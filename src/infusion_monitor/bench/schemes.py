"""对照基准的两套方案，共用产品主路径，只差「记忆」这一步。

两套方案用的是**同一个** :class:`.ColorBandPredictor`（同一个确定性阈值核）与
**同一条** ``init_state`` → ``add_new_points_or_box`` → ``propagate_in_video``，
差别只有 :func:`~..core.propagate.propagate_in_video` 的 ``memory_prompt`` 开关：

===================== ============================================== =========
方案                  每帧提示来自                                     记忆
===================== ============================================== =========
memory_propagation    上一帧掩膜导出的外扩 ROI + 质心正点（线上主路径） 有
single_frame_threshold 首帧锚点提示，每帧重新起算                      无
===================== ============================================== =========

**为什么阈值核换成 ``ColorBandPredictor`` 而不是产品线上那个**：线上几何核按
「高饱和、中等亮度」找输液袋里的液体，这是输液架的场景先验；公开 VOS 语料是自然
视频（牛、无人机、漂移板），那套阈值对它没有意义，两套方案会一起退化成同一个乱码，
对照就失去意义。这里的阈值核改用**由提示区域自估的色彩模型**——阈值带本身是同一个
「饱和度/亮度/色相带 → 形态学 → 碎片后处理 → 选块」的确定性家族，参数在两套方案间
完全一致，唯一随帧变化的是**提示框落在哪里**。这恰好把变量收敛到记忆机制上。

碎片后处理直接复用 :mod:`..core.postprocess` 的 ``postprocess_mask``，与产品主路径
同一段实现（孔洞填充与孤岛剔除两条独立路径，两个阈值互不影响）。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..core.postprocess import postprocess_mask
from ..core.predictor import MaskPrediction, Prompt, mask_bbox, mask_centroid

MEMORY_SCHEME = "memory_propagation"
SINGLE_FRAME_SCHEME = "single_frame_threshold"

#: 两套方案共用的碎片后处理阈值。与线上 :class:`..core.geom_kernel.GeomKernelParams`
#: 取同量级：太小则每个噪点都成为独立连通域，太大则把真实的小目标也剔掉。
MAX_HOLE_AREA = 2048
MAX_SPRINKLE_AREA = 192


@dataclass(frozen=True)
class ColorBandParams:
    """色彩带阈值核参数。两套方案共用同一份取值——见模块 docstring 的理由。"""

    #: 提示区域内各通道的中位绝对偏差倍数（相对中位数）。取值 1.5 而非更宽：
    #: 自然视频里主体与背景常有相近色相，放宽阈值带会把两者并成一块
    #: （实测 ``camel`` 第 0 帧一度把 7 万像素的主体选成 27 万像素）。
    spread_sigma: float = 1.5
    #: 绝对下限：自然视频的纹理噪声大，容差不能收到 0，否则整幅都被判成背景。
    spread_floor: float = 16.0
    #: 色相带半宽（OpenCV H，0–179）。0 表示不设色相约束。
    hue_tolerance: int = 14
    #: 开/闭运算核尺寸。**闭运算关掉**：它的作用是把细缝补上，但在本对照里
    #: 副作用远大于收益——主体与背景之间几十像素的过渡带会被它桥成一整块，
    #: 连通域选中后直接吞掉半幅画面。开运算保留，用于抹掉椒盐噪点。
    open_ksize: int = 3
    close_ksize: int = 0
    #: 连通域最小面积（px²）：低于此值按噪点丢掉。
    min_component_area: int = 150
    #: 正提示点到候选连通域质心的最大距离（占画面对角线比例）。超出即判「这一帧看不见」。
    max_pick_distance: float = 0.45
    #: 采样圆盘半径（占画幅对角线比例）：模型样本取正提示点周围这一圈纯主体像素。
    seed_radius_fraction: float = 0.04
    #: 采样框面积上限（占画幅比例）：掐断「选错块 → 框变大 → 更易选错」的正反馈。
    max_sample_fraction: float = 0.25
    #: 置信分：候选面积与提示框面积之比的压缩值，仅用于日志与排序，不进指标。
    area_ratio_score_cap: float = 4.0


class ColorBandPredictor:
    """按提示区域自估色彩模型的确定性阈值核。

    实现 ``PerPointMaskPredictor`` 协议（``predict``），因此可以直接喂给
    :func:`~..core.propagate.propagate_in_video`，与线上几何核**同接口**。
    记分卡里 ``name = "color_band"``，与线上 ``geom_kernel`` 并列注册。
    """

    name = "color_band"

    def __init__(self, params: ColorBandParams | None = None) -> None:
        self.params = params or ColorBandParams()

    def predict(self, image: np.ndarray, prompt: Prompt) -> list[MaskPrediction]:
        params = self.params
        hsv = cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_RGB2HSV)
        height, width = hsv.shape[:2]

        samples = self._sample(hsv, prompt)
        if samples.size == 0:
            # 提示框落在画面外或压不进像素：这一帧没有可依据的观测，返回空而不是硬选一块。
            return [MaskPrediction(mask=np.zeros((height, width), dtype=bool), score=0.0)]

        # 第一遍：用**中位数 + MAD**估色彩模型。
        # 不能用均值 + 标准差：提示框总会带进背景像素，均值被拉走、标准差被撑大，
        # 阈值带放宽到整幅画面都落进来，选块时只能挑到贴着提示点的那一大片背景。
        # 中位数与 MAD 对背景污染不敏感，代价是丢掉分布的长尾——对找主体恰好合适。
        model = self._robust_model(samples)
        mask = self._segment(hsv, model, prompt)
        # 第二遍：拿第一遍选中的那一块重新估模型。框里的背景被剔掉之后，
        # 阈值带自然收窄到主体自身。这一步两套方案完全一致。
        if mask.any():
            refined = self._robust_model(hsv[mask].astype(np.float32))
            mask = self._segment(hsv, refined, prompt)
        return [MaskPrediction(mask=mask, score=self._score(mask, prompt, (height, width)))]

    # ── 内部四段 ──────────────────────────────────────────────────────

    def _robust_model(self, samples: np.ndarray) -> np.ndarray:
        """由像素样本估色彩模型，返回 ``[H, S, V]`` 的中心值与逐通道容差。

        容差取 ``max(spread_floor, spread_sigma × 1.4826 × MAD)``；``1.4826`` 是把
        中位绝对偏差换算成正态标准差的系数。色相走环形距离，容差再截到 90°
        （超过半圈就等于不约束）。
        """
        params = self.params
        median = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - median), axis=0)
        tolerance = np.maximum(params.spread_floor, params.spread_sigma * 1.4826 * mad)
        tolerance[0] = min(float(tolerance[0]), 90.0)
        return np.concatenate([median, tolerance])

    def _segment(self, hsv: np.ndarray, model: np.ndarray, prompt: Prompt) -> np.ndarray:
        """按给定色彩模型阈值化 → 形态学 → 碎片后处理 → 选块，返回最终掩膜。"""
        params = self.params
        center = model[:3]
        tolerance = model[3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # OpenCV 的 H 是环形量：跨 0/179 的色相要绕回来比，不能直接卡上下界。
        dh = np.abs(h.astype(np.int16) - int(round(float(center[0]))))
        dh = np.minimum(dh, 180 - dh)
        candidates = dh <= tolerance[0]
        candidates &= (s >= center[1] - tolerance[1]) & (s <= center[1] + tolerance[1])
        candidates &= (v >= center[2] - tolerance[2]) & (v <= center[2] + tolerance[2])

        if params.open_ksize > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_RECT, (params.open_ksize, params.open_ksize)
            )
            candidates = cv2.morphologyEx(
                candidates.astype(np.uint8), cv2.MORPH_OPEN, k
            ).astype(bool)
        if params.close_ksize > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_RECT, (params.close_ksize, params.close_ksize)
            )
            candidates = cv2.morphologyEx(
                candidates.astype(np.uint8), cv2.MORPH_CLOSE, k
            ).astype(bool)

        candidates = postprocess_mask(
            candidates,
            max_hole_area=MAX_HOLE_AREA,
            max_sprinkle_area=MAX_SPRINKLE_AREA,
        ).mask
        return self._pick(candidates, prompt, hsv.shape[:2])

    def _sample(self, hsv: np.ndarray, prompt: Prompt) -> np.ndarray:
        """取**正提示点周围一个圆盘**内的 HSV 像素作为色彩模型样本。

        为什么不用整个提示框：框里必然有背景，而背景一旦进了样本，均值/中位数被拉走、
        离散度被撑大，阈值带宽到整片沙地和骆驼都落进来，选块时只能挑到贴着提示点的
        那一大片背景——实测在 ``camel`` 上第 0 帧就把 7 万像素的主体选成 27 万像素。
        质心点周围的圆盘则基本落在主体内部，阈值带自然收紧。

        圆盘半径取 ``seed_radius_fraction × 对角线``，并按提示框短边的一半封顶：
        目标越小（比如漂移板上的无人机）越要跟着缩，否则一个圆盘就把目标连同背景全吃进来。

        没有正提示点时（纯框提示）退回按框采样，并受 ``max_sample_fraction`` 封顶——
        记忆提示框是上一帧掩膜外扩得来的，选错一块就会一帧帧变大、采样越来越杂，
        最后彻底锁死在背景上；封顶把这条正反馈掐断。
        """
        height, width = hsv.shape[:2]
        points = prompt.positive_points
        box = prompt.clipped_box((height, width))

        if points:
            cx, cy = float(points[0].x), float(points[0].y)
            radius = self.params.seed_radius_fraction * float(np.linalg.norm((height, width)))
            if box is not None:
                bx0, by0, bx1, by1 = box
                radius = min(radius, 0.4 * min(bx1 - bx0, by1 - by0))
            radius = max(radius, 2.0)
            x0 = int(max(0, min(width - 1, round(cx - radius))))
            x1 = int(max(x0 + 1, min(width, round(cx + radius))))
            y0 = int(max(0, min(height - 1, round(cy - radius))))
            y1 = int(max(y0 + 1, min(height, round(cy + radius))))
            patch = hsv[y0:y1, x0:x1]
            yy, xx = np.ogrid[y0:y1, x0:x1]
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
            patch = patch[disk]
        elif box is not None:
            bx0, by0, bx1, by1 = box
            budget = self.params.max_sample_fraction * height * width
            area = (bx1 - bx0) * (by1 - by0)
            if area > budget > 0:
                scale = float(np.sqrt(budget / area))
                cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
                half_w, half_h = (bx1 - bx0) * scale / 2.0, (by1 - by0) * scale / 2.0
                bx0 = int(max(0, min(width - 1, round(cx - half_w))))
                bx1 = int(max(bx0 + 1, min(width, round(cx + half_w))))
                by0 = int(max(0, min(height - 1, round(cy - half_h))))
                by1 = int(max(by0 + 1, min(height, round(cy + half_h))))
            patch = hsv[by0:by1, bx0:bx1]
        else:
            return np.empty((0, 3), dtype=np.float32)

        if patch.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return patch.reshape(-1, 3).astype(np.float32)

    def _pick(
        self, candidates: np.ndarray, prompt: Prompt, shape: tuple[int, int]
    ) -> np.ndarray:
        """在候选连通域里选一块：离正提示点最近的那块，且不得远到离谱。"""
        empty = np.zeros(shape, dtype=bool)
        if not np.any(candidates):
            return empty

        points = prompt.positive_points
        if not points:
            return empty
        px, py = float(points[0].x), float(points[0].y)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidates.astype(np.uint8), connectivity=8
        )
        diagonal = float(np.linalg.norm(shape))
        prompt_box = prompt.clipped_box(shape)
        prompt_area = 0
        if prompt_box is not None:
            bx0, by0, bx1, by1 = prompt_box
            prompt_area = max(1, (bx1 - bx0) * (by1 - by0))

        best_label, best_distance, best_area = 0, None, 0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.params.min_component_area:
                continue
            cx, cy = centroids[label]
            distance = float(np.hypot(cx - px, cy - py))
            if distance > self.params.max_pick_distance * diagonal:
                continue
            if best_distance is None or distance < best_distance:
                best_label, best_distance, best_area = label, distance, area

        if best_label == 0:
            return empty
        return labels == best_label

    def _score(self, mask: np.ndarray, prompt: Prompt, shape: tuple[int, int]) -> float:
        """置信分：选块面积与提示框面积之比。仅供日志与排序，不进评测指标。"""
        if not mask.any():
            return 0.0
        prompt_box = prompt.clipped_box(shape)
        if prompt_box is None:
            return 0.0
        bx0, by0, bx1, by1 = prompt_box
        prompt_area = max(1, (bx1 - bx0) * (by1 - by0))
        ratio = int(np.count_nonzero(mask)) / prompt_area
        return float(min(ratio, self.params.area_ratio_score_cap))


def anchor_from_mask(mask: np.ndarray) -> Prompt:
    """由真值首帧的某对象掩膜起锚。

    起锚提示 = 该掩膜的质心正点 + 外接框。产品线上由护士点一下或由首帧枚举给候选框，
    这里用真值框只是为了**让两套方案从同一个起点出发**，起锚的额外信息量不会偏向任何一方。
    """
    box = mask_bbox(mask)
    centroid = mask_centroid(mask)
    if box is None or centroid is None:
        raise ValueError("起锚掩膜为空，无法构造提示")
    x0, y0, x1, y1 = box
    cx, cy = centroid
    return Prompt.from_box(
        float(x0), float(y0), float(x1), float(y1), point=(float(cx), float(cy))
    )


def overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """两掩膜的 IoU。起锚提示用不到产品主路径的 ``mask_iou`` 时可作诊断。"""
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0