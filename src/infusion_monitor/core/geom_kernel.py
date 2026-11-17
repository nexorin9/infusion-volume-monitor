"""确定性几何核：液体区域阈值带 + 形态学开闭 + 孔洞填充 / 孤岛剔除。

**为什么需要它**：本轮环境没有 GPU、也没有生产权重，但主路径不能因此停摆——护士上传
一段短片必须当次就拿到可核对的面积比曲线。几何核是**确定性基线**，既是无权重时的默认
实现，也是模型不可达时的降级路径（见 task 12/42），不是临时凑数。

**与源实现的语义对应**（只借参数语义，不搬代码、不跑源权重）：

* 源 ``SAM2ImagePredictor.__init__`` 的两个阈值（``sam2_image_predictor.py:37-40``）
  —— ``max_hole_area``：掩膜里面积不超过它的孔洞被**填上**；``max_sprinkle_area``：
  面积不超过它的碎斑被**删掉**。二者单位是源实现 low-res 掩膜上的像素数，本实现统一
  换算到**工作分辨率**下的像素面积（``GeomKernelParams`` 里的两个参数）。
* 源 ``masks = masks > self.mask_threshold``（``sam2_image_predictor.py:436``）——
  阈值化出二值掩膜；本实现对应 HSV 阈值带 :func:`liquid_candidates`。

**与源实现的一处有意偏离**：源里 ``max_sprinkle_area`` 会删掉**所有**小连通域。本实现
只删「不含正提示点」的碎斑——液面低至 2% 时那袋本身就是一条细窄连通域，删掉它等于把
真数据当噪声抹平。这条偏离在 tests/test_propagate.py 与 tests/test_postprocess.py 里有专门用例锁住。

形态学收尾本身已拆到 :mod:`.postprocess`（连通域、形态特征、两个阈值、粘连分离），
本模块只保留「阈值带 + 形态学开闭 + 取块」这几步与预测器协议的适配。

纯 numpy/OpenCV，不新增第三方依赖。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .postprocess import (
    AdhesionParams,
    apply_adhesion_cuts,
    fill_small_holes,
    find_adhesion_cuts,
    remove_small_sprinkles,
)
from .predictor import BatchCapablePredictor, MaskPrediction, Prompt

__all__ = [
    "GeomKernelParams",
    "GeometricMaskPredictor",
    "PreparedFrame",
    "liquid_candidates",
    "liquid_candidates_at",
    "fill_small_holes",
    "remove_small_sprinkles",
    "circularity",
    "estimate_cavity",
    "CavityEstimate",
]


@dataclass(frozen=True)
class GeomKernelParams:
    """几何核阈值。全部按「工作分辨率下的像素 / 像素²」计。"""

    #: 饱和度下限（OpenCV HSV 的 S，0–255）：液体比透明袋体明显更饱和。
    saturation_min: int = 96
    #: 亮度下限：排除暗背景。
    value_min: int = 45
    #: 亮度上限：排除过曝高光与镜面反光。
    value_max: int = 250
    #: 孔洞面积上限（px²）——对应源实现的 ``max_hole_area``。
    #: 取 2048 而非极小值：手机拍到的镜面高光与标签反光常在这个量级，按 0（源默认值）
    #: 会让每一次反光都在面积曲线上打出一个假凹陷。
    max_hole_area: int = 2048
    #: 孤岛面积上限（px²）——对应源实现的 ``max_sprinkle_area``。
    max_sprinkle_area: int = 192
    #: 形态学开运算核尺寸：先抹掉椒盐噪点。
    open_ksize: int = 3
    #: 形态学闭运算核尺寸：再补上液面与袋壁之间的细缝。
    close_ksize: int = 7
    #: 腔体估计时，判定「非背景」的通道最大差值。
    background_delta: int = 28
    #: 连通域最小面积（px²）：低于此值直接当噪点，不进候选。
    min_component_area: int = 24
    #: 正提示点到连通域外接框中心的最大允许距离（占画面对角线比例）。
    #: 超出即视为「这一帧看不见它」，返回空而不是硬选一块不相干的区域。
    max_pick_distance: float = 0.9
    #: 粘连分离门限。输液架上两袋贴住时，闭运算会把袋间缝隙桥接成一块，让其中一个袋子的
    #: 面积比翻倍、毫升数与换液时点一起带偏；这里按「双峰 + 鞍部」判据把粘连的两袋拆开。
    #: 判定在**每个连通域内**做（不是整幅一次），因此两袋离得远时绝不会互相误伤。
    adhesion: AdhesionParams = field(default_factory=AdhesionParams)
    #: 粘连判定的最小面积门槛（px²）：小于此值的碎块直接跳过，粘不住也拆不动。
    adhesion_min_area: int = 1500


def liquid_candidates_at(
    image: np.ndarray, saturation_min: int, params: GeomKernelParams | None = None
) -> np.ndarray:
    """按给定**饱和度下限**取液体候选（亮度带固定不变）。

    抽成独立函数是为了让枚举管线能做稳定性扰动：把饱和度下限抬高/压低若干级，
    观察同一块区域的像素留下多少（对应源实现里对 logit 阈值做同款扰动）。

    返回与输入同尺寸的 ``bool`` 数组。
    """
    params = params or GeomKernelParams()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"期望 HxWx3 的 RGB 帧，实际形状 {image.shape}")
    hsv = cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    return (
        (saturation >= saturation_min)
        & (value >= params.value_min)
        & (value <= params.value_max)
    )


def liquid_candidates(
    image: np.ndarray, params: GeomKernelParams | None = None
) -> np.ndarray:
    """HSV 阈值带：饱和度够高、亮度既不欠曝也不过曝的像素即液体候选。

    返回与输入同尺寸的 ``bool`` 数组。
    """
    params = params or GeomKernelParams()
    return liquid_candidates_at(image, params.saturation_min, params)


def circularity(mask: np.ndarray) -> float:
    """圆度 ``4πA / P²``，取值 (0, 1]。用作**形状规整度分**，不是概率。"""
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0.0
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    perimeter = max(
        (cv2.arcLength(contour, True) for contour in contours), default=0.0
    )
    if perimeter <= 0:
        return 0.0
    return float(min(1.0, 4.0 * math.pi * area / (perimeter * perimeter)))


@dataclass(frozen=True)
class CavityEstimate:
    """袋腔估计结果。``area_px`` 是首帧定下的腔体像素面积，整段短片沿用同一个值。"""

    area_px: int
    mask: np.ndarray


def estimate_cavity(
    image: np.ndarray,
    prompt: Prompt,
    params: GeomKernelParams | None = None,
) -> CavityEstimate:
    """估计袋腔（袋体轮廓）面积，作为面积比的归一化分母。

    做法：用画面四周的背景色中位数当背景基准，通道最大差值超阈值即前景，取离正提示点
    最近的连通域并把孔洞全填上。**只在起锚帧估一次**，之后整段短片沿用——袋体随帧轻微
    形变，逐帧重估只会把光照抖动混进曲线里。
    """
    params = params or GeomKernelParams()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"期望 HxWx3 的 RGB 帧，实际形状 {image.shape}")

    height, width = image.shape[0], image.shape[1]
    ring = np.concatenate(
        [
            image[0, :, :].reshape(-1, 3),
            image[height - 1, :, :].reshape(-1, 3),
            image[:, 0, :].reshape(-1, 3),
            image[:, width - 1, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(ring, axis=0)
    delta = np.abs(image.astype(np.int16) - background.astype(np.int16)).max(axis=2)
    foreground = delta >= params.background_delta

    box = prompt.clipped_box(image.shape)
    if box is not None:
        keep = np.zeros_like(foreground)
        keep[box[1] : box[3], box[0] : box[2]] = True
        foreground &= keep

    component = _pick_component(foreground, prompt, params)
    if not component.any():
        # 兜底：提示框内的全部前景
        if box is not None:
            full = np.zeros_like(foreground)
            full[box[1] : box[3], box[0] : box[2]] = foreground[box[1] : box[3], box[0] : box[2]]
            component = full
        else:
            component = foreground
    # 腔体是一个整体，内部孔洞全填上（不设上限）：不设上限才不会让孔洞阈值把它切开
    component = fill_small_holes(component, max_area=int(np.count_nonzero(component)))
    return CavityEstimate(area_px=int(np.count_nonzero(component)), mask=component)


def _pick_component(
    foreground: np.ndarray, prompt: Prompt, params: GeomKernelParams
) -> np.ndarray:
    """在前景里选出提示所指的那一块：优先含正点的连通域，否则取离正点最近且够大的块。

    腔体估计（:func:`estimate_cavity`）走这条路——前景是**袋体**而不是液体，
    不适用孤岛剔除规则。
    """
    if not foreground.any():
        return np.zeros_like(foreground, dtype=bool)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(foreground, dtype=bool)

    return _select_from_labels(
        labels, stats, centroids, prompt, params, foreground.shape, drop_sprinkles=False
    )


def _select_from_labels(
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    prompt: Prompt,
    params: GeomKernelParams,
    shape: tuple[int, int],
    *,
    drop_sprinkles: bool,
) -> np.ndarray:
    """在**已算好的连通域标签**上按提示选块。

    与逐点整算（阈值带 → 形态学 → 填孔 → 剔孤岛 → 取块）等价：形态学与填孔是
    **逐帧**操作、与提示无关，可以预先算好；只有「剔孤岛」与「取块」依赖提示，
    在这里现算。枚举管线在同一帧上问几十个点，靠这个分层把单点成本压到 O(1)。

    ``drop_sprinkles=True`` 时启用孤岛剔除（对应源实现的 ``max_sprinkle_area``）：
    面积不超过阈值的碎斑被剔除，**除非它含正提示点**——液面低至 2% 的那袋
    本身就是一条细窄连通域，删掉等于把真数据当噪声抹平。
    """
    height, width = shape
    diagonal = math.hypot(height, width)
    positives = prompt.positive_points

    best_label: int | None = None
    best_key: tuple[int, float] | None = None
    for label in range(1, len(stats)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        contains_point = _contains_label_point(labels, label, positives)
        if drop_sprinkles and area <= params.max_sprinkle_area and not contains_point:
            continue
        if area < params.min_component_area:
            continue
        rejected_by_negative = prompt.negative_points and _near_any(
            float(centroids[label][0]), float(centroids[label][1]), prompt.negative_points
        )
        if rejected_by_negative and not contains_point:
            continue
        if contains_point:
            # 正点直接落在这块里：命中，块越大越可信
            key = (0, -float(area))
        elif positives:
            # 正点没落进去：只有离正点足够近才认，否则视为这一帧看不见它
            x0 = stats[label, cv2.CC_STAT_LEFT]
            y0 = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            cx, cy = x0 + w / 2.0, y0 + h / 2.0
            distance = min(math.hypot(cx - point.x, cy - point.y) for point in positives) / diagonal
            if distance > params.max_pick_distance:
                continue
            key = (1, distance)
        else:
            # 无点提示：取最大块
            key = (2, -float(area))
        if best_key is None or key < best_key:
            best_key = key
            best_label = label

    if best_label is None:
        return np.zeros((height, width), dtype=bool)
    return labels == best_label


def _contains_point(component: np.ndarray, points) -> bool:
    height, width = component.shape
    for point in points:
        x, y = int(round(point.x)), int(round(point.y))
        if 0 <= x < width and 0 <= y < height and component[y, x]:
            return True
    return False


def _contains_label_point(labels: np.ndarray, label: int, points) -> bool:
    """提示点是否落在**指定连通域**里（不必先切出整块掩膜）。"""
    height, width = labels.shape
    for point in points:
        x, y = int(round(point.x)), int(round(point.y))
        if 0 <= x < width and 0 <= y < height and labels[y, x] == label:
            return True
    return False


def _near_any(x: float, y: float, points) -> bool:
    return any(math.hypot(x - point.x, y - point.y) <= 12.0 for point in points)


@dataclass(frozen=True)
class PreparedFrame:
    """一帧的预处理结果：连通域标签 + 每块的面积/外接框/质心。

    枚举管线（:mod:`.enumerate`）在一帧上要问几十上百个网格点；把「阈值带 →
    形态学 → 填孔 → 连通域」做成**一次性**准备，逐点查询就退化成 O(1) 的标签查表。
    语义与逐点整算一致（见 :meth:`GeometricMaskPredictor.select`）。
    """

    labels: np.ndarray
    stats: np.ndarray
    centroids: np.ndarray
    count: int
    params: GeomKernelParams
    shape: tuple[int, int]


def _clip_to_window(mask: np.ndarray, window: tuple[int, int, int, int], prompt: Prompt) -> np.ndarray:
    """把选中的块裁进提示框，再按框内连通性取一块（含正提示点的那块）。

    提示框的作用是「别把旁边的袋子粘进来」。逐点整算是先把候选像素裁进框再算
    连通域；预处理路径只能在选块之后裁，裁完再在框内重新取连通块——两者等价。
    """
    x0, y0, x1, y1 = window
    clipped = np.zeros_like(mask)
    clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    if not clipped.any():
        return clipped

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        clipped.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return clipped

    chosen: int | None = None
    for label in range(1, count):
        if _contains_label_point(labels, label, prompt.positive_points):
            chosen = label
            break
    if chosen is None:
        chosen = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == chosen


class GeometricMaskPredictor(BatchCapablePredictor):
    """确定性几何核实现：液体阈值带 → 开闭 → 选块 → 填孔 → 剔孤岛。"""

    name = "geom_kernel"

    def __init__(self, params: GeomKernelParams | None = None) -> None:
        self.params = params or GeomKernelParams()

    # ── PerPointMaskPredictor 协议 ────────────────────────────────────
    def predict(self, image: np.ndarray, prompt: Prompt) -> list[MaskPrediction]:
        return self._prediction(self.prepare(image), prompt)

    def predict_batch(
        self, image: np.ndarray, prompts
    ) -> list[list[MaskPrediction]]:
        prepared = self.prepare(image)
        return [self._prediction(prepared, prompt) for prompt in prompts]

    # ── 枚举管线用的两段式接口 ────────────────────────────────────────
    def prepare(self, image: np.ndarray) -> PreparedFrame:
        """一次性准备一帧：阈值带 → 形态学开闭 → 填孔 → 粘连分离 → 连通域标签。

        这些步骤**与提示无关**，逐点整算时是重复劳动；预先算一次即可。
        """
        params = self.params
        candidates = liquid_candidates(image, params)

        # 粘连判定必须在闭运算**之前**取证据：两袋之间 4 px 的缝隙会被闭运算桥成和袋体
        # 一样厚的整块，届时它与「一个特别宽的袋」在几何上完全一样，无从判定。见 postprocess。
        adhesion_cuts = (
            find_adhesion_cuts(
                candidates,
                params.adhesion,
                min_area=params.adhesion_min_area,
            )
            if params.close_ksize > 0
            else []
        )

        if params.open_ksize > 0:
            open_k = cv2.getStructuringElement(
                cv2.MORPH_RECT, (params.open_ksize, params.open_ksize)
            )
            candidates = cv2.morphologyEx(
                candidates.astype(np.uint8), cv2.MORPH_OPEN, open_k
            ).astype(bool)
        if params.close_ksize > 0:
            close_k = cv2.getStructuringElement(
                cv2.MORPH_RECT, (params.close_ksize, params.close_ksize)
            )
            candidates = cv2.morphologyEx(
                candidates.astype(np.uint8), cv2.MORPH_CLOSE, close_k
            ).astype(bool)

        # 顺序对齐源实现：阈值化 → 碎片后处理 → 取目标块
        candidates = fill_small_holes(candidates, params.max_hole_area)

        # 切割必须在闭运算**之后**做，否则刚切开又被桥回去。
        if adhesion_cuts:
            candidates, _ = apply_adhesion_cuts(candidates, adhesion_cuts, params.adhesion)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidates.astype(np.uint8), connectivity=8
        )
        return PreparedFrame(
            labels=labels,
            stats=stats,
            centroids=centroids,
            count=int(count),
            params=params,
            shape=(int(image.shape[0]), int(image.shape[1])),
        )

    def select(self, prepared: PreparedFrame, prompt: Prompt) -> np.ndarray:
        """在预处理好的帧上按提示选块（提示相关的部分只有剔孤岛与取块）。"""
        selected = _select_from_labels(
            prepared.labels,
            prepared.stats,
            prepared.centroids,
            prompt,
            prepared.params,
            prepared.shape,
            drop_sprinkles=True,
        )
        if not selected.any():
            return selected
        box = prompt.clipped_box(prepared.shape)
        if box is not None:
            selected = _clip_to_window(selected, box, prompt)
        return selected

    def _prediction(self, prepared: PreparedFrame, prompt: Prompt) -> list[MaskPrediction]:
        selected = self.select(prepared, prompt)
        if not selected.any():
            return []
        return [MaskPrediction(mask=selected, score=round(circularity(selected), 4))]

    # ── 辅助 ──────────────────────────────────────────────────────────
    def estimate_cavity(self, image: np.ndarray, prompt: Prompt) -> CavityEstimate:
        return estimate_cavity(image, prompt, self.params)