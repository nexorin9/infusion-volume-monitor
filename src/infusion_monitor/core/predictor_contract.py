"""预测器协议契约：**一套断言，两个实现都跑**。

**为什么要有这一层**：业务层只认 :class:`~.predictor.PerPointMaskPredictor` 协议
（``predict`` / ``predict_batch``），主路径不认实现。换实现（确定性几何核 ↔ 生产权重）
是一次环境变量的事，但**换实现也是一次出数口径的变更**——几何核接上去的两个常见毛病
（掩膜内部降采样再放大、坐标系偏了几个像素）在类型层面完全合法：dtype 还是 ``bool``，
shape 还是 ``HxW``，函数签名一模一样，只有面积曲线会悄悄错掉几个百分点，等到护士长比对
两次巡视的毫升数才发现，那时已经查不出是哪一步错的。

所以契约不是「跑得通就算」，而是**把协议里那三条约定逐条钉成可执行的断言**：同尺寸同
坐标系（不许内部下采样）、结构完整（dtype/shape/score/排序）、退化提示不炸且不返回整帧。
任何实现改了其中一条，这套断言立刻红。

**三个实现差异是允许的**（断言据此设计，不要求两个实现出数一致）：

* ``predict(image, Prompt())``（空提示）的解释可以不同——几何核取最大块（首帧枚举就是
  靠这条起候选的），权重实现通常返回空列表。契约只要求**不抛异常、结构正确、绝不返回
  整帧**（返回整帧等于把面积比钉成 1.0，毫升数直接失真）；
* ``score`` 是形状规整度（0–1），不是概率，两个实现的数值不必相同；
* 掩膜边界允许有形态学差异，断言给的是带（IoU 下限 + 不串到邻块 + 质心误差上限）。

**PNG 往返**用 :func:`encode_mask_png` / :func:`decode_mask_png`：掩膜要落盘、要跨轮次
比对、要在问题单里当证据附件，编码口径必须是同一对函数，不能每个脚本各写一次
``cv2.imencode``——0/255 还是 0/1、灰度还是调色板，差一位就再也对不上。
冷归档（保留策略与容量测算）复用同一对函数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np

from .predictor import (
    MaskPrediction,
    PerPointMaskPredictor,
    Point,
    Prompt,
    mask_centroid,
    mask_iou,
)

__all__ = [
    "CONTRACT_CHECKS",
    "Blob",
    "ContractCase",
    "ContractCheckResult",
    "ContractReport",
    "ContractScene",
    "ContractViolation",
    "blob_scene",
    "blob_truth_mask",
    "check_batch_matches_single",
    "check_determinism_and_purity",
    "check_degenerate_prompts_do_not_raise",
    "check_empty_target_returns_empty_list",
    "check_mask_png_roundtrip",
    "check_point_lands_on_truth_blob",
    "check_predictor_surface",
    "check_prediction_structure",
    "contract_cases",
    "decode_mask_png",
    "encode_mask_png",
    "plain_scene",
    "run_predictor_contract",
    "summarise_reports",
]


#: 全图返回的兜底上限：一张 ``HxW`` 掩膜若盖住 95% 以上画面，那不是「找到了袋子」，
#: 是「没找到于是把整张图交上去了」。面积比会因此被钉成 1.0，毫升数直接失真。
FULL_FRAME_GUARD = 0.95

#: 契约检查项的固定顺序（报告与自检输出按这个顺序，不按执行顺序）。
CONTRACT_CHECKS: tuple[str, ...] = (
    "surface",
    "structure",
    "empty_target",
    "degenerate_prompts",
    "coordinate_alignment",
    "batch_consistency",
    "determinism_and_purity",
    "serialisation",
)


class ContractViolation(Exception):
    """实现违反了 :class:`~.predictor.PerPointMaskPredictor` 协议。

    单独一个异常类型而不是直接用 ``AssertionError``：契约既在 pytest 里跑（要漂亮的
    失败输出），也在病区侧自检里跑（要给运维一条能照做的中文提示），两边都要能单独捕获。
    """


# ─────────────────────────────────────────────────────────────────────────────
# 契约样例：一帧画布 + 若干提示，以及每个提示「应该落到哪块」
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Blob:
    """契约画面里的一块已知区域。"""

    key: str
    x0: int
    y0: int
    x1: int
    y1: int
    rgb: tuple[int, int, int]

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (float(self.x0), float(self.y0), float(self.x1), float(self.y1))

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, dtype=bool)
        out[self.y0 : self.y1, self.x0 : self.x1] = True
        return out


@dataclass(frozen=True)
class ContractScene:
    """一帧契约画面 + 画面里的已知区域。"""

    image: np.ndarray
    blobs: tuple[Blob, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.image.shape[0]), int(self.image.shape[1]))

    def blob(self, key: str) -> Blob:
        for blob in self.blobs:
            if blob.key == key:
                return blob
        raise KeyError(f"契约画面里没有区域 {key!r}，现有：{[b.key for b in self.blobs]}")

    def truth(self, key: str) -> np.ndarray:
        return self.blob(key).mask(self.shape)


@dataclass(frozen=True)
class ContractCase:
    """一个契约用例：一条提示 + 它应该落到哪块（或不应该落到任何块）。"""

    key: str
    prompt: Prompt
    scene: ContractScene
    expect_target: str | None = None
    forbid_target: str | None = None
    min_iou: float = 0.6
    centroid_tolerance: float = 10.0


#: 契约画面的底色：中性灰，饱和度为 0，任何按「液体阈值带」取候选的实现都不该把它当前景。
_BACKGROUND_RGB = (200, 200, 200)

#: 三块区域的颜色都落在 HSV 的 S≥96 / 45≤V≤250 带内——这不是巧合：契约画面必须让
#: **两种实现都看得见**，否则「实现没找到目标」和「画面里本来就没有目标」分不开。
_BLOBS: tuple[Blob, ...] = (
    Blob("a", 30, 52, 110, 196, (30, 120, 220)),
    Blob("b", 200, 52, 280, 196, (215, 90, 35)),
    # 小块专治「坐标系偏了几个像素」：24×24 的块上偏 6px 就足以把 IoU 打到 0.5 以下。
    Blob("c", 140, 96, 164, 120, (40, 170, 70)),
)


def blob_scene(width: int = 320, height: int = 240) -> ContractScene:
    """契约画面：三块已知区域铺在中性灰底上。**逐位可复现**，不引入随机量。"""

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = _BACKGROUND_RGB
    for blob in _BLOBS:
        image[blob.y0 : blob.y1, blob.x0 : blob.x1] = blob.rgb
    return ContractScene(image=image, blobs=_BLOBS)


def plain_scene(width: int = 320, height: int = 240) -> ContractScene:
    """「画面里根本没有可分割的东西」：整帧中性灰。用于「找不到目标 → 空列表」这一条。"""

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = _BACKGROUND_RGB
    return ContractScene(image=image, blobs=())


def blob_truth_mask(blob_key: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    """按契约画面默认尺寸取某块的真值掩膜（外部核对用）。"""

    scene = blob_scene()
    if shape is not None and tuple(shape) != scene.shape:
        raise ValueError(f"契约画面尺寸固定为 {scene.shape}，收到 {tuple(shape)}")
    return scene.truth(blob_key)


def contract_cases(scene: ContractScene | None = None) -> tuple[ContractCase, ...]:
    """共享用例集：两个实现跑的是**同一批提示**，不是各自造各自的例子。

    覆盖：点提示、框提示、点+框、正负点、浮点坐标、越界框、空提示、只有负点、
    画面外点——外加「找不到就返回空列表」用的纯色画面。
    """

    scene = scene or blob_scene()
    blob_a, blob_b, blob_c = scene.blob("a"), scene.blob("b"), scene.blob("c")
    ax, ay = blob_a.centre
    bx, by = blob_b.centre
    cx, cy = blob_c.centre
    height, width = scene.shape

    cases = [
        ContractCase("point_in_a", Prompt.from_point(ax, ay), scene, "a", min_iou=0.6),
        ContractCase("point_in_b", Prompt.from_point(bx, by), scene, "b", min_iou=0.6),
        # 小块 + 更严的质心门限：专门抓坐标系平移/缩放错位
        ContractCase(
            "point_in_small_c",
            Prompt.from_point(cx, cy),
            scene,
            "c",
            min_iou=0.5,
            centroid_tolerance=6.0,
        ),
        ContractCase(
            "float_point_half_pixel",
            Prompt.from_point(ax + 0.5, ay - 0.5),
            scene,
            "a",
            min_iou=0.6,
        ),
        ContractCase(
            "box_around_a",
            Prompt(box=blob_a.box),
            scene,
            "a",
            min_iou=0.6,
        ),
        ContractCase(
            "box_with_point",
            Prompt(points=(Point(ax, ay, 1),), box=blob_a.box),
            scene,
            "a",
            min_iou=0.6,
        ),
        # 正点在 a、负点在 b 的形心：b 本身就是一块合格候选，靠负点才排得掉
        ContractCase(
            "positive_plus_negative",
            Prompt(points=(Point(ax, ay, 1), Point(bx, by, 0))),
            scene,
            "a",
            forbid_target="b",
            min_iou=0.6,
        ),
        ContractCase(
            "box_partially_outside",
            Prompt(box=(-30.0, 40.0, blob_a.x1, blob_a.y1 + 14)),
            scene,
            "a",
            min_iou=0.55,
        ),
        # ── 退化提示：只要求「不炸 + 结构正确 + 不返回整帧」，不要求返回空 ──
        ContractCase("point_outside_frame", Prompt.from_point(-40.0, -40.0), scene),
        ContractCase(
            "point_far_outside_frame",
            Prompt.from_point(float(width) + 90.0, float(height) + 90.0),
            scene,
        ),
        ContractCase("point_on_bottom_edge", Prompt.from_point(float(width) - 0.5, float(height) - 0.5), scene),
        ContractCase("bare_prompt", Prompt(), scene),
        ContractCase("negative_only_prompt", Prompt.from_point(ax, ay, 0), scene),
        ContractCase("box_outside_frame", Prompt(box=(-200.0, -200.0, -100.0, -100.0)), scene),
        ContractCase("box_degenerate", Prompt(box=(10.0, 10.0, 10.0, 60.0)), scene),
    ]

    plain = plain_scene(width=scene.shape[1], height=scene.shape[0])
    cases.extend(
        [
            ContractCase("plain_point", Prompt.from_point(ax, ay), plain),
            ContractCase("plain_box", Prompt(box=blob_a.box), plain),
            ContractCase("plain_bare_prompt", Prompt(), plain),
        ]
    )
    return tuple(cases)


# ─────────────────────────────────────────────────────────────────────────────
# 掩膜 PNG 编解码：落盘、比对、送审附件共用的唯一口径
# ─────────────────────────────────────────────────────────────────────────────
def encode_mask_png(mask: np.ndarray) -> bytes:
    """把 ``bool`` 掩膜编成 0/255 的**单通道 8 位灰度 PNG**（PNG 无损，往返逐位相同）。

    非连续视图（切片出来的子矩阵）照样能编——调用方常拿 ``mask[::2, ::2]`` 或
    ``mask[:, ::-1]`` 直接落盘，不该在这里炸一个「layout 不连续」。
    """

    array = np.asarray(mask)
    if array.dtype != np.bool_:
        raise ContractViolation(
            f"掩膜必须是 bool 数组，实际 dtype={array.dtype}。"
            f"0/1 的 uint8 请先 `mask.astype(bool)`——1 与 True 在下游的按位运算里不等价。"
        )
    if array.ndim != 2:
        raise ContractViolation(f"掩膜必须是二维 HxW，实际形状 {array.shape}")
    payload = np.ascontiguousarray(array.astype(np.uint8) * 255)
    ok, buffer = cv2.imencode(".png", payload)
    if not ok:
        raise ContractViolation(f"掩膜编码失败：形状 {array.shape}")
    return buffer.tobytes()


def decode_mask_png(payload: bytes) -> np.ndarray:
    """把 PNG 解回 ``bool`` 掩膜；形状与 :func:`encode_mask_png` 一一对应。

    按**灰度**读而不是按彩色读：调色板 PNG 与 RGB PNG 在这里必须是同一条路，否则同一张
    掩膜换个来源就对不上。
    """

    raw = np.frombuffer(payload, dtype=np.uint8)
    if raw.size == 0:
        raise ContractViolation("PNG 载荷为空")
    decoded = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    # 不同 OpenCV 构建的 imdecode 返回值不一样：有的返回 (retval, img)，有的直接返回
    # 图像（失败时给 None 或空数组）。两种都接住，否则病区侧换一个 OpenCV 版本就会在
    # 解码这一行炸掉，而炸的地方离「掩膜存不下来」很远，很难往回找。
    if isinstance(decoded, tuple):
        ok, decoded = decoded
        if not ok:
            raise ContractViolation("PNG 载荷解码失败（截断或不是 PNG）")
    if decoded is None or getattr(decoded, "size", 0) == 0:
        raise ContractViolation("PNG 载荷解码失败（截断或不是 PNG）")
    if decoded.ndim != 2:
        raise ContractViolation(f"掩膜 PNG 解码后应为单通道灰度，实际形状 {decoded.shape}")
    return decoded > 127


# ─────────────────────────────────────────────────────────────────────────────
# 逐条断言：每条一个函数，失败抛 ContractViolation（消息是给人照做的）
# ─────────────────────────────────────────────────────────────────────────────
def _violation(check: str, predictor: object, detail: str) -> ContractViolation:
    return ContractViolation(f"[{check}] 实现 {getattr(predictor, 'name', type(predictor).__name__)}：{detail}")


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ContractViolation(f"帧必须是 numpy 数组，实际 {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ContractViolation(f"帧必须是 uint8，实际 {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ContractViolation(
            f"帧必须是 HxWx3 的 RGB，实际形状 {image.shape}。通道序转换只在读帧处做一次，"
            f"下游不再猜——这里收到别的形状说明读帧处漏了转换。"
        )


def check_predictor_surface(predictor: object) -> None:
    """协议表面：有名字、满足 ``PerPointMaskPredictor``、两个方法可调用。"""

    name = getattr(predictor, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise _violation("surface", predictor, "缺少非空字符串 name，错误日志与运行摘要都靠它区分实现")
    if not isinstance(predictor, PerPointMaskPredictor):
        missing = [m for m in ("predict", "predict_batch") if not callable(getattr(predictor, m, None))]
        raise _violation("surface", predictor, f"不满足 PerPointMaskPredictor 协议（缺 {missing or '类型标注'}）")
    for method in ("predict", "predict_batch"):
        if not callable(getattr(predictor, method, None)):
            raise _violation("surface", predictor, f"{method} 不可调用")


def check_prediction_structure(predictor: object, case: ContractCase) -> None:
    """输出结构：``list`` → ``MaskPrediction`` → bool 掩膜、原图同尺寸、score 在 [0,1]、按分降序。"""

    image = case.scene.image
    _validate_image(image)
    result = predictor.predict(image, case.prompt)
    if not isinstance(result, list):
        raise _violation(
            "structure", predictor, f"predict 必须返回 list（找不到目标时是空列表），实际 {type(result).__name__}"
        )
    scores: list[float] = []
    height, width = case.scene.shape
    frame_area = height * width
    for index, prediction in enumerate(result):
        if not isinstance(prediction, MaskPrediction):
            raise _violation(
                "structure", predictor, f"第 {index} 项不是 MaskPrediction，实际 {type(prediction).__name__}"
            )
        mask = np.asarray(prediction.mask)
        if mask.dtype != np.bool_:
            raise _violation(
                "structure",
                predictor,
                f"第 {index} 个掩膜 dtype={mask.dtype}；协议要求 bool（面积是整数计数，"
                f"0/1 的 uint8 会在按位与上悄悄变成 0/1 相乘而不是布尔与）",
            )
        if mask.shape != (height, width):
            raise _violation(
                "structure",
                predictor,
                f"第 {index} 个掩膜 shape={mask.shape}，输入图是 ({height}, {width})。"
                f"内部下采样再放大冒充原分辨率，会让提示坐标整体错位——必须直接出原尺寸。",
            )
        score = prediction.score
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise _violation("structure", predictor, f"第 {index} 项 score 不是数字：{score!r}")
        if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise _violation("structure", predictor, f"第 {index} 项 score={score} 越界（须为 0–1 的有限值）")
        scores.append(float(score))
        area = int(np.count_nonzero(mask))
        if area > FULL_FRAME_GUARD * frame_area:
            raise _violation(
                "structure",
                predictor,
                f"第 {index} 个掩膜盖住 {area}/{frame_area} = {area / frame_area:.1%} 的画面。"
                f"这是「没找到目标于是把整张图交上去了」：面积比会被钉成 1.0，毫升数与 ETA 一起失真。"
                f"找不到目标应当返回空列表。",
            )
    if scores != sorted(scores, reverse=True):
        raise _violation("structure", predictor, f"候选未按 score 降序排列：{scores}")


def check_empty_target_returns_empty_list(predictor: object, scene: ContractScene | None = None) -> None:
    """找不到目标时返回**空列表**，不是一张全零图、也不是整帧。

    用纯色画面触发：画面里确实没有可分割的东西，「没找到」是客观事实而不是实现的判断。
    """

    scene = scene or plain_scene()
    for prompt in (Prompt.from_point(60.0, 60.0), Prompt(box=(20.0, 20.0, 200.0, 200.0)), Prompt()):
        result = predictor.predict(scene.image, prompt)
        if not isinstance(result, list):
            raise _violation("empty_target", predictor, f"提示 {prompt} 下返回的不是 list")
        if result:
            raise _violation(
                "empty_target",
                predictor,
                f"纯色画面（无任何可分割区域）上提示 {prompt} 却返回了 {len(result)} 个候选，"
                f"首个面积 {int(np.count_nonzero(result[0].mask))}px。"
                f"空列表与全零图的差别在于：空列表会被上层记成「这一帧看不见」，"
                f"全零图会被算成 0% 余量——后者会让护士长看到一条并不存在的换液提醒。",
            )


def check_degenerate_prompts_do_not_raise(predictor: object, cases: Sequence[ContractCase]) -> None:
    """退化提示（画面外点、空提示、只有负点、零面积框）必须被裁剪处理，不得抛异常。

    协议第三条：「超出画面时由实现自行裁剪而不是报错」。护士是照着袋体随手点的，点落在
    画面外几像素是常事——为此把一次巡视判失败，等于让服务在真实用法上不可用。
    """

    for case in cases:
        try:
            result = predictor.predict(case.scene.image, case.prompt)
        except Exception as exc:  # noqa: BLE001 - 契约要的就是「任何异常都算违规」
            raise _violation(
                "degenerate_prompts",
                predictor,
                f"用例 {case.key} 抛出 {type(exc).__name__}: {exc}。"
                f"画面外的提示应被裁剪处理，不是让整次上传失败。",
            ) from exc
        if not isinstance(result, list):
            raise _violation("degenerate_prompts", predictor, f"用例 {case.key} 返回的不是 list")


def check_point_lands_on_truth_blob(predictor: object, cases: Sequence[ContractCase]) -> None:
    """坐标系一致：提示指哪块，掩膜就落在哪块附近，且**不串到邻块**。

    这条是契约里最要紧的一条。dtype 与 shape 都合规而坐标系平移了几个像素的实现，
    类型系统一个错都报不出来，但它会把每一个面积比都算错。三个判据一起看：
    IoU 下限（不许漏太多）、不与禁止区域相交（不许串块）、质心误差上限（不许平移）。
    """

    for case in cases:
        if case.expect_target is None:
            continue
        truth = case.scene.truth(case.expect_target)
        result = predictor.predict(case.scene.image, case.prompt)
        if not result:
            raise _violation(
                "coordinate_alignment",
                predictor,
                f"用例 {case.key}：提示明确指向区域 {case.expect_target}，实现却返回空列表",
            )
        mask = np.asarray(result[0].mask, dtype=bool)
        iou = mask_iou(mask, truth)
        if iou < case.min_iou:
            raise _violation(
                "coordinate_alignment",
                predictor,
                f"用例 {case.key}：与区域 {case.expect_target} 的 IoU={iou:.3f} < 下限 {case.min_iou}。"
                f"掩膜落在 {mask.sum()}px，真值区域 {truth.sum()}px——坐标系可能平移或缩放了。",
            )
        if case.forbid_target is not None:
            forbidden = case.scene.truth(case.forbid_target)
            overlap = int(np.count_nonzero(mask & forbidden))
            if overlap:
                leaked = overlap / max(1, int(np.count_nonzero(forbidden)))
                if leaked > 0.1:
                    raise _violation(
                        "coordinate_alignment",
                        predictor,
                        f"用例 {case.key}：掩膜有 {leaked:.1%} 落在禁止区域 {case.forbid_target} 里，"
                        f"提示并没有指向它——典型症状是把邻袋一起框进来了。",
                    )
        centre = mask_centroid(mask)
        wanted = case.scene.blob(case.expect_target).centre
        if centre is None:
            raise _violation("coordinate_alignment", predictor, f"用例 {case.key}：掩膜为空，无法核对质心")
        error = math.hypot(centre[0] - wanted[0], centre[1] - wanted[1])
        if error > case.centroid_tolerance:
            raise _violation(
                "coordinate_alignment",
                predictor,
                f"用例 {case.key}：掩膜质心 ({centre[0]:.1f}, {centre[1]:.1f}) 与目标区域中心 "
                f"({wanted[0]:.1f}, {wanted[1]:.1f}) 相差 {error:.1f}px > {case.centroid_tolerance}px",
            )


def _same_predictions(
    left: Sequence[MaskPrediction], right: Sequence[MaskPrediction]
) -> str | None:
    if len(left) != len(right):
        return f"候选个数不同：{len(left)} vs {len(right)}"
    for index, (a, b) in enumerate(zip(left, right)):
        if np.asarray(a.mask).shape != np.asarray(b.mask).shape:
            return f"第 {index} 个掩膜 shape 不同：{np.asarray(a.mask).shape} vs {np.asarray(b.mask).shape}"
        if not np.array_equal(np.asarray(a.mask, dtype=bool), np.asarray(b.mask, dtype=bool)):
            diff = int(np.count_nonzero(np.asarray(a.mask, dtype=bool) ^ np.asarray(b.mask, dtype=bool)))
            return f"第 {index} 个掩膜逐像素不同（差 {diff}px）"
        if abs(float(a.score) - float(b.score)) > 1e-9:
            return f"第 {index} 项 score 不同：{a.score} vs {b.score}"
    return None


def check_batch_consistency(predictor: object, cases: Sequence[ContractCase]) -> None:
    """批量与逐条必须逐像素一致（枚举管线在一帧上问几十个点，两条路不能给出两套数）。"""

    scene = cases[0].scene
    image = scene.image
    prompts = [case.prompt for case in cases]
    try:
        batched = predictor.predict_batch(image, prompts)
        empty = predictor.predict_batch(image, [])
    except Exception as exc:  # noqa: BLE001
        raise _violation(
            "batch_consistency", predictor, f"predict_batch 抛出 {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(batched, list) or len(batched) != len(prompts):
        raise _violation(
            "batch_consistency",
            predictor,
            f"predict_batch 必须对每条提示返回一个列表，实际 {type(batched).__name__}"
            f"/长度 {len(batched) if isinstance(batched, list) else 'n-a'}（输入 {len(prompts)} 条）",
        )
    if list(empty):
        raise _violation("batch_consistency", predictor, f"空提示列表应返回空列表，实际 {type(empty).__name__} {len(empty)}")
    for case, per_prompt, batched_one in zip(cases, prompts, batched):
        single = predictor.predict(image, per_prompt)
        diff = _same_predictions(list(batched_one), list(single))
        if diff is not None:
            raise _violation(
                "batch_consistency", predictor, f"用例 {case.key}：批量与逐条不一致——{diff}"
            )


def check_determinism_and_purity(predictor: object, cases: Sequence[ContractCase]) -> None:
    """同一张图同一提示跑两次结果逐位相同，且**不改动输入帧**。

    不改动输入帧这一条是被真实事故逼出来的：几何核的形态学若在原图上就地写回，第二袋
    的起锚就建立在一张已经被第一袋改脏的图上，曲线会从第一帧起就错。
    """

    scene = cases[0].scene
    for case in cases:
        image = scene.image
        before = image.copy()
        first = list(predictor.predict(image, case.prompt))
        if not np.array_equal(image, before):
            changed = int(np.count_nonzero(np.any(image != before, axis=2)))
            raise _violation(
                "determinism_and_purity",
                predictor,
                f"用例 {case.key}：predict 改动了传入的帧（{changed} 个像素不同）。"
                f"帧是调用方的数据，实现必须只读。",
            )
        second = list(predictor.predict(image, case.prompt))
        diff = _same_predictions(first, second)
        if diff is not None:
            raise _violation("determinism_and_purity", predictor, f"用例 {case.key}：两次调用结果不同——{diff}")


def check_mask_png_roundtrip(mask: np.ndarray | None = None) -> None:
    """掩膜 PNG 往返必须逐位相同（含非连续切片视图）。"""

    if mask is None:
        mask = blob_scene().truth("a")
    payload = encode_mask_png(mask)
    decoded = decode_mask_png(payload)
    if decoded.shape != np.asarray(mask).shape:
        raise ContractViolation(
            f"[serialisation] PNG 往返后形状变了：{np.asarray(mask).shape} → {decoded.shape}"
        )
    if not np.array_equal(decoded, np.asarray(mask, dtype=bool)):
        lost = int(np.count_nonzero(decoded ^ np.asarray(mask, dtype=bool)))
        raise ContractViolation(f"[serialisation] PNG 往返后掩膜丢了 {lost} 个像素（二值掩膜必须无损）")
    if encode_mask_png(decoded) != payload:
        raise ContractViolation("[serialisation] PNG 往返后再编码，字节不一致——编码口径不稳定")
    view = np.asarray(mask)[::2, ::2]
    roundtrip = decode_mask_png(encode_mask_png(view))
    if not np.array_equal(roundtrip, view):
        raise ContractViolation("[serialisation] 非连续切片的掩膜 PNG 往返不一致")


# ─────────────────────────────────────────────────────────────────────────────
# 汇总跑：两个实现跑同一套，出可打印的报告
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ContractCheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class ContractReport:
    """一个实现跑完整套契约的结果。"""

    impl: str
    results: tuple[ContractCheckResult, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.results if item.ok)

    @property
    def failures(self) -> tuple[ContractCheckResult, ...]:
        return tuple(item for item in self.results if not item.ok)

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        return {
            "impl": self.impl,
            "ok": self.ok,
            "passed": list(self.passed),
            "failed": [{"check": item.name, "detail": item.detail} for item in self.failures],
        }

    def summary(self) -> str:
        lines = [f"{self.impl}：{'通过' if self.ok else '未通过'}（{len(self.passed)}/{len(self.results)} 项）"]
        lines += [f"  ✅ {name}" for name in self.passed]
        lines += [f"  ❌ {item.name} — {item.detail}" for item in self.failures]
        return "\n".join(lines)


def run_predictor_contract(
    predictor: object,
    *,
    cases: Sequence[ContractCase] | None = None,
    scene: ContractScene | None = None,
) -> ContractReport:
    """把整套契约跑在一个实现上，**不中断**地收集每条结果。

    不在第一条失败就停：一个实现可能同时踩了两条（shape 不对 + 空画面乱返回），
    一次把两条报出来比来回改两轮省事。
    """

    cases = tuple(cases) if cases is not None else contract_cases(scene)
    blob_cases = [case for case in cases if case.scene.blobs]

    plan: tuple[tuple[str, object], ...] = (
        ("surface", lambda: check_predictor_surface(predictor)),
        ("structure", lambda: [check_prediction_structure(predictor, case) for case in blob_cases]),
        ("empty_target", lambda: check_empty_target_returns_empty_list(predictor)),
        ("degenerate_prompts", lambda: check_degenerate_prompts_do_not_raise(predictor, blob_cases)),
        ("coordinate_alignment", lambda: check_point_lands_on_truth_blob(predictor, blob_cases)),
        ("batch_consistency", lambda: check_batch_consistency(predictor, blob_cases)),
        ("determinism_and_purity", lambda: check_determinism_and_purity(predictor, blob_cases)),
        ("serialisation", lambda: check_mask_png_roundtrip()),
    )

    results: list[ContractCheckResult] = []
    for name, runner in plan:
        try:
            runner()
        except ContractViolation as exc:
            results.append(ContractCheckResult(name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001 - 实现里抛什么都算违规，只是消息更粗
            results.append(
                ContractCheckResult(name, False, f"[{name}] {type(exc).__name__}: {exc}")
            )
        else:
            results.append(ContractCheckResult(name, True))
    # 固定成 CONTRACT_CHECKS 的顺序输出，报告之间才能逐行对照
    order = {name: index for index, name in enumerate(CONTRACT_CHECKS)}
    results.sort(key=lambda item: order.get(item.name, len(order)))
    return ContractReport(impl=str(getattr(predictor, "name", type(predictor).__name__)), results=tuple(results))


def summarise_reports(reports: Iterable[ContractReport]) -> str:
    """把多个实现的报告并排打成一段文本（病区侧自检与测试失败输出共用）。"""

    blocks = [report.summary() for report in reports]
    broken = [report.impl for report in reports if not report.ok]
    tail = "全部实现均满足协议契约。" if not broken else f"未满足契约的实现：{'、'.join(broken)}"
    return "\n".join([*blocks, "", tail])