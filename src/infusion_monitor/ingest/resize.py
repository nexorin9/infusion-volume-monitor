"""一段短片进来时的**规模控制**：长边缩放、ROI 窗口、流式接收的字节/时长闸门。

三者解决的是同一件事——护士手机一次巡视拍 10–20 秒、1200 帧、原始长边动辄 4000px，
而病区那台服务器要在**峰值内存不随帧数线性膨胀**的前提下把它跑完。分开做就会各自
漏掉一半：

* **长边缩放**（:func:`plan_resize`）：主路径的每个像素都要过一遍形态学，4000px 的帧
  比 1280px 贵约 10 倍，而面积比、余量、ETA 全是**比例量**——按长边缩到上限后数值
  一字不变，只是像素数少了。缩放**只在落盘时做一次**，下游读到的永远是规范帧。
* **短边下限保护**：短边低于 :data:`~ingest.validate.MIN_SHORT_SIDE`（320px）时液面
  边界不可分辨，入站硬校验直接 422。缩放必须保证**缩完的短边仍在门限之上**——一根
  4000×300 的极端长条按长边缩到 1280 会变成 1280×96，缩得越准越没法用。这条约束让
  缩放比例从「按长边一刀切」变成「长边上限与短边下限谁优先」的显式取舍，取值和
  落选理由都记进 :class:`ResizePlan.reason`（源实现只用长边一条，见 spec.md 开源对照）。
* **ROI 窗口**（:class:`RoiTracker`）：一架 6 只袋子只占手机画面的一角，画面里其余的
  是隔壁床、输液杆、天花板——那些也是「液体色」，几何核在里面照单全收。按首帧候选框
  圈出输液架所在的那块再分割，误检少一大截，单帧算力还同比下降。窗口必须**随帧平滑**：
  逐帧抖动的裁剪框会让同一只袋子的面积曲线凭空多出几段假波谷，而毫升数与 ETA 就挂
  在这条曲线的下游。
* **流式闸门**（:class:`UploadBudget`）：整段上传按块落盘，超预算/超时立刻退回并指向
  续传接口（task 32 的 ``POST /api/v1/infusion/uploads``），而不是把一个大包读进内存
  再超时——病区网络断在 90% 是常态，不是异常。

**坐标系纪律**：本模块只做一次坐标变换，方向是「相机原始像素 → 规范帧像素」。
:func:`FrameGeometry.to_camera_point` 反向可算，因此响应里的候选框、ROI 框都能换算回
护士手机画面的原始像素；缩放比例本身也随 :class:`ResizeReport` 落库，换算链不丢。
掩膜一律**贴回整幅规范帧**（:meth:`WindowedPredictor.predict`），因此
``PerPointMaskPredictor`` 协议第 2 条（掩膜与原图同尺寸同坐标系）不被破窗——ROI 少喂
的是画面，不是坐标。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from ..core.predictor import MaskPrediction, Prompt, mask_bbox
from ..errors import ApiError

__all__ = [
    "DEFAULT_LONG_EDGE_MAX",
    "DEFAULT_SHORT_EDGE_FLOOR",
    "RESIZE_REASONS",
    "FrameGeometry",
    "RoiBox",
    "RoiParams",
    "RoiTracker",
    "RoiWindowState",
    "ResizePlan",
    "ResizePolicy",
    "ResizeReport",
    "UploadBudget",
    "WindowedPredictor",
    "plan_resize",
    "resize_array",
    "resize_frame_file",
    "resize_report_for_frames",
]

#: 长边缩放上限。1280 是主路径算力与液面边界分辨率的折中：几何核的形态学核是
#: 固定像素尺度的，再大只是更慢，不会更准（面积比是比例量，见 ``calibration/models``）。
DEFAULT_LONG_EDGE_MAX = 1280
#: 短边下限。与 ``ingest.validate.MIN_SHORT_SIDE`` 同一数值——缩放**不得**把画面缩到
#: 硬校验门限之下，否则同一段短片会因为「缩放策略」而不是「拍法」被判 422。
DEFAULT_SHORT_EDGE_FLOOR = 320

#: 缩放取值理由（进响应与审计，便于信息科解释「为什么这段没缩」）。
RESIZE_REASONS = (
    "long_edge_capped",        # 按长边上限缩到位
    "short_edge_floor_wins",   # 画面长宽比太极端，保短边优先，长边没缩到上限
    "short_edge_below_floor",  # 短边本就低于门限：不缩不放，原样交出去让校验/预检判
    "disabled",                # 配置关掉了缩放
)

_RESAMPLE = Image.Resampling.BILINEAR
#: 缩小超过一半时先做一次盒式降采样：肉眼无损、CPU 省一大截（与源产品的
#: ``ResizeLongestSide`` 之后统一走定长模型输入的口径同源，这里只是把它前移）。
_REDUCING_GAP = 2.0


# ── 缩放策略与取值 ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResizePolicy:
    """入站缩放策略。默认值对护士手机短片标定过（长边 1280 / 短边不低于 320）。"""

    long_edge_max: int = DEFAULT_LONG_EDGE_MAX
    short_edge_floor: int = DEFAULT_SHORT_EDGE_FLOOR
    enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "long_edge_max": self.long_edge_max,
            "short_edge_floor": self.short_edge_floor,
        }


@dataclass(frozen=True)
class ResizePlan:
    """一帧（或一段同尺寸帧）的缩放取值。尺寸一律 ``(width, height)``。"""

    original_size: tuple[int, int]
    output_size: tuple[int, int]
    scale: float
    reason: str | None = None

    @property
    def resized(self) -> bool:
        return self.output_size != self.original_size

    @property
    def short_side(self) -> int:
        return min(self.output_size)

    @property
    def long_side(self) -> int:
        return max(self.output_size)

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_width": self.original_size[0],
            "camera_height": self.original_size[1],
            "canonical_width": self.output_size[0],
            "canonical_height": self.output_size[1],
            "scale": round(self.scale, 6),
            "resized": self.resized,
            "reason": self.reason,
        }

    @classmethod
    def identity(cls, width: int, height: int, reason: str | None = None) -> "ResizePlan":
        return cls((int(width), int(height)), (int(width), int(height)), 1.0, reason)


def plan_resize(width: int, height: int, policy: ResizePolicy | None = None) -> ResizePlan:
    """算出该缩到多少，并说清为什么这么缩。

    约束有三条，冲突时**短边下限优先**：

    1. 长边不得超 ``policy.long_edge_max``（算力预算）；
    2. 短边不得低于 ``policy.short_edge_floor``（液面边界可分辨 + 不破入站硬校验）；
    3. 只缩不放（放大只会放大噪声，不会多出信息）。
    """
    policy = policy or ResizePolicy()
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"帧尺寸必须是正整数，收到 {width}×{height}")
    original = (width, height)
    if not policy.enabled:
        return ResizePlan.identity(width, height, "disabled")

    long_edge, short_edge = max(original), min(original)
    if long_edge <= policy.long_edge_max:
        # 不需要缩；短边本就低于门限时把理由记下来（画面交出去由校验/预检去判，
        # 这里**不**擅自放大——放大会把原本被 422 挡住的拍法放进来）。
        reason = "short_edge_below_floor" if short_edge < policy.short_edge_floor else None
        return ResizePlan.identity(width, height, reason)

    s_long = policy.long_edge_max / long_edge
    s_floor = policy.short_edge_floor / short_edge
    if s_floor > 1.0:
        # 短边已经低于门限：一缩更糟，原样放行。
        scale, reason = 1.0, "short_edge_below_floor"
    elif s_floor > s_long:
        # 长宽比太极端：缩到长边上限会把短边压到门限以下，此时保短边，长边超额。
        scale, reason = s_floor, "short_edge_floor_wins"
    else:
        scale, reason = s_long, "long_edge_capped"

    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    if min(out_w, out_h) < policy.short_edge_floor and scale > s_floor:
        # 取整可能再削掉一两像素，短边因此贴到门限以下——回头按短边门限重算。
        scale = s_floor
        out_w = max(1, int(round(width * scale)))
        out_h = max(1, int(round(height * scale)))
        reason = "short_edge_floor_wins"
    applied = max(out_w, out_h) / long_edge
    return ResizePlan(original, (out_w, out_h), applied, reason)


@dataclass(frozen=True)
class ResizeReport:
    """一段短片的缩放结论（以首帧为准——帧序列本来就该同尺寸）。"""

    plan: ResizePlan
    resized_frames: int
    total_frames: int

    @property
    def camera_size(self) -> tuple[int, int]:
        return self.plan.original_size

    @property
    def canonical_size(self) -> tuple[int, int]:
        return self.plan.output_size

    @property
    def scale(self) -> float:
        return self.plan.scale

    @property
    def resized(self) -> bool:
        return self.resized_frames > 0

    def as_dict(self) -> dict[str, Any]:
        payload = self.plan.as_dict()
        payload["resized_frames"] = self.resized_frames
        payload["total_frames"] = self.total_frames
        return payload


def _reduce_gap_for(scale: float) -> float | None:
    return _REDUCING_GAP if scale <= 0.5 else None


def resize_array(image: np.ndarray, plan: ResizePlan) -> np.ndarray:
    """按取值缩放一幅 ``HxWx3`` RGB 帧。

    走 PIL 而非 OpenCV：落盘那条路（:func:`resize_frame_file`）也是 PIL，两处同一套
    重采样，像素级一致——「测试里缩出来的样子」与「服务里缩出来的样子」不能有两副面孔。
    """
    if not plan.resized:
        return image
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"期望 HxWx3 的 RGB 帧，实际形状 {image.shape}")
    with Image.fromarray(image) as source:
        resized = source.resize(
            plan.output_size, _RESAMPLE, reducing_gap=_reduce_gap_for(plan.scale)
        )
        return np.asarray(resized, dtype=np.uint8)


def resize_frame_file(
    path: Path | str,
    policy: ResizePolicy | None = None,
    *,
    quality: int = 92,
    out_path: Path | str | None = None,
) -> ResizePlan:
    """就地把一帧缩到策略上限，返回实际用的取值。

    **不需要缩时一个字节都不重写**——已经合规的短片保持原样，不因「过了一趟缩放
    流水线」而被重新编码掉画质。这是流式解包这条路的默认行为：1200 帧里绝大多数
    根本不缩。
    """
    policy = policy or ResizePolicy()
    path = Path(path)
    with Image.open(path) as source:
        plan = plan_resize(source.width, source.height, policy)
        if not plan.resized:
            return plan
        resized = source.resize(
            plan.output_size, _RESAMPLE, reducing_gap=_reduce_gap_for(plan.scale)
        )
        target = Path(out_path) if out_path is not None else path
        if target == path:
            # 原子替换：解包中途被中断时磁盘上要么是原帧、要么是新帧，不会留半张。
            staging = path.with_name(path.name + ".resize-tmp")
        else:
            staging = target
        resized.save(staging, format="JPEG", quality=quality, subsampling=1)
        if target == path:
            os.replace(staging, path)
    return plan


def resize_report_for_frames(
    paths: Sequence[Path], policy: ResizePolicy | None = None
) -> ResizeReport | None:
    """已落盘的一组帧的缩放结论（分片续传提交时用；只读 JPEG 头，不解码像素）。"""
    if not paths:
        return None
    with Image.open(paths[0]) as first:
        plan = plan_resize(first.width, first.height, policy)
    return ResizeReport(plan=plan, resized_frames=0, total_frames=len(paths))


# ── ROI 窗口 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoiBox:
    """规范帧坐标系下的裁剪窗（左上角 + 宽高，整数像素）。"""

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_xyxy(cls, x0: float, y0: float, x1: float, y1: float) -> "RoiBox":
        left, top = int(math.floor(min(x0, x1))), int(math.floor(min(y0, y1)))
        right, bottom = int(math.ceil(max(x0, x1))), int(math.ceil(max(y0, y1)))
        return cls(left, top, max(1, right - left), max(1, bottom - top))

    @classmethod
    def full(cls, shape: tuple[int, int]) -> "RoiBox":
        height, width = shape
        return cls(0, 0, int(width), int(height))

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.xyxy

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def clip(self, shape: tuple[int, int]) -> "RoiBox | None":
        """裁到画面内；完全出画返回 None（**不**给一个空窗去喂分割）。"""
        height, width = shape
        x0, y0 = max(0, self.x), max(0, self.y)
        x1, y1 = min(int(width), self.x + self.width), min(int(height), self.y + self.height)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        return RoiBox(x0, y0, x1 - x0, y1 - y0)

    def is_full(self, shape: tuple[int, int]) -> bool:
        height, width = shape
        return self.x <= 0 and self.y <= 0 and self.width >= width and self.height >= height

    def expand_by(self, margin: int) -> "RoiBox":
        if margin <= 0:
            return self
        return RoiBox(
            self.x - margin, self.y - margin, self.width + 2 * margin, self.height + 2 * margin
        )

    def union(self, other: "RoiBox") -> "RoiBox":
        x0, y0 = min(self.x, other.x), min(self.y, other.y)
        x1, y1 = max(self.x + self.width, other.x + other.width), max(
            self.y + self.height, other.y + other.height
        )
        return RoiBox(x0, y0, x1 - x0, y1 - y0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def contains_box(self, other: "RoiBox", tolerance: int = 2) -> bool:
        """``other`` 是否整个落在本窗内（留 ``tolerance`` 像素余量）。"""
        ox0, oy0, ox1, oy1 = other.xyxy
        return (
            ox0 >= self.x + tolerance
            and oy0 >= self.y + tolerance
            and ox1 <= self.x + self.width - tolerance
            and oy1 <= self.y + self.height - tolerance
        )

    def moved_towards(self, target: "RoiBox", alpha: float, max_step: int) -> tuple["RoiBox", int]:
        """朝 ``target`` 走一步：先按 ``alpha`` 做指数平滑，再逐边夹住 ``max_step``。

        返回 ``(新窗口, 本次实际位移)``。位移以**四条边各自的最大移动量**计，信息科
        看一眼就知道窗口有没有在抖。
        """
        step = 0
        edges = []
        for current, wanted in zip(self.xyxy, target.xyxy):
            delta = int(round((wanted - current) * alpha))
            if delta > max_step:
                delta = max_step
            elif delta < -max_step:
                delta = -max_step
            edges.append(current + delta)
            step = max(step, abs(delta))
        return RoiBox.from_xyxy(*edges), step

    def touches(self, other: "RoiBox", tolerance: int = 2) -> bool:
        """``other`` 是否贴到本窗边缘（物体被窗边切了的信号）。"""
        ox0, oy0, ox1, oy1 = other.xyxy
        return (
            ox0 <= self.x + tolerance
            or oy0 <= self.y + tolerance
            or ox1 >= self.x + self.width - tolerance
            or oy1 >= self.y + self.height - tolerance
        )


@dataclass(frozen=True)
class RoiParams:
    """ROI 窗口的三个门限。取值按「6 袋挂架 + 手持微抖」标定。"""

    enabled: bool = True
    #: 窗口相对候选框向外放多少（按长边的比例）——袋子随液面下降会变矮，窗口要留余量
    margin_ratio: float = 0.12
    #: 单次调用窗口边缘最多移动多少（长边比例）：防抖，也防场景突变时一帧跳走
    max_step_ratio: float = 0.05
    #: 指数平滑系数：新观测的权重
    alpha: float = 0.35
    #: 窗口短边下限：比这更小的窗没有分辨力，宁可不开窗
    min_side: int = 96

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "margin_ratio": self.margin_ratio,
            "max_step_ratio": self.max_step_ratio,
            "alpha": self.alpha,
            "min_side": self.min_side,
        }


@dataclass
class RoiWindowState:
    """窗口的运行记录（进响应，让信息科能回答「这一段到底裁没剪」）。"""

    window: RoiBox | None = None
    applied_calls: int = 0
    advances: int = 0
    max_step_px: int = 0
    border_touches: int = 0
    #: 窗口可以**平移、可以涨**，但不得比这个初值更小（见 :meth:`RoiTracker.advance`）
    anchor_width: int | None = None
    anchor_height: int | None = None
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.fallback_reason is None,
            "applied": self.applied_calls > 0,
            "window": self.window.as_dict() if self.window else None,
            "applied_calls": self.applied_calls,
            "advances": self.advances,
            "max_step_px": self.max_step_px,
            "border_touches": self.border_touches,
            "anchor_width": self.anchor_width,
            "anchor_height": self.anchor_height,
            "fallback_reason": self.fallback_reason,
        }


class RoiTracker:
    """ROI 窗口的初值与逐帧平滑。

    初值来自**首帧枚举出的候选框**（护士不用点像素）：把认得上的袋子框并起来、外放
    一点余量、裁进画面。此后每一帧按上一帧真见到的掩膜框把窗口挪一小步。

    两条不能省的细节：

    * **窗口只涨不缩**（见 :meth:`advance`）：窗内观测完全不动窗，观测顶到窗边才向外扩。
      这一条同时挡掉两种走形——逐对象传播时窗口在几只袋子之间摆（每只袋子被切一刀），
      以及液面下降时窗口一路收到只剩液体那一小块（袋子上半截被裁，毫升数失真）。
      :attr:`RoiWindowState.anchor_width/anchor_height` 记下初值尺寸，进响应可核对。
    * **退化即退全幅**：候选框为空、窗口比 ``min_side`` 还小、或并集本来就等于整幅
      画面时都不开窗，并把理由如实记在 :class:`RoiWindowState.fallback_reason`——
      宁可少一层优化，也不能让「窗开错了」变成一条假的面积曲线。
    """

    def __init__(self, params: RoiParams | None = None, shape: tuple[int, int] | None = None):
        self.params = params or RoiParams()
        self.state = RoiWindowState()
        self._shape = shape

    @property
    def window(self) -> RoiBox | None:
        return self.state.window

    def observe_shape(self, shape: tuple[int, int]) -> None:
        self._shape = (int(shape[0]), int(shape[1]))

    def _require_shape(self) -> tuple[int, int]:
        if self._shape is None:
            raise ValueError("RoiTracker 尚未拿到画面尺寸；先调 observe_shape() 或 initial_window()")
        return self._shape

    def _margin_px(self, shape: tuple[int, int]) -> int:
        long_edge = max(shape)
        return int(round(self.params.margin_ratio * long_edge))

    def initial_window(
        self, boxes: Iterable[Sequence[int]], shape: tuple[int, int]
    ) -> RoiBox | None:
        """按首帧候选框定初值窗口；不成立就返回 None 并记下退全幅的理由。"""
        self.observe_shape(shape)
        state = self.state
        if not self.params.enabled:
            state.fallback_reason = "disabled"
            return None
        union: RoiBox | None = None
        for box in boxes:
            if box is None or len(tuple(box)) != 4:
                continue
            candidate = RoiBox.from_xyxy(*box)
            if candidate.area <= 0:
                continue
            union = candidate if union is None else union.union(candidate)
        if union is None:
            state.fallback_reason = "no_candidate_boxes"
            return None
        window = union.expand_by(self._margin_px(shape)).clip(shape)
        if window is None or window.short_side < self.params.min_side:
            state.fallback_reason = "window_too_small"
            return None
        if window.is_full(shape):
            state.fallback_reason = "covers_full_frame"
            return None
        state.window = window
        state.anchor_width = window.width
        state.anchor_height = window.height
        state.fallback_reason = None
        return window

    def advance(self, boxes: Iterable[Sequence[int] | None]) -> RoiBox | None:
        """按本帧真见到的掩膜框调整窗口；无窗口时什么都不做（保持退全幅）。

        不变量只有一条：**窗口只涨不缩**（除非涨满整幅）。由此两条推论都是免费的：

        * **窗内的观测完全不动窗**。逐对象传播时一次 ``predict`` 只会看到一只袋子，若跟着
          这一只重定位，窗口就会在几只袋子之间来回摆——四袋挂架上，轮到最左边那只时窗心
          偏左（把它左侧切掉），轮到最右边那只时又偏右，于是每只袋子都在自己的那一半
          时间里被切了一刀，面积比随之失真。初始窗口是全部候选框的并集，本来就罩住整架。
        * **液面下降不会让窗口收窄**。液体掩膜一路变矮时 ``touches`` 全程不响（掩膜是
          **离开**边框，不是贴上），早期实现却顺着它一路收到只剩液体那一小块，把袋子上
          半截——正是标定要量的腔体上沿——裁了出去，毫升数随之失真（实测面积比末值
          0.5252 被压成 0.2063）。

        增长仍然走指数平滑 + 单步硬上限：单帧误检不会把窗口一帧撑满，那正是平滑要挡的。
        """
        state = self.state
        if state.window is None or not self.params.enabled:
            return state.window
        shape = self._require_shape()
        current = state.window
        seen = [RoiBox.from_xyxy(*box) for box in boxes if box is not None]
        if not seen:
            return current
        touching = any(current.touches(box) for box in seen)
        if all(current.contains_box(box) and not touching for box in seen):
            return current
        if touching:
            state.border_touches += 1

        union = seen[0]
        for box in seen[1:]:
            union = union.union(box)
        # 「窗 ∪ 观测」只会变大——这就是不变量本身，不需要另写一段垫尺寸的逻辑
        target = current.union(union).expand_by(self._margin_px(shape)).clip(shape)
        if target is None or target.short_side < self.params.min_side:
            return current
        moved, step = current.moved_towards(
            target, self.params.alpha, max(1, int(self.params.max_step_ratio * max(shape)))
        )
        state.advances += 1
        state.max_step_px = max(state.max_step_px, step)
        state.window = moved
        return moved

    def note_applied(self) -> None:
        self.state.applied_calls += 1


@dataclass(frozen=True)
class FrameGeometry:
    """一次上传的帧几何：相机原始尺寸 ↔ 规范帧尺寸 ↔ 分析窗口。"""

    camera_size: tuple[int, int]
    canonical_size: tuple[int, int]
    plan: ResizePlan
    roi: RoiBox | None = None

    @property
    def scale(self) -> float:
        return self.plan.scale

    @property
    def resized(self) -> bool:
        return self.plan.resized

    def to_camera_point(self, x: float, y: float) -> tuple[float, float]:
        """规范帧坐标 → 相机原始像素坐标（缩放与 ROI 都不影响这一条换算的线性性）。"""
        if self.scale <= 0:
            return (x, y)
        return (x / self.scale, y / self.scale)

    def to_camera_box(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        x0, y0 = self.to_camera_point(box[0], box[1])
        x1, y1 = self.to_camera_point(box[2], box[3])
        return (x0, y0, x1, y1)

    def as_dict(self) -> dict[str, Any]:
        payload = self.plan.as_dict()
        payload["roi"] = self.roi.as_dict() if self.roi else None
        return payload


class WindowedPredictor:
    """**只把输液架那块画面喂进预测器**的装饰器，掩膜照旧贴回整幅规范帧。

    这是 :class:`RoiTracker` 挂在主路径上的方式：预测器协议不变（点/框进、同尺寸掩膜
    出），业务层（``core/propagate.py``、``core/multi_object.py``）一行不用改。换句话说
    ROI 少喂的是**画面**，不是**坐标**——面积比、槽位几何、候选框、跨会话重锚全部还在
    同一个坐标系里。

    其余属性（``estimate_cavity`` 等）原样转发给内层实现：袋腔面积在整幅图上量与在
    窗内量是同一个像素数（袋子完整落在窗内），因此不需要跟着平移，硬转发反而更不容易错。
    """

    def __init__(self, inner: Any, tracker: RoiTracker):
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "tracker", tracker)
        object.__setattr__(self, "calls", 0)

    # 协议入口
    def predict(self, image: np.ndarray, prompt: Prompt) -> Sequence[MaskPrediction]:
        tracker = self.tracker
        tracker.observe_shape(image.shape[:2])
        window = tracker.window
        if window is not None:
            # 再按**这一帧真实的尺寸**夹一次：帧序列本该同尺寸，真混了尺寸时
            # （只收了前几片、或混尺寸 zip）宁可退整帧也不能拿一个越界的窗去裁——
            # 那会让掩膜尺寸与画面对不上，面积比直接算错。
            window = window.clip(image.shape[:2])
        if window is None or window.is_full(image.shape[:2]):
            predictions = self.inner.predict(image, prompt)
        else:
            tracker.note_applied()
            crop = image[
                window.y : window.y + window.height, window.x : window.x + window.width
            ]
            shifted = _shift_prompt(prompt, -window.x, -window.y)
            raw = self.inner.predict(crop, shifted) if crop.size else []
            # 必须交回 **list**：协议第 1 条（找不到目标是空列表，不是 tuple/生成器），
            # 契约套件会按类型断言，别在装饰器这一层把 list 变成 tuple。
            predictions = [
                _paste_prediction(item, window, image.shape[:2]) for item in raw
            ]
        tracker.advance(
            mask_bbox(item.mask) for item in predictions if getattr(item, "mask", None) is not None
        )
        object.__setattr__(self, "calls", self.calls + 1)
        return predictions

    def __getattr__(self, name: str) -> Any:  # 转发 estimate_cavity 等其余能力
        try:
            inner = object.__getattribute__(self, "inner")
        except AttributeError:  # pragma: no cover - 构造未完成
            raise AttributeError(name) from None
        return getattr(inner, name)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<WindowedPredictor inner={type(self.inner).__name__} window={self.tracker.window}>"


def _shift_prompt(prompt: Prompt, dx: int, dy: int) -> Prompt:
    return Prompt(
        points=tuple(
            type(point)(point.x + dx, point.y + dy, point.label) for point in prompt.points
        ),
        box=None
        if prompt.box is None
        else (prompt.box[0] + dx, prompt.box[1] + dy, prompt.box[2] + dx, prompt.box[3] + dy),
    )


def _paste_mask(mask: np.ndarray, window: RoiBox, shape: tuple[int, int]) -> np.ndarray:
    """把窗内掩膜贴回整幅规范帧（掩膜与原图同尺寸——协议第 2 条）。"""
    canvas = np.zeros((int(shape[0]), int(shape[1])), dtype=bool)
    target_h = min(window.height, canvas.shape[0] - window.y)
    target_w = min(window.width, canvas.shape[1] - window.x)
    if target_h > 0 and target_w > 0:
        canvas[window.y : window.y + target_h, window.x : window.x + target_w] = mask[
            :target_h, :target_w
        ]
    return canvas


def _paste_prediction(
    item: MaskPrediction, window: RoiBox, shape: tuple[int, int]
) -> MaskPrediction:
    """贴回整帧并**原样保留 score**——形状规整度是几何核的证据，不能在搬运里抹成 0。"""
    if tuple(item.mask.shape[:2]) != (window.height, window.width):
        raise ValueError(
            f"预测器返回的掩膜 {tuple(item.mask.shape[:2])} 与输入裁剪窗 "
            f"{(window.height, window.width)} 不同尺寸；协议要求掩膜与输入同尺寸同坐标系"
        )
    return MaskPrediction(mask=_paste_mask(item.mask, window, shape), score=item.score)


# ── 流式接收的字节/时长闸门 ────────────────────────────────────────────────


@dataclass(frozen=True)
class UploadBudget:
    """整段上传的规模闸门。超限立刻退回并指向续传接口，而不是读进内存再超时。"""

    max_bytes: int | None = None
    max_seconds: float | None = None

    def check(self, received: int, elapsed: float) -> None:
        if self.max_bytes is not None and received > self.max_bytes:
            raise ApiError(
                "UPLOAD_BUDGET_EXCEEDED",
                f"这一段已收到 {received / 1024 / 1024:.0f}MB，超过单次整段上传上限 "
                f"{self.max_bytes / 1024 / 1024:.0f}MB",
            )
        if self.max_seconds is not None and elapsed > self.max_seconds:
            raise ApiError(
                "UPLOAD_TIMEOUT",
                f"接收这一段已用 {elapsed:.0f} 秒，超过单次整段上传时限 {self.max_seconds:.0f} 秒",
            )

    def as_dict(self) -> dict[str, Any]:
        return {"max_bytes": self.max_bytes, "max_seconds": self.max_seconds}