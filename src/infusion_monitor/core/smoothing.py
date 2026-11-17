"""面积序列平滑：中值 + 一阶平滑、物理单调约束、缺帧受限重建。

**为什么必须有这一层。** 几何核的逐帧面积比会因反光高光、袋体形变、滴斗空管而抖动，
而毫升数与 ETA 是这条曲线的**线性**下游：袋型标定把面积比映射成毫升、滴速把毫升
换算成分钟，中间隔着的任何一帧假尖峰都会直接变成「快见底了」的换液点名。所以平滑
不是显示层的修饰，而是**点名与毫升数可信的前提**。

**三条纪律**（与 :mod:`.occlusion`、:mod:`.propagate` 同一套口径，宁可少说也不说假）：

1. **raw 与 smoothed 双序列并存。** 只回平滑值等于把「原始观测到底是什么」藏起来了；
   两者都要出，且逐帧对齐，护士/信息科能自己核。
2. **看见的帧只降级、不消失。** 物理上不可能的上升段标 ``outlier`` 并**保留原值**，
   绝不悄悄抹平或钳到前一帧——抹平了就没人知道这一帧拍坏了。
3. **看不见的帧不插值。** 遮挡期留 ``null``。受限重建是**单独一条序列**、逐帧标
   ``reconstructed``，只在「前后端点都是可信观测」且缺帧数落在 ``[min_gap_frames,
   max_gap_frames]`` 时才填；端点不全、超长、或两端液面反而升高（换液嫌疑）时**拒绝
   重建**并给出原因码。

**下游读哪条序列。** :attr:`SmoothingResult.trusted` 是给毫升/ETA/换液点名用的工作
序列：尖峰已被中值吸收、反常帧退到最近一次可信读数、看不见的帧仍是 ``None``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "FLAG_MISSING",
    "FLAG_OBSERVED",
    "FLAG_OUTLIER",
    "FLAG_RECONSTRUCTED",
    "FLAG_SPIKE",
    "GapFill",
    "FrameFlag",
    "SmoothingParams",
    "SmoothingResult",
    "first_order",
    "find_gaps",
    "median_filter",
    "reconstruct_gaps",
    "smooth_series",
]

#: 逐帧标记（:attr:`FrameFlag.label` 会用「+」拼多个，如 ``outlier+reconstructed``）。
FLAG_OBSERVED = "observed"
FLAG_MISSING = "missing"
FLAG_SPIKE = "spike"
FLAG_OUTLIER = "outlier"
FLAG_RECONSTRUCTED = "reconstructed"

#: 受限重建的拒绝原因（稳定码，护士端与日志按码取文案）。
REJECT_NO_FRONT = "no_trusted_front_endpoint"
REJECT_NO_BACK = "no_trusted_back_endpoint"
REJECT_TOO_SHORT = "gap_shorter_than_min"
REJECT_TOO_LONG = "gap_longer_than_max"
REJECT_ORDER = "endpoint_order_invalid"
REJECT_SPAN = "endpoint_span_too_small"


@dataclass(frozen=True)
class SmoothingParams:
    """平滑参数。取值理由都写在字段旁，不留「经验值」这种没头没尾的数。

    默认值按**护士手机 10–20 秒短片**（12–1200 帧、320px 短边）标定：
    输液袋在一段短片里液面变化量本身就很小（千分之几到百分之一），
    因此「单帧允许的上升」必须比一次真实抖动更小，否则形变会被当成坏帧。
    """

    median_window: int = 5
    """中值滤波窗口（按**观测点**计，``None`` 不占名额）。默认 5 而不是 3，是实测换来的：
    镜面高光打在液面上挖出来的洞往往**连着两帧**都在（液面在动，高光跟着动），
    3 点窗口只能吃掉一半（实测可信序列仍掉 0.024），5 点窗口把它压到 0.003 以内。
    代价是等斜坡上多滞后半个窗口——真实滴速下 10–20 秒的液面变化只有千分之几，
    这点滞后换掉一整段假凹陷划算。"""

    alpha: float = 0.6
    """一阶平滑中新观测的权重。偏大（0.6）是为了**不拖慢响应**：末帧读数是毫升数与
    ETA 的输入，压得太狠会让「快见底」迟到几个巡视轮次。"""

    rise_tolerance: float = 0.004
    """单帧允许的上升量（面积比）。袋体晃动 + 分割抖动实测在 0.002 以内，0.004 留一倍
    余量；再大就会把滴斗空管造成的假面积跳变当成真实上升放过。"""

    spike_tolerance: float = 0.02
    """原始值与中值偏离超过此值即标 ``spike``（反光高光典型偏离 0.03–0.08）。"""

    min_gap_frames: int = 2
    """缺 1 帧不值得编一个数出来（那多半是一次检出抖动而不是真的被挡），留 ``None``。"""

    max_gap_frames: int = 10
    """受限重建的最大缺帧数。超过就拒绝重建并报 ``gap_longer_than_max``——
    遮挡 10 帧（约 5 秒）之后两端液面差已经大到线性外推没有意义。"""

    def as_dict(self) -> dict[str, float | int]:
        return {
            "median_window": self.median_window,
            "alpha": self.alpha,
            "rise_tolerance": self.rise_tolerance,
            "spike_tolerance": self.spike_tolerance,
            "min_gap_frames": self.min_gap_frames,
            "max_gap_frames": self.max_gap_frames,
        }


@dataclass(frozen=True)
class FrameFlag:
    """逐帧标记。多个标记可同时成立（反常帧又被重建命中就是 ``outlier+reconstructed``）。"""

    index: int
    observed: bool = False
    spike: bool = False
    outlier: bool = False
    reconstructed: bool = False

    @property
    def label(self) -> str:
        parts: list[str] = []
        if self.reconstructed:
            parts.append(FLAG_RECONSTRUCTED)
        if self.outlier:
            parts.append(FLAG_OUTLIER)
        elif self.spike:
            parts.append(FLAG_SPIKE)
        if parts:
            return "+".join(parts)
        return FLAG_OBSERVED if self.observed else FLAG_MISSING

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "label": self.label,
            "observed": self.observed,
            "spike": self.spike,
            "outlier": self.outlier,
            "reconstructed": self.reconstructed,
        }


@dataclass(frozen=True)
class GapFill:
    """一段缺帧的处理结果：填了还是没填、为什么不填。"""

    start: int
    end: int
    reason: str
    filled: bool

    @property
    def length(self) -> int:
        """这段缺帧**占了几帧**（列表下标口径）。"""
        return self.end - self.start + 1

    def as_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "reason": self.reason,
            "filled": self.filled,
        }


@dataclass(frozen=True)
class SmoothingResult:
    """四条逐帧序列（都按帧号一一对齐，长度与输入相同）加诊断信息。

    * ``raw`` —— 原始观测，一个字节都没改；遮挡帧是 ``None``；
    * ``smoothed`` —— 中值 + 一阶平滑的工作序列；**反常帧保留原值**并标 ``outlier``；
    * ``trusted`` —— 下游（毫升/ETA/换液点名）真正读的那条：尖峰已被吸收、反常帧退到
      最近一次可信读数、看不见的帧仍是 ``None``；
    * ``reconstructed`` —— 受限重建的补值，只在被重建的帧上有值，其余为 ``None``。
    """

    frame_indices: list[int]
    raw: list[float | None]
    smoothed: list[float | None]
    trusted: list[float | None]
    reconstructed: list[float | None]
    flags: list[FrameFlag]
    gaps: list[GapFill] = field(default_factory=list)
    params: SmoothingParams = field(default_factory=SmoothingParams)

    # ── 派生读数 ─────────────────────────────────────────────────────
    @property
    def outlier_frames(self) -> list[int]:
        return [flag.index for flag in self.flags if flag.outlier]

    @property
    def spike_frames(self) -> list[int]:
        return [flag.index for flag in self.flags if flag.spike and not flag.outlier]

    @property
    def reconstructed_frames(self) -> list[int]:
        return [flag.index for flag in self.flags if flag.reconstructed]

    @property
    def missing_frames(self) -> list[int]:
        return [flag.index for flag in self.flags if not flag.observed]

    @property
    def degraded(self) -> bool:
        """是否存在需要护士留意的帧（尖峰 / 反常 / 缺帧）。"""
        return any(
            flag.spike or flag.outlier or not flag.observed for flag in self.flags
        )

    @property
    def latest_raw(self) -> float | None:
        for value in reversed(self.raw):
            if value is not None:
                return value
        return None

    @property
    def latest_trusted(self) -> float | None:
        """末次可信读数（= 毫升与 ETA 的输入）；整段没有可信观测时为 ``None``。"""
        for value in reversed(self.trusted):
            if value is not None:
                return value
        return None

    @property
    def drain_ratio_per_frame(self) -> float | None:
        """整段的平均降幅（面积比/帧，正数表示在漏）。只有两个可信观测点起算。"""
        points = [
            (index, value)
            for index, value in enumerate(self.trusted)
            if value is not None
        ]
        if len(points) < 2:
            return None
        (first_index, first_value), (last_index, last_value) = points[0], points[-1]
        span = self.frame_indices[last_index] - self.frame_indices[first_index]
        if span <= 0:
            return None
        return (first_value - last_value) / span

    def frame_label(self, index: int) -> str:
        return self.flags[index].label if 0 <= index < len(self.flags) else FLAG_MISSING

    def as_dict(self) -> dict[str, object]:
        """供响应与日志使用的诊断摘要（不含四条长序列本身，它们各有字段）。"""
        return {
            "outlier_frames": self.outlier_frames,
            "spike_frames": self.spike_frames,
            "reconstructed_frames": self.reconstructed_frames,
            "missing_frame_count": len(self.missing_frames),
            "drain_ratio_per_frame": self.drain_ratio_per_frame,
            "latest_raw": self.latest_raw,
            "latest_trusted": self.latest_trusted,
            "degraded": self.degraded,
            "gaps": [gap.as_dict() for gap in self.gaps],
            "params": self.params.as_dict(),
        }


#: 中值窗口至少要凑够几个点才成立（凑不够就没有中值可言，退回原值）。
MIN_MEDIAN_POINTS = 3


def median_filter(values: Sequence[float | None], *, window: int) -> list[float | None]:
    """中值滤波：孤立尖峰被吸收，平台段与单调段基本不动。

    窗口按**观测点**计（``None`` 不占名额），因此遮挡段不会被窗口边缘抹出假梯度。

    两个边界口径（都实测踩过坑）：

    * 窗口贴到序列两端时**整体内移**而不是截短——截短会得到偶数点窗口，取中间两点
      均值当「中值」，首帧于是被系统性地往上抬半分，第二帧紧跟着被判成下降；
    * 可用观测点铺不满窗口时（短段、或窗口大于整段观测数）该点**原样返回**：取不出
      稳定中值，硬取会把换液后的新高抹成旧液面，后面再拿它当重建端点就是假桥。
    """
    if window < 1:
        raise ValueError("median_window 至少为 1")
    observed_index = [i for i, value in enumerate(values) if value is not None]
    out: list[float | None] = [None] * len(values)
    width = max(window, MIN_MEDIAN_POINTS)
    if width % 2 == 0:  # 偶数窗口会退化成取均值，口径含糊，直接抬到下一个奇数
        width += 1
    half = width // 2
    available = len(observed_index)
    for slot, index in enumerate(observed_index):
        if available < MIN_MEDIAN_POINTS or width > available:
            out[index] = float(values[index])  # type: ignore[arg-type]
            continue
        left = max(0, min(slot - half, available - width))
        neighbourhood = sorted(
            float(values[observed_index[position]]) for position in range(left, left + width)
        )
        out[index] = neighbourhood[len(neighbourhood) // 2]
    return out


def first_order(
    values: Sequence[float | None], *, alpha: float, restart: Sequence[int] | None = None
) -> list[float | None]:
    """一阶平滑：``s[i] = alpha * v[i] + (1 - alpha) * s[i-1]``，``None`` 原样留空。

    缺帧不参与状态推进也不被补值——缺一帧就等于这一次没测到，不是「沿用上一帧」。

    ``restart`` 里的下标**重新起算**状态（直接取该帧观测，不与历史加权）。调用方
    :func:`smooth_series` 在两处标它：一处是长遮挡之后（上一帧观测已经是几十帧前的
    读数），一处是反常帧之后（中间夹着物理上不可能的观测，状态连续性已经断了——
    换液后的新袋若还和旧袋的液位加权，报出来的余量两头都不像）。
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha 须落在 (0, 1]")
    breakpoints = set(restart or ())
    out: list[float | None] = [None] * len(values)
    state: float | None = None
    for index, value in enumerate(values):
        if value is None:
            continue
        state = float(value) if state is None or index in breakpoints else (
            alpha * float(value) + (1.0 - alpha) * state
        )
        out[index] = state
    return out


def find_gaps(values: Sequence[float | None]) -> list[tuple[int, int]]:
    """返回连续 ``None`` 段的 ``(start, end)`` 闭区间列表（按**列表下标**）。

    需要按帧号算跨度的场合（重建、一阶平滑的隔太久重起算）见各自函数——两者口径
    不同：这里是「哪几格空着」，那里是「空了多久」。
    """
    gaps: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value is None:
            if start is None:
                start = index
        elif start is not None:
            gaps.append((start, index - 1))
            start = None
    if start is not None:
        gaps.append((start, len(values) - 1))
    return gaps


def reconstruct_gaps(
    trusted: Sequence[float | None],
    *,
    params: SmoothingParams | None = None,
    frame_indices: Sequence[int] | None = None,
    endpoints: Sequence[bool] | None = None,
) -> tuple[list[float | None], list[GapFill]]:
    """受限重建：只在**前后端点都是可信观测**且缺帧数落在区间内时线性补值。

    返回 ``(补值序列, 每段缺帧的处理记录)``。拒绝重建的五种原因都带稳定码：

    * :data:`REJECT_NO_FRONT` —— 前面没看到过（起锚前或一开拍就看不见）；
    * :data:`REJECT_NO_BACK` —— 后面还没恢复（这段短片到遮挡就结束了）；
    * :data:`REJECT_TOO_SHORT` —— 缺帧数不足 ``min_gap_frames``；
    * :data:`REJECT_TOO_LONG` —— 超过 ``max_gap_frames``；
    * :data:`REJECT_SPAN` —— 两个端点在**帧号上**挨得太近，线性外推没有意义；
    * :data:`REJECT_ORDER` —— 后端点比前端点还高。输液液面只会降，这说明中间可能
      换过袋（:mod:`.reacquire` 的场景），补出来的线是假的，一律不补。
    """
    p = params or SmoothingParams()
    indices = list(frame_indices) if frame_indices is not None else list(range(len(trusted)))
    out: list[float | None] = [None] * len(trusted)
    records: list[GapFill] = []
    def _endpoint(index: int) -> float | None:
        """能当重建端点的帧：本身有可信值，且 ``endpoints`` 没把它排除掉。"""
        if index < 0 or index >= len(trusted):
            return None
        value = trusted[index]
        if value is None:
            return None
        if endpoints is not None and not endpoints[index]:
            return None
        return value

    for start, end in find_gaps(trusted):
        front = _endpoint(start - 1)
        back = _endpoint(end + 1)
        if front is None:
            records.append(GapFill(start, end, REJECT_NO_FRONT, False))
            continue
        if back is None:
            records.append(GapFill(start, end, REJECT_NO_BACK, False))
            continue
        span = indices[end + 1] - indices[start - 1]
        if span <= 0:
            records.append(GapFill(start, end, REJECT_SPAN, False))
            continue
        # 跨度一律按**帧号**算：重传只留下部分帧时，列表上挨着的两格在时间上可能隔着
        # 好几秒，按下标判「只缺 1 帧」就会把一段长时间遮挡当成小抖动补上。
        missing = span - 1
        if missing < p.min_gap_frames:
            records.append(GapFill(start, end, REJECT_TOO_SHORT, False))
            continue
        if missing > p.max_gap_frames:
            records.append(GapFill(start, end, REJECT_TOO_LONG, False))
            continue
        if back > front:
            records.append(GapFill(start, end, REJECT_ORDER, False))
            continue
        for index in range(start, end + 1):
            left_span = indices[index] - indices[start - 1]
            out[index] = front + (back - front) * (left_span / span)
        records.append(GapFill(start, end, "reconstructed_from_endpoints", True))
    return out, records


def _mark_outliers(
    medians: Sequence[float | None], *, rise_tolerance: float
) -> tuple[list[float | None], list[int]]:
    """按物理单调约束标出反常帧：液面只会降，升上去的一帧只可能是拍坏了。

    信封（envelope）取**历史最高可信液面**而不是上一帧值：袋体晃动会让相邻帧上下抖
    零点几个百分点，用上一帧做基准会把正常抖动一路累积成「全程反常」。反常帧不推进
    信封，异常值也不会顺着信封往上抬。
    """
    trusted: list[float | None] = [None] * len(medians)
    outliers: list[int] = []
    envelope: float | None = None
    for index, value in enumerate(medians):
        if value is None:
            continue
        if envelope is not None and value > envelope + rise_tolerance:
            trusted[index] = None
            outliers.append(index)
            continue
        trusted[index] = value
        envelope = value if envelope is None else max(envelope, value)
    return trusted, outliers


def smooth_series(
    raw: Sequence[float | None],
    *,
    frame_indices: Sequence[int] | None = None,
    params: SmoothingParams | None = None,
) -> SmoothingResult:
    """一袋面积比曲线：原始观测 → 中值 → 一阶平滑 → 单调约束 → 缺帧受限重建。

    对应 ``core/propagate.py`` 产出的 ``BagCurve.area_ratio_series``：长度与输入完全
    一致，**任何一帧都不会被删除或移位**。
    """
    p = params or SmoothingParams()
    values: list[float | None] = [None if v is None else float(v) for v in raw]
    indices = list(frame_indices) if frame_indices is not None else list(range(len(values)))

    medians = median_filter(values, window=p.median_window)
    trusted_medians, outlier_frames = _mark_outliers(
        medians, rise_tolerance=p.rise_tolerance
    )
    # 重新起算的两处：上一帧观测离得太远（长遮挡），或中间夹着反常帧（换液嫌疑）。
    observed_index = [i for i, v in enumerate(trusted_medians) if v is not None]
    restart: set[int] = set()
    outliers_before: list[int] = []
    seen_outliers = 0
    for index in range(len(trusted_medians)):
        if index in outlier_frames:
            seen_outliers += 1
        outliers_before.append(seen_outliers)
    for position, index in enumerate(observed_index):
        if position == 0:
            restart.add(index)
            continue
        previous = observed_index[position - 1]
        if indices[index] - indices[previous] > p.max_gap_frames:
            restart.add(index)
        elif outliers_before[index] > outliers_before[previous]:
            restart.add(index)
    ema = first_order(trusted_medians, alpha=p.alpha, restart=sorted(restart))

    outlier_set = set(outlier_frames)
    smoothed: list[float | None] = [None] * len(values)
    trusted: list[float | None] = [None] * len(values)
    endpoint_ok: list[bool] = [False] * len(values)
    last_trusted: float | None = None
    for index, value in enumerate(values):
        observed = value is not None
        is_outlier = index in outlier_set
        if is_outlier:
            # 保留原值：反常帧不许被平滑抹平，护士端要看得见这一帧拍坏了。
            smoothed[index] = value
            trusted[index] = last_trusted
        elif observed and ema[index] is not None:
            smoothed[index] = ema[index]
            trusted[index] = ema[index]
            last_trusted = ema[index]
            endpoint_ok[index] = True

    # 重建端点取**可信序列**（平滑之后），不取中值序列：补出来的线要正好接在两端
    # 平滑值上，否则补线与观测段不在同一水平上，接缝处会出现一个假的台阶——
    # 看着像液面回升，比不补还糟。
    reconstructed, gaps = reconstruct_gaps(
        trusted, params=p, frame_indices=indices, endpoints=endpoint_ok
    )

    flags: list[FrameFlag] = []
    for index, value in enumerate(values):
        observed = value is not None
        is_outlier = index in outlier_set
        spike = False
        if observed and medians[index] is not None and ema[index] is not None:
            spike = abs(value - medians[index]) > p.spike_tolerance
        flags.append(
            FrameFlag(
                index=index,
                observed=observed,
                spike=spike,
                outlier=is_outlier,
                reconstructed=reconstructed[index] is not None,
            )
        )

    return SmoothingResult(
        frame_indices=indices,
        raw=values,
        smoothed=smoothed,
        trusted=trusted,
        reconstructed=reconstructed,
        flags=flags,
        gaps=gaps,
        params=p,
    )