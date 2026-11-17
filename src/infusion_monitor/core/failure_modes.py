"""拍摄质量分类：把「拍不好」变成护士能照做的下一步。

**为什么单独一层**：主路径已经能对拍得不好的短片给出面积比曲线，但那些数字**不可信**，
却看不出哪里不可信。与其默默给一个偏低的毫升数，不如把「这段拍糊了」讲清楚，并给出一句
护士当场就能照做的补拍动作。分类只**降级并标记**，从不拒绝上传——拒绝权只留给硬校验
（帧数 / 分辨率 / 帧序，见 :mod:`..ingest.validate`）。

**六类失败模式**（按「该先修哪个」排序，``PRIORITY`` 是唯一的排序真源）：

===================  =========================  ==========================================
码                    业务说法                     判据
===================  =========================  ==========================================
``bag_out_of_frame``  袋体出框                    液体/腔体贴到画面边界
``motion_blur``      运动模糊                   拉普拉斯方差（清晰度）低于门限
``glare_overexposed``  反光过曝                   近白无彩色像素占比超门限
``bad_angle``        角度不正                   液面线最小二乘拟合的倾角超门限
``low_light``        暗光                       平均亮度与暗像素占比同时过低
``empty_drip_chamber`` 滴斗空管                  腔体底部窄带填充率过低（**info**，不是拍摄问题）
===================  =========================  ==========================================

**与源主路径的关系**：源产品的 ``stability_score_thresh`` / ``pred_iou_thresh`` 把
「不确定的预测」筛掉、并让掩膜解码在稳定性不够时退回多候选而不是硬选一个（见
``github_ref/sam2/sam2/build_sam.py`` 的 ``dynamic_multimask_via_stability`` 与
``github_ref/sam2/sam2/automatic_mask_generator.py`` 的两个阈值）。那层做的是**预测空间**
的可信门控；本模块把同一套「不硬信、标出来并说明」的语义搬到**拍摄空间**：输入本身拍坏时
不产出可信结论，而是标 ``degraded`` 并给出一条可执行的补救。

**算力**：只均匀采样 ``sample_frames`` 帧（默认 5，与序列长度无关），每帧一次预测，
腔体只在首采样帧估一次。1200 帧与 24 帧的质检开销相同。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from .geom_kernel import estimate_cavity
from .predictor import PerPointMaskPredictor, Prompt

__all__ = [
    "FAILURE_CODES",
    "PRIORITY",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "CaptureQuality",
    "FailureFlag",
    "FailureModeParams",
    "FrameMeasurement",
    "classify_capture",
    "classify_measurements",
    "measure_frame",
    "render_reshoot_text",
    "sample_indices",
    "surface_tilt",
]


#: 严重度：``warning`` 会让本次结果标 ``degraded``；``info`` 只解释现象，不建议重拍。
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

#: 六类失败模式（机器可读码）。
FAILURE_CODES = (
    "bag_out_of_frame",
    "motion_blur",
    "glare_overexposed",
    "bad_angle",
    "low_light",
    "empty_drip_chamber",
)

#: 补拍优先级：**只提示最该先修的那一条**。排在前面的修好后，后面的多半会自动消失。
PRIORITY = {
    "bag_out_of_frame": 0,
    "motion_blur": 1,
    "glare_overexposed": 2,
    "bad_angle": 3,
    "low_light": 4,
    "empty_drip_chamber": 5,
}


@dataclass(frozen=True)
class FailureModeParams:
    """分类门限。全部按「工作分辨率下的像素 / 像素² / 度」计，可在配置面覆盖。"""

    #: 均匀采样多少帧做质检（与序列长度无关，保证恒定算力）。
    sample_frames: int = 5
    #: 一段里至少多少比例的采样帧命中才算这一段有问题（抗单帧误判）。
    min_hit_ratio: float = 0.34
    # ── 反光过曝 ────────────────────────────────────────────────────
    #: 近白判定：三通道最小值 ≥ 该值即算「白到没有颜色」。
    glare_channel_min: int = 242
    #: 近白像素占全画面的比例门限。实测：带反光高光的短片 ≈0.0030，其余拍法 0.0000。
    glare_ratio_threshold: float = 0.0015
    # ── 暗光 ────────────────────────────────────────────────────────
    #: 暗像素判定的灰度门限。
    dark_luma: int = 34
    #: 暗像素占比**参考线**——只进证据，不单独触发（实测它在各拍法间没有区分力：
    #: 夜间 0.56 与「站得太近」0.58 几乎一样，单独拿它判会把好片误杀）。
    dark_ratio_threshold: float = 0.62
    #: 全画面平均灰度门限。实测：夜间暗光 ≈42–44，正常拍法 ≥80，间隔足够大。
    low_light_mean_luma: float = 46.0
    # ── 运动模糊 ────────────────────────────────────────────────────
    #: 拉普拉斯方差（清晰度）下限。实测：拖影 ≈87，轻微手抖 ≈242，取中间值 150。
    sharpness_min: float = 150.0
    # ── 角度不正 ────────────────────────────────────────────────────
    #: 液面线倾角门限（度）。实测：歪 20° 的拍法 5–24°，正常拍法 ≤2.1°。
    tilt_angle_deg: float = 8.0
    #: 参与倾角拟合的最少列数（太少则判不出，返回 None 而不是猜一个）。
    tilt_min_columns: int = 16
    # ── 袋体出框 ────────────────────────────────────────────────────
    #: 掩膜外接框距画面边界多少像素内算「贴边」。
    border_px: int = 3
    # ── 滴斗空管 ────────────────────────────────────────────────────
    #: 腔体底部若干行的平均宽度 / 最宽行宽度。低于门限说明底部收成一条细杆。
    #: 实测：接空滴斗 ≈0.26，正常袋体 ≈0.99，轻微手抖 ≥0.73。
    drip_tail_ratio: float = 0.55
    #: 取腔体底部这么多行做窄带统计。
    drip_tail_rows: int = 8

    def as_dict(self) -> dict[str, Any]:
        """配置摘要（进运行摘要与响应，便于信息科核对现场为什么被标 degraded）。"""
        return {
            "sample_frames": self.sample_frames,
            "min_hit_ratio": self.min_hit_ratio,
            "glare_ratio_threshold": self.glare_ratio_threshold,
            "dark_ratio_threshold": self.dark_ratio_threshold,
            "low_light_mean_luma": self.low_light_mean_luma,
            "sharpness_min": self.sharpness_min,
            "tilt_angle_deg": self.tilt_angle_deg,
            "border_px": self.border_px,
            "drip_tail_ratio": self.drip_tail_ratio,
        }


@dataclass(frozen=True)
class FrameMeasurement:
    """一帧的拍摄质量测量值。全部是**观测量**，不含业务判断。"""

    luma_mean: float
    dark_ratio: float
    glare_ratio: float
    sharpness: float
    #: 液面线倾角（度）；有效列数不足时为 None（判不出就说判不出）。
    tilt_deg: float | None
    #: 液体/腔体外接框到画面边界的最小距离（px）；没量到掩膜时为 None。
    border_gap_px: int | None
    #: 腔体底部窄带填充率（0–1）；判不出时为 None。
    tail_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "luma_mean": round(self.luma_mean, 2),
            "dark_ratio": round(self.dark_ratio, 4),
            "glare_ratio": round(self.glare_ratio, 6),
            "sharpness": round(self.sharpness, 2),
            "tilt_deg": None if self.tilt_deg is None else round(self.tilt_deg, 2),
            "border_gap_px": self.border_gap_px,
            "tail_ratio": None if self.tail_ratio is None else round(self.tail_ratio, 4),
        }


@dataclass(frozen=True)
class FailureFlag:
    """一条失败模式：机器码 + 中文说明 + 补拍建议 + 支撑证据。"""

    code: str
    severity: str
    label_zh: str
    detail_zh: str
    suggestion_zh: str
    hit_frames: int
    sampled_frames: int
    evidence: dict[str, Any]

    @property
    def hit_ratio(self) -> float:
        return self.hit_frames / self.sampled_frames if self.sampled_frames else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "label_zh": self.label_zh,
            "detail_zh": self.detail_zh,
            "suggestion_zh": self.suggestion_zh,
            "hit_frames": self.hit_frames,
            "sampled_frames": self.sampled_frames,
            "hit_ratio": round(self.hit_ratio, 4),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CaptureQuality:
    """一段短片（对某一只袋子）的拍摄质量结论。"""

    flags: tuple[FailureFlag, ...] = ()
    sampled_frames: int = 0
    #: 存在 warning 级命中即降级；只有 info 级（空滴斗）不算拍坏。
    degraded: bool = False
    #: **只提示最该补的那一条**；没有 warning 时为 None。
    primary: FailureFlag | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "degraded": self.degraded,
            "sampled_frames": self.sampled_frames,
            "primary": None if self.primary is None else self.primary.to_dict(),
            "flags": [flag.to_dict() for flag in self.flags],
        }


# ── 测量 ────────────────────────────────────────────────────────────────


def sample_indices(total: int, limit: int) -> tuple[int, ...]:
    """从 ``total`` 帧里均匀取 ``limit`` 个下标（含首末）。采样数恒定，不随序列变长。"""
    if total <= 0:
        return ()
    if total <= limit:
        return tuple(range(total))
    step = (total - 1) / (limit - 1)
    seen: list[int] = []
    for i in range(limit):
        index = int(round(i * step))
        if index not in seen:
            seen.append(index)
    return tuple(seen)


def _luma(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def surface_tilt(liquid_mask: np.ndarray, min_columns: int) -> float | None:
    """液面线倾角（度）。

    对每一列取液体的最上一行做最小二乘拟合。输液袋挂正时液面在画面里是横平的；手机歪了
    20°，液面就跟着歪 20°。有效列不足 ``min_columns``（例如只剩一小条液柱）时返回
    ``None``——判不出就说不判，不猜一个角度出来。

    入参只需要「一列里哪些像素属于液体」的布尔掩膜，不必是清理干净的连通域：上传前预检
    （:mod:`..ingest.quality`）拿不到清理过的掩膜，用的是未做形态学收尾的饱和度带，
    两者量的是同一条液面线。
    """
    columns = np.flatnonzero(liquid_mask.any(axis=0))
    if columns.size < min_columns:
        return None
    rows = liquid_mask.argmax(axis=0)[columns].astype(np.float64)
    xs = columns.astype(np.float64)
    # 居中化后再拟合，避免大坐标下的数值抵消。
    xs = xs - xs.mean()
    rows = rows - rows.mean()
    denominator = float((xs * xs).sum())
    if denominator <= 0.0:
        return None
    slope = float((xs * rows).sum() / denominator)
    return math.degrees(math.atan(abs(slope)))


def _border_gap(mask: np.ndarray | None) -> int | None:
    """掩膜外接框到画面边界的最小距离（像素）；没量到掩膜时为 None。

    出框给的是**距离**而不是布尔：贴边 1px 和贴边 3px 对信息科是两回事——
    前者基本等于拍全了只是差一线，后者才是整只袋子被切掉要重拍。
    """
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    return int(min(xs.min(), ys.min(), width - 1 - xs.max(), height - 1 - ys.max()))


def _bottom_tail_ratio(mask: np.ndarray, rows_considered: int) -> float | None:
    """腔体底部窄带填充率 = 底部若干行的平均宽度 / 最宽行宽度。

    挂着滴斗时，袋体下面接一条细管，最底下那几行只剩管子那么窄（≈0.26）；袋体本身的
    底行接近满宽（≈1.0）。这个量只看形状，不看颜色，因此换袋型、换颜色都不影响判定。
    """
    if not mask.any():
        return None
    widths = mask.sum(axis=1).astype(np.float64)
    max_width = float(widths.max())
    if max_width <= 0.0:
        return None
    last_row = int(np.flatnonzero(widths)[-1])
    start = max(0, last_row - rows_considered + 1)
    band = widths[start : last_row + 1]
    if band.size == 0:
        return None
    return float(band.mean() / max_width)


def measure_frame(
    image: np.ndarray,
    liquid_mask: np.ndarray | None,
    cavity_mask: np.ndarray | None,
    params: FailureModeParams | None = None,
) -> FrameMeasurement:
    """量一帧。``liquid_mask`` / ``cavity_mask`` 为 None 时该类判据记不出（返回 None/False）。"""
    params = params or FailureModeParams()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"期望 HxWx3 的 RGB 帧，实际形状 {image.shape}")

    gray = _luma(image)
    total = float(gray.size)
    channel_min = image.min(axis=2)
    glare_ratio = float(np.count_nonzero(channel_min >= params.glare_channel_min)) / total

    combined: np.ndarray | None = None
    if liquid_mask is not None or cavity_mask is not None:
        combined = np.zeros(image.shape[:2], dtype=bool)
        if liquid_mask is not None and liquid_mask.shape == combined.shape:
            combined |= liquid_mask
        if cavity_mask is not None and cavity_mask.shape == combined.shape:
            combined |= cavity_mask

    return FrameMeasurement(
        luma_mean=float(gray.mean()),
        dark_ratio=float(np.count_nonzero(gray <= params.dark_luma)) / total,
        glare_ratio=glare_ratio,
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        tilt_deg=None if liquid_mask is None else surface_tilt(liquid_mask, params.tilt_min_columns),
        border_gap_px=_border_gap(combined),
        tail_ratio=None if cavity_mask is None else _bottom_tail_ratio(cavity_mask, params.drip_tail_rows),
    )


# ── 分类 ────────────────────────────────────────────────────────────────

#: 每类的固定文案：中文说明 + 护士能照做的补拍动作。
#: ``detail`` 说「现在是什么情况」，``suggestion`` 说「下一步怎么做」。
#: 占位符 ``{n}`` / ``{total}`` 由 :func:`_fill` 填，其余键一律来自 ``evidence``——
#: 模板里出现一个 evidence 里没有的键就会原样留下（宁可留占位符也不编数字）。
_TEXT: dict[str, tuple[str, str, str, str]] = {
    # code -> (严重度, 业务说法, 现在是什么情况, 下一步怎么做)
    "bag_out_of_frame": (
        SEVERITY_WARNING,
        "袋体出框",
        "有 {n}/{total} 帧里袋子没拍全（离画面边只剩 {border_gap_px}px，门限 {threshold}px），量到的液面偏小。",
        "往后退半步，让整只输液袋连同挂点一起进画面，稳住 10–20 秒再重拍这一段。",
    ),
    "motion_blur": (
        SEVERITY_WARNING,
        "运动模糊",
        "有 {n}/{total} 帧画面发糊（清晰度 {sharpness}，门限 {threshold}），袋口和液面边界化开了。",
        "把手机靠在输液架上或用手肘夹住，稳住 10–20 秒再松开重拍。",
    ),
    "glare_overexposed": (
        SEVERITY_WARNING,
        "反光过曝",
        "有 {n}/{total} 帧袋子上打出反光白斑（近白像素占 {ratio}），液面上被打出假凹陷。",
        "换个角度避开顶灯直射，或用手／床单在袋子上方挡一下光再重拍。",
    ),
    "bad_angle": (
        SEVERITY_WARNING,
        "角度不正",
        "有 {n}/{total} 帧手机是歪的（液面线偏 {tilt_deg}°，门限 {threshold}°），量到的液面高度会偏。",
        "手机正对输液架，让液面横平着进画面再重拍一次。",
    ),
    "low_light": (
        SEVERITY_WARNING,
        "暗光",
        "有 {n}/{total} 帧画面太暗（最暗那帧平均灰度 {luma_mean}，门限 {threshold}），袋身和背景分不开。",
        "把床头灯或手电打开，离输液架近一点再重拍。",
    ),
    "empty_drip_chamber": (
        SEVERITY_INFO,
        "滴斗空管",
        "这只袋子下面接着空滴斗，袋体与滴斗连成一块，腔体被撑大、面积比因此偏低。",
        "这是已知偏差，不是你拍的问题，不用重拍——毫升数会略偏小，余量判定不受影响。",
    ),
    # ── 只有「上传前预检」（ingest/quality.py）才发得出的两类 ──────────────
    # 这一张表是**两层质检共用**的补拍文案真源：袋级分类（这里）与序列级预检
    # （ingest/quality.py）都从这里取文案，护士端因此只看到一套措辞。预检独有的一类
    # 故意不进 FAILURE_CODES——袋级分类拿不到「短边多少像素」这类画面级量。
    "low_resolution": (
        SEVERITY_WARNING,
        "分辨率偏低",
        "有 {n}/{total} 帧画面短边只有 {short_side}px（门限 {threshold}px），"
        "液面边界糊在一起，余量高低分不清。",
        "把手机往输液架走近一步、或开 2× 变焦，让袋体占满画面再重拍这一段。",
    ),
}


def render_reshoot_text(
    code: str, hit_frames: int, total: int, evidence: dict[str, Any] | None = None
) -> dict[str, str] | None:
    """补拍文案的**单一真源**：袋级分类与上传前预检都从这里取。

    返回 ``{"severity", "label_zh", "detail_zh", "suggestion_zh"}``；``evidence``
    里的键填进模板，模板里出现一个 ``evidence`` 没有的键就原样留下占位符
    （宁可少一个数，也不编一个数出来）。未知 ``code`` 返回 ``None``。
    """
    entry = _TEXT.get(code)
    if entry is None:
        return None
    severity, label, detail, suggestion = entry
    payload = evidence or {}
    return {
        "severity": severity,
        "label_zh": label,
        "detail_zh": _fill(detail, hit_frames, total, **payload),
        "suggestion_zh": _fill(suggestion, hit_frames, total, **payload),
    }

#: 空滴斗且**能量出**撑大幅度时的补充文案（此时才写百分数，量不出就不编）。
_DRIP_OVERFILL_TEXT = (
    "这只袋子下面接着空滴斗，袋体与滴斗连成一块，腔体被撑大约 {overfill}%，面积比因此偏低。",
)


def _hit_glare(measure: FrameMeasurement, params: FailureModeParams) -> bool:
    return measure.glare_ratio >= params.glare_ratio_threshold


def _hit_low_light(measure: FrameMeasurement, params: FailureModeParams) -> bool:
    """只看全画面平均灰度。暗像素占比在实测里没有区分力（见参数字段注释），不参与判定。"""
    return measure.luma_mean < params.low_light_mean_luma


def _hit_blur(measure: FrameMeasurement, params: FailureModeParams) -> bool:
    return measure.sharpness < params.sharpness_min


def _hit_tilt(measure: FrameMeasurement, params: FailureModeParams) -> bool:
    return measure.tilt_deg is not None and measure.tilt_deg > params.tilt_angle_deg


def _hit_border(measure: FrameMeasurement, params: FailureModeParams) -> bool:
    return measure.border_gap_px is not None and measure.border_gap_px <= params.border_px


def _hit_drip(measure: FrameMeasurement, params: FailureModeParams) -> bool:
    return measure.tail_ratio is not None and measure.tail_ratio < params.drip_tail_ratio


#: 每类的命中判据与证据取值。抽取成表而不是散在 if 里：新增一类只改这一处 + ``_TEXT``。
_JUDGES: dict[str, tuple[Any, Any, Any]] = {
    "bag_out_of_frame": (_hit_border, "border_gap_px", "border_px"),
    "motion_blur": (_hit_blur, "sharpness", "sharpness_min"),
    "glare_overexposed": (_hit_glare, "glare_ratio", "glare_ratio_threshold"),
    "bad_angle": (_hit_tilt, "tilt_deg", "tilt_angle_deg"),
    "low_light": (_hit_low_light, "luma_mean", "low_light_mean_luma"),
    "empty_drip_chamber": (_hit_drip, "tail_ratio", "drip_tail_ratio"),
}


def classify_measurements(
    measurements: Sequence[FrameMeasurement],
    params: FailureModeParams | None = None,
    *,
    drip_overfill_ratio: float | None = None,
) -> CaptureQuality:
    """把若干帧的测量值归成六类失败模式，按优先级排序，**只给一条最该补的建议**。

    ``drip_overfill_ratio`` 是空滴斗把腔体撑大的相对幅度（由调用方用两次腔体估计给出）；
    为 None 时文案里不写具体百分数。
    """
    params = params or FailureModeParams()
    sampled = len(measurements)
    if sampled == 0:
        return CaptureQuality()

    hits: dict[str, list[FrameMeasurement]] = {code: [] for code in FAILURE_CODES}
    for measure in measurements:
        for code, (judge, _field, _threshold) in _JUDGES.items():
            if judge(measure, params):
                hits[code].append(measure)

    # 空滴斗判据是「腔体底部收成细杆」，靠的是外接框底部的形状。手机歪了或袋子被切出
    # 画面时，底行本来就没几行像素，同一个量会把「歪拍的袋角」也读成细杆——这两种帧的
    # 底部窄带不可信，直接从滴斗判定里剔掉，而不是让护士去为一条假线索重拍。
    unreliable = {
        id(measure)
        for code in ("bad_angle", "bag_out_of_frame")
        for measure in hits.get(code, ())
    }
    hits["empty_drip_chamber"] = [
        measure for measure in hits["empty_drip_chamber"] if id(measure) not in unreliable
    ]

    # 暗光帧判不出糊不糊：拉普拉斯方差跟着对比度一起掉，光线不足和手抖在这一个量上
    # 分不开。实测夜间短片（平均灰度 27）清晰度 107、低于门限 150，于是被判成「运动模糊」，
    # 护士收到的建议会是「把手机靠住」——可她真正要做的第一件事是**把灯打开**。
    # 暗光帧的模糊判定整条剔掉，让 low_light 当唯一病因；灯打开之后这一段重新判，
    # 真的还糊仍会被下一次质检点出来。
    dark_frames = {id(measure) for measure in hits.get("low_light", ())}
    hits["motion_blur"] = [
        measure for measure in hits["motion_blur"] if id(measure) not in dark_frames
    ]

    flags: list[FailureFlag] = []
    for code in FAILURE_CODES:
        matched = hits[code]
        if not matched:
            continue
        evidence = _evidence(code, matched, params, drip_overfill_ratio)
        text = render_reshoot_text(code, len(matched), sampled, evidence)
        if text is None:  # pragma: no cover - FAILURE_CODES 与文案表一一对应
            continue
        if text["severity"] == SEVERITY_WARNING and (
            len(matched) / sampled < params.min_hit_ratio
        ):
            # 少数几帧偶发不算「这段拍坏了」——不足以让护士重拍整段。
            continue
        if code == "empty_drip_chamber" and "overfill" in evidence:
            text = {
                **text,
                "detail_zh": _fill(_DRIP_OVERFILL_TEXT, len(matched), sampled, **evidence),
            }
        flags.append(
            FailureFlag(
                code=code,
                severity=text["severity"],
                label_zh=text["label_zh"],
                detail_zh=text["detail_zh"],
                suggestion_zh=text["suggestion_zh"],
                hit_frames=len(matched),
                sampled_frames=sampled,
                evidence=evidence,
            )
        )

    flags.sort(key=lambda flag: PRIORITY[flag.code])
    warnings = [flag for flag in flags if flag.severity == SEVERITY_WARNING]
    return CaptureQuality(
        flags=tuple(flags),
        sampled_frames=sampled,
        degraded=bool(warnings),
        primary=warnings[0] if warnings else None,
    )


def _fill(text: str, hit_frames: int, total: int, **evidence: Any) -> str:
    """把文案模板里的 ``{n}`` / ``{total}`` / 证据占位符填上。缺项直接原样留下占位，
    不伪造数字——宁可文案里少一个数，也不给一个编出来的数。"""
    values: dict[str, Any] = {"n": hit_frames, "total": total}
    values.update(evidence)
    try:
        return text.format(**values)
    except (KeyError, IndexError):
        return text


def _evidence(
    code: str,
    matched: Sequence[FrameMeasurement],
    params: FailureModeParams,
    drip_overfill_ratio: float | None,
) -> dict[str, Any]:
    """证据：命中帧的实测值 + 对应门限。护士与信息科据此判断是「拍坏了」还是「门限太严」。"""
    _judge, field, threshold_key = _JUDGES[code]
    evidence: dict[str, Any] = {}
    if field is not None:
        values = [getattr(measure, field) for measure in matched]
        values = [value for value in values if value is not None]
        if values:
            # 暗光看**最暗**的那帧（离门限最近），其余看**最严重**的那帧。
            worst = min(values) if code == "low_light" else max(values)
            evidence[field] = round(worst, 4)
    if threshold_key is not None:
        evidence["threshold"] = round(float(getattr(params, threshold_key)), 4)
    if code == "low_light":
        evidence["dark_ratio"] = round(max(m.dark_ratio for m in matched), 4)
    if code == "glare_overexposed":
        evidence["ratio"] = f"{max(m.glare_ratio for m in matched) * 100:.2f}%"
    if code == "empty_drip_chamber":
        tails = [m.tail_ratio for m in matched if m.tail_ratio is not None]
        if tails:
            evidence["tail_ratio"] = round(min(tails), 4)
        if drip_overfill_ratio is not None:
            evidence["overfill"] = f"{drip_overfill_ratio * 100:.1f}%"
    return evidence


def classify_capture(
    frames: Sequence[np.ndarray],
    predictor: PerPointMaskPredictor,
    prompt: Prompt,
    params: FailureModeParams | None = None,
    *,
    reference_cavity_area: int | None = None,
) -> CaptureQuality:
    """跑一遍质检：均匀采样 → 逐帧预测液体掩膜 → 量 → 归类。

    ``reference_cavity_area`` 是「去掉滴斗后同一只袋子应有的腔体面积」；给了它才能把
    滴斗撑大腔体的幅度写进文案。腔体只在**首个采样帧**估一次（与主路径同一口径）。
    """
    params = params or FailureModeParams()
    indices = sample_indices(len(frames), max(1, params.sample_frames))
    if not indices:
        return CaptureQuality()

    kernel_params = getattr(predictor, "params", None)
    measurements: list[FrameMeasurement] = []
    cavity_area: int | None = None
    for position, index in enumerate(indices):
        image = frames[index]
        liquid_mask: np.ndarray | None = None
        if position == 0:
            predictions = predictor.predict(image, prompt)
            if predictions:
                liquid_mask = predictions[0].mask
            cavity = estimate_cavity(image, prompt, kernel_params)
            cavity_area = cavity.area_px
            cavity_mask = cavity.mask
        else:
            predictions = predictor.predict(image, prompt)
            if predictions:
                liquid_mask = predictions[0].mask
            cavity_mask = None
        measurements.append(measure_frame(image, liquid_mask, cavity_mask, params))

    overfill: float | None = None
    if cavity_area and reference_cavity_area:
        overfill = (cavity_area - reference_cavity_area) / float(reference_cavity_area)
    return classify_measurements(measurements, params, drip_overfill_ratio=overfill)