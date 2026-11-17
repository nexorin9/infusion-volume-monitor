"""体积不确定度与置信门禁：毫升数可信才敢拿去点名换液。

**为什么要有这一层。** ``estimated_volume_ml`` 是护士唯一会照着做的数字——看余量够不够
就决定要不要换袋。本模块给这个数字配一条**区间**和一个**置信度**；置信度低于门禁时，
``estimated_volume_ml`` 在响应里**整体不出现**（不是 null 占位，见 ``api.schemas`` 的
``exclude_none``），和 :mod:`calibration.estimator` 的 ``uncalibrated`` 门禁同一套语义，
但**原因不同、处置也不同**：``uncalibrated`` 是「信息科还没给这个袋型定标」，补一次登记
就好；置信门禁是「这一段的观测本身撑不起这个数」，重拍一次就好。两者的提示文案必须让
护士分得清，否则一律「不出数」就变成了「这袋子坏了」。

**不确定度从三条证据合成**（各条独立、按平方和相加）：

============================  ==========================================
证据                          含义与取值依据
============================  ==========================================
腔体几何版本                   几何换算的模型误差。腔体模式下取
``geometry_model_sigma``；几何登记了却**没版本号**（``geometry_revision < 1``）时无法判断
                              历史读数是否还成立，直接判门禁——参数改了不升版本，是本产品
                              唯一无法自证的情况。分段带模式下模型误差由拟合残差
                              ``max_relative_error`` 给出（是实测值，不是拍的）。
面积比方差                    本段可信序列的离散度 ``σ_fill``，乘以当前液位处的
                              体积对面积比灵敏度 ``dV/dA``。**低液位天然吃亏**：
                              同样的分割噪声在满袋上只有几个毫升，在 2% 的液面上等于
                              十几毫升——这不是阈值拍脑袋定的，是灵敏度算出来的。
遮挡帧占比                    被挡住的帧不参与统计，末帧读数等于跨着空档外推；占比越高，
                              外推段越长，给的不确定度按 ``occlusion_sigma_gain`` 线性放大。
                              超过 ``high_occlusion_ratio`` 直接判 ``high_occlusion`` 门禁。
============================  ==========================================

``σ_fill`` 的**地板值**（``sigma_fill_floor``）取 0.010，依据是合成回归里五类拍法实测的
面积比偏差：固定机位 0.0084 / 手持 0.0073 / 反光高光 0.0096 / 袋体形变 0.0120 / 夜间暗光
0.0001——取最差的一类作地板，宁可把区间放宽，也不因为「这一段看着很稳」就收窄到没有余量。

**门禁原因码**只有 ``low_confidence`` / ``high_occlusion`` 两种（加上 ``estimator`` 里既有的
``uncalibrated`` 共三类，护士端各有一句不同的话）；更细的成因放在
:attr:`UncertaintyVerdict.reason` 与 ``factors`` 里，信息科据此判断是「拍坏了」还是
「门限太严」。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - 只为类型标注，运行时不 import 造成环
    from .models import BagTypeCalibration

__all__ = [
    "UncertaintyParams",
    "UncertaintyVerdict",
    "ObservationQuality",
    "GATE_OK",
    "GATE_LOW_CONFIDENCE",
    "GATE_HIGH_OCCLUSION",
    "GATES",
    "evaluate_volume_uncertainty",
    "band_sensitivity",
    "reason_hint_zh",
]

#: 门禁结论：三个取值与 ``store.schema.CALIBRATION_STATES`` 里的后两个对齐。
GATE_OK = "ok"
GATE_LOW_CONFIDENCE = "low_confidence"
GATE_HIGH_OCCLUSION = "high_occlusion"
GATES = (GATE_OK, GATE_LOW_CONFIDENCE, GATE_HIGH_OCCLUSION)

#: 成因码：比门禁细一层，给信息科看，不给护士端直接显示。
REASON_GEOMETRY = "geometry_version_missing"
REASON_OCCLUDED = "occluded_too_much"
REASON_SPREAD = "relative_uncertainty_too_high"


@dataclass(frozen=True)
class UncertaintyParams:
    """不确定度门限。取值理由写在字段旁，不留「经验值」这种没头没尾的数。"""

    min_confidence: float = 0.60
    """放行所需的最低置信度。0.60 的含义是「区间半宽不超过余量的四成」——
    再宽的话，护士看着 180±80 不知道该不该换，那就等于没给数。"""

    high_occlusion_ratio: float = 0.35
    """遮挡帧占比超过此值即判 ``high_occlusion``。0.35 是折中：病房里推车、洗手、
    护士背身挡一下，10–20 秒里挡掉三分之一是常事，仍应给个（更宽的）区间；
    超过三分之一说明这次拍的基本是遮挡本身，余量是外推出来的。"""

    sigma_fill_floor: float = 0.010
    """面积比噪声地板（合成回归实测最差一类拍法的偏差 0.0120，取 0.010 留一点余量）。
    本段实测离散度比它小的时候仍按它算——「这一段看着很稳」不等于分割没有系统偏差。"""

    geometry_model_sigma: float = 0.030
    """腔体几何换算自身的相对误差（3%）。袋腔外形是登记时量出来的，登记精度就是
    这个数；它与实测残差（``LearnedResidual``）分开记，避免把「登记得糙」和
    「袋子挂壁」混成同一个数。"""

    occlusion_sigma_gain: float = 0.20
    """遮挡占比 → 相对不确定度的增益。遮挡越多，能用的观测越少、末帧读数越靠外推；
    35% 遮挡约加 7 个百分点相对不确定度，区间大约放宽三成，与「挡得越多越不该照着这个
    数换袋」的直觉一致。"""

    sigma_ceiling: float = 0.35
    """置信度归零的相对不确定度。``confidence = 1 - σ_rel / sigma_ceiling``（下限 0），
    与 ``min_confidence`` 合起来才构成门限：门限等价于 ``σ_rel > 0.14``。"""

    cover: float = 1.96
    """区间的覆盖系数，取正态 95%。不做「±σ」那种只有 68% 的说法——护士看的是
    「这个数大概靠不靠谱」，95% 的口径才敢写进提示。"""

    min_samples_for_variance: int = 3
    """少于这么多可信观测就不算方差（1–2 个点的「方差」是噪声不是证据），退回地板值。"""

    def as_dict(self) -> dict[str, float | int]:
        return {
            "min_confidence": self.min_confidence,
            "high_occlusion_ratio": self.high_occlusion_ratio,
            "sigma_fill_floor": self.sigma_fill_floor,
            "geometry_model_sigma": self.geometry_model_sigma,
            "occlusion_sigma_gain": self.occlusion_sigma_gain,
            "sigma_ceiling": self.sigma_ceiling,
            "cover": self.cover,
            "min_samples_for_variance": self.min_samples_for_variance,
        }


@dataclass(frozen=True)
class ObservationQuality:
    """这一段观测的质量证据（主路径从曲线与平滑结果汇总）。

    只放**事实**（序列、帧数、遮挡帧数），不确定度怎么合成由
    :func:`evaluate_volume_uncertainty` 决定——本类不持有参数、不出结论。
    """

    area_ratio_samples: tuple[float, ...] = ()
    """本段可信面积比观测（看不见的帧已剔除）。"""
    frame_count: int = 0
    occluded_frame_count: int = 0

    @property
    def occlusion_ratio(self) -> float:
        if self.frame_count <= 0:
            return 0.0
        return min(max(self.occluded_frame_count / self.frame_count, 0.0), 1.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_count": len(self.area_ratio_samples),
            "frame_count": self.frame_count,
            "occluded_frame_count": self.occluded_frame_count,
            "occlusion_ratio": round(self.occlusion_ratio, 4),
        }


def _detrended_stddev(samples: Sequence[float]) -> float:
    """可信序列对**线性降幅**的残差标准差——这一段到底抖得有多厉害。

    直接取整段标准差是错的：输液本来就在漏，20 秒里液面实打实降了几个百分点，整段
    标准差量到的是「漏了多少」，不是「量得准不准」，拿它当噪声会让每个正常读数都被
    说成不可信。这里先按最小二乘拟合一条直线（真实降幅近似线性），再取残差的标准差，
    剩下的才是分割与平滑的抖动。两个点恰好能拟合出一条直线，残差恒为 0——这时退回
    地板值，不拿 0 当「完美观测」。
    """
    n = len(samples)
    if n < 3:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(samples) / n
    sxx = sum((index - mean_x) ** 2 for index in range(n))
    if sxx <= 0:
        return 0.0
    sxy = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(samples))
    slope = sxy / sxx
    residuals = [
        value - (mean_y + slope * (index - mean_x)) for index, value in enumerate(samples)
    ]
    var = sum(value**2 for value in residuals) / (n - 2)
    return math.sqrt(max(var, 0.0))


def band_sensitivity(
    bands: Sequence[Sequence[float]], fill_ratio: float
) -> float | None:
    """分段带在 ``fill_ratio`` 处的 ``dV/dA``（毫升 / 单位面积比）。

    取包含该面积比的那一段斜率；落在最外一档时用相邻段的斜率外推。带为空或横坐标
    不递增（不该发生，但旧库里可能有）时返回 ``None``——**不给数比给一个错的斜率好**。
    """
    points = [(float(pair[0]), float(pair[1])) for pair in bands if len(pair) >= 2]
    if len(points) < 2:
        return None
    for index in range(len(points) - 1):
        (x0, y0), (x1, y1) = points[index], points[index + 1]
        if x1 <= x0:
            return None
        if x0 <= fill_ratio <= x1:
            return (y1 - y0) / (x1 - x0)
    # 落在最外一档：借紧邻的那一段。
    if fill_ratio < points[0][0]:
        return (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
    return (points[-1][1] - points[-2][1]) / (points[-1][0] - points[-2][0])


def reason_hint_zh(gate: str, reason: str, *, occlusion_ratio: float) -> str:
    """护士端一句可执行的话。三个门禁三种说法，护士据此知道下一步该做什么。"""
    if gate == GATE_LOW_CONFIDENCE and reason == REASON_GEOMETRY:
        return (
            "这个袋型的外形参数还没登记版本号，改过以后就对不上历史读数了。"
            "先让信息科补登记版本号，毫升数会自己回来。"
        )
    if gate == GATE_HIGH_OCCLUSION:
        return (
            f"这一段有 {occlusion_ratio:.0%} 的时间被挡着，剩下的毫升是外推出来的。"
            "等挡开以后对准输液架再拍 10–20 秒就好。"
        )
    if gate == GATE_LOW_CONFIDENCE:
        return (
            "余量太小或这一段抖动太大，算出来的毫升数落在一段挺宽的范围里，"
            "够不够换液还说不准。等液面再低一些、或把手机端稳了重拍一次。"
        )
    if gate == GATE_OK:
        return "毫升数在可信区间内，可按余量安排换液。"
    return "依据不足，先不出数。"


@dataclass(frozen=True)
class UncertaintyVerdict:
    """一袋的体积置信结论。``gate != GATE_OK`` 时调用方**不得**给出毫升数。"""

    volume_ml: float
    confidence: float
    relative_sigma: float
    sigma_ml: float
    low_ml: float
    high_ml: float
    gate: str = GATE_OK
    reason: str = ""
    hint_zh: str = ""
    factors: dict[str, Any] = field(default_factory=dict)

    @property
    def gated(self) -> bool:
        return self.gate != GATE_OK

    def as_dict(self) -> dict[str, Any]:
        """给落库与响应用的扁平字典（区间与置信度都保留，门禁不通过时也不抹掉）。"""
        return {
            "confidence": round(self.confidence, 4),
            "relative_sigma": round(self.relative_sigma, 4),
            "sigma_ml": round(self.sigma_ml, 2),
            "low_ml": round(self.low_ml, 1),
            "high_ml": round(self.high_ml, 1),
            "gate": self.gate,
            "reason": self.reason,
            "hint_zh": self.hint_zh,
            "factors": self.factors,
        }


def _model_sigma(calibration: "BagTypeCalibration | None", estimation_mode: str) -> tuple[float, str]:
    """模型项（相对）：分段带用实测拟合残差，腔体用登记精度。返回 ``(sigma, 来源)``。"""
    if calibration is None:
        return 0.0, "no_calibration"
    if estimation_mode == "cavity":
        return 0.030, "cavity_geometry"
    return max(float(calibration.max_relative_error), 0.0), "band_fit_residual"


def evaluate_volume_uncertainty(
    *,
    volume_ml: float,
    fill_ratio: float,
    calibration: "BagTypeCalibration | None",
    estimation_mode: str = "band",
    sensitivity_ml_per_ratio: float | None = None,
    observation: ObservationQuality | None = None,
    geometry_revision: int | None = None,
    params: UncertaintyParams | None = None,
) -> UncertaintyVerdict:
    """合成体积不确定度，给出区间与置信度，并判定是否该门禁。

    ``geometry_revision`` 是**腔体几何的版本号**：``< 1`` 表示「登记过但没升版本」，
    此时无法判断历史读数是否还成立，直接判门禁（成因 ``geometry_version_missing``）。
    分段带模式没有几何参数，用拟合残差作模型项，不受此门禁影响。
    """
    p = params or UncertaintyParams()
    occl = observation.occlusion_ratio if observation is not None else 0.0

    if estimation_mode == "cavity":
        model_sigma, model_source = p.geometry_model_sigma, "cavity_geometry"
    else:
        model_sigma, model_source = _model_sigma(calibration, estimation_mode)

    factors: dict[str, Any] = {
        "estimation_mode": estimation_mode,
        "model_sigma": round(model_sigma, 5),
        "model_source": model_source,
        "occlusion_ratio": round(occl, 4),
    }
    factors["sigma_sensitivity_rel"] = 0.0

    def verdict_of(
        sigma_rel: float,
        *,
        gate: str,
        reason: str,
        sigma_fill: float | None = None,
        sensitivity: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> UncertaintyVerdict:
        sigma_rel = max(0.0, min(sigma_rel, p.sigma_ceiling))
        confidence = max(0.0, 1.0 - sigma_rel / p.sigma_ceiling)
        sigma_value = volume_ml * sigma_rel
        half = p.cover * sigma_value
        factors_local = dict(factors)
        if sigma_fill is not None:
            factors_local["sigma_area_ratio"] = round(sigma_fill, 5)
        if sensitivity is not None:
            factors_local["sensitivity_ml_per_ratio"] = round(sensitivity, 2)
        if extra:
            factors_local.update(extra)
        factors_local["relative_sigma"] = round(sigma_rel, 5)
        return UncertaintyVerdict(
            volume_ml=volume_ml,
            confidence=confidence,
            relative_sigma=sigma_rel,
            sigma_ml=sigma_value,
            low_ml=max(volume_ml - half, 0.0),
            high_ml=volume_ml + half,
            gate=gate,
            reason=reason,
            hint_zh=reason_hint_zh(gate, reason, occlusion_ratio=occl),
            factors=factors_local,
        )

    # ① 几何版本缺失：几何换算的模型误差无法界定，先挡下来（门禁原因仍是 low_confidence，
    #    成因码让信息科知道要去补登记，而不是去调门限）。
    if estimation_mode == "cavity" and (geometry_revision or 0) < 1:
        factors["geometry_revision"] = int(geometry_revision or 0)
        return verdict_of(
            p.sigma_ceiling,
            gate=GATE_LOW_CONFIDENCE,
            reason=REASON_GEOMETRY,
            extra={"geometry_revision": int(geometry_revision or 0)},
        )

    # ② 面积比方差 → 体积不确定度：离散度与分割噪声地板平方和，再乘当前液位的灵敏度。
    #    低液位就是在这里被挡下的——同样的 σ_fill，灵敏度一样，分子上的毫升数小一大截。
    samples = tuple(
        float(value) for value in (observation.area_ratio_samples if observation else ()) if value is not None
    )
    observed_std = (
        _detrended_stddev(samples) if len(samples) >= p.min_samples_for_variance else 0.0
    )
    sigma_fill = math.sqrt(observed_std**2 + p.sigma_fill_floor**2)
    factors["sample_count"] = len(samples)
    factors["residual_std_after_drain_fit"] = round(observed_std, 5)

    sensitivity = sensitivity_ml_per_ratio
    if sensitivity is None and calibration is not None and estimation_mode == "band":
        sensitivity = band_sensitivity(calibration.bands, fill_ratio)
    #: 灵敏度折成的**相对**不确定度（无量纲，与另两项同量纲才能平方和）。
    #: 除以 volume_ml 是这一项的要害：同样的分割噪声，满袋上只有百分之几，
    #: 落到见底的小数上就是几成——低液位被门禁挡下是这个除法算出来的，不是拍阈值。
    sigma_sensitivity_rel = 0.0
    if sensitivity is not None and volume_ml > 0:
        sigma_sensitivity_rel = abs(float(sensitivity)) * sigma_fill / volume_ml
        factors["sigma_sensitivity_rel"] = round(sigma_sensitivity_rel, 5)

    # ③ 遮挡：占比超门限直接判门禁；未超则按增益加宽区间。
    sigma_occ = occl * p.occlusion_sigma_gain
    relative_sigma = math.sqrt(
        model_sigma**2 + sigma_sensitivity_rel**2 + sigma_occ**2
    )

    if occl >= p.high_occlusion_ratio:
        return verdict_of(
            max(relative_sigma, p.sigma_ceiling),
            gate=GATE_HIGH_OCCLUSION,
            reason=REASON_OCCLUDED,
            sigma_fill=sigma_fill,
            sensitivity=sensitivity,
            extra={"high_occlusion_ratio": p.high_occlusion_ratio},
        )

    confidence = max(0.0, 1.0 - min(relative_sigma, p.sigma_ceiling) / p.sigma_ceiling)
    if confidence < p.min_confidence:
        return verdict_of(
            relative_sigma,
            gate=GATE_LOW_CONFIDENCE,
            reason=REASON_SPREAD,
            sigma_fill=sigma_fill,
            sensitivity=sensitivity,
            extra={
                "min_confidence": p.min_confidence,
                "equivalent_relative_sigma_limit": round(
                    p.sigma_ceiling * (1.0 - p.min_confidence), 5
                ),
            },
        )

    return verdict_of(
        relative_sigma,
        gate=GATE_OK,
        reason="",
        sigma_fill=sigma_fill,
        sensitivity=sensitivity,
    )