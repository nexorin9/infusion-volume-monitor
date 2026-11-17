"""毫升数与预计耗尽时点（ETA）的估计，以及**门禁语义**。

本模块只做三件事：

1. **滴速来源优先级**：输注医嘱文本 → MQTT 泵速遥测 → ``unavailable``。两个来源
   **对不上**时（相对偏差超门限）不靠「优先」硬压一头，而是判 ``rate_conflict``：
   不给时点、把两个值交出去请人工确认。三者都缺时**不给猜出来的时点**，并把该袋的
   换液点名标为抑制（``pickup_suppressed``）——「还剩多少、还能撑多久」是护士唯一会
   照着做的事，给一个编出来的时点比不给更危险。
2. **体积 → ETA**：``eta_minutes = ceil(volume_ml / rate_ml_per_h * 60)``，向上取整，
   宁可早一分钟提醒，不可晚一分钟漏掉。
3. **门禁**：无标定记录 → ``calibration_state=uncalibrated``，``estimated_volume_ml``
   与 ``eta_minutes`` 都**不出数**（字段整体不出现，不是 null 占位）。响应序列化
   统一走 ``exclude_none``（见 ``api/schemas.dump_model``）。

滴速文本怎么解析、两个来源怎么比，见 :mod:`calibration.rate_parser` 与
:mod:`calibration.eta`；本模块只把「来源 → 毫升 → 时点」串起来。

**置信门禁**（``low_confidence`` / ``high_occlusion``）由 :mod:`calibration.uncertainty`
接上：三条证据（腔体几何版本、面积比方差、遮挡帧占比）合成体积不确定度，低于门禁就
**连毫升数一起不出**——数不可信时「还剩多少、还能撑多久」两个数都不能给。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cavity import LearnedResidual
from .eta import (
    DEFAULT_CONFLICT_TOLERANCE,
    ETA_SOURCES,
    EtaEstimate,
    RateConflict,
    RateResolution,
    estimate_eta,
    resolve_rate_sources,
)
from .models import BagTypeCalibration, VolumeReading
from .rate_parser import ParsedRate, parse_rate, parse_rate_text
from .uncertainty import (
    GATE_OK,
    ObservationQuality,
    UncertaintyParams,
    UncertaintyVerdict,
    band_sensitivity,
    evaluate_volume_uncertainty,
)
from .versions import DEFAULT_REVIEW_THRESHOLD, judge_deviation

__all__ = [
    "RateResolution",
    "RateConflict",
    "VolumeEstimate",
    "TelemetrySample",
    "TelemetryCache",
    "parse_order_rate",
    "parse_rate",
    "parse_rate_text",
    "resolve_rate",
    "estimate_bag",
    "recompute_for_new_version",
    "ETA_SOURCES",
    "SUPPRESSION_REASONS",
    "UNOBSERVED",
    "ObservationQuality",
    "UncertaintyVerdict",
]

#: 体积门禁的三种状态（``uncalibrated`` 是「没定标」，后两个是「观测撑不起这个数」）。
STATE_LOW_CONFIDENCE = "low_confidence"
STATE_HIGH_OCCLUSION = "high_occlusion"

#: 抑制换液点名的原因码。
SUPPRESSION_REASONS = (
    "uncalibrated",
    "eta_unavailable",
    "rate_conflict",
    "no_observation",
    "overfilled",
    "low_confidence",
    "high_occlusion",
)

#: 标定状态（与 ``store.schema.CALIBRATION_STATES`` 保持一致）。
STATE_UNCALIBRATED = "uncalibrated"
STATE_CALIBRATED = "calibrated"

#: 本轮尚未起锚 / 整段看不见时的填充比占位（不是 0，0 会被读成「袋是空的」）。
UNOBSERVED = None


@dataclass(frozen=True)
class TelemetrySample:
    """一条泵速遥测。``ts`` 用于重连后去重（同一份遥测不重复消费）。"""

    rate_ml_per_h: float
    device_sn: str = ""
    ts: str = ""
    #: 这条读数是从哪条通道进来的（``mqtt`` / ``webhook``）。只用于观测：
    #: 院区切了 webhook 兜底之后，运维要能从袋位响应里看出「现在是 webhook 在喂」。
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "rate_ml_per_h": round(self.rate_ml_per_h, 3),
            "device_sn": self.device_sn,
            "ts": self.ts,
            "source": self.source,
        }


class TelemetryCache:
    """按 ``(rack_id, bag_seq)`` 记最近一条遥测的进程内缓存。

    视觉主路径**不依赖**它：遥测缺失只降低交叉校验能力，ETA 仍可由医嘱速率给出。
    task 11/39 的 MQTT 订阅泵与重连去重往这里 ``publish``，本轮只留缓存与读接口。
    """

    def __init__(self, *, max_entries: int = 4096) -> None:
        self._latest: dict[tuple[str, str], TelemetrySample] = {}
        self._max_entries = max_entries

    def publish(self, rack_id: str, bag_seq: str, sample: TelemetrySample) -> bool:
        """写入遥测；同一 ``(rack, bag)`` 的旧时间戳直接丢弃（重连去重的兜底）。"""
        key = (rack_id, bag_seq)
        previous = self._latest.get(key)
        if previous is not None and sample.ts and previous.ts and sample.ts <= previous.ts:
            return False
        if len(self._latest) >= self._max_entries:
            oldest = min(self._latest, key=lambda item: self._latest[item].ts or "")
            self._latest.pop(oldest, None)
        self._latest[key] = sample
        return True

    def get(self, rack_id: str, bag_seq: str) -> TelemetrySample | None:
        return self._latest.get((rack_id, bag_seq))

    def rate_ml_per_h(self, rack_id: str, bag_seq: str) -> float | None:
        sample = self.get(rack_id, bag_seq)
        return sample.rate_ml_per_h if sample else None

    def count(self) -> int:
        return len(self._latest)

    def clear(self) -> None:
        self._latest.clear()


@dataclass(frozen=True)
class VolumeEstimate:
    """一袋的体积与 ETA 估计结果；门禁不通过时各数字字段为 ``None``（响应里不出现）。"""

    calibration_state: str
    calibration_version: int | None = None
    bag_type_code: str | None = None
    fill_ratio: float | None = None
    estimated_volume_ml: float | None = None
    eta_minutes: int | None = None
    eta_source: str = "unavailable"
    rate_ml_per_h: float | None = None
    rate_reason: str = ""
    pickup_suppressed: bool = True
    suppression_reason: str | None = "no_observation"
    near_empty: bool = False
    near_full: bool = False
    overfilled: bool = False
    band_index: int | None = None
    #: 毫升数是怎么来的：``band`` = 分段带（经验标定），``cavity`` = 腔体几何换算。
    estimation_mode: str = "band"
    #: 腔体反解出的真实高度比与残留量（仅 ``cavity`` 模式有值）。
    cavity_height_fraction: float | None = None
    resident_ml: float | None = None
    #: 自校准结论摘要（仅该袋型登记过腔体参数时有值）。
    self_calibration: dict[str, Any] | None = None
    #: 医嘱速率与泵速遥测对不上的证据（仅冲突时有值）。
    rate_conflict: dict[str, Any] | None = None
    #: 体积不确定度与置信结论（区间、置信度、门禁成因）。**门禁不通过时照样给**，
    #: 护士端要靠它显示「为什么这次没有毫升数」以及下一步该做什么。
    uncertainty: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """给仓储层落库用的扁平字典（``None`` 表示「这一格没有数」，不写 null 占位语义）。"""
        return {
            "fill_ratio": self.fill_ratio,
            "estimated_volume_ml": self.estimated_volume_ml,
            "eta_minutes": self.eta_minutes,
            "eta_source": self.eta_source,
            "rate_ml_per_h": self.rate_ml_per_h,
            "rate_source": self.eta_source,
            "rate_reason": self.rate_reason,
            "calibration_state": self.calibration_state,
            "calibration_version": self.calibration_version,
            "pickup_suppressed": self.pickup_suppressed,
            "suppression_reason": self.suppression_reason,
            "near_empty": self.near_empty,
            "overfilled": self.overfilled,
            "volume_confidence": (
                self.uncertainty["confidence"] if self.uncertainty else None
            ),
            "volume_low_ml": self.uncertainty["low_ml"] if self.uncertainty else None,
            "volume_high_ml": self.uncertainty["high_ml"] if self.uncertainty else None,
            "uncertainty_gate": self.uncertainty["gate"] if self.uncertainty else None,
            "uncertainty_json": self.uncertainty,
        }


def parse_order_rate(text: str | None) -> float | None:
    """从输注医嘱文本里抽 ``ml/h``；抽不出返回 ``None``（不猜、不取默认值）。

    完整写法（滴/分、滴系数、全角半角、按体重、推注）见
    :mod:`calibration.rate_parser`。
    """
    return parse_rate(text)


def resolve_rate(
    order_text: str | None,
    telemetry_rate_ml_per_h: float | None,
    *,
    telemetry_state: str = "",
    conflict_tolerance: float = DEFAULT_CONFLICT_TOLERANCE,
) -> RateResolution:
    """滴速来源优先级：**医嘱文本 → MQTT 遥测 → unavailable**，并对不上的两个来源判冲突。

    医嘱优先于遥测：医嘱是开单时写定的治疗决定，泵速遥测只是执行侧的实际值；
    两者**明显对不上**时（相对偏差超 ``conflict_tolerance``）不取中间值糊弄，
    而是判 ``rate_conflict``：不给时点，把两个值交出去请人工确认。
    """
    return resolve_rate_sources(
        order_text,
        telemetry_rate_ml_per_h,
        conflict_tolerance=conflict_tolerance,
        telemetry_state=telemetry_state,
    )


def _gated_estimate(
    *,
    gate: str,
    verdict: UncertaintyVerdict,
    base: dict[str, Any],
) -> VolumeEstimate:
    """门禁不通过时的统一返回形态。

    ``estimated_volume_ml`` 与 ``eta_minutes`` **都是 None**（响应里字段整体不出现）——
    毫升不可信时，由它推出来的「还能撑多久」同样不可信，两个数一起撤。区间与置信度
    仍然回传：护士端要显示「大概多少、为什么这次没有准数」，信息科要靠它判断是拍坏了
    还是门限太严。
    """
    payload = dict(base)
    payload.update(
        calibration_state=gate,
        estimated_volume_ml=None,
        eta_minutes=None,
        pickup_suppressed=True,
        suppression_reason=gate,
        uncertainty=verdict.as_dict(),
    )
    return VolumeEstimate(**payload)


def estimate_bag(
    fill_ratio: float | None,
    calibration: BagTypeCalibration | None,
    rate: RateResolution,
    *,
    bag_type_code: str | None = None,
    learned_residual: LearnedResidual | None = None,
    observation: ObservationQuality | None = None,
    uncertainty_params: UncertaintyParams | None = None,
) -> VolumeEstimate:
    """一袋的毫升数与 ETA；门禁不通过时**不出数**。

    三条门禁（返回的各数字字段为 ``None``，响应序列化时字段整体不出现）：

    * 整段看不见（``fill_ratio is None``）→ 什么都不给，点名抑制，原因 ``no_observation``；
    * 无生效标定记录 → ``calibration_state=uncalibrated``，原因 ``uncalibrated``；
    * 面积比超装（几何核把袋外液体也算进来了）→ 不给数，原因 ``overfilled``，
      宁可让护士重拍一张，也不给一个「1000 毫升」的假读数。

    **毫升数走哪条路**由袋型有没有登记腔体参数决定：登记了走
    :mod:`calibration.cavity` 的腔体几何（``estimation_mode="cavity"``，可带自校准
    学出来的残留补偿），没登记走分段带（``"band"``）。两条路的门禁语义完全一致——
    有腔体参数不等于可以放宽「不出数」的规矩。
    """
    code = bag_type_code or (calibration.bag_type_code if calibration else None)
    if fill_ratio is None:
        return VolumeEstimate(
            calibration_state=STATE_UNCALIBRATED if calibration is None else STATE_CALIBRATED,
            calibration_version=calibration.version if calibration else None,
            bag_type_code=code,
            eta_source=rate.source,
            rate_ml_per_h=rate.rate_ml_per_h,
            rate_reason=rate.reason,
            pickup_suppressed=True,
            suppression_reason="no_observation",
            rate_conflict=rate.conflict.as_dict() if rate.conflict else None,
        )

    if calibration is None:
        return VolumeEstimate(
            calibration_state=STATE_UNCALIBRATED,
            bag_type_code=code,
            fill_ratio=float(fill_ratio),
            eta_source=rate.source,
            rate_ml_per_h=rate.rate_ml_per_h,
            rate_reason=rate.reason,
            pickup_suppressed=True,
            suppression_reason="uncalibrated",
        )

    # 腔体模型优先（袋型登记了外形参数时）；残差补偿只在证据充分时生效。
    if calibration.cavity is not None:
        return _estimate_by_cavity(
            float(fill_ratio),
            calibration,
            rate,
            code,
            learned_residual,
            observation=observation,
            uncertainty_params=uncertainty_params,
        )

    reading: VolumeReading = calibration.estimated_volume_ml(fill_ratio)
    if reading.overfilled:
        return VolumeEstimate(
            calibration_state=STATE_CALIBRATED,
            calibration_version=calibration.version,
            bag_type_code=code,
            fill_ratio=float(fill_ratio),
            eta_source=rate.source,
            rate_ml_per_h=rate.rate_ml_per_h,
            rate_reason=rate.reason,
            pickup_suppressed=True,
            suppression_reason="overfilled",
            overfilled=True,
        )

    base: dict[str, Any] = {
        "calibration_state": STATE_CALIBRATED,
        "calibration_version": calibration.version,
        "bag_type_code": code,
        "fill_ratio": float(fill_ratio),
        "near_empty": reading.near_empty,
        "near_full": reading.near_full,
        "overfilled": False,
        "band_index": reading.band_index,
        "eta_source": rate.source,
        "rate_ml_per_h": rate.rate_ml_per_h,
        "rate_reason": rate.reason,
        "rate_conflict": rate.conflict.as_dict() if rate.conflict else None,
    }
    verdict = evaluate_volume_uncertainty(
        volume_ml=reading.volume_ml,
        fill_ratio=float(fill_ratio),
        calibration=calibration,
        estimation_mode="band",
        sensitivity_ml_per_ratio=band_sensitivity(calibration.bands, float(fill_ratio)),
        observation=observation,
        params=uncertainty_params,
    )
    if verdict.gate != GATE_OK:
        return _gated_estimate(gate=verdict.gate, verdict=verdict, base=base)

    eta = estimate_eta(reading.volume_ml, rate)
    return VolumeEstimate(
        **base,
        estimated_volume_ml=reading.volume_ml,
        eta_minutes=eta.eta_minutes,
        pickup_suppressed=eta.pickup_suppressed,
        suppression_reason=eta.suppression_reason,
        uncertainty=verdict.as_dict(),
    )


#: 袋腔模型实例按 (袋型, 全部外形参数) 缓存：几何只依赖参数，重建没有意义，
#: 而每次上传都要反解，建 257×257 的查找表不划算。
_CAVITY_MODEL_CACHE: dict[str, Any] = {}


def _cavity_model_for(calibration: BagTypeCalibration) -> Any:
    from .cavity import CavityModel

    spec = calibration.cavity
    key = spec.to_json()
    model = _CAVITY_MODEL_CACHE.get(key)
    if model is None:
        model = CavityModel(spec)
        _CAVITY_MODEL_CACHE[key] = model
    return model


def _cavity_sensitivity(
    model: Any, fill_ratio: float, residual_ml: float
) -> float | None:
    """腔体模式在当前液位处的 ``dV/dA``（毫升 / 单位面积比），中心差分。

    分段带的斜率是现成的，腔体是反解出来的，没有解析导数——用 ±0.002 面积比的中心
    差分估一个就够：不确定度要的是「同样的分割噪声会折成多少毫升」，量级对了即可，
    而中心差分在这个平滑曲面上比单边差分稳得多。
    """
    step = 0.002
    low = model.volume(max(fill_ratio - step, 0.0), learned_residual_ml=residual_ml)
    high = model.volume(min(fill_ratio + step, 1.0), learned_residual_ml=residual_ml)
    if low.readable_ml is None or high.readable_ml is None:
        return None
    return (high.readable_ml - low.readable_ml) / (2.0 * step)


def _estimate_by_cavity(
    fill_ratio: float,
    calibration: BagTypeCalibration,
    rate: RateResolution,
    code: str | None,
    learned_residual: LearnedResidual | None,
    *,
    observation: ObservationQuality | None = None,
    uncertainty_params: UncertaintyParams | None = None,
) -> VolumeEstimate:
    """腔体几何换算。门禁与分段带那条路同口径，只是毫升数的来源不同。"""
    model = _cavity_model_for(calibration)
    residual = learned_residual.effective_residual_ml if learned_residual else 0.0
    reading = model.volume(fill_ratio, learned_residual_ml=residual)
    summary = learned_residual.as_dict() if learned_residual else None

    if reading.overfilled or reading.readable_ml is None:
        return VolumeEstimate(
            calibration_state=STATE_CALIBRATED,
            calibration_version=calibration.version,
            bag_type_code=code,
            fill_ratio=fill_ratio,
            eta_source=rate.source,
            rate_ml_per_h=rate.rate_ml_per_h,
            rate_reason=rate.reason,
            pickup_suppressed=True,
            suppression_reason="overfilled",
            overfilled=True,
            estimation_mode="cavity",
            cavity_height_fraction=reading.height_fraction,
            resident_ml=reading.resident_ml,
            self_calibration=summary,
        )

    base: dict[str, Any] = {
        "calibration_state": STATE_CALIBRATED,
        "calibration_version": calibration.version,
        "bag_type_code": code,
        "fill_ratio": fill_ratio,
        "near_empty": reading.near_empty,
        "near_full": reading.near_full,
        "overfilled": False,
        "estimation_mode": "cavity",
        "cavity_height_fraction": reading.height_fraction,
        "resident_ml": reading.resident_ml,
        "self_calibration": summary,
        "eta_source": rate.source,
        "rate_ml_per_h": rate.rate_ml_per_h,
        "rate_reason": rate.reason,
        "rate_conflict": rate.conflict.as_dict() if rate.conflict else None,
    }
    verdict = evaluate_volume_uncertainty(
        volume_ml=reading.readable_ml,
        fill_ratio=fill_ratio,
        calibration=calibration,
        estimation_mode="cavity",
        sensitivity_ml_per_ratio=_cavity_sensitivity(model, fill_ratio, residual),
        observation=observation,
        geometry_revision=calibration.cavity.geometry_revision if calibration.cavity else 0,
        params=uncertainty_params,
    )
    if verdict.gate != GATE_OK:
        return _gated_estimate(gate=verdict.gate, verdict=verdict, base=base)

    eta = estimate_eta(reading.readable_ml, rate)
    return VolumeEstimate(
        **base,
        estimated_volume_ml=reading.readable_ml,
        eta_minutes=eta.eta_minutes,
        pickup_suppressed=eta.pickup_suppressed,
        suppression_reason=eta.suppression_reason,
        uncertainty=verdict.as_dict(),
    )


def _confidence_of(estimate: VolumeEstimate) -> float | None:
    return estimate.uncertainty["confidence"] if estimate.uncertainty else None


def _interval_of(estimate: VolumeEstimate, key: str) -> Any:
    return estimate.uncertainty.get(key) if estimate.uncertainty else None


def _observation_from_session(session: dict[str, Any]) -> ObservationQuality | None:
    """从会话行复原观测证据（标定改版重算专用）。

    主路径的证据来自曲线与平滑结果，那两样在重算时已经不在手边；会话行里存着可信
    序列、帧数与遮挡帧数，够复原同一份口径。序列读不出来时返回 ``None``——让不确定度
    退回地板值，而不是把一条读不出来的行当成「零遮挡」当成最好的情况。
    """
    import json as _json

    raw = session.get("area_ratio_series")
    samples: list[float] = []
    if raw:
        try:
            values = _json.loads(raw) if isinstance(raw, str) else list(raw)
        except (ValueError, TypeError):
            values = []
        samples = [float(v) for v in values if v is not None]
    frame_count = int(session.get("frame_count") or 0)
    occluded = int(session.get("frame_count_occluded") or 0)
    if not samples and not frame_count:
        return None
    return ObservationQuality(
        area_ratio_samples=tuple(samples),
        frame_count=frame_count,
        occluded_frame_count=occluded,
    )


def recompute_for_new_version(
    store: Any,
    calibration: BagTypeCalibration,
    previous_version: int | None,
    *,
    review_threshold: float | None = None,
) -> dict[str, Any]:
    """标定改版后，把**历史会话**的毫升数按新版本重算，并留版本轨迹。

    * 只重算 ``bag_type_code`` 命中、``fill_ratio`` 非空、状态不是 ``replaced`` 的会话；
      已换袋的会话（``replaced``）保留当时的数字，不追溯修改；
    * 重算结果与差异（改前 / 改后 / 差多少）逐条写进 append-only 的
      ``calibration_recalc_log``，供护理部与信息科回答「这袋当时为什么报 180 毫升」；
    * **首次登记**（此前没有生效版本）与**改版**记不同的原因码：前者是「补录标定后把
      之前 uncalibrated 的会话补上毫升数」（``calibration_registered``），后者是
      「换版本后按新曲线重算」（``calibration_version_changed``）。两者都不写
      ``previous_volume_ml``，因为改前根本**没有数**——不是 0；
    * 会话行与重算日志在**同一个事务**里落盘，不允许日志缺、业务却已改。

    **偏差闸门（task 27）**：新旧读数相对差超过 ``review_threshold``
    （默认 0.15）时**不覆盖**旧读数——那多半意味着新版本本身有问题，而不是
    「旧版本终于被修正」。这类会话照常留一条 ``review_required=1`` 的重算轨迹
    （``new_volume_ml`` 是「本该给出的数」），旧读数原样保留并被标成待复核，
    差异逐条落 ``calibration_version_alert``，等护理部结案后再由复核入口落库。
    闸门只拦**改版**（``previous_volume_ml`` 已有数）；首次登记没有旧读数，直接生效。
    """
    reason = (
        "calibration_registered"
        if previous_version is None
        else "calibration_version_changed"
    )
    sessions = store.sessions.list_by_bag_type(calibration.bag_type_code)
    changed: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    skipped = 0
    # 重算与主路径必须用同一套残留补偿，否则改版后历史数字会与新上传的对不上。
    learned = None
    deviations = getattr(store, "deviations", None)
    if deviations is not None and calibration.cavity is not None:
        learned = deviations.summary(
            calibration.bag_type_code, usable_capacity_ml=calibration.usable_capacity_ml
        )
    for session in sessions:
        if session.get("fill_ratio") is None or session.get("state") == "replaced":
            skipped += 1
            continue
        rate = RateResolution(
            rate_ml_per_h=session.get("rate_ml_per_h"),
            source=session.get("rate_source") or "unavailable",
        )
        # 与主路径同口径：从会话行复原这一段的观测证据（可信序列、帧数、遮挡帧数），
        # 否则改版重算会拿一条「没有证据」的记录去重算，凭空比上传那一刻宽一截区间。
        observation = _observation_from_session(session)
        estimate = estimate_bag(
            session["fill_ratio"],
            calibration,
            rate,
            bag_type_code=calibration.bag_type_code,
            learned_residual=learned,
            observation=observation,
        )
        previous_volume = session.get("estimated_volume_ml")
        verdict = judge_deviation(
            previous_volume,
            estimate.estimated_volume_ml,
            threshold=(
                DEFAULT_REVIEW_THRESHOLD if review_threshold is None else review_threshold
            ),
        )
        store.calibrations.log_recalc(
            calibration=calibration,
            previous_version=previous_version,
            session=session,
            previous_volume_ml=previous_volume,
            new_volume_ml=estimate.estimated_volume_ml,
            reason=reason,
            relative_deviation=verdict.relative_deviation,
            review_required=verdict.review_required,
        )
        if verdict.review_required:
            # 不覆盖：旧读数原样留着，只把「有一版更大的数待复核」这件事记下来。
            store.sessions.update_calibration(
                session["session_id"],
                {
                    "calibration_review_required": 1,
                    "calibration_review_deviation": verdict.relative_deviation,
                    "calibration_review_version": calibration.version,
                    "calibration_review_proposed_ml": estimate.estimated_volume_ml,
                },
            )
            store.calibrations.raise_alert(
                calibration=calibration,
                previous_version=previous_version,
                session=session,
                previous_volume_ml=previous_volume,
                proposed_volume_ml=estimate.estimated_volume_ml,
                relative_deviation=verdict.relative_deviation or 0.0,
                threshold=verdict.threshold,
                severity=verdict.severity,
            )
            held.append(
                {
                    "session_id": session["session_id"],
                    "bag_seq": session.get("bag_seq"),
                    "previous_volume_ml": previous_volume,
                    "proposed_volume_ml": estimate.estimated_volume_ml,
                    "relative_deviation": verdict.relative_deviation,
                    "severity": verdict.severity,
                }
            )
            continue
        store.sessions.update_calibration(
            session["session_id"],
            {
                "fill_ratio": estimate.fill_ratio,
                "estimated_volume_ml": estimate.estimated_volume_ml,
                "eta_minutes": estimate.eta_minutes,
                "eta_source": estimate.eta_source,
                "calibration_state": estimate.calibration_state,
                "calibration_version": estimate.calibration_version,
                "calibration_review_required": 0,
                "calibration_review_deviation": None,
                "calibration_review_version": None,
                "calibration_review_proposed_ml": None,
                "volume_confidence": _confidence_of(estimate),
                "volume_low_ml": _interval_of(estimate, "low_ml"),
                "volume_high_ml": _interval_of(estimate, "high_ml"),
                "uncertainty_gate": _interval_of(estimate, "gate"),
                "uncertainty_json": estimate.uncertainty,
            },
        )
        changed.append(
            {
                "session_id": session["session_id"],
                "bag_seq": session.get("bag_seq"),
                "previous_volume_ml": previous_volume,
                "new_volume_ml": estimate.estimated_volume_ml,
                "relative_deviation": verdict.relative_deviation,
                "applied": True,
            }
        )
    return {
        "bag_type_code": calibration.bag_type_code,
        "previous_version": previous_version,
        "new_version": calibration.version,
        "recalculated_count": len(changed),
        "skipped_count": skipped,
        "review_required_count": len(held),
        "review_threshold": (
            DEFAULT_REVIEW_THRESHOLD if review_threshold is None else review_threshold
        ),
        "changes": changed,
        "held_for_review": held,
    }