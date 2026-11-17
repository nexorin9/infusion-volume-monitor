"""ETA 引擎：医嘱速率 × 泵速遥测 →「还能撑多久」，以及两者对不上时的显式降级。

滴速有**两个**来源：

* **输注医嘱**（``order_text``）——开单时写定的治疗决定；
* **泵速遥测**（``mqtt_telemetry``）——输液泵执行侧的实际值。

两条来源，一个结论：

* **只有一条**（或两条一致）→ 按医嘱给时点。医嘱是治疗决定，遥测只是执行状态；
* **两条都缺** → ``eta_source=unavailable``，**不给时点**。给一个猜出来的时点
  比不给更危险——护士照着「还早着呢」不换液，袋子就见底了；
* **两条对不上**（相对偏差超门限）→ 标 ``rate_conflict``，**同样不给时点**，
  并把两个值连同偏差一起交出去，请人工确认。

为什么不冲突的时候也不给时点：医嘱写 40 ml/h、泵上实际跑 20 ml/h 时，
按医嘱算出的「还剩 2 小时」在泵的实际速度下是「还剩 4 小时」；按泵速算，
又与医嘱对不上，可能泵速本身没调对。**取中间值更是两头不靠**——那个数字
既不是医嘱说的，也不是泵在跑的，只会让两个都像对的。所以这里不取中间值：
把冲突显式抛出来，由护士确认到底是医嘱没改还是泵没调。

相对偏差按 ``|医嘱 − 遥测| / max(医嘱, 遥测)`` 计算（对称，不因分母取谁而偏心）。
默认门限 20%：院内泵速与医嘱的常见偏差在 10% 以内，超过两成已经不是舍入误差。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .rate_parser import ParsedRate, parse_rate_text

__all__ = [
    "DEFAULT_CONFLICT_TOLERANCE",
    "ETA_SOURCES",
    "EtaEstimate",
    "RateConflict",
    "RateResolution",
    "eta_minutes_for",
    "estimate_eta",
    "relative_deviation",
    "resolve_rate_sources",
]

#: 医嘱速率与泵速遥测的相对偏差门限（0.20 = 20%）。超过即判冲突。
DEFAULT_CONFLICT_TOLERANCE = 0.20

#: 滴速来源，取值写进 ``bag_session.eta_source`` 与响应，便于回溯「这个时点哪来的」。
ETA_SOURCES = ("order_text", "mqtt_telemetry", "conflict", "unavailable")


@dataclass(frozen=True)
class RateConflict:
    """医嘱速率与泵速遥测对不上的证据。"""

    order_rate_ml_per_h: float
    telemetry_rate_ml_per_h: float
    relative_deviation: float
    tolerance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_rate_ml_per_h": round(self.order_rate_ml_per_h, 4),
            "telemetry_rate_ml_per_h": round(self.telemetry_rate_ml_per_h, 4),
            "relative_deviation": round(self.relative_deviation, 4),
            "tolerance": round(self.tolerance, 4),
            "message": (
                f"医嘱 {self.order_rate_ml_per_h:g} ml/h 与泵速遥测 "
                f"{self.telemetry_rate_ml_per_h:g} ml/h 相对偏差 "
                f"{self.relative_deviation * 100:.1f}%，超过 {self.tolerance * 100:.0f}% 门限——"
                "两个来源对不上，不取中间值，请人工确认"
            ),
        }


@dataclass(frozen=True)
class RateResolution:
    """滴速来源解析结果：用了哪一条、为什么，以及有没有冲突。

    ``rate_ml_per_h`` 为 ``None`` 表示**没有可用的滴速**（两个来源都缺，或两者冲突）；
    ``conflict`` 非空时说明是哪两个值对不上。
    """

    rate_ml_per_h: float | None = None
    source: str = "unavailable"
    reason: str = ""
    conflict: RateConflict | None = None
    parsed_rate: ParsedRate | None = None
    #: 由本模块给出的抑制原因码（rate_conflict / eta_unavailable）；无则为空。
    suppression_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.rate_ml_per_h is not None and self.rate_ml_per_h > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rate_ml_per_h": (
                round(self.rate_ml_per_h, 4) if self.rate_ml_per_h is not None else None
            ),
            "source": self.source,
            "reason": self.reason,
            "conflict": self.conflict.as_dict() if self.conflict else None,
            "suppression_reason": self.suppression_reason,
        }


@dataclass(frozen=True)
class EtaEstimate:
    """一袋的「还能撑多久」结论。

    门禁不通过时 ``eta_minutes`` 为 ``None``（响应里字段整体不出现），
    ``suppression_reason`` 说清是哪一道门禁。
    """

    eta_minutes: int | None = None
    eta_source: str = "unavailable"
    rate_ml_per_h: float | None = None
    rate_reason: str = ""
    conflict: RateConflict | None = None
    suppression_reason: str | None = None
    pickup_suppressed: bool = False
    parsed_rate: ParsedRate | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "eta_minutes": self.eta_minutes,
            "eta_source": self.eta_source,
            "rate_ml_per_h": (
                round(self.rate_ml_per_h, 4) if self.rate_ml_per_h is not None else None
            ),
            "rate_reason": self.rate_reason,
            "conflict": self.conflict.as_dict() if self.conflict else None,
            "suppression_reason": self.suppression_reason,
            "pickup_suppressed": self.pickup_suppressed,
        }


def relative_deviation(left: float, right: float) -> float:
    """两个速率的相对偏差（对称口径：分母取较大者）。"""
    base = max(abs(left), abs(right))
    if base <= 0:
        return 0.0
    return abs(left - right) / base


def eta_minutes_for(volume_ml: float, rate_ml_per_h: float) -> int:
    """毫升 → 分钟，向上取整（宁可早提醒，不可晚漏掉）。"""
    if rate_ml_per_h <= 0:
        raise ValueError("滴速必须为正数")
    if volume_ml <= 0:
        return 0
    return int(math.ceil(volume_ml / rate_ml_per_h * 60.0))


def resolve_rate_sources(
    order_text: str | None,
    telemetry_rate_ml_per_h: float | None,
    *,
    conflict_tolerance: float = DEFAULT_CONFLICT_TOLERANCE,
    telemetry_state: str = "",
) -> RateResolution:
    """按「医嘱 → 泵速遥测」定来源，并对不上的两个来源显式判冲突。

    医嘱优先于遥测：医嘱是开单时写定的治疗决定，泵速遥测只是执行侧的实际值；
    但两者**明显对不上**时不靠「优先」这条规则硬压一头——那是冲突，要人工确认。
    """
    parsed = parse_rate_text(order_text)
    order_rate = parsed.rate_ml_per_h if parsed.available else None
    telemetry_rate: float | None = None
    if telemetry_rate_ml_per_h is not None:
        try:
            candidate = float(telemetry_rate_ml_per_h)
        except (TypeError, ValueError):
            candidate = 0.0
        if candidate > 0:
            telemetry_rate = candidate

    if order_rate is not None and telemetry_rate is not None:
        deviation = relative_deviation(order_rate, telemetry_rate)
        if deviation > conflict_tolerance:
            conflict = RateConflict(
                order_rate_ml_per_h=order_rate,
                telemetry_rate_ml_per_h=telemetry_rate,
                relative_deviation=deviation,
                tolerance=conflict_tolerance,
            )
            return RateResolution(
                rate_ml_per_h=None,
                source="conflict",
                reason=conflict.as_dict()["message"],
                conflict=conflict,
                parsed_rate=parsed,
                suppression_reason="rate_conflict",
            )
        return RateResolution(
            rate_ml_per_h=order_rate,
            source="order_text",
            reason=(
                f"医嘱 {order_rate:g} ml/h 与泵速遥测 {telemetry_rate:g} ml/h 一致"
                f"（相对偏差 {deviation * 100:.1f}%，门限 {conflict_tolerance * 100:.0f}%），按医嘱给时点"
            ),
            parsed_rate=parsed,
        )

    if order_rate is not None:
        return RateResolution(
            rate_ml_per_h=order_rate,
            source="order_text",
            reason=parsed.reason,
            parsed_rate=parsed,
        )

    if telemetry_rate is not None:
        suffix = f"（{telemetry_state}）" if telemetry_state else ""
        return RateResolution(
            rate_ml_per_h=telemetry_rate,
            source="mqtt_telemetry",
            reason=f"医嘱未给可解析速率，退到泵速遥测{suffix}",
            parsed_rate=parsed,
        )

    return RateResolution(
        rate_ml_per_h=None,
        source="unavailable",
        reason=(
            f"{parsed.reason}；泵速遥测也没有——不给猜出来的时点"
            if parsed.rate_ml_per_h is None
            else "输注医嘱未给可解析速率，泵速遥测也没有——不给猜出来的时点"
        ),
        parsed_rate=parsed,
        suppression_reason="eta_unavailable",
    )


def estimate_eta(volume_ml: float | None, resolution: RateResolution) -> EtaEstimate:
    """由余量毫升与已定来源推出「还能撑多久」。

    三条门禁（``eta_minutes`` 为 ``None``，响应里字段整体不出现）：

    * 没有余量观测（``volume_ml is None``）→ ``no_observation``；
    * 两个滴速来源对不上 → ``rate_conflict``（不取中间值）；
    * 两个来源都给不出滴速 → ``eta_unavailable``。
    """
    if volume_ml is None:
        return EtaEstimate(
            eta_minutes=None,
            eta_source=resolution.source,
            rate_ml_per_h=resolution.rate_ml_per_h,
            rate_reason=resolution.reason,
            conflict=resolution.conflict,
            suppression_reason="no_observation",
            pickup_suppressed=True,
            parsed_rate=resolution.parsed_rate,
        )

    if resolution.conflict is not None:
        return EtaEstimate(
            eta_minutes=None,
            eta_source="conflict",
            rate_ml_per_h=None,
            rate_reason=resolution.reason,
            conflict=resolution.conflict,
            suppression_reason="rate_conflict",
            pickup_suppressed=True,
            parsed_rate=resolution.parsed_rate,
        )

    if not resolution.available:
        return EtaEstimate(
            eta_minutes=None,
            eta_source="unavailable",
            rate_ml_per_h=None,
            rate_reason=resolution.reason,
            conflict=None,
            suppression_reason="eta_unavailable",
            pickup_suppressed=True,
            parsed_rate=resolution.parsed_rate,
        )

    return EtaEstimate(
        eta_minutes=eta_minutes_for(float(volume_ml), float(resolution.rate_ml_per_h or 0.0)),
        eta_source=resolution.source,
        rate_ml_per_h=resolution.rate_ml_per_h,
        rate_reason=resolution.reason,
        conflict=None,
        suppression_reason=None,
        pickup_suppressed=False,
        parsed_rate=resolution.parsed_rate,
    )
