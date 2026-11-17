"""ETA 引擎：医嘱速率、泵速遥测与两者冲突时的降级（task 26）。

「还能撑多久」是护士最想拿到的结果，但它只在**滴速可信**时才能给。三类用例：

* **单一来源**：只有医嘱，或医嘱认不出而遥测在——照常给时点；
* **两个来源都缺**：``eta_source=unavailable``，**不给时点**（给一个编出来的时点
  比不给更危险），毫升数照给、点名抑制；
* **两个来源对不上**：判 ``rate_conflict``——**不取中间值**，不给时点，把两个值
  连同偏差交出去请人工确认，并且**抑制该袋的换液点名**。
"""

from __future__ import annotations

import json

import pytest

from infusion_monitor.calibration.estimator import estimate_bag, resolve_rate
from infusion_monitor.calibration.eta import (
    DEFAULT_CONFLICT_TOLERANCE,
    EtaEstimate,
    RateResolution,
    estimate_eta,
    eta_minutes_for,
    relative_deviation,
    resolve_rate_sources,
)
from infusion_monitor.calibration.models import BagTypeCalibration, CalibrationSample
from infusion_monitor.identity.state_machine import ChangeParams, evaluate_change
from tests.support import EXAMPLE_FIELDS, build_frame_zip

TRACKS_URL = "/api/v1/infusion/tracks"
CALIBRATIONS_URL = "/api/v1/infusion/calibrations"

#: 一份演示用的标定记录（占位样例，非真实称重数据）。
TYPE_A_SAMPLES = (
    CalibrationSample(0.05, 50.0, "演示用占位样例"),
    CalibrationSample(0.25, 250.0, "演示用占位样例"),
    CalibrationSample(0.50, 500.0, "演示用占位样例"),
    CalibrationSample(0.75, 750.0, "演示用占位样例"),
    CalibrationSample(0.95, 950.0, "演示用占位样例"),
)


def type_a() -> BagTypeCalibration:
    return BagTypeCalibration.from_samples(
        "TYPE-A", 1000.0, TYPE_A_SAMPLES, calibrated_by="EXAMPLE-NURSE-01"
    )


# ── 1. 单一来源 ─────────────────────────────────────────────────────────────


def test_order_text_only_gives_eta():
    resolution = resolve_rate_sources("40ml/h", None)

    assert resolution.source == "order_text"
    assert resolution.rate_ml_per_h == pytest.approx(40.0)
    assert resolution.conflict is None
    assert resolution.available is True


def test_telemetry_only_is_used_when_order_text_has_no_rate():
    resolution = resolve_rate_sources("常规输液", 60.0, telemetry_state="connected")

    assert resolution.source == "mqtt_telemetry"
    assert resolution.rate_ml_per_h == pytest.approx(60.0)
    assert "遥测" in resolution.reason


def test_both_missing_means_no_time_point():
    resolution = resolve_rate_sources("常规输液", None)

    assert resolution.source == "unavailable"
    assert resolution.rate_ml_per_h is None
    assert resolution.available is False
    assert resolution.suppression_reason == "eta_unavailable"


def test_zero_or_negative_telemetry_is_not_a_rate():
    """泵停了（0 ml/h）不等于「速率是 0」——那会让 ETA 变成无穷大。"""
    for value in (0.0, -5.0):
        resolution = resolve_rate_sources("常规输液", value)
        assert resolution.source == "unavailable"


# ── 2. 两个来源一致 ─────────────────────────────────────────────────────────


def test_two_sources_agreeing_prefers_the_order():
    resolution = resolve_rate_sources("40ml/h", 42.0)

    assert resolution.source == "order_text"
    assert resolution.rate_ml_per_h == pytest.approx(40.0)
    assert resolution.conflict is None


def test_deviation_exactly_at_tolerance_is_not_a_conflict():
    """门限取「超过」而不是「达到」：40 与 32 正好 20%，按医嘱走。"""
    resolution = resolve_rate_sources("40ml/h", 32.0)

    assert resolution.source == "order_text"
    assert resolution.conflict is None


def test_relative_deviation_is_symmetric():
    assert relative_deviation(40.0, 20.0) == pytest.approx(0.5)
    assert relative_deviation(20.0, 40.0) == pytest.approx(0.5)
    assert relative_deviation(40.0, 40.0) == pytest.approx(0.0)


# ── 3. 两个来源冲突 ─────────────────────────────────────────────────────────


def test_conflicting_sources_are_flagged_and_no_middle_value_is_taken():
    """医嘱 40 ml/h、泵上跑 20 ml/h：不取中间值 30，给出冲突证据。"""
    resolution = resolve_rate_sources("40ml/h", 20.0)

    assert resolution.source == "conflict"
    assert resolution.rate_ml_per_h is None
    assert resolution.suppression_reason == "rate_conflict"
    conflict = resolution.conflict
    assert conflict is not None
    assert conflict.order_rate_ml_per_h == pytest.approx(40.0)
    assert conflict.telemetry_rate_ml_per_h == pytest.approx(20.0)
    assert conflict.relative_deviation == pytest.approx(0.5)
    assert conflict.tolerance == pytest.approx(DEFAULT_CONFLICT_TOLERANCE)
    assert "不取中间值" in conflict.as_dict()["message"]


def test_conflict_tolerance_is_configurable():
    """门限可调：放宽到 60% 后 40 与 20（偏差 50%）不再算冲突。"""
    resolution = resolve_rate_sources("40ml/h", 20.0, conflict_tolerance=0.6)

    assert resolution.source == "order_text"
    assert resolution.conflict is None


def test_conflict_keeps_the_volume_but_withholds_the_time_point():
    """冲突时毫升数照给（与滴速无关），时点不给——护士至少知道还剩多少。"""
    record = type_a()
    estimate = estimate_bag(0.60, record, resolve_rate("40ml/h", 20.0))

    assert estimate.calibration_state == "calibrated"
    assert estimate.estimated_volume_ml == pytest.approx(600.0)
    assert estimate.eta_minutes is None
    assert estimate.eta_source == "conflict"
    assert estimate.rate_ml_per_h is None
    assert estimate.pickup_suppressed is True
    assert estimate.suppression_reason == "rate_conflict"
    assert estimate.rate_conflict is not None
    assert estimate.rate_conflict["order_rate_ml_per_h"] == pytest.approx(40.0)
    assert estimate.rate_conflict["telemetry_rate_ml_per_h"] == pytest.approx(20.0)


def test_conflict_suppresses_the_change_pickup():
    """冲突时该袋**不点名**：余量已经到紧急线，但依据不足，不催换。

    对照组把抑制去掉，同样的余量会点名——说明抑制确实是冲突带来的。
    """
    session = {
        "bag_seq": "B01",
        "session_id": "s-1",
        "state": "monitoring",
        "area_ratio_series": [0.12, 0.11, 0.10, 0.09, 0.08],
        "estimated_volume_ml": 20.0,  # ≤ margin_ml（30 ml），条件 B 会触发
        "eta_minutes": None,  # 冲突 → 不给时点
        "pickup_suppressed": True,
        "suppression_reason": "rate_conflict",
    }

    suppressed = evaluate_change(session)
    assert suppressed.changed is False
    assert suppressed.target_state == "monitoring"
    assert "rate_conflict" in suppressed.reason_detail

    control = {**session, "pickup_suppressed": False, "suppression_reason": None}
    unconflicted = evaluate_change(control)
    assert unconflicted.changed is True
    assert unconflicted.trigger_reason == "volume_below_margin"


# ── 4. ETA 计算本身 ─────────────────────────────────────────────────────────


def test_eta_rounds_up_so_nurse_is_told_early():
    """60 ml ÷ 7 ml/h = 514.28 分钟 → 向上取整 515，宁可早提醒。"""
    assert eta_minutes_for(60.0, 7.0) == 515
    estimate = estimate_bag(0.10, type_a(), resolve_rate_sources("7ml/h", None))
    # 标定带在 0.10 处读出 100 ml：100 ÷ 7 × 60 = 857.14 → 858
    assert estimate.estimated_volume_ml == pytest.approx(100.0)
    assert estimate.eta_minutes == 858
    assert estimate.eta_minutes == eta_minutes_for(100.0, 7.0)


def test_eta_for_empty_bag_is_zero_minutes():
    assert eta_minutes_for(0.0, 40.0) == 0


def test_eta_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        eta_minutes_for(100.0, 0.0)


def test_estimate_eta_without_observation_is_suppressed():
    estimate = estimate_eta(None, resolve_rate_sources("40ml/h", None))

    assert estimate.eta_minutes is None
    assert estimate.suppression_reason == "no_observation"
    assert estimate.pickup_suppressed is True


def test_estimate_eta_without_rate_is_suppressed():
    estimate = estimate_eta(500.0, resolve_rate_sources("常规输液", None))

    assert estimate.eta_minutes is None
    assert estimate.eta_source == "unavailable"
    assert estimate.suppression_reason == "eta_unavailable"
    assert estimate.pickup_suppressed is True


def test_estimate_eta_on_conflict_withholds_time_point():
    estimate = estimate_eta(500.0, resolve_rate_sources("40ml/h", 20.0))

    assert estimate.eta_minutes is None
    assert estimate.eta_source == "conflict"
    assert estimate.suppression_reason == "rate_conflict"
    assert estimate.conflict is not None


def test_eta_estimate_serialises_without_leaking_plaintext():
    payload = EtaEstimate(
        eta_minutes=10,
        eta_source="order_text",
        rate_ml_per_h=40.0,
    ).as_dict()

    assert payload["eta_minutes"] == 10
    assert payload["conflict"] is None
    assert payload["suppression_reason"] is None


def test_resolve_rate_is_the_thin_wrapper_over_sources():
    assert resolve_rate("40ml/h", None) == resolve_rate_sources("40ml/h", None)


def test_rate_resolution_keeps_parsed_rate_for_traceability():
    resolution = resolve_rate_sources("20滴/分", None)

    assert isinstance(resolution, RateResolution)
    assert resolution.parsed_rate is not None
    assert resolution.parsed_rate.source_form == "drops_per_min"
    assert resolution.parsed_rate.drop_factor_gtt_per_ml == pytest.approx(20.0)


# ── 5. 端到端：上传时医嘱与泵速遥测对不上 ──────────────────────────────────


def _register_type_a(client) -> None:
    response = client.post(
        CALIBRATIONS_URL,
        json={
            "bag_type_code": "TYPE-A",
            "capacity_ml": 1000.0,
            "samples": [
                {"measured_fill_ratio": item.measured_fill_ratio,
                 "known_volume_ml": item.known_volume_ml,
                 "source": item.source}
                for item in TYPE_A_SAMPLES
            ],
            "calibrated_by": "EXAMPLE-NURSE-01",
            "note": "演示用占位样例",
        },
    )
    assert response.status_code == 201, response.text


def _upload(client, *, rate_text: str) -> dict:
    fields = dict(EXAMPLE_FIELDS)
    fields["slots"] = json.dumps(
        [{"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-A", "rate_text": rate_text}]
    )
    files = {"frames": ("rack.zip", build_frame_zip(frame_count=12), "application/zip")}
    response = client.post(TRACKS_URL, data=fields, files=files)
    assert response.status_code == 201, response.text
    return response.json()


def test_upload_with_conflicting_telemetry_suppresses_pickup(store_client):
    """真实主路径：医嘱 40 ml/h、泵速遥测 20 ml/h → 冲突、不给时点、点名抑制。"""
    client, _observer = store_client
    _register_type_a(client)

    # 泵速遥测先到（架号来自主题，袋序 B01），与医嘱的 40 ml/h 差一倍
    client.app.state.telemetry.publish(
        "R-EXAMPLE-07", "B01", _telemetry_sample(20.0)
    )

    payload = _upload(client, rate_text="40ml/h")
    bag = payload["bags"][0]

    assert bag["eta_source"] == "conflict"
    assert "eta_minutes" not in bag
    assert "rate_ml_per_h" not in bag
    assert bag["pickup_suppressed"] is True
    assert bag["suppression_reason"] == "rate_conflict"
    assert bag["rate_conflict"]["order_rate_ml_per_h"] == pytest.approx(40.0)
    assert bag["rate_conflict"]["telemetry_rate_ml_per_h"] == pytest.approx(20.0)
    # 毫升数与滴速无关，照给
    assert bag["estimated_volume_ml"] is not None


def test_upload_with_matching_telemetry_gives_eta(store_client):
    """两个来源一致 → 照常给时点，点名不被压住。"""
    client, _observer = store_client
    _register_type_a(client)
    client.app.state.telemetry.publish(
        "R-EXAMPLE-07", "B01", _telemetry_sample(40.0)
    )

    payload = _upload(client, rate_text="40ml/h")
    bag = payload["bags"][0]

    assert bag["eta_source"] == "order_text"
    assert bag["eta_minutes"] is not None
    assert bag["rate_ml_per_h"] == pytest.approx(40.0)
    assert "rate_conflict" not in bag


def test_conflict_is_persisted_for_audit(store_client):
    """冲突证据落库：事后能回答「这袋当时为什么不给时点」。"""
    client, observer = store_client
    _register_type_a(client)
    client.app.state.telemetry.publish(
        "R-EXAMPLE-07", "B01", _telemetry_sample(20.0)
    )

    payload = _upload(client, rate_text="40ml/h")
    session_id = payload["bags"][0]["session_id"]

    row = observer.execute(
        "SELECT eta_minutes, eta_source, rate_conflict_json, pickup_suppressed, "
        "suppression_reason FROM bag_session WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    assert row["eta_minutes"] is None
    assert row["eta_source"] == "conflict"
    assert row["pickup_suppressed"] == 1
    assert row["suppression_reason"] == "rate_conflict"
    conflict = json.loads(row["rate_conflict_json"])
    assert conflict["order_rate_ml_per_h"] == pytest.approx(40.0)
    assert conflict["telemetry_rate_ml_per_h"] == pytest.approx(20.0)


def _telemetry_sample(rate_ml_per_h: float):
    from infusion_monitor.calibration.estimator import TelemetrySample

    return TelemetrySample(
        rate_ml_per_h=rate_ml_per_h,
        device_sn="PUMP-EXAMPLE-01",
        ts="2026-10-22T09:41:00+08:00",
    )
