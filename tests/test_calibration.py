"""M3 袋型标定、毫升换算与 uncalibrated 门禁。

覆盖 task.json 步骤 1–6：

1. 标定记录（袋型 / 腔体参数 / 样例 / 标定人 / 标定时间 / 版本号）
2. fill_ratio → estimated_volume_ml（分段带 + 弯月面补偿）、滴速 → eta_minutes
3. 滴速来源优先级：医嘱文本 → MQTT 遥测；都缺 → eta_source=unavailable 且抑制点名
4. 门禁：无标定记录时响应里 **estimated_volume_ml 字段整体不出现**
5. 改版后历史会话按新版本重算，版本轨迹可查
6. 端到端：真实上传 → 标定后出数；未标定时不出数
"""

from __future__ import annotations

import json
import math

import pytest

from infusion_monitor.calibration.estimator import (
    TelemetryCache,
    TelemetrySample,
    estimate_bag,
    parse_order_rate,
    recompute_for_new_version,
    resolve_rate,
)
from infusion_monitor.calibration.models import (
    BagTypeCalibration,
    CalibrationSample,
    fit_bands,
)
from infusion_monitor.store import CalibrationRepository, SessionRepository

from .support import synth

# ── 标定样例：分段带（输液袋上的刻度带本身就是分段线性的） ──────────────────
TYPE_A_SAMPLES = (
    CalibrationSample(0.10, 60.0, source="称重法"),
    CalibrationSample(0.35, 220.0, source="称重法"),
    CalibrationSample(0.60, 390.0, source="换液确认反推"),
    CalibrationSample(0.85, 540.0, source="称重法"),
)


def type_a(**kwargs) -> BagTypeCalibration:
    params = {
        "capacity_ml": 600.0,
        "samples": TYPE_A_SAMPLES,
        "calibrated_by": "OP-NURSE-01",
        "calibrated_at": "2026-10-22T08:00:00+08:00",
    }
    params.update(kwargs)
    return BagTypeCalibration.from_samples("TYPE-A", **params)


# ── 1 标定记录本身 ──────────────────────────────────────────────────────────


def test_calibration_record_carries_geometry_and_provenance():
    record = type_a(dead_volume_ml=20.0, meniscus_offset_ratio=0.03)

    assert record.bag_type_code == "TYPE-A"
    assert record.capacity_ml == 600.0
    assert record.usable_capacity_ml == 580.0  # 标称 − 管路残留
    assert record.dead_volume_ml == 20.0
    assert record.meniscus_offset_ratio == 0.03
    assert record.calibrated_by == "OP-NURSE-01"
    assert record.calibrated_at == "2026-10-22T08:00:00+08:00"
    assert record.version == 1
    assert len(record.samples) == 4
    # 袋腔几何 = 分段带的节点，两端补齐到 [0,1]
    assert record.bands[0] == (0.0, 0.0)
    assert record.bands[-1][0] == 1.0
    assert record.bands[-1][1] == pytest.approx(580.0)
    assert [ratio for ratio, _ in record.bands] == sorted(r for r, _ in record.bands)


def test_fit_is_exact_at_sample_points_and_monotone():
    record = type_a()
    for sample in TYPE_A_SAMPLES:
        reading = record.estimated_volume_ml(sample.measured_fill_ratio)
        assert reading.volume_ml == pytest.approx(sample.known_volume_ml, abs=1e-6)

    volumes = [record.estimated_volume_ml(r / 20).volume_ml for r in range(21)]
    assert volumes == sorted(volumes)  # 单调不增
    assert record.max_relative_error < 0.01


def test_no_samples_falls_back_to_straight_line_and_says_so():
    record = type_a(samples=())
    assert record.bands == ((0.0, 0.0), (1.0, 600.0))
    assert record.max_relative_error == 1.0  # 明确告诉标定人「这条没拟合」
    assert record.estimated_volume_ml(0.5).volume_ml == pytest.approx(300.0)


def test_meniscus_offset_shifts_the_reading_down():
    """液面挂壁 → 量到的面积比偏高 → 补偿后应当读得更小。"""
    plain = type_a()
    compensated = type_a(meniscus_offset_ratio=0.05)

    assert compensated.estimated_volume_ml(0.50).volume_ml < plain.estimated_volume_ml(0.50).volume_ml
    assert compensated.estimated_volume_ml(0.50).effective_fill_ratio == pytest.approx(0.45)
    # 补偿量取 0.05 时，应等于按 0.45 查表
    assert compensated.estimated_volume_ml(0.50).volume_ml == pytest.approx(
        plain.estimated_volume_ml(0.45).volume_ml
    )


def test_band_index_and_edge_flags():
    record = type_a()
    assert record.band_index_of(0.0) == 0
    assert record.band_index_of(1.0) == len(record.bands) - 1
    assert record.estimated_volume_ml(0.03).near_empty is True
    assert record.estimated_volume_ml(0.99).near_full is True
    assert record.estimated_volume_ml(0.50).near_empty is False


def test_duplicate_ratio_samples_take_median_and_record_residual():
    samples = [
        CalibrationSample(0.5, 300.0, source="称重法"),
        CalibrationSample(0.5, 320.0, source="称重法"),
        CalibrationSample(0.5, 310.0, source="称重法"),
    ]
    fit = fit_bands(samples, 600.0)
    knots = dict(fit.bands)
    assert knots[0.5] == pytest.approx(310.0)  # 中位数抗单次读数毛刺
    assert fit.max_relative_error > 0  # 端点外推的偏差如实记下来，不静默拉直


# ── 2 体积 → ETA ────────────────────────────────────────────────────────────


def test_volume_and_eta_from_calibrated_bag():
    record = type_a()
    rate = resolve_rate("5ml/h", None)

    estimate = estimate_bag(0.60, record, rate, bag_type_code="TYPE-A")

    assert estimate.calibration_state == "calibrated"
    assert estimate.calibration_version == 1
    assert estimate.estimated_volume_ml == pytest.approx(390.0)
    assert estimate.eta_minutes == 4680  # 390 ml ÷ 5 ml/h = 78 小时 = 4680 分钟
    assert estimate.eta_source == "order_text"
    assert estimate.pickup_suppressed is False
    assert estimate.suppression_reason is None
    # 读数不出 14 位小数：0.1 ml 就够，不拿假精度糊弄护士
    assert estimate.estimated_volume_ml == round(estimate.estimated_volume_ml, 1)


def test_eta_rounds_up_so_nurse_is_told_early():
    record = type_a()
    estimate = estimate_bag(0.10, record, resolve_rate("7ml/h", None))
    # 60 ml / 7 ml/h = 514.28 min → 向上取整 515
    assert estimate.eta_minutes == 515


# ── 3 滴速来源优先级 ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("5ml/h", 5.0),
        ("5 ml/h", 5.0),
        ("250毫升/小时", 250.0),
        ("40 毫升每小时", 40.0),
        ("  8 ML / H  ", 8.0),
        ("缓慢静滴", None),
        ("", None),
        (None, None),
        ("0ml/h", None),
    ],
)
def test_order_rate_parsing(text, expected):
    assert parse_order_rate(text) == expected


def test_rate_priority_order_text_wins_over_telemetry():
    """医嘱与泵速遥测**一致**时按医嘱给时点（偏差在门限内）。"""
    resolved = resolve_rate("5ml/h", 5.2)
    assert resolved.source == "order_text"
    assert resolved.rate_ml_per_h == 5.0
    assert resolved.conflict is None


def test_rate_conflict_is_flagged_instead_of_silently_picking_one():
    """两个来源明显对不上时判冲突：不给时点、不取中间值（task 26）。"""
    resolved = resolve_rate("5ml/h", 60.0)
    assert resolved.source == "conflict"
    assert resolved.rate_ml_per_h is None
    assert resolved.conflict is not None
    assert resolved.conflict.order_rate_ml_per_h == 5.0
    assert resolved.conflict.telemetry_rate_ml_per_h == 60.0


def test_rate_falls_back_to_telemetry_when_order_unparsable():
    resolved = resolve_rate("常规输液", 60.0, telemetry_state="connected")
    assert resolved.source == "mqtt_telemetry"
    assert resolved.rate_ml_per_h == 60.0
    assert "遥测" in resolved.reason


def test_rate_unavailable_when_both_missing():
    resolved = resolve_rate("常规输液", None)
    assert resolved.source == "unavailable"
    assert resolved.rate_ml_per_h is None
    assert resolved.available is False


def test_missing_rate_suppresses_pickup_but_keeps_volume():
    """给不出滴速时：毫升数照给，时点不给，点名抑制——不拿猜出来的时点催换袋。"""
    record = type_a()
    estimate = estimate_bag(0.60, record, resolve_rate("常规输液", None))

    assert estimate.estimated_volume_ml == pytest.approx(390.0)
    assert estimate.eta_minutes is None
    assert estimate.eta_source == "unavailable"
    assert estimate.pickup_suppressed is True
    assert estimate.suppression_reason == "eta_unavailable"


def test_telemetry_cache_keeps_latest_and_rejects_stale():
    cache = TelemetryCache()
    assert cache.publish("R-1", "B01", TelemetrySample(60.0, ts="2026-10-22T08:00:00+08:00"))
    # 重连后重放的旧时间戳直接丢弃，不让过期遥测覆盖新值
    assert not cache.publish("R-1", "B01", TelemetrySample(999.0, ts="2026-10-22T07:00:00+08:00"))
    assert cache.rate_ml_per_h("R-1", "B01") == 60.0
    assert cache.publish("R-1", "B01", TelemetrySample(80.0, ts="2026-10-22T09:00:00+08:00"))
    assert cache.rate_ml_per_h("R-1", "B01") == 80.0
    assert cache.rate_ml_per_h("R-1", "B99") is None


# ── 4 uncalibrated 门禁 ─────────────────────────────────────────────────────


def test_uncalibrated_gate_withholds_volume_and_eta():
    estimate = estimate_bag(0.60, None, resolve_rate("5ml/h", None), bag_type_code="TYPE-X")

    assert estimate.calibration_state == "uncalibrated"
    assert estimate.estimated_volume_ml is None
    assert estimate.eta_minutes is None
    assert estimate.pickup_suppressed is True
    assert estimate.suppression_reason == "uncalibrated"
    assert estimate.fill_ratio == 0.60  # 观测值照给，只是不换算


def test_no_observation_withholds_everything():
    estimate = estimate_bag(None, type_a(), resolve_rate("5ml/h", None))

    assert estimate.fill_ratio is None
    assert estimate.estimated_volume_ml is None
    assert estimate.suppression_reason == "no_observation"


def test_overfilled_ratio_is_refused_rather_than_rounded():
    estimate = estimate_bag(1.35, type_a(), resolve_rate("5ml/h", None))

    assert estimate.overfilled is True
    assert estimate.estimated_volume_ml is None
    assert estimate.suppression_reason == "overfilled"


# ── 5 改版重算与版本轨迹 ─────────────────────────────────────────────────────


def _seed_session(repo, **overrides):
    payload = {
        "rack_id": "R-07",
        "bed_no": "12",
        "infusion_order_no": "ORDER-0001",
        "shift_key": "2026-10-22-AM",
        "bag_seq": "B01",
        "slot_index": 0,
        "bag_type_code": "TYPE-A",
        "frames_dir": "/tmp/frames",
        "frame_count": 24,
        "captured_at": "2026-10-22T08:00:00+08:00",
        "state": "monitoring",
    }
    payload.update(overrides)
    return repo.upsert_session(payload)


def test_new_version_supersedes_old_and_keeps_history(conn):
    repo = CalibrationRepository(conn)
    first, superseded, _status = repo.save(type_a())
    assert first.version == 1 and superseded is None

    second, superseded_version, _status = repo.save(
        type_a(capacity_ml=650.0, note="同型换批次后重新标定")
    )
    assert second.version == 2
    assert superseded_version == 1
    assert repo.get_active("TYPE-A").version == 2  # 同时只有一个生效版本
    assert repo.get(first.calibration_id).active is False  # 旧版保留可查，不被覆盖
    assert [record.version for record in repo.list_by_code("TYPE-A")] == [1, 2]


def test_version_change_recomputes_history_and_leaves_trail(conn):
    sessions = SessionRepository(conn)
    calibrations = CalibrationRepository(conn)

    _seed_session(sessions)
    _seed_session(sessions, bag_seq="B02", slot_index=1)
    # 已换袋的会话保留当时的读数，不追溯修改
    _seed_session(sessions, bag_seq="B03", slot_index=2, state="replaced")
    sessions.update_calibration(
        sessions.find_by_rack_shift_bag("R-07", "2026-10-22-AM", "B01")["session_id"],
        {"fill_ratio": 0.60, "estimated_volume_ml": 390.0, "eta_minutes": 78,
         "eta_source": "order_text", "rate_ml_per_h": 5.0, "rate_source": "order_text"},
    )
    sessions.update_calibration(
        sessions.find_by_rack_shift_bag("R-07", "2026-10-22-AM", "B02")["session_id"],
        {"fill_ratio": 0.10, "estimated_volume_ml": 60.0, "eta_minutes": 12,
         "eta_source": "order_text", "rate_ml_per_h": 5.0, "rate_source": "order_text"},
    )
    replaced_id = sessions.find_by_rack_shift_bag(
        "R-07", "2026-10-22-AM", "B03"
    )["session_id"]
    sessions.update_calibration(replaced_id, {"fill_ratio": 0.35, "estimated_volume_ml": 220.0})

    class _Store:
        pass

    store = _Store()
    store.sessions = sessions
    store.calibrations = calibrations

    calibrations.save(type_a())  # v1：会话上的读数就是按它算的
    # 改版 = 重新标定：样例整体偏高（换批次后袋腔略大），读数必须跟着动
    revised = tuple(
        CalibrationSample(s.measured_fill_ratio, s.known_volume_ml * 1.08, s.source)
        for s in TYPE_A_SAMPLES
    )
    new_record, superseded_version, _status = calibrations.save(
        type_a(capacity_ml=650.0, samples=revised)
    )
    summary = recompute_for_new_version(store, new_record, superseded_version)

    assert summary["recalculated_count"] == 2
    assert summary["skipped_count"] == 1  # replaced 的会话不追溯

    b01 = sessions.find_by_rack_shift_bag("R-07", "2026-10-22-AM", "B01")
    assert b01["calibration_version"] == 2
    assert b01["estimated_volume_ml"] == pytest.approx(
        new_record.estimated_volume_ml(0.60).volume_ml
    )
    assert b01["estimated_volume_ml"] == pytest.approx(390.0 * 1.08)  # 读数跟着新版本走
    assert b01["eta_minutes"] == math.ceil(b01["estimated_volume_ml"] / 5.0 * 60)

    # 版本轨迹 append-only：改前 / 改后 / 差多少都在
    trail = calibrations.list_recalc("TYPE-A")
    assert len(trail) == 2
    first_trail = trail[0]
    assert first_trail["previous_version"] == 1
    assert first_trail["new_version"] == 2
    assert first_trail["previous_volume_ml"] == pytest.approx(390.0)
    assert first_trail["delta_ml"] == pytest.approx(
        first_trail["new_volume_ml"] - first_trail["previous_volume_ml"]
    )
    assert first_trail["reason"] == "calibration_version_changed"
    # 轨迹只追加：再存一版后旧行一条不少
    calibrations.save(type_a(capacity_ml=700.0))
    assert len(calibrations.list_recalc("TYPE-A")) == 2


def test_first_registration_is_logged_differently_from_a_version_change(conn):
    """首次登记（此前没有生效版本）与改版记不同原因码——审计不能把两者混成一句话。"""
    sessions = SessionRepository(conn)
    calibrations = CalibrationRepository(conn)

    # 上传时还没登记标定：只有面积比，没有毫升数
    _seed_session(sessions)
    session_id = sessions.find_by_rack_shift_bag(
        "R-07", "2026-10-22-AM", "B01"
    )["session_id"]
    sessions.update_calibration(session_id, {"fill_ratio": 0.60})

    class _Store:
        pass

    store = _Store()
    store.sessions = sessions
    store.calibrations = calibrations

    first, superseded, _status = calibrations.save(type_a())
    assert superseded is None
    summary = recompute_for_new_version(store, first, superseded)

    assert summary["recalculated_count"] == 1
    assert sessions.get(session_id)["estimated_volume_ml"] == pytest.approx(390.0)

    row = calibrations.list_recalc("TYPE-A")[0]
    assert row["reason"] == "calibration_registered"
    assert row["previous_version"] is None
    # 改前**没有数**，不是 0 —— 轨迹里如实留空，不写 0 冒充
    assert row["previous_volume_ml"] is None
    assert row["delta_ml"] is None
    assert row["new_volume_ml"] == pytest.approx(390.0)


# ── 6 端到端：真实上传路径 ───────────────────────────────────────────────────


def _upload(client, slots: list[dict]):
    frames = synth.build_rack_zip(synth.default_rack_scene(24))
    return client.post(
        "/api/v1/infusion/tracks",
        data={
            "rack_id": "R-07",
            "infusion_order_no": "ORDER-0001",
            "bed_no": "12",
            "captured_at": "2026-10-22T08:00:00+08:00",
            "slots": json.dumps(slots, ensure_ascii=False),
        },
        files={"frames": ("frames.zip", frames, "application/zip")},
    )


def test_upload_without_calibration_has_no_volume_field(client):
    response = _upload(
        client,
        [
            {"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-A", "rate_text": "5ml/h"},
            {"bag_seq": "B02", "slot_index": 1, "bag_type_code": "TYPE-A", "rate_text": "5ml/h"},
        ],
    )
    assert response.status_code == 201
    for bag in response.json()["bags"]:
        assert bag["calibration_state"] == "uncalibrated"
        assert "estimated_volume_ml" not in bag  # 门禁：字段整体不出现
        assert "eta_minutes" not in bag
        assert bag["pickup_suppressed"] is True
        assert bag["suppression_reason"] == "uncalibrated"
        assert bag["area_ratio_series"]  # 只保留曲线本身


def test_upload_after_calibration_returns_ml_and_eta(client):
    created = client.post(
        "/api/v1/infusion/calibrations",
        json={
            "bag_type_code": "TYPE-A",
            "capacity_ml": 600.0,
            "dead_volume_ml": 20.0,
            "samples": [
                {"measured_fill_ratio": 0.10, "known_volume_ml": 58.0, "source": "称重法"},
                {"measured_fill_ratio": 0.60, "known_volume_ml": 378.0, "source": "称重法"},
                {"measured_fill_ratio": 0.90, "known_volume_ml": 520.0, "source": "换液确认反推"},
            ],
            "calibrated_by": "OP-NURSE-01",
            "note": "首版：按袋上刻度带登记",
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["version"] == 1
    assert body["usable_capacity_ml"] == pytest.approx(580.0)
    assert body["bands"][0] == [0.0, 0.0]

    response = _upload(
        client,
        [
            {"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-A", "rate_text": "5ml/h"},
            {"bag_seq": "B02", "slot_index": 1, "bag_type_code": "TYPE-A", "rate_text": "5ml/h"},
        ],
    )
    assert response.status_code == 201
    bags = response.json()["bags"]
    assert [bag["calibration_state"] for bag in bags] == ["calibrated", "calibrated"]
    assert all("estimated_volume_ml" in bag for bag in bags)
    assert all("eta_minutes" in bag for bag in bags)
    assert all(bag["eta_source"] == "order_text" for bag in bags)
    assert all(bag["pickup_suppressed"] is False for bag in bags)
    # 两袋液位不同 → 毫升数与 ETA 必须不同（没串袋）
    assert bags[0]["estimated_volume_ml"] != bags[1]["estimated_volume_ml"]

    # 落库行与响应一致，且带上标定版本
    stored = client.get(f"/api/v1/infusion/sessions/{bags[0]['session_id']}").json()
    assert stored["calibration_version"] == 1
    assert stored["rate_ml_per_h"] == 5.0
    assert stored["estimated_volume_ml"] == pytest.approx(bags[0]["estimated_volume_ml"])


def test_upload_without_rate_keeps_ml_but_drops_eta(client):
    client.post(
        "/api/v1/infusion/calibrations",
        json={
            "bag_type_code": "TYPE-B",
            "capacity_ml": 250.0,
            "samples": [{"measured_fill_ratio": 0.5, "known_volume_ml": 125.0}],
            "calibrated_by": "OP-NURSE-02",
        },
    )
    response = _upload(
        client,
        [{"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-B", "rate_text": "常规输液"}],
    )
    assert response.status_code == 201
    bag = response.json()["bags"][0]
    assert bag["estimated_volume_ml"] is not None
    assert "eta_minutes" not in bag
    assert bag["eta_source"] == "unavailable"
    assert bag["pickup_suppressed"] is True
    assert bag["suppression_reason"] == "eta_unavailable"


def test_telemetry_only_rate_gives_eta_from_telemetry(client):
    client.post(
        "/api/v1/infusion/calibrations",
        json={
            "bag_type_code": "TYPE-C",
            "capacity_ml": 100.0,
            "samples": [{"measured_fill_ratio": 0.5, "known_volume_ml": 50.0}],
            "calibrated_by": "OP-NURSE-03",
        },
    )
    client.app.state.telemetry.publish(
        "R-07", "B01", TelemetrySample(60.0, device_sn="PUMP-9", ts="2026-10-22T08:00:00+08:00")
    )
    response = _upload(
        client, [{"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-C"}]
    )
    bag = response.json()["bags"][0]
    assert bag["eta_source"] == "mqtt_telemetry"
    assert bag["rate_ml_per_h"] == 60.0
    assert bag["estimated_volume_ml"] is not None


def test_calibration_endpoints_expose_versions_and_recalc_trail(client):
    client.post(
        "/api/v1/infusion/calibrations",
        json={
            "bag_type_code": "TYPE-D",
            "capacity_ml": 500.0,
            "samples": [{"measured_fill_ratio": 0.5, "known_volume_ml": 250.0}],
            "calibrated_by": "OP-NURSE-04",
        },
    )
    listing = client.get("/api/v1/infusion/calibrations?bag_type_code=TYPE-D").json()
    assert listing["count"] == 1
    assert listing["calibrations"][0]["version"] == 1

    _upload(
        client, [{"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-D", "rate_text": "10ml/h"}]
    )
    second = client.post(
        "/api/v1/infusion/calibrations",
        json={
            "bag_type_code": "TYPE-D",
            "capacity_ml": 560.0,
            "samples": [
                {"measured_fill_ratio": 0.5, "known_volume_ml": 280.0},
                {"measured_fill_ratio": 0.9, "known_volume_ml": 500.0},
            ],
            "calibrated_by": "OP-NURSE-05",
            "note": "换批次",
        },
    ).json()
    assert second["version"] == 2
    assert second["superseded_version"] == 1
    assert second["recalculated_sessions"] == 1

    active = client.get("/api/v1/infusion/calibrations").json()
    assert [item["bag_type_code"] for item in active["calibrations"]] == ["TYPE-D"]
    assert active["calibrations"][0]["version"] == 2

    trail = client.get("/api/v1/infusion/calibrations/recalc?bag_type_code=TYPE-D").json()
    assert trail["count"] == 1
    assert trail["recalc_log"][0]["previous_version"] == 1
    assert trail["recalc_log"][0]["new_version"] == 2
    assert trail["recalc_log"][0]["rack_id"] == "R-07"


def test_calibration_rejects_out_of_range_samples(client):
    response = client.post(
        "/api/v1/infusion/calibrations",
        json={
            "bag_type_code": "TYPE-E",
            "capacity_ml": 500.0,
            "samples": [{"measured_fill_ratio": 1.8, "known_volume_ml": 250.0}],
            "calibrated_by": "OP-NURSE-06",
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "INGEST_INVALID"