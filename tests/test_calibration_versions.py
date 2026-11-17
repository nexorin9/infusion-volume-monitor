"""M3 标定版本管理：生效区间唯一性、改版重算与偏差复核闸门。

覆盖 task 27 的三条硬断言：

1. **生效区间唯一性**：同一 ``bag_type_code`` 的生效区间不允许相交；预约登记
   （未来才生效）不动当天读数，到点才转正；
2. **版本切换触发重算**：改版后历史会话按新版本重算，差异落 append-only 轨迹，
   可追溯到「谁在何时标的第几版」；
3. **偏差告警**：相对差超阈值（默认 15%）**不静默覆盖**旧读数，而是保留原值 +
   挂待复核标记 + 落告警，人工结案后才生效。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from infusion_monitor.calibration.models import BagTypeCalibration, CalibrationSample
from infusion_monitor.calibration.versions import (
    DEFAULT_REVIEW_THRESHOLD,
    OverlappingVersionWindow,
    VersionWindow,
    active_at,
    assert_no_overlapping_window,
    classify_all,
    classify_version,
    judge_deviation,
    parse_timestamp,
    plan_supersede,
    promote_due,
    relative_deviation,
)
from infusion_monitor.store import CalibrationRepository, SessionRepository

TYPE_A_SAMPLES = (
    CalibrationSample(0.10, 60.0, source="称重法"),
    CalibrationSample(0.35, 220.0, source="称重法"),
    CalibrationSample(0.60, 390.0, source="称重法"),
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


def _shifted_samples(factor: float) -> tuple[CalibrationSample, ...]:
    return tuple(
        CalibrationSample(s.measured_fill_ratio, s.known_volume_ml * factor, s.source)
        for s in TYPE_A_SAMPLES
    )


def _seed_session(sessions: SessionRepository, bag_seq: str = "B01", **kwargs) -> str:
    """按 SessionRepository 的字段口径插一条监测中的会话，返回 session_id。"""
    params = {
        "rack_id": "R-07",
        "bed_no": "07-3",
        "infusion_order_no": "EXAMPLE-ORDER-0027",
        "shift_key": "2026-10-22-AM",
        "bag_seq": bag_seq,
        "slot_index": 0,
        "bag_type_code": "TYPE-A",
        "frames_dir": f"/tmp/{bag_seq}",
        "frame_count": 20,
        "captured_at": "2026-10-22T09:41:00+08:00",
    }
    params.update(kwargs)
    return sessions.upsert_session(params)["session_id"]


class _Store:
    """``recompute_for_new_version`` 只需要这两个仓储，故不必起整个应用。"""

    def __init__(self, conn):
        self.sessions = SessionRepository(conn)
        self.calibrations = CalibrationRepository(conn)
        self.audit_chain = None
        self.deviations = None


# ── 1 生效区间：唯一性与状态判定 ────────────────────────────────────────────


def test_overlapping_windows_are_rejected_for_the_same_bag_type(conn):
    """同一袋型的两版区间相交 → 拒收，不允许「两条曲线同时生效」。"""
    repo = CalibrationRepository(conn)
    first, _, status = repo.save(type_a())
    assert status == "active"

    # 直接落在 v1 区间内（v1 从 08:00 起生效、无停用时刻）
    clash = VersionWindow(effective_from="2026-10-23T08:00:00+08:00")
    with pytest.raises(OverlappingVersionWindow) as excinfo:
        assert_no_overlapping_window(repo.list_by_code("TYPE-A"), clash, bag_type_code="TYPE-A")

    issues = excinfo.value.issues
    assert issues and issues[0].code == "version_window_overlap"
    assert "v1" in issues[0].reason
    # 建议文案要能照做：指出该把生效时刻放到旧版之后
    assert "同一袋型同时只能有一个生效版本" in issues[0].suggestion
    assert first.version == 1


def test_save_rejects_a_backdated_window(conn):
    """把新版本的生效时刻登记到**过去**：交接点早于旧版的生效时刻，等于伪造记录。

    改版交接本身不冲突（旧版会被收口到新版生效时刻），但**回填**生效时刻会要求
    把旧版的有效期往回拉长——那不是交接，是把历史改掉，必须挡在落库前。
    """
    repo = CalibrationRepository(conn)
    repo.save(type_a())

    with pytest.raises(OverlappingVersionWindow):
        repo.save(type_a(capacity_ml=650.0, effective_from="2026-10-20T08:00:00+08:00"))

    assert [r.version for r in repo.list_by_code("TYPE-A")] == [1]
    assert repo.get_active("TYPE-A").capacity_ml == 600.0  # 旧版没被动过


def test_immediate_revision_closes_the_previous_window_without_overlap(conn):
    """改版交接：旧版收口成 [08:00, 改版时刻)，新区间从改版时刻起——两者不相交。"""
    repo = CalibrationRepository(conn)
    repo.save(type_a())
    second, superseded, status = repo.save(type_a(capacity_ml=650.0))

    assert status == "active"
    assert superseded == 1
    old = repo.get_active("TYPE-A")
    assert old.version == 2
    history = {r.version: r for r in repo.list_by_code("TYPE-A")}
    v1_window = VersionWindow(
        effective_from=history[1].effective_from, effective_to=history[1].effective_to
    )
    v2_window = VersionWindow(
        effective_from=history[2].effective_from, effective_to=history[2].effective_to
    )
    assert not v1_window.overlaps(v2_window)
    assert v1_window.end() is not None  # 旧版已被收口，不再「至今」
    assert second.version == 2


def test_scheduled_version_leaves_todays_reading_untouched(conn):
    """预约登记：未来才生效的版本 active=0，当前生效版本一点不动。"""
    repo = CalibrationRepository(conn)
    repo.save(type_a())
    tomorrow = (datetime.now().astimezone() + timedelta(days=1)).isoformat(timespec="seconds")

    future, superseded, status = repo.save(type_a(capacity_ml=650.0, effective_from=tomorrow))

    assert status == "scheduled"
    assert superseded == 1  # 上一版仍是被取代关系，但当下还在用
    assert future.active is False
    assert repo.get_active("TYPE-A").version == 1  # 今天算出来的毫升数不变
    v1 = next(r for r in repo.list_by_code("TYPE-A") if r.version == 1)
    # 旧版区间按计划**事先**收口到新版生效时刻，但它此刻仍在生效
    assert v1.effective_to == tomorrow
    assert v1.active is True
    assert classify_version(v1, datetime.now().astimezone()).status == "active"
    assert classify_version(future, datetime.now().astimezone()).status == "scheduled"


def test_promotion_happens_when_the_scheduled_version_becomes_due(conn):
    """到点转正：交接由版本查询动作完成，但**不重算**历史会话的毫升数。"""
    repo = CalibrationRepository(conn)
    repo.save(type_a())
    soon = (datetime.now().astimezone() + timedelta(minutes=1)).isoformat(timespec="seconds")
    repo.save(type_a(capacity_ml=650.0, effective_from=soon))

    records = repo.list_by_code("TYPE-A")
    assert promote_due(records, at=parse_timestamp(soon) - timedelta(minutes=5)) == []  # 还没到

    due = promote_due(records, at=parse_timestamp(soon) + timedelta(hours=1))
    assert [view.version for view in due] == [2]

    superseded = repo.promote(due[0].calibration_id, effective_from=soon)
    assert superseded == 1
    assert repo.get_active("TYPE-A").version == 2
    assert classify_version(next(r for r in repo.list_by_code("TYPE-A") if r.version == 1),
                            parse_timestamp(soon)).status == "expired"


def test_classify_marks_scheduled_active_and_expired():
    """版本状态按时刻判定：还没到 = scheduled，已过停用时刻 = expired。"""
    record = BagTypeCalibration.from_samples(
        "TYPE-B",
        capacity_ml=500.0,
        samples=TYPE_A_SAMPLES,
        calibrated_at="2026-11-01T00:00:00+08:00",
        effective_from="2026-11-01T00:00:00+08:00",
        effective_to="2026-12-01T00:00:00+08:00",
    )
    assert classify_version(record, parse_timestamp("2026-10-31T23:00:00+08:00")).status == "scheduled"
    assert classify_version(record, parse_timestamp("2026-11-15T00:00:00+08:00")).status == "active"
    # 交接后旧版被翻成非生效且已过停用时刻 → expired
    retired = replace(record, active=False)
    assert classify_version(retired, parse_timestamp("2026-12-02T00:00:00+08:00")).status == "expired"
    # 已收口但**仍在生效**（预约版没到点）→ 报 active，而不是提前宣布过期
    assert classify_version(record, parse_timestamp("2026-12-02T00:00:00+08:00")).status == "active"

    views = classify_all([record])
    assert views[0].status in ("scheduled", "active", "expired")
    assert views[0].as_dict()["effective_from"] == "2026-11-01T00:00:00+08:00"


def test_active_at_picks_the_single_effective_version():
    record = BagTypeCalibration.from_samples(
        "TYPE-C", 500.0, TYPE_A_SAMPLES, calibrated_at="2026-11-01T00:00:00+08:00"
    )
    assert active_at([record], parse_timestamp("2026-11-05T00:00:00+08:00")) is record
    # 生效前没有任何版本可用 → 调用方走 uncalibrated 分支，不猜
    assert active_at([record], parse_timestamp("2026-10-20T00:00:00+08:00")) is None


def test_window_overlap_is_half_open():
    """半开区间：交接点不重叠——旧版 12:00 结束、新版 12:00 开始，二者不冲突。"""
    old = VersionWindow(effective_from="2026-11-01T00:00:00+08:00", effective_to="2026-11-02T00:00:00+08:00")
    new = VersionWindow(effective_from="2026-11-02T00:00:00+08:00")
    assert not old.overlaps(new)
    assert not new.overlaps(old)
    assert old.contains(parse_timestamp("2026-11-01T12:00:00+08:00"))
    assert not old.contains(parse_timestamp("2026-11-02T00:00:00+08:00"))


def test_plan_supersede_closes_previous_at_the_new_start():
    """预约版与即刻改版都要把旧版区间收口到新版起点——区间是计划，不是事后补记。"""
    now = datetime.now().astimezone()
    future = (now + timedelta(days=1)).isoformat(timespec="seconds")

    superseded = type_a(version=1, calibration_id="cal-1", effective_from="2026-10-22T08:00:00+08:00")
    close_to, status = plan_supersede(superseded, VersionWindow(effective_from=future))
    assert status == "scheduled"
    assert close_to == future  # 预约：区间已收口，但 active 不动

    close_to, status = plan_supersede(superseded, VersionWindow(effective_from=now.isoformat()))
    assert status == "active"
    assert parse_timestamp(close_to) is not None

    # 首次登记没有旧版要收口
    assert plan_supersede(None, VersionWindow(effective_from=now.isoformat())) == (None, "active")


def test_naive_timestamps_are_read_as_local_not_utc():
    """「2026-11-01 零点」按院内本地时区理解；一律按 UTC 解会把交接点错开一个时差。"""
    local_tz = datetime.now().astimezone().tzinfo
    naive = parse_timestamp("2026-11-01T00:00:00")
    assert naive == datetime(2026, 11, 1, 0, 0, tzinfo=local_tz)
    # 带 Z 的按 UTC 原样理解，不受本机时区影响
    assert parse_timestamp("2026-11-01T00:00:00Z") == datetime(
        2026, 11, 1, 0, 0, tzinfo=timezone.utc
    )
    # 两种写法描述的是同一时刻时，判定结果必须一致
    stamp = datetime.now().astimezone().replace(microsecond=0)
    same_instant = (stamp - datetime(2026, 11, 1, tzinfo=stamp.tzinfo)).days and stamp
    assert parse_timestamp(stamp.isoformat()) == parse_timestamp(same_instant.isoformat())


# ── 2 版本切换触发重算 ──────────────────────────────────────────────────────


def test_version_switch_recomputes_history_and_keeps_the_trail(conn):
    """改版触发重算，轨迹逐条可查（改前/改后/差多少/哪个版本/谁标的）。"""
    store = _Store(conn)
    sessions, repo = store.sessions, store.calibrations
    _seed_session(sessions, "B01")
    _seed_session(sessions, "B02", slot_index=1)
    # 已换袋的会话保留当时的读数，不追溯修改
    _seed_session(sessions, "B03", slot_index=2, state="replaced")
    for bag_seq, ratio, ml in (("B01", 0.60, 390.0), ("B02", 0.10, 60.0), ("B03", 0.35, 220.0)):
        sessions.update_calibration(
            sessions.find_by_rack_shift_bag("R-07", "2026-10-22-AM", bag_seq)["session_id"],
            {"fill_ratio": ratio, "estimated_volume_ml": ml},
        )

    repo.save(type_a())
    # 换批次后袋腔略大：样例整体高 8%，仍在 15% 闸门内 → 直接覆盖
    new_record, previous_version, status = repo.save(
        type_a(capacity_ml=650.0, samples=_shifted_samples(1.08), calibrated_by="OP-NURSE-02")
    )
    assert status == "active"
    summary = recompute(repo, store, new_record, previous_version)

    assert summary["recalculated_count"] == 2  # 已换袋的 B03 不追溯
    assert summary["review_required_count"] == 0
    trail = repo.list_recalc("TYPE-A")
    assert {row["session_id"] for row in trail} == {
        sessions.find_by_rack_shift_bag("R-07", "2026-10-22-AM", bag)["session_id"]
        for bag in ("B01", "B02")
    }
    for row in trail:
        assert row["reason"] == "calibration_version_changed"
        assert row["previous_version"] == 1 and row["new_version"] == 2
        assert row["previous_volume_ml"] is not None
        assert row["new_volume_ml"] > row["previous_volume_ml"]  # 袋腔大了，读数跟着涨
        assert row["delta_ml"] == pytest.approx(row["new_volume_ml"] - row["previous_volume_ml"], abs=0.05)
        assert row["review_required"] == 0
        # 复核标记在自动覆盖后清干净，不留悬空状态
        assert sessions.get(row["session_id"])["calibration_review_required"] == 0


def recompute(repo, store, record, previous_version):
    from infusion_monitor.calibration.estimator import recompute_for_new_version

    return recompute_for_new_version(store, record, previous_version)


# ── 3 偏差告警：不静默覆盖，等人工复核 ──────────────────────────────────────


def test_relative_deviation_uses_the_larger_side_as_denominator():
    """袋快见底时报 3 ml：按旧值当分母会算出 19 倍，凭空造出一条 critical。"""
    assert relative_deviation(3.0, 60.0) == pytest.approx(57.0 / 60.0)
    assert relative_deviation(3.0, 3.0) == 0.0
    assert relative_deviation(None, 100.0) is None  # 「本来就没数」不是「差得离谱」
    assert relative_deviation(100.0, None) is None


def test_judge_deviation_escalates_beyond_the_error_threshold():
    ok = judge_deviation(390.0, 420.0)  # 7.1%：闸门内
    assert ok.review_required is False and ok.severity == "notice"

    review = judge_deviation(220.0, 275.0)  # 20.0%：过 15% 闸门、不到 25% 错误线 → review
    assert review.review_required is True
    assert review.severity == "review"
    assert review.threshold == DEFAULT_REVIEW_THRESHOLD

    critical = judge_deviation(220.0, 100.0)  # 54.5% > 25%
    assert critical.severity == "critical"

    first_time = judge_deviation(None, 390.0)  # 首次登记没有旧读数 → 不算偏差
    assert first_time.review_required is False and first_time.relative_deviation is None


def test_large_deviation_does_not_overwrite_the_old_reading(conn):
    """超阈值的那几条：旧读数原样保留，会话挂待复核，告警逐条落库。"""
    store = _Store(conn)
    sessions, repo = store.sessions, store.calibrations
    session_id = _seed_session(sessions, "B01")
    sessions.update_calibration(session_id, {"fill_ratio": 0.60, "estimated_volume_ml": 390.0})

    repo.save(type_a())
    # 这次改版把样例整体拉高 40%：多半是新版本自己错了，不是旧版本终于被修正
    new_record, previous_version, _status = repo.save(
        type_a(capacity_ml=650.0, samples=_shifted_samples(1.40))
    )
    summary = recompute(repo, store, new_record, previous_version)

    assert summary["review_required_count"] == 1
    assert summary["recalculated_count"] == 0  # 一个都没自动生效
    held = summary["held_for_review"][0]
    assert held["previous_volume_ml"] == 390.0
    assert held["proposed_volume_ml"] > 500.0

    session = sessions.get(session_id)
    assert session["estimated_volume_ml"] == 390.0  # 旧读数没被覆盖
    assert session["calibration_review_required"] == 1
    assert session["calibration_review_version"] == 2
    assert session["calibration_review_deviation"] == pytest.approx(
        (held["proposed_volume_ml"] - 390.0) / held["proposed_volume_ml"], abs=1e-6
    )

    alerts = repo.list_alerts("TYPE-A")
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["resolution"] == "pending"  # append-only，等护理部结案
    assert alert["session_id"] == session_id
    assert alert["previous_volume_ml"] == 390.0
    assert alert["relative_deviation"] > DEFAULT_REVIEW_THRESHOLD
    assert alert["threshold"] == DEFAULT_REVIEW_THRESHOLD
    assert alert["severity"] in ("review", "critical")

    # 重算轨迹照样留一行，但要标明「新读数未生效」
    trail = repo.list_recalc("TYPE-A")
    assert len(trail) == 1 and trail[0]["review_required"] == 1
    assert trail[0]["previous_volume_ml"] == 390.0


def test_review_resolution_applies_the_new_reading_and_records_who(conn):
    """复核通过 → 新读数落库、待复核标记清掉、告警结案并留下复核人。"""
    store = _Store(conn)
    sessions, repo = store.sessions, store.calibrations
    session_id = _seed_session(sessions, "B01")
    sessions.update_calibration(session_id, {"fill_ratio": 0.60, "estimated_volume_ml": 390.0})
    repo.save(type_a())
    new_record, previous_version, _ = repo.save(
        type_a(capacity_ml=650.0, samples=_shifted_samples(1.40))
    )
    summary = recompute(repo, store, new_record, previous_version)
    proposed = summary["held_for_review"][0]["proposed_volume_ml"]
    alert_id = repo.list_alerts("TYPE-A")[0]["alert_id"]

    resolved = repo.resolve_alert(alert_id, "applied", by="OP-NURSE-03", note="复称确认新批次袋腔更大")
    assert resolved["resolution"] == "applied"
    assert resolved["resolved_by"] == "OP-NURSE-03"
    assert resolved["resolution_note"] == "复称确认新批次袋腔更大"

    sessions.update_calibration(
        session_id,
        {
            "estimated_volume_ml": proposed,
            "calibration_version": 2,
            "calibration_review_required": 0,
            "calibration_review_deviation": None,
            "calibration_review_version": None,
            "calibration_review_proposed_ml": None,
        },
    )
    session = sessions.get(session_id)
    assert session["estimated_volume_ml"] == pytest.approx(proposed, abs=0.01)
    assert session["calibration_version"] == 2
    assert session["calibration_review_required"] == 0
    assert repo.count_alerts("pending") == 0


def test_dismissed_review_keeps_the_old_reading(conn):
    """复核否决 → 新版本有问题，旧读数继续沿用；重复结案显式报错而不是默默改写。"""
    store = _Store(conn)
    sessions, repo = store.sessions, store.calibrations
    session_id = _seed_session(sessions, "B01")
    sessions.update_calibration(session_id, {"fill_ratio": 0.60, "estimated_volume_ml": 390.0})
    repo.save(type_a())
    new_record, previous_version, _ = repo.save(
        type_a(capacity_ml=650.0, samples=_shifted_samples(1.40))
    )
    recompute(repo, store, new_record, previous_version)
    alert_id = repo.list_alerts("TYPE-A")[0]["alert_id"]

    dismissed = repo.resolve_alert(alert_id, "dismissed", by="OP-NURSE-04", note="样例称错了")
    assert dismissed["resolution"] == "dismissed"
    assert sessions.get(session_id)["estimated_volume_ml"] == 390.0  # 旧读数仍在

    with pytest.raises(ValueError, match="不可重复处理"):
        repo.resolve_alert(alert_id, "applied", by="OP-NURSE-05")
    with pytest.raises(ValueError, match="非法告警结案状态"):
        repo.resolve_alert(alert_id, "pending", by="OP-NURSE-05")


# ── 4 HTTP 契约：登记回执 / 版本查询 / 复核入口 ─────────────────────────────


def test_http_registration_reports_status_and_reviews(client):
    """POST /calibrations 回执带版本状态、生效区间与复核计数。"""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    body = {
        "bag_type_code": "TYPE-A",
        "capacity_ml": 600.0,
        "samples": [
            {"measured_fill_ratio": s.measured_fill_ratio, "known_volume_ml": s.known_volume_ml}
            for s in TYPE_A_SAMPLES
        ],
        "calibrated_by": "OP-NURSE-01",
        "effective_from": now,
    }
    first = client.post("/api/v1/infusion/calibrations", json=body)
    assert first.status_code == 201
    assert first.json()["version"] == 1
    assert first.json()["version_status"] == "active"
    assert first.json()["review_threshold"] == DEFAULT_REVIEW_THRESHOLD

    versions = client.get("/api/v1/infusion/calibrations/TYPE-A/versions").json()
    assert versions["active_version"] == 1
    assert versions["scheduled_versions"] == []
    assert versions["count"] == 1
    assert versions["versions"][0]["calibrated_by"] == "OP-NURSE-01"
    assert versions["versions"][0]["status"] == "active"

    unknown = client.get("/api/v1/infusion/calibrations/TYPE-NOPE/versions").json()
    assert unknown["registered"] is False
    assert unknown["active_version"] is None and unknown["count"] == 0


def test_http_scheduled_registration_does_not_change_todays_version(client):
    """预约登记：回执 status=scheduled，查询端点把它排进 scheduled_versions。"""
    body = {
        "bag_type_code": "TYPE-B",
        "capacity_ml": 500.0,
        "samples": [{"measured_fill_ratio": 0.5, "known_volume_ml": 260.0}],
        "calibrated_by": "OP-NURSE-02",
    }
    assert client.post("/api/v1/infusion/calibrations", json=body).json()["version_status"] == "active"

    future = (datetime.now().astimezone() + timedelta(days=7)).isoformat(timespec="seconds")
    body["capacity_ml"] = 520.0
    body["effective_from"] = future
    scheduled = client.post("/api/v1/infusion/calibrations", json=body)
    assert scheduled.status_code == 201
    assert scheduled.json()["version_status"] == "scheduled"
    assert scheduled.json()["recalculated_sessions"] == 0  # 预约版不重算

    versions = client.get("/api/v1/infusion/calibrations/TYPE-B/versions").json()
    assert versions["active_version"] == 1
    assert versions["scheduled_versions"] == [2]
    assert [v["status"] for v in versions["versions"]] == ["active", "scheduled"]
    assert versions["versions"][1]["effective_from"] == future


def test_http_overlapping_window_is_rejected_with_actionable_issues(client):
    """区间相交走 409，detail 给三段式 issue，调用方知道该把生效时刻挪到哪。"""
    body = {
        "bag_type_code": "TYPE-C",
        "capacity_ml": 500.0,
        "samples": [{"measured_fill_ratio": 0.5, "known_volume_ml": 260.0}],
        "calibrated_by": "OP-NURSE-01",
    }
    assert client.post("/api/v1/infusion/calibrations", json=body).status_code == 201

    # 回填生效时刻（落在 v1 生效之前）→ 409
    body["capacity_ml"] = 520.0
    body["effective_from"] = "2020-01-01T00:00:00+08:00"
    clash = client.post("/api/v1/infusion/calibrations", json=body)
    assert clash.status_code == 409
    detail = clash.json()["detail"]
    assert detail["error_code"] == "CALIBRATION_WINDOW_CONFLICT"
    issue = detail["issues"][0]
    assert issue["code"] == "version_window_overlap"
    assert "同一袋型同时只能有一个生效版本" in issue["suggestion"]

    # 库里仍然只有一版，没有落下第二行
    versions = client.get("/api/v1/infusion/calibrations/TYPE-C/versions").json()
    assert versions["count"] == 1


def test_http_review_endpoints_reject_unknown_alert(client):
    """复核入口对不存在的告警给 404 与可照做的建议，而不是 500。"""
    response = client.post(
        "/api/v1/infusion/calibrations/TYPE-A/reviews/999",
        json={"resolution": "applied", "reviewed_by": "OP-NURSE-01"},
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["error_code"] == "NOT_FOUND"
    assert detail["issues"][0]["code"] == "alert_not_found"


def test_http_versions_endpoint_promotes_a_due_scheduled_version(store_client):
    """到期未生效的版本在查询时转正，并留一行自动交接的审计。

    用 ``store_client`` 而不是 ``client``：TestClient 的 SQLite 连接建在门户线程上，
    用例线程改不了它（跨线程用同一连接会抛 ProgrammingError）。这里要拨动生效时刻，
    故另开一个指到同一文件库的观察连接。
    """
    client, observer = store_client
    body = {
        "bag_type_code": "TYPE-D",
        "capacity_ml": 500.0,
        "samples": [{"measured_fill_ratio": 0.5, "known_volume_ml": 260.0}],
        "calibrated_by": "OP-NURSE-01",
    }
    client.post("/api/v1/infusion/calibrations", json=body)

    # 预约一个 1 分钟后生效的版本：登记时当天读数仍按 v1 走（不重算）
    soon = (datetime.now().astimezone() + timedelta(minutes=1)).isoformat(timespec="seconds")
    body["capacity_ml"] = 520.0
    body["effective_from"] = soon
    scheduled = client.post("/api/v1/infusion/calibrations", json=body)
    assert scheduled.json()["version_status"] == "scheduled"
    assert scheduled.json()["recalculated_sessions"] == 0

    before = client.get("/api/v1/infusion/calibrations/TYPE-D/versions").json()
    assert before["active_version"] == 1 and before["scheduled_versions"] == [2]
    assert before["promoted_versions"] == []

    # 越过生效时刻再查：交接完成，v2 转正；旧版区间早已按计划收口，故此刻即 expired
    past = (datetime.now().astimezone() - timedelta(minutes=1)).isoformat(timespec="seconds")
    observer.execute(
        "UPDATE bag_type_calibration SET effective_from = ? WHERE bag_type_code = ? "
        "AND version = 2",
        (past, "TYPE-D"),
    )
    promoted = client.get("/api/v1/infusion/calibrations/TYPE-D/versions").json()
    assert promoted["promoted_versions"] == [2]
    assert promoted["active_version"] == 2
    assert promoted["scheduled_versions"] == []
    assert [v["status"] for v in promoted["versions"]] == ["expired", "active"]

    # 交接留痕：append-only 审计链里能查到这一行，标了自动转正
    rows = observer.execute(
        "SELECT action, outcome, detail FROM audit_log WHERE action = ?",
        ("calibration.version_superseded",),
    ).fetchall()
    assert rows, "自动转正必须留审计，否则事后说不清算的哪一版"
    assert rows[-1]["outcome"] == "accepted"
    assert '"auto_promoted":true' in rows[-1]["detail"].replace(" ", "")


def test_http_double_resolution_is_a_conflict_not_a_server_error(store_client):
    """同一条告警结案两次 → 409 而不是 500：它有资源，只是已经被人处理过了。"""
    client, observer = store_client
    body = {
        "bag_type_code": "TYPE-E",
        "capacity_ml": 500.0,
        "samples": [{"measured_fill_ratio": 0.5, "known_volume_ml": 260.0}],
        "calibrated_by": "OP-NURSE-01",
    }
    client.post("/api/v1/infusion/calibrations", json=body)
    future = (datetime.now().astimezone() + timedelta(days=1)).isoformat(timespec="seconds")
    body["capacity_ml"] = 520.0
    body["effective_from"] = future
    client.post("/api/v1/infusion/calibrations", json=body)

    # 直接造一条已结案的告警（走仓储层建版 + 告警写入，主路径由 verify_task27 覆盖）
    alert_id = observer.execute(
        "INSERT INTO calibration_version_alert (calibration_id, bag_type_code, "
        "previous_version, new_version, session_id, bag_seq, rack_id, bed_no, "
        "previous_volume_ml, proposed_volume_ml, delta_ml, relative_deviation, threshold, "
        "severity, resolution, resolved_by, resolved_at, raised_at) VALUES "
        "('cal-x', 'TYPE-E', 1, 2, 'sess-x', 'B01', 'R-1', '07-1', 260.0, 380.0, 120.0, "
        "0.3158, 0.15, 'critical', 'dismissed', 'OP-NURSE-09', '2026-10-22T10:00:00+08:00', "
        "'2026-10-22T09:00:00+08:00')"
    ).lastrowid
    response = client.post(
        f"/api/v1/infusion/calibrations/TYPE-E/reviews/{alert_id}",
        json={"resolution": "applied", "reviewed_by": "OP-NURSE-10"},
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error_code"] == "STATE_CONFLICT"
    assert detail["issues"][0]["code"] == "review_already_resolved"
    assert "OP-NURSE-09" in detail["issues"][0]["reason"]  # 谁先处理的照实说


def test_error_code_table_documents_the_new_conflict():
    """新错误码进错误码表与文档生成器，挂接方不会拿着一份过期码表调接口。"""
    from infusion_monitor.errors import ERROR_CODES, error_code_table

    assert ERROR_CODES["CALIBRATION_WINDOW_CONFLICT"] == 409
    table = error_code_table()
    assert "`CALIBRATION_WINDOW_CONFLICT`" in table
    assert "| 409 |" in table
