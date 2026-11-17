"""存储层测试：建表、幂等迁移、幂等加列、三张表的读写往返。"""

from __future__ import annotations

import pytest

from infusion_monitor.store import (
    EventRepository,
    InferenceAuditRepository,
    SessionRepository,
    apply_migrations,
    connect,
    current_version,
    ensure_column,
    table_columns,
    transaction,
)
from infusion_monitor.store.schema import SCHEMA_VERSION


def _session_payload(**overrides):
    payload = {
        "rack_id": "RACK-A-01",
        "bed_no": "12-3",
        "infusion_order_no": "<EXAMPLE_ORDER>",
        "shift_key": "2026-10-22-AM",
        "bag_seq": "B01",
        "slot_index": 0,
        "bag_type_code": "500ml_soft",
        "frames_dir": "var/frames/sess-demo",
        "frame_count": 30,
        "captured_at": "2026-10-22T09:12:00+08:00",
        "area_ratio_series": [0.42, 0.41, 0.40, 0.40, 0.39],
        "area_ratio_raw": [0.43, 0.40, 0.41, 0.39, 0.38],
        "occlusion_flags": ["visible", "visible", "partial", "visible", "visible"],
    }
    payload.update(overrides)
    return payload


# ── 建表与迁移 ──────────────────────────────────────────────────────────────

def test_migrations_create_three_tables(conn):
    """三张业务表 + 迁移记录表一次建齐。"""
    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"bag_session", "infusion_event", "inference_audit", "schema_migrations"} <= names
    assert current_version(conn) == SCHEMA_VERSION


def test_migrations_are_idempotent(conn):
    """重复跑迁移不报错、版本不重复累加、不重复加列。"""
    first = apply_migrations(conn)
    second = apply_migrations(conn)
    third = apply_migrations(conn)

    assert first["added_columns"] == []
    assert second["added_columns"] == []
    assert third["version"] == SCHEMA_VERSION
    rows = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    assert rows == 1


def test_ensure_column_adds_then_skips(conn):
    """幂等加列：缺列补齐，再跑一次跳过。"""
    assert ensure_column(conn, "bag_session", "eta_source_probe", "TEXT") is True
    assert "eta_source_probe" in table_columns(conn, "bag_session")
    assert ensure_column(conn, "bag_session", "eta_source_probe", "TEXT") is False


def test_ensure_column_on_missing_table_raises(conn):
    """对不存在的表加列要显式报错，不静默吞掉。"""
    with pytest.raises(RuntimeError):
        ensure_column(conn, "no_such_table", "x", "TEXT")


def test_connect_creates_parent_directory(tmp_path):
    """首次启动自动建库目录，重复连接不报错。"""
    dsn = f"sqlite:///{tmp_path / 'var' / 'nested' / 'infusion.db'}"
    conn = connect(dsn)
    apply_migrations(conn)
    conn.close()
    again = connect(dsn)
    apply_migrations(again)
    assert again.execute("SELECT COUNT(*) FROM bag_session").fetchone()[0] == 0
    again.close()


# ── 会话仓储 ────────────────────────────────────────────────────────────────

def test_session_upsert_and_read_back(conn):
    """会话写入后可按 session_id 读回，JSON 列还原为列表。"""
    repo = SessionRepository(conn)
    written = repo.upsert_session(_session_payload())

    assert written["session_id"].startswith("sess_")
    assert written["state"] == "monitoring"
    assert written["calibration_state"] == "uncalibrated"
    assert written["estimated_volume_ml"] is None
    assert written["area_ratio_series"] == [0.42, 0.41, 0.4, 0.4, 0.39]
    assert written["occlusion_flags"][2] == "partial"
    assert written["session_version"] == 1

    fetched = repo.get(written["session_id"])
    assert fetched is not None
    assert fetched["rack_id"] == "RACK-A-01"
    assert fetched["infusion_order_no"] == "<EXAMPLE_ORDER>"
    assert repo.count() == 1


def test_repeat_upload_overwrites_same_rack_shift_bag(conn):
    """同 rack+shift+bag_seq 重复上传走覆盖式更新，不追加重复会话。"""
    repo = SessionRepository(conn)
    first = repo.upsert_session(_session_payload())
    second = repo.upsert_session(
        _session_payload(frame_count=45, area_ratio_series=[0.21, 0.20])
    )

    assert repo.count() == 1
    assert second["session_id"] == first["session_id"]
    assert second["session_version"] == 2
    assert second["frame_count"] == 45
    assert second["area_ratio_series"] == [0.21, 0.20]
    assert repo.find_by_rack_shift_bag("RACK-A-01", "2026-10-22-AM", "B01")["session_version"] == 2


def test_illegal_state_is_rejected(conn):
    """非法状态写入直接拒绝，不静默落库。"""
    repo = SessionRepository(conn)
    with pytest.raises(ValueError):
        repo.upsert_session(_session_payload(state="replaced_pending"))
    assert repo.count() == 0


def test_state_update_and_listing(conn):
    """状态推进与按状态查询可读，trigger_reason 随状态落库。"""
    repo = SessionRepository(conn)
    session = repo.upsert_session(_session_payload())
    updated = repo.update_state(
        session["session_id"],
        "needs_change",
        trigger_reason="eta_within_window",
        decision_source="threshold_fallback",
    )
    assert updated["state"] == "needs_change"
    assert updated["trigger_reason"] == "eta_within_window"

    assert len(repo.list_by_state("needs_change")) == 1
    assert repo.count_by_state() == {"needs_change": 1}
    assert [row["bag_seq"] for row in repo.list_by_order("<EXAMPLE_ORDER>")] == ["B01"]

    with pytest.raises(ValueError):
        repo.update_state(session["session_id"], "not_a_state")


def test_area_series_update_keeps_raw_and_flags(conn):
    """回写面积序列同时保留原始观测与遮挡标记，供人工核对。"""
    repo = SessionRepository(conn)
    session = repo.upsert_session(_session_payload())
    repo.update_area_series(
        session["session_id"],
        [0.30, 0.29, 0.28],
        raw=[0.31, 0.30, 0.28],
        occlusion_flags=["visible", "occluded", "visible"],
        frame_count_occluded=1,
    )
    row = repo.get(session["session_id"])
    assert row["area_ratio_series"] == [0.3, 0.29, 0.28]
    assert row["area_ratio_raw"] == [0.31, 0.3, 0.28]
    assert row["frame_count_occluded"] == 1


def test_update_state_rejects_unknown_field(conn):
    """不允许写白名单外的字段，避免绕过状态机语义改数据。"""
    repo = SessionRepository(conn)
    session = repo.upsert_session(_session_payload())
    with pytest.raises(ValueError):
        repo.update_state(session["session_id"], "needs_change", rack_id="HACK")


# ── 事件仓储（幂等键）────────────────────────────────────────────────────────

def _event_payload(**overrides):
    payload = {
        "client_event_id": "cid-0001",
        "session_id": "sess_demo",
        "rack_id": "RACK-A-01",
        "bed_no": "12-3",
        "infusion_order_no": "<EXAMPLE_ORDER>",
        "bag_seq": "B01",
        "fill_ratio": 0.08,
        "estimated_volume_ml": 24.0,
        "eta_minutes": 6,
        "trigger_reason": "eta_within_window",
        "decision_source": "threshold_fallback",
        "operator_id": "nurse_007",
        "occurred_at": "2026-10-22T09:40:00+08:00",
    }
    payload.update(overrides)
    return payload


def test_event_append_and_read_back(conn):
    """出站事件按字段规范落库，payload 快照可读回。"""
    repo = EventRepository(conn)
    row, created = repo.append_event(_event_payload())

    assert created is True
    assert row["event_id"].startswith("evt_")
    assert row["writeback_status"] == "pending"
    assert row["attempt_count"] == 0
    assert row["payload"]["operator_id"] == "nurse_007"
    assert repo.count() == 1


def test_duplicate_client_event_id_is_idempotent(conn):
    """同 client_event_id 重复确认不新增事件行。"""
    repo = EventRepository(conn)
    first, created_first = repo.append_event(_event_payload())
    second, created_second = repo.append_event(_event_payload(operator_id="nurse_008"))

    assert created_first is True
    assert created_second is False
    assert second["event_id"] == first["event_id"]
    assert second["payload"]["operator_id"] == "nurse_007"  # 保留首次快照，不被覆盖
    assert repo.count() == 1


def test_writeback_status_transitions(conn):
    """写回状态流转与重试队列筛选可用。"""
    repo = EventRepository(conn)
    row, _ = repo.append_event(_event_payload())
    sent = repo.mark_writeback(
        row["event_id"], "pending_writeback", attempt_count=3, last_error="HTTP 503"
    )
    assert sent["writeback_status"] == "pending_writeback"
    assert sent["attempt_count"] == 3
    assert repo.list_by_writeback_status("pending_writeback")[0]["event_id"] == row["event_id"]
    assert repo.count_by_writeback_status() == {"pending_writeback": 1}

    with pytest.raises(ValueError):
        repo.mark_writeback(row["event_id"], "weird_status")


def test_failed_transaction_rolls_back(conn):
    """事务内异常整体回滚：不留半写状态（不静默改数据）。"""
    repo = SessionRepository(conn)
    before = repo.count()
    with pytest.raises(RuntimeError):
        with transaction(conn):
            repo.upsert_session(_session_payload(bag_seq="B01"))
            repo.upsert_session(_session_payload(bag_seq="B02"))
            raise RuntimeError("注入失败：模拟写回途中抛错")
    assert repo.count() == before
    assert repo.find_by_rack_shift_bag("RACK-A-01", "2026-10-22-AM", "B01") is None
    assert repo.find_by_rack_shift_bag("RACK-A-01", "2026-10-22-AM", "B02") is None


# ── 推理审计仓储 ─────────────────────────────────────────────────────────────

def test_inference_audit_append_only(conn):
    """推理审计落 request_id/model/prompt_hash/latency/decision_source。"""
    repo = InferenceAuditRepository(conn)
    audit_id = repo.append(
        {
            "request_id": "req-001",
            "session_id": "sess_demo",
            "bag_seq": "B01",
            "model": "test-model",
            "prompt_hash": "sha256:abc123",
            "latency_ms": 812,
            "prompt_tokens": 320,
            "completion_tokens": 64,
            "decision_source": "threshold_fallback",
            "fallback_reason": "read_timeout",
            "pending_manual_review": True,
        }
    )
    assert audit_id > 0
    rows = repo.list_by_request("req-001")
    assert rows[0]["decision_source"] == "threshold_fallback"
    assert rows[0]["fallback_reason"] == "read_timeout"
    assert rows[0]["pending_manual_review"] == 1
    assert repo.count_by_decision_source() == {"threshold_fallback": 1}

    repo.append(
        {
            "request_id": "req-002",
            "model": "test-model",
            "prompt_hash": "sha256:def456",
            "decision_source": "model_decided",
            "decision": "monitor",
            "confidence": 0.81,
        }
    )
    assert repo.count() == 2
    assert [row["request_id"] for row in repo.list_recent(1)] == ["req-002"]


def test_no_patient_identifiable_fields_in_schema(conn):
    """三张表不落患者姓名等可识别明细（安全边界，配置面即固定）。"""
    columns = set(table_columns(conn, "bag_session")) | set(table_columns(conn, "infusion_event"))
    for forbidden in ("patient_name", "patient_id", "id_card", "name", "phone"):
        assert forbidden not in columns