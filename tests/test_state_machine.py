"""袋级状态机的验收测试：``monitoring → needs_change → replaced``。

四组断言：

1. **三态端到端**：服务算出 ``needs_change`` → 护士确认 → ``replaced``，每一步都有
   append-only 审计行，且 ``replaced`` **只能**由带 ``operator_id`` 的确认推进；
2. **非法迁移 409**：``monitoring`` 直跳 ``replaced``、``replaced`` 回退、自环，全部
   拒绝**并**写审计行（被拒的动作也要留痕，飞检查的就是这个）；
3. **needs_change 的双条件**：只有「连续 N 帧低于阈值」不够，余量也得到紧急线才行；
   单帧毛刺不触发；
4. **确认回写边界**：未确认的 ``needs_change`` 不产生出站事件；确认后给出的事件载荷
   带幂等键，且模型路径无权推进任何状态。
"""

from __future__ import annotations

import pytest

from infusion_monitor.errors import ApiError
from infusion_monitor.identity.state_machine import (
    ALLOWED_TRANSITIONS,
    ChangeParams,
    IllegalTransition,
    apply_decision,
    check_transition,
    confirm_replacement,
    evaluate_change,
)
from infusion_monitor.store import AuditChain, SessionRepository

DECISION_URL = "/api/v1/infusion/decision"


def _session_payload(**overrides):
    payload = {
        "rack_id": "R-EXAMPLE-09",
        "bed_no": "EXAMPLE-BED-09",
        "infusion_order_no": "EXAMPLE-ORDER-0009",
        "shift_key": "2026-10-22-AM",
        "bag_seq": "B01",
        "slot_index": 0,
        "bag_type_code": "TYPE-A",
        "frames_dir": "/tmp/frames",
        "frame_count": 16,
        "captured_at": "2026-10-22T09:41:00+08:00",
        "state": "monitoring",
    }
    payload.update(overrides)
    return payload


def _seed(conn, **overrides) -> dict:
    return SessionRepository(conn).upsert_session(_session_payload(**overrides))


def _near_empty_session(conn, **overrides) -> dict:
    """一袋已接近排空、滴速与标定都在手的会话（会被判定点名）。"""
    row = _seed(conn, **overrides)
    SessionRepository(conn).update_area_series(row["session_id"], [0.08, 0.06, 0.05, 0.04])
    SessionRepository(conn).update_calibration(
        row["session_id"],
        {
            "fill_ratio": 0.04,
            "estimated_volume_ml": 12.0,
            "eta_minutes": 12,
            "eta_source": "order_text",
            "rate_ml_per_h": 60.0,
            "calibration_state": "calibrated",
            "calibration_version": 1,
            "pickup_suppressed": False,
            "suppression_reason": None,
        },
    )
    return SessionRepository(conn).get(row["session_id"])


# ── 1. 三态端到端 ─────────────────────────────────────────────────────────


def test_three_states_end_to_end(conn):
    """monitoring → needs_change（服务算）→ replaced（护士确认），逐步留痕。"""
    sessions = SessionRepository(conn)
    audit = AuditChain(conn)
    record = _near_empty_session(conn)
    assert record["state"] == "monitoring"

    decision = evaluate_change(record)
    assert decision.changed is True
    assert decision.target_state == "needs_change"
    assert decision.trigger_reason == "near_empty_observed"
    assert decision.consecutive_frames >= ChangeParams().n_frames

    updated, changed = apply_decision(
        sessions, decision, audit=audit, request_id="req-flow-1"
    )
    assert changed is True
    assert updated["state"] == "needs_change"
    assert updated["trigger_reason"] == "near_empty_observed"

    confirmation = confirm_replacement(
        sessions,
        updated["session_id"],
        operator_id="N-EXAMPLE-1",
        audit=audit,
        request_id="req-flow-1",
    )
    assert confirmation.state == "replaced"
    assert sessions.get(record["session_id"])["state"] == "replaced"

    # 审计链按动作类型齐全，且都指回同一个会话
    counts = audit.count_by_action()
    assert counts["state.change_proposed"] == 1
    assert counts["state.confirmed"] == 1
    rows = audit.list_by_target(record["session_id"])
    assert [row["action"] for row in rows] == [
        "state.change_proposed",
        "state.confirmed",
    ]
    assert rows[0]["actor"].startswith("service:")
    assert rows[1]["actor"] == "nurse:N-EXAMPLE-1"
    assert rows[1]["from_state"] == "needs_change"
    assert rows[1]["to_state"] == "replaced"
    assert all(row["outcome"] == "accepted" for row in rows)
    # 只追加：audit_id 单调递增，时间也是
    ids = [row["audit_id"] for row in rows]
    assert ids == sorted(ids)


def test_replaced_requires_a_nurse(conn):
    """没有确认人就推不到 replaced（没有这只手就没有换液事件）。"""
    sessions = SessionRepository(conn)
    record = _near_empty_session(conn)
    sessions.apply_state_transition(record["session_id"], "needs_change")

    with pytest.raises(ApiError) as excinfo:
        confirm_replacement(sessions, record["session_id"], operator_id="")
    assert excinfo.value.error_code == "INGEST_INVALID"
    assert excinfo.value.issues[0].field == "operator_id"
    assert sessions.get(record["session_id"])["state"] == "needs_change", "被拒的确认不该改状态"


def test_confirmation_payload_carries_idempotency_key(conn):
    """确认后给出的出站载荷带幂等键与护士 id（task 10/11 直接吃它）。"""
    sessions = SessionRepository(conn)
    record = _near_empty_session(conn)
    sessions.apply_state_transition(record["session_id"], "needs_change")
    confirmation = confirm_replacement(
        sessions, record["session_id"], operator_id="N-EXAMPLE-1", now="2026-10-22T11:05:00+08:00"
    )

    payload = confirmation.event_payload
    assert payload["client_event_id"] == f"infusion-{record['session_id']}-B01-replaced"
    assert payload["event_type"] == "infusion.bag.changed"
    assert payload["bag_seq"] == "B01"
    assert payload["operator_id"] == "N-EXAMPLE-1"
    assert payload["occurred_at"] == "2026-10-22T11:05:00+08:00"
    assert payload["trigger_reason"] == "nurse_confirmed"
    # 载荷里不带患者可识别明细：只有床号、袋序、数量与事件
    assert set(payload) <= {
        "client_event_id", "event_type", "session_id", "rack_id", "bed_no",
        "infusion_order_no", "bag_seq", "fill_ratio", "estimated_volume_ml",
        "eta_minutes", "trigger_reason", "decision_source", "operator_id", "occurred_at",
    }


def test_needs_change_alone_writes_no_outbound_event(conn):
    """服务算出的 needs_change 不产生任何出站事件（未确认永不出站）。"""
    sessions = SessionRepository(conn)
    record = _near_empty_session(conn)
    apply_decision(sessions, evaluate_change(record))

    events = conn.execute("SELECT COUNT(*) AS n FROM infusion_event").fetchone()["n"]
    assert events == 0, "未确认的点名被写进了出站事件表"


# ── 2. 非法迁移 409 ───────────────────────────────────────────────────────


def test_illegal_transition_is_rejected_and_audited(conn):
    """monitoring 直跳 replaced → 409，且留一行 rejected 审计。"""
    sessions = SessionRepository(conn)
    audit = AuditChain(conn)
    record = _seed(conn)

    with pytest.raises(IllegalTransition) as excinfo:
        confirm_replacement(
            sessions, record["session_id"], operator_id="N-EXAMPLE-1",
            audit=audit, request_id="req-illegal-1",
        )
    assert excinfo.value.status_code == 409
    assert excinfo.value.error_code == "STATE_CONFLICT"
    assert "monitoring" in excinfo.value.message
    assert sessions.get(record["session_id"])["state"] == "monitoring", "非法迁移改了状态"

    rows = audit.list_by_target(record["session_id"])
    assert len(rows) == 1
    assert rows[0]["action"] == "state.transition_rejected"
    assert rows[0]["outcome"] == "rejected"
    assert (rows[0]["from_state"], rows[0]["to_state"]) == ("monitoring", "replaced")
    assert rows[0]["detail"]["reason"] == "illegal_transition"


def test_replaced_is_terminal(conn):
    """replaced 是终态：任何回退都被拒。"""
    sessions = SessionRepository(conn)
    record = _near_empty_session(conn)
    sessions.apply_state_transition(record["session_id"], "needs_change")
    confirm_replacement(sessions, record["session_id"], operator_id="N-EXAMPLE-1")

    for target in ("monitoring", "needs_change", "replaced"):
        with pytest.raises(IllegalTransition):
            check_transition("replaced", target)


def test_self_transition_is_rejected(conn):
    """自环也是非法迁移：重复确认不是「再确认一次」，而是重复事件。"""
    assert ALLOWED_TRANSITIONS["replaced"] == frozenset()
    with pytest.raises(IllegalTransition):
        check_transition("monitoring", "monitoring")
    with pytest.raises(IllegalTransition):
        check_transition("needs_change", "needs_change")


def test_illegal_transition_maps_to_409_over_http(store_client):
    """HTTP 侧：已换袋的会话再判定 → 409，detail 是统一的错误契约。"""
    client, conn = store_client
    from tests.test_reanchor import TWO_BAG_SLOTS, _descending_scene, _upload

    upload = _upload(
        client, _descending_scene(16), shift_key="2026-10-22-AM", slots=TWO_BAG_SLOTS
    )
    bag = upload.json()["bags"][0]
    sessions = SessionRepository(conn)
    sessions.apply_state_transition(bag["session_id"], "needs_change")
    confirm_replacement(sessions, bag["session_id"], operator_id="N-EXAMPLE-1")

    response = client.post(DECISION_URL, json={"session_id": bag["session_id"]})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error_code"] == "STATE_CONFLICT"
    assert detail["issues"][0]["code"] == "illegal_transition"
    assert "replaced" in detail["message"]

    # 被拒的判定同样留痕（飞检查的是「系统是不是在不该动的时候动了」）
    rows = [
        row
        for row in AuditChain(conn).list_by_target(bag["session_id"])
        if row["action"] == "state.transition_rejected"
    ]
    assert len(rows) == 1
    assert rows[0]["outcome"] == "rejected"


def test_decision_endpoint_reports_no_change_for_a_healthy_bag(store_client):
    """健康的一袋：判定不触发，但把判定明细如实回给护士端（可核对为什么没点名）。"""
    client, conn = store_client
    from tests.test_reanchor import TWO_BAG_SLOTS, _upload
    from tests.support import default_rack_scene

    upload = _upload(
        client, default_rack_scene(16), shift_key="2026-10-22-AM", slots=TWO_BAG_SLOTS
    )
    bag = upload.json()["bags"][0]
    response = client.post(DECISION_URL, json={"session_id": bag["session_id"]})

    assert response.status_code == 200
    result = response.json()
    assert result["changed"] is False
    assert result["state"] == "monitoring"
    assert "trigger_reason" not in result
    assert result["decision"]["consecutive_frames"] == 0
    assert SessionRepository(conn).get(bag["session_id"])["state"] == "monitoring"


def test_decision_endpoint_rejects_unknown_session(store_client):
    client, _ = store_client
    response = client.post(DECISION_URL, json={"session_id": "sess_nope"})
    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "NOT_FOUND"


# ── 3. needs_change 的双条件 ──────────────────────────────────────────────


def test_single_frame_spike_does_not_trigger(conn):
    """单帧毛刺（反光/晃一下）不触发：连续帧不够就是不够。"""
    sessions = SessionRepository(conn)
    row = _seed(conn)

    # 毛刺夹在中间：末尾液面正常，连续计数为 0
    sessions.update_area_series(row["session_id"], [0.40, 0.02, 0.38, 0.36])
    assert evaluate_change(sessions.get(row["session_id"])).consecutive_frames == 0

    # 毛刺正好落在最后一帧：只有 1 帧，仍不够 3 帧
    sessions.update_area_series(row["session_id"], [0.40, 0.38, 0.36, 0.02])
    decision = evaluate_change(sessions.get(row["session_id"]))
    assert decision.consecutive_frames == 1
    assert decision.changed is False
    assert "只有 1 帧低于阈值" in decision.reason_detail


def test_low_level_but_enough_volume_does_not_trigger(conn):
    """液面低了但余量还够（未标定、无滴速）→ 不点名，只压住时点。"""
    sessions = SessionRepository(conn)
    row = _seed(conn)
    sessions.update_area_series(row["session_id"], [0.12, 0.11, 0.10, 0.09])
    decision = evaluate_change(sessions.get(row["session_id"]))
    assert decision.changed is False
    assert decision.consecutive_frames >= 3
    assert decision.margin_kind == "none"
    assert "紧急线" in decision.reason_detail


def test_eta_window_triggers_with_volume_margin(conn):
    """条件 B 走 ETA：预计撑不到窗口内即点名，触发码是 eta_within_window。"""
    sessions = SessionRepository(conn)
    row = _seed(conn)
    sessions.update_area_series(row["session_id"], [0.12, 0.11, 0.10, 0.09])
    sessions.update_calibration(
        row["session_id"],
        {
            "fill_ratio": 0.09,
            "estimated_volume_ml": 90.0,     # 毫升还高于 30 ml 余量线
            "eta_minutes": 18,               # 但只剩 18 分钟
            "eta_source": "order_text",
            "calibration_state": "calibrated",
            "calibration_version": 1,
            "pickup_suppressed": False,
        },
    )
    decision = evaluate_change(sessions.get(row["session_id"]))
    assert decision.changed is True
    assert decision.trigger_reason == "eta_within_window"
    assert decision.margin_kind == "eta"
    assert decision.margin_value == 18


def test_volume_margin_triggers(conn):
    """条件 B 走毫升线：剩余 12 ml（ETA 还早）也点名。"""
    sessions = SessionRepository(conn)
    row = _seed(conn)
    sessions.update_area_series(row["session_id"], [0.11, 0.10, 0.09, 0.08])
    sessions.update_calibration(
        row["session_id"],
        {
            "fill_ratio": 0.08,
            "estimated_volume_ml": 12.0,
            "eta_minutes": 240,
            "eta_source": "order_text",
            "calibration_state": "calibrated",
            "calibration_version": 1,
            "pickup_suppressed": False,
        },
    )
    decision = evaluate_change(sessions.get(row["session_id"]))
    assert decision.changed is True
    assert decision.trigger_reason == "volume_below_margin"
    assert decision.margin_kind == "volume"
    assert decision.margin_value == 12.0


def test_gated_bag_is_not_named_unless_near_empty(conn):
    """标定门禁压住时不点名；但近空是直接观测，仍可点名（不给编出来的毫升数催人）。"""
    sessions = SessionRepository(conn)
    gated = _seed(conn, bag_seq="B01", slot_index=0)
    sessions.update_area_series(gated["session_id"], [0.10, 0.09, 0.08, 0.07])
    sessions.update_calibration(
        gated["session_id"],
        {"pickup_suppressed": True, "suppression_reason": "uncalibrated",
         "calibration_state": "uncalibrated"},
    )
    decision = evaluate_change(sessions.get(gated["session_id"]))
    assert decision.changed is False
    assert decision.suppression_reason == "uncalibrated"
    assert "标定门禁" in decision.reason_detail

    near_empty = _seed(conn, bag_seq="B02", slot_index=1)
    sessions.update_area_series(near_empty["session_id"], [0.05, 0.04, 0.03, 0.02])
    sessions.update_calibration(
        near_empty["session_id"],
        {"pickup_suppressed": True, "suppression_reason": "uncalibrated",
         "calibration_state": "uncalibrated"},
    )
    decision = evaluate_change(sessions.get(near_empty["session_id"]))
    assert decision.changed is True
    assert decision.trigger_reason == "near_empty_observed"


def test_reupload_does_not_silently_reset_a_pending_pickup(conn):
    """重传不重置业务状态：刚被点名要换袋，下一轮又拍一次不能把点名抹掉。"""
    sessions = SessionRepository(conn)
    row = _near_empty_session(conn)
    sessions.apply_state_transition(row["session_id"], "needs_change")

    again = sessions.upsert_session(_session_payload(frame_count=20))
    assert again["state"] == "needs_change"
    assert again["session_version"] == 2


def test_decision_is_idempotent_once_already_named(conn):
    """已处于 needs_change 时再判定：changed=false（不重复点名、不重复审计）。"""
    sessions = SessionRepository(conn)
    audit = AuditChain(conn)
    record = _near_empty_session(conn)
    apply_decision(sessions, evaluate_change(record), audit=audit, request_id="req-a")

    again, changed = apply_decision(
        sessions, evaluate_change(sessions.get(record["session_id"])),
        audit=audit, request_id="req-b",
    )
    assert changed is False
    assert again["state"] == "needs_change"
    assert audit.count_by_action()["state.change_proposed"] == 1


def test_occluded_frames_do_not_break_the_streak(conn):
    """被遮挡的帧（null）不算「低」也不打断连续计数——液面在遮挡前后都是低的。

    判据是**可见观测**构成的序列：一次两秒的遮挡不该把已经攒够的证据清零，
    否则护士在遮挡频繁的病房里永远等不到点名。
    """
    sessions = SessionRepository(conn)
    row = _seed(conn)
    sessions.update_area_series(row["session_id"], [0.08, None, 0.06, None, 0.04])
    decision = evaluate_change(sessions.get(row["session_id"]))
    assert decision.consecutive_frames == 3
    assert decision.latest_fill_ratio == 0.04, "末次观测要取最后那一帧，不是遮挡前的老读数"

    # 但遮挡后液面回升会立刻断掉连续计数（null 不会被当成低帧凑数）
    sessions.update_area_series(row["session_id"], [0.08, None, 0.06, None, 0.40])
    assert evaluate_change(sessions.get(row["session_id"])).consecutive_frames == 0


def test_no_observation_is_not_reported_as_zero(conn):
    """整段看不见 → 不点名，不报 0 毫升（这是两件完全不同的事）。"""
    sessions = SessionRepository(conn)
    row = _seed(conn)
    sessions.update_area_series(row["session_id"], [None, None, None, None])
    decision = evaluate_change(sessions.get(row["session_id"]))
    assert decision.changed is False
    assert decision.latest_fill_ratio is None
    assert "没有可见帧" in decision.reason_detail


def test_audit_chain_rejects_unknown_action(conn):
    """审计动作码受白名单约束（后续任务只允许扩充，不允许拼错码混进去）。"""
    with pytest.raises(ValueError):
        AuditChain(conn).append(
            request_id="r", actor="nurse:N", action="state.maybe", target="sess-1",
            outcome="accepted",
        )