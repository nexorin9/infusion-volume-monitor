"""换液点名载荷与确认接口的验收测试（task 31）。

四组断言，对应任务书里的四条硬要求：

1. **排序**——点名清单按 ETA 升序 → 床号 → 袋序，取不到滴速的袋**排在最后而不是
   最前**（按 0 分钟混进去等于给一个没滴速的袋编了时点）；只有 ``needs_change`` 进
   清单；名次 1 起且跟着顺序走。
2. **字段完整**——清单每一行都带齐 ``bag_seq / trigger_reason / eta_minutes /
   confidence / bed_no`` 以及幂等键与确认入参；``writeback_queued`` 恒假。
3. **降级文案**——未标定袋型时逐字给出「依据面积比，未标定袋型」并**不出现**
   毫升数；没测到时置信度**不出现**而不是 0。
4. **重复确认幂等**——同一个确认点两次只产生一条事件、只外发一次，第二次返回
   ``idempotent_hit=true`` 且事件 ID 与首次完全相同；终端自编幂等键被 422 拒收。

另有一条对照：**未确认的 ``needs_change`` 在点名清单里看得见，但绝不出站**——查一次
清单之后出站替身收到的请求数必须是 0、事件表必须是 0 行。
"""

from __future__ import annotations

import pytest

from infusion_monitor.api.pickup_payload import (
    NEEDS_CHANGE_STATE,
    build_pickup_item,
    build_pickup_list,
    observation_confidence,
    pickup_sort_key,
    read_pickup_list,
)
from infusion_monitor.api.transition import expected_client_event_id
from infusion_monitor.identity.state_machine import build_client_event_id
from infusion_monitor.store.repositories import SessionRepository
from tests.support import NurseExecStub

OPERATOR = "N-EXAMPLE-31"
TOKEN = "task31-token-not-real"

#: 点名清单承诺给到护士端的每一项。少一项，护士端就得自己猜——所以钉死。
REQUIRED_ITEM_KEYS = {
    "bag_seq",
    "session_id",
    "state",
    "bed_no",
    "trigger_reason",
    "eta_minutes",
    "confidence",
    "confidence_source",
    "evidence_note",
    "client_event_id",
    "writeback_queued",
    "writeback_hold_reason",
    "confirm_hint",
    "rank",
}


def session_row(**overrides):
    """一条点名中的会话行（纯字典，够 :func:`build_pickup_item` 用）。"""
    row = {
        "session_id": "sess_example_31",
        "rack_id": "R-EXAMPLE-31",
        "bed_no": "EXAMPLE-BED-31",
        "infusion_order_no": "EXAMPLE-ORDER-0031",
        "shift_key": "2026-10-22-AM",
        "bag_seq": "B01",
        "slot_index": 0,
        "bag_type_code": "TYPE-A",
        "frames_dir": "/tmp/frames",
        "frame_count": 16,
        "captured_at": "2026-10-22T09:41:00+08:00",
        "area_ratio_series": [0.12, 0.11, 0.10, 0.09] * 4,
        "fill_ratio": 0.09,
        "estimated_volume_ml": 52.2,
        "eta_minutes": 26,
        "eta_source": "order_text",
        "calibration_state": "calibrated",
        "calibration_version": 2,
        "volume_confidence": 0.87,
        "trigger_reason": "eta_within_window",
        "state": NEEDS_CHANGE_STATE,
        "updated_at": "2026-10-22T09:41:30+00:00",
    }
    row.update(overrides)
    return row


# ── 1. 排序 ──────────────────────────────────────────────────────────────────


class TestOrdering:
    def test_orders_by_eta_then_bed_then_bag(self):
        rows = [
            session_row(session_id="s-a", bag_seq="B02", bed_no="BED-3", eta_minutes=40),
            session_row(session_id="s-b", bag_seq="B01", bed_no="BED-3", eta_minutes=20),
            session_row(session_id="s-c", bag_seq="B01", bed_no="BED-2", eta_minutes=20),
            session_row(session_id="s-d", bag_seq="B03", bed_no="BED-1", eta_minutes=90),
        ]
        listing = build_pickup_list(rows, generated_at="2026-10-22T10:00:00+08:00")
        assert [item.bag_seq + "@" + str(item.bed_no) for item in listing.items] == [
            "B01@BED-2",
            "B01@BED-3",
            "B02@BED-3",
            "B03@BED-1",
        ]

    def test_bags_without_eta_sort_last_not_first(self):
        """没有滴速就没有「几点见底」——排到最前等于编了一个时点。"""
        rows = [
            session_row(session_id="s-a", bag_seq="B01", eta_minutes=None),
            session_row(session_id="s-b", bag_seq="B02", eta_minutes=5),
        ]
        listing = build_pickup_list(rows)
        assert [item.bag_seq for item in listing.items] == ["B02", "B01"]
        assert listing.items[-1].eta_minutes is None

    def test_sort_key_is_stable_and_total(self):
        """同 ETA、同床、同袋序的两条会话（两架挂同一床）顺序也不能随入参顺序跳。"""
        rows = [
            session_row(session_id="s-a", bag_seq="B01", bed_no="BED-1", eta_minutes=20),
            session_row(session_id="s-b", bag_seq="B01", bed_no="BED-1", eta_minutes=20),
        ]
        first = [item.session_id for item in build_pickup_list(rows).items]
        second = [item.session_id for item in build_pickup_list(list(reversed(rows))).items]
        # 排序键是全序（末位含会话 ID），倒着喂进去结果必须一样：护理执行台按清单逐条
        # 核对，顺序跳一次护士就会怀疑自己漏了哪一袋。
        assert first == second == ["s-a", "s-b"]
        assert pickup_sort_key(build_pickup_item(rows[0])) < pickup_sort_key(
            build_pickup_item(rows[1])
        )

    def test_only_needs_change_bags_are_listed(self):
        rows = [
            session_row(session_id="s-a", bag_seq="B01", state=NEEDS_CHANGE_STATE),
            session_row(session_id="s-b", bag_seq="B02", state="monitoring"),
            session_row(session_id="s-c", bag_seq="B03", state="replaced"),
        ]
        listing = build_pickup_list(rows)
        assert [item.bag_seq for item in listing.items] == ["B01"]
        assert listing.count == 1

    def test_rank_is_one_based_and_follows_the_order(self):
        rows = [
            session_row(session_id="s-a", bag_seq="B01", bed_no="BED-9", eta_minutes=90),
            session_row(session_id="s-b", bag_seq="B02", bed_no="BED-1", eta_minutes=10),
        ]
        listing = build_pickup_list(rows)
        assert [(item.bag_seq, item.rank) for item in listing.items] == [("B02", 1), ("B01", 2)]

    def test_empty_listing_is_a_valid_answer(self):
        listing = build_pickup_list([], generated_at="2026-10-22T10:00:00+08:00")
        assert listing.count == 0
        assert listing.to_json()["items"] == []
        assert listing.to_json()["writeback_queued_count"] == 0

    def test_read_pickup_list_only_touches_the_needed_change_state(self, conn):
        """读路径只按状态取，不全表扫——点名清单不该把 monitoring 的会话也捞出来。"""
        sessions = SessionRepository(conn)
        _seed(sessions, session_id="sess_rd_named", bag_seq="B01")
        _seed(sessions, session_id="sess_rd_mon", bag_seq="B02", state="monitoring")
        _seed(sessions, session_id="sess_rd_rep", bag_seq="B03", state="replaced")

        asked: list[str] = []

        class _Spy:
            def list_by_state(self, state, limit=100):
                asked.append(state)
                return [r for r in (sessions.get(r) for r in
                        ("sess_rd_named", "sess_rd_mon", "sess_rd_rep")) if r["state"] == state][:limit]

        listing = read_pickup_list(_Spy())
        assert asked == [NEEDS_CHANGE_STATE]
        assert [item.bag_seq for item in listing.items] == ["B01"]


# ── 2. 字段完整 ──────────────────────────────────────────────────────────────


class TestItemFields:
    def test_item_carries_every_promised_field(self):
        payload = build_pickup_item(session_row()).to_json()
        assert REQUIRED_ITEM_KEYS <= set(payload), sorted(REQUIRED_ITEM_KEYS - set(payload))
        assert payload["bag_seq"] == "B01"
        assert payload["bed_no"] == "EXAMPLE-BED-31"
        assert payload["trigger_reason"] == "eta_within_window"
        assert payload["eta_minutes"] == 26
        assert payload["confidence"] == 0.87

    def test_client_event_id_follows_the_fixed_rule(self):
        """幂等键由 (session_id, bag_seq) 固定生成——清单和确认端点必须算出同一个。"""
        row = session_row(session_id="sess_x", bag_seq="B07")
        assert build_pickup_item(row).client_event_id == build_client_event_id("sess_x", "B07")
        assert build_pickup_item(row).client_event_id == expected_client_event_id("sess_x", "B07")

    def test_confirm_hint_is_exactly_what_the_confirm_endpoint_wants(self):
        item = build_pickup_item(session_row(session_id="sess_y", bag_seq="B02"))
        hint = item.to_json()["confirm_hint"]
        assert set(hint) == {"session_id", "bag_seq", "client_event_id"}
        assert hint["session_id"] == "sess_y"
        assert hint["bag_seq"] == "B02"
        assert hint["client_event_id"] == expected_client_event_id("sess_y", "B02")

    def test_named_bag_is_never_marked_as_queued_for_writeback(self):
        payload = build_pickup_item(session_row()).to_json()
        assert payload["writeback_queued"] is False
        assert "未确认" in payload["writeback_hold_reason"]

    def test_volume_confidence_wins_when_the_bag_is_calibrated(self):
        item = build_pickup_item(session_row(volume_confidence=0.73))
        assert item.confidence == 0.73
        assert item.confidence_source == "volume"

    def test_observation_confidence_is_used_when_there_is_no_millilitre(self):
        item = build_pickup_item(
            session_row(
                calibration_state="uncalibrated",
                calibration_version=None,
                estimated_volume_ml=None,
                volume_confidence=None,
            )
        )
        assert item.confidence_source == "observation"
        assert item.confidence is not None and 0.0 < item.confidence <= 1.0
        assert item.confidence_factors["visible_frames"] == 16

    def test_eta_is_absent_when_no_rate_was_resolved(self):
        item = build_pickup_item(session_row(eta_minutes=None, eta_source="unavailable"))
        assert item.eta_minutes is None
        assert "未取到滴速" in item.evidence_note


# ── 3. 降级文案 ──────────────────────────────────────────────────────────────


class TestDegradedText:
    def test_uncalibrated_bag_says_area_ratio_not_a_fake_millilitre(self):
        """未标定袋型：给依据、给点名，**不给毫升数**，也不让护士以为那数不存在。"""
        item = build_pickup_item(
            session_row(
                calibration_state="uncalibrated",
                calibration_version=None,
                estimated_volume_ml=None,
                volume_confidence=None,
            )
        )
        assert "依据面积比，未标定袋型" in item.evidence_note
        assert "TYPE-A" in item.evidence_note
        assert item.estimated_volume_ml is None
        assert item.evidence_basis == ("area_ratio",)

    def test_unregistered_bag_type_still_names_the_fallback(self):
        item = build_pickup_item(
            session_row(
                bag_type_code=None,
                calibration_state="uncalibrated",
                estimated_volume_ml=None,
                volume_confidence=None,
            )
        )
        assert "依据面积比，未标定袋型（未登记袋型）" in item.evidence_note

    def test_calibrated_bag_names_the_version_it_used(self):
        item = build_pickup_item(session_row())
        assert "TYPE-A" in item.evidence_note and "2 版标定" in item.evidence_note
        assert "calibration" in item.evidence_basis

    def test_gated_bag_explains_why_the_millilitre_is_withheld(self):
        item = build_pickup_item(
            session_row(
                calibration_state="high_occlusion",
                estimated_volume_ml=None,
                volume_confidence=None,
                suppression_reason="high_occlusion",
            )
        )
        assert item.estimated_volume_ml is None
        assert "面积比" in item.evidence_note
        assert "high_occlusion" in item.evidence_note

    def test_confidence_is_absent_rather_than_zero_when_nothing_was_observed(self):
        confidence, factors = observation_confidence(
            session_row(area_ratio_series=[None, None, None, None], volume_confidence=None)
        )
        assert confidence is None, "整段看不见时给 0 等于说『测过了但不可信』"
        assert factors["reason"] == "no_observation"

    def test_occlusion_lowers_observation_confidence(self):
        clear, _ = observation_confidence(
            session_row(area_ratio_series=[0.1] * 16, volume_confidence=None)
        )
        occluded, factors = observation_confidence(
            session_row(
                area_ratio_series=[0.1] * 8 + [None] * 8, volume_confidence=None
            )
        )
        assert occluded < clear
        assert factors["occlusion_ratio"] == 0.5

    def test_low_fill_costs_confidence_because_the_band_is_thinnest_there(self):
        high, _ = observation_confidence(
            session_row(area_ratio_series=[0.4] * 16, volume_confidence=None)
        )
        low, factors = observation_confidence(
            session_row(area_ratio_series=[0.02] * 16, volume_confidence=None)
        )
        assert low < high
        assert factors["low_fill_penalty"] > 0


# ── 4 & 5. 接口层：点名清单只读 + 确认幂等（真实 HTTP + 真实替身对端） ────────


def _seed(sessions: SessionRepository, **overrides):
    """落一条会话并推到 ``needs_change``（点名后状态）。"""
    row = session_row(**overrides)
    row.pop("updated_at", None)
    stored = sessions.upsert_session(row)
    sessions.update_area_series(
        stored["session_id"],
        row["area_ratio_series"],
        occlusion_flags=[False] * len(row["area_ratio_series"]),
    )
    if row.get("state") != NEEDS_CHANGE_STATE:
        return sessions.get(stored["session_id"])
    return sessions.apply_state_transition(
        stored["session_id"], NEEDS_CHANGE_STATE, trigger_reason="eta_within_window"
    )


@pytest.fixture
def live(store_client):
    """TestClient + 观察连接 + 真 socket 的出站替身（出站客户端是正式实现）。"""
    client, observer = store_client
    from infusion_monitor.adapters.nurse_exec import NurseExecClient

    with NurseExecStub(status=201) as stub:
        original = client.app.state.nurse_exec
        replacement = NurseExecClient(
            base_url=f"{stub.base_url}/api/v1/infusion/events", api_token=TOKEN
        )
        client.app.state.nurse_exec = replacement
        try:
            yield client, observer, stub
        finally:
            replacement.close()
            client.app.state.nurse_exec = original


def _confirm_body(hint: dict, **overrides):
    body = {
        "session_id": hint["session_id"],
        "bag_seq": hint["bag_seq"],
        "operator_id": OPERATOR,
        "client_event_id": hint["client_event_id"],
    }
    body.update(overrides)
    return body


class TestPickupsEndpointIsReadOnly:
    def test_named_bags_are_visible_in_the_listing(self, live):
        client, observer, _stub = live
        _seed(SessionRepository(observer), bag_seq="B01", bed_no="BED-1")
        payload = client.get("/api/v1/infusion/pickups").json()
        assert payload["count"] == 1
        assert payload["items"][0]["bag_seq"] == "B01"
        assert payload["items"][0]["state"] == NEEDS_CHANGE_STATE
        assert payload["without_ml_count"] == 0

    def test_listing_a_pickup_sends_nothing_and_queues_nothing(self, live):
        """点名清单只读：查一次之后，出站替身收到的请求数必须是 0、事件表 0 行。"""
        client, observer, stub = live
        for index in range(3):
            _seed(
                SessionRepository(observer),
                session_id=f"sess_ro_{index}",
                bag_seq=f"B0{index + 1}",
                bed_no=f"BED-{index + 1}",
            )
        for _ in range(3):
            assert client.get("/api/v1/infusion/pickups").status_code == 200
        assert stub.call_count == 0, "点名清单这一层绝不能触发任何出站请求"
        assert observer.execute("SELECT COUNT(*) FROM infusion_event").fetchone()[0] == 0

    def test_monitoring_and_replaced_bags_stay_out_of_the_listing(self, live):
        client, observer, _stub = live
        sessions = SessionRepository(observer)
        _seed(sessions, session_id="sess_mon", bag_seq="B01", state="monitoring")
        _seed(sessions, session_id="sess_named", bag_seq="B02")
        payload = client.get("/api/v1/infusion/pickups").json()
        assert [item["bag_seq"] for item in payload["items"]] == ["B02"]

    def test_rack_filter_narrows_the_listing(self, live):
        client, observer, _stub = live
        sessions = SessionRepository(observer)
        _seed(sessions, session_id="sess_r1", bag_seq="B01", rack_id="R-EXAMPLE-31")
        _seed(sessions, session_id="sess_r2", bag_seq="B02", rack_id="R-EXAMPLE-32")
        payload = client.get("/api/v1/infusion/pickups?rack_id=R-EXAMPLE-32").json()
        assert [item["bag_seq"] for item in payload["items"]] == ["B02"]

    def test_uncalibrated_bag_is_listed_with_the_degraded_note_and_no_millilitre(self, live):
        client, observer, _stub = live
        _seed(
            SessionRepository(observer),
            calibration_state="uncalibrated",
            calibration_version=None,
            estimated_volume_ml=None,
            volume_confidence=None,
        )
        item = client.get("/api/v1/infusion/pickups").json()["items"][0]
        assert "依据面积比，未标定袋型" in item["evidence_note"]
        assert "estimated_volume_ml" not in item, "无标定时毫升数字段必须整体不出现"
        assert item["confidence_source"] == "observation"

    def test_oversized_limit_is_refused_with_a_next_step(self, live):
        client, _observer, _stub = live
        response = client.get("/api/v1/infusion/pickups?limit=101")
        assert response.status_code == 422
        issue = response.json()["detail"]["issues"][0]
        assert issue["code"] == "limit_too_large"
        assert "rack_id" in issue["suggestion"]


class TestTransitionIdempotency:
    def test_confirm_hints_from_the_listing_are_accepted_verbatim(self, live):
        client, observer, stub = live
        _seed(SessionRepository(observer), bag_seq="B01")
        hint = client.get("/api/v1/infusion/pickups").json()["items"][0]["confirm_hint"]

        response = client.post("/api/v1/infusion/transition", json=_confirm_body(hint))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["created"] is True
        assert body["idempotent_hit"] is False
        assert body["state"] == "replaced"
        assert body["client_event_id"] == hint["client_event_id"]
        assert body["writeback"]["status"] == "sent"
        assert stub.call_count == 1
        assert stub.last().authorization == f"Bearer {TOKEN}"
        assert stub.last().json()["bag_seq"] == "B01"

    def test_replaying_the_same_confirmation_changes_nothing(self, live):
        """病区 Wi-Fi 抖动会让同一个点击重发好几次：第二次必须悄无声息地拿到同一个结果。"""
        client, observer, stub = live
        _seed(SessionRepository(observer), bag_seq="B01")
        hint = client.get("/api/v1/infusion/pickups").json()["items"][0]["confirm_hint"]
        body = _confirm_body(hint)

        first = client.post("/api/v1/infusion/transition", json=body).json()
        second = client.post("/api/v1/infusion/transition", json=body).json()

        assert second["created"] is False
        assert second["idempotent_hit"] is True
        assert second["event_id"] == first["event_id"]
        assert second["client_event_id"] == first["client_event_id"]
        assert second["confirmed_at"] == first["confirmed_at"]
        assert stub.call_count == 1, "重放不得再发一次出站请求"
        assert observer.execute("SELECT COUNT(*) FROM infusion_event").fetchone()[0] == 1

    def test_a_terminal_bag_cannot_be_confirmed_twice_under_another_key(self, live):
        """幂等键按 (会话, 袋序) 定：换一袋就是一个新事件，换回旧袋就是重放。"""
        client, observer, stub = live
        _seed(SessionRepository(observer), bag_seq="B01")
        hint = client.get("/api/v1/infusion/pickups").json()["items"][0]["confirm_hint"]
        client.post("/api/v1/infusion/transition", json=_confirm_body(hint))
        again = client.post(
            "/api/v1/infusion/transition",
            json=_confirm_body(hint, client_event_id="infusion-made-up-by-terminal"),
        )
        assert again.status_code == 422
        assert again.json()["detail"]["issues"][0]["code"] == "client_event_id_mismatch"
        assert observer.execute("SELECT COUNT(*) FROM infusion_event").fetchone()[0] == 1
        assert stub.call_count == 1

    def test_client_event_id_may_be_omitted_entirely(self, live):
        """终端不填键时服务端按规则生成——不因为少填一个字段就让护士换不了液。"""
        client, observer, stub = live
        row = _seed(SessionRepository(observer), bag_seq="B01")
        body = {"session_id": row["session_id"], "bag_seq": "B01", "operator_id": OPERATOR}
        payload = client.post("/api/v1/infusion/transition", json=body).json()
        assert payload["client_event_id"] == expected_client_event_id(row["session_id"], "B01")
        assert stub.call_count == 1

    def test_wrong_bag_seq_is_refused_as_a_cross_bag_write(self, live):
        client, observer, stub = live
        row = _seed(SessionRepository(observer), bag_seq="B01")
        response = client.post(
            "/api/v1/infusion/transition",
            json={"session_id": row["session_id"], "bag_seq": "B02", "operator_id": OPERATOR},
        )
        assert response.status_code == 403
        assert response.json()["detail"]["issues"][0]["code"] == "bag_not_in_session"
        assert observer.execute("SELECT COUNT(*) FROM infusion_event").fetchone()[0] == 0
        assert stub.call_count == 0

    def test_illegal_transition_is_409(self, live):
        """monitoring 直跳 replaced：还没到换液点就报已换，直接 409 并留审计。"""
        client, observer, stub = live
        row = _seed(SessionRepository(observer), bag_seq="B01", state="monitoring")
        response = client.post(
            "/api/v1/infusion/transition",
            json={"session_id": row["session_id"], "bag_seq": "B01", "operator_id": OPERATOR},
        )
        assert response.status_code == 409
        assert response.json()["detail"]["issues"][0]["code"] == "illegal_transition"
        assert observer.execute("SELECT COUNT(*) FROM infusion_event").fetchone()[0] == 0
        assert stub.call_count == 0

    def test_missing_operator_is_refused(self, live):
        client, observer, _stub = live
        row = _seed(SessionRepository(observer), bag_seq="B01")
        response = client.post(
            "/api/v1/infusion/transition",
            json={"session_id": row["session_id"], "bag_seq": "B01"},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["issues"][0]["field"] == "operator_id"
        assert observer.execute("SELECT COUNT(*) FROM infusion_event").fetchone()[0] == 0

    def test_pickup_evidence_survives_the_confirmation(self, live):
        """确认会把 trigger_reason 覆盖成 nurse_confirmed——点名依据要在这里留一份。"""
        client, observer, _stub = live
        _seed(SessionRepository(observer), bag_seq="B01", eta_minutes=26)
        hint = client.get("/api/v1/infusion/pickups").json()["items"][0]["confirm_hint"]
        evidence = client.post(
            "/api/v1/infusion/transition", json=_confirm_body(hint)
        ).json()["pickup_evidence"]
        assert evidence["previous_state"] == NEEDS_CHANGE_STATE
        assert evidence["previous_trigger_reason"] == "eta_within_window"
        assert evidence["previous_eta_minutes"] == 26

    def test_writeback_failure_ends_in_pending_writeback_not_a_silent_drop(self, live):
        """对端 5xx：退避耗尽后置 pending_writeback，事件仍在库里等重放。"""
        client, observer, stub = live
        _seed(SessionRepository(observer), bag_seq="B01")
        stub.queue_status(500, 500, 500)
        hint = client.get("/api/v1/infusion/pickups").json()["items"][0]["confirm_hint"]
        payload = client.post("/api/v1/infusion/transition", json=_confirm_body(hint)).json()

        assert payload["created"] is True, "状态与事件都已落库，对端不可达不该把确认也判失败"
        assert payload["writeback"]["status"] == "pending_writeback"
        assert payload["writeback"]["ok"] is False
        assert payload["writeback"]["attempts"] == 3
        assert payload["writeback"]["next_retry_at"]
        rows = observer.execute(
            "SELECT writeback_status, attempt_count FROM infusion_event"
        ).fetchall()
        assert [dict(row) for row in rows] == [
            {"writeback_status": "pending_writeback", "attempt_count": 3}
        ]

    def test_openapi_publishes_both_endpoints(self, live):
        client, _observer, _stub = live
        spec = client.get("/openapi.json").json()
        assert spec["openapi"].startswith("3.1")
        assert "/api/v1/infusion/pickups" in spec["paths"]
        assert "/api/v1/infusion/transition" in spec["paths"]
        assert "TransitionIn" in spec["components"]["schemas"]
        assert "PickupListOut" in spec["components"]["schemas"]