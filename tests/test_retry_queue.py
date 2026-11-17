"""待写回队列：退避次数、欠账落库、重放幂等、重放后清理（task 37）。

**这个模块兑现的是一句承诺**：护士点过「已换液」，护理执行平台没收到时，
这笔账**不会消失**——它落在盘上、看得见、能重放，重放也不会写出第二条换液事件。

四组断言（与 task 37 的验收点一一对应）：

1. **退避次数**——指数表 1s / 4s / 16s；尝试预算 = 首次 + ≤2 次重试；
   预算用完即进死信并置 ``pending_writeback``；
2. **pending 落库**——欠账行带载荷快照、已尝试次数、下次可重试时刻、错误摘要；
   错误摘要**不含凭据**；
3. **重放幂等**——重放只投已存在的事件，对端拿到的幂等键与首次相同；
   已送达的重放**一次 HTTP 都不发**；
4. **重放成功后队列清理**——那一行从「欠账」改成「已结清」，待处理清单里不再出现，
   业务事件行数不变。

替身只出现在**测试数据层**：真实护理执行平台端点本轮未实连（见 progress.txt）。
"""

from __future__ import annotations

import json

import pytest

from infusion_monitor.adapters.nurse_exec import NurseExecClient
from infusion_monitor.adapters.retry_queue import (
    BACKOFF_SCHEDULE_SECONDS,
    MAX_ATTEMPTS,
    PENDING_STATUSES,
    RETRY_BUDGET,
    STATUS_DEAD_LETTER,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_SENT,
    WritebackQueue,
    backoff_seconds,
    retry_at,
)
from infusion_monitor.adapters.writeback import backoff_seconds as inline_backoff
from infusion_monitor.adapters.writeback import deliver_event
from infusion_monitor.store import (
    AuditChain,
    EventRepository,
    SessionRepository,
    WritebackQueueRepository,
)
from tests.support import NurseExecStub

TOKEN = "retry-queue-token-not-real"
OPERATOR = "N-EXAMPLE-37"
REPLAYER = "OPS-EXAMPLE-37"
REQUEST_ID = "req-retry-queue-0037"


class _RecordingSleeper:
    """把退避等待记下来而不真睡——测试不该为指数退避付出 5 秒墙钟。"""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


@pytest.fixture
def stub():
    with NurseExecStub(status=201) as running:
        yield running


@pytest.fixture
def client(stub):
    return NurseExecClient(base_url=f"{stub.base_url}/api/v1/infusion/events", api_token=TOKEN)


@pytest.fixture
def repos(conn):
    sessions = SessionRepository(conn)
    events = EventRepository(conn)
    audit = AuditChain(conn)
    queue = WritebackQueue(WritebackQueueRepository(conn), events)
    return sessions, events, audit, queue


def _seed_session(sessions: SessionRepository, *, state: str = "needs_change") -> dict:
    """一袋已点名的会话（``state`` 可切到 replaced 以过出站门禁）。"""
    payload = {
        "rack_id": "R-EXAMPLE-37",
        "bed_no": "EXAMPLE-BED-37",
        "infusion_order_no": "EXAMPLE-ORDER-0037",
        "shift_key": "2026-10-23-AM",
        "bag_seq": "B01",
        "slot_index": 0,
        "bag_type_code": "TYPE-A",
        "frames_dir": "/tmp/frames-37",
        "frame_count": 16,
        "captured_at": "2026-10-23T09:41:00+08:00",
        "fill_ratio": 0.18,
        "estimated_volume_ml": 104.4,
        "eta_minutes": 26,
        "state": "monitoring",
    }
    row = sessions.upsert_session(payload)
    if state == "needs_change":
        sessions.apply_state_transition(row["session_id"], "needs_change")
    elif state == "replaced":
        sessions.apply_state_transition(row["session_id"], "needs_change")
        sessions.apply_state_transition(row["session_id"], "replaced")
    return sessions.get(row["session_id"])


def _confirmed_event(sessions, events, *, operator_id: str = OPERATOR):
    """走真幂等入口确认换液，拿到已落库的出站事件行。

    会话先停在 ``needs_change``——``replaced`` 是终态，只能由这一次确认推进，
    不能先置好再确认（那才是非法迁移）。
    """
    from infusion_monitor.identity.idempotency import confirm_replacement_once

    record = _seed_session(sessions, state="needs_change")
    outcome = confirm_replacement_once(
        sessions, events, record["session_id"], bag_seq="B01", operator_id=operator_id
    )
    return record, events.get(str(outcome.event_id))


# ── 1. 退避次数 ──────────────────────────────────────────────────────────────


def test_backoff_table_is_one_four_sixteen_and_shared_with_the_writeback_path():
    """退避表是**同一张**：1s / 4s / 16s，且与出站内联退避同源。

    两处一旦各写一份，就会出现「表上写着 16s 可重试、实际 4s 就重试」的错位——
    而错位只在真断网时显形，那时没人有空查表。
    """
    assert BACKOFF_SCHEDULE_SECONDS == (1.0, 4.0, 16.0)
    assert backoff_seconds(1) == 1.0
    assert backoff_seconds(2) == 4.0
    assert backoff_seconds(3) == 16.0
    # 末项之后不再增长：再长的等待也等不来一个不同的答案。
    assert backoff_seconds(4) == 16.0
    assert backoff_seconds(99) == 16.0
    # 与出站内联退避同一条曲线（不是另一张长得像的表）。
    assert BACKOFF_SCHEDULE_SECONDS == tuple(inline_backoff(i) for i in range(3))


def test_attempt_budget_is_first_attempt_plus_two_retries():
    """预算：首次 + ≤2 次重试 = 3 次。用完就进死信，不再自动发。"""
    assert MAX_ATTEMPTS == 3
    assert RETRY_BUDGET == 2


def test_delivery_stops_at_the_budget_and_marks_the_event_pending_writeback(
    stub, client, repos
):
    """对端一直 5xx：首次 + 2 次重试后停手，事件置 pending_writeback。"""
    sessions, events, audit, queue = repos
    _, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(503)

    sleeper = _RecordingSleeper()
    outcome = deliver_event(
        sessions,
        events,
        client,
        event_row["event_id"],
        audit=audit,
        request_id=REQUEST_ID,
        sleeper=sleeper,
    )

    assert stub.call_count == 3, "退避预算用完后不该再发"
    assert outcome.attempts == 3
    assert outcome.attempt_kinds == ["http_5xx", "http_5xx", "http_5xx"]
    assert sleeper.delays == [1.0, 4.0], "退避间隔必须是指数的 1s / 4s"
    assert outcome.writeback_status == "pending_writeback"

    stored = events.get(event_row["event_id"])
    assert stored["writeback_status"] == "pending_writeback"
    assert stored["attempt_count"] == 3


def test_exhausted_budget_lands_the_debt_in_the_queue_as_dead_letter(
    stub, client, repos
):
    """预算耗尽 → 队列里出现一行死信，事件仍是 pending_writeback。"""
    sessions, events, audit, queue = repos
    _, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(503)

    deliver_event(
        sessions,
        events,
        client,
        event_row["event_id"],
        audit=audit,
        request_id=REQUEST_ID,
        sleeper=_RecordingSleeper(),
        queue=queue,
    )

    entry = queue.get(event_row["event_id"])
    assert entry is not None, "出站失败却没有落进待写回队列——这笔账会消失"
    assert entry.status == STATUS_DEAD_LETTER
    assert entry.dead_letter_reason == "retries_exhausted"
    assert entry.attempt_count == 3


def test_business_rejection_is_not_retried_and_is_queued_as_failed(stub, client, repos):
    """对端 4xx 是业务拒绝：同一份载荷再发只会再被拒一次，所以只发一次。"""
    sessions, events, audit, queue = repos
    _, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(400, body=json.dumps({"message": "输注单号不存在"}))

    sleeper = _RecordingSleeper()
    outcome = deliver_event(
        sessions,
        events,
        client,
        event_row["event_id"],
        audit=audit,
        request_id=REQUEST_ID,
        sleeper=sleeper,
        queue=queue,
    )

    assert stub.call_count == 1, "4xx 不该退避重试"
    assert sleeper.delays == []
    assert outcome.writeback_status == "failed"

    entry = queue.get(event_row["event_id"])
    assert entry is not None
    assert entry.status == STATUS_FAILED
    assert entry.dead_letter_reason == "business_rejected"


# ── 2. pending 落库 ──────────────────────────────────────────────────────────


def test_pending_entry_carries_payload_attempts_deadline_and_error(
    stub, client, repos
):
    """欠账行要能回答四个问题：发了什么、试了几次、什么时候能再试、错在哪。"""
    sessions, events, audit, queue = repos
    _, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(503)

    deliver_event(
        sessions,
        events,
        client,
        event_row["event_id"],
        audit=audit,
        request_id=REQUEST_ID,
        sleeper=_RecordingSleeper(),
        queue=queue,
    )

    row = queue.repository.get_by_event_id(event_row["event_id"])
    assert row is not None
    assert row["event_id"] == event_row["event_id"]
    assert row["client_event_id"] == event_row["client_event_id"]
    assert row["attempt_count"] == 3
    assert row["next_retry_at"], "欠账行没写下次可重试时刻"
    assert row["last_error"] and "503" in row["last_error"]
    assert row["status"] == STATUS_DEAD_LETTER
    assert row["payload"].get("bag_seq") == "B01"
    assert row["payload"].get("infusion_order_no") == "EXAMPLE-ORDER-0037"
    assert row["rack_id"] == "R-EXAMPLE-37"
    assert row["bed_no"] == "EXAMPLE-BED-37"


def test_pending_entry_never_carries_the_credential(stub, client, repos):
    """错误摘要会被导进飞检材料，凭据不该出现在里面。"""
    sessions, events, audit, queue = repos
    _, event_row = _confirmed_event(sessions, events)
    # 有些网关会把请求头原样回显——那正是 token 会顺着进库、进日志的路径。
    stub.set_default_status(503, body=json.dumps({"echo": f"Authorization: Bearer {TOKEN}"}))

    deliver_event(
        sessions,
        events,
        client,
        event_row["event_id"],
        audit=audit,
        request_id=REQUEST_ID,
        sleeper=_RecordingSleeper(),
        queue=queue,
    )

    row = queue.repository.get_by_event_id(event_row["event_id"])
    assert TOKEN not in (row["last_error"] or "")
    assert "***" in (row["last_error"] or ""), "凭据应被抹成掩码而不是原样留着"


def test_a_first_time_success_leaves_no_queue_debt(stub, client, repos):
    """一次就送到的出站不该在队列里留痕——没欠账就没有欠账行。"""
    sessions, events, audit, queue = repos
    _, event_row = _confirmed_event(sessions, events)

    deliver_event(
        sessions,
        events,
        client,
        event_row["event_id"],
        audit=audit,
        request_id=REQUEST_ID,
        queue=queue,
    )

    assert queue.get(event_row["event_id"]) is None
    assert queue.repository.count() == 0


def test_unconfirmed_session_never_enters_the_queue(stub, client, repos):
    """未确认的换液不是欠账：它压根没打算发出去，不该出现在待写回清单里。"""
    sessions, events, audit, queue = repos
    record = _seed_session(sessions, state="needs_change")
    from infusion_monitor.identity.idempotency import confirm_replacement_once

    outcome = confirm_replacement_once(
        sessions, events, record["session_id"], bag_seq="B01", operator_id=OPERATOR
    )

    deliver_event(
        sessions,
        events,
        client,
        str(outcome.event_id),
        audit=audit,
        request_id=REQUEST_ID,
        queue=queue,
    )

    assert queue.get(str(outcome.event_id)) is None
    assert queue.repository.count() == 0


def test_reconcile_recovers_a_half_attempted_event_after_a_restart(repos):
    """进程在重试到一半时重启：事件表里还欠着，队列要能把它补回来。"""
    sessions, events, audit, queue = repos
    _, event_row = _confirmed_event(sessions, events)
    # 模拟「发过一次、进程就被杀了」：事件停在 pending 且已试过一次。
    events.mark_writeback(event_row["event_id"], "pending", attempt_count=1)

    assert queue.get(event_row["event_id"]) is None
    recovered = queue.reconcile(events)
    assert len(recovered) == 1

    entry = queue.get(event_row["event_id"])
    assert entry is not None
    assert entry.status == STATUS_QUEUED
    assert entry.attempt_count == 1
    assert entry.next_retry_at


# ── 3. 重放幂等 ──────────────────────────────────────────────────────────────


def test_replay_resends_the_same_idempotency_key_and_creates_no_second_event(
    stub, client, repos
):
    """重放投的是**已存在**的事件：幂等键不变，业务事件不新增。"""
    import asyncio

    sessions, events, audit, queue = repos
    _, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(503)
    deliver_event(
        sessions,
        events,
        client,
        event_row["event_id"],
        audit=audit,
        request_id=REQUEST_ID,
        sleeper=_RecordingSleeper(),
        queue=queue,
    )
    assert stub.call_count == 3

    # 对端恢复了。
    stub.set_default_status(201)
    stub.reset()
    before = events.count()

    from infusion_monitor.adapters.retry_queue import replay_entry

    result = asyncio.run(
        replay_entry(
            queue,
            sessions,
            events,
            client,
            event_row["event_id"],
            operator_id=REPLAYER,
            audit=audit,
            request_id=REQUEST_ID,
        )
    )

    assert result.ok is True
    assert result.replayed is True
    assert result.idempotent_hit is False
    assert stub.call_count == 1, "人工重放只发一次"
    sent = stub.last()
    assert sent.idempotency_key == event_row["client_event_id"]
    assert sent.json()["client_event_id"] == event_row["client_event_id"]
    assert events.count() == before, "重放写出了第二条业务事件"


def test_replaying_an_already_delivered_debt_sends_nothing(stub, client, repos):
    """已送达的重放：一次 HTTP 都不发，直接回幂等命中。"""
    import asyncio

    sessions, events, audit, queue = repos
    _, event_row = _confirmed_event(sessions, events)
    deliver_event(
        sessions,
        events,
        client,
        event_row["event_id"],
        audit=audit,
        request_id=REQUEST_ID,
        queue=queue,
    )
    assert stub.call_count == 1
    stub.reset()

    from infusion_monitor.adapters.retry_queue import replay_entry

    result = asyncio.run(
        replay_entry(
            queue,
            sessions,
            events,
            client,
            event_row["event_id"],
            operator_id=REPLAYER,
            audit=audit,
            request_id=REQUEST_ID,
        )
    )

    assert result.idempotent_hit is True
    assert result.replayed is False
    assert stub.call_count == 0, "已送达的事件被重发了一次"
    assert events.count() == 1


# ── 4. 重放成功后队列清理 ────────────────────────────────────────────────────


def test_successful_replay_clears_the_entry_from_the_pending_view(
    stub, client, repos
):
    """重放成功后：那一行结清（sent），待处理清单里不再出现，事件行数不变。"""
    import asyncio

    sessions, events, audit, queue = repos
    _, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(503)
    deliver_event(
        sessions,
        events,
        client,
        event_row["event_id"],
        audit=audit,
        request_id=REQUEST_ID,
        sleeper=_RecordingSleeper(),
        queue=queue,
    )
    assert len(queue.list_pending()) == 1

    stub.set_default_status(201)
    from infusion_monitor.adapters.retry_queue import replay_entry

    asyncio.run(
        replay_entry(
            queue,
            sessions,
            events,
            client,
            event_row["event_id"],
            operator_id=REPLAYER,
            audit=audit,
            request_id=REQUEST_ID,
        )
    )

    entry = queue.get(event_row["event_id"])
    assert entry is not None, "结清后这一行不该被删掉——它是「欠账何时结清」的凭据"
    assert entry.status == STATUS_SENT
    assert entry.dead_letter_reason is None
    assert queue.list_pending() == [], "已送达的欠账还挂在待处理清单里"
    assert queue.count_by_status() == {STATUS_SENT: 1}
    assert events.count() == 1
    assert events.get(event_row["event_id"])["writeback_status"] == "sent"


def test_failed_replay_keeps_the_debt_in_the_pending_view(stub, client, repos):
    """重放仍失败：欠账留在清单里，尝试次数往上走，绝不静默消失。"""
    import asyncio

    sessions, events, audit, queue = repos
    _, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(503)
    deliver_event(
        sessions,
        events,
        client,
        event_row["event_id"],
        audit=audit,
        request_id=REQUEST_ID,
        sleeper=_RecordingSleeper(),
        queue=queue,
    )

    from infusion_monitor.adapters.retry_queue import replay_entry

    result = asyncio.run(
        replay_entry(
            queue,
            sessions,
            events,
            client,
            event_row["event_id"],
            operator_id=REPLAYER,
            audit=audit,
            request_id=REQUEST_ID,
        )
    )

    assert result.ok is False
    entry = queue.get(event_row["event_id"])
    assert entry is not None
    assert entry.status == STATUS_DEAD_LETTER
    assert entry.attempt_count == 4, "重放失败也要计入尝试次数"
    assert len(queue.list_pending()) == 1


def test_pending_statuses_exclude_sent():
    """已送达的不算欠账：待处理清单里不该出现 sent。"""
    assert STATUS_SENT not in PENDING_STATUSES
    assert set(PENDING_STATUSES) == {STATUS_QUEUED, STATUS_DEAD_LETTER, STATUS_FAILED}


def test_retry_at_is_a_future_timestamp():
    """下次可重试时刻必须是将来——写成过去等于「立刻重试」的死循环。"""
    from infusion_monitor.store.repositories import utc_now

    moment = retry_at(1)
    assert moment > utc_now()


# ── HTTP 端点：查与重放 ─────────────────────────────────────────────────────


@pytest.fixture
def http_app(stub, credential_env, tmp_path):
    """TestClient + 指向**同一个库文件**的观察连接（出站指向本地替身对端）。"""
    from fastapi.testclient import TestClient

    from infusion_monitor.api.service import create_app
    from infusion_monitor.config import Settings
    from infusion_monitor.store import apply_migrations, connect

    dsn = f"sqlite:///{tmp_path / 'pending.db'}"
    app = create_app(Settings(), database_url=dsn, frames_root=tmp_path / "frames")
    observer = connect(dsn)
    apply_migrations(observer)
    with TestClient(app) as test_client:
        # 出站指向替身对端：真实协议、真实路径、真实 Bearer 头，只是对端是本地替身。
        test_client.app.state.nurse_exec = NurseExecClient(
            base_url=f"{stub.base_url}/api/v1/infusion/events", api_token=TOKEN
        )
        yield test_client, observer
    observer.close()


def _seed_debt(observer) -> str:
    """造一笔「欠着对端」的换液事件：会话已确认、事件已落库、队列里是死信。"""
    sessions = SessionRepository(observer)
    events = EventRepository(observer)
    queue_repo = WritebackQueueRepository(observer)

    from infusion_monitor.identity.idempotency import confirm_replacement_once

    record = _seed_session(sessions, state="needs_change")
    outcome = confirm_replacement_once(
        sessions, events, record["session_id"], bag_seq="B01", operator_id=OPERATOR
    )
    event_row = events.get(str(outcome.event_id))
    queue_repo.enqueue(
        {
            "event_id": event_row["event_id"],
            "client_event_id": event_row["client_event_id"],
            "session_id": event_row["session_id"],
            "rack_id": event_row["rack_id"],
            "bed_no": event_row["bed_no"],
            "bag_seq": event_row["bag_seq"],
            "payload": event_row["payload"],
            "attempt_count": 3,
            "next_retry_at": retry_at(3),
            "last_error": "http_5xx: HTTP 503",
            "status": STATUS_DEAD_LETTER,
            "dead_letter_reason": "retries_exhausted",
        }
    )
    return str(outcome.event_id)


def test_pending_endpoint_lists_the_debt_and_nothing_else(http_app):
    """清单只列**还没送达**的欠账，已送达的不出现。"""
    client, observer = http_app
    event_id = _seed_debt(observer)

    response = client.get("/api/v1/infusion/pending")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["dead_letter_count"] == 1
    assert body["by_status"] == {STATUS_DEAD_LETTER: 1}
    item = body["items"][0]
    assert item["event_id"] == event_id
    assert item["status"] == STATUS_DEAD_LETTER
    assert item["attempt_count"] == 3
    assert item["replayable"] is True
    assert item["last_error"]
    # 清单里不该出现凭据
    assert TOKEN not in response.text


def test_pending_endpoint_can_filter_by_rack_and_status(http_app):
    """按架、按状态过滤：信息科先看某一架。"""
    client, observer = http_app
    _seed_debt(observer)

    same_rack = client.get("/api/v1/infusion/pending", params={"rack_id": "R-EXAMPLE-37"})
    assert same_rack.json()["count"] == 1
    other = client.get("/api/v1/infusion/pending", params={"rack_id": "R-OTHER"})
    assert other.json()["count"] == 0
    queued = client.get("/api/v1/infusion/pending", params={"status": STATUS_QUEUED})
    assert queued.json()["count"] == 0
    bad = client.get("/api/v1/infusion/pending", params={"status": "sent"})
    assert bad.status_code == 422
    assert bad.json()["detail"]["error_code"] == "INGEST_INVALID"


def test_replay_endpoint_sends_the_debt_once_and_clears_it(http_app, stub):
    """重放：发出一次、结清那一行、清单里不再出现、业务事件不新增。"""
    client, observer = http_app
    event_id = _seed_debt(observer)
    events = EventRepository(observer)
    before = events.count()

    response = client.post(
        f"/api/v1/infusion/pending/{event_id}/replay",
        json={"operator_id": REPLAYER},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["replayed"] is True
    assert body["idempotent_hit"] is False
    assert body["writeback_status"] == "sent"
    assert body["queue_status"] == STATUS_SENT

    assert stub.call_count == 1
    sent = stub.last()
    assert sent.authorization == f"Bearer {TOKEN}"
    assert sent.idempotency_key == body["client_event_id"]
    assert sent.json()["client_event_id"] == body["client_event_id"]

    # 队列清理：那一行结清，待处理清单空了，业务事件行数不变。
    listing = client.get("/api/v1/infusion/pending").json()
    assert listing["count"] == 0
    assert events.count() == before


def test_replay_endpoint_is_idempotent_on_a_second_call(http_app, stub):
    """再点一次重放：幂等命中，一次 HTTP 都不发。"""
    client, observer = http_app
    event_id = _seed_debt(observer)

    first = client.post(
        f"/api/v1/infusion/pending/{event_id}/replay", json={"operator_id": REPLAYER}
    )
    assert first.status_code == 200
    assert stub.call_count == 1

    second = client.post(
        f"/api/v1/infusion/pending/{event_id}/replay", json={"operator_id": REPLAYER}
    )
    assert second.status_code == 200, second.text
    assert second.json()["idempotent_hit"] is True
    assert second.json()["replayed"] is False
    assert stub.call_count == 1, "已送达的欠账被重发了一次"


def test_replay_endpoint_returns_404_for_an_unknown_debt(http_app):
    """队列里没有这一笔 → 404，并说清「先看清单」。"""
    client, _observer = http_app

    response = client.post(
        "/api/v1/infusion/pending/evt-does-not-exist/replay",
        json={"operator_id": REPLAYER},
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["error_code"] == "NOT_FOUND"
    assert detail["issues"][0]["code"] == "pending_entry_not_found"


def test_replay_endpoint_requires_an_operator_id(http_app):
    """「谁补的」必须能追溯到人：operator_id 必填。"""
    client, observer = http_app
    event_id = _seed_debt(observer)

    response = client.post(f"/api/v1/infusion/pending/{event_id}/replay", json={})
    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "INGEST_INVALID"


def test_replay_endpoint_reports_a_peer_rejection_without_dropping_the_debt(
    http_app, stub
):
    """对端业务拒绝：回 409 WRITEBACK_REJECTED，欠账仍在清单里。"""
    client, observer = http_app
    event_id = _seed_debt(observer)
    stub.set_default_status(400, body=json.dumps({"message": "输注单号不存在"}))

    response = client.post(
        f"/api/v1/infusion/pending/{event_id}/replay", json={"operator_id": REPLAYER}
    )
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["error_code"] == "WRITEBACK_REJECTED"
    assert "http_4xx" in detail["issues"][0]["reason"]

    listing = client.get("/api/v1/infusion/pending").json()
    assert listing["count"] == 1, "被拒的欠账从清单里消失了"
    assert listing["items"][0]["event_id"] == event_id


def test_replay_writes_an_audit_row_naming_the_operator(http_app):
    """谁补的要留痕：审计链里有一行 writeback.replayed，点名操作人。"""
    client, observer = http_app
    event_id = _seed_debt(observer)

    client.post(f"/api/v1/infusion/pending/{event_id}/replay", json={"operator_id": REPLAYER})

    audit = AuditChain(observer)
    rows = audit.list_by_action("writeback.replayed")
    assert len(rows) == 1
    assert rows[0]["actor"] == f"operator:{REPLAYER}"
    assert rows[0]["detail"]["operator_id"] == REPLAYER
    assert rows[0]["detail"]["event_id"] == event_id


def test_an_idempotent_replay_writes_no_extra_audit_row(http_app):
    """幂等命中**不写**「重放过」的审计行。

    什么都没发生（一次 HTTP 都没发）却留下痕迹，飞检材料里就会出现
    「同一条事件送出两次」的错觉。
    """
    client, observer = http_app
    event_id = _seed_debt(observer)

    client.post(f"/api/v1/infusion/pending/{event_id}/replay", json={"operator_id": REPLAYER})
    client.post(f"/api/v1/infusion/pending/{event_id}/replay", json={"operator_id": REPLAYER})

    audit = AuditChain(observer)
    assert len(audit.list_by_action("writeback.replayed")) == 1
    assert audit.list_by_action("writeback.replay_failed") == []


def test_summary_reports_the_queue_debt(http_app):
    """运行摘要是护士长看「欠了几笔」的地方。"""
    client, observer = http_app
    _seed_debt(observer)

    body = client.get("/api/v1/summary").json()
    assert body["writeback"]["queue_by_status"] == {STATUS_DEAD_LETTER: 1}
    assert body["writeback"]["queue_pending_count"] == 1
    assert body["writeback"]["queue_dead_letter_count"] == 1


def test_openapi_documents_the_pending_endpoints(http_app):
    """两个新端点进 /docs，错误码与真实返回一致。"""
    client, _observer = http_app
    spec = client.get("/openapi.json").json()

    assert "/api/v1/infusion/pending" in spec["paths"]
    assert "/api/v1/infusion/pending/{event_id}/replay" in spec["paths"]
    replay_responses = spec["paths"]["/api/v1/infusion/pending/{event_id}/replay"]["post"][
        "responses"
    ]
    assert "404" in replay_responses
    assert "409" in replay_responses
    assert "WRITEBACK_REJECTED：" in replay_responses["409"]["description"]
