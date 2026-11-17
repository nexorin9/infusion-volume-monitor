"""出站写回契约测试：对**本地替身对端**（真实 TCP 端口）核验请求契约与失败处理。

替身只出现在测试数据层。**出站客户端与字段规范是正式实现**：真实协议、真实端点
路径、真实 Bearer 凭据位；真实护理执行平台端点本轮未实连。

七组断言：

1. **请求头**——``Authorization: Bearer`` 取自环境变量配置面、``Content-Type``、
   ``Idempotency-Key`` 与请求体 ``client_event_id`` 同值；
2. **端点与字段**——POST 到 ``/api/v1/infusion/events``；请求体**全量 14 个字段**
   逐个核对（名字、类型、值），字段规范文档与模型同源、与对端接口文档的对应表齐全；
3. **门禁语义**——未标定袋型时 ``estimated_volume_ml`` / ``eta_minutes`` **整体不出现**，
   不是 ``null``；
4. **幂等语义**——重放同一次确认不产生第二条事件、已 sent 的事件不重复外发；
5. **退避次数**——对端持续 5xx 时首次 + 2 次重试，退避间隔为指数（1s / 4s），
   耗尽后落 ``pending_writeback`` 并写审计；
6. **失败分类（六类各一条）**——业务拒绝 4xx、对端 5xx、连接超时、读超时、
   **半开连接**、连接错误；每一类都断言「可重试与否 + 终态 + 按类别计数」与
   :data:`FAILURE_CLASSES` 这张表逐项一致，不允许实现与表各说各话；
7. **未确认永不出站**——会话还在 ``needs_change`` 时一次 HTTP 都不发，事件行保留，
   只留一行 ``writeback.refused`` 审计。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from infusion_monitor.adapters.nurse_exec import (
    DECISION_SOURCES_OUTBOUND,
    FAILURE_CLASSES,
    OUTBOUND_EVENT_SCHEMA,
    PEER_FIELD_BINDINGS,
    RETRYABLE_FAILURE_KINDS,
    TRIGGER_REASONS_OUTBOUND,
    NurseExecClient,
    OutboundEvent,
    business_rejection_reason,
    classify_exception,
    contract_markdown,
    failure_class,
    is_retryable,
    scrub,
)
from infusion_monitor.adapters.writeback import (
    BACKOFF_BASE_SECONDS,
    MAX_RETRIES,
    RETRYABLE_KINDS,
    WritebackRefused,
    confirm_and_deliver,
    count_kinds,
    deliver_event,
)
from infusion_monitor.store import (
    AuditChain,
    EventRepository,
    SessionRepository,
    apply_migrations,
    connect,
)
from tests.support import NurseExecStub

#: 写回的每一层动作都必须能按 request_id 追溯到这次调用
REQUEST_ID = "req-contract-0011"

TOKEN = "contract-token-not-real"
OPERATOR = "N-EXAMPLE-11"


@pytest.fixture
def stub():
    with NurseExecStub(status=201) as running:
        yield running


def _stub_client(stub, **kwargs) -> NurseExecClient:
    """指向替身对端的出站客户端（端点路径与凭据位与生产完全一致）。"""
    return NurseExecClient(
        base_url=f"{stub.base_url}/api/v1/infusion/events", api_token=TOKEN, **kwargs
    )


@pytest.fixture
def client(stub):
    return _stub_client(stub)


@pytest.fixture
def repos(conn):
    apply_migrations(conn)
    return SessionRepository(conn), EventRepository(conn), AuditChain(conn)


def _seed_named_session(sessions: SessionRepository, *, state: str = "needs_change") -> dict:
    """一袋已点名的会话。``state`` 可切到 monitoring/replaced 以验门禁。"""
    payload = {
        "rack_id": "R-EXAMPLE-11",
        "bed_no": "EXAMPLE-BED-11",
        "infusion_order_no": "EXAMPLE-ORDER-0011",
        "shift_key": "2026-10-22-AM",
        "bag_seq": "B01",
        "slot_index": 0,
        "bag_type_code": "TYPE-A",
        "frames_dir": "/tmp/frames",
        "frame_count": 16,
        "captured_at": "2026-10-22T09:41:00+08:00",
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
    """走真幂等入口确认换液，拿到已落库的出站事件行。"""
    from infusion_monitor.identity.idempotency import confirm_replacement_once

    record = _seed_named_session(sessions)
    outcome = confirm_replacement_once(
        sessions, events, record["session_id"], bag_seq="B01", operator_id=operator_id
    )
    return record, events.get(str(outcome.event_id))


class _RecordingSleeper:
    """把退避等待记下来而不真睡——测试不该为指数退避付出 5 秒墙钟。"""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# ── 1. 请求头 ────────────────────────────────────────────────────────────────


def test_authorization_header_uses_configured_token(stub, client, repos):
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)

    deliver_event(sessions, events, client, event_row["event_id"], audit=audit, request_id=REQUEST_ID)

    assert stub.call_count == 1
    request = stub.last()
    assert request.authorization == f"Bearer {TOKEN}"
    assert request.headers["content-type"].startswith("application/json")
    assert request.headers["accept"] == "application/json"


def test_idempotency_key_header_matches_body_field(stub, client, repos):
    """幂等键在请求头与请求体里必须同值：对端任一侧去重都不会漏。"""
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)

    deliver_event(sessions, events, client, event_row["event_id"], audit=audit, request_id=REQUEST_ID)

    request = stub.last()
    assert request.idempotency_key == request.json()["client_event_id"]
    assert request.idempotency_key == event_row["client_event_id"]


def test_token_never_leaks_into_error_text():
    """凭据不进日志、不进 last_error：有些网关会把请求头原样回显。"""
    echoed = f'{{"detail":"upstream rejected Authorization: Bearer {TOKEN}"}}'
    assert TOKEN not in scrub(echoed, TOKEN)
    assert "***" in scrub(echoed, TOKEN)


# ── 2. 端点与字段 ─────────────────────────────────────────────────────────────


def test_posts_to_events_endpoint(stub, client, repos):
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)

    deliver_event(sessions, events, client, event_row["event_id"], audit=audit, request_id=REQUEST_ID)

    assert stub.last().method == "POST"
    assert stub.last().path == "/api/v1/infusion/events"


def test_request_body_matches_outbound_schema(stub, client, repos):
    """请求体**全量字段**逐个核对：名字、标量类型与取值，不放过任何一个。"""
    sessions, events, audit = repos
    record, event_row = _confirmed_event(sessions, events)

    deliver_event(sessions, events, client, event_row["event_id"], audit=audit, request_id=REQUEST_ID)

    body = stub.last().json()
    schema_names = {spec.name for spec in OUTBOUND_EVENT_SCHEMA}
    # 这一袋所有字段都有值，所以出站体应当**一个不多一个不少**地覆盖字段表。
    assert set(body) == schema_names, f"出站体与字段表对不上：多 {set(body) - schema_names} 少 {schema_names - set(body)}"
    required = {spec.name for spec in OUTBOUND_EVENT_SCHEMA if spec.required}
    assert required <= set(body), "必填字段缺了"
    numeric = {spec.name for spec in OUTBOUND_EVENT_SCHEMA if spec.type in ("float", "int")}
    for name in numeric & set(body):
        assert isinstance(body[name], (int, float)) and not isinstance(body[name], bool)
    text = {spec.name for spec in OUTBOUND_EVENT_SCHEMA if spec.type in ("str", "enum")}
    for name in text & set(body):
        assert isinstance(body[name], str) and body[name], f"{name} 应是非空字符串"

    # 逐字段取值核对：事件 ID 与幂等键必须来自事件行本身，不能由请求侧另生成。
    assert body["event_id"] == event_row["event_id"]
    assert body["client_event_id"] == event_row["client_event_id"]
    assert body["session_id"] == record["session_id"]
    assert body["event_type"] == "infusion.bag.changed"
    assert body["rack_id"] == "R-EXAMPLE-11"
    assert body["bed_no"] == "EXAMPLE-BED-11"
    assert body["infusion_order_no"] == "EXAMPLE-ORDER-0011"
    assert body["bag_seq"] == "B01"
    assert body["operator_id"] == OPERATOR
    assert body["fill_ratio"] == pytest.approx(0.18)
    assert body["estimated_volume_ml"] == pytest.approx(104.4)
    assert body["eta_minutes"] == 26 and isinstance(body["eta_minutes"], int)
    assert body["trigger_reason"] in TRIGGER_REASONS_OUTBOUND
    assert body["decision_source"] in DECISION_SOURCES_OUTBOUND
    # occurred_at 必须是可解析的 ISO8601，且带时区（对端按它排序执行记录）。
    assert datetime.fromisoformat(body["occurred_at"]).tzinfo is not None


def test_every_enumerated_field_carries_an_allowed_value(stub, client, repos):
    """带枚举的字段，取值必须落在枚举表内：对端按码分派，枚举外的码会被当成未知事件。"""
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)

    deliver_event(sessions, events, client, event_row["event_id"], audit=audit, request_id=REQUEST_ID)

    body = stub.last().json()
    for spec in OUTBOUND_EVENT_SCHEMA:
        if spec.enum and spec.name in body:
            assert body[spec.name] in spec.enum, f"{spec.name}={body[spec.name]} 不在枚举 {spec.enum} 内"


def test_peer_field_bindings_cover_the_schema_exactly():
    """字段 → 对端接口文档的对应表必须与字段表一一对应：接挂时逐行对的就是它。"""
    assert set(PEER_FIELD_BINDINGS) == {spec.name for spec in OUTBOUND_EVENT_SCHEMA}
    for name, binding in PEER_FIELD_BINDINGS.items():
        assert binding.location.strip(), f"{name} 缺对端契约落点"
        assert binding.note.strip(), f"{name} 缺对端侧约定"


def test_idempotency_key_is_bound_to_both_peer_locations():
    """幂等键在请求体字段与请求头两处都有明确落点，且文档写清「必须同值」。"""
    binding = PEER_FIELD_BINDINGS["client_event_id"]
    assert "Idempotency-Key" in binding.location and "clientEventId" in binding.location
    doc = contract_markdown()
    assert "`client_event_id` | 请求体 clientEventId + 请求头 Idempotency-Key" in doc


def test_contract_markdown_file_is_in_sync_with_the_schema():
    """落盘的契约文档必须等于代码生成的版本——手改文档会当场被抓出来。"""
    doc_path = Path(__file__).resolve().parents[2] / "docs" / "nurse-exec-contract.md"
    assert doc_path.exists(), "出站契约文档缺失"
    assert contract_markdown() == doc_path.read_text(encoding="utf-8"), (
        "docs/nurse-exec-contract.md 与 OUTBOUND_EVENT_SCHEMA 已漂移，"
        "请重新生成：PYTHONPATH=src python -c "
        "\"from infusion_monitor.adapters.nurse_exec import contract_markdown; "
        "print(contract_markdown(), end='')\" > docs/nurse-exec-contract.md"
    )


def test_contract_markdown_lists_the_failure_table_and_summary_format():
    doc = contract_markdown()
    for kind in FAILURE_CLASSES:
        assert f"| `{kind}` |" in doc, f"契约文档缺失败类别 {kind}"
    assert "## 错误摘要格式" in doc and "attempts=" in doc and "counts=" in doc


def test_outbound_enums_cover_the_sources_they_mirror():
    """出站枚举必须**覆盖**上游的来源表，而不是各抄一份后慢慢漂移。"""
    from infusion_monitor.gateway.audit import DECISION_SOURCES
    from infusion_monitor.identity.state_machine import TRIGGER_REASONS

    assert set(DECISION_SOURCES) <= set(DECISION_SOURCES_OUTBOUND)
    assert set(TRIGGER_REASONS) <= set(TRIGGER_REASONS_OUTBOUND)
    # 写回这一步的结论已被护士确认取代，因此出站独有的那一个来源码。
    assert "nurse_confirmed" in DECISION_SOURCES_OUTBOUND
    assert "nurse_confirmed" in TRIGGER_REASONS_OUTBOUND


def test_outbound_model_rejects_unknown_field():
    """多发一个字段就是「对端收到我们没打算让它看到的东西」：直接拒绝。"""
    with pytest.raises(Exception):
        OutboundEvent(
            event_id="evt_1", client_event_id="k", rack_id="r", bed_no="b",
            infusion_order_no="o", bag_seq="B01", operator_id="n", occurred_at="t",
            patient_name="不该出现的字段",
        )


def test_contract_markdown_is_generated_from_same_schema():
    """契约文档由 schema 生成：文档表格与模型字段必须同源（防手改漂移）。"""
    doc = contract_markdown()
    for spec in OUTBOUND_EVENT_SCHEMA:
        assert f"| `{spec.name}` |" in doc, f"契约文档缺字段 {spec.name}"
    assert {spec.name for spec in OUTBOUND_EVENT_SCHEMA} == set(OutboundEvent.model_fields)
    assert "Idempotency-Key" in doc and "null" in doc


# ── 3. 门禁语义（未标定 → 字段整体不出现）─────────────────────────────────────


def test_missing_volume_fields_are_omitted_not_null(stub, client, repos):
    """未标定袋型时，出站载荷里也不能出现 null —— 对端会把 null 当 0 读成「袋子空了」。"""
    sessions, events, audit = repos
    record = sessions.upsert_session(
        {
            "rack_id": "R-EXAMPLE-11", "bed_no": "EXAMPLE-BED-11",
            "infusion_order_no": "EXAMPLE-ORDER-0011", "shift_key": "2026-10-22-PM",
            "bag_seq": "B02", "slot_index": 1, "bag_type_code": "TYPE-B",
            "frames_dir": "/tmp/f", "frame_count": 16,
            "captured_at": "2026-10-22T13:41:00+08:00",
            "estimated_volume_ml": None, "eta_minutes": None, "fill_ratio": None,
            "calibration_state": "uncalibrated",
        }
    )
    sessions.apply_state_transition(record["session_id"], "needs_change")
    from infusion_monitor.identity.idempotency import confirm_replacement_once

    outcome = confirm_replacement_once(
        sessions, events, record["session_id"], bag_seq="B02", operator_id=OPERATOR
    )

    deliver_event(sessions, events, client, str(outcome.event_id), audit=audit, request_id=REQUEST_ID)

    body = stub.last().json()
    assert "estimated_volume_ml" not in body
    assert "eta_minutes" not in body
    assert "fill_ratio" not in body
    assert "estimated_volume_ml" not in json.dumps(body)  # 连 null 都没有


# ── 4. 幂等语义 ──────────────────────────────────────────────────────────────


def test_replayed_confirmation_does_not_create_second_event(stub, client, repos):
    """护士的按钮重试：同一次确认仍只有一条事件，且已 sent 的不重复外发。"""
    sessions, events, audit = repos
    record, event_row = _confirmed_event(sessions, events)

    first = deliver_event(sessions, events, client, event_row["event_id"], audit=audit, request_id=REQUEST_ID)
    second = deliver_event(sessions, events, client, event_row["event_id"], audit=audit, request_id=REQUEST_ID)

    assert first.writeback_status == "sent" and first.attempts == 1
    assert second.kind == "already_sent"
    assert stub.call_count == 1, "已送出的事件被重复外发了"
    assert events.count() == 1
    # 整条链路重放（确认 + 写回）仍只有一条事件
    _, outcome = confirm_and_deliver(
        sessions, events, client, record["session_id"], bag_seq="B01", operator_id=OPERATOR
    )
    assert events.count() == 1
    assert outcome.writeback_status == "sent"
    assert stub.call_count == 1


def test_pending_event_is_retried_on_replayed_confirmation(stub, client, repos):
    """上次没送出去时，护士再点一次正是我们希望它重试的时机。"""
    sessions, events, audit = repos
    record, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(503)
    sleeper = _RecordingSleeper()

    first = deliver_event(
        sessions, events, client, event_row["event_id"],
        max_retries=0, backoff_base=0.0, sleeper=sleeper,
    )
    assert first.writeback_status == "pending_writeback"

    stub.set_default_status(200)
    second = deliver_event(
        sessions, events, client, event_row["event_id"],
        backoff_base=0.0, sleeper=sleeper,
    )
    assert second.writeback_status == "sent"
    assert events.count() == 1


# ── 5. 退避次数 ──────────────────────────────────────────────────────────────


def test_exhausted_retries_land_in_pending_writeback(stub, client, repos):
    """持续 5xx：首次 + 2 次重试（指数退避 1s / 4s），耗尽后 pending_writeback。"""
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(503, body='{"detail":"upstream busy"}')
    sleeper = _RecordingSleeper()

    outcome = deliver_event(
        sessions, events, client, event_row["event_id"], sleeper=sleeper, audit=audit, request_id=REQUEST_ID
    )

    assert outcome.attempts == MAX_RETRIES + 1 == 3
    assert sleeper.delays == [BACKOFF_BASE_SECONDS, BACKOFF_BASE_SECONDS * 4]
    assert stub.call_count == 3
    assert outcome.writeback_status == "pending_writeback"
    assert outcome.next_retry_at, "耗尽后必须给出可重放的时间点"
    assert "503" in (outcome.last_error or "")

    stored = events.get(event_row["event_id"])
    assert stored["writeback_status"] == "pending_writeback"
    assert stored["attempt_count"] == 3
    assert TOKEN not in (stored["last_error"] or "")
    assert events.count() == 1, "失败的写回把事件弄丢了——这是最不可接受的"

    actions = [row["action"] for row in audit.list_by_target(str(event_row["session_id"]))]
    assert actions.count("writeback.retry") == 2, "每次退避重试都要留一行"
    assert actions[-1] == "writeback.pending"
    retry_rows = [
        row for row in audit.list_by_target(str(event_row["session_id"]))
        if row["action"] == "writeback.retry"
    ]
    assert [row["detail"]["backoff_seconds"] for row in retry_rows] == [1.0, 4.0]


def test_backoff_is_exponential():
    from infusion_monitor.adapters.writeback import backoff_seconds

    assert backoff_seconds(0) == 1.0
    assert backoff_seconds(1) == 4.0
    assert backoff_seconds(2) == 16.0


# ── 6. 失败分类 ──────────────────────────────────────────────────────────────


def test_business_rejection_is_not_retried(stub, client, repos):
    """4xx 是对端的业务拒绝：同一份载荷重发只会再被拒一次，不浪费病区带宽。"""
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(422, body='{"detail":"unknown infusion_order_no"}')
    sleeper = _RecordingSleeper()

    outcome = deliver_event(
        sessions, events, client, event_row["event_id"], sleeper=sleeper,
        audit=audit, request_id=REQUEST_ID,
    )

    assert outcome.attempts == 1
    assert sleeper.delays == []
    assert stub.call_count == 1
    assert outcome.writeback_status == "failed"
    assert "unknown infusion_order_no" in (outcome.last_error or "")
    actions = [row["action"] for row in audit.list_by_target(str(event_row["session_id"]))]
    assert actions[-1] == "writeback.rejected"


def test_read_timeout_is_retried_and_then_pending(stub, client, repos):
    """对端慢（读超时）属可重试：重试一次成功就不该留下待处理项。"""
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)
    # 第 1 次：替身慢一拍 → 客户端读超时；第 2 次：对端 503；第 3 次：正常应答
    stub.queue_response(201, latency=0.6)
    stub.queue_status(503)
    slow = NurseExecClient(
        base_url=f"{stub.base_url}/api/v1/infusion/events", api_token=TOKEN, timeout=0.2
    )
    stub.set_default_status(201)

    outcome = deliver_event(
        sessions, events, slow, event_row["event_id"], sleeper=_RecordingSleeper()
    )

    assert outcome.writeback_status == "sent"
    assert outcome.attempts >= 2
    assert "timeout" in outcome.attempt_kinds[0]


def test_unreachable_endpoint_is_retried_then_pending(repos):
    """对端完全不可达：退避后置 pending_writeback，事件一行不少。

    这里不断言具体失败类别——沙箱/网络栈对「连接被拒」是立刻 RST 还是丢包超时
    的处理不同，断言某个类别会让测试随环境漂移。真正要守住的是**处置**：
    传输层失败一律可重试、次数有上限、耗尽后事件不丢。
    """
    import socket

    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()
    offline = NurseExecClient(
        base_url=f"http://127.0.0.1:{dead_port}/api/v1/infusion/events",
        api_token=TOKEN,
        timeout=0.3,
    )
    sleeper = _RecordingSleeper()

    outcome = deliver_event(
        sessions, events, offline, event_row["event_id"], sleeper=sleeper, audit=audit, request_id=REQUEST_ID
    )

    assert outcome.kind in RETRYABLE_KINDS, f"传输层失败不该被当业务拒绝：{outcome.kind}"
    assert outcome.writeback_status == "pending_writeback"
    assert outcome.attempts == 3
    assert len(sleeper.delays) == 2
    assert events.count() == 1
    stored = events.get(event_row["event_id"])
    assert stored["attempt_count"] == 3
    assert TOKEN not in (stored["last_error"] or "")


def test_connect_error_is_classified_as_retryable():
    """真正的连接错误（DNS/对端不存在）单独归一类，且算可重试。"""
    import httpx

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("name or service not known", request=request)

    offline = NurseExecClient(
        base_url="https://nurse-exec.invalid/api/v1/infusion/events",
        api_token=TOKEN,
        client=httpx.Client(transport=httpx.MockTransport(boom)),
    )
    event = OutboundEvent(
        event_id="evt_x", client_event_id="k-x", rack_id="r", bed_no="b",
        infusion_order_no="o", bag_seq="B01", operator_id="n", occurred_at="t",
    )

    attempt = offline.attempt(event)

    assert attempt.ok is False
    assert attempt.kind == "connect_error"
    assert attempt.kind in RETRYABLE_KINDS
    assert TOKEN not in (attempt.error or "")


def test_retryable_kinds_exclude_business_rejection():
    assert "http_4xx" not in RETRYABLE_KINDS
    assert "http_5xx" in RETRYABLE_KINDS
    assert "read_timeout" in RETRYABLE_KINDS


# ── 7. 未确认永不出站 ────────────────────────────────────────────────────────


def test_unconfirmed_needs_change_is_never_sent(stub, client, repos):
    """本服务只建议换液。没点确认就发事件，对端会真的换新袋。"""
    sessions, events, audit = repos
    record = _seed_named_session(sessions, state="needs_change")

    # 手工塞一条「未经确认」的事件行（正常路径不会产生这样的行），验证门禁本身有效
    event, _ = events.append_event(
        {
            "client_event_id": f"probe-{record['session_id']}",
            "session_id": record["session_id"],
            "rack_id": record["rack_id"],
            "bed_no": record["bed_no"],
            "infusion_order_no": record["infusion_order_no"],
            "bag_seq": "B01",
            "operator_id": OPERATOR,
            "occurred_at": "2026-10-22T11:05:00+08:00",
        }
    )

    outcome = deliver_event(sessions, events, client, event["event_id"], audit=audit, request_id=REQUEST_ID)

    assert outcome.refused is True
    assert outcome.refusal_reason == "session_not_confirmed"
    assert stub.call_count == 0, "未确认的换液发出了 HTTP 请求"
    stored = events.get(event["event_id"])
    assert stored["writeback_status"] == "pending", "拒发不应把事件标成已处理"
    assert events.count() == 1
    actions = [row["action"] for row in audit.list_by_target(record["session_id"])]
    assert "writeback.refused" in actions


def test_missing_event_raises_refused(repos):
    sessions, events, audit = repos
    offline = NurseExecClient(base_url="http://127.0.0.1:1/x", api_token=TOKEN)
    with pytest.raises(WritebackRefused):
        deliver_event(sessions, events, offline, "evt_not_there")


def test_successful_writeback_is_audited_with_attempt_count(stub, client, repos):
    sessions, events, audit = repos
    record, event_row = _confirmed_event(sessions, events)

    deliver_event(sessions, events, client, event_row["event_id"], audit=audit, request_id=REQUEST_ID)

    rows = audit.list_by_target(record["session_id"])
    sent = [row for row in rows if row["action"] == "writeback.sent"]
    assert len(sent) == 1
    assert sent[0]["outcome"] == "accepted"
    assert sent[0]["detail"]["attempts"] == 1
    assert sent[0]["detail"]["client_event_id"] == event_row["client_event_id"]
    assert TOKEN not in json.dumps(sent[0]["detail"], ensure_ascii=False)


# ── 8. 六类失败各一条 ────────────────────────────────────────────────────────


class _ScriptedTransport:
    """按脚本依次抛异常或返回应答，并记录每次**真实组装出来的请求**。

    连接超时、读超时、连接错误这三类没法让替身造出来（替身总是能应答），
    所以它们走注入的传输层。**注入的只是传输层**：报头与请求体照常由产品代码
    组装，与真实出站完全一致。
    """

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.script.pop(0) if self.script else 201
        if isinstance(step, int):
            return httpx.Response(step, json={"accepted": True}, request=request)
        raise step("injected by the outbound contract test", request=request)


def _injected_client(script, *, timeout: float = 5.0):
    transport = _ScriptedTransport(script)
    client = NurseExecClient(
        base_url="https://nurse-exec.invalid/api/v1/infusion/events",
        api_token=TOKEN,
        timeout=timeout,
        client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    return client, transport


#: 六类失败 × 期望处置。表里的可重试与终态直接取自 :data:`FAILURE_CLASSES`，
#: 因此下面那条断言不是在复述实现，而是在钉住「实现 == 这张表」。
SIX_FAILURES = (
    "http_4xx", "http_5xx", "connect_timeout", "read_timeout", "half_open", "connect_error",
)


def test_failure_table_covers_exactly_the_six_peer_failure_shapes():
    for kind in SIX_FAILURES:
        assert kind in FAILURE_CLASSES, f"失败分类表缺 {kind}"
        assert failure_class(kind).description.strip()
        assert failure_class(kind).operator_hint.strip()
    assert failure_class("http_4xx").retryable is False, "业务拒绝重发只会再被拒一次"
    assert failure_class("http_5xx").retryable is True
    for kind in SIX_FAILURES[2:]:
        assert failure_class(kind).retryable is True, f"{kind} 属传输层，值得退避重试"


def test_retryable_kinds_is_derived_from_the_failure_table_not_recopied():
    """退避名单必须是分类表的投影：两处各写一份，加一种失败就会漏改一处。"""
    assert set(RETRYABLE_KINDS) == set(RETRYABLE_FAILURE_KINDS)
    assert set(RETRYABLE_KINDS) == {k for k, s in FAILURE_CLASSES.items() if s.retryable}
    assert "http_4xx" not in RETRYABLE_KINDS
    assert "http_unexpected" not in RETRYABLE_KINDS, "看不懂的状态码不该被盲目重发"


def test_half_open_is_a_class_of_its_own_and_flags_peer_may_have_applied(stub, repos):
    """半开连接：对端把连接掐了却没给应答——请求多半已经送到对端了。

    这条是幂等键不可省的直接证据：它既值得重试，又不能保证重发不会在对端变成两条。
    """
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)
    spec = failure_class("half_open")
    assert spec.retryable is True
    assert spec.peer_may_have_applied is True
    stub.queue_half_close(1)
    stub.set_default_status(201)
    sleeper = _RecordingSleeper()

    outcome = deliver_event(
        sessions, events, _stub_client(stub), event_row["event_id"], sleeper=sleeper,
        audit=audit, request_id=REQUEST_ID,
    )

    assert outcome.attempt_kinds[0] == "half_open"
    assert outcome.kind_counts["half_open"] == 1
    assert outcome.attempts == 2, "半开连接应退避重试一次"
    assert outcome.writeback_status == "sent", "重试拿到应答就不该留欠账"
    assert stub.call_count == 2, "替身没真收到两次请求，说明不是走了真 socket"
    # 重试时幂等键必须与第一次完全一致：对端靠它认出「这是同一条，不是第二条」。
    keys = {request.idempotency_key for request in stub.requests}
    assert keys == {event_row["client_event_id"]}


def test_half_open_exhausted_lands_pending_and_says_peer_may_have_applied(stub, repos):
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)
    stub.queue_half_close(MAX_RETRIES + 1)
    sleeper = _RecordingSleeper()

    outcome = deliver_event(
        sessions, events, _stub_client(stub), event_row["event_id"], sleeper=sleeper,
        audit=audit, request_id=REQUEST_ID,
    )

    assert outcome.attempts == MAX_RETRIES + 1
    assert outcome.writeback_status == "pending_writeback"
    assert outcome.kind_counts == {"half_open": MAX_RETRIES + 1}
    assert outcome.peer_may_have_applied is True
    assert "kind=half_open" in (outcome.last_error or "")
    assert f"attempts={MAX_RETRIES + 1}" in (outcome.last_error or "")
    stored = events.get(event_row["event_id"])
    assert stored["writeback_status"] == "pending_writeback"
    assert stored["attempt_count"] == MAX_RETRIES + 1


@pytest.mark.parametrize("exc_type,kind", [
    (httpx.ConnectTimeout, "connect_timeout"),
    (httpx.ReadTimeout, "read_timeout"),
    (httpx.WriteTimeout, "write_timeout"),
    (httpx.PoolTimeout, "pool_timeout"),
    (httpx.RemoteProtocolError, "half_open"),
    (httpx.ReadError, "half_open"),
    (httpx.WriteError, "half_open"),
    (httpx.CloseError, "half_open"),
    (httpx.ConnectError, "connect_error"),
    (httpx.UnsupportedProtocol, "protocol_error"),
])
def test_transport_exceptions_map_to_distinct_failure_kinds(exc_type, kind):
    """建连阶段与应答阶段分开：处置完全不同，混成一种会让排查方向跑偏。"""
    assert classify_exception(exc_type("injected", request=None)) == kind
    assert is_retryable(kind) is FAILURE_CLASSES[kind].retryable


def test_connect_and_read_timeouts_are_counted_separately(repos):
    """「连接超时 1 次 + 读超时 1 次」不能被压成「失败 2 次」。

    前者查病区网络配置，后者查对端处理速度；分开数信息科才知道该找谁。
    """
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)
    client, transport = _injected_client(
        [httpx.ConnectTimeout, httpx.ReadTimeout, 201]
    )
    sleeper = _RecordingSleeper()

    outcome = deliver_event(
        sessions, events, client, event_row["event_id"], sleeper=sleeper,
        audit=audit, request_id=REQUEST_ID,
    )

    assert outcome.attempt_kinds == ["connect_timeout", "read_timeout", "ok"]
    assert outcome.kind_counts == {"connect_timeout": 1, "read_timeout": 1, "ok": 1}
    assert outcome.writeback_status == "sent" and outcome.attempts == 3
    assert sleeper.delays == [BACKOFF_BASE_SECONDS, BACKOFF_BASE_SECONDS * 4]
    # 每次重试都带着同一个幂等键——两类超时都可能「对端其实已经收到了」。
    assert len(transport.requests) == 3
    assert {request.headers["Idempotency-Key"] for request in transport.requests} == {
        event_row["client_event_id"]
    }


def test_read_timeout_alone_exhausts_into_pending(repos):
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)
    client, _ = _injected_client([httpx.ReadTimeout] * (MAX_RETRIES + 1))

    outcome = deliver_event(
        sessions, events, client, event_row["event_id"], sleeper=_RecordingSleeper(),
        audit=audit, request_id=REQUEST_ID,
    )

    assert outcome.kind_counts == {"read_timeout": MAX_RETRIES + 1}
    assert outcome.writeback_status == "pending_writeback"
    assert outcome.peer_may_have_applied is True


def test_connect_error_alone_exhausts_into_pending(repos):
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)
    client, _ = _injected_client([httpx.ConnectError] * (MAX_RETRIES + 1))

    outcome = deliver_event(
        sessions, events, client, event_row["event_id"], sleeper=_RecordingSleeper(),
        audit=audit, request_id=REQUEST_ID,
    )

    assert outcome.kind_counts == {"connect_error": MAX_RETRIES + 1}
    assert outcome.writeback_status == "pending_writeback"
    assert outcome.peer_may_have_applied is False, "连都没连上，对端不可能收到过"


def test_unexpected_status_code_is_not_retried(stub, repos):
    """3xx 落成最终响应（本服务不跟随重定向）：盲目重发只会再来一次看不懂。"""
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(302)
    sleeper = _RecordingSleeper()

    outcome = deliver_event(
        sessions, events, _stub_client(stub), event_row["event_id"], sleeper=sleeper,
        audit=audit, request_id=REQUEST_ID,
    )

    assert outcome.kind == "http_unexpected"
    assert outcome.attempts == 1 and sleeper.delays == []
    assert stub.call_count == 1, "跟随了重定向会把换液事件发到别处"
    assert outcome.writeback_status == failure_class("http_unexpected").terminal_status == "failed"


def test_client_does_not_inherit_the_workstation_system_proxy():
    """出站端点由配置明确给出，不该被工作站的系统代理悄悄接管。

    除了「病区事件被绕出院内网络」这条安全上的理由，更实际的一条是**失败分类会
    失真**：代理会把「对端掐了连接不给应答」统一翻成 502，于是半开连接与读超时
    全被记成对端 5xx，信息科照着错误方向排查病区网络。
    """
    client = NurseExecClient(base_url="https://nurse-exec.invalid/x", api_token=TOKEN)
    try:
        assert client._http().trust_env is False
    finally:
        client.close()


def test_protocol_error_is_not_retried(repos):
    """base_url 少写 http:// 属于配置问题：重试多少次都是同一条错误。"""
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)
    broken = NurseExecClient(
        base_url="nurse-exec.invalid/api/v1/infusion/events", api_token=TOKEN, timeout=1.0
    )
    sleeper = _RecordingSleeper()

    outcome = deliver_event(
        sessions, events, broken, event_row["event_id"], sleeper=sleeper,
        audit=audit, request_id=REQUEST_ID,
    )

    assert outcome.kind == "protocol_error"
    assert outcome.attempts == 1 and sleeper.delays == []
    assert outcome.writeback_status == "failed"


def test_business_rejection_reason_is_read_out_of_the_peer_body(repos):
    """4xx 时把**对端给的原因**抽出来，信息科才有得查。"""
    for body, expected in (
        ('{"detail":"unknown infusion_order_no"}', "unknown infusion_order_no"),
        ('{"message":"bag_seq already replaced"}', "bag_seq already replaced"),
        ('{"detail":[{"loc":["body","bed_no"],"msg":"field required","type":"value_error.missing"}]}',
         "field required（body/bed_no）"),
        ("opaque text from the gateway", "opaque text from the gateway"),
    ):
        response = httpx.Response(422, content=body.encode("utf-8"))
        assert business_rejection_reason(response) == expected
        response = httpx.Response(422, content=b"")
        assert business_rejection_reason(response) is None


def test_four_xx_records_peer_reason_and_does_not_retry(stub, repos):
    sessions, events, audit = repos
    record, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(422, body='{"detail":"unknown infusion_order_no"}')
    sleeper = _RecordingSleeper()

    outcome = deliver_event(
        sessions, events, _stub_client(stub), event_row["event_id"], sleeper=sleeper,
        audit=audit, request_id=REQUEST_ID,
    )

    assert outcome.writeback_status == "failed"
    assert outcome.attempts == 1 and sleeper.delays == []
    assert outcome.status_code == 422
    assert outcome.rejection_reason == "unknown infusion_order_no"
    assert outcome.kind_counts == {"http_4xx": 1}
    summary = outcome.last_error or ""
    assert "kind=http_4xx" in summary
    assert "status=422" in summary
    assert "attempts=1" in summary
    assert "counts=http_4xx=1" in summary
    assert "reason=unknown infusion_order_no" in summary
    rejected = [
        row for row in audit.list_by_target(record["session_id"])
        if row["action"] == "writeback.rejected"
    ]
    assert rejected[-1]["detail"]["rejection_reason"] == "unknown infusion_order_no"
    assert rejected[-1]["detail"]["status_code"] == 422


def test_five_xx_summary_carries_status_attempts_and_counts(stub, repos):
    sessions, events, audit = repos
    _, event_row = _confirmed_event(sessions, events)
    stub.set_default_status(503, body='{"detail":"upstream busy"}')

    outcome = deliver_event(
        sessions, events, _stub_client(stub), event_row["event_id"],
        sleeper=_RecordingSleeper(), audit=audit, request_id=REQUEST_ID,
    )

    assert outcome.writeback_status == "pending_writeback"
    assert outcome.kind_counts == {"http_5xx": MAX_RETRIES + 1}
    summary = outcome.last_error or ""
    assert "kind=http_5xx" in summary and "status=503" in summary
    assert f"attempts={MAX_RETRIES + 1}" in summary
    assert f"counts=http_5xx={MAX_RETRIES + 1}" in summary
    assert "last=http_5xx" in summary and "upstream busy" in summary


def _summary_of_echo(text: str) -> str:
    """把一段对端回显当作错误文本，交给摘要函数走一遍（脱敏在真实链路上也会走）。"""
    from infusion_monitor.adapters.nurse_exec import failure_summary

    return failure_summary(
        "http_4xx", status_code=401, attempts=1, reason=text, last_error=text, secrets=(TOKEN,)
    )


def test_failure_summary_never_carries_the_credential():
    """对端回显了 Bearer token（有的网关会），摘要里也不能有。"""
    echoed = f'{{"detail":"rejected Authorization: Bearer {TOKEN}"}}'
    summary = _summary_of_echo(echoed)
    assert TOKEN not in summary
    assert "***" in summary
    assert scrub(summary, TOKEN) == summary, "摘要自身已脱敏，外层再脱敏一次不应改变它"


def test_count_kinds_keeps_first_seen_order():
    assert count_kinds(["b", "a", "b", "c", "a", "b"]) == {"b": 3, "a": 2, "c": 1}
    assert count_kinds([]) == {}