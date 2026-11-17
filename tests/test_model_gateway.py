"""M6 模型网关的验收测试：协议兼容性、四类失败各自分类降级、状态不被推进。

六组断言：

1. **协议兼容**：请求真的按 OpenAI-compatible 打到 ``{base}/chat/completions``，
   报头带 Bearer，请求体形状是 ``model`` / ``messages`` / ``temperature=0`` /
   ``max_tokens`` / ``stream=false`` / ``response_format={"type":"json_object"}``；
   响应能解析出结论与 token 用量；
2. **四类失败各自可区分**：连接超时 / 读超时 / 5xx / 输出不可解析，各自落独立的
   ``fallback_reason``，且都是「首次 + ≤2 次重试」后降级；
3. **重试是值得的**：5xx 后重发成功时，采信模型结论而不是降级；
4. **模型不得推进业务状态**：无论模型可用、不可用、还是它想输出 ``replaced``，
   会话状态与出站事件表都纹丝不动；
5. **审计可核对**：每次调用留一行 ``inference_audit``，来源、降级原因、token、
   延迟都对得上；凭据不进错误文本；
6. **不发模型也能跑**：``call_model=False`` 走 ``threshold_only``，一次 HTTP 都不发。
"""

from __future__ import annotations

import pytest

from infusion_monitor.gateway.audit import GatewayAuditor, prompt_digest
from infusion_monitor.gateway.model_gateway import (
    GATEWAY_MAX_RETRIES,
    ModelGateway,
    build_messages,
    classify_response,
)
from infusion_monitor.gateway.schemas import (
    CONFIDENCE_FLOOR,
    DECISIONS,
    OUTPUT_SCHEMA,
    OUTPUT_TRIGGER_REASONS,
    ModelDecision,
    ModelOutputRejected,
    contract_markdown,
    parse_model_output,
)
from infusion_monitor.identity.state_machine import ChangeParams, evaluate_change
from infusion_monitor.store import EventRepository, InferenceAuditRepository, SessionRepository

from .support.llm_stub import LLMStub, completion_body

NEEDS_CHANGE = {
    "decision": "needs_change",
    "reason": "液位连续偏低且余量已进入紧急线",
    "confidence": 0.86,
    "trigger_reason": "near_empty_observed",
}
NO_CHANGE = {
    "decision": "no_change",
    "reason": "余量充足，本轮不用换",
    "confidence": 0.91,
    "trigger_reason": None,
}


# ── 夹具 ────────────────────────────────────────────────────────────────────


@pytest.fixture
def stub():
    with LLMStub() as running:
        yield running


def _gateway(stub: LLMStub, **kwargs) -> ModelGateway:
    defaults = dict(
        base_url=stub.base_url,
        api_key="test-key-not-real",
        model="test-model",
        timeout=1.0,
        connect_timeout=1.0,
        backoff_base=0.0,
    )
    defaults.update(kwargs)
    return ModelGateway(**defaults)


def _session_payload(**overrides):
    payload = {
        "rack_id": "R-EXAMPLE-12",
        "bed_no": "EXAMPLE-BED-12",
        "infusion_order_no": "EXAMPLE-ORDER-0012",
        "shift_key": "2026-10-23-AM",
        "bag_seq": "B01",
        "slot_index": 0,
        "bag_type_code": "TYPE-A",
        "frames_dir": "/tmp/frames",
        "frame_count": 16,
        "captured_at": "2026-10-23T09:41:00+08:00",
        "state": "monitoring",
    }
    payload.update(overrides)
    return payload


def _seed_session(conn, **overrides) -> dict:
    row = SessionRepository(conn).upsert_session(_session_payload(**overrides))
    sessions = SessionRepository(conn)
    sessions.update_area_series(row["session_id"], [0.08, 0.06, 0.05, 0.04, 0.03])
    return sessions.get(row["session_id"])


# ── 1. 协议兼容性 ───────────────────────────────────────────────────────────


def test_request_hits_openai_compatible_endpoint_with_bearer(stub):
    stub.script_json(NO_CHANGE)
    gateway = _gateway(stub)
    try:
        gateway.complete(build_messages({"bag_seq": "B01", "bed_no": "EXAMPLE-BED-12"}))
    finally:
        gateway.close()

    assert stub.call_count == 1
    record = stub.requests[0]
    assert record.method == "POST"
    assert record.path == "/v1/chat/completions"
    assert record.authorization == "Bearer test-key-not-real"
    assert record.headers["content-type"] == "application/json"


def test_request_body_matches_openai_compatible_shape(stub):
    stub.script_json(NO_CHANGE)
    gateway = _gateway(stub)
    try:
        gateway.complete(build_messages({"bag_seq": "B01"}), max_tokens=888)
    finally:
        gateway.close()

    body = stub.requests[0].payload
    assert body["model"] == "test-model"
    assert body["response_format"] == {"type": "json_object"}
    assert body["stream"] is False
    assert body["temperature"] == 0.0, "同一段观测重跑必须给同一结论，温度不能是随机的"
    assert body["max_tokens"] == 888
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    system = body["messages"][0]["content"]
    # 输出契约写在提示词里：协议只保证「是个 JSON 对象」，字段由提示词约束
    assert "needs_change" in system and "replaced" in system
    assert 'confidence' in system and 'trigger_reason' in system


def test_response_is_parsed_into_decision_and_usage(stub):
    stub.script_json(NEEDS_CHANGE, prompt_tokens=311, completion_tokens=57)
    gateway = _gateway(stub)
    try:
        result = gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    assert result.ok is True
    assert result.kind == "ok"
    assert result.status_code == 200
    assert result.prompt_tokens == 311
    assert result.completion_tokens == 57
    decision = parse_model_output(result.content)
    assert decision.decision == "needs_change"
    assert decision.trigger_reason == "near_empty_observed"
    assert decision.usable is True


def test_app_exposes_exactly_one_gateway(client):
    """业务层只有一个模型入口：``app.state.model_gateway``。

    多一个入口就等于把「换端点只改配置」这件事又漏一处——而且没人会发现是哪一处。
    """
    gateway = client.app.state.model_gateway
    assert isinstance(gateway, ModelGateway)
    assert gateway is client.app.state.model_gateway
    assert gateway.model == "test-model"
    assert gateway.auditor is not None, "网关必须自带推理审计，否则来源查不到"
    assert gateway.describe()["endpoint"] == "https://llm.test-hospital.invalid/v1/chat/completions"


def test_endpoint_derives_from_base_url():
    gateway = ModelGateway(base_url="https://llm.example.invalid/v1/", api_key="k", model="m")
    assert gateway.endpoint == "https://llm.example.invalid/v1/chat/completions"


def test_gateway_is_built_from_settings(credential_env):
    from infusion_monitor.config import Settings

    settings = Settings()
    gateway = ModelGateway.from_settings(settings)
    assert gateway.base_url == "https://llm.test-hospital.invalid/v1"
    assert gateway.endpoint.endswith("/chat/completions")
    assert gateway.model == "test-model"
    assert gateway.timeout == 8.0
    # 网关对象会进日志/异常栈，repr 里不能有明文 key
    assert "test-key-not-real" not in repr(gateway)
    assert gateway.describe()["api_key"] == "***"
    assert gateway.headers()["Authorization"] == "Bearer test-key-not-real"


def test_classify_response_separates_4xx_from_5xx():
    class _R:
        def __init__(self, code):
            self.status_code = code

    assert classify_response(_R(200)) == (True, "ok")
    assert classify_response(_R(429)) == (False, "http_4xx")
    assert classify_response(_R(503)) == (False, "http_5xx")
    assert classify_response(_R(101)) == (False, "http_unexpected")


# ── 2. 四类失败各自降级 ────────────────────────────────────────────────────


def _offline_gateway(exc_factory, **kwargs) -> ModelGateway:
    """构造一个传输层直接抛异常的网关。

    **为什么要注入传输层而不是连一个没人监听的端口**：本环境的出站流量被代理
    接管，连接被拒不会表现为 ``httpx.ConnectError``（实测表现为 ReadTimeout，甚至
    给不存在的域名回 502），拿它当「连接失败」会让断言测到的是代理的行为而不是
    我们的分类。真实端点连不上时 httpx 抛的就是 :class:`httpx.ConnectError`，这里
    直接抛它，把分类逻辑钉在真实异常类型上。其余三类（5xx / 读超时 / 解析失败）
    仍走真实 socket 的 :class:`LLMStub`。
    """
    import httpx

    def _handler(request):
        raise exc_factory("simulated transport failure")

    defaults = dict(
        base_url="http://llm.invalid/v1",
        api_key="test-key-not-real",
        model="test-model",
        timeout=0.5,
        connect_timeout=0.5,
        backoff_base=0.0,
        client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    defaults.update(kwargs)
    return ModelGateway(**defaults)


def test_connect_error_degrades_with_its_own_reason(stub):
    """连都连不上（端点没起）与连上了但超时是两种排查方向，码必须分开。"""
    import httpx

    gateway = _offline_gateway(lambda m: httpx.ConnectError(m))
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-connect")
    finally:
        gateway.close()

    assert advice.decision_source == "threshold_fallback"
    assert advice.fallback_reason == "connect_error"
    assert advice.attempts == GATEWAY_MAX_RETRIES + 1
    assert advice.pending_manual_review is True
    assert advice.usable is False
    assert advice.deterministic is deterministic


def test_connect_timeout_has_its_own_reason(stub):
    """连不上与连上了没等到回，必须是两个码（前者看端点没起，后者看端点忙）。"""
    import httpx

    gateway = _offline_gateway(lambda m: httpx.ConnectTimeout(m))
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-ctimeout")
    finally:
        gateway.close()

    assert advice.fallback_reason == "connect_timeout"


def test_read_timeout_degrades_with_its_own_reason(stub):
    stub.script_json(NEEDS_CHANGE, delay=0.6)
    gateway = _gateway(stub, timeout=0.15)
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-read")
    finally:
        gateway.close()

    assert advice.decision_source == "threshold_fallback"
    assert advice.fallback_reason == "read_timeout"
    assert advice.attempts == GATEWAY_MAX_RETRIES + 1
    assert len(advice.attempt_kinds) == GATEWAY_MAX_RETRIES + 1
    assert stub.call_count == GATEWAY_MAX_RETRIES + 1


def test_http_5xx_degrades_after_two_retries(stub):
    stub.script_flood(503, times=5)
    gateway = _gateway(stub)
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-5xx")
    finally:
        gateway.close()

    assert advice.decision_source == "threshold_fallback"
    assert advice.fallback_reason == "http_5xx"
    assert advice.attempts == GATEWAY_MAX_RETRIES + 1, "首次 + 2 次重试"
    assert advice.attempt_kinds == ["http_5xx"] * 3
    assert len(advice.backoff_delays) == GATEWAY_MAX_RETRIES
    assert stub.call_count == 3, "退避重试耗尽就停，不再无限刷端点"


def test_parse_error_degrades_and_is_distinguishable(stub):
    stub.script_raw(200, "抱歉，我无法回答这个问题。")
    gateway = _gateway(stub)
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-parse")
    finally:
        gateway.close()

    assert advice.decision_source == "threshold_fallback"
    assert advice.fallback_reason == "parse_error"
    assert advice.error and "JSON" in advice.error
    # 解析失败是随机的，值得重发一次
    assert advice.attempts == GATEWAY_MAX_RETRIES + 1


def test_four_failure_kinds_are_mutually_distinguishable(stub):
    """四类失败的 ``fallback_reason`` 必须互不相同——运维据此分头排查。"""
    import httpx

    seen = set()

    stub.script_flood(503, times=5)
    gateway = _gateway(stub)
    observation, deterministic = _observation_from_session()
    try:
        seen.add(gateway.advise(observation, deterministic=deterministic, request_id="r-a").fallback_reason)
    finally:
        gateway.close()

    stub.script_raw(200, "not json at all")
    gateway = _gateway(stub)
    try:
        seen.add(gateway.advise(observation, deterministic=deterministic, request_id="r-b").fallback_reason)
    finally:
        gateway.close()

    slow = LLMStub()
    slow.script_json(NO_CHANGE, delay=0.5)
    with slow:
        gateway = _gateway(slow, timeout=0.12)
        try:
            seen.add(gateway.advise(observation, deterministic=deterministic, request_id="r-c").fallback_reason)
        finally:
            gateway.close()

    unreachable = _offline_gateway(lambda m: httpx.ConnectError(m))
    try:
        seen.add(unreachable.advise(observation, deterministic=deterministic, request_id="r-d").fallback_reason)
    finally:
        unreachable.close()

    assert seen == {"http_5xx", "parse_error", "read_timeout", "connect_error"}


def test_http_4xx_is_not_retried(stub):
    """4xx 是确定性的业务拒绝：重发只会再被拒一次，不浪费端点。"""
    stub.script_flood(401, times=5)
    gateway = _gateway(stub)
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-4xx")
    finally:
        gateway.close()

    assert advice.fallback_reason == "http_4xx"
    assert advice.attempts == 1
    assert stub.call_count == 1


# ── 3. 重试恢复 ─────────────────────────────────────────────────────────────


def test_retry_recovers_after_transient_5xx(stub):
    stub.script_flood(503, times=1)
    stub.script_json(NEEDS_CHANGE)
    gateway = _gateway(stub)
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-recover")
    finally:
        gateway.close()

    assert stub.call_count == 2
    assert advice.decision_source == "model_decided"
    assert advice.fallback_reason is None
    assert advice.decision == "needs_change"
    assert advice.trigger_reason == "near_empty_observed"
    assert advice.usable is True
    assert advice.attempt_kinds == ["http_5xx", "ok"]


def test_model_may_say_no_change_and_that_is_accepted(stub):
    stub.script_json(NO_CHANGE)
    gateway = _gateway(stub)
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-no")
    finally:
        gateway.close()

    assert advice.decision_source == "model_decided"
    assert advice.decision == "no_change"
    assert advice.trigger_reason is None
    assert advice.usable is True


# ── 4. 模型不得推进业务状态 ─────────────────────────────────────────────────


def test_model_cannot_emit_replaced(stub):
    """模型输出 ``replaced`` 必须被拒——枚举里没有它，不是被忽略。"""
    stub.script_json({"decision": "replaced", "reason": "该换了", "confidence": 0.99})
    gateway = _gateway(stub)
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-replaced")
    finally:
        gateway.close()

    assert advice.decision_source == "threshold_fallback"
    assert advice.fallback_reason == "parse_error"
    assert advice.decision is None
    assert advice.usable is False
    assert "replaced" not in DECISIONS


@pytest.mark.parametrize(
    "payload",
    [
        {"decision": "replaced", "reason": "该换了", "confidence": 0.99},
        {"decision": "needs_change", "reason": "", "confidence": 0.9, "trigger_reason": "near_empty_observed"},
        {"decision": "needs_change", "reason": "换", "confidence": 1.8, "trigger_reason": "near_empty_observed"},
        {"decision": "needs_change", "reason": "换", "confidence": -0.1, "trigger_reason": "near_empty_observed"},
        {"decision": "needs_change", "reason": "换", "confidence": 0.9, "trigger_reason": "我编的码"},
        {"decision": "no_change", "reason": "ok", "confidence": 0.9, "unknown_field": 1},
        {"decision": "no_change", "confidence": 0.9},  # 缺 reason
    ],
)
def test_structured_output_is_validated_strictly(payload):
    """解析层就把不合契约的输出挡掉（不是捡认识的字段凑合用）。"""
    with pytest.raises(ModelOutputRejected):
        parse_model_output(payload)


@pytest.mark.parametrize("raw", ["", "   ", "抱歉，我无法回答", "{\"unclosed\": ", "[1,2,3]"])
def test_non_object_or_unparseable_replies_are_rejected(raw):
    with pytest.raises(ModelOutputRejected):
        parse_model_output(raw)


def test_missing_trigger_reason_parses_but_is_not_usable(stub):
    """缺 trigger_reason 在**可用性**层被拦，不在解析层。

    两层分开是有意的：``trigger_reason`` 可以合法地为 ``None``（``no_change`` 就是），
    所以它不是解析错误；但 ``needs_change`` 少了它，下游按码分支就没有可走的路，
    于是在 ``advise`` 里判为不可用并退阈值路径。
    """
    parsed = parse_model_output({"decision": "needs_change", "reason": "余量不足", "confidence": 0.9})
    assert parsed.usable is False

    stub.script_json({"decision": "needs_change", "reason": "余量不足", "confidence": 0.9})
    gateway = _gateway(stub)
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-notrigger")
    finally:
        gateway.close()

    assert advice.decision_source == "threshold_fallback"
    assert advice.usable is False
    assert advice.error and "incomplete_advice" in advice.error


def test_fenced_json_is_accepted(stub):
    """模型爱加 ```json 围栏——内容是好的，不该为一个围栏就丢掉结论。"""
    import json as _json

    fenced = "```json\n" + _json.dumps(NEEDS_CHANGE, ensure_ascii=False) + "\n```"
    stub.script_json(fenced)
    gateway = _gateway(stub)
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-fence")
    finally:
        gateway.close()

    assert advice.decision_source == "model_decided"
    assert advice.decision == "needs_change"


def test_low_confidence_is_not_taken_at_face_value(stub):
    stub.script_json(dict(NEEDS_CHANGE, confidence=0.2))
    gateway = _gateway(stub)
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-low")
    finally:
        gateway.close()

    assert advice.decision_source == "threshold_fallback"
    assert advice.usable is False
    assert advice.confidence is None
    assert CONFIDENCE_FLOOR > 0.2


def test_state_machine_state_not_advanced_by_any_gateway_path(stub, conn):
    """核心红线：网关的四条路径（可用 / 5xx / 解析失败 / 不调模型）都不动状态。"""
    stub.script_json(NEEDS_CHANGE)
    gateway = _gateway(stub, auditor=GatewayAuditor(InferenceAuditRepository(conn)))
    row = _seed_session(conn)
    deterministic = evaluate_change(row, ChangeParams(n_frames=3, change_threshold_ratio=0.15))
    observation = dict(row)
    observation["threshold_decision"] = deterministic.to_json()
    try:
        for call_model in (True, False):
            advice = gateway.advise(
                observation, deterministic=deterministic,
                request_id=f"r-state-{call_model}", call_model=call_model,
            )
            after = SessionRepository(conn).get(row["session_id"])
            assert after["state"] == "monitoring", "网关不得把 monitoring 推走"
            assert after.get("decision_source") in (None, "", "threshold_only")
    finally:
        gateway.close()

    assert EventRepository(conn).count() == 0, "模型不得写出站事件"


def test_degradation_leaves_state_and_event_table_untouched(stub, conn):
    """模型不可达时最危险的写法是「顺手把状态推进到 replaced」——这里守住它。"""
    stub.script_flood(503, times=5)
    gateway = _gateway(stub, auditor=GatewayAuditor(InferenceAuditRepository(conn)))
    row = _seed_session(conn)
    deterministic = evaluate_change(row, ChangeParams(n_frames=3, change_threshold_ratio=0.15))
    observation = dict(row)
    observation["threshold_decision"] = deterministic.to_json()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-degrade")
    finally:
        gateway.close()

    assert advice.decision_source == "threshold_fallback"
    assert advice.pending_manual_review is True
    after = SessionRepository(conn).get(row["session_id"])
    assert after["state"] == "monitoring"
    assert EventRepository(conn).count() == 0


# ── 5. 审计与凭据 ───────────────────────────────────────────────────────────


def test_inference_audit_records_call_metadata(stub, conn):
    stub.script_json(NEEDS_CHANGE, prompt_tokens=222, completion_tokens=33)
    gateway = _gateway(stub, auditor=GatewayAuditor(InferenceAuditRepository(conn)))
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(
            observation, deterministic=deterministic, request_id="r-audit",
            prompt_version="infusion-change-v1",
        )
    finally:
        gateway.close()

    rows = InferenceAuditRepository(conn).list_by_request("r-audit")
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "test-model"
    assert row["decision_source"] == "model_decided"
    assert row["prompt_version"] == "infusion-change-v1"
    assert row["prompt_hash"] == advice.prompt_hash and row["prompt_hash"]
    assert row["prompt_tokens"] == 222 and row["completion_tokens"] == 33
    assert row["latency_ms"] is not None and row["latency_ms"] >= 0
    assert row["decision"] == "needs_change"
    assert row["bag_seq"] == "B01"
    assert row["pending_manual_review"] == 0
    assert advice.audit_written is True


def test_inference_audit_records_fallback_reason(stub, conn):
    stub.script_flood(503, times=5)
    gateway = _gateway(stub, auditor=GatewayAuditor(InferenceAuditRepository(conn)))
    observation, deterministic = _observation_from_session()
    try:
        gateway.advise(observation, deterministic=deterministic, request_id="r-fb")
    finally:
        gateway.close()

    rows = InferenceAuditRepository(conn).list_by_request("r-fb")
    assert len(rows) == 1
    assert rows[0]["decision_source"] == "threshold_fallback"
    assert rows[0]["fallback_reason"] == "http_5xx"
    assert rows[0]["pending_manual_review"] == 1
    assert rows[0]["decision"] is None
    assert InferenceAuditRepository(conn).count_by_decision_source()["threshold_fallback"] == 1


def test_threshold_only_call_leaves_a_traceable_row(conn):
    """压根不调模型也要留痕：否则「这轮为什么没有模型意见」无从回答。"""
    gateway = _gateway(LLMStub(), auditor=GatewayAuditor(InferenceAuditRepository(conn)))
    observation, deterministic = _observation_from_session()
    advice = gateway.advise(
        observation, deterministic=deterministic, request_id="r-none", call_model=False
    )
    assert advice.decision_source == "threshold_only"
    assert advice.attempts == 0
    rows = InferenceAuditRepository(conn).list_by_request("r-none")
    assert rows[0]["decision_source"] == "threshold_only"
    assert rows[0]["fallback_reason"] is None


def test_api_key_never_appears_in_error_text(stub):
    secret = "sk-super-secret-value"
    stub.script_raw(401, '{"error":"unauthorized","echo":{"authorization":"Bearer ' + secret + '"}}')
    gateway = _gateway(stub, api_key=secret)
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-leak")
    finally:
        gateway.close()

    assert advice.error is not None
    assert secret not in advice.error
    assert "***" in advice.error


def test_prompt_digest_is_stable_and_sensitive():
    assert prompt_digest("abc") == prompt_digest("abc")
    assert prompt_digest("abc") != prompt_digest("abd")
    assert len(prompt_digest("abc")) == 16


def test_audit_failure_does_not_break_the_conclusion(stub, conn):
    """审计写不进去不能让结论跟着倒——但要如实标出来。"""
    class _Broken:
        def append(self, record):
            raise __import__("sqlite3").OperationalError("database is locked")

        def list_by_request(self, request_id):
            return []

        def list_recent(self, limit=50):
            return []

        def count(self):
            return 0

        def count_by_decision_source(self):
            return {}

    stub.script_json(NO_CHANGE)
    gateway = _gateway(stub, auditor=GatewayAuditor(_Broken()))
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(observation, deterministic=deterministic, request_id="r-broken")
    finally:
        gateway.close()

    assert advice.decision_source == "model_decided"
    assert advice.usable is True
    assert advice.audit_written is False, "写审计失败必须如实暴露，不能假装写上了"


# ── 6. 不调模型也能跑 ───────────────────────────────────────────────────────


def test_call_model_false_sends_no_http(stub):
    stub.script_json(NO_CHANGE)
    gateway = _gateway(stub)
    observation, deterministic = _observation_from_session()
    try:
        advice = gateway.advise(
            observation, deterministic=deterministic, request_id="r-off", call_model=False
        )
    finally:
        gateway.close()

    assert stub.call_count == 0, "threshold_only 必须一次 HTTP 都不发"
    assert advice.decision_source == "threshold_only"
    assert advice.decision is None
    assert advice.deterministic is deterministic
    assert advice.deterministic.target_state == "needs_change", "确定性结论照旧可用"


def test_deterministic_conclusion_is_always_carried(stub):
    """四条路径下 deterministic 都在：降级时它就是最终结论，不能是 None。"""
    row_payload = None
    for call_model in (True, False):
        stub.requests.clear()
        stub.script_json(NO_CHANGE)
        gateway = _gateway(stub)
        observation, deterministic = _observation_from_session()
        row_payload = deterministic
        try:
            advice = gateway.advise(
                observation, deterministic=deterministic, request_id=f"r-det-{call_model}",
                call_model=call_model,
            )
        finally:
            gateway.close()
        assert advice.deterministic is row_payload
        assert advice.to_json()["deterministic"] is not None


# ── 输出契约文档 ────────────────────────────────────────────────────────────


def test_contract_markdown_lists_every_model_field():
    doc = contract_markdown()
    for name in ("decision", "reason", "confidence", "trigger_reason"):
        assert f"`{name}`" in doc
    assert "无 replaced" in doc
    assert "/chat/completions" in doc
    assert "response_format" in doc


def test_contract_doc_is_in_sync_with_the_model():
    """``docs/llm-contract.md`` 与模型同源：改字段会失败，而不是让人肉比对一张表。"""
    from pathlib import Path

    doc_path = Path(__file__).resolve().parents[1] / "docs" / "llm-contract.md"
    assert doc_path.exists(), "输出契约文档缺失"
    assert contract_markdown() == doc_path.read_text(encoding="utf-8"), (
        "docs/llm-contract.md 与 OUTPUT_SCHEMA/ModelDecision 已漂移，请重新生成该文档"
    )


def test_schema_contract_and_pydantic_model_agree():
    """机读契约（发给端点 / 写进文档的）与 Pydantic 模型必须说同一件事。"""
    schema = OUTPUT_SCHEMA
    assert set(schema["properties"]) == set(ModelDecision.model_fields)
    assert set(schema["required"]) == {"decision", "reason", "confidence"}
    assert schema["properties"]["decision"]["enum"] == list(DECISIONS)
    assert schema["properties"]["trigger_reason"]["enum"] == list(OUTPUT_TRIGGER_REASONS) + [None]


def _observation_from_session() -> tuple[dict, object]:
    """不带库的最小观测（只用四类失败那几条路径，它们不关心落库）。"""
    decision = evaluate_change(
        {
            "session_id": "sess-gw",
            "bag_seq": "B01",
            "state": "monitoring",
            "area_ratio_series": [0.08, 0.06, 0.05, 0.04, 0.03],
            "eta_minutes": 12,
            "estimated_volume_ml": 12.0,
        },
        ChangeParams(n_frames=3, change_threshold_ratio=0.15),
    )
    observation = {
        "session_id": "sess-gw",
        "bag_seq": "B01",
        "bed_no": "EXAMPLE-BED-12",
        "area_ratio_series": [0.08, 0.06, 0.05, 0.04, 0.03],
        "threshold_decision": decision.to_json(),
    }
    return observation, decision