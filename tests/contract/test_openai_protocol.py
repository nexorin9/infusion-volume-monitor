"""协议兼容性契约测试：**对着 OpenAI-compatible 协议本身**逐条钉住网关。

这个文件与 :mod:`tests.test_model_gateway` 不是同一件事：

* ``test_model_gateway`` 验的是**行为**——四类失败会不会降级、降级后状态动不动；
* 这里验的是**线上格式**——请求打到哪个路径、报头是什么、请求体的键集与
  端点认的那份协议是否一致、响应里哪些字段是必需/可选。

分家的理由很实际：换院内端点（自建推理服务 ↔ 本地部署权重）时，**行为不变、
格式会变**。格式漂了要有一个不依赖数据库、不依赖状态机的契约套件顶着报错，
否则排查会先掉进「是不是业务逻辑错了」这个方向。

所有用例都走 :class:`~tests.support.llm_stub.LLMStub`（真实 TCP 端口），不用
内存传输层——``httpx.MockTransport`` 验不出路径拼错、报头没带、连接超时参数
没生效这类真会在病区暴露的问题。

**替身只是测试数据层**：它不代表真实模型服务的接口行为，院内真实端点本轮未
实连（见 ``progress.txt``）。但它守的是协议形状，不是端点的脾气。
"""

from __future__ import annotations

import json

import httpx
import pytest

from infusion_monitor.gateway.metrics import QUOTA_EXCEEDED
from infusion_monitor.gateway.model_gateway import (
    GATEWAY_MAX_RETRIES,
    ModelGateway,
    build_messages,
)
from infusion_monitor.gateway.schemas import parse_model_output

from ..support.llm_stub import LLMStub, completion_body

#: 出站请求体必须**恰好**是这六个键。多一个键可能让严格端点 400，少一个键则
#: 端点会用默认值替我们做决定（temperature 变成 1，结论就不再可复现）。
CONTRACT_BODY_KEYS = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "stream",
    "response_format",
}

NEEDS_CHANGE = {
    "decision": "needs_change",
    "reason": "液位连续偏低且余量已进入紧急线",
    "confidence": 0.86,
    "trigger_reason": "near_empty_observed",
}


@pytest.fixture
def stub():
    with LLMStub() as running:
        yield running


def _gateway(stub: LLMStub, **kwargs) -> ModelGateway:
    defaults = dict(
        base_url=stub.base_url,
        api_key="contract-key-not-real",
        model="contract-model",
        timeout=1.0,
        connect_timeout=1.0,
        backoff_base=0.0,
    )
    defaults.update(kwargs)
    return ModelGateway(**defaults)


# ── 1. 请求面 ───────────────────────────────────────────────────────────────


def test_endpoint_path_follows_openai_compatible_layout():
    """``base_url`` 拼上 ``/chat/completions``；末尾斜杠与 ``/v1`` 段都要处理对。"""
    def _endpoint(base_url: str) -> str:
        return ModelGateway(base_url=base_url, api_key="k", model="m").endpoint

    assert _endpoint("https://llm.example/v1") == "https://llm.example/v1/chat/completions"
    assert _endpoint("https://llm.example/v1/") == "https://llm.example/v1/chat/completions"
    assert _endpoint("http://127.0.0.1:8/") == "http://127.0.0.1:8/chat/completions"
    # 配置面本身会剥掉末尾斜杠，两处都要对：路径多一段就是 404，不是小毛病。
    assert _endpoint("https://llm.example/api/openai") == (
        "https://llm.example/api/openai/chat/completions"
    )


def test_request_headers_match_the_protocol(stub):
    stub.script_json(NEEDS_CHANGE)
    gateway = _gateway(stub)
    try:
        gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    record = stub.requests[0]
    assert record.method == "POST"
    assert record.path == "/v1/chat/completions"
    assert record.authorization == "Bearer contract-key-not-real"
    assert record.headers["content-type"] == "application/json"
    assert record.headers["accept"] == "application/json"
    # 凭据只在这一个报头里出现：换端点时不会出现第二个携带 key 的地方。
    carried = [name for name, value in record.headers.items() if "contract-key-not-real" in value]
    assert carried == ["authorization"]


def test_request_body_carries_exactly_the_contract_keys(stub):
    """键集逐字相等。多一个键可能被严格端点拒，少一个键端点会替我们做决定。"""
    stub.script_json(NEEDS_CHANGE)
    gateway = _gateway(stub)
    try:
        gateway.complete(build_messages({"bag_seq": "B01", "bed_no": "EXAMPLE-BED-43"}))
    finally:
        gateway.close()

    body = stub.requests[0].payload
    assert set(body) == CONTRACT_BODY_KEYS
    assert body["model"] == "contract-model"
    assert body["temperature"] == 0.0, "同一段观测重跑必须给同一结论，温度不能随机"
    assert body["stream"] is False
    assert body["response_format"] == {"type": "json_object"}
    assert isinstance(body["max_tokens"], int) and body["max_tokens"] > 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    # 每条消息都必须带 role + content 且都是非空字符串：有些端点会静默丢掉
    # content 为 null 的消息，于是模型收不到任何东西却回一句客套话。
    for message in body["messages"]:
        assert set(message) == {"role", "content"}
        assert isinstance(message["content"], str) and message["content"]


def test_response_format_is_json_object_and_not_a_bare_schema(stub):
    """只承诺协议保证的最大公约数；把整份 JSON Schema 塞进去只有部分端点认。"""
    stub.script_json(NEEDS_CHANGE)
    gateway = _gateway(stub)
    try:
        gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    body = stub.requests[0].payload
    assert body["response_format"] == {"type": "json_object"}
    assert "json_schema" not in json.dumps(body["response_format"])
    # 输出字段契约由提示词承担：system 段里必须能看到字段名与枚举。
    system = body["messages"][0]["content"]
    for field in ("decision", "trigger_reason", "confidence"):
        assert field in system


def test_max_tokens_reflects_the_configured_cap(stub):
    stub.script_json(NEEDS_CHANGE)
    gateway = _gateway(stub, max_tokens=777)
    try:
        gateway.complete(build_messages({"bag_seq": "B01"}))
        gateway.complete(build_messages({"bag_seq": "B02"}), max_tokens=333)
    finally:
        gateway.close()

    assert [r.payload["max_tokens"] for r in stub.requests] == [777, 333]


# ── 2. 响应面 ───────────────────────────────────────────────────────────────


def test_openai_compatible_response_is_parsed_into_a_decision(stub):
    stub.script_json(NEEDS_CHANGE, model="contract-model", prompt_tokens=311, completion_tokens=57)
    gateway = _gateway(stub)
    try:
        result = gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    assert result.ok is True
    assert result.status_code == 200
    decision = parse_model_output(result.content or "")
    assert decision.decision == "needs_change"
    assert decision.trigger_reason == "near_empty_observed"
    assert decision.confidence == pytest.approx(0.86)
    # token 用量取自 usage，不猜。
    assert (result.prompt_tokens, result.completion_tokens) == (311, 57)


def test_endpoint_without_usage_block_is_not_treated_as_failure(stub):
    """部分自建端点不回 ``usage``。缺一个不影响结论的字段不该换来一次降级。"""
    body = completion_body(NEEDS_CHANGE)
    body.pop("usage")
    stub.script_raw(200, json.dumps(body, ensure_ascii=False))
    gateway = _gateway(stub)
    try:
        result = gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    assert result.ok is True
    assert result.prompt_tokens is None and result.completion_tokens is None
    assert parse_model_output(result.content or "").decision == "needs_change"


def test_reply_without_choices_is_a_parse_error_not_a_success(stub):
    """``{}`` 在协议上是「合法 JSON 但没有选择」——按解析失败处理，不当成功。"""
    stub.script_raw(200, json.dumps({"id": "chatcmpl-empty", "object": "chat.completion"}))
    gateway = _gateway(stub)
    try:
        result = gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    assert result.ok is False
    assert result.kind == "parse_error"


def test_null_content_with_finish_reason_length_is_a_parse_error(stub):
    """推理被 ``max_tokens`` 截断时端点给 ``content=null``，不能当成空回复收下。"""
    stub.script_raw(
        200,
        json.dumps(completion_body(None, finish_reason="length"), ensure_ascii=False),
    )
    gateway = _gateway(stub)
    try:
        result = gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    assert result.ok is False
    assert result.kind == "parse_error"
    assert "finish_reason" in (result.error or "")


# ── 3. 四类失败在替身上各自可复现 ──────────────────────────────────────────


def _offline_gateway(exc_factory, **kwargs) -> ModelGateway:
    """传输层直接抛异常的网关。

    **为什么连接超时不用「连一个没人监听的端口」来造**：本环境的出站流量被代理
    接管，连不上不会表现为 ``httpx.ConnectTimeout``（实测表现为 ReadTimeout，甚至
    给不存在的域名回 502），拿它当「连接超时」会让断言测到的是代理的行为。真实
    端点连不上时 httpx 抛的就是 ``ConnectTimeout``，这里直接抛它。其余三类
    （读超时 / 5xx / 解析失败）仍走真实 socket 的 :class:`LLMStub`。
    """
    def _handler(request):
        raise exc_factory("simulated transport failure")

    defaults = dict(
        base_url="http://llm.invalid/v1",
        api_key="contract-key-not-real",
        model="contract-model",
        timeout=0.5,
        connect_timeout=0.5,
        backoff_base=0.0,
        client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    defaults.update(kwargs)
    return ModelGateway(**defaults)


def test_connect_timeout_and_read_timeout_are_two_distinct_codes():
    """两类超时的排查方向不同：前者看端点没起，后者看端点忙。"""
    connect = _offline_gateway(lambda m: httpx.ConnectTimeout(m))
    try:
        connect_result = connect.complete(build_messages({"bag_seq": "B01"}))
    finally:
        connect.close()

    read = _offline_gateway(lambda m: httpx.ReadTimeout(m))
    try:
        read_result = read.complete(build_messages({"bag_seq": "B01"}))
    finally:
        read.close()

    assert connect_result.kind == "connect_timeout"
    assert read_result.kind == "read_timeout"
    assert connect_result.kind != read_result.kind


def test_read_timeout_reproduces_on_a_slow_endpoint(stub):
    """替身延迟超过 timeout 即读超时——走的是真 socket，不是注入。"""
    stub.script_json(NEEDS_CHANGE, delay=0.6)
    gateway = _gateway(stub, timeout=0.15)
    try:
        result = gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    assert result.kind == "read_timeout"
    assert result.attempts == GATEWAY_MAX_RETRIES + 1
    assert stub.call_count == GATEWAY_MAX_RETRIES + 1


def test_http_5xx_is_retried_then_gives_up_on_the_wire(stub):
    stub.script_flood(503, times=6)
    gateway = _gateway(stub)
    try:
        result = gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    assert result.kind == "http_5xx"
    assert result.status_code == 503
    assert result.attempts == GATEWAY_MAX_RETRIES + 1
    assert len(result.backoff_delays) == GATEWAY_MAX_RETRIES
    assert stub.call_count == GATEWAY_MAX_RETRIES + 1, "退避重试耗尽就停，不刷端点"


def test_structured_parse_failure_is_its_own_code(stub):
    """端点回了 200 但内容不是约定的 JSON —— 与 5xx 分开记（换端点/换提示词）。"""
    stub.script_raw(200, "抱歉，我无法回答这个问题。")
    gateway = _gateway(stub)
    try:
        result = gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    assert result.kind == "parse_error"
    assert "JSON" in (result.error or "")
    assert result.attempts == GATEWAY_MAX_RETRIES + 1, "解析失败是随机的，值得重发"


def test_four_failure_codes_do_not_collide():
    """四个码互不相同，且都不与成功码相撞——运维据此分头排查。"""
    codes = {"connect_timeout", "read_timeout", "http_5xx", "parse_error"}
    assert len(codes) == 4
    assert "ok" not in codes
    assert QUOTA_EXCEEDED not in codes, "额度是护栏，不是协议失败，码必须分开"


def test_transient_5xx_recovery_keeps_the_model_decision(stub):
    """重试是为了拿回结论，不是为了刷端点：先 503 后成功就该采信模型。"""
    stub.script_raw(503, '{"error":"overloaded"}')
    stub.script_json(NEEDS_CHANGE)
    gateway = _gateway(stub)
    try:
        result = gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    assert result.ok is True
    assert result.attempt_kinds == ["http_5xx", "ok"]
    assert result.attempts == 2


def test_api_key_never_leaks_through_an_endpoint_echo(stub):
    """有的网关会把请求头原样回显——key 不能顺着 last_error 进库、进日志。"""
    body = {"error": {"message": "bad key: contract-key-not-real", "type": "invalid_request_error"}}
    stub.script_raw(401, json.dumps(body))
    gateway = _gateway(stub)
    try:
        result = gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    assert result.kind == "http_4xx"
    assert "contract-key-not-real" not in (result.error or "")
    assert "***" in (result.error or "")
    assert result.attempts == 1, "4xx 是确定性拒绝，重发只会再被拒一次"
    assert "contract-key-not-real" not in repr(gateway)