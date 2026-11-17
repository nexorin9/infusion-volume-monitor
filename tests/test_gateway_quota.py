"""成本护栏与用量统计：模型不能悄悄烧钱，也不能悄悄变慢。

两组断言：

1. **超限就退阈值路径**（护栏）——每日调用上限、单请求 token 上限被越过时**一次
   HTTP 都不发**，落 ``quota_exceeded``，结论照常由确定性路径给出，且**不推进
   任何业务状态**。护栏只挡不改写：不做静默截断，审计里记的 ``max_tokens``
   与端点实际收到的必须一致。
2. **用量统计对得上**（观测）——调用数、token 用量、延迟分位、失败率按
   ``request_id`` 与模型名聚合，并出现在运行摘要里。

**为什么护栏默认就在**：模型不可用时系统照常工作（退阈值路径），这恰恰是最容易
没人发现「它一天调了八万次」的场景——功能全绿、账单在涨。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from infusion_monitor.gateway.metrics import (
    QUOTA_EXCEEDED,
    GatewayMetrics,
    LatencyWindow,
    QuotaGuard,
    QuotaPolicy,
    percentile,
)
from infusion_monitor.gateway.model_gateway import ModelGateway, build_messages
from infusion_monitor.identity.state_machine import ChangeParams, evaluate_change
from infusion_monitor.store import EventRepository, SessionRepository

from .support.llm_stub import LLMStub, completion_body

NEEDS_CHANGE = {
    "decision": "needs_change",
    "reason": "液位连续偏低且余量已进入紧急线",
    "confidence": 0.86,
    "trigger_reason": "near_empty_observed",
}

CST = timezone(timedelta(hours=8))


# ── 夹具 ────────────────────────────────────────────────────────────────────


@pytest.fixture
def stub():
    with LLMStub() as running:
        yield running


def _gateway(stub: LLMStub, *, quota=None, **kwargs) -> ModelGateway:
    defaults = dict(
        base_url=stub.base_url,
        api_key="quota-key-not-real",
        model="quota-model",
        timeout=1.0,
        connect_timeout=1.0,
        backoff_base=0.0,
        max_tokens=1200,
        metrics=GatewayMetrics(),
        quota=quota,
    )
    defaults.update(kwargs)
    return ModelGateway(**defaults)


def _observation():
    decision = evaluate_change(
        {
            "session_id": "sess-quota",
            "bag_seq": "B01",
            "state": "monitoring",
            "area_ratio_series": [0.08, 0.06, 0.05, 0.04, 0.03],
            "eta_minutes": 12,
            "estimated_volume_ml": 12.0,
        },
        ChangeParams(n_frames=3, change_threshold_ratio=0.15),
    )
    observation = {
        "session_id": "sess-quota",
        "bag_seq": "B01",
        "bed_no": "EXAMPLE-BED-43",
        "area_ratio_series": [0.08, 0.06, 0.05, 0.04, 0.03],
        "threshold_decision": decision.to_json(),
    }
    return observation, decision


def _seed_session(conn, **overrides) -> dict:
    payload = {
        "rack_id": "R-EXAMPLE-43",
        "bed_no": "EXAMPLE-BED-43",
        "infusion_order_no": "EXAMPLE-ORDER-0043",
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
    row = SessionRepository(conn).upsert_session(payload)
    sessions = SessionRepository(conn)
    sessions.update_area_series(row["session_id"], [0.08, 0.06, 0.05, 0.04, 0.03])
    return sessions.get(row["session_id"])


# ── 1. 每日调用上限 ──────────────────────────────────────────────────────────


def test_daily_call_limit_blocks_the_call_before_any_http(stub):
    """超限后一次请求都不发——护栏省下的正是这些请求。"""
    stub.script_json(NEEDS_CHANGE)
    guard = QuotaGuard(QuotaPolicy(daily_call_limit=2, max_tokens_per_request=0))
    gateway = _gateway(stub, quota=guard)
    observation, deterministic = _observation()
    try:
        first = gateway.advise(observation, deterministic=deterministic, request_id="r-1")
        second = gateway.advise(observation, deterministic=deterministic, request_id="r-2")
        blocked = gateway.advise(observation, deterministic=deterministic, request_id="r-3")
    finally:
        gateway.close()

    assert first.decision_source == "model_decided"
    assert second.decision_source == "model_decided"
    assert stub.call_count == 2, "第 3 次不该出网"

    assert blocked.decision_source == "threshold_fallback"
    assert blocked.fallback_reason == QUOTA_EXCEEDED
    assert blocked.usable is False
    assert blocked.attempts == 0
    assert blocked.attempt_kinds == []
    assert blocked.pending_manual_review is True
    # 护栏退的也是确定性结论：护士拿到的仍然是可核对的阈值判定。
    assert blocked.deterministic is deterministic
    assert stub.call_count == 2


def test_quota_block_reports_the_limit_and_how_to_lift_it(stub):
    guard = QuotaGuard(QuotaPolicy(daily_call_limit=1))
    stub.script_json(NEEDS_CHANGE)
    gateway = _gateway(stub, quota=guard)
    observation, deterministic = _observation()
    try:
        gateway.advise(observation, deterministic=deterministic, request_id="r-1")
        blocked = gateway.advise(observation, deterministic=deterministic, request_id="r-2")
    finally:
        gateway.close()

    # 报错文本要能直接让人去改配置，而不是只说一句「超限了」。
    assert blocked.error and "LLM_DAILY_CALL_LIMIT" in blocked.error
    assert "1" in blocked.error
    assert "quota-key-not-real" not in blocked.error


def test_quota_block_does_not_advance_business_state(stub, conn):
    """护栏不是模型，也不是业务动作：会话状态与出站事件表都必须纹丝不动。"""
    session = _seed_session(conn)
    guard = QuotaGuard(QuotaPolicy(daily_call_limit=1))
    stub.script_json(NEEDS_CHANGE)
    gateway = _gateway(stub, quota=guard)
    observation, deterministic = _observation()
    try:
        gateway.advise(observation, deterministic=deterministic, request_id="r-1")
        gateway.advise(observation, deterministic=deterministic, request_id="r-2")
    finally:
        gateway.close()

    assert SessionRepository(conn).get(session["session_id"])["state"] == "monitoring"
    assert EventRepository(conn).count() == 0


def test_daily_budget_resets_on_the_next_business_day(stub):
    """护栏按「今天还能花多少」服务，不按「这个进程一共花过多少」。"""
    clock = {"now": datetime(2026, 10, 23, 21, 30, tzinfo=CST)}
    guard = QuotaGuard(QuotaPolicy(daily_call_limit=1), clock=lambda: clock["now"])

    assert guard.authorize().allowed is True
    assert guard.authorize().allowed is False
    assert guard.as_dict()["business_day"] == "2026-10-23"
    assert guard.as_dict()["exhausted"] is True

    # 跨到院内次日零点后额度归零。
    clock["now"] = datetime(2026, 10, 24, 0, 5, tzinfo=CST)
    assert guard.usage_today() == 0
    assert guard.authorize().allowed is True
    assert guard.as_dict()["business_day"] == "2026-10-24"


def test_business_day_boundary_is_the_hospital_midnight_not_utc():
    """UTC 的日界落在病区凌晨两点。20:30+08:00 在 UTC 里已是「明天」。"""
    guard = QuotaGuard(QuotaPolicy(daily_call_limit=5), clock=lambda: datetime(2026, 10, 23, 20, 30, tzinfo=CST))
    assert guard.as_dict()["business_day"] == "2026-10-23"
    late = QuotaGuard(QuotaPolicy(daily_call_limit=5), clock=lambda: datetime(2026, 10, 23, 23, 59, tzinfo=CST))
    assert late.as_dict()["business_day"] == "2026-10-23"


def test_zero_limit_means_no_guard_not_zero_calls(stub):
    """限值 ≤ 0 表示不限制。把它当「一次都不能调」会让服务直接不干活。"""
    stub.script_json(NEEDS_CHANGE)
    guard = QuotaGuard(QuotaPolicy(daily_call_limit=0, max_tokens_per_request=0))
    gateway = _gateway(stub, quota=guard)
    observation, deterministic = _observation()
    try:
        results = [
            gateway.advise(observation, deterministic=deterministic, request_id=f"r-{i}")
            for i in range(5)
        ]
    finally:
        gateway.close()

    assert all(r.decision_source == "model_decided" for r in results)
    assert stub.call_count == 5
    assert guard.remaining_today() is None
    assert guard.as_dict()["exhausted"] is False


def test_gateway_without_quota_guard_is_unaffected(stub):
    """没挂护栏的网关（如单测手工构造）照常发请求——护栏是可选件，不是隐式开关。"""
    stub.script_json(NEEDS_CHANGE)
    gateway = _gateway(stub, quota=None)
    try:
        result = gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    assert result.ok is True
    assert stub.call_count == 1


# ── 2. 单请求 token 上限 ────────────────────────────────────────────────────


def test_per_request_token_cap_blocks_instead_of_silently_clamping(stub):
    """静默截断会让审计记的 max_tokens 与端点实际收到的不是一回事。"""
    stub.script_json(NEEDS_CHANGE)
    guard = QuotaGuard(QuotaPolicy(max_tokens_per_request=1200))
    gateway = _gateway(stub, quota=guard, max_tokens=2000)
    observation, deterministic = _observation()
    try:
        blocked = gateway.advise(observation, deterministic=deterministic, request_id="r-tokens")
    finally:
        gateway.close()

    assert blocked.fallback_reason == QUOTA_EXCEEDED
    assert blocked.attempts == 0
    assert stub.call_count == 0, "超上限的请求根本不该出网"
    assert "LLM_MAX_TOKENS_PER_REQUEST" in (blocked.error or "")
    assert "2000" in (blocked.error or "") and "1200" in (blocked.error or "")
    assert blocked.deterministic is deterministic


def test_explicit_over_cap_call_is_blocked_at_the_call_site(stub):
    """``complete(max_tokens=…)`` 越过上限同样挡住，而不是发出去。"""
    stub.script_json(NEEDS_CHANGE)
    guard = QuotaGuard(QuotaPolicy(max_tokens_per_request=1200))
    gateway = _gateway(stub, quota=guard)
    try:
        ok = gateway.complete(build_messages({"bag_seq": "B01"}), max_tokens=1200)
        over = gateway.complete(build_messages({"bag_seq": "B02"}), max_tokens=5000)
    finally:
        gateway.close()

    assert ok.ok is True
    assert over.ok is False
    assert over.kind == QUOTA_EXCEEDED
    assert stub.call_count == 1


def test_token_cap_boundary_is_inclusive(stub):
    """等于上限放行、大 1 个挡住：门限两侧都要有断言，否则调参时不知差在哪。"""
    stub.script_json(NEEDS_CHANGE)
    guard = QuotaGuard(QuotaPolicy(max_tokens_per_request=800))
    gateway = _gateway(stub, quota=guard)
    try:
        assert gateway.complete(build_messages({"bag_seq": "B01"}), max_tokens=800).ok is True
        assert gateway.complete(build_messages({"bag_seq": "B02"}), max_tokens=801).ok is False
    finally:
        gateway.close()
    assert stub.call_count == 1


def test_token_cap_does_not_consume_the_daily_budget(stub):
    """被 token 上限挡下的调用没花出去，不该占掉当天的调用名额。"""
    guard = QuotaGuard(QuotaPolicy(daily_call_limit=2, max_tokens_per_request=100))
    assert guard.authorize(max_tokens=100).allowed is True
    assert guard.authorize(max_tokens=5000).allowed is False
    assert guard.usage_today() == 1, "被挡的那次没有记账"
    assert guard.authorize(max_tokens=100).allowed is True
    assert guard.authorize(max_tokens=100).allowed is False, "额度只剩 1 次"


# ── 3. 用量与延迟统计 ───────────────────────────────────────────────────────


def test_token_usage_totals_are_exact(stub):
    """端点报多少就记多少，不估算、不四舍五入。"""
    stub.script_json(NEEDS_CHANGE, prompt_tokens=300, completion_tokens=50)
    stub.script_json(NEEDS_CHANGE, prompt_tokens=200, completion_tokens=40)
    metrics = GatewayMetrics()
    gateway = _gateway(stub, metrics=metrics)
    observation, deterministic = _observation()
    try:
        gateway.advise(observation, deterministic=deterministic, request_id="r-a")
        gateway.advise(observation, deterministic=deterministic, request_id="r-b")
    finally:
        gateway.close()

    assert metrics.prompt_tokens == 500
    assert metrics.completion_tokens == 90
    assert metrics.total_tokens == 590
    assert metrics.usage_missing == 0
    assert metrics.invocations == 2


def test_missing_usage_is_counted_not_invented(stub):
    """端点不回 usage 时，用量口径要**说没有**，不能补 0 冒充「没用」。"""
    body = completion_body(NEEDS_CHANGE)
    body.pop("usage")
    stub.script_raw(200, json.dumps(body, ensure_ascii=False))
    metrics = GatewayMetrics()
    gateway = _gateway(stub, metrics=metrics)
    try:
        result = gateway.complete(build_messages({"bag_seq": "B01"}))
    finally:
        gateway.close()

    assert result.ok is True
    assert metrics.prompt_tokens == 0 and metrics.completion_tokens == 0
    assert metrics.usage_missing == 2, "prompt 与 completion 两项各算一次缺失"


def test_failure_rate_and_kinds_are_separated(stub):
    """失败率要有分母，四类失败要能分开看（排查方向不同）。"""
    metrics = GatewayMetrics()
    gateway = _gateway(stub, metrics=metrics, max_retries=0)
    try:
        stub.script_json(NEEDS_CHANGE)
        gateway.complete(build_messages({"bag_seq": "B01"}), request_id="r-ok")
        stub.reset()
        stub.script_flood(503, times=3)
        gateway.complete(build_messages({"bag_seq": "B02"}), request_id="r-5xx")
        stub.reset()
        stub.script_raw(200, "不是 JSON")
        gateway.complete(build_messages({"bag_seq": "B03"}), request_id="r-parse")
    finally:
        gateway.close()

    assert metrics.invocations == 3
    assert metrics.calls_ok == 1 and metrics.calls_failed == 2
    assert metrics.failure_rate == pytest.approx(2 / 3, abs=1e-4)
    assert metrics.by_kind == {"ok": 1, "http_5xx": 1, "parse_error": 1}
    # 一次调用里的 N 次退避重试是「一次调用发了 N 个请求」，不是 N 次调用。
    assert metrics.requests_sent == 3


def test_metrics_aggregate_by_model_name(stub):
    """换端点/换模型名时用量要分开算——两个模型共用一个计数就查不出是谁烧的。"""
    first = GatewayMetrics()
    second = GatewayMetrics()
    stub.script_json(NEEDS_CHANGE, prompt_tokens=100, completion_tokens=10)
    gateway_a = _gateway(stub, metrics=first)
    gateway_b = _gateway(stub, metrics=second, model="another-model")
    try:
        gateway_a.complete(build_messages({"bag_seq": "B01"}))
        gateway_b.complete(build_messages({"bag_seq": "B02"}))
    finally:
        gateway_a.close()
        gateway_b.close()

    assert set(first.by_model) == {"quota-model"}
    assert set(second.by_model) == {"another-model"}
    assert first.by_model["quota-model"].total_tokens == 110
    assert second.by_model["another-model"].total_tokens == 110
    merged = GatewayMetrics()
    merged.note_result(model="m1", kind="ok", ok=True, requests_sent=1, latency_ms=10)
    merged.note_result(model="m2", kind="http_5xx", ok=False, requests_sent=1, latency_ms=40)
    assert set(merged.by_model) == {"m1", "m2"}
    assert merged.by_model["m2"].failure_rate == 1.0
    assert merged.by_model["m1"].failure_rate == 0.0


def test_metrics_aggregate_by_request_id(stub):
    """按 request_id 能把「这一轮巡视调了哪些模型结论」捞出来。"""
    stub.script_json(NEEDS_CHANGE)
    stub.script_flood(503, times=3)
    metrics = GatewayMetrics()
    gateway = _gateway(stub, metrics=metrics, max_retries=0)
    observation, deterministic = _observation()
    try:
        gateway.advise(observation, deterministic=deterministic, request_id="r-round-1")
        gateway.advise(observation, deterministic=deterministic, request_id="r-round-2")
    finally:
        gateway.close()

    decided = metrics.request_stats("r-round-1")
    degraded = metrics.request_stats("r-round-2")
    assert decided["decision_source"] == "model_decided"
    assert decided["ok"] is True
    assert degraded["decision_source"] == "threshold_fallback"
    assert degraded["fallback_reason"] == "http_5xx"
    assert metrics.request_stats("r-never-happened") is None


def test_request_buckets_are_bounded():
    """LRU 上限：病区这台机器还要跑视觉，不能让统计吃满内存。"""
    metrics = GatewayMetrics(request_buckets=4)
    for index in range(10):
        metrics.note_result(
            model="m", kind="ok", ok=True, requests_sent=1, latency_ms=index, request_id=f"r{index}"
        )

    assert metrics.request_stats("r0") is None, "最早的应被挤出"
    assert metrics.request_stats("r9") is not None
    snapshot = metrics.snapshot()
    assert snapshot["tracked_requests"] == 4
    assert snapshot["request_bucket_limit"] == 4
    # 计数仍然是精确累计的——只有明细被挤出。
    assert snapshot["invocations"] == 10


def test_latency_percentiles_are_nearest_rank_over_the_window():
    """分位取真实出现过的值：运维拿它去对照日志能对上。"""
    window = LatencyWindow(maxlen=1024)
    for value in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        window.add(value)

    stats = window.as_dict()
    assert stats["count"] == 10
    assert stats["p50_ms"] == 50
    assert stats["p95_ms"] == 100
    assert stats["max_ms"] == 100
    assert stats["mean_ms"] == 55.0
    assert percentile([], 0.95) is None
    assert percentile([7], 0.95) == 7


def test_latency_window_is_bounded_but_counts_stay_exact():
    window = LatencyWindow(maxlen=8)
    for value in range(20):
        window.add(value)

    assert len(window) == 8
    assert window.as_dict()["max_ms"] == 19, "窗口里是最新的那 8 个"
    assert window.as_dict()["count"] == 8


def test_slow_endpoint_shows_up_in_p95_not_in_mean():
    """端点从 200ms 变成 4s：均值与 p50 都被快请求压住，只有高分位会跟着涨。"""
    metrics = GatewayMetrics()
    for _ in range(94):
        metrics.note_result(model="m", kind="ok", ok=True, requests_sent=1, latency_ms=200)
    for _ in range(6):
        metrics.note_result(model="m", kind="ok", ok=True, requests_sent=1, latency_ms=4000)

    latency = metrics.snapshot()["latency"]
    assert latency["p50_ms"] == 200
    assert latency["p95_ms"] == 4000
    assert latency["max_ms"] == 4000
    assert latency["mean_ms"] == 428.0


def test_quota_blocks_are_visible_in_metrics(stub):
    """护栏省下的请求要能被看见——「没花」也是一个可核对的事实。"""
    stub.script_json(NEEDS_CHANGE)
    guard = QuotaGuard(QuotaPolicy(daily_call_limit=1))
    metrics = GatewayMetrics()
    gateway = _gateway(stub, quota=guard, metrics=metrics)
    observation, deterministic = _observation()
    try:
        gateway.advise(observation, deterministic=deterministic, request_id="r-1")
        gateway.advise(observation, deterministic=deterministic, request_id="r-2")
    finally:
        gateway.close()

    snapshot = metrics.snapshot()
    assert snapshot["invocations"] == 2
    assert snapshot["requests_sent"] == 1, "被挡的那次没有出网"
    assert snapshot["quota_blocked"] == 1
    assert snapshot["by_kind"][QUOTA_EXCEEDED] == 1
    assert snapshot["by_decision_source"] == {"model_decided": 1, "threshold_fallback": 1}
    assert snapshot["by_model"]["quota-model"]["quota_blocked"] == 1


def test_threshold_only_call_is_not_counted_as_a_model_invocation(stub):
    """压根没调模型的一轮不该计进「模型调用数」——那个数是给算账用的。"""
    metrics = GatewayMetrics()
    gateway = _gateway(stub, metrics=metrics, quota=QuotaGuard(QuotaPolicy(daily_call_limit=1)))
    observation, deterministic = _observation()
    try:
        advice = gateway.advise(
            observation, deterministic=deterministic, request_id="r-only", call_model=False
        )
    finally:
        gateway.close()

    assert advice.decision_source == "threshold_only"
    assert metrics.invocations == 0
    assert metrics.requests_sent == 0
    assert metrics.snapshot()["by_decision_source"] == {"threshold_only": 1}
    assert metrics.request_stats("r-only")["decision_source"] == "threshold_only"
    # 护栏不该被压根没发生的调用扣掉额度。
    assert gateway.quota.usage_today() == 0
    assert stub.call_count == 0


# ── 4. 进运行摘要 ───────────────────────────────────────────────────────────


def test_summary_exposes_runtime_metrics_and_quota(client, stub):
    """运行摘要要能回答「今天烧了多少、还剩多少额度」。"""
    stub.script_json(NEEDS_CHANGE)
    gateway = client.app.state.model_gateway
    # 把应用里的网关指向本地替身跑一次真调用（替身是测试数据层，不是最终对接边界）。
    gateway.base_url = stub.base_url
    gateway.client = None
    observation, deterministic = _observation()
    try:
        gateway.advise(observation, deterministic=deterministic, request_id="r-summary")
    finally:
        gateway.close()

    body = client.get("/api/v1/summary").json()
    model = body["model"]
    assert model["runtime_metrics"]["invocations"] == 1
    assert model["runtime_metrics"]["requests_sent"] == 1
    assert model["runtime_metrics"]["total_tokens"] > 0
    assert model["runtime_metrics"]["latency"]["count"] >= 1
    assert model["runtime_metrics"]["by_decision_source"]["model_decided"] == 1
    assert model["quota"]["policy"]["daily_call_limit"] == 2000
    assert model["quota"]["used_today"] == 1
    assert model["quota"]["remaining_today"] == 1999
    assert model["quota"]["exhausted"] is False
    assert model["quota"]["business_day"]


def test_summary_quota_exhaustion_is_visible(client, stub):
    """额度耗尽后摘要要能一眼看到，否则「模型为什么一直在降级」无从查起。"""
    gateway = client.app.state.model_gateway
    gateway.quota.policy = QuotaPolicy(daily_call_limit=1, max_tokens_per_request=0)
    stub.script_json(NEEDS_CHANGE)
    gateway.base_url = stub.base_url
    gateway.client = None
    observation, deterministic = _observation()
    try:
        gateway.advise(observation, deterministic=deterministic, request_id="r-1")
        gateway.advise(observation, deterministic=deterministic, request_id="r-2")
    finally:
        gateway.close()

    model = client.get("/api/v1/summary").json()["model"]
    assert model["quota"]["exhausted"] is True
    assert model["quota"]["remaining_today"] == 0
    assert model["runtime_metrics"]["quota_blocked"] == 1
    assert model["runtime_metrics"]["by_kind"][QUOTA_EXCEEDED] == 1


def test_summary_never_leaks_the_api_key(client):
    """运行摘要是给运维看的，凭据不能出现在里面。"""
    body = client.get("/api/v1/summary").text
    assert "test-key-not-real" not in body
    assert "test-token-not-real" not in body
    assert "test-password-not-real" not in body
    # 掩码视图仍在：运维要能看到「配了、但看不到明文」。
    assert "llm_api_key" in body