"""M6 决策融合仲裁层的验收测试：一致才采信，其余一律回落，状态永远不由模型推进。

六组断言：

1. **一致采信**：模型结论与阈值一致且置信达门禁 → 只采信它的**措辞**，结论与
   触发原因码仍取阈值路径；模型给的原因码不一致时被丢弃并留档；
2. **冲突回落**：阈值说该换、模型说不换（或反过来）→ 回落，``decision_source``
   如实从 ``model_decided`` 改写成 ``threshold_fallback``；
3. **低置信回落**：自评置信度低于**采信**门禁 → 回落，与网关那道**可用性**门禁
   是两道不同的门（同一个数、同一个来源，问的不是同一个问题）；
4. **端点不可达回落**：5xx / 解析失败 / 没调模型各自落自己的原因码；
5. **状态红线**：四条路径调完之后会话恒为 ``monitoring``、出站事件表恒 0 行，
   且 :func:`apply_to_state` 的返回值里永远不可能有 ``replaced``；
6. **仲裁可解释**：采信与回落各落一行 append-only 审计，detail 里带着原因码、
   双方结论与置信度——护士长与信息科复核「这袋为什么这么说」看的就是它。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from infusion_monitor.gateway.audit import DECISION_SOURCES, GatewayAuditor
from infusion_monitor.gateway.fusion import (
    ARBITRATION_CODES,
    FUSION_AUDIT_ACTIONS,
    Arbitration,
    FusionParams,
    apply_to_state,
    arbitrate,
    fuse,
    fuse_round,
    outcome_counts,
    threshold_decision_of,
)
from infusion_monitor.gateway.model_gateway import ModelGateway
from infusion_monitor.gateway.schemas import CONFIDENCE_FLOOR
from infusion_monitor.identity.state_machine import ChangeParams, evaluate_change
from infusion_monitor.store import (
    EventRepository,
    InferenceAuditRepository,
    SessionRepository,
)
from infusion_monitor.store.audit_chain import AuditChain

from .support.llm_stub import LLMStub

# 与阈值路径一致：液面连续 5 帧 ≤0.05 → near_empty_observed
NEEDS_CHANGE_NEAR_EMPTY = {
    "decision": "needs_change",
    "reason": "液面已经低到袋底，护士到床旁就该看着换了",
    "confidence": 0.88,
    "trigger_reason": "near_empty_observed",
}
# 与阈值路径相反：阈值判该换，模型说不换
NO_CHANGE = {
    "decision": "no_change",
    "reason": "余量还够撑到下一轮，不用换",
    "confidence": 0.93,
    "trigger_reason": None,
}
# 自评置信度不达门禁
LOW_CONFIDENCE = {
    "decision": "needs_change",
    "reason": "看着快没了",
    "confidence": 0.35,
    "trigger_reason": "near_empty_observed",
}
# 越界：模型说出了业务层词表以外的结论
OUT_OF_CONTRACT = {
    "decision": "replaced",
    "reason": "已经换好了",
    "confidence": 0.99,
    "trigger_reason": "near_empty_observed",
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
        "rack_id": "R-EXAMPLE-42",
        "bed_no": "EXAMPLE-BED-42",
        "infusion_order_no": "EXAMPLE-ORDER-0042",
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


def _seed(conn, *, bag_seq="B01", ratios=(0.08, 0.06, 0.05, 0.04, 0.03), **overrides) -> dict:
    sessions = SessionRepository(conn)
    row = sessions.upsert_session(_session_payload(bag_seq=bag_seq, **overrides))
    sessions.update_area_series(row["session_id"], list(ratios))
    return sessions.get(row["session_id"])


def _threshold(row, **overrides) -> object:
    """按阈值路径算出一条可判定的结论（needs_change / no_change 都试得出）。"""
    return evaluate_change(row, ChangeParams(**overrides))


def _observation(row, deterministic) -> dict:
    observation = dict(row)
    observation["threshold_decision"] = deterministic.to_json()
    return observation


def _run(stub, conn, *, model_reply, params=None, call_model=True, **seed_kwargs):
    """跑一次「调模型 → 仲裁」，返回 (结论, 会话, 阈值结论)。"""
    # 每次先清脚本：script_* 是追加，游标只随请求前进，不清就会拿到上一轮的回复，
    # 症状伪装成「仲裁逻辑不对」，方向完全跑偏。
    stub.reset()
    stub.script_json(model_reply)
    audit = AuditChain(conn)
    gateway = _gateway(stub, auditor=GatewayAuditor(InferenceAuditRepository(conn)))
    row = _seed(conn, **seed_kwargs)
    deterministic = _threshold(row)
    try:
        conclusion = arbitrate(
            gateway,
            _observation(row, deterministic),
            deterministic=deterministic,
            request_id="r-fusion",
            params=params,
            call_model=call_model,
            audit=audit,
            session=row,
        )
    finally:
        gateway.close()
    return conclusion, SessionRepository(conn).get(row["session_id"]), deterministic


# ── 1. 一致才采信，且只采信措辞 ──────────────────────────────────────────────


def test_agreement_accepts_model_wording_but_keeps_threshold_conclusion(stub, conn):
    conclusion, _, deterministic = _run(stub, conn, model_reply=NEEDS_CHANGE_NEAR_EMPTY)

    assert deterministic.target_state == "needs_change"
    assert deterministic.trigger_reason == "near_empty_observed"
    assert conclusion.arbitration.accepted is True
    assert conclusion.arbitration.code == "agreed_with_threshold"
    assert conclusion.decision_source == "model_decided"
    assert conclusion.decision == "needs_change"
    # 措辞取模型的（这是模型唯一被采信的东西）
    assert conclusion.reason == NEEDS_CHANGE_NEAR_EMPTY["reason"]
    # 结论与触发原因码永远取阈值路径（纪律 1）
    assert conclusion.trigger_reason == "near_empty_observed"
    assert conclusion.arbitration.threshold_decision == "needs_change"
    assert conclusion.arbitration.model_decision == "needs_change"
    assert conclusion.arbitration.trigger_reason_matched is True
    assert conclusion.arbitration.trigger_reason_discarded is None
    assert conclusion.pending_manual_review is False


def test_agreeing_decision_still_discards_a_different_trigger_reason(stub, conn):
    """结论一致但原因码不同：采信措辞，原因码仍取阈值的，差异留档不丢。"""
    reply = dict(NEEDS_CHANGE_NEAR_EMPTY, trigger_reason="eta_within_window")
    conclusion, _, _ = _run(stub, conn, model_reply=reply)

    assert conclusion.arbitration.accepted is True
    assert conclusion.trigger_reason == "near_empty_observed", "触发原因码永远取阈值路径"
    assert conclusion.arbitration.trigger_reason_matched is False
    assert conclusion.arbitration.trigger_reason_discarded == "eta_within_window"
    assert "eta_within_window" in conclusion.arbitration.detail


def test_model_trigger_reason_on_a_no_change_reply_is_never_promoted(stub, conn):
    """模型在「不用换」时还给了原因码：丢掉，结论与原因码都不采信。"""
    reply = dict(NO_CHANGE, trigger_reason="eta_within_window")
    conclusion, _, _ = _run(stub, conn, model_reply=reply, ratios=(0.9, 0.8, 0.7, 0.6, 0.5))

    assert conclusion.decision == "no_change"
    assert conclusion.trigger_reason is None
    assert conclusion.arbitration.trigger_reason_discarded == "eta_within_window"


# ── 2. 冲突回落 ─────────────────────────────────────────────────────────────


def test_conflict_falls_back_to_threshold_and_downgrades_the_source(stub, conn):
    """阈值判该换、模型说不换：以阈值为准，来源如实改写为 threshold_fallback。"""
    conclusion, _, deterministic = _run(stub, conn, model_reply=NO_CHANGE)

    assert deterministic.target_state == "needs_change"
    assert conclusion.arbitration.accepted is False
    assert conclusion.arbitration.code == "conflicts_with_threshold"
    assert conclusion.decision == "needs_change", "冲突时以阈值路径为准"
    assert conclusion.trigger_reason == "near_empty_observed"
    assert conclusion.reason == deterministic.reason_detail, "回落时展示阈值路径那句可复算的话"
    assert conclusion.decision_source == "threshold_fallback"
    assert conclusion.pending_manual_review is True
    assert "conflict" not in conclusion.reason.lower()


def test_reverse_conflict_also_falls_back(stub, conn):
    """反方向同样回落：阈值说不用换时，模型说该换也不能把状态推上去。"""
    conclusion, _, deterministic = _run(
        stub, conn, model_reply=NEEDS_CHANGE_NEAR_EMPTY, ratios=(0.9, 0.8, 0.7, 0.6, 0.5)
    )

    assert deterministic.target_state == "monitoring"
    assert conclusion.arbitration.code == "conflicts_with_threshold"
    assert conclusion.decision == "no_change"
    assert conclusion.decision_source == "threshold_fallback"
    assert conclusion.trigger_reason is None


# ── 3. 低置信回落 ───────────────────────────────────────────────────────────


def test_confidence_below_the_arbitration_floor_falls_back(stub, conn):
    """采信门禁调高于可用性门禁时，「过了校验但不敢用」也回落。"""
    params = FusionParams(confidence_floor=0.90)
    conclusion, _, _ = _run(
        stub, conn, model_reply=dict(NEEDS_CHANGE_NEAR_EMPTY, confidence=0.75), params=params
    )

    assert conclusion.arbitration.code == "below_confidence_floor"
    assert conclusion.arbitration.model_confidence == 0.75
    assert conclusion.decision_source == "threshold_fallback"
    assert conclusion.decision == "needs_change"


def test_default_floor_matches_the_gateway_usability_floor(stub, conn):
    """两道门默认同源同值：可用性门禁过了的模型，默认也过得了采信门禁。"""
    assert FusionParams().confidence_floor == CONFIDENCE_FLOOR
    conclusion, _, _ = _run(stub, conn, model_reply=dict(NEEDS_CHANGE_NEAR_EMPTY, confidence=0.61))
    assert conclusion.arbitration.accepted is True

    low = fuse(
        _stub_advice(LOW_CONFIDENCE, usable=True), deterministic=_stub_decision("needs_change")
    )
    assert low.arbitration.code == "below_confidence_floor"


def test_output_missing_a_trigger_reason_is_not_called_a_transport_failure():
    """协议层是好的、内容不够格，与端点不可达是两回事，原因码必须分开。"""
    unusable = fuse(_stub_advice(NO_CHANGE, usable=False), deterministic=_stub_decision("needs_change"))
    assert unusable.arbitration.code == "model_output_unusable"
    assert unusable.decision_source == "threshold_fallback"
    assert unusable.decision == "needs_change"


# ── 4. 端点不可达 / 越界 / 没调模型 ──────────────────────────────────────────


def test_endpoint_failure_falls_back_with_its_own_reason_code(stub, conn):
    stub.script_flood(503, times=5)
    gateway = _gateway(stub, auditor=GatewayAuditor(InferenceAuditRepository(conn)))
    row = _seed(conn)
    deterministic = _threshold(row)
    try:
        conclusion = arbitrate(
            gateway,
            _observation(row, deterministic),
            deterministic=deterministic,
            request_id="r-fusion-503",
            audit=AuditChain(conn),
            session=row,
        )
    finally:
        gateway.close()

    assert conclusion.arbitration.code == "gateway_degraded"
    assert "http_5xx" in conclusion.arbitration.detail
    assert conclusion.decision_source == "threshold_fallback"
    assert conclusion.decision == "needs_change"


def test_out_of_contract_decision_is_rejected_at_the_arbitration_layer(stub, conn):
    """越界结论即使绕过了网关的枚举校验（换端点/换提示词的可能），仲裁仍挡下。"""
    advice = _stub_advice(OUT_OF_CONTRACT, usable=True)
    conclusion = fuse(advice, deterministic=_stub_decision("needs_change"))

    assert conclusion.arbitration.code == "model_out_of_contract"
    assert conclusion.decision_source == "threshold_fallback"
    assert conclusion.decision == "needs_change"


def test_threshold_only_is_a_normal_state_not_a_fallback(stub, conn):
    """压根没调模型：正常路径（院内还没部署端点），不是故障、不落审计行。"""
    conclusion, _, _ = _run(stub, conn, model_reply=NEEDS_CHANGE_NEAR_EMPTY, call_model=False)

    assert conclusion.arbitration.code == "model_not_called"
    assert conclusion.arbitration.accepted is False
    assert conclusion.decision_source == "threshold_only"
    assert conclusion.decision == "needs_change"
    assert stub.call_count == 0, "call_model=False 时一次 HTTP 都不该发"
    actions = [row["action"] for row in AuditChain(conn).list_by_request("r-fusion")]
    assert actions == [], "没调模型就没有「模型被采信/被挡下」这回事可记"


def test_missing_threshold_conclusion_is_not_treated_as_agreement(stub, conn):
    """没有可比对的阈值结论（终态/无观测）时不存在「一致」，不采信。"""
    advice = _stub_advice(NEEDS_CHANGE_NEAR_EMPTY)
    conclusion = fuse(advice, deterministic=_stub_decision("replaced"))

    assert threshold_decision_of(_stub_decision("replaced")) is None, (
        "replaced 是护士点出来的动作，压根不该映射成模型词表里的结论"
    )
    assert conclusion.arbitration.code == "no_threshold_conclusion"
    assert conclusion.decision is None, "终态不该被模型结论重新打开"
    assert conclusion.decision_source == "threshold_fallback"


# ── 5. 决策来源三态 + 状态红线 ──────────────────────────────────────────────


def test_decision_source_three_states_are_all_observable(stub, conn):
    """三态都能在真实调用里观察到，且都落在出站字段规范允许的取值内。"""
    from infusion_monitor.adapters.nurse_exec import DECISION_SOURCES_OUTBOUND

    agreed, _, _ = _run(stub, conn, model_reply=NEEDS_CHANGE_NEAR_EMPTY)
    only, _, _ = _run(stub, conn, model_reply=NEEDS_CHANGE_NEAR_EMPTY, call_model=False)
    conflict, _, _ = _run(stub, conn, model_reply=NO_CHANGE)

    seen = {agreed.decision_source, only.decision_source, conflict.decision_source}
    assert seen == {"model_decided", "threshold_only", "threshold_fallback"}
    assert seen <= set(DECISION_SOURCES)
    for conclusion in (agreed, only, conflict):
        # 出站事件与响应带的是同一个值，且必须是契约里登记过的码
        assert conclusion.to_outbound_fields()["decision_source"] in DECISION_SOURCES_OUTBOUND
        assert conclusion.to_response_fields()["decision_source"] == conclusion.decision_source


def test_no_path_advances_state_and_apply_to_state_never_returns_replaced(stub, conn):
    """核心红线：四条路径调完之后状态与出站事件表纹丝不动。"""
    for reply, call_model, params in (
        (NEEDS_CHANGE_NEAR_EMPTY, True, None),
        (NO_CHANGE, True, None),
        (LOW_CONFIDENCE, True, None),
        (NEEDS_CHANGE_NEAR_EMPTY, False, None),
    ):
        conclusion, row, _ = _run(
            stub, conn, model_reply=reply, call_model=call_model, params=params
        )
        assert row["state"] == "monitoring", "模型路径不得把 monitoring 推走"
        target, changed = apply_to_state(conclusion, current_state=row["state"])
        assert target in ("monitoring", "needs_change"), "建议状态里永远不该有 replaced"
        assert changed is (conclusion.decision == "needs_change" and row["state"] != "needs_change")
        if target == "needs_change":
            # 就算模型判了 needs_change，服务侧也只是「建议」，真正迁移仍要护士确认
            assert row["state"] == "monitoring"
    assert EventRepository(conn).count() == 0, "模型不得写出站事件"


def test_fusion_module_holds_no_write_path(stub, conn):
    """静态红线：仲裁层不 import 仓储、不碰 sqlite——它压根没有推进状态的代码路径。"""
    source = Path(__file__).resolve().parents[1] / "src" / "infusion_monitor" / "gateway" / "fusion.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    for banned in ("sqlite3", "infusion_monitor.store.repositories",
                   "infusion_monitor.store.db", "infusion_monitor.identity.state_machine"
                   ".confirm_replacement"):
        assert banned not in imported, f"仲裁层不得依赖 {banned}：它只能给建议，不能写状态"


# ── 6. 仲裁可解释：append-only 审计 ────────────────────────────────────────


def test_accepted_and_rejected_are_audited_with_reasons(stub, conn):
    audit = AuditChain(conn)
    gateway = _gateway(stub, auditor=GatewayAuditor(InferenceAuditRepository(conn)))
    row = _seed(conn)
    deterministic = _threshold(row)
    try:
        for reply, request_id in ((NEEDS_CHANGE_NEAR_EMPTY, "r-acc"), (NO_CHANGE, "r-rej")):
            stub.script_json(reply)
            arbitrate(
                gateway,
                _observation(row, deterministic),
                deterministic=deterministic,
                request_id=request_id,
                audit=audit,
                session=row,
            )
    finally:
        gateway.close()

    accepted = audit.list_by_request("r-acc")
    rejected = audit.list_by_request("r-rej")
    assert [row["action"] for row in accepted] == [FUSION_AUDIT_ACTIONS["accepted"]]
    assert [row["action"] for row in rejected] == [FUSION_AUDIT_ACTIONS["rejected"]]
    assert accepted[0]["outcome"] == "accepted" and rejected[0]["outcome"] == "rejected"

    detail = rejected[0]["detail"]
    assert detail["code"] == "conflicts_with_threshold"
    assert detail["threshold_decision"] == "needs_change"
    assert detail["model_decision"] == "no_change"
    assert detail["decision"] == "needs_change", "回落后写进审计的仍是阈值路径的结论"
    assert detail["trigger_reason"] == "near_empty_observed"
    assert detail["confidence"] == pytest.approx(0.93)
    # 审计行能定位到具体这一袋（导审计包按架/按床过滤用得到这两列）
    assert rejected[0]["bag_seq"] == "B01"
    assert rejected[0]["rack_id"] == "R-EXAMPLE-42"
    assert rejected[0]["bag_key"] == "EXAMPLE-ORDER-0042|B01"


def test_one_bag_blowing_up_does_not_cost_the_round_its_other_bags(stub, conn):
    """单袋失败不牵连同轮其余袋：一轮巡视有六袋，不能因一袋模型抽风就都没了提示。"""

    class _ExplodesOnB02:
        def __init__(self, inner):
            self.inner = inner

        def advise(self, observation, **kwargs):
            if observation.get("bag_seq") == "B02":
                raise RuntimeError("模型端点在这一袋上炸了")
            return self.inner.advise(observation, **kwargs)

    stub.script_json(NEEDS_CHANGE_NEAR_EMPTY)
    gateway = _gateway(stub)
    first = _seed(conn, bag_seq="B01")
    second = _seed(conn, bag_seq="B02", slot_index=1)
    try:
        results = fuse_round(
            _ExplodesOnB02(gateway), [first, second], request_id="r-round"
        )
    finally:
        gateway.close()

    assert set(results) == {"B01"}
    assert results["B01"].decision == "needs_change"
    assert outcome_counts(results.values()) == {"agreed_with_threshold": 1}


# ── 辅助 ────────────────────────────────────────────────────────────────────


def _stub_decision(target_state: str):
    from infusion_monitor.identity.state_machine import ChangeDecision

    return ChangeDecision(
        bag_seq="B01",
        session_id="sess-fusion",
        target_state=target_state,
        changed=target_state == "needs_change",
        trigger_reason="near_empty_observed" if target_state == "needs_change" else None,
        consecutive_frames=5,
        latest_fill_ratio=0.03,
        change_threshold_ratio=0.15,
        reason_detail="连续 5 帧低于 0.15，余量判定=near_empty",
        decided_at="2026-10-23T09:41:00+08:00",
    )


def _stub_advice(payload: dict, *, usable: bool | None = None):
    """直接构造一条 :class:`GatewayAdvice`，用于绕过网关的契约校验来测仲裁层自身。

    ``usable=None`` 表示「按网关那道可用性门禁推」；显式传 ``True/False`` 才能造出
    「过了门禁却不够格采信」「压根没给可执行的触发原因码」这两类输入。
    """
    from infusion_monitor.gateway.model_gateway import GatewayAdvice

    return GatewayAdvice(
        request_id="r-direct",
        model="test-model",
        prompt_hash="0123456789abcdef",
        prompt_version="infusion_change@v1",
        decision_source="model_decided",
        pending_manual_review=False,
        deterministic=_stub_decision("needs_change"),
        decision=payload["decision"],
        reason=payload["reason"],
        trigger_reason=payload.get("trigger_reason"),
        confidence=payload["confidence"],
        usable=payload["confidence"] >= CONFIDENCE_FLOOR if usable is None else usable,
        latency_ms=12,
        prompt_tokens=256,
        completion_tokens=41,
        audit_written=True,
    )


# 仲裁码的唯一性：多写一个码而忘了在文档/统计里登记时，这里当场报错
def test_arbitration_codes_are_unique_and_audit_actions_are_registered():
    assert len(set(ARBITRATION_CODES)) == len(ARBITRATION_CODES)
    assert set(FUSION_AUDIT_ACTIONS) == {"accepted", "rejected"}
    from infusion_monitor.store.schema import AUDIT_ACTIONS

    assert set(FUSION_AUDIT_ACTIONS.values()) <= set(AUDIT_ACTIONS)
    assert isinstance(Arbitration(accepted=True, code="agreed_with_threshold", detail="",
                                  threshold_decision="needs_change"), Arbitration)