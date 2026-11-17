"""决策融合仲裁层：模型不是裁判，只是阈值结论的「复述与措辞」。

**为什么网关（task 12）之后还要这一层**：网关能把一次调用的成败、来源与降级
原因如实记下来，但「模型给了一条通过契约校验的结论」并不等于「这条结论该被
采信」。一个自评 0.95 的模型，在一段被反光糊住的短片上完全可能说「不用换」，
而阈值路径刚拿同一段证据判了「该换」。谁说了算？答案是**阈值路径**——它背后是
可复算的连续帧数、余量门限与近空观测；模型只有一句话，而且那句话可能来自
上一轮的截图。

四条纪律，每条都对应一个会在病区出事的地方：

1. **模型只能改措辞，不能改结论**。:func:`fuse` 返回的 ``decision`` 与
   ``trigger_reason`` **永远**来自阈值路径；采信时模型贡献的是给护士看的那句
   人话（``reason``）。触发原因码更是一律不采信——``eta_within_window`` 与
   ``near_empty_observed`` 在护士端是两条不同的提示文案，让一段无法复算的文本
   决定护士看到什么，等于把点名判据交给了一个说不清依据的环节。
2. **冲突、越界、低置信一律回落**。回落原因码写进 :attr:`Arbitration.code`，
   并落一行 ``inference.fusion_accepted`` / ``inference.fusion_rejected``
   审计（动作码登记在 :data:`store.schema.AUDIT_ACTIONS`，链在
   :mod:`store.audit_chain`）。护士长问「这袋是模型说的还是算出来的」，答案在库里。
3. **模型路径绝不推进业务状态**。:func:`apply_to_state` 是本模块唯一给出状态
   建议的函数，它的返回值里**永远不可能有 ``replaced``**：``needs_change →
   replaced`` 只能由护士在终端确认触发（见
   :func:`~identity.state_machine.confirm_replacement`）。本模块不 import 任何
   仓储、没有任何写状态的方法——这条纪律不是靠自觉，是靠没有那条代码路径。
4. **回落不是故障，是常态**。一个病区一周里模型有几次没调成、几次与阈值冲突，
   都是正常信息：它们说明阈值路径在独立工作。把它做成**看得见**的三态
   （见 :data:`DECISION_SOURCES`）比做成一个「成功/失败」开关有用得多。

**本模块不做的事**：不重算阈值结论（那是调用方用
:func:`~identity.state_machine.evaluate_change` 算好后传进来的）、不调 HTTP
（那是网关的事）、不碰数据库（审计行由调用方按已算好的结论补写）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..identity.state_machine import ChangeParams
from .audit import DECISION_SOURCES
from .model_gateway import GatewayAdvice
from .schemas import CONFIDENCE_FLOOR, DECISIONS

__all__ = [
    "ARBITRATION_CODES",
    "FUSED_ACCEPTED",
    "FUSION_AUDIT_ACTIONS",
    "SERVICE_ACTOR",
    "Arbitration",
    "FusionParams",
    "FusedConclusion",
    "apply_to_state",
    "arbitrate",
    "fuse",
    "fuse_round",
    "outcome_counts",
    "threshold_decision_of",
]

#: 仲裁 actor 标识（审计行用它区分「服务算的」与「护士点的」）。
SERVICE_ACTOR = "service:decision-fusion"

#: 采信：模型结论与阈值结论一致，且自评置信度达门禁。
FUSED_ACCEPTED = "agreed_with_threshold"

#: 仲裁结果码。前一个是采信，其余都是回落——每个码对应一个具体的失败/回避场景，
#: 统计时按码分开看（「这周回落了几次、因为什么」），合并成一个码就答不上来了。
ARBITRATION_CODES: tuple[str, ...] = (
    FUSED_ACCEPTED,               # 一致且置信达标 → 采信模型的人话措辞
    "conflicts_with_threshold",   # 结论与阈值相反（阈值说换、模型说不换，反之亦然）
    "below_confidence_floor",     # 模型自评置信度不达门禁
    "model_output_unusable",      # 缺 trigger_reason 等，没到可用这一步
    "model_out_of_contract",      # 越界：模型说了业务层枚举之外的结论
    "gateway_degraded",           # 端点不可达 / 5xx / 输出不可解析，网关已降级
    "model_not_called",           # 这一轮压根没调模型（院内还没部署端点）
    "no_threshold_conclusion",    # 没有阈值结论可比（这一段观测不足以判定）
)

#: 仲裁落审计用的两个动作码。**采信与回落各一个**：飞检与护理部问的是两件不同的
#: 事——「模型参与了结论」与「模型被挡在结论之外」，合并成一个码就答不出后者。
FUSION_AUDIT_ACTIONS: dict[str, str] = {
    "accepted": "inference.fusion_accepted",
    "rejected": "inference.fusion_rejected",
}

#: 阈值状态 → 模型词表的结论。**只认前两个**；``replaced`` 是护士的动作，
#: 出现在阈值判定里说明调用方传错了东西，当越界处理。
_STATE_TO_DECISION: dict[str, str] = {
    "monitoring": "no_change",
    "needs_change": "needs_change",
}


@dataclass(frozen=True)
class FusionParams:
    """仲裁门禁。

    ``confidence_floor`` 与网关的可用性门禁同源（:data:`CONFIDENCE_FLOOR`）：
    「够不够格说」已经判过一次，这里判的是「说了之后敢不敢用」，两道门禁的
    门槛相同但问的不是同一个问题——一道挡解析层，一道挡采信层。
    """

    confidence_floor: float = CONFIDENCE_FLOOR
    #: 关掉它等于「模型说什么都采信」。留这个开关只是为了让门禁本身可测，
    #: 不是给部署用的旋钮——真要关，正确做法是把端点摘掉走 ``threshold_only``。
    require_agreement: bool = True

    @classmethod
    def from_settings(cls, settings: Any) -> "FusionParams":
        """从配置面取置信门禁（缺省沿用 :data:`CONFIDENCE_FLOOR`）。"""
        floor = getattr(settings, "llm_fusion_confidence_floor", None)
        return cls(confidence_floor=float(floor) if floor is not None else CONFIDENCE_FLOOR)


@dataclass(frozen=True)
class Arbitration:
    """一次仲裁的**理由**。护士长与信息科复核时看的就是这个对象。"""

    #: 是否采信模型的措辞。注意这不是「模型调用成功与否」——调用成功但结论冲突
    #: 一样是 False。
    accepted: bool
    code: str
    detail: str
    #: 阈值路径的结论（``no_change`` / ``needs_change``）。
    threshold_decision: str | None
    #: 模型给的结论；压根没调成或不可用时为 ``None``。
    model_decision: str | None = None
    model_confidence: float | None = None
    #: 模型给的触发原因码与阈值是否一致。**不一致也只是丢弃**（见纪律 1），
    #: 单独记一个字段是为了让「它天天给不同的原因码」这件事能被统计出来。
    trigger_reason_matched: bool | None = None
    #: 被丢弃的模型触发原因码（仅在采信、但原因码不一致时非空）。
    trigger_reason_discarded: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "code": self.code,
            "detail": self.detail,
            "threshold_decision": self.threshold_decision,
            "model_decision": self.model_decision,
            "model_confidence": self.model_confidence,
            "trigger_reason_matched": self.trigger_reason_matched,
            "trigger_reason_discarded": self.trigger_reason_discarded,
        }


@dataclass(frozen=True)
class FusedConclusion:
    """仲裁后的**有效结论**：护士端显示什么、出站事件带什么，都取自这里。

    字段分两半：左边（``decision`` / ``trigger_reason`` / ``threshold_reason``）
    是可复算的证据，右边（``model_*``）是模型的原话与元数据。**左边永远不是
    右边推出来的**——这是本模块存在的全部理由。
    """

    decision: str | None
    trigger_reason: str | None
    reason: str
    decision_source: str
    arbitration: Arbitration
    pending_manual_review: bool
    request_id: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    #: 模型侧原话与元数据（采信时 ``reason`` 就是它）。
    model_decision: str | None = None
    model_reason: str | None = None
    model_trigger_reason: str | None = None
    model_confidence: float | None = None
    model_error: str | None = None
    attempts: int = 0
    attempt_kinds: list[str] = field(default_factory=list)
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    audit_written: bool = False

    # ── 对外形状 ──────────────────────────────────────────────────────────
    def to_json(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "trigger_reason": self.trigger_reason,
            "reason": self.reason,
            "decision_source": self.decision_source,
            "pending_manual_review": self.pending_manual_review,
            "arbitration": self.arbitration.to_json(),
            "request_id": self.request_id,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "model_decision": self.model_decision,
            "model_reason": self.model_reason,
            "model_trigger_reason": self.model_trigger_reason,
            "model_confidence": self.model_confidence,
            "model_error": self.model_error,
            "attempts": self.attempts,
            "attempt_kinds": list(self.attempt_kinds),
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "audit_written": self.audit_written,
        }

    def to_response_fields(self) -> dict[str, Any]:
        """挂到上传响应每只袋子上的那几个字段（其余字段在响应模型里各有一列）。"""
        return {
            "decision_source": self.decision_source,
            "decision": self.decision,
            "reason": self.reason,
            "arbitration": self.arbitration.to_json(),
            "pending_manual_review": self.pending_manual_review,
        }

    def to_outbound_fields(self) -> dict[str, Any]:
        """并进出站事件载荷的字段。

        只有 ``decision_source`` 一列真的进载荷（其余由
        :data:`~infusion_monitor.adapters.nurse_exec.OUTBOUND_EVENT_SCHEMA` 定义），
        带上理由码是为了让审计包与推理审计能对上同一次仲裁。
        """
        return {"decision_source": self.decision_source}


# ── 阈值结论 → 模型词表 ────────────────────────────────────────────────────


def _threshold_target(deterministic: Any) -> str | None:
    """取出阈值路径判定的目标状态（``monitoring`` / ``needs_change``）。"""
    if deterministic is None:
        return None
    if isinstance(deterministic, Mapping):
        return deterministic.get("target_state") or deterministic.get("state")
    return getattr(deterministic, "target_state", None) or getattr(deterministic, "state", None)


def threshold_decision_of(deterministic: Any) -> str | None:
    """阈值结论 → 模型词表的结论；无法映射时返回 ``None``。

    ``replaced`` 映射成 ``None`` 而不是某个结论：那是护士点出来的动作，阈值
    路径压根不该产出它，看到了按越界处理（见 :func:`_out_of_contract`）。
    """
    target = _threshold_target(deterministic)
    if target is None:
        return None
    return _STATE_TO_DECISION.get(str(target))


def _threshold_reason(deterministic: Any) -> str:
    """阈值路径的人话说明（回落时展示给护士的那句）。"""
    if isinstance(deterministic, Mapping):
        detail = deterministic.get("reason_detail") or ""
    else:
        detail = getattr(deterministic, "reason_detail", "") or ""
    if detail:
        return str(detail)
    decision = threshold_decision_of(deterministic)
    return "阈值路径判定该换液" if decision == "needs_change" else "阈值路径判定不用换"


def _threshold_trigger(deterministic: Any) -> str | None:
    if isinstance(deterministic, Mapping):
        return deterministic.get("trigger_reason")
    return getattr(deterministic, "trigger_reason", None)


# ── 仲裁 ────────────────────────────────────────────────────────────────────


def fuse(
    advice: GatewayAdvice,
    *,
    deterministic: Any = None,
    params: FusionParams | None = None,
    audit: Any | None = None,
    request_id: str | None = None,
    session: Mapping[str, Any] | None = None,
) -> FusedConclusion:
    """把一条 :class:`GatewayAdvice` 与阈值结论合成一条**有效结论**。

    :param deterministic: 阈值路径已经算好的判定
        （:class:`~identity.state_machine.ChangeDecision`，或等价的 mapping）。
        缺省取 ``advice.deterministic``——网关当初就是拿着它去问模型的，
        同一个对象，不重算也不重取。
    :param audit: :class:`~store.audit_chain.AuditChain`，落一行仲裁留痕。
        压根没调模型时**不写**：审计链记的是「发生过什么」，不是「什么都没发生」。
    :returns: :class:`FusedConclusion`。**本函数不推进任何业务状态。**
    """
    params = params or FusionParams()
    threshold = deterministic if deterministic is not None else advice.deterministic
    threshold_decision = threshold_decision_of(threshold)
    source = advice.decision_source if advice.decision_source in DECISION_SOURCES else "threshold_only"
    fallback_reason = advice.fallback_reason

    arbitration = _arbitrate(advice, threshold_decision, params, source, fallback_reason)

    # 调成了、也过了契约校验，但仲裁没过（冲突/低置信/无可比对结论）：结论实际上
    # 是阈值路径给的，如实把来源改写成 threshold_fallback——库里若还写着
    # model_decided，护士长问「这条谁说的」就得到了一个错的答案。
    if not arbitration.accepted and source == "model_decided":
        source = "threshold_fallback"

    # 有效结论：左边两列永远来自阈值路径（纪律 1）。
    decision = threshold_decision
    trigger_reason = _threshold_trigger(threshold) if threshold_decision == "needs_change" else None
    if arbitration.accepted and advice.reason:
        reason = advice.reason
    else:
        reason = _threshold_reason(threshold)

    audit_written = _record_audit(
        audit=audit,
        request_id=request_id or advice.request_id,
        arbitration=arbitration,
        advice=advice,
        conclusion_decision=decision,
        trigger_reason=trigger_reason,
        reason=reason,
        session=session or {},
    )

    return FusedConclusion(
        decision=decision,
        trigger_reason=trigger_reason,
        reason=reason,
        decision_source=source,
        arbitration=arbitration,
        # 模型缺席或被挡下时，护士端的这句话就不是模型写的——标出来，
        # 让护士长知道该由谁兜这一眼。
        pending_manual_review=(
            advice.pending_manual_review or (source != "threshold_only" and not arbitration.accepted)
        ),
        request_id=advice.request_id,
        model=advice.model,
        prompt_version=advice.prompt_version,
        prompt_hash=advice.prompt_hash,
        model_decision=advice.decision,
        model_reason=advice.reason,
        model_trigger_reason=advice.trigger_reason,
        model_confidence=advice.confidence,
        model_error=advice.error,
        attempts=advice.attempts,
        attempt_kinds=list(advice.attempt_kinds),
        latency_ms=advice.latency_ms,
        prompt_tokens=advice.prompt_tokens,
        completion_tokens=advice.completion_tokens,
        audit_written=audit_written,
    )


def _arbitrate(
    advice: GatewayAdvice,
    threshold_decision: str | None,
    params: FusionParams,
    source: str,
    fallback_reason: str | None,
) -> Arbitration:
    """仲裁规则本体。四道门，按「先问有没有得比、再问敢不敢用」的顺序过。"""
    model_decision = advice.decision
    model_confidence = advice.confidence

    def _rejected(code: str, detail: str, **extra: Any) -> Arbitration:
        return Arbitration(
            accepted=False,
            code=code,
            detail=detail,
            threshold_decision=threshold_decision,
            model_decision=model_decision,
            model_confidence=model_confidence,
            **extra,
        )

    # ① 这一轮压根没调模型 —— 正常路径，不是回落。
    if source == "threshold_only":
        return Arbitration(
            accepted=False,
            code="model_not_called",
            detail="这一轮没有调用模型（院内未开启或未部署端点），结论由阈值路径给出",
            threshold_decision=threshold_decision,
        )

    # ② 没有阈值结论可比：观测不足以判定时不存在「一致」这回事。
    if threshold_decision is None:
        return _rejected(
            "no_threshold_conclusion",
            "阈值路径没有给出可比对的结论（观测不足或状态已是终态），不采信模型",
        )

    # ③ 越界：模型说了业务层词表以外的结论。网关那道枚举校验本该先挡掉，
    #    这里再挡一次是给「换端点/换提示词后校验被绕过」留的后手。
    if model_decision is not None and model_decision not in DECISIONS:
        return _rejected(
            "model_out_of_contract",
            f"模型给出的结论 {model_decision!r} 不在允许的取值内，按越界回落",
        )

    # ④ 端点/协议层没成：压根没有可用的模型结论。找信息科。
    if source != "model_decided" or model_decision is None:
        return _rejected(
            "gateway_degraded",
            f"模型调用未成（{fallback_reason or source}），回落阈值路径",
        )

    # ⑤ 协议层是好的，**内容**不够格（needs_change 却没给触发原因码）：改提示词，
    #    不是找信息科——与 ④ 的排查方向不同，原因码必须分开记。
    if not advice.usable:
        return _rejected(
            "model_output_unusable",
            f"模型给了结论 {model_decision!r} 但缺可执行的触发原因码，按不可用回落阈值路径",
        )

    # ⑤ 低置信：模型自己都说没把握，那就没理由采信它。
    confidence = model_confidence if model_confidence is not None else 0.0
    if confidence < params.confidence_floor:
        return _rejected(
            "below_confidence_floor",
            f"模型自评置信度 {confidence:.2f} 低于门禁 {params.confidence_floor:.2f}，回落阈值路径",
        )

    # ⑥ 一致性：这是采信与否的最后一道门，也是最重要的一道。
    if params.require_agreement and model_decision != threshold_decision:
        return _rejected(
            "conflicts_with_threshold",
            f"模型判「{model_decision}」，阈值路径判「{threshold_decision}」，"
            "两者冲突，一律以阈值路径为准",
        )

    # 采信：一致且达门禁。触发原因码仍不采信（纪律 1），不一致就丢弃并留档。
    model_trigger = advice.trigger_reason
    threshold_trigger = _threshold_trigger(advice.deterministic)
    matched: bool | None = None
    discarded: str | None = None
    if model_trigger is not None:
        matched = model_trigger == threshold_trigger
        if not matched:
            discarded = model_trigger
    detail = (
        f"模型判「{model_decision}」与阈值路径一致且置信度 {confidence:.2f} 达门禁，"
        "采信模型给护士看的措辞；结论与触发原因码仍取阈值路径"
    )
    if discarded is not None:
        detail += f"（模型另给的原因码 {discarded} 已丢弃）"
    return Arbitration(
        accepted=True,
        code=FUSED_ACCEPTED,
        detail=detail,
        threshold_decision=threshold_decision,
        model_decision=model_decision,
        model_confidence=confidence,
        trigger_reason_matched=matched,
        trigger_reason_discarded=discarded,
    )


def _record_audit(
    *,
    audit: Any | None,
    request_id: str | None,
    arbitration: Arbitration,
    advice: GatewayAdvice,
    conclusion_decision: str | None,
    trigger_reason: str | None,
    reason: str,
    session: Mapping[str, Any],
) -> bool:
    """落一行仲裁审计。写不进去就如实说 ``False``，由调用方决定要不要上报。"""
    if audit is None or not request_id:
        return False
    # 压根没调模型不落行：这一轮没有「模型被采信/被挡下」这回事可记。
    if arbitration.code == "model_not_called":
        return False
    from ..store.schema import AUDIT_ACTIONS

    action = FUSION_AUDIT_ACTIONS["accepted" if arbitration.accepted else "rejected"]
    if action not in AUDIT_ACTIONS:  # pragma: no cover - 登记漏了就当场炸，不静默丢
        raise AssertionError(f"仲裁动作码 {action} 未登记进 AUDIT_ACTIONS")
    try:
        audit.append(
            request_id=request_id,
            actor=SERVICE_ACTOR,
            action=action,
            target=str(
                session.get("session_id")
                or getattr(advice.deterministic, "session_id", None)
                or "-"
            ),
            outcome="accepted" if arbitration.accepted else "rejected",
            bag_seq=session.get("bag_seq") or getattr(advice.deterministic, "bag_seq", None),
            bag_key=session.get("infusion_order_no")
            and f"{session.get('infusion_order_no')}|{session.get('bag_seq')}",
            rack_id=session.get("rack_id"),
            bed_no=session.get("bed_no"),
            from_state=session.get("state"),
            to_state=session.get("state"),
            detail={
                "code": arbitration.code,
                "decision": conclusion_decision,
                "threshold_decision": arbitration.threshold_decision,
                "model_decision": arbitration.model_decision,
                "trigger_reason": trigger_reason,
                "reason": reason,
                "decision_source": advice.decision_source,
                "fallback_reason": advice.fallback_reason,
                "model": advice.model,
                "prompt_hash": advice.prompt_hash,
                "confidence": arbitration.model_confidence,
                "latency_ms": advice.latency_ms,
                "attempts": advice.attempts,
            },
        )
    except Exception:  # noqa: BLE001 - 审计写不进不该把已经算出的结论带倒
        return False
    return True


# ── 一句话入口 ──────────────────────────────────────────────────────────────


def arbitrate(
    gateway: Any,
    observation: Mapping[str, Any],
    *,
    deterministic: Any,
    request_id: str,
    params: FusionParams | None = None,
    call_model: bool = True,
    audit: Any | None = None,
    session: Mapping[str, Any] | None = None,
    max_retries: int | None = None,
) -> FusedConclusion:
    """调一次模型（由网关去调）并立刻仲裁，返回一条**有效结论**。

    这是业务层该用的唯一入口：它把「问模型」与「敢不敢用」收在一处，避免
    某处只调了网关的 :meth:`~gateway.model_gateway.ModelGateway.advise` 就把
    模型结论当结论用。**本函数不推进任何业务状态。**
    """
    advice = gateway.advise(
        observation,
        deterministic=deterministic,
        request_id=request_id,
        call_model=call_model,
        max_retries=max_retries,
    )
    return fuse(
        advice,
        deterministic=deterministic,
        params=params,
        audit=audit,
        request_id=request_id,
        session=session,
    )


def fuse_round(
    gateway: Any,
    candidates: Sequence[Mapping[str, Any]],
    *,
    request_id: str,
    params: FusionParams | None = None,
    call_model: bool = True,
    audit: Any | None = None,
    change_params: ChangeParams | None = None,
) -> dict[str, FusedConclusion]:
    """一轮巡视里对若干袋依次仲裁，返回 ``{bag_seq: FusedConclusion}``。

    **逐袋独立、单袋失败不影响其余**：某一袋的仲裁抛异常时它自己落空，
    其余袋照样给结论——一轮巡视有六袋，不能因为一袋的模型抽风就让另外五袋
    的护栏文案也没了。这一条是刻意选的容错边界，不是顺手写的。
    """
    from ..identity.state_machine import evaluate_change

    results: dict[str, FusedConclusion] = {}
    for item in candidates:
        bag_seq = str(item.get("bag_seq") or "")
        if not bag_seq:
            continue
        try:
            decision = evaluate_change(item, change_params or ChangeParams())
            # 阈值结论一并进观测：模型看到的是「阈值路径判了什么」，而不是让它从
            # 同一堆数字里重算一遍——否则两边各算一次，「一致」就成了两次巧合。
            observation = {**item, "threshold_decision": decision.to_json()}
            results[bag_seq] = arbitrate(
                gateway,
                observation,
                deterministic=decision,
                request_id=request_id,
                params=params,
                call_model=call_model,
                audit=audit,
                session=item,
            )
        except Exception:  # noqa: BLE001 - 单袋失败不牵连同轮其余袋
            continue
    return results


# ── 状态建议：模型路径到不了 replaced ──────────────────────────────────────


def apply_to_state(conclusion: FusedConclusion, *, current_state: str) -> tuple[str, bool]:
    """仲裁结论 → 建议状态，返回 ``(目标状态, 是否需要迁移)``。

    **返回值里永远不可能有 ``replaced``**：``needs_change → replaced`` 是护士在
    终端上点的确认（见
    :func:`~identity.state_machine.confirm_replacement`），模型与仲裁层都没有
    那条路径。这也是为什么本模块不 import 仓储——没有写状态的方法，也就没有
    「不小心推进一下」的机会。
    """
    current = str(current_state or "monitoring")
    if current == "replaced" or conclusion.decision != "needs_change":
        return current, False
    return "needs_change", current != "needs_change"


def outcome_counts(conclusions: Iterable[FusedConclusion]) -> dict[str, int]:
    """一轮巡视里按仲裁码统计次数（护士长问「这周回落了几次、因为什么」）。"""
    counts: dict[str, int] = {}
    for conclusion in conclusions:
        code = conclusion.arbitration.code
        counts[code] = counts.get(code, 0) + 1
    return counts