"""袋级业务状态机：``monitoring → needs_change → replaced``。

**状态的所有者是护理执行平台，不是本服务**：本服务只负责算出 ``needs_change``
（依据余量曲线与滴速的建议），而 ``replaced`` **只能由护士在终端确认触发**——
中间隔着一次真人的手。没有这一步，一次误报就会变成一次没人执行的拔管/换袋。

三个状态各自对应现场的一种事实：

============ ==========================================================
状态          含义
============ ==========================================================
monitoring    在看，还没到要提醒的程度
needs_change  服务算出来「这袋快见底了」，等人来换；**未确认前不出站**
replaced      护士确认已经换过；终态，不可回退
============ ==========================================================

``needs_change`` 的判定是**双条件**而不是单看液位：

* **条件 A（稳）**：连续 ``n_frames`` 帧的可见液面都低于 ``change_threshold_ratio``。
  单帧毛刺（反光、滴斗晃一下）不许触发点名——误报一次，护士下次就不看了。
* **条件 B（急）**：余量确实快不够了——ETA 已进入 ``eta_window_minutes``，或剩余
  毫升数低于 ``margin_ml``，或液面已低到 ``near_empty_ratio``。

抑制门禁在判定之前：被标定门禁压住（``uncalibrated`` / ``eta_unavailable`` /
``no_observation`` / ``overfilled``）的袋**不点名**——拿一个不可信的毫升数去催护士
换袋，比不提醒更糟。唯一的例外是 ``near_empty``：「袋子里几乎没液体了」是直接观测，
不需要毫升换算，也就不受标定门禁约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..errors import ApiError, Issue
from ..store.repositories import utc_now

__all__ = [
    "ALLOWED_TRANSITIONS",
    "ChangeDecision",
    "ChangeParams",
    "IllegalTransition",
    "ReplacementConfirmation",
    "ServiceActor",
    "apply_decision",
    "build_client_event_id",
    "check_transition",
    "confirm_replacement",
    "evaluate_change",
]


def build_client_event_id(session_id: str, bag_seq: str) -> str:
    """出站换液事件的幂等键。

    **键里不能有时间戳**：护士点一次「已换液」，病区 Wi-Fi 抖动会让同一个请求重发
    好几次。若把确认时刻拼进键，每次重试都算出不同的键——那不叫幂等键，那叫每次
    都当成新事件，护理执行平台上就会出现三条「B01 已换」而 B01 只真的换过一次。

    键由 (会话, 袋序) 唯一确定：同一袋只会被换一次，重放必然算出同一个键；
    换袋后对端会给新的 ``bag_seq``，自然会得到新的键。
    """
    return f"infusion-{session_id}-{bag_seq}-replaced"

#: 允许的状态迁移。**只允许这两条边**，其余（含自环）一律拒绝。
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "monitoring": frozenset({"needs_change"}),
    "needs_change": frozenset({"replaced"}),
    "replaced": frozenset(),  # 终态
}

#: 服务侧动作在审计里的 actor 标识。
SERVICE_ACTOR = "service:change-detector"
#: 触发原因码：判定通过时必然给出其中一条，护士端按码给不同的提示文案。
TRIGGER_REASONS = (
    "eta_within_window",    # 预计撑不到本轮巡视结束
    "volume_below_margin",  # 剩余毫升数低于余量下限
    "near_empty_observed",  # 液面已低到近空（直接观测，不经毫升换算）
)


class ServiceActor:
    """审计 actor 常量（避免各处散落字符串字面量）。"""

    DETECTOR = SERVICE_ACTOR


@dataclass(frozen=True)
class ChangeParams:
    """``needs_change`` 判定门限。"""

    #: 条件 A：连续多少帧低于阈值才算稳
    n_frames: int = 3
    #: 条件 A：液位阈值（面积比）
    change_threshold_ratio: float = 0.15
    #: 条件 B：ETA 进入多少分钟内算急
    eta_window_minutes: int = 30
    #: 条件 B：剩余毫升低于多少算急
    margin_ml: float = 30.0
    #: 条件 B：液面低到多少算「快空了」（且该观测不受标定门禁约束）
    near_empty_ratio: float = 0.06


@dataclass(frozen=True)
class ChangeDecision:
    """一次判定结论。``target_state`` 是判定后应当处于的状态（可能不变）。"""

    bag_seq: str
    session_id: str | None
    target_state: str
    changed: bool
    trigger_reason: str | None
    consecutive_frames: int
    latest_fill_ratio: float | None
    change_threshold_ratio: float
    #: 触发条件 B 的那一条：eta / volume / near_empty / none
    margin_kind: str = "none"
    margin_value: float | None = None
    suppression_reason: str | None = None
    reason_detail: str = ""
    decided_at: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "bag_seq": self.bag_seq,
            "session_id": self.session_id,
            "target_state": self.target_state,
            "changed": self.changed,
            "trigger_reason": self.trigger_reason,
            "consecutive_frames": self.consecutive_frames,
            "latest_fill_ratio": self.latest_fill_ratio,
            "change_threshold_ratio": self.change_threshold_ratio,
            "margin_kind": self.margin_kind,
            "margin_value": self.margin_value,
            "suppression_reason": self.suppression_reason,
            "reason_detail": self.reason_detail,
            "decided_at": self.decided_at,
        }


class IllegalTransition(ApiError):
    """非法状态迁移（``monitoring`` 直跳 ``replaced``、``replaced`` 回退等）。

    继承 :class:`~infusion_monitor.errors.ApiError` 是为了**直接**被 API 层映射成
    409（``STATE_CONFLICT``），调用方不需要在每个入口各写一遍映射。
    """

    def __init__(self, current: str, target: str, *, session_id: str | None = None,
                 bag_seq: str | None = None, audit: Any | None = None,
                 request_id: str | None = None) -> None:
        allowed = sorted(ALLOWED_TRANSITIONS.get(current, frozenset()))
        hint = {
            "replaced": "这袋已确认换过，终态不可回退；新一轮巡视请让对端给新的 bag_seq。",
            "monitoring": "还没到换液点，不能直接跳到已换；请先等服务算出换液点名，"
                          "再由护士在终端确认。",
        }.get(current, "当前状态不允许这次迁移。")
        super().__init__(
            "STATE_CONFLICT",
            f"状态迁移 {current} → {target} 不合法"
            + (f"（{current} 只允许 → {allowed}）" if allowed else f"（{current} 是终态）")
            + f"。{hint}",
            issues=[
                Issue(
                    code="illegal_transition",
                    field="state",
                    reason=f"当前状态 {current}，请求迁移到 {target}",
                    suggestion=hint,
                )
            ],
        )
        self.current = current
        self.target = target
        # 被拒的迁移同样留一行审计：飞检查的是「系统是不是在不该动的时候动了」
        if audit is not None and request_id:
            audit.append(
                request_id=request_id,
                actor=SERVICE_ACTOR,
                action="state.transition_rejected",
                target=session_id or "-",
                outcome="rejected",
                bag_seq=bag_seq,
                from_state=current,
                to_state=target,
                detail={"allowed": allowed, "reason": "illegal_transition"},
            )


def check_transition(current: str, target: str, **kwargs: Any) -> None:
    """校验迁移合法性；不合法抛 :class:`IllegalTransition`（带审计写入）。"""
    if current == target:
        raise IllegalTransition(current, target, **kwargs)
    if target not in ALLOWED_TRANSITIONS.get(current, frozenset()):
        raise IllegalTransition(current, target, **kwargs)


# ── needs_change 判定 ─────────────────────────────────────────────────────


def _visible_ratios(session: Mapping[str, Any]) -> list[float]:
    """取可见帧的液面序列（被遮挡的 null 帧不算观测，不参与「连续 N 帧」计数）。"""
    series = session.get("area_ratio_series") or []
    values: list[float] = []
    for item in series:
        if item is None:
            continue
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        values.append(value)
    return values


def _consecutive_below(values: Sequence[float], threshold: float) -> tuple[int, float | None]:
    """从末尾往回数连续低于阈值的帧数，返回 (帧数, 末次液面)。

    ``latest`` 取**最后**一个观测值（序列末端），不是倒着数时先碰到的那个——两者在
    整段都低于阈值时并不相同，用错了会把「刚测到的液面」当成八帧前的旧读数。
    """
    streak = 0
    for value in reversed(values):
        if value > threshold:
            break
        streak += 1
    if streak == 0:
        return 0, values[-1] if values else None
    return streak, values[-1]


def _margin(session: Mapping[str, Any], latest: float, params: ChangeParams) -> tuple[str, float | None]:
    """条件 B：余量是否真的不够了。返回 (种类, 数值)。

    顺序按「越直接观测越优先」：近空是眼睛能看见的，ETA 是推算的。
    """
    if latest <= params.near_empty_ratio:
        return "near_empty", latest
    eta = session.get("eta_minutes")
    if eta is not None and float(eta) <= params.eta_window_minutes:
        return "eta", float(eta)
    volume = session.get("estimated_volume_ml")
    if volume is not None and float(volume) <= params.margin_ml:
        return "volume", float(volume)
    return "none", None


def evaluate_change(
    session: Mapping[str, Any], params: ChangeParams | None = None, *, now: str | None = None
) -> ChangeDecision:
    """对一袋会话做一次换液点名判定（纯函数，不写库、不推进状态）。

    返回的 ``target_state`` 是判定后应当处于的状态；``changed=False`` 表示
    「维持原状」，此时 ``trigger_reason`` 为空。
    """
    p = params or ChangeParams()
    bag_seq = str(session.get("bag_seq") or "")
    session_id = session.get("session_id")
    current_state = str(session.get("state") or "monitoring")
    values = _visible_ratios(session)
    streak, latest = _consecutive_below(values, p.change_threshold_ratio)
    # 抑制门禁认的是**原因码**：`pickup_suppressed` 的库默认值是 1，只看布尔位会把
    # 还没算过体积的会话也当成「被压住」，理由却是空的。估计器两者一起写，故取与。
    suppression_reason = session.get("suppression_reason")
    suppressed = bool(session.get("pickup_suppressed")) and bool(suppression_reason)

    def _decision(
        target_state: str,
        changed: bool,
        trigger_reason: str | None,
        margin_kind: str = "none",
        margin_value: float | None = None,
        detail: str = "",
    ) -> ChangeDecision:
        return ChangeDecision(
            bag_seq=bag_seq,
            session_id=session_id,
            target_state=target_state,
            changed=changed,
            trigger_reason=trigger_reason,
            consecutive_frames=streak,
            latest_fill_ratio=latest,
            change_threshold_ratio=p.change_threshold_ratio,
            margin_kind=margin_kind,
            margin_value=margin_value,
            suppression_reason=suppression_reason,
            reason_detail=detail,
            decided_at=now or utc_now(),
        )

    if current_state == "replaced":
        return _decision(current_state, False, None, detail="已换袋终态，不再参与点名判定")

    if not values:
        return _decision(current_state, False, None, detail="没有可见帧的液面观测")

    kind, value = _margin(session, latest if latest is not None else values[-1], p)
    if suppressed and kind != "near_empty":
        return _decision(
            current_state,
            False,
            None,
            detail=f"该袋被标定门禁压住（{suppression_reason}），不给换液点名",
        )

    if streak < p.n_frames:
        return _decision(
            current_state,
            False,
            None,
            detail=f"只有 {streak} 帧低于阈值，少于要求的连续 {p.n_frames} 帧",
        )
    if kind == "none":
        # 条件 A 满足但条件 B 不满足：液面是低了，可余量还够撑到下一轮巡视。
        # 双条件的意义就在这里——只有条件 A 的话，护士会在每次长输注里被反复点名。
        return _decision(
            current_state,
            False,
            None,
            kind,
            value,
            detail=f"连续 {streak} 帧低于阈值，但余量还没到紧急线（ETA/毫升/近空均未触发）",
        )

    reason = {
        "near_empty": "near_empty_observed",
        "eta": "eta_within_window",
        "volume": "volume_below_margin",
    }[kind]
    detail = f"连续 {streak} 帧低于 {p.change_threshold_ratio:.2f}，余量判定={kind}"
    if current_state == "needs_change":
        return _decision("needs_change", False, None, kind, value, "已处于换液点名状态")
    return _decision("needs_change", True, reason, kind, value, detail)


def apply_decision(
    sessions: Any,
    decision: ChangeDecision,
    *,
    audit: Any | None = None,
    request_id: str | None = None,
    decision_source: str = "threshold_only",
) -> tuple[dict[str, Any], bool]:
    """把判定落到库上：需要点名就推 ``monitoring → needs_change``。

    返回 (会话行, 是否真的推了状态)。被拒的迁移**不写业务行**，只写审计行——
    状态推不动时系统要留痕，而不是把非法迁移硬写进去。
    """
    record = sessions.get(decision.session_id) if decision.session_id else None
    if record is None:
        raise ApiError("NOT_FOUND", f"session_id={decision.session_id} 不存在；无法应用判定")
    if not decision.changed:
        return record, False

    check_transition(
        str(record.get("state") or "monitoring"),
        decision.target_state,
        session_id=record["session_id"],
        bag_seq=record.get("bag_seq"),
        audit=audit,
        request_id=request_id,
    )
    # 审计行与状态推进**同事务**（task 33）：点名判出来之后如果审计写不进去，
    # 这一袋就保持 monitoring、下一轮重新判，而不是变成一个查无实据的 needs_change。
    audit_record = None
    if audit is not None and request_id:
        audit_record = {
            "request_id": request_id,
            "actor": SERVICE_ACTOR,
            "action": "state.change_proposed",
            "target": record["session_id"],
            "outcome": "accepted",
            "bag_seq": record.get("bag_seq"),
            "bag_key": record.get("infusion_order_no") and f"{record.get('infusion_order_no')}|{record.get('bag_seq')}",
            "rack_id": record.get("rack_id"),
            "bed_no": record.get("bed_no"),
            "from_state": "monitoring",
            "to_state": decision.target_state,
            "detail": decision.to_json(),
        }
    updated = sessions.apply_state_transition(
        record["session_id"],
        decision.target_state,
        audit=audit if audit_record is not None else None,
        audit_record=audit_record,
        trigger_reason=decision.trigger_reason,
        decision_source=decision_source,
    )
    return updated, True


# ── replaced：只能由护士确认 ───────────────────────────────────────────────


@dataclass(frozen=True)
class ReplacementConfirmation:
    """护士确认后的结果。``client_event_id`` 是出站幂等键的原料（task 10 落库）。"""

    session_id: str
    bag_seq: str
    state: str
    operator_id: str
    confirmed_at: str
    event_payload: dict[str, Any]


def confirm_replacement(
    sessions: Any,
    session_id: str,
    *,
    operator_id: str,
    audit: Any | None = None,
    request_id: str | None = None,
    now: str | None = None,
) -> ReplacementConfirmation:
    """护士确认「这袋已经换掉了」：``needs_change → replaced``。

    **本函数是 ``replaced`` 的唯一入口**：入参里的 ``operator_id`` 必填——没有护士
    确认就没有换液事件。确认成功后返回出站事件载荷（``infusion.bag.changed``），
    实际 HTTP 写回由 M5 适配层执行（task 11），幂等键去重由 task 10 承接。
    """
    if not operator_id:
        raise ApiError(
            "INGEST_INVALID",
            "operator_id 不能为空：换液状态只能由护士在终端确认后推进",
            issues=[
                Issue(
                    code="field_missing",
                    field="operator_id",
                    reason="请求里没有确认人",
                    suggestion="请在移动护理终端上用当前登录的工号确认换液；系统不接受无人确认的换液。",
                )
            ],
        )
    record = sessions.get(session_id)
    if record is None:
        raise ApiError("NOT_FOUND", f"session_id={session_id} 不存在；请用上传响应里 bags[].session_id")

    check_transition(
        str(record.get("state") or "monitoring"),
        "replaced",
        session_id=session_id,
        bag_seq=record.get("bag_seq"),
        audit=audit,
        request_id=request_id,
    )
    stamp = now or utc_now()
    # 幂等键由 (会话, 袋序) 唯一确定，不含确认时刻——重放同一次确认必然算出同一个键
    # （生成规则见 build_client_event_id；task 10 的查重直接吃这个值）。
    client_event_id = build_client_event_id(session_id, str(record.get("bag_seq") or ""))
    # 与点名同一条纪律（task 33）：确认动作与 state.confirmed 审计行同事务。
    # 「谁在几点确认换了哪一袋」是飞检最常问的一句话，这句话对应的审计行不允许缺席。
    audit_record = None
    if audit is not None and request_id:
        audit_record = {
            "request_id": request_id,
            "actor": f"nurse:{operator_id}",
            "action": "state.confirmed",
            "target": session_id,
            "outcome": "accepted",
            "bag_seq": record.get("bag_seq"),
            "bag_key": record.get("infusion_order_no")
            and f"{record.get('infusion_order_no')}|{record.get('bag_seq')}",
            "rack_id": record.get("rack_id"),
            "bed_no": record.get("bed_no"),
            "from_state": "needs_change",
            "to_state": "replaced",
            "detail": {"client_event_id": client_event_id, "occurred_at": stamp},
            "occurred_at": stamp,
        }
    updated = sessions.apply_state_transition(
        session_id,
        "replaced",
        audit=audit if audit_record is not None else None,
        audit_record=audit_record,
        trigger_reason="nurse_confirmed",
        decision_source="nurse_confirmed",
    )
    payload = {
        "client_event_id": client_event_id,
        "event_type": "infusion.bag.changed",
        "session_id": session_id,
        "rack_id": record.get("rack_id"),
        "bed_no": record.get("bed_no"),
        "infusion_order_no": record.get("infusion_order_no"),
        "bag_seq": record.get("bag_seq"),
        "fill_ratio": record.get("fill_ratio"),
        "estimated_volume_ml": record.get("estimated_volume_ml"),
        "eta_minutes": record.get("eta_minutes"),
        "trigger_reason": "nurse_confirmed",
        "decision_source": "nurse_confirmed",
        "operator_id": operator_id,
        "occurred_at": stamp,
    }
    return ReplacementConfirmation(
        session_id=session_id,
        bag_seq=str(record.get("bag_seq") or ""),
        state=updated.get("state", "replaced"),
        operator_id=operator_id,
        confirmed_at=stamp,
        event_payload=payload,
    )