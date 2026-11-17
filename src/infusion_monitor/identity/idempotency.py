"""幂等键、并发重放与越权边界。

三条约束，每条都对应现场一种**真的会发生**的失败方式——不是防御性编程的洁癖：

1. **同一次确认只产生一条出站事件**。护士在移动护理终端上点「已换液」，网络超时、
   屏幕卡住、病区 Wi-Fi 抖动都会让同一个点击被重发好几次。若每次重发都追加一条事件，
   护理执行平台上就会出现三条「B01 已换」，而 B01 只真的换过一次；对账时没人说得清
   哪条是真的。幂等键 :func:`build_client_event_id` 由 (会话, 袋序) 唯一确定，
   重放必然算出同一个键。

2. **并发两路确认必须串行化**。病区两台终端可能同时显示同一条点名（同一次巡视的
   推送到了两个护士的手机）。若两路都通过检查就会写出两条事件。做法是：读路径先判
   一次 → 写路径拿到 ``BEGIN IMMEDIATE`` 写锁后**在锁内复核**前提 → 输的那一路回到
   读路径重判，落到「幂等命中」或 409。谁赢都不会双写。

3. **拒绝越权写入**。确认请求必须点名是哪一袋（``bag_seq``），带了就必须与会话里的
   袋序一致。带错袋序意味着「给这一轮的 B01 会话写一条 B02 的换液事件」——轻则事件
   挂错袋、重则把别人的换液记录改成自己的。程序不猜、不兼容：403 + 一行 rejected 审计。

**被拒的动作同样要留审计**（:mod:`store.audit_chain`）：飞检查的不是「系统做对了什么」，
而是「系统是不是在不该动的时候动了」。因此所有**失败**判定都发生在事务**外**——
事务一旦开启再抛异常就会连审计行一起回滚掉，恰好把最该留痕的那次尝试抹掉。
事务里只装成功路径的写入，失败则整笔回滚（此时还没有任何东西需要保留）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..errors import ApiError, Issue, field_suggestion
from ..store.db import transaction
from .state_machine import (
    ReplacementConfirmation,
    build_client_event_id,
    check_transition,
    confirm_replacement,
)

__all__ = [
    "MAX_CONFIRM_ATTEMPTS",
    "TransitionOutcome",
    "authorize_session_write",
    "build_client_event_id",
    "confirm_replacement_once",
]

#: 锁内复核失败后的重试上限。正常情况下第 2 次就能定下来（读路径看到已落库的事件）；
#: 3 次都拿不到结论说明有别的东西在持续改状态，宁可明确报错也不无限重试。
MAX_CONFIRM_ATTEMPTS = 3


class _LostRace(Exception):
    """内部信号：拿到写锁后复核发现「别人先到了」，回读路径重判一次。

    不对外暴露，也不写审计——它不是一次被拒的请求，只是本路径需要重走一步。
    """


@dataclass(frozen=True)
class TransitionOutcome:
    """一次幂等确认的结果。

    ``idempotent_hit=True`` 表示这次调用**没有**产生新事件：同一次确认的重放，
    返回的是既有事件行。调用方（task 31 的 HTTP 端点）据此回 ``{"idempotent_hit": true}``。
    """

    session_id: str
    bag_seq: str
    state: str
    client_event_id: str
    event_id: str | None
    created: bool
    idempotent_hit: bool
    operator_id: str
    occurred_at: str
    event_payload: dict[str, Any]
    session: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "bag_seq": self.bag_seq,
            "state": self.state,
            "client_event_id": self.client_event_id,
            "event_id": self.event_id,
            "created": self.created,
            "idempotent_hit": self.idempotent_hit,
            "operator_id": self.operator_id,
            "occurred_at": self.occurred_at,
            "writeback_status": "pending" if self.created else None,
        }


# ── 越权边界 ──────────────────────────────────────────────────────────────


def _reject_write(
    *,
    record: Mapping[str, Any] | None,
    error_code: str,
    code: str,
    field_name: str,
    reason: str,
    suggestion: str,
    message: str,
    detail: Mapping[str, Any],
    audit: Any | None,
    request_id: str | None,
    actor: str,
) -> ApiError:
    """构造一个拒绝异常并落一行 rejected 审计。

    审计与异常是同一件事的两面：光回错误码说不清「谁在什么时候试图给哪一袋写什么」，
    光有审计不回错误码又会让调用方以为写成功了。

    ``error_code`` 区分两类拒绝：``INGEST_INVALID``（422，缺必填字段——按入站契约
    统一处理）与 ``WRITE_NOT_ALLOWED``（403，字段填了但指向别的袋/别的架——这是
    越权，不是填错）。
    """
    if audit is not None and request_id:
        audit.append(
            request_id=request_id,
            actor=actor,
            action="session.write_rejected",
            target=str(record["session_id"]) if record else "-",
            outcome="rejected",
            bag_seq=record.get("bag_seq") if record else None,
            detail={"code": code, "reason": reason, **dict(detail)},
        )
    return ApiError(
        error_code,
        message,
        issues=[Issue(code=code, field=field_name, reason=reason, suggestion=suggestion)],
    )


def authorize_session_write(
    record: Mapping[str, Any],
    *,
    bag_seq: str | None,
    rack_id: str | None = None,
    audit: Any | None = None,
    request_id: str | None = None,
    operator_id: str = "",
) -> None:
    """确认这条写操作确实是在动**这一袋**。不合法则抛 403（并留审计）。

    * ``bag_seq`` 为空 → 不指明是哪一袋的换液确认，不接受；
    * ``bag_seq`` 与会话里的袋序不一致 → 跨袋写入，拒绝；
    * ``rack_id`` 给了且与会话不符 → 跨输液架写入（一病区终端误操作到另一架），拒绝。
    """
    actor = f"nurse:{operator_id}" if operator_id else "nurse:unknown"

    if not bag_seq:
        raise _reject_write(
            record=record,
            error_code="INGEST_INVALID",
            code="bag_seq_missing",
            field_name="bag_seq",
            reason="请求没有点名是哪一袋被换掉",
            suggestion=field_suggestion("bag_seq"),
            message="换液确认必须点名袋序：系统不接受「不指明是哪一袋」的换液记录。",
            detail={"submitted_bag_seq": None},
            audit=audit,
            request_id=request_id,
            actor=actor,
        )

    owned = str(record.get("bag_seq") or "")
    if str(bag_seq) != owned:
        raise _reject_write(
            record=record,
            error_code="WRITE_NOT_ALLOWED",
            code="bag_not_in_session",
            field_name="bag_seq",
            reason=(
                f"这次确认写的是 {bag_seq}，但会话 {record['session_id']} 属于 {owned}"
                f"（输液架 {record.get('rack_id')} {record.get('shift_key')} 轮次）"
            ),
            suggestion=(
                f"请确认护士点的确实是 {owned} 这一袋。若确实换的是另一袋，"
                "请先在候选列表里认领正确袋序，再重新发起确认。"
            ),
            message=f"拒绝把 {bag_seq} 的换液事件写到属于 {owned} 的会话上。",
            detail={"submitted_bag_seq": str(bag_seq), "session_bag_seq": owned},
            audit=audit,
            request_id=request_id,
            actor=actor,
        )

    if rack_id and str(rack_id) != str(record.get("rack_id") or ""):
        raise _reject_write(
            record=record,
            error_code="WRITE_NOT_ALLOWED",
            code="rack_not_in_session",
            field_name="rack_id",
            reason=(
                f"请求的输液架是 {rack_id}，但会话 {record['session_id']} "
                f"属于 {record.get('rack_id')}"
            ),
            suggestion="请核对床号与输液架号；跨输液架确认会把换液记到别的架子上。",
            message=f"拒绝把 {rack_id} 的换液事件写到属于 {record.get('rack_id')} 的会话上。",
            detail={"submitted_rack_id": str(rack_id), "session_rack_id": record.get("rack_id")},
            audit=audit,
            request_id=request_id,
            actor=actor,
        )


# ── 幂等确认 ──────────────────────────────────────────────────────────────


def _outcome(
    record: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    client_event_id: str,
    operator_id: str,
    created: bool,
    payload: Mapping[str, Any] | None = None,
) -> TransitionOutcome:
    return TransitionOutcome(
        session_id=str(record["session_id"]),
        bag_seq=str(record.get("bag_seq") or ""),
        state=str(record.get("state") or "monitoring"),
        client_event_id=client_event_id,
        event_id=event.get("event_id"),
        created=created,
        idempotent_hit=not created,
        operator_id=operator_id,
        occurred_at=str(event.get("occurred_at") or ""),
        event_payload=dict(payload or event.get("payload") or {}),
        session=dict(record),
    )


def confirm_replacement_once(
    sessions: Any,
    events: Any,
    session_id: str,
    *,
    bag_seq: str,
    operator_id: str,
    rack_id: str | None = None,
    audit: Any | None = None,
    request_id: str | None = None,
    now: str | None = None,
    max_attempts: int = MAX_CONFIRM_ATTEMPTS,
) -> TransitionOutcome:
    """护士确认换液的**幂等入口**：推进 ``needs_change → replaced`` 并落出站事件。

    这是 ``confirm_replacement`` 之上的入口，额外负责三件事：幂等键查重、并发互斥、
    越权边界。三者都靠**同一把库写锁**（``BEGIN IMMEDIATE``）串起来，因此**状态推进、
    事件落库、审计写入在同一个事务里**——不存在「状态变了但事件没写」或反之。

    返回的 :class:`TransitionOutcome` 里 ``idempotent_hit`` 直接回答「这次是不是重放」。
    重放**不是错误**：返回 200 + 既有事件，而不是 409——护士的按钮重试应该悄无声息
    地拿到同一个结果，而不是弹一个红框让他以为换液没成功、于是再点一次。
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
                    suggestion=field_suggestion("operator_id"),
                )
            ],
        )

    conn = sessions.conn
    for _attempt in range(max(1, max_attempts)):
        record = sessions.get(session_id)
        if record is None:
            raise ApiError(
                "NOT_FOUND",
                f"session_id={session_id} 不存在；请用上传响应里 bags[].session_id",
            )

        authorize_session_write(
            record,
            bag_seq=bag_seq,
            rack_id=rack_id,
            audit=audit,
            request_id=request_id,
            operator_id=operator_id,
        )

        client_event_id = build_client_event_id(session_id, str(record.get("bag_seq") or ""))

        # ── 读路径：幂等键命中即为重放，直接返回既有事件，不碰事务 ──────────
        existing = events.find_by_client_event_id(client_event_id)
        if existing is not None:
            if audit is not None and request_id:
                audit.append(
                    request_id=request_id,
                    actor=f"nurse:{operator_id}",
                    action="event.replay_ignored",
                    target=session_id,
                    outcome="accepted",
                    bag_seq=record.get("bag_seq"),
                    detail={
                        "client_event_id": client_event_id,
                        "event_id": existing.get("event_id"),
                        "idempotent_hit": True,
                        "note": "同一次确认的重放：读回既有事件，未新增事件行",
                    },
                    occurred_at=existing.get("occurred_at"),
                )
            return _outcome(
                record, existing, client_event_id=client_event_id, operator_id=operator_id,
                created=False,
            )

        # ── 非法迁移 409：在事务外判，rejected 审计才不会被回滚抹掉 ─────────
        check_transition(
            str(record.get("state") or "monitoring"),
            "replaced",
            session_id=session_id,
            bag_seq=record.get("bag_seq"),
            audit=audit,
            request_id=request_id,
        )

        # ── 写路径：拿写锁后复核前提，成功才落库 ──────────────────────────
        confirmation: ReplacementConfirmation
        try:
            with transaction(conn):
                locked = sessions.get(session_id)
                locked_state = str(locked.get("state") or "monitoring")
                if locked_state != str(record.get("state") or "monitoring") or (
                    events.find_by_client_event_id(client_event_id) is not None
                ):
                    # 拿到锁的这段时间里别人推进了状态或落了事件：本路径作废，
                    # 回到读路径按「幂等命中 / 409」重新判定，绝不在这里硬写。
                    raise _LostRace()

                confirmation = confirm_replacement(
                    sessions,
                    session_id,
                    operator_id=operator_id,
                    audit=audit,
                    request_id=request_id,
                    now=now,
                )
                event, created = events.append_event(confirmation.event_payload)
                if created and audit is not None and request_id:
                    audit.append(
                        request_id=request_id,
                        actor=f"nurse:{operator_id}",
                        action="event.recorded",
                        target=session_id,
                        outcome="accepted",
                        bag_seq=confirmation.bag_seq,
                        detail={
                            "client_event_id": client_event_id,
                            "event_id": event.get("event_id"),
                            "idempotent_hit": False,
                            "writeback_status": event.get("writeback_status"),
                        },
                        occurred_at=confirmation.confirmed_at,
                    )
        except _LostRace:
            continue

        if created:
            _record_self_calibration_evidence(
                sessions, session_id, audit=audit, request_id=request_id,
                operator_id=operator_id,
            )

        return _outcome(
            sessions.get(session_id) or record,
            event,
            client_event_id=client_event_id,
            operator_id=operator_id,
            created=created,
            payload=confirmation.event_payload,
        )

    # 重试到上限仍未定下来：别让调用方猜，直接说明这一轮有别的写入在打架
    raise ApiError(
        "STATE_CONFLICT",
        f"session_id={session_id} 的换液确认连续 {max_attempts} 次都没能定下来"
        "（同一会话上有别的写入在并发推进）。请稍后重试，或联系信息科查该会话的审计链。",
        issues=[
            Issue(
                code="concurrent_write_contention",
                field="session_id",
                reason=f"幂等键 {build_client_event_id(session_id, bag_seq)} 在 "
                f"{max_attempts} 次尝试内既没落库也没确认已存在",
                suggestion="请稍候重试一次；若反复出现，把 rack_id + 时间窗报给信息科，"
                "由内审按审计链核对这一会话的改动记录。",
            )
        ],
    )

def _record_self_calibration_evidence(
    sessions: Any,
    session_id: str,
    *,
    audit: Any | None,
    request_id: str | None,
    operator_id: str,
) -> None:
    """换液确认成功后，留一条自校准样本（护士说「这袋输完了」是唯一的残留证据来源）。

    三条纪律：

    * **不与确认争先后**。确认与出站事件已经在同一事务里落定了，这里是**之后**单独
      一次写入——标定是辅助账本，它写不进去不该让护士的确认失败；
    * **写失败必须留痕**。样本丢了就补一条 ``calibration.deviation_failed`` 的 rejected
      审计行，让复核的人看见「这里本该有一条证据而没有」，而不是以为没发生过；
    * **只写一次**。重放路径根本不会走到这里（``created`` 为假），加上
      ``calibration_deviation.session_id`` 的唯一索引，同一会话不可能留下两条。
    """
    from ..store.repositories import DeviationRepository

    try:
        record = sessions.get(session_id)
        if not record:
            return
        DeviationRepository(sessions.conn).record_session(
            record,
            note=f"护士 {operator_id} 确认换液；确认即视为该袋已输完，模型当时给的毫升数即残差",
        )
    except Exception as exc:  # 辅助账本写不进去，不能拖垮确认；留审计说明缺了一条证据
        if audit is not None and request_id:
            try:
                audit.append(
                    request_id=request_id,
                    actor=f"nurse:{operator_id}",
                    action="calibration.deviation_failed",
                    target=session_id,
                    outcome="rejected",
                    bag_seq=(sessions.get(session_id) or {}).get("bag_seq"),
                    detail={
                        "reason": "自校准样本未写入 calibration_deviation",
                        "error": f"{type(exc).__name__}: {exc}",
                        "note": "换液确认与出站事件已落库；这条缺失的证据请人工补登记",
                    },
                )
            except Exception:  # 审计也写不进去时只能放弃，不能连带把确认回滚
                pass
