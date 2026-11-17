"""护士确认换液：``POST /api/v1/infusion/transition``。

**这是 ``replaced`` 的唯一 HTTP 入口。** 系统自己判出来的 ``needs_change`` 只是建议——
中间必须隔着护士点一下。没有这一步，一次误报就变成一次没人执行的拔管换袋。

一次确认做三件事，且**只在这一条路径上发生**：

1. **推进状态** ``needs_change → replaced``（:func:`identity.state_machine.confirm_replacement`）；
2. **落一条出站事件**（幂等键由 (会话, 袋序) 唯一确定，见
   :func:`~infusion_monitor.identity.state_machine.build_client_event_id`）；
3. **写回护理执行平台**（真实 HTTP + Bearer 凭据，5xx 指数退避 ≤2 次后置
   ``pending_writeback``）。

**幂等键生成规则由服务端唯一持有。** 病区 Wi-Fi 抖动会让同一个点击重发好几次，若把
确认时刻拼进键，每次重试都是一条新事件，护理执行平台上就会出现三条「B01 已换」而 B01
只真的换过一次。终端**可以**把点名清单里的 ``client_event_id`` 原样回填（本项目就这么
发），但**必须与服务端算出来的一致**——不一致直接 422，不接受终端自编的键。

**重放不是错误。** 同一个确认再点一次，返回 200 + 既有事件 + ``idempotent_hit=true``，
而不是弹个红框让护士以为换液没成功、于是再点第三次。

**点名依据随确认一起留痕**：确认会覆盖会话上的 ``trigger_reason``，所以这里在确认
之前先读出「这袋当初是被凭什么点名的」，连同确认一起写进响应——护士长事后复盘
「谁在什么时候按什么依据确认的换液」有据可查，而不只是看到一个 ``nurse_confirmed``。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Header, Request

from ..adapters.retry_queue import retry_at
from ..adapters.writeback import WritebackOutcome, WritebackRefused, deliver_event_async
from ..errors import ApiError, Issue
from ..identity.idempotency import authorize_session_write, confirm_replacement_once
from ..identity.state_machine import build_client_event_id
from ..logging_utils import request_id_of
from ..store.repositories import utc_now
from .schemas import TransitionIn, TransitionOut, WritebackOut

__all__ = ["expected_client_event_id", "router"]

logger = logging.getLogger("infusion_monitor.api.transition")

router = APIRouter(prefix="/api/v1/infusion", tags=["identity"])


def expected_client_event_id(session_id: str, bag_seq: str) -> str:
    """确认入参里 ``client_event_id`` 的**唯一**正确取值。

    独立成一个函数是为了让它可被对照测试直接钉住：幂等键规则散在拼接表达式里的
    时候，改一次格式没人会发现，而改一次格式就是一次全量换液事件重复的来源。
    """
    return build_client_event_id(session_id, bag_seq)


def _writeback_out(outcome: WritebackOutcome) -> WritebackOut:
    return WritebackOut(
        status=outcome.writeback_status,
        ok=outcome.ok,
        attempts=outcome.attempts,
        kind=outcome.kind,
        refused=outcome.refused,
        reason=outcome.refusal_reason,
        next_retry_at=outcome.next_retry_at,
    )


@router.post(
    "/transition",
    response_model=TransitionOut,
    response_model_exclude_none=True,
    summary="护士确认换液（needs_change → replaced 并写回执行平台）",
    description=(
        "护士在移动护理终端上点「已换液」时调用。\n\n"
        "* 幂等键由服务端按 (session_id, bag_seq) 固定生成；入参可原样回填点名清单里的"
        "`client_event_id`，与规则不一致时返回 422 而不是按调用方的键入账；\n"
        "* 重复确认返回 200 与 `idempotent_hit=true`，不产生第二条事件；\n"
        "* 非法迁移（monitoring 直跳 replaced、replaced 回退）返回 409 并留审计行；\n"
        "* 写回失败指数退避 ≤2 次，耗尽后事件置 `pending_writeback` 并告警——**绝不丢弃**。"
    ),
)
async def confirm_transition(
    payload: TransitionIn,
    request: Request,
    x_request_id: str | None = Header(default=None),
) -> TransitionOut:
    state = request.app.state
    store = state.store
    request_id = (x_request_id or request.headers.get("X-Request-Id") or "").strip()
    request_id = request_id or request_id_of(request)

    record = store.sessions.get(payload.session_id)
    if record is None:
        raise ApiError(
            "NOT_FOUND",
            f"session_id={payload.session_id} 不存在；请用点名清单 items[].session_id",
            issues=[
                Issue(
                    code="session_not_found",
                    field="session_id",
                    reason=f"库里没有 session_id={payload.session_id}",
                    suggestion="请用 GET /api/v1/infusion/pickups 返回的 items[].session_id。",
                )
            ],
        )

    # 越权判定放在最前面：袋序对不上就别谈幂等键——用错误的袋序算出来的键本身
    # 就是错的，先拒掉才不会让一个错键有机会被写进事件表。
    authorize_session_write(
        record,
        bag_seq=payload.bag_seq,
        rack_id=payload.rack_id,
        audit=store.audit_chain,
        request_id=request_id,
        operator_id=payload.operator_id,
    )

    expected_key = expected_client_event_id(str(record["session_id"]), payload.bag_seq)
    if payload.client_event_id and payload.client_event_id != expected_key:
        raise ApiError(
            "INGEST_INVALID",
            "client_event_id 与服务端按 (session_id, bag_seq) 生成的键不一致，已拒绝",
            issues=[
                Issue(
                    code="client_event_id_mismatch",
                    field="client_event_id",
                    reason=(
                        f"请求给了 {payload.client_event_id}，"
                        f"而这袋的幂等键按规则应为 {expected_key}"
                    ),
                    suggestion=(
                        "幂等键不要在终端自编：直接回填 GET /api/v1/infusion/pickups 里"
                        "该条目的 client_event_id，或整个字段留空由服务端生成。"
                        "自编的键会让同一次换液在对端变成多条事件。"
                    ),
                )
            ],
        )

    # 点名依据：确认会把 trigger_reason 覆盖成 nurse_confirmed，先留一份。
    pickup_evidence = {
        "previous_state": str(record.get("state") or "monitoring"),
        "previous_trigger_reason": record.get("trigger_reason"),
        "previous_eta_minutes": record.get("eta_minutes"),
        "previous_fill_ratio": record.get("fill_ratio"),
        "bed_no": record.get("bed_no"),
        "rack_id": record.get("rack_id"),
    }

    # 两步分开调（而不是直接用同步的 confirm_and_deliver）：确认是纯本地库操作，
    # 留在处理函数线程里；写回要发 HTTP，放进 await 让出事件循环——否则同进程上的
    # 上传请求会被一次同步网络调用卡住。SQLite 连接的线程亲和性也因此不受影响。
    transition = confirm_replacement_once(
        store.sessions,
        store.events,
        str(record["session_id"]),
        bag_seq=payload.bag_seq,
        operator_id=payload.operator_id,
        rack_id=payload.rack_id,
        audit=store.audit_chain,
        request_id=request_id,
        now=payload.confirmed_at,
    )

    if not transition.event_id:
        # 幂等命中但事件行缺失：异常形态。不猜、不补发，如实回报让信息科查审计链。
        outcome = WritebackOutcome(
            event_id="",
            client_event_id=transition.client_event_id,
            writeback_status="pending",
            ok=False,
            refused=True,
            refusal_reason="event_missing",
            last_error="幂等命中但事件行缺失，未写回；请查审计链。",
        )
    else:
        try:
            outcome = await deliver_event_async(
                store.sessions,
                store.events,
                state.nurse_exec,
                str(transition.event_id),
                audit=store.audit_chain,
                request_id=request_id,
                queue=store.writeback_queue,
            )
        except WritebackRefused as refusal:
            # 状态已推进但出站被门禁拒发（例如会话在并发里被改回非 replaced）：
            # **不把确认回滚**——护士确实已经换了，让护士重做到比让护士以为没换强。
            # 事件留在库里等写回队列处理，这里如实说明。
            logger.warning(
                "换液确认后写回被拒 event=%s reason=%s",
                transition.event_id,
                refusal.reason,
            )
            # 这笔事件确实欠着对端一次确认 → 落进待写回队列（queued，等自动重试）。
            # 不落队列的话它就只躺在事件表里一个 pending 状态位上，没人看得见。
            event_row = store.events.get(str(transition.event_id)) or {}
            if event_row:
                store.writeback_queue.record(
                    event_row,
                    status="queued",
                    attempt_count=int(event_row.get("attempt_count") or 0),
                    error=refusal.message,
                    reason=None,
                    next_retry_at=retry_at(int(event_row.get("attempt_count") or 0)),
                )
            outcome = WritebackOutcome(
                event_id=str(transition.event_id),
                client_event_id=transition.client_event_id,
                writeback_status="pending_writeback",
                ok=False,
                refused=True,
                refusal_reason=refusal.reason,
                last_error=refusal.message,
            )

    logger.info(
        "换液确认 session=%s bag=%s operator=%s created=%s idempotent_hit=%s writeback=%s",
        transition.session_id,
        transition.bag_seq,
        payload.operator_id,
        transition.created,
        transition.idempotent_hit,
        outcome.writeback_status,
    )

    return TransitionOut(
        session_id=transition.session_id,
        bag_seq=transition.bag_seq,
        state=transition.state,
        operator_id=transition.operator_id,
        client_event_id=transition.client_event_id,
        event_id=transition.event_id,
        created=transition.created,
        idempotent_hit=transition.idempotent_hit,
        confirmed_at=transition.occurred_at or utc_now(),
        pickup_evidence=pickup_evidence,
        writeback=_writeback_out(outcome),
    )