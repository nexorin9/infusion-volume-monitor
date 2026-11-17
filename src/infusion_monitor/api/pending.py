"""待写回队列的两个端点：查与重放。

* ``GET /api/v1/infusion/pending`` —— **只读**：信息科看「有哪些换液事件还没送到
  护理执行平台、欠了多久、最后一次错在哪」。它不写任何东西，也不推进任何状态。
* ``POST /api/v1/infusion/pending/{event_id}/replay`` —— **人工重放**：把某一笔
  欠账再送一次。

**谁会用**：护士长看清单（「今天有几袋换液没写回去」），信息科做处置（「对端恢复了，
把这几条补送上去」）。两个角色问的问题不一样，所以分成两个端点而不是一个带参数的。

**重放的三条纪律**（写在这里，因为它们是「绝不静默丢事件」这个承诺的兑现方式）：

1. **不产生第二条业务事件**。重放只投**已存在**的 ``event_id``，请求体里的
   ``client_event_id`` 与首次完全相同——对端凭同一个键去重，护理执行平台上不会
   出现「B01 换袋三条」。
2. **已送达的不重发**。队列状态为 ``sent`` 时直接回 ``idempotent_hit=true``，
   **一次 HTTP 都不发**。信息科连点两下不会多出一条写回。
3. **谁来补的要留痕**。入参必须带 ``operator_id``，审计链里落一行
   ``writeback.replayed``（或 ``writeback.replay_failed``）——飞检查的是
   「这条欠账最后是谁补的」，不是「系统有没有重试过」。

**失败怎么报**：重放没送出去时统一回 409 ``WRITEBACK_REJECTED``——从入站角度看，
「对端不可达」与「对端业务拒绝」是同一件事：护理执行平台此刻没有接受这笔换液。
两者的区别放在 ``issues[].reason`` 与响应体的 ``kind`` 里（``http_4xx`` 是业务拒绝，
``http_5xx`` / ``read_timeout`` 是对端不可达），处置方式不一样：前者要先改输注单号
或袋序，后者等对端恢复再点一次重放。**这笔欠账仍留在清单里，不会丢。**
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Header, Query, Request

from ..adapters.retry_queue import (
    PENDING_STATUSES,
    STATUS_FAILED,
    STATUS_SENT,
    WritebackQueue,
    replay_entry,
    summarize,
)
from ..errors import ApiError, Issue
from ..logging_utils import request_id_of
from ..store.repositories import utc_now
from .schemas import PendingItemOut, PendingListOut, PendingReplayIn, PendingReplayOut

__all__ = ["router", "build_pending_list"]

logger = logging.getLogger("infusion_monitor.api.pending")

router = APIRouter(prefix="/api/v1/infusion", tags=["writeback"])


def build_pending_list(
    queue: WritebackQueue,
    *,
    statuses: tuple[str, ...] = PENDING_STATUSES,
    rack_id: str | None = None,
    bed_no: str | None = None,
    limit: int = 100,
) -> dict:
    """把队列读成一份可直接回给信息科的清单（纯读）。"""
    entries = queue.list_pending(statuses=statuses, rack_id=rack_id, bed_no=bed_no, limit=limit)
    summary = summarize(entries)
    return {
        "count": summary["count"],
        "generated_at": utc_now(),
        "by_status": summary["by_status"],
        "dead_letter_count": summary["dead_letter_count"],
        "queued_count": summary["queued_count"],
        "failed_count": summary["failed_count"],
        "items": [entry.to_json() for entry in entries],
    }


@router.get(
    "/pending",
    response_model=PendingListOut,
    response_model_exclude_none=True,
    summary="待写回清单（只读）：还没送到护理执行平台的换液事件",
    description=(
        "列出所有**还没送达**的出站换液事件：欠了多久、试过几次、最后一次错在哪。\n\n"
        "* 三种状态：`queued`（等自动重试）/ `dead_letter`（自动重试耗尽，等人处理）"
        "/ `failed`（对端业务拒绝，等人处理）；已送达的 `sent` 不出现在清单里；\n"
        "* **本端点只读**：不重发、不改状态、不推进任何业务状态；\n"
        "* 要补送请调 `POST /api/v1/infusion/pending/{event_id}/replay`。"
    ),
)
async def list_pending(
    request: Request,
    rack_id: str | None = Query(default=None, description="只列这一架的欠账"),
    bed_no: str | None = Query(default=None, description="只列这一床的欠账"),
    status: str | None = Query(
        default=None, description="只列某种状态：queued / dead_letter / failed"
    ),
    limit: int = Query(default=100, ge=1, le=500, description="单次最多返回多少条"),
) -> PendingListOut:
    store = request.app.state.store
    request_id = request_id_of(request)
    statuses = PENDING_STATUSES
    if status:
        if status not in PENDING_STATUSES:
            raise ApiError(
                "INGEST_INVALID",
                f"status={status} 不是待写回清单里的状态",
                issues=[
                    Issue(
                        code="status_not_listed",
                        field="status",
                        reason=f"允许的状态：{', '.join(PENDING_STATUSES)}",
                        suggestion="去掉 status 看全部，或改成上面三种之一。",
                    )
                ],
            )
        statuses = (status,)
    payload = build_pending_list(
        store.writeback_queue,
        statuses=statuses,
        rack_id=rack_id,
        bed_no=bed_no,
        limit=limit,
    )
    logger.info(
        "待写回清单 request=%s count=%d rack=%s bed=%s",
        request_id,
        payload["count"],
        rack_id or "-",
        bed_no or "-",
    )
    return PendingListOut(**payload)


@router.post(
    "/pending/{event_id}/replay",
    response_model=PendingReplayOut,
    response_model_exclude_none=True,
    summary="人工重放一笔没送出去的换液事件",
    description=(
        "护理执行平台恢复后，由信息科把某一笔欠账再送一次。\n\n"
        "* **幂等**：重放不产生第二条业务事件，对端收到的 `client_event_id` 与首次"
        "完全相同；这笔早已送达时直接返回 `idempotent_hit=true`，一次 HTTP 都不发；\n"
        "* 人工重放**只发一次**（不等退避），请按对端当时的状态重试；\n"
        "* 没送出去时返回 409 `WRITEBACK_REJECTED`（对端不可达与业务拒绝同码，"
        "区别在 issues[].reason 与 kind 里）—— 欠账仍留在清单里，不会丢；\n"
        "* 谁补的要留痕：`operator_id` 必填，审计链落 `writeback.replayed`。"
    ),
)
async def replay_pending(
    event_id: str,
    payload: PendingReplayIn,
    request: Request,
    x_request_id: str | None = Header(default=None),
) -> PendingReplayOut:
    state = request.app.state
    store = state.store
    request_id = (x_request_id or request.headers.get("X-Request-Id") or "").strip()
    request_id = request_id or request_id_of(request)

    result = await replay_entry(
        store.writeback_queue,
        store.sessions,
        store.events,
        state.nurse_exec,
        event_id,
        operator_id=payload.operator_id,
        audit=store.audit_chain,
        request_id=request_id,
    )

    if not result.found:
        raise ApiError(
            "NOT_FOUND",
            f"待写回清单里没有 event_id={event_id} 这一笔",
            issues=[
                Issue(
                    code="pending_entry_not_found",
                    field="event_id",
                    reason=(
                        "队列里没有这一笔：它可能已经送达并结清，"
                        "或者从来没有出站过（护士还没确认换液）。"
                    ),
                    suggestion=(
                        "先 GET /api/v1/infusion/pending 看清单里的 event_id；"
                        "清单里没有的不用重放。"
                    ),
                )
            ],
        )

    entry = store.writeback_queue.get(event_id)
    target = str((entry.session_id if entry else None) or event_id)
    bag_seq = entry.bag_seq if entry else None
    if result.replayed:
        store.audit_chain.append(
            request_id=request_id,
            actor=f"operator:{payload.operator_id}",
            action="writeback.replayed",
            target=target,
            outcome="accepted",
            bag_seq=bag_seq,
            rack_id=entry.rack_id if entry else None,
            bed_no=entry.bed_no if entry else None,
            detail={
                "event_id": event_id,
                "client_event_id": result.client_event_id,
                "operator_id": payload.operator_id,
                "attempts": result.attempts,
                "queue_attempt_count": result.queue_attempt_count,
            },
        )
    elif not result.ok:
        store.audit_chain.append(
            request_id=request_id,
            actor=f"operator:{payload.operator_id}",
            action="writeback.replay_failed",
            target=target,
            outcome="rejected",
            bag_seq=bag_seq,
            rack_id=entry.rack_id if entry else None,
            bed_no=entry.bed_no if entry else None,
            detail={
                "event_id": event_id,
                "client_event_id": result.client_event_id,
                "operator_id": payload.operator_id,
                "kind": result.kind,
                "error": result.last_error,
                "queue_status": result.queue_status,
            },
        )
    # 幂等命中（这笔早已送达、本次一次 HTTP 都没发）：**不写审计行**。
    # 什么都没发生却留下「有人重放过」的痕迹，飞检材料里就会出现
    # 「同一条事件送出两次」的错觉——与出站侧 ``already_sent`` 不写审计同一口径。

    logger.info(
        "写回重放 event=%s operator=%s ok=%s kind=%s queue=%s",
        event_id,
        payload.operator_id,
        result.ok,
        result.kind,
        result.queue_status,
    )

    if not result.ok:
        # 对端不可达与对端业务拒绝**共用一个错误码**：从「入站」的角度看，
        # 这两件事是同一件——护理执行平台此刻没有接受这笔换液事件。
        # 区别放在 issues[].reason 与响应体的 kind 里（http_4xx 是业务拒绝，
        # http_5xx / read_timeout 是对端不可达），处置方式不一样：
        # 前者要先改输注单号或袋序，后者等对端恢复再点一次重放。
        # 这笔欠账仍留在待写回清单里，**不会丢**。
        raise ApiError(
            "WRITEBACK_REJECTED",
            f"护理执行平台此刻没有接受这笔换液事件（{result.kind}），欠账仍在待写回清单里",
            issues=[
                Issue(
                    code="peer_not_accepted",
                    field="event_id",
                    reason=result.last_error or result.kind,
                    suggestion=(
                        "kind=http_4xx 时核对这袋的输注单号、床号与袋序是否与护理执行平台"
                        "登记的一致；kind 是 5xx 或超时时先确认对端服务在跑，"
                        "恢复后在待写回清单里点重放即可——这笔事件不会被丢掉。"
                    ),
                )
            ],
        )

    return PendingReplayOut(**result.to_json())
