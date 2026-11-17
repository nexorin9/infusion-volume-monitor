"""待写回队列：出站欠账的持久台账、死信与人工重放。

**这个模块回答一个问题：护士点过「已换液」，护理执行平台到底收到了没有？**

出站写回有一个绕不开的现实：病区网络会断、护理执行平台会 5xx、机器会在重试的
半路上被重启。这三件事里任何一件发生，「护士已经换过袋、平台却不知道」就会变成
一条**静默丢失**的换液记录——护士端显示已换，护理执行平台上没有这一袋执行记录，
对账时两边对不上，而且没人知道是哪一袋。

所以「绝不静默丢事件」不能只是一句口号，它得落在**盘上**：

* 队列表 ``writeback_queue`` 一行记一笔欠账：载荷快照、已尝试次数、下次可重试
  时刻、最后一次错误摘要、当前状态；
* 出站失败**先落队列再返回**，进程重启后欠账还在（不是进程内存里的一张清单）；
* 自动重试按指数退避（:data:`BACKOFF_SCHEDULE_SECONDS` = 1s / 4s / 16s）走，
  尝试预算用完（首次 + ≤2 次重试）就进**死信** ``dead_letter``，等人来看；
* 信息科可以查 ``GET /api/v1/infusion/pending``、重放 ``POST …/pending/{event_id}/replay``。
  **重放幂等**：重放成功不产生第二条业务事件，只把队列那一行从欠账改成已送达。

**队列不产生事件。** ``infusion_event`` 是事件的唯一真源；队列只是同一笔事
「还欠着一次确认」的另一份台账。两张表靠 ``event_id`` 一一对应（唯一索引），
所以「重放会不会写出第二条换液事件」这个问题的答案是结构性的：不会。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from ..store.repositories import utc_now
from .nurse_exec import OutboundEvent, WritebackAttempt
from .writeback import backoff_seconds as _inline_backoff_seconds

if TYPE_CHECKING:  # pragma: no cover - 仅类型标注
    from .writeback import WritebackOutcome

__all__ = [
    "BACKOFF_SCHEDULE_SECONDS",
    "MAX_ATTEMPTS",
    "PENDING_STATUSES",
    "RETRY_BUDGET",
    "STATUS_DEAD_LETTER",
    "STATUS_FAILED",
    "STATUS_QUEUED",
    "STATUS_SENT",
    "WritebackQueue",
    "backoff_seconds",
    "drain_due",
    "replay_entry",
    "retry_at",
]

logger = logging.getLogger("infusion_monitor.adapters.retry_queue")

#: 指数退避表：第 N 次尝试失败后，等第 N 项秒再试。末项之后不再变（一直等 16s）。
#: 1s / 4s / 16s 这个序列对应「病区网络抖一下」的量级——一两分钟就能自愈，
#: 而拖到分钟级以上的重试已经没有意义，该交给人处理。
#:
#: 这张表**不是另起一份**：它取自 :func:`adapters.writeback.backoff_seconds`
#: （基数 1s、倍数 4 的同一条指数曲线），只是把「重试序号」换成了「已尝试次数」。
#: 两处必须一致——一处决定「等多久再试」，一处决定「这条欠账什么时候能自动重试」，
#: 对不上就会出现「表上写着 16s 可重试、实际 4s 后就重试」的错位。
BACKOFF_SCHEDULE_SECONDS: tuple[float, ...] = (
    _inline_backoff_seconds(0),
    _inline_backoff_seconds(1),
    _inline_backoff_seconds(2),
)

#: 一次出站的**总尝试预算**：首次 + ≤2 次重试。用完即进死信。
MAX_ATTEMPTS = 3
#: 预算里允许的重试次数（总预算减掉首次那一次）。
RETRY_BUDGET = MAX_ATTEMPTS - 1

STATUS_QUEUED = "queued"
STATUS_DEAD_LETTER = "dead_letter"
STATUS_SENT = "sent"
STATUS_FAILED = "failed"

#: 待处理清单默认包含的状态：三种都**还没送达**。``sent`` 不在其中——已送达的
#: 事情不欠账，摆在待办里只会让人以为还有活没干。
PENDING_STATUSES: tuple[str, ...] = (STATUS_QUEUED, STATUS_DEAD_LETTER, STATUS_FAILED)


def backoff_seconds(attempt_count: int) -> float:
    """第 ``attempt_count`` 次尝试之后、下一次尝试之前等多少秒。

    ``attempt_count`` 是**已经试过几次**：试过 1 次等 1s，试过 2 次等 4s，
    试过 3 次及以上等 16s（末项不再增长——再长的等待也等不来一个不同的答案，
    该让人来看了）。
    """
    index = max(0, int(attempt_count) - 1)
    return _inline_backoff_seconds(min(index, len(BACKOFF_SCHEDULE_SECONDS) - 1))


def retry_at(attempt_count: int, *, now: datetime | None = None) -> str:
    """下一次可重试时刻（ISO8601）。"""
    base = now or datetime.now(timezone.utc)
    return (base + timedelta(seconds=backoff_seconds(attempt_count))).isoformat(timespec="seconds")


@dataclass
class QueueEntry:
    """队列里的一行（读出来的形状）。"""

    queue_id: str
    event_id: str
    client_event_id: str
    status: str
    attempt_count: int = 0
    session_id: str | None = None
    rack_id: str | None = None
    bed_no: str | None = None
    bag_seq: str | None = None
    next_retry_at: str | None = None
    last_error: str | None = None
    dead_letter_reason: str | None = None
    last_attempt_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "QueueEntry":
        return cls(
            queue_id=str(row.get("queue_id") or ""),
            event_id=str(row.get("event_id") or ""),
            client_event_id=str(row.get("client_event_id") or ""),
            status=str(row.get("status") or STATUS_QUEUED),
            attempt_count=int(row.get("attempt_count") or 0),
            session_id=row.get("session_id"),
            rack_id=row.get("rack_id"),
            bed_no=row.get("bed_no"),
            bag_seq=row.get("bag_seq"),
            next_retry_at=row.get("next_retry_at"),
            last_error=row.get("last_error"),
            dead_letter_reason=row.get("dead_letter_reason"),
            last_attempt_at=row.get("last_attempt_at"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            payload=dict(row.get("payload") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        """出参形状。**不含载荷全文**：清单是给人看的，载荷按需单独取。"""
        return {
            "queue_id": self.queue_id,
            "event_id": self.event_id,
            "client_event_id": self.client_event_id,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "session_id": self.session_id,
            "rack_id": self.rack_id,
            "bed_no": self.bed_no,
            "bag_seq": self.bag_seq,
            "next_retry_at": self.next_retry_at,
            "last_error": self.last_error,
            "dead_letter_reason": self.dead_letter_reason,
            "last_attempt_at": self.last_attempt_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "replayable": self.status in (STATUS_DEAD_LETTER, STATUS_FAILED, STATUS_QUEUED),
        }


class WritebackQueue:
    """待写回队列的门面：仓储 + 策略。

    策略只有三条，其余都是存取：

    1. **只有欠了账的事件才进队列**。未确认的换液（``pending``）不是欠账——
       护士还没点确认，它压根没打算发出去；把它摆进待办里只会让人以为漏了事。
    2. **可达的失败进 ``queued``，耗尽预算的进 ``dead_letter``**。前者等自动重试，
       后者等人。
    3. **送达即出清单**（``sent`` 不在待处理清单里），但那一行**留在库里**：
       它是「这笔欠账什么时候结清的」的唯一凭据，删掉就再也说不清了。
    """

    def __init__(self, repository: Any, events: Any | None = None) -> None:
        self._repo = repository
        self._events = events

    @property
    def repository(self) -> Any:
        return self._repo

    # ── 记 ────────────────────────────────────────────────────────────
    def record(
        self,
        event_row: Mapping[str, Any],
        *,
        status: str,
        attempt_count: int,
        error: str | None = None,
        reason: str | None = None,
        next_retry_at: str | None = None,
    ) -> dict[str, Any]:
        """把一笔出站欠账写进队列（upsert，同一事件只有一行）。"""
        payload = dict(event_row.get("payload") or {})
        for key in ("event_id", "client_event_id", "session_id", "rack_id", "bed_no", "bag_seq"):
            if payload.get(key) is None and event_row.get(key) is not None:
                payload[key] = event_row.get(key)
        row, _created = self._repo.enqueue(
            {
                "event_id": event_row.get("event_id"),
                "client_event_id": event_row.get("client_event_id"),
                "session_id": event_row.get("session_id"),
                "rack_id": event_row.get("rack_id"),
                "bed_no": event_row.get("bed_no"),
                "bag_seq": event_row.get("bag_seq"),
                "payload": payload,
                "attempt_count": int(attempt_count),
                "next_retry_at": next_retry_at,
                "last_error": error,
                "status": status,
                # 停在「等人处理」两种状态的原因码：自动重试耗尽 / 对端业务拒绝。
                "dead_letter_reason": reason if status in (STATUS_DEAD_LETTER, STATUS_FAILED) else None,
                "last_attempt_at": utc_now(),
            }
        )
        return row

    def record_outcome(self, event_row: Mapping[str, Any], outcome: "WritebackOutcome") -> dict[str, Any] | None:
        """把一次出站结果折成队列状态。

        * **送达**（``sent``）：若这一笔此前在队列里（说明它欠过账），把它结清；
          没有就什么都不做——一次就成功的出站不该在队列里留下痕迹。
        * **拒发**（会话未确认）：不是欠账，不入队。
        * **对端业务拒绝**（4xx，``failed``）：入队并标 ``failed``，等人处理；
          重发同一份载荷只会再被拒一次，但「它被拒过、拒的原因是什么」必须留着。
        * **退避耗尽**（``pending_writeback``）：入队并标 ``dead_letter``，等人工重放。

        **尝试次数是累计的**：这一轮发了几次，就往上加几次。写本轮的单次次数会让
        「一共试过几次」永远停在 1——信息科看着像只试过一次，实际可能已经试了五次，
        处置判断就跟着错。
        """
        if outcome.refused:
            return None
        event_id = str(event_row.get("event_id"))
        existing = self._repo.get_by_event_id(event_id)
        attempts = int(outcome.attempts or 0)
        total = attempts + int((existing or {}).get("attempt_count") or 0)
        if outcome.ok:
            if existing is None:
                return None
            return self._repo.update(
                existing["queue_id"],
                status=STATUS_SENT,
                attempt_count=total,
                last_error=None,
                dead_letter_reason=None,
                last_attempt_at=utc_now(),
            )
        status = STATUS_FAILED if outcome.writeback_status == "failed" else STATUS_DEAD_LETTER
        reason = "business_rejected" if status == STATUS_FAILED else "retries_exhausted"
        return self.record(
            event_row,
            status=status,
            attempt_count=total,
            error=outcome.last_error,
            reason=reason,
            next_retry_at=outcome.next_retry_at or retry_at(total),
        )

    def reconcile(self, events: Any | None = None) -> list[dict[str, Any]]:
        """从事件表补齐队列里缺的行。

        **为什么需要这一步**：出站的尝试发生在网络调用上，进程可能在重试到一半时
        被重启——那一笔事件停在 ``pending`` 且 ``attempt_count>0``，但队列行还没
        落盘（队列是在一次投递**结束**时才写的）。重启后没有人知道还欠着这笔。
        这里按事件表反查，把这类事件补进队列：不丢、不猜、只按已落库的字段补。

        只补 ``pending`` 且 ``attempt_count>0`` 的：``pending`` 且一次没试过的是
        「护士还没确认」，那不是欠账；``pending_writeback`` 的按定义已经有队列行。
        """
        source = events if events is not None else self._events
        if source is None:
            return []
        created: list[dict[str, Any]] = []
        for row in source.list_by_writeback_status("pending"):
            attempts = int(row.get("attempt_count") or 0)
            if attempts <= 0:
                continue
            if self._repo.get_by_event_id(str(row.get("event_id"))) is not None:
                continue
            entry = self.record(
                row,
                status=STATUS_QUEUED,
                attempt_count=attempts,
                error=row.get("last_error"),
                reason=None,
                next_retry_at=retry_at(attempts),
            )
            created.append(entry)
        return created

    # ── 读 ────────────────────────────────────────────────────────────
    def get(self, event_id: str) -> QueueEntry | None:
        row = self._repo.get_by_event_id(event_id)
        return QueueEntry.from_row(row) if row else None

    def list_pending(
        self,
        *,
        statuses: Sequence[str] = PENDING_STATUSES,
        rack_id: str | None = None,
        bed_no: str | None = None,
        limit: int = 100,
    ) -> list[QueueEntry]:
        rows = self._repo.list_pending(
            statuses=statuses, rack_id=rack_id, bed_no=bed_no, limit=limit
        )
        return [QueueEntry.from_row(row) for row in rows]

    def due(self, *, now: str | None = None, limit: int = 20) -> list[QueueEntry]:
        """到期可自动重试的条目（``queued`` 且 ``next_retry_at`` 已到）。

        只有 ``queued`` 会被自动重试：``dead_letter`` 的自动预算已经用完，
        再自动发下去就是「无限重试」——那正是这个模块要避免的事。
        """
        moment = now or utc_now()
        rows = self._repo.list_by_status(STATUS_QUEUED, due_before=moment, limit=limit)
        return [QueueEntry.from_row(row) for row in rows]

    def count_by_status(self) -> dict[str, int]:
        return self._repo.count_by_status()

    # ── 改 ────────────────────────────────────────────────────────────
    def mark_sent(self, event_id: str, *, attempt_count: int | None = None) -> dict[str, Any] | None:
        existing = self._repo.get_by_event_id(event_id)
        if existing is None:
            return None
        fields: dict[str, Any] = {
            "status": STATUS_SENT,
            "last_error": None,
            "dead_letter_reason": None,
            "last_attempt_at": utc_now(),
        }
        if attempt_count is not None:
            fields["attempt_count"] = int(attempt_count)
        return self._repo.update(existing["queue_id"], **fields)

    def mark_dead_letter(self, event_id: str, *, error: str | None = None,
                         reason: str = "retries_exhausted") -> dict[str, Any] | None:
        existing = self._repo.get_by_event_id(event_id)
        if existing is None:
            return None
        return self._repo.update(
            existing["queue_id"],
            status=STATUS_DEAD_LETTER,
            last_error=error if error is not None else existing.get("last_error"),
            dead_letter_reason=reason,
            last_attempt_at=utc_now(),
        )


# ── 自动重试与人工重放 ──────────────────────────────────────────────────────


@dataclass
class ReplayResult:
    """一次重放的结果。"""

    event_id: str
    client_event_id: str
    found: bool
    replayed: bool
    idempotent_hit: bool = False
    writeback_status: str = "pending"
    ok: bool = False
    kind: str = "ok"
    attempts: int = 0
    reason: str | None = None
    last_error: str | None = None
    queue_status: str | None = None
    queue_attempt_count: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "client_event_id": self.client_event_id,
            "found": self.found,
            "replayed": self.replayed,
            "idempotent_hit": self.idempotent_hit,
            "writeback_status": self.writeback_status,
            "ok": self.ok,
            "kind": self.kind,
            "attempts": self.attempts,
            "reason": self.reason,
            "last_error": self.last_error,
            "queue_status": self.queue_status,
            "queue_attempt_count": self.queue_attempt_count,
        }


async def replay_entry(
    queue: WritebackQueue,
    sessions: Any,
    events: Any,
    client: Any,
    event_id: str,
    *,
    operator_id: str,
    audit: Any | None = None,
    request_id: str | None = None,
) -> ReplayResult:
    """信息科手动重放一笔没送出去的换液事件。

    **重放幂等的三条保证**：

    1. **不产生第二条业务事件**。重放只投**已存在的** ``event_id``，从不新建
       事件行；对端拿到的 ``client_event_id`` 与首次完全相同，重复消费由对端
       凭同一个键去重。
    2. **已送达的不重发**。队列状态为 ``sent``（或事件本身已 ``sent``）时直接
       返回 ``idempotent_hit=true``，**一次 HTTP 都不发**。
    3. **只更新队列状态**。送达后那一行从「欠账」改成「已结清」，待处理清单
       里不再出现；事件表里的行数不变。

    单次重放**只发一次**（``max_retries=0``）：这是人在点按钮，应该立刻给他
    「送到了 / 没送到」的答复，而不是让请求线程睡 5 秒。
    """
    from .writeback import deliver_event_async

    entry = queue.get(event_id)
    event_row = events.get(event_id)

    if event_row is None:
        return ReplayResult(
            event_id=event_id,
            client_event_id=entry.client_event_id if entry else "",
            found=False,
            replayed=False,
            reason="event_not_found",
            last_error=f"事件 {event_id} 已不在事件表里，无法重放；请查审计链。",
        )

    if entry is None:
        # 队列里没有这一笔，但事件本身已经送达 → 幂等命中（**不发 HTTP**）。
        # 这不是错误：这一笔压根没欠过账，或者早就结清了。
        if str(event_row.get("writeback_status")) == "sent":
            return ReplayResult(
                event_id=event_id,
                client_event_id=str(event_row.get("client_event_id") or ""),
                found=True,
                replayed=False,
                idempotent_hit=True,
                ok=True,
                writeback_status="sent",
                kind="already_sent",
                reason="already_sent",
            )
        return ReplayResult(
            event_id=event_id,
            client_event_id=str(event_row.get("client_event_id") or ""),
            found=False,
            replayed=False,
            reason="pending_entry_not_found",
            last_error=(
                f"队列里没有 event_id={event_id} 这一笔；"
                "它可能已经送达并结清，或从来没有出站过。"
            ),
        )

    if entry.status == STATUS_SENT or str(event_row.get("writeback_status")) == "sent":
        return ReplayResult(
            event_id=event_id,
            client_event_id=entry.client_event_id,
            found=True,
            replayed=False,
            idempotent_hit=True,
            ok=True,
            writeback_status="sent",
            kind="already_sent",
            reason="already_sent",
            queue_status=entry.status,
            queue_attempt_count=entry.attempt_count,
        )

    outcome = await deliver_event_async(
        sessions,
        events,
        client,
        event_id,
        audit=audit,
        request_id=request_id,
        max_retries=0,
        expected_state=None,
        queue=queue,
    )
    updated = queue.get(event_id)
    return ReplayResult(
        event_id=event_id,
        client_event_id=entry.client_event_id,
        found=True,
        replayed=outcome.ok,
        idempotent_hit=False,
        writeback_status=outcome.writeback_status,
        ok=outcome.ok,
        kind=outcome.kind,
        attempts=outcome.attempts,
        reason=outcome.refusal_reason,
        last_error=outcome.last_error,
        queue_status=updated.status if updated else None,
        queue_attempt_count=updated.attempt_count if updated else entry.attempt_count,
    )


async def drain_due(
    queue: WritebackQueue,
    sessions: Any,
    events: Any,
    client: Any,
    *,
    audit: Any | None = None,
    request_id: str | None = None,
    now: str | None = None,
    limit: int = 20,
) -> list[ReplayResult]:
    """把**到期**的队列条目各发一次（自动重试的一拍）。

    只动 ``queued`` 且 ``next_retry_at`` 已到的条目；``dead_letter`` 不在其中——
    它的自动预算已经用完，再自动发下去就是无限重试。

    每次重试同样只发一次（``max_retries=0``）：退避表管的是**两次重试之间等多久**，
    不是一次重试内部试几次。把两者混在一起会算错预算。
    """
    results: list[ReplayResult] = []
    for entry in queue.due(now=now, limit=limit):
        result = await replay_entry(
            queue,
            sessions,
            events,
            client,
            entry.event_id,
            operator_id="service:writeback-retry",
            audit=audit,
            request_id=request_id,
        )
        results.append(result)
    return results


def event_from_entry(entry: QueueEntry) -> OutboundEvent:
    """从队列里的载荷快照装配出站事件。

    **按快照发，不按当前代码重新解释**：事件落库那一刻的字段形态才是对账依据；
    用新代码重新解释旧载荷，会让「发了什么」和「记了什么」对不上。
    """
    payload = dict(entry.payload)
    payload.setdefault("event_id", entry.event_id)
    payload.setdefault("client_event_id", entry.client_event_id)
    return OutboundEvent.from_event_row(payload)


def attempts_to_dead_letter(attempt_count: int) -> bool:
    """这次尝试之后是否已经用完自动重试预算。"""
    return int(attempt_count) >= MAX_ATTEMPTS


def summarize(entries: Iterable[QueueEntry]) -> dict[str, Any]:
    """把队列压成一组可核对的数（运行摘要与端点响应共用）。"""
    by_status: dict[str, int] = {}
    for entry in entries:
        by_status[entry.status] = by_status.get(entry.status, 0) + 1
    return {
        "count": sum(by_status.values()),
        "by_status": by_status,
        "dead_letter_count": by_status.get(STATUS_DEAD_LETTER, 0),
        "queued_count": by_status.get(STATUS_QUEUED, 0),
        "failed_count": by_status.get(STATUS_FAILED, 0),
    }


def attempt_kind_of(attempt: WritebackAttempt) -> str:
    """一次尝试的结果分类（透传，便于调用方只依赖本模块的常量）。"""
    return attempt.kind
