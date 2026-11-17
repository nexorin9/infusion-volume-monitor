"""写回投递策略：谁能被发出去、失败怎么退避、耗尽后落在哪。

三条规则，每条对应一个真实会在病区发生的事故：

1. **未确认的 needs_change 永不出站**。本服务只「建议该换液」，换没换是护士
   点一下的事。没点确认就发事件，等于系统替护士签字换掉了某一袋——对端收到
   就会真的换新袋，病人手上那袋还在输。所以 :func:`deliver_event` 在**发请求前**
   先查会话状态：不是 ``replaced`` 一律拒发（``writeback.refused``），事件行留在
   库里等护士确认，绝不悄悄删掉也绝不发出去。

2. **退避重试 ≤2 次，耗尽置 pending_writeback**。病区网络抖一下是常事，对端
   5xx 或读超时重试一次多半就好。重试次数要有上限：无限重试等于把一条永远发不
   出去的事件挂在服务里无限刷日志，而信息科从运行摘要里根本看不出有这么一条。
   耗尽之后保留事件、置 ``pending_writeback``、写审计——**丢事件是不可接受的**，
   让运维知道「有 3 条没送出去」是可以的。**该哪类失败值得重试**由
   :data:`~adapters.nurse_exec.FAILURE_CLASSES` 说了算，本模块不另判。

3. **错误摘要可复核且不含凭据**。``last_error`` 走
   :func:`~adapters.nurse_exec.failure_summary`：状态码、尝试次数、**按类别分别
   计数**、对端给的业务拒绝原因、最后一次错误文本，一行齐活；再经
   :func:`~adapters.nurse_exec.scrub` 抹掉 token 值。飞检与内审都会导出这张表，
   凭据不该出现在里面。分开计数是有用的：「连接超时 ×3」指向病区网络，
   「读超时 ×3」指向对端处理慢，写成一个「失败 3 次」两边都查不到。

**线程模型**：HTTP 调用是阻塞的，SQLite 连接带线程亲和性。因此
:func:`deliver_event_async` 只把**纯网络**的那一步丢进线程池，落库与审计回到
调用线程——绝不出现「在工作线程里写主线程的连接」。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Sequence

from .nurse_exec import (
    RETRYABLE_FAILURE_KINDS,
    NurseExecClient,
    OutboundEvent,
    WritebackAttempt,
    failure_class,
    failure_summary,
    scrub,
)

logger = logging.getLogger("infusion_monitor.adapters.writeback")

__all__ = [
    "BACKOFF_BASE_SECONDS",
    "BACKOFF_FACTOR",
    "MAX_RETRIES",
    "RETRYABLE_KINDS",
    "WritebackOutcome",
    "WritebackRefused",
    "backoff_seconds",
    "confirm_and_deliver",
    "count_kinds",
    "deliver_event",
    "deliver_event_async",
]

#: 初始尝试之外的重试次数上限。
MAX_RETRIES = 2
#: 指数退避基数（秒）：第 1 次重试等 base，第 2 次等 base × factor。
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_FACTOR = 4

#: 值得退避重试的失败类别。**从 nurse_exec 的 :data:`FAILURE_CLASSES` 派生**，
#: 不在本模块另抄一份——两处各写一张名单，加一种失败时漏改一处就会出现
#: 「分类说不可重试、退避却照发」或反过来的矛盾处置。
RETRYABLE_KINDS = RETRYABLE_FAILURE_KINDS

#: 会话必须处于此状态才允许出站。
CONFIRMED_STATE = "replaced"


class WritebackRefused(Exception):
    """拒发：事件不满足出站前提（会话未被护士确认）。**不发任何 HTTP 请求。**"""

    def __init__(self, reason: str, message: str, *, detail: Mapping[str, Any] | None = None) -> None:
        self.reason = reason
        self.message = message
        self.detail = dict(detail or {})
        super().__init__(message)


@dataclass
class WritebackOutcome:
    """一次投递的结果。``writeback_status`` 是库里的终态。"""

    event_id: str
    client_event_id: str
    writeback_status: str
    ok: bool
    attempts: int = 0
    kind: str = "ok"
    refused: bool = False
    refusal_reason: str | None = None
    last_error: str | None = None
    next_retry_at: str | None = None
    attempt_kinds: list[str] = field(default_factory=list)
    backoff_delays: list[float] = field(default_factory=list)
    #: **按类别分别计数**：``{"connect_timeout": 2, "read_timeout": 1}``。
    #: 连接超时与读超时的病根完全不同（病区网络 vs 对端处理慢），分开数
    #: 信息科才知道该查哪一头；合成一个总数就只剩「失败了 3 次」。
    kind_counts: dict[str, int] = field(default_factory=dict)
    #: 最后一次尝试拿到的 HTTP 状态码（传输层失败时为 None）。
    status_code: int | None = None
    #: 4xx 时从对端响应体抽出的原因；业务拒绝才认它。
    rejection_reason: str | None = None
    #: 对端可能已经收下这条事件（读超时/写超时/半开连接）——重发靠幂等键兜底。
    peer_may_have_applied: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "client_event_id": self.client_event_id,
            "writeback_status": self.writeback_status,
            "ok": self.ok,
            "attempts": self.attempts,
            "kind": self.kind,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "last_error": self.last_error,
            "next_retry_at": self.next_retry_at,
            "attempt_kinds": list(self.attempt_kinds),
            "backoff_delays": list(self.backoff_delays),
            "kind_counts": dict(self.kind_counts),
            "status_code": self.status_code,
            "rejection_reason": self.rejection_reason,
            "peer_may_have_applied": self.peer_may_have_applied,
        }

    def to_transition_json(self) -> dict[str, Any]:
        """给 ``POST …/transition`` 响应用的形状（task 31 复用）。"""
        return {
            "writeback_status": self.writeback_status,
            "writeback_ok": self.ok,
            "writeback_attempts": self.attempts,
            "writeback_kind": self.kind,
            "writeback_kind_counts": dict(self.kind_counts),
            "writeback_status_code": self.status_code,
            "writeback_rejection_reason": self.rejection_reason,
            "writeback_peer_may_have_applied": self.peer_may_have_applied,
            "writeback_refused": self.refused,
            "writeback_reason": self.refusal_reason,
            "next_retry_at": self.next_retry_at,
        }


def count_kinds(kinds: Sequence[str]) -> dict[str, int]:
    """按失败类别分别计数，保持首次出现的顺序。"""
    counts: dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def backoff_seconds(retry_index: int, *, base: float = BACKOFF_BASE_SECONDS,
                    factor: float = BACKOFF_FACTOR) -> float:
    """第 ``retry_index`` 次重试前等待的秒数（从 0 起）：base × factor^n。"""
    return base * (factor ** max(0, retry_index))


def _iso_in(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(timespec="seconds")


def _audit(audit: Any, request_id: str | None, **kwargs: Any) -> None:
    if audit is None:
        return
    try:
        audit.append(request_id=request_id or f"auto-{kwargs.get('target', '-')}", **kwargs)
    except ValueError:
        # 审计动作码尚未登记时不吞掉业务结果：写回状态已经落库，
        # 审计漏一行不该反过来把写回变成失败（task 33 会把码补齐）。
        pass


def _load_event(events: Any, event_id: str) -> dict[str, Any]:
    row = events.get(event_id)
    if row is None:
        raise WritebackRefused(
            "event_not_found",
            f"event_id={event_id} 不在库中；没有可写回的事件。",
            detail={"event_id": event_id},
        )
    return row


def _authorize_outbound(row: Mapping[str, Any], sessions: Any) -> None:
    """出站前的门禁：**会话必须是护士已确认的 replaced**。

    这是本模块最重要的一行。系统自己判出来的 ``needs_change`` 只是一条建议，
    对端收到写回会真的换新袋——所以没有护士确认就没有 HTTP 请求。
    """
    session_id = row.get("session_id")
    if not session_id:
        return  # 无会话关联的事件（如对端预置回执）按既有形态放行
    record = sessions.get(str(session_id))
    if record is None:
        raise WritebackRefused(
            "session_not_found",
            f"事件 {row.get('event_id')} 指向的会话 {session_id} 已不存在，拒发。",
            detail={"session_id": session_id},
        )
    state = str(record.get("state") or "monitoring")
    if state != CONFIRMED_STATE:
        raise WritebackRefused(
            "session_not_confirmed",
            (
                f"会话 {session_id} 当前是 {state}，不是 {CONFIRMED_STATE}："
                "换液必须由护士在终端确认后才写回，未确认的换液点名不会出站。"
            ),
            detail={
                "session_id": session_id,
                "bag_seq": row.get("bag_seq"),
                "session_state": state,
            },
        )


def _finalise(
    *,
    events: Any,
    row: Mapping[str, Any],
    outcome: WritebackOutcome,
    audit: Any,
    request_id: str | None,
    queue: Any | None = None,
) -> WritebackOutcome:
    """把投递结果落库 + 写审计（只在这里碰数据库）。

    落库与审计**不共用事务**：写回结果是对端的事实，审计是对这次动作的记录，
    两者分开写才能保证「对端已收但审计写失败」不会反过来把状态回滚成 pending
    ——那会导致同一条事件被重发一遍，而对端已经收过了。

    ``queue`` 是待写回队列（task 37）：出站欠账在这里落盘。**队列与事件表分开写**
    是有意的——队列是「提醒与重放」的台账，它写失败不该把已经送达的事实回滚；
    真写失败时记一行警告日志，由运行摘要的待写回条数暴露出来，不静默。
    """
    events.mark_writeback(
        outcome.event_id,
        outcome.writeback_status,
        attempt_count=outcome.attempts,
        next_retry_at=outcome.next_retry_at,
        last_error=outcome.last_error,
    )
    if queue is not None:
        try:
            queue.record_outcome(row, outcome)
        except Exception as exc:  # pragma: no cover - 队列写失败不得回滚已送达的事实
            logger.warning(
                "待写回队列落盘失败 event=%s error=%s",
                outcome.event_id,
                type(exc).__name__,
            )
    actor = "service:writeback"
    target = str(row.get("session_id") or outcome.event_id)
    bag_seq = row.get("bag_seq")
    # 每次失败后的退避重试单独留一行：一条事件重试两次 vs 重试一次，在飞检时
    # 讲的是两回事（前者说明病区网络当时确实有问题）。
    for index, kind in enumerate(outcome.attempt_kinds[:-1]):
        _audit(
            audit, request_id, actor=actor, action="writeback.retry", target=target,
            outcome="accepted", bag_seq=bag_seq,
            detail={
                "event_id": outcome.event_id,
                "client_event_id": outcome.client_event_id,
                "attempt": index + 1,
                "kind": kind,
                "backoff_seconds": outcome.backoff_delays[index]
                if index < len(outcome.backoff_delays) else None,
            },
        )
    if outcome.refused:
        _audit(
            audit, request_id, actor=actor, action="writeback.refused", target=target,
            outcome="rejected", bag_seq=bag_seq,
            detail={"reason": outcome.refusal_reason, "client_event_id": outcome.client_event_id,
                    "message": str(outcome.last_error or "")[:200]},
        )
    elif outcome.ok:
        _audit(
            audit, request_id, actor=actor, action="writeback.sent", target=target,
            outcome="accepted", bag_seq=bag_seq,
            detail={"event_id": outcome.event_id, "client_event_id": outcome.client_event_id,
                    "attempts": outcome.attempts, "kind": outcome.kind},
        )
    elif outcome.writeback_status == "failed":
        _audit(
            audit, request_id, actor=actor, action="writeback.rejected", target=target,
            outcome="rejected", bag_seq=bag_seq,
            detail={"event_id": outcome.event_id, "client_event_id": outcome.client_event_id,
                    "kind": outcome.kind, "attempts": outcome.attempts,
                    "status_code": outcome.status_code,
                    "kind_counts": dict(outcome.kind_counts),
                    "rejection_reason": outcome.rejection_reason,
                    "error": outcome.last_error},
        )
    else:
        _audit(
            audit, request_id, actor=actor, action="writeback.pending", target=target,
            outcome="accepted", bag_seq=bag_seq,
            detail={"event_id": outcome.event_id, "client_event_id": outcome.client_event_id,
                    "kind": outcome.kind, "attempts": outcome.attempts,
                    "status_code": outcome.status_code,
                    "kind_counts": dict(outcome.kind_counts),
                    "peer_may_have_applied": outcome.peer_may_have_applied,
                    "error": outcome.last_error, "next_retry_at": outcome.next_retry_at},
        )
    return outcome


def _run_attempts(
    client: NurseExecClient,
    event: OutboundEvent,
    *,
    max_retries: int,
    backoff_base: float,
    sleeper: Callable[[float], None],
) -> tuple[WritebackAttempt, int, list[str], list[float]]:
    """跑「首次 + ≤max_retries 次重试」，返回 (最后一次尝试, 次数, 类别轨迹, 退避轨迹)。

    只发 HTTP，不碰库。
    """
    kinds: list[str] = []
    delays: list[float] = []
    total = max(0, max_retries) + 1
    attempt = client.attempt(event)
    kinds.append(attempt.kind)
    used = 1
    while not attempt.ok and attempt.kind in RETRYABLE_KINDS and used < total:
        delay = backoff_seconds(used - 1, base=backoff_base)
        delays.append(delay)
        sleeper(delay)
        attempt = client.attempt(event)
        kinds.append(attempt.kind)
        used += 1
    return attempt, used, kinds, delays


def _refused_outcome(row: Mapping[str, Any], refusal: WritebackRefused,
                     client: NurseExecClient) -> WritebackOutcome:
    """拒发 outcome：状态**不动**（保持 pending 等护士确认），只留一行审计。"""
    return WritebackOutcome(
        event_id=str(row["event_id"]),
        client_event_id=str(row.get("client_event_id") or ""),
        writeback_status=str(row.get("writeback_status") or "pending"),
        ok=False,
        refused=True,
        refusal_reason=refusal.reason,
        last_error=scrub(refusal.message, client.api_token),
    )


def _already_sent_outcome(row: Mapping[str, Any]) -> WritebackOutcome:
    """已送出的事件不重复外发。

    对端凭 ``client_event_id`` 也会去重，但重发只是白白占一条病区带宽，还会让
    对端侧的接收时间戳变得难以追溯（护理部对账时看到同一袋三个接收时间会来问）。
    """
    return WritebackOutcome(
        event_id=str(row["event_id"]),
        client_event_id=str(row.get("client_event_id") or ""),
        writeback_status="sent",
        ok=True,
        attempts=int(row.get("attempt_count") or 0),
        kind="already_sent",
    )


def _outcome_from_attempt(
    row: Mapping[str, Any],
    event: OutboundEvent,
    attempt: WritebackAttempt,
    used: int,
    kinds: Sequence[str],
    delays: Sequence[float],
    client: NurseExecClient,
) -> WritebackOutcome:
    """最后一次尝试 → 库里的终态。可重试耗尽落 pending_writeback，业务拒绝落 failed。

    终态一律取自 :data:`~adapters.nurse_exec.FAILURE_CLASSES`，不在这里再判一次
    「这类该不该重试」——分类与处置是同一张表的两个投影。
    """
    counts = count_kinds(kinds)
    common = {
        "event_id": str(row["event_id"]),
        "client_event_id": event.client_event_id,
        "attempts": used,
        "kind": attempt.kind,
        "attempt_kinds": list(kinds),
        "backoff_delays": list(delays),
        "kind_counts": counts,
        "status_code": attempt.status_code,
        "rejection_reason": attempt.rejection_reason,
        "peer_may_have_applied": attempt.peer_may_have_applied,
    }
    if attempt.ok:
        return WritebackOutcome(writeback_status="sent", ok=True, last_error=None, **common)
    # 耗尽时的摘要四件套：状态码、尝试次数、按类别分别计数、对端原因 / 最后错误。
    summary = failure_summary(
        attempt.kind,
        status_code=attempt.status_code,
        attempts=used,
        counts=counts,
        reason=attempt.rejection_reason,
        last_error=attempt.error,
        secrets=(client.api_token,),
    )
    spec = failure_class(attempt.kind)
    if spec.retryable:
        delay = backoff_seconds(max(0, used - 1))
        return WritebackOutcome(
            writeback_status="pending_writeback", ok=False, last_error=summary,
            next_retry_at=_iso_in(delay), **common,
        )
    return WritebackOutcome(
        writeback_status=spec.terminal_status, ok=False, last_error=summary, **common
    )


def _prepare(
    sessions: Any,
    events: Any,
    client: NurseExecClient,
    event_id: str,
    expected_state: str | None,
) -> tuple[dict[str, Any] | None, OutboundEvent | None, WritebackOutcome | None]:
    """出站前的读取与门禁。返回 (事件行, 待发事件, 已定的 outcome)。

    三个返回值互斥：有 outcome 就直接落审计返回；否则按 (row, event) 走网络。
    """
    row = _load_event(events, event_id)
    if expected_state is not None:
        try:
            _authorize_outbound(row, sessions)
        except WritebackRefused as refusal:
            return row, None, _refused_outcome(row, refusal, client)
    if str(row.get("writeback_status")) == "sent":
        return row, None, _already_sent_outcome(row)
    return row, OutboundEvent.from_event_row(row), None


def _settle(events: Any, row: Mapping[str, Any] | None, outcome: WritebackOutcome,
            audit: Any | None, request_id: str | None, queue: Any | None = None) -> WritebackOutcome:
    """门禁拒发要落审计；已送出的重放**什么都不写**。

    重放一次不该多出一条 ``writeback.sent``——那会让飞检材料里出现「同一条事件送
    出两次」的错觉。拒发则相反：系统不动过一次，是要留痕的。
    """
    if outcome.kind == "already_sent":
        return outcome
    return _finalise(
        events=events, row=row or {}, outcome=outcome, audit=audit, request_id=request_id,
        queue=queue,
    )


def deliver_event(
    sessions: Any,
    events: Any,
    client: NurseExecClient,
    event_id: str,
    *,
    audit: Any | None = None,
    request_id: str | None = None,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
    expected_state: str | None = CONFIRMED_STATE,
    queue: Any | None = None,
) -> WritebackOutcome:
    """投递一条已落库的换液事件（同步版：网络调用与落库同在调用线程）。

    :param expected_state: 出站门禁要求的状态；传 ``None`` 可关掉门禁（仅供
        信息科重放已确认事件的运维路径使用，见 task 37）。
    :param queue: 待写回队列（:class:`adapters.retry_queue.WritebackQueue`）。
        给了它，出站欠账才会落盘——**不给就等于没有队列**，退避耗尽的事件只剩
        事件表里的一个状态位，没人看得见也重放不了。
    :raises WritebackRefused: 事件不存在（未落库的 event_id 无法投递）。
    """
    row, event, decided = _prepare(sessions, events, client, event_id, expected_state)
    if decided is not None:
        return _settle(events, row, decided, audit, request_id, queue)
    attempt, used, kinds, delays = _run_attempts(
        client, event, max_retries=max_retries, backoff_base=backoff_base, sleeper=sleeper
    )
    outcome = _outcome_from_attempt(row or {}, event, attempt, used, kinds, delays, client)
    return _finalise(
        events=events, row=row or {}, outcome=outcome, audit=audit, request_id=request_id,
        queue=queue,
    )


async def deliver_event_async(
    sessions: Any,
    events: Any,
    client: NurseExecClient,
    event_id: str,
    *,
    audit: Any | None = None,
    request_id: str | None = None,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE_SECONDS,
    sleeper: Callable[[float], None] | None = None,
    expected_state: str | None = CONFIRMED_STATE,
    queue: Any | None = None,
) -> WritebackOutcome:
    """投递事件的异步外壳：**只在网络那一步用线程池**，落库回到调用线程。

    出站 HTTP 是阻塞调用，放在事件循环里会卡住同一进程上的上传请求；但 SQLite
    连接带线程亲和性，又不能在别的线程里读写。折中：读事件行、判门禁、落库、
    写审计全部留在调用线程，把「发 HTTP + 退避等待」整段放进线程池。
    """
    from starlette.concurrency import run_in_threadpool

    row, event, decided = _prepare(sessions, events, client, event_id, expected_state)
    if decided is not None:
        return _settle(events, row, decided, audit, request_id, queue)
    # 走零参 lambda 而不是逐个透传：``_run_attempts`` 的三个参数是 keyword-only 的，
    # 直接按位置传会在真发请求时抛 TypeError（task 31 接上 HTTP 端点时才暴露出来）。
    attempt, used, kinds, delays = await run_in_threadpool(
        lambda: _run_attempts(
            client,
            event,
            max_retries=max_retries,
            backoff_base=backoff_base,
            sleeper=sleeper or time.sleep,
        )
    )
    outcome = _outcome_from_attempt(row or {}, event, attempt, used, kinds, delays, client)
    return _finalise(
        events=events, row=row or {}, outcome=outcome, audit=audit, request_id=request_id,
        queue=queue,
    )


def confirm_and_deliver(
    sessions: Any,
    events: Any,
    client: NurseExecClient,
    session_id: str,
    *,
    bag_seq: str,
    operator_id: str,
    rack_id: str | None = None,
    audit: Any | None = None,
    request_id: str | None = None,
    now: str | None = None,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
    queue: Any | None = None,
) -> tuple[Any, WritebackOutcome]:
    """护士确认换液 → 落出站事件 → 立即写回。**主路径的一条龙**。

    返回 ``(TransitionOutcome, WritebackOutcome)``。``TransitionOutcome.created``
    为 ``False`` 说明这是一次重放：事件早已落库，此时按它当前的写回状态决定要不要
    再发一次（已 sent 就不再发；pending/pending_writeback/failed 说明上次没送出去，
    护士再点一次正是我们希望它重试的时机）。
    """
    from ..identity.idempotency import confirm_replacement_once

    transition = confirm_replacement_once(
        sessions,
        events,
        session_id,
        bag_seq=bag_seq,
        operator_id=operator_id,
        rack_id=rack_id,
        audit=audit,
        request_id=request_id,
        now=now,
    )
    if not transition.event_id:
        # 幂等命中但没有事件行（异常形态）：不猜、不发，如实回报。
        return transition, WritebackOutcome(
            event_id="",
            client_event_id=transition.client_event_id,
            writeback_status="pending",
            ok=False,
            refused=True,
            refusal_reason="event_missing",
            last_error="幂等命中但事件行缺失，未写回；请查审计链。",
        )
    outcome = deliver_event(
        sessions,
        events,
        client,
        str(transition.event_id),
        audit=audit,
        request_id=request_id,
        max_retries=max_retries,
        backoff_base=backoff_base,
        sleeper=sleeper,
        queue=queue,
    )
    return transition, outcome