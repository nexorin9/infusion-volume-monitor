"""保留期策略与过期清理：帧目录归档 / 删除、推理审计裁剪。

**这个模块解决什么问题**：输液架每一轮巡视都留一段 10–20 秒短片，一轮几十帧、
一个输液架一个目录。一间病区一天十几轮、全院几十间，帧目录按月计就会把病区服务器
的盘吃满。容量测算见 ``scripts/estimate_storage.py``，本模块负责「到点收」。

**三条硬纪律**（每条都有测试钉住，写在这里是为了让下一个人别改坏）：

1. **未完成的会话一个都不删。** ``state`` 不是 ``replaced`` 的会话（``monitoring`` /
   ``needs_change``）帧目录原样保留——护士还指着它核对余量，点名刚算出来的袋子里
   就有它。
2. **有写回欠账的会话一个都不删。** 出站事件还没送出去（``pending`` /
   ``pending_writeback`` / ``failed``）或待写回队列里还挂着（``queued`` /
   ``dead_letter``）的会话，帧目录留着：信息科重放那笔欠账时还要靠原始帧核对。
3. **审计链永不删。** ``audit_log`` 是飞检留档（append-only），本模块**没有任何**
   触碰它的路径。只清 ``inference_audit``——那是模型调用元数据，不是业务留档。

**一个输液架一个目录**：一段短片里几只袋子共用**同一个帧目录**
（``frames_root/{rack}_{order}_{shift}``），所以清理的单位是**目录**而不是会话——
一个目录里只要还有一只袋子没到清理条件，整个目录都留着（否则会把还在核对的那只袋
的帧一起删掉）。被拦下的原因会逐袋记在结果里。

**像素归零、读数留下**：帧目录过期后，会话行**保留**（面积比曲线、毫升数、ETA、
换液结论都还在），只把像素清掉，并在 ``retention_archive`` 留一行台账。护士长复盘
「上周三那袋还剩多少」靠的是会话行，不是那堆 jpg。

**落盘顺序**：先动文件、再写库。写库失败时下一轮会把这些会话记成 ``absent``
（目录本来就不在），像素不会丢——反过来先写库再删文件，删失败就再也没人管那堆像素。
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .db import transaction
from .repositories import utc_now

__all__ = [
    "RetentionPolicy",
    "RetentionResult",
    "BLOCK_REASONS",
    "run_retention",
    "plan_retention",
    "retention_overview",
]

logger = logging.getLogger(__name__)

#: 出站事件里「还没了结」的状态：这几种的会话帧目录一律留着。
_UNSETTLED_EVENT_STATES = ("pending", "pending_writeback", "failed")

#: 待写回队列里「还欠着」的状态。
_BLOCKING_QUEUE_STATES = ("queued", "dead_letter")

#: 会话被留下的原因码（出现在结果与台账里，信息科看得懂为什么没清）。
BLOCK_REASONS = {
    "within_retention": "还在保留期内，不动",
    "session_not_finished": "会话还没结束（monitoring / needs_change），护士还在核对",
    "writeback_unsettled": "出站换液事件还没送达对端",
    "writeback_queue_pending": "待写回队列里还挂着这笔欠账",
    "no_event": "这条会话没有对应的出站事件（状态不一致，先留着人工看）",
    "bad_timestamp": "会话的 captured_at 解析不了，跳过并保留",
    "no_frames_dir": "会话没有记录帧目录（保留，不动）",
    "frames_shared_with_active_session": "同一段短片的另一只袋子还没到清理条件，整个目录一起留着",
}


@dataclass(frozen=True)
class RetentionPolicy:
    """保留期策略。

    默认值按病区实际节奏定：帧目录留 30 天（覆盖一次月度护理质控复盘 + 换袋争议
    追溯），推理审计留 90 天（要够覆盖一轮飞检准备期）。
    """

    session_retention_days: float = 30.0
    inference_audit_retention_days: float = 90.0
    archive_root: Path | None = None
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_retention_days": self.session_retention_days,
            "inference_audit_retention_days": self.inference_audit_retention_days,
            "archive_root": str(self.archive_root) if self.archive_root else None,
            "dry_run": self.dry_run,
        }


@dataclass
class RetentionResult:
    """一次清理的结果。**被留下的会话也在里面**（连同原因码）——只报删了多少，
    不报留下了多少，信息科就无法判断清理是不是配错了。"""

    policy: RetentionPolicy
    cutoff: str
    archived: list[dict[str, Any]] = field(default_factory=list)
    purged: list[dict[str, Any]] = field(default_factory=list)
    absent: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    kept_sessions: int = 0
    pruned_inference_audit: int = 0
    freed_bytes: int = 0
    frames_root: str = ""
    directories_touched: int = 0

    @property
    def touched_sessions(self) -> int:
        return len(self.archived) + len(self.purged) + len(self.absent)

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.as_dict(),
            "cutoff": self.cutoff,
            "frames_root": self.frames_root,
            "archived_sessions": self.archived,
            "purged_sessions": self.purged,
            "absent_sessions": self.absent,
            "blocked_sessions": self.blocked,
            "blocked_count": len(self.blocked),
            "kept_sessions": self.kept_sessions,
            "pruned_inference_audit": self.pruned_inference_audit,
            "freed_bytes": self.freed_bytes,
            "touched_sessions": self.touched_sessions,
            "directories_touched": self.directories_touched,
            "dry_run": self.policy.dry_run,
        }


# ── 时间解析 ──────────────────────────────────────────────────────────────
def _parse_captured_at(value: Any) -> datetime | None:
    """解析会话的 ``captured_at``（终端给的 ISO8601）。

    带时区偏移的按偏移换算成 UTC；**不带时区的按 UTC 处理**（与本产品落库的
    ``utc_now()`` 同一口径）。留 30 天的保留期，差 8 小时不影响结果。
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utcnow(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


# ── 目录体积 ──────────────────────────────────────────────────────────────
def _directory_bytes(path: Path) -> int:
    """目录里所有文件的字节和（不跟随符号链接，避免算进别处的盘）。"""
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_symlink():
                continue
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:  # pragma: no cover - 清理途中文件被别处删掉
            continue
    return total


# ── 事件与队列的欠账索引 ─────────────────────────────────────────────────
def _unsettled_index(conn: Any) -> dict[str, list[str]]:
    """按会话列出「出站还没了结」的状态码。

    一个会话可能有多条事件（重放会产生同幂等键的事件，但会话与袋序一一对应，
    正常只有一个）；这里取全部状态码，任一未了结即拦住。
    """
    rows = conn.execute(
        "SELECT session_id, writeback_status FROM infusion_event "
        "WHERE session_id IS NOT NULL AND writeback_status IN (?, ?, ?)",
        _UNSETTLED_EVENT_STATES,
    ).fetchall()
    index: dict[str, list[str]] = {}
    for row in rows:
        index.setdefault(row["session_id"], []).append(row["writeback_status"])
    return index


def _queue_pending_index(conn: Any) -> dict[str, list[str]]:
    """按会话列出「待写回队列里还挂着」的状态码。"""
    placeholders = ", ".join("?" for _ in _BLOCKING_QUEUE_STATES)
    rows = conn.execute(
        f"SELECT session_id, status FROM writeback_queue "
        f"WHERE session_id IS NOT NULL AND status IN ({placeholders})",
        _BLOCKING_QUEUE_STATES,
    ).fetchall()
    index: dict[str, list[str]] = {}
    for row in rows:
        index.setdefault(row["session_id"], []).append(row["status"])
    return index


def _event_counts(conn: Any) -> dict[str, int]:
    """每个会话的出站事件总数（用来判断「这条会话根本没有事件」）。"""
    rows = conn.execute(
        "SELECT session_id, COUNT(*) AS n FROM infusion_event "
        "WHERE session_id IS NOT NULL GROUP BY session_id"
    ).fetchall()
    return {row["session_id"]: int(row["n"]) for row in rows}


# ── 主流程 ────────────────────────────────────────────────────────────────
def plan_retention(
    conn: Any,
    *,
    policy: RetentionPolicy,
    now: datetime | None = None,
    measure_bytes: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], datetime]:
    """算出「哪些帧目录可以清、哪些必须留着」，**不动任何文件与数据**。

    返回（可清理的会话、留着的会话、截止时刻）。可清理的会话按目录分组——
    同一段短片的几只袋子共用一个帧目录，清理以目录为单位。

    ``measure_bytes=False`` 时不遍历目录（运行摘要轮询用，避免每次都扫一遍盘）。
    """
    reference = _utcnow(now)
    cutoff = reference - timedelta(days=policy.session_retention_days)
    cutoff_iso = _iso(cutoff)

    unsettled = _unsettled_index(conn)
    queue_pending = _queue_pending_index(conn)
    event_counts = _event_counts(conn)

    rows = conn.execute(
        "SELECT session_id, rack_id, bed_no, bag_seq, shift_key, captured_at, "
        "       frames_dir, frame_count, state "
        "FROM bag_session WHERE frames_retention_state = 'kept'"
    ).fetchall()

    # ── 逐会话判定 ────────────────────────────────────────────────────
    members: list[dict[str, Any]] = []
    for row in rows:
        session_id = row["session_id"]
        captured = _parse_captured_at(row["captured_at"])
        base = {
            "session_id": session_id,
            "rack_id": row["rack_id"],
            "bed_no": row["bed_no"],
            "bag_seq": row["bag_seq"],
            "shift_key": row["shift_key"],
            "captured_at": row["captured_at"],
            "frames_dir": row["frames_dir"] or "",
            "frame_count": int(row["frame_count"] or 0),
            "state": row["state"],
            "cutoff": cutoff_iso,
        }
        if captured is None:
            members.append({**base, "eligible": False, "reason": "bad_timestamp", "detail": []})
            continue
        if captured >= cutoff:
            members.append({**base, "eligible": False, "reason": "within_retention", "detail": []})
            continue
        if row["state"] != "replaced":
            members.append(
                {**base, "eligible": False, "reason": "session_not_finished", "detail": []}
            )
            continue
        statuses = unsettled.get(session_id)
        if statuses:
            members.append(
                {**base, "eligible": False, "reason": "writeback_unsettled", "detail": statuses}
            )
            continue
        queued = queue_pending.get(session_id)
        if queued:
            members.append(
                {**base, "eligible": False, "reason": "writeback_queue_pending", "detail": queued}
            )
            continue
        if not event_counts.get(session_id):
            members.append({**base, "eligible": False, "reason": "no_event", "detail": []})
            continue
        if not row["frames_dir"]:
            members.append({**base, "eligible": False, "reason": "no_frames_dir", "detail": []})
            continue
        members.append({**base, "eligible": True, "reason": "", "detail": []})

    # ── 按帧目录分组：目录里只要有一只袋子没到条件，整个目录都留着 ──────
    groups: dict[str, list[dict[str, Any]]] = {}
    for member in members:
        groups.setdefault(member["frames_dir"] or f"__no_dir__:{member['session_id']}", []).append(member)

    eligible: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for directory, group in groups.items():
        all_ready = all(item["eligible"] for item in group)
        for item in group:
            if item["eligible"] and all_ready:
                eligible.append(item)
            elif item["eligible"]:
                blocked.append(_blocked_entry(item, "frames_shared_with_active_session"))
            else:
                blocked.append(_blocked_entry(item, item["reason"], item["detail"]))

    # ── 目录级信息（字节数与归档目标路径按目录算一次，不按袋数重复算）──
    for item in eligible:
        source = Path(item["frames_dir"])
        item["exists"] = source.is_dir()
        item["bytes"] = (
            _directory_bytes(source) if (measure_bytes and source.is_dir()) else 0
        )
        item["archive_path"] = ""
        if policy.archive_root is not None and source.is_dir():
            month = (item["captured_at"] or "")[:7] or "unknown"
            item["archive_path"] = str(Path(policy.archive_root) / month / source.name)

    return eligible, blocked, cutoff


def retention_overview(
    conn: Any,
    *,
    policy: RetentionPolicy,
    now: datetime | None = None,
) -> dict[str, Any]:
    """保留期概览：**只读**，给运行摘要用。

    回答值班的人三个问题：盘上还剩多少帧、下一次清理会清掉哪几袋、为什么有的袋
    清不掉（按原因码分组计数，不给明细——明细在清理脚本的输出里）。

    概览**不遍历目录算字节**（摘要被轮询得很勤，扫盘太贵）；字节数由
    ``scripts/estimate_storage.py`` 在运维主动测算时给。
    """
    eligible, blocked, cutoff = plan_retention(conn, policy=policy, now=now, measure_bytes=False)

    by_state: dict[str, int] = {}
    for row in conn.execute(
        "SELECT frames_retention_state, COUNT(*) AS n FROM bag_session GROUP BY frames_retention_state"
    ).fetchall():
        by_state[row["frames_retention_state"]] = int(row["n"])

    blocked_by_reason: dict[str, int] = {}
    for item in blocked:
        blocked_by_reason[item["reason"]] = blocked_by_reason.get(item["reason"], 0) + 1

    archived = conn.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(bytes), 0) AS total FROM retention_archive"
    ).fetchone()

    return {
        "policy": policy.as_dict(),
        "cutoff": _iso(cutoff),
        "sessions_by_frames_state": by_state,
        "eligible_for_next_purge": len(eligible),
        "blocked_by_reason": blocked_by_reason,
        "archived_records": int(archived["n"]),
        "archived_bytes": int(archived["total"]),
    }


def _blocked_entry(
    item: Mapping[str, Any],
    reason: str,
    detail: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "session_id": item["session_id"],
        "rack_id": item.get("rack_id"),
        "bed_no": item.get("bed_no"),
        "bag_seq": item.get("bag_seq"),
        "captured_at": item.get("captured_at"),
        "state": item.get("state"),
        "frames_dir": item.get("frames_dir"),
        "reason": reason,
        "reason_text": BLOCK_REASONS.get(reason, reason),
        "detail": list(detail or []),
    }


def run_retention(
    conn: Any,
    frames_root: str | Path | None = None,
    *,
    policy: RetentionPolicy | None = None,
    audit: Any = None,
    request_id: str | None = None,
    now: datetime | None = None,
) -> RetentionResult:
    """按保留期清理：帧目录归档/删除 + 推理审计裁剪。

    ``audit`` 传 :class:`store.audit_chain.AuditChain` 时，每一次归档/删除都会在
    审计链留一行（谁在什么时候因为什么清了哪袋的帧）。
    """
    policy = policy or RetentionPolicy()
    eligible, blocked, cutoff = plan_retention(conn, policy=policy, now=now)

    result = RetentionResult(
        policy=policy,
        cutoff=_iso(cutoff),
        blocked=blocked,
        frames_root=str(frames_root) if frames_root else "",
    )
    result.kept_sessions = len(blocked)

    # 同一目录只处理一次（一只输液架几只袋子共用一个目录）
    by_dir: dict[str, list[dict[str, Any]]] = {}
    for item in eligible:
        by_dir.setdefault(item["frames_dir"], []).append(item)

    for directory, items in by_dir.items():
        outcome = _handle_directory(
            conn, items, policy=policy, audit=audit, request_id=request_id
        )
        result.directories_touched += 1
        result.freed_bytes += int(outcome.get("bytes") or 0)
        bucket = {
            "archived": result.archived,
            "deleted": result.purged,
            "absent": result.absent,
        }[outcome["action"]]
        for item in items:
            bucket.append({**item, "action": outcome["action"], "archive_path": outcome["archive_path"]})

    result.pruned_inference_audit = _prune_inference_audit(
        conn,
        retention_days=policy.inference_audit_retention_days,
        dry_run=policy.dry_run,
        audit=audit,
        request_id=request_id,
        now=now,
    )

    logger.info(
        "保留期清理完成 cutoff=%s archived=%d purged=%d absent=%d kept=%d freed_bytes=%d",
        result.cutoff,
        len(result.archived),
        len(result.purged),
        len(result.absent),
        result.kept_sessions,
        result.freed_bytes,
    )
    return result


def _handle_directory(
    conn: Any,
    items: list[dict[str, Any]],
    *,
    policy: RetentionPolicy,
    audit: Any,
    request_id: str | None,
) -> dict[str, Any]:
    """处理一个帧目录：归档 / 删除 / 记为不在。**同一目录的所有会话一起改。**

    **dry-run 只算不动**：不碰文件、不写库、不写审计——否则「先看看会清掉什么」
    这个动作本身就会把数据改了。
    """
    first = items[0]
    source = Path(first["frames_dir"])
    stamp = utc_now()

    if not source.is_dir():
        # 目录本来就不在（上一轮删了、或老库没留帧）：记一笔，别每次都重扫。
        if not policy.dry_run:
            with transaction(conn):
                for item in items:
                    _mark_session(conn, item["session_id"], "absent", stamp)
                    _record_archive(
                        conn,
                        item,
                        action="absent",
                        archive_path="",
                        size=0,
                        stamp=stamp,
                        policy_days=policy.session_retention_days,
                    )
        return {"action": "absent", "bytes": 0, "archive_path": ""}

    size = int(first.get("bytes") or 0)
    archive_path = ""
    action = "deleted"

    if not policy.dry_run:
        if policy.archive_root is not None:
            destination = Path(first["archive_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            archive_path = str(destination)
            action = "archived"
        else:
            shutil.rmtree(source, ignore_errors=True)

        with transaction(conn):
            for item in items:
                if action == "archived":
                    conn.execute(
                        "UPDATE bag_session SET frames_dir = ?, frames_retention_state = ?, "
                        "frames_purged_at = ?, updated_at = ? WHERE session_id = ?",
                        (archive_path, "archived", stamp, stamp, item["session_id"]),
                    )
                else:
                    conn.execute(
                        "UPDATE bag_session SET frames_retention_state = ?, frames_purged_at = ?, "
                        "updated_at = ? WHERE session_id = ?",
                        ("purged", stamp, stamp, item["session_id"]),
                    )
                _record_archive(
                    conn,
                    item,
                    action=action,
                    archive_path=archive_path,
                    size=size,
                    stamp=stamp,
                    policy_days=policy.session_retention_days,
                )

        if audit is not None and request_id:
            for item in items:
                audit.append(
                    request_id=request_id,
                    actor="service:retention",
                    action=(
                        "session.frames_archived" if action == "archived" else "session.frames_purged"
                    ),
                    target=item["session_id"],
                    bag_seq=item.get("bag_seq"),
                    rack_id=item.get("rack_id"),
                    bed_no=item.get("bed_no"),
                    outcome="accepted",
                    detail={
                        "reason": "retention_expired",
                        "frames_dir": item["frames_dir"],
                        "archive_path": archive_path,
                        "bytes": size,
                        "frame_count": item.get("frame_count"),
                        "policy_days": policy.session_retention_days,
                    },
                )

    return {"action": action, "bytes": size, "archive_path": archive_path}


def _mark_session(conn: Any, session_id: str, state: str, stamp: str) -> None:
    conn.execute(
        "UPDATE bag_session SET frames_retention_state = ?, frames_purged_at = ?, "
        "updated_at = ? WHERE session_id = ?",
        (state, stamp, stamp, session_id),
    )


def _record_archive(
    conn: Any,
    item: Mapping[str, Any],
    *,
    action: str,
    archive_path: str,
    size: int,
    stamp: str,
    policy_days: float,
) -> None:
    conn.execute(
        "INSERT INTO retention_archive (session_id, rack_id, bed_no, bag_seq, shift_key, "
        "captured_at, frames_dir, archive_path, frame_count, bytes, action, reason, "
        "policy_days, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            item["session_id"],
            item.get("rack_id"),
            item.get("bed_no"),
            item.get("bag_seq"),
            item.get("shift_key"),
            item.get("captured_at"),
            item.get("frames_dir"),
            archive_path or None,
            int(item.get("frame_count") or 0),
            int(size),
            action,
            "retention_expired",
            float(policy_days),
            stamp,
        ),
    )


def _prune_inference_audit(
    conn: Any,
    *,
    retention_days: float,
    dry_run: bool,
    audit: Any,
    request_id: str | None,
    now: datetime | None,
) -> int:
    """裁掉超期的推理审计元数据。

    **只清元数据，不碰审计链**：``audit_log`` 是飞检留档，任何保留期都不动它。
    ``inference_audit.created_at`` 一律由本服务的 ``utc_now()`` 写入（同一格式的
    UTC ISO 串），所以字符串比较与时间比较等价。
    """
    reference = _utcnow(now)
    cutoff = _iso(reference - timedelta(days=retention_days))
    rows = conn.execute(
        "SELECT audit_id FROM inference_audit WHERE created_at < ?", (cutoff,)
    ).fetchall()
    if not rows:
        return 0
    if dry_run:
        return len(rows)

    ids = [int(row["audit_id"]) for row in rows]
    with transaction(conn):
        conn.execute(
            "DELETE FROM inference_audit WHERE audit_id IN (%s)" % ",".join("?" for _ in ids),
            ids,
        )

    if audit is not None and request_id:
        audit.append(
            request_id=request_id,
            actor="service:retention",
            action="audit.inference_pruned",
            target=f"inference_audit:{cutoff}",
            outcome="accepted",
            detail={
                "reason": "retention_expired",
                "pruned_rows": len(ids),
                "retention_days": retention_days,
                "cutoff": cutoff,
            },
        )
    return len(ids)
