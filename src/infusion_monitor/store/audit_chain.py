"""append-only 审计链：谁、在什么时候、对哪一袋做了什么。

**为什么被拒绝的动作也要留一行**：护理质控与飞检查的不是「系统做对了什么」，
而是「系统是不是在不该动的时候动了」。一次 ``monitoring → replaced`` 的直跳被拒，
正是这类问题存在的证据；只记成功动作的日志回答不了「护士为什么没看到换液事件」。

三条纪律：

1. **只追加**：本模块不提供 update / delete，审计行一旦写入就不可变；
2. **与业务写入同事务**：``store.db.transaction`` 可重入嵌套，业务写成功而审计缺失
   会随外层事务一起回滚——不允许出现「状态变了但查不到谁改的」。需要把审计和业务写
   绑死在一个事务里时用 :meth:`AuditChain.insert`（自己在给定连接上写，不开事务），
   由调用方的 ``with transaction(conn)`` 统一收口，见
   ``repositories.SessionRepository.apply_state_transition``；
3. **不含可识别明细**：只落床号、袋序、动作与结果，不落患者姓名/住院号，
   ``detail`` 里也禁止调用方塞自由文本进来（见 :func:`append`）。

**导审计包这一层（task 33）回答的是护理部与内审的问法**：飞检不会问「
``writeback.retry`` 这个动作码是什么」，它问「这一架当天换了几次液、有没有
没写回去的、有没有人动过不该动的」。因此本模块把动作码归成 :data:`AUDIT_CATEGORIES`
七个**问句类别**（认领 / 标定确认 / 点名 / 护士确认 / 写回成功 / 写回失败 / 挂起），
导出时可按类别取；再给一条 :meth:`AuditChain.chain_digest` 滚动摘要，让「行有没有被
人删过」这件事当场可验——append-only 只是一句纪律，摘要是它的证据。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from .db import transaction
from .schema import AUDIT_ACTIONS, AUDIT_LOG_COLUMNS

#: 导出时逐列取值用的列名（顺序即 CSV 列序，见 ``store.audit_export``）。
AUDIT_EXPORT_COLUMNS: tuple[str, ...] = tuple(name for name, _ in AUDIT_LOG_COLUMNS)

__all__ = [
    "AUDIT_CATEGORIES",
    "AUDIT_EXPORT_COLUMNS",
    "AuditChain",
    "actions_in_categories",
    "category_label",
    "category_of",
    "detail_digest",
    "payload_digest",
]


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def detail_digest(detail: Mapping[str, Any]) -> str:
    """``detail`` 的稳定摘要（键排序后 sha256 前 16 位）。

    键排序是刻意的：同一份内容换个键序也要给出同一个摘要，否则「改没改过」
    就得靠肉眼比对 JSON 了。
    """
    blob = _dumps(dict(detail)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def payload_digest(payload: Mapping[str, Any]) -> str:
    """出站/入站载荷的摘要别名（与 :func:`detail_digest` 同口径）。"""
    return detail_digest(payload)


# ── 动作码 → 护理部/内审的问句类别 ─────────────────────────────────────────
#: 七类，覆盖 task 33 要求留痕的全部现场问法。**一个动作码只属于一类**：
#: 导审计包时按类别取行，若一个码同时落在两类里，同一行会在两张单子里各出现一次，
#: 内审对账时会以为发生了两件事。
#:
#: 未列在下面的码（分片续传、遥测通道、重锚等）**不属于这七类**，导出时按
#: ``category=其他`` 归入 ``other``，不静默丢弃——飞检问「这一段还有没有别的动作」
#: 时，答不上来比答「有」更糟。
AUDIT_CATEGORIES: dict[str, tuple[str, ...]] = {
    # 认领：护士在候选列表里指认「这一行是 B01」
    "claim": ("candidate.claimed", "candidate.corrected"),
    # 标定确认：袋型标定的登记、改版与人工复核结论（谁来定的、定的是哪一版）
    "calibration": (
        "calibration.version_created",
        "calibration.version_scheduled",
        "calibration.version_superseded",
        "calibration.review_required",
        "calibration.review_applied",
        "calibration.review_dismissed",
        "calibration.window_rejected",
        "calibration.deviation_failed",
    ),
    # 点名：系统算出 needs_change（判了、被闸门挡下、还是被每轮上限截断）
    "pickup": (
        "state.change_proposed",
        "pickup.named",
        "pickup.suppressed",
        "pickup.truncated",
    ),
    # 护士确认：人把 needs_change 推进到 replaced，并落出站事件
    "confirmation": (
        "state.confirmed",
        "state.transition_rejected",
        "session.write_rejected",
        "event.recorded",
        "event.replay_ignored",
    ),
    # 写回成功：对端 2xx，事件真的到了护理执行平台
    "writeback_sent": ("writeback.sent", "writeback.replayed"),
    # 写回失败：拒发（未确认）、对端 4xx 业务拒绝、可重试失败、人工重放仍未送达
    "writeback_failed": (
        "writeback.refused",
        "writeback.rejected",
        "writeback.retry",
        "writeback.replay_failed",
    ),
    # 挂起：退避重试耗尽，等信息科重放——这是护士长最需要看见的一类
    "writeback_pending": ("writeback.pending",),
}

#: 不属于上述七类、但仍要出现在审计包里的动作码（其余码归此类）。
OTHER_CATEGORY = "other"

_ACTION_TO_CATEGORY: dict[str, str] = {
    action: category for category, actions in AUDIT_CATEGORIES.items() for action in actions
}

_CATEGORY_LABELS: dict[str, str] = {
    "claim": "候选认领",
    "calibration": "标定确认",
    "pickup": "换液点名",
    "confirmation": "护士确认",
    "writeback_sent": "写回成功",
    "writeback_failed": "写回失败",
    "writeback_pending": "写回挂起",
    OTHER_CATEGORY: "其他动作",
}


def category_of(action: str) -> str:
    """动作码 → 类别键。未登记的码落 :data:`OTHER_CATEGORY`，不报错也不丢。"""
    return _ACTION_TO_CATEGORY.get(action, OTHER_CATEGORY)


def category_label(category: str) -> str:
    """类别键 → 中文标签（导出 CSV 的表头与 manifest 用）。"""
    return _CATEGORY_LABELS.get(category, category)


def actions_in_categories(categories: Sequence[str]) -> list[str]:
    """把类别键展开成动作码；``other`` 展开成「未归类的全部动作码」。

    这样「导全部动作」和「导这七类动作」能拼成同一套 SQL 参数，不必在调用方
    手工维护第二份动作码清单（那份清单迟早和 :data:`AUDIT_ACTIONS` 走偏）。
    """
    unknown = [c for c in categories if c not in AUDIT_CATEGORIES and c != OTHER_CATEGORY]
    if unknown:
        raise ValueError(f"未知审计类别：{unknown}（可用 {sorted(AUDIT_CATEGORIES) + [OTHER_CATEGORY]}）")
    wanted = {action for c in categories for action in AUDIT_CATEGORIES.get(c, ())}
    if OTHER_CATEGORY in categories:
        wanted.update(a for a in AUDIT_ACTIONS if a not in _ACTION_TO_CATEGORY)
    return sorted(wanted)


class AuditChain:
    """``audit_log`` 读写：只追加，不更新不删除。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def append(
        self,
        *,
        request_id: str,
        actor: str,
        action: str,
        target: str,
        outcome: str,
        bag_seq: str | None = None,
        bag_key: str | None = None,
        rack_id: str | None = None,
        bed_no: str | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
        detail: Mapping[str, Any] | None = None,
        occurred_at: str | None = None,
    ) -> int:
        """追加一行审计。``action`` 必须取自 :data:`store.schema.AUDIT_ACTIONS`。

        ``outcome`` 取 ``accepted`` / ``rejected``；被拒的行与被接受的行走同一张表，
        区别只在 outcome 与 detail 里的 reason。

        ``bag_key`` 是「输注单号|袋序」：换液点名引擎靠它查上一次点名时刻，
        落在独立列而不是 ``detail`` 里，是为了让「距上一次点名多久」能用索引查，
        而不是把整条审计链读进内存扫。

        ``rack_id`` / ``bed_no`` 冗余落列（task 33）：导审计包的两个主过滤维度。
        不传不影响写入，只会让该行在「按架 / 按床」过滤时走不到索引。
        """
        with transaction(self._conn) as conn:
            return self.insert(
                conn,
                request_id=request_id,
                actor=actor,
                action=action,
                target=target,
                outcome=outcome,
                bag_seq=bag_seq,
                bag_key=bag_key,
                rack_id=rack_id,
                bed_no=bed_no,
                from_state=from_state,
                to_state=to_state,
                detail=detail,
                occurred_at=occurred_at,
            )

    def insert(self, conn: sqlite3.Connection, **fields: Any) -> int:
        """在**给定连接**上写一行审计，**不开自己的事务**。

        与 :meth:`append` 的区别就是这一个：调用方若已经在 ``transaction(conn)``
        里，这里写进去的行与业务行同生共死；审计写失败（磁盘满、表被锁、字段
        超长）时业务行一起回滚，不会留下「状态变了但查不到谁改的」。

        校验在写之前做完，非法动作码 / 非法结果在**没有**任何行被写下的情况下
        就抛错——这条路径上抛异常是要触发回滚的，不能半写。
        """
        action = fields.get("action")
        if action not in AUDIT_ACTIONS:
            raise ValueError(f"未知审计动作：{action}（允许 {AUDIT_ACTIONS}）")
        outcome = fields.get("outcome")
        if outcome not in ("accepted", "rejected"):
            raise ValueError(f"非法审计结果：{outcome}")
        payload = dict(fields.get("detail") or {})
        # occurred_at 由调用方给（多数路径用 store.utc_now），缺省在下方补
        from .repositories import utc_now  # 局部导入避免循环

        row = {
            "request_id": fields.get("request_id"),
            "actor": fields.get("actor"),
            "action": action,
            "target": fields.get("target"),
            "bag_seq": fields.get("bag_seq"),
            "bag_key": fields.get("bag_key"),
            "rack_id": fields.get("rack_id"),
            "bed_no": fields.get("bed_no"),
            "from_state": fields.get("from_state"),
            "to_state": fields.get("to_state"),
            "outcome": outcome,
            "detail": _dumps(payload),
            "payload_hash": detail_digest(payload),
            "occurred_at": fields.get("occurred_at") or utc_now(),
        }
        columns = ", ".join(row)
        placeholders = ", ".join(f":{name}" for name in row)
        cursor = conn.execute(
            f"INSERT INTO audit_log ({columns}) VALUES ({placeholders})", row
        )
        return int(cursor.lastrowid or 0)

    def append_many(self, records: Iterable[Mapping[str, Any]]) -> int:
        """批量追加，返回写入行数（每行仍各自独立走事务）。"""
        return sum(1 for _ in (self.append(**dict(record)) for record in records))

    # ── 读 ────────────────────────────────────────────────────────────
    @staticmethod
    def _hydrate(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        raw = data.get("detail") or "{}"
        try:
            data["detail"] = json.loads(raw)
        except json.JSONDecodeError:
            data["detail"] = {"_raw": raw}
        return data

    def list_by_target(self, target: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE target = ? ORDER BY audit_id ASC", (target,)
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def list_by_request(self, request_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE request_id = ? ORDER BY audit_id ASC",
            (request_id,),
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def list_by_action(self, action: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE action = ? ORDER BY audit_id ASC LIMIT ?",
            (action, limit),
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def list_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM audit_log ORDER BY audit_id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0])

    # ── 导审计包：过滤查询 ──────────────────────────────────────────────
    def query(
        self,
        *,
        rack_id: str | None = None,
        bed_no: str | None = None,
        occurred_from: str | None = None,
        occurred_to: str | None = None,
        actions: Sequence[str] | None = None,
        outcomes: Sequence[str] | None = None,
        bag_seq: str | None = None,
        target: str | None = None,
        request_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """按架 / 床 / 时间窗 / 动作 / 结果过滤，返回**时间序单调**的审计行。

        排序是 ``occurred_at ASC, audit_id ASC``：``occurred_at`` 由调用方给，
        两行可能拿到同一个时刻（同一毫秒里的一次确认 + 一次出站），此时靠
        ``audit_id`` 定序，否则同一批导出两次会得到不同的行序——护理部拿着两版
        审计包对不上，就不知道该信哪一版。

        ``rack_id`` / ``bed_no`` 走 ``a.列`` 与 ``s.列`` 的 COALESCE：本表没有这两列的
        历史行（旧库或调用方未传）回查 ``bag_session`` 兜底，两类行都能按架/床导出，
        只是历史行走不了索引。
        """
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if rack_id:
            clauses.append("COALESCE(a.rack_id, s.rack_id) = :rack_id")
            params["rack_id"] = rack_id
        if bed_no:
            clauses.append("COALESCE(a.bed_no, s.bed_no) = :bed_no")
            params["bed_no"] = bed_no
        if occurred_from:
            clauses.append("a.occurred_at >= :occurred_from")
            params["occurred_from"] = occurred_from
        if occurred_to:
            clauses.append("a.occurred_at <= :occurred_to")
            params["occurred_to"] = occurred_to
        if actions:
            names = [name for name in actions]
            clauses.append("a.action IN (%s)" % ", ".join(f":a{i}" for i in range(len(names))))
            params.update({f"a{i}": name for i, name in enumerate(names)})
        if outcomes:
            marks = [name for name in outcomes]
            clauses.append("a.outcome IN (%s)" % ", ".join(f":o{i}" for i in range(len(marks))))
            params.update({f"o{i}": name for i, name in enumerate(marks)})
        if bag_seq:
            clauses.append("a.bag_seq = :bag_seq")
            params["bag_seq"] = bag_seq
        if target:
            clauses.append("a.target = :target")
            params["target"] = target
        if request_id:
            clauses.append("a.request_id = :request_id")
            params["request_id"] = request_id

        # SELECT 显式列出来而不是 ``a.*``：架号 / 床号要用 COALESCE 兜底同名，
        # 而 SQLite 的 ``dict(Row)`` 遇到重名列只保留靠后的那个——写成 ``a.*``
        # 加两个别名的话，能不能兜底就取决于列顺序这种没人会读的东西了。
        selected = [f"a.{name}" for name in AUDIT_EXPORT_COLUMNS if name not in ("rack_id", "bed_no")]
        selected.append("COALESCE(a.rack_id, s.rack_id) AS rack_id")
        selected.append("COALESCE(a.bed_no, s.bed_no) AS bed_no")
        selected.append("COALESCE(s.bag_type_code, '') AS bag_type_code")
        sql = (
            f"SELECT {', '.join(selected)} FROM audit_log a "
            "LEFT JOIN bag_session s ON s.session_id = a.target"
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY a.occurred_at ASC, a.audit_id ASC"
        if limit is not None and limit > 0:
            sql += " LIMIT :limit"
            params["limit"] = int(limit)
        rows = self._conn.execute(sql, params).fetchall()
        hydrated = [self._hydrate(row) for row in rows]
        for row in hydrated:
            row["category"] = category_of(str(row.get("action") or ""))
        return hydrated

    @staticmethod
    def chain_digest(rows: Sequence[Mapping[str, Any]]) -> str:
        """按行序滚动 sha256 的链摘要——**append-only 的证据**，不是口号。

        逐行把 ``(audit_id, action, target, outcome, payload_hash, occurred_at)``
        喂进摘要并把结果继续喂下一行。改动任何一行（改结果、换时刻）都会让摘要变；
        **删掉中间一行**同样会变——因为后续行的链式输入整体前移。这是「只追加」唯一
        能被第三方当场验证的形式：留档人重算一遍，对不上就说明这段审计被动过。

        空集合返回固定值 ``empty-chain`` 而不是空串：CSV 包里的空结果也该有个
        能填进「链摘要」栏的值，而不是让内审以为导出坏了。
        """
        running = ""
        for row in rows:
            piece = "|".join(
                str(row.get(key) or "")
                for key in ("audit_id", "action", "target", "outcome", "payload_hash", "occurred_at")
            )
            running = hashlib.sha256((running + "\x1f" + piece).encode("utf-8")).hexdigest()
        return running[:32] if running else "empty-chain"

    def count_by_action(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT action, COUNT(*) AS n FROM audit_log GROUP BY action"
        ).fetchall()
        return {row["action"]: int(row["n"]) for row in rows}

    def count_by_outcome(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT outcome, COUNT(*) AS n FROM audit_log GROUP BY outcome"
        ).fetchall()
        return {row["outcome"]: int(row["n"]) for row in rows}