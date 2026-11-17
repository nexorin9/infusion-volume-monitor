"""仓储层：袋级会话、出站事件、推理审计的读写方法。

写法约束（后续任务沿用）：

* 会话定位键是 ``rack_id + shift_key + bag_seq``：重复上传**覆盖**而非追加；
* 覆盖之前先把被盖掉的那一版快照进 ``bag_session_history``（task 32）；
* 出站事件以 ``client_event_id`` 为幂等键，重复确认返回既有行、不新增；
* 审计只追加，不更新不删除；
* 三张表的写入都在 ``transaction()`` 里，异常回滚，不静默改数据。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..calibration.cavity import (
    MAX_RESIDUE_EVIDENCE_RATIO,
    DeviationObservation,
    LearnedResidual,
    deviation_observation_from_session,
    learn_residual,
)
from ..calibration.models import BagTypeCalibration
from ..calibration.versions import (
    VersionWindow,
    assert_no_overlapping_window,
    plan_supersede,
)
from .db import transaction
from .schema import (
    CALIBRATION_STATES,
    BAG_STATES,
    WRITEBACK_QUEUE_STATES,
    WRITEBACK_STATES,
)

__all__ = [
    "SessionRepository",
    "EventRepository",
    "InferenceAuditRepository",
    "CalibrationRepository",
    "DeviationRepository",
    "UploadRepository",
    "utc_now",
    "new_id",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _merged_resize(existing: Any, incoming: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """合并两片的缩放结论：取值以第一片为准，``resized_frames`` 逐片累加。"""
    if incoming is None:
        return dict(existing) if isinstance(existing, Mapping) else None
    if not isinstance(existing, Mapping):
        return dict(incoming)
    merged = dict(existing)
    merged["resized_frames"] = int(existing.get("resized_frames") or 0) + int(
        incoming.get("resized_frames") or 0
    )
    return merged


def _loads(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


class SessionRepository:
    """``bag_session`` 读写。会话键：rack_id + shift_key + bag_seq。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        """底层连接。供 :mod:`identity.idempotency` 开跨仓储的同一事务用。"""
        return self._conn

    # ── 写 ────────────────────────────────────────────────────────────
    def upsert_session(
        self,
        payload: Mapping[str, Any],
        *,
        on_version_archived: Callable[[str, int, dict], None] | None = None,
    ) -> dict:
        """覆盖式写入一袋的会话；同一 (rack_id, shift_key, bag_seq) 只保留一行。

        重传整段序列时沿用同一 session_id 并把 ``session_version`` 加一，
        **被盖掉的那一版先快照进 ``bag_session_history``**（task 32）：护士长要回答
        「十分钟前那次报的是多少毫升」时，唯一能查的地方就是那张表。

        ``on_version_archived(session_id, version, snapshot)`` 在**同一事务内**被调用，
        供调用方把「谁在什么时刻重放了这一段」写进审计链；快照与覆盖同生共死，
        不会出现「版本盖掉了、审计没有」的中间态。

        **重传不重置业务状态**：入站只带「这条会话属于哪一袋」，不带状态。若已有行
        处于 ``needs_change`` 或 ``replaced``，重传保留原状态——护士刚被点名要换袋，
        下一轮又拍了一次就把点名抹掉，等于系统自己把自己的结论撤了。
        """
        now = utc_now()
        state = payload.get("state", "monitoring")
        if state not in BAG_STATES:
            raise ValueError(f"非法状态：{state}（允许 {BAG_STATES}）")
        calibration_state = payload.get("calibration_state", "uncalibrated")
        if calibration_state not in CALIBRATION_STATES:
            raise ValueError(f"非法标定状态：{calibration_state}")

        rack_id = payload["rack_id"]
        shift_key = payload["shift_key"]
        bag_seq = payload["bag_seq"]

        with transaction(self._conn) as conn:
            existing = conn.execute(
                "SELECT * FROM bag_session "
                "WHERE rack_id = ? AND shift_key = ? AND bag_seq = ?",
                (rack_id, shift_key, bag_seq),
            ).fetchone()
            if existing is None:
                session_id = payload.get("session_id") or new_id("sess")
                version = 1
                created_at = now
            else:
                session_id = existing["session_id"]
                version = int(existing["session_version"]) + 1
                created_at = existing["created_at"]
                if state == "monitoring" and existing["state"] != "monitoring":
                    state = existing["state"]
                # 覆盖前先把旧版本快照留档：整段重放要能回答「上一版是多少」。
                snapshot = self._archive_version(conn, existing, archived_at=now)
                if on_version_archived is not None:
                    on_version_archived(session_id, int(existing["session_version"]), snapshot)

            row = {
                "session_id": session_id,
                "rack_id": rack_id,
                "bed_no": payload["bed_no"],
                "infusion_order_no": payload["infusion_order_no"],
                "shift_key": shift_key,
                "bag_seq": bag_seq,
                "slot_index": int(payload.get("slot_index", 0)),
                "bag_type_code": payload.get("bag_type_code"),
                "frames_dir": payload.get("frames_dir", ""),
                "frame_count": int(payload.get("frame_count", 0)),
                "captured_at": payload["captured_at"],
                "area_ratio_series": _dumps(payload.get("area_ratio_series", [])),
                "area_ratio_raw": _dumps(payload.get("area_ratio_raw", [])),
                "occlusion_flags": _dumps(payload.get("occlusion_flags", [])),
                "frame_count_occluded": int(payload.get("frame_count_occluded", 0)),
                "fill_ratio": payload.get("fill_ratio"),
                "estimated_volume_ml": payload.get("estimated_volume_ml"),
                "eta_minutes": payload.get("eta_minutes"),
                "eta_source": payload.get("eta_source"),
                "calibration_state": calibration_state,
                "calibration_version": payload.get("calibration_version"),
                "decision_source": payload.get("decision_source"),
                "trigger_reason": payload.get("trigger_reason"),
                "state": state,
                "client_event_id": payload.get("client_event_id"),
                "session_version": version,
                "quality_flags": _dumps(payload.get("quality_flags", [])),
                # 帧几何（task 35）：相机原始尺寸 ↔ 规范帧尺寸 ↔ ROI 末态窗。
                # 缺列时留 NULL 而不是补 0/1：旧行（缩放策略上线前落的）没量过这两件事，
                # 补默认值会让「量过 1.0」与「没量过」在跨轮比对时长得一模一样。
                "camera_width": payload.get("camera_width"),
                "camera_height": payload.get("camera_height"),
                "canonical_width": payload.get("canonical_width"),
                "canonical_height": payload.get("canonical_height"),
                "frame_scale": payload.get("frame_scale"),
                "roi_window_json": _dumps(payload.get("roi_window")),
                "created_at": created_at,
                "updated_at": now,
            }
            columns = ", ".join(row)
            placeholders = ", ".join(f":{name}" for name in row)
            updates = ", ".join(
                f"{name} = excluded.{name}"
                for name in row
                if name not in ("session_id", "created_at")
            )
            conn.execute(
                f"INSERT INTO bag_session ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(rack_id, shift_key, bag_seq) DO UPDATE SET {updates}",
                row,
            )
        return self.get(session_id) or {}

    def update_state(self, session_id: str, state: str, **fields: Any) -> dict:
        """推进会话状态并附带 ``trigger_reason`` 等字段（调用方负责校验合法性）。"""
        if state not in BAG_STATES:
            raise ValueError(f"非法状态：{state}")
        allowed = {
            "trigger_reason",
            "decision_source",
            "fill_ratio",
            "estimated_volume_ml",
            "eta_minutes",
            "eta_source",
            "calibration_state",
            "calibration_version",
            "client_event_id",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"不可写字段：{sorted(unknown)}")
        assignments = ["state = :state", "updated_at = :updated_at"]
        params: dict[str, Any] = {"session_id": session_id, "state": state, "updated_at": utc_now()}
        for key, value in fields.items():
            assignments.append(f"{key} = :{key}")
            params[key] = value
        with transaction(self._conn) as conn:
            conn.execute(
                f"UPDATE bag_session SET {', '.join(assignments)} WHERE session_id = :session_id",
                params,
            )
        return self.get(session_id) or {}

    def update_area_series(self, session_id: str, series: Sequence[float], *, raw: Sequence[float] | None = None,
                           occlusion_flags: Sequence[Any] | None = None,
                           frame_count_occluded: int | None = None) -> None:
        """回写面积比序列（平滑值 + 原始值都留，供核对）。"""
        assignments = ["area_ratio_series = :series", "updated_at = :updated_at"]
        params: dict[str, Any] = {
            "session_id": session_id,
            "series": _dumps(list(series)),
            "updated_at": utc_now(),
        }
        if raw is not None:
            assignments.append("area_ratio_raw = :raw")
            params["raw"] = _dumps(list(raw))
        if occlusion_flags is not None:
            assignments.append("occlusion_flags = :flags")
            params["flags"] = _dumps(list(occlusion_flags))
        if frame_count_occluded is not None:
            assignments.append("frame_count_occluded = :occluded")
            params["occluded"] = int(frame_count_occluded)
        with transaction(self._conn) as conn:
            conn.execute(
                f"UPDATE bag_session SET {', '.join(assignments)} WHERE session_id = :session_id",
                params,
            )

    # ── 跨巡视轮次：起锚与重锚（task 9） ────────────────────────────────
    def save_anchor(
        self,
        session_id: str,
        *,
        candidate_id: str | None = None,
        bbox: Sequence[int] | None = None,
        descriptor: Mapping[str, Any] | None = None,
        candidates: Sequence[Mapping[str, Any]] | None = None,
        claim_state: str | None = None,
        claimed_by: str | None = None,
    ) -> dict:
        """落「这一袋据以起锚的候选 + 它的外观描述子」。

        描述子是**下一轮认出「还是那一袋」的唯一依据**，必须落库：只存在内存里的
        描述子在下一轮巡视时已经没了，重锚就退化成只看挂点位置。

        ``descriptor={}`` 表示**清空**描述子（认领纠错后原区域已看不见液体时用），
        与 ``None``（不改这一列）区分开。
        """
        assignments: list[str] = []
        params: dict[str, Any] = {"session_id": session_id, "updated_at": utc_now()}
        if candidate_id is not None:
            assignments.append("anchor_candidate_id = :anchor_candidate_id")
            params["anchor_candidate_id"] = candidate_id
        if bbox is not None:
            assignments.append("anchor_bbox = :anchor_bbox")
            params["anchor_bbox"] = _dumps([int(v) for v in bbox])
        if descriptor is not None:
            assignments.append("descriptor_json = :descriptor_json")
            params["descriptor_json"] = _dumps(dict(descriptor))
        if candidates is not None:
            assignments.append("candidates_json = :candidates_json")
            params["candidates_json"] = _dumps([dict(item) for item in candidates])
        if claim_state is not None:
            assignments.append("claim_state = :claim_state")
            params["claim_state"] = claim_state
        if claimed_by is not None:
            assignments.append("claimed_by = :claimed_by")
            params["claimed_by"] = claimed_by
        if not assignments:
            return self.get(session_id) or {}
        assignments.append("updated_at = :updated_at")
        with transaction(self._conn) as conn:
            conn.execute(
                f"UPDATE bag_session SET {', '.join(assignments)} WHERE session_id = :session_id",
                params,
            )
        return self.get(session_id) or {}

    def save_reanchor(self, session_id: str, verdict: Mapping[str, Any]) -> dict:
        """落跨轮次重锚的判定结果（含**拒绝猜测**的原因码）。"""
        allowed = {
            "reanchor_state",
            "reanchor_confidence",
            "reanchor_runner_up",
            "reanchor_reason",
            "previous_session_id",
            "previous_shift_key",
        }
        unknown = set(verdict) - allowed
        if unknown:
            raise ValueError(f"不可写字段：{sorted(unknown)}")
        if not verdict:
            return self.get(session_id) or {}
        assignments = [f"{name} = :{name}" for name in verdict]
        assignments.append("updated_at = :updated_at")
        params: dict[str, Any] = dict(verdict)
        params["session_id"] = session_id
        params["updated_at"] = utc_now()
        with transaction(self._conn) as conn:
            conn.execute(
                f"UPDATE bag_session SET {', '.join(assignments)} WHERE session_id = :session_id",
                params,
            )
        return self.get(session_id) or {}

    def apply_state_transition(
        self,
        session_id: str,
        target: str,
        *,
        audit: Any | None = None,
        audit_record: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> dict:
        """在**同一事务**里推进状态并写审计。

        状态合法性由调用方（``identity/state_machine``）先判；这里只负责落库，
        保证「状态变了但审计没写」不可能发生——两者共用一个 ``transaction()``。

        ``audit_record`` 给了就必须在这一次事务里落 ``audit_log``（走
        ``audit.insert``，不开自己的事务）：审计写失败（磁盘满、库被锁、行超长）
        会把上面那条 UPDATE 一起回滚。护理质控问「这袋为什么被推成 needs_change」时，
        查得到一条 ``state.change_proposed`` 与查不到，是两种完全不同的话——所以宁可
        让这次点名整体失败重做，也不留一个查无实据的状态。

        只给 ``audit`` 不给 ``audit_record`` 时审计不会被写：显式传记录，避免调用方
        以为自己挂了审计链其实什么都没记（``audit`` 在这里是给 ``insert`` 用的句柄）。
        """
        if target not in BAG_STATES:
            raise ValueError(f"非法状态：{target}")
        allowed = {"trigger_reason", "decision_source", "client_event_id"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"不可写字段：{sorted(unknown)}")
        if audit_record is not None and audit is None:
            raise ValueError("给了 audit_record 就必须给 audit 句柄，否则这行审计写不出去")
        assignments = ["state = :state", "updated_at = :updated_at"]
        params: dict[str, Any] = {"session_id": session_id, "state": target, "updated_at": utc_now()}
        for key, value in fields.items():
            assignments.append(f"{key} = :{key}")
            params[key] = value
        with transaction(self._conn) as conn:
            conn.execute(
                f"UPDATE bag_session SET {', '.join(assignments)} WHERE session_id = :session_id",
                params,
            )
            if audit_record is not None and audit is not None:
                audit.insert(conn, **dict(audit_record))
        return self.get(session_id) or {}

    # ── 读 ────────────────────────────────────────────────────────────
    @staticmethod
    def _hydrate_version(row: sqlite3.Row | None) -> dict | None:
        """把 ``bag_session_history`` 行还原成与 ``bag_session`` 同形的可读 dict。"""
        if row is None:
            return None
        data = dict(row)
        for key in ("area_ratio_series", "area_ratio_raw", "occlusion_flags"):
            data[key] = _loads(data.get(key), [])
        return data

    @staticmethod
    def _archive_version(
        conn: sqlite3.Connection, row: sqlite3.Row, *, archived_at: str
    ) -> dict:
        """把被覆盖的旧版本快照进 ``bag_session_history``（append-only）。

        同一 ``(session_id, session_version)`` 只留一份：连打三次重放时，第 2 版被盖
        两次也只能有一条快照，否则「旧版本」自己也会重影，护士长查到两条同样的数字
        反而不知道该信哪条。
        """
        conn.execute(
            "INSERT OR IGNORE INTO bag_session_history ("
            "  session_id, session_version, rack_id, shift_key, bag_seq, state,"
            "  trigger_reason, frames_dir, frame_count, captured_at, area_ratio_series,"
            "  area_ratio_raw, occlusion_flags, fill_ratio, estimated_volume_ml,"
            "  eta_minutes, calibration_state, calibration_version, client_event_id,"
            "  archived_reason, archived_at"
            ") VALUES ("
            "  :session_id, :session_version, :rack_id, :shift_key, :bag_seq, :state,"
            "  :trigger_reason, :frames_dir, :frame_count, :captured_at, :area_ratio_series,"
            "  :area_ratio_raw, :occlusion_flags, :fill_ratio, :estimated_volume_ml,"
            "  :eta_minutes, :calibration_state, :calibration_version, :client_event_id,"
            "  'replay', :archived_at"
            ")",
            {
                "session_id": row["session_id"],
                "session_version": int(row["session_version"]),
                "rack_id": row["rack_id"],
                "shift_key": row["shift_key"],
                "bag_seq": row["bag_seq"],
                "state": row["state"],
                "trigger_reason": row["trigger_reason"],
                "frames_dir": row["frames_dir"],
                "frame_count": int(row["frame_count"]),
                "captured_at": row["captured_at"],
                "area_ratio_series": row["area_ratio_series"],
                "area_ratio_raw": row["area_ratio_raw"],
                "occlusion_flags": row["occlusion_flags"],
                "fill_ratio": row["fill_ratio"],
                "estimated_volume_ml": row["estimated_volume_ml"],
                "eta_minutes": row["eta_minutes"],
                "calibration_state": row["calibration_state"],
                "calibration_version": row["calibration_version"],
                "client_event_id": row["client_event_id"],
                "archived_at": archived_at,
            },
        )
        snapshot = conn.execute(
            "SELECT * FROM bag_session_history WHERE session_id = ? AND session_version = ?",
            (row["session_id"], int(row["session_version"])),
        ).fetchone()
        return SessionRepository._hydrate_version(snapshot) or {}

    def list_versions(self, session_id: str) -> list[dict]:
        """按版本号升序列出这条会话被覆盖掉的全部旧版本（append-only 快照）。"""
        rows = self._conn.execute(
            "SELECT * FROM bag_session_history WHERE session_id = ? "
            "ORDER BY session_version ASC",
            (session_id,),
        ).fetchall()
        return [self._hydrate_version(row) for row in rows]

    def count_versions(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM bag_session_history WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["n"])

    @staticmethod
    def _hydrate(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        data = dict(row)
        for key, fallback in (
            ("area_ratio_series", []),
            ("area_ratio_raw", []),
            ("occlusion_flags", []),
            ("quality_flags", []),
            ("candidates_json", []),
        ):
            data[key] = _loads(data.get(key), fallback)
        data["descriptor"] = _loads(data.get("descriptor_json"), {})
        data["anchor_bbox"] = _loads(data.get("anchor_bbox"), None)
        if data.get("pickup_suppressed") is not None:
            data["pickup_suppressed"] = bool(data["pickup_suppressed"])
        return data

    def get(self, session_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM bag_session WHERE session_id = ?", (session_id,)
        ).fetchone()
        return self._hydrate(row)

    def find_by_rack_shift_bag(self, rack_id: str, shift_key: str, bag_seq: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM bag_session WHERE rack_id = ? AND shift_key = ? AND bag_seq = ?",
            (rack_id, shift_key, bag_seq),
        ).fetchone()
        return self._hydrate(row)

    def list_by_order(self, infusion_order_no: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM bag_session WHERE infusion_order_no = ? ORDER BY bag_seq ASC",
            (infusion_order_no,),
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def list_previous_rounds(
        self, rack_id: str, infusion_order_no: str, *, exclude_shift: str
    ) -> list[dict]:
        """取同一输液架、同一输注单里，每个袋序**最近一次**的历史会话（重锚的候选集合）。

        两个关键约束：

        * ``exclude_shift`` 排除本轮轮次键——同一轮重传是覆盖，不是新的一轮，
          自己跟自己重锚必然满分对齐，会把「认不准」的情形测不出来；
        * **每个袋序只留最近一条**。同一袋在上午、下午两轮各有一条会话，若把两轮都
          当候选，那它们长得几乎一样，分差必然过小 → 每轮都判 ``ambiguous_slot``，
          重锚就永远接不上。重锚要回答的是「这个袋序现在是什么状态」，不是「历史上出现过几次」；
        * 「最近」按 ``captured_at``（护士实际拍摄时刻，调用方给的 ISO8601）排，而不是
          服务端写入时刻：``utc_now`` 只到秒，几轮在同一秒内落库时按写入时刻排会随机
          挑到任意一轮。拍摄时刻才是「哪一轮更新」的业务事实。
        """
        rows = self._conn.execute(
            "SELECT * FROM bag_session WHERE session_id IN ("
            "  SELECT session_id FROM ("
            "    SELECT session_id, ROW_NUMBER() OVER ("
            "      PARTITION BY bag_seq ORDER BY captured_at DESC, updated_at DESC"
            "    ) AS rn FROM bag_session"
            "    WHERE rack_id = ? AND infusion_order_no = ? AND shift_key != ?"
            "  ) WHERE rn = 1"
            ") ORDER BY bag_seq ASC",
            (rack_id, infusion_order_no, exclude_shift),
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def list_by_state(self, state: str, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM bag_session WHERE state = ? ORDER BY updated_at DESC LIMIT ?",
            (state, limit),
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def count_by_state(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT state, COUNT(*) AS n FROM bag_session GROUP BY state"
        ).fetchall()
        return {row["state"]: int(row["n"]) for row in rows}

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM bag_session").fetchone()[0])

    def list_by_bag_type(self, bag_type_code: str) -> list[dict]:
        """按袋型取历史会话（标定改版后重算的对象集合）。"""
        rows = self._conn.execute(
            "SELECT * FROM bag_session WHERE bag_type_code = ? ORDER BY created_at ASC",
            (bag_type_code,),
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def update_calibration(self, session_id: str, payload: Mapping[str, Any]) -> dict:
        """回写标定与体积结果（不推进业务状态）。

        ``estimated_volume_ml``/``eta_minutes`` 为 ``None`` 时**写成 NULL**——这是
        「门禁未通过」的落库形态：库里明确没有数，响应序列化时字段整体不出现。
        """
        allowed = {
            "fill_ratio",
            "estimated_volume_ml",
            "eta_minutes",
            "eta_source",
            "rate_ml_per_h",
            "rate_source",
            "rate_reason",
            "rate_conflict_json",
            "calibration_state",
            "calibration_version",
            "pickup_suppressed",
            "suppression_reason",
            "calibration_review_required",
            "calibration_review_deviation",
            "calibration_review_version",
            "calibration_review_proposed_ml",
            "volume_confidence",
            "volume_low_ml",
            "volume_high_ml",
            "uncertainty_gate",
            "uncertainty_json",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"不可写字段：{sorted(unknown)}")
        if not payload:
            return self.get(session_id) or {}
        assignments = ["updated_at = :updated_at"]
        params: dict[str, Any] = {"session_id": session_id, "updated_at": utc_now()}
        for key, value in payload.items():
            assignments.append(f"{key} = :{key}")
            if key == "pickup_suppressed" and value is not None:
                params[key] = int(value)
            elif key == "uncertainty_json" and value is not None and not isinstance(value, str):
                # 传 dict 与传 JSON 串都收：调用方（主路径 / 改版重算）各写各的容易漏。
                params[key] = _dumps(value)
            else:
                params[key] = value
        with transaction(self._conn) as conn:
            conn.execute(
                f"UPDATE bag_session SET {', '.join(assignments)} WHERE session_id = :session_id",
                params,
            )
        return self.get(session_id) or {}

    #: 拍摄质量可回写字段（白名单，与 :meth:`update_calibration` 同一套规矩）。
    QUALITY_FIELDS = ("quality_flags", "capture_degraded")

    def update_quality(self, session_id: str, quality_flags: Sequence[Any], degraded: bool) -> dict:
        """回写拍摄质量结论（不推进业务状态，也不改体积读数）。

        与体积读数分开写：拍坏了照样给曲线和毫升数，只是护士端把补拍建议顶到提示条上——
        质检结果**不能**反过来改掉已经算出来的读数，否则「拍不好」就变成了「查不出数」，
        那是两回事（后者有独立的门禁，见 :mod:`..calibration.estimator`）。
        """
        params: dict[str, Any] = {
            "session_id": session_id,
            "updated_at": utc_now(),
            "quality_flags": _dumps(list(quality_flags)),
            "capture_degraded": 1 if degraded else 0,
        }
        with transaction(self._conn) as conn:
            conn.execute(
                "UPDATE bag_session SET updated_at = :updated_at, "
                "quality_flags = :quality_flags, capture_degraded = :capture_degraded "
                "WHERE session_id = :session_id",
                params,
            )
        return self.get(session_id) or {}

    def update_frame_geometry(
        self,
        session_id: str,
        *,
        camera_width: int,
        camera_height: int,
        canonical_width: int,
        canonical_height: int,
        scale: float,
        roi_window: Mapping[str, Any] | None = None,
    ) -> dict:
        """回写这一袋所属那段短片的帧几何（task 35）。

        相机原始尺寸、规范帧尺寸、缩放比与 ROI 末态窗。护士端把候选框画回原始照片时
        乘 ``1/scale``；信息科问「昨天和今天的读数怎么一个说 1280 一个说 640」时，
        答案在这几列里，而不在某段日志里。
        """
        params: dict[str, Any] = {
            "session_id": session_id,
            "updated_at": utc_now(),
            "camera_width": int(camera_width),
            "camera_height": int(camera_height),
            "canonical_width": int(canonical_width),
            "canonical_height": int(canonical_height),
            "scale": float(scale),
            "roi": _dumps(dict(roi_window) if roi_window else None),
        }
        with transaction(self._conn) as conn:
            conn.execute(
                "UPDATE bag_session SET updated_at = :updated_at, "
                "camera_width = :camera_width, camera_height = :camera_height, "
                "canonical_width = :canonical_width, canonical_height = :canonical_height, "
                "frame_scale = :scale, roi_window_json = :roi "
                "WHERE session_id = :session_id",
                params,
            )
        return self.get(session_id) or {}

    #: 上传前预检可回写字段（白名单，与 :meth:`update_quality` 同一套规矩）。
    PRECHECK_FIELDS = ("precheck_score", "precheck_flags", "precheck_degraded")

    def update_precheck(
        self,
        session_id: str,
        *,
        score: float | None,
        precheck_flags: Sequence[Any],
        degraded: bool,
    ) -> dict:
        """回写**上传前预检**结论（整段一段，逐袋各存一份）。

        与袋级质量结论分开写：``quality_flags`` 答「哪一只袋子没拍好」，
        ``precheck_*`` 答「这一整段值不值得信」。护士复盘时这两个问题都会问
        （「B02 那次为什么建议我重拍」对「这一轮整体拍得怎么样」），分开存才答得清。
        """
        params: dict[str, Any] = {
            "session_id": session_id,
            "updated_at": utc_now(),
            "precheck_score": None if score is None else float(score),
            "precheck_flags": _dumps(list(precheck_flags)),
            "precheck_degraded": 1 if degraded else 0,
        }
        with transaction(self._conn) as conn:
            conn.execute(
                "UPDATE bag_session SET updated_at = :updated_at, "
                "precheck_score = :precheck_score, precheck_flags = :precheck_flags, "
                "precheck_degraded = :precheck_degraded "
                "WHERE session_id = :session_id",
                params,
            )
        return self.get(session_id) or {}


class UploadRepository:
    """``upload_chunk`` 读写：分片续传的半成品台账（task 32）。

    这一行回答的是「刚才那段拍到一半断网了，接着从第几帧传」——所以已收帧号是
    **逐帧**清单而不是「收到 3 个分片」：病区 Wi-Fi 是按帧丢的，只知道分片数
    没法告诉终端该补哪一段。

    重复帧的处理写在这一层而不是调用方：**先到先得**，后到的同号帧直接丢弃并计数，
    不覆盖已收的那一份（覆盖会让「这一帧到底是哪一次拍的」说不清）。
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    # ── 写 ────────────────────────────────────────────────────────────
    def open_upload(self, payload: Mapping[str, Any]) -> dict:
        """登记一次可续传的上传（还没收到任何帧）。

        ``expires_at`` 由调用方按配置的保留期算好传进来：保留期是**运维配置**，
        仓储层不该知道「病区一般留多久」。
        """
        now = utc_now()
        upload_id = payload.get("upload_id") or new_id("upl")
        with transaction(self._conn) as conn:
            conn.execute(
                "INSERT INTO upload_chunk ("
                "  upload_id, rack_id, bed_no, infusion_order_no, shift_key, captured_at,"
                "  slots_json, client_event_id, session_key, staging_dir, status,"
                "  expected_frames, received_frames, frame_times_json, duplicate_count,"
                "  chunk_count, total_bytes, session_ids_json, created_at, updated_at,"
                "  expires_at, completed_at"
                ") VALUES ("
                "  :upload_id, :rack_id, :bed_no, :infusion_order_no, :shift_key, :captured_at,"
                "  :slots_json, :client_event_id, :session_key, :staging_dir, 'pending',"
                "  :expected_frames, '[]', '{}', 0, 0, 0, '[]', :created_at, :updated_at,"
                "  :expires_at, NULL"
                ")",
                {
                    "upload_id": upload_id,
                    "rack_id": payload["rack_id"],
                    "bed_no": payload["bed_no"],
                    "infusion_order_no": payload["infusion_order_no"],
                    "shift_key": payload["shift_key"],
                    "captured_at": payload["captured_at"],
                    "slots_json": payload["slots_json"],
                    "client_event_id": payload.get("client_event_id"),
                    "session_key": payload["session_key"],
                    "staging_dir": payload["staging_dir"],
                    "expected_frames": payload.get("expected_frames"),
                    "created_at": now,
                    "updated_at": now,
                    "expires_at": payload["expires_at"],
                },
            )
        return self.get(upload_id) or {}

    def record_chunk(
        self,
        upload_id: str,
        *,
        accepted_frames: Sequence[int],
        frame_times: Mapping[int, str],
        duplicate_frames: Sequence[int],
        added_bytes: int,
        expires_at: str,
        resize: Mapping[str, Any] | None = None,
    ) -> dict:
        """登记一个分片的结果：新增帧号、逐帧时间、去重丢弃的帧号。

        **在一个事务里重算**：读—判—写三段都在 ``BEGIN IMMEDIATE`` 内，两路终端
        同时补同一个分片时后到的那一路会看到前一路刚写下的帧号并照单去重，
        不会双双把同一帧号记成「已收」而实际只落了一份。

        ``resize`` 是本片的缩放结论（task 35）：**取值（相机尺寸/规范尺寸/缩放比/理由）
        以第一片为准**——相机原始尺寸是整段短片的属性，不该由最后一片说了算；
        ``resized_frames`` 则**逐片累加**，提交时报的是「整段里有几帧被重写过」，
        与整段上传那条路同一个口径。
        """
        now = utc_now()
        with transaction(self._conn) as conn:
            row = conn.execute(
                "SELECT received_frames, frame_times_json, duplicate_count, chunk_count, "
                "resize_json FROM upload_chunk WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
            if row is None:
                raise KeyError(upload_id)
            received = sorted(set(_loads(row["received_frames"], [])) | set(accepted_frames))
            times: dict[str, str] = dict(_loads(row["frame_times_json"], {}))
            times.update({str(index): value for index, value in frame_times.items()})
            merged_resize = _merged_resize(_loads(row["resize_json"], None), resize)
            conn.execute(
                "UPDATE upload_chunk SET received_frames = :frames, frame_times_json = :times,"
                " duplicate_count = duplicate_count + :duplicates, chunk_count = chunk_count + 1,"
                " total_bytes = total_bytes + :bytes, updated_at = :updated_at,"
                " expires_at = :expires_at,"
                " resize_json = COALESCE(:resize, resize_json)"
                " WHERE upload_id = :upload_id",
                {
                    "frames": _dumps(received),
                    "times": _dumps(times),
                    "duplicates": len(duplicate_frames),
                    "bytes": int(added_bytes),
                    "updated_at": now,
                    "expires_at": expires_at,
                    "resize": _dumps(merged_resize) if merged_resize else None,
                    "upload_id": upload_id,
                },
            )
        return self.get(upload_id) or {}

    def mark_completed(self, upload_id: str, session_ids: Sequence[str]) -> dict:
        """标记已提交：半成品转成正式袋级会话，不再受清理策略扫。"""
        now = utc_now()
        with transaction(self._conn) as conn:
            conn.execute(
                "UPDATE upload_chunk SET status = 'completed', session_ids_json = :sessions,"
                " completed_at = :completed_at, updated_at = :updated_at WHERE upload_id = :upload_id",
                {
                    "sessions": _dumps([str(item) for item in session_ids]),
                    "completed_at": now,
                    "updated_at": now,
                    "upload_id": upload_id,
                },
            )
        return self.get(upload_id) or {}

    def mark_abandoned(self, upload_ids: Sequence[str]) -> int:
        """保留期到点：把半成品结案为 ``abandoned``（不删行，留「谁传了一半」的痕迹）。"""
        now = utc_now()
        ids = list(upload_ids)
        if not ids:
            return 0
        with transaction(self._conn) as conn:
            conn.executemany(
                "UPDATE upload_chunk SET status = 'abandoned', updated_at = ? "
                "WHERE upload_id = ? AND status = 'pending'",
                [(now, upload_id) for upload_id in ids],
            )
        return len(ids)

    # ── 读 ────────────────────────────────────────────────────────────
    @staticmethod
    def _hydrate(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        data = dict(row)
        data["received_frames"] = _loads(data.get("received_frames"), [])
        data["frame_times"] = _loads(data.get("frame_times_json"), {})
        data["session_ids"] = _loads(data.get("session_ids_json"), [])
        data["slots"] = _loads(data.get("slots_json"), [])
        data.pop("frame_times_json", None)
        data.pop("slots_json", None)
        data.pop("session_ids_json", None)
        return data

    def get(self, upload_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM upload_chunk WHERE upload_id = ?", (upload_id,)
        ).fetchone()
        return self._hydrate(row)

    def list_by_rack(self, rack_id: str, *, shift_key: str | None = None) -> list[dict]:
        if shift_key is None:
            rows = self._conn.execute(
                "SELECT * FROM upload_chunk WHERE rack_id = ? ORDER BY created_at DESC",
                (rack_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM upload_chunk WHERE rack_id = ? AND shift_key = ? "
                "ORDER BY created_at DESC",
                (rack_id, shift_key),
            ).fetchall()
        return [self._hydrate(row) for row in rows]

    def list_expired(self, *, before: str, status: str = "pending") -> list[dict]:
        """保留期已到、仍是该状态的半成品（清理策略的输入）。"""
        rows = self._conn.execute(
            "SELECT * FROM upload_chunk WHERE status = ? AND expires_at <= ? "
            "ORDER BY expires_at ASC",
            (status, before),
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def count_pending(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM upload_chunk WHERE status = 'pending'"
        ).fetchone()
        return int(row["n"])

    def count_by_status(self, status: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM upload_chunk WHERE status = ?", (status,)
        ).fetchone()
        return int(row["n"])

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM upload_chunk").fetchone()
        return int(row["n"])


class EventRepository:
    """``infusion_event`` 读写。``client_event_id`` 为幂等键。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def append_event(self, payload: Mapping[str, Any]) -> tuple[dict, bool]:
        """追加出站事件，返回 (事件行, 是否新增)。

        ``client_event_id`` 已存在时返回既有行且 ``created=False`` —— 重复确认
        不产生第二条事件（task 10 的幂等语义）。

        **查重在事务内做**：``BEGIN IMMEDIATE`` 先拿到写锁，再查 ``client_event_id``
        再决定是否 INSERT。把这次查重放在事务外会留下一个窗口——两路并发都在事务外
        查到「不存在」，然后各自 INSERT，第二个撞 ``UNIQUE`` 报错。一条幂等路径不该
        靠异常兜底，那说明它本来就没打算重放。撞上了仍然是幂等命中（读回既有行），
        只是那是最后一道保险而不是主要路径。
        """
        now = utc_now()
        client_event_id = payload["client_event_id"]
        status = payload.get("writeback_status", "pending")
        if status not in WRITEBACK_STATES:
            raise ValueError(f"非法写回状态：{status}")
        row = {
            "event_id": payload.get("event_id") or new_id("evt"),
            "client_event_id": client_event_id,
            "session_id": payload.get("session_id"),
            "event_type": payload.get("event_type", "infusion.bag.changed"),
            "rack_id": payload["rack_id"],
            "bed_no": payload["bed_no"],
            "infusion_order_no": payload["infusion_order_no"],
            "bag_seq": payload["bag_seq"],
            "fill_ratio": payload.get("fill_ratio"),
            "estimated_volume_ml": payload.get("estimated_volume_ml"),
            "eta_minutes": payload.get("eta_minutes"),
            "trigger_reason": payload.get("trigger_reason"),
            "decision_source": payload.get("decision_source"),
            "operator_id": payload["operator_id"],
            "occurred_at": payload["occurred_at"],
            "writeback_status": status,
            "attempt_count": int(payload.get("attempt_count", 0)),
            "next_retry_at": payload.get("next_retry_at"),
            "last_error": payload.get("last_error"),
            "payload_json": _dumps(dict(payload)),
            "created_at": now,
            "updated_at": now,
        }
        columns = ", ".join(row)
        placeholders = ", ".join(f":{name}" for name in row)
        try:
            with transaction(self._conn) as conn:
                # 锁内查重：两路并发里先拿到写锁的那一路插入，后到的一路在这里读回既有行
                existing = self.find_by_client_event_id(client_event_id)
                if existing is not None:
                    return existing, False
                conn.execute(
                    f"INSERT INTO infusion_event ({columns}) VALUES ({placeholders})", row
                )
        except sqlite3.IntegrityError:
            # 兜底而非主路径：万一锁没兜住（例如有人绕过仓储层直写），读回既有行仍是幂等命中
            existing = self.find_by_client_event_id(client_event_id)
            if existing is None:
                raise
            return existing, False
        return self.get(row["event_id"]) or {}, True

    def mark_writeback(self, event_id: str, status: str, *, attempt_count: int | None = None,
                       next_retry_at: str | None = None, last_error: str | None = None) -> dict:
        if status not in WRITEBACK_STATES:
            raise ValueError(f"非法写回状态：{status}")
        assignments = ["writeback_status = :status", "updated_at = :updated_at"]
        params: dict[str, Any] = {
            "event_id": event_id,
            "status": status,
            "updated_at": utc_now(),
        }
        if attempt_count is not None:
            assignments.append("attempt_count = :attempt_count")
            params["attempt_count"] = int(attempt_count)
        if next_retry_at is not None:
            assignments.append("next_retry_at = :next_retry_at")
            params["next_retry_at"] = next_retry_at
        if last_error is not None:
            assignments.append("last_error = :last_error")
            params["last_error"] = last_error
        with transaction(self._conn) as conn:
            conn.execute(
                f"UPDATE infusion_event SET {', '.join(assignments)} WHERE event_id = :event_id",
                params,
            )
        return self.get(event_id) or {}

    def get(self, event_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM infusion_event WHERE event_id = ?", (event_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["payload"] = _loads(data.pop("payload_json"), {})
        return data

    def find_by_client_event_id(self, client_event_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM infusion_event WHERE client_event_id = ?", (client_event_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["payload"] = _loads(data.pop("payload_json"), {})
        return data

    def list_by_writeback_status(self, status: str, *, due_before: str | None = None) -> list[dict]:
        if due_before is None:
            rows = self._conn.execute(
                "SELECT * FROM infusion_event WHERE writeback_status = ? ORDER BY created_at ASC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM infusion_event WHERE writeback_status = ? "
                "AND (next_retry_at IS NULL OR next_retry_at <= ?) ORDER BY created_at ASC",
                (status, due_before),
            ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["payload"] = _loads(data.pop("payload_json"), {})
            out.append(data)
        return out

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM infusion_event").fetchone()[0])

    def count_by_writeback_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT writeback_status, COUNT(*) AS n FROM infusion_event GROUP BY writeback_status"
        ).fetchall()
        return {row["writeback_status"]: int(row["n"]) for row in rows}


class WritebackQueueRepository:
    """``writeback_queue`` 读写：待写回队列（一笔出站欠账一行）。

    **它不产生业务事件**。``infusion_event`` 才是事件的唯一真源；队列只是把
    「这笔写回还欠着对端吗」这件事单独记一份，好让信息科按架、按床、按到期时刻
    去查，而不是在一堆已送达的事件里翻。

    一笔事件在队列里最多一行（``event_id`` 唯一）：``enqueue`` 走 upsert，
    重试一次就更新一次，不追加。
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    # ── 写 ────────────────────────────────────────────────────────────
    def enqueue(self, payload: Mapping[str, Any]) -> tuple[dict, bool]:
        """入队（upsert）。返回 (队列行, 是否新建)。

        同一 ``event_id`` 再次入队是**更新**而不是追加：一条欠账在它的生命周期里
        只有一行，重试次数、下次重试时刻、错误摘要都在这一行上滚动。
        """
        now = utc_now()
        event_id = payload["event_id"]
        status = payload.get("status", "queued")
        if status not in WRITEBACK_QUEUE_STATES:
            raise ValueError(f"非法队列状态：{status}")
        row = {
            "queue_id": payload.get("queue_id") or new_id("wbq"),
            "event_id": event_id,
            "client_event_id": payload.get("client_event_id") or "",
            "session_id": payload.get("session_id"),
            "rack_id": payload.get("rack_id"),
            "bed_no": payload.get("bed_no"),
            "bag_seq": payload.get("bag_seq"),
            "payload_json": _dumps(dict(payload.get("payload") or {})),
            "attempt_count": int(payload.get("attempt_count", 0)),
            "next_retry_at": payload.get("next_retry_at"),
            "last_error": payload.get("last_error"),
            "status": status,
            "dead_letter_reason": payload.get("dead_letter_reason"),
            "last_attempt_at": payload.get("last_attempt_at") or now,
            "created_at": now,
            "updated_at": now,
        }
        columns = ", ".join(row)
        placeholders = ", ".join(f":{name}" for name in row)
        with transaction(self._conn) as conn:
            existing = self.get_by_event_id(event_id)
            if existing is None:
                conn.execute(
                    f"INSERT INTO writeback_queue ({columns}) VALUES ({placeholders})", row
                )
                return self.get(row["queue_id"]) or {}, True
            assignments = ", ".join(
                f"{name} = :{name}"
                for name in row
                if name not in ("queue_id", "event_id", "created_at")
            )
            conn.execute(
                f"UPDATE writeback_queue SET {assignments} WHERE event_id = :event_id", row
            )
        return self.get_by_event_id(event_id) or {}, False

    def update(self, queue_id: str, **fields: Any) -> dict:
        """更新队列行（白名单字段）。返回更新后的行。"""
        allowed = {
            "status",
            "attempt_count",
            "next_retry_at",
            "last_error",
            "dead_letter_reason",
            "last_attempt_at",
            "payload_json",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"不可写字段：{sorted(unknown)}")
        if "status" in fields and fields["status"] not in WRITEBACK_QUEUE_STATES:
            raise ValueError(f"非法队列状态：{fields['status']}")
        params: dict[str, Any] = {"queue_id": queue_id, "updated_at": utc_now()}
        assignments = ["updated_at = :updated_at"]
        for name, value in fields.items():
            assignments.append(f"{name} = :{name}")
            params[name] = value
        with transaction(self._conn) as conn:
            conn.execute(
                f"UPDATE writeback_queue SET {', '.join(assignments)} WHERE queue_id = :queue_id",
                params,
            )
        return self.get(queue_id) or {}

    # ── 读 ────────────────────────────────────────────────────────────
    @staticmethod
    def _hydrate(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["payload"] = _loads(data.pop("payload_json"), {})
        return data

    def get(self, queue_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM writeback_queue WHERE queue_id = ?", (queue_id,)
        ).fetchone()
        return self._hydrate(row) if row is not None else None

    def get_by_event_id(self, event_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM writeback_queue WHERE event_id = ?", (event_id,)
        ).fetchone()
        return self._hydrate(row) if row is not None else None

    def list_by_status(
        self,
        status: str,
        *,
        due_before: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        if due_before is None:
            rows = self._conn.execute(
                "SELECT * FROM writeback_queue WHERE status = ? ORDER BY created_at ASC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM writeback_queue WHERE status = ? "
                "AND (next_retry_at IS NULL OR next_retry_at <= ?) "
                "ORDER BY created_at ASC LIMIT ?",
                (status, due_before, limit),
            ).fetchall()
        return [self._hydrate(row) for row in rows]

    def list_pending(
        self,
        *,
        statuses: Sequence[str] = ("queued", "dead_letter", "failed"),
        rack_id: str | None = None,
        bed_no: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """待处理清单：默认返回**还没送达**的三种状态（``sent`` 不在其中）。"""
        clauses = ["status IN (%s)" % ", ".join(f":s{i}" for i in range(len(statuses)))]
        params: dict[str, Any] = {f"s{i}": s for i, s in enumerate(statuses)}
        if rack_id:
            clauses.append("rack_id = :rack_id")
            params["rack_id"] = rack_id
        if bed_no:
            clauses.append("bed_no = :bed_no")
            params["bed_no"] = bed_no
        params["limit"] = int(limit)
        rows = self._conn.execute(
            "SELECT * FROM writeback_queue WHERE " + " AND ".join(clauses)
            + " ORDER BY created_at ASC LIMIT :limit",
            params,
        ).fetchall()
        return [self._hydrate(row) for row in rows]

    def count_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM writeback_queue GROUP BY status"
        ).fetchall()
        return {row["status"]: int(row["n"]) for row in rows}

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM writeback_queue").fetchone()[0])


class InferenceAuditRepository:
    """``inference_audit`` 只追加，不更新不删除。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def append(self, record: Mapping[str, Any]) -> int:
        row = {
            "request_id": record["request_id"],
            "session_id": record.get("session_id"),
            "bag_seq": record.get("bag_seq"),
            "model": record["model"],
            "prompt_hash": record["prompt_hash"],
            "prompt_version": record.get("prompt_version"),
            "latency_ms": record.get("latency_ms"),
            "prompt_tokens": record.get("prompt_tokens"),
            "completion_tokens": record.get("completion_tokens"),
            "decision_source": record.get("decision_source", "threshold_only"),
            "fallback_reason": record.get("fallback_reason"),
            "pending_manual_review": int(bool(record.get("pending_manual_review", False))),
            "decision": record.get("decision"),
            "confidence": record.get("confidence"),
            "created_at": record.get("created_at") or utc_now(),
        }
        columns = ", ".join(row)
        placeholders = ", ".join(f":{name}" for name in row)
        with transaction(self._conn) as conn:
            cursor = conn.execute(
                f"INSERT INTO inference_audit ({columns}) VALUES ({placeholders})", row
            )
        return int(cursor.lastrowid or 0)

    def list_by_request(self, request_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM inference_audit WHERE request_id = ? ORDER BY audit_id ASC",
            (request_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_recent(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM inference_audit ORDER BY audit_id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM inference_audit").fetchone()[0])

    def count_by_decision_source(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT decision_source, COUNT(*) AS n FROM inference_audit GROUP BY decision_source"
        ).fetchall()
        return {row["decision_source"]: int(row["n"]) for row in rows}


class CalibrationRepository:
    """``bag_type_calibration`` 读写：**同袋型同时只有一个生效版本**。

    保存新版本时把旧版本置为 ``active=0`` 而不是更新它——历史版本要能原样取回回答
    「当时那条曲线是谁标的、什么时候标的、用的哪几条样例」。
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def next_version(self, bag_type_code: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM bag_type_calibration "
            "WHERE bag_type_code = ?",
            (bag_type_code,),
        ).fetchone()
        return int(row["v"]) + 1

    def save(
        self,
        record: BagTypeCalibration,
        *,
        created_at: str | None = None,
        review_threshold: float | None = None,
    ) -> tuple[BagTypeCalibration, int | None, str]:
        """保存一个版本，返回 ``(落库后的记录, 被取代的旧版本号, 版本状态)``。

        版本号由仓储层分配（同袋型自增），**忽略**记录里自带的 version；
        落库、旧版本失活与生效区间收口在**同一事务**里，不允许出现「两个生效版本」。

        生效区间（task 27）：

        * ``effective_from`` 留空 = 立刻生效（与既有行为逐字一致），旧版收到
          ``effective_to = 新版生效时刻`` 后失活；
        * ``effective_from`` 在未来 = **预约登记**：新版 ``active=0``、状态
          ``scheduled``，当前生效版本**一点不动**，当天算出来的毫升数不变；
        * 新登记的区间与既有版本相交且不是改版交接点 → 拒收（409，见
          :func:`calibration.versions.assert_no_overlapping_window`）。
        """
        stamp = created_at or record.calibrated_at or utc_now()
        with transaction(self._conn) as conn:
            version = self.next_version(record.bag_type_code)
            calibration_id = record.calibration_id or new_id("cal")
            existing = self.list_by_code(record.bag_type_code, conn=conn)
            superseded = next((item for item in existing if item.active), None)
            window = VersionWindow(
                effective_from=record.effective_from or stamp,
                effective_to=record.effective_to,
            )
            superseded_to, status = plan_supersede(superseded, window)
            # 改版交接本身不算冲突（旧版会被收口到新版生效时刻）；其余相交一律拒收。
            assert_no_overlapping_window(
                existing,
                window,
                bag_type_code=record.bag_type_code,
                closing=(
                    {superseded.calibration_id: superseded_to}
                    if superseded is not None and superseded_to is not None
                    else {}
                ),
            )
            stored = replace(
                record,
                version=version,
                calibration_id=calibration_id,
                effective_from=window.effective_from,
                effective_to=window.effective_to,
                active=status == "active",
            )
            if superseded is not None and status == "active":
                # 交接：旧版收口并失活。
                conn.execute(
                    "UPDATE bag_type_calibration SET active = 0, effective_to = ? "
                    "WHERE calibration_id = ?",
                    (superseded_to, superseded.calibration_id),
                )
            elif superseded is not None:
                # 预约登记：**事先**把旧版的生效区间收口到新版的生效时刻（区间是
                # 计划，不是事后补记），但**不**动 active —— 当天算出来的毫升数一点
                # 不变，到点由 promote_due 只翻标志完成交接。提前换版会让当晚还在
                # 监测的病区凭空多出几百毫升的差值。
                conn.execute(
                    "UPDATE bag_type_calibration SET effective_to = ? "
                    "WHERE calibration_id = ?",
                    (superseded_to, superseded.calibration_id),
                )
            row = stored.to_row(created_at=stamp, active=status == "active")
            columns = list(row)
            conn.execute(
                f"INSERT INTO bag_type_calibration ({', '.join(columns)}) "
                f"VALUES ({', '.join(':' + name for name in columns)})",
                row,
            )
        return stored, (superseded.version if superseded else None), status

    def promote(self, calibration_id: str, *, effective_from: str) -> int | None:
        """把一个到期未生效的版本转正，并把前一版按 ``effective_from`` 收口。

        返回被取代的旧版本号。**不重算任何会话的毫升数**：交接当天的读数变化必须
        走偏差闸门由人工复核，不能由一次版本交接悄悄改掉病区正在看的数字。
        """
        with transaction(self._conn) as conn:
            row = conn.execute(
                "SELECT * FROM bag_type_calibration WHERE calibration_id = ?",
                (calibration_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"版本不存在：{calibration_id}")
            promoted = BagTypeCalibration.from_row(row)
            if promoted.active:
                return None
            previous = conn.execute(
                "SELECT * FROM bag_type_calibration WHERE bag_type_code = ? AND active = 1 "
                "AND version < ? ORDER BY version DESC LIMIT 1",
                (promoted.bag_type_code, promoted.version),
            ).fetchone()
            conn.execute(
                "UPDATE bag_type_calibration SET active = 1 WHERE calibration_id = ?",
                (calibration_id,),
            )
            superseded_version: int | None = None
            if previous is not None:
                superseded_version = int(previous["version"])
                conn.execute(
                    "UPDATE bag_type_calibration SET active = 0, effective_to = ? "
                    "WHERE calibration_id = ?",
                    (effective_from, previous["calibration_id"]),
                )
        return superseded_version

    def get_active(
        self, bag_type_code: str, *, conn: sqlite3.Connection | None = None
    ) -> BagTypeCalibration | None:
        handle = conn or self._conn
        row = handle.execute(
            "SELECT * FROM bag_type_calibration WHERE bag_type_code = ? AND active = 1 "
            "ORDER BY version DESC LIMIT 1",
            (bag_type_code,),
        ).fetchone()
        return BagTypeCalibration.from_row(row) if row else None

    def get(self, calibration_id: str) -> BagTypeCalibration | None:
        row = self._conn.execute(
            "SELECT * FROM bag_type_calibration WHERE calibration_id = ?", (calibration_id,)
        ).fetchone()
        return BagTypeCalibration.from_row(row) if row else None

    def list_by_code(
        self, bag_type_code: str, *, conn: sqlite3.Connection | None = None
    ) -> list[BagTypeCalibration]:
        handle = conn or self._conn
        rows = handle.execute(
            "SELECT * FROM bag_type_calibration WHERE bag_type_code = ? ORDER BY version ASC",
            (bag_type_code,),
        ).fetchall()
        return [BagTypeCalibration.from_row(row) for row in rows]

    def list_active(self) -> list[BagTypeCalibration]:
        rows = self._conn.execute(
            "SELECT * FROM bag_type_calibration WHERE active = 1 ORDER BY bag_type_code ASC"
        ).fetchall()
        return [BagTypeCalibration.from_row(row) for row in rows]

    def count(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM bag_type_calibration").fetchone()[0]
        )

    # ── 改版重算轨迹（append-only） ────────────────────────────────────
    def log_recalc(
        self,
        *,
        calibration: BagTypeCalibration,
        previous_version: int | None,
        session: Mapping[str, Any],
        previous_volume_ml: float | None,
        new_volume_ml: float | None,
        reason: str = "calibration_version_changed",
        relative_deviation: float | None = None,
        review_required: bool = False,
    ) -> int:
        """追加一条重算轨迹。**只追加**：改版不改旧记录，回答「当时为什么报这个数」。

        ``review_required=True`` 表示这一条**没有**覆盖旧读数（偏差超阈值被闸门拦下），
        ``new_volume_ml`` 是「本该给出的数」而不是「已经生效的数」。
        """
        delta = None
        if previous_volume_ml is not None and new_volume_ml is not None:
            delta = float(new_volume_ml) - float(previous_volume_ml)
        with transaction(self._conn) as conn:
            cursor = conn.execute(
                """
                INSERT INTO calibration_recalc_log (
                    calibration_id, bag_type_code, previous_version, new_version,
                    session_id, bag_seq, rack_id, bed_no,
                    previous_volume_ml, new_volume_ml, delta_ml, reason,
                    relative_deviation, review_required, recalculated_at
                ) VALUES (
                    :calibration_id, :bag_type_code, :previous_version, :new_version,
                    :session_id, :bag_seq, :rack_id, :bed_no,
                    :previous_volume_ml, :new_volume_ml, :delta_ml, :reason,
                    :relative_deviation, :review_required, :recalculated_at
                )
                """,
                {
                    "calibration_id": calibration.calibration_id,
                    "bag_type_code": calibration.bag_type_code,
                    "previous_version": previous_version,
                    "new_version": calibration.version,
                    "session_id": session["session_id"],
                    "bag_seq": session.get("bag_seq"),
                    "rack_id": session.get("rack_id"),
                    "bed_no": session.get("bed_no"),
                    "previous_volume_ml": previous_volume_ml,
                    "new_volume_ml": new_volume_ml,
                    "delta_ml": delta,
                    "reason": reason,
                    "relative_deviation": relative_deviation,
                    "review_required": 1 if review_required else 0,
                    "recalculated_at": utc_now(),
                },
            )
        return int(cursor.lastrowid or 0)

    def list_recalc(
        self, bag_type_code: str | None = None, limit: int = 200
    ) -> list[dict]:
        if bag_type_code:
            rows = self._conn.execute(
                "SELECT * FROM calibration_recalc_log WHERE bag_type_code = ? "
                "ORDER BY recalc_id ASC LIMIT ?",
                (bag_type_code, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM calibration_recalc_log ORDER BY recalc_id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_recalc(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM calibration_recalc_log").fetchone()[0]
        )

    # ── 改版偏差告警（task 27，append-only）──────────────────────────────
    def raise_alert(
        self,
        *,
        calibration: BagTypeCalibration,
        previous_version: int | None,
        session: Mapping[str, Any],
        previous_volume_ml: float | None,
        proposed_volume_ml: float | None,
        relative_deviation: float,
        threshold: float,
        severity: str,
    ) -> int:
        """登记一条「改版偏差超阈值、旧读数未覆盖」的告警。只追加，不改旧行。"""
        delta = None
        if previous_volume_ml is not None and proposed_volume_ml is not None:
            delta = float(proposed_volume_ml) - float(previous_volume_ml)
        with transaction(self._conn) as conn:
            cursor = conn.execute(
                """
                INSERT INTO calibration_version_alert (
                    calibration_id, bag_type_code, previous_version, new_version,
                    session_id, bag_seq, rack_id, bed_no,
                    previous_volume_ml, proposed_volume_ml, delta_ml,
                    relative_deviation, threshold, severity, resolution, raised_at
                ) VALUES (
                    :calibration_id, :bag_type_code, :previous_version, :new_version,
                    :session_id, :bag_seq, :rack_id, :bed_no,
                    :previous_volume_ml, :proposed_volume_ml, :delta_ml,
                    :relative_deviation, :threshold, :severity, 'pending', :raised_at
                )
                """,
                {
                    "calibration_id": calibration.calibration_id,
                    "bag_type_code": calibration.bag_type_code,
                    "previous_version": previous_version,
                    "new_version": calibration.version,
                    "session_id": session["session_id"],
                    "bag_seq": session.get("bag_seq"),
                    "rack_id": session.get("rack_id"),
                    "bed_no": session.get("bed_no"),
                    "previous_volume_ml": previous_volume_ml,
                    "proposed_volume_ml": proposed_volume_ml,
                    "delta_ml": delta,
                    "relative_deviation": float(relative_deviation),
                    "threshold": float(threshold),
                    "severity": severity,
                    "raised_at": utc_now(),
                },
            )
        return int(cursor.lastrowid or 0)

    def list_alerts(
        self,
        bag_type_code: str | None = None,
        *,
        resolution: str | None = None,
        session_id: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[Any] = []
        if bag_type_code:
            clauses.append("bag_type_code = ?")
            params.append(bag_type_code)
        if resolution:
            clauses.append("resolution = ?")
            params.append(resolution)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM calibration_version_alert{where} ORDER BY alert_id ASC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_alert(self, alert_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM calibration_version_alert WHERE alert_id = ?", (alert_id,)
        ).fetchone()
        return dict(row) if row else None

    def resolve_alert(
        self, alert_id: int, resolution: str, *, by: str, note: str = ""
    ) -> dict:
        """结掉一条告警：``applied``（复核通过，按新读数落库）或 ``dismissed``（否决）。

        告警表本身 append-only：**结案写的是同一行的 resolution 列**，且只允许
        ``pending → applied/dismissed`` 一次迁移；重复结案会显式报错而不是默默改写，
        否则「谁按的」在两次结案后就说不清了。
        """
        if resolution not in ("applied", "dismissed"):
            raise ValueError(f"非法告警结案状态：{resolution}（只允许 applied / dismissed）")
        with transaction(self._conn) as conn:
            row = conn.execute(
                "SELECT * FROM calibration_version_alert WHERE alert_id = ?", (alert_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"告警不存在：alert_id={alert_id}")
            if row["resolution"] != "pending":
                raise ValueError(
                    f"告警 {alert_id} 已于 {row['resolved_at']} 结案为 "
                    f"{row['resolution']}（复核人 {row['resolved_by']}），不可重复处理"
                )
            conn.execute(
                "UPDATE calibration_version_alert SET resolution = ?, resolved_by = ?, "
                "resolved_at = ?, resolution_note = ? WHERE alert_id = ?",
                (resolution, by, utc_now(), note, alert_id),
            )
        return self.get_alert(alert_id) or {}

    def count_alerts(self, resolution: str | None = None) -> int:
        if resolution:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM calibration_version_alert WHERE resolution = ?",
                (resolution,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM calibration_version_alert").fetchone()
        return int(row[0])

class DeviationRepository:
    """``calibration_deviation`` 读写：**append-only**，只追加不修改不删除。

    每条样本 = 一次「护士确认已换液」留下的残留证据。这张表回答两个问题：

    * 「这个袋型的残留偏差是怎么定出来的？」——逐条样本可追到床号、袋序、巡视时刻；
    * 「为什么不生效？」——不可用的样本（换液前看得很高、没有读数、没声明袋型）
      同样入库并带原因码，不静默丢弃，否则复核的人会以为「没记录 = 没发生」。
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ── 写入 ────────────────────────────────────────────────────────────
    def append(
        self, observation: DeviationObservation, *, note: str = ""
    ) -> tuple[int, bool]:
        """追加一条样本，返回 ``(deviation_id, created)``。

        * 同一 ``session_id`` 只留一条（表上有唯一索引）：重放同一次确认不会再加样本，
          与 ``infusion_event`` 的幂等口径一致；
        * ``sample_count`` 在**同一事务内**按该袋型已有行数 +1 计算，append-only 且
          本仓储不提供删除，权威值与 ``COUNT(*)`` 一致（回归用例逐条核对过）；
        * 样本不可用时照写不误，带 ``ineligible_reason``——「为什么这条没参与定标」
          本身就是要留档的信息。
        """
        if not observation.session_id:
            # 没有会话就无从去重也无处追溯，这种样本不入库。
            return 0, False
        existing = self.find_by_session(observation.session_id)
        if existing is not None:
            return int(existing["deviation_id"]), False

        eligible = observation.eligible
        with transaction(self._conn) as conn:
            count_row = conn.execute(
                "SELECT COUNT(*) AS n FROM calibration_deviation WHERE bag_type_code = ?",
                (observation.bag_type_code,),
            ).fetchone()
            cursor = conn.execute(
                """
                INSERT INTO calibration_deviation (
                    bag_type_code, session_id, bag_seq, rack_id, bed_no, source,
                    last_fill_ratio, model_volume_ml, residual_ml, eligible,
                    ineligible_reason, calibration_version, sample_count, note, recorded_at
                ) VALUES (
                    :bag_type_code, :session_id, :bag_seq, :rack_id, :bed_no, :source,
                    :last_fill_ratio, :model_volume_ml, :residual_ml, :eligible,
                    :ineligible_reason, :calibration_version, :sample_count, :note, :recorded_at
                )
                """,
                {
                    "bag_type_code": observation.bag_type_code,
                    "session_id": observation.session_id,
                    "bag_seq": observation.bag_seq or None,
                    "rack_id": observation.rack_id or None,
                    "bed_no": observation.bed_no or None,
                    "source": observation.source,
                    "last_fill_ratio": observation.last_fill_ratio,
                    "model_volume_ml": observation.model_volume_ml,
                    "residual_ml": observation.residual_ml,
                    "eligible": 1 if eligible else 0,
                    "ineligible_reason": None if eligible else _ineligible_reason(observation),
                    "calibration_version": observation.calibration_version,
                    "sample_count": int(count_row["n"]) + 1,
                    "note": note,
                    "recorded_at": observation.occurred_at or utc_now(),
                },
            )
        return int(cursor.lastrowid or 0), True

    def record_session(
        self, session: Mapping[str, Any], *, note: str = ""
    ) -> tuple[int, bool]:
        """由一条会话行构造证据并入库；会话没袋型或没读数时直接返回 ``(0, False)``。"""
        observation = deviation_observation_from_session(session)
        if observation is None:
            return 0, False
        return self.append(observation, note=note)

    # ── 读取 ────────────────────────────────────────────────────────────
    def find_by_session(self, session_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM calibration_deviation WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_samples(
        self, bag_type_code: str | None = None, limit: int = 200
    ) -> list[dict]:
        if bag_type_code:
            rows = self._conn.execute(
                "SELECT * FROM calibration_deviation WHERE bag_type_code = ? "
                "ORDER BY deviation_id ASC LIMIT ?",
                (bag_type_code, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM calibration_deviation ORDER BY deviation_id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def observations(
        self, bag_type_code: str, limit: int = 500
    ) -> list[DeviationObservation]:
        """按袋型取出全部样本（含不可用的），交给 :func:`calibration.cavity.learn_residual`。"""
        rows = self.list_samples(bag_type_code, limit=limit)
        return [
            DeviationObservation(
                bag_type_code=str(row["bag_type_code"]),
                last_fill_ratio=row["last_fill_ratio"],
                model_volume_ml=row["model_volume_ml"],
                session_id=str(row["session_id"] or ""),
                bag_seq=str(row["bag_seq"] or ""),
                rack_id=str(row["rack_id"] or ""),
                bed_no=str(row["bed_no"] or ""),
                calibration_version=row["calibration_version"],
                source=str(row["source"]),
                occurred_at=str(row["recorded_at"] or ""),
            )
            for row in rows
        ]

    def summary(
        self, bag_type_code: str, *, usable_capacity_ml: float
    ) -> LearnedResidual:
        """该袋型的自校准结论（残差值、样本数、是否生效、为什么）。"""
        return learn_residual(
            self.observations(bag_type_code), usable_capacity_ml=usable_capacity_ml
        )

    def count(self, bag_type_code: str | None = None) -> int:
        if bag_type_code:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM calibration_deviation WHERE bag_type_code = ?",
                (bag_type_code,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM calibration_deviation").fetchone()
        return int(row["n"])


def _ineligible_reason(observation: DeviationObservation) -> str:
    """样本不可用的原因码。判定顺序与 ``DeviationObservation.eligible`` 相反着走。"""
    if observation.model_volume_ml is None or float(observation.model_volume_ml) <= 0.0:
        return "no_model_volume"
    if observation.last_fill_ratio is None:
        return "no_fill_ratio"
    if float(observation.last_fill_ratio) > MAX_RESIDUE_EVIDENCE_RATIO:
        return "not_near_empty"
    return "missing_bag_type"
