"""保留期策略的测试：过期清理、非过期保留、未完成会话不被误删。

三条硬纪律各有一组用例钉住（见 :mod:`store.retention` 的模块说明）：
未完成的会话不删、有写回欠账的会话不删、审计链永不删。
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from infusion_monitor.store import (
    AuditChain,
    EventRepository,
    RetentionPolicy,
    SessionRepository,
    apply_migrations,
    connect,
    run_retention,
)
from infusion_monitor.store.retention import BLOCK_REASONS, plan_retention

#: 固定参照时刻：2026-10-23 08:00 UTC。用它算「多久以前」，测试不依赖真实时钟。
NOW = datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def _make_frames(directory: Path, count: int = 3, size: int = 512) -> Path:
    """造一个假的帧目录（内容是占位字节，不是真 JPEG——本模块只关心目录与体积）。"""
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"frame_{index:05d}.jpg").write_bytes(b"\x00" * size)
    return directory


def _session(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    frames_root: Path,
    days_ago: float,
    state: str = "replaced",
    bag_seq: str = "B01",
    shared_key: str = "rack-shared",
    captured_offset_hours: float = 0.0,
) -> dict:
    """落一条会话（含真实帧目录）。``shared_key`` 决定它与谁共用帧目录。"""
    captured = NOW - timedelta(days=days_ago, hours=captured_offset_hours)
    frames_dir = frames_root / shared_key
    if not frames_dir.exists():
        _make_frames(frames_dir)
    repo = SessionRepository(conn)
    row = repo.upsert_session(
        {
            "rack_id": "R-EXAMPLE-01",
            "bed_no": "012",
            "infusion_order_no": "EXAMPLE-ORDER-0001",
            "shift_key": "2026-10-20-AM",
            "bag_seq": bag_seq,
            "slot_index": 0,
            "bag_type_code": "TYPE-A",
            "frames_dir": str(frames_dir),
            "frame_count": 3,
            "captured_at": _iso(captured),
            "state": state,
            "calibration_state": "calibrated",
        }
    )
    generated_id = row["session_id"]
    conn.execute(
        "UPDATE bag_session SET session_id = ? WHERE session_id = ?",
        (session_id, generated_id),
    )
    # 关联的出站事件：默认已送达（sent），有写回欠账的用例另建
    EventRepository(conn).append_event(
        {
            "client_event_id": f"evt-{session_id}",
            "session_id": session_id,
            "rack_id": "R-EXAMPLE-01",
            "bed_no": "012",
            "infusion_order_no": "EXAMPLE-ORDER-0001",
            "bag_seq": bag_seq,
            "operator_id": "N-EXAMPLE-7",
            "occurred_at": _iso(captured),
            "writeback_status": "sent",
            "payload_json": "{}",
        }
    )
    return {"session_id": session_id, "frames_dir": str(frames_dir), "bag_seq": bag_seq}


@pytest.fixture
def store(tmp_path):
    """一份已迁移的临时库 + 帧根目录。"""
    connection = connect("sqlite:///:memory:")
    apply_migrations(connection)
    yield connection, tmp_path / "frames"
    connection.close()


# ── 一、过期清理 ──────────────────────────────────────────────────────────
def test_expired_session_frames_are_purged(store):
    """过了保留期、且一切了结的会话：帧目录被删，读数留在库里。"""
    conn, frames_root = store
    item = _session(
        conn, session_id="sess-old", frames_root=frames_root, days_ago=45, shared_key="old-dir"
    )
    frames_dir = Path(item["frames_dir"])
    assert frames_dir.is_dir()

    result = run_retention(
        conn,
        frames_root,
        policy=RetentionPolicy(session_retention_days=30.0),
        now=NOW,
    )

    assert len(result.purged) == 1
    assert result.purged[0]["session_id"] == "sess-old"
    assert not frames_dir.exists(), "过期帧目录应当被删除"
    assert result.freed_bytes == 3 * 512, "释放字节数按目录实际体积算"

    row = conn.execute(
        "SELECT frames_retention_state, frames_purged_at, fill_ratio FROM bag_session "
        "WHERE session_id = ?",
        ("sess-old",),
    ).fetchone()
    assert row["frames_retention_state"] == "purged"
    assert row["frames_purged_at"], "应当留下清理时刻"

    # 台账有一行
    archive = conn.execute(
        "SELECT action, reason, bytes, archive_path FROM retention_archive WHERE session_id = ?",
        ("sess-old",),
    ).fetchone()
    assert archive["action"] == "deleted"
    assert archive["reason"] == "retention_expired"
    assert archive["bytes"] == 3 * 512
    assert archive["archive_path"] is None


def test_purged_session_keeps_its_readings(store):
    """像素归零、读数留下：会话行的面积比与毫升数不受清理影响。"""
    conn, frames_root = store
    item = _session(
        conn, session_id="sess-read", frames_root=frames_root, days_ago=60, shared_key="read-dir"
    )
    conn.execute(
        "UPDATE bag_session SET area_ratio_series = ?, estimated_volume_ml = ? WHERE session_id = ?",
        ('[0.58,0.44]', 273.2, item["session_id"]),
    )

    run_retention(
        conn,
        frames_root,
        policy=RetentionPolicy(session_retention_days=30.0),
        now=NOW,
    )

    row = conn.execute(
        "SELECT area_ratio_series, estimated_volume_ml FROM bag_session WHERE session_id = ?",
        (item["session_id"],),
    ).fetchone()
    assert row["area_ratio_series"] == "[0.58,0.44]"
    assert row["estimated_volume_ml"] == 273.2


def test_archive_mode_moves_frames_instead_of_deleting(store, tmp_path):
    """配了归档根目录：帧被搬进归档区，会话行的 frames_dir 指向新位置。"""
    conn, frames_root = store
    archive_root = tmp_path / "archive"
    item = _session(
        conn, session_id="sess-arch", frames_root=frames_root, days_ago=45, shared_key="arch-dir"
    )
    original = Path(item["frames_dir"])

    result = run_retention(
        conn,
        frames_root,
        policy=RetentionPolicy(session_retention_days=30.0, archive_root=archive_root),
        now=NOW,
    )

    assert len(result.archived) == 1
    assert not original.exists()
    moved = Path(
        conn.execute(
            "SELECT frames_dir FROM bag_session WHERE session_id = ?", ("sess-arch",)
        ).fetchone()["frames_dir"]
    )
    assert moved.is_dir(), "归档后的帧目录应当还在"
    assert str(archive_root) in str(moved)
    assert len(list(moved.glob("frame_*.jpg"))) == 3

    row = conn.execute(
        "SELECT frames_retention_state FROM bag_session WHERE session_id = ?", ("sess-arch",)
    ).fetchone()
    assert row["frames_retention_state"] == "archived"


def test_dry_run_changes_nothing(store):
    """dry-run 只报不动：不删文件、不改会话状态、不写台账。"""
    conn, frames_root = store
    item = _session(
        conn, session_id="sess-dry", frames_root=frames_root, days_ago=45, shared_key="dry-dir"
    )
    frames_dir = Path(item["frames_dir"])

    result = run_retention(
        conn,
        frames_root,
        policy=RetentionPolicy(session_retention_days=30.0, dry_run=True),
        now=NOW,
    )

    assert len(result.purged) == 1, "dry-run 也要报出会清哪一袋"
    assert frames_dir.is_dir(), "dry-run 不得删文件"
    row = conn.execute(
        "SELECT frames_retention_state FROM bag_session WHERE session_id = ?",
        ("sess-dry",),
    ).fetchone()
    assert row["frames_retention_state"] == "kept"
    assert conn.execute("SELECT COUNT(*) FROM retention_archive").fetchone()[0] == 0


def test_second_run_does_not_touch_the_same_session_twice(store):
    """清理过的会话不会被二次处理（保留状态记住了）。"""
    conn, frames_root = store
    _session(
        conn, session_id="sess-twice", frames_root=frames_root, days_ago=45, shared_key="twice-dir"
    )

    first = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )
    second = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )

    assert len(first.purged) == 1
    assert second.purged == []
    assert second.absent == [], "第二次不该再把它当成待清理对象"


# ── 二、非过期保留 ────────────────────────────────────────────────────────
def test_recent_session_is_kept(store):
    """还在保留期内的会话：一个字节都不动。"""
    conn, frames_root = store
    item = _session(
        conn, session_id="sess-new", frames_root=frames_root, days_ago=3, shared_key="new-dir"
    )
    frames_dir = Path(item["frames_dir"])

    result = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )

    assert result.purged == []
    assert result.archived == []
    assert frames_dir.is_dir()
    reasons = {item["session_id"]: item["reason"] for item in result.blocked}
    assert reasons["sess-new"] == "within_retention"


def test_custom_retention_window_is_respected(store):
    """保留期改成 7 天：第 10 天的会话就该被清掉。"""
    conn, frames_root = store
    _session(
        conn, session_id="sess-10d", frames_root=frames_root, days_ago=10, shared_key="ten-dir"
    )

    result = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=7.0), now=NOW
    )

    assert len(result.purged) == 1


def test_blocked_reasons_are_reported(store):
    """被留下的会话要带原因码：只报删了多少、不报留下多少等于没报。"""
    conn, frames_root = store
    _session(
        conn,
        session_id="sess-monitor",
        frames_root=frames_root,
        days_ago=45,
        state="monitoring",
        shared_key="mon-dir",
    )

    result = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )

    assert result.kept_sessions == 1
    blocked = result.blocked[0]
    assert blocked["session_id"] == "sess-monitor"
    assert blocked["reason"] == "session_not_finished"
    assert blocked["reason_text"] == BLOCK_REASONS["session_not_finished"]


# ── 三、未完成 / 有欠账的会话不被误删 ─────────────────────────────────────
@pytest.mark.parametrize("state", ["monitoring", "needs_change"])
def test_unfinished_session_is_never_purged(store, state):
    """没结束的会话（monitoring / needs_change）帧目录一个都不动。"""
    conn, frames_root = store
    item = _session(
        conn,
        session_id=f"sess-{state}",
        frames_root=frames_root,
        days_ago=90,
        state=state,
        shared_key=f"{state}-dir",
    )
    frames_dir = Path(item["frames_dir"])

    result = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )

    assert result.purged == []
    assert frames_dir.is_dir(), "未完成的会话帧目录必须保留"
    row = conn.execute(
        "SELECT frames_retention_state FROM bag_session WHERE session_id = ?",
        (item["session_id"],),
    ).fetchone()
    assert row["frames_retention_state"] == "kept"


@pytest.mark.parametrize("status", ["pending", "pending_writeback", "failed"])
def test_session_with_unsettled_writeback_is_not_purged(store, status):
    """出站事件还没了结的会话：帧留着，信息科重放时还要靠它核对。"""
    conn, frames_root = store
    item = _session(
        conn, session_id="sess-unsettled", frames_root=frames_root, days_ago=90, shared_key="unsettled-dir"
    )
    conn.execute(
        "UPDATE infusion_event SET writeback_status = ? WHERE session_id = ?",
        (status, item["session_id"]),
    )
    frames_dir = Path(item["frames_dir"])

    result = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )

    assert result.purged == []
    assert frames_dir.is_dir()
    assert result.blocked[0]["reason"] == "writeback_unsettled"
    assert result.blocked[0]["detail"] == [status]


def test_session_with_queued_writeback_is_not_purged(store):
    """待写回队列里还挂着欠账的会话同样保留。"""
    conn, frames_root = store
    item = _session(
        conn, session_id="sess-queued", frames_root=frames_root, days_ago=90, shared_key="queued-dir"
    )
    conn.execute(
        "INSERT INTO writeback_queue (queue_id, event_id, client_event_id, session_id, rack_id, "
        "bed_no, bag_seq, payload_json, status, created_at, updated_at) "
        "VALUES ('q-1', 'evt-x', 'evt-x', ?, 'R-EXAMPLE-01', '012', 'B01', '{}', 'queued', ?, ?)",
        (item["session_id"], _iso(NOW), _iso(NOW)),
    )
    frames_dir = Path(item["frames_dir"])

    result = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )

    assert result.purged == []
    assert frames_dir.is_dir()
    assert result.blocked[0]["reason"] == "writeback_queue_pending"


def test_session_without_outbound_event_is_not_purged(store):
    """没有出站事件的会话（状态不一致）保留并报出来，不静默清掉。"""
    conn, frames_root = store
    item = _session(
        conn, session_id="sess-noevent", frames_root=frames_root, days_ago=90, shared_key="noevent-dir"
    )
    conn.execute("DELETE FROM infusion_event WHERE session_id = ?", (item["session_id"],))
    frames_dir = Path(item["frames_dir"])

    result = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )

    assert result.purged == []
    assert frames_dir.is_dir()
    assert result.blocked[0]["reason"] == "no_event"


def test_shared_frames_dir_is_kept_while_sibling_bag_is_active(store):
    """同一段短片的几只袋子共用一个帧目录：只要有一只没到条件，整个目录都留着。"""
    conn, frames_root = store
    shared = frames_root / "shared-dir"
    _make_frames(shared)

    sessions = SessionRepository(conn)
    for bag_seq, state, session_id in (
        ("B01", "replaced", "sess-shared-done"),
        ("B02", "monitoring", "sess-shared-active"),
    ):
        sessions.upsert_session(
            {
                "rack_id": "R-EXAMPLE-02",
                "bed_no": "013",
                "infusion_order_no": "EXAMPLE-ORDER-0002",
                "shift_key": "2026-10-20-AM",
                "bag_seq": bag_seq,
                "slot_index": 0,
                "bag_type_code": "TYPE-A",
                "frames_dir": str(shared),
                "frame_count": 3,
                "captured_at": _iso(NOW - timedelta(days=45)),
                "state": state,
                "calibration_state": "calibrated",
            }
        )
        conn.execute(
            "UPDATE bag_session SET session_id = ? WHERE bag_seq = ?", (session_id, bag_seq)
        )
        EventRepository(conn).append_event(
            {
                "client_event_id": f"evt-{session_id}",
                "session_id": session_id,
                "rack_id": "R-EXAMPLE-02",
                "bed_no": "013",
                "infusion_order_no": "EXAMPLE-ORDER-0002",
                "bag_seq": bag_seq,
                "operator_id": "N-EXAMPLE-7",
                "occurred_at": _iso(NOW - timedelta(days=45)),
                "writeback_status": "sent",
                "payload_json": "{}",
            }
        )

    result = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )

    assert result.purged == [], "还有一只袋子没结束时，整个目录都不能删"
    assert shared.is_dir()
    reasons = {item["session_id"]: item["reason"] for item in result.blocked}
    assert reasons["sess-shared-done"] == "frames_shared_with_active_session"
    assert reasons["sess-shared-active"] == "session_not_finished"


def test_shared_dir_is_purged_once_when_all_bags_are_done(store):
    """两只袋子都了结时，目录只处理一次（字节数不按袋数重复算）。"""
    conn, frames_root = store
    shared = frames_root / "shared-done-dir"
    _make_frames(shared, count=4, size=100)

    sessions = SessionRepository(conn)
    for bag_seq, session_id in (("B01", "sess-sd-1"), ("B02", "sess-sd-2")):
        sessions.upsert_session(
            {
                "rack_id": "R-EXAMPLE-03",
                "bed_no": "014",
                "infusion_order_no": "EXAMPLE-ORDER-0003",
                "shift_key": "2026-10-20-AM",
                "bag_seq": bag_seq,
                "slot_index": 0,
                "bag_type_code": "TYPE-A",
                "frames_dir": str(shared),
                "frame_count": 4,
                "captured_at": _iso(NOW - timedelta(days=45)),
                "state": "replaced",
                "calibration_state": "calibrated",
            }
        )
        conn.execute(
            "UPDATE bag_session SET session_id = ? WHERE bag_seq = ?", (session_id, bag_seq)
        )
        EventRepository(conn).append_event(
            {
                "client_event_id": f"evt-{session_id}",
                "session_id": session_id,
                "rack_id": "R-EXAMPLE-03",
                "bed_no": "014",
                "infusion_order_no": "EXAMPLE-ORDER-0003",
                "bag_seq": bag_seq,
                "operator_id": "N-EXAMPLE-7",
                "occurred_at": _iso(NOW - timedelta(days=45)),
                "writeback_status": "sent",
                "payload_json": "{}",
            }
        )

    result = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )

    assert len(result.purged) == 2, "两只袋子都记为已清理"
    assert result.directories_touched == 1, "一个目录只动一次"
    assert result.freed_bytes == 400, "字节数按目录算一次，不按袋数翻倍"
    assert not shared.exists()


# ── 四、审计链与推理审计 ──────────────────────────────────────────────────
def test_audit_chain_is_never_pruned(store):
    """审计链是飞检留档：任何保留期都不动它。"""
    conn, frames_root = store
    chain = AuditChain(conn)
    chain.append(
        request_id="req-1",
        actor="nurse:N-EXAMPLE-7",
        action="state.confirmed",
        target="sess-keep",
        outcome="accepted",
        occurred_at=_iso(NOW - timedelta(days=400)),
    )
    _session(
        conn, session_id="sess-keep", frames_root=frames_root, days_ago=90, shared_key="keep-dir"
    )

    run_retention(
        conn,
        frames_root,
        policy=RetentionPolicy(session_retention_days=30.0, inference_audit_retention_days=1.0),
        now=NOW,
    )

    assert conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 1


def test_expired_inference_audit_rows_are_pruned(store):
    """推理审计（模型调用元数据）按保留期裁掉，并记一行审计。"""
    conn, frames_root = store
    repo = SessionRepository(conn)
    conn.execute(
        "INSERT INTO inference_audit (request_id, model, prompt_hash, decision_source, created_at) "
        "VALUES ('req-old', 'm', 'h', 'threshold_only', ?)",
        (_iso(NOW - timedelta(days=200)),),
    )
    conn.execute(
        "INSERT INTO inference_audit (request_id, model, prompt_hash, decision_source, created_at) "
        "VALUES ('req-new', 'm', 'h', 'threshold_only', ?)",
        (_iso(NOW - timedelta(days=2)),),
    )

    result = run_retention(
        conn,
        frames_root,
        policy=RetentionPolicy(inference_audit_retention_days=90.0),
        audit=AuditChain(conn),
        request_id="retention-cron",
        now=NOW,
    )

    assert result.pruned_inference_audit == 1
    remaining = conn.execute("SELECT request_id FROM inference_audit").fetchall()
    assert [row["request_id"] for row in remaining] == ["req-new"]
    actions = [
        row["action"]
        for row in conn.execute("SELECT action FROM audit_log").fetchall()
    ]
    assert "audit.inference_pruned" in actions


def test_cleanup_writes_audit_rows(store):
    """归档/删除在审计链留痕（谁在什么时候因为什么清了哪袋的帧）。"""
    conn, frames_root = store
    _session(
        conn, session_id="sess-audit", frames_root=frames_root, days_ago=45, shared_key="audit-dir"
    )

    run_retention(
        conn,
        frames_root,
        policy=RetentionPolicy(session_retention_days=30.0),
        audit=AuditChain(conn),
        request_id="retention-cron",
        now=NOW,
    )

    row = conn.execute(
        "SELECT action, target, detail FROM audit_log WHERE action = 'session.frames_purged'"
    ).fetchone()
    assert row is not None
    assert row["target"] == "sess-audit"
    assert "retention_expired" in row["detail"]


# ── 五、边界 ──────────────────────────────────────────────────────────────
def test_unparsable_captured_at_is_kept(store):
    """captured_at 解析不了的会话：保留并报出来，不当成过期清掉。"""
    conn, frames_root = store
    item = _session(
        conn, session_id="sess-badts", frames_root=frames_root, days_ago=45, shared_key="badts-dir"
    )
    conn.execute(
        "UPDATE bag_session SET captured_at = ? WHERE session_id = ?",
        ("不是时间", item["session_id"]),
    )
    frames_dir = Path(item["frames_dir"])

    result = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )

    assert result.purged == []
    assert frames_dir.is_dir()
    assert result.blocked[0]["reason"] == "bad_timestamp"


def test_missing_frames_directory_is_recorded_as_absent(store):
    """目录本来就不在（老库没留帧）：记一笔 absent，别每次都重扫。"""
    conn, frames_root = store
    item = _session(
        conn, session_id="sess-absent", frames_root=frames_root, days_ago=45, shared_key="absent-dir"
    )
    shutil.rmtree(item["frames_dir"])

    result = run_retention(
        conn, frames_root, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )

    assert len(result.absent) == 1
    assert result.purged == []
    row = conn.execute(
        "SELECT frames_retention_state FROM bag_session WHERE session_id = ?",
        (item["session_id"],),
    ).fetchone()
    assert row["frames_retention_state"] == "absent"


def test_plan_retention_does_not_mutate(store):
    """plan 只算不动：会话状态、台账、文件都不该变。"""
    conn, frames_root = store
    _session(
        conn, session_id="sess-plan", frames_root=frames_root, days_ago=45, shared_key="plan-dir"
    )

    eligible, blocked, cutoff = plan_retention(
        conn, policy=RetentionPolicy(session_retention_days=30.0), now=NOW
    )

    assert len(eligible) == 1
    assert blocked == []
    assert cutoff == NOW - timedelta(days=30)
    row = conn.execute(
        "SELECT frames_retention_state FROM bag_session WHERE session_id = ?", ("sess-plan",)
    ).fetchone()
    assert row["frames_retention_state"] == "kept"
    assert conn.execute("SELECT COUNT(*) FROM retention_archive").fetchone()[0] == 0
