"""幂等键、并发重放与越权边界的验收测试。

六组断言，每组都盯住「事件表行数不变」这一个不变量：

1. **重复确认**：同一次确认发两次，第二次返回 ``idempotent_hit=true``，
   ``infusion_event`` 仍只有一行，且两次的 ``event_id`` 是同一个；
2. **重复上传**：同 ``rack+shift+bag_seq`` 重传走覆盖式更新（``session_version`` +1、
   会话行数仍为 1），帧目录**不追加**——含「先传长的再传短的」反例，专治陈旧帧；
3. **并发双确认**：两线程各持一条连接同时确认，恰好一路 created，另一路幂等命中或 409；
4. **写锁串行化**：一路持锁未提交时，另一路必须等——这是「不双写」的物理保证；
5. **越权边界**：缺 ``bag_seq`` → 422；袋序不属于该会话 → 403；跨输液架 → 403；
   三者都留 rejected 审计，且状态与事件表**纹丝不动**；
6. **非法字段写入**：仓储层的字段白名单挡住「顺手多写一列」这类写入。
"""

from __future__ import annotations

import threading

import pytest

from infusion_monitor.errors import ApiError
from infusion_monitor.identity.idempotency import (
    authorize_session_write,
    confirm_replacement_once,
)
from infusion_monitor.identity.state_machine import build_client_event_id
from infusion_monitor.store import (
    AuditChain,
    EventRepository,
    SessionRepository,
    apply_migrations,
    connect,
)
from tests.support import build_rack_zip, default_rack_scene

TRACKS_URL = "/api/v1/infusion/tracks"

TWO_BAG_SLOTS = (
    '[{"bag_seq":"B01","slot_index":0,"bag_type_code":"TYPE-A","rate_text":"5ml/h"},'
    '{"bag_seq":"B02","slot_index":1,"bag_type_code":"TYPE-A","rate_text":"5ml/h"}]'
)

OPERATOR = "N-EXAMPLE-10"


def _upload_round(client, scene, *, shift_key: str, captured_at: str, slots: str, **fields):
    """上传一段短片。拍摄时刻显式给出——重锚按它排序，不按服务端写入时刻。"""
    payload = {
        "rack_id": "R-EXAMPLE-10",
        "infusion_order_no": "EXAMPLE-ORDER-0010",
        "bed_no": "EXAMPLE-BED-10",
        "captured_at": captured_at,
        "shift_key": shift_key,
        "slots": slots,
        **fields,
    }
    return client.post(
        TRACKS_URL,
        data=payload,
        files={"frames": ("frames.zip", build_rack_zip(scene), "application/zip")},
    )


def _seed_named_session(conn, **overrides) -> dict:
    """一袋已点名、需要换液的会话（needs_change 是 replaced 的唯一合法来源）。"""
    payload = {
        "rack_id": "R-EXAMPLE-10",
        "bed_no": "EXAMPLE-BED-10",
        "infusion_order_no": "EXAMPLE-ORDER-0010",
        "shift_key": "2026-10-22-AM",
        "bag_seq": "B01",
        "slot_index": 0,
        "bag_type_code": "TYPE-A",
        "frames_dir": "/tmp/frames",
        "frame_count": 16,
        "captured_at": "2026-10-22T09:41:00+08:00",
        "state": "monitoring",
    }
    payload.update(overrides)
    sessions = SessionRepository(conn)
    row = sessions.upsert_session(payload)
    sessions.apply_state_transition(row["session_id"], "needs_change")
    return sessions.get(row["session_id"])


def _repos(conn) -> tuple[SessionRepository, EventRepository, AuditChain]:
    return SessionRepository(conn), EventRepository(conn), AuditChain(conn)


def _event_count(conn) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM infusion_event").fetchone()["n"])


# ── 1. 重复确认 ───────────────────────────────────────────────────────────


def test_repeated_confirmation_writes_one_event(conn):
    """同一次确认发两次：第二次幂等命中，事件表仍一行，event_id 不变。"""
    sessions, events, audit = _repos(conn)
    record = _seed_named_session(conn)

    first = confirm_replacement_once(
        sessions, events, record["session_id"],
        bag_seq="B01", operator_id=OPERATOR, audit=audit, request_id="req-10-a",
        now="2026-10-22T11:05:00+08:00",
    )
    assert first.created is True
    assert first.idempotent_hit is False
    assert first.state == "replaced"
    assert _event_count(conn) == 1

    second = confirm_replacement_once(
        sessions, events, record["session_id"],
        bag_seq="B01", operator_id=OPERATOR, audit=audit, request_id="req-10-b",
        now="2026-10-22T11:05:30+08:00",   # 重放发生在 30 秒之后，键仍然必须相同
    )
    assert second.idempotent_hit is True
    assert second.created is False
    assert _event_count(conn) == 1, "重复确认多写了一条出站事件"
    assert second.event_id == first.event_id
    # 响应形状就是 task 31 的 HTTP 端点要回的那一份
    assert second.to_json()["idempotent_hit"] is True
    assert second.to_json()["event_id"] == first.event_id
    # 重放**不改**原事件的时刻：事件记的是事实发生的时刻，不是请求到达的时刻
    assert second.occurred_at == first.occurred_at == "2026-10-22T11:05:00+08:00"
    assert sessions.get(record["session_id"])["state"] == "replaced"


def test_idempotency_key_ignores_the_confirmation_timestamp(conn):
    """幂等键只由 (会话, 袋序) 决定：确认时刻变了，键不能变。"""
    sessions, events, audit = _repos(conn)
    record = _seed_named_session(conn)
    confirm_replacement_once(
        sessions, events, record["session_id"],
        bag_seq="B01", operator_id=OPERATOR, now="2026-10-22T11:05:00+08:00",
    )
    # 库里唯一的键 == 按规则算出来的键
    stored = events.find_by_client_event_id(build_client_event_id(record["session_id"], "B01"))
    assert stored is not None
    assert stored["client_event_id"] == f"infusion-{record['session_id']}-B01-replaced"
    # 换袋后对端给新的 bag_seq → 自然是新键，不会把两次换液并成一条
    assert build_client_event_id(record["session_id"], "B02") != stored["client_event_id"]


def test_replay_is_audited_without_adding_an_event(conn):
    """重放要留痕（护士点了两次是事实），但只留审计行、不留第二条事件。"""
    sessions, events, audit = _repos(conn)
    record = _seed_named_session(conn)
    confirm_replacement_once(
        sessions, events, record["session_id"],
        bag_seq="B01", operator_id=OPERATOR, audit=audit, request_id="req-10-a",
    )
    confirm_replacement_once(
        sessions, events, record["session_id"],
        bag_seq="B01", operator_id=OPERATOR, audit=audit, request_id="req-10-b",
    )

    counts = audit.count_by_action()
    assert counts["state.confirmed"] == 1, "确认动作被记了两次"
    assert counts["event.recorded"] == 1
    assert counts["event.replay_ignored"] == 1
    assert _event_count(conn) == 1

    replay = [r for r in audit.list_by_target(record["session_id"])
              if r["action"] == "event.replay_ignored"][0]
    assert replay["request_id"] == "req-10-b", "重放要能按 request_id 追到是哪一次点击"
    assert replay["outcome"] == "accepted"
    assert replay["detail"]["idempotent_hit"] is True
    assert replay["detail"]["note"]


# ── 2. 重复上传 ───────────────────────────────────────────────────────────


def test_repeated_upload_overwrites_instead_of_appending(store_client):
    """同 rack+shift+bag_seq 重传：会话行数不变、版本 +1、帧目录不追加。"""
    client, conn = store_client
    upload = _upload_round(client, default_rack_scene(20), shift_key="2026-10-22-AM",
                           captured_at="2026-10-22T09:41:00+08:00", slots=TWO_BAG_SLOTS)
    assert upload.status_code == 201, upload.text
    bag = upload.json()["bags"][0]

    sessions = SessionRepository(conn)
    first = sessions.get(bag["session_id"])
    frame_dir = first["frames_dir"]
    from pathlib import Path

    assert len(list(Path(frame_dir).glob("frame_*.jpg"))) == 20

    again = _upload_round(client, default_rack_scene(20), shift_key="2026-10-22-AM",
                          captured_at="2026-10-22T09:41:00+08:00", slots=TWO_BAG_SLOTS)
    assert again.status_code == 201, again.text
    assert again.json()["bags"][0]["session_id"] == bag["session_id"], "重传建了新会话"

    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM bag_session WHERE rack_id = ? AND shift_key = ? AND bag_seq = ?",
        (first["rack_id"], first["shift_key"], "B01"),
    ).fetchone()["n"]
    assert rows == 1, "重传把会话拆成了两行"
    second = sessions.get(bag["session_id"])
    assert second["session_version"] == first["session_version"] + 1
    assert len(list(Path(frame_dir).glob("frame_*.jpg"))) == 20, "帧目录被追加了"


def test_shorter_reupload_does_not_leave_stale_frames(store_client):
    """先传 20 帧再传 12 帧：目录里只剩本次的 12 帧，不残留上一次的 8 帧。

    这是覆盖式上传最容易漏的一角——只写新帧不删旧帧，磁盘上会留下一段「本轮不存在
    的长视频」，之后重锚、复算、取证读到的都是错的东西。
    """
    from pathlib import Path

    client, conn = store_client
    long_round = _upload_round(client, default_rack_scene(20), shift_key="2026-10-22-AM",
                               captured_at="2026-10-22T09:41:00+08:00", slots=TWO_BAG_SLOTS)
    bag = long_round.json()["bags"][0]
    frame_dir = Path(SessionRepository(conn).get(bag["session_id"])["frames_dir"])

    short_round = _upload_round(client, default_rack_scene(12), shift_key="2026-10-22-AM",
                                captured_at="2026-10-22T09:41:00+08:00", slots=TWO_BAG_SLOTS)
    assert short_round.status_code == 201
    stored = SessionRepository(conn).get(bag["session_id"])
    assert stored["frame_count"] == 12
    names = sorted(p.name for p in frame_dir.glob("frame_*.jpg"))
    assert len(names) == 12, f"上一轮的帧没清干净：{len(names)} 个"
    assert names[-1] == "frame_00011.jpg", "残留了上一轮的第 13–20 帧"


def test_reupload_does_not_duplicate_outbound_events(store_client):
    """重传一整段短片不产生出站事件（入站重放与出站幂等是两条独立的路）。"""
    client, conn = store_client
    upload = _upload_round(client, default_rack_scene(16), shift_key="2026-10-22-AM",
                           captured_at="2026-10-22T09:41:00+08:00", slots=TWO_BAG_SLOTS)
    assert upload.status_code == 201
    _upload_round(client, default_rack_scene(16), shift_key="2026-10-22-AM",
                  captured_at="2026-10-22T09:41:00+08:00", slots=TWO_BAG_SLOTS)
    assert _event_count(conn) == 0


# ── 3. 并发双确认 ─────────────────────────────────────────────────────────


def test_concurrent_double_confirmation_writes_one_event(tmp_path, store_client):
    """两线程各持一条连接、同一屏障起跑地确认同一袋：事件表恒为一行。"""
    client, conn = store_client
    upload = _upload_round(client, default_rack_scene(16), shift_key="2026-10-22-AM",
                           captured_at="2026-10-22T09:41:00+08:00", slots=TWO_BAG_SLOTS)
    session_id = upload.json()["bags"][0]["session_id"]
    # 确认的唯一合法来源是被点名（needs_change）：先让服务算一次点名
    SessionRepository(conn).apply_state_transition(session_id, "needs_change")
    dsn = f"sqlite:///{tmp_path / 'store.db'}"

    barrier = threading.Barrier(2)
    outcomes: list[object] = []
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        worker_conn = connect(dsn)
        try:
            sessions = SessionRepository(worker_conn)
            events = EventRepository(worker_conn)
            barrier.wait(timeout=30)
            try:
                outcomes.append(
                    confirm_replacement_once(
                        sessions, events, session_id,
                        bag_seq="B01", operator_id=f"{OPERATOR}-{index}",
                        audit=AuditChain(worker_conn),
                        request_id=f"req-concurrent-{index}",
                    )
                )
            except ApiError as exc:
                # 契约允许的另一种落点：409（幂等键命中与状态检查之间的极窄窗口）
                errors.append(exc)
        finally:
            worker_conn.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive(), "并发确认线程卡住了（写锁没释放？）"

    assert _event_count(conn) == 1, f"并发双确认写了 {_event_count(conn)} 条事件"
    assert len(outcomes) + len(errors) == 2, "有一路既没成功也没按契约报 409"
    for exc in errors:
        assert exc.status_code == 409 and exc.error_code == "STATE_CONFLICT"

    # 恰好一路真正落库；命中的一路 idempotent_hit 为 True
    created = [o for o in outcomes if getattr(o, "created", False)]
    hits = [o for o in outcomes if getattr(o, "idempotent_hit", False)]
    assert len(created) == 1
    assert len(created) + len(hits) + len(errors) == 2
    # 确认审计只有一条：不管几路来请求，事实只发生过一次
    assert AuditChain(conn).count_by_action()["state.confirmed"] == 1
    assert SessionRepository(conn).get(session_id)["state"] == "replaced"


def test_concurrent_confirmations_stay_single_event_across_rounds(tmp_path, store_client):
    """连打 5 轮并发双确认：每一轮的事件表都恰好一行，不随轮数增长。"""
    client, conn = store_client
    dsn = f"sqlite:///{tmp_path / 'store.db'}"
    sessions = SessionRepository(conn)

    for round_index in range(5):
        shift = f"2026-10-22-R{round_index}"
        captured_at = f"2026-10-22T{round_index:02d}:41:00+08:00"
        upload = _upload_round(client, default_rack_scene(16), shift_key=shift,
                               captured_at=captured_at, slots=TWO_BAG_SLOTS)
        session_id = upload.json()["bags"][0]["session_id"]
        sessions.apply_state_transition(session_id, "needs_change")
        before = _event_count(conn)

        barrier = threading.Barrier(2)
        lock = threading.Lock()

        def worker(index: int, session_id=session_id, barrier=barrier, lock=lock) -> None:
            worker_conn = connect(dsn)
            try:
                try:
                    confirm_replacement_once(
                        SessionRepository(worker_conn), EventRepository(worker_conn),
                        session_id, bag_seq="B01", operator_id=f"{OPERATOR}-{index}",
                    )
                except ApiError as exc:  # 409 也是契约内的落点
                    assert exc.status_code == 409
            finally:
                worker_conn.close()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive()

        assert _event_count(conn) == before + 1, (
            f"第 {round_index} 轮并发确认后事件数不是 +1（{_event_count(conn)} vs {before}）"
        )
        assert sessions.get(session_id)["state"] == "replaced"


def test_write_lock_serialises_two_connections(tmp_path):
    """物理保证：一路持 ``BEGIN IMMEDIATE`` 未提交时，另一路写不进去。

    前一组用例断言的是「结果对」；这组断言的是「机制真的挡住了」。若哪天有人把
    ``BEGIN IMMEDIATE`` 换成延迟事务，上面的用例仍可能侥幸通过，这组不会。

    SQLite 连接带线程亲和性，所以两条连接都**在自己的线程里**打开；写者线程先
    把连接建好（``ready``）再等放行，避免把 ``connect()`` 里的 PRAGMA 也算进
    「被锁挡住」的时间里。
    """
    dsn = f"sqlite:///{tmp_path / 'lock.db'}"
    seed = connect(dsn)
    apply_migrations(seed)
    seed.close()

    ready = threading.Event()      # 写者连接已建立
    go = threading.Event()         # 放行写者去写
    acquired = threading.Event()   # 持锁线程已拿到写锁
    released = threading.Event()   # 允许持锁线程提交
    finished = threading.Event()   # 写者写完

    def _insert(request_id: str, action: str, digest: str, at: str) -> None:
        return (
            "INSERT INTO audit_log (request_id, actor, action, target, outcome, detail, "
            f"payload_hash, occurred_at) VALUES ('{request_id}', 'service', '{action}', 's', "
            f"'accepted', '{{}}', '{digest}', '{at}')"
        )

    def holder() -> None:
        conn = connect(dsn)
        try:
            # 手动开事务并**不提交**：模拟一路正在写、还没写完
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(_insert("r2", "state.confirmed", "h2", "2026-10-22T11:05:01+08:00"))
            acquired.set()
            released.wait(timeout=30)
            conn.execute("COMMIT")
        finally:
            conn.close()

    def writer() -> None:
        conn = connect(dsn)
        try:
            ready.set()
            go.wait(timeout=30)
            conn.execute(_insert("r3", "event.recorded", "h3", "2026-10-22T11:05:02+08:00"))
        finally:
            finished.set()
            conn.close()

    other = threading.Thread(target=writer)
    other.start()
    assert ready.wait(timeout=30), "写者连接没建起来"

    thread = threading.Thread(target=holder)
    thread.start()
    assert acquired.wait(timeout=30), "持锁线程没能拿到写锁"

    go.set()
    # 持锁线程还没放锁 → 另一路此刻必须还写不进去
    assert not finished.wait(timeout=1.5), "另一路在写锁未释放时就写进去了（没有串行化）"
    released.set()
    thread.join(timeout=30)
    assert finished.wait(timeout=30), "放锁后另一路仍然写不进去"
    other.join(timeout=30)

    checker = connect(dsn)
    try:
        rows = checker.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE action IN "
            "('state.confirmed', 'event.recorded')"
        ).fetchone()["n"]
        assert rows == 2, "放锁后两行都在（各自提交，没有互相吞掉）"
    finally:
        checker.close()


# ── 4. 越权边界 ───────────────────────────────────────────────────────────


def test_missing_bag_seq_is_rejected_with_an_audit_row(conn):
    """不点名袋序的确认不接受：422 + rejected 审计，状态与事件表都不动。"""
    sessions, events, audit = _repos(conn)
    record = _seed_named_session(conn)

    with pytest.raises(ApiError) as excinfo:
        confirm_replacement_once(
            sessions, events, record["session_id"],
            bag_seq="", operator_id=OPERATOR, audit=audit, request_id="req-10-no-bag",
        )
    assert excinfo.value.status_code == 422
    assert excinfo.value.error_code == "INGEST_INVALID"
    assert excinfo.value.issues[0].code == "bag_seq_missing"
    assert excinfo.value.issues[0].field == "bag_seq"
    assert excinfo.value.issues[0].suggestion

    assert sessions.get(record["session_id"])["state"] == "needs_change"
    assert _event_count(conn) == 0
    rejected = [r for r in audit.list_by_target(record["session_id"])
                if r["outcome"] == "rejected"]
    assert len(rejected) == 1
    assert rejected[0]["action"] == "session.write_rejected"
    assert rejected[0]["detail"]["code"] == "bag_seq_missing"


def test_writing_another_bags_replacement_is_forbidden(conn):
    """给 B01 的会话写 B02 的换液事件 → 403 + rejected 审计，两边都不变。"""
    sessions, events, audit = _repos(conn)
    record = _seed_named_session(conn)
    sessions.upsert_session(
        {
            "rack_id": "R-EXAMPLE-10", "bed_no": "EXAMPLE-BED-10",
            "infusion_order_no": "EXAMPLE-ORDER-0010", "shift_key": "2026-10-22-AM",
            "bag_seq": "B02", "slot_index": 1, "frames_dir": "/tmp/frames",
            "frame_count": 16, "captured_at": "2026-10-22T09:41:00+08:00",
        }
    )

    with pytest.raises(ApiError) as excinfo:
        confirm_replacement_once(
            sessions, events, record["session_id"],
            bag_seq="B02", operator_id=OPERATOR, audit=audit, request_id="req-10-cross",
        )
    assert excinfo.value.status_code == 403
    assert excinfo.value.error_code == "WRITE_NOT_ALLOWED"
    issue = excinfo.value.issues[0]
    assert issue.code == "bag_not_in_session"
    assert "B01" in issue.reason and "B02" in issue.reason
    assert issue.suggestion

    assert sessions.get(record["session_id"])["state"] == "needs_change"
    assert _event_count(conn) == 0, "越权写入留下了事件行"
    rejected = [r for r in audit.list_by_target(record["session_id"])
                if r["outcome"] == "rejected"]
    assert rejected[0]["detail"]["submitted_bag_seq"] == "B02"
    assert rejected[0]["detail"]["session_bag_seq"] == "B01"


def test_cross_rack_confirmation_is_forbidden(conn):
    """跨输液架确认（病区终端误操作到隔壁架）同样拒绝。"""
    sessions, events, audit = _repos(conn)
    record = _seed_named_session(conn)

    with pytest.raises(ApiError) as excinfo:
        confirm_replacement_once(
            sessions, events, record["session_id"],
            bag_seq="B01", rack_id="R-EXAMPLE-99", operator_id=OPERATOR,
            audit=audit, request_id="req-10-cross-rack",
        )
    assert excinfo.value.status_code == 403
    assert excinfo.value.issues[0].code == "rack_not_in_session"
    assert _event_count(conn) == 0
    assert sessions.get(record["session_id"])["state"] == "needs_change"


def test_authorize_accepts_a_matching_bag(conn):
    """袋序对得上就放行——这道闸只拦跨袋写入，不拦正常确认。"""
    record = _seed_named_session(conn)
    authorize_session_write(record, bag_seq="B01", operator_id=OPERATOR)
    authorize_session_write(
        record, bag_seq="B01", rack_id="R-EXAMPLE-10", operator_id=OPERATOR
    )


def test_illegal_transition_is_rejected_before_any_write(conn):
    """monitoring 直跳 replaced：409 + rejected 审计，事件表不新增。"""
    sessions, events, audit = _repos(conn)
    sessions.upsert_session(
        {
            "rack_id": "R-EXAMPLE-10", "bed_no": "EXAMPLE-BED-10",
            "infusion_order_no": "EXAMPLE-ORDER-0010", "shift_key": "2026-10-22-AM",
            "bag_seq": "B01", "slot_index": 0, "frames_dir": "/tmp/frames",
            "frame_count": 16, "captured_at": "2026-10-22T09:41:00+08:00",
        }
    )
    record = sessions.find_by_rack_shift_bag(
        "R-EXAMPLE-10", "2026-10-22-AM", "B01"
    )
    assert record["state"] == "monitoring"

    with pytest.raises(ApiError) as excinfo:
        confirm_replacement_once(
            sessions, events, record["session_id"],
            bag_seq="B01", operator_id=OPERATOR, audit=audit, request_id="req-10-skip",
        )
    assert excinfo.value.status_code == 409
    assert excinfo.value.error_code == "STATE_CONFLICT"
    assert excinfo.value.issues[0].code == "illegal_transition"
    assert _event_count(conn) == 0
    rejected = [r for r in audit.list_by_target(record["session_id"])
                if r["outcome"] == "rejected"]
    assert rejected[0]["action"] == "state.transition_rejected"


def test_replaced_session_cannot_be_confirmed_under_another_bag_seq(conn):
    """已换袋终态：换个袋序点不回来；同一袋重放则是幂等命中而不是 409。

    两条路的**性质不同**，不能混：换个袋序是「对另一袋动手」，要硬拒（403）；
    同一袋重放是「同一次确认又到了一遍」，应当悄无声息地返回同一个结果——
    护士点了两次不该弹红框说换液失败，否则他多半会再点第三次。
    """
    sessions, events, audit = _repos(conn)
    record = _seed_named_session(conn)
    first = confirm_replacement_once(
        sessions, events, record["session_id"],
        bag_seq="B01", operator_id=OPERATOR, audit=audit, request_id="req-10-first",
    )
    assert first.created is True

    with pytest.raises(ApiError) as excinfo:
        confirm_replacement_once(
            sessions, events, record["session_id"],
            bag_seq="B02", operator_id=OPERATOR, audit=audit, request_id="req-10-other-bag",
        )
    assert excinfo.value.status_code == 403
    assert excinfo.value.issues[0].code == "bag_not_in_session"

    replay = confirm_replacement_once(
        sessions, events, record["session_id"],
        bag_seq="B01", operator_id=OPERATOR, audit=audit, request_id="req-10-same-bag",
    )
    assert replay.idempotent_hit is True
    assert replay.state == "replaced"
    assert _event_count(conn) == 1


# ── 5. 非法字段写入 ───────────────────────────────────────────────────────


def test_repository_rejects_fields_outside_the_whitelist(conn):
    """仓储层挡住「顺手多写一列」：白名单外的字段直接报错，不进库。"""
    sessions = SessionRepository(conn)
    record = _seed_named_session(conn)

    with pytest.raises(ValueError, match="不可写字段"):
        sessions.apply_state_transition(
            record["session_id"], "needs_change", estimated_volume_ml=999.0
        )
    with pytest.raises(ValueError, match="不可写字段"):
        sessions.update_state(record["session_id"], "replaced", state_of_the_world="x")
    # 被拒的写入没进库：估计毫升数还是确认前那个值
    assert sessions.get(record["session_id"])["estimated_volume_ml"] is None


def test_confirm_without_operator_is_rejected(conn):
    """没人确认就没有换液事件（即使袋序完全合法）。"""
    sessions, events, audit = _repos(conn)
    record = _seed_named_session(conn)
    with pytest.raises(ApiError) as excinfo:
        confirm_replacement_once(
            sessions, events, record["session_id"], bag_seq="B01", operator_id=""
        )
    assert excinfo.value.status_code == 422
    assert excinfo.value.issues[0].field == "operator_id"
    assert _event_count(conn) == 0


def test_unknown_session_is_not_found(conn):
    sessions, events, _ = _repos(conn)
    with pytest.raises(ApiError) as excinfo:
        confirm_replacement_once(
            sessions, events, "sess_does_not_exist", bag_seq="B01", operator_id=OPERATOR
        )
    assert excinfo.value.status_code == 404


# ── 6. 事务一致性 ─────────────────────────────────────────────────────────


def test_state_and_event_land_in_the_same_transaction(conn):
    """状态推进与事件落库同生共死：事务里失败不会留下「状态变了但没事件」。"""
    sessions, events, audit = _repos(conn)
    record = _seed_named_session(conn)

    class _Boom(RuntimeError):
        pass

    original_append = audit.append
    original_insert = audit.insert
    calls = {"n": 0}

    def exploding(*args, **kwargs):
        # 在「确认」审计之后、「事件落库」之前炸掉——这是最容易留下半写的位置。
        # 两个入口都要拦：``state.confirmed`` 走 insert（与状态推进同事务，task 33），
        # ``event.recorded`` 走 append。只拦一个的话，这条用例其实没在测它想测的东西。
        calls["n"] += 1
        if calls["n"] == 2:
            raise _Boom("审计写入中断")
        return original_append(*args, **kwargs) if not args else original_insert(*args, **kwargs)

    audit.append = exploding  # type: ignore[method-assign]
    audit.insert = exploding  # type: ignore[method-assign]
    try:
        with pytest.raises(_Boom):
            confirm_replacement_once(
                sessions, events, record["session_id"],
                bag_seq="B01", operator_id=OPERATOR, audit=audit, request_id="req-10-boom",
            )
    finally:
        audit.append = original_append  # type: ignore[method-assign]
        audit.insert = original_insert  # type: ignore[method-assign]

    assert sessions.get(record["session_id"])["state"] == "needs_change", "半写了状态"
    assert _event_count(conn) == 0, "半写了事件"
    # 重来一次应当正常成功（半成品没有留下任何锁或脏数据）
    outcome = confirm_replacement_once(
        sessions, events, record["session_id"],
        bag_seq="B01", operator_id=OPERATOR, audit=audit, request_id="req-10-retry",
    )
    assert outcome.created is True
    assert _event_count(conn) == 1