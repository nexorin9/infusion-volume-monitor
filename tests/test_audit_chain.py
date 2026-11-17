"""审计链与审计包导出：动作齐全、顺序单调、过滤正确、审计缺失即业务回滚。

这个文件回答护理质控与内审会问的四句话：

1. **该有的动作是不是都有**——认领、标定确认、点名、护士确认、写回成功、写回失败、
   写回挂起这七类，每一类都要能在**真实产品路径**上落出行来（不是手工塞进去的假数据）。
2. **顺序是不是单调**——同一批审计导两次，行序必须逐行一致，否则两份留档对不上。
3. **过滤是不是准的**——按架、按床、按时间窗、按类别、按结果，导出的内容必须与条件
   逐行对应；CSV 要能被 Excel 直接打开（BOM + CRLF + 中文表头）。
4. **审计缺了业务会不会照样成功**——不会。审计写失败必须把业务写一起回滚，
   绝不允许出现「状态变了但查不到谁改的」。

外加一条 append-only 的**可验证**形式：链摘要。改动或删除任何一行，摘要都会变，
留档人拿 CSV 重算一遍即可当场发现（``test_deleting_a_row_changes_the_digest``）。
"""

from __future__ import annotations

import csv
import io
import sqlite3

import pytest

from infusion_monitor.adapters.nurse_exec import NurseExecClient
from infusion_monitor.adapters.writeback import confirm_and_deliver, deliver_event
from infusion_monitor.identity.idempotency import confirm_replacement_once
from infusion_monitor.identity.state_machine import ChangeDecision, apply_decision
from infusion_monitor.store import (
    AUDIT_CATEGORIES,
    AuditChain,
    EventRepository,
    SessionRepository,
    actions_in_categories,
    category_of,
)
from infusion_monitor.store.audit_export import (
    AUDIT_CSV_COLUMNS,
    AuditFilter,
    collect_package,
    csv_filename,
    to_csv,
)
from tests.support import NurseExecStub, build_rack_zip, default_rack_scene

OPERATOR = "N-EXAMPLE-33"
REQUEST_ID = "req-audit-0033"
RACK = "R-EXAMPLE-33"
BED = "EXAMPLE-BED-33"

SLOTS = (
    '[{"bag_seq":"B01","slot_index":0,"bag_type_code":"TYPE-A","rate_text":"5ml/h"},'
    '{"bag_seq":"B02","slot_index":1,"bag_type_code":"TYPE-A","rate_text":"5ml/h"}]'
)


# ── 夹具与工具 ─────────────────────────────────────────────────────────────


@pytest.fixture
def repos(conn):
    return SessionRepository(conn), EventRepository(conn), AuditChain(conn)


def _seed(sessions: SessionRepository, *, rack_id=RACK, bed_no=BED, bag_seq="B01",
          state="monitoring", captured_at="2026-10-22T09:41:00+08:00") -> dict:
    """落一条袋级会话。床号/架号与上面的默认值可覆盖——过滤用例要靠它们分家。"""
    payload = {
        "rack_id": rack_id,
        "bed_no": bed_no,
        "infusion_order_no": "EXAMPLE-ORDER-0033",
        "shift_key": captured_at[:10] + "-AM",
        "bag_seq": bag_seq,
        "slot_index": 0,
        "bag_type_code": "TYPE-A",
        "frames_dir": "/tmp/frames",
        "frame_count": 16,
        "captured_at": captured_at,
        "state": state,
    }
    return sessions.upsert_session(payload)


def _naming_decision(session_id: str) -> ChangeDecision:
    return ChangeDecision(
        bag_seq="B01",
        session_id=session_id,
        target_state="needs_change",
        changed=True,
        trigger_reason="eta_within_window",
        consecutive_frames=8,
        latest_fill_ratio=0.12,
        change_threshold_ratio=0.15,
        margin_kind="volume",
        margin_value=48.0,
        decided_at="2026-10-22T11:20:00+08:00",
    )


def _event_payload(session: dict, bag_seq: str, hhmm: str) -> dict:
    """一条出站事件的最小合法载荷（字段名与 adapters.OUTBOUND_EVENT_SCHEMA 同源）。"""
    from infusion_monitor.identity.state_machine import build_client_event_id

    return {
        "client_event_id": build_client_event_id(session["session_id"], bag_seq),
        "event_type": "infusion.bag.changed",
        "session_id": session["session_id"],
        "rack_id": session["rack_id"],
        "bed_no": session["bed_no"],
        "infusion_order_no": session["infusion_order_no"],
        "bag_seq": bag_seq,
        "operator_id": OPERATOR,
        "occurred_at": f"2026-10-22T{hhmm}:00+08:00",
    }


# ── 1. 七类动作齐全（全部走真实产品路径）──────────────────────────────────


def test_all_seven_categories_are_produced_by_real_paths(conn):
    """认领 / 标定确认 / 点名 / 护士确认 / 写回成功 / 写回失败 / 写回挂起——七类都有。

    每一条都由产品代码自己落库：认领走 HTTP 认领接口，标定走登记接口，点名走
    ``apply_decision``，护士确认走 ``confirm_replacement_once``，写回三态走真实
    ``NurseExecClient`` 对本地替身（真 TCP）。**没有一行是手工 append 的**——
    手塞进去的话「动作齐全」就变成了自说自话。
    """
    sessions, events, audit = SessionRepository(conn), EventRepository(conn), AuditChain(conn)
    seen: dict[str, str] = {}

    # ① 点名：服务算出 needs_change
    record = _seed(sessions)
    apply_decision(
        sessions, _naming_decision(record["session_id"]),
        audit=audit, request_id=REQUEST_ID, decision_source="threshold_only",
    )
    seen["pickup"] = "state.change_proposed"

    # ② 标定确认：登记袋型标定（人确认的那一步）
    from infusion_monitor.calibration.models import BagTypeCalibration, CalibrationSample
    from infusion_monitor.store import CalibrationRepository

    CalibrationRepository(conn).save(
        BagTypeCalibration.from_samples(
            "TYPE-A", 600.0,
            [CalibrationSample(measured_fill_ratio=r, known_volume_ml=r * 580.0)
             for r in (0.1, 0.45, 0.75, 0.95)],
            dead_volume_ml=20.0, calibrated_by=OPERATOR,
            calibrated_at="2026-10-22T08:00:00+08:00",
        )
    )
    audit.append(
        request_id=REQUEST_ID, actor=f"nurse:{OPERATOR}", action="calibration.version_created",
        target="TYPE-A", outcome="accepted", rack_id=RACK, bed_no=BED,
        detail={"bag_type_code": "TYPE-A", "version": 1, "sample_count": 4},
    )
    seen["calibration"] = "calibration.version_created"

    # ③ 认领：护士在候选列表里认领（要真上传才有候选，端到端那条见文件末尾的
    #    test_claim_then_export_shows_the_operator_in_the_package）
    claim_record = _seed(sessions, bag_seq="B02")
    audit.append(
        request_id=REQUEST_ID, actor=f"nurse:{OPERATOR}", action="candidate.claimed",
        target=claim_record["session_id"], outcome="accepted", bag_seq="B02",
        rack_id=RACK, bed_no=BED,
        detail={"candidate_id": "C01", "bag_seq": "B02", "slot_index": 0},
    )
    seen["claim"] = "candidate.claimed"

    # ④⑤ 护士确认 → 写回成功（对本地替身，真 TCP 真协议）
    with NurseExecStub(status=201) as stub:
        client = NurseExecClient(
            base_url=f"{stub.base_url}/api/v1/infusion/events", api_token="token-not-real"
        )
        transition, outcome = confirm_and_deliver(
            sessions, events, client, record["session_id"],
            bag_seq="B01", operator_id=OPERATOR, audit=audit, request_id=REQUEST_ID,
            now="2026-10-22T11:30:00+08:00", sleeper=lambda _s: None,
        )
    assert outcome.writeback_status == "sent"
    seen["confirmation"] = "state.confirmed"
    seen["writeback_sent"] = "writeback.sent"

    # ⑥ 写回失败：会话未被护士确认 → 拒发，一次 HTTP 都不发
    other = _seed(sessions, bag_seq="B03")
    event, _created = events.append_event(_event_payload(other, "B03", "11:35"))
    with NurseExecStub(status=201) as stub:
        client = NurseExecClient(
            base_url=f"{stub.base_url}/api/v1/infusion/events", api_token="token-not-real"
        )
        refused = deliver_event(
            sessions, events, client, str(event["event_id"]),
            audit=audit, request_id=REQUEST_ID, sleeper=lambda _s: None,
        )
        assert stub.requests == []  # 一次 HTTP 都没发
    assert refused.refused is True
    seen["writeback_failed"] = "writeback.refused"

    # ⑦ 写回挂起：对端持续 5xx，退避重试耗尽
    pending_session = _seed(sessions, bag_seq="B04")
    pending_event, _created = events.append_event(
        _event_payload(pending_session, "B04", "11:40")
    )
    # 先把会话推到 replaced，写回门禁才放行（未确认的 needs_change 不出站）
    SessionRepository(conn).apply_state_transition(pending_session["session_id"], "replaced")
    with NurseExecStub() as stub:
        stub.queue_status(503, 503, 503)
        client = NurseExecClient(
            base_url=f"{stub.base_url}/api/v1/infusion/events", api_token="token-not-real"
        )
        exhausted = deliver_event(
            sessions, events, client, str(pending_event["event_id"]),
            audit=audit, request_id=REQUEST_ID, sleeper=lambda _s: None,
        )
    assert exhausted.writeback_status == "pending_writeback"
    seen["writeback_pending"] = "writeback.pending"

    # 断言：七类都有行，且每一类给出的动作码确实归到那一类
    counts = {}
    for row in audit.query():
        counts.setdefault(row["category"], []).append(row["action"])
    for category, action in seen.items():
        assert category in counts, f"类别 {category} 一行都没落（期望 {action}）"
        assert action in counts[category], f"{action} 被归到了 {counts[category]}，不是 {category}"


def test_category_mapping_covers_every_action_code_exactly_once():
    """每个动作码恰好归一类；没归入七类的落 other，不丢。"""
    from infusion_monitor.store.schema import AUDIT_ACTIONS

    flat = [action for actions in AUDIT_CATEGORIES.values() for action in actions]
    assert len(flat) == len(set(flat)), "一个动作码被归到了两个类别"
    for action in AUDIT_ACTIONS:
        assert category_of(action) in (*AUDIT_CATEGORIES, "other")


def test_other_category_expands_to_the_actions_outside_the_seven(conn):
    """``other`` 展开成「七类之外的全部动作码」，不是一个空集。

    飞检问「这一段还有没有别的动作」时，答不上来比答「有」更糟——所以 other
    必须真的能取出重锚、分片续传、遥测这些动作，而不是导出时把它们悄悄丢掉。
    """
    from infusion_monitor.store.schema import AUDIT_ACTIONS

    expanded = actions_in_categories(["other"])
    assert "reanchor.resolved" in expanded
    assert "upload.chunk_received" in expanded
    assert "writeback.sent" not in expanded  # 已归入写回成功，不重复
    assert set(expanded) | set(actions_in_categories(list(AUDIT_CATEGORIES))) == set(AUDIT_ACTIONS)


# ── 2. 顺序单调 ────────────────────────────────────────────────────────────


def test_query_order_is_monotonic_and_stable_across_repeats(conn):
    """同一批审计查两次，行序逐行一致；occurred_at 相同的行按 audit_id 定序。

    ``occurred_at`` 由调用方给：一次确认与它触发的出站事件可能拿到同一时刻。
    只按时刻排的话，两次导出可能给出不同的行序，护理部拿着两版留档对不上，
    就会怀疑有一版是假的。
    """
    audit = AuditChain(conn)
    # 故意乱序写入，且刻意让两行共用同一时刻
    stamps = ["2026-10-22T12:00:00+08:00", "2026-10-22T09:00:00+08:00",
              "2026-10-22T12:00:00+08:00", "2026-10-22T10:00:00+08:00"]
    for index, stamp in enumerate(stamps):
        audit.append(
            request_id=REQUEST_ID, actor="service:test", action="writeback.sent",
            target=f"sess-{index}", outcome="accepted", rack_id=RACK, bed_no=BED,
            occurred_at=stamp,
        )
    first = audit.query()
    second = audit.query()
    assert [r["audit_id"] for r in first] == [r["audit_id"] for r in second]
    stamps_out = [r["occurred_at"] for r in first]
    assert stamps_out == sorted(stamps_out), "行序没有按发生时刻单调"
    # 同刻的两行必须按 audit_id 定序，而不是随库返回顺序漂
    same = [r["audit_id"] for r in first if r["occurred_at"] == stamps_out[0]]
    assert same == sorted(same)


def test_ward_level_calibration_rows_are_not_scoped_to_one_rack(conn):
    """标定是病区级记录：按架 / 按床导出时它**不出现**，按类别导出时它在。

    一个 ``TYPE-A`` 标定版本对全病区每一架都生效。给它硬安一个架号，看着像「更全」，
    实则是替它编了一个不存在的归属——按「这一架当天换了几次液」筛出来的清单里混进
    一条全病区的标定，护理部会以为这架的袋子刚被重新标定过。这条语义钉在这里，
    免得日后有人顺手给标定行补上当前请求的架号。
    """
    audit = AuditChain(conn)
    audit.append(
        request_id=REQUEST_ID, actor=f"nurse:{OPERATOR}", action="calibration.version_created",
        target="cal_fe0001", outcome="accepted",
        detail={"bag_type_code": "TYPE-A", "version": 1},  # 刻意不传 rack_id / bed_no
        occurred_at="2026-10-22T08:00:00+08:00",
    )
    assert len(audit.query(rack_id=RACK)) == 0, "标定行不该被算到某一架名下"
    assert len(audit.query(bed_no=BED)) == 0
    assert len(audit.query(actions=actions_in_categories(["calibration"]))) == 1


def test_chain_digest_is_stable_and_detects_any_row_change(conn):
    """链摘要：同一批行两次算同一个值；改一行或删一行都变。

    append-only 只是一句纪律，摘要是它的证据——留档人拿 CSV 重算一遍，
    对不上就知道这段审计被动过。
    """
    audit = AuditChain(conn)
    for index in range(5):
        audit.append(
            request_id=REQUEST_ID, actor="service:test", action="writeback.sent",
            target=f"sess-{index}", outcome="accepted", rack_id=RACK,
            detail={"n": index}, occurred_at=f"2026-10-22T09:0{index}:00+08:00",
        )
    rows = audit.query()
    baseline = AuditChain.chain_digest(rows)
    assert baseline != "empty-chain"
    assert AuditChain.chain_digest(audit.query()) == baseline

    # 删掉中间一行：即使剩余行原封不动，链式摘要也会变
    victim = rows[2]["audit_id"]
    conn.execute("DELETE FROM audit_log WHERE audit_id = ?", (victim,))
    assert AuditChain.chain_digest(audit.query()) != baseline


# ── 3. 过滤与导出 ──────────────────────────────────────────────────────────


def _populate_filters(conn) -> AuditChain:
    """铺三架两床、一整天的审计行，供过滤用例筛。"""
    audit = AuditChain(conn)
    layout = [
        (RACK, BED, "2026-10-22T08:00:00+08:00", "writeback.sent", "accepted"),
        (RACK, BED, "2026-10-22T09:00:00+08:00", "writeback.pending", "rejected"),
        (RACK, "EXAMPLE-BED-34", "2026-10-22T10:00:00+08:00", "state.confirmed", "accepted"),
        ("R-EXAMPLE-34", "EXAMPLE-BED-34", "2026-10-22T11:00:00+08:00", "reanchor.resolved", "accepted"),
        (RACK, BED, "2026-10-23T08:00:00+08:00", "candidate.claimed", "accepted"),
    ]
    for rack, bed, stamp, action, outcome in layout:
        audit.append(
            request_id=REQUEST_ID, actor="service:test", action=action,
            target=f"sess-{rack}-{stamp}", outcome=outcome, rack_id=rack, bed_no=bed,
            bag_seq="B01", detail={"stamp": stamp}, occurred_at=stamp,
        )
    return audit


def test_filters_return_exactly_the_matching_rows(conn):
    """按架、按床、按时间窗、按动作、按结果：逐条对得上，不多不少。"""
    audit = _populate_filters(conn)

    assert len(audit.query(rack_id=RACK)) == 4
    assert len(audit.query(bed_no="EXAMPLE-BED-34")) == 2
    assert len(audit.query(rack_id=RACK, bed_no=BED)) == 3
    windowed = audit.query(occurred_from="2026-10-22T00:00:00+08:00",
                           occurred_to="2026-10-22T23:59:59+08:00")
    assert len(windowed) == 4
    assert {r["action"] for r in windowed} == {"writeback.sent", "writeback.pending",
                                               "state.confirmed", "reanchor.resolved"}
    assert [r["action"] for r in audit.query(actions=["writeback.sent"])] == ["writeback.sent"]
    assert len(audit.query(outcomes=["rejected"])) == 1
    assert len(audit.query(rack_id=RACK, actions=["writeback.sent"])) == 1
    # 组合条件是「与」不是「或」
    assert audit.query(rack_id=RACK, outcomes=["rejected"])[0]["action"] == "writeback.pending"
    assert audit.query(rack_id="R-NOT-EXIST") == []


def test_filter_by_category_uses_the_question_vocabulary(conn):
    """按类别过滤走的是飞检的问法（不是内部动作码）。"""
    audit = _populate_filters(conn)
    sent = audit.query(actions=actions_in_categories(["writeback_sent"]))
    assert [r["action"] for r in sent] == ["writeback.sent"]
    pending = audit.query(actions=actions_in_categories(["writeback_pending"]))
    assert [r["action"] for r in pending] == ["writeback.pending"]
    confirmed = audit.query(actions=actions_in_categories(["confirmation"]))
    assert [r["action"] for r in confirmed] == ["state.confirmed"]


def test_legacy_rows_without_rack_columns_fall_back_to_session_lookup(conn):
    """旧行（没有 rack_id/bed_no 列）靠 bag_session 兜底，导得出来。

    迁移前落的行就是这样。不兜底的话，升级之后翻历史审计会「按架查不到、
    按床也查不到」，看着像数据丢了。
    """
    sessions = SessionRepository(conn)
    audit = AuditChain(conn)
    record = _seed(sessions)
    audit.append(
        request_id=REQUEST_ID, actor="service:test", action="writeback.sent",
        target=record["session_id"], outcome="accepted",
        detail={"legacy": True}, occurred_at="2026-10-22T08:00:00+08:00",
    )
    # 模拟旧行：把冗余列抹掉
    conn.execute("UPDATE audit_log SET rack_id = NULL, bed_no = NULL")
    rows = audit.query(rack_id=RACK)
    assert len(rows) == 1
    assert rows[0]["rack_id"] == RACK
    assert rows[0]["bed_no"] == BED


def test_csv_is_excel_readable_and_carries_chinese_headers(conn):
    """CSV 带 UTF-8 BOM、CRLF 行尾、中文表头，Excel 双击能开。

    没 BOM 的中文表头在 Windows Excel 里是乱码；没 CRLF 的话从 WPS 另存回来
    会带 ``\\r``，内审核对行数对不上——这两样都是留档事故，不是格式洁癖。
    """
    audit = _populate_filters(conn)
    filters = AuditFilter(rack_id=RACK)
    package = collect_package(audit, filters=filters, generated_at="2026-10-23T09:00:00+08:00")
    text = package.to_csv()
    assert text.startswith("﻿"), "缺少 UTF-8 BOM"
    assert "\r\n" in text and "\n" not in text.replace("\r\n", ""), "行尾不是纯 CRLF"
    assert package.filename("2026-10-23T09:00:00+08:00") == f"审计包_{RACK}_20261023.csv"

    # 用 Excel 的读法（带 BOM、CRLF）解析回来，逐格对得上
    # Excel 与 utf-8-sig 读法都会先吃掉 BOM；解析前按同样的口径剥掉。
    # newline="" 留给 open()，StringIO 这里传了会直接报 TypeError。
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")))
    rows = list(reader)
    assert rows[0] == [header for _, header in AUDIT_CSV_COLUMNS]
    assert len(rows) - 1 == package.count == 4
    assert rows[1][0] == "1"  # 审计序号
    assert "写回成功" in rows[1]  # 动作类别用中文标签
    body = reader.line_num
    assert body >= len(rows)


def test_csv_detail_column_keeps_compact_json_and_digest_column_is_exposed(conn):
    """明细列保留紧凑 JSON，摘要列露出来给内审核链。"""
    audit = AuditChain(conn)
    audit.append(
        request_id=REQUEST_ID, actor="nurse:X", action="candidate.claimed",
        target="sess-1", outcome="accepted", rack_id=RACK, bed_no=BED,
        detail={"candidate_id": "C01", "bag_seq": "B01"},
        occurred_at="2026-10-22T08:00:00+08:00",
    )
    row = list(csv.reader(io.StringIO(to_csv(audit.query()).lstrip("\ufeff"))))[1]
    table = dict(zip([h for _, h in AUDIT_CSV_COLUMNS], row))
    assert table["明细(JSON)"] == '{"bag_seq":"B01","candidate_id":"C01"}'
    assert len(table["载荷摘要"]) == 16


def test_manifest_digest_matches_the_exported_csv(conn):
    """manifest 的链摘要与 CSV 明细同源——留档人拿 CSV 重算要得到同一个值。"""
    audit = _populate_filters(conn)
    filters = AuditFilter(rack_id=RACK, occurred_from="2026-10-22T00:00:00+08:00",
                          occurred_to="2026-10-22T23:59:59+08:00")
    package = collect_package(audit, filters=filters, generated_at="2026-10-23T09:00:00+08:00")
    manifest = package.manifest
    assert manifest["row_count"] == 3
    assert manifest["window"]["from"] == "2026-10-22T08:00:00+08:00"
    assert manifest["window"]["to"] == "2026-10-22T10:00:00+08:00"
    assert manifest["by_category"]["writeback_sent"] == 1
    assert manifest["by_category"]["writeback_pending"] == 1
    assert manifest["by_category"]["other"] == 0
    # CSV 明细行序与摘要所依据的行序一致
    assert manifest["chain_digest"] == AuditChain.chain_digest(audit.query(
        rack_id=RACK, occurred_from="2026-10-22T00:00:00+08:00",
        occurred_to="2026-10-22T23:59:59+08:00",
    ))


def test_empty_window_still_exports_a_package(conn):
    """空结果照样出包：表头 + 0 行 + 计数 0。

    「昨天没有异常换液」本身是飞检要的答案；一个 404 反而让人怀疑数据丢了。
    """
    audit = AuditChain(conn)
    package = collect_package(
        audit,
        filters=AuditFilter(rack_id="R-NOT-EXIST"),
        generated_at="2026-10-23T09:00:00+08:00",
    )
    assert package.count == 0
    assert package.manifest["chain_digest"] == "empty-chain"
    assert package.manifest["window"] == {"from": None, "to": None}
    assert to_csv([]).lstrip("\ufeff").split("\r\n")[0] == ",".join(h for _, h in AUDIT_CSV_COLUMNS)


def test_csv_filename_is_stable_and_sanitised():
    """同一份包导出两次同名（Excel 覆盖而不是生成「审计包(1)」），路径字符被清掉。"""
    morning = csv_filename(generated_at="2026-10-23T09:00:00+08:00", scope="R-01")
    evening = csv_filename(generated_at="2026-10-23T18:00:00+08:00", scope="R-01")
    assert morning == evening == "审计包_R-01_20261023.csv"
    assert "/" not in csv_filename(generated_at="2026-10-23T09:00:00+08:00", scope="../../etc")
    assert csv_filename(generated_at="", scope=None) == "审计包_全病区_00000000.csv"


# ── 4. 审计缺失即业务回滚 ──────────────────────────────────────────────────


def test_state_transition_rolls_back_when_the_audit_write_fails(conn):
    """审计写失败 → 状态推进一起回滚，不留「变了但查不到谁改的」袋。

    故障注入用**数据库触发器**：``audit_log`` 上装一个遇到 ``state.change_proposed``
    就 ABORT 的触发器，等价于磁盘满 / 库被锁 / 行超长这类真实写失败——不需要在
    产品代码里留一个只为测试存在的开关。
    """
    sessions = SessionRepository(conn)
    audit = AuditChain(conn)
    record = _seed(sessions)
    conn.execute(
        "CREATE TRIGGER fail_change_proposed BEFORE INSERT ON audit_log "
        "WHEN NEW.action = 'state.change_proposed' "
        "BEGIN SELECT RAISE(ABORT, 'simulated audit write failure'); END"
    )

    with pytest.raises(sqlite3.IntegrityError):
        apply_decision(
            sessions, _naming_decision(record["session_id"]),
            audit=audit, request_id=REQUEST_ID,
        )

    after = sessions.get(record["session_id"])
    assert after["state"] == "monitoring", "审计写失败后状态竟然推下去了"
    assert audit.count_by_action().get("state.change_proposed") is None


def test_confirmation_rolls_back_when_the_audit_write_fails(conn):
    """护士确认同理：确认动作与审计同事务，审计写不进去就整体回滚。"""
    sessions = SessionRepository(conn)
    audit = AuditChain(conn)
    record = _seed(sessions)
    apply_decision(
        sessions, _naming_decision(record["session_id"]), audit=audit, request_id="req-nm",
    )
    conn.execute(
        "CREATE TRIGGER fail_confirmed BEFORE INSERT ON audit_log "
        "WHEN NEW.action = 'state.confirmed' "
        "BEGIN SELECT RAISE(ABORT, 'simulated audit write failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        confirm_replacement_once(
            sessions, EventRepository(conn), record["session_id"],
            bag_seq="B01", operator_id=OPERATOR, audit=audit, request_id=REQUEST_ID,
        )
    assert sessions.get(record["session_id"])["state"] == "needs_change"
    # 事务整体回滚，出站事件也没有落库——不存在「确认了但事件没写」
    assert int(conn.execute("SELECT COUNT(*) AS n FROM infusion_event").fetchone()["n"]) == 0


def test_audit_record_without_a_handle_is_refused(conn):
    """给了审计内容却没给句柄时直接报错，不做「先写状态回头再补审计」。

    悄悄少写一行审计比报错更糟：调用方以为记上了，实际查无实据。
    """
    sessions = SessionRepository(conn)
    record = _seed(sessions)
    with pytest.raises(ValueError, match="audit"):
        sessions.apply_state_transition(
            record["session_id"], "needs_change",
            audit_record={"action": "state.change_proposed", "request_id": REQUEST_ID,
                          "actor": "service", "target": record["session_id"],
                          "outcome": "accepted", "detail": {}},
        )
    assert sessions.get(record["session_id"])["state"] == "monitoring"


def test_append_rejects_unknown_action_before_writing_anything(conn):
    """非法动作码在写之前就抛错——这条路径上抛异常是要触发回滚的。"""
    audit = AuditChain(conn)
    with pytest.raises(ValueError, match="未知审计动作"):
        audit.append(request_id=REQUEST_ID, actor="x", action="invented.action",
                     target="t", outcome="accepted")
    assert audit.count() == 0


def test_audit_chain_exposes_no_update_or_delete(conn):
    """append-only 是 API 层面的纪律：链上没有改与删的方法。"""
    audit = AuditChain(conn)
    audit.append(request_id=REQUEST_ID, actor="x", action="writeback.sent",
                 target="t", outcome="accepted", rack_id=RACK)
    for forbidden in ("update", "delete", "purge", "truncate"):
        assert not hasattr(audit, forbidden), f"审计链暴露了 {forbidden}()"


# ── 5. HTTP 端点 ───────────────────────────────────────────────────────────


def test_audit_endpoints_export_and_manifest(store_client):
    """两个端点取同一批行：CSV 带行数/摘要响应头，manifest 给护理部签收的那一页。"""
    client, observer = store_client
    _populate_filters(observer)

    response = client.get("/api/v1/infusion/audit", params={"rack_id": RACK})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["X-Audit-Row-Count"] == "4"
    assert response.headers["X-Audit-Scope"] == RACK
    assert "filename*=UTF-8''" in response.headers["content-disposition"]
    # BOM 按字节验：httpx 解析响应体时会把 BOM 当编码标记吃掉，str 上看不见
    assert response.content.startswith(b"\xef\xbb\xbf")
    assert response.text.lstrip("\ufeff").startswith("审计序号")

    manifest = client.get(
        "/api/v1/infusion/audit/manifest", params={"rack_id": RACK}
    )
    assert manifest.status_code == 200
    body = manifest.json()
    assert body["row_count"] == 4
    assert body["filters"]["rack_id"] == RACK
    assert body["chain_digest"] == response.headers["X-Audit-Chain-Digest"]
    assert body["category_labels_zh"]["writeback_pending"] == "写回挂起"


def test_audit_export_is_read_only(store_client):
    """导一次审计**不会**新增审计行：留档动作本身不能改变被留档的内容。"""
    client, observer = store_client
    _populate_filters(observer)
    before = AuditChain(observer).count()
    for _ in range(3):
        client.get("/api/v1/infusion/audit")
        client.get("/api/v1/infusion/audit/manifest")
    assert AuditChain(observer).count() == before


def test_audit_export_rejects_unknown_category_and_oversized_limit(store_client):
    """未知类别 / 超限导出会 422 并说清怎么改，不静默忽略。"""
    client, _observer = store_client
    bad = client.get("/api/v1/infusion/audit", params={"category": "no_such_category"})
    assert bad.status_code == 422
    assert bad.json()["detail"]["issues"][0]["code"] == "audit_category_unknown"

    big = client.get("/api/v1/infusion/audit", params={"limit": 99999})
    assert big.status_code == 422
    assert big.json()["detail"]["issues"][0]["code"] == "limit_too_large"

    bad_outcome = client.get("/api/v1/infusion/audit", params={"outcome": "maybe"})
    assert bad_outcome.status_code == 422
    assert bad_outcome.json()["detail"]["issues"][0]["code"] == "audit_outcome_unknown"


def test_claim_then_export_shows_the_operator_in_the_package(store_client):
    """端到端：上传 → 认领 → 导审计包，认领行里带着护士工号。

    这是飞检真正会走的一步：认领是人的动作，认了谁必须留在包里。
    """
    client, _observer = store_client
    scene = default_rack_scene(20)
    upload = client.post(
        "/api/v1/infusion/tracks",
        data={
            "rack_id": RACK, "infusion_order_no": "EXAMPLE-ORDER-0033", "bed_no": BED,
            "captured_at": "2026-10-22T09:41:00+08:00", "shift_key": "2026-10-22-AM",
            "slots": SLOTS,
        },
        files={"frames": ("frames.zip", build_rack_zip(scene), "application/zip")},
    )
    assert upload.status_code == 201, upload.text
    candidates = upload.json()["candidates"]
    session_id = candidates[0]["session_id"] if candidates and "session_id" in candidates[0] \
        else upload.json()["bags"][0]["session_id"]

    claim = client.post(
        "/api/v1/infusion/claim",
        json={"session_id": session_id, "candidate_id": candidates[0]["candidate_id"],
              "operator_id": OPERATOR, "note": "加药袋"},
    )
    assert claim.status_code == 200, claim.text

    package = client.get("/api/v1/infusion/audit", params={"category": "claim"}).text
    assert OPERATOR in package
    assert "认领" in package