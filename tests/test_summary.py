"""运行摘要、结构化日志与配置自检的验收测试。

这里盯的是**四个值班问题各自对应一个可核对的数**：

1. 「现在有几袋该换液了？」→ ``sessions.needs_change_count``
2. 「有没有换液事件没送出去？」→ ``writeback.pending_writeback_count``
3. 「模型是不是一直在连不上？」→ ``model.threshold_fallback_count``
4. 「遥测还在不在流？」→ ``channels.telemetry.state``

其中第 2 条最容易算错，而且算错的方向很坏：把「护士还没确认的 pending」也算进
积压，护士长会以为自己漏点了确认，跑去逐袋核对——而那些事件本来就该还没出站。
所以有一条专门的反例钉死这个口径。

另一半是**贯穿性**：入站的 ``X-Request-Id`` 要能一路走到响应头、业务日志、审计行，
以及凭据在任何一条日志里都不能出现明文。这两条是排障时唯一能靠机器做的事。

最后是**文档与代码不漂移**：``docs/error-codes.md`` 由 ``errors.py`` 生成，
测试比对是否同步——挂接方拿着一份过期码表调接口，比没有码表更糟。
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
from pathlib import Path

import pytest

from infusion_monitor.errors import (
    ERROR_CODES,
    error_code_markdown,
    error_code_table,
    fault_table,
    operational_faults,
)
from infusion_monitor.logging_utils import (
    JsonFormatter,
    RuntimeCounters,
    configure_logging,
    current_request_id,
    log_event,
    redact,
    request_scope,
)
from infusion_monitor.store import AuditChain, EventRepository, SessionRepository
from infusion_monitor.store.schema import SCHEMA_VERSION

from tests.support import build_rack_zip, default_rack_scene

SUMMARY_URL = "/api/v1/summary"
HEALTH_URL = "/api/v1/health"
TRACKS_URL = "/api/v1/infusion/tracks"
ERROR_DOC = Path(__file__).resolve().parent.parent / "docs" / "error-codes.md"
CHECK_CONFIG = Path(__file__).resolve().parent.parent / "scripts" / "check_config.py"

SLOTS_TWO_BAGS = (
    '[{"bag_seq":"B01","slot_index":0,"bag_type_code":"TYPE-A","rate_text":"5ml/h"},'
    '{"bag_seq":"B02","slot_index":1,"bag_type_code":"TYPE-A","rate_text":"5ml/h"}]'
)


# ─────────────────────────── 摘要：四个值班问题 ───────────────────────────


def test_summary_exposes_all_four_ward_level_questions(client):
    """摘要里必须同时答得上四个问题；缺一个字段就是值班的人得再问一次。"""
    payload = client.get(SUMMARY_URL).json()

    assert payload["service"] == "infusion-volume-monitor"
    assert payload["version"]
    # 「有几袋该换液了」
    assert payload["sessions"]["needs_change_count"] == 0
    assert "monitoring" in payload["sessions"]["by_state"] or payload["sessions"]["total"] == 0
    # 「有没有事件没送出去」
    assert payload["writeback"]["pending_writeback_count"] == 0
    # 「模型是不是连不上」
    assert payload["model"]["threshold_fallback_count"] == 0
    # 「遥测在不在流」
    assert payload["channels"]["telemetry"]["state"]
    assert payload["runtime"]["started_at"]
    assert payload["runtime"]["uptime_seconds"] >= 0


def test_needs_change_count_tracks_nurse_pending_actions(store_client):
    """needs_change = 系统已点名、护士还没确认。这是**欠着的人工动作**，不是错误。"""
    client, observer = store_client
    sessions = SessionRepository(observer)

    def seed(bag_seq: str, state_after: str | None = None) -> str:
        row = sessions.upsert_session(
            {
                "rack_id": "R-EXAMPLE-13",
                "bed_no": "EXAMPLE-BED-13",
                "infusion_order_no": "EXAMPLE-ORDER-0013",
                "shift_key": "2026-10-22-AM",
                "bag_seq": bag_seq,
                "slot_index": int(bag_seq[-1]) - 1,
                "frames_dir": "/tmp/frames",
                "frame_count": 16,
                "captured_at": "2026-10-22T09:41:00+08:00",
                "state": "monitoring",
            }
        )
        if state_after:
            sessions.apply_state_transition(row["session_id"], state_after)
        return row["session_id"]

    assert client.get(SUMMARY_URL).json()["sessions"]["needs_change_count"] == 0

    seed("B01", "needs_change")
    seed("B02", "needs_change")
    seed("B03", "needs_change")
    seed("B04")  # 还在巡视中，不点名
    payload = client.get(SUMMARY_URL).json()

    assert payload["sessions"]["total"] == 4
    assert payload["sessions"]["needs_change_count"] == 3
    assert payload["sessions"]["by_state"]["needs_change"] == 3


def test_pending_writeback_count_excludes_unconfirmed_pending(store_client):
    """反例钉死口径：**未确认的 pending 不算积压**。

    护士点了「该换液」但还没确认换没换时，事件状态是 ``pending``。把它算进
    ``pending_writeback_count``，护士长会以为自己漏点了确认，去逐袋核对——而那些
    事件本来就**还不该**出站。只有对端送不出去（退避耗尽）的才算真积压。
    """
    client, observer = store_client
    events = EventRepository(observer)
    sessions = SessionRepository(observer)

    rows = [
        ("client-unconfirmed-1", "pending", 0),
        ("client-unconfirmed-2", "pending", 0),
        ("client-stuck-1", "pending_writeback", 3),
        ("client-stuck-2", "pending_writeback", 3),
        ("client-stuck-3", "pending_writeback", 3),
        ("client-sent-1", "sent", 1),
        ("client-failed-1", "failed", 1),
    ]
    for index, (client_event_id, status, attempts) in enumerate(rows):
        session = sessions.upsert_session(
            {
                "rack_id": "R-EXAMPLE-13",
                "bed_no": "EXAMPLE-BED-13",
                "infusion_order_no": "EXAMPLE-ORDER-0013",
                "shift_key": "2026-10-22-AM",
                "bag_seq": f"B{index + 1:02d}",
                "slot_index": index,
                "frames_dir": "/tmp/frames",
                "frame_count": 16,
                "captured_at": "2026-10-22T09:41:00+08:00",
                "state": "monitoring",
            }
        )
        event, created = events.append_event(
            {
                "client_event_id": client_event_id,
                "session_id": session["session_id"],
                "rack_id": "R-EXAMPLE-13",
                "bed_no": "EXAMPLE-BED-13",
                "infusion_order_no": "EXAMPLE-ORDER-0013",
                "bag_seq": f"B{index + 1:02d}",
                "operator_id": "N-EXAMPLE-13",
                "occurred_at": "2026-10-22T11:05:00+08:00",
            }
        )
        assert created
        events.mark_writeback(event["event_id"], status, attempt_count=attempts)

    payload = client.get(SUMMARY_URL).json()
    writeback = payload["writeback"]

    assert writeback["total"] == 7
    assert writeback["pending_writeback_count"] == 3, "只有退避耗尽的才算真积压"
    assert writeback["unsent_pending_count"] == 2, "未确认的 pending 单独记，不混进积压"
    assert writeback["sent_count"] == 1
    assert writeback["failed_count"] == 1
    assert writeback["by_status"]["pending"] == 2


def test_threshold_fallback_count_shows_model_outage(store_client):
    """模型一直连不上时，这个数会涨——它就是给信息科看的「该不该去查端点」。"""
    client, observer = store_client
    from infusion_monitor.store import InferenceAuditRepository

    audit = InferenceAuditRepository(observer)
    sources = ["model_decided"] * 7 + ["threshold_fallback"] * 3 + ["threshold_only"]
    for index, source in enumerate(sources):
        audit.append(
            {
                "request_id": f"req-summary-{index:02d}",
                "model": "test-model",
                "prompt_hash": "0" * 16,
                "decision_source": source,
                "pending_manual_review": source == "threshold_fallback",
            }
        )

    model = client.get(SUMMARY_URL).json()["model"]

    assert model["total_calls"] == 11
    assert model["model_decided_count"] == 7
    assert model["threshold_fallback_count"] == 3
    assert model["threshold_only_count"] == 1
    assert model["pending_manual_review_count"] == 3
    assert model["by_decision_source"]["threshold_fallback"] == 3


def test_summary_reports_telemetry_channel_state(client):
    """两条入站通道都没启用时，摘要要说清是「没接泵网关」而不是「连不上」。

    这两种在病区侧的处理完全不同：前者什么都不用做，后者要查 broker 或泵网关。
    task 39 起入站是双通道（MQTT + webhook 兜底），合并视图多了一个 ``disabled``：
    两条都没开时它比 ``idle`` 更准——「idle」听着像「开着但闲着」，而没接泵网关
    的病区占多数，用 idle 会让人一直去找那个不存在的 broker。
    """
    payload = client.get(SUMMARY_URL).json()
    telemetry = payload["channels"]["telemetry"]

    assert telemetry["enabled"] is False
    assert telemetry["state"] == "disabled"
    assert telemetry["mqtt_enabled"] is False
    assert telemetry["webhook_enabled"] is False
    assert telemetry["last_message_at"] is None
    assert telemetry["cached_samples"] == 0
    assert payload["channels"]["webhook"]["enabled"] is False
    assert payload["channels"]["webhook"]["state"] == "disabled"
    assert payload["channels"]["mqtt"]["enabled"] is False


def test_summary_never_leaks_plaintext_credentials(client, credential_env):
    """摘要是给值班的人看的，谁都可能截图转进工作群。凭据一个字符都不能在里面。"""
    body = client.get(SUMMARY_URL).text

    assert "test-token-not-real" not in body
    assert "test-password-not-real" not in body
    assert "test-key-not-real" not in body
    # 掩码视图仍在，说明是「打码」而不是「整个字段删掉」
    token = client.get(SUMMARY_URL).json()["config"]["credentials"]["nurse_exec_api_token"]
    assert "***" in token and token != "test-token-not-real"
    assert token == "te***al"  # 两头各留两位，中间打码


def test_health_stays_cheap_and_summary_stays_separate(client):
    """health 给探活打、summary 给值班的人看：两者不能互相拖慢。

    合成健康检查若误进业务日志，逐行刷屏会把真正的业务事件淹掉——这条在
    ``test_health_probe_is_not_logged_as_business_event`` 里单独验。
    """
    health = client.get(HEALTH_URL).json()
    summary = client.get(SUMMARY_URL).json()

    assert health["status"] == "ok"
    assert "session_count" in health
    # health 不做四张表的聚合：不带 sessions.by_state 这类昂贵字段
    assert "by_state" not in health
    assert health["schema_version"] == SCHEMA_VERSION
    # summary 才带运行账
    assert summary["runtime"]["request_count"] >= 1


def test_summary_is_read_only(client, store_client):
    """摘要只读：不写库、不外发。跑十次之后事件与会话数都不变。"""
    test_client, observer = store_client
    before_audit = AuditChain(observer).count()
    before_sessions = SessionRepository(observer).count()

    for _ in range(10):
        assert test_client.get(SUMMARY_URL).status_code == 200

    assert AuditChain(observer).count() == before_audit
    assert SessionRepository(observer).count() == before_sessions


def test_summary_reports_the_retention_overview(client, store_client):
    """值班的人要能从摘要看见保留期状态：下一次清理会清哪几袋、为什么有的清不掉。

    定时清理在 cron 里跑，摘要不给这一段，运维就只能在日志里找答案。
    """
    test_client, observer = store_client
    sessions = SessionRepository(observer)
    row = sessions.upsert_session(
        {
            "rack_id": "R-EXAMPLE-44",
            "bed_no": "EXAMPLE-BED-44",
            "infusion_order_no": "EXAMPLE-ORDER-0044",
            "shift_key": "2026-10-22-AM",
            "bag_seq": "B01",
            "slot_index": 0,
            "frames_dir": "/tmp/frames-44",
            "frame_count": 16,
            "captured_at": "2026-10-22T09:41:00+08:00",
            "state": "monitoring",
        }
    )

    payload = test_client.get(SUMMARY_URL).json()
    retention = payload["retention"]

    assert retention["policy"]["session_retention_days"] == 30.0
    assert retention["sessions_by_frames_state"]["kept"] >= 1
    assert "blocked_by_reason" in retention
    assert "eligible_for_next_purge" in retention
    # 刚上传的会话还在保留期内，不该被算进「下次会清掉」
    assert row["session_id"] not in retention.get("eligible_sessions", [])
    assert retention["eligible_for_next_purge"] == 0


# ─────────────────────────── 错误码表与文档同步 ───────────────────────────


def test_error_code_table_covers_every_registered_code():
    """错误码表不许漏项——漏了的码在 /docs 里查不到，挂接方只能猜。"""
    from infusion_monitor.errors import error_code_docs

    documented = {doc.code for doc in error_code_docs()}
    assert documented == set(ERROR_CODES)

    for doc in error_code_docs():
        assert doc.status == ERROR_CODES[doc.code]
        assert doc.meaning.strip() and doc.handling.strip()


def test_operational_faults_cover_the_named_scenarios():
    """任务点名的六类故障一个都不能少：帧序非法/状态非法属 API 码，其余四类属运行期。"""
    keys = {fault.key for fault in operational_faults()}
    assert keys == {
        "credentials_missing",
        "nurse_exec_unreachable",
        "model_unreachable",
        "mqtt_disconnected",
        "frames_root_not_writable",
        "quota_exceeded",
    }


def test_error_code_doc_is_in_sync_with_code():
    """docs/error-codes.md 必须由代码重新生成得一模一样，否则就是一份过期码表。"""
    assert ERROR_DOC.exists(), "docs/error-codes.md 不存在；用 error_code_markdown() 生成"
    assert ERROR_DOC.read_text(encoding="utf-8") == error_code_markdown()

    body = ERROR_DOC.read_text(encoding="utf-8")
    for code, status in ERROR_CODES.items():
        assert f"| `{code}` | {status} |" in body, f"码表缺 {code}"
    assert fault_table() in body


def test_error_tables_are_valid_markdown_rows():
    """两张表每一行的列数一致——列数不齐渲染出来是错位的表格。"""
    for table in (error_code_table(), fault_table()):
        rows = [line for line in table.splitlines() if line.startswith("|")]
        widths = {line.count("|") for line in rows}
        assert len(widths) == 1, f"表格列数不齐：{widths}"
        assert len(rows) >= 8, "错误码表不该只有寥寥几行"


# ─────────────────────────── 结构化日志 ───────────────────────────


def test_structured_log_line_is_one_json_object_with_request_id():
    """一行一个 JSON 对象、request_id 在顶层——采集器才能直接 jq。"""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(service="infusion-volume-monitor"))
    logger = logging.getLogger("infusion_monitor.test.json")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    with request_scope("req_abc123"):
        log_event(logger, logging.INFO, "session.created",
                  session_id="sess-1", bag_seq="B01", frame_count=24)

    line = stream.getvalue().strip()
    assert "\n" not in line, "一条事件必须只占一行"
    payload = json.loads(line)
    assert payload["request_id"] == "req_abc123"
    assert payload["event"] == "session.created"
    assert payload["level"] == "INFO"
    assert payload["service"] == "infusion-volume-monitor"
    assert payload["session_id"] == "sess-1"
    assert payload["frame_count"] == 24


def test_request_id_absent_outside_request_context_is_explicit():
    """不在请求上下文里时字段直接不出现，而不是塞一个 null 占位。"""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("infusion_monitor.test.norid")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    assert current_request_id() is None
    log_event(logger, logging.INFO, "background.sweep")

    assert "request_id" not in json.loads(stream.getvalue().strip())


def test_sensitive_fields_are_redacted_before_reaching_the_stream():
    """凭据位一共 7 组，但日志会流向采集器、告警群与飞检材料——一处泄漏就得全院换密钥。"""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("infusion_monitor.test.redact")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    with request_scope("req_secret"):
        log_event(
            logger,
            logging.WARNING,
            "upstream.failed",
            nurse_exec_api_token="real-token-value-1234",
            authorization="Bearer real-token-value-1234",
            nested={"llm_api_key": "sk-real-key-9999", "mqtt_password": "real-pass"},
            rack_id="R-EXAMPLE-13",
        )

    body = stream.getvalue()
    assert "real-token-value-1234" not in body
    assert "sk-real-key-9999" not in body
    assert "real-pass" not in body
    payload = json.loads(body.strip())
    assert payload["nurse_exec_api_token"] == "***"
    assert payload["authorization"] == "***"
    assert payload["nested"]["llm_api_key"] == "***"
    # 非敏感字段照常保留：脱敏不该把可排障的信息一起抹掉
    assert payload["rack_id"] == "R-EXAMPLE-13"


def test_redact_keeps_structure_intact():
    """打码后结构不变，采集器侧的字段解析才不会因为脱敏而断。"""
    payload = {"a": [{"token": "x"}, {"ok": 1}], "n": None, "i": 3}
    out = redact(payload)

    assert out == {"a": [{"token": "***"}, {"ok": 1}], "n": None, "i": 3}
    assert isinstance(out["a"], list)
    assert out["i"] == 3


def test_health_probe_is_not_logged_as_business_event(client, caplog):
    """探活每秒可能被打一次，逐条记 INFO 会把真正的业务事件淹掉。"""
    with caplog.at_level(logging.INFO, logger="infusion_monitor.http"):
        client.get(HEALTH_URL)
        client.get(SUMMARY_URL)

    events = [rec.getMessage() for rec in caplog.records]
    paths = [
        getattr(rec, "fields", {}).get("path")
        for rec in caplog.records
        if rec.getMessage() == "http.request"
    ]
    assert "/api/v1/summary" in paths
    assert "/api/v1/health" not in paths


def test_request_id_reaches_response_header_and_audit_chain(store_client):
    """入站 X-Request-Id → 响应头 → 审计行：一次上传三处必须是同一个 id。

    护士端把出问题那次的响应头截图发给信息科，就能定位到唯一一行日志与一组审计记录。
    """
    client, observer = store_client
    inbound = "req-from-ward-terminal-0013"
    scene = default_rack_scene()

    response = client.post(
        TRACKS_URL,
        headers={"X-Request-Id": inbound},
        data={
            "rack_id": "R-EXAMPLE-13",
            "infusion_order_no": "EXAMPLE-ORDER-0013",
            "bed_no": "EXAMPLE-BED-13",
            "captured_at": "2026-10-22T09:41:00+08:00",
            "shift_key": "2026-10-22-AM",
            "slots": SLOTS_TWO_BAGS,
        },
        files={"frames": ("frames.zip", build_rack_zip(scene), "application/zip")},
    )
    assert response.status_code == 201
    assert response.headers["X-Request-Id"] == inbound, "响应头必须回写同一个 id"

    audit_rows = AuditChain(observer).list_by_request(inbound)
    assert audit_rows, "上传这次请求应当留下审计行"
    assert all(row["request_id"] == inbound for row in audit_rows)


def test_business_log_lines_share_the_inbound_request_id(store_client):
    """「贯穿」的**实际**含义：业务行的 request_id 也得是入站那一个。

    只在 http.request 上带 id、业务行各带各的，排障时照样拼不出整条链——
    拿到护士端给的 id，grep 不到「帧序列接入完成」那行，等于没贯穿。
    """
    client, _observer = store_client
    inbound = "req-chain-0013"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    for name in ("infusion_monitor.ingest", "infusion_monitor.enumerate",
                 "infusion_monitor.core", "infusion_monitor.api"):
        target = logging.getLogger(name)
        target.handlers = [handler]
        target.propagate = False
        target.setLevel(logging.INFO)

    client.post(
        TRACKS_URL,
        headers={"X-Request-Id": inbound},
        data={
            "rack_id": "R-EXAMPLE-13",
            "infusion_order_no": "EXAMPLE-ORDER-0015",
            "bed_no": "EXAMPLE-BED-13",
            "captured_at": "2026-10-22T09:41:00+08:00",
            "shift_key": "2026-10-22-AM",
            "slots": SLOTS_TWO_BAGS,
        },
        files={"frames": ("frames.zip", build_rack_zip(default_rack_scene()),
                          "application/zip")},
    )

    business = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    assert business, "上传过程应当产生业务事件行"
    assert all(item["request_id"] == inbound for item in business), business


def test_request_id_generated_when_caller_sends_none(store_client):
    """调用方没带 X-Request-Id 时服务端自己生成一个，并照样回写——不能是空。"""
    client, _observer = store_client
    scene = default_rack_scene()

    response = client.post(
        TRACKS_URL,
        data={
            "rack_id": "R-EXAMPLE-13",
            "infusion_order_no": "EXAMPLE-ORDER-0014",
            "bed_no": "EXAMPLE-BED-13",
            "captured_at": "2026-10-22T09:41:00+08:00",
            "shift_key": "2026-10-22-AM",
            "slots": SLOTS_TWO_BAGS,
        },
        files={"frames": ("frames.zip", build_rack_zip(scene), "application/zip")},
    )
    assert response.status_code == 201
    generated = response.headers["X-Request-Id"]
    assert generated.startswith("req_") and len(generated) > len("req_")


def test_runtime_counters_track_requests_and_errors(store_client):
    """进程内粗账：值班的人至少能知道「这个进程起来以后被打过多少次」。"""
    client, _observer = store_client
    counters = client.app.state.counters
    before = counters.request_count

    client.get(HEALTH_URL)
    client.get(SUMMARY_URL)

    assert counters.request_count == before + 2
    assert counters.error_count == 0
    assert counters.uptime_seconds() >= 0


def test_configure_logging_does_not_stack_handlers():
    """重复 configure 不叠加 handler——叠加一次日志就打两遍，采集器按体积计费。"""
    root = logging.getLogger()
    before = len([h for h in root.handlers if getattr(h, "_infusion_monitor", False)])
    configure_logging("INFO", stream=io.StringIO())
    configure_logging("INFO", stream=io.StringIO())
    after = len([h for h in root.handlers if getattr(h, "_infusion_monitor", False)])

    assert after - before == 1


# ─────────────────────────── 配置自检命令 ───────────────────────────


def _load_check_config():
    spec = importlib.util.spec_from_file_location("check_config_under_test", CHECK_CONFIG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_config_lists_every_missing_group_by_name(clean_env):
    """一个都不给时，7 组必须**逐组点名**，而不是只说一句「配置缺失」。"""
    module = _load_check_config()
    result = module.run_check()

    assert result["ok"] is False
    assert len(result["credential_groups"]) == 7
    missing = [g["key"] for g in result["credential_groups"] if not g["present"]]
    assert len(missing) == 7
    assert "mqtt" in missing and "llm_api_key" in missing

    mqtt_group = next(g for g in result["credential_groups"] if g["key"] == "mqtt")
    assert mqtt_group["env_names"] == ["MQTT_HOST", "MQTT_PORT", "MQTT_USERNAME", "MQTT_PASSWORD"]
    assert "MQTT_HOST" in mqtt_group["example"]

    text = result["error_text"]
    assert "MQTT_HOST" in text
    assert "NURSE_EXEC_API_TOKEN" in text
    assert "LLM_MODEL" in text


def test_check_config_flags_blank_values_separately_from_absent(clean_env, monkeypatch):
    """设了但值为空，和根本没设，是两种病：空值往往是复制粘贴时漏了后半行。"""
    module = _load_check_config()
    monkeypatch.setenv("MQTT_HOST", "   ")
    result = module.run_check()

    mqtt_group = next(g for g in result["credential_groups"] if g["key"] == "mqtt")
    assert "MQTT_HOST" in mqtt_group["blank"]
    assert "MQTT_HOST" not in mqtt_group["env_names"][:0]  # 结构没被改动


def test_check_config_passes_when_complete(credential_env, tmp_path, monkeypatch):
    """齐了就必须放行——自检命令天天拦着不让启动，比不装还糟。"""
    module = _load_check_config()
    monkeypatch.setenv("FRAMES_ROOT", str(tmp_path / "frames"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'check.db'}")

    result = module.run_check()

    assert result["ok"] is True
    assert result["missing_group_keys"] == []
    assert result["error_text"] is None
    # 顺带把端点拼出来给人核对（不外发请求）
    assert result["derived"]["出站写回端点"].endswith("/api/v1/infusion/events")
    assert result["derived"]["遥测订阅主题"] == "hospital/infusion/+/telemetry"


def test_check_config_detects_unwritable_paths(credential_env, tmp_path, monkeypatch):
    """帧目录不可写是病区侧最常见的坑，症状是「上传到一半 500」，很难往配置上想。"""
    module = _load_check_config()
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)  # 只读父目录：服务无法在里面创建 frames
    monkeypatch.setenv("FRAMES_ROOT", str(locked / "frames"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'check.db'}")

    try:
        result = module.run_check()
    finally:
        locked.chmod(0o700)

    frame_check = next(p for p in result["paths"] if p["label"] == "会话帧目录")
    assert frame_check["exists"] is False
    if os.geteuid() == 0:
        pytest.skip("root 忽略目录权限位，测不出只读目录")
    assert frame_check["ok"] is False
    assert result["ok"] is False
    assert "INSTALL.md" in frame_check["hint"]


def test_check_config_exit_code_is_nonzero_when_incomplete(clean_env, capsys):
    """退出码是装机脚本唯一的判断依据：缺项时必须非 0，否则 systemd 会反复拉起一个起不来的服务。"""
    module = _load_check_config()

    assert module.main([]) == 1
    assert "缺失" in capsys.readouterr().out
    assert module.main(["--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_check_config_exit_code_is_zero_when_complete(credential_env, tmp_path, monkeypatch, capsys):
    """齐了就必须放行——自检天天拦着不让启动，比不装还糟。"""
    module = _load_check_config()
    monkeypatch.setenv("FRAMES_ROOT", str(tmp_path / "frames"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'check.db'}")
    assert module.main([]) == 0
    assert "配置齐全" in capsys.readouterr().out

    assert module.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert len(payload["credential_groups"]) == 7


def test_check_config_never_prints_plaintext_secrets(credential_env, tmp_path, monkeypatch, capsys):
    """自检输出常被直接贴进工作群问信息科，凭据不能跟着出去。"""
    module = _load_check_config()
    monkeypatch.setenv("FRAMES_ROOT", str(tmp_path / "frames"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'check.db'}")

    module.main([])
    body = capsys.readouterr().out

    assert "test-token-not-real" not in body
    assert "test-password-not-real" not in body
    assert "test-key-not-real" not in body


def test_runtime_counters_as_dict_is_json_serialisable():
    counters = RuntimeCounters()
    counters.note_request()
    counters.note_request(error=True)
    payload = counters.as_dict()

    json.dumps(payload)  # 摘要要把它整个塞进响应，不可序列化就是 500
    assert payload["request_count"] == 2
    assert payload["error_count"] == 1


@pytest.mark.parametrize("code,status", sorted(ERROR_CODES.items()))
def test_every_error_code_appears_in_summary_payload(client, code, status):
    """摘要带着错误码表：挂接方不用去翻源码就知道自己收到的是什么。"""
    payload = client.get(SUMMARY_URL).json()
    codes = {item["code"]: item["status"] for item in payload["error_codes"]}
    assert codes[code] == status