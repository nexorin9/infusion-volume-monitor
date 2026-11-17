"""对接端到端演练（task 40）：把一条换液事件从「护士拍片」跑到「对端收到」。

**替身只在测试数据层。** :class:`~tests.support.nurse_exec_stub.NurseExecStub` 起在
``127.0.0.1`` 的随机端口上，路径、报头、JSON 体、超时与半开连接全走真 socket。
**出站客户端与字段规范是正式实现**（真实协议、真实端点
``POST {NURSE_EXEC_BASE_URL}/api/v1/infusion/events``、真实凭据位
``Authorization: Bearer $NURSE_EXEC_API_TOKEN``）；**真实护理执行平台端点本轮未实连**。
替身不代表对端的接口行为，接挂时以 :mod:`infusion_monitor.adapters.nurse_exec` 里的
字段规范与对应表为准。

与 :mod:`tests.contract.test_writeback_contract` 的分工：那边逐条钉字段与失败分类，
这里跑**完整一条业务链**并做失败注入——

* :func:`test_unconfirmed_pickup_never_leaves_the_host`：点名了但护士没点确认，
  替身收到 **0** 个请求、事件表 **0** 行。没确认就出站等于系统替护士签字换袋；
* :func:`test_confirmed_replacement_reaches_the_peer`：确认 → 替身收到**全量字段**，
  ``Authorization`` 与 ``Idempotency-Key`` 逐项核对（键与请求体同值），一次成功；
* :func:`test_injected_5xx_backs_off_twice_then_lands_in_pending`：注入连续 5xx →
  首次 + **2** 次指数退避重试（1s / 4s）→ ``pending_writeback``，事件**一行不少**，
  审计链与运行摘要同步更新；
* :func:`test_injected_business_rejection_is_not_retried`：注入 4xx → 一次都不重试，
  落 ``failed`` 并把对端给的原因抽出来；
* :func:`test_injected_half_close_is_retried_under_the_same_idempotency_key`：
  注入半开连接（对端掐了连接不给应答）→ 退避重试成功，两次都带**同一个**幂等键，
  对端不会因此收成两条；
* :func:`test_replay_after_the_peer_recovers_delivers_the_same_event`：对端恢复后
  人工重放 → 送达、事件表仍是 1 行（不产生第二条业务事件）、重放后再重放幂等命中
  且**一次 HTTP 都不发**。

演练全程不依赖外网：出站端点是本机替身，模型端点指向一个关闭的本地端口（必然连不上，
走确定性阈值路径），MQTT 默认不开。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from infusion_monitor.adapters.nurse_exec import (
    OUTBOUND_EVENT_SCHEMA,
    RETRYABLE_FAILURE_KINDS,
    NurseExecClient,
)
from infusion_monitor.adapters import writeback as writeback_module
from infusion_monitor.api.service import create_app
from infusion_monitor.config import Settings, reset_settings_cache
from infusion_monitor.store import AuditChain, EventRepository, apply_migrations, connect
from tests.support import BagSpec, NurseExecStub, RackScene, build_rack_zip

TRACKS_URL = "/api/v1/infusion/tracks"
TRANSITION_URL = "/api/v1/infusion/transition"
PICKUPS_URL = "/api/v1/infusion/pickups"
PENDING_URL = "/api/v1/infusion/pending"
SUMMARY_URL = "/api/v1/summary"

#: 一份占位但非真实的凭据位（与 .env.example 同形，不含任何院内真实值）。
TOKEN = "e2e-token-not-real"
OPERATOR = "N-EXAMPLE-40"
REPLAY_OPERATOR = "IT-EXAMPLE-40"
RACK_ID = "R-EXAMPLE-40"
BED_NO = "EXAMPLE-BED-40"
ORDER_NO = "EXAMPLE-ORDER-0040"

#: B01 排到近空（会被点名），B02 还满着（不该被点名）。
#: 两袋放在同一段短片里，演练才能证明「只写回被点名的那一袋」，而不是把整架都发出去。
DRILL_SCENE = RackScene(
    bags=(
        BagSpec("B01", 0.30, 0.28, 0.22, 0.03, (92, 148, 196)),
        BagSpec("B02", 0.72, 0.26, 0.80, 0.72, (64, 128, 190)),
    ),
    frame_count=20,
)


def upload_fields(**overrides: Any) -> dict[str, str]:
    """上传的业务头。``slots`` 带滴速文本，让 ETA 有医嘱来源而不是退回 unavailable。"""
    fields = {
        "rack_id": RACK_ID,
        "infusion_order_no": ORDER_NO,
        "bed_no": BED_NO,
        "captured_at": "2026-10-22T09:41:00+08:00",
        "slots": json.dumps(
            [
                {"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-A", "rate_text": "5ml/h"},
                {"bag_seq": "B02", "slot_index": 1, "bag_type_code": "TYPE-A", "rate_text": "5ml/h"},
            ],
            ensure_ascii=False,
        ),
    }
    fields.update(overrides)
    return fields


@dataclass
class Drill:
    """一次演练的现场：应用 + 观察连接 + 替身对端。"""

    client: TestClient
    observer: Any
    stub: NurseExecStub

    # ── 读侧 ────────────────────────────────────────────────────────────
    def session(self, bag_seq: str) -> dict[str, Any]:
        row = self.observer.execute(
            "SELECT * FROM bag_session WHERE rack_id = ? AND bag_seq = ?", (RACK_ID, bag_seq)
        ).fetchone()
        return dict(row) if row else {}

    def events(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.observer.execute("SELECT * FROM infusion_event ORDER BY rowid")
        ]

    def audit_actions(self, session_id: str) -> list[str]:
        return [row["action"] for row in AuditChain(self.observer).list_by_target(session_id)]

    def audit_rows(self, action: str) -> list[dict[str, Any]]:
        """按动作码取审计行。**append-only 正序**（``audit_id`` 升序）——

        退避重试要断言的是「先 1.0s 再 4.0s」的先后次序，所以不能走 ``list_recent``：
        那是给运维看最近动态的，返回的是倒序。
        """
        return AuditChain(self.observer).list_by_action(action)

    # ── 写侧（走真实 HTTP）────────────────────────────────────────────────
    def upload(self, zip_payload: bytes, **overrides: Any) -> dict[str, Any]:
        response = self.client.post(
            TRACKS_URL,
            data=upload_fields(**overrides),
            files={"frames": ("rack.zip", zip_payload, "application/zip")},
        )
        assert response.status_code == 201, response.text
        return response.json()

    def pickup(self) -> dict[str, Any]:
        payload = self.client.get(PICKUPS_URL).json()
        return payload["items"][0] if payload["items"] else {}

    def confirm(self, hint: dict[str, Any], **overrides: Any) -> dict[str, Any]:
        body = {
            "session_id": hint["session_id"],
            "bag_seq": hint["bag_seq"],
            "operator_id": OPERATOR,
            "client_event_id": hint["client_event_id"],
        }
        body.update(overrides)
        response = self.client.post(TRANSITION_URL, json=body)
        assert response.status_code == 200, response.text
        return response.json()

    def summary(self) -> dict[str, Any]:
        return self.client.get(SUMMARY_URL).json()


@pytest.fixture
def drill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Drill]:
    """起替身对端 → 把出站端点指向它 → 建应用（真 SQLite 文件库 + 观察连接）。

    端点**经配置面注入**（``NURSE_EXEC_BASE_URL`` 环境变量），而不是替换
    ``app.state.nurse_exec``：走配置面才能证明「信息科改一个环境变量就换了对端」
    这条路径真的通。模型端点指向一个**关闭的本地端口**，必然连不上——演练不依赖
    外网，也不需要院内模型凭证。
    """
    from tests.conftest import CREDENTIAL_ENV

    with NurseExecStub(status=201) as stub:
        for name, value in CREDENTIAL_ENV.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("NURSE_EXEC_BASE_URL", stub.base_url)
        monkeypatch.setenv("NURSE_EXEC_API_TOKEN", TOKEN)
        monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{_closed_port()}/v1")
        monkeypatch.setenv("MQTT_ENABLED", "false")
        # 后台重试泵不在本演练范围内（task 37）：关掉它，重试次数只由这次投递决定。
        monkeypatch.setenv("WRITEBACK_RETRY_ENABLED", "false")
        reset_settings_cache()

        dsn = f"sqlite:///{tmp_path / 'drill.db'}"
        app = create_app(Settings(), database_url=dsn, frames_root=tmp_path / "frames")
        observer = connect(dsn)
        apply_migrations(observer)
        with TestClient(app) as client:
            yield Drill(client=client, observer=observer, stub=stub)
        observer.close()


def _closed_port() -> int:
    """借一个端口号再立刻放掉：连它必然被拒，且不依赖外网。"""
    import socket

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class _SleepRecorder:
    """记录退避等待而不真睡。

    只替掉 :mod:`adapters.writeback` 里的 ``time``：退避**次数与间隔**照常按产品代码
    的公式算出来并逐条断言，演练不必为此付出 5 秒墙钟。这不是把退避改成假的——
    ``_run_attempts`` 仍然按 ``backoff_seconds`` 算出 1.0 / 4.0 才调到这里。
    """

    def __init__(self) -> None:
        self.delays: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(writeback_module, "time", self)


@pytest.fixture
def backoff_clock(monkeypatch: pytest.MonkeyPatch) -> _SleepRecorder:
    recorder = _SleepRecorder()
    recorder.install(monkeypatch)
    return recorder


def _drill_zip() -> bytes:
    return build_rack_zip(DRILL_SCENE)


# ── 1. 未确认的换液永不出站 ──────────────────────────────────────────────────


def test_unconfirmed_pickup_never_leaves_the_host(drill: Drill):
    """点名只是**建议**。护士没点确认，一次 HTTP 都不能发出去。

    对端收到就会真的换新袋——那一刻病人手上那袋还在输。一次误报若被系统自行签收，
    后果是拔管换袋这件事没人做过。
    """
    body = drill.upload(_drill_zip())
    named = next(bag for bag in body["bags"] if bag["bag_seq"] == "B01")
    assert named["state"] == "needs_change", "演练前提没成立：这一袋没有被点名"
    assert named["trigger_reason"] == "near_empty_observed"

    hint = drill.pickup()
    assert hint["bag_seq"] == "B01", "只有近空的那袋该进点名清单"

    # 只看清单、不点确认：查清单这一层同样不该触发任何出站请求。
    assert drill.client.get(PICKUPS_URL).status_code == 200
    assert drill.stub.call_count == 0, "点名清单这一层发出了出站请求"
    assert drill.events() == [], "未经确认就落了一条换液事件"

    # 顺带证明另一袋没被牵连：B02 还满着，全程 monitoring，不进清单、不出站。
    assert drill.session("B02")["state"] == "monitoring"
    assert [item["bag_seq"] for item in drill.client.get(PICKUPS_URL).json()["items"]] == ["B01"]


# ── 2. 确认换液 → 写回成功，替身收到的字段逐项核对 ───────────────────────────


def test_confirmed_replacement_reaches_the_peer(drill: Drill):
    """上传 → 点名 → 确认 → 替身收到**全量字段**，报头与幂等键逐项核对。"""
    drill.upload(_drill_zip())
    hint = drill.pickup()

    result = drill.confirm(hint)

    assert result["state"] == "replaced"
    assert result["created"] is True and result["idempotent_hit"] is False
    assert result["writeback"]["status"] == "sent"
    assert result["writeback"]["ok"] is True
    assert result["writeback"]["attempts"] == 1, "一次成功就不该有第二次尝试"
    assert drill.stub.call_count == 1

    request = drill.stub.last()
    # 报头：Bearer 凭据来自环境变量配置面；幂等键与请求体同字段同值。
    assert request.authorization == f"Bearer {TOKEN}"
    assert request.headers["content-type"].startswith("application/json")
    assert request.path == "/api/v1/infusion/events"
    assert request.idempotency_key == request.json()["client_event_id"]

    body = request.json()
    schema_names = {spec.name for spec in OUTBOUND_EVENT_SCHEMA}
    assert set(body) <= schema_names, f"出了字段规范之外的字段：{set(body) - schema_names}"
    required = {spec.name for spec in OUTBOUND_EVENT_SCHEMA if spec.required}
    assert required <= set(body), f"必填字段缺了：{required - set(body)}"

    event_row = drill.events()[0]
    session_row = drill.session("B01")
    assert body["event_id"] == event_row["event_id"]
    assert body["client_event_id"] == hint["client_event_id"]
    assert body["session_id"] == session_row["session_id"]
    assert body["event_type"] == "infusion.bag.changed"
    assert body["rack_id"] == RACK_ID and body["bed_no"] == BED_NO
    assert body["infusion_order_no"] == ORDER_NO and body["bag_seq"] == "B01"
    assert body["operator_id"] == OPERATOR
    assert body["trigger_reason"] == "nurse_confirmed"
    # 载荷里不许出现患者姓名/住院号这类可识别明细。
    assert "patient" not in json.dumps(body, ensure_ascii=False).lower()

    # 落库与摘要：写回一次成功就结清，不该在待写回清单里留欠账。
    assert event_row["writeback_status"] == "sent"
    assert event_row["attempt_count"] == 1
    assert TOKEN not in (event_row["last_error"] or "")
    writeback = drill.summary()["writeback"]
    assert writeback["sent_count"] == 1 and writeback["pending_writeback_count"] == 0
    assert writeback["queue_pending_count"] == 0
    assert drill.client.get(PENDING_URL).json()["count"] == 0

    actions = drill.audit_actions(session_row["session_id"])
    assert "state.confirmed" in actions and "writeback.sent" in actions
    assert "writeback.pending" not in actions


def test_uncalibrated_bag_type_omits_volume_fields_outbound(drill: Drill, backoff_clock):
    """未登记袋型标定时，出站载荷里**不能**出现毫升与时点，连 null 都没有。

    对端若把缺省读成 0，就会把「我们没测出来」读成「袋子是空的」。
    """
    drill.upload(_drill_zip())
    hint = drill.pickup()

    result = drill.confirm(hint)

    assert result["writeback"]["status"] == "sent"
    body = drill.stub.last().json()
    assert "estimated_volume_ml" not in body
    assert "eta_minutes" not in body
    assert "estimated_volume_ml" not in json.dumps(body, ensure_ascii=False)


# ── 3. 失败注入：连续 5xx → 退避 2 次 → pending_writeback ────────────────────


def test_injected_5xx_backs_off_twice_then_lands_in_pending(drill: Drill, backoff_clock):
    """对端持续 5xx：首次 + 2 次指数退避重试，耗尽后置 pending_writeback。

    这是「绝不静默丢事件」最关键的一条：退避耗尽**不等于**可以把这笔换液丢掉——
    护士已经换了，平台还不知道。事件必须留在库里、状态可见、摘要里数得出来。
    """
    drill.upload(_drill_zip())
    hint = drill.pickup()
    session_id = hint["session_id"]
    drill.stub.set_default_status(503, body='{"detail":"upstream busy"}')

    result = drill.confirm(hint)

    assert result["writeback"]["status"] == "pending_writeback"
    assert result["writeback"]["ok"] is False
    assert result["writeback"]["attempts"] == 3, "首次 + 2 次重试"
    assert drill.stub.call_count == 3, "替身没真收到三次请求，说明不是走的真 socket"
    # 退避间隔由产品代码的指数公式算出：基数 1s、倍数 4 → 1.0 / 4.0。
    assert backoff_clock.delays == [1.0, 4.0]

    # 事件一行不少，状态与错误摘要都落库。
    stored = drill.events()[0]
    assert len(drill.events()) == 1, "失败的写回把换液事件弄丢了"
    assert stored["writeback_status"] == "pending_writeback"
    assert stored["attempt_count"] == 3
    assert "kind=http_5xx" in (stored["last_error"] or "")
    assert "status=503" in (stored["last_error"] or "")
    assert "counts=http_5xx=3" in (stored["last_error"] or "")
    assert TOKEN not in (stored["last_error"] or ""), "凭据混进了 last_error"

    # 审计链：每次退避重试留一行，最后落一行 pending。
    actions = drill.audit_actions(session_id)
    assert actions.count("writeback.retry") == 2
    assert actions[-1] == "writeback.pending"
    retry_rows = drill.audit_rows("writeback.retry")
    assert [row["detail"]["backoff_seconds"] for row in retry_rows] == [1.0, 4.0]

    # 运行摘要与待写回清单同步：护士长问「今天有几袋换液没写回去」看的就是这里。
    writeback = drill.summary()["writeback"]
    assert writeback["pending_writeback_count"] == 1
    assert writeback["sent_count"] == 0
    assert writeback["queue_by_status"].get("dead_letter") == 1
    pending = drill.client.get(PENDING_URL).json()
    assert pending["count"] == 1 and pending["dead_letter_count"] == 1
    item = pending["items"][0]
    assert item["event_id"] == stored["event_id"]
    assert item["bag_seq"] == "B01" and item["dead_letter_reason"] == "retries_exhausted"
    assert item["attempt_count"] == 3
    assert TOKEN not in json.dumps(item, ensure_ascii=False)

    # 每次尝试都带同一个幂等键：对端即便中途收到过，也只会当成同一条。
    assert {request.idempotency_key for request in drill.stub.requests} == {
        stored["client_event_id"]
    }


def test_injected_business_rejection_is_not_retried(drill: Drill, backoff_clock):
    """4xx 是对端的业务拒绝：同一份载荷、同一把 token 重发只会再被拒一次。"""
    drill.upload(_drill_zip())
    hint = drill.pickup()
    drill.stub.set_default_status(422, body='{"detail":"unknown infusion_order_no"}')

    result = drill.confirm(hint)

    assert result["writeback"]["status"] == "failed"
    assert result["writeback"]["attempts"] == 1
    assert backoff_clock.delays == [], "业务拒绝不该退避重试"
    assert drill.stub.call_count == 1
    stored = drill.events()[0]
    assert stored["writeback_status"] == "failed"
    assert "reason=unknown infusion_order_no" in (stored["last_error"] or "")
    # 业务拒绝与「退避耗尽」在清单里分列：前者要改载荷，后者等对端恢复。
    assert drill.client.get(PENDING_URL).json()["failed_count"] == 1
    assert drill.summary()["writeback"]["failed_count"] == 1


def test_injected_read_timeout_is_retried_then_sent(drill: Drill, backoff_clock):
    """对端慢一拍（读超时）属可重试：退避重试拿到应答就不该留下欠账。"""
    drill.upload(_drill_zip())
    hint = drill.pickup()
    drill.stub.queue_response(201, latency=0.6)   # 客户端 0.2s 超时，替身随后才应答
    drill.stub.set_default_status(201)
    slow = NurseExecClient(
        base_url=f"{drill.stub.base_url}/api/v1/infusion/events",
        api_token=TOKEN,
        timeout=0.2,
    )
    original = drill.client.app.state.nurse_exec
    drill.client.app.state.nurse_exec = slow
    try:
        result = drill.confirm(hint)
    finally:
        slow.close()
        drill.client.app.state.nurse_exec = original

    assert result["writeback"]["status"] == "sent"
    assert result["writeback"]["attempts"] >= 2
    assert drill.stub.call_count >= 2
    assert drill.events()[0]["writeback_status"] == "sent"
    assert drill.client.get(PENDING_URL).json()["count"] == 0


def test_injected_half_close_is_retried_under_the_same_idempotency_key(drill: Drill, backoff_clock):
    """半开连接：对端把连接掐了却没给应答——请求多半**已经送到对端**了。

    这条既是「可重试」，也是「幂等键不可省」的直接证据：重试必须带同一个
    ``client_event_id`` / ``Idempotency-Key``，否则对端会收成两条换液事件。
    """
    drill.upload(_drill_zip())
    hint = drill.pickup()
    drill.stub.queue_half_close(1)
    drill.stub.set_default_status(201)

    result = drill.confirm(hint)

    assert result["writeback"]["status"] == "sent", "重试拿到应答就不该留欠账"
    assert result["writeback"]["attempts"] == 2
    assert drill.stub.call_count == 2
    assert backoff_clock.delays == [1.0], "半开连接只退避重试一次"
    assert "half_open" in RETRYABLE_FAILURE_KINDS
    keys = {request.idempotency_key for request in drill.stub.requests}
    assert keys == {hint["client_event_id"]}, "重试换了幂等键，对端会收成两条"
    assert drill.events()[0]["writeback_status"] == "sent"


# ── 4. 对端恢复后人工重放 ────────────────────────────────────────────────────


def test_replay_after_the_peer_recovers_delivers_the_same_event(drill: Drill, backoff_clock):
    """对端恢复后信息科点重放：送达、**不产生第二条业务事件**、再点一次幂等命中。"""
    drill.upload(_drill_zip())
    hint = drill.pickup()
    drill.stub.set_default_status(503, body='{"detail":"upstream busy"}')
    failed = drill.confirm(hint)
    assert failed["writeback"]["status"] == "pending_writeback"
    assert len(drill.events()) == 1

    # 对端恢复，重放前 pending_writeback 已耗尽退避——人工重放只发一次，不等退避。
    drill.stub.set_default_status(201)
    replay = drill.client.post(
        f"{PENDING_URL}/{failed['event_id']}/replay", json={"operator_id": REPLAY_OPERATOR}
    )
    assert replay.status_code == 200, replay.text
    body = replay.json()
    assert body["replayed"] is True and body["ok"] is True
    assert body["writeback_status"] == "sent"
    assert body["queue_status"] == "sent"
    assert body["client_event_id"] == hint["client_event_id"], "重放必须带同一个幂等键"

    # 对端这次收到的是同一个键、同一个 event_id——它去重后仍是一条。
    request = drill.stub.last()
    assert request.idempotency_key == hint["client_event_id"]
    assert request.json()["event_id"] == failed["event_id"]

    assert len(drill.events()) == 1, "重放写出了第二条业务事件"
    assert drill.events()[0]["writeback_status"] == "sent"
    assert drill.client.get(PENDING_URL).json()["count"] == 0, "已送达的不该留在待办里"
    writeback = drill.summary()["writeback"]
    assert writeback["pending_writeback_count"] == 0 and writeback["sent_count"] == 1

    # 谁补的要留痕。
    replayed = drill.audit_rows("writeback.replayed")
    assert len(replayed) == 1
    assert replayed[0]["detail"]["operator_id"] == REPLAY_OPERATOR
    assert replayed[0]["request_id"]

    # 再点一次：早已送达，一次 HTTP 都不发。
    calls_before = drill.stub.call_count
    again = drill.client.post(
        f"{PENDING_URL}/{failed['event_id']}/replay", json={"operator_id": REPLAY_OPERATOR}
    )
    assert again.status_code == 200, again.text
    assert again.json()["idempotent_hit"] is True
    assert drill.stub.call_count == calls_before, "已送达的事件被重复外发了"
    assert len(drill.events()) == 1


# ── 5. 演练自身的边界 ────────────────────────────────────────────────────────


def test_drill_uses_the_configured_endpoint_not_a_local_shortcut(drill: Drill):
    """出站端点来自 ``NURSE_EXEC_BASE_URL`` 配置面，不是测试替身硬塞进去的对象。

    接挂时信息科改的就是这个环境变量；它不通，改什么都没用。
    """
    client: NurseExecClient = drill.client.app.state.nurse_exec
    assert client.endpoint == f"{drill.stub.base_url}/api/v1/infusion/events"
    assert client.api_token == TOKEN
    assert drill.client.get("/openapi.json").json()["openapi"].startswith("3.1")


def test_drill_writes_no_credential_into_any_observable_surface(drill: Drill, backoff_clock):
    """演练全程跑一遍后，摘要、清单、审计与事件表里都搜不到明文凭据。"""
    drill.upload(_drill_zip())
    hint = drill.pickup()
    drill.stub.set_default_status(500, body='{"detail":"boom"}')
    drill.confirm(hint)

    surfaces = [
        json.dumps(drill.summary(), ensure_ascii=False),
        drill.client.get(PENDING_URL).text,
        json.dumps(drill.events(), ensure_ascii=False, default=str),
        json.dumps(drill.audit_rows("writeback.pending"), ensure_ascii=False, default=str),
    ]
    for text in surfaces:
        assert TOKEN not in text, "明文凭据出现在可观测面上"