"""入站遥测韧性：MQTT 断线重连、跨通道去重、webhook 兜底、遥测缺失时视觉主路径仍可用。

四组断言：

1. **断线重连**——指数退避的形状、退避窗口内不再打扰 broker、稳定连接后清零、
   「调用返回了」不等于「连上了」、重试上限、每次尝试都落审计；
2. **去重**——重连后 broker 重投的同一份遥测（哪怕已经换了输液架）、webhook 与
   MQTT 双通道送来的同一份、`ts` 缺失固件的载荷摘要兜底、乱序不覆盖新值；
3. **webhook 兜底通道**——未配密钥就没有端点、验签在解析之前、时间戳进签名串、
   重放窗口、架号不一致整条拒收、重复投递幂等；
4. **遥测缺失时视觉主路径仍可用**——一条遥测都没有，余量与 ETA 照常给出来，
   `telemetry_state=absent` 如实报出；新窗口内的遥测报 fresh，超窗报 stale。

本轮**未实连真实 MQTT broker 与真实院内泵网关**：重连回调由注入的假 client 驱动，
webhook 走真 HTTP（TestClient 发真请求、真中间件、真异常处理），签名算法与协议面
是真实实现。
"""

from __future__ import annotations

import csv
import io
import json
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from infusion_monitor.adapters.mqtt_telemetry import (
    DEFAULT_TOPIC_PREFIX,
    PENDING_AUDIT_MAX,
    ReconnectPolicy,
    TelemetryChannel,
    TelemetryDedup,
    classify_sample,
    dedup_key,
)
from infusion_monitor.adapters.webhook_telemetry import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    WebhookTelemetryReceiver,
    contract_markdown,
    sign_payload,
    verify_signature,
)
from infusion_monitor.calibration.estimator import TelemetrySample
from infusion_monitor.store import AuditChain, apply_migrations, connect
from tests.support import synth

TOPIC = "hospital/infusion/R-EXAMPLE-39/telemetry"
RACK = "R-EXAMPLE-39"
SECRET = "webhook-secret-not-real"

CONTRACT_DOC = Path(__file__).resolve().parent.parent / "docs" / "webhook-contract.md"
SUMMARY_URL = "/api/v1/summary"
TRACKS_URL = "/api/v1/infusion/tracks"


# ── 夹具 ────────────────────────────────────────────────────────────────────

class _FakeMqttClient:
    """记录连接动作的假 paho client（不需要真 broker）。"""

    def __init__(self) -> None:
        self.connected: list[tuple[str, int, int]] = []
        self.reconnects: list[int] = []
        self.subscribed: list[tuple[str, int]] = []
        self.loop_started = False
        self.fail_next_reconnect: Exception | None = None

    def username_pw_set(self, username: str, password: str) -> None:
        self.username = username

    def reconnect_delay_set(self, min_delay: int = 1, max_delay: int = 120) -> None:
        self.reconnect_window = (min_delay, max_delay)

    def subscribe(self, topic: str, qos: int = 0) -> tuple[int, int]:
        self.subscribed.append((topic, qos))
        return (0, qos)

    def connect_async(self, host: str, port: int, keepalive: int = 60) -> None:
        self.connected.append((host, port, keepalive))

    def loop_start(self) -> None:
        self.loop_started = True

    def loop_stop(self) -> None:
        self.loop_started = False

    def disconnect(self) -> None:
        return None

    def reconnect(self) -> None:
        if self.fail_next_reconnect is not None:
            error, self.fail_next_reconnect = self.fail_next_reconnect, None
            raise error
        self.reconnects.append(len(self.reconnects))


class _Clock:
    """可控的单调时钟：重连退避要按真实时间推进，用真时钟测就得 sleep。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def audit():
    connection = connect("sqlite:///:memory:")
    apply_migrations(connection)
    yield AuditChain(connection)
    connection.close()


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def channel(clock, audit):
    """未启动的 MQTT 通道：假 client + 假单调时钟 + 真审计链。"""
    client_holder: dict[str, _FakeMqttClient] = {}

    def factory() -> _FakeMqttClient:
        client_holder["client"] = _FakeMqttClient()
        return client_holder["client"]

    chan = TelemetryChannel(
        host="mqtt.test.invalid",
        port=1883,
        prefix=DEFAULT_TOPIC_PREFIX,
        client_factory=factory,
        audit=audit,
        monotonic=clock,
        clock=lambda: "2026-10-23T09:41:00+08:00",
    )
    chan._fake_client_holder = client_holder  # 测试侧取用假 client
    return chan


def _payload(**overrides) -> dict:
    base = {
        "bag_seq": "B01",
        "pump_rate_ml_per_h": 40.0,
        "device_sn": "PUMP-EXAMPLE-1",
        "ts": "2026-10-23T09:40:00+08:00",
    }
    base.update(overrides)
    return base


def _message(payload, topic: str = TOPIC):
    return SimpleNamespace(topic=topic, payload=json.dumps(payload).encode("utf-8"))


# ── 组 1：断线重连 ─────────────────────────────────────────────────────────

def test_backoff_is_exponential_and_capped():
    policy = ReconnectPolicy(base_seconds=1.0, factor=2.0, max_seconds=32.0)
    assert [policy.delay_for(n) for n in range(1, 8)] == [1, 2, 4, 8, 16, 32, 32]


def test_supervisor_reconnects_with_growing_backoff(channel, clock):
    channel.start()
    client = channel._fake_client_holder["client"]
    channel.on_disconnect(client, None, None, 0)

    delays = []
    for _ in range(6):
        verdict = channel.supervise_once()
        assert verdict.action == "reconnect", verdict
        assert verdict.issued is True
        delays.append(verdict.delay_seconds)
        clock.advance(verdict.delay_seconds + 0.1)  # 走完退避窗口再来一次

    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    assert len(client.reconnects) == 6
    assert channel.reconnect.attempts == 6
    assert channel.reconnect.as_dict()["backoff_delays"] == delays


def test_no_reconnect_attempt_inside_the_backoff_window(channel, clock):
    channel.start()
    client = channel._fake_client_holder["client"]
    channel.on_disconnect(client, None, None, 0)
    first = channel.supervise_once()
    assert first.delay_seconds == 1.0

    # 退避还没走完：不能再敲 broker
    verdict = channel.supervise_once()
    assert verdict.action == "wait"
    assert verdict.reason == "backoff"
    assert len(client.reconnects) == 1

    clock.advance(1.0)
    assert channel.supervise_once().action == "reconnect"
    assert len(client.reconnects) == 2


def test_reconnect_failure_is_recorded_not_raised(channel):
    channel.start()
    client = channel._fake_client_holder["client"]
    channel.on_disconnect(client, None, None, 0)
    client.fail_next_reconnect = ConnectionResetError("broker refused")

    verdict = channel.supervise_once()  # 不抛异常：连不上是常态

    assert verdict.action == "reconnect"
    assert verdict.issued is False
    assert verdict.reason == "error"
    assert "broker refused" in channel.state()["last_error"]
    assert channel.state()["reconnect"]["last_outcome"] == "error"


def test_issued_reconnect_is_not_reported_as_connected(channel, clock):
    """``reconnect()`` 没抛异常不等于连上了。状态只能由 on_connect 回调改。"""
    channel.start()
    client = channel._fake_client_holder["client"]
    channel.on_disconnect(client, None, None, 0)
    channel.supervise_once()
    assert channel.state()["state"] == "disconnected"

    clock.advance(2.0)
    channel.supervise_once()
    channel.on_connect(client, None, None, 0)
    assert channel.state()["state"] == "connected"
    assert channel.reconnect.last_outcome == "connected"
    # 连上 ≠ 恢复：退避计数要等「这条连接稳定站住」之后才清零（见下一次断开）
    assert channel.reconnect.attempts == 2
    clock.advance(301.0)
    channel.on_disconnect(client, None, None, 0)
    assert channel.reconnect.attempts == 0


def test_stable_connection_resets_backoff(channel, clock):
    policy = ReconnectPolicy(reset_after_seconds=300.0)
    channel.reconnect.policy = policy
    channel.start()
    client = channel._fake_client_holder["client"]
    channel.on_disconnect(client, None, None, 0)
    for _ in range(3):
        channel.supervise_once()
        clock.advance(99.0)

    channel.on_connect(client, None, None, 0)   # 连上
    clock.advance(301.0)                        # 连稳超过 reset_after
    channel.on_disconnect(client, None, None, 0)

    assert channel.reconnect.attempts == 0
    verdict = channel.supervise_once()
    assert verdict.delay_seconds == 1.0  # 从基准重新开始，而不是接着 8s 退


def test_flapping_connection_keeps_escalating(channel, clock):
    """「连上又掉」在稳定窗口内不算恢复，退避必须继续涨。"""
    channel.reconnect.policy = ReconnectPolicy(reset_after_seconds=300.0)
    channel.start()
    client = channel._fake_client_holder["client"]

    delays = []
    for _ in range(4):
        channel.on_disconnect(client, None, None, 0)
        verdict = channel.supervise_once()
        delays.append(verdict.delay_seconds)
        channel.on_connect(client, None, None, 0)  # 刚连上（不足 300s）
        clock.advance(10.0)

    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_reconnect_stops_after_max_attempts(channel, clock):
    channel.reconnect.policy = ReconnectPolicy(max_attempts=2)
    channel.start()
    client = channel._fake_client_holder["client"]
    channel.on_disconnect(client, None, None, 0)

    channel.supervise_once()
    clock.advance(5.0)
    channel.supervise_once()
    clock.advance(60.0)
    verdict = channel.supervise_once()

    assert verdict.action == "wait"
    assert verdict.reason == "attempts_exhausted"
    assert len(client.reconnects) == 2
    assert channel.state()["reconnect"]["exhausted"] is True


def test_client_is_rebuilt_when_the_first_connect_never_succeeded(channel, clock):
    """start() 就失败时（_client 为 None），重连要自己把 client 重建出来。"""
    channel.start()
    channel._client = None
    channel._state = "disconnected"

    verdict = channel.supervise_once()

    assert verdict.action == "reconnect"
    assert channel._client is not None
    assert channel._fake_client_holder["client"].connected


def test_every_reconnect_attempt_is_audited(channel, audit):
    channel.start()
    client = channel._fake_client_holder["client"]
    channel.on_disconnect(client, None, None, 0)
    channel.supervise_once()

    chain = audit._conn.execute(  # noqa: SLF001 —— 测试侧直接读审计表
        "SELECT action, outcome, detail FROM audit_log WHERE action = ?",
        ("telemetry.reconnect_attempted",),
    ).fetchall()
    assert len(chain) == 1
    assert chain[0]["outcome"] == "accepted"
    detail = json.loads(chain[0]["detail"])
    assert detail["attempt"] == 1
    assert detail["delay_seconds"] == 1.0
    assert detail["issued"] is True


def test_supervisor_thread_starts_and_stops(channel):
    thread = channel.start_supervisor(interval_seconds=0.01)
    assert thread.is_alive()
    channel.stop_supervisor(timeout=1.0)
    assert not thread.is_alive()


def test_audit_from_another_thread_is_queued_not_written_directly(channel, audit):
    """paho 回调与重连线程不在事件循环线程上，不能直接写主连接。

    这条不是洁癖：SQLite 连接带线程亲和性，异线程直接写会抛
    ``ProgrammingError``，而这个异常是从 paho 的网络循环线程里抛出去的——
    线程就此死掉，遥测从此静默消失，连一行审计都留不下，比不写还难查。
    正确形态是「异线程入队 → 事件循环线程落库」，本用例钉的就是这两拍。
    """
    def rows_for(action: str) -> int:
        return audit._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM audit_log WHERE action = ?", (action,)
        ).fetchone()[0]

    errors: list[BaseException] = []

    def _from_another_thread() -> None:
        try:
            channel.on_disconnect(None, None, None, 7)
            channel.supervise_once()
        except BaseException as exc:  # noqa: BLE001 —— 用例要看见任何异常
            errors.append(exc)

    thread = threading.Thread(target=_from_another_thread, name="fake-paho-loop")
    thread.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert errors == []                      # 异线程不得抛
    assert rows_for("telemetry.channel_down") == 0   # 还没落库
    stats = channel.pending_audit_stats()
    assert stats["queued"] == 2              # 断开 + 一次重连尝试
    assert stats["dropped"] == 0

    assert channel.flush_pending_audit() == 2
    assert channel.pending_audit_stats()["queued"] == 0
    assert rows_for("telemetry.channel_down") == 1
    assert rows_for("telemetry.reconnect_attempted") == 1


def test_pending_audit_buffer_is_bounded_and_reports_drops(channel, audit):
    """缓冲满了丢最老的一条并计数——丢多少摆在摘要里，不静默。"""
    def _fill() -> None:
        for index in range(PENDING_AUDIT_MAX + 3):
            channel._audit_append(
                "telemetry.channel_down", "accepted", {"seq": index}
            )

    thread = threading.Thread(target=_fill, name="fake-paho-loop-flood")
    thread.start()
    thread.join(timeout=10.0)

    stats = channel.pending_audit_stats()
    assert stats["queued"] == PENDING_AUDIT_MAX
    assert stats["dropped"] == 3
    assert stats["max_queued"] == PENDING_AUDIT_MAX
    assert channel.flush_pending_audit() == PENDING_AUDIT_MAX
    # 留下的必须是**最新**的那几条：留档里「刚才还在断」比「刚才断过」有用
    kept = sorted(
        json.loads(row["detail"])["seq"]
        for row in audit._conn.execute(  # noqa: SLF001
            "SELECT detail FROM audit_log WHERE action = ?", ("telemetry.channel_down",)
        ).fetchall()
    )
    assert kept == list(range(3, PENDING_AUDIT_MAX + 3))


def test_health_flushes_the_queue_and_summary_reports_it(client):
    """运维轮询的这两个端点就是落库时机：health 静默落库，summary 如实报条数。"""
    channel = client.app.state.telemetry_channel
    channel._pending_audit.append(
        {"action": "telemetry.channel_down", "outcome": "accepted", "detail": {"stage": "test"}}
    )

    assert channel.pending_audit_stats()["queued"] == 1
    assert client.get("/api/v1/health").json()["status"] == "ok"
    assert channel.pending_audit_stats()["queued"] == 0

    channel._pending_audit.append(
        {"action": "telemetry.reconnect_attempted", "outcome": "accepted",
         "detail": {"attempt": 1}}
    )
    payload = client.get(SUMMARY_URL).json()
    assert payload["runtime"] is None or payload["runtime"] is not None  # 摘要不依赖运行计数
    assert payload["flushed_channel_audit"] == 1
    assert payload["channels"]["mqtt"]["pending_audit"]["queued"] == 0
    export = client.get(
        "/api/v1/infusion/audit", params={"action": "telemetry.reconnect_attempted"}
    )
    assert export.headers["X-Audit-Row-Count"] == "1"


# ── 组 2：去重（重连重投 + 跨通道）────────────────────────────────────────

def test_redelivery_after_reconnect_is_dropped_even_on_a_new_rack(channel):
    """broker 重投的是上一次会话的消息，重连时输液架可能已经换了。

    缓存按 (rack, bag) 比，架号一变就完全比不出来——这正是要按 (device_sn, ts)
    去重的原因：同一台泵同一时刻的读数只有一份事实。
    """
    first = channel.submit(TOPIC, _payload(), source="mqtt")
    assert first.accepted and first.reason == "accepted"

    replay_topic = "hospital/infusion/R-EXAMPLE-99/telemetry"
    replay = channel.submit(replay_topic, _payload(), source="mqtt")

    assert replay.accepted is False
    assert replay.reason == "duplicate"
    assert replay.basis == "device_ts"
    assert channel.received == 1
    assert channel.duplicates == 1
    assert channel.cache.count() == 1
    assert channel.cache.get(RACK, "B01") is not None
    assert channel.cache.get("R-EXAMPLE-99", "B01") is None


def test_same_reading_from_two_channels_counts_once(channel):
    """webhook 兜底与 MQTT 指向同一个泵网关时，同一条遥测只算一条。"""
    via_mqtt = channel.submit(TOPIC, _payload(), source="mqtt")
    via_webhook = channel.submit(TOPIC, _payload(), source="webhook")

    assert via_mqtt.accepted is True
    assert via_webhook.reason == "duplicate"
    assert channel.received == 1
    assert channel.cache.get(RACK, "B01").source == "mqtt"  # 先到的那条为准


def test_undated_firmware_dedups_by_payload_digest(channel):
    """有的泵固件不发 ts：内容完全一致的两条按重复处理。"""
    first = channel.submit(TOPIC, {"bag_seq": "B01", "pump_rate_ml_per_h": 40.0}, source="mqtt")
    same = channel.submit(TOPIC, {"bag_seq": "B01", "pump_rate_ml_per_h": 40.0}, source="webhook")
    changed = channel.submit(TOPIC, {"bag_seq": "B01", "pump_rate_ml_per_h": 30.0}, source="webhook")

    assert first.accepted and first.basis == "payload_digest"
    assert same.reason == "duplicate"
    assert changed.accepted is True
    assert channel.received == 2
    assert channel.state()["dedup"]["by_basis"]["payload_digest"] == 1


def test_newer_telemetry_is_not_dropped(channel):
    older = channel.submit(TOPIC, _payload(ts="2026-10-23T09:30:00+08:00"), source="mqtt")
    newer = channel.submit(TOPIC, _payload(ts="2026-10-23T09:40:00+08:00"), source="mqtt")

    assert older.accepted and newer.accepted
    assert channel.duplicates == 0
    assert channel.cache.rate_ml_per_h(RACK, "B01") == 40.0


def test_out_of_order_telemetry_does_not_overwrite_the_newer_reading(channel):
    channel.submit(TOPIC, _payload(ts="2026-10-23T09:40:00+08:00"), source="mqtt")
    late = channel.submit(
        TOPIC, _payload(pump_rate_ml_per_h=20.0, ts="2026-10-23T09:30:00+08:00"), source="mqtt"
    )

    assert late.accepted is False
    assert late.reason == "out_of_order"
    assert channel.duplicates == 0  # 乱序不是重投，两个计数不能混
    assert channel.cache.rate_ml_per_h(RACK, "B01") == 40.0


def test_dedup_table_is_bounded_and_the_cache_still_guards_the_evicted_entry():
    """去重表挤掉老条目后，仍由缓存层的 ts 比较兜底——两道闸是独立的。"""
    dedup = TelemetryDedup(max_entries=2)
    channel = TelemetryChannel(
        prefix=DEFAULT_TOPIC_PREFIX, clock=lambda: "2026-10-23T09:41:00+08:00"
    )
    channel.dedup = dedup

    stamps = ("2026-10-23T09:38:00+08:00", "2026-10-23T09:39:00+08:00", "2026-10-23T09:40:00+08:00")
    for stamp in stamps:
        assert channel.submit(TOPIC, _payload(ts=stamp), source="mqtt").accepted is True
    assert dedup.size() == 2  # 最老的一条被挤出去了

    replay_old = channel.submit(TOPIC, _payload(ts=stamps[0]), source="mqtt")
    assert replay_old.reason == "out_of_order"  # 去重表认不得，缓存层仍挡得住
    assert channel.cache.rate_ml_per_h(RACK, "B01") == 40.0
    assert channel.received == 3
    assert dedup.as_dict()["entries"] == 2


def test_duplicate_drop_is_audited(channel, audit):
    channel.submit(TOPIC, _payload(), source="mqtt")
    channel.submit(TOPIC, _payload(), source="mqtt")

    rows = audit._conn.execute(  # noqa: SLF001
        "SELECT action, detail FROM audit_log WHERE action = ?", ("telemetry.duplicate_dropped",)
    ).fetchall()
    assert len(rows) == 1
    detail = json.loads(rows[0]["detail"])
    assert detail["basis"] == "device_ts"
    assert detail["device_sn"] == "PUMP-EXAMPLE-1"


def test_dedup_key_falls_back_to_rack_bag_without_device_sn():
    from infusion_monitor.adapters.mqtt_telemetry import TelemetryReading

    reading = TelemetryReading(rack_id="R-1", bag_seq="B01", pump_rate_ml_per_h=40.0, ts="t")
    key, basis = dedup_key(reading)
    assert key == ("R-1:B01", "t")
    assert basis == "device_ts"


# ── 组 3：webhook 兜底通道 ─────────────────────────────────────────────────

def _receiver(channel, audit=None, **kwargs) -> WebhookTelemetryReceiver:
    return WebhookTelemetryReceiver(
        channel, secret=kwargs.pop("secret", SECRET), audit=audit, **kwargs
    )


def _signed(body: bytes, *, secret: str = SECRET, timestamp: str = "") -> dict[str, str]:
    headers = {}
    if secret:
        headers[SIGNATURE_HEADER] = sign_payload(secret, body, timestamp)
    if timestamp:
        headers[TIMESTAMP_HEADER] = timestamp
    return headers


def test_endpoint_does_not_exist_without_a_secret(client):
    response = client.post(
        f"/api/v1/infusion/telemetry/webhook/{RACK}", content=json.dumps(_payload())
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["error_code"] == "NOT_FOUND"
    assert detail["issues"][0]["code"] == "webhook_disabled"
    assert "WEBHOOK_SIGNING_SECRET" in detail["issues"][0]["suggestion"]


def test_valid_webhook_is_accepted_and_lands_in_the_cache(client_factory):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        body = json.dumps(_payload()).encode("utf-8")
        response = client.post(
            f"/api/v1/infusion/telemetry/webhook/{RACK}",
            content=body,
            headers={**_signed(body), "content-type": "application/json"},
        )
        assert response.status_code == 202
        payload = response.json()
        assert payload["accepted"] is True
        assert payload["duplicate"] is False
        assert payload["rack_id"] == RACK  # 架号取自路径段
        assert payload["bag_seq"] == "B01"
        assert payload["source"] == "webhook"
        cache = client.app.state.telemetry
        assert cache.rate_ml_per_h(RACK, "B01") == 40.0
        assert cache.get(RACK, "B01").source == "webhook"


def test_duplicate_delivery_is_idempotent(client_factory):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        body = json.dumps(_payload()).encode("utf-8")
        url = f"/api/v1/infusion/telemetry/webhook/{RACK}"
        headers = {**_signed(body), "content-type": "application/json"}
        first = client.post(url, content=body, headers=headers)
        second = client.post(url, content=body, headers=headers)

        assert first.status_code == second.status_code == 202
        assert second.json()["duplicate"] is True
        assert second.json()["accepted"] is False
        assert client.app.state.telemetry_channel.received == 1
        assert client.app.state.telemetry_channel.duplicates == 1


@pytest.mark.parametrize(
    "signature,reason",
    [
        (None, "signature_missing"),
        ("", "signature_missing"),
        ("sha256=not-hex", "signature_malformed"),
        ("md5=" + "a" * 64, "signature_malformed"),
        ("a" * 63, "signature_malformed"),
    ],
)
def test_bad_signatures_are_rejected_before_parsing(client_factory, signature, reason):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        body = json.dumps(_payload()).encode("utf-8")
        headers = {"content-type": "application/json"}
        if signature is not None:
            headers[SIGNATURE_HEADER] = signature
        response = client.post(
            f"/api/v1/infusion/telemetry/webhook/{RACK}", content=body, headers=headers
        )

        assert response.status_code == 401
        detail = response.json()["detail"]
        assert detail["error_code"] == "WEBHOOK_SIGNATURE_INVALID"
        assert detail["issues"][0]["code"] == reason
        assert client.app.state.telemetry.count() == 0  # 验签在解析之前


def test_wrong_secret_and_tampered_body_are_rejected(client_factory):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        url = f"/api/v1/infusion/telemetry/webhook/{RACK}"
        body = json.dumps(_payload()).encode("utf-8")

        wrong_secret = client.post(
            url,
            content=body,
            headers={**_signed(body, secret="another-secret"), "content-type": "application/json"},
        )
        assert wrong_secret.status_code == 401
        assert wrong_secret.json()["detail"]["error_code"] == "WEBHOOK_SIGNATURE_INVALID"

        tampered = body.replace(b"40.0", b"99.0")
        signed_then_changed = client.post(
            url,
            content=tampered,
            headers={**_signed(body), "content-type": "application/json"},
        )
        assert signed_then_changed.status_code == 401
        assert client.app.state.telemetry.count() == 0


def test_replay_window_is_enforced_and_timestamp_is_inside_the_signature(client_factory):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        url = f"/api/v1/infusion/telemetry/webhook/{RACK}"
        now = int(time.time())

        fresh_body = json.dumps(_payload(ts="2026-10-23T09:50:00+08:00")).encode("utf-8")
        fresh = client.post(
            url,
            content=fresh_body,
            headers={**_signed(fresh_body, timestamp=str(now)), "content-type": "application/json"},
        )
        assert fresh.status_code == 202
        assert fresh.json()["timestamp_enforced"] is True

        stale_body = json.dumps(_payload(ts="2026-10-23T09:51:00+08:00")).encode("utf-8")
        stale = client.post(
            url,
            content=stale_body,
            headers={
                **_signed(stale_body, timestamp=str(now - 3600)),
                "content-type": "application/json",
            },
        )
        assert stale.status_code == 401
        assert stale.json()["detail"]["error_code"] == "WEBHOOK_TIMESTAMP_REJECTED"
        assert stale.json()["detail"]["issues"][0]["code"] == "timestamp_out_of_window"

        # 时间戳在签名串里：换了时间戳头，签名就地对不上
        swapped_body = json.dumps(_payload(ts="2026-10-23T09:52:00+08:00")).encode("utf-8")
        headers = _signed(swapped_body, timestamp=str(now))
        headers[TIMESTAMP_HEADER] = str(now - 1)
        swapped = client.post(url, content=swapped_body, headers={**headers, "content-type": "application/json"})
        assert swapped.status_code == 401


def test_rack_id_in_payload_must_match_the_path(client_factory):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        body = json.dumps(_payload(rack_id="R-SOMEONE-ELSE")).encode("utf-8")
        response = client.post(
            f"/api/v1/infusion/telemetry/webhook/{RACK}",
            content=body,
            headers={**_signed(body), "content-type": "application/json"},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error_code"] == "TELEMETRY_REJECTED"
        assert detail["issues"][0]["code"] == "rack_mismatch"
        assert client.app.state.telemetry.count() == 0


def test_payload_with_matching_rack_id_is_accepted(client_factory):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        body = json.dumps(_payload(rack_id=RACK)).encode("utf-8")
        response = client.post(
            f"/api/v1/infusion/telemetry/webhook/{RACK}",
            content=body,
            headers={**_signed(body), "content-type": "application/json"},
        )
        assert response.status_code == 202


@pytest.mark.parametrize(
    "payload_body,reason",
    [
        ({"pump_rate_ml_per_h": 40.0}, "field_missing"),
        ({"bag_seq": "B01"}, "field_missing"),
        ({"bag_seq": "B01", "pump_rate_ml_per_h": 0}, "rate_not_positive"),
        ({"bag_seq": "B01", "pump_rate_ml_per_h": -5}, "rate_not_positive"),
    ],
)
def test_unusable_payloads_are_rejected_one_by_one(client_factory, payload_body, reason):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        body = json.dumps(payload_body).encode("utf-8")
        response = client.post(
            f"/api/v1/infusion/telemetry/webhook/{RACK}",
            content=body,
            headers={**_signed(body), "content-type": "application/json"},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error_code"] == "TELEMETRY_REJECTED"
        assert detail["issues"][0]["code"] == "payload_rejected"
        assert client.app.state.telemetry.count() == 0


def test_oversized_body_is_rejected_with_413(client_factory):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET, WEBHOOK_MAX_BODY_BYTES="256") as client:
        body = json.dumps(_payload(pad="x" * 1024)).encode("utf-8")
        response = client.post(
            f"/api/v1/infusion/telemetry/webhook/{RACK}",
            content=body,
            headers={**_signed(body), "content-type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["detail"]["error_code"] == "WEBHOOK_BODY_TOO_LARGE"


def test_malformed_json_is_rejected(client_factory):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        body = b"{not json"
        response = client.post(
            f"/api/v1/infusion/telemetry/webhook/{RACK}",
            content=body,
            headers={**_signed(body), "content-type": "application/json"},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["issues"][0]["code"] == "payload_not_json"


def test_rejection_is_audited_with_a_reason(client_factory, tmp_path):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET, DATABASE_URL=f"sqlite:///{tmp_path}/a.db") as client:
        body = json.dumps(_payload(rack_id="R-OTHER")).encode("utf-8")
        client.post(
            f"/api/v1/infusion/telemetry/webhook/{RACK}",
            content=body,
            headers={**_signed(body), "content-type": "application/json"},
        )
        # 审计包导出走产品自己的只读端点（而不是从测试线程直接摸 app.state 的连接：
        # 那条连接建在 TestClient 的门户线程上，跨线程用会抛 ProgrammingError，
        # 断言到的就不是审计链而是线程模型了）。
        export = client.get(
            "/api/v1/infusion/audit", params={"action": "telemetry.webhook_rejected"}
        )
        assert export.status_code == 200
        assert export.headers["X-Audit-Row-Count"] == "1"
        rows = list(csv.DictReader(io.StringIO(export.text.lstrip("﻿"))))
        assert len(rows) == 1
        assert rows[0]["结果"] == "rejected"
        assert rows[0]["动作码"] == "telemetry.webhook_rejected"
        assert json.loads(rows[0]["明细(JSON)"])["reason"] == "rack_mismatch"


def test_secret_never_appears_in_a_response(client_factory):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        body = json.dumps(_payload()).encode("utf-8")
        response = client.post(
            f"/api/v1/infusion/telemetry/webhook/{RACK}",
            content=body,
            headers={**_signed(body, secret="wrong-secret"), "content-type": "application/json"},
        )
        assert SECRET not in response.text
        assert "wrong-secret" not in response.text


def test_verify_signature_helper_covers_every_rejection_reason():
    body = b'{"a": 1}'
    good = sign_payload(SECRET, body, "1000")
    assert verify_signature(SECRET, body, good, timestamp="1000", now=1000.0).ok is True
    assert verify_signature("other", body, good, timestamp="1000", now=1000.0).reason == (
        "signature_mismatch"
    )
    assert verify_signature(SECRET, body, None).reason == "signature_missing"
    assert verify_signature(SECRET, body, "sha256=xyz").reason == "signature_malformed"
    assert verify_signature(SECRET, body, good, timestamp="abc", now=1000.0).reason == (
        "timestamp_malformed"
    )
    assert verify_signature(SECRET, body, good, timestamp="1", now=1000.0).reason == (
        "timestamp_out_of_window"
    )
    # 不带时间戳 → 只对 body 验签，但没有重放保护（响应里的 timestamp_enforced=false）。
    # 注意签名要按**不带时间戳**的口径算：签名串是 ts.body，带了时间戳签出来的签名
    # 在这条路径上本来就该对不上（换掉时间戳头签名就失效，正是要守的性质）。
    unsigned_ts = sign_payload(SECRET, body)
    assert verify_signature(SECRET, body, good).ok is False
    assert verify_signature(SECRET, body, unsigned_ts).ok is True
    assert verify_signature(SECRET, body, unsigned_ts).as_dict()["timestamp_enforced"] is False


def test_webhook_rejection_is_audited_by_the_receiver(channel, audit):
    receiver = _receiver(channel, audit=audit)
    receiver.handle(RACK, b"{}", signature=None)
    rows = audit._conn.execute(  # noqa: SLF001
        "SELECT detail FROM audit_log WHERE action = ?", ("telemetry.webhook_rejected",)
    ).fetchall()
    assert json.loads(rows[0]["detail"])["reason"] == "signature_missing"


def test_receiver_state_tracks_activity(channel):
    receiver = _receiver(channel, secret="")
    assert receiver.state()["enabled"] is False
    assert receiver.state()["state"] == "disabled"
    assert receiver.handle(RACK, b"{}", signature=None).http_status == 404

    receiver = _receiver(channel)
    assert receiver.state()["enabled"] is True
    assert receiver.state()["state"] == "idle"

    body = json.dumps(_payload()).encode("utf-8")
    receiver.handle(RACK, body, signature=sign_payload(SECRET, body))
    assert receiver.state()["state"] == "active"
    assert receiver.state()["accepted"] == 1
    assert receiver.state()["last_rack_id"] == RACK


def test_contract_doc_is_in_sync_with_the_code():
    assert CONTRACT_DOC.exists(), "docs/webhook-contract.md 不存在；用 contract_markdown() 生成"
    assert CONTRACT_DOC.read_text(encoding="utf-8") == contract_markdown()


# ── 组 4：遥测缺失时视觉主路径仍可用 ──────────────────────────────────────

def _upload(client, slots, *, rack_id=RACK, captured_at="2026-10-23T09:41:00+08:00"):
    frames = synth.build_rack_zip(synth.default_rack_scene(20))
    return client.post(
        TRACKS_URL,
        data={
            "rack_id": rack_id,
            "infusion_order_no": "EXAMPLE-ORDER-0039",
            "bed_no": "12",
            "captured_at": captured_at,
            "slots": json.dumps(slots, ensure_ascii=False),
        },
        files={"frames": ("frames.zip", frames, "application/zip")},
    )


def _register_type_a_calibration(client) -> None:
    """登记 TYPE-A 标定。

    这组用例验的是**遥测**的门禁语义，不是标定门禁；不登记标定的话
    ``suppression_reason`` 会被 ``uncalibrated`` 先占住，看不到遥测那一条。
    """
    client.post(
        "/api/v1/infusion/calibrations",
        json={
            "bag_type_code": "TYPE-A",
            "capacity_ml": 600,
            "dead_volume_ml": 20,
            "samples": [
                {"measured_fill_ratio": f, "known_volume_ml": f * 600, "source": "称重法"}
                for f in (0.1, 0.5, 0.9)
            ],
            "calibrated_by": "OP-EXAMPLE-1",
        },
    )


def test_visual_path_works_with_zero_telemetry(client):
    """一条遥测都没有：余量曲线照常、ETA 退回医嘱速率、telemetry_state 如实报 absent。"""
    _register_type_a_calibration(client)
    response = _upload(
        client,
        [{"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-A", "rate_text": "5ml/h"}],
    )
    assert response.status_code == 201
    bag = response.json()["bags"][0]
    assert len(bag["area_ratio_series"]) >= 12
    assert bag["telemetry_state"] == "absent"
    assert "telemetry_age_seconds" not in bag
    assert bag["eta_source"] == "order_text"
    assert bag["eta_minutes"] > 0


def test_upload_reports_fresh_and_stale_telemetry(client_factory):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        _register_type_a_calibration(client)
        body = json.dumps(_payload(ts="2026-10-23T09:40:00+08:00")).encode("utf-8")
        client.post(
            f"/api/v1/infusion/telemetry/webhook/{RACK}",
            content=body,
            headers={**_signed(body), "content-type": "application/json"},
        )
        response = _upload(
            client,
            [{"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-A", "rate_text": "5ml/h"}],
        )
        bag = response.json()["bags"][0]
        assert bag["telemetry_state"] == "fresh"
        assert bag["telemetry_source"] == "webhook"
        assert 50 <= bag["telemetry_age_seconds"] <= 70
        # 医嘱 5ml/h 与泵速 40ml/h 差一个数量级 → 判冲突，且不给时点
        assert bag["eta_source"] == "conflict"
        assert "eta_minutes" not in bag
        assert bag["suppression_reason"] == "rate_conflict"

    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        _register_type_a_calibration(client)
        old = json.dumps(_payload(ts="2026-10-23T09:20:00+08:00")).encode("utf-8")
        client.post(
            f"/api/v1/infusion/telemetry/webhook/{RACK}",
            content=old,
            headers={**_signed(old), "content-type": "application/json"},
        )
        bag = _upload(
            client,
            [{"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-A", "rate_text": "5ml/h"}],
        ).json()["bags"][0]
        assert bag["telemetry_state"] == "stale"
        assert bag["telemetry_age_seconds"] >= 1200


def test_telemetry_outage_does_not_break_upload_summary_or_health(client):
    """遥测通道整个挂掉时，探活、摘要、上传三件事都照常。"""
    channel = client.app.state.telemetry_channel
    channel._state = "disconnected"
    channel.on_disconnect(None, None, None, 7)

    assert client.get("/api/v1/health").json()["status"] == "ok"
    payload = client.get(SUMMARY_URL).json()
    assert payload["channels"]["mqtt"]["state"] == "disconnected"
    assert payload["channels"]["telemetry"]["enabled"] is False
    assert _upload(
        client,
        [{"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-A", "rate_text": "5ml/h"}],
    ).status_code == 201


def test_summary_shows_both_channels_after_a_webhook_delivery(client_factory):
    with client_factory(WEBHOOK_SIGNING_SECRET=SECRET) as client:
        body = json.dumps(_payload()).encode("utf-8")
        client.post(
            f"/api/v1/infusion/telemetry/webhook/{RACK}",
            content=body,
            headers={**_signed(body), "content-type": "application/json"},
        )
        channels = client.get(SUMMARY_URL).json()["channels"]
        assert channels["webhook"]["enabled"] is True
        assert channels["webhook"]["state"] == "active"
        assert channels["telemetry"]["enabled"] is True
        assert channels["telemetry"]["state"] == "connected"
        assert channels["telemetry"]["last_message_at"] is not None
        assert channels["mqtt"]["reconnect"]["policy"]["max_seconds"] == 32.0


@pytest.mark.parametrize(
    "sample,reference,expected",
    [
        (None, "2026-10-23T09:41:00+08:00", "absent"),
        (TelemetrySample(40.0, ts="2026-10-23T09:40:30+08:00"), "2026-10-23T09:41:00+08:00", "fresh"),
        (TelemetrySample(40.0, ts="2026-10-23T09:20:00+08:00"), "2026-10-23T09:41:00+08:00", "stale"),
        (TelemetrySample(40.0, ts=""), "2026-10-23T09:41:00+08:00", "undated"),
        (TelemetrySample(40.0, ts="not-a-timestamp"), "2026-10-23T09:41:00+08:00", "undated"),
    ],
)
def test_freshness_classification(sample, reference, expected):
    view = classify_sample(sample, reference=reference, fresh_seconds=300.0)
    assert view.state == expected


def test_freshness_uses_capture_time_not_wall_clock():
    """排队补传的一段短片：拿墙上时钟比会把完全有效的读数判成几小时前的。"""
    sample = TelemetrySample(40.0, ts="2026-10-23T09:40:00+08:00")
    view = classify_sample(sample, reference="2026-10-23T09:41:00+08:00", fresh_seconds=300.0)
    assert view.state == "fresh"
    assert view.age_seconds == pytest.approx(60.0, abs=1.0)