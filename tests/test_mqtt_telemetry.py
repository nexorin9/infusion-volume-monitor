"""入站 MQTT 遥测订阅的验收测试。

六组断言：

1. **主题解析**——输液架号取自主题的 ``+`` 段，段数不对/架号为空即拒；
2. **载荷解析**——必填字段、类型、非正滴速；坏消息只拒这一条；
3. **消息派发与去重**——一条合法遥测进缓存；重连后 broker 重投同一份（``ts`` 不变）
   不被二次消费；
4. **通道状态**——连接/断开/坏消息分别写审计，``state()`` 给出运维能一眼看懂的快照；
5. **视觉路径独立可用**——遥测一行没有时，ETA 仍由医嘱速率给出（``eta_source=order_text``）；
6. **断线不拖垮启动**——broker 不可达时 ``start()`` 不抛异常，状态记 disconnected。

本轮**未实连真实 MQTT broker**：回调与去重用注入的假 client 驱动，协议面
（paho 版本、主题、载荷 schema）是真实实现，连接行为由 paho 负责。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from infusion_monitor.adapters.mqtt_telemetry import (
    TelemetryChannel,
    TelemetryRejected,
    parse_rack_from_topic,
    parse_telemetry_payload,
)
from infusion_monitor.calibration.estimator import TelemetryCache, resolve_rate
from infusion_monitor.store import AuditChain, apply_migrations, connect

TOPIC = "hospital/infusion/R-EXAMPLE-12/telemetry"


class _FakeMqttClient:
    """记录 subscribe / 连接动作的假 paho client（不需要真 broker）。"""

    def __init__(self) -> None:
        self.subscribed: list[tuple[str, int]] = []
        self.connected: list[tuple[str, int, int]] = []
        self.loop_started = False
        self.loop_stopped = False
        self.disconnected = False

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
        self.loop_stopped = True

    def disconnect(self) -> None:
        self.disconnected = True


def _message(payload, topic: str = TOPIC):
    raw = payload if isinstance(payload, (bytes, str)) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(topic=topic, payload=raw.encode("utf-8") if isinstance(raw, str) else raw)


def _valid(**overrides) -> dict:
    data = {
        "bag_seq": "B01",
        "pump_rate_ml_per_h": 40.0,
        "device_sn": "PUMP-EXAMPLE-001",
        "ts": "2026-10-22T09:41:00+08:00",
    }
    data.update(overrides)
    return data


@pytest.fixture
def channel():
    cache = TelemetryCache()
    return TelemetryChannel(
        host="mqtt.test-hospital.invalid",
        port=1883,
        username="test-user",
        password="test-password-not-real",
        topic=TOPIC,
        cache=cache,
        client_factory=_FakeMqttClient,
        clock=lambda: "2026-10-22T09:41:05+00:00",
    )


@pytest.fixture
def audit():
    conn = connect("sqlite:///:memory:")
    apply_migrations(conn)
    yield AuditChain(conn)
    conn.close()


# ── 1-2. 解析 ───────────────────────────────────────────────────────────────


def test_rack_id_comes_from_topic_segment():
    assert parse_rack_from_topic(TOPIC) == "R-EXAMPLE-12"
    assert parse_rack_from_topic("hospital/infusion/R-07/telemetry") == "R-07"


@pytest.mark.parametrize(
    "topic", ["telemetry", "hospital/infusion", "hospital/infusion//telemetry/extra/"]
)
def test_malformed_topic_is_rejected(topic):
    with pytest.raises(TelemetryRejected) as caught:
        parse_rack_from_topic(topic)
    assert caught.value.reason == "topic_mismatch"


def test_topic_missing_rack_segment_is_rejected():
    """架号段缺失时**不能**把前缀里的 ``infusion`` 当成架号收下。

    这条是回归用例：早先的实现取「末段往前数第二个非空段」，
    ``hospital/infusion/telemetry`` 会解析出 ``rack_id="infusion"`` 并写进缓存——
    一条看起来完全正常、却永远配不上任何一袋的泵速，排查时没人会想到是主题少段。
    """
    with pytest.raises(TelemetryRejected) as caught:
        parse_rack_from_topic("hospital/infusion/telemetry")
    assert caught.value.reason == "topic_mismatch"
    # 也不能悄悄截成两层主题就收下
    with pytest.raises(TelemetryRejected):
        parse_rack_from_topic("hospital/telemetry")
    # 前缀里的词当架号仍然拒（即使段数凑够）
    with pytest.raises(TelemetryRejected):
        parse_rack_from_topic("infusion/R-01/telemetry")


def test_topic_prefix_mismatch_is_rejected():
    """换了 ``MQTT_TOPIC_PREFIX`` 的院区不该收到别的院区主题的遥测。"""
    prefix = "ward-a/infusion"
    assert parse_rack_from_topic("ward-a/infusion/R-09/telemetry", prefix) == "R-09"
    with pytest.raises(TelemetryRejected) as caught:
        parse_rack_from_topic("hospital/infusion/R-09/telemetry", prefix)
    assert caught.value.reason == "topic_mismatch"


def test_channel_uses_its_own_topic_prefix():
    """通道按自己的主题前缀解析（多院区/多病区部署不该互相吃对方的遥测）。"""
    channel = TelemetryChannel(cache=TelemetryCache(), prefix="ward-a/infusion")
    assert channel.topic == "ward-a/infusion/+/telemetry"
    channel.on_message(None, None, _message(_valid(), topic="ward-a/infusion/R-09/telemetry"))
    assert channel.received == 1 and channel.rejected == 0
    # 默认前缀下的主题对这个通道是陌生主题
    channel.on_message(None, None, _message(_valid()))
    assert channel.rejected == 1
    assert channel.cache.get("R-09", "B01").rate_ml_per_h == 40.0


def test_valid_payload_is_parsed_with_all_fields():
    reading = parse_telemetry_payload(TOPIC, _valid())
    assert reading.rack_id == "R-EXAMPLE-12"
    assert reading.bag_seq == "B01"
    assert reading.pump_rate_ml_per_h == 40.0
    assert reading.device_sn == "PUMP-EXAMPLE-001"
    assert reading.ts == "2026-10-22T09:41:00+08:00"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"pump_rate_ml_per_h": 40.0}, "field_missing"),
        ({"bag_seq": "B01"}, "field_missing"),
        (_valid(pump_rate_ml_per_h="快"), "rate_not_number"),
        (_valid(pump_rate_ml_per_h=0), "rate_not_positive"),
        (_valid(pump_rate_ml_per_h=-5), "rate_not_positive"),
        ("{不是 json", "payload_not_json"),
        (b"\xff\xfe", "payload_not_utf8"),
        (json.dumps([1, 2, 3]), "payload_not_object"),
    ],
)
def test_bad_payloads_are_rejected_with_reason(payload, reason):
    """泵网关固件版本不一，坏消息是常事——每条都要有明确的拒绝理由。"""
    with pytest.raises(TelemetryRejected) as caught:
        parse_telemetry_payload(TOPIC, payload)
    assert caught.value.reason == reason


# ── 3. 派发与去重 ───────────────────────────────────────────────────────────


def test_message_flows_into_cache(channel):
    channel.on_message(None, None, _message(_valid()))

    sample = channel.cache.get("R-EXAMPLE-12", "B01")
    assert sample is not None and sample.rate_ml_per_h == 40.0
    assert channel.received == 1
    assert channel.state()["last_message_at"] == "2026-10-22T09:41:05+00:00"


def test_replayed_message_is_not_consumed_twice(channel):
    """断线重连后 broker 重投同一份遥测：消费两次会让「泵速一直在跳」成立。"""
    channel.on_message(None, None, _message(_valid()))
    channel.on_message(None, None, _message(_valid()))

    assert channel.received == 1
    assert channel.duplicates == 1
    assert channel.cache.count() == 1


def test_newer_sample_replaces_older(channel):
    """按 ts 去重：新的那条进来，旧的被顶掉（护士改滴速后遥测要能反映出来）。"""
    channel.on_message(None, None, _message(_valid(pump_rate_ml_per_h=40.0)))
    channel.on_message(
        None, None, _message(_valid(pump_rate_ml_per_h=20.0, ts="2026-10-22T09:46:00+08:00"))
    )

    assert channel.received == 2
    assert channel.cache.rate_ml_per_h("R-EXAMPLE-12", "B01") == 20.0


def test_bad_message_does_not_stop_later_messages(channel, audit):
    channel._audit = audit
    channel.on_message(None, None, _message({"bag_seq": "B01"}))
    channel.on_message(None, None, _message(_valid()))

    assert channel.rejected == 1
    assert channel.received == 1, "一条坏消息不该把整条订阅通道带掉"
    assert channel.cache.count() == 1


# ── 4. 通道状态与审计 ───────────────────────────────────────────────────────


def test_connect_subscribes_topic_and_audits(channel, audit):
    channel._audit = audit
    channel.start()
    client = channel._client

    channel.on_connect(client, None, None, 0)

    assert client.subscribed == [(TOPIC, 1)]
    assert channel.state()["state"] == "connected"
    actions = [row["action"] for row in audit.list_by_target(TOPIC)]
    assert "telemetry.channel_up" in actions


def test_failed_connect_is_recorded_without_raising(channel, audit):
    channel._audit = audit
    channel.start()
    client = channel._client

    channel.on_connect(client, None, None, 5)

    assert channel.state()["state"] == "disconnected"
    assert channel.state()["last_error"] == "connect_return_code=5"
    actions = [row["action"] for row in audit.list_by_target(TOPIC)]
    assert "telemetry.channel_down" in actions


def test_disconnect_audits_and_flips_state(channel, audit):
    channel._audit = audit
    channel.start()
    channel.on_connect(channel._client, None, None, 0)

    channel.on_disconnect(channel._client, None, 0, 7)

    assert channel.state()["state"] == "disconnected"
    assert channel.connected is False
    rows = [row for row in audit.list_by_target(TOPIC) if row["action"] == "telemetry.channel_down"]
    assert rows and rows[-1]["detail"]["reason"]


def test_rejected_message_is_audited(channel, audit):
    channel._audit = audit
    channel.on_message(None, None, _message({"bag_seq": "B01"}, topic=TOPIC))

    rows = [row for row in audit.list_by_target(TOPIC) if row["action"] == "telemetry.message_rejected"]
    assert len(rows) == 1
    assert rows[0]["outcome"] == "rejected"
    assert rows[0]["detail"]["reason"] == "field_missing"
    assert rows[0]["detail"]["missing"] == ["pump_rate_ml_per_h"]


# ── 5. 视觉路径独立可用 ─────────────────────────────────────────────────────


def test_eta_still_comes_from_order_text_without_telemetry(channel):
    """遥测一行没有时，ETA 仍由医嘱速率给出——遥测只是交叉校验，不是前置条件。"""
    assert channel.cache.count() == 0
    resolution = resolve_rate("5ml/h", channel.cache.rate_ml_per_h("R-EXAMPLE-12", "B01"))

    assert resolution.source == "order_text"
    assert resolution.rate_ml_per_h == 5.0
    assert channel.state()["state"] == "idle"


def test_telemetry_is_used_when_order_text_has_no_rate(channel):
    channel.on_message(None, None, _message(_valid(pump_rate_ml_per_h=40.0)))

    resolution = resolve_rate("常规输液", channel.cache.rate_ml_per_h("R-EXAMPLE-12", "B01"))

    assert resolution.source == "mqtt_telemetry"
    assert resolution.rate_ml_per_h == 40.0


# ── 6. 启动与停止 ───────────────────────────────────────────────────────────


def test_start_is_safe_when_broker_is_unreachable(channel):
    """连不上不抛异常：服务照常起来，遥测状态由 state() 如实汇报。"""
    assert channel.start() is False or channel.state()["state"] in ("connecting", "connected")
    assert channel._client is not None
    channel.stop()
    assert channel.state()["state"] == "stopped"


def test_start_failure_is_audited_not_raised(audit):
    class _ExplodingClient(_FakeMqttClient):
        def connect_async(self, *args, **kwargs):
            raise OSError("no route to host")

    broken = TelemetryChannel(
        host="mqtt.invalid", port=1883, topic=TOPIC,
        client_factory=_ExplodingClient, audit=audit,
    )

    assert broken.start() is False
    assert broken.state()["state"] == "disconnected"
    assert "no route to host" in broken.state()["last_error"]
    rows = [row for row in audit.list_by_target(TOPIC) if row["action"] == "telemetry.channel_down"]
    assert rows and rows[-1]["detail"]["stage"] == "start"


def test_settings_derive_topic_and_credentials(credential_env):
    """凭据与主题都来自配置面（环境变量），模块里不出现明文。"""
    from infusion_monitor.config import Settings

    settings = Settings()
    channel = TelemetryChannel(settings, TelemetryCache())

    assert channel.topic == "hospital/infusion/+/telemetry"
    assert channel.host == settings.mqtt_host
    assert channel.port == settings.mqtt_port
    # 密码只在建立连接时使用，不出现在 state() 这类运维快照里
    assert "test-password-not-real" not in json.dumps(channel.state(), ensure_ascii=False)