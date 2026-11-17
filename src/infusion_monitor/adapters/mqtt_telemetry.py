"""入站遥测：订阅输液泵泵速（MQTT 主通道 + webhook 兜底共用本模块的解析与去重）。

**遥测是交叉校验，不是主路径。** 护士拍一段短片就能得到余量与 ETA（滴速来自
输注医嘱）；泵速遥测只是让 ETA 多一个独立来源，能发现「医嘱写 40 ml/h、泵上跑
的是 20 ml/h」这类对不上的情况。所以这里的每一条降级路径都必须是「视觉结果照常
给出来」，而不是「遥测没来就卡住」。

主题与载荷（与院内输液泵网关的约定）::

    订阅：hospital/infusion/+/telemetry     （``+`` = 输液架号段）
    载荷：{"bag_seq": "B01", "pump_rate_ml_per_h": 40.0,
           "device_sn": "PUMP-<EXAMPLE>", "ts": "2026-10-22T09:41:00+08:00"}

五条纪律：

1. **架号从主题来，不从载荷猜**。同一架有 6 袋，只靠 ``bag_seq`` 无法定位到
   会话；主题里的架号段才是权威的输液架标识。webhook 兜底通道把**路径段**当架号，
   载荷里若也带 ``rack_id`` 且与路径不符即拒收——同一条纪律，两种传输。
2. **坏消息只丢一条，不拖垮订阅线程**。泵网关发来的字段缺失或类型不对的载荷很
   常见（有的固件版本少发 ``ts``）。一条坏消息只该让那一条被拒并计数，绝不该
   抛进 paho 的回调里把整个连接搞掉——那会连带整层遥测静默消失。
3. **去重按 (device_sn, ts)**。断线重连后 broker 重投的是**上一次会话**里的消息，
   而重连之后可能已经换了输液架——那时缓存是空的，靠 ``(rack, bag)`` 比是比不出来的。
   见 :class:`TelemetryDedup`。
4. **断线自己会退避重连**。指数退避（默认 1s/2s/4s…封顶 32s），连稳 ``reset_after``
   才把退避清零。重连是本服务自己的职责：把「等 broker 自己重试」当恢复策略，等于
   赌院区的 broker 配了自动重连，而默认很多没配。
5. **新鲜度要报出来**。遥测缺失只降低交叉校验能力（ETA 退回医嘱速率），但那条
   「降级了」不该只写在内部日志里——每个袋位都在响应里带 ``telemetry_state``，
   护士端能直接显示「泵速遥测 12 分钟前那次的」。
"""

from __future__ import annotations

import collections
import hashlib
import json
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from ..calibration.estimator import TelemetryCache, TelemetrySample

__all__ = [
    "DEFAULT_TOPIC_PREFIX",
    "PENDING_AUDIT_MAX",
    "TELEMETRY_TOPIC_TEMPLATE",
    "ReconnectPolicy",
    "ReconnectSupervisor",
    "ReconnectVerdict",
    "TelemetryChannel",
    "TelemetryDedup",
    "TelemetryReading",
    "TelemetryRejected",
    "TelemetrySampleView",
    "TelemetrySubmission",
    "TELEMETRY_STATES",
    "classify_sample",
    "dedup_key",
    "parse_rack_from_topic",
    "parse_telemetry_payload",
]

#: 遥测主题模板（``{prefix}`` 来自 ``MQTT_TOPIC_PREFIX``，默认 ``hospital/infusion``）。
TELEMETRY_TOPIC_TEMPLATE = "{prefix}/+/telemetry"

#: 默认主题前缀。与 ``Settings.mqtt_topic_prefix`` 的默认值一致。
DEFAULT_TOPIC_PREFIX = "hospital/infusion"

#: 跨线程审计的待写缓冲上限（条）。**为什么不能直接写**：主 SQLite 连接建在事件循环
#: 线程上（见 ``api/service.create_app`` 的说明），而 paho 的 ``on_connect`` /
#: ``on_message`` / ``on_disconnect`` 与本模块的重连监督线程都跑在**别的**线程上，
#: 拿主连接写会抛 ``ProgrammingError: SQLite objects created in a thread can only be
#: used in that same thread``。而这个异常是从 paho 回调里抛出去的——paho 的网络
#: 循环线程就此死掉，遥测从此静默消失，连一行审计都留不下来，比不写还难查。
#: 因此异线程的审计先入队，由事件循环线程在 :meth:`TelemetryChannel.flush_pending_audit`
#: 里落库；缓冲满了丢最老的一条并计数，计数摆在运行摘要里，不静默。
PENDING_AUDIT_MAX = 512

#: 载荷里必填的字段。
_REQUIRED_FIELDS = ("bag_seq", "pump_rate_ml_per_h")


class TelemetryRejected(ValueError):
    """一条遥测消息不可用（字段缺失 / 类型不对 / 主题不匹配）。

    只影响这一条消息：订阅线程记一次 ``rejected`` 计数后继续收下一条。
    """

    def __init__(self, reason: str, detail: Mapping[str, Any] | None = None) -> None:
        self.reason = reason
        self.detail = dict(detail or {})
        super().__init__(reason)


@dataclass(frozen=True)
class TelemetryReading:
    """一条解析后的遥测。"""

    rack_id: str
    bag_seq: str
    pump_rate_ml_per_h: float
    device_sn: str = ""
    ts: str = ""
    topic: str = ""
    #: 进来的是哪条通道（``mqtt`` / ``webhook``）。只用于观测与审计，不参与解析。
    source: str = "mqtt"

    def to_sample(self) -> TelemetrySample:
        return TelemetrySample(
            rate_ml_per_h=self.pump_rate_ml_per_h, device_sn=self.device_sn, ts=self.ts
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rack_id": self.rack_id,
            "bag_seq": self.bag_seq,
            "pump_rate_ml_per_h": self.pump_rate_ml_per_h,
            "device_sn": self.device_sn,
            "ts": self.ts,
        }


# ── 去重（task 39）────────────────────────────────────────────────────────

#: ``ts`` 缺失时用于兜底去重的载荷摘要口径：只取「泵报了什么」，不含通道与主题，
#: 这样 MQTT 与 webhook 两条路送来的同一份遥测摘要相同、互为重复。
_DIGEST_FIELDS = ("rack_id", "bag_seq", "pump_rate_ml_per_h", "device_sn")


def dedup_key(reading: TelemetryReading) -> tuple[tuple[str, str], str]:
    """去重键 + 去重口径。

    ``ts`` 在：``(device_sn, ts)``。``device_sn`` 缺失时退回 ``rack:bag``——泵没报序列号
    时，同一袋同一时刻的读数仍然是唯一的一份事实。

    ``ts`` 不在：``(device_sn|rack:bag, 载荷摘要)``。有的泵固件不发 ``ts``（那是
    task 11 就在对接的院内固件差异），此时内容完全一致的两条消息对 ETA 没有任何增量
    信息，按重复处理是安全的：代价只是一条真的重复上报被记进 ``duplicates`` 而不是
    ``received``，两个计数都摆在运行摘要里，运维看得见。

    **为什么不能只靠缓存层的 ts 比较**：缓存按 ``(rack, bag)`` 比，只能挡住「同一袋
    的重投」。断线重连后 broker 重投的是上一次会话里的各袋消息，重连时很可能已经换
    了输液架——那批消息在缓存里是空的，每一条都会被当成新读数吃进去。跨通道同理：
    webhook 兜底与 MQTT 若指向同一个泵网关，同一份遥测会从两条路各进来一次。
    """
    subject = reading.device_sn or f"{reading.rack_id}:{reading.bag_seq}"
    if reading.ts:
        return (subject, reading.ts), "device_ts"
    digest = hashlib.sha256(
        json.dumps(
            {name: getattr(reading, name) for name in _DIGEST_FIELDS},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return (subject, digest), "payload_digest"


class TelemetryDedup:
    """按 :func:`dedup_key` 记住「这份遥测收过了」，有界、可观测。

    容量按条数封顶（默认 8192，够两台泵连续跑一周），满了丢最老的一条：重复窗口
    只对**刚重连完那一小段**有意义，隔了一整天再收到同一份消息，缓存层的 ts 比较
    照样能挡住。
    """

    def __init__(self, *, max_entries: int = 8192) -> None:
        self._seen: dict[tuple[str, str], str] = {}
        self._max_entries = max_entries
        self.dropped = 0
        self.by_basis: dict[str, int] = {"device_ts": 0, "payload_digest": 0}

    def seen(self, reading: TelemetryReading) -> tuple[bool, str]:
        """这条收过没有。收过则计数并返回 ``(True, 去重口径)``。"""
        key, basis = dedup_key(reading)
        if key in self._seen:
            self.dropped += 1
            self.by_basis[basis] = self.by_basis.get(basis, 0) + 1
            return True, basis
        if len(self._seen) >= self._max_entries:
            self._seen.pop(next(iter(self._seen)))
        self._seen[key] = basis
        return False, basis

    def size(self) -> int:
        return len(self._seen)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": len(self._seen),
            "max_entries": self._max_entries,
            "dropped": self.dropped,
            "by_basis": dict(self.by_basis),
        }


# ── 断线重连（task 39）─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReconnectPolicy:
    """指数退避重连参数。

    ``max_attempts=0`` 表示一直重试（默认）：遥测是交叉校验输入，放弃订阅等于永久
    少一个校验来源，而病区侧的 broker 停机会被信息科当独立故障处理。设成正数则是
    「重试 N 次就停，把断线摆到运行摘要里等人看」。
    """

    base_seconds: float = 1.0
    factor: float = 2.0
    max_seconds: float = 32.0
    max_attempts: int = 0
    #: 连稳这么久，才把退避计数清零。短于这个时间的「连上又掉」仍然继续退避，
    #: 否则 broker 抖动时会退到 32s 封顶再也没机会快速恢复。
    reset_after_seconds: float = 300.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_seconds": self.base_seconds,
            "factor": self.factor,
            "max_seconds": self.max_seconds,
            "max_attempts": self.max_attempts,
            "reset_after_seconds": self.reset_after_seconds,
        }

    def delay_for(self, attempt: int) -> float:
        """第 ``attempt`` 次失败后该等多久（``attempt`` 从 1 起）。"""
        if attempt <= 0:
            return 0.0
        delay = self.base_seconds * (self.factor ** (attempt - 1))
        return float(min(delay, self.max_seconds))


class ReconnectSupervisor:
    """退避账本：什么时候该再连一次、连上之后要不要把退避清零。

    与 paho 自带的 ``reconnect_delay_set`` 是**互补**而不是重复：那条路要求 broker
    侧会话还在、loop 线程还活着；病区场景里最常见的是「broker 重启导致会话失效」，
    那时只有自己重新发起 ``connect`` 才走得通。这一层因此显式记状态、显式退避，
    并把每次尝试写进审计。
    """

    def __init__(
        self,
        policy: ReconnectPolicy | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.policy = policy or ReconnectPolicy()
        self._clock = clock or time.monotonic
        self.attempts = 0
        self.delays: list[float] = []
        self.last_attempt_at: float | None = None
        self.last_outcome: str | None = None
        self._connected_since: float | None = None
        self._next_due_at: float | None = None

    def on_connect(self, now: float | None = None) -> None:
        at = self._clock() if now is None else now
        self._connected_since = at
        self._next_due_at = None

    def on_disconnect(self, now: float | None = None) -> bool:
        """记一次断开。返回 True 表示这条连接此前是**稳定的**（退避清零）。"""
        at = self._clock() if now is None else now
        stable = (
            self._connected_since is not None
            and at - self._connected_since >= self.policy.reset_after_seconds
        )
        if stable:
            self.attempts = 0
            self.delays.clear()
        self._connected_since = None
        self._next_due_at = at
        return stable

    def due(self, now: float | None = None) -> bool:
        at = self._clock() if now is None else now
        if self._next_due_at is None:
            return True
        if self.policy.max_attempts and self.attempts >= self.policy.max_attempts:
            return False
        return at >= self._next_due_at

    def exhausted(self) -> bool:
        return bool(self.policy.max_attempts) and self.attempts >= self.policy.max_attempts

    def record_attempt(self, issued: bool, now: float | None = None) -> float:
        """记一次尝试，返回这次用掉的退避时长（秒）。

        ``issued=False`` 表示这次尝试连发都没发出去（异常已在调用方记下）；
        ``issued=True`` 只说明**调用返回了**，连没连上要等 ``on_connect`` 回调——
        这两件事混为一谈会让运行摘要里的「重连成功」变成一句谎话。
        """
        at = self._clock() if now is None else now
        self.attempts += 1
        delay = self.policy.delay_for(self.attempts)
        self.delays.append(delay)
        self.last_attempt_at = at
        self.last_outcome = "issued" if issued else "error"
        self._next_due_at = at + delay
        return delay

    def mark_connected(self) -> None:
        """回调确认连上了。

        **这里刻意不清零 ``attempts``**：连上不等于恢复。broker 抖动时最常见的就是
        「刚连上就掉」，而这一阶段若每次都清零，就会陷入「连上 → 掉 → 1 秒后又连上
        → 掉」的固定频率重试，正好压在 broker 最虚的时候。真正的清零发生在
        :meth:`on_disconnect`——只有确认这条连接**稳定站住了**（时长 ≥
        ``reset_after_seconds``）之后才归零。
        """
        self._next_due_at = None
        self.last_outcome = "connected"
        self.on_connect()

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "backoff_delays": list(self.delays),
            "last_outcome": self.last_outcome,
            "exhausted": self.exhausted(),
            "policy": self.policy.as_dict(),
        }


@dataclass(frozen=True)
class ReconnectVerdict:
    """一次重连判定的结果（可观测：每一次尝试都有一份可回读的记录）。"""

    action: str
    reason: str
    issued: bool = False
    attempt: int = 0
    delay_seconds: float = 0.0
    next_attempt_in_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "issued": self.issued,
            "attempt": self.attempt,
            "delay_seconds": round(self.delay_seconds, 3),
            "next_attempt_in_seconds": (
                round(self.next_attempt_in_seconds, 3)
                if self.next_attempt_in_seconds is not None
                else None
            ),
        }


# ── 新鲜度（task 39）───────────────────────────────────────────────────────

#: 遥测新鲜度状态。``absent`` = 这一袋从来没有任何遥测（ETA 退回医嘱速率）；
#: ``fresh`` / ``stale`` = 有读数且在/不在新鲜窗口内；``undated`` = 泵没报时间戳，
#: 新鲜度无从判断。**这四个都不阻断出数**：遥测只降低交叉校验能力，不决定余量。
TELEMETRY_STATES = ("absent", "fresh", "stale", "undated")


@dataclass(frozen=True)
class TelemetrySampleView:
    """一袋的遥测现状（写进上传响应的 ``telemetry_state``）。"""

    state: str
    rate_ml_per_h: float | None = None
    device_sn: str = ""
    ts: str = ""
    age_seconds: float | None = None
    #: 读数从哪条通道进来的（``mqtt`` / ``webhook`` / 缓存里没有来源则空）。
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "rate_ml_per_h": self.rate_ml_per_h,
            "device_sn": self.device_sn,
            "ts": self.ts,
            "age_seconds": round(self.age_seconds, 1) if self.age_seconds is not None else None,
            "source": self.source,
        }


def _parse_ts(raw: str) -> datetime | None:
    """解析遥测时间戳。带不带时区都收；解析不了返回 ``None``（不猜）。"""
    text = (raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def classify_sample(
    sample: TelemetrySample | None,
    *,
    reference: str | datetime | None = None,
    fresh_seconds: float = 300.0,
    source: str = "",
) -> TelemetrySampleView:
    """判断一条缓存里的遥测「有多新」。

    ``reference`` 是**比较基准**，默认取本次上传的 ``captured_at`` 而不是墙上时钟：
    病区网络断过、上传是排队补传的那一段，墙上时钟会把一份完全有效的读数判成
    几小时前的陈旧数据，并让护士端挂一条没必要看的黄标。

    新鲜度**不参与门禁**：``stale`` 的读数照样用于医嘱/泵速的一致性交叉校验，只是
    状态如实报出去，让护士端自己决定要不要采信。
    """
    if sample is None:
        return TelemetrySampleView(state="absent", source=source)
    # 来源默认取**读数自己带的那个**（缓存 publish 时就标了 mqtt / webhook）。
    # 不取的话，调用方每次都得把通道名再传一遍；而漏传的那一次，护士端看到的就是
    # 「有遥测但不知道从哪来的」——比报 absent 还让人误会。
    origin = source or getattr(sample, "source", "") or ""
    ts = _parse_ts(sample.ts)
    if ts is None:
        return TelemetrySampleView(
            state="undated", rate_ml_per_h=sample.rate_ml_per_h,
            device_sn=sample.device_sn, ts=sample.ts, source=origin,
        )
    if isinstance(reference, datetime):
        ref = reference
    else:
        ref = _parse_ts(reference or "") or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    age = (ref - ts).total_seconds()
    return TelemetrySampleView(
        state="fresh" if 0 <= age <= fresh_seconds else "stale",
        rate_ml_per_h=sample.rate_ml_per_h,
        device_sn=sample.device_sn,
        ts=sample.ts,
        age_seconds=age,
        source=origin,
    )


def parse_rack_from_topic(topic: str, prefix: str = DEFAULT_TOPIC_PREFIX) -> str:
    """从主题里取输液架号：``hospital/infusion/<rack>/telemetry``。

    **必须严格是「前缀 + 架号段 + telemetry」三段余一**，而不是「末段往前数第二个
    非空段」。后者在架号段缺失时会把前缀里的 ``infusion`` 当成架号——缓存里躺着
    一条 ``rack_id="infusion"`` 的泵速，看起来完全正常，ETA 交叉校验却永远配不
    上任何一袋，排查时没人会想到是主题少了一段。

    因此这里按前缀切开并要求余下恰好 ``["<rack>", "telemetry"]``：段数不对、架号
    段为空、末段不是 telemetry、前缀不符，一律拒收——宁可这条不看，也不要把一个
    错的架号写进缓存。
    """
    expected = TELEMETRY_TOPIC_TEMPLATE.format(prefix=prefix.strip("/"))
    marker = f"{prefix.strip('/')}/"
    trimmed = topic.strip()
    detail = {"topic": topic, "expected": expected}
    if not trimmed.startswith(marker):
        raise TelemetryRejected("topic_mismatch", detail)
    rest = trimmed[len(marker):].split("/")
    if len(rest) != 2 or rest[0] == "" or rest[1] != "telemetry":
        raise TelemetryRejected("topic_mismatch", detail)
    return rest[0]


def parse_telemetry_payload(topic: str, raw: Any, prefix: str = DEFAULT_TOPIC_PREFIX) -> TelemetryReading:
    """解析一条 MQTT 消息。``raw`` 为 bytes/str/已解码的 dict 都接受。

    ``pump_rate_ml_per_h <= 0`` 一律拒：负滴速与零滴速都意味着「泵停了」，把它
    当成有效速率会把 ETA 算成无穷大，然后一路报「还早得很」直到袋子空了。
    """
    rack_id = parse_rack_from_topic(topic, prefix)
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TelemetryRejected("payload_not_utf8", {"topic": topic}) from exc
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TelemetryRejected("payload_not_json", {"topic": topic}) from exc
    if not isinstance(raw, Mapping):
        raise TelemetryRejected("payload_not_object", {"topic": topic, "got": type(raw).__name__})

    missing = [name for name in _REQUIRED_FIELDS if raw.get(name) in (None, "")]
    if missing:
        raise TelemetryRejected(
            "field_missing", {"topic": topic, "missing": missing, "received": sorted(map(str, raw))}
        )
    try:
        rate = float(raw["pump_rate_ml_per_h"])
    except (TypeError, ValueError) as exc:
        raise TelemetryRejected(
            "rate_not_number", {"topic": topic, "value": str(raw.get("pump_rate_ml_per_h"))}
        ) from exc
    if rate <= 0:
        raise TelemetryRejected("rate_not_positive", {"topic": topic, "rate": rate})

    return TelemetryReading(
        rack_id=rack_id,
        bag_seq=str(raw["bag_seq"]),
        pump_rate_ml_per_h=rate,
        device_sn=str(raw.get("device_sn") or ""),
        ts=str(raw.get("ts") or ""),
        topic=topic,
    )


@dataclass(frozen=True)
class TelemetrySubmission:
    """一条遥测投进通道之后的结果。

    ``reason`` 取值：``accepted`` / ``duplicate`` / ``out_of_order``，或解析层给出的
    拒收码（``field_missing`` / ``rate_not_positive`` / …，见 :class:`TelemetryRejected`）。
    调用方（MQTT 回调、webhook 接收器）只按这份结论记账，不各自再判一遍。
    """

    reading: TelemetryReading | None = None
    accepted: bool = False
    reason: str = ""
    basis: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "basis": self.basis,
            "rack_id": self.reading.rack_id if self.reading else None,
            "bag_seq": self.reading.bag_seq if self.reading else None,
        }


class TelemetryChannel:
    """MQTT 订阅通道：连接状态、消息派发、去重与重连都收在这里。

    ``client_factory`` 供测试注入一个假 client（不需要真 broker 也能验回调与
    去重逻辑）；不传时按 :mod:`paho.mqtt` 建真连接。

    **断线不影响主路径**：连接失败/断开只把 ``state`` 记成 ``disconnected`` 并写
    一行审计，ETA 仍由医嘱速率给出（``eta_source=order_text``）。
    """

    def __init__(
        self,
        settings: Any | None = None,
        cache: TelemetryCache | None = None,
        *,
        host: str = "",
        port: int = 1883,
        username: str = "",
        password: str = "",
        topic: str = "",
        prefix: str = "",
        client_factory: Callable[[], Any] | None = None,
        clock: Callable[[], str] | None = None,
        audit: Any | None = None,
        dedup: TelemetryDedup | None = None,
        reconnect_policy: ReconnectPolicy | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if settings is not None:
            host = host or settings.mqtt_host
            port = port or settings.mqtt_port
            username = username or settings.mqtt_username
            password = password or settings.mqtt_password.get_secret_value()
            topic = topic or settings.mqtt_telemetry_topic
            prefix = prefix or getattr(settings, "mqtt_topic_prefix", "")
            if reconnect_policy is None:
                reconnect_policy = getattr(settings, "reconnect_policy", None)
        self.host = host
        self.port = int(port)
        self.username = username
        self._password = password
        self.prefix = prefix.strip("/") or DEFAULT_TOPIC_PREFIX
        self.topic = topic or TELEMETRY_TOPIC_TEMPLATE.format(prefix=self.prefix)
        self.cache = cache if cache is not None else TelemetryCache()
        self._client_factory = client_factory
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self._monotonic = monotonic or time.monotonic
        self._audit = audit
        self.dedup = dedup if dedup is not None else TelemetryDedup()
        self.reconnect = ReconnectSupervisor(reconnect_policy, clock=self._monotonic)
        self._client: Any | None = None
        self._lock = threading.Lock()
        self._state = "idle"
        self._last_message_at: str | None = None
        self._last_connected_at: str | None = None
        self._last_error: str | None = None
        self.received = 0
        self.rejected = 0
        self.duplicates = 0
        self._reconnect_thread: threading.Thread | None = None
        self._supervisor_stop: threading.Event | None = None
        # 建通道的那条线程就是**唯一能直接写审计的线程**（服务里即事件循环线程）。
        # 别的线程（paho 回调、退避重连线程）来的审计一律先入队，见 PENDING_AUDIT_MAX。
        self._owner_thread = threading.get_ident()
        self._pending_audit: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=PENDING_AUDIT_MAX
        )
        self._pending_lock = threading.Lock()
        self.pending_dropped = 0

    # ── 状态 ────────────────────────────────────────────────────────────
    def state(self) -> dict[str, Any]:
        """通道状态，供运行摘要一眼看出「遥测到底在不在流」。"""
        return {
            "state": self._state,
            "topic": self.topic,
            "host": self.host,
            "port": self.port,
            "last_message_at": self._last_message_at,
            "last_connected_at": self._last_connected_at,
            "last_error": self._last_error,
            "received": self.received,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "cached_samples": self.cache.count(),
            "dedup": self.dedup.as_dict(),
            "reconnect": self.reconnect.as_dict(),
            "pending_audit": self.pending_audit_stats(),
        }

    @property
    def connected(self) -> bool:
        return self._state == "connected"

    def pending_audit_stats(self) -> dict[str, Any]:
        """异线程审计的积压：待落库多少条、缓冲上限多少、历史上丢了多少。

        ``dropped > 0`` 意味着有人得去看 broker（异线程审计只来自 paho 回调与重连
        线程，也就是「通道出事」这一类事）；丢掉的条数摆在这里，不靠日志才发现。
        """
        with self._pending_lock:
            queued = len(self._pending_audit)
        return {
            "queued": queued,
            "max_queued": PENDING_AUDIT_MAX,
            "dropped": self.pending_dropped,
        }

    def flush_pending_audit(self) -> int:
        """把异线程攒下的审计落库。**必须在事件循环线程上调用**。

        返回落库条数。一条写失败不影响其余（审计写失败不该让摘要或探活跟着报错）。
        调用点见 ``api/summary.py`` 与 ``api/health.py``——运维本来就在轮询这两个
        端点，顺手把攒下的记录落库，比另起一个后台线程更省一条活的部件。
        """
        with self._pending_lock:
            if not self._pending_audit:
                return 0
            batch = list(self._pending_audit)
            self._pending_audit.clear()
        written = 0
        for fields in batch:
            if self._append_audit(**fields):
                written += 1
        return written

    def _append_audit(self, *, action: str, outcome: str, detail: Mapping[str, Any]) -> bool:
        """真正写一行审计。返回是否写成功（未知动作码按丢弃计，不抛）。"""
        if self._audit is None:
            return False
        try:
            self._audit.append(
                request_id=f"mqtt-{self._clock().replace(':', '').replace('-', '')}",
                actor="service:telemetry",
                action=action,
                target=self.topic,
                outcome=outcome,
                detail=detail,
            )
        except ValueError:
            return False
        return True

    def _audit_append(self, action: str, outcome: str, detail: Mapping[str, Any]) -> None:
        """记一条通道审计：同线程直接写，异线程入队等事件循环落库。

        **为什么不能图省事一律直接写**：见 :data:`PENDING_AUDIT_MAX`。这条路径上
        抛出的异常会顺着 paho 的网络循环线程抛出去，把整个订阅打死。
        """
        if self._audit is None:
            return
        if threading.get_ident() == self._owner_thread:
            self._append_audit(action=action, outcome=outcome, detail=detail)
            return
        with self._pending_lock:
            if len(self._pending_audit) == self._pending_audit.maxlen:
                self.pending_dropped += 1
            self._pending_audit.append(
                {"action": action, "outcome": outcome, "detail": dict(detail)}
            )

    # ── 生命周期 ────────────────────────────────────────────────────────
    def _build_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        import paho.mqtt.client as mqtt  # 延迟导入：没启用遥测的部署不必装

        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="infusion-monitor")
        except AttributeError:  # paho 1.x 回退
            client = mqtt.Client(client_id="infusion-monitor")
        if self.username:
            client.username_pw_set(self.username, self._password)
        client.reconnect_delay_set(min_delay=1, max_delay=32)
        return client

    def start(self) -> bool:
        """连上并开始收。**连不上不抛异常**：只记状态，视觉主路径照常。

        返回是否已连上（连上与否都算启动成功——遥测是可选输入）。
        """
        if self._client is not None:
            return self.connected
        client = self._build_client()
        client.on_connect = self.on_connect
        client.on_disconnect = self.on_disconnect
        client.on_message = self.on_message
        try:
            client.connect_async(self.host, self.port, keepalive=60)
            client.loop_start()
        except Exception as exc:  # noqa: BLE001 —— 任何连不上都不该拖垮服务启动
            self._state = "disconnected"
            self._last_error = f"{type(exc).__name__}: {exc}"
            self.reconnect.on_disconnect(self._monotonic())
            self._audit_append(
                "telemetry.channel_down", "accepted",
                {"stage": "start", "error": self._last_error, "topic": self.topic},
            )
            return False
        self._client = client
        self._state = "connecting"
        return self.connected

    def start_supervisor(self, *, interval_seconds: float = 5.0) -> threading.Thread:
        """起后台线程按退避表重连（服务在 ``MQTT_ENABLED=true`` 时调用）。

        **为什么不能靠 paho 自己重连**：broker 重启会让已有会话失效，那时 paho 的
        loop 只会反复对着一份死会话重试，永远走不到新的 CONNACK。这一层显式地重新
        ``connect``，并把每次尝试记进审计——运维要能回答「昨天夜里它到底试了几次、
        退到多久」。
        """
        if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
            return self._reconnect_thread
        stop_event = threading.Event()
        self._supervisor_stop = stop_event

        def _loop() -> None:
            while not stop_event.wait(interval_seconds):
                try:
                    self.supervise_once()
                except Exception as exc:  # noqa: BLE001 —— 监控线程不允许把异常抛出去
                    self._last_error = f"supervisor: {type(exc).__name__}"
                    self._audit_append(
                        "telemetry.channel_down", "accepted",
                        {"stage": "supervise", "error": self._last_error},
                    )

        thread = threading.Thread(target=_loop, name="telemetry-reconnect", daemon=True)
        self._reconnect_thread = thread
        thread.start()
        return thread

    def stop_supervisor(self, timeout: float = 2.0) -> None:
        stop_event = getattr(self, "_supervisor_stop", None)
        if stop_event is None:
            return
        stop_event.set()
        thread = self._reconnect_thread
        self._reconnect_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def supervise_once(self) -> ReconnectVerdict:
        """走一拍重连判定：该不该连、连了没有、下一次什么时候。

        「调用返回了」不等于「连上了」——所以 ``issued`` 与 ``state`` 分开报：
        ``issued=True`` 只说明 ``reconnect()``／``connect_async()`` 没抛异常，
        真正连上与否由 ``on_connect`` 回调改状态。运行摘要里因此永远能看到
        「尝试了 N 次、最后一次什么时候」，而不是一个乐观的「已恢复」。
        """
        if self._state == "connected":
            return ReconnectVerdict(action="none", reason="connected")
        if self._state == "stopped":
            return ReconnectVerdict(action="none", reason="stopped")
        now = self._monotonic()
        if not self.reconnect.due(now):
            return ReconnectVerdict(
                action="wait",
                reason="backoff" if not self.reconnect.exhausted() else "attempts_exhausted",
                attempt=self.reconnect.attempts,
            )
        client = self._client
        issued = False
        try:
            if client is None:
                client = self._build_client()
                client.on_connect = self.on_connect
                client.on_disconnect = self.on_disconnect
                client.on_message = self.on_message
                client.connect_async(self.host, self.port, keepalive=60)
                client.loop_start()
                self._client = client
                self._state = "connecting"
                issued = True
            else:
                client.reconnect()
                issued = True
        except Exception as exc:  # noqa: BLE001 —— 连不上是常态，不是异常事件
            self._state = "disconnected"
            self._last_error = f"reconnect: {type(exc).__name__}: {exc}"[:200]
            issued = False
        delay = self.reconnect.record_attempt(issued, now)
        self._audit_append(
            "telemetry.reconnect_attempted", "accepted" if issued else "rejected",
            {
                "attempt": self.reconnect.attempts,
                "issued": issued,
                "delay_seconds": delay,
                "error": self._last_error,
                "topic": self.topic,
            },
        )
        return ReconnectVerdict(
            action="reconnect",
            reason="issued" if issued else "error",
            issued=issued,
            attempt=self.reconnect.attempts,
            delay_seconds=delay,
            next_attempt_in_seconds=delay,
        )

    def stop(self) -> None:
        self.stop_supervisor()
        client = self._client
        self._client = None
        if client is None:
            self._state = "stopped"
            return
        for method in ("loop_stop", "disconnect"):
            try:
                getattr(client, method)()
            except Exception:  # noqa: BLE001 —— 停机路径不允许再抛
                pass
        self._state = "stopped"

    # ── paho 回调（签名兼容 paho 2.x CallbackAPIVersion.VERSION2）─────────
    def on_connect(self, client: Any, userdata: Any, flags: Any, reason_code: Any = 0,
                   properties: Any = None) -> None:
        rc = int(getattr(reason_code, "value", reason_code) or 0)
        with self._lock:
            self._state = "connected" if rc == 0 else "disconnected"
            self._last_error = None if rc == 0 else f"connect_return_code={rc}"
            if rc == 0:
                self._last_connected_at = self._clock()
        if rc == 0:
            self.reconnect.mark_connected()
        else:
            self.reconnect.on_disconnect(self._monotonic())
        try:
            client.subscribe(self.topic, qos=1)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"subscribe_failed: {type(exc).__name__}"
        self._audit_append(
            "telemetry.channel_up" if rc == 0 else "telemetry.channel_down",
            "accepted" if rc == 0 else "rejected",
            {"return_code": rc, "topic": self.topic},
        )

    def on_disconnect(self, client: Any, userdata: Any, *args: Any) -> None:
        stable = self.reconnect.on_disconnect(self._monotonic())
        with self._lock:
            self._state = "disconnected"
            self._last_error = f"disconnect: {'/'.join(str(a) for a in args if a is not None)}"[:200]
        self._audit_append(
            "telemetry.channel_down", "accepted",
            {"topic": self.topic, "reason": self._last_error, "stable_before_drop": stable},
        )

    def on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        """一条遥测进缓存。坏消息只拒这一条，不影响后续消息。"""
        topic = getattr(msg, "topic", "") or ""
        raw = getattr(msg, "payload", b"")
        self.submit(topic, raw, source="mqtt")

    # ── 主动推入（webhook 兜底通道共用同一套解析、去重与缓存）───────────
    def ingest(self, topic: str, raw: Any, *, source: str = "webhook") -> bool:
        """不走 MQTT 也能喂一条遥测（webhook 通道复用同一套解析与去重）。

        返回是否被缓存接受（重复的返回 False）。
        """
        return self.submit(topic, raw, source=source).accepted

    def submit(self, topic: str, raw: Any, *, source: str = "mqtt") -> "TelemetrySubmission":
        """解析 → 去重 → 缓存，返回一份**可回读**的结论。

        三步的顺序是刻意的：**先去重再进缓存**。反过来会出现「重复消息把缓存的
        ts 比较刷新了一遍」这种只在断线重连后才显形的怪现象，而且缓存里那条最新
        读数的时间戳来源已经说不清了。

        MQTT 回调与 webhook 兜底都走这一个入口，因此「同一条遥测从两条路进来」在
        解析、去重、缓存三层都是同一套逻辑，不会出现两个通道给出不同结论的情况。
        """
        try:
            reading = parse_telemetry_payload(topic, raw, self.prefix)
        except TelemetryRejected as rejection:
            self.rejected += 1
            self._last_message_at = self._clock()
            self._audit_append(
                "telemetry.message_rejected", "rejected",
                {"topic": topic, "reason": rejection.reason, "source": source,
                 **rejection.detail},
            )
            return TelemetrySubmission(reason=rejection.reason)
        reading = replace(reading, source=source)
        self._last_message_at = self._clock()
        duplicate, basis = self.dedup.seen(reading)
        if duplicate:
            self.duplicates += 1
            self._audit_append(
                "telemetry.duplicate_dropped", "rejected",
                {
                    "topic": topic,
                    "source": source,
                    "basis": basis,
                    "device_sn": reading.device_sn,
                    "bag_seq": reading.bag_seq,
                    "ts": reading.ts,
                },
            )
            return TelemetrySubmission(reading=reading, reason="duplicate", basis=basis)
        sample = replace(reading.to_sample(), source=source)
        accepted = self.cache.publish(reading.rack_id, reading.bag_seq, sample)
        if accepted:
            self.received += 1
            return TelemetrySubmission(
                reading=reading, accepted=True, reason="accepted", basis=basis
            )
        # 同一 (架, 袋) 上更旧的时间戳：不是重投，是乱序到达。不算重复，
        # 但也不该覆盖缓存里更新的那条。
        self._last_error = f"out_of_order_telemetry: {reading.bag_seq} ts={reading.ts}"
        return TelemetrySubmission(reading=reading, reason="out_of_order", basis=basis)