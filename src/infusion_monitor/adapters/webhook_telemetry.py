"""入站 webhook 兜底通道：没有 MQTT 的院区也能把泵速遥测送进来。

**为什么要有这条通道**：有的院区根本没有 MQTT broker，输液泵网关只提供一个
HTTP 回调地址；也有的病区 broker 由集团统一管，病区侧没有订阅权限。这两种情况下
「订阅主题」这条路是死的，但遥测本身（医嘱写 40 ml/h、泵上跑的是 20 ml/h）仍然
值得交叉校验一次。所以入站是**双通道**：

    MQTT    POST/订阅  hospital/infusion/<rack>/telemetry
    webhook POST      /api/v1/infusion/telemetry/webhook/<rack>

两条路**共用同一份解析、同一套去重、同一个缓存**（:mod:`adapters.mqtt_telemetry`
的 ``TelemetryChannel``）。这不是省事，是必须：同一条遥测从两条路各进来一次是常态
（切通道的灰度期最明显），两个各写一套的解析器必然在某次固件差异上给出不同结论，
而「两条遥测不一致」这种问题排查起来比「没收到遥测」贵得多。

**四条纪律**：

1. **架号取自路径段，不取自载荷**。与 MQTT 取架号段是同一条纪律；载荷里若也带
   ``rack_id`` 且与路径不符，**拒收**——一份跨架的遥测会写进另一架的缓存，而缓存里
   躺着的东西事后根本看不出是哪来的。
2. **先验签再解析**。签名不过的请求连 JSON 都不解：未验签的载荷一旦开始解析，
   就已经在替攻击者消耗 CPU 了。签名比对用 :func:`hmac.compare_digest`（常数时间），
   不用 ``==``。
3. **重放窗口靠时间戳**。带 ``X-Infusion-Timestamp``（epoch 秒）时，签名串把它拼在
   最前面（``{ts}.{body}``），并且要求请求时间与服务端相差不超过
   ``WEBHOOK_SIGNATURE_TOLERANCE_SECONDS``（默认 300 秒）。**为什么时间戳要在签名
   串里**：否则攻击者可以把一条真消息的时间戳头换掉而签名照样成立，窗口形同虚设。
   不带时间戳的请求按只对 body 验签放行（院内有些网关确实不会加这个头），但此时
   没有重放保护——这一点在响应与文档里都写明，不让部署方误以为有。
4. **通道是可选的**。``WEBHOOK_SIGNING_SECRET`` 没配就**没有这个端点**（404），
   而不是配一个默认密钥。不是「装了就开着」的接口。

签名密钥一律从环境变量读（:mod:`infusion_monitor.config`），代码与配置文件里
不出现明文，错误文本与审计里也不回显密钥。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from .mqtt_telemetry import (
    DEFAULT_TOPIC_PREFIX,
    TelemetryChannel,
    TelemetryReading,
)

__all__ = [
    "DEFAULT_TOLERANCE_SECONDS",
    "MAX_BODY_BYTES",
    "SIGNATURE_HEADER",
    "TIMESTAMP_HEADER",
    "TELEMETRY_PAYLOAD_SCHEMA",
    "WEBHOOK_REJECTION_REASONS",
    "WebhookOutcome",
    "WebhookSignature",
    "WebhookTelemetryReceiver",
    "contract_markdown",
    "sign_payload",
    "verify_signature",
]

#: 请求头名（大小写不敏感，由 HTTP 层归一）。写在这里而不是散在代码里，
#: 是为了让 :func:`contract_markdown` 生成的那份对接文档与代码永远一致。
SIGNATURE_HEADER = "x-infusion-signature"
TIMESTAMP_HEADER = "x-infusion-timestamp"

#: 签名串的算法标识，写在签名头里。对接方看到 ``sha256=`` 就知道按 HMAC-SHA256 算。
_SIGNATURE_SCHEME = "sha256"

#: 重放窗口默认值（秒）。5 分钟：一轮巡视 1–2 小时，泵网关的消息重投不会超过这个尺度；
#: 太宽会让抓到的旧消息一直能重放，太窄则钟不准的病区会自己把自己的真消息拒掉。
DEFAULT_TOLERANCE_SECONDS = 300.0

#: 单条 webhook 请求体上限（字节）。一条遥测几十字节，64KB 足够；这条闸门是为了
#: 挡住「忘了验签就先读 body」的实现里那个无上限的 ``await request.body()``。
MAX_BODY_BYTES = 64 * 1024

#: 入站载荷字段规范（与 MQTT 那条路**逐字段一致**）。``rack_id`` 在 webhook 侧来自路径，
#: 载荷里带它是可选的，带了就必须与路径一致。
TELEMETRY_PAYLOAD_SCHEMA: tuple[tuple[str, str, bool, str], ...] = (
    ("bag_seq", "string", True, "袋序标识，须与输注医嘱/护理单元系统给出的一致"),
    ("pump_rate_ml_per_h", "number", True, "泵当前执行的滴速（ml/h）；≤0 一律拒收"),
    ("device_sn", "string", False, "输液泵序列号；缺时退化为按 (架, 袋, 载荷) 去重"),
    ("ts", "string", False, "ISO8601 时间戳；缺时新鲜度报 undated 而不是假装很新"),
    ("rack_id", "string", False, "输液架标识；带了就必须与 URL 路径段一致，否则整条拒收"),
)

#: 拒收原因码（写进审计与响应 detail，运维按这个去问泵网关厂商）。
WEBHOOK_REJECTION_REASONS: tuple[tuple[str, str], ...] = (
    ("webhook_disabled", "未配置 WEBHOOK_SIGNING_SECRET，本部署没有这条通道"),
    ("body_too_large", "请求体超过 64KB 上限"),
    ("signature_missing", "缺少签名头"),
    ("signature_malformed", "签名头格式不对（应为 sha256=<hex>）"),
    ("signature_mismatch", "签名不匹配：密钥不一致或请求体在签名后被改动"),
    ("timestamp_malformed", "时间戳头不是 epoch 秒"),
    ("timestamp_out_of_window", "请求时间与服务端相差超过重放窗口（对端时钟可能不准）"),
    ("payload_not_json", "请求体不是合法 JSON"),
    ("payload_not_object", "请求体不是 JSON 对象"),
    ("rack_mismatch", "载荷里的 rack_id 与 URL 路径段不一致"),
)


@dataclass(frozen=True)
class WebhookSignature:
    """验签结论。``ok=False`` 时 ``reason`` 一定在 :data:`WEBHOOK_REJECTION_REASONS` 里。"""

    ok: bool
    reason: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, **dict(self.detail)}


def _signing_payload(body: bytes, timestamp: str) -> bytes:
    """签名串：``{ts}.{body}``（带时间戳时）或 ``{body}``。"""
    return f"{timestamp}.".encode("utf-8") + body if timestamp else body


def sign_payload(secret: str, body: bytes, timestamp: str = "") -> str:
    """算签名头的内容（``sha256=<hex>``）。对端（泵网关）按同一算法算即可。"""
    digest = hmac.new(
        secret.encode("utf-8"), _signing_payload(body, timestamp), hashlib.sha256
    ).hexdigest()
    return f"{_SIGNATURE_SCHEME}={digest}"


def verify_signature(
    secret: str,
    body: bytes,
    signature: str | None,
    *,
    timestamp: str = "",
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
    now: float | None = None,
) -> WebhookSignature:
    """验签 + 重放窗口。**返回结论，不抛异常**——每一种失败都是一个可判定的拒收原因。

    比对顺序是「先看格式、再看时间窗、最后比字节」：这样一条明显格式不对的请求
    不会被拿去做一次 HMAC，也就不会被当成「慢」的来源拖住线程。
    """
    if not signature:
        return WebhookSignature(False, "signature_missing", {"header": SIGNATURE_HEADER})
    raw = signature.strip()
    if "=" in raw:
        scheme, _, provided = raw.partition("=")
        if scheme.strip().lower() != _SIGNATURE_SCHEME:
            return WebhookSignature(False, "signature_malformed", {"scheme": scheme[:32]})
    else:
        provided = raw
    provided = provided.strip().lower()
    if len(provided) != 64 or any(ch not in "0123456789abcdef" for ch in provided):
        return WebhookSignature(False, "signature_malformed", {"length": len(provided)})

    if timestamp:
        try:
            sent_at = float(timestamp)
        except (TypeError, ValueError):
            return WebhookSignature(False, "timestamp_malformed", {"timestamp": str(timestamp)[:32]})
        current = time.time() if now is None else now
        skew = abs(current - sent_at)
        if skew > tolerance_seconds:
            return WebhookSignature(
                False,
                "timestamp_out_of_window",
                {"skew_seconds": round(skew, 1), "tolerance_seconds": tolerance_seconds},
            )

    expected = hmac.new(
        secret.encode("utf-8"), _signing_payload(body, timestamp), hashlib.sha256
    ).hexdigest()
    # 常数时间比较：普通 ``!=`` 会在第一个不同字节处提前返回，长度可测；
    # 签名头是可以被反复试探的公开输入，这条不能省。
    if not hmac.compare_digest(expected, provided):
        return WebhookSignature(False, "signature_mismatch", {})
    return WebhookSignature(True, "", {"timestamp_enforced": bool(timestamp)})


@dataclass(frozen=True)
class WebhookOutcome:
    """一条 webhook 的处理结论（直接作为响应体）。"""

    accepted: bool
    duplicate: bool = False
    reason: str = ""
    http_status: int = 202
    reading: TelemetryReading | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "accepted": self.accepted,
            "duplicate": self.duplicate,
        }
        if not self.accepted:
            payload["reason"] = self.reason
        if self.reading is not None:
            payload.update(
                {
                    "rack_id": self.reading.rack_id,
                    "bag_seq": self.reading.bag_seq,
                    "pump_rate_ml_per_h": self.reading.pump_rate_ml_per_h,
                    "device_sn": self.reading.device_sn or None,
                    "ts": self.reading.ts or None,
                    "source": "webhook",
                }
            )
        payload.update(self.detail)
        return payload


class WebhookTelemetryReceiver:
    """webhook 兜底通道：验签 → 解析 → 走 MQTT 那条通道的同一套去重与缓存。

    ``channel`` 是共享的 :class:`~infusion_monitor.adapters.mqtt_telemetry.TelemetryChannel`，
    所以接收计数、去重表、缓存与审计都是同一份——两个通道在运行摘要里是可分的两个
    视图，但**状态只有一份**（一份遥测不会因为走了哪条路而变成两条）。
    """

    def __init__(
        self,
        channel: TelemetryChannel,
        *,
        secret: str = "",
        tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
        max_body_bytes: int = MAX_BODY_BYTES,
        clock: Callable[[], str] | None = None,
        audit: Any | None = None,
        prefix: str = "",
    ) -> None:
        self.channel = channel
        self.secret = secret
        self.tolerance_seconds = float(tolerance_seconds)
        self.max_body_bytes = int(max_body_bytes)
        self.prefix = (prefix or channel.prefix or DEFAULT_TOPIC_PREFIX).strip("/")
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self._audit = audit
        self.accepted = 0
        self.rejected = 0
        self.duplicates = 0
        self.last_message_at: str | None = None
        self.last_rack_id: str | None = None
        self.last_signature_ok: bool | None = None

    # ── 状态 ────────────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return bool(self.secret)

    def state(self) -> dict[str, Any]:
        """通道状态（运行摘要里 ``channels.webhook`` 的来源）。

        ``state`` 对 webhook 的含义与 MQTT 不同：HTTP 推送没有「连着」这回事，
        只有「最近有没有在收」。因此 ``active`` = 新鲜窗口内收到过消息，
        ``idle`` = 配了但没收到过或已经很久没有。
        """
        return {
            "enabled": self.enabled,
            "state": self._activity_state(),
            "path": "/api/v1/infusion/telemetry/webhook/{rack_id}",
            "signature_header": SIGNATURE_HEADER,
            "timestamp_header": TIMESTAMP_HEADER,
            "tolerance_seconds": self.tolerance_seconds,
            "max_body_bytes": self.max_body_bytes,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "last_message_at": self.last_message_at,
            "last_rack_id": self.last_rack_id,
            "last_signature_ok": self.last_signature_ok,
        }

    def _activity_state(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.last_message_at is None:
            return "idle"
        return "active"

    # ── 处理 ────────────────────────────────────────────────────────────
    def topic_for(self, rack_id: str) -> str:
        """把路径段还原成等价的 MQTT 主题，让两条路走同一套解析。"""
        return f"{self.prefix}/{rack_id}/telemetry"

    def handle(
        self,
        rack_id: str,
        body: bytes,
        *,
        signature: str | None = None,
        timestamp: str = "",
        now: float | None = None,
    ) -> WebhookOutcome:
        """处理一条 webhook 请求。**任何失败都返回结论**，不抛异常。

        闸门顺序：通道开没开 → 体量 → 验签/时间窗 → JSON 形状 → 架号一致 →
        解析与去重。前三道在**碰载荷之前**就返回。
        """
        if not self.enabled:
            return self._reject("webhook_disabled", rack_id, http_status=404)
        if len(body) > self.max_body_bytes:
            return self._reject(
                "body_too_large", rack_id, http_status=413,
                detail={"bytes": len(body), "limit": self.max_body_bytes},
            )
        verdict = verify_signature(
            self.secret,
            body,
            signature,
            timestamp=timestamp,
            tolerance_seconds=self.tolerance_seconds,
            now=now,
        )
        self.last_signature_ok = verdict.ok
        if not verdict.ok:
            status = 401 if verdict.reason != "body_too_large" else 413
            return self._reject(
                verdict.reason, rack_id, http_status=status, detail=dict(verdict.detail)
            )

        try:
            payload = json.loads(body.decode("utf-8"))
        except UnicodeDecodeError:
            return self._reject("payload_not_json", rack_id, http_status=422)
        except json.JSONDecodeError:
            return self._reject("payload_not_json", rack_id, http_status=422)
        if not isinstance(payload, Mapping):
            return self._reject("payload_not_object", rack_id, http_status=422)
        claimed = str(payload.get("rack_id") or "").strip()
        if claimed and claimed != rack_id:
            # 跨架遥测：宁可整条拒，也不让一份来源不明的读数进缓存。
            return self._reject(
                "rack_mismatch", rack_id, http_status=422,
                detail={"payload_rack_id": claimed, "path_rack_id": rack_id},
            )

        result = self.channel.submit(self.topic_for(rack_id), dict(payload), source="webhook")
        # 重放保护有没有真的生效，如实回给对端：带时间戳且过了窗口校验才算有。
        # 对不上就说明「抓到的旧消息可以重放」，这条通道自己心里得知道。
        replay_detail = {"timestamp_enforced": bool(verdict.detail.get("timestamp_enforced"))}
        if result.reading is None:
            # 解析层拒收（缺字段、滴速非正等）：细因由通道那边给出，审计已落。
            self.rejected += 1
            return WebhookOutcome(
                accepted=False,
                reason="payload_rejected",
                http_status=422,
                detail={**replay_detail, "channel_reason": result.reason or "see_audit"},
            )
        self.last_message_at = self._clock()
        self.last_rack_id = rack_id
        if not result.accepted:
            # 重复不是错误：断线重投、MQTT/webhook 双通道并存都会走到这里。
            # 回 202 + duplicate=true，让对端**不要**把它当失败去无限重投。
            self.duplicates += 1
            return WebhookOutcome(
                accepted=False,
                duplicate=True,
                reason=result.reason or "duplicate",
                reading=result.reading,
                detail=replay_detail,
            )
        self.accepted += 1
        return WebhookOutcome(accepted=True, reading=result.reading, detail=replay_detail)

    def _reject(
        self,
        reason: str,
        rack_id: str,
        *,
        http_status: int = 401,
        detail: Mapping[str, Any] | None = None,
    ) -> WebhookOutcome:
        self.rejected += 1
        self.last_message_at = self._clock()
        self.last_rack_id = rack_id
        if self._audit is not None:
            try:
                self._audit.append(
                    request_id=f"webhook-{self._clock().replace(':', '').replace('-', '')}",
                    actor="service:telemetry-webhook",
                    action="telemetry.webhook_rejected",
                    target=rack_id,
                    outcome="rejected",
                    detail={"reason": reason, **dict(detail or {})},
                )
            except ValueError:
                pass
        return WebhookOutcome(
            accepted=False,
            reason=reason,
            http_status=http_status,
            detail=dict(detail or {}),
        )


def contract_markdown() -> str:
    """由上面的常量生成 ``docs/webhook-contract.md``（与出站那份一个做法）。

    对接文档与代码各写一份、然后慢慢漂移，是挂接方最怕的事：拿着过期文档调接口，
    排查半天最后发现是文档错了。文档由代码生成，并有测试比对是否同步。
    """
    lines = [
        "# 入站 webhook 兜底通道 · 对接契约",
        "",
        "> 本文件由 `infusion_monitor.adapters.webhook_telemetry.contract_markdown()` 从",
        "> 代码里的常量生成，`tests/test_mqtt_resilience.py` 会比对是否同步。",
        "> **请改代码，不要改本文。**",
        "",
        "## 什么时候才需要这条通道",
        "",
        "本服务默认从 MQTT 订阅 `hospital/infusion/<rack>/telemetry`。**只有**下面两种",
        "院区才需要 webhook：病区没有 MQTT broker，或病区侧没有 broker 订阅权限。",
        "两个通道同时开着也可以——同一份遥测从两条路进来会被去重表认出来（按",
        "`device_sn + ts`）。",
        "",
        "## 端点",
        "",
        "```",
        "POST /api/v1/infusion/telemetry/webhook/{rack_id}",
        f"{SIGNATURE_HEADER}: sha256=<hex>",
        f"{TIMESTAMP_HEADER}: <epoch 秒，可选>",
        "Content-Type: application/json",
        "```",
        "",
        "未配置 `WEBHOOK_SIGNING_SECRET` 的部署**没有这个端点**（返回 404",
        "`webhook_disabled`），不会用一个默认密钥把它开着。",
        "",
        "## 签名算法",
        "",
        "```python",
        "signing_input = f\"{timestamp}.\".encode() + body if timestamp else body",
        f"signature = \"{_SIGNATURE_SCHEME}=\" + hmac_sha256(secret, signing_input).hexdigest()",
        "```",
        "",
        "* 时间戳**在签名串里**（`ts.body`），因此换掉时间戳头会让签名失效——",
        "  否则重放窗口形同虚设。",
        f"* 带了 `{TIMESTAMP_HEADER}` 时，重放窗口为 ±{int(DEFAULT_TOLERANCE_SECONDS)} 秒",
        "  （`WEBHOOK_SIGNATURE_TOLERANCE_SECONDS` 可调）。对端时钟不准会自己拒掉自己的",
        "  真消息——请先对时，而不是把这个头去掉。",
        "* 不带时间戳时只对 body 验签，**没有重放保护**：抓到的旧消息可以重放。",
        "  这条通道只影响泵速交叉校验，不影响余量与换液判定。",
        "",
        "## 载荷字段",
        "",
        "| 字段 | 类型 | 必填 | 说明 |",
        "|------|------|------|------|",
    ]
    for name, kind, required, note in TELEMETRY_PAYLOAD_SCHEMA:
        lines.append(f"| `{name}` | {kind} | {'是' if required else '否'} | {note} |")
    lines += [
        "",
        "`rack_id` 取自 **URL 路径段**；载荷里带了就必须与路径一致，否则整条拒收",
        "（`rack_mismatch`）——一份跨架的遥测写进另一架的缓存，事后查不出来源。",
        "",
        "## 响应与拒收原因",
        "",
        "| 情况 | 状态码 | 响应里的 `reason` |",
        "|------|--------|------------------|",
        "| 新读数 | 202 | `accepted=true` |",
        "| 重复（同一 `device_sn+ts`，或同架同袋的更旧时间戳） | 202 | `duplicate=true` |",
    ]
    status_by_reason = {
        "webhook_disabled": "404",
        "body_too_large": "413",
        "signature_missing": "401",
        "signature_malformed": "401",
        "signature_mismatch": "401",
        "timestamp_malformed": "401",
        "timestamp_out_of_window": "401",
        "payload_not_json": "422",
        "payload_not_object": "422",
        "rack_mismatch": "422",
        "payload_rejected": "422",
    }
    for reason, meaning in WEBHOOK_REJECTION_REASONS:
        lines.append(f"| {meaning} | {status_by_reason.get(reason, '422')} | `{reason}` |")
    lines += [
        "| 载荷字段缺失 / 滴速非正 | 422 | `payload_rejected`（细因见审计链） |",
        "",
        "## 这条通道坏了会怎样",
        "",
        "**什么都不发生——视觉主路径照常。** 余量与换液判定只用短片；滴速退回输注",
        "医嘱文本（`eta_source=order_text`），每个袋位的 `telemetry_state` 如实报",
        "`absent`，运行摘要 `channels.webhook.state` 变成 `idle`。没有遥测不等于",
        "没有结论。",
        "",
        "## 未实连范围",
        "",
        "本轮只对本地 HTTP 替身与测试客户端跑过签名/去重/拒收路径；**真实院内泵网关",
        "的 webhook 端点未实连**。替身在测试数据层，不是最终对接方式。",
        "",
    ]
    return "\n".join(lines)