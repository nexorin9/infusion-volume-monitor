"""入站 webhook 兜底通道的 HTTP 面：``POST /api/v1/infusion/telemetry/webhook/{rack_id}``。

**为什么要有它**：有的病区没有 MQTT broker，或 broker 由集团统一管、病区侧没有订阅
权限。那些院区里「医嘱写 40 ml/h、泵上跑的是 20 ml/h」这类对不上照样会发生，
所以泵速遥测这条交叉校验不能整条砍掉——它得换个门进来。

**这个端点做什么**：验签 → 校验 → 把这一条遥测交给 MQTT 那条通道的同一套解析、去重
与缓存。它**不**推进任何业务状态，也不返回余量：遥测只影响 ETA 的来源与交叉校验，
而 ETA 是下一次上传短片时才算的。一句话说清楚的是：**没收到遥测，什么都不会坏。**

对接契约（端点、签名算法、载荷字段、拒收原因码）见 ``docs/webhook-contract.md``，
该文件由 :func:`infusion_monitor.adapters.webhook_telemetry.contract_markdown`
从代码生成，有测试比对是否同步。

**未实连范围**：真实院内泵网关的 webhook 端点本轮未实连；签名、去重、拒收路径
只对本地测试客户端跑过。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from ..adapters.webhook_telemetry import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    WebhookTelemetryReceiver,
)
from ..errors import ApiError, Issue
from ..logging_utils import request_id_of
from .schemas import TelemetryWebhookOut

__all__ = ["router", "webhook_path"]

logger = logging.getLogger("infusion_monitor.api.webhooks")

router = APIRouter(prefix="/api/v1/infusion/telemetry", tags=["telemetry"])

#: 文档与错误提示里引用的端点模板（与 router 的 prefix + path 保持一致）。
webhook_path = "/api/v1/infusion/telemetry/webhook/{rack_id}"


def _issue(code: str, reason: str, suggestion: str) -> tuple[str, tuple[Issue, ...]]:
    return code, (Issue(code=reason, field="signature", reason=reason, suggestion=suggestion),)


@router.post(
    "/webhook/{rack_id}",
    response_model=TelemetryWebhookOut,
    response_model_exclude_none=True,
    status_code=202,
    summary="入站泵速遥测（webhook 兜底通道）",
    description=(
        "给**没有 MQTT 的院区**用的入站通道：输液泵网关把泵速 POST 过来，"
        "本服务按与 MQTT 相同的 schema 解析、去重并缓存，用于 ETA 的交叉校验。\n\n"
        "**架号取自 URL 路径段**，不取自载荷；载荷里若也带 `rack_id` 且与路径不一致，"
        "整条拒收。\n\n"
        "签名：`HMAC-SHA256(WEBHOOK_SIGNING_SECRET, ts + \".\" + body)`，"
        "结果写作 `sha256=<hex>` 放 `X-Infusion-Signature`。带 `X-Infusion-Timestamp`"
        "时一并校验重放窗口（默认 ±300 秒）。**验签在解析之前**，不通过的请求体不会被读进缓存。\n\n"
        "重复投递回 `duplicate=true`（202），不是错误——断线重投与双通道并存都会走到这里。\n\n"
        "未配置 `WEBHOOK_SIGNING_SECRET` 的部署没有这个端点（404 `webhook_disabled`）。"
    ),
)
async def telemetry_webhook(rack_id: str, request: Request) -> dict:
    """一条遥测进缓存。返回的是**这一条**的结论，不涉及任何业务状态。"""
    receiver: WebhookTelemetryReceiver | None = getattr(
        request.app.state, "telemetry_webhook", None
    )
    if receiver is None or not receiver.enabled:
        code, issues = _issue(
            "NOT_FOUND",
            "webhook_disabled",
            "本部署未配置 WEBHOOK_SIGNING_SECRET，这条兜底通道没有开启；"
            "改用 MQTT 订阅，或请信息科在 .env 里配置签名密钥后重启。",
        )
        raise ApiError(code, "webhook 兜底通道未开启", issues=issues)

    body = await request.body()
    signature = request.headers.get(SIGNATURE_HEADER)
    timestamp = request.headers.get(TIMESTAMP_HEADER, "")
    outcome = receiver.handle(
        rack_id,
        body,
        signature=signature,
        timestamp=timestamp,
    )
    payload = outcome.as_dict()
    if outcome.duplicate:
        # 重复投递按 202 回，不当错误抛：对端拿 4xx/5xx 会当成失败去重投，
        # 而重复的成因往往就是「上次重投了没收到应答」——那会变成无限重投。
        logger.info(
            "telemetry.webhook duplicate rack=%s bag=%s reason=%s request_id=%s",
            outcome.reading.rack_id if outcome.reading else rack_id,
            outcome.reading.bag_seq if outcome.reading else "-",
            outcome.reason, request_id_of(request),
        )
        return payload
    if not outcome.accepted:
        code, issues = _webhook_error(outcome.reason)
        logger.info(
            "telemetry.webhook rejected reason=%s rack=%s request_id=%s",
            outcome.reason, rack_id, request_id_of(request),
        )
        raise ApiError(code, "入站遥测未被接受", issues=issues)
    logger.info(
        "telemetry.webhook accepted rack=%s bag=%s duplicate=%s request_id=%s",
        outcome.reading.rack_id if outcome.reading else rack_id,
        outcome.reading.bag_seq if outcome.reading else "-",
        outcome.duplicate, request_id_of(request),
    )
    return payload


def _webhook_error(reason: str) -> tuple[str, tuple[Issue, ...]]:
    """把拒收原因码映射成 (错误码, detail.issues)。

    映射集中在这一处，而不是散在各个 ``raise`` 上：响应里的状态码、错误码表与
    ``detail.issues`` 描述必须一致，而这三个东西手工同步过一次就会漏。
    """
    if reason in ("signature_missing", "signature_malformed", "signature_mismatch"):
        return _issue(
            "WEBHOOK_SIGNATURE_INVALID",
            reason,
            "签名不对：核对两端的密钥是否同一份（本服务读 WEBHOOK_SIGNING_SECRET）"
            "与签名串口径——带时间戳时签 `ts.body`，不带时签 body。",
        )
    if reason in ("timestamp_malformed", "timestamp_out_of_window"):
        return _issue(
            "WEBHOOK_TIMESTAMP_REJECTED",
            reason,
            "先给病区对时（NTP），不要去掉时间戳头：去掉之后签名只覆盖请求体，"
            "等于放弃重放保护。",
        )
    if reason == "body_too_large":
        return _issue(
            "WEBHOOK_BODY_TOO_LARGE",
            reason,
            "一条泵速遥测只有几十字节；超限说明对端发错了地址或塞了别的东西进来。"
            "本服务不落盘也不解析超限请求体。",
        )
    return _issue(
        "TELEMETRY_REJECTED",
        reason or "payload_rejected",
        "只影响这一条遥测：余量与换液判定照常，滴速退回输注医嘱文本。"
        "核对载荷字段（bag_seq / pump_rate_ml_per_h 必填）与架号；"
        "滴速 ≤ 0 一律拒收，把「泵停了」当成有效速率会把 ETA 算成无穷大。",
    )