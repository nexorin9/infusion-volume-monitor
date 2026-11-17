"""``GET /api/v1/summary``：运行摘要——护士长与信息科一眼看到「现在是什么状况」。

值班的人最常问的几句话各对应一个字段，且都必须是**可核对的数**而不是形容词：

| 值班的人问 | 摘要里的字段 |
|-----------|-------------|
| 现在有几袋该换液了？ | ``sessions.needs_change_count`` |
| 有没有换液事件没送出去？ | ``writeback.pending_writeback_count`` |
| 模型是不是一直在连不上？ | ``model.threshold_fallback_count`` |
| 遥测还在不在流？ | ``channels.telemetry.state`` / ``last_message_at`` |
| 今天模型还剩多少额度？ | ``model.quota.used_today`` / ``remaining_today`` / ``exhausted`` |

**为什么单独一个端点而不挂在 health 上**：health 是给探活与负载均衡看的，它必须
永远便宜、永远 200；摘要要查四张表的聚合，还要带上配置与通道状态。混在一起的结果
是探活每次都拖着一串聚合查询跑，摘要在对端眼里又只是「一个偶尔会慢的 health」。

**只报事实，不报警**。阈值降级次数高、出站积压多，是让**人**判断的事——摘要负责
把数摆出来，不在这里下结论。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from .. import __version__
from ..errors import error_code_docs, operational_faults
from ..logging_utils import RuntimeCounters
from ..store.retention import retention_overview

__all__ = ["router", "build_summary"]

router = APIRouter(tags=["health"])

#: 待换液状态。会话停在 needs_change 说明系统已点名、护士还没确认——它是「欠着的人工
#: 动作」，不是错误，但它积压起来就是巡视质量问题的早期信号。
NEEDS_CHANGE_STATE = "needs_change"


def _counters(request: Request) -> RuntimeCounters | None:
    return getattr(request.app.state, "counters", None)


def _latest_timestamp(*states: dict[str, Any] | None) -> str | None:
    """两条通道里较新的 ``last_message_at``（ISO8601 字符串可直接按字典序比）。"""
    seen = [
        str(state["last_message_at"])
        for state in states
        if state and state.get("last_message_at")
    ]
    return max(seen) if seen else None


def _merged_telemetry_state(
    mqtt: dict[str, Any] | None,
    webhook: dict[str, Any] | None,
    mqtt_enabled: bool,
    webhook_enabled: bool,
) -> str:
    """两条入站通道的合并状态。

    ``disabled`` 只在**两条都没开**时出现——那是「本部署没接泵网关」，与「接了但
    断了」在值班的人那里是两件完全不同的事（前者什么都不用做，后者要查 broker 或
    泵网关）。任一条在流就是 ``connected``。
    """
    if not mqtt_enabled and not webhook_enabled:
        return "disabled"
    if (mqtt or {}).get("state") == "connected" or (webhook or {}).get("state") == "active":
        return "connected"
    return "disconnected"


def build_summary(request: Request) -> dict[str, Any]:
    """组装运行摘要。除「把异线程攒下的通道审计落库」外是纯读，不外发任何请求。"""
    state = request.app.state
    store = state.store
    sessions = store.sessions
    events = store.events
    audit = store.audit
    calibrations = store.calibrations

    # paho 回调与退避重连线程不在事件循环线程上，不能直接写主连接（见
    # ``mqtt_telemetry.PENDING_AUDIT_MAX``）。运维本来就在轮询这个端点，
    # 顺手把攒下的通道记录落库，比另起一个后台线程更省一条活的部件。
    telemetry_channel = getattr(state, "telemetry_channel", None)
    flushed_audit = (
        telemetry_channel.flush_pending_audit() if telemetry_channel is not None else 0
    )

    by_state = sessions.count_by_state()
    by_status = events.count_by_writeback_status()
    by_source = audit.count_by_decision_source()
    pending_manual_review = sum(
        1 for row in audit.list_recent(limit=500) if row.get("pending_manual_review")
    )

    telemetry_state = telemetry_channel.state() if telemetry_channel is not None else None
    telemetry_cache = getattr(state, "telemetry", None)
    cached_samples = telemetry_cache.count() if telemetry_cache is not None else 0
    telemetry_webhook = getattr(state, "telemetry_webhook", None)
    webhook_state = telemetry_webhook.state() if telemetry_webhook is not None else None
    mqtt_enabled = bool(state.settings.mqtt_enabled)
    webhook_enabled = bool(state.settings.webhook_enabled)

    # 待写回队列（task 37）：欠账的**队列视角**。事件表给的是「最后一次投递的
    # 结果」，队列给的是「还欠着几笔、其中几笔已经自动重试耗尽等人处理」——
    # 护士长问「今天有几袋换液没写回去」时看的是后者。
    queue = getattr(store, "writeback_queue", None)
    queue_by_status = queue.count_by_status() if queue is not None else {}

    counters = _counters(request)

    # 网关的进程内用量与成本护栏（gateway/metrics.py）。**与上面那些落库的数不是
    # 同一口径**：sessions/writeback 查的是全量历史，这里记的是「这个进程启动
    # 以来烧了多少」——所以两组数并列而不相加，重启会清零这件事也不需要解释。
    # 值班的人问的是「模型是不是一直在连不上」「额度还剩多少」，答的是这里。
    gateway = getattr(state, "model_gateway", None)
    gateway_metrics = getattr(gateway, "metrics", None)
    gateway_quota = getattr(gateway, "quota", None)

    payload: dict[str, Any] = {
        "service": "infusion-volume-monitor",
        "version": __version__,
        "runtime": counters.as_dict() if counters is not None else None,
        # 这次请求顺手落库的异线程审计条数：上一次轮询到现在，paho 回调与重连线程
        # 攒下的记录。非 0 说明期间通道那边确实有事发生过。
        "flushed_channel_audit": flushed_audit,
        "sessions": {
            "total": sessions.count(),
            "by_state": by_state,
            "needs_change_count": int(by_state.get(NEEDS_CHANGE_STATE, 0)),
            "confirmed_count": int(by_state.get("replaced", 0)),
            "calibration_suppressed_count": sum(
                1 for row in sessions.list_by_state("monitoring", limit=500)
                if row.get("pickup_suppressed") and not row.get("trigger_reason")
            ),
        },
        "writeback": {
            "total": events.count(),
            "by_status": by_status,
            # 只数「等着重放」的：pending 是护士还没确认的，还没到出站阶段，
            # 把它算进积压会让护士长以为自己漏点了确认。
            "pending_writeback_count": int(by_status.get("pending_writeback", 0)),
            "unsent_pending_count": int(by_status.get("pending", 0)),
            "sent_count": int(by_status.get("sent", 0)),
            "failed_count": int(by_status.get("failed", 0)),
            # 队列视角（task 37）：欠账条数与其中「自动重试已耗尽、等人处理」的条数。
            "queue_by_status": queue_by_status,
            "queue_pending_count": sum(
                count for status, count in queue_by_status.items() if status != "sent"
            ),
            "queue_dead_letter_count": int(queue_by_status.get("dead_letter", 0)),
        },
        "model": {
            "total_calls": audit.count(),
            "by_decision_source": by_source,
            "threshold_fallback_count": int(by_source.get("threshold_fallback", 0)),
            "model_decided_count": int(by_source.get("model_decided", 0)),
            "threshold_only_count": int(by_source.get("threshold_only", 0)),
            "pending_manual_review_count": pending_manual_review,
            # 本进程自启动以来的用量/延迟/失败率（token、p50、p95、失败率）。
            # 没挂网关统计（如单测里手工造的网关）时这两项整体不出现。
            "runtime_metrics": gateway_metrics.snapshot() if gateway_metrics else None,
            # 成本护栏：今日已用/上限/是否已耗尽。超限后走确定性阈值路径，结论照常。
            "quota": gateway_quota.as_dict() if gateway_quota else None,
        },
        "calibrations": {
            "active_count": len(calibrations.list_active()),
            "total_versions": calibrations.count(),
            "recalc_count": calibrations.count_recalc(),
        },
        # 保留期（task 44）：定时清理在 cron 里跑，值班的人从这里看「盘上还剩
        # 多少帧、下一次清理会清哪几袋、为什么有的袋清不掉」。**只读**——
        # 摘要不删任何东西，清理只由 scripts/purge_expired_sessions.py 做。
        "retention": retention_overview(
            store.conn, policy=state.settings.retention_policy
        ),
        "channels": {
            # 合并视图：入站遥测整体在不在流。值班的人只问一句「遥测还在不在流」，
            # 不需要先知道院区用的是 MQTT 还是 webhook。
            "telemetry": {
                "enabled": mqtt_enabled or webhook_enabled,
                "mqtt_enabled": mqtt_enabled,
                "webhook_enabled": webhook_enabled,
                "state": _merged_telemetry_state(
                    telemetry_state, webhook_state, mqtt_enabled, webhook_enabled
                ),
                "last_message_at": _latest_timestamp(telemetry_state, webhook_state),
                "cached_samples": cached_samples,
                "received": (telemetry_state or {}).get("received", 0),
                "rejected": (telemetry_state or {}).get("rejected", 0),
                "duplicates": (telemetry_state or {}).get("duplicates", 0),
                "fresh_seconds": state.settings.telemetry_fresh_seconds,
            },
            # 两条入站通道各自的明细。**状态只有一份**（同一个缓存与去重表），
            # 这里分列只是为了让运维一眼看出「断的是 MQTT，还是 webhook 一直没来过」。
            "mqtt": {
                "enabled": mqtt_enabled,
                **(telemetry_state or {"state": "unavailable"}),
            },
            "webhook": {
                "enabled": webhook_enabled,
                **(webhook_state or {"state": "unavailable"}),
            },
        },
        "error_codes": [
            {"code": doc.code, "status": doc.status} for doc in error_code_docs()
        ],
        "operational_faults": [doc.key for doc in operational_faults()],
    }
    # 配置摘要只给掩码：describe() 已经把三个密钥替换成 xx**xx。
    payload["config"] = state.settings.describe()
    return payload


@router.get("/api/v1/summary", summary="运行摘要")
async def summary(request: Request) -> dict:
    return build_summary(request)