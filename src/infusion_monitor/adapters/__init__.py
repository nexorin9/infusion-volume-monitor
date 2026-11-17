"""M5 对接层：与院内既有系统对接的两个方向的适配器。

* :mod:`adapters.nurse_exec` —— **出站 HTTP 写回**。护士确认换液后，把一条
  ``infusion.bag.changed`` 事件 POST 到护理执行平台，带
  ``Authorization: Bearer $NURSE_EXEC_API_TOKEN``。字段规范与对端接口文档的
  对应表落在 ``docs/nurse-exec-contract.md``（由本模块的 schema 生成）。
* :mod:`adapters.writeback` —— 写回的**投递策略**：指数退避重试、超次置
  ``pending_writeback`` 并告警、未确认状态拒发。绝不静默丢事件。
* :mod:`adapters.retry_queue` —— 写回欠账的**持久台账**：待写回队列、死信与
  人工重放。退避表 1s / 4s / 16s、尝试预算 3 次（首次 + ≤2 次重试）；
  队列不产生业务事件，重放幂等。
* :mod:`adapters.mqtt_telemetry` —— **入站 MQTT 订阅**输液泵泵速遥测：
  连接状态、去重（``device_sn + ts``）、断线指数退避重连。遥测只用于与医嘱速率
  交叉校验，**视觉主路径不依赖它**：遥测缺失时 ETA 仍由医嘱速率给出。
* :mod:`adapters.webhook_telemetry` —— 入站的 **webhook 兜底通道**：给没有
  MQTT broker 的院区用，与 MQTT **共用同一个** :class:`TelemetryChannel`
  （同一份解析、去重表与缓存）。对接契约由 ``docs/webhook-contract.md`` 生成。

四个模块都不含患者姓名等可识别明细，也不把凭据写进任何错误文本或日志。
"""

from __future__ import annotations

from .mqtt_telemetry import (
    DEFAULT_TOPIC_PREFIX,
    PENDING_AUDIT_MAX,
    TELEMETRY_STATES,
    TELEMETRY_TOPIC_TEMPLATE,
    ReconnectPolicy,
    ReconnectSupervisor,
    TelemetryChannel,
    TelemetryDedup,
    TelemetryReading,
    TelemetryRejected,
    TelemetrySampleView,
    classify_sample,
    dedup_key,
    parse_rack_from_topic,
    parse_telemetry_payload,
)
from .webhook_telemetry import (
    SIGNATURE_HEADER as WEBHOOK_SIGNATURE_HEADER,
)
from .webhook_telemetry import (
    TIMESTAMP_HEADER as WEBHOOK_TIMESTAMP_HEADER,
)
from .webhook_telemetry import (
    WEBHOOK_REJECTION_REASONS,
    WebhookSignature,
    WebhookTelemetryReceiver,
    sign_payload,
    verify_signature,
)
from .nurse_exec import (
    DECISION_SOURCES_OUTBOUND,
    FAILURE_CLASSES,
    OUTBOUND_EVENT_SCHEMA,
    PEER_FIELD_BINDINGS,
    RETRYABLE_FAILURE_KINDS,
    TRIGGER_REASONS_OUTBOUND,
    FailureClass,
    NurseExecClient,
    OutboundEvent,
    PeerFieldBinding,
    WritebackAttempt,
    business_rejection_reason,
    build_outbound_body,
    classify_exception,
    classify_response,
    contract_markdown,
    failure_class,
    failure_summary,
    is_retryable,
)
from .retry_queue import (
    BACKOFF_SCHEDULE_SECONDS,
    MAX_ATTEMPTS,
    PENDING_STATUSES,
    RETRY_BUDGET,
    QueueEntry,
    WritebackQueue,
    backoff_seconds,
    drain_due,
    replay_entry,
    retry_at,
)
from .writeback import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_FACTOR,
    MAX_RETRIES,
    RETRYABLE_KINDS,
    WritebackOutcome,
    WritebackRefused,
    confirm_and_deliver,
    deliver_event,
)

__all__ = [
    "BACKOFF_BASE_SECONDS",
    "BACKOFF_FACTOR",
    "BACKOFF_SCHEDULE_SECONDS",
    "DECISION_SOURCES_OUTBOUND",
    "DEFAULT_TOPIC_PREFIX",
    "FAILURE_CLASSES",
    "MAX_ATTEMPTS",
    "MAX_RETRIES",
    "OUTBOUND_EVENT_SCHEMA",
    "PEER_FIELD_BINDINGS",
    "PENDING_STATUSES",
    "RETRYABLE_FAILURE_KINDS",
    "RETRYABLE_KINDS",
    "RETRY_BUDGET",
    "TELEMETRY_TOPIC_TEMPLATE",
    "TRIGGER_REASONS_OUTBOUND",
    "PENDING_AUDIT_MAX",
    "ReconnectPolicy",
    "ReconnectSupervisor",
    "TELEMETRY_STATES",
    "TelemetryDedup",
    "TelemetrySampleView",
    "WEBHOOK_REJECTION_REASONS",
    "WEBHOOK_SIGNATURE_HEADER",
    "WEBHOOK_TIMESTAMP_HEADER",
    "WebhookSignature",
    "WebhookTelemetryReceiver",
    "classify_sample",
    "dedup_key",
    "sign_payload",
    "verify_signature",
    "FailureClass",
    "NurseExecClient",
    "OutboundEvent",
    "PeerFieldBinding",
    "QueueEntry",
    "TelemetryChannel",
    "TelemetryReading",
    "TelemetryRejected",
    "WritebackAttempt",
    "WritebackOutcome",
    "WritebackQueue",
    "WritebackRefused",
    "backoff_seconds",
    "build_outbound_body",
    "business_rejection_reason",
    "classify_exception",
    "classify_response",
    "confirm_and_deliver",
    "contract_markdown",
    "deliver_event",
    "drain_due",
    "failure_class",
    "failure_summary",
    "is_retryable",
    "parse_rack_from_topic",
    "parse_telemetry_payload",
    "replay_entry",
    "retry_at",
]