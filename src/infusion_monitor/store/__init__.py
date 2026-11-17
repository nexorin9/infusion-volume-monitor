"""存储层：SQLite schema、幂等迁移、仓储方法、append-only 审计链、保留期策略。"""

from __future__ import annotations

from .audit_chain import (
    AUDIT_CATEGORIES,
    AuditChain,
    actions_in_categories,
    category_label,
    category_of,
    detail_digest,
    payload_digest,
)
from .db import connect, connect_dsn, sqlite_path_from_dsn, transaction
from .dsn import (
    BackendUnavailableError,
    DatabaseDsn,
    UnsupportedDsnError,
    parse_dsn,
    redact_dsn,
    sqlite_dsn,
)
from .migrations import apply_migrations, current_version, ensure_column, table_columns
from .repositories import (
    CalibrationRepository,
    DeviationRepository,
    EventRepository,
    InferenceAuditRepository,
    SessionRepository,
    UploadRepository,
    WritebackQueueRepository,
    new_id,
    utc_now,
)
from .retention import (
    BLOCK_REASONS,
    RetentionPolicy,
    RetentionResult,
    plan_retention,
    run_retention,
)
from .schema import (
    AUDIT_ACTIONS,
    BAG_STATES,
    CALIBRATION_STATES,
    RETENTION_STATES,
    SCHEMA_VERSION,
    UPLOAD_STATUSES,
    WRITEBACK_QUEUE_STATES,
    WRITEBACK_STATES,
)

__all__ = [
    "AUDIT_ACTIONS",
    "AUDIT_CATEGORIES",
    "AuditChain",
    "actions_in_categories",
    "category_label",
    "category_of",
    "BAG_STATES",
    "BackendUnavailableError",
    "BLOCK_REASONS",
    "CALIBRATION_STATES",
    "CalibrationRepository",
    "DatabaseDsn",
    "DeviationRepository",
    "EventRepository",
    "InferenceAuditRepository",
    "RETENTION_STATES",
    "RetentionPolicy",
    "RetentionResult",
    "SCHEMA_VERSION",
    "SessionRepository",
    "UPLOAD_STATUSES",
    "UnsupportedDsnError",
    "UploadRepository",
    "WRITEBACK_QUEUE_STATES",
    "WRITEBACK_STATES",
    "WritebackQueueRepository",
    "apply_migrations",
    "connect",
    "connect_dsn",
    "current_version",
    "detail_digest",
    "ensure_column",
    "new_id",
    "open_store",
    "parse_dsn",
    "payload_digest",
    "plan_retention",
    "redact_dsn",
    "run_retention",
    "sqlite_dsn",
    "sqlite_path_from_dsn",
    "table_columns",
    "transaction",
    "utc_now",
]


def open_store(database_url: str):
    """打开连接并幂等迁移，返回六个仓储（会话/上传/事件/推理/标定/审计链）。

    启动时统一走这里，保证「重复启动不报错」。
    """
    conn = connect(database_url)
    apply_migrations(conn)
    return (
        conn,
        SessionRepository(conn),
        UploadRepository(conn),
        EventRepository(conn),
        InferenceAuditRepository(conn),
        CalibrationRepository(conn),
        DeviationRepository(conn),
        AuditChain(conn),
    )