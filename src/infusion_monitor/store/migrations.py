"""幂等迁移：建表、建索引、加列，重复启动不报错。

不引入 Alembic 等迁移框架——病区侧单机部署只需要「重复启动安全」这一条；
``schema_migrations`` 表记录已应用版本，``ensure_column`` 按列存在性补齐。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from .schema import (
    AUDIT_LOG_COLUMNS,
    BAG_SESSION_COLUMNS,
    BAG_SESSION_HISTORY_COLUMNS,
    BAG_TYPE_CALIBRATION_COLUMNS,
    CALIBRATION_DEVIATION_COLUMNS,
    CALIBRATION_RECALC_COLUMNS,
    CALIBRATION_VERSION_ALERT_COLUMNS,
    CREATE_INDEX_SQL,
    CREATE_TABLE_SQL,
    INFUSION_EVENT_COLUMNS,
    INFERENCE_AUDIT_COLUMNS,
    RETENTION_ARCHIVE_COLUMNS,
    SCHEMA_VERSION,
    UPLOAD_CHUNK_COLUMNS,
    WRITEBACK_QUEUE_COLUMNS,
)

__all__ = [
    "apply_migrations",
    "current_version",
    "ensure_column",
    "table_columns",
    "TABLE_COLUMNS",
]

#: 表 → 列定义（DDL 类型片段）。加列时从这里取类型。
TABLE_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "bag_session": BAG_SESSION_COLUMNS,
    "infusion_event": INFUSION_EVENT_COLUMNS,
    "writeback_queue": WRITEBACK_QUEUE_COLUMNS,
    "inference_audit": INFERENCE_AUDIT_COLUMNS,
    "bag_type_calibration": BAG_TYPE_CALIBRATION_COLUMNS,
    "calibration_recalc_log": CALIBRATION_RECALC_COLUMNS,
    "calibration_deviation": CALIBRATION_DEVIATION_COLUMNS,
    "calibration_version_alert": CALIBRATION_VERSION_ALERT_COLUMNS,
    "upload_chunk": UPLOAD_CHUNK_COLUMNS,
    "bag_session_history": BAG_SESSION_HISTORY_COLUMNS,
    "audit_log": AUDIT_LOG_COLUMNS,
    "retention_archive": RETENTION_ARCHIVE_COLUMNS,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def table_columns(conn: Any, table: str) -> set[str]:
    """返回表当前实际存在的列名集合。

    **不写 ``PRAGMA table_info``**：那是 SQLite 方言。改成「零行查询读游标的列名」，
    两种后端都支持（``SELECT * FROM t LIMIT 0``），迁移层因此不挑方言。
    """
    try:
        cursor = conn.execute(f"SELECT * FROM {table} LIMIT 0")
    except Exception as exc:  # 表不存在（两种后端抛的异常类型不同）
        if _is_missing_table(exc):
            return set()
        raise
    return {description[0] for description in cursor.description}


def _is_missing_table(exc: Exception) -> bool:
    """判断异常是不是「表不存在」。

    SQLite 报 ``no such table``，PostgreSQL 报 ``relation ... does not exist``；
    这里按文本判断，免得为了一个探测动作把 psycopg 变成硬依赖。
    """
    text = str(exc).lower()
    return "no such table" in text or "does not exist" in text or "undefinedtable" in text


def current_version(conn: Any) -> int:
    """读取已应用的 schema 版本；未迁移过返回 0。

    直接查表而不是查 ``sqlite_master``——后者是 SQLite 方言；表还不存在时按 0 算。
    """
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    except Exception as exc:
        if _is_missing_table(exc):
            return 0
        raise
    if row is None:
        return 0
    return int(row[0] or 0)


def ensure_column(conn: Any, table: str, column: str, ddl_type: str) -> bool:
    """幂等加列：列已存在则跳过并返回 False；新增返回 True。

    SQLite 不支持 ``ADD COLUMN IF NOT EXISTS``，故先查列是否存在（方言无关的做法见
    :func:`table_columns`）。``ALTER TABLE ... ADD COLUMN`` 两边都支持。
    """
    existing = table_columns(conn, table)
    if not existing:
        raise RuntimeError(f"表不存在，无法加列：{table}")
    if column in existing:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
    return True


def _ensure_declared_columns(conn: Any) -> list[str]:
    """把 ``schema.py`` 里声明但库里没有的列补齐，返回新增列名。"""
    added: list[str] = []
    for table, columns in TABLE_COLUMNS.items():
        existing = table_columns(conn, table)
        for name, ddl_type in columns:
            if name not in existing:
                ensure_column(conn, table, name, ddl_type)
                added.append(f"{table}.{name}")
    return added


def apply_migrations(conn: Any, note: str = "init") -> dict:
    """建表 + 补列 + 建索引 + 记录版本。重复调用结果不变且不抛异常。

    **顺序要紧**：先补列再建索引——新加的列（如 ``frames_retention_state``）可能被
    新索引引用，列还不存在时建索引会直接报 ``no such column``。
    """
    for ddl in CREATE_TABLE_SQL:
        conn.execute(ddl)
    added_columns = _ensure_declared_columns(conn)
    for ddl in CREATE_INDEX_SQL:
        try:
            conn.execute(ddl)
        except Exception as exc:
            # 旧库上索引定义已变更时保留旧索引，不阻断启动
            if "exist" in str(exc).lower():
                continue
            raise
    version = current_version(conn)
    if version < SCHEMA_VERSION:
        # ON CONFLICT 是标准 SQL（SQLite 3.24+ / PostgreSQL 9.5+ 都支持），
        # 不写 SQLite 独有的 INSERT OR REPLACE——迁移层因此不挑方言。
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at, note) VALUES (?, ?, ?) "
            "ON CONFLICT(version) DO UPDATE SET applied_at = excluded.applied_at, "
            "note = excluded.note",
            (SCHEMA_VERSION, _now(), note),
        )
    return {
        "version": max(version, SCHEMA_VERSION),
        "added_columns": added_columns,
        "tables": sorted(TABLE_COLUMNS),
    }