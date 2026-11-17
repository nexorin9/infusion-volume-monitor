"""数据库连接与事务辅助。

病区侧单机部署默认 SQLite；写操作串行化由 ``BEGIN IMMEDIATE`` 保证。
换 PostgreSQL 时只改连接串（见 ``store/dsn.py``），仓储层不感知方言差异。

**方言只在本模块出现**：``BEGIN IMMEDIATE`` / ``PRAGMA`` 是 SQLite 独有的写法，
它们封装在这里，``store/repositories.py`` 只发标准 SQL。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .dsn import BackendUnavailableError, DatabaseDsn, parse_dsn

__all__ = [
    "BackendUnavailableError",
    "connect",
    "connect_dsn",
    "transaction",
    "sqlite_path_from_dsn",
]


def sqlite_path_from_dsn(database_url: str) -> str:
    """从 ``DATABASE_URL`` 取 SQLite 文件路径；非 SQLite 形态交给 DSN 层处理。"""
    dsn = parse_dsn(database_url)
    if not dsn.is_sqlite:
        raise ValueError(f"非 SQLite 连接串，请在 store/dsn.py 层处理：{database_url}")
    return dsn.sqlite_path or ":memory:"


def connect_dsn(dsn: DatabaseDsn, *, create_parents: bool = True) -> Any:
    """按已解析的连接串建连接。"""
    if dsn.is_sqlite:
        return _connect_sqlite(dsn, create_parents=create_parents)
    return _connect_postgresql(dsn)


def _connect_sqlite(dsn: DatabaseDsn, *, create_parents: bool = True) -> sqlite3.Connection:
    """打开（必要时创建）SQLite 连接并套上统一的 pragma。"""
    path = dsn.sqlite_path or ":memory:"
    if path != ":memory:" and create_parents:
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _connect_postgresql(dsn: DatabaseDsn) -> Any:
    """连 PostgreSQL。

    驱动（``psycopg``）不随默认依赖安装：病区侧单机部署用 SQLite，不需要它；
    真要换 PostgreSQL 时装一次驱动即可，**产品代码不用改**。
    驱动缺失时点名报错并给出安装命令，不静默退回 SQLite——那会把数据写进错误的库。
    """
    try:
        import psycopg  # type: ignore[import-not-found]
        from psycopg.rows import dict_row  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - 本机未装该驱动
        raise BackendUnavailableError(
            "DATABASE_URL 指向 PostgreSQL，但本机没装 PostgreSQL 驱动。"
            "安装一次即可（产品代码无需改动）：pip install 'psycopg[binary]'"
        ) from exc

    conn = psycopg.connect(**dsn.connection_kwargs())
    # dict_row 支持 row["列名"] 与 dict(row)，与 sqlite3.Row 的用法一致，
    # 仓储层因此不需要为两种后端各写一套取值逻辑。
    conn.row_factory = dict_row
    return conn


def connect(database_url: str, *, create_parents: bool = True) -> Any:
    """打开数据库连接（SQLite 默认；PostgreSQL 只改连接串）。"""
    return connect_dsn(parse_dsn(database_url), create_parents=create_parents)


def _in_transaction(conn: Any) -> bool:
    """当前连接是否已在事务里（SQLite 与 psycopg 各问各的）。"""
    if isinstance(conn, sqlite3.Connection):
        return bool(conn.in_transaction)
    info = getattr(conn, "info", None)  # pragma: no cover - 需 psycopg
    if info is None:  # pragma: no cover
        return False
    import psycopg  # type: ignore[import-not-found]  # pragma: no cover

    return info.transaction_status != psycopg.pq.TransactionStatus.IDLE


@contextmanager
def transaction(conn: Any) -> Iterator[Any]:
    """写事务：并发写显式串行化，异常回滚不静默改数据。

    SQLite 走 ``BEGIN IMMEDIATE``（拿写锁再读，避免两路确认同时通过检查后双写）；
    PostgreSQL 走行锁 + ``READ COMMITTED`` 的默认事务——那里行锁本身就是串行化的，
    ``IMMEDIATE`` 是 SQLite 独有的语义，硬套过去会报语法错。

    可重入嵌套：已在事务中时只借用外层事务（不重复 BEGIN），由最外层决定提交或回滚
    —— 让「业务写入 + 审计写入同事务」这类组合能靠嵌套调用自然实现。
    """
    if _in_transaction(conn):
        yield conn
        return

    if isinstance(conn, sqlite3.Connection):
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return

    with conn.transaction():  # pragma: no cover - 需 psycopg
        yield conn