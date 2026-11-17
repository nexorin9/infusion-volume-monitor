"""数据库连接串抽象的测试：SQLite 默认、PostgreSQL 只改配置、仓储层不感知方言。

「换 PostgreSQL 只改连接串」这句话要有牙齿，就得能验证三件事：
连接串解析对、**凭据不外泄**、仓储层里**没有**方言分支。第三条用源码断言守住——
它比任何运行时测试都直接。
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pytest

from infusion_monitor.store import (
    BackendUnavailableError,
    UnsupportedDsnError,
    apply_migrations,
    connect,
    parse_dsn,
    redact_dsn,
    sqlite_dsn,
)

#: 仓储层源码的绝对路径（用于方言无关性断言）。
REPOSITORIES = Path(__file__).resolve().parents[1] / "src" / "infusion_monitor" / "store" / "repositories.py"


# ── 解析 ────────────────────────────────────────────────────────────────
def test_sqlite_dsn_variants_are_parsed():
    """SQLite 的三种写法都要认：相对路径、绝对路径、内存库。"""
    relative = parse_dsn("sqlite:///./var/infusion_monitor.db")
    assert relative.is_sqlite
    assert relative.sqlite_path == "./var/infusion_monitor.db"

    absolute = parse_dsn("sqlite:////var/lib/infusion-monitor/infusion_monitor.db")
    assert absolute.sqlite_path == "/var/lib/infusion-monitor/infusion_monitor.db"

    memory = parse_dsn("sqlite:///:memory:")
    assert memory.sqlite_path == ":memory:"


def test_relative_sqlite_path_is_not_turned_absolute():
    """``sqlite:///./var/x.db`` 里的相对路径不能被当成根目录下的绝对路径。"""
    dsn = parse_dsn("sqlite:///./var/x.db")
    assert not dsn.sqlite_path.startswith("/")


def test_postgresql_dsn_is_parsed():
    """PostgreSQL 连接串：主机、端口、库名、用户都要解出来。"""
    dsn = parse_dsn("postgresql://nurse:secret@db.internal:5433/infusion?sslmode=require")
    assert dsn.is_postgresql
    assert dsn.host == "db.internal"
    assert dsn.port == 5433
    assert dsn.database == "infusion"
    assert dsn.user == "nurse"
    assert dsn.options["sslmode"] == "require"


def test_postgres_alias_is_accepted():
    """``postgres://`` 是 libpq 的别名，与 ``postgresql://`` 等价。"""
    dsn = parse_dsn("postgres://u:p@h:5432/d")
    assert dsn.is_postgresql
    assert dsn.database == "d"


def test_postgresql_default_port():
    """没写端口时用 5432（与 libpq 一致）。"""
    assert parse_dsn("postgresql://u:p@h/d").port == 5432


def test_unknown_scheme_is_rejected_by_name():
    """不认识的 scheme 直接点名报错，**不静默退回 SQLite**。

    把拼错的 PostgreSQL 连接串当成 SQLite 用，会让服务在错误的库上安静地写数据——
    比启动失败难查得多。
    """
    with pytest.raises(UnsupportedDsnError) as excinfo:
        parse_dsn("mysql://user:pw@host/db")
    message = str(excinfo.value)
    assert "mysql" in message
    assert "sqlite" in message and "postgresql" in message


def test_empty_dsn_is_rejected():
    with pytest.raises(UnsupportedDsnError):
        parse_dsn("")


# ── 凭据不外泄 ──────────────────────────────────────────────────────────
def test_describe_never_exposes_the_password():
    """describe() 是给运行摘要与启动日志用的，绝不能带明文密码。"""
    dsn = parse_dsn("postgresql://nurse:hunter2@db.internal/infusion")
    described = dsn.describe()
    assert "hunter2" not in str(described)
    assert described["password"] == "***"
    assert described["user"] == "nurse"


def test_redact_dsn_masks_the_password():
    """连接串本身也要能打日志：密码段被换成 ***。"""
    masked = redact_dsn("postgresql://nurse:hunter2@db.internal:5432/infusion")
    assert "hunter2" not in masked
    assert masked == "postgresql://nurse:***@db.internal:5432/infusion"


def test_repr_of_dsn_does_not_leak_the_password():
    """dataclass 的 repr 会被日志与异常栈打出来——密码不能出现在里面。"""
    dsn = parse_dsn("postgresql://nurse:hunter2@db.internal/infusion")
    assert "hunter2" not in repr(dsn)


def test_connection_kwargs_are_the_only_place_with_the_plain_password():
    """明文密码只出现在真正建连接的那一刻用的参数里。"""
    dsn = parse_dsn("postgresql://nurse:hunter2@db.internal/infusion")
    assert dsn.connection_kwargs()["password"] == "hunter2"
    assert "hunter2" not in str(dsn.describe())


# ── 连接 ────────────────────────────────────────────────────────────────
def test_connect_still_works_for_sqlite(tmp_path):
    """默认路径（SQLite）完全不受影响。"""
    path = tmp_path / "nested" / "infusion_monitor.db"
    conn = connect(sqlite_dsn(path))
    try:
        assert isinstance(conn, sqlite3.Connection)
        assert path.exists()
        apply_migrations(conn)
        assert conn.execute("SELECT COUNT(*) FROM bag_session").fetchone()[0] == 0
    finally:
        conn.close()


def test_connect_postgresql_without_driver_names_the_driver(monkeypatch):
    """本机没装 PostgreSQL 驱动时点名说清楚，不静默退回 SQLite。

    退回去意味着连接写进错误的库——那种故障要到数据对不上时才发现。
    """
    # 让 ``import psycopg`` 失败，模拟「病区机上没装驱动」
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(BackendUnavailableError) as excinfo:
        connect("postgresql://nurse:pw@db.internal/infusion")
    message = str(excinfo.value)
    assert "psycopg" in message
    assert "pip install" in message


def test_postgresql_url_never_silently_falls_back_to_sqlite(monkeypatch):
    """连接串指向 PostgreSQL 时，绝不能返回一个 SQLite 连接。

    无论驱动装没装、库连不连得上，结果都必须是「报错」而不是「换个库接着写」。
    """
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(BackendUnavailableError):
        connect("postgresql://nurse:pw@db.internal/infusion")


def test_sqlite_dsn_helper_builds_the_url():
    assert sqlite_dsn("./var/x.db") == "sqlite:///./var/x.db"
    assert sqlite_dsn(":memory:") == "sqlite:///:memory:"


# ── 仓储层不感知方言 ─────────────────────────────────────────────────────
def test_repository_layer_has_no_dialect_specific_sql():
    """仓储层里不许出现 SQLite 独有的写法。

    ``BEGIN IMMEDIATE`` / ``PRAGMA`` / ``sqlite_master`` / ``INSERT OR REPLACE``
    都是 SQLite 方言；它们一旦漏进仓储层，「换 PostgreSQL 只改连接串」就是空话。
    方言只允许出现在 store/dsn.py 与 store/db.py 两个文件里。
    """
    source = REPOSITORIES.read_text(encoding="utf-8")
    # 去掉文档字符串与注释再查：模块说明里提到这些词不算违规。
    code = re.sub(r'""".*?"""', "", source, flags=re.S)
    code = re.sub(r"#.*", "", code)
    for forbidden in ("BEGIN IMMEDIATE", "PRAGMA", "sqlite_master", "INSERT OR REPLACE", "AUTOINCREMENT"):
        assert forbidden not in code, f"仓储层出现了 SQLite 方言写法：{forbidden}"


def test_dialect_knowledge_is_confined_to_the_connection_and_schema_layers():
    """方言知识只允许待在两个地方：连接层（dsn/db）与 DDL 层（schema）。

    * **连接层**（``dsn.py`` / ``db.py``）：解析连接串、选驱动、开事务——本来就该知道方言；
    * **DDL 层**（``schema.py``）：表结构是 SQLite 方言（``AUTOINCREMENT`` 等），
      换 PostgreSQL 时这一处要翻译，其余模块不用动；
    * **仓储层与迁移层**（``repositories.py`` / ``migrations.py``）：零方言分支。
    """
    store_dir = REPOSITORIES.parent
    for path in sorted(store_dir.glob("*.py")):
        if path.name in {"dsn.py", "db.py", "schema.py"}:
            continue
        code = re.sub(r'""".*?"""', "", path.read_text(encoding="utf-8"), flags=re.S)
        code = re.sub(r"#.*", "", code)
        for forbidden in ("BEGIN IMMEDIATE", "PRAGMA", "sqlite_master", "INSERT OR REPLACE", "AUTOINCREMENT"):
            assert forbidden not in code, f"{path.name} 里出现了 SQLite 方言写法：{forbidden}"


def test_schema_ddl_is_the_single_place_dialect_specific_ddl_lives():
    """``AUTOINCREMENT`` 这类方言 DDL 集中在 schema.py 一处，换库时只改这一处。"""
    store_dir = REPOSITORIES.parent
    hits = []
    for path in sorted(store_dir.glob("*.py")):
        code = re.sub(r'""".*?"""', "", path.read_text(encoding="utf-8"), flags=re.S)
        code = re.sub(r"#.*", "", code)
        if "AUTOINCREMENT" in code:
            hits.append(path.name)
    assert hits == ["schema.py"], f"方言 DDL 散落在 {hits}，应当集中在 schema.py"


def test_transaction_serialises_writers_on_sqlite(tmp_path):
    """SQLite 上写事务仍然串行化（``BEGIN IMMEDIATE`` 拿写锁）。"""
    path = tmp_path / "x.db"
    first = connect(sqlite_dsn(path))
    second = connect(sqlite_dsn(path))
    try:
        apply_migrations(first)
        first.execute("BEGIN IMMEDIATE")
        first.execute(
            "INSERT INTO bag_session (session_id, rack_id, bed_no, infusion_order_no, "
            "shift_key, bag_seq, frames_dir, captured_at, created_at, updated_at) "
            "VALUES ('s1','r','b','o','s','B01','/tmp/x','t','t','t')"
        )
        # 另一条连接在锁释放前写不进去
        with pytest.raises(sqlite3.OperationalError):
            second.execute(
                "INSERT INTO bag_session (session_id, rack_id, bed_no, infusion_order_no, "
                "shift_key, bag_seq, frames_dir, captured_at, created_at, updated_at) "
                "VALUES ('s2','r','b','o','s','B02','/tmp/x','t','t','t')"
            )
        first.execute("ROLLBACK")
    finally:
        first.close()
        second.close()
