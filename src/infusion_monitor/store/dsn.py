"""数据库连接串抽象：SQLite 默认，PostgreSQL 只改配置。

**为什么要这一层**：病区侧单机部署用 SQLite 就够（写入毫秒级，见 INSTALL.md
「SQLite 的并发边界」），但迟早会有人问「能不能换成 PostgreSQL」。如果方言判断
散落在仓储层的每个 ``if`` 里，这个问题的答案就是「不能，改不动」。所以方言只在本
模块与 :mod:`store.db` 两处出现：

* :func:`parse_dsn` 把连接串解析成 :class:`DatabaseDsn`（**不连库**，只解析）；
* :mod:`store.db` 按 ``dsn.backend`` 决定连哪个驱动、事务怎么开。

**仓储层（``store/repositories.py``）不感知方言**：它只发标准 SQL（``SELECT`` /
``INSERT`` / ``UPDATE`` / ``DELETE``），方言差异全部由本模块吸收。

安全纪律：连接串里可能带密码。:meth:`DatabaseDsn.describe` 与 :func:`redact_dsn`
**永远不吐明文**——日志、运行摘要、启动自检里出现的连接串必须是掩码后的。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qsl, unquote, urlsplit

__all__ = [
    "BackendUnavailableError",
    "DatabaseDsn",
    "UnsupportedDsnError",
    "parse_dsn",
    "redact_dsn",
    "sqlite_dsn",
]

#: 认可的连接串 scheme → 规范后端名。
_SCHEMES = {
    "sqlite": "sqlite",
    "sqlite3": "sqlite",
    "postgres": "postgresql",
    "postgresql": "postgresql",
}

#: PostgreSQL 的默认端口（与 libpq 一致）。
_DEFAULT_PG_PORT = 5432

#: 连接串里的密码片段（``scheme://user:password@host``）。
_PG_PASSWORD_RE = re.compile(r"(?<=://)([^:/@\s]+):([^@/\s]+)@")


class UnsupportedDsnError(ValueError):
    """连接串的 scheme 不认识——启动时点名报错，不静默退回 SQLite。"""


class BackendUnavailableError(RuntimeError):
    """连接串指向的后端在本机不可用（驱动没装 / 库连不上）。"""


@dataclass(frozen=True)
class DatabaseDsn:
    """解析后的连接串。

    只描述「连哪里」，不负责连。**明文密码只出现在 ``_password`` 一个字段里**，
    它不进 ``repr``、不参与比较；``raw`` 存的是**掩码后**的连接串——这个对象
    随时可能被日志、异常栈或运行摘要打印出去，不该带着明文。
    """

    raw: str
    backend: str
    sqlite_path: str | None = None
    host: str | None = None
    port: int | None = None
    database: str | None = None
    user: str | None = None
    options: Mapping[str, str] = field(default_factory=dict)
    _password: str | None = field(default=None, repr=False, compare=False)

    # ── 判别 ───────────────────────────────────────────────────────────
    @property
    def is_sqlite(self) -> bool:
        return self.backend == "sqlite"

    @property
    def is_postgresql(self) -> bool:
        return self.backend == "postgresql"

    # ── 对外可见的摘要（永远无明文凭据）───────────────────────────────
    def describe(self) -> dict[str, Any]:
        """可安全打印的连接串摘要（不含任何明文密码）。"""
        if self.is_sqlite:
            return {
                "backend": "sqlite",
                "path": self.sqlite_path,
                "raw": self.raw,
            }
        return {
            "backend": "postgresql",
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": "***" if self._password else "(未配置)",
            "options": dict(self.options),
            "raw": redact_dsn(self.raw),
        }

    def connection_kwargs(self) -> dict[str, Any]:
        """给驱动用的连接参数（**含明文密码**，仅在真正建连接的那一刻用）。"""
        if self.is_sqlite:
            return {"database": self.sqlite_path or ":memory:"}
        kwargs: dict[str, Any] = {"dbname": self.database or ""}
        if self.host:
            kwargs["host"] = self.host
        if self.port:
            kwargs["port"] = self.port
        if self.user:
            kwargs["user"] = self.user
        if self._password:
            kwargs["password"] = self._password
        if self.options:
            kwargs["options"] = dict(self.options)
        return kwargs


def redact_dsn(raw: str) -> str:
    """把连接串里的密码替换成 ``***``，其余原样返回。

    非 PostgreSQL 形态原样返回（SQLite 连接串不带密码）。
    """
    return _PG_PASSWORD_RE.sub(lambda m: f"{m.group(1)}:***@", raw)


def sqlite_dsn(path: str | Path) -> str:
    """由文件路径拼 SQLite 连接串（``sqlite:///abs/path`` 或 ``sqlite:///rel/path``）。"""
    text = str(path)
    if text == ":memory:":
        return "sqlite:///:memory:"
    return f"sqlite:///{text}"


def parse_dsn(database_url: str) -> DatabaseDsn:
    """解析连接串，**不连库**。

    支持：

    * ``sqlite:///./var/infusion_monitor.db``（默认，相对路径相对于进程工作目录）
    * ``sqlite:///:memory:`` / ``sqlite://``（内存库）
    * ``postgresql://user:pass@host:5432/dbname?sslmode=require``
    * ``postgres://…``（libpq 的别名，等同 postgresql）

    不认识的 scheme 抛 :class:`UnsupportedDsnError` 并把原串回显——**不静默退回
    SQLite**：把一个拼错的 PostgreSQL 连接串当成 SQLite 用，会让服务在错误的库上
    安静地写数据，比启动失败难查得多。
    """
    raw = (database_url or "").strip()
    if not raw:
        raise UnsupportedDsnError("数据库连接串为空（DATABASE_URL 未配置）")

    scheme = raw.split("://", 1)[0].lower() if "://" in raw else ""
    backend = _SCHEMES.get(scheme)
    if backend is None:
        raise UnsupportedDsnError(
            f"不认识的数据库连接串 scheme：{scheme or '(空)'}；"
            f"可用 sqlite:///… 或 postgresql://…（原串：{redact_dsn(raw)}）"
        )

    if backend == "sqlite":
        return _parse_sqlite(raw)

    return _parse_postgresql(raw)


def _parse_sqlite(raw: str) -> DatabaseDsn:
    # SQLite 的路径部分**不能**用 urlsplit 解析：`sqlite:///./var/x.db` 会被拆成
    # 绝对路径 `/./var/x.db`，把相对路径变成根目录下的路径——这正是老实现里
    # 手工切前缀的原因，这里保持同一口径。
    for prefix in ("sqlite:///", "sqlite://"):
        if raw.startswith(prefix):
            path = raw[len(prefix) :] or ":memory:"
            return DatabaseDsn(raw=raw, backend="sqlite", sqlite_path=path)

    # sqlite:xxx（无斜杠）这种形态不认：路径边界说不清。
    raise UnsupportedDsnError(f"SQLite 连接串形态不认：{redact_dsn(raw)}")


def _parse_postgresql(raw: str) -> DatabaseDsn:
    parts = urlsplit(raw)
    database = unquote(parts.path.lstrip("/")) or None
    user: str | None = None
    password: str | None = None
    host = parts.hostname
    port = parts.port

    if parts.username:
        user = unquote(parts.username)
    if parts.password:
        password = unquote(parts.password)
    if host is None and parts.netloc:
        # `postgresql:///dbname` 这种只给库名的写法：host 留空，交给驱动走本地 socket。
        host = None

    options = {key: value for key, value in parse_qsl(parts.query)}

    return DatabaseDsn(
        raw=redact_dsn(raw),
        backend="postgresql",
        host=host,
        port=port or _DEFAULT_PG_PORT,
        database=database,
        user=user,
        options=options,
        _password=password,
    )
