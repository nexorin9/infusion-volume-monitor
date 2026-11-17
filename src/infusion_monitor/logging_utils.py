"""结构化日志（JSON 行）+ ``request_id`` 全链路贯穿。

病区侧上机后，信息科遇到的第一个问题永远是「昨晚那次到底怎么回事」。答案不该在
某个人的滚动屏幕里——它应该在服务自己的日志里，按 ``request_id`` 一捞就是一整条链：
入站上传 → 重锚判定 → 换液点名 → 护士确认 → 出站写回 → 模型判定与降级。

**为什么是 JSON 行而不是人读的表格**：日志要被 ``grep``/``jq``/日志采集器接住，
一行一个 JSON 对象时排障脚本能直接 ``jq 'select(.event=="writeback.pending")'``，
不用正则去啃人读格式。代价是肉眼看略费劲——所以保留 ``configure_logging`` 的
``human=True`` 一行速览开关，排障时临时打开即可，不必改格式。

**request_id 怎么贯穿**：入站请求头 ``X-Request-Id`` 优先（调用方可自带，便于把
移动护理终端那一侧的日志和服务日志对上），没有就生成一个。它存进 :mod:`contextvars`
的上下文变量，于是**任何**一层代码不用层层传参就能拿到同一份 id：业务日志、
出站写回审计、推理审计表全都挂同一个 id。中间件把它回写到响应头 ``X-Request-ID``，
护士端把出问题那次的响应头截图发给信息科，就能定位到唯一一行。

**凭据不进日志**：:func:`redact` 按键名把 token / password / api_key / 授权头
换成 ``***``。凭据位只有 7 组，但日志会流向采集器、告警群与飞检材料——一处泄漏
就要全院换密钥。
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterable, Iterator, Mapping, MutableMapping

__all__ = [
    "HEADER_NAME",
    "JSONFormatter",
    "RuntimeCounters",
    "current_request_id",
    "configure_logging",
    "install_request_logging",
    "log_event",
    "masked_fields",
    "new_request_id",
    "redact",
    "request_id_of",
    "request_scope",
]


#: 入站请求头与出站响应头共用的名字（大小写不敏感，HTTP 头本身如此）。
HEADER_NAME = "X-Request-Id"

#: 进程内 ``request_id``。上下文变量而非线程局部量：FastAPI 的 ``async def``
#: 处理函数与退避重试用的线程池会在不同上下文里跑，只用 threading.local 会在
#: ``run_in_threadpool`` 段里丢掉 id，出站审计与入站日志就对不上了。
_REQUEST_ID: ContextVar[str | None] = ContextVar("infusion_request_id", default=None)

#: 日志里必须打码的键名（小写比较）。命中即整体替换，不做部分匹配保留尾巴——
#: 半截 token 同样能拼出可用的凭据。
_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "mqtt_password",
        "llm_api_key",
        "nurse_exec_api_token",
        "password",
        "secret",
        "token",
        "webhook_signing_secret",
    }
)

_MASK = "***"

#: 第三方库的 INFO 噪声（uvicorn 每条访问日志、httpx 每次连接）在病区侧没有价值，
#: 反而把真正的业务事件淹掉。默认压到 WARNING，需要时再放开。
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "hpack",
    "paho.mqtt.client",
    "multipart",
    "uvicorn.access",
)


def new_request_id() -> str:
    """生成一个请求 id（16 位十六进制，与库里其它 id 前缀风格一致）。"""
    return f"req_{uuid.uuid4().hex[:16]}"


def current_request_id() -> str | None:
    """当前上下文里的 ``request_id``；不在请求上下文内返回 None。"""
    return _REQUEST_ID.get()


def set_request_id(value: str) -> Token:
    return _REQUEST_ID.set(value)


def reset_request_id(token: Token) -> None:
    _REQUEST_ID.reset(token)


@contextmanager
def request_scope(request_id: str) -> Iterator[str]:
    """把 ``request_id`` 绑到当前上下文（中间件与测试都用它）。"""
    token = _REQUEST_ID.set(request_id)
    try:
        yield request_id
    finally:
        _REQUEST_ID.reset(token)


def request_id_of(request: Any = None) -> str:
    """取本次请求的 id：上下文变量 → 入站请求头 → 新生成。

    顺序有讲究：中间件先进上下文，所以路由里调它拿到的一定是同一个 id；
    直接在无中间件的场合调用（例如单元测试直接调业务函数）时回落到请求头。
    """
    existing = current_request_id()
    if existing:
        return existing
    header = None
    if request is not None:
        try:
            header = request.headers.get(HEADER_NAME)
        except Exception:  # pragma: no cover - 非 Request 对象（测试替身）
            header = None
    value = (header or "").strip()
    return value or new_request_id()


def redact(value: Any, *, extra_keys: Iterable[str] = ()) -> Any:
    """递归打码：命中敏感键名的值整条替换成 ``***``。

    保持结构不变（dict / list 照旧递归），这样采集器侧的字段解析不会因为脱敏而断。
    """
    blocked = _SENSITIVE_KEYS | {k.lower() for k in extra_keys}
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in blocked:
                out[str(key)] = _MASK
            else:
                out[str(key)] = redact(item, extra_keys=extra_keys)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(item, extra_keys=extra_keys) for item in value]
    return value


def masked_fields(record: Mapping[str, Any], extra_keys: Iterable[str] = ()) -> dict[str, Any]:
    """:func:`redact` 的 dict 快捷方式（日志字段本来就是 dict）。"""
    return redact(dict(record), extra_keys=extra_keys)


class JsonFormatter(logging.Formatter):
    """一行一个 JSON 对象。

    ``exc_info`` 被折成 ``error`` 字段（异常类型 + 消息，不带整段 traceback——
    traceback 逐条进采集器会把量放大几十倍，``LOG_DEBUG=1`` 时才带全文）。
    """

    def __init__(self, *, service: str = "infusion-volume-monitor",
                 include_traceback: bool = False) -> None:
        super().__init__()
        self.service = service
        self.include_traceback = include_traceback

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003 - 覆写标准库接口
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        request_id = current_request_id() or getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        for key, value in getattr(record, "fields", {}).items():
            if key not in payload:
                payload[key] = value
        if record.exc_info:
            exc = record.exc_info[1]
            payload["error"] = f"{type(exc).__name__}: {exc}" if exc else "unknown"
            if self.include_traceback:
                payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload), ensure_ascii=False, default=str)


class HumanFormatter(logging.Formatter):
    """排障时的一行速览：``时间 级别 request_id 事件 k=v ...``。

    与 JSON 行同源（同一个 ``fields`` 约定），只是给人看。采集器仍应指向 JSON。
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        fields = redact(dict(getattr(record, "fields", {})))
        rendered = " ".join(f"{k}={_short(v)}" for k, v in fields.items())
        request_id = current_request_id() or getattr(record, "request_id", None) or "-"
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        prefix = f"{stamp} {record.levelname:<7} {request_id:<20} {record.getMessage()}"
        return f"{prefix} {rendered}".rstrip()

    def formatTime(self, record, datefmt=None):  # noqa: A003, N802
        return time.strftime(datefmt or "%H:%M:%S", time.localtime(record.created))


def _short(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def configure_logging(
    level: str | int = "INFO",
    *,
    stream: Any = None,
    human: bool = False,
    service: str = "infusion-volume-monitor",
    noisy_loggers: Iterable[str] = _NOISY_LOGGERS,
) -> logging.Logger:
    """装好根 logger：一条 JSON 行 handler，重复调用不会叠加。

    ``human=True`` 出一行速览（排障用），默认 JSON 行（采集器用）。
    """
    root = logging.getLogger()
    root.setLevel(level if isinstance(level, int) else str(level).upper())
    formatter: logging.Formatter = (
        HumanFormatter()
        if human
        else JsonFormatter(service=service, include_traceback=level == "DEBUG")
    )
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(formatter)
    # 认准我们自己的 handler 再替换：pytest 等已经装过 handler 的场景下，
    # 盲删会把它们的输出也一起干掉。
    for existing in list(root.handlers):
        if getattr(existing, "_infusion_monitor", False):
            root.removeHandler(existing)
    handler._infusion_monitor = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("infusion_monitor")


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """发一条结构化日志：``log_event(log, logging.INFO, "session.created", session_id=…)``。

    ``fields`` 里的键会成为 JSON 顶层键；敏感键自动打码。``exc_info=True`` 可传入
    以附上异常摘要。
    """
    if not logger.isEnabledFor(level):
        return
    payload = {k: v for k, v in fields.items() if k not in {"exc_info", "stack_info"}}
    if fields.get("exc_info"):
        payload.setdefault("error", "see_exception")
    logger.log(
        level,
        event,
        extra={"event": event, "fields": payload},
        exc_info=fields.get("exc_info"),
    )


class RuntimeCounters:
    """进程内计数器（请求数、错误数、启动时刻）。

    只做「这个进程启动以来发生了什么」级别的粗账。跨重启的统计一律查 SQLite，
    不在内存里攒——护士长要看的是今天一整天，不是重启之后。
    """

    __slots__ = ("started_at", "started_monotonic", "request_count", "error_count", "last_request_at")

    def __init__(self) -> None:
        from datetime import datetime, timezone

        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.started_monotonic = time.monotonic()
        self.request_count = 0
        self.error_count = 0
        self.last_request_at: str | None = None

    def note_request(self, *, error: bool = False) -> None:
        self.request_count += 1
        if error:
            self.error_count += 1
        from datetime import datetime, timezone

        self.last_request_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def uptime_seconds(self) -> float:
        return round(time.monotonic() - self.started_monotonic, 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "uptime_seconds": self.uptime_seconds(),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "last_request_at": self.last_request_at,
        }


class RequestLoggingMiddleware:
    """给每个请求绑 ``request_id``、记账、并在响应头回写。

    走**纯 ASGI** 而不是 ``BaseHTTPMiddleware``：后者的响应体回放会让上传大 zip 的
    请求多一份内存拷贝，且在流式响应上有已知的边界问题。本服务只有 JSON 响应，
    自己包 ``send`` 就够了。
    """

    def __init__(self, app: Any, *, logger: logging.Logger | None = None,
                 counters: RuntimeCounters | None = None,
                 slow_request_ms: float = 2000.0) -> None:
        self.app = app
        self.logger = logger or logging.getLogger("infusion_monitor.http")
        self.counters = counters or RuntimeCounters()
        self.slow_request_ms = slow_request_ms

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        incoming = (headers.get(HEADER_NAME.lower()) or "").strip()
        request_id = incoming or new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id

        status_code = 500
        started = time.monotonic()

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
                raw = list(message.get("headers", []))
                raw.append((HEADER_NAME.encode("latin-1"), request_id.encode("latin-1")))
                message = {**message, "headers": raw}
            await send(message)

        with request_scope(request_id):
            try:
                await self.app(scope, receive, send_wrapper)
            except Exception:
                self.counters.note_request(error=True)
                self._emit(request_id, headers, scope, 500, started, failed=True)
                raise
            self.counters.note_request(error=status_code >= 500)
            self._emit(request_id, headers, scope, status_code, started)

    def _emit(self, request_id: str, headers: Mapping[str, str], scope: Mapping[str, Any],
              status_code: int, started: float, *, failed: bool = False) -> None:
        duration_ms = round((time.monotonic() - started) * 1000, 2)
        # 健康检查每秒可能被探活打一次，逐条记 INFO 会把真正的业务事件冲掉。
        if scope.get("path", "").endswith("/api/v1/health") and not failed:
            return
        level = logging.WARNING if failed or status_code >= 500 else logging.INFO
        log_event(
            self.logger,
            level,
            "http.error" if failed else "http.request",
            method=scope.get("method", "-"),
            path=scope.get("path", "-"),
            status=status_code,
            duration_ms=duration_ms,
            client=headers.get("x-forwarded-for") or headers.get("host", "-"),
            slow=bool(duration_ms >= self.slow_request_ms),
            request_id=request_id,
        )


def install_request_logging(app: Any, *, logger: logging.Logger | None = None,
                            slow_request_ms: float = 2000.0) -> RuntimeCounters:
    """把中间件与计数器装到 FastAPI 应用上，返回计数器供运行摘要读取。"""
    counters = RuntimeCounters()
    app.state.counters = counters
    app.add_middleware(RequestLoggingMiddleware, logger=logger, counters=counters,
                       slow_request_ms=slow_request_ms)
    return counters