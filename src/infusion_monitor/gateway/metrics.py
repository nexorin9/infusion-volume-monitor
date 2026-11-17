"""网关可观测性与成本护栏：调用数、token 用量、延迟分位、失败率、每日上限。

这个模块解决两个会在病区出事、而**不会报错**的问题：

1. **模型悄悄烧钱**。模型不可用时系统照常工作（退阈值路径），这恰恰是最容易
   没人发现「它一天调了八万次」的场景——功能全绿、账单在涨。所以调用数、token
   用量、失败率是**进程内**即时可查的量，不是月底才有人翻的账单。
2. **模型悄悄变慢**。端点从 300ms 变成 4s，护士的反应是「今天电脑好卡」，不会
   去查是模型。所以延迟按分位（p50/p95/max）而不是均值上报：均值被一串 20ms
   的快请求压住，p95 才会跟着端点变慢一起涨。

三条纪律：

1. **护栏只挡，不改写**。单请求 token 上限被越过时**不静默截断**成上限内的值——
   静默截断会让审计里记的 ``max_tokens`` 与端点实际收到的不是一回事，出事时
   查不出来。挡住 → 落 ``quota_exceeded`` → 退确定性阈值路径 → 护士照常拿到结论。
2. **额度按院内工作日算**。UTC 的日界落在病区凌晨两点，那时正是护士交班前点名
   换液的时段；按 ``+08:00`` 切日才对得上「今天」这个说法。
3. **内存有界**。延迟样本、每请求桶都带上限（滑动窗口 + LRU）。病区这台机器要
   同时跑视觉与网关，不能因为统计而慢慢吃满内存；token/调用数是**精确累计**，
   丢的只有延迟分布的历史样本。

**进程内 vs 落库**：这里只回答「现在这个进程烧了多少」，跨重启的历史口径在
``inference_audit`` 表（见 :mod:`gateway.audit`）。两者不互相替代——护士长问
「今天一共调了多少次」看摘要，问「昨天为什么降级」查审计表。
"""

from __future__ import annotations

import math
import threading
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

__all__ = [
    "DEFAULT_LATENCY_WINDOW",
    "DEFAULT_REQUEST_BUCKETS",
    "QUOTA_EXCEEDED",
    "GatewayMetrics",
    "LatencyWindow",
    "ModelStats",
    "QuotaDecision",
    "QuotaGuard",
    "QuotaPolicy",
    "RequestStats",
    "percentile",
]

#: 降级原因码：额度或 token 上限被越过。与 ``gateway.audit.FALLBACK_REASONS``
#: 同名同义——审计表里那一条 ``fallback_reason`` 就是它。
QUOTA_EXCEEDED = "quota_exceeded"

#: 延迟滑动窗口长度。1024 个样本足够把 p95 估稳（分位数的有效样本下限就在这个
#: 量级），也足够小到几十个模型统计桶加起来仍在几 MB 之内。
DEFAULT_LATENCY_WINDOW = 1024

#: 按 request_id 保留的调用明细条数（LRU）。一次巡视十几袋，一天几十轮，128 足够
#: 覆盖「护士长正在看的这一轮」；再多就只是历史，而历史在审计表里。
DEFAULT_REQUEST_BUCKETS = 256

#: 院内日界偏移（小时）。中国大陆固定 UTC+8，不依赖 tzdata——病区服务器上少装一个
#: 时区库就少一个「时间不对」的排查方向。
BUSINESS_TZ_OFFSET_HOURS = 8


def percentile(values: list[int] | tuple[int, ...] | deque[int], q: float) -> int | None:
    """最近秩分位数（nearest-rank），空样本返回 ``None``。

    为什么不用「线性插值」那些包：分位数在这里是给**人**看的运行指标，最近秩
    的定义是「第 ⌈q×n⌉ 个样本」，用不着插值；而且返回真实出现过的那个延迟值，
    运维拿它去对照日志能对上，插值出来的 137ms 在日志里并不存在。
    """
    if not values:
        return None
    ordered = sorted(values)
    if q <= 0:
        return int(ordered[0])
    rank = min(len(ordered), max(1, math.ceil(q * len(ordered))))
    return int(ordered[rank - 1])


@dataclass
class LatencyWindow:
    """有界的延迟样本窗。统计口径：p50 / p95 / max / mean / 样本数。"""

    maxlen: int = DEFAULT_LATENCY_WINDOW
    samples: deque[int] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.samples, deque):
            self.samples = deque(self.samples, maxlen=self.maxlen)
        else:
            self.samples = deque(self.samples, maxlen=self.maxlen)

    def add(self, latency_ms: int) -> None:
        if latency_ms is None:
            return
        self.samples.append(int(latency_ms))

    def __len__(self) -> int:
        return len(self.samples)

    def as_dict(self) -> dict[str, Any]:
        if not self.samples:
            return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None, "mean_ms": None}
        ordered = sorted(self.samples)
        return {
            "count": len(ordered),
            "p50_ms": percentile(ordered, 0.50),
            "p95_ms": percentile(ordered, 0.95),
            "max_ms": int(ordered[-1]),
            "mean_ms": round(sum(ordered) / len(ordered), 1),
        }


@dataclass
class ModelStats:
    """单个模型名的桶：调用数、token、延迟、失败率。"""

    model: str
    invocations: int = 0
    requests_sent: int = 0
    calls_ok: int = 0
    calls_failed: int = 0
    quota_blocked: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_missing: int = 0
    latency: LatencyWindow = field(default_factory=LatencyWindow)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def failure_rate(self) -> float:
        return round(self.calls_failed / self.invocations, 4) if self.invocations else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "invocations": self.invocations,
            "requests_sent": self.requests_sent,
            "calls_ok": self.calls_ok,
            "calls_failed": self.calls_failed,
            "failure_rate": self.failure_rate,
            "quota_blocked": self.quota_blocked,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "usage_missing": self.usage_missing,
            "latency": self.latency.as_dict(),
        }


@dataclass
class RequestStats:
    """单个 ``request_id`` 的调用明细（LRU 里的一项）。"""

    request_id: str
    model: str
    kind: str
    ok: bool
    attempts: int = 0
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    decision_source: str | None = None
    fallback_reason: str | None = None
    blocked: bool = False
    at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "kind": self.kind,
            "ok": self.ok,
            "attempts": self.attempts,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "decision_source": self.decision_source,
            "fallback_reason": self.fallback_reason,
            "blocked": self.blocked,
            "at": self.at,
        }


def _now_iso(clock: Callable[[], datetime] | None = None) -> str:
    return (clock() if clock else datetime.now(timezone.utc)).isoformat(timespec="seconds")


class GatewayMetrics:
    """网关的进程内用量与延迟统计。线程安全。

    **一次调用记两笔、互不重复**：传输层事实（发了几个请求、延迟多少、token 多少）
    由 :meth:`note_result` 记；结论来源三态（模型定的 / 降级 / 没调）由
    :meth:`note_decision` 记。调用数只由前者累加，后者只动分类计数——否则直连
    :meth:`~gateway.model_gateway.ModelGateway.complete` 的路径会漏记，而经
    ``advise`` 的路径会重复记。
    """

    def __init__(
        self,
        *,
        latency_window: int = DEFAULT_LATENCY_WINDOW,
        request_buckets: int = DEFAULT_REQUEST_BUCKETS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._latency_window = latency_window
        self._request_buckets = max(1, request_buckets)
        self.invocations = 0
        self.requests_sent = 0
        self.calls_ok = 0
        self.calls_failed = 0
        self.quota_blocked = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage_missing = 0
        self.by_kind: Counter[str] = Counter()
        self.by_decision_source: Counter[str] = Counter()
        self.by_model: dict[str, ModelStats] = {}
        self._latency = LatencyWindow(maxlen=latency_window)
        self._requests: "OrderedDict[str, RequestStats]" = OrderedDict()

    # ── 记一笔 ────────────────────────────────────────────────────────────
    def _model_bucket(self, model: str) -> ModelStats:
        bucket = self.by_model.get(model)
        if bucket is None:
            bucket = ModelStats(model=model, latency=LatencyWindow(maxlen=self._latency_window))
            self.by_model[model] = bucket
        return bucket

    def note_result(
        self,
        *,
        model: str,
        kind: str,
        ok: bool,
        requests_sent: int = 0,
        latency_ms: int = 0,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        request_id: str | None = None,
        blocked: bool = False,
    ) -> None:
        """记一次网关调用（无论成败）。

        :param requests_sent: 实际发出的 HTTP 次数（含退避重试）。额度挡住时为 0
            ——这正是「护栏省下的钱」，值得单独看。
        :param blocked: 是否因额度/上限被挡下（一次 HTTP 都没发）。
        """
        with self._lock:
            self.invocations += 1
            self.requests_sent += max(0, int(requests_sent))
            if blocked:
                self.quota_blocked += 1
            if ok:
                self.calls_ok += 1
            else:
                self.calls_failed += 1
            self.by_kind[kind] += 1
            if prompt_tokens is None:
                self.usage_missing += 1
            else:
                self.prompt_tokens += int(prompt_tokens)
            if completion_tokens is None:
                self.usage_missing += 1
            else:
                self.completion_tokens += int(completion_tokens)
            if latency_ms:
                self._latency.add(latency_ms)

            bucket = self._model_bucket(model)
            bucket.invocations += 1
            bucket.requests_sent += max(0, int(requests_sent))
            if blocked:
                bucket.quota_blocked += 1
            if ok:
                bucket.calls_ok += 1
            else:
                bucket.calls_failed += 1
            if prompt_tokens is not None:
                bucket.prompt_tokens += int(prompt_tokens)
            if completion_tokens is None:
                bucket.usage_missing += 1
            if completion_tokens is not None:
                bucket.completion_tokens += int(completion_tokens)
            if latency_ms:
                bucket.latency.add(latency_ms)

            if request_id:
                self._touch(
                    RequestStats(
                        request_id=request_id,
                        model=model,
                        kind=kind,
                        ok=ok,
                        attempts=int(requests_sent),
                        latency_ms=int(latency_ms or 0),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        blocked=blocked,
                        at=_now_iso(self._clock),
                    )
                )

    def note_decision(
        self,
        *,
        model: str,
        decision_source: str,
        request_id: str | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        """记一次结论收口（``advise`` 路径）。**不**累加调用数。"""
        with self._lock:
            self.by_decision_source[decision_source] += 1
            if request_id:
                entry = self._requests.get(request_id)
                if entry is not None:
                    entry.decision_source = decision_source
                    entry.fallback_reason = fallback_reason
                else:
                    self._touch(
                        RequestStats(
                            request_id=request_id,
                            model=model,
                            kind="ok" if decision_source == "model_decided" else "settled",
                            ok=decision_source == "model_decided",
                            decision_source=decision_source,
                            fallback_reason=fallback_reason,
                            at=_now_iso(self._clock),
                        )
                    )

    def _touch(self, entry: RequestStats) -> None:
        self._requests[entry.request_id] = entry
        self._requests.move_to_end(entry.request_id)
        while len(self._requests) > self._request_buckets:
            self._requests.popitem(last=False)

    # ── 查 ────────────────────────────────────────────────────────────────
    def request_stats(self, request_id: str) -> dict[str, Any] | None:
        """单个请求的明细；已被 LRU 挤出则返回 ``None``（历史在审计表里）。"""
        with self._lock:
            entry = self._requests.get(request_id)
            return entry.as_dict() if entry else None

    def recent_requests(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._requests.values())[-max(0, limit):]
            return [item.as_dict() for item in items]

    @property
    def failure_rate(self) -> float:
        return round(self.calls_failed / self.invocations, 4) if self.invocations else 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def snapshot(self) -> dict[str, Any]:
        """进运行摘要的快照（纯读，无副作用）。"""
        with self._lock:
            return {
                "invocations": self.invocations,
                "requests_sent": self.requests_sent,
                "calls_ok": self.calls_ok,
                "calls_failed": self.calls_failed,
                "failure_rate": self.failure_rate,
                "quota_blocked": self.quota_blocked,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "usage_missing": self.usage_missing,
                "by_kind": dict(self.by_kind),
                "by_decision_source": dict(self.by_decision_source),
                "by_model": {name: stat.as_dict() for name, stat in self.by_model.items()},
                "latency": self._latency.as_dict(),
                "tracked_requests": len(self._requests),
                "request_bucket_limit": self._request_buckets,
            }

    def reset(self) -> None:
        """清零（测试与热加载用）。"""
        with self._lock:
            self.invocations = 0
            self.requests_sent = 0
            self.calls_ok = 0
            self.calls_failed = 0
            self.quota_blocked = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.usage_missing = 0
            self.by_kind = Counter()
            self.by_decision_source = Counter()
            self.by_model = {}
            self._latency = LatencyWindow(maxlen=self._latency_window)
            self._requests = OrderedDict()


# ── 成本护栏 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QuotaPolicy:
    """两条护栏，都可以用环境变量关掉（限值 ≤ 0 即不限制）。

    :param daily_call_limit: 每日模型调用上限（``LLM_DAILY_CALL_LIMIT``）。
        按院内工作日（UTC+8）切日。0 = 不限制。
    :param max_tokens_per_request: 单请求 ``max_tokens`` 上限
        （``LLM_MAX_TOKENS_PER_REQUEST``）。0 = 不限制。
    """

    daily_call_limit: int = 0
    max_tokens_per_request: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "daily_call_limit": self.daily_call_limit,
            "max_tokens_per_request": self.max_tokens_per_request,
        }


@dataclass(frozen=True)
class QuotaDecision:
    """一次调用的放行判定。``allowed=False`` 时 ``reason`` 恒为 ``quota_exceeded``。"""

    allowed: bool
    reason: str | None = None
    detail: str | None = None
    used_today: int = 0
    limit: int = 0
    remaining: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "detail": self.detail,
            "used_today": self.used_today,
            "limit": self.limit,
            "remaining": self.remaining,
        }


class QuotaGuard:
    """按院内工作日记账的上限闸门。线程安全。

    **为什么自己记而不是去问端点**：问端点要发一次 HTTP 才知道额度，那一次本身
    就是超限后的第一笔开销；而院方给的额度上限本来就是我们与信息科之间的约定，
    端点侧未必有额度概念。本地记账的代价是「网关重启后今天已用的量会清零」，
    所以上限要按**病区一天的总调用量**留够余量，而不是贴着瞬时峰值配。
    """

    def __init__(
        self,
        policy: QuotaPolicy | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.policy = policy or QuotaPolicy()
        self._clock = clock
        self._lock = threading.Lock()
        self._day = self._business_day()
        self._used = 0

    # ── 内部 ──────────────────────────────────────────────────────────────
    def _now(self) -> datetime:
        return self._clock() if self._clock else datetime.now(timezone.utc)

    def _business_day(self) -> str:
        local = self._now().astimezone(timezone(timedelta(hours=BUSINESS_TZ_OFFSET_HOURS)))
        return local.date().isoformat()

    def _roll(self) -> None:
        today = self._business_day()
        if today != self._day:
            # 跨日即清零：护栏按「今天还能花多少」服务，不是「这个进程一共花过多少」。
            self._day = today
            self._used = 0

    # ── 用 ────────────────────────────────────────────────────────────────
    def authorize(self, *, max_tokens: int | None = None) -> QuotaDecision:
        """占一次调用名额并放行；被挡时**不**记账（它本来也没花出去）。"""
        with self._lock:
            self._roll()
            limit = self.policy.daily_call_limit
            token_cap = self.policy.max_tokens_per_request

            # 先看单请求上限：那是配置错配，属于「怎么调都不会自己好」的一类。
            if token_cap > 0 and max_tokens is not None and int(max_tokens) > token_cap:
                return QuotaDecision(
                    allowed=False,
                    reason=QUOTA_EXCEEDED,
                    detail=(
                        f"quota_exceeded: 本次请求 max_tokens={int(max_tokens)} 超过单请求上限 "
                        f"{token_cap}（LLM_MAX_TOKENS_PER_REQUEST）。护栏不做静默截断——"
                        f"把配置调大，或把这次要的输出长度降到上限内"
                    ),
                    used_today=self._used,
                    limit=token_cap,
                    remaining=None,
                )

            if limit > 0 and self._used >= limit:
                return QuotaDecision(
                    allowed=False,
                    reason=QUOTA_EXCEEDED,
                    detail=(
                        f"quota_exceeded: 今日模型调用已达上限 {limit} 次"
                        f"（LLM_DAILY_CALL_LIMIT，院内工作日 {self._day}）。"
                        f"本轮走确定性阈值路径；需要放开请调高上限"
                    ),
                    used_today=self._used,
                    limit=limit,
                    remaining=0,
                )

            self._used += 1
            return QuotaDecision(
                allowed=True,
                used_today=self._used,
                limit=limit,
                remaining=(limit - self._used) if limit > 0 else None,
            )

    def usage_today(self) -> int:
        with self._lock:
            self._roll()
            return self._used

    @property
    def day(self) -> str:
        with self._lock:
            self._roll()
            return self._day

    def remaining_today(self) -> int | None:
        """今天还能调几次；不限时为 ``None``。"""
        limit = self.policy.daily_call_limit
        if limit <= 0:
            return None
        with self._lock:
            self._roll()
            return max(0, limit - self._used)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            self._roll()
            return {
                "policy": self.policy.as_dict(),
                "business_day": self._day,
                "used_today": self._used,
                "remaining_today": (max(0, self.policy.daily_call_limit - self._used)
                                    if self.policy.daily_call_limit > 0 else None),
                "exhausted": self.policy.daily_call_limit > 0
                and self._used >= self.policy.daily_call_limit,
            }

    def reset(self) -> None:
        """清零当日记账（测试与人工放行用）。"""
        with self._lock:
            self._day = self._business_day()
            self._used = 0


def merge_usage(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """把 ``inference_audit`` 的若干行折成用量摘要（跨重启的历史口径）。

    与 :class:`GatewayMetrics` 的进程内口径**并存**：这个走库，所以算的是
    「落过审计的那些次」，两者的分母天然不同，摘要里分开列、不相加。
    """
    prompt = completion = 0
    missing = 0
    latencies: list[int] = []
    for row in rows:
        p, c = row.get("prompt_tokens"), row.get("completion_tokens")
        if p is None:
            missing += 1
        else:
            prompt += int(p)
        if c is None:
            missing += 1
        else:
            completion += int(c)
        if row.get("latency_ms") is not None:
            latencies.append(int(row["latency_ms"]))
    return {
        "rows": len(rows),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "usage_missing": missing,
        "latency": LatencyWindow(maxlen=len(latencies) or 1).as_dict()
        if latencies
        else {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None, "mean_ms": None},
    }