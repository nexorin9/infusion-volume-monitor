"""单一模型网关（OpenAI-compatible）：业务层只调它，不碰任何厂商 SDK。

**为什么只有一个网关**：院内端点可能是院方自建的推理服务、也可能是本地部署的
开源权重，但协议是同一种（``POST {LLM_BASE_URL}/chat/completions`` +
``Authorization: Bearer``）。业务层一旦在某个模块里直接 import 某个厂商 SDK，
换端点就要动业务代码——那正是「模型不可替换」的开始。所以这个模块是**唯一**发起
模型调用的地方，``base_url`` / ``api_key`` / ``model`` / ``timeout`` 全部配置化，
换服务只改环境变量。

四条纪律，每条对应一个真会在病区出事的地方：

1. **四类失败分开记，处置统一**。连接超时 / 读超时 / 5xx / 输出不可解析各自有
   独立的 :attr:`GatewayAttempt.kind`（排查方向不同：前两类看网络与端点负载，
   5xx 看端点自身，解析失败看提示词与端点版本），但**处置一样**——退避重试
   ≤2 次，仍不成则退确定性阈值路径。解析失败也重试是因为它是随机的（同一个模型
   重发一次常就给对了 JSON），而 4xx 业务拒绝是确定的（同一份请求重发只会再被
   拒一次），后者不重试。
2. **模型不得推进业务状态**。:meth:`ModelGateway.advise` 只产出
   :class:`GatewayAdvice` 这个**提示对象**，它没有写状态的代码路径，也拿不到状态
   仓储；``needs_change → replaced`` 仍然只能由护士在终端确认触发。降级路径同样
   不推进：模型不可用不是让系统自己往前走的理由。
3. **不可达必须留痕**。每一次调用（成功、降级、根本没调）都落一行
   ``inference_audit``，含 ``request_id`` / ``model`` / ``prompt_hash`` /
   ``latency_ms`` / token 用量 / ``decision_source`` / ``fallback_reason``。
   护士长问「这句话谁说的」，答案在库里，不用猜。
4. **凭据不进错误文本**。端点报错时可能把请求头原样回显（有的网关会），Bearer
   key 就会顺着 ``last_error`` 进库、进日志、进飞检材料。所有错误摘要经
   :func:`~adapters.nurse_exec.scrub` 抹一遍。

**线程模型**：HTTP 是阻塞调用，而 SQLite 连接带线程亲和性。同步版
:meth:`advise` 供后台线程与脚本调用；异步版 :meth:`advise_async` 只把**纯网络**
那一段丢进线程池，审计落库回到调用线程——绝不出现「在工作线程里写主线程的连接」。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import httpx

from ..adapters.nurse_exec import scrub
from .audit import GatewayAuditRecord, GatewayAuditor, prompt_digest
from .metrics import QUOTA_EXCEEDED, GatewayMetrics, QuotaGuard, QuotaPolicy
from .prompts import DEFAULT_TEMPLATE_NAME, PromptRegistry, default_registry
from .schemas import CONFIDENCE_FLOOR, ModelOutputRejected, parse_model_output

__all__ = [
    "BACKOFF_BASE_SECONDS",
    "BACKOFF_FACTOR",
    "GATEWAY_MAX_RETRIES",
    "OBSERVATION_FIELDS",
    "RETRYABLE_KINDS",
    "GatewayAdvice",
    "GatewayAttempt",
    "GatewayResult",
    "ModelGateway",
    "PromptMessages",
    "build_messages",
    "build_prompt",
    "classify_response",
    "extract_usage",
]

#: 初始尝试之外的重试次数上限。与出站写回同一条上限：无限重试等于把一次模型
#: 故障放大成病区侧的一次长时间卡顿，而结论在降级路径上本来就有。
GATEWAY_MAX_RETRIES = 2
#: 指数退避基数（秒）。比写回的小：模型调用本来就该是毫秒级的，让护士等 1 秒
#: 再看一个本来就有的确定性答案没有意义。
BACKOFF_BASE_SECONDS = 0.2
BACKOFF_FACTOR = 4

#: 值得退避重试的失败类别。``parse_error`` 在其中（随机输出，重发可能就对）；
#: ``http_4xx`` 不在其中（确定性的业务拒绝，重发只会再被拒一次）。
RETRYABLE_KINDS = (
    "connect_timeout",
    "read_timeout",
    "connect_error",
    "http_5xx",
    "parse_error",
)


def backoff_seconds(retry_index: int, *, base: float = BACKOFF_BASE_SECONDS,
                    factor: int = BACKOFF_FACTOR) -> float:
    """第 ``retry_index`` 次重试前等待的秒数（从 0 起）：base × factor^n。"""
    return base * (factor ** max(0, retry_index))


# ── 请求装配 ────────────────────────────────────────────────────────────────

#: 提示词里放哪些观测字段。**白名单**：模型只看得到判定需要的聚合量，
#: 不放帧序列、不放图像、不放床号以外的任何明细——一次巡视有 6 袋、每袋几十帧，
#: 把原始帧塞进提示词既不现实，也会把床位现场数据复制到模型侧。
OBSERVATION_FIELDS: tuple[str, ...] = (
    "bag_seq",
    "bed_no",
    "fill_ratio",
    "estimated_volume_ml",
    "eta_minutes",
    "eta_source",
    "calibration_state",
    "occluded_frame_count",
    "visible_frame_count",
    "frame_count",
    "latest_ratios",
    "threshold_decision",
    "change_threshold_ratio",
)


@dataclass(frozen=True)
class PromptMessages:
    """一次提示的装配结果：消息正文 + 模板身份 + 两类摘要。

    ``template_id`` 是审计里记的 ``prompt_version``；``content_hash`` 是
    「模板 + 本次观测」的摘要，同一份输入重跑必须相同。
    """

    messages: list[dict[str, str]]
    template_id: str
    template_version: str
    template_hash: str
    content_hash: str

    @property
    def rendered(self) -> str:
        """渲染后的完整正文（消息拼起来，用于算 ``prompt_hash``）。"""
        return "\n".join(m["content"] for m in self.messages)


def build_prompt(
    observation: Mapping[str, Any],
    *,
    registry: PromptRegistry | None = None,
    template_name: str | None = None,
    template_version: str | None = None,
) -> PromptMessages:
    """按**模板名**装配 OpenAI-compatible ``messages``。

    提示词正文来自 :mod:`gateway.prompts` 的模板注册表（随包分发的
    ``templates/*.md``），业务层只给模板名与观测——**改提示词不用改业务代码**，
    也不用改这里。模板里的输出契约由 schema 渲染填入，枚举不会与 Pydantic
    模型漂移（见 :func:`~gateway.prompts.render_output_contract`）。

    输出契约写在提示词里、而不是塞进 ``response_format``：OpenAI-compatible
    协议里 ``response_format`` 的通用形态只有 ``{"type": "json_object"}``，
    把整份 JSON Schema 塞进去只有部分端点认；写法统一成「协议保证对象、契约
    写进提示词」，换端点时不会有一末端点突然不认请求。
    """
    reg = registry or default_registry()
    rendered = reg.render(template_name, version=template_version)
    payload = {field: observation.get(field) for field in OBSERVATION_FIELDS}
    user = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    messages = [
        {"role": "system", "content": rendered.text},
        {"role": "user", "content": user},
    ]
    return PromptMessages(
        messages=messages,
        template_id=rendered.template_id,
        template_version=rendered.version,
        template_hash=rendered.template_hash,
        content_hash=rendered.content_hash,
    )


def build_messages(
    observation: Mapping[str, Any],
    *,
    registry: PromptRegistry | None = None,
    template_name: str | None = None,
    template_version: str | None = None,
) -> list[dict[str, str]]:
    """把一袋的观测装配成 OpenAI-compatible ``messages``（只要正文，不要摘要）。

    需要模板身份与摘要时用 :func:`build_prompt`。
    """
    return build_prompt(
        observation,
        registry=registry,
        template_name=template_name,
        template_version=template_version,
    ).messages


def extract_usage(payload: Any) -> tuple[int | None, int | None]:
    """从响应里取 (prompt_tokens, completion_tokens)；取不到就是 ``None``。

    取不到**不判错**：有些自建端点不回 ``usage``。为了让「端点没回 token 用量」
    而让整次调用降级，是拿一个不影响结论的缺失去换一个确定的结论。
    """
    if not isinstance(payload, Mapping):
        return None, None
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None, None

    def _int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return _int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens"))


def extract_content(payload: Any) -> str:
    """从响应里取 ``choices[0].message.content``。"""
    if not isinstance(payload, Mapping):
        raise ModelOutputRejected("模型响应不是 JSON 对象")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelOutputRejected("模型响应里没有 choices，无法取回复内容")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ModelOutputRejected("choices[0] 不是对象")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ModelOutputRejected("choices[0].message 缺失")
    content = message.get("content")
    if content is None:
        # 部分端点在推理超限时给 null + finish_reason=length
        raise ModelOutputRejected(
            f"choices[0].message.content 为空（finish_reason={first.get('finish_reason')!r}）"
        )
    if not isinstance(content, str):
        raise ModelOutputRejected("choices[0].message.content 不是字符串")
    return content


def classify_response(response: httpx.Response) -> tuple[bool, str]:
    """把 HTTP 响应分类为 (是否成功, 失败类别)。与出站写回同口径。"""
    code = response.status_code
    if 200 <= code < 300:
        return True, "ok"
    if 400 <= code < 500:
        return False, "http_4xx"
    if 500 <= code < 600:
        return False, "http_5xx"
    return False, "http_unexpected"


# ── 一次尝试的结果 ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GatewayAttempt:
    """一次模型调用的结果（纯网络 + 纯解析，不碰数据库）。"""

    ok: bool
    kind: str
    content: str | None = None
    status_code: int | None = None
    error: str | None = None
    elapsed_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "status_code": self.status_code,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class GatewayResult:
    """「首次 + ≤N 次重试」跑完之后的总结果。"""

    ok: bool
    kind: str
    content: str | None = None
    attempts: int = 0
    attempt_kinds: list[str] = field(default_factory=list)
    backoff_delays: list[float] = field(default_factory=list)
    latency_ms: int = 0
    status_code: int | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "attempts": self.attempts,
            "attempt_kinds": list(self.attempt_kinds),
            "backoff_delays": list(self.backoff_delays),
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
            "error": self.error,
        }


# ── 提示结论 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GatewayAdvice:
    """模型给出（或没能给出）的一条**提示**。

    **它不是业务状态**：这个对象里没有任何写状态的方法，也拿不到仓储。要把它
    变成 ``needs_change``，得走 :func:`~identity.state_machine.apply_decision`；
    要变成 ``replaced``，还得再等一次护士确认（task 42 的仲裁层负责这件事的
    接线）。:attr:`deterministic` 始终带着阈值路径的结论——降级时它就是最终结论，
    所以无论模型在不在，这个字段都不是 ``None``。
    """

    request_id: str
    model: str
    prompt_hash: str
    prompt_version: str | None
    decision_source: str
    pending_manual_review: bool
    deterministic: Any
    #: 模型侧字段（不可达或不可用时为 None）
    decision: str | None = None
    reason: str | None = None
    trigger_reason: str | None = None
    confidence: float | None = None
    usable: bool = False
    fallback_reason: str | None = None
    error: str | None = None
    attempts: int = 0
    attempt_kinds: list[str] = field(default_factory=list)
    backoff_delays: list[float] = field(default_factory=list)
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    audit_written: bool = False

    def to_json(self) -> dict[str, Any]:
        """给响应与出站事件用的形状。``deterministic`` 只带摘要，不整块序列化。"""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "decision_source": self.decision_source,
            "usable": self.usable,
            "decision": self.decision,
            "reason": self.reason,
            "trigger_reason": self.trigger_reason,
            "confidence": self.confidence,
            "pending_manual_review": self.pending_manual_review,
            "fallback_reason": self.fallback_reason,
            "error": self.error,
            "attempts": self.attempts,
            "attempt_kinds": list(self.attempt_kinds),
            "backoff_delays": list(self.backoff_delays),
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "audit_written": self.audit_written,
            "deterministic": self.deterministic,
        }


# ── 网关本体 ────────────────────────────────────────────────────────────────


@dataclass
class ModelGateway:
    """唯一的模型调用边界（OpenAI-compatible）。

    ``client`` 供契约测试注入本地替身；不传时用真实 HTTP。**产品代码不因为测试
    注入而改变行为**——注入的只是传输层，请求报头与请求体照常组装。
    """

    base_url: str
    #: 凭据位。``repr=False``：网关对象会出现在异常栈和日志里，dataclass 的默认
    #: repr 会把这一行原样打出来，key 就跟着进了日志——这条纪律不能靠"记得别
    #: 打印对象"，得靠类型本身不给你打印的机会。
    api_key: str = field(repr=False)
    model: str
    timeout: float = 8.0
    connect_timeout: float = 3.0
    max_retries: int = GATEWAY_MAX_RETRIES
    backoff_base: float = BACKOFF_BASE_SECONDS
    temperature: float = 0.0
    max_tokens: int = 1200
    client: httpx.Client | None = field(default=None, repr=False)
    auditor: GatewayAuditor | None = field(default=None, repr=False)
    #: 成本护栏（每日调用上限 + 单请求 token 上限）。留空即不限。
    quota: QuotaGuard | None = field(default=None, repr=False)
    #: 进程内用量与延迟统计。留空则不统计（此时 :meth:`describe` 不带这两项）。
    metrics: GatewayMetrics | None = field(default=None, repr=False)
    #: 提示词模板注册表。留空则用随包分发的默认注册表（``templates/*.md``）。
    #: 换成自定义注册表即可在**不动业务代码**的前提下换一套提示词。
    prompts: PromptRegistry | None = field(default=None, repr=False)
    _owns_client: bool = field(default=False, init=False, repr=False)

    def prompt_registry(self) -> PromptRegistry:
        """本网关用的模板注册表（缺省取随包分发的默认注册表，只读一次）。"""
        if self.prompts is None:
            self.prompts = default_registry()
        return self.prompts

    @classmethod
    def from_settings(cls, settings: Any, **kwargs: Any) -> "ModelGateway":
        """从 :class:`~infusion_monitor.config.Settings` 构造（凭据只走配置面）。"""
        kwargs.setdefault(
            "quota",
            QuotaGuard(
                QuotaPolicy(
                    daily_call_limit=int(getattr(settings, "llm_daily_call_limit", 0) or 0),
                    max_tokens_per_request=int(
                        getattr(settings, "llm_max_tokens_per_request", 1200) or 0
                    ),
                )
            ),
        )
        kwargs.setdefault("metrics", GatewayMetrics())
        return cls(
            base_url=str(settings.llm_base_url).rstrip("/"),
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            timeout=float(settings.llm_timeout),
            max_tokens=int(getattr(settings, "llm_max_tokens_per_request", 1200)),
            **kwargs,
        )

    # ── 协议面 ────────────────────────────────────────────────────────────
    @property
    def endpoint(self) -> str:
        """OpenAI-compatible 补全端点。"""
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def headers(self) -> dict[str, str]:
        """请求头。凭据只在这里出现一次。"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def describe(self) -> dict[str, Any]:
        """可安全打印的网关摘要（key 只给掩码）。"""
        payload = {
            "endpoint": self.endpoint,
            "model": self.model,
            "timeout": self.timeout,
            "connect_timeout": self.connect_timeout,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": "***",
        }
        # 护栏与统计：没挂就不出现这两项，不拿 null 占位糊弄读者。
        if self.quota is not None:
            payload["quota"] = self.quota.as_dict()
        if self.metrics is not None:
            payload["metrics"] = self.metrics.snapshot()
        return payload

    def build_body(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int | None = None
    ) -> dict[str, Any]:
        """出站请求体（OpenAI-compatible ``/chat/completions``）。

        ``temperature=0`` 是刻意的：同一个观测重跑两次应该给同一个结论，否则
        「同一段短片为什么这次说该换、下次说不用换」没法排查。``response_format``
        只要 ``{"type": "json_object"}``——这是协议保证的最大公约数，输出字段契约
        由提示词模板承担（见 :func:`build_prompt` 的说明）。
        """
        return {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "temperature": self.temperature,
            "max_tokens": int(max_tokens or self.max_tokens),
            "stream": False,
            "response_format": {"type": "json_object"},
        }

    def _http(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client(
                timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout),
                follow_redirects=False,
            )
            self._owns_client = True
        return self.client

    def _clean(self, text: str) -> str:
        return scrub(text, self.api_key)

    # ── 一次尝试 ───────────────────────────────────────────────────────────
    def attempt(self, messages: Sequence[Mapping[str, str]],
                *, max_tokens: int | None = None) -> GatewayAttempt:
        """发一次请求并解析回复。**不重试**（退避在 :meth:`complete`）。

        只做网络与协议，不碰数据库，因此可以在工作线程里跑。
        """
        body = self.build_body(messages, max_tokens=max_tokens)
        started = time.perf_counter()
        try:
            response = self._http().post(
                self.endpoint, json=body, headers=self.headers(), timeout=self.timeout
            )
        except httpx.TimeoutException as exc:
            kind = "connect_timeout" if isinstance(exc, httpx.ConnectTimeout) else "read_timeout"
            return GatewayAttempt(
                ok=False,
                kind=kind,
                error=self._clean(f"{kind}: {type(exc).__name__}"),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        except httpx.HTTPError as exc:
            return GatewayAttempt(
                ok=False,
                kind="connect_error",
                error=self._clean(f"connect_error: {type(exc).__name__}: {exc}"),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        ok, kind = classify_response(response)
        elapsed = int((time.perf_counter() - started) * 1000)
        if not ok:
            excerpt = self._clean(response.text[:200])
            return GatewayAttempt(
                ok=False,
                kind=kind,
                status_code=response.status_code,
                error=f"{kind}: HTTP {response.status_code} {excerpt}".strip(),
                elapsed_ms=elapsed,
            )
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            return GatewayAttempt(
                ok=False,
                kind="parse_error",
                status_code=response.status_code,
                error="parse_error: 响应体不是合法 JSON",
                elapsed_ms=elapsed,
            )
        prompt_tokens, completion_tokens = extract_usage(payload)
        try:
            content = extract_content(payload)
        except ModelOutputRejected as exc:
            return GatewayAttempt(
                ok=False,
                kind="parse_error",
                status_code=response.status_code,
                error=f"parse_error: {exc}",
                elapsed_ms=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        return GatewayAttempt(
            ok=True,
            kind="ok",
            content=content,
            status_code=response.status_code,
            elapsed_ms=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_retries: int | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        max_tokens: int | None = None,
        request_id: str | None = None,
    ) -> GatewayResult:
        """跑「首次 + ≤N 次重试」，返回最后一次尝试的总结果。只发网络，不碰库。

        **护栏在这一层生效**：每日额度或单请求 token 上限被越过时，**一次 HTTP 都
        不发**，直接返回 ``kind="quota_exceeded"`` 的失败结果，由调用方按既有降级
        路径收口（结论仍来自确定性阈值，状态仍不推进）。
        """
        effective_max_tokens = int(max_tokens or self.max_tokens)
        if self.quota is not None:
            decision = self.quota.authorize(max_tokens=effective_max_tokens)
            if not decision.allowed:
                blocked = GatewayResult(
                    ok=False,
                    kind=decision.reason or QUOTA_EXCEEDED,
                    attempts=0,
                    attempt_kinds=[],
                    backoff_delays=[],
                    latency_ms=0,
                    error=self._clean(decision.detail or decision.reason or "quota_exceeded"),
                )
                self._note(blocked, request_id=request_id, blocked=True)
                return blocked

        kinds: list[str] = []
        delays: list[float] = []
        total = max(0, self.max_retries if max_retries is None else max_retries) + 1
        attempt = self.attempt(messages, max_tokens=effective_max_tokens)
        kinds.append(attempt.kind)
        used = 1
        while not attempt.ok and attempt.kind in RETRYABLE_KINDS and used < total:
            delay = backoff_seconds(used - 1, base=self.backoff_base)
            delays.append(delay)
            sleeper(delay)
            attempt = self.attempt(messages, max_tokens=effective_max_tokens)
            kinds.append(attempt.kind)
            used += 1
        result = GatewayResult(
            ok=attempt.ok,
            kind=attempt.kind,
            content=attempt.content,
            attempts=used,
            attempt_kinds=kinds,
            backoff_delays=delays,
            latency_ms=attempt.elapsed_ms,
            status_code=attempt.status_code,
            error=self._clean(attempt.error) if attempt.error else None,
            prompt_tokens=attempt.prompt_tokens,
            completion_tokens=attempt.completion_tokens,
        )
        self._note(result, request_id=request_id)
        return result

    def _note(self, result: GatewayResult, *, request_id: str | None,
              blocked: bool = False) -> None:
        """把一次调用的传输层事实记进统计。**不**在这一层记结论来源三态。"""
        if self.metrics is None:
            return
        self.metrics.note_result(
            model=self.model,
            kind=result.kind,
            ok=result.ok,
            requests_sent=0 if blocked else result.attempts,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            request_id=request_id,
            blocked=blocked,
        )

    # ── 提示（降级路径在此收口）──────────────────────────────────────────
    def advise(
        self,
        observation: Mapping[str, Any],
        *,
        deterministic: Any,
        request_id: str,
        prompt_version: str | None = None,
        call_model: bool = True,
        confidence_floor: float = CONFIDENCE_FLOOR,
        max_retries: int | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        template_name: str | None = None,
        template_version: str | None = None,
    ) -> GatewayAdvice:
        """给一袋产出一条提示；模型不可用就退确定性阈值路径。

        :param deterministic: 阈值路径已经算好的结论（由调用方用
            :func:`~identity.state_machine.evaluate_change` 算出并传进来）。
            网关**不重算**它——重算就等于有两套阈值参数在跑，早晚会对不上。
        :param call_model: 传 ``False`` 走 ``threshold_only``：压根不调模型，一次
            HTTP 都不发。院内还没部署端点时用这个模式，服务照常工作。
        :param template_name: 提示词模板名（默认取
            :data:`~gateway.prompts.DEFAULT_TEMPLATE_NAME`）。
        :param template_version: 模板版本；省略时取该名字下版本号最大的那份。
        :param prompt_version: 覆盖写进审计的版本标识。默认记**模板标识**
            （形如 ``infusion_change@v1``）——审计要能回答「这条结论是哪版提示词
            给的」，只记一个裸版本号不够。
        :returns: :class:`GatewayAdvice`。**无论模型在不在、给不给得对，这个方法
            都不推进任何业务状态。**
        """
        prompt = build_prompt(
            observation,
            registry=self.prompt_registry(),
            template_name=template_name,
            template_version=template_version,
        )
        messages = prompt.messages
        digest = prompt_digest(prompt.rendered)
        version = prompt_version or prompt.template_id

        if not call_model:
            return self._settle(
                request_id=request_id,
                prompt_version=version,
                prompt_hash=digest,
                deterministic=deterministic,
                source="threshold_only",
                result=None,
            )

        result = self.complete(
            messages, max_retries=max_retries, sleeper=sleeper, request_id=request_id
        )
        if not result.ok:
            return self._settle(
                request_id=request_id,
                prompt_version=version,
                prompt_hash=digest,
                deterministic=deterministic,
                source="threshold_fallback",
                result=result,
            )

        # 走完了网络才谈解析：解析失败同样归 parse_error，且同样可重试。
        try:
            decision = parse_model_output(result.content or "")
        except ModelOutputRejected as exc:
            failed = GatewayResult(
                ok=False,
                kind="parse_error",
                content=None,
                attempts=result.attempts,
                attempt_kinds=list(result.attempt_kinds) + ["parse_error"],
                backoff_delays=list(result.backoff_delays),
                latency_ms=result.latency_ms,
                status_code=result.status_code,
                error=self._clean(f"parse_error: {exc}"),
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
            return self._settle(
                request_id=request_id,
                prompt_version=version,
                prompt_hash=digest,
                deterministic=deterministic,
                source="threshold_fallback",
                result=failed,
            )

        # 自评置信度过低不作数：模型说「我只有 45% 把握」却仍被采信，等于没设门禁。
        if decision.confidence < confidence_floor or not decision.usable:
            low = GatewayResult(
                ok=False,
                kind="parse_error",
                content=None,
                attempts=result.attempts,
                attempt_kinds=list(result.attempt_kinds),
                backoff_delays=list(result.backoff_delays),
                latency_ms=result.latency_ms,
                status_code=result.status_code,
                error=self._clean(
                    f"below_confidence_floor: confidence={decision.confidence} "
                    f"< {confidence_floor}" if decision.confidence < confidence_floor
                    else f"incomplete_advice: decision={decision.decision} 但缺 trigger_reason"
                ),
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
            return self._settle(
                request_id=request_id,
                prompt_version=version,
                prompt_hash=digest,
                deterministic=deterministic,
                source="threshold_fallback",
                result=low,
            )

        return self._settle(
            request_id=request_id,
            prompt_version=version,
            prompt_hash=digest,
            deterministic=deterministic,
            source="model_decided",
            result=result,
            decision=decision,
        )

    def _settle(
        self,
        *,
        request_id: str,
        prompt_version: str | None,
        prompt_hash: str,
        deterministic: Any,
        source: str,
        result: GatewayResult | None,
        decision: Any | None = None,
    ) -> GatewayAdvice:
        """把一次调用收成一条 :class:`GatewayAdvice`，并落一行推理审计。

        ``decision_source`` 的含义：``model_decided`` = 模型给出了一条通过校验的
        结论；``threshold_fallback`` = 调了但没成（网络、5xx、解析、置信度门禁），
        带着具体 ``fallback_reason``；``threshold_only`` = 压根没调。
        task 42 会把 ``model_decided`` 进一步细化为「模型与阈值一致才采信」。
        """
        degraded = source != "model_decided"
        record = GatewayAuditRecord(
            request_id=request_id,
            model=self.model,
            prompt_hash=prompt_hash,
            prompt_version=prompt_version,
            decision_source=source,
            latency_ms=result.latency_ms if result else 0,
            prompt_tokens=result.prompt_tokens if result else None,
            completion_tokens=result.completion_tokens if result else None,
            fallback_reason=result.kind if (degraded and result) else None,
            # 降级一律要人来兜一眼：模型不可用意味着点名文案不是模型写的。
            pending_manual_review=degraded,
            decision=decision.decision if decision else None,
            confidence=decision.confidence if decision else None,
            session_id=str(deterministic.session_id) if getattr(deterministic, "session_id", None) else None,
            bag_seq=getattr(deterministic, "bag_seq", None),
        )
        audit_written = False
        if self.auditor is not None:
            audit_written = self.auditor.record(record) is not None
        if self.metrics is not None:
            self.metrics.note_decision(
                model=self.model,
                decision_source=source,
                request_id=request_id,
                fallback_reason=record.fallback_reason,
            )

        return GatewayAdvice(
            request_id=request_id,
            model=self.model,
            prompt_hash=prompt_hash,
            prompt_version=prompt_version,
            decision_source=source,
            pending_manual_review=degraded,
            deterministic=deterministic,
            decision=decision.decision if decision else None,
            reason=decision.reason if decision else None,
            trigger_reason=decision.trigger_reason if decision else None,
            confidence=decision.confidence if decision else None,
            usable=decision is not None and not degraded,
            fallback_reason=result.kind if (degraded and result) else None,
            error=result.error if (degraded and result) else None,
            attempts=result.attempts if result else 0,
            attempt_kinds=list(result.attempt_kinds) if result else [],
            backoff_delays=list(result.backoff_delays) if result else [],
            latency_ms=result.latency_ms if result else 0,
            prompt_tokens=result.prompt_tokens if result else None,
            completion_tokens=result.completion_tokens if result else None,
            audit_written=audit_written,
        )

    async def advise_async(self, observation: Mapping[str, Any], **kwargs: Any) -> GatewayAdvice:
        """异步外壳：**只在网络那一段用线程池**，审计落库回到调用线程。

        与 :func:`~adapters.writeback.deliver_event_async` 同一个折中：模型调用是
        阻塞的，不能占着事件循环；而 SQLite 连接带线程亲和性，不能让落库跑到
        别的线程上去。
        """
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(_advise_sync, self, observation, **kwargs)

    def close(self) -> None:
        if self.client is not None and self._owns_client:
            self.client.close()
            self.client = None

    def __enter__(self) -> "ModelGateway":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _advise_sync(gateway: ModelGateway, observation: Mapping[str, Any], **kwargs: Any) -> GatewayAdvice:
    """:meth:`ModelGateway.advise` 的可 pickle 包装（线程池需要一个顶层函数）。"""
    return gateway.advise(observation, **kwargs)