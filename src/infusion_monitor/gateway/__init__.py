"""M6 模型网关：业务层与模型之间**唯一**的边界。

业务层只 import 本包，不 import 任何厂商 SDK。换端点（院内自建推理服务 / 本地
部署的权重 / 另一个 OpenAI-compatible 服务）只改 ``LLM_BASE_URL`` 等环境变量。

* :mod:`gateway.schemas` —— 模型输出的结构化契约与强校验；
* :mod:`gateway.prompts` —— 提示词模板注册表（模板从文件加载、版本化、渲染留摘要）；
* :mod:`gateway.audit` —— 每次调用的推理审计（来源、延迟、token、降级原因）；
* :mod:`gateway.metrics` —— 进程内用量/延迟统计与成本护栏（每日调用上限、单请求 token 上限）；
* :mod:`gateway.model_gateway` —— 客户端、四类失败分类、退避重试与降级收口；
* :mod:`gateway.fusion` —— 仲裁层：模型不是裁判，只在结论一致且置信达门禁时采信其措辞。
"""

from __future__ import annotations

from .audit import (
    DECISION_SOURCES,
    FALLBACK_REASONS,
    GatewayAuditRecord,
    GatewayAuditor,
    prompt_digest,
)
from .fusion import (
    ARBITRATION_CODES,
    FUSED_ACCEPTED,
    FUSION_AUDIT_ACTIONS,
    Arbitration,
    FusionParams,
    FusedConclusion,
    apply_to_state,
    arbitrate,
    fuse,
    fuse_round,
    outcome_counts,
    threshold_decision_of,
)
from .metrics import (
    DEFAULT_LATENCY_WINDOW,
    DEFAULT_REQUEST_BUCKETS,
    QUOTA_EXCEEDED,
    GatewayMetrics,
    LatencyWindow,
    ModelStats,
    QuotaDecision,
    QuotaGuard,
    QuotaPolicy,
    RequestStats,
    percentile,
)
from .model_gateway import (
    BACKOFF_BASE_SECONDS,
    GATEWAY_MAX_RETRIES,
    OBSERVATION_FIELDS,
    RETRYABLE_KINDS,
    GatewayAdvice,
    GatewayAttempt,
    GatewayResult,
    ModelGateway,
    PromptMessages,
    build_messages,
    build_prompt,
    classify_response,
)
from .prompts import (
    DEFAULT_TEMPLATE_NAME,
    TEMPLATE_DIR,
    PromptRegistry,
    PromptRegistryError,
    PromptTemplate,
    PromptTemplateMissing,
    RenderedPrompt,
    default_registry,
    render_output_contract,
    template_catalog_markdown,
)
from .schemas import (
    CONFIDENCE_FLOOR,
    DECISIONS,
    OUTPUT_SCHEMA,
    ModelDecision,
    ModelOutputRejected,
    extract_json_object,
    parse_model_output,
)

__all__ = [
    "ARBITRATION_CODES",
    "BACKOFF_BASE_SECONDS",
    "CONFIDENCE_FLOOR",
    "DECISIONS",
    "DECISION_SOURCES",
    "DEFAULT_TEMPLATE_NAME",
    "FALLBACK_REASONS",
    "FUSED_ACCEPTED",
    "FUSION_AUDIT_ACTIONS",
    "DEFAULT_LATENCY_WINDOW",
    "DEFAULT_REQUEST_BUCKETS",
    "GATEWAY_MAX_RETRIES",
    "OBSERVATION_FIELDS",
    "OUTPUT_SCHEMA",
    "QUOTA_EXCEEDED",
    "RETRYABLE_KINDS",
    "TEMPLATE_DIR",
    "Arbitration",
    "FusionParams",
    "FusedConclusion",
    "GatewayAdvice",
    "GatewayAttempt",
    "GatewayAuditRecord",
    "GatewayAuditor",
    "GatewayMetrics",
    "GatewayResult",
    "LatencyWindow",
    "ModelStats",
    "QuotaDecision",
    "QuotaGuard",
    "QuotaPolicy",
    "RequestStats",
    "ModelDecision",
    "ModelGateway",
    "ModelOutputRejected",
    "PromptMessages",
    "PromptRegistry",
    "PromptRegistryError",
    "PromptTemplate",
    "PromptTemplateMissing",
    "RenderedPrompt",
    "apply_to_state",
    "arbitrate",
    "build_messages",
    "build_prompt",
    "classify_response",
    "default_registry",
    "extract_json_object",
    "fuse",
    "fuse_round",
    "outcome_counts",
    "parse_model_output",
    "percentile",
    "prompt_digest",
    "render_output_contract",
    "template_catalog_markdown",
    "threshold_decision_of",
]