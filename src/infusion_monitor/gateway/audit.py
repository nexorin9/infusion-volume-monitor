"""推理审计：每次模型调用都留一行可核对的元数据。

**为什么模型不在线上也要留行**：护士看到「系统说这袋该换了」，护士长或信息科
一定会问「这句话是模型说的还是算出来的」。``inference_audit`` 就是这个问题的
答案——``decision_source`` 一列把来源写死：``model_decided`` / ``threshold_fallback`` /
``threshold_only``。降级不是故障，恰恰是最需要留痕的那一种：它说明端点当时不可用、
我们没有因此停摆，而且每一次降级都能按 ``fallback_reason`` 分开统计。

两条纪律：

1. **写审计失败不阻断业务**。结论已经算出来了，把它连带业务一起回滚重来一遍，
   对护士没有任何好处。所以 :meth:`GatewayAuditor.record` 吞掉写库异常并返回
   ``None``，由 :attr:`GatewayAdvice.audit_written` 如实暴露「这行没写上」，
   而不是假装写上了。业务要拿这条做判断时，看得见它缺了。
2. **只落调用元数据，不落提示词全文**。``prompt_hash`` 是渲染后提示词的摘要：
   够用来证明「同一次提示词模板 + 同一份输入」重跑得到同一结论，又不必把袋位
   曲线之类的现场数据复制一份到第二张表。真正要复现时按模板 + 当时输入重渲即可。
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "DECISION_SOURCES",
    "FALLBACK_REASONS",
    "GatewayAuditRecord",
    "GatewayAuditor",
    "prompt_digest",
]

#: 结论来源三态。``threshold_only`` = 这一轮压根没走模型（确定性路径自己下了结论）。
DECISION_SOURCES: tuple[str, ...] = ("model_decided", "threshold_fallback", "threshold_only")

#: 降级原因码。与 ``store.schema`` 里 ``inference_audit.fallback_reason`` 的注释
#: 同口径；``connect_timeout`` 是那条注释里 ``timeout`` 的精确化——把「连不上」和
#: 「连上了但没等到回复」分开记，是因为两者的排查方向不同。
FALLBACK_REASONS: tuple[str, ...] = (
    "connect_timeout",
    "read_timeout",
    "connect_error",
    "http_5xx",
    "http_4xx",
    "http_unexpected",
    "parse_error",
    "quota_exceeded",  # task 43 的调用上限
)


def prompt_digest(text: str) -> str:
    """渲染后提示词的稳定摘要（sha256 前 16 位）。

    键序与空白都会影响摘要——这是有意的：提示词改一个空格就可能改结论，
    摘要变了才查得出「换了提示词之后结果不一样」。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class GatewayAuditRecord:
    """一行 ``inference_audit``。字段与表列一一对应。"""

    request_id: str
    model: str
    prompt_hash: str
    decision_source: str
    prompt_version: str | None = None
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    fallback_reason: str | None = None
    pending_manual_review: bool = False
    decision: str | None = None
    confidence: float | None = None
    session_id: str | None = None
    bag_seq: str | None = None
    created_at: str | None = None
    #: 不落库、只在返回里带给调用方的补充信息（尝试次数、退避轨迹）。
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """转成 ``InferenceAuditRepository.append`` 吃的字典。"""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "decision_source": self.decision_source,
            "prompt_version": self.prompt_version,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "fallback_reason": self.fallback_reason,
            "pending_manual_review": self.pending_manual_review,
            "decision": self.decision,
            "confidence": self.confidence,
            "session_id": self.session_id,
            "bag_seq": self.bag_seq,
            "created_at": self.created_at,
        }


@dataclass
class GatewayAuditor:
    """把 :class:`GatewayAuditRecord` 写进 ``inference_audit``（只追加）。"""

    #: :class:`~infusion_monitor.store.repositories.InferenceAuditRepository`
    repository: Any

    def record(self, record: GatewayAuditRecord) -> int | None:
        """写一行审计，返回 ``audit_id``；写库失败返回 ``None``（不抛）。"""
        try:
            return int(self.repository.append(record.to_row()) or 0) or None
        except sqlite3.Error:
            # 见模块文档纪律 1：结论已经算出来了，让审计写不进去不该把业务一起带倒。
            return None

    def list_by_request(self, request_id: str) -> list[dict]:
        return self.repository.list_by_request(request_id)