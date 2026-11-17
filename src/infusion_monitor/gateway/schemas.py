"""模型输出的结构化契约：模型能说的话只有两句，且都必须落在枚举里。

**为什么校验要这么严**：模型的结论只进 `inference_audit` 与点名载荷，它**不推进
业务状态**。但如果放任它说自由文本，`needs_change` 后面可能跟着一句不可执行的
原因码，下游按码分支时就有了没走过的路。与其让下游兜底，不如在网关边界上把它
挡掉：校验不过就是「模型输出不可解析」（``parse_error``），退确定性阈值路径，
给出的仍然是正确的那个结论。

三条纪律：

1. **模型说不出 ``replaced``**。:data:`DECISIONS` 里只有 ``no_change`` 与
   ``needs_change``；换液确认是护士在终端上点出来的动作，不是一段文本能表达
   的状态。模型返回 ``replaced`` 会被枚举校验拒掉（见
   :meth:`ModelDecision.from_model` 的用例），而不是被悄悄忽略——否则
   "模型说要换" 这件事连日志里都不会留下。
2. **枚举来自业务层而不是另抄一份**。:data:`DECISIONS` 与 ``trigger_reason``
   的取值分别对齐 :mod:`identity.state_machine` 的状态集合与触发原因码，两边
   不会再各长出一套词表。
3. **未知字段按错误处理**（``extra="forbid"``）。模型多给一个字段，多半说明它
   在按另一套 schema 回答（比如把某个老版本的提示词当成当前版本）；这时它对
   当前契约的理解已经不可信了，不该只把认识的字段捡出来用。
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..identity.state_machine import TRIGGER_REASONS

__all__ = [
    "CONFIDENCE_FLOOR",
    "DECISIONS",
    "OUTPUT_SCHEMA",
    "ModelDecision",
    "ModelOutputRejected",
    "OUTPUT_TRIGGER_REASONS",
    "contract_markdown",
    "extract_json_object",
    "output_contract",
]


class ModelOutputRejected(ValueError):
    """模型输出不可解析或不符合契约。

    归入 ``parse_error`` 一类：不论是压根没有 JSON、JSON 里缺字段、还是
    ``decision`` 取了枚举外的值，对我们的处置都一样——**不采信**，退阈值路径。
    分成不同类别只会让降级逻辑多出几条分支，而它们的后果一模一样。
    """

    def __init__(self, message: str, *, raw_excerpt: str | None = None) -> None:
        self.raw_excerpt = (raw_excerpt or "")[:200]
        super().__init__(message)


#: 模型能给出的结论。**刻意不含 ``replaced``**——换液确认只属于护士。
DECISIONS: tuple[str, ...] = ("no_change", "needs_change")

#: 模型可以引用的触发原因码，直接复用业务层的词表（不在此处另立一套）。
OUTPUT_TRIGGER_REASONS: tuple[str, ...] = tuple(TRIGGER_REASONS)

#: 低于此置信度的模型结论不采信（归回落阈值路径），与 task 42 的仲裁门禁同源。
CONFIDENCE_FLOOR = 0.60


class ModelDecision(BaseModel):
    """模型的结构化结论。**只是一个提示，不是业务事实。**"""

    model_config = ConfigDict(extra="forbid")

    decision: str = Field(description="no_change / needs_change；没有 replaced 这一项")
    reason: str = Field(description="给护士看的一句话理由（中文，不含患者可识别明细）")
    confidence: float = Field(description="模型自评置信度，落在 0–1")
    trigger_reason: str | None = Field(
        default=None,
        description="needs_change 时的触发原因码，取值同业务层词表；no_change 时留空",
    )

    @field_validator("decision")
    @classmethod
    def _known_decision(cls, value: str) -> str:
        if value not in DECISIONS:
            raise ValueError(
                f"decision 只能是 {' / '.join(DECISIONS)}；收到 {value!r}"
                "（换液确认只能由护士触发，模型无权给出）"
            )
        return value

    @field_validator("trigger_reason")
    @classmethod
    def _known_trigger_reason(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if value not in OUTPUT_TRIGGER_REASONS:
            raise ValueError(
                f"trigger_reason 取值须在 {' / '.join(OUTPUT_TRIGGER_REASONS)} 内；收到 {value!r}"
            )
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence 须落在 0–1；收到 {value!r}")
        return value

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason 不能是空串：护士端要显示这句话")
        return value.strip()

    @property
    def usable(self) -> bool:
        """是否够格采信：``needs_change`` 必须带上可执行的触发原因码。"""
        return self.decision != "needs_change" or bool(self.trigger_reason)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型回复里取出 JSON 对象。

    OpenAI-compatible 端点即便设了 ``response_format=json_object``，也常见两种
    「不是纯 JSON」的返回：包在 ```json 围栏里，或者前后带一句客套话。对这两种
    做一次温和的收拾，比把它们当解析失败降级掉更划算——内容本身是好的，扔了可惜。
    实在取不出 JSON 才抛 :class:`ModelOutputRejected`。
    """
    if not isinstance(text, str) or not text.strip():
        raise ModelOutputRejected("模型回复为空", raw_excerpt=text if isinstance(text, str) else "")
    candidate = _FENCE_RE.sub("", text).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        # 退一步：截取最外层 {...}（模型常在 JSON 前后加解释文字）
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ModelOutputRejected("模型回复里找不到 JSON 对象", raw_excerpt=candidate)
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelOutputRejected(f"模型回复不是合法 JSON：{exc.msg}", raw_excerpt=candidate) from exc
    if not isinstance(parsed, dict):
        raise ModelOutputRejected("模型回复的 JSON 顶层不是对象", raw_excerpt=candidate)
    return parsed


def parse_model_output(raw: Mapping[str, Any] | str) -> ModelDecision:
    """模型回复 → :class:`ModelDecision`；任何不合契约的形态都抛 :class:`ModelOutputRejected`。

    **统一成一类错误**是有意的：解析失败、缺字段、枚举越界、多了字段，对调用方的
    处置完全相同（不采信、退阈值路径）。把它们拆成四类错误码只会在降级分支里
    多出几条走法相同、后果相同的路。
    """
    payload = extract_json_object(raw) if isinstance(raw, str) else dict(raw)
    try:
        return ModelDecision(**payload)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        field_name = ".".join(str(p) for p in first.get("loc", ())) or "(顶层)"
        raise ModelOutputRejected(
            f"模型输出不符合契约：字段 {field_name} — {first.get('msg', exc)}",
            raw_excerpt=json.dumps(payload, ensure_ascii=False),
        ) from exc


#: 输出契约的机读描述（发给端点与写进文档的是同一份，避免手抄漂移）。
OUTPUT_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "reason", "confidence"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": list(DECISIONS),
            "description": "no_change = 不用换；needs_change = 建议换袋。无 replaced。",
        },
        "reason": {
            "type": "string",
            "minLength": 1,
            "description": "给护士看的一句话中文理由，不含患者姓名/住院号等明细",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "模型自评置信度；低于 0.60 不采信，退阈值路径",
        },
        "trigger_reason": {
            "type": ["string", "null"],
            "enum": list(OUTPUT_TRIGGER_REASONS) + [None],
            "description": "needs_change 时必填，取值同业务层触发原因码；no_change 时留空",
        },
    },
}


def output_contract() -> dict[str, Any]:
    """``response_format`` 里实际发出的那份（副本，调用方随意改不影响契约真源）。"""
    return json.loads(json.dumps(OUTPUT_SCHEMA))


def contract_markdown() -> str:
    """由 :data:`OUTPUT_SCHEMA` 生成输出契约文档（契约测试比对文档与模型字段）。"""
    lines = [
        "# 模型输出契约 · 输液余量判定",
        "",
        "> 本文件由 `infusion_monitor.gateway.schemas.contract_markdown()` 从",
        "> `OUTPUT_SCHEMA` 与 `ModelDecision` 生成，**请勿手改**。",
        "",
        "## 调用形态",
        "",
        "| 项 | 值 |",
        "|------|------|",
        "| 方法与路径 | `POST {LLM_BASE_URL}/chat/completions` |",
        "| 鉴权 | `Authorization: Bearer $LLM_API_KEY` |",
        "| 响应约束 | `response_format={\"type\": \"json_object\"}` |",
        "| 重试 | 首次 + ≤2 次指数退避；耗尽即退确定性阈值路径 |",
        "",
        "## 输出字段",
        "",
        "| 字段 | 类型 | 必填 | 取值 | 说明 |",
        "|------|------|------|------|------|",
        f"| `decision` | enum | 是 | {' / '.join(DECISIONS)} | 结论。**无 replaced**：换液确认只属于护士 |",
        f"| `reason` | str | 是 | 任意非空 | 给护士看的一句话理由 |",
        f"| `confidence` | number | 是 | 0–1 | 自评置信度，低于 {CONFIDENCE_FLOOR} 不采信 |",
        f"| `trigger_reason` | str/null | 否 | {' / '.join(OUTPUT_TRIGGER_REASONS)} | needs_change 时必填 |",
        "",
        "## 降级语义",
        "",
        "* 校验不过（缺字段 / 枚举越界 / 多出字段 / 不是 JSON）一律归 `parse_error`，**不采信**；",
        "* 降级后 `decision_source=threshold_fallback` 且 `pending_manual_review=true`；",
        "* 降级**不推进业务状态**：`needs_change → replaced` 仍只能由护士确认触发；",
        "* 推理元数据落 `inference_audit`（request_id / model / prompt_hash / latency_ms /",
        "  token 用量 / decision_source / fallback_reason），供信息科与护理部复核。",
        "",
    ]
    return "\n".join(lines)