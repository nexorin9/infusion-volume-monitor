# 模型输出契约 · 输液余量判定

> 本文件由 `infusion_monitor.gateway.schemas.contract_markdown()` 从
> `OUTPUT_SCHEMA` 与 `ModelDecision` 生成，**请勿手改**。

## 调用形态

| 项 | 值 |
|------|------|
| 方法与路径 | `POST {LLM_BASE_URL}/chat/completions` |
| 鉴权 | `Authorization: Bearer $LLM_API_KEY` |
| 响应约束 | `response_format={"type": "json_object"}` |
| 重试 | 首次 + ≤2 次指数退避；耗尽即退确定性阈值路径 |

## 输出字段

| 字段 | 类型 | 必填 | 取值 | 说明 |
|------|------|------|------|------|
| `decision` | enum | 是 | no_change / needs_change | 结论。**无 replaced**：换液确认只属于护士 |
| `reason` | str | 是 | 任意非空 | 给护士看的一句话理由 |
| `confidence` | number | 是 | 0–1 | 自评置信度，低于 0.6 不采信 |
| `trigger_reason` | str/null | 否 | eta_within_window / volume_below_margin / near_empty_observed | needs_change 时必填 |

## 降级语义

* 校验不过（缺字段 / 枚举越界 / 多出字段 / 不是 JSON）一律归 `parse_error`，**不采信**；
* 降级后 `decision_source=threshold_fallback` 且 `pending_manual_review=true`；
* 降级**不推进业务状态**：`needs_change → replaced` 仍只能由护士确认触发；
* 推理元数据落 `inference_audit`（request_id / model / prompt_hash / latency_ms /
  token 用量 / decision_source / fallback_reason），供信息科与护理部复核。
