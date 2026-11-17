# 提示词模板注册表

> 本文件由 `infusion_monitor.gateway.prompts.template_catalog_markdown()` 生成，
> **请勿手改**。模板正文在 `src/infusion_monitor/gateway/templates/`，
> 改提示词改那里，本文件随之重新生成。

## 已注册模板

| 模板 | 版本 | 适用场景 | 正文摘要 |
|------|------|----------|----------|
| `infusion_change` | `v1` | 单袋换液判定——护士上传一段短片后，对输液架上每一袋判断现在是否需要换液 | `f03d4006c276b58c` |

## 占位符

| 模板 | 占位符 | 由谁提供 |
|------|--------|----------|
| `infusion_change@v1` | `{{output_contract}}` | 输出契约（由 `gateway.schemas.OUTPUT_SCHEMA` 渲染） |

## 渲染后的输出契约

```json
{"decision": "no_change" | "needs_change", "reason": "给护士看的一句话中文理由", "confidence": 0.0-1.0, "trigger_reason": null | "eta_within_window" | "volume_below_margin" | "near_empty_observed"}
```

## 版本与审计

* 每次调用落一行 `inference_audit`：`prompt_version` = 模板标识（形如 `infusion_change@v1`），`prompt_hash` = 渲染后正文的摘要；
* `prompt_hash` 相同 ⇒ 模板版本与本次输入都相同 ⇒ 结论可复现；
* `prompt_version` 变了而 `prompt_hash` 变了 ⇒ 提示词改过，
  先比对模板 diff 再讨论结论差异；
* 模板版本变了但 `prompt_hash` 没变 ⇒ 改动没有进入渲染路径（可疑）。
