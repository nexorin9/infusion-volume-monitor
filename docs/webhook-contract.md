# 入站 webhook 兜底通道 · 对接契约

> 本文件由 `infusion_monitor.adapters.webhook_telemetry.contract_markdown()` 从
> 代码里的常量生成，`tests/test_mqtt_resilience.py` 会比对是否同步。
> **请改代码，不要改本文。**

## 什么时候才需要这条通道

本服务默认从 MQTT 订阅 `hospital/infusion/<rack>/telemetry`。**只有**下面两种
院区才需要 webhook：病区没有 MQTT broker，或病区侧没有 broker 订阅权限。
两个通道同时开着也可以——同一份遥测从两条路进来会被去重表认出来（按
`device_sn + ts`）。

## 端点

```
POST /api/v1/infusion/telemetry/webhook/{rack_id}
x-infusion-signature: sha256=<hex>
x-infusion-timestamp: <epoch 秒，可选>
Content-Type: application/json
```

未配置 `WEBHOOK_SIGNING_SECRET` 的部署**没有这个端点**（返回 404
`webhook_disabled`），不会用一个默认密钥把它开着。

## 签名算法

```python
signing_input = f"{timestamp}.".encode() + body if timestamp else body
signature = "sha256=" + hmac_sha256(secret, signing_input).hexdigest()
```

* 时间戳**在签名串里**（`ts.body`），因此换掉时间戳头会让签名失效——
  否则重放窗口形同虚设。
* 带了 `x-infusion-timestamp` 时，重放窗口为 ±300 秒
  （`WEBHOOK_SIGNATURE_TOLERANCE_SECONDS` 可调）。对端时钟不准会自己拒掉自己的
  真消息——请先对时，而不是把这个头去掉。
* 不带时间戳时只对 body 验签，**没有重放保护**：抓到的旧消息可以重放。
  这条通道只影响泵速交叉校验，不影响余量与换液判定。

## 载荷字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `bag_seq` | string | 是 | 袋序标识，须与输注医嘱/护理单元系统给出的一致 |
| `pump_rate_ml_per_h` | number | 是 | 泵当前执行的滴速（ml/h）；≤0 一律拒收 |
| `device_sn` | string | 否 | 输液泵序列号；缺时退化为按 (架, 袋, 载荷) 去重 |
| `ts` | string | 否 | ISO8601 时间戳；缺时新鲜度报 undated 而不是假装很新 |
| `rack_id` | string | 否 | 输液架标识；带了就必须与 URL 路径段一致，否则整条拒收 |

`rack_id` 取自 **URL 路径段**；载荷里带了就必须与路径一致，否则整条拒收
（`rack_mismatch`）——一份跨架的遥测写进另一架的缓存，事后查不出来源。

## 响应与拒收原因

| 情况 | 状态码 | 响应里的 `reason` |
|------|--------|------------------|
| 新读数 | 202 | `accepted=true` |
| 重复（同一 `device_sn+ts`，或同架同袋的更旧时间戳） | 202 | `duplicate=true` |
| 未配置 WEBHOOK_SIGNING_SECRET，本部署没有这条通道 | 404 | `webhook_disabled` |
| 请求体超过 64KB 上限 | 413 | `body_too_large` |
| 缺少签名头 | 401 | `signature_missing` |
| 签名头格式不对（应为 sha256=<hex>） | 401 | `signature_malformed` |
| 签名不匹配：密钥不一致或请求体在签名后被改动 | 401 | `signature_mismatch` |
| 时间戳头不是 epoch 秒 | 401 | `timestamp_malformed` |
| 请求时间与服务端相差超过重放窗口（对端时钟可能不准） | 401 | `timestamp_out_of_window` |
| 请求体不是合法 JSON | 422 | `payload_not_json` |
| 请求体不是 JSON 对象 | 422 | `payload_not_object` |
| 载荷里的 rack_id 与 URL 路径段不一致 | 422 | `rack_mismatch` |
| 载荷字段缺失 / 滴速非正 | 422 | `payload_rejected`（细因见审计链） |

## 这条通道坏了会怎样

**什么都不发生——视觉主路径照常。** 余量与换液判定只用短片；滴速退回输注
医嘱文本（`eta_source=order_text`），每个袋位的 `telemetry_state` 如实报
`absent`，运行摘要 `channels.webhook.state` 变成 `idle`。没有遥测不等于
没有结论。

## 未实连范围

本轮只对本地 HTTP 替身与测试客户端跑过签名/去重/拒收路径；**真实院内泵网关
的 webhook 端点未实连**。替身在测试数据层，不是最终对接方式。
