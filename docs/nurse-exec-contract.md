# 出站写回契约 · 护理执行平台

> 本文件由 `infusion_monitor.adapters.nurse_exec.contract_markdown()` 从
> `OUTBOUND_EVENT_SCHEMA` 生成，**请勿手改**——契约测试会比对本文件与模型字段。

## 端点与鉴权

| 项 | 值 |
|------|------|
| 方法与路径 | `POST {NURSE_EXEC_BASE_URL}/api/v1/infusion/events` |
| 鉴权 | `Authorization: Bearer $NURSE_EXEC_API_TOKEN` |
| 内容类型 | `application/json`（UTF-8） |
| 幂等头 | `Idempotency-Key: <client_event_id>`，与请求体同字段同值 |
| 重定向 | 不跟随（3xx 视为失败，避免把载荷发到别处） |
| 系统代理 | 不继承（`trust_env=False`）：端点由 `NURSE_EXEC_BASE_URL` 明确给出，工作站的系统代理不得接管病区换液事件，也不得把传输层失败统一翻成 502 |

## 字段规范

| 字段 | 类型 | 必填 | 可空 | 说明 |
|------|------|------|------|------|
| `event_id` | str | 是 | 否 | 本服务生成的事件 ID，写回后供对端回查与对账 |
| `client_event_id` | str | 是 | 否 | 幂等键：同一袋只换一次，重放必得同一个值 |
| `event_type` | enum | 是 | 否 | 事件类型，当前固定 infusion.bag.changed |
| `session_id` | str | 否 | 是 | 本轮巡视的袋级会话 ID，便于回查曲线 |
| `rack_id` | str | 是 | 否 | 输液架编号 |
| `bed_no` | str | 是 | 否 | 床号 |
| `infusion_order_no` | str | 是 | 否 | 输注单号（对端给的权威标识） |
| `bag_seq` | str | 是 | 否 | 被换掉的袋序 |
| `fill_ratio` | float | 否 | 是 | 触发确认时的面积比（0–1）；测不出则不出现 |
| `estimated_volume_ml` | float | 否 | 是 | 标定后的剩余毫升；未标定袋型则不出现 |
| `eta_minutes` | int | 否 | 是 | 预计耗尽分钟数；滴速不可得则不出现 |
| `trigger_reason` | str | 否 | 是 | 点名/确认原因码 |
| `decision_source` | str | 否 | 是 | 结论来源 |
| `operator_id` | str | 是 | 否 | 确认换液的护士工号；没有确认人就没有这条事件 |
| `occurred_at` | str | 是 | 否 | 事件发生时刻（ISO8601） |

## 缺省语义

* 可空且拿不到值的字段（`estimated_volume_ml` / `eta_minutes` / `fill_ratio`）在请求体里**整体不出现**，不是 `null`。对端若把缺省读成 0，会把「没测出来」读成「袋是空的」。
* `event_type` 当前只取值 `infusion.bag.changed`。
* 载荷只含床号、袋序、数量与事件，**不含患者姓名/住院号等可识别明细**。

## 字段枚举

| 字段 | 允许取值 |
|------|----------|
| `event_type` | `infusion.bag.changed` |
| `trigger_reason` | `eta_within_window` / `volume_below_margin` / `near_empty_observed` / `nurse_confirmed` |
| `decision_source` | `model_decided` / `threshold_fallback` / `threshold_only` / `nurse_confirmed` |

## 与对端接口文档的对应表

接挂时逐行对这张表：左边是本服务发出的字段名，右边是对端在自己接口文档里的落点与约定。名字相似不等于语义相同——`occurred_at` 是业务发生时刻而不是对端接收时刻，`clientEventId` 与 `Idempotency-Key` 必须是同一个值。

| 本服务字段 | 对端契约位置 | 对端侧约定 |
|------|------|----------|
| `event_id` | 请求体 eventId | 本服务生成的回查号；对端原样落库，不参与去重 |
| `client_event_id` | 请求体 clientEventId + 请求头 Idempotency-Key | **幂等键**：两处必须同值。对端凭它去重，因此重试安全（同一袋只换一次） |
| `event_type` | 请求体 eventType | 对端按码分派到各自的处理流程；未知码会被拒 |
| `session_id` | 请求体 sessionId | 本服务的袋级会话 ID；对端只做回查用，不做业务键 |
| `rack_id` | 请求体 rackId | 输液架编号，与床号一起定位巡视频次 |
| `bed_no` | 请求体 bedNo | 床号；不传患者姓名与住院号 |
| `infusion_order_no` | 请求体 infusionOrderNo | 输注单号；对端校验它是否属于该床位的在输医嘱 |
| `bag_seq` | 请求体 bagSeq | 被换掉的袋序；新袋由对端按自身规则编号 |
| `fill_ratio` | 请求体 fillRatio（可选） | 缺省即「本轮没测出来」，对端**不得**按 0 处理 |
| `estimated_volume_ml` | 请求体 estimatedVolumeMl（可选） | 未标定袋型则整个字段不出现，不是 null |
| `eta_minutes` | 请求体 etaMinutes（可选） | 滴速不可得则不出现；对端按缺省走巡视节奏 |
| `trigger_reason` | 请求体 triggerReason（可选） | 原因码；对端写进执行记录，飞检时对得上 |
| `decision_source` | 请求体 decisionSource（可选） | 结论来源；对端不据此改流程，只做留痕 |
| `operator_id` | 请求体 operatorId | 护士工号；对端据此把执行记录落到具体的人头上 |
| `occurred_at` | 请求体 occurredAt | 业务发生时刻（护士点确认那一刻），不是对端的接收时刻 |

## 失败分类

| 失败类别 | 可重试 | 终态 | 对端可能已收下 | 现场长什么样 |
|------|------|------|------|------|
| `ok` | 否 | `sent` | 不会 | 对端 2xx 收下了这条事件 |
| `http_4xx` | 否 | `failed` | 不会 | 对端业务拒绝：同一份载荷、同一把 token 重发仍会被拒 |
| `http_5xx` | 是 | `pending_writeback` | 不会 | 对端内部错误，与本次载荷无关 |
| `http_unexpected` | 否 | `failed` | 不会 | 看不懂的状态码（1xx 落成最终响应 / 3xx，本服务不跟随重定向） |
| `connect_timeout` | 是 | `pending_writeback` | 不会 | TCP 建连就没连上：病区到对端的网络不通或防火墙拦了 |
| `read_timeout` | 是 | `pending_writeback` | **会** | 对端收下了请求但没在时限内回应答：它慢，或者它卡在自己的库里 |
| `write_timeout` | 是 | `pending_writeback` | **会** | 载荷发到一半断了：请求体可能只写到一部分，对端多半已按格式错误拒收 |
| `pool_timeout` | 是 | `pending_writeback` | 不会 | 连接池等不到可用连接：本服务自己占着连接，尚未发到对端 |
| `half_open` | 是 | `pending_writeback` | **会** | 半开连接：对端把连接掐了却没给应答（复用的连接早已失效、对端进程重启、中间的负载均衡重置了空闲连接） |
| `connect_error` | 是 | `pending_writeback` | 不会 | 连接被拒 / DNS 解析失败：地址写错、对端没起、或被拒在门外 |
| `protocol_error` | 否 | `failed` | 不会 | 协议层就不成立（最常见：base_url 少了 http:// 前缀） |

> `对端可能已收下 = 会` 的三类（`read_timeout` / `write_timeout` / `half_open`）正是**幂等键不可省**的原因：它们既值得重试，又不能保证重发不会在对端变成两条。

## 错误摘要格式

退避耗尽后写进 `last_error`、并在待写回列表与飞检材料里展示的一行摘要：

```
kind=<失败类别> status=<HTTP 状态码，可无> attempts=<尝试次数> counts=<类别=次数,...> reason=<对端给的原因> last=<最后一次错误文本>
```

* `counts` **按类别分别计数**，因此「连接超时 3 次」与「读超时 3 次」在摘要里长得不一样——前者查网络配置，后者查对端处理速度。
* 整条摘要经 `scrub()` 脱敏，凭据值不会进这条文本。

## 响应约定

| 状态码 | 本服务的处理 |
|------|------|
| 2xx | 写回置 `sent` |
| 4xx | 业务拒绝，**不重试**，事件置 `failed`，并从响应体抽出对端原因记进摘要 |
| 5xx | 可重试；退避重试耗尽后置 `pending_writeback` 并告警 |
| 连接/读取超时 | 可重试；同上 |
| 半开连接 | 可重试；请求很可能已到对端，靠 `Idempotency-Key` 保证不重复 |
