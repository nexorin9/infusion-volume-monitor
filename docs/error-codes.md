# 错误码与运行期故障对照

> 本文件由 `src/infusion_monitor/errors.py` 的 `ERROR_CODES` / `_ERROR_CODE_DOCS` /
> `_OPERATIONAL_FAULTS` 生成，`tests/test_summary.py` 会比对是否同步。**请改代码，不要改本文。**

两件事要分开看：

* **API 错误码**：调用方拿到了 HTTP 错误响应。响应体恒为
  `{"detail": {"error_code": …, "message": …, "issues": [{"code","field","reason","suggestion"}]}}`。
* **运行期故障**：请求**成功返回**，但服务内部某条链路已经不可用。它们只出现在
  运行摘要 `GET /api/v1/summary` 与启动日志里。把两者混成一张表，值班的就会对着
  「HTTP 200 但摘要说没连上」这种组合查半天。

## API 错误码

| 错误码 | HTTP | 含义 | 调用方怎么处理 |
|--------|------|------|----------------|
| `INGEST_INVALID` | 422 | 帧序非法：帧数不在 12–1200、短边小于 320px、时间戳非单调递增、槽位超过 6 袋，或必填字段缺失。 | 看 detail.issues[].suggestion 逐条照做后重传；同一次上传的问题会一次性全部返回，不必来回试。 |
| `FRAMES_ZIP_REJECTED` | 422 | 帧 zip 结构不安全：含路径穿越成员、符号链接、异常压缩比（zip 炸弹）或成员命名不合约定。 | 重新打包后重传。服务端不会把 zip 解到临时目录以外，收到该码说明这一包已被整体拒收。 |
| `NOT_FOUND` | 404 | 会话或袋位不存在，或其帧目录已被保留期清理。 | 确认 session_id 来自本次上传响应；已被清理的会话需要重新巡视上传一段短片。 |
| `STATE_CONFLICT` | 409 | 业务状态迁移非法：例如 monitoring 直跳 replaced、已换袋的会话再次确认。 | 新一轮巡视请让对端给新的 bag_seq；同一袋的重复确认请复用同一个 client_event_id。 |
| `WRITE_NOT_ALLOWED` | 403 | 越权写入：确认换液时指定的袋序或输液架不属于该会话。 | 按 detail.issues[].reason 里点出的两个袋序核对终端缓存的 session_id，勿凭记忆点确认。 |
| `CALIBRATION_WINDOW_CONFLICT` | 409 | 标定生效区间与既有版本相交：同一袋型被登记了两个同时生效的版本。 | 按 detail.issues[].suggestion 把本次登记的生效时刻挪到旧版之后，或给旧版补一个 effective_to 明确停用时刻；需要同时生效请先核对是否真的换了袋型。 |
| `UPLOAD_REJECTED` | 422 | 分片被拒：成员命名不合约定、单片帧数超限，或分片里的帧号是从 0 重新编号的（那样无法判断该接在哪一段后面）。 | 只拒这一片：已经收下的分片不受影响。按 detail.issues[].suggestion 改成整段全局序号命名后重传该片即可。 |
| `UPLOAD_NOT_OPEN` | 409 | 这个 upload_id 已结案（completed 或 abandoned），不再接收分片。 | 已提交过的无需重传；被保留期清理掉的请重新 POST /api/v1/infusion/uploads 开一次。 |
| `UPLOAD_ALREADY_COMPLETED` | 409 | 同一 upload_id 已经提交过一次，重复提交不会重算。 | 拿提交响应里的 bags[].session_id 去查余量；确实要重传整段请直接 POST /api/v1/infusion/tracks（那是覆盖式更新，会留下旧版本快照）。 |
| `UPLOAD_TOO_MANY_PENDING` | 409 | 未提交的分片上传已堆到上限：多半是网络断过之后终端反复重开。 | 先 POST /api/v1/infusion/uploads/cleanup 回收过保留期的半成品，或让护士重新拍一段整的直接走 /tracks。 |
| `UPLOAD_BUDGET_EXCEEDED` | 413 | 这一段超过了单次整段上传的字节上限（UPLOAD_MAX_BYTES，默认 160MB）。**已收到的部分不作废**，只是这一包整体被退回。 | 改走分片续传：POST /api/v1/infusion/uploads 开一次拿 upload_id → POST …/{upload_id}/chunks 逐片补 → GET …/{upload_id} 查断点（received_frames / missing_frames）→ POST …/{upload_id}/commit 提交；提交走的是与 /tracks 相同的主路径与缩放策略，两条路给同一段短片得到同一份数字。 |
| `UPLOAD_TIMEOUT` | 408 | 接收这一段超过了单次整段上传的时限（UPLOAD_MAX_SECONDS，默认 120 秒）。病区网络断在传输中途是常态，不是异常。 | 同上改走分片续传（POST /api/v1/infusion/uploads 起）；已收到的分片留在暂存目录，GET …/{upload_id} 能查到断点在哪一帧。 |
| `UPSTREAM_UNAVAILABLE` | 502 | 出站对端不可达：护理执行平台连接超时、读超时或 5xx。 | 本服务已把事件置 pending_writeback 并按指数退避重试 ≤2 次，不会丢事件；信息科在运行摘要里看待写回条数，对端恢复后可重放。 |
| `INTERNAL` | 500 | 未预期内部错误，已记日志。 | 把响应头里的 X-Request-Id 交给信息科，可据此定位到唯一一行日志。 |
| `WRITEBACK_REJECTED` | 409 | 护理执行平台**业务拒绝**了这条换液事件（4xx）：载荷或业务状态不被接受。重放同一份载荷只会被再拒一次。 | 先看 detail.issues[].reason 里对端给的原因（袋序不存在、输注单号不匹配、状态不允许等），核对该袋的输注单号与床号后再决定是改载荷还是改对端；这笔欠账仍留在待写回清单里，不会被丢掉。 |
| `WEBHOOK_SIGNATURE_INVALID` | 401 | 入站 webhook 兜底通道的签名不通过：缺签名头、格式不是 `sha256=<hex>`、或与请求体算出来的不一致。**请求体不会被解析**，一条都没进缓存。 | 核对两端的签名密钥是否同一份（本服务读 `WEBHOOK_SIGNING_SECRET`）与签名串口径：带时间戳时签的是 `ts.body`，不带时签的是 body。密钥不一致就别去改请求体。 |
| `WEBHOOK_TIMESTAMP_REJECTED` | 401 | webhook 请求带的时间戳与服务端相差超过重放窗口（默认 ±300 秒）。 | 先对时（病区 NTP），别把时间戳头去掉——去掉之后签名就只覆盖 body，等于放弃重放保护（响应里的 `timestamp_enforced` 会变成 false）。 |
| `WEBHOOK_BODY_TOO_LARGE` | 413 | webhook 请求体超过上限（默认 64KB）。 | 一条泵速遥测只有几十字节，超限说明对端塞了别的东西进来（或发错了地址）。确认对端打的是这个端点；本服务不落盘、不解析超限请求体。 |
| `TELEMETRY_REJECTED` | 422 | 遥测载荷不可用：`bag_seq` 或 `pump_rate_ml_per_h` 缺失/类型不对、滴速 ≤ 0，或载荷里的 `rack_id` 与 URL 路径段不一致。 | 只影响这一条遥测，视觉主路径不受影响：滴速退回输注医嘱文本，袋位响应里 `telemetry_state=absent`。核对泵网关发的字段名与架号；滴速 ≤ 0 一律拒收——把「泵停了」当成有效速率会把 ETA 算成无穷大。 |

## 运行期故障

| 故障 | 含义 | 影响面 | 处理方式 |
|------|------|--------|----------|
| `credentials_missing` | 7 组凭据位缺项或为空。 | 服务起不来（fail fast），不存在「起来了但拿不到出站凭据」的中间态。 | 按报错逐行点名补齐环境变量后重启；`python scripts/check_config.py` 可在启动前自查。 |
| `nurse_exec_unreachable` | 护理执行平台出站不可达（连接超时 / 读超时 / 5xx）。 | 换液事件进重试队列并置 pending_writeback；护士侧的点名与确认不受影响。 | 看运行摘要的 pending_writeback_count；对端恢复后由信息科重放，服务不自动无限重试。 |
| `model_unreachable` | 模型网关不可达、超时、返回非 JSON 或结构化校验不通过。 | 该次判定退确定性阈值路径，标 decision_source=threshold_fallback + pending_manual_review=true。**不会**推进业务状态、**不会**写回。 | 阈值路径给出的结论仍可用但需人工复核；看运行摘要的 threshold_fallback_count 是否异常升高。 |
| `mqtt_disconnected` | 输液泵遥测 MQTT 断连（未启用则恒为 stopped）。 | 视觉主路径完全不受影响；ETA 退回医嘱速率，交叉校验能力下降。 | 看运行摘要的 channels.telemetry.last_message_at 判断遥测是否在流；确认 broker 地址与凭据后重启订阅。 |
| `frames_root_not_writable` | 会话帧目录不可写（磁盘满 / 目录权限不对）。 | 上传会以 500 失败；已落库的历史会话不受影响。 | 看磁盘剩余空间与目录属主；见 INSTALL.md 的目录权限一节。 |
| `quota_exceeded` | 模型网关单日调用上限被触发。 | 该次判定直接走阈值路径并记 reason=quota_exceeded，模型不被调用。 | 按需调 LLM_DAILY_CALL_LIMIT，或在非巡视高峰期再放开。 |

## 排障入口

| 想知道 | 看哪里 |
|--------|--------|
| 某一次请求整条链发生了什么 | 日志里按 `request_id` 过滤；响应头 `X-Request-Id` 是同一个 id |
| 现在有几袋等着换、有几条没送出去 | `GET /api/v1/summary` 的 `sessions.needs_change_count` 与 `writeback.pending_writeback_count` |
| 服务能不能用 | `GET /api/v1/health` |
| 配置齐不齐 | `python scripts/check_config.py`（退出码 0 = 齐） |
| 7 组凭据位分别是什么 | `.env.example`，本仓库不含任何真实值 |
