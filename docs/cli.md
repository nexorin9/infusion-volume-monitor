# 命令 / 配置全表（自 README 外移）

本文件是**速查全表**：环境变量、接口端点与错误降级的逐条口径。
README「命令 / API / 配置说明」只留最常用的十几行，日常排障按需查这里；
`GET /api/v1/error-codes` 返回的机器可读码表与 `docs/error-codes.md` 同源。

## 关键配置项（全表）

### 关键配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `NURSE_EXEC_BASE_URL` | 无（必填） | 写回地址 `POST {base}/api/v1/infusion/events` |
| `NURSE_EXEC_API_TOKEN` | 无（必填） | `Authorization: Bearer` 头凭据 |
| `MQTT_HOST` / `MQTT_PORT` | 无（必填） | 遥测订阅 broker |
| `MQTT_TOPIC_PREFIX` | `hospital/infusion` | 订阅 `hospital/infusion/+/telemetry` |
| `LLM_BASE_URL` | 无（必填） | OpenAI-compatible 端点（含 `/v1`），调用 `{base}/chat/completions` |
| `LLM_API_KEY` | 无（必填） | 模型网关 `Authorization: Bearer` 凭据 |
| `LLM_MODEL` / `LLM_TIMEOUT` | 无（必填） / `8.0` | 模型名与单请求超时秒数 |
| `LLM_MAX_TOKENS_PER_REQUEST` | `1200` | 单请求 `max_tokens` 上限；越过即退阈值路径（**不静默截断**）。设 `0` 关闭该护栏 |
| `LLM_DAILY_CALL_LIMIT` | `2000` | 每日模型调用上限，按院内工作日（UTC+8）切；越过即退阈值路径，一次 HTTP 都不发。设 `0` 关闭该护栏 |
| `LLM_FUSION_ENABLED` | `false` | 是否在点名后请模型复述一遍结论做一致性仲裁。**关掉不影响任何结论**：模型缺席时 `decision_source` 恒为 `threshold_only` |
| `LLM_FUSION_CONFIDENCE_FLOOR` | `0.60` | 采信门禁：模型自评置信度低于它一律回落阈值路径（与网关的可用性门禁同源，问的不是同一个问题） |
| `MQTT_ENABLED` | `false` | 是否启动遥测订阅线程；关闭时 ETA 退回医嘱速率，视觉主路径不受影响 |
| `MQTT_RECONNECT_BASE_SECONDS` / `MQTT_RECONNECT_FACTOR` / `MQTT_RECONNECT_MAX_SECONDS` | `1.0` / `2.0` / `32.0` | 断线重连的指数退避形状（1s → 2s → 4s …封顶 32s） |
| `MQTT_RECONNECT_MAX_ATTEMPTS` | `0` | 0 = 一直重试（默认）。设成正数则是「试 N 次就停，把断线摆进运行摘要等人看」 |
| `MQTT_RECONNECT_RESET_SECONDS` | `300.0` | 这条连接稳定站住多久才把退避清零；短于它的「连上又掉」继续退避 |
| `TELEMETRY_FRESH_SECONDS` | `300.0` | 遥测新鲜窗口（秒）。超过即在响应里报 `telemetry_state=stale`，**但不参与门禁** |
| `WEBHOOK_SIGNING_SECRET` | 未设（通道关闭） | webhook 兜底通道的 HMAC 密钥。**未设就没有这个端点**（404 `webhook_disabled`），不用默认密钥开着 |
| `WEBHOOK_SIGNATURE_TOLERANCE_SECONDS` | `300.0` | 带时间戳头的 webhook 重放窗口。对端时钟不准会自己拒掉自己的真消息——先对时，别去掉那个头 |
| `WEBHOOK_MAX_BODY_BYTES` | `65536` | 单条 webhook 请求体上限；超限 `413`，不落盘也不解析 |
| `WRITEBACK_TIMEOUT` | `5.0` | 出站写回单次请求超时（秒） |
| `WRITEBACK_RETRY_ENABLED` | `true` | 是否后台重试到期的待写回条目。关掉后只剩人工重放一条路 |
| `WRITEBACK_RETRY_INTERVAL_SECONDS` | `5.0` | 待写回队列的扫描间隔（秒） |
| `FRAMES_ROOT` | `./var/frames` | 会话帧目录根；分片续传的半成品在它下面的 `_staging/` |
| `UPLOAD_RESUME_RETENTION_HOURS` | `24.0` | 开着没提交的分片上传保留多久，到点由清理接口 / `scripts/purge_partial_uploads.py` 回收 |
| `UPLOAD_RESUME_MAX_PENDING` | `50` | 未提交的分片上传数上限；终端断网反复重开会撞到，超限的新开启请求被拒并指路清理 |
| `MODEL_IMPL` | `geom_kernel` | 分割实现选择；**只有显式设置才算覆盖**，代码里的默认值不会盖住注册表 |
| `MODEL_ID` | `geom_kernel_base` | 换档只改这一项（切暗光档 / 切生产权重档） |
| `MODEL_CONFIG_PATH` | 未设 | 覆盖该档登记的配置文件；显式填了却不存在 → 启动即报错 |
| `MODEL_CHECKPOINT` | 未设 | 覆盖该档登记的权重；权重档缺权重 → 启动即报错并点名缺哪个变量 |
| `MODEL_REGISTRY_ROOT` | `./models` | 注册表根目录；相对路径按它解析，写绝对路径则原样用 |
| `MODEL_REGISTRY_JSON` | 未设 | JSON 对象，按 `model_id` 覆盖/追加注册表；写坏时启动即失败并给示例 |
| `UNCERTAINTY_MIN_CONFIDENCE` | `0.60` | 放行毫升数所需的最低置信度。门限等价于「相对不确定度 > 0.14」 |
| `UNCERTAINTY_HIGH_OCCLUSION_RATIO` | `0.35` | 遮挡帧占比超过此值即判 `high_occlusion`、不出毫升数 |
| `UNCERTAINTY_SIGMA_FILL_FLOOR` | `0.010` | 面积比噪声地板（合成回归实测最差拍法偏差 0.0120） |
| `UNCERTAINTY_GEOMETRY_MODEL_SIGMA` | `0.030` | 腔体几何换算自身的相对误差（登记精度） |
| `UNCERTAINTY_OCCLUSION_GAIN` | `0.20` | 遮挡占比换算成相对不确定度的增益 |
| `UNCERTAINTY_SIGMA_CEILING` | `0.35` | 置信度归零的相对不确定度，与最低置信度合成最终门限 |
| `SMOOTHING_MEDIAN_WINDOW` | `5` | 面积比序列的中值滤波窗口（按观测点计）。5 才能吃掉连着两帧的镜面高光 |
| `SMOOTHING_ALPHA` | `0.6` | 一阶平滑中新观测的权重；压太小会让换液点名迟到一两个巡视轮次 |
| `SMOOTHING_RISE_TOLERANCE` | `0.004` | 单帧允许的面积比上升量；超过即判反常帧（液面物理上只会降） |
| `SMOOTHING_SPIKE_TOLERANCE` | `0.02` | 原始值与中值偏离超过此值标尖峰 |
| `SMOOTHING_MIN_GAP_FRAMES` | `2` | 缺帧少于此数不做受限重建（保持 `null`） |
| `SMOOTHING_MAX_GAP_FRAMES` | `10` | 受限重建的最大缺帧数；不得小于 `SMOOTHING_MIN_GAP_FRAMES`，否则启动即报错 |
| `QUALITY_SAMPLE_FRAMES` | `5` | 每段均匀采样几帧做拍摄质检；与短片长度无关，长短片的质检开销相同 |
| `QUALITY_MIN_HIT_RATIO` | `0.34` | 至少这个比例的采样帧命中才算这段有问题（抗单帧偶发误判） |
| `QUALITY_GLARE_RATIO_THRESHOLD` | `0.0015` | 近白无彩色像素占比超此值判「反光过曝」 |
| `QUALITY_LOW_LIGHT_MEAN_LUMA` | `46.0` | 全画面平均灰度低于此值判「暗光」 |
| `QUALITY_SHARPNESS_MIN` | `150.0` | 拉普拉斯方差下限，低于此值判「运动模糊」（暗光帧不参与该判定） |
| `QUALITY_TILT_ANGLE_DEG` | `8.0` | 液面线倾角超此值判「角度不正」 |
| `QUALITY_DRIP_TAIL_RATIO` | `0.55` | 腔体底部窄带填充率低于此值判「滴斗空管」（info 级，不建议重拍） |
| `MODEL_SATURATION_MIN` | `96` | 液体的饱和度下限；几何核与上传前预检**共用**这一个门限，保证两层量的是同一批液体 |
| `PRECHECK_MIN_SHORT_SIDE` | `480` | 上传前预检的短边**质量**下限（硬下限 320 仍由入站校验把关，低于它直接 `422`） |
| `PRECHECK_DEGRADED_SCORE` | `0.70` | 五项子分加权平均低于此值即标 degraded，即便一类都没命中 |
| `FRAME_RESIZE_ENABLED` | `true` | 解包落盘时是否把长边缩到上限；关掉则原样处理（算力换不来更准的数字） |
| `FRAME_LONG_EDGE_MAX` | `1280` | 长边缩放上限（px）。形态学核是固定像素尺度的，再大只更慢不会更准 |
| `FRAME_SHORT_EDGE_FLOOR` | `320` | 短边下限（px）。缩放**不得**把画面缩到这条线以下，否则同一段短片会因为「缩放策略」而不是「拍法」被判 422 |
| `ROI_ENABLED` | `true` | 是否只把输液架那块画面喂给预测器。关掉则整幅处理，数字相同、算力更高 |
| `ROI_MARGIN_RATIO` / `ROI_MAX_STEP_RATIO` | `0.12` / `0.05` | 窗口外放余量、单步移动上限（均按长边比例）。液面下降要留余量，场景突变不能一步跳走 |
| `ROI_ALPHA` / `ROI_MIN_SIDE` | `0.35` / `96` | 窗口增长时的平滑系数；窗口短边下限，比这更小没有分辨力，宁可不开窗 |
| `UPLOAD_MAX_BYTES` | `167772160`（160MB） | 单次整段上传字节上限；超限 `413` 并指路分片续传 |
| `UPLOAD_MAX_SECONDS` | `120.0` | 单次整段上传接收时限（秒）；超限 `408` 并指路分片续传 |
| `DATABASE_URL` | `sqlite:///./var/infusion_monitor.db` | 连接串；换 PostgreSQL 只改此项 |
| `RETENTION_SESSION_DAYS` | `30.0` | 输液架帧目录保留天数。到点后帧被清掉，**读数仍留在库里**；会话还没结束或换液事件还没送达的，一个都不动 |
| `RETENTION_INFERENCE_AUDIT_DAYS` | `90.0` | 模型调用元数据的保留天数。审计链（飞检留档）任何保留期都不动 |
| `RETENTION_ARCHIVE_ROOT` | 未设（到期直接删除） | 帧目录归档根目录；设了就按月分目录搬过去而不是删掉，飞检期内还能调原始帧 |

## CLI 子命令 / 接口端点（全表）

### CLI 子命令 / 接口端点

| 命令或路径 | 用途 | 关键参数 |
|------------|------|----------|
| `GET /api/v1/health` | 健康检查：schema 版本、帧目录可写、会话总数 | 无 |
| `POST /api/v1/infusion/tracks` | 上传帧序列，返回每袋余量曲线与点名 | `rack_id`/`infusion_order_no`/`bed_no`/`captured_at`/`slots`/`frames`(zip)/`client_event_id` |
| `POST /api/v1/infusion/calibrations` | 登记袋型标定（首次建版 / 再次改版）；带 `cavity{}` 时毫升数改走袋型腔体几何换算 | `bag_type_code`/`capacity_ml`/`dead_volume_ml`/`samples[]`/`cavity{}`/`calibrated_by`/`effective_from`（留空=立刻生效，填未来时刻=预约登记） |
| `GET /api/v1/infusion/calibrations` | 查生效中的标定记录 | `bag_type_code`（可选，带上则返回该袋型全部版本） |
| `GET /api/v1/infusion/calibrations/recalc` | 查标定改版后的重算轨迹 | `bag_type_code`（可选） |
| `GET /api/v1/infusion/calibrations/{袋型}/versions` | 查某袋型当前生效版本、历史版本与各自生效区间（到点的预约版本自动转正） | 无 |
| `GET /api/v1/infusion/calibrations/{袋型}/reviews` | 查该袋型的改版偏差待复核清单（新旧读数、相对差、是否已结案） | `resolution`（可选） |
| `POST /api/v1/infusion/calibrations/{袋型}/reviews/{alert_id}` | 复核结案：`applied` 按新读数落库、`dismissed` 沿用旧读数 | `resolution`/`reviewed_by`/`note` |
| `GET /api/v1/infusion/calibrations/deviations` | 查自校准残留偏差的样本与结论（哪条参与定标、哪条没参与及原因） | `bag_type_code` |
| `GET /api/v1/infusion/sessions/{id}` | 查会话当前状态与本轮点名 | 无 |
| `GET /api/v1/infusion/pickups` | 病区换液点名清单（只读）：按 ETA 升序 → 床号 → 袋序排好，每条带点名依据与确认入参 | `rack_id`（可选，按架过滤）/`limit`（≤100） |
| `POST /api/v1/infusion/transition` | 护士确认换液，推进到 `replaced` 并幂等写回 | `session_id`/`bag_seq`/`operator_id`/`client_event_id`（可选，原样回填清单里的值）/`rack_id`（可选） |
| `POST /api/v1/infusion/claim` | 在候选列表里认领一袋（或改绑），按认领的候选**重算整条曲线** | `session_id`/`candidate_id`/`operator_id`/`note` |
| `POST /api/v1/infusion/decision` | 重算某一袋的换液点名判定（双条件），只产出 `needs_change` | `session_id` |
| `GET /api/v1/infusion/pending` | 待写回清单（只读）：还没送到护理执行平台的换液事件——欠了多久、试了几次、最后一次错在哪 | `rack_id`（可选）/`bed_no`（可选）/`status`（可选：`queued`/`dead_letter`/`failed`）/`limit`（≤500） |
| `POST /api/v1/infusion/pending/{event_id}/replay` | 人工重放一笔欠账（幂等：只投已存在的事件，不产生第二条换液记录） | `operator_id`（必填，重放人；谁补的要留痕） |
| `GET /api/v1/infusion/audit` | 导出审计包明细（CSV，只读；带 BOM，Excel 双击能开） | `rack_id`/`bed_no`/`occurred_from`/`occurred_to`/`category`/`action`/`outcome`/`bag_seq`/`limit` |
| `GET /api/v1/infusion/audit/manifest` | 审计包首页（只读）：行数、七类动作分布、时间窗、链摘要 | 与上一行同一套过滤条件 |
| `POST /api/v1/infusion/uploads` | 开启一次可续传的上传，拿 `upload_id` | 与 `/tracks` 同口径的袋位声明 + `expected_frames`（申报总帧数才能算「还缺哪几帧」） |
| `POST /api/v1/infusion/uploads/{id}/chunks` | 补传一个分片（zip，成员按**整段全局序号**命名） | `chunk`/`frame_times`（可选）/`frame_offset`（终端只能按片内编号时传） |
| `GET /api/v1/infusion/uploads/{id}` | 断点查询：已收帧号、还缺帧号、重复计数、分片数 | 无 |
| `POST /api/v1/infusion/uploads/{id}/commit` | 提交：已收齐的帧走**与 `/tracks` 相同**的主路径算余量（响应同形） | 无 |
| `POST /api/v1/infusion/uploads/cleanup` | 回收保留期已到的半成品（cron / 运维） | `retention_hours`（可选，覆盖配置值）/`dry_run` |
| `GET /api/v1/infusion/sessions/{id}/versions` | 查整段重放前的旧版本快照（护士长回查「上一版是多少毫升」） | 无 |
| `POST /api/v1/infusion/telemetry/webhook/{rack_id}` | 入站泵速遥测的 **webhook 兜底通道**（给没有 MQTT broker 的病区）；与 MQTT 共用同一份解析与去重 | `bag_seq`/`pump_rate_ml_per_h`（必填）/`device_sn`/`ts`/`rack_id`（带了就必须与路径一致）；头 `X-Infusion-Signature`（必填）、`X-Infusion-Timestamp`（可选，带了就校验重放窗口） |
| `GET /api/v1/summary` | 运行摘要：待换液袋数、待写回积压、模型降级次数、通道状态（值班入口） | 无 |
| `GET /api/v1/error-codes` | 错误码表（机器可读）：每个码的 HTTP 状态、含义、处理方式、会返回它的接口清单 | 无 |
| `GET /docs` | OpenAPI 3.1 契约页：每个接口的 responses 里已逐条写明它会返回的错误码 | 无 |
| `python scripts/purge_expired_sessions.py` | 保留期清理（cron / 运维）：到点的输液架帧目录归档或删除，读数留在库里 | `--retention-days`/`--archive-root`/`--dry-run`/`--json` |
| `python scripts/estimate_storage.py` | 容量测算：按病区规模算帧目录占用与各表增长，并给出换库判据 | `--wards`/`--frames-per-round`/`--frames-root`（用真实帧实测）/`--database`/`--json` |

## 错误与降级（全表）

### 错误与降级

| 场景 | 表现 | 处理方式 |
|------|------|----------|
| 帧数 < 12 或 > 1200、短边 < 320px、时间戳逆序、槽位 > 6 | `422`，detail 为 `field + reason + suggestion` 三段式 | 按建议补拍角度重拍，不静默丢弃 |
| 整段上传超过 `UPLOAD_MAX_BYTES`（默认 160MB） | `413 UPLOAD_BUDGET_EXCEEDED` | 改走分片续传（`POST /api/v1/infusion/uploads` 起）；**已收到的部分不作废**，只是这一包整体退回 |
| 整段上传超过 `UPLOAD_MAX_SECONDS`（默认 120 秒） | `408 UPLOAD_TIMEOUT` | 同上。病区网络断在传输中途是常态，不是异常；已收到的分片留在暂存目录，`GET …/{upload_id}` 查得到断点 |
| ROI 窗口开不起来（没候选框 / 窗太小 / 并集等于整幅） | 整幅处理，`roi.fallback_reason` 如实报出是哪一条 | 不为了省算力去猜一个窗——窗开错会直接变成一条假的面积曲线 |
| 状态迁移非法（`monitoring` 直跳 `replaced`、`replaced` 回退） | `409` + 审计行（`outcome="rejected"`） | 走合法路径：先产出点名，再由护士确认 |
| 缺帧就提交（已申报总帧数但没收齐） | `422 chunk_gap`，detail 点名缺哪几帧 | 按 `missing_frames` 补传那几片；**已收到的分片不重传、不丢** |
| 分片顶着一个已收到的帧号、内容却不是同一帧 | `422 chunk_frame_conflict`（只拒这一片） | 多半是该片把帧号从 0 重排：按整段全局序号命名，或显式传 `frame_offset`；若帧确实已在断点清单里，直接跳过本片 |
| 审计导出：`category` 拼错或 `outcome` 不是 accepted/rejected | `422`，issue 为 `audit_category_unknown` / `audit_outcome_unknown` 并列出可用取值 | 按 issue 里的清单改拼写；类别用 `claim`/`calibration`/`pickup`/`confirmation`/`writeback_sent`/`writeback_failed`/`writeback_pending`/`other` |
| 这一袋从来没收到过泵速遥测 | `telemetry_state="absent"`，`telemetry_age_seconds` **不出现**；`eta_source` 退回 `order_text` | **什么都不用做**：余量曲线、毫升换算、换液判定全都不受影响；滴速少一个交叉校验来源而已 |
| 遥测收到但超出新鲜窗口 / 泵没报时间戳 | `telemetry_state="stale"` / `"undated"`，仍照常用于医嘱与泵速的一致性校验 | 过期读数**照样参与**冲突判定（医嘱 5 ml/h、泵上 40 ml/h 这种更要查）；新鲜度只报状态，不作门禁 |
| MQTT 断线 / 一直没接上 | 运行摘要 `channels.mqtt.state="disconnected"` + 退避表（尝试次数、退避时长、最后一次结果） | 本服务自己按指数退避重连（1s→32s 封顶），运维要做的只是查 broker；要停掉重连把 `MQTT_RECONNECT_MAX_ATTEMPTS` 设成正数 |
| webhook 请求签名不对 / 缺签名头 / 格式不对 | `401 WEBHOOK_SIGNATURE_INVALID`，issue 为 `signature_missing`/`signature_malformed`/`signature_mismatch` | 核对两端密钥是否同一份，以及签名串口径：带时间戳时签 `ts.body`，不带时签 body。**验签在解析之前**，不通过的请求体不会被读进缓存 |
| webhook 时间戳超出重放窗口 / 不是 epoch 秒 | `401 WEBHOOK_TIMESTAMP_REJECTED` | 先给病区对时（NTP），**不要去掉时间戳头**——去掉之后签名只覆盖请求体，等于放弃重放保护 |
| webhook 请求体超 `WEBHOOK_MAX_BODY_BYTES` | `413 WEBHOOK_BODY_TOO_LARGE`，不落盘也不解析 | 一条泵速遥测只有几十字节；超限说明对端发错了地址 |
| webhook 载荷里的 `rack_id` 与路径段不一致 | `422 TELEMETRY_REJECTED`，issue 为 `rack_mismatch` | 架号以 **URL 路径段**为准。宁可整条拒，也不让一份跨架的遥测写进另一架的缓存（事后查不出来源） |
| webhook 重复投递（同设备序列号 + 同时间戳） | `202` + `{"duplicate": true, "accepted": false}` | **不是错误**：对端拿 4xx/5xx 会当成失败去重投，而重复的成因往往就是「上次重投没收到应答」，那会变成无限重投 |
| 异线程（broker 回调 / 重连线程）攒下的通道审计积压 | 运行摘要 `channels.mqtt.pending_audit.queued` / `.dropped`；探活与摘要请求会顺手落库 | 主库连接带线程亲和性，异线程直接写会把 paho 的网络线程打死（遥测从此静默消失）。所以异线程只入队、由事件循环线程落库；缓冲上限 512 条，满了丢最老的并计数——**丢多少摆在摘要里，不静默** |
| 审计导出：`limit` 超单次上限 | `422 limit_too_large`，提示改按架/床/时间窗分段导 | 分段导出的每段各自带链摘要，合并时按 `audit_id` 排序即可复核 |
| 审计写盘失败（磁盘满/库被锁） | 业务写入**一起回滚**，接口报错而不是「改了但查不到谁改的」 | 排查磁盘与库文件权限后重做那一步；状态没变，重做不会重复计入 |
| 对已完成/已清理的上传继续传片或提交 | `409 UPLOAD_ALREADY_COMPLETED` / `UPLOAD_NOT_OPEN` / `NOT_FOUND` | 已提交过的不用重传；要重传整段走 `/tracks`（覆盖式，留旧版本快照） |
| 保留期清理撞上还在核对的会话 | 该袋被**留下**并带原因码（`session_not_finished` / `writeback_unsettled` / `writeback_queue_pending` / `frames_shared_with_active_session`） | 不用管：护士确认换液、事件送达对端之后，下一轮清理自然会收走。**没确认过的帧不会被删** |
| 帧目录被清理后再来认领候选 | `404`，issue 为 `frames_gone`，提示重新上传这一轮短片 | 读数（毫升数 / ETA / 换液结论）仍在库里可用；只有原始像素没了。要留像素就配 `RETENTION_ARCHIVE_ROOT` |
| `DATABASE_URL` 指向 PostgreSQL 但本机没装驱动 | 启动即报错并点名 `pip install 'psycopg[binary]'` | **不会静默退回 SQLite**——那种故障要到数据对不上时才发现。装一次驱动即可，产品代码不用改 |
| `DATABASE_URL` 的 scheme 拼错（如 `mysql://`） | 启动即报错并列出可用写法 | 按报错改；不认识的连接串一律不猜 |
| 未提交的分片上传堆到上限 | `409 UPLOAD_TOO_MANY_PENDING` | 先跑清理接口或 `scripts/purge_partial_uploads.py`；多半是断网后终端反复重开 |
| 认领了本轮不存在的候选 | `422`，issue 为 `candidate_not_found`，列出本轮可认领的候选 | 从上传响应的 `candidates` 里选一行 |
| 认不出「还是不是上一轮那袋」 | `reanchor_state="unbound"` + 原因码（`ambiguous_slot`/`low_confidence`/`bag_seq_mismatch`/`no_candidate`） | **不猜**：护士在候选列表里点一行认领，系统按新候选重算曲线 |
| 挂点对得上、但外观变了（换新袋的典型样子） | `reanchor_reason="appearance_changed"`，`reanchor_driver` 点名是哪一项对不上（`colour`/`level`/`label`） | **不猜**：转人工认领，响应里**不出现** `previous_session_id`，新袋不会接到旧曲线上 |
| 同一挂点上的袋子比上一轮还满（液柱升高） | 直接判 `appearance_changed`（液位一票否决），不信「颜色对得上」 | 请护士确认是否换过袋；系统不把新袋的满袋当成旧袋的余量 |
| 标签区这轮看不清（整段偏暗 / 观察带出画 / 全被挡） | `descriptor_mode="color_only"`：标签项**退出加权**（不是按 0 分算），档位写进响应与审计 | 颜色与液位照常参与判定；要更稳的下轮换个角度拍 |
| 标签区有→无（袋子被翻面） | 标签项判 0 并标 `reanchor_driver="label"`，但**单凭这一项不解除绑定** | 翻面不换袋，接上是对的；真换袋时颜色或液位会同时塌，那时才判认不准 |
| 换完袋后上一轮的袋 | `reanchor_state="fresh"`，不挂旧会话 | 新袋从本轮起单独一条曲线，不接到旧袋上 |
| 某只袋子认错了挂点 / 中间挂位空着 | 该袋 `slot_state` 为 `mismatched`/`below_score`，空位报 `unfilled`，两处都 `claim_required` | **不自动改绑**：在候选列表里点一行认领；把 A 袋的余量报给 B 袋比报不出数字更危险 |
| 人工重放没送出去（对端不可达 / 业务拒绝） | `409 WRITEBACK_REJECTED`，`issues[].reason` 写明失败类别（`http_4xx` 是业务拒绝，`http_5xx`/超时是对端不可达） | 按 reason 分头处置：业务拒绝先核对输注单号、床号与袋序；对端不可达等恢复后再点一次重放。**这笔欠账仍留在待写回清单里，不会丢** |
| 某只袋子整段被挡住 | 该袋 `slot_state="no_evidence"`、曲线整段留 `null`，**不占**任何挂点 | 该挂点单独报「这段没拍到」；把镜头对准这一挂点补拍一段，同架其余袋子的读数不受牵连 |
| 反光高光在液面上挖出一个洞 | 该帧进 `smoothing.spike_frames`，原始序列保留原值，可信序列照真实缓降走 | 毫升数与 ETA 不吃这一帧；护士端会看到「这段拍糊了」提示，建议换个角度补拍 |
| 液面出现物理上不可能的上升（滴斗空管 / 中间换过袋） | 该帧进 `smoothing.outlier_frames`，**保留原值**但退到最近一次可信读数 | 不会把升上去的液面当成新袋的满袋余量；换液后请重新上传一轮 |
| 遮挡段前后都拿不到可信读数（挡太久 / 中间换袋） | `area_ratio_reconstructed` 该段保持 `null`，`smoothing.gaps[].reason` 给出原因码 | **不补**：把镜头对准这一挂点补拍一段，比给一条编出来的曲线有用 |
| 液面在换液阈值附近反复抖（反光 / 滴斗晃动） | 该袋 `pickup_suppress_reason="hysteresis_hold"`，`pickup_detail` 写明卡在滞回带第几轮 | **不点名**：判不准就不催。连续 3 轮都停在带内才按「确已跌破」收口（`hysteresis_resolved`） |
| 这袋已经点过名、液位还没回升 | `pickup_suppress_reason="latched"` | 不重复点名——误报一次就耗尽信任，等护士换袋（对端给新的 `bag_seq` 自动开新锁） |
| 距上一次点名不足 15 分钟（含进程重启后） | `pickup_suppress_reason="interval_not_elapsed"`，detail 给出实际间隔分钟数 | 上一次点名时刻从审计链读（带 `bag_key` 列），重启不清零；护士可在该袋换完后立即再传一轮 |
| 一轮里该换液的袋超过 3 个 | 多出的按 ETA 升序截断，`pickup_truncated=true`、`pickup_rank` 给到名次，审计记 `pickup.truncated` | **截断不是丢弃**：下一轮重新参与，`pickup_detail` 写明排在第几位被截断，护士长能回答「为什么 C01 没被点名」 |
| 同 `client_event_id` 重复确认 | `200` + `{"idempotent_hit": true}`，返回既有 `event_id` | 事件表不新增行，重复提交安全；护士按钮重试不弹红框 |
| 终端自己编了一个幂等键（与规则算出来的不一致） | `422`，issue 为 `client_event_id_mismatch`，并把「这袋按规则应该是什么键」写在 `reason` 里 | 原样回填点名清单里的 `client_event_id`，或整个字段留空由服务端生成。**自编的键会把一次换液在对端变成多条事件** |
| 点名清单 `limit` 要得太大 | `422`，issue 为 `limit_too_large`（上限 100） | 改用 `rack_id` 按架拉取，或把 `limit` 调到 100 以内——一张几百行的清单护士长也不会用 |
| 这一袋这一段完全没测到（整段被挡 / 袋腔没枚举出） | 置信度字段**不出现**（不是 0），`confidence_factors.reason="no_observation"` | 0 会被读成「测过了但不可信」；不出现才是实话。对准这一挂点补拍 10–20 秒 |
| 两台终端同时确认同一袋 | 恰好一路落库，另一路幂等命中（或极窄窗口下 `409`） | 写库事务串行化 + 锁内复核前提，不双写 |
| 确认请求没点名袋序 | `422`，issue 为 `bag_seq_missing` | 补 `bag_seq`（与上传响应 `bags[].bag_seq` 一致） |
| 确认写的袋序不属于该会话 / 输液架对不上 | `403 WRITE_NOT_ALLOWED` + `outcome="rejected"` 审计 | 先在候选列表认领正确袋序，再重新确认；系统不替人猜 |
| 袋型无标定记录 | 响应中**不出现** `estimated_volume_ml`，`calibration_state="uncalibrated"`，`pickup_suppressed=true` | 先补标定记录（补录后该袋型历史会话自动重算）；在此之前只按曲线判断 |
| 滴速给不出来（医嘱与遥测都没有） | 毫升数照给，`eta_minutes` 不出现，`eta_source="unavailable"`，该袋不点名 | 补录输注医嘱速率；不拿猜出来的时点催换袋 |
| 医嘱速率与泵速遥测对不上（相对偏差 > 20%） | 毫升数照给，`eta_minutes` **不出现**，`eta_source="conflict"`，`rate_conflict` 带出医嘱值 / 泵速值 / 偏差，该袋不点名 | **不取中间值**：请护士确认是医嘱没改还是泵没调，改完重传一轮；按错的速率算出的时点比没有时点更危险 |
| 余量太小或这一段抖动太大，算出来的毫升数落在一段宽区间里 | `estimated_volume_ml` **不出现**，`calibration_state="low_confidence"`，`volume_uncertainty` 给出区间、置信度与成因 `relative_uncertainty_too_high` | 门限不是拍脑袋定的（相对不确定度 > 0.14，由「灵敏度 × 噪声 ÷ 毫升数」算出）：余量约 41 ml 以下天然更严。等液面再低一些、或把手机端稳了重拍 |
| 这段挡得太久（遮挡帧占比 ≥ 35%） | `estimated_volume_ml` **不出现**，`calibration_state="high_occlusion"`，成因 `occluded_too_much`，区间仍在 | 等挡开以后对准输液架再拍 10–20 秒；10–20 秒里挡掉三分之一以内仍会出数，只是区间更宽 |
| 袋型的腔体外形参数登记过、但没有版本号 | `estimated_volume_ml` **不出现**，成因 `geometry_version_missing`，提示点名让信息科补登记版本号 | 外形参数改了不升版本，就说不清历史读数是否还成立；补登记一个版本号，毫升数自己回来（改版重算同步走） |
| 短片拍得不好（出框 / 糊 / 反光 / 歪 / 暗光） | 该袋 `capture_degraded=true`，`capture_quality` 给出问题码 + 中文补拍建议；**仍返回 `201`**，曲线与毫升数照给 | 按 `capture_quality[0].suggestion_zh` 重拍这一段。**不拒收**——拍坏与查不出数是两回事，拒绝权只留给帧数/分辨率/帧序这些硬校验 |
| 夜里拍得又暗又糊 | 只报 `low_light`，**不报** `motion_blur` | 先开灯重拍；灯亮之后若仍糊，下一轮质检会单独点出「运动模糊」 |
| 整段拍得都不太行（上传前预检） | 响应顶层 `precheck.state="degraded"`，给出 0–1 `quality_score`、五项子分与命中的问题；每袋另有 `precheck_score` / `precheck_degraded` | 按 `precheck.primary.suggestion_zh` 重拍；质量分低**不影响**袋里的曲线与毫升数，只表示这些数字要打问号 |
| 短边低于质量下限（默认 480px，仍 ≥ 硬下限 320px） | `precheck` 段报 `low_resolution` 并建议贴近或开 2× 变焦；**仍返回 `201`** | 320px 是硬下限（低于它直接 `422`）；480px 是质量下限，只提示不拒收 |
| 输液架上两袋贴到只剩 ≈2 px 缝（已完全看不出缝） | 合成一块，那一块的面积比偏大 | 请护士把两袋拉开一点重拍；系统在像素上无法区分「贴住的两袋」与「一个很宽的袋」，不假装分得开 |
| 选了权重档但权重文件不在盘上（拷贝漏了 / 挂载点没挂） | 服务**启动即失败**，点名 `MODEL_CHECKPOINT`、期望路径与「为什么不退回几何核」 |把权重放到该路径并保证进程可读，或把 `MODEL_ID` 改回几何核档；`python scripts/check_config.py` 可在启动前查 |
| `MODEL_ID` 写错 / `MODEL_IMPL` 指名未注册的实现 / `MODEL_REGISTRY_JSON` 不是合法 JSON | 启动即失败，报错列出当前可用的 id 与已注册的实现，并给出 JSON 示例 | 照报错改配置；映射表写坏时不会静默按内置表启动 |
| 标定改版 | 历史会话按新版本重算，差异写入重算轨迹 | `GET /calibrations/recalc` 可核对每条会话改前/改后的毫升数 |
| 新登记的标定生效时刻落在旧版生效之前（回填历史） | `409 CALIBRATION_WINDOW_CONFLICT`，issue 点出冲突的是第几版 | 同一袋型不能有两个版本同时生效；把生效时刻挪到旧版之后，或给旧版补 `effective_to` |
| 改版后同一袋的毫升读数相对差超过 15% | 该袋**旧读数原样保留**，挂 `calibration_review_required`，告警进 `GET /calibrations/{袋型}/reviews` 待复核清单 | 差这么大量多半是新版本自己有问题，不是旧版本终于被修正；护理部复核后 `POST …/reviews/{alert_id}` 结案 |
| 同一条告警被复核两次 | `409` + `review_already_resolved`，报出先处理的复核人与时刻 | 先查该条的 `resolution`；已结案的不必重复提交 |
| 面积比超上限（几何核把袋外液体或挂壁高光算进前景） | `overfilled=true`，毫升与 ETA 字段**整体不出现**，`pickup_suppressed=true` | 请把镜头对准袋体中部避开袋外反光重拍；宁可让护士白跑一趟，也不给一个 1000 毫升的假读数 |
| 袋型登记了腔体参数，但自校准证据不足 / 互相打架 | 读数照给、标 `resident_ml`，但**不扣**残差；`GET /calibrations/deviations` 里能看到结论与原因 | 证据攒够 ≥3 条且离散度在上限内才自动生效；打架时请复核该袋型的腔体外形参数与换液确认记录 |
| 液面低到腔体里那点液体与残留同一量级（≤1 ml） | 报 `0` 毫升并标 `residual_limited=true` | 按输完处理；要核对就翻 `GET /calibrations/deviations` 里该袋型的残留结论 |
| 出站写回 5xx / 连接或读超时 | 指数退避重试（默认 1s、4s）≤2 次后置 `pending_writeback`，事件行保留 | 进待写回队列，可由信息科重放；绝不静默丢事件 |
| 出站写回 4xx（对端业务拒绝） | 不重试，事件置 `failed`，`last_error` 里带对端自己给的原因（`reason=`） | 同一份载荷重发只会再被拒一次；按对端返回的拒绝原因改载荷后重放 |
| 半开连接（对端把连接掐了却没给应答） | 归 `half_open` 一类，退避重试后仍失败则 `pending_writeback` | **请求多半已经送到对端**，所以重试必须靠 `Idempotency-Key` 兜底；先确认对端去重生效，再重放 |
| 连接超时与读超时混在一起看不出病根 | `last_error` 的 `counts=` **按类别分别计数**（`connect_timeout=3` vs `read_timeout=3`） | 前者查病区网段到对端的连通性，后者查对端处理速度——合成一个总数两边都查不到 |
| 工作站配了系统代理（病区机器常见） | 出站客户端不继承系统代理（`trust_env=False`） | 换液事件按 `NURSE_EXEC_BASE_URL` 直达护理执行平台；否则事件被绕出院内网络，且代理会把传输层失败统一翻成 502 让人查错方向 |
| 对端回了看不懂的状态码（3xx / 未知码）或 `base_url` 少写 `http://` | 归 `http_unexpected` / `protocol_error`，**不重试**，事件置 `failed` | 盲目重发只会再来一次同样的错；先核对对端网关约定与 `NURSE_EXEC_BASE_URL` |
| 会话尚未被护士确认就要求写回 | 拒发：一次 HTTP 都不发，事件保持 `pending` + 拒发审计 | 这是门禁不是故障；护士确认换液后才会写回 |
| 写回错误摘要里混入凭据 | 出站失败时网关可能回显请求头 | 摘要统一过脱敏，token 值不落库、不进日志 |
| 模型网关不可达 / 超时 / 5xx / 输出不可解析 | 退确定性阈值路径，`decision_source="threshold_fallback"`，`fallback_reason` 记 `connect_error` / `connect_timeout` / `read_timeout` / `http_5xx` / `parse_error` | 四类分开记便于分头排查；退避重试 ≤2 次后降级，**不写回、不推进换液状态**，标 `pending_manual_review` |
| 7 组凭据位缺项或为空 | 进程直接起不来，并逐组点名缺哪一个变量名 | 先跑 `python scripts/check_config.py`（退出码 0 = 齐）；它还会查帧目录与库目录**能不能写** |
| 输液泵遥测断连（MQTT） | `channels.telemetry.state="disconnected"`，视觉主路径不受影响 | ETA 退回医嘱速率；看 `last_message_at` 判断遥测是否在流，确认 broker 后重启订阅 |
| 请求报错但只拿到一句英文 | `detail.error_code` 与 `detail.issues[]` 三段式 | 完整对照见 [`docs/error-codes.md`](error-codes.md)（由代码生成，与 `/docs` 描述一致） |
| 模型给出的结论缺 `trigger_reason` 或自评置信度低于 0.60 | 同上，按 `parse_error` 降级 | 模型说「我只有 45% 把握」不该被采信；结论退回阈值路径，人来兜一眼 |
| 模型输出 `replaced`（越权要换液） | 契约校验拒收，降级阈值路径 | 换液确认只能由护士在终端点出来，模型**没有**这个权力 |
| 今日模型调用已达 `LLM_DAILY_CALL_LIMIT`（默认 2000 次） | **一次 HTTP 都不发**，退阈值路径，`fallback_reason="quota_exceeded"`，`attempts=0` | 护栏只挡不改数据：结论照常由阈值路径给出，换液判定不受影响。看运行摘要 `model.quota.used_today / remaining_today / exhausted`；额度按**院内工作日（UTC+8）**切，跨日自动归零 |
| 本次请求 `max_tokens` 越过 `LLM_MAX_TOKENS_PER_REQUEST`（默认 1200） | 同上退阈值路径，报错文本里同时报出「要的数」与「上限」 | **不做静默截断**——截断会让审计记的 `max_tokens` 与端点实际收到的不是一回事，出事查不出来。把配置调大，或把这次要的输出长度降到上限内 |
| 端点变慢（p95 延迟悄悄涨） | 运行摘要 `model.runtime_metrics.latency.p50_ms / p95_ms / max_ms` | 上报分位而不是均值：均值被一串快请求压住，p95 才会跟着端点一起涨。护栏被动用后还有 `quota_blocked` 与 `requests_sent` 可对账——「没花出去的钱」也是一个可核对的事实 |
| 输液泵遥测缺失 | ETA 仍可由医嘱速率给出，`eta_source` 标明来源 | 视觉主路径不受影响 |
| 遥测主题缺段或前缀对不上（如少写架号段） | 只拒这一条并计 `telemetry.message_rejected`，不写进缓存 | 核对泵网关的订阅路径是否与 `MQTT_TOPIC_PREFIX` 一致；错架号的遥测只会污染 ETA 交叉校验 |
| 凭据缺失 | 启动失败并按名列出缺哪一项 | 按 `.env.example` 补齐 |

## 对接现有系统

### 对接现有系统

- **出站（写）**：`POST {NURSE_EXEC_BASE_URL}/api/v1/infusion/events`，
  `Authorization: Bearer $NURSE_EXEC_API_TOKEN`，幂等键同时出现在请求头
  `Idempotency-Key` 与请求体 `client_event_id`（两侧同值，对端任一侧去重都不漏）。
  出站只发生在护士确认触发 `replaced` 之后；字段规范落盘在 `docs/nurse-exec-contract.md`
  （由代码里的字段表生成，改字段只改一处）。
- **入站（上传）**：`POST /api/v1/infusion/tracks`，multipart/form-data，帧为 zip。
- **入站（遥测）**：MQTT `hospital/infusion/+/telemetry`，字段 `bag_seq`/`pump_rate_ml_per_h`/`device_sn`/`ts`。
- **模型判定（出站，仅提示）**：`POST {LLM_BASE_URL}/chat/completions`，
  `Authorization: Bearer $LLM_API_KEY`，请求带 `response_format={"type":"json_object"}`。
  业务层只经**单一模型网关**调用（换院内自建端点或本地部署权重只改环境变量，
  不动业务代码）；返回字段契约落盘在 `docs/llm-contract.md`（由代码里的 schema 生成）。
  模型的结论只是一条提示，**不推进任何业务状态**——`needs_change → replaced`
  永远只由护士确认触发。
- 全部凭据位读环境变量，代码与配置文件中不出现明文；自动化测试只连本地测试替身，
  真实护理执行平台端点需院内联调环境接通。

## 相关文档

| 文件 | 内容 |
|------|------|
| [`error-codes.md`](error-codes.md) | 错误码 → HTTP 状态 / 含义 / 处理方式（由 `errors.py` 生成） |
| [`nurse-exec-contract.md`](nurse-exec-contract.md) | 出站写回契约：端点、鉴权、14 个字段、11 类失败分类 |
| [`llm-contract.md`](llm-contract.md) | 模型输出契约：字段、枚举、置信门禁、降级语义 |
| [`webhook-contract.md`](webhook-contract.md) | 入站 webhook 兜底通道：签名口径、载荷、拒收原因码 |
| [`prompt-templates.md`](prompt-templates.md) | 提示词模板注册表：模板名 / 版本 / 适用场景 |
| [`INSTALL.md`](../INSTALL.md) | 环境变量在病区机器上的落地方式（systemd、目录权限） |
