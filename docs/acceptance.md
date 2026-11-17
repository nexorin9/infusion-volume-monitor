# 验收与自检全表（自 README 外移）

README「验收与自检」只留代表性十几行；下面这张全表逐条给出**可复跑的命令**与
**期望结果**，覆盖上传、标定、体积门禁、幂等并发、出站写回、审计链、遥测、
模型网关、保留期与评测取数。

跑法约定：

```bash
cd infusion-volume-monitor
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pytest -q                                     # 全量
PYTHONPATH=src:. python3 var/verify_task44.py # 单条真进程复核（var/ 下按任务编号命名）
```

## 验收全表

## 验收与自检

| 验收项 | 命令 | 期望结果 |
|--------|------|----------|
| 上传端到端可用 | `curl -X POST …/api/v1/infusion/tracks -F frames=@rack_01.zip …` | 返回 `session_id` 与每袋 `area_ratio_series` |
| 保留期清理：过期的清、没到期的留 | `PYTHONPATH=src:. python3 var/verify_task44.py` | 真进程跑清理脚本：过期帧目录被删且读数仍在库里；保留期内、未完成、有写回欠账的会话帧目录原样保留并带原因码 |
| 保留期清理：dry-run 不动手 | `python3 scripts/purge_expired_sessions.py --dry-run` | 报出会清哪几袋，但文件、会话状态、归档台账一概不变 |
| 容量测算能算出数 | `python3 scripts/estimate_storage.py --wards 3` | 输出帧目录占用、各表每天/每年增长与换库判据；数字自洽（每天字节 = 每天帧数 × 单帧字节） |
| 硬校验拦截 | 上传帧数不足 / 尺寸过小 / 时间戳逆序的序列 | `422`，detail 含 `field + reason + suggestion` |
| 六类拍摄问题各命中 | `python3 -m pytest tests/test_failure_modes.py -q` | 出框/糊/反光/歪/暗光/滴斗空管逐条命中，`suggestion_zh` 非空且是能照做的动作 |
| 正常拍摄不误判 | 同上（`test_clean_capture_reports_nothing`） | 干净拍法零命中，`capture_quality` 为空且 `capture_degraded=false` |
| 上传前预检：坏片不拒收 | `PYTHONPATH=src:. python3 var/verify_task34.py` | 真 uvicorn + curl 传抖动短片 → HTTP 201，`precheck.degraded=true` 命中 `motion_blur`，质量分 < 1.0，逐袋 `precheck_score` 与整段一致 |
| 上传前预检：好片满分、判不出不猜 | 同上（18/18 项 OK） | 干净短片 `precheck.state="ok"`、零命中、满分 1.0；六袋液面高度不同也不会读出假倾角（干净拍法倾角 < 1°）；画面里量不出液面时倾角子项整体缺席 |
| 拍坏不拒收 | 上传一段手持抖动的短片 | 仍 `201`，曲线与 `fill_ratio` 照给，该袋 `capture_degraded=true` 并带补拍建议 |
| 毫升数带可信区间 | 上传常规液位短片（袋型已登记标定） | `estimated_volume_ml` 出现且落在 `volume_uncertainty.low_ml ~ high_ml` 之间，`gate="ok"`、`confidence ≥ 0.60` |
| 三类低置信各自不出数 | `python3 -m pytest tests/test_uncertainty.py -q` | 小液位 / 多次遮挡 / 几何版本缺失各自不出数，成因码与护士端文案互不相同 |
| 门禁是算出来的不是拍出来的 | 同上（`test_low_confidence_gate_threshold_is_derived_not_chosen`） | 20% 液位（85 ml）放行、5%（29 ml）挡下，交点落在余量约 41 ml（= 灵敏度 580 × 0.010 ÷ 0.14） |
| 区间随条件收紧 | 同上（遮挡占比 / 序列抖动 / 液位三组） | 区间半宽随条件变差单调变宽，抖动叠低液位时跨过门禁 |
| 门禁语义进契约 | `curl …/api/v1/infusion/tracks -F frames=@small_fill.zip` | 响应**无** `estimated_volume_ml` 与 `eta_minutes` 两个键（不是 null），但有 `volume_uncertainty.reason/hint_zh` |
| 暗光不误报成糊 | `test_dark_frames_are_not_reported_as_motion_blur` | 夜间短片只报 `low_light`，建议是「打开灯」而非「把手机靠住」 |
| 无标定门禁 | 查询未标定袋型的会话 | 响应中无 `estimated_volume_ml`，`calibration_state="uncalibrated"` |
| 标定后出数 | `POST /api/v1/infusion/calibrations` 后重传同一段短片 | 两袋各带 `estimated_volume_ml` / `calibration_version` / `eta_source` |
| 滴速缺失降级 | `slots` 里 `rate_text` 写「常规输液」且无遥测 | 毫升数照给，`eta_minutes` 不出现，`pickup_suppressed=true` |
| 模型网关协议形状 | `python3 -m pytest tests/test_model_gateway.py -q` | 请求打在 `{LLM_BASE_URL}/chat/completions`，带 Bearer 与 `response_format={"type":"json_object"}` |
| 四类失败各自分类 | 同上（`test_four_failure_kinds_are_mutually_distinguishable`） | `connect_error`/`read_timeout`/`http_5xx`/`parse_error` 四个 `fallback_reason` 互不相同 |
| 出站请求体键集与协议一致 | `python3 -m pytest tests/contract/test_openai_protocol.py -q` | 请求体**恰好**是 `model`/`messages`/`temperature`/`max_tokens`/`stream`/`response_format` 六个键；`response_format={"type":"json_object"}`，`temperature=0` |
| 四类失败在替身上各自可复现 | 同上 | 读超时（替身延迟 > timeout）、5xx（退避重试 ≤2 次）、解析失败（200 但非 JSON）各走真 socket 复现；连接超时用注入的传输层钉在真实异常类型上 |
| 协议边角不误判 | 同上（`test_endpoint_without_usage_block…` / `test_reply_without_choices…` / `test_null_content…`） | 端点不回 `usage` **不**判失败；`{}`（无 choices）与 `content=null + finish_reason=length` 判 `parse_error`，不当成功收下 |
| 每日上限超限不出网 | `python3 -m pytest tests/test_gateway_quota.py -q` | 第 3 次调用（上限 2）`fallback_reason="quota_exceeded"`、`attempts=0`，替身请求数仍为 2；报错文本点名 `LLM_DAILY_CALL_LIMIT` |
| 单请求上限不静默截断 | 同上（`test_per_request_token_cap_blocks_instead_of_silently_clamping`） | `max_tokens=5000` 越上限 1200 被挡、零出网，报错里同时给出 5000 与 1200；恰好等于 1200 放行 |
| 护栏与降级都不推进状态 | `PYTHONPATH=src:. python3 var/verify_task43.py` | 36/36 OK：真 uvicorn + 真 SQLite 下，会话恒 `monitoring`、`infusion_event` 恒 0 行，上传链路仍 201 |
| 用量与延迟统计对得上 | 同上 | `runtime_metrics` 给出 `invocations=5 / requests_sent=3 / quota_blocked=2`、token 精确到个位、延迟 p50 与 p95 有样本；被挡下的调用 `latency=0` 不进窗口 |
| 额度按院内工作日切 | 同上（`test_business_day_boundary…`） | 23:59+08:00 仍算 10-23，跨零点后额度归零（UTC 日界会落在病区凌晨两点） |
| 摘要口径分得清 | `curl …/api/v1/summary` | `model.runtime_metrics`（本进程）与 `model.total_calls`（落库的审计行）**并列不相加**；`model.quota` 给出 `used_today / remaining_today / exhausted` |
| 模型不可达不拖垮业务 | 端点持续 5xx 后调用网关 | `decision_source="threshold_fallback"`、`pending_manual_review=true`，会话仍 `monitoring`、出站事件表 0 行 |
| 模型无权推进状态 | 让替身返回 `{"decision":"replaced",…}` | 契约拒收并降级阈值路径，`decisions` 枚举内无 `replaced` |
| 仲裁只采信措辞 | `LLM_FUSION_ENABLED=true` 后让替身与阈值给出一致的结论 | 响应 `decision_source="model_decided"`、`arbitration.code="agreed_with_threshold"`；结论与 `trigger_reason` 仍取阈值路径 |
| 冲突即回落 | 同上但让替身判「不用换」 | `decision_source` 如实改写 `threshold_fallback`、`arbitration.code="conflicts_with_threshold"`，审计链落一行 `inference.fusion_rejected` |
| 调过模型也不推进状态 | 上面两条跑完之后查库 | 会话恒 `needs_change`、全库无 `replaced`、`infusion_event` 恒 0 行（未确认的换液永不出站） |
| 标定改版重算 | 同袋型再次 `POST /calibrations` | 历史会话毫升数按新版本更新，`GET /calibrations/recalc` 有改前/改后轨迹 |
| 生效区间唯一 | 登记一个生效时刻早于旧版的标定 | `409 CALIBRATION_WINDOW_CONFLICT`，issue 点名冲突版本与建议；库中仍只有一版 |
| 改大了不静默覆盖 | 登记一个让读数差 15% 以上的标定 | 该袋旧毫升数原样保留 + `calibration_review_required`，告警进 `GET /calibrations/{袋型}/reviews` 的 pending 清单 |
| 复核结案闭环 | `POST /calibrations/{袋型}/reviews/{alert_id}` | `applied` 后新读数落库、告警结案、审计链留下复核人；重复结案 `409` |
| 版本与区间可查 | `GET /api/v1/infusion/calibrations/TYPE-A/versions` | 给出当前生效版本、历史版本与各自 `effective_from`/`effective_to`，到点的预约版本自动转正 |
| 腔体换算比朴素线性更准 | `pytest tests/test_cavity.py -k bias` | 同一液位下四个常用袋型的误差跨度：朴素做法 >10 ml，腔体换算 <2 ml |
| 低液位不虚报 | 同上（`-k convergence` / `-k residual_limited`） | 自校准生效后逐档残差 ≤0.8% 有效容量；扣到只剩残留量级时报 0 并标 `residual_limited` |
| 证据不足不自我修正 | 同上（`-k insufficient` / `-k contradicts`） | 样本 <3 条 → `insufficient_samples` 且 `applied=false`；离散度超限 → `conflicting` 且不生效 |
| 超装不给数 | 同上（`-k overfill`） | 面积比 1.05 → `overfilled=true`、`readable_ml is None`，`reason` 带可照做的重拍动作 |
| 自校准走真实确认路径 | `python3 var/verify_task25.py` | 真 uvicorn + 真 curl 上传 + 真 SQLite：三次确认攒够样本后生效，重放不新增样本，生效前后读数差 = 学到的残差 |
| 幂等确认 | 同一 `client_event_id` 连发两次 transition | 第二次 `idempotent_hit=true`，事件表行数不变 |
| 并发双确认 | 两线程各持一条连接同时确认同一袋 | 事件表恰好 1 行，确认审计恰好 1 条 |
| 覆盖式重传 | 同 `rack+shift+bag_seq` 先传 20 帧再传 12 帧 | 会话行数不变、`session_version` +1、帧目录只剩本次的 12 帧 |
| 越权写入 | 给 B01 的会话确认 `bag_seq=B02` | `403 WRITE_NOT_ALLOWED` + `rejected` 审计，事件表不新增行 |
| 出站写回契约 | `pytest tests/contract/test_writeback_contract.py` | 本地替身收到 `Bearer` 凭据头 + `Idempotency-Key` 同值；请求体**全量 14 个字段**与出站字段表逐一对应，且每个字段都有对端接口文档对应项 |
| 六类失败各自落地 | 同上（含 `var/verify_task38.py`） | 业务拒绝 `failed` 不重试并带对端原因；5xx / 连接超时 / 读超时 / 半开连接 / 连接错误退避 ≤2 次后 `pending_writeback`，且 `kind_counts` 按类别分开计数 |
| 半开连接不再混进「连接失败」 | 同上 | 真实对端掐连接 → `kind_counts={"half_open":3}`、`peer_may_have_applied=true`，重试一次即送达的不留欠账 |
| 契约文档不漂移 | 同上 | `docs/nurse-exec-contract.md` 与代码生成结果逐字一致（含失败分类表与错误摘要格式） |
| 待写回队列：退避与死信 | `python3 -m pytest tests/test_retry_queue.py -q` | 退避表 1s/4s/16s、预算 = 首次 + ≤2 次重试；对端持续 5xx 时替身收到 3 次请求后停手，欠账落 `dead_letter` 且错误摘要里没有凭据 |
| 待写回队列：重放幂等 | 同上（`test_replay_endpoint_is_idempotent_on_a_second_call`） | 二次重放 `idempotent_hit=true`、**一次 HTTP 都不发**，事件表仍 1 行 |
| 待写回队列：端到端复核 | `PYTHONPATH=src:. python3 var/verify_task37.py` | 真 uvicorn + 真替身对端 + 真 SQLite：31/31 项 OK——确认换液后对端 503 → 清单里查得到欠账 → 对端恢复后重放 → 那一行结清、业务事件仍 1 条 |
| 对接端到端演练 | `python3 -m pytest tests/e2e/test_writeback_e2e.py -q` | 上传 → 点名 → 确认 → 替身收到全量字段；未确认时替身收到 **0** 个请求、事件表 0 行 |
| 演练：失败注入 | 同上 | 连续 5xx → 退避 **1.0s / 4.0s** 两次后落 `pending_writeback`，事件一行不少；4xx 一次都不重试、落 `failed` 并抽出对端原因；半开连接退避重试成功且两次带**同一个**幂等键 |
| 演练：摘要与清单同步 | 同上 | 5xx 耗尽后 `GET /api/v1/summary` 的 `pending_writeback_count=1`、`queue_by_status.dead_letter=1`，`GET /api/v1/infusion/pending` 列出的那一行 `dead_letter_reason=retries_exhausted`、`attempt_count=3` |
| 演练：凭据不出门 | 同上（`test_drill_writes_no_credential_into_any_observable_surface`） | 跑完整条链路后，摘要 / 待写回清单 / 事件表 / 审计行里都搜不到明文 Bearer token |
| 演练：独立复核（不采信 pytest） | `PYTHONPATH=src:. python3 var/verify_task40.py` | 真 uvicorn 进程 + 真 curl 上传 + 真 TCP 替身：43/43 项 OK，退避**真睡** 5.04 秒（1s + 4s），四次失败注入与人工重放逐条核对 |
| 审计七类动作齐全 | `python3 -m pytest tests/test_audit_chain.py -q` | 认领/标定确认/点名/护士确认/写回成功/写回失败/写回挂起七类均由**真实产品路径**落出行来，无一手工 append |
| 审计顺序单调 | 同上（`test_query_order_is_monotonic_and_stable_across_repeats`） | 同一批导两次行序逐行一致；同刻的两行按 `audit_id` 定序，不随库返回顺序漂 |
| 审计过滤与 CSV 可留档 | 同上（`test_filters_return_exactly_the_matching_rows` / `test_csv_is_excel_readable…`） | 按架/床/时间窗/类别/结果逐条对得上；CSV 带 UTF-8 BOM、CRLF 行尾、中文表头，Excel 双击能开 |
| 审计缺失即业务回滚 | 同上（`test_state_transition_rolls_back_when_the_audit_write_fails`） | 触发器让审计写失败 → 点名整体失败，会话仍 `monitoring`，不落 `state.change_proposed` |
| 审计包可验真伪 | `curl -OJ '…/api/v1/infusion/audit?rack_id=R-07'` | 响应头 `X-Audit-Row-Count` / `X-Audit-Chain-Digest` 与 manifest 一致；删掉任一行后摘要改变 |
| 未确认不出站 | 会话停在 `needs_change` 时触发写回 | 一次 HTTP 都不发，事件保持 `pending` + `writeback.refused` 审计 |
| 遮挡后接回同一袋 | `pytest tests/test_memory_bank.py -q` | 遮挡 3 帧后恢复点判 `reappeared`，`bag_seq` 不变、空档仍留 `null` |
| 尖峰被吸收、原值不改 | `pytest tests/test_smoothing.py -q`（`test_glare_spike_is_absorbed_and_flagged`） | `raw[2]==0.880` 照留，可信序列在该帧贴着真实缓降走 |
| 反常帧只降级不消失 | 同上（`test_trusted_series_never_rises_even_when_observations_do`） | 观测升到 0.55 时 `outlier_frames` 点名且 `smoothed` 保留 0.55，`trusted` 全程不上升 |
| 缺帧不插值、补了有据 | 同上（`test_gap_inside_the_window_is_reconstructed_and_marked` / `test_gap_longer_than_max_is_refused`） | 3 帧缺帧补值并标 `reconstructed`；12 帧缺帧保持 `null` 且原因码为 `gap_longer_than_max` |
| 平滑参数可整组配置 | `SMOOTHING_MAX_GAP_FRAMES=1` 重启后重传同一段遮挡短片 | `smoothing.params.max_gap_frames=1`，该段不再补线（`gap_longer_than_max`） |
| 挡太久后再找回 | `python3 var/verify_task17.py` | 遮挡 10 帧 → 交出轨迹 → 第 13 帧（交出帧 + 冷却 2 帧）接回同一 `bag_seq`，接回帧与干净跑法同值（实测差 0.0000），遮挡段与冷却段仍留 `null` |
| 新袋不得接到旧曲线 | 同上（第 3 组） | 换液后新袋更满、药水颜色不同 → 曲线全程留 `null`，保持 `lost` 等人工认领 |
| 6 袋一次拍完分得清 | `python3 var/verify_task18.py` | 6 袋全部 `slot_state=assigned` 且各配自己挂点、6 条曲线互不相同、逐帧预测次数 = 帧数 × 6 |
| 掩膜按 id 归位 | `pytest tests/test_multi_object.py -k id_not_by_visit_order` | 起锚顺序整个倒过来后，每个 `bag_seq` 的曲线逐点不变 |
| 遮挡不牵连别袋 | 同上（`-k not_degrade_other_bags`） | 全程遮挡那袋 `no_evidence` 且不占挂点、该挂点 `unfilled`；另外两袋仍 `assigned` 且余量读数不退化成满袋 1.0 |
| 挂点对不上不自动改绑 | 同上（`-k declared_slot_mismatch`） | `slot_state=mismatched` + `claim_required=true`，`slot_detail` 同时点出声明挂点与实际停在哪一列 |
| 中途塞新袋被拒 | 同上（`-k object_id_set_is_frozen`） | 起锚后再加新 `obj_id` 抛 `ObjectIdSetFrozen`；同 id 改绑仍允许 |
| 再获取路径可关 | `pytest tests/test_reacquire.py -q` | 传 `reacquire=None` 时整条路径关闭，交出后曲线全程不复活；无遮挡短片零触发 |
| 记忆定长且会衰减 | `python3 var/verify_task16.py` | 30 帧与 200 帧的库大小同为 4 条、掩膜字节数相同；衰减到门限以下时返回「无依据」 |
| 外观描述子三档位 | `python3 -m pytest tests/test_descriptor.py -q` | 带印刷笔画 → `full`；空膜 → `full` + `present=false`；整段偏暗 → `color_only`（**不是**「没有标签」）；区域太小 → `geometry_only` 且不自动接上 |
| 换袋必低置信 | `python3 var/verify_task29.py` | 同挂点换新袋 → `unbound/appearance_changed`、`needs_claim`、响应**无** `previous_session_id`；同架没换袋的那只仍 `bound` |
| 液柱升高一票否决 | 同上（第 4 组） | 颜色完全相同、袋子比上一轮满 → 仍判 `unbound/appearance_changed`，`reanchor_driver="level"` |
| 缺证据不当 0 分 | 同上（第 6 组） | 旧描述子（无标签字段）读回为 `color_only`，标签项退出加权，`total=1.0`（不是按缺满分算） |
| 扫错地方不污染余量 | 同上（第 4 组） | 与记忆矛盾的观测判遮挡、`area_ratio` 为 `null`、不进库但留档；下一帧本袋自己接回 |
| 退避与待写回 | 替身持续返回 503 | 3 次尝试、退避 1s/4s，事件置 `pending_writeback` 且事件表不丢行 |
| 遥测入站 | `pytest tests/test_mqtt_telemetry.py` | 主题解析出架号、坏消息只拒一条、重复遥测不二次消费；无遥测时 ETA 仍由医嘱速率给出 |
| 断线重连按退避表走 | `python3 -m pytest tests/test_mqtt_resilience.py -q`（`-k backoff or reconnect or flapping`） | 退避序列 1/2/4/8/16/32s 封顶；退避窗口内**不再敲 broker**；「调用返回了」不等于「连上了」（`issued` 与 `connected` 分开报）；连稳 300s 后才清零；试够次数报 `attempts_exhausted`；每次尝试都落审计 |
| 换架之后重投不重复消费 | 同上（`-k redelivery or dedup`） | 同一份遥测换个输液架再进来仍被 `(device_sn, ts)` 认出来；MQTT 与 webhook 双通道送来的同一份同样只消费一次；泵固件不发 `ts` 时按载荷摘要兜底；乱序到达不覆盖更新的读数 |
| webhook 兜底通道 | 同上（`-k webhook`） | 未配密钥 → 404 且端点不存在；验签在解析之前（不通过的请求体不进缓存）；时间戳在签名串里（换掉就失效）+ ±300s 重放窗口；跨架 `rack_mismatch` 整条拒收；重复投递回 202 + `duplicate=true` 而非错误 |
| 遥测没了不影响主路径 | 同上（`-k telemetry or zero_telemetry`） | 一条遥测都没有时：余量曲线照常、`eta_source=order_text` 且 `eta_minutes` 照给、`telemetry_state="absent"`；通道整个断开时探活仍 200、摘要仍可读、上传仍 201 |
| 异线程审计不打死订阅 | 同上（`-k audit_from_another_thread or pending_audit`） | 从另一个线程调 broker 回调与重连：**不抛异常、不直接写库**，只入队；探活/摘要请求把队列落库并如实报条数；缓冲上限 512 条，满了丢最老并计数（`pending_audit.dropped`） |
| 非法迁移 | `monitoring → replaced` 直跳 | `409` + 审计行 |
| 出站退避 | 对端持续 5xx | 退避 ≤2 次后置 `pending_writeback`，不丢事件 |
| 模型降级 | 网关不可达 | `decision_source="threshold_fallback"`，状态未被推进 |
| 凭据自检 | `python scripts/check_config.py` | 全空环境退出码 1 并逐组点名缺哪个变量名；7 组齐全退出码 0 |
| 运行摘要 | `curl -sS .../api/v1/summary` | 待换液袋数、待写回积压、模型降级次数、遥测通道状态四个数都在 |
| 请求可追溯 | 上传带 `X-Request-Id` | 响应头回写同一 id，业务日志行与审计行都挂这个 id |
| 凭据不进日志 | 检索日志全文 | 三个密钥的明文均不可见（摘要、启动行、错误摘要同样只给掩码） |
| 错误码表不漂移 | `pytest tests/test_summary.py` | `docs/error-codes.md` 与代码重新生成的逐字一致 |
| 断点续传与重放 | `PYTHONPATH=src:. python3 var/verify_task32.py` | 20/20 OK：真 uvicorn + 真 SQLite + 真文件，缺帧提交 422 但已收分片保留、重发同一片 0 新增 12 去重且磁盘字节未变、顶号异内容 422、整段重放版本 1→2 且事件表不新增、清理只回收 `_staging` |
| 换实现不破协议 | `pytest tests/contract/test_predictor_contract.py -q` | 45 passed；几何核 8/8 全绿，权重档位如实 skip 并写明缺什么；反例对照证明契约抓得住破协议的实现 |
| 主路径掩膜可落盘 | `python3 var/verify_task24.py` | 真上传短片上驱动面积曲线的那张掩膜 PNG 往返逐位无损、二次编码字节一致；两袋面积比曲线与主干基线逐位一致 |
| 取评测基础不整包下载 | `python scripts/fetch_vos_subset.py --dry-run` | 只读中央目录即报出可选规模，帧负载字节下发为 0 |
| 取数幂等可重跑 | `python scripts/fetch_vos_subset.py --relax-min-frames`（连跑两次） | 首次 1892 个文件 / 77.7s；第二次新取 0、复用 1892、4.6s |
| 坏数据不冒充真数据 | `pytest tests/test_fetch_subset.py -q` | 注入坏字节 → `skipped` + 坏文件不进正式路径；断网 → 磁盘只剩 `manifest.json` |
| 规模门禁不偷换 | 同上（`-k shortfall`） | 要求每序列 150 帧、语料最长 104 帧 → 缺口写进 manifest 与文档顶部，不静默降门槛 |
| 误报被挡住 | `pytest tests/test_change_detector.py -q` | 39 passed：单帧毛刺不触发、滞回带按轮次消解、15 分钟内不重复点名、每轮 3 袋上限按 ETA 截断且留名次 |
| 真实链路复核 | `PYTHONPATH=src:. python3 var/verify_task30.py` | 19/19 OK：真 uvicorn + 真 SQLite，四袋见底只点 3 袋、第四袋留名次，下一轮不重复点名，出站事件表始终为 0 行 |
| 点名清单只读 | `PYTHONPATH=src:. python3 var/verify_task31.py` | 51/51 OK：查完清单替身收到 0 个请求、事件表 0 行；清单按 ETA 升序、只含真被点名的袋 |
| 确认一次换液 = 一条事件 | 同上 | 首次 `created=true` + 替身收到 1 条带 Bearer 的 `POST /api/v1/infusion/events`；重放 `idempotent_hit=true`、`event_id` 与 `confirmed_at` 与首次逐字相同、替身仍只收 1 条 |
| 终端自编幂等键被拒 | `pytest tests/test_pickup_payload.py -q`（37 passed） | `422 client_event_id_mismatch` 且事件表不新增行；对端持续 5xx 时确认仍成功、事件置 `pending_writeback` 等重放 |
| 测试通过 | `python3 -m pytest` | 全部用例通过（Phase 4 后续 task 补齐实际数量） |

