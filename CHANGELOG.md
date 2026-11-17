# 变更日志

本文件由沌构机（ChaosForge）在每轮交付时自动生成与维护，**不需要人工编辑**。条目按交付顺序排列，面向「这个项目现在能做什么」。版本块最新的在最上方。

## [0.1.0] - 2026-10-24

**本轮目标**：护士对着输液架拍一段 10–20 秒短片上传，得到每袋的余量曲线、还能撑多久的估时，以及"该换液了"的点名；确认换液后自动把换液事件写回护理执行平台

| 给谁用 | 什么时候用 |
|--------|------------|
| 病区护士 / 助理护士 | 每 1–2 小时一轮输液巡视，站在输液架前拍 10–20 秒短片 |

### 跑通主干

- **项目子目录脚手架与中文 README 完整骨架**
  创建 infusion-volume-monitor/ 项目子目录（与 task.json 顶层 project 逐字一致），复制二维码、建 .gitignore、依赖与配置样例、docs/concepts 与 docs/screenshots，并 Write 完整章节骨架的中文 README（含关注区块）。所有项目文件一律落在子目录内，禁止在 round 根目录创建 src/、README.md、requirements.txt 等
  - 证据：test -f qrcode.jpg && grep -q '## 关注我们' README.md && grep -q 'qrcode.jpg' README.md 通过；README 基线实测 12870 字符 / 13 个 ## / 18 个 ### / 59 行表格 / 22 个代码块 / 1 张 ASCII 架构图（≥5000/10/8/5/8/1 全部达标），341 行（≤350）；…

### 新增能力

- **配置面（7 组凭据位）与 SQLite 存储 schema**
  实现 pydantic-settings 配置面（凭据一律读环境变量，缺失即启动失败并列出缺哪个，不做静默默认值）与 SQLite schema（袋级 session / 出站事件 / 推理审计三表 + 索引 + 幂等迁移）。这是出站真实调用与凭据位的落地前提
  - 证据：python3 -m pytest tests/test_config.py tests/test_store.py -q → 28 passed（0 failed）。覆盖：7 组凭据位登记齐全（test_credential_groups_cover_seven_slots）；全空环境下 get_settings() 抛 MissingCredentialError 且报错文本点名缺 MQTT…
- **M1 帧序列接入与主干最窄端到端路径**
  实现入站 multipart 帧序列接入（解包、校验、落库），并让前 3 个任务结束时能用一条 curl 命令上传帧序列并拿到可核对 JSON 响应——主干第一次端到端产出
  - 证据：python3 -m pytest tests/test_ingest_validate.py -q → 35 passed；全量 python3 -m pytest -q → 63 passed（11 config + 17 store + 35 ingest，0 failed，见 logs/task3-pytest-full.log）。主干最窄端到端已跑通：真实 uvicorn（127.0.…
- **M2 主路径内核：预测器协议 + 确定性几何核 + 逐帧传播**
  落地融合后产品的第一个可用能力：一次上传得到每袋跨帧一致的 area_ratio_series（面积比曲线）。可插拔 PerPointMaskPredictor 协议 + 确定性几何核实现，对应源主路径 init_state → add_new_points_or_box → propagate_in_video 的传播语义
  - 证据：python3 -m pytest tests/test_propagate.py -q → 20 passed（协议 4 / 身份 3 / 几何容差 5 / 碎片语义 3 / 接口 4 / 工具 2，0 failed）；全量 python3 -m pytest → 83 passed（63→83，0 failed，日志 logs/task4-pytest-full.log）。真 uvicorn…
- **M2 首帧无提示枚举管线与源实现对照测试**
  把首帧「不提示也把画面里的袋子全切出来」这条枚举管线写进产品：护士在列表里认领而不是逐袋点像素。实现须与源仓库枚举管线的算法语义对齐，并用对照测试守住关键行为
  - 证据：python3 -m pytest tests/test_enumerate.py -q → 15 passed（召回 2 / 源行为对照 4 / 低分剔除 1 / 聚类绑定 3 / 网格 1 / 接口 1 / 孔洞回归 1 + 稳定分 1，0 failed）；全量 python3 -m pytest → 98 passed（83→98，0 failed）。源行为对照：batched_nms 与…
- **M2 遮挡标记/恢复与长序列恒定算力**
  遮挡不丢弃：遮挡帧显式标记、恢复后接回同一对象（记忆特征续接语义对齐 github_ref/sam2/sam2/modeling/memory_attention.py 的 MemoryAttentionLayer._forward_ca，读上一帧记忆特征）；并量化数百至上千帧序列下的恒定算力。公开子集下载与基准对照分别由任务 21 / 22 承接
  - 证据：python3 -m pytest tests/test_occlusion.py -q → 22 passed（0 failed）；全量 python3 -m pytest → 120 passed（98→120，0 failed，日志 logs/task6-pytest-full.log，本轮触达共享底座 core/propagate.py 与对外契约 BagSlotOut 故跑全量）。容量…
- **M3 袋型标定、毫升换算与 uncalibrated 门禁**
  把掩膜变成可核对的毫升数：袋型标定为前置条件，无标定时强制 uncalibrated 且响应中不得出现 estimated_volume_ml，避免用不可信的数字驱动换液提醒
  - 证据：python3 -m pytest tests/test_calibration.py -q → 34 passed（0 failed）；全量 python3 -m pytest → 154 passed（120→154，0 failed，日志 logs/task7-pytest-full.log；本轮触达共享底座 store/schema+migrations+repositories、api…
- **M3 合成真值回归与数据脚本**
  自建参数化渲染的输液袋短片（已知真值液位与容量）作为回归基准，覆盖反光、形变、背景花色、夜间暗光与滴斗空管等边界；替代物只作测试数据层，不充当最终对接边界
  - 证据：python3 -m pytest tests/test_calibration_regression.py -q → 35 passed（0 failed，2.6s）；全量 python3 -m pytest -q → 189 passed（154→189，0 failed，12.9s，日志 logs/task8-pytest-full.log）。三层回归各自归因并通过：①几何层 test_a…
- **M4 跨会话重锚、槽位认领与状态机**
  同一输液架下一轮巡视时把同一袋接上；同槽位多候选或认领不到时拒绝猜测，标 unbound 要求护士认领。状态所有者是护理执行平台，本服务只算 needs_change，replaced 只能由护士确认
  - 证据：python3 -m pytest tests/test_reanchor.py tests/test_state_machine.py -q → 36 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 224 passed（189→224，0 failed，日志 logs/task9-pytest-full.log；本轮触达…
- **M4 幂等键、并发重放与越权边界**
  同 client_event_id 重复确认不产生第二条出站事件；同 rack+shift+bag_seq 重复上传覆盖会话而非追加；并发重放与越权写入被显式拒绝并记审计
  - 证据：python3 -m pytest tests/test_idempotency.py → 19 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 243 passed（224→243，0 failed，19.44s，日志 logs/task10-pytest-full.log；本轮触达共享底座 store/repositor…
- **M5 对接层：出站 HTTP 写回与 MQTT 遥测入站**
  本轮落生产强制项：出站 HTTP 客户端真实调用 + Bearer 凭据 + 字段规范落盘 + 指数退避重试 + pending_writeback 告警；入站 MQTT 订阅泵速遥测，断线时视觉路径独立可用
  - 证据：全量 python3 -m pytest → 292 passed（289→292，0 failed，27.1s，日志 logs/task11-pytest-full.log）；python3 var/verify_task11.py 退出码 0，四条主路径主张独立复核通过（日志 logs/task11-verify-independent.log，真 socket 替身 + 真 SQLite，…
- **M6 单一 ModelGateway（OpenAI-compatible）与降级**
  ai_role=core 的落点：业务层只调单一可替换网关，配置化 base_url/api_key/model/timeout，结构化输出强校验，四类错误分开记，退确定性阈值路径且绝不推进业务状态
  - 证据：python3 -m pytest tests/test_model_gateway.py -q → 45 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 337 passed（292→337，0 failed，40.7s，日志 logs/task12-pytest-full.log；本轮新增 ModelGateway 并触…
- **可观测性、运行摘要与病区侧部署**
  结构化日志贯穿 request_id、错误码表与运行摘要（会话数/待换液数/pending_writeback 数/fallback 次数），加 Dockerfile 与院内安装说明，让信息科能在病区侧上机运行
  - 证据：python3 -m pytest tests/test_summary.py → 37 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 374 passed（337→374，0 failed，41.6s，日志 logs/task13-pytest-full.log；本轮触达共享底座 api/service.py 应用工厂、…
- **M2 掩膜几何后处理：连通域、孔洞填充与孤岛剔除**
  把几何核的形态学后处理拆成独立可测模块：连通域分析、孔洞填充阈值、孤岛剔除阈值与形态特征提取。两个阈值的语义对齐 github_ref/sam2/sam2/sam2_image_predictor.py:484-511 的 max_hole_area / max_sprinkle_area，并覆盖反光碎斑、袋体形变粘连两类失败用例
  - 证据：python3 -m pytest tests/test_postprocess.py -q → 29 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 403 passed（374→403，0 failed，45.5s，日志 logs/task14-pytest-full.log；本轮触达共享底座 core/geom_ker…
- **M2 帧间关联与轨迹 ID 管理**
  把逐帧掩膜串成跨帧一致的轨迹：IoU 与质心双判据关联、轨迹 ID 分配与终止、一致性评分。对应源主路径 add_new_points_or_box → propagate_in_video 的语义（github_ref/sam2/tools/vos_inference.py:137,208,224）
  - 证据：python3 -m pytest tests/test_tracking.py -q → 17 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 420 passed（403→420，0 failed，46.2s，日志 logs/task15-pytest-full.log；本轮新增模块并改动共享底座 core/__init…
- **M2 记忆特征库与相似度加权**
  把「记住上一帧」做成可测模块：每条轨迹维护 top-k 记忆帧、按相似度加权贡献、记忆衰减与置信融合。语义对齐 github_ref/sam2/sam2/modeling/memory_attention.py 的 MemoryAttentionLayer._forward_ca（读上一帧记忆特征参与当前帧推理）
  - 证据：python3 -m pytest tests/test_memory_bank.py -q → 22 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 442 passed（420→442，0 failed，53.6s，日志 logs/task16-pytest-full.log；本轮动共享底座 core/occlusion…
- **M2 轨迹丢失后的再获取与局部重初始化**
  遮挡超过丢失窗口后（任务 6 置 lost），要能在下一段帧里把这袋重新找回来，而不是从零新建一条轨迹与旧曲线接错
  - 证据：python3 -m pytest tests/test_reacquire.py -q → 23 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 465 passed（442→465，0 failed，56.5s，日志 logs/task17-pytest-full.log、logs/task17-junit.xml；本轮…
- **M2 多袋并行传播与槽位-轨迹分配**
  单架 ≤6 袋并行传播之后必须回答「哪条轨迹是哪个槽位」。分配歧义时拒绝猜测，交人工认领
  - 证据：python3 -m pytest tests/test_multi_object.py → 22 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 487 passed（465→487，0 failed，61.6s，日志 logs/task18-pytest-full.log；本轮动共享底座 core/propagate.p…
- **M2 面积序列平滑、单调性约束与缺帧重建**
  原始面积比抖动会直接污染毫升与 ETA。实现中值+一阶平滑、输液物理单调不增约束，以及遮挡期的缺帧占位与受限重建
  - 证据：python3 -m pytest tests/test_smoothing.py -q → 40 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest -q → 527 passed（487→527，0 failed，日志 logs/task19-pytest-full.log；本轮触达共享底座 config.py、api/sche…
- **M2 鲁棒性用例集与失败模式分类（补拍建议）**
  把「拍不好」变成护士能照做的下一步：按反光过曝/暗光/运动模糊/角度不正/袋体出框/滴斗空管分类，每类给出具体补拍建议，而不是报错或静默给出不可信结果
  - 证据：python3 -m pytest tests/test_failure_modes.py -q → 22 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest 退出码 0、收集 549 条（527→549，0 failed，日志 logs/task20-pytest-full.log / .txt；本轮触达共享底座 store/sc…
- **M2 公开 VOS 子集数据获取脚本（可重跑 · 断网降级）**
  真实院内短片本轮不可得，用公开 VOS 数据集子集提供可复现的评测基础：脚本化下载、完整性校验、幂等重跑，断网时降级为 skipped 并如实标注，绝不伪造数据。数据规模写进验收
  - 证据：python3 -m pytest tests/test_fetch_subset.py → 18 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 567 passed（549→567，0 failed，77.4s，日志 logs/task21-pytest-full.log）。【真实公开归档实连，非替身】python sc…
- **M2 基准：IoU / J&F 计算器与阈值单帧基线对照**
  核心深度 (b)：在同一公开子集上，把逐帧传播方案与『阈值单帧』基础方案跑出可复现的对照结果，指标口径对齐 github_ref/sam2/sam2/benchmark.py 的评测方式，结果写入 docs/benchmark-vos.md
  - 证据：python3 -m pytest tests/test_benchmark_metrics.py → 20 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 587 passed（567→587，0 failed，89.0s，日志 logs/task22-pytest-full.log；本任务触达共享底座 core/prop…
- **M2 生产配置加载器与预测器注册表**
  把源产品 build_sam 的 model_id → config/checkpoint 映射（github_ref/sam2/sam2/build_sam.py:424-455）移植成 env 化配置加载与可插拔注册表，让生产权重与确定性几何核共用同一协议、同一入口
  - 证据：python3 -m pytest tests/test_predictor_registry.py → 31 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 618 passed（587→618，0 failed，88.9s，日志 logs/task23-pytest-full.log；本轮触达共享底座 config.py…
- **M2 预测器契约测试套件（两个实现跑同一协议）**
  保证生产权重路径与确定性几何核在协议层完全一致：同一组契约断言同时跑两个实现，任何一方破坏协议（输出 dtype/shape/坐标系/空输入行为）立刻失败
  - 证据：python3 -m pytest tests/contract/test_predictor_contract.py → 45 passed / 10 skipped（0 failed，skip 全是 weights 档位且带真实原因）；全量 env -u DATABASE_URL python3 -m pytest → 663 passed（618→663，0 failed，90.8s，日志…
- **M3 袋型腔体几何模型与自校准**
  把「面积比 → 毫升」从朴素线性外推升级为按袋型腔体几何换算，并支持用护士已确认的换液事件反推自校准项——这是把不可信数字变成可核对数字的关键
  - 证据：python3 -m pytest tests/test_cavity.py -q → 33 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 696 passed / 10 skipped（0 failed，92.9s，日志 logs/task25-pytest-full.log；本轮新增 calibration/cavit…
- **M3 ETA 引擎：医嘱速率解析与遥测交叉校验**
  「还能撑多久」是护士最想拿到的结果：由输注医嘱速率与泵速遥测共同推 ETA，两者冲突或缺失时显式降级，不给猜出来的时点
  - 证据：python3 -m pytest tests/test_rate_parser.py -q → 34 passed；python3 -m pytest tests/test_eta.py -q → 23 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 754 passed / 10 skipped（707→754，0 fa…
- **M3 标定版本管理与历史重算**
  标定记录会改版：版本化 + 历史会话重算 + 审计轨迹 + 偏差告警，避免同一袋在不同时间得到两个不可比的毫升数
  - 证据：python3 -m pytest tests/test_calibration_versions.py -q → 23 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 788 passed / 13 skipped（754→788，0 failed；13 skip 全部是 weights 档位与缺公开语料的既有跳过，非本轮…
- **M3 体积不确定度与置信门禁**
  毫升数要可信才敢拿去点名换液：给 estimated_volume_ml 配不确定度与置信区间，低于门禁一律不出数——与 uncalibrated 门禁互补但原因不同
  - 证据：python3 -m pytest tests/test_uncertainty.py -q → 20 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 808 passed / 10 skipped / 0 failed（788→808，JUnit logs/task28-junit.xml，日志 logs/task28-p…
- **M4 外观描述子与跨会话重锚打分**
  下一轮巡视要认出「还是那一袋」：槽位几何 + 外观描述子（颜色分布、液位特征、标签区存在性）联合打分并给出置信度，任务 9 的重锚在此获得可测的打分实现
  - 证据：python3 -m pytest tests/test_descriptor.py tests/test_reanchor_score.py -q → 23 passed（描述子 10 + 打分 13，0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 820 passed / 10 skipped（0 failed，95.2s，日志 lo…
- **M4 needs_change 判定引擎与误报抑制**
  误报一次就耗尽护士的信任：连续 N 帧 + 阈值余量双条件、滞回带去抖、每袋最小点名间隔与每轮点名上限，全部只产出建议不推进状态
  - 证据：python3 -m pytest tests/test_change_detector.py → 39 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 859 passed / 10 skipped（820→859，0 failed，96.4s，日志 logs/task30-pytest-full.log；本轮触达共享底座…
- **M4 换液点名载荷与确认接口幂等键**
  把「该换液了」变成护士点一下就完成的事：点名载荷（2–3 袋、含 trigger_reason 与 eta）与确认入参的幂等键生成规则
  - 证据：python3 -m pytest tests/test_pickup_payload.py -q → 37 passed（0 failed；排序 7 / 字段完整 7 / 降级文案 7 / 只读清单 6 / 确认幂等 10）；全量 env -u DATABASE_URL python3 -m pytest → 906 tests / 0 failures / 0 errors / 10 ski…
- **M4 会话重放、断点续跑与重复帧去重**
  病区网络不稳，上传中断要能续：分片续传、重复帧去重、整段重放不产生重复业务状态
  - 证据：python3 -m pytest tests/test_resume.py -q → 19 passed（续传补齐 6 / 重复帧去重 4 / 重放不双写 3 / 半成品清理 4 / upload_id 白名单 1 / 仓储层事务直测 1，0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 919 passed / 10 skipped /…
- **M4 审计链（append-only）与审计包导出**
  护理质控与飞检语境要能回答「谁在什么时候认领/确认/触发点名/写回成功或失败」：append-only 审计链 + 可导出审计包
  - 证据：python3 -m pytest tests/test_audit_chain.py -q → 23 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 942 passed / 10 skipped / 0 failed（119.27s，日志 logs/task33-pytest-full.log；本轮触达共享底座 stor…
- **M1 帧质量预检与补拍建议**
  拍摄质量决定后面所有数字的可信度：模糊、过曝、过暗、分辨率、角度预检给分与建议，但不因质量差直接拒绝上传
  - 证据：python3 -m pytest tests/test_quality.py -q → 27 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 969 passed / 10 skipped / 0 failed（942→969，123.35s，日志 logs/task34-pytest-full.log；本轮触达共享底座…
- **M1 缩放、ROI 与流式上传**
  手机上传 1200 帧不能一次性读进内存：长边限制、ROI 裁剪与流式 multipart，峰值内存受控（对应深度预算 (c) 的真实规模输入）
  - 证据：python3 -m pytest tests/test_resize.py → 38 passed（0 failed，11.3s）；全量 env -u DATABASE_URL python3 -m pytest → 1009 passed / 10 skipped / 0 failed（135.80s，969→1009，日志 logs/task35-pytest-full.log；本轮触达共…
- **M1 入站错误契约与 OpenAPI 响应模型**
  院内系统挂接看的是契约：显式 Pydantic 响应模型、结构化 422 detail（字段 + 原因 + 建议）与稳定错误码表，并与 /docs 的 OpenAPI 描述一致
  - 证据：python3 -m pytest tests/test_openapi_schema.py → 22 passed（必填字段 6 / 错误结构 11 / OpenAPI 可生成 5，0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 1031 passed / 10 skipped / 0 failed（1009→1031，137.4s，日…
- **M5 写回重试队列持久化、死信与重放端点**
  出站失败不能丢：队列落 SQLite、指数退避、超次进死信并可由信息科重放，这是「绝不静默丢事件」的落地形态
  - 证据：python3 -m pytest tests/test_retry_queue.py -q → 27 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 1059 passed / 10 skipped / 0 failed（1031→1059，日志 logs/task37-pytest-full.log，149s）；独立复核…
- **M5 出站契约扩充与失败分类**
  对端各种失败长得不一样：4xx 业务拒绝、5xx、连接超时、读超时、半开连接要分开处理，不混成一种盲目重试
  - 证据：python3 -m pytest tests/contract/test_writeback_contract.py → 51 passed（0 failed，原 20 条扩到 51：六类失败各一条 + 传输层异常映射参数化 10 条 + 请求体全量 14 字段断言 + 对应表/枚举/文档同步守卫）；全量 env -u DATABASE_URL python3 -m pytest → 1090…
- **M5 入站 webhook 兜底与 MQTT 重连去重**
  有的院区给 webhook 而不是 MQTT：入站双通道 + MQTT 断线重连 + 遥测去重，且遥测缺失不得阻断视觉主路径
  - 证据：python3 -m pytest tests/test_mqtt_resilience.py → 56 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 1150 passed / 10 skipped / 0 failed（1090→1150，152.7s，日志 logs/task39-pytest-full.log；本轮…
- **M5 对接端到端演练与失败注入**
  用本地替身对端跑完整写回链路并注入失败，证明退避、pending 与告警真的生效。替身只出现在测试数据层，不是最终对接边界
  - 证据：python3 -m pytest tests/e2e/test_writeback_e2e.py -q → 10 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 1160 passed / 10 skipped / 0 failed（1150→1160，162.4s，日志 logs/task40-pytest-full.l…
- **M6 提示词模板注册表与 prompt 版本化**
  模型输入要可复现：模板注册表 + prompt_hash 版本化 + 输出 schema 契约化，改提示词不改业务代码
  - 证据：python3 -m pytest tests/test_prompts.py → 21 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 1181 passed / 10 skipped / 0 failed（1160→1181，164.22s，日志 logs/task41-pytest-full.log；本轮触达共享底座…
- **M6 决策融合仲裁层与 decision_source 三态**
  模型不是裁判：只在确定性阈值与模型结论一致的区间采信，冲突、越界或低置信一律回落阈值路径，并如实标注来源，绝不让模型推进业务状态
  - 证据：python3 -m pytest tests/test_fusion.py → 18 passed（0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 1199 passed / 10 skipped（0 failed，172.7s，日志 logs/task42-pytest-full.log；本轮触达共享底座 store/schema A…
- **M6 网关可观测性、成本护栏与协议兼容性测试**
  模型不能悄悄烧钱或悄悄变慢：token 与延迟统计、每日调用上限、超限自动降级，以及用本地 OpenAI-compatible 替身做协议兼容性测试
  - 证据：python3 -m pytest tests/test_gateway_quota.py tests/contract/test_openai_protocol.py -q → 41 passed（护栏与统计 25 / 协议契约 16，0 failed）；全量 env -u DATABASE_URL python3 -m pytest → 1240 passed / 10 skipped /…
- **存储演进：保留策略、归档与容量测算**
  帧目录与审计表会长大：保留期、归档与容量测算提前算清；SQLite 换 PostgreSQL 只动连接串，不改仓储层
  - 证据：全量 env -u DATABASE_URL python3 -m pytest → 1283 passed / 10 skipped / 0 failed（日志 logs/task44-pytest-full.log；本轮触达共享底座 store/schema.py 表与列、store/migrations.py 迁移顺序、store/db.py 连接与事务、config.py 配置面与 ap…
- **采集真实运行截图到 docs/screenshots/**
  倒数第三个任务。跑真实全流程（上传帧序列 → 面积曲线与毫升/ETA → 护士确认换液 → 出站 HTTP 调用）并采集真实运行截图落 docs/screenshots/，在 README「概念 / 实测对照 → 实测证据」中引用。禁止用概念图或设计稿冒充实测截图
  - 证据：docs/screenshots/ 下 5 张真实 PNG 已落盘（01-upload-and-curves 1352×5084 / 02-pickup-and-transition 1100×1910 / 03-outbound-writeback 776×714 / 04-service-log 1100×622 / 05-candidate-claim 1100×2416），README「…
- **README 补全：填实骨架占位符并过丰度下限**
  倒数第二个任务。把 Task 1 骨架里所有「（Phase 4 后续 task 补齐）」填实（配置表、命令/端点表、错误与降级表、验收与自检表、完整目录树逐行注释、输出样例），并跑基数自检；发布态不得残留占位符
  - 证据：README 丰度自检实测：wc -l=350（≤350）、wc -m=21976（≥5000）、grep -c '^## '=13（≥10）、grep -c '^### '=21（≥8）、grep -c '^|'=52（≥5）、grep -c '^```'=28（≥8）、ASCII 架构/数据流图=1（≥1）、被引用实测截图=6（≥1，docs/screenshots/ 5 张 + 正文 1…

### 收尾与发布

- **发布前清理**
  最后一个任务。清理运行时数据、日志、缓存与临时文件，保证 infusion-volume-monitor/ 可直接作为 GitHub 仓库内容发布；保留源码、文档、配置与必要脱敏样例
  - 证据：发布前清理实测（infusion-volume-monitor/）：①清理后全量 env -u DATABASE_URL python3 -m pytest → 1280 passed / 13 skipped / 0 failed（1293 用例，211.94s，完整输出 round 根 logs/task16-pytest.log，另存 logs/task47-pytest-full.log…

### 交付证据

| 项 | 值 |
|----|----|
| 版本 | 0.1.0 |
| 轮次 | R014 |
| 任务完成 | 47/47 |
| 交付状态 | 全部完成 |
| 综合评分 | 7.4 |
| 发布 | 未发布 |
