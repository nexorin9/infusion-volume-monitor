"""业务表的 DDL 与列定义。

``bag_session``             袋级会话：面积比序列、标定结果与业务状态跨巡视轮次留存。
``infusion_event``          出站换液事件：``client_event_id`` 唯一索引即幂等键，
                            写回状态（pending / sent / failed）落在同一张表。
``inference_audit``         每次模型推理的调用元数据：request_id / model / prompt_hash /
                            latency_ms / decision_source，支撑「谁在什么时候得到什么结论」。
``upload_chunk``            分片续传的半成品上传：开着没提交的那一段帧挂在哪、
                            已收哪些帧号、保留到什么时候（task 32）。
``bag_session_history``     会话被整段重放前的旧版本快照，append-only（task 32）。
``bag_type_calibration``    袋型标定（分段带 + 腔体外形参数）。
``calibration_recalc_log``  标定改版后的历史会话重算轨迹，append-only。
``calibration_deviation``   用护士确认的换液事件反推出来的残留偏差样本，append-only。
``calibration_version_alert`` 标定改版后偏差超阈值、未自动覆盖的告警，append-only。
``audit_log``               审计链。

所有表都**不含患者姓名等可识别明细**：只落床号、袋序、数量与事件。
"""

from __future__ import annotations

__all__ = [
    "SCHEMA_VERSION",
    "BAG_SESSION_COLUMNS",
    "INFUSION_EVENT_COLUMNS",
    "WRITEBACK_QUEUE_COLUMNS",
    "WRITEBACK_QUEUE_STATES",
    "INFERENCE_AUDIT_COLUMNS",
    "BAG_TYPE_CALIBRATION_COLUMNS",
    "CALIBRATION_RECALC_COLUMNS",
    "CALIBRATION_DEVIATION_COLUMNS",
    "CALIBRATION_VERSION_ALERT_COLUMNS",
    "UPLOAD_CHUNK_COLUMNS",
    "BAG_SESSION_HISTORY_COLUMNS",
    "RETENTION_ARCHIVE_COLUMNS",
    "RETENTION_STATES",
    "UPLOAD_STATUSES",
    "AUDIT_LOG_COLUMNS",
    "AUDIT_ACTIONS",
    "CREATE_TABLE_SQL",
    "CREATE_INDEX_SQL",
]

#: 当前 schema 版本；``migrations`` 按此幂等推进。
#: 2 = 新增 ``calibration_deviation`` 表 + ``bag_type_calibration.cavity_json`` 列。
#: 3 = 新增 ``calibration_version_alert`` 表 + 标定/会话的生效区间与偏差复核列（task 27）。
#: 4 = 体积不确定度列（置信度 / 区间 / 门禁），置信门禁不通过时 ``estimated_volume_ml``
#:     写成 NULL——和 ``uncalibrated`` 门禁同一套落库形态（task 28）。
#: 5 = 新增 ``upload_chunk``（分片续传半成品）与 ``bag_session_history``（重放前的旧
#:     版本快照）两表（task 32）。
#: 6 = 新增 ``writeback_queue``（待写回队列：出站欠账的持久台账 + 死信 + 重放凭据，
#:     task 37）。
#: 7 = 新增 ``retention_archive``（帧目录归档台账）+ 会话的帧保留状态列（task 44）。
SCHEMA_VERSION = 7

#: 袋级会话的业务状态（所有者是护理执行平台，本服务只算 needs_change）。
BAG_STATES = ("monitoring", "needs_change", "replaced")
#: 标定状态门禁：无标定时响应中不出现 estimated_volume_ml。后两个是**置信门禁**——
#: 标定在，但这一段的观测撑不起那个毫升数（见 calibration.uncertainty）。
CALIBRATION_STATES = ("uncalibrated", "calibrated", "low_confidence", "high_occlusion")
#: 出站写回状态（任务 11/37 推进）。
WRITEBACK_STATES = ("pending", "sent", "failed", "pending_writeback")

#: 待写回队列的状态（task 37）。它**不是**第二个事件源，而是同一笔出站欠账的台账：
#: ``infusion_event`` 说「这件事发生过没有」，队列说「还欠着对端一次确认没有」。
WRITEBACK_QUEUE_STATES = ("queued", "dead_letter", "sent", "failed")

# ── bag_session ────────────────────────────────────────────────────────────
#: 新增列走 ``migrations.ensure_column`` 幂等补齐，不破坏已有库。
BAG_SESSION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("session_id", "TEXT PRIMARY KEY"),
    ("rack_id", "TEXT NOT NULL"),
    ("bed_no", "TEXT NOT NULL"),
    ("infusion_order_no", "TEXT NOT NULL"),
    ("shift_key", "TEXT NOT NULL"),  # 巡视轮次键（如 2026-10-22-AM），与 rack/bag_seq 共同定位会话
    ("bag_seq", "TEXT NOT NULL"),  # 对端给的袋序标识，产品侧不自行编号
    ("slot_index", "INTEGER NOT NULL DEFAULT 0"),
    ("bag_type_code", "TEXT"),
    ("frames_dir", "TEXT NOT NULL"),
    ("frame_count", "INTEGER NOT NULL DEFAULT 0"),
    ("captured_at", "TEXT NOT NULL"),
    ("area_ratio_series", "TEXT NOT NULL DEFAULT '[]'"),  # JSON 数组：平滑后序列
    ("area_ratio_raw", "TEXT NOT NULL DEFAULT '[]'"),  # JSON 数组：原始观测，保留供核对
    ("occlusion_flags", "TEXT NOT NULL DEFAULT '[]'"),  # JSON 数组：逐帧可见度标记
    ("frame_count_occluded", "INTEGER NOT NULL DEFAULT 0"),
    ("fill_ratio", "REAL"),
    ("estimated_volume_ml", "REAL"),  # 未标定时为 NULL，响应中字段整体不出现
    ("eta_minutes", "INTEGER"),
    ("eta_source", "TEXT"),  # order_text / mqtt_telemetry / conflict / unavailable
    ("rate_ml_per_h", "REAL"),  # 解析出的滴速；标定改版重算 ETA 时复用
    ("rate_source", "TEXT"),  # 与 eta_source 同值：速率的来源优先级结果
    ("rate_reason", "TEXT"),  # 速率退到下一来源 / 给不出来的原因文案
    ("rate_conflict_json", "TEXT"),  # JSON：医嘱与泵速遥测对不上的证据（task 26）
    ("pickup_suppressed", "INTEGER NOT NULL DEFAULT 1"),  # 1 = 抑制该袋换液点名
    ("suppression_reason", "TEXT"),  # uncalibrated / eta_unavailable / rate_conflict / no_observation / overfilled
    ("calibration_state", "TEXT NOT NULL DEFAULT 'uncalibrated'"),
    ("calibration_version", "INTEGER"),
    # ── 改版偏差复核闸门（task 27）───────────────────────────────────────
    ("calibration_review_required", "INTEGER NOT NULL DEFAULT 0"),  # 1 = 新版本读数待复核
    ("calibration_review_deviation", "REAL"),  # 被拦下那次的相对差
    ("calibration_review_version", "INTEGER"),  # 待复核的新版本号
    ("calibration_review_proposed_ml", "REAL"),  # 待复核的新读数（未生效）
    # ── 体积不确定度与置信门禁（task 28）────────────────────────────────
    ("volume_confidence", "REAL"),  # 0–1；没算过不确定度时为 NULL（不是 0）
    ("volume_low_ml", "REAL"),  # 95% 区间下界
    ("volume_high_ml", "REAL"),  # 95% 区间上界
    ("uncertainty_gate", "TEXT"),  # ok / low_confidence / high_occlusion
    ("uncertainty_json", "TEXT"),  # JSON：置信度、成因与逐项证据（护士端文案也在里面）
    ("decision_source", "TEXT"),  # model_decided / threshold_fallback / threshold_only
    ("trigger_reason", "TEXT"),
    ("state", "TEXT NOT NULL DEFAULT 'monitoring'"),
    ("client_event_id", "TEXT"),
    ("session_version", "INTEGER NOT NULL DEFAULT 1"),  # 重放产生新版本，旧版本保留可查
    ("quality_flags", "TEXT NOT NULL DEFAULT '[]'"),
    ("capture_degraded", "INTEGER NOT NULL DEFAULT 0"),  # 1 = 拍坏了（出框/糊/反光/歪/暗光任一命中）
    # ── 上传前帧质量预检（task 34）───────────────────────────────────
    # 序列级结论，逐袋各存一份（同一段短片的每一只袋子共享同一个预检结论）：
    # 「这次拍摄质量分多少、护士端要不要顶提示条」在护士复盘时是按袋查的。
    ("precheck_score", "REAL"),  # 0–1；没跑过预检时为 NULL（不是 0）
    ("precheck_flags", "TEXT NOT NULL DEFAULT '[]'"),
    ("precheck_degraded", "INTEGER NOT NULL DEFAULT 0"),  # 1 = 整段拍得不可信
    # ── 入站缩放与 ROI（task 35）─────────────────────────────────────
    # 响应里的框与面积都是**规范帧**口径；相机原始尺寸与缩放比一并落库，护士端画在
    # 原始照片上时乘 1/scale 即可。改了 FRAME_LONG_EDGE_MAX 之后这两列就能看出
    # 「上一轮和这一轮根本不是同一个坐标系」，不必靠猜。
    ("camera_width", "INTEGER"),  # 相机原始宽度（像素）
    ("camera_height", "INTEGER"),  # 相机原始高度（像素）
    ("canonical_width", "INTEGER"),  # 分析用规范帧宽度（像素）
    ("canonical_height", "INTEGER"),  # 分析用规范帧高度（像素）
    ("frame_scale", "REAL"),  # 缩放比 = 规范长边 / 原始长边
    ("roi_window_json", "TEXT"),  # JSON {x,y,width,height}：末态 ROI 窗（规范帧坐标）
    # ── 跨巡视轮次重锚（task 9）：下一轮还认不认得出这是同一袋 ─────────────
    ("anchor_candidate_id", "TEXT"),  # 本袋据以起锚的候选 ID（认领后可被纠错改写）
    ("anchor_bbox", "TEXT"),  # JSON [x0, y0, x1, y1]：起锚框，原图像素坐标
    ("candidates_json", "TEXT NOT NULL DEFAULT '[]'"),  # 本轮首帧枚举候选（认领时按 ID 回查）
    ("descriptor_json", "TEXT NOT NULL DEFAULT '{}'"),  # 外观描述子（横向中心/袋底/宽度/色相直方图）
    ("reanchor_state", "TEXT"),  # bound（续上同一袋）/ unbound（认不准）/ fresh（新袋）
    ("reanchor_confidence", "REAL"),
    ("reanchor_runner_up", "REAL"),  # 次高分：与最高分的差距过小即判认不准
    ("reanchor_reason", "TEXT"),
    ("previous_session_id", "TEXT"),  # 续上的上一轮会话（可空 = 新袋）
    ("previous_shift_key", "TEXT"),
    ("claim_state", "TEXT NOT NULL DEFAULT 'auto'"),  # auto / claimed / corrected
    ("claimed_by", "TEXT"),
    # ── 帧目录保留状态（task 44）──────────────────────────────────────
    # 帧目录是「像素」，会话行是「读数」：保留期到了清掉像素，**读数留在库里**
    # （护士长复盘「上周三那袋还剩多少」靠的是这一行，不是那堆 jpg）。
    ("frames_retention_state", "TEXT NOT NULL DEFAULT 'kept'"),  # kept / archived / purged / absent
    ("frames_purged_at", "TEXT"),  # 帧目录被归档或删除的时刻
    ("created_at", "TEXT NOT NULL"),
    ("updated_at", "TEXT NOT NULL"),
)

# ── infusion_event ─────────────────────────────────────────────────────────
INFUSION_EVENT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("event_id", "TEXT PRIMARY KEY"),
    ("client_event_id", "TEXT NOT NULL UNIQUE"),  # 幂等键：重复确认不产生第二条事件
    ("session_id", "TEXT"),
    ("event_type", "TEXT NOT NULL DEFAULT 'infusion.bag.changed'"),
    ("rack_id", "TEXT NOT NULL"),
    ("bed_no", "TEXT NOT NULL"),
    ("infusion_order_no", "TEXT NOT NULL"),
    ("bag_seq", "TEXT NOT NULL"),
    ("fill_ratio", "REAL"),
    ("estimated_volume_ml", "REAL"),
    ("eta_minutes", "INTEGER"),
    ("trigger_reason", "TEXT"),
    ("decision_source", "TEXT"),
    ("operator_id", "TEXT NOT NULL"),  # 护士确认人；状态推进的唯一触发源
    ("occurred_at", "TEXT NOT NULL"),
    ("writeback_status", "TEXT NOT NULL DEFAULT 'pending'"),
    ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
    ("next_retry_at", "TEXT"),
    ("last_error", "TEXT"),  # 失败摘要；禁止含凭据
    ("payload_json", "TEXT NOT NULL"),  # 出站字段规范快照，便于对账与重放
    ("created_at", "TEXT NOT NULL"),
    ("updated_at", "TEXT NOT NULL"),
)

# ── writeback_queue ────────────────────────────────────────────────────────
#: 待写回队列（task 37）。一条出站事件在队列里**最多一行**（``event_id`` 唯一）：
#: 队列不产生新事件，只回答「这笔写回还欠着对端吗、欠多久了、谁该去看」。
#:
#: 为什么要这张表而不是只靠 ``infusion_event.writeback_status``：事件表记的是
#: 「最后一次投递的结果」，它不记**这次欠账从什么时候开始、下次什么时候能再试**。
#: 病区断网两小时后进程重启，信息科要回答的是「那段时间有几笔换液没写回去、现在能不能补」
#: ——这个问题的答案必须落在盘上，不能靠进程内存。
WRITEBACK_QUEUE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("queue_id", "TEXT PRIMARY KEY"),
    ("event_id", "TEXT NOT NULL UNIQUE"),  # 与 infusion_event 一一对应
    ("client_event_id", "TEXT NOT NULL"),  # 幂等键：重放时与首次完全相同
    ("session_id", "TEXT"),
    ("rack_id", "TEXT"),
    ("bed_no", "TEXT"),
    ("bag_seq", "TEXT"),
    ("payload_json", "TEXT NOT NULL"),  # 出站载荷快照：重放按**当时**的形态发
    ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),  # 已尝试次数（含首次）
    ("next_retry_at", "TEXT"),  # 下一次可自动重试的时刻
    ("last_error", "TEXT"),  # 最后错误摘要；禁止含凭据
    ("status", "TEXT NOT NULL DEFAULT 'queued'"),  # queued / dead_letter / sent / failed
    ("dead_letter_reason", "TEXT"),  # 死信原因码
    ("last_attempt_at", "TEXT"),
    ("created_at", "TEXT NOT NULL"),
    ("updated_at", "TEXT NOT NULL"),
)

# ── inference_audit ────────────────────────────────────────────────────────
INFERENCE_AUDIT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("audit_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("request_id", "TEXT NOT NULL"),
    ("session_id", "TEXT"),
    ("bag_seq", "TEXT"),
    ("model", "TEXT NOT NULL"),
    ("prompt_hash", "TEXT NOT NULL"),
    ("prompt_version", "TEXT"),
    ("latency_ms", "INTEGER"),
    ("prompt_tokens", "INTEGER"),
    ("completion_tokens", "INTEGER"),
    ("decision_source", "TEXT NOT NULL"),  # model_decided / threshold_fallback / threshold_only
    ("fallback_reason", "TEXT"),  # timeout / read_timeout / http_5xx / parse_error / quota_exceeded
    ("pending_manual_review", "INTEGER NOT NULL DEFAULT 0"),
    ("decision", "TEXT"),  # 模型给出的结论（仅提示，不推进业务状态）
    ("confidence", "REAL"),
    ("created_at", "TEXT NOT NULL"),
)

# ── bag_type_calibration ────────────────────────────────────────────────────
#: 袋型标定记录：**同袋型同时只有一个生效版本**（active=1），历史版本保留可查。
BAG_TYPE_CALIBRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("calibration_id", "TEXT PRIMARY KEY"),
    ("bag_type_code", "TEXT NOT NULL"),
    ("version", "INTEGER NOT NULL"),
    ("capacity_ml", "REAL NOT NULL"),  # 标签标称容量
    ("usable_capacity_ml", "REAL NOT NULL"),  # 有效容量 = 标称 − 死角残留
    ("dead_volume_ml", "REAL NOT NULL DEFAULT 0"),  # 管路与滴斗残留
    ("meniscus_offset_ratio", "REAL NOT NULL DEFAULT 0"),  # 弯月面挂壁补偿
    ("bands_json", "TEXT NOT NULL DEFAULT '[]'"),  # 标定样例拟合出的分段线性带
    ("cavity_json", "TEXT"),  # 腔体外形参数（JSON）；为空则走分段带，见 calibration.cavity
    ("samples_json", "TEXT NOT NULL DEFAULT '[]'"),  # 标定样例原件，供复核
    ("max_relative_error", "REAL NOT NULL DEFAULT 0"),
    ("calibrated_by", "TEXT NOT NULL"),
    ("calibrated_at", "TEXT NOT NULL"),
    ("active", "INTEGER NOT NULL DEFAULT 1"),
    # ── 生效区间（task 27）：左闭右开 [effective_from, effective_to) ────────
    # 同一 bag_type_code 的区间不允许相交；空 effective_to = 仍然生效。
    # 老库无这两列时用 calibrated_at / NULL 兜底，见 calibration.versions.window_of。
    ("effective_from", "TEXT"),
    ("effective_to", "TEXT"),
    ("note", "TEXT"),
    ("created_at", "TEXT NOT NULL"),
)

# ── calibration_recalc_log ──────────────────────────────────────────────────
#: 标定改版后的历史会话重算轨迹：**append-only**，只追加不更新不删除。
CALIBRATION_RECALC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("recalc_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("calibration_id", "TEXT NOT NULL"),
    ("bag_type_code", "TEXT NOT NULL"),
    ("previous_version", "INTEGER"),
    ("new_version", "INTEGER NOT NULL"),
    ("session_id", "TEXT NOT NULL"),
    ("bag_seq", "TEXT"),
    ("rack_id", "TEXT"),
    ("bed_no", "TEXT"),
    ("previous_volume_ml", "REAL"),
    ("new_volume_ml", "REAL"),
    ("delta_ml", "REAL"),
    ("reason", "TEXT NOT NULL"),
    # ── 偏差闸门（task 27）────────────────────────────────────────────────
    ("relative_deviation", "REAL"),  # 相对差；超过阈值时不自动覆盖
    ("review_required", "INTEGER NOT NULL DEFAULT 0"),  # 1 = 本行被闸门拦下，待复核
    ("recalculated_at", "TEXT NOT NULL"),
)

# ── calibration_version_alert ──────────────────────────────────────────────
#: 改版偏差告警：**append-only**。每行 = 「新版会把这一袋从多少毫升改成多少毫升、
#: 相对差多少、因此没自动生效、等谁来复核」。老读数原样保留，所以这张表是唯一
#: 能回答「那天为什么报了 180 毫升、现在为什么还是 180 毫升」的地方。
CALIBRATION_VERSION_ALERT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("alert_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("calibration_id", "TEXT NOT NULL"),
    ("bag_type_code", "TEXT NOT NULL"),
    ("previous_version", "INTEGER"),
    ("new_version", "INTEGER NOT NULL"),
    ("session_id", "TEXT NOT NULL"),
    ("bag_seq", "TEXT"),
    ("rack_id", "TEXT"),
    ("bed_no", "TEXT"),
    ("previous_volume_ml", "REAL"),  # 保留中的旧读数（未被覆盖）
    ("proposed_volume_ml", "REAL"),  # 新版本会给出的数；未生效
    ("delta_ml", "REAL"),
    ("relative_deviation", "REAL NOT NULL"),
    ("threshold", "REAL NOT NULL"),
    ("severity", "TEXT NOT NULL"),  # review / critical
    ("resolution", "TEXT NOT NULL"),  # pending / applied / dismissed
    ("resolved_by", "TEXT"),
    ("resolved_at", "TEXT"),
    ("resolution_note", "TEXT"),
    ("raised_at", "TEXT NOT NULL"),
)

# ── calibration_deviation ──────────────────────────────────────────────────
#: 自校准样本：**append-only**，一条 = 一次「护士确认已换液」留下的残留证据。
#:
#: 为什么单独一张表而不是往标定记录里改数：标定带是**称重/登记**得来的经验值，
#: 自校准是**运行中从确认事件**反推的偏差项，两者来源与责任人不同；把后者混进前者
#: 就再也说不清「这条曲线当初是怎么来的」。这张表只追加不修改，护理部要复核时
#: 逐条能追到是哪一床、哪一袋、第几次巡视确认的。
CALIBRATION_DEVIATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("deviation_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("bag_type_code", "TEXT NOT NULL"),
    ("session_id", "TEXT"),
    ("bag_seq", "TEXT"),
    ("rack_id", "TEXT"),
    ("bed_no", "TEXT"),
    ("source", "TEXT NOT NULL"),  # confirmed_replacement（护士确认换液）
    ("last_fill_ratio", "REAL"),  # 确认前最后一次观测的面积比
    ("model_volume_ml", "REAL"),  # 模型当时给出的毫升数
    ("residual_ml", "REAL NOT NULL"),  # 反推的残留偏差项（真实可读余量按 0 算）
    ("eligible", "INTEGER NOT NULL DEFAULT 0"),  # 是否可作证据（换液前看得很低）
    ("ineligible_reason", "TEXT"),
    ("calibration_version", "INTEGER"),
    # 写入时的累计样本数（同一袋型）。append-only 且仓储层不提供删除，权威值以
    # COUNT(*) 为准；这里冗余一份是为了导出的台账不用再回查一次库。
    ("sample_count", "INTEGER NOT NULL DEFAULT 0"),
    ("note", "TEXT"),
    ("recorded_at", "TEXT NOT NULL"),
)

#: 样本不可用的原因码（写进 ``ineligible_reason``，随行留档）。
DEVIATION_INELIGIBLE_REASONS = (
    "not_near_empty",        # 换液前看得很高：多半是医嘱变更或误点，不是残留证据
    "no_fill_ratio",         # 这一袋没走到体积换算，没有面积比
    "no_model_volume",       # 模型当时没给出毫升数（无标定），没有偏差可学
    "missing_bag_type",      # 袋位没声明袋型
)

# ── upload_chunk（task 32：分片续传与半成品清理）────────────────────────────
#: 一次上传的生命周期：**开着没提交**的半成品留在磁盘上（护士拍到一半网络断了），
#: 这一行就是那半成品的台账——已收哪些帧号、缺哪些、保留到什么时候。
#:
#: 只有 ``pending`` 会被清理策略扫；``completed`` 的暂存目录在提交那一刻就删了，
#: 这一行留作「这个 upload_id 已经变成哪几个 session」的索引。
UPLOAD_CHUNK_COLUMNS: tuple[tuple[str, str], ...] = (
    ("upload_id", "TEXT PRIMARY KEY"),
    ("rack_id", "TEXT NOT NULL"),
    ("bed_no", "TEXT NOT NULL"),
    ("infusion_order_no", "TEXT NOT NULL"),
    ("shift_key", "TEXT NOT NULL"),
    ("captured_at", "TEXT NOT NULL"),
    ("slots_json", "TEXT NOT NULL"),  # 业务头（袋位声明），提交时原样喂回主干
    ("client_event_id", "TEXT"),
    ("session_key", "TEXT NOT NULL"),  # upload_key：提交后帧落到这个目录，与 /tracks 同布局
    ("staging_dir", "TEXT NOT NULL"),  # 半成品帧目录（frames_root/_staging/{upload_id}）
    ("status", "TEXT NOT NULL DEFAULT 'pending'"),  # pending / completed / abandoned
    ("expected_frames", "INTEGER"),  # 终端报的整段帧数；给了才能算缺哪几帧
    ("received_frames", "TEXT NOT NULL DEFAULT '[]'"),  # JSON 数组：已收帧号，升序
    ("frame_times_json", "TEXT NOT NULL DEFAULT '{}'"),  # JSON：帧号 → ISO8601 时间
    ("duplicate_count", "INTEGER NOT NULL DEFAULT 0"),  # 被去重丢掉的重复帧累计数
    ("chunk_count", "INTEGER NOT NULL DEFAULT 0"),  # 收到过的分片数（含全重复的空片）
    ("total_bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("session_ids_json", "TEXT NOT NULL DEFAULT '[]'"),  # 提交后产生的 session_id 列表
    # 收第一片时记下的缩放结论（task 35）：分片路与整段路走**同一套缩放策略**，
    # 因此同一段短片无论怎么传，提交后的几何与数字都一样；这份结论随行落库，
    # 提交时原样带回响应，而不是事后拿规范帧反推「原始大概是多大」（那是猜）。
    ("resize_json", "TEXT"),  # JSON：{camera_width,…,scale,reason}
    ("created_at", "TEXT NOT NULL"),
    ("updated_at", "TEXT NOT NULL"),
    ("expires_at", "TEXT NOT NULL"),  # 保留期到点即由清理策略回收
    ("completed_at", "TEXT"),
)

#: 半成品上传的状态。``abandoned`` 是**清理策略结案**后的终态——不是失败：
#: 病区网络断在第 7 帧时那一段本来就该被丢掉，护士重新拍一段更省事。
UPLOAD_STATUSES = ("pending", "completed", "abandoned")

# ── bag_session_history（task 32：整段重放前的旧版本快照）──────────────────
#: **append-only**。整段重放（同 rack + 轮次 + 袋序再传一次）会让 ``bag_session``
#: 的 session_version 加一并覆盖旧值；护士长要回答「十分钟前那次报的是多少毫升」
#: 时，唯一能查的地方就是这里——它记的是**被覆盖掉的那一版**。
BAG_SESSION_HISTORY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("history_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("session_id", "TEXT NOT NULL"),
    ("session_version", "INTEGER NOT NULL"),
    ("rack_id", "TEXT NOT NULL"),
    ("shift_key", "TEXT NOT NULL"),
    ("bag_seq", "TEXT NOT NULL"),
    ("state", "TEXT"),
    ("trigger_reason", "TEXT"),
    ("frames_dir", "TEXT"),
    ("frame_count", "INTEGER NOT NULL DEFAULT 0"),
    ("captured_at", "TEXT"),
    ("area_ratio_series", "TEXT NOT NULL DEFAULT '[]'"),
    ("area_ratio_raw", "TEXT NOT NULL DEFAULT '[]'"),
    ("occlusion_flags", "TEXT NOT NULL DEFAULT '[]'"),
    ("fill_ratio", "REAL"),
    ("estimated_volume_ml", "REAL"),
    ("eta_minutes", "INTEGER"),
    ("calibration_state", "TEXT"),
    ("calibration_version", "INTEGER"),
    ("client_event_id", "TEXT"),
    ("archived_reason", "TEXT NOT NULL DEFAULT 'replay'"),  # 目前只有整段重放一种
    ("archived_at", "TEXT NOT NULL"),
)

# ── retention_archive（task 44：帧目录归档台账）────────────────────────────
#: **append-only**。一行 = 一次「这袋的帧目录被归档/删除了」的记录。
#:
#: 为什么删完像素还要留这张表：帧目录是像素，删了就没了；但「哪一袋的帧在什么时候
#: 因为什么被清掉了」必须可查——护士长复盘时看到某袋的曲线突然没有原始帧，
#: 信息科要能回答「是保留期到了，不是当时没拍到」。没有这张表就只能靠猜。
RETENTION_ARCHIVE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("archive_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("session_id", "TEXT NOT NULL"),
    ("rack_id", "TEXT"),
    ("bed_no", "TEXT"),
    ("bag_seq", "TEXT"),
    ("shift_key", "TEXT"),
    ("captured_at", "TEXT"),
    ("frames_dir", "TEXT"),  # 清理前的帧目录
    ("archive_path", "TEXT"),  # 归档后的路径；直接删除时为 NULL
    ("frame_count", "INTEGER NOT NULL DEFAULT 0"),
    ("bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("action", "TEXT NOT NULL"),  # archived / deleted / absent
    ("reason", "TEXT NOT NULL"),  # retention_expired
    ("policy_days", "REAL NOT NULL"),
    ("archived_at", "TEXT NOT NULL"),
)

#: 帧目录的保留状态。
#: ``kept`` = 还在保留期内；``archived`` = 已搬到归档目录；``purged`` = 已删除；
#: ``absent`` = 清理时发现目录本来就不在（无需动作，记一笔免得每次都重扫）。
RETENTION_STATES = ("kept", "archived", "purged", "absent")

# ── audit_log ──────────────────────────────────────────────────────────────
#: append-only 审计链：谁、在什么时候、对哪一袋做了什么、结果如何、被拒的原因。
#: **只追加，不更新不删除**——护理质控与飞检要回答的是「当时发生了什么」，
#: 而不是「现在这条记录长什么样」；被拒绝的非法迁移同样要留一行。
AUDIT_LOG_COLUMNS: tuple[tuple[str, str], ...] = (
    ("audit_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("request_id", "TEXT NOT NULL"),
    ("actor", "TEXT NOT NULL"),  # nurse:<operator_id> / service / info-admin
    ("action", "TEXT NOT NULL"),  # 取值见 AUDIT_ACTIONS
    ("target", "TEXT NOT NULL"),  # session_id 或 client_event_id
    ("bag_seq", "TEXT"),
    ("bag_key", "TEXT"),  # 输注单号|袋序：换液点名的持久闸门按它查上一次点名时刻
    # 架号与床号**冗余**落在审计行上（task 33）：飞检与护理部导审计包问的是
    # 「这一架 / 这一床当天发生过什么」，而不是「这个 session_id 发生过什么」；
    # 每次审计都回查 bag_session 才能回答，那正是导 3 万行审计包时的慢路径。
    # 缺这两列的历史行由导出侧的 COALESCE(a.rack_id, s.rack_id) 兜底。
    ("rack_id", "TEXT"),
    ("bed_no", "TEXT"),
    ("from_state", "TEXT"),
    ("to_state", "TEXT"),
    ("outcome", "TEXT NOT NULL"),  # accepted / rejected
    ("detail", "TEXT NOT NULL DEFAULT '{}'"),  # JSON：本次动作的补充信息
    ("payload_hash", "TEXT NOT NULL"),  # detail 的 sha256：改一个字就能看出来
    ("occurred_at", "TEXT NOT NULL"),
)

#: 审计动作码。后续任务只在此扩充，不复用既有码表达新语义。
AUDIT_ACTIONS: tuple[str, ...] = (
    "candidate.claimed",        # 护士在候选列表里认领了一行
    "candidate.corrected",      # 认领后又改绑到另一个候选
    "reanchor.resolved",        # 跨轮次重锚的判定结果（含拒绝猜测的原因）
    "state.change_proposed",    # 服务算出 needs_change
    "state.confirmed",          # 护士确认已换袋 → replaced
    "state.transition_rejected",  # 非法迁移被拒（monitoring 直跳 replaced 等）
    "event.recorded",           # 出站事件首次落库（幂等键在此写入，task 10）
    "event.replay_ignored",     # 同一幂等键重放：读回既有事件，未新增行（task 10）
    "session.write_rejected",   # 越权写入被拒：缺 bag_seq / 袋序不属于该会话（task 10）
    "writeback.sent",           # 出站写回成功（对端 2xx，task 11）
    "writeback.retry",          # 写回失败但可重试：本次退避后还要再发（task 11）
    "writeback.pending",        # 退避重试耗尽：置 pending_writeback 等信息科重放（task 11）
    "writeback.rejected",       # 对端 4xx 业务拒绝：不重试，事件置 failed 留底（task 11）
    "writeback.refused",        # 会话未被护士确认：拒发，一次 HTTP 都没发（task 11）
    "writeback.replayed",       # 信息科人工重放送达（task 37）
    "writeback.replay_failed",  # 信息科人工重放未送达：对端仍不可达/仍拒绝（task 37）
    "telemetry.channel_up",     # MQTT 遥测通道已连上并订阅（task 11）
    "telemetry.channel_down",   # MQTT 遥测通道断开/启动失败（task 11）
    "telemetry.message_rejected",  # 单条遥测载荷不可用：只拒这一条（task 11）
    "telemetry.duplicate_dropped",  # 重投/跨通道重来的同一份遥测：按 device_sn+ts 去重（task 39）
    "telemetry.reconnect_attempted",  # 退避表上的一次重连尝试（issued 与连没连上分开记）（task 39）
    "telemetry.webhook_rejected",     # webhook 兜底通道拒收：验签/时间窗/架号不符（task 39）
    "calibration.deviation_failed",  # 自校准样本没写进库：换液确认仍成立，但这条证据丢了
    "calibration.version_created",   # 新版本登记成功（含首次建版）
    "calibration.version_scheduled", # 登记了未来才生效的版本；当天读数不变
    "calibration.version_superseded",  # 旧版本被新版取代并收口生效区间
    "calibration.review_required",   # 改版偏差超阈值：该袋旧读数保留，等人工复核
    "calibration.review_applied",    # 人工复核通过：新读数按复核结论落库
    "calibration.review_dismissed",  # 人工复核否决：新版本有问题，旧读数继续沿用
    "calibration.window_rejected",   # 生效区间与既有版本相交，登记被拒
    "pickup.named",                  # 本轮真的点了这袋的名（最小间隔闸门的读数来源）
    "pickup.suppressed",             # 判了但被滞回/间隔/上限挡下：不点名，并留原因码
    "pickup.truncated",              # 按每轮上限截断（保留 ETA 与排名，下一轮重来）
    "upload.opened",                 # 开启一次可续传的上传（拿到 upload_id）
    "upload.chunk_received",         # 收到一个分片：新增/去重各多少帧都留痕
    "upload.completed",              # 提交成功：半成品转成正式袋级会话
    "upload.rejected",               # 分片被拒（帧号命名/越界/上传已关闭），只拒这一片
    "upload.abandoned",              # 保留期到点回收半成品：磁盘回收，未完成的这一段作废
    "session.version_archived",      # 整段重放覆盖前把旧版本快照进 bag_session_history
    "inference.fusion_accepted",     # 仲裁采信：模型结论与阈值一致且置信达门禁，只采信其措辞（task 42）
    "inference.fusion_rejected",     # 仲裁回落：冲突/越界/低置信/端点不可用，一律以阈值路径为准（task 42）
    "session.frames_archived",       # 保留期到点：帧目录搬到归档目录（读数留在库里，task 44）
    "session.frames_purged",         # 保留期到点：帧目录已删除（读数留在库里，task 44）
    "audit.inference_pruned",        # 保留期到点：推理审计元数据被裁掉（只清元数据，不碰审计链，task 44）
)

_COLUMN_TYPES = {
    "bag_session": dict(BAG_SESSION_COLUMNS),
    "infusion_event": dict(INFUSION_EVENT_COLUMNS),
    "writeback_queue": dict(WRITEBACK_QUEUE_COLUMNS),
    "inference_audit": dict(INFERENCE_AUDIT_COLUMNS),
    "bag_type_calibration": dict(BAG_TYPE_CALIBRATION_COLUMNS),
    "calibration_recalc_log": dict(CALIBRATION_RECALC_COLUMNS),
    "calibration_deviation": dict(CALIBRATION_DEVIATION_COLUMNS),
    "calibration_version_alert": dict(CALIBRATION_VERSION_ALERT_COLUMNS),
    "upload_chunk": dict(UPLOAD_CHUNK_COLUMNS),
    "bag_session_history": dict(BAG_SESSION_HISTORY_COLUMNS),
    "audit_log": dict(AUDIT_LOG_COLUMNS),
}

CREATE_TABLE_SQL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS bag_session (
        session_id            TEXT PRIMARY KEY,
        rack_id               TEXT NOT NULL,
        bed_no                TEXT NOT NULL,
        infusion_order_no     TEXT NOT NULL,
        shift_key             TEXT NOT NULL,
        bag_seq               TEXT NOT NULL,
        slot_index            INTEGER NOT NULL DEFAULT 0,
        bag_type_code         TEXT,
        frames_dir            TEXT NOT NULL,
        frame_count           INTEGER NOT NULL DEFAULT 0,
        captured_at           TEXT NOT NULL,
        area_ratio_series     TEXT NOT NULL DEFAULT '[]',
        area_ratio_raw        TEXT NOT NULL DEFAULT '[]',
        occlusion_flags       TEXT NOT NULL DEFAULT '[]',
        frame_count_occluded  INTEGER NOT NULL DEFAULT 0,
        fill_ratio            REAL,
        estimated_volume_ml   REAL,
        eta_minutes           INTEGER,
        eta_source            TEXT,
        rate_ml_per_h         REAL,
        rate_source           TEXT,
        rate_reason           TEXT,
        rate_conflict_json    TEXT,
        pickup_suppressed     INTEGER NOT NULL DEFAULT 1,
        suppression_reason    TEXT,
        calibration_state     TEXT NOT NULL DEFAULT 'uncalibrated',
        calibration_version   INTEGER,
        decision_source       TEXT,
        trigger_reason        TEXT,
        state                 TEXT NOT NULL DEFAULT 'monitoring',
        client_event_id       TEXT,
        session_version       INTEGER NOT NULL DEFAULT 1,
        quality_flags         TEXT NOT NULL DEFAULT '[]',
        capture_degraded      INTEGER NOT NULL DEFAULT 0,
        precheck_score        REAL,
        precheck_flags        TEXT NOT NULL DEFAULT '[]',
        precheck_degraded     INTEGER NOT NULL DEFAULT 0,
        camera_width          INTEGER,
        camera_height         INTEGER,
        canonical_width       INTEGER,
        canonical_height      INTEGER,
        frame_scale           REAL,
        roi_window_json       TEXT,
        anchor_candidate_id   TEXT,
        anchor_bbox           TEXT,
        candidates_json       TEXT NOT NULL DEFAULT '[]',
        descriptor_json       TEXT NOT NULL DEFAULT '{}',
        reanchor_state        TEXT,
        reanchor_confidence   REAL,
        reanchor_runner_up    REAL,
        reanchor_reason       TEXT,
        previous_session_id   TEXT,
        previous_shift_key    TEXT,
        claim_state           TEXT NOT NULL DEFAULT 'auto',
        claimed_by            TEXT,
        created_at            TEXT NOT NULL,
        updated_at            TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS infusion_event (
        event_id             TEXT PRIMARY KEY,
        client_event_id      TEXT NOT NULL UNIQUE,
        session_id           TEXT,
        event_type           TEXT NOT NULL DEFAULT 'infusion.bag.changed',
        rack_id              TEXT NOT NULL,
        bed_no               TEXT NOT NULL,
        infusion_order_no    TEXT NOT NULL,
        bag_seq              TEXT NOT NULL,
        fill_ratio           REAL,
        estimated_volume_ml  REAL,
        eta_minutes          INTEGER,
        trigger_reason       TEXT,
        decision_source      TEXT,
        operator_id          TEXT NOT NULL,
        occurred_at          TEXT NOT NULL,
        writeback_status     TEXT NOT NULL DEFAULT 'pending',
        attempt_count        INTEGER NOT NULL DEFAULT 0,
        next_retry_at        TEXT,
        last_error           TEXT,
        payload_json         TEXT NOT NULL,
        created_at           TEXT NOT NULL,
        updated_at           TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS inference_audit (
        audit_id               INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id             TEXT NOT NULL,
        session_id             TEXT,
        bag_seq                TEXT,
        model                  TEXT NOT NULL,
        prompt_hash            TEXT NOT NULL,
        prompt_version         TEXT,
        latency_ms             INTEGER,
        prompt_tokens          INTEGER,
        completion_tokens      INTEGER,
        decision_source        TEXT NOT NULL,
        fallback_reason        TEXT,
        pending_manual_review  INTEGER NOT NULL DEFAULT 0,
        decision               TEXT,
        confidence             REAL,
        created_at             TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version     INTEGER PRIMARY KEY,
        applied_at  TEXT NOT NULL,
        note        TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bag_type_calibration (
        calibration_id       TEXT PRIMARY KEY,
        bag_type_code        TEXT NOT NULL,
        version              INTEGER NOT NULL,
        capacity_ml          REAL NOT NULL,
        usable_capacity_ml   REAL NOT NULL,
        dead_volume_ml       REAL NOT NULL DEFAULT 0,
        meniscus_offset_ratio REAL NOT NULL DEFAULT 0,
        bands_json           TEXT NOT NULL DEFAULT '[]',
        samples_json         TEXT NOT NULL DEFAULT '[]',
        max_relative_error   REAL NOT NULL DEFAULT 0,
        calibrated_by        TEXT NOT NULL,
        calibrated_at        TEXT NOT NULL,
        active               INTEGER NOT NULL DEFAULT 1,
        effective_from       TEXT,
        effective_to         TEXT,
        note                 TEXT,
        created_at           TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS calibration_recalc_log (
        recalc_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        calibration_id    TEXT NOT NULL,
        bag_type_code     TEXT NOT NULL,
        previous_version  INTEGER,
        new_version       INTEGER NOT NULL,
        session_id        TEXT NOT NULL,
        bag_seq           TEXT,
        rack_id           TEXT,
        bed_no            TEXT,
        previous_volume_ml REAL,
        new_volume_ml     REAL,
        delta_ml          REAL,
        reason            TEXT NOT NULL,
        relative_deviation REAL,
        review_required   INTEGER NOT NULL DEFAULT 0,
        recalculated_at   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS calibration_version_alert (
        alert_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        calibration_id      TEXT NOT NULL,
        bag_type_code       TEXT NOT NULL,
        previous_version    INTEGER,
        new_version         INTEGER NOT NULL,
        session_id          TEXT NOT NULL,
        bag_seq             TEXT,
        rack_id             TEXT,
        bed_no              TEXT,
        previous_volume_ml  REAL,
        proposed_volume_ml  REAL,
        delta_ml            REAL,
        relative_deviation  REAL NOT NULL,
        threshold           REAL NOT NULL,
        severity            TEXT NOT NULL,
        resolution          TEXT NOT NULL,
        resolved_by         TEXT,
        resolved_at         TEXT,
        resolution_note     TEXT,
        raised_at           TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS calibration_deviation (
        deviation_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        bag_type_code     TEXT NOT NULL,
        session_id        TEXT,
        bag_seq           TEXT,
        rack_id           TEXT,
        bed_no            TEXT,
        source            TEXT NOT NULL,
        last_fill_ratio   REAL,
        model_volume_ml   REAL,
        residual_ml       REAL NOT NULL,
        eligible          INTEGER NOT NULL DEFAULT 0,
        ineligible_reason TEXT,
        calibration_version INTEGER,
        sample_count      INTEGER NOT NULL DEFAULT 0,
        note              TEXT,
        recorded_at       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS upload_chunk (
        upload_id         TEXT PRIMARY KEY,
        rack_id           TEXT NOT NULL,
        bed_no            TEXT NOT NULL,
        infusion_order_no TEXT NOT NULL,
        shift_key         TEXT NOT NULL,
        captured_at       TEXT NOT NULL,
        slots_json        TEXT NOT NULL,
        client_event_id   TEXT,
        session_key       TEXT NOT NULL,
        staging_dir       TEXT NOT NULL,
        status            TEXT NOT NULL DEFAULT 'pending',
        expected_frames   INTEGER,
        received_frames   TEXT NOT NULL DEFAULT '[]',
        frame_times_json  TEXT NOT NULL DEFAULT '{}',
        duplicate_count   INTEGER NOT NULL DEFAULT 0,
        chunk_count       INTEGER NOT NULL DEFAULT 0,
        total_bytes       INTEGER NOT NULL DEFAULT 0,
        session_ids_json  TEXT NOT NULL DEFAULT '[]',
        resize_json       TEXT,
        created_at        TEXT NOT NULL,
        updated_at        TEXT NOT NULL,
        expires_at        TEXT NOT NULL,
        completed_at      TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bag_session_history (
        history_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id         TEXT NOT NULL,
        session_version    INTEGER NOT NULL,
        rack_id            TEXT NOT NULL,
        shift_key          TEXT NOT NULL,
        bag_seq            TEXT NOT NULL,
        state              TEXT,
        trigger_reason     TEXT,
        frames_dir         TEXT,
        frame_count        INTEGER NOT NULL DEFAULT 0,
        captured_at        TEXT,
        area_ratio_series  TEXT NOT NULL DEFAULT '[]',
        area_ratio_raw     TEXT NOT NULL DEFAULT '[]',
        occlusion_flags    TEXT NOT NULL DEFAULT '[]',
        fill_ratio         REAL,
        estimated_volume_ml REAL,
        eta_minutes        INTEGER,
        calibration_state  TEXT,
        calibration_version INTEGER,
        client_event_id    TEXT,
        archived_reason    TEXT NOT NULL DEFAULT 'replay',
        archived_at        TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        audit_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id   TEXT NOT NULL,
        actor        TEXT NOT NULL,
        action       TEXT NOT NULL,
        target       TEXT NOT NULL,
        bag_seq      TEXT,
        bag_key      TEXT,
        rack_id      TEXT,
        bed_no       TEXT,
        from_state   TEXT,
        to_state     TEXT,
        outcome      TEXT NOT NULL,
        detail       TEXT NOT NULL DEFAULT '{}',
        payload_hash TEXT NOT NULL,
        occurred_at  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS writeback_queue (
        queue_id           TEXT PRIMARY KEY,
        event_id           TEXT NOT NULL UNIQUE,
        client_event_id    TEXT NOT NULL,
        session_id         TEXT,
        rack_id            TEXT,
        bed_no             TEXT,
        bag_seq            TEXT,
        payload_json       TEXT NOT NULL,
        attempt_count      INTEGER NOT NULL DEFAULT 0,
        next_retry_at      TEXT,
        last_error         TEXT,
        status             TEXT NOT NULL DEFAULT 'queued',
        dead_letter_reason TEXT,
        last_attempt_at    TEXT,
        created_at         TEXT NOT NULL,
        updated_at         TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS retention_archive (
        archive_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id    TEXT NOT NULL,
        rack_id       TEXT,
        bed_no        TEXT,
        bag_seq       TEXT,
        shift_key     TEXT,
        captured_at   TEXT,
        frames_dir    TEXT,
        archive_path  TEXT,
        frame_count   INTEGER NOT NULL DEFAULT 0,
        bytes         INTEGER NOT NULL DEFAULT 0,
        action        TEXT NOT NULL,
        reason        TEXT NOT NULL,
        policy_days   REAL NOT NULL,
        archived_at   TEXT NOT NULL
    )
    """,
)

CREATE_INDEX_SQL: tuple[str, ...] = (
    # 会话定位：同一输液架、同一巡视轮次、同一袋序 —— 重复上传走覆盖式更新
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_bag_session_rack_shift_bag "
    "ON bag_session (rack_id, shift_key, bag_seq)",
    "CREATE INDEX IF NOT EXISTS ix_bag_session_state ON bag_session (state)",
    "CREATE INDEX IF NOT EXISTS ix_bag_session_order ON bag_session (infusion_order_no, bed_no)",
    # 幂等键：client_event_id 的唯一约束即重复确认的去重依据
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_infusion_event_client_event_id "
    "ON infusion_event (client_event_id)",
    "CREATE INDEX IF NOT EXISTS ix_infusion_event_writeback "
    "ON infusion_event (writeback_status, next_retry_at)",
    "CREATE INDEX IF NOT EXISTS ix_infusion_event_session ON infusion_event (session_id, bag_seq)",
    # 待写回队列（task 37）：一条事件一行；到期可自动重试的按 (status, next_retry_at) 取，
    # 信息科按架/按床查欠账时走另外两条索引。
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_writeback_queue_event "
    "ON writeback_queue (event_id)",
    "CREATE INDEX IF NOT EXISTS ix_writeback_queue_status ON writeback_queue (status, next_retry_at)",
    "CREATE INDEX IF NOT EXISTS ix_writeback_queue_rack ON writeback_queue (rack_id, status)",
    "CREATE INDEX IF NOT EXISTS ix_writeback_queue_bed ON writeback_queue (bed_no, status)",
    # 审计按 request_id 与时间检索
    "CREATE INDEX IF NOT EXISTS ix_inference_audit_request ON inference_audit (request_id)",
    "CREATE INDEX IF NOT EXISTS ix_inference_audit_created ON inference_audit (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_inference_audit_source ON inference_audit (decision_source)",
    # 标定：同袋型同版本唯一；按袋型取生效版本
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_bag_type_calibration_code_version "
    "ON bag_type_calibration (bag_type_code, version)",
    "CREATE INDEX IF NOT EXISTS ix_bag_type_calibration_active "
    "ON bag_type_calibration (bag_type_code, active)",
    # 标定重算轨迹按袋型与时间检索
    "CREATE INDEX IF NOT EXISTS ix_calibration_recalc_code "
    "ON calibration_recalc_log (bag_type_code, recalculated_at)",
    # 自校准样本：按袋型与记录时间取全部证据（含不可用样本，便于复核为什么没生效）
    "CREATE INDEX IF NOT EXISTS ix_calibration_deviation_code "
    "ON calibration_deviation (bag_type_code, recorded_at)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_calibration_deviation_session "
    "ON calibration_deviation (session_id)",
    # 偏差告警：按袋型取待复核清单（resolution=pending 就是护理部要处理的那批）
    "CREATE INDEX IF NOT EXISTS ix_calibration_alert_pending "
    "ON calibration_version_alert (bag_type_code, resolution, raised_at)",
    "CREATE INDEX IF NOT EXISTS ix_calibration_alert_session "
    "ON calibration_version_alert (session_id, raised_at)",
    "CREATE INDEX IF NOT EXISTS ix_bag_session_calibration "
    "ON bag_session (bag_type_code, state)",
    # 跨轮次重锚：按输液架 + 轮次取上一轮同架会话
    "CREATE INDEX IF NOT EXISTS ix_bag_session_rack_shift "
    "ON bag_session (rack_id, shift_key)",
    # 审计链：按对象与时间检索；append-only，无更新无删除路径
    "CREATE INDEX IF NOT EXISTS ix_audit_log_target ON audit_log (target, occurred_at)",
    "CREATE INDEX IF NOT EXISTS ix_audit_log_bagkey ON audit_log (bag_key, action, audit_id)",
    "CREATE INDEX IF NOT EXISTS ix_audit_log_action ON audit_log (action, occurred_at)",
    "CREATE INDEX IF NOT EXISTS ix_audit_log_request ON audit_log (request_id)",
    # 导审计包的两个主过滤维度（task 33）：按架、按床各走时间序，末端 LIMIT 不用回表
    "CREATE INDEX IF NOT EXISTS ix_audit_log_rack_time ON audit_log (rack_id, occurred_at)",
    "CREATE INDEX IF NOT EXISTS ix_audit_log_bed_time ON audit_log (bed_no, occurred_at)",
    # 半成品上传：只扫 pending 且已过保留期的（清理策略），以及按输液架查未完成的那几段
    "CREATE INDEX IF NOT EXISTS ix_upload_chunk_status ON upload_chunk (status, expires_at)",
    "CREATE INDEX IF NOT EXISTS ix_upload_chunk_rack ON upload_chunk (rack_id, shift_key)",
    # 旧版本快照：按会话取全部被覆盖掉的版本（append-only，无更新无删除路径）
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_bag_session_history_version "
    "ON bag_session_history (session_id, session_version)",
    "CREATE INDEX IF NOT EXISTS ix_bag_session_history_bag "
    "ON bag_session_history (rack_id, bag_seq, archived_at)",
    # 保留策略：只扫「还在保留期内以外、且帧目录还在」的会话
    "CREATE INDEX IF NOT EXISTS ix_bag_session_retention "
    "ON bag_session (frames_retention_state, captured_at)",
    # 归档台账：按会话与按时间检索（append-only，无更新无删除路径）
    "CREATE INDEX IF NOT EXISTS ix_retention_archive_session "
    "ON retention_archive (session_id, archived_at)",
    "CREATE INDEX IF NOT EXISTS ix_retention_archive_time "
    "ON retention_archive (archived_at)",
)


def column_type(table: str, column: str) -> str | None:
    """取列的 DDL 类型片段；表或列不存在返回 None（供幂等加列判断）。"""
    cols = _COLUMN_TYPES.get(table)
    if cols is None:
        raise KeyError(f"未知表：{table}")
    return cols.get(column)