"""入站请求与出站响应的 Pydantic 模型（OpenAPI 契约的第一版）。

**门禁语义进契约**：``estimated_volume_ml`` / ``eta_minutes`` 等字段在无标定或低置信
时是「字段整体不出现」，不是返回 ``null`` 占位——用 ``exclude_none=True`` 序列化即可。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BagSlotOut",
    "CandidateOut",
    "SmoothingOut",
    "SlotSpecIn",
    "TrackUploadForm",
    "track_upload_body_schema",
    "TrackUploadResponse",
    "HealthResponse",
    "ErrorIssueOut",
    "ErrorDetailOut",
    "ErrorCodeOut",
    "ErrorCodeListOut",
    "CalibrationSampleIn",
    "CalibrationCreateIn",
    "CalibrationOut",
    "ReviewResolveIn",
    "ClaimIn",
    "ClaimOut",
    "DecisionIn",
    "DecisionOut",
    "PickupItemOut",
    "PickupListOut",
    "TransitionIn",
    "TransitionOut",
    "WritebackOut",
]


class SlotSpecIn(BaseModel):
    """``slots`` JSON 数组里的一项：一个输液架挂点上的袋子。

    ``bag_type_code`` 决定能否换算出毫升数（没登记标定就只给面积比），
    ``rate_text`` 是输注医嘱里的滴速文本，是 ETA 的第一优先来源。
    """

    model_config = ConfigDict(extra="forbid")

    bag_seq: str = Field(description="袋序标识，由医嘱/护理单元系统给出")
    slot_index: int = Field(ge=0, description="输液架上的挂点序号，从 0 开始")
    bag_type_code: str | None = Field(default=None, description="袋型编码；未登记标定时仍可传")
    rate_text: str | None = Field(
        default=None, description="输注医嘱里的滴速文本，如「5ml/h」；抽不出就退到泵速遥测"
    )


class TrackUploadForm(BaseModel):
    """``POST /api/v1/infusion/tracks`` 的入站 multipart/form-data 字段（必填清单真源）。

    这个模型**不**直接作为 FastAPI 的 ``Form()`` 参数使用——那样生成的 OpenAPI 只声明
    ``application/x-www-form-urlencoded``，挂接方看不出这是个 multipart 上传。真正的
    契约由 :func:`track_upload_body_schema` 展开：文本字段取自这里的 ``model_json_schema``，
    二进制字段 ``frames`` 单独按 ``{type: string, format: binary}`` 声明，
    再由 ``api/contract.py`` 挂回 ``multipart/form-data`` 的 requestBody。

    必填五项 + 帧 zip 缺一不可；缺任何一项都返回 422，detail.issues 里点名该字段。
    """

    model_config = ConfigDict(extra="forbid")

    rack_id: str = Field(description="输液架编号，与护理执行平台登记一致（不是床号）")
    infusion_order_no: str = Field(
        description="输注单号；权威标识由医嘱/护理单元系统给出，本服务不自行编号"
    )
    bed_no: str = Field(description="该输液架当前对应的床号")
    captured_at: str = Field(description="拍摄时刻，ISO8601，如 2026-10-22T09:41:00+08:00")
    slots: str = Field(
        description=(
            'JSON 数组，元素见 SlotSpecIn，如 [{"bag_seq":"B01","slot_index":0,'
            '"bag_type_code":"TYPE-A","rate_text":"5ml/h"}]；单架最多 6 袋'
        )
    )
    client_event_id: str | None = Field(default=None, description="幂等键（可选）；重传同一段请复用同一个值")
    shift_key: str | None = Field(
        default=None, description="巡视轮次键（可选）；不传时按 captured_at 的日期生成"
    )
    frame_times: str | None = Field(
        default=None, description="与帧数等长的 ISO8601 时间数组 JSON（可选）；缺省取 zip 成员时间戳"
    )

    @classmethod
    def required_field_names(cls) -> list[str]:
        return ["rack_id", "infusion_order_no", "bed_no", "captured_at", "slots"]


def track_upload_body_schema() -> dict[str, Any]:
    """入站 multipart 体的 JSON Schema（文本字段取自模型 + 二进制 ``frames``）。"""
    scalar = TrackUploadForm.model_json_schema(ref_template="#/components/schemas/{model}")
    required = TrackUploadForm.required_field_names()
    properties = {
        name: prop
        for name, prop in scalar.get("properties", {}).items()
    }
    properties["frames"] = {
        "type": "string",
        "format": "binary",
        "description": (
            "帧 zip，成员命名为 frame_00000.jpg 起的连续序号，只含帧图片；"
            f"帧数 {12}–{1200}、短边 ≥320px、时间戳单调递增、单架 ≤6 袋"
        ),
    }
    schema: dict[str, Any] = {
        "type": "object",
        "title": "TrackUploadForm",
        "description": TrackUploadForm.__doc__ or "",
        "properties": properties,
        "required": [*required, "frames"],
        "additionalProperties": False,
    }
    slot_schema = SlotSpecIn.model_json_schema(ref_template="#/components/schemas/{model}")
    slot_schema.pop("$defs", None)
    schema["x-slots-item-schema"] = slot_schema
    return schema


class BagSlotOut(BaseModel):
    """一个袋位的接入结果 + 跨帧面积比曲线。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="袋级会话 ID；transition/claim 用它定位")
    bag_seq: str = Field(description="袋序标识，由医嘱/护理单元系统给出")
    slot_index: int = Field(description="输液架上的挂点序号，从 0 开始")
    bag_type_code: str | None = Field(default=None, description="袋型编码，未登记时为空")
    frames_dir: str = Field(description="本次上传解包后的帧目录")
    frame_count: int = Field(description="该袋关联的帧数（同一段上传内各袋共享）")
    state: str = Field(description="业务状态：monitoring / needs_change / replaced")
    calibration_state: str = Field(
        description=(
            "标定与置信状态：uncalibrated（没定标）/ calibrated（可信）/ "
            "low_confidence（定标在但置信不够）/ high_occlusion（这段挡得太久）"
        )
    )
    session_version: int = Field(description="同一会话的第几版（重传递增）")
    frame_indices: list[int] = Field(
        default_factory=list, description="曲线对应的帧序号，与 area_ratio_series 一一对齐"
    )
    area_ratio_series: list[float | None] = Field(
        default_factory=list,
        description=(
            "逐帧液体面积占袋腔的面积比；该帧看不见这袋时为 null（标 occluded），"
            "不插值也不沿用上一帧数值"
        ),
    )
    occlusion_flags: list[bool] = Field(
        default_factory=list, description="逐帧是否可见；true 表示该帧被遮挡或未检出"
    )
    visibility_states: list[str | None] = Field(
        default_factory=list,
        description=(
            "逐帧可见度：visible（完整可见）/ partial（只看到部分袋体）/ occluded（完全看不见，"
            "面积比留 null）/ reappeared（遮挡后接回同一袋）/ lost（超过丢失窗口、轨迹已交出"
            "待再获取）；起锚前的帧为 null"
        ),
    )
    track_handoff: dict[str, Any] | None = Field(
        default=None,
        description=(
            "轨迹交出单：连续遮挡超过丢失窗口时给出，含 obj_id/lost_frame/last_seen_frame/"
            "bbox/centroid/reference_area_ratio/reason，交由再获取路径认领；"
            "未交出时字段整体不出现（与 estimated_volume_ml 同一套门禁语义）"
        ),
    )
    cavity_area_px: int = Field(default=0, description="首帧估出的袋腔像素面积（面积比的分母）")
    visible_frame_count: int = Field(default=0, description="成功检出的帧数")
    occluded_frame_count: int = Field(default=0, description="被遮挡或未检出的帧数")
    # ── 体积与 ETA（门禁：不满足条件时下面几个字段整体不出现） ──────────────
    fill_ratio: float | None = Field(
        default=None,
        description=(
            "末次可见帧的液体占袋腔面积比（观测值，不经标定换算）；"
            "整段看不见时不出现"
        ),
    )
    estimated_volume_ml: float | None = Field(
        default=None,
        description=(
            "按袋型标定换算出的袋内余量毫升数。**无生效标定记录时不出现**"
            "（calibration_state=uncalibrated），不给 null 占位"
        ),
    )
    calibration_version: int | None = Field(
        default=None,
        description="这个毫升数用的是该袋型第几版标定；改版后数字变了可据此回溯"
    )
    eta_minutes: int | None = Field(
        default=None,
        description="按当前滴速预计耗尽的分钟数，向上取整；滴速或标定缺失时不出现"
    )
    eta_source: str | None = Field(
        default=None,
        description=(
            "滴速来源：order_text（输注医嘱文本）/ mqtt_telemetry（泵速遥测）/ "
            "conflict（医嘱与泵速遥测对不上，不给时点）/ "
            "unavailable（两者都缺，不给时点）"
        ),
    )
    rate_ml_per_h: float | None = Field(
        default=None, description="本次采用的滴速（ml/h）；两个来源都给不出、或两者冲突时不出现"
    )
    rate_reason: str | None = Field(
        default=None, description="滴速退到下一来源、或给不出来时的原因说明"
    )
    rate_conflict: dict[str, Any] | None = Field(
        default=None,
        description=(
            "医嘱速率与泵速遥测对不上的证据（order_rate_ml_per_h / "
            "telemetry_rate_ml_per_h / relative_deviation / tolerance / message）；"
            "两者一致或只有一个来源时不出现"
        ),
    )
    telemetry_state: str | None = Field(
        default=None,
        description=(
            "泵速遥测现状：absent（这一袋从来没有任何遥测）/ fresh（新鲜窗口内）/ "
            "stale（超过新鲜窗口）/ undated（泵没报时间戳，新鲜度无从判断）。"
            "**遥测缺失不阻断任何结论**：ETA 退回输注医嘱速率，余量与换液判定照常"
        ),
    )
    telemetry_age_seconds: float | None = Field(
        default=None, description="这条遥测距本次上传拍摄时刻多少秒；无遥测或时间戳不可解析时不出现"
    )
    telemetry_source: str | None = Field(
        default=None, description="这条遥测是从哪条通道进来的（mqtt / webhook）"
    )
    pickup_suppressed: bool = Field(
        default=True,
        description=(
            "是否抑制该袋的换液点名：true 表示「依据不足以点名」，护士端只展示不催换。"
            "滴速缺失、医嘱与泵速对不上、未标定、整段看不见、超装五种情况均为 true"
        ),
    )
    suppression_reason: str | None = Field(
        default=None,
        description=(
            "抑制原因码：uncalibrated / eta_unavailable / rate_conflict / no_observation / "
            "overfilled / low_confidence / high_occlusion；未抑制时不出现。"
            "前五个是「缺东西」（没标定、没滴速、整段没看见、超装），"
            "后两个是「有标定但这一段的观测撑不起那个数」——护士端据此显示不同的补拍/补登记提示"
        ),
    )
    volume_uncertainty: UncertaintyOut | None = Field(
        default=None,
        description=(
            "这个毫升数的可信区间与置信结论（几何版本 / 面积比方差 / 遮挡占比三证合成）。"
            "**门禁不通过时照样出现**——它就是护士端显示「为什么这次没有毫升数、"
            "下一步该做什么」的依据；毫升数被门禁挡下时它一定在"
        ),
    )
    near_empty: bool = Field(
        default=False, description="余量已进入近空区间（面积比 ≤ 6%），提醒护士别等"
    )
    overfilled: bool = Field(
        default=False, description="面积比超装（几何核可能把袋外液体算进来），已拒绝出数"
    )
    estimation_mode: str = Field(
        default="band",
        description="毫升数来源：band=分段带（经验标定），cavity=袋型腔体几何换算",
    )
    cavity_height_fraction: float | None = Field(
        default=None,
        description="腔体反解出的真实高度比（0=空，1=满）；仅 estimation_mode=cavity 时出现",
    )
    resident_ml: float | None = Field(
        default=None,
        description="已扣除的残留量（袋壁挂液 + 自校准残差），单位 ml；仅 cavity 模式出现",
    )
    predictor: str = Field(default="", description="实际使用的预测器实现名（geom_kernel 等）")
    candidate_state: str = Field(
        default="needs_claim",
        description=(
            "首帧枚举的认领状态：bound 表示唯一命中某个候选并已据此起锚；"
            "claimed 表示护士已在候选列表里认领或改绑；"
            "needs_claim 表示没找到或认不准，需护士在候选列表里认领"
        ),
    )
    candidate_id: str | None = Field(
        default=None, description="绑到本槽位的候选 ID（认领时回传它）"
    )
    # ── 跨巡视轮次重锚：下一轮还认不认得出这是同一袋 ────────────────────────
    reanchor_state: str | None = Field(
        default=None,
        description=(
            "与上一轮同架会话的重锚结果：bound（高置信续上同一袋曲线）/ "
            "fresh（上一轮已换掉，本轮是新袋）/ unbound（认不准，需护士认领）"
        ),
    )
    reanchor_reason: str | None = Field(
        default=None,
        description=(
            "重锚判定的原因码：first_round / previous_replaced / anchored / "
            "low_confidence / appearance_changed（位置对得上但外观变了，多半换了新袋）/ "
            "ambiguous_slot / bag_seq_mismatch / "
            "appearance_unavailable / no_candidate"
        ),
    )
    reanchor_confidence: float | None = Field(
        default=None, description="重锚打分（0–1）；认不准时也会给出，供护士判断要不要自己认"
    )
    reanchor_runner_up: float | None = Field(
        default=None, description="次高分；与最高分差值过小即判认不准（ambiguous_slot）"
    )
    reanchor_driver: str | None = Field(
        default=None,
        description=(
            "把重锚分拖下去的那一项（geometry / colour / level / label）："
            "护士端据此提示「是颜色不一样」还是「液位比上一轮还高（多半是换了新袋）」"
        ),
    )
    descriptor_mode: str | None = Field(
        default=None,
        description=(
            "外观描述子档位：full（颜色+液位+标签区都可用）/ "
            "color_only（看不出标签区，已降级为几何+颜色+液位）/ "
            "geometry_only（区域太小，外观不可用，一律不自动接上）"
        ),
    )
    previous_session_id: str | None = Field(
        default=None, description="续上的上一轮会话 ID；新袋（fresh）时不出现"
    )
    previous_shift_key: str | None = Field(
        default=None, description="上一轮巡视的轮次键（如 2026-10-22-AM）"
    )
    claim_state: str = Field(
        default="auto", description="auto（程序自动起锚）/ claimed（护士认领）/ corrected（认领后改绑）"
    )
    # ── 换液点名的闸门结论：点名了没、为什么没点 ──────────────────────────
    trigger_reason: str | None = Field(
        default=None,
        description=(
            "点名触发原因码：eta_within_window（预计撑不到本轮巡视结束）/ "
            "volume_below_margin（余量低于下限）/ near_empty_observed（液面已近空）/ "
            "nurse_confirmed（护士确认已换袋）。未点名时不出现"
        ),
    )
    pickup_state: str | None = Field(
        default=None,
        description=(
            "本轮点名结论：named（点了名，等护士确认）/ suppressed（判了但被闸门挡下）/ "
            "null 表示这一袋本来就没到点名条件"
        ),
    )
    pickup_suppress_reason: str | None = Field(
        default=None,
        description=(
            "不点名的原因码：state_replaced / no_observation / no_trigger / "
            "hysteresis_hold / hysteresis_resolved / latched / interval_not_elapsed / "
            "round_cap_exceeded。护士长据此回答「这袋这轮为什么没被点名」"
        ),
    )
    pickup_detail: str | None = Field(
        default=None, description="点名或抑制的人话说明（可直接显示在护士端提示条）"
    )
    pickup_rank: int | None = Field(
        default=None, description="本轮点名序号（按 ETA 升序，从 1 开始）；被截断时给排到的名次"
    )
    pickup_truncated: bool = Field(
        default=False, description="是否因每轮点名上限被截断（截断不是丢弃，下一轮重新参与）"
    )
    # ── 决策来源三态：这句话是算出来的还是模型写的（gateway/fusion.py）─────
    decision_source: str = Field(
        default="threshold_only",
        description=(
            "结论来源：threshold_only（本轮没有模型参与，结论完全由确定性阈值路径给出，"
            "院内未部署端点时的常态）/ model_decided（模型结论与阈值一致且置信达门禁，"
            "只采信它给护士看的措辞，结论与触发原因码仍取阈值路径）/ "
            "threshold_fallback（调了但未成或与阈值冲突，一律回落阈值路径）。"
            "**无论哪一态，needs_change → replaced 都只能由护士确认触发**"
        ),
    )
    decision_reason: str | None = Field(
        default=None,
        description=(
            "点名理由的一句话中文说明：采信模型措辞时是模型那句，否则是阈值路径的"
            "可复算说明（连续几帧低于多少、余量判定是哪一条）"
        ),
    )
    arbitration: dict[str, Any] | None = Field(
        default=None,
        description=(
            "仲裁留痕：agreed_with_threshold（一致采信）/ conflicts_with_threshold（冲突回落）/ "
            "below_confidence_floor（低置信回落）/ model_out_of_contract（越界回落）/ "
            "gateway_degraded（端点不可达回落）/ model_not_called（本轮没调模型）/ "
            "no_threshold_conclusion（没有可比对的阈值结论）。"
            "护士长与信息科复核「这袋为什么这么说」看的就是它"
        ),
    )
    # ── 槽位-轨迹分配：这条曲线是不是真的落在它声明的那个挂点上 ──────────────
    slot_state: str | None = Field(
        default=None,
        description=(
            "多袋并行传播后的槽位分配结论：assigned（确实落在声明的挂点竖列里）/ "
            "mismatched（停在别的挂点那一列，可能拍到隔壁那只）/ ambiguous（分不出先后）/ "
            "no_evidence（整段没看清）/ below_score（三通道加权没过门槛）/ "
            "unfilled（这个挂点没拍到）/ extra_track（多出一条不在任何挂点列上的轨迹）。"
            "**非 assigned 一律 needs_claim**，不默认取分数最高的"
        ),
    )
    slot_assigned_index: int | None = Field(
        default=None, description="实际判定的挂点序号（0 起）；与 slot_index 不一致即袋序错位"
    )
    slot_score: float | None = Field(
        default=None, description="槽位分配总分（0–1，三个通道加权）"
    )
    slot_margin: float | None = Field(
        default=None, description="最佳与次佳挂点的分差；过小即判 ambiguous 不猜"
    )
    slot_reason: str | None = Field(
        default=None,
        description=(
            "分配原因码：assigned / declared_slot_mismatch / ambiguous_slot / "
            "ambiguous_track / track_never_seen / slot_empty / extra_track / no_slot_declared"
        ),
    )
    slot_detail: str | None = Field(
        default=None, description="给护士看的中文说明：现在是什么情况、下一步点哪里"
    )
    # ── 面积序列平滑：原始观测与工作序列并存，抖动与反常都看得见 ────────────
    area_ratio_smoothed: list[float | None] = Field(
        default_factory=list,
        description=(
            "平滑后的工作序列（中值 + 一阶平滑），逐帧与 area_ratio_series 对齐；"
            "看不见的帧仍是 null。反常帧（液面不该上升却上升了）**保留原值**并在下"
            "方 smoothing 里点名，不悄悄抹平"
        ),
    )
    area_ratio_reconstructed: list[float | None] = Field(
        default_factory=list,
        description=(
            "受限重建的补值序列：只有「前后端点都是可信观测」且缺帧数在 "
            "2–10 之间的帧才会有值，其余为 null。重建只用于看清这段的下降速度，"
            "**不作为余量读数**；端点不全、超长或中间疑似换袋时拒绝重建"
        ),
    )
    smoothing: SmoothingOut | None = Field(
        default=None,
        description=(
            "平滑诊断：尖峰帧 / 反常帧 / 重建帧 / 每段缺帧的处理原因 / 平均降幅。"
            "原始序列始终在 area_ratio_series，两者可逐帧核对"
        ),
    )
    # ── 拍摄质量：拍不好时给补拍建议，而不是默默给一个不可信的毫升数 ────
    capture_degraded: bool = Field(
        default=False,
        description=(
            "本次拍摄被判为「拍坏了」（反光/暗光/糊/歪/出框任一命中）。"
            "**仍然照常返回曲线与体积**，只是这些数字要打问号——"
            "护士端应把 capture_quality 里那一条补拍建议顶到提示条上"
        ),
    )
    capture_quality: list[QualityFlagOut] = Field(
        default_factory=list,
        description=(
            "命中的拍摄问题（按「该先修哪个」排序，最多 6 条）。第一条是 primary——"
            "修好它，后面的多半自动消失。空滴斗为 info 级：只解释现象，不建议重拍"
        ),
    )
    precheck_score: float | None = Field(
        default=None,
        description="上传前预检给整段短片的 0–1 质量分；未跑预检时为 null（不是 0）",
    )
    precheck_degraded: bool = Field(
        default=False,
        description="上传前预检判整段「值不值得信」。与 capture_degraded 分开报：本袋拍坏了 ≠ 整段拍坏了",
    )


class QualityFlagOut(BaseModel):
    """一条拍摄问题 + 一句护士能照做的补拍建议。"""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        description=(
            "问题码：bag_out_of_frame（袋体出框）/ motion_blur（运动模糊）/ "
            "glare_overexposed（反光过曝）/ bad_angle（角度不正）/ low_light（暗光）/ "
            "empty_drip_chamber（滴斗空管，info 级）/ low_resolution（短边分辨率偏低，"
            "仅上传前预检给出）"
        )
    )
    severity: str = Field(description="warning（建议重拍）/ info（只解释，不建议重拍）")
    label_zh: str = Field(description="中文说法，给护士看的名字")
    detail_zh: str = Field(description="现在是什么情况：命中几帧、实测值多少、门限多少")
    suggestion_zh: str = Field(description="下一步怎么做：一句能照做的补拍动作")
    hit_frames: int = Field(description="采样帧里命中该问题的帧数")
    sampled_frames: int = Field(description="本次质检采样了多少帧")
    hit_ratio: float = Field(description="命中占比（hit_frames / sampled_frames）")
    evidence: dict[str, Any] = Field(
        default_factory=dict, description="实测值与门限，便于信息科判断是拍坏了还是门限太严"
    )


class UncertaintyOut(BaseModel):
    """体积不确定度与置信门禁结论（``BagSlotOut.volume_uncertainty``）。"""

    model_config = ConfigDict(extra="forbid")

    confidence: float = Field(
        description="置信度（0–1）：1 = 区间零宽，越低说明这个毫升数越只能当参考"
    )
    relative_sigma: float = Field(
        description="相对不确定度（区间半宽 ÷ 毫升数 ÷ 覆盖系数）；门限等价于它超过 0.14"
    )
    sigma_ml: float = Field(description="绝对不确定度（毫升，1σ）")
    low_ml: float = Field(description="95% 区间下界（毫升，不低于 0）")
    high_ml: float = Field(description="95% 区间上界（毫升）")
    gate: str = Field(
        description="门禁结论：ok（放行）/ low_confidence / high_occlusion；后两者毫升数不出"
    )
    reason: str = Field(
        default="",
        description=(
            "成因码（给信息科）：geometry_version_missing（腔体几何没登记版本号）/ "
            "occluded_too_much（遮挡占比过高）/ relative_uncertainty_too_high（相对不确定度过大）；"
            "放行时为空串"
        ),
    )
    hint_zh: str = Field(description="护士端一句可执行的话：现在是什么情况、下一步做什么")
    factors: dict[str, Any] = Field(
        default_factory=dict,
        description="逐项证据：模型项、面积比噪声、灵敏度、遮挡占比与各自门限",
    )


class SmoothingOut(BaseModel):
    """一袋曲线的平滑诊断（``BagSlotOut.smoothing``）。"""

    model_config = ConfigDict(extra="forbid")

    outlier_frames: list[int] = Field(
        default_factory=list,
        description=(
            "反常帧（液面物理上只会降，这些帧却升了上去——多半是滴斗空管或袋体"
            "被挡）：不参与下游读数，但原值仍在 area_ratio_series 里"
        ),
    )
    spike_frames: list[int] = Field(
        default_factory=list, description="尖峰帧（反光高光造成，已被中值吸收）"
    )
    reconstructed_frames: list[int] = Field(
        default_factory=list, description="被受限重建补值的帧号"
    )
    missing_frame_count: int = Field(
        default=0, description="看不见的帧数（这些帧在三条序列里都是 null）"
    )
    drain_ratio_per_frame: float | None = Field(
        default=None, description="整段平均降幅（面积比/帧，正数表示在漏）；可信观测不足两点时为 null"
    )
    latest_raw: float | None = Field(
        default=None, description="末次原始观测值（未经平滑）"
    )
    latest_trusted: float | None = Field(
        default=None, description="末次可信读数——毫升数与 ETA 用的就是这个"
    )
    degraded: bool = Field(
        default=False, description="存在尖峰/反常/缺帧时为 true（护士端据此提示本段拍得一般）"
    )
    gaps: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "每段缺帧的处理结果：start/end/length/reason/filled。reason 为 "
            "reconstructed_from_endpoints 表示补了，其余为拒绝原因码"
            "（no_trusted_front_endpoint / no_trusted_back_endpoint / "
            "gap_shorter_than_min / gap_longer_than_max / endpoint_order_invalid）"
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="本次生效的平滑参数（来自环境变量配置面）"
    )


class CandidateOut(BaseModel):
    """首帧无提示枚举出的一个候选（护士在列表里认领的那一行）。"""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(description="候选 ID；认领时回传")
    bbox: list[int] = Field(
        description="外接框 [x0, y0, x1, y1]，原图像素坐标（左上角为原点）"
    )
    area: int = Field(description="候选面积（像素²）")
    stability_score: float = Field(
        description="稳定性分：阈值扰动下这块区域的保持比例，越接近 1 越可靠"
    )
    slot_index: int | None = Field(
        default=None, description="聚到的挂点序号；认不准时为空"
    )
    bound: bool = Field(default=False, description="是否已唯一绑定到某个挂点")


class FrameQualityOut(BaseModel):
    """采样帧的拍摄质量观测量（信息科核对「当时到底拍成什么样」用）。"""

    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(description="该帧在整段短片里的下标")
    width: int
    height: int
    short_side: int = Field(description="短边像素；低于质量下限即触发 low_resolution")
    luma_mean: float = Field(description="全画面平均灰度（0–255）")
    dark_ratio: float = Field(description="暗像素占比；只进证据，不单独触发")
    glare_ratio: float = Field(description="近白（无彩色）像素占比，反光过曝的判据")
    sharpness: float = Field(description="拉普拉斯方差；越低越糊")
    tilt_deg: float | None = Field(
        default=None,
        description="液面线倾角（度）；画面里找不到可量的液面线时为 null（判不出就说判不出）",
    )


class PrecheckOut(BaseModel):
    """上传前预检结论：整段短片一份（不是逐袋一份）。"""

    model_config = ConfigDict(extra="forbid")

    state: str = Field(description="ok（本段可按面值用）/ degraded（读数要打问号）")
    quality_score: float = Field(
        description=(
            "0–1 质量分：五个子分（清晰度/亮度/反光/分辨率/角度）加权平均。"
            "**只降级不拒绝**——分低照样返回曲线与体积，只是护士端顶补拍提示条"
        )
    )
    degraded: bool = Field(description="命中任一类问题，或总分低于门禁即 true")
    sampled_frames: int = Field(description="本次预检采样了多少帧（恒定，与短片长度无关）")
    sub_scores: dict[str, float] = Field(
        default_factory=dict,
        description="各子分（0–1）；量不出来的子项不出现在这里——判不出 ≠ 拍得差",
    )
    primary: QualityFlagOut | None = Field(
        default=None, description="最该先补的那一条建议（护士端提示条只顶这一条）"
    )
    flags: list[QualityFlagOut] = Field(
        default_factory=list, description="命中的拍摄问题（按「该先修哪个」排序）"
    )
    frames: list[FrameQualityOut] = Field(
        default_factory=list, description="采样帧的观测量，供信息科复核门限是否过严"
    )


class RoiBoxOut(BaseModel):
    """ROI 裁剪窗（**规范帧**坐标，不是相机原始像素）。"""

    model_config = ConfigDict(extra="forbid")

    x: int
    y: int
    width: int
    height: int


class FrameGeometryOut(BaseModel):
    """入站缩放结论：相机原始尺寸 ↔ 分析用规范帧尺寸。

    响应里的候选框、掩膜面积、面积比全部是**规范帧**口径；护士端要画在原始照片上时，
    乘 ``1/scale`` 换算回去即可（``FrameGeometry.to_camera_point`` 是这条换算的实现）。
    """

    model_config = ConfigDict(extra="forbid")

    camera_width: int = Field(description="相机原始宽度（像素）")
    camera_height: int = Field(description="相机原始高度（像素）")
    canonical_width: int = Field(description="分析用规范帧宽度（像素）")
    canonical_height: int = Field(description="分析用规范帧高度（像素）")
    scale: float = Field(description="缩放比 = 规范长边 / 原始长边；1.0 即未缩放")
    resized: bool = Field(description="本次上传是否真的缩放过")
    reason: str | None = Field(
        default=None,
        description=(
            "取值理由：long_edge_capped（按长边上限缩到位）/ short_edge_floor_wins"
            "（长宽比太极端，保短边）/ short_edge_below_floor（短边本就低于门限，不缩不放）"
            "/ disabled（配置关掉了缩放）；未缩放且短边达标时为空"
        ),
    )
    resized_frames: int = Field(description="1200 帧里有几帧被重写过（其余一个字节没碰）")
    total_frames: int = Field(description="本段总帧数")
    scale_mismatch_with_previous_round: bool = Field(
        default=False,
        description=(
            "上一轮会话的规范帧尺寸与本轮不同（多半是中途改过 FRAME_LONG_EDGE_MAX）。"
            "**如实告知而非静默**：跨轮几何比对此时不可靠，建议整院统一时段内不要改这个配置"
        ),
    )


class RoiOut(BaseModel):
    """ROI 窗口的运行记录（信息科可据此回答「这一段到底裁没裁」）。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(description="ROI 是否开启")
    applied: bool = Field(description="本段是否真的裁剪过（没候选框/窗太小会退全幅）")
    window: RoiBoxOut | None = Field(default=None, description="末态裁剪窗（规范帧坐标）")
    applied_calls: int = Field(description="有多少次预测走了裁剪后的画面")
    advances: int = Field(description="窗口被平滑了多少次")
    max_step_px: int = Field(description="单次平滑里窗口边缘的最大位移（像素）")
    border_touches: int = Field(
        description="掩膜贴到窗边的次数；不为 0 说明袋子被窗切过（系统改为只向外扩窗）"
    )
    anchor_width: int | None = Field(
        default=None, description="窗口初值宽（像素）。窗口**只涨不缩**，末态不小于它"
    )
    anchor_height: int | None = Field(
        default=None, description="窗口初值高（像素）。窗口**只涨不缩**，末态不小于它"
    )
    fallback_reason: str | None = Field(
        default=None,
        description=(
            "不开窗的理由：disabled / no_candidate_boxes（首帧没认出袋子）/ window_too_small / "
            "covers_full_frame（并集本来就是整幅画面，裁了等于没裁）"
        ),
    )


class TrackUploadResponse(BaseModel):
    """``POST /api/v1/infusion/tracks`` 的响应。"""

    model_config = ConfigDict(extra="forbid")

    upload_key: str = Field(description="本次上传的稳定标识（重传同一段不变）")
    rack_id: str
    bed_no: str
    infusion_order_no: str
    captured_at: str = Field(description="拍摄时刻（原样回显调用方给的 ISO8601 串）")
    shift_key: str = Field(description="巡视轮次键；未传时按 captured_at 的日期生成")
    frame_count: int
    first_frame: str = Field(description="首帧文件名")
    last_frame: str = Field(description="末帧文件名")
    frame_time_source: str = Field(
        description="逐帧时间的来源：frame_times 表单字段 或 zip 成员时间戳"
    )
    bags: list[BagSlotOut]
    candidates: list[CandidateOut] = Field(
        default_factory=list,
        description=(
            "首帧无提示枚举出的候选列表（护士在列表里认领，不用逐袋点像素）；"
            "低稳定分的候选已被剔除，不会出现在这里"
        ),
    )
    precheck: PrecheckOut | None = Field(
        default=None,
        description=(
            "上传前帧质量预检结论（整段一份）。分低**不**影响 bags 里的读数——"
            "拍得糊与查不出数是两回事，后者有独立门禁（无标定/无滴速/无观测）"
        ),
    )
    geometry: FrameGeometryOut | None = Field(
        default=None,
        description=(
            "入站缩放结论：相机原始尺寸与分析用规范帧尺寸。响应里的框与面积都是"
            "**规范帧**口径，换算回原始照片乘 1/scale"
        ),
    )
    roi: RoiOut | None = Field(
        default=None,
        description="ROI 裁剪窗的运行记录；退化时给出 fallback_reason 而不是假装裁了",
    )
    accepted_at: str


class HealthResponse(BaseModel):
    """``GET /api/v1/health``。"""

    model_config = ConfigDict(extra="allow")

    status: str
    service: str
    version: str
    schema_version: int
    database: str
    frames_root: str
    session_count: int


class ErrorIssueOut(BaseModel):
    """422 detail 里的一条问题：字段 + 原因 + 下一步怎么做。"""

    code: str = Field(description="机器可读细分码，如 frames_too_few")
    field: str
    reason: str
    suggestion: str


class ErrorDetailOut(BaseModel):
    """统一错误 detail。"""

    error_code: str
    message: str
    issues: list[ErrorIssueOut] = Field(default_factory=list)


def dump_model(model: BaseModel) -> dict[str, Any]:
    """统一序列化口径：``None`` 字段整体不出现（门禁语义）。"""
    return model.model_dump(exclude_none=True)


class ErrorCodeOut(BaseModel):
    """错误码表的一行（机器可读，供挂接方拉取后写分支）。"""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="错误码，如 INGEST_INVALID")
    status: int = Field(description="对应 HTTP 状态码")
    meaning: str = Field(description="这个错误码意味着什么")
    handling: str = Field(description="调用方下一步怎么处理")
    routes: list[str] = Field(
        default_factory=list,
        description="会返回这个码的接口，形如 POST /api/v1/infusion/tracks",
    )


class ErrorCodeListOut(BaseModel):
    """``GET /api/v1/error-codes`` 的响应。与 ``/docs`` 的 descriptions 同源。"""

    model_config = ConfigDict(extra="forbid")

    error_codes: list[ErrorCodeOut]
    operational_faults: list[dict[str, str]] = Field(
        default_factory=list,
        description="运行期故障（请求成功返回但链路不可用），只出现在运行摘要与启动日志",
    )


class CalibrationSampleIn(BaseModel):
    """一条标定样例：量到的面积比 + 已知的真实毫升数。"""

    model_config = ConfigDict(extra="forbid")

    measured_fill_ratio: float = Field(
        ge=0.0, le=1.0, description="量出来的液体占袋腔面积比"
    )
    known_volume_ml: float = Field(ge=0.0, description="对应的袋内真实毫升数")
    source: str = Field(
        default="unspecified",
        description="样例出处：称重法 / 换液确认反推 等，供护理部核对",
    )
    captured_at: str | None = Field(default=None, description="样例采集时刻 ISO8601")


class CavitySpecIn(BaseModel):
    """登记袋型腔体外形与残留参数（可选）。登记后毫升数走腔体几何换算。"""

    model_config = ConfigDict(extra="forbid")

    capacity_ml: float = Field(gt=0.0, description="标签标称容量（满袋）")
    cavity_height_mm: float = Field(default=200.0, gt=0.0, description="有效腔高（液柱能爬到的最高处）")
    dead_volume_ml: float = Field(default=0.0, ge=0.0, description="管路与滴斗残留（在袋腔之外）")
    meniscus_offset_ratio: float = Field(
        default=0.0, ge=0.0, lt=1.0, description="弯月面挂壁：量到的面积比高于真实高度比的量"
    )
    shape_exponent: float = Field(default=4.0, gt=1.0, description="横截面超椭圆指数，越大越接近矩形")
    shoulder_center: float = Field(default=0.45, ge=-1.0, le=1.0, description="肩部中心（归一化高度）")
    shoulder_half: float = Field(default=0.55, gt=0.0, le=1.0, description="肩部半宽（归一化高度）")
    taper: float = Field(default=0.20, ge=0.0, le=0.8, description="透视收窄：袋顶比袋底窄的比例")
    bulge: float = Field(default=0.055, ge=0.0, le=0.5, description="软袋鼓胀：充盈后下部被顶鼓的幅度")
    film_holdup_ml: float = Field(default=0.0, ge=0.0, description="袋壁挂液（永远流不出来的那部分）")
    geometry_revision: int | None = Field(
        default=None,
        ge=1,
        description=(
            "腔体几何的版本号。**不填按第 1 版登记**（首次登记即第 1 版）；"
            "改外形参数时必须升版，否则门禁会挡下毫升数——参数改了不升版本，"
            "谁也说不清历史读数是不是还成立"
        ),
    )
    label_zh: str = Field(default="", description="袋型中文名，护士端展示用")
    note: str = Field(default="", description="登记说明（外形参数从哪来的、留了什么偏差）")


class CalibrationCreateIn(BaseModel):
    """登记一个袋型的标定记录（新建或改版）。"""

    model_config = ConfigDict(extra="forbid")

    bag_type_code: str = Field(min_length=1, description="袋型编码，与输注单 slots 的 bag_type_code 对应")
    capacity_ml: float = Field(gt=0.0, description="标签标称容量（满袋）")
    dead_volume_ml: float = Field(
        default=0.0, ge=0.0, description="管路与滴斗残留（袋内测不到的部分）"
    )
    meniscus_offset_ratio: float = Field(
        default=0.0,
        description="弯月面挂壁补偿：液面爬在袋壁上，量到的面积比高于真实高度比",
    )
    samples: list[CalibrationSampleIn] = Field(
        default_factory=list, description="标定样例；为空时按「线性到满容量」兜底"
    )
    cavity: CavitySpecIn | None = Field(
        default=None,
        description=(
            "可选：袋型腔体外形与残留参数。登记后毫升数由腔体几何反解得出"
            "（estimation_mode=cavity），不登记则走分段带（band）"
        ),
    )
    calibrated_by: str = Field(min_length=1, description="标定人（工号或姓名），写进版本轨迹")
    calibrated_at: str | None = Field(default=None, description="标定时刻 ISO8601，默认取当前时间")
    note: str = Field(default="", description="标定说明（用的什么方法、留了什么偏差）")
    effective_from: str | None = Field(
        default=None,
        description=(
            "生效时刻（ISO8601）。留空 = 立刻生效（前一版同时收口）；"
            "填未来时刻 = 预约登记，当天算出来的毫升数一点不变"
        ),
    )
    effective_to: str | None = Field(
        default=None,
        description="停用时刻（ISO8601，开区间不含此刻）；留空表示仍然生效",
    )


class CalibrationOut(BaseModel):
    """标定记录的回执（含拟合出的分段带与版本信息）。"""

    model_config = ConfigDict(extra="forbid")

    calibration_id: str
    bag_type_code: str
    version: int = Field(description="该袋型的第几版；每次登记自增")
    superseded_version: int | None = Field(
        default=None, description="被本版取代的旧版本号；首次登记时不出现"
    )
    capacity_ml: float
    usable_capacity_ml: float
    dead_volume_ml: float
    meniscus_offset_ratio: float
    bands: list[list[float]] = Field(description="拟合出的分段线性带 [[面积比, 毫升], ...]")
    estimation_mode: str = Field(
        default="band", description="本版毫升数走哪条路：band=分段带，cavity=腔体几何换算"
    )
    cavity: dict | None = Field(
        default=None, description="登记的腔体外形参数快照；未登记时不出现"
    )
    sample_count: int
    max_relative_error: float = Field(description="样例最大相对偏差；超阈值需复核")
    calibrated_by: str
    calibrated_at: str
    note: str = ""
    active: bool
    recalculated_sessions: int = Field(
        default=0, description="本次改版触发重算的历史会话数"
    )
    # ── 版本与偏差闸门（task 27）───────────────────────────────────────
    version_status: str = Field(
        default="active",
        description="active=现在就用这一版；scheduled=已登记但还没到生效时刻",
    )
    effective_from: str | None = Field(default=None, description="本版生效时刻")
    effective_to: str | None = Field(default=None, description="本版停用时刻；仍在生效时不出现")
    superseded_to: str | None = Field(
        default=None, description="被本版取代的旧版,其停用时刻已收到这个值"
    )
    review_required_sessions: int = Field(
        default=0, description="偏差超阈值未自动覆盖、仍在等复核的会话数"
    )
    review_threshold: float = Field(
        default=0.15, description="本次采用的相对差复核阈值（可由环境变量覆盖）"
    )

class ReviewResolveIn(BaseModel):
    """人工复核一条改版偏差告警的入参。"""

    model_config = ConfigDict(extra="forbid")

    resolution: Literal["applied", "dismissed"] = Field(
        description=(
            "applied=复核通过，按新版本的读数落库；"
            "dismissed=复核否决（多半是新版本有问题），旧读数继续沿用"
        )
    )
    reviewed_by: str = Field(min_length=1, description="复核人（工号或姓名），写进审计链")
    note: str = Field(default="", description="复核结论说明，供护理部与内审留档")


class ClaimIn(BaseModel):
    """在候选列表里认领一袋（或改绑到别的候选）。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, description="上传响应里 bags[].session_id")
    candidate_id: str = Field(min_length=1, description="要认领的候选 ID（取自响应里的 candidates）")
    operator_id: str = Field(
        min_length=1, description="确认护士的工号；写进 append-only 审计链，可追溯到人"
    )
    note: str = Field(default="", description="认领备注（可选，如「这个是加药的袋子」）")


class ClaimOut(BaseModel):
    """认领后的这一袋：曲线与体积都是**按新候选重算**的结果。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    bag_seq: str
    slot_index: int
    candidate_id: str
    bbox: list[int] | None = Field(default=None, description="认领候选的外接框 [x0,y0,x1,y1]")
    claim_state: str = Field(description="claimed（首次认领）/ corrected（认领后改绑）")
    corrected: bool = Field(default=False, description="是否属于改绑（此前已认领过别的候选）")
    claimed_by: str
    frame_count: int = Field(description="重算时使用的帧数")
    area_ratio_series: list[float | None] = Field(
        default_factory=list, description="按认领候选重算的逐帧液面；被遮挡帧为 null"
    )
    area_ratio_smoothed: list[float | None] = Field(
        default_factory=list,
        description="按新候选重算后的平滑工作序列（口径与上传响应一致：反常帧保留原值并被点名）",
    )
    smoothing: SmoothingOut | None = Field(
        default=None, description="平滑诊断：尖峰/反常/重建帧、缺帧处理原因、平均降幅"
    )
    fill_ratio: float | None = None
    estimated_volume_ml: float | None = Field(
        default=None, description="无生效标定时不出现（门禁语义与上传响应一致）"
    )
    calibration_version: int | None = None
    eta_minutes: int | None = None
    eta_source: str | None = None
    pickup_suppressed: bool = True
    suppression_reason: str | None = None
    volume_uncertainty: UncertaintyOut | None = Field(
        default=None,
        description="认领后重算的可信区间与置信结论；与接入主路径同一套门限，不因认领换口径",
    )
    descriptor_refreshed: bool = Field(
        default=False,
        description="外观描述子是否同步刷新；false 表示该候选处已看不到液体，下一轮不会自动接上",
    )
    decided_at: str


class DecisionIn(BaseModel):
    """重算某一袋的换液点名判定。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, description="要判定的袋级会话 ID")


class DecisionOut(BaseModel):
    """判定结果。``changed=false`` 表示维持原状——这也是绝大多数轮次的结果。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    bag_seq: str
    bed_no: str | None = None
    state: str = Field(description="判定后的业务状态：monitoring / needs_change")
    changed: bool
    trigger_reason: str | None = Field(
        default=None,
        description="触发原因码：eta_within_window / volume_below_margin / near_empty_observed；"
        "不触发时不出现",
    )
    decision: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "判定明细：连续帧数、当前液面、阈值、余量判定种类与数值、"
            "被压住时的原因说明"
        ),
    )
    estimated_volume_ml: float | None = None
    eta_minutes: int | None = None
    pickup_suppressed: bool = True
    decided_at: str


class PickupItemOut(BaseModel):
    """点名清单里的一行。**它是一条建议**——未确认换液不出站。"""

    model_config = ConfigDict(extra="forbid")

    bag_seq: str = Field(description="袋序标识，与上传响应 bags[].bag_seq 一致")
    session_id: str = Field(description="袋级会话 ID；确认换液时用它定位")
    state: str = Field(description="恒为 needs_change：只有这个状态会出现在点名清单里")
    rack_id: str | None = Field(default=None, description="输液架号")
    bed_no: str | None = Field(default=None, description="床号；护士按床号核对点名结果")
    slot_index: int = Field(default=0, description="输液架上的挂点序号，从 0 开始")
    infusion_order_no: str | None = Field(default=None, description="输注单号（权威标识由对端给）")
    bag_type_code: str | None = Field(default=None, description="袋型编码，未登记时为空")
    calibration_state: str = Field(
        description="标定状态：uncalibrated / calibrated / low_confidence / high_occlusion"
    )
    trigger_reason: str | None = Field(
        default=None,
        description="点名原因码：eta_within_window / volume_below_margin / near_empty_observed",
    )
    eta_minutes: int | None = Field(
        default=None, description="预计耗尽分钟数；取不到滴速时**不出现**（不给猜的时点）"
    )
    fill_ratio: float | None = Field(default=None, description="末次可见帧的液面面积比")
    estimated_volume_ml: float | None = Field(
        default=None,
        description="余量毫升数；**无生效标定记录时不出现**，改看 evidence_note 里的降级说明",
    )
    confidence: float | None = Field(
        default=None, description="置信度 0–1；完全没测到时**不出现**（不是 0）"
    )
    confidence_source: str = Field(
        description="置信度出处：volume（有标定，取体积区间结论）/ observation（无毫升数，取观测级）"
    )
    confidence_factors: dict[str, Any] = Field(
        default_factory=dict,
        description="观测级置信的可复算因子：可见帧数/总帧数、遮挡占比、覆盖度、低液位惩罚",
    )
    evidence_note: str = Field(
        default="",
        description=(
            "点名依据的人话说明。未标定袋型时逐字给出「依据面积比，未标定袋型 …」，"
            "而不是给一个假的毫升数"
        ),
    )
    evidence_basis: list[str] = Field(
        default_factory=list,
        description="依据清单：area_ratio 恒在，calibration / rate_telemetry 按门禁增减",
    )
    client_event_id: str = Field(
        description="确认换液时用的幂等键，由服务端按 (session_id, bag_seq) 固定生成"
    )
    writeback_queued: bool = Field(
        default=False, description="恒为 false：未确认的点名绝不进入写回队列"
    )
    writeback_hold_reason: str = Field(default="", description="为什么还压着不出站")
    confirm_hint: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "护士端确认换液时该原样回填的入参（session_id / bag_seq / client_event_id）；"
            "直接回填就不会因为终端自编键而把一次换液拆成多条事件"
        ),
    )
    captured_at: str | None = Field(default=None, description="本轮短片的拍摄时刻")
    updated_at: str | None = Field(default=None, description="会话最近一次写入时刻")
    rank: int = Field(default=0, description="本清单内的次序，1 起；按 ETA 升序 → 床号 → 袋序")


class PickupListOut(BaseModel):
    """病区换液点名清单（只读）。"""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(description="点名条数")
    bed_count: int = Field(description="涉及床位数")
    generated_at: str = Field(description="清单生成时刻")
    items: list[PickupItemOut] = Field(default_factory=list, description="点名条目，已排好序")
    writeback_queued_count: int = Field(
        default=0, description="恒为 0：清单这一层不产生任何写回动作"
    )
    hold_reason: str = Field(default="", description="未确认不出站的说明")
    without_ml_count: int = Field(
        default=0, description="其中因未标定/被门禁压住而给不出毫升数的条数"
    )


class PendingItemOut(BaseModel):
    """待写回队列里的一笔欠账。**不含患者可识别明细**，只有架号、床号与袋序。"""

    model_config = ConfigDict(extra="forbid")

    queue_id: str = Field(description="队列行 ID（内部用，重放按 event_id 定位）")
    event_id: str = Field(description="出站事件 ID；重放时用它定位")
    client_event_id: str = Field(description="幂等键：与首次写回完全相同，对端据此去重")
    status: str = Field(
        description="欠账状态：queued（等自动重试）/ dead_letter（自动重试耗尽，等人）"
        "/ failed（对端业务拒绝，等人）"
    )
    attempt_count: int = Field(description="已尝试次数（含首次）")
    session_id: str | None = Field(default=None, description="袋级会话 ID")
    rack_id: str | None = Field(default=None, description="输液架号")
    bed_no: str | None = Field(default=None, description="床号")
    bag_seq: str | None = Field(default=None, description="袋序")
    next_retry_at: str | None = Field(default=None, description="下一次可自动重试时刻")
    last_error: str | None = Field(default=None, description="最后一次错误摘要（不含凭据）")
    dead_letter_reason: str | None = Field(
        default=None, description="死信原因码：retries_exhausted / business_rejected"
    )
    last_attempt_at: str | None = Field(default=None, description="最后一次尝试时刻")
    created_at: str | None = Field(default=None, description="入队时刻")
    updated_at: str | None = Field(default=None, description="最后更新时刻")
    replayable: bool = Field(description="是否可由信息科手动重放")


class PendingListOut(BaseModel):
    """待写回清单（只读）。"""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(description="欠账条数")
    generated_at: str = Field(description="清单生成时刻")
    by_status: dict[str, int] = Field(default_factory=dict, description="按状态计数")
    dead_letter_count: int = Field(default=0, description="自动重试耗尽、等人处理的条数")
    queued_count: int = Field(default=0, description="等自动重试的条数")
    failed_count: int = Field(default=0, description="被对端业务拒绝的条数")
    items: list[PendingItemOut] = Field(default_factory=list, description="欠账条目")


class PendingReplayIn(BaseModel):
    """人工重放的入参。"""

    model_config = ConfigDict(extra="forbid")

    operator_id: str = Field(
        min_length=1,
        description=(
            "执行重放的信息科人员标识（工号/账号）；**必填**——"
            "「谁把这条欠账补送出去的」必须能追溯到人"
        ),
    )


class PendingReplayOut(BaseModel):
    """重放结果。``idempotent_hit=true`` 表示这笔早已送达，没有再发一次。"""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(description="被重放的出站事件 ID")
    client_event_id: str = Field(description="幂等键；与首次写回完全相同")
    found: bool = Field(description="队列里是否存在这一笔")
    replayed: bool = Field(description="本次是否真的发出了一次写回请求")
    idempotent_hit: bool = Field(description="true=早已送达，本次没有再发（幂等命中）")
    writeback_status: str = Field(
        description="重放后的写回状态：sent / pending_writeback / failed"
    )
    ok: bool = Field(description="重放是否送达")
    kind: str = Field(default="ok", description="最后一次尝试的结果分类")
    attempts: int = Field(default=0, description="本次尝试次数（人工重放只发一次）")
    reason: str | None = Field(default=None, description="被拒发或失败的原因")
    last_error: str | None = Field(default=None, description="错误摘要（不含凭据）")
    queue_status: str | None = Field(default=None, description="重放后的队列状态")
    queue_attempt_count: int = Field(default=0, description="重放后的累计尝试次数")


class TransitionIn(BaseModel):
    """护士确认换液的入参。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, description="点名清单 items[].session_id")
    bag_seq: str = Field(
        min_length=1, description="被换掉的那一袋；与会话对不上直接 403（越权写入）"
    )
    operator_id: str = Field(
        min_length=1, description="确认护士的工号；**必填**——没有护士确认就没有换液事件"
    )
    client_event_id: str | None = Field(
        default=None,
        description=(
            "可选。原样回填点名清单里的 client_event_id 即可；留空由服务端按 "
            "(session_id, bag_seq) 生成。给了但与规则不一致 → 422，不按调用方的键入账"
        ),
    )
    rack_id: str | None = Field(
        default=None, description="可选。填了就做跨输液架防护（对不上直接 403）"
    )
    confirmed_at: str | None = Field(
        default=None, description="可选，确认时刻（ISO8601）；不传按服务端当前时刻"
    )


class WritebackOut(BaseModel):
    """出站写回的结果。**失败也如实返回**，绝不静默丢弃事件。"""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        description="写回状态：sent（对端已收）/ pending_writeback（退避耗尽，等重放）/ "
        "failed（业务拒绝，不重试）/ pending（还没发）"
    )
    ok: bool
    attempts: int = Field(default=0, description="实际尝试次数（首次 + ≤2 次重试）")
    kind: str = Field(default="ok", description="最后一次尝试的结果分类")
    refused: bool = Field(default=False, description="是否因出站门禁被拒发")
    reason: str | None = Field(default=None, description="被拒发或失败的原因")
    next_retry_at: str | None = Field(default=None, description="下一次重试时刻；已送达时不出现")


class TransitionOut(BaseModel):
    """确认结果。``idempotent_hit=true`` 表示这是重放，没有产生新事件。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    bag_seq: str
    state: str = Field(description="确认后的状态：replaced（终态）")
    operator_id: str = Field(description="确认护士工号，写进审计链可追溯到人")
    client_event_id: str = Field(description="出站幂等键；重放时与首次完全相同")
    event_id: str | None = Field(default=None, description="出站事件 ID")
    created: bool = Field(description="本次是否真的产生了一条新事件")
    idempotent_hit: bool = Field(description="true=同一次确认的重放，未新增事件")
    confirmed_at: str = Field(description="确认时刻（ISO8601）")
    pickup_evidence: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "点名依据留痕：确认前的状态、触发原因码、ETA 与液面。"
            "确认会把 trigger_reason 覆盖成 nurse_confirmed，所以这里先留一份"
        ),
    )
    writeback: WritebackOut | None = Field(
        default=None, description="出站写回结果；退避耗尽时为 pending_writeback 等重放"
    )


# ── 分片续传与断点（task 32）──────────────────────────────────────────────


class UploadOpenOut(BaseModel):
    """``POST /api/v1/infusion/uploads`` 的响应：拿到 upload_id 才能续传。"""

    model_config = ConfigDict(extra="forbid")

    upload_id: str = Field(description="这次续传上传的标识；分片按它归集")
    status: str = Field(description="pending（开着等分片）/ completed / abandoned")
    session_key: str = Field(description="提交后帧将落到的目录名，与 /tracks 的 upload_key 同规则")
    expected_frames: int | None = Field(
        default=None, description="本次申报的总帧数；空表示终端没报，断点清单为空"
    )
    received_frames: list[int] = Field(default_factory=list, description="已收帧号（刚开启时为空）")
    missing_frames: list[int] = Field(default_factory=list, description="还缺的帧号")
    expires_at: str = Field(description="保留期到点即被清理策略回收")


class UploadChunkAckOut(BaseModel):
    """``POST …/{upload_id}/chunks`` 的响应：这一片收下哪些、丢了几帧重复。"""

    model_config = ConfigDict(extra="forbid")

    upload_id: str
    accepted_frames: list[int] = Field(description="本次真正落盘的帧号")
    duplicates_dropped: list[int] = Field(
        description="因帧号重复被丢弃的帧号（先到先得，后到的这一份不覆盖）"
    )
    duplicate_count_total: int = Field(description="本次上传累计去重丢掉的帧数")
    received_frames: list[int] = Field(description="截至本次的已收帧号全集")
    received_count: int
    missing_frames: list[int] = Field(description="还缺哪些帧；申报了总帧数时才有内容")
    added_bytes: int = Field(description="本次落盘的字节数")
    expires_at: str


class UploadStatusOut(BaseModel):
    """``GET …/{upload_id}`` 的响应：断点查询。"""

    model_config = ConfigDict(extra="forbid")

    upload_id: str
    status: str
    rack_id: str
    shift_key: str
    session_key: str
    expected_frames: int | None = None
    received_frames: list[int] = Field(default_factory=list)
    received_count: int = 0
    missing_frames: list[int] = Field(default_factory=list)
    duplicate_count: int = 0
    chunk_count: int = Field(default=0, description="收到过的分片数（含全重复的空片）")
    session_ids: list[str] = Field(
        default_factory=list, description="提交后产出的袋级会话（未提交时为空）"
    )
    created_at: str
    updated_at: str
    expires_at: str
    completed_at: str | None = None


class SessionVersionOut(BaseModel):
    """整段重放前的旧版本快照（``bag_session_history`` 的一行）。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    session_version: int
    state: str | None = None
    trigger_reason: str | None = None
    captured_at: str | None = None
    frame_count: int = 0
    fill_ratio: float | None = None
    estimated_volume_ml: float | None = None
    eta_minutes: int | None = None
    calibration_state: str | None = None
    calibration_version: int | None = None
    area_ratio_series: list[float | None] = Field(default_factory=list)
    archived_at: str = Field(description="被覆盖的时刻（ISO8601）")
    archived_reason: str = Field(default="replay", description="replay=同架同轮次整段重放")


class SessionVersionsOut(BaseModel):
    """``GET …/sessions/{id}/versions``：当前版本 + 全部被盖掉的旧版本。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    current_version: int = Field(description="bag_session 上的当前版本号")
    archived_count: int = Field(description="被盖掉的旧版本数（0 = 这一袋还没重放过）")
    versions: list[SessionVersionOut] = Field(default_factory=list)


class CleanupResultOut(BaseModel):
    """``POST …/uploads/cleanup`` 的响应：半成品回收结论。"""

    model_config = ConfigDict(extra="forbid")

    expired_uploads: list[str] = Field(default_factory=list)
    expired_count: int = 0
    freed_directories: list[str] = Field(default_factory=list)
    freed_bytes: int = 0
    kept_completed: int = Field(description="已完成的上传数：一个都没动")
    dry_run: bool = Field(default=False, description="true = 只报不删")


class AuditFilterOut(BaseModel):
    """审计包导出条件回显（让人知道这份包到底筛了什么）。"""

    model_config = ConfigDict(extra="forbid")

    rack_id: str | None = None
    bed_no: str | None = None
    occurred_from: str | None = None
    occurred_to: str | None = None
    actions: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    bag_seq: str | None = None
    limit: int | None = None


class AuditManifestOut(BaseModel):
    """审计包首页：护理部打印签收、内审核对摘要用。"""

    model_config = ConfigDict(extra="forbid")

    generated_at: str = Field(description="导出时刻")
    scope_zh: str = Field(description="导出范围的人话描述")
    scope_key: str = Field(description="范围标识（架号或床号；全量时为『全病区』）")
    row_count: int = Field(description="本次导出的审计行数")
    window: dict[str, str | None] = Field(description="覆盖的时间窗：首行与末行的发生时刻")
    by_category: dict[str, int] = Field(description="七类动作各多少条（other = 其余动作）")
    category_labels_zh: dict[str, str] = Field(description="类别键的中文名")
    by_action: dict[str, int] = Field(description="按动作码的分布")
    chain_digest: str = Field(description="链摘要：留档人重算一遍，对不上说明审计被动过")
    digest_algorithm: str = Field(description="摘要算法说明")
    note_zh: str = Field(default="", description="留档须知")
    filters: AuditFilterOut = Field(description="本包用的过滤条件")


class TelemetryWebhookOut(BaseModel):
    """入站 webhook 兜底通道的响应（一条遥测一行的结论）。

    ``accepted=true`` 且 ``duplicate=false`` 才是「真的收下了」；重复的那条回
    ``duplicate=true`` 且 ``accepted=false``——两条都不是错误，返回码都是 202。
    """

    model_config = ConfigDict(extra="allow")

    accepted: bool = Field(description="这条遥测是否作为新读数进了缓存")
    duplicate: bool = Field(
        description=(
            "是否被去重表判为重复（同一 device_sn+ts，或同架同袋的更旧时间戳）。"
            "重连重投与跨通道重复都走这一条"
        )
    )
    reason: str | None = Field(
        default=None, description="未接受时的原因码；接受时字段不出现"
    )
    rack_id: str | None = Field(default=None, description="输液架标识（取自 URL 路径段）")
    bag_seq: str | None = Field(default=None, description="袋序标识")
    pump_rate_ml_per_h: float | None = Field(default=None, description="泵当前滴速（ml/h）")
    device_sn: str | None = Field(default=None, description="输液泵序列号；载荷没带时不出现")
    ts: str | None = Field(default=None, description="泵上报的时间戳；没带时不出现")
    source: str | None = Field(default=None, description="通道标识，恒为 webhook")
    timestamp_enforced: bool | None = Field(
        default=None, description="是否校验了重放窗口（请求带了时间戳头才为 true）"
    )
