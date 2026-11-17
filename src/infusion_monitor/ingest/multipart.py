"""入站帧序列接入：``POST /api/v1/infusion/tracks``（multipart/form-data）。

这是**主干最窄的一条端到端路径**：一次上传 → 校验 → 解包 → 落袋级会话 →
逐帧传播 → 袋型标定换算 → 返回可核对的 JSON（``upload_key`` / ``frame_count`` /
每袋 ``session_id``、``area_ratio_series`` 与标定后的 ``estimated_volume_ml`` /
``eta_minutes``）。点名与确认（task 9/31 起）都挂在这条路径上，不另起第二条推理路径。

**帧 zip 按块落临时文件再解包**，不整包读进内存——1200 帧时峰值内存不随帧数线性膨胀。
**边收边量**（:func:`save_upload_to_temp` 的 ``budget``）：超字节/超时限立刻退回并指向
分片续传。**落盘即规范化**（:mod:`.resize`）：解包写每一帧时顺手把长边缩到上限，短边有
下限保护，缩放比例随响应与 ``bag_session`` 落库。**只把输液架那块喂给预测器**
（:class:`~.resize.WindowedPredictor`）：ROI 窗口随帧平滑，掩膜照旧贴回整幅规范帧。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, UploadFile

from ..api.schemas import (
    BagSlotOut,
    CandidateOut,
    FrameGeometryOut,
    FrameQualityOut,
    PrecheckOut,
    QualityFlagOut,
    RoiBoxOut,
    RoiOut,
    SessionVersionOut,
    SessionVersionsOut,
    SmoothingOut,
    TrackUploadResponse,
    UncertaintyOut,
)
from ..adapters.mqtt_telemetry import classify_sample
from ..calibration.estimator import estimate_bag, resolve_rate
from ..calibration.uncertainty import ObservationQuality
from ..core.enumerate import enumerate_candidates
from ..core.failure_modes import classify_capture, sample_indices
from ..core.multi_object import (
    TrackAnchor,
    assign_slots,
    propagate_batch,
    slot_columns,
)
from ..core.predictor import Prompt, mask_bbox
from ..core.occlusion import handoff_to_dict
from ..core.propagate import load_rgb, slot_seed_prompt
from ..core.smoothing import smooth_series
from ..errors import ApiError, Issue, ValidationRejected
from ..identity.reanchor import (
    SlotObservation,
    describe_region,
    descriptor_from_session,
    reanchor_slots,
)
from ..identity.state_machine import apply_decision
from ..gateway.audit import DECISION_SOURCES
from ..gateway.fusion import FusionParams, fuse_round
from ..logging_utils import request_id_of
from ..store.repositories import new_id, utc_now
from .quality import precheck_frames
from .resize import (
    FrameGeometry,
    ResizePlan,
    RoiTracker,
    UploadBudget,
    WindowedPredictor,
    resize_report_for_frames,
)
from .validate import (
    SlotSpec,
    UploadMeta,
    collect_frame_issues,
    parse_frame_times,
    parse_upload_fields,
)
from .zip_frames import extract_frames, upload_key

__all__ = ["router", "save_upload_to_temp", "process_track_upload"]

logger = logging.getLogger("infusion_monitor.ingest")

router = APIRouter(prefix="/api/v1/infusion", tags=["ingest"])

_COPY_CHUNK = 1024 * 1024


def _confidence_of(estimate: Any) -> float | None:
    """体积置信度；没有算过不确定度（例如整段没观测）时为 ``None``（不是 0）。"""
    return estimate.uncertainty["confidence"] if estimate.uncertainty else None


def _interval_of(estimate: Any, key: str) -> Any:
    """从不确定度结论里取一个字段；没有结论时为 ``None``。"""
    return estimate.uncertainty.get(key) if estimate.uncertainty else None


def _fuse_round(
    state: Any,
    records: list[dict[str, Any]],
    *,
    picked_bags: set[str],
    request_id: str,
) -> dict[str, Any]:
    """可选的决策融合仲裁（默认不启用），返回 ``{bag_seq: FusedConclusion}``。

    **默认关闭**（``LLM_FUSION_ENABLED=false``）：返回空字典，主路径一次 HTTP
    都不发。这不是省事——护士上传短片到看到点名之间只隔一次巡视的间隔，
    让每一轮都去问一次模型，会把端点的抖动直接变成护士侧的等待。开启后模型
    也只贡献措辞（见 :mod:`gateway.fusion`），关掉不影响任何结论。
    """
    gateway = getattr(state, "model_gateway", None)
    settings = getattr(state, "settings", None)
    if gateway is None or settings is None or not getattr(settings, "llm_fusion_enabled", False):
        return {}
    store = getattr(state, "store", None)
    return fuse_round(
        gateway,
        [item for item in records if str(item.get("bag_seq") or "") in picked_bags],
        request_id=request_id,
        params=FusionParams.from_settings(settings),
        audit=getattr(store, "audit_chain", None),
    )


async def save_upload_to_temp(
    upload: UploadFile, directory: Path, budget: UploadBudget | None = None
) -> Path:
    """把上传的 zip 按块写到临时文件，避免整包进内存。

    ``budget`` 给了就**边收边量**：超字节上限报 413、超时限报 408，两者都指向分片
    续传（task 32 的 ``POST /api/v1/infusion/uploads``）。病区网络断在传输中途是常态，
    那一刻最该做的是立刻退回并告诉护士「改走分片，已收到的分片不受影响」，而不是
    把大包继续读进内存再慢慢超时。
    """
    directory.mkdir(parents=True, exist_ok=True)
    handle, raw_path = tempfile.mkstemp(prefix="upload-", suffix=".zip", dir=str(directory))
    temp_path = Path(raw_path)
    started = time.monotonic()
    received = 0
    try:
        with os.fdopen(handle, "wb") as sink:
            while True:
                chunk: bytes = await upload.read(_COPY_CHUNK)
                if not chunk:
                    break
                sink.write(chunk)
                received += len(chunk)
                if budget is not None:
                    budget.check(received, time.monotonic() - started)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _parse_times(zip_times: tuple[datetime, ...], raw_frame_times: Any):
    """优先用调用方显式给的逐帧时间；否则退回 zip 成员时间戳。"""
    explicit = parse_frame_times(raw_frame_times, len(zip_times))
    if explicit:
        parsed = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in explicit]
        return tuple(parsed), "frame_times"
    return zip_times, "zip_member"


@router.post(
    "/tracks",
    response_model=TrackUploadResponse,
    response_model_exclude_none=True,
    status_code=201,
    summary="上传输液架短片帧序列",
    description=(
        "移动护理终端把一轮巡视拍到的帧序列打包成 zip 上传。服务端做硬校验"
        "（帧数 / 分辨率 / 时间序 / 袋位上限），落袋级会话后逐帧传播，返回每袋的 "
        "area_ratio_series 与被遮挡帧标记。不合格时返回 422，detail 里逐条给出字段、"
        "原因与补拍建议。"
    ),
)
async def upload_tracks(
    request: Request,
    rack_id: Annotated[str, Form(description="输液架编号，与护理执行平台登记一致")],
    infusion_order_no: Annotated[str, Form(description="输注单号（权威标识由对端给出）")],
    bed_no: Annotated[str, Form(description="床号")],
    captured_at: Annotated[str, Form(description="拍摄时刻，ISO8601")],
    slots: Annotated[
        str,
        Form(
            description=(
                'JSON 数组，如 [{"bag_seq":"B01","slot_index":0,"bag_type_code":"TYPE-A",'
                '"rate_text":"5ml/h"}]；bag_type_code 决定能否换出毫升数，'
                "rate_text 是输注医嘱里的滴速（ETA 的第一优先来源）"
            )
        ),
    ],
    frames: Annotated[UploadFile, File(description="帧 zip，成员命名为 frame_00000.jpg 起连续")],
    client_event_id: Annotated[str | None, Form(description="幂等键（可选）")] = None,
    shift_key: Annotated[str | None, Form(description="巡视轮次键（可选，默认取拍摄日期）")] = None,
    frame_times: Annotated[
        str | None, Form(description="与帧数等长的 ISO8601 时间数组 JSON（可选）")
    ] = None,
) -> TrackUploadResponse:
    state = request.app.state
    meta: UploadMeta = parse_upload_fields(
        {
            "rack_id": rack_id,
            "infusion_order_no": infusion_order_no,
            "bed_no": bed_no,
            "captured_at": captured_at,
            "slots": slots,
            "client_event_id": client_event_id,
            "shift_key": shift_key,
        }
    )

    spool_root = Path(state.upload_spool)
    temp_path = await save_upload_to_temp(frames, spool_root, state.settings.upload_budget)
    try:
        if temp_path.stat().st_size == 0:
            raise ValidationRejected(
                [
                    Issue(
                        code="frames_empty_file",
                        field="frames",
                        reason="上传的 frames 文件为 0 字节",
                        suggestion="上传过程中断断了。请检查病区网络后重新上传这一段短片。",
                    )
                ]
            )

        key = upload_key(meta.rack_id, meta.infusion_order_no, meta.shift_key)
        frames_dir = Path(state.settings.frames_root) / key
        extracted = extract_frames(
            temp_path, frames_dir, resize_policy=state.settings.resize_policy
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return process_track_upload(
        state=state, request=request, meta=meta, extracted=extracted, frame_times=frame_times
    )


def process_track_upload(
    *,
    state: Any,
    request: Request,
    meta: UploadMeta,
    extracted: Any,
    frame_times: str | None = None,
) -> TrackUploadResponse:
    """一次接入的**唯一**主路径：校验 → 落袋级会话 → 分割 → 标定 → 换液点名。

    ``POST /tracks``（整段上传）与 ``POST /uploads/{id}/commit``（分片续传提交）
    都进这一个函数：分片模式不是「简化版主路径」，它只是把「帧从哪来」这一步换了
    种方式，后面的每一个数字仍出自同一条链路。传进来已解包的 ``extracted``
    （:class:`ingest.zip_frames.ExtractedFrames`），本函数负责硬校验与后续全部处理。
    """
    times, time_source = _parse_times(extracted.times, frame_times)
    collect_frame_issues(
        names=extracted.names, times=times, paths=extracted.paths
    )
    key = upload_key(meta.rack_id, meta.infusion_order_no, meta.shift_key)
    frames_dir = Path(extracted.paths[0]).parent
    request_id = request_id_of(request)

    sessions = state.store.sessions
    bags: list[BagSlotOut] = []
    #: 换液点名的输入（整轮收尾时统一判定，见循环后的 detect_round 调用）
    detection_inputs: list[dict[str, Any]] = []
    records: list[dict] = []

    def _archive_previous(session_id: str, version: int, snapshot: dict) -> None:
        """整段重放盖掉旧版本时的审计行（与快照同事务，见 repositories.upsert_session）。"""
        state.store.audit_chain.append(
            request_id=request_id,
            actor="client:nursing-terminal",
            action="session.version_archived",
            target=session_id,
            outcome="accepted",
            bag_seq=snapshot.get("bag_seq"),
            from_state=snapshot.get("state"),
            detail={
                "archived_version": version,
                "new_version": version + 1,
                "reason": snapshot.get("archived_reason") or "replay",
                "previous_fill_ratio": snapshot.get("fill_ratio"),
                "previous_volume_ml": snapshot.get("estimated_volume_ml"),
                "previous_eta_minutes": snapshot.get("eta_minutes"),
                "previous_frame_count": snapshot.get("frame_count"),
                "captured_at": snapshot.get("captured_at"),
                "archived_at": snapshot.get("archived_at"),
            },
        )

    for slot in meta.slots:
        record = sessions.upsert_session(
            {
                "rack_id": meta.rack_id,
                "bed_no": meta.bed_no,
                "infusion_order_no": meta.infusion_order_no,
                "shift_key": meta.shift_key,
                "bag_seq": slot.bag_seq,
                "slot_index": slot.slot_index,
                "bag_type_code": slot.bag_type_code,
                "frames_dir": str(frames_dir),
                "frame_count": extracted.frame_count,
                "captured_at": meta.captured_at,
                "state": "monitoring",
                "calibration_state": "uncalibrated",
                "client_event_id": meta.client_event_id,
            },
            on_version_archived=_archive_previous,
        )
        records.append(record)

    # 主路径内核：首帧无提示枚举 → 起锚 → 逐帧传播 → 每袋一条面积比曲线。
    #
    # 起锚优先用**枚举出来的候选**：护士不用逐袋点像素，程序把画面里可能是袋子的
    # 区域找出来，唯一命中的槽位直接据此起锚；认不准的槽位才退回挂点顺序的
    # 槽位几何先验，并在响应里标 needs_claim 等护士认领。
    first_frame = load_rgb(extracted.paths[0])
    frame_shape = (int(first_frame.shape[0]), int(first_frame.shape[1]))

    # 帧几何（task 35）：相机原始尺寸 ↔ 分析用规范帧尺寸。缩放在解包落盘时就做完了，
    # 这里只把结论取出来——它决定响应里的框怎么换算回护士手机上的原始照片，也决定
    # 「上一轮和这一轮是不是同一个坐标系」。分片续传那条路把收第一片时的结论带了回来
    # （ingest/resume.py）；实在拿不到（例如缩放策略上线前开着的半成品）就按盘上尺寸
    # 如实报，camera 与 canonical 一致，**不**反推一个假的原始尺寸。
    resize_report = extracted.resize or resize_report_for_frames(extracted.paths)

    enumeration = enumerate_candidates(
        first_frame,
        state.predictor,
        slot_count=len(meta.slots),
    )
    bound_candidates = enumeration.bound_by_slot

    seeds: list[tuple[str, Prompt]] = []
    bound_by_slot: dict[int, str] = {}
    for slot in meta.slots:
        candidate = bound_candidates.get(slot.slot_index)
        if candidate is not None:
            seeds.append(
                (
                    slot.bag_seq,
                    Prompt.from_point(candidate.centre_x, candidate.centre_y),
                )
            )
            bound_by_slot[slot.slot_index] = candidate.candidate_id
        else:
            seeds.append(
                (
                    slot.bag_seq,
                    slot_seed_prompt(first_frame.shape, slot.slot_index, len(meta.slots)),
                )
            )
    prompt_by_seq = {bag_seq: prompt for bag_seq, prompt in seeds}

    # ROI 窗口：把**首帧枚举出的全部候选框**与**每一只袋子在起锚帧上真正看到的掩膜框**
    # 并起来、外放余量，裁进画面。画面里其余位置（隔壁床、输液杆）同样是液体色，
    # 不框住就是让几何核去那儿也认几只袋子。
    #
    # 两条取值纪律，都是「宁可把窗开大，不可把窗开小」：
    #
    # ① **候选一律取，不分认领与否**。窗口开大只多算几个像素，开小会出事：认不出候选
    #    的槽位退回槽位几何先验起锚，先验点常落在两只袋子之间的空档上，一旦窗外那只袋子
    #    只剩一条贴着窗边的窄缝，几何核 nearest-component 就会锁到那条缝上（实测面积比
    #    从 0.59 一路歪到 0.02 再爬回 0.54）。误认一个候选最多让窗口接近全幅——退回
    #    「不裁」，安全；认漏一只则直接错数。
    # ② **并上起锚帧真实掩膜**。认不出候选的槽位（液体柱高、或与相邻袋粘连）其袋体可能
    #    比任何候选框都高，窗口按候选框一开，那只袋子的液体柱上半截就落在窗外，前几帧的
    #    面积比被削掉一大截，等窗口反应过来长回去，曲线开头已经歪了。这里按起锚提示在
    #    **整幅规范帧**上各问一次预测器：每袋一次、只做起锚帧，比逐帧裁剪省下的算力还多。
    roi_tracker = RoiTracker(state.settings.roi_params)
    roi_boxes = [candidate.bbox for candidate in enumeration.candidates]
    for _, prompt in seeds:
        for prediction in state.predictor.predict(first_frame, prompt) or ():
            box = mask_bbox(prediction.mask)
            if box is not None:
                roi_boxes.append(box)
    roi_tracker.initial_window(roi_boxes, frame_shape)
    # 装饰器只在裁剪时换掉喂进去的画面；掩膜照旧贴回整幅规范帧，面积比、槽位几何、
    # 候选框、跨会话重锚全都在同一个坐标系里（协议第 2 条不被破窗）。
    predictor = WindowedPredictor(state.predictor, roi_tracker)

    # 拍摄质量用的采样帧：**只取固定几帧且只解一次**，整架共用。
    # 质检开销因此与短片长度、与袋数都不成正比——1200 帧的架和 24 帧的架一样快，
    # 六袋的架也不会把同一段画面反复解码六遍。
    # 上传前预检（ingest/quality.py）与袋级分类（core/failure_modes.py）看的是**同一批**
    # 画面：预检答「这一整段值不值得信」，袋级分类答「哪一只袋子没拍好」，两层共用一次解码。
    quality_params = state.settings.quality_params
    quality_indices = sample_indices(len(extracted.paths), max(1, quality_params.sample_frames))
    quality_frames = [load_rgb(extracted.paths[index]) for index in quality_indices]
    precheck = precheck_frames(
        quality_frames,
        state.settings.precheck_params,
        frame_indices=quality_indices,
    )

    # 多袋并行：整架的袋子在**一个状态、一遍传播**里跑完（不是逐袋各跑一遍），
    # 随后单独验一次「哪条曲线是哪只袋子」——认不准的进护士认领列表，不默认取最高分。
    batch = propagate_batch(
        list(extracted.paths),
        [
            TrackAnchor(slot.bag_seq, prompt, slot.slot_index)
            for slot, (_obj_id, prompt) in zip(meta.slots, seeds)
        ],
        predictor,
    )
    curves = dict(batch.curves)
    slot_result = assign_slots(
        batch.views(),
        slot_columns(len(meta.slots), first_frame.shape[1], [s.bag_seq for s in meta.slots]),
        frame_shape=batch.frame_shape,
    )
    assignments = {item.obj_id: item for item in slot_result.assignments}

    # 跨巡视轮次重锚：这一轮拍到的还是不是上一轮那几只袋子。
    #
    # 判定认不准时保持 needs_claim 交护士认领，不默认取分数最高的那只（见 reanchor.py）。
    # 已确认换袋的历史袋不进入打分集合——它只用来判断「本轮是不是新袋」。
    candidate_rows = [
        {
            "candidate_id": item.candidate_id,
            "bbox": list(item.bbox),
            "area": item.area,
            "stability_score": round(float(item.stability_score), 6),
            "point": [float(item.point[0]), float(item.point[1])],
            "slot_index": item.slot_index,
            "bound": bool(item.bound),
        }
        for item in enumeration.candidates
    ]
    by_candidate_id = {item.candidate_id: item for item in enumeration.candidates}
    history = sessions.list_previous_rounds(
        meta.rack_id, meta.infusion_order_no, exclude_shift=meta.shift_key
    )
    # 帧几何在此定型（ROI 窗口也已随传播走到末态）。
    geometry = FrameGeometry(
        camera_size=resize_report.plan.original_size,
        canonical_size=resize_report.plan.output_size,
        plan=resize_report.plan,
        roi=roi_tracker.window,
    )
    # 跨轮坐标系是否一致：上一轮落库时记的规范帧尺寸与本轮不同（多半是中途改过
    # FRAME_LONG_EDGE_MAX），跨轮几何比对此时不可靠。**如实告知而不是静默**——
    # 两套坐标系下的候选框直接比，比出来的高置信度是假的。
    geometry_mismatch = any(
        row.get("canonical_width") not in (None, geometry.canonical_size[0])
        or row.get("canonical_height") not in (None, geometry.canonical_size[1])
        for row in history
    )
    if geometry_mismatch:
        logger.warning(
            "跨轮帧几何不一致：上一轮规范帧 %s×%s，本轮 %s×%s；跨轮重锚的几何项不可靠，"
            "建议整院统一时段内不要改 FRAME_LONG_EDGE_MAX",
            history[0].get("canonical_width"),
            history[0].get("canonical_height"),
            geometry.canonical_size[0],
            geometry.canonical_size[1],
        )
    # 含已换袋的历史袋：函数内部按状态分流（换掉的只用于判「这是新袋」）
    previous_descriptors = [
        descriptor
        for descriptor in (descriptor_from_session(row) for row in history)
        if descriptor is not None
    ]
    observations: list[SlotObservation] = []
    for slot in meta.slots:
        candidate_id = bound_by_slot.get(slot.slot_index)
        candidate = by_candidate_id.get(candidate_id) if candidate_id else None
        descriptor = None
        if candidate is not None and candidate.mask is not None and candidate.mask.any():
            descriptor = describe_region(
                first_frame,
                candidate.mask,
                bag_seq=slot.bag_seq,
                slot_index=slot.slot_index,
                shift_key=meta.shift_key,
            )
        observations.append(
            SlotObservation(
                slot_index=slot.slot_index,
                bag_seq=slot.bag_seq,
                descriptor=descriptor,
                candidate_id=candidate_id,
            )
        )
    verdicts = {
        verdict.slot_index: verdict
        for verdict in reanchor_slots(
            observations, previous_descriptors, previous_seen=bool(history)
        )
    }

    # 生效标定按袋型取一次（同一架多袋同型不重复查库）；缺记录即 None —— 走
    # uncalibrated 分支，响应里不出现 estimated_volume_ml。
    slots_by_seq: dict[str, SlotSpec] = {slot.bag_seq: slot for slot in meta.slots}
    calibrations: dict[str, Any] = {}
    residuals: dict[str, Any] = {}
    for slot in meta.slots:
        code = slot.bag_type_code
        if not code or code in calibrations:
            continue
        record = state.store.calibrations.get_active(code)
        calibrations[code] = record
        # 自校准残留补偿：只对登记了腔体参数的袋型有意义（没腔体参数就没得反解）。
        # 证据不足 / 互相打架时 summary 返回 applied=False，主路径照常出数、不修正。
        if record is not None and record.cavity is not None:
            residuals[code] = state.store.deviations.summary(
                code, usable_capacity_ml=record.usable_capacity_ml
            )

    for record in records:
        curve = curves.get(record["bag_seq"])
        slot_index = int(record.get("slot_index") or 0)
        verdict = verdicts.get(slot_index)
        candidate_id = bound_by_slot.get(slot_index)
        candidate = by_candidate_id.get(candidate_id) if candidate_id else None
        # 槽位-轨迹分配结论：这条曲线是不是真的落在它声明的那个挂点上。
        assignment = assignments.get(record["bag_seq"])
        # 面积序列平滑：原始观测照留（area_ratio_series），平滑后的工作序列供
        # 毫升/ETA/换液点名读。缺帧不插值，反常帧保留原值并标出来。
        smoothing = (
            smooth_series(
                curve.area_ratio_series,
                frame_indices=curve.frame_indices,
                params=state.settings.smoothing_params,
            )
            if curve is not None
            else None
        )

        # 起锚信息与外观描述子落库：描述子是下一轮认出「还是那一袋」的唯一依据。
        # 候选列表**对每个袋位都落**（哪怕这一位没绑上）——没绑上的正是护士要认领的，
        # 候选只存在响应里的话，护士点认领时服务端已经无从回查了。
        anchor_payload: dict[str, Any] = {}
        if candidate_rows:
            anchor_payload["candidates"] = candidate_rows
        if candidate is not None:
            descriptor = next(
                (o.descriptor for o in observations if o.candidate_id == candidate_id), None
            )
            anchor_payload.update(
                candidate_id=candidate_id,
                bbox=list(candidate.bbox),
                descriptor=descriptor.to_json() if descriptor is not None else {},
            )
        if anchor_payload:
            sessions.save_anchor(record["session_id"], **anchor_payload)
        if verdict is not None:
            sessions.save_reanchor(
                record["session_id"],
                {
                    "reanchor_state": verdict.anchor_state,
                    "reanchor_confidence": verdict.confidence,
                    "reanchor_runner_up": verdict.runner_up,
                    "reanchor_reason": verdict.reason,
                    "previous_session_id": verdict.matched_session_id
                    if verdict.bound
                    else None,
                    "previous_shift_key": verdict.previous_shift_key if verdict.bound else None,
                },
            )
            state.store.audit_chain.append(
                request_id=request_id,
                actor="service:reanchor",
                action="reanchor.resolved",
                target=record["session_id"],
                outcome="accepted" if verdict.anchor_state in ("bound", "fresh") else "rejected",
                bag_seq=record["bag_seq"],
                detail={
                    "reason": verdict.reason,
                    "anchor_state": verdict.anchor_state,
                    "confidence": verdict.confidence,
                    "runner_up": verdict.runner_up,
                    # 拖分项与描述子档位：事后复盘「当时为什么没接上」靠这两项，
                    # 只留一个 reason 码说不清是颜色变了还是液位变高了。
                    "driver": verdict.driver,
                    "descriptor_mode": verdict.descriptor_mode,
                    "scores": [
                        [bag, round(total, 6), round(geometry, 6)]
                        for bag, total, geometry in verdict.scores
                    ],
                },
            )

        if curve is not None:
            sessions.update_area_series(
                record["session_id"],
                curve.area_ratio_series,
                raw=curve.area_ratio_series,
                occlusion_flags=curve.occlusion_flags,
                frame_count_occluded=curve.occluded_frame_count,
            )
            record = sessions.get(record["session_id"]) or record

        # 体积与 ETA：滴速优先级 = 输注医嘱文本 → 泵速遥测 → 给不出来就不给时点。
        # 轨迹已交出（lost）时按「没有观测」处理，不用丢失前的旧读数冒充当前余量。
        slot = slots_by_seq.get(record["bag_seq"])
        # 遥测现状先算出来再定滴速：ETA 的「来源与理由」要能说清这条泵速读数有多新，
        # 而**新鲜度不参与门禁**——余量与换液判定不依赖它，stale 的读数照样用于
        # 与医嘱的一致性交叉校验，只是状态如实报给护士端。
        telemetry_view = classify_sample(
            state.telemetry.get(meta.rack_id, record["bag_seq"]),
            reference=meta.captured_at,
            fresh_seconds=state.settings.telemetry_fresh_seconds,
        )
        rate = resolve_rate(
            slot.rate_text if slot else None,
            telemetry_view.rate_ml_per_h,
            telemetry_state=telemetry_view.state,
        )
        fill_ratio = (
            smoothing.latest_trusted
            if curve is not None and smoothing is not None and not curve.is_lost
            else None
        )
        # 体积不确定度要拿得到**这一段的观测证据**：可信序列的离散度、整段帧数与
        # 遮挡帧数。证据从平滑结果与曲线汇总——曲线没有就没有（标定改版重算走那条
        # 路径时同理，从会话行里复原），绝不因为「拿不到证据」就当成完美观测。
        observation = (
            ObservationQuality(
                area_ratio_samples=tuple(
                    value for value in smoothing.trusted if value is not None
                ),
                frame_count=len(curve.frame_indices) if curve else 0,
                occluded_frame_count=curve.occluded_frame_count if curve else 0,
            )
            if curve is not None
            else None
        )
        estimate = estimate_bag(
            fill_ratio,
            calibrations.get(record.get("bag_type_code")),
            rate,
            bag_type_code=record.get("bag_type_code"),
            learned_residual=residuals.get(record.get("bag_type_code")),
            observation=observation,
            uncertainty_params=state.settings.uncertainty_params,
        )
        sessions.update_calibration(
            record["session_id"],
            {
                "fill_ratio": estimate.fill_ratio,
                "estimated_volume_ml": estimate.estimated_volume_ml,
                "eta_minutes": estimate.eta_minutes,
                "eta_source": estimate.eta_source,
                "rate_ml_per_h": estimate.rate_ml_per_h,
                "rate_source": estimate.eta_source,
                "rate_reason": estimate.rate_reason,
                "rate_conflict_json": (
                    json.dumps(estimate.rate_conflict, ensure_ascii=False)
                    if estimate.rate_conflict
                    else None
                ),
                "calibration_state": estimate.calibration_state,
                "calibration_version": estimate.calibration_version,
                "pickup_suppressed": estimate.pickup_suppressed,
                "suppression_reason": estimate.suppression_reason,
                "volume_confidence": _confidence_of(estimate),
                "volume_low_ml": _interval_of(estimate, "low_ml"),
                "volume_high_ml": _interval_of(estimate, "high_ml"),
                "uncertainty_gate": _interval_of(estimate, "gate"),
                "uncertainty_json": (
                    json.dumps(estimate.uncertainty, ensure_ascii=False)
                    if estimate.uncertainty
                    else None
                ),
            },
        )
        record = sessions.get(record["session_id"]) or record

        # 换液点名判定在**整轮收尾时**统一做（见循环后的 detect_round）：
        # 每轮点名上限与 ETA 排序都是「这一架所有袋子一起看」才成立的口径，
        # 单袋边算边点名会把上限退化成「每袋各点各的」。
        # 判定读的是**可信序列**：反光尖峰不会把「还早着呢」顶成「快见底了」，
        # 看不见的帧仍然是 null（那一段不计入连续帧数）。
        detection_inputs.append(
            {
                **record,
                "area_ratio_series": smoothing.trusted
                if smoothing
                else record.get("area_ratio_series"),
            }
        )

        # 拍摄质量分类：拍不好时给一句能照做的补拍建议，**不拒绝上传**。
        #
        # 放在体积换算与换液点名**之后**是故意的：质检只解释「这些数字该打多少问号」，
        # 不能反过来把已经算出来的曲线和毫升数扣掉——「拍坏了」和「查不出数」是两件事，
        # 后者有独立的门禁（无标定 / 无滴速 / 无观测，见 calibration.estimator）。
        prompt = prompt_by_seq.get(record["bag_seq"])
        quality = (
            classify_capture(
                quality_frames,
                state.predictor,
                prompt,
                quality_params,
            )
            if prompt is not None and quality_frames
            else None
        )
        if quality is not None:
            sessions.update_quality(
                record["session_id"],
                [flag.to_dict() for flag in quality.flags],
                quality.degraded,
            )
        # 预检结论逐袋各存一份：同一段短片的每一只袋子共享同一个预检结果，护士复盘时
        # 按袋查得到「这轮拍得怎么样」，而不是只在上传响应里看一次就没了。
        sessions.update_precheck(
            record["session_id"],
            score=precheck.score if precheck.sampled_frames else None,
            precheck_flags=[flag.to_dict() for flag in precheck.flags],
            degraded=precheck.degraded,
        )
        # 帧几何逐袋各存一份：相机原始尺寸、规范帧尺寸、缩放比与 ROI 末态窗。
        # 这几列是「上一轮报 1280×960、这一轮报 640×480」时唯一的对账依据。
        sessions.update_frame_geometry(
            record["session_id"],
            camera_width=geometry.camera_size[0],
            camera_height=geometry.camera_size[1],
            canonical_width=geometry.canonical_size[0],
            canonical_height=geometry.canonical_size[1],
            scale=geometry.scale,
            roi_window=roi_tracker.window.as_dict() if roi_tracker.window else None,
        )

        bags.append(
            BagSlotOut(
                session_id=record["session_id"],
                bag_seq=record["bag_seq"],
                slot_index=record["slot_index"],
                bag_type_code=record["bag_type_code"],
                frames_dir=record["frames_dir"],
                frame_count=record["frame_count"],
                state=record["state"],
                calibration_state=record["calibration_state"],
                session_version=record["session_version"],
                frame_indices=curve.frame_indices if curve else [],
                area_ratio_series=curve.area_ratio_series if curve else [],
                area_ratio_smoothed=smoothing.smoothed if smoothing else [],
                area_ratio_reconstructed=smoothing.reconstructed if smoothing else [],
                smoothing=SmoothingOut(**smoothing.as_dict()) if smoothing else None,
                occlusion_flags=curve.occlusion_flags if curve else [],
                visibility_states=curve.visibility_states if curve else [],
                track_handoff=handoff_to_dict(curve.handoff) if curve else None,
                cavity_area_px=curve.cavity_area_px if curve else 0,
                visible_frame_count=curve.visible_frame_count if curve else 0,
                occluded_frame_count=curve.occluded_frame_count if curve else 0,
                fill_ratio=estimate.fill_ratio,
                estimated_volume_ml=estimate.estimated_volume_ml,
                calibration_version=estimate.calibration_version,
                eta_minutes=estimate.eta_minutes,
                eta_source=estimate.eta_source,
                rate_ml_per_h=estimate.rate_ml_per_h,
                rate_reason=estimate.rate_reason or None,
                rate_conflict=estimate.rate_conflict,
                # 遥测现状（task 39）：缺了它，护士端只能从「eta_source=order_text」
                # 反推泵没来消息；有了它可以直接显示「泵速遥测 12 分钟前那次的」。
                telemetry_state=telemetry_view.state,
                telemetry_age_seconds=telemetry_view.age_seconds,
                telemetry_source=telemetry_view.source or None,
                pickup_suppressed=estimate.pickup_suppressed,
                suppression_reason=estimate.suppression_reason,
                volume_uncertainty=(
                    UncertaintyOut(**estimate.uncertainty) if estimate.uncertainty else None
                ),
                near_empty=estimate.near_empty,
                overfilled=estimate.overfilled,
                estimation_mode=estimate.estimation_mode,
                cavity_height_fraction=estimate.cavity_height_fraction,
                resident_ml=estimate.resident_ml,
                capture_degraded=bool(quality.degraded) if quality is not None else False,
                precheck_score=precheck.score if precheck.sampled_frames else None,
                precheck_degraded=precheck.degraded,
                capture_quality=(
                    [QualityFlagOut(**flag.to_dict()) for flag in quality.flags]
                    if quality is not None
                    else []
                ),
                predictor=state.predictor.name,
                candidate_state=(
                    "needs_claim"
                    if (verdict is not None and verdict.anchor_state == "unbound")
                    or record.get("slot_index") not in bound_by_slot
                    or (assignment is not None and assignment.claim_required)
                    else "bound"
                ),
                slot_state=assignment.state if assignment is not None else None,
                slot_assigned_index=assignment.slot_index if assignment is not None else None,
                slot_score=assignment.score if assignment is not None else None,
                slot_margin=assignment.margin if assignment is not None else None,
                slot_reason=assignment.reason if assignment is not None else None,
                slot_detail=assignment.detail if assignment is not None else None,
                candidate_id=candidate_id,
                reanchor_state=verdict.anchor_state if verdict else None,
                reanchor_reason=verdict.reason if verdict else None,
                reanchor_confidence=verdict.confidence if verdict else None,
                reanchor_runner_up=verdict.runner_up if verdict else None,
                reanchor_driver=verdict.driver if verdict else None,
                descriptor_mode=verdict.descriptor_mode if verdict else None,
                previous_session_id=(
                    verdict.matched_session_id if verdict and verdict.bound else None
                ),
                previous_shift_key=(
                    verdict.previous_shift_key if verdict and verdict.bound else None
                ),
                claim_state=record.get("claim_state") or "auto",
            )
        )

    # ── 换液点名：整轮判定（滞回带 / 每袋最小间隔 / 每轮上限）──────────────
    #
    # 放进循环之后有两个硬理由：
    # ①「每轮最多点 N 袋」和「按 ETA 升序」都是**这一架所有袋子一起看**才成立的口径；
    # ②滞回带按轮次累计，单袋边算边判没有「上一轮」的上下文可读。
    #
    # 本引擎**只产出建议**：被点的袋推 `monitoring → needs_change`，
    # `replaced` 仍然只能由护士在终端确认——未确认的 needs_change 一次 HTTP 都不发。
    detection = state.change_detector.detect_round(
        detection_inputs,
        round_key=meta.shift_key,
        now=utc_now(),
    )
    picked = 0
    for gate in detection.gates:
        target = gate.session_id
        if not target:
            continue
        record = sessions.get(target)
        if record is None:
            continue
        if gate.picked and gate.decision is not None and gate.decision.changed:
            # 只推 monitoring → needs_change。replaced 仍然只能由护士确认。
            apply_decision(
                sessions,
                gate.decision,
                audit=state.store.audit_chain,
                request_id=request_id,
            )
            picked += 1
            continue
        if gate.proposal is None:
            # 本来就没打算点这袋（双条件没满足 / 已终态 / 没有观测）：
            # 这类不进点名审计，否则审计链会被「什么都没发生」淹没。
            continue
        reason = gate.suppress_reason or "no_trigger"
        truncated = reason == "round_cap_exceeded"
        state.store.audit_chain.append(
            request_id=request_id,
            actor="service:change-detector",
            action="pickup.truncated" if truncated else "pickup.suppressed",
            target=target,
            outcome="accepted" if truncated else "rejected",
            bag_seq=gate.bag_seq,
            bag_key=gate.bag_key,
            from_state=record.get("state"),
            to_state=record.get("state"),
            detail={
                "reason": reason,
                "round_key": meta.shift_key,
                "detail": gate.detail,
                "band_dwell_rounds": gate.band_dwell_rounds,
                "latched": gate.latched,
                "since_minutes": gate.since_minutes,
                "eta_minutes": gate.eta_minutes,
                "rank": gate.proposal.rank,
            },
        )
    # 响应里带上闸门结论：护士长要能回答「这袋这轮为什么没被点名」
    for bag in bags:
        gate = detection.gate_for(bag.bag_seq)
        if gate is None:
            continue
        bag.pickup_state = "named" if gate.picked else "suppressed"
        bag.pickup_suppress_reason = gate.suppress_reason
        bag.pickup_detail = gate.detail
        bag.pickup_rank = gate.proposal.rank if gate.proposal is not None else None
        bag.pickup_truncated = gate.suppress_reason == "round_cap_exceeded"
        if gate.picked and gate.proposal is not None:
            # 状态与触发原因是在**这一轮收尾时**才落库的，而 bags 是在袋级循环里
            # 造的（那时还没点名）。不回写这两项，护士端看到的就是「点了名但状态还是
            # 监测中、也没有为什么点名」——那等于没点名。
            record = sessions.get(bag.session_id) or {}
            bag.state = record.get("state") or bag.state
            bag.trigger_reason = record.get("trigger_reason") or gate.proposal.trigger_reason

    # ── M6 决策融合仲裁：模型不是裁判，只是结论的措辞与第二意见 ────────────
    #
    # 放在点名**之后**是硬理由：仲裁的对象是「点名判定」这个已算好的结论，
    # 放在它之前跑就变成让模型参与判定本身了。默认关闭（LLM_FUSION_ENABLED=false）
    # ——关闭时下面这段一行都不执行，响应里 decision_source 恒为 threshold_only，
    # 行为与院内没部署模型端点时逐字一致。
    fused = _fuse_round(
        state,
        detection_inputs,
        picked_bags={gate.bag_seq for gate in detection.gates if gate.picked},
        request_id=request_id,
    )
    for bag in bags:
        record = sessions.get(bag.session_id) or {}
        conclusion = fused.get(bag.bag_seq)
        if conclusion is not None:
            bag.decision_source = conclusion.decision_source
            bag.decision_reason = conclusion.reason
            bag.arbitration = conclusion.arbitration.to_json()
        else:
            # 没开仲裁（或这袋没被点名）时，如实报「这条结论来自阈值路径」。
            bag.decision_source = str(record.get("decision_source") or "threshold_only")
            if bag.decision_source not in DECISION_SOURCES:
                bag.decision_source = "threshold_only"

    logger.info(
        "换液点名判定完成 rack=%s shift=%s named=%d truncated=%d suppressed=%d",
        meta.rack_id,
        meta.shift_key,
        picked,
        len(detection.truncated),
        len([gate for gate in detection.gates if gate.proposal is None or not gate.picked]),
    )

    logger.info(
        "帧序列接入完成 rack=%s order=%s frames=%d bags=%d candidates=%d shift=%s "
        "camera=%sx%s canonical=%sx%s roi=%s",
        meta.rack_id,
        meta.infusion_order_no,
        extracted.frame_count,
        len(bags),
        len(enumeration.candidates),
        meta.shift_key,
        geometry.camera_size[0],
        geometry.camera_size[1],
        geometry.canonical_size[0],
        geometry.canonical_size[1],
        roi_tracker.window.as_dict() if roi_tracker.window else "full",
    )
    return TrackUploadResponse(
        upload_key=key,
        rack_id=meta.rack_id,
        bed_no=meta.bed_no,
        infusion_order_no=meta.infusion_order_no,
        captured_at=meta.captured_at,
        shift_key=meta.shift_key,
        frame_count=extracted.frame_count,
        first_frame=extracted.names[0],
        last_frame=extracted.names[-1],
        frame_time_source=time_source,
        bags=bags,
        candidates=[
            CandidateOut(
                candidate_id=candidate.candidate_id,
                bbox=list(candidate.bbox),
                area=candidate.area,
                stability_score=candidate.stability_score,
                slot_index=candidate.slot_index,
                bound=candidate.bound,
            )
            for candidate in enumeration.candidates
        ],
        precheck=(
            PrecheckOut(
                state=precheck.state,
                quality_score=precheck.score,
                degraded=precheck.degraded,
                sampled_frames=precheck.sampled_frames,
                sub_scores=dict(precheck.sub_scores),
                primary=(
                    QualityFlagOut(**precheck.primary.to_dict()) if precheck.primary else None
                ),
                flags=[QualityFlagOut(**flag.to_dict()) for flag in precheck.flags],
                frames=[FrameQualityOut(**item.to_dict()) for item in precheck.frames],
            )
            if precheck.sampled_frames
            else None
        ),
        geometry=FrameGeometryOut(
            camera_width=geometry.camera_size[0],
            camera_height=geometry.camera_size[1],
            canonical_width=geometry.canonical_size[0],
            canonical_height=geometry.canonical_size[1],
            scale=round(geometry.scale, 6),
            resized=resize_report.resized,
            reason=resize_report.plan.reason,
            resized_frames=resize_report.resized_frames,
            total_frames=resize_report.total_frames,
            scale_mismatch_with_previous_round=geometry_mismatch,
        ),
        roi=RoiOut(
            enabled=state.settings.roi_params.enabled,
            applied=roi_tracker.state.applied_calls > 0,
            window=(
                RoiBoxOut(**roi_tracker.window.as_dict()) if roi_tracker.window else None
            ),
            applied_calls=roi_tracker.state.applied_calls,
            advances=roi_tracker.state.advances,
            max_step_px=roi_tracker.state.max_step_px,
            border_touches=roi_tracker.state.border_touches,
            anchor_width=roi_tracker.state.anchor_width,
            anchor_height=roi_tracker.state.anchor_height,
            fallback_reason=roi_tracker.state.fallback_reason,
        ),
        accepted_at=utc_now(),
    )


@router.get(
    "/sessions/{session_id}",
    summary="查袋级会话当前状态",
    description="按上传响应里的 session_id 查询本轮接入结果与业务状态。",
)
async def get_session(session_id: str, request: Request) -> dict:
    record = request.app.state.store.sessions.get(session_id)
    if record is None:
        raise ApiError(
            "NOT_FOUND",
            f"session_id={session_id} 不存在；请用上传响应里 bags[].session_id 查询",
        )
    return record

@router.get(
    "/sessions/{session_id}/versions",
    summary="查这一袋被整段重放过几次、旧版本各是多少",
    description=(
        "整段重放（同输液架 + 同轮次 + 同袋序再传一次）会让会话版本加一，"
        "被盖掉的那一版留在版本历史里：护士长问「十分钟前报的是多少毫升」时，"
        "答案在这里而不是在已经变了的面板上。"
    ),
)
async def get_session_versions(session_id: str, request: Request) -> SessionVersionsOut:
    sessions = request.app.state.store.sessions
    record = sessions.get(session_id)
    if record is None:
        raise ApiError(
            "NOT_FOUND",
            f"session_id={session_id} 不存在；请用上传响应里 bags[].session_id 查询",
        )
    versions = [
        SessionVersionOut(
            session_id=item["session_id"],
            session_version=int(item["session_version"]),
            state=item.get("state"),
            trigger_reason=item.get("trigger_reason"),
            captured_at=item.get("captured_at"),
            frame_count=int(item.get("frame_count") or 0),
            fill_ratio=item.get("fill_ratio"),
            estimated_volume_ml=item.get("estimated_volume_ml"),
            eta_minutes=item.get("eta_minutes"),
            calibration_state=item.get("calibration_state"),
            calibration_version=item.get("calibration_version"),
            area_ratio_series=list(item.get("area_ratio_series") or []),
            archived_at=item["archived_at"],
            archived_reason=item.get("archived_reason") or "replay",
        )
        for item in sessions.list_versions(session_id)
    ]
    return SessionVersionsOut(
        session_id=session_id,
        current_version=int(record.get("session_version") or 1),
        archived_count=len(versions),
        versions=versions,
    )
