"""槽位认领与换液点名判定：``/api/v1/infusion/claim`` 与 ``/decision``。

**护士端的动作是「点一行」，不是「逐袋点像素」**：上传响应里已经给了首帧枚举出的
候选列表（位置框 + 稳定分），认领就是指定「这个候选是 B01」。点错了就再点一次——
:class:`api.claim.claim_candidate` 会**按新候选重算这一袋的整条曲线**，而不是只改个
标签糊弄过去。认不准时程序宁可把 `candidate_state` 留在 `needs_claim`，也不替人猜。

**换液点名判定**（``/decision``）走的是 :mod:`identity.state_machine` 的确定性双条件
路径：连续 N 帧低于液位阈值 **且** 余量确实不够了。它只产出 `needs_change`，
`replaced` 只能由护士确认（task 31 的确认接口承接），未确认的点名不会出站。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, Request

from ..calibration.estimator import RateResolution, estimate_bag
from ..calibration.uncertainty import ObservationQuality
from ..core.predictor import Prompt
from ..core.propagate import analyse_frames, load_rgb
from ..core.smoothing import smooth_series
from ..errors import ApiError, Issue
from ..identity.reanchor import describe_region
from ..logging_utils import request_id_of
from ..identity.state_machine import (
    ChangeDecision,
    ChangeParams,
    IllegalTransition,
    apply_decision,
    evaluate_change,
)
from ..store.repositories import new_id, utc_now
from .schemas import ClaimIn, ClaimOut, DecisionIn, DecisionOut, SmoothingOut, UncertaintyOut

__all__ = ["router"]

logger = logging.getLogger("infusion_monitor.api.claim")

router = APIRouter(prefix="/api/v1/infusion", tags=["identity"])


def _request_id(request: Request, header: str | None) -> str:
    """取本次请求的 id：入站 ``X-Request-Id`` 优先，中间件上下文次之，都没有就生成。

    与 ``logging_utils`` 共用同一份 id——审计行、运行日志与响应头必须指向同一条链，
    否则排障时要跨三个地方手动对齐时间戳。
    """
    value = header or request.headers.get("X-Request-Id")
    value = (value or "").strip()
    return value or request_id_of(request)


def _frame_paths(session: dict[str, Any]) -> list[Path]:
    root = Path(session.get("frames_dir") or "")
    if not root.is_dir():
        raise ApiError(
            "NOT_FOUND",
            f"session_id={session['session_id']} 的帧目录已不在（可能被保留期清理过），"
            "无法重算曲线",
            issues=[
                Issue(
                    code="frames_gone",
                    field="frames_dir",
                    reason="该会话的帧目录不存在",
                    suggestion="请重新上传这一轮巡视的短片后再认领。",
                )
            ],
        )
    return sorted(root.glob("frame_*.jpg"))


def _rate_from_session(session: dict[str, Any]) -> RateResolution:
    """从会话行还原上次接入时已解出的滴速（不重新解析医嘱，也不猜）。"""
    rate = session.get("rate_ml_per_h")
    if rate is None or float(rate) <= 0:
        return RateResolution(
            rate_ml_per_h=None,
            source="unavailable",
            reason=session.get("rate_reason") or "该袋没有可用滴速，不给时点",
        )
    return RateResolution(
        rate_ml_per_h=float(rate),
        source=str(session.get("rate_source") or session.get("eta_source") or "order_text"),
        reason=session.get("rate_reason") or "",
    )


@router.post(
    "/claim",
    response_model=ClaimOut,
    response_model_exclude_none=True,
    summary="在候选列表里认领一袋（或改绑到别的候选）",
    description=(
        "护士在上传响应给出的候选列表里点一行完成认领：指定 candidate_id 属于该会话的"
        "袋序。程序据此**重算这一袋的整条面积比曲线与毫升/ETA**，不是只改标签。"
        "认错了就再调一次（同一候选重复认领按幂等处理）。"
    ),
)
async def claim_candidate(
    payload: ClaimIn, request: Request, x_request_id: str | None = Header(default=None)
) -> ClaimOut:
    state = request.app.state
    store = state.store
    request_id = _request_id(request, x_request_id)

    record = store.sessions.get(payload.session_id)
    if record is None:
        raise ApiError(
            "NOT_FOUND",
            f"session_id={payload.session_id} 不存在；请用上传响应里 bags[].session_id",
        )

    candidates = record.get("candidates_json") or []
    candidate = next(
        (item for item in candidates if item.get("candidate_id") == payload.candidate_id), None
    )
    if candidate is None:
        known = ", ".join(str(item.get("candidate_id")) for item in candidates) or "（无）"
        raise ApiError(
            "INGEST_INVALID",
            f"candidate_id={payload.candidate_id} 不在本轮候选列表里",
            issues=[
                Issue(
                    code="candidate_not_found",
                    field="candidate_id",
                    reason=f"本轮可认领的候选只有：{known}",
                    suggestion="请从上传响应的 candidates 列表里选一行；"
                    "若列表为空，说明这一挂点本轮没枚举出可认的袋子，请重新补拍。",
                )
            ],
        )

    point = candidate.get("point") or []
    if len(point) < 2:
        raise ApiError(
            "INGEST_INVALID",
            f"candidate_id={payload.candidate_id} 缺少起锚点，无法重算曲线",
        )

    paths = _frame_paths(record)
    predictor = state.predictor
    prompt = Prompt.from_point(float(point[0]), float(point[1]))
    curves = analyse_frames(paths, [(str(record["bag_seq"]), prompt)], predictor)
    if not curves:
        raise ApiError(
            "INTERNAL",
            "按该候选重算曲线没有得到任何轨迹；请重新上传这一轮短片",
        )
    curve = curves[0]
    # 平滑口径与上传主路径共用同一份实现与参数：认领后重算出来的毫升数不能走
    # 另一套平滑规则，否则「认领前后两个数不一样」就没法向护士解释了。
    smoothing = smooth_series(
        curve.area_ratio_series,
        frame_indices=curve.frame_indices,
        params=state.settings.smoothing_params,
    )

    sessions = store.sessions
    sessions.update_area_series(
        record["session_id"],
        curve.area_ratio_series,
        raw=curve.area_ratio_series,
        occlusion_flags=curve.occlusion_flags,
        frame_count_occluded=curve.occluded_frame_count,
    )

    # 认领后重算体积与 ETA：滴速沿用上次接入已解出的结果，不在这里另起一套解析口径。
    calibration = None
    learned = None
    if record.get("bag_type_code"):
        calibration = store.calibrations.get_active(str(record["bag_type_code"]))
        if calibration is not None and calibration.cavity is not None:
            learned = store.deviations.summary(
                str(record["bag_type_code"]),
                usable_capacity_ml=calibration.usable_capacity_ml,
            )
    fill_ratio = smoothing.latest_trusted if not curve.is_lost else None
    # 认领后重算用的是**同一条**不确定度门禁：认领只是换了起锚的候选，观测证据
    # （本段序列、帧数、遮挡帧数）没变，置信口径不能因为认领了就换一套。
    observation = ObservationQuality(
        area_ratio_samples=tuple(v for v in smoothing.trusted if v is not None),
        frame_count=len(curve.frame_indices) if curve else 0,
        occluded_frame_count=curve.occluded_frame_count if curve else 0,
    )
    estimate = estimate_bag(
        fill_ratio,
        calibration,
        _rate_from_session(record),
        bag_type_code=record.get("bag_type_code"),
        learned_residual=learned,
        observation=observation,
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
            "calibration_state": estimate.calibration_state,
            "calibration_version": estimate.calibration_version,
            "pickup_suppressed": estimate.pickup_suppressed,
            "suppression_reason": estimate.suppression_reason,
            "volume_confidence": (
                estimate.uncertainty["confidence"] if estimate.uncertainty else None
            ),
            "volume_low_ml": (
                estimate.uncertainty.get("low_ml") if estimate.uncertainty else None
            ),
            "volume_high_ml": (
                estimate.uncertainty.get("high_ml") if estimate.uncertainty else None
            ),
            "uncertainty_gate": (
                estimate.uncertainty.get("gate") if estimate.uncertainty else None
            ),
            "uncertainty_json": estimate.uncertainty,
        },
    )

    # 描述子按新候选重算：认领是纠错，外观描述子必须跟着改，否则下一轮重锚又认回旧的。
    first_frame = load_rgb(paths[0])
    predictions = predictor.predict(first_frame, prompt)
    mask = predictions[0].mask if predictions else None
    bbox = candidate.get("bbox")
    if mask is not None and mask.any():
        descriptor = describe_region(
            first_frame,
            mask,
            bag_seq=str(record["bag_seq"]),
            slot_index=int(record.get("slot_index") or 0),
            session_id=str(record["session_id"]),
            shift_key=str(record.get("shift_key") or ""),
            state=str(record.get("state") or "monitoring"),
        )
    else:
        # 掩膜为空（候选框还留在库里但这块已经看不见液体）：只换绑定，不写描述子，
        # 让下一轮走「外观不可用 → 不自动接」的分支，而不是留一个过期的描述子。
        descriptor = None

    already_claimed = record.get("claim_state") in ("claimed", "corrected")
    same_candidate = record.get("anchor_candidate_id") == payload.candidate_id
    corrected = already_claimed and not same_candidate
    sessions.save_anchor(
        record["session_id"],
        candidate_id=str(payload.candidate_id),
        bbox=[int(v) for v in bbox] if bbox else None,
        # 掩膜为空时写空描述子而不是留着旧的：认领是纠错，留旧描述子会让下一轮
        # 重锚又认回刚才被否掉的那一只。
        descriptor=descriptor.to_json() if descriptor is not None else {},
        claim_state="corrected" if corrected else "claimed",
        claimed_by=payload.operator_id,
    )
    sessions.save_reanchor(
        record["session_id"],
        {
            "reanchor_state": "bound" if descriptor is not None else "unbound",
            "reanchor_reason": "claimed" if descriptor is not None else "appearance_unavailable",
            "reanchor_confidence": None,
            "reanchor_runner_up": None,
        },
    )
    store.audit_chain.append(
        request_id=request_id,
        actor=f"nurse:{payload.operator_id}",
        action="candidate.corrected" if corrected else "candidate.claimed",
        target=str(record["session_id"]),
        outcome="accepted",
        bag_seq=record.get("bag_seq"),
        from_state=None,
        to_state=None,
        detail={
            "candidate_id": payload.candidate_id,
            "previous_candidate_id": record.get("anchor_candidate_id"),
            "corrected": corrected,
            "slot_index": record.get("slot_index"),
            "note": payload.note or "",
        },
    )
    updated = store.sessions.get(record["session_id"]) or record

    logger.info(
        "槽位认领 session=%s bag=%s candidate=%s corrected=%s",
        record["session_id"],
        record.get("bag_seq"),
        payload.candidate_id,
        corrected,
    )
    return ClaimOut(
        session_id=str(updated["session_id"]),
        bag_seq=str(updated["bag_seq"]),
        slot_index=int(updated.get("slot_index") or 0),
        candidate_id=str(payload.candidate_id),
        bbox=[int(v) for v in bbox] if bbox else None,
        claim_state="corrected" if corrected else "claimed",
        corrected=corrected,
        claimed_by=payload.operator_id,
        frame_count=len(curve.frame_indices),
        area_ratio_series=curve.area_ratio_series,
        area_ratio_smoothed=smoothing.smoothed,
        smoothing=SmoothingOut(**smoothing.as_dict()),
        fill_ratio=estimate.fill_ratio,
        estimated_volume_ml=estimate.estimated_volume_ml,
        calibration_version=estimate.calibration_version,
        eta_minutes=estimate.eta_minutes,
        eta_source=estimate.eta_source,
        pickup_suppressed=estimate.pickup_suppressed,
        suppression_reason=estimate.suppression_reason,
        volume_uncertainty=(
            UncertaintyOut(**estimate.uncertainty) if estimate.uncertainty else None
        ),
        descriptor_refreshed=descriptor is not None,
        decided_at=utc_now(),
    )


@router.post(
    "/decision",
    response_model=DecisionOut,
    response_model_exclude_none=True,
    summary="重算某一袋的换液点名判定",
    description=(
        "按「连续 N 帧低于液位阈值 **且** 余量确实不够」双条件重算 needs_change。"
        "被标定门禁压住（未标定 / 滴速缺失 / 整段看不见 / 超装）的袋不会被点名——"
        "只压住时点、保留可核对的部分；近空液面是直接观测，不受该门禁约束。"
    ),
)
async def decide_change(
    payload: DecisionIn, request: Request, x_request_id: str | None = Header(default=None)
) -> DecisionOut:
    state = request.app.state
    store = state.store
    request_id = _request_id(request, x_request_id)

    record = store.sessions.get(payload.session_id)
    if record is None:
        raise ApiError(
            "NOT_FOUND",
            f"session_id={payload.session_id} 不存在；请用上传响应里 bags[].session_id",
        )

    # 已换袋是终态：再对这一袋做判定等于想让状态回退，直接 409（并留审计行）。
    if str(record.get("state") or "monitoring") == "replaced":
        raise IllegalTransition(
            "replaced",
            "monitoring",
            session_id=str(record["session_id"]),
            bag_seq=record.get("bag_seq"),
            audit=store.audit_chain,
            request_id=request_id,
        )

    decision: ChangeDecision = evaluate_change(
        record, ChangeParams(), now=utc_now()
    )
    updated, changed = apply_decision(
        store.sessions,
        decision,
        audit=store.audit_chain,
        request_id=request_id,
    )
    return DecisionOut(
        session_id=str(updated["session_id"]),
        bag_seq=str(updated["bag_seq"]),
        bed_no=updated.get("bed_no"),
        state=str(updated.get("state") or "monitoring"),
        changed=changed,
        trigger_reason=decision.trigger_reason,
        decision=decision.to_json(),
        estimated_volume_ml=updated.get("estimated_volume_ml"),
        eta_minutes=updated.get("eta_minutes"),
        pickup_suppressed=bool(updated.get("pickup_suppressed")),
        decided_at=decision.decided_at,
    )