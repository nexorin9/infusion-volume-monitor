"""袋型标定登记与查询：``/api/v1/infusion/calibrations``。

**为什么要有这几个端点**：没有标定记录，系统就只能给护士一条面积比曲线——「还剩
42%」不能拿去决定要不要换袋；有了标定记录才有毫升数与预计耗尽时点。标定记录由
**护理部/信息科**在系统上线时按袋型登记、按批次改版，业务代码不参与换算系数。

* ``POST /calibrations``：登记一个版本（首次=建版，重复=改版）。改版会**立刻把该
  袋型的历史会话按新版本重算**，差异逐条写 append-only 的版本轨迹；
* ``GET  /calibrations``：生效中的标定记录（可按袋型看历史版本）；
* ``GET  /calibrations/recalc``：改版重算轨迹，回答「这袋当时为什么报 180 毫升」。

改版**不回滚已换袋会话**的数字（``replaced`` 的会话保留当时的读数），只重算仍在
监测中的会话——护理部核对的是「当时的数对不对」，不是「现在该是多少」。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from ..calibration.cavity import CavitySpec
from ..calibration.estimator import recompute_for_new_version
from ..calibration.models import BagTypeCalibration, CalibrationSample
from ..calibration.versions import classify_all, promote_due
from ..errors import ApiError, Issue
from ..store.repositories import utc_now
from .schemas import CalibrationCreateIn, CalibrationOut, ReviewResolveIn

__all__ = ["router"]

router = APIRouter(prefix="/api/v1/infusion", tags=["calibration"])


def _previous_record(store: Any, bag_type_code: str, version: int) -> BagTypeCalibration | None:
    """取该袋型的第 ``version`` 版（改版收口后核对 effective_to 用）。"""
    for record in store.calibrations.list_by_code(bag_type_code):
        if record.version == version:
            return record
    return None


def superseded_calibration_id(store: Any, stored: BagTypeCalibration, version: int) -> str:
    previous = _previous_record(store, stored.bag_type_code, version)
    return previous.calibration_id if previous else f"{stored.bag_type_code}#v{version}"


def _to_record(payload: CalibrationCreateIn) -> BagTypeCalibration:
    samples = tuple(
        CalibrationSample(
            measured_fill_ratio=item.measured_fill_ratio,
            known_volume_ml=item.known_volume_ml,
            source=item.source,
            captured_at=item.captured_at,
        )
        for item in payload.samples
    )
    cavity = None
    if payload.cavity is not None:
        cavity = CavitySpec(
            bag_type_code=payload.bag_type_code,
            capacity_ml=payload.cavity.capacity_ml,
            cavity_height_mm=payload.cavity.cavity_height_mm,
            dead_volume_ml=payload.cavity.dead_volume_ml,
            meniscus_offset_ratio=payload.cavity.meniscus_offset_ratio,
            shape_exponent=payload.cavity.shape_exponent,
            shoulder_center=payload.cavity.shoulder_center,
            shoulder_half=payload.cavity.shoulder_half,
            taper=payload.cavity.taper,
            bulge=payload.cavity.bulge,
            film_holdup_ml=payload.cavity.film_holdup_ml,
            # 版本号由登记入口分配：首次登记即第 1 版，改外形参数时由调用方显式升版。
            # 不在这里编造版本——没登记版本的几何会被置信门禁挡下毫升数（见
            # calibration.uncertainty），那是刻意的：宁可不出数，也不给一个
            # 「参数可能已经改过、自己却不知道」的数字。
            geometry_revision=payload.cavity.geometry_revision or 1,
            label_zh=payload.cavity.label_zh,
            registered_by=payload.calibrated_by,
            registered_at=payload.calibrated_at or utc_now(),
            note=payload.cavity.note,
        )
    return BagTypeCalibration.from_samples(
        payload.bag_type_code,
        payload.capacity_ml,
        samples,
        dead_volume_ml=payload.dead_volume_ml,
        meniscus_offset_ratio=payload.meniscus_offset_ratio,
        calibrated_by=payload.calibrated_by,
        calibrated_at=payload.calibrated_at or utc_now(),
        note=payload.note,
        cavity=cavity,
        effective_from=payload.effective_from,
        effective_to=payload.effective_to,
    )


@router.post(
    "/calibrations",
    response_model=CalibrationOut,
    response_model_exclude_none=True,
    status_code=201,
    summary="登记袋型标定（首次建版 / 再次改版）",
    description=(
        "把「量到的面积比」与「已知的真实毫升数」成对登记成标定样例，拟合成该袋型的"
        "换算曲线。同一袋型再次登记即改版：旧版本失活但保留可查，历史会话按新版本"
        "重算并在版本轨迹里留下改前/改后的差异。"
    ),
)
async def create_calibration(payload: CalibrationCreateIn, request: Request) -> CalibrationOut:
    store = request.app.state.store
    settings = request.app.state.settings
    record = _to_record(payload)
    stored, superseded, status = store.calibrations.save(record)
    if status == "active":
        recalc = recompute_for_new_version(
            store, stored, superseded, review_threshold=settings.calibration_review_threshold
        )
    else:
        # 预约生效：今天不动任何读数，也不重算——提前换版会让当晚还在监测的
        # 病区凭空多出几百毫升的差值，那不是修正，是事故。
        recalc = {
            "recalculated_count": 0,
            "skipped_count": 0,
            "held_for_review": [],
            "review_required_count": 0,
            "review_threshold": settings.calibration_review_threshold,
        }
    audit = store.audit_chain
    audit.append(
        request_id=request.headers.get("x-request-id") or "calibration-register",
        actor=payload.calibrated_by,
        action="calibration.version_created",
        target=stored.calibration_id,
        outcome="accepted",
        detail={
            "bag_type_code": stored.bag_type_code,
            "version": stored.version,
            "superseded_version": superseded,
            "version_status": status,
            "effective_from": stored.effective_from,
        },
    )
    if superseded is not None:
        audit.append(
            request_id=request.headers.get("x-request-id") or "calibration-register",
            actor=payload.calibrated_by,
            action="calibration.version_superseded",
            target=superseded_calibration_id(store, stored, superseded),
            outcome="accepted",
            detail={"bag_type_code": stored.bag_type_code, "new_version": stored.version},
        )
    if status == "scheduled":
        audit.append(
            request_id=request.headers.get("x-request-id") or "calibration-register",
            actor=payload.calibrated_by,
            action="calibration.version_scheduled",
            target=stored.calibration_id,
            outcome="accepted",
            detail={
                "bag_type_code": stored.bag_type_code,
                "version": stored.version,
                "effective_from": stored.effective_from,
            },
        )
    superseded_to = None
    if superseded is not None:
        previous = _previous_record(store, stored.bag_type_code, superseded)
        superseded_to = previous.effective_to if previous else None
    return CalibrationOut(
        version_status=status,
        effective_from=stored.effective_from,
        effective_to=stored.effective_to,
        superseded_to=superseded_to,
        review_required_sessions=recalc["review_required_count"],
        review_threshold=recalc["review_threshold"],
        calibration_id=stored.calibration_id,
        bag_type_code=stored.bag_type_code,
        version=stored.version,
        superseded_version=superseded,
        capacity_ml=stored.capacity_ml,
        usable_capacity_ml=stored.usable_capacity_ml,
        dead_volume_ml=stored.dead_volume_ml,
        meniscus_offset_ratio=stored.meniscus_offset_ratio,
        bands=[[x, y] for x, y in stored.bands],
        estimation_mode="cavity" if stored.cavity is not None else "band",
        cavity=stored.cavity.as_dict() if stored.cavity else None,
        sample_count=len(stored.samples),
        max_relative_error=stored.max_relative_error,
        calibrated_by=stored.calibrated_by,
        calibrated_at=stored.calibrated_at,
        note=stored.note,
        active=stored.active,
        recalculated_sessions=recalc["recalculated_count"],
    )


@router.get(
    "/calibrations",
    summary="查标定记录",
    description=(
        "不带 bag_type_code 时返回全部生效中的记录；带上则返回该袋型的全部版本"
        "（含已失效版本），便于核对「这条曲线是谁标的、什么时候标的」。"
    ),
)
async def list_calibrations(
    request: Request,
    bag_type_code: str | None = Query(default=None, description="按袋型过滤"),
) -> dict[str, Any]:
    store = request.app.state.store
    if bag_type_code:
        records = store.calibrations.list_by_code(bag_type_code)
        return {
            "bag_type_code": bag_type_code,
            "count": len(records),
            "calibrations": [record.describe() for record in records],
        }
    records = store.calibrations.list_active()
    return {
        "count": len(records),
        "calibrations": [record.describe() for record in records],
    }


@router.get(
    "/calibrations/recalc",
    summary="查标定改版后的重算轨迹",
    description=(
        "append-only 记录：每次改版重算了哪些会话、改前多少毫升、改后多少毫升。"
        "只追加不修改，可直接导出给护理部与内审留档。"
    ),
)
async def list_recalc(
    request: Request,
    bag_type_code: str | None = Query(default=None, description="按袋型过滤"),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict[str, Any]:
    store = request.app.state.store
    rows = store.calibrations.list_recalc(bag_type_code, limit=limit)
    return {
        "bag_type_code": bag_type_code,
        "count": len(rows),
        "recalc_log": rows,
    }


@router.get(
    "/calibrations/deviations",
    summary="查自校准残留偏差的样本与结论",
    description=(
        "每一行是一次「护士确认已换液」留下的残留证据：换液前量到的面积比、"
        "模型当时给出的毫升数，以及反推出来的残差。confirmed 表示该条参与了定标，"
        "ineligible_reason 说明为什么这条没参与（换液前看得很高 / 没有读数 / 没声明袋型）。"
        "样本只追加不修改，可直接导出给护理部复核。"
    ),
)
async def list_deviations(
    request: Request,
    bag_type_code: str = Query(min_length=1, description="袋型编码"),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict[str, Any]:
    store = request.app.state.store
    rows = store.deviations.list_samples(bag_type_code, limit=limit)
    calibration = store.calibrations.get_active(bag_type_code)
    learned = store.deviations.summary(
        bag_type_code,
        usable_capacity_ml=(
            calibration.usable_capacity_ml if calibration else 0.0
        ),
    )
    return {
        "bag_type_code": bag_type_code,
        "count": len(rows),
        "deviations": rows,
        "learned_residual": learned.as_dict(),
        "cavity_registered": calibration is not None and calibration.cavity is not None,
    }


@router.get(
    "/calibrations/{bag_type_code}/versions",
    summary="查某袋型的生效版本与历史版本",
    description=(
        "给信息科与护理部核对用：这一袋型现在用的是第几版、什么时候由谁标的、"
        "还有哪几版排着队没生效（scheduled），历史版本各自的生效区间是什么。"
        "同一袋型任何时刻只可能有一个 active 版本——这条约束在登记时就已挡住。"
    ),
)
async def list_versions(
    bag_type_code: str, request: Request
) -> dict[str, Any]:
    store = request.app.state.store
    records = store.calibrations.list_by_code(bag_type_code)
    promoted = _promote_due(store, bag_type_code, records, request)
    if promoted:
        records = store.calibrations.list_by_code(bag_type_code)
    views = classify_all(records)
    alerts = store.calibrations.list_alerts(bag_type_code, limit=500)
    return {
        "bag_type_code": bag_type_code,
        "registered": bool(records),
        "count": len(views),
        "active_version": next((v.version for v in views if v.status == "active"), None),
        "scheduled_versions": [v.version for v in views if v.status == "scheduled"],
        "versions": [view.as_dict() for view in views],
        "promoted_versions": promoted,
        "review_threshold": request.app.state.settings.calibration_review_threshold,
        "pending_reviews": sum(1 for a in alerts if a["resolution"] == "pending"),
    }


@router.post(
    "/calibrations/{bag_type_code}/reviews/{alert_id}",
    summary="复核一条改版偏差告警",
    description=(
        "改版后某一袋的毫升读数相对差超过阈值，系统没有自动覆盖旧读数，"
        "而是把新旧两个数、相对差都留在这里等护理部结案。"
        "applied = 复核通过，按新读数落库；dismissed = 复核否决，旧读数继续沿用。"
        "两种结案都会写 append-only 审计链，可追到是谁按的。"
    ),
)
async def resolve_review(
    bag_type_code: str, alert_id: int, payload: ReviewResolveIn, request: Request
) -> dict[str, Any]:
    store = request.app.state.store
    alert = store.calibrations.get_alert(alert_id)
    if alert is None or alert["bag_type_code"] != bag_type_code:
        raise ApiError(
            "NOT_FOUND",
            f"告警不存在：{bag_type_code} / alert_id={alert_id}",
            issues=[
                Issue(
                    code="alert_not_found",
                    field="alert_id",
                    reason="该袋型下没有这条改版偏差告警",
                    suggestion="先用 GET /calibrations/{bag_type_code}/reviews 看待复核清单里的 alert_id。",
                )
            ],
        )
    if alert["resolution"] != "pending":
        # 已被结案：先报冲突再去碰会话，否则「重复处理」会被误报成「会话不存在」。
        raise ApiError(
            "STATE_CONFLICT",
            f"告警 {alert_id} 已于 {alert['resolved_at']} 结案为 {alert['resolution']}"
            f"（复核人 {alert['resolved_by']}），不可重复处理",
            issues=[
                Issue(
                    code="review_already_resolved",
                    field="resolution",
                    reason=f"已由 {alert['resolved_by']} 于 {alert['resolved_at']} 结案为 {alert['resolution']}",
                    suggestion="先用 GET /calibrations/{bag_type_code}/reviews 看这条的 resolution，"
                    "已结案的不必重复提交。",
                )
            ],
        )
    session = store.sessions.get(alert["session_id"])
    if session is None:
        raise ApiError(
            "NOT_FOUND",
            f"告警对应的会话已不存在：{alert['session_id']}",
            issues=[
                Issue(
                    code="session_gone",
                    field="alert_id",
                    reason="告警指向的袋位会话已被保留期清理",
                    suggestion="该袋已不在监测中，直接把这条按 dismissed 结案并注明原因。",
                )
            ],
        )
    request_id = request.headers.get("x-request-id") or f"calibration-review-{alert_id}"

    if payload.resolution == "applied":
        # 复核通过：把待复核的新读数真正落下去，并清掉待复核标记。
        store.sessions.update_calibration(
            alert["session_id"],
            {
                "estimated_volume_ml": alert["proposed_volume_ml"],
                "calibration_version": alert["new_version"],
                "calibration_review_required": 0,
                "calibration_review_deviation": None,
                "calibration_review_version": None,
                "calibration_review_proposed_ml": None,
            },
        )
        store.calibrations.log_recalc(
            calibration=_record_by_version(store, bag_type_code, alert["new_version"]),
            previous_version=alert["previous_version"],
            session=session,
            previous_volume_ml=alert["previous_volume_ml"],
            new_volume_ml=alert["proposed_volume_ml"],
            reason="calibration_review_applied",
            relative_deviation=alert["relative_deviation"],
            review_required=False,
        )
        action = "calibration.review_applied"
    else:
        # 否决：新版本有问题，旧读数原样沿用，只把「这条不采信」记下来。
        store.sessions.update_calibration(
            alert["session_id"],
            {
                "calibration_review_required": 0,
                "calibration_review_deviation": None,
                "calibration_review_version": None,
                "calibration_review_proposed_ml": None,
            },
        )
        action = "calibration.review_dismissed"

    try:
        resolved = store.calibrations.resolve_alert(
            alert_id, payload.resolution, by=payload.reviewed_by, note=payload.note
        )
    except ValueError as exc:
        # 重复结案/非法结案状态：这条告警已经有人处理过了，不是内部故障。
        raise ApiError(
            "STATE_CONFLICT",
            str(exc),
            issues=[
                Issue(
                    code="review_already_resolved",
                    field="resolution",
                    reason=str(exc),
                    suggestion="先用 GET /calibrations/{bag_type_code}/reviews 看这条的 resolution，"
                    "已结案的不必重复提交。",
                )
            ],
        ) from exc
    store.audit_chain.append(
        request_id=request_id,
        actor=payload.reviewed_by,
        action=action,
        target=alert["session_id"],
        bag_seq=alert["bag_seq"],
        outcome="accepted",
        detail={
            "alert_id": alert_id,
            "bag_type_code": bag_type_code,
            "new_version": alert["new_version"],
            "previous_volume_ml": alert["previous_volume_ml"],
            "proposed_volume_ml": alert["proposed_volume_ml"],
            "relative_deviation": alert["relative_deviation"],
            "threshold": alert["threshold"],
            "note": payload.note,
        },
    )
    return {
        "alert_id": alert_id,
        "session_id": alert["session_id"],
        "resolution": payload.resolution,
        "resolved_by": payload.reviewed_by,
        "resolved_at": resolved["resolved_at"],
        "effective_volume_ml": (
            resolved["proposed_volume_ml"]
            if payload.resolution == "applied"
            else resolved["previous_volume_ml"]
        ),
    }


@router.get(
    "/calibrations/{bag_type_code}/reviews",
    summary="查某袋型的改版偏差告警（待复核清单）",
    description=(
        "每行 = 「新版本会把这一袋从多少毫升改成多少毫升、相对差多少、"
        "因此没有自动生效」。resolution=pending 即护理部待处理的清单；"
        "旧读数在这期间一直沿用，所以这不是故障，是有意不覆盖。"
    ),
)
async def list_reviews(
    bag_type_code: str,
    request: Request,
    resolution: str | None = Query(default=None, description="pending / applied / dismissed"),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict[str, Any]:
    store = request.app.state.store
    rows = store.calibrations.list_alerts(bag_type_code, resolution=resolution, limit=limit)
    return {
        "bag_type_code": bag_type_code,
        "resolution": resolution,
        "count": len(rows),
        "pending_count": store.calibrations.count_alerts("pending"),
        "reviews": rows,
    }


def _promote_due(
    store: Any, bag_type_code: str, records: list[BagTypeCalibration], request: Request
) -> list[int]:
    """把已经到生效时刻的预约版本转正，并收口前一版的区间。

    **只转正、不重算**：交接当天的读数变化必须走偏差闸门（人工复核），
    不能由一次查询动作悄悄改掉病区看到的毫升数。
    """
    due = promote_due(records)
    if not due:
        return []
    request_id = request.headers.get("x-request-id") or "calibration-versions"
    for view in due:
        superseded = store.calibrations.promote(
            view.calibration_id, effective_from=view.window.effective_from
        )
        store.audit_chain.append(
            request_id=request_id,
            actor="system",
            action="calibration.version_superseded",
            target=view.calibration_id,
            outcome="accepted",
            detail={
                "bag_type_code": bag_type_code,
                "promoted_to": view.version,
                "effective_from": view.window.effective_from,
                "superseded_version": superseded,
                "auto_promoted": True,
            },
        )
    return [view.version for view in due]


def _record_by_version(
    store: Any, bag_type_code: str, version: int | None
) -> BagTypeCalibration:
    record = _previous_record(store, bag_type_code, int(version or 0))
    if record is None:
        # 版本行被清理过也要能结案：重算轨迹只需要袋型与版本号，够落字段即可。
        raise ApiError(
            "NOT_FOUND",
            f"版本不存在：{bag_type_code} v{version}",
            issues=[
                Issue(
                    code="version_not_found",
                    field="alert_id",
                    reason="告警指向的标定版本在库里已经取不到了",
                    suggestion="先查 GET /calibrations/{bag_type_code}/versions 核对版本号，"
                    "再决定 applied 还是 dismissed。",
                )
            ],
        )
    return record
