"""换液点名载荷与病区点名清单：``GET /api/v1/infusion/pickups``。

**这一层只读。** 判定在接入时做完（:mod:`identity.change_detector`），本模块把库里
``needs_change`` 的袋读出来，整理成护士长在护士站一眼能扫完、逐条能核对的清单。
它**不写任何东西**：不推进状态、不落出站事件、不碰写回队列——所以「查一次点名清单」
这件事在任何情况下都不可能把一条未确认的换液推给护理执行平台。

清单本身要解决三件现场的事：

1. **顺序**。六袋同时挂在那儿，护士长要知道先去哪一床。按 **ETA 升序**排（最急的排
   头），同 ETA 按 **床号**、再按袋序，保证同一轮里顺序可复现——同一份清单刷新两次
   顺序不能跳。取不到滴速的袋排最后而不是混进中间：它没有「几点见底」，
   排进 ETA 序列等于编了一个时点。
2. **依据**。每条点名都要能回答「凭什么点它」。给 ``trigger_reason`` 原因码 + 一句
   人话 ``evidence_note`` + 一个 ``confidence``。
3. **能不能拿这个数去换袋**。未标定袋型时**没有毫升数**，此时清单**照常点名**（护士
   看得见这袋快空了），但必须说清「依据面积比，未标定袋型」——不给假毫升数，也不
   让护士以为那个数字不存在。

**幂等键在清单里就发下去**（``client_event_id`` + ``confirm_hint``）：护士在终端点
「已换液」时直接把这一条原样回填，确认接口按同一规则校验（见
:mod:`infusion_monitor.api.transition`）。规则由服务端唯一持有
（:func:`~infusion_monitor.identity.state_machine.build_client_event_id`），终端自己
编一个键只会把同一次换液拆成两次。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from fastapi import APIRouter, Query, Request

from ..errors import ApiError, Issue
from ..identity.state_machine import build_client_event_id
from ..logging_utils import request_id_of
from ..store.repositories import utc_now
from .schemas import PickupItemOut, PickupListOut

__all__ = [
    "NEEDS_CHANGE_STATE",
    "PickupItem",
    "PickupList",
    "build_pickup_item",
    "build_pickup_list",
    "observation_confidence",
    "pickup_sort_key",
    "read_pickup_list",
    "router",
]

logger = logging.getLogger("infusion_monitor.api.pickup")

router = APIRouter(prefix="/api/v1/infusion", tags=["identity"])

#: 清单里出现的状态。只有这一个状态会被点名——``monitoring`` 还没到提醒的程度，
#: ``replaced`` 已经换过了（终态不再参与判定，见 ``change_detector``）。
NEEDS_CHANGE_STATE = "needs_change"

#: 观测级置信度的两个系数，写在这里是为了让「为什么这条读数可信/不可信」可复算：
#: 遮挡占比每高 1，观测置信按 :data:`OCCLUSION_PENALTY` 打折；帧数覆盖不足时先打折到
#: :data:`COVERAGE_FLOOR`，再随可见帧数线性补回。
OCCLUSION_PENALTY = 0.5
COVERAGE_FLOOR = 0.7
#: 可见帧达到这个数就算「覆盖充分」（接入下限是 12 帧）。
COVERAGE_TARGET = 12
#: 液面低于 :data:`LOW_FILL_RATIO` 时几何核读数最不稳（细窄连通域接近碎斑阈值），
#: 按线性比例再扣一点置信——点名时恰恰都是快见底的袋，这一项对清单里最急的那几条最贴。
LOW_FILL_RATIO = 0.10
LOW_FILL_PENALTY = 0.15


# ── 置信与降级文案 ──────────────────────────────────────────────────────────


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parsed_json(raw: Any) -> dict[str, Any]:
    """``uncertainty_json`` 落库是字符串，读回来要解一次；解不了就当没有。"""
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
        except ValueError:
            return {}
        return dict(value) if isinstance(value, Mapping) else {}
    return {}


def _visible_ratios(session: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for item in session.get("area_ratio_series") or []:
        if item is None:
            continue
        value = _as_float(item)
        if value is not None:
            values.append(value)
    return values


def observation_confidence(session: Mapping[str, Any]) -> tuple[float | None, dict[str, Any]]:
    """**观测级**置信度：这袋的液面读数本身有多可信（与毫升换算无关）。

    三个因子，全部可复算，不含任何模型输出：

    * **可见帧占比**——被遮挡的帧留 ``null`` 不插值，占比越低说明这一段越靠猜；
    * **帧数覆盖**——看得见但只有两三帧，和看得见十六帧，不该给同一个数；
    * **低液位惩罚**——液面低到 10% 以下时袋体在画面里只剩一条窄带，边界最容易被
      孔洞/孤岛阈值误伤，因此再扣一点。

    返回 ``(置信度, 因子明细)``。**完全没有可见观测时返回 ``(None, …)``**——不给
    一个 0 冒充「测过了但不可信」，护士看到的是「这袋这一段没测到」。
    """
    series = list(session.get("area_ratio_series") or [])
    total = len(series)
    visible = _visible_ratios(session)
    if total == 0 or not visible:
        return None, {
            "visible_frames": len(visible),
            "total_frames": total,
            "occlusion_ratio": None,
            "coverage": 0.0,
            "low_fill_penalty": 0.0,
            "reason": "no_observation",
        }

    occlusion_ratio = round(1.0 - len(visible) / total, 4)
    coverage = min(1.0, len(visible) / COVERAGE_TARGET)
    fill = visible[-1]
    low_fill = min(max((LOW_FILL_RATIO - fill) / LOW_FILL_RATIO, 0.0), 1.0)
    penalty = round(LOW_FILL_PENALTY * low_fill, 4)

    base = 1.0 - OCCLUSION_PENALTY * occlusion_ratio
    value = base * (COVERAGE_FLOOR + (1.0 - COVERAGE_FLOOR) * coverage) - penalty
    return (
        round(max(0.0, min(1.0, value)), 4),
        {
            "visible_frames": len(visible),
            "total_frames": total,
            "occlusion_ratio": occlusion_ratio,
            "coverage": round(coverage, 4),
            "low_fill_penalty": penalty,
            "reason": "observed",
        },
    )


def _evidence_note(session: Mapping[str, Any]) -> tuple[str, list[str]]:
    """点名依据的人话说明 + 依据清单（``area_ratio`` 恒在，其余按门禁增减）。

    未标定袋型时文案里逐字出现「依据面积比，未标定袋型」：护士要能一眼分清
    「这个结论是量出来的」和「这个结论是看出来的」。
    """
    notes: list[str] = []
    basis = ["area_ratio"]
    calibration_state = str(session.get("calibration_state") or "uncalibrated")
    bag_type = str(session.get("bag_type_code") or "")

    if calibration_state == "calibrated":
        version = _as_int(session.get("calibration_version"))
        if bag_type and version:
            notes.append(f"按 {bag_type} 第 {version} 版标定换算毫升")
        else:
            notes.append("按袋型标定换算毫升")
        basis.append("calibration")
    elif calibration_state == "uncalibrated":
        target = f" {bag_type}" if bag_type else "（未登记袋型）"
        notes.append(f"依据面积比，未标定袋型{target}，给不出毫升数")
    else:
        # low_confidence / high_occlusion：不是「没标定」，是「这一段撑不起那个数」。
        uncertainty = _parsed_json(session.get("uncertainty_json"))
        hint = str(uncertainty.get("hint_zh") or "").strip()
        reason = str(session.get("suppression_reason") or calibration_state)
        notes.append(hint or f"依据面积比判断；这一段的观测给不出可信毫升数（{reason}）")

    eta_source = str(session.get("eta_source") or "")
    if eta_source == "unavailable":
        notes.append("未取到滴速，不给预计耗尽时点")
    elif eta_source == "mqtt_telemetry":
        notes.append("预计时点取自泵速遥测")
        basis.append("rate_telemetry")

    return "；".join(notes), basis


# ── 单条点名 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PickupItem:
    """清单里的一行。**它是一条建议，不是一个已发生的事实**。"""

    bag_seq: str
    session_id: str
    state: str
    client_event_id: str
    rack_id: str | None = None
    bed_no: str | None = None
    slot_index: int = 0
    infusion_order_no: str | None = None
    bag_type_code: str | None = None
    calibration_state: str = "uncalibrated"
    trigger_reason: str | None = None
    eta_minutes: int | None = None
    fill_ratio: float | None = None
    estimated_volume_ml: float | None = None
    confidence: float | None = None
    confidence_source: str = "observation"
    confidence_factors: dict[str, Any] = field(default_factory=dict)
    evidence_note: str = ""
    evidence_basis: tuple[str, ...] = ("area_ratio",)
    captured_at: str | None = None
    updated_at: str | None = None
    rank: int = 0

    def to_json(self) -> dict[str, Any]:
        """出参形状。门禁不通过的字段保持 ``None``，由 ``exclude_none`` 整体不出。"""
        return {
            "bag_seq": self.bag_seq,
            "session_id": self.session_id,
            "state": self.state,
            "rack_id": self.rack_id,
            "bed_no": self.bed_no,
            "slot_index": self.slot_index,
            "infusion_order_no": self.infusion_order_no,
            "bag_type_code": self.bag_type_code,
            "calibration_state": self.calibration_state,
            "trigger_reason": self.trigger_reason,
            "eta_minutes": self.eta_minutes,
            "fill_ratio": self.fill_ratio,
            "estimated_volume_ml": self.estimated_volume_ml,
            "confidence": self.confidence,
            "confidence_source": self.confidence_source,
            "confidence_factors": dict(self.confidence_factors),
            "evidence_note": self.evidence_note,
            "evidence_basis": list(self.evidence_basis),
            "client_event_id": self.client_event_id,
            # 未确认不出站：这两项是清单对「确认回写」维度的正面回答，
            # 而不是一句承诺——护士端可以照着 writeback_queued 直接决定要不要弹确认框。
            "writeback_queued": False,
            "writeback_hold_reason": "未确认换液不出站：护士在终端确认后才写回护理执行平台",
            "confirm_hint": {
                "session_id": self.session_id,
                "bag_seq": self.bag_seq,
                "client_event_id": self.client_event_id,
            },
            "captured_at": self.captured_at,
            "updated_at": self.updated_at,
            "rank": self.rank,
        }


def pickup_sort_key(item: PickupItem) -> tuple[bool, int, str, str, str]:
    """排序口径：**ETA 升序 → 床号 → 袋序 → 会话 ID**。

    取不到 ETA 的袋排最后（``eta_minutes is None`` 落在 ``True`` 那一侧）而不是按 0
    分钟混进最前——那等于给一个没有滴速的袋编了一个「马上就见底」的时点。

    末位补一个会话 ID：同一张床挂两架、同架同袋序的两轮会话都可能出现，只靠前三项
    排序是不全序，清单顺序会随数据库返回顺序变。护士照着清单逐条核对时顺序跳一次，
    就会怀疑自己漏了一袋——所以宁可多一个字段也要把它变成全序。
    """
    eta = item.eta_minutes
    return (
        eta is None,
        eta if eta is not None else 0,
        item.bed_no or "",
        item.bag_seq,
        item.session_id,
    )


def build_pickup_item(session: Mapping[str, Any], *, rank: int = 0) -> PickupItem:
    """把一条 ``needs_change`` 会话行整理成清单条目（纯函数，不读库不写库）。"""
    session_id = str(session.get("session_id") or "")
    bag_seq = str(session.get("bag_seq") or "")
    note, basis = _evidence_note(session)
    observation, factors = observation_confidence(session)

    # 置信度优先用体积置信（有标定时的 95% 区间结论）；没有毫升数时退回观测级置信，
    # 并把来源标出来——护士看到的数字必须能对上它的出处。
    volume_confidence = _as_float(session.get("volume_confidence"))
    if volume_confidence is not None:
        confidence, source = volume_confidence, "volume"
    else:
        confidence, source = observation, "observation"

    return PickupItem(
        bag_seq=bag_seq,
        session_id=session_id,
        state=str(session.get("state") or NEEDS_CHANGE_STATE),
        client_event_id=build_client_event_id(session_id, bag_seq),
        rack_id=session.get("rack_id"),
        bed_no=session.get("bed_no"),
        slot_index=int(session.get("slot_index") or 0),
        infusion_order_no=session.get("infusion_order_no"),
        bag_type_code=session.get("bag_type_code"),
        calibration_state=str(session.get("calibration_state") or "uncalibrated"),
        trigger_reason=session.get("trigger_reason"),
        eta_minutes=_as_int(session.get("eta_minutes")),
        fill_ratio=_as_float(session.get("fill_ratio")),
        # 门禁：没有生效标定 / 被门禁压住时不给毫升数，与上传响应、decision 响应同一条语义。
        estimated_volume_ml=_as_float(session.get("estimated_volume_ml")),
        confidence=confidence,
        confidence_source=source,
        confidence_factors=factors,
        evidence_note=note,
        evidence_basis=tuple(basis),
        captured_at=session.get("captured_at"),
        updated_at=session.get("updated_at"),
        rank=rank,
    )


def build_pickup_list(
    sessions: Iterable[Mapping[str, Any]], *, generated_at: str | None = None
) -> "PickupList":
    """按口径排好序的病区点名清单（纯函数）。

    只收 ``needs_change``：``monitoring`` 还没到提醒程度，``replaced`` 已换过——
    两者出现在清单里都是噪声，而噪声会让人开始跳过整张清单。
    """
    items = [
        build_pickup_item(session)
        for session in sessions
        if str(session.get("state") or "") == NEEDS_CHANGE_STATE
    ]
    items.sort(key=pickup_sort_key)
    ranked = [
        PickupItem(**{**item.__dict__, "rank": index})
        for index, item in enumerate(items, start=1)
    ]
    return PickupList(generated_at=generated_at or utc_now(), items=tuple(ranked))


@dataclass(frozen=True)
class PickupList:
    """一份点名清单。它**不携带任何出站动作**——``writeback_queued_count`` 恒为 0。"""

    generated_at: str
    items: tuple[PickupItem, ...]

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def beds(self) -> list[str]:
        seen: list[str] = []
        for item in self.items:
            if item.bed_no and item.bed_no not in seen:
                seen.append(item.bed_no)
        return seen

    def item_for(self, bag_seq: str) -> PickupItem | None:
        for item in self.items:
            if item.bag_seq == bag_seq:
                return item
        return None

    def to_json(self) -> dict[str, Any]:
        uncalibrated = [item for item in self.items if item.estimated_volume_ml is None]
        return {
            "count": self.count,
            "bed_count": len(self.beds),
            "generated_at": self.generated_at,
            "items": [item.to_json() for item in self.items],
            # 清单这一层不产生任何写回：把这件事显式写进响应，而不是让调用方去猜。
            "writeback_queued_count": 0,
            "hold_reason": "未确认的 needs_change 只在清单里可见，绝不进入写回队列",
            "without_ml_count": len(uncalibrated),
        }


def read_pickup_list(
    sessions: Any,
    *,
    rack_id: str | None = None,
    limit: int = 100,
    states: Sequence[str] = (NEEDS_CHANGE_STATE,),
) -> PickupList:
    """从仓储读点名清单。``rack_id`` 给了就只看这一架（床旁终端按架核对用）。"""
    collected: list[Mapping[str, Any]] = []
    for state in states:
        collected.extend(sessions.list_by_state(state, limit=limit))
    if rack_id:
        collected = [row for row in collected if str(row.get("rack_id") or "") == rack_id]
    return build_pickup_list(collected)


# ── HTTP ────────────────────────────────────────────────────────────────────


@router.get(
    "/pickups",
    response_model=PickupListOut,
    response_model_exclude_none=True,
    summary="病区换液点名清单（只读）",
    description=(
        "读出当前所有处于 needs_change 的袋，按 ETA 升序 → 床号 → 袋序排好，"
        "每条带 trigger_reason、依据说明与置信度。\n\n"
        "**本端点只读**：不推进状态、不落出站事件、不进写回队列。"
        "护士在终端确认换液请调 `POST …/transition`。"
    ),
)
async def list_pickups(
    request: Request,
    rack_id: str | None = Query(default=None, description="只列这一架；不传则列全病区"),
    limit: int = Query(default=100, ge=1, le=500, description="单状态最多读取多少条"),
) -> PickupListOut:
    store = request.app.state.store
    request_id = request_id_of(request)
    if limit > 100:
        raise ApiError(
            "INGEST_INVALID",
            f"limit={limit} 超出单次上限 100；请按输液架用 rack_id 过滤，"
            "或分批拉取——一张几百行的清单护士长也不会用",
            issues=[
                Issue(
                    code="limit_too_large",
                    field="limit",
                    reason=f"请求 limit={limit}，上限 100",
                    suggestion="改用 rack_id 按架拉取，或把 limit 调到 100 以内。",
                )
            ],
        )
    listing = read_pickup_list(store.sessions, rack_id=rack_id, limit=limit)
    logger.info(
        "点名清单 request=%s count=%d rack=%s", request_id, listing.count, rack_id or "-"
    )
    return PickupListOut(**listing.to_json())