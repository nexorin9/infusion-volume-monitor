"""多袋并行传播与槽位-轨迹分配：一架 ≤6 袋一次传播，传完必须回答「哪条曲线是哪只袋子」。

**这一层为什么存在**：输液架上不是一只袋子。一个输液架最多挂 6 只，护士拍一段 10–20 秒的
短片，画面里同时有 6 个液体区域在动。前面的层各自解决一件事——:mod:`.enumerate` 在**首帧**
把「画面里可能是袋子的区域」找出来，:mod:`.propagate` 让**每条轨迹**逐帧保持身份，
:mod:`.reanchor` 判断「这一轮拍的是不是上一轮那几只」。但没有任何一层回答那个最朴素的问题：

    传完之后，画面左边那条下降的曲线，到底是**第 2 只挂点上的袋子**还是第 3 只？

答错的代价很具体：把 B02 的毫升数报给 B03，护士按错的数字去换液。多袋并行之后，轨迹与
挂点的对应关系必须**单独验一次**，认不准就交护士在列表里点一行——**绝不默认取分数最高的那个**。

**与源主路径的对应关系**（只借调用链与语义，不搬代码、不跑源权重）：

==============================================  ==========================================================
本模块                                          源主路径（代码证据）
==============================================  ==========================================================
:func:`propagate_batch` 的**一个 state、一遍传播**  ``vos_inference.py:137`` 的 ``inference_state``：一次视频
                                                只起一个状态，所有对象都挂在**同一个**状态里
:func:`propagate_batch` 起锚时的**一个循环加全部对象**  ``vos_inference.py:205-210`` 的 ``add_new_mask(... obj_id=...)``
                                                在一个 ``for`` 循环里把每个对象逐个加进同一个状态
:attr:`BatchPropagation.frames` 的逐帧 id→掩膜    ``vos_inference.py:224-231``：
                                                ``for out_frame_idx, out_obj_ids, out_mask_logits in
                                                predictor.propagate_in_video(...)``，然后
                                                ``{out_obj_id: (out_mask_logits[i] > thresh) ... for i,
                                                out_obj_id in enumerate(out_obj_ids)}``——**按对象 id 取回
                                                自己那份掩膜**，不是按位置猜
:attr:`BatchPropagation.anchor_ids`（身份集合冻结）  ``vos_inference.py:193-204``：``object_ids_set`` 在**第一条**输入帧
                                                上定死，后面出现新 ``object_id`` 直接 ``RuntimeError``
:func:`assign_slots` 的槽位-轨迹分配               源主路径**没有**这一层（它按 ``obj_id`` 存盘，不需要回答
                                                「这是哪个物理位置」）。这是医院侧必然存在的一段：
                                                输注单上的袋序必须对得上挂点，认不准就交人工认领
==============================================  ==========================================================

**四条纪律**（与 :mod:`.reacquire` / :mod:`.occlusion` 一致）：

1. **一次传播、一份状态**：6 只袋子共用一个状态、一遍 ``propagate_in_video``，不是每只袋子各
   跑一遍（那会让算力随袋数线性膨胀，也让「同一帧的多只袋子」失去可比性）。
2. **按 id 连接，不按顺序连接**：每帧的掩膜按 ``obj_id`` 归位。两袋 id 顺序颠倒一次，曲线就会
   整体对调——这种错误在数值上看不出来，所以连接键只允许是 id。
3. **认不准就不认**：三个通道（槽位竖列先验 / 质心垂向漂移 / 面积量级）任一硬门限不过即淘汰；
   前两名分差不足 ``min_margin`` 一律判歧义；**不默认取最高分**。
4. **整段没看清 = 没有依据**：全程被遮挡的轨迹**不给槽位**，也**不占**别人的槽位——它的槽位按
   ``unfilled`` 单独报出来，护士据此知道「这一挂点这段没拍到」，而不是被别袋的曲线顶替。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .predictor import Prompt, mask_centroid
from .propagate import (
    DEFAULT_REACQUIRE,
    BagCurve,
    ObjectTrack,
    PropagationState,
    add_new_points_or_box,
    init_state,
    load_rgb,
    propagate_in_video,
)
from .reacquire import ReacquireParams

__all__ = [
    "ASSIGNED",
    "MISMATCHED",
    "AMBIGUOUS",
    "BELOW_SCORE",
    "NO_EVIDENCE",
    "UNFILLED",
    "EXTRA_TRACK",
    "REASON_ASSIGNED",
    "REASON_SLOT_PRIOR",
    "REASON_CENTROID_DRIFT",
    "REASON_AREA_SCALE",
    "REASON_AMBIGUOUS_SLOT",
    "REASON_AMBIGUOUS_TRACK",
    "REASON_NEVER_SEEN",
    "REASON_DECLARED_SLOT_MISMATCH",
    "REASON_SLOT_EMPTY",
    "REASON_EXTRA_TRACK",
    "REASON_NO_SLOT",
    "MAX_BAGS_PER_RACK",
    "SlotAssignmentParams",
    "SlotColumn",
    "TrackAnchor",
    "TrackView",
    "SlotScore",
    "SlotAssignment",
    "SlotAssignmentResult",
    "BatchFrame",
    "BatchPropagation",
    "slot_columns",
    "assign_slots",
    "propagate_batch",
]

logger = logging.getLogger("infusion_monitor.core.multi_object")

#: 一个输液架最多挂几只袋子。与 ``ingest.validate.MAX_SLOTS`` 保持一致（有测试钉住），
#: 在这里另立一份是为了让 core 层不必反向依赖 ingest 层。
MAX_BAGS_PER_RACK = 6

# ── 分配状态 ────────────────────────────────────────────────────────────────
#: 轨迹确实停在它声明的那个挂点竖列里。
ASSIGNED = "assigned"
#: 轨迹没停在它声明的竖列里（袋序错位 / 拍到别袋），**不自动改绑**。
MISMATCHED = "mismatched"
#: 前两名分差不足，不猜。
AMBIGUOUS = "ambiguous"
#: 三个通道加权后没过总门槛。
BELOW_SCORE = "below_score"
#: 整段都没看清，没有依据判它在哪个槽位。
NO_EVIDENCE = "no_evidence"
#: 声明了这个挂点，但画面里没有任何轨迹落进来。
UNFILLED = "unfilled"
#: 画面里多出一条不落在任何挂点竖列上的轨迹（并袋、杂物、邻架入画）。
EXTRA_TRACK = "extra_track"

# ── 原因码（稳定，供护士端与审计引用） ───────────────────────────────────────
REASON_ASSIGNED = "assigned"
REASON_SLOT_PRIOR = "slot_prior_rejected"
REASON_CENTROID_DRIFT = "centroid_drift_rejected"
REASON_AREA_SCALE = "area_scale_rejected"
REASON_AMBIGUOUS_SLOT = "ambiguous_slot"
REASON_AMBIGUOUS_TRACK = "ambiguous_track"
REASON_NEVER_SEEN = "track_never_seen"
REASON_DECLARED_SLOT_MISMATCH = "declared_slot_mismatch"
REASON_SLOT_EMPTY = "slot_empty"
REASON_EXTRA_TRACK = "extra_track"
REASON_NO_SLOT = "no_slot_declared"


@dataclass(frozen=True)
class SlotAssignmentParams:
    """槽位-轨迹分配的门限与权重。取值理由写在字段旁，全部可由病区实测调整。"""

    #: 三通道权重和为 1：槽位竖列先验最重（挂点是物理事实），质心漂移次之
    #: （袋子挂在架上，垂向不该乱跑），面积量级最轻（只挡并袋/碎块）。
    slot_prior_weight: float = 0.45
    centroid_weight: float = 0.30
    area_weight: float = 0.25
    #: 槽位先验的容差，单位是**槽位列半宽**。0.55 ≈ 轨迹中心偏离列心半列宽以内拿满分。
    slot_centre_tolerance: float = 0.55
    #: 质心垂向漂移容差，单位是**画面高度**。输液架是固定挂点，液面下降不该让质心跑 18%。
    centroid_drift_tolerance: float = 0.18
    #: 面积量级容差：与同架其他轨迹的中位面积偏离超过该比例即判并袋/碎块（软打分）。
    area_scale_tolerance: float = 0.60
    #: 面积量级**硬门限**：偏离中位面积超过这个倍数即直接淘汰（不参与加权）。
    #: 只挡明显不是一只袋子的东西——并袋成一块（≈2×）或半袋残块（≈0.5×）。
    #: 软容差只影响打分高低，硬门限才让证据齐备也不给过。
    max_area_deviation: float = 2.0
    #: 三个通道各自的硬门限（先于加权淘汰，避免被其他通道的高分抬过门槛）。
    min_slot_score: float = 0.20
    min_centroid_score: float = 0.25
    #: 加权总分门槛。
    min_total: float = 0.45
    #: 最佳与次优分差小于它即判歧义。0.08 ≈ 槽位列宽的 5% 量级。
    min_margin: float = 0.08
    #: 单架袋数上限（超过即拒绝整段传播，不静默截断）。
    max_bags: int = MAX_BAGS_PER_RACK


@dataclass(frozen=True)
class SlotColumn:
    """一个挂点在画面上的竖列（几何先验，与 :func:`..propagate.slot_seed_prompt` 同口径）。"""

    slot_index: int
    #: 竖列中心的横坐标（像素）。
    centre_x: float
    #: 竖列半宽（像素）。打分按「偏离列心 ÷ 半宽」归一化。
    half_width: float
    #: 输注单上的袋序（若有）。用于把分配结果说成护士听得懂的话。
    bag_seq: str | None = None


def slot_columns(
    slot_count: int,
    width: int,
    bag_seqs: Sequence[str] | None = None,
) -> tuple[SlotColumn, ...]:
    """按挂点顺序把画面竖向等分成 ``slot_count`` 条竖列。

    口径与 :func:`..propagate.slot_seed_prompt` 严格一致（同一个 ``step = width / count``），
    两处若不一致，「起锚用的先验」和「分配用的先验」就会互相打架。
    """
    count = max(1, int(slot_count))
    step = width / count
    columns = []
    for index in range(count):
        x0 = index * step
        x1 = (index + 1) * step
        bag_seq = None if bag_seqs is None or index >= len(bag_seqs) else bag_seqs[index]
        columns.append(
            SlotColumn(
                slot_index=index,
                centre_x=(x0 + x1) / 2.0,
                half_width=max(1e-6, (x1 - x0) / 2.0),
                bag_seq=bag_seq,
            )
        )
    return tuple(columns)


@dataclass(frozen=True)
class TrackAnchor:
    """一条轨迹的起锚信息：给谁、在哪一帧起、用什么提示、声明它是哪个挂点。"""

    obj_id: str
    prompt: Prompt
    slot_index: int | None = None


@dataclass(frozen=True)
class TrackView:
    """一条轨迹的只读快照（分配打分只看得到这些量，不持有掩膜）。

    刻意**只保留统计量**：逐帧质心与面积都在 :class:`..propagate.BagCurve` 里算完就压成
    均值，本对象不留任何逐帧数组——恒定算力。
    """

    obj_id: str
    declared_slot: int | None
    #: 整段里真正看见的帧数。
    seen_frames: int
    #: 可见帧质心的均值（没看见过则为 None）。
    centroid: tuple[float, float] | None
    #: 首次可见帧的质心——垂向漂移的参照。
    anchor_centroid: tuple[float, float] | None
    #: 可见帧面积的均值（px²）。
    mean_area_px: float
    #: 末次可见帧的面积比。
    area_ratio: float | None
    #: 整段都没看见过（被交出过且没接回，或全程遮挡）。
    never_seen: bool

    @classmethod
    def from_curve(cls, curve: BagCurve, declared_slot: int | None) -> "TrackView":
        centroids = getattr(curve, "frame_centroids", None) or []
        areas = [value for value in (getattr(curve, "frame_areas_px", None) or []) if value]
        seen = [
            centroid
            for centroid, ratio in zip(centroids, curve.area_ratio_series)
            if centroid is not None and ratio is not None
        ]
        mean_centroid = None
        if seen:
            mean_centroid = (
                float(np.mean([item[0] for item in seen])),
                float(np.mean([item[1] for item in seen])),
            )
        return cls(
            obj_id=curve.obj_id,
            declared_slot=declared_slot,
            seen_frames=len(seen),
            centroid=mean_centroid,
            anchor_centroid=seen[0] if seen else None,
            mean_area_px=float(np.mean(areas)) if areas else 0.0,
            area_ratio=curve.latest_ratio,
            never_seen=not seen,
        )


@dataclass(frozen=True)
class SlotScore:
    """一对（轨迹, 挂点）的打分与判据拆解，便于护士端与复核逐条看。"""

    track_id: str
    slot_index: int
    total: float
    slot_score: float
    centroid_score: float
    area_score: float

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "slot_index": self.slot_index,
            "total": round(self.total, 4),
            "slot_score": round(self.slot_score, 4),
            "centroid_score": round(self.centroid_score, 4),
            "area_score": round(self.area_score, 4),
        }


@dataclass(frozen=True)
class SlotAssignment:
    """一条轨迹的分配结论。``state == ASSIGNED`` 时 ``slot_index`` 才可信。"""

    obj_id: str
    declared_slot: int | None
    slot_index: int | None
    state: str
    reason: str
    score: float | None = None
    margin: float | None = None
    runner_up_slot: int | None = None
    runner_up_track: str | None = None
    evidence: dict = field(default_factory=dict)
    #: 中文说明：给护士看的一句话，能照着做。
    detail: str = ""

    @property
    def assigned(self) -> bool:
        return self.state == ASSIGNED

    @property
    def claim_required(self) -> bool:
        """是否必须让护士在列表里点一行认领。

        ``assigned`` 与 ``no_evidence`` 之外的都算：``mismatched`` 是「对不上」，
        ``ambiguous``/``below_score`` 是「分不出」，``extra_track`` 是「多出来一条」，
        ``unfilled`` 是「这个挂点没拍到」。这些情形下产品给不出可信的袋序对应。
        """
        return self.state != ASSIGNED

    def to_dict(self) -> dict:
        return {
            "obj_id": self.obj_id,
            "bag_seq": self.evidence.get("bag_seq"),
            "declared_slot": self.declared_slot,
            "slot_index": self.slot_index,
            "state": self.state,
            "reason": self.reason,
            "score": None if self.score is None else round(float(self.score), 4),
            "margin": None if self.margin is None else round(float(self.margin), 4),
            "runner_up_slot": self.runner_up_slot,
            "runner_up_track": self.runner_up_track,
            "claim_required": self.claim_required,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class SlotAssignmentResult:
    """一次分配的全体结论：逐轨迹 + 逐挂点 + 护士认领列表。"""

    assignments: tuple[SlotAssignment, ...]
    slot_states: dict[int, SlotAssignment]

    def by_obj(self, obj_id: str) -> SlotAssignment | None:
        for item in self.assignments:
            if item.obj_id == obj_id:
                return item
        return None

    def by_slot(self, slot_index: int) -> SlotAssignment | None:
        return self.slot_states.get(slot_index)

    @property
    def claimable(self) -> tuple[SlotAssignment, ...]:
        return tuple(item for item in self.assignments if item.claim_required)

    def claim_rows(self) -> list[dict]:
        """护士认领列表：点一行认领一只袋子，不是逐袋点像素。

        只列**需要认领**的行；已确认的行不在列表里出现，免得护士在一屏里翻已经认过的。
        """
        rows = []
        for item in self.claimable:
            rows.append(
                {
                    "bag_seq": item.evidence.get("bag_seq"),
                    "slot_index": item.declared_slot,
                    "track_id": item.obj_id,
                    "state": item.state,
                    "reason": item.reason,
                    "score": None if item.score is None else round(float(item.score), 4),
                    "detail": item.detail,
                }
            )
        return rows

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for item in self.assignments:
            counts[item.state] = counts.get(item.state, 0) + 1
        return {
            "track_count": len(self.assignments),
            "slot_count": len(self.slot_states),
            "by_state": counts,
            "claim_required": sum(1 for item in self.assignments if item.claim_required),
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "assignments": [item.to_dict() for item in self.assignments],
        }


def _pair_score(
    view: TrackView,
    slot: SlotColumn,
    median_area_px: float,
    frame_height: float,
    params: SlotAssignmentParams,
) -> SlotScore | None:
    """给一对（轨迹, 挂点）打分并过硬门限；任一门限不过返回 ``None``（不猜）。

    三个通道各管一件事，互不代替：

    1. **槽位竖列先验**：轨迹质心 x 离这条竖列的列心有多远（挂点是物理事实，最重）；
    2. **质心垂向漂移**：整段可见帧的质心 y 与首次可见帧差多少（袋子挂在架上，不该乱跑）；
    3. **面积量级**：与同架其他轨迹的中位面积是否同档（挡住并袋成一块、或碎块起轨迹）。
    """
    if view.centroid is None or view.anchor_centroid is None:
        return None

    centre_x, _centre_y = view.centroid
    slot_score = max(0.0, 1.0 - abs(centre_x - slot.centre_x) / (slot.half_width * params.slot_centre_tolerance))
    if slot_score < params.min_slot_score:
        return None

    drift = abs(view.centroid[1] - view.anchor_centroid[1])
    tolerance = max(1e-6, frame_height * params.centroid_drift_tolerance)
    centroid_score = max(0.0, 1.0 - drift / tolerance)
    if centroid_score < params.min_centroid_score:
        return None

    if median_area_px > 0 and view.mean_area_px > 0:
        ratio = view.mean_area_px / median_area_px
        deviation = abs(ratio - 1.0)
        # 硬门限：偏离中位面积超过 max_area_deviation 倍即淘汰（并袋 / 残块）。
        # 这一条在加权之前判——不然一条并成两袋的轨迹即使面积离谱，仍能靠槽位与质心
        # 两个通道凑过 min_total（实测 0.75 ≥ 0.45），并袋就会变成一条「看起来很稳」的曲线。
        if deviation > params.max_area_deviation:
            return None
        area_score = max(0.0, 1.0 - deviation / params.area_scale_tolerance)
    else:
        # 没有同架可比的中位面积（只剩一条轨迹）：这一路**不参与**加权，
        # 而不是当成 0 分把证据齐备的轨迹系统性压到门槛以下。
        area_score = -1.0

    channels = [
        (params.slot_prior_weight, slot_score),
        (params.centroid_weight, centroid_score),
    ]
    if area_score >= 0.0:
        channels.append((params.area_weight, area_score))
    weight_sum = sum(weight for weight, _ in channels)
    total = sum(weight * value for weight, value in channels) / weight_sum

    if total < params.min_total:
        return None
    return SlotScore(
        track_id=view.obj_id,
        slot_index=slot.slot_index,
        total=round(total, 6),
        slot_score=slot_score,
        centroid_score=centroid_score,
        area_score=area_score,
    )


def _detail(
    view: TrackView,
    slot: SlotColumn | None,
    state: str,
    reason: str,
    runner_up_slot: int | None = None,
) -> str:
    """给护士看的一句话：说清楚「现在是什么情况、下一步做什么」。"""
    bag_seq = view.obj_id
    if state == ASSIGNED and slot is not None:
        return f"第 {slot.slot_index + 1} 个挂点上的 {bag_seq} 与这条曲线对得上，直接用。"
    if state == MISMATCHED and slot is not None:
        return (
            f"{bag_seq} 声明在第 {view.declared_slot + 1} 个挂点，但这条曲线停在第 "
            f"{slot.slot_index + 1} 个挂点那一列——可能拍到隔壁那只了，请在列表里点一行认领。"
        )
    if state == AMBIGUOUS:
        target = f"第 {(runner_up_slot or 0) + 1} 个挂点" if runner_up_slot is not None else "这个挂点"
        return f"{target} 上有两只分不出先后，不猜，请在列表里点一行认领。"
    if state == NO_EVIDENCE:
        return f"{bag_seq} 整段都没看清，不给它配挂点，请把镜头对准这一挂点重拍一段。"
    if state == UNFILLED:
        return f"第 {(slot.slot_index if slot else 0) + 1} 个挂点这段没拍到袋子，请确认这一挂位。"
    if state == EXTRA_TRACK:
        return f"画面里多出一条轨迹（{bag_seq}），输注单上没给它挂点，请确认它是不是袋子。"
    return f"{bag_seq} 暂时配不到挂点（{reason}），请在列表里点一行认领。"


def assign_slots(
    views: Sequence[TrackView],
    columns: Sequence[SlotColumn],
    *,
    frame_shape: tuple[int, int] | None = None,
    params: SlotAssignmentParams | None = None,
) -> SlotAssignmentResult:
    """把「每条轨迹属于哪个挂点」算出来，并给出**认不准就不认**的结论。

    算法是三步，全程不猜：

    1. **逐轨迹消歧**：每条轨迹先看自己在所有挂点上的得分，前两名分差不足 ``min_margin``
       即判 ``ambiguous``，直接出局（不参与后面的配对）；
    2. **逐挂点消歧**：每个挂点同样看前两名轨迹，分差不足即把这条挂点整个标成待认领；
    3. **全局一对一贪心配对**：剩下的（轨迹, 挂点）按总分从高到低配，一条轨迹只占一个挂点、
       一个挂点只接一条轨迹。

    剩下的收尾：整段没看清的轨迹给 ``no_evidence`` 且**不占**任何挂点；没被任何轨迹命中的
    挂点给 ``unfilled``；不落在任何挂点列上的轨迹给 ``extra_track``。
    """
    params = params or SlotAssignmentParams()
    height = float(frame_shape[0]) if frame_shape else 640.0

    areas = [view.mean_area_px for view in views if view.mean_area_px > 0]
    median_area = float(np.median(areas)) if areas else 0.0

    columns_by_index = {column.slot_index: column for column in columns}
    # ── 步骤 1：逐轨迹打分 + 逐轨迹消歧 ──────────────────────────────────────
    scores_by_track: dict[str, dict[int, SlotScore]] = {}
    ambiguous: dict[str, SlotAssignment] = {}
    never_seen: dict[str, SlotAssignment] = {}
    for view in views:
        if view.never_seen:
            never_seen[view.obj_id] = SlotAssignment(
                obj_id=view.obj_id,
                declared_slot=view.declared_slot,
                slot_index=None,
                state=NO_EVIDENCE,
                reason=REASON_NEVER_SEEN,
                evidence={"bag_seq": view.obj_id, "seen_frames": 0},
                detail=_detail(view, None, NO_EVIDENCE, REASON_NEVER_SEEN),
            )
            continue
        per_slot: dict[int, SlotScore] = {}
        for column in columns:
            score = _pair_score(view, column, median_area, height, params)
            if score is not None:
                per_slot[column.slot_index] = score
        scores_by_track[view.obj_id] = per_slot
        ranked = sorted(per_slot.values(), key=lambda item: (-item.total, item.slot_index))
        if len(ranked) >= 2 and (ranked[0].total - ranked[1].total) < params.min_margin:
            ambiguous[view.obj_id] = SlotAssignment(
                obj_id=view.obj_id,
                declared_slot=view.declared_slot,
                slot_index=None,
                state=AMBIGUOUS,
                reason=REASON_AMBIGUOUS_SLOT,
                score=ranked[0].total,
                margin=ranked[0].total - ranked[1].total,
                runner_up_slot=ranked[1].slot_index,
                evidence={
                    "bag_seq": view.obj_id,
                    "best_slot": ranked[0].slot_index,
                    "runner_up_slot": ranked[1].slot_index,
                },
                detail=_detail(view, None, AMBIGUOUS, REASON_AMBIGUOUS_SLOT, ranked[1].slot_index),
            )
        elif not ranked:
            ambiguous[view.obj_id] = SlotAssignment(
                obj_id=view.obj_id,
                declared_slot=view.declared_slot,
                slot_index=None,
                state=BELOW_SCORE,
                reason=REASON_SLOT_PRIOR,
                evidence={"bag_seq": view.obj_id},
                detail=_detail(view, None, BELOW_SCORE, REASON_SLOT_PRIOR),
            )

    # ── 步骤 2：逐挂点消歧 ──────────────────────────────────────────────────
    live = {
        obj_id: per_slot
        for obj_id, per_slot in scores_by_track.items()
        if obj_id not in ambiguous
    }
    ambiguous_slots: dict[int, str] = {}
    for column in columns:
        ranked = sorted(
            (
                (per_slot[column.slot_index].total, obj_id)
                for obj_id, per_slot in live.items()
                if column.slot_index in per_slot
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if len(ranked) >= 2 and (ranked[0][0] - ranked[1][0]) < params.min_margin:
            ambiguous_slots[column.slot_index] = ranked[1][1]

    # ── 步骤 3：全局一对一贪心配对 ──────────────────────────────────────────
    pairs: list[tuple[float, str, int]] = []
    for obj_id, per_slot in live.items():
        for slot_index, score in per_slot.items():
            if slot_index in ambiguous_slots:
                continue
            pairs.append((score.total, obj_id, slot_index))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    taken_tracks: set[str] = set()
    taken_slots: set[int] = set()
    accepted: dict[str, SlotAssignment] = {}
    runner_up_by_track: dict[str, int] = {}
    for _total, obj_id, slot_index in pairs:
        if obj_id in taken_tracks or slot_index in taken_slots:
            other = runner_up_by_track.get(obj_id)
            if other is None:
                runner_up_by_track[obj_id] = slot_index
            continue
        view = next(item for item in views if item.obj_id == obj_id)
        column = columns_by_index[slot_index]
        per_slot = scores_by_track[obj_id]
        others = sorted(
            (score.total for index, score in per_slot.items() if index != slot_index),
            reverse=True,
        )
        taken_tracks.add(obj_id)
        taken_slots.add(slot_index)
        # 配到**别的**挂点 = 袋序错位。这一条**不自动改绑**：换液点名与上一轮曲线都按
        # bag_seq 走，认错了就是把 A 袋的毫升数报给 B 袋。错位一律交护士在列表里认领。
        # 未声明挂点的轨迹（输注单上根本没有这一袋）归 extra_track，不算「错位」。
        if view.declared_slot is None:
            state, reason = EXTRA_TRACK, REASON_NO_SLOT
        elif slot_index == view.declared_slot:
            state, reason = ASSIGNED, REASON_ASSIGNED
        else:
            state, reason = MISMATCHED, REASON_DECLARED_SLOT_MISMATCH
        accepted[obj_id] = SlotAssignment(
            obj_id=obj_id,
            declared_slot=view.declared_slot,
            slot_index=slot_index,
            state=state,
            reason=reason,
            score=per_slot[slot_index].total,
            margin=(per_slot[slot_index].total - others[0]) if others else None,
            runner_up_slot=(
                min(
                    (index for index in per_slot if index != slot_index),
                    key=lambda index: -per_slot[index].total,
                )
                if others
                else None
            ),
            evidence={
                "bag_seq": view.obj_id,
                "slot_score": round(per_slot[slot_index].slot_score, 4),
                "centroid_score": round(per_slot[slot_index].centroid_score, 4),
                "area_score": (
                    round(per_slot[slot_index].area_score, 4)
                    if per_slot[slot_index].area_score >= 0
                    else None
                ),
                "seen_frames": view.seen_frames,
                "mean_area_px": round(view.mean_area_px, 1),
                "observed_slot": slot_index,
            },
            detail=_detail(
                view,
                column,
                ASSIGNED if view.declared_slot is not None and slot_index == view.declared_slot
                else (EXTRA_TRACK if view.declared_slot is None else MISMATCHED),
                REASON_ASSIGNED if view.declared_slot is not None and slot_index == view.declared_slot
                else (REASON_NO_SLOT if view.declared_slot is None else REASON_DECLARED_SLOT_MISMATCH),
            ),
        )

    # ── 收尾：没配上的轨迹各归其因，没被认领的挂点报 unfilled ────────────────
    assignments: list[SlotAssignment] = []
    for view in views:
        if view.obj_id in accepted:
            assignments.append(accepted[view.obj_id])
            continue
        if view.obj_id in never_seen:
            assignments.append(never_seen[view.obj_id])
            continue
        if view.obj_id in ambiguous:
            assignments.append(ambiguous[view.obj_id])
            continue
        # 到这里：至少有一个挂点过了门限（否则上一轮已归入 ambiguous/below_score），
        # 但在全局配对里被别人先占了 → 它跟声明的挂点对不上，或落在所有挂点列之外。
        per_slot = scores_by_track.get(view.obj_id, {})
        if not per_slot:
            assignments.append(
                SlotAssignment(
                    obj_id=view.obj_id,
                    declared_slot=view.declared_slot,
                    slot_index=None,
                    state=BELOW_SCORE,
                    reason=REASON_SLOT_PRIOR,
                    detail=_detail(view, None, BELOW_SCORE, REASON_SLOT_PRIOR),
                )
            )
            continue
        best_slot = min(per_slot, key=lambda index: -per_slot[index].total)
        if best_slot in ambiguous_slots:
            # 这个挂点本身就有争议（步骤 2 里两条轨迹分不出先后），于是这一列**谁也没配到**。
            # 这种情形报歧义而不是「错位」：错位说的是位置对不上，这里说的是位置对得上但
            # 认不出是哪一只。按「错位」报会把争议硬塞给护士去比大小，方向就错了。
            rival = ambiguous_slots[best_slot]
            assignments.append(
                SlotAssignment(
                    obj_id=view.obj_id,
                    declared_slot=view.declared_slot,
                    slot_index=None,
                    state=AMBIGUOUS,
                    reason=REASON_AMBIGUOUS_SLOT,
                    score=per_slot[best_slot].total,
                    runner_up_slot=best_slot,
                    runner_up_track=None if rival == view.obj_id else rival,
                    evidence={"bag_seq": view.obj_id, "contested_slot": best_slot},
                    detail=(
                        f"第 {best_slot + 1} 个挂点上两条轨迹分不出先后"
                        f"（这一只与 {rival}），不猜，请点一行认领。"
                    ),
                )
            )
            continue
        if view.declared_slot is None:
            assignments.append(
                SlotAssignment(
                    obj_id=view.obj_id,
                    declared_slot=None,
                    slot_index=None,
                    state=EXTRA_TRACK,
                    reason=REASON_NO_SLOT,
                    score=per_slot[best_slot].total,
                    detail=_detail(view, None, EXTRA_TRACK, REASON_NO_SLOT),
                )
            )
            continue
        if best_slot in taken_slots:
            # 这个挂点已经被别的轨迹拿走了：两条轨迹在抢同一列。**不给任何一条第二个
            # slot_index**（否则响应里会出现两个袋次指向同一个挂点），按「抢同一个挂点」
            # 报歧义或错位，交给护士认领。
            holder = next(
                obj for obj in taken_tracks if accepted[obj].slot_index == best_slot
            )
            if best_slot == view.declared_slot:
                assignments.append(
                    SlotAssignment(
                        obj_id=view.obj_id,
                        declared_slot=view.declared_slot,
                        slot_index=None,
                        state=AMBIGUOUS,
                        reason=REASON_AMBIGUOUS_TRACK,
                        score=per_slot[best_slot].total,
                        runner_up_slot=best_slot,
                        runner_up_track=holder,
                        detail=(
                            f"第 {best_slot + 1} 个挂点上两条轨迹抢同一个位置"
                            f"（{view.obj_id} 与 {holder}），不猜，请点一行认领。"
                        ),
                    )
                )
                continue
            assignments.append(
                SlotAssignment(
                    obj_id=view.obj_id,
                    declared_slot=view.declared_slot,
                    slot_index=None,
                    state=MISMATCHED,
                    reason=REASON_DECLARED_SLOT_MISMATCH,
                    score=per_slot[best_slot].total,
                    runner_up_track=holder,
                    evidence={"bag_seq": view.obj_id, "observed_slot": best_slot},
                    detail=_detail(
                        view, columns_by_index[best_slot], MISMATCHED, REASON_DECLARED_SLOT_MISMATCH
                    ),
                )
            )
            continue
        column = columns_by_index[best_slot]
        assignments.append(
            SlotAssignment(
                obj_id=view.obj_id,
                declared_slot=view.declared_slot,
                slot_index=best_slot,
                state=MISMATCHED,
                reason=REASON_DECLARED_SLOT_MISMATCH,
                score=per_slot[best_slot].total,
                runner_up_track=ambiguous_slots.get(view.declared_slot),
                evidence={"bag_seq": view.obj_id, "observed_slot": best_slot},
                detail=_detail(view, column, MISMATCHED, REASON_DECLARED_SLOT_MISMATCH),
            )
        )

    slot_states: dict[int, SlotAssignment] = {}
    for column in columns:
        holder = next(
            (item for item in accepted.values() if item.slot_index == column.slot_index),
            None,
        )
        if holder is not None:
            slot_states[column.slot_index] = holder
            continue
        rival = ambiguous_slots.get(column.slot_index)
        if rival is not None:
            slot_states[column.slot_index] = SlotAssignment(
                obj_id=rival,
                declared_slot=column.slot_index,
                slot_index=column.slot_index,
                state=AMBIGUOUS,
                reason=REASON_AMBIGUOUS_TRACK,
                runner_up_slot=column.slot_index,
                evidence={"bag_seq": rival, "slot_index": column.slot_index},
                detail=f"第 {column.slot_index + 1} 个挂点上有两条候选分不出先后，不猜，请点一行认领。",
            )
            continue
        slot_states[column.slot_index] = SlotAssignment(
            obj_id="",
            declared_slot=column.slot_index,
            slot_index=column.slot_index,
            state=UNFILLED,
            reason=REASON_SLOT_EMPTY,
            evidence={"slot_index": column.slot_index, "bag_seq": column.bag_seq},
            detail=f"第 {column.slot_index + 1} 个挂点这段没拍到袋子，请确认这一挂位。",
        )

    return SlotAssignmentResult(
        assignments=tuple(assignments), slot_states=slot_states
    )


@dataclass(frozen=True)
class BatchFrame:
    """一帧的多对象产出（对应源 ``propagate_in_video`` 每次 yield 的三元组）。

    ``object_ids`` 是**这一帧实际有掩膜的对象 id**，顺序按起锚顺序固定；掩膜本身按 id 归位，
    不按位置猜（源实现的 ``{out_obj_id: logits[i] ... for i, out_obj_id in enumerate(out_obj_ids)}``）。
    """

    frame_index: int
    frame_name: str
    object_ids: tuple[str, ...]
    areas_px: dict[str, int]


@dataclass(frozen=True)
class BatchPropagation:
    """一次多袋并行传播的产出。"""

    curves: dict[str, BagCurve]
    ordered: tuple[BagCurve, ...]
    frames: tuple[BatchFrame, ...]
    anchor_ids: frozenset[str]
    slot_of: dict[str, int | None]
    frame_shape: tuple[int, int]

    def views(self) -> tuple[TrackView, ...]:
        return tuple(
            TrackView.from_curve(self.curves[obj_id], self.slot_of.get(obj_id))
            for obj_id in self.ordered_ids
        )

    @property
    def ordered_ids(self) -> tuple[str, ...]:
        return tuple(curve.obj_id for curve in self.ordered)

    @property
    def predict_calls(self) -> int:
        """整段传播共发起多少次预测（= 帧数 × 对象数，不含再获取的重扫）。"""
        return sum(len(frame.object_ids) for frame in self.frames)


def propagate_batch(
    frame_paths: Sequence[Path],
    anchors: Sequence[TrackAnchor | tuple[str, Prompt]],
    predictor,
    *,
    anchor_frame: int = 0,
    reacquire: ReacquireParams | None | object = DEFAULT_REACQUIRE,
    params: SlotAssignmentParams | None = None,
    visibility_params=None,
    _slot_of: dict[str, int] | None = None,
) -> BatchPropagation:
    """**一个状态、一遍传播**跑完整架的袋子。

    这是多袋并行的唯一入口：与逐袋各跑一遍不同，这里 6 只袋子共用一个
    :class:`..propagate.PropagationState`、一次 :func:`..propagate.propagate_in_video`，
    因此①同一帧的多只袋子互相可比（面积量级先验要靠这个），②算力不随袋数线性膨胀，
    ③掩膜按 ``obj_id`` 归位而不是按遍历顺序。

    起锚时把身份集合**冻结**（源 ``vos_inference.py:193-204`` 的 ``object_ids_set``）：起锚之后
    再往状态里加新 id 会抛 :class:`..propagate.ObjectIdSetFrozen`——中途多出一只袋子只能走
    :mod:`.reacquire` 的再获取或护士的认领，**不能凭空插进传播里**。
    """
    params = params or SlotAssignmentParams()
    if not frame_paths:
        return BatchPropagation({}, (), (), frozenset(), {}, (0, 0))
    normalised: list[TrackAnchor] = []
    for item in anchors:
        normalised.append(
            item if isinstance(item, TrackAnchor) else TrackAnchor(obj_id=item[0], prompt=item[1])
        )
    if len({anchor.obj_id for anchor in normalised}) != len(normalised):
        raise ValueError("同一批锚点的 obj_id 必须互不相同")
    if len(normalised) > params.max_bags:
        raise ValueError(
            f"单架最多 {params.max_bags} 只袋子，收到 {len(normalised)} 只；"
            "请拆成两次上传，不要把两个输液架混在一段短片里"
        )

    first = load_rgb(frame_paths[0])
    shape = first.shape
    state: PropagationState = init_state([path.name for path in frame_paths], shape)

    cavities: dict[str, int] = {}
    if hasattr(predictor, "estimate_cavity"):
        estimator = predictor.estimate_cavity
        for anchor in normalised:
            estimate = estimator(first, anchor.prompt)
            cavities[anchor.obj_id] = int(getattr(estimate, "area_px", 0))
    if not cavities:
        cavities = {anchor.obj_id: 1 for anchor in normalised}
        logger.warning("袋腔估计不可用，area_ratio 退化为归一化的液体面积占比")
    else:
        # **按袋降级，不按架降级**：某一袋在起锚帧就看不见（全程被挡板盖住、或压根没挂）时
        # 它的袋腔估不出来，但那只袋子没有面积比 ≠ 同一架其余袋子的面积比也没了。早期实现
        # 是「任意一袋估不出来就整架退化成面积比 1.0」，一份遮挡会把另外五只袋子的余量
        # 一起变成假读数——这正是「因为缺一条证据而系统性拖累其余」的老毛病。
        degraded = [obj_id for obj_id, area in cavities.items() if area <= 0]
        if degraded:
            logger.warning(
                "袋腔估计不可用，以下袋子的 area_ratio 退化为归一化的液体面积占比：%s",
                ",".join(sorted(degraded)),
            )
        cavities = {obj_id: area if area > 0 else 1 for obj_id, area in cavities.items()}

    slot_of: dict[str, int | None] = {}
    # 一个循环把整架袋子加进同一个状态——源实现 add_new_mask 的那个 for 循环。
    for index, anchor in enumerate(normalised):
        slot_index = (
            anchor.slot_index
            if anchor.slot_index is not None
            else (_slot_of or {}).get(anchor.obj_id, index)
        )
        slot_of[anchor.obj_id] = slot_index
        add_new_points_or_box(
            state,
            frame_index=anchor_frame,
            obj_id=anchor.obj_id,
            prompt=anchor.prompt,
            cavity_area_px=cavities[anchor.obj_id],
            visibility_params=visibility_params,
            slot_index=slot_index,
        )
    # 身份集合在此冻结：起锚之后再加新 id 一律拒绝（纪律：不猜、不凭空多一只袋子）
    state.frozen_object_ids = frozenset(state.objects)

    by_name = {path.name: path for path in frame_paths}
    frame_records: list[BatchFrame] = []
    per_frame_centroids: dict[str, list[tuple[float, float] | None]] = {
        anchor.obj_id: [] for anchor in normalised
    }
    per_frame_areas: dict[str, list[int | None]] = {
        anchor.obj_id: [] for anchor in normalised
    }
    for result in propagate_in_video(
        state,
        predictor,
        lambda name: load_rgb(by_name[name]),
        reacquire=reacquire,
    ):
        areas_px: dict[str, int] = {}
        for obj_id, track in state.objects.items():
            mask = result.masks.get(obj_id)
            centroid = mask_centroid(mask) if mask is not None else None
            per_frame_centroids[obj_id].append(centroid)
            per_frame_areas[obj_id].append(None if mask is None else int(np.count_nonzero(mask)))
            if mask is not None:
                areas_px[obj_id] = int(np.count_nonzero(mask))
        frame_records.append(
            BatchFrame(
                frame_index=result.frame_index,
                frame_name=result.frame_name,
                object_ids=tuple(
                    obj_id for obj_id in state.objects if obj_id in result.masks
                ),
                areas_px=areas_px,
            )
        )

    ordered = tuple(
        _curve_with_geometry(
            track,
            per_frame_centroids[track.obj_id],
            per_frame_areas[track.obj_id],
        )
        for track in state.objects.values()
    )
    return BatchPropagation(
        curves={curve.obj_id: curve for curve in ordered},
        ordered=ordered,
        frames=tuple(frame_records),
        anchor_ids=frozenset(state.objects),
        slot_of=slot_of,
        frame_shape=shape,
    )


def _curve_with_geometry(
    track: ObjectTrack,
    centroids: list[tuple[float, float] | None],
    areas: list[int | None],
) -> BagCurve:
    """把一条轨迹的产出压成 :class:`..propagate.BagCurve`（含逐帧质心/面积，供分配层用）。"""
    return BagCurve(
        obj_id=track.obj_id,
        frame_indices=list(range(track.frame_count)),
        area_ratio_series=list(track.ratios),
        occlusion_flags=list(track.occlusion_flags),
        cavity_area_px=track.cavity_area_px,
        scores=list(track.scores),
        anchor=(
            (track.prompt.positive_points[0].x, track.prompt.positive_points[0].y)
            if track.prompt.positive_points
            else None
        ),
        visibility_states=list(track.visibility.states),
        handoff=track.visibility.handoff,
        reacquires=list(track.reacquires),
        frame_centroids=centroids,
        frame_areas_px=areas,
    )