"""再获取：丢失窗口用尽后，把这只袋子从后续帧里找回来（不新建轨迹、不接错曲线）。

**这条路径为什么必须存在**：:mod:`.occlusion` 的可见度状态机在连续遮挡超过
``lost_window_frames`` 后把轨迹**交出**（``ReacquireHandoff``）并冻结曲线。冻结是对的——
拿一条编出来的曲线去算毫升和 ETA 比没有曲线更危险；但「交出」本身不解决问题：病人还挂着
输液，袋子迟早会回到画面里。**交出之后由谁接回来、凭什么接**，就是本模块的职责。

**与源主路径的对应关系**（只借调用链与语义，不搬代码、不跑源权重）：

===========================================  ==========================================================
本模块                                       源主路径（代码证据）
===========================================  ==========================================================
:meth:`LostTrackView.from_handoff`          ``vos_inference.py:145-173`` 的 ``use_all_masks``：输入提示
                                             **不必只在首帧**，可以在后续帧补一段掩膜继续跟同一条对象
:attr:`ReacquireVerdict.state` 的身份不变性  ``vos_inference.py:193-204``：``object_ids_set`` 在**第一条**输入帧
                                             上定死，后面出现新 ``object_id`` 直接 ``RuntimeError``。
                                             本产品等价物：再获取只把候选接到**原 ``bag_seq``**，
                                             中途凭空多出一条 id 是不可能的
:func:`reacquire_tracks` 的局部重枚举        :mod:`.enumerate`（task 5 的首帧枚举管线），对应
                                             ``automatic_mask_generator.py`` 的 ``_process_crop``：
                                             裁一块 ROI、在这块里按网格扫、补漏、换回原图坐标
:func:`reacquire_tracks` 的「重新给提示」    ``vos_inference.py:205-210`` 的
                                             ``predictor.add_new_mask(frame_idx=…, obj_id=…, mask=…)``：
                                             换一帧、换一条提示、**对象 id 不变**
再获取为什么必要                              ``memory_attention.py:119-161`` 的 ``num_obj_ptr_tokens``：
                                             对象指针记忆读不到时不参与推理，等外部重新给提示
===========================================  ==========================================================

**四条纪律**（与 :mod:`.occlusion` / :mod:`.propagate` 一致）：

1. **不新建轨迹**：再获取成功只是把候选接回**原来的** ``bag_seq`` 与原来的记忆库；
   失败则保持 ``lost``，由护士在列表里认领（:mod:`..ingest` 的 ``/claim``）。
2. **不接错**：三条独立证据（记忆几何 / 槽位几何先验 / 外观描述子）加权打分，任一条
   硬门限不过即拒绝；前两名分差不足 ``min_margin`` 一律判歧义，**不默认取最高分**。
3. **不把新袋接到旧曲线**：换液后新袋**更满**，面积比会不降反升。候选余量比交出时的参考
   余量高出 ``max_ratio_regain`` 以上即判「换成了新袋」，直接拒绝接回并要求人工认领。
4. **恒定算力**：一次只服务 ``batch_scan_limit`` 条 lost 轨迹，且只在**一条**并集 ROI 上
   跑一次枚举（不是每条轨迹各扫一遍）；ROI 面积有上限，与丢失轨迹条数无关。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..identity.reanchor import BagDescriptor, describe_region, score_pair
from .enumerate import (
    EnumerationParams,
    MaskCandidate,
    enumerate_candidates,
)
from .memory_bank import MemoryEntry
from .occlusion import LOST, ReacquireHandoff, VisibilityParams
from .predictor import MaskPrediction, PerPointMaskPredictor, mask_centroid

__all__ = [
    "REACQUIRED",
    "UNBOUND",
    "STILL_LOST",
    "REACQUIRE_STATES",
    "REASON_OK",
    "REASON_NOT_LOST",
    "REASON_ROI_EMPTY",
    "REASON_BELOW_SCORE",
    "REASON_AMBIGUOUS",
    "REASON_SLOT_PRIOR",
    "REASON_MEMORY_MISMATCH",
    "REASON_REFILL_GAIN",
    "REASON_TAKEN",
    "REASON_BATCH_LIMIT",
    "REASON_NO_REFERENCE",
    "ReacquireParams",
    "LostTrackView",
    "ReacquireCandidate",
    "ReacquireVerdict",
    "ReacquireBatch",
    "roi_for_tracks",
    "reacquire_tracks",
]

logger = logging.getLogger("infusion_monitor.core.reacquire")

#: 在后续帧找回同一条轨迹，曲线接回原来的 ``bag_seq``。
REACQUIRED = "reacquired"
#: 找到了东西但认不准（歧义 / 门限不过 / 外观不可用）——交人工认领，不接回。
UNBOUND = "unbound"
#: 本轮没能找回（ROI 里没东西、或候选被别人先占），保持丢失状态。
#: 与 :data:`.occlusion.LOST` 同值——「没接回来」在可见度序列里就仍然记 lost。
STILL_LOST = LOST

REACQUIRE_STATES = (REACQUIRED, UNBOUND, STILL_LOST)

# ── 原因码（进审计与护士端，均为可读中文短语的稳定串）──────────────────
#: 接回成功。
REASON_OK = "reacquired"
#: 这条轨迹还没被交出（``handoff is None``）——不触发再获取，避免与遮挡恢复打架。
REASON_NOT_LOST = "not_lost_no_reacquire"
#: ROI 内一个候选都没枚举出来。
REASON_ROI_EMPTY = "roi_no_candidate"
#: 有候选，但最高分没过 ``min_score``。
REASON_BELOW_SCORE = "below_threshold"
#: 前两名分差不足 ``min_margin``——认不准，不猜。
REASON_AMBIGUOUS = "ambiguous_candidates"
#: 候选离该袋位的槽位中心太远。
REASON_SLOT_PRIOR = "slot_prior_violation"
#: 候选与「最后可见那一帧」的几何对不上（不是同一只袋子）。
REASON_MEMORY_MISMATCH = "memory_geometry_mismatch"
#: 候选余量反而回升——这是换上了新袋，不能接到旧曲线上。
REASON_REFILL_GAIN = "refilled_bag_not_connected"
#: 最佳候选已被另一条 lost 轨迹先认走（一条候选只接一条轨迹）。
REASON_TAKEN = "candidate_taken_by_other_track"
#: 本轮 lost 轨迹数超过 ``batch_scan_limit``，超出的不参与本轮再获取。
REASON_BATCH_LIMIT = "batch_limit_exceeded"
#: 交出单里连参考余量都没有（从未成功看见过），无从判断「是不是换袋」。
REASON_NO_REFERENCE = "no_reference_ratio"


@dataclass(frozen=True)
class ReacquireParams:
    """再获取门限。默认值按合成场景标定，真机上由信息科按病房实测调整。

    **每一项都能解释给护士听**，出问题能一眼看出是哪一项把这次接回挡掉了。
    """

    # ── ROI 与算力 ──
    #: 丢失位置外扩比例（袋体可能被挪动、被家属挡住后重新挂回）。
    roi_pad_ratio: float = 0.45
    #: 外扩的像素下限（近景小袋也要留一点余量）。
    roi_min_pad_px: int = 24
    #: ROI 面积上限（像素）——恒定算力的闸，与丢失轨迹条数无关。
    roi_max_area_px: int = 480 * 480
    #: ROI 内重扫的网格密度（比首帧整幅稀疏：区域小得多，袋子也少得多）。
    points_per_side: int = 4
    #: 补漏重扫的密度倍率。
    refill_points_scale: int = 2
    #: 候选稳定分下限，与首帧枚举同口径（低分块不拿回来当答案）。
    stability_score_thresh: float = 0.90
    #: 一次最多服务几条 lost 轨迹（再多留到下一帧）。
    batch_scan_limit: int = 4
    #: 交出后再等几帧才试再获取——避免在同一片遮挡上反复试。
    retry_gap_frames: int = 2

    # ── 三条证据的权重 ──
    memory_weight: float = 0.45
    slot_weight: float = 0.25
    appearance_weight: float = 0.30

    # ── 硬门限 ──
    #: 与「最后可见那一帧」的吻合度下限（袋底 / 质心 / 横向宽度三项）。
    min_memory_agreement: float = 0.35
    #: 外观分硬门限（仅当外观**可用**时生效）：低于它即判「不是同一只袋子」。
    #:
    #: 这条比 :mod:`..identity.reanchor` 的 ``min_confidence`` 更严，是**刻意**的：
    #: 重锚错了下一轮还能纠正，而再获取错了会把两只袋子的曲线接成一条，毫升和 ETA
    #: 从此全是假的。同挂点、同袋型、同颜色的两只袋子色相直方图交集本就在 0.8 以上，
    #: 0.30 已经是「明显不像」的水位。
    min_appearance: float = 0.30
    #: 槽位几何先验分下限。
    min_slot_score: float = 0.25
    #: 槽位中心距容差（占画面宽度比例）。
    slot_centre_tolerance: float = 0.22
    #: 总分门槛。
    min_score: float = 0.62
    #: 与次优候选的最小分差——不足即判歧义。
    min_margin: float = 0.08
    #: 外观不可用时在总分门槛上加的惩罚（几何证据必须更强才敢接）。
    appearance_penalty: float = 0.10
    #: 余量回升多少即判「换上了新袋」（面积比绝对增量）。
    max_ratio_regain: float = 0.12


@dataclass(frozen=True)
class LostTrackView:
    """一条 lost 轨迹的**只读快照**（不复制记忆、不持有第二份状态）。

    三个构造入口：
    * :meth:`from_handoff` —— 从 :class:`~.occlusion.ReacquireHandoff` + 槽位信息构造；
    * :meth:`from_tracker` —— 直接从 :class:`~.occlusion.OcclusionTracker` 取（自动判是否已交出）；
    * 直接构造 —— 单测里用程序拼的框与参考余量。
    """

    obj_id: str
    slot_index: int | None
    slot_count: int
    handoff: ReacquireHandoff | None
    descriptor: BagDescriptor | None = None

    @property
    def lost(self) -> bool:
        """是否已被交出。**没交出的轨迹不参与再获取**（避免与遮挡恢复两套逻辑打架）。"""
        return self.handoff is not None

    @property
    def expected_centre_x(self) -> float | None:
        """该袋位的横向中心（占画面宽度比例）——挂点按袋序竖分画面。"""
        if self.slot_index is None or self.slot_count < 1:
            return None
        return (self.slot_index + 0.5) / self.slot_count

    @property
    def reference_entry(self) -> MemoryEntry | None:
        """把交出单复原成一条「最后可见那一帧」的记忆条目。

        复用 :class:`~.memory_bank.MemoryEntry` 而不是另写一套几何比对，口径因此与
        遮挡恢复路径完全一致（袋底 / 质心 / 横向宽度三项，对液面下降不敏感）。
        """
        if self.handoff is None:
            return None
        bbox = self.handoff.bbox
        if bbox is None:
            return None
        centroid = self.handoff.centroid
        if centroid is None:
            centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        ratio = self.handoff.reference_area_ratio or 0.0
        return MemoryEntry(
            frame_index=self.handoff.last_seen_frame,
            bbox=bbox,
            centroid=centroid,
            area=int(round(ratio * self.handoff.cavity_area_px)),
            ratio=ratio,
            mask=None,
        )

    @classmethod
    def from_handoff(
        cls,
        handoff: ReacquireHandoff,
        *,
        slot_index: int | None = None,
        slot_count: int = 1,
        descriptor: BagDescriptor | None = None,
    ) -> "LostTrackView":
        return cls(
            obj_id=handoff.obj_id,
            slot_index=slot_index,
            slot_count=max(1, slot_count),
            handoff=handoff,
            descriptor=descriptor,
        )

    @classmethod
    def from_tracker(
        cls,
        obj_id: str,
        tracker: Any,
        *,
        slot_index: int | None = None,
        slot_count: int = 1,
        descriptor: BagDescriptor | None = None,
    ) -> "LostTrackView":
        """从 :class:`~.occlusion.OcclusionTracker` 取快照。

        ``tracker.handoff is None`` 时 ``lost`` 为 ``False``——调用方据此知道这条轨迹
        还走正常路径，本轮不该碰它。
        """
        handoff = getattr(tracker, "handoff", None)
        return cls(
            obj_id=obj_id,
            slot_index=slot_index,
            slot_count=max(1, slot_count),
            handoff=handoff,
            descriptor=descriptor,
        )


@dataclass(frozen=True, eq=False)
class ReacquireCandidate:
    """一条候选与某条 lost 轨迹的配对结果（含它凭什么占这个分数）。"""

    candidate_id: str
    obj_id: str
    bbox: tuple[int, int, int, int]
    area: int
    centroid: tuple[float, float] | None
    area_ratio: float
    stability_score: float
    memory_score: float
    slot_score: float | None
    appearance_score: float | None
    total: float
    #: 原图坐标系下的掩膜。接回曲线时由 :meth:`as_prediction` 还原成一次预测，
    #: 因此**不要求调用方自己再去找那张掩膜**。
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool), repr=False)
    roi: tuple[int, int, int, int] | None = None

    def as_prediction(self) -> MaskPrediction:
        """还原成一次预测，供 :meth:`~.occlusion.OcclusionTracker.observe` 吸收。

        ``score`` 传稳定分而不是形状规整度：两者同源（都来自几何核的一帧输出），
        但稳定分额外回答了「这块区域离阈值边缘有多远」，接回时更该看它。
        """
        return MaskPrediction(mask=self.mask, score=float(self.stability_score))


@dataclass(frozen=True)
class ReacquireVerdict:
    """一条 lost 轨迹的再获取判定。"""

    obj_id: str
    frame_index: int
    state: str
    reason: str
    confidence: float | None = None
    runner_up: float | None = None
    candidate: ReacquireCandidate | None = None
    #: 中文说明，直接给护士端与审计看。
    detail: str = ""
    #: 参与本轮打分的候选数（0 = ROI 里什么都没有）。
    candidate_count: int = 0

    @property
    def reacquired(self) -> bool:
        return self.state == REACQUIRED

    def to_dict(self) -> dict[str, Any]:
        return {
            "obj_id": self.obj_id,
            "frame_index": self.frame_index,
            "state": self.state,
            "reason": self.reason,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "runner_up": None if self.runner_up is None else round(self.runner_up, 4),
            "candidate_count": self.candidate_count,
            "candidate": None if self.candidate is None else {
                "candidate_id": self.candidate.candidate_id,
                "bbox": list(self.candidate.bbox),
                "area": self.candidate.area,
                "area_ratio": self.candidate.area_ratio,
                "stability_score": round(self.candidate.stability_score, 4),
                "memory_score": round(self.candidate.memory_score, 4),
                "slot_score": (
                    None if self.candidate.slot_score is None else round(self.candidate.slot_score, 4)
                ),
                "appearance_score": (
                    None
                    if self.candidate.appearance_score is None
                    else round(self.candidate.appearance_score, 4)
                ),
            },
            "detail": self.detail,
        }


@dataclass
class ReacquireBatch:
    """一次批量再获取的产出。"""

    frame_index: int
    roi: tuple[int, int, int, int] | None = None
    verdicts: list[ReacquireVerdict] = field(default_factory=list)
    scanned: int = 0
    #: 本轮被 ``batch_scan_limit`` 挡在外面的轨迹数（下一轮再试，不静默丢弃）。
    deferred: int = 0

    def by_obj(self) -> dict[str, ReacquireVerdict]:
        return {verdict.obj_id: verdict for verdict in self.verdicts}

    @property
    def reacquired(self) -> list[ReacquireVerdict]:
        return [verdict for verdict in self.verdicts if verdict.reacquired]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "roi": None if self.roi is None else list(self.roi),
            "scanned": self.scanned,
            "deferred": self.deferred,
            "verdicts": [verdict.to_dict() for verdict in self.verdicts],
        }


# ── ROI ────────────────────────────────────────────────────────────────────


def roi_for_tracks(
    views: Sequence[LostTrackView], shape: tuple[int, int], params: ReacquireParams
) -> tuple[int, int, int, int] | None:
    """把若干条 lost 轨迹的丢失位置并成**一块** ROI（一次枚举服务多条轨迹）。

    面积有上限：外扩比例按 0.7 逐档收，收到下限仍超面积就把区域裁成以画面中心为心的
    同尺寸窗口——「宁可少看一点」也不让单帧耗时随丢失条数膨胀。
    """
    boxes = [view.handoff.bbox for view in views if view.lost and view.handoff and view.handoff.bbox]
    if not boxes:
        return None
    height, width = shape
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)

    ratio = params.roi_pad_ratio
    while True:
        pad_x = max(params.roi_min_pad_px, int(round((x1 - x0) * ratio)))
        pad_y = max(params.roi_min_pad_px, int(round((y1 - y0) * ratio)))
        rx0 = max(0, x0 - pad_x)
        ry0 = max(0, y0 - pad_y)
        rx1 = min(width, x1 + pad_x)
        ry1 = min(height, y1 + pad_y)
        area = (rx1 - rx0) * (ry1 - ry0)
        if area <= params.roi_max_area_px or ratio <= 0.0:
            break
        ratio *= 0.7
    if area > params.roi_max_area_px:
        span = int(np.sqrt(params.roi_max_area_px))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        rx0 = max(0, min(width - span, cx - span // 2))
        ry0 = max(0, min(height - span, cy - span // 2))
        rx1 = min(width, rx0 + span)
        ry1 = min(height, ry0 + span)
        logger.info(
            "丢失区域并集 %d px² 超过上限，裁成中心 %dx%d 窗口", area, rx1 - rx0, ry1 - ry0
        )
    return (rx0, ry0, rx1, ry1)


# ── 打分 ───────────────────────────────────────────────────────────────────


def _pair_score(
    view: LostTrackView,
    candidate: MaskCandidate,
    prediction: MaskPrediction,
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    params: ReacquireParams,
    visibility: VisibilityParams,
) -> ReacquireCandidate | None:
    """给一对（lost 轨迹, 候选）打分并过硬门限；任一门限不过返回 ``None``。

    返回 ``None`` 时**不带原因**——原因由调用方按「本轨迹一个候选都没活下来」的层级
    汇总后给出，避免同一个理由在每条候选上刷屏。
    """
    reference = view.reference_entry
    if reference is None or view.handoff is None:
        return None
    # 交出单里没有参考余量 → 这条轨迹从未有过一次可信读数，无从判断「面前这块是不是
    # 原来那只袋子」。宁可交给护士认领，也不靠位置对得上就接回（纪律：不猜）。
    if view.handoff.reference_area_ratio is None:
        return None
    cavity = max(1, int(view.handoff.cavity_area_px))
    area_ratio = int(np.count_nonzero(candidate.mask)) / cavity

    # 纪律 3：换液后新袋更满，余量会不降反升。接回就等于把两袋的曲线接成一条。
    regain = area_ratio - float(view.handoff.reference_area_ratio)
    if regain > params.max_ratio_regain:
        return None

    memory_score = reference.agreement(prediction, visibility)
    if memory_score < params.min_memory_agreement:
        return None

    slot_score: float | None = None
    expected_centre = view.expected_centre_x
    if expected_centre is not None:
        centre_x = (candidate.bbox[0] + candidate.bbox[2]) / 2.0 / max(1, image.shape[1])
        slot_score = max(
            0.0, 1.0 - abs(centre_x - expected_centre) / max(1e-6, params.slot_centre_tolerance)
        )
        if slot_score < params.min_slot_score:
            return None

    descriptor = describe_region(
        image, candidate.mask, bag_seq=view.obj_id, slot_index=view.slot_index or 0
    )
    appearance_score: float | None = None
    if view.descriptor is not None and view.descriptor.appearance_available:
        _total, _geometry, appearance = score_pair(descriptor, view.descriptor)
        appearance_score = appearance
        # 外观可用却明显不像——换了一只药袋。位置对得上不代表是同一袋，拒。
        if appearance_score is not None and appearance_score < params.min_appearance:
            return None

    # 只把**这一路证据真的存在**的通道计入分子分母。槽位序号未知（slot_score=None）时
    # 若还把它的权重算进分母，总分会凭空少掉四分之一，证据齐备的轨迹也会被门槛挡死——
    # 这类「因为缺一条证据而系统性低估」比缺证据本身更危险。
    channels = [(params.memory_weight, memory_score)]
    if slot_score is not None:
        channels.append((params.slot_weight, slot_score))
    if appearance_score is not None:
        channels.append((params.appearance_weight, appearance_score))
    weight_sum = sum(weight for weight, _ in channels)
    total = sum(weight * value for weight, value in channels) / weight_sum

    return ReacquireCandidate(
        candidate_id=candidate.candidate_id,
        obj_id=view.obj_id,
        bbox=candidate.bbox,
        area=candidate.area,
        centroid=mask_centroid(candidate.mask),
        area_ratio=round(area_ratio, 5),
        stability_score=candidate.stability_score,
        memory_score=round(memory_score, 4),
        slot_score=None if slot_score is None else round(slot_score, 4),
        appearance_score=None if appearance_score is None else round(appearance_score, 4),
        total=round(total, 4),
        mask=candidate.mask,
        roi=roi,
    )


# ── 主流程 ─────────────────────────────────────────────────────────────────


def reacquire_tracks(
    views: Sequence[LostTrackView],
    image: np.ndarray,
    predictor: PerPointMaskPredictor,
    *,
    frame_index: int,
    params: ReacquireParams | None = None,
    visibility: VisibilityParams | None = None,
) -> ReacquireBatch:
    """在 ``image`` 这一帧上，把所有 lost 轨迹再获取一遍。

    流程：

    1. 只取**已交出**（``handoff is not None``）的轨迹；未交出的判
       ``not_lost_no_reacquire`` 交回调用方，本函数不碰它（纪律：不与遮挡恢复打架）；
    2. 超出 ``batch_scan_limit`` 的轨迹记 ``batch_limit_exceeded``，下一帧再试；
    3. 全部丢失位置并成**一块** ROI，只跑**一次**枚举（复用 :mod:`.enumerate`，
       传 ``region`` 限制在这块区域内）；
    4. 每条轨迹按「记忆几何 + 槽位先验 + 外观」三条证据给每个候选打分，硬门限淘汰；
    5. 全局按分数贪心分配：**一条候选只接一条轨迹，一条轨迹只接一条候选**；
    6. 接回的候选还要求与次优拉开 ``min_margin``，否则判歧义不猜。

    返回的 :class:`ReacquireBatch` 只给判定与证据；**是否把曲线接回去由调用方决定**
    （:meth:`~.occlusion.OcclusionTracker.reacquire_from` 是唯一入口）。

    ``frame_index`` 是本次再获取发生在哪一帧——它写进每条判定，便于审计把「第几帧接回的」
    与可见度序列对上。
    """
    params = params or ReacquireParams()
    visibility = visibility or VisibilityParams()

    eligible: list[LostTrackView] = []
    verdicts: dict[str, ReacquireVerdict] = {}
    for view in views:
        if not view.lost:
            verdicts[view.obj_id] = ReacquireVerdict(
                view.obj_id,
                frame_index,
                UNBOUND,
                REASON_NOT_LOST,
                detail="这条轨迹还没被交出，仍走正常遮挡恢复路径，再获取不介入",
            )
            continue
        eligible.append(view)

    eligible.sort(key=lambda item: (item.handoff.lost_frame if item.handoff else 0, item.obj_id))
    scanned = eligible[: params.batch_scan_limit]
    deferred = eligible[params.batch_scan_limit :]
    for view in deferred:
        verdicts[view.obj_id] = ReacquireVerdict(
            view.obj_id,
            frame_index,
            STILL_LOST,
            REASON_BATCH_LIMIT,
            detail=(
                f"本轮 lost 轨迹 {len(eligible)} 条超过一次服务上限 "
                f"{params.batch_scan_limit} 条，留到后续帧再试"
            ),
        )

    batch = ReacquireBatch(
        frame_index=frame_index, verdicts=[], scanned=len(scanned), deferred=len(deferred)
    )
    if not scanned:
        batch.verdicts = [verdicts[key] for key in sorted(verdicts)]
        return batch

    shape = image.shape[:2]
    roi = roi_for_tracks(scanned, (shape[0], shape[1]), params)
    batch.frame_index = frame_index
    batch.roi = roi
    if roi is None:
        for view in scanned:
            verdicts[view.obj_id] = ReacquireVerdict(
                view.obj_id,
                frame_index,
                STILL_LOST,
                REASON_ROI_EMPTY,
                detail="交出单里没有可用的丢失位置，无从在附近重扫",
            )
        batch.verdicts = [verdicts[key] for key in sorted(verdicts)]
        return batch

    enumeration_params = EnumerationParams(
        points_per_side=params.points_per_side,
        refill_points_scale=params.refill_points_scale,
        stability_score_thresh=params.stability_score_thresh,
    )
    result = enumerate_candidates(
        image, predictor, params=enumeration_params, region=roi
    )
    if not result.candidates:
        for view in scanned:
            verdicts[view.obj_id] = ReacquireVerdict(
                view.obj_id,
                frame_index,
                STILL_LOST,
                REASON_ROI_EMPTY,
                detail=f"在丢失位置附近（{roi}）没扫出像输液袋的区域，保持丢失待人工认领",
            )
        batch.verdicts = [verdicts[key] for key in sorted(verdicts)]
        return batch

    # 打分：每条轨迹保留自己「过了硬门限」的候选，按总分倒序
    scored: dict[str, list[ReacquireCandidate]] = {}
    for view in scanned:
        ranked: list[ReacquireCandidate] = []
        for candidate in result.candidates:
            prediction = MaskPrediction(mask=candidate.mask, score=candidate.stability_score)
            pair = _pair_score(view, candidate, prediction, image, roi, params, visibility)
            if pair is not None:
                ranked.append(pair)
        ranked.sort(key=lambda item: (-item.total, item.candidate_id))
        scored[view.obj_id] = ranked

    # 全局贪心分配：一条候选只接一条轨迹
    assignments: dict[str, ReacquireCandidate] = {}
    used: set[str] = set()
    pairs = sorted(
        (
            (pair.total, view.obj_id, pair)
            for view in scanned
            for pair in scored[view.obj_id]
        ),
        key=lambda item: (-item[0], item[1], item[2].candidate_id),
    )
    for total, obj_id, pair in pairs:
        if obj_id in assignments or pair.candidate_id in used:
            continue
        assignments[obj_id] = pair
        used.add(pair.candidate_id)

    for view in scanned:
        obj_id = view.obj_id
        ranked = scored[obj_id]
        if not ranked:
            verdicts[obj_id] = ReacquireVerdict(
                obj_id,
                frame_index,
                STILL_LOST,
                _failure_reason(view, params),
                candidate_count=len(result.candidates),
                detail=_failure_detail(view, params, len(result.candidates)),
            )
            continue
        best = assignments.get(obj_id)
        if best is None:
            verdicts[obj_id] = ReacquireVerdict(
                obj_id,
                frame_index,
                STILL_LOST,
                REASON_TAKEN,
                confidence=ranked[0].total,
                candidate_count=len(result.candidates),
                detail=(
                    f"附近扫到的候选已被另一条丢失轨迹先认走（{ranked[0].candidate_id}），"
                    "本轨迹保持丢失待人工认领"
                ),
            )
            continue
        remaining = [item for item in ranked if item.candidate_id != best.candidate_id]
        runner_up = remaining[0].total if remaining else None
        gate = params.min_score + (
            0.0 if best.appearance_score is not None else params.appearance_penalty
        )
        if best.total < gate:
            verdicts[obj_id] = ReacquireVerdict(
                obj_id,
                frame_index,
                UNBOUND,
                REASON_BELOW_SCORE,
                confidence=best.total,
                runner_up=runner_up,
                candidate=best,
                candidate_count=len(result.candidates),
                detail=(
                    f"最佳候选总分 {best.total:.2f} 未达门槛 {gate:.2f}"
                    + ("（外观不可用，已加罚）" if best.appearance_score is None else "")
                    + "，不自动接回"
                ),
            )
            continue
        if runner_up is not None and (best.total - runner_up) < params.min_margin:
            verdicts[obj_id] = ReacquireVerdict(
                obj_id,
                frame_index,
                UNBOUND,
                REASON_AMBIGUOUS,
                confidence=best.total,
                runner_up=runner_up,
                candidate=best,
                candidate_count=len(result.candidates),
                detail=(
                    f"最佳候选 {best.candidate_id}（{best.total:.2f}）与次优 "
                    f"{remaining[0].candidate_id}（{runner_up:.2f}）分差不足 "
                    f"{params.min_margin:.2f}，认不准，不默认取最高分"
                ),
            )
            continue
        verdicts[obj_id] = ReacquireVerdict(
            obj_id,
            frame_index,
            REACQUIRED,
            REASON_OK,
            confidence=best.total,
            runner_up=runner_up,
            candidate=best,
            candidate_count=len(result.candidates),
            detail=(
                f"在丢失位置附近找回同一袋（候选 {best.candidate_id}，"
                f"余量 {best.area_ratio:.3f}，总分 {best.total:.2f}），曲线接回原 bag_seq"
            ),
        )

    batch.verdicts = [verdicts[key] for key in sorted(verdicts)]
    logger.info(
        "再获取 frame=%s ROI=%s 候选=%d 接回=%d 保持丢失=%d",
        frame_index,
        roi,
        len(result.candidates),
        len(batch.reacquired),
        sum(1 for v in batch.verdicts if v.state == STILL_LOST),
    )
    return batch


def _failure_reason(view: LostTrackView, params: ReacquireParams) -> str:
    """所有候选都被硬门限淘汰时，给出**最具诊断价值**的那一条原因。

    顺序即优先级：先说「是不是换袋了」（这条最需要人工介入），再说身份对不上。
    """
    handoff = view.handoff
    if handoff is None:
        return REASON_NOT_LOST
    if handoff.reference_area_ratio is None:
        return REASON_NO_REFERENCE
    return REASON_BELOW_SCORE


def _failure_detail(view: LostTrackView, params: ReacquireParams, candidate_count: int) -> str:
    handoff = view.handoff
    if handoff is not None and handoff.reference_area_ratio is None:
        return "这条轨迹从未成功看见过，没有可比的参考余量，需人工认领"
    return (
        f"附近扫到 {candidate_count} 个候选，但没有一个同时过得了记忆几何 / 槽位先验 / "
        f"外观三条门限，保持丢失待人工认领"
    )