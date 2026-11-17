"""帧间关联与轨迹 ID 管理：把逐帧掩膜接成一条条 ID 稳定的轨迹。

**与源主路径的对应关系**（只借调用链与语义，不搬代码、不跑源权重）：

| 本模块 | 源主路径 | 代码证据 |
|--------|----------|----------|
| :meth:`TrackManager.seed` | ``predictor.add_new_mask(..., obj_id=...)``：首帧提示建立对象 | ``vos_inference.py:205-210``；``benchmark.py:63-69`` |
| :meth:`TrackManager.step` | ``for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(...)`` | ``vos_inference.py:224-231`` |
| 一条轨迹一个 ID、跨帧不变 | ``out_obj_ids`` 在整段视频里恒定，掩膜逐帧变而身份不变 | ``vos_inference.py:227-231`` |
| **不重新分割、不另起推理路径** | 源循环里后续帧只调传播，不调分割器 | ``vos_inference.py:224-226`` |

本模块**不做分割**。它只吃已经分好的每帧掩膜（:mod:`.propagate` 的产出）做关联、
ID 分配与终止判断；分割在哪里做、给什么提示，全由 :mod:`.propagate` 决定。这里不 import
任何预测器，也没有任何推理入口——测试 ``test_module_opens_no_inference_path`` 把这条钉住。

**判据取舍（本模块的核心设计）**：关联打分用 IoU、质心、袋底三个量，但**权重刻意偏向
几何**（IoU 0.25 / 质心 0.40 / 袋底 0.35），并且几何不通过时 IoU 一票否决。两个原因：

1. **液面下降会让 IoU 天然衰减**——袋越输越矮，掩膜越小，越接近换液时越接不回来；
2. **两袋贴住时 IoU 会交叉**——粘成一块的掩膜对两条轨迹的 IoU 几乎相同，此时只有
   质心与**袋底纵坐标**（袋底固定、液面上升，是最稳的锚）能分出谁是谁。

**四条纪律**（与 :mod:`.propagate` / :mod:`.occlusion` 一致）：

1. **恒定算力**：每条轨迹只留**最后一份**掩膜与几何量，不攒掩膜序列；600 帧与 1200 帧
   的峰值内存同量级。
2. **不静默丢帧**：没关联上的帧在轨迹上留 ``None`` 占位，序列长度不变，不插值。
3. **不静默硬接**：抖动超阈值时降置信并打 ``inconsistent`` 标记，**保留原值**——
   既不把假尖峰抹平，也不因此把这一帧拒之门外。
4. **不猜**：同一帧里两个候选得分差小于 ``ambiguity_margin`` 即判歧义，宁可让轨迹这一帧
   落空（进下一帧重试或交人工认领），也不按「取最高分」硬接。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .postprocess import component_features, label_components
from .predictor import mask_iou

__all__ = [
    "MATCHED",
    "GEOMETRY_VETO",
    "BELOW_SCORE",
    "AMBIGUOUS",
    "MISSED",
    "TERMINATED_BY_WINDOW",
    "TERMINATED_BY_REQUEST",
    "TrackingParams",
    "Detection",
    "Association",
    "FrameAssociation",
    "Track",
    "TrackManager",
    "association",
    "detections_from_mask",
    "track_masks_sequence",
]

logger = logging.getLogger("infusion_monitor.core.tracking")

#: 关联成功。
MATCHED = "matched"
#: 几何判据不通过（质心/袋底超容差）——IoU 再高也一票否决。
GEOMETRY_VETO = "geometry_veto"
#: 几何通过但总分低于 ``min_match_score``。
BELOW_SCORE = "below_score"
#: 两个候选得分接近，本帧不猜。
AMBIGUOUS = "ambiguous"
#: 本帧没有可关联的候选（终止后继续未命中也记此原因）。
MISSED = "missed"
#: 连续未命中超过窗口，轨迹终止。
TERMINATED_BY_WINDOW = "missed_window_exceeded"
#: 调用方显式退役（换袋、护士认领到新袋等）。
TERMINATED_BY_REQUEST = "retired_by_caller"


@dataclass(frozen=True)
class TrackingParams:
    """关联与终止的门限。取值理由写在字段旁，全部可由病区实测调整。"""

    #: IoU 达到该值即算「完全重叠」，据此把 IoU 归一化到 0–1。
    iou_threshold: float = 0.30
    #: 三项打分权重。几何权重高于 IoU，理由见模块 docstring。
    iou_weight: float = 0.25
    centroid_weight: float = 0.40
    bottom_weight: float = 0.35
    #: 质心位移容差（像素）。超出即不是原来那袋。
    centroid_tolerance_px: float = 60.0
    #: 袋底纵坐标容差（像素）。袋底固定、液面上升，是最稳的锚。
    bottom_tolerance_px: float = 40.0
    #: 关联成功的总分门限。
    min_match_score: float = 0.45
    #: 候选得分差小于它即判歧义，不猜。0.08 ≈ 质心差 5 px 量级。
    ambiguity_margin: float = 0.08
    #: 连续未命中多少帧后终止轨迹。0.5 s/帧 × 8 = 4 s，与遮挡丢失窗口同量级。
    max_missed_frames: int = 8
    #: 面积比跨帧抖动容差。超过即标 ``inconsistent`` 并降置信，**但保留原值**。
    jitter_ratio: float = 0.05
    #: 一致性评分的 EMA 系数（越小越平滑、越迟钝）。
    consistency_alpha: float = 0.35
    #: 出过 ``inconsistent`` 的轨迹置信上限——不足以驱动换液点名，要人工看一眼。
    inconsistent_confidence_cap: float = 0.60
    #: 单个连通域最小面积（px²），低于此值不进检测集合。
    min_area_px: int = 24
    #: 轨迹 ID 前缀。
    id_prefix: str = "T"


@dataclass(frozen=True)
class Detection:
    """一帧里的一个观测：连通域的几何量 + 面积比。

    ``mask`` 可选。带掩膜时 IoU 参与打分；不带时只用几何（权重按可用项重新归一化）。
    检测对象**不持有跨帧状态**——本模块不攒检测序列（恒定算力）。
    """

    detection_id: str
    frame_index: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area_px: int
    ratio: float
    mask: np.ndarray | None = None

    @property
    def bottom(self) -> int:
        """袋底纵坐标——本产品的「最稳的锚」。"""
        return self.bbox[3]


def detections_from_mask(
    mask: np.ndarray,
    frame_index: int,
    *,
    cavity_area_px: int = 0,
    params: TrackingParams | None = None,
    detection_prefix: str = "D",
) -> list[Detection]:
    """把一帧掩膜切成若干检测（复用 :mod:`.postprocess` 的连通域特征，不重复实现）。

    ``cavity_area_px > 0`` 时按袋腔算出面积比，否则面积比就是归一化面积（仅供调试）。
    面积小到看不出是袋子的连通域直接丢掉——宁可漏，不可拿碎斑起一条轨迹。
    """
    params = params or TrackingParams()
    labelled = label_components(mask)
    detections: list[Detection] = []
    for label in range(1, labelled.count):
        component = labelled.label_mask(label)
        feature = component_features(labelled, label, mask=component)
        if feature.area_px < params.min_area_px:
            continue
        ratio = feature.area_px / cavity_area_px if cavity_area_px > 0 else 1.0
        detections.append(
            Detection(
                detection_id=f"{detection_prefix}{label:02d}",
                frame_index=frame_index,
                bbox=feature.bbox,
                centroid=feature.centroid,
                area_px=feature.area_px,
                ratio=min(1.0, ratio),
                mask=component,
            )
        )
    detections.sort(key=lambda d: d.bbox[0])
    return detections


@dataclass(frozen=True)
class Association:
    """一次「轨迹 × 检测」配对的判定结果，含判据拆解（便于复核与调试）。"""

    track_id: str
    detection_id: str
    score: float
    matched: bool
    reason: str
    iou: float | None = None
    centroid_distance: float = 0.0
    bottom_distance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "detection_id": self.detection_id,
            "score": round(self.score, 4),
            "matched": self.matched,
            "reason": self.reason,
            "iou": None if self.iou is None else round(self.iou, 4),
            "centroid_distance": round(self.centroid_distance, 2),
            "bottom_distance": round(self.bottom_distance, 2),
        }


@dataclass(frozen=True)
class FrameAssociation:
    """一帧的关联产出（成功配对 + 落空的轨迹 + 没被认领的检测）。"""

    frame_index: int
    #: 本帧**成功**的配对（已按分数降序，贪心一对一的结果）。
    matches: tuple[Association, ...] = ()
    #: 本帧落空的轨迹（无候选、被更高分抢走、或歧义不猜）。
    missed_track_ids: tuple[str, ...] = ()
    #: 因歧义未决的轨迹（护士端据此提示需人工认领）。
    ambiguous_track_ids: tuple[str, ...] = ()
    #: 本帧没被任何轨迹认领的检测（新出现的袋位或碎块）。
    unmatched_detection_ids: tuple[str, ...] = ()

    @property
    def is_ambiguous(self) -> bool:
        """本帧是否有轨迹因歧义未决。"""
        return bool(self.ambiguous_track_ids)

    @property
    def match_count(self) -> int:
        return len(self.matches)

    def detection_for(self, track_id: str) -> str | None:
        """查某条轨迹本帧接到了哪个检测；本帧落空则返回 ``None``。"""
        for match in self.matches:
            if match.track_id == track_id:
                return match.detection_id
        return None


@dataclass
class Track:
    """一条被持续关联的轨迹（一条 = 一个稳定的轨迹 ID）。"""

    track_id: str
    first_frame: int
    last_seen_frame: int
    last_updated_frame: int
    cavity_area_px: int = 0
    hits: int = 1
    misses: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    centroid: tuple[float, float] = (0.0, 0.0)
    ratio: float = 0.0
    last_mask: np.ndarray | None = None
    #: 一致性评分（0–1）。抖动越小越接近 1。
    consistency: float = 1.0
    #: 抖动超阈值的帧号（保留原值，只打标记，不丢帧）。
    inconsistent_frames: list[int] = field(default_factory=list)
    #: 每帧面积比（未命中帧留 None 占位，序列长度恒等于帧数）。
    ratios: list[float | None] = field(default_factory=list)
    terminated: bool = False
    terminate_reason: str | None = None
    #: 出过 ``inconsistent`` 时的置信上限（由 :class:`TrackingParams` 下发）
    inconsistent_cap: float = 0.60

    @property
    def track_age(self) -> int:
        """轨迹年龄（帧）：自建立起经过的帧数，落空的帧也算在内。"""
        return self.last_updated_frame - self.first_frame + 1

    @property
    def bottom(self) -> int:
        return self.bbox[3]

    @property
    def is_active(self) -> bool:
        return not self.terminated

    @property
    def confidence(self) -> float:
        """置信：一致性评分；出过 ``inconsistent`` 则封顶。"""
        if self.inconsistent_frames:
            return min(self.consistency, self.inconsistent_cap)
        return self.consistency

    @property
    def is_inconsistent(self) -> bool:
        return bool(self.inconsistent_frames)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "first_frame": self.first_frame,
            "last_seen_frame": self.last_seen_frame,
            "last_updated_frame": self.last_updated_frame,
            "track_age": self.track_age,
            "hits": self.hits,
            "misses": self.misses,
            "consistency": round(self.consistency, 4),
            "confidence": round(self.confidence, 4),
            "inconsistent_frames": list(self.inconsistent_frames),
            "inconsistent": self.is_inconsistent,
            "terminated": self.terminated,
            "terminate_reason": self.terminate_reason,
        }


def association(
    track: Track, detection: Detection, params: TrackingParams | None = None
) -> Association:
    """判一对「轨迹 × 检测」能不能接上，返回打分与判据拆解。

    **几何不通过即否决**：质心或袋底超出容差时，即便 IoU = 1.0（两袋粘连时对谁都是 1.0）
    也不接——这正是「不猜」的落点。
    """
    params = params or TrackingParams()
    dx = detection.centroid[0] - track.centroid[0]
    dy = detection.centroid[1] - track.centroid[1]
    centroid_distance = float(np.hypot(dx, dy))
    bottom_distance = float(abs(detection.bottom - track.bottom))

    iou: float | None = None
    if track.last_mask is not None and detection.mask is not None:
        iou = mask_iou(track.last_mask, detection.mask)

    centroid_score = max(0.0, 1.0 - centroid_distance / max(1e-6, params.centroid_tolerance_px))
    bottom_score = max(0.0, 1.0 - bottom_distance / max(1e-6, params.bottom_tolerance_px))

    terms: list[tuple[float, float]] = [
        (params.centroid_weight, centroid_score),
        (params.bottom_weight, bottom_score),
    ]
    if iou is not None:
        terms.append((params.iou_weight, min(1.0, iou / max(1e-6, params.iou_threshold))))
    total_weight = sum(weight for weight, _ in terms) or 1.0
    score = sum(weight * value for weight, value in terms) / total_weight

    geometry_ok = (
        centroid_distance <= params.centroid_tolerance_px
        and bottom_distance <= params.bottom_tolerance_px
    )
    if not geometry_ok:
        reason = GEOMETRY_VETO
        matched = False
    elif score < params.min_match_score:
        reason = BELOW_SCORE
        matched = False
    else:
        reason = MATCHED
        matched = True
    return Association(
        track_id=track.track_id,
        detection_id=detection.detection_id,
        score=score,
        matched=matched,
        reason=reason,
        iou=iou,
        centroid_distance=centroid_distance,
        bottom_distance=bottom_distance,
    )


class TrackManager:
    """轨迹管理器：首帧起 ID，之后逐帧只做关联。

    每帧的处理顺序固定为：**算全部候选 → 去掉歧义 → 按分数贪心一对一配对 → 更新轨迹**。
    贪心按 ``(-score, track_id, detection_id)`` 排序，因此同一输入必得同一结果，
    不依赖字典遍历顺序——这是「可复现」的前提。
    """

    def __init__(
        self,
        *,
        frame_count: int = 0,
        cavity_area_px: int = 0,
        params: TrackingParams | None = None,
    ) -> None:
        self.params = params or TrackingParams()
        self.cavity_area_px = int(cavity_area_px)
        self.frame_count = int(frame_count)
        self.tracks: dict[str, Track] = {}
        self._issued = 0

    # ── 建立 ──────────────────────────────────────────────────────────
    def seed(self, frame_index: int, detections: Sequence[Detection]) -> list[Track]:
        """首帧建立轨迹并分配 ID。

        ID 按质心 ``(x, y)`` 排序后依次分配——画面左边的袋子拿 ``T01``，与掩膜标签顺序、
        连通域编号都无关，因此同一场景重跑得到的 ID 完全一致。
        """
        if not detections:
            return []
        ordered = sorted(detections, key=lambda d: (round(d.centroid[0], 3), round(d.centroid[1], 3)))
        created: list[Track] = []
        for detection in ordered:
            self._issued += 1
            track = Track(
                track_id=f"{self.params.id_prefix}{self._issued:02d}",
                first_frame=frame_index,
                last_seen_frame=frame_index,
                last_updated_frame=frame_index,
                cavity_area_px=self.cavity_area_px,
                bbox=detection.bbox,
                centroid=detection.centroid,
                ratio=detection.ratio,
                last_mask=detection.mask,
                ratios=[detection.ratio],
                inconsistent_cap=self.params.inconsistent_confidence_cap,
            )
            self.tracks[track.track_id] = track
            created.append(track)
        return created

    # ── 关联 ──────────────────────────────────────────────────────────
    def step(
        self, frame_index: int, detections: Sequence[Detection]
    ) -> FrameAssociation:
        """把本帧检测关联到已有轨迹上。"""
        candidates: list[Association] = []
        for track in self.tracks.values():
            if not track.is_active:
                continue
            for detection in detections:
                verdict = association(track, detection, self.params)
                if verdict.matched:
                    candidates.append(verdict)

        by_track: dict[str, list[Association]] = {}
        for verdict in candidates:
            by_track.setdefault(verdict.track_id, []).append(verdict)

        # 歧义先摘掉：同一轨迹有两个分数接近的候选时，本帧不接，等下一帧或交人工认领
        ambiguous_tracks: set[str] = set()
        for track_id, verdicts in by_track.items():
            if len(verdicts) < 2:
                continue
            verdicts.sort(key=lambda v: (-v.score, v.detection_id))
            if verdicts[0].score - verdicts[1].score < self.params.ambiguity_margin:
                ambiguous_tracks.add(track_id)

        candidates = [v for v in candidates if v.track_id not in ambiguous_tracks]
        candidates.sort(key=lambda v: (-v.score, v.track_id, v.detection_id))

        matched_tracks: dict[str, Association] = {}
        taken_detections: set[str] = set()
        for verdict in candidates:
            if verdict.track_id in matched_tracks or verdict.detection_id in taken_detections:
                continue
            matched_tracks[verdict.track_id] = verdict
            taken_detections.add(verdict.detection_id)

        by_detection = {d.detection_id: d for d in detections}
        for track_id, verdict in matched_tracks.items():
            self._absorb(self.tracks[track_id], frame_index, by_detection[verdict.detection_id])

        missed: list[str] = sorted(ambiguous_tracks)
        for track in self.tracks.values():
            if not track.is_active or track.track_id in matched_tracks:
                continue
            gap = max(1, frame_index - track.last_updated_frame)
            self._record_miss(track, frame_index, gap)
            missed.append(track.track_id)

        frame_result = FrameAssociation(
            frame_index=frame_index,
            matches=tuple(matched_tracks.values()),
            missed_track_ids=tuple(sorted(set(missed))),
            ambiguous_track_ids=tuple(sorted(ambiguous_tracks)),
            unmatched_detection_ids=tuple(
                sorted(d.detection_id for d in detections if d.detection_id not in taken_detections)
            ),
        )
        if ambiguous_tracks:
            logger.info(
                "帧 %d 关联歧义：%s 本帧不猜，等下一帧或人工认领",
                frame_index,
                ",".join(sorted(ambiguous_tracks)),
            )
        return frame_result

    # ── 内部状态推进 ──────────────────────────────────────────────────
    def _absorb(self, track: Track, frame_index: int, detection: Detection) -> None:
        track.misses = 0
        track.hits += 1
        track.last_seen_frame = frame_index
        track.last_updated_frame = frame_index
        track.bbox = detection.bbox
        track.centroid = detection.centroid
        track.ratio = detection.ratio
        track.last_mask = detection.mask
        self._score_consistency(track, frame_index, detection.ratio)
        self._extend(track, frame_index, detection.ratio)

    def _record_miss(self, track: Track, frame_index: int, gap: int = 1) -> None:
        track.misses += max(1, gap)
        track.last_updated_frame = frame_index
        # 落空的帧留 None 占位：不插值、不沿用上一帧数值，序列长度不变
        for offset in range(gap):
            self._extend(track, frame_index - gap + 1 + offset, None)
        if track.misses > self.params.max_missed_frames:
            track.terminated = True
            track.terminate_reason = TERMINATED_BY_WINDOW
            logger.info(
                "轨迹 %s 连续未命中 %d 帧，终止（%s）",
                track.track_id,
                track.misses,
                TERMINATED_BY_WINDOW,
            )

    def _score_consistency(
        self, track: Track, frame_index: int, ratio: float
    ) -> None:
        previous = track.ratios[-1] if track.ratios else None
        if previous is None:
            track.consistency = 1.0
            return
        jump = abs(ratio - previous)
        alpha = self.params.consistency_alpha
        if jump <= self.params.jitter_ratio:
            track.consistency = (1.0 - alpha) * track.consistency + alpha * 1.0
            return
        # 抖动：降置信 + 打标记，但**原值照记**（不静默丢帧，也不硬接平滑值）
        penalty = min(1.0, (jump - self.params.jitter_ratio) / max(1e-6, self.params.jitter_ratio))
        track.consistency = (1.0 - alpha) * track.consistency + alpha * (1.0 - penalty)
        track.inconsistent_frames.append(frame_index)

    def _extend(self, track: Track, frame_index: int, ratio: float | None) -> None:
        if self.frame_count > 0:
            while len(track.ratios) <= frame_index:
                track.ratios.append(None)
            track.ratios[frame_index] = ratio
        else:
            track.ratios.append(ratio)

    # ── 终止与查询 ────────────────────────────────────────────────────
    def retire(self, track_id: str, reason: str = TERMINATED_BY_REQUEST) -> bool:
        """显式退役一条轨迹（换袋、护士把槽位认领到新袋）。返回是否确实改了状态。"""
        track = self.tracks.get(track_id)
        if track is None or track.terminated:
            return False
        track.terminated = True
        track.terminate_reason = reason
        return True

    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.is_active]

    def terminated_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.terminated]

    def finalize(self, frame_index: int) -> None:
        """序列结束：未终止的轨迹显式收尾（标注而非静默消失）。"""
        for track in self.tracks.values():
            track.last_updated_frame = max(track.last_updated_frame, frame_index)


def track_masks_sequence(
    masks_by_frame: Iterable[Mapping[str, np.ndarray]],
    *,
    cavity_area_px: int = 0,
    params: TrackingParams | None = None,
) -> TrackManager:
    """把 :mod:`.propagate` 逐帧产出的掩膜接成轨迹（医院侧入口）。

    ``masks_by_frame`` 每一项是「本帧各对象的掩膜」映射，键与值来自同一次传播——
    **本函数不调用任何预测器**：分割已在传播里做完，这里只做帧间关联。

    键里带 ``bag_seq`` 的（:mod:`.propagate` 的 ``obj_id`` 即袋序）在第一帧直接按袋序
    建轨迹；不带任何提示的裸序列则走「首帧枚举 → 质心排序分配 ID」。
    """
    manager = TrackManager(cavity_area_px=cavity_area_px, params=params)
    seeded = False
    last_frame = -1
    for frame_index, frame_masks in enumerate(masks_by_frame):
        last_frame = frame_index
        detections = [
            Detection(
                detection_id=f"F{frame_index:05d}-{key}",
                frame_index=frame_index,
                bbox=_bbox(mask),
                centroid=_centroid(mask),
                area_px=int(np.count_nonzero(mask)),
                ratio=(int(np.count_nonzero(mask)) / cavity_area_px if cavity_area_px else 1.0),
                mask=mask,
            )
            for key, mask in sorted(frame_masks.items())
            if np.any(mask)
        ]
        if not seeded:
            manager.seed(frame_index, detections)
            seeded = True
            continue
        manager.step(frame_index, detections)
    manager.finalize(last_frame)
    return manager


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean()))