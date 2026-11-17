"""记忆特征库：每条轨迹维护 top-k 记忆帧，按相似度加权参与当前帧证据。

**与源主路径的对应关系**（只借调用链与语义，不搬代码、不跑源权重）：

| 本模块 | 源主路径 | 代码证据 |
|--------|----------|----------|
| :class:`MemoryBank`——保存「上一帧/前几帧」的记忆特征 | ``MemoryAttentionLayer._forward_ca(tgt, memory, ...)``：``q`` 来自当前帧、``k/v`` 来自 ``memory``（上一帧的编码特征），输出因此被记忆条件化 | ``memory_attention.py:66-81`` |
| :meth:`MemoryBank.write`——记忆只在「真的看见了同一袋」时更新 | ``MemoryAttention.forward(..., memory=...)`` 每步换新记忆，不拿任意一帧往里塞 | ``memory_attention.py:119-161`` |
| :meth:`MemoryBank.match` / :meth:`fused_agreement`——按相似度加权汇总记忆当本帧证据 | cross-attention 的注意力权重（越相关的记忆贡献越大） | ``memory_attention.py:74-80`` |
| :attr:`MemoryEntry.drain_per_frame`——记忆里的下降趋势 | 记忆参与推理，因此「上一帧什么样」能约束「这一帧该是什么样」 | ``memory_attention.py:94`` |

**为什么不是「留一份上一帧」就够**（task 6 的留白）：只留最新一帧时，新鲜度与可靠性绑在
一起——被挡了一帧、护士手一晃，最新记忆本身就脏了，恢复判断只能整体放弃。这里把记忆
拆成 **top-k 条**（固定容量，与帧数无关），并把两件事分开：

* **新鲜度**由 ``decay`` 决定（越久远权重越低）；
* **可靠性**由 ``similarity`` 决定（与库里共识对不上的不参与加权）。

于是「最新一帧脏了，但前几帧还干净」时，恢复判断仍有依据；而**全库都脏**（场景切换、
拍到别的东西）时，加权结果自然塌下去 → 判遮挡 → 走丢失窗口交出轨迹。

**库里分两类条目，用途不同**（这是本模块最关键的一条约定）：

============  ==========================================  ==========================
条目          干什么                                       谁在用
============  ==========================================  ==========================
``trusted``   袋体确实完整可见，面积比可信                 身份判断（写入门禁、观测门禁、
（visible）  ——**投身份判断的票**                        恢复时的加权吻合度）
``untrusted`` 被压住一半，只看得见部分袋体——几何照常更新   位置提示（``latest``）：那一帧
（partial）   以维持跟踪，但**不投身份判断的票**          确实看得见袋子在那儿，只是读数偏低
============  ==========================================  ==========================

不这么分就会出真事故：被压住一半的帧把共识拉扁，遮一结束恢复正常的帧反而被判成
「拍到了别的东西」（本轮实测 sim 0.467 vs 门限 0.55）。用采信条目投票就没这个问题。

**它管不了什么（别指望它）**：几何记忆分不出「同高度、同样式的另一袋」。输液架上两袋
袋底与宽度一致时，仅靠位置相似度最高也有 0.6（宽度项 0.3 + 袋底项 0.3）——那是
**槽位分配**（task 18）与**外观描述子**（task 29）的活，本模块不越界去猜。

**四条纪律**（与 :mod:`.propagate`、:mod:`.occlusion` 同一套）：

1. **不引入第二套推理入口**：记忆库是 :class:`.occlusion.OcclusionTracker` **唯一**持有的
   状态（替换原来的单个 ``_memory`` 字段），下一帧的提示仍由
   :meth:`.propagate.ObjectTrack.follow_up_prompt` 那一处产出，库只提供内容。
2. **不引入第二份状态**：``latest`` 返回的就是库里那一条对象本身，不是副本。
3. **恒定算力**：容量固定 ``capacity``（默认 4）条，每帧最多 ``capacity`` 次比对；
   600 帧与 1200 帧的库大小完全相同。
4. **不猜**：写入前先比对共识，对不上**直接拒收**（不是收进来再打个低分）——
   错误记忆一旦进了库，就会把下一帧的提示带到别的地方去。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "GeometryTolerance",
    "MemoryParams",
    "MemoryEntry",
    "MemoryMatch",
    "MemoryQuery",
    "MemoryWrite",
    "MemoryBank",
    "REJECT_SCENE_SWITCH",
    "REJECT_FIRST_SIGHTING",
    "REJECT_EMPTY_BANK",
]

#: 观测与共识对不上 → 拒收（场景切换 / 拍到了别的东西）。
REJECT_SCENE_SWITCH = "scene_switch_rejected"
#: 库是空的，本来就没有可比的共识 → 首见必收。
REJECT_FIRST_SIGHTING = "first_sighting"
#: 库里一条不剩（例如刚被清空）→ 没有依据可比。
REJECT_EMPTY_BANK = "empty_bank"


@runtime_checkable
class GeometryTolerance(Protocol):
    """:meth:`MemoryEntry.agreement` 需要的两个容差（由 :class:`.occlusion.VisibilityParams` 提供）。"""

    centroid_tolerance_px: float
    bottom_tolerance_px: float


@dataclass(frozen=True)
class MemoryParams:
    """记忆库门限。取值理由写在字段旁，全部可由病区实测调整。"""

    #: top-k 容量。4 条 ≈ 2 秒的视觉记忆，够跨过一次「护士手经过」而不至于把很久以前的
    #: 液位当现状；再大对判断没有帮助，只是白占内存。
    capacity: int = 4
    #: **写入门禁**：与共识相似度低于此值就拒收（库宁缺一条也不收脏的）。
    #: 0.20 的来历：合成渲染实测相邻帧 0.92–0.98、袋底被压住 40% 的半遮挡帧 0.30
    #: （见 progress.txt 的 task 16 实测表），而「整袋跑到画面另一头」约 0.0–0.15。
    #: 门禁卡在「荒唐」与「反常」之间：宁可让一条脏记忆挤占一格，也不让它改写下一帧的提示。
    min_similarity: float = 0.20
    #: **观测门禁**（比写入门禁严）：读数与滴速趋势对得上、位置却与记忆差得远时
    #: 判「拍到了别的东西」。留 0.55 是因为合法帧实测最低 0.916，而真·场景切换
    #: （换到另一槽位、另一袋型）只有 0.0–0.15——0.55 落在两者中间，两头都留了余量。
    min_consistency: float = 0.55
    #: 每过一帧，权重乘一次——新鲜度衰减。
    decay_per_frame: float = 0.88
    #: 权重低于此值的记忆参与不了加权判断（等价于「太旧了不作数」）。
    min_entry_weight: float = 0.02


@dataclass
class MemoryEntry:
    """一条记忆：某一帧被确认看见时留下的几何、掩膜、面积比与置信。

    字段语义与 :class:`.occlusion.TrackMemory` 一致（``TrackMemory`` 是本类的别名），
    外加三项记忆库专用信息：:attr:`score`（该帧掩膜的形状规整度）、:attr:`trusted`
    （这一帧的面积比是否被采信——partial 帧为 ``False``）、:attr:`similarity`
    （写入时与库共识的相似度）。
    """

    frame_index: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    ratio: float
    mask: np.ndarray | None = None
    #: 每帧面积比下降量（EMA）。用于预测「这帧按滴速应该剩多少」。
    drain_per_frame: float = 0.0
    #: 这条轨迹被连续确认了多少帧（≥2 才开始用趋势判 partial，避免首帧抖动误判）。
    hits: int = 1
    #: 该帧掩膜的形状规整度（0–1），进加权。
    score: float = 0.0
    #: 本条的面积比是否被采信。partial 帧几何更新但面积比沿用上一条可信值。
    trusted: bool = True
    #: 写入时与库共识的相似度（首见为 1.0）。
    similarity: float = 1.0

    def predicted_ratio(self, gap: int = 1) -> float:
        """按记忆里的下降趋势外推 ``gap`` 帧后的预期面积比。"""
        return max(0.0, self.ratio - self.drain_per_frame * max(0, gap))

    def agreement(self, prediction: "PredictionLike", tolerance: GeometryTolerance) -> float:
        """当前掩膜与这条记忆有多吻合（0–1）。

        刻意**对液面下降不敏感**：液面降了会让外接框变矮、质心上移，若直接用 IoU 会
        越接近换液时越接不回来。因此改用三个与液面无关的量：质心位置、**袋底纵坐标**
        （袋体挂在架子上，袋底不随液面移动）与横向宽度。
        """
        from .predictor import mask_centroid  # 局部导入：predictor 不依赖本模块

        bbox = prediction.bbox
        if bbox is None:
            return 0.0
        px0, py0, px1, py1 = bbox
        mx0, my0, mx1, my1 = self.bbox
        centre = mask_centroid(prediction.mask)
        pcx, pcy = centre if centre is not None else ((px0 + px1) / 2.0, (py0 + py1) / 2.0)
        distance = math.hypot(pcx - self.centroid[0], pcy - self.centroid[1])
        positional = max(0.0, 1.0 - distance / max(1e-6, tolerance.centroid_tolerance_px))
        bottom = max(0.0, 1.0 - abs(py1 - my1) / max(1e-6, tolerance.bottom_tolerance_px))
        pred_width = max(1, px1 - px0)
        mem_width = max(1, mx1 - mx0)
        width = min(pred_width, mem_width) / max(pred_width, mem_width)
        return 0.4 * positional + 0.3 * bottom + 0.3 * width

    def weight_of(self, now_frame: int, decay_per_frame: float) -> float:
        """新鲜度 × 可靠性 × 形状规整度——库内一条记忆参与加权的实际权重。"""
        age = max(0, now_frame - self.frame_index)
        return decay_per_frame**age * self.similarity * (0.5 + 0.5 * max(0.0, min(1.0, self.score)))

    def mask_iou(self, other: np.ndarray | None) -> float | None:
        """与另一张掩膜的 IoU；任一方没有掩膜时返回 ``None``（不假装有依据）。"""
        if self.mask is None or other is None or self.mask.shape != other.shape:
            return None
        intersection = int(np.count_nonzero(self.mask & other))
        if intersection == 0:
            return 0.0
        union = int(np.count_nonzero(self.mask | other))
        return intersection / union if union else 0.0

    def age(self, now_frame: int) -> int:
        return max(0, now_frame - self.frame_index)


class PredictionLike(Protocol):
    """:meth:`MemoryEntry.agreement` 只需要 :class:`.predictor.MaskPrediction` 的 ``mask`` 与 ``bbox``。"""

    mask: np.ndarray
    score: float

    @property
    def bbox(self) -> tuple[int, int, int, int] | None: ...


@dataclass(frozen=True)
class MemoryMatch:
    """一条记忆对当前观测的匹配结果（含它凭什么占这么多权重）。"""

    entry: MemoryEntry
    similarity: float
    weight: float
    age: int
    mask_iou: float | None = None

    def to_dict(self) -> dict:
        return {
            "frame_index": self.entry.frame_index,
            "similarity": round(self.similarity, 4),
            "weight": round(self.weight, 4),
            "age": self.age,
            "mask_iou": None if self.mask_iou is None else round(self.mask_iou, 4),
            "trusted": self.entry.trusted,
        }


@dataclass(frozen=True)
class MemoryQuery:
    """记忆库对当前观测的一次汇总回答——「我是谁、我凭什么这么确信」。"""

    obj_id: str
    entry_count: int
    total_weight: float
    #: 加权共识几何（袋底/宽度锚定，与单条记忆相比更抗一次脏帧）。
    bbox: tuple[int, int, int, int] | None
    centroid: tuple[float, float] | None
    #: 最新一条**被采信**的面积比（不做加权平均：液面在降，平均值没有物理意义）。
    ratio: float | None
    #: 与观测的加权吻合度（0–1）；库为空或全部对不上时为 ``None``。
    agreement: float | None
    matches: tuple[MemoryMatch, ...] = ()
    rejected: tuple[MemoryEntry, ...] = ()
    reason: str = ""

    @property
    def supported(self) -> bool:
        """有没有记忆真的站在这一边（权重够、且能给出正吻合度）。"""
        return self.agreement is not None and self.agreement > 0.0 and self.total_weight > 0.0

    def to_dict(self) -> dict:
        return {
            "obj_id": self.obj_id,
            "entry_count": self.entry_count,
            "total_weight": round(self.total_weight, 4),
            "bbox": list(self.bbox) if self.bbox else None,
            "centroid": list(self.centroid) if self.centroid else None,
            "ratio": self.ratio,
            "agreement": None if self.agreement is None else round(self.agreement, 4),
            "matches": [match.to_dict() for match in self.matches],
            "rejected": [entry.frame_index for entry in self.rejected],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MemoryWrite:
    """写入结果。``accepted=False`` 时**库里没有多这一条**（拒收，不是低分收）。"""

    accepted: bool
    reason: str
    similarity: float | None = None
    evicted: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "evicted": list(self.evicted),
        }


@dataclass
class MemoryBank:
    """一条轨迹的记忆库（top-k 条，按相似度与新鲜度加权）。

    **唯一持有者**是 :class:`.occlusion.OcclusionTracker`——本类不自己跑推理，只存内容、
    只回答「这一帧观测与记忆对不对得上」。
    """

    obj_id: str
    params: MemoryParams = field(default_factory=MemoryParams)
    entries: list[MemoryEntry] = field(default_factory=list)
    #: 被拒收的观测（只留最近 ``rejected_keep`` 条，供运行摘要与 task 17 再获取排查用）。
    rejected: list[MemoryEntry] = field(default_factory=list)
    rejected_total: int = 0

    def __post_init__(self) -> None:
        if self.params.capacity < 1:
            raise ValueError(f"记忆库容量必须 ≥1，收到 {self.params.capacity}")

    # ── 只读视图 ────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.entries)

    @property
    def is_empty(self) -> bool:
        return not self.entries

    @property
    def latest(self) -> MemoryEntry | None:
        """最新一条被采信的记录（追下一帧提示、判 partial 都用它）。"""
        return self.entries[-1] if self.entries else None

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(entry.frame_index for entry in self.entries)

    def reference_ratio(self) -> float | None:
        """最新一条被采信的面积比；partial 帧本身不作为基准（读数被压低了）。"""
        for entry in reversed(self.entries):
            if entry.trusted:
                return entry.ratio
        return self.entries[-1].ratio if self.entries else None

    # ── 加权：新鲜度 × 可靠性 × 形状规整度 ──────────────────────────────
    def weight_of(self, entry: MemoryEntry, now_frame: int) -> float:
        return entry.weight_of(now_frame, self.params.decay_per_frame)

    def identity_entries(self) -> list[MemoryEntry]:
        """参与**身份判断**的条目：只取采信的；一条都没有时退回全部。

        被压住一半的帧几何是失真的（掩膜矮一大截、质心上移），让它参与投票会把共识拉偏
        ——遮一结束恢复正常的那一帧反倒被判成「拍到了别的东西」。
        """
        trusted = [entry for entry in self.entries if entry.trusted]
        return trusted or self.entries

    def _consensus(self, now_frame: int) -> tuple[tuple[float, float, float, float], tuple[float, float], float]:
        """按权重汇总**采信条目**的几何（袋底/宽度锚定）。库空时抛错，由调用方先判空。"""
        entries = self.identity_entries()
        weights = [self.weight_of(entry, now_frame) for entry in entries]
        total = sum(weights) or 1.0
        box = [0.0, 0.0, 0.0, 0.0]
        centre = [0.0, 0.0]
        for entry, weight in zip(entries, weights):
            for index in range(4):
                box[index] += entry.bbox[index] * weight
            centre[0] += entry.centroid[0] * weight
            centre[1] += entry.centroid[1] * weight
        return (
            tuple(value / total for value in box),  # type: ignore[return-value]
            (centre[0] / total, centre[1] / total),
            total,
        )

    def similarity_to_consensus(
        self,
        prediction: PredictionLike,
        tolerance: GeometryTolerance,
        *,
        now_frame: int = 0,
    ) -> float:
        """观测与库共识有多吻合（0–1）。**写入门禁**：低于阈值即拒收。

        对**单条**记忆取加权平均会高估一致性（库里若只剩一条脏记忆，它和自己当然像），
        所以这里比的是加权共识——单条库等价于那条记忆。
        """
        if not self.entries:
            return 1.0
        box, centre, _ = self._consensus(now_frame)
        anchor = MemoryEntry(
            frame_index=-1,
            bbox=tuple(int(round(value)) for value in box),  # type: ignore[arg-type]
            centroid=(centre[0], centre[1]),
            area=0,
            ratio=0.0,
        )
        return anchor.agreement(prediction, tolerance)

    # ── 写入 ────────────────────────────────────────────────────────────
    def write(
        self,
        entry: MemoryEntry,
        tolerance: GeometryTolerance,
        *,
        now_frame: int | None = None,
    ) -> MemoryWrite:
        """写入一条记忆。**对不上共识直接拒收**——错误记忆不进库。

        拒收而不是「收进来打低分」是刻意的：库里的东西会被拿来出下一帧的提示，
        一条脏记忆会把提示带到别的地方，后面每一帧都跟着错。
        """
        now = entry.frame_index if now_frame is None else now_frame
        if not self.entries:
            self.entries.append(entry)
            return MemoryWrite(True, REJECT_FIRST_SIGHTING, 1.0)

        similarity = self.similarity_to_consensus(entry, tolerance, now_frame=now)
        entry.similarity = similarity
        if similarity < self.params.min_similarity:
            self.reject(entry, reason=REJECT_SCENE_SWITCH, similarity=similarity)
            return MemoryWrite(False, REJECT_SCENE_SWITCH, similarity)

        self.entries.append(entry)
        evicted = self._prune(now)
        return MemoryWrite(True, "accepted", similarity, evicted)

    def reject(self, entry: MemoryEntry, *, reason: str, similarity: float) -> None:
        """记一次拒收（**不写进库**，只留档供运行摘要与排查用）。

        :meth:`write` 与 :class:`.occlusion.OcclusionTracker` 的观测门禁共用这一条路径，
        所以「拍到了别的东西」在库这一侧也有据可查——而不是只有上一层的判定。
        """
        entry.similarity = similarity
        self.rejected_total += 1
        self.rejected.append(entry)
        del self.rejected[:-4]  # 只留最近 4 条，恒定内存

    def _prune(self, now_frame: int) -> tuple[int, ...]:
        """按权重裁到 top-k；**最新一条永远保留**（不然下一帧就没提示可给了）。

        权重随年龄单调下降，所以正常情况下淘汰的就是最旧的那几条——但排序必须真按
        权重来，不能「保留最旧的 + 保留最新的」（那样会为了留两条陈的记忆而挤掉新鲜的）。
        """
        if len(self.entries) <= self.params.capacity:
            return ()
        newest = self.entries[-1]
        # 采信条目优先留下：半遮挡连着好几帧时，不先把完整可见的那些挤出去
        ranked = sorted(
            self.entries,
            key=lambda entry: (entry.trusted, self.weight_of(entry, now_frame), entry.frame_index),
        )
        room = self.params.capacity - 1
        others = [entry for entry in ranked if entry is not newest]
        kept = others[-room:] if room > 0 else []
        kept_ids = {id(entry) for entry in kept}
        evicted = tuple(
            entry.frame_index for entry in self.entries if id(entry) not in kept_ids
        )
        self.entries = [*kept, newest]
        return evicted

    # ── 查询 ────────────────────────────────────────────────────────────
    def match(
        self,
        prediction: PredictionLike,
        tolerance: GeometryTolerance,
        *,
        now_frame: int,
    ) -> tuple[MemoryMatch, ...]:
        """库内每条**采信**记忆对这次观测的匹配（按权重降序）。"""
        mask = getattr(prediction, "mask", None)
        matches = []
        for entry in self.identity_entries():
            similarity = entry.agreement(prediction, tolerance)
            weight = self.weight_of(entry, now_frame) * similarity
            matches.append(
                MemoryMatch(
                    entry=entry,
                    similarity=similarity,
                    weight=weight,
                    age=entry.age(now_frame),
                    mask_iou=entry.mask_iou(mask),
                )
            )
        return tuple(sorted(matches, key=lambda item: item.weight, reverse=True))

    def fused_agreement(
        self,
        prediction: PredictionLike,
        tolerance: GeometryTolerance,
        *,
        now_frame: int,
    ) -> tuple[float | None, tuple[MemoryMatch, ...]]:
        """按相似度 × 新鲜度加权汇总吻合度。

        返回 ``(None, ())`` 表示「没有任何记忆站在这一边」——调用方据此走遮挡而不是猜。
        权重全在阈值之下（记忆太旧）同样返回 ``None``。
        """
        matches = self.match(prediction, tolerance, now_frame=now_frame)
        usable = [item for item in matches if item.weight >= self.params.min_entry_weight]
        if not usable:
            return None, matches
        total = sum(item.weight for item in usable)
        if total <= 0.0:
            return None, matches
        return sum(item.weight * item.similarity for item in usable) / total, matches

    def query(
        self,
        prediction: PredictionLike | None = None,
        tolerance: GeometryTolerance | None = None,
        *,
        now_frame: int = 0,
    ) -> MemoryQuery:
        """一次完整的记忆回答（给运行摘要、task 17 再获取与调试用）。"""
        if not self.entries:
            return MemoryQuery(
                self.obj_id, 0, 0.0, None, None, None, None, (), tuple(self.rejected), "记忆库为空"
            )
        box, centre, total = self._consensus(now_frame)
        bbox = (int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3])))
        if prediction is None or tolerance is None:
            return MemoryQuery(
                self.obj_id,
                len(self.entries),
                total,
                bbox,
                (centre[0], centre[1]),
                self.reference_ratio(),
                None,
                (),
                tuple(self.rejected),
                f"{len(self.entries)} 条记忆的加权共识；未给观测故不判吻合度"
                + (f"；已拒收 {self.rejected_total} 次对不上的观测" if self.rejected_total else ""),
            )
        agreement, matches = self.fused_agreement(
            prediction, tolerance, now_frame=now_frame
        )
        return MemoryQuery(
            self.obj_id,
            len(self.entries),
            total,
            bbox,
            (centre[0], centre[1]),
            self.reference_ratio(),
            agreement,
            matches,
            tuple(self.rejected),
            f"{len(matches)} 条记忆参与加权"
            + (f"，已拒收 {self.rejected_total} 次对不上的观测" if self.rejected_total else ""),
        )

    def describe(self) -> dict:
        return {
            "obj_id": self.obj_id,
            "capacity": self.params.capacity,
            "min_similarity": self.params.min_similarity,
            "decay_per_frame": self.params.decay_per_frame,
            "entries": [
                {
                    "frame_index": entry.frame_index,
                    "ratio": entry.ratio,
                    "score": round(entry.score, 4),
                    "trusted": entry.trusted,
                    "similarity": round(entry.similarity, 4),
                }
                for entry in self.entries
            ],
            "rejected_total": self.rejected_total,
        }