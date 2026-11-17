"""遮挡标记、恢复续接与轨迹交出（visible → partial → occluded → reappeared → lost）。

**与源主路径的对应关系**（只借调用链与语义，不搬代码、不跑源权重）：

| 本模块 | 源主路径 | 代码证据 |
|--------|----------|----------|
| :class:`.memory_bank.MemoryBank`——top-k 条记忆（掩膜/外接框/面积比） | ``MemoryAttentionLayer._forward_ca(tgt, memory, ...)``：把上一帧的 ``memory`` 作为 cross-attention 的 key/value 参与当前帧推理 | ``memory_attention.py:66-81`` |
| :attr:`VisibilityVerdict.used_memory`——当前帧结论被记忆条件化 | 同上（``q`` 来自当前帧、``k/v`` 来自记忆，输出因此被记忆影响） | ``memory_attention.py:74-80`` |
| :attr:`OcclusionTracker.handoff`——记忆失效后交出轨迹 | ``MemoryAttention.forward(..., num_obj_ptr_tokens=...)``：对象指针记忆读不到时不再参与推理，等待外部重新给提示 | ``memory_attention.py:122,153`` |
| 遮挡帧留 ``None``、不丢弃也不插值 | 传播循环逐帧 yield，对象当前帧没检出不补值 | ``vos_inference.py:224-231`` |

记忆由 :class:`.memory_bank.MemoryBank` **单一持有**（本类不再自己存第二份）：库负责
top-k、相似度加权与拒收脏记忆，本类只负责「拿库里的证据做判定」。

**四条纪律**（第 1、2、3 条与 :mod:`.propagate` 一致）：

1. **恒定算力**：每条轨迹只留**固定条数**（默认 4 条）的记忆，不攒掩膜序列。
   状态与曲线本身每帧只占一个标量/指针，600 帧与 1200 帧的峰值内存同量级。
2. **遮挡不丢弃**：遮挡帧在面积序列里**保留位置、值为 ``None``**，状态显式记 ``occluded``；
   **不插值、不沿用上一帧数值**——拿一条假曲线去算毫升和 ETA 比没有曲线更危险。
3. **不猜**：恢复点若与记忆对不上（换了袋子、串了槽位、场景切换），宁可继续记遮挡直到
   超过丢失窗口，**绝不**静默新建一条轨迹把两袋的曲线接在一起；也**绝不**把这一次脏观测
   写进记忆库（否则下一帧的提示就跟着跑偏）。
4. **交出而非新建**：超过丢失窗口置 ``lost`` 并冻结轨迹，产出 :class:`ReacquireHandoff`
   交给 task 17 的再获取路径；在被认领之前，后续帧一律记 ``lost``，曲线不复活。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .memory_bank import (
    REJECT_SCENE_SWITCH,
    MemoryBank,
    MemoryEntry,
    MemoryParams,
    MemoryWrite,
)
from .predictor import MaskPrediction, mask_centroid

__all__ = [
    "VISIBLE",
    "PARTIAL",
    "OCCLUDED",
    "REAPPEARED",
    "LOST",
    "VISIBILITY_STATES",
    "HANDOFF_REASON",
    "VisibilityParams",
    "MemoryEntry",
    "TrackMemory",
    "ReacquireHandoff",
    "VisibilityVerdict",
    "OcclusionTracker",
    "summarise_states",
    "handoff_to_dict",
]

logger = logging.getLogger("infusion_monitor.core.occlusion")

#: 本帧完整可见：掩膜在，且面积比与「记忆 + 滴速趋势」的预期一致。
VISIBLE = "visible"
#: 本帧只看到部分袋体：掩膜在，但面积比明显低于趋势（手/被子压住一角）。
PARTIAL = "partial"
#: 本帧完全看不见：没有掩膜，面积比留 ``None``，不插值。
OCCLUDED = "occluded"
#: 遮挡后按记忆比对接回同一袋（``obj_id`` 不变，曲线在恢复点续上，中间仍留空）。
REAPPEARED = "reappeared"
#: 连续遮挡超过丢失窗口，轨迹已交出，等 task 17 的再获取路径认领。
LOST = "lost"

#: 全部可见度状态（起锚前的帧为 ``None``，表示「还没开始跟」）。
VISIBILITY_STATES = (VISIBLE, PARTIAL, OCCLUDED, REAPPEARED, LOST)

#: 交出轨迹的原因串；task 17 的再获取路径按此筛选。
HANDOFF_REASON = "occlusion_lost_window_exceeded"


@dataclass(frozen=True)
class VisibilityParams:
    """可见度判定的门限。取值理由写在字段旁，全部可由病区实测调整。"""

    #: 面积比低于「记忆预测值」的该倍数即判 partial。0.75 ≈ 少了四分之一。
    partial_ratio: float = 0.75
    #: 恢复点与记忆的吻合度门限（0–1）。低于它不接回，继续记遮挡。
    resume_agreement: float = 0.50
    #: 质心位移容差（像素）。超出即认为不是原来那袋。
    centroid_tolerance_px: float = 60.0
    #: 袋底纵向位移容差（像素）。袋底固定、液面上升，所以袋底是最稳的锚。
    bottom_tolerance_px: float = 40.0
    #: 连续遮挡多少帧后置 lost。0.5 s/帧 × 8 = 4 s，超过护士手经过的时长。
    lost_window_frames: int = 8
    #: 每帧液面下降量的指数平滑系数（用于「按趋势应该有这么多」）。
    drain_ema_alpha: float = 0.3
    #: 每帧面积比最多可下降多少仍算「滴速造成」。超过则归因遮挡，不进趋势。
    #: 0.5 s 一帧时 0.04/帧 ≈ 每秒 8%，已远快于任何输液泵的正常滴速。
    max_drain_per_frame: float = 0.04


#: 记忆条目本体在 :mod:`.memory_bank`（库负责 top-k / 加权 / 拒收），这里保留旧名。
TrackMemory = MemoryEntry


@dataclass(frozen=True)
class ReacquireHandoff:
    """轨迹交出单：丢失窗口用尽后交给 task 17 的再获取路径。

    交出的是**同一条轨迹的续接线索**（最后可见帧、几何、参考面积比），不是「新建一条
    曲线」的许可——认领与否由再获取路径决定。
    """

    obj_id: str
    lost_frame: int
    last_seen_frame: int
    missing_frames: int
    bbox: tuple[int, int, int, int] | None
    centroid: tuple[float, float] | None
    cavity_area_px: int
    reference_area_ratio: float | None
    reason: str = HANDOFF_REASON


@dataclass(frozen=True)
class VisibilityVerdict:
    """一帧的可见度判定结果。"""

    frame_index: int
    state: str
    area_ratio: float | None
    score: float
    #: 与记忆的吻合度（0–1）。恢复点用它决定「还是不是同一袋」。
    confidence: float = 0.0
    #: 本帧判定是否用到了记忆（partial 与 reappeared 一定用到）。
    used_memory: bool = False
    #: 中文原因，直接给护士端/审计看。
    reason: str = ""


@dataclass
class OcclusionTracker:
    """一条轨迹的可见度状态机。

    状态迁移::

        (起锚) ──首帧检出──> visible <──┐
                            │ partial <┤
                            ▼          │ 面积比≈趋势且吻合度够
                          occluded ────┘  → reappeared（同一条轨迹接回）
                            │ 连续遮挡 > lost_window_frames
                            ▼
                          lost ──认领（task 17 再获取）──> visible

    每帧 :meth:`observe` 一次，产出 :class:`VisibilityVerdict`，并把
    ``area_ratio``（遮挡时为 ``None``）与 ``state`` 写进本轨迹的序列。
    """

    obj_id: str
    cavity_area_px: int
    frame_count: int
    params: VisibilityParams = field(default_factory=VisibilityParams)
    memory_params: MemoryParams = field(default_factory=MemoryParams)
    ratios: list[float | None] = field(default_factory=list)
    states: list[str | None] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    _bank: MemoryBank = field(default=None, repr=False)  # type: ignore[assignment]
    _missing_streak: int = 0
    _handoff: ReacquireHandoff | None = field(default=None, repr=False)
    _reacquired_at: int | None = field(default=None, repr=False)
    #: 最近一次写入记忆的结果（拒收时带原因，给运行摘要与 task 17 排查用）
    last_write: MemoryWrite | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.frame_count < 1:
            raise ValueError("frame_count 必须 ≥1")
        if self.cavity_area_px <= 0:
            raise ValueError(f"{self.obj_id} 的袋腔面积非法：{self.cavity_area_px}")
        self.ratios = [None] * self.frame_count
        self.states = [None] * self.frame_count
        self.scores = [0.0] * self.frame_count
        self._bank = MemoryBank(obj_id=self.obj_id, params=self.memory_params)

    # ── 只读视图 ────────────────────────────────────────────────────────
    @property
    def bank(self) -> MemoryBank:
        """本轨迹的记忆库（**唯一**一份记忆状态）。"""
        return self._bank

    @property
    def memory(self) -> MemoryEntry | None:
        """最新一条被采信的记忆（追下一帧提示、判 partial 都用它）。"""
        return self._bank.latest

    def memory_query(self, frame_index: int, prediction: MaskPrediction | None = None):
        """记忆库对当前帧的一次汇总（给运行摘要、task 17 再获取与调试用）。"""
        return self._bank.query(
            prediction, self.params if prediction is not None else None, now_frame=frame_index
        )

    @property
    def handoff(self) -> ReacquireHandoff | None:
        """丢失窗口用尽后产出的交出单；未被认领前一直可查。"""
        return self._handoff

    @property
    def is_lost(self) -> bool:
        return self.states and LOST in self.states

    @property
    def reacquired_at(self) -> int | None:
        """被再获取路径认领的那一帧；``None`` 表示还没人认领（曲线仍冻结）。"""
        return self._reacquired_at

    @property
    def missing_frame_count(self) -> int:
        """已观测但看不见的帧数（occluded + lost）。

        未起锚 / 还没观测到的帧不计入——那些位置本来就该是空的。
        """
        return sum(
            1 for ratio, state in zip(self.ratios, self.states) if ratio is None and state is not None
        )

    def counts_by_state(self) -> dict[str, int]:
        """各状态的帧数——给运行摘要与护士端一眼看「这段拍糊了多少」。"""
        tally = {state: 0 for state in VISIBILITY_STATES}
        for state in self.states:
            if state in tally:
                tally[state] += 1
        return tally

    def reacquire_from(self, frame_index: int) -> None:
        """task 17 的再获取路径找到这袋后调用：解除 lost，重新按记忆续接曲线。

        **只有再获取路径能调用它**——遮挡期间不得借此偷偷把曲线接上。
        """
        if not 0 <= frame_index < self.frame_count:
            raise ValueError(f"frame_index={frame_index} 越界（共 {self.frame_count} 帧）")
        self._reacquired_at = frame_index
        self._missing_streak = 0
        memory = self._bank.latest
        if memory is not None:
            memory.frame_index = min(memory.frame_index, frame_index)

    # ── 主循环：每帧一次 ───────────────────────────────────────────────
    def observe(self, frame_index: int, prediction: MaskPrediction | None) -> VisibilityVerdict:
        """吸收一帧观测，返回可见度判定。

        ``prediction`` 为 ``None`` 或空掩膜即「这一帧看不见它」——与
        :meth:`.propagate.ObjectTrack.absorb` 的约定一致。
        """
        if not 0 <= frame_index < self.frame_count:
            raise ValueError(f"frame_index={frame_index} 越界（共 {self.frame_count} 帧）")

        verdict = self._decide(frame_index, prediction)
        self.ratios[frame_index] = verdict.area_ratio
        self.states[frame_index] = verdict.state
        self.scores[frame_index] = verdict.score
        return verdict

    def _decide(self, frame_index: int, prediction: MaskPrediction | None) -> VisibilityVerdict:
        missing = prediction is None or prediction.is_empty

        # 已 lost 且未被再获取认领：曲线冻结，不偷偷续上
        if self._handoff is not None and self._reacquired_at is None:
            self._missing_streak += 1
            return VisibilityVerdict(
                frame_index,
                LOST,
                None,
                0.0,
                confidence=0.0,
                used_memory=True,
                reason="轨迹已交出，等待认领或再获取；本帧不续接曲线",
            )

        if missing:
            return self._decide_missing(frame_index)

        area = int(prediction.area)  # type: ignore[union-attr]
        ratio = round(area / float(self.cavity_area_px), 5)
        score = float(prediction.score)  # type: ignore[union-attr]

        # 首见：没有记忆可依，直接记 visible（不存在「接错对象」的可能）
        if self._bank.is_empty:
            self._commit(frame_index, prediction, ratio)  # type: ignore[arg-type]
            return VisibilityVerdict(
                frame_index,
                VISIBLE,
                ratio,
                score,
                confidence=1.0,
                used_memory=False,
                reason="本帧首次检出，记为完整可见",
            )

        gap = self._missing_streak + 1
        if self._missing_streak > 0:
            agreement, matches = self._bank.fused_agreement(
                prediction, self.params, now_frame=frame_index  # type: ignore[arg-type]
            )
            if agreement is not None and agreement >= self.params.resume_agreement:
                self._commit(frame_index, prediction, ratio, gap)  # type: ignore[arg-type]
                return VisibilityVerdict(
                    frame_index,
                    REAPPEARED,
                    ratio,
                    score,
                    confidence=round(agreement, 4),
                    used_memory=True,
                    reason=(
                        f"遮挡 {self._missing_streak} 帧后按记忆比对接回同一袋"
                        f"（{len(matches)} 条记忆加权吻合度 {agreement:.2f}）"
                    ),
                )
            # 对不上：不猜，继续记遮挡（不写记忆、不出面积比）
            self._missing_streak += 1
            shown = 0.0 if agreement is None else agreement
            return self._missing_verdict(
                frame_index,
                score,
                f"恢复位置与记忆对不上（加权吻合度 {shown:.2f} < {self.params.resume_agreement:.2f}），"
                "不猜接回，继续记遮挡",
            )

        # 连续可见：先判 partial（读数被压低），再判「还是不是记忆中那一袋」。
        # 顺序有讲究：半遮挡（被子/手压住）会让掩膜矮一大截、质心跟着上移，
        # 单看几何与「跑到别处」区分度有限；但它的面积比一定低于滴速趋势，
        # 先被 partial 规则接住，剩下的才是真正的场景切换。
        latest = self._bank.latest
        predicted = latest.predicted_ratio(1)  # type: ignore[union-attr]
        if latest.hits >= 2 and predicted > 0.0 and ratio < predicted * self.params.partial_ratio:
            observed_share = ratio / predicted
            # 面积比不写回记忆：这一帧的读数被遮挡压低，拿它当基线会让本次遮挡自我掩盖
            self._commit(frame_index, prediction, ratio, trust_ratio=False)  # type: ignore[arg-type]
            return VisibilityVerdict(
                frame_index,
                PARTIAL,
                ratio,
                score,
                confidence=round(observed_share, 4),
                used_memory=True,
                reason=(
                    f"本帧只看到部分袋体（面积比 {ratio:.3f} 明显低于按滴速预期的 {predicted:.3f}），"
                    "建议换个角度补拍"
                ),
            )

        # 面积比与滴速趋势对得上，位置却与记忆差得远：手机扫到别处了。收下来等于把
        # 错误记忆写进库（下一帧的提示也会跟着跑偏），并给护士端一个假的余量数字，
        # 所以按「看不见」处理——不写记忆、不出面积比。
        consistency = self._bank.similarity_to_consensus(
            prediction, self.params, now_frame=frame_index  # type: ignore[arg-type]
        )
        if consistency < self.memory_params.min_consistency:
            self._missing_streak += 1
            self._bank.reject(
                self._entry_for(frame_index, prediction, ratio),  # type: ignore[arg-type]
                reason=REJECT_SCENE_SWITCH,
                similarity=consistency,
            )
            # last_write 记的是「上一帧观测交给记忆库之后的处置」，收与拒都算
            self.last_write = MemoryWrite(False, REJECT_SCENE_SWITCH, consistency)
            return self._missing_verdict(
                frame_index,
                score,
                f"本帧读数与滴速趋势一致，但位置与记忆里的这一袋对不上"
                f"（相似度 {consistency:.2f} < {self.memory_params.min_consistency:.2f}），"
                "判为拍到了别的东西；不写入记忆、不出面积比",
            )

        self._commit(frame_index, prediction, ratio)  # type: ignore[arg-type]
        return VisibilityVerdict(
            frame_index,
            VISIBLE,
            ratio,
            score,
            confidence=1.0,
            used_memory=False,
            reason="本帧完整可见，面积比与滴速趋势一致",
        )

    def _decide_missing(self, frame_index: int) -> VisibilityVerdict:
        self._missing_streak += 1
        return self._missing_verdict(frame_index, 0.0, self._missing_reason())

    def _missing_reason(self) -> str:
        if self._missing_streak > self.params.lost_window_frames:
            return (
                f"连续遮挡 {self._missing_streak} 帧已超过丢失窗口 "
                f"{self.params.lost_window_frames} 帧，轨迹交出待再获取"
            )
        return f"本帧被遮挡或未检出（连续第 {self._missing_streak} 帧），面积比留空不插值"

    def _missing_verdict(self, frame_index: int, score: float, reason: str) -> VisibilityVerdict:
        memory = self._bank.latest
        state = OCCLUDED
        if self._missing_streak > self.params.lost_window_frames:
            state = LOST
            if self._handoff is None:
                self._handoff = ReacquireHandoff(
                    obj_id=self.obj_id,
                    lost_frame=frame_index,
                    last_seen_frame=memory.frame_index if memory else -1,
                    missing_frames=self._missing_streak,
                    bbox=memory.bbox if memory else None,
                    centroid=memory.centroid if memory else None,
                    cavity_area_px=self.cavity_area_px,
                    reference_area_ratio=self._bank.reference_ratio(),
                )
                logger.info(
                    "%s 在第 %d 帧超出丢失窗口，交出轨迹供再获取", self.obj_id, frame_index
                )
        return VisibilityVerdict(
            frame_index,
            state,
            None,
            score,
            confidence=0.0,
            used_memory=memory is not None,
            reason=reason,
        )

    def _commit(
        self,
        frame_index: int,
        prediction: MaskPrediction,
        ratio: float,
        gap: int = 1,
        *,
        trust_ratio: bool = True,
    ) -> None:
        """把一帧确认为「看见了」，写入记忆库（含下降趋势 EMA）。

        ``trust_ratio=False`` 用于 partial 帧：几何（外接框/质心/掩膜）照常更新以维持
        跟踪，但**面积比不写回**——那一帧的读数被遮挡压低了，拿它当基线会让本次遮挡
        「把自己学成正常」，之后每一帧都判成 visible。

        写入本身可能仍被 :class:`.memory_bank.MemoryBank` **拒收**（与库共识对不上）；
        正常路径不会走到这里（``_decide`` 已先做污染判定），拒收只作为兜底并如实记进
        :attr:`last_write`，不静默吞掉。
        """
        previous = self._bank.latest
        self._missing_streak = 0
        entry = self._entry_for(frame_index, prediction, ratio)
        if previous is None:
            self.last_write = self._bank.write(entry, self.params, now_frame=frame_index)
            return

        drain = previous.drain_per_frame
        trusted_ratio = ratio
        if trust_ratio:
            # 趋势只吸收「滴速造成」的下降：每帧降幅超过物理上限的（多半是被挡了一块）
            # 一律不进 EMA，否则一次遮挡会被当成袋子在猛漏。
            per_frame = (previous.ratio - ratio) / max(1, gap)
            if 0.0 < per_frame <= self.params.max_drain_per_frame:
                alpha = self.params.drain_ema_alpha
                drain += alpha * (per_frame - drain)
        else:
            trusted_ratio = previous.ratio

        entry.ratio = trusted_ratio
        entry.trusted = trust_ratio
        entry.drain_per_frame = drain
        entry.hits = previous.hits + 1
        self.last_write = self._bank.write(entry, self.params, now_frame=frame_index)

    @staticmethod
    def _entry_for(
        frame_index: int, prediction: MaskPrediction, ratio: float
    ) -> MemoryEntry:
        """把一帧观测组装成记忆条目（几何 + 掩膜 + 置信 + 面积比）。"""
        return MemoryEntry(
            frame_index=frame_index,
            bbox=prediction.bbox or (0, 0, 0, 0),
            centroid=mask_centroid(prediction.mask) or (0.0, 0.0),
            area=prediction.area,
            ratio=ratio,
            mask=prediction.mask,
            score=prediction.score,
        )


def summarise_states(states: Sequence[str | None]) -> str:
    """把可见度序列压成一段可读摘要（给日志与运行摘要用）。"""
    if not states:
        return "空序列"
    runs: list[tuple[str | None, int]] = []
    for state in states:
        if runs and runs[-1][0] == state:
            runs[-1] = (state, runs[-1][1] + 1)
        else:
            runs.append((state, 1))
    parts = [f"{state or '未起锚'}×{count}" for state, count in runs]
    return " → ".join(parts)


def handoff_to_dict(handoff: ReacquireHandoff | None) -> dict | None:
    """交出单的 JSON 形态（进接口响应与审计）；``None`` 原样返回。"""
    if handoff is None:
        return None
    x0, y0, x1, y1 = handoff.bbox or (0, 0, 0, 0)
    return {
        "obj_id": handoff.obj_id,
        "lost_frame": handoff.lost_frame,
        "last_seen_frame": handoff.last_seen_frame,
        "missing_frames": handoff.missing_frames,
        "bbox": [x0, y0, x1, y1],
        "centroid": list(handoff.centroid) if handoff.centroid else None,
        "cavity_area_px": handoff.cavity_area_px,
        "reference_area_ratio": handoff.reference_area_ratio,
        "reason": handoff.reason,
    }