"""跨巡视轮次的袋体重锚：这一轮拍到的候选，还是上一轮那一只袋子吗？

**现场问题是这样的**：护士每 1–2 小时拍一轮。第一轮 B01 在挂点 0，第二轮换个角度
拍，B01 还在挂点 0 但框歪了一点、液面低了一截；护士长在床旁点「B01 剩余 42%」，
面板得续上**同一条**曲线，而不是新起一条从满袋开始的线。反过来，换了新袋之后，
新袋绝不能接到旧袋的曲线上——那会让护士看到「换完液反而更满了」。

**本模块只做判定，不做打分**：打分（几何 + 颜色 + 液位 + 标签区四项加权、缺证据的
项退出加权）全在 :mod:`identity.reanchor_score`；描述子（颜色直方图 / 液位特征 /
标签区存在性，含降级档位）在 :mod:`identity.descriptor`。这里负责的是三件业务事：

1. **分流**：这一袋是首次巡视 / 上一轮已换掉（fresh）/ 还是进入打分；
2. **判定**：够不够格接上（同 :mod:`identity.reanchor_score` 的门限）；
3. **拒绝猜测**：认不准就把指针收回来，要求护士在候选列表里认领。

**拒绝猜测的硬规则**（宁可让护士点一行，也不接错）：

1. 最高分 < ``min_confidence`` 且位置对不上 → ``unbound(low_confidence)``；
2. 位置对得上但外观变了（颜色 / 液位 / 标签有塌了，或**液柱比上一轮还高**）→
   ``unbound(appearance_changed)``——**换袋必低置信**，新袋绝不会被接到旧曲线上；
3. 最高分与次高分之差 < ``min_margin`` → ``unbound(ambiguous_slot)``；
   **同槽位多候选时禁止默认取置信度最高者**——「看起来最像」不等于「就是它」；
4. 外观不可用（区域太小 / 像素不足）→ ``unbound(appearance_unavailable)``，
   不拿几何分硬接；
5. 命中的上一轮袋序与对端本次给的 ``bag_seq`` 不一致 → ``unbound(bag_seq_mismatch)``。
   系统认出这是上一轮的 B01，可对端把新袋编成了 B03，那说明两边的账对不上，
   得让人看一眼，而不是把曲线接到一个护士马上会否认的袋序上。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .descriptor import BagAppearance, describe_region, descriptor_from_session
from .reanchor_score import (
    CandidateScore,
    ScoreParams,
    rank_candidates,
    score_pair,
)

__all__ = [
    "ANCHOR_STATES",
    "REANCHOR_REASONS",
    "BagAppearance",
    "BagDescriptor",
    "ReanchorParams",
    "ReanchorVerdict",
    "SlotObservation",
    "describe_region",
    "descriptor_from_session",
    "reanchor_slots",
    "score_pair",
]

#: 重锚判定结果。
ANCHOR_STATES = ("bound", "unbound", "fresh")

#: 判定原因码（响应与审计里都带这一项，护士端据此给不同的提示文案）。
REANCHOR_REASONS = (
    "first_round",             # 这一输液架没有可比的历史会话
    "previous_replaced",       # 上一轮的袋都已换掉，本轮是新袋
    "anchored",                # 高置信且无歧义地接上了同一袋
    "low_confidence",          # 最高分不够，不敢接
    "appearance_changed",      # 位置对得上、但外观变了（多半是换了新袋）→ 人工认领
    "ambiguous_slot",          # 前两名分差太小，同槽位多候选
    "bag_seq_mismatch",        # 认出的袋与对端给的袋序对不上
    "appearance_unavailable",  # 外观描述子不可用（区域太小/像素不足），一律不自动接
    "no_candidate",            # 本轮这个挂点根本没枚举出候选
)

#: 打分门限的旧名（新代码用 :class:`identity.reanchor_score.ScoreParams`）。
ReanchorParams = ScoreParams

#: 描述子的旧名：现在由 :mod:`identity.descriptor` 定义，此处保持导入路径不变。
BagDescriptor = BagAppearance


@dataclass(frozen=True)
class SlotObservation:
    """本轮某个挂点上实际观测到的东西（认不出就是 ``descriptor=None``）。"""

    slot_index: int
    bag_seq: str
    descriptor: BagAppearance | None
    candidate_id: str | None = None


@dataclass(frozen=True)
class ReanchorVerdict:
    """一个挂位的重锚判定。``anchor_state`` 决定响应里它算不算已绑定。"""

    slot_index: int
    bag_seq: str
    anchor_state: str
    reason: str
    confidence: float | None = None
    runner_up: float | None = None
    matched_bag_seq: str | None = None
    matched_session_id: str | None = None
    previous_shift_key: str | None = None
    #: 全部历史候选的打分（倒序），供护士核对「系统凭什么这么判」
    scores: tuple[tuple[str, float, float], ...] = ()
    #: 本轮描述子档位（full / color_only / geometry_only）
    descriptor_mode: str | None = None
    #: 拖分最多的那一项（geometry / colour / level / label）；认准时也带出，
    #: 护士端据此提示「是颜色不一样 / 是液位比上一轮高了」
    driver: str | None = None

    @property
    def bound(self) -> bool:
        return self.anchor_state == "bound"

    def to_json(self) -> dict[str, Any]:
        return {
            "slot_index": self.slot_index,
            "bag_seq": self.bag_seq,
            "anchor_state": self.anchor_state,
            "reason": self.reason,
            "confidence": self.confidence,
            "runner_up": self.runner_up,
            "matched_bag_seq": self.matched_bag_seq,
            "matched_session_id": self.matched_session_id,
            "previous_shift_key": self.previous_shift_key,
        }


# ── 判定 ──────────────────────────────────────────────────────────────────


def reanchor_slots(
    observations: Sequence[SlotObservation],
    previous: Sequence[BagAppearance],
    params: ScoreParams | None = None,
    *,
    previous_seen: bool | None = None,
) -> list[ReanchorVerdict]:
    """逐挂位给出重锚判定（不猜、不默认取最高分）。

    ``previous`` 传该输液架**全部**历史会话的描述子（含已换袋的）。函数内部按状态分流：

    * 这个袋序上一轮就**确认换掉了**、且没有同袋序仍在监测 → ``fresh``：旧曲线已封口，
      现在挂上去的是新袋，绝不把新袋接到旧曲线上；
    * 仍在监测的历史袋才进入打分集合——**同槽位多候选时禁止默认取分数最高的那个**。

    ``previous_seen`` 只影响「本架有没有历史会话」的判断（``first_round`` 与
    ``previous_replaced`` 两种提示的区别），默认按 ``previous`` 是否为空推断。
    """
    p = params or ScoreParams()
    seen = bool(previous) if previous_seen is None else previous_seen
    return [_verdict_for(observation, previous, p, seen) for observation in observations]


def _verdict_for(
    observation: SlotObservation,
    previous: Sequence[BagAppearance],
    params: ScoreParams,
    previous_seen: bool = False,
) -> ReanchorVerdict:
    """单个挂位的判定。

    **未绑定时不带出 ``matched_session_id``**：那个字段一旦有值，上层就可能当成
    「已经接上了上一轮那条曲线」——认不准就必须连指针都不给。
    """
    if observation.descriptor is None:
        return ReanchorVerdict(
            slot_index=observation.slot_index,
            bag_seq=observation.bag_seq,
            anchor_state="unbound",
            reason="no_candidate",
        )

    live = [item for item in previous if item.state != "replaced"]
    same_seq_live = [item for item in live if item.bag_seq == observation.bag_seq]
    same_seq_closed = [
        item for item in previous if item.bag_seq == observation.bag_seq and item.state == "replaced"
    ]
    if same_seq_closed and not same_seq_live:
        return ReanchorVerdict(
            slot_index=observation.slot_index,
            bag_seq=observation.bag_seq,
            anchor_state="fresh",
            reason="previous_replaced",
            descriptor_mode=observation.descriptor.descriptor_mode,
        )
    if not live:
        return ReanchorVerdict(
            slot_index=observation.slot_index,
            bag_seq=observation.bag_seq,
            anchor_state="fresh",
            reason="previous_replaced" if previous_seen else "first_round",
            descriptor_mode=observation.descriptor.descriptor_mode,
        )

    ranked = rank_candidates(observation.descriptor, live, params)
    best = ranked.best
    assert best is not None  # live 非空 ⇒ 至少一个候选
    second_score = ranked.runner_up.total if ranked.runner_up is not None else None

    def _verdict(state: str, reason: str, **kwargs: Any) -> ReanchorVerdict:
        bound = state == "bound"
        return ReanchorVerdict(
            slot_index=observation.slot_index,
            bag_seq=observation.bag_seq,
            anchor_state=state,
            reason=reason,
            confidence=best.total,
            runner_up=second_score,
            matched_bag_seq=best.bag_seq,
            matched_session_id=best.session_id if bound else None,
            previous_shift_key=best.shift_key if bound else None,
            scores=tuple((item.bag_seq, item.total, item.geometry) for item in ranked.ordered),
            descriptor_mode=observation.descriptor.descriptor_mode if observation.descriptor else None,
            driver=best.driver,
            **kwargs,
        )

    if not ranked.appearance_available:
        return _verdict("unbound", "appearance_unavailable")
    if best.level_vetoed:
        # 液柱比上一轮还高：同一袋的液面只会下降，升高只有一种解释——换新袋了。
        # 这条不走「总分不够」的软判断，直接一票否决（权重会把这么强的信号稀释掉）。
        return _verdict("unbound", "appearance_changed")
    if best.total < params.min_confidence:
        # 位置对得上、外观塌了 → 归因成「换袋」，护士端提示语不一样
        appearance = best.appearance
        if (
            best.geometry >= params.geometry_trust
            and appearance is not None
            and appearance < params.appearance_changed_threshold
        ):
            return _verdict("unbound", "appearance_changed")
        return _verdict("unbound", "low_confidence")
    if second_score is not None and (best.total - second_score) < params.min_margin:
        return _verdict("unbound", "ambiguous_slot")
    if best.bag_seq != observation.bag_seq:
        return _verdict("unbound", "bag_seq_mismatch")
    return _verdict("bound", "anchored")


def score_details(ranked_scores: Sequence[CandidateScore]) -> list[dict[str, Any]]:
    """把排序结果转成响应/审计用的 JSON 列表（逐项留痕，护士端可展开核对）。"""
    return [item.to_json() for item in ranked_scores]