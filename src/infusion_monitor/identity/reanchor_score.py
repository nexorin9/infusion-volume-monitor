"""跨轮次重锚打分：把「这一轮这几只袋子」和「上一轮那几只」排出个高下。

打分是**加权平均**而不是「几何分 × 外观分」这种写死的两项之和：某一侧的某项证据
不存在时（比如标签区没看清、库里是旧版描述子），那一项**退出加权**而不是当 0 分
——0 分的意思是「看着不像」，缺证据的意思是「没看见」，把两者混同会系统性低估
证据齐备的那一侧。

四项判据（详见 :mod:`identity.descriptor`）：

======  ==========================  ============================================
项      权重默认                    惩罚什么
======  ==========================  ============================================
几何    0.34                        横向中心 / 袋底 / 横向宽度（对液位下降不敏感）
颜色    0.30                        色相直方图交集
液位    0.22                        **液柱变高**（同一袋只会往下走）
标签    0.14                        标签区有无 / 面积 / 质心位移
======  ==========================  ============================================

**液位只罚「升高」**：两轮之间液面只会下降，升高只有一种解释——换新袋了；而
「掉得多」是正常业务（两小时没巡视），拿它扣分等于罚护士巡视间隔长。反过来，被手
挡住的袋子看到的液面偏低，也不会被误罚——保守方向一致。

**拒绝猜测的排序语义**：排名只回答「最像哪个」，**不回答「就是哪个」**。判定
（够不够高置信、与次高分差够不够大）留在 :mod:`identity.reanchor`，那里连同
「同槽位多候选不许默认取最高分」一起执行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .descriptor import BagAppearance

__all__ = [
    "SCORE_TERMS",
    "CandidateScore",
    "RankedCandidates",
    "ScoreParams",
    "colour_score",
    "geometry_score",
    "label_score",
    "level_score",
    "rank_candidates",
    "score_pair",
]

#: 打分项（响应与审计里 ``driver`` 字段取值也用这套名字）。
SCORE_TERMS = ("geometry", "colour", "level", "label")


@dataclass(frozen=True)
class ScoreParams:
    """重锚打分门限。默认值按合成场景标定，真机上由信息科按病房实测调整。"""

    geometry_weight: float = 0.34
    colour_weight: float = 0.30
    level_weight: float = 0.22
    label_weight: float = 0.14
    #: 几何三项各自的惩罚系数（输入都是 0–1 的画面比例差）
    centre_x_penalty: float = 4.0
    bottom_y_penalty: float = 2.5
    width_penalty: float = 3.0
    #: 液柱允许的「免费」浮动（机位微动 / 弯月面受反光影响），超出才计罚
    level_rise_floor: float = 0.02
    #: 液柱升高多少算「升满了这一项」（归一化到 0 分）
    level_rise_tolerance: float = 0.08
    #: 液柱升高超限时是否**一票否决**（不靠加权稀释掉）。默认开：液面在同一袋里
    #: 只会下降，升高只有一种解释——换了新袋。这条是物理约束，不是打分倾向。
    level_veto: bool = True
    #: 标签区面积差超过该值算「面积对不上」
    label_coverage_tolerance: float = 0.25
    #: 标签区质心横移超过该值算「位置对不上」
    label_shift_tolerance: float = 0.10
    #: 低于该总分不接（``unbound/low_confidence``）
    min_confidence: float = 0.62
    #: 最高分与次高分之差下限，低于即 ``unbound/ambiguous_slot``
    min_margin: float = 0.08
    #: 几何分达到该值才算「位置对得上」——只有这时外观变化才值得单独归因
    geometry_trust: float = 0.75
    #: 外观（颜色/液位/标签加权后）低于该值且位置对得上 → ``appearance_changed``
    appearance_changed_threshold: float = 0.50


# ── 各项判据 ──────────────────────────────────────────────────────────────


def geometry_score(
    current: BagAppearance, previous: BagAppearance, p: ScoreParams | None = None
) -> float:
    """三项几何差异的线性惩罚，全部落在 [0, 1]。

    用线性惩罚而不是高斯：三项差异都能解释给护士听（「横向偏了 12%、袋底偏了 4%），
    出了问题能一眼看出是哪一项把分打下去的。
    """
    params = p or ScoreParams()
    penalty = (
        abs(current.centre_x - previous.centre_x) * params.centre_x_penalty
        + abs(current.bottom_y - previous.bottom_y) * params.bottom_y_penalty
        + abs(current.width - previous.width) * params.width_penalty
    )
    return max(0.0, 1.0 - penalty)


def colour_score(
    current: BagAppearance, previous: BagAppearance
) -> float | None:
    """色相直方图交集（∈[0,1]）；任一侧外观不可用返回 ``None``（不是 0）。"""
    if not (current.appearance_available and previous.appearance_available):
        return None
    return float(sum(min(a, b) for a, b in zip(current.hue_hist, previous.hue_hist)))


def level_score(
    current: BagAppearance, previous: BagAppearance, p: ScoreParams | None = None
) -> float | None:
    """液柱高度一致性：只罚升高。任一侧液位不可用返回 ``None``。"""
    params = p or ScoreParams()
    if not (current.level_available and previous.level_available):
        return None
    # 液柱变**高**才是异常：同一袋只会往下走。本轮比上一轮高 = 换了新袋（或拍到了别的东西）。
    rise = current.liquid_height - previous.liquid_height
    if rise <= params.level_rise_floor:
        return 1.0
    over = rise - params.level_rise_floor
    return max(0.0, 1.0 - over / max(1e-6, params.level_rise_tolerance))


def label_score(
    current: BagAppearance, previous: BagAppearance, p: ScoreParams | None = None
) -> float | None:
    """标签区一致性：有无、面积、质心三项。任一侧不可判返回 ``None``。

    「上一轮有标签、这轮没有」直接判 0：同一袋的标签不会凭空消失（除非袋子被翻面，
    那也正该让人看一眼，不该悄悄接着旧曲线）。
    """
    params = p or ScoreParams()
    if not (current.label_available and previous.label_available):
        return None
    if current.label_present != previous.label_present:
        return 0.0
    coverage_penalty = min(
        1.0, abs(current.label_coverage - previous.label_coverage) / params.label_coverage_tolerance
    )
    shift_penalty = min(
        1.0, abs(current.label_centre_x - previous.label_centre_x) / params.label_shift_tolerance
    )
    return max(0.0, 1.0 - 0.6 * coverage_penalty - 0.4 * shift_penalty)


# ── 候选打分 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CandidateScore:
    """一个历史候选的打分明细（护士端「系统凭什么这么判」就靠它）。"""

    bag_seq: str
    total: float
    geometry: float
    colour: float | None
    level: float | None
    label: float | None
    #: 实际参与加权的项（缺席的证据不在其中）
    terms_used: tuple[str, ...]
    #: 拖分最多的那一项；``None`` 表示没有可用外观项
    driver: str | None
    #: 液柱比上一轮**高**（超容差）→ 物理上讲不通，一票否决（见 ``level_veto``）
    level_vetoed: bool = False
    session_id: str | None = None
    shift_key: str | None = None
    state: str = "monitoring"
    current_mode: str = "full"
    previous_mode: str = "full"

    @property
    def appearance(self) -> float | None:
        """外观合成（颜色/液位/标签加权平均）；无任何外观项时为 ``None``。"""
        parts = [(self.colour, "colour"), (self.level, "level"), (self.label, "label")]
        usable = [(value, name) for value, name in parts if value is not None]
        if not usable:
            return None
        return float(sum(value for value, _ in usable) / len(usable))

    def to_json(self) -> dict[str, Any]:
        return {
            "bag_seq": self.bag_seq,
            "session_id": self.session_id,
            "total": round(self.total, 6),
            "geometry": round(self.geometry, 6),
            "colour": None if self.colour is None else round(self.colour, 6),
            "level": None if self.level is None else round(self.level, 6),
            "label": None if self.label is None else round(self.label, 6),
            "terms_used": list(self.terms_used),
            "driver": self.driver,
            "level_vetoed": self.level_vetoed,
        }

    def explain(self) -> str:
        """一句人话：这一分是被哪一项拖下去的。"""
        if self.driver is None:
            return "位置对不上"
        return f"{self.driver} 项对不上"


def score_candidate(
    current: BagAppearance, previous: BagAppearance, p: ScoreParams | None = None
) -> CandidateScore:
    """给一对「本轮候选 / 历史袋」打分，逐项留痕。"""
    params = p or ScoreParams()
    geometry = geometry_score(current, previous, params)
    colour = colour_score(current, previous)
    level = level_score(current, previous, params)
    label = label_score(current, previous, params)

    weights = {"geometry": params.geometry_weight, "colour": params.colour_weight}
    values: dict[str, float | None] = {"geometry": geometry, "colour": colour}
    if level is not None:
        weights["level"] = params.level_weight
        values["level"] = level
    if label is not None:
        weights["label"] = params.label_weight
        values["label"] = label

    used = tuple(name for name in SCORE_TERMS if values.get(name) is not None)
    total_weight = sum(weights[name] for name in used)
    total = (
        sum(weights[name] * float(values[name]) for name in used) / total_weight
        if total_weight
        else geometry
    )
    # 拖分最多的项：按「权重 × 掉分」比，谁把平均分拉得最狠谁背锅。
    driver = min(
        (name for name in used if name != "geometry"),
        key=lambda name: float(values[name]),
        default=None,
    )
    return CandidateScore(
        bag_seq=previous.bag_seq,
        total=total,
        geometry=geometry,
        colour=colour,
        level=level,
        label=label,
        terms_used=used,
        driver=driver,
        level_vetoed=bool(params.level_veto and level is not None and level == 0.0),
        session_id=previous.session_id,
        shift_key=previous.shift_key,
        state=previous.state,
        current_mode=current.descriptor_mode,
        previous_mode=previous.descriptor_mode,
    )


@dataclass(frozen=True)
class RankedCandidates:
    """一组历史候选的排序结果。``margin`` 是最高分与次高分之差（无次高分时为 None）。"""

    ordered: tuple[CandidateScore, ...]

    @property
    def best(self) -> CandidateScore | None:
        return self.ordered[0] if self.ordered else None

    @property
    def runner_up(self) -> CandidateScore | None:
        return self.ordered[1] if len(self.ordered) > 1 else None

    @property
    def margin(self) -> float | None:
        if len(self.ordered) < 2:
            return None
        return self.ordered[0].total - self.ordered[1].total

    @property
    def appearance_available(self) -> bool:
        """第一名是否具备外观证据（没有就不许自动接上）。"""
        best = self.best
        return best is not None and best.colour is not None


def rank_candidates(
    current: BagAppearance,
    previous: Sequence[BagAppearance],
    params: ScoreParams | None = None,
) -> RankedCandidates:
    """给本轮候选排个序（高分在前）。

    **只排序不判定**：返回谁最像、差多少；「够不够格接上」由
    :func:`identity.reanchor.reanchor_slots` 连同拒绝猜测规则一起判。
    """
    p = params or ScoreParams()
    scored = [score_candidate(current, item, p) for item in previous]
    scored.sort(key=lambda item: (-item.total, item.bag_seq))
    return RankedCandidates(tuple(scored))


def score_pair(
    current: BagAppearance, previous: BagAppearance, params: ScoreParams | None = None
) -> tuple[float, float, float | None]:
    """兼容旧三元组 ``(总分, 几何分, 颜色分)``。

    第三个返回值仍是**颜色分**而不是外观合成：既有调用方
    （:mod:`core.reacquire` 的再获取、几何口径单测）拿它跟「颜色明显不像」的门槛比，
    语义就是颜色。它为 ``None`` 表示**颜色不可用**——调用方据此走
    ``appearance_unavailable``，绝不把它当 0 分（0 是「看着不像」，两者含义不同）。
    """
    scored = score_candidate(current, previous, params)
    return scored.total, scored.geometry, scored.colour