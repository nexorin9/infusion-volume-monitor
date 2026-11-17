"""袋型腔体几何：面积比 → 毫升，以及**用护士确认的换液事件反推自校准项**。

**为什么不能拿面积比直接乘容量。** 几何核量到的是「液体在画面上的投影面积占袋腔
投影面积的比例」。同一个 ``fill_ratio=0.5``，在圆角方袋上离满袋还有小半袋，在扁袋上
几乎就是满袋——因为输液袋的横截面不是一个矩形：袋腔上部收窄（挂耳与封边），软袋
充盈后下部又被顶鼓。这两项都**只改变观感几何、不改变真实体积**，于是「面积比」与
「体积分数」之间差着一段**随袋型而变**的系统偏差（task 8 实测：同一个 5% 液位，
TYPE-D 扁袋比 TYPE-C 硬瓶高出 7.6 个百分点）。把面积比线性乘容量的做法，误差正好
落在这段偏差上，而毫升数是要拿去决定「要不要换袋」的，不能带这种偏差。

**本模块的做法**：按袋型登记一条腔体剖面（超椭圆横截面 + 透视收窄 + 软袋鼓胀），
用它把观测量反解回真实高度比，再按**未变形腔体**积分出体积分数，最后乘有效容量。
反解用「查表 + 二分」而不是拟合——几何关系是解析的，多项式拟合会把袋型差异糊掉。

**残留不是噪声。** 低液位时有两部分液体永远流不出来：袋壁挂液（膜）与出液口/管路/
滴斗残留（常数）。前者随浸润面积增长、后者是常数，量级都在毫升级——放在 500 ml 袋子
上不算大，可它恰好落在「快没了、要不要换袋」这个最要紧的区段里。因此本模块把它们
**显式建模**（``film_holdup_ml`` 与自校准学出来的 ``learned_residual_ml``），可读
体积 = 腔体几何体积 − 残留，读数落到残留量级以下时标 ``residual_limited`` 并给 0，
不给一个「还剩 1.7 ml」这种没人能核对的数字。

**自校准只用护士自己确认过的事实。** 护士点了「已换液」，就等于确认这袋是输完才换的
——那一刻真实可读余量是 0，而系统当时按模型给了一个数。这个差就是该袋型的残留偏差项。
把这类样本按袋型攒起来取中位数（不是平均：护士也常因医嘱变更提前换袋，那是离群点，
中位数不会被它拽走），攒够 ``MIN_DEVIATION_SAMPLES`` 条才允许生效；证据互相打架
（离散度过大）时**不生效**，只把样本摆出来请人复核。

**与标定带的关系**：``models.BagTypeCalibration`` 的分段带是**经验标定**（称重法或
人工登记的样例），本模块是**几何标定**（袋型外形参数）。两者互补：袋型登记时带了
``cavity`` 参数就走腔体模型（``estimation_mode="cavity"``），没带就走原有的分段带
（``estimation_mode="band"``），主路径不因有没有腔体参数而改形状。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .models import NEAR_EMPTY_RATIO, NEAR_FULL_RATIO, OVERFILL_RATIO, clamp

__all__ = [
    "CavitySpec",
    "CavityModel",
    "CavityReading",
    "DeviationObservation",
    "LearnedResidual",
    "learn_residual",
    "deviation_observation_from_session",
    "GRID",
    "MONOTONE_TOL",
    "MIN_DEVIATION_SAMPLES",
    "MAX_RESIDUE_EVIDENCE_RATIO",
    "READABLE_FLOOR_ML",
    "LEARNING_STATES",
]

#: 腔体剖面的高度采样点数。2049 给 1/2048 的高度分辨率——离散误差要远小于
#: 分割本身的测量误差（后者在 1 个百分点量级）。
GRID = 2049

#: 面积比查找表允许的格内微凹上限。超过它说明登记的外形参数本身有问题，
#: 应当当场报错而不是照着反解。
MONOTONE_TOL = 1e-3

#: 横截面超椭圆的肩部位置与半宽（归一化高度）。软袋在袋体中下部最宽，往上经封边
#: 收窄成挂耳——这正是「同一个面积比在不同袋型上代表的量不一样」的来源。
SHOULDER_CENTER = 0.45
SHOULDER_HALF = 0.55

#: 自校准生效所需的最少样本数。少于这个数就只有一个点——那不叫标定，叫巧合。
MIN_DEVIATION_SAMPLES = 3

#: 只有「换液前最后一次观测已经很低」的确认才算是残留的证据。半袋被换掉，原因多半
#: 是医嘱变更或病人要下床，与「这袋流不干净」无关，拿它当证据会把偏差学歪。
MAX_RESIDUE_EVIDENCE_RATIO = 0.30

#: 样本离散度上限（相对有效容量）。超过就判证据互相打架，不自动生效。
MAX_DEVIATION_SPREAD_RATIO = 0.06

#: 低于这个毫升数就不给读数了：这是滴斗里那点液体的量级，报出来只会让人误以为
#: 「还能挤出来几毫升」。
READABLE_FLOOR_ML = 1.0

#: 自校准结论状态。
LEARNING_STATES = (
    "no_evidence",            # 一条可用的证据都没有
    "insufficient_samples",   # 有证据但不够 MIN_DEVIATION_SAMPLES 条
    "conflicting",            # 证据互相打架，不自动生效，交人工复核
    "learned",                # 已生效
)


# ── 腔体几何 ────────────────────────────────────────────────────────────────


def _superellipse_width(h: Sequence[float], exponent: float, center: float, half: float):
    """归一化横截面宽度：超椭圆上半部。袋体中下部接近矩形，肩部收成圆角。"""
    heights = np.asarray(h, dtype=float)
    u = np.clip((heights - center) / half, -1.0, 1.0)
    return np.power(np.maximum(0.0, 1.0 - np.power(np.abs(u), exponent)), 1.0 / exponent)


@dataclass(frozen=True)
class CavitySpec:
    """一个袋型的腔体外形与残留参数（登记一次，之后只读）。

    ``capacity_ml`` 是标签标称容量，``dead_volume_ml`` 是袋腔**之外**的管路与滴斗残留；
    两者之差是有效容量（与 :class:`models.BagTypeCalibration` 同一口径，读数即袋内
    可用液体量，不再二次扣减）。
    """

    bag_type_code: str
    capacity_ml: float
    #: 有效腔高（毫米）：液柱能爬到的最高处与袋底的距离。登记后主要用于核对与复现，
    #: 体积换算按归一化高度做，不依赖绝对毫米数。
    cavity_height_mm: float = 200.0
    dead_volume_ml: float = 0.0
    #: 弯月面挂壁：液面爬在袋壁上，量到的面积比高于真实高度比。
    meniscus_offset_ratio: float = 0.0
    #: 横截面超椭圆指数（越大越接近矩形）。
    shape_exponent: float = 4.0
    #: 肩部中心与半宽（归一化高度）。
    shoulder_center: float = SHOULDER_CENTER
    shoulder_half: float = SHOULDER_HALF
    #: 透视收窄：袋顶比袋底窄的比例（纯光学，不改变真实体积）。
    taper: float = 0.20
    #: 软袋鼓胀：充盈后下部被顶鼓的幅度（观感形变，同样不改变真实体积）。
    bulge: float = 0.055
    #: 袋壁挂液：膜挂住的那部分（可读体积 = 几何体积 − 残留）。
    film_holdup_ml: float = 0.0
    #: 腔体几何的**版本号**：0 = 登记过但没升版本。由标定登记入口分配（首次登记即第 1 版），
    #: 不由本类自己发明——历史 ``cavity_json`` 里没有这个键时读成 0。
    #: 几何换算的毫升数带不确定度，模型误差无法自证时门禁会挡下读数（见
    #: :mod:`calibration.uncertainty` 的 ``geometry_version_missing``），
    #: 补登记一个版本号即恢复。
    geometry_revision: int = 0
    label_zh: str = ""
    registered_by: str = ""
    registered_at: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.bag_type_code:
            raise ValueError("bag_type_code 不能为空")
        if self.capacity_ml <= 0:
            raise ValueError(f"capacity_ml 必须为正，收到 {self.capacity_ml}")
        if self.cavity_height_mm <= 0:
            raise ValueError(f"cavity_height_mm 必须为正，收到 {self.cavity_height_mm}")
        if self.dead_volume_ml < 0 or self.dead_volume_ml >= self.capacity_ml:
            raise ValueError(
                f"dead_volume_ml 必须在 [0, capacity_ml)，收到 {self.dead_volume_ml}"
            )
        if self.shape_exponent <= 1.0:
            raise ValueError(f"shape_exponent 必须大于 1（超椭圆），收到 {self.shape_exponent}")
        if not 0.0 < self.shoulder_half <= 1.0:
            raise ValueError(f"shoulder_half 必须在 (0,1]，收到 {self.shoulder_half}")
        if abs(self.shoulder_center) > 1.0:
            raise ValueError(f"shoulder_center 必须在 [-1,1]，收到 {self.shoulder_center}")
        if not 0.0 <= self.taper <= 0.8:
            raise ValueError(f"taper 必须在 [0,0.8]，收到 {self.taper}")
        if not 0.0 <= self.bulge <= 0.5:
            raise ValueError(f"bulge 必须在 [0,0.5]，收到 {self.bulge}")
        if self.film_holdup_ml < 0:
            raise ValueError(f"film_holdup_ml 不能为负，收到 {self.film_holdup_ml}")
        if self.film_holdup_ml >= self.usable_capacity_ml:
            raise ValueError(
                f"film_holdup_ml={self.film_holdup_ml} 已达到甚至超过有效容量 "
                f"{self.usable_capacity_ml}，这袋就没有可读余量了"
            )
        if not 0.0 <= self.meniscus_offset_ratio < 1.0:
            raise ValueError(
                f"meniscus_offset_ratio 必须在 [0,1)，收到 {self.meniscus_offset_ratio}"
            )

    @property
    def usable_capacity_ml(self) -> float:
        return max(self.capacity_ml - self.dead_volume_ml, 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bag_type_code": self.bag_type_code,
            "capacity_ml": round(self.capacity_ml, 3),
            "cavity_height_mm": round(self.cavity_height_mm, 3),
            "dead_volume_ml": round(self.dead_volume_ml, 3),
            "meniscus_offset_ratio": round(self.meniscus_offset_ratio, 6),
            "shape_exponent": round(self.shape_exponent, 4),
            "shoulder_center": round(self.shoulder_center, 4),
            "shoulder_half": round(self.shoulder_half, 4),
            "taper": round(self.taper, 4),
            "bulge": round(self.bulge, 4),
            "film_holdup_ml": round(self.film_holdup_ml, 3),
            "geometry_revision": int(self.geometry_revision),
            "label_zh": self.label_zh,
            "registered_by": self.registered_by,
            "registered_at": self.registered_at,
            "note": self.note,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CavitySpec":
        return cls(
            bag_type_code=str(raw["bag_type_code"]),
            capacity_ml=float(raw["capacity_ml"]),
            cavity_height_mm=float(raw.get("cavity_height_mm", 200.0)),
            dead_volume_ml=float(raw.get("dead_volume_ml", 0.0)),
            meniscus_offset_ratio=float(raw.get("meniscus_offset_ratio", 0.0)),
            shape_exponent=float(raw.get("shape_exponent", 4.0)),
            shoulder_center=float(raw.get("shoulder_center", SHOULDER_CENTER)),
            shoulder_half=float(raw.get("shoulder_half", SHOULDER_HALF)),
            taper=float(raw.get("taper", 0.20)),
            bulge=float(raw.get("bulge", 0.055)),
            film_holdup_ml=float(raw.get("film_holdup_ml", 0.0)),
            geometry_revision=int(raw.get("geometry_revision", 0)),
            label_zh=str(raw.get("label_zh", "")),
            registered_by=str(raw.get("registered_by", "")),
            registered_at=str(raw.get("registered_at", "")),
            note=str(raw.get("note", "")),
        )

    @classmethod
    def from_json(cls, text: str | None) -> "CavitySpec | None":
        """从标定记录里取出腔体参数；没有登记或内容为空返回 ``None``（走分段带）。"""
        if not text:
            return None
        raw = json.loads(text)
        if not raw:
            return None
        return cls.from_dict(raw)


@dataclass(frozen=True)
class CavityReading:
    """一次腔体换算的读数。门禁不通过时 ``readable_ml`` 为 ``None``（不出数）。"""

    area_ratio: float
    #: 扣掉弯月面挂壁后的有效面积比（反解实际用的自变量）。
    effective_area_ratio: float
    #: 反解出的真实高度比（0=空，1=满）。
    height_fraction: float
    #: 腔体几何体积（未扣残留）。
    cavity_ml: float
    #: 袋壁挂液 + 自校准残留。
    resident_ml: float
    #: 可读体积 = 腔体几何体积 − 残留。
    readable_ml: float | None
    residual_source: str
    residual_limited: bool
    near_empty: bool
    near_full: bool
    overfilled: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "area_ratio": round(self.area_ratio, 6),
            "effective_area_ratio": round(self.effective_area_ratio, 6),
            "height_fraction": round(self.height_fraction, 6),
            "cavity_ml": round(self.cavity_ml, 2),
            "resident_ml": round(self.resident_ml, 2),
            "readable_ml": round(self.readable_ml, 1) if self.readable_ml is not None else None,
            "residual_source": self.residual_source,
            "residual_limited": self.residual_limited,
            "near_empty": self.near_empty,
            "near_full": self.near_full,
            "overfilled": self.overfilled,
            "reason": self.reason,
        }


class CavityModel:
    """按 :class:`CavitySpec` 建的腔体几何。**只读、可缓存**，没有可变状态。

    **为什么不做成逐点积分。** 投影半宽里的鼓胀项 ``1 + bulge·(液位 − 高度)`` 带着液位，
    「对液位再积分一遍」会让每查一个读数都要重算整条剖面。这里把它拆成定长前缀和：

    ``Σ w·taper·(1 + bulge·(L−h)) = A + bulge·(L·B − C)``

    三个前缀和 ``A / B / C`` 构造一次就算好，之后每次查询只是取个下标——一次上传几十
    袋、上百帧都走这条路径，常数时间很重要。
    """

    __slots__ = (
        "spec",
        "_h",
        "_a",
        "_b",
        "_c",
        "_cum_true",
        "_lut_ratio",
        "_lut_level",
    )

    def __init__(self, spec: CavitySpec) -> None:
        self.spec = spec
        heights = (np.arange(GRID) + 0.5) / GRID
        self._h = heights
        width = _superellipse_width(
            heights, spec.shape_exponent, spec.shoulder_center, spec.shoulder_half
        )
        # 透视收窄只随高度变；鼓胀随液位变，拆进前缀和。
        weighted = width * (1.0 + spec.taper * (0.5 - heights))
        zero = np.zeros(1, dtype=float)
        self._a = np.concatenate((zero, np.cumsum(weighted)))
        self._b = np.concatenate((zero, np.cumsum(weighted * heights)))
        self._c = np.concatenate((zero, np.cumsum(weighted * heights * heights)))
        self._cum_true = np.concatenate((zero, np.cumsum(width)))

        ratios = np.array([self._area_ratio_at(float(level)) for level in heights])
        # 单调性是反解成立的前提：面积比不单调，二分就会有多解，拿到哪一解全看数值
        # 噪声。允许极小的格内微凹（离散化 + 鼓胀项的解析余项，量级 ~1e-4），超过
        # 容忍度说明登记的外形参数有问题，当场报错而不是等到读数悄悄跳档。
        if ratios[-1] - ratios[0] <= 0.0 or float(np.min(np.diff(ratios))) < -MONOTONE_TOL:
            raise ValueError(
                "腔体剖面给出的面积比不是单调递增的，反解会多解；请检查 shape_exponent / "
                "taper / bulge 的登记值"
            )
        self._lut_ratio = np.maximum.accumulate(ratios)
        self._lut_level = heights

    # ── 观测量（与真值的定义一致，可直接与几何核读数对拍） ──────────────
    def _cut(self, level: float) -> int:
        """液面以下有多少个剖面采样点（``searchsorted`` 二分定位）。"""
        return int(np.searchsorted(self._h, level, side="left"))

    def _area_ratio_at(self, level: float) -> float:
        cut = self._cut(level)
        bulge = self.spec.bulge
        numerator = self._a[cut] + bulge * (level * self._b[cut] - self._c[cut])
        denominator = self._a[-1] + bulge * (level * self._b[-1] - self._c[-1])
        if denominator <= 0:
            return 0.0
        return float(numerator / denominator)

    def area_ratio_at(self, level: float) -> float:
        """真实液位 ``level`` 对应的理论投影面积比（完美分割器量到的值）。"""
        return self._area_ratio_at(clamp(level, 0.0, 1.0))

    def volume_fraction_at(self, level: float) -> float:
        """真实液位对应的**体积**分数（按未变形腔体积分，不含透视与鼓胀）。"""
        cut = self._cut(clamp(level, 0.0, 1.0))
        total = float(self._cum_true[-1])
        if total <= 0:
            return 0.0
        return float(self._cum_true[cut] / total)

    # ── 反解：观测量 → 真实高度比 ─────────────────────────────────────
    def height_fraction(self, area_ratio: float) -> float:
        """投影面积比 → 真实高度比（查表给初值 + 二分细化）。

        ``area_ratio_at`` 单调，二分收敛唯一解；40 次二分把高度压到 1e-12 量级，
        剩下的误差全来自网格离散（半格 ≈ 1/4096 高度，在 500 ml 袋上不到 0.12 ml）。
        """
        target = clamp(area_ratio, 0.0, 1.0)
        if target <= 0.0:
            return 0.0
        level = float(np.interp(target, self._lut_ratio, self._lut_level))
        low, high = 0.0, 1.0
        for _ in range(40):
            mid = (low + high) / 2.0
            if self._area_ratio_at(mid) < target:
                low = mid
            else:
                high = mid
        return clamp((low + high) / 2.0, 0.0, 1.0)

    # ── 主换算 ─────────────────────────────────────────────────────────
    def volume(
        self, area_ratio: float | None, *, learned_residual_ml: float = 0.0
    ) -> CavityReading:
        """投影面积比 → 可读毫升数（含残留补偿与三类边界处理）。

        三类边界都不「自圆其说」：

        * ``fill_ratio > OVERFILL_RATIO`` —— 几何核把袋外的液体或挂壁高光算进了前景，
          此时 ``readable_ml`` 出 ``None`` 并 ``overfilled=True``，宁可让护士重拍；
        * 高度反解为 0（画面里几乎没有液体）—— 报 0 毫升并标 ``near_empty``；
        * 可读体积落到残留量级以下 —— 报 0 毫升并标 ``residual_limited``，不报
          「还剩 1.7 ml」这种谁也核对不了的数。
        """
        spec = self.spec
        if area_ratio is None:
            return CavityReading(
                area_ratio=0.0,
                effective_area_ratio=0.0,
                height_fraction=0.0,
                cavity_ml=0.0,
                resident_ml=round(learned_residual_ml, 2),
                readable_ml=None,
                residual_source="self_calibrated" if learned_residual_ml else "none",
                residual_limited=True,
                near_empty=True,
                near_full=False,
                overfilled=False,
                reason="no_observation",
            )

        raw = float(area_ratio)
        if raw > OVERFILL_RATIO:
            return CavityReading(
                area_ratio=round(raw, 6),
                effective_area_ratio=0.0,
                height_fraction=1.0,
                cavity_ml=spec.usable_capacity_ml,
                resident_ml=round(learned_residual_ml, 2),
                readable_ml=None,
                residual_source="self_calibrated" if learned_residual_ml else "none",
                residual_limited=False,
                near_empty=False,
                near_full=True,
                overfilled=True,
                reason=(
                    f"量到的面积比 {raw:.3f} 超过 {OVERFILL_RATIO}（袋腔装不下这么多），"
                    "不给毫升数；请把镜头对准袋体中部重拍一张，避开袋外反光"
                ),
            )

        effective = clamp(raw - spec.meniscus_offset_ratio, 0.0, 1.0)
        level = self.height_fraction(effective)
        volume_fraction = self.volume_fraction_at(level)
        cavity_ml = volume_fraction * spec.usable_capacity_ml
        # 残留 = 袋壁挂液（随浸润面积增长）+ 自校准学出来的常数残留。
        resident_ml = spec.film_holdup_ml * volume_fraction + max(learned_residual_ml, 0.0)
        readable = cavity_ml - resident_ml
        residual_source = (
            "self_calibrated" if learned_residual_ml > 0 else
            ("registered_film" if spec.film_holdup_ml > 0 else "none")
        )

        if level <= 0.0:
            return CavityReading(
                area_ratio=round(raw, 6),
                effective_area_ratio=round(effective, 6),
                height_fraction=0.0,
                cavity_ml=0.0,
                resident_ml=round(resident_ml, 2),
                readable_ml=0.0,
                residual_source=residual_source,
                residual_limited=True,
                near_empty=True,
                near_full=False,
                overfilled=False,
                reason="cavity_empty",
            )

        if readable <= READABLE_FLOOR_ML:
            return CavityReading(
                area_ratio=round(raw, 6),
                effective_area_ratio=round(effective, 6),
                height_fraction=round(level, 6),
                cavity_ml=round(cavity_ml, 2),
                resident_ml=round(resident_ml, 2),
                readable_ml=0.0,
                residual_source=residual_source,
                residual_limited=True,
                near_empty=True,
                near_full=False,
                overfilled=False,
                reason=(
                    f"腔体里还看得到液体，但都在残留量级（约 {resident_ml:.1f} ml）以内，"
                    "按 0 毫升报；这袋可以按输完处理"
                ),
            )

        return CavityReading(
            area_ratio=round(raw, 6),
            effective_area_ratio=round(effective, 6),
            height_fraction=round(level, 6),
            cavity_ml=round(cavity_ml, 2),
            resident_ml=round(resident_ml, 2),
            readable_ml=round(readable, 1),
            residual_source=residual_source,
            residual_limited=False,
            near_empty=effective <= NEAR_EMPTY_RATIO,
            near_full=effective >= NEAR_FULL_RATIO,
            overfilled=False,
            reason="cavity_model",
        )


# ── 自校准：用护士确认的换液事件反推残留偏差项 ──────────────────────────────


@dataclass(frozen=True)
class DeviationObservation:
    """一条「护士确认已换液」带来的残留证据。"""

    bag_type_code: str
    #: 确认前最后一次观测到的面积比。
    last_fill_ratio: float | None
    #: 模型当时给出的毫升数——按「换液即输完」它应当是 0，两者之差就是残留。
    model_volume_ml: float | None
    session_id: str = ""
    bag_seq: str = ""
    rack_id: str = ""
    bed_no: str = ""
    calibration_version: int | None = None
    source: str = "confirmed_replacement"
    occurred_at: str = ""

    @property
    def eligible(self) -> bool:
        """这条证据能不能拿来学残留。

        需要同时满足：换液前已经看得很低（否则多半是医嘱变更导致的提前换袋）、
        模型当时确实给了一个正数（否则没有偏差可学）。
        """
        return (
            self.last_fill_ratio is not None
            and self.model_volume_ml is not None
            and 0.0 <= float(self.last_fill_ratio) <= MAX_RESIDUE_EVIDENCE_RATIO
            and float(self.model_volume_ml) > 0.0
        )

    @property
    def residual_ml(self) -> float:
        """这条证据反推出的残留偏差（真实可读余量按 0 算）。"""
        return max(float(self.model_volume_ml or 0.0), 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bag_type_code": self.bag_type_code,
            "session_id": self.session_id,
            "bag_seq": self.bag_seq,
            "rack_id": self.rack_id,
            "bed_no": self.bed_no,
            "last_fill_ratio": (
                round(float(self.last_fill_ratio), 6)
                if self.last_fill_ratio is not None
                else None
            ),
            "model_volume_ml": (
                round(float(self.model_volume_ml), 3)
                if self.model_volume_ml is not None
                else None
            ),
            "residual_ml": round(self.residual_ml, 3),
            "eligible": self.eligible,
            "calibration_version": self.calibration_version,
            "source": self.source,
            "occurred_at": self.occurred_at,
        }


@dataclass(frozen=True)
class LearnedResidual:
    """自校准结论：残留偏差项、样本数，以及**是否允许生效**。"""

    residual_ml: float = 0.0
    sample_count: int = 0
    skipped_count: int = 0
    spread_ml: float = 0.0
    state: str = "no_evidence"
    applied: bool = False
    reason: str = ""

    @property
    def effective_residual_ml(self) -> float:
        """真正参与换算的残留值；证据不足或互相打架时按 0（照常出数，只是不修正）。"""
        return self.residual_ml if self.applied else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "residual_ml": round(self.residual_ml, 3),
            "effective_residual_ml": round(self.effective_residual_ml, 3),
            "sample_count": self.sample_count,
            "skipped_count": self.skipped_count,
            "spread_ml": round(self.spread_ml, 3),
            "state": self.state,
            "applied": self.applied,
            "reason": self.reason,
        }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def learn_residual(
    observations: Iterable[DeviationObservation],
    *,
    usable_capacity_ml: float,
    min_samples: int = MIN_DEVIATION_SAMPLES,
) -> LearnedResidual:
    """由确认换液的证据反推该袋型的残留偏差项。

    口径与理由：

    * **只取中位数**：护士会因医嘱变更、病人下床等原因提前换袋，那条样本是离群点，
      平均会被拽走，中位数不会；
    * **样本不足不生效**：一条样本只能说明「这一袋是这样」，不能说明「这个袋型是这样」；
    * **证据打架不生效**：离散度超过 ``MAX_DEVIATION_SPREAD_RATIO`` 说明我们对这袋型
      的理解有问题，此时报出中位数供人复核，但不自动改所有读数。
    """
    rows = list(observations)
    eligible = [row for row in rows if row.eligible]
    skipped = len(rows) - len(eligible)
    if not eligible:
        return LearnedResidual(
            sample_count=0,
            skipped_count=skipped,
            state="no_evidence",
            reason="还没有「换液前已经很低」的确认记录，无法反推残留",
        )
    values = [row.residual_ml for row in eligible]
    median = _median(values)
    spread = max(values) - min(values)
    if len(eligible) < min_samples:
        return LearnedResidual(
            residual_ml=round(median, 3),
            sample_count=len(eligible),
            skipped_count=skipped,
            spread_ml=round(spread, 3),
            state="insufficient_samples",
            applied=False,
            reason=(
                f"只有 {len(eligible)} 条可用确认（需 {min_samples} 条），"
                "先记下偏差，不改读数"
            ),
        )
    limit = max(usable_capacity_ml * MAX_DEVIATION_SPREAD_RATIO, 1.0)
    if spread > limit:
        return LearnedResidual(
            residual_ml=round(median, 3),
            sample_count=len(eligible),
            skipped_count=skipped,
            spread_ml=round(spread, 3),
            state="conflicting",
            applied=False,
            reason=(
                f"{len(eligible)} 条确认反推出的残差从 {min(values):.1f} ml 到 {max(values):.1f} ml，"
                f"跨度 {spread:.1f} ml 超过上限 {limit:.1f} ml，证据互相打架；"
                "请复核该袋型的腔体参数与换液确认记录后再定"
            ),
        )
    return LearnedResidual(
        residual_ml=round(median, 3),
        sample_count=len(eligible),
        skipped_count=skipped,
        spread_ml=round(spread, 3),
        state="learned",
        applied=True,
        reason=f"按 {len(eligible)} 条换液确认的中位数取残留偏差",
    )


def deviation_observation_from_session(
    session: Mapping[str, Any], *, source: str = "confirmed_replacement"
) -> DeviationObservation | None:
    """由一条会话行构造残留证据；**缺袋型或缺读数就返回 None**（不编造）。

    袋型与读数都由上传时的袋位声明给出，缺任何一个都说明这条会话没有走到体积换算，
    拿它当残留证据只会往标定里灌噪声。
    """
    bag_type_code = str(session.get("bag_type_code") or "").strip()
    if not bag_type_code:
        return None
    fill_ratio = session.get("fill_ratio")
    volume_ml = session.get("estimated_volume_ml")
    return DeviationObservation(
        bag_type_code=bag_type_code,
        last_fill_ratio=None if fill_ratio is None else float(fill_ratio),
        model_volume_ml=None if volume_ml is None else float(volume_ml),
        session_id=str(session.get("session_id") or ""),
        bag_seq=str(session.get("bag_seq") or ""),
        rack_id=str(session.get("rack_id") or ""),
        bed_no=str(session.get("bed_no") or ""),
        calibration_version=session.get("calibration_version"),
        source=source,
        occurred_at=str(session.get("state_changed_at") or session.get("captured_at") or ""),
    )