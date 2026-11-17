"""袋型标定记录：面积比 → 毫升的换算关系，以及它的版本与出处。

**为什么必须有标定记录**：几何核给出的是「液体占袋腔的面积比」。面积比本身换算不出
毫升——同一袋型不同批次的袋腔形状、弯月面挂壁、管路与滴斗残留都不一样。本模块把这层
关系做成**可登记、可核对、可版本化**的标定记录：

* ``CalibrationSample`` 一条标定样例 = 「量出来的面积比」+「已知的真实毫升数」
  （称重法、或护士确认换液后按「这袋输完了」反推）；
* ``fit_bands`` 把样例拟合成一条**分段线性**曲线——输液袋上那几道刻度带本身就是分段
  线性的，标定做的事就是把带子的 ml 数登记下来，不是硬套一个线性公式；
* ``BagTypeCalibration.interpolate_ml`` 由面积比反查毫升，含弯月面挂壁补偿
  （``meniscus_offset_ratio``：液面爬在袋壁上，量到的面积比高于真实高度比）；
* 死角残留 ``dead_volume_ml`` 只体现在 ``usable_capacity_ml``（= 标签容量 − 残留）上，
  标定带子里的毫升数就是**袋内**液体量，不再二次扣减。

**没有标定记录就不出毫升数**：调用方拿不到本模块的记录时应走 ``uncalibrated`` 分支，
响应里 ``estimated_volume_ml`` 字段整体不出现，而不是给一个假数字（见 estimator.py）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # cavity 反过来 import 本模块的常量，运行时再 import 会成环
    from .cavity import CavitySpec

__all__ = [
    "CalibrationSample",
    "BandFit",
    "BagTypeCalibration",
    "VolumeReading",
    "fit_bands",
    "clamp",
    "MIN_SAMPLES_FOR_FIT",
]

#: 少于这个数目的样例不足以拟合分段带，按「线性 + 满容量」兜底。
MIN_SAMPLES_FOR_FIT = 2

#: 面积比判为「快满了 / 快空了」的阈值，供护士端文案与自校准取材用。
NEAR_EMPTY_RATIO = 0.06
NEAR_FULL_RATIO = 0.94
#: 面积比超过该值即视为超装（几何核把袋外液体也算了进去）。
OVERFILL_RATIO = 1.02


def clamp(value: float, low: float, high: float) -> float:
    """把 ``value`` 夹到 ``[low, high]``；NaN 归到 ``low``（不外推）。"""
    if value is None or math.isnan(value):
        return low
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class CalibrationSample:
    """一条标定样例：量到的面积比 与 已知毫升数 成对。"""

    measured_fill_ratio: float
    known_volume_ml: float
    source: str = "unspecified"
    captured_at: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.measured_fill_ratio <= 1.0:
            raise ValueError(
                f"measured_fill_ratio 必须在 [0,1]，收到 {self.measured_fill_ratio}"
            )
        if self.known_volume_ml < 0:
            raise ValueError(f"known_volume_ml 不能为负，收到 {self.known_volume_ml}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "measured_fill_ratio": round(self.measured_fill_ratio, 6),
            "known_volume_ml": round(self.known_volume_ml, 3),
            "source": self.source,
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CalibrationSample":
        return cls(
            measured_fill_ratio=float(raw["measured_fill_ratio"]),
            known_volume_ml=float(raw["known_volume_ml"]),
            source=str(raw.get("source") or "unspecified"),
            captured_at=raw.get("captured_at"),
        )


@dataclass(frozen=True)
class BandFit:
    """分段线性带的拟合结果。"""

    bands: tuple[tuple[float, float], ...]
    max_relative_error: float
    mean_relative_error: float
    sample_count: int
    extrapolated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "bands": [[round(x, 6), round(y, 3)] for x, y in self.bands],
            "max_relative_error": round(self.max_relative_error, 6),
            "mean_relative_error": round(self.mean_relative_error, 6),
            "sample_count": self.sample_count,
            "extrapolated": self.extrapolated,
        }


def _dedupe_samples(
    samples: Iterable[CalibrationSample],
) -> tuple[tuple[float, float], ...]:
    """按面积比去重；同一面积比取毫升数中位数（量多次时取中位，抗单次读数毛刺）。"""
    grouped: dict[float, list[float]] = {}
    for sample in samples:
        ratio = round(clamp(sample.measured_fill_ratio, 0.0, 1.0), 6)
        grouped.setdefault(ratio, []).append(float(sample.known_volume_ml))
    knots: list[tuple[float, float]] = []
    for ratio in sorted(grouped):
        values = sorted(grouped[ratio])
        middle = len(values) // 2
        median = (
            values[middle]
            if len(values) % 2 == 1
            else (values[middle - 1] + values[middle]) / 2.0
        )
        knots.append((ratio, median))
    return tuple(knots)


def fit_bands(
    samples: Sequence[CalibrationSample],
    usable_capacity_ml: float,
    *,
    tolerance: float = 0.02,
) -> BandFit:
    """由标定样例拟合分段线性带 ``(面积比, 袋内毫升)``。

    拟合口径：

    * 两端补齐 —— 起点补 ``(0, 0)``、终点补 ``(1, usable_capacity_ml)``，保证
      ``[0,1]`` 上处处可查，不留「样例没覆盖就不给数」的空洞；
    * 面积比相同的多条样例取中位数（``_dedupe_samples``）；
    * 相邻节点间的残差超过 ``tolerance`` 时**不静默拉直**，而是把偏差如实记进
      ``max_relative_error`` 交给标定人复核——袋腔形状不规则是真实存在的，宁可让
      偏差可见，也不要拟合出一条看起来很漂亮但对不上刻度的曲线。
    """
    knots = _dedupe_samples(samples)
    extrapolated = False
    if not knots:
        return BandFit(
            bands=((0.0, 0.0), (1.0, usable_capacity_ml)),
            max_relative_error=1.0,
            mean_relative_error=1.0,
            sample_count=0,
            extrapolated=True,
        )

    if knots[0][0] > 0.0:
        knots = ((0.0, 0.0),) + knots
        extrapolated = True
    if knots[-1][0] < 1.0:
        knots = knots + ((1.0, usable_capacity_ml),)
        extrapolated = True

    errors: list[float] = []
    scale = max(usable_capacity_ml, 1.0)
    for ratio, known in knots:
        if ratio <= 0.0:
            continue
        errors.append(abs(_interpolate(knots, ratio) - known) / scale)
    for sample in samples:
        predicted = _interpolate(knots, clamp(sample.measured_fill_ratio, 0.0, 1.0))
        errors.append(
            abs(predicted - sample.known_volume_ml) / max(sample.known_volume_ml, 1.0)
        )

    return BandFit(
        bands=knots,
        max_relative_error=max(errors) if errors else 0.0,
        mean_relative_error=(sum(errors) / len(errors)) if errors else 0.0,
        sample_count=len(samples),
        extrapolated=extrapolated,
    )


def _interpolate(knots: Sequence[tuple[float, float]], ratio: float) -> float:
    """在节点间线性插值（两端按端点斜率外推并夹到容量上限）。"""
    if not knots:
        return 0.0
    if ratio <= knots[0][0]:
        x0, y0 = knots[0]
        x1, y1 = knots[1] if len(knots) > 1 else knots[0]
        slope = (y1 - y0) / (x1 - x0) if x1 > x0 else 0.0
        return max(0.0, y0 + slope * (ratio - x0))
    if ratio >= knots[-1][0]:
        x0, y0 = knots[-2] if len(knots) > 1 else knots[-1]
        x1, y1 = knots[-1]
        slope = (y1 - y0) / (x1 - x0) if x1 > x0 else 0.0
        return y1 + slope * (ratio - x1)
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x0 <= ratio <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * ((ratio - x0) / (x1 - x0))
    return knots[-1][1]


@dataclass(frozen=True)
class VolumeReading:
    """一次换算的读数：毫升数 + 换算经过哪一段带 + 边界提示。"""

    volume_ml: float
    #: 扣掉弯月面挂壁后的有效面积比（换算实际用的自变量）。
    effective_fill_ratio: float
    #: 落在哪一段带（带序号从 0 起），护士端可据此提示「按第 2 格刻度估的」。
    band_index: int
    near_empty: bool
    near_full: bool
    overfilled: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "volume_ml": round(self.volume_ml, 2),
            "effective_fill_ratio": round(self.effective_fill_ratio, 6),
            "band_index": self.band_index,
            "near_empty": self.near_empty,
            "near_full": self.near_full,
            "overfilled": self.overfilled,
        }


@dataclass(frozen=True)
class BagTypeCalibration:
    """一个袋型在某版本下的标定记录。**同袋型同时只有一个生效版本。**"""

    bag_type_code: str
    version: int
    capacity_ml: float
    usable_capacity_ml: float
    dead_volume_ml: float
    meniscus_offset_ratio: float
    bands: tuple[tuple[float, float], ...]
    samples: tuple[CalibrationSample, ...] = ()
    max_relative_error: float = 0.0
    calibrated_by: str = ""
    calibrated_at: str = ""
    note: str = ""
    calibration_id: str = ""
    active: bool = True
    #: 生效区间（左闭右开）。留空时用 :func:`calibration.versions.window_of` 的
    #: 兜底口径（from=calibrated_at、to=无），老库升级后行为不变。
    effective_from: str | None = None
    effective_to: str | None = None
    #: 腔体外形参数（可选）。登记了就走腔体几何换算（``estimation_mode="cavity"``），
    #: 没登记就走分段带（``"band"``）——两条路互不依赖，见 :mod:`calibration.cavity`。
    cavity: "CavitySpec | None" = None

    # ── 构造 ──────────────────────────────────────────────────────────────
    @classmethod
    def from_samples(
        cls,
        bag_type_code: str,
        capacity_ml: float,
        samples: Sequence[CalibrationSample],
        *,
        dead_volume_ml: float = 0.0,
        meniscus_offset_ratio: float = 0.0,
        calibrated_by: str = "",
        calibrated_at: str = "",
        note: str = "",
        version: int = 1,
        calibration_id: str = "",
        cavity: "CavitySpec | None" = None,
        effective_from: str | None = None,
        effective_to: str | None = None,
    ) -> "BagTypeCalibration":
        """由样例拟合出一条标定记录（``version`` 由仓储层分配，不在这里定）。"""
        usable = max(capacity_ml - dead_volume_ml, 0.0)
        fit = fit_bands(samples, usable)
        return cls(
            bag_type_code=bag_type_code,
            version=version,
            capacity_ml=capacity_ml,
            usable_capacity_ml=usable,
            dead_volume_ml=dead_volume_ml,
            meniscus_offset_ratio=meniscus_offset_ratio,
            bands=fit.bands,
            samples=tuple(samples),
            max_relative_error=fit.max_relative_error,
            calibrated_by=calibrated_by,
            calibrated_at=calibrated_at,
            note=note,
            calibration_id=calibration_id,
            cavity=cavity,
            effective_from=effective_from,
            effective_to=effective_to,
        )

    # ── 换算 ──────────────────────────────────────────────────────────────
    def estimated_volume_ml(self, fill_ratio: float | None) -> VolumeReading:
        """面积比 → 袋内毫升数。弯月面补偿后按分段带插值。"""
        if fill_ratio is None:
            return VolumeReading(
                volume_ml=0.0,
                effective_fill_ratio=0.0,
                band_index=0,
                near_empty=True,
                near_full=False,
                overfilled=False,
            )
        raw = float(fill_ratio)
        effective = clamp(raw - self.meniscus_offset_ratio, 0.0, 1.0)
        # 毫升数是给护士看的读数，保留 0.1 ml 就够；不出 14 位小数的假精度
        volume = round(
            clamp(_interpolate(self.bands, effective), 0.0, self.usable_capacity_ml), 1
        )
        return VolumeReading(
            volume_ml=volume,
            effective_fill_ratio=effective,
            band_index=self.band_index_of(effective),
            near_empty=effective <= NEAR_EMPTY_RATIO,
            near_full=effective >= NEAR_FULL_RATIO,
            overfilled=raw > OVERFILL_RATIO,
        )

    def band_index_of(self, fill_ratio: float) -> int:
        """面积比落在第几段带（从 0 起）。"""
        for index in range(len(self.bands) - 1, -1, -1):
            if fill_ratio >= self.bands[index][0]:
                return index
        return 0

    def describe(self) -> dict[str, Any]:
        """给 API 与审计用的快照（不含任何凭据）。"""
        return {
            "calibration_id": self.calibration_id,
            "bag_type_code": self.bag_type_code,
            "version": self.version,
            "capacity_ml": round(self.capacity_ml, 3),
            "usable_capacity_ml": round(self.usable_capacity_ml, 3),
            "dead_volume_ml": round(self.dead_volume_ml, 3),
            "meniscus_offset_ratio": round(self.meniscus_offset_ratio, 6),
            "bands": [[round(x, 6), round(y, 3)] for x, y in self.bands],
            "cavity": self.cavity.as_dict() if self.cavity else None,
            "sample_count": len(self.samples),
            "max_relative_error": round(self.max_relative_error, 6),
            "calibrated_by": self.calibrated_by,
            "calibrated_at": self.calibrated_at,
            "note": self.note,
            "active": self.active,
            "effective_from": self.effective_from or self.calibrated_at or None,
            "effective_to": self.effective_to,
        }

    def to_row(self, *, created_at: str, active: bool) -> dict[str, Any]:
        """落 ``bag_type_calibration`` 表的一行。"""
        import json

        return {
            "calibration_id": self.calibration_id,
            "bag_type_code": self.bag_type_code,
            "version": self.version,
            "capacity_ml": self.capacity_ml,
            "usable_capacity_ml": self.usable_capacity_ml,
            "dead_volume_ml": self.dead_volume_ml,
            "meniscus_offset_ratio": self.meniscus_offset_ratio,
            "bands_json": json.dumps(
                [[x, y] for x, y in self.bands], separators=(",", ":")
            ),
            "cavity_json": self.cavity.to_json() if self.cavity else None,
            "samples_json": json.dumps(
                [sample.as_dict() for sample in self.samples],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "max_relative_error": self.max_relative_error,
            "calibrated_by": self.calibrated_by,
            "calibrated_at": self.calibrated_at,
            "active": 1 if active else 0,
            "effective_from": self.effective_from or self.calibrated_at,
            "effective_to": self.effective_to,
            "note": self.note,
            "created_at": created_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "BagTypeCalibration":
        """从数据库行还原标定记录。

        ``cavity_json`` 是后加的列：旧库里可能整列不存在，取不到就当没登记腔体参数
        （走分段带），不因此报错——升级不能让已经在跑的服务起不来。
        """
        import json

        from .cavity import CavitySpec

        bands = tuple(
            (float(item[0]), float(item[1])) for item in json.loads(row["bands_json"])
        )
        samples = tuple(
            CalibrationSample.from_dict(item) for item in json.loads(row["samples_json"])
        )
        try:
            raw_cavity = row["cavity_json"]
        except (IndexError, KeyError):
            raw_cavity = None
        cavity: CavitySpec | None = None
        if raw_cavity:
            try:
                cavity = CavitySpec.from_json(str(raw_cavity))
            except (json.JSONDecodeError, KeyError, ValueError):
                # 登记时的参数不合法也不该让整条标定读不出来：退到分段带。
                cavity = None
        return cls(
            calibration_id=row["calibration_id"],
            bag_type_code=row["bag_type_code"],
            version=int(row["version"]),
            capacity_ml=float(row["capacity_ml"]),
            usable_capacity_ml=float(row["usable_capacity_ml"]),
            dead_volume_ml=float(row["dead_volume_ml"]),
            meniscus_offset_ratio=float(row["meniscus_offset_ratio"]),
            bands=bands,
            samples=samples,
            max_relative_error=float(row["max_relative_error"]),
            calibrated_by=row["calibrated_by"],
            calibrated_at=row["calibrated_at"],
            note=row["note"] or "",
            active=bool(row["active"]),
            cavity=cavity,
            effective_from=_optional(row, "effective_from") or row["calibrated_at"],
            effective_to=_optional(row, "effective_to"),
        )


def _optional(row: Any, key: str) -> str | None:
    """取一列；老库缺列时返回 ``None``（升级不该让已在跑的服务起不来）。"""
    try:
        value = row[key]
    except (IndexError, KeyError):
        return None
    return str(value) if value else None


@dataclass
class BagTypeCalibrationBundle:
    """仓储层返回的标定记录集合：生效版本 + 该袋型的历史版本。"""

    active: BagTypeCalibration | None = None
    history: list[BagTypeCalibration] = field(default_factory=list)

    @property
    def versions(self) -> list[int]:
        return [record.version for record in self.history]