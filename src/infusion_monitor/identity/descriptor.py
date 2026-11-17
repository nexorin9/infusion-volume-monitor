"""外观描述子：这一轮拍到的这只袋子，长什么样——颜色、液位、标签区。

**这一层要回答的问题**：下一轮巡视时，手机里这两只袋子里，哪一只还是刚才那一只？
判据不是「挂点序号」（床位会挪、袋序会重排），也不是「掩膜框高度」（候选框是液体
区域，液面一降框就变矮，等于把「液位低」误判成「不是同一袋」），而是三组**液面下降
不该改变、换袋一定会改变**的特征：

1. **颜色分布**——候选区域内的 HSV 色相直方图（12 bin）。同一袋的液体颜色跨轮基本
   不变；换袋（不同药品 / 不同厂家）色相分布明显不同。
2. **液位特征**——弯月面（``liquid_top_y``）与液柱高度（``liquid_height``）。**只惩罚
   液柱变高、不惩罚变低**：同一袋只会往下走，上去只有一种解释——换了新袋。反过来
   「掉得多」是正常的（两小时没巡视），拿它扣分等于惩罚正常业务。
3. **标签区存在性**——袋体上沿的观察带里有没有印刷笔画（标签贴纸）。**不依赖 OCR**：
   只统计「亮膜区里偏暗的笔画占比」与标签区质心，不去认字。认字是 OCR 的活，本项目
   不引这个依赖；判据只要「同一袋跨轮稳定、换袋就会变」就够了。

**降级口径**（``descriptor_mode``，响应与审计都带）：

* ``full``——颜色 + 液位 + 标签区都拿得到，打分用三项；
* ``color_only``——区域够大但**看不出来有没有标签**（整段太暗 / 观察带出画 / 全被
  挡住），降级为「几何 + 颜色 + 液位」，标签项退出加权并在结果里如实标出；
* ``geometry_only``——区域小到像素不足，颜色与液位都不可信：``appearance_available``
  为假，上游一律**不自动接上**（宁可让护士点一行，也不接错袋）。

**为什么标签区只统计「有没有、不认字」**：护士换袋时经常是「同一家、同一规格、只是袋
数不同」，颜色和液位都可能看不出差别；反过来不同厂家的同色药袋，标签贴纸的印刷笔画
位置与覆盖面积往往不同。判据够用、可解释、不引新依赖，比「认得出标签上的字」更划算。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = [
    "DESCRIPTOR_MODES",
    "HUE_BINS",
    "LABEL_BAND_FRACTION",
    "MIN_APPEARANCE_PIXELS",
    "MIN_LABEL_INK",
    "BagAppearance",
    "LabelFeatures",
    "descriptor_from_session",
    "describe_region",
    "extract_appearance",
]

#: 色相直方图 bin 数（OpenCV HSV 的 H 取 0–179）。
HUE_BINS = 12
#: 低于此像素数不取直方图：几十个像素的直方图是噪声，不是外观。
MIN_APPEARANCE_PIXELS = 200

#: 标签观察带高度 = 液柱高度 × 该系数（观察带取在弯月面**上方**的袋体空膜区）。
#: 取**偏大**的值是故意的：带子开得高一点，越过袋子上沿的那几行会被下面的
#: ``ROW_FILM_COVERAGE`` 逐行判掉。液柱快见底时带子跟着变矮，标签贴在袋体中上部就会
#: 整块落在带外——那种场景只能拿「液柱高度」去定位观察带，那就完全看不全袋子了。
LABEL_BAND_FRACTION = 2.0
#: 观察带像素数下限：太窄的带子判不出「有没有印刷笔画」，如实标成不可用。
MIN_LABEL_BAND_PIXELS = 120
#: 观察带整体明度下限（0–255）：整段都黑（夜间未开灯 / 袋体被手挡）→ 标签不可判。
MIN_LABEL_BAND_VALUE = 40.0
#: 笔画占比达到该值才算「有标签区」（0–1）。
MIN_LABEL_INK = 0.03
#: 相对明度阈值：亮膜（标签/空膜）与深色笔画的分界，都按本观察带的 90 分位归一，
#: 免得换个拍摄亮度整套判据就失效。
FILM_VALUE_RATIO = 0.60
INK_VALUE_RATIO = 0.38

#: 描述子档位（响应与审计里都带，护士端据此提示「这次没看标签区」）。
DESCRIPTOR_MODES = ("full", "color_only", "geometry_only")


@dataclass(frozen=True)
class LabelFeatures:
    """标签区观测结果。``available=False`` 时其余字段一律为默认值，不参与打分。"""

    available: bool
    present: bool
    #: 亮膜（袋体/标签贴纸）占观察带的比例
    coverage: float
    #: 印刷笔画占观察带的比例（不认字，只统计深色细笔画）
    ink: float
    centre_x: float
    centre_y: float

    def to_json(self) -> dict[str, Any]:
        return {
            "label_available": self.available,
            "label_present": self.present,
            "label_coverage": round(self.coverage, 6),
            "label_ink": round(self.ink, 6),
            "label_centre_x": round(self.centre_x, 6),
            "label_centre_y": round(self.centre_y, 6),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any] | None) -> "LabelFeatures":
        """从落库形态还原。旧行没有这些键 → **一律按不可用**处理（不猜标签在不在）。

        降级方向只有一个：宁可少用一项判据，也绝不把「没存」当成「没有标签」。
        """
        if not payload:
            return _LABEL_UNAVAILABLE
        available = bool(payload.get("label_available", False))
        if not available:
            return _LABEL_UNAVAILABLE
        return cls(
            available=True,
            present=bool(payload.get("label_present", False)),
            coverage=float(payload.get("label_coverage", 0.0)),
            ink=float(payload.get("label_ink", 0.0)),
            centre_x=float(payload.get("label_centre_x", 0.0)),
            centre_y=float(payload.get("label_centre_y", 0.0)),
        )


_LABEL_UNAVAILABLE = LabelFeatures(False, False, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class BagAppearance:
    """一只袋子的可对外描述子（**不含任何患者明细**，只有位置、颜色与液位统计）。"""

    bag_seq: str
    slot_index: int
    #: 横向中心（占画面宽度的比例）
    centre_x: float
    #: 袋底纵坐标（占画面高度的比例）——液面下降时不动
    bottom_y: float
    #: 袋体横向宽度（占画面宽度的比例）
    width: float
    #: 归一化色相直方图；为空表示外观不可用
    hue_hist: tuple[float, ...] = ()
    mean_saturation: float = 0.0
    mean_value: float = 0.0
    session_id: str | None = None
    shift_key: str | None = None
    state: str = "monitoring"
    # ── 液位特征（本任务新增）─────────────────────────────────────────────
    #: 弯月面纵坐标（占画面高度的比例）；掩膜上沿
    liquid_top_y: float = 0.0
    #: 液柱高度（占画面高度的比例）= 袋底到弯月面；同一袋只减不增
    liquid_height: float = 0.0
    #: 液体面积 / 掩膜外接框面积（≈1；软袋鼓胀 / 折皱 / 遮挡时会明显小于 1）
    liquid_fill_fraction: float = 0.0
    # ── 标签区（本任务新增）───────────────────────────────────────────────
    label_available: bool = False
    label_present: bool = False
    label_coverage: float = 0.0
    label_ink: float = 0.0
    label_centre_x: float = 0.0
    label_centre_y: float = 0.0

    @property
    def appearance_available(self) -> bool:
        """颜色是否可用（区域像素够不够）。不可用时上游一律不自动接上。"""
        return len(self.hue_hist) == HUE_BINS

    @property
    def level_available(self) -> bool:
        """液位特征是否可用（掩膜非空且有高度）。"""
        return self.liquid_height > 0.0

    @property
    def descriptor_mode(self) -> str:
        """描述子档位：``full`` / ``color_only`` / ``geometry_only``。"""
        if not self.appearance_available:
            return "geometry_only"
        return "full" if self.label_available else "color_only"

    def to_json(self) -> dict[str, Any]:
        """落库形态（JSON 可序列化）。字段是**超集**：旧读者只认前 8 个键仍能读。"""
        payload: dict[str, Any] = {
            "bag_seq": self.bag_seq,
            "slot_index": self.slot_index,
            "centre_x": round(self.centre_x, 6),
            "bottom_y": round(self.bottom_y, 6),
            "width": round(self.width, 6),
            "hue_hist": [round(v, 6) for v in self.hue_hist],
            "mean_saturation": round(self.mean_saturation, 6),
            "mean_value": round(self.mean_value, 6),
            "liquid_top_y": round(self.liquid_top_y, 6),
            "liquid_height": round(self.liquid_height, 6),
            "liquid_fill_fraction": round(self.liquid_fill_fraction, 6),
            "descriptor_mode": self.descriptor_mode,
            "state": self.state,
        }
        payload.update(self.label_features().to_json())
        return payload

    def label_features(self) -> LabelFeatures:
        return LabelFeatures(
            available=self.label_available,
            present=self.label_present,
            coverage=self.label_coverage,
            ink=self.label_ink,
            centre_x=self.label_centre_x,
            centre_y=self.label_centre_y,
        )

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        bag_seq: str,
        slot_index: int,
        session_id: str | None = None,
        shift_key: str | None = None,
        state: str = "monitoring",
    ) -> "BagAppearance | None":
        """从落库形态还原；几何三项缺失一律返回 None（不拿默认值凑一个描述子）。"""
        if not payload:
            return None
        try:
            centre = float(payload["centre_x"])
            bottom = float(payload["bottom_y"])
            width = float(payload["width"])
        except (KeyError, TypeError, ValueError):
            return None
        hist = payload.get("hue_hist") or []
        label = LabelFeatures.from_json(payload)
        return cls(
            bag_seq=bag_seq,
            slot_index=slot_index,
            centre_x=centre,
            bottom_y=bottom,
            width=width,
            hue_hist=tuple(float(v) for v in hist),
            mean_saturation=float(payload.get("mean_saturation", 0.0)),
            mean_value=float(payload.get("mean_value", 0.0)),
            session_id=session_id,
            shift_key=shift_key,
            state=state,
            liquid_top_y=float(payload.get("liquid_top_y", 0.0)),
            liquid_height=float(payload.get("liquid_height", 0.0)),
            liquid_fill_fraction=float(payload.get("liquid_fill_fraction", 0.0)),
            label_available=label.available,
            label_present=label.present,
            label_coverage=label.coverage,
            label_ink=label.ink,
            label_centre_x=label.centre_x,
            label_centre_y=label.centre_y,
        )


# ── 颜色 ──────────────────────────────────────────────────────────────────


def _hue_histogram(pixels: np.ndarray) -> tuple[tuple[float, ...], float, float]:
    """像素 → (归一化色相直方图, 平均饱和度, 平均明度)。

    走 OpenCV 的 RGB→HSV 而不是自己写色彩空间转换：本项目已在分割主路径上依赖它，
    颜色口径与几何核保持一致，不另起一套会漂的 HSV 实现。
    """
    import cv2  # 局部导入：本模块的纯序列化部分不需要图像库

    if pixels.size == 0:
        return (), 0.0, 0.0
    rgb = pixels.reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    counts, _ = np.histogram(hsv[:, 0, 0], bins=HUE_BINS, range=(0, 180))
    total = float(counts.sum())
    hist = tuple((counts / total).tolist()) if total else tuple([0.0] * HUE_BINS)
    return hist, float(hsv[:, 0, 1].mean()), float(hsv[:, 0, 2].mean())


# ── 标签区 ────────────────────────────────────────────────────────────────


def detect_label(
    image: np.ndarray, x0: int, x1: int, liquid_top: int, liquid_bottom: int
) -> LabelFeatures:
    """在弯月面上方的袋体带里找标签区（**只判存在，不认字**）。

    观察带取 ``liquid_top`` 上方、横向覆盖液柱宽度的一段袋体。带内按本段 90 分位
    明度归一后切「亮膜 / 深色笔画」——这样夜间暗光与白天强光下判据一致。某一行
    **一个亮膜像素都没有**说明这一行已经越过袋子上沿（看到的是背景），整行不计入，
    避免把背景误当笔画；判「有没有亮膜像素」而不是判「亮膜占多数」，是因为印满字的
    标签行亮膜本来就不到一半，按占比判会把标签本身当成背景丢出去。
    """
    import cv2

    height, width = image.shape[:2]
    band_height = max(4, int(round(LABEL_BAND_FRACTION * max(1, liquid_bottom - liquid_top))))
    x_start, x_end = max(0, int(x0)), min(width, int(x1) + 1)
    band_bottom = max(0, min(height - 1, int(liquid_top) - 1))
    band_top = max(0, band_bottom - band_height + 1)
    if x_end <= x_start or band_bottom <= band_top:
        return _LABEL_UNAVAILABLE
    band = image[band_top : band_bottom + 1, x_start:x_end]
    if band.shape[0] * band.shape[1] < MIN_LABEL_BAND_PIXELS or band.size == 0:
        return _LABEL_UNAVAILABLE

    hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    bright_ref = float(np.percentile(value, 90))
    if bright_ref < MIN_LABEL_BAND_VALUE:
        return _LABEL_UNAVAILABLE  # 整段太暗：判不出有没有标签，如实说不可用

    film = value >= FILM_VALUE_RATIO * bright_ref
    ink = value < INK_VALUE_RATIO * bright_ref
    # 逐行判断「这一行还在袋子上」：整行一个亮膜像素都没有就是已经在袋子上沿之外。
    in_bag = film.any(axis=1)
    if not in_bag.any():
        return _LABEL_UNAVAILABLE
    valid = np.repeat(in_bag[:, None], film.shape[1], axis=1)
    area = float(valid.sum())
    coverage = float((film & valid).sum() / area)
    ink_ratio = float((ink & valid).sum() / area)
    present = ink_ratio >= MIN_LABEL_INK

    anchor = (ink & valid) if present else (film & valid)
    ys, xs = np.nonzero(anchor)
    if xs.size:
        centre_x = float((xs.mean() + x_start) / max(1, width))
        centre_y = float((ys.mean() + band_top) / max(1, height))
    else:  # 理论上不会走到（valid 非空即有像素），留一个可解释的退化值
        centre_x, centre_y = 0.0, 0.0
    return LabelFeatures(
        available=True,
        present=present,
        coverage=coverage,
        ink=ink_ratio,
        centre_x=centre_x,
        centre_y=centre_y,
    )


# ── 描述子 ────────────────────────────────────────────────────────────────


def extract_appearance(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    bag_seq: str,
    slot_index: int,
    session_id: str | None = None,
    shift_key: str | None = None,
    state: str = "monitoring",
) -> BagAppearance:
    """把一段候选区域变成可跨轮次比对的描述子。

    ``bbox`` 取**候选框的袋底**（框下沿）与横向中心、横向宽度。区域太小（像素不足）
    时直方图留空并标 ``geometry_only``——上游据此直接拒绝自动接上，而不是拿一条
    抖动的短直方图去凑分。
    """
    ys, xs = np.nonzero(mask)
    height, width = image.shape[:2]
    if xs.size:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
    else:  # 空掩膜不该走到这里，给一个退化但可解释的几何
        x0 = y0 = 0
        x1 = y1 = 0

    pixels = image[mask] if mask.any() else np.empty((0, 3), dtype=image.dtype)
    if pixels.shape[0] >= MIN_APPEARANCE_PIXELS:
        hist, sat, val = _hue_histogram(pixels)
    else:
        hist, sat, val = (), 0.0, 0.0

    bbox_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    liquid_top_y = y0 / max(1, height)
    bottom_y = (y1 + 1) / max(1, height)
    label = detect_label(image, x0, x1, y0, y1) if xs.size else _LABEL_UNAVAILABLE
    return BagAppearance(
        bag_seq=bag_seq,
        slot_index=slot_index,
        centre_x=((x0 + x1) / 2.0) / max(1, width),
        bottom_y=bottom_y,
        width=(x1 - x0 + 1) / max(1, width),
        hue_hist=hist,
        mean_saturation=sat,
        mean_value=val,
        session_id=session_id,
        shift_key=shift_key,
        state=state,
        liquid_top_y=liquid_top_y,
        liquid_height=max(0.0, bottom_y - liquid_top_y),
        liquid_fill_fraction=float(xs.size) / bbox_area,
        label_available=label.available,
        label_present=label.present,
        label_coverage=label.coverage,
        label_ink=label.ink,
        label_centre_x=label.centre_x,
        label_centre_y=label.centre_y,
    )


#: 兼容旧调用名（core/propagate.py、core/reacquire.py、api/claim.py 都用它）。
describe_region = extract_appearance


def descriptor_from_session(session: Mapping[str, Any]) -> BagAppearance | None:
    """从 ``bag_session`` 行还原描述子；描述子缺失返回 None（不当成「长得一样」）。"""
    raw = session.get("descriptor")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    return BagAppearance.from_json(
        raw,
        bag_seq=str(session.get("bag_seq") or ""),
        slot_index=int(session.get("slot_index") or 0),
        session_id=session.get("session_id"),
        shift_key=session.get("shift_key"),
        state=str(session.get("state") or "monitoring"),
    )