"""掩膜几何后处理：连通域标记、形态特征、孔洞填充、孤岛剔除与粘连分离。

这个模块把几何核里「阈值带之后的形态学收尾」拆成独立可测的一层。它是纯 numpy/OpenCV
实现，不引入新依赖，也不碰任何业务语义（不做换液判定、不认袋序）。

**与源实现的语义对应**（只借参数语义，不搬代码、不跑源权重）：

* 源 ``SAM2ImagePredictor.__init__``（``sam2_image_predictor.py:37-40``）的
  ``max_hole_area``：掩膜里面积不超过它的**孔洞**被填上 → :func:`fill_small_holes`；
* 同处的 ``max_sprinkle_area``：面积不超过它的**碎斑**被删掉 → :func:`remove_small_sprinkles`；
* 源 ``masks = masks > self.mask_threshold``（``sam2_image_predictor.py:436``）二值化之后
  才做上述两步 —— 本实现的 :func:`postprocess_mask` 沿用同一顺序（先填孔、后剔孤岛）。

源实现的两个阈值**互不影响**：填孔只动掩膜内部的封闭区域，剔孤岛只评估掩膜之外的连通域。
本实现保持这一性质，并用 ``tests/test_postprocess.py`` 把「调一个阈值，另一个的行为逐像素
不变」钉住——这是拆分出来的最大收益：两条路径此前耦合在几何核里，改阈值很容易连累另一条。

**三处有意偏离**（都写在各自 docstring 里）：

1. 孤岛剔除保留**含正提示点**的那块（源实现删光全部小连通域）——液面低至 2% 的那袋本身
   就是一条细窄连通域，删掉等于把真数据当噪声抹平。
2. 粘连分离（:func:`split_adhesion`）是源实现没有的能力：源模型给的是逐物体掩膜，护士拍
   到的相邻两袋却会被形态学闭运算连成一块。输液架上两袋贴在一起会让一个袋子的面积比翻倍，
   进而把毫升数与换液时点全部带偏，因此这里给出可测的分离路径。
3. 形态特征里没有 OCR 或外观语义，纯几何量，避免把「看起来像」写死进分割层。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

__all__ = [
    "ComponentFeatures",
    "LabelledComponents",
    "AdhesionReport",
    "AdhesionCut",
    "PostprocessResult",
    "AdhesionParams",
    "label_components",
    "component_features",
    "extract_features",
    "fill_small_holes",
    "remove_small_sprinkles",
    "postprocess_mask",
    "detect_adhesion",
    "split_adhesion",
    "find_adhesion_cuts",
    "apply_adhesion_cuts",
    "split_adhered_components",
]


# ── 连通域与形态特征 ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class LabelledComponents:
    """一帧掩膜的连通域标签图。``labels`` 里 0 是背景，1..count-1 是各块。"""

    labels: np.ndarray
    stats: np.ndarray
    centroids: np.ndarray
    count: int
    shape: tuple[int, int]

    def label_mask(self, label: int) -> np.ndarray:
        """取出单块（返回独立副本，调用方改它不会污染标签图）。"""
        return self.labels == int(label)


@dataclass(frozen=True)
class ComponentFeatures:
    """单块连通域的形态特征。

    全部是**几何量**，不是概率也不含业务判断：

    * ``circularity`` 4πA/P²：形状规整度；
    * ``hull_ratio`` A/A_hull（solidity）：凹陷程度，粘连的袋间缝隙会让它明显小于单袋；
    * ``extent`` A/A_bbox：填充程度，细窄液柱与方块状袋体差别明显；
    * ``aspect_ratio`` 宽/高：竖直的输液袋远小于 1，横躺的袋子大于 1；
    * ``neck_width`` 最窄处宽度：粘连检测的核心判据（见 :func:`detect_adhesion`）。
    """

    label: int
    area_px: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    perimeter_px: float
    circularity: float
    hull_ratio: float
    extent: float
    aspect_ratio: float
    touches_border: bool

    def to_dict(self) -> dict:
        """给调试/审计用的普通 dict（JSON 可序列化）。"""
        return {
            "label": self.label,
            "area_px": self.area_px,
            "bbox": list(self.bbox),
            "centroid": [round(self.centroid[0], 2), round(self.centroid[1], 2)],
            "perimeter_px": round(self.perimeter_px, 2),
            "circularity": round(self.circularity, 4),
            "hull_ratio": round(self.hull_ratio, 4),
            "extent": round(self.extent, 4),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "touches_border": self.touches_border,
        }


def label_components(mask: np.ndarray, connectivity: int = 8) -> LabelledComponents:
    """连通域标记。返回标签图与每块的面积/外接框/质心，供后续按块取用。"""
    binary = mask.astype(np.uint8, copy=False)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity
    )
    return LabelledComponents(
        labels=labels,
        stats=stats,
        centroids=centroids,
        count=int(count),
        shape=(int(mask.shape[0]), int(mask.shape[1])),
    )


def component_features(
    labelled: LabelledComponents, label: int, mask: np.ndarray | None = None
) -> ComponentFeatures:
    """算单块的形态特征。``mask`` 省略时用标签图现切，避免重复求轮廓。"""
    binary = labelled.label_mask(label) if mask is None else mask.astype(bool)
    area = int(np.count_nonzero(binary))
    if area <= 0:
        raise ValueError(f"label={label} 在掩膜里没有像素")

    x = int(labelled.stats[label, cv2.CC_STAT_LEFT])
    y = int(labelled.stats[label, cv2.CC_STAT_TOP])
    w = int(labelled.stats[label, cv2.CC_STAT_WIDTH])
    h = int(labelled.stats[label, cv2.CC_STAT_HEIGHT])

    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))

    hull = cv2.convexHull(np.vstack(contours)) if contours else None
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0

    circularity = 0.0 if perimeter <= 0 else min(1.0, 4.0 * math.pi * area / perimeter**2)
    hull_ratio = 0.0 if hull_area <= 0 else min(1.0, area / hull_area)
    box_area = float(max(1, w * h))
    height, width = labelled.shape
    return ComponentFeatures(
        label=int(label),
        area_px=area,
        bbox=(x, y, x + w, y + h),
        centroid=(float(labelled.centroids[label][0]), float(labelled.centroids[label][1])),
        perimeter_px=perimeter,
        circularity=float(circularity),
        hull_ratio=float(hull_ratio),
        extent=float(min(1.0, area / box_area)),
        aspect_ratio=float(max(w, 1) / max(h, 1)),
        touches_border=bool(
            x <= 0 or y <= 0 or x + w >= width or y + h >= height
        ),
    )


def extract_features(mask: np.ndarray, connectivity: int = 8) -> list[ComponentFeatures]:
    """一帧掩膜的全部连通域形态特征，按面积从大到小排。"""
    labelled = label_components(mask, connectivity)
    features = [
        component_features(labelled, label) for label in range(1, labelled.count)
    ]
    features.sort(key=lambda f: f.area_px, reverse=True)
    return features


# ── 阈值一：孔洞填充（对应源实现 max_hole_area） ─────────────────────────


def fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """把面积 ``≤ max_area`` 的孔洞填上（对应源实现的 ``max_hole_area``）。

    做法：先从画面边界把「与外界连通」的背景标出来，剩下的才是真正的孔洞；只填
    够小的孔洞——袋体标签贴、气泡穿过液面造成的空洞若面积很大，那是真实结构，不动。
    ``max_area <= 0`` 时原样返回（与源实现的 ``> 0`` 才生效同义）。

    本函数**只写掩膜内部**：外部连通域与碎斑一个都不动，因此与
    :func:`remove_small_sprinkles` 的阈值互不影响。
    """
    if max_area <= 0 or not mask.any():
        return mask.copy()

    inverse = (~mask).astype(np.uint8)
    height, width = mask.shape
    # 外围补一圈「背景」，再从 (0,0) 漫水：这样标出来的才是**真正与外界连通**的背景，
    # 被包在里面的才是孔洞。（外围若补 0，漫水会漏进掩膜本身，外部背景就分不出来了。）
    padded = np.ones((height + 2, width + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = inverse
    flood_mask = np.zeros((height + 4, width + 4), dtype=np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 2)
    outside = padded[1:-1, 1:-1] == 2

    holes = inverse.astype(bool) & ~outside
    if not holes.any():
        return mask.copy()

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8), connectivity=8
    )
    filled = mask.copy()
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= max_area:
            filled |= labels == label
    return filled


# ── 阈值二：孤岛剔除（对应源实现 max_sprinkle_area） ─────────────────────


def remove_small_sprinkles(
    mask: np.ndarray,
    max_area: int,
    *,
    protected: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    """删掉面积 ``≤ max_area`` 且不含受保护点的碎斑（对应源实现的 ``max_sprinkle_area``）。

    ``protected`` 是正提示点坐标：**压住这些点所在的连通域**。这是相对源实现「删光
    全部小连通域」的有意偏离——液面低至 2% 的那袋本身就是细窄连通域，删掉等于把真数据
    当噪声抹平。

    本函数**只删掩膜之外的碎斑**：内部孔洞原样保留，因此与 :func:`fill_small_holes`
    的阈值互不影响。
    """
    if max_area <= 0 or not mask.any():
        return mask.copy()

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(mask)
    height, width = mask.shape
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > max_area:
            kept |= labels == label
            continue
        if any(
            0 <= int(round(y)) < height
            and 0 <= int(round(x)) < width
            and labels[int(round(y)), int(round(x))] == label
            for x, y in protected
        ):
            kept |= labels == label
    return kept


@dataclass(frozen=True)
class PostprocessResult:
    """两步后处理的结果与「被改过没有」的标记。

    ``holes_filled`` / ``sprinkles_removed`` 与源实现给未改动掩膜打 1 分、被改动打 0 分的
    做法同源：枚举管线据此决定候选在 NMS 里排前还是排后。
    """

    mask: np.ndarray
    holes_filled: int
    sprinkles_removed: int
    changed: bool


def postprocess_mask(
    mask: np.ndarray,
    *,
    max_hole_area: int,
    max_sprinkle_area: int,
    protected: tuple[tuple[float, float], ...] = (),
) -> PostprocessResult:
    """按源实现的顺序做两步：先填孔（``max_hole_area``）后剔孤岛（``max_sprinkle_area``）。"""
    filled = fill_small_holes(mask, max_hole_area)
    holes_filled = int(np.count_nonzero(filled) - np.count_nonzero(mask))

    sprinkled = remove_small_sprinkles(filled, max_sprinkle_area, protected=protected)
    sprinkles_removed = int(np.count_nonzero(filled) - np.count_nonzero(sprinkled))

    return PostprocessResult(
        mask=sprinkled,
        holes_filled=holes_filled,
        sprinkles_removed=sprinkles_removed,
        changed=holes_filled > 0 or sprinkles_removed > 0,
    )


# ── 粘连分离：两袋贴在一起不得合成一块 ─────────────────────────────────


@dataclass(frozen=True)
class AdhesionParams:
    """粘连判定门限。全部按工作分辨率下的像素计。"""

    #: 两个峰的最小间距（px）。小于此值视为同一条脊线上的噪声，不算两袋。
    min_separation: float = 12.0
    #: 峰的最小距离变换值（px）。太浅的局部极值不算「一袋的中心」。
    min_peak_radius: float = 4.0
    #: 峰高相对最大峰的下限（相对判据，抗不同分辨率）。
    peak_ratio: float = 0.55
    #: 颈宽 / 较小峰高 的上限。鞍部明显低于两峰才判粘连，单袋的脊线过不了这一关。
    neck_ratio: float = 0.82
    #: 只拆两块（输液架上贴住的是相邻两袋；再多留给任务 15 的轨迹分配）。
    max_parts: int = 2
    #: 局部极大值判定用的膨胀核边长。
    peak_ksize: int = 5
    #: 两个**独立块**之间算「贴住」的最大缝隙（px）。``MORPH_CLOSE`` 核边长 7 的半径是 3，
    #: 能桥接的缝隙上限约 6 px，这里留一点余量。超过这个缝隙闭运算桥不上，也就不必拆。
    max_pair_gap: float = 8.0
    #: 切割缝的宽度（px）：缝上这一条两边都不给。只切不留缝的话，切口两侧的像素上下相邻，
    #: 八连通下两块依然连成一块，等于没切。被丢掉的正是闭运算桥出来的桥面，本就不属于任何一袋。
    seam_width: float = 2.0


@dataclass(frozen=True)
class AdhesionReport:
    """一次粘连判定的结论。``split`` 为假时其余字段没有意义。"""

    mask: np.ndarray
    split: bool
    parts: int
    seeds: tuple[tuple[int, int], ...]
    neck_width: float
    peak_radius: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "parts": self.parts,
            "seeds": [list(s) for s in self.seeds],
            "neck_width": round(self.neck_width, 2),
            "peak_radius": round(self.peak_radius, 2),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AdhesionCut:
    """一处粘连的**切割方案**：在哪块区域、按哪条缝切。

    判定与切割分成两步，是因为**闭运算会把粘连的证据抹平**：两袋之间 4 px 的缝隙经过
    ``MORPH_CLOSE`` 之后被桥成一个和袋体一样厚的整块，此时再判「有两个袋」已经无从下手
    ——它和「一个特别宽的袋」在几何上完全一样。所以证据必须在闭运算**之前**的掩膜上取，
    切割则必须在闭运算**之后**的掩膜上做，否则刚切开又被桥回去。

    ``seam`` 是切割缝：``("vertical", x)`` 表示沿 x = x 处竖着切（输液架上相邻两袋几乎都是
    这种情况），``None`` 则退回按两个 ``seeds`` 做 Voronoi 切分。
    """

    bbox: tuple[int, int, int, int]
    seeds: tuple[tuple[int, int], tuple[int, int]]
    seam: tuple[str, float] | None
    neck_width: float
    peak_radius: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "seeds": [list(s) for s in self.seeds],
            "seam": None if self.seam is None else [self.seam[0], round(self.seam[1], 2)],
            "neck_width": round(self.neck_width, 2),
            "peak_radius": round(self.peak_radius, 2),
            "reason": self.reason,
        }


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    """到掩膜边界的距离图，**单位是像素**。

    ``maskSize`` 必须取 5：OpenCV 的 ``maskSize=3`` 走固定点近似，返回值是放大过 900 多倍
    的定点数（实测一条半宽 40 px 的袋体中心值 36044），直接拿来和 ``min_separation`` 这类
    像素门限比，等于门限全部失效。``maskSize=5`` 返回真正的 float32 欧氏距离。
    """
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)


def _local_peaks(distance: np.ndarray, params: AdhesionParams) -> np.ndarray:
    """距离图的局部极大值坐标（行优先，即 y 在前 x 在后）。

    用「与自身膨胀结果相等」判定极大值：比逐点比邻域更稳，也不会在平顶区报出一片点。
    """
    ksize = max(3, int(params.peak_ksize) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dilated = cv2.dilate(distance, kernel)
    peaks = np.argwhere((distance >= dilated - 1e-6) & (distance > 0))
    return peaks if peaks.size else np.empty((0, 2), dtype=int)


def _neck_width(distance: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    """两峰连线上的最小距离值——即颈部的「半宽」。

    沿线等距取样：样本点太少会跨过鞍部，太密只是白费算力，32 段足够平直地跨过袋间缝隙。
    """
    y0, x0 = a
    y1, x1 = b
    steps = 32
    ys = np.rint(np.linspace(y0, y1, steps)).astype(int)
    xs = np.rint(np.linspace(x0, x1, steps)).astype(int)
    ys = np.clip(ys, 0, distance.shape[0] - 1)
    xs = np.clip(xs, 0, distance.shape[1] - 1)
    return float(distance[ys, xs].min())


def detect_adhesion(
    mask: np.ndarray, params: AdhesionParams | None = None
) -> AdhesionReport:
    """判断一块掩膜里是否粘着两个物体（输液架上两袋贴在一起）。

    判据是**双峰 + 鞍部**：单袋的距离图沿中线是一条脊，峰值连续、鞍部不低于峰；两袋粘连
    则是两个独立的峰，中间有个明显下凹的颈。只看「有两个峰」会把单袋的脊线误判成两袋，
    只看「凹」又会被袋体自身的形状变化骗到，两个条件一起才稳。

    通过判据时返回 :func:`split_adhesion` 的分离结果，否则原样返回并说明原因。
    """
    params = params or AdhesionParams()
    empty = np.zeros(mask.shape, dtype=bool)

    if not mask.any():
        return AdhesionReport(
            mask=mask.copy(), split=False, parts=1, seeds=(),
            neck_width=0.0, peak_radius=0.0, reason="empty_mask",
        )

    distance = _distance_transform(mask)
    peaks = _local_peaks(distance, params)
    if peaks.size == 0:
        return AdhesionReport(
            mask=mask.copy(), split=False, parts=1, seeds=(),
            neck_width=0.0, peak_radius=0.0, reason="no_peak",
        )

    radius_floor = max(params.min_peak_radius, params.peak_ratio * float(distance.max()))
    heights = distance[peaks[:, 0], peaks[:, 1]]
    keep = heights >= radius_floor
    peaks, heights = peaks[keep], heights[keep]
    if peaks.shape[0] == 0:
        return AdhesionReport(
            mask=mask.copy(), split=False, parts=1, seeds=(),
            neck_width=0.0, peak_radius=0.0, reason="no_peak",
        )

    seeds = _lobe_seeds(distance, peaks, heights, radius_floor, params)
    if len(seeds) < 2:
        return AdhesionReport(
            mask=mask.copy(), split=False, parts=1, seeds=(),
            neck_width=0.0, peak_radius=float(heights.max()),
            reason="single_lobe",
        )

    a, b = seeds[0], seeds[1]
    neck = _neck_width(distance, a, b)
    peak_radius = float(min(distance[a[0], a[1]], distance[b[0], b[1]]))
    if neck > params.neck_ratio * peak_radius:
        return AdhesionReport(
            mask=mask.copy(), split=False, parts=1, seeds=(),
            neck_width=neck, peak_radius=peak_radius, reason="no_neck",
        )

    split_mask = _voronoi_split(mask, (a, b))
    return AdhesionReport(
        mask=split_mask,
        split=True,
        parts=2,
        seeds=(a, b),
        neck_width=neck,
        peak_radius=peak_radius,
        reason=f"adhesion_neck_{neck:.1f}px",
    )


def _lobe_seeds(
    distance: np.ndarray,
    peaks: np.ndarray,
    heights: np.ndarray,
    radius_floor: float,
    params: AdhesionParams,
) -> list[tuple[int, int]]:
    """把距离图按「高值连通区」聚成瓣，每瓣取最高峰作种子。

    **为什么不能只按峰间距聚类**：一整块实心矩形的距离图沿中线是一条**平台**，
    ``dilate`` 判极大值会在平台上报出一排相隔十几像素的峰。纯按距离聚类会把同一块袋体
    拆成「两个峰」，进而误判粘连。改按 ``distance >= radius_floor`` 的**连通区**聚类：
    一袋的脊线在阈值之上是连通的（并成一瓣），两袋之间的桥只有几像素宽、在阈值之下
    （自然分成两瓣）。
    """
    band = (distance >= radius_floor).astype(np.uint8)
    _, band_labels = cv2.connectedComponents(band, connectivity=8)
    lobe_of_peak = band_labels[peaks[:, 0], peaks[:, 1]]

    best: dict[int, int] = {}
    for index, lobe in enumerate(lobe_of_peak):
        lobe = int(lobe)
        if lobe not in best or heights[index] > heights[best[lobe]]:
            best[lobe] = index

    # 瓣心离得太近（小凸起被阈值切下来）时不拆：拆出来的多半是袋体自身的凹凸。
    ranked = sorted(best.items(), key=lambda item: -float(heights[item[1]]))
    seeds: list[tuple[int, int]] = []
    for _, index in ranked:
        y, x = int(peaks[index][0]), int(peaks[index][1])
        if all(math.hypot(y - sy, x - sx) >= params.min_separation for sy, sx in seeds):
            seeds.append((y, x))
        if len(seeds) >= params.max_parts:
            break
    return seeds


def _voronoi_split(
    mask: np.ndarray, seeds: tuple[tuple[int, int], ...], seam_width: float = 2.0
) -> np.ndarray:
    """按到各峰的欧氏距离把掩膜一分为二（两峰 Voronoi 分割）。

    种子取距离图的局部极大值——两袋各自最厚的地方。缝上 ``seam_width`` 宽的一圈像素
    两边都不给（判据是到两峰的距离之差 ``≤ seam_width``：点到中垂线的垂距是该差的一半，
    这样切出来的缝正好 ``seam_width`` 宽）。只在掩膜内分配像素，掩膜外的像素一律不进。
    """
    first, second = seeds[0], seeds[1]
    height, width = mask.shape
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    d_first = np.sqrt((ys - first[0]) ** 2 + (xs - first[1]) ** 2)
    d_second = np.sqrt((ys - second[0]) ** 2 + (xs - second[1]) ** 2)
    return mask & (np.abs(d_first - d_second) > seam_width)


def _seam_split(
    mask: np.ndarray, seam: tuple[str, float], seam_width: float = 2.0
) -> np.ndarray:
    """沿轴对齐的缝切开：缝上 ``seam_width`` 宽的一圈两边都不给，其余按左右/上下各归一边。"""
    orientation, position = seam
    half = seam_width / 2.0
    coordinate = np.arange(mask.shape[1] if orientation == "vertical" else mask.shape[0])
    keep = np.abs(coordinate.astype(np.float32) - position) > half
    if orientation == "vertical":
        return mask & keep[None, :]
    return mask & keep[:, None]


def split_adhesion(
    mask: np.ndarray, params: AdhesionParams | None = None
) -> AdhesionReport:
    """粘连分离的显式入口：判定 + 分割，等价于 :func:`detect_adhesion`。"""
    return detect_adhesion(mask, params)


def find_adhesion_cuts(
    mask: np.ndarray,
    params: AdhesionParams | None = None,
    *,
    min_area: int = 1500,
) -> list[AdhesionCut]:
    """在**仍看得到袋间缝隙**的掩膜上找出所有粘连处，返回切割方案。

    粘连有两种形态，两条都要认：

    1. **一块两瓣**（袋壁贴合、液面已经连通）：用「双峰 + 鞍部」判，见 :func:`detect_adhesion`；
    2. **两块贴住**（隔着几像素的袋壁或缝隙）：形态学闭运算会把这几像素桥上，两个袋变成
       一块。判据就是**两块之间的缝隙宽度** ≤ ``max_pair_gap``。

    两种形态都必须在**形态学闭运算之前**判定：桥上之后它与「一个特别宽的袋」在几何上完全
    一样，证据就没了。切割则放在闭运算之后做，否则刚切开又被桥回去。

    **逐连通域判，不整幅判**：整幅上两袋各有一个峰，看上去正好「双峰」，但两峰之间的连线
    要穿过背景（距离为 0），会被判成极窄的颈而把整幅一刀两半。逐块判定没有这个问题：
    分得开的两袋是两个块，各自单瓣，直接判不粘连。

    小于 ``min_area`` 的碎块跳过：它们粘不成两袋，逐块判也只是白花算力。

    **已知的边界**：缝隙窄到 ``≈2 px`` 及以下时，它在阈值化那一步就已经被 JPEG 压缩的
    振铃填平，阈值图里根本不存在这条缝——此时两袋与「一个特别宽的袋」在像素上不可区分，
    任何几何判据都无能为力，只能靠护士拉开距离再拍。这类场景不在本轮承诺范围内，
    与其猜一个缝不如如实合成一块（块面积偏大，但不会把两个袋子都算丢）。
    """
    if not mask.any():
        return []

    params = params or AdhesionParams()
    labelled = label_components(mask)
    cuts: list[AdhesionCut] = []

    solid = [
        label
        for label in range(1, labelled.count)
        if int(labelled.stats[label, cv2.CC_STAT_AREA]) >= min_area
    ]

    for label in solid:
        # 裁到外接框再判：距离变换与峰值搜索都按块的大小算，一帧里块越小省得越多。
        # **必须留 2 px 背景余量**：距离变换量的是到最近「零像素」的距离，若裁得刚好贴住
        # 外接框，整块裁片全是前景、一个零像素都没有，算出来的距离是垃圾（本项目实测该情形
        # 下 d.max() 直接顶到 65534），既慢又会把每一块都判成一堆「峰」。
        bbox = _bbox_of(labelled, label)
        x0, y0, x1, y1 = bbox
        px0, py0 = max(0, x0 - 2), max(0, y0 - 2)
        px1, py1 = min(mask.shape[1], x1 + 2), min(mask.shape[0], y1 + 2)
        crop = np.zeros((py1 - py0, px1 - px0), dtype=bool)
        crop[...] = labelled.labels[py0:py1, px0:px1] == label

        report = detect_adhesion(crop, params)
        if not report.split or len(report.seeds) < 2:
            continue
        cuts.append(
            AdhesionCut(
                bbox=bbox,
                seeds=(
                    (report.seeds[0][0] + py0, report.seeds[0][1] + px0),
                    (report.seeds[1][0] + py0, report.seeds[1][1] + px0),
                ),
                seam=None,
                neck_width=report.neck_width,
                peak_radius=report.peak_radius,
                reason=report.reason,
            )
        )

    for first, second in itertools.combinations(solid, 2):
        cut = _pair_cut(labelled, first, second, params)
        if cut is not None:
            cuts.append(cut)
    return cuts


def _bbox_of(labelled: LabelledComponents, label: int) -> tuple[int, int, int, int]:
    x = int(labelled.stats[label, cv2.CC_STAT_LEFT])
    y = int(labelled.stats[label, cv2.CC_STAT_TOP])
    w = int(labelled.stats[label, cv2.CC_STAT_WIDTH])
    h = int(labelled.stats[label, cv2.CC_STAT_HEIGHT])
    return x, y, x + w, y + h


def _pair_cut(
    labelled: LabelledComponents,
    first: int,
    second: int,
    params: AdhesionParams,
) -> AdhesionCut | None:
    """两块贴得够近时的切割方案；离得远返回 ``None``。

    先用外接框距离做 O(1) 粗筛，只有粗筛说「有可能」才去裁剪 + 距离变换量精确缝隙——
    画幅上通常只有两三块够大，块数一多粗筛就能把绝大多数配对挡在算力之外。
    """
    bx0, by0, bx1, by1 = _bbox_of(labelled, first)
    cx0, cy0, cx1, cy1 = _bbox_of(labelled, second)
    box_gap = math.hypot(
        max(0, max(bx0, cx0) - min(bx1, cx1)),
        max(0, max(by0, cy0) - min(by1, cy1)),
    )
    if box_gap > params.max_pair_gap + 1.0:
        return None

    height, width = labelled.shape
    x0, y0 = max(0, min(bx0, cx0)), max(0, min(by0, cy0))
    x1, y1 = min(width, max(bx1, cx1)), min(height, max(by1, cy1))
    crop = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    crop |= labelled.labels[y0:y1, x0:x1] == first
    crop |= labelled.labels[y0:y1, x0:x1] == second

    first_local = labelled.labels[y0:y1, x0:x1] == first
    second_local = labelled.labels[y0:y1, x0:x1] == second
    if not first_local.any() or not second_local.any():
        return None

    # 缝隙 = A 的边界到 B 的最近距离（距离变换在 ~A 上求，再在 B 上取最小）。
    to_second = cv2.distanceTransform(
        (~first_local).astype(np.uint8), cv2.DIST_L2, 3
    )
    gap = float(to_second[second_local].min())
    if gap > params.max_pair_gap:
        return None

    seed_a = _deepest_point(first_local)
    seed_b = _deepest_point(second_local)

    # 切割缝取**轴对齐**：输液架上相邻两袋几乎总是左右并排（缝隙是竖的），偶尔上下错位。
    # 两袋一旦有高差，两峰的垂直平分线就会斜着穿过袋体，把两个袋子都切坏；沿缝隙中线
    # 竖切则与袋体形状无关。
    gap_x = max(0, max(bx0, cx0) - min(bx1, cx1))
    gap_y = max(0, max(by0, cy0) - min(by1, cy1))
    if gap_x >= gap_y:
        seam = ("vertical", (max(bx0, cx0) + min(bx1, cx1)) / 2.0)
    else:
        seam = ("horizontal", (max(by0, cy0) + min(by1, cy1)) / 2.0)

    return AdhesionCut(
        bbox=(x0, y0, x1, y1),
        seeds=((seed_a[0] + y0, seed_a[1] + x0), (seed_b[0] + y0, seed_b[1] + x0)),
        seam=seam,
        neck_width=gap,
        peak_radius=float(min(seed_a[2], seed_b[2])),
        reason=f"pair_gap_{gap:.1f}px",
    )


def _deepest_point(mask: np.ndarray) -> tuple[int, int, float]:
    """块内离自身边界最远的点（``(y, x, 距离)``），用作 Voronoi 切割的种子。

    取「最厚处」而不是质心：袋体被标签、液面刻度切成不规则块时质心会偏到别处，
    而距离变换的最大值点永远落在实体内部的中轴上。
    """
    distance = _distance_transform(mask)
    flat = int(np.argmax(distance))
    y, x = divmod(flat, mask.shape[1])
    return int(y), int(x), float(distance[y, x])


def apply_adhesion_cuts(
    mask: np.ndarray,
    cuts: Sequence[AdhesionCut],
    params: AdhesionParams | None = None,
) -> tuple[np.ndarray, int]:
    """按切割方案切开掩膜，返回 ``(新掩膜, 实际切开处数)``。

    优先用轴对齐的缝（``cut.seam``），没有才退回按两个峰的 Voronoi 切分。只在切割框内
    动手，框外的像素一个不改；方案落在空白处（例如换了分辨率后位置对不上）就跳过，
    不硬切。
    """
    if not cuts:
        return mask.copy(), 0
    seam_width = (params.seam_width if params is not None else AdhesionParams().seam_width)
    out = mask.copy()
    applied = 0
    for cut in cuts:
        x0, y0, x1, y1 = cut.bbox
        window = np.zeros(mask.shape, dtype=bool)
        window[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        split = (
            _seam_split(window, cut.seam, seam_width)
            if cut.seam is not None
            else _voronoi_split(window, cut.seeds, seam_width)
        )
        if np.array_equal(split, window):
            continue
        out[y0:y1, x0:x1] &= ~window[y0:y1, x0:x1]
        out |= split
        applied += 1
    return out, applied


def split_adhered_components(
    mask: np.ndarray,
    params: AdhesionParams | None = None,
    *,
    min_area: int = 1500,
) -> tuple[np.ndarray, int]:
    """一步版的粘连分离（判定与切割都在同一份掩膜上）。

    主路径用不到它：几何核要先在闭运算**前**的掩膜上取证据、在闭运算**后**的掩膜上切割，
    走的是 :func:`find_adhesion_cuts` + :func:`apply_adhesion_cuts` 两段式。一步版留给
    形状已知、不再做形态学闭合的场景（例如单元测试与离线分析）。
    """
    cuts = find_adhesion_cuts(mask, params, min_area=min_area)
    return apply_adhesion_cuts(mask, cuts)