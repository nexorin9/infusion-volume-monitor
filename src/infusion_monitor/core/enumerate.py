"""首帧无提示枚举：护士在列表里认领，而不是逐袋点像素。

**为什么有这条管线**：输液架上的袋子由护理单元登记了挂点顺序，但护士端只拍得到
一段短片——「哪一块是 B01、哪一块是 B02」在画面里是几何与外观的事。逐袋点像素
既慢又不可复核；让程序把画面里**可能是袋子**的区域全找出来、给每块打一个稳定性
分，护士只需要在一张列表里点「这一行是 B01」。

**与源主路径的对应**（只借调用链与参数语义，不搬代码、不跑源权重）：

===========================================  ==========================================
本模块                                       源实现（代码证据）
===========================================  ==========================================
:func:`build_point_grid`                     归一化网格布点 ``automatic_mask_generator.py:110-118``
:func:`calculate_stability_score`            ``calculate_stability_score`` ``automatic_mask_generator.py:337-341``
:func:`batched_nms`                          ``torchvision.ops.boxes.batched_nms`` ``automatic_mask_generator.py:279-285``
:func:`postprocess_small_regions`            孔洞 / 孤岛两条路径 + 再 NMS ``automatic_mask_generator.py:386-435``
:func:`enumerate_candidates` 的补漏重扫段     ``_generate_masks`` → ``_process_crop`` → 换回原图坐标 ``automatic_mask_generator.py:224-292``
===========================================  ==========================================

**三处有意的医院侧偏离**（逐条写明原因，不藏着）：

1. **稳定分的扰动对象**：源在 logit 上扰动阈值；几何核没有 logit，改扰动**饱和度
   下限**（几何核的决定性阈值）。公式与判定方向完全一致——分子是「阈值抬高后仍算
   液体且落在本候选内的像素数」，分母是「阈值压低后算液体且落在本候选内的像素数」，
   比值越接近 1 说明这块区域离阈值边缘越远、越可靠。
2. **补漏裁剪框的取法**：源是规则切块（``generate_crop_boxes``）；本产品改成「已找到
   的候选所在区域外扩后重扫」。规则切块在输液架照片上会把一只袋子切成两半，切出来
   的半袋掩膜比找不到更糟；「按已找到的东西去邻域补漏」才是院内场景里有对应动作的
   做法（护士说的「旁边那袋也拍到了吧」）。
3. **孤岛剔除保留提示点**：与几何核既有语义一致（液面低至 2% 的那袋本身就很小，
   删掉等于把真数据当噪声抹平）。

**产品位置**：一次上传后，首帧枚举给出候选列表，每条带 ``stability_score`` 与
``bbox``；低分候选直接剔除，不拿低置信项打扰护士。候选按挂点几何聚类，唯一命中的
才算绑定，认不准的留给护士在列表里认领（认领与纠错接口见 task 9）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .geom_kernel import (
    GeomKernelParams,
    fill_small_holes,
    liquid_candidates_at,
    remove_small_sprinkles,
)
from .predictor import PerPointMaskPredictor, Prompt, mask_bbox

__all__ = [
    "EnumerationParams",
    "MaskCandidate",
    "EnumerationResult",
    "build_point_grid",
    "calculate_stability_score",
    "batched_nms",
    "postprocess_small_regions",
    "enumerate_candidates",
    "cluster_by_slot",
    "clip_region",
]

logger = logging.getLogger("infusion_monitor.core.enumerate")


@dataclass(frozen=True)
class EnumerationParams:
    """枚举管线参数。

    默认值按「手机拍的输液架近景」调过，**不是**源实现的原值——源默认每边 32 个
    网格点（1024 点）是为整幅自然图像准备的，画面里只有几只输液袋，8×8=64 点已经
    足够密，再密只是白烧算力。
    """

    #: 每边网格点数 → 总点数 = points_per_side²。
    points_per_side: int = 8
    #: 补漏重扫时的网格密度倍率（源实现 ``crop_n_points_downscale_factor`` 的反向用法：
    #: 裁剪层里点更密而不是更疏）。
    refill_points_scale: int = 2
    #: 稳定分下限：低于此值的候选直接剔除，不进候选列表。
    stability_score_thresh: float = 0.90
    #: 稳定分的扰动量（饱和度级数）。源实现是 logit 单位的 1.0。
    stability_score_offset: int = 24
    #: 框 IoU 阈值：超过它的框视为同一个东西，只留分数高的那个。
    box_nms_thresh: float = 0.70
    #: 跨裁剪层合并时的框 IoU 阈值（源实现 ``crop_nms_thresh``）。
    crop_nms_thresh: float = 0.70
    #: 碎片后处理：孔洞填充的面积门限（对应 ``max_hole_area``）。
    min_hole_area: int = 2048
    #: 碎片后处理：孤岛剔除的面积门限（对应 ``max_sprinkle_area``）。
    min_sprinkle_area: int = 192
    #: 补漏重扫时，候选并集的外扩比例。
    refill_pad_ratio: float = 0.25
    #: 单架最多几袋（与入站硬校验一致）。
    max_slots: int = 6
    #: 同一槽位最多保留几个候选（超出的按稳定分截断）。
    max_candidates_per_slot: int = 3


@dataclass(frozen=True, eq=False)
class MaskCandidate:
    """一个候选：首帧上「看起来像一只输液袋」的液体区域。

    ``mask`` 是**原图坐标系**下的 bool 数组；``bbox`` 是 ``(x0, y0, x1, y1)``。
    """

    candidate_id: str
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    stability_score: float
    point: tuple[float, float]
    crop_box: tuple[int, int, int, int]
    slot_index: int | None = None
    bound: bool = False
    fragments_removed: bool = False

    @property
    def centre_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def centre_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0


@dataclass(frozen=True)
class EnumerationResult:
    """一次首帧枚举的产出。"""

    candidates: tuple[MaskCandidate, ...]
    slot_candidates: dict[int, tuple[MaskCandidate, ...]]
    params: EnumerationParams

    @property
    def bound_by_slot(self) -> dict[int, MaskCandidate]:
        """唯一命中的槽位 → 候选。认不准的槽位**不出现**在这里，由上层标 unbound。"""
        return {
            index: items[0]
            for index, items in self.slot_candidates.items()
            if len(items) == 1
        }


def build_point_grid(points_per_side: int, shape: tuple[int, int]) -> np.ndarray:
    """归一化网格布点 → 原图像素坐标。

    口径对齐源实现的 ``build_point_grid``：点在 [0,1]² 上均匀分布，首末点各留
    半个间距的边距（``offset = 1 / (2 * n)``），x 在列上变化、y 在行上变化。
    """
    if points_per_side < 1:
        raise ValueError(f"points_per_side 至少为 1，收到 {points_per_side}")
    height, width = shape
    offset = 1.0 / (2.0 * points_per_side)
    coords = np.linspace(offset, 1.0 - offset, points_per_side)
    grid_x, grid_y = np.meshgrid(coords, coords, indexing="xy")
    points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    points[:, 0] *= width
    points[:, 1] *= height
    return points


def calculate_stability_score(
    high: np.ndarray, low: np.ndarray, mask: np.ndarray
) -> float:
    """稳定分：掩膜在阈值扰动下的保持比例。

    源实现的公式（在 logit 上）：``intersections / unions``，其中
    ``intersections = (masks > t + offset)`` 的像素数，
    ``unions = (masks > t - offset)`` 的像素数。

    本实现的等价式：``high`` 是把饱和度下限**抬高** ``offset`` 后的液体候选，
    ``low`` 是**压低**后的液体候选，``mask`` 是本候选自身。分子分母都限制在
    本候选内，因此比值落在 (0, 1]——比值越接近 1，说明这块区域离阈值边缘越远、
    越可靠；越低说明它大部分像素都贴着阈值，是「勉强像液体」的边缘块。
    """
    denominator = int(np.count_nonzero(low & mask))
    if denominator == 0:
        return 0.0
    numerator = int(np.count_nonzero(high & mask))
    return float(numerator) / float(denominator)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> list[int]:
    """贪心 NMS，返回保留的下标（按分数从高到低选）。

    与 ``torchvision.ops.boxes.batched_nms`` 同口径：被选中的框会抑制与它 IoU
    **大于**阈值的框（IoU ≤ 阈值的保留）。本项目所有框同属一类，源实现里那层
    「按类别分组」在这里退化为单组。
    """
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    areas = np.maximum(
        0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    )

    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        union = areas[current] + areas[rest] - intersection
        iou = np.where(union > 0, intersection / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def postprocess_small_regions(
    candidates: Sequence[MaskCandidate],
    *,
    min_hole_area: int,
    min_sprinkle_area: int,
    nms_thresh: float,
) -> list[MaskCandidate]:
    """碎片后处理 + 再做一次 NMS（源实现 ``postprocess_small_regions`` 的口径）。

    源实现的顺序：先 ``remove_small_regions(mode="holes")`` 填小孔洞，再
    ``remove_small_regions(mode="islands")`` 删小连通域；**给没被改动的掩膜打 1 分、
    被改的打 0 分**，再按这个分数做一次 NMS（NMS 因此偏好没被动过的那一块）；最后
    只给被改动的掩膜重算框。

    本实现的对应：孔洞填充与孤岛剔除复用几何核已有的两个函数（语义分别对应源实现
    的 ``max_hole_area`` / ``max_sprinkle_area``）；掩膜用 bool 数组而非 RLE；
    「被改动」的判定与打分完全一致。孤岛剔除保留含提示点的那块（见模块头偏离说明 3）。
    """
    if not candidates:
        return []

    new_masks: list[np.ndarray] = []
    scores: list[float] = []
    changed_flags: list[bool] = []
    for candidate in candidates:
        mask = candidate.mask
        filled = fill_small_holes(mask, min_hole_area)
        changed = not np.array_equal(filled, mask)
        sprinkled = remove_small_sprinkles(
            filled, min_sprinkle_area, protected=(candidate.point,)
        )
        changed = changed or not np.array_equal(sprinkled, filled)
        new_masks.append(sprinkled)
        scores.append(0.0 if changed else 1.0)
        changed_flags.append(changed)

    boxes = np.array(
        [mask_bbox(mask) or (0, 0, 0, 0) for mask in new_masks], dtype=float
    )
    keep = batched_nms(boxes, np.asarray(scores, dtype=float), nms_thresh)

    result: list[MaskCandidate] = []
    for index in keep:
        candidate = candidates[index]
        mask = new_masks[index]
        result.append(
            replace(
                candidate,
                mask=mask,
                bbox=mask_bbox(mask) or candidate.bbox,
                area=int(np.count_nonzero(mask)),
                fragments_removed=changed_flags[index],
            )
        )
    return result


def _kernel_params(predictor: PerPointMaskPredictor) -> GeomKernelParams:
    """取预测器用的几何核阈值；非几何核实现退回默认值。"""
    params = getattr(predictor, "params", None)
    if isinstance(params, GeomKernelParams):
        return params
    return GeomKernelParams()


def _scan_region(
    image: np.ndarray,
    predictor: PerPointMaskPredictor,
    params: EnumerationParams,
    kernel: GeomKernelParams,
    *,
    region: tuple[int, int, int, int] | None,
    points_per_side: int,
) -> list[MaskCandidate]:
    """在一个区域（整幅或 ROI）上按网格布点扫一遍，产出候选。

    对应源的 ``_process_crop``：裁剪 → 按裁剪尺寸缩放网格点 → 逐点出掩膜 →
    稳定性过滤 → 框 NMS → 把框与点换回原图坐标。
    """
    height, width = image.shape[:2]
    if region is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0, x1, y1 = region
    x1 = min(x1, width)
    y1 = min(y1, height)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return []

    grid = build_point_grid(points_per_side, crop.shape[:2])
    prompts = [Prompt.from_point(float(x), float(y)) for x, y in grid]
    predictions = predictor.predict_batch(crop, prompts)

    # 稳定性扰动用的两张「阈值抬高压低」候选图——整层只算一次，逐候选只做交集计数。
    high = liquid_candidates_at(
        crop, kernel.saturation_min + params.stability_score_offset, kernel
    )
    low = liquid_candidates_at(
        crop, kernel.saturation_min - params.stability_score_offset, kernel
    )

    found: list[MaskCandidate] = []
    for (point_x, point_y), preds in zip(grid, predictions):
        if not preds:
            continue
        best = preds[0]
        if best.is_empty:
            continue
        local_mask = best.mask
        stability = calculate_stability_score(high, low, local_mask)
        if stability < params.stability_score_thresh:
            # 低稳定分候选直接剔除：不拿低置信项去打扰护士。
            continue
        local_bbox = mask_bbox(local_mask)
        if local_bbox is None:
            continue

        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[y0:y1, x0:x1] = local_mask
        found.append(
            MaskCandidate(
                candidate_id="",
                mask=full_mask,
                bbox=(
                    local_bbox[0] + x0,
                    local_bbox[1] + y0,
                    local_bbox[2] + x0,
                    local_bbox[3] + y0,
                ),
                area=best.area,
                stability_score=round(float(stability), 4),
                point=(float(point_x + x0), float(point_y + y0)),
                crop_box=(x0, y0, x1, y1),
            )
        )

    # 同一只袋子会被很多个网格点命中——框 NMS 只留一个（分数 = 稳定分）。
    boxes = np.array([candidate.bbox for candidate in found], dtype=float).reshape(
        -1, 4
    )
    scores = np.array(
        [candidate.stability_score for candidate in found], dtype=float
    )
    keep = batched_nms(boxes, scores, params.box_nms_thresh)
    return [found[index] for index in keep]


def _union_roi(
    candidates: Sequence[MaskCandidate],
    pad_ratio: float,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """候选外接框的并集，按比例外扩后裁回画面。

    外扩是为了把「紧挨着的那只袋子」也扫进来——补漏要补的是邻居，不是自己。
    """
    height, width = shape
    x0 = min(candidate.bbox[0] for candidate in candidates)
    y0 = min(candidate.bbox[1] for candidate in candidates)
    x1 = max(candidate.bbox[2] for candidate in candidates)
    y1 = max(candidate.bbox[3] for candidate in candidates)
    pad_x = int(round((x1 - x0) * pad_ratio))
    pad_y = int(round((y1 - y0) * pad_ratio))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def clip_region(
    region: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """把区域裁进画面；裁完退化（宽或高 ≤0）返回 ``None``。

    局部重枚举（:mod:`.reacquire`）要把补漏重扫限制在调用方给的区域内，否则
    「只在丢失那只袋子周围找」会退化成整幅重扫，白烧算力还把别处的袋子收进来。
    """
    height, width = shape
    x0 = max(0, min(int(region[0]), width))
    y0 = max(0, min(int(region[1]), height))
    x1 = max(0, min(int(region[2]), width))
    y1 = max(0, min(int(region[3]), height))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _merge_across_crops(
    primary: Sequence[MaskCandidate],
    extra: Sequence[MaskCandidate],
    iou_threshold: float,
) -> list[MaskCandidate]:
    """合并不同裁剪层的结果（源实现按 ``1 / 裁剪框面积`` 打分做跨框 NMS）。

    源实现：``scores = 1 / box_area(data["crop_boxes"])``——裁剪框越小分越高，
    于是同一件东西有多份候选时，保留来自更小裁剪框（看得更细）的那份。
    """
    combined = list(primary) + list(extra)
    if not combined:
        return []
    boxes = np.array([candidate.bbox for candidate in combined], dtype=float).reshape(
        -1, 4
    )
    scores = np.array(
        [
            1.0
            / max(
                1.0,
                float(
                    (candidate.crop_box[2] - candidate.crop_box[0])
                    * (candidate.crop_box[3] - candidate.crop_box[1])
                ),
            )
            for candidate in combined
        ],
        dtype=float,
    )
    keep = batched_nms(boxes, scores, iou_threshold)
    return [combined[index] for index in keep]


def cluster_by_slot(
    candidates: Sequence[MaskCandidate],
    slot_count: int,
    shape: tuple[int, int],
) -> dict[int, list[MaskCandidate]]:
    """按挂点几何把候选聚到槽位上。

    槽位几何沿用主路径的先验（见 ``propagate.slot_seed_prompt``）：画面按挂点顺序
    竖向等分 ``slot_count`` 条，候选外接框中心落在哪一条就归到哪个槽位。

    **不猜**：某个槽位聚到 0 个或多个候选都算「认不准」，上层据此标 unbound 让护士
    在列表里认领——这里只负责把候选摆到正确的槽位旁边。
    """
    count = max(1, slot_count)
    _, width = shape
    step = width / count
    buckets: dict[int, list[MaskCandidate]] = {index: [] for index in range(count)}
    for candidate in candidates:
        index = int(candidate.centre_x / step)
        index = min(count - 1, max(0, index))
        buckets[index].append(candidate)
    return buckets


def enumerate_candidates(
    image: np.ndarray,
    predictor: PerPointMaskPredictor,
    *,
    params: EnumerationParams | None = None,
    slot_count: int | None = None,
    region: tuple[int, int, int, int] | None = None,
) -> EnumerationResult:
    """首帧枚举：把画面里可能是输液袋的液体区域全找出来。

    流程（对齐源实现 ``_generate_masks`` 的骨架）：

    1. 整幅网格布点 → 每点出掩膜；
    2. 按稳定分过滤（低分候选直接剔除，不进候选列表）；
    3. 框 NMS 抑制重复；
    4. 碎片后处理（孔洞 / 孤岛两条独立路径）+ 再 NMS；
    5. **补漏重扫**：在已找到候选所在区域用更密的网格再扫一遍（对应源的
       ``_process_crop`` 补漏），坐标换回原图后与已有候选合并。

    ``slot_count`` 给了就把候选按挂点几何聚类（认不准的槽位不绑定）。

    ``region`` 给了就**只在这块局部区域里枚举**，且补漏重扫也被限制在同一区域内
    （:mod:`.reacquire` 的「丢失轨迹局部重枚举」直接复用本函数，不另起一套扫描）。
    """
    params = params or EnumerationParams()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"期望 HxWx3 的 RGB 帧，实际形状 {image.shape}")
    height, width = image.shape[:2]
    kernel = _kernel_params(predictor)
    scan_region = clip_region(region, (height, width)) if region is not None else None
    if region is not None and scan_region is None:
        raise ValueError(f"region={region} 与画面 {(height, width)} 无交集")

    found = _scan_region(
        image,
        predictor,
        params,
        kernel,
        region=scan_region,
        points_per_side=params.points_per_side,
    )
    refined = postprocess_small_regions(
        found,
        min_hole_area=params.min_hole_area,
        min_sprinkle_area=params.min_sprinkle_area,
        nms_thresh=params.box_nms_thresh,
    )

    if refined:
        roi = _union_roi(refined, params.refill_pad_ratio, (height, width))
        if scan_region is not None:
            # 补漏也只在调用方给的区域内做：局部重枚举不该顺手把整幅画面重扫一遍
            bounded = clip_region(roi, (height, width))
            roi = _intersect_regions(bounded, scan_region) if bounded else None
        refill = (
            []
            if roi is None
            else _scan_region(
                image,
                predictor,
                params,
                kernel,
                region=roi,
                points_per_side=params.points_per_side * params.refill_points_scale,
            )
        )
        merged = _merge_across_crops(refined, refill, params.crop_nms_thresh)
    else:
        merged = []

    merged.sort(key=lambda candidate: candidate.stability_score, reverse=True)
    numbered = tuple(
        replace(candidate, candidate_id=f"C{index + 1:02d}")
        for index, candidate in enumerate(merged)
    )

    if slot_count is None:
        annotated = numbered
        slot_candidates: dict[int, tuple[MaskCandidate, ...]] = {}
    else:
        buckets = cluster_by_slot(numbered, slot_count, (height, width))
        # 唯一命中的槽位才算绑定；同槽位多个候选或一个都没有都算「认不准」，
        # 上层据此标 unbound 让护士在列表里认领——本模块不猜。
        slot_of: dict[str, int] = {}
        bound_of: dict[str, bool] = {}
        for index, items in buckets.items():
            for candidate in items:
                slot_of[candidate.candidate_id] = index
                bound_of[candidate.candidate_id] = len(items) == 1
        annotated = tuple(
            replace(
                candidate,
                slot_index=slot_of.get(candidate.candidate_id),
                bound=bound_of.get(candidate.candidate_id, False),
            )
            for candidate in numbered
        )
        by_id = {candidate.candidate_id: candidate for candidate in annotated}
        slot_candidates = {
            index: tuple(
                sorted(
                    (by_id[item.candidate_id] for item in items),
                    key=lambda candidate: candidate.stability_score,
                    reverse=True,
                )[: params.max_candidates_per_slot]
            )
            for index, items in buckets.items()
        }

    logger.info(
        "枚举完成 候选=%d 绑定=%d 槽位=%d 区域=%s",
        len(annotated),
        sum(1 for items in slot_candidates.values() if len(items) == 1),
        len(slot_candidates),
        "整幅" if scan_region is None else f"x{scan_region[0]}..{scan_region[2]}",
    )
    return EnumerationResult(
        candidates=annotated,
        slot_candidates=slot_candidates,
        params=params,
    )


def _intersect_regions(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """两块区域的交集；不相交返回 ``None``。"""
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)
