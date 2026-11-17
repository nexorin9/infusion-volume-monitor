"""首帧无提示枚举管线的验收测试。

覆盖四组断言：

1. **候选召回对照人工标注**：把候选跟「按场景参数独立重建的标注」逐袋比对，
   Top-K 命中才算过（枚举不是给出一堆框就完事，是要把每一只袋子都找出来）。
2. **与源实现同一输入的行为对照**：NMS 直接跟 ``torchvision.ops.boxes.batched_nms``
   对跑（源实现用的就是它）；碎片后处理断言源实现写明的两条语义——「孔洞」「孤岛」
   两条独立路径互不影响，以及「没被动过的掩膜在 NMS 里胜出」。
3. **低稳定分候选被剔除**：贴着阈值的边缘块不进候选列表。
4. **按槽位聚类与绑定**：唯一命中才绑定，认不准的不猜。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from infusion_monitor.core.enumerate import (
    EnumerationParams,
    MaskCandidate,
    batched_nms,
    build_point_grid,
    calculate_stability_score,
    cluster_by_slot,
    enumerate_candidates,
    postprocess_small_regions,
)
from infusion_monitor.core.geom_kernel import GeometricMaskPredictor, fill_small_holes
from tests.support import (
    EXAMPLE_FIELDS,
    BagSpec,
    RackScene,
    build_rack_zip,
    default_rack_scene,
    render_rack_frame,
)

TRACKS_URL = "/api/v1/infusion/tracks"

#: 贴阈值的「勉强像液体」的饱和度：S ≈ 99，抬阈值 24 级后整块都掉出去。
MARGINAL_RGB = (150, 210, 245)
#: 扎实的液体色：S ≈ 135，抬阈值 24 级后整块都还在。
SOLID_RGB = (92, 148, 196)


# ── 标注：从场景参数独立重建的液体真值 ────────────────────────────────────


def truth_mask(scene: RackScene, bag: BagSpec, slot: int, frame_index: int = 0) -> np.ndarray:
    """按场景参数重建某袋的液体真值掩膜（与检测器无关的人工标注口径）。

    这里刻意**不复用**渲染代码，而是按几何参数重新推一遍液面位置——检测器与标注
    共用同一份渲染真值时，召回率就只是自说自话。
    """
    height, width = scene.height, scene.width
    margin_x = 0.0
    half = bag.width_ratio / 2.0
    x0 = int(round((bag.x_center - half) * width))
    x1 = int(round((bag.x_center + half) * width))
    y0 = int(round(0.12 * height))
    y1 = int(round(0.90 * height))
    fill = scene.fill_at(bag.bag_seq, frame_index)
    liquid_top = y1 - int(round((y1 - y0) * max(0.0, min(1.0, fill))))
    mask = np.zeros((height, width), dtype=bool)
    mask[liquid_top : y1 - 3, x0 + 3 : x1 - 3] = True
    del margin_x
    return mask


def iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 0.0


def solid_panel(
    width: int = 480,
    height: int = 640,
    panels: tuple[tuple[int, int, int, int, tuple[int, int, int]], ...] = (),
) -> np.ndarray:
    """造一张只含若干纯色矩形（=液体块）的画面。"""
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    for x0, y0, x1, y1, color in panels:
        canvas[y0:y1, x0:x1] = color
    return canvas


# ── 1. 候选召回对照人工标注 ────────────────────────────────────────────────


def test_candidates_recall_every_bag_against_annotation():
    scene = default_rack_scene()
    frame = render_rack_frame(scene, 0)
    predictor = GeometricMaskPredictor()

    result = enumerate_candidates(frame, predictor, slot_count=len(scene.bags))

    assert result.candidates, "枚举管线一个候选都没给出"
    for slot, bag in enumerate(scene.bags):
        annotation = truth_mask(scene, bag, slot)
        best = max(result.candidates, key=lambda c: iou(c.mask, annotation))
        assert iou(best.mask, annotation) >= 0.60, (
            f"{bag.bag_seq} 没被召回：最好的候选 IoU="
            f"{iou(best.mask, annotation):.3f}，候选数={len(result.candidates)}"
        )


def test_top_k_recall_matches_annotation_ordering():
    """Top-K 命中：按稳定分排序后，标注里的每只袋子都要落在前 K 个候选里。"""
    scene = default_rack_scene()
    frame = render_rack_frame(scene, 0)
    predictor = GeometricMaskPredictor()

    result = enumerate_candidates(frame, predictor, slot_count=len(scene.bags))
    top_k = result.candidates[: max(2, len(scene.bags))]

    for slot, bag in enumerate(scene.bags):
        annotation = truth_mask(scene, bag, slot)
        assert any(iou(candidate.mask, annotation) >= 0.60 for candidate in top_k), (
            f"{bag.bag_seq} 不在前 K 个候选里"
        )


def test_every_candidate_carries_stability_score_and_bbox():
    scene = default_rack_scene()
    frame = render_rack_frame(scene, 0)

    result = enumerate_candidates(frame, GeometricMaskPredictor(), slot_count=2)

    for candidate in result.candidates:
        assert candidate.candidate_id.startswith("C")
        assert 0.0 < candidate.stability_score <= 1.0
        x0, y0, x1, y1 = candidate.bbox
        assert 0 <= x0 < x1 <= frame.shape[1]
        assert 0 <= y0 < y1 <= frame.shape[0]
        assert candidate.area > 0


# ── 2. 与源实现同一输入的行为对照 ──────────────────────────────────────────


def test_nms_matches_torchvision_batched_nms():
    """源实现用的就是 ``torchvision.ops.boxes.batched_nms``——直接对跑。"""
    torchvision_ops = pytest.importorskip("torchvision.ops.boxes")

    boxes = np.array(
        [
            [10.0, 10.0, 110.0, 210.0],
            [12.0, 14.0, 108.0, 205.0],  # 与 0 高度重合 → 应被抑制
            [300.0, 100.0, 380.0, 260.0],
            [302.0, 104.0, 379.0, 258.0],  # 与 2 高度重合 → 应被抑制
            [10.0, 300.0, 60.0, 420.0],
        ]
    )
    scores = np.array([0.9, 0.8, 0.7, 0.95, 0.6])

    mine = batched_nms(boxes, scores, 0.70)

    import torch

    reference = torchvision_ops.batched_nms(
        torch.as_tensor(boxes, dtype=torch.float32),
        torch.as_tensor(scores, dtype=torch.float32),
        torch.zeros(len(boxes), dtype=torch.float32),
        iou_threshold=0.70,
    ).tolist()
    assert sorted(mine) == sorted(reference)


def test_nms_keeps_box_at_threshold_and_drops_above():
    """口径：IoU ≤ 阈值的保留，> 阈值的抑制（与 torchvision 一致）。"""
    boxes = np.array([[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 50.0, 100.0]])
    scores = np.array([1.0, 0.5])
    # 第二个框与第一个的 IoU = 50*100 / (100*100 + 50*100 - 50*100) = 0.5
    assert batched_nms(boxes, scores, 0.50) == [0, 1]
    assert batched_nms(boxes, scores, 0.49) == [0]


def test_stability_score_follows_source_formula():
    """稳定分口径 = 源实现 ``calculate_stability_score`` 的分子/分母定义。

    源：``intersections = (masks > t + offset)`` 的像素数，
    ``unions = (masks > t - offset)`` 的像素数，比值即稳定分。
    """

    def reference(high: np.ndarray, low: np.ndarray, mask: np.ndarray) -> float:
        intersections = int(np.count_nonzero(high & mask))
        unions = int(np.count_nonzero(low & mask))
        return intersections / unions if unions else 0.0

    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    high = np.zeros_like(mask)
    high[10:25, 10:30] = True
    low = np.zeros_like(mask)
    low[10:30, 10:35] = True

    assert calculate_stability_score(high, low, mask) == pytest.approx(
        reference(high, low, mask)
    )
    # 分子是分母的子集 → 比值落在 (0, 1]
    assert 0.0 < calculate_stability_score(high, low, mask) <= 1.0


def test_postprocess_holes_and_islands_thresholds_are_independent():
    """源实现把孔洞与孤岛拆成两条独立调用（``mode="holes"`` / ``mode="islands"``）。

    这里的断言是：调孔洞门限只改变孔洞的处理结果，调孤岛门限只改变孤岛的处理结果
    ——两条路径互不影响（不是一条阈值管两件事）。
    """
    mask = np.zeros((120, 120), dtype=bool)
    mask[20:100, 20:100] = True  # 主体 80×80 = 6400 px
    mask[40:46, 40:46] = False  # 被包住的孔洞：6×6 = 36 px
    mask[5:13, 5:13] = True  # 独立的小块（孤岛）：8×8 = 64 px

    candidate = MaskCandidate(
        candidate_id="C01",
        mask=mask,
        bbox=(5, 5, 100, 100),
        area=int(np.count_nonzero(mask)),
        stability_score=1.0,
        point=(60.0, 60.0),
        crop_box=(0, 0, 120, 120),
    )

    def area_after(min_hole_area: int, min_sprinkle_area: int) -> int:
        processed = postprocess_small_regions(
            [candidate],
            min_hole_area=min_hole_area,
            min_sprinkle_area=min_sprinkle_area,
            nms_thresh=0.7,
        )
        return int(np.count_nonzero(processed[0].mask))

    base = 6400 - 36 + 64  # 主体 − 孔洞 + 孤岛（孔洞本来就是空的）
    hole_size = 36
    island_size = 64

    assert area_after(64, 0) == base + hole_size  # 孔洞被填上，孤岛留着
    assert area_after(0, 0) == base  # 两条路径都不生效
    assert area_after(0, 192) == base - island_size  # 孤岛被删，孔洞留着
    assert area_after(64, 192) == base - island_size + hole_size  # 两条路径都生效

    # 独立性：只动孔洞门限 → 面积差恰为孔洞大小；只动孤岛门限 → 面积差恰为孤岛大小
    assert area_after(64, 0) - area_after(0, 0) == hole_size
    assert area_after(64, 192) - area_after(0, 192) == hole_size
    assert area_after(64, 192) - area_after(64, 0) == -island_size


def test_postprocess_prefers_unchanged_mask_in_nms():
    """源实现给「没被后处理改动的掩膜」打 1 分、被改的打 0 分，NMS 因此偏好前者。"""
    base = np.zeros((100, 100), dtype=bool)
    base[20:80, 20:80] = True

    unchanged = MaskCandidate(
        candidate_id="C01",
        mask=base,
        bbox=(20, 20, 80, 80),
        area=int(np.count_nonzero(base)),
        stability_score=1.0,
        point=(50.0, 50.0),
        crop_box=(0, 0, 100, 100),
    )
    speckled = base.copy()
    speckled[50:53, 50:53] = False  # 一个会被填上的小孔洞 → 被判为「被改动」
    changed = MaskCandidate(
        candidate_id="C02",
        mask=speckled,
        bbox=(20, 20, 80, 80),
        area=int(np.count_nonzero(speckled)),
        stability_score=1.0,
        point=(50.0, 50.0),
        crop_box=(0, 0, 100, 100),
    )

    kept = postprocess_small_regions(
        [changed, unchanged], min_hole_area=64, min_sprinkle_area=0, nms_thresh=0.7
    )
    assert len(kept) == 1
    assert kept[0].candidate_id == "C01"


# ── 3. 低稳定分候选被剔除 ──────────────────────────────────────────────────


def test_low_stability_candidate_is_dropped_from_the_list():
    """贴着饱和度阈值的边缘块稳定分为 0 → 直接剔除，不进候选列表。"""
    frame = solid_panel(
        panels=(
            (40, 120, 220, 520, SOLID_RGB),
            (280, 120, 440, 520, MARGINAL_RGB),
        )
    )
    predictor = GeometricMaskPredictor()

    result = enumerate_candidates(frame, predictor, slot_count=2)

    kept_boxes = [candidate.bbox for candidate in result.candidates]
    assert any(box[0] < 200 for box in kept_boxes), "扎实的液体块没被枚举出来"
    assert not any(box[0] > 250 for box in kept_boxes), (
        "贴着阈值的边缘块本该被剔除，却出现在候选列表里"
    )


# ── 4. 按槽位聚类与绑定 ────────────────────────────────────────────────────


def test_slot_clustering_binds_only_unique_hits():
    scene = default_rack_scene()
    frame = render_rack_frame(scene, 0)

    result = enumerate_candidates(frame, GeometricMaskPredictor(), slot_count=2)

    assert set(result.slot_candidates) == {0, 1}
    for index, items in result.slot_candidates.items():
        assert len(items) == 1, f"槽位 {index} 聚到了多个候选"
    bound = result.bound_by_slot
    assert set(bound) == {0, 1}
    for candidate in result.candidates:
        assert candidate.bound is True
        assert candidate.slot_index in (0, 1)


def test_ambiguous_slot_is_not_bound():
    """同一槽位里挤了两个候选 → 认不准，不绑定（不猜）。"""
    frame = solid_panel(
        panels=(
            (30, 120, 110, 520, SOLID_RGB),
            (140, 120, 220, 520, SOLID_RGB),
        )
    )

    result = enumerate_candidates(frame, GeometricMaskPredictor(), slot_count=2)

    assert len(result.candidates) >= 2
    slot_zero = result.slot_candidates.get(0, ())
    if len(slot_zero) > 1:
        assert 0 not in result.bound_by_slot
        assert all(candidate.bound is False for candidate in slot_zero)
    # 无论聚到几个，绑定与否都必须在结果里如实标出来
    for candidate in result.candidates:
        assert isinstance(candidate.bound, bool)


def test_cluster_by_slot_uses_geometry_not_order():
    frame = solid_panel(
        panels=(
            (300, 120, 440, 520, SOLID_RGB),
            (40, 120, 180, 520, SOLID_RGB),
        )
    )
    predictor = GeometricMaskPredictor()
    result = enumerate_candidates(frame, predictor, slot_count=2)

    buckets = cluster_by_slot(result.candidates, 2, frame.shape[:2])
    for index, items in buckets.items():
        for candidate in items:
            centre = candidate.centre_x
            expected = 0 if centre < frame.shape[1] / 2 else 1
            assert index == expected


def test_hole_filling_only_fills_enclosed_holes():
    """只有**被包住的**背景才算孔洞；连着画面边界的背景不是孔洞。

    这条锁住枚举管线依赖的孔洞语义：碎片后处理会把掩膜里的洞填上——把画面边缘的
    背景也填上，等于在袋体之外凭空添一块液体，曲线立刻失真。
    """
    mask = np.zeros((60, 60), dtype=bool)
    mask[0:60, 0:50] = True  # 大片实心
    mask[0:5, 50:60] = False  # 右上角 50px² 的背景（连着画面边界，不是孔洞）
    mask[20:24, 20:24] = False  # 被包住的 16px² 真孔洞

    filled = fill_small_holes(mask, max_area=64)

    assert filled[22, 22], "被包住的小孔应被填上"
    assert not filled[2, 55], "连着画面边界的背景不该被当成孔洞填上"


# ── 5. 网格布点口径 ────────────────────────────────────────────────────────


def test_point_grid_layout_matches_source_convention():
    grid = build_point_grid(4, (100, 200))

    assert grid.shape == (16, 2)
    assert grid[:, 0].min() > 0 and grid[:, 0].max() < 200
    assert grid[:, 1].min() > 0 and grid[:, 1].max() < 100
    # 首末点各留半个间距的边距（offset = 1 / (2n)）
    assert grid[0, 0] == pytest.approx(200 / 8)
    assert grid[0, 1] == pytest.approx(100 / 8)
    # x 在列上变化（同一行 y 相同）
    assert len({round(float(value), 6) for value in grid[:4, 1]}) == 1


# ── 6. 枚举真的接进了产品主路径 ────────────────────────────────────────────


def test_upload_response_exposes_candidates_and_binding(client):
    scene = default_rack_scene()
    payload = build_rack_zip(scene)
    fields = dict(EXAMPLE_FIELDS)
    fields["slots"] = json.dumps(
        [
            {"bag_seq": bag.bag_seq, "slot_index": slot, "bag_type_code": "TYPE-A"}
            for slot, bag in enumerate(scene.bags)
        ]
    )

    response = client.post(
        TRACKS_URL, data=fields, files={"frames": ("rack.zip", payload, "application/zip")}
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["candidates"], "上传响应里没有候选列表"
    for candidate in body["candidates"]:
        assert set(candidate) >= {"candidate_id", "bbox", "area", "stability_score"}
        assert candidate["stability_score"] > 0
        assert len(candidate["bbox"]) == 4
    for bag in body["bags"]:
        assert bag["candidate_state"] in {"bound", "needs_claim"}
        if bag["candidate_state"] == "bound":
            assert bag["candidate_id"] in {c["candidate_id"] for c in body["candidates"]}
