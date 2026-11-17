"""``core.postprocess`` 的测试：连通域/形态特征 + 两个独立阈值 + 粘连分离。

任务 14 的验收点逐条对应下面的用例组：

1. **两个阈值各自的边界上下沿** —— 面积正好等于阈值 / 差一个像素的孔洞与碎斑；
2. **两个阈值互不影响** —— 调一个，另一个的输出逐像素不变；
3. **三类反例** —— 反光碎斑要剔、袋体形变造成的粘连不得误合并、液面低至 2% 的细窄
   连通域不得当碎斑删掉；
4. **主路径真的用上了它** —— 几何核 ``prepare()`` 在贴住的两袋上给出两块而不是一块，
   在正常画面上不误切。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from infusion_monitor.core.geom_kernel import (
    GeomKernelParams,
    GeometricMaskPredictor,
)
from infusion_monitor.core.postprocess import (
    AdhesionParams,
    apply_adhesion_cuts,
    component_features,
    detect_adhesion,
    extract_features,
    fill_small_holes,
    find_adhesion_cuts,
    label_components,
    postprocess_mask,
    remove_small_sprinkles,
)

LIQUID_RGB = (200, 40, 30)


# ── 掩膜与画面的构造 ─────────────────────────────────────────────────────


def blank(height: int = 120, width: int = 200) -> np.ndarray:
    return np.zeros((height, width), dtype=bool)


def box(mask: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    """在 ``mask`` 上画一个实心矩形（**就地修改**）。"""
    mask[y0:y1, x0:x1] = True
    return mask


def frame(rects, height: int = 160, width: int = 220) -> np.ndarray:
    """按矩形列表画一帧：深色背景 + 高饱和「液体」块（几何核认得的结构）。"""
    image = np.full((height, width, 3), 30, dtype=np.uint8)
    for y0, x0, y1, x1 in rects:
        image[y0:y1, x0:x1] = LIQUID_RGB
    return image


def solid_with_hole(hole_area: int) -> np.ndarray:
    """一块 40×40 的实心掩膜，中间挖一个正方形的孔（面积 = ``hole_area``）。"""
    mask = box(blank(), 20, 20, 60, 60)
    side = int(round(hole_area**0.5))
    assert side * side == hole_area, "本用例用正方形孔，方便手算面积"
    mask[30 : 30 + side, 30 : 30 + side] = False
    assert int(np.count_nonzero(mask)) == 1600 - hole_area
    return mask


def bounds(prepared, label: int) -> tuple[int, int, int, int]:
    """取一块的 ``(x0, y0, x1, y1)``（stats 里存的是 left/top/width/height）。"""
    row = prepared.stats[label]
    left, top = int(row[cv2.CC_STAT_LEFT]), int(row[cv2.CC_STAT_TOP])
    return left, top, left + int(row[cv2.CC_STAT_WIDTH]), top + int(row[cv2.CC_STAT_HEIGHT])


# ── 连通域标记与形态特征 ─────────────────────────────────────────────────


def test_label_components_marks_each_block_and_keeps_geometry():
    mask = box(blank(), 10, 10, 30, 50)  # 20×40
    box(mask, 60, 90, 100, 110)  # 40×20
    labelled = label_components(mask)

    assert labelled.count == 3  # 背景 + 两块
    areas = sorted(int(labelled.stats[i, cv2.CC_STAT_AREA]) for i in range(1, 3))
    assert areas == [800, 800]
    first = labelled.label_mask(1)
    assert first.sum() == 800 and not (first & (labelled.labels == 2)).any()
    # label_mask 返回独立副本，改它不会污染标签图
    first[:] = False
    assert labelled.label_mask(1).sum() == 800


def test_component_features_match_hand_computed_geometry():
    mask = box(blank(80, 80), 10, 20, 50, 50)  # 40 高 × 30 宽实心矩形
    labelled = label_components(mask)
    features = component_features(labelled, 1)

    assert features.area_px == 1200
    assert features.bbox == (20, 10, 50, 50)
    assert features.centroid == pytest.approx((34.5, 29.5), abs=0.01)  # OpenCV 给的是像素坐标均值
    assert features.extent == pytest.approx(1.0)  # 填满外接框
    assert features.hull_ratio == pytest.approx(1.0, abs=0.01)  # 凸包就是自己
    assert features.aspect_ratio == pytest.approx(30 / 40)  # 宽/高
    assert 0.6 < features.circularity <= 1.0  # 矩形圆度约 0.785
    assert features.touches_border is False
    assert component_features(labelled, 1, mask=mask).area_px == 1200


def test_component_features_flag_border_touching_block():
    mask = box(blank(60, 60), 0, 0, 20, 30)
    features = component_features(label_components(mask), 1)
    assert features.touches_border is True


def test_extract_features_orders_by_area_and_splits_concave_block():
    mask = box(blank(60, 120), 5, 5, 25, 25)  # 小块 20×20
    box(mask, 5, 60, 55, 110)  # 大块 50×50
    mask[20:40, 85:110] = False  # 在大块右缘啃一个凹口（不切断）

    features = extract_features(mask)
    assert [f.area_px for f in features] == [2000, 400]
    concave = features[0]
    assert concave.hull_ratio < 0.95, "被啃掉一块的凸包率必须低于 1"
    assert concave.extent < 1.0 and features[1].extent == pytest.approx(1.0)


def test_extract_features_rejects_label_without_pixels():
    labelled = label_components(blank(40, 40))
    with pytest.raises(ValueError, match="没有像素"):
        component_features(labelled, 1)


# ── 阈值一：孔洞填充（边界上下沿） ───────────────────────────────────────


def test_hole_exactly_at_threshold_is_filled_and_one_more_is_not():
    mask = solid_with_hole(64)

    assert fill_small_holes(mask, 64)[34, 34], "面积正好等于阈值的孔洞要填"
    assert not fill_small_holes(mask, 63)[34, 34], "面积比阈值大一个像素就不填"
    assert np.count_nonzero(fill_small_holes(mask, 64)) == 1600


def test_hole_filling_is_off_when_threshold_is_zero():
    mask = solid_with_hole(64)
    assert np.array_equal(fill_small_holes(mask, 0), mask)
    assert np.array_equal(fill_small_holes(mask, -5), mask)


def test_hole_filling_never_touches_the_outside():
    """挖一个与画面边界连通的缺口：它不是孔洞，多大的阈值都不该被填。"""
    mask = box(blank(60, 60), 10, 10, 50, 50)
    mask[30:, 40:] = False  # 从右下角啃掉一块，通到画面外

    filled = fill_small_holes(mask, max_area=100_000)
    assert not filled[40, 45], "与外界连通的缺口不是孔洞"
    assert np.count_nonzero(filled) == np.count_nonzero(mask)


def test_fill_small_holes_keeps_a_large_real_cavity():
    """袋体标签贴造成的大空洞是真实结构，不能被小阈值顺手填掉。"""
    mask = box(blank(80, 80), 10, 10, 70, 70)
    mask[20:40, 20:40] = False  # 400 px 的标签区

    assert not fill_small_holes(mask, 64)[30, 30]
    assert fill_small_holes(mask, 400)[30, 30]


# ── 阈值二：孤岛剔除（边界上下沿） ──────────────────────────────────────


def test_sprinkle_exactly_at_threshold_is_removed_and_one_more_is_kept():
    mask = box(blank(60, 120), 10, 10, 50, 50)  # 主体 1600 px
    box(mask, 10, 60, 14, 68)  # 32 px 碎斑
    assert int(np.count_nonzero(mask[10:14, 60:68])) == 32

    assert not remove_small_sprinkles(mask, 32)[12, 64], "面积正好等于阈值的碎斑要剔"
    assert remove_small_sprinkles(mask, 31)[12, 64], "面积比阈值大一个像素就留"


def test_sprinkle_removal_is_off_when_threshold_is_zero():
    mask = box(blank(60, 60), 10, 10, 50, 50)
    box(mask, 30, 30, 34, 38)
    assert np.array_equal(remove_small_sprinkles(mask, 0), mask)


def test_protected_point_spares_its_own_block_only():
    mask = box(blank(60, 120), 10, 10, 50, 50)  # 主体
    box(mask, 30, 30, 34, 38)  # 碎斑 A（在主体外，单独一块）
    box(mask, 30, 60, 34, 68)  # 碎斑 B

    kept = remove_small_sprinkles(mask, 64, protected=((33.5, 33.5),))
    assert kept[33, 33], "含正提示点的那块要留下"
    assert not kept[33, 63], "没被压住的碎斑照剔"
    assert kept[30, 30], "主体不受影响"


# ── 两个阈值互不影响 ─────────────────────────────────────────────────────


def _mixed_mask() -> np.ndarray:
    """主体 40×40（内含 4 px 孔洞）+ 一块 16 px 碎斑，三样东西各占一块。"""
    mask = box(blank(80, 160), 10, 10, 50, 50)
    box(mask, 60, 60, 62, 68)
    mask[20:22, 20:22] = False
    return mask


def _dropped_blocks(mask: np.ndarray, after: np.ndarray) -> list[int]:
    """按面积返回「被剔除掉的块」——比逐像素比较更能说明剔除的判定没被动过。"""
    before = label_components(mask)
    remaining = label_components(after)
    return sorted(
        int(before.stats[label, cv2.CC_STAT_AREA])
        for label in range(1, before.count)
        if not (remaining.labels == label).any()
    )


def test_hole_threshold_does_not_change_which_blocks_count_as_sprinkles():
    """填孔只会给已有块**加像素**，不会凭空多出一块；因此「哪些块算碎斑」与孔洞阈值无关。

    注意不能直接比逐像素结果——填上 4 px 孔洞后输出本就多 4 个像素，那正是填孔该干的事。
    """
    mask = _mixed_mask()
    for hole_area in (0, 4, 4_096):
        filled = fill_small_holes(mask, hole_area)
        dropped = _dropped_blocks(filled, remove_small_sprinkles(filled, 64))
        assert dropped == [16], f"改孔洞阈值 {hole_area} 竟然改变了「哪些块算碎斑」"


def test_sprinkle_threshold_does_not_change_hole_filling():
    """剔碎斑只删整块外部连通域，删完剩下的孔洞原样保留；因此孔洞填充与碎斑阈值无关。"""
    mask = _mixed_mask()
    baseline = fill_small_holes(mask, 64)
    # 未剔掉碎斑的那些阈值，输出必须逐像素一致
    for sprinkle_area in (0, 8, 15):
        assert np.array_equal(
            fill_small_holes(remove_small_sprinkles(mask, sprinkle_area), 64), baseline
        ), f"改碎斑阈值 {sprinkle_area} 竟然改变了孔洞填充的结果"

    # 即使碎斑整块被剔掉（阈值正好等于它的 16 px），主体里的孔洞仍按同一套规则填
    body = np.zeros_like(mask)
    body[5:55, 5:55] = True
    removed = remove_small_sprinkles(mask, 16)
    assert np.array_equal(fill_small_holes(removed, 64)[body], baseline[body])
    # 孔洞确实填上了：主体 1600 px（含补上的 4 px 孔洞）+ 还在的 16 px 碎斑
    assert int(np.count_nonzero(baseline)) == 1600 + 16


def test_postprocess_mask_reports_what_each_threshold_changed():
    mask = _mixed_mask()

    result = postprocess_mask(mask, max_hole_area=64, max_sprinkle_area=64)
    assert result.holes_filled == 4
    assert result.sprinkles_removed == 16
    assert result.changed is True
    assert int(np.count_nonzero(result.mask)) == 1600

    untouched = postprocess_mask(mask, max_hole_area=0, max_sprinkle_area=0)
    assert (untouched.holes_filled, untouched.sprinkles_removed) == (0, 0)
    assert untouched.changed is False
    assert np.array_equal(untouched.mask, mask)


# ── 反例一：反光造成的高对比碎斑要剔掉 ──────────────────────────────────


def test_specular_highlight_inside_liquid_is_absorbed_not_kept_as_a_speck():
    """镜面反光在液面上打出一个小洞：孔洞填充把它吸收掉，曲线不出现假凹陷。"""
    liquid = box(blank(120, 120), 10, 10, 110, 110)  # 液体主体
    highlight = box(blank(120, 120), 40, 40, 56, 56)  # 16×16 = 256 px 高光
    liquid &= ~highlight

    filled = fill_small_holes(liquid, GeomKernelParams().max_hole_area)
    assert int(np.count_nonzero(filled)) == 100 * 100
    assert int(np.count_nonzero(filled) - np.count_nonzero(liquid)) == 256

    # 若把阈值压到高光之下，它就会在面积曲线上留下一个 256 px 的凹陷
    unfilled = fill_small_holes(liquid, 128)
    assert int(np.count_nonzero(unfilled)) == 100 * 100 - 256


def test_highlight_spatter_outside_the_bag_is_removed_as_a_sprinkle():
    """袋体旁的高对比反光碎点不属于任何一袋，按孤岛剔除。"""
    mask = box(blank(120, 160), 20, 20, 100, 80)
    for y, x in ((30, 100), (44, 108), (58, 96), (72, 104)):
        box(mask, y, x, y + 5, x + 5)  # 每块 25 px

    cleaned = remove_small_sprinkles(mask, GeomKernelParams().max_sprinkle_area)
    assert int(np.count_nonzero(cleaned)) == 80 * 60, "四块反光碎点都该被剔掉"
    assert int(np.count_nonzero(mask)) == 80 * 60 + 4 * 25


# ── 反例二：袋体形变造成的两袋粘连不得误合并 ────────────────────────────


def test_two_touching_bags_are_separated_by_the_kernel():
    """两袋隔着 4 px 缝：闭运算会桥上，粘连分离必须把它切回两块。"""
    rects = [(20, 40, 120, 100), (20, 104, 120, 164)]
    prepared = GeometricMaskPredictor().prepare(frame(rects))

    assert prepared.count - 1 == 2, "贴住的两袋必须给出两块，不能合成一块"
    areas = sorted(int(prepared.stats[i, cv2.CC_STAT_AREA]) for i in (1, 2))
    assert abs(areas[0] - 6000) <= 150 and abs(areas[1] - 6000) <= 150
    first, second = sorted(bounds(prepared, i) for i in (1, 2))
    assert first[0] == 40 and second[0] >= 103, "切缝落在两袋之间"
    assert first[2] <= 102 and second[2] == 164, "右袋右缘不因切割而缩水"


def test_single_bag_is_never_cut_in_two():
    prepared = GeometricMaskPredictor().prepare(frame([(20, 40, 120, 100)]))
    assert prepared.count - 1 == 1
    assert int(prepared.stats[1, cv2.CC_STAT_AREA]) == 6000


def test_bags_far_apart_are_left_alone():
    prepared = GeometricMaskPredictor().prepare(
        frame([(20, 10, 120, 50), (20, 140, 120, 180)])
    )
    assert prepared.count - 1 == 2
    assert sorted(int(prepared.stats[i, cv2.CC_STAT_AREA]) for i in (1, 2)) == [4000, 4000]


def test_offset_bags_are_cut_on_the_seam_not_on_a_tilted_line():
    """两袋有高差时，垂直平分线会斜穿袋体；切割必须沿轴对齐的缝。"""
    prepared = GeometricMaskPredictor().prepare(
        frame([(30, 40, 120, 100), (20, 104, 120, 164)])
    )
    assert prepared.count - 1 == 2
    left_label = next(
        i for i in (1, 2) if int(prepared.stats[i, cv2.CC_STAT_LEFT]) == 40
    )
    left = prepared.labels == left_label
    assert left[:, 101:].sum() == 0, "切割线之后不该还留着左袋的像素"
    assert left[:, 95].sum() > 0, "左袋主体必须留住"
    assert int(prepared.stats[left_label, cv2.CC_STAT_TOP]) == 30
    assert int(prepared.stats[left_label, cv2.CC_STAT_HEIGHT]) <= 100


def test_three_bags_in_a_row_all_get_separated():
    prepared = GeometricMaskPredictor().prepare(
        frame([(20, 20, 120, 80), (20, 84, 120, 144), (20, 148, 120, 208)])
    )
    assert prepared.count - 1 == 3
    assert max(int(prepared.stats[i, cv2.CC_STAT_WIDTH]) for i in (1, 2, 3)) <= 63


def test_one_block_with_a_real_neck_is_split_on_the_double_peak():
    """一块两瓣（袋壁已贴合、液面连通）：靠「双峰 + 鞍部」判据拆开。"""
    mask = blank(160, 200)
    box(mask, 20, 40, 120, 100)
    box(mask, 20, 106, 120, 166)
    box(mask, 60, 100, 80, 106)  # 6 px 宽的连接颈

    report = detect_adhesion(mask)
    assert report.split is True
    assert report.parts == 2
    assert report.neck_width <= AdhesionParams().neck_ratio * report.peak_radius
    assert len(np.unique(label_components(report.mask).labels[report.mask])) == 2

    prepared = GeometricMaskPredictor().prepare(
        np.where(report.mask[..., None], LIQUID_RGB, np.uint8(30)).astype(np.uint8)
    )
    assert prepared.count - 1 == 2


def test_solid_block_with_a_shallow_dent_is_not_split():
    """袋体自身的凹凸（不是两袋粘连）不该被切开：鞍部不够深。"""
    mask = box(blank(120, 200), 20, 40, 100, 160)
    mask[60, 100:104] = False  # 4 px 的浅缺口，不构成颈

    assert detect_adhesion(mask).split is False


def test_pair_gap_threshold_bounds_which_pairs_are_cuts():
    mask = blank(120, 220)
    box(mask, 20, 40, 100, 100)
    box(mask, 20, 104, 100, 164)  # 4 px 缝 → 贴住
    box(mask, 20, 190, 100, 215)  # 26 px 缝 → 离得开

    cuts = find_adhesion_cuts(mask)
    assert len(cuts) == 1, "26 px 缝的那一对不算贴住"
    assert cuts[0].seam is not None and cuts[0].seam[0] == "vertical"
    assert cuts[0].neck_width <= AdhesionParams().max_pair_gap

    # 切割方案是在「桥上之前」取的，因此要作用在桥上之后的掩膜上（几何核 prepare() 的顺序）。
    kernel = np.ones((7, 7), np.uint8)
    bridged = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    assert label_components(bridged).count - 1 == 2, "闭运算确实把前两袋桥成了一块"

    split, applied = apply_adhesion_cuts(bridged, cuts)
    assert applied == 1
    assert label_components(split).count - 1 == 3


def test_small_blocks_are_skipped_by_the_min_area_gate():
    mask = blank(80, 120)
    box(mask, 20, 20, 30, 24)  # 40 px 碎斑
    box(mask, 20, 26, 30, 30)  # 40 px 碎斑
    assert find_adhesion_cuts(mask, min_area=1500) == []
    assert find_adhesion_cuts(mask, min_area=20), "门槛放到 20 时才进入判定"


# ── 反例三：液面低至 2% 的细窄连通域不得当碎斑删掉 ──────────────────────


def test_two_percent_fill_survives_sprinkle_removal_even_without_a_prompt():
    """细窄液柱 60×6 = 360 px，远大于默认阈值 192，本来就不该被剔——这里钉住上沿。"""
    params = GeomKernelParams()
    strip = box(blank(120, 160), 100, 40, 106, 100)  # 2% 液位
    assert int(np.count_nonzero(strip)) == 360
    assert 360 > params.max_sprinkle_area
    assert np.array_equal(remove_small_sprinkles(strip, params.max_sprinkle_area), strip)

    # 把阈值抬到液柱之上（模拟院内换更高分辨率的相机），提示点仍然保住它
    kept = remove_small_sprinkles(strip, 400, protected=((70.0, 103.0),))
    assert np.array_equal(kept, strip), "含正提示点的细窄液柱必须留下"


def test_two_percent_fill_survives_the_kernel_end_to_end():
    prepared = GeometricMaskPredictor().prepare(frame([(114, 40, 120, 100)]))
    assert prepared.count - 1 == 1
    assert int(prepared.stats[1, cv2.CC_STAT_AREA]) >= 360


def test_a_real_speck_next_to_the_low_fill_strip_is_still_removed():
    """细窄液柱留着，但它旁边的反光碎点仍要被剔——两件事不能互相牺牲。"""
    mask = box(blank(120, 160), 114, 40, 120, 100)  # 低液位
    box(mask, 40, 20, 43, 24)  # 9 px 反光碎点

    cleaned = remove_small_sprinkles(mask, 192)
    assert int(np.count_nonzero(cleaned)) == 360