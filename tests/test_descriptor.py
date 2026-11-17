"""外观描述子（颜色 / 液位 / 标签区）的验收测试。

五组断言：

1. **颜色**：区域够大才有直方图，够小则整档降为 ``geometry_only``（不自动接上）；
2. **液位**：弯月面与液柱高度随液面下降而下降，袋底与横向宽度不动；
3. **标签区**：带印刷笔画时 ``label_present=True`` 且档位 ``full``；无笔画（空膜）
   时 ``label_present=False``、档位降为 ``color_only``；整段过暗时如实标**不可用**
   （不把「没看清」当成「没有标签」）；
4. **降级方向唯一**：宁可少用一项判据，也绝不把「没存」当成「没有标签」——
   旧版描述子（无标签字段）读回来必须是 ``color_only`` 而不是 ``full``；
5. **落库往返**：JSON 往返不丢字段，``descriptor_mode`` 与得分用到的项一致。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
from infusion_monitor.core.predictor import Prompt
from infusion_monitor.identity.descriptor import (
    DESCRIPTOR_MODES,
    HUE_BINS,
    MIN_APPEARANCE_PIXELS,
    BagAppearance,
    LabelFeatures,
    descriptor_from_session,
    describe_region,
    detect_label,
    extract_appearance,
)
from tests.support import BagSpec, RackScene, default_rack_scene, render_rack_frame


def _liquid_mask(image: np.ndarray, x_center: float) -> np.ndarray:
    """用几何核在画面里切出液体掩膜（走产品主路径，不手搓掩膜）。"""
    point = (x_center * image.shape[1], image.shape[0] * 0.55)
    return GeometricMaskPredictor().predict(image, Prompt.from_point(*point))[0].mask


def _descriptor_at(scene: RackScene, frame_index: int, slot: int = 0) -> BagAppearance:
    image = np.asarray(render_rack_frame(scene, frame_index))
    return describe_region(
        image,
        _liquid_mask(image, scene.bags[slot].x_center),
        bag_seq=scene.bags[slot].bag_seq,
        slot_index=slot,
        shift_key="EXAMPLE-SHIFT",
    )


# ── 1. 颜色 ───────────────────────────────────────────────────────────────


def test_colour_histogram_is_built_only_for_big_enough_regions():
    scene = default_rack_scene(16)
    image = np.asarray(render_rack_frame(scene, 0))
    bag = scene.bags[0]
    full = describe_region(image, _liquid_mask(image, bag.x_center), bag_seq="B01", slot_index=0)

    assert len(full.hue_hist) == HUE_BINS
    assert full.appearance_available
    assert sum(full.hue_hist) == pytest.approx(1.0, abs=1e-6)
    assert full.descriptor_mode == "full"

    # 只留一小撮像素（低于 MIN_APPEARANCE_PIXELS）→ 颜色不可用
    tiny = np.zeros_like(full_mask := _liquid_mask(image, bag.x_center))
    ys, xs = np.nonzero(full_mask)
    tiny[ys.min() : ys.min() + 8, xs.min() : xs.min() + 8] = True
    thin = describe_region(image, tiny, bag_seq="B01", slot_index=0)
    assert thin.hue_hist == ()
    assert not thin.appearance_available
    assert thin.descriptor_mode == "geometry_only"


# ── 2. 液位特征 ───────────────────────────────────────────────────────────


def test_liquid_level_features_track_the_drop_but_not_the_bag_body():
    """液面下降时弯月面下移、液柱变矮；袋底与横向宽度不动（那才是袋子的身份）。"""
    scene = default_rack_scene(16)
    high = _descriptor_at(scene, 0)
    low = _descriptor_at(scene, scene.frame_count - 1)

    assert low.liquid_top_y > high.liquid_top_y, "液面降了，弯月面却没往下走"
    assert low.liquid_height < high.liquid_height
    assert high.level_available and low.level_available
    assert low.centre_x == pytest.approx(high.centre_x, abs=0.01)
    assert low.bottom_y == pytest.approx(high.bottom_y, abs=0.02)
    assert low.width == pytest.approx(high.width, abs=0.02)
    # 软袋里的液体铺满掩膜外接框（不是稀疏的碎块）
    assert high.liquid_fill_fraction > 0.9


# ── 3. 标签区 ─────────────────────────────────────────────────────────────


def _draw_labelled_bag(
    *, liquid_rgb: tuple[int, int, int], label_ink: bool, dark: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """画一只带标签贴纸的袋子，返回 (画面, 液体掩膜)。

    袋体 = 亮膜；标签贴纸上的印刷笔画 = 亮膜里的深色细横线。``label_ink=False``
    就是「空膜、没贴标签」；``dark=True`` 模拟夜间没开灯（整段偏暗）。
    """
    scale = 0.12 if dark else 1.0
    canvas = np.full((240, 320, 3), int(30 * scale), dtype=np.uint8)
    body = tuple(int(c * scale) for c in (198, 210, 216))
    x0, x1 = 90, 230
    liquid_top, bottom = 150, 220
    cv2.rectangle(canvas, (x0, 40), (x1, bottom), body, -1)  # 袋体（含弯月面上方的空膜）
    cv2.rectangle(canvas, (x0, liquid_top), (x1, bottom), liquid_rgb, -1)  # 液体
    if label_ink:
        ink = tuple(int(c * scale * 0.25) for c in (40, 40, 40))
        for row in (70, 82, 94, 106):  # 标签上的印刷横线
            cv2.rectangle(canvas, (x0 + 18, row), (x1 - 40, row + 3), ink, -1)
    mask = np.zeros(canvas.shape[:2], dtype=bool)
    mask[liquid_top:bottom, x0:x1] = True
    return canvas, mask


def test_label_region_is_detected_without_ocr():
    """带印刷笔画 → label_present=True 且档位 full；不依赖任何 OCR 依赖。"""
    image, mask = _draw_labelled_bag(liquid_rgb=(120, 170, 210), label_ink=True)
    descriptor = extract_appearance(image, mask, bag_seq="B01", slot_index=0)

    label = descriptor.label_features()
    assert label.available and label.present
    assert label.ink >= 0.03
    assert 0.0 < label.coverage <= 1.0
    assert descriptor.descriptor_mode == "full"
    assert 0.2 < label.centre_x < 0.8


def test_plain_film_without_ink_still_yields_a_usable_label_observation():
    """空膜（没贴标签）是**一条结论**，不是「判不出来」：present=False 但档位仍 full。

    降级只在「看不清」时发生：``color_only`` 意味着这一轮根本没拿到标签区信息，
    不能拿来当「这只袋子上没有标签」用。两者混同会让换过标签的袋被判成同一袋。
    """
    image, mask = _draw_labelled_bag(liquid_rgb=(120, 170, 210), label_ink=False)
    descriptor = extract_appearance(image, mask, bag_seq="B01", slot_index=0)

    label = descriptor.label_features()
    assert label.available
    assert not label.present
    assert label.ink < 0.03
    assert descriptor.descriptor_mode == "full"
    assert descriptor.appearance_available and descriptor.level_available


def test_unusable_label_information_degrades_to_color_only():
    """标签区判不出来时降级为「几何+颜色+液位」，并如实标 descriptor_mode。"""
    image, mask = _draw_labelled_bag(liquid_rgb=(120, 170, 210), label_ink=True, dark=True)
    descriptor = extract_appearance(image, mask, bag_seq="B01", slot_index=0)

    assert not descriptor.label_available
    assert not descriptor.label_present
    assert descriptor.descriptor_mode == "color_only"


def test_dark_band_reports_label_unavailable_instead_of_guessing():
    """整段太暗判不出有没有标签时如实说不可用——不把「没看清」当成「没有标签」。"""
    image, mask = _draw_labelled_bag(liquid_rgb=(120, 170, 210), label_ink=True, dark=True)
    label = detect_label(image, 90, 230, 150, 220)
    assert not label.available
    assert label.ink == 0.0 and label.coverage == 0.0

    descriptor = extract_appearance(image, mask, bag_seq="B01", slot_index=0)
    assert descriptor.descriptor_mode in DESCRIPTOR_MODES
    assert not descriptor.label_present


def test_label_detection_ignores_rows_above_the_bag():
    """观察带越过袋子上沿看到背景时，背景的暗像素不算「印刷笔画」。"""
    image, mask = _draw_labelled_bag(liquid_rgb=(120, 170, 210), label_ink=False)
    # 液体顶放到很靠上 → 观察带整段落在背景（暗）上
    high_liquid = np.zeros_like(mask)
    high_liquid[30:220, 90:230] = True
    assert not detect_label(image, 90, 230, 30, 220).available


# ── 4. 降级方向唯一 ───────────────────────────────────────────────────────


def test_legacy_descriptor_without_label_fields_degrades_to_color_only():
    """旧版落库描述子（只有颜色几何 8 个键）读回来必须降级，不得当成 full。"""
    legacy = {
        "bag_seq": "B01",
        "slot_index": 0,
        "centre_x": 0.3,
        "bottom_y": 0.7,
        "width": 0.3,
        "hue_hist": [0.0] * HUE_BINS,
        "mean_saturation": 80.0,
        "mean_value": 140.0,
        "state": "monitoring",
    }
    restored = BagAppearance.from_json(legacy, bag_seq="B01", slot_index=0)
    assert restored is not None
    assert restored.appearance_available
    assert not restored.label_available
    assert restored.descriptor_mode == "color_only"
    assert restored.label_features() == LabelFeatures(False, False, 0.0, 0.0, 0.0, 0.0)


# ── 5. 落库往返 ───────────────────────────────────────────────────────────


def test_descriptor_round_trips_through_json_without_losing_features():
    scene = default_rack_scene(16)
    original = _descriptor_at(scene, 4, slot=1)
    payload = original.to_json()

    assert payload["descriptor_mode"] in DESCRIPTOR_MODES
    assert "label_available" in payload and "liquid_height" in payload
    restored = BagAppearance.from_json(payload, bag_seq="B02", slot_index=1, session_id="s1")
    assert restored is not None
    assert restored.hue_hist == pytest.approx(original.hue_hist)
    assert restored.liquid_top_y == pytest.approx(original.liquid_top_y)
    assert restored.liquid_height == pytest.approx(original.liquid_height)
    assert restored.liquid_fill_fraction == pytest.approx(original.liquid_fill_fraction)
    assert restored.label_present == original.label_present
    assert restored.label_coverage == pytest.approx(original.label_coverage)
    assert restored.label_centre_x == pytest.approx(original.label_centre_x)
    assert restored.descriptor_mode == original.descriptor_mode


def test_descriptor_from_session_returns_none_when_absent():
    """库里没有描述子 → None（不当成「长得一样」）。"""
    assert descriptor_from_session({"bag_seq": "B01", "slot_index": 0, "descriptor": {}}) is None
    assert descriptor_from_session({"bag_seq": "B01", "slot_index": 0}) is None
    assert descriptor_from_session({"bag_seq": "B01", "descriptor": "{不是 JSON"}) is None