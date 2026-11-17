"""缩放、ROI 窗口与流式上传的用例（task 35）。

四组断言，与 ``task.json`` 的 steps 一一对应：

1. **缩放尺寸**：长边缩到上限、短边有下限保护、原始与缩放后尺寸都记着；
2. **ROI 框随帧平滑**：抖动不放大、突变不跳变、贴边只向外扩不退全幅；
3. **600 / 1200 帧峰值内存**：帧数翻倍而峰值不跟着翻倍（流式那条纪律的硬指标）；
4. **协议不破窗**：ROI 装饰器出来的掩膜仍与整幅规范帧同尺寸同坐标系。
"""

from __future__ import annotations

import io
import json
import tracemalloc
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from infusion_monitor.core.predictor import MaskPrediction, Prompt, mask_bbox
from infusion_monitor.errors import ApiError
from infusion_monitor.ingest.resize import (
    FrameGeometry,
    ResizePlan,
    ResizePolicy,
    ResizeReport,
    RoiBox,
    RoiParams,
    RoiTracker,
    UploadBudget,
    WindowedPredictor,
    plan_resize,
    resize_array,
    resize_frame_file,
    resize_report_for_frames,
)
from infusion_monitor.ingest.validate import MIN_SHORT_SIDE
from infusion_monitor.ingest.zip_frames import extract_frames
from tests.support.synth import default_rack_scene, render_rack_frame

# ── 1. 缩放尺寸 ────────────────────────────────────────────────────────────


def test_long_edge_is_capped_and_both_sizes_are_recorded():
    plan = plan_resize(4000, 3000)
    assert plan.output_size == (1280, 960)
    assert plan.original_size == (4000, 3000)
    assert plan.resized and plan.reason == "long_edge_capped"
    assert plan.scale == pytest.approx(1280 / 4000, rel=1e-6)
    # 原始与缩放后尺寸都在报告里，护士端画回原照片要靠它
    assert plan.as_dict()["camera_width"] == 4000
    assert plan.as_dict()["canonical_width"] == 1280


def test_short_side_floor_protects_extreme_aspect_ratio():
    """4000×800：按长边缩到 1280 短边只剩 256px，跌破门限 → 保短边、长边超额。"""
    plan = plan_resize(4000, 800)
    assert plan.reason == "short_edge_floor_wins"
    assert plan.output_size == (1600, MIN_SHORT_SIDE)
    assert plan.short_side >= MIN_SHORT_SIDE
    assert plan.output_size[0] > 1280  # 长边超额放行，短边守住


def test_frames_below_the_floor_are_passed_through_untouched():
    """短边本就低于门限：不缩、不放，原样交出去让校验/预检去判。

    4000×300 就是源实现那条「只按长边一刀切」会踩的坑：缩到 1280 长边后短边只剩
    96px，比缩之前还不可用。这里一个字节都不动。
    """
    plan = plan_resize(4000, 300)
    assert not plan.resized
    assert plan.output_size == (4000, 300)
    assert plan.reason == "short_edge_below_floor"


def test_disabled_policy_never_resizes():
    plan = plan_resize(4000, 3000, ResizePolicy(enabled=False))
    assert not plan.resized and plan.reason == "disabled"
    assert plan.output_size == (4000, 3000)


def test_small_frames_keep_their_exact_bytes(tmp_path):
    """已经合规的短片一个字节都不重写——过一趟缩放流水线不该掉画质。"""
    source = tmp_path / "frame_00000.jpg"
    Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(source, format="JPEG")
    before = source.read_bytes()
    plan = resize_frame_file(source, ResizePolicy())
    assert not plan.resized
    assert source.read_bytes() == before


def test_resized_frame_keeps_the_liquid_line_and_scales_down(tmp_path):
    """缩放不改变「哪一行是液面」——按缩放比换算回去的行号应落在 2 行之内。"""
    scene = default_rack_scene(frame_count=2, width=1920, height=2560)  # 护士手机原生长边
    frame = render_rack_frame(scene, 0)
    assert frame.shape[:2] == (2560, 1920)
    plan = plan_resize(1920, 2560)  # 长边是 height=2560，缩到 1280
    assert plan.output_size == (960, 1280)
    assert plan.scale == pytest.approx(0.5)
    scaled = resize_array(frame, plan)
    assert scaled.shape[:2] == (1280, 960)
    assert max(scaled.shape[:2]) == 1280

    # 判据刻意与画面宽度无关：数「这一行有几个 B01 药色像素」，门槛固定 3 个。
    # 用「占宽度的比例」当门槛会随缩放一起变，测出来的是门槛漂移而不是液面漂移。
    bag_rgb = np.array(scene.bags[0].colour_at(0), dtype=np.int16)

    def liquid_centroid_row(image: np.ndarray) -> float:
        """液体像素质心的行号。

        **不用「第一行液体」**：弯月面是个很薄的楔形，缩小一半之后最上面那几行只剩
        一两个像素，重采样又把它们抹掉，于是「第一行」测的是抗锯齿而不是液面。
        质心按像素数加权，对这种边缘效应不敏感，缩放前后才可比。
        """
        distance = np.abs(image.astype(np.int16) - bag_rgb).max(axis=2)
        per_row = (distance <= 24).sum(axis=1).astype(np.float64)
        return float((per_row * np.arange(len(per_row))).sum() / per_row.sum())

    row0 = liquid_centroid_row(frame)
    row1 = liquid_centroid_row(scaled)
    assert row1 == pytest.approx(row0 * plan.scale, abs=4)


def test_extract_frames_normalises_on_disk_and_reports_both_sizes(tmp_path):
    """落盘即规范化：解包后的帧已经是规范帧，报告里记着原始尺寸。"""
    scene = default_rack_scene(frame_count=14, width=1920, height=1280)
    zip_path = tmp_path / "frames.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        base = datetime(2026, 10, 22, 9, 41, 0)
        for index in range(scene.frame_count):
            frame = render_rack_frame(scene, index)
            encoded = io.BytesIO()
            Image.fromarray(frame, mode="RGB").save(encoded, format="JPEG", quality=90)
            info = zipfile.ZipInfo(
                f"frame_{index:05d}.jpg", date_time=(base + timedelta(seconds=index)).timetuple()[:6]
            )
            archive.writestr(info, encoded.getvalue())

    dest = tmp_path / "out"
    extracted = extract_frames(zip_path, dest, resize_policy=ResizePolicy())

    with Image.open(dest / "frame_00000.jpg") as on_disk:
        assert (on_disk.width, on_disk.height) == (1280, 853)
    assert extracted.resize is not None
    assert extracted.resize.resized_frames == scene.frame_count
    assert extracted.resize.plan.original_size == (1920, 1280)
    assert extracted.resize.plan.output_size == (1280, 853)
    # 不给策略就一个字节都不动（回归对照：同一条路关掉缩放仍是原图）
    plain = extract_frames(zip_path, tmp_path / "plain", resize_policy=None)
    with Image.open(tmp_path / "plain" / "frame_00000.jpg") as on_disk:
        assert (on_disk.width, on_disk.height) == (1920, 1280)
    assert plain.resize is None


def test_mixed_size_zip_reports_the_first_frame_as_canonical(tmp_path):
    """帧序列本来就该同尺寸；真混了也只按首帧报结论，不假装整段一致。"""
    zip_path = tmp_path / "mixed.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (width, height) in enumerate([(1920, 1280), (640, 480)]):
            buffer = io.BytesIO()
            Image.fromarray(
                np.zeros((height, width, 3), dtype=np.uint8), mode="RGB"
            ).save(buffer, format="JPEG")
            archive.writestr(f"frame_{index:05d}.jpg", buffer.getvalue())
    extracted = extract_frames(zip_path, tmp_path / "mixed", resize_policy=ResizePolicy())
    assert extracted.resize.plan.output_size == (1280, 853)
    assert extracted.resize.resized_frames == 1  # 第二帧已合规，一个字节没碰


# ── 2. ROI 框随帧平滑 ─────────────────────────────────────────────────────


def test_initial_window_covers_bound_candidates_plus_margin():
    tracker = RoiTracker(RoiParams(), (640, 480))
    window = tracker.initial_window([(100, 100, 200, 400), (300, 100, 400, 400)], (640, 480))
    assert window is not None
    margin = round(640 * 0.12)  # 640 长边 × 12% 余量
    assert window.x == 100 - margin
    assert window.x + window.width == 400 + margin
    assert window.height == 300 + 2 * margin  # 框 (100,100)-(400,400) 高 300
    assert window.contains_point(150, 250) and window.contains_point(350, 250)


def test_jittery_boxes_do_not_move_the_window_by_their_own_amplitude():
    """逐帧抖 ±10px 的框，窗口必须跟着 EMA 走而不是跟着抖。"""
    tracker = RoiTracker(RoiParams(alpha=0.35, max_step_ratio=0.05), (640, 480))
    window = tracker.initial_window([(150, 120, 350, 500)], (640, 480))
    assert window is not None
    for index in range(24):
        wobble = 10 if index % 2 else -10
        tracker.advance([(150 + wobble, 120, 350 + wobble, 500)])
    # 逐次位移被 EMA 压到抖动幅度以下，且永远不超过硬上限（640 × 0.05 = 32px）
    assert tracker.state.max_step_px <= 32
    assert tracker.state.max_step_px < 10
    # 抖 24 帧后窗口没跑偏：仍在首帧窗口附近（没有累积成一次漂移）
    assert tracker.window is not None
    assert abs(tracker.window.x - window.x) <= 32


def test_scene_change_cannot_jump_the_window():
    """场景突变时单次位移被硬上限夹住——不夹的话面积曲线上会凭空多一段波谷。"""
    tracker = RoiTracker(RoiParams(alpha=1.0, max_step_ratio=0.05), (640, 480))
    tracker.initial_window([(40, 40, 120, 200)], (640, 480))
    before = tracker.window
    tracker.advance([(400, 400, 600, 620)])
    assert tracker.window is not None and before is not None
    assert tracker.state.max_step_px == 32  # 640 × 0.05
    assert abs(tracker.window.x - before.x) <= 32


def test_mask_touching_the_border_expands_instead_of_shrinking():
    """掩膜贴窗边说明框被切了：只向外扩，不向内收（否则窗口会越收越切）。"""
    tracker = RoiTracker(RoiParams(alpha=1.0, max_step_ratio=1.0), (640, 480))
    window = tracker.initial_window([(150, 120, 350, 500)], (640, 480))
    assert window is not None
    tracker.advance([(window.x, window.y, window.x + 60, window.y + 60)])  # 贴左上角
    assert tracker.state.border_touches == 1
    assert tracker.window is not None
    assert tracker.window.width >= window.width  # 没有被这批残缺框收窄


def test_roi_degrades_to_full_frame_instead_of_guessing():
    for boxes, params, reason in [
        ([], RoiParams(), "no_candidate_boxes"),
        # 余量设 0 才能让 10×10 的框原样成为窗口（默认 12% 余量会把它撑到 163px）
        ([(10, 10, 20, 20)], RoiParams(margin_ratio=0.0), "window_too_small"),
        ([(0, 0, 480, 640)], RoiParams(), "covers_full_frame"),  # 640×480 的整幅画面
    ]:
        tracker = RoiTracker(params, (640, 480))
        assert tracker.initial_window(boxes, (640, 480)) is None
        assert tracker.state.fallback_reason == reason


def test_disabled_roi_records_the_reason():
    tracker = RoiTracker(RoiParams(enabled=False), (640, 480))
    assert tracker.initial_window([(10, 10, 100, 100)], (640, 480)) is None
    assert tracker.state.fallback_reason == "disabled"


def test_window_never_shrinks_while_the_liquid_drains():
    """液面一路下降时窗口不得跟着收——收掉的正是标定要量的腔体上沿。

    这是踩过的坑：初始窗口按候选框开，液体掩膜随后一路变矮，原实现跟着它一路收到
    515 行只剩 318 行（裁掉了袋子上半截）。贴边检测挡不住这条——掩膜是**离开**边框，
    不是贴上。窗口可以平移、可以涨，不得比初值更小。
    """
    tracker = RoiTracker(RoiParams(alpha=0.35, max_step_ratio=0.05), (640, 480))
    window = tracker.initial_window([(150, 100, 350, 460)], (640, 480))
    assert window is not None
    for index in range(24):
        tracker.advance([(150, 100 + 8 * index, 350, 460)])
        current = tracker.window
        assert current is not None
        assert current.width >= window.width, f"第 {index} 帧宽度缩到 {current.width}"
        assert current.height >= window.height, f"第 {index} 帧高度缩到 {current.height}"


def test_window_follows_a_moved_object_without_ever_resizing():
    """整架挪出画面时窗口跟着长过去，但尺寸只增不减。

    相机平移不该让 ROI 失效，但平移也不该让窗口缩水——缩水就是切袋子。
    """
    tracker = RoiTracker(RoiParams(alpha=1.0, max_step_ratio=1.0), (640, 480))
    window = tracker.initial_window([(300, 100, 480, 460)], (640, 480))
    assert window is not None
    tracker.advance([(60, 100, 200, 460)])
    current = tracker.window
    assert current is not None
    assert current.width >= window.width and current.height >= window.height
    assert current.contains_point(130, 280)  # 确实跟到了新位置


def test_window_does_not_swing_between_bags_when_propagating_one_at_a_time():
    """逐对象传播时窗口不能跟着当前这一只袋子重定位。

    四袋挂架上，若轮到最左边那只时窗心偏左、轮到最右边那只时又偏右，每只袋子都会在
    自己那一半时间里被切一刀。窗口罩住整架就不动；只有观测真的掉到窗外才整体跟过去。
    """
    boxes = [(32, 384, 148, 430), (179, 387, 296, 430), (326, 392, 442, 430)]
    tracker = RoiTracker(RoiParams(alpha=1.0, max_step_ratio=1.0), (640, 480))
    window = tracker.initial_window(boxes, (640, 480))
    assert window is not None
    # 逐对象地轮着传，每一帧只看到一只袋子
    for _ in range(4):
        for box in boxes:
            tracker.advance([box])
    current = tracker.window
    assert current is not None
    for box in boxes:
        assert current.contains_point(*[(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]), (
            f"窗口 {current.as_dict()} 把 {box} 切出去了"
        )


# ── 4. 协议不破窗：ROI 装饰器仍满足预测器契约 ─────────────────────────────


class _FixedMaskPredictor:
    """「点落在框里就画出这个框」的最小实现。

    框坐标**按拿到的画面**解释——真实预测器就是这样工作的（它不知道自己被裁过）。
    把它按整幅图坐标写死，测的就不是「裁剪+贴回」而是「谁记得自己被裁了」，两回事。
    """

    def __init__(self, box: tuple[int, int, int, int]):
        self.box = box
        self.seen_shapes: list[tuple[int, int]] = []

    def predict(self, image, prompt: Prompt):
        self.seen_shapes.append(tuple(image.shape[:2]))
        point = prompt.points[0]
        x0, y0, x1, y1 = self.box
        if not (x0 <= point.x < x1 and y0 <= point.y < y1):
            return []
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[y0:y1, x0:x1] = True
        return [MaskPrediction(mask=mask, score=0.75)]


def test_windowed_predictor_pastes_masks_back_to_full_frame():
    tracker = RoiTracker(RoiParams(margin_ratio=0.0), (480, 640))
    window = tracker.initial_window([(100, 100, 300, 400)], (480, 640))
    assert window is not None
    # 提示点给整幅坐标 (140, 200)：装饰器应把它平移进裁剪窗 (40, 100) 后再问内层
    inner = _FixedMaskPredictor((40, 40, 160, 260))
    windowed = WindowedPredictor(inner, tracker)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    predictions = windowed.predict(frame, Prompt.from_point(140, 200))

    assert len(predictions) == 1
    # 掩膜与整幅规范帧同尺寸（协议第 2 条），且贴回了正确的位置
    assert predictions[0].mask.shape == (480, 640)
    assert predictions[0].score == 0.75
    assert mask_bbox(predictions[0].mask) == (140, 140, 260, 360)
    # 内层拿到的是裁剪后的画面（少喂了无关区域），坐标也同步平移了
    assert inner.seen_shapes == [(300, 200)]


def test_windowed_predictor_clips_a_stale_window_instead_of_mis_sizing_masks():
    """帧尺寸变了（混尺寸 zip）时宁可退整帧，也不能让掩膜与画面对不上。"""
    tracker = RoiTracker(RoiParams(margin_ratio=0.0), (480, 640))
    tracker.initial_window([(100, 100, 300, 400)], (480, 640))
    inner = _FixedMaskPredictor((40, 40, 160, 260))  # 裁剪窗内的坐标
    windowed = WindowedPredictor(inner, tracker)
    # 下一帧变成 480×320：按旧尺寸开的窗在宽度上越界
    smaller = np.zeros((480, 320, 3), dtype=np.uint8)
    predictions = windowed.predict(smaller, Prompt.from_point(140, 200))
    assert predictions[0].mask.shape == (480, 320)
    assert mask_bbox(predictions[0].mask) == (140, 140, 260, 360)


def test_windowed_predictor_falls_back_to_full_frame_without_a_window():
    tracker = RoiTracker(RoiParams(), (480, 640))
    inner = _FixedMaskPredictor((140, 140, 260, 360))  # 全幅坐标
    windowed = WindowedPredictor(inner, tracker)
    predictions = windowed.predict(np.zeros((480, 640, 3), dtype=np.uint8), Prompt.from_point(200, 250))
    assert inner.seen_shapes == [(480, 640)]
    assert predictions[0].mask.shape == (480, 640)
    assert tracker.state.applied_calls == 0


def test_windowed_predictor_satisfies_the_structural_half_of_the_contract():
    """契约套件（task 24）的**结构那几条**在 ROI 装饰器上再跑一遍：换包装不换协议。

    只跑结构/表面这几条，不跑 ``determinism``：ROI 装饰器**有状态**（窗口逐帧平滑），
    同一提示连问两次裁剪窗可能已经挪了半格，输出因此可以不同——这是设计，不是违约。
    真正要守住的是「掩膜与整幅规范帧同尺寸、dtype 是 bool、score 在 0–1、找不到目标
    返回空列表」这四条，它们一旦破了，面积比与毫升数就在错的坐标系上算了。
    """
    from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
    from infusion_monitor.core.predictor_contract import (
        check_empty_target_returns_empty_list,
        check_predictor_surface,
        check_prediction_structure,
        contract_cases,
        plain_scene,
    )

    tracker = RoiTracker(RoiParams(margin_ratio=0.0), (240, 320))
    tracker.initial_window([(30, 20, 290, 200)], (240, 320))
    wrapped = WindowedPredictor(GeometricMaskPredictor(), tracker)
    assert tracker.window is not None, "本用例需要一个真的开起来的窗口"

    check_predictor_surface(wrapped)
    for case in contract_cases():
        check_prediction_structure(wrapped, case)
    check_empty_target_returns_empty_list(wrapped, plain_scene())


def test_estimate_cavity_is_delegated_unchanged():
    """袋腔在整幅图上量与在窗内量是同一片像素——转发比裁一遍更不容易错。"""
    tracker = RoiTracker(RoiParams(), (480, 640))
    tracker.initial_window([(100, 100, 300, 400)], (480, 640))
    inner = _FixedMaskPredictor((140, 140, 260, 360))  # 全幅坐标
    windowed = WindowedPredictor(inner, tracker)
    assert hasattr(windowed, "estimate_cavity") is False  # 内层没有就不假装有

    class _WithCavity(_FixedMaskPredictor):
        def estimate_cavity(self, image, prompt):
            self.seen_shapes.append(tuple(image.shape[:2]))
            return type("Cavity", (), {"area_px": 1234})()

    tracker2 = RoiTracker(RoiParams(), (480, 640))
    tracker2.initial_window([(100, 100, 300, 400)], (480, 640))
    inner2 = _WithCavity((140, 140, 260, 360))
    wrapped2 = WindowedPredictor(inner2, tracker2)
    assert wrapped2.estimate_cavity(np.zeros((480, 640, 3), dtype=np.uint8), Prompt.from_point(1, 1)).area_px == 1234
    assert inner2.seen_shapes == [(480, 640)]  # 没被裁


def test_mask_bbox_of_a_pasted_prediction_stays_in_canonical_coordinates():
    tracker = RoiTracker(RoiParams(margin_ratio=0.0), (480, 640))
    tracker.initial_window([(100, 100, 300, 400)], (480, 640))
    windowed = WindowedPredictor(_FixedMaskPredictor((40, 40, 160, 260)), tracker)
    predictions = windowed.predict(
        np.zeros((480, 640, 3), dtype=np.uint8), Prompt.from_point(140, 200)
    )
    assert mask_bbox(predictions[0].mask) == (140, 140, 260, 360)


def test_frame_geometry_maps_canonical_points_back_to_camera_pixels():
    plan = plan_resize(4000, 3000)
    geometry = FrameGeometry(
        camera_size=plan.original_size, canonical_size=plan.output_size, plan=plan
    )
    assert geometry.to_camera_point(320, 240) == pytest.approx((1000, 750))
    assert geometry.to_camera_box((320, 240, 640, 480)) == pytest.approx((1000, 750, 2000, 1500))


# ── 3. 流式：600 / 1200 帧峰值内存 ─────────────────────────────────────────


def _build_clip(directory: Path, scene, *, start_index: int = 0) -> Path:
    zip_path = directory / f"clip-{scene.frame_count}.zip"
    base = datetime(2026, 10, 22, 9, 41, 0)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(scene.frame_count):
            frame = render_rack_frame(scene, index)
            encoded = io.BytesIO()
            Image.fromarray(frame, mode="RGB").save(encoded, format="JPEG", quality=80)
            info = zipfile.ZipInfo(
                f"frame_{start_index + index:05d}.jpg",
                date_time=(base + timedelta(seconds=index)).timetuple()[:6],
            )
            archive.writestr(info, encoded.getvalue())
    return zip_path


@pytest.mark.parametrize(("frames", "budget_mb"), [(600, 24), (1200, 32)])
def test_peak_memory_does_not_grow_with_frame_count(tmp_path, frames, budget_mb):
    """帧数翻倍而峰值内存不同量级地涨——这是流式那条纪律的硬指标。

    阈值按实测留了两倍余量：1200 帧 × 320×480 的解包峰值约 16MB。
    """
    scene = default_rack_scene(frame_count=frames)
    zip_path = _build_clip(tmp_path, scene)
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        extracted = extract_frames(zip_path, tmp_path / f"out-{frames}", resize_policy=ResizePolicy())
        peak = tracemalloc.get_traced_memory()[1] - baseline
    finally:
        tracemalloc.stop()
    assert extracted.frame_count == frames
    assert peak < budget_mb * 1024 * 1024, f"{frames} 帧峰值 {peak / 1024 / 1024:.1f}MB 超预算"


def test_upload_budget_rejects_oversized_packets_with_a_resume_pointer():
    budget = UploadBudget(max_bytes=1000, max_seconds=None)
    with pytest.raises(ApiError) as caught:
        budget.check(1500, 0.1)
    assert caught.value.error_code == "UPLOAD_BUDGET_EXCEEDED"
    assert caught.value.status_code == 413


def test_upload_budget_rejects_slow_packets_with_a_resume_pointer():
    budget = UploadBudget(max_bytes=None, max_seconds=2.0)
    with pytest.raises(ApiError) as caught:
        budget.check(10, 2.5)
    assert caught.value.error_code == "UPLOAD_TIMEOUT"
    assert caught.value.status_code == 408


def test_budget_error_docs_point_at_the_resume_endpoints():
    from infusion_monitor.errors import error_code_docs

    docs = {item.code: item for item in error_code_docs()}
    for code in ("UPLOAD_BUDGET_EXCEEDED", "UPLOAD_TIMEOUT"):
        assert code in docs
        assert "/api/v1/infusion/uploads" in docs[code].handling
    assert docs["UPLOAD_BUDGET_EXCEEDED"].status == 413
    assert docs["UPLOAD_TIMEOUT"].status == 408


# ── 端到端：响应里真的带着 geometry 与 roi ────────────────────────────────


def _post_clip(client, scene, **kwargs):
    from tests.support.synth import build_rack_zip

    slots = json.dumps(
        [
            {"bag_seq": bag.bag_seq, "slot_index": index, "bag_type_code": "TYPE-A"}
            for index, bag in enumerate(scene.bags)
        ],
        ensure_ascii=False,
    )
    return client.post(
        "/api/v1/infusion/tracks",
        data={
            "rack_id": "R-EXAMPLE-35",
            "infusion_order_no": "EXAMPLE-ORDER-0035",
            "bed_no": "EXAMPLE-BED-35",
            "captured_at": "2026-10-22T09:41:00+08:00",
            "slots": slots,
        },
        files={"frames": ("frames.zip", build_rack_zip(scene, **kwargs), "application/zip")},
    )


def test_upload_response_reports_geometry_and_roi(client):
    scene = default_rack_scene(frame_count=16)
    response = _post_clip(client, scene)
    assert response.status_code == 201
    payload = response.json()

    assert payload["geometry"]["camera_width"] == 480
    assert payload["geometry"]["canonical_width"] == 480  # 默认短片不缩
    assert payload["geometry"]["scale"] == 1.0
    assert payload["geometry"]["resized"] is False
    assert payload["geometry"]["total_frames"] == 16

    # 首帧认出了袋子 → 窗口开得起来；认不出就退全幅并说明理由，两种都要如实报
    assert payload["roi"]["enabled"] is True
    if payload["roi"]["applied"]:
        assert payload["roi"]["window"] is not None
        assert payload["roi"]["applied_calls"] > 0
        # 窗口只涨不缩：末态尺寸不小于初值，信息科据此可核对这条不变量
        window = payload["roi"]["window"]
        assert window["width"] >= payload["roi"]["anchor_width"]
        assert window["height"] >= payload["roi"]["anchor_height"]
        assert payload["roi"].get("fallback_reason") is None
    else:
        assert payload["roi"]["fallback_reason"] in (
            "no_candidate_boxes",
            "window_too_small",
            "covers_full_frame",
        )
    # 曲线照给：ROI 只是少喂了画面，不改坐标、不改数字
    for bag in payload["bags"]:
        assert len(bag["area_ratio_series"]) == 16


def test_upload_response_geometry_survives_a_big_clip(client):
    """原生长边 1920 的一段：落盘即缩到 1280，响应同时给出两套尺寸。"""
    scene = default_rack_scene(frame_count=16, width=1920, height=1280)
    response = _post_clip(client, scene)
    assert response.status_code == 201
    payload = response.json()
    assert payload["geometry"]["camera_width"] == 1920
    assert payload["geometry"]["camera_height"] == 1280
    assert payload["geometry"]["canonical_width"] == 1280
    assert payload["geometry"]["canonical_height"] == 853
    assert payload["geometry"]["resized"] is True
    assert payload["geometry"]["resized_frames"] == 16
    assert payload["geometry"]["reason"] == "long_edge_capped"
    # 未标定袋型的门禁语义不受影响
    for bag in payload["bags"]:
        assert bag["calibration_state"] == "uncalibrated"
        assert "estimated_volume_ml" not in bag


def test_session_row_records_frame_geometry(store_client):
    client, observer = store_client
    scene = default_rack_scene(frame_count=14, width=1920, height=1280)
    response = _post_clip(client, scene)
    assert response.status_code == 201
    session_id = response.json()["bags"][0]["session_id"]

    row = observer.execute(
        "SELECT camera_width, camera_height, canonical_width, canonical_height, frame_scale, "
        "roi_window_json FROM bag_session WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    assert (row["camera_width"], row["camera_height"]) == (1920, 1280)
    assert (row["canonical_width"], row["canonical_height"]) == (1280, 853)
    assert row["frame_scale"] == pytest.approx(1280 / 1920, rel=1e-6)
    assert json.loads(row["roi_window_json"]) in (None, {}) or "x" in (
        json.loads(row["roi_window_json"]) or {"x": 0}
    )


def test_chunked_upload_produces_the_same_numbers_as_a_whole_clip(credential_env, tmp_path):
    """分 3 片传与整段传走同一套缩放策略，因此给同一份数字。"""
    from fastapi.testclient import TestClient

    from infusion_monitor.api.service import create_app
    from infusion_monitor.config import Settings
    from tests.support.synth import build_rack_zip

    import zipfile as _zipfile

    scene = default_rack_scene(frame_count=16, width=1920, height=1280)
    payload = build_rack_zip(scene)
    slots = json.dumps(
        [{"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-A"},
         {"bag_seq": "B02", "slot_index": 1, "bag_type_code": "TYPE-A"}],
        ensure_ascii=False,
    )

    def _client(tag: str) -> TestClient:
        app = create_app(
            Settings(),
            database_url=f"sqlite:///{tmp_path / f'{tag}.db'}",
            frames_root=tmp_path / f"frames-{tag}",
        )
        return TestClient(app)

    head = {
        "rack_id": "R-EXAMPLE-35",
        "infusion_order_no": "EXAMPLE-ORDER-0035",
        "bed_no": "EXAMPLE-BED-35",
        "captured_at": "2026-10-22T09:41:00+08:00",
        "slots": slots,
    }

    with _client("whole") as whole:
        whole_response = whole.post(
            "/api/v1/infusion/tracks",
            data=head,
            files={"frames": ("frames.zip", payload, "application/zip")},
        )
        assert whole_response.status_code == 201

    with _client("chunks") as chunked:
        opened = chunked.post(
            "/api/v1/infusion/uploads", data={**head, "expected_frames": "16"}
        )
        assert opened.status_code == 201
        upload_id = opened.json()["upload_id"]

        source = _zipfile.ZipFile(io.BytesIO(payload))
        bounds = [(0, 6), (6, 11), (11, 16)]
        for low, high in bounds:
            buffer = io.BytesIO()
            with _zipfile.ZipFile(buffer, "w", compression=_zipfile.ZIP_DEFLATED) as piece:
                for index in range(low, high):
                    # 连 ZipInfo 一起搬：成员时间戳是**这一段唯一的拍摄时长证据**
                    # （时间序校验靠它），重新 writestr(名字, 字节) 会把所有帧的时间
                    # 抹成同一刻，提交时就会被 frame_times_flat 挡下。
                    info = source.getinfo(f"frame_{index:05d}.jpg")
                    piece.writestr(info, source.read(info.filename))
            ack = chunked.post(
                f"/api/v1/infusion/uploads/{upload_id}/chunks",
                files={"chunk": ("chunk.zip", buffer.getvalue(), "application/zip")},
            )
            assert ack.status_code == 200
        committed = chunked.post(f"/api/v1/infusion/uploads/{upload_id}/commit")
        assert committed.status_code == 201

    whole_body = whole_response.json()
    chunk_body = committed.json()
    assert chunk_body["geometry"] == whole_body["geometry"]
    for left, right in zip(whole_body["bags"], chunk_body["bags"]):
        assert left["area_ratio_series"] == right["area_ratio_series"]
        assert left["fill_ratio"] == right["fill_ratio"]


def test_report_for_frames_does_not_decode_pixels(tmp_path):
    scene = default_rack_scene(frame_count=2)
    paths = []
    for index in range(2):
        path = tmp_path / f"frame_{index:05d}.jpg"
        Image.fromarray(render_rack_frame(scene, index), mode="RGB").save(path, format="JPEG")
        paths.append(path)
    report = resize_report_for_frames(paths)
    assert isinstance(report, ResizeReport)
    assert report.plan.output_size == (480, 640)
    assert report.resized_frames == 0 and report.total_frames == 2


def test_resize_report_identity_helper_round_trips():
    plan = ResizePlan.identity(640, 480)
    assert plan.resized is False
    assert plan.as_dict()["canonical_width"] == 640
    assert RoiBox.full((480, 640)).is_full((480, 640))


# ── 5. ROI 只省算力，不改数字 ───────────────────────────────────────────
#
# 这一组是本任务最硬的约束：**开着 ROI 与关着 ROI 必须给出同一份数字**。
# ROI 少喂的是画面，不是坐标——面积比、余量、ETA 全都挂在同一条曲线上，窗开错一点，
# 曲线就歪一点，而歪掉的量会一路传到毫升数与换液点名。


def _post_with_settings(credential_env, tmp_path, tag, scene, *, roi_enabled):
    from fastapi.testclient import TestClient

    from infusion_monitor.api.service import create_app
    from infusion_monitor.config import Settings
    from tests.support.synth import build_rack_zip

    app = create_app(
        Settings(roi_enabled=roi_enabled),
        database_url=f"sqlite:///{tmp_path / f'{tag}.db'}",
        frames_root=tmp_path / f"frames-{tag}",
    )
    with TestClient(app) as test_client:
        slots = json.dumps(
            [
                {"bag_seq": bag.bag_seq, "slot_index": index, "bag_type_code": "TYPE-A"}
                for index, bag in enumerate(scene.bags)
            ],
            ensure_ascii=False,
        )
        response = test_client.post(
            "/api/v1/infusion/tracks",
            data={
                "rack_id": "R-EXAMPLE-35",
                "infusion_order_no": "EXAMPLE-ORDER-0035",
                "bed_no": "EXAMPLE-BED-35",
                "captured_at": "2026-10-22T09:41:00+08:00",
                "slots": slots,
            },
            files={"frames": ("frames.zip", build_rack_zip(scene), "application/zip")},
        )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize("frame_count", [16, 24])
def test_roi_on_and_off_give_the_same_curve(credential_env, tmp_path, frame_count):
    """两只袋子、两种帧数：两条路逐点相同（开窗省算力，不许改数）。"""
    scene = default_rack_scene(frame_count=frame_count)
    with_roi = _post_with_settings(credential_env, tmp_path, "roi-on", scene, roi_enabled=True)
    without = _post_with_settings(credential_env, tmp_path, "roi-off", scene, roi_enabled=False)
    assert with_roi["roi"]["enabled"] is True
    assert without["roi"]["applied"] is False
    assert [b["area_ratio_series"] for b in with_roi["bags"]] == [
        b["area_ratio_series"] for b in without["bags"]
    ]
    assert [b["fill_ratio"] for b in with_roi["bags"]] == [b["fill_ratio"] for b in without["bags"]]
    # 帧几何与袋型标定门禁当然也不受开关影响
    assert with_roi["geometry"] == without["geometry"]
    assert all(bag["calibration_state"] == "uncalibrated" for bag in with_roi["bags"])


def _roi_keeps_the_whole_mask(windowed, frame, prompt, expected_mask, tolerance=0.005) -> bool:
    """窗内预测的掩膜与整幅预测一致（比掩膜面积，最能说明有没有被切）。"""
    full = int(expected_mask.sum())
    cropped = windowed.predict(frame, prompt)[0].mask
    return abs(int(cropped.sum()) - full) <= max(4, int(full * tolerance))


def test_a_slot_prior_landing_between_two_bags_still_anchors_the_right_one():
    """槽位先验点落在两只袋子中间的空档上时，ROI 不得把锚锁到窗外那条窄缝上。

    实测过的坏法：窗外那只袋子只剩贴着窗边 4px 的一条，几何核 nearest-component 选中它，
    面积比从 0.59 一路歪到 0.02 再爬回 0.54——袋子的 ml 与 ETA 一起废掉。
    窗口开大只多算几个像素，所以初值取**全部**候选框而不是只取认领上的那几个。
    """
    from infusion_monitor.core.enumerate import enumerate_candidates
    from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
    from infusion_monitor.ingest.multipart import slot_seed_prompt

    scene = default_rack_scene(frame_count=20)
    frame = render_rack_frame(scene, 0)
    assert frame.shape[:2] == (640, 480)
    predictor = GeometricMaskPredictor()
    # 一个槽位：枚举器认不出归属，两个候选都是 unbound——窗口不能只取 bound 的
    enumeration = enumerate_candidates(frame, predictor, slot_count=1)
    assert enumeration.candidates and not any(c.bound for c in enumeration.candidates)

    prior = slot_seed_prompt(frame.shape, 0, 1)
    full_mask = predictor.predict(frame, prior)[0].mask

    tracker = RoiTracker(RoiParams())
    window = tracker.initial_window([c.bbox for c in enumeration.candidates], frame.shape[:2])
    assert window is not None, "认不出归属的候选也必须撑起窗口"
    # 窗口罩住两条候选：先验点再落空档，几何核仍按整幅画面认到同一只袋子
    for candidate in enumeration.candidates:
        assert window.contains_point(candidate.centre_x, candidate.centre_y)
    assert _roi_keeps_the_whole_mask(
        WindowedPredictor(predictor, tracker), frame, prior, full_mask
    )