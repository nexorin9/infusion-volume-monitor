"""M2 主路径内核测试：预测器协议 / 确定性几何核 / 逐帧传播。

覆盖三类断言：

1. **身份**：同一袋跨帧 id 不变，遮挡后接回同一袋，不换轨也不新建轨迹；
2. **几何**：面积比曲线对得上渲染真值；反光与袋体形变两类拍摄变体下容差可控；
3. **碎片语义**：孔洞填充与孤岛剔除两条阈值各自独立，且「液面低至 2% 的细窄连通域」
   不会被当成碎斑删掉。

用到的画面是**合成渲染帧**（``tests/support/synth.py``），不是真实院内短片——真实短片
需医院授权采集，本轮以合成 + 公开 VOS 子集替代。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from infusion_monitor.core.geom_kernel import (
    GeometricMaskPredictor,
    fill_small_holes,
    liquid_candidates,
    remove_small_sprinkles,
)
from infusion_monitor.core.predictor import (
    MaskPrediction,
    PerPointMaskPredictor,
    Point,
    Prompt,
)
from infusion_monitor.core.propagate import (
    add_new_points_or_box,
    analyse_frames,
    init_state,
    load_rgb,
    propagate_in_video,
    slot_seed_prompt,
)
from tests.support.synth import (
    BagSpec,
    RackScene,
    build_rack_zip,
    default_rack_scene,
    write_rack_frames,
)

FRAME_COUNT = 18


@pytest.fixture(scope="module")
def predictor() -> GeometricMaskPredictor:
    return GeometricMaskPredictor()


def _seeds(scene: RackScene, shape):
    return [
        (bag.bag_seq, slot_seed_prompt(shape, slot, len(scene.bags)))
        for slot, bag in enumerate(scene.bags)
    ]


def _run(tmp_path: Path, scene: RackScene, predictor, **variant) -> dict:
    paths = write_rack_frames(tmp_path, scene, **variant)
    shape = load_rgb(paths[0]).shape
    return {curve.obj_id: curve for curve in analyse_frames(paths, _seeds(scene, shape), predictor)}


def _errors(curve, scene: RackScene) -> list[float]:
    return [
        abs(ratio - scene.fill_at(curve.obj_id, index))
        for index, ratio in enumerate(curve.area_ratio_series)
        if ratio is not None
    ]


# ── 协议 ────────────────────────────────────────────────────────────────
def test_predictor_satisfies_protocol(predictor):
    assert isinstance(predictor, PerPointMaskPredictor)
    assert predictor.name == "geom_kernel"


def test_predict_batch_matches_single_predict(tmp_path, predictor):
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    paths = write_rack_frames(tmp_path, scene)
    image = load_rgb(paths[0])
    prompts = [slot_seed_prompt(image.shape, slot, len(scene.bags)) for slot in range(len(scene.bags))]
    batched = predictor.predict_batch(image, prompts)
    single = [predictor.predict(image, prompt) for prompt in prompts]
    assert len(batched) == len(single)
    for left, right in zip(batched, single):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert np.array_equal(a.mask, b.mask)
            assert a.score == pytest.approx(b.score)


def test_prediction_stays_in_input_coordinate_system(tmp_path, predictor):
    """掩膜必须与原图同尺寸——坐标一平移，点/框提示就指错地方了。"""
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    paths = write_rack_frames(tmp_path, scene)
    image = load_rgb(paths[0])
    prompt = slot_seed_prompt(image.shape, 0, len(scene.bags))
    prediction = predictor.predict(image, prompt)[0]
    assert prediction.mask.shape == image.shape[:2]
    assert prediction.mask.dtype == np.bool_


def test_returns_empty_instead_of_guessing(predictor):
    """画面里根本没有袋体时返回空列表，而不是硬选一块区域。"""
    blank = np.full((640, 480, 3), 24, dtype=np.uint8)
    assert predictor.predict(blank, Prompt.from_point(240.0, 500.0)) == []


# ── 身份 ────────────────────────────────────────────────────────────────
def test_identity_stable_across_frames(tmp_path, predictor):
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    curves = _run(tmp_path, scene, predictor)
    assert set(curves) == {"B01", "B02"}
    # 两袋液位下降速度不同：若传播把两袋串了号，曲线会重合或对调
    assert curves["B01"].area_ratio_series != curves["B02"].area_ratio_series
    # B02 起点更高、下降更慢：末帧仍是 B02 高。串号的话这里会反过来。
    assert curves["B02"].latest_ratio > curves["B01"].latest_ratio
    for curve in curves.values():
        assert curve.obj_id in {"B01", "B02"}
        assert curve.occluded_frame_count == 0


def test_occluded_frames_are_marked_not_dropped(tmp_path, predictor):
    """遮挡帧显式标记、面积比留空，序列长度不变，恢复后接回同一袋。"""
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    curves = _run(tmp_path, scene, predictor, occluders=[(0, 5, 8)])
    bag = curves["B01"]

    assert len(bag.area_ratio_series) == scene.frame_count
    assert bag.frame_indices == list(range(scene.frame_count))
    assert [i for i, flag in enumerate(bag.occlusion_flags) if flag] == [5, 6, 7]
    assert [bag.area_ratio_series[i] for i in (5, 6, 7)] == [None, None, None]
    assert bag.occluded_frame_count == 3
    # 恢复后仍是同一条轨迹（同一 obj_id），且重新给出了面积比
    assert bag.area_ratio_series[8] is not None
    assert bag.area_ratio_series[-1] is not None
    # 另一袋不受遮挡影响
    assert curves["B02"].occluded_frame_count == 0


def test_propagate_yields_every_frame(tmp_path, predictor):
    """逐帧产出：帧数与输入一致，未起锚的帧不产出掩膜。"""
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    paths = write_rack_frames(tmp_path, scene)
    first = load_rgb(paths[0])
    state = init_state([path.name for path in paths], first.shape)
    cavity = predictor.estimate_cavity(first, slot_seed_prompt(first.shape, 0, len(scene.bags)))
    add_new_points_or_box(
        state,
        frame_index=0,
        obj_id="B01",
        prompt=slot_seed_prompt(first.shape, 0, len(scene.bags)),
        cavity_area_px=cavity.area_px,
    )
    by_name = {path.name: path for path in paths}
    results = list(propagate_in_video(state, predictor, lambda name: load_rgb(by_name[name])))
    assert [result.frame_index for result in results] == list(range(FRAME_COUNT))
    assert all("B01" in result.masks for result in results)
    assert all(result.masks["B01"].shape == first.shape[:2] for result in results)


# ── 几何容差 ────────────────────────────────────────────────────────────
def test_area_ratio_tracks_rendered_truth(tmp_path, predictor):
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    curves = _run(tmp_path, scene, predictor)
    for curve in curves.values():
        errors = _errors(curve, scene)
        assert errors, "曲线不应整段为空"
        assert max(errors) < 0.06, f"{curve.obj_id} 偏离真值 {max(errors):.4f}"


def test_reflection_highlight_does_not_dip_the_curve(tmp_path, predictor):
    """镜面高光在液体掩膜上打出一个孔洞，孔洞填充应把它补回来。"""
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    clean = _run(tmp_path / "clean", scene, predictor)
    glary = _run(tmp_path / "glare", scene, predictor, highlight=True)
    for bag_seq in ("B01", "B02"):
        baseline = clean[bag_seq].area_ratio_series
        glared = glary[bag_seq].area_ratio_series
        assert len(glared) == len(baseline)
        deviation = max(abs(a - b) for a, b in zip(baseline, glared))
        assert deviation < 0.02, f"{bag_seq} 反光造成的偏差 {deviation:.4f} 过大"
        assert max(_errors(glary[bag_seq], scene)) < 0.06


def test_deformation_keeps_identity_and_acceptable_error(tmp_path, predictor):
    """袋体逐帧摆动：身份不串、曲线仍近似单调、误差在容差内。"""
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    curves = _run(tmp_path, scene, predictor, deform=1.0)
    assert set(curves) == {"B01", "B02"}
    for curve in curves.values():
        visible = [r for r in curve.area_ratio_series if r is not None]
        assert max(_errors(curve, scene)) < 0.12
        # 输液液面物理上不增：允许极小的观测抖动，不允许整体上升
        # 允许 0.03 以内的观测抖动（袋体摆动本来就会让液面像素数上下浮动），
        # 但串号或跟丢新目标会带来远大于此的跳变
        increases = [b - a for a, b in zip(visible, visible[1:]) if b - a > 0.03]
        assert not increases, f"{curve.obj_id} 曲线在形变下出现整体上升：{increases}"


def test_dark_scene_still_tracked(tmp_path, predictor):
    """夜间暗光（亮度整体压到 55%）不应让几何核失效。"""
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    curves = _run(tmp_path, scene, predictor, dark=0.55)
    for curve in curves.values():
        assert curve.occluded_frame_count == 0
        assert max(_errors(curve, scene)) < 0.06


def test_low_fill_two_percent_not_treated_as_sprinkle(tmp_path, predictor):
    """液面低至 2% 时那袋本身是一条细窄连通域，不得被当成碎斑删掉。"""
    scene = RackScene(
        bags=(BagSpec("B01", 0.5, 0.30, 0.20, 0.02, (92, 148, 196)),),
        frame_count=FRAME_COUNT,
    )
    curves = _run(tmp_path, scene, predictor)
    curve = curves["B01"]
    assert curve.occluded_frame_count == 0
    assert curve.latest_ratio is not None
    assert abs(curve.latest_ratio - 0.02) < 0.02
    assert max(_errors(curve, scene)) < 0.03


# ── 碎片后处理：孔洞 / 孤岛两条阈值各自独立 ──────────────────────────────
def test_hole_threshold_fills_only_small_holes():
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:50, 10:50] = True
    mask[20:24, 20:24] = False  # 16px² 的小孔
    mask[30:45, 30:45] = False  # 225px² 的大孔
    filled = fill_small_holes(mask, max_area=64)
    assert filled[22, 22], "小孔应被填上"
    assert not filled[32, 32], "大孔不该被填上"
    assert fill_small_holes(mask, max_area=0).tolist() == mask.tolist(), "阈值 0 相当于关闭"


def test_sprinkle_threshold_drops_sprinkles_but_not_prompted_object():
    mask = np.zeros((80, 120), dtype=bool)
    mask[10:70, 10:70] = True  # 主块，3600px²
    mask[30:33, 100:104] = True  # 9px² 的碎斑
    dropped = remove_small_sprinkles(mask, max_area=32)
    assert dropped[50, 50], "主块不该被动"
    assert not dropped[31, 102], "碎斑应被删"

    # 阈值压到连主块都够不着时，靠提示点把真对象压住——这正是相对源实现
    # 「删光全部小连通域」的有意偏离：液面低至 2% 时那袋本身就很小
    protected_sprinkle = remove_small_sprinkles(mask, max_area=4, protected=((101.5, 31.5),))
    assert protected_sprinkle[31, 102], "压住提示点的连通域不该被删"
    protected_object = remove_small_sprinkles(mask, max_area=16, protected=((50.0, 50.0),))
    assert protected_object[50, 50], "压住的主对象不该被删"
    assert not protected_object[31, 102], "没被压住的碎斑仍应删掉"


def test_sprinkle_and_hole_thresholds_are_independent():
    """调一个阈值不改变另一个的行为（源实现里两者是各自独立的开关）。"""
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:50, 10:50] = True
    mask[20:23, 20:23] = False   # 9px² 的小孔
    mask[30:45, 30:45] = False   # 225px² 的大孔
    mask[52:54, 52:54] = True    # 4px² 的独立碎斑

    # 开孔洞填充：小孔填上；关孤岛剔除：碎斑留在
    holes_filled = fill_small_holes(mask, 64)
    assert holes_filled[21, 21] and not holes_filled[32, 32]
    assert holes_filled[52, 52], "孔洞填充不该动独立碎斑"

    # 开孤岛剔除：碎斑删掉；孔洞保持原样（孔洞是背景，不是连通域）
    sprinkles_removed = remove_small_sprinkles(mask, 64)
    assert not sprinkles_removed[52, 52]
    assert not sprinkles_removed[21, 21] and not sprinkles_removed[32, 32]

    # 调孔洞阈值（0 → 8，两值都不足以填 225px² 的大孔）不改变孤岛行为
    assert remove_small_sprinkles(fill_small_holes(mask, 0), 64).tolist() == (
        remove_small_sprinkles(fill_small_holes(mask, 8), 64).tolist()
    )
    # 调孤岛阈值：阈值 64 时 4px² 的碎斑被删、阈值 2 时留下——而孔洞的处理只取决于
    # max_hole_area，不受孤岛阈值影响
    sprinkled = remove_small_sprinkles(mask, 64)
    spared = remove_small_sprinkles(mask, 2)
    assert not sprinkled[52, 52] and spared[52, 52]
    assert fill_small_holes(sprinkled, 64)[21, 21] and fill_small_holes(spared, 64)[21, 21]
    assert not fill_small_holes(sprinkled, 64)[32, 32]
    assert not fill_small_holes(spared, 64)[32, 32]


def test_liquid_candidates_separates_liquid_from_bag_body(tmp_path, predictor):
    """阈值带要能把液体（高饱和）与透明袋体（低饱和）分开。"""
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    paths = write_rack_frames(tmp_path, scene)
    image = load_rgb(paths[0])
    candidates = liquid_candidates(image, predictor.params)
    body = np.zeros(image.shape[:2], dtype=bool)
    body[: image.shape[0] // 3] = True  # 袋体上部（透明、几乎无色）
    assert candidates[body].mean() < 0.02, "袋体上部不该被判成液体"


# ── 接口层：一次上传拿到每袋曲线 ──────────────────────────────────────────
def test_tracks_response_carries_area_ratio_series(client):
    payload = build_rack_zip(default_rack_scene(frame_count=FRAME_COUNT))
    response = client.post(
        "/api/v1/infusion/tracks",
        data={
            "rack_id": "R-EXAMPLE-07",
            "infusion_order_no": "EXAMPLE-ORDER-0001",
            "bed_no": "EXAMPLE-BED-07",
            "captured_at": "2026-10-22T09:41:00+08:00",
            "slots": '[{"bag_seq": "B01", "slot_index": 0}, {"bag_seq": "B02", "slot_index": 1}]',
        },
        files={"frames": ("frames.zip", payload, "application/zip")},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert len(body["bags"]) == 2
    for bag in body["bags"]:
        assert bag["predictor"] == "geom_kernel"
        assert len(bag["area_ratio_series"]) == FRAME_COUNT
        assert bag["frame_indices"] == list(range(FRAME_COUNT))
        assert len(bag["occlusion_flags"]) == FRAME_COUNT
        assert bag["cavity_area_px"] > 0
        assert bag["visible_frame_count"] == FRAME_COUNT
        assert bag["occluded_frame_count"] == 0
        assert min(bag["area_ratio_series"]) > 0
    # 袋间不能串号：两条曲线必须不同
    assert body["bags"][0]["area_ratio_series"] != body["bags"][1]["area_ratio_series"]

    # 面积曲线要落库，能按 session_id 回读
    session_id = body["bags"][0]["session_id"]
    stored = client.get(f"/api/v1/infusion/sessions/{session_id}").json()
    assert stored["area_ratio_series"] == body["bags"][0]["area_ratio_series"]


def test_uncalibrated_upload_has_no_volume_estimate(client):
    """未标定袋型时不得出现毫升数——曲线有了也不能驱动换液提醒。"""
    payload = build_rack_zip(default_rack_scene(frame_count=FRAME_COUNT))
    response = client.post(
        "/api/v1/infusion/tracks",
        data={
            "rack_id": "R-EXAMPLE-07",
            "infusion_order_no": "EXAMPLE-ORDER-0001",
            "bed_no": "EXAMPLE-BED-07",
            "captured_at": "2026-10-22T09:41:00+08:00",
            "slots": '[{"bag_seq": "B01", "slot_index": 0}]',
        },
        files={"frames": ("frames.zip", payload, "application/zip")},
    )
    bag = response.json()["bags"][0]
    assert bag["calibration_state"] == "uncalibrated"
    assert "estimated_volume_ml" not in bag
    assert "eta_minutes" not in bag


def test_mask_prediction_score_is_shape_only():
    prediction = MaskPrediction(mask=np.ones((10, 10), dtype=bool), score=0.7854)
    assert prediction.area == 100
    assert prediction.bbox == (0, 0, 10, 10)
    assert not prediction.is_empty
    assert MaskPrediction(mask=np.zeros((4, 4), dtype=bool)).bbox is None


def test_prompt_box_clipping():
    prompt = Prompt.from_box(-10.0, 5.0, 999.0, 12.0)
    assert prompt.clipped_box((20, 30)) == (0, 5, 30, 12)
    degenerate = Prompt(points=(Point(1.0, 1.0, 1),))
    assert degenerate.clipped_box((20, 30)) is None