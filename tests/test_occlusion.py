"""M2 遮挡标记、恢复续接与长序列恒定算力测试。

四组断言：

1. **遮挡标记**：完全看不见的帧显式记 ``occluded``，面积比留 ``None``，序列长度不变，
   **不插值、不沿用上一帧数值**；
2. **部分可见**：被挡了一块的帧记 ``partial``（还能看见，不是完全看不见），且**连续多帧
   都不会自我掩盖**——被压低的读数不进趋势基线；
3. **恢复续接**：遮挡后按上一帧记忆比对接回**同一袋**（``bag_seq`` 不变、中间仍留空、
   恢复点与干净拍摄同值）；对不上就**不猜**，超过丢失窗口则 ``lost`` 并交出轨迹；
4. **恒定算力**：600 帧与 1200 帧序列的峰值内存与单帧耗时同量级，不随序列长度线性膨胀。

画面是**合成渲染帧**（``tests/support/synth.py``），不是真实院内短片——真实短片需医院
授权采集，本轮以合成 + 公开 VOS 子集替代。
"""

from __future__ import annotations

import resource
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
from infusion_monitor.core.occlusion import (
    HANDOFF_REASON,
    LOST,
    OCCLUDED,
    PARTIAL,
    REAPPEARED,
    VISIBLE,
    OcclusionTracker,
    TrackMemory,
    VisibilityParams,
    handoff_to_dict,
    summarise_states,
)
from infusion_monitor.core.predictor import MaskPrediction
from infusion_monitor.core.propagate import analyse_frames, load_rgb, slot_seed_prompt
from tests.support.synth import (
    RackScene,
    build_rack_zip,
    default_rack_scene,
    write_rack_frames,
)

FRAME_COUNT = 18
HEIGHT, WIDTH = 240, 320
CAVITY = 10_000


# ── 构造器 ──────────────────────────────────────────────────────────────
def _pred(box: tuple[int, int, int, int], score: float = 0.72) -> MaskPrediction:
    """一个矩形掩膜预测；``box`` 为 ``(x0, y0, x1, y1)``。"""
    x0, y0, x1, y1 = box
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return MaskPrediction(mask=mask, score=score)


def _tracker(frame_count: int = 12, **params) -> OcclusionTracker:
    return OcclusionTracker(
        obj_id="B01", cavity_area_px=CAVITY, frame_count=frame_count, params=VisibilityParams(**params)
    )


def _box(area_px: int, *, y1: int = 200, x0: int = 100) -> tuple[int, int, int, int]:
    """由目标面积与固定的袋底/左边界反推出外接框（模拟液面下降：袋底不动、顶部上移）。"""
    width = 100
    height = max(1, area_px // width)
    return (x0, y1 - height, x0 + width, y1)


def _ratio(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


@pytest.fixture
def predictor() -> GeometricMaskPredictor:
    return GeometricMaskPredictor()


def _seeds(scene: RackScene, shape):
    return [
        (bag.bag_seq, slot_seed_prompt(shape, slot, len(scene.bags)))
        for slot, bag in enumerate(scene.bags)
    ]


def _run(tmp_path: Path, scene: RackScene, predictor, **variant):
    paths = write_rack_frames(tmp_path, scene, **variant)
    shape = load_rgb(paths[0]).shape
    return {c.obj_id: c for c in analyse_frames(paths, _seeds(scene, shape), predictor)}


# ── 1. 状态机：标记 ──────────────────────────────────────────────────────
def test_first_sighting_is_visible_without_memory():
    tracker = _tracker()
    verdict = tracker.observe(0, _pred(_box(8000)))
    assert verdict.state == VISIBLE
    assert not verdict.used_memory, "首帧没有记忆可依，不该声称用了记忆"
    assert tracker.memory is not None


def test_missing_frames_are_marked_and_left_empty():
    """完全看不见的帧：记 occluded、面积比留空，序列长度不变，不插值。"""
    tracker = _tracker()
    tracker.observe(0, _pred(_box(8000)))
    for index in (1, 2, 3):
        verdict = tracker.observe(index, None)
        assert verdict.state == OCCLUDED
        assert verdict.area_ratio is None

    assert tracker.ratios == [8000 / CAVITY, None, None, None, None, None, None, None, None, None, None, None]
    assert len(tracker.states) == tracker.frame_count
    assert [s for s in tracker.states if s == OCCLUDED] == [OCCLUDED] * 3
    assert tracker.missing_frame_count == 3
    # 关键：空档里不能出现任何数值——假曲线比没有曲线更危险
    assert tracker.ratios[1:4] == [None, None, None]


def test_empty_mask_counts_as_missing():
    tracker = _tracker()
    empty = MaskPrediction(mask=np.zeros((HEIGHT, WIDTH), dtype=bool))
    assert tracker.observe(0, empty).state == OCCLUDED
    assert tracker.ratios[0] is None


# ── 2. 状态机：部分可见 ──────────────────────────────────────────────────
def test_partial_when_ratio_falls_below_memory_trend():
    """面积比明显低于「记忆 + 滴速趋势」→ partial，且判定用到了记忆。"""
    tracker = _tracker()
    for index, area in enumerate((8000, 7800, 7600)):  # 建立下降趋势
        tracker.observe(index, _pred(_box(area)))
    assert tracker.memory is not None and tracker.memory.drain_per_frame > 0

    verdict = tracker.observe(3, _pred(_box(3000)))  # 骤降到 3000
    assert verdict.state == PARTIAL
    assert verdict.used_memory
    assert "补拍" in verdict.reason
    # 面积比照给（这一帧确实看得见），只是被判为读数偏低
    assert verdict.area_ratio == pytest.approx(3000 / CAVITY)


def test_partial_reading_does_not_overwrite_trusted_level():
    """被压低的读数不进基线——否则本次遮挡会把自己学成正常。"""
    tracker = _tracker()
    for index, area in enumerate((8000, 7800, 7600)):
        tracker.observe(index, _pred(_box(area)))
    trusted = tracker.memory.ratio  # type: ignore[union-attr]
    tracker.observe(3, _pred(_box(3000)))
    assert tracker.memory.ratio == pytest.approx(trusted)
    # 但几何照常更新，跟踪不丢
    assert tracker.memory.bbox == _box(3000)  # type: ignore[union-attr]


def test_sustained_partial_does_not_mask_itself():
    """连续 4 帧只看到部分袋体：每一帧都要继续报 partial，而不是「学成正常」。"""
    tracker = _tracker()
    for index, area in enumerate((8000, 7800, 7600)):
        tracker.observe(index, _pred(_box(area)))
    states = [tracker.observe(index, _pred(_box(3000))).state for index in range(3, 7)]
    assert states == [PARTIAL] * 4


def test_drain_trend_ignores_physically_impossible_drop():
    """输液泵不会在 0.5 秒内快 8 倍——超物理上限的降幅不进趋势，只算遮挡。"""
    tracker = _tracker(max_drain_per_frame=0.04)
    for index, area in enumerate((8000, 7800, 7600, 7400)):
        tracker.observe(index, _pred(_box(area)))
    learned = tracker.memory.drain_per_frame  # type: ignore[union-attr]
    assert learned < 0.04
    tracker.observe(4, _pred(_box(2000)))
    assert tracker.memory.drain_per_frame == pytest.approx(learned)


def test_normal_drain_is_not_flagged_partial():
    """按正常滴速下降不该被误报成「只看到部分袋体」。"""
    tracker = _tracker()
    states = []
    area = 8000
    for index in range(10):
        verdict = tracker.observe(index, _pred(_box(area)))
        states.append(verdict.state)
        area -= 200  # 每帧 -0.02，符合物理上限
    assert PARTIAL not in states
    assert states == [VISIBLE] * 10


# ── 3. 恢复续接 ─────────────────────────────────────────────────────────
def test_agreement_is_insensitive_to_draining():
    """液面下降会压扁外接框；若用 IoU，越接近换液越接不回来。袋底 + 宽度才是稳的锚。"""
    memory = TrackMemory(
        frame_index=4, bbox=_box(8000), centroid=(150.0, 150.0), area=8000, ratio=0.8
    )
    drained = _pred(_box(2000))  # 面积掉到 1/4，袋底仍固定、宽度不变
    assert memory.agreement(drained, VisibilityParams()) >= 0.5
    moved = _pred((20, 20, 120, 120))  # 整袋跑到别处
    assert memory.agreement(moved, VisibilityParams()) < 0.5


def test_reappearance_reattaches_same_object():
    """遮挡后接回同一袋：状态记 reappeared、中间仍留空、obj_id 不变。"""
    tracker = _tracker()
    tracker.observe(0, _pred(_box(8000)))
    tracker.observe(1, _pred(_box(7800)))
    for index in (2, 3, 4):
        tracker.observe(index, None)

    verdict = tracker.observe(5, _pred(_box(7400)))
    assert verdict.state == REAPPEARED
    assert verdict.used_memory and verdict.confidence >= 0.5
    assert tracker.ratios[2:5] == [None, None, None], "空档不得被回填"
    assert tracker.ratios[5] == pytest.approx(7400 / CAVITY)
    assert tracker.obj_id == "B01"


def test_reappearance_returns_to_visible_after_recovery():
    """reappeared 只标记恢复那一帧，之后回到正常判定。"""
    tracker = _tracker()
    tracker.observe(0, _pred(_box(8000)))
    tracker.observe(1, None)
    states = [tracker.observe(index, _pred(_box(7800 - 100 * index))).state for index in (2, 3, 4)]
    assert states == [REAPPEARED, VISIBLE, VISIBLE]


def test_mismatch_after_occlusion_is_not_guessed():
    """遮挡后在别处冒出来的东西不接回旧曲线——宁可继续记遮挡。"""
    tracker = _tracker()
    tracker.observe(0, _pred(_box(8000)))
    tracker.observe(1, None)
    verdict = tracker.observe(2, _pred((10, 10, 110, 110)))  # 完全不同的位置
    assert verdict.state == OCCLUDED
    assert verdict.area_ratio is None, "对不上就不能给面积比"
    assert "不猜" in verdict.reason


# ── 4. 丢失窗口与轨迹交出 ────────────────────────────────────────────────
def test_lost_window_emits_single_handoff():
    tracker = _tracker(frame_count=30, lost_window_frames=3)
    tracker.observe(0, _pred(_box(8000)))
    for index in range(1, 30):
        tracker.observe(index, None)

    assert tracker.states[1:4] == [OCCLUDED] * 3
    assert tracker.states[4] == LOST, "连续遮挡 4 帧 > 窗口 3 帧 → lost"
    assert tracker.handoff is not None
    handoff = tracker.handoff
    assert handoff.obj_id == "B01"
    assert handoff.lost_frame == 4
    assert handoff.last_seen_frame == 0
    assert handoff.missing_frames == 4
    assert handoff.reason == HANDOFF_REASON
    assert handoff.reference_area_ratio == pytest.approx(8000 / CAVITY)
    assert handoff.bbox is not None and len(handoff.bbox) == 4


def test_lost_track_is_frozen_not_silently_reborn():
    """交出之后即使袋子又出现在画面里，曲线也不复活——不静默新建轨迹接两袋。"""
    tracker = _tracker(frame_count=20, lost_window_frames=2)
    tracker.observe(0, _pred(_box(8000)))
    for index in (1, 2, 3, 4):
        tracker.observe(index, None)
    assert tracker.handoff is not None, "连续遮挡 4 帧 > 窗口 2 帧应交出"
    # 第 5 帧起袋子回到画面、位置也对得上——但没有认领就不接回
    for index in range(5, 20):
        tracker.observe(index, _pred(_box(7000)))

    assert tracker.states[-1] == LOST
    assert tracker.ratios[-1] is None, "lost 之后不得偷偷把曲线接回去"
    assert [s for s in tracker.states if s == VISIBLE] == [VISIBLE]
    assert tracker.handoff.lost_frame == 3


def test_reacquire_entry_reopens_frozen_track():
    """task 17 的再获取路径认领后，轨迹才能重新接上。"""
    tracker = _tracker(frame_count=20, lost_window_frames=2)
    tracker.observe(0, _pred(_box(8000)))
    for index in range(1, 6):
        tracker.observe(index, None)
    assert tracker.handoff is not None

    tracker.reacquire_from(6)
    verdict = tracker.observe(7, _pred(_box(6000)))
    assert verdict.state == VISIBLE
    assert tracker.ratios[7] == pytest.approx(6000 / CAVITY)


def test_handoff_to_dict_and_summary_helpers():
    tracker = _tracker(frame_count=12, lost_window_frames=1)
    tracker.observe(0, _pred(_box(8000)))
    tracker.observe(1, None)
    tracker.observe(2, None)
    payload = handoff_to_dict(tracker.handoff)
    assert payload is not None
    assert set(payload) == {
        "obj_id",
        "lost_frame",
        "last_seen_frame",
        "missing_frames",
        "bbox",
        "centroid",
        "cavity_area_px",
        "reference_area_ratio",
        "reason",
    }
    assert payload["bbox"] == list(_box(8000))
    assert handoff_to_dict(None) is None
    assert summarise_states([VISIBLE, VISIBLE, OCCLUDED]) == "visible×2 → occluded×1"
    assert tracker.counts_by_state()[LOST] == 1


def test_tracker_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="frame_count"):
        OcclusionTracker(obj_id="B01", cavity_area_px=CAVITY, frame_count=0)
    with pytest.raises(ValueError, match="袋腔面积"):
        OcclusionTracker(obj_id="B01", cavity_area_px=0, frame_count=4)
    tracker = _tracker(frame_count=4)
    with pytest.raises(ValueError, match="越界"):
        tracker.observe(9, _pred(_box(8000)))


# ── 5. 主路径：渲染场景 ──────────────────────────────────────────────────
def test_rendered_full_occlusion_marks_and_reattaches(tmp_path, predictor):
    """挡板盖住第 5–8 帧：标记而非丢弃；恢复后接回同一袋，末值与干净拍摄一致。"""
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    occluded = _run(tmp_path / "occ", scene, predictor, occluders=[(0, 5, 8)])
    clean = _run(tmp_path / "clean", scene, predictor)
    bag = occluded["B01"]

    assert len(bag.area_ratio_series) == scene.frame_count
    assert bag.visibility_states[5:8] == [OCCLUDED] * 3
    assert bag.area_ratio_series[5:8] == [None, None, None]
    assert bag.visibility_states[8] == REAPPEARED
    assert bag.occluded_frame_count == 3
    # 恢复点续上同一条曲线：末值与从未被遮挡时逐位相同（没接到别的东西上）
    assert bag.latest_ratio == pytest.approx(clean["B01"].latest_ratio, abs=1e-5)
    assert bag.obj_id == "B01"
    # 另一袋不受影响
    assert occluded["B02"].occluded_frame_count == 0
    assert PARTIAL not in bag.visibility_states


def test_rendered_partial_occlusion_is_partial_not_occluded(tmp_path, predictor):
    """袋下段被压住：看得见 → partial（不是 occluded），且连续多帧都如实标记。"""
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    curves = _run(tmp_path / "part", scene, predictor, partial_occluders=[(0, 5, 8, 0.40)])
    bag = curves["B01"]

    assert bag.visibility_states[5] == PARTIAL
    assert bag.visibility_states[6] == PARTIAL, "遮挡持续时不能自我掩盖成 visible"
    assert bag.occluded_frame_count == 0, "还看得见就不该记成完全看不见"
    assert all(ratio is not None for ratio in bag.area_ratio_series)
    # 恢复后回到正常判定，且末值与干净拍摄一致
    assert bag.visibility_states[-1] == VISIBLE
    clean = _run(tmp_path / "clean", scene, predictor)
    assert bag.latest_ratio == pytest.approx(clean["B01"].latest_ratio, abs=1e-5)


def test_rendered_long_occlusion_hands_off_track(tmp_path, predictor):
    """交出后**没有认领就不接回**——本用例把再获取路径显式关掉，单独盯可见度这一层。

    关掉再获取（task 17 的 :mod:`.reacquire`）之后，遮挡层就是唯一裁判：袋子第 16 帧
    重新入画也**不得**把曲线接回去。这条契约与 task 17 的再获取用例是一对——那边开着
    再获取路径，证明同一段短片能被有据地接回同一条轨迹。
    """
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    paths = write_rack_frames(tmp_path / "lost", scene, occluders=[(0, 3, 16)])
    shape = load_rgb(paths[0]).shape
    curves = {
        c.obj_id: c
        for c in analyse_frames(paths, _seeds(scene, shape), predictor, reacquire=None)
    }
    bag = curves["B01"]

    assert LOST in bag.visibility_states
    assert bag.is_lost and bag.handoff is not None
    assert bag.handoff.obj_id == "B01"
    assert bag.handoff.lost_frame == 11  # 连续遮挡 9 帧 > 窗口 8 帧
    assert bag.handoff.last_seen_frame == 2
    # 袋子在 16 帧以后又露出来了，但没有认领就不接回
    assert bag.area_ratio_series[16] is None
    assert bag.visibility_states[16] == LOST
    assert bag.reacquires == [], "再获取关着就不该有任何再获取判定"
    # 另一袋完全不受牵连
    assert curves["B02"].occluded_frame_count == 0


def test_api_response_carries_visibility_states(client):
    """接口层把逐帧可见度与轨迹交出单带给护士端。"""
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    payload = build_rack_zip(scene, occluders=[(0, 3, 16)])
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
    bags = {bag["bag_seq"]: bag for bag in response.json()["bags"]}

    handed_off = bags["B01"]
    assert LOST in handed_off["visibility_states"]
    assert handed_off["track_handoff"]["obj_id"] == "B01"
    assert handed_off["track_handoff"]["reason"] == HANDOFF_REASON
    assert handed_off["track_handoff"]["lost_frame"] > handed_off["track_handoff"]["last_seen_frame"]
    assert len(handed_off["visibility_states"]) == scene.frame_count
    assert len(handed_off["occlusion_flags"]) == scene.frame_count

    healthy = bags["B02"]
    # 未交出时字段**整体不出现**（与 estimated_volume_ml 同一套门禁语义），不是 null 占位
    assert "track_handoff" not in healthy
    assert healthy["occluded_frame_count"] == 0
    assert LOST not in healthy["visibility_states"]


# ── 6. 恒定算力：600 帧 vs 1200 帧 ──────────────────────────────────────
def _peak_rss_bytes() -> int:
    """进程 RSS 高水位（macOS 返回字节，Linux 返回 KB）。"""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _measure(paths, seeds, predictor) -> dict:
    rss_before = _peak_rss_bytes()
    tracemalloc.start()
    started = time.perf_counter()
    curves = analyse_frames(paths, seeds, predictor)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "elapsed": elapsed,
        "ms_per_frame": elapsed / len(paths) * 1000.0,
        "peak_bytes": peak,
        "rss_growth": _peak_rss_bytes() - rss_before,
        "curve": curves[0],
    }


@pytest.fixture(scope="module")
def long_scene(tmp_path_factory) -> tuple[RackScene, list[Path]]:
    """一段 1200 帧的合成输液架短片（600 帧那一跑复用它的前 600 帧）。"""
    base = default_rack_scene(frame_count=1200)
    scene = RackScene(bags=base.bags, frame_count=1200, width=320, height=240)
    directory = tmp_path_factory.mktemp("long-frames")
    return scene, write_rack_frames(directory, scene)


def test_long_sequence_has_constant_compute(long_scene, predictor):
    """恒定算力：帧数翻倍，峰值内存与单帧耗时都不同量级地涨。"""
    scene, paths = long_scene
    shape = load_rgb(paths[0]).shape
    seeds = _seeds(scene, shape)

    short = _measure(paths[:600], seeds, predictor)
    long = _measure(paths, seeds, predictor)

    assert short["curve"].obj_id == long["curve"].obj_id == "B01"
    assert len(short["curve"].visibility_states) == 600
    assert len(long["curve"].visibility_states) == 1200
    # 千帧长片里不该冒出遮挡/丢失（没有遮挡就别瞎标）
    assert set(long["curve"].visibility_states) <= {VISIBLE, PARTIAL}

    # 单帧耗时恒定：实测 600 帧 ≈1.9 ms/帧、1200 帧 ≈1.85 ms/帧
    assert short["ms_per_frame"] < 40.0, f"单帧耗时 {short['ms_per_frame']:.2f} ms 超阈值"
    assert long["ms_per_frame"] < 40.0, f"单帧耗时 {long['ms_per_frame']:.2f} ms 超阈值"
    assert (
        long["ms_per_frame"] <= short["ms_per_frame"] * 1.5
    ), f"单帧耗时随帧数变慢：{short['ms_per_frame']:.2f} → {long['ms_per_frame']:.2f} ms/帧"

    # 峰值内存恒定：实测两者都是 ≈1.9 MB（线性实现会翻倍到 ≈3.9 MB）
    assert long["peak_bytes"] <= short["peak_bytes"] * 1.5 + 2_000_000
    # 进程侧同样不膨胀：缓存每帧掩膜的实现会在多出的 600 帧上多占上百 MB
    assert long["rss_growth"] <= 64_000_000, f"RSS 增长 {long['rss_growth']} 疑似随帧数膨胀"