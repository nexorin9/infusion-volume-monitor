"""M2 帧间关联与轨迹 ID 管理测试。

六组断言：

1. **ID 稳定**：连续 30 帧两袋，轨迹 ID 恒定、逐帧命中、序列长度恒等于帧数；
2. **粘连分离不串号**：两袋贴住粘成一块后重新分开，两条轨迹各自回到自己那一袋，
   既不合并成一条、也不左右互换；
3. **落空帧不丢**：短暂丢失后重入，``track_id`` 不变、落空帧留 ``None`` 占位、
   ``last_seen_frame`` 停在最后一次真看见的那帧而 ``track_age`` 继续走；
4. **终止可测**：连续未命中超过窗口即终止并给出原因，终止后不复活；
5. **判据取舍**：同一帧两袋 IoU 交叉时，几何判据一票否决 IoU（左右两个方向各测一次，
   证明不是「一律偏左」）；抖动超阈值降置信并打 ``inconsistent``，但**原值保留、不丢帧**；
6. **不另起推理路径**：本模块源码里没有任何预测器入口。

画面是**合成渲染帧**（``tests/support/synth.py``）与程序构造的掩膜，不是真实院内短片——
真实短片需医院授权采集，本轮以合成 + 公开 VOS 子集替代。
"""

from __future__ import annotations

import ast

from pathlib import Path

import cv2
import numpy as np
import pytest

from infusion_monitor.core.geom_kernel import GeomKernelParams, GeometricMaskPredictor, liquid_candidates
from infusion_monitor.core.postprocess import postprocess_mask
from infusion_monitor.core.predictor import Point, Prompt
from infusion_monitor.core.propagate import add_new_points_or_box, init_state, propagate_in_video
from infusion_monitor.core.tracking import (
    AMBIGUOUS,
    GEOMETRY_VETO,
    MATCHED,
    TERMINATED_BY_WINDOW,
    Detection,
    FrameAssociation,
    Track,
    TrackManager,
    TrackingParams,
    association,
    detections_from_mask,
    track_masks_sequence,
)
from tests.support.synth import BagSpec, RackScene, render_rack_frame

FRAME_COUNT = 30
HEIGHT, WIDTH = 640, 480
LIQUID_A = (92, 148, 196)
LIQUID_B = (64, 128, 190)

TRACKING = Path(__file__).resolve().parents[1] / "src" / "infusion_monitor" / "core" / "tracking.py"


# ── 构造器 ──────────────────────────────────────────────────────────────


def _rect(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _det(
    det_id: str,
    box: tuple[int, int, int, int],
    *,
    ratio: float = 0.5,
    frame_index: int = 0,
    mask: np.ndarray | None = None,
) -> Detection:
    x0, y0, x1, y1 = box
    return Detection(
        detection_id=det_id,
        frame_index=frame_index,
        bbox=box,
        centroid=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        area_px=(x1 - x0) * (y1 - y0),
        ratio=ratio,
        mask=mask,
    )


def _scene(width_ratio: float, left: float, right: float) -> RackScene:
    """同一液位曲线、不同袋间距的场景——只有几何变了，填充趋势完全一致。"""
    return RackScene(
        bags=(
            BagSpec("B01", left, width_ratio, 0.62, 0.40, LIQUID_A),
            BagSpec("B02", right, width_ratio, 0.75, 0.55, LIQUID_B),
        ),
        frame_count=FRAME_COUNT,
        width=WIDTH,
        height=HEIGHT,
    )


SEPARATE = _scene(0.26, 0.28, 0.72)
ADHERED = _scene(0.38, 0.33, 0.67)


def _liquid_mask(frame: np.ndarray) -> np.ndarray:
    """一帧画面 → 液体掩膜（走产品主路径的同一套几何核与形态学后处理）。"""
    params = GeomKernelParams()
    return postprocess_mask(
        liquid_candidates(frame, params),
        max_hole_area=params.max_hole_area,
        max_sprinkle_area=params.max_sprinkle_area,
    ).mask


def _detections_for(frame: np.ndarray, frame_index: int, cavity: int) -> list[Detection]:
    return detections_from_mask(_liquid_mask(frame), frame_index, cavity_area_px=cavity)


def _run(
    scene: RackScene,
    frames: dict[int, RackScene],
    *,
    merged: tuple[int, int] | None = None,
    **param_overrides,
) -> TrackManager:
    """按帧号取场景逐帧推进。

    ``frames`` 给「哪些帧用哪个场景」；``merged=(起, 止)`` 把该段的两个连通域并成**一块**
    ——模拟粘连检测没切开、主路径交到关联层的就是一整块的情形。
    """
    defaults = {"centroid_tolerance_px": 70.0, "bottom_tolerance_px": 40.0}
    manager = TrackManager(params=TrackingParams(**{**defaults, **param_overrides}))
    for index in range(scene.frame_count):
        source = frames.get(index, scene)
        detections = _detections_for(render_rack_frame(source, index), index, cavity=74_000)
        if merged and merged[0] <= index < merged[1] and len(detections) > 1:
            detections = [_merged_blob(detections, index)]
        if index == 0:
            manager.seed(0, detections)
        else:
            manager.step(index, detections)
    manager.finalize(scene.frame_count - 1)
    return manager


def _merged_blob(detections: list[Detection], frame_index: int) -> Detection:
    """把同帧的若干连通域并成一块：掩膜取并集、面积取和、质心按面积加权。

    取的是**掩膜并集**而不是外接矩形——两袋之间的背景不属于任何一块，填进去会让
    这一块凭空多出几万个像素，质心也跟着偏。并集再过一次小尺度闭运算把袋间那几像素
    的接缝桥上：袋子贴住时像素上本就连着，留缝只是渲染分辨率的产物。
    """
    union = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for detection in detections:
        union |= detection.mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    union = cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    area = sum(d.area_px for d in detections)
    cx = sum(d.centroid[0] * d.area_px for d in detections) / area
    cy = sum(d.centroid[1] * d.area_px for d in detections) / area
    rows = np.flatnonzero(union.any(axis=1))
    cols = np.flatnonzero(union.any(axis=0))
    return Detection(
        detection_id="D-blob",
        frame_index=frame_index,
        bbox=(int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1),
        centroid=(cx, cy),
        area_px=int(np.count_nonzero(union)),
        ratio=min(1.0, area / 74_000),
        mask=union,
    )


# ── 1. ID 稳定 ──────────────────────────────────────────────────────────


def test_thirty_frames_keep_one_id_per_bag():
    manager = _run(SEPARATE, {})
    assert sorted(manager.tracks) == ["T01", "T02"]
    for track in manager.tracks.values():
        assert track.hits == FRAME_COUNT, f"{track.track_id} 未逐帧命中"
        assert track.misses == 0
        assert track.last_seen_frame == FRAME_COUNT - 1
        assert track.track_age == FRAME_COUNT
        assert len(track.ratios) == FRAME_COUNT
        assert all(ratio is not None for ratio in track.ratios), "不该有落空帧"
        assert track.terminated is False
        assert track.confidence == pytest.approx(1.0, abs=1e-6)


def test_ids_follow_the_bag_not_the_label_or_the_slot():
    """ID 按质心左右排序分配，与连通域编号顺序无关：同一场景重跑必得同一套 ID。"""
    first = _run(SEPARATE, {})
    second = _run(SEPARATE, {})
    assert [t.track_id for t in first.tracks.values()] == [t.track_id for t in second.tracks.values()]
    left = min(first.tracks.values(), key=lambda t: t.centroid[0])
    right = max(first.tracks.values(), key=lambda t: t.centroid[0])
    assert (left.track_id, right.track_id) == ("T01", "T02")
    # 液位下降越多的袋子末帧面积比越低——ID 没跟着读数高低串位
    assert left.ratios[-1] < right.ratios[-1]


def test_module_opens_no_inference_path():
    """本模块不做分割：AST 层面不得 import 任何预测器，也不得调用推理入口。

    只看**代码**不看 docstring——模块头那张源码证据表会提到源仓库的调用名，
    但那是文档，不是本产品的推理路径。
    """
    tree = ast.parse(TRACKING.read_text(encoding="utf-8"))
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
        elif isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            called.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))

    assert "GeometricMaskPredictor" not in imported
    assert not ({"predict", "predict_batch", "propagate_in_video", "analyse_frames"} & called)
    # 本模块只允许从同层的 postprocess（连通域特征）与 predictor（纯几何工具）取用；
    # 一旦改成从 propagate / geom_kernel 借道，就是另起了一条分割推理路径
    assert imported & {"propagate", "geom_kernel", "enumerate"} == set()


# ── 2. 粘连分离不串号 ───────────────────────────────────────────────────


def test_adhesion_split_does_not_cross_ids():
    """第 10–14 帧两袋贴住、粘成一块；分开之后两条轨迹各回各袋，ID 不互换。"""
    adhered_frames = {index: ADHERED for index in range(10, 15)}
    manager = _run(SEPARATE, adhered_frames, merged=(10, 15))

    left = min(manager.tracks.values(), key=lambda t: t.centroid[0])
    right = max(manager.tracks.values(), key=lambda t: t.centroid[0])
    assert (left.track_id, right.track_id) == ("T01", "T02"), "ID 不得因粘连而互换"

    blob = _merged_blob(
        _detections_for(render_rack_frame(ADHERED, 12), 12, cavity=74_000), 12
    )
    assert len([d for d in _detections_for(render_rack_frame(ADHERED, 12), 12, 74_000)]) == 2
    assert blob.area_px > 0 and len(detections_from_mask(blob.mask, 12)) == 1, "粘连段确实只剩一块"

    # 粘住的那 5 帧：合并块正好落在两条轨迹中间，默认容差下几何对谁都否决——
    # 两条都落空（**不猜**），曲线留 None 占位，既不硬接一半也不把新块起成第三条轨迹
    assert left.ratios[10:15] == [None] * 5 and right.ratios[10:15] == [None] * 5
    assert len(manager.tracks) == 2

    # 同一段几何、容差放宽到 200 px：一块只归**一条**轨迹，另一条仍落空
    generous = _run(
        SEPARATE, adhered_frames, merged=(10, 15), centroid_tolerance_px=200.0
    )
    slots = [t.ratios[10:15] for t in generous.tracks.values()]
    assert sum(1 for slot in slots if all(v is not None for v in slot)) == 1, "一块不能被两条轨迹同时认领"
    assert sum(1 for slot in slots if all(v is None for v in slot)) == 1, "另一条必须如实落空"

    # 「不串号」的硬证据：与**没粘连**的干净跑法逐点对照——两条轨迹在粘连段之外的读数
    # 必须逐点一致。若 ID 在粘连处左右互换，这里必然对不上。
    clean = _run(SEPARATE, {})
    outside = [index for index in range(FRAME_COUNT) if not 10 <= index < 15]
    for track_id in ("T01", "T02"):
        expected = [clean.tracks[track_id].ratios[i] for i in outside]
        actual = [manager.tracks[track_id].ratios[i] for i in outside]
        assert actual == pytest.approx(expected), f"{track_id} 在粘连段之外的读数被换到别的袋上"

    left_ratio, right_ratio = left.ratios, right.ratios
    assert left_ratio[0] < right_ratio[0], "左袋起始液位更低（0.62 vs 0.75）"
    assert left_ratio[-1] < right_ratio[-1], "左袋也降得更低（0.40 vs 0.55）"
    for index in range(15, FRAME_COUNT):
        assert left.ratios[index] is not None and right.ratios[index] is not None
    assert left.centroid[0] < right.centroid[0], "两条轨迹仍在各自那一侧"


def test_adhesion_blob_is_claimed_by_at_most_one_track():
    """粘成一块时同一帧不得有两条轨迹抢同一块——一块只归一条轨迹。"""
    manager = TrackManager(params=TrackingParams(centroid_tolerance_px=200.0))
    manager.seed(
        0,
        [
            _det("D-left", (60, 70, 170, 560), ratio=0.62, mask=_rect(60, 70, 170, 560)),
            _det("D-right", (310, 70, 420, 560), ratio=0.75, mask=_rect(310, 70, 420, 560)),
        ],
    )
    blob = _det("D-blob", (60, 70, 420, 560), ratio=0.68, frame_index=1, mask=_rect(60, 70, 420, 560))
    result = manager.step(1, [blob])

    claimed = [m.track_id for m in result.matches if m.detection_id == "D-blob"]
    assert len(claimed) == 1, "一块掩膜只能归一条轨迹"
    assert claimed[0] == "T01", "质心更近的左袋应拿到这一块"
    assert result.missed_track_ids == ("T02",)
    assert manager.tracks["T02"].misses == 1
    assert manager.tracks["T02"].ratios[1] is None, "落空帧留 None 占位，不插值"


# ── 3. 落空帧不丢、短丢重入 ─────────────────────────────────────────────


def test_missed_frame_keeps_slot_and_reenters_with_same_id():
    manager = TrackManager(params=TrackingParams(max_missed_frames=8))
    box = (60, 70, 170, 560)
    manager.seed(0, [_det("D0", box, ratio=0.60, mask=_rect(*box))])
    for index in (1, 2, 3):
        manager.step(index, [])
    assert manager.tracks["T01"].ratios[1:] == [None, None, None], "落空帧留 None，序列长度不变"
    assert manager.tracks["T01"].last_seen_frame == 0, "last_seen_frame 停在最后真看见的那帧"
    assert manager.tracks["T01"].track_age == 4, "track_age 继续走，落空的帧也算年龄"
    assert manager.tracks["T01"].misses == 3

    moved = (66, 74, 176, 562)
    manager.step(4, [_det("D4", moved, ratio=0.58, frame_index=4, mask=_rect(*moved))])
    track = manager.tracks["T01"]
    assert track.track_id == "T01", "短暂丢失后重入必须是同一条轨迹"
    assert track.last_seen_frame == 4 and track.misses == 0 and track.hits == 2
    assert track.track_age == 5
    assert track.ratios[4] == pytest.approx(0.58)


def test_new_bag_after_a_gap_is_not_glued_to_the_old_curve():
    """槽位上换了一袋新袋：它与旧轨迹远，判几何否决，不把新袋接到旧曲线上。"""
    manager = TrackManager(params=TrackingParams())
    old_box = (60, 70, 170, 560)
    manager.seed(0, [_det("D0", old_box, ratio=0.60, mask=_rect(*old_box))])
    new_box = (300, 70, 410, 560)
    result = manager.step(1, [_det("D1", new_box, ratio=0.05, frame_index=1, mask=_rect(*new_box))])
    assert result.match_count == 0
    assert result.missed_track_ids == ("T01",)
    assert result.unmatched_detection_ids == ("D1",), "认不上的新袋如实报为未认领"
    assert manager.tracks["T01"].ratios[1] is None
    assert manager.tracks["T01"].misses == 1


# ── 4. 终止可测 ─────────────────────────────────────────────────────────


def test_track_is_terminated_after_the_miss_window_and_stays_dead():
    params = TrackingParams(max_missed_frames=3)
    manager = TrackManager(params=params)
    box = (60, 70, 170, 560)
    manager.seed(0, [_det("D0", box, ratio=0.60, mask=_rect(*box))])

    for index in range(1, params.max_missed_frames + 1):
        result = manager.step(index, [])
        assert manager.tracks["T01"].terminated is False, f"第 {index} 帧还不该终止（还能容忍）"
        assert result.missed_track_ids == ("T01",)

    result = manager.step(params.max_missed_frames + 1, [])
    assert manager.tracks["T01"].terminated is True
    assert manager.tracks["T01"].terminate_reason == TERMINATED_BY_WINDOW
    assert manager.tracks["T01"].misses == params.max_missed_frames + 1
    assert manager.terminated_tracks() and not manager.active_tracks()

    # 终止后即使袋子回到原位也不复活——不静默新建、也不复活旧轨迹
    manager.step(
        params.max_missed_frames + 2,
        [_det("D6", box, ratio=0.30, frame_index=params.max_missed_frames + 2, mask=_rect(*box))],
    )
    assert manager.tracks["T01"].terminated is True
    assert manager.tracks["T01"].ratios[-1] is None
    assert len(manager.tracks) == 1, "不得凭空多出一条轨迹"


def test_retire_by_caller_is_recorded_with_its_own_reason():
    manager = TrackManager()
    box = (60, 70, 170, 560)
    manager.seed(0, [_det("D0", box, ratio=0.60, mask=_rect(*box))])
    assert manager.retire("T01") is True
    assert manager.retire("T01") is False, "重复退役不改变状态"
    assert manager.tracks["T01"].terminate_reason == "retired_by_caller"
    assert manager.step(1, [_det("D1", box, ratio=0.5, frame_index=1, mask=_rect(*box))]).match_count == 0


# ── 5. 判据取舍：IoU 交叉时几何说了算 ───────────────────────────────────


@pytest.mark.parametrize(
    "track_box, memory_box, left_box, right_box, expected",
    [
        # 轨迹是**左**袋：上一帧记忆是两袋粘成的一块，这一帧重新分开。
        # 右边那块与记忆的 IoU 明显更高，但几何上这袋靠左 → 接左边。
        (
            (20, 40, 80, 200),
            (20, 40, 200, 200),
            (20, 40, 80, 200),
            (60, 40, 200, 200),
            "D-left",
        ),
        # 轨迹是**右**袋：对称情形，右边胜出，证明不是「一律偏左」。
        (
            (140, 40, 200, 200),
            (20, 40, 200, 200),
            (20, 40, 140, 200),
            (140, 40, 200, 200),
            "D-right",
        ),
    ],
)
def test_iou_crossing_defers_to_geometry(track_box, memory_box, left_box, right_box, expected):
    params = TrackingParams()
    memory = _rect(*memory_box)
    track = Track(
        track_id="T01",
        first_frame=0,
        last_seen_frame=0,
        last_updated_frame=0,
        bbox=track_box,
        centroid=((track_box[0] + track_box[2]) / 2.0, (track_box[1] + track_box[3]) / 2.0),
        ratio=0.5,
        last_mask=memory,
    )
    left = _det("D-left", left_box, mask=_rect(*left_box))
    right = _det("D-right", right_box, mask=_rect(*right_box))

    left_verdict = association(track, left, params)
    right_verdict = association(track, right, params)

    # 前提：IoU 确实交叉——「错的」那块与记忆的重叠比「对的」更多
    wrong = right_verdict if expected == "D-left" else left_verdict
    right_one = left_verdict if expected == "D-left" else right_verdict
    assert wrong.iou > right_one.iou, "本用例的前提就是 IoU 交叉"

    # 判据取舍：几何一票否决 IoU
    assert wrong.matched is False and wrong.reason == GEOMETRY_VETO
    assert right_one.matched is True and right_one.reason == MATCHED

    manager = TrackManager(params=params)
    manager.tracks["T01"] = track
    result = manager.step(1, [left, right])
    assert result.detection_for("T01") == expected
    assert len(result.unmatched_detection_ids) == 1


def test_ambiguous_pair_is_not_guessed():
    """一袋被管路阴影切成两块，两块都与锚点等距：不按「取最高分」硬接，本帧落空。"""
    manager = TrackManager(params=TrackingParams())
    whole = _det("D-whole", (20, 40, 120, 200), ratio=0.5, mask=_rect(20, 40, 120, 200))
    manager.seed(0, [whole])

    # 液体被管路/标签阴影切成两块，两块质心到锚点都是 20 px——打分成完全一样
    left = _det("D-part-a", (20, 40, 80, 200), ratio=0.3, frame_index=1)
    right = _det("D-part-b", (60, 40, 120, 200), ratio=0.2, frame_index=1)
    result = manager.step(1, [left, right])

    assert result.match_count == 0
    assert result.ambiguous_track_ids == ("T01",)
    assert result.is_ambiguous is True
    assert result.missed_track_ids == ("T01",)
    assert set(result.unmatched_detection_ids) == {"D-part-a", "D-part-b"}, "认领不了就都不认领"
    track = manager.tracks["T01"]
    assert track.ratios[1] is None, "歧义帧留 None，不取分数高的一半硬接"
    assert track.last_seen_frame == 0

    # 下一帧阴影消失、恢复成完整一块：同一条轨迹原样接上，不用重新起 ID
    manager.step(2, [_det("D-back", (20, 40, 120, 200), ratio=0.48, frame_index=2, mask=_rect(20, 40, 120, 200))])
    assert manager.tracks["T01"].track_id == "T01"
    assert manager.tracks["T01"].ratios[1] is None and manager.tracks["T01"].ratios[2] == pytest.approx(0.48)


def test_jitter_marks_inconsistent_without_dropping_frame_or_value():
    manager = TrackManager(params=TrackingParams(jitter_ratio=0.05))
    box = (60, 70, 170, 560)
    manager.seed(0, [_det("D0", box, ratio=0.60, mask=_rect(*box))])
    spike_box = (60, 120, 170, 560)  # 袋没动，只是液面突然"跳"了一截
    result = manager.step(
        1, [_det("D1", spike_box, ratio=0.12, frame_index=1, mask=_rect(*spike_box))]
    )

    track = manager.tracks["T01"]
    assert result.detection_for("T01") == "D1", "抖动帧仍被关联，不静默丢帧"
    assert track.ratios[1] == pytest.approx(0.12), "保留原值，不平滑、不插值"
    assert len(track.ratios) == 2
    assert track.inconsistent_frames == [1] and track.is_inconsistent is True
    assert track.consistency < 1.0, "抖动应降一致性评分"
    assert track.confidence <= 0.60, "出过一次抖动的轨迹不足以驱动换液点名"
    assert track.last_seen_frame == 1


def test_consistency_recovers_when_frames_are_stable_again():
    """抖过一次会降分；读数重新稳下来后评分能爬回去，但不一致帧号仍留档不抹掉。"""
    manager = TrackManager(params=TrackingParams(jitter_ratio=0.05))
    box = (60, 70, 170, 560)
    spike_box = (60, 120, 170, 560)
    manager.seed(0, [_det("D0", box, ratio=0.60, mask=_rect(*box))])
    manager.step(1, [_det("D1", spike_box, ratio=0.12, frame_index=1, mask=_rect(*spike_box))])
    spiked = manager.tracks["T01"]
    assert spiked.consistency < 1.0 and spiked.inconsistent_frames == [1]

    # 回落那一帧相对上一帧同样是跳变，照实记——一帧尖峰会把前后两帧都标出来
    manager.step(2, [_det("D2", box, ratio=0.58, frame_index=2, mask=_rect(*box))])
    settled = manager.tracks["T01"].consistency
    for index, ratio in enumerate([0.57, 0.56, 0.55], start=3):
        manager.step(index, [_det(f"D{index}", box, ratio=ratio, frame_index=index, mask=_rect(*box))])

    recovered = manager.tracks["T01"]
    assert recovered.consistency > settled, "读数稳下来后一致性应回升"
    assert recovered.inconsistent_frames == [1, 2], "不一致帧号是留档，不因为后来稳了就抹掉"
    assert recovered.confidence <= 0.60, "留过不一致的轨迹置信仍封顶，要人工看一眼"
    assert recovered.misses == 0 and len(recovered.ratios) == 6


def test_association_is_reproducible_regardless_of_input_order():
    """同一输入无论检测给什么顺序，贪心结果都一致（排序键显式，不靠字典序）。"""
    box_a, box_b = (60, 70, 170, 560), (300, 70, 410, 560)

    def run(order):
        manager = TrackManager()
        manager.seed(0, [_det("D-left", box_a, mask=_rect(*box_a)), _det("D-right", box_b, mask=_rect(*box_b))])
        detections = [_det("E-b", box_b, frame_index=1, mask=_rect(*box_b)), _det("E-a", box_a, frame_index=1, mask=_rect(*box_a))]
        return {m.track_id: m.detection_id for m in manager.step(1, order).matches}

    assert run([_det("E-a", box_a, frame_index=1, mask=_rect(*box_a)), _det("E-b", box_b, frame_index=1, mask=_rect(*box_b))]) == run(
        [_det("E-b", box_b, frame_index=1, mask=_rect(*box_b)), _det("E-a", box_a, frame_index=1, mask=_rect(*box_a))]
    )


# ── 6. 与真实主路径对接：只做关联，不重新分割 ───────────────────────────


def test_tracks_real_main_path_output_without_resegmenting():
    """直接吃 :mod:`.propagate` 逐帧产出的掩膜，接成两条 ID 稳定的轨迹。"""
    predictor = GeometricMaskPredictor()
    frames = [render_rack_frame(SEPARATE, index) for index in range(FRAME_COUNT)]
    names = [f"f{index:03d}.png" for index in range(FRAME_COUNT)]
    state = init_state(names, frames[0].shape)
    for slot, bag in enumerate(SEPARATE.bags):
        add_new_points_or_box(
            state,
            frame_index=0,
            obj_id=bag.bag_seq,
            prompt=Prompt(points=(Point(WIDTH * bag.x_center, HEIGHT * 0.7, 1),)),
            cavity_area_px=74_000,
        )

    propagated = [
        {obj_id: result.masks[obj_id] for obj_id in sorted(result.masks)}
        for result in propagate_in_video(state, predictor, lambda name: frames[names.index(name)])
    ]
    manager = track_masks_sequence(propagated, cavity_area_px=74_000)
    assert sorted(manager.tracks) == ["T01", "T02"]
    for track in manager.tracks.values():
        assert len(track.ratios) == FRAME_COUNT
        assert track.misses == 0 and track.terminated is False
        assert all(ratio is not None for ratio in track.ratios)
        assert track.ratios[0] > track.ratios[-1], "面积比应随液面下降而走低"
    left = min(manager.tracks.values(), key=lambda t: t.centroid[0])
    right = max(manager.tracks.values(), key=lambda t: t.centroid[0])
    assert left.ratios[0] < right.ratios[0], "右袋起始液位更高（0.75 vs 0.62）"


def test_frame_association_helpers_report_both_sides():
    result = FrameAssociation(frame_index=3)
    assert result.match_count == 0 and result.is_ambiguous is False
    assert result.detection_for("T01") is None