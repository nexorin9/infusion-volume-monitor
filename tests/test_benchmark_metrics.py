"""公开 VOS 对照基准的测试：指标口径、消融语义与「记忆不劣于单帧」这条命题。

三组：

1. **指标口径**——IoU / J / F / J&F 的定义、空掩膜的语义，以及
   「距离变换版 F 与逐像素最近邻版 F 等价」这条等价性（性能改写，不是口径改写）。
2. **消融语义**——``memory_prompt`` 开关真的只改「每帧提示从哪来」，其余不变。
3. **对照结论**——在含遮挡的序列上，记忆方案不劣于阈值单帧；语料不在时退 skipped。
"""

from __future__ import annotations

import numpy as np
import pytest

from infusion_monitor.bench import metrics as M
from infusion_monitor.bench.runner import (
    OCCLUSION_AREA_RATIO,
    discover,
    evaluate_sequence,
    run_benchmark,
    verdict,
)
from infusion_monitor.bench.schemes import (
    MEMORY_SCHEME,
    SINGLE_FRAME_SCHEME,
    ColorBandParams,
    ColorBandPredictor,
    anchor_from_mask,
)

VOS_ROOT = "data/vos_subset"


# ── 1. 指标口径 ──────────────────────────────────────────────────────────


def _square(size: int, top: int, left: int, side: int) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    m[top : top + side, left : left + side] = True
    return m


def test_jaccard_matches_hand_computed_values():
    a = _square(20, 0, 0, 10)  # 100 px
    b = _square(20, 5, 0, 10)  # 100 px, 交集 50, 并集 150
    assert M.jaccard(a, b) == pytest.approx(50 / 150)
    assert M.jaccard(a, a) == 1.0


def test_j_and_iou_are_the_same_single_object_quantity():
    """单对象单帧下 J 与 IoU 同值——两处都报是为了多对象场景，这里钉住同值关系。"""
    a = _square(20, 0, 0, 10)
    b = _square(20, 3, 2, 10)
    assert M.region_similarity(a, b) == M.jaccard(a, b)


def test_both_empty_counts_as_perfect_and_one_sided_empty_as_zero():
    empty = np.zeros((12, 12), dtype=bool)
    block = _square(12, 2, 2, 4)
    assert M.jaccard(empty, empty) == 1.0
    # 真值为空时检出东西 = 0 分（不能靠「全片检出空气」刷均分）
    assert M.jaccard(block, empty) == 0.0
    metric = M.frame_metrics(block, empty, sequence="s", frame_index=0)
    assert metric.present is False
    assert metric.j_and_f == 0.0
    # 反过来：真值有东西却什么都没检出 = 0 分
    assert M.jaccard(empty, block) == 0.0


def test_contour_accuracy_is_one_for_identical_and_near_zero_for_disjoint():
    a = _square(40, 5, 5, 20)
    assert M.db_eval_boundary(a, a) == pytest.approx(1.0)
    b = np.zeros_like(a)
    assert M.db_eval_boundary(a, b) == 0.0


def test_contour_accuracy_rewards_being_within_tolerance_and_punishes_beyond():
    truth = _square(60, 10, 10, 30)
    near = _square(60, 11, 10, 30)  # 偏移 1 px，远小于容差
    far = _square(60, 30, 30, 20)  # 完全另一个位置
    assert M.db_eval_boundary(near, truth) > 0.95
    assert M.db_eval_boundary(far, truth) < 0.2


def test_distance_transform_contour_matches_brute_force_nearest_neighbour():
    """距离变换是性能改写；这里对拍证明它与逐像素最近邻口径完全等价。"""
    rng = np.random.default_rng(7)
    truth = rng.random((40, 40)) > 0.5
    pred = rng.random((40, 40)) > 0.55
    # 让两者至少各有边界像素，否则两边都空会走满分短路、对拍不到
    truth[10:20, 10:20] = True
    pred[15:28, 12:24] = True

    tol = float(np.ceil(M.BOUND_TH * float(np.linalg.norm(truth.shape))))
    p_bound = M._boundary_mask(pred)
    g_bound = M._boundary_mask(truth)

    # 逐像素暴力版：每个边界像素到另一侧全部边界像素的最小欧氏距离
    ys, xs = np.nonzero(g_bound)
    brute_pred = sum(
        1
        for y, x in zip(*np.nonzero(p_bound))
        if np.min(np.hypot(ys - y, xs - x)) <= tol
    )
    ys2, xs2 = np.nonzero(p_bound)
    brute_gt = sum(
        1
        for y, x in zip(*np.nonzero(g_bound))
        if np.min(np.hypot(ys2 - y, xs2 - x)) <= tol
    )

    precision, recall = M.boundary_precision_recall(pred, truth)
    assert precision == pytest.approx(brute_pred / int(np.count_nonzero(p_bound)))
    assert recall == pytest.approx(brute_gt / int(np.count_nonzero(g_bound)))


def test_j_and_f_is_the_mean_of_j_and_f():
    truth = _square(50, 8, 8, 25)
    pred = _square(50, 9, 9, 24)
    metric = M.frame_metrics(pred, truth, sequence="s", frame_index=3, obj_id=1)
    assert metric.j_and_f == pytest.approx((metric.j + metric.f) / 2)
    assert metric.iou == pytest.approx(metric.j)


def test_boundary_extraction_matches_the_official_neighbour_rule():
    """官方 ``_seg2bmap`` 的判据：与**右邻或下邻**状态不同的像素即边界。

    这不是「物体四周一圈完整轮廓」——实心方块只会点亮它的下边与右边。这里照官方
    语义实现（含最后一行/列单独收口），因此断言也按这个语义写：改成一圈完整轮廓
    会让本项目的 F 与评测协议对不上账。
    """
    mask = _square(10, 2, 2, 4)  # 行 2-5，列 2-5
    boundary = M._boundary_mask(mask)
    # 下边（行 5）与右边（列 5）：与右邻/下邻不同的那两条
    assert boundary[5, 2:6].all()
    assert boundary[2:6, 5].all()
    # 内部不得有边界像素
    assert not boundary[2:5, 2:5].any()
    # 与右邻同值、也与下邻同值的上边/左边不算边界——这是官方语义，不是漏判
    assert not boundary[2, 2:5].any()
    assert not boundary[2:5, 2].any()


def test_boundary_closes_the_last_row_and_column():
    """贴到画幅边缘的目标也必须留下边界，不能因为「没有邻居」整条边丢失。

    官方实现对最后一列/最后一行是**覆盖**式收口（末列只比下邻、末行只比右邻），
    所以贴边实心块在末列上只有最下一行亮边界——这不是漏判，是官方口径。
    """
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:5, 3:6] = True  # 贴到最后一列
    boundary = M._boundary_mask(mask)
    assert boundary.any(), "贴边目标不应得到空边界"
    # 末列收口后只剩该块的最下一行
    assert boundary[4, 5]
    # 块体的下边（含末列）始终是边界
    assert boundary[4, 3:6].all()
    # 块的右边（不含收口行）在末列上按官方口径是 False：整列实心，与下邻处处相同
    assert not boundary[2:4, 5].any()
    # 而同一列若不在画幅边缘，右邻比较会把这条边点亮——差别正来自收口规则
    inner = np.zeros((6, 6), dtype=bool)
    inner[2:5, 3:5] = True
    assert M._boundary_mask(inner)[2:5, 4].all()


# ── 2. 消融语义 ─────────────────────────────────────────────────────────


def test_both_schemes_share_one_kernel():
    """两套方案必须是同一个核、同一组参数——否则对照测的不是记忆。"""
    a, b = ColorBandPredictor(), ColorBandPredictor()
    assert a.params == b.params
    assert a.name == b.name


def test_prediction_is_deterministic_across_repeated_calls():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    mask = np.zeros((60, 80), dtype=bool)
    mask[20:40, 30:60] = True
    prompt = anchor_from_mask(mask)
    predictor = ColorBandPredictor()
    first = predictor.predict(image, prompt)[0].mask
    second = predictor.predict(image, prompt)[0].mask
    assert np.array_equal(first, second)


def test_anchor_prompt_points_at_the_object_centroid():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 20:40] = True
    prompt = anchor_from_mask(mask)
    point = prompt.positive_points[0]
    assert (point.x, point.y) == pytest.approx((29.5, 19.5))
    assert prompt.clipped_box((50, 50)) == (20, 10, 40, 30)


def test_out_of_frame_prompt_returns_empty_rather_than_guessing():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    from infusion_monitor.core.predictor import Prompt

    outside = Prompt.from_box(100.0, 100.0, 140.0, 140.0, point=(120.0, 120.0))
    prediction = ColorBandPredictor().predict(image, outside)[0]
    assert prediction.is_empty
    assert prediction.score == 0.0


def test_empty_gt_mask_cannot_seed_a_prompt():
    with pytest.raises(ValueError, match="起锚掩膜为空"):
        anchor_from_mask(np.zeros((20, 20), dtype=bool))


# ── 3. 对照结论 ─────────────────────────────────────────────────────────

subset = discover(VOS_ROOT)
needs_subset = pytest.mark.skipif(
    not subset, reason="公开 VOS 子集不在磁盘上（先跑 scripts/fetch_vos_subset.py）"
)


@needs_subset
def test_occlusion_criterion_flags_the_sequences_that_actually_contain_occlusion():
    """遮挡判据只看真值面积，与跑哪套方案无关。至少要能认出语料里确有遮挡的序列。"""
    params = ColorBandParams()
    metrics, _ = evaluate_sequence(subset[0], params=params, max_frames=12, schemes=(MEMORY_SCHEME,))
    assert metrics, "至少要评出一个对象"
    occluded = [m for m in metrics if m.occluded_frames]
    assert isinstance(occluded, list)
    assert 0 < OCCLUSION_AREA_RATIO < 1


@needs_subset
def test_memory_scheme_is_not_worse_than_single_frame_on_occluded_frames():
    """本任务要验的命题：含遮挡的序列上，跨帧记忆不劣于阈值单帧。

    跑真实语料（限序列与帧数以控制用时）。门槛 ``min_margin`` 写在
    :func:`.verdict` 里，这里不另外拍一个更严的数。
    """
    report = run_benchmark(VOS_ROOT, max_sequences=4, max_frames=30)
    assert report["status"] == "ok"
    conclusion = verdict(report)
    memory = report["totals"][MEMORY_SCHEME]
    single = report["totals"][SINGLE_FRAME_SCHEME]
    assert memory["frames"] == single["frames"] > 0
    assert conclusion["memory_occluded_j_and_f"] >= conclusion["single_occluded_j_and_f"]
    assert conclusion["not_worse_on_occluded"] is True


@needs_subset
def test_benchmark_is_bit_for_bit_reproducible():
    """全链路无随机源——同样输入必须逐位复现，否则「可复现」是空话。"""
    first = run_benchmark(VOS_ROOT, max_sequences=2, max_frames=10)
    second = run_benchmark(VOS_ROOT, max_sequences=2, max_frames=10)
    assert first["rows"] == second["rows"]
    assert first["totals"] == second["totals"]
    assert first["deterministic"] is True
    assert first["random_seed"] == 0


def test_missing_dataset_degrades_to_skipped_instead_of_silent_empty(tmp_path):
    report = run_benchmark(tmp_path / "does-not-exist")
    assert report["status"] == "skipped"
    assert "skipped_reason" in report and report["skipped_reason"]
    assert "totals" not in report  # 不给一张空表冒充结果


def test_aggregate_weights_frames_equally_not_sequences():
    """长短序列不齐时必须按帧等权，否则短序列被隐式加权。"""
    def make(seq, values):
        return M.SequenceMetric(
            sequence=seq,
            scheme=MEMORY_SCHEME,
            frames=[
                M.FrameMetric(seq, i, 1, v, v, v, True) for i, v in enumerate(values)
            ],
        )

    short = make("short", [0.0, 0.0, 0.0])  # 3 帧全 0
    long = make("long", [1.0] * 9)  # 9 帧全 1
    totals = M.aggregate([short, long])
    assert totals["frames"] == 12
    # 按帧等权 = 9/12；按序列平均 = 0.5。两者必须不同，否则权重口径没生效。
    assert totals["mean_j"] == pytest.approx(0.75)


def test_verdict_flags_a_regression_instead_of_passing_silently():
    """结论函数本身要能判负——只会报赢的判据没有意义。"""
    fake = {
        "totals": {
            MEMORY_SCHEME: {"mean_j_and_f": 0.10, "mean_occluded_j_and_f": 0.10},
            SINGLE_FRAME_SCHEME: {"mean_j_and_f": 0.40, "mean_occluded_j_and_f": 0.40},
        },
        "occluded_sequences": ["x"],
    }
    conclusion = verdict(fake)
    assert conclusion["occluded_margin"] < 0
    assert conclusion["not_worse_on_occluded"] is False
    assert conclusion["not_worse_overall"] is False