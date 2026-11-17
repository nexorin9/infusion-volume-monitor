"""M2 面积序列平滑测试：中值/一阶平滑、物理单调约束、缺帧受限重建。

对应 task 19「原始面积比抖动会直接污染毫升与 ETA」。四组断言：

1. **双序列并存** —— raw 一个字节都没改，smoothed/trusted/reconstructed 与它逐帧对齐；
2. **尖峰被点名** —— 反光高光被中值吸收并标 ``spike``，滴斗空管造成的假面积跳变
   （物理上不可能的上升）被标 ``outlier`` 且**保留原值**，可信序列不跟着往上走；
3. **缺帧不插值** —— 遮挡帧三条序列里都是 ``null``；只有前后端点都是可信观测且
   缺帧数在区间内才受限重建，其余五种情况给拒绝原因码；
4. **接进主路径** —— 上传响应的毫升数/ETA 读可信序列、曲线诊断随响应返回、参数
   可由环境变量整组覆盖。

画面是**合成渲染帧**（``tests/support/synth.py``），不是真实院内短片——真实短片需医院
授权采集，本轮以合成 + 公开 VOS 子集替代。
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import pytest

from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
from infusion_monitor.core.propagate import analyse_frames, load_rgb, slot_seed_prompt
from infusion_monitor.core.smoothing import (
    FLAG_MISSING,
    FLAG_OBSERVED,
    FLAG_OUTLIER,
    FLAG_RECONSTRUCTED,
    FLAG_SPIKE,
    MIN_MEDIAN_POINTS,
    REJECT_NO_BACK,
    REJECT_NO_FRONT,
    REJECT_ORDER,
    REJECT_TOO_LONG,
    REJECT_TOO_SHORT,
    SmoothingParams,
    find_gaps,
    first_order,
    median_filter,
    reconstruct_gaps,
    smooth_series,
)
from tests.support.synth import BagSpec, RackScene, default_rack_scene, write_rack_frames

FRAME_COUNT = 18


def _draining(frames: int = 24, start: float = 0.60, step: float = 0.004) -> list[float]:
    return [round(start - step * index, 6) for index in range(frames)]


def _curve_from_scene(tmp_path: Path, scene: RackScene, predictor, **variant):
    paths = write_rack_frames(tmp_path, scene, **variant)
    shape = load_rgb(paths[0]).shape
    seeds = [
        (bag.bag_seq, slot_seed_prompt(shape, slot, len(scene.bags)))
        for slot, bag in enumerate(scene.bags)
    ]
    return {
        curve.obj_id: curve for curve in analyse_frames(paths, seeds, predictor)
    }


# ── 1. 双序列并存：不改原值、不删帧、不移位 ──────────────────────────────
def test_raw_series_is_returned_verbatim():
    raw = _draining(10)
    result = smooth_series(raw)
    assert result.raw == raw
    assert len(result.smoothed) == len(result.trusted) == len(result.reconstructed) == len(raw)
    assert len(result.flags) == len(raw)


def test_series_lengths_never_shrink_when_frames_go_missing():
    raw = [0.60, 0.58, None, None, 0.52, 0.50]
    result = smooth_series(raw)
    for series in (result.raw, result.smoothed, result.trusted, result.reconstructed):
        assert len(series) == len(raw)
    assert result.raw[2] is None and result.smoothed[2] is None


def test_median_filter_lag_on_a_monotone_ramp_is_bounded_by_half_the_window():
    """中值滤波保边缘、代价是**等斜坡会滞后约半个窗口**——固有性质，不是缺陷。

    但滞后必须被窗口半宽约束住：滞后再大就把「降了多少」算小，ETA 跟着偏。
    """
    raw = _draining(9)
    window = SmoothingParams().median_window
    medians = median_filter(raw, window=window)
    assert medians == sorted(medians, reverse=True), "平滑不得把单调下降弄成上升"
    bound = (window // 2) * abs(raw[0] - raw[1]) + 1e-9
    assert max(abs(m - r) for m, r in zip(medians, raw)) <= bound


def test_median_filter_does_not_inflate_the_first_frame():
    """窗口贴边时整体内移而不是截短——截短只剩两点，退化成取均值当「中值」。

    首帧被抬高 → 紧接着的正常帧就成了「下降」，再被误判成反常，后面整条可信序列
    都会跟着歪。这条断言钉住修复后的口径：首帧取前三点的**中值**（0.596），
    而不是均值（0.598），且整段不会被判出任何反常帧。
    """
    raw = [0.600, 0.596, 0.592, 0.588, 0.584, 0.580]
    medians = median_filter(raw, window=3)
    assert medians[0] == 0.596
    assert medians[0] != (raw[0] + raw[1]) / 2
    result = smooth_series(raw)
    assert result.outlier_frames == []


def test_median_filter_passes_through_when_there_is_no_median_to_take():
    """可用观测点不足 3 个时原样返回：三个观测点里取不出稳定中值。"""
    assert median_filter([0.5, 0.9], window=3) == [0.5, 0.9]


def test_first_order_rejects_out_of_range_alpha():
    with pytest.raises(ValueError):
        first_order([0.5, 0.4], alpha=0.0)
    with pytest.raises(ValueError):
        median_filter([0.5], window=0)


# ── 2. 尖峰与物理单调约束 ────────────────────────────────────────────────
def test_glare_spike_is_absorbed_and_flagged():
    """反光高光：单帧面积比跳高 0.3，中值吸收，可信序列继续跟住真实下降。"""
    raw = [0.600, 0.596, 0.880, 0.588, 0.584, 0.580]
    result = smooth_series(raw)
    assert result.spike_frames == [2]
    assert result.outlier_frames == []
    assert result.flags[2].label == FLAG_SPIKE
    # 原始值一点没动——护士/信息科要能看到这一帧拍成了什么样
    assert result.raw[2] == 0.880
    # 可信序列既没跟着尖峰跳上去，也没掉下来
    assert max(result.trusted) <= 0.601
    assert result.trusted[2] < 0.600
    assert result.flags[2].reconstructed is False


def test_physically_impossible_rise_is_flagged_but_kept():
    """液面只会降。观测到上升 = 这一帧拍坏了（滴斗空管 / 袋体被挡）。"""
    raw = [0.500, 0.520, 0.550, 0.540]
    result = smooth_series(raw)
    assert 2 in result.outlier_frames
    assert result.raw == raw
    assert result.smoothed[2] == 0.550, "反常帧保留原值，不许被平滑抹平"
    assert result.flags[2].label == FLAG_OUTLIER


def test_trusted_series_never_rises_even_when_observations_do():
    raw = [0.500, 0.520, 0.550, 0.540]
    result = smooth_series(raw)
    values = [value for value in result.trusted if value is not None]
    assert values == sorted(values, reverse=True), "可信序列是下游读的那条，绝不能上升"


def test_outlier_frames_fall_back_to_the_last_trusted_reading():
    raw = [0.500, 0.520, 0.550]
    result = smooth_series(raw)
    assert result.trusted[1] == pytest.approx(result.trusted[0])
    assert result.trusted[2] == pytest.approx(result.trusted[0])
    assert result.latest_trusted < result.latest_raw


def test_envelope_uses_running_max_not_previous_frame():
    """袋体晃动会让相邻帧上下抖零点几个百分点；用上一帧当基准会一路累积成「全程反常」。"""
    base = 0.50
    raw = [base + (0.002 if index % 2 else -0.002) for index in range(8)]
    result = smooth_series(raw, params=SmoothingParams(rise_tolerance=0.004))
    assert result.outlier_frames == []


# ── 3. 缺帧：留 null + 受限重建 ──────────────────────────────────────────
def test_single_occluded_frame_stays_null_in_every_series():
    """缺 1 帧不补：四条序列里都是 null，标 missing。"""
    raw = [0.60, 0.58, None, 0.50, 0.48]
    result = smooth_series(raw)
    assert result.smoothed[2] is None
    assert result.trusted[2] is None
    assert result.reconstructed[2] is None
    assert result.frame_label(2) == FLAG_MISSING
    assert result.missing_frames == [2]


def test_short_gap_is_not_reconstructed():
    raw = [0.60, 0.58, None, 0.56]
    result = smooth_series(raw)
    assert result.reconstructed == [None, None, None, None]
    assert [gap.reason for gap in result.gaps] == [REJECT_TOO_SHORT]


def test_gap_inside_the_window_is_reconstructed_and_marked():
    raw = [0.60, 0.58, 0.56, None, None, None, 0.50, 0.48]
    result = smooth_series(raw)
    front, back = result.trusted[2], result.trusted[6]
    assert result.reconstructed[3:6] == pytest.approx(
        [front + (back - front) * step / 4.0 for step in (1, 2, 3)]
    ), "补线要正好接在两端的平滑值上（端点取可信序列，不取中值序列）"
    assert result.reconstructed[2] is None and result.reconstructed[6] is None
    assert result.reconstructed_frames == [3, 4, 5]
    assert result.frame_label(4) == FLAG_RECONSTRUCTED
    assert result.gaps[0].filled is True
    assert result.gaps[0].reason == "reconstructed_from_endpoints"


def test_reconstruction_stays_physically_monotone():
    raw = [0.60, 0.58, 0.56, None, None, None, 0.50, 0.48]
    result = smooth_series(raw)
    values = [
        result.trusted[i] if result.trusted[i] is not None else result.reconstructed[i]
        for i in range(len(raw))
    ]
    values = [value for value in values if value is not None]
    assert values == sorted(values, reverse=True)


def test_gap_longer_than_max_is_refused():
    raw = [0.60, 0.58] + [None] * 12 + [0.30, 0.28]
    result = smooth_series(raw)
    assert all(value is None for value in result.reconstructed)
    assert [gap.reason for gap in result.gaps] == [REJECT_TOO_LONG]


def test_gap_without_back_endpoint_is_refused():
    raw = [0.60, 0.58, None, None, None]
    result = smooth_series(raw)
    assert all(value is None for value in result.reconstructed)
    assert [gap.reason for gap in result.gaps] == [REJECT_NO_BACK]


def test_gap_without_front_endpoint_is_refused():
    raw = [None, None, None, 0.50, 0.48]
    result = smooth_series(raw)
    assert [gap.reason for gap in result.gaps] == [REJECT_NO_FRONT]


def test_refill_between_endpoints_blocks_reconstruction():
    """换过袋的那几帧被点名为反常之后，后端点就不成立——宁可留空也不补一条假线。"""
    raw = [0.60, 0.58, 0.56, 0.54, 0.52, None, None, None, 0.90, 0.88, 0.86, 0.84]
    result = smooth_series(raw)
    assert all(value is None for value in result.reconstructed)
    reasons = [gap.reason for gap in result.gaps]
    assert REJECT_ORDER in reasons or REJECT_NO_BACK in reasons
    assert 8 in result.outlier_frames, "换液后的新袋整段被标反常，不接到旧曲线上"


def test_find_gaps_returns_closed_intervals():
    assert find_gaps([1.0, None, None, 2.0, None]) == [(1, 2), (4, 4)]


def test_all_missing_series_has_no_reading():
    result = smooth_series([None, None, None])
    assert result.latest_raw is None
    assert result.latest_trusted is None
    assert result.drain_ratio_per_frame is None
    assert result.degraded is True


def test_drain_rate_uses_frame_indices_not_list_positions():
    """帧号不连续时（重传只留下部分帧），降幅要按帧号差算，不是按列表下标差。"""
    params = SmoothingParams(min_gap_frames=1, max_gap_frames=20)
    sparse, _ = reconstruct_gaps(
        [0.60, None, 0.48],
        frame_indices=[0, 1, 12],
        endpoints=[True, False, True],
        params=params,
    )
    # 补值按帧号跨度插值：第 1 帧落在 0.60→0.48 这条线的 1/12 处
    assert sparse[1] == pytest.approx(0.60 - 0.12 / 12)
    contiguous, _ = reconstruct_gaps(
        [0.60, None, 0.48],
        frame_indices=[0, 1, 2],
        endpoints=[True, False, True],
        params=params,
    )
    # 按列表下标算会得到中点 0.54 —— 两者必须不同，否则这条断言没有区分力
    assert contiguous[1] == pytest.approx(0.54)

    # 平均降幅同样按帧号跨度：重传只剩三格时，跨度是 12 帧而不是 2 格
    result = smooth_series([0.60, None, 0.48], frame_indices=[0, 1, 12], params=params)
    assert result.drain_ratio_per_frame == pytest.approx(
        (result.trusted[0] - result.trusted[2]) / 12
    )
    assert result.drain_ratio_per_frame < 0.02, "按列表下标算会得到 0.036 量级"


def test_reconstruct_rejects_degenerate_frame_span():
    values, gaps = reconstruct_gaps(
        [0.60, None, 0.50],
        params=SmoothingParams(min_gap_frames=1, max_gap_frames=10),
        frame_indices=[5, 5, 5],
    )
    assert values == [None, None, None]
    assert gaps[0].filled is False


def test_min_median_points_is_at_least_three():
    assert MIN_MEDIAN_POINTS == 3


# ── 4. 接进主路径 ────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def predictor() -> GeometricMaskPredictor:
    return GeometricMaskPredictor()


def _non_increasing(values: list[float], tolerance: float = 1e-5) -> bool:
    """单调不增判定（带浮点容差：分割结果本身就有 1e-6 量级的抖动）。"""
    return all(
        later <= earlier + tolerance for earlier, later in zip(values, values[1:])
    )


def _lag(result, curve) -> float:
    """平滑序列与原始观测的最大逐帧偏差（只统计两边都有值的帧）。"""
    return max(
        abs(value - raw)
        for value, raw in zip(result.trusted, curve.area_ratio_series)
        if value is not None and raw is not None
    )


def test_smoothing_does_not_move_a_slow_draining_curve(tmp_path, predictor):
    """真实速度的短片（18 帧共降 0.5 个百分点）：逐帧偏差 < 0.001。

    这一条才是「能不能上线」的判据：真实滴速下 10–20 秒的液面变化本来就在千分之几，
    平滑带来的滞后必须远小于它，否则等于把余量读数系统性拖慢。
    """
    scene = RackScene(
        bags=(BagSpec("B01", 0.28, 0.30, 0.620, 0.615, (92, 148, 196)),),
        frame_count=FRAME_COUNT,
    )
    curve = _curve_from_scene(tmp_path / "slow", scene, predictor)["B01"]
    result = smooth_series(curve.area_ratio_series, frame_indices=curve.frame_indices)
    assert _lag(result, curve) < 0.001
    assert _non_increasing([value for value in result.trusted if value is not None])


def test_smoothing_lag_is_bounded_by_the_window_width_on_a_steep_ramp(tmp_path, predictor):
    """合成场景是压力级斜坡（18 帧降 22 个百分点，每帧约 1.3%——真实短片不可能）。

    任何一阶平滑都会滞后，滞后量级约「窗口半宽 + (1-alpha)/alpha 个步长」。这条断言
    钉住它的**上界**：超过窗口宽度就说明平滑在系统性拖慢读数，「降了多少」会被算小。
    真实滴速下每帧只有千分之几，同样的滞后换算过来是 0.0006，够不着换液判定阈值。
    """
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    curve = _curve_from_scene(tmp_path / "steep", scene, predictor)["B01"]
    step = abs(curve.area_ratio_series[-1] - curve.area_ratio_series[0]) / (FRAME_COUNT - 1)
    result = smooth_series(curve.area_ratio_series, frame_indices=curve.frame_indices)
    window = SmoothingParams().median_window
    assert _lag(result, curve) <= (window / 2 + 2) * step
    assert result.raw == curve.area_ratio_series


def test_glare_variant_is_absorbed_without_touching_the_raw_series(tmp_path, predictor):
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    curve = _curve_from_scene(tmp_path / "glare", scene, predictor, highlight=True)["B01"]
    result = smooth_series(curve.area_ratio_series, frame_indices=curve.frame_indices)
    assert result.raw == curve.area_ratio_series, "原始序列一个字节都没改"
    assert _non_increasing([v for v in result.trusted if v is not None]), "可信序列不因高光而抖动"
    step = abs(curve.area_ratio_series[-1] - curve.area_ratio_series[0]) / (FRAME_COUNT - 1)
    window = SmoothingParams().median_window
    assert _lag(result, curve) <= (window / 2 + 2) * step


def test_occluded_rendered_gap_is_reconstructed_only_when_recovered(tmp_path, predictor):
    """挡板盖住的帧：三条序列都留 null，恢复后按端点受限重建并逐帧标名。"""
    scene = default_rack_scene(frame_count=FRAME_COUNT)
    curve = _curve_from_scene(tmp_path / "occl", scene, predictor, occluders=[(0, 6, 8)])["B01"]
    gap = [i for i, value in enumerate(curve.area_ratio_series) if value is None]
    assert gap, "这一段应当有被挡住的帧"
    result = smooth_series(curve.area_ratio_series, frame_indices=curve.frame_indices)
    assert all(result.smoothed[i] is None and result.trusted[i] is None for i in gap)
    assert result.reconstructed_frames == gap
    assert [result.frame_label(i) for i in gap] == [FLAG_RECONSTRUCTED] * len(gap)
    assert result.gaps[0].filled is True
    # 补出来的线不与任何一端的实测值打架
    front, back = result.trusted[gap[0] - 1], result.trusted[gap[-1] + 1]
    assert result.reconstructed[gap[0]] == pytest.approx(front + (back - front) / (len(gap) + 1))


def test_rendered_refill_is_flagged_and_ema_restarts(tmp_path, predictor):
    """第 9 帧起换新袋：上升的那几帧被点名，一阶平滑在异常块之后**重新起算**。

    不重新起算的话，新袋的读数会一直和旧袋的液位加权，报出来的余量两头都不像。
    """
    scene = RackScene(
        bags=(
            BagSpec("B01", 0.28, 0.30, 0.62, 0.40, (92, 148, 196), refill_at=9, refill_to=0.95),
        ),
        frame_count=FRAME_COUNT,
    )
    curve = _curve_from_scene(tmp_path / "refill", scene, predictor, occluders=[(0, 5, 8)])["B01"]
    result = smooth_series(curve.area_ratio_series, frame_indices=curve.frame_indices)
    assert result.outlier_frames, "换液后的液面上升必须被点名"
    assert max(result.outlier_frames) >= 9
    # 异常块之后的第一帧：可信值 == 中值（一阶平滑重新起算，不与旧液位加权）
    resume = max(result.outlier_frames) + 1
    assert result.trusted[resume] == pytest.approx(
        median_filter(result.raw, window=3)[resume]
    )
    # 换液段之后不再补线：不会拿旧袋的液面去桥接新袋
    assert not [index for index in result.reconstructed_frames if index > max(result.outlier_frames)]


def test_upload_response_carries_both_series_and_diagnostics(client):
    body = _upload(client, bag_type_code=None)
    bag = body["bags"][0]
    assert len(bag["area_ratio_series"]) == len(bag["area_ratio_smoothed"])
    assert len(bag["area_ratio_reconstructed"]) == len(bag["area_ratio_series"])
    assert bag["area_ratio_series"] != bag["area_ratio_smoothed"], "双序列不是同一份数据"
    smoothing = bag["smoothing"]
    assert set(smoothing) >= {
        "outlier_frames",
        "spike_frames",
        "reconstructed_frames",
        "missing_frame_count",
        "drain_ratio_per_frame",
        "latest_raw",
        "latest_trusted",
        "gaps",
        "params",
    }
    assert smoothing["params"]["median_window"] == 5


def test_upload_millilitres_come_from_the_trusted_series(client):
    _register_type_a(client)
    body = _upload(client, bag_type_code="TYPE-A")
    bag = body["bags"][0]
    assert bag["calibration_state"] == "calibrated"
    assert bag["smoothing"]["latest_trusted"] == pytest.approx(bag["fill_ratio"], abs=1e-9)
    assert bag["estimated_volume_ml"] > 0


def test_uncalibrated_bag_still_reports_the_smoothed_series(client):
    body = _upload(client, bag_type_code=None)
    bag = body["bags"][0]
    assert bag["calibration_state"] == "uncalibrated"
    assert "estimated_volume_ml" not in bag, "门禁语义不能被平滑绕过去"
    assert bag["area_ratio_smoothed"]


def test_smoothing_params_are_env_configurable(client_factory, monkeypatch):
    monkeypatch.setenv("SMOOTHING_MEDIAN_WINDOW", "7")
    monkeypatch.setenv("SMOOTHING_ALPHA", "0.9")
    monkeypatch.setenv("SMOOTHING_MAX_GAP_FRAMES", "4")
    with client_factory() as client:
        body = _upload(client, bag_type_code=None)
        params = body["bags"][0]["smoothing"]["params"]
    assert params["median_window"] == 7
    assert params["alpha"] == 0.9
    assert params["max_gap_frames"] == 4


@pytest.mark.parametrize(
    "name, value",
    [
        ("SMOOTHING_ALPHA", "1.5"),          # 一阶平滑权重只允许 (0, 1]
        ("SMOOTHING_RISE_TOLERANCE", "1.2"),  # 容差只允许 [0, 1)
        ("SMOOTHING_MEDIAN_WINDOW", "0"),     # 窗口至少 1
    ],
)
def test_smoothing_params_reject_nonsense(credential_env, monkeypatch, name, value):
    """平滑参数配错就是「还早着呢」被误报成「快见底了」，所以要**当场拒绝**而不是照单全收。"""
    from pydantic import ValidationError

    from infusion_monitor.config import Settings

    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError) as excinfo:
        Settings()
    assert name.lower() in str(excinfo.value).lower()


def test_max_gap_must_cover_min_gap(credential_env, monkeypatch):
    from pydantic import ValidationError

    from infusion_monitor.config import Settings

    monkeypatch.setenv("SMOOTHING_MIN_GAP_FRAMES", "8")
    monkeypatch.setenv("SMOOTHING_MAX_GAP_FRAMES", "4")
    with pytest.raises(ValidationError):
        Settings()


def test_smoothing_cost_does_not_grow_with_frame_count():
    """恒定算力：帧数翻 8 倍，单帧耗时不能线性膨胀（任务 6/15/16 的同一条纪律）。"""
    short = _draining(120)
    long = _draining(960)

    def _best_ms(series: list[float], repeats: int = 5) -> float:
        """取若干次里的最好一次：单帧只有几微秒，调度抖动比计算本身大得多。"""
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            smooth_series(series)
            best = min(best, (time.perf_counter() - start) / len(series) * 1000.0)
        return best

    _best_ms(short)  # 预热，避免把首次 import/JIT 计进单帧耗时
    short_ms = _best_ms(short)
    long_ms = _best_ms(long)
    assert long_ms < short_ms * 3.0, f"单帧耗时 {short_ms:.5f} → {long_ms:.5f} ms 涨得太多"


def test_smoothing_peak_memory_is_bounded():
    """1200 帧曲线的峰值内存受控（不是把整段复制多份）。"""
    series = _draining(1200)
    tracemalloc.start()
    try:
        smooth_series(series)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024, f"峰值 {peak / 1024:.0f} KB 超出预算"


def test_settings_expose_smoothing_params(credential_env):
    from infusion_monitor.config import get_settings

    settings = get_settings()
    assert isinstance(settings.smoothing_params, SmoothingParams)
    assert settings.describe()["smoothing"]["alpha"] == settings.smoothing_alpha


def _register_type_a(client) -> None:
    """登记一条 TYPE-A 标定（样例值，非真实称重数据），让毫升数能出得来。"""
    created = client.post(
        "/api/v1/infusion/calibrations",
        json={
            "bag_type_code": "TYPE-A",
            "capacity_ml": 600.0,
            "dead_volume_ml": 20.0,
            "samples": [
                {"measured_fill_ratio": 0.10, "known_volume_ml": 58.0, "source": "称重法"},
                {"measured_fill_ratio": 0.60, "known_volume_ml": 378.0, "source": "称重法"},
                {"measured_fill_ratio": 0.90, "known_volume_ml": 520.0, "source": "换液确认反推"},
            ],
            "calibrated_by": "OP-NURSE-EXAMPLE",
            "note": "演示用样例值，非真实称重数据",
        },
    )
    assert created.status_code == 201, created.text


def _upload(client, *, bag_type_code: str | None, **variant):
    """用合成帧走一次真实上传，返回解析后的响应体。"""
    import json

    from tests.support import synth

    scene = default_rack_scene(frame_count=FRAME_COUNT)
    frames = synth.build_rack_zip(scene, **variant)
    slots = [{"bag_seq": bag.bag_seq, "slot_index": slot} for slot, bag in enumerate(scene.bags)]
    if bag_type_code:
        for item in slots:
            item["bag_type_code"] = bag_type_code
            item["rate_text"] = "5ml/h"
    response = client.post(
        "/api/v1/infusion/tracks",
        data={
            "rack_id": "R-SMOOTH-01",
            "infusion_order_no": "EXAMPLE-ORDER-0019",
            "bed_no": "B-EXAMPLE-19",
            "captured_at": "2026-01-01T09:00:00+08:00",
            "slots": json.dumps(slots, ensure_ascii=False),
        },
        files={"frames": ("frames.zip", frames, "application/zip")},
    )
    assert response.status_code == 201, response.text
    return response.json()