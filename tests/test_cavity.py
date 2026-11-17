"""袋型腔体几何与自校准的回归。

真值来源是 :mod:`infusion_monitor.synth` 的 :class:`~infusion_monitor.synth.CavityModel`
——那是**合成短片生成器**自己的积分实现，与被测的 :mod:`infusion_monitor.calibration.cavity`
是两条独立代码路径（一条按剖面积分给真值，一条用查表 + 二分做反解）。两者参数同族，
所以「反解是否真能拿回积分给的那份体积」是可证伪的，而不是自说自话。
"""

from __future__ import annotations

import json

import pytest

from infusion_monitor.calibration.cavity import (
    MAX_RESIDUE_EVIDENCE_RATIO,
    MIN_DEVIATION_SAMPLES,
    READABLE_FLOOR_ML,
    CavityModel,
    CavitySpec,
    DeviationObservation,
    deviation_observation_from_session,
    learn_residual,
)
from infusion_monitor.calibration.estimator import (
    RateResolution,
    estimate_bag,
)
from infusion_monitor.calibration.models import BagTypeCalibration, CalibrationSample
from infusion_monitor.store.repositories import DeviationRepository
from infusion_monitor.synth import BAG_TYPE_CATALOG, bag_profile, cavity_model
from infusion_monitor.store.migrations import apply_migrations, current_version
from infusion_monitor.store.schema import SCHEMA_VERSION

from .support import synth

#: 参与回归的液位：2% 是「快见底」，90% 是「快满」，中间几档覆盖常规巡视频段。
LEVELS = (0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)

#: 毫升数预算（相对标签容量）。与 task 8 的毫升回归同一量级：0.8%。
ML_BUDGET = 0.008


def spec_for(profile) -> CavitySpec:
    """合成袋型档案 → 腔体参数登记（生产侧拿到的就是这份外形参数）。"""
    return CavitySpec(
        bag_type_code=profile.bag_type_code,
        capacity_ml=profile.capacity_ml,
        dead_volume_ml=profile.dead_volume_ml,
        shape_exponent=profile.shape_p,
        taper=profile.perspective,
        bulge=profile.bulge,
        label_zh=profile.label_zh,
        registered_by="OP-NURSE-01",
        # 登记入口分配的腔体几何版本号（首次登记 = 第 1 版）。几何换算的毫升数带
        # 不确定度，**参数改了不升版本就出不了数**（calibration.uncertainty 的
        # geometry_version_missing 门禁），所以这里必须带版本。
        geometry_revision=1,
    )


# ── 1 多袋型换算回归 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("profile", BAG_TYPE_CATALOG, ids=lambda p: p.bag_type_code)
def test_cavity_inversion_recovers_true_volume_for_every_bag_type(profile):
    """观测量（面积比）反解出的体积，与真值积分的误差逐档在预算内。"""
    truth = cavity_model(profile)
    model = CavityModel(spec_for(profile))
    worst = 0.0
    for level in LEVELS:
        true_area = truth.projected_area_ratio(level)
        true_ml = truth.true_volume_ml(level)
        reading = model.volume(true_area)
        assert reading.readable_ml is not None, f"{profile.bag_type_code}@{level} 不该被门禁拦下"
        error = abs(reading.readable_ml - true_ml) / profile.capacity_ml
        worst = max(worst, error)
        assert error <= ML_BUDGET, (
            f"{profile.bag_type_code} 液位 {level:.0%}：腔体反解 {reading.readable_ml:.1f} ml "
            f"对真值 {true_ml:.1f} ml，相对误差 {error:.4%} 超过预算 {ML_BUDGET:.2%}"
        )
    assert worst < ML_BUDGET  # 真跑过一遍，不是恰好达标


@pytest.mark.parametrize("profile", BAG_TYPE_CATALOG, ids=lambda p: p.bag_type_code)
def test_cavity_beats_naive_linear_under_measurement_noise(profile):
    """分割有 ±1 个百分点误差时，腔体换算比「面积比 × 有效容量」更贴真值。

    预算放到 2%（噪声进腔体反解会被非线性放大），但仍要求**逐档都优于**朴素做法。
    """
    truth = cavity_model(profile)
    model = CavityModel(spec_for(profile))
    usable = profile.usable_capacity_ml
    cavity_errors: list[float] = []
    naive_errors: list[float] = []
    for level in LEVELS:
        if level < 0.10:
            continue  # 近空段另有用例（见 test_near_empty_is_flagged_not_extrapolated）
        true_area = truth.projected_area_ratio(level)
        true_ml = truth.true_volume_ml(level)
        for noise in (-0.01, 0.0, 0.01):
            measured = min(max(true_area + noise, 0.0), 1.0)
            predicted = model.volume(measured).readable_ml or 0.0
            cavity_errors.append(abs(predicted - true_ml))
            naive_errors.append(abs(measured * usable - true_ml))
    assert sum(cavity_errors) < sum(naive_errors), (
        f"{profile.bag_type_code}：腔体总误差 {sum(cavity_errors):.1f} ml 没有优于朴素做法 "
        f"{sum(naive_errors):.1f} ml"
    )
    assert max(cavity_errors) <= usable * 0.02, (
        f"{profile.bag_type_code}：最差一档 {max(cavity_errors):.1f} ml 超过 2% 预算"
    )


def test_naive_linear_carries_a_bag_type_systematic_bias_that_cavity_removes():
    """为什么值得多这一层：同一个液位，不同袋型的朴素误差差出一大截。

    task 8 实测过「观测量与真实体积分数之差随袋型变化」，这里换个角度：直接看毫升。
    """
    level = 0.30
    naive = {}
    cavity = {}
    for profile in BAG_TYPE_CATALOG:
        truth = cavity_model(profile)
        true_area = truth.projected_area_ratio(level)
        true_ml = truth.true_volume_ml(level)
        naive[profile.bag_type_code] = abs(true_area * profile.usable_capacity_ml - true_ml)
        cavity[profile.bag_type_code] = abs(
            CavityModel(spec_for(profile)).volume(true_area).readable_ml - true_ml
        )

    naive_spread = max(naive.values()) - min(naive.values())
    cavity_spread = max(cavity.values()) - min(cavity.values())
    assert naive_spread > 10.0, f"朴素做法本应有明显的袋型间偏差，实测 {naive_spread:.1f} ml"
    assert cavity_spread < 2.0, f"腔体换算的袋型间残差实测 {cavity_spread:.2f} ml，应压到 2 ml 内"
    assert cavity_spread * 4 < naive_spread


def test_volume_is_monotone_in_fill_ratio_and_anchored_at_both_ends():
    profile = bag_profile("TYPE-A")
    model = CavityModel(spec_for(profile))
    previous = None
    for step in range(0, 101):
        reading = model.volume(step / 100.0)
        value = reading.readable_ml
        assert value is not None
        if previous is not None:
            assert value >= previous - 1e-6, f"面积比涨了毫升数反而跌：{step / 100:.2f}"
        previous = value
    assert model.volume(0.0).readable_ml == 0.0
    assert model.volume(1.0).readable_ml == pytest.approx(profile.usable_capacity_ml, abs=0.5)


# ── 2 残留补偿（低液位的系统性偏差不当噪声忽略） ─────────────────────────────


def test_film_holdup_subtracts_proportionally_and_reports_its_source():
    profile = bag_profile("TYPE-A")
    spec = spec_for(profile)
    spec = CavitySpec(**{**spec.__dict__, "film_holdup_ml": 6.0})
    model = CavityModel(spec)
    truth = cavity_model(profile)

    reading = model.volume(truth.projected_area_ratio(0.50))
    expected_fraction = truth.volume_fraction(0.50)
    assert reading.resident_ml == pytest.approx(6.0 * expected_fraction, abs=0.05)
    # 与真值的差留在几何反解自身的精度内（预算 0.5% 有效容量），不多不少。
    assert reading.readable_ml == pytest.approx(
        truth.true_volume_ml(0.50) - 6.0 * expected_fraction, abs=profile.usable_capacity_ml * 0.005
    )
    assert reading.residual_source == "registered_film"


def test_near_empty_is_flagged_not_extrapolated():
    """液面 2% 与几乎排空：都报可核对的结果，不外推也不假装还有几毫升。"""
    profile = bag_profile("TYPE-B")
    model = CavityModel(spec_for(profile))
    truth = cavity_model(profile)

    at_2pct = model.volume(truth.projected_area_ratio(0.02))
    assert at_2pct.readable_ml is not None
    assert at_2pct.readable_ml > 0
    # 近空段腔体收窄，同样的面积比误差会被反解放大——所以必须同时报出近空标记，
    # 由护士端与不确定度引擎（task 28）据此放宽余量口径，而不是当普通读数用。
    assert at_2pct.height_fraction == pytest.approx(0.02, abs=0.01)

    # 液面低到腔体里那点液体与残留同一量级时：报 0 + residual_limited，而不是外推。
    drained_level = 0.0
    for step in range(1, 200):
        level = step / 4000.0
        if model.volume(truth.projected_area_ratio(level)).cavity_ml <= READABLE_FLOOR_ML:
            drained_level = level
            break
    assert drained_level > 0.0
    drained = model.volume(truth.projected_area_ratio(drained_level))
    assert drained.residual_limited is True
    assert drained.near_empty is True
    assert drained.readable_ml == 0.0
    assert "残留" in drained.reason or "0 毫升" in drained.reason

    empty = model.volume(0.0)
    assert empty.readable_ml == 0.0
    assert empty.residual_limited is True
    assert empty.reason == "cavity_empty"


def test_learned_residual_is_reported_as_its_own_source():
    profile = bag_profile("TYPE-A")
    model = CavityModel(spec_for(profile))
    truth = cavity_model(profile)
    reading = model.volume(
        truth.projected_area_ratio(0.50), learned_residual_ml=9.0
    )
    assert reading.resident_ml == pytest.approx(9.0, abs=0.01)
    assert reading.residual_source == "self_calibrated"
    assert reading.readable_ml == pytest.approx(
        truth.true_volume_ml(0.50) - 9.0, abs=profile.usable_capacity_ml * 0.005
    )


# ── 3 超装：显式告警而不是自圆其说 ───────────────────────────────────────────


def test_overfill_ratio_is_alarmed_not_rounded_down():
    model = CavityModel(spec_for(bag_profile("TYPE-A")))
    reading = model.volume(1.05)
    assert reading.overfilled is True
    assert reading.readable_ml is None, "超装时不得给毫升数"
    assert "重拍" in reading.reason
    assert "装不下" in reading.reason


def test_ratio_just_under_overfill_threshold_is_clamped_to_capacity():
    profile = bag_profile("TYPE-A")
    model = CavityModel(spec_for(profile))
    reading = model.volume(1.01)
    assert reading.overfilled is False
    assert reading.near_full is True
    assert reading.readable_ml == pytest.approx(profile.usable_capacity_ml, abs=0.2)


def test_missing_observation_is_not_zero_ml():
    reading = CavityModel(spec_for(bag_profile("TYPE-A"))).volume(None)
    assert reading.readable_ml is None
    assert reading.reason == "no_observation"


# ── 4 自校准：证据 → 偏差项 → 偏差收敛 ──────────────────────────────────────


def _level_for_true_volume(truth, target_ml: float) -> float:
    """反查「腔体里真的有 target_ml 液体」的液位。

    用二分而不是查表：这条辅助线要在 4 个袋型上都成立，查表还得给不同袋型配不同的
    粗细，徒增一处可能与被测实现同源的偏差。
    """
    low, high = 0.0, 1.0
    for _ in range(40):
        mid = (low + high) / 2.0
        if truth.true_volume_ml(mid) < target_ml:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _observation(profile, model, truth, residue_ml: float, *, tag: str = "") -> DeviationObservation:
    """造一条「护士确认换液」证据：这袋当时真实可读余量就是 ``residue_ml``（= 流不出的残留）。

    证据里记的 ``model_volume_ml`` 是**模型当时给的数**——也就是还没学残留时虚报的量。
    """
    level = _level_for_true_volume(truth, residue_ml)
    reading = model.volume(truth.projected_area_ratio(level))
    # 容差取有效容量的 0.5%：被测模型与真值生成器对「鼓胀项」的离散化方式不同
    # （一边逐格积分、一边用前缀和解析求和），差值落在这个量级，属已知建模差异。
    assert reading.readable_ml == pytest.approx(
        residue_ml, abs=profile.usable_capacity_ml * 0.005
    )
    return DeviationObservation(
        bag_type_code=profile.bag_type_code,
        last_fill_ratio=level,
        model_volume_ml=reading.readable_ml,
        session_id=f"sess-{tag or residue_ml}",
        bag_seq="B01",
        rack_id="R-07",
        bed_no="12",
        occurred_at="2026-10-22T20:15:00+08:00",
    )


def test_self_calibration_needs_enough_evidence_before_it_moves_any_number():
    profile = bag_profile("TYPE-A")
    model = CavityModel(spec_for(profile))
    truth = cavity_model(profile)
    samples = [
        _observation(profile, model, truth, residue_ml=value, tag=f"r{index}")
        for index, value in enumerate((6.8, 7.0, 7.1, 7.2))
    ]

    one = learn_residual(samples[:1], usable_capacity_ml=profile.usable_capacity_ml)
    assert one.state == "insufficient_samples"
    assert one.applied is False
    assert one.effective_residual_ml == 0.0
    assert one.sample_count == 1

    two = learn_residual(samples[:2], usable_capacity_ml=profile.usable_capacity_ml)
    assert two.applied is False
    assert "2 条" in two.reason

    three = learn_residual(
        samples[:MIN_DEVIATION_SAMPLES], usable_capacity_ml=profile.usable_capacity_ml
    )
    assert three.state == "learned"
    assert three.applied is True
    assert three.sample_count == MIN_DEVIATION_SAMPLES
    assert three.residual_ml == pytest.approx(7.0, abs=1.5)


def test_self_calibration_closes_the_low_end_bias_after_convergence():
    """自校准生效后：低液位不再虚报，全量程与真值（已扣残留）逐点对齐。"""
    profile = bag_profile("TYPE-A")
    model = CavityModel(spec_for(profile))
    truth = cavity_model(profile)
    residue = 7.0

    samples = [
        _observation(profile, model, truth, residue_ml=value, tag=f"r{index}")
        for index, value in enumerate((6.6, 6.9, 7.0, 7.2, 7.4))
    ]
    learned = learn_residual(samples, usable_capacity_ml=profile.usable_capacity_ml)
    assert learned.applied is True

    uncorrected_errors = []
    corrected_errors = []
    for level in LEVELS:
        measured = truth.projected_area_ratio(level)
        true_readable = truth.true_volume_ml(level) - residue
        before = model.volume(measured).readable_ml or 0.0
        after = model.volume(
            measured, learned_residual_ml=learned.effective_residual_ml
        ).readable_ml or 0.0
        uncorrected_errors.append(abs(before - max(true_readable, 0.0)))
        corrected_errors.append(abs(after - max(true_readable, 0.0)))

    assert sum(corrected_errors) < sum(uncorrected_errors) * 0.2, (
        f"自校准后总残差 {sum(corrected_errors):.2f} ml 应收敛到原来的 1/5 以内，"
        f"实测未收敛前 {sum(uncorrected_errors):.2f} ml"
    )
    assert max(corrected_errors) <= profile.usable_capacity_ml * 0.008
    # 低液位的虚报是最要紧的：2% 那档本来差一整个残留量，校正后必须落到预算内。
    assert corrected_errors[0] < 1.0, f"2% 液位校正后仍差 {corrected_errors[0]:.2f} ml"


def test_learned_residual_pushes_the_readout_into_the_residual_limited_band():
    profile = bag_profile("TYPE-A")
    model = CavityModel(spec_for(profile))
    truth = cavity_model(profile)
    learned = learn_residual(
        [
            _observation(profile, model, truth, residue_ml=value, tag=f"r{index}")
            for index, value in enumerate((7.9, 8.0, 8.1))
        ],
        usable_capacity_ml=profile.usable_capacity_ml,
    )
    drained_level = _level_for_true_volume(truth, 8.0)
    reading = model.volume(
        truth.projected_area_ratio(drained_level),
        learned_residual_ml=learned.effective_residual_ml,
    )
    assert reading.residual_limited is True
    assert reading.readable_ml == 0.0
    assert reading.residual_source == "self_calibrated"


def test_evidence_that_contradicts_itself_is_not_applied():
    profile = bag_profile("TYPE-A")
    model = CavityModel(spec_for(profile))
    truth = cavity_model(profile)
    # 三条证据差得离谱：我们对这个袋型的理解有问题，不是数据有噪声。
    samples = [
        _observation(profile, model, truth, residue_ml=2.0, tag="a"),
        _observation(profile, model, truth, residue_ml=48.0, tag="b"),
        _observation(profile, model, truth, residue_ml=7.0, tag="c"),
    ]
    learned = learn_residual(samples, usable_capacity_ml=profile.usable_capacity_ml)
    assert learned.state == "conflicting"
    assert learned.applied is False
    assert learned.effective_residual_ml == 0.0
    assert "复核" in learned.reason
    # 打架时仍报出中位数，供人判断该往哪边查。
    assert learned.residual_ml == pytest.approx(7.0, abs=1.5)


def test_replacement_that_was_not_run_out_is_not_treated_as_residue_evidence():
    """半袋被换掉（医嘱变更、病人下床）不能拿来定残留。"""
    profile = bag_profile("TYPE-A")
    model = CavityModel(spec_for(profile))
    truth = cavity_model(profile)
    samples = [
        _observation(profile, model, truth, residue_ml=7.0, tag="a"),
        _observation(profile, model, truth, residue_ml=7.0, tag="b"),
        DeviationObservation(
            bag_type_code="TYPE-A",
            last_fill_ratio=0.72,
            model_volume_ml=300.0,
            session_id="sess-early",
        ),
    ]
    learned = learn_residual(samples, usable_capacity_ml=profile.usable_capacity_ml)
    assert learned.sample_count == 2
    assert learned.skipped_count == 1
    assert learned.applied is False
    assert samples[-1].eligible is False


def test_observation_without_fill_ratio_or_bag_type_is_not_evidence():
    assert deviation_observation_from_session({"bag_type_code": "TYPE-A"}) is not None
    missing_type = deviation_observation_from_session(
        {"bag_type_code": None, "fill_ratio": 0.1, "estimated_volume_ml": 40.0}
    )
    assert missing_type is None
    partial = deviation_observation_from_session(
        {"bag_type_code": "TYPE-A", "fill_ratio": None, "estimated_volume_ml": None}
    )
    assert partial is not None
    assert partial.eligible is False


# ── 5 主路径：estimate_bag 走腔体模式时的门禁与出数 ─────────────────────────


def _cavity_calibration(profile, **kwargs) -> BagTypeCalibration:
    return BagTypeCalibration.from_samples(
        profile.bag_type_code,
        profile.capacity_ml,
        [CalibrationSample(0.10, 60.0, source="称重法")],
        dead_volume_ml=profile.dead_volume_ml,
        calibrated_by="OP-NURSE-01",
        cavity=spec_for(profile),
        **kwargs,
    )


def test_estimate_bag_uses_cavity_mode_when_the_bag_type_registered_geometry():
    profile = bag_profile("TYPE-C")
    calibration = _cavity_calibration(profile)
    rate = RateResolution(rate_ml_per_h=5.0, source="order_text")
    estimate = estimate_bag(0.55, calibration, rate)
    assert estimate.estimation_mode == "cavity"
    assert estimate.estimated_volume_ml is not None
    assert estimate.cavity_height_fraction is not None
    assert 0.0 < estimate.cavity_height_fraction < 1.0
    assert estimate.eta_minutes is not None and estimate.eta_minutes > 0


def test_cavity_mode_keeps_the_same_gates_as_band_mode():
    profile = bag_profile("TYPE-C")
    calibration = _cavity_calibration(profile)
    rate = RateResolution(rate_ml_per_h=5.0, source="order_text")

    unobserved = estimate_bag(None, calibration, rate)
    assert unobserved.estimated_volume_ml is None
    assert unobserved.suppression_reason == "no_observation"

    overfilled = estimate_bag(1.08, calibration, rate)
    assert overfilled.estimated_volume_ml is None
    assert overfilled.overfilled is True
    assert overfilled.suppression_reason == "overfilled"

    no_rate = estimate_bag(
        0.5, calibration, RateResolution(rate_ml_per_h=None, source="unavailable")
    )
    assert no_rate.estimated_volume_ml is not None, "缺滴速只压住时点，毫升数仍要给"
    assert no_rate.eta_minutes is None
    assert no_rate.suppression_reason == "eta_unavailable"

    uncalibrated = estimate_bag(0.5, None, rate)
    assert uncalibrated.estimated_volume_ml is None
    assert uncalibrated.calibration_state == "uncalibrated"


def test_band_mode_is_unchanged_when_no_geometry_is_registered():
    profile = bag_profile("TYPE-A")
    calibration = BagTypeCalibration.from_samples(
        "TYPE-A",
        profile.capacity_ml,
        [CalibrationSample(0.50, 240.0, source="称重法")],
        dead_volume_ml=profile.dead_volume_ml,
        calibrated_by="OP-NURSE-01",
    )
    estimate = estimate_bag(0.5, calibration, RateResolution(5.0, "order_text"))
    assert estimate.estimation_mode == "band"
    assert estimate.estimated_volume_ml == pytest.approx(240.0, abs=1.0)
    assert estimate.cavity_height_fraction is None


# ── 6 存储：append-only、自校准样本与样本数 ────────────────────────────────


def test_deviation_table_is_created_and_migrations_stay_idempotent(conn):
    first = apply_migrations(conn)
    second = apply_migrations(conn)
    assert "calibration_deviation" in first["tables"]
    assert second["added_columns"] == []
    # schema 3 = calibration_version_alert 与生效区间/复核列（task 27）
    # schema 5 = upload_chunk（分片续传）与 bag_session_history（重放前快照，task 32）
    # schema 6 = writeback_queue（待写回队列 / 死信 / 重放台账，task 37）
    # schema 7 = retention_archive（帧目录归档台账）与会话的帧保留状态列（task 44）
    assert current_version(conn) == SCHEMA_VERSION == 7
    for table in ("upload_chunk", "bag_session_history", "writeback_queue", "retention_archive"):
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone() is not None
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='calibration_deviation'"
    ).fetchone()
    assert rows is not None


def test_deviation_append_is_idempotent_per_session_and_counts_samples(conn):
    repo = DeviationRepository(conn)
    observation = DeviationObservation(
        bag_type_code="TYPE-A",
        last_fill_ratio=0.05,
        model_volume_ml=7.4,
        session_id="sess-1",
        bag_seq="B01",
        rack_id="R-07",
        bed_no="12",
        calibration_version=1,
        occurred_at="2026-10-22T20:15:00+08:00",
    )
    deviation_id, created = repo.append(observation)
    assert created is True and deviation_id > 0

    again_id, again_created = repo.append(observation)
    assert again_created is False
    assert again_id == deviation_id
    assert repo.count("TYPE-A") == 1, "同一次确认重放不得留下第二条样本"

    for index in range(2, 6):
        repo.append(
            DeviationObservation(
                bag_type_code="TYPE-A",
                last_fill_ratio=0.05,
                model_volume_ml=7.0 + index * 0.1,
                session_id=f"sess-{index}",
            )
        )
    assert repo.count("TYPE-A") == 5
    # 冗余的 sample_count 必须与 COUNT(*) 一致（append-only 且仓储层不提供删除）。
    stored = repo.list_samples("TYPE-A")
    assert [row["sample_count"] for row in stored] == [1, 2, 3, 4, 5]
    assert all(row["eligible"] == 1 for row in stored)
    assert all(row["ineligible_reason"] is None for row in stored)

    learned = repo.summary("TYPE-A", usable_capacity_ml=480.0)
    assert learned.applied is True
    assert learned.sample_count == 5
    assert learned.residual_ml == pytest.approx(7.4, abs=0.01)


def test_ineligible_samples_are_kept_with_a_reason(conn):
    repo = DeviationRepository(conn)
    repo.append(
        DeviationObservation(
            bag_type_code="TYPE-B",
            last_fill_ratio=0.81,
            model_volume_ml=700.0,
            session_id="sess-early",
        )
    )
    repo.append(
        DeviationObservation(
            bag_type_code="TYPE-B",
            last_fill_ratio=0.06,
            model_volume_ml=None,
            session_id="sess-nomodel",
        )
    )
    rows = {row["session_id"]: row for row in repo.list_samples("TYPE-B")}
    assert rows["sess-early"]["eligible"] == 0
    assert rows["sess-early"]["ineligible_reason"] == "not_near_empty"
    assert rows["sess-nomodel"]["ineligible_reason"] == "no_model_volume"
    learned = repo.summary("TYPE-B", usable_capacity_ml=970.0)
    assert learned.state == "no_evidence"
    assert learned.skipped_count == 2
    assert repo.count() == 2, "不可用的样本同样要留档，不能悄悄丢掉"


def test_record_session_refuses_rows_without_a_bag_type(conn):
    repo = DeviationRepository(conn)
    assert repo.record_session({"fill_ratio": 0.1, "estimated_volume_ml": 30.0}) == (0, False)
    assert repo.count() == 0
    deviation_id, created = repo.record_session(
        {
            "bag_type_code": "TYPE-A",
            "session_id": "sess-x",
            "bag_seq": "B01",
            "rack_id": "R-07",
            "bed_no": "12",
            "fill_ratio": 0.05,
            "estimated_volume_ml": 6.8,
            "calibration_version": 2,
        }
    )
    assert created is True and deviation_id > 0
    stored = repo.find_by_session("sess-x")
    assert stored["residual_ml"] == pytest.approx(6.8, abs=0.01)
    assert stored["calibration_version"] == 2


def register_cavity_via_repo(conn) -> None:
    """仓储层登记一条带腔体参数的 TYPE-A 标定（API 侧那条路由另有用例覆盖）。"""
    from infusion_monitor.store.repositories import CalibrationRepository

    profile = bag_profile("TYPE-A")
    record = BagTypeCalibration.from_samples(
        "TYPE-A",
        profile.capacity_ml,
        [CalibrationSample(0.10, 46.0, source="称重法")],
        dead_volume_ml=profile.dead_volume_ml,
        calibrated_by="OP-NURSE-01",
        cavity=spec_for(profile),
    )
    CalibrationRepository(conn).save(record)


# ── 7 端到端：登记腔体参数 → 上传 → 护士确认换液 → 样本入库 ──────────────────


CAVITY_PAYLOAD = {
    "capacity_ml": 500.0,
    "cavity_height_mm": 210.0,
    "dead_volume_ml": 20.0,
    "shape_exponent": 4.0,
    "taper": 0.20,
    "bulge": 0.055,
    "film_holdup_ml": 0.0,
    "label_zh": "500 ml 软袋（圆角方袋）",
    "note": "按院内常用袋型登记",
}


def _register_cavity(client) -> dict:
    response = client.post(
        "/api/v1/infusion/calibrations",
        json={
            "bag_type_code": "TYPE-A",
            "capacity_ml": 500.0,
            "dead_volume_ml": 20.0,
            "cavity": CAVITY_PAYLOAD,
            "calibrated_by": "OP-NURSE-01",
            "note": "首版：带腔体外形参数",
        },
    )
    assert response.status_code == 201
    return response.json()


def _upload(client, *, shift="08:00:00", rack="R-07"):
    frames = synth.build_rack_zip(synth.default_rack_scene(24))
    return client.post(
        "/api/v1/infusion/tracks",
        data={
            "rack_id": rack,
            "infusion_order_no": "ORDER-0001",
            "bed_no": "12",
            "captured_at": f"2026-10-22T{shift}+08:00",
            "slots": json.dumps(
                [
                    {
                        "bag_seq": "B01",
                        "slot_index": 0,
                        "bag_type_code": "TYPE-A",
                        "rate_text": "5ml/h",
                    },
                    {
                        "bag_seq": "B02",
                        "slot_index": 1,
                        "bag_type_code": "TYPE-B",
                        "rate_text": "5ml/h",
                    },
                ],
                ensure_ascii=False,
            ),
        },
        files={"frames": ("frames.zip", frames, "application/zip")},
    )


def test_registered_cavity_drives_the_upload_response(client):
    body = _register_cavity(client)
    assert body["estimation_mode"] == "cavity"
    assert body["cavity"]["shape_exponent"] == pytest.approx(4.0)
    assert body["cavity"]["registered_by"] == "OP-NURSE-01"

    response = _upload(client)
    assert response.status_code == 201
    bags = {bag["bag_seq"]: bag for bag in response.json()["bags"]}

    registered = bags["B01"]
    assert registered["calibration_state"] == "calibrated"
    assert registered["estimation_mode"] == "cavity"
    assert "estimated_volume_ml" in registered
    assert registered["cavity_height_fraction"] > 0
    assert registered["resident_ml"] == pytest.approx(0.0, abs=0.01)
    # 未登记腔体参数的袋型仍走原来的分段带，形状没变。
    unregistered = bags["B02"]
    assert unregistered["calibration_state"] == "uncalibrated"
    assert "estimated_volume_ml" not in unregistered

    spec = client.get("/openapi.json").json()
    assert spec["openapi"] == "3.1.0"
    assert "/api/v1/infusion/calibrations/deviations" in spec["paths"]


def _seed_near_empty_session(conn, *, session_id: str, shift: str, fill: float, ml: float):
    """造一条「已经算过体积、袋型已登记」的近空会话（与上传主路径写的是同一批列）。"""
    from infusion_monitor.store.repositories import SessionRepository

    repository = SessionRepository(conn)
    record = repository.upsert_session(
        {
            "session_id": session_id,
            "rack_id": "R-07",
            "bed_no": "12",
            "infusion_order_no": "ORDER-0001",
            "shift_key": shift,
            "bag_seq": "B01",
            "slot_index": 0,
            "bag_type_code": "TYPE-A",
            "frames_dir": f"/tmp/{session_id}",
            "frame_count": 24,
            "captured_at": f"2026-10-22T{shift}+08:00",
            "fill_ratio": fill,
            "estimated_volume_ml": ml,
            "eta_minutes": None,
            "eta_source": "order_text",
            "rate_ml_per_h": 5.0,
            "calibration_state": "calibrated",
            "calibration_version": 1,
        }
    )
    # needs_change 是 replaced 的唯一合法来源：先让会话走到「已点名」。
    repository.apply_state_transition(record["session_id"], "needs_change")
    return repository.get(record["session_id"])


def test_confirming_a_replacement_records_self_calibration_evidence(conn):
    """护士点「已换液」= 确认这袋输完了：这条事实要落成自校准样本。"""
    from infusion_monitor.identity.idempotency import confirm_replacement_once
    from infusion_monitor.store.audit_chain import AuditChain
    from infusion_monitor.store.repositories import CalibrationRepository, EventRepository, SessionRepository

    register_cavity_via_repo(conn)
    sessions = SessionRepository(conn)
    events = EventRepository(conn)
    deviations = DeviationRepository(conn)
    audit = AuditChain(conn)
    seeded = _seed_near_empty_session(
        conn, session_id="sess-confirm", shift="2026-10-22-AM", fill=0.05, ml=7.6
    )

    outcome = confirm_replacement_once(
        sessions, events, seeded["session_id"],
        bag_seq="B01", operator_id="N-7", audit=audit, request_id="req-selfcal",
    )
    assert outcome.created is True
    assert deviations.count("TYPE-A") == 1
    sample = deviations.list_samples("TYPE-A")[0]
    assert sample["session_id"] == seeded["session_id"]
    assert sample["residual_ml"] == pytest.approx(7.6, abs=0.01)
    assert sample["eligible"] == 1
    assert sample["last_fill_ratio"] == pytest.approx(0.05, abs=1e-6)

    learned = deviations.summary("TYPE-A", usable_capacity_ml=480.0)
    assert learned.applied is False
    assert learned.state == "insufficient_samples"

    # 重放同一次确认：不产生第二条样本（与出站事件的幂等口径一致）。
    replay = confirm_replacement_once(
        sessions, events, seeded["session_id"],
        bag_seq="B01", operator_id="N-7", audit=audit, request_id="req-replay",
    )
    assert replay.idempotent_hit is True
    assert deviations.count("TYPE-A") == 1
    assert CalibrationRepository(conn).count() == 1


def test_enough_confirmations_flip_the_residual_correction_on(conn):
    """三次「换液前已经很低」的确认之后，残留补偿才真正生效。"""
    from infusion_monitor.identity.idempotency import confirm_replacement_once
    from infusion_monitor.store.audit_chain import AuditChain
    from infusion_monitor.store.repositories import EventRepository, SessionRepository

    register_cavity_via_repo(conn)
    sessions, events, deviations = SessionRepository(conn), EventRepository(conn), DeviationRepository(conn)
    audit = AuditChain(conn)

    for index, (ml, tag) in enumerate(((7.0, "a"), (7.2, "b"), (6.9, "c"))):
        seeded = _seed_near_empty_session(
            conn,
            session_id=f"sess-{tag}",
            shift=f"2026-10-22-R{index}",
            fill=0.05,
            ml=ml,
        )
        confirm_replacement_once(
            sessions, events, seeded["session_id"],
            bag_seq="B01", operator_id="N-7", audit=audit, request_id=f"req-{tag}",
        )

    learned = deviations.summary("TYPE-A", usable_capacity_ml=480.0)
    assert learned.applied is True
    assert learned.state == "learned"
    assert learned.sample_count == 3
    assert learned.residual_ml == pytest.approx(7.0, abs=0.01)

    # 生效之后，上传读数真的按残差扣掉了（而不是只写了个样本不落地）。
    from infusion_monitor.calibration.estimator import RateResolution, estimate_bag
    from infusion_monitor.store.repositories import CalibrationRepository

    calibration = CalibrationRepository(conn).get_active("TYPE-A")
    assert calibration is not None and calibration.cavity is not None
    # 取常规巡视频段（30%）：5% 那种见底液位会被置信门禁挡下不出数——那是另一条
    # 门禁（tests/test_uncertainty.py 覆盖），本用例要验的是残差扣减确实生效。
    measured = 0.30
    with_correction = estimate_bag(
        measured, calibration, RateResolution(5.0, "order_text"), learned_residual=learned
    )
    without = estimate_bag(measured, calibration, RateResolution(5.0, "order_text"))
    assert with_correction.resident_ml == pytest.approx(7.0, abs=0.01)
    assert with_correction.estimated_volume_ml == pytest.approx(
        without.estimated_volume_ml - 7.0, abs=0.2
    )


def test_response_never_leaks_credentials(client):
    _register_cavity(client)
    response = _upload(client)
    text = response.text
    for secret in (
        "test-token-not-real",
        "test-password-not-real",
        "test-key-not-real",
    ):
        assert secret not in text