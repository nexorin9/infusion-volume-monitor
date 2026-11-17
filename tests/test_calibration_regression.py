"""M3 合成真值回归：面积比 → 毫升 换算链的可核对偏差预算。

**这些数据是程序合成的，不是真实院内短片。** 真实短片须经医院授权采集；本轮自动化
验收以「合成 + 公开 VOS 数据集」替代，**替身不是最终验收边界**（见
``docs/synthetic-data.md`` 与 README「数据来源与许可」）。

回归分三层，各自归因：

1. **几何层**（:func:`test_apparent_area_ratio_is_not_volume_fraction`）——量到的面积比
   与真实体积分数**不是**一个东西，且偏差随袋型与液位变化。这是标定必须走分段
   线性带、而不是 ``面积比 × 容量`` 一条直线的物理理由。
2. **分割层**（各变体用例）——几何核在反光、手持、形变、夜间暗光下的分割误差，
   逐变体给出容差。
3. **换算层**（标定回归）——按「护士称重登记样例」的实际流程拟合标定带，再在
   **未参与拟合**的液位上预测毫升，与真值比偏差。

阈值全部写成「占标签容量的百分比」——护士端看到的是毫升数，用容量百分比做预算才
对得上「这袋还剩多少」的直觉。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from infusion_monitor.calibration.estimator import RateResolution, estimate_bag
from infusion_monitor.calibration.models import BagTypeCalibration, CalibrationSample
from infusion_monitor.synth import (
    BAG_TYPE_CATALOG,
    EXAMPLE_CAPTURED_AT,
    BagPlan,
    SceneSpec,
    bag_profile,
    cavity_model,
    decode_jpeg,
    encode_jpeg,
    frame_zip,
    hold_level,
    manifest_from_spec,
    render_scene,
    scene_manifest,
    spec_from_manifest,
    write_scene,
)

from .support import synth

#: 登记标定样例的液位（护士按袋上刻度带 / 称重法取的那几格）。
CALIBRATION_LEVELS = (0.15, 0.35, 0.55, 0.75, 0.95)
#: 预测用的留出液位——**不参与拟合**，否则就是在拿训练点报成绩。
HOLDOUT_LEVELS = (0.05, 0.25, 0.45, 0.65, 0.85)

#: 换算层预算：留出液位上的毫升误差占标签容量的比例上限。
ML_ERROR_BUDGET = 0.008
#: 分割层预算：面积比绝对误差上限（固定机位基线，不含手遮挡）。
AREA_RATIO_BUDGET = 0.02


def _single_bag_scene(bag_type_code: str, level: float, **kwargs) -> SceneSpec:
    """单袋、钉死在某一液位上的静止场景（回归的基本取样单元）。"""
    return SceneSpec(
        scene_id=f"regression-{bag_type_code}-{level:.2f}",
        bags=(BagPlan("B01", 0, bag_type_code, level, level, **kwargs),),
        frame_count=1,
        width=480,
        height=640,
        seed=20261022,
        variant="fixed",
    )


def _true_volume_ml(bag_type_code: str, level: float) -> float:
    return cavity_model(bag_profile(bag_type_code)).true_volume_ml(level)


def _naive_volume_ml(bag_type_code: str, area_ratio: float) -> float:
    """朴素换算：可用容量 × 面积比。回归要证明它不够准。"""
    profile = bag_profile(bag_type_code)
    return round(min(area_ratio * profile.usable_capacity_ml, profile.usable_capacity_ml), 1)


def _fit_calibration(bag_type_code: str, *, jitter: float = 0.0, seed: int = 7) -> BagTypeCalibration:
    """按真实登记流程拟合一条标定记录：逐液位量面积比，配上称重得到的真实毫升。"""
    profile = bag_profile(bag_type_code)
    rng = np.random.default_rng(seed)
    samples = []
    for level in CALIBRATION_LEVELS:
        scene = _single_bag_scene(bag_type_code, level)
        measurement = synth.measure_scene(scene, "B01")
        assert measurement.observed, f"{bag_type_code} 在 {level} 这一格没有观测到液体"
        ratio = measurement.area_ratio + (float(rng.uniform(-jitter, jitter)) if jitter else 0.0)
        samples.append(
            CalibrationSample(
                measured_fill_ratio=max(0.0, min(1.0, ratio)),
                known_volume_ml=_true_volume_ml(bag_type_code, level),
                source="synthetic_truth",
            )
        )
    return BagTypeCalibration.from_samples(
        bag_type_code,
        profile.capacity_ml,
        samples,
        dead_volume_ml=profile.dead_volume_ml,
        calibrated_by="SYNTH-CALIBRATOR",
        calibrated_at=EXAMPLE_CAPTURED_AT,
        note="合成真值回归标定（非真实称重数据）",
    )


# ── 1. 几何层：为什么不能拿一条直线换算 ─────────────────────────────────


def test_apparent_area_ratio_is_not_volume_fraction():
    """面积比与真实体积分数之差是**系统性**的，且随袋型与液位变化。

    若两者相等，标定就不必分段；这条断言是后面所有换算精度结论的前提。
    """
    ratios = {}
    for profile in BAG_TYPE_CATALOG:
        model = cavity_model(profile)
        # 低液位处扁袋（TYPE-D）的透视+鼓胀偏差最大，硬瓶（TYPE-C）最小
        low_fill = model.projected_area_ratio(0.05) / model.volume_fraction(0.05)
        ratios[profile.bag_type_code] = low_fill
        # 观测量高估真实体积，低液位处最明显
        assert low_fill > 1.0, f"{profile.bag_type_code} 低液位的观测量没有高估"

    spread = max(ratios.values()) - min(ratios.values())
    assert spread > 0.03, f"不同袋型在同一液位上的偏差应明显不同，实测跨度仅 {spread:.4f}"
    # 扁袋软袋偏差最大、硬瓶最小——与「软袋更容易读偏」一致
    assert ratios["TYPE-D"] > ratios["TYPE-C"]


def test_naive_linear_mapping_exceeds_budget_at_low_fill():
    """朴素换算在低液位上超预算——这就是分段带存在的理由。"""
    worst = 0.0
    for profile in BAG_TYPE_CATALOG:
        for level in (0.05, 0.10, 0.15):
            ratio = cavity_model(profile).projected_area_ratio(level)
            naive = _naive_volume_ml(profile.bag_type_code, ratio)
            true_ml = _true_volume_ml(profile.bag_type_code, level)
            worst = max(worst, abs(naive - true_ml) / profile.capacity_ml)
    assert worst > ML_ERROR_BUDGET, f"朴素换算本应超预算，实测最大偏差仅 {worst:.4f}"


# ── 2. 换算层：标定回归 ─────────────────────────────────────────────────


@pytest.mark.parametrize("bag_type_code", [profile.bag_type_code for profile in BAG_TYPE_CATALOG])
def test_calibration_recovers_millilitres_on_holdout_levels(bag_type_code: str):
    """拟合 → 留出液位预测，逐袋型断言毫升误差落在预算内。"""
    profile = bag_profile(bag_type_code)
    calibration = _fit_calibration(bag_type_code)

    errors = []
    for level in HOLDOUT_LEVELS:
        scene = _single_bag_scene(bag_type_code, level)
        measurement = synth.measure_scene(scene, "B01")
        assert measurement.observed
        predicted = calibration.estimated_volume_ml(measurement.area_ratio).volume_ml
        true_ml = _true_volume_ml(bag_type_code, level)
        errors.append((level, predicted, true_ml, abs(predicted - true_ml) / profile.capacity_ml))

    worst = max(errors, key=lambda item: item[3])
    assert worst[3] <= ML_ERROR_BUDGET, (
        f"{bag_type_code} 在液位 {worst[0]} 上偏差 {worst[3] * 100:.2f}% 标签容量"
        f"（预测 {worst[1]:.1f} ml，真值 {worst[2]:.1f} ml），超预算 {ML_ERROR_BUDGET * 100:.1f}%"
    )


def test_band_fit_beats_naive_linear_mapping():
    """分段带在留出液位上必须明显优于朴素直线——否则标定没白做。"""
    band_errors: list[float] = []
    naive_errors: list[float] = []
    for profile in BAG_TYPE_CATALOG:
        code = profile.bag_type_code
        calibration = _fit_calibration(code)
        for level in HOLDOUT_LEVELS:
            measurement = synth.measure_scene(_single_bag_scene(code, level), "B01")
            true_ml = _true_volume_ml(code, level)
            band = calibration.estimated_volume_ml(measurement.area_ratio).volume_ml
            naive = _naive_volume_ml(code, measurement.area_ratio)
            band_errors.append(abs(band - true_ml) / profile.capacity_ml)
            naive_errors.append(abs(naive - true_ml) / profile.capacity_ml)

    mean_band = float(np.mean(band_errors))
    mean_naive = float(np.mean(naive_errors))
    assert mean_band <= ML_ERROR_BUDGET
    assert mean_band * 2 < mean_naive, (
        f"分段带均值偏差 {mean_band * 100:.2f}% 未显著优于朴素直线的 {mean_naive * 100:.2f}%"
    )


def test_noisy_calibration_samples_stay_within_looser_budget():
    """标定样例带读数抖动时仍要可用——真实登记是人工读数，不是精确值。"""
    for profile in BAG_TYPE_CATALOG:
        code = profile.bag_type_code
        calibration = _fit_calibration(code, jitter=0.004, seed=31)
        for level in HOLDOUT_LEVELS:
            measurement = synth.measure_scene(_single_bag_scene(code, level), "B01")
            predicted = calibration.estimated_volume_ml(measurement.area_ratio).volume_ml
            true_ml = _true_volume_ml(code, level)
            assert abs(predicted - true_ml) / profile.capacity_ml <= 0.02, (
                f"{code} 在液位 {level} 上带抖动的标定偏差过大"
            )


def test_eta_follows_from_the_recovered_volume():
    """换算准了，ETA 才准：毫升 → 分钟按医嘱速率推，向上取整不晚提醒。"""
    calibration = _fit_calibration("TYPE-A")
    rate = RateResolution(rate_ml_per_h=5.0, source="order_text")
    measurement = synth.measure_scene(_single_bag_scene("TYPE-A", 0.45), "B01")
    estimate = estimate_bag(measurement.area_ratio, calibration, rate, bag_type_code="TYPE-A")

    true_ml = _true_volume_ml("TYPE-A", 0.45)
    assert estimate.estimated_volume_ml == pytest.approx(true_ml, abs=4.0)
    assert estimate.eta_minutes == math.ceil(estimate.estimated_volume_ml / 5.0 * 60)
    assert estimate.pickup_suppressed is False


# ── 3. 边界用例 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("bag_type_code", ["TYPE-A", "TYPE-C"])
def test_two_percent_fill_is_measured_not_discarded(bag_type_code: str):
    """液面低至 2%：细窄连通域不能被当碎斑删掉。"""
    scene = _single_bag_scene(bag_type_code, 0.02)
    measurement = synth.measure_scene(scene, "B01")
    truth = render_scene(scene).truth_rows("B01")[0]

    assert measurement.observed, "2% 液位被当成看不见——真数据被当噪声抹平了"
    assert measurement.liquid_area_px > 1000, "2% 液位不该只剩几个像素"
    assert abs(measurement.area_ratio - truth["projected_area_ratio"]) <= 0.005


def test_nearly_drained_bag_reports_no_observation_not_zero():
    """袋体几乎排空：报「未观测到」，不报 0 毫升。

    0 会被护士读成「袋空了、可以拔管」，而真相是这一帧根本没看清——这是两种完全
    不同的临床含义，不能混。
    """
    measurement = synth.measure_scene(_single_bag_scene("TYPE-A", 0.005), "B01")
    assert not measurement.observed
    assert measurement.area_ratio is None

    estimate = estimate_bag(
        measurement.area_ratio, _fit_calibration("TYPE-A"), RateResolution(5.0, "order_text")
    )
    assert estimate.estimated_volume_ml is None
    assert estimate.eta_minutes is None
    assert estimate.suppression_reason == "no_observation"
    assert estimate.pickup_suppressed is True


def test_missing_calibration_record_withholds_the_millilitre_number():
    """标定记录缺失：只有面积比，毫升字段整体不出现。"""
    measurement = synth.measure_scene(_single_bag_scene("TYPE-D", 0.55), "B01")
    estimate = estimate_bag(
        measurement.area_ratio, None, RateResolution(5.0, "order_text"), bag_type_code="TYPE-D"
    )
    assert estimate.calibration_state == "uncalibrated"
    assert estimate.estimated_volume_ml is None
    assert estimate.eta_minutes is None
    assert estimate.suppression_reason == "uncalibrated"


def test_empty_drip_chamber_bias_depends_on_which_anchor_is_used():
    """空滴斗接在袋口：列先验起锚会把滴斗算进腔体，面积比被系统性压低。

    这是本轮量化出来的一条真实结论，不是构造出来的用例——主路径必须用首帧枚举出
    的袋体候选框起锚，退回竖列先验会让挂空滴斗的架一律读少 3 个百分点（约 15 ml）。
    """
    profile = bag_profile("TYPE-A")
    scenes = {
        "no_chamber": _single_bag_scene("TYPE-A", 0.60),
        "with_chamber": _single_bag_scene("TYPE-A", 0.60, drip_chamber=True),
    }

    truth_ratio = {}
    box_ratio = {}
    column_ratio = {}
    for name, scene in scenes.items():
        truth_ratio[name] = render_scene(scene).truth_rows("B01")[0]["projected_area_ratio"]
        box = synth.measure_scene(scene, "B01", anchor="bag_box")
        column = synth.measure_scene(scene, "B01", anchor="slot_column")
        assert box.observed and column.observed
        box_ratio[name], column_ratio[name] = box.area_ratio, column.area_ratio

    # 候选框起锚：滴斗落在框外，挂不挂滴斗测得的面积比一致
    assert abs(box_ratio["with_chamber"] - box_ratio["no_chamber"]) <= 0.01
    # 列先验起锚：滴斗撑大腔体，测得的面积比被系统性压低
    column_drop = column_ratio["no_chamber"] - column_ratio["with_chamber"]
    assert column_drop > 0.02, f"空滴斗对列先验起锚的影响应明显，实测仅 {column_drop:.4f}"
    ml_error = column_drop * profile.usable_capacity_ml
    assert ml_error > 10.0, f"折成毫升应超过 10 ml，实测 {ml_error:.1f} ml"


@pytest.mark.parametrize(
    "variant,expected_slack",
    [
        ("fixed", AREA_RATIO_BUDGET),
        ("handheld", AREA_RATIO_BUDGET + 0.005),
        ("highlight", AREA_RATIO_BUDGET + 0.005),
        ("deform", AREA_RATIO_BUDGET + 0.01),
        ("night", AREA_RATIO_BUDGET + 0.005),
    ],
)
def test_shooting_variant_keeps_area_ratio_within_budget(variant: str, expected_slack: float):
    """五类拍法下面积比仍在容差内——补拍建议才不必每次都发。"""
    scene = replace(
        _single_bag_scene("TYPE-B", 0.55),
        variant=variant,
    )
    measurement = synth.measure_scene(scene, "B01")
    truth = render_scene(scene).truth_rows("B01")[0]
    assert measurement.observed
    drift = abs(measurement.area_ratio - truth["projected_area_ratio"])
    assert drift <= expected_slack, f"{variant} 变体下分割偏差 {drift:.4f} 超出 {expected_slack:.4f}"


def test_hand_occlusion_hides_liquid_and_is_reported_as_such():
    """手挡住袋体：面积比掉下来是**真值**，真值清单如实标注被挡。"""
    scene = replace(
        _single_bag_scene("TYPE-A", 0.60),
        variant="hand_occlusion",
        occlude_frames=(("B01", 0, 1),),
    )
    rendered = render_scene(scene)
    truth = rendered.truth_rows("B01")[0]
    assert truth["hand_occluded"] is True

    measurement = synth.measure_scene(scene, "B01")
    assert measurement.observed
    # 手挡住的是袋体下半部分，量到的可见液体必然少于真值——差异是真实信息，不是缺陷
    assert measurement.area_ratio < truth["projected_area_ratio"] - 0.05


def test_label_patch_does_not_punch_a_false_dent_in_the_curve():
    """袋上贴的输液标签会在液面上打出小孔洞，孔洞填充要把它吸收掉。"""
    plain = synth.measure_scene(_single_bag_scene("TYPE-A", 0.70, label=False), "B01")
    labelled = synth.measure_scene(_single_bag_scene("TYPE-A", 0.70, label=True), "B01")
    assert labelled.observed and plain.observed
    assert abs(labelled.area_ratio - plain.area_ratio) <= 0.01


# ── 4. 可复现性与产物约定 ────────────────────────────────────────────────


def test_same_seed_renders_identical_frames():
    """固定随机种子 → 逐字节可复现，回归结论才能被别人重跑出来。"""
    scene = SceneSpec(
        scene_id="repro",
        bags=(
            BagPlan("B01", 0, "TYPE-A", 0.80, 0.30, drip_chamber=True),
            BagPlan("B02", 1, "TYPE-C", 0.65, 0.25, label=True),
        ),
        frame_count=6,
        variant="handheld",
        seed=4242,
    )
    first = render_scene(scene)
    second = render_scene(scene)
    assert [encode_jpeg(f) for f in first.frames] == [encode_jpeg(f) for f in second.frames]
    for bag_seq in ("B01", "B02"):
        assert all(
            np.array_equal(a, b) for a, b in zip(first.liquid_masks[bag_seq], second.liquid_masks[bag_seq])
        )


def test_scene_reproduces_from_its_own_manifest(tmp_path: Path):
    """由 manifest 还原参数再渲染，产物与原产物逐字节一致。"""
    scene = SceneSpec(
        scene_id="manifest-round-trip",
        bags=(
            BagPlan("B01", 0, "TYPE-D", 0.70, 0.35),
            BagPlan("B02", 1, "TYPE-A", 0.55, 0.20, drip_chamber=True, label=True),
        ),
        frame_count=5,
        variant="night",
        seed=99,
    )
    rendered = render_scene(scene)
    manifest = scene_manifest(rendered)
    root = write_scene(tmp_path / "a", rendered)

    restored = write_scene(tmp_path / "b", render_scene(spec_from_manifest(manifest)))
    original_zip = (root / "frames.zip").read_bytes()
    restored_zip = (restored / "frames.zip").read_bytes()
    assert hashlib.sha256(original_zip).hexdigest() == hashlib.sha256(restored_zip).hexdigest()
    assert json.loads((restored / "manifest.json").read_text(encoding="utf-8"))["truth"] == manifest["truth"]


def test_ground_truth_uses_object_per_frame_directory_layout(tmp_path: Path):
    """真值按 {scene_id}/{bag_seq}/{frame}.png 落盘，与源产品评测器读取的 GT 目录一致。"""
    scene = SceneSpec(
        scene_id="layout",
        bags=(BagPlan("B01", 0, "TYPE-A", 0.6, 0.6), BagPlan("B02", 1, "TYPE-B", 0.4, 0.4)),
        frame_count=3,
    )
    root = write_scene(tmp_path, render_scene(scene))
    for bag_seq in ("B01", "B02"):
        masks = sorted((root / "GT" / bag_seq).glob("*.png"))
        assert [p.name for p in masks] == ["00000.png", "00001.png", "00002.png"]
    assert len(list((root / "RGB").glob("frame_*.jpg"))) == 3
    assert (root / "frames.zip").exists()


def test_generated_zip_satisfies_the_ingress_contract(tmp_path: Path):
    """合成产物本身要能过入站解包，否则拿它回归等于没走产品链路。"""
    from infusion_monitor.ingest.zip_frames import extract_frames

    scene = SceneSpec(
        scene_id="ingress",
        bags=(BagPlan("B01", 0, "TYPE-A", 0.6, 0.5),),
        frame_count=12,
    )
    archive = tmp_path / "frames.zip"
    archive.write_bytes(frame_zip(render_scene(scene)))

    extracted = extract_frames(archive, tmp_path / "unpacked")
    assert extracted.names == tuple(f"frame_{i:05d}.jpg" for i in range(12))
    assert extracted.frame_count == 12
    assert all(path.exists() for path in extracted.paths)


@pytest.mark.parametrize("variant", ["fixed", "handheld", "highlight", "deform", "hand_occlusion", "night"])
def test_every_variant_renders_and_stays_inside_the_frame(variant: str):
    """六种拍法都要渲得出来，且袋体不越框（越框会被几何核当成出框丢掉）。"""
    scene = SceneSpec(
        scene_id=f"variant-{variant}",
        bags=tuple(
            BagPlan(f"B{i + 1:02d}", i, profile.bag_type_code, 0.85, 0.35)
            for i, profile in enumerate(BAG_TYPE_CATALOG[:6])
        ),
        frame_count=8,
        variant=variant,
        occlude_frames=(("B02", 2, 5),) if variant == "hand_occlusion" else (),
    )
    rendered = render_scene(scene)
    height, width = rendered.frames[0].shape[:2]
    for index in range(scene.frame_count):
        for plan in scene.bags:
            mask = rendered.cavity_masks[plan.bag_seq][index]
            ys, xs = np.nonzero(mask)
            assert xs.min() >= 0 and xs.max() < width
            assert ys.min() >= 0 and ys.max() < height
            assert np.count_nonzero(mask) > 0, f"{variant}/{plan.bag_seq} 第 {index} 帧袋体没渲出来"


def test_single_rack_holds_at_most_six_bags():
    """单架 ≤6 袋是首版钉死的边界，超了要当场报错而不是悄悄渲成 7 个。"""
    six = tuple(BagPlan(f"B{i + 1:02d}", i, "TYPE-A", 0.6, 0.4) for i in range(6))
    SceneSpec(scene_id="six", bags=six, frame_count=2)
    with pytest.raises(ValueError, match="最多 6 袋"):
        SceneSpec(scene_id="seven", bags=six + (BagPlan("B07", 6, "TYPE-A", 0.6, 0.4),), frame_count=2)


def test_fill_trajectory_walks_down_and_stays_monotone():
    """液位一路下降：真值清单里的毫升与面积比都必须非增。"""
    scene = SceneSpec(
        scene_id="drain",
        bags=(BagPlan("B01", 0, "TYPE-A", 0.90, 0.20),),
        frame_count=14,
    )
    rows = render_scene(scene).truth_rows("B01")
    volumes = [row["true_volume_ml"] for row in rows]
    ratios = [row["projected_area_ratio"] for row in rows]
    assert volumes == sorted(volumes, reverse=True)
    assert ratios == sorted(ratios, reverse=True)
    assert volumes[0] > volumes[-1]


# ── 5. 脱敏：合成数据不得夹带可被误认为真实的标识 ─────────────────────────


#: 会被当成「真实院内标识」的形态：批号、住院号、处方号、身份证、具体日期。
_BANNED_PATTERNS = (
    re.compile(r"\bB\d{8,}\b"),                       # 类批号 B20261022
    re.compile(r"\b\d{8,10}\b"),                       # 长数字：住院号 / 处方号 / 身份证
    re.compile(r"\b(19|20)\d{2}[-/年]\d{1,2}[-/月]\d{1,2}\b"),  # 具体日期
    re.compile(r"\bZ\d{6,}\b"),                        # 住院号常见前缀
)

#: 清单里唯一允许出现的长数字：随机种子（渲染参数，不是任何人的标识）。
_NON_IDENTIFYING_KEYS = ("seed",)


def _scan_for_identifiers(manifest: dict) -> str:
    """剔除渲染参数后的清单正文（脱敏扫描只看这部分）。"""
    scannable = {k: v for k, v in manifest.items() if k not in _NON_IDENTIFYING_KEYS}
    return json.dumps(scannable, ensure_ascii=False)


def test_manifest_carries_no_identifiable_identifiers():
    """清单里的一切标识用 EXAMPLE-* / <EXAMPLE_*> 占位。"""
    scene = SceneSpec(
        scene_id="anonymity",
        bags=(
            BagPlan("B01", 0, "TYPE-A", 0.6, 0.4, drip_chamber=True, label=True),
            BagPlan("B02", 1, "TYPE-D", 0.5, 0.3),
        ),
        frame_count=4,
        variant="night",
    )
    manifest = scene_manifest(render_scene(scene))
    body = _scan_for_identifiers(manifest)

    assert EXAMPLE_CAPTURED_AT in body
    for pattern in _BANNED_PATTERNS:
        found = pattern.search(body)
        assert found is None, f"清单里出现疑似真实标识：{found.group(0)!r}（{pattern.pattern}）"


def test_fixture_manifest_is_anonymised_and_reproducible():
    """仓库里随附的样例清单：占位标识 + 可按 seed 逐位复现。"""
    fixture = Path(__file__).parent / "fixtures" / "rack-synth-scene" / "manifest.json"
    manifest = json.loads(fixture.read_text(encoding="utf-8"))
    body = _scan_for_identifiers(manifest)

    assert manifest["synthetic"] is True
    assert EXAMPLE_CAPTURED_AT in body
    for pattern in _BANNED_PATTERNS:
        found = pattern.search(body)
        assert found is None, f"样例清单里出现疑似真实标识：{found.group(0)!r}"

    spec = spec_from_manifest(manifest)
    assert spec.seed == manifest["seed"]
    rendered = render_scene(spec)
    assert [row["true_volume_ml"] for row in rendered.truth_rows(spec.bags[0].bag_seq)] == [
        row["true_volume_ml"] for row in manifest["truth"][spec.bags[0].bag_seq]
    ]