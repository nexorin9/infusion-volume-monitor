"""体积不确定度与置信门禁（task 28）。

覆盖三件事：

1. **三类低置信场景各自不出数**——小液位、多次遮挡、几何版本缺失。三者的处置不同，
   所以门禁成因码与护士端文案也必须不同（补拍 / 等挡开 / 补登记版本号）。
2. **正常场景出数，且区间随条件收紧**——遮挡占比、序列抖动、几何版本齐全度各自
   收紧时区间确实收窄，「区间是个装饰」这种实现要被抓出来。
3. **门禁语义进接口契约**——被门禁挡下时 ``estimated_volume_ml`` 在响应里**整体不出现**
   （不是 null 占位），而 ``volume_uncertainty`` 必须在场，否则护士端无从解释。

取值不是拍脑袋来的：``test_low_confidence_gate_threshold_is_derived_not_chosen``
把主路径实测（见 progress.txt 的量测表）钉在断言里——门限落点由 dV/dA ÷ V 决定。
"""

from __future__ import annotations

import json
import math

import pytest

from infusion_monitor.calibration.cavity import CavitySpec
from infusion_monitor.calibration.estimator import (
    SUPPRESSION_REASONS,
    ObservationQuality,
    RateResolution,
    UncertaintyParams,
    band_sensitivity,
    estimate_bag,
    evaluate_volume_uncertainty,
)
from infusion_monitor.calibration.models import BagTypeCalibration, CalibrationSample

CAPACITY_ML = 600.0
DEAD_VOLUME_ML = 20.0
USABLE_ML = CAPACITY_ML - DEAD_VOLUME_ML


def band_calibration(**kwargs) -> BagTypeCalibration:
    """四条样例拟合出的分段带（容量 600、残留 20 → 有效 580）。"""
    return BagTypeCalibration.from_samples(
        "TYPE-A",
        CAPACITY_ML,
        [
            CalibrationSample(0.10, 58.0, source="称重法"),
            CalibrationSample(0.45, 280.0, source="称重法"),
            CalibrationSample(0.75, 440.0, source="称重法"),
            CalibrationSample(0.95, 540.0, source="称重法"),
        ],
        dead_volume_ml=DEAD_VOLUME_ML,
        calibrated_by="OP-NURSE-28",
        **kwargs,
    )


def cavity_calibration(*, geometry_revision: int = 1) -> BagTypeCalibration:
    spec = CavitySpec(
        bag_type_code="TYPE-C",
        capacity_ml=500.0,
        dead_volume_ml=15.0,
        label_zh="软袋 500",
        registered_by="OP-NURSE-28",
        geometry_revision=geometry_revision,
    )
    return BagTypeCalibration.from_samples(
        "TYPE-C",
        500.0,
        [CalibrationSample(0.50, 240.0, source="称重法")],
        dead_volume_ml=15.0,
        calibrated_by="OP-NURSE-28",
        cavity=spec,
    )


def observation(
    fill: float, *, frames: int = 24, occluded: int = 0, jitter: float = 0.0
) -> ObservationQuality:
    """一段「液位从 fill 微降到 fill-0.01」的观测；``jitter`` 是叠在降幅之上的抖动。"""
    samples = []
    for index in range(frames):
        base = fill - 0.01 * index / max(frames - 1, 1)
        wobble = jitter * math.sin(index * 1.7)
        samples.append(max(base + wobble, 1e-4))
    return ObservationQuality(
        area_ratio_samples=tuple(samples),
        frame_count=frames,
        occluded_frame_count=occluded,
    )


RATE = RateResolution(rate_ml_per_h=40.0, source="order_text")


# ── 1 三类低置信场景：各自不出数，成因与文案各不相同 ────────────────────────


def test_small_fill_withholds_the_number_because_the_interval_swallows_it():
    """小液位：同样的分割噪声落到见底的小数上，相对不确定度就大到没法照着换袋。"""
    record = band_calibration()
    obs = observation(0.06)
    estimate = estimate_bag(0.06, record, RATE, observation=obs)

    assert estimate.estimated_volume_ml is None, "低置信不该出毫升数"
    assert estimate.eta_minutes is None, "毫升不可信时，时点同样不能给"
    assert estimate.pickup_suppressed is True
    assert estimate.suppression_reason == "low_confidence"
    assert estimate.calibration_state == "low_confidence"
    assert estimate.uncertainty is not None, "门禁不通过也必须回传区间与成因"
    assert estimate.uncertainty["reason"] == "relative_uncertainty_too_high"
    assert estimate.uncertainty["confidence"] < 0.60
    # 区间仍然给得出「大概还有多少」，只是宽到不能当准数用。
    assert estimate.uncertainty["low_ml"] < estimate.uncertainty["high_ml"]


def test_repeated_occlusion_withholds_the_number_under_its_own_reason():
    """多次遮挡：占比过高单独成门禁，与「相对不确定度过大」分开报，护士才知道该重拍。"""
    record = band_calibration()
    obs = observation(0.60, frames=20, occluded=9)
    estimate = estimate_bag(0.60, record, RATE, observation=obs)

    assert estimate.estimated_volume_ml is None
    assert estimate.eta_minutes is None
    assert estimate.suppression_reason == "high_occlusion"
    assert estimate.calibration_state == "high_occlusion"
    assert estimate.uncertainty["reason"] == "occluded_too_much"
    assert estimate.uncertainty["factors"]["occlusion_ratio"] == pytest.approx(0.45)
    assert "挡" in estimate.uncertainty["hint_zh"]


def test_missing_geometry_revision_withholds_the_number_until_the_version_is_registered():
    """几何版本缺失：参数改过没有版本号，谁也说不清历史读数还成不成立。"""
    legacy = cavity_calibration(geometry_revision=0)
    assert legacy.cavity is not None and legacy.cavity.geometry_revision == 0

    estimate = estimate_bag(
        0.55,
        legacy,
        RATE,
        observation=observation(0.55),
    )
    assert estimate.estimated_volume_ml is None
    assert estimate.suppression_reason == "low_confidence"
    assert estimate.uncertainty["reason"] == "geometry_version_missing"
    assert estimate.uncertainty["factors"]["geometry_revision"] == 0
    assert "版本号" in estimate.uncertainty["hint_zh"]

    # 补登记版本号（同一份外形参数）后，同一段观测自己就出数了——门禁是可恢复的。
    fixed = cavity_calibration(geometry_revision=1)
    recovered = estimate_bag(0.55, fixed, RATE, observation=observation(0.55))
    assert recovered.estimated_volume_ml is not None
    assert recovered.uncertainty["gate"] == "ok"


def test_band_mode_is_not_gated_by_a_geometry_version_it_never_had():
    """分段带模式没有几何参数，用拟合残差当模型项；不能因为「没有几何版本号」被挡。"""
    estimate = estimate_bag(0.60, band_calibration(), RATE, observation=observation(0.60))
    assert estimate.estimation_mode == "band"
    assert estimate.estimated_volume_ml is not None
    assert estimate.uncertainty["gate"] == "ok"
    assert estimate.uncertainty["factors"]["model_source"] == "band_fit_residual"


def test_the_three_reasons_are_distinguishable_from_each_other():
    """三个场景给出三个不同的成因码与三段不同的提示，护士端不能收到同一句废话。"""
    record = band_calibration()
    low_fill = estimate_bag(0.06, record, RATE, observation=observation(0.06))
    occluded = estimate_bag(
        0.60, record, RATE, observation=observation(0.60, frames=20, occluded=9)
    )
    unversioned = estimate_bag(
        0.55, cavity_calibration(geometry_revision=0), RATE, observation=observation(0.55)
    )

    reasons = {
        low_fill.uncertainty["reason"],
        occluded.uncertainty["reason"],
        unversioned.uncertainty["reason"],
    }
    assert len(reasons) == 3, "成因码必须互不相同，否则护士端只能收到一句通用废话"
    hints = {
        low_fill.uncertainty["hint_zh"],
        occluded.uncertainty["hint_zh"],
        unversioned.uncertainty["hint_zh"],
    }
    assert len(hints) == 3
    assert all(hints)
    # 两类抑制原因码都在既有的白名单里（响应契约与数据库枚举认它们）。
    assert "low_confidence" in SUPPRESSION_REASONS
    assert "high_occlusion" in SUPPRESSION_REASONS


# ── 2 正常场景出数，区间随条件收紧 ──────────────────────────────────────────


def test_normal_conditions_produce_a_number_and_a_usable_interval():
    record = band_calibration()
    estimate = estimate_bag(0.60, record, RATE, observation=observation(0.60))

    # 0.60 落在 [0.45, 0.75] 那一段里：280 + (0.60-0.45)/(0.75-0.45) × (440-280) = 360
    assert estimate.estimated_volume_ml == pytest.approx(360.0, abs=1.0)
    assert estimate.uncertainty["gate"] == "ok"
    assert estimate.uncertainty["confidence"] >= 0.60
    low = estimate.uncertainty["low_ml"]
    high = estimate.uncertainty["high_ml"]
    assert low < estimate.estimated_volume_ml < high
    # 95% 区间：半宽 = 覆盖系数 × σ，且 σ = 相对不确定度 × 毫升数。
    # 区间上下界在 as_dict 里四舍五入到 0.1 ml，比较时留一点舍入余量。
    half = (high - low) / 2
    assert half == pytest.approx(
        1.96 * estimate.uncertainty["relative_sigma"] * estimate.estimated_volume_ml,
        abs=0.06,
    )


def test_interval_tightens_as_occlusion_ratio_falls():
    """遮挡占比下降 → 区间收窄。这是「区间随条件收紧」里最直接的一条。"""
    record = band_calibration()
    widths = []
    for occluded in (0, 2, 5):
        estimate = estimate_bag(
            0.60, record, RATE, observation=observation(0.60, frames=20, occluded=occluded)
        )
        assert estimate.estimated_volume_ml is not None, "未过门限时照常出数"
        widths.append(estimate.uncertainty["high_ml"] - estimate.uncertainty["low_ml"])
    assert widths[0] < widths[1] < widths[2], f"区间没随遮挡占比收紧：{widths}"


def test_interval_tightens_as_the_series_stops_jittering():
    """序列抖动下降 → 去趋势残差变小 → 区间收窄。抖动大到一定程度会被门禁挡下。"""
    record = band_calibration()
    widths = []
    for jitter in (0.0002, 0.006, 0.015):
        estimate = estimate_bag(0.60, record, RATE, observation=observation(0.60, jitter=jitter))
        assert estimate.estimated_volume_ml is not None, "未过门限时照常出数"
        widths.append(estimate.uncertainty["high_ml"] - estimate.uncertainty["low_ml"])
    assert widths[0] < widths[1] < widths[2], f"区间没随抖动收紧：{widths}"

    # 抖动叠加低液位：同样的抖动在中液位还放得住，到见底的小数上就该拒绝出数。
    # 门禁不是「抖动阈值」，是抖动除以余量。
    assert (
        estimate_bag(
            0.20, record, RATE, observation=observation(0.20, jitter=0.05)
        ).estimated_volume_ml
        is None
    )


def test_interval_tightens_with_liquid_level_at_the_same_absolute_noise():
    """同一个绝对噪声，在满袋上只有百分之几，在见底的小数上就是几成。"""
    record = band_calibration()
    high = estimate_bag(0.80, record, RATE, observation=observation(0.80))
    low = estimate_bag(0.25, record, RATE, observation=observation(0.25))

    assert high.estimated_volume_ml is not None and low.estimated_volume_ml is not None
    assert (
        high.uncertainty["relative_sigma"] < low.uncertainty["relative_sigma"]
    ), "同样的 σ_fill，高液位的相对不确定度必须更小"


def test_low_confidence_gate_threshold_is_derived_not_chosen():
    """门限落点由 dV/dA ÷ V 算出来，不是拍上去的常量。

    门限等价于 ``相对不确定度 > 0.35 × (1 - 0.60) = 0.14``；对本袋型即
    ``灵敏度 × σ_fill ÷ 毫升数 > 0.14``，在首段（灵敏度 580、σ_fill 0.010）下解出
    **余量 ≈ 41 ml** 以下才挡——所以放行的是 20%（85 ml）、挡下的是 5%（29 ml）。
    改动灵敏度公式或地板值时这条会先炸：说明确实有人动了取值依据，而不是顺手把
    断言改成「怎么都能过」。
    """
    record = band_calibration()
    # 0.20 落在 [0.10, 0.45] 段：斜率 = (280-58)/(0.45-0.10)
    assert band_sensitivity(record.bands, 0.20) == pytest.approx(
        (280.0 - 58.0) / 0.35, rel=1e-6
    )
    # 0.60 落在 [0.45, 0.75] 段：斜率 = (440-280)/(0.75-0.45)
    assert band_sensitivity(record.bands, 0.60) == pytest.approx(
        (440.0 - 280.0) / 0.30, rel=1e-6
    )

    allowed = estimate_bag(0.20, record, RATE, observation=observation(0.20))
    blocked = estimate_bag(0.05, record, RATE, observation=observation(0.05))

    assert allowed.estimated_volume_ml is not None
    assert allowed.uncertainty["confidence"] >= 0.60
    assert blocked.estimated_volume_ml is None
    assert (
        allowed.uncertainty["relative_sigma"] < blocked.uncertainty["relative_sigma"]
    )
    # 交叉点确实落在「余量约 41 ml」附近，不是随便挑的两个点：
    # 放行袋 84.6 ml（相对 σ 0.0522）在线下，挡下袋 29 ml（相对 σ 0.1667）在线上，
    # 两者之间就是门限解出来的余量分界。
    assert allowed.estimated_volume_ml > 41.4 > 29.0
    assert allowed.uncertainty["relative_sigma"] < 0.14
    assert blocked.uncertainty["relative_sigma"] > 0.14


# ── 3 合成逻辑本身的边界 ────────────────────────────────────────────────────


def test_drain_is_not_mistaken_for_noise():
    """整段标准差量到的是「漏了多少」，不是「量得准不准」——去趋势之后才是抖动。

    没有这一步的实现会把每一个正常读数都判成不可信（实测：80% 液位 24 帧的整段
    标准差 0.059，把它当噪声的话任何袋都出不了数）。
    """
    flat = estimate_bag(
        0.60,
        band_calibration(),
        RATE,
        observation=ObservationQuality(
            area_ratio_samples=(0.60, 0.55, 0.50, 0.45, 0.40, 0.35), frame_count=6
        ),
    )
    jittered = estimate_bag(
        0.60,
        band_calibration(),
        RATE,
        observation=ObservationQuality(
            area_ratio_samples=(0.60, 0.58, 0.53, 0.50, 0.42, 0.35), frame_count=6
        ),
    )
    # 完美线性下降：残差 0，σ 只剩地板值；第二个有肉眼可见的抖动，σ 更大。
    assert (
        flat.uncertainty["factors"]["residual_std_after_drain_fit"] < 1e-9
    )
    assert jittered.uncertainty["factors"]["residual_std_after_drain_fit"] > 0.02
    assert (
        flat.uncertainty["relative_sigma"] < jittered.uncertainty["relative_sigma"]
    )


def test_two_samples_are_not_enough_to_claim_the_series_is_steady():
    """两个点恰好拟合出一条直线，残差恒为 0——不能拿它当「这一段完美稳定」的证据。"""
    verdict = evaluate_volume_uncertainty(
        volume_ml=300.0,
        fill_ratio=0.5,
        calibration=band_calibration(),
        observation=ObservationQuality(
            area_ratio_samples=(0.50, 0.48), frame_count=2
        ),
    )
    params = UncertaintyParams()
    assert verdict.factors["residual_std_after_drain_fit"] == 0.0
    assert verdict.factors["sigma_area_ratio"] == pytest.approx(
        params.sigma_fill_floor, rel=1e-6
    )


def test_band_sensitivity_refuses_a_degenerate_band():
    """带坏掉时给 None（退回地板值单独算），不给一个错的斜率。"""
    assert band_sensitivity([], 0.5) is None
    assert band_sensitivity([(0.0, 0.0)], 0.5) is None
    assert band_sensitivity([(0.0, 0.0), (0.0, 100.0)], 0.5) is None


def test_occlusion_ratio_is_clamped_and_safe_without_frames():
    assert ObservationQuality().occlusion_ratio == 0.0
    assert ObservationQuality(occluded_frame_count=5, frame_count=0).occlusion_ratio == 0.0
    assert (
        ObservationQuality(occluded_frame_count=99, frame_count=10).occlusion_ratio == 1.0
    )


def test_uncertainty_parameters_reject_nonsense_thresholds():
    """门限本身也要能被钉住：把置信度门限抬到 1.0 以上等于把门禁永久关上。"""
    from pydantic import ValidationError

    from infusion_monitor.config import Settings

    base = {
        "nurse_exec_base_url": "https://nurse-exec.example.invalid",
        "nurse_exec_api_token": "t",
        "mqtt_host": "b",
        "mqtt_port": 1883,
        "mqtt_username": "u",
        "mqtt_password": "p",
        "llm_base_url": "https://llm.example.invalid/v1",
        "llm_api_key": "k",
        "llm_model": "m",
        "llm_timeout": 5,
    }
    assert Settings(**base).uncertainty_params.min_confidence == 0.60
    for field in (
        "uncertainty_min_confidence",
        "uncertainty_high_occlusion_ratio",
        "uncertainty_sigma_fill_floor",
        "uncertainty_sigma_ceiling",
    ):
        with pytest.raises(ValidationError):
            Settings(**base, **{field: -1.0})


# ── 4 门禁语义进接口契约与存储 ──────────────────────────────────────────────


def _register_band(client, *, bag_type_code: str = "TYPE-A") -> dict:
    response = client.post(
        "/api/v1/infusion/calibrations",
        json={
            "bag_type_code": bag_type_code,
            "capacity_ml": CAPACITY_ML,
            "dead_volume_ml": DEAD_VOLUME_ML,
            "calibrated_by": "OP-NURSE-28",
            "samples": [
                {"measured_fill_ratio": 0.10, "known_volume_ml": 58.0, "source": "称重法"},
                {"measured_fill_ratio": 0.45, "known_volume_ml": 280.0, "source": "称重法"},
                {"measured_fill_ratio": 0.75, "known_volume_ml": 440.0, "source": "称重法"},
                {"measured_fill_ratio": 0.95, "known_volume_ml": 540.0, "source": "称重法"},
            ],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _upload(client, *, fill: float = 0.60, rack: str = "R-EXAMPLE-28"):
    payload = {
        "rack_id": rack,
        "infusion_order_no": "EXAMPLE-ORDER-0028",
        "bed_no": "B-EXAMPLE-28",
        "captured_at": "2026-01-01T09:41:00+08:00",
        "slots": json.dumps(
            [
                {
                    "bag_seq": "B01",
                    "slot_index": 0,
                    "bag_type_code": "TYPE-A",
                    "rate_text": "40ml/h",
                }
            ]
        ),
    }
    return client.post(
        "/api/v1/infusion/tracks",
        data=payload,
        files={"frames": ("frames.zip", _sample_zip(fill), "application/zip")},
    )


def test_upload_at_a_normal_level_returns_the_number_with_its_interval(client):
    _register_band(client)
    response = _upload(client, fill=0.60)
    assert response.status_code == 201, response.text
    bag = response.json()["bags"][0]

    assert bag["estimated_volume_ml"] is not None
    interval = bag["volume_uncertainty"]
    assert interval["gate"] == "ok"
    assert interval["confidence"] >= 0.60
    assert interval["low_ml"] < bag["estimated_volume_ml"] < interval["high_ml"]
    assert interval["reason"] == ""


def test_gated_upload_drops_the_field_entirely_but_keeps_the_reason(client):
    """被门禁挡下：``estimated_volume_ml`` **字段整体不出现**（不是 null），
    而 ``volume_uncertainty`` 必须在场——否则护士端只能看到「没数」，无从解释。"""
    _register_band(client)
    response = _upload(client, fill=0.04, rack="R-EXAMPLE-28B")
    assert response.status_code == 201, response.text
    bag = response.json()["bags"][0]

    assert "estimated_volume_ml" not in bag, "门禁不通过时字段不得出现"
    assert "eta_minutes" not in bag
    assert bag["pickup_suppressed"] is True
    assert bag["suppression_reason"] in ("low_confidence", "high_occlusion", "no_observation")
    interval = bag["volume_uncertainty"]
    assert interval is not None
    assert interval["gate"] != "ok"
    assert interval["hint_zh"]

    # 区间仍在：护士端据此显示「大概还剩多少、为什么没有准数」。
    assert interval["low_ml"] <= interval["high_ml"]


def test_gated_result_is_persisted_so_the_next_round_can_be_compared(store_client):
    _, observer = store_client
    client = store_client[0]
    _register_band(client)
    response = _upload(client, fill=0.04, rack="R-EXAMPLE-28C")
    assert response.status_code == 201
    session_id = response.json()["bags"][0]["session_id"]
    row = observer.execute(
        "SELECT estimated_volume_ml, eta_minutes, calibration_state, suppression_reason,"
        " volume_confidence, volume_low_ml, volume_high_ml, uncertainty_gate,"
        " uncertainty_json FROM bag_session WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    assert row["estimated_volume_ml"] is None, "门禁不通过：库里明确没有数"
    assert row["eta_minutes"] is None
    assert row["suppression_reason"] in ("low_confidence", "high_occlusion", "no_observation")
    assert row["uncertainty_gate"] in ("low_confidence", "high_occlusion", "ok")
    evidence = json.loads(row["uncertainty_json"])
    assert evidence["reason"] in (
        "relative_uncertainty_too_high",
        "occluded_too_much",
        "geometry_version_missing",
    )


def test_uncalibrated_and_low_confidence_are_two_different_doorways(client):
    """两种「不出数」必须能被护士端分开：没定标 → 补登记；观测不够 → 重拍。"""
    # 没登记标定：uncalibrated，且没有置信区间（无从谈置信）。
    bare = _upload(client, rack="R-EXAMPLE-28D")
    assert bare.status_code == 201
    bag = bare.json()["bags"][0]
    assert "estimated_volume_ml" not in bag
    assert bag["suppression_reason"] == "uncalibrated"
    assert bag.get("volume_uncertainty") is None

    # 登记了但这段观测撑不住：low_confidence，带区间与成因。
    _register_band(client)
    gated = _upload(client, fill=0.04, rack="R-EXAMPLE-28E").json()["bags"][0]
    assert gated["suppression_reason"] != "uncalibrated"
    assert gated["volume_uncertainty"]["reason"] == "relative_uncertainty_too_high"


def test_openapi_documents_the_interval_contract(client):
    schema = client.get("/openapi.json").json()
    bag_schema = schema["components"]["schemas"]["BagSlotOut"]
    assert "volume_uncertainty" in bag_schema["properties"]
    interval = schema["components"]["schemas"]["UncertaintyOut"]
    assert set(interval["required"]) >= {
        "confidence",
        "relative_sigma",
        "low_ml",
        "high_ml",
        "gate",
    }


def _sample_zip(fill: float):
    """按给定液位渲染一段可上传的帧序列 zip（复用主路径同款渲染器）。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("scripts").resolve()))
    import tempfile

    import make_sample_track

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as handle:
        path = Path(handle.name)
    make_sample_track.build(path, 24, 640, 480, fill, bags=1)
    return path.open("rb")