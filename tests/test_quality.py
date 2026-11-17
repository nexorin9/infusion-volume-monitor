"""上传前帧质量预检测试（task 34）。

对应 task 34「拍摄质量决定后面所有数字的可信度」。六组断言：

1. **五类各有一条命中用例** —— 模糊 / 过曝 / 暗光 / 歪 / 分辨率偏低，每条都断言命中帧数、
   非空中文补拍建议、实测值与门限进证据；
2. **正常拍摄不误判** —— 干净拍法零命中、总分满分，且门限收窄后确实开始命中（证明门限
   在起作用，不是「怎么拍都不报」）；
3. **共用文案层** —— 预检的建议逐字等于 :func:`core.failure_modes.render_reshoot_text`
   的输出，且与袋级分类对同一段短片给出**同一句话**（两层措辞不能分叉）；
4. **只降级不拒绝** —— 拍坏的短片照样 HTTP 201、曲线照给，预检分低**不**吞掉读数；
5. **落库** —— 质量分与 flag 写进 ``bag_session``，护士复盘时回读得到；
6. **恒定算力 + 判不出就不猜** —— 采样帧数与短片长度无关；画面里没有可量的液面时
   倾角子项整体缺席，而不是给一个编出来的角度。

画面是**合成渲染帧**（``tests/support/synth.py``），不是真实院内短片——真实短片需医院授权采集。
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from infusion_monitor.core.failure_modes import (
    FailureModeParams,
    classify_capture,
    render_reshoot_text,
)
from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
from infusion_monitor.core.propagate import slot_seed_prompt
from infusion_monitor.ingest.quality import (
    PRECHECK_CODES,
    PrecheckParams,
    measure_frame_quality,
    precheck_frames,
    precheck_sample_indices,
)
from tests.support.synth import (
    EXAMPLE_FIELDS,
    default_rack_scene,
    render_rack_frame,
)

SCENE = default_rack_scene(frame_count=20)


def precheck(*, scene=None, params: PrecheckParams | None = None, **variant):
    """渲染一段短片 → 跑一遍上传前预检。"""
    scene = scene or SCENE
    frames = [render_rack_frame(scene, index, **variant) for index in range(scene.frame_count)]
    return precheck_frames(frames, params)


def codes_of(quality) -> set[str]:
    return {flag.code for flag in quality.flags}


# ── 1. 五类命中用例 ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "variant,expected",
    [
        pytest.param({"blur": 5}, "motion_blur", id="blur"),
        pytest.param({"highlight": True}, "glare_overexposed", id="glare"),
        pytest.param({"dark": 0.32}, "low_light", id="low_light"),
        pytest.param({"tilt_deg": 20.0}, "bad_angle", id="tilt"),
    ],
)
def test_each_problem_is_detected_with_an_actionable_suggestion(variant, expected):
    """命中的每一类都要带一句护士照做就能改好的中文建议 + 支撑证据。"""
    quality = precheck(**variant)

    assert expected in codes_of(quality), f"{expected} 没命中，实际 {codes_of(quality)}"
    flag = next(item for item in quality.flags if item.code == expected)
    assert flag.suggestion_zh.strip(), "命中了却没给补拍建议"
    assert flag.detail_zh.strip()
    assert flag.label_zh.strip()
    assert flag.severity == "warning"
    # 证据里必须有实测值与门限，信息科据此判断是「拍坏了」还是「门限太严」；
    # 且中文文案里的占位符要真被填上，不能把 ``{threshold}`` 原样发给护士。
    assert "threshold" in flag.evidence
    assert "{" not in flag.detail_zh and "{" not in flag.suggestion_zh
    assert flag.hit_frames == flag.sampled_frames == 5
    assert flag.hit_ratio == pytest.approx(1.0)


def test_low_resolution_is_flagged_but_still_accepted_by_the_hard_validator():
    """短边低于质量下限即提示；只要还过得了硬校验（≥320px）就照收。"""
    small = default_rack_scene(frame_count=20, width=400, height=520)
    quality = precheck(scene=small)

    assert "low_resolution" in codes_of(quality)
    flag = next(item for item in quality.flags if item.code == "low_resolution")
    assert flag.evidence["short_side"] == 400
    assert flag.evidence["threshold"] == 480
    assert flag.suggestion_zh.strip()

    # 分辨率子分只是打折，不是清零：320 的硬下限仍归入站校验管
    assert quality.sub_scores["resolution"] == pytest.approx(400 / 480, abs=1e-3)
    assert quality.degraded is True


def test_bags_at_different_fill_levels_do_not_fake_a_tilt():
    """**六只袋子液面高度各不相同**：把它们硬拟合成一条直线，量到的是「挂得高矮差多少」，
    不是「手机歪了多少度」。实测本项若不分组，干净拍法会读出 19–22° 的假倾角。"""
    quality = precheck()

    assert "bad_angle" not in codes_of(quality)
    assert all(frame.tilt_deg is not None for frame in quality.frames)
    assert max(frame.tilt_deg for frame in quality.frames) < 1.0


def test_tilt_is_measured_against_the_true_tilt():
    """歪 20° 的拍法量到的倾角贴着 20°（不是某个经验系数）。"""
    quality = precheck(tilt_deg=20.0)
    assert quality.frames[0].tilt_deg == pytest.approx(20.0, abs=1.5)


def test_tiny_tilt_below_the_threshold_is_not_reported():
    """歪 5° 拍得歪但还能用：默认门限 8° 不报——预检不是「越严越好」。"""
    quality = precheck(tilt_deg=5.0)

    assert "bad_angle" not in codes_of(quality)
    assert quality.degraded is False


# ── 2. 正常拍摄不误判 / 门限在起作用 ─────────────────────────────────────


def test_clean_capture_passes_with_a_full_score():
    """正常拍摄：零命中、总分满分、不降级。"""
    quality = precheck()

    assert quality.flags == ()
    assert quality.degraded is False
    assert quality.state == "ok"
    assert quality.score == pytest.approx(1.0)
    assert set(quality.sub_scores) == {"sharpness", "low_light", "glare", "resolution", "tilt"}


def test_narrowing_a_threshold_starts_reporting_clean_capture():
    """门限收紧后干净拍法确实开始命中——证明这些门限在起作用，不是「怎么拍都满分」。"""
    strict = PrecheckParams(sharpness_min=5000.0)
    quality = precheck(params=strict)

    assert "motion_blur" in codes_of(quality)
    assert quality.score < 1.0


def test_dark_frames_are_not_reported_as_blurry():
    """暗光帧的清晰度量不可信：光线不足与手抖在这一个量上分不开。

    夜里的短片清晰度 107 低于门限 150，若照直报成「运动模糊」，护士收到的第一句建议会是
    「把手机靠住」——而她真正该做的第一件事是**把灯打开**。与袋级分类同一条规则。
    """
    quality = precheck(dark=0.32)

    assert "low_light" in codes_of(quality)
    assert "motion_blur" not in codes_of(quality)
    low_light = next(item for item in quality.flags if item.code == "low_light")
    assert "把灯" in low_light.suggestion_zh or "灯" in low_light.suggestion_zh


def test_score_gate_degrades_even_when_no_problem_is_hit():
    """五类都没命中但总分低于门禁时照样降级——「每项都差一点」同样让余量读数打问号。"""
    quality = precheck(tilt_deg=5.0, params=PrecheckParams(degraded_score=0.9))

    assert quality.flags == (), "歪 5° 不到 bad_angle 门限，不该命中任何一类"
    assert quality.degraded is True
    assert quality.state == "degraded"
    assert quality.primary is None


def test_few_odd_frames_do_not_condemn_the_whole_clip():
    """只有一帧糊不算「这一段拍坏了」——不足以让护士重拍整段。"""
    frames = [render_rack_frame(SCENE, index, blur=5 if index == 3 else 0) for index in range(20)]
    measured_indices = [3, 7, 11, 15, 19]  # 只有 3 号帧是糊的

    quality = precheck_frames(
        [frames[index] for index in measured_indices], frame_indices=measured_indices
    )

    assert "motion_blur" not in codes_of(quality)
    assert quality.degraded is False


# ── 3. 共用文案层 ────────────────────────────────────────────────────────


def test_precheck_text_comes_from_the_shared_reshoot_table():
    """每条建议都逐字等于共用文案表的输出——两层质检不会对护士说两套话。"""
    cases = [
        ({"blur": 5}, "motion_blur"),
        ({"highlight": True}, "glare_overexposed"),
        ({"dark": 0.32}, "low_light"),
        ({"tilt_deg": 20.0}, "bad_angle"),
    ]
    for variant, code in cases:
        flag = next(item for item in precheck(**variant).flags if item.code == code)
        expected = render_reshoot_text(code, flag.hit_frames, flag.sampled_frames, flag.evidence)
        assert flag.suggestion_zh == expected["suggestion_zh"]
        assert flag.detail_zh == expected["detail_zh"]
        assert flag.label_zh == expected["label_zh"]
        assert flag.severity == expected["severity"]


def test_bag_level_classifier_and_precheck_wording_never_diverge():
    """同一段糊片，袋级分类与上传前预检给护士的是**同一句话**。"""
    variant = {"blur": 5}
    precheck_flag = next(
        item for item in precheck(**variant).flags if item.code == "motion_blur"
    )

    frames = [render_rack_frame(SCENE, index, **variant) for index in range(SCENE.frame_count)]
    prompt = slot_seed_prompt(frames[0].shape, 0, len(SCENE.bags))
    capture = classify_capture(frames, GeometricMaskPredictor(), prompt, FailureModeParams())
    bag_flag = next(item for item in capture.flags if item.code == "motion_blur")

    assert bag_flag.suggestion_zh == precheck_flag.suggestion_zh


def test_every_precheck_code_has_shared_text():
    """新增一类时漏写共用文案表会被这里挡住（响应里就会出现一条空建议）。"""
    for code in PRECHECK_CODES:
        text = render_reshoot_text(code, 1, 5, {})
        assert text is not None, code
        assert text["suggestion_zh"].strip(), code
        assert text["detail_zh"].strip(), code


# ── 4. 只降级不拒绝 ──────────────────────────────────────────────────────


def _upload(client, *, scene=None, **variant):
    """把一段（可能拍坏的）短片真传上去，返回响应 JSON。"""
    from tests.support.synth import build_rack_zip

    payload = build_rack_zip(scene or SCENE, **variant)
    response = client.post(
        "/api/v1/infusion/tracks",
        data={
            "rack_id": EXAMPLE_FIELDS["rack_id"],
            "infusion_order_no": EXAMPLE_FIELDS["infusion_order_no"],
            "bed_no": EXAMPLE_FIELDS["bed_no"],
            "captured_at": EXAMPLE_FIELDS["captured_at"],
            "slots": '[{"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-A"},'
            ' {"bag_seq": "B02", "slot_index": 1, "bag_type_code": "TYPE-A"}]',
        },
        files={"frames": ("frames.zip", payload, "application/zip")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_bad_capture_is_accepted_and_still_returns_readings(client):
    """拍坏了照样收：HTTP 201、曲线照给；预检分低不吞读数，也不冒充「查不出数」。"""
    body = _upload(client, blur=5)
    precheck_out = body["precheck"]
    bag = body["bags"][0]

    assert precheck_out["state"] == "degraded"
    assert precheck_out["degraded"] is True
    assert precheck_out["quality_score"] < 1.0
    assert any(flag["code"] == "motion_blur" for flag in precheck_out["flags"])
    assert precheck_out["primary"]["code"] == "motion_blur"

    # 逐袋字段与整段结论一致（护士端按袋展示时不会两处打架）
    assert bag["precheck_score"] == pytest.approx(precheck_out["quality_score"])
    assert bag["precheck_degraded"] is True

    # 降级不抹读数：曲线在、填充比在；未标定时毫升数仍按老规矩整体不出现
    assert len(bag["area_ratio_series"]) == SCENE.frame_count
    assert bag["fill_ratio"] is not None
    assert "estimated_volume_ml" not in bag


def test_clean_upload_reports_precheck_as_ok(client):
    """干净上传也带 precheck 段（契约稳定），只是 ok、零命中。"""
    body = _upload(client)

    assert body["precheck"]["state"] == "ok"
    assert body["precheck"]["degraded"] is False
    assert body["precheck"]["flags"] == []
    assert body["precheck"]["quality_score"] == pytest.approx(1.0)
    assert body["precheck"]["sampled_frames"] == 5
    # 采样帧观测量随行返回，信息科可复核门限是否过严
    assert len(body["precheck"]["frames"]) == 5
    assert all(item["short_side"] == SCENE.width for item in body["precheck"]["frames"])
    for bag in body["bags"]:
        assert bag["precheck_degraded"] is False


def test_openapi_documents_the_precheck_contract(client):
    """precheck 段进 OpenAPI——护士端据此决定要不要顶补拍提示条。"""
    spec = client.get("/openapi.json").json()
    assert spec["openapi"].startswith("3.1")
    assert "PrecheckOut" in spec["components"]["schemas"]
    properties = spec["components"]["schemas"]["PrecheckOut"]["properties"]
    assert {"quality_score", "degraded", "flags", "primary", "sub_scores"} <= set(properties)
    assert "precheck" in spec["components"]["schemas"]["TrackUploadResponse"]["properties"]


# ── 5. 落库 ──────────────────────────────────────────────────────────────


def test_precheck_result_is_persisted_per_bag(store_client):
    """质量分与 flag 落库：会话回读拿得到，不只是响应里看一次就没了。"""
    client, connection = store_client
    body = _upload(client, blur=5)

    for bag in body["bags"]:
        row = connection.execute(
            "SELECT precheck_score, precheck_flags, precheck_degraded "
            "FROM bag_session WHERE session_id = ?",
            (bag["session_id"],),
        ).fetchone()
        assert row is not None
        assert row["precheck_degraded"] == 1
        assert row["precheck_score"] == pytest.approx(bag["precheck_score"])
        stored = json.loads(row["precheck_flags"])
        assert any(item["code"] == "motion_blur" for item in stored)
        assert all(item["suggestion_zh"].strip() for item in stored)

    # 整段一份结论，逐袋各存一份：同一段短片的两只袋子读数必须一致
    scores = {
        connection.execute(
            "SELECT precheck_score FROM bag_session WHERE session_id = ?",
            (bag["session_id"],),
        ).fetchone()["precheck_score"]
        for bag in body["bags"]
    }
    assert len(scores) == 1


def test_session_endpoint_returns_the_precheck_fields(client):
    """``GET /sessions/{id}`` 同样能查到预检结论（护士复盘走的就是这条）。"""
    body = _upload(client, dark=0.32)
    record = client.get(f"/api/v1/infusion/sessions/{body['bags'][0]['session_id']}").json()

    assert record["precheck_degraded"] == 1
    assert record["precheck_score"] == pytest.approx(body["bags"][0]["precheck_score"])
    assert any(item["code"] == "low_light" for item in json.loads(record["precheck_flags"]))


# ── 6. 恒定算力 + 判不出就不猜 ───────────────────────────────────────────


def test_sampling_cost_is_independent_of_clip_length():
    """1200 帧的架和 24 帧的架预检开销相同：只量固定几帧，与序列长度无关。"""
    frame = render_rack_frame(SCENE, 0)

    assert len(precheck_sample_indices(24)) == 5
    assert len(precheck_sample_indices(1200)) == 5
    sampled = precheck_sample_indices(1200)
    assert sampled[0] == 0 and sampled[-1] == 1199  # 含首末，覆盖整段

    short = precheck_frames([frame] * 24)
    long = precheck_frames([frame] * 1200)
    assert short.sampled_frames == long.sampled_frames == 5
    assert short.score == pytest.approx(long.score)

    start = time.perf_counter()
    precheck_frames([frame] * 24)
    short_cost = time.perf_counter() - start
    start = time.perf_counter()
    precheck_frames([frame] * 1200)
    long_cost = time.perf_counter() - start
    # 帧数 ×50，单帧成本不应随之上涨（宽松上界，给 CI 抖动留余量）
    assert long_cost < max(short_cost * 3.0, 0.5)


def test_frame_without_liquid_yields_no_tilt_instead_of_a_guessed_angle():
    """画面里找不到可量的液面线时，倾角判不出就说不判——不猜一个角度，也不给 0 分。"""
    background = np.full((240, 320, 3), 32, dtype=np.uint8)
    background[10:20, 10:20] = (200, 30, 30)  # 零星高饱和噪点，量不出液面
    measured = measure_frame_quality(background, 0)

    assert measured.tilt_deg is None

    quality = precheck_frames([background] * 20)
    assert quality.frames[0].tilt_deg is None
    assert "tilt" not in quality.sub_scores, "判不出 ≠ 拍得差，不该给 0 分"
    assert "bad_angle" not in codes_of(quality)
    # 权重摊回其余子项，总分仍只按量得到的四项算
    assert set(quality.sub_scores) == {"sharpness", "low_light", "glare", "resolution"}


def test_measure_frame_rejects_a_non_rgb_frame():
    """入参形状不对就直接报错，不静默按灰度算。"""
    with pytest.raises(ValueError):
        measure_frame_quality(np.zeros((10, 10), dtype=np.uint8))


def test_precheck_params_come_from_config_and_share_the_bag_classifier_thresholds(credential_env):
    """配置面是唯一入口，且两层质检共用同一组门限——不会出现「预检说好、袋级说坏」。"""
    import os

    from infusion_monitor.config import Settings, reset_settings_cache

    settings = Settings()
    precheck_params = settings.precheck_params
    bag_params = settings.quality_params
    assert precheck_params.sharpness_min == bag_params.sharpness_min
    assert precheck_params.tilt_angle_deg == bag_params.tilt_angle_deg
    assert precheck_params.glare_ratio_threshold == bag_params.glare_ratio_threshold
    assert precheck_params.low_light_mean_luma == bag_params.low_light_mean_luma
    assert precheck_params.sample_frames == bag_params.sample_frames
    # 预检量的液面必须是主路径量的那批液体
    assert precheck_params.saturation_min == settings.model_saturation_min

    os.environ["QUALITY_SHARPNESS_MIN"] = "4000.0"
    os.environ["PRECHECK_MIN_SHORT_SIDE"] = "600"
    os.environ["PRECHECK_DEGRADED_SCORE"] = "0.9"
    reset_settings_cache()
    try:
        tuned = Settings()
        assert tuned.precheck_params.sharpness_min == 4000.0
        assert tuned.quality_params.sharpness_min == 4000.0  # 两层同时改
        assert tuned.precheck_params.min_short_side == 600
        assert tuned.precheck_params.degraded_score == 0.9
        assert tuned.describe()["precheck"]["min_short_side"] == 600
        # 门限改了，结论真的跟着变
        assert "motion_blur" in codes_of(precheck(params=tuned.precheck_params))
        assert precheck(params=tuned.precheck_params).degraded is True
    finally:
        os.environ.pop("QUALITY_SHARPNESS_MIN", None)
        os.environ.pop("PRECHECK_MIN_SHORT_SIDE", None)
        os.environ.pop("PRECHECK_DEGRADED_SCORE", None)
        reset_settings_cache()


def test_invalid_precheck_settings_are_rejected(credential_env):
    """质量分门禁落在 (0, 1] 之外、短边下限离谱的值，启动即拒而不是照单全收。"""
    from pydantic import ValidationError

    from infusion_monitor.config import Settings

    for field, value in (
        ("PRECHECK_DEGRADED_SCORE", "1.5"),
        ("PRECHECK_DEGRADED_SCORE", "0"),
        ("PRECHECK_MIN_SHORT_SIDE", "0"),
    ):
        with pytest.raises(ValidationError):
            Settings(**{field.lower(): value})


def test_empty_clip_yields_a_zero_sample_precheck():
    """没有帧可量时给空结论（0 帧、不降级），而不是假装满分。"""
    quality = precheck_frames([])

    assert quality.sampled_frames == 0
    assert quality.degraded is False
    assert quality.flags == ()
    assert quality.score == 0.0