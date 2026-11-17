"""M2 鲁棒性用例集与失败模式分类测试。

对应 task 20「把『拍不好』变成护士能照做的下一步」。五组断言：

1. **六类各有一条命中用例** —— 袋体出框 / 运动模糊 / 反光过曝 / 角度不正 / 暗光 /
   滴斗空管，每条都断言命中帧数、非空中文补拍建议、以及门禁语义（info 级不算拍坏）；
2. **正常拍摄不误判** —— 干净拍法零命中，且门限收窄后确实会开始命中（证明阈值在起作用，
   不是「怎么拍都不报」）；
3. **只给一条最该补的建议** —— 多类同时命中时按 ``PRIORITY`` 取 primary，其余仍在 flags 里；
4. **暗光不误报成糊** —— 暗光帧的清晰度量不可信，不该让护士去「把手机靠住」；
5. **只降级不拒绝** —— 拍坏的上传仍然 HTTP 201、曲线照给、毫升数照给（未标定时仍不出
   数字），拒绝权只留给硬校验。

画面是**合成渲染帧**（``tests/support/synth.py``，任务 8 的 ``infusion_monitor.synth``
渲染器），不是真实院内短片——真实短片需医院授权采集。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from infusion_monitor import synth as task8_synth
from infusion_monitor.core.failure_modes import (
    FAILURE_CODES,
    PRIORITY,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    FailureModeParams,
    classify_capture,
    classify_measurements,
    measure_frame,
    sample_indices,
)
from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
from infusion_monitor.core.propagate import slot_seed_prompt
from tests.support.synth import (
    EXAMPLE_FIELDS,
    RackScene,
    anchor_prompt,
    build_rack_zip,
    default_rack_scene,
    render_rack_frame,
)

SCENE = default_rack_scene(frame_count=20)
PREDICTOR = GeometricMaskPredictor()


def frames_of(**variant) -> list[np.ndarray]:
    """按拍摄变体渲染一整段短片（默认 20 帧，和默认场景一致）。"""
    return [render_rack_frame(SCENE, index, **variant) for index in range(SCENE.frame_count)]


def classify(*, slot: int = 0, params: FailureModeParams | None = None, **variant):
    """跑一遍质检：渲染 → 用该袋的槽位先验起锚 → 分类。"""
    frames = frames_of(**variant)
    return classify_capture(
        frames,
        PREDICTOR,
        slot_seed_prompt(frames[0].shape, slot, 2),
        params,
    )


def codes_of(quality) -> list[str]:
    return [flag.code for flag in quality.flags]


# ── 1. 六类失败模式各有一条命中用例 ────────────────────────────────────────


def test_clean_capture_reports_nothing():
    """干净拍法零命中——分类器不能变成「怎么拍都报警」的噪声源。"""
    quality = classify()
    assert quality.flags == ()
    assert quality.degraded is False
    assert quality.primary is None


@pytest.mark.parametrize(
    ("variant", "expected", "slot"),
    [
        pytest.param({"crop": (0, 180, SCENE.height, SCENE.width)}, "bag_out_of_frame", 0, id="out_of_frame"),
        pytest.param({"blur": 3}, "motion_blur", 0, id="motion_blur"),
        pytest.param({"highlight": True}, "glare_overexposed", 0, id="glare"),
        pytest.param({"tilt_deg": 20.0}, "bad_angle", 0, id="tilt"),
        pytest.param({"dark": 0.32}, "low_light", 0, id="low_light"),
    ],
)
def test_each_shoot_condition_is_detected_with_actionable_advice(variant, expected, slot):
    """五类拍摄问题各命中一条，且给的是**护士能照做的动作**而不是报错。"""
    quality = classify(**variant, slot=slot)
    assert expected in codes_of(quality), f"期望命中 {expected}，实得 {codes_of(quality)}"
    flag = next(item for item in quality.flags if item.code == expected)
    assert flag.severity == SEVERITY_WARNING
    assert flag.suggestion_zh.strip(), "补拍建议不能为空"
    # 建议里必须出现护士能照做的动词，而不是「无法处理 / 质量不达标」这类推诿
    assert any(word in flag.suggestion_zh for word in ("重拍", "拍", "打开", "挡", "退", "靠"))
    # detail 说清现在是什么情况，且带证据数字（不是空话）
    assert flag.detail_zh.strip()
    assert flag.hit_frames > 0
    assert flag.sampled_frames == quality.sampled_frames
    assert flag.evidence.get("threshold") is not None


def test_out_of_frame_keeps_the_other_bag_clean():
    """只切掉一只袋子时，只有那一只被判出框——不能连坐整架。"""
    cropped = classify(crop=(0, 180, SCENE.height, SCENE.width), slot=0)
    assert "bag_out_of_frame" in codes_of(cropped)
    untouched = classify(slot=1)  # 同一段画面，另一只袋子完整入镜
    assert "bag_out_of_frame" not in codes_of(untouched)


def test_empty_drip_chamber_is_info_not_degraded():
    """空滴斗是**已知的量偏差**，不是护士拍错了：info 级，不标 degraded，不建议重拍。"""
    scenes = {}
    for drip in (False, True):
        spec = task8_synth.SceneSpec(
            scene_id=f"drip-{int(drip)}",
            frame_count=16,
            bags=(
                task8_synth.BagPlan("B01", 0, "TYPE-A", 0.60, 0.40, drip_chamber=drip),
            ),
        )
        rendered = task8_synth.render_scene(spec)
        # 列先验起锚：滴斗落在框外，会被并进腔体——这正是要识别的那条偏差路径
        scenes[drip] = classify_capture(
            rendered.frames,
            PREDICTOR,
            anchor_prompt(spec, "B01", 0, style="slot_column"),
            FailureModeParams(),
        )

    assert "empty_drip_chamber" not in codes_of(scenes[False])
    flagged = scenes[True]
    assert "empty_drip_chamber" in codes_of(flagged)

    flag = next(item for item in flagged.flags if item.code == "empty_drip_chamber")
    assert flag.severity == SEVERITY_INFO
    assert flag.evidence["tail_ratio"] < flag.evidence["threshold"]
    # info 级不构成「拍坏了」：degraded 仍为 False，primary 仍是 None
    assert flagged.degraded is False
    assert flagged.primary is None
    # 文案说清「不用重拍」，别让护士白跑一趟
    assert "不用重拍" in flag.suggestion_zh


# ── 2. 门限在起作用（不是恒不命中） ────────────────────────────────────────


def test_thresholds_actually_gate_detection():
    """门限收窄到 0 之后干净拍法也会命中——证明 2 是「门限筛掉了」而不是「压根没算」。"""
    permissive = FailureModeParams(glare_ratio_threshold=0.0, sharpness_min=0.0)
    assert classify(params=permissive).flags != ()


def test_min_hit_ratio_suppresses_a_single_frame_glitch():
    """只有个别帧偶发不合格不足以让护士重拍整段（``min_hit_ratio``）。"""
    params = FailureModeParams(min_hit_ratio=1.1)  # 永远达不到
    assert "glare_overexposed" not in codes_of(classify(highlight=True, params=params))
    # 同一个拍法、默认门限下确实命中——排除「这条压根测不出来」
    assert "glare_overexposed" in codes_of(classify(highlight=True))


def test_unmeasurable_frame_is_reported_as_such_not_guessed():
    """判不出倾角/底带宽时给 None，不编一个数字出来。"""
    frame = render_rack_frame(SCENE, 0)
    blank = np.zeros(frame.shape[:2], dtype=bool)  # 没量到液体
    measure = measure_frame(frame, blank, blank)
    assert measure.tilt_deg is None
    assert measure.tail_ratio is None  # 全空掩膜 → 判不出，不是 0.0
    assert measure.border_gap_px is None
    empty = measure_frame(frame, None, None)
    assert empty.tilt_deg is None
    assert empty.border_gap_px is None


def test_sample_indices_is_constant_cost_for_long_clips():
    """采样帧数与序列长度无关——1200 帧和 24 帧的质检开销相同。"""
    assert len(sample_indices(1200, 5)) == 5
    assert len(sample_indices(24, 5)) == 5
    assert sample_indices(3, 5) == (0, 1, 2)
    assert sample_indices(0, 5) == ()


# ── 3. 只提示最该补的那一条 ────────────────────────────────────────────────


def test_primary_follows_priority_and_only_one_suggestion_is_promoted():
    """多类同时命中时按 PRIORITY 取 primary——护士端把这一条顶到提示条就够。"""
    # 歪 + 糊 + 暗，三条一起上：出框 > 糊 > 反光 > 歪 > 暗光
    quality = classify(tilt_deg=20.0, blur=5, dark=0.32)
    assert quality.degraded is True

    order = [PRIORITY[code] for code in codes_of(quality)]
    assert order == sorted(order), "flags 必须按优先级排序"

    # 暗光把模糊判据带走了（见下一节），所以这里应剩歪 + 暗
    assert codes_of(quality) == ["bad_angle", "low_light"]
    assert len([f for f in quality.flags if f.severity == SEVERITY_WARNING]) == 2
    assert quality.primary.code == "bad_angle"  # 歪比暗先修：先把手机摆正


def test_info_flags_never_become_primary():
    """info 级（空滴斗）即使在首位也不占 primary——护士端提示条只放值得重拍的那条。"""
    params = FailureModeParams()
    measures = [measure_frame(render_rack_frame(SCENE, i), None, None, params) for i in range(4)]
    quality = classify_measurements(measures, params)
    assert all(flag.severity == SEVERITY_INFO for flag in quality.flags) or quality.flags == ()
    assert quality.primary is None or quality.primary.severity == SEVERITY_WARNING


# ── 4. 暗光不该被说成「糊」 ────────────────────────────────────────────────


def test_dark_frames_are_not_reported_as_motion_blur():
    """暗光帧的清晰度不可测：判成「糊」会让护士去靠手机，而真正要做的第一件事是开灯。

    实测：夜间短片平均灰度 27、清晰度 107（低于门限 150），两者只靠清晰度分不开。
    """
    quality = classify(dark=0.32)
    assert "low_light" in codes_of(quality)
    assert "motion_blur" not in codes_of(quality)
    assert quality.primary.code == "low_light"
    assert "灯" in quality.primary.suggestion_zh

    # 灯打开之后同样的抖动仍会被判成糊——不是把运动模糊整类关掉了
    lit_blurry = classify(blur=5)
    assert "motion_blur" in codes_of(lit_blurry)


# ── 5. 只降级不拒绝 ────────────────────────────────────────────────────────


def _upload(client, **variant):
    """把一段（拍坏的）短片真传上去，返回响应 JSON。"""
    scene = default_rack_scene(frame_count=20)
    payload = build_rack_zip(scene, **variant)
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


def test_degraded_capture_is_accepted_and_still_returns_readings(client):
    """拍坏了照样收：HTTP 201、曲线照给、体积照给，只是多一条补拍建议。"""
    body = _upload(client, blur=5)
    bag = body["bags"][0]

    assert bag["capture_degraded"] is True
    assert bag["capture_quality"], "拍坏了却没给任何提示"
    flag = next(item for item in bag["capture_quality"] if item["code"] == "motion_blur")
    assert flag["suggestion_zh"].strip()
    assert flag["severity"] == SEVERITY_WARNING

    # 降级**不**抹掉读数：曲线在、填充比在
    assert len(bag["area_ratio_series"]) == 20
    assert bag["fill_ratio"] is not None

    # 未登记标定时毫升数仍按老规矩整体不出现——拍得糊不等于查不出数，两条门禁互不吞并
    assert "estimated_volume_ml" not in bag


def test_clean_upload_reports_capture_quality_empty(client):
    """干净上传也带这两个字段（契约稳定），只是为空且未降级。"""
    body = _upload(client)
    for bag in body["bags"]:
        assert bag["capture_degraded"] is False
        assert bag["capture_quality"] == []


def test_capture_quality_survives_rereading_the_session(store_client):
    """质检结论落库：会话回读拿得到，不只是响应里的一次性字段。"""
    client, connection = store_client
    body = _upload(client, blur=5)
    session_id = body["bags"][0]["session_id"]

    row = connection.execute(
        "SELECT capture_degraded, quality_flags FROM bag_session WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    assert row is not None
    assert row["capture_degraded"] == 1
    import json

    stored = json.loads(row["quality_flags"])
    assert any(item["code"] == "motion_blur" for item in stored)
    assert all(item["suggestion_zh"].strip() for item in stored)


# ── 6. 契约与自检 ──────────────────────────────────────────────────────────


def test_every_failure_code_has_a_priority_and_a_text_entry():
    """六类都在优先级表里，且文案不重名——新增一类时漏改任一处都会被这里挡住。

    ``_TEXT`` 自 task 34 起是**两层质检共用的文案真源**（袋级分类 + 上传前预检），
    因此允许比 ``FAILURE_CODES`` 多出预检独有的条目；但袋级的每一类都必须在里面，
    多出来的那些必须正好是预检会发的那几类——多一条少一条都会在这里挡住。
    """
    from infusion_monitor.core.failure_modes import _TEXT
    from infusion_monitor.ingest.quality import PRECHECK_CODES

    assert set(PRIORITY) == set(FAILURE_CODES)
    assert set(FAILURE_CODES) <= set(_TEXT)
    assert set(_TEXT) - set(FAILURE_CODES) == set(PRECHECK_CODES) - set(FAILURE_CODES)
    assert sorted(PRIORITY.values()) == list(range(len(FAILURE_CODES)))
    for code, (severity, label, detail, suggestion) in _TEXT.items():
        assert severity in (SEVERITY_INFO, SEVERITY_WARNING)
        assert label.strip() and detail.strip() and suggestion.strip(), code


def test_quality_params_reach_the_algorithm_from_config(credential_env):
    """配置面是唯一入口：环境变量改了，分类门限真的跟着变。"""
    import os

    from infusion_monitor.config import Settings, reset_settings_cache

    os.environ["QUALITY_TILT_ANGLE_DEG"] = "3.0"
    reset_settings_cache()
    try:
        params = Settings().quality_params
        assert params.tilt_angle_deg == 3.0
        assert params.sharpness_min == FailureModeParams().sharpness_min  # 未改的保持默认
        # 同一段歪 5° 的短片：默认门限 8° 不报（拍歪了但还能用），配置收到 3° 之后报
        assert "bad_angle" not in codes_of(classify(tilt_deg=5.0))
        assert "bad_angle" in codes_of(classify(tilt_deg=5.0, params=params))
    finally:
        os.environ.pop("QUALITY_TILT_ANGLE_DEG", None)
        reset_settings_cache()


def test_openapi_documents_the_capture_quality_contract(client):
    """两个新字段进 OpenAPI——护士端据此把补拍建议顶到提示条。"""
    body = client.get("/openapi.json").json()
    assert body["openapi"].startswith("3.1")
    properties = body["components"]["schemas"]["BagSlotOut"]["properties"]
    assert {"capture_degraded", "capture_quality"} <= set(properties)
    assert "QualityFlagOut" in body["components"]["schemas"]


def test_capture_quality_cost_does_not_grow_with_clip_length():
    """质检只采样固定几帧：长片质检耗时与短片同量级（不随序列线性膨胀）。"""

    def elapsed(frame_count: int) -> float:
        scene: RackScene = default_rack_scene(frame_count=frame_count)
        frames = [render_rack_frame(scene, i) for i in range(frame_count)]
        start = cv2.getTickCount()
        classify_capture(frames, PREDICTOR, slot_seed_prompt(frames[0].shape, 0, 2))
        return (cv2.getTickCount() - start) / cv2.getTickFrequency()

    short = elapsed(20)
    long = elapsed(120)
    assert long <= max(short * 3.0, 0.5), f"短片 {short:.3f}s / 长片 {long:.3f}s，超出恒定算力预期"