"""M2 再获取测试：丢失轨迹在后续帧里被找回来，且绝不接错曲线。

六组断言：

1. **与遮挡恢复不打架**：还没交出（丢失窗口未用尽）的轨迹**一条也不碰**，走的是
   :mod:`.occlusion` 的恢复路径；只有 ``handoff`` 在的才进入再获取；
2. **单条再获取成功**：人为遮挡超过丢失窗口 → 置 lost → 袋子在后面几帧回到画面 →
   接回**同一个** ``bag_seq``，曲线在原位置续上（遮挡段仍留 ``None``，不插值）；
3. **批量再获取**：两条轨迹同时 lost，一次枚举服务两条，各自接回自己那条，
   **不互相串号**（一条候选只接一条轨迹）；
4. **失败保持 lost**：附近扫不到 / 全部门限不过 / 最佳与次优分差不足 → 保持 lost，
   要求人工认领，**不静默新建轨迹**；
5. **不得把新袋接到旧曲线**：换液后新袋更满，余量回升超过门限 → 直接拒绝接回；
6. **主路径接线**：``propagate_in_video`` 在真实帧序列上自动触发再获取，
   传 ``reacquire=None`` 则整条路径关闭（对照组曲线全程不复活）。

画面是**合成渲染帧**（``tests/support/synth.py``），不是真实院内短片——真实短片需
医院授权采集，本轮以合成 + 公开 VOS 子集替代。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from infusion_monitor.core.enumerate import enumerate_candidates
from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
from infusion_monitor.core.occlusion import (
    LOST,
    OCCLUDED,
    VISIBLE,
    OcclusionTracker,
    ReacquireHandoff,
    VisibilityParams,
)
from infusion_monitor.core.propagate import (
    REACQUIRE_GAP_FRAMES,
    analyse_frames,
    slot_seed_prompt,
)
from infusion_monitor.core.reacquire import (
    REASON_AMBIGUOUS,
    REASON_BATCH_LIMIT,
    REASON_NO_REFERENCE,
    REASON_NOT_LOST,
    REASON_ROI_EMPTY,
    REACQUIRED,
    STILL_LOST,
    UNBOUND,
    LostTrackView,
    ReacquireParams,
    reacquire_tracks,
    roi_for_tracks,
)
from infusion_monitor.identity.reanchor import describe_region
from tests.support.synth import (
    BagSpec,
    RackScene,
    default_rack_scene,
    render_rack_frame,
    write_rack_frames,
)

FRAME_COUNT = 24
PREDICTOR = GeometricMaskPredictor()


# ── 构造器 ──────────────────────────────────────────────────────────────
def _handoff(
    obj_id: str = "B01",
    *,
    bbox: tuple[int, int, int, int] | None,
    cavity_area_px: int,
    reference_area_ratio: float | None,
    lost_frame: int = 8,
    last_seen_frame: int = 5,
    missing_frames: int = 3,
) -> ReacquireHandoff:
    return ReacquireHandoff(
        obj_id=obj_id,
        lost_frame=lost_frame,
        last_seen_frame=last_seen_frame,
        missing_frames=missing_frames,
        bbox=bbox,
        centroid=(
            None if bbox is None else ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        ),
        cavity_area_px=cavity_area_px,
        reference_area_ratio=reference_area_ratio,
    )


def _facts(scene: RackScene, frame_index: int = 0) -> dict[str, dict]:
    """用**产品主路径的同一套**渲染 + 几何核算出每袋的真实几何。

    门限用例不能拿手编的框去喂：框对不上真实袋体，再合理的门限也只会「全部拒绝」，
    那测的就不是门限而是坐标错了。
    """
    image = render_rack_frame(scene, frame_index)
    out: dict[str, dict] = {}
    for index, bag in enumerate(scene.bags):
        prompt = slot_seed_prompt(image.shape, index, len(scene.bags))
        cavity = int(PREDICTOR.estimate_cavity(image, prompt).area_px)
        preds = PREDICTOR.predict(image, prompt)
        best = preds[0] if preds and not preds[0].is_empty else None
        out[bag.bag_seq] = {
            "slot_index": index,
            "bbox": best.bbox if best is not None else None,
            "cavity": max(1, cavity),
            "ratio": (best.area / max(1, cavity)) if best is not None else None,
        }
    return out


def _view_for(
    scene: RackScene,
    obj_id: str = "B01",
    *,
    frame_index: int = 0,
    reference_frame: int | None = None,
    descriptor=None,
) -> LostTrackView:
    """按场景真实几何造一条「已交出」的轨迹快照。

    ``reference_frame`` 指定参考余量取自哪一帧（默认与 ``frame_index`` 同帧）。换袋
    场景里新袋更满，正是拿它当参考余量才会触发「余量回升」这一条拒绝理由。
    """
    facts = _facts(scene)
    info = facts[obj_id]
    reference = info["ratio"] if reference_frame is None else _facts(scene, reference_frame)[obj_id]["ratio"]
    return LostTrackView.from_handoff(
        _handoff(
            obj_id,
            bbox=info["bbox"],
            cavity_area_px=info["cavity"],
            reference_area_ratio=reference,
        ),
        slot_index=info["slot_index"],
        slot_count=len(scene.bags),
        descriptor=descriptor,
    )


def _scene_with_gap(
    gap: tuple[int, int], *, frame_count: int = FRAME_COUNT, refill: bool = False
) -> RackScene:
    """两袋场景；B01 在 ``gap`` 区间被挡板完全盖住。

    ``refill=True`` 时遮挡之后换成**一只更满的新袋**（换液后的画面），用来验证
    「不把新袋接到旧曲线」。
    """
    scene = default_rack_scene(frame_count=frame_count)
    bag = scene.bags[0]
    if refill:
        # 第 gap 结束那一帧起，这个挂点上换成一只全新且更满的袋子（护士已经换液）
        bag = BagSpec(
            bag.bag_seq,
            bag.x_center,
            bag.width_ratio,
            bag.fill_start,
            bag.fill_end,
            bag.liquid_rgb,
            refill_at=gap[1],
            refill_to=0.95,
            refill_rgb=(198, 96, 88),
        )
    return RackScene(bags=(bag, scene.bags[1]), frame_count=frame_count)


def _frames(tmp_path: Path, scene: RackScene, occluders, **variant) -> list[Path]:
    return write_rack_frames(tmp_path / "frames", scene, occluders=occluders, **variant)


def _read(path: Path) -> np.ndarray:
    """读一帧 JPEG 为 ``HxWx3`` ``uint8`` RGB（与产品主路径同一口径：产品读的就是 JPEG）。"""
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _image(scene: RackScene, frame_index: int = 0) -> np.ndarray:
    """就地渲染一帧 RGB（门限级用例走这条，省去落盘往返）。"""
    return render_rack_frame(scene, frame_index)


# ── 1. 与遮挡恢复不打架 ─────────────────────────────────────────────────


def test_untracked_is_left_untouched():
    """还没交出的轨迹不进再获取——不与遮挡恢复路径抢同一袋。"""
    tracker = OcclusionTracker(
        obj_id="B01", cavity_area_px=40_000, frame_count=12, params=VisibilityParams()
    )
    view = LostTrackView.from_tracker("B01", tracker, slot_index=0, slot_count=2)
    assert view.lost is False

    batch = reacquire_tracks([view], np.zeros((240, 320, 3), dtype=np.uint8), PREDICTOR, frame_index=3)

    assert batch.scanned == 0
    assert batch.roi is None, "没有交出单就不该做 ROI"
    assert [v.state for v in batch.verdicts] == [UNBOUND]
    assert batch.verdicts[0].reason == REASON_NOT_LOST
    assert tracker.handoff is None, "再获取不得凭空造出交出单"


def test_occlusion_still_recovers_itself_before_lost():
    """遮挡未超窗口时，恢复点仍按记忆接回——再获取根本没被触发。"""
    scene = _scene_with_gap((5, 8))
    paths = _frames(Path("/tmp/ivm-reacquire-a"), scene, [(0, 5, 8)])
    curves = {
        curve.obj_id: curve
        for curve in analyse_frames(paths, _seeds(scene), PREDICTOR, reacquire=None)
    }
    b01 = curves["B01"]
    assert b01.handoff is None, "遮挡 3 帧 < 丢失窗口 8 帧，不该交出"
    assert b01.reacquires == []
    assert b01.area_ratio_series[8] is not None, "第 8 帧已恢复"
    assert b01.area_ratio_series[5:8] == [None, None, None], "遮挡段留空不插值"


def _seeds(scene: RackScene):
    from infusion_monitor.core.predictor import Prompt

    return [
        (bag.bag_seq, slot_seed_prompt((scene.height, scene.width), index, len(scene.bags)))
        for index, bag in enumerate(scene.bags)
    ]


# ── 2. 单条再获取成功 ───────────────────────────────────────────────────


def test_single_track_reacquires_and_keeps_same_bag_seq(tmp_path: Path):
    """遮挡超窗口 → lost → 袋子回来 → 接回同一袋，曲线续上。"""
    scene = _scene_with_gap((3, 12))
    paths = _frames(tmp_path, scene, [(0, 3, 12)])
    curves = {
        curve.obj_id: curve
        for curve in analyse_frames(paths, _seeds(scene), PREDICTOR)
    }
    b01 = curves["B01"]

    assert b01.handoff is not None, "遮挡 9 帧 > 丢失窗口 8 帧应交出"
    assert b01.handoff.lost_frame == 11, f"第 11 帧起 lost，实得 {b01.handoff.lost_frame}"
    assert b01.reacquired is True, f"袋子在第 12 帧回来，应被找回；判定={[v.to_dict() for v in b01.reacquires]}"

    # 第 12 帧袋子已经回到画面，但再获取有 2 帧冷却（避免在同一片遮挡上反复试），
    # 因此最早在第 13 帧接回——晚一点，但绝不早于「袋子真的回来」。
    frame_back = b01.reacquired_frame
    assert frame_back == b01.handoff.lost_frame + REACQUIRE_GAP_FRAMES, (
        f"应恰好在冷却窗口后的第一帧接回，实得 {frame_back}"
    )
    assert b01.area_ratio_series[frame_back] is not None, "接回帧要有面积比"
    assert b01.area_ratio_series[3:frame_back] == [None] * (frame_back - 3), (
        "遮挡段与冷却段仍留空，不插值"
    )
    lost_at = b01.handoff.lost_frame
    assert b01.visibility_states[3:lost_at] == [OCCLUDED] * (lost_at - 3), (
        f"第 3–{lost_at - 1} 帧遮挡，实得 {b01.visibility_states[3:lost_at]}"
    )
    assert b01.visibility_states[lost_at:frame_back] == [LOST] * (frame_back - lost_at), (
        "冷却期内保持 lost（不偷偷复活、也不乱试）"
    )
    assert b01.still_lost is False, "接回后不再提示「需要重新认领」"
    assert b01.is_lost is True, "但交出这一事实留在曲线上（审计要看）"


def test_reacquired_curve_matches_clean_run(tmp_path: Path):
    """接回后的读数与「这段从没被挡住」的干净跑法一致——接的是同一袋，不是新袋。"""
    scene = _scene_with_gap((3, 12))
    gap_frames = _frames(tmp_path / "gap", scene, [(0, 3, 12)])
    clean_frames = _frames(tmp_path / "clean", scene, [])

    gap = {c.obj_id: c for c in analyse_frames(gap_frames, _seeds(scene), PREDICTOR)}["B01"]
    clean = {c.obj_id: c for c in analyse_frames(clean_frames, _seeds(scene), PREDICTOR)}["B01"]

    assert gap.reacquired is True
    back = gap.reacquired_frame
    assert gap.area_ratio_series[back] == pytest.approx(clean.area_ratio_series[back], abs=0.02)
    assert gap.area_ratio_series[23] == pytest.approx(clean.area_ratio_series[23], abs=0.02)


def test_reacquire_verdict_reports_its_evidence():
    """判定要能回答「凭什么接回」——三条证据的分项都在。"""
    scene = _scene_with_gap((0, 0))
    image = _image(scene)
    facts = _facts(scene)["B01"]
    forced = LostTrackView.from_handoff(
        _handoff("B01", bbox=facts["bbox"], cavity_area_px=facts["cavity"],
                 reference_area_ratio=None),
        slot_index=0,
        slot_count=2,
    )
    batch = reacquire_tracks([forced], image, PREDICTOR, frame_index=1)
    payload = batch.verdicts[0].to_dict()
    assert payload["obj_id"] == "B01"
    assert payload["state"] == STILL_LOST, "没有参考余量就没有依据，保持 lost"
    assert payload["reason"] == REASON_NO_REFERENCE
    assert payload["detail"], "判定必须带中文说明，不能只有状态码"
    assert set(payload) == {
        "obj_id",
        "frame_index",
        "state",
        "reason",
        "confidence",
        "runner_up",
        "candidate_count",
        "candidate",
        "detail",
    }


# ── 3. 批量再获取 ───────────────────────────────────────────────────────


def test_two_lost_tracks_reacquire_independently(tmp_path: Path):
    """两条轨迹同时 lost：一次枚举服务两条，各自接回自己那条，不串号。"""
    scene = _scene_with_gap((3, 12))
    paths = _frames(tmp_path, scene, [(0, 3, 12), (1, 3, 12)])
    curves = {
        curve.obj_id: curve for curve in analyse_frames(paths, _seeds(scene), PREDICTOR)
    }
    for bag in ("B01", "B02"):
        assert curves[bag].handoff is not None, f"{bag} 应交出"
        assert curves[bag].reacquired is True, (
            f"{bag} 应被找回；判定={[v.to_dict() for v in curves[bag].reacquires]}"
        )
        assert curves[bag].area_ratio_series[curves[bag].reacquired_frame] is not None

    back = curves["B01"].reacquired_frame
    assert curves["B01"].area_ratio_series[back] != curves["B02"].area_ratio_series[back], (
        "两袋曲线应各不相同（没串号）"
    )
    assert curves["B01"].reacquired_frame == curves["B02"].reacquired_frame
    for bag, curve in curves.items():
        winner = [v for v in curve.reacquires if v.reacquired]
        assert winner, f"{bag} 至少有一条 reacquired 判定"
        assert winner[0].candidate is not None
        assert winner[0].candidate.bbox, "接回的候选要带框，便于护士核对是哪一块"
        assert winner[0].candidate.slot_score is not None, "槽位先验分必须出现"
        assert winner[0].candidate.memory_score is not None, "记忆几何分必须出现"


def test_one_candidate_cannot_serve_two_tracks():
    """一条候选只接一条轨迹：两条轨迹抢同一块时，第二条保持 lost。"""
    scene = _scene_with_gap((0, 0))
    image = _image(scene)
    facts = _facts(scene)
    # 两条轨迹交出在同一块（人为制造的歧义）
    shared = facts["B01"]["bbox"]
    views = [
        LostTrackView.from_handoff(
            _handoff("B01", bbox=shared, cavity_area_px=facts["B01"]["cavity"],
                     reference_area_ratio=facts["B01"]["ratio"]), slot_index=0, slot_count=2),
        LostTrackView.from_handoff(
            _handoff("B02", bbox=shared, cavity_area_px=facts["B02"]["cavity"],
                     reference_area_ratio=facts["B02"]["ratio"]), slot_index=1, slot_count=2),
    ]
    batch = reacquire_tracks(views, image, PREDICTOR, frame_index=4)
    winners = [v for v in batch.verdicts if v.reacquired]
    assert len(winners) <= 1, "一条候选不能同时接两条轨迹"
    ids = [v.candidate.candidate_id for v in winners]
    assert len(set(ids)) == len(ids), "同一候选不能被两条轨迹同时接走"


def test_batch_limit_defers_extra_tracks():
    """超过一次服务上限的轨迹记 batch_limit_exceeded 并留到后续帧，不静默丢弃。"""
    scene = _scene_with_gap((0, 0))
    image = _image(scene)
    info = _facts(scene)["B01"]
    views = [
        LostTrackView.from_handoff(
            _handoff(f"B0{index}", bbox=info["bbox"], cavity_area_px=info["cavity"],
                     reference_area_ratio=info["ratio"], lost_frame=index),
            slot_index=index,
            slot_count=6,
        )
        for index in range(1, 6)
    ]
    batch = reacquire_tracks(views, image, PREDICTOR, frame_index=9, params=ReacquireParams(batch_scan_limit=2))
    assert batch.scanned == 2
    assert batch.deferred == 3
    deferred = [v for v in batch.verdicts if v.reason == REASON_BATCH_LIMIT]
    assert len(deferred) == 3
    assert all(v.state == STILL_LOST for v in deferred)


# ── 4. 失败保持 lost ────────────────────────────────────────────────────


def test_roi_with_nothing_stays_lost():
    """ROI 里一块也扫不出来 → 保持 lost，要求人工认领。"""
    scene = _scene_with_gap((0, 0))
    image = _image(scene)
    blank = np.zeros_like(image)  # 全黑画面：没有任何像液体的区域
    batch = reacquire_tracks([_view_for(scene)], blank, PREDICTOR, frame_index=7)

    verdict = batch.verdicts[0]
    assert verdict.state == STILL_LOST
    assert verdict.reason == REASON_ROI_EMPTY
    assert verdict.reacquired is False
    assert verdict.detail


def test_candidates_failing_all_gates_stay_lost():
    """候选存在但三条门限全过不了 → 不接回，且给出的是「保持丢失」而不是「已接回」。"""
    scene = _scene_with_gap((0, 0))
    image = _image(scene)
    facts = _facts(scene)
    # 交出位置放到画面另一侧的输液袋上——位置与袋序都对不上，三条门限全过不了
    views = [
        LostTrackView.from_handoff(
            _handoff("B01", bbox=facts["B02"]["bbox"],
                     cavity_area_px=facts["B02"]["cavity"],
                     reference_area_ratio=facts["B02"]["ratio"]),
            slot_index=0,
            slot_count=2,
        )
    ]
    batch = reacquire_tracks(views, image, PREDICTOR, frame_index=7)
    assert batch.verdicts[0].state in {STILL_LOST, UNBOUND}
    assert batch.verdicts[0].reacquired is False


def test_ambiguous_candidates_are_not_guessed():
    """两只一模一样、液位也一样的袋子并排 → 分不出谁是谁，判歧义不猜。

    这是再获取最容易「看起来能连、其实连错」的场景：几何分相同、体积余量相同、
    槽位序号又未知（``slot_index=None``）时，两条候选的总分逐位相同——此时任何
    「取最高分」的实现都是抛硬币。产品必须显式判歧义并交给护士认领。
    """
    liquid = (92, 148, 196)
    identical = RackScene(
        bags=(
            BagSpec("B01", 0.28, 0.30, 0.62, 0.40, liquid),
            BagSpec("B02", 0.72, 0.30, 0.62, 0.40, liquid),
        ),
        frame_count=FRAME_COUNT,
    )
    image = _image(identical)
    facts = _facts(identical)
    # 丢失位置横跨两只袋子；槽位序号未知 → 槽位这一路证据不存在
    wide = (
        min(facts["B01"]["bbox"][0], facts["B02"]["bbox"][0]),
        facts["B01"]["bbox"][1],
        max(facts["B01"]["bbox"][2], facts["B02"]["bbox"][2]),
        facts["B01"]["bbox"][3],
    )
    view = LostTrackView.from_handoff(
        _handoff(
            "B01",
            bbox=wide,
            cavity_area_px=facts["B01"]["cavity"],
            reference_area_ratio=facts["B01"]["ratio"],
        ),
        slot_index=None,
        slot_count=2,
    )

    # 外观描述子可用（两只袋子同色 → 外观也分不出高下），这样总分才够得着门槛，
    # 逼出「分差不足」这一条——而不是先被总分门槛挡在门外。
    result = enumerate_candidates(image, PREDICTOR, slot_count=2)
    own = next(c for c in result.candidates if c.bbox[0] < 240)
    descriptor = describe_region(image, own.mask, bag_seq="B01", slot_index=0)
    view = LostTrackView.from_handoff(
        view.handoff, slot_index=None, slot_count=2, descriptor=descriptor
    )

    probe = reacquire_tracks(
        [view], image, PREDICTOR, frame_index=7, params=ReacquireParams(min_margin=0.0)
    ).verdicts[0]
    assert probe.candidate_count >= 2, "先确认 ROI 里确实有两个候选可比"
    assert probe.runner_up is not None, "两个候选都过门限时必须有次优分"
    assert probe.confidence >= ReacquireParams().min_score, (
        f"探针用例的总分要先过门槛（实测 {probe.confidence}）"
    )
    assert probe.runner_up == pytest.approx(probe.confidence), (
        f"两条候选分数应逐位相同（实测 {probe.confidence} vs {probe.runner_up}）"
    )

    strict = reacquire_tracks([view], image, PREDICTOR, frame_index=7).verdicts[0]
    assert strict.reacquired is False, "分差为 0 < min_margin，必须判歧义"
    assert strict.state == UNBOUND
    assert strict.reason == REASON_AMBIGUOUS
    assert "不默认取最高分" in strict.detail

    # 收紧分差门限不改变结论：两条候选分不出高下，多试几次也还是认不准
    for margin in (0.001, 0.01, 0.05):
        again = reacquire_tracks(
            [view], image, PREDICTOR, frame_index=7,
            params=ReacquireParams(min_margin=margin),
        ).verdicts[0]
        assert again.reacquired is False and again.reason == REASON_AMBIGUOUS


def test_below_threshold_is_not_connected():
    """总分没过门槛 → 不接回（门限抬高只加罚，不放水）。"""
    scene = _scene_with_gap((0, 0))
    image = _image(scene)
    batch = reacquire_tracks(
        [_view_for(scene)], image, PREDICTOR, frame_index=7,
        params=ReacquireParams(min_score=1.01),
    )
    verdict = batch.verdicts[0]
    assert verdict.reacquired is False
    assert verdict.state == UNBOUND
    assert verdict.reason == "below_threshold"


# ── 5. 不得把新袋接到旧曲线 ─────────────────────────────────────────────


def test_refilled_bag_is_not_connected_to_old_curve(tmp_path: Path):
    """换液后新袋更满 → 余量回升 → 拒绝接回旧曲线，保持 lost 交人工认领。"""
    scene = _scene_with_gap((3, 12), refill=True)
    paths = _frames(tmp_path, scene, [(0, 3, 12)])
    curves = {c.obj_id: c for c in analyse_frames(paths, _seeds(scene), PREDICTOR)}
    b01 = curves["B01"]

    assert b01.handoff is not None, "前半段遮挡应交出"
    assert b01.reacquired is False, (
        f"第 12 帧回来的是一只更满的新袋，不得接到旧曲线；判定="
        f"{[v.to_dict() for v in b01.reacquires]}"
    )
    assert b01.reacquires, "应有再获取判定记录（判了但拒绝）"
    assert b01.reacquires[-1].state in {UNBOUND, STILL_LOST}
    assert b01.area_ratio_series[12] is None, "被拒的新袋不得让曲线复活"


def test_refill_gate_is_the_thing_that_blocks_it():
    """把回升门限放到极大值后，「换袋」这一路就不再是拒绝理由——证明该门限真的在起作用。"""
    scene = _scene_with_gap((0, 0))
    image = _image(scene, frame_index=23)
    info = _facts(scene, frame_index=23)["B01"]
    # 旧袋只剩一点点（参考余量 0.02），候选却明显更满 → 命中换袋门限
    left = LostTrackView.from_handoff(
        _handoff("B01", bbox=info["bbox"], cavity_area_px=info["cavity"],
                 reference_area_ratio=0.02),
        slot_index=0, slot_count=2,
    )
    blocked = reacquire_tracks([left], image, PREDICTOR, frame_index=7)
    relaxed = reacquire_tracks(
        [left], image, PREDICTOR, frame_index=7, params=ReacquireParams(max_ratio_regain=5.0)
    )
    assert blocked.verdicts[0].reacquired is False
    assert relaxed.verdicts[0].reacquired is True, (
        "放宽回升门限后同一块应能接回——说明拒绝对应的是回升这一条"
    )
    assert relaxed.verdicts[0].candidate is not None


def test_refill_reason_code_is_reachable():
    """「换成了新袋」这条原因码可复现：参考余量极低 + 真候选更满 → 不接回。"""
    scene = _scene_with_gap((0, 0))
    image = _image(scene)
    result = enumerate_candidates(image, PREDICTOR, slot_count=2)
    assert result.candidates, "先确认真有候选可用"

    info = _facts(scene)["B01"]
    batch = reacquire_tracks(
        [LostTrackView.from_handoff(
            _handoff("B01", bbox=info["bbox"], cavity_area_px=info["cavity"],
                     reference_area_ratio=0.02),
            slot_index=0, slot_count=2)],
        image, PREDICTOR, frame_index=7,
    )
    assert batch.verdicts[0].reacquired is False
    # 原因要么直接落在换袋门限，要么落在其余门限——但**绝不能**是已接回
    assert batch.verdicts[0].state in {UNBOUND, STILL_LOST}


# ── 6. 主路径接线 ───────────────────────────────────────────────────────


def test_main_path_can_be_switched_off_for_comparison(tmp_path: Path):
    """对照组：关掉再获取后，整条曲线在遮挡之后全程不复活。"""
    scene = _scene_with_gap((3, 12))
    paths = _frames(tmp_path, scene, [(0, 3, 12)])
    curves = {c.obj_id: c for c in analyse_frames(paths, _seeds(scene), PREDICTOR, reacquire=None)}
    b01 = curves["B01"]
    assert b01.handoff is not None
    assert b01.reacquired is False
    assert b01.area_ratio_series[12] is None, "关闭再获取时袋子回来也不接"
    assert b01.visibility_states[12] == LOST
    assert b01.still_lost is True, "整段都没接回来 → 等护士认领"


def test_reacquire_path_is_zero_cost_without_occlusion(tmp_path: Path):
    """无遮挡的正常拍摄不该触发再获取（到期集合为空 → 一次枚举都不跑）。"""
    scene = _scene_with_gap((0, 0))
    paths = _frames(tmp_path, scene, [])
    curves = {c.obj_id: c for c in analyse_frames(paths, _seeds(scene), PREDICTOR)}
    for curve in curves.values():
        assert curve.reacquires == [], f"{curve.obj_id} 无遮挡却产生了再获取判定"
        assert curve.handoff is None


def test_roi_is_bounded_by_params():
    """ROI 面积有上限：丢 6 条轨迹也不会把单帧耗时顶穿。"""
    params = ReacquireParams()
    views = [
        LostTrackView.from_handoff(
            _handoff(
                f"B{index}",
                bbox=(10 + index * 4, 10, 60 + index * 4, 90),
                cavity_area_px=10_000,
                reference_area_ratio=0.4,
                lost_frame=index,
            ),
            slot_index=index,
            slot_count=6,
        )
        for index in range(6)
    ]
    roi = roi_for_tracks(views, (240, 320), params)
    assert roi is not None
    assert (roi[2] - roi[0]) * (roi[3] - roi[1]) <= params.roi_max_area_px


def test_roi_none_when_no_position_known():
    """交出单里没有位置 → 不猜，直接不扫。"""
    view = LostTrackView.from_handoff(
        _handoff("B01", bbox=None, cavity_area_px=40_000, reference_area_ratio=None),
        slot_index=0,
        slot_count=2,
    )
    assert view.reference_entry is None
    assert roi_for_tracks([view], (240, 320), ReacquireParams()) is None


def test_reference_entry_is_built_from_the_handoff():
    """交出单复原出的记忆条目能回答「最后可见那一帧长什么样」。"""
    handoff = _handoff(
        "B01", bbox=(60, 40, 200, 420), cavity_area_px=40_000, reference_area_ratio=0.40
    )
    entry = LostTrackView.from_handoff(handoff, slot_index=0, slot_count=2).reference_entry
    assert entry is not None
    assert entry.bbox == (60, 40, 200, 420)
    assert entry.centroid == (130.0, 230.0)
    assert entry.frame_index == handoff.last_seen_frame
    assert entry.ratio == pytest.approx(0.40)
    assert entry.area == pytest.approx(16_000)


def test_descriptor_gates_reacquire(tmp_path: Path):
    """挂上外观描述子后，外观分会真正参与打分（分项字段非空且与不挂时不同）。"""
    scene = _scene_with_gap((0, 0))
    image = _image(scene)
    plain = reacquire_tracks([_view_for(scene)], image, PREDICTOR, frame_index=7).verdicts[0]

    result = enumerate_candidates(image, PREDICTOR, slot_count=2)
    assert result.candidates
    candidate = result.candidates[0]
    descriptor = describe_region(image, candidate.mask, bag_seq="B01", slot_index=0)
    assert descriptor.appearance_available, "候选区域足够大，外观应可用"

    with_desc = reacquire_tracks(
        [_view_for(scene, descriptor=descriptor)], image, PREDICTOR, frame_index=7
    ).verdicts[0]

    assert plain.reacquired is True
    assert with_desc.reacquired is True
    assert with_desc.candidate is not None
    assert with_desc.candidate.appearance_score is not None, "挂上描述子后外观分必须出现"
    assert plain.candidate is not None
    assert plain.candidate.appearance_score is None
    assert with_desc.candidate.total != plain.candidate.total, "加入外观权重后总分应变化"


def test_missing_channel_is_renormalised_not_penalised():
    """缺一条证据时，权重按**剩下的通道重新归一**，而不是白白扣掉那一路的权重。

    槽位序号未知（``slot_index=None``）时槽位那一路不存在；若仍把它算进分母，总分
    会凭空少 ``slot_weight``×槽位分，证据齐备的轨迹也会被 ``min_score`` 挡死。
    这条用例把两种口径摆在一起对比，防止回退。
    """
    scene = _scene_with_gap((0, 0))
    image = _image(scene)
    result = enumerate_candidates(image, PREDICTOR, slot_count=2)
    own = next(c for c in result.candidates if c.bbox[0] < 240)
    descriptor = describe_region(image, own.mask, bag_seq="B01", slot_index=0)

    with_slot = reacquire_tracks(
        [_view_for(scene, descriptor=descriptor)], image, PREDICTOR, frame_index=7
    ).verdicts[0]
    without_slot = reacquire_tracks(
        [LostTrackView.from_handoff(
            _view_for(scene, descriptor=descriptor).handoff,
            slot_index=None,
            slot_count=2,
            descriptor=descriptor,
        )],
        image,
        PREDICTOR,
        frame_index=7,
    ).verdicts[0]

    assert with_slot.reacquired is True
    assert without_slot.reacquired is True, "少了槽位这一路不该反而接不回来"
    assert with_slot.candidate is not None and without_slot.candidate is not None
    assert without_slot.candidate.total >= with_slot.candidate.total, (
        f"去掉槽位分后总分应不降（实测 {without_slot.candidate.total} < "
        f"{with_slot.candidate.total}）——权重被重复扣了"
    )
    assert without_slot.candidate.slot_score is None


def test_appearance_unavailable_raises_the_gate(tmp_path: Path):
    """外观不可用不当作 0 分，而是给几何证据加罚——门槛抬高、判定如实说明。"""
    scene = _scene_with_gap((0, 0))
    image = _image(scene)
    verdict = reacquire_tracks(
        [_view_for(scene)], image, PREDICTOR, frame_index=7,
        params=ReacquireParams(appearance_penalty=0.9),
    ).verdicts[0]
    assert verdict.reacquired is False
    assert "外观" in verdict.detail