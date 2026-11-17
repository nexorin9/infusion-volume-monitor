"""重锚打分与判定的验收测试（几何 + 颜色 + 液位 + 标签区四项加权）。

**三类场景**（任务书要求的主场景）：

1. **同袋跨轮**——位置不变、液面下降、颜色不变 → 高置信接上，曲线续在同一条上；
2. **同槽位换新袋**——位置不变但颜色变了、液位比上一轮还高 → 低置信、判
   ``appearance_changed``，要求护士认领，**新袋绝不接到旧曲线上**；
3. **换袋又换位**——新袋挂到了另一个挂点 → 认不出，unbound 且不给出历史会话指针。

**另有四组口径断言**：缺证据的项退出加权（不当 0 分）、标签有→无判 0、
候选排序与分差（拒绝猜最高分）、四类判据的归因（``driver``）说得清。
"""

from __future__ import annotations

import numpy as np
import pytest

from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
from infusion_monitor.core.predictor import Prompt
from infusion_monitor.identity.descriptor import BagAppearance, describe_region
from infusion_monitor.identity.reanchor import SlotObservation, reanchor_slots
from infusion_monitor.identity.reanchor_score import (
    ScoreParams,
    colour_score,
    geometry_score,
    label_score,
    level_score,
    rank_candidates,
    score_candidate,
)
from tests.support import BagSpec, RackScene, build_rack_zip, render_rack_frame

TRACKS_URL = "/api/v1/infusion/tracks"

#: 两轮巡视的拍摄时刻（重锚按 captured_at 排先后）。
CAPTURED_AT = {
    "AM": "2026-10-22T09:41:00+08:00",
    "PM": "2026-10-22T13:41:00+08:00",
}


# ── 样例工具 ──────────────────────────────────────────────────────────────


def _scene(
    *,
    b01_fill: tuple[float, float],
    b01_rgb: tuple[int, int, int] = (92, 148, 196),
    swap_slots: bool = False,
    frames: int = 16,
) -> RackScene:
    """两袋场景。``swap_slots`` 把两只袋子的挂点对调（换袋又换位用）。"""
    b01 = BagSpec("B01", 0.28, 0.30, *b01_fill, b01_rgb)
    b02 = BagSpec("B02", 0.72, 0.30, 0.75, 0.55, (64, 128, 190))
    if swap_slots:  # 挂点对调：B02 挂到左边、B01 挂到右边
        b01 = BagSpec("B01", 0.72, 0.30, *b01_fill, b01_rgb)
        b02 = BagSpec("B02", 0.28, 0.30, 0.75, 0.55, (64, 128, 190))
    return RackScene(bags=(b01, b02), frame_count=frames)


def _descriptor(
    scene: RackScene, frame_index: int, slot: int, *, session_id: str, shift_key: str
) -> BagAppearance:
    """走产品主路径（几何核切掩膜 → 描述子），不手搓像素。"""
    image = np.asarray(render_rack_frame(scene, frame_index))
    point = (scene.bags[slot].x_center * image.shape[1], image.shape[0] * 0.55)
    mask = GeometricMaskPredictor().predict(image, Prompt.from_point(*point))[0].mask
    return describe_region(
        image,
        mask,
        bag_seq=scene.bags[slot].bag_seq,
        slot_index=slot,
        session_id=session_id,
        shift_key=shift_key,
    )


def _previous_round() -> tuple[BagAppearance, BagAppearance]:
    scene = _scene(b01_fill=(0.62, 0.62))
    return (
        _descriptor(scene, 0, 0, session_id="sess-am-B01", shift_key="2026-10-22-AM"),
        _descriptor(scene, 0, 1, session_id="sess-am-B02", shift_key="2026-10-22-AM"),
    )


# ── 场景 1：同袋跨轮（高置信） ────────────────────────────────────────────


def test_same_bag_across_rounds_scores_high_and_binds():
    previous = _previous_round()
    scene = _scene(b01_fill=(0.40, 0.40))  # 液位降了，颜色与位置不变
    current = _descriptor(scene, 0, 0, session_id="sess-pm-B01", shift_key="2026-10-22-PM")

    ranked = rank_candidates(current, previous)
    assert [item.bag_seq for item in ranked.ordered] == ["B01", "B02"]
    best = ranked.best
    assert best.bag_seq == "B01", "同袋跨轮应排在自己前面"
    assert best.total >= 0.95
    assert best.geometry > 0.95
    assert best.level == 1.0, "液面下降不该扣分"
    assert best.appearance is not None and best.appearance > 0.9
    assert ranked.margin is not None and ranked.margin > ScoreParams().min_margin

    verdict = reanchor_slots([SlotObservation(0, "B01", current)], previous)[0]
    assert (verdict.anchor_state, verdict.reason) == ("bound", "anchored")
    assert verdict.matched_session_id == "sess-am-B01"
    assert verdict.previous_shift_key == "2026-10-22-AM"
    assert verdict.descriptor_mode in ("full", "color_only")


# ── 场景 2：同槽位换新袋（低置信 + 转人工认领） ────────────────────────────


def test_new_bag_in_same_slot_scores_low_and_requires_claim():
    previous = _previous_round()
    # 同一挂点，但换成几乎满的新袋 + 另一种药液颜色
    scene = _scene(b01_fill=(0.98, 0.98), b01_rgb=(206, 120, 86))
    current = _descriptor(scene, 0, 0, session_id="sess-pm-B01", shift_key="2026-10-22-PM")

    verdict = reanchor_slots([SlotObservation(0, "B01", current)], previous)[0]
    assert verdict.anchor_state == "unbound"
    assert verdict.reason == "appearance_changed", "换袋要归因到外观变化，不是「分不够」"
    assert verdict.matched_session_id is None, "认不准还给出历史会话指针 = 把新袋接到旧曲线上"
    assert verdict.previous_shift_key is None
    assert verdict.confidence < ScoreParams().min_confidence

    # 归因说得清：颜色与液位两项都塌了
    best = rank_candidates(current, previous).best
    assert best.geometry >= ScoreParams().geometry_trust, "位置没变，不该靠几何归因"
    assert best.colour is not None and best.colour < 0.5
    assert best.level is not None and best.level == 0.0, "液柱比上一轮高 = 换了新袋"
    assert best.driver in ("colour", "level")


def test_liquid_level_that_rises_is_the_strongest_signal_for_a_new_bag():
    """同一袋液柱只会变矮；变高（含颜色没变）同样必须低置信。"""
    previous = [_previous_round()[0]]
    scene = _scene(b01_fill=(0.90, 0.90))  # 颜色仍是同一种，只是袋子满了
    current = _descriptor(scene, 0, 0, session_id="sess-pm-B01", shift_key="2026-10-22-PM")

    assert current.liquid_height > previous[0].liquid_height
    assert level_score(current, previous[0]) == 0.0
    verdict = reanchor_slots([SlotObservation(0, "B01", current)], previous)[0]
    assert verdict.anchor_state == "unbound"
    assert verdict.reason == "appearance_changed"


# ── 场景 3：换袋又换位（unbound） ─────────────────────────────────────────


def test_bag_replaced_and_moved_to_another_slot_is_unbound():
    previous = _previous_round()
    scene = _scene(b01_fill=(0.95, 0.95), b01_rgb=(206, 120, 86), swap_slots=True)
    current = _descriptor(scene, 0, 1, session_id="sess-pm-B01", shift_key="2026-10-22-PM")

    ranked = rank_candidates(current, previous)
    verdict = reanchor_slots([SlotObservation(1, "B01", current)], previous)[0]
    assert verdict.anchor_state == "unbound"
    assert verdict.matched_session_id is None
    assert verdict.reason in ("low_confidence", "appearance_changed", "bag_seq_mismatch")
    # 位置也变了：几何项扣分，几何分必须低于「位置可信」的门槛
    assert ranked.best.geometry < ScoreParams().geometry_trust
    # 排序仍然给出「最像谁」供护士自己认领（只排序不判定）
    assert [item.bag_seq for item in ranked.ordered][0] in {"B01", "B02"}


# ── 口径：缺证据的项退出加权，不当 0 分 ───────────────────────────────────


def test_missing_label_evidence_is_dropped_not_scored_zero():
    """旧版描述子没有标签字段 → 标签项退出加权；总分不能被一个「没有的证据」压低。"""
    hist = [0.0] * 12
    hist[3] = 1.0
    payload = {
        "bag_seq": "B01",
        "slot_index": 0,
        "centre_x": 0.3,
        "bottom_y": 0.7,
        "width": 0.3,
        "hue_hist": hist,
        "liquid_height": 0.2,
        "liquid_top_y": 0.5,
        "state": "monitoring",
    }
    legacy = BagAppearance.from_json(payload, bag_seq="B01", slot_index=0)
    fresh = BagAppearance.from_json(payload, bag_seq="B01", slot_index=0)
    assert legacy.descriptor_mode == "color_only"
    assert legacy.label_available is False

    scored = score_candidate(fresh, legacy)
    assert scored.label is None
    assert "label" not in scored.terms_used
    assert scored.total >= ScoreParams().min_confidence, "缺一条证据不该把总分按缺满分算"


def test_identical_descriptors_score_one_on_every_term():
    previous = _previous_round()[0]
    scored = score_candidate(previous, previous)
    assert scored.total == pytest.approx(1.0)
    assert scored.geometry == pytest.approx(1.0)
    assert scored.colour == pytest.approx(1.0)
    assert scored.level == pytest.approx(1.0)
    assert scored.terms_used == ("geometry", "colour", "level", "label")


# ── 口径：标签区有 → 无 ───────────────────────────────────────────────────


def test_label_appearing_or_disappearing_between_rounds_scores_zero():
    """标签有→无：标签项判 0 并被标成拖分项。

    但**单靠这一项不解除绑定**：袋子被翻面、标签转到看不见的那一面，同样是这个结果，
    而那种情况接上旧曲线是对的。标签是四项里最弱的一项（权重最低），它负责的是
    「同色换品牌」这种拿颜色看不出来的情形——真到该拒的时候，颜色或液位会同时塌。
    """
    base = _previous_round()[0]
    with_label = BagAppearance(**{**base.__dict__, "label_available": True, "label_present": True,
                                  "label_coverage": 0.9, "label_ink": 0.05})
    without_label = BagAppearance(**{**base.__dict__, "label_available": True, "label_present": False,
                                     "label_coverage": 0.9, "label_ink": 0.0})
    assert label_score(without_label, with_label) == 0.0
    assert label_score(with_label, without_label) == 0.0
    assert label_score(with_label, with_label) == pytest.approx(1.0)

    scored = score_candidate(without_label, with_label)
    assert scored.label == 0.0
    assert scored.driver == "label"
    assert 0.80 < scored.total < 1.0, "扣掉的是标签那一份权重，不是整条判据"
    assert reanchor_slots([SlotObservation(0, "B01", without_label)], [with_label])[0].bound


def test_label_flip_together_with_a_colour_change_is_rejected():
    """换袋（颜色变 + 标签没了一起来）→ 判 appearance_changed，要求认领。"""
    base = _previous_round()[0]
    flipped = BagAppearance(
        **{
            **base.__dict__,
            "hue_hist": tuple(1.0 if i == 9 else 0.0 for i in range(12)),
            "label_available": True,
            "label_present": False,
            "label_coverage": 0.9,
            "label_ink": 0.0,
        }
    )
    labelled = BagAppearance(
        **{
            **base.__dict__,
            "label_available": True,
            "label_present": True,
            "label_coverage": 0.9,
            "label_ink": 0.05,
        }
    )
    verdict = reanchor_slots([SlotObservation(0, "B01", flipped)], [labelled])[0]
    assert verdict.anchor_state == "unbound"
    assert verdict.reason == "appearance_changed"
    assert verdict.matched_session_id is None


# ── 口径：排序与拒绝猜最高分 ─────────────────────────────────────────────


def test_two_look_alike_history_bags_are_ambiguous_not_best_guess():
    """两个几乎一样的历史袋 → 分差不够 → ambiguous，绝不默认取最高分。"""
    scene = _scene(b01_fill=(0.62, 0.62))
    b01 = _descriptor(scene, 0, 0, session_id="sess-am-B01", shift_key="2026-10-22-AM")
    twin = BagAppearance(**{**b01.__dict__, "bag_seq": "B02", "session_id": "sess-am-B02",
                            "centre_x": b01.centre_x + 0.002})
    current = _descriptor(
        _scene(b01_fill=(0.45, 0.45)), 0, 0, session_id="sess-pm-B01", shift_key="2026-10-22-PM"
    )

    ranked = rank_candidates(current, [b01, twin])
    assert ranked.margin is not None and ranked.margin < ScoreParams().min_margin
    verdict = reanchor_slots([SlotObservation(0, "B01", current)], [b01, twin])[0]
    assert (verdict.anchor_state, verdict.reason) == ("unbound", "ambiguous_slot")
    assert verdict.matched_session_id is None
    assert len(verdict.scores) == 2, "排序明细要留给护士核对"


def test_score_breakdown_is_json_serialisable_for_audit():
    previous = _previous_round()
    current = _descriptor(_scene(b01_fill=(0.45, 0.45)), 0, 0, session_id="s", shift_key="PM")
    payload = rank_candidates(current, previous).best.to_json()
    assert set(payload) >= {"bag_seq", "total", "geometry", "colour", "level", "label", "driver"}
    assert isinstance(payload["terms_used"], list)


# ── 端到端：换袋不会把新袋接到旧曲线上 ───────────────────────────────────


def _upload(client, scene: RackScene, *, shift: str, slots: str):
    return client.post(
        TRACKS_URL,
        data={
            "rack_id": "R-EXAMPLE-29",
            "infusion_order_no": "EXAMPLE-ORDER-0029",
            "bed_no": "EXAMPLE-BED-29",
            "captured_at": CAPTURED_AT[shift],
            "shift_key": f"2026-10-22-{shift}",
            "slots": slots,
        },
        files={"frames": ("frames.zip", build_rack_zip(scene), "application/zip")},
    )


TWO_BAG_SLOTS = (
    '[{"bag_seq":"B01","slot_index":0,"bag_type_code":"TYPE-A","rate_text":"5ml/h"},'
    '{"bag_seq":"B02","slot_index":1,"bag_type_code":"TYPE-A","rate_text":"5ml/h"}]'
)


def test_upload_after_visual_bag_swap_asks_for_claim_and_keeps_curves_separate(store_client):
    """第二轮 B01 被换成新袋（对端没推进袋序）→ 响应 unbound/needs_claim，不指历史会话。"""
    client, conn = store_client
    first = _upload(client, _scene(b01_fill=(0.62, 0.62)), shift="AM", slots=TWO_BAG_SLOTS)
    assert first.status_code == 201, first.text
    assert {bag["reanchor_state"] for bag in first.json()["bags"]} == {"fresh"}

    swapped = _scene(b01_fill=(0.98, 0.98), b01_rgb=(206, 120, 86))
    second = _upload(client, swapped, shift="PM", slots=TWO_BAG_SLOTS)
    assert second.status_code == 201, second.text
    by_seq = {bag["bag_seq"]: bag for bag in second.json()["bags"]}

    b01 = by_seq["B01"]
    assert b01["reanchor_state"] == "unbound"
    assert b01["reanchor_reason"] == "appearance_changed"
    assert b01["candidate_state"] == "needs_claim"
    assert "previous_session_id" not in b01, "认不准却给出历史会话指针"
    assert b01["reanchor_driver"] in ("colour", "level")
    assert b01["descriptor_mode"] in ("full", "color_only")
    assert b01["reanchor_confidence"] < 0.62

    # 同一架上没换袋的那一只照常续上（打分不是一刀切地拒）
    assert by_seq["B02"]["reanchor_state"] == "bound"
    assert by_seq["B02"]["previous_shift_key"] == "2026-10-22-AM"


def test_round_trip_stores_new_descriptor_fields_for_the_next_round(store_client):
    """描述子（含液位与标签区）落库，下一轮从库里读回来仍能打分。"""
    from infusion_monitor.identity.descriptor import descriptor_from_session
    from infusion_monitor.store import SessionRepository

    client, conn = store_client
    first = _upload(client, _scene(b01_fill=(0.62, 0.62)), shift="AM", slots=TWO_BAG_SLOTS)
    row = SessionRepository(conn).get(first.json()["bags"][0]["session_id"])
    descriptor = descriptor_from_session(row)

    assert descriptor is not None
    assert descriptor.liquid_height > 0
    assert descriptor.label_available, "合成短片里标签区看得见，应判 full"
    assert descriptor.descriptor_mode == "full"
    assert score_candidate(descriptor, descriptor).total == pytest.approx(1.0)