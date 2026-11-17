"""跨巡视轮次重锚与槽位认领的验收测试。

四组断言：

1. **同袋重锚成功**：同一输液架下一轮巡视还是那几只袋子，几何 + 外观高置信接上，
   曲线续在同一个 ``bag_seq`` 上（``previous_session_id`` 指向上一轮会话）；
2. **换袋后重锚到新 bag_seq**：上一轮已确认换掉的袋**不再**被接上，本轮判 ``fresh``；
3. **unbound 强制认领**：同槽位多候选 / 分差不够 / 袋序对不上，一律拒绝猜测，
   护士在候选列表里点一行完成认领（按新候选**重算整条曲线**，不是改标签）；
4. **打分口径**：袋底/横向宽度对液面下降不敏感；外观不可用时绝不自动接上。
"""

from __future__ import annotations

import numpy as np
import pytest

from infusion_monitor.identity.reanchor import (
    BagDescriptor,
    ReanchorParams,
    SlotObservation,
    describe_region,
    descriptor_from_session,
    reanchor_slots,
    score_pair,
)
from infusion_monitor.identity.state_machine import confirm_replacement
from infusion_monitor.store import AuditChain, CalibrationRepository, SessionRepository
from tests.support import BagSpec, RackScene, build_rack_zip, default_rack_scene

TRACKS_URL = "/api/v1/infusion/tracks"
CLAIM_URL = "/api/v1/infusion/claim"


class _Store:
    """测试线程侧的存储句柄（指向与 app 同一个库文件）。"""

    def __init__(self, conn) -> None:
        self.conn = conn
        self.sessions = SessionRepository(conn)
        self.audit_chain = AuditChain(conn)
        self.calibrations = CalibrationRepository(conn)


# ── 样例工具 ──────────────────────────────────────────────────────────────


#: 三轮巡视的拍摄时刻（对应 AM / PM / NIGHT）。重锚按 captured_at 排先后，
#: 不用服务端写入时刻——``utc_now`` 只到秒，测试里几轮会在同一秒落库。
SHIFT_CAPTURED_AT = {
    "2026-10-22-AM": "2026-10-22T09:41:00+08:00",
    "2026-10-22-PM": "2026-10-22T13:41:00+08:00",
    "2026-10-22-NIGHT": "2026-10-22T19:41:00+08:00",
}


def _upload(client, scene: RackScene, *, shift_key: str, slots: str, **fields):
    """上传一段短片，返回响应 JSON。"""
    payload = {
        "rack_id": "R-EXAMPLE-09",
        "infusion_order_no": "EXAMPLE-ORDER-0009",
        "bed_no": "EXAMPLE-BED-09",
        "captured_at": SHIFT_CAPTURED_AT[shift_key],
        "shift_key": shift_key,
        "slots": slots,
        **fields,
    }
    return client.post(
        TRACKS_URL,
        data=payload,
        files={"frames": ("frames.zip", build_rack_zip(scene), "application/zip")},
    )


TWO_BAG_SLOTS = (
    '[{"bag_seq":"B01","slot_index":0,"bag_type_code":"TYPE-A","rate_text":"5ml/h"},'
    '{"bag_seq":"B02","slot_index":1,"bag_type_code":"TYPE-A","rate_text":"5ml/h"}]'
)


def _descending_scene(frame_count: int = 16) -> RackScene:
    """两袋、位置不变、液位比第一轮更低的一段（第二轮巡视的现实情况）。"""
    return RackScene(
        bags=(
            BagSpec("B01", 0.28, 0.30, 0.34, 0.20, (92, 148, 196)),
            BagSpec("B02", 0.72, 0.30, 0.40, 0.26, (64, 128, 190)),
        ),
        frame_count=frame_count,
    )


def _descriptor(bag_seq: str, *, centre_x: float, hue_peak: int, session_id: str | None = None,
                state: str = "monitoring") -> BagDescriptor:
    """构造一个可控的描述子（直方图集中在指定色相 bin 上）。"""
    hist = [0.0] * 12
    hist[hue_peak] = 1.0
    return BagDescriptor(
        bag_seq=bag_seq,
        slot_index=0,
        centre_x=centre_x,
        bottom_y=0.70,
        width=0.30,
        hue_hist=tuple(hist),
        session_id=session_id,
        state=state,
    )


# ── 1. 同袋重锚成功 ───────────────────────────────────────────────────────


def test_same_bag_reanchors_across_rounds(store_client):
    """同一输液架下一轮巡视：两袋都高置信续上，且指向上一轮的会话。"""
    client, conn = store_client
    first = _upload(client, default_rack_scene(16), shift_key="2026-10-22-AM", slots=TWO_BAG_SLOTS)
    assert first.status_code == 201, first.text
    assert [bag["reanchor_state"] for bag in first.json()["bags"]] == ["fresh", "fresh"]
    assert {bag["reanchor_reason"] for bag in first.json()["bags"]} == {"first_round"}

    second = _upload(client, _descending_scene(16), shift_key="2026-10-22-PM", slots=TWO_BAG_SLOTS)
    assert second.status_code == 201, second.text
    payload = second.json()

    by_seq = {bag["bag_seq"]: bag for bag in payload["bags"]}
    for bag_seq, bag in by_seq.items():
        assert bag["reanchor_state"] == "bound", bag
        assert bag["reanchor_reason"] == "anchored"
        assert bag["reanchor_confidence"] >= 0.62
        assert bag["previous_shift_key"] == "2026-10-22-AM"
        assert bag["previous_session_id"], "接上了却没有指回上一轮会话"

    # 续的是同一袋：两袋各自接回**自己**的那一个，不是串到对方身上
    sessions = SessionRepository(conn)
    seen = set()
    for bag_seq, bag in by_seq.items():
        previous = sessions.get(bag["previous_session_id"])
        assert previous is not None
        assert previous["bag_seq"] == bag_seq, "接到了对方的袋子上"
        assert previous["shift_key"] == "2026-10-22-AM"
        seen.add(previous["session_id"])
    assert len(seen) == 2, "两袋接到了同一个历史会话"

    # 落库：描述子与判定都留痕，下一轮还能接着比
    for bag in payload["bags"]:
        row = sessions.get(bag["session_id"])
        assert row["reanchor_state"] == "bound"
        assert row["anchor_candidate_id"] == bag["candidate_id"]
        assert row["descriptor"]["hue_hist"], "描述子没落库，下一轮就没法认同一袋"
        assert row["candidates_json"], "候选没落库，护士点认领时服务端无从回查"
    # 两轮各两袋 → 4 条重锚判定审计
    assert AuditChain(conn).count_by_action()["reanchor.resolved"] == 4


def test_descriptor_survives_round_trip_through_store(store_client):
    """描述子落库再读回仍可用于打分（JSON 往返不能把 bag_seq/状态丢掉）。"""
    client, conn = store_client
    first = _upload(client, default_rack_scene(16), shift_key="2026-10-22-AM", slots=TWO_BAG_SLOTS)
    session_id = first.json()["bags"][0]["session_id"]
    row = SessionRepository(conn).get(session_id)

    restored = descriptor_from_session(row)
    assert restored is not None
    assert restored.bag_seq == row["bag_seq"]
    assert restored.session_id == row["session_id"]
    assert restored.appearance_available

    # 库里没描述子的旧行 → 返回 None（不当成「长得一样」）
    assert descriptor_from_session({"bag_seq": "B01", "slot_index": 0, "descriptor": {}}) is None
    assert descriptor_from_session({"bag_seq": "B01", "slot_index": 0}) is None


# ── 2. 换袋后重锚到新 bag_seq ─────────────────────────────────────────────


def test_replaced_bag_is_not_reanchored_new_round_gets_fresh(store_client):
    """上一轮确认换掉的袋不再被接上；本轮同挂点的新袋判 fresh 且不挂旧会话。"""
    client, conn = store_client
    first = _upload(client, default_rack_scene(16), shift_key="2026-10-22-AM", slots=TWO_BAG_SLOTS)
    assert first.status_code == 201, first.text
    store = _Store(conn)
    b01 = next(bag for bag in first.json()["bags"] if bag["bag_seq"] == "B01")
    store.sessions.apply_state_transition(b01["session_id"], "needs_change")
    confirm_replacement(
        store.sessions,
        b01["session_id"],
        operator_id="N-EXAMPLE-1",
        audit=store.audit_chain,
        request_id="req-replace-1",
    )

    # 第二轮：B01 换成了新袋，B02 还是原来那只
    second = _upload(client, _descending_scene(16), shift_key="2026-10-22-PM", slots=TWO_BAG_SLOTS)
    assert second.status_code == 201, second.text
    by_seq = {bag["bag_seq"]: bag for bag in second.json()["bags"]}

    assert by_seq["B01"]["reanchor_state"] == "fresh", "已换掉的袋被接了回来"
    assert by_seq["B01"]["reanchor_reason"] == "previous_replaced"
    assert "previous_session_id" not in by_seq["B01"]
    assert by_seq["B02"]["reanchor_state"] == "bound"

    # 新袋没有被接到旧袋曲线上：B01 的描述子与旧会话的描述子不同源
    row = store.sessions.get(by_seq["B01"]["session_id"])
    assert row["previous_session_id"] is None
    assert row["state"] == "monitoring"


# ── 3. unbound：拒绝猜测，强制走人工认领 ──────────────────────────────────


def test_ambiguous_slot_is_not_bound_and_requires_claim():
    """两个几乎一样的历史袋 → 分差不够 → unbound，绝不默认取分数最高的那个。"""
    current = _descriptor("B01", centre_x=0.30, hue_peak=3)
    previous = [
        _descriptor("B01", centre_x=0.30, hue_peak=3, session_id="sess-a"),
        _descriptor("B02", centre_x=0.302, hue_peak=3, session_id="sess-b"),
    ]
    verdicts = reanchor_slots(
        [SlotObservation(0, "B01", current)], previous, ReanchorParams()
    )
    verdict = verdicts[0]

    assert verdict.anchor_state == "unbound"
    assert verdict.reason == "ambiguous_slot"
    assert verdict.matched_session_id is None
    assert len(verdict.scores) == 2
    assert verdict.confidence - verdict.runner_up < 0.08


def test_low_confidence_is_not_bound():
    """看着完全不像（位置差得远、色相完全不同）→ low_confidence。"""
    current = _descriptor("B01", centre_x=0.10, hue_peak=3)
    previous = [_descriptor("B01", centre_x=0.80, hue_peak=11, session_id="sess-a")]
    verdict = reanchor_slots(
        [SlotObservation(0, "B01", current)], previous, ReanchorParams()
    )[0]
    assert verdict.anchor_state == "unbound"
    assert verdict.reason == "low_confidence"
    assert verdict.confidence < 0.62


def test_bag_seq_mismatch_requires_manual_check():
    """认出的袋与对端给的 bag_seq 对不上 → 要求人工核对，不把曲线接到错的袋序上。"""
    current = _descriptor("B03", centre_x=0.30, hue_peak=3)
    previous = [_descriptor("B01", centre_x=0.30, hue_peak=3, session_id="sess-a")]
    verdict = reanchor_slots(
        [SlotObservation(0, "B03", current)], previous, ReanchorParams()
    )[0]
    assert verdict.anchor_state == "unbound"
    assert verdict.reason == "bag_seq_mismatch"
    assert verdict.matched_bag_seq == "B01"  # 分数给得出来，但不据此改袋序


def test_slot_without_candidate_is_unbound():
    """本轮这个挂点没枚举出候选 → no_candidate（连打分都不做）。"""
    verdict = reanchor_slots(
        [SlotObservation(1, "B02", None)], [_descriptor("B01", centre_x=0.3, hue_peak=3)]
    )[0]
    assert verdict.anchor_state == "unbound"
    assert verdict.reason == "no_candidate"
    assert verdict.scores == ()


def test_claim_recomputes_curve_and_leaves_audit(store_client):
    """护士在候选列表里点一行：按该候选重算整条曲线，并留 append-only 审计行。"""
    client, conn = store_client
    store = _Store(conn)
    scene = default_rack_scene(16)
    upload = _upload(client, scene, shift_key="2026-10-22-AM", slots=TWO_BAG_SLOTS)
    assert upload.status_code == 201, upload.text
    payload = upload.json()
    bag = payload["bags"][0]
    candidate_id = bag["candidate_id"]
    assert candidate_id, "没有绑上候选就没法测认领"

    before = store.sessions.get(bag["session_id"])
    claim = client.post(
        CLAIM_URL,
        json={
            "session_id": bag["session_id"],
            "candidate_id": candidate_id,
            "operator_id": "N-EXAMPLE-7",
            "note": "床旁核对后确认是这袋",
        },
    )
    assert claim.status_code == 200, claim.text
    result = claim.json()
    assert result["claim_state"] == "claimed"
    assert result["claimed_by"] == "N-EXAMPLE-7"
    assert len(result["area_ratio_series"]) == before["frame_count"]
    assert result["area_ratio_series"] == before["area_ratio_series"]

    after = store.sessions.get(bag["session_id"])
    assert after["claim_state"] == "claimed"
    assert after["claimed_by"] == "N-EXAMPLE-7"
    assert after["anchor_candidate_id"] == candidate_id
    assert after["anchor_bbox"], "认领的框没落库，护士端点错了没法纠错"

    rows = store.audit_chain.list_by_target(bag["session_id"])
    claimed = [row for row in rows if row["action"] == "candidate.claimed"]
    assert len(claimed) == 1
    assert claimed[0]["actor"] == "nurse:N-EXAMPLE-7"
    assert claimed[0]["detail"]["candidate_id"] == candidate_id
    assert claimed[0]["payload_hash"], "审计行缺摘要，无法核对内容有没有被改过"


def test_repeat_claim_of_same_candidate_is_idempotent(client):
    """同一候选重复认领按幂等处理（不算改绑，不重复纠正）。"""
    scene = default_rack_scene(16)
    upload = _upload(client, scene, shift_key="2026-10-22-AM", slots=TWO_BAG_SLOTS)
    bag = upload.json()["bags"][0]
    body = {
        "session_id": bag["session_id"],
        "candidate_id": bag["candidate_id"],
        "operator_id": "N-EXAMPLE-7",
    }
    first = client.post(CLAIM_URL, json=body).json()
    second = client.post(CLAIM_URL, json=body).json()

    assert first["corrected"] is False
    assert second["corrected"] is False
    assert second["claim_state"] == "claimed"


def test_claim_rejects_unknown_candidate(client):
    """认领一个不存在的候选 → 422，并列出本轮可认领的候选（护士照着就能改对）。"""
    upload = _upload(client, default_rack_scene(16), shift_key="2026-10-22-AM", slots=TWO_BAG_SLOTS)
    bag = upload.json()["bags"][0]
    response = client.post(
        CLAIM_URL,
        json={
            "session_id": bag["session_id"],
            "candidate_id": "C-NOT-THERE",
            "operator_id": "N-EXAMPLE-7",
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error_code"] == "INGEST_INVALID"
    assert detail["issues"][0]["code"] == "candidate_not_found"
    assert bag["candidate_id"] in detail["issues"][0]["reason"]


def test_claim_rejects_unknown_session(client):
    response = client.post(
        CLAIM_URL,
        json={"session_id": "sess_nope", "candidate_id": "C01", "operator_id": "N-EXAMPLE-7"},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "NOT_FOUND"


# ── 4. 打分口径 ───────────────────────────────────────────────────────────


def test_geometry_features_ignore_liquid_level_drop():
    """液面下降时袋底与横向宽度不动（候选框是液体区域，框高会变——刻意不用它）。"""
    from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
    from infusion_monitor.core.predictor import Prompt
    from tests.support import render_rack_frame

    scene = default_rack_scene(16)
    high = np.asarray(render_rack_frame(scene, 0))
    low = np.asarray(render_rack_frame(scene, scene.frame_count - 1))
    point = (scene.bags[0].x_center * scene.width, scene.height * 0.55)

    descriptors = []
    for image in (high, low):
        prediction = GeometricMaskPredictor().predict(image, Prompt.from_point(*point))[0]
        descriptors.append(describe_region(image, prediction.mask, bag_seq="B01", slot_index=0))
    high_desc, low_desc = descriptors

    assert high_desc.centre_x == pytest.approx(low_desc.centre_x, abs=0.01)
    assert high_desc.bottom_y == pytest.approx(low_desc.bottom_y, abs=0.02)
    assert high_desc.width == pytest.approx(low_desc.width, abs=0.02)
    total, geometry, appearance = score_pair(low_desc, high_desc)
    assert appearance is not None and appearance > 0.9
    assert total > 0.9, "同一袋液面下降后不该被判成换袋"


def test_appearance_unavailable_is_never_auto_bound():
    """区域太小 → 外观不可用 → 一律 unbound（不拿几何分硬接）。"""
    thin = BagDescriptor(
        bag_seq="B01", slot_index=0, centre_x=0.3, bottom_y=0.7, width=0.3, hue_hist=()
    )
    previous = [_descriptor("B01", centre_x=0.3, hue_peak=3)]
    total, geometry, appearance = score_pair(thin, previous[0])
    assert appearance is None
    verdict = reanchor_slots([SlotObservation(0, "B01", thin)], previous)[0]
    assert verdict.anchor_state == "unbound"
    assert verdict.reason == "appearance_unavailable"


def test_first_round_vs_replaced_round_have_distinct_reasons():
    """首次巡视与「历史都换掉了」都不接上一轮，但给护士的提示不同。"""
    observation = SlotObservation(0, "B01", _descriptor("B01", centre_x=0.3, hue_peak=3))
    first = reanchor_slots([observation], [], previous_seen=False)[0]
    after_replace = reanchor_slots([observation], [], previous_seen=True)[0]
    assert (first.reason, after_replace.reason) == ("first_round", "previous_replaced")
    assert first.anchor_state == after_replace.anchor_state == "fresh"

def test_only_the_latest_round_per_bag_is_a_reanchor_candidate(store_client):
    """同一袋在两轮里各有一条会话时，只拿最近一条当候选——否则每轮都判「认不准」。

    这是实跑第三轮才发现的缺陷：上午、下午两条会话长得几乎一样，分差必然过小，
    重锚就永远接不上上一轮——重锚要回答的是「这个袋序现在是什么状态」。
    """
    client, conn = store_client
    first = _upload(client, default_rack_scene(16), shift_key="2026-10-22-AM", slots=TWO_BAG_SLOTS)
    assert first.status_code == 201
    second = _upload(client, _descending_scene(16), shift_key="2026-10-22-PM", slots=TWO_BAG_SLOTS)
    assert second.status_code == 201
    third = _upload(client, _descending_scene(16), shift_key="2026-10-22-NIGHT", slots=TWO_BAG_SLOTS)
    assert third.status_code == 201, third.text

    payload = third.json()
    for bag in payload["bags"]:
        assert bag["reanchor_state"] == "bound", bag
        assert bag["previous_shift_key"] == "2026-10-22-PM", "接的不是最近一轮"

    # 候选集合每个袋序只有一条
    history = SessionRepository(conn).list_previous_rounds(
        "R-EXAMPLE-09", "EXAMPLE-ORDER-0009", exclude_shift="2026-10-22-NIGHT"
    )
    assert len(history) == 2
    assert {row["bag_seq"] for row in history} == {"B01", "B02"}
