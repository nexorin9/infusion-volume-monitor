"""M2 多袋并行传播与槽位-轨迹分配：一架 ≤6 袋一次传播，传完必须回答「哪条曲线是哪只袋子」。

分两组断言：

**A. 多袋并行（真实传播，跑在合成渲染帧上）**

1. **一个状态、一遍传播**：6 只袋子共用一次 ``propagate_batch``，预测次数恒等于
   帧数 × 袋数（不是逐袋各跑一遍），逐帧 ``object_ids`` 顺序固定；
2. **按 id 连接，不按顺序连接**：把起锚顺序整个倒过来，每个 obj_id 的曲线逐点不变
   （按位置连接的话，倒一次顺序两条曲线就整体对调，而数值上看不出来）；
3. **身份集合冻结**：起锚之后往同一份状态里塞第 7 个 id 会被拒
   （对应源实现 ``object_ids_set`` 在首帧定死）；
4. **袋数上限**：超过 6 袋整段拒绝并说明原因，不静默截断；上限与入站校验同源。

**B. 槽位-轨迹分配（三个必答用例 + 纪律）**

5. **6 袋全部可见**：一一对应到各自挂点，全部 ``assigned``，无需认领；
6. **某袋完全遮挡**：全程看不见的那条**不给挂点也不占别人的挂点**（``no_evidence`` +
   该挂点报 ``unfilled``），另外几袋仍各自 ``assigned``——不被牵连、不被顶替；
7. **袋序错位（插空位）**：声明的挂点空着 / 袋子与声明对不上 → 报待认领
   （``below_score`` / ``mismatched``），**不自动改绑**；
8. **不猜**：位置有争议时判 ``ambiguous``，不默认取分数最高的那个；一条挂点最多接一条轨迹。

画面是**合成渲染帧**（``tests/support/synth.py``），不是真实院内短片——真实短片需医院
授权采集，本轮以合成 + 公开 VOS 子集（任务 21/22）替代。
"""

from __future__ import annotations

import pytest

from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
from infusion_monitor.core.multi_object import (
    AMBIGUOUS,
    ASSIGNED,
    BELOW_SCORE,
    EXTRA_TRACK,
    MAX_BAGS_PER_RACK,
    MISMATCHED,
    NO_EVIDENCE,
    REASON_ASSIGNED,
    REASON_DECLARED_SLOT_MISMATCH,
    REASON_NEVER_SEEN,
    REASON_SLOT_EMPTY,
    SlotAssignmentParams,
    TrackAnchor,
    TrackView,
    assign_slots,
    propagate_batch,
    slot_columns,
)
from infusion_monitor.core.propagate import (
    BagCurve,
    ObjectIdSetFrozen,
    add_new_points_or_box,
    analyse_frames,
    init_state,
    slot_seed_prompt,
)
from infusion_monitor.ingest.validate import MAX_SLOTS
from tests.support.synth import BagSpec, RackScene, write_rack_frames

WIDTH, HEIGHT = 480, 640

# 袋子的液体色：必须落在几何核的饱和度带内（saturation ≥ 96/255），否则整袋根本切不出来
# ——那是取样问题，不是分配问题。踩过一次：淡紫 (130,130,190) 的 S≈80 被整袋漏掉，
# 一度误判成「最右边那只袋子认不出来」。_palette_is_in_band 把这个坑钉成断言。
_COLOURS = (
    (40, 120, 200),
    (55, 120, 200),
    (70, 120, 200),
    (85, 120, 200),
    (100, 120, 200),
    (115, 120, 200),
)


def _palette_is_in_band() -> None:
    """所有取样颜色都在几何核的饱和度带内（B 通道取最大值时的标准 HSV 饱和度）。"""
    for red, _green, blue in _COLOURS:
        saturation = (max(red, blue) - min(red, blue)) / max(1, blue)
        assert saturation * 255 >= 96, f"取样色 {(red, _green, blue)} 落在饱和度带外"


def rack(count: int, *, width_ratio: float = 0.13, frame_count: int = 16) -> RackScene:
    """等距挂 ``count`` 只袋子的场景（第 i 只在第 i 列的列心附近）。"""
    _palette_is_in_band()
    return RackScene(
        bags=tuple(
            BagSpec(
                f"B{i + 1:02d}",
                (2 * i + 1) / (2.0 * count),
                width_ratio,
                0.85 - 0.02 * i,
                0.80 - 0.02 * i,
                _COLOURS[i % len(_COLOURS)],
            )
            for i in range(count)
        ),
        frame_count=frame_count,
    )


@pytest.fixture(scope="module")
def predictor() -> GeometricMaskPredictor:
    return GeometricMaskPredictor()


def run(
    tmp_path,
    scene: RackScene,
    predictor: GeometricMaskPredictor,
    *,
    declared_slots: list[int] | None = None,
    slot_count: int | None = None,
    render: dict | None = None,
):
    """渲染 → 一次并行传播 → 槽位分配。返回 (batch, result)。"""
    paths = write_rack_frames(tmp_path, scene, **(render or {}))
    count = slot_count or len(declared_slots or scene.bags)
    declared = declared_slots if declared_slots is not None else list(range(len(scene.bags)))
    seqs = [f"B{index + 1:02d}" for index in range(len(declared))]
    from infusion_monitor.core.propagate import load_rgb

    first = load_rgb(paths[0])
    anchors = [
        TrackAnchor(seq, slot_seed_prompt(first.shape, slot, count), slot)
        for seq, slot in zip(seqs, declared)
    ]
    batch = propagate_batch(paths, anchors, predictor)
    columns = slot_columns(count, WIDTH, seqs)
    return batch, assign_slots(batch.views(), columns, frame_shape=batch.frame_shape)


# ── A. 多袋并行 ───────────────────────────────────────────────────────────


def test_six_bags_are_assigned_one_to_one(tmp_path, predictor):
    """用例 1：6 袋全部可见一一对应，每袋一条互不相同的曲线，且都无需认领。"""
    batch, result = run(tmp_path, rack(6), predictor)

    assert result.summary()["track_count"] == 6
    assert result.summary()["by_state"] == {ASSIGNED: 6}
    assert result.summary()["claim_required"] == 0
    assert [item.slot_index for item in result.assignments] == [0, 1, 2, 3, 4, 5]
    assert result.claim_rows() == []
    # 每袋一条曲线，且各袋读数互不相同（没有两袋串成同一条）
    for curve in batch.ordered:
        assert curve.visible_frame_count == 16
        assert curve.occluded_frame_count == 0
    last_values = [curve.latest_ratio for curve in batch.ordered]
    assert len(set(last_values)) == 6, "六条曲线读数不该相同（说明有串袋）"


def test_six_bags_propagate_in_one_pass(tmp_path, predictor):
    """6 只袋子共用一次传播：预测次数 = 帧数 × 袋数，逐帧对象集合顺序固定。"""
    batch, _ = run(tmp_path, rack(6), predictor)

    assert len(batch.frames) == 16
    assert batch.predict_calls == 16 * 6
    assert batch.anchor_ids == frozenset(f"B{i:02d}" for i in range(1, 7))
    for frame in batch.frames:
        # 顺序按起锚顺序固定，且逐帧一致——掩膜按 id 归位，不靠遍历顺序
        assert frame.object_ids == batch.ordered_ids
        assert set(frame.areas_px) == set(batch.ordered_ids)


def test_masks_are_joined_by_id_not_by_visit_order(tmp_path, predictor):
    """起锚顺序整个倒过来，每个 obj_id 的曲线逐点不变。

    按位置连接的话，倒一次顺序两条曲线就整体对调，而数值上完全看不出来。
    """
    scene = rack(3)
    forward, _ = run(tmp_path, scene, predictor)

    paths = write_rack_frames(tmp_path / "reversed", scene)
    from infusion_monitor.core.propagate import load_rgb

    first = load_rgb(paths[0])
    order = [2, 0, 1]
    reversed_batch = propagate_batch(
        paths,
        [
            TrackAnchor(f"B{index + 1:02d}", slot_seed_prompt(first.shape, index, 3), index)
            for index in order
        ],
        predictor,
    )
    assert reversed_batch.ordered_ids == tuple(f"B{index + 1:02d}" for index in order)
    for obj_id in forward.ordered_ids:
        assert reversed_batch.curves[obj_id].area_ratio_series == (
            forward.curves[obj_id].area_ratio_series
        ), f"{obj_id} 的曲线随起锚顺序变了——掩膜是按位置而不是按 id 归位的"


def test_object_id_set_is_frozen_after_anchoring(tmp_path, predictor):
    """起锚之后塞新 id 会被拒（对应源实现 object_ids_set 在首帧定死）。"""
    scene = rack(2)
    paths = write_rack_frames(tmp_path, scene)
    from infusion_monitor.core.propagate import load_rgb

    first = load_rgb(paths[0])
    state = init_state([path.name for path in paths], first.shape)
    add_new_points_or_box(
        state,
        frame_index=0,
        obj_id="B01",
        prompt=slot_seed_prompt(first.shape, 0, 2),
        cavity_area_px=1000,
        slot_index=0,
    )
    state.frozen_object_ids = frozenset(state.objects)

    with pytest.raises(ObjectIdSetFrozen) as excinfo:
        add_new_points_or_box(
            state,
            frame_index=0,
            obj_id="B09",
            prompt=slot_seed_prompt(first.shape, 1, 2),
            cavity_area_px=1000,
            slot_index=1,
        )
    assert "B09" in str(excinfo.value)
    # 已冻结的 id 再加一次（改绑同一只袋子的提示）仍然允许
    add_new_points_or_box(
        state,
        frame_index=0,
        obj_id="B01",
        prompt=slot_seed_prompt(first.shape, 0, 2),
        cavity_area_px=2000,
        slot_index=0,
    )
    assert state.objects["B01"].cavity_area_px == 2000


def test_more_than_six_bags_is_rejected_with_a_reason(tmp_path, predictor):
    """超过 6 袋整段拒绝并说明原因（不静默截断）；上限与入站校验同源。"""
    assert MAX_BAGS_PER_RACK == MAX_SLOTS == 6

    paths = write_rack_frames(tmp_path, rack(7))
    from infusion_monitor.core.propagate import load_rgb

    first = load_rgb(paths[0])
    anchors = [
        TrackAnchor(f"B{index + 1:02d}", slot_seed_prompt(first.shape, index, 7), index)
        for index in range(7)
    ]
    with pytest.raises(ValueError) as excinfo:
        propagate_batch(paths, anchors, predictor)
    message = str(excinfo.value)
    assert "6" in message and "拆成两次上传" in message


def test_duplicate_object_ids_are_rejected(tmp_path, predictor):
    paths = write_rack_frames(tmp_path, rack(2))
    from infusion_monitor.core.propagate import load_rgb

    first = load_rgb(paths[0])
    anchors = [
        TrackAnchor("B01", slot_seed_prompt(first.shape, 0, 2), 0),
        TrackAnchor("B01", slot_seed_prompt(first.shape, 1, 2), 1),
    ]
    with pytest.raises(ValueError, match="互不相同"):
        propagate_batch(paths, anchors, predictor)


def test_curve_carries_per_frame_geometry_but_no_masks(tmp_path, predictor):
    """曲线带逐帧质心/面积（分配层要用），但不因此攒下掩膜（恒定算力边界不动）。"""
    batch, _ = run(tmp_path, rack(3), predictor)

    for curve in batch.ordered:
        assert len(curve.frame_centroids) == len(curve.area_ratio_series) == 16
        assert len(curve.frame_areas_px) == 16
        assert all(value is not None for value in curve.frame_centroids)
        # 逐帧只留质心与面积两个标量，不得攒下整段掩膜（恒定算力的边界不动）
        fields = set(BagCurve.__dataclass_fields__)
        assert "frame_centroids" in fields and "frame_areas_px" in fields
        assert not any(
            field.endswith("masks") or field.endswith("mask") for field in fields
        ), "曲线不得攒整段掩膜（恒定算力）"


def test_analyse_frames_is_the_same_implementation_as_propagate_batch(tmp_path, predictor):
    """analyse_frames 就是 propagate_batch 的薄包装：两个入口数值不得分叉。"""
    scene = rack(3)
    paths = write_rack_frames(tmp_path, scene)
    from infusion_monitor.core.propagate import load_rgb

    first = load_rgb(paths[0])
    seeds = [(f"B{index + 1:02d}", slot_seed_prompt(first.shape, index, 3)) for index in range(3)]

    legacy = analyse_frames(paths, seeds, predictor)
    batch = propagate_batch(paths, seeds, predictor)
    assert [curve.obj_id for curve in legacy] == list(batch.ordered_ids)
    for left, right in zip(legacy, batch.ordered):
        assert left.area_ratio_series == right.area_ratio_series
        assert left.frame_centroids == right.frame_centroids


# ── B. 槽位-轨迹分配 ──────────────────────────────────────────────────────


def test_fully_occluded_bag_is_not_assigned_and_does_not_take_another_slot(tmp_path, predictor):
    """用例 2：中间那袋全程被挡板盖住——不给挂点、不占别人的挂点，其余各归各位。"""
    scene = rack(3, width_ratio=0.20)
    batch, result = run(
        tmp_path,
        scene,
        predictor,
        render={"occluders": [(1, 0, scene.frame_count)]},
    )

    occluded = result.by_obj("B02")
    assert occluded.state == NO_EVIDENCE
    assert occluded.reason == REASON_NEVER_SEEN
    assert occluded.slot_index is None
    assert occluded.claim_required is True

    # 其余两袋仍各自配到自己声明的挂点，没被顶替
    assert result.by_obj("B01").slot_index == 0
    assert result.by_obj("B03").slot_index == 2
    assert {item.obj_id for item in result.assignments} == {"B01", "B02", "B03"}
    # 那一挂点单独报「没拍到」，而不是被别袋的曲线悄悄顶掉
    assert result.by_slot(1).state != ASSIGNED
    assert result.by_slot(1).reason == REASON_SLOT_EMPTY
    assert result.by_slot(0).obj_id == "B01"
    assert result.by_slot(2).obj_id == "B03"
    assert [row["bag_seq"] for row in result.claim_rows()] == ["B02"]
    # 曲线本身也整段留空，不插值
    assert batch.curves["B02"].area_ratio_series == [None] * scene.frame_count


def test_occluded_bag_does_not_degrade_other_bags_to_fake_readings(tmp_path, predictor):
    """一只袋子没有袋腔估计，不得把同一架其余袋子的余量一起变成假读数。

    早期实现是「任意一袋估不出袋腔 → 整架 area_ratio 退化成 1.0」，于是一份遮挡会把
    另外两只袋子的毫升数一起变成假的（1.0 = 满袋），这正是「因为缺一条证据而系统性
    拖累其余」的老毛病。
    """
    scene = rack(3, width_ratio=0.20)
    batch, _ = run(
        tmp_path, scene, predictor, render={"occluders": [(1, 0, scene.frame_count)]}
    )

    for obj_id in ("B01", "B03"):
        ratios = [value for value in batch.curves[obj_id].area_ratio_series if value is not None]
        assert ratios, f"{obj_id} 整段没有读数"
        assert max(ratios) < 0.95, f"{obj_id} 的读数退化成满袋（袋腔估计失效牵连了全架）"
        # 真实走向是下降的；逐帧 ±0.005 的几何噪声不算「上升」，
        # 真正的上升段异常与插值由任务 19 的平滑/单调性层负责判，不在本任务里断言
        assert ratios[-1] < ratios[0] - 0.005
        assert all(
            later <= earlier + 0.01 for earlier, later in zip(ratios, ratios[1:])
        ), "余量出现了明显上升段"


def test_empty_hook_is_reported_and_needs_claim_instead_of_guessing(tmp_path, predictor):
    """用例 3（上）：声明了 3 个挂点，实际只挂 0 与 2 列——空位报 unfilled，认领不猜。"""
    scene = RackScene(
        bags=(
            BagSpec("B01", 1 / 6.0, 0.20, 0.85, 0.80, _COLOURS[0]),
            BagSpec("B02", 5 / 6.0, 0.20, 0.65, 0.60, _COLOURS[2]),
        ),
        frame_count=16,
    )
    _, result = run(tmp_path, scene, predictor, declared_slots=[0, 1, 2])

    assert result.by_obj("B01").slot_index == 0
    empty = result.by_obj("B02")
    assert empty.claim_required is True
    assert empty.state in {NO_EVIDENCE, BELOW_SCORE}
    assert result.by_slot(1).reason == REASON_SLOT_EMPTY
    assert result.by_slot(1).claim_required is True
    # 空位没有被别袋占走
    assert result.by_slot(0).obj_id == "B01"


def test_slot_columns_geometry_matches_the_seed_prompt_prior():
    """分配用的竖列先验与起锚用的 slot_seed_prompt 必须同一套几何。

    两处若各算一套，「起锚按 A 先验、分配按 B 先验」就会自己跟自己打架。
    """
    for count in (1, 2, 3, 6):
        columns = slot_columns(count, WIDTH)
        assert len(columns) == count
        for column in columns:
            seed = slot_seed_prompt((HEIGHT, WIDTH), column.slot_index, count)
            box = seed.box
            assert box is not None
            assert box[0] == pytest.approx(column.centre_x - column.half_width, abs=1e-6)
            assert box[2] == pytest.approx(column.centre_x + column.half_width, abs=1e-6)


# ── 分配纪律（纯函数层：直接构造 TrackView，钉住语义而不是钉住某个场景） ────


def _view(
    obj_id: str,
    declared: int | None,
    centre_x: float,
    *,
    area: float = 22000.0,
    seen: int = 14,
    never_seen: bool = False,
) -> TrackView:
    return TrackView(
        obj_id=obj_id,
        declared_slot=declared,
        seen_frames=seen,
        centroid=None if never_seen else (centre_x, 400.0),
        anchor_centroid=None if never_seen else (centre_x, 400.0),
        mean_area_px=0.0 if never_seen else area,
        area_ratio=None if never_seen else 0.7,
        never_seen=never_seen,
    )


def _assign(views, count: int = 4, params: SlotAssignmentParams | None = None):
    return assign_slots(
        views,
        slot_columns(count, WIDTH, [f"B{index + 1:02d}" for index in range(count)]),
        frame_shape=(HEIGHT, WIDTH),
        params=params,
    )


def test_track_in_its_own_column_is_assigned():
    result = _assign([_view("B01", 0, 0.125 * WIDTH), _view("B02", 1, 0.375 * WIDTH)])

    first = result.by_obj("B01")
    assert first.state == ASSIGNED
    assert first.reason == REASON_ASSIGNED
    assert first.slot_index == 0
    assert first.claim_required is False
    assert first.detail  # 护士看得懂的一句话
    assert result.summary()["claim_required"] == 0


def test_contested_hook_is_ambiguous_rather_than_best_score():
    """两条轨迹在同一列上分不出先后 → 判歧义，**不**默认取分数最高的那个。"""
    result = _assign([_view("B01", 0, 61.4, area=22000), _view("B02", 0, 63.4, area=22400)])

    assert {item.state for item in result.assignments} == {AMBIGUOUS}
    # 谁都没拿到挂点：不给「看起来最像」的那只开后门
    assert all(item.slot_index is None for item in result.assignments)
    assert result.by_slot(0).state == AMBIGUOUS
    assert result.summary()["claim_required"] == 2


def test_near_tied_tracks_on_a_boundary_are_ambiguous():
    """两条轨迹都贴着同一条列边界、分差不足 → 歧义，不配对。"""
    result = _assign([_view("B02", 1, 155.0), _view("B03", 2, 205.0)])

    assert {item.state for item in result.assignments} == {AMBIGUOUS}
    assert result.by_slot(1).claim_required is True
    assert result.by_slot(1).obj_id in {"B02", "B03"}


def test_declared_slot_mismatch_is_flagged_not_silently_swapped():
    """袋子实际在第 4 列、输液单记在第 2 个挂点 → 报错位交认领，**不自动改绑**。"""
    result = _assign(
        [_view("B02", 1, 0.875 * WIDTH, area=21000), _view("B03", 3, 0.125 * WIDTH, area=23000)]
    )

    first = result.by_obj("B02")
    assert first.state == MISMATCHED
    assert first.reason == REASON_DECLARED_SLOT_MISMATCH
    assert first.declared_slot == 1
    assert first.slot_index == 3  # 落在哪一列如实报出来，但状态是待认领
    assert first.claim_required is True
    assert "第 2 个挂点" in first.detail and "第 4 个挂点" in first.detail


def test_track_without_declared_slot_is_an_extra_track():
    """输注单上没有的那只袋子（邻架/点滴瓶/杂物）→ extra_track，不算「错位」。"""
    result = _assign([_view("B01", 0, 0.125 * WIDTH), _view("B99", None, 0.90 * WIDTH, area=23000)])

    extra = result.by_obj("B99")
    assert extra.state == EXTRA_TRACK
    assert extra.claim_required is True
    assert result.by_obj("B01").state == ASSIGNED


def test_merged_blob_area_is_rejected_by_the_area_gate():
    """面积离群 4 倍（两袋并成一块）→ 面积硬门限直接淘汰，不靠加权抬过门槛。"""
    result = _assign(
        [
            _view("B01", 0, 0.125 * WIDTH, area=22000),
            _view("B02", 1, 0.375 * WIDTH, area=22000),
            _view("B03", 2, 0.625 * WIDTH, area=90000),
        ]
    )

    assert result.by_obj("B03").state == BELOW_SCORE
    assert result.by_obj("B03").slot_index is None
    assert result.by_obj("B03").claim_required is True
    assert result.by_obj("B01").state == ASSIGNED
    assert result.by_obj("B02").state == ASSIGNED
    # 放宽硬门限后它就能过——说明确实是面积通道在拦，不是别的
    relaxed = _assign(
        [
            _view("B01", 0, 0.125 * WIDTH, area=22000),
            _view("B02", 1, 0.375 * WIDTH, area=22000),
            _view("B03", 2, 0.625 * WIDTH, area=90000),
        ],
        params=SlotAssignmentParams(max_area_deviation=10.0),
    )
    assert relaxed.by_obj("B03").state == ASSIGNED


def test_one_hook_holds_at_most_one_track():
    """全局一对一：任何挂点最多被一条轨迹占用（响应里不会出现两个袋次指向同一挂点）。"""
    views = [_view(f"B{index:02d}", 0, 0.125 * WIDTH) for index in range(1, 4)]
    result = _assign(views)

    occupied = [item.slot_index for item in result.assignments if item.slot_index is not None]
    assert len(occupied) == len(set(occupied))
    assert len(occupied) <= 1
    for slot_index, item in result.slot_states.items():
        holders = [
            obj for obj in result.assignments if obj.slot_index == slot_index and obj.assigned
        ]
        assert len(holders) <= 1
        if item.state == ASSIGNED:
            assert item.obj_id in holders


def test_never_seen_track_gets_no_evidence_and_claims_no_hook():
    views = [_view("B01", 0, 0.125 * WIDTH), _view("B02", 1, 0.375 * WIDTH, never_seen=True)]
    result = _assign(views)

    blind = result.by_obj("B02")
    assert blind.state == NO_EVIDENCE
    assert blind.slot_index is None
    assert blind.evidence["seen_frames"] == 0
    assert blind.claim_required is True
    assert result.by_obj("B01").state == ASSIGNED


def test_claim_rows_only_list_what_needs_claiming():
    result = _assign(
        [_view("B01", 0, 0.125 * WIDTH), _view("B02", 1, 0.375 * WIDTH, never_seen=True)]
    )

    rows = result.claim_rows()
    assert [row["bag_seq"] for row in rows] == ["B02"]
    assert rows[0]["state"] == NO_EVIDENCE
    assert rows[0]["track_id"] == "B02"
    assert rows[0]["slot_index"] == 1
    assert rows[0]["detail"]
    # 汇总与逐条一致
    assert result.summary()["claim_required"] == len(result.claimable) == len(rows)
    assert result.to_dict()["summary"] == result.summary()


def test_every_slot_reports_a_state():
    """每个挂点都要有结论（空位也要说出来），不能只报有袋子的那些。"""
    result = _assign([_view("B01", 0, 0.125 * WIDTH)])

    assert set(result.slot_states) == {0, 1, 2, 3}
    assert [result.by_slot(index).reason for index in (1, 2, 3)] == [REASON_SLOT_EMPTY] * 3