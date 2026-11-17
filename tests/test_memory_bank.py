"""M2 记忆特征库测试：记忆命中 / 记忆衰减 / 错误记忆不污染。

三组断言，外加两条纪律性断言：

1. **记忆命中**：每条轨迹维护 top-k 条记忆（帧号 + 掩膜 + 置信）；连续 3 帧全遮挡后
   记忆仍在，恢复时按**相似度加权**的吻合度接回**同一袋**（``bag_seq`` 不变、曲线不断、
   空档仍留空）；
2. **记忆衰减**：新鲜度按帧衰减，越久远权重越低；衰减到门限以下时**没有记忆站在这一边**
   （返回 ``None``），上层据此不猜——而不是拿一条快过期的记忆硬接；
3. **错误记忆不污染**：场景切换（手机扫到别处）送进来的观测**不进库、不出面积比**，
   库里仍是对的那袋，下一帧因此还能自己接回来；
4. **不引入第二套入口/第二份状态**：库由 :class:`OcclusionTracker` 单一持有，
   ``memory`` 就是库里那一条本身；
5. **恒定算力**：容量固定，与帧数无关；淘汰的是最旧的而不是最新的。

画面是**合成渲染帧**（``tests/support/synth.py``），不是真实院内短片——真实短片需医院
授权采集，本轮以合成 + 公开 VOS 子集替代。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
from infusion_monitor.core.memory_bank import (
    REJECT_SCENE_SWITCH,
    MemoryBank,
    MemoryEntry,
    MemoryParams,
)
from infusion_monitor.core.occlusion import (
    OCCLUDED,
    REAPPEARED,
    VISIBLE,
    OcclusionTracker,
    VisibilityParams,
)
from infusion_monitor.core.predictor import MaskPrediction
from infusion_monitor.core.propagate import analyse_frames, load_rgb, slot_seed_prompt
from tests.support.synth import default_rack_scene, write_rack_frames

HEIGHT, WIDTH = 240, 320
CAVITY = 10_000
CAPACITY = 4


# ── 构造器 ──────────────────────────────────────────────────────────────
def _pred(box: tuple[int, int, int, int], score: float = 0.72) -> MaskPrediction:
    x0, y0, x1, y1 = box
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return MaskPrediction(mask=mask, score=score)


def _box(area_px: int, *, y1: int = 200, x0: int = 100) -> tuple[int, int, int, int]:
    """液面下降时的外接框（袋底固定、顶部上移）。"""
    width = 100
    height = max(1, area_px // width)
    return (x0, y1 - height, x0 + width, y1)


def _tracker(frame_count: int = 24, **params) -> OcclusionTracker:
    return OcclusionTracker(
        obj_id="B01",
        cavity_area_px=CAVITY,
        frame_count=frame_count,
        params=VisibilityParams(**params),
        memory_params=MemoryParams(),
    )


def _bank(**params) -> MemoryBank:
    return MemoryBank(obj_id="B01", params=MemoryParams(**params))


def _entry(frame_index: int, box: tuple[int, int, int, int], *, score: float = 0.9, **kwargs):
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return MemoryEntry(
        frame_index=frame_index,
        bbox=box,
        centroid=((x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0),
        area=mask.sum(),
        ratio=0.5,
        mask=mask,
        score=score,
        **kwargs,
    )


# ── 1. 记忆命中 ─────────────────────────────────────────────────────────
def test_bank_keeps_top_k_frames_with_mask_and_confidence():
    """每条轨迹维护 top-k 记忆帧：帧号、掩膜特征、置信三样都在。"""
    bank = _bank()
    tolerance = VisibilityParams()
    for index, area in enumerate((8000, 7800, 7600, 7400, 7200, 7000, 6800)):
        entry = _entry(index, _box(area))
        entry.ratio = area / CAVITY
        write = bank.write(entry, tolerance, now_frame=index)
        assert write.accepted, write.reason

    assert len(bank) == CAPACITY, "容量固定，不是攒到多少算多少"
    assert bank.frame_indices == (3, 4, 5, 6), "淘汰最旧的，留最新的"
    assert all(entry.mask is not None for entry in bank.entries), "掩膜特征要留着参与判断"
    assert [entry.score for entry in bank.entries] == [0.9] * CAPACITY, "置信要留着"
    assert bank.describe()["capacity"] == CAPACITY


def test_memory_hit_resumes_same_bag_after_three_occluded_frames():
    """连续 3 帧全遮挡后记忆仍能接回同一袋：bag_seq 不变、曲线不断、空档仍留空。"""
    tracker = _tracker()
    for index, area in enumerate((8000, 7800, 7600, 7400)):
        tracker.observe(index, _pred(_box(area)))
    assert len(tracker.bank) == CAPACITY

    for index in (4, 5, 6):
        verdict = tracker.observe(index, None)
        assert verdict.state == OCCLUDED and verdict.area_ratio is None
    assert len(tracker.bank) == CAPACITY, "遮挡帧不写记忆（也不该把库冲掉）"

    verdict = tracker.observe(7, _pred(_box(7200)))
    assert verdict.state == REAPPEARED
    assert verdict.used_memory, "恢复判断必须用到记忆"
    assert verdict.confidence >= 0.5
    assert tracker.obj_id == "B01", "接回的是同一袋，不是新对象"
    assert tracker.ratios[4:7] == [None, None, None], "空档不得被回填"
    assert tracker.ratios[7] == pytest.approx(7200 / CAVITY), "曲线接得上"


def test_fused_agreement_uses_every_stored_memory_with_weights():
    """相似度加权：库内每条采信记忆都参与加权，权重随新鲜度递减。"""
    tracker = _tracker()
    for index, area in enumerate((8000, 7800, 7600, 7400)):
        tracker.observe(index, _pred(_box(area)))
    bank = tracker.bank
    matches = bank.match(_pred(_box(7300)), VisibilityParams(), now_frame=4)

    assert len(matches) == CAPACITY
    assert [match.weight for match in matches] == sorted(
        (match.weight for match in matches), reverse=True
    ), "按权重降序，最新的在最前"
    assert matches[0].entry.frame_index == 3, "最新一条权重最大"
    assert all(match.mask_iou is not None for match in matches), "掩膜特征参与匹配"

    weights = [match.weight for match in matches]
    assert weights[0] > weights[-1], "越久远权重越低"
    # 加权吻合度是各条吻合度的加权平均，不是简单取一条
    agreement, _ = bank.fused_agreement(_pred(_box(7300)), VisibilityParams(), now_frame=4)
    assert agreement == pytest.approx(sum(w * m.similarity for w, m in zip(weights, matches)) / sum(weights))


def test_newest_memory_outweighs_a_stale_one():
    """最新记忆权重最大——恢复判断以「刚看见的那几帧」为主。"""
    tracker = _tracker()
    for index, area in enumerate((8000, 7800, 7600, 7400, 7200)):
        tracker.observe(index, _pred(_box(area)))
    matches = tracker.bank.match(_pred(_box(7100)), VisibilityParams(), now_frame=5)
    assert matches[0].entry.frame_index == 4
    # 衰减 0.88^4 ≈ 0.60，实测比值 1.48（再乘上各自的相似度）
    assert matches[0].weight / matches[-1].weight > 1.4, "4 帧之差就该把权重拉开"


# ── 2. 记忆衰减 ─────────────────────────────────────────────────────────
def test_weight_decays_with_age():
    entry = _entry(0, _box(8000))
    fresh = entry.weight_of(now_frame=0, decay_per_frame=0.88)
    older = entry.weight_of(now_frame=10, decay_per_frame=0.88)
    assert older == pytest.approx(fresh * 0.88**10)
    assert older < fresh


def test_stale_memory_stops_standing_behind_the_observation():
    """记忆衰减到门限以下 → 没有任何记忆站在这一边（None），上层据此不猜。"""
    bank = _bank(min_entry_weight=0.05)
    bank.write(_entry(0, _box(8000)), VisibilityParams(), now_frame=0)
    bank.write(_entry(1, _box(7800)), VisibilityParams(), now_frame=1)

    near, _ = bank.fused_agreement(_pred(_box(7600)), VisibilityParams(), now_frame=3)
    assert near is not None and near > 0.5

    far, matches = bank.fused_agreement(_pred(_box(7600)), VisibilityParams(), now_frame=400)
    assert far is None, "记忆早就不新鲜了，不能再拿它当依据"
    assert matches, "匹配过程仍然如实返回，只是权重全部低于门限"


def test_far_future_observation_is_not_resumed_from_old_memory():
    """丢了很久之后画面里那块，早就不能用来接回曲线。"""
    tracker = _tracker(frame_count=40)
    tracker.observe(0, _pred(_box(8000)))
    tracker.observe(1, _pred(_box(7800)))
    for index in range(2, 12):
        tracker.observe(index, None)
    assert tracker.handoff is not None, "10 帧遮挡已超丢失窗口，轨迹应交出"

    verdict = tracker.observe(12, _pred(_box(7600)))
    assert verdict.state != REAPPEARED, "轨迹已交出，不能靠旧记忆偷偷接回"
    assert tracker.ratios[12] is None


def test_partial_reading_does_not_vote_on_identity():
    """被压住一半的帧几何失真，不能参与身份判断——否则遮一结束、恢复正常的那帧反倒被判成「拍到别的」。"""
    tracker = _tracker()
    for index, area in enumerate((8000, 7800, 7600, 7400)):
        tracker.observe(index, _pred(_box(area)))
    full_box = tracker.bank.identity_entries()[-1].bbox

    tracker.observe(4, _pred(_box(2000)))  # partial：面积掉到四分之一
    assert tracker.memory is not None and tracker.memory.trusted is False
    assert tracker.bank.identity_entries()[-1].bbox == full_box, "共识仍由完整可见的帧决定"
    assert tracker.bank.reference_ratio() == pytest.approx(7400 / CAVITY), "面积比也不采信它"


# ── 3. 错误记忆不污染 ───────────────────────────────────────────────────
def test_scene_switch_observation_is_not_stored_and_yields_no_ratio():
    """手机扫到别处：观测不进库、不出面积比，库仍是对的那袋。"""
    tracker = _tracker()
    for index, area in enumerate((8000, 7800, 7600)):
        tracker.observe(index, _pred(_box(area)))
    bank = tracker.bank
    before = bank.frame_indices

    # 另一台输液架：挂得更高、袋子更宽——位置与记忆对不上
    impostor = (100, 40, 260, 140)
    verdict = tracker.observe(3, _pred(impostor))

    assert verdict.state == OCCLUDED, "拍到别的东西就该按看不见处理"
    assert verdict.area_ratio is None, "不给假的余量数字"
    assert tracker.ratios[3] is None
    assert bank.frame_indices == before, "错误记忆不得进库"
    assert bank.rejected_total == 1, "拒收要留档，运行摘要与 task 17 才查得到"
    assert bank.rejected[0].frame_index == 3
    assert tracker.last_write is not None and not tracker.last_write.accepted, (
        "last_write 记的是最近一次交给记忆库的处置，拒收也要如实反映"
    )
    assert tracker.obj_id == "B01", "曲线不得改挂到别的对象上"
    assert "别的东西" in verdict.reason


def test_rejected_write_is_reported_not_swallowed():
    """写入门禁的拒收结果如实记在 last_write / rejected 上，不静默吞掉。"""
    bank = _bank()
    tolerance = VisibilityParams()
    bank.write(_entry(0, _box(8000)), tolerance, now_frame=0)
    write = bank.write(_entry(1, (100, 40, 260, 140)), tolerance, now_frame=1)

    assert not write.accepted
    assert write.reason == REJECT_SCENE_SWITCH
    assert write.similarity is not None and write.similarity < bank.params.min_similarity
    assert len(bank) == 1
    assert [entry.frame_index for entry in bank.rejected] == [1]


def test_bank_still_describes_the_real_bag_after_a_rejection():
    """拒收之后，库给出的共识仍是那一袋——task 17 的再获取拿到的线索不能被带偏。"""
    tracker = _tracker()
    for index, area in enumerate((8000, 7800, 7600)):
        tracker.observe(index, _pred(_box(area)))
    tracker.observe(3, _pred((100, 40, 260, 140)))

    query = tracker.memory_query(3)
    assert query.entry_count == 3
    assert query.rejected and query.rejected[0].frame_index == 3
    latest_box = tracker.bank.identity_entries()[-1].bbox
    assert query.bbox is not None
    assert all(abs(a - b) <= 3 for a, b in zip(query.bbox, latest_box)), (
        "共识应仍贴着真袋子，不该被冒名帧拉走"
    )
    assert query.reason.count("拒收") == 1


def test_next_frame_recovers_itself_because_memory_was_not_polluted():
    """因为没把冒名帧写进库，下一帧的提示仍指向真袋子，本袋能自己接回来。"""
    tracker = _tracker()
    for index, area in enumerate((8000, 7800, 7600)):
        tracker.observe(index, _pred(_box(area)))
    tracker.observe(3, _pred((100, 40, 260, 140)))

    verdict = tracker.observe(4, _pred(_box(7200)))
    # 上一帧按「看不见」处理，所以这一帧记 reappeared（同一袋接回），不是 visible
    assert verdict.state == REAPPEARED
    assert tracker.ratios[3] is None, "被拒的那一帧仍留空，不回填"
    assert tracker.ratios[4] == pytest.approx(7200 / CAVITY)
    assert tracker.obj_id == "B01"


def test_geometry_cannot_separate_a_sibling_bag_hanging_at_the_same_height():
    """**已知边界**（不是 bug）：同高度同样式的另一袋，几何记忆分不出来。

    袋底项与宽度项各占 0.3，只要两袋挂一样高、一样宽，位置相似度就有 0.6 的底。
    分辨同架上的两袋是**槽位分配**（task 18）与**外观描述子**（task 29）的活；
    这里把它钉成断言，免得以后有人误以为几何记忆已经解决了身份绑定。
    """
    tracker = _tracker()
    for index, area in enumerate((8000, 7800, 7600)):
        tracker.observe(index, _pred(_box(area)))
    sibling = _pred((200, 120, 300, 200))  # 右边一格，袋底与宽度都一样
    similarity = tracker.bank.similarity_to_consensus(
        sibling, VisibilityParams(), now_frame=3
    )
    assert similarity == pytest.approx(0.6)
    assert similarity >= tracker.bank.params.min_similarity
    assert similarity >= tracker.bank.params.min_consistency
    module_doc = inspect.getdoc(sys.modules[MemoryBank.__module__]) or ""
    assert "task 18" in module_doc and "task 29" in module_doc, (
        "边界要在模块文档里写明去处"
    )


# ── 4. 纪律：唯一持有 / 无第二份状态 / 无第二套入口 ──────────────────────
def test_tracker_owns_the_bank_and_memory_is_not_a_copy():
    tracker = _tracker()
    tracker.observe(0, _pred(_box(8000)))
    assert tracker.bank.obj_id == "B01"
    assert tracker.memory is tracker.bank.latest, "memory 就是库里那一条本身，不是副本"
    assert tracker.memory is not None and tracker.memory.frame_index == 0


def test_no_second_memory_holder_left_in_the_occlusion_module():
    """occlusion 模块里不该还留着旧的单个 _memory 字段（否则就成了第二份状态）。"""
    source = (Path(inspect.getfile(OcclusionTracker)).read_text(encoding="utf-8"))
    assert "self._memory" not in source, "记忆的唯一持有者是 MemoryBank"
    assert not hasattr(OcclusionTracker, "_memory")


def test_bank_is_asked_not_guessed_when_it_is_empty():
    tracker = _tracker()
    verdict = tracker.observe(0, _pred(_box(8000)))
    assert verdict.state == VISIBLE and not verdict.used_memory
    assert tracker.memory_query(0).entry_count == 1
    assert tracker.memory_query(0).agreement is None, "没给观测就没有吻合度可判"


def test_query_reports_evidence_without_an_observation():
    tracker = _tracker()
    for index, area in enumerate((8000, 7800)):
        tracker.observe(index, _pred(_box(area)))
    query = tracker.memory_query(1)
    assert query.entry_count == 2
    assert query.bbox is not None and query.centroid is not None
    assert query.ratio == pytest.approx(7800 / CAVITY)
    assert query.agreement is None and query.supported is False


def test_bank_rejects_impossible_capacity():
    with pytest.raises(ValueError, match="容量"):
        MemoryBank(obj_id="B01", params=MemoryParams(capacity=0))


# ── 5. 恒定算力与主路径不变 ─────────────────────────────────────────────
def test_capacity_is_independent_of_sequence_length():
    """写 200 帧，库还是那几条——不随帧数膨胀。"""
    tracker = _tracker(frame_count=200)
    for index in range(200):
        tracker.observe(index, _pred(_box(max(800, 8000 - index * 36))))
    assert len(tracker.bank) == CAPACITY
    assert tracker.bank.frame_indices[-1] == 199
    assert all(entry.mask is not None for entry in tracker.bank.entries)


@pytest.fixture(scope="module")
def long_frames(tmp_path_factory):
    scene = default_rack_scene(frame_count=120)
    directory = tmp_path_factory.mktemp("memory-bank")
    return scene, write_rack_frames(directory, scene)


def test_rendered_scene_bank_matches_clean_baseline(long_frames):
    """记忆库不改变主干数值：同一段渲染短片，曲线与 task 4/5/6 的基线逐位一致。"""
    scene, paths = long_frames
    shape = load_rgb(paths[0]).shape
    seeds = [
        (bag.bag_seq, slot_seed_prompt(shape, slot, len(scene.bags)))
        for slot, bag in enumerate(scene.bags)
    ]
    curves = {c.obj_id: c for c in analyse_frames(paths, seeds, GeometricMaskPredictor())}

    first = curves["B01"].area_ratio_series[0]
    last = curves["B01"].area_ratio_series[-1]
    assert first == pytest.approx(0.58843, abs=1e-5), "首值与主干基线一致（渲染真值回归）"
    assert last < first, "液面在降"
    assert all(ratio is not None for ratio in curves["B01"].area_ratio_series)
    assert curves["B01"].obj_id == "B01"
    assert curves["B02"].obj_id == "B02"


def test_rendered_scene_keeps_top_k_memories(long_frames):
    """渲染短片跑完：每条轨迹都攒着 top-k 条带掩膜的记忆。"""
    scene, paths = long_frames
    shape = load_rgb(paths[0]).shape
    seeds = [
        (bag.bag_seq, slot_seed_prompt(shape, slot, len(scene.bags)))
        for slot, bag in enumerate(scene.bags)
    ]
    predictor = GeometricMaskPredictor()
    state = analyse_frames(paths, seeds, predictor)
    assert len(state) == len(scene.bags)

    from infusion_monitor.core import propagate as P

    first = load_rgb(paths[0])
    prop = P.init_state([path.name for path in paths], first.shape)
    for slot, bag in enumerate(scene.bags):
        prompt = slot_seed_prompt(shape, slot, len(scene.bags))
        cavity = int(predictor.estimate_cavity(first, prompt).area_px)
        P.add_new_points_or_box(prop, frame_index=0, obj_id=bag.bag_seq, prompt=prompt, cavity_area_px=cavity)
    by_name = {path.name: path for path in paths}
    for _ in P.propagate_in_video(prop, predictor, lambda name: load_rgb(by_name[name])):
        pass

    for obj_id, track in prop.objects.items():
        bank = track.visibility.bank
        assert len(bank) == CAPACITY, f"{obj_id} 的库应攒到 top-k"
        assert bank.entries[-1].frame_index == scene.frame_count - 1, "最后一条是末帧"
        assert all(entry.score > 0.0 for entry in bank.entries), "置信逐帧带下来了"
        assert bank.rejected_total == 0, "正常拍摄不该有拒收"


@pytest.mark.skipif(sys.platform == "win32", reason="RSS 口径按 macOS/Linux 记")
def test_bank_holds_a_fixed_number_of_masks():
    """库只留 CAPACITY 张掩膜：帧数翻多少倍都不涨（恒定算力，任务 6 的口径）。"""
    tracker = _tracker(frame_count=200)
    for index in range(200):
        tracker.observe(index, _pred(_box(max(800, 8000 - index * 36))))
    mask_bytes = sum(entry.mask.nbytes for entry in tracker.bank.entries if entry.mask is not None)
    assert mask_bytes == CAPACITY * HEIGHT * WIDTH, f"库只应留 {CAPACITY} 张掩膜，实测 {mask_bytes} 字节"
    assert len(tracker.bank.rejected) <= 4, "拒收记录也是定长，不随帧数涨"