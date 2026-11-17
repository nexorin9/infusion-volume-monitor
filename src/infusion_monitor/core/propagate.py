"""逐帧传播：起锚 → 记住 → 传播（一次上传得到每袋跨帧一致的面积比曲线）。

**与源主路径的对应关系**（只借调用链与语义，不搬代码、不跑源权重）：

| 本模块 | 源主路径 | 代码证据 |
|--------|----------|----------|
| :func:`init_state` | ``predictor.init_state(video_path=...)`` | ``vos_inference.py:137``；``benchmark.py:45`` |
| :func:`add_new_points_or_box` | ``predictor.add_new_points_or_box(...)`` / ``add_new_mask`` | ``benchmark.py:63-69``；``vos_inference.py:205-210`` |
| :func:`propagate_in_video` | ``for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(...)`` | ``vos_inference.py:224-231`` |

源实现里「记住上一帧」由记忆特征参与下一帧推理完成（``memory_attention.py`` 的
``MemoryAttentionLayer._forward_ca``）；本产品的等价物是 :class:`.occlusion.TrackMemory`
——上一帧掩膜同时充当「记忆」（下一帧的 ROI 与提示点）与「续接依据」（找不到目标时
不新建轨迹，而是接着上一帧的 ROI 再试；恢复点用记忆比对确认还是同一袋）。记忆由
:class:`.occlusion.OcclusionTracker` 单一持有，本模块只转发，不复制第二份状态；
记忆的 top-k、相似度加权与衰减在 :mod:`.memory_bank` 内实现，提示仍只由本模块的
:meth:`ObjectTrack.follow_up_prompt` 一处产出。

**三条纪律**：

1. **恒定算力**：一次只保留每条轨迹的上一帧掩膜，不把整段掩膜序列攒在内存里，
   1200 帧与 24 帧的峰值内存同量级。
2. **遮挡不丢弃**：某帧找不到目标时该帧标 ``occluded``、面积比留 ``None``，**不插值、
   不沿用上一帧数值**——拿一条假曲线去算毫升和 ETA 比没有曲线更危险。
3. **不猜**：远离提示的连通域宁可不要（返回空 → 标遮挡），也不硬接一块不相干区域。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
from PIL import Image

from ..identity.reanchor import BagDescriptor, describe_region

from .occlusion import OcclusionTracker, ReacquireHandoff, VisibilityParams
from .predictor import (
    MaskPrediction,
    PerPointMaskPredictor,
    Point,
    Prompt,
    mask_centroid,
)
from .reacquire import (
    REACQUIRED,
    LostTrackView,
    ReacquireParams,
    ReacquireVerdict,
    reacquire_tracks,
)

__all__ = [
    "PropagationState",
    "ObjectTrack",
    "FrameResult",
    "BagCurve",
    "init_state",
    "add_new_points_or_box",
    "propagate_in_video",
    "slot_seed_prompt",
    "analyse_frames",
    "load_rgb",
]

logger = logging.getLogger("infusion_monitor.core.propagate")

#: 跟随上一帧掩膜时的 ROI 外扩比例（袋体轻微晃动 + 液面下降都在这个余量里）
ROI_PAD_RATIO = 0.35

#: 丢失后再等几帧才试再获取（避免在同一片遮挡上反复试）。
REACQUIRE_GAP_FRAMES = 2

#: 哨兵：``reacquire`` 省略时取 :class:`~.reacquire.ReacquireParams` 默认值（**开启**）；
#: 显式传 ``None`` 才关闭整条再获取路径（回归对照与无遮挡场景用）。
DEFAULT_REACQUIRE = object()


def load_rgb(path: Path | str) -> np.ndarray:
    """读一帧 JPEG 为 ``HxWx3`` ``uint8`` RGB 数组。通道序只在这里转换一次。"""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def slot_seed_prompt(
    shape: tuple[int, int],
    slot_index: int,
    slot_count: int,
    *,
    focus_y: float = 0.78,
) -> Prompt:
    """按挂点顺序把画面竖向等分为 ``slot_count`` 条，给出该袋位的起锚提示。

    这是**确定性起锚**：输液架上的袋子按袋序从左到右挂，槽位几何本身就给了位置先验，
    不需要护士逐袋点像素。首帧候选枚举（task 5）会用更准的候选框替换/细化它，
    在那之前这条先验保证主干可跑。
    """
    height, width = shape[0], shape[1]
    count = max(1, slot_count)
    step = width / count
    x0 = int(round(slot_index * step))
    x1 = int(round((slot_index + 1) * step))
    x0, x1 = max(0, min(x0, width)), max(1, min(x1, width))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    centre = (x0 + x1) / 2.0
    return Prompt(points=(Point(centre, height * focus_y, 1),), box=(x0, 0.0, float(x1), float(height)))


@dataclass
class ObjectTrack:
    """一条被持续跟踪的袋轨迹（一条 = 一个 ``bag_seq``）。

    可见度与「记忆」由 :class:`.occlusion.OcclusionTracker` 单一持有（记忆本体在
    :mod:`.memory_bank`），本类只做转发——不复制第二份状态。
    """

    obj_id: str
    anchor_frame: int
    prompt: Prompt
    cavity_area_px: int
    frame_count: int
    visibility: OcclusionTracker
    ratios: list[float | None] = field(default_factory=list)
    occlusion_flags: list[bool] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    #: 该袋在输液架上的挂点序号（0 起）。再获取的槽位几何先验用它；未知时为 ``None``。
    slot_index: int | None = None
    #: 本条轨迹的再获取判定历史（按帧号倒序，最新在前）。
    reacquires: list[ReacquireVerdict] = field(default_factory=list)

    @property
    def last_mask(self) -> np.ndarray | None:
        memory = self.visibility.memory
        return memory.mask if memory is not None else None

    @property
    def last_bbox(self) -> tuple[int, int, int, int] | None:
        memory = self.visibility.memory
        return memory.bbox if memory is not None else None

    @property
    def last_seen_frame(self) -> int | None:
        memory = self.visibility.memory
        return memory.frame_index if memory is not None else None

    def prepare(self) -> None:
        self.ratios = [None] * self.frame_count
        self.occlusion_flags = [True] * self.frame_count
        self.scores = [0.0] * self.frame_count

    def follow_up_prompt(self, shape: tuple[int, int]) -> Prompt | None:
        """由上一帧掩膜导出本帧提示：外扩 ROI + 质心正点。"""
        memory = self.visibility.memory
        if memory is None:
            return None
        height, width = shape[0], shape[1]
        x0, y0, x1, y1 = memory.bbox
        pad_x = max(4.0, (x1 - x0) * ROI_PAD_RATIO)
        pad_y = max(4.0, (y1 - y0) * ROI_PAD_RATIO)
        centre = mask_centroid(memory.mask) if memory.mask is not None else None
        cx, cy = centre if centre is not None else ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        return Prompt(
            points=(Point(cx, cy, 1),),
            box=(
                float(max(0.0, x0 - pad_x)),
                float(max(0.0, y0 - pad_y)),
                float(min(float(width), x1 + pad_x)),
                float(min(float(height), y1 + pad_y)),
            ),
        )

    def absorb(self, frame_index: int, prediction: MaskPrediction | None) -> None:
        """吸收一帧的观测结果。``None`` 或空掩膜 = 这一帧看不见它。"""
        if self.cavity_area_px <= 0:
            # 袋腔估不出来时没有可信分母，一律按「看不见」处理，不给假面积比
            prediction = None
        self.visibility.observe(frame_index, prediction)
        self.ratios[frame_index] = self.visibility.ratios[frame_index]
        self.occlusion_flags[frame_index] = self.visibility.ratios[frame_index] is None
        self.scores[frame_index] = self.visibility.scores[frame_index]

    def reacquire_due(self, frame_index: int, gap_frames: int = REACQUIRE_GAP_FRAMES) -> bool:
        """这一帧该不该试再获取。

        三个条件缺一不可：① 已交出（``handoff`` 在）；② 本帧在丢失窗口之后；③ 还没被
        认领过。**没交出的轨迹一条也不碰**——遮挡未超窗口时是 :mod:`.occlusion` 的
        恢复路径在管，两套逻辑同时跑会把同一袋接成两条。
        """
        handoff = self.visibility.handoff
        if handoff is None:
            return False
        if self.visibility.reacquired_at is not None:
            return False
        return frame_index >= handoff.lost_frame + gap_frames


@dataclass
class FrameResult:
    """一帧的传播产出（对应源 ``propagate_in_video`` 每次 yield 的三元组）。"""

    frame_index: int
    frame_name: str
    masks: dict[str, np.ndarray]
    scores: dict[str, float]


@dataclass
class PropagationState:
    """一次传播的全部状态（对应源 ``inference_state``）。"""

    frame_names: tuple[str, ...]
    height: int
    width: int
    objects: dict[str, ObjectTrack] = field(default_factory=dict)
    #: 起锚后冻结的身份集合（源 ``vos_inference.py:193-204`` 的 ``object_ids_set``）。
    #: 非 ``None`` 时再往里加新 id 会抛 :class:`ObjectIdSetFrozen`——中途凭空多出一只袋子
    #: 只能走 :mod:`.reacquire` 的再获取或护士认领，不能插进正在跑的这一遍传播。
    frozen_object_ids: frozenset[str] | None = None

    @property
    def frame_count(self) -> int:
        return len(self.frame_names)


class ObjectIdSetFrozen(RuntimeError):
    """起锚之后再往同一份传播状态里加**新**对象 id。

    对应源实现的 ``RuntimeError``（``vos_inference.py:198-204``）：``object_ids_set`` 在第一条
    输入帧上定死，后面出现新 ``object_id`` 直接报错并提示改用「后出现对象」的开关。
    本产品没有那个开关——多出来的那只袋子要靠再获取或护士认领接进来。
    """


def init_state(frame_names: Sequence[str], shape: tuple[int, int]) -> PropagationState:
    """起状态：登记帧序列与画面尺寸，暂不挂任何对象。"""
    if not frame_names:
        raise ValueError("帧序列为空，无法起传播状态")
    height, width = shape[0], shape[1]
    return PropagationState(
        frame_names=tuple(frame_names), height=height, width=width, objects={}
    )


def add_new_points_or_box(
    state: PropagationState,
    *,
    frame_index: int,
    obj_id: str,
    prompt: Prompt,
    cavity_area_px: int,
    visibility_params: VisibilityParams | None = None,
    slot_index: int | None = None,
) -> ObjectTrack:
    """在指定帧给一个对象下首个提示（源主路径的第二段）。"""
    if not 0 <= frame_index < state.frame_count:
        raise ValueError(f"frame_index={frame_index} 越界（共 {state.frame_count} 帧）")
    if cavity_area_px <= 0:
        raise ValueError(f"obj_id={obj_id} 的腔体面积非法：{cavity_area_px}")
    frozen = state.frozen_object_ids
    if frozen is not None and obj_id not in frozen:
        raise ObjectIdSetFrozen(
            f"obj_id={obj_id} 不在起锚时冻结的身份集合里（{sorted(frozen)}）；"
            "传播途中不能凭空多出一只袋子，请走再获取或让护士在列表里认领"
        )
    track = ObjectTrack(
        obj_id=obj_id,
        anchor_frame=frame_index,
        prompt=prompt,
        cavity_area_px=int(cavity_area_px),
        frame_count=state.frame_count,
        slot_index=slot_index,
        visibility=OcclusionTracker(
            obj_id=obj_id,
            cavity_area_px=int(cavity_area_px),
            frame_count=state.frame_count,
            params=visibility_params or VisibilityParams(),
        ),
    )
    track.prepare()
    state.objects[obj_id] = track
    return track


def propagate_in_video(
    state: PropagationState,
    predictor: PerPointMaskPredictor,
    read_frame: Callable[[str], np.ndarray],
    *,
    reacquire: ReacquireParams | None | object = DEFAULT_REACQUIRE,
    memory_prompt: bool = True,
) -> Iterator[FrameResult]:
    """逐帧传播（源主路径的第三段）：逐帧 yield 每个对象的掩膜。

    ``read_frame`` 由调用方给出（本产品从解包后的帧目录按名读），因此本模块不关心
    帧从哪来——内存里的一串数组、公开 VOS 子集的帧、还是磁盘上的 JPEG。

    **再获取**：某一帧吸收完之后，若有轨迹已被交出（丢失窗口用尽）且过了
    ``REACQUIRE_GAP_FRAMES`` 的冷却，就在**这一帧**上跑一次 :mod:`.reacquire` 的
    局部重枚举；判定为 ``reacquired`` 的才调 :meth:`OcclusionTracker.reacquire_from`
    解冻并把该帧重记为「看见了」。判不出来就保持 lost，下一帧再试——**失败不静默、
    也不新建轨迹**。传 ``reacquire=None`` 即整条路径关闭（回归对照用）。

    **记忆消融**：``memory_prompt=False`` 时每一帧都退回首帧锚点提示，不再用上一帧
    掩膜导出的提示。这是公开子集对照基准里「阈值单帧」基础方案的入口——与产品主路径
    走**同一段代码、同一套阈值核**，唯一差别就是这一行。默认 ``True``，线上行为不变。
    """
    # 省略 = 开启（取默认参数）；显式 None = 关闭整条路径
    enabled = reacquire is not None
    reacquire_params = ReacquireParams() if reacquire is DEFAULT_REACQUIRE else reacquire
    for index, name in enumerate(state.frame_names):
        image = read_frame(name)
        masks: dict[str, np.ndarray] = {}
        scores: dict[str, float] = {}
        for obj_id, track in state.objects.items():
            if index < track.anchor_frame:
                continue
            prompt = (
                track.prompt
                if index == track.anchor_frame or not memory_prompt
                else track.follow_up_prompt(image.shape)
            )
            if prompt is None:
                track.absorb(index, None)
                continue
            predictions = predictor.predict(image, prompt)
            best = predictions[0] if predictions else None
            if best is not None and best.is_empty:
                best = None
            track.absorb(index, best)
            if best is not None:
                masks[obj_id] = best.mask
                scores[obj_id] = best.score
        if enabled:
            _run_reacquire(
                state, predictor, image, index, reacquire_params, masks, scores, read_frame
            )
        yield FrameResult(index, name, masks, scores)


def _run_reacquire(
    state: PropagationState,
    predictor: PerPointMaskPredictor,
    image: np.ndarray,
    frame_index: int,
    params: ReacquireParams,
    masks: dict[str, np.ndarray],
    scores: dict[str, float],
    read_frame: Callable[[str], np.ndarray] | None = None,
) -> None:
    """在本帧上尝试把所有 lost 轨迹再获取一遍（主路径钩子）。

    只在「至少有一条轨迹到期」时进入，因此正常拍摄（无遮挡）这条路径**零开销**。

    外观描述子在**丢失前最后一帧**就地取：那一帧的图按需重读一次（每条 lost 轨迹
    最多一次，之后接回与否都不再读），配上记忆库里留着的掩膜算出色相直方图。取不到
    就如实留空，由 :mod:`.reacquire` 按「外观不可用」加罚处理，而不是假装没有这条证据。
    """
    due = [track for track in state.objects.values() if track.reacquire_due(frame_index)]
    if not due:
        return
    slot_count = len(state.objects)
    views = []
    for track in due:
        view = LostTrackView.from_tracker(
            track.obj_id,
            track.visibility,
            slot_index=track.slot_index,
            slot_count=slot_count,
        )
        views.append(
            view if view.descriptor is not None else replace(view, descriptor=_last_seen_descriptor(
                state, track, read_frame
            ))
        )
    batch = reacquire_tracks(
        views, image, predictor, frame_index=frame_index, params=params
    )
    by_obj = batch.by_obj()
    for track in due:
        verdict = by_obj.get(track.obj_id)
        if verdict is None:
            continue
        track.reacquires.append(verdict)
        if verdict.state != REACQUIRED or verdict.candidate is None:
            continue
        # 接回：解冻 → 重记这一帧。bag_seq 不变、记忆库不新建，接的是原来那条曲线。
        track.visibility.reacquire_from(frame_index)
        track.absorb(frame_index, verdict.candidate.as_prediction())
        masks[track.obj_id] = verdict.candidate.mask
        scores[track.obj_id] = float(verdict.candidate.stability_score)


def _last_seen_descriptor(
    state: PropagationState,
    track: ObjectTrack,
    read_frame: Callable[[str], np.ndarray] | None,
) -> BagDescriptor | None:
    """用「丢失前最后一帧」的图 + 记忆库里的掩膜算外观描述子。

    三种情况都返回 ``None``（不假装有证据）：① 记忆库为空或那条没留掩膜；② 没有
    ``read_frame``（调用方只给了内存帧）；③ 那一帧已经不在帧序列里。
    """
    handoff = track.visibility.handoff
    memory = track.visibility.memory
    if handoff is None or memory is None or memory.mask is None or read_frame is None:
        return None
    index = memory.frame_index
    if not 0 <= index < state.frame_count:
        return None
    try:
        image = read_frame(state.frame_names[index])
    except (OSError, KeyError):  # 帧已被清掉：外观不可用，如实留空
        return None
    if memory.mask.shape != image.shape[:2]:
        return None
    return describe_region(
        image, memory.mask, bag_seq=track.obj_id, slot_index=track.slot_index or 0
    )


@dataclass
class BagCurve:
    """一条轨迹的产出：给护士端的面积比曲线。"""

    obj_id: str
    frame_indices: list[int]
    area_ratio_series: list[float | None]
    occlusion_flags: list[bool]
    cavity_area_px: int
    scores: list[float]
    anchor: tuple[float, float] | None = None
    #: 逐帧可见度（visible / partial / occluded / reappeared / lost；起锚前为 None）
    visibility_states: list[str | None] = field(default_factory=list)
    #: 超出丢失窗口后交出的轨迹（交 :mod:`.reacquire` 的再获取路径）
    handoff: ReacquireHandoff | None = None
    #: 再获取判定历史（被交出后每帧一次，按帧号顺序）
    reacquires: list[ReacquireVerdict] = field(default_factory=list)
    #: 逐帧掩膜质心（看不见的帧为 ``None``）。给 :mod:`.multi_object` 的槽位分配用。
    #: 只有质心与面积两个标量，**不存掩膜**——恒定算力的边界不动。
    frame_centroids: list[tuple[float, float] | None] = field(default_factory=list)
    #: 逐帧掩膜面积（px²， 看不见的帧为 ``None``）。
    frame_areas_px: list[int | None] = field(default_factory=list)

    @property
    def visible_frame_count(self) -> int:
        return sum(1 for flag in self.occlusion_flags if not flag)

    @property
    def occluded_frame_count(self) -> int:
        return sum(1 for flag in self.occlusion_flags if flag)

    @property
    def is_lost(self) -> bool:
        """轨迹是否被交出过（护士端据此提示「这段拍糊了，袋子交出过」）。

        交出**不等于**最终没找回来：:mod:`.reacquire` 可能在后面的帧把它接了回去，
        那要看 :attr:`reacquired` / :attr:`still_lost`。
        """
        return self.handoff is not None

    @property
    def reacquired(self) -> bool:
        """是否被再获取接回过（护士端据此不必提示「这袋需要重新认领」）。"""
        return any(verdict.reacquired for verdict in self.reacquires)

    @property
    def still_lost(self) -> bool:
        """交出之后**整段都没接回来**——保持丢失，等护士在列表里认领。"""
        return self.is_lost and not self.reacquired

    @property
    def reacquired_frame(self) -> int | None:
        """接回发生在哪一帧；从未接回为 ``None``。"""
        for verdict in self.reacquires:
            if verdict.reacquired:
                return verdict.frame_index
        return None

    @property
    def latest_ratio(self) -> float | None:
        for ratio in reversed(self.area_ratio_series):
            if ratio is not None:
                return ratio
        return None


def analyse_frames(
    frame_paths: Sequence[Path],
    seed_prompts: Sequence[tuple[str, Prompt]],
    predictor: PerPointMaskPredictor,
    *,
    anchor_frame: int = 0,
    slot_indices: Sequence[int] | None = None,
    reacquire: ReacquireParams | None | object = DEFAULT_REACQUIRE,
) -> list[BagCurve]:
    """医院侧入口：帧路径 + 每袋起锚提示 → 每袋的面积比曲线。

    只在起锚帧估一次袋腔；之后每帧只保留上一帧掩膜，内存不随帧数线性膨胀。

    ``slot_indices`` 给每袋的挂点序号（再获取的槽位几何先验、:mod:`.multi_object` 的槽位
    分配都用它）；不给就按起锚顺序当作 0,1,2…——这与 :func:`slot_seed_prompt` 的竖向等分
    口径一致。``reacquire`` 省略即开启再获取；显式传 ``None`` 关闭。

    实现上本函数就是 :func:`..multi_object.propagate_batch` 的薄包装（按起锚顺序返回曲线
    列表）。多袋并行只有一个实现，两个入口不会各走一套算法、数值也就不会分叉。
    """
    if not frame_paths:
        return []
    from .multi_object import propagate_batch  # 延迟导入：避免两个模块互相 import

    slot_of = (
        {obj_id: int(slot_indices[index]) for index, (obj_id, _) in enumerate(seed_prompts)}
        if slot_indices is not None
        else {obj_id: index for index, (obj_id, _) in enumerate(seed_prompts)}
    )
    batch = propagate_batch(
        frame_paths,
        [
            (obj_id, prompt)
            for obj_id, prompt in seed_prompts
        ],
        predictor,
        anchor_frame=anchor_frame,
        reacquire=reacquire,
        _slot_of=slot_of,
    )
    return list(batch.ordered)