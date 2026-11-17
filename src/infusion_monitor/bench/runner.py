"""在公开 VOS 子集上跑「记忆传播 vs 阈值单帧」的可复现对照。

两套方案都走产品主路径 ``init_state`` → ``add_new_points_or_box`` →
``propagate_in_video``，同一个阈值核、同一套碎片后处理，唯一差别是每帧提示从哪来。
指标口径见 :mod:`.metrics`；GT 目录结构沿用 ``sav_evaluator.py`` 文档化的
``{video_id}/{frame_id}.png``。

**确定性**：整条链路没有随机数——阈值核、连通域选择、距离变换全是确定性算子，
色彩模型由提示框内像素的均值与标准差直接给出。``seed`` 仍照记（见 ``RANDOM_SEED``），
是为了让「随机种子」这一栏有实数可填，也方便日后给任何抽样式增强留位。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.propagate import (
    add_new_points_or_box,
    init_state,
    load_rgb,
    propagate_in_video,
)
from .metrics import (
    BOUND_TH,
    SequenceMetric,
    aggregate,
    frame_metrics,
    summarise,
)
from .schemes import (
    MAX_HOLE_AREA,
    MAX_SPRINKLE_AREA,
    MEMORY_SCHEME,
    SINGLE_FRAME_SCHEME,
    ColorBandParams,
    ColorBandPredictor,
    anchor_from_mask,
)

#: 记录在结果里的随机种子。本链路无随机源，写 0 是为了让「种子」一栏不是空的。
RANDOM_SEED = 0

#: 真值对象面积掉到本序列峰值这个比例以下即记为「遮挡帧」。
OCCLUSION_AREA_RATIO = 0.5

#: 单序列最多评几个对象。首帧标注里有几个就评几个，但设个上限避免多目标序列把
#: 基准跑成压测。公开语料里绝大多数序列首帧只有一个对象。
MAX_OBJECTS = 3

IMAGE_SUBDIR = ("JPEGImages", "480p")
ANNOTATION_SUBDIR = ("Annotations", "480p")


@dataclass
class SequenceSpec:
    """一个待评序列在磁盘上的位置与规模。"""

    name: str
    image_dir: Path
    annotation_dir: Path
    frame_names: list[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.frame_names)


def _read_gt(path: Path) -> np.ndarray:
    """读一张调色板标注 PNG，取**调色板索引**而不是 RGB。

    DAVIS 的标注是多值调色板图：索引 0 是背景，1、2、3… 是不同对象。用 ``IMREAD_*``
    会把调色板展开成 RGB，颜色值与对象号不再一一对应（真值里出现 128 这种值就是
    展开后的颜色而非对象号）。``Image.open(...).convert`` 之前拿 ``np.array`` 拿到的
    才是索引。
    """
    with Image.open(path) as image:
        return np.asarray(image)


def discover(root: Path | str = "data/vos_subset") -> list[SequenceSpec]:
    """列出落盘在磁盘上的全部序列（有图且有标注才算）。"""
    base = Path(root)
    image_root = base.joinpath(*IMAGE_SUBDIR)
    annotation_root = base.joinpath(*ANNOTATION_SUBDIR)
    if not image_root.is_dir():
        return []
    specs: list[SequenceSpec] = []
    for image_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        annotation_dir = annotation_root / image_dir.name
        if not annotation_dir.is_dir():
            continue
        frames = sorted(
            p.stem
            for p in annotation_dir.glob("*.png")
            if (image_dir / f"{p.stem}.jpg").is_file()
        )
        if frames:
            specs.append(SequenceSpec(image_dir.name, image_dir, annotation_dir, frames))
    return specs


def _object_ids(mask: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(mask) if int(v) > 0)


def _gt_mask(root: Path, frame: str, obj_id: int, shape: tuple[int, int]) -> np.ndarray:
    """取某帧某对象的真值掩膜；该帧没有这个对象时给全空掩膜。"""
    raw = _read_gt(root / f"{frame}.png")
    mask = np.zeros(shape, dtype=bool)
    if raw.shape != shape:
        # 标注与图像分辨率不一致：按画布尺寸对齐，多余部分截掉、短缺的补空。
        canvas = np.zeros(shape, dtype=raw.dtype)
        h = min(raw.shape[0], shape[0])
        w = min(raw.shape[1], shape[1])
        canvas[:h, :w] = raw[:h, :w]
        raw = canvas
    mask[...] = raw == obj_id
    return mask


def _run_scheme(
    spec: SequenceSpec,
    obj_id: int,
    scheme: str,
    params: ColorBandParams,
    *,
    max_frames: int | None = None,
) -> tuple[list[tuple[int, np.ndarray]], dict[str, object]]:
    """跑一套方案，返回 ``[(帧号, 预测掩膜), ...]`` 与运行统计。

    帧号沿用真值帧名里的原始序号，不从 0 重排——评测表里「第几帧」要能对回磁盘上的
    ``00042.png``，重排会让人查不到原始帧。
    """
    frames = spec.frame_names if max_frames is None else spec.frame_names[:max_frames]
    first = load_rgb(spec.image_dir / f"{frames[0]}.jpg")
    shape = (first.shape[0], first.shape[1])

    gt_first = _gt_mask(spec.annotation_dir, frames[0], obj_id, shape)
    if not gt_first.any():
        return [], {"skipped": "首帧真值为空"}

    state = init_state(frames, shape)
    add_new_points_or_box(
        state,
        frame_index=0,
        obj_id=f"obj{obj_id}",
        prompt=anchor_from_mask(gt_first),
        # 面积比曲线的分母。用锚帧真值面积，两个方案起点完全一致，
        # 且不随方案变化——指标只用掩膜，不用这里的面积比。
        cavity_area_px=int(np.count_nonzero(gt_first)),
    )

    predictor = ColorBandPredictor(params)
    read_cache: dict[str, np.ndarray] = {}

    def read_frame(name: str) -> np.ndarray:
        cached = read_cache.get(name)
        if cached is None:
            cached = load_rgb(spec.image_dir / f"{name}.jpg")
            read_cache.clear()  # 只留一帧：基准要跑上千帧，别把整段序列读进内存
            read_cache[name] = cached
        return cached

    started = time.perf_counter()
    predictions: list[tuple[int, np.ndarray]] = []
    empty_frames = 0
    for result in propagate_in_video(
        state,
        predictor,
        read_frame,
        reacquire=None,  # 关闭再获取：两套方案只允许「记忆」这一个变量
        memory_prompt=scheme == MEMORY_SCHEME,
    ):
        mask = result.masks.get(f"obj{obj_id}")
        if mask is None:
            empty_frames += 1
            mask = np.zeros(shape, dtype=bool)
        predictions.append((result.frame_index, mask))
    elapsed = time.perf_counter() - started

    stats = {
        "scheme": scheme,
        "frames": len(predictions),
        "empty_frames": empty_frames,
        "elapsed_s": round(elapsed, 3),
        "ms_per_frame": round(1000 * elapsed / max(1, len(predictions)), 2),
    }
    return predictions, stats


def evaluate_sequence(
    spec: SequenceSpec,
    *,
    params: ColorBandParams | None = None,
    max_frames: int | None = None,
    schemes: tuple[str, ...] = (MEMORY_SCHEME, SINGLE_FRAME_SCHEME),
) -> tuple[list[SequenceMetric], list[dict[str, object]]]:
    """在一个序列上跑完两套方案并算指标。"""
    params = params or ColorBandParams()
    frames = spec.frame_names if max_frames is None else spec.frame_names[:max_frames]
    first = load_rgb(spec.image_dir / f"{frames[0]}.jpg")
    shape = (first.shape[0], first.shape[1])

    obj_ids = _object_ids(_read_gt(spec.annotation_dir / f"{frames[0]}.png"))[:MAX_OBJECTS]
    results: list[SequenceMetric] = []
    run_stats: list[dict[str, object]] = []

    for obj_id in obj_ids:
        # 先算一遍真值面积峰值，用来判哪些帧算「被遮挡」。判定只依赖真值，
        # 与跑哪套方案无关——两套方案面对同一批遮挡帧。
        areas = {
            name: int(np.count_nonzero(_gt_mask(spec.annotation_dir, name, obj_id, shape)))
            for name in frames
        }
        peak = max(areas.values()) if areas else 0
        occluded = frozenset(
            index
            for index, name in enumerate(frames)
            if 0 < areas[name] <= OCCLUSION_AREA_RATIO * peak
        )

        for scheme in schemes:
            predictions, stats = _run_scheme(
                spec, obj_id, scheme, params, max_frames=max_frames
            )
            if not predictions:
                continue
            run_stats.append({"sequence": spec.name, "obj_id": obj_id, **stats})
            metric = SequenceMetric(
                sequence=f"{spec.name}#obj{obj_id}", scheme=scheme, occluded_frames=occluded
            )
            for frame_index, pred_mask in predictions:
                truth = _gt_mask(
                    spec.annotation_dir, frames[frame_index], obj_id, shape
                )
                metric.frames.append(
                    frame_metrics(
                        pred_mask,
                        truth,
                        sequence=metric.sequence,
                        frame_index=frame_index,
                        obj_id=obj_id,
                    )
                )
            results.append(metric)
    return results, run_stats


def run_benchmark(
    root: Path | str = "data/vos_subset",
    *,
    max_sequences: int | None = None,
    max_frames: int | None = None,
    params: ColorBandParams | None = None,
    schemes: tuple[str, ...] = (MEMORY_SCHEME, SINGLE_FRAME_SCHEME),
) -> dict[str, object]:
    """跑完整对照，返回可直接落 ``docs/benchmark-vos.md`` 的结果字典。

    语料不在磁盘上时**不静默返回空**——返回 ``status="skipped"`` 并带上原因，
    留给 CLI 写进文档状态块。
    """
    specs = discover(root)
    if not specs:
        return {
            "status": "skipped",
            "skipped_reason": f"{root} 下没有找到可用的公开 VOS 子集（先跑 scripts/fetch_vos_subset.py）",
            "dataset": None,
        }
    if max_sequences is not None:
        specs = specs[:max_sequences]

    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    totals: dict[str, dict[str, object]] = {}
    per_sequence: dict[str, list[SequenceMetric]] = {}
    run_stats: list[dict[str, object]] = []

    for spec in specs:
        metrics, stats = evaluate_sequence(
            spec, params=params, max_frames=max_frames, schemes=schemes
        )
        run_stats.extend(stats)
        for metric in metrics:
            rows.append(metric.to_row())
            per_sequence.setdefault(metric.scheme, []).append(metric)

    for scheme, metrics in per_sequence.items():
        totals[scheme] = aggregate(metrics)

    occluded_sequences = sorted(
        {m.sequence for metrics in per_sequence.values() for m in metrics if m.occluded}
    )

    report = {
        "status": "ok",
        "dataset": {
            "name": "DAVIS 2017 trainval 480p",
            "root": str(root),
            "sequences": [s.name for s in specs],
            "sequence_count": len(specs),
            "frames_per_sequence": {s.name: s.frame_count for s in specs},
            "license_note": "DAVIS 由 ETH Zürich CVG 与日内瓦大学发布；本仓库只放本地评测副本，不 redistribution。",
        },
        "kernel": {
            "predictor": "color_band",
            "hue_tolerance": (params or ColorBandParams()).hue_tolerance,
            "spread_sigma": (params or ColorBandParams()).spread_sigma,
            "max_hole_area": MAX_HOLE_AREA,
            "max_sprinkle_area": MAX_SPRINKLE_AREA,
            "bound_th": BOUND_TH,
            "occlusion_area_ratio": OCCLUSION_AREA_RATIO,
            "max_objects_per_sequence": MAX_OBJECTS,
            "reacquire": "disabled（两套方案只允许「记忆」一个变量）",
        },
        "random_seed": RANDOM_SEED,
        "deterministic": True,
        "rows": rows,
        "totals": totals,
        "occluded_sequences": occluded_sequences,
        "run_stats": run_stats,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    report["table"] = summarise(report)
    return report


def verdict(report: dict[str, object]) -> dict[str, object]:
    """对照结论：记忆方案在含遮挡序列上是否不劣于阈值单帧。

    判定用**含遮挡帧上的 J&F**（含遮挡序列的全体帧），门槛 ``MIN_MARGIN = 0.0``——
    即「不劣」。这不是宽松放水，是本任务要验的那条命题本身；实测数值由文档记录。
    """
    totals = report.get("totals", {})  # type: ignore[assignment]
    memory = totals.get(MEMORY_SCHEME, {})
    single = totals.get(SINGLE_FRAME_SCHEME, {})
    min_margin = 0.0

    def value(block: dict[str, object], key: str) -> float:
        return float(block.get(key, 0.0))  # type: ignore[arg-type]

    overall_margin = value(memory, "mean_j_and_f") - value(single, "mean_j_and_f")
    occluded_margin = value(memory, "mean_occluded_j_and_f") - value(
        single, "mean_occluded_j_and_f"
    )
    return {
        "memory_mean_j_and_f": value(memory, "mean_j_and_f"),
        "single_mean_j_and_f": value(single, "mean_j_and_f"),
        "overall_margin": round(overall_margin, 4),
        "memory_occluded_j_and_f": value(memory, "mean_occluded_j_and_f"),
        "single_occluded_j_and_f": value(single, "mean_occluded_j_and_f"),
        "occluded_margin": round(occluded_margin, 4),
        "min_margin": min_margin,
        "not_worse_on_occluded": occluded_margin >= min_margin,
        "not_worse_overall": overall_margin >= min_margin,
        "occluded_sequences": report.get("occluded_sequences", []),
    }