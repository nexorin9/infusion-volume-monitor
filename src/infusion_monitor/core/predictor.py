"""可插拔的「给点/框 → 出掩膜」预测器协议。

**为什么要有这一层**：生产权重实现与确定性几何核实现（:mod:`.geom_kernel`）必须能被
同一条主路径调用——业务层只认本协议，不直接依赖任何一种实现。源产品主路径的对应语义
是「加提示即出掩膜」：``Prompt`` → ``predict()``；逐帧串联（提示 → 记住 → 传播）由
:mod:`.propagate` 承担，与源主路径 ``init_state → add_new_points_or_box →
propagate_in_video`` 的三段一一对应。

**约定（三条，缺一条就是实现坏了）**：

1. ``image`` 一律是 ``HxWx3`` ``uint8``、**RGB** 通道序；通道序转换只在读帧处做一次，
   下游一律不猜。
2. 返回的 ``MaskPrediction.mask`` 与原图**同尺寸同坐标系**的 ``bool`` 数组；不允许
   内部下采样后放大来冒充原分辨率（坐标一旦平移，点/框提示就指错了地方）。
3. 提示坐标以**原图像素**为单位，允许浮点；框为 ``(x0, y0, x1, y1)``，超出画面时
   由实现自行裁剪而不是报错。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "Point",
    "Prompt",
    "MaskPrediction",
    "PerPointMaskPredictor",
    "mask_bbox",
    "mask_centroid",
    "mask_iou",
]


@dataclass(frozen=True)
class Point:
    """一个提示点。``label=1`` 为正点（属于目标），``label=0`` 为负点（不属于目标）。"""

    x: float
    y: float
    label: int = 1

    def as_tuple(self) -> tuple[float, float, int]:
        return (self.x, self.y, self.label)


@dataclass(frozen=True)
class Prompt:
    """一次提示：若干点 + 可选框。点为空的框提示也合法。"""

    points: tuple[Point, ...] = ()
    box: tuple[float, float, float, float] | None = None

    @classmethod
    def from_point(cls, x: float, y: float, label: int = 1) -> "Prompt":
        return cls(points=(Point(x, y, label),))

    @classmethod
    def from_box(
        cls,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        point: tuple[float, float] | None = None,
    ) -> "Prompt":
        points = (Point(point[0], point[1], 1),) if point is not None else ()
        return cls(points=points, box=(x0, y0, x1, y1))

    @property
    def positive_points(self) -> tuple[Point, ...]:
        return tuple(point for point in self.points if point.label == 1)

    @property
    def negative_points(self) -> tuple[Point, ...]:
        return tuple(point for point in self.points if point.label == 0)

    def clipped_box(self, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
        """把框裁进画面；退化（宽或高 ≤0）时返回 ``None``。"""
        if self.box is None:
            return None
        height, width = shape[0], shape[1]
        x0 = int(max(0, min(width, min(self.box[0], self.box[2]))))
        x1 = int(max(0, min(width, max(self.box[0], self.box[2]))))
        y0 = int(max(0, min(height, min(self.box[1], self.box[3]))))
        y1 = int(max(0, min(height, max(self.box[1], self.box[3]))))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)


@dataclass(frozen=True)
class MaskPrediction:
    """一次预测的结果。``score`` 是**形状规整度**（0–1），不是概率。"""

    mask: np.ndarray
    score: float = 0.0

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def is_empty(self) -> bool:
        return self.area == 0

    @property
    def bbox(self) -> tuple[int, int, int, int] | None:
        return mask_bbox(self.mask)


@runtime_checkable
class PerPointMaskPredictor(Protocol):
    """「给点/框 → 出掩膜」的统一协议。

    任何实现都必须满足：同一张图 + 同一个提示，输出掩膜的坐标系与输入图一致；
    找不到目标时返回**空列表**（而不是一张全零图），由上层决定标记遮挡还是别的处理。
    """

    name: str

    def predict(self, image: np.ndarray, prompt: Prompt) -> list[MaskPrediction]:
        """单个提示 → 该提示下的全部候选掩膜（按 score 降序）。"""

    def predict_batch(
        self, image: np.ndarray, prompts: Sequence[Prompt]
    ) -> list[list[MaskPrediction]]:
        """批量提示。默认逐条走 ``predict``，实现可覆写以复用图像侧的计算。"""


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """掩膜的紧致外接框 ``(x0, y0, x1, y1)``；空掩膜返回 ``None``。"""
    if not np.any(mask):
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """掩膜的像素质心 ``(x, y)``；空掩膜返回 ``None``。"""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """两掩膜的交并比。面积均为 0 时按「无交集」记 0，不返回 1。"""
    intersection = np.count_nonzero(left & right)
    if intersection == 0:
        return 0.0
    union = np.count_nonzero(left | right)
    return float(intersection) / float(union) if union else 0.0


class BatchCapablePredictor:
    """给 ``PerPointMaskPredictor`` 补一个可复用的批量默认实现。"""

    name = "batch_default"

    def predict(self, image: np.ndarray, prompt: Prompt) -> list[MaskPrediction]:
        raise NotImplementedError

    def predict_batch(
        self, image: np.ndarray, prompts: Sequence[Prompt]
    ) -> list[list[MaskPrediction]]:
        return [self.predict(image, prompt) for prompt in prompts]