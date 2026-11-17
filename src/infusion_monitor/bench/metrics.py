"""逐帧 IoU / 区域相似度 J / 轮廓准确度 F / J&F 计算器。

**口径来源**。``github_ref/sam2/sav_dataset/sav_evaluator.py:11-13`` 明确写明该评测器
转述自两套公开评测实现（vos-benchmark 与 davis2017-evaluation），并在其 15-41 行把
``GT_ROOT`` / ``PRED_ROOT`` 的目录结构约定为「``{video_id}/{frame_id}.png`` 的调色板
PNG，标注可含多个对象」——本模块的 ``load_gt_mask`` 与 :mod:`.runner` 直接沿用同一
约定，因此本项目落盘的公开子集可以被同一套口径直接消费。

**三个指标的定义**（与上述转述同源）：

``IoU``
    交并比。本项目自带的面积比曲线之外的**几何**口径；单帧单对象时与 J 同值，
    多对象逐对象累计时才有区别，故两处都给。

``J``（region similarity，区域相似度）
    ``|P ∩ G| / |P ∪ G|``。对掩膜质量敏感——错在物体内部与错在边界同样扣分。

``F``（contour accuracy，轮廓准确度）
    轮廓精度与召回的调和平均：把预测与真值的**边界像素**互相投影，落进容差
    ``bound_th × 图像对角线`` 内即算命中。容差默认 ``0.008``（``BOUND_TH``），
    与 DAVIS 官方评测一致。只看得见轮廓，对「整块糊掉」不敏感。

``J&F``
    ``(J + F) / 2``。两个口径同时报告，避免单看一个被钻空子。

**实现说明**：官方轮廓评测用逐像素双重循环找最近边界像素，在 480p × 90 帧 × 10 序列
的规模下是数分钟量级的纯 Python 循环。本模块改用 OpenCV 精确欧氏距离变换求「到最近
边界像素的距离」，语义与逐像素最近邻完全等价（``DIST_L2`` + 精确 mask size 不存在
``maskSize`` 近似），并有对拍测试守住这条等价性。**这是性能改写，不是口径改写。**
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: 轮廓容差系数（乘图像对角线）。DAVIS 官方评测默认值。
BOUND_TH = 0.008

#: 空掩膜的分数语义。真值为空时，**空预测算满分**（没东西可错），非空预测算 0。
#: 这一条是评测协议里最容易写错的地方：写反了会让「什么都没检出」反而拿高分。
EMPTY_TRUE_POSITIVE = 1.0
EMPTY_TRUE_NEGATIVE = 0.0


def _as_mask(value: np.ndarray) -> np.ndarray:
    """把任意标注图折成 ``bool`` 掩膜，顺带挡掉形状与维度不对的输入。"""
    array = np.asarray(value)
    if array.ndim == 3:
        # 调色板 PNG 若被按 RGB 读进来，退化成「任一通道非零即前景」；
        # 调用方拿不准该用哪种读法时，这条兜底至少不会静默算错。
        array = array.any(axis=2)
    if array.ndim != 2:
        raise ValueError(f"期望二维掩膜，实际形状 {array.shape}")
    return array.astype(bool)


def jaccard(pred: np.ndarray, truth: np.ndarray) -> float:
    """交并比 ``|P ∩ G| / |P ∪ G|``。空对空记 1.0。"""
    p = _as_mask(pred)
    g = _as_mask(truth)
    if p.shape != g.shape:
        raise ValueError(f"掩膜形状不一致：pred {p.shape} vs gt {g.shape}")
    union = int(np.count_nonzero(p | g))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(p & g) / union)


#: 区域相似度与 IoU 在单对象单帧上同定义；此处单列一个名字，
#: 是为了让 J&F 的口径在代码里读起来就是评测协议里的那个 J。
region_similarity = jaccard


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """取掩膜的一像素宽边界环（右邻/下邻异或的并集，四边补齐）。

    这是官方 ``_seg2bmap`` 的同一判据：一个像素属于边界，当且仅当它自己的前景状态
    与**右邻或下邻**不同。两个方向必须用两张独立的移位数组——共用一张会让第二次
    赋值把第一次的结果整个盖掉，右边界就此消失（实测边界只剩上下两条边）。
    最后一行/列要单独收口，否则右下角漏判。
    """
    m = _as_mask(mask)
    east = np.zeros_like(m)
    south = np.zeros_like(m)
    east[:, :-1] = m[:, 1:]
    south[:-1, :] = m[1:, :]
    boundary = np.logical_xor(m, east) | np.logical_xor(m, south)
    # 最后一行没有下邻、最后一列没有右邻：都视作与背景相邻。
    boundary[-1, :] = np.logical_xor(m[-1, :], east[-1, :])
    boundary[:, -1] = np.logical_xor(m[:, -1], south[:, -1])
    return boundary


def _distance_to(mask_points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """每个像素到 ``mask_points`` 中最近像素的精确欧氏距离。

    ``cv2.distanceTransform`` 求的是「到最近**零**像素」的距离，所以先把待测点置零、
    其余置一。``DIST_L2`` + ``cv2.DIST_MASK_PRECISE`` 是精确解，不带 3×3/5×5 掩模近似。
    """
    src = np.ones(shape, dtype=np.uint8)
    src[mask_points] = 0
    import cv2  # 局部导入：本模块的纯数值部分不应拖起 OpenCV

    return cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def db_eval_boundary(pred: np.ndarray, truth: np.ndarray, bound_th: float = BOUND_TH) -> float:
    """轮廓准确度 F：边界像素在容差内互相命中的调和平均。

    返回 ``F``；若需要分别看精度与召回，用 :func:`boundary_precision_recall`。
    """
    precision, recall = boundary_precision_recall(pred, truth, bound_th)
    if precision + recall == 0.0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def boundary_precision_recall(
    pred: np.ndarray, truth: np.ndarray, bound_th: float = BOUND_TH
) -> tuple[float, float]:
    """(轮廓精度, 轮廓召回)。

    精度 = 预测边界像素中，容差内能对上真值边界的比例；召回对称。
    两边边界都为空记 (1.0, 1.0)；只有一边为空记 (0.0, 0.0)——「有一个有轮廓，另一个
    没有」不可能互相命中。
    """
    p = _as_mask(pred)
    g = _as_mask(truth)
    if p.shape != g.shape:
        raise ValueError(f"掩膜形状不一致：pred {p.shape} vs gt {g.shape}")
    p_bound = _boundary_mask(p)
    g_bound = _boundary_mask(g)
    p_count = int(np.count_nonzero(p_bound))
    g_count = int(np.count_nonzero(g_bound))
    if p_count == 0 and g_count == 0:
        return EMPTY_TRUE_POSITIVE, EMPTY_TRUE_POSITIVE
    if p_count == 0 or g_count == 0:
        return EMPTY_TRUE_NEGATIVE, EMPTY_TRUE_NEGATIVE

    tolerance = float(np.ceil(bound_th * float(np.linalg.norm(p.shape))))
    # 到真值边界的距离：预测边界里落进容差的即命中
    dist_to_gt = _distance_to(g_bound, p.shape)
    precision = float(np.count_nonzero(p_bound & (dist_to_gt <= tolerance)) / p_count)
    dist_to_pred = _distance_to(p_bound, p.shape)
    recall = float(np.count_nonzero(g_bound & (dist_to_pred <= tolerance)) / g_count)
    return precision, recall


@dataclass(frozen=True)
class FrameMetric:
    """单帧、单对象的评测结果。

    ``present`` 记的是**真值侧**这一帧有没有该对象——评测端据此区分「这一帧本就不该
    检出」与「该检出却漏了」，两种情况的 J 都可能是 0，但含义完全不同。
    """

    sequence: str
    frame_index: int
    obj_id: int
    iou: float
    j: float
    f: float
    present: bool

    @property
    def j_and_f(self) -> float:
        return (self.j + self.f) / 2.0

    def to_row(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "frame_index": self.frame_index,
            "obj_id": self.obj_id,
            "iou": self.iou,
            "j": self.j,
            "f": self.f,
            "j_and_f": self.j_and_f,
            "present": self.present,
        }


def frame_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    sequence: str,
    frame_index: int,
    obj_id: int = 1,
) -> FrameMetric:
    """算一帧单对象的 IoU / J / F。

    真值为空时：空预测记满分（``j=f=1``），非空预测记 0。这条门禁让「全片都检出空气」
    这种退化方案拿不到虚高的均分。
    """
    p = _as_mask(pred)
    g = _as_mask(truth)
    present = bool(np.count_nonzero(g) > 0)
    if not present:
        score = EMPTY_TRUE_POSITIVE if not np.count_nonzero(p) else EMPTY_TRUE_NEGATIVE
        return FrameMetric(sequence, frame_index, obj_id, score, score, score, False)
    return FrameMetric(
        sequence=sequence,
        frame_index=frame_index,
        obj_id=obj_id,
        iou=jaccard(p, g),
        j=region_similarity(p, g),
        f=db_eval_boundary(p, g),
        present=True,
    )


@dataclass
class SequenceMetric:
    """一个序列在一套方案下的汇总。``frames`` 保留逐帧明细，便于定位是第几帧崩的。"""

    sequence: str
    scheme: str
    frames: list[FrameMetric] = field(default_factory=list)
    occluded_frames: frozenset[int] = frozenset()

    @property
    def evaluated(self) -> list[FrameMetric]:
        """计入均分的帧：真值侧确实有对象的那些。

        真值为空的帧不进均分——把「这帧没有目标」当成一次失误，会把检测器的空转率
        混进分割质量里，两头都不准。真值侧的信息由 ``occluded_frames`` 单独承担。
        """
        return [f for f in self.frames if f.present]

    @property
    def occluded(self) -> list[FrameMetric]:
        return [f for f in self.evaluated if f.frame_index in self.occluded_frames]

    def _mean(self, rows: list[FrameMetric], attr: str) -> float:
        if not rows:
            return 0.0
        values = [getattr(f, attr) if attr != "j_and_f" else f.j_and_f for f in rows]
        return float(np.mean(values))

    @property
    def mean_iou(self) -> float:
        return self._mean(self.evaluated, "iou")

    @property
    def mean_j(self) -> float:
        return self._mean(self.evaluated, "j")

    @property
    def mean_f(self) -> float:
        return self._mean(self.evaluated, "f")

    @property
    def mean_j_and_f(self) -> float:
        return self._mean(self.evaluated, "j_and_f")

    @property
    def mean_occluded_j_and_f(self) -> float:
        return self._mean(self.occluded, "j_and_f")

    @property
    def occlusion_cover(self) -> float:
        """含遮挡序列的判定：真值对象面积掉到本序列峰值一半以下的帧占比。"""
        return len(self.occluded) / len(self.evaluated) if self.evaluated else 0.0

    def to_row(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "scheme": self.scheme,
            "frames": len(self.evaluated),
            "mean_iou": round(self.mean_iou, 4),
            "mean_j": round(self.mean_j, 4),
            "mean_f": round(self.mean_f, 4),
            "mean_j_and_f": round(self.mean_j_and_f, 4),
            "occluded_frames": len(self.occluded),
            "mean_occluded_j_and_f": round(self.mean_occluded_j_and_f, 4),
        }


def aggregate(sequences: list[SequenceMetric]) -> dict[str, object]:
    """跨序列汇总：**按帧等权**，不是各序列均分再平均。

    各序列帧数不同（90–104 帧），先按序列求均值再平均会隐式给短序列更高权重。
    这里把所有计入均分的帧摊平一起求均值，与「逐帧指标的平均」这一说法一致。
    """
    frames = [f for seq in sequences for f in seq.evaluated]
    occluded = [f for seq in sequences for f in seq.occluded]

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "sequences": len(sequences),
        "frames": len(frames),
        "mean_iou": round(mean([f.iou for f in frames]), 4),
        "mean_j": round(mean([f.j for f in frames]), 4),
        "mean_f": round(mean([f.f for f in frames]), 4),
        "mean_j_and_f": round(mean([f.j_and_f for f in frames]), 4),
        "occluded_frames": len(occluded),
        "mean_occluded_j_and_f": round(mean([f.j_and_f for f in occluded]), 4),
    }


def summarise(report: dict[str, object]) -> str:
    """把对照结果压成一张可粘贴的 markdown 表。"""
    lines = [
        "| 方案 | 序列 | 计入帧 | IoU | J | F | J&F | 遮挡帧 J&F |",
        "|------|------|--------|-----|---|---|-----|------------|",
    ]
    for row in report.get("rows", []):  # type: ignore[union-attr]
        lines.append(
            "| {scheme} | {sequence} | {frames} | {mean_iou:.4f} | {mean_j:.4f} | "
            "{mean_f:.4f} | {mean_j_and_f:.4f} | {mean_occluded_j_and_f:.4f} |".format(**row)
        )
    totals = report.get("totals", {})
    for scheme, total in totals.items():  # type: ignore[union-attr]
        lines.append(
            "| **{scheme}（全体）** | — | {frames} | {mean_iou:.4f} | {mean_j:.4f} | "
            "{mean_f:.4f} | **{mean_j_and_f:.4f}** | {mean_occluded_j_and_f:.4f} |".format(
                scheme=scheme, **total
            )
        )
    return "\n".join(lines)