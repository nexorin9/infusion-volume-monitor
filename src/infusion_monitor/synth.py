"""合成输液架短片渲染器 —— **数据层，不是产品代码路径**。

产品运行时**不 import 本模块**：这里的每一帧都是程序画出来的，不含任何患者信息，
也不代表真实输液袋的成像。真实院内短片必须经医院授权采集；本轮以「合成 + 公开
VOS 数据集」替代自动化验收所需的数据（见 ``docs/synthetic-data.md``）——**替身不是
最终验收边界**。

为什么要有它
------------

几何核给出的是面积比，标定带把面积比换成毫升。这条链路上每一步都会引入误差，而
「换完液到底还剩多少毫升」是要护士照着做事的数。本模块提供**已知真值**的短片：
给定液位，渲染出来的液体区域像素数与真实体积分数都能算出来，于是

    分割误差（面积比）  与  换算误差（毫升）  可以分开量、分开归因。

几何模型
--------

袋腔用**偏心超椭圆**描述：横向半宽剖面 ``(1 - |u|^p)^(1/p)``，``u`` 以袋腔中部
（略偏下，0.45）为中心归一。``p`` 越小越接近椭圆（软袋鼓胀），越大越接近方袋/硬瓶。

真实体积分数按未变形腔体的宽度剖面积分给出：

    体积分数(level) = ∫₀^level w(y) dy / ∫₀¹ w(y) dy

而**投影**面积比另带两个观感修正，它们都不会增加袋里真实的液体量：

* ``perspective`` —— 相机略低于袋中，袋顶比袋底窄（纯光学效应）；
* ``bulge`` —— 软袋充盈后下部被液体顶鼓（袋体形变，体积变化由袋型标定吸收）。

因此「量到的面积比」与「真实体积分数」之间存在**系统性偏差**，且偏差随袋型与液位
变化——这正是标定要走分段线性带、而不是一条 ``面积比 × 容量`` 直线的物理原因。
``tests/test_calibration_regression.py`` 用本渲染器把这层关系量出来。

目录约定
--------

真值按 ``{scene_id}/{object_id}/{frame_id}.png`` 落盘（对齐源产品评测器读取的
「一段视频 / 一个对象 / 一帧一张二值掩膜」GT 目录规范，见
``github_ref/sam2/sav_dataset/sav_evaluator.py`` 的 ``GT_ROOT`` 结构），使后续
IoU / J&F 对照可以直接消费本脚本的产物，不必再转一次格式。
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "BagTypeProfile",
    "BagPlan",
    "SceneSpec",
    "CavityModel",
    "RenderedScene",
    "BAG_TYPE_CATALOG",
    "VARIANTS",
    "bag_profile",
    "cavity_model",
    "level_at",
    "render_scene",
    "hold_level",
    "encode_jpeg",
    "decode_jpeg",
    "frame_zip",
    "write_scene",
    "scene_manifest",
    "spec_from_manifest",
    "manifest_from_spec",
    "EXAMPLE_CAPTURED_AT",
    "SYNTH_BASE_TIME",
]

#: 合成数据的固定基准时刻（**不是**真实拍摄时间；仅用于 zip 成员时间戳，保证可复现）。
SYNTH_BASE_TIME = datetime(2020, 1, 1, 0, 0, 0)

#: manifest 里 ``captured_at`` 的占位值——不写具体日期，避免被误读成真实拍摄时间。
EXAMPLE_CAPTURED_AT = "<EXAMPLE_CAPTURED_AT>"

#: 宽度剖面的积分网格分辨率（分段求和，误差远小于像素量化误差）。
GRID = 2048

#: 超椭圆的中心与半高：袋腔中线略偏下（0.45），底部近平、顶部收成肩。
SHOULDER_CENTER = 0.45
SHOULDER_HALF = 0.55

#: 袋体在画面中的纵向占位（顶部 y 比例、高度比例）。
BAG_TOP_RATIO = 0.09
BAG_HEIGHT_RATIO = 0.76

#: 渲染用色。液体饱和度必须过几何核的 ``saturation_min=96``；袋体刻意低饱和。
LIQUID_RGB = (120, 175, 225)
BAG_BODY_RGB = (205, 214, 219)
LABEL_RGB = (232, 232, 228)
HIGHLIGHT_RGB = (255, 255, 255)
HAND_RGB = (198, 166, 142)
TUBE_RGB = (196, 200, 204)

#: 夜间变体的逐通道衰减（偏蓝，暗光下的病区灯光）。
NIGHT_GAIN = (0.42, 0.46, 0.55)

#: 支持的拍摄变体。``fixed`` 是基线，其余覆盖 task 8 的不利拍法与 task 20 的失败模式集。
VARIANTS = (
    "fixed",
    "handheld",
    "highlight",
    "deform",
    "hand_occlusion",
    "night",
    "motion_blur",
    "tilt",
    "out_of_frame",
)

#: ``tilt`` 变体的倾角（度）：手机歪这么多，液面线在画面里就跟着歪这么多。
TILT_DEG = 20.0
#: ``out_of_frame`` 变体把整幅画面下移这么多（占画面高的比例），袋子底部被切出画面。
OUT_OF_FRAME_SHIFT_RATIO = 0.28
#: ``motion_blur`` 变体的水平拖影长度（像素）。
MOTION_BLUR_LENGTH = 13


@dataclass(frozen=True)
class BagTypeProfile:
    """一个袋型的腔体几何与观感参数。

    ``capacity_ml`` 是标签容量，``dead_volume_ml`` 是管路与滴斗残留（够不着的那部分），
    两者之差才是 ``usable_capacity_ml``——标定带里的毫升数按袋内液体量登记，不再二次
    扣减残留。
    """

    bag_type_code: str
    label_zh: str
    capacity_ml: float
    dead_volume_ml: float
    shape_p: float
    perspective: float
    bulge: float
    width_ratio: float

    @property
    def usable_capacity_ml(self) -> float:
        return max(self.capacity_ml - self.dead_volume_ml, 0.0)


#: 常见院内袋型（外形与容量取公开常识，不含任何具体批号/厂商/效期）。
BAG_TYPE_CATALOG: tuple[BagTypeProfile, ...] = (
    BagTypeProfile("TYPE-A", "500 ml 软袋（圆角方袋）", 500.0, 20.0, 4.0, 0.20, 0.055, 0.62),
    BagTypeProfile("TYPE-B", "1000 ml 软袋（长袋）", 1000.0, 30.0, 3.2, 0.24, 0.070, 0.64),
    BagTypeProfile("TYPE-C", "250 ml 硬瓶（方瓶）", 250.0, 15.0, 6.0, 0.12, 0.000, 0.48),
    BagTypeProfile("TYPE-D", "250 ml 软袋（扁袋）", 250.0, 12.0, 4.6, 0.28, 0.090, 0.72),
)

_BY_CODE = {profile.bag_type_code: profile for profile in BAG_TYPE_CATALOG}


def bag_profile(bag_type_code: str) -> BagTypeProfile:
    """按袋型代码取档；未知代码直接报错，不静默回落到默认袋型。"""
    try:
        return _BY_CODE[bag_type_code]
    except KeyError as exc:  # pragma: no cover - 防御性分支
        known = ", ".join(sorted(_BY_CODE))
        raise KeyError(f"未知袋型 {bag_type_code!r}，可选：{known}") from exc


@dataclass(frozen=True)
class BagPlan:
    """一个挂位上的袋子：挂哪、什么袋型、液位怎么走、挂了什么附件。"""

    bag_seq: str
    slot_index: int
    bag_type_code: str
    fill_start: float
    fill_end: float
    #: 袋口下接管路与滴斗。**空滴斗**——接上但没液——是常见的误判来源：
    #: 袋体与滴斗连成同一块前景，腔体估计会被撑大、面积比被压低。
    drip_chamber: bool = False
    #: 护士贴的输液标签（一小块浅色贴纸，会在液面上打出孔洞）。
    label: bool = False

    @property
    def profile(self) -> BagTypeProfile:
        return bag_profile(self.bag_type_code)


@dataclass(frozen=True)
class SceneSpec:
    """一段合成短片的完整参数。**全部参数写进 manifest，因此可逐位复现。**"""

    scene_id: str
    bags: tuple[BagPlan, ...]
    frame_count: int = 12
    width: int = 480
    height: int = 640
    seed: int = 20261022
    variant: str = "fixed"
    #: ``(bag_seq, 起始帧, 结束帧)``：该区间用手的形状挡住这一袋（只改画面，不改真值）。
    occlude_frames: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.bags:
            raise ValueError("scene 至少要有一个袋子")
        if self.frame_count < 1:
            raise ValueError(f"frame_count 至少 1，收到 {self.frame_count}")
        if self.variant not in VARIANTS:
            raise ValueError(f"未知拍摄变体 {self.variant!r}，可选：{', '.join(VARIANTS)}")
        slots = sorted(plan.slot_index for plan in self.bags)
        if slots != list(range(len(self.bags))):
            raise ValueError(f"slot_index 必须是从 0 开始的连续整数，收到 {slots}")
        if len(self.bags) > 6:
            raise ValueError(f"单架最多 6 袋，收到 {len(self.bags)}")


def level_at(plan: BagPlan, frame_count: int, frame_index: int) -> float:
    """该袋在该帧的**真值液面高度**（袋腔高度占比，0=空、1=满）。"""
    if frame_count <= 1:
        return plan.fill_start
    t = frame_index / (frame_count - 1)
    return max(0.0, min(1.0, plan.fill_start + (plan.fill_end - plan.fill_start) * t))


def hold_level(spec: SceneSpec, bag_seq: str, level: float) -> SceneSpec:
    """把某袋钉在指定液位上（其余袋保持不动），用于**逐液位**取样。

    标定样例要在一组固定液位上量「面积比 ↔ 真实毫升」，短片里液位一路下降会把这组
    配对关系糊掉；钉住之后每个液位都是一段静止画面，回归才可复现。
    """
    bags = tuple(
        replace(plan, fill_start=level, fill_end=level)
        if plan.bag_seq == bag_seq
        else plan
        for plan in spec.bags
    )
    return replace(spec, bags=bags, variant="fixed", occlude_frames=())


class CavityModel:
    """袋腔几何：给定液位，算出真实体积分数与投影面积比。"""

    __slots__ = ("profile", "h", "nominal", "taper", "_total")

    def __init__(self, profile: BagTypeProfile, *, grid: int = GRID) -> None:
        h = (np.arange(grid) + 0.5) / grid
        u = np.clip((h - SHOULDER_CENTER) / SHOULDER_HALF, -1.0, 1.0)
        p = profile.shape_p
        nominal = np.power(np.maximum(0.0, 1.0 - np.power(np.abs(u), p)), 1.0 / p)
        self.profile = profile
        self.h = h
        self.nominal = nominal
        #: 透视收窄：袋顶比袋底窄 ``perspective``（纯光学，不改变真实体积）。
        self.taper = 1.0 + profile.perspective * (0.5 - h)
        self._total = float(nominal.sum())

    # ── 真实量 ─────────────────────────────────────────────────────────
    def volume_fraction(self, level: float) -> float:
        """真实体积分数（未变形腔体的宽度剖面积分比）。"""
        cut = int(round(max(0.0, min(1.0, float(level))) * len(self.h)))
        return float(self.nominal[:cut].sum() / self._total)

    def true_volume_ml(self, level: float) -> float:
        return self.volume_fraction(level) * self.profile.usable_capacity_ml

    # ── 观测量 ─────────────────────────────────────────────────────────
    def projected_width(self, level: float) -> np.ndarray:
        """逐高度的投影半宽（相对袋腔宽度）。

        ``bulge`` 建模「软袋充盈后下部被顶鼓」：鼓出的幅度与该处**上方**的液柱高度
        成正比，下部鼓得多、液面之上不鼓。这项形变改变的是**观感**几何，真实毫升仍
        按未变形腔体积分给出——两者之差正是标定带要吸收的系统偏差。
        """
        level = max(0.0, min(1.0, float(level)))
        head = np.maximum(0.0, level - self.h)
        return self.nominal * self.taper * (1.0 + self.profile.bulge * head)

    def projected_area_ratio(self, level: float) -> float:
        """理论投影面积比（完美分割器量到的值）。"""
        width = self.projected_width(level)
        cut = int(round(max(0.0, min(1.0, float(level))) * len(self.h)))
        total = float(width.sum())
        return float(width[:cut].sum() / total) if total > 0 else 0.0

    def width_at(self, heights: np.ndarray, level: float) -> np.ndarray:
        """按任意高度数组插值出投影半宽（渲染与真值掩膜共用同一条剖面）。"""
        return np.interp(heights, self.h, self.projected_width(level))


_MODEL_CACHE: dict[str, CavityModel] = {}


def cavity_model(profile: BagTypeProfile) -> CavityModel:
    """袋型 → 腔体模型（按袋型缓存，模型本身是只读的）。"""
    model = _MODEL_CACHE.get(profile.bag_type_code)
    if model is None or model.profile is not profile:
        model = CavityModel(profile)
        _MODEL_CACHE[profile.bag_type_code] = model
    return model


# ── 渲染 ────────────────────────────────────────────────────────────────


def _bag_box(spec: SceneSpec, plan: BagPlan, frame_index: int) -> tuple[int, int, int, int]:
    """袋体外接框。``deform`` 变体下逐帧摆动与鼓缩（真值掩膜跟着一起变，仍是精确的）。"""
    column = spec.width / len(spec.bags)
    centre_x = (plan.slot_index + 0.5) * column
    box_w = column * plan.profile.width_ratio
    top = spec.height * BAG_TOP_RATIO
    height = spec.height * BAG_HEIGHT_RATIO
    bottom = top + height

    if spec.variant == "deform":
        rng = np.random.default_rng(spec.seed * 131 + frame_index * 17 + plan.slot_index)
        wobble = float(rng.uniform(-1.0, 1.0))
        box_w *= 1.0 + 0.055 * wobble
        height *= 1.0 + 0.030 * wobble
        centre_x += 0.018 * wobble * column
        top += spec.height * 0.006 * wobble
    centre_x += column * 0.012 * math.sin(2 * math.pi * (frame_index / 11.0 + 0.3 * plan.slot_index))
    top += spec.height * 0.004 * math.sin(2 * math.pi * (frame_index / 7.0 + 0.2 * plan.slot_index))

    x0 = int(round(centre_x - box_w / 2.0))
    x1 = int(round(centre_x + box_w / 2.0))
    y0 = int(round(top))
    y1 = int(round(top + height))
    x0 = max(0, min(x0, spec.width - 2))
    x1 = max(x0 + 2, min(x1, spec.width))
    y0 = max(0, min(y0, spec.height - 4))
    y1 = max(y0 + 4, min(y1, spec.height))
    return x0, y0, x1, y1


def _height_map(y0: int, y1: int) -> np.ndarray:
    """把行号映射成袋腔高度比例：底行为 0、顶行为 1。"""
    rows = np.arange(y0, y1)
    span = max(1, (y1 - 1) - y0)
    return (y1 - 1 - rows) / float(span)


def _draw_bag(
    canvas: np.ndarray,
    spec: SceneSpec,
    plan: BagPlan,
    frame_index: int,
    model: CavityModel,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """把一袋画到画布上，返回 ``(腔体掩膜, 液体掩膜)``（真值，按构造精确）。"""
    x0, y0, x1, y1 = _bag_box(spec, plan, frame_index)
    level = level_at(plan, spec.frame_count, frame_index)
    heights = _height_map(y0, y1)
    half = model.width_at(heights, level) * (x1 - x0) / 2.0
    dx = np.abs(np.arange(x0, x1) - (x0 + x1 - 1) / 2.0)
    cavity_local = dx[None, :] <= half[:, None]
    liquid_local = cavity_local & (heights[:, None] <= level)
    if not cavity_local.any():
        return None, None

    view = canvas[y0:y1, x0:x1]
    view[cavity_local] = BAG_BODY_RGB
    view[liquid_local] = LIQUID_RGB

    # 输液标签：贴在袋体上部的一小块浅色贴纸，会在液面上打出一个小孔洞。
    if plan.label:
        lw, lh = max(6, int((x1 - x0) * 0.22)), max(8, int((y1 - y0) * 0.075))
        lx = (x0 + x1) // 2 - lw // 2
        ly = y0 + int((y1 - y0) * 0.16)
        canvas[ly : ly + lh, lx : lx + lw] = LABEL_RGB

    # 镜面反光高光：落在液体区里的亮斑。几何核靠孔洞填充吸收它（面积 ≤ max_hole_area）。
    if spec.variant == "highlight" and level > 0.35:
        hy0 = y0 + int((y1 - y0) * (1.0 - level) * 0.9)
        hy1 = min(y1 - 2, hy0 + max(6, int((y1 - y0) * 0.05)))
        hx0 = x0 + max(3, int((x1 - x0) * 0.30))
        hx1 = min(x1 - 3, hx0 + max(6, int((x1 - x0) * 0.16)))
        if hy1 > hy0 and hx1 > hx0:
            cv2.ellipse(
                canvas,
                ((hx0 + hx1) // 2, (hy0 + hy1) // 2),
                (max(3, (hx1 - hx0) // 2), max(2, (hy1 - hy0) // 2)),
                0,
                0,
                360,
                HIGHLIGHT_RGB,
                -1,
            )

    # 滴斗与管路：接在袋口下方，**空管**（无液体）。它与袋体连成同一块前景，
    # 会把腔体估计撑大——这是本模块刻意保留的已知偏差，见 docs/synthetic-data.md。
    if plan.drip_chamber:
        tube_w = max(3, int((x1 - x0) * 0.06))
        tx = (x0 + x1) // 2 - tube_w // 2
        chamber_w = max(8, int((x1 - x0) * 0.26))
        cx = (x0 + x1) // 2 - chamber_w // 2
        canvas[y1 - 2 : min(spec.height, y1 + 14), tx : tx + tube_w] = TUBE_RGB
        cy1 = min(spec.height, y1 + int((spec.height - y1) * 0.92))
        canvas[min(spec.height, y1 + 6) : cy1, cx : cx + chamber_w] = TUBE_RGB

    cavity = np.zeros(canvas.shape[:2], dtype=bool)
    liquid = np.zeros(canvas.shape[:2], dtype=bool)
    cavity[y0:y1, x0:x1] = cavity_local
    liquid[y0:y1, x0:x1] = liquid_local
    return cavity, liquid


def _handheld_affine(spec: SceneSpec, frame_index: int) -> np.ndarray:
    """手持抖动：平移 + 微小旋转。同一仿射同时作用在画面与真值掩膜上，两者不错位。"""
    rng = np.random.default_rng(spec.seed * 977 + frame_index * 53)
    angle = float(rng.uniform(-1.4, 1.4))
    dx = float(rng.uniform(-3.5, 3.5))
    dy = float(rng.uniform(-3.0, 3.0))
    matrix = cv2.getRotationMatrix2D((spec.width / 2.0, spec.height / 2.0), angle, 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return matrix


def _apply_affine_mask(mask: np.ndarray, matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    warped = cv2.warpAffine(
        mask.astype(np.uint8) * 255,
        matrix,
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return warped > 127


def _tilt_affine(spec: SceneSpec) -> np.ndarray:
    """固定倾角的手机拍摄：绕画面中心旋转 ``TILT_DEG``，画面与真值掩膜同步变换。"""
    return cv2.getRotationMatrix2D(
        (spec.width / 2.0, spec.height / 2.0), TILT_DEG, 1.0
    )


def _out_of_frame_affine(spec: SceneSpec) -> np.ndarray:
    """站得太近 / 举得不够高：整幅画面下移，袋子底部被切出画面。"""
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, spec.height * OUT_OF_FRAME_SHIFT_RATIO]])
    return matrix


def _motion_blur_kernel() -> np.ndarray:
    """水平拖影核（长度 ``MOTION_BLUR_LENGTH``，中间权重高）。"""
    length = MOTION_BLUR_LENGTH | 1
    axis = np.arange(length, dtype=np.float64) - (length - 1) / 2.0
    weights = np.exp(-0.5 * (axis / max(1.0, length / 6.0)) ** 2)
    kernel = np.zeros((1, length), dtype=np.float64)
    kernel[0] = weights / weights.sum()
    return kernel


def _draw_hand(canvas: np.ndarray, plan: BagPlan, box: tuple[int, int, int, int]) -> None:
    """在袋体侧下方画一只手（肤色，饱和度低于几何核门限，不会被当成液体）。"""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    palm = ((x0 + x1) // 2 + int(width * 0.22), y0 + int(height * 0.86))
    cv2.ellipse(canvas, palm, (max(6, int(width * 0.30)), max(8, int(height * 0.12))), 12, 0, 360, HAND_RGB, -1)
    for index in range(3):
        fx = palm[0] - int(width * 0.14) + index * int(width * 0.13)
        fy = palm[1] - int(height * 0.10) - index * int(height * 0.012)
        cv2.ellipse(canvas, (fx, fy), (max(3, int(width * 0.055)), max(6, int(height * 0.10))), 22, 0, 360, HAND_RGB, -1)


@dataclass
class RenderedScene:
    """一段渲染结果：画面、真值掩膜与逐帧真值清单。"""

    spec: SceneSpec
    frames: list[np.ndarray]
    cavity_masks: dict[str, list[np.ndarray]]
    liquid_masks: dict[str, list[np.ndarray]]

    def truth_rows(self, bag_seq: str) -> list[dict[str, Any]]:
        """逐帧真值：真实液面高度、体积分数、真实毫升、理论面积比与像素计数。"""
        plan = next(item for item in self.spec.bags if item.bag_seq == bag_seq)
        model = cavity_model(plan.profile)
        rows: list[dict[str, Any]] = []
        for index in range(self.spec.frame_count):
            cavity = self.cavity_masks[bag_seq][index]
            liquid = self.liquid_masks[bag_seq][index]
            cavity_px = int(np.count_nonzero(cavity))
            liquid_px = int(np.count_nonzero(liquid))
            level = level_at(plan, self.spec.frame_count, index)
            rows.append(
                {
                    "frame_index": index,
                    "true_height_fraction": round(level, 6),
                    "true_volume_fraction": round(model.volume_fraction(level), 6),
                    "true_volume_ml": round(model.true_volume_ml(level), 2),
                    "cavity_area_px": cavity_px,
                    "liquid_area_px": liquid_px,
                    "projected_area_ratio": round(liquid_px / cavity_px, 6) if cavity_px else 0.0,
                    "hand_occluded": self._is_occluded(bag_seq, index),
                }
            )
        return rows

    def _is_occluded(self, bag_seq: str, frame_index: int) -> bool:
        return any(
            target == bag_seq and start <= frame_index < end
            for target, start, end in self.spec.occlude_frames
        )


def render_scene(spec: SceneSpec) -> RenderedScene:
    """渲染整段短片。**相同 spec 必得相同结果**（所有随机量都由 ``spec.seed`` 派生）。"""
    height, width = spec.height, spec.width
    frames: list[np.ndarray] = []
    cavity_masks: dict[str, list[np.ndarray]] = {plan.bag_seq: [] for plan in spec.bags}
    liquid_masks: dict[str, list[np.ndarray]] = {plan.bag_seq: [] for plan in spec.bags}

    for frame_index in range(spec.frame_count):
        rng = np.random.default_rng(spec.seed * 7919 + frame_index)
        base = rng.integers(18, 40, size=(height, width, 1), dtype=np.uint8)
        canvas = np.repeat(base, 3, axis=2)
        per_frame_cavity: dict[str, np.ndarray | None] = {}
        per_frame_liquid: dict[str, np.ndarray | None] = {}

        for plan in spec.bags:
            model = cavity_model(plan.profile)
            per_frame_cavity[plan.bag_seq], per_frame_liquid[plan.bag_seq] = _draw_bag(
                canvas, spec, plan, frame_index, model
            )

        if spec.variant == "hand_occlusion":
            for plan in spec.bags:
                if any(
                    target == plan.bag_seq and start <= frame_index < end
                    for target, start, end in spec.occlude_frames
                ):
                    _draw_hand(canvas, plan, _bag_box(spec, plan, frame_index))

        if spec.variant == "night":
            noise = rng.normal(0.0, 2.6, size=(height, width, 1)).astype(np.float32)
            canvas = np.clip(
                canvas.astype(np.float32) * np.asarray(NIGHT_GAIN, dtype=np.float32) + noise,
                0,
                255,
            ).astype(np.uint8)

        if spec.variant in ("handheld", "tilt", "out_of_frame"):
            matrix = {
                "handheld": lambda: _handheld_affine(spec, frame_index),
                "tilt": lambda: _tilt_affine(spec),
                "out_of_frame": lambda: _out_of_frame_affine(spec),
            }[spec.variant]()
            canvas = cv2.warpAffine(canvas, matrix, (width, height), borderValue=(26, 26, 26))
            for plan in spec.bags:
                for table in (per_frame_cavity, per_frame_liquid):
                    original = table[plan.bag_seq]
                    table[plan.bag_seq] = (
                        None if original is None else _apply_affine_mask(original, matrix, (height, width))
                    )

        # 拖影只改画面观感：真实液位没变，真值掩膜保持原样（回归仍测「产品读到的那张糊图」）。
        if spec.variant == "motion_blur":
            canvas = cv2.filter2D(canvas, -1, _motion_blur_kernel())

        frames.append(canvas)
        for plan in spec.bags:
            frame_cavity = per_frame_cavity[plan.bag_seq]
            frame_liquid = per_frame_liquid[plan.bag_seq]
            if frame_cavity is None:
                frame_cavity = np.zeros((height, width), dtype=bool)
            if frame_liquid is None:
                frame_liquid = np.zeros((height, width), dtype=bool)
            cavity_masks[plan.bag_seq].append(frame_cavity)
            liquid_masks[plan.bag_seq].append(frame_liquid)

    return RenderedScene(spec, frames, cavity_masks, liquid_masks)


# ── 编解码与打包 ────────────────────────────────────────────────────────


def encode_jpeg(frame: np.ndarray, *, quality: int = 88) -> bytes:
    """按上传链路的口径编码（JPEG 有损）。回归测的是**产品实际读到的帧**，不是原图。"""
    buffer = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def decode_jpeg(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def frame_zip(
    rendered: RenderedScene,
    *,
    start: datetime = SYNTH_BASE_TIME,
    step_seconds: float = 0.5,
    prefix: str = "frame_",
) -> bytes:
    """把画面打成可上传的 zip（成员 ``frame_%05d.jpg``，时间戳单调递增）。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, frame in enumerate(rendered.frames):
            moment = start + timedelta(seconds=step_seconds * index)
            info = zipfile.ZipInfo(f"{prefix}{index:05d}.jpg", date_time=moment.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, encode_jpeg(frame))
    return buffer.getvalue()


def write_scene(
    out_dir: Path | str,
    rendered: RenderedScene,
    *,
    zip_name: str = "frames.zip",
) -> Path:
    """落盘一套可复现的产物。

    ``RGB/`` 画面帧 · ``GT/{bag_seq}/`` 逐袋真值二值掩膜 · ``frames.zip`` 可直接上传
    · ``manifest.json`` 真值清单。GT 目录沿用 ``{scene_id}/{object_id}/{frame_id}.png``。
    """
    root = Path(out_dir) / rendered.spec.scene_id
    (root / "RGB").mkdir(parents=True, exist_ok=True)
    for plan in rendered.spec.bags:
        (root / "GT" / plan.bag_seq).mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(rendered.frames):
        (root / "RGB" / f"frame_{index:05d}.jpg").write_bytes(encode_jpeg(frame))
    for plan in rendered.spec.bags:
        for index in range(rendered.spec.frame_count):
            mask = rendered.liquid_masks[plan.bag_seq][index]
            cv2.imwrite(
                str(root / "GT" / plan.bag_seq / f"{index:05d}.png"),
                (mask.astype(np.uint8) * 255),
            )
    (root / zip_name).write_bytes(frame_zip(rendered))
    (root / "manifest.json").write_text(
        json.dumps(scene_manifest(rendered), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def manifest_from_spec(spec: SceneSpec, frame_count: int | None = None) -> dict[str, Any]:
    """不渲染也能给出参数级 manifest（供 ``--reproduce`` 与 fixture 用）。"""
    return {
        "scene_id": spec.scene_id,
        "synthetic": True,
        "synthetic_notice": (
            "本清单由程序合成，不含患者信息；真实院内短片须经医院授权采集。"
            "合成数据是本轮自动化验收的替身，不是最终验收边界。"
        ),
        "seed": spec.seed,
        "variant": spec.variant,
        "frames": spec.frame_count if frame_count is None else frame_count,
        "frame_size": [spec.width, spec.height],
        "captured_at": EXAMPLE_CAPTURED_AT,
        "frame_step_seconds": 0.5,
        "bags": [
            {
                "bag_seq": plan.bag_seq,
                "slot_index": plan.slot_index,
                "bag_type_code": plan.bag_type_code,
                "bag_type_label": plan.profile.label_zh,
                "capacity_ml": plan.profile.capacity_ml,
                "dead_volume_ml": plan.profile.dead_volume_ml,
                "usable_capacity_ml": round(plan.profile.usable_capacity_ml, 3),
                "fill_start": plan.fill_start,
                "fill_end": plan.fill_end,
                "drip_chamber": plan.drip_chamber,
                "label": plan.label,
            }
            for plan in spec.bags
        ],
        "occlude_frames": [list(item) for item in spec.occlude_frames],
    }


def scene_manifest(rendered: RenderedScene) -> dict[str, Any]:
    """完整 manifest：参数 + 逐袋逐帧真值（液位、体积、理论面积比、像素计数）。"""
    manifest = manifest_from_spec(rendered.spec)
    manifest["truth"] = {
        plan.bag_seq: rendered.truth_rows(plan.bag_seq) for plan in rendered.spec.bags
    }
    return manifest


def spec_from_manifest(manifest: Mapping[str, Any]) -> SceneSpec:
    """由 manifest 还原 spec——``--reproduce`` 与确定性回归都靠它。"""
    bags = tuple(
        BagPlan(
            bag_seq=str(item["bag_seq"]),
            slot_index=int(item["slot_index"]),
            bag_type_code=str(item["bag_type_code"]),
            fill_start=float(item["fill_start"]),
            fill_end=float(item["fill_end"]),
            drip_chamber=bool(item.get("drip_chamber", False)),
            label=bool(item.get("label", False)),
        )
        for item in manifest["bags"]
    )
    return SceneSpec(
        scene_id=str(manifest["scene_id"]),
        bags=bags,
        frame_count=int(manifest["frames"]),
        width=int(manifest["frame_size"][0]),
        height=int(manifest["frame_size"][1]),
        seed=int(manifest["seed"]),
        variant=str(manifest.get("variant", "fixed")),
        occlude_frames=tuple(
            (str(item[0]), int(item[1]), int(item[2])) for item in manifest.get("occlude_frames", [])
        ),
    )