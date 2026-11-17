"""程序合成帧序列（测试与本地联调用）。

**不是**真实院内短片，也不是最终验收边界：真实短片需医院授权采集，本轮以
合成 + 公开 VOS 数据集替代（见 README 与 ``docs/benchmark-vos.md``）。这里只生成
**能通过硬校验的形状正确的帧**——真实袋体渲染与真值回归由 task 8 的
``scripts/synthesize_rack.py`` 承担。

样例里的一切标识都用 ``<EXAMPLE_*>`` / ``EXAMPLE-*`` 占位，不编造可被误认为
真实的住院号、处方号、批号或日期。
"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "EXAMPLE_FIELDS",
    "make_frame_image",
    "build_frame_zip",
    "zip_bytes",
    "BagSpec",
    "RackScene",
    "render_rack_frame",
    "build_rack_zip",
    "write_rack_frames",
    "default_rack_scene",
    "FillMeasurement",
    "anchor_prompt",
    "measure_scene",
]

#: 一份可直接喂给接口的样例业务头（占位标识，不含任何院内真实值）。
EXAMPLE_FIELDS = {
    "rack_id": "R-EXAMPLE-07",
    "infusion_order_no": "EXAMPLE-ORDER-0001",
    "bed_no": "EXAMPLE-BED-07",
    "captured_at": "2026-10-22T09:41:00+08:00",
    "slots": '[{"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-A"}]',
    "client_event_id": "example-upload-0001",
}


def make_frame_image(width: int, height: int, *, seed: int, fill_level: float = 0.6) -> Image.Image:
    """生成一帧「输液袋占位画面」：深色背景 + 上亮下暗的矩形袋体。

    形状对几何核足够（后续 task 4 的分割算法就吃这种结构），但它**不是**真实
    输液袋渲染，也不含任何患者信息。
    """
    rng = np.random.default_rng(seed)
    canvas = rng.integers(24, 48, size=(height, width, 3), dtype=np.uint8)
    margin_x, margin_y = width // 10, height // 10
    bag_w = width - 2 * margin_x
    bag_h = height - 2 * margin_y
    liquid_top = margin_y + int(bag_h * (1.0 - fill_level))
    canvas[margin_y:liquid_top, margin_x : margin_x + bag_w] = (196, 208, 214)
    canvas[liquid_top : margin_y + bag_h, margin_x : margin_x + bag_w] = (86, 142, 190)
    return Image.fromarray(canvas, mode="RGB")


def build_frame_zip(
    *,
    frame_count: int = 12,
    width: int = 360,
    height: int = 480,
    start: datetime | None = None,
    step_seconds: float = 0.5,
    seed: int = 20261022,
    names: Sequence[str] | None = None,
    start_offsets: Sequence[int] | None = None,
) -> bytes:
    """在内存里打一个帧 zip。

    ``start_offsets`` 用于构造时间逆序等非法序列（偏移量加在基础时间上）。
    """
    base = start or datetime(2026, 10, 22, 9, 41, 0)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(frame_count):
            name = names[index] if names else f"frame_{index:05d}.jpg"
            image = make_frame_image(width, height, seed=seed + index)
            moment = base + timedelta(seconds=step_seconds * index)
            info = zipfile.ZipInfo(name, date_time=moment.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            buffer_image = io.BytesIO()
            image.save(buffer_image, format="JPEG", quality=85)
            archive.writestr(info, buffer_image.getvalue())
    return buffer.getvalue()


def zip_bytes(*, extra_members: dict[str, bytes] | None = None, **kwargs) -> bytes:
    """在 :func:`build_frame_zip` 基础上追加额外成员（用于不安全 zip 的反例）。"""
    payload = build_frame_zip(**kwargs)
    if not extra_members:
        return payload
    # 重新打包含额外成员的包：解出原有帧再重写，保持同样的成员顺序
    source = zipfile.ZipFile(io.BytesIO(payload))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for info in source.infolist():
            archive.writestr(info.filename, source.read(info.filename))
        for name, data in extra_members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


# ── 多袋输液架场景（task 4 起）─────────────────────────────────────────────
#
# 上面那套是「一帧一个矩形」的占位形状；这里渲染的是**可辨认的多袋输液架**：袋体、
# 液面、挂点位置都随挂点顺序排开，几何核在它上面要能分出哪块是哪袋。
#
# 仍然**不是真实院内短片**——真实短片需医院授权采集。带真值的全参数渲染与回归由
# task 8 的 ``scripts/synthesize_rack.py`` 承担；这里只提供几何核用例需要的
# 形状与可注入的拍摄变体。

#: 透明袋体本色：低饱和、偏亮。液体与它的差别主要落在饱和度上。
BAG_BODY_RGB = (198, 210, 216)


@dataclass(frozen=True)
class BagSpec:
    """一个挂点上的袋子。``x_center``/``width_ratio`` 是相对画面宽度的比例。

    ``refill_at``/``refill_to``/``refill_rgb`` 描述**换液**：从第 ``refill_at`` 帧起，
    这个挂点上换成一只全新的袋子（液位从 ``refill_to`` 重新开始、颜色也不同）。
    三个字段默认 ``None`` / 同色，即退化为单只袋子从头放到尾——既有场景零影响。
    """

    bag_seq: str
    x_center: float
    width_ratio: float
    fill_start: float
    fill_end: float
    liquid_rgb: tuple[int, int, int]
    refill_at: int | None = None
    refill_to: float = 0.95
    refill_rgb: tuple[int, int, int] | None = None

    def colour_at(self, frame_index: int) -> tuple[int, int, int]:
        """该帧袋子里的液体颜色（换液后是新袋，颜色随之改变）。"""
        if self.refill_at is not None and self.refill_rgb is not None and frame_index >= self.refill_at:
            return self.refill_rgb
        return self.liquid_rgb


@dataclass(frozen=True)
class RackScene:
    """一段短片的场景定义：几袋、每袋怎么走液位、画面多大。"""

    bags: tuple[BagSpec, ...]
    frame_count: int = 24
    width: int = 480
    height: int = 640
    seed: int = 20261022

    def fill_at(self, bag_seq: str, frame_index: int) -> float:
        """该袋在该帧的**真值液位比例**（袋腔高度占比），测试据此核对曲线。

        换液（``refill_at``）之后按新袋重算：液位从 ``refill_to`` 起、到段末落到
        ``fill_end``，而不是接着旧袋的下降趋势。
        """
        bag = next(item for item in self.bags if item.bag_seq == bag_seq)
        if bag.refill_at is not None and frame_index >= bag.refill_at:
            span = max(1, self.frame_count - 1 - bag.refill_at)
            t = (frame_index - bag.refill_at) / span
            return max(0.0, min(1.0, bag.refill_to + (bag.fill_end - bag.refill_to) * t))
        t = frame_index / max(1, self.frame_count - 1)
        return max(0.0, min(1.0, bag.fill_start + (bag.fill_end - bag.fill_start) * t))


def default_rack_scene(frame_count: int = 24, **kwargs) -> RackScene:
    """默认双袋场景：两袋液位下降速度不同，便于断言「不同袋给出不同的曲线」。"""
    return RackScene(
        bags=(
            BagSpec("B01", 0.28, 0.30, 0.62, 0.40, (92, 148, 196)),
            BagSpec("B02", 0.72, 0.30, 0.75, 0.55, (64, 128, 190)),
        ),
        frame_count=frame_count,
        **kwargs,
    )


def _bag_box(
    scene: RackScene, bag: BagSpec, slot: int, frame_index: int, deform: float
) -> tuple[int, int, int, int]:
    """袋体外接框。``deform`` 让袋体在帧间轻微摆动，模拟手持与袋体形变。"""
    width, height = scene.width, scene.height
    wobble = math.sin(2 * math.pi * (frame_index / 9.0 + 0.35 * slot))
    centre = bag.x_center + deform * 0.02 * wobble
    half = (
        bag.width_ratio
        / 2
        * (1.0 + deform * 0.06 * math.sin(2 * math.pi * (frame_index / 7.0 + 0.2 * slot)))
    )
    x0 = max(0, int(round((centre - half) * width)))
    x1 = min(width, int(round((centre + half) * width)))
    y0 = max(0, int(round(0.12 * height)))
    y1 = min(height, int(round(0.90 * height)))
    return x0, y0, x1, y1


def _fill_slanted_liquid(
    canvas: np.ndarray,
    x0: int,
    x1: int,
    top: int,
    bottom: int,
    floor: int,
    colour: tuple[int, int, int],
    tilt_deg: float,
) -> None:
    """把液体区画成**上沿倾斜**的多边形（手机歪着拍时液面在画面里就是斜的）。

    逐列填而不是整体旋转画布：旋转会顺带引入重采样模糊，把「歪」和「糊」两个问题
    混在一条用例里，分不开。这里只让液面线本身倾斜——正是 ``_surface_tilt`` 要量的那个量。
    """
    if x1 <= x0 or bottom <= top:
        return
    xs = np.arange(x0, x1)
    centre = (x0 + x1 - 1) / 2.0
    slope = math.tan(math.radians(abs(tilt_deg)))
    tops = np.rint(top + slope * (xs - centre)).astype(np.int64)
    np.clip(tops, max(0, floor), bottom - 1, out=tops)
    for x, start in zip(xs, tops):
        canvas[start:bottom, x] = colour


def render_rack_frame(
    scene: RackScene,
    frame_index: int,
    *,
    highlight: bool = False,
    deform: float = 0.0,
    dark: float = 1.0,
    occluders: Sequence[tuple[int, int, int]] = (),
    partial_occluders: Sequence[tuple[int, int, int, float]] = (),
    tilt_deg: float = 0.0,
    crop: tuple[int, int, int, int] | None = None,
    blur: int = 0,
) -> np.ndarray:
    """渲染一帧 ``HxWx3`` RGB。

    ``highlight`` 打出镜面反光高光（落在液体区内，会把液体掩膜打出一个孔洞）；
    ``deform`` 控制袋体逐帧摆动幅度；``dark < 1`` 模拟夜间暗光；
    ``occluders`` 是 ``(挂点序号, 起始帧, 结束帧)`` 三元组——该区间内用不透明挡板
    完全盖住那一袋，用来验证「遮挡帧标记而不丢弃」；
    ``partial_occluders`` 是 ``(挂点序号, 起始帧, 结束帧, 覆盖比例)`` 四元组——用不透明
    挡板盖住袋体**下部**该比例（被单/家属的手压在袋下段），袋子仍看得见但可见液体变少，
    用来验证「只看到部分袋体」与「完全看不见」是两回事；
    ``tilt_deg`` 让液面线按该角度倾斜（手机歪着拍，默认 0 不歪）；
    ``crop`` 是 ``(y0, x0, y1, x1)`` 取景框——画面被裁到这一块，落在框外的袋子被切掉，
    用来构造「袋子没拍全、贴到画面边」（默认 None 不裁）；
    ``blur`` 是手持抖动：沿一条斜线做方向平均（真拖影，边缘被抹开）。
    ``deform`` 只让整只袋子平移摆动——画面本身仍然锐利，所以**测不出糊**，
    两者不是一回事，想验运动模糊必须用 ``blur``。
    """
    height, width = scene.height, scene.width
    rng = np.random.default_rng(scene.seed + 7919 * frame_index)
    # 背景：低饱和的暗噪声（病区夜间灯光下的墙面/床帘）
    canvas = np.repeat(rng.integers(18, 40, size=(height, width, 1), dtype=np.uint8), 3, axis=2)

    for slot, bag in enumerate(scene.bags):
        x0, y0, x1, y1 = _bag_box(scene, bag, slot, frame_index, deform)
        if x1 <= x0 or y1 <= y0:
            continue
        if any(slot == o_slot and start <= frame_index < end for o_slot, start, end in occluders):
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (118, 118, 118), -1)
            continue

        cv2.rectangle(canvas, (x0, y0), (x1, y1), BAG_BODY_RGB, -1)
        fill = scene.fill_at(bag.bag_seq, frame_index)
        liquid_top = y1 - int(round((y1 - y0) * max(0.0, min(1.0, fill))))
        inner_x0, inner_x1 = x0 + 3, x1 - 3
        if inner_x1 > inner_x0 and y1 - 3 > liquid_top:
            if abs(tilt_deg) < 1e-6:
                cv2.rectangle(
                    canvas, (inner_x0, liquid_top), (inner_x1, y1 - 3), bag.colour_at(frame_index), -1
                )
            else:
                _fill_slanted_liquid(
                    canvas,
                    inner_x0,
                    inner_x1,
                    liquid_top,
                    y1 - 3,
                    floor=y0,
                    colour=bag.colour_at(frame_index),
                    tilt_deg=tilt_deg,
                )
        covered = next(
            (
                fraction
                for o_slot, start, end, fraction in partial_occluders
                if o_slot == slot and start <= frame_index < end
            ),
            None,
        )
        if covered is not None:
            cut = y1 - int(round((y1 - y0) * max(0.0, min(1.0, covered))))
            if cut < y1:
                cv2.rectangle(canvas, (x0, cut), (x1, y1), (118, 118, 118), -1)
        if highlight and (y1 - 3 - liquid_top) > 60:
            hx0 = x0 + max(2, (x1 - x0) // 4)
            hx1 = min(x1 - 3, hx0 + max(6, (x1 - x0) // 10))
            hy0 = liquid_top + max(6, (y1 - liquid_top) // 4)
            hy1 = min(y1 - 3, hy0 + max(6, (y1 - liquid_top) // 14))
            if hx1 > hx0 and hy1 > hy0:
                cv2.rectangle(canvas, (hx0, hy0), (hx1, hy1), (255, 255, 255), -1)

    if dark != 1.0:
        canvas = np.clip(canvas.astype(np.float32) * float(dark), 0, 255).astype(np.uint8)
    if blur > 0:
        # 手持抖动：沿一条斜线做方向平均。边缘被抹开 → 拉普拉斯方差下降（真的糊了）。
        # kernel 必须自己归一：cv2.filter2D 不做归一化，不除会把画面加成一片纯白。
        size = blur * 2 + 1
        kernel = np.zeros((size, size), np.float32)
        cv2.line(kernel, (0, 0), (size - 1, size - 1), 1.0, 1)
        canvas = cv2.filter2D(canvas, -1, kernel / kernel.sum())
    if crop is not None:
        # 模拟取景框没对准：画面只剩 crop 这一块，落在框外的袋子被切掉。
        top, left, bottom, right = crop
        canvas = canvas[max(0, top) : bottom, max(0, left) : right]
    return canvas


def build_rack_zip(
    scene: RackScene | None = None,
    *,
    start: datetime | None = None,
    step_seconds: float = 0.5,
    **variant,
) -> bytes:
    """把整个场景打成可上传的帧 zip（成员仍命名为 ``frame_%05d.jpg``）。"""
    scene = scene or default_rack_scene()
    base = start or datetime(2026, 10, 22, 9, 41, 0)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(scene.frame_count):
            frame = render_rack_frame(scene, index, **variant)
            moment = base + timedelta(seconds=step_seconds * index)
            info = zipfile.ZipInfo(f"frame_{index:05d}.jpg", date_time=moment.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            encoded = io.BytesIO()
            Image.fromarray(frame, mode="RGB").save(encoded, format="JPEG", quality=85)
            archive.writestr(info, encoded.getvalue())
    return buffer.getvalue()


def write_rack_frames(directory: Path, scene: RackScene | None = None, **variant) -> list[Path]:
    """把场景逐帧落成 JPEG，返回按帧序排列的路径列表。"""
    scene = scene or default_rack_scene()
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(scene.frame_count):
        frame = render_rack_frame(scene, index, **variant)
        path = directory / f"frame_{index:05d}.jpg"
        Image.fromarray(frame, mode="RGB").save(path, format="JPEG", quality=85)
        paths.append(path)
    return paths


# ── 带真值的回归测量（task 8）──────────────────────────────────────────────
#
# 上面两套渲染器服务于「形状正确即可」的契约用例；带真值的那套在
# ``infusion_monitor.synth``（数据层），由 ``scripts/synthesize_rack.py`` 驱动。
# 这里补上**测量**侧：用产品主路径的同一套几何核去量带真值的合成短片，
# 让「分割误差」与「换算误差」能分开归因。


@dataclass(frozen=True)
class FillMeasurement:
    """一次测量的结果：面积比 + 观测状态。``observed=False`` 即「这一帧看不见」。"""

    area_ratio: float | None
    cavity_area_px: int
    liquid_area_px: int
    anchor: str

    @property
    def observed(self) -> bool:
        return self.area_ratio is not None


def _synth_bag_box(spec, bag_seq: str, frame_index: int):
    from infusion_monitor.synth import _bag_box

    plan = next(item for item in spec.bags if item.bag_seq == bag_seq)
    return _bag_box(spec, plan, frame_index)


def anchor_prompt(spec, bag_seq: str, frame_index: int, *, style: str = "bag_box"):
    """起锚提示。

    ``bag_box`` —— 首帧枚举给出的**袋体候选框** + 袋心正点（产品主路径）：
    提示框裁在袋体上，管路与滴斗落在框外，不会被算进腔体。

    ``slot_column`` —— 挂点竖列先验（``core.propagate.slot_seed_prompt`` 的退化路径）：
    框覆盖整列整高，管路与滴斗会被并进前景。

    两种起锚的差别就是空滴斗边界用例要量的东西。
    """
    from infusion_monitor.core.predictor import Point, Prompt

    x0, y0, x1, y1 = _synth_bag_box(spec, bag_seq, frame_index)
    centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    if style == "slot_column":
        return Prompt(points=(Point(centre[0], centre[1], 1),), box=(0.0, 0.0, float(spec.width), float(spec.height)))
    return Prompt(
        points=(Point(centre[0], centre[1], 1),),
        box=(float(x0), float(y0), float(x1), float(y1)),
    )


def measure_scene(spec, bag_seq: str, *, frame_index: int = 0, anchor: str = "bag_box"):
    """渲染合成短片并用产品主路径的几何核量出该袋的面积比。

    走 ``infusion_monitor.synth`` 的带真值渲染器（``scripts/synthesize_rack.py``
    的内核），帧按上传链路口径过一遍 JPEG 编解码——产品读的是 JPEG，回归就该测
    JPEG 往返后的结果，而不是原图。
    """
    import numpy as np

    from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
    from infusion_monitor.synth import decode_jpeg, encode_jpeg, render_scene

    rendered = render_scene(spec)
    frame = decode_jpeg(encode_jpeg(rendered.frames[frame_index]))
    prompt = anchor_prompt(spec, bag_seq, frame_index, style=anchor)
    predictor = GeometricMaskPredictor()

    cavity = predictor.estimate_cavity(frame, prompt)
    predictions = predictor.predict(frame, prompt)
    mask = predictions[0].mask if predictions else None
    if mask is None or not mask.any() or cavity.area_px <= 0:
        return FillMeasurement(None, cavity.area_px, 0, anchor)
    liquid_px = int(np.count_nonzero(mask))
    return FillMeasurement(liquid_px / cavity.area_px, cavity.area_px, liquid_px, anchor)