#!/usr/bin/env python3
"""生成一段可上传的样例帧 zip（本地联调与 README 快速开始用）。

    python scripts/make_sample_track.py --out /tmp/rack.zip --frames 24

生成的是**合成的占位帧**——形状符合入站契约（``frame_%05d.jpg``、短边 ≥320px、
时间递增），但**不是真实院内短片**：真实短片需医院授权采集，本轮以合成数据 +
公开 VOS 数据集替代（见 ``docs/benchmark-vos.md``）。真实袋体渲染与真值回归见
``scripts/synthesize_rack.py``。

标识一律用 ``EXAMPLE-*`` 占位，不编造可被误认为真实的住院号、处方号或批号。
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_FRAMES = 24
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 640
DEFAULT_FILL = 0.62
DEFAULT_BAGS = 2

#: 各袋的液位色（饱和度足够高，几何核靠它把液体与透明袋体分开）
LIQUID_RGB = ((92, 148, 196), (64, 128, 190), (70, 150, 110))
BAG_BODY_RGB = (198, 210, 216)
#: 不透明挡板色：灰、无饱和度，几何核不会把它当成液体
OCCLUDER_RGB = (118, 118, 118)


def render_frame(
    width: int,
    height: int,
    index: int,
    fill: float,
    bags: int = DEFAULT_BAGS,
    frames: int = DEFAULT_FRAMES,
    occluders: tuple[tuple[int, int, int], ...] = (),
) -> Image.Image:
    """一帧占位画面：低饱和暗背景 + 若干个液位随帧缓慢下降的输液袋。

    多个袋时按挂点顺序从左到右排列，每袋液位下降速度不同——上传后各袋给出的
    ``area_ratio_series`` 应当彼此不同，可以直接看出有没有串袋。

    ``occluders`` 是 ``(挂点序号, 起始帧, 结束帧)`` 三元组：该区间内用不透明挡板盖住
    那一袋（模拟被单/手臂挡在镜头前），用于演示「遮挡帧标记而不丢弃、恢复后接回同一袋」。
    """
    rng = np.random.default_rng(20261022 + index)
    background = np.repeat(
        rng.integers(18, 40, size=(height, width, 1), dtype=np.uint8), 3, axis=2
    )
    canvas = np.asarray(background, dtype=np.uint8).copy()

    band = width / bags
    bag_w = int(band * 0.62)
    bag_h = int(height * 0.78)
    top, bottom = int(height * 0.12), int(height * 0.90)

    for slot in range(bags):
        centre = int((slot + 0.5) * band)
        x0 = max(0, centre - bag_w // 2)
        x1 = min(width, centre + bag_w // 2)
        if x1 <= x0:
            continue
        if any(
            slot == o_slot and start <= index < end for o_slot, start, end in occluders
        ):
            canvas[top:bottom, x0:x1] = OCCLUDER_RGB
            continue
        canvas[top:bottom, x0:x1] = BAG_BODY_RGB
        # 液面随帧下降：末帧比首帧低 25%–40%（每袋不同）
        span = 0.25 + 0.15 * slot
        level = max(0.0, fill * (1.0 - span * index / max(1, frames - 1)))
        liquid_top = bottom - int((bottom - top) * level)
        if bottom - 3 > liquid_top:
            canvas[liquid_top : bottom - 3, x0 + 3 : x1 - 3] = LIQUID_RGB[slot % len(LIQUID_RGB)]
    return Image.fromarray(canvas, mode="RGB")


def build(
    out_path: Path,
    frames: int,
    width: int,
    height: int,
    fill: float,
    bags: int = DEFAULT_BAGS,
    occluders: tuple[tuple[int, int, int], ...] = (),
) -> Path:
    base = datetime.now().replace(microsecond=0) - timedelta(seconds=int(frames * 0.5))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(frames):
            image = render_frame(
                width, height, index, fill, bags=bags, frames=frames, occluders=occluders
            )
            moment = base + timedelta(seconds=0.5 * index)
            info = zipfile.ZipInfo(
                f"frame_{index:05d}.jpg", date_time=moment.timetuple()[:6]
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            from io import BytesIO

            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            archive.writestr(info, buffer.getvalue())
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成可上传的样例帧 zip")
    parser.add_argument("--out", required=True, type=Path, help="输出 zip 路径")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="帧数（12–1200）")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fill", type=float, default=DEFAULT_FILL, help="首袋的液位比例 0–1")
    parser.add_argument("--bags", type=int, default=DEFAULT_BAGS, help="同架袋数（1–6）")
    parser.add_argument(
        "--occlude",
        default="",
        metavar="SLOT:START:END",
        help="用不透明挡板盖住某袋的某段帧（挂点序号:起始帧:结束帧），可重复；"
        "用于演示遮挡帧标记与恢复续接",
    )
    args = parser.parse_args(argv)

    occluders = []
    for item in filter(None, args.occlude.split(",")):
        parts = item.split(":")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            parser.error(f"--occlude 需形如 0:5:8（挂点:起始帧:结束帧），收到 {item!r}")
        slot, start, end = (int(part) for part in parts)
        if not 0 <= slot < args.bags:
            parser.error(f"--occlude 的挂点序号 {slot} 超出袋数 {args.bags}")
        if not 0 <= start < end <= args.frames:
            parser.error(f"--occlude 的帧区间 {start}:{end} 须落在 0–{args.frames}")
        occluders.append((slot, start, end))


    if not 12 <= args.frames <= 1200:
        parser.error("--frames 须落在 12–1200（低于 12 帧会被入站硬校验退回）")
    if min(args.width, args.height) < 320:
        parser.error("短边不得小于 320px，否则会被入站硬校验退回")
    if not 1 <= args.bags <= 6:
        parser.error("--bags 须落在 1–6（单架上限 6 袋）")

    path = build(
        args.out,
        args.frames,
        args.width,
        args.height,
        args.fill,
        bags=args.bags,
        occluders=tuple(occluders),
    )
    print(f"已生成 {path}（{args.frames} 帧，{args.width}×{args.height}，{args.bags} 袋）")
    if occluders:
        print(f"已注入遮挡：{occluders}（该区间内该袋在响应里记 occluded/partial）")
    print("上传时 slots 按袋序给出；bag_type_code 决定能否换出毫升数，rate_text 是滴速：")
    slots = [
        {
            "bag_seq": f"B{index + 1:02d}",
            "slot_index": index,
            "bag_type_code": "TYPE-A",
            "rate_text": "5ml/h",
        }
        for index in range(args.bags)
    ]
    print("  " + __import__("json").dumps(slots, ensure_ascii=False))
    print(
        "  （没有登记 bag_type_code 的标定记录时，响应里不出现 estimated_volume_ml）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())