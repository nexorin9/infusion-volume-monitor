#!/usr/bin/env python3
"""参数化合成输液架短片 + 真值清单（数据脚本）。

    # 默认场景：双袋、24 帧、固定机位
    python scripts/synthesize_rack.py --out data/synth

    # 单架 6 袋、夜间暗光、其中两袋被手挡住
    python scripts/synthesize_rack.py --out data/synth --scene-id night-6bag \
        --bags 6 --frames 30 --variant night --occlude 1:4:12 --occlude 3:18:26

    # 只看真值，不落帧（干跑）
    python scripts/synthesize_rack.py --out /tmp/x --dry-run

    # 由 manifest 逐位复现（固定随机种子，产物字节一致）
    python scripts/synthesize_rack.py --reproduce data/synth/<scene_id>/manifest.json --out data/repro

产物::

    <out>/<scene_id>/
    ├── RGB/frame_%05d.jpg        # 画面帧（可重新打包上传）
    ├── GT/<bag_seq>/%05d.png     # 该袋在该帧的真实液体区域二值掩膜
    ├── frames.zip                # 可直接 POST /api/v1/infusion/tracks 的帧包
    └── manifest.json             # 真值清单（液位、体积、理论面积比、像素计数）

GT 目录沿用 ``{scene_id}/{object_id}/{frame_id}.png``，与源产品评测器读取的 GT 目录
一致，后续 IoU / J&F 对照可直接消费。

**这是合成替身，不是真实院内数据。** 真实短片必须经医院授权采集，本轮自动化验收
以合成 + 公开 VOS 数据集替代（见 ``docs/synthetic-data.md``）。清单里的一切标识用
``EXAMPLE-*`` / ``<EXAMPLE_*>`` 占位，不编造可被误认为真实的住院号、处方号、批号
或拍摄日期。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from infusion_monitor.synth import (  # noqa: E402
    BAG_TYPE_CATALOG,
    VARIANTS,
    BagPlan,
    SceneSpec,
    manifest_from_spec,
    render_scene,
    spec_from_manifest,
    write_scene,
)

#: 默认场景：两袋不同袋型、液位下降速度不同——上传后各袋曲线必然不同，可直接看有没有串袋。
DEFAULT_PLAN = (
    BagPlan("B01", 0, "TYPE-A", 0.62, 0.38),
    BagPlan("B02", 1, "TYPE-B", 0.75, 0.52, drip_chamber=True, label=True),
)


def build_plan(bag_count: int, bag_types: str | None, drip_chamber: bool, label: bool) -> tuple[BagPlan, ...]:
    """按袋数拼出默认布袋方案。``--bag-types`` 可指定循环使用的袋型代码。"""
    if bag_types:
        codes = [item.strip() for item in bag_types.split(",") if item.strip()]
    else:
        codes = [profile.bag_type_code for profile in BAG_TYPE_CATALOG]
    plans: list[BagPlan] = []
    for index in range(bag_count):
        code = codes[index % len(codes)]
        plans.append(
            BagPlan(
                bag_seq=f"B{index + 1:02d}",
                slot_index=index,
                bag_type_code=code,
                # 液位一路下降，且各袋速度不同：便于断言「不同袋给出不同的曲线」
                fill_start=round(0.82 - 0.05 * (index % 4), 3),
                fill_end=round(0.46 - 0.05 * (index % 3), 3),
                drip_chamber=drip_chamber and index % 2 == 1,
                label=label and index % 2 == 0,
            )
        )
    return tuple(plans)


def parse_occlusion(items: list[str]) -> tuple[tuple[str, int, int], ...]:
    """``--occlude B01:4:12`` → 该袋第 4–11 帧被手挡住。"""
    parsed: list[tuple[str, int, int]] = []
    for item in items:
        parts = item.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--occlude 需要 B01:起始帧:结束帧 三段，收到 {item!r}")
        parsed.append((parts[0].strip(), int(parts[1]), int(parts[2])))
    return tuple(parsed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="参数化合成输液架短片并输出真值清单（数据替身，非真实院内数据）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", default="data/synth", help="输出根目录（默认 data/synth）")
    parser.add_argument("--scene-id", default="synth-rack", help="场景编号，作为输出子目录名")
    parser.add_argument("--frames", type=int, default=24, help="帧数（入站契约要求 12–1200）")
    parser.add_argument("--bags", type=int, default=2, help="单架袋数，1–6")
    parser.add_argument("--bag-types", default=None, help="袋型代码，逗号分隔循环使用，如 TYPE-A,TYPE-C")
    parser.add_argument("--width", type=int, default=480, help="画面宽")
    parser.add_argument("--height", type=int, default=640, help="画面高（短边须 ≥320px）")
    parser.add_argument("--seed", type=int, default=20261022, help="随机种子（决定全部随机量，改了就不可复现）")
    parser.add_argument(
        "--variant",
        default="fixed",
        choices=VARIANTS,
        help="拍摄变体：fixed 固定机位 / handheld 手持抖动 / highlight 反光高光 / "
        "deform 袋体形变 / hand_occlusion 手与管路遮挡 / night 夜间暗光",
    )
    parser.add_argument("--occlude", action="append", default=[], metavar="B01:4:12", help="手遮挡区间，可重复")
    parser.add_argument("--drip-chamber", action="store_true", help="给偶数位袋挂上空滴斗与管路")
    parser.add_argument("--label", action="store_true", help="给奇数位袋贴输液标签")
    parser.add_argument("--reproduce", default=None, help="由既有 manifest.json 逐位复现")
    parser.add_argument("--dry-run", action="store_true", help="只打印真值清单，不落帧")
    parser.add_argument("--json", action="store_true", help="把 manifest 打到标准输出")
    args = parser.parse_args(argv)

    if args.reproduce:
        manifest = json.loads(Path(args.reproduce).read_text(encoding="utf-8"))
        spec = spec_from_manifest(manifest)
    else:
        if not 1 <= args.bags <= 6:
            raise SystemExit(f"--bags 需在 1–6 之间，收到 {args.bags}")
        spec = SceneSpec(
            scene_id=args.scene_id,
            bags=build_plan(args.bags, args.bag_types, args.drip_chamber, args.label),
            frame_count=args.frames,
            width=args.width,
            height=args.height,
            seed=args.seed,
            variant=args.variant,
            occlude_frames=parse_occlusion(args.occlude),
        )

    if args.dry_run:
        print(json.dumps(manifest_from_spec(spec), ensure_ascii=False, indent=2))
        return 0

    rendered = render_scene(spec)
    root = write_scene(Path(args.out), rendered)
    manifest = manifest_from_spec(spec)
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"场景 {spec.scene_id}（{spec.variant}，seed={spec.seed}）已写入 {root}")
    print(f"  帧 {spec.frame_count} 张 · {spec.width}×{spec.height} · 袋 {len(spec.bags)} 个")
    for plan in spec.bags:
        truth = rendered.truth_rows(plan.bag_seq)
        first, last = truth[0], truth[-1]
        print(
            f"  {plan.bag_seq} {plan.bag_type_code}（{plan.profile.label_zh}）"
            f" 可用容量 {plan.profile.usable_capacity_ml:.0f} ml"
            f" 液面 {first['true_height_fraction']:.2f}→{last['true_height_fraction']:.2f}"
            f" 真值 {first['true_volume_ml']:.0f}→{last['true_volume_ml']:.0f} ml"
            f" 理论面积比 {first['projected_area_ratio']:.3f}→{last['projected_area_ratio']:.3f}"
        )
    print("  真值：manifest.json · 掩膜：GT/<bag_seq>/%05d.png · 可上传：frames.zip")
    print("  提示：以上为程序合成数据，非真实院内短片；详见 docs/synthetic-data.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())