#!/usr/bin/env python3
"""公开 VOS 子集的脚本化获取（可重跑 · 断点续传 · 断网降级）。

真实院内短片须医院授权采集，本轮自动化验收用公开视频分割语料的一个子集当评测
基础。本脚本不下载整包（约 794 MB），而是先读远端 ZIP 中央目录、算出要取的成员
字节区间，再逐个 HTTP Range 取下来，因此 `--sequences` / `--max-bytes` 限定规模是
真少下。详见 `src/infusion_monitor/vos_dataset.py` 与 `docs/benchmark-vos.md`。

    # 试算：只看中央目录里能选到什么（不下任何字节）
    python scripts/fetch_vos_subset.py --dry-run

    # 取 10 个最长序列（DAVIS-2017 最长 104 帧，达不到 150 帧门禁 → 需显式放松）
    python scripts/fetch_vos_subset.py --relax-min-frames --max-sequences 10

    # 只取指定序列的前 N 帧，限总字节
    python scripts/fetch_vos_subset.py --sequences koala,sheep --max-bytes 30000000

    # 幂等重跑：已在盘且 CRC32 对得上的一律跳过
    python scripts/fetch_vos_subset.py --relax-min-frames

    # 机器可读结果 + 刷新文档顶部状态块
    python scripts/fetch_vos_subset.py --relax-min-frames --json --update-doc

产物::

    data/vos_subset/
    ├── JPEGImages/480p/<seq>/%05d.jpg     # 画面帧
    ├── Annotations/480p/<seq>/%05d.png    # 首帧/逐帧标注（评测真值）
    └── manifest.json                      # 逐文件 bytes + CRC32 + 规模 + 门禁达成情况

**降级纪律**：断网或校验不过时状态为 ``skipped``，原因写进 ``docs/benchmark-vos.md``
顶部与 manifest，绝不用合成帧、占位图或拷贝冒充真数据（合成回归另见
``docs/synthetic-data.md``）。``data/`` 已在 .gitignore 中排除，子集不入库。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from infusion_monitor.vos_dataset import (  # noqa: E402
    DEFAULT_ARCHIVE_URL,
    DEFAULT_ROOT,
    HttpRangeReader,
    fetch_subset,
    plan_subset,
    read_central_directory,
    update_docs_banner,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOC = PROJECT_ROOT / "docs" / "benchmark-vos.md"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="下载并校验公开 VOS 子集（断网降级为 skipped，不伪造数据）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default=DEFAULT_ROOT, help=f"落盘根目录（默认 {DEFAULT_ROOT}）")
    p.add_argument("--url", default=DEFAULT_ARCHIVE_URL, help="远端归档地址（可用本地 http:// 归档做离线演练）")
    p.add_argument("--sequences", default=None,
                   help="逗号分隔的序列名；缺省按帧数从长到短自动选")
    p.add_argument("--min-frames", type=int, default=150, help="每序列帧数门禁（默认 150）")
    p.add_argument("--max-sequences", type=int, default=10, help="最多取几个序列（默认 10，0 表示不限）")
    p.add_argument("--max-bytes", type=int, default=None, help="总字节上限，超过则截断（不静默：manifest 里记实际值）")
    p.add_argument("--relax-min-frames", action="store_true",
                   help="门禁达不成时显式放松，取语料内最长的一批（缺口记进 manifest 与文档，不静默降门槛）")
    p.add_argument("--timeout", type=float, default=30.0, help="单次请求超时秒数（默认 30）")
    p.add_argument("--retries", type=int, default=2, help="失败重试次数（默认 2）")
    p.add_argument("--dry-run", action="store_true", help="只读中央目录并试算选片，不下载任何字节")
    p.add_argument("--json", action="store_true", help="以 JSON 打印结果（供 CI/编排层消费）")
    p.add_argument("--update-doc", action="store_true", help="把状态块刷新到 docs/benchmark-vos.md 顶部")
    p.add_argument("--doc", default=str(DEFAULT_DOC), help="配合 --update-doc 指定文档路径")
    return p


def _print_human(report) -> None:
    status = report.status.upper()
    print(f"状态：{status}")
    if report.reason:
        print(f"原因：{report.reason}")
    if report.plan is not None:
        gate = "达成" if report.plan.min_frames_gate_satisfied else "未达成"
        print(f"每序列帧数门禁：要求 {report.plan.requested_min_frames} 帧 → {gate}")
        if report.plan.relaxed:
            print("取数方式：--relax-min-frames 显式放松")
        print(f"选中序列：{len(report.plan.sequences)} 个，"
              f"合计 {report.plan.total_frames} 帧 / {report.plan.total_bytes / 1e6:.1f} MB")
        for seq in report.plan.sequences[:15]:
            print(f"  - {seq.name:<22} {seq.frames:>4} 帧  {seq.total_bytes / 1e6:>6.2f} MB")
    if report.status == "ready":
        print(f"落盘：{report.root}")
        print(f"文件：新取 {report.downloaded_files} / 复用 {report.reused_files} / 续传 {report.resumed_files}"
              f"，共 {report.bytes_on_disk / 1e6:.1f} MB，耗时 {report.elapsed_s:.1f}s")
        bad = report.verify.get("decode_failures", [])
        print(f"解码校验：{len(report.sequences)} 个序列，失败 {len(bad)} 个")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = fetch_subset(
        url=args.url,
        root=Path(args.out),
        sequences=[s.strip() for s in args.sequences.split(",") if s.strip()] if args.sequences else None,
        min_frames=args.min_frames,
        max_sequences=args.max_sequences or None,
        max_bytes=args.max_bytes,
        relax_min_frames=args.relax_min_frames,
        timeout=args.timeout,
        retries=args.retries,
        dry_run=args.dry_run,
    )
    if args.update_doc:
        update_docs_banner(Path(args.doc), report)
        if not args.json:
            print(f"已刷新状态块：{args.doc}")
    if args.json:
        print(json.dumps(report.to_json(), ensure_ascii=False, indent=2))
    else:
        _print_human(report)
    # skipped / planned 都不是"可用评测基础"，返回非 0 让 CI 与编排层看得见
    return 0 if report.status == "ready" else (2 if report.status == "skipped" else 0)


if __name__ == "__main__":
    raise SystemExit(main())