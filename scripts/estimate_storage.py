#!/usr/bin/env python3
"""容量测算：按病区规模估算帧目录占用与数据库表增长。

**为什么要有这个命令**：帧目录是像素库，一天几 GB 很正常；信息科在申请磁盘、
决定保留期、或者判断「该不该换 PostgreSQL」时，需要一个能算出来的数，而不是
「先跑着看看」。

**数字从哪来**（重要）：

* 默认的「单帧字节数」是**本产品渲染器在规范分辨率（长边 1280px）下的实测值**
  （约 156 KiB/帧，JPEG 质量 88）。它是**下界**：合成画面是平色，真手机照片
  纹理多，同分辨率下通常更大。
* 想拿部署环境的真值，用 ``--frames-root`` 直接量盘上的真实帧，或用
  ``--database`` 量真实库文件——两个都是实测，不是估的。
* 表增长按每行的字段量估（含两段面积序列 JSON），同样可以用 ``--database``
  里的真实行数校正。

用法::

    python scripts/estimate_storage.py                              # 默认规模
    python scripts/estimate_storage.py --wards 3 --beds-per-ward 30
    python scripts/estimate_storage.py --frames-root ./var/frames    # 用真实帧实测
    python scripts/estimate_storage.py --database ./var/infusion_monitor.db
    python scripts/estimate_storage.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infusion_monitor.config import MissingCredentialError, get_settings  # noqa: E402

#: 规范分析分辨率的长边（与 config.FRAME_LONG_EDGE_MAX 一致）。
CANONICAL_LONG_EDGE = 1280

#: 单帧字节数的默认值：渲染器实测（长边 1280、JPEG q88）后向上取整到 KiB。
#: **这是下界**——合成画面平色，真手机照片同分辨率下更大。
DEFAULT_FRAME_BYTES = 156 * 1024

#: 每轮上传的帧数三档（护士端行为不同：抽样上传 vs 整段录像）。
#: 低位 = 终端抽样（12–24 帧）；典型 = 10 秒 × 30 帧/秒；高位 = 20 秒 × 30 帧/秒。
FRAME_LOW = 24
FRAME_TYPICAL = 300
FRAME_HIGH = 600

#: 每轮每袋的默认袋数（一架输液架挂 1–3 袋常见，取 2）。
DEFAULT_BAGS_PER_ROUND = 2

#: 每床每天的换液次数（500 ml 袋按 60 ml/h 约 8 小时一袋，取 3 次/天估算）。
DEFAULT_REPLACEMENTS_PER_BED_PER_DAY = 3

#: 每张表的单行字节估算（含 JSON 列）。用 ``--database`` 可用实测校正。
ROW_BYTES = {
    "bag_session": 6 * 1024,       # 两段面积序列 JSON 占大头
    "infusion_event": 2 * 1024,    # 含出站载荷快照
    "writeback_queue": 2 * 1024,
    "inference_audit": 0.3 * 1024,
    "audit_log": 0.5 * 1024,
    "upload_chunk": 1 * 1024,
}

#: 换 PostgreSQL 的判据（与 INSTALL.md「SQLite 的并发边界」一致）。
PG_SESSION_ROWS_THRESHOLD = 500_000
PG_QPS_THRESHOLD = 20


def _human_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"


def _measure_frame_bytes(frames_root: Path) -> tuple[int, int, int]:
    """量盘上真实的帧：返回（平均字节，帧数，目录数）。没有帧就报 0。"""
    total = 0
    count = 0
    directories = 0
    for frame in frames_root.rglob("frame_*.jpg"):
        try:
            total += frame.stat().st_size
            count += 1
        except OSError:
            continue
    for _ in frames_root.iterdir() if frames_root.is_dir() else ():
        directories += 1
    average = int(total / count) if count else 0
    return average, count, directories


def _measure_database(database_path: Path) -> dict:
    """量真实库：文件大小与各表行数（用来校正「每行多少字节」）。"""
    if not database_path.is_file():
        return {}
    conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        counts = {}
        for table in tables:
            try:
                counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            except sqlite3.Error:
                continue
    finally:
        conn.close()
    return {
        "file_bytes": database_path.stat().st_size,
        "row_counts": counts,
        "total_rows": sum(counts.values()),
    }


def estimate(
    *,
    wards: int,
    beds_per_ward: int,
    rounds_per_day: int,
    bags_per_round: int,
    frames_per_round: int,
    frame_bytes: int,
    retention_days: int,
    replacements_per_bed_per_day: int,
    frame_bytes_source: str,
    database_measurement: dict | None = None,
) -> dict:
    """算出一天的帧量、保留期内的帧占用、每天各表新增行数与库增长。"""
    rounds_total = wards * rounds_per_day
    bags_per_day = rounds_total * bags_per_round
    frames_per_day = rounds_total * frames_per_round
    bytes_per_day = frames_per_day * frame_bytes

    frame_disk = bytes_per_day * retention_days

    rows_per_day = {
        "bag_session": bags_per_day,
        "infusion_event": wards * beds_per_ward * replacements_per_bed_per_day,
        "writeback_queue": wards * beds_per_ward * replacements_per_bed_per_day,
        "inference_audit": bags_per_day,
        "audit_log": bags_per_day * 6,
        "upload_chunk": rounds_total,
    }
    db_bytes_per_day = sum(rows_per_day[name] * size for name, size in ROW_BYTES.items())

    measured = database_measurement or {}
    per_row_note = ""
    if measured.get("total_rows"):
        observed = measured["file_bytes"] / measured["total_rows"]
        if observed > 0:
            db_bytes_per_day = sum(rows_per_day.values()) * observed
            per_row_note = (
                f"按实测库文件 {measured['file_bytes'] / 1024:.0f} KiB / "
                f"{measured['total_rows']} 行 ≈ {observed:.0f} B/行 校正"
            )

    year_rows = sum(rows_per_day.values()) * 365
    session_rows_year = rows_per_day["bag_session"] * 365

    verdicts = []
    if session_rows_year >= PG_SESSION_ROWS_THRESHOLD:
        verdicts.append(
            f"一年后会话表约 {session_rows_year:,} 行，已过 {PG_SESSION_ROWS_THRESHOLD:,} 行的"
            "换库判据——跨病区集中部署就该换 PostgreSQL"
        )
    else:
        verdicts.append(
            f"一年后会话表约 {session_rows_year:,} 行，未过 {PG_SESSION_ROWS_THRESHOLD:,} 行的"
            "换库判据（单机 SQLite 够用）"
        )
    if rounds_total >= PG_QPS_THRESHOLD * 60:
        verdicts.append(
            f"全天 {rounds_total:,} 轮上传，折合平均 {rounds_total / 86400:.2f} QPS，"
            f"远低于 {PG_QPS_THRESHOLD} QPS 的换库判据"
        )

    return {
        "assumptions": {
            "wards": wards,
            "beds_per_ward": beds_per_ward,
            "rounds_per_day_per_ward": rounds_per_day,
            "bags_per_round": bags_per_round,
            "frames_per_round": frames_per_round,
            "frame_bytes": frame_bytes,
            "frame_bytes_source": frame_bytes_source,
            "retention_days": retention_days,
            "replacements_per_bed_per_day": replacements_per_bed_per_day,
        },
        "frames": {
            "rounds_per_day": rounds_total,
            "bags_per_day": bags_per_day,
            "frames_per_day": frames_per_day,
            "bytes_per_day": bytes_per_day,
            "bytes_per_day_human": _human_bytes(bytes_per_day),
            "disk_within_retention": frame_disk,
            "disk_within_retention_human": _human_bytes(frame_disk),
            "disk_one_year_no_retention": bytes_per_day * 365,
            "disk_one_year_no_retention_human": _human_bytes(bytes_per_day * 365),
        },
        "database": {
            "rows_per_day": rows_per_day,
            "bytes_per_day": int(db_bytes_per_day),
            "bytes_per_day_human": _human_bytes(db_bytes_per_day),
            "bytes_per_year": int(db_bytes_per_day * 365),
            "bytes_per_year_human": _human_bytes(db_bytes_per_day * 365),
            "note": per_row_note or "按各表字段量估算（含 JSON 列）",
            "measured": measured,
        },
        "verdicts": verdicts,
    }


def _render(payload: dict) -> str:
    assumptions = payload["assumptions"]
    frames = payload["frames"]
    database = payload["database"]
    lines = [
        "输液余量监测系统 · 容量测算",
        "=" * 52,
        "",
        "【测算口径】",
        f"  病区数            {assumptions['wards']} 个",
        f"  每病区床位        {assumptions['beds_per_ward']} 张",
        f"  每病区日均巡视轮次 {assumptions['rounds_per_day_per_ward']} 轮（1–2 小时一轮）",
        f"  每轮袋数          {assumptions['bags_per_round']} 袋",
        f"  每轮帧数          {assumptions['frames_per_round']} 帧",
        f"  单帧字节          {assumptions['frame_bytes']:,} B（{assumptions['frame_bytes_source']}）",
        f"  帧保留期          {assumptions['retention_days']} 天",
        "",
        "【帧目录】",
        f"  每天上传          {frames['rounds_per_day']:,} 轮 / {frames['bags_per_day']:,} 袋 "
        f"/ {frames['frames_per_day']:,} 帧",
        f"  每天新增          {frames['bytes_per_day_human']}",
        f"  保留期内占用      {frames['disk_within_retention_human']}（保留 {assumptions['retention_days']} 天）",
        f"  不做保留期        {frames['disk_one_year_no_retention_human']}（一年）",
        "",
        "【数据库】",
        f"  每天新增行数      {database['rows_per_day']}",
        f"  每天增长          {database['bytes_per_day_human']}",
        f"  一年增长          {database['bytes_per_year_human']}",
        f"  口径              {database['note']}",
        "",
        "【换 PostgreSQL 的判据】",
    ]
    lines.extend(f"  - {verdict}" for verdict in payload["verdicts"])
    lines.extend(
        [
            "",
            "  判据出处：INSTALL.md「SQLite 的并发边界」——换库只改 DATABASE_URL，",
            "  仓储层不感知方言（store/dsn.py 是唯一的方言入口）。",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按病区规模估算帧目录占用与数据库表增长")
    parser.add_argument("--wards", type=int, default=1, help="病区数（默认 1）")
    parser.add_argument("--beds-per-ward", type=int, default=30, help="每病区床位（默认 30）")
    parser.add_argument(
        "--rounds-per-day", type=int, default=12, help="每病区日均巡视轮次（默认 12，1–2 小时一轮）"
    )
    parser.add_argument("--bags-per-round", type=int, default=DEFAULT_BAGS_PER_ROUND, help="每轮袋数")
    parser.add_argument(
        "--frames-per-round",
        type=int,
        default=FRAME_TYPICAL,
        help=f"每轮帧数（默认 {FRAME_TYPICAL}：10 秒 × 30 帧/秒；抽样上传可按 24 算）",
    )
    parser.add_argument(
        "--avg-frame-bytes",
        type=int,
        default=None,
        help="单帧字节数；不给则用渲染器实测默认值",
    )
    parser.add_argument(
        "--frames-root",
        default=None,
        help="真实帧目录；给了就用盘上实测的单帧字节（比默认值准）",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="真实 SQLite 库文件；给了就用实测行数校正每行字节",
    )
    parser.add_argument("--retention-days", type=int, default=30, help="帧目录保留天数（默认 30）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    frame_bytes = args.avg_frame_bytes
    source = "手工指定"
    if frame_bytes is None and args.frames_root:
        measured, count, _ = _measure_frame_bytes(Path(args.frames_root))
        if measured:
            frame_bytes = measured
            source = f"实测 {args.frames_root}（{count} 帧）"
    if frame_bytes is None:
        frame_bytes = DEFAULT_FRAME_BYTES
        source = (
            f"渲染器实测下界（长边 {CANONICAL_LONG_EDGE}px、JPEG q88，"
            "平色合成画面，真照片更大）"
        )

    database_measurement = None
    if args.database:
        database_measurement = _measure_database(Path(args.database))
        if not database_measurement:
            print(f"警告：读不到数据库文件 {args.database}，按估算口径出结果", file=sys.stderr)

    payload = estimate(
        wards=args.wards,
        beds_per_ward=args.beds_per_ward,
        rounds_per_day=args.rounds_per_day,
        bags_per_round=args.bags_per_round,
        frames_per_round=args.frames_per_round,
        frame_bytes=frame_bytes,
        retention_days=args.retention_days,
        replacements_per_bed_per_day=DEFAULT_REPLACEMENTS_PER_BED_PER_DAY,
        frame_bytes_source=source,
        database_measurement=database_measurement,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(_render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
