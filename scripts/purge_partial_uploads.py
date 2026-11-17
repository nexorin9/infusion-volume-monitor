#!/usr/bin/env python3
"""回收保留期已到的分片续传半成品（病区侧 cron / 信息科手工跑）。

为什么要有这个命令：分片续传开着没提交的那一段帧会留在 ``FRAMES_ROOT/_staging/`` 下，
护士当天没补完就没人管它。定时跑一次，按 ``UPLOAD_RESUME_RETENTION_HOURS`` 把到点的
半成品结案为 ``abandoned`` 并删掉它的暂存目录——**只删暂存目录**，已完成会话的帧
一个都不碰。

用法::

    python scripts/purge_partial_uploads.py                # 按配置的保留期回收
    python scripts/purge_partial_uploads.py --dry-run      # 只报不删
    python scripts/purge_partial_uploads.py --retention-hours 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infusion_monitor.config import get_settings  # noqa: E402
from infusion_monitor.ingest.resume import purge_expired  # noqa: E402
from infusion_monitor.store import UploadRepository, apply_migrations, connect  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="回收保留期已到的分片续传半成品")
    parser.add_argument(
        "--retention-hours",
        type=float,
        default=None,
        help="保留小时数；缺省取配置项 UPLOAD_RESUME_RETENTION_HOURS",
    )
    parser.add_argument("--dry-run", action="store_true", help="只报不删")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 cron 采集）")
    args = parser.parse_args(argv)

    settings = get_settings()
    conn = connect(settings.database_url)
    apply_migrations(conn)
    uploads = UploadRepository(conn)
    result = purge_expired(
        uploads,
        Path(settings.frames_root),
        retention_hours=args.retention_hours,
        dry_run=args.dry_run,
    )
    payload = result.as_dict()
    payload["retention_hours"] = (
        settings.upload_resume_retention_hours
        if args.retention_hours is None
        else args.retention_hours
    )
    conn.close()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    print(
        f"保留期 {payload['retention_hours']} 小时"
        f"{'（dry-run，未删除）' if args.dry_run else ''}："
        f"回收 {payload['expired_count']} 段未提交的上传，"
        f"释放 {payload['freed_bytes'] / 1024:.0f} KiB；"
        f"已完成的上传 {payload['kept_completed']} 段，一个都没动。"
    )
    for upload_id in payload["expired_uploads"]:
        print(f"  - {upload_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())