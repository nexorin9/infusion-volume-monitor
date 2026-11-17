#!/usr/bin/env python3
"""按保留期清理过期的输液架帧目录（病区侧 cron / 信息科手工跑）。

为什么要有这个命令：一轮巡视留一段 10–20 秒短片，一个输液架一个目录。一间病区
一天十几轮、全院几十间，帧目录按月计就会把病区服务器的盘吃满。保留期到点把帧
清掉（可选搬到归档目录），**读数留在库里**——护士长复盘「上周三那袋还剩多少」
看的是会话行，不是那堆 jpg。

**不会误删的三类**（有测试钉住）：会话还没结束的、出站事件还没送达对端的、
待写回队列里还挂着欠账的；审计链（飞检留档）任何保留期都不动。

用法::

    python scripts/purge_expired_sessions.py                 # 按配置清理
    python scripts/purge_expired_sessions.py --dry-run       # 只报不删
    python scripts/purge_expired_sessions.py --retention-days 7
    python scripts/purge_expired_sessions.py --archive-root /var/lib/infusion-monitor/archive
    python scripts/purge_expired_sessions.py --json          # 供 cron 采集
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infusion_monitor.config import MissingCredentialError, get_settings  # noqa: E402
from infusion_monitor.store import (  # noqa: E402
    AuditChain,
    RetentionPolicy,
    apply_migrations,
    connect,
    run_retention,
)

#: 凭据位缺失时的兜底：保留期清理只碰存储，不需要模型网关或出站凭据。
_FALLBACK_DATABASE_URL = "sqlite:///./var/infusion_monitor.db"
_FALLBACK_FRAMES_ROOT = "./var/frames"
_FALLBACK_SESSION_DAYS = "30"
_FALLBACK_AUDIT_DAYS = "90"
_FALLBACK_ARCHIVE_ROOT = ""


def _storage_settings(args: argparse.Namespace) -> tuple[str, str, float, float, str]:
    """取存储相关的配置。

    **凭据缺失不挡清理**：保留期任务只碰数据库与帧目录，不发请求、不调模型。
    护理部的模型网关还没配好时，磁盘照样该收——所以这里退回环境变量与默认值，
    而不是让整个 cron 因为「少一个 LLM_API_KEY」就失败。
    """
    database_url = args.database_url or os.environ.get("DATABASE_URL", "")
    frames_root = args.frames_root or os.environ.get("FRAMES_ROOT", "")
    session_days = os.environ.get("RETENTION_SESSION_DAYS", "")
    audit_days = os.environ.get("RETENTION_INFERENCE_AUDIT_DAYS", "")
    archive_root = os.environ.get("RETENTION_ARCHIVE_ROOT", "")

    try:
        settings = get_settings()
    except MissingCredentialError:
        settings = None

    if settings is not None:
        database_url = database_url or settings.database_url
        frames_root = frames_root or settings.frames_root
        session_days = session_days or str(settings.retention_session_days)
        audit_days = audit_days or str(settings.retention_inference_audit_days)
        archive_root = archive_root or (settings.retention_archive_root or "")

    return (
        database_url or _FALLBACK_DATABASE_URL,
        frames_root or _FALLBACK_FRAMES_ROOT,
        float(session_days or _FALLBACK_SESSION_DAYS),
        float(audit_days or _FALLBACK_AUDIT_DAYS),
        archive_root or _FALLBACK_ARCHIVE_ROOT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按保留期清理过期的输液架帧目录")
    parser.add_argument(
        "--retention-days",
        type=float,
        default=None,
        help="帧目录保留天数；缺省取配置项 RETENTION_SESSION_DAYS",
    )
    parser.add_argument(
        "--audit-retention-days",
        type=float,
        default=None,
        help="推理审计保留天数；缺省取配置项 RETENTION_INFERENCE_AUDIT_DAYS",
    )
    parser.add_argument("--archive-root", default=None, help="归档根目录；不给则到期直接删除")
    parser.add_argument("--database-url", default=None, help="数据库连接串；缺省取配置项")
    parser.add_argument("--frames-root", default=None, help="帧目录根路径；缺省取配置项")
    parser.add_argument("--dry-run", action="store_true", help="只报不删（不碰文件、不写库）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 cron 采集）")
    args = parser.parse_args(argv)

    database_url, frames_root, session_days, audit_days, archive_root = _storage_settings(args)

    policy = RetentionPolicy(
        session_retention_days=args.retention_days if args.retention_days is not None else session_days,
        inference_audit_retention_days=(
            args.audit_retention_days if args.audit_retention_days is not None else audit_days
        ),
        archive_root=Path(args.archive_root or archive_root) if (args.archive_root or archive_root) else None,
        dry_run=args.dry_run,
    )

    conn = connect(database_url)
    apply_migrations(conn)
    audit = AuditChain(conn)
    result = run_retention(
        conn,
        frames_root,
        policy=policy,
        audit=audit,
        request_id="retention-cron",
    )
    conn.close()

    payload = result.as_dict()
    payload["database_url"] = database_url
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    print(
        f"保留期 {policy.session_retention_days} 天"
        f"{'（dry-run，未删除）' if policy.dry_run else ''}，截止 {result.cutoff}："
        f"归档 {len(result.archived)} 袋、删除 {len(result.purged)} 袋、"
        f"目录本就不在 {len(result.absent)} 袋；"
        f"保留 {result.kept_sessions} 袋（未完成 / 有写回欠账 / 还在保留期内），"
        f"释放 {result.freed_bytes / 1024 / 1024:.1f} MiB；"
        f"裁掉推理审计 {result.pruned_inference_audit} 行。"
    )
    for item in result.blocked[:20]:
        print(f"  保留 {item['session_id']}：{item['reason_text']}")
    if len(result.blocked) > 20:
        print(f"  …… 另有 {len(result.blocked) - 20} 袋未列出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
