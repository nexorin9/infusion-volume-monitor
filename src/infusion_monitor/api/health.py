"""``GET /api/v1/health``：运维与挂接方探活。

只报「能不能用」的事实（schema 版本、库可达、帧目录可写），不报业务内容；
凭据一律给掩码或压根不出现。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from .. import __version__
from ..store.migrations import current_version

__all__ = ["router"]

router = APIRouter(tags=["health"])


@router.get("/api/v1/health", summary="健康检查")
async def health(request: Request) -> dict:
    # 同步处理函数会被 Starlette 丢进线程池执行，而 SQLite 连接建在事件循环线程上；
    # 全部走 async 处理函数，保证仓储调用与建连在同一线程（见 api/service.py 的说明）。
    return _health_payload(request)


def _health_payload(request: Request) -> dict:
    state = request.app.state
    conn = state.store.conn
    sessions = state.store.sessions
    frames_root = Path(state.settings.frames_root)
    # 探活被 Docker healthcheck 与监控系统按分钟级轮询，正好是异线程审计的落库时机
    # （见 mqtt_telemetry.PENDING_AUDIT_MAX）。单批最多 512 行、一次事务，写不动也
    # 不拖慢探活；这里不做也不影响任何结论，只是让通道记录落库慢一拍。
    channel = getattr(state, "telemetry_channel", None)
    if channel is not None:
        channel.flush_pending_audit()
    return {
        "status": "ok",
        "service": "infusion-volume-monitor",
        "version": __version__,
        "schema_version": current_version(conn),
        "database": "sqlite",
        "frames_root": str(frames_root),
        "frames_root_writable": frames_root.exists() and frames_root.is_dir(),
        "session_count": sessions.count(),
    }