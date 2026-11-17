"""进程入口：先装好结构化日志，再拉起 uvicorn。

**为什么单独一个模块**：日志配置只应该发生在真正的进程入口，不该在
``import infusion_monitor.api.service`` 时就发生。那条 import 链上还有 pytest——
测试里改一次 ``root.setLevel``，整套套件的日志级别与 ``caplog`` 就跟着变了，
排障时会怀疑人生。把配置放在这里，``python -m infusion_monitor.server`` 与
Dockerfile 的 CMD 都走它，测试则完全不碰。

环境变量：

* ``LOG_LEVEL``（默认 ``INFO``；``DEBUG`` 时日志带完整 traceback）
* ``LOG_FORMAT``（默认 ``json``；设成 ``human`` 出一行速览，方便现场排障）
* ``HOST`` / ``PORT``（默认 ``0.0.0.0:8000``）
"""

from __future__ import annotations

import logging
import os

__all__ = ["app", "build_app", "main"]


def build_app():
    """构造 FastAPI 应用（薄壳，让 ``uvicorn infusion_monitor.server:app`` 也能用）。"""
    from .api.service import create_app

    return create_app()


#: ASGI 应用对象。凭据在 lifespan 启动时才校验，因此 import 本模块不会因缺环境变量失败。
app = build_app()


def main() -> int:
    import uvicorn

    from .logging_utils import configure_logging

    configure_logging(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        human=os.environ.get("LOG_FORMAT", "json").lower() == "human",
    )
    logging.getLogger("infusion_monitor").debug(
        "日志格式：%s，级别：%s",
        os.environ.get("LOG_FORMAT", "json"),
        os.environ.get("LOG_LEVEL", "INFO"),
    )
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "*"),
        # log_config=None 是刻意的：uvicorn 默认会给自己装一套独立的 handler 并切断
        # 向 root 传播，于是「Uvicorn running on …」这类启动行会以纯文本混进 JSON 日志流，
        # 采集器按行解析就会在服务**启动那一刻**报错——恰好是最需要看日志的时候。
        # 交回 root 之后，整个进程（uvicorn + 本服务）只有一个格式。
        log_config=None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - 进程入口
    raise SystemExit(main())