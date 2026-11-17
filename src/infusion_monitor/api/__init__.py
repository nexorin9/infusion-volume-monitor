"""HTTP API 层：服务装配、错误契约、健康检查与响应模型。

交付形态是**无界面 HTTP API 服务**::

    uvicorn infusion_monitor.api:app        # 常驻进程，契约页在 /docs

> 本包里的子模块**不能叫 ``app``**：``infusion_monitor.api.app`` 一旦被导入，
> Python 会把它注册成包属性 ``api.app``，``uvicorn infusion_monitor.api:app``
> 就拿到模块而不是 ASGI 对象（``TypeError: 'module' object is not callable``）。
> 服务装配因此放在 ``api/service.py``。

本模块对 ``create_app`` / ``app`` 采用惰性导出：``ingest.multipart`` 需要
``api.schemas``，而 ``api.service`` 又需要 ``ingest.multipart``，import 期不能互拉。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from .service import AppStore, create_app

__all__ = ["AppStore", "app", "create_app"]

_LAZY_ATTRS = ("AppStore", "create_app", "app")


def __getattr__(name: str) -> Any:
    # 用 importlib 而不是 `from .service import ...`：后者在惰性场景下会再走一次 __getattr__
    if name in _LAZY_ATTRS:
        return getattr(importlib.import_module(f"{__name__}.service"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")