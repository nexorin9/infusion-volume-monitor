#!/usr/bin/env python3
"""配置自检：启动前把 7 组凭据位挨个点一遍，缺哪个说哪个。

**为什么要有这个命令**：服务本身在缺凭据时会直接起不来（这是刻意的——带着半个
配置跑起来的服务比起不来的服务更危险）。但「起不来」对值班的护士长不是信息：「进程
退出码 1」这句话没法拿去问信息科。这个命令把同样的检查提前跑一遍并翻译成人话：
缺哪组、缺哪个变量名、示例怎么写、在哪改。

    python scripts/check_config.py           # 人读；退出码 0 = 齐，1 = 缺
    python scripts/check_config.py --json    # 机器读（挂到装机脚本里）

顺带做两件**只读**的落地检查：帧目录能不能写、SQLite 文件所在目录能不能建。
这两处配错在病区侧最常见（``/var/lib`` 没权限、挂载点没挂上），而它们的症状是
「上传到一半 500」，很难往配置上想。

本命令**不外发任何请求**：它不连护理执行平台，也不连模型网关。检查出站端点只是把
URL 拼出来看一眼——「地址对不对」要看对端返回，那是 ``scripts/demo_stub_peer.py``
与契约测试的活。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from infusion_monitor.config import CREDENTIAL_GROUPS, require_settings  # noqa: E402
from infusion_monitor.core.config_map import describe_registry  # noqa: E402


def _check_path(label: str, raw: str, *, need_dir: bool) -> dict[str, Any]:
    """只读地看一眼落地目录能不能用（不创建、不删除）。

    目录本身还不存在时，往上找一层能写的父目录：服务启动时会自动建目录，父目录能写
    就行。真写不进去（``/var/lib`` 没权限、挂载点没挂上）在这里就报出来，而不是
    等到上传到一半才 500。
    """
    path = Path(raw).expanduser()
    probe = path if need_dir else path.parent
    result: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "writable": False,
        "ok": False,
        "hint": "",
    }
    if probe.exists():
        result["writable"] = os.access(probe, os.W_OK)
        result["ok"] = result["writable"]
        result["hint"] = "" if result["ok"] else f"{probe} 不可写；见 INSTALL.md 的「目录权限」一节"
        return result

    # probe 还不存在：父目录能写就够了，启动时会自动创建
    parent = probe.parent
    parent_ok = parent.exists() and os.access(parent, os.W_OK)
    result["writable"] = parent_ok
    result["ok"] = parent_ok
    if parent_ok:
        result["hint"] = f"{probe} 还不存在，启动时会自动创建"
    elif not parent.exists():
        result["hint"] = f"{parent} 还不存在，服务无法创建 {probe}；请先手工建目录"
    else:
        result["hint"] = f"{parent} 不可写，服务无法创建 {probe}；见 INSTALL.md 的「目录权限」一节"
    return result


def run_check() -> dict[str, Any]:
    """跑一遍全部检查，返回一份可 JSON 序列化的结果（不抛异常）。"""
    settings, missing, blank = require_settings()
    groups = [
        {
            "key": group.key,
            "label": group.label,
            "env_names": list(group.env_names),
            "example": group.example,
            "present": group not in missing,
            "blank": [name for name in group.env_names if name in set(blank)],
        }
        for group in CREDENTIAL_GROUPS
    ]
    paths: list[dict[str, Any]] = []
    derived: dict[str, Any] = {}
    if settings is not None:
        paths = [
            _check_path("会话帧目录", settings.frames_root, need_dir=True),
            _check_path("SQLite 数据库", settings.sqlite_path or settings.database_url,
                        need_dir=False),
        ]
        derived = {
            "出站写回端点": settings.nurse_exec_events_url,
            "遥测订阅主题": settings.mqtt_telemetry_topic,
            "模型端点": f"{settings.llm_base_url}（模型 {settings.llm_model}）",
            "MQTT 订阅线程": "已启用" if settings.mqtt_enabled else "未启用（视觉主路径不受影响）",
        }

    path_failures = [item for item in paths if not item["ok"]]
    # 预测器注册表：换实现只改 MODEL_IMPL / MODEL_ID，出错是启动期错误。在这里就
    # 把它摊开——「这台机现在会起哪个实现、配置文件在不在、权重在不在」不用等服务起。
    model = None
    if settings is not None:
        try:
            model = describe_registry(settings)
        except Exception as exc:  # 注册表坏了也不能让自检本身崩掉
            model = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    model_failures = []
    if isinstance(model, dict) and model.get("current", {}).get("error"):
        model_failures = [{"label": "预测器注册表", "hint": model["current"]["message"]}]

    ok = settings is not None and not path_failures and not model_failures
    return {
        "ok": ok,
        "credential_groups": groups,
        "missing_group_keys": [group.key for group in missing],
        "blank_env_names": list(blank),
        "paths": paths,
        "model_registry": model,
        "derived": derived,
        "error_text": None if ok else _error_text(missing, blank, path_failures, model_failures),
    }


def _error_text(missing, blank, path_failures, model_failures=()) -> str:
    lines: list[str] = []
    if missing:
        lines.append("启动失败：以下凭据位缺失或为空（不设默认值，请配置后重启）")
        for group in missing:
            absent = [n for n in group.env_names if not os.environ.get(n, "").strip()]
            named = absent or [n for n in group.env_names]
            lines.append(f"  - [{group.key}] {group.label}：缺 {'、'.join(named)}")
            lines.append(f"    示例：{group.example}")
    for item in path_failures:
        lines.append(f"  - [{item['label']}] {item['hint']}")
    for item in model_failures:
        lines.append(f"  - [{item['label']}] {item['hint']}")
    return "\n".join(lines)


def _render(result: dict[str, Any]) -> str:
    lines = ["输液余量监测系统 · 配置自检", ""]
    for group in result["credential_groups"]:
        if group["present"] and not group["blank"]:
            mark, note = "OK  ", ""
        elif group["blank"]:
            mark, note = "空值", f"（已设但为空：{'、'.join(group['blank'])}）"
        else:
            mark, note = "缺失", ""
        names = "、".join(group["env_names"])
        lines.append(f"[{mark}] {group['label']:<28} {names}{note}")
        if mark != "OK  ":
            lines.append(f"       示例：{group['example']}")
    lines.append("")
    if result["paths"]:
        lines.append("落地目录：")
        for item in result["paths"]:
            mark = "OK  " if item["ok"] else "不可写"
            lines.append(f"[{mark}] {item['label']}：{item['path']}")
            if item["hint"]:
                lines.append(f"       {item['hint']}")
        lines.append("")
    registry = result.get("model_registry")
    if isinstance(registry, dict) and registry.get("entries"):
        lines.append("预测器注册表（model_id → 实现 / 配置 / 权重）：")
        current = registry.get("current", {})
        chosen = current.get("model_id")
        for model_id, entry in registry["entries"].items():
            mark = "当前" if model_id == chosen else "    "
            config_mark = "在" if entry["config_present"] else "缺"
            ckpt = "无权重" if not entry["checkpoint_name"] else (
                "在" if entry["checkpoint_present"] else "缺")
            lines.append(
                f"[{mark}] {model_id:<24} 实现={entry['impl']:<11} "
                f"配置={config_mark} 权重={ckpt}"
            )
        if current.get("error"):
            lines.append(f"       当前选择不可用：{current['error']} — {current.get('message', '')}")
        lines.append("")
    if result["derived"]:
        lines.append("按当前配置推出的端点（不外发请求，仅供核对）：")
        for key, value in result["derived"].items():
            lines.append(f"  - {key}：{value}")
        lines.append("")
    if result["ok"]:
        lines.append("结论：配置齐全，可以启动。")
    else:
        lines.append("结论：配置不完整，服务**不会**以缺凭据的状态启动。")
        lines.append(result["error_text"] or "")
        lines.append("")
        lines.append("改 .env（或 systemd 的 EnvironmentFile）后重跑本命令复查。")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="启动前自检 7 组凭据位与落地目录；退出码 0 表示配置齐全。",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON（供装机脚本判断）")
    args = parser.parse_args(argv)

    result = run_check()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(_render(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())