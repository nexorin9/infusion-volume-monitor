"""生产权重预测器的加载入口：读配置 → 验权重 → 交给适配器。

**这一层的边界**：它不实现推理。推理栈（权重格式、解包、算子、设备搬运）由院内或
本项目后续接的**适配器**提供，通过 :func:`register_weights_adapter` 注册；本模块负责
把「配置读对了没有、权重在不在、能不能读」这些**部署期就会炸**的问题在启动时报清楚，
而不是让主路径跑到一半才抛一个看不懂的异常。

**为什么不静默退回几何核**：退回几何核看起来「服务还能用」，但同一个 ``MODEL_ID``
在两台机上会给出不同的毫升数与 ETA——护士长比对两次巡视记录时才发现，这条路走错根本
查不出来。所以 :class:`~.config_map.WeightsRuntimeUnavailable` 只报错，不退。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .config_map import ModelSpec, WeightsRuntimeUnavailable
from .predictor import PerPointMaskPredictor

__all__ = [
    "WeightsAdapter",
    "WeightsBuildError",
    "load_weights_predictor",
    "read_model_config",
    "register_weights_adapter",
    "weights_adapter_names",
]


class WeightsBuildError(WeightsRuntimeUnavailable):
    """配置/权重本身不合格（读不了、解不出、字段缺）。"""


@dataclass(frozen=True)
class WeightsAdapter:
    """一个权重适配器：吃配置与权重路径，吐一个满足协议的实现。"""

    name: str
    factory: Callable[[Mapping, Path | None, ModelSpec], PerPointMaskPredictor]
    #: 运行这个适配器需要装的东西，报错时直接点名，不让运维去猜。
    requires: tuple[str, ...] = ()

    def describe(self) -> dict:
        return {"name": self.name, "requires": list(self.requires)}


_ADAPTERS: dict[str, WeightsAdapter] = {}


def register_weights_adapter(
    name: str,
    factory: Callable[[Mapping, Path | None, ModelSpec], PerPointMaskPredictor],
    *,
    requires: tuple[str, ...] = (),
) -> None:
    """注册权重适配器。院内自研权重接进来时只需调一次，不必改主路径。"""

    key = name.strip()
    if not key:
        raise ValueError("适配器名不能为空")
    if key in _ADAPTERS and _ADAPTERS[key].factory is not factory:
        raise ValueError(f"适配器 {key!r} 已注册；替换实现会改变出数，须由部署者显式决定")
    _ADAPTERS[key] = WeightsAdapter(name=key, factory=factory, requires=tuple(requires))


def weights_adapter_names() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def read_model_config(path: Path) -> dict:
    """读推理配置：``.yaml`` / ``.yml`` 走 yaml，``.json`` 走标准库。

    yaml 解析器不在依赖清单里（几何核根本不需要它），所以缺解析器时报的是一条能照做的
    错误，而不是 ``ModuleNotFoundError`` 让运维自己猜装什么。
    """

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # 可选依赖：只在权重实现里用
        except ModuleNotFoundError as exc:  # pragma: no cover - 取决于环境
            raise WeightsBuildError(
                f"配置文件 {path} 是 YAML，但本机没装解析器。请在病区侧机上执行 "
                f"`pip install pyyaml` 后重启，或把该配置改写成 JSON 并把 MODEL_CONFIG_PATH 指向它。"
            ) from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise WeightsBuildError(
                f"配置文件 {path} 不是合法 JSON：{exc.msg}（第 {exc.lineno} 行第 {exc.colno} 列）"
            ) from exc
    else:
        raise WeightsBuildError(
            f"配置文件 {path} 后缀是 {suffix or '（无）'}；只认 .yaml / .yml / .json。"
        )

    if not isinstance(data, dict):
        raise WeightsBuildError(
            f"配置文件 {path} 顶层必须是「键 → 值」的映射，实际是 {type(data).__name__}。"
        )
    return data


def load_weights_predictor(
    spec: ModelSpec,
    *,
    adapters: Mapping[str, WeightsAdapter] | None = None,
) -> PerPointMaskPredictor:
    """按 :class:`~.config_map.ModelSpec` 构建生产权重实现。

    顺序刻意与源主路径一致（读配置 → 装权重 → 定设备 → 切 eval），只是每一步失败都
    换成一条能照做的中文错误。配置里缺 ``backend`` 时按 ``torch`` 处理。
    """

    if spec.config_path is None or not spec.config_path.is_file():
        # resolve_model 已经拦过一次；走到这里说明有人绕过解析层直接调了本函数。
        raise WeightsBuildError(
            f"model_id={spec.model_id!r} 的配置文件不存在或未指定，无法构建权重实现。"
            f"请设 MODEL_CONFIG_PATH 指向实际存在的配置文件。"
        )

    config = read_model_config(spec.config_path)
    backend = str(config.get("backend", "torch")).strip() or "torch"

    table = dict(adapters) if adapters is not None else dict(_ADAPTERS)
    adapter = table.get(backend)
    if adapter is None:
        raise WeightsRuntimeUnavailable(
            f"配置 {spec.config_path} 指定 backend={backend!r}，但本机没有对应适配器。"
            f"已注册：{'、'.join(sorted(table)) or '（无）'}。"
            f"请改配置里的 backend，或在进程启动前调 register_weights_adapter({backend!r}, factory) "
            f"注册院区侧的权重实现。本服务不会静默退回几何核——同一 model_id 在两台机上"
            f"会给出不同的毫升数。",
            needs=adapter.requires if adapter else (),
        )

    if spec.checkpoint_path is None or not spec.checkpoint_path.is_file():
        raise WeightsBuildError(
            f"权重文件缺失：{spec.checkpoint_path if spec.checkpoint_path else '（未指定）'}。"
            f"model_id={spec.model_id!r} 走权重实现，必须先有权重文件。"
        )
    if spec.checkpoint_path.stat().st_size == 0:
        raise WeightsBuildError(
            f"权重文件 {spec.checkpoint_path} 是 0 字节（多半是拷贝中断）。"
            f"请重新拷贝后再启动。"
        )

    try:
        predictor = adapter.factory(config, spec.checkpoint_path, spec)
    except WeightsRuntimeUnavailable:
        raise
    except Exception as exc:  # 适配器内部炸了——把它翻译成部署可读的错误
        raise WeightsRuntimeUnavailable(
            f"权重适配器 {backend!r} 加载 {spec.checkpoint_path} 失败：{type(exc).__name__}: {exc}。"
            f"请核对该权重的版本与本机运行时是否匹配。",
            needs=adapter.requires,
        ) from exc

    if not hasattr(predictor, "predict") or not hasattr(predictor, "predict_batch"):
        raise WeightsBuildError(
            f"权重适配器 {backend!r} 产出的对象不满足 PerPointMaskPredictor 协议"
            f"（缺 predict / predict_batch）；主路径不会用一个不满足协议的预测器。"
        )
    return predictor