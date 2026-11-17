"""生产配置加载器与预测器注册表：``model_id → (配置文件, 权重)`` 的映射语义移植。

**移植的是哪一段**：源产品主路径的构建入口不接收「一堆参数」，只接收一个
``model_id``，然后用一张映射表把它翻成「哪个配置文件 + 哪个权重文件」，再按这个顺序
建模型：读配置 → 装载权重 → 放到指定设备 → 切到 eval。查表的那一步（``build_sam.py``
里的 ``model_id → (yaml, .pt)`` 映射表与 ``model_id`` 构建入口）与后处理默认值的拼装
（``apply_postprocessing=True`` 时追加的那几条后处理开关）被原样搬到这里：换成病区
侧自己的表、换成环境变量可覆盖，但**「按名字建模型」这条主路径的形状不变**。

**为什么医院侧需要这一层**：输液余量监测要在病区侧机上换实现——本机先用确定性几何核
跑通业务，生产权重到位后只改 ``MODEL_IMPL`` / ``MODEL_ID`` 两个环境变量，不改业务
代码；如果两个实现各自在自己的分支里 ``import``，换实现就变成一次发版。

**三条纪律**（与 config.py 一致）：

1. **不做静默默认值**。权重实现选了却没给权重路径、或权重文件不在盘上，直接抛
   :class:`CheckpointMissingError`，消息里点名缺哪个环境变量、期望路径在哪、怎么配；
2. **映射表可由环境变量覆盖**（``MODEL_REGISTRY_JSON``），供院内多环境切换与试点；
3. **注册表只做解析与装配，不做业务判断**。它产出的是一个满足
   :class:`~.predictor.PerPointMaskPredictor` 协议的对象，掩膜合不合理由调用方的
   契约测试说了算。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .predictor import PerPointMaskPredictor

__all__ = [
    "CheckpointMissingError",
    "ConfigFileMissingError",
    "MODEL_IDS",
    "ModelConfigError",
    "ModelEntry",
    "ModelSpec",
    "PredictorFactory",
    "RegistryOverrideInvalid",
    "UnknownModelIdError",
    "UnknownModelImplError",
    "WeightsRuntimeUnavailable",
    "DEFAULT_POSTPROCESS_OVERRIDES",
    "build_predictor",
    "describe_registry",
    "load_registry",
    "predictor_impl_names",
    "register_predictor",
    "registry_json_help",
    "resolve_model",
]


# ─────────────────────────────────────────────────────────────────────────────
# 异常：一律带「缺什么 / 期望什么 / 怎么配」三段，不留「未知错误」
# ─────────────────────────────────────────────────────────────────────────────
class ModelConfigError(Exception):
    """模型配置与注册表相关错误的基类。"""


class UnknownModelIdError(ModelConfigError):
    """请求的 ``model_id`` 不在注册表里。"""


class UnknownModelImplError(ModelConfigError):
    """``MODEL_IMPL`` 指名的实现没注册过。"""


class ConfigFileMissingError(ModelConfigError):
    """配置文件缺失（权重实现必填；几何核只在显式覆盖时必填）。"""


class CheckpointMissingError(ModelConfigError):
    """权重文件缺失或路径为空。"""


class WeightsRuntimeUnavailable(ModelConfigError):
    """权重运行时/适配器不可用（配置与权重都在，但本机装不上对应的推理栈）。"""

    def __init__(self, message: str, *, needs: Iterable[str] = ()) -> None:
        self.needs = tuple(needs)
        super().__init__(message)


class RegistryOverrideInvalid(ModelConfigError):
    """``MODEL_REGISTRY_JSON`` 不是合法 JSON，或结构不对。"""


# ─────────────────────────────────────────────────────────────────────────────
# 注册表：model_id → (实现名, 配置文件, 权重文件)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelEntry:
    """注册表里的一行：``model_id`` 对应哪个实现、哪份配置、哪份权重。

    ``config_name`` / ``checkpoint_name`` 是**注册表根目录下的相对路径**；写成绝对
    路径也接受（试点时权重挂在院区共享盘上的常见情形）。
    """

    model_id: str
    impl: str
    config_name: str | None = None
    checkpoint_name: str | None = None
    requires_config: bool = False
    requires_checkpoint: bool = False
    description: str = ""

    def config_path(self, root: Path) -> Path | None:
        return _under(root, self.config_name) if self.config_name else None

    def checkpoint_path(self, root: Path) -> Path | None:
        return _under(root, self.checkpoint_name) if self.checkpoint_name else None

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "impl": self.impl,
            "config_name": self.config_name,
            "checkpoint_name": self.checkpoint_name,
            "requires_config": self.requires_config,
            "requires_checkpoint": self.requires_checkpoint,
            "description": self.description,
        }


def _under(root: Path, name: str) -> Path:
    candidate = Path(name)
    return candidate if candidate.is_absolute() else root / candidate


#: 内置映射表。**保留源表的结构**（一个名字 → 一份配置 + 一份权重，规模越大配置与
#: 权重各一份），但名字与文件名全部换成本产品自己的——源项目的模型名不写进产品代码。
_BUILTIN_REGISTRY: tuple[ModelEntry, ...] = (
    ModelEntry(
        model_id="geom_kernel_base",
        impl="geom_kernel",
        config_name="geom_kernel.yaml",
        description="确定性几何核（本机默认实现；无权重，阈值走 SMOOTHING_*/QUALITY_*）",
    ),
    ModelEntry(
        model_id="geom_kernel_low_light",
        impl="geom_kernel",
        config_name="geom_kernel_low_light.yaml",
        description="夜间/暗光档几何核：阈值带与形态学核按暗光合成场景重新标定",
    ),
    ModelEntry(
        model_id="weights_base",
        impl="weights",
        config_name="weights_base.yaml",
        checkpoint_name="weights_base.pt",
        requires_config=True,
        requires_checkpoint=True,
        description="生产权重（基础档）：病区侧机默认生产实现",
    ),
    ModelEntry(
        model_id="weights_base_plus",
        impl="weights",
        config_name="weights_base_plus.yaml",
        checkpoint_name="weights_base_plus.pt",
        requires_config=True,
        requires_checkpoint=True,
        description="生产权重（增强档）：带优先级 2 的工作站",
    ),
    ModelEntry(
        model_id="weights_large",
        impl="weights",
        config_name="weights_large.yaml",
        checkpoint_name="weights_large.pt",
        requires_config=True,
        requires_checkpoint=True,
        description="生产权重（大档）：值班机；本机无 GPU 时按配置回退",
    ),
)

#: 可见的 model_id 列表（报错文案与自检脚本用；不含任何凭据）。
MODEL_IDS: tuple[str, ...] = tuple(entry.model_id for entry in _BUILTIN_REGISTRY)

#: 构建时追加的后处理开关（源主路径 ``apply_postprocessing=True`` 那一段的语义）。
#: 记在 spec 上是为了让「这次到底按什么参数跑的」可回溯，而不是埋在某个函数体里。
DEFAULT_POSTPROCESS_OVERRIDES: tuple[str, ...] = (
    "dynamic_multimask_via_stability=true",
    "dynamic_multimask_stability_delta=0.05",
    "dynamic_multimask_stability_thresh=0.98",
    "binarize_mask_from_pts_for_mem_enc=true",
    "fill_hole_area=8",
)


def registry_json_help() -> str:
    """``MODEL_REGISTRY_JSON`` 的示例文案（报错与自检直接引用，不各写一份）。"""
    return (
        'MODEL_REGISTRY_JSON=\'{"weights_pilot": {"impl": "weights", '
        '"config": "weights_pilot.yaml", "checkpoint": "weights_pilot.pt", '
        '"requires_config": true, "requires_checkpoint": true}}\''
    )


def load_registry(overrides: Mapping[str, object] | str | None = None) -> dict[str, ModelEntry]:
    """载入注册表：内置表 + 环境变量覆盖（同 id 覆盖同名，新增 id 直接加进来）。

    ``overrides`` 传 ``None`` 时读 ``MODEL_REGISTRY_JSON`` 环境变量。覆盖项的键是
    ``model_id``，值支持两种写法：

    * 完整对象：``{"impl": ..., "config": ..., "checkpoint": ..., "requires_config": ...}``
    * 简写字符串：``"weights"``（只换实现名，其余沿用同名内置项）

    解析失败一律抛 :class:`RegistryOverrideInvalid` 并把出错位置与示例一起带出来，
    不静默忽略——映射表被环境变量写坏却照常按内置表启动，是最难查的一类故障。
    """

    return load_registry_with_overrides(overrides)[0]


def load_registry_with_overrides(
    overrides: Mapping[str, object] | str | None = None,
) -> tuple[dict[str, ModelEntry], frozenset[str]]:
    """同 :func:`load_registry`，但一并返回**被覆盖过的 id 集合**。

    分开返回而不是把覆盖标记塞进 ``ModelEntry``：注册表项要保持可哈希、可当常量用，
    「这一行是不是被环境变量改过」属于运行期事实，记在 spec 上（``ModelSpec.overridden``）。
    """

    raw: str | None
    if overrides is None:
        raw = os.environ.get("MODEL_REGISTRY_JSON") or None
    elif isinstance(overrides, str):
        raw = overrides or None
    else:
        raw = json.dumps(overrides, ensure_ascii=False)

    registry: dict[str, ModelEntry] = {entry.model_id: entry for entry in _BUILTIN_REGISTRY}
    if raw is None:
        return registry, frozenset()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryOverrideInvalid(
            "MODEL_REGISTRY_JSON 不是合法 JSON："
            f"{exc.msg}（第 {exc.lineno} 行第 {exc.colno} 列）。"
            f"正确写法示例：{registry_json_help()}"
        ) from exc

    if not isinstance(parsed, dict):
        raise RegistryOverrideInvalid(
            f"MODEL_REGISTRY_JSON 顶层必须是 {{model_id: 覆盖项}} 的对象，"
            f"实际拿到 {type(parsed).__name__}。正确写法示例：{registry_json_help()}"
        )

    for model_id, patch in parsed.items():
        registry[model_id] = _merge_entry(registry.get(model_id), str(model_id), patch)
    return registry, frozenset(str(key) for key in parsed)


def _merge_entry(base: ModelEntry | None, model_id: str, patch: object) -> ModelEntry:
    if isinstance(patch, str):
        patch = {"impl": patch}
    if not isinstance(patch, dict):
        raise RegistryOverrideInvalid(
            f"MODEL_REGISTRY_JSON 里 {model_id} 的覆盖项必须是对象或实现名字符串，"
            f"实际拿到 {type(patch).__name__}。正确写法示例：{registry_json_help()}"
        )

    impl = patch.get("impl")
    config = patch.get("config", getattr(base, "config_name", None))
    checkpoint = patch.get("checkpoint", getattr(base, "checkpoint_name", None))
    impl_name = str(impl or getattr(base, "impl", "") or "")
    if not impl_name:
        raise RegistryOverrideInvalid(
            f"MODEL_REGISTRY_JSON 里 {model_id} 既没有 impl，也没有同名内置项可继承。至少要写 impl。"
        )
    if config is not None and not isinstance(config, str):
        raise RegistryOverrideInvalid(f"MODEL_REGISTRY_JSON 里 {model_id} 的 config 必须是字符串路径")
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise RegistryOverrideInvalid(f"MODEL_REGISTRY_JSON 里 {model_id} 的 checkpoint 必须是字符串路径")

    return ModelEntry(
        model_id=model_id,
        impl=impl_name,
        config_name=config,
        checkpoint_name=checkpoint,
        # 覆盖项没显式写这两个门限时按「有没有文件」推导：写了 config 就认为要用它，
        # 写了 checkpoint 就认为要走权重路径，避免改了路径却没跟着改门限。
        requires_config=bool(patch.get("requires_config", config is not None)),
        requires_checkpoint=bool(patch.get("requires_checkpoint", checkpoint is not None)),
        description=str(patch.get("description", getattr(base, "description", "") or "")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 解析与校验：model_id → ModelSpec（每一步不通过都点名缺什么）
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    """一次构建决策的完整快照：谁、按什么配置、从哪加载权重、带了哪些后处理开关。"""

    model_id: str
    impl: str
    config_path: Path | None = None
    checkpoint_path: Path | None = None
    requires_config: bool = False
    requires_checkpoint: bool = False
    config_explicit: bool = False
    checkpoint_explicit: bool = False
    apply_postprocessing: bool = True
    overrides: tuple[str, ...] = ()
    description: str = ""
    #: 注册表项被环境变量覆盖过——运行摘要里如实标出来，便于回溯「当时配的是什么」。
    overridden: bool = False

    @property
    def config_present(self) -> bool:
        return self.config_path is not None and self.config_path.is_file()

    @property
    def checkpoint_present(self) -> bool:
        return self.checkpoint_path is not None and self.checkpoint_path.is_file()

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "impl": self.impl,
            "config_path": str(self.config_path) if self.config_path else None,
            "config_present": self.config_present,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "checkpoint_present": self.checkpoint_present,
            "apply_postprocessing": self.apply_postprocessing,
            "overrides": list(self.overrides),
            "overridden": self.overridden,
            "description": self.description,
        }


def resolve_model(
    model_id: str,
    *,
    impl: str | None = None,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    registry_root: str | Path = "./models",
    registry: Mapping[str, ModelEntry] | None = None,
    apply_postprocessing: bool = True,
) -> ModelSpec:
    """把 ``model_id`` 解析成可构建的 :class:`ModelSpec`，并就地做硬校验。

    ``impl`` / ``config_path`` / ``checkpoint_path`` 为 ``None`` 表示**未显式覆盖**，
    走注册表；给了非空值就是环境变量覆盖，优先级最高。几何核的配置文件默认允许缺席
    （阈值走环境变量），但**显式指定了一个不存在的路径时同样报错**——运维以为配上了
    却没人告诉他文件没到位，这类故障在病区侧要等到第一次上传失败才暴露。
    """

    overridden_ids: frozenset[str]
    if registry is not None:
        table: dict[str, ModelEntry] = dict(registry)
        overridden_ids = frozenset()
    else:
        table, overridden_ids = load_registry_with_overrides()
    root = Path(registry_root)

    entry = table.get(model_id)
    if entry is None:
        known = "、".join(sorted(table)) or "（空）"
        raise UnknownModelIdError(
            f"未知 model_id={model_id!r}。当前注册表可用的是：{known}。"
            f"改用其中之一，或用 MODEL_REGISTRY_JSON 追加该 id，示例：{registry_json_help()}"
        )

    resolved_impl = impl or entry.impl
    if not resolved_impl.strip():
        raise UnknownModelImplError(
            f"model_id={model_id!r} 解析出的实现名为空。请设 MODEL_IMPL，或在 "
            f"MODEL_REGISTRY_JSON 里补 impl。示例：{registry_json_help()}"
        )

    config_explicit = config_path is not None and str(config_path).strip() != ""
    checkpoint_explicit = checkpoint_path is not None and str(checkpoint_path).strip() != ""

    config_p: Path | None
    if config_explicit:
        config_p = _under(root, str(config_path))
    elif entry.config_name:
        config_p = entry.config_path(root)
    else:
        config_p = None

    checkpoint_p: Path | None
    if checkpoint_explicit:
        checkpoint_p = _under(root, str(checkpoint_path))
    elif entry.checkpoint_name:
        checkpoint_p = entry.checkpoint_path(root)
    else:
        checkpoint_p = None

    spec = ModelSpec(
        model_id=model_id,
        impl=resolved_impl,
        config_path=config_p,
        checkpoint_path=checkpoint_p,
        requires_config=entry.requires_config,
        requires_checkpoint=entry.requires_checkpoint,
        config_explicit=config_explicit,
        checkpoint_explicit=checkpoint_explicit,
        apply_postprocessing=apply_postprocessing,
        overrides=DEFAULT_POSTPROCESS_OVERRIDES if apply_postprocessing else (),
        description=entry.description,
        overridden=bool(
            impl
            or config_explicit
            or checkpoint_explicit
            or model_id in overridden_ids
            or model_id not in MODEL_IDS
        ),
    )

    _validate(spec, root)
    return spec


def _validate(spec: ModelSpec, root: Path) -> None:
    """硬校验：实现名 → 配置文件 → 权重文件，按「越基础越先报」的顺序一次性说清楚。

    实现名排在最前是有意的：配了一个不存在的实现时，先报「这个实现没注册」比先报
    「这个实现要的权重没找到」更接近根因——后者只是换了个名字后必然出现的连带错误。
    """

    if spec.impl not in _PREDICTOR_FACTORIES:
        raise UnknownModelImplError(
            f"未知实现 MODEL_IMPL={spec.impl!r}（model_id={spec.model_id!r}）。"
            f"已注册的实现：{'、'.join(predictor_impl_names())}。"
            f"若这是院区自研实现，先调 register_predictor({spec.impl!r}, factory) 注册再启动。"
        )

    if spec.requires_config or spec.config_explicit:
        if spec.config_path is None:
            raise ConfigFileMissingError(
                f"model_id={spec.model_id!r} 需要一份推理配置，但 MODEL_CONFIG_PATH 为空且"
                f"注册表项也没登记配置。请设 MODEL_CONFIG_PATH=<路径>（相对路径按注册表根目录 "
                f"{root} 解析），或在 MODEL_REGISTRY_JSON 里补 config。"
            )
        if not spec.config_path.is_file():
            where = "显式指定" if spec.config_explicit else "注册表项登记"
            raise ConfigFileMissingError(
                f"找不到推理配置文件 {spec.config_path}（{where}）。"
                f"请把配置文件放到该路径，或改 MODEL_CONFIG_PATH / MODEL_REGISTRY_JSON 指向实际位置。"
            )

    if not spec.requires_checkpoint:
        return

    if spec.checkpoint_path is None:
        raise CheckpointMissingError(
            f"model_id={spec.model_id!r} 走的是权重实现（impl={spec.impl!r}），必须有权重文件。"
            f"请设 MODEL_CHECKPOINT=<权重路径>（相对路径按注册表根目录 {root} 解析），"
            f"或在 MODEL_REGISTRY_JSON 里给该 id 补 checkpoint。"
        )
    if not spec.checkpoint_path.is_file():
        where = "显式指定" if spec.checkpoint_explicit else "注册表项登记"
        raise CheckpointMissingError(
            f"找不到权重文件 {spec.checkpoint_path}（{where}）。"
            f"请把权重放到该路径并保证进程可读（病区侧机的属主与权限见 INSTALL.md），"
            f"或改 MODEL_CHECKPOINT / MODEL_REGISTRY_JSON 指向实际位置。缺失时本服务不会"
            f"静默退回几何核——换个实现出数，毫升数与 ETA 会变，必须由部署者显式决定。"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 预测器注册表：实现名 → 工厂；两个实现共用同一个协议、同一入口
# ─────────────────────────────────────────────────────────────────────────────
PredictorFactory = Callable[[ModelSpec], PerPointMaskPredictor]


def _geom_kernel_factory(spec: ModelSpec) -> PerPointMaskPredictor:
    """确定性几何核：无权重、可在任意一台病区侧机上跑起来。"""
    from .geom_kernel import GeometricMaskPredictor

    return GeometricMaskPredictor()


def _weights_factory(spec: ModelSpec) -> PerPointMaskPredictor:
    """生产权重实现：走 :mod:`.weights_predictor` 的配置/权重校验与适配器注册表。"""
    from .weights_predictor import load_weights_predictor

    return load_weights_predictor(spec)


_PREDICTOR_FACTORIES: dict[str, PredictorFactory] = {
    "geom_kernel": _geom_kernel_factory,
    "weights": _weights_factory,
}


def register_predictor(impl: str, factory: PredictorFactory) -> None:
    """注册一个预测器实现。同名覆盖用于试点切换，不覆盖时抛错以免静默顶掉。"""

    name = impl.strip()
    if not name:
        raise ValueError("实现名不能为空")
    if name in _PREDICTOR_FACTORIES and _PREDICTOR_FACTORIES[name] is not factory:
        raise ValueError(
            f"实现名 {name!r} 已注册；如需替换请先确认是有意为之（换实现会改变出数，"
            f"必须由部署者显式决定，而不是被一次 import 悄悄顶掉）"
        )
    _PREDICTOR_FACTORIES[name] = factory


def predictor_impl_names() -> tuple[str, ...]:
    """已注册的实现名（自检与报错文案用）。"""
    return tuple(sorted(_PREDICTOR_FACTORIES))


def _selection_from_settings(settings: object | None) -> dict:
    """从配置面上读出「选择哪一项 + 覆盖了什么」。

    **只有显式设置过的字段才算覆盖**（pydantic 把来自环境变量/构造参数的字段记在
    ``model_fields_set`` 里，代码里的默认值不算）。否则 ``model_impl`` 的默认值会永远
    盖住注册表：把 ``MODEL_ID`` 改成 ``weights_base`` 却忘了改 ``MODEL_IMPL``，构建出来的
    就成了几何核——换实现变成一个静默的陷阱。
    """

    if settings is None:
        return {
            "model_id": "geom_kernel_base",
            "impl": None,
            "config_path": None,
            "checkpoint_path": None,
            "registry_root": "./models",
            "registry_json": None,
        }

    provided = set(getattr(settings, "model_fields_set", ()) or ())
    impl = getattr(settings, "model_impl", None) or None
    return {
        "model_id": getattr(settings, "model_id", None) or "geom_kernel_base",
        "impl": impl if "model_impl" in provided else None,
        "config_path": getattr(settings, "model_config_path", None) or None,
        "checkpoint_path": getattr(settings, "model_checkpoint", None) or None,
        "registry_root": getattr(settings, "model_registry_root", None) or "./models",
        "registry_json": getattr(settings, "model_registry_json", None) or None,
    }


def build_predictor(
    settings: object | None = None,
    *,
    registry: Mapping[str, ModelEntry] | None = None,
    apply_postprocessing: bool = True,
) -> PerPointMaskPredictor:
    """**业务层取预测器的唯一入口**：按配置面解析 → 校验 → 调工厂。

    传 ``settings``（:class:`infusion_monitor.config.Settings`）时用
    ``MODEL_IMPL`` / ``MODEL_ID`` / ``MODEL_CONFIG_PATH`` / ``MODEL_CHECKPOINT`` /
    ``MODEL_REGISTRY_ROOT`` / ``MODEL_REGISTRY_JSON`` 六个配置项；不传则全部走内置默认
    （等价于 ``MODEL_ID=geom_kernel_base``，注册表把它解析成几何核）。
    """

    selection = _selection_from_settings(settings)
    effective_registry = registry
    if selection["registry_json"]:
        effective_registry = load_registry(selection["registry_json"])

    spec = resolve_model(
        selection["model_id"],
        impl=selection["impl"],
        config_path=selection["config_path"],
        checkpoint_path=selection["checkpoint_path"],
        registry_root=selection["registry_root"],
        registry=effective_registry,
        apply_postprocessing=apply_postprocessing,
    )

    factory = _PREDICTOR_FACTORIES[spec.impl]  # resolve_model 已校验过注册情况
    predictor = factory(spec)
    if not hasattr(predictor, "predict") or not hasattr(predictor, "predict_batch"):
        raise ModelConfigError(
            f"实现 {spec.impl!r} 产出的对象不满足 PerPointMaskPredictor 协议"
            f"（缺 predict / predict_batch）；主路径不会用一个不满足协议的预测器。"
        )
    return predictor


def describe_registry(
    settings: object | None = None,
    *,
    registry: Mapping[str, ModelEntry] | None = None,
) -> dict:
    """注册表与当前选择的可打印摘要（**不含任何凭据**，只有路径与门限）。

    给 ``scripts/check_config.py`` 与运行摘要用：启动前就能看见「这台机现在会起哪个
    实现、配置文件在不在、权重在不在」，而不是等第一次上传才炸。
    """

    table = dict(registry) if registry is not None else load_registry()
    root = Path(_registry_root(settings))
    entries: dict[str, dict] = {}
    for model_id, entry in sorted(table.items()):
        config_p = entry.config_path(root)
        checkpoint_p = entry.checkpoint_path(root)
        entries[model_id] = {
            **entry.as_dict(),
            "config_path": str(config_p) if config_p else None,
            "config_present": bool(config_p and config_p.is_file()),
            "checkpoint_path": str(checkpoint_p) if checkpoint_p else None,
            "checkpoint_present": bool(checkpoint_p and checkpoint_p.is_file()),
        }
    try:
        selection = _selection_from_settings(settings)
        current = resolve_model(
            selection["model_id"],
            impl=selection["impl"],
            config_path=selection["config_path"],
            checkpoint_path=selection["checkpoint_path"],
            registry_root=selection["registry_root"],
            registry=table,
        ).as_dict()
    except ModelConfigError as exc:
        current = {"error": type(exc).__name__, "message": str(exc)}
    return {
        "registry_root": _registry_root(settings),
        "registered_impls": list(predictor_impl_names()),
        "entries": entries,
        "current": current,
    }


def _registry_root(settings: object | None) -> str:
    return getattr(settings, "model_registry_root", None) or "./models"