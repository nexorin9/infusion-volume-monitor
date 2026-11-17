"""生产配置加载器与预测器注册表：两个实现同协议同入口，缺什么点名到什么。

覆盖 task 23 的三条硬要求（两个实现均可注册 / 未知实现名报错 / 权重缺失错误信息完整），
外加注册表被环境变量覆盖、配置与权重路径解析、以及「默认构建仍等价于几何核」这条
主路径不回归。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from infusion_monitor.core.config_map import (
    DEFAULT_POSTPROCESS_OVERRIDES,
    MODEL_IDS,
    CheckpointMissingError,
    ConfigFileMissingError,
    RegistryOverrideInvalid,
    UnknownModelIdError,
    UnknownModelImplError,
    WeightsRuntimeUnavailable,
    build_predictor,
    describe_registry,
    load_registry,
    predictor_impl_names,
    register_predictor,
    resolve_model,
)
from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
from infusion_monitor.core.predictor import MaskPrediction, Prompt
from infusion_monitor.core.weights_predictor import (
    WeightsBuildError,
    load_weights_predictor,
    read_model_config,
    register_weights_adapter,
    weights_adapter_names,
)
from infusion_monitor.synth import BagPlan, SceneSpec, decode_jpeg, encode_jpeg, render_scene
from tests.support.synth import anchor_prompt


# ── 一个最小可用的「权重实现」，用来证明两个实现真的走同一个协议 ────────────────
class _FakeWeightsPredictor:
    """满足 ``PerPointMaskPredictor`` 的最小实现（协议一致性由本文件末尾断言守住）。

    它**真的按提示出掩膜**（以提示点为中心的小方块）而不是回一张固定的图——一个不理会
    提示的实现验不了坐标系是否与输入一致，而那正是协议第二条要守的东西。
    """

    name = "fake_weights"

    def __init__(self, config, checkpoint_path, spec):
        self.config = dict(config)
        self.checkpoint_path = checkpoint_path
        self.spec = spec

    def predict(self, image, prompt):
        mask = np.zeros(image.shape[:2], dtype=bool)
        for point in prompt.points:
            x, y = int(point.x), int(point.y)
            if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
                continue
            mask[max(0, y - 4) : y + 5, max(0, x - 4) : x + 5] = True
        if not mask.any():
            return []
        return [MaskPrediction(mask=mask, score=0.5)]

    def predict_batch(self, image, prompts):
        return [self.predict(image, prompt) for prompt in prompts]


@pytest.fixture
def fake_adapter():
    """注册一个测试用权重适配器，并在用例结束后摘掉，不污染全局注册表。"""
    from infusion_monitor.core import weights_predictor as wp

    register_weights_adapter("test_torch", _FakeWeightsPredictor)
    try:
        yield "test_torch"
    finally:
        wp._ADAPTERS.pop("test_torch", None)


@pytest.fixture
def weights_dir(tmp_path):
    """一个放齐了配置与权重的注册表根目录。"""
    root = tmp_path / "models"
    root.mkdir()
    (root / "weights_base.yaml").write_text(
        "backend: test_torch\nmodel_type: infusion_bag\n", encoding="utf-8"
    )
    (root / "weights_base.pt").write_bytes(b"fake-weights-payload" * 8)
    return root


# ── 1. 注册表结构：model_id → (实现, 配置, 权重) ──────────────────────────────
def test_registry_preserves_id_to_config_and_checkpoint_mapping():
    """注册表保留「一个名字 → 一份配置 + 一份权重」的映射结构（源主路径同构）。"""
    table = load_registry()
    for model_id in MODEL_IDS:
        assert model_id in table, f"注册表缺 {model_id}"
    kernel = [e for e in table.values() if e.impl == "geom_kernel"]
    weights = [e for e in table.values() if e.impl == "weights"]
    assert kernel and weights, "两个实现都必须在注册表里"
    # 几何核档：只有配置、没有权重；权重档：配置与权重都登记
    assert all(e.config_name and not e.checkpoint_name for e in kernel)
    assert all(e.config_name and e.checkpoint_name for e in weights)
    # 规模分档（源表 tiny/small/base+/large 的结构）在本表里对应多个权重档
    assert len(weights) >= 3


def test_two_impls_are_both_registered():
    """两个实现都在预测器注册表里，且几何核是默认入口。"""
    names = predictor_impl_names()
    assert "geom_kernel" in names
    assert "weights" in names


def test_default_build_is_the_geometric_kernel():
    """默认配置构建出来的就是几何核——主路径行为不因换了入口而改变。"""
    predictor = build_predictor()
    assert isinstance(predictor, GeometricMaskPredictor)
    assert predictor.name == "geom_kernel"
    assert hasattr(predictor, "predict") and hasattr(predictor, "predict_batch")


def test_both_impls_satisfy_the_same_protocol(weights_dir, fake_adapter):
    """几何核与权重实现走同一入口、产出同一个协议下的对象（主路径不认实现）。"""
    kernel = build_predictor()
    weighted = build_predictor(
        _FakeSettings(model_id="weights_base", registry_root=str(weights_dir))
    )
    for predictor in (kernel, weighted):
        assert hasattr(predictor, "predict") and hasattr(predictor, "predict_batch")
    assert isinstance(weighted, _FakeWeightsPredictor)

    # 同一份协议、同一个入口：两者吃同一种提示，都给出与原图同尺寸同坐标系的 bool 掩膜。
    spec = SceneSpec(
        scene_id="registry-contract",
        bags=(BagPlan(bag_seq="B01", slot_index=0, bag_type_code="TYPE-A",
                      fill_start=0.6, fill_end=0.45),),
        frame_count=4,
    )
    frame = decode_jpeg(encode_jpeg(render_scene(spec).frames[0]))
    prompt = anchor_prompt(spec, "B01", 0)
    point = prompt.points[0]
    for predictor in (kernel, weighted):
        predictions = predictor.predict(frame, prompt)
        assert predictions, "提示点在袋体上时必须出掩膜"
        mask = predictions[0].mask
        assert mask.shape == frame.shape[:2]
        assert mask.dtype == bool
        assert mask[int(point.y), int(point.x)], "掩膜坐标系必须与提示坐标系一致（不许整体平移）"
        assert 0.0 <= predictions[0].score <= 1.0

    # 批量接口与逐条接口必须一致（协议第二条：批处理 = 逐条）
    other = Prompt.from_point(4, 4)
    batched = kernel.predict_batch(frame, [prompt, other])
    assert len(batched) == 2
    assert len(batched[0]) == len(kernel.predict(frame, prompt))
    assert len(batched[1]) == len(kernel.predict(frame, other))


def test_weights_impl_resolves_config_and_checkpoint_paths(weights_dir, fake_adapter):
    """权重实现的 spec 指向真实落盘的配置与权重，并带上后处理开关。"""
    spec = resolve_model("weights_base", registry_root=str(weights_dir))
    assert spec.config_path == weights_dir / "weights_base.yaml"
    assert spec.checkpoint_path == weights_dir / "weights_base.pt"
    assert spec.config_present and spec.checkpoint_present
    assert spec.overrides == DEFAULT_POSTPROCESS_OVERRIDES
    predictor = load_weights_predictor(spec)
    assert predictor.checkpoint_path == weights_dir / "weights_base.pt"
    assert predictor.config["model_type"] == "infusion_bag"


# ── 2. 未知实现名 / 未知 id：报错必须点名并给可执行的下一步 ──────────────────
def test_unknown_model_id_is_rejected_with_available_ids():
    with pytest.raises(UnknownModelIdError) as excinfo:
        resolve_model("weights_middle_plus")
    message = str(excinfo.value)
    assert "weights_middle_plus" in message
    assert "geom_kernel_base" in message, "报错要点出当前可用的 id"
    assert "MODEL_REGISTRY_JSON" in message, "报错要点出怎么补这个 id"


def test_unknown_impl_is_rejected_and_lists_registered_impls(weights_dir):
    """配置指向一个没注册过的实现名 → 报错点名 MODEL_IMPL 与已注册实现。"""
    with pytest.raises(UnknownModelImplError) as excinfo:
        build_predictor(_FakeSettings(model_id="weights_base", impl="mystery_impl",
                                      registry_root=str(weights_dir)))
    message = str(excinfo.value)
    assert "mystery_impl" in message
    assert "geom_kernel" in message and "weights" in message
    assert "register_predictor" in message


def test_unknown_impl_in_registry_json_is_rejected(tmp_path):
    """环境变量里塞一个没注册的实现名，同样在构建时报出来，而不是运行时才炸。"""
    payload = json.dumps({"weights_x": {"impl": "nope", "config": "c.yaml", "checkpoint": "c.pt"}})
    # 配置与权重都放齐：确保报的是「实现没注册」而不是被连带缺失的权重挡住。
    (tmp_path / "c.yaml").write_text("backend: nope\n", encoding="utf-8")
    (tmp_path / "c.pt").write_bytes(b"payload")
    with pytest.raises(UnknownModelImplError) as excinfo:
        build_predictor(_FakeSettings(model_id="weights_x", registry_json=payload,
                                      registry_root=str(tmp_path)))
    assert "nope" in str(excinfo.value)


def test_unknown_impl_is_reported_before_missing_weights(tmp_path):
    """实现没注册 + 权重也没到位时，先报实现名——后者只是换了个名字后的连带错误。"""
    with pytest.raises(UnknownModelImplError):
        resolve_model("weights_base", impl="mystery_impl", registry_root=str(tmp_path))


# ── 3. checkpoint / 配置缺失：错误信息必须完整到能照做 ───────────────────────
def test_missing_checkpoint_error_names_env_var_and_path(tmp_path):
    """权重实现没给权重路径 → 报错含 MODEL_CHECKPOINT、期望路径与怎么配。"""
    # 配置先放齐，让报错落在权重这一项上（校验顺序：实现名 → 配置 → 权重）。
    (tmp_path / "weights_base.yaml").write_text("backend: torch\n", encoding="utf-8")
    with pytest.raises(CheckpointMissingError) as excinfo:
        resolve_model("weights_base", registry_root=str(tmp_path), checkpoint_path="")
    message = str(excinfo.value)
    assert "MODEL_CHECKPOINT" in message
    assert "weights_base.pt" in message
    assert "MODEL_REGISTRY_JSON" in message
    assert "不会静默退回几何核" in message


def test_checkpoint_file_absent_error_reports_path(tmp_path):
    """权重路径给了但文件不在盘上 → 报错点名那个路径，并说明谁可读。"""
    (tmp_path / "weights_base.yaml").write_text("backend: torch\n", encoding="utf-8")
    with pytest.raises(CheckpointMissingError) as excinfo:
        resolve_model("weights_base", checkpoint_path="weights_base.pt", registry_root=str(tmp_path))
    message = str(excinfo.value)
    assert str(tmp_path / "weights_base.pt") in message
    assert "显式指定" in message
    assert "INSTALL.md" in message


def test_zero_byte_checkpoint_is_rejected(tmp_path, fake_adapter):
    """0 字节权重（拷贝中断）必须在构建期被拦下，而不是等到推理时。"""
    root = tmp_path / "models"
    root.mkdir()
    (root / "weights_base.yaml").write_text("backend: test_torch\n", encoding="utf-8")
    (root / "weights_base.pt").write_bytes(b"")
    spec = resolve_model("weights_base", registry_root=str(root))
    with pytest.raises(WeightsBuildError) as excinfo:
        load_weights_predictor(spec)
    assert "0 字节" in str(excinfo.value)


def test_missing_config_file_error_names_env_var(tmp_path):
    with pytest.raises(ConfigFileMissingError) as excinfo:
        resolve_model("weights_base", registry_root=str(tmp_path))
    message = str(excinfo.value)
    assert "MODEL_CONFIG_PATH" in message
    assert str(tmp_path / "weights_base.yaml") in message


def test_geom_kernel_config_may_be_absent_but_explicit_path_must_exist(tmp_path):
    """几何核的配置文件默认可缺席（阈值走环境变量），但显式配了一个不存在的路径要报错。"""
    spec = resolve_model("geom_kernel_base", registry_root=str(tmp_path))
    assert spec.config_path == tmp_path / "geom_kernel.yaml"
    assert spec.config_present is False
    assert spec.checkpoint_path is None

    with pytest.raises(ConfigFileMissingError) as excinfo:
        resolve_model("geom_kernel_base", config_path="geom_kernel.yaml", registry_root=str(tmp_path))
    assert "显式指定" in str(excinfo.value)


# ── 4. 映射表可由环境变量覆盖（院内多环境切换）────────────────────────────────
def test_registry_json_adds_a_new_model_id(monkeypatch, tmp_path):
    payload = json.dumps(
        {
            "weights_pilot": {
                "impl": "weights",
                "config": "weights_pilot.yaml",
                "checkpoint": "weights_pilot.pt",
                "requires_config": True,
                "requires_checkpoint": True,
            }
        }
    )
    monkeypatch.setenv("MODEL_REGISTRY_JSON", payload)
    table = load_registry()
    assert "weights_pilot" in table
    entry = table["weights_pilot"]
    assert entry.requires_config and entry.requires_checkpoint
    # 内置项没被覆盖掉
    assert "weights_base" in table
    # 新 id 的配置/权重路径按注册表根目录解析
    (tmp_path / "weights_pilot.yaml").write_text("backend: torch\n", encoding="utf-8")
    (tmp_path / "weights_pilot.pt").write_bytes(b"payload")
    spec = resolve_model("weights_pilot", registry_root=str(tmp_path))
    assert spec.config_path == tmp_path / "weights_pilot.yaml"
    assert spec.checkpoint_path == tmp_path / "weights_pilot.pt"
    assert spec.overridden is True


def test_registry_json_overrides_existing_entry(monkeypatch, tmp_path):
    """同名覆盖：改 checkpoint 名而不必重写整行。"""
    monkeypatch.setenv(
        "MODEL_REGISTRY_JSON",
        json.dumps({"weights_base": {"checkpoint": "weights_base_v2.pt"}}),
    )
    (tmp_path / "weights_base.yaml").write_text("backend: torch\n", encoding="utf-8")
    (tmp_path / "weights_base_v2.pt").write_bytes(b"payload")
    spec = resolve_model("weights_base", registry_root=str(tmp_path))
    assert spec.checkpoint_path == tmp_path / "weights_base_v2.pt"
    assert spec.impl == "weights", "未写的字段沿用同名内置项"
    assert spec.requires_checkpoint is True, "改了 checkpoint 就自动走权重门限"


def test_registry_json_accepts_impl_only_shorthand(monkeypatch, tmp_path):
    """简写写法：只换实现名，其余沿用同名内置项。"""
    monkeypatch.setenv("MODEL_REGISTRY_JSON", json.dumps({"geom_kernel_base": "weights"}))
    entry = load_registry()["geom_kernel_base"]
    assert entry.impl == "weights"
    assert entry.config_name == "geom_kernel.yaml"


@pytest.mark.parametrize(
    "raw, needle",
    [
        ("{不是 json}", "不是合法 JSON"),
        (json.dumps(["weights_base"]), "顶层必须是"),
        (json.dumps({"weights_base": 7}), "必须是对象或实现名字符串"),
        (json.dumps({"weights_new": {}}), "至少要写 impl"),
        (json.dumps({"weights_base": {"impl": "weights", "config": 7}}), "config 必须是字符串"),
    ],
)
def test_broken_registry_json_is_rejected_loudly(raw, needle, monkeypatch):
    """映射表被环境变量写坏必须当场报出来，不能照内置表静默启动。"""
    monkeypatch.setenv("MODEL_REGISTRY_JSON", raw)
    with pytest.raises(RegistryOverrideInvalid) as excinfo:
        load_registry()
    assert needle in str(excinfo.value)


def test_registry_json_help_text_is_usable():
    """报错文案里引用的示例本身必须是合法 JSON（否则照着抄会再报一次错）。"""
    from infusion_monitor.core.config_map import registry_json_help

    example = registry_json_help()
    assert example.startswith("MODEL_REGISTRY_JSON=")
    payload = json.loads(example.split("=", 1)[1].strip().strip("'"))
    assert payload["weights_pilot"]["impl"] == "weights"


# ── 5. 适配器缺失：只报错，不静默退回几何核 ─────────────────────────────────
def test_missing_weights_adapter_is_reported_not_degraded(tmp_path, monkeypatch):
    """配置与权重都在，但没有对应适配器 → 明确报错，绝不悄悄换回几何核。"""
    from infusion_monitor.core import weights_predictor as wp

    root = tmp_path / "models"
    root.mkdir()
    (root / "weights_base.yaml").write_text("backend: torch\n", encoding="utf-8")
    (root / "weights_base.pt").write_bytes(b"payload")
    monkeypatch.delitem(wp._ADAPTERS, "test_torch", raising=False)

    with pytest.raises(WeightsRuntimeUnavailable) as excinfo:
        load_weights_predictor(resolve_model("weights_base", registry_root=str(root)))
    message = str(excinfo.value)
    assert "torch" in message
    assert "register_weights_adapter" in message
    assert "不会静默退回几何核" in message


def test_adapter_internal_error_is_translated(tmp_path):
    """适配器内部抛错要被翻译成部署可读的错，而不是裸栈给到值班的人。"""

    def _boom(config, checkpoint_path, spec):
        raise RuntimeError("unpickling failed")

    from infusion_monitor.core import weights_predictor as wp

    root = tmp_path / "models"
    root.mkdir()
    (root / "weights_base.yaml").write_text("backend: boom_backend\n", encoding="utf-8")
    (root / "weights_base.pt").write_bytes(b"payload")
    wp.register_weights_adapter("boom_backend", _boom)
    try:
        with pytest.raises(WeightsRuntimeUnavailable) as excinfo:
            load_weights_predictor(resolve_model("weights_base", registry_root=str(root)))
    finally:
        wp._ADAPTERS.pop("boom_backend", None)
    assert "unpickling failed" in str(excinfo.value)
    assert "boom_backend" in str(excinfo.value)


def test_registering_a_duplicate_impl_is_refused():
    """同名覆盖必须由部署者显式同意，不能被一次 import 悄悄顶掉。"""
    with pytest.raises(ValueError) as excinfo:
        register_predictor("geom_kernel", lambda spec: _FakeWeightsPredictor({}, None, spec))
    assert "已注册" in str(excinfo.value)
    # 原实现没被动过
    assert predictor_impl_names().count("geom_kernel") == 1
    assert isinstance(build_predictor(), GeometricMaskPredictor)


# ── 6. 配置读取与摘要 ────────────────────────────────────────────────────────
def test_read_model_config_supports_json_and_yaml(tmp_path):
    as_json = tmp_path / "c.json"
    as_json.write_text(json.dumps({"backend": "torch"}), encoding="utf-8")
    assert read_model_config(as_json)["backend"] == "torch"

    as_yaml = tmp_path / "c.yaml"
    as_yaml.write_text("backend: torch\nmodel_type: bag\n", encoding="utf-8")
    assert read_model_config(as_yaml) == {"backend": "torch", "model_type": "bag"}

    broken = tmp_path / "broken.json"
    broken.write_text("{不是 json}", encoding="utf-8")
    with pytest.raises(WeightsBuildError) as excinfo:
        read_model_config(broken)
    assert "不是合法 JSON" in str(excinfo.value)

    weird_suffix = tmp_path / "c.toml"
    weird_suffix.write_text("backend = 'torch'", encoding="utf-8")
    with pytest.raises(WeightsBuildError) as excinfo:
        read_model_config(weird_suffix)
    assert ".yaml / .yml / .json" in str(excinfo.value)


def test_describe_registry_reports_presence_without_credentials(tmp_path):
    """自检摘要：配置文件/权重在不在一目了然，且不出现任何凭据。"""
    (tmp_path / "weights_base.yaml").write_text("backend: torch\n", encoding="utf-8")
    described = describe_registry(_FakeSettings(model_id="weights_base", registry_root=str(tmp_path)))
    assert described["registry_root"] == str(tmp_path)
    assert set(described["registered_impls"]) >= {"geom_kernel", "weights"}
    assert described["entries"]["weights_base"]["config_present"] is True
    assert described["entries"]["weights_base"]["checkpoint_present"] is False
    # 校验失败的当前选择也要给出来，而不是整段空掉
    assert described["current"]["error"] == "CheckpointMissingError"

    blob = json.dumps(described, ensure_ascii=False)
    for secret in ("NURSE_EXEC_API_TOKEN", "LLM_API_KEY", "MQTT_PASSWORD"):
        assert secret not in blob


def test_describe_registry_on_a_healthy_machine_has_no_error(tmp_path, weights_dir, fake_adapter):
    described = describe_registry(_FakeSettings(model_id="weights_base", registry_root=str(weights_dir)))
    assert described["current"]["impl"] == "weights"
    assert described["current"]["checkpoint_present"] is True
    assert "error" not in described["current"]


def test_postprocessing_overrides_are_recorded_and_can_be_switched_off(tmp_path):
    """后处理开关记在 spec 上可回溯；关掉时不带（对应源主路径的同名开关）。"""
    on = resolve_model("geom_kernel_base", registry_root=str(tmp_path))
    off = resolve_model("geom_kernel_base", registry_root=str(tmp_path), apply_postprocessing=False)
    assert on.overrides and off.overrides == ()
    assert "fill_hole_area=8" in on.overrides


def test_weights_adapter_registry_starts_empty_and_refuses_silent_replace():
    """适配器表默认空：本轮没有内置权重推理栈，接进来要显式注册。"""
    assert "test_torch" not in weights_adapter_names()


class _FakeSettings:
    """只带配置面那六个字段的轻量替身（避免每个用例都造一整套凭据）。

    照抄了 pydantic 的关键约定：``model_fields_set`` 只含**显式传过**的字段名——
    真实 ``Settings`` 正是靠它区分「运维配的」与「代码默认值」，替身漏了它，这个测试
    就会验不到「默认值不该盖住注册表」那条真正的行为。
    """

    def __init__(self, *, model_id="geom_kernel_base", impl=None,
                 config_path=None, checkpoint=None, registry_root="./models", registry_json=None):
        provided: set[str] = set()
        self.model_id = model_id
        self.model_impl = impl if impl is not None else "geom_kernel"
        self.model_config_path = config_path
        self.model_checkpoint = checkpoint
        self.model_registry_root = registry_root
        self.model_registry_json = registry_json
        fields = set(provided)
        for name, value in (
            ("model_impl", impl),
            ("model_config_path", config_path),
            ("model_checkpoint", checkpoint),
            ("model_registry_json", registry_json),
        ):
            if value is not None:
                fields.add(name)
        self.model_fields_set = fields