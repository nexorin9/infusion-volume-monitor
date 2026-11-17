"""Task 41 验收测试：提示词模板注册表、版本化与输出 schema 契约。

三组断言（对应 task.json 的三步）：

1. **模板可渲染**：模板真的从文件读进来（不是写死在 Python 里），占位符全部对上，
   缺上下文/缺模板时**报错并说清缺什么**，而不是渲染出一段半截提示词；
2. **hash 稳定**：同一份文件重复加载摘要不变；同一份上下文重复渲染摘要不变；
   模板正文变了摘要就变；``prompt_hash`` 与模板版本一起落进 ``inference_audit``，
   同一份观测重跑给同一个摘要（可复现）；
3. **schema 契约与模型一致**：提示词里那行输出契约的枚举、取值范围、字段集合
   与 :data:`gateway.schemas.OUTPUT_SCHEMA` / ``ModelDecision`` **逐项相等**——
   模板正文里不许写死任何一个枚举值。

另外一条**业务代码不拼字符串**的断言：换一个自定义注册表，网关发出去的提示词
就换了一整套，而网关、业务层、审计表的代码一个字都没改。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infusion_monitor.gateway.audit import GatewayAuditor
from infusion_monitor.gateway.model_gateway import (
    ModelGateway,
    build_messages,
    build_prompt,
)
from infusion_monitor.gateway.prompts import (
    DEFAULT_TEMPLATE_NAME,
    TEMPLATE_DIR,
    PromptRegistry,
    PromptRegistryError,
    PromptTemplateMissing,
    default_registry,
    load_template,
    render_output_contract,
    render_text,
    template_catalog_markdown,
)
from infusion_monitor.gateway.schemas import (
    CONFIDENCE_FLOOR,
    DECISIONS,
    OUTPUT_SCHEMA,
    OUTPUT_TRIGGER_REASONS,
    ModelDecision,
)
from infusion_monitor.store import InferenceAuditRepository

from .support.llm_stub import LLMStub

NO_CHANGE = {
    "decision": "no_change",
    "reason": "余量充足，本轮不用换",
    "confidence": 0.91,
    "trigger_reason": None,
}

TEMPLATE_HEADER = """---
name: {name}
version: {version}
scenario: {scenario}
---

"""


def _write_template(directory: Path, *, name: str, version: str, body: str,
                    scenario: str = "自定义模板（测试用）") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.{version}.md"
    path.write_text(
        TEMPLATE_HEADER.format(name=name, version=version, scenario=scenario) + body,
        encoding="utf-8",
    )
    return path


# ── 1. 模板可渲染 ───────────────────────────────────────────────────────────


def test_default_registry_loads_templates_from_disk():
    """模板从**文件**加载：注册表里至少有一份随包分发的模板，元信息齐全。"""
    registry = default_registry()
    templates = registry.templates()
    assert templates, "模板目录里没有可用的模板"
    template = registry.get(DEFAULT_TEMPLATE_NAME)
    assert template.name == DEFAULT_TEMPLATE_NAME
    assert template.version
    assert template.scenario.strip(), "模板必须写明适用场景，审的人才知道它在什么位置用"
    assert template.body.strip()
    assert Path(template.source).is_file(), "模板必须来自磁盘上的文件"
    assert Path(template.source).parent == TEMPLATE_DIR
    assert template.template_id == f"{template.name}@{template.version}"


def test_template_is_loaded_from_a_file_not_hardcoded(tmp_path):
    """换个目录放一份模板，注册表就能用它——正文不在 Python 代码里。"""
    body = "自定义提示词：{{output_contract}}\n只回答 JSON。"
    _write_template(tmp_path, name="custom", version="v1", body=body)
    registry = PromptRegistry.from_directory(tmp_path)
    rendered = registry.render("custom")
    assert rendered.text.startswith("自定义提示词：")
    assert "{{" not in rendered.text
    assert rendered.template_id == "custom@v1"


def test_render_substitutes_output_contract_placeholder():
    rendered = default_registry().render()
    assert "{{" not in rendered.text, "占位符必须全部替换干净"
    assert render_output_contract() in rendered.text, "输出契约要由 schema 渲染进模板"


def test_unknown_placeholder_is_rejected():
    """半渲染的提示词最坏：它不报错，只是让模型少收到一条约束。"""
    with pytest.raises(PromptRegistryError) as excinfo:
        render_text("请判断 {{output_contract}} 与 {{nope}}", {"output_contract": "{}"})
    assert "nope" in str(excinfo.value)


def test_missing_template_reports_available_names():
    registry = default_registry()
    with pytest.raises(PromptTemplateMissing) as excinfo:
        registry.get("no_such_template")
    message = str(excinfo.value)
    assert "no_such_template" in message
    assert DEFAULT_TEMPLATE_NAME in message, "报错要告诉调用方现在有哪些模板可用"


def test_missing_version_reports_available_versions(tmp_path):
    _write_template(tmp_path, name="multi", version="v1", body="正文 v1")
    _write_template(tmp_path, name="multi", version="v2", body="正文 v2")
    registry = PromptRegistry.from_directory(tmp_path)
    assert registry.versions("multi") == ["v1", "v2"]
    assert registry.get("multi").version == "v2", "默认取版本号最大的那份"
    with pytest.raises(PromptTemplateMissing) as excinfo:
        registry.get("multi", "v9")
    assert "v1" in str(excinfo.value) and "v2" in str(excinfo.value)


def test_template_without_required_metadata_is_rejected(tmp_path):
    path = tmp_path / "broken.md"
    path.write_text("---\nname: broken\n---\n正文\n", encoding="utf-8")
    with pytest.raises(PromptRegistryError) as excinfo:
        load_template(path)
    assert "version" in str(excinfo.value)


def test_duplicate_template_with_different_body_is_rejected(tmp_path):
    path = _write_template(tmp_path, name="dup", version="v1", body="正文一")
    template = load_template(path)
    registry = PromptRegistry([template])
    with pytest.raises(PromptRegistryError):
        registry.register(
            type(template)(name="dup", version="v1", scenario="x", body="正文二")
        )


# ── 2. hash 稳定 ────────────────────────────────────────────────────────────


def test_template_hash_is_stable_across_loads(tmp_path):
    path = _write_template(tmp_path, name="stable", version="v1", body="提示词正文\n第二行\n")
    first = load_template(path)
    second = load_template(path)
    assert first.body_hash == second.body_hash
    assert first.body_hash != load_template(
        _write_template(tmp_path, name="stable2", version="v1", body="改了正文\n")
    ).body_hash


def test_render_hash_depends_only_on_template_and_context(tmp_path):
    _write_template(tmp_path, name="ctx", version="v1", body="上下文：{{extra}}")
    registry = PromptRegistry.from_directory(tmp_path)
    first = registry.render("ctx", context={"extra": "甲"})
    second = registry.render("ctx", context={"extra": "甲"})
    third = registry.render("ctx", context={"extra": "乙"})
    assert first.content_hash == second.content_hash, "同一上下文必须渲染出同一个摘要"
    assert first.content_hash != third.content_hash, "上下文变了摘要必须变"
    assert first.template_hash == second.template_hash
    assert first.template_hash == third.template_hash, "模板摘要与上下文无关"


def test_prompt_hash_and_template_version_land_in_inference_audit(conn):
    """每次调用都要留下「哪版提示词 + 同一份输入的摘要」。"""
    with LLMStub() as stub:
        stub.script_json(NO_CHANGE)
        gateway = ModelGateway(
            base_url=stub.base_url,
            api_key="test-key-not-real",
            model="test-model",
            timeout=1.0,
            connect_timeout=1.0,
            backoff_base=0.0,
            auditor=GatewayAuditor(InferenceAuditRepository(conn)),
        )
        observation = {"bag_seq": "B01", "bed_no": "EXAMPLE-BED-12", "fill_ratio": 0.42}
        try:
            advice = gateway.advise(observation, deterministic=None, request_id="r-prompts")
        finally:
            gateway.close()

    rows = InferenceAuditRepository(conn).list_by_request("r-prompts")
    assert len(rows) == 1
    row = rows[0]
    assert row["prompt_version"] == f"{DEFAULT_TEMPLATE_NAME}@v1", "审计要记到模板版本"
    assert row["prompt_hash"] == advice.prompt_hash and row["prompt_hash"]
    assert advice.prompt_version == row["prompt_version"]


def test_same_observation_reproduces_the_same_prompt_hash(conn):
    """同输入可复现：同一份观测渲染两次，``prompt_hash`` 必须逐字相同。"""
    with LLMStub() as stub:
        stub.script_json(NO_CHANGE)
        stub.script_json(NO_CHANGE)
        gateway = ModelGateway(
            base_url=stub.base_url,
            api_key="test-key-not-real",
            model="test-model",
            timeout=1.0,
            connect_timeout=1.0,
            backoff_base=0.0,
            auditor=GatewayAuditor(InferenceAuditRepository(conn)),
        )
        observation = {"bag_seq": "B01", "bed_no": "EXAMPLE-BED-12", "fill_ratio": 0.42}
        try:
            first = gateway.advise(observation, deterministic=None, request_id="r-rep-1")
            second = gateway.advise(observation, deterministic=None, request_id="r-rep-2")
        finally:
            gateway.close()

    assert first.prompt_hash == second.prompt_hash
    assert first.prompt_version == second.prompt_version


def test_changing_the_template_changes_the_recorded_prompt_hash(conn, tmp_path):
    """换模板 ⇒ 审计里的摘要与版本都变（不然「结论变了」根本查不出原因）。"""
    _write_template(tmp_path, name="other", version="v1", body="另一套提示词\n{{output_contract}}")
    other_registry = PromptRegistry.from_directory(tmp_path)
    observation = {"bag_seq": "B01", "bed_no": "EXAMPLE-BED-12", "fill_ratio": 0.42}

    baseline = build_prompt(observation, registry=default_registry())
    other = build_prompt(observation, registry=other_registry, template_name="other")
    assert baseline.content_hash != other.content_hash
    assert other.template_id == "other@v1"


# ── 3. schema 契约与模型一致 ────────────────────────────────────────────────


def test_contract_line_lists_exactly_the_schema_enums():
    contract = render_output_contract()
    for decision in DECISIONS:
        assert f'"{decision}"' in contract, f"契约行里必须有 decision 取值 {decision}"
    for reason in OUTPUT_TRIGGER_REASONS:
        assert f'"{reason}"' in contract, f"契约行里必须有 trigger_reason 取值 {reason}"
    assert '"replaced"' not in contract, "replaced 不是模型能给的结论，绝不能出现在取值里"


def test_contract_fields_match_model_and_schema():
    contract = render_output_contract()
    fields_in_contract = {
        part.split(":")[0].strip().strip('"')
        for part in contract.strip("{}").split(", ")
    }
    assert fields_in_contract == set(ModelDecision.model_fields)
    assert fields_in_contract == set(OUTPUT_SCHEMA["properties"])


def test_contract_confidence_range_matches_schema():
    contract = render_output_contract()
    spec = OUTPUT_SCHEMA["properties"]["confidence"]
    assert f'{float(spec["minimum"])}-{float(spec["maximum"])}' in contract
    assert spec["minimum"] <= CONFIDENCE_FLOOR <= spec["maximum"], (
        "置信门禁必须落在契约声明的取值范围内，否则门禁与提示词互相矛盾"
    )


def test_schema_enums_agree_with_pydantic_model():
    assert tuple(OUTPUT_SCHEMA["properties"]["decision"]["enum"]) == DECISIONS
    schema_reasons = tuple(
        r for r in OUTPUT_SCHEMA["properties"]["trigger_reason"]["enum"] if r is not None
    )
    assert schema_reasons == OUTPUT_TRIGGER_REASONS
    # 模型给的结论里不允许出现 replaced
    with pytest.raises(Exception):
        ModelDecision(
            decision="replaced", reason="想直接确认换液", confidence=0.9
        )


def test_template_body_does_not_hardcode_enum_values():
    """模板正文只留占位符：枚举值全部由 schema 渲染，不会与模型漂移。"""
    template = default_registry().get(DEFAULT_TEMPLATE_NAME)
    for value in (*DECISIONS, *OUTPUT_TRIGGER_REASONS):
        assert f'"{value}"' not in template.body, (
            f"模板正文里写死了枚举值 {value}，它应该来自 schema 渲染"
        )
    assert "{{output_contract}}" in template.body


def test_contract_markdown_and_catalog_stay_in_sync():
    """目录文档由注册表生成：改了模板要重新生成，否则审阅的人看的是旧的。"""
    doc_path = Path(__file__).resolve().parents[1] / "docs" / "prompt-templates.md"
    assert doc_path.exists(), "docs/prompt-templates.md 不存在；用 template_catalog_markdown() 生成"
    assert doc_path.read_text(encoding="utf-8") == template_catalog_markdown(), (
        "docs/prompt-templates.md 与模板注册表已漂移，请重新生成该文档"
    )


# ── 业务代码不拼字符串 ──────────────────────────────────────────────────────


def test_swapping_the_registry_swaps_the_prompt_without_touching_business_code(tmp_path):
    """换一套提示词 = 换一个注册表；业务层与网关代码一个字都不用改。"""
    _write_template(
        tmp_path,
        name="night_shift",
        version="v3",
        body="夜班专用提示词：{{output_contract}}\n只回答 JSON。",
    )
    registry = PromptRegistry.from_directory(tmp_path)
    observation = {"bag_seq": "B01", "bed_no": "EXAMPLE-BED-12", "fill_ratio": 0.42}

    default_messages = build_messages(observation)
    night_messages = build_messages(
        observation, registry=registry, template_name="night_shift"
    )

    assert "夜班专用提示词" in night_messages[0]["content"]
    assert "夜班专用提示词" not in default_messages[0]["content"]
    # 业务代码没变：两条消息的形状与观测载荷完全一致
    assert [m["role"] for m in night_messages] == [m["role"] for m in default_messages]
    assert json.loads(night_messages[1]["content"]) == json.loads(default_messages[1]["content"])


def test_gateway_sends_the_registry_prompt_to_the_endpoint(tmp_path):
    """真发一次：替身收到的 system 消息就是注册表渲染出来的那份。"""
    _write_template(
        tmp_path,
        name="probe",
        version="v1",
        body="探针提示词 {{output_contract}}",
    )
    registry = PromptRegistry.from_directory(tmp_path)
    with LLMStub() as stub:
        stub.script_json(NO_CHANGE)
        gateway = ModelGateway(
            base_url=stub.base_url,
            api_key="test-key-not-real",
            model="test-model",
            timeout=1.0,
            connect_timeout=1.0,
            backoff_base=0.0,
            prompts=registry,
        )
        try:
            gateway.advise(
                {"bag_seq": "B01"},
                deterministic=None,
                request_id="r-probe",
                template_name="probe",
            )
        finally:
            gateway.close()

    system = stub.requests[0].payload["messages"][0]["content"]
    assert "探针提示词" in system
    assert render_output_contract() in system
