"""提示词模板注册表：模板从文件加载、版本化、渲染后留摘要。

**为什么提示词要落成文件**：模型输入是这套系统里**唯一**一段「改了会改变提示
内容」却不被任何测试盯着的文本。把它写死在业务代码里，改一个字要翻三个模块，
diff 里也看不出到底改了什么；审阅提示词的人还得先读 Python 才能找到它。落成
``templates/<name>.<version>.md`` 之后，改提示词 = 改一个文件（可 review、可
diff、可回滚），模板正文里的输出契约由 :mod:`gateway.schemas` 渲染进来，
**枚举不会与 Pydantic 模型漂移**。

四条纪律：

1. **模板名 + 版本进审计**。每次调用把 ``prompt_hash``（渲染后正文摘要）与
   ``prompt_version``（模板标识，形如 ``infusion_change@v1``）落进
   ``inference_audit``。护士长问「这条结论是哪版提示词给的」，库里查得到；
   「上周和这周的结论不一样」，先看版本有没有变，再看提示词摘要有没有变。
2. **占位符必须全部对上**。模板里写了 ``{{...}}`` 却没有对应上下文，注册表直接
   报错——半渲染的提示词会静悄悄地改变模型行为，而且改的是临床提示。
3. **业务代码不拼字符串**。网关按**模板名**取用（见 :data:`DEFAULT_TEMPLATE_NAME`），
   换模板只改注册表指向，不动业务层任何一个文件。
4. **正文摘要与渲染摘要分开**。:attr:`PromptTemplate.body_hash` 只认模板本身
   （改了正文就变），:attr:`RenderedPrompt.content_hash` 认「模板 + 本次上下文」
   （同输入重跑必须相同）。两个摘要用途不同，混用就查不出「结论变了是因为
   提示词变了，还是因为这次拍得不一样」。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import OUTPUT_SCHEMA

__all__ = [
    "DEFAULT_TEMPLATE_NAME",
    "TEMPLATE_DIR",
    "PromptRegistry",
    "PromptRegistryError",
    "PromptTemplate",
    "PromptTemplateMissing",
    "RenderedPrompt",
    "default_registry",
    "load_template",
    "render_output_contract",
    "render_text",
    "template_catalog_markdown",
    "template_hash",
]

#: 模板目录（随包分发）。模板正文是 ``.md`` 文件，便于在仓库里直接 review 与 diff。
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

#: 网关默认取用的模板名。**业务代码只认这个名字**，不认正文。
DEFAULT_TEMPLATE_NAME = "infusion_change"

#: 模板文件的后缀。
TEMPLATE_SUFFIX = ".md"


# ── 错误 ────────────────────────────────────────────────────────────────────


class PromptRegistryError(RuntimeError):
    """模板注册表层面的问题（渲染不了、缺上下文、元信息不全）。"""


class PromptTemplateMissing(PromptRegistryError):
    """按名字/版本找不到模板。"""


# ── 摘要 ────────────────────────────────────────────────────────────────────


def template_hash(text: str) -> str:
    """模板正文的稳定摘要（sha256 前 16 位）。

    键序与空白都影响摘要——这是有意的：提示词改一个空格就可能改结论，
    摘要变了才查得出「换了提示词之后结果不一样」。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── 输出契约（从 schema 渲染，模板不自己抄一份）────────────────────────────


def _short_hint(description: Any) -> str:
    """从 schema 的 description 里取一句短提示（截到第一个分句）。"""
    if not isinstance(description, str) or not description.strip():
        return ""
    text = description.strip()
    for sep in ("；", "。", "，", ",", ";", "."):
        if sep in text:
            head = text.split(sep)[0].strip()
            if head:
                return head
    return text


def _contract_value(spec: Mapping[str, Any]) -> str:
    """把一个字段的 schema 渲染成提示词里的一行取值说明。"""
    enum = spec.get("enum")
    if isinstance(enum, list) and enum:
        values = [v for v in enum if v is not None]
        joined = " | ".join(f'"{v}"' for v in values)
        if None in enum:
            return f"null | {joined}" if joined else "null"
        return joined
    stype = spec.get("type")
    types = stype if isinstance(stype, list) else [stype]
    if "number" in types or "integer" in types:
        low = spec.get("minimum")
        high = spec.get("maximum")
        if low is not None and high is not None:
            return f"{float(low)}-{float(high)}"
    if "string" in types:
        hint = _short_hint(spec.get("description"))
        return f'"{hint}"' if hint else '"字符串"'
    return "见下方约束"


def render_output_contract(schema: Mapping[str, Any] = OUTPUT_SCHEMA) -> str:
    """把输出契约渲染成提示词里的一行字段说明。

    **枚举与取值范围只有一个真源**：:data:`~gateway.schemas.OUTPUT_SCHEMA`。
    模板正文里不写死任何一个枚举值（写了就会与 Pydantic 模型漂移），只写
    ``{{output_contract}}`` 占位符，由本函数在渲染时填进去。

    字段顺序取 ``properties`` 的声明顺序（decision / reason / confidence /
    trigger_reason），四个字段**都**出现在契约行里——``trigger_reason`` 不在
    schema 的 ``required`` 里（``no_change`` 时允许为空），但漏掉它，模型在
    ``needs_change`` 时就没有可填的原因码了。
    """
    props = schema.get("properties") or {}
    parts: list[str] = []
    for field_name, spec in props.items():
        if not isinstance(spec, Mapping):
            continue
        parts.append(f'"{field_name}": {_contract_value(spec)}')
    return "{" + ", ".join(parts) + "}"


# ── 模板 ────────────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@dataclass(frozen=True)
class PromptTemplate:
    """一个提示词模板：名字、版本、正文、适用场景。

    ``source`` 记模板文件路径——审阅时能直接找到那个文件；``body_hash`` 是正文
    摘要，改一个字就变。
    """

    name: str
    version: str
    scenario: str
    body: str
    source: str = ""

    @property
    def template_id(self) -> str:
        """模板标识：``name@version``（审计里记的就是它）。"""
        return f"{self.name}@{self.version}"

    @property
    def body_hash(self) -> str:
        """模板正文摘要（与渲染上下文无关）。"""
        return template_hash(self.body)

    @property
    def placeholders(self) -> list[str]:
        """模板里用到的占位符名（去重、保序）。"""
        seen: list[str] = []
        for match in _PLACEHOLDER_RE.finditer(self.body):
            key = match.group(1)
            if key not in seen:
                seen.append(key)
        return seen

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "template_id": self.template_id,
            "scenario": self.scenario,
            "body_hash": self.body_hash,
            "source": self.source,
        }


@dataclass(frozen=True)
class RenderedPrompt:
    """渲染结果：正文 + 两类摘要 + 模板身份。"""

    template_id: str
    name: str
    version: str
    scenario: str
    text: str
    template_hash: str
    content_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "version": self.version,
            "scenario": self.scenario,
            "template_hash": self.template_hash,
            "content_hash": self.content_hash,
        }


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """拆出 ``---`` 头信息与正文。

    头信息是 ``key: value`` 行（不是 YAML——少一个依赖，格式也够用）：
    ``name`` / ``version`` / ``scenario`` 三项是必需的，缺任何一项都当模板损坏
    处理，不做「缺了就用默认值」的宽容处理——模板写错名字会静悄悄地取不到。
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise PromptRegistryError(f"模板头信息行缺少 ':'：{stripped!r}")
        key, _, value = stripped.partition(":")
        meta[key.strip()] = value.strip()
    return meta, text[match.end():]


def load_template(path: str | Path) -> PromptTemplate:
    """从文件读一个模板。"""
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")
    meta, body = _parse_front_matter(raw)
    missing = [key for key in ("name", "version", "scenario") if not meta.get(key)]
    if missing:
        raise PromptRegistryError(
            f"模板 {file_path} 头信息缺少：{'、'.join(missing)}"
            "（name / version / scenario 三项都必填）"
        )
    body = body.strip("\n")
    if not body.strip():
        raise PromptRegistryError(f"模板 {file_path} 正文为空")
    return PromptTemplate(
        name=meta["name"],
        version=meta["version"],
        scenario=meta["scenario"],
        body=body,
        source=str(file_path),
    )


def render_text(body: str, context: Mapping[str, Any]) -> str:
    """替换 ``{{name}}`` 占位符；**有占位符没有上下文就报错**。

    半渲染的提示词是最坏的一种错：它不报错，只是让模型收到一段缺了约束的
    文字，然后给出一个看起来合理但不该采信的结论。
    """
    missing: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            missing.append(key)
            return match.group(0)
        value = context[key]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    rendered = _PLACEHOLDER_RE.sub(_sub, body)
    if missing:
        raise PromptRegistryError(
            "模板占位符没有对应上下文："
            + "、".join(sorted(set(missing)))
            + "（补上上下文，或把占位符从模板里去掉）"
        )
    return rendered


def _version_sort_key(version: str) -> tuple[int, ...]:
    digits = re.findall(r"\d+", version)
    return tuple(int(d) for d in digits) if digits else (0,)


# ── 注册表 ──────────────────────────────────────────────────────────────────


class PromptRegistry:
    """模板注册表：按名字（可选版本）取用并渲染。

    同一个名字可以注册多个版本（``v1`` / ``v2``），**默认取版本号最大的那个**；
    指定版本时精确取那一份。注册同名同版本但正文不同的模板会被拒绝——那说明
    两处来源在打架，必须有人先说清楚哪份算数。
    """

    def __init__(self, templates: Iterable[PromptTemplate] = ()) -> None:
        self._templates: dict[str, dict[str, PromptTemplate]] = {}
        for template in templates:
            self.register(template)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "PromptRegistry":
        """从目录读入全部 ``*.md`` 模板（目录不存在就返回空注册表）。"""
        registry = cls()
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return registry
        for path in sorted(dir_path.glob(f"*{TEMPLATE_SUFFIX}")):
            registry.register(load_template(path))
        return registry

    def register(self, template: PromptTemplate) -> PromptTemplate:
        versions = self._templates.setdefault(template.name, {})
        existing = versions.get(template.version)
        if existing is not None and existing.body != template.body:
            raise PromptRegistryError(
                f"模板 {template.template_id} 重复注册且正文不一致"
                f"（{existing.source} vs {template.source}）"
            )
        versions[template.version] = template
        return template

    def names(self) -> list[str]:
        return sorted(self._templates)

    def versions(self, name: str) -> list[str]:
        versions = self._templates.get(name) or {}
        return sorted(versions, key=_version_sort_key)

    def get(self, name: str | None = None, version: str | None = None) -> PromptTemplate:
        """取模板；``version`` 省略时取该名字下版本号最大的那份。"""
        template_name = name or DEFAULT_TEMPLATE_NAME
        versions = self._templates.get(template_name)
        if not versions:
            raise PromptTemplateMissing(
                f"没有名为 {template_name!r} 的提示词模板；已注册："
                f"{'、'.join(self.names()) or '（空）'}"
            )
        if version is None:
            version = max(sorted(versions), key=_version_sort_key)
        template = versions.get(version)
        if template is None:
            raise PromptTemplateMissing(
                f"模板 {template_name!r} 没有版本 {version!r}；可用版本："
                f"{'、'.join(self.versions(template_name))}"
            )
        return template

    def default_context(self) -> dict[str, str]:
        """渲染时默认带上的上下文。

        目前只有 ``output_contract``——它从 schema 渲染，是模板与 Pydantic 模型
        不漂移的那道桥。
        """
        return {"output_contract": render_output_contract()}

    def render(
        self,
        name: str | None = None,
        *,
        version: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> RenderedPrompt:
        """渲染模板，返回正文与两类摘要。"""
        template = self.get(name, version)
        merged: dict[str, Any] = dict(self.default_context())
        merged.update(context or {})
        text = render_text(template.body, merged)
        return RenderedPrompt(
            template_id=template.template_id,
            name=template.name,
            version=template.version,
            scenario=template.scenario,
            text=text,
            template_hash=template.body_hash,
            content_hash=template_hash(text),
        )

    def templates(self) -> list[PromptTemplate]:
        """全部模板（按名字、版本排序）。"""
        out: list[PromptTemplate] = []
        for name in self.names():
            for version in self.versions(name):
                out.append(self._templates[name][version])
        return out

    def describe(self) -> list[dict[str, str]]:
        return [template.to_dict() for template in self.templates()]


_DEFAULT_REGISTRY: PromptRegistry | None = None


def default_registry() -> PromptRegistry:
    """默认注册表（读包内 ``templates/`` 目录，进程内只读一次）。"""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = PromptRegistry.from_directory(TEMPLATE_DIR)
    return _DEFAULT_REGISTRY


# ── 目录文档（供审阅）────────────────────────────────────────────────────────


def template_catalog_markdown(registry: PromptRegistry | None = None) -> str:
    """生成模板目录文档：谁在用、什么版本、什么时候用、占位符是什么。"""
    reg = registry or default_registry()
    lines = [
        "# 提示词模板注册表",
        "",
        "> 本文件由 `infusion_monitor.gateway.prompts.template_catalog_markdown()` 生成，",
        "> **请勿手改**。模板正文在 `src/infusion_monitor/gateway/templates/`，",
        "> 改提示词改那里，本文件随之重新生成。",
        "",
        "## 已注册模板",
        "",
        "| 模板 | 版本 | 适用场景 | 正文摘要 |",
        "|------|------|----------|----------|",
    ]
    templates = reg.templates()
    if not templates:
        lines.append("| （无） | — | — | — |")
    for template in templates:
        lines.append(
            f"| `{template.name}` | `{template.version}` | {template.scenario} | `{template.body_hash}` |"
        )
    lines += [
        "",
        "## 占位符",
        "",
        "| 模板 | 占位符 | 由谁提供 |",
        "|------|--------|----------|",
    ]
    for template in templates:
        for placeholder in template.placeholders or ["（无）"]:
            provider = "输出契约（由 `gateway.schemas.OUTPUT_SCHEMA` 渲染）" if placeholder == "output_contract" else "调用方传入"
            lines.append(f"| `{template.template_id}` | `{{{{{placeholder}}}}}` | {provider} |")
    lines += [
        "",
        "## 渲染后的输出契约",
        "",
        "```json",
        render_output_contract(),
        "```",
        "",
        "## 版本与审计",
        "",
        "* 每次调用落一行 `inference_audit`：`prompt_version` = 模板标识"
        "（形如 `infusion_change@v1`），`prompt_hash` = 渲染后正文的摘要；",
        "* `prompt_hash` 相同 ⇒ 模板版本与本次输入都相同 ⇒ 结论可复现；",
        "* `prompt_version` 变了而 `prompt_hash` 变了 ⇒ 提示词改过，",
        "  先比对模板 diff 再讨论结论差异；",
        "* 模板版本变了但 `prompt_hash` 没变 ⇒ 改动没有进入渲染路径（可疑）。",
        "",
    ]
    return "\n".join(lines)
