"""出站写回客户端：把护士确认过的换液事件 POST 给护理执行平台。

**这是正式实现，不是测试替身**：真实协议（HTTP/1.1 + JSON）、真实端点
``POST {NURSE_EXEC_BASE_URL}/api/v1/infusion/events``、真实凭据位
``Authorization: Bearer $NURSE_EXEC_API_TOKEN``。凭据只从环境变量读，
不落配置、不进日志、不进 ``last_error``（见 :func:`scrub`）。

四条纪律：

1. **字段规范与代码同源**。:data:`OUTBOUND_EVENT_SCHEMA` 是出站字段的唯一真源：
   :class:`OutboundEvent` 按它做必填/类型校验，``docs/nurse-exec-contract.md``
   由它生成（:func:`contract_markdown`）。三者不可能各写一份然后慢慢漂移——
   契约测试直接断言「文档表格 == 模型字段 == 实际请求体」。
2. **幂等键随请求透传**。``client_event_id`` 同时出现在请求体的字段里和
   ``Idempotency-Key`` 请求头里，两者必须同值：对端凭它去重，我们这边的重试
   与对端的重复消费靠的是同一个键，而不是两套约定。
3. **门禁语义一路带到线上**。``estimated_volume_ml`` / ``eta_minutes`` 为 ``None``
   时（未标定袋型、滴速不可得）在请求体里**整体不出现**，不是 ``null`` 占位——
   对端若把 null 当 0 处理，就会把「我们没测出来」读成「袋是空的」。
4. **失败要能说清是哪一类**。4xx 是对端业务拒绝（重试只会再被拒一次），5xx 与
   连接/读取超时才值得退避重试；「半开连接」单列一类，因为请求多半已经到对端，
   只有幂等键能保证重发安全。全部口径写在 :data:`FAILURE_CLASSES` 一张表里，
   本模块只负责**分类**，退避与终态由 :mod:`adapters.writeback` 执行。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import httpx
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DECISION_SOURCES_OUTBOUND",
    "EVENT_TYPES",
    "FAILURE_CLASSES",
    "OUTBOUND_EVENT_SCHEMA",
    "PEER_FIELD_BINDINGS",
    "RETRYABLE_FAILURE_KINDS",
    "TRIGGER_REASONS_OUTBOUND",
    "FailureClass",
    "FieldSpec",
    "NurseExecClient",
    "OutboundEvent",
    "PeerFieldBinding",
    "WritebackAttempt",
    "business_rejection_reason",
    "build_outbound_body",
    "classify_exception",
    "classify_response",
    "contract_markdown",
    "failure_class",
    "failure_summary",
    "is_retryable",
    "scrub",
]

#: 出站事件类型。目前只有换液一种；将来加「加液」「暂停」时在此扩充，
#: 对端按 ``event_type`` 分派，不靠猜字段。
EVENT_TYPES = ("infusion.bag.changed",)

#: 出站载荷里 ``trigger_reason`` 的取值 = 判定通过时的三条原因码 + 护士确认。
TRIGGER_REASONS_OUTBOUND = (
    "eta_within_window",
    "volume_below_margin",
    "near_empty_observed",
    "nurse_confirmed",
)

#: 出站载荷里 ``decision_source`` 的取值。前三个来自模型网关（结论由谁下的），
#: 第四个是出站独有的：事件走到写回这一步，结论已被护士确认取代。
#: 契约测试断言本表**覆盖**网关侧的 :data:`~infusion_monitor.gateway.audit.DECISION_SOURCES`，
#: 网关新增取值而这里忘了补，会当场报错而不是悄悄发一个对端不认识的码。
DECISION_SOURCES_OUTBOUND = (
    "model_decided",
    "threshold_fallback",
    "threshold_only",
    "nurse_confirmed",
)


@dataclass(frozen=True)
class FieldSpec:
    """出站字段规范的一条：名字、类型、必填、可空、枚举、含义。

    可空（nullable）与缺省（required=False）是两件事：``fill_ratio`` 是**可空**
    的数字（测不出来就是没有），``operator_id`` 是**必填**的（没有确认人就不
    该有这条事件）。字段缺失与显式 null 在对端是两种语义，这里逐字段写死。

    ``enum`` 非空时，该字段的取值必须落在枚举里——出站侧不写「大概就是这几个」，
    对端要按码分派，枚举之外的码会被当成未知事件。
    """

    name: str
    type: str
    required: bool
    nullable: bool
    description: str
    enum: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "nullable": self.nullable,
            "description": self.description,
        }
        if self.enum:
            data["enum"] = list(self.enum)
        return data


#: 出站字段规范（``docs/nurse-exec-contract.md`` 由此生成）。
OUTBOUND_EVENT_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("event_id", "str", True, False, "本服务生成的事件 ID，写回后供对端回查与对账"),
    FieldSpec("client_event_id", "str", True, False, "幂等键：同一袋只换一次，重放必得同一个值"),
    FieldSpec("event_type", "enum", True, False, "事件类型，当前固定 infusion.bag.changed",
              EVENT_TYPES),
    FieldSpec("session_id", "str", False, True, "本轮巡视的袋级会话 ID，便于回查曲线"),
    FieldSpec("rack_id", "str", True, False, "输液架编号"),
    FieldSpec("bed_no", "str", True, False, "床号"),
    FieldSpec("infusion_order_no", "str", True, False, "输注单号（对端给的权威标识）"),
    FieldSpec("bag_seq", "str", True, False, "被换掉的袋序"),
    FieldSpec("fill_ratio", "float", False, True, "触发确认时的面积比（0–1）；测不出则不出现"),
    FieldSpec("estimated_volume_ml", "float", False, True, "标定后的剩余毫升；未标定袋型则不出现"),
    FieldSpec("eta_minutes", "int", False, True, "预计耗尽分钟数；滴速不可得则不出现"),
    FieldSpec("trigger_reason", "str", False, True, "点名/确认原因码", TRIGGER_REASONS_OUTBOUND),
    FieldSpec("decision_source", "str", False, True, "结论来源", DECISION_SOURCES_OUTBOUND),
    FieldSpec("operator_id", "str", True, False, "确认换液的护士工号；没有确认人就没有这条事件"),
    FieldSpec("occurred_at", "str", True, False, "事件发生时刻（ISO8601）"),
)

_SCHEMA_BY_NAME = {spec.name: spec for spec in OUTBOUND_EVENT_SCHEMA}


@dataclass(frozen=True)
class PeerFieldBinding:
    """一条出站字段与**对端接口文档**的对应关系。

    字段规范回答「我们发什么」，对应关系回答「对端在它自己的接口文档里管这个叫
    什么、放哪儿、按什么语义处理」。两件事必须一起落盘：接挂的时候，信息科拿着
    对端的文档逐行对，才能确认两边说的是同一件事，而不是「名字长得像」。
    """

    #: 在对端接口文档里的落点：请求体字段、请求头，还是两处都有。
    location: str
    #: 对端侧的约定（为什么这么放、缺省时对端怎么处理）。
    note: str


#: 字段 → 对端接口文档对应表。键与 :data:`OUTBOUND_EVENT_SCHEMA` 一一对应
#: （契约测试逐字段断言不多不少），文档里生成成一张表。
PEER_FIELD_BINDINGS: dict[str, PeerFieldBinding] = {
    "event_id": PeerFieldBinding(
        "请求体 eventId", "本服务生成的回查号；对端原样落库，不参与去重"
    ),
    "client_event_id": PeerFieldBinding(
        "请求体 clientEventId + 请求头 Idempotency-Key",
        "**幂等键**：两处必须同值。对端凭它去重，因此重试安全（同一袋只换一次）",
    ),
    "event_type": PeerFieldBinding(
        "请求体 eventType", "对端按码分派到各自的处理流程；未知码会被拒"
    ),
    "session_id": PeerFieldBinding(
        "请求体 sessionId", "本服务的袋级会话 ID；对端只做回查用，不做业务键"
    ),
    "rack_id": PeerFieldBinding("请求体 rackId", "输液架编号，与床号一起定位巡视频次"),
    "bed_no": PeerFieldBinding("请求体 bedNo", "床号；不传患者姓名与住院号"),
    "infusion_order_no": PeerFieldBinding(
        "请求体 infusionOrderNo", "输注单号；对端校验它是否属于该床位的在输医嘱"
    ),
    "bag_seq": PeerFieldBinding("请求体 bagSeq", "被换掉的袋序；新袋由对端按自身规则编号"),
    "fill_ratio": PeerFieldBinding(
        "请求体 fillRatio（可选）", "缺省即「本轮没测出来」，对端**不得**按 0 处理"
    ),
    "estimated_volume_ml": PeerFieldBinding(
        "请求体 estimatedVolumeMl（可选）", "未标定袋型则整个字段不出现，不是 null"
    ),
    "eta_minutes": PeerFieldBinding(
        "请求体 etaMinutes（可选）", "滴速不可得则不出现；对端按缺省走巡视节奏"
    ),
    "trigger_reason": PeerFieldBinding(
        "请求体 triggerReason（可选）", "原因码；对端写进执行记录，飞检时对得上"
    ),
    "decision_source": PeerFieldBinding(
        "请求体 decisionSource（可选）", "结论来源；对端不据此改流程，只做留痕"
    ),
    "operator_id": PeerFieldBinding(
        "请求体 operatorId", "护士工号；对端据此把执行记录落到具体的人头上"
    ),
    "occurred_at": PeerFieldBinding(
        "请求体 occurredAt", "业务发生时刻（护士点确认那一刻），不是对端的接收时刻"
    ),
}


class OutboundEvent(BaseModel):
    """出站事件载荷。字段与 :data:`OUTBOUND_EVENT_SCHEMA` 一一对应。"""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    client_event_id: str
    event_type: str = "infusion.bag.changed"
    session_id: str | None = None
    rack_id: str
    bed_no: str
    infusion_order_no: str
    bag_seq: str
    fill_ratio: float | None = None
    estimated_volume_ml: float | None = None
    eta_minutes: int | None = None
    trigger_reason: str | None = None
    decision_source: str | None = None
    operator_id: str
    occurred_at: str

    def to_body(self) -> dict[str, Any]:
        """出站请求体：**None 字段整体不出现**（不是 null 占位）。"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_event_row(cls, row: Mapping[str, Any]) -> "OutboundEvent":
        """从 ``infusion_event`` 行装配。

        取行里已落库的 ``payload_json`` 快照优先，缺失时回退到列值——这样字段
        规范变更后重放旧事件，仍按事件**当时**的形态发出去，而不是用新代码
        重新解释一遍旧数据（对账时两者对不上比发不出去更麻烦）。
        """
        payload: Mapping[str, Any] = row.get("payload") or {}
        data = {spec.name: payload.get(spec.name, row.get(spec.name)) for spec in OUTBOUND_EVENT_SCHEMA}
        data["event_id"] = data.get("event_id") or row.get("event_id")
        data["event_type"] = data.get("event_type") or row.get("event_type") or "infusion.bag.changed"
        return cls(**data)


def build_outbound_body(row: Mapping[str, Any]) -> OutboundEvent:
    """装配出站事件（对外暴露的语义名，内部仍走 :meth:`OutboundEvent.from_event_row`）。"""
    return OutboundEvent.from_event_row(row)


def scrub(text: str, *secrets: str | None) -> str:
    """从任意文本里抹掉凭据值。

    出站失败时我们会把对端的状态码与响应片段写进 ``last_error``，方便信息科
    排查；但如果对端把请求头原样回显（有的网关会），Bearer token 就会进库、
    进日志、进飞检材料。抹一遍比事后解释便宜。
    """
    out = text
    for secret in secrets:
        if secret:
            out = out.replace(secret, "***")
    return out


@dataclass(frozen=True)
class WritebackAttempt:
    """一次出站尝试的结果（纯网络层，不碰数据库）。"""

    ok: bool
    kind: str
    status_code: int | None = None
    error: str | None = None
    elapsed_ms: int = 0
    #: 对端响应体片段（已脱敏），成功与失败都留一小段便于排查。
    response_excerpt: str | None = None
    idempotency_key: str | None = None
    #: 4xx 时从响应体抽出的对端原因（已脱敏）；业务拒绝才认它。
    rejection_reason: str | None = None
    #: 该类别是否可重试（由 :data:`FAILURE_CLASSES` 派生，不另存一份判断）。
    retryable: bool = False
    #: 对端可能已经收下并处理了这条事件——幂等键存在的理由。
    peer_may_have_applied: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "status_code": self.status_code,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "retryable": self.retryable,
            "rejection_reason": self.rejection_reason,
            "peer_may_have_applied": self.peer_may_have_applied,
        }


@dataclass(frozen=True)
class FailureClass:
    """一类失败的处置口径：能不能重试、落到什么终态、对端可能有没有收到。

    把这些口径写进代码而不是散在退避逻辑里，是因为退避次数、终态与
    ``last_error`` 措辞必须是**同一张表**的三个投影：以前它们分散在
    :mod:`adapters.writeback` 的 if 分支里，加一种失败就得改三处，漏一处就
    变成「不可重试的失败被退避重试三次」或「重试耗尽却记成 failed」。
    """

    kind: str
    #: 是否值得退避重试。False = 重发只会得到同一个结果。
    retryable: bool
    #: 重试耗尽后（或压根不重试）事件行落到哪个状态。
    terminal_status: str
    #: 对端**可能已经**收下并处理了这条事件，只是应答没回来。
    #: 这一列就是幂等键存在的理由：只有这类失败才既「可能重发」又「不能重发两遍」。
    peer_may_have_applied: bool
    #: 中文说明：这类失败在病区现场长什么样。
    description: str
    #: 信息科看到 ``last_error`` 里这个类别时，下一步该做什么。
    operator_hint: str


#: 失败分类总表。**唯一真源**：退避名单（:data:`RETRYABLE_FAILURE_KINDS`）、
#: 错误摘要、契约文档的失败分类表都从这里派生。
FAILURE_CLASSES: dict[str, FailureClass] = {
    "ok": FailureClass(
        "ok", False, "sent", False,
        "对端 2xx 收下了这条事件",
        "无需处理；信息科可在对端按 event_id 回查",
    ),
    "http_4xx": FailureClass(
        "http_4xx", False, "failed", False,
        "对端业务拒绝：同一份载荷、同一把 token 重发仍会被拒",
        "看 last_error 里的 reason（对端自己给的原因）；改载荷或找护理执行平台值班",
    ),
    "http_5xx": FailureClass(
        "http_5xx", True, "pending_writeback", False,
        "对端内部错误，与本次载荷无关",
        "已退避重试；耗尽后由信息科在待写回列表重放",
    ),
    "http_unexpected": FailureClass(
        "http_unexpected", False, "failed", False,
        "看不懂的状态码（1xx 落成最终响应 / 3xx，本服务不跟随重定向）",
        "查对端网关是否改了状态码约定；不要盲目重发",
    ),
    "connect_timeout": FailureClass(
        "connect_timeout", True, "pending_writeback", False,
        "TCP 建连就没连上：病区到对端的网络不通或防火墙拦了",
        "查病区网段到对端的连通性与端口放通；请求没有离开本机，对端没有收到",
    ),
    "read_timeout": FailureClass(
        "read_timeout", True, "pending_writeback", True,
        "对端收下了请求但没在时限内回应答：它慢，或者它卡在自己的库里",
        "重试安全由幂等键保证；对端凭 Idempotency-Key 去重",
    ),
    "write_timeout": FailureClass(
        "write_timeout", True, "pending_writeback", True,
        "载荷发到一半断了：请求体可能只写到一部分，对端多半已按格式错误拒收",
        "查病区链路 MTU 与代理；重试安全由幂等键保证",
    ),
    "pool_timeout": FailureClass(
        "pool_timeout", True, "pending_writeback", False,
        "连接池等不到可用连接：本服务自己占着连接，尚未发到对端",
        "查本机并发写回数；不是对端的问题",
    ),
    "half_open": FailureClass(
        "half_open", True, "pending_writeback", True,
        "半开连接：对端把连接掐了却没给应答（复用的连接早已失效、对端进程重启、"
        "中间的负载均衡重置了空闲连接）",
        "请求**很可能已经到对端**；重试安全完全依赖幂等键，重放前先确认对端去重生效",
    ),
    "connect_error": FailureClass(
        "connect_error", True, "pending_writeback", False,
        "连接被拒 / DNS 解析失败：地址写错、对端没起、或被拒在门外",
        "核对 NURSE_EXEC_BASE_URL 与 DNS；请求没有到达对端",
    ),
    "protocol_error": FailureClass(
        "protocol_error", False, "failed", False,
        "协议层就不成立（最常见：base_url 少了 http:// 前缀）",
        "改 NURSE_EXEC_BASE_URL；这是配置问题，重试多少次都一样",
    ),
}

#: 值得退避重试的失败类别（由 :data:`FAILURE_CLASSES` 派生，不另抄一份）。
RETRYABLE_FAILURE_KINDS: tuple[str, ...] = tuple(
    kind for kind, spec in FAILURE_CLASSES.items() if spec.retryable
)


def failure_class(kind: str) -> FailureClass:
    """查失败类别；未知类别按「不可重试的看不懂」处理（不盲目重发）。"""
    return FAILURE_CLASSES.get(kind, FAILURE_CLASSES["http_unexpected"])


def is_retryable(kind: str) -> bool:
    """该类别是否值得退避重试。"""
    return failure_class(kind).retryable


def _excerpt(text: str | None, limit: int = 200) -> str:
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat[:limit]


def _reason_from_payload(payload: Any, *, limit: int = 3) -> str | None:
    """从对端响应体的 JSON 里抽出人话原因。

    对端可能回 ``{"detail": "unknown infusion_order_no"}``、也可能是框架默认的
    ``{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}``，还有的网关
    直接把原因放在 ``message``。挨个试一遍都试不到，才退回原始文本——宁可给一段
    难读的原文，也不要在 ``last_error`` 里只留一个「HTTP 422」。
    """
    if isinstance(payload, str):
        return _excerpt(payload) or None
    if isinstance(payload, list):
        parts = [got for got in (_reason_from_payload(item, limit=limit) for item in payload) if got]
        return "; ".join(parts[:limit]) or None
    if isinstance(payload, dict):
        for key in ("detail", "message", "msg", "error", "reason", "code"):
            if key in payload:
                got = _reason_from_payload(payload.get(key), limit=limit)
                if got:
                    loc = payload.get("loc")
                    if isinstance(loc, list) and loc:
                        return f"{got}（{'/'.join(str(part) for part in loc)}）"
                    return got
    return None


def business_rejection_reason(response: httpx.Response) -> str | None:
    """4xx 时从响应体抽出**对端给的原因**（已脱敏）。

    信息科看到「对端拒了」没有用——他要能直接拿着这句话去问对端为什么。
    """
    text = _excerpt(response.text if response.content else "")
    if not text:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return text
    return _reason_from_payload(payload) or text


def classify_response(response: httpx.Response) -> tuple[bool, str]:
    """把 HTTP 响应分类为 (是否成功, 失败类别)。

    类别决定**要不要退避重试**（口径见 :data:`FAILURE_CLASSES`）：

    * ``http_4xx`` —— 对端的业务拒绝。重试只会再被拒一次（同一份载荷、同一把
      token），只会把「对端不收这个事件」拖成一串重复拒绝。不可重试。
    * ``http_5xx`` —— 对端内部错误。与本服务的载荷无关，重试有意义。可重试。
    * ``http_2xx`` / ``http_3xx`` 之外的怪状态码（如 1xx 落成最终响应）按不可
      重试处理：看不懂的响应不该被盲目重发。
    """
    code = response.status_code
    if 200 <= code < 300:
        return True, "ok"
    if 400 <= code < 500:
        return False, "http_4xx"
    if 500 <= code < 600:
        return False, "http_5xx"
    return False, "http_unexpected"


#: 传输层异常 → 失败类别。**建连阶段**与**应答阶段**分开，因为处置完全不同：
#: 建连超时说明请求根本没离开本机；读超时说明对端很可能已经收下了。
_TIMEOUT_KIND_BY_EXC: tuple[tuple[type[BaseException], str], ...] = (
    (httpx.ConnectTimeout, "connect_timeout"),
    (httpx.ReadTimeout, "read_timeout"),
    (httpx.WriteTimeout, "write_timeout"),
    (httpx.PoolTimeout, "pool_timeout"),
)

#: 半开连接的表现：请求发出去了，对端（或中间的网关）把连接掐了却没给应答。
#: 复用一条早已失效的 keep-alive 连接时最常见——httpx 侧表现为
#: ``RemoteProtocolError: Server disconnected without sending a response``。
_HALF_OPEN_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.CloseError,
)


def classify_exception(exc: BaseException) -> str:
    """把传输层异常分类成失败类别。**半开连接单独成一类**。

    以前这些异常统统落进 ``connect_error``，于是「连接压根没建起来」和
    「连上了但对端把话说到一半掐了」在 ``last_error`` 里长得一模一样。信息科
    照着这句话去查网络方向就查错了：半开连接的病根通常在**对端或中间网关**
    （连接池回收、对端进程重启、LB 重置空闲连接），本机网络是好的；而它的
    危险之处在于**请求多半已经送到对端**，必须靠幂等键才敢重发。
    """
    for exc_type, kind in _TIMEOUT_KIND_BY_EXC:
        if isinstance(exc, exc_type):
            return kind
    if isinstance(exc, httpx.TimeoutException):
        return "read_timeout"  # 未来新增的超时子类：按最常见的一种归
    if isinstance(exc, _HALF_OPEN_EXCEPTIONS):
        return "half_open"
    if isinstance(exc, httpx.UnsupportedProtocol):
        return "protocol_error"
    return "connect_error"


def failure_summary(
    kind: str,
    *,
    status_code: int | None = None,
    attempts: int = 1,
    counts: Mapping[str, int] | None = None,
    reason: str | None = None,
    last_error: str | None = None,
    secrets: Sequence[str | None] = (),
) -> str:
    """拼一条**可复核且不含凭据**的错误摘要。

    退避耗尽后这条文本会进 ``last_error``、进待写回列表、进飞检材料，所以：

    * **状态码、尝试次数、各类别次数、最后错误文本**四件套齐——只写「发送失败」
      等于让信息科自己再去翻日志；
    * 逐类别计数（连接超时几次、读超时几次）能区分「网络不通」和「对端慢」；
    * 全部经 :func:`scrub` 抹掉 token 值。
    """
    parts = [f"kind={kind}"]
    if status_code is not None:
        parts.append(f"status={status_code}")
    parts.append(f"attempts={attempts}")
    if counts:
        parts.append("counts=" + ",".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if reason:
        parts.append(f"reason={_excerpt(reason, 160)}")
    if last_error:
        parts.append(f"last={_excerpt(last_error, 200)}")
    return scrub(" ".join(parts), *secrets)


@dataclass
class NurseExecClient:
    """护理执行平台出站客户端。

    ``transport`` 供契约测试注入本地替身；不传时用真实 HTTP。**产品代码不因为
    测试注入而改变行为**——注入的只是传输层，报头与请求体照常组装。
    """

    base_url: str
    api_token: str
    timeout: float = 5.0
    connect_timeout: float = 3.0
    client: httpx.Client | None = None
    _owns_client: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: Any, **kwargs: Any) -> "NurseExecClient":
        """从 :class:`~infusion_monitor.config.Settings` 构造（凭据只走配置面）。"""
        return cls(
            base_url=settings.nurse_exec_events_url,
            api_token=settings.nurse_exec_api_token.get_secret_value(),
            timeout=getattr(settings, "writeback_timeout", 5.0),
            **kwargs,
        )

    @property
    def endpoint(self) -> str:
        return self.base_url

    def _http(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client(
                timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout),
                headers={"Accept": "application/json"},
                follow_redirects=False,
                # 不继承宿主机/工作站的系统代理。出站端点由 NURSE_EXEC_BASE_URL
                # 明确给出，护理执行平台就在院内网段里；让一个工作台上的系统代理
                # 悄悄接管换液事件有两条害处：病区事件被绕出院内网络，以及
                # 失败分类失真——代理会把「对端掐了连接不给应答」统一翻成 502，
                # 于是半开连接、读超时全都被记成对端 5xx，信息科照着错误方向排查。
                trust_env=False,
            )
            self._owns_client = True
        return self.client

    def headers_for(self, event: OutboundEvent) -> dict[str, str]:
        """出站请求头。

        ``Idempotency-Key`` 与请求体里的 ``client_event_id`` 同值：对端按请求头
        去重（网关/中间件的通用位置），按字段去重（业务侧的最终依据），两者
        必须是同一个键——不一致就会出现「网关放过去了、业务侧又收了两条」。
        """
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Idempotency-Key": event.client_event_id,
        }

    def attempt(self, event: OutboundEvent, *, transport_only: bool = True) -> WritebackAttempt:
        """发一次出站请求，返回分类结果。**不重试**（退避在 writeback 层）。

        只做网络与协议，不碰数据库，因此可以在工作线程里跑（见
        :func:`adapters.writeback.deliver_event_async`）：SQLite 连接带线程
        亲和性，阻塞的 HTTP 调用不能待在事件循环线程上，也不能让落库回到另一条
        线程。
        """
        body = event.to_body()
        headers = self.headers_for(event)
        started = time.perf_counter()
        try:
            response = self._http().post(
                self.endpoint, json=body, headers=headers, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            kind = classify_exception(exc)
            return WritebackAttempt(
                ok=False,
                kind=kind,
                error=scrub(f"{kind}: {type(exc).__name__}: {exc}", self.api_token),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                idempotency_key=event.client_event_id,
                retryable=failure_class(kind).retryable,
                peer_may_have_applied=failure_class(kind).peer_may_have_applied,
            )

        ok, kind = classify_response(response)
        spec = failure_class(kind)
        excerpt = scrub(_excerpt(response.text), self.api_token)
        reason = None
        if not ok and kind == "http_4xx":
            reason = scrub(business_rejection_reason(response) or "", self.api_token) or None
        detail = f"HTTP {response.status_code} {excerpt}".strip()
        return WritebackAttempt(
            ok=ok,
            kind=kind,
            status_code=response.status_code,
            error=None if ok else scrub(f"{kind}: {detail}", self.api_token),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            response_excerpt=excerpt,
            idempotency_key=event.client_event_id,
            rejection_reason=reason,
            retryable=spec.retryable,
            peer_may_have_applied=spec.peer_may_have_applied,
        )

    def close(self) -> None:
        if self.client is not None and self._owns_client:
            self.client.close()
            self.client = None

    def __enter__(self) -> "NurseExecClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def contract_markdown() -> str:
    """由 :data:`OUTBOUND_EVENT_SCHEMA` 生成出站契约 Markdown。

    生成的文档与代码同源：改字段只改 :data:`OUTBOUND_EVENT_SCHEMA`，契约测试
    会发现文档与模型不一致，而不是让人肉比对一张几百行的表。
    """
    lines = [
        "# 出站写回契约 · 护理执行平台",
        "",
        "> 本文件由 `infusion_monitor.adapters.nurse_exec.contract_markdown()` 从",
        "> `OUTBOUND_EVENT_SCHEMA` 生成，**请勿手改**——契约测试会比对本文件与模型字段。",
        "",
        "## 端点与鉴权",
        "",
        "| 项 | 值 |",
        "|------|------|",
        "| 方法与路径 | `POST {NURSE_EXEC_BASE_URL}/api/v1/infusion/events` |",
        "| 鉴权 | `Authorization: Bearer $NURSE_EXEC_API_TOKEN` |",
        "| 内容类型 | `application/json`（UTF-8） |",
        "| 幂等头 | `Idempotency-Key: <client_event_id>`，与请求体同字段同值 |",
        "| 重定向 | 不跟随（3xx 视为失败，避免把载荷发到别处） |",
        "| 系统代理 | 不继承（`trust_env=False`）：端点由 `NURSE_EXEC_BASE_URL` 明确给出，"
        "工作站的系统代理不得接管病区换液事件，也不得把传输层失败统一翻成 502 |",
        "",
        "## 字段规范",
        "",
        "| 字段 | 类型 | 必填 | 可空 | 说明 |",
        "|------|------|------|------|------|",
    ]
    for spec in OUTBOUND_EVENT_SCHEMA:
        lines.append(
            f"| `{spec.name}` | {spec.type} | {'是' if spec.required else '否'} "
            f"| {'是' if spec.nullable else '否'} | {spec.description} |"
        )
    lines += [
        "",
        "## 缺省语义",
        "",
        "* 可空且拿不到值的字段（`estimated_volume_ml` / `eta_minutes` / `fill_ratio`）"
        "在请求体里**整体不出现**，不是 `null`。对端若把缺省读成 0，会把「没测出来」"
        "读成「袋是空的」。",
        "* `event_type` 当前只取值 `infusion.bag.changed`。",
        "* 载荷只含床号、袋序、数量与事件，**不含患者姓名/住院号等可识别明细**。",
        "",
        "## 字段枚举",
        "",
        "| 字段 | 允许取值 |",
        "|------|----------|",
    ]
    for spec in OUTBOUND_EVENT_SCHEMA:
        if spec.enum:
            lines.append(f"| `{spec.name}` | " + " / ".join(f"`{value}`" for value in spec.enum) + " |")
    lines += [
        "",
        "## 与对端接口文档的对应表",
        "",
        "接挂时逐行对这张表：左边是本服务发出的字段名，右边是对端在自己接口文档里"
        "的落点与约定。名字相似不等于语义相同——`occurred_at` 是业务发生时刻而不是"
        "对端接收时刻，`clientEventId` 与 `Idempotency-Key` 必须是同一个值。",
        "",
        "| 本服务字段 | 对端契约位置 | 对端侧约定 |",
        "|------|------|----------|",
    ]
    for spec in OUTBOUND_EVENT_SCHEMA:
        binding = PEER_FIELD_BINDINGS[spec.name]
        lines.append(f"| `{spec.name}` | {binding.location} | {binding.note} |")
    lines += [
        "",
        "## 失败分类",
        "",
        "| 失败类别 | 可重试 | 终态 | 对端可能已收下 | 现场长什么样 |",
        "|------|------|------|------|------|",
    ]
    for spec in FAILURE_CLASSES.values():
        lines.append(
            f"| `{spec.kind}` | {'是' if spec.retryable else '否'} | `{spec.terminal_status}` "
            f"| {'**会**' if spec.peer_may_have_applied else '不会'} | {spec.description} |"
        )
    lines += [
        "",
        "> `对端可能已收下 = 会` 的三类（`read_timeout` / `write_timeout` / `half_open`）"
        "正是**幂等键不可省**的原因：它们既值得重试，又不能保证重发不会在对端变成两条。",
        "",
        "## 错误摘要格式",
        "",
        "退避耗尽后写进 `last_error`、并在待写回列表与飞检材料里展示的一行摘要：",
        "",
        "```",
        "kind=<失败类别> status=<HTTP 状态码，可无> attempts=<尝试次数> "
        "counts=<类别=次数,...> reason=<对端给的原因> last=<最后一次错误文本>",
        "```",
        "",
        "* `counts` **按类别分别计数**，因此「连接超时 3 次」与「读超时 3 次」在摘要里"
        "长得不一样——前者查网络配置，后者查对端处理速度。",
        "* 整条摘要经 `scrub()` 脱敏，凭据值不会进这条文本。",
        "",
        "## 响应约定",
        "",
        "| 状态码 | 本服务的处理 |",
        "|------|------|",
        "| 2xx | 写回置 `sent` |",
        "| 4xx | 业务拒绝，**不重试**，事件置 `failed`，并从响应体抽出对端原因记进摘要 |",
        "| 5xx | 可重试；退避重试耗尽后置 `pending_writeback` 并告警 |",
        "| 连接/读取超时 | 可重试；同上 |",
        "| 半开连接 | 可重试；请求很可能已到对端，靠 `Idempotency-Key` 保证不重复 |",
        "",
    ]
    return "\n".join(lines)