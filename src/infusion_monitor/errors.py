"""入站错误契约（中立层）：结构化 issue + 拒绝异常。

放在包顶层而不是 ``api/`` 下，因为 ``ingest/`` 也要用同一套错误契约抛错，
而方向必须是 ``ingest → errors``、``api → errors``，不能出现 ``ingest → api``。

**422 的 detail 一律是三段式**：``field``（哪个字段）+ ``reason``（为什么不行）
+ ``suggestion``（护士照着做能得到什么）。调用方不需要解析一句英文报错才知道
下一步该干什么——这是 spec「入站错误契约」的要求。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "Issue",
    "ValidationRejected",
    "ApiError",
    "ERROR_CODES",
    "OPERATIONAL_FAULTS",
    "ErrorCodeDoc",
    "FaultDoc",
    "error_code_docs",
    "error_code_table",
    "fault_table",
    "field_suggestion",
]


#: 错误码表（与 /docs 的 OpenAPI 描述逐一对应）。后续任务只允许在此扩充。
ERROR_CODES: dict[str, int] = {
    "INGEST_INVALID": 422,       # 入站帧序列/槽位/字段校验不通过
    "FRAMES_ZIP_REJECTED": 422,  # zip 结构不安全或成员不符合命名约定
    "NOT_FOUND": 404,            # 会话/袋位不存在
    "STATE_CONFLICT": 409,       # 业务状态迁移非法
    "WRITE_NOT_ALLOWED": 403,    # 越权写入：袋序/输液架不属于该会话（任务 10）
    "CALIBRATION_WINDOW_CONFLICT": 409,  # 标定生效区间与既有版本相交（任务 27）
    "UPLOAD_REJECTED": 422,      # 分片被拒：命名不合约定 / 片过大 / 片内编号从 0 重排（任务 32）
    "UPLOAD_NOT_OPEN": 409,      # 这次续传上传已结案，不再接收分片（任务 32）
    "UPLOAD_ALREADY_COMPLETED": 409,  # 已提交过，重复提交不再算一遍（任务 32）
    "UPLOAD_TOO_MANY_PENDING": 409,    # 未提交的分片上传堆到上限（任务 32）
    "UPLOAD_BUDGET_EXCEEDED": 413,    # 整段上传超出字节预算（任务 35）
    "UPLOAD_TIMEOUT": 408,           # 整段上传接收超时（任务 35）
    "UPSTREAM_UNAVAILABLE": 502,  # 出站对端不可达：失败类别与处置见 adapters.nurse_exec.FAILURE_CLASSES（任务 38）
    "WRITEBACK_REJECTED": 409,   # 对端业务拒绝：重放同一份载荷仍会被拒（任务 37）
    "WEBHOOK_SIGNATURE_INVALID": 401,   # webhook 签名缺失/格式不对/不匹配（任务 39）
    "WEBHOOK_TIMESTAMP_REJECTED": 401,  # webhook 时间戳不在重放窗口内（任务 39）
    "WEBHOOK_BODY_TOO_LARGE": 413,      # webhook 请求体超限（任务 39）
    "TELEMETRY_REJECTED": 422,          # 遥测载荷字段缺失/类型不对/滴速非正（任务 39）
    "INTERNAL": 500,             # 未预期内部错误
}

#: API 错误码 → 它对调用方意味着什么。``docs/error-codes.md`` 由本表生成，
#: 不要手改文档——文档与代码对不上正是挂接方最怕的事（拿着一份过期码表调接口）。
_ERROR_CODE_DOCS: tuple[tuple[str, str, str], ...] = (
    (
        "INGEST_INVALID",
        "帧序非法：帧数不在 12–1200、短边小于 320px、时间戳非单调递增、槽位超过 6 袋，"
        "或必填字段缺失。",
        "看 detail.issues[].suggestion 逐条照做后重传；同一次上传的问题会一次性全部返回，"
        "不必来回试。",
    ),
    (
        "FRAMES_ZIP_REJECTED",
        "帧 zip 结构不安全：含路径穿越成员、符号链接、异常压缩比（zip 炸弹）或成员命名不合约定。",
        "重新打包后重传。服务端不会把 zip 解到临时目录以外，收到该码说明这一包已被整体拒收。",
    ),
    (
        "NOT_FOUND",
        "会话或袋位不存在，或其帧目录已被保留期清理。",
        "确认 session_id 来自本次上传响应；已被清理的会话需要重新巡视上传一段短片。",
    ),
    (
        "STATE_CONFLICT",
        "业务状态迁移非法：例如 monitoring 直跳 replaced、已换袋的会话再次确认。",
        "新一轮巡视请让对端给新的 bag_seq；同一袋的重复确认请复用同一个 client_event_id。",
    ),
    (
        "WRITE_NOT_ALLOWED",
        "越权写入：确认换液时指定的袋序或输液架不属于该会话。",
        "按 detail.issues[].reason 里点出的两个袋序核对终端缓存的 session_id，勿凭记忆点确认。",
    ),
    (
        "CALIBRATION_WINDOW_CONFLICT",
        "标定生效区间与既有版本相交：同一袋型被登记了两个同时生效的版本。",
        "按 detail.issues[].suggestion 把本次登记的生效时刻挪到旧版之后，"
        "或给旧版补一个 effective_to 明确停用时刻；需要同时生效请先核对是否真的换了袋型。",
    ),
    (
        "UPLOAD_REJECTED",
        "分片被拒：成员命名不合约定、单片帧数超限，或分片里的帧号是从 0 重新编号的"
        "（那样无法判断该接在哪一段后面）。",
        "只拒这一片：已经收下的分片不受影响。按 detail.issues[].suggestion 改成整段全局"
        "序号命名后重传该片即可。",
    ),
    (
        "UPLOAD_NOT_OPEN",
        "这个 upload_id 已结案（completed 或 abandoned），不再接收分片。",
        "已提交过的无需重传；被保留期清理掉的请重新 POST /api/v1/infusion/uploads 开一次。",
    ),
    (
        "UPLOAD_ALREADY_COMPLETED",
        "同一 upload_id 已经提交过一次，重复提交不会重算。",
        "拿提交响应里的 bags[].session_id 去查余量；确实要重传整段请直接 POST "
        "/api/v1/infusion/tracks（那是覆盖式更新，会留下旧版本快照）。",
    ),
    (
        "UPLOAD_TOO_MANY_PENDING",
        "未提交的分片上传已堆到上限：多半是网络断过之后终端反复重开。",
        "先 POST /api/v1/infusion/uploads/cleanup 回收过保留期的半成品，"
        "或让护士重新拍一段整的直接走 /tracks。",
    ),
    (
        "UPLOAD_BUDGET_EXCEEDED",
        "这一段超过了单次整段上传的字节上限（UPLOAD_MAX_BYTES，默认 160MB）。"
        "**已收到的部分不作废**，只是这一包整体被退回。",
        "改走分片续传：POST /api/v1/infusion/uploads 开一次拿 upload_id → "
        "POST …/{upload_id}/chunks 逐片补 → GET …/{upload_id} 查断点（received_frames / "
        "missing_frames）→ POST …/{upload_id}/commit 提交；提交走的是与 /tracks 相同的"
        "主路径与缩放策略，两条路给同一段短片得到同一份数字。",
    ),
    (
        "UPLOAD_TIMEOUT",
        "接收这一段超过了单次整段上传的时限（UPLOAD_MAX_SECONDS，默认 120 秒）。"
        "病区网络断在传输中途是常态，不是异常。",
        "同上改走分片续传（POST /api/v1/infusion/uploads 起）；已收到的分片留在暂存目录，"
        "GET …/{upload_id} 能查到断点在哪一帧。",
    ),
    (
        "UPSTREAM_UNAVAILABLE",
        "出站对端不可达：护理执行平台连接超时、读超时或 5xx。",
        "本服务已把事件置 pending_writeback 并按指数退避重试 ≤2 次，不会丢事件；"
        "信息科在运行摘要里看待写回条数，对端恢复后可重放。",
    ),
    (
        "INTERNAL",
        "未预期内部错误，已记日志。",
        "把响应头里的 X-Request-Id 交给信息科，可据此定位到唯一一行日志。",
    ),
    (
        "WRITEBACK_REJECTED",
        "护理执行平台**业务拒绝**了这条换液事件（4xx）：载荷或业务状态不被接受。"
        "重放同一份载荷只会被再拒一次。",
        "先看 detail.issues[].reason 里对端给的原因（袋序不存在、输注单号不匹配、"
        "状态不允许等），核对该袋的输注单号与床号后再决定是改载荷还是改对端；"
        "这笔欠账仍留在待写回清单里，不会被丢掉。",
    ),
    (
        "WEBHOOK_SIGNATURE_INVALID",
        "入站 webhook 兜底通道的签名不通过：缺签名头、格式不是 `sha256=<hex>`、"
        "或与请求体算出来的不一致。**请求体不会被解析**，一条都没进缓存。",
        "核对两端的签名密钥是否同一份（本服务读 `WEBHOOK_SIGNING_SECRET`）与签名串口径："
        "带时间戳时签的是 `ts.body`，不带时签的是 body。密钥不一致就别去改请求体。",
    ),
    (
        "WEBHOOK_TIMESTAMP_REJECTED",
        "webhook 请求带的时间戳与服务端相差超过重放窗口（默认 ±300 秒）。",
        "先对时（病区 NTP），别把时间戳头去掉——去掉之后签名就只覆盖 body，"
        "等于放弃重放保护（响应里的 `timestamp_enforced` 会变成 false）。",
    ),
    (
        "WEBHOOK_BODY_TOO_LARGE",
        "webhook 请求体超过上限（默认 64KB）。",
        "一条泵速遥测只有几十字节，超限说明对端塞了别的东西进来（或发错了地址）。"
        "确认对端打的是这个端点；本服务不落盘、不解析超限请求体。",
    ),
    (
        "TELEMETRY_REJECTED",
        "遥测载荷不可用：`bag_seq` 或 `pump_rate_ml_per_h` 缺失/类型不对、滴速 ≤ 0，"
        "或载荷里的 `rack_id` 与 URL 路径段不一致。",
        "只影响这一条遥测，视觉主路径不受影响：滴速退回输注医嘱文本，袋位响应里 "
        "`telemetry_state=absent`。核对泵网关发的字段名与架号；"
        "滴速 ≤ 0 一律拒收——把「泵停了」当成有效速率会把 ETA 算成无穷大。",
    ),
)


@dataclass(frozen=True)
class ErrorCodeDoc:
    code: str
    status: int
    meaning: str
    handling: str


@dataclass(frozen=True)
class FaultDoc:
    key: str
    meaning: str
    impact: str
    handling: str


#: **运行期故障**不是 HTTP 错误：请求照样成功返回，但服务内部某条链路已经不可用。
#: 它们只出现在运行摘要与启动日志里。把两者混成一张表，值班的就会对着
#: 「HTTP 200 但摘要说没连上」这种组合查半天。
_OPERATIONAL_FAULTS: tuple[tuple[str, str, str, str], ...] = (
    (
        "credentials_missing",
        "7 组凭据位缺项或为空。",
        "服务起不来（fail fast），不存在「起来了但拿不到出站凭据」的中间态。",
        "按报错逐行点名补齐环境变量后重启；`python scripts/check_config.py` 可在启动前自查。",
    ),
    (
        "nurse_exec_unreachable",
        "护理执行平台出站不可达（连接超时 / 读超时 / 5xx）。",
        "换液事件进重试队列并置 pending_writeback；护士侧的点名与确认不受影响。",
        "看运行摘要的 pending_writeback_count；对端恢复后由信息科重放，服务不自动无限重试。",
    ),
    (
        "model_unreachable",
        "模型网关不可达、超时、返回非 JSON 或结构化校验不通过。",
        "该次判定退确定性阈值路径，标 decision_source=threshold_fallback + "
        "pending_manual_review=true。**不会**推进业务状态、**不会**写回。",
        "阈值路径给出的结论仍可用但需人工复核；看运行摘要的 threshold_fallback_count 是否异常升高。",
    ),
    (
        "mqtt_disconnected",
        "输液泵遥测 MQTT 断连（未启用则恒为 stopped）。",
        "视觉主路径完全不受影响；ETA 退回医嘱速率，交叉校验能力下降。",
        "看运行摘要的 channels.telemetry.last_message_at 判断遥测是否在流；"
        "确认 broker 地址与凭据后重启订阅。",
    ),
    (
        "frames_root_not_writable",
        "会话帧目录不可写（磁盘满 / 目录权限不对）。",
        "上传会以 500 失败；已落库的历史会话不受影响。",
        "看磁盘剩余空间与目录属主；见 INSTALL.md 的目录权限一节。",
    ),
    (
        "quota_exceeded",
        "模型网关单日调用上限被触发。",
        "该次判定直接走阈值路径并记 reason=quota_exceeded，模型不被调用。",
        "按需调 LLM_DAILY_CALL_LIMIT，或在非巡视高峰期再放开。",
    ),
)


def error_code_docs() -> tuple[ErrorCodeDoc, ...]:
    """错误码表的结构化视图（测试与文档生成共用）。"""
    return tuple(
        ErrorCodeDoc(code=code, status=ERROR_CODES[code], meaning=meaning, handling=handling)
        for code, meaning, handling in _ERROR_CODE_DOCS
        if code in ERROR_CODES
    )


def operational_faults() -> tuple[FaultDoc, ...]:
    return tuple(
        FaultDoc(key=key, meaning=meaning, impact=impact, handling=handling)
        for key, meaning, impact, handling in _OPERATIONAL_FAULTS
    )


def error_code_table() -> str:
    """生成 ``docs/error-codes.md`` 的 API 错误码小节（Markdown 表格）。"""
    lines = [
        "| 错误码 | HTTP | 含义 | 调用方怎么处理 |",
        "|--------|------|------|----------------|",
    ]
    for doc in error_code_docs():
        lines.append(f"| `{doc.code}` | {doc.status} | {doc.meaning} | {doc.handling} |")
    return "\n".join(lines)


def fault_table() -> str:
    """生成 ``docs/error-codes.md`` 的运行期故障小节。"""
    lines = [
        "| 故障 | 含义 | 影响面 | 处理方式 |",
        "|------|------|--------|----------|",
    ]
    for doc in operational_faults():
        lines.append(f"| `{doc.key}` | {doc.meaning} | {doc.impact} | {doc.handling} |")
    return "\n".join(lines)


def error_code_markdown() -> str:
    """整份 ``docs/error-codes.md`` 的正文（文档同步测试据此比对）。"""
    return f"""# 错误码与运行期故障对照

> 本文件由 `src/infusion_monitor/errors.py` 的 `ERROR_CODES` / `_ERROR_CODE_DOCS` /
> `_OPERATIONAL_FAULTS` 生成，`tests/test_summary.py` 会比对是否同步。**请改代码，不要改本文。**

两件事要分开看：

* **API 错误码**：调用方拿到了 HTTP 错误响应。响应体恒为
  `{{"detail": {{"error_code": …, "message": …, "issues": [{{"code","field","reason","suggestion"}}]}}}}`。
* **运行期故障**：请求**成功返回**，但服务内部某条链路已经不可用。它们只出现在
  运行摘要 `GET /api/v1/summary` 与启动日志里。把两者混成一张表，值班的就会对着
  「HTTP 200 但摘要说没连上」这种组合查半天。

## API 错误码

{error_code_table()}

## 运行期故障

{fault_table()}

## 排障入口

| 想知道 | 看哪里 |
|--------|--------|
| 某一次请求整条链发生了什么 | 日志里按 `request_id` 过滤；响应头 `X-Request-Id` 是同一个 id |
| 现在有几袋等着换、有几条没送出去 | `GET /api/v1/summary` 的 `sessions.needs_change_count` 与 `writeback.pending_writeback_count` |
| 服务能不能用 | `GET /api/v1/health` |
| 配置齐不齐 | `python scripts/check_config.py`（退出码 0 = 齐） |
| 7 组凭据位分别是什么 | `.env.example`，本仓库不含任何真实值 |
"""


#: 兼容旧引用：部分测试按名字取这个常量。
OPERATIONAL_FAULTS = tuple(
    FaultDoc(key=k, meaning=m, impact=i, handling=h)
    for k, m, i, h in _OPERATIONAL_FAULTS
)


@dataclass(frozen=True)
class Issue:
    """一条可核对的入站问题。

    :param code: 机器可读的细分码（如 ``frames_too_few``），测试按它断言；
    :param field: 出问题的入站字段（``frames`` / ``slots`` / ``captured_at`` …）；
    :param reason: 为什么不行（写事实，带上实测值）；
    :param suggestion: 护士或调用方**下一步照着做什么**的中文建议。
    """

    code: str
    field: str
    reason: str
    suggestion: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "field": self.field,
            "reason": self.reason,
            "suggestion": self.suggestion,
        }


class ValidationRejected(Exception):
    """校验不通过：带一串 issue，由 API 层转成 422。"""

    def __init__(self, issues: Sequence[Issue], *, error_code: str = "INGEST_INVALID") -> None:
        if not issues:
            raise ValueError("ValidationRejected 至少要带一条 issue")
        self.issues: tuple[Issue, ...] = tuple(issues)
        self.error_code = error_code
        super().__init__("；".join(f"{i.field}: {i.reason}" for i in self.issues))

    def as_detail(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": "入站帧序列未通过校验，请按每条 issue 的 suggestion 调整后重传",
            "issues": [issue.as_dict() for issue in self.issues],
        }


class ApiError(Exception):
    """非校验类 API 错误（404 / 409 / 5xx），detail 结构与 422 保持一致。"""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        status_code: int | None = None,
        issues: Iterable[Issue] = (),
    ) -> None:
        self.error_code = error_code
        self.message = message
        self.status_code = status_code or ERROR_CODES.get(error_code, 500)
        self.issues: tuple[Issue, ...] = tuple(issues)
        super().__init__(message)

    def as_detail(self) -> dict[str, Any]:
        detail: dict[str, Any] = {"error_code": self.error_code, "message": self.message}
        if self.issues:
            detail["issues"] = [issue.as_dict() for issue in self.issues]
        return detail


#: 常见入站字段的通用建议文案。校验器给不出更具体的建议时用这里的兜底。
_DEFAULT_SUGGESTION = "请按接口文档的字段要求补齐后重传；不确定时联系信息科查接口版本。"

_FIELD_SUGGESTIONS: dict[str, str] = {
    "rack_id": "rack_id 不能为空。请填输液架上贴的架号（与护理执行平台登记的一致），不要填床号。",
    "bed_no": "bed_no 不能为空。请填该输液架当前对应的床号。",
    "infusion_order_no": "infusion_order_no 不能为空。请填医嘱/护理单元系统给的输注单号，产品侧不自行编号。",
    "captured_at": "captured_at 须为 ISO8601 时间（如 2026-10-22T09:41:00+08:00），请用本机时间而非 UTC 缩写。",
    "slots": "slots 须为 JSON 数组，每项含 bag_seq 与 slot_index；单架最多 6 袋，请按架分批上传。",
    "frames": "frames 须为 zip，成员命名为 frame_00000.jpg 起的连续序号，只含帧图片。",
    "frame_times": "frame_times 须为与帧数等长的 ISO8601 时间数组，按帧顺序给出。",
    "shift_key": "shift_key 不传时按 captured_at 的日期自动生成；如需按巡视轮次区分请显式传 YYYY-MM-DD-AM/PM。",
    "client_event_id": "client_event_id 建议传调用方生成的唯一幂等键（重传同一段请复用同一个值）。",
    "session_id": "session_id 不存在。请用上传响应里的 bags[].session_id。",
    "bag_seq": "bag_seq 不能为空。请填这一袋在输液架上的袋序（与上传响应里 bags[].bag_seq 一致）；"
               "系统不接受「不指明是哪一袋」的换液确认。",
    "operator_id": "operator_id 不能为空。请用移动护理终端上当前登录的护士工号确认换液。",
    "body": _DEFAULT_SUGGESTION,
}


def field_suggestion(field_name: str) -> str:
    """取某字段的通用建议文案（校验器没给出更具体建议时的兜底）。"""
    return _FIELD_SUGGESTIONS.get(field_name, _DEFAULT_SUGGESTION)