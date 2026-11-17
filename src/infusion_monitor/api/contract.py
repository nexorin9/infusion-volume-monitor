"""OpenAPI 契约与错误码表的一致性（``/docs`` 里看到的就是服务真实返回的）。

FastAPI 默认给每个 operation 挂一个 422 ``HTTPValidationError``
（``{"detail": [{"loc": …, "msg": …}]}``），而本服务真实返回的是
``{"detail": {"error_code": …, "message": …, "issues": [{code, field, reason, suggestion}]}}``。
两者不一致的后果很实际：挂接方照 ``/docs`` 写分支，线上 422 一条也匹配不上。

所以这里做三件事，源都在 :mod:`infusion_monitor.errors` 与
:mod:`infusion_monitor.api.schemas`：

1. **逐条注入错误响应**：把 ``ROUTE_ERROR_CODES`` 里登记的错误码，按
   :func:`~infusion_monitor.errors.error_code_docs` 的中文说明写进每个 operation 的
   ``responses``，并指向真正的 :class:`ErrorDetailOut`。挂不上的码在这里直接抛
   ``ValueError``——宁可在建应用时炸，也不要让码表和接口悄悄对不上。
2. **补上入站 multipart 体**：``/tracks`` 的 requestBody 换成显式的
   :func:`~infusion_monitor.api.schemas.track_upload_body_schema`（带必填清单与字段说明）。
3. **机器可读码表**：``GET /api/v1/error-codes``，挂接方不必去爬 HTML 文档。

不变式（``tests/test_openapi_schema.py`` 逐条断言）：

* ``set(ERROR_CODES)`` 与 OpenAPI 里出现过的错误码**恰好相等**——不多不少；
* 每个错误码至少被一个 operation 记录，且描述里带着码名；
* 每个 operation 的 422 都指向 ``ErrorDetailOut``，不再指向 ``HTTPValidationError``；
* 业务代码里 ``raise ApiError("X")`` / ``error_code="X"`` 的 X 一定在码表里，且被
  至少一个 operation 记录（新增错误码忘了登记文档，测试会红）。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from fastapi import APIRouter, FastAPI
from fastapi.openapi.utils import get_openapi

from ..errors import ERROR_CODES, error_code_docs, operational_faults
from .schemas import (
    ErrorCodeListOut,
    ErrorCodeOut,
    ErrorDetailOut,
    ErrorIssueOut,
    track_upload_body_schema,
)

__all__ = [
    "ROUTE_ERROR_CODES",
    "OUTBOUND_ONLY_CODES",
    "ERROR_DETAIL_REF",
    "custom_openapi",
    "install_error_contract",
    "error_codes_router",
    "documented_routes",
    "codes_documented_for",
]

#: 错误 detail 的 schema 引用名。FastAPI 按类名注册组件，所以直接用模型名。
ERROR_DETAIL_REF = "#/components/schemas/ErrorDetailOut"

#: (METHOD, 路径模板) → 该接口可能返回的错误码。**登记错不报错、登记漏了测试会红**，
#: 这是刻意的：路由表人手维护，业务侧的 ``raise`` 由 AST 扫描交叉核对。
ROUTE_ERROR_CODES: dict[tuple[str, str], tuple[str, ...]] = {
    ("POST", "/api/v1/infusion/tracks"): (
        "INGEST_INVALID",
        "FRAMES_ZIP_REJECTED",
        "UPLOAD_BUDGET_EXCEEDED",
        "UPLOAD_TIMEOUT",
        "INTERNAL",
    ),
    ("GET", "/api/v1/infusion/sessions/{session_id}"): ("NOT_FOUND", "INTERNAL"),
    ("GET", "/api/v1/infusion/sessions/{session_id}/versions"): ("NOT_FOUND", "INTERNAL"),
    ("POST", "/api/v1/infusion/uploads"): ("INGEST_INVALID", "UPLOAD_TOO_MANY_PENDING"),
    ("POST", "/api/v1/infusion/uploads/{upload_id}/chunks"): (
        "UPLOAD_NOT_OPEN",
        "UPLOAD_REJECTED",
        "NOT_FOUND",
        "INTERNAL",
    ),
    ("GET", "/api/v1/infusion/uploads/{upload_id}"): ("NOT_FOUND",),
    ("POST", "/api/v1/infusion/uploads/{upload_id}/commit"): (
        "UPLOAD_NOT_OPEN",
        "UPLOAD_ALREADY_COMPLETED",
        "UPLOAD_REJECTED",
        "INGEST_INVALID",
        "FRAMES_ZIP_REJECTED",
        "NOT_FOUND",
    ),
    ("POST", "/api/v1/infusion/uploads/cleanup"): ("INTERNAL",),
    ("POST", "/api/v1/infusion/calibrations"): ("CALIBRATION_WINDOW_CONFLICT", "INGEST_INVALID"),
    ("GET", "/api/v1/infusion/calibrations"): (),
    ("GET", "/api/v1/infusion/calibrations/recalc"): (),
    ("GET", "/api/v1/infusion/calibrations/deviations"): (),
    ("GET", "/api/v1/infusion/calibrations/{bag_type_code}/versions"): ("NOT_FOUND",),
    ("POST", "/api/v1/infusion/calibrations/{bag_type_code}/reviews/{alert_id}"): (
        "NOT_FOUND",
        "STATE_CONFLICT",
        "INGEST_INVALID",
    ),
    ("GET", "/api/v1/infusion/calibrations/{bag_type_code}/reviews"): ("NOT_FOUND",),
    ("POST", "/api/v1/infusion/claim"): ("NOT_FOUND", "INGEST_INVALID"),
    ("POST", "/api/v1/infusion/decision"): ("NOT_FOUND", "STATE_CONFLICT", "WRITE_NOT_ALLOWED"),
    ("POST", "/api/v1/infusion/transition"): (
        "NOT_FOUND",
        "STATE_CONFLICT",
        "WRITE_NOT_ALLOWED",
        "INGEST_INVALID",
    ),
    ("GET", "/api/v1/infusion/pickups"): ("NOT_FOUND",),
    ("GET", "/api/v1/infusion/pending"): ("INGEST_INVALID",),
    ("POST", "/api/v1/infusion/pending/{event_id}/replay"): (
        "NOT_FOUND",
        "WRITEBACK_REJECTED",
        "INGEST_INVALID",
    ),
    ("GET", "/api/v1/infusion/audit"): ("INGEST_INVALID",),
    ("GET", "/api/v1/infusion/audit/manifest"): ("INGEST_INVALID",),
    ("POST", "/api/v1/infusion/telemetry/webhook/{rack_id}"): (
        "NOT_FOUND",
        "WEBHOOK_SIGNATURE_INVALID",
        "WEBHOOK_TIMESTAMP_REJECTED",
        "WEBHOOK_BODY_TOO_LARGE",
        "TELEMETRY_REJECTED",
    ),
    ("GET", "/api/v1/error-codes"): (),
    ("GET", "/api/v1/health"): (),
    ("GET", "/api/v1/summary"): ("INTERNAL",),
}

#: 不进 ``responses`` 的码：它们是**出站**故障（对端不可达），由运行摘要暴露，
#: 不会出现在本服务的入站响应里。留着它们是为了让码表完整可查。
OUTBOUND_ONLY_CODES = ("UPSTREAM_UNAVAILABLE",)


def _doc_text(code: str) -> str:
    for doc in error_code_docs():
        if doc.code == code:
            return f"{doc.meaning}\n\n处理方式：{doc.handling}"
    raise KeyError(code)


def codes_documented_for(method: str, path: str) -> tuple[str, ...]:
    """某接口登记的错误码。"""
    return ROUTE_ERROR_CODES.get((method.upper(), path), ())


def documented_routes() -> dict[tuple[str, str], tuple[str, ...]]:
    """已登记的接口 → 错误码（供 ``GET /api/v1/error-codes`` 反查）。"""
    return dict(ROUTE_ERROR_CODES)


def validate_route_table() -> None:
    """建应用时核对：登记的码存在、每个码都有人认领。

    认领不到只有两种可能——码表里加了一个没人会抛的码，或者忘了把新错误码登记到
    接口上。两种都该在这里炸出来，而不是等挂接方在 /docs 里找不到。
    """
    unknown = sorted(
        {code for codes in ROUTE_ERROR_CODES.values() for code in codes if code not in ERROR_CODES}
    )
    if unknown:
        raise ValueError(f"接口表登记了码表里没有的错误码：{unknown}")
    covered = {code for codes in ROUTE_ERROR_CODES.values() for code in codes}
    uncovered = sorted(set(ERROR_CODES) - covered - set(OUTBOUND_ONLY_CODES))
    if uncovered:
        raise ValueError(f"错误码表里有码没被任何接口登记：{uncovered}")


def _error_response(codes: tuple[str, ...]) -> dict[str, Any]:
    """一组共处同一 HTTP 状态码的错误码 → 一个 OpenAPI response 对象。

    OpenAPI 一个 status 只能有一个 response，而码表里 422 上就挂着
    ``INGEST_INVALID`` 与 ``FRAMES_ZIP_REJECTED`` 两个码。硬写会互相覆盖，
    挂接方在 /docs 里就只看得到最后一个——所以按状态码归并，描述里逐码点名。
    """
    if not codes:
        raise ValueError("错误响应至少要带一个码")
    blocks = [f"{code}：{_doc_text(code)}" for code in codes]
    primary = codes[0]
    return {
        "description": "\n\n".join(blocks),
        "content": {
            "application/json": {
                "schema": {"$ref": ERROR_DETAIL_REF},
                "example": {
                    "detail": {
                        "error_code": primary,
                        "message": _doc_text(primary).splitlines()[0],
                        "issues": [
                            {
                                "code": "frames_too_few",
                                "field": "frames",
                                "reason": "本次只收到 6 帧，少于下限 12 帧",
                                "suggestion": "请把手机对准输液架连续拍 10–20 秒…",
                            }
                        ],
                    }
                },
            }
        },
    }


def _group_by_status(codes: Iterable[str]) -> dict[int, tuple[str, ...]]:
    """错误码 → HTTP 状态码 → 归并后的码组（组内顺序稳定，便于比对文档）。"""
    grouped: dict[int, list[str]] = {}
    for code in codes:
        grouped.setdefault(ERROR_CODES[code], []).append(code)
    return {status: tuple(group) for status, group in sorted(grouped.items())}


def _patch_operations(spec: dict[str, Any]) -> None:
    for path, operations in spec.get("paths", {}).items():
        for method, operation in operations.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            responses = operation.setdefault("responses", {})
            codes = codes_documented_for(method, path)
            # 框架级校验错误（缺字段 / 类型不对）在 ``api/errors.py`` 里被统一折成
            # INGEST_INVALID 三段式，所以**每个**接口都真的会吐这个 422。
            # 不显式登记的那些接口，也要把它顶掉 FastAPI 默认的 HTTPValidationError。
            for status, group in _group_by_status((*codes, "INGEST_INVALID")).items():
                responses[str(status)] = _error_response(group)


def _swap_request_bodies(spec: dict[str, Any]) -> None:
    """把 /tracks 的入站体换成显式 schema（保持 multipart/form-data 不变）。"""
    schemas = spec.setdefault("components", {}).setdefault("schemas", {})
    schemas["TrackUploadForm"] = track_upload_body_schema()
    operation = spec.get("paths", {}).get("/api/v1/infusion/tracks", {}).get("post")
    if not operation:
        return
    body = operation.get("requestBody")
    if not body:
        return
    multipart = body.get("content", {}).get("multipart/form-data")
    if not multipart:
        return
    multipart["schema"] = {"$ref": "#/components/schemas/TrackUploadForm"}
    body["required"] = True


def custom_openapi(app: FastAPI) -> dict[str, Any]:
    """本服务的 OpenAPI 文档真源（在 ``app.openapi`` 上装好）。"""
    if app.openapi_schema:
        return app.openapi_schema
    validate_route_table()
    spec = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schemas = spec.setdefault("components", {}).setdefault("schemas", {})
    issue_schema = ErrorIssueOut.model_json_schema(ref_template="#/components/schemas/{model}")
    issue_schema.pop("$defs", None)
    detail_schema = ErrorDetailOut.model_json_schema(ref_template="#/components/schemas/{model}")
    detail_schema.pop("$defs", None)
    schemas["ErrorIssueOut"] = issue_schema
    schemas["ErrorDetailOut"] = detail_schema
    _patch_operations(spec)
    _swap_request_bodies(spec)
    app.openapi_schema = spec
    return spec


def install_error_contract(app: FastAPI) -> None:
    """把上面那套错误码契约装到应用上。"""
    validate_route_table()
    app.openapi = lambda: custom_openapi(app)  # type: ignore[method-assign]


error_codes_router = APIRouter(prefix="/api/v1", tags=["contract"])


@error_codes_router.get(
    "/error-codes",
    response_model=ErrorCodeListOut,
    summary="错误码表（机器可读）",
    description=(
        "与 /docs 各接口 responses 里写的错误码同源（同一份 ERROR_CODES）。"
        "挂接方拉一次就能把分支写全，不必去爬 HTML 文档。"
    ),
)
async def list_error_codes() -> ErrorCodeListOut:
    docs = error_code_docs()
    routes_by_code: dict[str, list[str]] = {}
    for (method, path), codes in ROUTE_ERROR_CODES.items():
        for code in codes:
            routes_by_code.setdefault(code, []).append(f"{method} {path}")
    rows = [
        ErrorCodeOut(
            code=doc.code,
            status=doc.status,
            meaning=doc.meaning,
            handling=doc.handling,
            routes=sorted(routes_by_code.get(doc.code, ())),
        )
        for doc in docs
    ]
    return ErrorCodeListOut(
        error_codes=rows,
        operational_faults=[
            {
                "key": fault.key,
                "meaning": fault.meaning,
                "impact": fault.impact,
                "handling": fault.handling,
            }
            for fault in operational_faults()
        ],
    )


def codes_in_openapi(spec: Mapping[str, Any]) -> set[str]:
    """OpenAPI 文档里出现过的错误码集合（按 responses 描述里点名的码判定）。"""
    found: set[str] = set()
    for operations in spec.get("paths", {}).values():
        for method, operation in operations.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            for response in operation.get("responses", {}).values():
                description = str(response.get("description", ""))
                found.update(code for code in ERROR_CODES if f"{code}：" in description)
    return found


def codes_by_status_in_openapi(spec: Mapping[str, Any]) -> dict[int, set[str]]:
    """HTTP 状态码 → 该状态下文档里点名的错误码集合。"""
    grouped: dict[int, set[str]] = {}
    for path, operations in spec.get("paths", {}).items():
        for method, operation in operations.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            for status, response in operation.get("responses", {}).items():
                if not str(status).isdigit():
                    continue
                description = str(response.get("description", ""))
                names = {code for code in ERROR_CODES if f"{code}：" in description}
                if names:
                    grouped.setdefault(int(status), set()).update(names)
    return grouped


def all_operations(spec: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    for path, operations in spec.get("paths", {}).items():
        for method in operations:
            if method.lower() in {"get", "post", "put", "patch", "delete"}:
                yield method.upper(), path