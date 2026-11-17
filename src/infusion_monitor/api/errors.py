"""异常 → HTTP 响应的统一映射。

目标：**调用方拿到 422 时，不需要猜**。detail 恒为

```json
{"error_code": "INGEST_INVALID",
 "message": "…",
 "issues": [{"code": "frames_too_few", "field": "frames",
             "reason": "本次只收到 6 帧，少于下限 12 帧",
             "suggestion": "请把手机对准输液架连续拍 10–20 秒…"}]}
```

FastAPI 自带的校验错误（缺表单字段、类型不对）也映射到同一形状，
否则调用方会看到两套互不兼容的错误结构。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..errors import ERROR_CODES, ApiError, Issue, ValidationRejected, field_suggestion

__all__ = ["install_error_handlers", "ERROR_CODES"]

logger = logging.getLogger("infusion_monitor.api")


def _json(status_code: int, detail: dict) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def _from_validation_error(exc: RequestValidationError) -> tuple[list[Issue], int]:
    """把 FastAPI 的校验错误折成三段式 issue。"""
    issues: list[Issue] = []
    status_code = 422
    for error in exc.errors():
        location = [str(part) for part in error.get("loc", ()) if part not in ("body",)]
        name = location[-1] if location else "body"
        kind = error.get("type", "")
        code = {
            "missing": "field_missing",
            "missing_part": "field_missing",
            "string_type": "field_type_mismatch",
            "int_type": "field_type_mismatch",
            "value_error": "field_malformed",
        }.get(kind, "field_invalid")
        if code == "field_type_mismatch":
            suggestion = f"{name} 收到的类型不对；请按接口文档的字段要求传值。"
        else:
            suggestion = field_suggestion(name)
        issues.append(
            Issue(
                code=code,
                field=name,
                reason=f"{error.get('msg', '字段不合法')}",
                suggestion=suggestion,
            )
        )
    if not issues:
        issues.append(
            Issue(
                code="request_invalid",
                field="body",
                reason="请求体未通过校验",
                suggestion=field_suggestion("body"),
            )
        )
        if status_code != 422:
            status_code = 422
    return issues, status_code


def install_error_handlers(app: FastAPI) -> None:
    """把四类异常统一成同一套 detail 结构。"""

    @app.exception_handler(ValidationRejected)
    async def _on_validation_rejected(_: Request, exc: ValidationRejected) -> JSONResponse:
        return _json(ERROR_CODES.get(exc.error_code, 422), exc.as_detail())

    @app.exception_handler(ApiError)
    async def _on_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _json(exc.status_code, exc.as_detail())

    @app.exception_handler(RequestValidationError)
    async def _on_request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        issues, status_code = _from_validation_error(exc)
        return _json(
            status_code,
            {
                "error_code": "INGEST_INVALID",
                "message": "请求未通过校验，请按每条 issue 的 suggestion 调整后重传",
                "issues": [issue.as_dict() for issue in issues],
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _on_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        error_code = {404: "NOT_FOUND", 409: "STATE_CONFLICT", 422: "INGEST_INVALID"}.get(
            exc.status_code, "INTERNAL" if exc.status_code >= 500 else "INGEST_INVALID"
        )
        return _json(exc.status_code, {"error_code": error_code, "message": str(exc.detail)})

    @app.exception_handler(Exception)
    async def _on_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # 只记异常类型与消息，不把凭据或患者明细带进日志
        logger.exception("未预期内部错误：%s", type(exc).__name__)
        return _json(
            500,
            {
                "error_code": "INTERNAL",
                "message": "服务内部错误，已记录；请携带 request_id 联系信息科排查",
            },
        )