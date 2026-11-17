"""审计包导出：``GET /api/v1/infusion/audit`` 与 ``…/audit/manifest``。

**这一层只读。** 它导出的是已经落在 ``audit_log`` 里的行，一个字都不写回：飞检
留档的动作本身不能改变被留档的内容——否则「查一次有没有人动过」会把这次查询也
变成一条新记录，下一次再查就永远对不上。

**两个端点，一份数据的两种读法**：

* ``…/audit`` → **CSV 明细**（UTF-8 BOM + CRLF，Excel 双击能开），响应头带上
  行数、链摘要与文件名，浏览器直接落成「审计包_<范围>_<日期>.csv」；
* ``…/audit/manifest`` → **包的首页 JSON**：导了多少行、七类动作各多少条、链摘要。
  护理部打印这页签收，内审把摘要抄进底稿。

**过滤维度就是飞检的问法**：按输液架（「这一架当天换了几次液」）、按床号（「这一床
这轮有没有人动过」）、按时间窗（「昨天上午 AM 班那一段」）、按类别（「有没有写回
挂起的」）。``category`` 用 :data:`~store.audit_chain.AUDIT_CATEGORIES` 的七类
问句名，而不是内部动作码——飞检不会问「``writeback.retry`` 是什么」。

**空结果照样出包**（表头 + 0 行 + manifest 计数 0）：「昨天没有异常换液」本身
就是飞检要的答案，一个 404 反而会让人怀疑数据丢了。
"""

from __future__ import annotations

import logging
from urllib.parse import quote

from fastapi import APIRouter, Query, Request, Response

from ..errors import ApiError, Issue
from ..logging_utils import request_id_of
from ..store.audit_chain import AUDIT_CATEGORIES, OTHER_CATEGORY, actions_in_categories
from ..store.audit_export import AuditFilter, collect_package
from ..store.repositories import utc_now
from .schemas import AuditFilterOut, AuditManifestOut

__all__ = ["ALL_CATEGORIES", "MAX_EXPORT_ROWS", "build_filter", "router"]

logger = logging.getLogger("infusion_monitor.api.audit")

router = APIRouter(prefix="/api/v1/infusion", tags=["audit"])

#: 可用的类别键。导出端点对外暴露的词汇表就是这个，别的拼法一律 422 而不是静默忽略。
ALL_CATEGORIES: tuple[str, ...] = tuple(AUDIT_CATEGORIES) + (OTHER_CATEGORY,)

#: 单次导出行数上限。病区侧单机库留三年也就十万级，但**一屏装不下的包没人看**，
#: 而且「导全量」通常是忘了加过滤条件——超限直接报错并说清怎么筛，比给一个
#: 十几 MB 的 CSV 让它在 Excel 里卡死强。
MAX_EXPORT_ROWS = 20000

#: 时间窗跨度上限（天）。跨月的导出会带上别的班次，留档时对不上「这一班」的口径。
MAX_WINDOW_DAYS = 93


def _ascii_header(value: str) -> str:
    """把值转成能进 HTTP 头的 ASCII。

    HTTP 头按 latin-1 编码，而范围标识可能是「全病区」这类中文——直接塞进去会在
    响应构造时抛 ``UnicodeEncodeError``，把一次正常的导出变成 500。非 ASCII 一律
    百分号编码：只出现在响应头里，展示仍以 manifest 与文件名里的原文为准。
    """
    try:
        value.encode("latin-1")
        return value
    except UnicodeEncodeError:
        return quote(value, safe="")


def build_filter(
    *,
    rack_id: str | None,
    bed_no: str | None,
    occurred_from: str | None,
    occurred_to: str | None,
    categories: tuple[str, ...],
    actions: tuple[str, ...],
    outcomes: tuple[str, ...],
    bag_seq: str | None,
    limit: int,
) -> AuditFilter:
    """把查询参数合成 :class:`~store.audit_export.AuditFilter`，顺带做校验。

    ``category`` 与 ``action`` **不是替代关系而是并集**：类别展开成动作码后与显式
    动作码取并集（``category=confirmation&action=pickup.named`` = 两类都要）。
    写成交集的话，护理部想「只要护士确认、只要 state.confirmed」就得先查文档才知道
    这俩是同一个东西。
    """
    unknown = [c for c in categories if c not in ALL_CATEGORIES]
    if unknown:
        raise ApiError(
            "INGEST_INVALID",
            f"未知的审计类别：{'、'.join(unknown)}",
            issues=[
                Issue(
                    code="audit_category_unknown",
                    field="category",
                    reason=f"请求里的类别不在可用清单内：{unknown}",
                    suggestion=f"可用类别：{'、'.join(ALL_CATEGORIES)}（每个类别的中文名见 "
                    f"GET /api/v1/infusion/audit/manifest 的 category_labels_zh）。",
                )
            ],
        )
    bad_outcomes = [o for o in outcomes if o not in ("accepted", "rejected")]
    if bad_outcomes:
        raise ApiError(
            "INGEST_INVALID",
            f"未知的审计结果：{'、'.join(bad_outcomes)}",
            issues=[
                Issue(
                    code="audit_outcome_unknown",
                    field="outcome",
                    reason=f"结果只能是 accepted / rejected，收到：{bad_outcomes}",
                    suggestion="accepted=动作被接受；rejected=动作被拒（含被闸门压下的点名）。",
                )
            ],
        )
    if limit > MAX_EXPORT_ROWS:
        raise ApiError(
            "INGEST_INVALID",
            f"limit={limit} 超出单次导出上限 {MAX_EXPORT_ROWS}",
            issues=[
                Issue(
                    code="limit_too_large",
                    field="limit",
                    reason=f"请求 limit={limit}，上限 {MAX_EXPORT_ROWS}",
                    suggestion="请用 rack_id / bed_no / 时间窗缩小范围后分段导出，"
                    "分段导出的每段各自带链摘要，合并时按 audit_id 排序即可复核。",
                )
            ],
        )
    expanded = list(actions)
    if categories:
        expanded.extend(actions_in_categories(categories))
    # 去重后按动作码排序：同一组条件导出两次要得到同一份 SQL 参数顺序。
    merged = tuple(sorted(set(expanded)))
    return AuditFilter(
        rack_id=rack_id,
        bed_no=bed_no,
        occurred_from=occurred_from,
        occurred_to=occurred_to,
        actions=merged,
        outcomes=tuple(outcomes),
        bag_seq=bag_seq,
        limit=limit,
    )


@router.get(
    "/audit",
    summary="导出审计包明细（CSV，只读）",
    description=(
        "按输液架 / 床号 / 时间窗 / 类别导出审计链明细，返回 UTF-8 BOM 的 CSV。\n\n"
        "响应头 `X-Audit-Row-Count` 给行数、`X-Audit-Chain-Digest` 给链摘要"
        "（留档人重算一遍，对不上说明这段审计被动过）。\n\n"
        "**只读端点**：不写任何一行审计，查询本身不改变被留档的内容。"
    ),
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}, "description": "审计包 CSV 明细"}},
)
async def export_audit_csv(
    request: Request,
    rack_id: str | None = Query(default=None, description="只导这一架"),
    bed_no: str | None = Query(default=None, description="只导这一床"),
    occurred_from: str | None = Query(default=None, description="起始时刻（含），ISO8601"),
    occurred_to: str | None = Query(default=None, description="截止时刻（含），ISO8601"),
    category: list[str] | None = Query(
        default=None,
        description="按问句类别过滤，可重复：claim / calibration / pickup / confirmation / "
        "writeback_sent / writeback_failed / writeback_pending / other",
    ),
    action: list[str] | None = Query(default=None, description="按动作码过滤，可重复"),
    outcome: list[str] | None = Query(
        default=None, description="按结果过滤：accepted / rejected，可重复"
    ),
    bag_seq: str | None = Query(default=None, description="只导这一袋序"),
    limit: int = Query(default=5000, ge=1, description="最多导多少行"),
) -> Response:
    store = request.app.state.store
    request_id = request_id_of(request)
    filters = build_filter(
        rack_id=rack_id,
        bed_no=bed_no,
        occurred_from=occurred_from,
        occurred_to=occurred_to,
        categories=tuple(category or ()),
        actions=tuple(action or ()),
        outcomes=tuple(outcome or ()),
        bag_seq=bag_seq,
        limit=limit,
    )
    generated_at = utc_now()
    package = collect_package(store.audit_chain, filters=filters, generated_at=generated_at)
    filename = package.filename(generated_at)
    manifest = package.manifest
    logger.info(
        "审计包导出 request=%s rows=%d scope=%s digest=%s",
        request_id,
        package.count,
        manifest.get("scope_key"),
        manifest.get("chain_digest"),
    )
    return Response(
        content=package.to_csv(),
        media_type="text/csv; charset=utf-8",
        headers={
            # 中文文件名走 RFC 5987 的 filename*：content-disposition 的 filename
            # 用 latin-1，中文放进去在部分浏览器上会变成一串问号。
            "Content-Disposition": f"attachment; filename=audit-package.csv; filename*=UTF-8''{quote(filename)}",
            "X-Audit-Row-Count": str(package.count),
            "X-Audit-Chain-Digest": str(manifest.get("chain_digest") or ""),
            "X-Audit-Scope": _ascii_header(str(manifest.get("scope_key") or "")),
            "X-Audit-Generated-At": generated_at,
            "X-Request-Id": request_id,
        },
    )


@router.get(
    "/audit/manifest",
    response_model=AuditManifestOut,
    summary="审计包首页（manifest，只读）",
    description=(
        "与 `…/audit` 用**同一套过滤条件**取同一批行，只返回包的首页信息：行数、"
        "七类动作分布、覆盖时间窗与链摘要。护理部打印这页签收，内审把摘要抄进底稿。"
    ),
)
async def audit_manifest(
    request: Request,
    rack_id: str | None = Query(default=None, description="只导这一架"),
    bed_no: str | None = Query(default=None, description="只导这一床"),
    occurred_from: str | None = Query(default=None, description="起始时刻（含），ISO8601"),
    occurred_to: str | None = Query(default=None, description="截止时刻（含），ISO8601"),
    category: list[str] | None = Query(default=None, description="按问句类别过滤，可重复"),
    action: list[str] | None = Query(default=None, description="按动作码过滤，可重复"),
    outcome: list[str] | None = Query(default=None, description="按结果过滤，可重复"),
    bag_seq: str | None = Query(default=None, description="只导这一袋序"),
    limit: int = Query(default=5000, ge=1, le=MAX_EXPORT_ROWS, description="最多计多少行"),
) -> AuditManifestOut:
    store = request.app.state.store
    request_id = request_id_of(request)
    filters = build_filter(
        rack_id=rack_id,
        bed_no=bed_no,
        occurred_from=occurred_from,
        occurred_to=occurred_to,
        categories=tuple(category or ()),
        actions=tuple(action or ()),
        outcomes=tuple(outcome or ()),
        bag_seq=bag_seq,
        limit=limit,
    )
    package = collect_package(store.audit_chain, filters=filters, generated_at=utc_now())
    logger.info(
        "审计包首页 request=%s rows=%d scope=%s", request_id, package.count,
        package.manifest.get("scope_key"),
    )
    return AuditManifestOut(**dict(package.manifest), filters=AuditFilterOut(**filters.to_json()))