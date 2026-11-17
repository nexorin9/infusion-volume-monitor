"""审计包导出：护理部与内审留档用的 CSV + manifest。

**一份审计包要能被两个完全不同的人用**：护理部要在 Excel 里筛「这一架当天换了几次液」，
内审要能当场判断「这段记录有没有被人动过」。所以导出分两件东西：

1. **CSV 明细**（:func:`to_csv`）——一行一条审计，列固定顺序，中文表头，UTF-8 BOM +
   CRLF 行尾。这两样都不是审美选择：院内留档的机器几乎全是 Windows 上的 Excel，
   没有 BOM 的中文表头在 Excel 里是乱码，没有 CRLF 的话从 WPS 另存后再被 Python
   读回来会多出 ``\\r``，内审核对行数时对不上。
2. **manifest**（:func:`build_manifest`）——包的首页：导了多少行、覆盖什么时间窗、
   七类动作各多少条、以及一条 :meth:`~store.audit_chain.AuditChain.chain_digest`
   链摘要。护理部打印这页签收，内审留档时记下摘要；将来任何人重算一遍摘要，
   对得上就说明这段审计没被动过（哪怕中间删掉一行也会变）。

**导出的口径纪律**：

- 只导**已发生的动作**，不导「本来该发生什么」。`pickup.suppressed` 之所以在包里，
  是因为它记录了「判了但被闸门挡下」这个**发生过**的判定，不是待办清单；
- 空结果**照样出包**（只有表头 + manifest 计数 0）。飞检问「昨天有没有异常换液」时，
  一份说「0 条」的空包是有力的证据，一个 404 让人以为系统坏了或数据丢了；
- **不是所有动作都属于某一架某一床**。袋型标定是**病区级**记录：一个 ``TYPE-A``
  的标定版本对全病区每一架都生效，硬给它安一个架号/床号等于替它编一个不存在的
  归属。所以标定类审计行的 ``rack_id`` / ``bed_no`` 恒为空，**按架或按床导出时不会
  出现在包里**——按「这一架当天换了几次液」筛出来的清单里混进一条全病区的标定是
  噪声；要查标定留痕请用不带架/床过滤的导出。
- ``detail`` 列保留紧凑 JSON 不做展平：护理部看 `evidence_note` 这种人话字段，
  内审按 ``payload_hash`` 核链，两边要的东西不一样，硬压成一列谁都看不清。
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..store.audit_chain import (
    AUDIT_CATEGORIES,
    AUDIT_EXPORT_COLUMNS,
    AuditChain,
    category_label,
    category_of,
)

__all__ = [
    "AUDIT_CSV_COLUMNS",
    "AuditFilter",
    "AuditPackage",
    "build_manifest",
    "collect_package",
    "csv_filename",
    "to_csv",
]

#: CSV 列 = (列名, 中文表头)。顺序即文件里的列序，**只增不改**：留档的 CSV 一旦
#: 换了列序，上一版导出的解析脚本会静默读错列而不是报错，那比报错更难发现。
#: ``detail`` / ``payload_hash`` 留给内审核链与复现，表头写清机器可读。
AUDIT_CSV_COLUMNS: tuple[tuple[str, str], ...] = (
    ("audit_id", "审计序号"),
    ("occurred_at", "发生时刻"),
    ("actor", "操作人"),
    ("category", "动作类别"),
    ("action", "动作码"),
    ("outcome", "结果"),
    ("target", "对象"),
    ("rack_id", "输液架"),
    ("bed_no", "床号"),
    ("bag_seq", "袋序"),
    ("bag_type_code", "袋型"),
    ("bag_key", "输注单号|袋序"),
    ("from_state", "原状态"),
    ("to_state", "新状态"),
    ("request_id", "请求号"),
    ("payload_hash", "载荷摘要"),
    ("detail", "明细(JSON)"),
)


@dataclass(frozen=True)
class AuditFilter:
    """导出条件。原样进 :meth:`AuditChain.query`，也原样回显进 manifest。"""

    rack_id: str | None = None
    bed_no: str | None = None
    occurred_from: str | None = None
    occurred_to: str | None = None
    actions: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    bag_seq: str | None = None
    limit: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "rack_id": self.rack_id,
            "bed_no": self.bed_no,
            "occurred_from": self.occurred_from,
            "occurred_to": self.occurred_to,
            "actions": list(self.actions),
            "outcomes": list(self.outcomes),
            "bag_seq": self.bag_seq,
            "limit": self.limit,
        }

    def describe_zh(self) -> str:
        """一句人话，写在 manifest 里给留档的人看（不是给机器解析的）。"""
        parts: list[str] = []
        if self.rack_id:
            parts.append(f"输液架 {self.rack_id}")
        if self.bed_no:
            parts.append(f"床号 {self.bed_no}")
        if self.occurred_from or self.occurred_to:
            parts.append(f"时间窗 {self.occurred_from or '起始'} ~ {self.occurred_to or '至今'}")
        if self.actions:
            parts.append(f"动作 {'、'.join(self.actions)}")
        if self.outcomes:
            parts.append(f"结果 {'、'.join(self.outcomes)}")
        if self.bag_seq:
            parts.append(f"袋序 {self.bag_seq}")
        if self.limit:
            parts.append(f"最多 {self.limit} 行")
        return "；".join(parts) or "全量（无过滤条件）"


@dataclass(frozen=True)
class AuditPackage:
    """一份审计包：明细行 + 包的首页信息。"""

    rows: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]

    @property
    def count(self) -> int:
        return len(self.rows)

    def to_csv(self) -> str:
        return to_csv(self.rows)

    def filename(self, generated_at: str) -> str:
        return csv_filename(generated_at=generated_at, scope=self.manifest.get("scope_key"))


def _cell(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    return str(value)


def csv_filename(*, generated_at: str, scope: str | None = None) -> str:
    """文件名：``审计包_<范围>_<日期>.csv``。中文名是刻意的——护理部归档时按名索骥。

    日期取生成日（``YYYY-MM-DD``），范围取架号 / 床号；都没有就写 ``全病区``。
    同一份包导出两次得到同名文件，Excel 会提示覆盖而不是生成 ``审计包(1).csv``，
    避免留档目录里躺着两份同名的不同内容。
    """
    day = (generated_at or "")[:10].replace("-", "") or "00000000"
    who = scope or "全病区"
    safe = "".join(ch for ch in who if ch.isalnum() or ch in "-_·（）()") or "全病区"
    return f"审计包_{safe}_{day}.csv"


def to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """明细行 → CSV 文本（带 UTF-8 BOM，CRLF 行尾）。

    ``csv.writer`` 的换行参数必须是 ``\\r\\n`` 的字面值：Windows 上 Excel 靠它认
    行尾，Linux 上 Python 的 ``open(..., newline="")`` 也靠它避免读到 ``\\r\\n`` 之外
    的怪东西。``lineterminator`` 显式写死，不跟平台默认值走。
    """
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow([header for _, header in AUDIT_CSV_COLUMNS])
    for row in rows:
        category = row.get("category") or category_of(str(row.get("action") or ""))
        cells: list[str] = []
        for key, _header in AUDIT_CSV_COLUMNS:
            if key == "category":
                cells.append(category_label(str(category)))
            elif key == "detail":
                detail = row.get("detail")
                cells.append(
                    json.dumps(detail, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    if isinstance(detail, Mapping)
                    else str(detail or "")
                )
            else:
                cells.append(_cell(row, key))
        writer.writerow(cells)
    # BOM 由这里显式写死：Excel 靠它认出 UTF-8，缺了中文表头在 Windows 上是乱码。
    return "\ufeff" + buffer.getvalue()


def build_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    filters: AuditFilter,
    generated_at: str,
) -> dict[str, Any]:
    """包的首页：范围、行数、七类动作分布、链摘要。

    ``chain_digest`` 与 :mod:`store.audit_export` 的 CSV 用的是**同一批行、同一个
    顺序**，所以留档人拿 CSV 重算一遍摘要应当得到同一个值——这一条由
    ``tests/test_audit_chain.py::test_manifest_digest_matches_the_exported_csv`` 守住。
    """
    by_category: dict[str, int] = {name: 0 for name in AUDIT_CATEGORIES}
    by_category["other"] = 0
    by_action: dict[str, int] = {}
    for row in rows:
        category = str(row.get("category") or category_of(str(row.get("action") or "")))
        by_category[category] = by_category.get(category, 0) + 1
        action = str(row.get("action") or "")
        by_action[action] = by_action.get(action, 0) + 1
    stamps = [str(row.get("occurred_at") or "") for row in rows]
    scope = filters.rack_id or filters.bed_no or "全病区"
    return {
        "generated_at": generated_at,
        "scope_zh": filters.describe_zh(),
        "scope_key": scope,
        "row_count": len(rows),
        "window": {
            "from": min(stamps) if stamps else None,
            "to": max(stamps) if stamps else None,
        },
        "by_category": by_category,
        "category_labels_zh": {name: category_label(name) for name in by_category},
        "by_action": dict(sorted(by_action.items())),
        "chain_digest": AuditChain.chain_digest(rows),
        "digest_algorithm": "rolling-sha256(occurred_at,audit_id) 前 32 位",
        "note_zh": (
            "本包为只追加审计链的导出：行序按发生时刻单调。"
            "chain_digest 可由本包明细重算；对不上说明这段审计被改动过。"
        ),
    }


def collect_package(
    audit: AuditChain, *, filters: AuditFilter, generated_at: str
) -> AuditPackage:
    """按过滤条件取行并组装成包。这是导出唯一入口——CSV 与 manifest 必须同源。"""
    rows = audit.query(
        rack_id=filters.rack_id,
        bed_no=filters.bed_no,
        occurred_from=filters.occurred_from,
        occurred_to=filters.occurred_to,
        actions=filters.actions or None,
        outcomes=filters.outcomes or None,
        bag_seq=filters.bag_seq,
        limit=filters.limit,
    )
    return AuditPackage(
        rows=tuple(rows),
        manifest=build_manifest(rows, filters=filters, generated_at=generated_at),
    )