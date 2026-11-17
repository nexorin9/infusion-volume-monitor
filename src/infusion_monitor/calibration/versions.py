"""标定版本管理：生效区间、版本状态与改版偏差告警。

**为什么同一袋型不能有两个生效版本**：毫升数是拿去决定「要不要换袋」的读数。
两条曲线同时生效，护士长按 A 算出来是 180 ml、按 B 算是 260 ml，谁也说不清算的谁，
最后只能谁都不信。本模块把「同一袋型同时只有一个生效版本」从口头约定变成
**数据结构上的硬约束**——登记新版本时由 :func:`assert_no_overlapping_window`
检查生效区间是否与既有版本相交，相交即拒收（而不是靠调用方记得先停用旧的）。

**生效区间是半开区间 ``[effective_from, effective_to)``**：今天登记的新版本默认
``effective_from = 现在``、旧版本自动收到 ``effective_to = 新版本生效时刻``。
留空 ``effective_to`` 表示「仍然生效」。半开区间的好处是**同一时刻**交接时
新旧两版不会有一帧同时生效（闭区间会在交接点重叠 1 帧）。

**未来生效的版本（预约登记）**：护理部换了一批新袋型，可以提前把标定录进去但
指定「下周一零点起生效」。这时 ``active=0``、状态 ``scheduled``，**当天算出来的
毫升数一点不变**；到期后由 :func:`promote_due` 把它转正。提前换版会让当晚还在
监测的病区凭空多出几百毫升的读数差，那不是修正，是事故。

**改版偏差告警**：改版后历史会话要按新版本重算，但**不是所有重算都该立刻生效**。
新版本与旧版本对同一袋给出的毫升数差得超过 15%（``DEFAULT_REVIEW_THRESHOLD``）
时，说明改的不是小数点、而是标定方法或袋型本身——这种差值很可能是**新版本错了**
而不是「旧版本终于被修正」。所以超阈值的那几条**不静默覆盖**：旧读数原样保留，
会话被标成待复核（``calibration_review_required``），差异逐条写 append-only 的
``calibration_version_alert``，等护理部确认新版本无误后再走复核应用入口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from ..errors import Issue, ValidationRejected
from .models import BagTypeCalibration

__all__ = [
    "DEFAULT_REVIEW_THRESHOLD",
    "VERSION_STATUSES",
    "REVIEW_THRESHOLD_ERROR",
    "VersionWindow",
    "VersionView",
    "DeviationVerdict",
    "OverlappingVersionWindow",
    "parse_timestamp",
    "window_of",
    "classify_version",
    "classify_all",
    "active_at",
    "assert_no_overlapping_window",
    "plan_supersede",
    "promote_due",
    "relative_deviation",
    "judge_deviation",
]

#: 改版后同一袋的毫升读数相对差超过这个比例即**不自动覆盖**，改为告警并要求复核。
DEFAULT_REVIEW_THRESHOLD = 0.15

#: 相对差超过它时的取值也一并记录，方便院方自己调口径而不用改代码。
REVIEW_THRESHOLD_ERROR = 0.25

#: 版本在某一时刻的状态。``scheduled`` 还没生效，``superseded`` 已被新版取代，
#: ``expired`` 是显式设了 ``effective_to`` 且已到期。
VERSION_STATUSES: tuple[str, ...] = ("active", "scheduled", "superseded", "expired")


def parse_timestamp(value: str | None) -> datetime | None:
    """解析 ISO8601 时刻；带时区的按原时区算，naive 的按**院内本地时区**理解。

    登记人写的是「2026-11-01 零点」而不是「2026-11-01T00:00:00Z」——按 UTC 解会把
    生效时刻错开 8 小时，版本在护士下班后才切换，排查起来极其费劲。
    """
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"时间格式不合法：{value!r}（应为 ISO8601，如 2026-11-01T00:00:00+08:00）") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_local_tz())
    return parsed


def _local_tz() -> timezone:
    """本机时区。登记人与服务同机部署，取本机时区即可。"""
    return datetime.now().astimezone().tzinfo or timezone.utc


def utc_now_iso() -> str:
    """当前时刻（ISO8601 带时区），版本默认生效时刻用它。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class VersionWindow:
    """一个版本的生效区间（左闭右开）。"""

    effective_from: str
    effective_to: str | None = None

    def start(self) -> datetime | None:
        return parse_timestamp(self.effective_from)

    def end(self) -> datetime | None:
        return parse_timestamp(self.effective_to)

    def contains(self, at: datetime) -> bool:
        start = self.start()
        end = self.end()
        if start is not None and at < start:
            return False
        if end is not None and at >= end:
            return False
        return True

    def overlaps(self, other: "VersionWindow") -> bool:
        """两个半开区间是否相交。``None`` 端点按 ±∞ 处理。"""
        left = self.start()
        right = self.end()
        other_left = other.start()
        other_right = other.end()
        if right is not None and other_left is not None and right <= other_left:
            return False
        if other_right is not None and left is not None and other_right <= left:
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {"effective_from": self.effective_from, "effective_to": self.effective_to}


@dataclass(frozen=True)
class VersionView:
    """一条标定记录在某一时刻的版本状态（列表与查询端点的返回形态）。"""

    calibration_id: str
    bag_type_code: str
    version: int
    status: str
    window: VersionWindow
    calibrated_by: str
    calibrated_at: str
    usable_capacity_ml: float
    max_relative_error: float
    note: str = ""
    estimation_mode: str = "band"

    def as_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "bag_type_code": self.bag_type_code,
            "version": self.version,
            "status": self.status,
            "effective_from": self.window.effective_from,
            "effective_to": self.window.effective_to,
            "calibrated_by": self.calibrated_by,
            "calibrated_at": self.calibrated_at,
            "usable_capacity_ml": round(self.usable_capacity_ml, 3),
            "max_relative_error": round(self.max_relative_error, 6),
            "estimation_mode": self.estimation_mode,
            "note": self.note,
        }


@dataclass(frozen=True)
class DeviationVerdict:
    """改版后单袋的偏差判定结果。"""

    relative_deviation: float | None
    threshold: float
    review_required: bool
    severity: str  # none / notice / review / critical
    previous_volume_ml: float | None = None
    proposed_volume_ml: float | None = None

    @property
    def auto_apply(self) -> bool:
        """是否可以直接覆盖旧读数（偏差在阈值内）。"""
        return not self.review_required

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_deviation": (
                None if self.relative_deviation is None else round(self.relative_deviation, 6)
            ),
            "threshold": self.threshold,
            "review_required": self.review_required,
            "severity": self.severity,
            "previous_volume_ml": self.previous_volume_ml,
            "proposed_volume_ml": self.proposed_volume_ml,
        }


def window_of(record: BagTypeCalibration) -> VersionWindow:
    """取记录的生效区间（老记录没有区间时用登记时刻兜底）。"""
    return VersionWindow(
        effective_from=getattr(record, "effective_from", None) or record.calibrated_at or "",
        effective_to=getattr(record, "effective_to", None),
    )


def classify_version(record: BagTypeCalibration, at: datetime) -> VersionView:
    """判定一条记录在 ``at`` 时刻的状态。"""
    window = window_of(record)
    start = window.start()
    end = window.end()
    if start is not None and at < start:
        status = "scheduled"
    elif not record.active and end is not None and at >= end:
        status = "expired"
    elif record.active:
        # 库里的 active 说了算：预约版还没到点时，旧版区间虽已按计划收口，
        # 但此刻确实还在用它，报「active」才是护士看到的事实。
        status = "active"
    else:
        status = "superseded"
    return VersionView(
        calibration_id=record.calibration_id,
        bag_type_code=record.bag_type_code,
        version=record.version,
        status=status,
        window=window,
        calibrated_by=record.calibrated_by,
        calibrated_at=record.calibrated_at,
        usable_capacity_ml=record.usable_capacity_ml,
        max_relative_error=record.max_relative_error,
        note=record.note,
        estimation_mode="cavity" if record.cavity is not None else "band",
    )


def classify_all(
    records: Iterable[BagTypeCalibration], at: datetime | None = None
) -> list[VersionView]:
    """把一个袋型的全部版本按时刻排好（版本号升序）。"""
    moment = at or datetime.now(_local_tz())
    return sorted(
        (classify_version(record, moment) for record in records),
        key=lambda view: view.version,
    )


def active_at(
    records: Iterable[BagTypeCalibration], at: datetime | None = None
) -> BagTypeCalibration | None:
    """取 ``at`` 时刻真正生效的那一版；没有则 ``None``（调用方走 ``uncalibrated``）。"""
    moment = at or datetime.now(_local_tz())
    hits = [record for record in records if window_of(record).contains(moment)]
    if not hits:
        return None
    return max(hits, key=lambda record: record.version)


class OverlappingVersionWindow(ValidationRejected):
    """登记的生效区间与既有版本相交——同一袋型不允许两个版本同时生效。

    走 409 而不是 422：请求本身**格式都对**，冲突的是业务事实（这条曲线和另一条
    同时生效），挂接方要改的是生效时刻而不是请求结构。
    """

    def __init__(self, issues: Sequence[Issue]) -> None:
        super().__init__(issues, error_code="CALIBRATION_WINDOW_CONFLICT")

    def as_detail(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": "标定生效区间与既有版本相交；同一袋型不能有两个版本同时生效",
            "issues": [issue.as_dict() for issue in self.issues],
        }


def assert_no_overlapping_window(
    existing: Iterable[BagTypeCalibration],
    window: VersionWindow,
    *,
    bag_type_code: str,
    exclude_calibration_id: str | None = None,
    closing: Mapping[str, Any] | None = None,
) -> None:
    """检查新登记的 ``window`` 是否与既有版本相交；相交即拒收。

    ``closing`` 是**改版交接**通道：``{calibration_id: 即将写入的 effective_to}``。
    交接本身不算冲突——旧版会被收口到新版的生效时刻，半开区间天然不重叠。
    但收口时刻必须**不早于旧版的生效时刻**，否则等于把旧版的历史有效期往回拉长，
    那不是交接，是伪造记录；这种情况照常拒收。

    除此之外的任何相交都是真冲突：要么是两版同时生效，要么是把生效区间
    登记到了历史里（那条记录早已不在用了）。
    """
    closing = dict(closing or {})
    clashes: list[BagTypeCalibration] = []
    for record in existing:
        if exclude_calibration_id and record.calibration_id == exclude_calibration_id:
            continue
        if record.calibration_id in closing:
            effective = window_of(record)
            close_at = parse_timestamp(closing[record.calibration_id])
            if close_at is not None and effective.start() is not None and close_at < effective.start():
                clashes.append(record)  # 回填生效区间：交接时刻早于旧版生效时刻
            continue
        if not window.overlaps(window_of(record)):
            continue
        clashes.append(record)
    if not clashes:
        return
    issues = [
        Issue(
            code="version_window_overlap",
            field="effective_from",
            reason=(
                f"{bag_type_code} 的 v{clash.version}"
                f"（{clash.calibrated_at} 登记）生效区间 "
                f"[{window_of(clash).effective_from}, "
                f"{window_of(clash).effective_to or '至今'}) "
                f"与本次登记的 [{window.effective_from}, "
                f"{window.effective_to or '至今'}) 相交"
            ),
            suggestion=(
                "同一袋型同时只能有一个生效版本。请把本次登记的生效时刻设在 "
                f"v{clash.version} 生效之后（如 2026-11-01T00:00:00+08:00），"
                "或先给该版本补一个 effective_to 明确它的停用时刻。"
            ),
        )
        for clash in clashes
    ]
    raise OverlappingVersionWindow(issues)


def plan_supersede(
    superseded: BagTypeCalibration | None, new_window: VersionWindow
) -> tuple[str | None, str]:
    """算出旧版应当如何收口，返回 ``(旧版 effective_to, 新版状态)``。

    * ``superseded`` 为 ``None``（首次登记）→ 旧版无需收口，新版直接 ``active``；
    * 新版生效时刻在**未来** → 旧版的 ``effective_to`` **事先**收到那个时刻、
      新版 ``scheduled``。事先收口而不是等到交接日才收，是因为区间是**计划**：
      「从 T 起改用 v2」在登记那一刻就已经决定了，v1 从 T 起就该失效。真正到点
      由 :func:`promote_due` 只需翻 ``active`` 标志，不必再动区间；
    * 新版即刻生效 → 旧版同时失活并收口（半开区间不重叠）。
    """
    if superseded is None:
        return None, "active"
    start = new_window.start()
    now = datetime.now(_local_tz())
    if start is not None and start > now:
        return new_window.effective_from, "scheduled"
    return new_window.effective_from, "active"


def promote_due(
    records: Sequence[BagTypeCalibration], at: datetime | None = None
) -> list[VersionView]:
    """找出已经到生效时刻、但仍标着 ``scheduled`` 的版本。

    服务重启或调用查询端点时调用一次即可：到期版本由 ``CalibrationRepository``
    置 ``active=1``、把上一版收口。**不自动重算历史会话**——交接当天的读数变化
    必须走复核闸门（:func:`judge_deviation`），不能由定时任务悄悄改掉。
    """
    moment = at or datetime.now(_local_tz())
    due: list[VersionView] = []
    for record in records:
        window = window_of(record)
        start = window.start()
        if start is None or moment < start:
            continue
        if window.contains(moment) and not record.active:
            due.append(classify_version(record, moment))
    return due


def relative_deviation(previous_ml: float | None, new_ml: float | None) -> float | None:
    """两次读数的相对差。

    分母取**两者较大值**而不是旧值：旧值极小时（袋快见底，报 3 ml）用旧值当分母
    会把「3 → 60」算成 19 倍，凭空造出一条 critical 告警。任一侧缺失或非正数
    返回 ``None``——「本来就没数」不是「差得离谱」。
    """
    if previous_ml is None or new_ml is None:
        return None
    previous = float(previous_ml)
    proposed = float(new_ml)
    if previous < 0 or proposed < 0:
        return None
    scale = max(abs(previous), abs(proposed), 1.0)
    return abs(proposed - previous) / scale


def judge_deviation(
    previous_ml: float | None,
    proposed_ml: float | None,
    *,
    threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> DeviationVerdict:
    """判定一次改版重算是否可以直接覆盖旧读数。

    * 没有旧读数（首次登记后补上毫升数）→ 直接生效，不算偏差；
    * 相对差 ≤ ``threshold`` → 直接覆盖（``severity="none"``）；
    * 超过 ``threshold`` → **不覆盖**，等复核（``review``）；超过
      ``REVIEW_THRESHOLD_ERROR`` 再升一档（``critical``），让偏差特大的版本
      在列表里一眼能被排到最前面。
    """
    deviation = relative_deviation(previous_ml, proposed_ml)
    if deviation is None:
        return DeviationVerdict(
            relative_deviation=None,
            threshold=threshold,
            review_required=False,
            severity="none",
            previous_volume_ml=previous_ml,
            proposed_volume_ml=proposed_ml,
        )
    if deviation > REVIEW_THRESHOLD_ERROR:
        severity = "critical"
    elif deviation > threshold:
        severity = "review"
    else:
        severity = "notice" if deviation > 0.0 else "none"
    return DeviationVerdict(
        relative_deviation=deviation,
        threshold=threshold,
        review_required=severity in ("review", "critical"),
        severity=severity,
        previous_volume_ml=previous_ml,
        proposed_volume_ml=proposed_ml,
    )