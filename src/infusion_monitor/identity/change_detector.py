"""换液点名判定引擎：把「这袋该换了」变成护士**肯信**的一条提醒。

**误报一次就耗尽信任。** 这是本模块存在的唯一理由。一个每轮巡视都把同一袋点名的
系统，护士第三次看到就会连内容一起跳过——那时它和没有提醒没有区别。所以本模块在
:func:`~infusion_monitor.identity.state_machine.evaluate_change` 的**双条件**之上
（连续 N 帧 + 余量 margin）再加三道闸：

================ ==============================================================
闸门              挡住的是
================ ==============================================================
**滞回带**        面积比在阈值附近来回抖时判不了——阈值上下的反光与滴斗晃动，
                 每轮都重新触发一次点名。它按轮次耐心消解，而不是当噪声丢掉。
**每袋最小间隔**  同一袋在 15 分钟内只点一次名。间隔用**审计链里的上一次点名
                 时刻**算，进程重启也不清零（持久闸门）。
**每轮点名上限**  一轮最多点 3 袋，超出的按 ETA 升序截断，**并把被截断的记下来**。
                 截断不是丢弃：护士长要能回答「为什么 C03 没被点名」。
================ ==============================================================

**本模块只产出建议**：它返回一堆 ``PickupProposal``，由调用方决定要不要推进状态。
未确认的 ``needs_change`` 不出站、不迁移 ``replaced``——这条边界在
``state_machine`` 里，本模块只负责**少点名、点名有据、点名可解释**。

三道闸都是**可解释**的：每袋都给出一条 ``suppress_reason`` 码和一句人话 ``detail``，
护士长与信息科复盘时看到的是「这袋为什么这轮没被点名」，而不是一片沉默。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .state_machine import ChangeDecision, ChangeParams, evaluate_change

__all__ = [
    "AuditPickupHistory",
    "BagGate",
    "ChangeDetector",
    "DetectorParams",
    "InMemoryPickupHistory",
    "PickupHistory",
    "PickupProposal",
    "PICKUP_ACTIONS",
    "RoundDetection",
    "SUPPRESS_REASONS",
    "bag_key",
    "band_bounds",
    "classify_ratio",
    "detect_round",
    "parse_ts",
]

#: 点名审计动作码。``named`` 是持久闸门（最小间隔）的读数来源。
PICKUP_ACTIONS = ("pickup.named", "pickup.suppressed", "pickup.truncated")

#: 抑制原因码。**不点名也是一条结论**，必须能说清是哪一道闸挡的。
SUPPRESS_REASONS = (
    "state_replaced",        # 已确认换过（终态），不再参与点名
    "no_observation",        # 这一段没有可信液面观测
    "no_trigger",            # 双条件未满足（连续帧不够 / 余量还够）
    "hysteresis_hold",       # 液面在滞回带内反复，判不了，等下一轮
    "hysteresis_resolved",   # 带内耐心耗尽，按「确已跌破」收口
    "latched",               # 这袋已经点名过、液位未回升，不重复点名
    "interval_not_elapsed",  # 距上一次点名不足最小间隔（读审计链，跨重启有效）
    "round_cap_exceeded",    # 本轮点名数已到上限，被截断（记入 truncated）
)


def parse_ts(value: str | None) -> datetime | None:
    """宽松解析 ISO8601 时间串；解析不了返回 ``None``（**不当成当前时刻**）。"""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        # 库里的 captured_at 是带时区的；这里兜底按 UTC 处理，避免比较时报错
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def bag_key(session: Mapping[str, Any]) -> str:
    """一袋的业务身份：``输注单号|袋序``。

    刻意**不含** ``rack_id`` / ``slot_index``：输液架挪个位置、槽位重排，本轮还是
    同一袋；而换袋后对端给的新 ``bag_seq`` 必然算出新键——这正是「点名过就别再点」
    该有的边界。用 bag_seq 单键会漏掉「同一输液单上第 2 袋换成了第 1 袋」这种情况。
    """
    return f"{session.get('infusion_order_no') or ''}|{session.get('bag_seq') or ''}"


@dataclass(frozen=True)
class DetectorParams:
    """点名判定引擎的门限（判定本身沿用 :class:`ChangeParams` 的双条件）。"""

    #: 双条件门限：连续 N 帧低于阈值 + 余量 margin 充足
    change: ChangeParams = field(default_factory=ChangeParams)
    #: 滞回带宽度（面积比）。阈值 ±半带宽内视为「判不了」。
    #: 取 0.03 的依据：反光与滴斗晃动在几何核上给出的读数抖动实测在 0.012 量级，
    #: 带宽取其两倍余量；而液面每帧的自然下降约 0.003–0.008，落在带内约 4–10 轮，
    #: 耐心轮次取 3 表示「真在排空时不会等太久、纯抖动时也不会被点」。
    hysteresis_ratio: float = 0.03
    #: 滞回带内的耐心轮次：连续这么多轮都在带内，就按「确已跌破」收口
    band_patience_rounds: int = 3
    #: 每袋两次点名的最小间隔（分钟）
    min_pickup_interval_minutes: int = 15
    #: 每轮点名上限（袋）
    max_pickups_per_round: int = 3


def band_bounds(threshold: float, width: float) -> tuple[float, float]:
    """滞回带上下沿：``(带下沿, 带上沿)``。

    带宽是**对称**的：阈值 ±width/2。带上沿高于阈值的那一半不是为了「迟一点才认输」，
    而是给回读留出余量——液面真的回升到阈值以上才算解除锁定，停在阈值上方一丁点
    就解锁等于没锁。
    """
    half = max(width, 0.0) / 2.0
    return threshold - half, threshold + half


def classify_ratio(ratio: float, threshold: float, width: float) -> str:
    """把一次读数判成 ``below`` / ``in_band`` / ``above``。"""
    low, high = band_bounds(threshold, width)
    if ratio <= low:
        return "below"
    if ratio >= high:
        return "above"
    return "in_band"


# ── 持久闸门：上一次点名时刻 ───────────────────────────────────────────────


class PickupHistory(Protocol):
    """「这袋上一次被点名是什么时候」的读数面。"""

    def last_pickup_at(self, key: str) -> datetime | None:  # pragma: no cover - 协议
        ...

    def record_pickup(self, key: str, at: datetime, *, meta: Mapping[str, Any] | None = None) -> None:
        ...  # pragma: no cover - 协议


class InMemoryPickupHistory:
    """进程内实现：单测与「本轮刚点名、下一轮就要判间隔」的场景。"""

    def __init__(self) -> None:
        self._stamps: dict[str, datetime] = {}

    def last_pickup_at(self, key: str) -> datetime | None:
        return self._stamps.get(key)

    def record_pickup(
        self, key: str, at: datetime, *, meta: Mapping[str, Any] | None = None
    ) -> None:
        self._stamps[key] = at


class AuditPickupHistory:
    """审计链实现（**生产用**）：重启、换进程都不丢最小间隔。

    点名动作 ``pickup.named`` 是一行 append-only 审计，带 ``bag_key`` 列；
    读最近一次命名时刻只要一次索引查询，不用扫全表。
    """

    def __init__(self, conn: sqlite3.Connection, *, audit: Any | None = None) -> None:
        self._conn = conn
        self._audit = audit

    def last_pickup_at(self, key: str) -> datetime | None:
        row = self._conn.execute(
            "SELECT occurred_at FROM audit_log "
            "WHERE action = 'pickup.named' AND bag_key = ? "
            "ORDER BY audit_id DESC LIMIT 1",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return parse_ts(row["occurred_at"])

    def record_pickup(
        self, key: str, at: datetime, *, meta: Mapping[str, Any] | None = None
    ) -> None:
        if self._audit is None:
            return
        payload = dict(meta or {})
        payload["bag_key"] = key
        self._audit.append(
            request_id=str(payload.get("request_id") or "service:change-detector"),
            actor="service:change-detector",
            action="pickup.named",
            target=str(payload.get("session_id") or key),
            outcome="accepted",
            bag_seq=payload.get("bag_seq"),
            bag_key=key,
            detail=payload,
            occurred_at=at.isoformat(timespec="seconds"),
        )


# ── 结论结构 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PickupProposal:
    """一条换液点名建议（**未确认**：护士点「已换」之后才会出站）。"""

    bag_seq: str
    session_id: str | None
    rack_id: str | None
    bed_no: str | None
    bag_key: str
    trigger_reason: str
    eta_minutes: int | None
    fill_ratio: float | None
    consecutive_frames: int
    margin_kind: str
    rank: int | None = None
    selected: bool = False
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "bag_seq": self.bag_seq,
            "session_id": self.session_id,
            "bed_no": self.bed_no,
            "trigger_reason": self.trigger_reason,
            "eta_minutes": self.eta_minutes,
            "fill_ratio": self.fill_ratio,
            "consecutive_frames": self.consecutive_frames,
            "margin_kind": self.margin_kind,
            "rank": self.rank,
            "selected": self.selected,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class BagGate:
    """一袋在某一轮的闸门结论：点名了没、为什么。"""

    bag_seq: str
    session_id: str | None
    bag_key: str
    picked: bool
    suppress_reason: str | None
    detail: str
    decisive_below: bool = False
    band_dwell_rounds: int = 0
    latched: bool = False
    since_minutes: float | None = None
    eta_minutes: int | None = None
    fill_ratio: float | None = None
    trigger_reason: str | None = None
    proposal: PickupProposal | None = None
    decision: ChangeDecision | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "bag_seq": self.bag_seq,
            "session_id": self.session_id,
            "bag_key": self.bag_key,
            "picked": self.picked,
            "suppress_reason": self.suppress_reason,
            "detail": self.detail,
            "decisive_below": self.decisive_below,
            "band_dwell_rounds": self.band_dwell_rounds,
            "latched": self.latched,
            "since_minutes": self.since_minutes,
            "eta_minutes": self.eta_minutes,
            "fill_ratio": self.fill_ratio,
            "trigger_reason": self.trigger_reason,
            "proposal": self.proposal.to_json() if self.proposal else None,
        }


@dataclass(frozen=True)
class RoundDetection:
    """一轮巡视的点名结果。``pickups`` 已按上限截断，``truncated`` 是被截断的那些。"""

    round_key: str
    decided_at: str
    pickups: tuple[PickupProposal, ...]
    truncated: tuple[PickupProposal, ...]
    gates: tuple[BagGate, ...]
    params: DetectorParams

    @property
    def suppressed(self) -> tuple[BagGate, ...]:
        return tuple(gate for gate in self.gates if not gate.picked)

    def gate_for(self, bag_seq: str) -> BagGate | None:
        for gate in self.gates:
            if gate.bag_seq == bag_seq:
                return gate
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "round_key": self.round_key,
            "decided_at": self.decided_at,
            "pickups": [item.to_json() for item in self.pickups],
            "truncated": [item.to_json() for item in self.truncated],
            "suppressed": [gate.to_json() for gate in self.suppressed],
            "params": {
                "change_threshold_ratio": self.params.change.change_threshold_ratio,
                "n_frames": self.params.change.n_frames,
                "hysteresis_ratio": self.params.hysteresis_ratio,
                "band_patience_rounds": self.params.band_patience_rounds,
                "min_pickup_interval_minutes": self.params.min_pickup_interval_minutes,
                "max_pickups_per_round": self.params.max_pickups_per_round,
            },
        }


# ── 引擎 ──────────────────────────────────────────────────────────────────


@dataclass
class _Latch:
    """某一袋的滞回状态：带内停留轮数 + 是否已点名且尚未解除。"""

    dwell: int = 0
    latched: bool = False
    last_side: str = "above"


class ChangeDetector:
    """一轮一轮地点名。**纯判定**，写库与审计由调用方负责。

    状态分两层，刻意分开：

    * **持久闸门**（最小间隔）落在 :class:`PickupHistory`，生产实现读审计链——
      进程重启后不会因为内存清空而重新点一遍名；
    * **软闸门**（滞回锁定）留在本对象内存里：它依赖同一进程内连续几轮的曲线，
      重启后重新积累即可，不值得为它建表。
    """

    def __init__(
        self,
        history: PickupHistory | None = None,
        params: DetectorParams | None = None,
    ) -> None:
        self.params = params or DetectorParams()
        self.history: PickupHistory = history or InMemoryPickupHistory()
        self._latches: dict[str, _Latch] = {}

    # ── 单袋闸门 ────────────────────────────────────────────────────────

    def _gate(
        self, session: Mapping[str, Any], *, now: datetime
    ) -> BagGate:
        p = self.params
        seq = str(session.get("bag_seq") or "")
        key = bag_key(session)
        latch = self._latches.setdefault(key, _Latch())
        values = [
            float(item)
            for item in (session.get("area_ratio_series") or [])
            if item is not None
        ]
        latest = values[-1] if values else None
        threshold = p.change.change_threshold_ratio

        decision = evaluate_change(session, p.change, now=now.isoformat(timespec="seconds"))

        def _hold(reason: str, detail: str, **extra: Any) -> BagGate:
            return BagGate(
                bag_seq=seq,
                session_id=session.get("session_id"),
                bag_key=key,
                picked=False,
                suppress_reason=reason,
                detail=detail,
                fill_ratio=latest,
                eta_minutes=_eta_of(session),
                trigger_reason=None,
                decision=decision,
                **extra,
            )

        if str(session.get("state") or "monitoring") == "replaced":
            latch.latched = False
            latch.dwell = 0
            return _hold("state_replaced", "这袋已确认换过，终态不再参与点名判定")

        if not values:
            latch.dwell = 0
            return _hold("no_observation", "这一段没有可信的液面观测，点名没有依据")

        low, high = band_bounds(threshold, p.hysteresis_ratio)
        side = classify_ratio(latest, threshold, p.hysteresis_ratio)

        # 滞回带的消解：连续读数始终落在带内时，耐心轮次耗尽就按「确已跌破」收口。
        # 不耗尽就按原阈值判——抖动是成片出现的，一两帧回到带上不代表袋子又满了。
        if side == "in_band":
            latch.dwell += 1
        else:
            latch.dwell = 0
        latch.last_side = side

        decisive_streak = _trailing_below(values, low)
        decided = decision.changed and decision.trigger_reason is not None

        if not decided:
            # 没有点名诉求：只有液面确实回到带上才解除锁定。
            if latch.latched and latest is not None and latest >= high:
                latch.latched = False
            return _hold(
                "no_trigger",
                decision.reason_detail or "双条件未满足，不点名",
                decisive_below=False,
                band_dwell_rounds=latch.dwell,
                latched=latch.latched,
            )

        if decisive_streak < p.change.n_frames:
            if latch.dwell >= p.band_patience_rounds:
                # 带内磨够了轮次：认定它确实在排空，只是液位卡在阈值附近读不准
                latch.dwell = 0
                reason = "hysteresis_resolved"
                detail = (
                    f"液面连续 {p.band_patience_rounds} 轮停在滞回带 "
                    f"[{low:.3f}, {high:.3f}] 内，按已跌破收口点名"
                )
            else:
                return _hold(
                    "hysteresis_hold",
                    f"液面在滞回带 [{low:.3f}, {high:.3f}] 内反复"
                    f"（已停留 {latch.dwell}/{p.band_patience_rounds} 轮），判不了，等下一轮",
                    decisive_below=False,
                    band_dwell_rounds=latch.dwell,
                    latched=latch.latched,
                )
        else:
            reason = None
            detail = decision.reason_detail

        if latch.latched:
            return _hold(
                "latched",
                "这袋已点名过、液位未回升到滞回带上沿，不重复点名",
                decisive_below=True,
                band_dwell_rounds=0,
                latched=True,
            )

        last = self.history.last_pickup_at(key)
        since = (now - last).total_seconds() / 60.0 if last is not None else None
        if last is not None and since is not None and since < p.min_pickup_interval_minutes:
            return _hold(
                "interval_not_elapsed",
                f"距上一次点名只过了 {since:.1f} 分钟，少于最小间隔 "
                f"{p.min_pickup_interval_minutes} 分钟",
                decisive_below=True,
                since_minutes=round(since, 2),
                latched=False,
            )

        proposal = PickupProposal(
            bag_seq=seq,
            session_id=session.get("session_id"),
            rack_id=session.get("rack_id"),
            bed_no=session.get("bed_no"),
            bag_key=key,
            trigger_reason=decision.trigger_reason or "",
            eta_minutes=_eta_of(session),
            fill_ratio=latest,
            consecutive_frames=decision.consecutive_frames,
            margin_kind=decision.margin_kind,
            detail=f"{reason + '：' if reason else ''}{detail}",
        )
        # 注意这里**不**置 latch：这一袋还可能被「每轮点名上限」截断，
        # 截断的袋下一轮还要重新参与，先锁上等于把没点成的名记成点过了。
        return BagGate(
            bag_seq=seq,
            session_id=session.get("session_id"),
            bag_key=key,
            picked=True,
            suppress_reason=None,
            detail=proposal.detail,
            decisive_below=True,
            band_dwell_rounds=0,
            latched=False,
            since_minutes=round(since, 2) if since is not None else None,
            eta_minutes=proposal.eta_minutes,
            fill_ratio=latest,
            trigger_reason=proposal.trigger_reason,
            proposal=proposal,
            decision=decision,
        )

    # ── 整轮 ────────────────────────────────────────────────────────────

    def detect_round(
        self,
        sessions: Iterable[Mapping[str, Any]],
        *,
        round_key: str = "",
        now: str | datetime | None = None,
        commit: bool = True,
    ) -> RoundDetection:
        """判定一整轮巡视里所有袋的点名结论（含上限截断）。

        ``commit=True`` 时把通过的点名写进 :class:`PickupHistory`（最小间隔闸门
        从这一刻起生效）；只做演练/复盘时传 ``False``。
        """
        stamp = _coerce_now(now)
        items = list(sessions)
        gates = [self._gate(session, now=stamp) for session in items]

        # 排序口径：**ETA 升序**（最急的先点），同 ETA 按床号、再按袋序，
        # 保证同一轮里点名顺序可复现——护士看到的顺序不该随字典序跳。
        candidates = sorted(
            (gate.proposal for gate in gates if gate.picked and gate.proposal is not None),
            key=lambda item: (
                item.eta_minutes is None,
                item.eta_minutes if item.eta_minutes is not None else 0,
                item.bed_no or "",
                item.bag_seq,
            ),
        )
        cap = max(self.params.max_pickups_per_round, 0)
        picked = [
            _with_rank(item, rank + 1, True) for rank, item in enumerate(candidates[:cap])
        ]
        dropped = [
            _with_rank(item, position + 1, False)
            for position, item in enumerate(candidates[cap:], start=cap)
        ]

        picked_by_seq = {item.bag_seq: item for item in picked}
        dropped_by_seq = {item.bag_seq: item for item in dropped}
        final_gates: list[BagGate] = []
        for gate in gates:
            if gate.bag_seq in dropped_by_seq:
                item = dropped_by_seq[gate.bag_seq]
                # 截断不是丢弃：理由、排名与 ETA 都留着，下一轮直接重新参与
                final_gates.append(
                    replace(
                        gate,
                        picked=False,
                        suppress_reason="round_cap_exceeded",
                        proposal=item,
                        detail=(
                            f"本轮已点名 {len(picked)} 袋（上限 {cap}），"
                            f"{gate.bag_seq} 按 ETA 升序排在第 {item.rank} 位被截断；"
                            "下一轮会重新参与"
                        ),
                    )
                )
            elif gate.bag_seq in picked_by_seq:
                item = picked_by_seq[gate.bag_seq]
                final_gates.append(replace(gate, proposal=item, detail=item.detail))
            else:
                final_gates.append(gate)

        for item in picked:
            # 真正点了名的袋才进锁定：下一轮液位没回升就不再重复点名
            self._latches.setdefault(item.bag_key, _Latch()).latched = True
        if commit:
            for item in picked:
                self.history.record_pickup(
                    item.bag_key,
                    stamp,
                    meta={
                        "session_id": item.session_id,
                        "bag_seq": item.bag_seq,
                        "round_key": round_key,
                        "trigger_reason": item.trigger_reason,
                        "eta_minutes": item.eta_minutes,
                        "rank": item.rank,
                    },
                )

        return RoundDetection(
            round_key=round_key,
            decided_at=stamp.isoformat(timespec="seconds"),
            pickups=tuple(picked),
            truncated=tuple(dropped),
            gates=tuple(final_gates),
            params=self.params,
        )

    def snapshot(self) -> dict[str, Any]:
        """滞回状态快照（供 ``/summary`` 与排障使用）。"""
        return {
            key: {"dwell": latch.dwell, "latched": latch.latched, "last_side": latch.last_side}
            for key, latch in self._latches.items()
        }


# ── 便捷入口 ──────────────────────────────────────────────────────────────

_DEFAULT_DETECTOR = ChangeDetector()


def detect_round(
    sessions: Iterable[Mapping[str, Any]],
    *,
    detector: ChangeDetector | None = None,
    round_key: str = "",
    now: str | datetime | None = None,
    commit: bool = True,
) -> RoundDetection:
    """模块级入口：默认用一个进程内单例（测试与脚本够用；生产请注入带审计的实例）。"""
    return (detector or _DEFAULT_DETECTOR).detect_round(
        sessions, round_key=round_key, now=now, commit=commit
    )


# ── 小工具 ────────────────────────────────────────────────────────────────


def _coerce_now(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = parse_ts(value)
    if parsed is not None:
        return parsed
    return datetime.now(timezone.utc)


def _trailing_below(values: Sequence[float], low: float) -> int:
    """末尾连续**确已跌破**带下沿的读数个数。"""
    streak = 0
    for value in reversed(values):
        if value > low:
            break
        streak += 1
    return streak


def _eta_of(session: Mapping[str, Any]) -> int | None:
    eta = session.get("eta_minutes")
    if eta is None:
        return None
    try:
        return int(eta)
    except (TypeError, ValueError):
        return None


def _with_rank(item: PickupProposal, rank: int | None, selected: bool) -> PickupProposal:
    return replace(item, rank=rank, selected=selected)