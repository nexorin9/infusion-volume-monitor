"""换液点名判定引擎的五组用例：连续帧、单帧毛刺、滞回带、间隔与上限、端到端无写回。

**误报一次就耗尽护士的信任**——这组用例守的不是「能不能点出名」，而是
「不该点的时候有没有真的忍住」。因此断言的重点大量落在**没点名**的那一侧：
抑制码、人话理由、被截断项是否留痕。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from infusion_monitor.identity.change_detector import (
    SUPPRESS_REASONS,
    AuditPickupHistory,
    ChangeDetector,
    DetectorParams,
    InMemoryPickupHistory,
    bag_key,
    band_bounds,
    classify_ratio,
)
from infusion_monitor.identity.state_machine import ChangeParams
from infusion_monitor.store.audit_chain import AuditChain
from infusion_monitor.store.db import connect
from infusion_monitor.store.migrations import apply_migrations

BASE = datetime(2026, 10, 22, 9, 41, tzinfo=timezone.utc)


def make_session(
    bag_seq: str = "B01",
    *,
    ratios: list[float] | None = None,
    eta_minutes: int | None = 8,
    bed_no: str = "12",
    order: str = "EXAMPLE-ORDER-0001",
    state: str = "monitoring",
    fill_ratio: float | None = None,
    volume_ml: float | None = None,
) -> dict:
    """造一条会话行。默认给一条**确已跌破**的曲线（0.12→0.08，连在阈值下沿之下，
    又高于近空线 0.06，所以触发原因是 ETA 而不是近空）。"""
    series = ratios if ratios is not None else [0.12, 0.11, 0.10, 0.09, 0.08]
    return {
        "session_id": f"sess-{order}-{bag_seq}",
        "rack_id": "R-EXAMPLE-07",
        "bed_no": bed_no,
        "infusion_order_no": order,
        "shift_key": "2026-10-22-AM",
        "bag_seq": bag_seq,
        "area_ratio_series": series,
        "eta_minutes": eta_minutes,
        "estimated_volume_ml": volume_ml,
        "state": state,
        "pickup_suppressed": False,
        "suppression_reason": None,
        "fill_ratio": fill_ratio if fill_ratio is not None else (series[-1] if series else None),
    }


def detector(**kwargs) -> ChangeDetector:
    params = DetectorParams(
        change=ChangeParams(),
        hysteresis_ratio=kwargs.pop("hysteresis_ratio", 0.03),
        band_patience_rounds=kwargs.pop("band_patience_rounds", 3),
        min_pickup_interval_minutes=kwargs.pop("min_pickup_interval_minutes", 15),
        max_pickups_per_round=kwargs.pop("max_pickups_per_round", 3),
    )
    history = kwargs.pop("history", None) or InMemoryPickupHistory()
    return ChangeDetector(history=history, params=params)


# ── 1. 连续帧触发 ──────────────────────────────────────────────────────────


class TestConsecutiveFrames:
    def test_consecutive_low_frames_with_margin_produces_named_pickup(self):
        det = detector()
        result = det.detect_round([make_session()], round_key="2026-10-22-AM", now=BASE)
        assert len(result.pickups) == 1
        proposal = result.pickups[0]
        assert proposal.bag_seq == "B01"
        assert proposal.trigger_reason == "eta_within_window"
        assert proposal.consecutive_frames == 5
        assert proposal.rank == 1
        assert proposal.selected is True
        assert result.gate_for("B01").decisive_below is True

    def test_every_named_pickup_carries_a_trigger_reason(self):
        """点名必带 trigger_reason——护士端按码给不同文案，缺了就没法显示。"""
        det = detector()
        result = det.detect_round(
            [
                make_session("B01", ratios=[0.09] * 5, eta_minutes=5),
                make_session("B02", ratios=[0.09] * 5, eta_minutes=200, volume_ml=12.0, bed_no="13"),
                make_session("B03", ratios=[0.09] * 5, eta_minutes=None, volume_ml=8.0, bed_no="14"),
            ],
            now=BASE,
        )
        reasons = {item.bag_seq: item.trigger_reason for item in result.pickups}
        assert reasons == {
            "B01": "eta_within_window",
            "B02": "volume_below_margin",
            "B03": "volume_below_margin",
        }
        assert all(item.trigger_reason for item in result.pickups)

    def test_near_empty_is_named_without_calibration_gate(self):
        """近空是直接观测：液面 ≤6% 时即使毫升换算被门禁压住也点名。"""
        det = detector()
        session = make_session("B01", ratios=[0.05, 0.04, 0.03, 0.02, 0.02])
        session["pickup_suppressed"] = True
        session["suppression_reason"] = "uncalibrated"
        session["eta_minutes"] = None
        session["estimated_volume_ml"] = None
        result = det.detect_round([session], now=BASE)
        assert len(result.pickups) == 1
        assert result.pickups[0].trigger_reason == "near_empty_observed"

    def test_margin_not_met_is_not_named(self):
        """条件 A 满足但条件 B 不满足：余量还够撑到下一轮，不点名。"""
        det = detector()
        result = det.detect_round(
            [make_session(ratios=[0.10] * 5, eta_minutes=400, volume_ml=260.0)], now=BASE
        )
        assert result.pickups == ()
        assert result.gate_for("B01").suppress_reason == "no_trigger"


# ── 2. 单帧毛刺不触发 ──────────────────────────────────────────────────────


class TestSingleFrameGlitch:
    def test_one_spike_does_not_trigger(self):
        """反光把某一帧顶到 0.02、其余 0.20：连续帧不够，不点名。"""
        det = detector()
        session = make_session(ratios=[0.20, 0.19, 0.02, 0.18, 0.17], eta_minutes=200, volume_ml=300.0)
        result = det.detect_round([session], now=BASE)
        assert result.pickups == ()
        assert result.gate_for("B01").suppress_reason == "no_trigger"

    def test_two_frames_still_below_n_frames(self):
        det = detector()
        session = make_session(ratios=[0.20, 0.19, 0.18, 0.02, 0.02], eta_minutes=200, volume_ml=300.0)
        result = det.detect_round([session], now=BASE)
        assert result.pickups == ()
        assert "只有 2 帧低于阈值，少于要求的连续 3 帧" in result.gate_for("B01").detail

    def test_glitch_does_not_latch_the_bag(self):
        """毛刺不该把袋子锁进点名状态——锁了会影响后续轮次的滞回行为。"""
        det = detector()
        det.detect_round(
            [make_session(ratios=[0.20, 0.02, 0.19, 0.18, 0.17])], now=BASE
        )
        snapshot = det.snapshot()
        assert all(entry["latched"] is False for entry in snapshot.values())


# ── 3. 滞回带 ──────────────────────────────────────────────────────────────


class TestHysteresis:
    def test_band_bounds_are_symmetric_around_threshold(self):
        low, high = band_bounds(0.15, 0.03)
        assert low == pytest.approx(0.135)
        assert high == pytest.approx(0.165)
        assert classify_ratio(0.10, 0.15, 0.03) == "below"
        assert classify_ratio(0.15, 0.15, 0.03) == "in_band"
        assert classify_ratio(0.20, 0.15, 0.03) == "above"

    def test_oscillation_around_threshold_is_held_not_named(self):
        """面积比在阈值附近来回抖：不点名，并说清卡在带内第几轮。"""
        det = detector()
        # 尾部三帧都 ≤ 换液阈值 0.15（判定认为该点），但都落在滞回带 [0.135, 0.165] 内：
        # 这正是反光/晃动把读数卡在阈值上下的形状——判得出来但不敢认。
        oscillating = [0.30, 0.16, 0.14, 0.15, 0.14]
        result = det.detect_round([make_session(ratios=oscillating, eta_minutes=8)], now=BASE)
        assert result.pickups == ()
        gate = result.gate_for("B01")
        assert gate.suppress_reason == "hysteresis_hold"
        assert gate.band_dwell_rounds == 1
        assert "滞回带" in gate.detail

    def test_band_patience_exhausted_resolves_as_low(self):
        """带内磨够耐心轮次：按「确已跌破」收口点名（否则真在排空的袋永远等不到）。"""
        det = detector(band_patience_rounds=3)
        series = [0.30, 0.16, 0.14, 0.15, 0.14]
        reasons = []
        for index in range(4):
            result = det.detect_round(
                [make_session(ratios=series, eta_minutes=8)],
                now=BASE + timedelta(minutes=30 * index),
            )
            gate = result.gate_for("B01")
            reasons.append("named" if gate.picked else gate.suppress_reason)
            if gate.picked:
                break
        # 前两轮耐心未满 → 仍 hold；第三轮耐心耗尽 → 收口点名
        assert reasons[:2] == ["hysteresis_hold", "hysteresis_hold"]
        assert reasons[2] == "named"

    def test_resolved_pickup_is_actually_named(self):
        det = detector(band_patience_rounds=2)
        series = [0.145, 0.15, 0.145]
        named = None
        for index in range(4):
            result = det.detect_round(
                [make_session(ratios=series, eta_minutes=8)],
                now=BASE + timedelta(minutes=30 * index),
            )
            if result.pickups:
                named = result
                break
        assert named is not None
        assert "按已跌破收口点名" in named.pickups[0].detail

    def test_named_bag_is_not_named_again_while_level_stays_low(self):
        """同一袋已点名、液位未回升：不重复点名（跨轮次的核心诉求）。"""
        det = detector()
        first = det.detect_round([make_session()], now=BASE)
        assert len(first.pickups) == 1
        later = det.detect_round([make_session()], now=BASE + timedelta(minutes=120))
        assert later.pickups == ()
        assert later.gate_for("B01").suppress_reason == "latched"

    def test_latch_releases_when_level_climbs_back_above_the_band(self):
        """液面回升到滞回带上沿之上才解除锁定（差一丁点不算）。"""
        det = detector()
        det.detect_round([make_session()], now=BASE)
        assert det.snapshot()[bag_key(make_session())]["latched"] is True
        det.detect_round(
            [make_session(ratios=[0.70, 0.68, 0.66])], now=BASE + timedelta(minutes=120)
        )
        assert det.snapshot()[bag_key(make_session())]["latched"] is False

    def test_replaced_bag_is_never_named_again(self):
        det = detector()
        result = det.detect_round([make_session(state="replaced")], now=BASE)
        assert result.pickups == ()
        assert result.gate_for("B01").suppress_reason == "state_replaced"


# ── 4. 间隔与上限 ──────────────────────────────────────────────────────────


class TestIntervalAndCap:
    def test_min_interval_blocks_a_second_pickup_inside_the_window(self):
        """15 分钟内不点第二次名。这里用「共享点名记录 + 新引擎实例」构造：
        进程重启后滞回锁定没了，但持久闸门仍认得上一次点名——那才是真实场景。"""
        history = InMemoryPickupHistory()
        first = detector(min_pickup_interval_minutes=15, history=history)
        assert len(first.detect_round([make_session()], now=BASE).pickups) == 1
        restarted = detector(min_pickup_interval_minutes=15, history=history)
        again = restarted.detect_round([make_session()], now=BASE + timedelta(minutes=10))
        assert again.pickups == ()
        assert again.gate_for("B01").suppress_reason == "interval_not_elapsed"

    def test_pickup_allowed_again_after_the_interval(self):
        history = InMemoryPickupHistory()
        detector(min_pickup_interval_minutes=15, history=history).detect_round(
            [make_session()], now=BASE
        )
        restarted = detector(min_pickup_interval_minutes=15, history=history)
        later = restarted.detect_round([make_session()], now=BASE + timedelta(minutes=16))
        assert [item.bag_seq for item in later.pickups] == ["B01"]

    def test_interval_gate_survives_a_restart_via_audit_chain(self):
        """**持久闸门**：换一个新实例（新进程），上次点名时刻从审计链读出来。"""
        conn = connect("sqlite:///:memory:")
        apply_migrations(conn)
        audit = AuditChain(conn)
        history = AuditPickupHistory(conn, audit=audit)
        first = ChangeDetector(history=history)
        assert len(first.detect_round([make_session()], now=BASE).pickups) == 1
        row = conn.execute(
            "SELECT bag_key, occurred_at FROM audit_log WHERE action = 'pickup.named'"
        ).fetchall()
        assert len(row) == 1 and row[0]["bag_key"] == bag_key(make_session())

        # 模拟进程重启：新连接 + 新引擎实例，内存里什么都不剩
        restarted = ChangeDetector(history=AuditPickupHistory(conn, audit=audit))
        assert restarted.snapshot() == {}
        again = restarted.detect_round([make_session()], now=BASE + timedelta(minutes=8))
        assert again.pickups == ()
        assert again.gate_for("B01").suppress_reason == "interval_not_elapsed"
        assert "8.0 分钟" in again.gate_for("B01").detail
        conn.close()

    def test_round_cap_keeps_the_most_urgent_by_eta(self):
        """每轮上限 3 袋：ETA 最小的先点，第四袋被截断且记明排第几。"""
        det = detector(max_pickups_per_round=3)
        sessions = [
            make_session("B01", eta_minutes=28, bed_no="11"),
            make_session("B02", eta_minutes=7, bed_no="12"),
            make_session("B03", eta_minutes=25, bed_no="13"),
            make_session("B04", eta_minutes=3, bed_no="14"),
        ]
        result = det.detect_round(sessions, now=BASE)
        assert [item.bag_seq for item in result.pickups] == ["B04", "B02", "B03"]
        assert [item.rank for item in result.pickups] == [1, 2, 3]
        assert [item.bag_seq for item in result.truncated] == ["B01"]
        assert result.truncated[0].rank == 4
        assert result.truncated[0].selected is False

    def test_truncated_bag_is_recorded_not_dropped(self):
        """截断 ≠ 丢弃：被截断的袋下一轮能重新参与，且留下原因码。"""
        det = detector(max_pickups_per_round=1)
        sessions = [
            make_session("B01", eta_minutes=20, bed_no="11"),
            make_session("B02", eta_minutes=5, bed_no="12"),
        ]
        first = det.detect_round(sessions, now=BASE)
        gate = first.gate_for("B01")
        assert gate.picked is False
        assert gate.suppress_reason == "round_cap_exceeded"
        assert "排在第 2 位被截断" in gate.detail
        assert "下一轮会重新参与" in gate.detail
        # 下一轮：B02 已被点名且锁定，B01 顺位拿到名额
        second = det.detect_round(sessions, now=BASE + timedelta(minutes=45))
        assert [item.bag_seq for item in second.pickups] == ["B01"]

    def test_bags_without_eta_sort_last_and_still_count(self):
        """拿不到 ETA 的袋排在最后，但不能因此被悄悄丢掉。"""
        det = detector(max_pickups_per_round=2)
        sessions = [
            make_session("B01", eta_minutes=None, volume_ml=10.0, bed_no="11"),
            make_session("B02", eta_minutes=20, bed_no="12"),
            make_session("B03", eta_minutes=9, bed_no="13"),
        ]
        result = det.detect_round(sessions, now=BASE)
        assert [item.bag_seq for item in result.pickups] == ["B03", "B02"]
        assert [item.bag_seq for item in result.truncated] == ["B01"]

    def test_round_cap_zero_names_nothing_but_records_truncation(self):
        det = detector(max_pickups_per_round=0)
        result = det.detect_round([make_session()], now=BASE)
        assert result.pickups == ()
        assert result.gate_for("B01").suppress_reason == "round_cap_exceeded"

    def test_dry_run_does_not_consume_the_interval_gate(self):
        """commit=False（复盘/演练）不写点名记录，也就不消耗最小间隔。"""
        history = InMemoryPickupHistory()
        det = ChangeDetector(history=history)
        det.detect_round([make_session()], now=BASE, commit=False)
        assert history.last_pickup_at(bag_key(make_session())) is None


# ── 5. 端到端：只产出建议，未确认不出站 ────────────────────────────────────


class TestEndToEndSuggestionOnly:
    def test_detector_writes_no_outbound_event_by_itself(self):
        """引擎跑完一轮后，出站事件表必须一行不增。"""
        conn = connect("sqlite:///:memory:")
        apply_migrations(conn)
        history = AuditPickupHistory(conn, audit=AuditChain(conn))
        det = ChangeDetector(history=history)
        sessions = [make_session(f"B0{index}") for index in (1, 2, 3)]
        result = det.detect_round(sessions, now=BASE)
        assert len(result.pickups) == 3
        count = conn.execute("SELECT COUNT(*) FROM infusion_event").fetchone()[0]
        assert count == 0
        conn.close()

    def test_only_named_audit_rows_are_written(self):
        """没到点名条件的不进点名审计——审计链只记「判过但被挡下」与「真的点了名」。"""
        conn = connect("sqlite:///:memory:")
        apply_migrations(conn)
        audit = AuditChain(conn)
        history = AuditPickupHistory(conn, audit=audit)
        det = ChangeDetector(history=history)
        det.detect_round(
            [
                make_session("B01"),
                make_session("B02", ratios=[0.20] * 5, eta_minutes=400, volume_ml=300.0),
            ],
            now=BASE,
        )
        rows = conn.execute(
            "SELECT action, outcome FROM audit_log WHERE action LIKE 'pickup.%'"
        ).fetchall()
        assert [(row["action"], row["outcome"]) for row in rows] == [
            ("pickup.named", "accepted")
        ]
        conn.close()

    def test_engine_never_advances_to_replaced(self):
        """未确认的 needs_change 不迁移 replaced——状态机的终态只认护士确认。"""
        conn = connect("sqlite:///:memory:")
        apply_migrations(conn)
        det = ChangeDetector(history=InMemoryPickupHistory())
        det.detect_round([make_session()], now=BASE)
        states = conn.execute("SELECT DISTINCT state FROM bag_session").fetchall()
        assert states == []
        conn.close()

    def test_round_detection_is_json_serialisable(self):
        """运行摘要与护士端文案都直接吃 to_json()，不能抛。"""
        det = detector()
        result = det.detect_round(
            [make_session("B01"), make_session("B02", ratios=[0.2] * 5, eta_minutes=400, volume_ml=300.0)],
            now=BASE,
        )
        payload = result.to_json()
        assert payload["round_key"] == ""
        assert payload["params"]["hysteresis_ratio"] == 0.03
        assert payload["params"]["max_pickups_per_round"] == 3
        assert len(payload["pickups"]) == 1
        assert payload["suppressed"][0]["bag_seq"] == "B02"
        assert payload["suppressed"][0]["suppress_reason"] == "no_trigger"

    def test_suppress_reason_codes_are_closed_set(self):
        """原因码是对外契约：新增要显式登记，不能随手写字符串。"""
        det = detector()
        result = det.detect_round(
            [
                make_session("B01", state="replaced"),
                make_session("B02", ratios=[0.30, 0.16, 0.14, 0.15, 0.14], eta_minutes=8),
                make_session("B03", ratios=[0.2] * 5, eta_minutes=400, volume_ml=300.0),
                make_session("B04", ratios=[]),
            ],
            now=BASE,
        )
        codes = {gate.suppress_reason for gate in result.suppressed if gate.suppress_reason}
        assert codes <= set(SUPPRESS_REASONS)
        assert {"state_replaced", "hysteresis_hold", "no_trigger", "no_observation"} <= codes


@pytest.mark.parametrize(
    "ratio,expected",
    [(0.0, "below"), (0.135, "below"), (0.136, "in_band"), (0.164, "in_band"), (0.165, "above")],
)
def test_classify_ratio_boundaries(ratio, expected):
    assert classify_ratio(ratio, 0.15, 0.03) == expected


def test_audit_history_is_a_noop_without_an_audit_chain():
    """只读形态（没传 audit）下 record_pickup 不该抛：读路径仍可用。"""
    conn = connect("sqlite:///:memory:")
    apply_migrations(conn)
    history = AuditPickupHistory(conn, audit=None)
    history.record_pickup("ORDER|B01", BASE)
    assert history.last_pickup_at("ORDER|B01") is None
    conn.close()


def test_parse_ts_rejects_garbage_without_crashing():
    from infusion_monitor.identity.change_detector import parse_ts

    assert parse_ts(None) is None
    assert parse_ts("") is None
    assert parse_ts("不是时间") is None
    assert parse_ts("2026-10-22T09:41:00Z") == BASE


def test_detector_is_usable_with_a_plain_sqlite_connection():
    """AuditPickupHistory 只依赖 sqlite3.Connection，不要求 AppStore。"""
    conn: sqlite3.Connection = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_migrations(conn)
    history = AuditPickupHistory(conn, audit=AuditChain(conn))
    det = ChangeDetector(history=history)
    det.detect_round([make_session()], now=BASE)
    assert history.last_pickup_at(bag_key(make_session())) == BASE
    conn.close()

# ── 6. 接口层：闸门结论必须出现在响应与审计里 ──────────────────────────────


def _low_rack_zip(frame_count: int = 16):
    """一架四袋都见底的短片（20% 液位起、16 帧内缓降），四袋液位各不相同。"""
    from tests.support.synth import BagSpec, RackScene, build_rack_zip

    scene = RackScene(
        bags=(
            BagSpec("B01", 0.14, 0.19, 0.13, 0.115, (92, 148, 196)),
            BagSpec("B02", 0.37, 0.19, 0.12, 0.105, (64, 128, 190)),
            BagSpec("B03", 0.60, 0.19, 0.11, 0.098, (80, 160, 140)),
            BagSpec("B04", 0.83, 0.19, 0.105, 0.092, (150, 120, 90)),
        ),
        frame_count=frame_count,
        width=640,
        height=480,
    )
    return build_rack_zip(scene)


class TestApiSurface:
    """走真实 HTTP 路径（TestClient + 真实 SQLite），不直接调引擎。"""

    def _upload(self, client, *, shift_key: str, zip_bytes=None):
        import json as _json

        # 先登记袋型标定：没有标定就没有毫升数与 ETA，换液点名的条件 B
        # （余量确实不够）根本无从谈起——那正是标定门禁该有的行为。
        if not getattr(client, "_calibration_ready", False):
            registered = client.post(
                "/api/v1/infusion/calibrations",
                json={
                    "bag_type_code": "TYPE-A",
                    "calibrated_by": "EXAMPLE-NURSE-7",
                    "capacity_ml": 1000.0,
                    "dead_volume_ml": 20.0,
                    "samples": [
                        {"measured_fill_ratio": 0.1, "known_volume_ml": 98.0},
                        {"measured_fill_ratio": 0.4, "known_volume_ml": 392.0},
                        {"measured_fill_ratio": 0.8, "known_volume_ml": 784.0},
                    ],
                },
            )
            assert registered.status_code == 201, registered.text
            client._calibration_ready = True  # type: ignore[attr-defined]

        return client.post(
            "/api/v1/infusion/tracks",
            data={
                "rack_id": "R-EXAMPLE-30",
                "infusion_order_no": "EXAMPLE-ORDER-0030",
                "bed_no": "EXAMPLE-BED-30",
                "captured_at": "2026-10-22T09:41:00+08:00",
                "shift_key": shift_key,
                "slots": _json.dumps(
                    [
                        {
                            "bag_seq": f"B0{index + 1}",
                            "slot_index": index,
                            "bag_type_code": "TYPE-A",
                            "rate_text": "300ml/h",
                        }
                        for index in range(4)
                    ]
                ),
            },
            files={"frames": ("frames.zip", zip_bytes or _low_rack_zip(), "application/zip")},
        )

    def test_round_cap_shows_up_in_the_upload_response(self, client):
        """四袋都该换液，但每轮上限 3 → 响应里第四袋标 truncated 而不是消失。"""
        response = self._upload(client, shift_key="2026-10-22-AM")
        assert response.status_code == 201
        bags = response.json()["bags"]
        assert len(bags) == 4
        named = [bag for bag in bags if bag.get("pickup_state") == "named"]
        truncated = [bag for bag in bags if bag.get("pickup_truncated")]
        assert len(named) == 3, [(b["bag_seq"], b.get("pickup_suppress_reason")) for b in bags]
        assert len(truncated) == 1
        assert truncated[0]["pickup_suppress_reason"] == "round_cap_exceeded"
        assert truncated[0]["pickup_rank"] == 4

    def test_named_bags_carry_state_and_trigger_reason(self, client):
        """点名之后状态要落到 needs_change 并带上原因码——否则护士端等于没收到点名。"""
        bags = self._upload(client, shift_key="2026-10-22-AM").json()["bags"]
        named = [bag for bag in bags if bag.get("pickup_state") == "named"]
        assert named, "四袋液位都低于阈值，应当有点名的袋"
        for bag in named:
            assert bag["state"] == "needs_change"
            assert bag["trigger_reason"] == "eta_within_window"
            assert bag["pickup_rank"] in (1, 2, 3)

    def test_pickup_audit_rows_are_written_for_named_and_truncated_only(
        self, store_client
    ):
        client, observer = store_client
        self._upload(client, shift_key="2026-10-22-AM")
        named = observer.execute(
            "SELECT bag_key FROM audit_log WHERE action = 'pickup.named'"
        ).fetchall()
        truncated = observer.execute(
            "SELECT bag_seq FROM audit_log WHERE action = 'pickup.truncated'"
        ).fetchall()
        assert len(named) == 3 and all(row[0] and "|" in row[0] for row in named)
        assert len(truncated) == 1
        assert observer.execute("SELECT COUNT(*) FROM infusion_event").fetchone()[0] == 0

    def test_second_round_does_not_rename_the_same_bags(self, client):
        """下一轮同一架：已点名的袋不再重复点名，被截断的袋反而拿到名额。"""
        self._upload(client, shift_key="2026-10-22-AM")
        second = self._upload(client, shift_key="2026-10-22-PM").json()["bags"]
        named = [bag for bag in second if bag.get("pickup_state") == "named"]
        assert [bag["bag_seq"] for bag in named] == ["B01"]
        held = {
            bag["bag_seq"]: bag.get("pickup_suppress_reason")
            for bag in second
            if bag["bag_seq"] != "B01"
        }
        assert set(held.values()) == {"latched"}
