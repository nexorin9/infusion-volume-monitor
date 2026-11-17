"""输注医嘱速率文本解析（task 26）。

医嘱文本来自 HIS，写法五花八门。解析器要认得出常见写法，也必须在**认不出**时
老实说认不出——猜一个滴速出来的 ETA 会让护士按错的时间安排换液。

三类用例：

* **认得出**：ml/h 的各种写法、滴/分与滴系数换算、ml/分、全角半角混写；
* **不该认**：体积数字（``500ml``）不是速率、按体重给的速率（缺体重）、推注；
* **认不出**：空文本、没有速率、非正速率——每个都要给出说得清的原因码。
"""

from __future__ import annotations

import pytest

from infusion_monitor.calibration.rate_parser import (
    DEFAULT_DROP_FACTOR_GTT_PER_ML,
    PRECISION_DROP_FACTOR_GTT_PER_ML,
    normalize_rate_text,
    parse_rate,
    parse_rate_text,
)

# ── 1. 常见的 ml/h 写法 ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("5ml/h", 5.0),
        ("5 ml/h", 5.0),
        ("250毫升/小时", 250.0),
        ("40 毫升每小时", 40.0),
        ("  8 ML / H  ", 8.0),
        ("125 ml/hr", 125.0),
        ("60ml/h", 60.0),
        ("1000 ml/hour", 1000.0),
        ("每小时 250 毫升", 250.0),
        ("0.5ml/h", 0.5),
        ("输液速度 40ml/h", 40.0),
        ("持续泵入 3ml/h", 3.0),
        ("0.9%氯化钠注射液 500ml 静脉滴注 40ml/h", 40.0),
        ("５ｍｌ／ｈ", 5.0),  # 全角
        ("０．５ｍｌ／ｈ", 0.5),  # 全角小数点
    ],
)
def test_common_ml_per_hour_forms(text, expected):
    assert parse_rate(text) == pytest.approx(expected)


def test_parsed_rate_records_matched_text_for_audit():
    """解析结果要带上命中的原文——事后对着医嘱核对时用得上。"""
    parsed = parse_rate_text("0.9%氯化钠 500ml 静脉滴注 40ml/h")

    assert parsed.rate_ml_per_h == pytest.approx(40.0)
    assert parsed.source_form == "ml_per_h"
    assert parsed.matched_text == "40ml/h"
    assert parsed.raw_text.startswith("0.9%")
    assert parsed.available is True


# ── 2. 滴/分与滴系数 ────────────────────────────────────────────────────────


def test_drops_per_minute_uses_default_drop_factor():
    """医嘱只写滴/分：按默认滴系数（20 滴/ml）换算，并把实际用的系数带出来。"""
    parsed = parse_rate_text("20滴/分")

    assert parsed.rate_ml_per_h == pytest.approx(60.0)  # 20 × 60 ÷ 20
    assert parsed.source_form == "drops_per_min"
    assert parsed.drops_per_min == pytest.approx(20.0)
    assert parsed.drop_factor_gtt_per_ml == pytest.approx(DEFAULT_DROP_FACTOR_GTT_PER_ML)
    assert "默认" in parsed.reason


def test_explicit_drop_factor_wins_over_default():
    """医嘱写明了滴系数就按写的算：20 滴/分、15 滴/ml → 80 ml/h。"""
    parsed = parse_rate_text("20滴/分，15滴/ml")

    assert parsed.rate_ml_per_h == pytest.approx(80.0)
    assert parsed.drop_factor_gtt_per_ml == pytest.approx(15.0)
    assert "默认" not in parsed.reason


def test_gtt_per_minute_english_form():
    assert parse_rate("20 gtt/min") == pytest.approx(60.0)


def test_precision_drop_factor_is_a_different_constant():
    """精密输液器 60 滴/ml 与普通 20 滴/ml 差 3 倍，两个常量都要在。"""
    assert PRECISION_DROP_FACTOR_GTT_PER_ML == pytest.approx(60.0)
    assert DEFAULT_DROP_FACTOR_GTT_PER_ML == pytest.approx(20.0)
    # 同一个滴/分，换个滴系数差 3 倍——这正是不该藏起来的假设
    assert parse_rate("20滴/分") == pytest.approx(60.0)
    assert parse_rate("20滴/分，60滴/ml") == pytest.approx(20.0)


def test_ml_per_minute_is_converted_to_hour():
    parsed = parse_rate_text("5ml/分")

    assert parsed.rate_ml_per_h == pytest.approx(300.0)
    assert parsed.source_form == "ml_per_min"


def test_mixed_text_prefers_ml_per_hour_and_keeps_drops():
    """同一段医嘱里既有 ml/h 又有滴/分：按 ml/h 为准，滴/分留着可回溯。"""
    parsed = parse_rate_text("40ml/h（或20滴/分）")

    assert parsed.rate_ml_per_h == pytest.approx(40.0)
    assert parsed.source_form == "mixed"
    assert parsed.drops_per_min == pytest.approx(20.0)


# ── 3. 不该当成速率的东西 ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "0.9%氯化钠注射液 500ml 静脉滴注",  # 只有体积
        "500ml",
        "生理盐水 250ml 静脉滴注",
    ],
)
def test_volume_is_not_mistaken_for_a_rate(text):
    """``500ml`` 是袋子的容量，不是速度：分母必须是时间单位。"""
    assert parse_rate(text) is None


def test_weight_based_rate_is_refused_not_guessed():
    """按体重给的速率需要体重，医嘱文本里没有——不按默认体重硬算。"""
    parsed = parse_rate_text("5ml/kg/h")

    assert parsed.rate_ml_per_h is None
    assert "体重" in parsed.reason


def test_bolus_has_no_eta():
    """推注不是持续输注，没有「还能撑多久」。"""
    parsed = parse_rate_text("生理盐水 20ml 静脉推注")

    assert parsed.rate_ml_per_h is None
    assert "推注" in parsed.reason


# ── 4. 认不出来时也要说清为什么 ────────────────────────────────────────────


def test_empty_text_is_reported_as_empty():
    for text in (None, "", "   "):
        parsed = parse_rate_text(text)
        assert parsed.rate_ml_per_h is None
        assert parsed.reason


def test_text_without_rate_says_so():
    parsed = parse_rate_text("常规输液")

    assert parsed.rate_ml_per_h is None
    assert "没有可解析的滴速" in parsed.reason
    assert parsed.source_form == "none"


def test_non_positive_rate_is_refused():
    """0 ml/h 不是「没写」，是写错了——两种都要拒绝，但原因码不同。"""
    parsed = parse_rate_text("0ml/h")

    assert parsed.rate_ml_per_h is None
    assert "非正数" in parsed.reason


def test_zero_drops_per_minute_is_refused():
    parsed = parse_rate_text("0滴/分")

    assert parsed.rate_ml_per_h is None
    assert "非正数" in parsed.reason


# ── 5. 归一化与便捷入口 ────────────────────────────────────────────────────


def test_normalisation_folds_full_width_and_case():
    assert normalize_rate_text("５ＭＬ／Ｈ") == "5ml/h"
    assert normalize_rate_text(None) == ""


def test_parse_rate_matches_parse_rate_text():
    text = "40ml/h"
    assert parse_rate(text) == parse_rate_text(text).rate_ml_per_h


def test_custom_default_drop_factor_is_honoured():
    parsed = parse_rate_text("20滴/分", default_drop_factor=PRECISION_DROP_FACTOR_GTT_PER_ML)

    assert parsed.rate_ml_per_h == pytest.approx(20.0)
    assert parsed.drop_factor_gtt_per_ml == pytest.approx(60.0)
