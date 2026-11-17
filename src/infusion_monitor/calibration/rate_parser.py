"""输注医嘱速率文本解析（ml/h）。

「还能撑多久」全靠滴速算得出来。国内病区里医嘱的滴速写法五花八门::

    5ml/h          5 ml/h          250毫升/小时      40 毫升每小时
    125 ml/hr      每小时 250 毫升   0.9%氯化钠 500ml 静脉滴注 40ml/h
    20滴/分         20 gtt/min       20滴/分，15滴/ml   5ml/kg/h

解析器只认**能唯一换算成 ml/h** 的写法，其余一律**说不出来**（``rate_ml_per_h=None``
+ 原因码），绝不用默认值糊弄——一个猜出来的滴速会让「还剩 40 分钟」变成「还剩 4 小时」。

三条纪律：

1. **单位先统一再匹配**：全角数字与全角斜杠先折成半角、大小写统一，避免
   「５ｍｌ／ｈ」这种写法漏认（医嘱文本来自 HIS，全角半角混用是常态）。
2. **滴/分要靠滴系数换算**：滴/分本身不是速率单位，换成 ml/h 必须知道滴管多少滴 1 ml。
   医嘱里写明了就按写的算（``20滴/分，15滴/ml`` → 80 ml/h）；没写就按默认滴系数
   换算，并**把实际用的系数放进结果里**——默认系数是有前提的，不该藏在代码里。
3. **按体重给的速率不给算**：``5ml/kg/h`` 需要体重，医嘱文本里没有体重，
   解析器不猜体重也不按 60kg 默认算，直接判「认不出来」。

体积数字（``500ml``）不会被当成速率：速率的分母必须是时间单位。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_DROP_FACTOR_GTT_PER_ML",
    "PRECISION_DROP_FACTOR_GTT_PER_ML",
    "ParsedRate",
    "normalize_rate_text",
    "parse_rate",
    "parse_rate_text",
]

#: 默认滴系数（滴/ml）。成人普通输液器 20 滴/ml 是行业惯例。
DEFAULT_DROP_FACTOR_GTT_PER_ML = 20.0

#: 精密输液器 / 儿童微量滴管常见 60 滴/ml。拿 20 当 60 用会差 3 倍，
#: 所以**默认取 20**，并在 ``reason`` 里说明用了哪个系数。
PRECISION_DROP_FACTOR_GTT_PER_ML = 60.0

#: 全角 → 半角。全角 ASCII 区（U+FF01–U+FF5E：数字、字母、句点、斜杠、括号……）
#: 与半角一一对应，偏移量固定；全角空格（U+3000）折成普通空格。
_FULLWIDTH_OFFSET = 0xFEE0
_TRANSLATION = {
    code: code - _FULLWIDTH_OFFSET for code in range(0xFF01, 0xFF5F)
}
_TRANSLATION[0x3000] = 0x20

_ML_UNIT = r"(?:ml|毫升|cc)"
_HOUR_UNIT = r"(?:hours|hour|hrs|hr|h|小时)"
_MIN_UNIT = r"(?:minutes|minute|mins|min|分钟|分)"
_DROP_UNIT = r"(?:gtt|drop|drops|滴)"

#: ``5ml/h`` / ``125 ml/hr`` / ``250毫升/小时``
_RE_ML_PER_HOUR = re.compile(rf"(\d+(?:\.\d+)?)\s*{_ML_UNIT}\s*/\s*{_HOUR_UNIT}")
#: ``40 毫升每小时`` / ``5毫升每小时``
_RE_ML_PER_HOUR_EVERY = re.compile(rf"(\d+(?:\.\d+)?)\s*{_ML_UNIT}\s*每\s*{_HOUR_UNIT}")
#: ``每小时 250 毫升``（数值在单位后面）
_RE_HOUR_PER_ML = re.compile(rf"每\s*{_HOUR_UNIT}\s*(\d+(?:\.\d+)?)\s*{_ML_UNIT}")
#: ``5ml/分`` / ``2 ml/min``
_RE_ML_PER_MIN = re.compile(rf"(\d+(?:\.\d+)?)\s*{_ML_UNIT}\s*/\s*{_MIN_UNIT}")
#: ``20滴/分`` / ``20 gtt/min``
_RE_DROPS_PER_MIN = re.compile(rf"(\d+(?:\.\d+)?)\s*{_DROP_UNIT}\s*/\s*{_MIN_UNIT}")
#: ``15滴/ml`` / ``20gtt/ml``（滴系数）
_RE_DROP_FACTOR = re.compile(rf"(\d+(?:\.\d+)?)\s*{_DROP_UNIT}\s*/\s*{_ML_UNIT}")
#: ``5ml/kg/h``（按体重，缺体重算不出来）
_RE_WEIGHT_BASED = re.compile(rf"(\d+(?:\.\d+)?)\s*{_ML_UNIT}\s*/\s*kg\b")
#: 推注不是输注，没有「还能撑多久」
_RE_BOLUS = re.compile(r"(?:静脉推注|静推|推注|iv\s*push|bolus)")


@dataclass(frozen=True)
class ParsedRate:
    """一次医嘱速率解析的结果。

    ``rate_ml_per_h`` 为 ``None`` 表示**认不出来**；``reason`` 说清为什么
    （空文本 / 按体重 / 推注 / 文本里没有速率 / 速率非正数）。
    """

    rate_ml_per_h: float | None = None
    #: 速率来自哪种写法：ml_per_h / ml_per_min / drops_per_min / mixed / none
    source_form: str = "none"
    #: 文本里写明的滴/分（写了就留着，换算过程可回溯）
    drops_per_min: float | None = None
    #: 换算用的滴系数（滴/ml）
    drop_factor_gtt_per_ml: float | None = None
    reason: str = ""
    raw_text: str = ""
    #: 命中的那段原文（写进审计，事后能对着看）
    matched_text: str = ""

    @property
    def available(self) -> bool:
        return self.rate_ml_per_h is not None and self.rate_ml_per_h > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rate_ml_per_h": (
                round(self.rate_ml_per_h, 4) if self.rate_ml_per_h is not None else None
            ),
            "source_form": self.source_form,
            "drops_per_min": self.drops_per_min,
            "drop_factor_gtt_per_ml": self.drop_factor_gtt_per_ml,
            "reason": self.reason,
            "matched_text": self.matched_text,
        }


def normalize_rate_text(text: str | None) -> str:
    """全角折半角 + 转小写。匹配之前先做这一步，全角半角混写才漏不掉。"""
    if not text:
        return ""
    return str(text).translate(_TRANSLATION).lower()


def _first_match(pattern: re.Pattern[str], norm: str) -> tuple[float, str] | None:
    """返回 ``(数值, 命中文本)``；没命中返回 ``None``。"""
    match = pattern.search(norm)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):  # pragma: no cover —— 正则保证能转 float
        return None
    return value, match.group(0).strip()


def parse_rate_text(
    text: str | None,
    *,
    default_drop_factor: float = DEFAULT_DROP_FACTOR_GTT_PER_ML,
) -> ParsedRate:
    """从输注医嘱文本里抽滴速（ml/h）。

    抽不出就返回 ``rate_ml_per_h=None`` 并给原因码，**不猜、不取默认值**。
    """
    raw = "" if text is None else str(text)
    norm = normalize_rate_text(raw)
    if not norm.strip():
        return ParsedRate(reason="医嘱文本为空，没有可解析的滴速", raw_text=raw)

    factor_match = _first_match(_RE_DROP_FACTOR, norm)
    drop_factor = factor_match[0] if factor_match else None
    drops_match = _first_match(_RE_DROPS_PER_MIN, norm)

    ml_per_hour = _first_match(_RE_ML_PER_HOUR, norm) or _first_match(
        _RE_ML_PER_HOUR_EVERY, norm
    ) or _first_match(_RE_HOUR_PER_ML, norm)
    if ml_per_hour is not None:
        value, matched = ml_per_hour
        if value <= 0:
            return ParsedRate(
                source_form="ml_per_h",
                reason=f"医嘱里的速率是 {value:g} ml/h，非正数，判为不可用",
                raw_text=raw,
                matched_text=matched,
            )
        mixed = drops_match is not None
        reason = f"医嘱文本直接写明 {value:g} ml/h"
        if mixed:
            reason += f"（同一段文本里另有 {drops_match[0]:g} 滴/分，按 ml/h 为准）"
        return ParsedRate(
            rate_ml_per_h=value,
            source_form="mixed" if mixed else "ml_per_h",
            drops_per_min=drops_match[0] if drops_match else None,
            drop_factor_gtt_per_ml=drop_factor,
            reason=reason,
            raw_text=raw,
            matched_text=matched,
        )

    ml_per_min = _first_match(_RE_ML_PER_MIN, norm)
    if ml_per_min is not None:
        value, matched = ml_per_min
        if value <= 0:
            return ParsedRate(
                source_form="ml_per_min",
                reason=f"医嘱里的速率是 {value:g} ml/分，非正数，判为不可用",
                raw_text=raw,
                matched_text=matched,
            )
        return ParsedRate(
            rate_ml_per_h=value * 60.0,
            source_form="ml_per_min",
            drops_per_min=drops_match[0] if drops_match else None,
            drop_factor_gtt_per_ml=drop_factor,
            reason=f"医嘱按分钟给速率：{value:g} ml/分 → {value * 60.0:g} ml/h",
            raw_text=raw,
            matched_text=matched,
        )

    if drops_match is not None:
        drops_per_min, matched = drops_match
        if drops_per_min <= 0:
            return ParsedRate(
                source_form="drops_per_min",
                reason=f"医嘱里的滴速是 {drops_per_min:g} 滴/分，非正数，判为不可用",
                raw_text=raw,
                matched_text=matched,
            )
        factor = drop_factor if (drop_factor and drop_factor > 0) else float(default_drop_factor)
        if factor <= 0:
            factor = DEFAULT_DROP_FACTOR_GTT_PER_ML
        rate = drops_per_min * 60.0 / factor
        reason = (
            f"医嘱按滴/分给速率：{drops_per_min:g} 滴/分 ÷ {factor:g} 滴/ml × 60 "
            f"→ {rate:g} ml/h"
        )
        if not drop_factor:
            reason += f"（医嘱未写滴系数，按默认 {factor:g} 滴/ml 换算）"
        return ParsedRate(
            rate_ml_per_h=rate,
            source_form="drops_per_min",
            drops_per_min=drops_per_min,
            drop_factor_gtt_per_ml=factor,
            reason=reason,
            raw_text=raw,
            matched_text=matched,
        )

    if _RE_WEIGHT_BASED.search(norm):
        return ParsedRate(
            reason="医嘱给的是按体重的速率（ml/kg/h），没有体重算不出来",
            raw_text=raw,
        )
    if _RE_BOLUS.search(norm):
        return ParsedRate(reason="医嘱是推注，不是持续输注，没有「还能撑多久」", raw_text=raw)
    return ParsedRate(reason="医嘱文本里没有可解析的滴速", raw_text=raw)


def parse_rate(
    text: str | None,
    *,
    default_drop_factor: float = DEFAULT_DROP_FACTOR_GTT_PER_ML,
) -> float | None:
    """只要一个 ml/h 数字的便捷入口；抽不出返回 ``None``。"""
    return parse_rate_text(text, default_drop_factor=default_drop_factor).rate_ml_per_h
