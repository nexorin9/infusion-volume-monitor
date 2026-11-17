"""入站字段与帧序列的硬校验。

三条纪律：

1. **硬校验只管拍不出来的东西**：帧数、分辨率、时间序、袋位上限。这些是不合格就不能
   算数的输入，宁可退回让护士重拍，也不产出不可信的数字。拍摄质量类（模糊/过曝）
   不在这里挡，由 ``ingest/quality.py`` 降级处理。
2. **每次拒绝都带补拍建议**：``Issue.suggestion`` 是护士照着做就能改好的中文提示，
   不是一句英文报错。
3. **不静默改数据**：不合格就整段拒绝并在响应里点名，不裁剪、不排序、不猜。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..errors import Issue, ValidationRejected, field_suggestion

__all__ = [
    "MIN_FRAMES",
    "MAX_FRAMES",
    "MIN_SHORT_SIDE",
    "MAX_SLOTS",
    "FRAME_NAME_PATTERN",
    "SlotSpec",
    "UploadMeta",
    "parse_upload_fields",
    "collect_frame_issues",
    "collect_field_issues",
    "validate_frame_times",
    "validate_frame_sizes",
    "derive_shift_key",
    "parse_frame_times",
]

#: 硬校验阈值（spec.md「最小数据格式与字段要求」）
MIN_FRAMES = 12          # 少于 12 帧不足以算跨帧曲线
MAX_FRAMES = 1200        # 上限，防止一次上传把内存和磁盘吃穿
MIN_SHORT_SIDE = 320     # 短边像素下限；低于此值液面边界不可分辨
MAX_SLOTS = 6            # 单架 ≤6 袋

#: zip 成员命名约定：frame_%05d.jpg
FRAME_NAME_PATTERN = re.compile(r"^frame_(\d{5})\.jpg$")

_IDENT_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


@dataclass(frozen=True)
class SlotSpec:
    """一个袋位（输液架上的一个挂点）。"""

    bag_seq: str
    slot_index: int
    bag_type_code: str | None = None
    #: 输注医嘱里的速率文本（如「5ml/h」）。ETA 的第一优先来源；抽不出就退到泵速遥测。
    rate_text: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bag_seq": self.bag_seq,
            "slot_index": self.slot_index,
            "bag_type_code": self.bag_type_code,
            "rate_text": self.rate_text,
        }


@dataclass(frozen=True)
class UploadMeta:
    """一次上传的业务头（帧之外的部分）。"""

    rack_id: str
    infusion_order_no: str
    bed_no: str
    captured_at: str          # 原样保留调用方给的 ISO8601 串
    captured_dt: datetime
    shift_key: str
    slots: tuple[SlotSpec, ...]
    client_event_id: str | None = None
    frame_times: tuple[str, ...] | None = None  # 调用方显式给的逐帧时间（可选）


# ── 业务头字段 ──────────────────────────────────────────────────────────────


def _require_identifier(
    raw: str | None, name: str, issues: list[Issue]
) -> str | None:
    value = (raw or "").strip()
    if not value:
        issues.append(
            Issue(
                code="field_missing",
                field=name,
                reason=f"{name} 为空",
                suggestion=field_suggestion(name),
            )
        )
        return None
    if not _IDENT_RE.match(value):
        issues.append(
            Issue(
                code="field_malformed",
                field=name,
                reason=f"{name}={value!r} 含非法字符（只允许字母、数字、下划线、连字符、点，长度 1–64）",
                suggestion=f"{name} 请按院内既有编号原样填写，不要加空格或中文说明文字。",
            )
        )
        return None
    return value


def _parse_captured_at(raw: str, issues: list[Issue]) -> datetime | None:
    text = (raw or "").strip()
    if not text:
        issues.append(
            Issue(
                code="field_missing",
                field="captured_at",
                reason="captured_at 为空",
                suggestion=field_suggestion("captured_at"),
            )
        )
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        issues.append(
            Issue(
                code="captured_at_not_iso8601",
                field="captured_at",
                reason=f"captured_at={text!r} 不是 ISO8601 时间",
                suggestion=field_suggestion("captured_at"),
            )
        )
        return None
    if parsed.tzinfo is None:
        # 相机时间戳常不带时区：按病区本地时间解释并显式标注，不猜 UTC
        parsed = parsed.astimezone()
    return parsed


def derive_shift_key(captured_dt: datetime, explicit: str | None) -> str:
    """巡视轮次键：显式值优先，否则取拍摄日期。

    会话定位键是 ``rack_id + shift_key + bag_seq``；同一轮巡视重传同一段是**覆盖**
    （task 10），跨轮重传则 session_version 递增、旧版本保留可查（task 32）。
    """
    text = (explicit or "").strip()
    if text:
        if not _IDENT_RE.match(text):
            raise ValueError(f"shift_key={text!r} 含非法字符")
        return text
    return date.fromtimestamp(captured_dt.timestamp()).isoformat()


def _parse_slots(raw: str, issues: list[Issue]) -> tuple[SlotSpec, ...]:
    text = (raw or "").strip()
    if not text:
        issues.append(
            Issue(
                code="field_missing",
                field="slots",
                reason="slots 为空",
                suggestion=field_suggestion("slots"),
            )
        )
        return ()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        issues.append(
            Issue(
                code="slots_not_json",
                field="slots",
                reason=f"slots 不是合法 JSON（{exc.msg}）",
                suggestion="slots 请传 JSON 数组字符串，例如 [{\"bag_seq\":\"B01\",\"slot_index\":0}]。",
            )
        )
        return ()
    if not isinstance(parsed, list) or not parsed:
        issues.append(
            Issue(
                code="slots_not_list",
                field="slots",
                reason="slots 须为非空数组",
                suggestion="slots 请传非空 JSON 数组，每一项对应输液架上的一个袋位。",
            )
        )
        return ()

    slots: list[SlotSpec] = []
    seen_seq: dict[str, int] = {}
    seen_index: dict[int, int] = {}
    for position, item in enumerate(parsed):
        if not isinstance(item, dict):
            issues.append(
                Issue(
                    code="slot_item_not_object",
                    field="slots",
                    reason=f"slots[{position}] 不是对象",
                    suggestion="slots 每一项都须是对象，含 bag_seq 与 slot_index。",
                )
            )
            continue
        bag_seq = str(item.get("bag_seq", "")).strip()
        slot_index = item.get("slot_index")
        if not bag_seq:
            issues.append(
                Issue(
                    code="slot_missing_bag_seq",
                    field="slots",
                    reason=f"slots[{position}] 缺 bag_seq",
                    suggestion="每个袋位都要带 bag_seq（袋序标识由医嘱/护理单元系统给出）。",
                )
            )
            continue
        if not isinstance(slot_index, int) or isinstance(slot_index, bool) or slot_index < 0:
            issues.append(
                Issue(
                    code="slot_bad_index",
                    field="slots",
                    reason=f"slots[{position}].slot_index={slot_index!r} 不是非负整数",
                    suggestion="slot_index 请按输液架上的挂点从 0 开始顺序填写。",
                )
            )
            continue
        if bag_seq in seen_seq:
            issues.append(
                Issue(
                    code="slot_duplicate_bag_seq",
                    field="slots",
                    reason=f"bag_seq={bag_seq} 重复出现（位置 {seen_seq[bag_seq]} 与 {position}）",
                    suggestion="同一袋在 slots 里只能出现一次；若已换袋请用新的 bag_seq。",
                )
            )
            continue
        if slot_index in seen_index:
            issues.append(
                Issue(
                    code="slot_duplicate_index",
                    field="slots",
                    reason=f"slot_index={slot_index} 被两袋占用（位置 {seen_index[slot_index]} 与 {position}）",
                    suggestion="一个挂点只能放一袋；请检查是否把同一挂点写了两遍。",
                )
            )
            continue
        bag_type = item.get("bag_type_code")
        rate_text = item.get("rate_text", item.get("order_rate_text"))
        seen_seq[bag_seq] = position
        seen_index[slot_index] = position
        slots.append(
            SlotSpec(
                bag_seq=bag_seq,
                slot_index=slot_index,
                bag_type_code=str(bag_type).strip() if bag_type not in (None, "") else None,
                rate_text=str(rate_text).strip() if rate_text not in (None, "") else None,
            )
        )

    if len(slots) > MAX_SLOTS:
        issues.append(
            Issue(
                code="too_many_slots",
                field="slots",
                reason=f"本次上报 {len(slots)} 个袋位，超过单架 {MAX_SLOTS} 袋上限",
                suggestion=f"请把这一架拆成每架不超过 {MAX_SLOTS} 袋分批拍摄上传。",
            )
        )
    return tuple(slots)


def parse_frame_times(raw: str | None, frame_count: int) -> tuple[str, ...] | None:
    """解析调用方显式给的逐帧时间；格式错误直接抛 422。"""
    text = (raw or "").strip()
    if not text:
        return None
    issues: list[Issue] = []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationRejected(
            [
                Issue(
                    code="frame_times_not_json",
                    field="frame_times",
                    reason=f"frame_times 不是合法 JSON（{exc.msg}）",
                    suggestion=field_suggestion("frame_times"),
                )
            ]
        ) from exc
    if not isinstance(parsed, list):
        raise ValidationRejected(
            [
                Issue(
                    code="frame_times_not_list",
                    field="frame_times",
                    reason="frame_times 须为数组",
                    suggestion=field_suggestion("frame_times"),
                )
            ]
        )
    if len(parsed) != frame_count:
        raise ValidationRejected(
            [
                Issue(
                    code="frame_times_length_mismatch",
                    field="frame_times",
                    reason=f"frame_times 有 {len(parsed)} 项，与帧数 {frame_count} 不一致",
                    suggestion="frame_times 请与 frames 内帧一一对应；不方便逐帧给时可省略，"
                    "服务会退回用打包时间。",
                )
            ]
        )
    times: list[str] = []
    for position, item in enumerate(parsed):
        value = str(item).strip()
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            issues.append(
                Issue(
                    code="frame_time_not_iso8601",
                    field="frame_times",
                    reason=f"frame_times[{position}]={value!r} 不是 ISO8601 时间",
                    suggestion="frame_times 每项都写成 2026-10-22T09:41:03+08:00 这样的完整时间。",
                )
            )
            continue
        times.append(value)
    if issues:
        raise ValidationRejected(issues)
    return tuple(times)


def parse_upload_fields(fields: Mapping[str, Any]) -> UploadMeta:
    """校验业务头字段并组装 :class:`UploadMeta`；任一条不通过即抛 422。"""
    issues: list[Issue] = []
    rack_id = _require_identifier(fields.get("rack_id"), "rack_id", issues)
    order_no = _require_identifier(
        fields.get("infusion_order_no"), "infusion_order_no", issues
    )
    bed_no = _require_identifier(fields.get("bed_no"), "bed_no", issues)
    captured_dt = _parse_captured_at(str(fields.get("captured_at", "")), issues)
    slots = _parse_slots(str(fields.get("slots", "")), issues)

    client_event_id: str | None = None
    raw_client_event_id = (str(fields.get("client_event_id", "")) or "").strip()
    if raw_client_event_id:
        if _IDENT_RE.match(raw_client_event_id):
            client_event_id = raw_client_event_id
        else:
            issues.append(
                Issue(
                    code="field_malformed",
                    field="client_event_id",
                    reason=f"client_event_id={raw_client_event_id!r} 含非法字符",
                    suggestion=field_suggestion("client_event_id"),
                )
            )

    if issues:
        raise ValidationRejected(issues)

    assert captured_dt is not None and rack_id and order_no and bed_no  # issues 已清空
    try:
        shift_key = derive_shift_key(captured_dt, fields.get("shift_key"))
    except ValueError as exc:
        raise ValidationRejected(
            [
                Issue(
                    code="shift_key_malformed",
                    field="shift_key",
                    reason=str(exc),
                    suggestion=field_suggestion("shift_key"),
                )
            ]
        ) from exc

    return UploadMeta(
        rack_id=rack_id,
        infusion_order_no=order_no,
        bed_no=bed_no,
        captured_at=str(fields.get("captured_at", "")).strip(),
        captured_dt=captured_dt,
        shift_key=shift_key,
        slots=slots,
        client_event_id=client_event_id,
    )


# ── 帧序列校验 ──────────────────────────────────────────────────────────────


def _format_frame_times(times: Sequence[datetime]) -> list[str]:
    out: list[str] = []
    for moment in times:
        out.append(moment.replace(tzinfo=None).isoformat(timespec="milliseconds"))
    return out


def validate_frame_times(times: Sequence[datetime]) -> None:
    """逐帧时间戳必须单调不减，且至少有一次严格递增。

    只要求**非递减**是因为 zip 成员时间戳精度有限（DOS 时间只有 2 秒一档），
    连续拍摄时相邻帧同秒是正常的；但出现回跳说明帧序乱了，必须退回重拍。
    """
    if not times:
        raise ValidationRejected(
            [
                Issue(
                    code="frame_times_missing",
                    field="frames",
                    reason="帧序列为空",
                    suggestion="请重新录制一段连续短片后上传。",
                )
            ]
        )
    issues: list[Issue] = []
    for position in range(1, len(times)):
        previous, current = times[position - 1], times[position]
        if current < previous:
            issues.append(
                Issue(
                    code="frame_times_out_of_order",
                    field="frames",
                    reason=(
                        f"第 {position} 帧时间 {_format_frame_times([current])[0]} "
                        f"早于第 {position - 1} 帧 {_format_frame_times([previous])[0]}（帧序逆序）"
                    ),
                    suggestion="帧序出现逆序。请重新录制一段连续短片，"
                    "不要跳帧、剪辑或分多次录制的片段拼接后上传。",
                )
            )
            break
    if all(a == b for a, b in zip(times, times[1:])):
        issues.append(
            Issue(
                code="frame_times_flat",
                field="frames",
                reason="所有帧的时间戳完全相同，无法确认拍摄时长",
                suggestion="请重新拍摄一段 10–20 秒的连续短片（建议 12 帧以上）再上传。",
            )
        )
    if issues:
        raise ValidationRejected(issues)


def validate_frame_sizes(paths: Sequence[Path], *, open_image: Any = None) -> None:
    """逐帧检查短边分辨率；低于下限即 422（附「靠近一点」的具体补拍建议）。"""
    if open_image is None:  # pragma: no cover - 仅在测试注入替身时省略
        from PIL import Image

        def open_image(path: Path) -> tuple[int, int]:
            with Image.open(path) as handle:
                return handle.size

    issues: list[Issue] = []
    for position, path in enumerate(paths):
        try:
            width, height = open_image(path)
        except Exception as exc:  # 损坏文件不能静默跳过
            issues.append(
                Issue(
                    code="frame_unreadable",
                    field="frames",
                    reason=f"第 {position} 帧无法解码为图片（{type(exc).__name__}）",
                    suggestion="该段视频里有损坏帧。请重新导出或重拍这段短片后上传。",
                )
            )
            continue
        short_side = min(int(width), int(height))
        if short_side < MIN_SHORT_SIDE:
            issues.append(
                Issue(
                    code="frame_too_small",
                    field="frames",
                    reason=(
                        f"第 {position} 帧短边 {short_side}px，低于下限 {MIN_SHORT_SIDE}px"
                        f"（尺寸 {width}×{height}）"
                    ),
                    suggestion=(
                        f"请靠近输液架或关闭广角/变焦，让画面短边不小于 {MIN_SHORT_SIDE} 像素；"
                        "站远一点只拍到一个小方块时，液面高低是看不出来的。"
                    ),
                )
            )
            if len(issues) >= 5:
                break
    if issues:
        raise ValidationRejected(issues)


def collect_frame_issues(
    *,
    names: Sequence[str],
    times: Sequence[datetime],
    paths: Sequence[Path],
    open_image: Any = None,
) -> list[Issue]:
    """把帧名、帧序、时间、分辨率四类检查的结果汇成一份 issue 列表。

    一次把**所有**问题都回给调用方，而不是让护士修一个报一个。
    """
    issues: list[Issue] = []

    indices: list[int] = []
    for name in names:
        match = FRAME_NAME_PATTERN.match(name)
        if match is None:
            issues.append(
                Issue(
                    code="frame_name_malformed",
                    field="frames",
                    reason=f"帧成员名 {name!r} 不符合 frame_%05d.jpg 约定",
                    suggestion="请把帧按顺序重命名为 frame_00000.jpg、frame_00001.jpg…后重新打包。",
                )
            )
        else:
            indices.append(int(match.group(1)))
    if indices and not any(i.code == "frame_name_malformed" for i in issues):
        ordered = sorted(indices)
        if ordered != list(range(len(ordered))):
            issues.append(
                Issue(
                    code="frame_sequence_broken",
                    field="frames",
                    reason=f"帧序号不连续：共 {len(ordered)} 帧，期望 frame_00000.jpg 起连续编号",
                    suggestion="打包时不要跳号或重号；重新导出这一段连续短片后再上传。",
                )
            )

    count = len(names)
    if count < MIN_FRAMES:
        issues.append(
            Issue(
                code="frames_too_few",
                field="frames",
                reason=f"本次只收到 {count} 帧，少于下限 {MIN_FRAMES} 帧",
                suggestion=(
                    f"请把手机对准输液架连续拍 10–20 秒（约 {MIN_FRAMES}–60 帧），"
                    "中途不要移动太快；帧数不够算不出余量是升还是降。"
                ),
            )
        )
    elif count > MAX_FRAMES:
        issues.append(
            Issue(
                code="frames_too_many",
                field="frames",
                reason=f"本次收到 {count} 帧，超过上限 {MAX_FRAMES} 帧",
                suggestion="请只上传这一轮巡视的短片（建议 20–120 帧），超长视频请分段拍摄。",
            )
        )

    try:
        validate_frame_times(times)
    except ValidationRejected as exc:
        issues.extend(exc.issues)

    if paths:
        try:
            validate_frame_sizes(paths, open_image=open_image)
        except ValidationRejected as exc:
            issues.extend(exc.issues)

    if issues:
        raise ValidationRejected(issues)
    return []


def collect_field_issues(fields: Mapping[str, Any]) -> list[Issue]:
    """:func:`parse_upload_fields` 的「只收集不抛」版本，供预览/自检使用。"""
    try:
        parse_upload_fields(fields)
    except ValidationRejected as exc:
        return list(exc.issues)
    return []


def iter_frame_paths(directory: Path, names: Iterable[str]) -> list[Path]:
    return [directory / name for name in names]