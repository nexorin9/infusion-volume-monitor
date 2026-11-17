"""分片续传与断点查询：病区网络不稳时，拍到一半的那段怎么接着传。

**为什么要有这一层**：病区 Wi-Fi 在走廊尽头会断，移动护理终端拍完一轮再一次性
上传很容易在第 7 帧丢掉——前面 7 帧是拍好的，重拍一遍护士不干。让上传能**分片、
可续、可重复**是把「网络问题」从护士身上挪走的唯一办法。

四条纪律：

1. **续传与整段上传走同一条主干**：``commit`` 只是把已收齐的帧按帧序拼好，再交给
   ``ingest/multipart.process_track_upload``——校验、分割、标定、点名全部复用，
   不另起第二条推理路径（分片模式不是「简化版主路径」）。
2. **重复帧先到先得，但先核对是不是同一帧**：同 ``upload_id`` 下同号帧只留一份，
   内容一致就安静去重（终端重试同一分片的正常情形）；内容不一致说明两端对帧号的
   理解已经分叉，当场 422 问清楚，而不是把整片悄悄丢掉让断点永远停在原地。
3. **帧号是全局的**：分片内的成员仍按整段的序号命名（第二片从 ``frame_00012.jpg`` 起），
   断点查询才能逐帧告诉终端「缺哪几帧」。
4. **半成品有保留期**：开着没提交的上传会留在磁盘上，``purge_expired`` 按配置的
   保留期回收暂存目录并结案为 ``abandoned``；已完成的会话目录一个都不碰。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastapi import APIRouter, File, Form, Request, UploadFile

from ..api.schemas import (
    CleanupResultOut,
    TrackUploadResponse,
    UploadChunkAckOut,
    UploadOpenOut,
    UploadStatusOut,
)
from ..errors import ApiError, Issue, ValidationRejected
from ..logging_utils import request_id_of
from ..store.repositories import UploadRepository, new_id
from .multipart import process_track_upload, save_upload_to_temp
from .resize import ResizePolicy, resize_frame_file
from .validate import (
    MAX_FRAMES,
    SlotSpec,
    UploadMeta,
    parse_frame_times,
    parse_upload_fields,
)
from .zip_frames import upload_key

__all__ = [
    "router",
    "ChunkOutcome",
    "PurgeResult",
    "missing_frames",
    "purge_expired",
    "staging_root",
]

logger = logging.getLogger("infusion_monitor.ingest.resume")

router = APIRouter(prefix="/api/v1/infusion/uploads", tags=["ingest"])

#: 暂存根目录名：与正式帧目录（``frames_root/{upload_key}``）分开，清理策略只扫这一层。
STAGING_DIRNAME = "_staging"

#: 单片帧数上限：病区终端按 ~50 帧一片切；超过多半是客户端把整段打成了一片。
MAX_CHUNK_FRAMES = 300

#: ``upload_id`` 是调用方给的标识，而它同时是**暂存目录名**（``_staging/{upload_id}``）
#: —— 因此必须先按白名单校一遍，不能让 ``../`` 之类的东西拼进磁盘路径。
UPLOAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _checked_upload_id(raw: str | None) -> str:
    """校验调用方自带的 ``upload_id``；不合约定即 422，不落任何盘。"""
    text = (raw or "").strip()
    if not text:
        return new_id("upl")
    if not UPLOAD_ID_PATTERN.match(text):
        raise ValidationRejected(
            [
                Issue(
                    code="upload_id_malformed",
                    field="upload_id",
                    reason=f"upload_id={text!r} 含非法字符或长度超限",
                    suggestion=(
                        "upload_id 只能包含字母、数字、下划线与连字符（1–64 位）；"
                        "不传则由服务端生成。"
                    ),
                )
            ]
        )
    return text


# ── 数据结构 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkOutcome:
    """一个分片入站后的结论。"""

    accepted: tuple[int, ...]
    duplicates: tuple[int, ...]
    total_bytes: int
    received: tuple[int, ...]
    duplicate_total: int

    @property
    def received_count(self) -> int:
        return len(self.received)


@dataclass
class PurgeResult:
    """一次半成品清理的结论（磁盘回收了多少、结案了几条）。"""

    expired: list[str] = field(default_factory=list)
    freed_directories: list[str] = field(default_factory=list)
    freed_bytes: int = 0
    kept_completed: int = 0
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "expired_uploads": list(self.expired),
            "expired_count": len(self.expired),
            "freed_directories": list(self.freed_directories),
            "freed_bytes": self.freed_bytes,
            "kept_completed": self.kept_completed,
            "dry_run": self.dry_run,
        }


def staging_root(frames_root: Path) -> Path:
    return Path(frames_root) / STAGING_DIRNAME


def frames_root_of(state: Any) -> Path:
    """帧目录根。

    取 ``state.settings.frames_root`` 而不是 ``state.frames_root``：``/tracks`` 与
    ``/health`` 读的都是前者（见 ``ingest/multipart.py`` 与 ``api/health.py``）。
    提交后的帧必须落在**与整段上传同一个根**下，否则同一袋的两次上传会分家，
    重锚就再也接不上了。
    """
    return Path(state.settings.frames_root)


def _utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _expires_at(retention_hours: float) -> str:
    return _iso(_utc_now_dt() + timedelta(hours=float(retention_hours)))


def missing_frames(received: Sequence[int], expected: int | None) -> list[int]:
    """还缺哪些帧号。``expected`` 未知时返回空列表（不给终端一个假的清单）。"""
    if expected is None:
        return []
    have = set(received)
    return [index for index in range(int(expected)) if index not in have]


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


# ── 半成品清理 ────────────────────────────────────────────────────────────


def purge_expired(
    uploads: UploadRepository,
    frames_root: Path,
    *,
    retention_hours: float | None = None,
    dry_run: bool = False,
    audit: Any = None,
    request_id: str | None = None,
) -> PurgeResult:
    """回收保留期已到的半成品上传。

    **只碰 ``upload_chunk.status='pending'`` 且 ``expires_at`` 已过的行**，
    而且只删它们自己的暂存目录（``frames_root/_staging/{upload_id}``）：

    * ``completed`` 的暂存目录在提交那一刻就已经删了，这一轮只计数不动它；
    * 正式帧目录（``frames_root/{upload_key}``）属于**已完成会话**，本函数从不遍历，
      因此不存在「清理半成品顺手删掉已算好的余量曲线」这种事。
    """
    result = PurgeResult(dry_run=dry_run)
    hours = 24.0 if retention_hours is None else float(retention_hours)
    cutoff = _iso(_utc_now_dt() - timedelta(hours=hours))

    for row in uploads.list_expired(before=cutoff, status="pending"):
        upload_id = row["upload_id"]
        result.expired.append(upload_id)
        staging = Path(row["staging_dir"])
        size = _directory_bytes(staging)
        result.freed_bytes += size
        if staging.exists():
            result.freed_directories.append(str(staging))
            if not dry_run:
                shutil.rmtree(staging, ignore_errors=True)
        if not dry_run:
            uploads.mark_abandoned([upload_id])
            if audit is not None and request_id:
                audit.append(
                    request_id=request_id,
                    actor="service:upload-cleanup",
                    action="upload.abandoned",
                    target=upload_id,
                    outcome="accepted",
                    detail={
                        "reason": "retention_expired",
                        "retention_hours": hours,
                        "received_count": len(row.get("received_frames") or []),
                        "expected_frames": row.get("expected_frames"),
                        "freed_bytes": size,
                    },
                )
            logger.info("半成品上传已回收 upload=%s frames=%s freed_bytes=%d", upload_id,
                        len(row.get("received_frames") or []), size)

    result.kept_completed = uploads.count_by_status("completed")
    return result


# ── 分片入站（内部函数，端点与测试共用）────────────────────────────────────


def _reject(code: str, reason: str, suggestion: str, *, error_code: str = "UPLOAD_REJECTED") -> None:
    raise ValidationRejected(
        [Issue(code=code, field="chunk", reason=reason, suggestion=suggestion)],
        error_code=error_code,
    )


def _same_frame_bytes(stored: Path, data: bytes) -> bool:
    """同号帧的「是不是同一帧」判定——**按字节**，不做像素级近似。

    为什么这么严：同号帧有两种来源，帧号上长得一模一样——终端**重试**一个已经收到
    过的分片（HTTP 层重试天然重发同一份字节，该安静去重），和终端把每个分片**从 0
    重新编号**（该当场问清楚）。而逐帧液面变化在整幅画面上只占千分之几的像素，
    「看起来像不像」根本分不开：一旦把不同内容当重复帧丢掉，断点就永远停在原地，
    且**没有任何报错**。宁可误伤一次重新打包（报 422，且文案说清两条可能与出路），
    也不静默吞掉一整片。
    """
    try:
        if stored.stat().st_size != len(data):
            return False
        return stored.read_bytes() == data
    except OSError:  # pragma: no cover - 暂存文件被外部删掉
        return False


def _parse_chunk_times(
    raw: str | None, indices: Sequence[int]
) -> dict[int, str]:
    """分片自带的逐帧时间（JSON 数组，与片内帧序一一对应）。"""
    explicit = parse_frame_times(raw, len(indices))
    if not explicit:
        return {}
    out: dict[int, str] = {}
    for index, value in zip(indices, explicit):
        out[index] = value
    return out


def ingest_chunk(
    uploads: UploadRepository,
    *,
    upload_id: str,
    zip_path: Path,
    staging_dir: Path,
    raw_frame_times: str | None = None,
    frame_offset: int | None = None,
    retention_hours: float = 24.0,
    resize_policy: ResizePolicy | None = None,
) -> ChunkOutcome:
    """收一个分片：解包 → 逐帧去重 → 落暂存目录 → 登记已收帧号。

    ``zip_path`` 必须已经由调用方落到临时文件（本函数只读它一次）。

    **与整段路走同一套缩放策略**（task 35）：收片时就把长边缩到上限，因此同一段短片
    分 3 片传和整段传，提交后得到**同一份**面积曲线与毫升数——病区网络好不好只影响
    传得多累，不影响数字。相机原始尺寸记在 ``upload_chunk.resize_json`` 里，提交时
    原样带回响应。
    """
    import zipfile

    from .zip_frames import chunk_members

    row = uploads.get(upload_id)
    if row is None:
        raise ApiError(
            "NOT_FOUND",
            f"upload_id={upload_id} 不存在或已被清理；请重新开启一次上传（POST /api/v1/infusion/uploads）",
        )
    if row["status"] != "pending":
        raise ApiError(
            "UPLOAD_NOT_OPEN",
            f"upload_id={upload_id} 已于 {row.get('completed_at') or row.get('updated_at')} "
            f"结案（status={row['status']}），不再接收分片；"
            "需要重传整段请直接 POST /api/v1/infusion/tracks。",
        )

    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    members = chunk_members(zip_path, frame_offset=frame_offset)
    if not members:
        _reject(
            "chunk_empty",
            "这个分片里没有任何帧",
            "请确认打包时选中了这一段帧；空分片不会推进断点。",
        )
    if len(members) > MAX_CHUNK_FRAMES:
        _reject(
            "chunk_too_large",
            f"单个分片有 {len(members)} 帧，超过单片上限 {MAX_CHUNK_FRAMES} 帧",
            f"请按每片不超过 {MAX_CHUNK_FRAMES} 帧切分后重传；整段上限仍是 {MAX_FRAMES} 帧。",
        )

    indices = [member.index for member in members]
    already: set[int] = set(row.get("received_frames") or [])
    explicit_times = _parse_chunk_times(raw_frame_times, indices)
    expected = row.get("expected_frames")
    accepted: list[int] = []
    duplicates: list[int] = []
    added_bytes = 0
    resize_payload: dict[str, Any] | None = None
    with zipfile.ZipFile(zip_path) as archive:
        for member in members:
            target = staging_dir / f"frame_{member.index:05d}.jpg"
            if expected is not None and member.index >= int(expected):
                _reject(
                    "frame_index_beyond_expected",
                    f"分片里出现第 {member.index} 帧，超出本次申报的 {int(expected)} 帧",
                    "请检查两端对这一段帧数的口径；不一致时以重新开启上传时申报的帧数为准。",
                )
            # 先到先得：已收过的帧号不覆盖。内容一致就安静去重（这是终端重试同一个
            # 分片的正常情形）；内容不一致说明两端对帧号的理解已经分叉，此时必须
            # 问清楚——照单去重会把整片悄悄丢掉，断点永远停在原地。
            if member.index in already or target.exists():
                if target.exists() and not _same_frame_bytes(
                    target, archive.read(member.name)
                ):
                    _reject(
                        "chunk_frame_conflict",
                        f"第 {member.index} 帧之前已经收到过，但这次传来的不是同一帧",
                        "两种可能：①这个分片把帧号从 0 重新排了一遍——请按整段的全局序号"
                        "命名（第二片从 frame_00012.jpg 起），只能按片内编号的终端请显式传 "
                        "frame_offset；②这一片是重新打包后再传的——若这些帧确实已经在 "
                        "断点查询（GET /api/v1/infusion/uploads/{upload_id}）的 received_frames "
                        "里，直接跳过本片即可。"
                        "已收到的帧一律不受影响。",
                    )
                duplicates.append(member.index)
                continue
            with archive.open(member.name) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink)
            if resize_policy is not None:
                # 落盘即规范化：与 /tracks 的 extract_frames 同一条策略，不缩的一帧不动
                plan = resize_frame_file(target, resize_policy)
                if resize_payload is None:
                    resize_payload = dict(plan.as_dict(), resized_frames=1 if plan.resized else 0)
                elif plan.resized:
                    resize_payload["resized_frames"] = int(
                        resize_payload.get("resized_frames") or 0
                    ) + 1
            added_bytes += target.stat().st_size
            accepted.append(member.index)

    # 片内时间优先用调用方给的；没给就退回 zip 成员时间戳（与 /tracks 同一口径）。
    frame_times: dict[int, str] = {
        member.index: explicit_times.get(member.index, member.time_iso)
        for member in members
    }

    record = uploads.record_chunk(
        upload_id,
        accepted_frames=accepted,
        frame_times=frame_times,
        duplicate_frames=duplicates,
        added_bytes=added_bytes,
        expires_at=_expires_at(retention_hours),
        resize=resize_payload,
    )
    return ChunkOutcome(
        accepted=tuple(sorted(accepted)),
        duplicates=tuple(sorted(duplicates)),
        total_bytes=added_bytes,
        received=tuple(record["received_frames"]),
        duplicate_total=int(record["duplicate_count"]),
    )


# ── 端点 ──────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=UploadOpenOut,
    status_code=201,
    summary="开启一次可续传的上传",
    description=(
        "拍到一半断网时先开一次上传拿到 upload_id，随后按分片补传；"
        "全部收齐后 POST …/{upload_id}/commit 提交，走与 /tracks 完全相同的主路径。"
    ),
)
async def open_upload(
    request: Request,
    rack_id: str = Form(description="输液架编号"),
    infusion_order_no: str = Form(description="输注单号"),
    bed_no: str = Form(description="床号"),
    captured_at: str = Form(description="拍摄时刻，ISO8601"),
    slots: str = Form(description="JSON 数组，袋位声明；与 /tracks 同一口径"),
    shift_key: str | None = Form(default=None, description="巡视轮次键（可选）"),
    client_event_id: str | None = Form(default=None, description="幂等键（可选）"),
    expected_frames: int | None = Form(
        default=None, description="本段总帧数；给了才能算出「还缺哪几帧」"
    ),
    upload_id: str | None = Form(default=None, description="自定义 upload_id（可选）"),
) -> UploadOpenOut:
    state = request.app.state
    meta: UploadMeta = parse_upload_fields(
        {
            "rack_id": rack_id,
            "infusion_order_no": infusion_order_no,
            "bed_no": bed_no,
            "captured_at": captured_at,
            "slots": slots,
            "client_event_id": client_event_id,
            "shift_key": shift_key,
        }
    )
    if expected_frames is not None and expected_frames <= 0:
        raise ValidationRejected(
            [
                Issue(
                    code="expected_frames_not_positive",
                    field="expected_frames",
                    reason=f"expected_frames={expected_frames} 不是正整数",
                    suggestion="帧数请按这一段实际拍到的张数填；不确定就留空。",
                )
            ]
        )

    pending = state.store.uploads.count_pending()
    limit = int(state.settings.upload_resume_max_pending)
    if pending >= limit:
        raise ApiError(
            "UPLOAD_TOO_MANY_PENDING",
            f"当前有 {pending} 段传了一半还没提交，超过上限 {limit}；"
            "多半是网络断了以后终端反复重开。请先 POST /api/v1/infusion/uploads/cleanup "
            "清理过保留期的半成品，或让护士重新拍一段整的直接走 /tracks。",
        )

    frames_root = frames_root_of(state)
    key = upload_key(meta.rack_id, meta.infusion_order_no, meta.shift_key)
    identifier = _checked_upload_id(upload_id)
    if state.store.uploads.get(identifier) is not None:
        raise ValidationRejected(
            [
                Issue(
                    code="upload_id_taken",
                    field="upload_id",
                    reason=f"upload_id={identifier} 已经被用过",
                    suggestion=(
                        "换一个 upload_id，或不传该字段让服务端生成；"
                        "对已存在的 upload_id 请用 POST …/{id}/chunks 继续补传。"
                    ),
                )
            ]
        )
    staging = staging_root(frames_root) / identifier
    row = state.store.uploads.open_upload(
        {
            "upload_id": identifier,
            "rack_id": meta.rack_id,
            "bed_no": meta.bed_no,
            "infusion_order_no": meta.infusion_order_no,
            "shift_key": meta.shift_key,
            "captured_at": meta.captured_at,
            "slots_json": json.dumps(
                [slot.as_dict() for slot in meta.slots], ensure_ascii=False
            ),
            "client_event_id": meta.client_event_id,
            "session_key": key,
            "staging_dir": str(staging),
            "expected_frames": expected_frames,
            "expires_at": _expires_at(state.settings.upload_resume_retention_hours),
        }
    )
    state.store.audit_chain.append(
        request_id=request_id_of(request),
        actor="client:nursing-terminal",
        action="upload.opened",
        target=identifier,
        outcome="accepted",
        bag_seq=None,
        detail={
            "rack_id": meta.rack_id,
            "shift_key": meta.shift_key,
            "expected_frames": expected_frames,
            "slots": len(meta.slots),
            "retention_hours": state.settings.upload_resume_retention_hours,
        },
    )
    return UploadOpenOut(
        upload_id=identifier,
        status=row["status"],
        session_key=key,
        expected_frames=expected_frames,
        received_frames=[],
        missing_frames=[],
        expires_at=row["expires_at"],
    )


@router.post(
    "/{upload_id}/chunks",
    response_model=UploadChunkAckOut,
    summary="补传一个分片",
    description=(
        "分片是 zip，成员仍按整段序号命名（第二片从 frame_00012.jpg 起）。"
        "重复帧号先到先得、后到的丢弃并计数，断点因此只前进不回退。"
    ),
)
async def put_chunk(
    request: Request,
    upload_id: str,
    chunk: UploadFile = File(description="帧分片 zip"),
    frame_times: str | None = Form(default=None, description="与片内帧一一对应的 ISO8601 时间数组"),
    frame_offset: int | None = Form(default=None, description="片内帧号偏移（客户端用片内编号时传）"),
) -> UploadChunkAckOut:
    state = request.app.state
    row = state.store.uploads.get(upload_id)
    if row is None:
        raise ApiError(
            "NOT_FOUND",
            f"upload_id={upload_id} 不存在或已被清理；请重新开启一次上传",
        )
    temp_path = await save_upload_to_temp(chunk, Path(state.upload_spool))
    try:
        outcome = ingest_chunk(
            state.store.uploads,
            upload_id=upload_id,
            zip_path=temp_path,
            staging_dir=row["staging_dir"],
            raw_frame_times=frame_times,
            frame_offset=frame_offset,
            retention_hours=state.settings.upload_resume_retention_hours,
            resize_policy=state.settings.resize_policy,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    state.store.audit_chain.append(
        request_id=request_id_of(request),
        actor="client:nursing-terminal",
        action="upload.chunk_received",
        target=upload_id,
        outcome="accepted",
        detail={
            "accepted_frames": len(outcome.accepted),
            "duplicates_dropped": len(outcome.duplicates),
            "received_count": outcome.received_count,
            "added_bytes": outcome.total_bytes,
        },
    )
    expected = row.get("expected_frames")
    return UploadChunkAckOut(
        upload_id=upload_id,
        accepted_frames=list(outcome.accepted),
        duplicates_dropped=list(outcome.duplicates),
        duplicate_count_total=outcome.duplicate_total,
        received_frames=list(outcome.received),
        received_count=outcome.received_count,
        missing_frames=missing_frames(outcome.received, expected),
        added_bytes=outcome.total_bytes,
        expires_at=state.store.uploads.get(upload_id)["expires_at"],
    )


@router.get(
    "/{upload_id}",
    response_model=UploadStatusOut,
    summary="断点查询：已收到哪些帧",
    description="终端据此只补传缺的那几片，不用整段重来。",
)
async def get_upload_status(upload_id: str, request: Request) -> UploadStatusOut:
    row = request.app.state.store.uploads.get(upload_id)
    if row is None:
        raise ApiError("NOT_FOUND", f"upload_id={upload_id} 不存在或已被清理")
    received = list(row["received_frames"])
    expected = row.get("expected_frames")
    return UploadStatusOut(
        upload_id=upload_id,
        status=row["status"],
        rack_id=row["rack_id"],
        shift_key=row["shift_key"],
        session_key=row["session_key"],
        expected_frames=expected,
        received_frames=received,
        received_count=len(received),
        missing_frames=missing_frames(received, expected),
        duplicate_count=int(row["duplicate_count"]),
        chunk_count=int(row["chunk_count"]),
        session_ids=list(row["session_ids"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        expires_at=row["expires_at"],
        completed_at=row.get("completed_at"),
    )


@router.post(
    "/{upload_id}/commit",
    response_model=TrackUploadResponse,
    response_model_exclude_none=True,
    status_code=201,
    summary="提交：已收齐的帧走主干算余量",
    description=(
        "把暂存的帧按帧序拼好后交给与 /tracks 相同的主路径：校验 → 分割 → 标定 → "
        "换液点名。帧数不足或帧序不连续仍返回 422，且补传不丢已收到的分片。"
    ),
)
async def commit_upload(upload_id: str, request: Request):
    """提交。响应与 ``POST /tracks`` 完全一致（TrackUploadResponse）。"""
    state = request.app.state
    row = state.store.uploads.get(upload_id)
    if row is None:
        raise ApiError("NOT_FOUND", f"upload_id={upload_id} 不存在或已被清理")
    if row["status"] == "completed":
        raise ApiError(
            "UPLOAD_ALREADY_COMPLETED",
            f"upload_id={upload_id} 已于 {row.get('completed_at')} 提交，产出 "
            f"{len(row['session_ids'])} 个袋级会话；重复提交不会再算一遍。",
        )
    if row["status"] != "pending":
        raise ApiError(
            "UPLOAD_NOT_OPEN",
            f"upload_id={upload_id} 已结案（status={row['status']}），无法提交；请重新开启一次上传。",
        )

    received = sorted(row["received_frames"])
    expected = row.get("expected_frames")
    gaps = missing_frames(received, expected)
    if gaps:
        raise ValidationRejected(
            [
                Issue(
                    code="chunk_gap",
                    field="frames",
                    reason=(
                        f"本次共 {len(received)} 帧，缺 {len(gaps)} 帧"
                        f"（{gaps[:5]}{'…' if len(gaps) > 5 else ''}）"
                    ),
                    suggestion=(
                        "请按 GET /api/v1/infusion/uploads/{upload_id} 返回的 "
                        "missing_frames 补传这几片；已收到的分片不会重传。"
                    ),
                )
            ]
        )

    frames_root = frames_root_of(state)
    session_key = row["session_key"]
    frames_dir = frames_root / session_key
    staging = Path(row["staging_dir"])
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in received:
        source = staging / f"frame_{index:05d}.jpg"
        target = frames_dir / f"frame_{index:05d}.jpg"
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
        paths.append(target)
    shutil.rmtree(staging, ignore_errors=True)

    times = _times_from_row(row, received)
    extracted = _extracted_from_paths(paths, times, row.get("resize_json"))
    meta = _meta_from_row(row)

    response = process_track_upload(state=state, request=request, meta=meta, extracted=extracted)
    session_ids = [bag.session_id for bag in response.bags]
    state.store.uploads.mark_completed(upload_id, session_ids)
    state.store.audit_chain.append(
        request_id=request_id_of(request),
        actor="client:nursing-terminal",
        action="upload.completed",
        target=upload_id,
        outcome="accepted",
        detail={
            "frame_count": len(received),
            "session_ids": session_ids,
            "session_key": session_key,
            "duplicate_count": int(row["duplicate_count"]),
        },
    )
    return response


@router.post(
    "/cleanup",
    response_model=CleanupResultOut,
    summary="回收保留期已到的半成品上传",
    description=(
        "病区侧按 cron 或信息科手工调用。只回收 _staging 下「开着没提交」的帧目录，"
        "已完成会话的帧目录一个都不碰；dry_run=true 只报不删。"
    ),
)
async def cleanup_uploads(
    request: Request,
    retention_hours: float | None = None,
    dry_run: bool = False,
) -> CleanupResultOut:
    state = request.app.state
    hours = (
        state.settings.upload_resume_retention_hours
        if retention_hours is None
        else float(retention_hours)
    )
    result = purge_expired(
        state.store.uploads,
        frames_root_of(state),
        retention_hours=hours,
        dry_run=dry_run,
        audit=state.store.audit_chain,
        request_id=request_id_of(request),
    )
    return CleanupResultOut(**result.as_dict())


# ── 提交时的组装（与 /tracks 同一套校验与推断入口）──────────────────────────


def _times_from_row(row: Mapping[str, Any], received: Sequence[int]) -> list[datetime]:
    """还原逐帧时间。

    **拿不到就是拿不到**：宁可 422 让人补传，也不按「平均间隔」编一条时间序列——
    帧序校验（单调不减 + 非全同）正是靠真实时间戳成立的，编出来的等于把校验关掉。
    """
    stored: dict[str, Any] = dict(row.get("frame_times") or {})
    out: list[datetime] = []
    missing: list[int] = []
    for index in received:
        value = stored.get(str(index))
        if not value:
            missing.append(index)
            continue
        out.append(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    if missing:
        raise ValidationRejected(
            [
                Issue(
                    code="frame_times_missing",
                    field="frames",
                    reason=(
                        f"有 {len(missing)} 帧没有时间戳（首帧 #{missing[0]}），无法校验帧序"
                    ),
                    suggestion=(
                        "补传时带上 frame_times（与片内帧一一对应的 ISO8601 数组），"
                        "或重新开启上传时申报整段时间；服务端不会替帧编时间。"
                    ),
                )
            ]
        )
    return out


def _extracted_from_paths(
    paths: Sequence[Path], times: Sequence[datetime], resize_json: str | None = None
):
    """把暂存目录里的帧拼成与 ``/tracks`` 同形的 :class:`ExtractedFrames`。

    ``resize_json`` 是**收第一片时**记下的缩放结论（task 35）：分片路上帧在收片时就已
    按同一策略缩到规范尺寸，提交时把当时的结论原样带回来，而不是拿已经缩过的帧反推
    「原始大概多大」——那是猜，猜出来的相机尺寸会一路流进响应和会话表。
    """
    from .resize import ResizePlan, ResizeReport
    from .zip_frames import ExtractedFrames

    report: ResizeReport | None = None
    payload = json.loads(resize_json) if resize_json else None
    if payload:
        plan = ResizePlan(
            original_size=(int(payload["camera_width"]), int(payload["camera_height"])),
            output_size=(
                int(payload["canonical_width"]),
                int(payload["canonical_height"]),
            ),
            scale=float(payload["scale"]),
            reason=payload.get("reason"),
        )
        report = ResizeReport(
            plan=plan,
            resized_frames=int(payload.get("resized_frames") or 0),
            total_frames=len(paths),
        )
    return ExtractedFrames(
        names=tuple(path.name for path in paths),
        paths=tuple(paths),
        times=tuple(times),
        total_bytes=sum(path.stat().st_size for path in paths),
        resize=report,
    )


def _meta_from_row(row: Mapping[str, Any]) -> UploadMeta:
    """从 ``upload_chunk`` 还原业务头（袋位声明原样喂回，不重新编号）。"""
    slots = tuple(
        SlotSpec(
            bag_seq=str(item["bag_seq"]),
            slot_index=int(item["slot_index"]),
            bag_type_code=item.get("bag_type_code"),
            rate_text=item.get("rate_text"),
        )
        for item in row.get("slots") or []
    )
    captured = str(row["captured_at"])
    captured_dt = datetime.fromisoformat(captured.replace("Z", "+00:00"))
    return UploadMeta(
        rack_id=row["rack_id"],
        infusion_order_no=row["infusion_order_no"],
        bed_no=row["bed_no"],
        captured_at=captured,
        captured_dt=captured_dt,
        shift_key=row["shift_key"],
        slots=slots,
        client_event_id=row.get("client_event_id"),
    )