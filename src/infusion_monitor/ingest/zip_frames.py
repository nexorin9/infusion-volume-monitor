"""帧 zip 的安全解包。

入站 zip 是**外部不可信输入**，这里只做四件事，任何一件不满足就 422 退回：

1. **成员命名**：只接受 ``frame_%05d.jpg``（zip 根下），拒收 ``__MACOSX/`` 等杂项；
2. **路径安全**：拒绝绝对路径、``..`` 穿越、反斜杠路径与符号链接成员；
3. **体积上限**：累计解压字节数封顶，防 zip 炸弹；
4. **落盘命名**：一律用规范名写盘，不使用 zip 内自带的名字。

帧的**目录约定**沿用源产品的 ``{video_id}/{frame_id}`` 形态：一次上传 = 一个
``upload_key`` 目录，同 ``rack_id + shift_key + infusion_order_no`` 的重传落到同一
目录并覆盖（``bag_session`` 侧同步走覆盖式更新，见 ``store/repositories.py``）。
"""

from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from ..errors import Issue, ValidationRejected
from .resize import ResizePolicy, ResizeReport, resize_frame_file
from .validate import FRAME_NAME_PATTERN, MAX_FRAMES

__all__ = [
    "MAX_UNCOMPRESSED_BYTES",
    "ChunkMember",
    "ExtractedFrames",
    "upload_key",
    "chunk_members",
    "extract_frames",
    "safe_member_name",
]

#: 单次上传解压后总字节上限（1200 帧 × 每帧 400KB ≈ 480MB，留一点余量）
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")
_READ_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class ExtractedFrames:
    """一次解包的结果。"""

    names: tuple[str, ...]          # 规范名，按帧号升序
    paths: tuple[Path, ...]         # 落盘绝对/相对路径，与 names 同序
    times: tuple[datetime, ...]     # 每帧时间（zip 成员时间戳，UTC 挂 naive）
    total_bytes: int
    #: 入站缩放结论（task 35）；未走缩放策略时为 None。**以首帧为准**——
    #: 帧序列本来就该同尺寸，真混了尺寸时后几帧的面积比不可比，由 validate 那层去判。
    resize: "ResizeReport | None" = None

    @property
    def frame_count(self) -> int:
        return len(self.names)


def upload_key(rack_id: str, infusion_order_no: str, shift_key: str) -> str:
    """一次上传的稳定目录名（同 rack + 轮次 + 单号必得同名，重传即覆盖）。

    目录名用可读前缀 + 短摘要：前缀方便运维在磁盘上认出是哪一袋，摘要保证
    截断后不会撞车。
    """
    parts = [rack_id, shift_key, infusion_order_no]
    stem = "_".join(_SAFE_SEGMENT.sub("-", part).strip("-") for part in parts)
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    stem = "-".join(filter(None, (stem[:80].strip("-"), digest)))
    return stem


def _reject(code: str, reason: str, suggestion: str) -> ValidationRejected:
    return ValidationRejected(
        [
            Issue(
                code=code,
                field="frames",
                reason=reason,
                suggestion=suggestion,
            )
        ],
        error_code="FRAMES_ZIP_REJECTED",
    )


def safe_member_name(raw: str) -> str | None:
    """成员名安全且符合约定则返回规范名，否则返回 None。"""
    if not raw or raw != raw.strip():
        return None
    if "/" in raw or "\\" in raw:
        return None
    if raw in (".", "..") or raw.startswith("."):
        return None
    if FRAME_NAME_PATTERN.match(raw) is None:
        return None
    return raw


def _check_zip_structure(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    if not infos:
        raise _reject(
            "frames_zip_empty",
            "frames zip 里没有任何成员",
            "请确认打包时选中了这一段短片导出的全部帧，而不是空目录。",
        )
    if len(infos) > MAX_FRAMES:
        raise _reject(
            "frames_too_many",
            f"frames zip 内有 {len(infos)} 个成员，超过上限 {MAX_FRAMES}",
            "请只上传这一轮巡视的短片（建议 20–120 帧），超长视频请分段拍摄。",
        )

    total = sum(info.file_size for info in infos)
    if total > MAX_UNCOMPRESSED_BYTES:
        raise _reject(
            "frames_zip_too_large",
            f"frames zip 解压后约 {total / 1024 / 1024:.0f}MB，超过上限 "
            f"{MAX_UNCOMPRESSED_BYTES / 1024 / 1024:.0f}MB",
            "请降低拍摄分辨率或缩短时长后重新上传。",
        )

    unsafe = [i.filename for i in infos if safe_member_name(i.filename) is None]
    if unsafe:
        shown = "、".join(unsafe[:3])
        raise _reject(
            "frame_member_unsafe",
            f"frames zip 含 {len(unsafe)} 个不符合约定的成员（如 {shown}）",
            "zip 内只允许 frame_00000.jpg 这类连续命名的帧图片，且不能带目录层级；"
            "请去掉手机打包时附带的 __MACOSX 等杂项后重新上传。",
        )

    symlinks = [i.filename for i in infos if (i.external_attr >> 16) & 0o170000 == 0o120000]
    if symlinks:
        raise _reject(
            "frame_member_symlink",
            f"frames zip 含符号链接成员：{'、'.join(symlinks[:3])}",
            "请用普通图片打包，不要在 zip 里放软链接或目录链接。",
        )


def _zip_member_time(info: zipfile.ZipInfo) -> datetime:
    """DOS 时间戳（2 秒精度）转 datetime；早于 1980 的按 1980 起算。"""
    try:
        year, month, day, hour, minute, second = info.date_time
    except (ValueError, TypeError):  # pragma: no cover - 结构损坏的 zip
        return datetime(1980, 1, 1)
    second = max(0, (second // 2) * 2)
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return datetime(1980, 1, 1) + timedelta(seconds=info.CRC & 0xFFFF)


def extract_frames(
    zip_path: Path, dest_dir: Path, *, resize_policy: ResizePolicy | None = None
) -> ExtractedFrames:
    """把帧 zip 安全解包到 ``dest_dir``，返回按帧号排序的结果。

    重复上传同一段（同一 ``upload_key``）时先清掉上一轮遗留的帧文件，再写新的——
    覆盖而非追加，避免磁盘上留下不属于本次上传的旧帧。

    **落盘即规范化（task 35）**：给了 ``resize_policy`` 就在**写这一帧的同时**决定要不要
    缩到长边上限（只读 JPEG 头，不额外解码像素），缩了才重写、没缩一个字节都不碰。
    于是「1200 帧 × 4000px」的手机短片在盘上、在内存里、进了几何核的每一处都已经是
    规范帧，而下游（校验、预检、传播、标定）**一处都不用改**——它们读到的从来就是缩放后
    的帧，几何与体积是比例量，数值一字不变。
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise _reject(
            "frames_zip_corrupt",
            f"frames 不是可读的 zip（{type(exc).__name__}）",
            "请重新打包这一段短片的帧；上传中途断开的文件会被判为损坏。",
        ) from exc

    with archive:
        _check_zip_structure(archive)
        infos = sorted(archive.infolist(), key=lambda i: FRAME_NAME_PATTERN.match(i.filename).group(1))

        incoming = {info.filename for info in infos}
        for stale in dest_dir.glob("frame_*.jpg"):
            if stale.name not in incoming:
                stale.unlink()

        names: list[str] = []
        paths: list[Path] = []
        times: list[datetime] = []
        total = 0
        resized_frames = 0
        first_plan = None
        for info in infos:
            target = dest_dir / info.filename
            with archive.open(info) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink, _READ_CHUNK)
            if resize_policy is not None:
                plan = resize_frame_file(target, resize_policy)
                if plan.resized:
                    resized_frames += 1
                if first_plan is None:
                    first_plan = plan
            total += target.stat().st_size
            names.append(info.filename)
            paths.append(target)
            times.append(_zip_member_time(info))

    report = (
        ResizeReport(plan=first_plan, resized_frames=resized_frames, total_frames=len(names))
        if first_plan is not None
        else None
    )
    return ExtractedFrames(
        names=tuple(names),
        paths=tuple(paths),
        times=tuple(times),
        total_bytes=total,
        resize=report,
    )


def frame_name_sequence(count: int) -> Sequence[str]:
    """第 i 帧的规范名（测试与脚本生成样例 zip 用）。"""
    return [f"frame_{index:05d}.jpg" for index in range(count)]


@dataclass(frozen=True)
class ChunkMember:
    """分片里的一个帧成员（task 32）。

    ``index`` 是**整段的全局帧号**，``number`` 是成员名里写的那个号——
    客户端用片内编号时两者相差一个 ``frame_offset``，断点查询与去重一律用全局号。
    """

    index: int
    number: int
    name: str
    time_iso: str


def chunk_members(zip_path: Path, *, frame_offset: int | None = None) -> tuple[ChunkMember, ...]:
    """读出分片内的帧成员（走**与整段上传同一套**结构校验）。

    安全约束一字不差：成员命名、路径穿越、符号链接、zip 炸弹四道闸都复用
    :func:`_check_zip_structure`——分片同样是外部不可信输入，放宽任何一条都等于
    给续传开一个后门。
    """
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise _reject(
            "chunk_not_readable",
            f"分片不是可读的 zip（{type(exc).__name__}）",
            "上传中途断开的分片会被判为损坏；请重传这一片，已收到的分片不受影响。",
        ) from exc

    with archive:
        _check_zip_structure(archive)
        infos = sorted(
            archive.infolist(),
            key=lambda i: int(FRAME_NAME_PATTERN.match(i.filename).group(1)),
        )
        offset = int(frame_offset or 0)
        return tuple(
            ChunkMember(
                index=int(FRAME_NAME_PATTERN.match(info.filename).group(1)) + offset,
                number=int(FRAME_NAME_PATTERN.match(info.filename).group(1)),
                name=info.filename,
                time_iso=_zip_member_time(info).isoformat(timespec="seconds"),
            )
            for info in infos
        )