"""公开 VOS 子集的脚本化获取：远端 ZIP 中央目录直读 + 按序列断点续传 + 完整性校验。

真实院内短片不可得（须医院授权采集），本模块负责把**公开**视频分割语料的一个
子集可复现地取到本地，作为逐帧身份传播的评测基础（IoU / J&F 的输入，见
``docs/benchmark-vos.md``）。它不做任何"数据补齐"：断网或校验不通过时如实降级为
``skipped`` 并把原因写进文档顶部，**绝不**用合成帧、占位图或拷贝冒充真数据。

为什么不整包下载再解压：完整归档约 794 MB，而按序列抽取只需取「中央目录里那几个
成员的字节区间」。本模块先读 ZIP 中央目录（尾部 EOCD → 中央目录），拿到每个成员
的 ``header_offset / compressed_size / crc32 / compress_type``，再逐个做 HTTP Range
取字节码。因此 ``--sequences`` 限定规模是真的少下，而不是下完再删。

断点续传：每个文件先写 ``<目标>.part``，追加写而不是重头再来；重启后按已有字节数
续写（``Range: bytes=<已写>-<尾>``）。校验：ZIP 头里带每个成员的 CRC32，落盘后逐个
重算比对；再按真实解码器打开一遍确认可解码与图像尺寸，写进 manifest。

完整性对账：``manifest.json`` 逐文件记录 ``path / bytes / crc32 / source``，
``tests/test_fetch_subset.py`` 拿它与磁盘实际内容逐条对账。
"""

from __future__ import annotations

import json
import os
import struct
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence
from zlib import crc32

__all__ = [
    "DEFAULT_ARCHIVE_URL",
    "DEFAULT_ROOT",
    "IMAGE_PREFIX",
    "ANNOTATION_PREFIX",
    "DATASET_NOTE",
    "ZipEntry",
    "ZipIndex",
    "HttpRangeReader",
    "RangeUnavailable",
    "SequencePlan",
    "SubsetPlan",
    "FetchReport",
    "read_central_directory",
    "plan_subset",
    "fetch_subset",
    "render_status_block",
    "update_docs_banner",
]

#: 公开视频分割基准的官方归档地址（DAVIS 2017 trainval 480p，约 794 MB）。
DEFAULT_ARCHIVE_URL = (
    "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
)

#: 落盘根目录（.gitignore 已排除 data/）。
DEFAULT_ROOT = "data/vos_subset"

#: 归档内两类成员的路径前缀（实测自远端中央目录）。
IMAGE_PREFIX = "DAVIS/JPEGImages/480p/"
ANNOTATION_PREFIX = "DAVIS/Annotations/480p/"

#: 归档自带 README/SOURCES 里的数据集与许可说明位置（不下载，只在文档里引用）。
DATASET_NOTE = (
    "DAVIS 2017 trainval 480p · DAVIS 数据集由 ETH Zürich CVG 与日内瓦大学发布，"
    "许可与引用要求见归档内 README.md / SOURCES.md；本仓库只放本地评测副本，"
    "不 redistribution。"
)

_EOCD_SIG = b"PK\x05\x06"
_CD_SIG = b"PK\x01\x02"
_LOCAL_SIG = b"PK\x03\x04"
_LOCAL_FIXED = 30  # 本地文件头固定长度（不含文件名与扩展字段）
_USER_AGENT = "infusion-volume-monitor/0.1 (+vos subset fetcher)"


class RangeUnavailable(RuntimeError):
    """远端不支持 Range（或不支持 HEAD），只能整包下载。"""


@dataclass(frozen=True)
class ZipEntry:
    """ZIP 中央目录里的一个成员。"""

    name: str
    compress_type: int
    compressed_size: int
    file_size: int
    crc32: int
    header_offset: int

    @property
    def is_dir(self) -> bool:
        return self.name.endswith("/")


@dataclass
class ZipIndex:
    """远端 ZIP 的中央目录索引。"""

    archive_size: int
    entries: dict[str, ZipEntry] = field(default_factory=dict)

    def under(self, prefix: str) -> list[ZipEntry]:
        return [e for name, e in self.entries.items() if name.startswith(prefix) and not e.is_dir]

    def sequence_of(self, entry: ZipEntry, prefix: str) -> str:
        return entry.name[len(prefix) :].split("/", 1)[0]


class HttpRangeReader:
    """按字节区间读远端文件的最小 HTTP 客户端（标准库，不引入新依赖）。"""

    def __init__(self, url: str, *, timeout: float = 30.0, retries: int = 2,
                 backoff: float = 1.5, opener: Callable[[urllib.request.Request], object] | None = None):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._open = opener or (lambda req: urllib.request.urlopen(req, timeout=timeout))
        self.bytes_served = 0
        self.requests = 0

    def _request(self, start: int | None, end: int | None) -> bytes:
        headers = {"User-Agent": _USER_AGENT}
        if start is not None:
            headers["Range"] = f"bytes={start}-{'' if end is None else end}"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, headers=headers)
                with self._open(req) as resp:  # type: ignore[union-attr]
                    data = resp.read()
                self.requests += 1
                self.bytes_served += len(data)
                return data
            except urllib.error.HTTPError as exc:
                if exc.code in (416, 501) and start is not None:
                    raise RangeUnavailable(f"{self.url} 不支持 Range（HTTP {exc.code}）") from exc
                last = exc
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(self.backoff**attempt)
        raise RuntimeError(f"取字节区间失败：{headers.get('Range', 'HEAD')} → {last!r}")

    def archive_size(self) -> int:
        try:
            req = urllib.request.Request(self.url, method="HEAD", headers={"User-Agent": _USER_AGENT})
            with self._open(req) as resp:  # type: ignore[union-attr]
                length = resp.headers.get("Content-Length")  # type: ignore[union-attr]
                self.requests += 1
            if length:
                return int(length)
        except Exception as exc:  # noqa: BLE001 - 有些站点禁 HEAD；本模块依赖 Range，如实报不可用
            raise RangeUnavailable(f"{self.url} 的 HEAD 不可用（{type(exc).__name__}: {exc}）") from exc
        raise RangeUnavailable(f"{self.url} 未给 Content-Length，无法定位中央目录")

    def read_range(self, start: int, end: int) -> bytes:
        if end < start:
            return b""
        return self._request(start, end)

    def stream_range(self, start: int, end: int):
        """流式读一个字节区间（供"一段块里切多个成员"用，避免逐成员建连接）。"""
        req = urllib.request.Request(
            self.url,
            headers={"User-Agent": _USER_AGENT, "Range": f"bytes={start}-{end}"},
        )
        resp = self._open(req)
        self.requests += 1
        return _RangeStream(resp, start, self)

    def read_tail(self, size: int) -> bytes:
        if size <= 0:
            return b""
        size_total = self.archive_size()
        start = max(0, size_total - size)
        return self.read_range(start, size_total - 1)


class _RangeStream:
    """带位置记账的区间流：短读即报错，不让"少几个字节"悄悄变成坏文件。"""

    def __init__(self, resp, start: int, reader: "HttpRangeReader"):
        self._resp = resp
        self._pos = start
        self._reader = reader

    def read_exact(self, size: int, what: str) -> bytes:
        buf = bytearray()
        while len(buf) < size:
            block = self._resp.read(size - len(buf))
            if not block:
                raise RuntimeError(f"{what} 在偏移 {self._pos} 处提前结束：只读到 {len(buf)}/{size} 字节")
            buf.extend(block)
            self._pos += len(block)
            self._reader.bytes_served += len(block)
        return bytes(buf)

    @property
    def position(self) -> int:
        return self._pos

    def close(self) -> None:
        self._resp.close()

    def __enter__(self) -> "_RangeStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _parse_central_directory(cd: bytes) -> dict[str, ZipEntry]:
    entries: dict[str, ZipEntry] = {}
    off = 0
    total = len(cd)
    while off + 46 <= total and cd[off : off + 4] == _CD_SIG:
        method = struct.unpack("<H", cd[off + 10 : off + 12])[0]
        crc = struct.unpack("<I", cd[off + 16 : off + 20])[0]
        csize, usize = struct.unpack("<II", cd[off + 20 : off + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[off + 28 : off + 34])
        header_offset = struct.unpack("<I", cd[off + 42 : off + 46])[0]
        name = cd[off + 46 : off + 46 + nlen].decode("utf-8", "replace")
        entries[name] = ZipEntry(
            name=name,
            compress_type=method,
            compressed_size=csize,
            file_size=usize,
            crc32=crc,
            header_offset=header_offset,
        )
        off += 46 + nlen + elen + clen
    if not entries:
        raise RuntimeError("中央目录解析为空：归档格式可能不是普通 ZIP")
    return entries


def read_central_directory(url: str = DEFAULT_ARCHIVE_URL, *, reader: HttpRangeReader | None = None,
                           probe_size: int = 64) -> ZipIndex:
    """读远端 ZIP 的中央目录，返回成员索引（不下载任何帧）。

    先只取尾部 64 字节找 EOCD（EOCD 里直接给出中央目录的偏移与长度），再**精确**取
    中央目录那几百 KB——不回读整段尾部，因此一个帧负载字节都不会被顺带下下来。
    归档带注释（EOCD 尾部 comment 字段）导致 64 字节不够时，退回读整段尾部再定位。
    """
    reader = reader or HttpRangeReader(url)
    size = reader.archive_size()
    eocd = _locate_eocd(reader, size, probe_size)
    cd_size, cd_offset = struct.unpack("<II", eocd[12:20])
    if cd_size == 0:
        raise RuntimeError("中央目录长度为 0：归档是空的或已损坏")
    cd = reader.read_range(cd_offset, cd_offset + cd_size - 1)
    if len(cd) != cd_size:
        raise RuntimeError(f"中央目录长度不符：期望 {cd_size} 字节，实得 {len(cd)}")
    return ZipIndex(archive_size=size, entries=_parse_central_directory(cd))


def _locate_eocd(reader: HttpRangeReader, size: int, probe_size: int) -> bytes:
    """在归档尾部定位 EOCD（22 字节固定头 + 可变长注释）。"""
    for window in (probe_size, 64 * 1024, size):
        if window <= 0 or window > size:
            window = size
        chunk = reader.read_range(size - window, size - 1)
        pos = chunk.rfind(_EOCD_SIG)
        if pos >= 0 and len(chunk) - pos >= 22:
            return chunk[pos : pos + 22]
        if window == size:
            break
    raise RuntimeError("尾部找不到 EOCD：不是 ZIP 归档或已损坏")


# --------------------------------------------------------------------------- 选片


@dataclass(frozen=True)
class SequencePlan:
    """一个序列的取数计划。"""

    name: str
    frames: int
    image_bytes: int
    annotation_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.image_bytes + self.annotation_bytes

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "frames": self.frames,
            "image_bytes": self.image_bytes,
            "annotation_bytes": self.annotation_bytes,
            "total_bytes": self.total_bytes,
        }


@dataclass
class SubsetPlan:
    """选片结果 + 门禁是否达成（不达成也如实记 shortfall，不静默降门槛）。"""

    sequences: list[SequencePlan]
    requested_min_frames: int
    max_sequences: int | None
    min_frames_gate_satisfied: bool
    shortfall_reason: str | None
    relaxed: bool
    corpus_longest_sequence: int | None = None
    corpus_sequences: int | None = None

    @property
    def satisfied(self) -> bool:
        """门禁未达成且未显式放松 → 不可取（调用方据此降级）。"""
        return self.min_frames_gate_satisfied or self.relaxed

    @property
    def total_bytes(self) -> int:
        return sum(s.total_bytes for s in self.sequences)

    @property
    def min_frames(self) -> int:
        return min((s.frames for s in self.sequences), default=0)

    @property
    def total_frames(self) -> int:
        return sum(s.frames for s in self.sequences)

    def to_json(self) -> dict:
        return {
            "sequences": [s.to_json() for s in self.sequences],
            "requested_min_frames": self.requested_min_frames,
            "effective_min_frames": self.min_frames,
            "max_sequences": self.max_sequences,
            "relaxed": self.relaxed,
            "gate_satisfied": self.min_frames_gate_satisfied,
            "shortfall_reason": self.shortfall_reason,
            "corpus_longest_sequence": self.corpus_longest_sequence,
            "corpus_sequences": self.corpus_sequences,
            "planned_bytes": self.total_bytes,
        }


def plan_subset(index: ZipIndex, *, sequences: Sequence[str] | None = None,
                min_frames: int = 150, max_sequences: int | None = 10,
                max_bytes: int | None = None, relax_min_frames: bool = False) -> SubsetPlan:
    """从中央目录里选序列。

    ``min_frames`` 是「每序列帧数」门禁（默认 150）。公开语料达不到时**不静默降门槛**：
    计划里如实记 ``gate_satisfied=false`` 与 shortfall 原因；只有调用方显式传
    ``relax_min_frames=True`` 才会退而取最长的那些序列，并在 manifest 与文档里留痕。
    """
    images: dict[str, list[ZipEntry]] = {}
    for entry in index.under(IMAGE_PREFIX):
        images.setdefault(index.sequence_of(entry, IMAGE_PREFIX), []).append(entry)
    annotations: dict[str, list[ZipEntry]] = {}
    for entry in index.under(ANNOTATION_PREFIX):
        annotations.setdefault(index.sequence_of(entry, ANNOTATION_PREFIX), []).append(entry)

    candidates: list[SequencePlan] = []
    for name, jpgs in images.items():
        anns = annotations.get(name) or []
        if not anns:  # 无标注的序列不能算评测基础，直接排除
            continue
        candidates.append(
            SequencePlan(
                name=name,
                frames=len(jpgs),
                image_bytes=sum(e.compressed_size for e in jpgs),
                annotation_bytes=sum(e.compressed_size for e in anns),
            )
        )
    corpus_longest = max((c.frames for c in candidates), default=0)
    reason: str | None = None

    if sequences:
        wanted = list(dict.fromkeys(sequences))
        by_name = {c.name: c for c in candidates}
        missing = [s for s in wanted if s not in by_name]
        if missing:
            raise KeyError(f"归档里没有这些序列：{'、'.join(missing)}")
        picked = [by_name[s] for s in wanted]
        gate_ok = all(p.frames >= min_frames for p in picked)
        reason = None if gate_ok else (
            f"指定序列里最短 {min(p.frames for p in picked)} 帧 < 门禁 {min_frames} 帧"
        )
    else:
        eligible = [c for c in candidates if c.frames >= min_frames]
        gate_ok = bool(eligible)
        if not eligible:
            reason = (
                f"公开语料无法满足门禁：归档 {len(candidates)} 个序列中最长的也只有 "
                f"{corpus_longest} 帧，少于要求的每序列 {min_frames} 帧"
                f"（实测自远端 ZIP 中央目录，非估计值）"
            )
        # 门禁未达成时取最长的一批，让"放松"变成显式决定而不是隐形替换
        ordered = eligible or sorted(candidates, key=lambda c: (-c.frames, c.name))
        picked = ordered
        if max_sequences:
            picked = picked[:max_sequences]
        if max_bytes:
            trimmed: list[SequencePlan] = []
            used = 0
            for cand in picked:
                if used + cand.total_bytes > max_bytes and trimmed:
                    break
                trimmed.append(cand)
                used += cand.total_bytes
            picked = trimmed

    return SubsetPlan(
        sequences=picked,
        requested_min_frames=min_frames,
        max_sequences=max_sequences,
        min_frames_gate_satisfied=gate_ok,
        shortfall_reason=reason,
        relaxed=relax_min_frames,
        corpus_longest_sequence=corpus_longest,
        corpus_sequences=len(candidates),
    )


# --------------------------------------------------------------------------- 取数


@dataclass
class FetchReport:
    """一次取数的完整结果；``status`` 只有 ready / skipped 两态。"""

    status: str
    reason: str | None = None
    archive_url: str = DEFAULT_ARCHIVE_URL
    root: str = ""
    plan: SubsetPlan | None = None
    files: list[dict] = field(default_factory=list)
    sequences: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    downloaded_files: int = 0
    reused_files: int = 0
    resumed_files: int = 0
    bytes_on_disk: int = 0
    verify: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_json(self) -> dict:
        return {
            "schema_version": 1,
            "generated_by": "scripts/fetch_vos_subset.py",
            "status": self.status,
            "skipped_reason": self.reason,
            "dataset": {
                "name": "DAVIS 2017 trainval 480p",
                "archive_url": self.archive_url,
                "license_note": DATASET_NOTE,
            },
            "root": self.root,
            "achieved": {
                "sequence_count": len(self.sequences),
                "frame_count": sum(s["frames"] for s in self.sequences),
                "min_frames_per_sequence": min((s["frames"] for s in self.sequences), default=0),
                "max_frames_per_sequence": max((s["frames"] for s in self.sequences), default=0),
                "downloaded_files": self.downloaded_files,
                "reused_files": self.reused_files,
                "resumed_files": self.resumed_files,
                "bytes_on_disk": self.bytes_on_disk,
                "elapsed_s": round(self.elapsed_s, 2),
            },
            "gate": self.plan.to_json() if self.plan else None,
            "sequences": self.sequences,
            "files": self.files,
            "failures": self.failures,
            "verify": self.verify,
        }


def _coalesce(entries: Sequence[ZipEntry], tolerance: int = 64 * 1024) -> list[list[ZipEntry]]:
    """把归档里**相邻**的成员并成一段块读，而不是一个成员一次请求。

    一个序列的帧在归档里是连着放的，逐个成员各发一次请求会让 946 帧变成 946 次往返
    （实测约 1 秒一次）。并成段块后同一序列通常只发一两次请求，块内逐成员解压校验。
    估算用中央目录里的名字长度 + 保守扩展字段上限；真正对齐与否在流里逐个断言，
    对不上就退回逐成员取（见 ``_fetch_run_with_fallback``），不会静默错位。
    """
    ordered = sorted(entries, key=lambda e: e.header_offset)
    runs: list[list[ZipEntry]] = []
    for entry in ordered:
        if runs:
            cur = runs[-1][-1]
            # 载荷尾的下界（未计本地头的扩展字段，故差值只会是 elen + 填充）
            lower_bound_end = cur.header_offset + _LOCAL_FIXED + len(cur.name.encode("utf-8")) + cur.compressed_size
            if 0 <= entry.header_offset - lower_bound_end <= tolerance:
                runs[-1].append(entry)
                continue
        runs.append([entry])
    return runs


def _fetch_run(reader: HttpRangeReader, entries: list[ZipEntry], root: Path) -> list[tuple[ZipEntry, str]]:
    """一段块里顺序切出多个成员（每成员各自解压 + CRC 校验后才落盘）。"""
    last_payload_start = _entry_payload_start(reader, entries[-1])
    end = last_payload_start + entries[-1].compressed_size - 1
    start = entries[0].header_offset
    results: list[tuple[ZipEntry, str]] = []
    with reader.stream_range(start, end) as stream:
        for entry in entries:
            if stream.position != entry.header_offset:
                raise RuntimeError(
                    f"块内位置错位：期望 {entry.name} 在 {entry.header_offset}，实际 {stream.position}"
                )
            head = stream.read_exact(_LOCAL_FIXED, entry.name)
            if head[:4] != _LOCAL_SIG:
                raise RuntimeError(f"{entry.name} 的本地文件头签名不对")
            nlen, elen = struct.unpack("<HH", head[26:30])
            stream.read_exact(nlen + elen, f"{entry.name} 文件名")
            raw = stream.read_exact(entry.compressed_size, entry.name)
            data = _decompress(raw, entry)
            if len(data) != entry.file_size:
                raise RuntimeError(f"{entry.name} 解压后长度 {len(data)} ≠ 声明 {entry.file_size}")
            actual = crc32(data) & 0xFFFFFFFF
            if actual != entry.crc32:
                raise RuntimeError(f"{entry.name} CRC32 不符：实得 {actual:#010x}，归档声明 {entry.crc32:#010x}")
            _write_member(root, entry, data)
            results.append((entry, "downloaded"))
    return results


def _write_member(root: Path, entry: ZipEntry, data: bytes) -> Path:
    dest = _local_path(root, entry.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staged = dest.with_suffix(dest.suffix + ".tmp")
    staged.write_bytes(data)
    os.replace(staged, dest)
    return dest


def _fetch_run_with_fallback(reader: HttpRangeReader, entries: list[ZipEntry], root: Path,
                             failures: list[dict]) -> list[tuple[ZipEntry, str]]:
    """块读失败（对齐不符、块内某成员坏掉）时退回逐成员取，不因为优化而取不到数。

    退回后仍逐成员单独记失败原因——只有真正有问题的成员进 failures，其余照常落盘。
    """
    if len(entries) == 1:
        entry = entries[0]
        dest = _local_path(root, entry.name)
        try:
            return [(entry, _fetch_entry(reader, entry, dest))]
        except Exception as exc:  # noqa: BLE001
            failures.append({"name": entry.name, "reason": f"{type(exc).__name__}: {exc}"})
            return []
    try:
        return _fetch_run(reader, entries, root)
    except Exception:  # noqa: BLE001 - 优化路径失败不是终局，退回保底路径
        out: list[tuple[ZipEntry, str]] = []
        for entry in entries:
            dest = _local_path(root, entry.name)
            try:
                out.append((entry, _fetch_entry(reader, entry, dest)))
            except Exception as exc:  # noqa: BLE001
                failures.append({"name": entry.name, "reason": f"{type(exc).__name__}: {exc}"})
        return out


def _entry_payload_start(reader: HttpRangeReader, entry: ZipEntry) -> int:
    head = reader.read_range(entry.header_offset, entry.header_offset + _LOCAL_FIXED - 1)
    if head[:4] != _LOCAL_SIG:
        raise RuntimeError(f"{entry.name} 的本地文件头签名不对")
    nlen, elen = struct.unpack("<HH", head[26:30])
    return entry.header_offset + _LOCAL_FIXED + nlen + elen


def _decompress(raw: bytes, entry: ZipEntry) -> bytes:
    if entry.compress_type == 0:
        return raw
    if entry.compress_type == 8:
        return zlib.decompress(raw, -15)
    raise RuntimeError(f"{entry.name} 用了不支持的压缩方式 {entry.compress_type}")


def _fetch_entry(reader: HttpRangeReader, entry: ZipEntry, dest: Path, *, chunk: int = 256 * 1024) -> str:
    """取一个成员到 ``dest``（含 .part 断点续传），返回 downloaded / resumed / reused。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and crc32(dest.read_bytes()) & 0xFFFFFFFF == entry.crc32:
        return "reused"
    part = dest.with_suffix(dest.suffix + ".part")
    payload_start = _entry_payload_start(reader, entry)
    data_start = payload_start
    data_end = payload_start + entry.compressed_size - 1

    resumed = False
    if part.exists() and part.stat().st_size:
        have = min(part.stat().st_size, entry.compressed_size)
        if have:
            data_start = payload_start + have
            resumed = True
    req = urllib.request.Request(
        reader.url,
        headers={"User-Agent": _USER_AGENT, "Range": f"bytes={data_start}-{data_end}"},
    )
    with reader._open(req) as resp:  # type: ignore[union-attr]
        with open(part, "ab" if resumed else "wb") as fh:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                fh.write(block)
                reader.bytes_served += len(block)
    reader.requests += 1

    payload = part.read_bytes()
    if len(payload) != entry.compressed_size:
        raise RuntimeError(f"{entry.name} 取回长度 {len(payload)} ≠ 声明 {entry.compressed_size}")
    data = _decompress(payload, entry)
    if len(data) != entry.file_size:
        raise RuntimeError(f"{entry.name} 解压后长度 {len(data)} ≠ 声明 {entry.file_size}")
    actual = crc32(data) & 0xFFFFFFFF
    if actual != entry.crc32:
        raise RuntimeError(f"{entry.name} CRC32 不符：实得 {actual:#010x}，归档声明 {entry.crc32:#010x}")
    # .part 存的是压缩字节流（续传要按字节接），落盘的必须是解压后的文件内容
    staged = dest.with_suffix(dest.suffix + ".tmp")
    staged.write_bytes(data)
    os.replace(staged, dest)
    part.unlink(missing_ok=True)
    return "resumed" if resumed else "downloaded"


def _local_path(root: Path, entry_name: str) -> Path:
    """归档内 ``DAVIS/JPEGImages/480p/<seq>/<f>`` → ``<root>/JPEGImages/480p/<seq>/<f>``。"""
    rel = entry_name.split("/", 1)[1] if entry_name.startswith("DAVIS/") else entry_name
    target = (root / rel).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise RuntimeError(f"成员路径越界：{entry_name}")
    return root / rel


def _verify_sequences(root: Path, planned: Sequence[SequencePlan], index: ZipIndex) -> dict:
    """解码器过一遍 + 首帧标注对账，写进 manifest 的 verify 段。"""
    from PIL import Image  # 延迟导入：仅校验阶段需要图像解码

    summary: dict = {"sequences": [], "decode_failures": []}
    for plan in planned:
        jpg_dir = _local_path(root, f"DAVIS{JPEG_PREFIX_SKIP}{plan.name}/")
        png_dir = _local_path(root, f"DAVIS{ANNOT_PREFIX_SKIP}{plan.name}/")
        jpgs = sorted(jpg_dir.glob("*.jpg"))
        pngs = sorted(png_dir.glob("*.png"))
        if not jpgs:
            summary["decode_failures"].append({"sequence": plan.name, "reason": "no_frames_on_disk"})
            continue
        width = height = None
        for path in jpgs:
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    width, height = img.size
            except Exception as exc:  # noqa: BLE001 - 记录即可，不掩盖
                summary["decode_failures"].append({"sequence": plan.name, "file": path.name, "reason": str(exc)})
                break
        ann_ok = False
        ann_size = None
        if pngs:
            try:
                with Image.open(pngs[0]) as img:
                    ann_size = img.size
                    img.verify()
                ann_ok = True
            except Exception as exc:  # noqa: BLE001
                summary["decode_failures"].append({"sequence": plan.name, "file": pngs[0].name, "reason": str(exc)})
        summary["sequences"].append(
            {
                "name": plan.name,
                "frames_on_disk": len(jpgs),
                "annotations_on_disk": len(pngs),
                "frame_size_px": [width, height],
                "annotation_size_px": list(ann_size) if ann_size else None,
                "first_frame": jpgs[0].name,
                "first_annotation": pngs[0].name if pngs else None,
                "has_first_frame_annotation": ann_ok,
            }
        )
    return summary


JPEG_PREFIX_SKIP = "/JPEGImages/480p/"
ANNOT_PREFIX_SKIP = "/Annotations/480p/"


def fetch_subset(*, url: str = DEFAULT_ARCHIVE_URL, root: Path | str = DEFAULT_ROOT,
                 sequences: Sequence[str] | None = None, min_frames: int = 150,
                 max_sequences: int | None = 10, max_bytes: int | None = None,
                 relax_min_frames: bool = False, timeout: float = 30.0,
                 retries: int = 2, dry_run: bool = False, max_failures: int = 3) -> FetchReport:
    """按计划取子集。断网 / 校验不过 → ``skipped``，不生成任何替代数据。"""
    started = time.time()
    root = Path(root)
    reader = HttpRangeReader(url, timeout=timeout, retries=retries)
    report = FetchReport(status="skipped", archive_url=url, root=str(root))

    try:
        index = read_central_directory(url, reader=reader)
    except Exception as exc:  # noqa: BLE001 - 断网/归档不可读，一律降级
        report.reason = f"读不到远端中央目录（{type(exc).__name__}: {exc}）"
        report.elapsed_s = time.time() - started
        _write_manifest(root, report)
        return report

    try:
        plan = plan_subset(
            index, sequences=sequences, min_frames=min_frames,
            max_sequences=max_sequences, max_bytes=max_bytes,
            relax_min_frames=relax_min_frames,
        )
    except KeyError as exc:
        report.reason = str(exc.args[0] if exc.args else exc)
        report.elapsed_s = time.time() - started
        _write_manifest(root, report)
        return report
    report.plan = plan

    if not plan.satisfied or not plan.sequences:
        report.reason = report.reason or plan.shortfall_reason or "没有选出任何序列"
        report.status = "skipped"
        report.elapsed_s = time.time() - started
        _write_manifest(root, report)
        return report

    if dry_run:
        report.status = "planned"
        report.reason = plan.shortfall_reason
        report.elapsed_s = time.time() - started
        return report

    root.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []

    for plan_seq in plan.sequences:
        wanted = [
            entry
            for prefix in (IMAGE_PREFIX, ANNOTATION_PREFIX)
            for entry in index.under(prefix)
            if index.sequence_of(entry, prefix) == plan_seq.name
        ]
        # 已在盘且 CRC32 对得上的直接算复用，不进任何请求
        pending: list[ZipEntry] = []
        for entry in wanted:
            dest = _local_path(root, entry.name)
            if dest.exists() and (crc32(dest.read_bytes()) & 0xFFFFFFFF) == entry.crc32:
                report.reused_files += 1
                report.files.append(_file_entry(dest, root, entry.crc32, "reused"))
            else:
                pending.append(entry)
        for run in _coalesce(pending):
            for entry, source in _fetch_run_with_fallback(reader, run, root, failures):
                report.downloaded_files += 1
                if source == "resumed":
                    report.resumed_files += 1
                report.files.append(
                    _file_entry(_local_path(root, entry.name), root, entry.crc32, source)
                )

    report.failures = failures
    report.verify = _verify_sequences(root, plan.sequences, index)
    verified_by_name = {item["name"]: item for item in report.verify.get("sequences", [])}
    merged: list[dict] = []
    for plan_seq in plan.sequences:
        item = plan_seq.to_json()
        item.update({k: v for k, v in verified_by_name.get(plan_seq.name, {}).items() if k != "name"})
        merged.append(item)
    report.sequences = merged
    report.bytes_on_disk = sum(f["bytes"] for f in report.files)
    if failures:
        report.status = "skipped"
        report.reason = (
            f"{len(failures)} 个成员取数或校验失败（如 {failures[0]['name']}: {failures[0]['reason']}）"
            "；已取到的部分保留在磁盘上，但本轮不视为可用评测基础"
        )
    else:
        report.status = "ready"
        report.reason = plan.shortfall_reason
    report.elapsed_s = time.time() - started
    _write_manifest(root, report)
    return report


def _file_entry(dest: Path, root: Path, declared_crc: int, source: str) -> dict:
    return {
        "path": str(dest.relative_to(root)),
        "bytes": dest.stat().st_size,
        "crc32": f"{crc32(dest.read_bytes()) & 0xFFFFFFFF:#010x}",
        "declared_crc32": f"{declared_crc:#010x}",
        "source": source,
    }


def _fetch_one(report: FetchReport, reader: HttpRangeReader, entry: ZipEntry, dest: Path,
               failures: list[dict], max_failures: int) -> None:
    try:
        source = _fetch_entry(reader, entry, dest)
    except Exception as exc:  # noqa: BLE001 - 逐个成员记原因
        failures.append({"name": entry.name, "reason": f"{type(exc).__name__}: {exc}"})
        return
    if source == "reused":
        report.reused_files += 1
    else:
        report.downloaded_files += 1
        if source == "resumed":
            report.resumed_files += 1
    report.files.append(_file_entry(dest, Path(report.root), entry.crc32, source))


def _write_manifest(root: Path, report: FetchReport) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    path.write_text(json.dumps(report.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- 文档顶部状态块

_BANNER_BEGIN = "<!-- vos-subset-status:begin -->"
_BANNER_END = "<!-- vos-subset-status:end -->"


def render_status_block(report: FetchReport) -> str:
    """生成写进 ``docs/benchmark-vos.md`` 顶部的状态块（skipped 也照写，如实留原因）。"""
    if report.status == "ready":
        seqs = report.sequences
        head = (
            f"> **公开 VOS 子集：可用** · {len(seqs)} 个序列 / "
            f"{sum(s['frames'] for s in seqs)} 帧 / {report.bytes_on_disk / 1e6:.1f} MB"
        )
    elif report.status == "planned":
        head = "> **公开 VOS 子集：仅试算（--dry-run），未落数据**"
    else:
        head = f"> **公开 VOS 子集：skipped（不可用）** · 原因：{report.reason}"

    lines = [_BANNER_BEGIN, head, ">"]
    if report.plan is not None:
        gate = "达成" if report.plan.min_frames_gate_satisfied else "未达成"
        lines.append(f"> - 每序列帧数门禁：要求 {report.plan.requested_min_frames} 帧 → **{gate}**")
        if report.plan.shortfall_reason:
            lines.append(f"> - 门禁缺口：{report.plan.shortfall_reason}")
        if report.plan.relaxed:
            lines.append("> - 取数方式：`--relax-min-frames` 显式放松后取语料内最长的一批序列（缺口如上，不静默替换）")
    if report.status == "ready":
        lines.append(f"> - 落盘位置：`{report.root}`（`data/` 已在 .gitignore 中排除，不入库）")
        lines.append(f"> - 取数统计：新取 {report.downloaded_files} 个文件 / 复用 {report.reused_files} 个 / "
                     f"断点续传 {report.resumed_files} 个；校验：逐文件 CRC32 对账 + 首帧标注对账 + 图像解码")
        lines.append(f"> - 本地取数耗时：{report.elapsed_s:.1f}s")
    lines.append(">")
    lines.append("> 复现命令见下文「获取子集」一节；断网时本块会如实改写为 `skipped` 并给出原因，")
    lines.append("> **不会**用合成帧或占位图冒充真数据（合成回归另见 `docs/synthetic-data.md`）。")
    lines.append(_BANNER_END)
    return "\n".join(lines)


def update_docs_banner(doc_path: Path | str, report: FetchReport) -> str:
    """把状态块写到文档顶部（已有则整块替换，保留正文）。"""
    path = Path(doc_path)
    block = render_status_block(report)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if _BANNER_BEGIN in text and _BANNER_END in text:
        start = text.index(_BANNER_BEGIN)
        end = text.index(_BANNER_END) + len(_BANNER_END)
        text = text[:start] + block + text[end:]
    else:
        text = block + "\n\n" + text if text.strip() else block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return block


def iter_plan_entries(index: ZipIndex, plan: SubsetPlan, root: Path) -> Iterable[tuple[ZipEntry, Path]]:
    """给测试与核对用：按计划列出 (成员, 落盘路径)。"""
    for seq in plan.sequences:
        for prefix in (IMAGE_PREFIX, ANNOTATION_PREFIX):
            for entry in sorted(index.under(prefix), key=lambda e: e.name):
                if index.sequence_of(entry, prefix) == seq.name:
                    yield entry, _local_path(root, entry.name)