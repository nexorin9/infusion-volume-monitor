"""本地替身归档源：真 TCP + 真 HTTP Range 的公开数据集归档服务器（**测试数据层**）。

为什么不用 mock：取数脚本的价值全在「按字节区间从远端 ZIP 里抽出该抽的成员」，
中央目录解析、Range 语义、CRC32 校验、断点续传这些都得真过一遍 socket 才能验。
用假响应对象只能验「我们调用了 read_range」，验不出服务器不支持 Range、206 状态码
写错、区间算错一位这类真会在病区机房里暴露的问题。

它只在测试里出现：**不是**最终数据来源，也不代表任何真实归档的行为。真实公开归档
本轮由 ``scripts/fetch_vos_subset.py`` 实连过（见 progress.txt），替身只让用例不必
每次都联网。

能力：记录每次请求与实际下发的字节区间（用来断言"只下了该下的"）、可按成员名注入
坏字节（用来制造 CRC 校验失败）、可整档停服（用来制造断网）。
"""

from __future__ import annotations

import io
import re
import threading
import zipfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from infusion_monitor.vos_dataset import ZipEntry

__all__ = ["VosArchiveStub", "build_stub_archive", "vos_archive_stub"]


def _jpeg_bytes(seed: int, size: tuple[int, int] = (48, 36)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (seed % 255, (seed * 7) % 255, (seed * 13) % 255)).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _png_bytes(seed: int, size: tuple[int, int] = (48, 36)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("P", size, 0)
    for y in range(size[1]):
        for x in range(size[0]):
            if (x + y + seed) % 3 == 0:
                img.putpixel((x, y), 1 + seed % 2)
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_stub_archive(sequences: dict[str, int], *, archive_name: str = "stub.zip") -> bytes:
    """造一个与公开归档同构的小 ZIP：``DAVIS/{JPEGImages,Annotations}/480p/<seq>/%05d.*``。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("DAVIS/README.md", b"stub archive for tests\n")
        for seq, frames in sequences.items():
            for i in range(frames):
                zf.writestr(f"DAVIS/JPEGImages/480p/{seq}/{i:05d}.jpg", _jpeg_bytes(i + len(seq)))
                zf.writestr(f"DAVIS/Annotations/480p/{seq}/{i:05d}.png", _png_bytes(i + len(seq)))
    return buf.getvalue()


@dataclass
class VosArchiveStub:
    """跑在真实端口上的归档替身。"""

    payload: bytes
    corrupt_members: set[str] = field(default_factory=set)
    fail_ranges: bool = False

    _server: ThreadingHTTPServer | None = field(default=None, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    requests: list[dict] = field(default_factory=list)
    bytes_served: int = 0
    entries: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        with zipfile.ZipFile(io.BytesIO(self.payload)) as zf:
            self.entries = {
                info.filename: ZipEntry(
                    name=info.filename,
                    compress_type=info.compress_type,
                    compressed_size=info.compress_size,
                    file_size=info.file_size,
                    crc32=info.CRC,
                    header_offset=info.header_offset,
                )
                for info in zf.infolist()
            }
        self._payload_starts = {
            name: entry.header_offset + 30 + _name_extra_len(self.payload, entry.header_offset)
            for name, entry in self.entries.items()
            if not entry.is_dir
        }

    # ---------------------------------------------------------------- 生命周期
    def start(self) -> "VosArchiveStub":
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # noqa: D102 - 静默，避免测试输出噪声
                return

            def do_HEAD(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Length", str(len(stub.payload)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

            def do_GET(self):  # noqa: N802
                if stub.fail_ranges:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                rng = self.headers.get("Range")
                stub.requests.append({"method": "GET", "range": rng})
                total = len(stub.payload)
                if rng:
                    m = re.match(r"bytes=(\d+)-(\d*)", rng)
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else total - 1
                    end = min(end, total - 1)
                    chunk = bytearray(stub.payload[start : end + 1])
                    bad_at = stub.corrupt_offset(start, end)
                    if bad_at is not None and chunk:
                        chunk[bad_at] ^= 0xFF  # 制造 CRC 不符
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(bytes(chunk))
                    stub.bytes_served += len(chunk)
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(total))
                    self.end_headers()
                    self.wfile.write(stub.payload)
                    stub.bytes_served += total

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # ---------------------------------------------------------------- 断言辅助
    @property
    def url(self) -> str:
        assert self._server is not None, "替身未启动"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/DAVIS-2017-stub.zip"

    def member_at(self, offset: int) -> str | None:
        for name, entry in self.entries.items():
            if entry.is_dir:
                continue
            start = self._payload_starts[name]
            if start <= offset < start + entry.compressed_size:
                return name
        return None

    def corrupt_offset(self, start: int, end: int) -> int | None:
        """区间内属于"坏成员"的字节（相对区间起点的下标）；没有则 None。

        按**重叠**判定而非"区间起点属于谁"：取数脚本会把相邻成员并成一段块读，
        坏字节出现在块中间也必须照样被投毒，否则用例测不到真实路径。
        """
        for name in self.corrupt_members:
            entry = self.entries[name]
            pstart = self._payload_starts[name]
            pend = pstart + entry.compressed_size - 1
            lo, hi = max(start, pstart), min(end, pend)
            if lo <= hi:
                rel = lo - start
                # 往后挪一点，避免正好落在压缩流头部把解压直接打崩（要的是 CRC 不符）
                return min(rel + entry.compressed_size // 2, hi - start)
        return None

    def frame_bytes_served(self) -> int:
        """实际下发里与「帧/标注成员负载区间」重叠的字节数（= 真下了多少数据内容）。"""
        total = 0
        for req in self.requests:
            if req["method"] != "GET" or not req["range"]:
                total += len(self.payload)  # 整包 GET
                continue
            m = re.match(r"bytes=(\d+)-(\d*)", req["range"])
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else len(self.payload) - 1
            for name, entry in self.entries.items():
                if entry.is_dir:
                    continue
                pstart = self._payload_starts[name]
                pend = pstart + entry.compressed_size - 1
                total += max(0, min(end, pend) - max(start, pstart) + 1)
        return total

    def reset_counters(self) -> None:
        self.requests.clear()
        self.bytes_served = 0


def _name_extra_len(archive: bytes, header_offset: int) -> int:
    import struct

    head = archive[header_offset : header_offset + 30]
    nlen, elen = struct.unpack("<HH", head[26:30])
    return nlen + elen


@pytest.fixture
def vos_archive_stub():
    """默认 4 个序列、每序列 5 帧的小归档替身（足够跑通全链路，又不拖慢用例）。"""
    stub = VosArchiveStub(payload=build_stub_archive({"alpha": 5, "beta": 5, "gamma": 4, "delta": 3})).start()
    try:
        yield stub
    finally:
        stub.stop()


def disk_files(root: Path) -> list[Path]:
    """列出落盘目录下全部文件（含 .part / manifest），供"没有伪造数据"类断言使用。"""
    return sorted(p for p in root.rglob("*") if p.is_file())