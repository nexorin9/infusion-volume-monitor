"""本地替身对端：护理执行平台的**测试替身**，跑在真实 TCP 端口上。

为什么不是 ``httpx.MockTransport``：契约测试要验的是「出站客户端真的按 HTTP
协议把请求发出去了」——报头、JSON 体、超时、重试后的多次请求都得真的过一遍
socket。用内存传输层只能验「我们调用了 post」，验不出路径拼错、头没带上、
超时参数不生效这类真会在病区暴露的问题。

替身只出现在**测试数据层**：它不是最终对接边界，也不代表真实护理执行平台的
接口行为。真实端点本轮未实连（见 progress.txt）。

能力：记录每次请求（方法/路径/报头/体），可脚本化状态码序列、响应延迟与
**半开连接**（收到请求却把连接掐了不给应答），用来制造 5xx、超时、4xx 业务拒绝
与传输层四类失败。
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["HALF_CLOSE", "RecordedRequest", "NurseExecStub"]

#: ``next_response`` 的哨兵状态码：表示「对端收下了请求却把连接掐了，不给应答」。
#: 真实病区里这一形态来自对端进程重启、负载均衡重置空闲连接、连接池回收——
#: httpx 侧表现为 ``RemoteProtocolError: Server disconnected without sending a
#: response``，不是超时也不是连接失败。
HALF_CLOSE = -1


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes
    received_at: float = field(default_factory=time.time)

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    @property
    def idempotency_key(self) -> str:
        return self.headers.get("idempotency-key", "")

    @property
    def authorization(self) -> str:
        return self.headers.get("authorization", "")


class _QuietServer(ThreadingHTTPServer):
    """客户端超时断开后 handler 写响应会 BrokenPipeError；那不是缺陷，别刷屏。"""

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        return


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # 静音默认的 stderr 日志
    def log_message(self, *args: Any) -> None:  # noqa: D102
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        stub: NurseExecStub = self.server.stub  # type: ignore[attr-defined]
        record = RecordedRequest(
            method="POST",
            path=self.path,
            headers={k.lower(): v for k, v in self.headers.items()},
            body=body,
        )
        status, payload, delay = stub.next_response(record)
        if delay:
            time.sleep(delay)
        if status == HALF_CLOSE:
            self._half_close()
            return
        raw = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _half_close(self) -> None:
        """半开连接：请求已读走，连接直接掐掉，不回任何字节。

        客户端侧就是 ``http.client.RemoteDisconnected``，被 httpx 包成
        ``RemoteProtocolError``——「对端不等我把话说完就挂了」。
        """
        self.close_connection = True
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.connection.close()
        except OSError:
            pass

    def do_GET(self) -> None:  # noqa: N802
        stub: NurseExecStub = self.server.stub  # type: ignore[attr-defined]
        status, payload, delay = stub.next_response(None)
        if delay:
            time.sleep(delay)
        if status == HALF_CLOSE:
            self._half_close()
            return
        raw = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class NurseExecStub:
    """本地替身对端。用作上下文管理器：

    >>> with NurseExecStub() as stub:
    >>>     ...  # stub.base_url 指向 127.0.0.1:<随机端口>
    """

    def __init__(self, *, status: int = 201, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.requests: list[RecordedRequest] = []
        self._lock = threading.Lock()
        self._response_queue: list[tuple[int, float]] = []
        self._default_status = status
        self._latency = 0.0
        self._body = json.dumps({"accepted": True}, ensure_ascii=False)
        self._server = _QuietServer((host, port), _Handler)
        self._server.stub = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # ── 生命周期 ────────────────────────────────────────────────────────
    def start(self) -> "NurseExecStub":
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> "NurseExecStub":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── 脚本化行为 ──────────────────────────────────────────────────────
    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def queue_status(self, *statuses: int) -> None:
        """把这些状态码**追加**到响应队列；队列用完后回到默认状态。

        传 :data:`HALF_CLOSE` 即排队一次半开连接（掐连接、不给应答）。
        """
        with self._lock:
            self._response_queue.extend((status, 0.0) for status in statuses)

    def queue_half_close(self, times: int = 1) -> None:
        """排队 ``times`` 次半开连接。"""
        self.queue_status(*([HALF_CLOSE] * times))

    def queue_response(self, status: int, *, latency: float = 0.0) -> None:
        """排队一次「状态码 + 该次响应前的延迟」。

        延迟用于制造读超时：客户端先放弃，替身随后才应答——与「对端慢」的真实
        情况一致，而不是把 socket 直接掐断。
        """
        with self._lock:
            self._response_queue.append((status, latency))

    def set_default_status(self, status: int, *, body: str | None = None) -> None:
        with self._lock:
            self._default_status = status
            if body is not None:
                self._body = body

    def set_latency(self, seconds: float) -> None:
        """每次响应前等待的秒数——用来制造读超时（对端慢，不是网络断）。"""
        self._latency = seconds

    def next_response(self, record: RecordedRequest | None) -> tuple[int, str, float]:
        if record is not None:
            with self._lock:
                self.requests.append(record)
        with self._lock:
            queued = self._response_queue.pop(0) if self._response_queue else None
        if queued is not None:
            status, delay = queued
        else:
            status, delay = self._default_status, self._latency
        return status, self._body, delay

    # ── 断言辅助 ────────────────────────────────────────────────────────
    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self.requests)

    def last(self) -> RecordedRequest:
        with self._lock:
            return self.requests[-1]

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()