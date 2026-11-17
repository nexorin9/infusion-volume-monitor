"""本地 OpenAI-compatible 替身：模型网关的**测试替身**，跑在真实 TCP 端口上。

为什么不是 ``httpx.MockTransport``：协议兼容性测试要验的是「网关真的按 HTTP
协议把请求发出去了」——报头、请求体形状、超时参数、重试后的多次请求，都得真的
过一遍 socket。用内存传输层只能验「我们调用了 post」，验不出路径拼错
（``/chat/completions`` 少写一段）、``response_format`` 没带、连接超时参数没生效
这类真会在病区暴露的问题。

替身只出现在**测试数据层**：它不是最终对接边界，也不代表真实模型服务的接口
行为。真实端点本轮未实连（见 progress.txt）。

能力：记录每次请求（方法/路径/报头/体），可脚本化状态码序列、响应体与响应延迟，
用来制造成功、5xx、连接超时、读超时与「返回了但不是 JSON」五类结局。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["LLMStub", "RecordedLLMRequest", "completion_body"]


@dataclass
class RecordedLLMRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes
    received_at: float = field(default_factory=time.time)

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    @property
    def authorization(self) -> str:
        return self.headers.get("authorization", "")

    @property
    def payload(self) -> dict[str, Any]:
        """解析后的请求体（断言 ``model`` / ``messages`` / ``response_format`` 用）。"""
        parsed = self.json()
        return parsed if isinstance(parsed, dict) else {}


def completion_body(
    content: Any,
    *,
    model: str = "test-model",
    prompt_tokens: int = 128,
    completion_tokens: int = 42,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """构造一个 OpenAI-compatible 补全响应。

    ``content`` 传字符串就原样放进 ``message.content``；传 dict/None 时序列化成
    字符串（真实端点也是这么给的），用来测「围栏 JSON」「空回复」这些形态。
    """
    if isinstance(content, str):
        text = content
    elif content is None:
        text = None
    else:
        text = json.dumps(content, ensure_ascii=False)
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 1730000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class _QuietServer(ThreadingHTTPServer):
    """客户端超时断开后 handler 写响应会 BrokenPipeError；那不是缺陷，别刷屏。"""

    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        return


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # noqa: D102
        return

    def _respond(self, record: RecordedLLMRequest | None) -> None:
        stub: LLMStub = self.server.stub  # type: ignore[attr-defined]
        status, payload, delay = stub.next_response(record)
        if delay:
            time.sleep(delay)
        raw = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        stub: LLMStub = self.server.stub  # type: ignore[attr-defined]
        record = RecordedLLMRequest(
            method="POST",
            path=self.path,
            headers={k.lower(): v for k, v in self.headers.items()},
            body=body,
        )
        self._respond(record)

    def do_GET(self) -> None:  # noqa: N802
        self._respond(None)


@dataclass
class LLMStub:
    """脚本化的 OpenAI-compatible 替身。

    ``script`` 按调用次序消费：每项是 ``(status, payload, delay_seconds)``。给空了
    就回最后一个（:attr:`repeat_last`）——重试类用例需要「前几次 5xx、之后成功」。
    """

    script: list[tuple[int, str, float]] = field(default_factory=list)
    repeat_last: bool = True
    host: str = "127.0.0.1"
    port: int = 0

    requests: list[RecordedLLMRequest] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _cursor: int = field(default=0, init=False)

    # ── 生命周期 ──────────────────────────────────────────────────────────
    def start(self) -> "LLMStub":
        server = _QuietServer((self.host, self.port), _Handler)
        server.stub = self  # type: ignore[attr-defined]
        self._server = server
        self.port = int(server.server_address[1])
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "LLMStub":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    # ── 响应脚本 ──────────────────────────────────────────────────────────
    def next_response(self, record: RecordedLLMRequest | None) -> tuple[int, str, float]:
        with self._lock:
            if record is not None:
                self.requests.append(record)
            if not self.script:
                return 200, json.dumps(completion_body({"decision": "no_change", "reason": "余量充足",
                                                         "confidence": 0.9}), ensure_ascii=False), 0.0
            index = min(self._cursor, len(self.script) - 1) if self.repeat_last else self._cursor
            if index >= len(self.script):
                index = len(self.script) - 1
            self._cursor += 1
            return self.script[index]

    # ── 断言辅助 ──────────────────────────────────────────────────────────
    def script_json(
        self,
        content: Any,
        *,
        status: int = 200,
        delay: float = 0.0,
        model: str = "test-model",
        prompt_tokens: int = 128,
        completion_tokens: int = 42,
    ) -> None:
        """排一次「返回某个补全体」的响应。"""
        self.script.append(
            (status, json.dumps(
                completion_body(content, model=model, prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens),
                ensure_ascii=False), delay)
        )

    def script_raw(self, status: int, payload: str, delay: float = 0.0) -> None:
        """排一次「返回任意原始文本」的响应（制造非 JSON 的解析失败）。"""
        self.script.append((status, payload, delay))

    def script_flood(self, status: int, times: int, payload: str = '{"error":"boom"}') -> None:
        """排 ``times`` 次同一响应（制造重试耗尽）。"""
        for _ in range(times):
            self.script.append((status, payload, 0.0))

    def reset(self) -> None:
        """清空脚本并把游标归零。

        同一个替身上连着跑第二组场景时**必须**先调它：``script_*`` 是追加而不是
        替换，游标只随请求前进，于是第二轮拿到的是上一轮排好的回复。这类错会伪装成
        「产品逻辑不对」，排查方向完全跑偏。
        """
        with self._lock:
            self.script.clear()
            self._cursor = 0

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self.requests)