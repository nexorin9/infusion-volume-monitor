"""起一个本地替身对端（护理执行平台的测试替身），把收到的请求打印出来。

用法::

    python3 scripts/demo_stub_peer.py --port 8899 > /tmp/stub.log 2>&1 &

然后把 ``NURSE_EXEC_BASE_URL`` 指到 ``http://127.0.0.1:8899`` 跑真链路。
**这是测试替身，不是最终对接边界**：真实护理执行平台端点本轮未实连。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from support.nurse_exec_stub import NurseExecStub  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="本地替身对端（打印收到的出站请求）")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--status", type=int, default=201)
    args = parser.parse_args()

    stub = NurseExecStub(status=args.status, port=args.port).start()
    print(f"stub listening on {stub.base_url}", flush=True)
    try:
        while True:
            time.sleep(0.5)
            if stub.call_count:
                request = stub.last()
                print("---", json.dumps({
                    "method": request.method,
                    "path": request.path,
                    "authorization": request.authorization,
                    "idempotency_key": request.idempotency_key,
                    "content_type": request.headers.get("content-type"),
                    "body": request.json(),
                }, ensure_ascii=False, indent=2), flush=True)
                stub.reset()
    except KeyboardInterrupt:
        pass
    finally:
        stub.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())