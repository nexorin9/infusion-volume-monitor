#!/usr/bin/env python3
"""跑通真实链路并采集**真实运行截图**（落 ``docs/screenshots/``）。

    python3 scripts/capture_demo_shots.py

这条脚本做的事与护士在床旁做的事同构：

1. 起一个**本地替身对端**（护理执行平台的测试替身，``scripts/demo_stub_peer.py``），
   它会打印自己真正收到的出站请求；
2. 起一个**真的 uvicorn 进程**（``python -m infusion_monitor.server``，结构化 JSON 日志）；
3. 用**真的 curl** 依次完成：登记袋型标定 → 上传帧序列 → 读换液点名清单 →
   护士确认换液（``/transition``）→ 重复确认一次看幂等；
4. 把这些命令的**真实 stdout/stderr** 落成会话记录，再由
   ``scripts/render_terminal_shot.py`` 渲染成 PNG。

**诚实边界**：截图里的文本全部是真跑出来的，但

* 对端是**本地替身**（真实护理执行平台端点本轮未实连）；
* 短片是**程序合成**的（真实院内短片须医院授权采集，见 ``docs/synthetic-data.md``）；
* 服务跑的是**确定性几何核**，不是任何神经分割模型（无 CUDA、无生产权重）。

截图本身不含任何手绘示意内容；概念图在 ``docs/concepts/``，两者不混用。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infusion_monitor.synth import BagPlan, SceneSpec, manifest_from_spec  # noqa: E402

#: 演示场景：5 个挂点、2 种已标定袋型 + 1 种**故意不标定**的袋型。
#: B01 液位低到 2% 一档，用来触发「该换液」点名；B04 用没登记标定的 TYPE-C，
#: 用来对照「没标定就不给毫升数」这条门禁。
DEMO_BAGS = (
    BagPlan("B01", 0, "TYPE-A", 0.15, 0.12),
    BagPlan("B02", 1, "TYPE-A", 0.55, 0.48),
    BagPlan("B03", 2, "TYPE-B", 0.72, 0.66),
    BagPlan("B04", 3, "TYPE-C", 0.30, 0.26),
    BagPlan("B05", 4, "TYPE-A", 0.40, 0.36),
)

CALIBRATIONS = [
    {
        "bag_type_code": "TYPE-A",
        "calibrated_by": "EXAMPLE-NURSE-45",
        "capacity_ml": 500.0,
        "dead_volume_ml": 20.0,
        "samples": [
            {"measured_fill_ratio": 0.1, "known_volume_ml": 48.0},
            {"measured_fill_ratio": 0.4, "known_volume_ml": 192.0},
            {"measured_fill_ratio": 0.8, "known_volume_ml": 384.0},
        ],
    },
    {
        "bag_type_code": "TYPE-B",
        "calibrated_by": "EXAMPLE-NURSE-45",
        "capacity_ml": 1000.0,
        "dead_volume_ml": 30.0,
        "samples": [
            {"measured_fill_ratio": 0.1, "known_volume_ml": 97.0},
            {"measured_fill_ratio": 0.4, "known_volume_ml": 388.0},
            {"measured_fill_ratio": 0.8, "known_volume_ml": 776.0},
        ],
    },
]

SLOTS = [
    {"bag_seq": plan.bag_seq, "slot_index": plan.slot_index,
     "bag_type_code": plan.bag_type_code, "rate_text": "300ml/h"}
    for plan in DEMO_BAGS
]

#: 第 5 段用的两袋场景（另一支渲染脚本，液体饱和度更高，首帧枚举能给出候选）
SLOTS_C = [
    {"bag_seq": "B01", "slot_index": 0, "bag_type_code": "TYPE-A", "rate_text": "300ml/h"},
    {"bag_seq": "B02", "slot_index": 1, "bag_type_code": "TYPE-A", "rate_text": "300ml/h"},
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Session:
    """一次真实会话记录：命令 + 该命令的真实输出。"""

    def __init__(self, env: dict[str, str] | None = None) -> None:
        self.lines: list[str] = []
        self.env = env or dict(os.environ)

    def note(self, text: str) -> None:
        self.lines.append(f"# {text}")

    def run(self, command: str, *, cwd: Path | None = None) -> str:
        """跑一条真命令，把命令本身与真实输出都记进会话记录。"""
        self.lines.append(f"$ {command}")
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=str(cwd or ROOT),
            capture_output=True,
            text=True,
            env=self.env,
        )
        output = (result.stdout or "") + (result.stderr or "")
        output = output.rstrip("\n")
        if output:
            self.lines.append(output)
        self.lines.append("")
        return output

    def raw(self, text: str) -> None:
        self.lines.append(text.rstrip("\n"))
        self.lines.append("")

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def wait_healthy(port: int, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="跑真实链路并采集真实运行截图")
    parser.add_argument("--out-dir", default="docs/screenshots", help="PNG 输出目录")
    parser.add_argument("--workdir", default="var/shotcapture", help="运行时工作目录（会被忽略）")
    parser.add_argument("--keep-workdir", action="store_true", help="保留运行时目录（调试用）")
    args = parser.parse_args()

    out_dir = (ROOT / args.out_dir).resolve()
    workdir = (ROOT / args.workdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    #: 会话记录里用**相对路径**（命令本来就是在项目根下跑的），截图里的命令才抄得走
    wd = args.workdir.rstrip("/")

    scene_id = "shot-45"
    spec = SceneSpec(
        scene_id=scene_id,
        bags=DEMO_BAGS,
        frame_count=20,
        width=640,
        height=480,
        seed=20261022,
        variant="fixed",
    )
    manifest_path = workdir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_from_spec(spec), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    port = free_port()
    stub_port = free_port()
    db_path = workdir / "store.db"
    frames_root = workdir / "frames"

    stub_log = (workdir / "stub-peer.log").open("w", encoding="utf-8")
    server_log = (workdir / "server.log").open("w", encoding="utf-8")

    env = dict(os.environ)
    env.update(
        {
            "DATABASE_URL": f"sqlite:///{db_path}",
            "FRAMES_ROOT": str(frames_root),
            "NURSE_EXEC_BASE_URL": f"http://127.0.0.1:{stub_port}/nurse-exec",
            "NURSE_EXEC_API_TOKEN": "shot45-demo-token-not-a-real-secret",
            "MQTT_HOST": "127.0.0.1",
            "MQTT_PORT": "1883",
            "MQTT_USERNAME": "shot45-demo-user",
            "MQTT_PASSWORD": "shot45-demo-pass",
            "LLM_BASE_URL": "http://127.0.0.1:9/llm",
            "LLM_API_KEY": "shot45-demo-key-not-a-real-secret",
            "LLM_MODEL": "shot45-demo-model",
            "LLM_TIMEOUT": "5",
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "LOG_FORMAT": "json",
            "LOG_LEVEL": "INFO",
            "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
        }
    )
    env.pop("MQTT_ENABLED", None)

    stub = subprocess.Popen(
        [sys.executable, "scripts/demo_stub_peer.py", "--port", str(stub_port)],
        cwd=str(ROOT), env=env, stdout=stub_log, stderr=subprocess.STDOUT,
    )
    server = subprocess.Popen(
        [sys.executable, "-m", "infusion_monitor.server"],
        cwd=str(ROOT), env=env, stdout=server_log, stderr=subprocess.STDOUT,
    )

    base = f"http://127.0.0.1:{port}"
    try:
        if not wait_healthy(port):
            print("服务未在预期时间内起来", file=sys.stderr)
            return 1

        # ── 第 1 张：健康检查 → 场景 → 标定 → 上传 → 面积曲线/毫升/ETA ─────
        shot1 = Session(env)
        shot1.note("输液余量监测服务 · 第 1 段：上传帧序列，拿到面积比曲线、毫升数与点名依据")
        shot1.run(f"curl -sS {base}/api/v1/health | jq '{{status, schema_version, frames_root_writable}}'")
        shot1.run(
            f"jq -c '[.bags[] | {{bag_seq, bag_type_code, fill_start, fill_end}}]' {wd}/manifest.json"
        )
        shot1.run(
            f"python3 scripts/synthesize_rack.py --reproduce {wd}/manifest.json --out {wd}/scene"
        )
        shot1.run(f"ls -1 {wd}/scene/{scene_id}")

        for payload in CALIBRATIONS:
            calib_path = workdir / f"calib-{payload['bag_type_code']}.json"
            calib_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            shot1.run(
                f"curl -sS -X POST {base}/api/v1/infusion/calibrations "
                f"-H 'Content-Type: application/json' -d @{calib_path} "
                f"-w '\\n→ HTTP %{{http_code}}\\n'"
            )

        shot1.run(
            f"curl -sS -X POST {base}/api/v1/infusion/tracks "
            f"-F rack_id=R-EXAMPLE-45 -F infusion_order_no=EXAMPLE-ORDER-0045 "
            f"-F bed_no=EXAMPLE-BED-45 -F captured_at=2026-10-22T09:41:00+08:00 "
            f"-F shift_key=2026-10-22-AM -F slots='{json.dumps(SLOTS, ensure_ascii=False)}' "
            f"-F frames=@{wd}/scene/{scene_id}/frames.zip "
            f"-o {wd}/upload.json -w '→ HTTP %{{http_code}}\\n'"
        )
        shot1.run(
            f"jq '{{upload_key, frame_count, bags: [.bags[] | {{bag_seq, state, calibration_state, "
            f"pickup_state, trigger_reason, fill_ratio, "
            f"estimated_volume_ml: (.estimated_volume_ml // \"—— 该字段不出现（袋型未标定）\"), "
            f"eta_minutes: (.eta_minutes // \"—— 该字段不出现（无滴速/未标定）\"), eta_source, "
            f"points: (.area_ratio_series|length), curve_head: (.area_ratio_series[0:3]), "
            f"curve_tail: (.area_ratio_series[-2:])}}]}}' {wd}/upload.json"
        )
        shot1.note(
            "B04 用的是 TYPE-C，这一型**故意没登记标定**——原始响应里 estimated_volume_ml / "
            "eta_minutes 两个键**整体不存在**（不是 null）：门禁不给不可信的数字，它也没被点名。"
            "下面这条命令看 B04 的原始对象"
        )
        shot1.run(
            f"jq '[.bags[] | {{bag_seq, calibration_state, has_ml: has(\"estimated_volume_ml\"), "
            f"has_eta: has(\"eta_minutes\"), suppression_reason}}]' {wd}/upload.json"
        )

        # ── 第 2 张：点名清单 → 护士确认换液 → 重复确认（幂等） ───────────
        shot2 = Session(env)
        shot2.note("第 2 段：换液点名清单 → 护士点「已换液」→ 重复点击看幂等")
        shot2.run(
            f"curl -sS {base}/api/v1/infusion/pickups -o {wd}/pickups.json -w '→ HTTP %{{http_code}}\\n'"
        )
        shot2.run(
            f"jq '{{count, writeback_queued_count, items: [.items[] | {{rank, bag_seq, bed_no, "
            f"eta_minutes, estimated_volume_ml, trigger_reason, confidence, writeback_queued}}]}}' "
            f"{wd}/pickups.json"
        )
        shot2.note("未确认前：清单里 writeback_queued 全为 false，出站队列计数为 0")
        shot2.run(
            f"curl -sS {base}/api/v1/infusion/pickups | jq '.items[0].confirm_hint'"
        )
        shot2.run(
            f"curl -sS -X POST {base}/api/v1/infusion/transition -H 'Content-Type: application/json' "
            f"-d \"$(jq -c '.items[0].confirm_hint + {{operator_id: \"N-EXAMPLE-45\"}}' "
            f"{wd}/pickups.json)\" -o {wd}/transition.json -w '→ HTTP %{{http_code}}\\n'"
        )
        shot2.run(
            f"jq '{{session_id, bag_seq, state, operator_id, client_event_id, event_id, created, "
            f"idempotent_hit, confirmed_at, pickup_evidence, writeback}}' {wd}/transition.json"
        )
        shot2.note("同一个确认再点一次（病区 Wi-Fi 抖动导致的重发）：幂等命中，不产生第二条事件")
        shot2.run(
            f"curl -sS -X POST {base}/api/v1/infusion/transition -H 'Content-Type: application/json' "
            f"-d \"$(jq -c '.items[0].confirm_hint + {{operator_id: \"N-EXAMPLE-45\"}}' "
            f"{wd}/pickups.json)\" -o {wd}/transition2.json -w '→ HTTP %{{http_code}}\\n'"
        )
        shot2.run(
            f"jq '{{created, idempotent_hit, event_id, confirmed_at, writeback_kind: .writeback.kind}}' "
            f"{wd}/transition2.json"
        )

        # ── 第 3 张：出站写回（替身对端真正收到的请求） ───────────────────
        shot3 = Session(env)
        shot3.note("第 3 段：出站写回——护理执行平台端点收到的真实请求（本地替身对端）")
        time.sleep(1.0)
        shot3.raw((workdir / "stub-peer.log").read_text(encoding="utf-8").strip())
        shot3.note("对端是本地替身：真实护理执行平台端点本轮未实连（见 docs/nurse-exec-contract.md）")

        # ── 第 4 张：服务侧结构化日志（同一 request_id 贯穿入站与出站） ────
        shot4 = Session(env)
        shot4.note("第 4 段：服务侧结构化日志——一条 JSON 一行，request_id 贯穿入站与出站")
        shot4.run(
            f"tail -n 14 {wd}/server.log | jq -c '{{ts, level, event, request_id}}'"
        )

        # ── 第 5 张：首帧候选列表 → 护士在列表里认领 → 曲线按新候选重算 ────
        shot5 = Session(env)
        shot5.note("第 5 段：首帧无提示枚举出的候选列表 → 护士在列表里认领（不用逐袋点像素）")
        shot5.run(
            f"python3 scripts/make_sample_track.py --out {wd}/rack2.zip --frames 20 --bags 2 --fill 0.55"
        )
        slots_c_path = workdir / "slots-c.json"
        slots_c_path.write_text(json.dumps(SLOTS_C, ensure_ascii=False), encoding="utf-8")
        shot5.run(
            f"curl -sS -X POST {base}/api/v1/infusion/tracks "
            f"-F rack_id=R-EXAMPLE-45C -F infusion_order_no=EXAMPLE-ORDER-0045C "
            f"-F bed_no=EXAMPLE-BED-45C -F captured_at=2026-10-22T13:41:00+08:00 "
            f"-F shift_key=2026-10-22-PM -F \"slots=$(cat {wd}/slots-c.json)\" "
            f"-F frames=@{wd}/rack2.zip -o {wd}/upload2.json -w '→ HTTP %{{http_code}}\\n'"
        )
        shot5.run(
            f"jq '[.candidates[] | {{candidate_id, bbox, stability_score, slot_index, bound}}]' "
            f"{wd}/upload2.json"
        )
        shot5.run(
            f"jq '.bags[] | {{bag_seq, session_id, candidate_state, candidate_id, claim_state}}' "
            f"{wd}/upload2.json"
        )
        shot5.note("护士点列表里的一行认领：曲线按新候选重算，认领人写进 append-only 审计链")
        shot5.run(
            f"curl -sS -X POST {base}/api/v1/infusion/claim -H 'Content-Type: application/json' "
            f"-d \"$(jq -c '{{session_id: .bags[0].session_id, candidate_id: .candidates[0].candidate_id, "
            f"operator_id: \"N-EXAMPLE-45\", note: \"床号侧的加药袋\"}}' {wd}/upload2.json)\" "
            f"-o {wd}/claim.json -w '→ HTTP %{{http_code}}\\n'"
        )
        shot5.run(
            f"jq '{{session_id, bag_seq, candidate_id, bbox, claim_state, corrected, claimed_by, "
            f"fill_ratio, estimated_volume_ml, eta_minutes, eta_source, descriptor_refreshed, "
            f"curve_head: (.area_ratio_series[0:3]), curve_tail: (.area_ratio_series[-3:]), "
            f"uncertainty: {{confidence: .volume_uncertainty.confidence, gate: .volume_uncertainty.gate, "
            f"hint: .volume_uncertainty.hint_zh}}}}' {wd}/claim.json"
        )

        # ── 渲染 PNG ────────────────────────────────────────────────────
        sys.path.insert(0, str(ROOT / "scripts"))
        from render_terminal_shot import ShotRenderer  # noqa: E402

        renderer = ShotRenderer(size=15)
        shots = [
            ("01-upload-and-curves.png", shot1, "输液余量监测 · 上传帧序列 → 面积曲线 / 毫升 / 点名依据"),
            ("02-pickup-and-transition.png", shot2, "输液余量监测 · 换液点名 → 护士确认 → 幂等重放"),
            ("03-outbound-writeback.png", shot3, "输液余量监测 · 出站写回：执行平台收到的真实请求"),
            ("04-service-log.png", shot4, "输液余量监测 · 服务侧结构化日志（JSON 行）"),
            ("05-candidate-claim.png", shot5, "输液余量监测 · 首帧候选列表 → 护士认领 → 曲线重算"),
        ]
        for filename, session, title in shots:
            transcript = workdir / filename.replace(".png", ".txt")
            transcript.write_text(session.text(), encoding="utf-8")
            path = renderer.render(session.text(), out_dir / filename, title)
            # 输出目录可能落在仓库之外（如 --out-dir /tmp/...），此时打印相对路径会抛
            # ValueError：relative_to() 要求目标在 ROOT 之下。用 relpath 兜住。
            try:
                shown = path.relative_to(ROOT)
            except ValueError:
                shown = path
            print(f"{shown}  ({path.stat().st_size // 1024} KB)")

        print("\n关键结论（来自真实响应）：")
        upload = json.loads((workdir / "upload.json").read_text(encoding="utf-8"))
        for bag in upload["bags"]:
            print(
                f"  {bag['bag_seq']} state={bag['state']:12s} "
                f"calibration={bag.get('calibration_state'):13s} "
                f"ml={bag.get('estimated_volume_ml')} eta={bag.get('eta_minutes')} "
                f"trigger={bag.get('trigger_reason')}"
            )
        pickups = json.loads((workdir / "pickups.json").read_text(encoding="utf-8"))
        print(f"  点名清单 count={pickups['count']} writeback_queued_count={pickups['writeback_queued_count']}")
        return 0
    finally:
        for proc in (server, stub):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        stub_log.close()
        server_log.close()
        if not args.keep_workdir:
            for path in (db_path,):
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
