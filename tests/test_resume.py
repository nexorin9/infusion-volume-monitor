"""分片续传、重复帧去重、整段重放与半成品清理（task 32）。

四组断言，各自对应 task steps 的四行：

1. **续传补齐**：开一次上传 → 传第一片 → 断点查询指出缺哪几帧 → 补第二片 → 提交，
   提交的产出与一次性 ``/tracks`` 完全同形（同一个主路径）。
2. **重复帧去重**：重传同一片 → 安静去重、已收帧号不变、磁盘上仍是那一份；
   换个内容却顶着同一个帧号 → 当场拒收并说清为什么（不静默丢片）。
3. **重放不双写**：同一 ``rack + 轮次 + 袋序`` 整段重传 → 版本加一、旧版本快照可查、
   事件表不新增行、已确认换袋的状态不被重置回监测中。
4. **半成品清理**：保留期到点的未提交上传被回收并结案；已完成会话的帧目录一个都不动。
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from infusion_monitor.ingest.resume import purge_expired, staging_root
from infusion_monitor.ingest.zip_frames import upload_key
from infusion_monitor.store.repositories import UploadRepository
from tests.support import EXAMPLE_FIELDS, default_rack_scene, render_rack_frame

BASE_TIME = datetime(2026, 10, 22, 9, 41, 0)
TRACKS_URL = "/api/v1/infusion/tracks"
UPLOADS_URL = "/api/v1/infusion/uploads"


# ── 测试脚手架 ────────────────────────────────────────────────────────────


def chunk_zip(
    start: int,
    stop: int,
    *,
    scene=None,
    renumber_from: int | None = None,
    duplicate_of: int | None = None,
    **render_kwargs,
) -> bytes:
    """打一个分片：成员按**整段全局序号**命名（``renumber_from`` 可改成片内编号）。

    ``duplicate_of`` = 把别处的某一帧也塞进这个包（构造「顶着同一个帧号、内容却不同」
    的分叉场景），重复同一帧两次则用 ``start > stop`` 的空区间表达不了，这里靠
    ``duplicate_of`` 指定重复出现的帧号。
    """
    scene = scene or default_rack_scene(24)
    quality = int(render_kwargs.pop("quality", 85))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        indices = list(range(start, stop))
        if duplicate_of is not None:
            indices = indices + [duplicate_of]
        for index in indices:
            number = index if renumber_from is None else renumber_from + (index - start)
            frame = render_rack_frame(scene, index, **render_kwargs)
            moment = BASE_TIME + timedelta(seconds=0.5 * index)
            info = zipfile.ZipInfo(f"frame_{number:05d}.jpg", date_time=moment.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            encoded = io.BytesIO()
            Image.fromarray(frame, mode="RGB").save(encoded, format="JPEG", quality=quality)
            archive.writestr(info, encoded.getvalue())
    return buffer.getvalue()


def full_zip(scene=None, **render_kwargs) -> bytes:
    """整段（一次性 ``/tracks`` 用）的帧 zip。"""
    scene = scene or default_rack_scene(24)
    quality = int(render_kwargs.pop("quality", 85))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(scene.frame_count):
            frame = render_rack_frame(scene, index, **render_kwargs)
            moment = BASE_TIME + timedelta(seconds=0.5 * index)
            info = zipfile.ZipInfo(f"frame_{index:05d}.jpg", date_time=moment.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            encoded = io.BytesIO()
            Image.fromarray(frame, mode="RGB").save(encoded, format="JPEG", quality=quality)
            archive.writestr(info, encoded.getvalue())
    return buffer.getvalue()


def open_form(**overrides) -> dict[str, str]:
    fields = dict(EXAMPLE_FIELDS)
    fields.update({k: str(v) for k, v in overrides.items()})
    return fields


def open_upload(client, **overrides):
    return client.post(UPLOADS_URL, data=open_form(**overrides))


def put_chunk(client, upload_id: str, payload: bytes, **fields):
    files = {"chunk": ("chunk.zip", payload, "application/zip")}
    return client.post(f"{UPLOADS_URL}/{upload_id}/chunks", data=fields, files=files)


def status_of(client, upload_id: str):
    response = client.get(f"{UPLOADS_URL}/{upload_id}")
    assert response.status_code == 200, response.text
    return response.json()


def issue_codes(response) -> set[str]:
    return {issue["code"] for issue in response.json()["detail"]["issues"]}


@pytest.fixture
def resume_client(client_factory, tmp_path):
    """帧目录指到 tmp_path 的 TestClient（不往仓库里的 ./var/frames 写）。"""
    with client_factory(FRAMES_ROOT=str(tmp_path / "frames")) as client:
        yield client


@pytest.fixture
def resume_store_client(credential_env, tmp_path):
    """TestClient（文件库 + tmp 帧目录）+ 一条指向同一库文件的观察连接。

    应用自己的 SQLite 连接带线程亲和性，只能在 TestClient 门户线程里用；测试线程
    要断言落库结果（事件表行数、upload_chunk 状态、审计动作）就得自己开一条。
    """
    from fastapi.testclient import TestClient

    from infusion_monitor.api.service import create_app
    from infusion_monitor.config import Settings
    from infusion_monitor.store import apply_migrations, connect

    dsn = f"sqlite:///{tmp_path / 'resume.db'}"
    app = create_app(
        Settings(frames_root=str(tmp_path / "frames")),
        database_url=dsn,
        frames_root=tmp_path / "frames",
    )
    observer = connect(dsn)
    apply_migrations(observer)
    with TestClient(app) as test_client:
        yield test_client, observer
    observer.close()


def frames_root_of(client) -> Path:
    return Path(client.app.state.settings.frames_root)


# ── 1. 续传补齐 ───────────────────────────────────────────────────────────


def test_resume_fills_gap_then_commit_runs_the_same_main_path(resume_client):
    client = resume_client
    opened = open_upload(client, expected_frames=24)
    assert opened.status_code == 201, opened.text
    upload_id = opened.json()["upload_id"]
    session_key = opened.json()["session_key"]
    assert opened.json()["received_frames"] == []

    first = put_chunk(client, upload_id, chunk_zip(0, 12))
    assert first.status_code == 200, first.text
    assert first.json()["accepted_frames"] == list(range(12))

    # 断点查询：已收 12 帧，明确点名还缺 12–23
    midway = status_of(client, upload_id)
    assert midway["received_count"] == 12
    assert midway["missing_frames"] == list(range(12, 24))

    second = put_chunk(client, upload_id, chunk_zip(12, 24))
    assert second.status_code == 200, second.text
    assert second.json()["missing_frames"] == []

    committed = client.post(f"{UPLOADS_URL}/{upload_id}/commit")
    assert committed.status_code == 201, committed.text
    body = committed.json()
    # 与 /tracks 同形的响应：同一个主路径算出来的余量曲线，不是简化版
    assert body["upload_key"] == session_key
    assert body["frame_count"] == 24
    assert len(body["bags"]) == 1
    assert len(body["bags"][0]["area_ratio_series"]) == 24

    done = status_of(client, upload_id)
    assert done["status"] == "completed"
    assert done["session_ids"] == [body["bags"][0]["session_id"]]
    assert done["completed_at"] is not None


def test_resume_and_whole_track_upload_agree(resume_client):
    """分片提交与一次性上传走同一条主干：同样的短片必须给出同一条曲线。"""
    client = resume_client
    whole = client.post(
        TRACKS_URL,
        data=open_form(shift_key="2026-10-22-AM"),
        files={"frames": ("rack.zip", full_zip(), "application/zip")},
    )
    assert whole.status_code == 201, whole.text

    opened = open_upload(client, expected_frames=24, shift_key="2026-10-22-PM")
    upload_id = opened.json()["upload_id"]
    assert put_chunk(client, upload_id, chunk_zip(0, 9)).status_code == 200
    assert put_chunk(client, upload_id, chunk_zip(9, 24)).status_code == 200
    chunked = client.post(f"{UPLOADS_URL}/{upload_id}/commit")
    assert chunked.status_code == 201, chunked.text

    whole_curve = whole.json()["bags"][0]["area_ratio_series"]
    chunked_curve = chunked.json()["bags"][0]["area_ratio_series"]
    assert chunked_curve == pytest.approx(whole_curve, abs=1e-9)


def test_commit_with_gap_keeps_already_received_chunks(resume_client):
    client = resume_client
    upload_id = open_upload(client, expected_frames=24).json()["upload_id"]
    put_chunk(client, upload_id, chunk_zip(0, 12))

    early = client.post(f"{UPLOADS_URL}/{upload_id}/commit")
    assert early.status_code == 422, early.text
    assert issue_codes(early) == {"chunk_gap"}
    detail = early.json()["detail"]["issues"][0]
    assert "缺 12 帧" in detail["reason"]
    assert "missing_frames" in detail["suggestion"]

    # 已收到的 12 帧仍在暂存目录里，补齐即可提交，不重来
    staging = staging_root(frames_root_of(client)) / upload_id
    assert len(list(staging.glob("frame_*.jpg"))) == 12
    assert put_chunk(client, upload_id, chunk_zip(12, 24)).status_code == 200
    assert client.post(f"{UPLOADS_URL}/{upload_id}/commit").status_code == 201


def test_commit_twice_is_rejected_and_chunks_of_closed_upload_are_refused(resume_client):
    client = resume_client
    upload_id = open_upload(client, expected_frames=12).json()["upload_id"]
    put_chunk(client, upload_id, chunk_zip(0, 12))
    assert client.post(f"{UPLOADS_URL}/{upload_id}/commit").status_code == 201

    again = client.post(f"{UPLOADS_URL}/{upload_id}/commit")
    assert again.status_code == 409, again.text
    assert again.json()["detail"]["error_code"] == "UPLOAD_ALREADY_COMPLETED"

    late = put_chunk(client, upload_id, chunk_zip(0, 12))
    assert late.status_code == 409, late.text
    assert late.json()["detail"]["error_code"] == "UPLOAD_NOT_OPEN"


def test_unknown_upload_id_is_404_not_silent_drop(resume_client):
    client = resume_client
    assert put_chunk(client, "upl_not_here", chunk_zip(0, 12)).status_code == 404
    assert client.get(f"{UPLOADS_URL}/upl_not_here").status_code == 404
    assert client.post(f"{UPLOADS_URL}/upl_not_here/commit").status_code == 404


# ── 2. 重复帧去重 ─────────────────────────────────────────────────────────


def test_resending_the_same_chunk_is_deduplicated_quietly(resume_client):
    client = resume_client
    upload_id = open_upload(client, expected_frames=24).json()["upload_id"]
    payload = chunk_zip(0, 12)
    assert put_chunk(client, upload_id, payload).status_code == 200

    staging = staging_root(frames_root_of(client)) / upload_id
    before = {path.name: path.read_bytes() for path in sorted(staging.glob("frame_*.jpg"))}
    assert len(before) == 12

    retry = put_chunk(client, upload_id, payload)
    assert retry.status_code == 200, retry.text
    body = retry.json()
    assert body["accepted_frames"] == []
    assert body["duplicates_dropped"] == list(range(12))
    assert body["received_frames"] == list(range(12))  # 断点不回退也不膨胀
    assert body["duplicate_count_total"] == 12

    after = {path.name: path.read_bytes() for path in sorted(staging.glob("frame_*.jpg"))}
    assert after == before  # 磁盘上仍是先到的那一份，字节未变


def test_duplicate_frame_inside_one_chunk_is_kept_once(resume_client):
    client = resume_client
    upload_id = open_upload(client, expected_frames=12).json()["upload_id"]
    # 同一个 zip 里出现两遍同一帧号（客户端切片脚本去重漏了）
    payload = chunk_zip(0, 11, duplicate_of=5)
    response = put_chunk(client, upload_id, payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["accepted_frames"] == list(range(11))
    assert body["duplicates_dropped"] == [5]
    assert body["received_count"] == 11
    staging = staging_root(frames_root_of(client)) / upload_id
    assert len(list(staging.glob("frame_*.jpg"))) == 11


def test_same_frame_number_with_different_content_is_refused(resume_client):
    """顶着已收到的帧号传另一段的内容 → 当场问清楚，绝不静默把整片丢掉。"""
    client = resume_client
    upload_id = open_upload(client, expected_frames=24).json()["upload_id"]
    put_chunk(client, upload_id, chunk_zip(0, 12))

    # 第二片被终端错误地从 0 重新编号：frame_00000…frame_00011，内容却是 12–23 帧
    misnumbered = chunk_zip(12, 24, renumber_from=0)
    rejected = put_chunk(client, upload_id, misnumbered)
    assert rejected.status_code == 422, rejected.text
    assert issue_codes(rejected) == {"chunk_frame_conflict"}
    detail = rejected.json()["detail"]["issues"][0]
    assert "frame_offset" in detail["suggestion"]

    # 断点没有因此被推歪
    assert status_of(client, upload_id)["received_count"] == 12

    # 显式声明 offset 时同样的内容能正确落到 12–23
    ok = put_chunk(client, upload_id, chunk_zip(12, 24, renumber_from=0), frame_offset="12")
    assert ok.status_code == 200, ok.text
    assert ok.json()["accepted_frames"] == list(range(12, 24))


def test_repacked_duplicate_is_reported_instead_of_silently_swallowed(resume_client):
    """重新打包后再传的同一帧：按字节判为「不是同一帧」→ 422，而不是安静去重。

    逐帧液面变化在整幅画面上只占千分之几的像素，像素级近似根本分不开「重试」与
    「按片重新编号」；宁可报一次 422（文案给出两条可能与出路），也不静默吞掉一整片。
    """
    client = resume_client
    upload_id = open_upload(client, expected_frames=24).json()["upload_id"]
    put_chunk(client, upload_id, chunk_zip(0, 12))

    repacked = chunk_zip(0, 12, quality=60)
    response = put_chunk(client, upload_id, repacked)
    assert response.status_code == 422, response.text
    assert issue_codes(response) == {"chunk_frame_conflict"}
    suggestion = response.json()["detail"]["issues"][0]["suggestion"]
    assert "frame_offset" in suggestion and "received_frames" in suggestion

    # 断点没有被推歪，原始那一份仍在
    assert status_of(client, upload_id)["received_count"] == 12
    staging = staging_root(frames_root_of(client)) / upload_id
    assert len(list(staging.glob("frame_*.jpg"))) == 12


def test_frame_offset_lets_chunk_local_numbering_work(resume_client):
    """终端只能按片内编号时：显式传 frame_offset，内容与全局编号一致即可接上。"""
    client = resume_client
    upload_id = open_upload(client, expected_frames=24).json()["upload_id"]
    put_chunk(client, upload_id, chunk_zip(0, 12))
    response = put_chunk(client, upload_id, chunk_zip(12, 24, renumber_from=0), frame_offset="12")
    assert response.status_code == 200, response.text
    assert response.json()["accepted_frames"] == list(range(12, 24))


# ── 3. 整段重放不双写 ─────────────────────────────────────────────────────


def test_whole_replay_bumps_version_archives_old_one(resume_client):
    client = resume_client
    first = client.post(
        TRACKS_URL,
        data=open_form(),
        files={"frames": ("rack.zip", full_zip(), "application/zip")},
    )
    assert first.status_code == 201, first.text
    session_id = first.json()["bags"][0]["session_id"]
    assert first.json()["bags"][0]["session_version"] == 1

    replay = client.post(
        TRACKS_URL,
        data=open_form(),
        files={"frames": ("rack.zip", full_zip(deform=0.03), "application/zip")},
    )
    assert replay.status_code == 201, replay.text
    body = replay.json()
    # 同一袋仍是同一个 session_id，版本加一
    assert body["bags"][0]["session_id"] == session_id
    assert body["bags"][0]["session_version"] == 2

    versions = client.get(f"/api/v1/infusion/sessions/{session_id}/versions")
    assert versions.status_code == 200, versions.text
    payload = versions.json()
    assert payload["current_version"] == 2
    assert payload["archived_count"] == 1
    archived = payload["versions"][0]
    assert archived["session_version"] == 1
    assert archived["archived_reason"] == "replay"
    assert archived["area_ratio_series"] == pytest.approx(
        first.json()["bags"][0]["area_ratio_series"], abs=1e-9
    )
    assert archived["archived_at"] is not None


def test_replay_after_nurse_confirmation_keeps_one_event_and_terminal_state(
    credential_env, tmp_path
):
    """已确认换袋后再整段重放：状态不被撤回，事件表不新增，出站也不重发。

    用真 socket 的出站替身（``NurseExecStub``）当护理执行平台：断言的是「重放之后
    替身仍只收到 1 次写回」，而不只是「事件表没多一行」。
    """
    from fastapi.testclient import TestClient

    from infusion_monitor.adapters.nurse_exec import NurseExecClient
    from infusion_monitor.api.service import create_app
    from infusion_monitor.config import Settings
    from infusion_monitor.store import SessionRepository, apply_migrations, connect
    from tests.support import NurseExecStub

    dsn = f"sqlite:///{tmp_path / 'replay.db'}"
    app = create_app(
        Settings(frames_root=str(tmp_path / "frames")),
        database_url=dsn,
        frames_root=tmp_path / "frames",
    )
    observer = connect(dsn)
    apply_migrations(observer)
    with TestClient(app) as client, NurseExecStub(status=201) as stub:
        client.app.state.nurse_exec = NurseExecClient(
            base_url=f"{stub.base_url}/api/v1/infusion/events",
            api_token="test-token-not-real",
        )
        payload = full_zip()
        first = client.post(
            TRACKS_URL,
            data=open_form(),
            files={"frames": ("rack.zip", payload, "application/zip")},
        )
        assert first.status_code == 201, first.text
        session_id = first.json()["bags"][0]["session_id"]
        bag_seq = first.json()["bags"][0]["bag_seq"]

        # 点名引擎把它推到 needs_change（与 /tracks 同一条判定链的产出），
        # 护士在终端确认 —— monitoring 直跳 replaced 会被状态机拒掉，这里不绕开。
        SessionRepository(observer).apply_state_transition(
            session_id, "needs_change", trigger_reason="eta_within_window"
        )
        confirmed = client.post(
            "/api/v1/infusion/transition",
            json={"session_id": session_id, "bag_seq": bag_seq, "operator_id": "N-EXAMPLE-32"},
        )
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["created"] is True
        assert confirmed.json()["state"] == "replaced"
        assert observer.execute("SELECT COUNT(*) FROM infusion_event").fetchone()[0] == 1
        assert len(stub.requests) == 1

        replay = client.post(
            TRACKS_URL,
            data=open_form(),
            files={"frames": ("rack.zip", payload, "application/zip")},
        )
        assert replay.status_code == 201, replay.text
        assert replay.json()["bags"][0]["session_version"] == 2
        assert replay.json()["bags"][0]["state"] == "replaced"  # 终态不被重传撤回

        # 幂等键由 (session_id, bag_seq) 决定：重放换了版本号也还是同一条事件
        assert observer.execute("SELECT COUNT(*) FROM infusion_event").fetchone()[0] == 1
        assert len(stub.requests) == 1  # 出站一次都没多发
        row = observer.execute(
            "SELECT state FROM bag_session WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert row["state"] == "replaced"

        # 旧版本留档：被盖掉的那一版原样可查（快照记的是**被覆盖前**那一刻的样子）
        archived = client.get(f"/api/v1/infusion/sessions/{session_id}/versions").json()
        assert archived["archived_count"] == 1
        assert archived["versions"][0]["session_version"] == 1
        assert archived["versions"][0]["state"] == "replaced"
        assert archived["versions"][0]["area_ratio_series"] == pytest.approx(
            first.json()["bags"][0]["area_ratio_series"], abs=1e-9
        )

        actions = [row["action"] for row in observer.execute("SELECT action FROM audit_log")]
        assert "session.version_archived" in actions
        assert "state.confirmed" in actions
    observer.close()


def test_replay_archives_only_once_per_version(resume_store_client):
    """连打三次重放：被盖掉的每一版各留一条快照，审计也各一行。"""
    client, observer = resume_store_client
    payload = full_zip()
    first = client.post(
        TRACKS_URL, data=open_form(), files={"frames": ("rack.zip", payload, "application/zip")}
    )
    session_id = first.json()["bags"][0]["session_id"]
    for _ in range(2):
        assert (
            client.post(
                TRACKS_URL,
                data=open_form(),
                files={"frames": ("rack.zip", payload, "application/zip")},
            ).status_code
            == 201
        )
    versions = client.get(f"/api/v1/infusion/sessions/{session_id}/versions").json()
    assert versions["current_version"] == 3
    assert [item["session_version"] for item in versions["versions"]] == [1, 2]
    assert observer.execute(
        "SELECT COUNT(*) FROM bag_session_history WHERE session_id = ?", (session_id,)
    ).fetchone()[0] == 2
    assert observer.execute(
        "SELECT COUNT(*) FROM audit_log WHERE action = 'session.version_archived'"
    ).fetchone()[0] == 2


# ── 4. 半成品清理 ─────────────────────────────────────────────────────────


def test_expired_partial_upload_is_purged_but_completed_frames_survive(
    resume_store_client,
):
    client, observer = resume_store_client
    root = frames_root_of(client)

    # 一段已经提交过的（帧必须活着）
    done_id = open_upload(client, expected_frames=12, shift_key="2026-10-22-AM").json()["upload_id"]
    put_chunk(client, done_id, chunk_zip(0, 12))
    committed = client.post(f"{UPLOADS_URL}/{done_id}/commit")
    assert committed.status_code == 201, committed.text
    session_key = committed.json()["upload_key"]
    session_dir = root / session_key
    assert len(list(session_dir.glob("frame_*.jpg"))) == 12

    # 一段只传了一半、过了保留期的
    stale_id = open_upload(client, expected_frames=24, shift_key="2026-10-22-PM").json()["upload_id"]
    put_chunk(client, stale_id, chunk_zip(0, 8))
    stale_staging = staging_root(root) / stale_id
    assert stale_staging.exists()

    observer.execute(
        "UPDATE upload_chunk SET expires_at = ? WHERE upload_id = ?",
        (
            (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat(timespec="seconds"),
            stale_id,
        ),
    )
    uploads = UploadRepository(observer)

    result = client.post(f"{UPLOADS_URL}/cleanup")
    assert result.status_code == 200, result.text
    payload = result.json()
    assert payload["expired_uploads"] == [stale_id]
    assert payload["freed_bytes"] > 0
    assert payload["kept_completed"] == 1

    assert not stale_staging.exists()                      # 半成品回收
    assert len(list((session_dir).glob("frame_*.jpg"))) == 12  # 已完成会话一帧不少
    assert uploads.get(stale_id)["status"] == "abandoned"
    assert uploads.get(done_id)["status"] == "completed"


def test_cleanup_respects_configured_retention_and_supports_dry_run(resume_store_client):
    client, observer = resume_store_client
    root = frames_root_of(client)
    upload_id = open_upload(client, expected_frames=24).json()["upload_id"]
    put_chunk(client, upload_id, chunk_zip(0, 8))

    # 保留期 24 小时、实际才过 1 小时 → 不该被收
    fresh = client.post(f"{UPLOADS_URL}/cleanup")
    assert fresh.status_code == 200
    assert fresh.json()["expired_count"] == 0
    assert (staging_root(root) / upload_id).exists()

    # dry_run 只报不删
    observer.execute("UPDATE upload_chunk SET expires_at = '2000-01-01T00:00:00+00:00'")
    dry = client.post(f"{UPLOADS_URL}/cleanup?dry_run=true").json()
    assert dry["expired_count"] == 1 and dry["dry_run"] is True
    assert (staging_root(root) / upload_id).exists()

    real = client.post(f"{UPLOADS_URL}/cleanup").json()
    assert real["expired_count"] == 1
    assert not (staging_root(root) / upload_id).exists()


def test_pending_upload_cap_refuses_new_open_and_points_to_cleanup(
    client_factory, tmp_path
):
    with client_factory(
        FRAMES_ROOT=str(tmp_path / "frames"), UPLOAD_RESUME_MAX_PENDING="2"
    ) as client:
        _assert_pending_cap(client)


def _assert_pending_cap(client) -> None:
    for index in range(2):
        response = open_upload(client, expected_frames=12, shift_key=f"2026-10-22-P{index}")
        assert response.status_code == 201, response.text
    blocked = open_upload(client, expected_frames=12, shift_key="2026-10-22-P9")
    assert blocked.status_code == 409, blocked.text
    detail = blocked.json()["detail"]
    assert detail["error_code"] == "UPLOAD_TOO_MANY_PENDING"
    assert "cleanup" in detail["message"]


def test_client_supplied_upload_id_must_be_a_safe_directory_name(resume_client):
    """upload_id 同时是暂存目录名：不合约定的字符不能拼进磁盘路径，已用过的要点名。"""
    client = resume_client
    for bad in ("../../etc/passwd", "a/b", "x" * 65, "upl with space"):
        response = client.post(UPLOADS_URL, data=open_form(upload_id=bad))
        assert response.status_code == 422, (bad, response.text[:120])
        assert issue_codes(response) == {"upload_id_malformed"}

    taken = open_upload(client, upload_id="upl_taken_32", expected_frames=24)
    assert taken.status_code == 201, taken.text
    staging = staging_root(frames_root_of(client)) / "upl_taken_32"
    again = client.post(UPLOADS_URL, data=open_form(upload_id="upl_taken_32"))
    assert again.status_code == 422, again.text
    assert issue_codes(again) == {"upload_id_taken"}
    assert staging.parent == staging_root(frames_root_of(client))  # 没拼出目录树外面


# ── 仓储层直测（不经 HTTP：事务与幂等口径）─────────────────────────────────


def test_upload_repository_dedupes_inside_one_transaction(conn, tmp_path):
    uploads = UploadRepository(conn)
    record = uploads.open_upload(
        {
            "rack_id": "R-EXAMPLE-32",
            "bed_no": "EXAMPLE-BED-32",
            "infusion_order_no": "EXAMPLE-ORDER-0032",
            "shift_key": "2026-10-22-AM",
            "captured_at": EXAMPLE_FIELDS["captured_at"],
            "slots_json": json.dumps([{"bag_seq": "B01", "slot_index": 0}]),
            "session_key": upload_key("R-EXAMPLE-32", "EXAMPLE-ORDER-0032", "2026-10-22-AM"),
            "staging_dir": str(tmp_path / "staging"),
            "expected_frames": 20,
            "expires_at": "2099-01-01T00:00:00+00:00",
        }
    )
    upload_id = record["upload_id"]
    uploads.record_chunk(
        upload_id,
        accepted_frames=[0, 1, 2],
        frame_times={0: "2026-10-22T09:41:00+08:00"},
        duplicate_frames=[],
        added_bytes=10,
        expires_at="2099-01-01T00:00:00+00:00",
    )
    uploads.record_chunk(
        upload_id,
        accepted_frames=[3],
        frame_times={3: "2026-10-22T09:41:01.5+08:00"},
        duplicate_frames=[0, 1],
        added_bytes=4,
        expires_at="2099-01-01T00:00:00+00:00",
    )
    row = uploads.get(upload_id)
    assert row["received_frames"] == [0, 1, 2, 3]
    assert row["duplicate_count"] == 2
    assert row["chunk_count"] == 2
    assert row["frame_times"]["3"] == "2026-10-22T09:41:01.5+08:00"
    assert uploads.count_pending() == 1
    uploads.mark_completed(upload_id, ["sess_x"])
    assert uploads.count_pending() == 0
    assert uploads.get(upload_id)["session_ids"] == ["sess_x"]


def test_purge_expired_only_touches_pending(tmp_path, conn):
    uploads = UploadRepository(conn)
    for index, status in enumerate(("pending", "completed", "pending")):
        uploads.open_upload(
            {
                "upload_id": f"upl_{index}",
                "rack_id": "R-EXAMPLE-32",
                "bed_no": "EXAMPLE-BED-32",
                "infusion_order_no": "EXAMPLE-ORDER-0032",
                "shift_key": "2026-10-22-AM",
                "captured_at": EXAMPLE_FIELDS["captured_at"],
                "slots_json": "[]",
                "session_key": f"key_{index}",
                "staging_dir": str(tmp_path / f"upl_{index}"),
                "expires_at": "2000-01-01T00:00:00+00:00",
            }
        )
        if status != "pending":
            uploads.mark_completed(f"upl_{index}", ["sess_x"])

    # 一个属于已完成会话的正式帧目录：清理策略必须连看都不看
    session_dir = tmp_path / "key_1"
    session_dir.mkdir(parents=True)
    (session_dir / "frame_00000.jpg").write_bytes(b"keep-me")

    result = purge_expired(uploads, tmp_path, retention_hours=0.0)
    assert sorted(result.expired) == ["upl_0", "upl_2"]
    assert uploads.get("upl_1")["status"] == "completed"
    assert (session_dir / "frame_00000.jpg").read_bytes() == b"keep-me"