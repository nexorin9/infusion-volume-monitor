"""入站帧序列的硬校验与接入端到端（task 3）。

覆盖：
* 正常序列通过 → 201，返回 ``session_id`` / 帧数 / 每袋槽位；
* 帧数不足、尺寸过小、时间戳逆序、slots 超 6 袋 → 422，且每条 issue 都带补拍建议；
* zip 结构不安全（路径穿越 / 杂项成员）→ 422 FRAMES_ZIP_REJECTED；
* 缺表单字段时 FastAPI 校验错误也映射成同一套三段式 detail。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from infusion_monitor.errors import ValidationRejected
from infusion_monitor.store.schema import SCHEMA_VERSION
from infusion_monitor.ingest.validate import (
    MAX_SLOTS,
    MIN_FRAMES,
    MIN_SHORT_SIDE,
    collect_frame_issues,
    parse_upload_fields,
    validate_frame_sizes,
    validate_frame_times,
)
from infusion_monitor.ingest.zip_frames import upload_key
from tests.support import EXAMPLE_FIELDS, build_frame_zip, zip_bytes

TRACKS_URL = "/api/v1/infusion/tracks"


def form_fields(**overrides) -> dict[str, str]:
    fields = dict(EXAMPLE_FIELDS)
    for key, value in overrides.items():
        if value is None:
            fields.pop(key, None)
        else:
            fields[key] = value
    return fields


def post_tracks(client, zip_payload: bytes, **overrides):
    fields = form_fields(**overrides)
    files = {"frames": ("rack.zip", zip_payload, "application/zip")}
    return client.post(TRACKS_URL, data=fields, files=files)


def codes(response) -> set[str]:
    return {issue["code"] for issue in response.json()["detail"]["issues"]}


def suggestions(response) -> list[str]:
    return [issue["suggestion"] for issue in response.json()["detail"]["issues"]]


# ── 正常路径 ────────────────────────────────────────────────────────────────


def test_valid_sequence_accepted_and_persisted(client, tmp_path):
    payload = build_frame_zip(frame_count=14)
    response = post_tracks(client, payload)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["frame_count"] == 14
    assert body["first_frame"] == "frame_00000.jpg"
    assert body["last_frame"] == "frame_00013.jpg"
    assert body["shift_key"] == "2026-10-22"  # 未显式给轮次时按拍摄日期生成
    assert body["frame_time_source"] == "zip_member"

    assert len(body["bags"]) == 1
    bag = body["bags"][0]
    assert bag["bag_seq"] == "B01"
    assert bag["slot_index"] == 0
    assert bag["bag_type_code"] == "TYPE-A"
    assert bag["frame_count"] == 14
    assert bag["state"] == "monitoring"
    assert bag["calibration_state"] == "uncalibrated"
    assert bag["session_id"].startswith("sess_")

    # 帧真的落盘了，且按 frame_%05d.jpg 规范命名
    frames_dir = Path(bag["frames_dir"])
    assert frames_dir.is_dir()
    assert sorted(p.name for p in frames_dir.glob("frame_*.jpg"))[:2] == [
        "frame_00000.jpg",
        "frame_00001.jpg",
    ]
    assert len(list(frames_dir.glob("frame_*.jpg"))) == 14


def test_upload_creates_one_session_per_slot(client):
    slots = json.dumps(
        [
            {"bag_seq": "B01", "slot_index": 0},
            {"bag_seq": "B02", "slot_index": 1},
            {"bag_seq": "B03", "slot_index": 2, "bag_type_code": "TYPE-B"},
        ]
    )
    response = post_tracks(client, build_frame_zip(frame_count=12), slots=slots)

    assert response.status_code == 201, response.text
    bags = response.json()["bags"]
    assert [bag["bag_seq"] for bag in bags] == ["B01", "B02", "B03"]
    assert [bag["slot_index"] for bag in bags] == [0, 1, 2]
    assert bags[2]["bag_type_code"] == "TYPE-B"
    # 三袋共享同一段上传的帧目录
    assert len({bag["frames_dir"] for bag in bags}) == 1
    assert len({bag["session_id"] for bag in bags}) == 3


def test_health_reports_schema_and_session_count(client):
    post_tracks(client, build_frame_zip(frame_count=12))
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["schema_version"] == SCHEMA_VERSION
    assert body["session_count"] == 1
    assert body["frames_root_writable"] is True


def test_session_lookup_returns_persisted_row(client):
    body = post_tracks(client, build_frame_zip(frame_count=12)).json()
    session_id = body["bags"][0]["session_id"]

    detail = client.get(f"/api/v1/infusion/sessions/{session_id}")
    assert detail.status_code == 200
    assert detail.json()["bag_seq"] == "B01"
    assert detail.json()["infusion_order_no"] == EXAMPLE_FIELDS["infusion_order_no"]

    missing = client.get("/api/v1/infusion/sessions/sess_not_exist")
    assert missing.status_code == 404
    assert missing.json()["detail"]["error_code"] == "NOT_FOUND"


def test_explicit_frame_times_take_precedence_over_zip_mtimes(client):
    base = datetime.fromisoformat("2026-10-22T09:41:00+08:00")
    frame_times = json.dumps(
        [(base + timedelta(seconds=0.4 * i)).isoformat() for i in range(12)]
    )
    body = post_tracks(client, build_frame_zip(frame_count=12), frame_times=frame_times).json()
    assert body["frame_time_source"] == "frame_times"


def test_explicit_shift_key_is_used(client):
    body = post_tracks(client, build_frame_zip(frame_count=12), shift_key="2026-10-22-AM").json()
    assert body["shift_key"] == "2026-10-22-AM"


# ── 422：帧数不足 ───────────────────────────────────────────────────────────


def test_too_few_frames_rejected_with_retake_hint(client):
    response = post_tracks(client, build_frame_zip(frame_count=6))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error_code"] == "INGEST_INVALID"
    assert "frames_too_few" in {issue["code"] for issue in detail["issues"]}
    issue = next(i for i in detail["issues"] if i["code"] == "frames_too_few")
    assert str(6) in issue["reason"]
    assert "连续拍" in issue["suggestion"]


# ── 422：分辨率过小 ─────────────────────────────────────────────────────────


def test_short_side_below_threshold_rejected_with_zoom_hint(client):
    response = post_tracks(client, build_frame_zip(frame_count=12, width=160, height=240))
    assert response.status_code == 422
    issue = next(i for i in response.json()["detail"]["issues"] if i["code"] == "frame_too_small")
    assert str(MIN_SHORT_SIDE) in issue["reason"]
    assert "靠近输液架" in issue["suggestion"]


# ── 422：时间戳逆序 ─────────────────────────────────────────────────────────


def test_frame_times_out_of_order_rejected(client):
    base = datetime.fromisoformat("2026-10-22T09:41:00+08:00")
    frame_times = [
        (base + timedelta(seconds=0.5 * i)).isoformat() for i in range(12)
    ]
    frame_times[7], frame_times[8] = frame_times[8], frame_times[7]  # 人为制造逆序

    response = post_tracks(client, build_frame_zip(frame_count=12), frame_times=json.dumps(frame_times))
    assert response.status_code == 422
    issue = next(
        i for i in response.json()["detail"]["issues"] if i["code"] == "frame_times_out_of_order"
    )
    assert "逆序" in issue["reason"]
    assert "不要跳帧" in issue["suggestion"]


def test_all_frames_sharing_one_timestamp_rejected():
    same = datetime(2026, 10, 22, 9, 41, 0)
    with pytest.raises(ValidationRejected) as excinfo:
        validate_frame_times([same] * 12)
    assert "frame_times_flat" in {issue.code for issue in excinfo.value.issues}


def test_equal_timestamps_allowed_but_flat_sequence_is_not():
    base = datetime(2026, 10, 22, 9, 41, 0)
    times = [base + timedelta(seconds=2 * i) for i in range(12)]
    validate_frame_times(times)  # 非递减：zip 2 秒精度下相邻同秒是正常的


# ── 422：袋位超限 ───────────────────────────────────────────────────────────


def test_more_than_six_slots_rejected(client):
    slots = json.dumps([{"bag_seq": f"B{index:02d}", "slot_index": index} for index in range(7)])
    response = post_tracks(client, build_frame_zip(frame_count=12), slots=slots)
    assert response.status_code == 422
    issue = next(i for i in response.json()["detail"]["issues"] if i["code"] == "too_many_slots")
    assert str(MAX_SLOTS) in issue["reason"]
    assert "分批" in issue["suggestion"]


def test_duplicate_slot_index_rejected(client):
    slots = json.dumps([{"bag_seq": "B01", "slot_index": 0}, {"bag_seq": "B02", "slot_index": 0}])
    response = post_tracks(client, build_frame_zip(frame_count=12), slots=slots)
    assert response.status_code == 422
    assert "slot_duplicate_index" in codes(response)


def test_duplicate_bag_seq_rejected(client):
    slots = json.dumps([{"bag_seq": "B01", "slot_index": 0}, {"bag_seq": "B01", "slot_index": 1}])
    response = post_tracks(client, build_frame_zip(frame_count=12), slots=slots)
    assert response.status_code == 422
    assert "slot_duplicate_bag_seq" in codes(response)


# ── 422：zip 结构不安全 ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "member",
    ["../escape.jpg", "/etc/passwd.jpg", "__MACOSX/._frame_00000.jpg", "readme.txt"],
)
def test_foreign_or_traversing_members_rejected(client, member):
    payload = zip_bytes(extra_members={member: b"not-a-frame"})
    response = post_tracks(client, payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error_code"] == "FRAMES_ZIP_REJECTED"
    assert detail["issues"][0]["code"] == "frame_member_unsafe"


def test_non_zip_upload_rejected(client):
    response = post_tracks(client, b"this is not a zip file")
    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "FRAMES_ZIP_REJECTED"


def test_empty_zip_rejected(client):
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w"):
        pass
    response = post_tracks(client, buffer.getvalue())
    assert response.status_code == 422
    assert "frames_zip_empty" in codes(response)


def test_zero_byte_frames_rejected(client):
    response = post_tracks(client, b"")
    assert response.status_code == 422
    assert "frames_empty_file" in codes(response)


# ── 422：业务头字段 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("field_name", ["rack_id", "infusion_order_no", "bed_no", "captured_at", "slots"])
def test_missing_required_field_maps_to_structured_detail(client, field_name):
    response = post_tracks(client, build_frame_zip(frame_count=12), **{field_name: None})
    assert response.status_code == 422
    issue = next(
        i for i in response.json()["detail"]["issues"] if i["field"] == field_name
    )
    assert issue["code"] == "field_missing"
    assert issue["suggestion"]


def test_non_iso8601_captured_at_rejected(client):
    response = post_tracks(client, build_frame_zip(frame_count=12), captured_at="22/10/2026 09:41")
    assert response.status_code == 422
    assert "captured_at_not_iso8601" in codes(response)


def test_slots_not_json_rejected(client):
    response = post_tracks(client, build_frame_zip(frame_count=12), slots="B01,B02")
    assert response.status_code == 422
    assert "slots_not_json" in codes(response)


# ── 一次把所有问题都回给护士，而不是修一个报一个 ───────────────────────────


def test_multiple_violations_reported_together(tmp_path):
    from PIL import Image

    frames = tmp_path / "small"
    frames.mkdir()
    paths = []
    base = datetime(2026, 10, 22, 9, 41, 0)
    times = [base + timedelta(seconds=0.5 * i) for i in range(5)]
    times[3], times[4] = times[4], times[3]
    for index in range(5):
        path = frames / f"frame_{index:05d}.jpg"
        Image.new("RGB", (200, 300), (10, 10, 10)).save(path, format="JPEG")
        paths.append(path)

    with pytest.raises(ValidationRejected) as excinfo:
        collect_frame_issues(
            names=[p.name for p in paths], times=times, paths=paths
        )

    found = {issue.code for issue in excinfo.value.issues}
    assert {"frames_too_few", "frame_times_out_of_order", "frame_too_small"} <= found


# ── 覆盖式重传：同 rack + 轮次 + 单号落同一目录且不追加重复帧 ───────────────


def test_reupload_overwrites_instead_of_appending(client):
    first = post_tracks(client, build_frame_zip(frame_count=12)).json()
    second = post_tracks(client, build_frame_zip(frame_count=20)).json()

    assert first["bags"][0]["session_id"] == second["bags"][0]["session_id"]
    assert second["bags"][0]["session_version"] == 2
    assert second["frame_count"] == 20

    frames_dir = Path(second["bags"][0]["frames_dir"])
    assert len(list(frames_dir.glob("frame_*.jpg"))) == 20

    detail = client.get(f"/api/v1/infusion/sessions/{second['bags'][0]['session_id']}").json()
    assert detail["frame_count"] == 20


def test_upload_key_is_stable_and_path_safe():
    first = upload_key("R-07A", "EXAMPLE-ORDER-0001", "2026-10-22-AM")
    again = upload_key("R-07A", "EXAMPLE-ORDER-0001", "2026-10-22-AM")
    other = upload_key("R-08B", "EXAMPLE-ORDER-0001", "2026-10-22-AM")
    assert first == again
    assert first != other
    assert "/" not in first and "<" not in first and ">" not in first


# ── 校验函数的单元口径 ─────────────────────────────────────────────────────


def test_size_validation_accepts_boundary(tmp_path):
    from PIL import Image

    path = tmp_path / "frame_00000.jpg"
    Image.new("RGB", (MIN_SHORT_SIDE, MIN_SHORT_SIDE + 40), (30, 30, 30)).save(path, format="JPEG")
    validate_frame_sizes([path])  # 短边正好等于下限应通过


def test_corrupt_frame_is_not_silently_skipped(tmp_path):
    broken = tmp_path / "frame_00000.jpg"
    broken.write_bytes(b"not an image at all")
    with pytest.raises(ValidationRejected) as excinfo:
        validate_frame_sizes([broken])
    assert "frame_unreadable" in {issue.code for issue in excinfo.value.issues}


def test_min_frame_count_is_twelve():
    assert MIN_FRAMES == 12
    names = [f"frame_{index:05d}.jpg" for index in range(MIN_FRAMES)]
    base = datetime(2026, 10, 22, 9, 41, 0)
    times = [base + timedelta(seconds=0.5 * i) for i in range(MIN_FRAMES)]
    assert collect_frame_issues(names=names, times=times, paths=[]) == []


def test_field_parser_reports_every_missing_field_at_once():
    with pytest.raises(ValidationRejected) as excinfo:
        parse_upload_fields({"rack_id": "", "infusion_order_no": "", "bed_no": ""})
    missing = {issue.field for issue in excinfo.value.issues}
    assert {"rack_id", "infusion_order_no", "bed_no", "captured_at", "slots"} <= missing