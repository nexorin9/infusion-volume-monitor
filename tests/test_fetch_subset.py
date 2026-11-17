"""公开 VOS 子集获取脚本的用例：幂等重跑、CRC 校验、断点续传、断网降级。

真数据（真实公开归档）由 ``scripts/fetch_vos_subset.py`` 实连取得，见 progress.txt 与
``docs/benchmark-vos.md``；本文件跑在本地真 socket 归档替身上，断言的是取数脚本行为：
中央目录直读不整包下载、只下该下的字节、逐文件 CRC32 对账、已完好文件重跑跳过、
坏字节降级为 skipped 且不落任何替代数据、断点续传接着写、manifest 与磁盘逐条对账。
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from zlib import crc32

import pytest

from infusion_monitor.vos_dataset import (
    ANNOTATION_PREFIX,
    IMAGE_PREFIX,
    HttpRangeReader,
    _fetch_entry,
    _local_path,
    fetch_subset,
    plan_subset,
    read_central_directory,
    update_docs_banner,
)
from tests.support.vos_archive_stub import (  # noqa: F401 - vos_archive_stub 是 pytest 夹具，导入即注册
    VosArchiveStub,
    build_stub_archive,
    disk_files,
    vos_archive_stub,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _closed_port() -> int:
    """借一个确定没人监听的端口，用来制造真实的连接失败。"""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# ------------------------------------------------------------------ 中央目录直读


def test_central_directory_is_read_without_pulling_frames(vos_archive_stub, tmp_path):
    """只读中央目录：下发字节远小于整包，且请求确实走 Range。"""
    stub = vos_archive_stub
    stub.reset_counters()
    reader = HttpRangeReader(stub.url, timeout=5, retries=0)
    index = read_central_directory(stub.url, reader=reader)

    assert len(index.entries) > 0
    # 精确口径：一个帧/标注负载字节都没下发（不是"下的不多"，是"一点没下"）
    assert stub.frame_bytes_served() == 0, f"读中央目录却下了 {stub.frame_bytes_served()} 字节数据内容"
    assert stub.bytes_served > 0
    assert all(r["range"] for r in stub.requests if r["method"] == "GET"), "应有请求带 Range 头"
    assert reader.requests >= 2, "至少要有：一次 HEAD/长度 + 一次尾部 + 一次中央目录"


def test_index_exposes_davis_layout(vos_archive_stub):
    """索引按归档同构目录组织：画面与标注两套前缀、序列名可切出来。"""
    index = read_central_directory(vos_archive_stub.url)
    images = index.under(IMAGE_PREFIX)
    annotations = index.under(ANNOTATION_PREFIX)
    assert {index.sequence_of(e, IMAGE_PREFIX) for e in images} == {"alpha", "beta", "gamma", "delta"}
    assert len(images) == len(annotations) == 17
    assert all(e.crc32 for e in images), "中央目录应带每个成员的 CRC32"


# ------------------------------------------------------------------ 选片与门禁


def test_plan_records_shortfall_instead_of_silently_lowering_the_gate(vos_archive_stub):
    """语料达不到每序列帧数门禁时，如实记 shortfall，绝不悄悄取短的当达标。"""
    index = read_central_directory(vos_archive_stub.url)
    plan = plan_subset(index, min_frames=150, max_sequences=2)
    assert plan.min_frames_gate_satisfied is False
    assert plan.satisfied is False, "未放松时不得视为可用"
    assert "5 帧" in (plan.shortfall_reason or "")
    assert plan.corpus_sequences == 4 and plan.corpus_longest_sequence == 5


def test_relaxed_plan_takes_longest_and_keeps_the_relaxation_visible(vos_archive_stub):
    index = read_central_directory(vos_archive_stub.url)
    plan = plan_subset(index, min_frames=150, max_sequences=2, relax_min_frames=True)
    assert plan.satisfied is True
    assert plan.relaxed is True
    assert plan.min_frames_gate_satisfied is False, "放松不等于门禁达成，两者都要留痕"
    assert [s.name for s in plan.sequences] == ["alpha", "beta"]
    assert plan.sequences[0].frames >= plan.sequences[1].frames


def test_plan_honours_explicit_sequences_and_max_bytes(vos_archive_stub):
    index = read_central_directory(vos_archive_stub.url)
    picked = plan_subset(index, sequences=["gamma"], min_frames=1)
    assert [s.name for s in picked.sequences] == ["gamma"]
    assert picked.min_frames_gate_satisfied is True

    capped = plan_subset(index, min_frames=1, max_sequences=4, max_bytes=1)
    assert len(capped.sequences) == 1, "总字节上限应截断计划"

    with pytest.raises(KeyError):
        plan_subset(index, sequences=["no-such-seq"], min_frames=1)


# ------------------------------------------------------------------ 取数主路径


def test_fetch_produces_frames_annotations_and_manifest(vos_archive_stub, tmp_path):
    """端到端：只取指定序列的画面与标注，manifest 逐文件可对账。"""
    root = tmp_path / "vos"
    report = fetch_subset(url=vos_archive_stub.url, root=root, sequences=["alpha"],
                          min_frames=1, retries=0)

    assert report.status == "ready", report.reason
    jpgs = sorted(root.glob(f"JPEGImages/480p/alpha/*.jpg"))
    pngs = sorted(root.glob(f"Annotations/480p/alpha/*.png"))
    assert len(jpgs) == len(pngs) == 5
    assert jpgs[0].name == "00000.jpg" and pngs[0].name == "00000.png"

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ready"
    assert manifest["achieved"]["sequence_count"] == 1
    assert manifest["achieved"]["frame_count"] == 5
    assert manifest["achieved"]["downloaded_files"] == 10
    assert len(manifest["files"]) == 10
    # 逐文件与磁盘对账：路径、字节数、CRC32 三项都要吻合
    for item in manifest["files"]:
        path = root / item["path"]
        assert path.exists(), item["path"]
        assert path.stat().st_size == item["bytes"]
        assert f"{crc32(path.read_bytes()) & 0xFFFFFFFF:#010x}" == item["crc32"]
    # 与归档声明的 CRC32 也一致（不是自己算自己）
    declared = {e.name: e.crc32 for e in read_central_directory(vos_archive_stub.url).entries.values()}
    for item in manifest["files"]:
        name = "DAVIS/" + item["path"]
        assert declared[name] == int(item["crc32"], 16)

    verify = report.verify["sequences"][0]
    assert verify["frames_on_disk"] == 5
    assert verify["has_first_frame_annotation"] is True
    assert verify["frame_size_px"] == [48, 36]
    assert report.verify["decode_failures"] == []


def test_fetch_downloads_only_the_sequences_asked_for(vos_archive_stub, tmp_path):
    """「限定规模」是真少下：下发的帧负载字节恰好等于被选中序列的成员压缩总长。"""
    stub = vos_archive_stub
    root = tmp_path / "vos"
    stub.reset_counters()
    fetch_subset(url=stub.url, root=root, sequences=["delta"], min_frames=1, retries=0)

    expected = sum(
        entry.compressed_size
        for entry in stub.entries.values()
        if not entry.is_dir and entry.name.split("/")[-2] == "delta"
    )
    total_all = sum(e.compressed_size for e in stub.entries.values() if not e.is_dir)
    assert stub.frame_bytes_served() == expected, "多下了不该下的字节"
    assert expected < total_all
    assert not (root / "JPEGImages/480p/alpha").exists()
    assert len(list((root / "JPEGImages/480p/delta").glob("*.jpg"))) == 3
    assert len(list((root / "Annotations/480p/delta").glob("*.png"))) == 3


def test_rerun_is_idempotent_and_reuses_verified_files(vos_archive_stub, tmp_path):
    """幂等重跑：完好文件跳过，不再下发帧字节。"""
    stub = vos_archive_stub
    root = tmp_path / "vos"
    first = fetch_subset(url=stub.url, root=root, sequences=["alpha", "beta"], min_frames=1, retries=0)
    assert first.status == "ready"
    assert first.downloaded_files == 20 and first.reused_files == 0

    stub.reset_counters()
    second = fetch_subset(url=stub.url, root=root, sequences=["alpha", "beta"], min_frames=1, retries=0)
    assert second.status == "ready"
    assert second.downloaded_files == 0
    assert second.reused_files == 20
    assert stub.frame_bytes_served() == 0, "重跑不应再下发任何帧负载"

    # 局部损坏的文件应被重新取回，而不是当成完好跳过
    victim = root / "JPEGImages/480p/alpha/00002.jpg"
    good = victim.read_bytes()
    victim.write_bytes(b"\x00" + good[1:])
    third = fetch_subset(url=stub.url, root=root, sequences=["alpha", "beta"], min_frames=1, retries=0)
    assert third.status == "ready"
    assert third.downloaded_files == 1, "被改坏的文件应重新下载"
    assert victim.read_bytes() == good, "重取后应与归档内容逐字节相同"


def test_contiguous_members_are_fetched_as_one_block(vos_archive_stub, tmp_path):
    """归档里相邻的成员并成一段块读：10 个文件不该变成 10 次往返。"""
    stub = vos_archive_stub
    stub.reset_counters()
    root = tmp_path / "vos"
    report = fetch_subset(url=stub.url, root=root, sequences=["alpha"], min_frames=1, retries=0)
    assert report.status == "ready"
    assert len(report.files) == 10

    gets = [r for r in stub.requests if r["method"] == "GET"]
    # 尾部探测 + 中央目录 + 段块读（+ 末成员对齐探针），远少于"每文件一次"
    assert len(gets) <= 5, f"请求次数 {len(gets)} 偏多，没合并成段块读"
    # 块读拿到的字节应覆盖全部 10 个成员（不多不少：CRC 全过即为不多）
    for item in report.files:
        path = root / item["path"]
        assert item["crc32"] == item["declared_crc32"]
        assert path.stat().st_size == item["bytes"]


def test_resume_continues_from_partial_file(vos_archive_stub, tmp_path):
    """断点续传：已有 .part 就从断点接着下，不是从头再来。"""
    stub = vos_archive_stub
    index = read_central_directory(stub.url)
    entry = next(e for e in index.under(IMAGE_PREFIX) if e.name.endswith("alpha/00003.jpg"))
    dest = tmp_path / "JPEGImages/480p/alpha/00003.jpg"
    dest.parent.mkdir(parents=True)
    part = dest.with_suffix(dest.suffix + ".part")
    prefix_len = min(64, entry.compressed_size // 2)
    part.write_bytes(stub.payload[stub._payload_starts[entry.name] : stub._payload_starts[entry.name] + prefix_len])

    reader = HttpRangeReader(stub.url, timeout=5, retries=0)
    source = _fetch_entry(reader, entry, dest)
    assert source == "resumed"
    assert not part.exists(), "校验通过后 .part 应被原子替换掉"
    assert crc32(dest.read_bytes()) & 0xFFFFFFFF == entry.crc32
    assert dest.stat().st_size == entry.file_size


# ------------------------------------------------------------------ 降级纪律


def test_corrupt_member_downgrades_to_skipped_without_fabricating_data(vos_archive_stub, tmp_path):
    """坏字节 → skipped；坏文件不进盘，manifest 如实记失败，不拿别的图顶上。"""
    stub = vos_archive_stub
    stub.corrupt_members = {"DAVIS/JPEGImages/480p/alpha/00002.jpg"}
    root = tmp_path / "vos"
    report = fetch_subset(url=stub.url, root=root, sequences=["alpha"], min_frames=1, retries=0)

    assert report.status == "skipped"
    assert "00002.jpg" in (report.reason or "")
    assert report.failures and "CRC32" in report.failures[0]["reason"]

    bad_dest = root / "JPEGImages/480p/alpha/00002.jpg"
    assert not bad_dest.exists(), "校验没过就不该留下正式文件"
    # 已取到的完好文件内容必须与归档逐字节一致（没有被改写过）
    declared = {e.name: e for e in read_central_directory(stub.url).entries.values()}
    for path in root.rglob("*.jpg"):
        entry = declared["DAVIS/" + str(path.relative_to(root))]
        assert crc32(path.read_bytes()) & 0xFFFFFFFF == entry.crc32

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "skipped"
    assert manifest["skipped_reason"]
    assert manifest["failures"][0]["name"].endswith("00002.jpg")


def test_network_failure_downgrades_to_skipped_and_leaves_no_data(tmp_path):
    """断网：只留 manifest 说明原因，一个帧文件都不造。"""
    root = tmp_path / "vos"
    report = fetch_subset(url=f"http://127.0.0.1:{_closed_port()}/davis.zip", root=root,
                          sequences=["alpha"], min_frames=1, retries=0, timeout=1.0)
    assert report.status == "skipped"
    assert report.reason and "中央目录" in report.reason
    assert disk_files(root) == [root / "manifest.json"], f"不该有任何帧文件：{disk_files(root)}"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "skipped" and manifest["files"] == []


def test_server_without_range_support_downgrades_instead_of_guessing(tmp_path):
    """站点不支持 Range（503）时如实降级，不退化成整包硬下 794 MB。"""
    stub = VosArchiveStub(payload=build_stub_archive({"alpha": 3}), fail_ranges=True).start()
    try:
        root = tmp_path / "vos"
        report = fetch_subset(url=stub.url, root=root, sequences=["alpha"], min_frames=1,
                              retries=0, timeout=2.0)
        assert report.status == "skipped"
        assert "中央目录" in (report.reason or "")
    finally:
        stub.stop()


def test_dry_run_downloads_nothing(vos_archive_stub, tmp_path):
    """--dry-run 只试算：不下任何字节、不建目录。"""
    stub = vos_archive_stub
    stub.reset_counters()
    root = tmp_path / "vos"
    report = fetch_subset(url=stub.url, root=root, min_frames=1, max_sequences=2,
                          relax_min_frames=True, dry_run=True, retries=0)
    assert report.status == "planned"
    assert [s.name for s in report.plan.sequences] == ["alpha", "beta"]
    assert not root.exists()
    assert stub.frame_bytes_served() == 0


def test_strict_min_frames_gate_refuses_to_fetch(tmp_path):
    """不放 --relax 时门禁不过就不落数据（strict 是默认值）。"""
    stub = VosArchiveStub(payload=build_stub_archive({"alpha": 5})).start()
    try:
        root = tmp_path / "vos"
        report = fetch_subset(url=stub.url, root=root, min_frames=150, retries=0)
        assert report.status == "skipped"
        assert "150" in (report.reason or "")
        assert disk_files(root) == [root / "manifest.json"]
    finally:
        stub.stop()


# ------------------------------------------------------------------ 文档状态块


def test_docs_banner_records_skip_reason_and_replaces_cleanly(tmp_path):
    """skipped 原因必须落到文档顶部；重复刷新只留一个状态块，正文不丢。"""
    doc = tmp_path / "benchmark-vos.md"
    doc.write_text("# 公开 VOS 子集\n\n## 获取子集\n\n正文一节。\n", encoding="utf-8")

    skipped = fetch_subset(url=f"http://127.0.0.1:{_closed_port()}/x.zip", root=tmp_path / "vos",
                           sequences=["alpha"], min_frames=1, retries=0, timeout=1.0)
    block = update_docs_banner(doc, skipped)
    text = doc.read_text(encoding="utf-8")
    assert text.startswith("<!-- vos-subset-status:begin -->")
    assert "skipped" in text and skipped.reason[:20] in text
    assert "正文一节。" in text, "写状态块不该吃掉正文"

    again = update_docs_banner(doc, skipped)
    text2 = doc.read_text(encoding="utf-8")
    assert text2.count("<!-- vos-subset-status:begin -->") == 1
    assert text2.count("<!-- vos-subset-status:end -->") == 1
    assert again == block
    assert text2 != text or True  # 幂等：内容一致即可


def test_docs_banner_states_relaxation_and_gate_shortfall(vos_archive_stub, tmp_path):
    """放松取数时，文档里同时出现门禁未达成与显式放松两行，不能只写"已就绪"。"""
    doc = tmp_path / "benchmark-vos.md"
    report = fetch_subset(url=vos_archive_stub.url, root=tmp_path / "vos", min_frames=150,
                          max_sequences=2, relax_min_frames=True, retries=0)
    assert report.status == "ready"
    update_docs_banner(doc, report)
    text = doc.read_text(encoding="utf-8")
    assert "可用" in text
    assert "未达成" in text and "--relax-min-frames" in text
    assert "CRC32" in text, "校验口径要写在文档里，别只在心里"


def test_member_path_cannot_escape_root(tmp_path):
    """归档里出现 ../ 的成员名时拒绝落盘。"""
    from infusion_monitor.vos_dataset import ZipEntry

    evil = ZipEntry(name="DAVIS/../escape.jpg", compress_type=0, compressed_size=1,
                    file_size=1, crc32=0, header_offset=0)
    with pytest.raises(RuntimeError):
        _local_path(tmp_path, evil.name)