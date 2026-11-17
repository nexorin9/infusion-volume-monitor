"""测试共用工具：程序合成帧序列 + 本地替身对端（**都不是**真实院内数据/真实系统）。"""

from __future__ import annotations

from .llm_stub import LLMStub, RecordedLLMRequest, completion_body
from .nurse_exec_stub import NurseExecStub, RecordedRequest
from .vos_archive_stub import VosArchiveStub, build_stub_archive, vos_archive_stub
from .synth import (
    EXAMPLE_FIELDS,
    BagSpec,
    FillMeasurement,
    RackScene,
    anchor_prompt,
    build_frame_zip,
    build_rack_zip,
    default_rack_scene,
    make_frame_image,
    measure_scene,
    render_rack_frame,
    write_rack_frames,
    zip_bytes,
)

__all__ = [
    "EXAMPLE_FIELDS",
    "VosArchiveStub",
    "build_stub_archive",
    "vos_archive_stub",
    "LLMStub",
    "BagSpec",
    "FillMeasurement",
    "NurseExecStub",
    "RecordedLLMRequest",
    "completion_body",
    "RackScene",
    "RecordedRequest",
    "anchor_prompt",
    "build_frame_zip",
    "build_rack_zip",
    "default_rack_scene",
    "make_frame_image",
    "measure_scene",
    "render_rack_frame",
    "write_rack_frames",
    "zip_bytes",
]
