"""OpenAPI 契约与错误码表的三组契约断言（task 36）。

分组：

1. **必填字段** —— 入站 multipart 的必填清单、帧 zip 的硬校验上限、以及
   ``estimated_volume_ml`` 这类**门禁字段**在契约里就是可选的、真实响应里干脆不出现。
2. **错误结构** —— 422/409/404/500 的 detail 恒为 ``error_code / message / issues[四段]``；
   错误码表与 ``/docs`` 各接口 responses **逐一对应**（不多不少）；业务代码里
   ``raise`` 的每个码都被登记过。
3. **OpenAPI 可生成** —— 文档能生成、能被 JSON 解析、两次生成一致、``/docs`` 可打开。

外加一条"文档说的和线上返回的是同一件事"的实跑核对（走 TestClient 真实发请求，
不是对着文档自说自话）。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from infusion_monitor.api.contract import (
    ERROR_DETAIL_REF,
    OUTBOUND_ONLY_CODES,
    ROUTE_ERROR_CODES,
    all_operations,
    codes_by_status_in_openapi,
    codes_documented_for,
    codes_in_openapi,
    validate_route_table,
)
from infusion_monitor.api.schemas import (
    BagSlotOut,
    ErrorDetailOut,
    ErrorIssueOut,
    SlotSpecIn,
    TrackUploadForm,
    TrackUploadResponse,
    track_upload_body_schema,
)
from infusion_monitor.errors import ERROR_CODES, error_code_docs, operational_faults
from tests.support import EXAMPLE_FIELDS, build_frame_zip

TRACKS_URL = "/api/v1/infusion/tracks"

#: 契约里「门禁字段」——条件不满足时字段整体不出现，而不是 null 占位。
GATED_FIELDS = (
    "estimated_volume_ml",
    "eta_minutes",
    "rate_ml_per_h",
    "calibration_version",
    "track_handoff",
    "fill_ratio",
)

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "infusion_monitor"


@pytest.fixture
def spec(client):
    """OpenAPI 文档（走真实应用工厂，不用静态文件）。"""
    return client.get("/openapi.json").json()


# ── 1. 必填字段 ────────────────────────────────────────────────────────────


def test_upload_body_documents_multipart_with_required_list(spec):
    body = spec["paths"][TRACKS_URL]["post"]["requestBody"]
    assert set(body["content"]) == {"multipart/form-data"}
    schema = spec["components"]["schemas"]["TrackUploadForm"]

    # 五个业务头字段 + 帧 zip 缺一不可；可选项不得混进必填清单。
    assert schema["required"] == [
        "rack_id",
        "infusion_order_no",
        "bed_no",
        "captured_at",
        "slots",
        "frames",
    ]
    assert set(schema["properties"]) == set(TrackUploadForm.model_fields) | {"frames"}
    assert schema["properties"]["frames"]["format"] == "binary"
    assert schema["additionalProperties"] is False

    # 每个字段都带人话说明，缺说明的字段挂接方只能去问信息科。
    undocumented = [
        name
        for name, prop in schema["properties"].items()
        if not str(prop.get("description", "")).strip()
    ]
    assert undocumented == []


def test_required_fields_match_the_validator_and_the_live_endpoint(client):
    """文档的必填清单必须等于校验器真会拒的那些字段。"""
    documented = set(track_upload_body_schema()["required"]) - {"frames"}
    assert documented == set(TrackUploadForm.required_field_names())

    payload = build_frame_zip(frame_count=14)
    files = {"frames": ("rack.zip", payload, "application/zip")}
    for field in sorted(documented):
        data = dict(EXAMPLE_FIELDS)
        data.pop(field)
        response = client.post(TRACKS_URL, data=data, files=files)
        assert response.status_code == 422, f"{field} 缺失却没被拒：{response.text}"
        issue_fields = {issue["field"] for issue in response.json()["detail"]["issues"]}
        assert field in issue_fields, (field, issue_fields)


def test_slots_item_schema_matches_the_validated_slot_spec(spec):
    item = spec["components"]["schemas"]["TrackUploadForm"]["x-slots-item-schema"]
    assert set(item["properties"]) == set(SlotSpecIn.model_fields)
    assert sorted(item["required"]) == ["bag_seq", "slot_index"]


def test_response_models_carry_the_documented_gated_fields(spec):
    bag_schema = spec["components"]["schemas"]["BagSlotOut"]
    for field in GATED_FIELDS:
        assert field in bag_schema["properties"], f"{field} 没进契约"
        assert field not in bag_schema.get("required", []), f"{field} 不该是必填"

    top = spec["components"]["schemas"]["TrackUploadResponse"]
    assert top["required"] == [
        "upload_key",
        "rack_id",
        "bed_no",
        "infusion_order_no",
        "captured_at",
        "shift_key",
        "frame_count",
        "first_frame",
        "last_frame",
        "frame_time_source",
        "bags",
        "accepted_at",
    ]


def test_gated_fields_are_absent_from_a_real_uncalibrated_response(client):
    """未登记袋型标定时，字段**整体不出现**——不是 null 占位。"""
    payload = build_frame_zip(frame_count=14)
    files = {"frames": ("rack.zip", payload, "application/zip")}
    response = client.post(TRACKS_URL, data=dict(EXAMPLE_FIELDS), files=files)
    assert response.status_code == 201, response.text

    bag = response.json()["bags"][0]
    assert bag["calibration_state"] == "uncalibrated"
    schema = BagSlotOut.model_json_schema()
    for field in ("estimated_volume_ml", "eta_minutes", "rate_ml_per_h", "calibration_version"):
        assert field in schema["properties"], f"{field} 没进 Pydantic 契约"
        assert field not in schema.get("required", []), f"{field} 不该是必填"
        assert field not in bag, f"{field} 门禁未满足却出现在响应里：{bag}"
        assert bag.get(field, "absent") == "absent"

    # 契约可选 ≠ 校验器放水：整个袋位对象仍能装回响应模型，值确实为 None。
    parsed = BagSlotOut.model_validate(bag)
    assert parsed.estimated_volume_ml is None
    assert parsed.calibration_state == "uncalibrated"
    # 反面：观测值该给的照给（门禁只压体积与时点，不压面积比）
    assert bag["fill_ratio"] > 0


# ── 2. 错误结构 ────────────────────────────────────────────────────────────


def test_error_detail_component_matches_the_three_part_contract(spec):
    detail = spec["components"]["schemas"]["ErrorDetailOut"]
    assert sorted(detail["required"]) == ["error_code", "message"]
    assert detail["properties"]["issues"]["items"] == {"$ref": "#/components/schemas/ErrorIssueOut"}

    issue = spec["components"]["schemas"]["ErrorIssueOut"]
    assert sorted(issue["required"]) == ["code", "field", "reason", "suggestion"]

    # 与 Pydantic 模型逐字段一致（文档不是手写的第二份真相）。
    assert set(issue["properties"]) == set(ErrorIssueOut.model_fields)
    assert set(detail["properties"]) == set(ErrorDetailOut.model_fields)


def test_no_operation_still_advertises_the_fastapi_default_error_shape(spec):
    """FastAPI 默认的 422 指向 HTTPValidationError，与真实返回体不符，必须换掉。"""
    for method, path in all_operations(spec):
        responses = spec["paths"][path][method.lower()].get("responses", {})
        for status, response in responses.items():
            schema = (
                response.get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            assert schema.get("$ref") != "#/components/schemas/HTTPValidationError", (
                f"{method} {path} 的 {status} 仍指向 FastAPI 默认结构"
            )


def test_error_code_table_matches_openapi_one_by_one(spec):
    documented = codes_in_openapi(spec)
    # 不多不少：出站故障码（UPSTREAM_UNAVAILABLE）按设计不进入站 responses。
    assert documented == set(ERROR_CODES) - set(OUTBOUND_ONLY_CODES)

    # 每一类状态码在 /docs 里都能查到对应的码与中文处理方式。
    for method, path in all_operations(spec):
        responses = spec["paths"][path][method.lower()]["responses"]
        for code in codes_documented_for(method, path):
            response = responses.get(str(ERROR_CODES[code]))
            assert response is not None, f"{method} {path} 少写了 {code} 的响应"
            description = response["description"]
            assert f"{code}：" in description, description
            assert len(description) > 40, f"{code} 的说明太短，挂接方看不出怎么处理"

    by_status = codes_by_status_in_openapi(spec)
    # 任务要求的四类：422 校验 / 409 状态非法 / 404 未找到 / 500 内部。
    for status in (422, 409, 404, 500):
        assert by_status.get(status), f"HTTP {status} 这一类在 /docs 里查不到任何错误码"
    # 共处同一状态码的码必须写进同一条响应（OpenAPI 一个 status 只放一个 response），
    # 否则后写的会顶掉先写的，挂接方在 /docs 里只看得到最后一个码。
    assert {"INGEST_INVALID", "FRAMES_ZIP_REJECTED"} <= by_status[422]


def test_shared_status_codes_are_merged_not_overwritten(spec):
    """/tracks 的两个 422 码都在同一条响应描述里点名。"""
    description = spec["paths"][TRACKS_URL]["post"]["responses"]["422"]["description"]
    for code in ("INGEST_INVALID", "FRAMES_ZIP_REJECTED"):
        assert f"{code}：" in description, description
    assert "处理方式：" in description  # 两段都带处理方式，不是只写一句


def test_every_operation_documents_the_framework_validation_shape(spec):
    """框架级校验错误恒映射成 INGEST_INVALID，所以每个接口都该写上 422。"""
    for method, path in all_operations(spec):
        responses = spec["paths"][path][method.lower()]["responses"]
        assert "422" in responses, f"{method} {path} 没写 422"
        assert "INGEST_INVALID：" in responses["422"]["description"]


def test_every_code_is_reachable_from_at_least_one_documented_route():
    validate_route_table()
    covered = {code for codes in ROUTE_ERROR_CODES.values() for code in codes}
    assert covered | set(OUTBOUND_ONLY_CODES) == set(ERROR_CODES)


def test_route_table_covers_every_operation_the_service_exposes(spec):
    registered = set(all_operations(spec))
    assert registered == set(ROUTE_ERROR_CODES), (
        f"接口表漏了 {sorted(registered - set(ROUTE_ERROR_CODES))}；"
        f"多写了 {sorted(set(ROUTE_ERROR_CODES) - registered)}"
    )


def test_raised_error_codes_are_all_documented():
    """业务代码里 ``raise`` 的每个码都必须登记到某个接口上。

    新增错误码却忘了写进接口表时，这里会红——这是本文件最有用的一条。
    """
    raised: set[str] = set()
    pattern = re.compile(r"""(?:ApiError\(\s*|error_code\s*=\s*)["']([A-Z_]+)["']""")
    for path in SRC_ROOT.rglob("*.py"):
        if path.name == "contract.py":  # 接口表本身就是登记处，不算 raise 点
            continue
        source = path.read_text(encoding="utf-8")
        ast.parse(source)  # 语法有问题时在这里就炸，不用等正则扫出半截字符串
        raised.update(pattern.findall(source))

    known = raised & set(ERROR_CODES)
    assert known, "扫描不到任何 raise 点，正则多半已经失效"
    undocumented = known - {code for codes in ROUTE_ERROR_CODES.values() for code in codes}
    assert undocumented == set(), f"这些码被真实抛出，却没登记进任何接口的文档：{sorted(undocumented)}"

    unknown = raised - set(ERROR_CODES)
    assert unknown == set(), f"代码里用了码表里没有的错误码：{sorted(unknown)}"


def test_live_error_responses_conform_to_the_documented_detail_shape(client):
    """线上真实返回的 422/404 与 /docs 里写的结构逐字段一致。"""
    payload = build_frame_zip(frame_count=6)  # 少于下限 12 帧
    files = {"frames": ("rack.zip", payload, "application/zip")}

    cases = [
        (client.post(TRACKS_URL, data=dict(EXAMPLE_FIELDS), files=files), 422, "INGEST_INVALID"),
        (
            client.post(
                TRACKS_URL,
                data={k: v for k, v in EXAMPLE_FIELDS.items() if k != "bed_no"},
                files=files,
            ),
            422,
            "INGEST_INVALID",
        ),
        (client.get("/api/v1/infusion/sessions/sess-does-not-exist"), 404, "NOT_FOUND"),
    ]

    saw_issue_list = False
    for response, status, expected_code in cases:
        assert response.status_code == status, response.text
        detail = response.json()["detail"]
        assert detail["error_code"] == expected_code
        assert detail["message"].strip()
        # issues 在契约里是可选的（404 这种一句话说得清的错不带 issues），
        # 但只要带了就必须逐条四段齐全。
        ErrorDetailOut.model_validate(detail)
        for issue in detail.get("issues", ()):
            saw_issue_list = True
            assert set(issue) == {"code", "field", "reason", "suggestion"}
            assert all(str(value).strip() for value in issue.values()), issue
            # 建议是护士照着做的中文话，不是英文报错
            assert re.search(r"[一-鿿]", issue["suggestion"]), issue
    assert saw_issue_list, "422 应当至少带一条可照做的 issues"


def test_error_codes_endpoint_serves_the_same_table(spec, client):
    response = client.get("/api/v1/error-codes")
    assert response.status_code == 200
    body = response.json()
    assert {row["code"] for row in body["error_codes"]} == set(ERROR_CODES)
    for row in body["error_codes"]:
        assert row["status"] == ERROR_CODES[row["code"]]
        assert row["handling"].strip() and row["meaning"].strip()
    assert {fault["key"] for fault in body["operational_faults"]} == {
        fault.key for fault in operational_faults()
    }

    # 端点里的说明与 /docs 里写的是同一份（同一份 ERROR_CODES 展开）。
    descriptions = {
        doc.code: f"{doc.meaning}\n\n处理方式：{doc.handling}" for doc in error_code_docs()
    }
    for method, path in all_operations(spec):
        responses = spec["paths"][path][method.lower()]["responses"]
        for code in codes_documented_for(method, path):
            expected = f"{code}：{descriptions[code]}"
            assert expected in responses[str(ERROR_CODES[code])]["description"]


def test_upstream_unavailable_is_documented_as_an_outbound_fault(spec, client):
    """出站对端不可达不是入站错误：码表里查得到，但它不出现在任何 responses 里。"""
    assert "UPSTREAM_UNAVAILABLE" in OUTBOUND_ONLY_CODES
    assert "UPSTREAM_UNAVAILABLE" not in codes_in_openapi(spec)

    fault = next(f for f in operational_faults() if f.key == "nurse_exec_unreachable")
    assert "pending_writeback" in fault.impact

    rows = {row["code"]: row for row in client.get("/api/v1/error-codes").json()["error_codes"]}
    assert rows["UPSTREAM_UNAVAILABLE"]["routes"] == []


def test_error_examples_in_openapi_carry_the_three_part_issue(spec):
    responses = spec["paths"][TRACKS_URL]["post"]["responses"]
    example = responses["422"]["content"]["application/json"]["example"]["detail"]
    assert example["error_code"] in {"INGEST_INVALID", "FRAMES_ZIP_REJECTED"}
    assert set(example["issues"][0]) == {"code", "field", "reason", "suggestion"}
    assert responses["422"]["content"]["application/json"]["schema"] == {"$ref": ERROR_DETAIL_REF}


# ── 3. OpenAPI 可生成 ──────────────────────────────────────────────────────


def test_openapi_is_generatable_and_parses(spec):
    assert spec["openapi"] == "3.1.0"
    assert spec["info"]["title"]
    json.dumps(spec)  # 能被 JSON 序列化：/docs 与 SDK 生成器都要吃它
    assert ERROR_DETAIL_REF.rsplit("/", 1)[-1] in spec["components"]["schemas"]


def test_openapi_generation_is_deterministic(client):
    first = client.get("/openapi.json").json()
    second = client.get("/openapi.json").json()
    assert json.dumps(first, sort_keys=True, ensure_ascii=False) == json.dumps(
        second, sort_keys=True, ensure_ascii=False
    )


def test_docs_page_renders(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


def test_bag_slot_out_forbids_unknown_fields():
    """契约收紧：多出来的字段直接判错，避免挂接方以为服务端认了它。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BagSlotOut.model_validate({"session_id": "s", "surprise": 1})


def test_documented_response_model_accepts_a_real_upload_body(client):
    """真实 201 响应能整个装回响应模型（required 清单没有过严或过松）。"""
    payload = build_frame_zip(frame_count=14)
    files = {"frames": ("rack.zip", payload, "application/zip")}
    response = client.post(TRACKS_URL, data=dict(EXAMPLE_FIELDS), files=files)
    assert response.status_code == 201, response.text

    model = TrackUploadResponse.model_validate(response.json())
    assert model.bags[0].bag_seq == "B01"
    assert len(model.bags[0].area_ratio_series) == model.bags[0].frame_count