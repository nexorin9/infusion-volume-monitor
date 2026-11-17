"""预测器契约测试套件：**同一套断言，两个实现都跑**。

覆盖 task 24 的四组断言：

1. **协议表面与输出结构**——``name``、满足 :class:`PerPointMaskPredictor`、``bool`` 掩膜、
   与原图同尺寸、``score ∈ [0,1]`` 且降序；「找不到目标」在纯色画面上返回**空列表**；
2. **退化提示**——画面外点、空提示、只有负点、零面积框：只裁剪不抛异常，且不返回整帧
   （返回整帧等于把面积比钉成 1.0，毫升数直接失真）；
3. **批量与逐条一致**、**两次调用逐位相同**、**不改动传入的帧**、**PNG 往返无损**；
4. **坐标系一致**——提示指哪块掩膜就落在哪块（IoU 下限 + 不串邻块 + 质心误差上限），
   这条专门抓「dtype 与 shape 都合规、但坐标系偏了几个像素」的实现。

**三个实现各自的位置**（矩阵在 :data:`CONTRACT_IMPLS` 里声明，加实现只改这一处）：

* ``geom_kernel`` —— 确定性几何核，**必须全绿**；
* ``weights`` —— 生产权重的真实构建路径（注册表 → 配置 → 权重 → 适配器）。本机没有权重与
  推理运行时，按要求**标 skip 并把真实原因写进跳过原因**，不拿替身冒充；
* ``contract_reference`` —— 测试层的协议替身（按背景色差取前景的独立实现），经**真实**
  ``register_weights_adapter`` / ``load_weights_predictor`` 入口构建，证明同一套断言在
  第二个**写法完全不同**的对象上照样跑得通。替身只在测试数据层，不是最终对接边界。

另有**反例对照组**：六个刻意写坏的实现，逐个断言契约能抓到它们——否则这套断言自己也可能
是「怎么都绿」的假契约。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from infusion_monitor.core import predictor_contract as contract
from infusion_monitor.core.config_map import (
    ModelConfigError,
    build_predictor,
    predictor_impl_names,
    resolve_model,
)
from infusion_monitor.core.geom_kernel import GeometricMaskPredictor
from infusion_monitor.core.predictor import (
    BatchCapablePredictor,
    MaskPrediction,
    Prompt,
    mask_centroid,
    mask_iou,
)
from infusion_monitor.core.predictor_contract import (
    CONTRACT_CHECKS,
    ContractCase,
    ContractReport,
    ContractScene,
    ContractViolation,
    blob_scene,
    contract_cases,
    decode_mask_png,
    encode_mask_png,
    run_predictor_contract,
)
from infusion_monitor.core.weights_predictor import load_weights_predictor, register_weights_adapter

#: 契约矩阵：实现名 → 怎么把它构造出来。加一个实现就在这里加一行，别处不许各造各的。
CONTRACT_IMPLS: tuple[str, ...] = ("geom_kernel", "weights", "contract_reference")


# ─────────────────────────────────────────────────────────────────────────────
# 协议替身：按背景色差取前景的独立实现（刻意与几何核写法不同）
# ─────────────────────────────────────────────────────────────────────────────
class ReferenceColorPredictor(BatchCapablePredictor):
    """最小但**真的按提示出掩膜**的实现。

    与几何核的差别是刻意的：几何核走 HSV 阈值带 + 形态学 + 连通域标签查表，这里走
    「画面四边中位数当背景基准 → 通道最大差值取前景 → 点所在连通域」。两段代码不共享，
    所以它们对同一批提示的独立一致才说明契约断言绑的是协议而不是某一份实现。
    """

    name = "contract_reference"

    def __init__(self, delta: int = 40, pick_radius: float = 40.0) -> None:
        self.delta = delta
        self.pick_radius = pick_radius

    def _foreground(self, image: np.ndarray) -> np.ndarray:
        border = np.concatenate(
            [image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]], axis=0
        ).astype(np.int16)
        background = np.median(border, axis=0)
        distance = np.abs(image.astype(np.int16) - background).max(axis=2)
        return distance >= self.delta

    def predict(self, image, prompt):
        foreground = self._foreground(image)
        window = prompt.clipped_box(image.shape[:2])
        if window is not None:
            masked = np.zeros_like(foreground)
            x0, y0, x1, y1 = window
            masked[y0:y1, x0:x1] = foreground[y0:y1, x0:x1]
            foreground = masked
        if not foreground.any():
            return []
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            return []

        chosen: int | None = None
        for point in prompt.positive_points:
            x, y = int(round(point.x)), int(round(point.y))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0] and labels[y, x] > 0:
                chosen = int(labels[y, x])
                break
        if chosen is None and prompt.positive_points:
            best = min(
                (
                    (
                        min(
                            float(np.hypot(centroids[label][0] - point.x, centroids[label][1] - point.y))
                            for point in prompt.positive_points
                        ),
                        int(label),
                    )
                    for label in range(1, count)
                ),
                default=None,
            )
            if best is not None and best[0] <= self.pick_radius:
                chosen = best[1]
        if chosen is None:
            chosen = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == chosen
        return [MaskPrediction(mask=mask, score=round(_shape_score(mask), 4))]


def _shape_score(mask: np.ndarray) -> float:
    """形状规整度：一个便宜的自洽定义（只要求在 0–1，跨实现不要求同值）。"""

    area = float(np.count_nonzero(mask))
    if area <= 0:
        return 0.0
    ys, xs = np.nonzero(mask)
    span = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    return float(min(1.0, 0.5 + 0.5 * area / max(1.0, span)))


def _make_weights_dir(tmp_path, backend: str = "torch"):
    """造一个配置 + 权重齐全的注册表根目录（权重内容不参与推理，只为过加载期校验）。

    ``backend`` 默认 ``torch``——这是院区侧真实配置里会写的值，也让 weights 档位的跳过原因
    读起来是真话（「配置指向 torch，但本机没有对应适配器」），而不是测试自己造的圈套。
    """

    root = tmp_path / "models"
    root.mkdir(exist_ok=True)
    (root / "weights_base.yaml").write_text(
        f"backend: {backend}\nmodel_type: infusion_bag\n", encoding="utf-8"
    )
    (root / "weights_base.pt").write_bytes(b"contract-test-payload" * 8)
    return root


# ─────────────────────────────────────────────────────────────────────────────
# 实现矩阵
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def geom_kernel():
    """确定性几何核：经业务层唯一入口 ``build_predictor()`` 取，主路径不认实现。"""

    predictor = build_predictor()
    assert isinstance(predictor, GeometricMaskPredictor)
    return predictor


def _reference_factory(config, checkpoint_path, spec) -> ReferenceColorPredictor:
    return ReferenceColorPredictor(delta=int(config.get("delta", 40)))


def _build_reference(tmp_path) -> ReferenceColorPredictor:
    """经真实 ``register_weights_adapter`` → ``load_weights_predictor`` 入口构建协议替身。"""

    from infusion_monitor.core import weights_predictor as wp

    register_weights_adapter("contract_reference", _reference_factory)
    try:
        spec = resolve_model(
            "weights_base", registry_root=_make_weights_dir(tmp_path, "contract_reference")
        )
        predictor = load_weights_predictor(spec)
    finally:
        wp._ADAPTERS.pop("contract_reference", None)
    assert predictor.name == "contract_reference"
    return predictor


def _build_weights(tmp_path):
    """生产权重的真实构建路径；本机没有权重与推理运行时时返回 ``None`` + 真实原因。

    不用替身顶替——顶替了就等于宣称权重路径验过了。契约矩阵保留该档位，接上权重适配器后
    同一套断言会自动跑起来。
    """

    root = _make_weights_dir(tmp_path)
    try:
        spec = resolve_model("weights_base", registry_root=root)
        return load_weights_predictor(spec), None
    except ModelConfigError as exc:
        return None, f"{type(exc).__name__}: {exc}"


@pytest.fixture
def weights_impl(tmp_path):
    """生产权重档位；本机无实现时按要求 skip，并把真实原因原样写进跳过原因。"""

    predictor, reason = _build_weights(tmp_path)
    if predictor is None:
        pytest.skip(f"本机未接入生产权重实现（{reason}）；契约矩阵保留该档位")
    return predictor


@pytest.fixture
def contract_reference(tmp_path):
    """协议替身：写法与几何核完全不同，经真实加载入口构建。"""

    return _build_reference(tmp_path)


@pytest.fixture(params=CONTRACT_IMPLS)
def any_impl(request, tmp_path):
    """三个档位逐一跑同一套断言。

    **按需构造**：weights 档位缺实现时只 skip 它自己，不牵连另外两个档位（早先把三个夹具
    一起请求，结果一个档位 skip 会把整组参数化用例全部 skip 掉）。
    """

    if request.param == "geom_kernel":
        return "geom_kernel", build_predictor()
    if request.param == "contract_reference":
        return "contract_reference", _build_reference(tmp_path)
    predictor, reason = _build_weights(tmp_path)
    if predictor is None:
        pytest.skip(f"本机未接入生产权重实现（{reason}）；契约矩阵保留该档位")
    return "weights", predictor


# ─────────────────────────────────────────────────────────────────────────────
# 1. 两个实现跑同一套契约
# ─────────────────────────────────────────────────────────────────────────────
def test_every_registered_impl_is_covered_by_the_contract_matrix():
    """注册表里新增了实现就必须进契约矩阵——漏掉的那一个会安静地不受协议约束。"""

    registered = set(predictor_impl_names())
    covered = {impl for impl in CONTRACT_IMPLS if impl in registered}
    assert registered <= covered, f"未进契约矩阵的实现：{sorted(registered - covered)}"


def test_contract_case_set_is_shared_not_per_impl(any_impl):
    """用例集只有一份：三个实现拿到的是同一个 ``contract_cases()``。"""

    impl, predictor = any_impl
    cases = contract_cases()
    assert len({case.key for case in cases}) == len(cases), "用例 key 必须唯一，否则报告对不上"
    assert impl in CONTRACT_IMPLS
    assert run_predictor_contract(predictor, cases=cases).ok


def test_geometric_kernel_passes_the_whole_contract(geom_kernel):
    """几何核必须全绿（这是硬要求，不是「跑得动就算」）。"""

    report = run_predictor_contract(geom_kernel)
    assert report.ok, contract.summarise_reports([report])
    assert report.passed == CONTRACT_CHECKS


@pytest.mark.parametrize("check", CONTRACT_CHECKS)
def test_each_contract_check_passes(any_impl, check):
    """逐项跑，便于失败时直接定位是哪一条被破坏。"""

    impl, predictor = any_impl
    report = run_predictor_contract(predictor)
    broken = [item for item in report.failures if item.name == check]
    assert not broken, f"{impl} 在 {check} 上被契约拒绝：{broken[0].detail}"
    assert check in report.passed


def test_report_is_stable_and_printable(any_impl):
    """报告可打印、顺序固定、失败项带原因——病区侧自检直接用它。"""

    impl, predictor = any_impl
    report = run_predictor_contract(predictor)
    assert isinstance(report, ContractReport)
    assert report.impl == predictor.name, "报告里的实现名就是 predicate 自己的 name"
    assert impl in CONTRACT_IMPLS
    text = contract.summarise_reports([report])
    for name in CONTRACT_CHECKS:
        assert name in text
    payload = report.as_dict()
    assert payload["ok"] is True and payload["failed"] == []
    assert set(payload) == {"impl", "ok", "passed", "failed"}


# ─────────────────────────────────────────────────────────────────────────────
# 2. 具体契约条目的定向用例（几何核跑，防止断言写成空壳）
# ─────────────────────────────────────────────────────────────────────────────
def test_mask_is_bool_and_same_size_as_the_frame(geom_kernel):
    """掩膜 dtype 与 shape 是协议里最容易悄悄坏掉的两项，单独钉一条。"""

    scene = blob_scene()
    prediction = geom_kernel.predict(scene.image, Prompt.from_point(70.0, 124.0))[0]
    assert prediction.mask.dtype == np.bool_
    assert prediction.mask.shape == scene.shape
    assert 0.0 <= prediction.score <= 1.0


def test_point_prompt_lands_on_the_right_blob_and_not_its_neighbour(geom_kernel):
    """坐标系一致：点在哪块，掩膜就落哪块，且不把邻块一起框进来。"""

    scene = blob_scene()
    mask = geom_kernel.predict(scene.image, Prompt.from_point(70.0, 124.0))[0].mask
    assert mask_iou(mask, scene.truth("a")) >= 0.6
    assert not np.any(mask & scene.truth("b"))
    centre = mask_centroid(mask)
    assert centre is not None
    assert abs(centre[0] - 70.0) <= 10.0 and abs(centre[1] - 124.0) <= 10.0


def test_empty_target_yields_empty_list_not_a_zero_mask(geom_kernel):
    """纯色画面上必须返回 ``[]``。返回全零图会被上层当成 0% 余量，凭空多出一条换液提醒。"""

    plain = contract.plain_scene()
    for prompt in (Prompt.from_point(60.0, 60.0), Prompt(box=(20.0, 20.0, 200.0, 200.0)), Prompt()):
        assert geom_kernel.predict(plain.image, prompt) == []


def test_degenerate_prompts_are_clipped_not_rejected(geom_kernel):
    """画面外的点与零面积框不抛异常——护士随手点偏几个像素是常事。"""

    scene = blob_scene()
    height, width = scene.shape
    for prompt in (
        Prompt.from_point(-40.0, -40.0),
        Prompt.from_point(float(width) + 90.0, float(height) + 90.0),
        Prompt(),
        Prompt.from_point(70.0, 124.0, 0),
        Prompt(box=(10.0, 10.0, 10.0, 60.0)),
        Prompt(box=(-200.0, -200.0, -100.0, -100.0)),
    ):
        result = geom_kernel.predict(scene.image, prompt)
        assert isinstance(result, list)
        for prediction in result:
            assert prediction.mask.shape == scene.shape
            assert prediction.mask.sum() <= 0.95 * height * width


def test_batch_matches_single_bit_for_bit(geom_kernel):
    """批量与逐条逐像素一致——枚举管线在一帧上问几十个点，两条路不能给出两套数。"""

    scene = blob_scene()
    cases = contract_cases(scene)
    batched = geom_kernel.predict_batch(scene.image, [case.prompt for case in cases])
    assert len(batched) == len(cases)
    for case, result in zip(cases, batched):
        single = geom_kernel.predict(scene.image, case.prompt)
        assert len(result) == len(single)
        for a, b in zip(result, single):
            assert np.array_equal(a.mask, b.mask) and a.score == b.score
    assert geom_kernel.predict_batch(scene.image, []) == []


def test_predict_does_not_mutate_the_input_frame(geom_kernel):
    """实现必须只读传入的帧：就地写回会让第二袋的起锚建立在被改脏的图上。"""

    scene = blob_scene()
    before = scene.image.copy()
    for case in contract_cases(scene):
        if case.scene.blobs:
            geom_kernel.predict(scene.image, case.prompt)
    assert np.array_equal(scene.image, before)


def test_two_identical_calls_give_identical_masks(geom_kernel):
    """幂等：重传同一段短片必须得到同一组掩膜，否则面积比曲线每次重跑都在动。"""

    scene = blob_scene()
    prompt = Prompt.from_point(70.0, 124.0)
    first, second = geom_kernel.predict(scene.image, prompt), geom_kernel.predict(scene.image, prompt)
    assert np.array_equal(first[0].mask, second[0].mask)
    assert first[0].score == second[0].score


# ─────────────────────────────────────────────────────────────────────────────
# 3. 掩膜 PNG 往返
# ─────────────────────────────────────────────────────────────────────────────
def test_mask_png_roundtrip_is_lossless():
    """0/255 单通道 PNG 往返逐位相同，且二次编码字节一致。"""

    mask = blob_scene().truth("a")
    payload = encode_mask_png(mask)
    decoded = decode_mask_png(payload)
    assert decoded.dtype == np.bool_
    assert decoded.shape == mask.shape
    assert np.array_equal(decoded, mask)
    assert encode_mask_png(decoded) == payload


def test_mask_png_roundtrip_handles_non_contiguous_views():
    """切片出来的非连续视图照样能编——落盘代码常直接拿子矩阵写文件。"""

    mask = blob_scene().truth("b")[::2, ::3]
    assert not mask.flags["C_CONTIGUOUS"]
    assert np.array_equal(decode_mask_png(encode_mask_png(mask)), mask)


def test_mask_png_rejects_wrong_dtype_and_shape():
    """0/1 的 uint8 不是 bool 掩膜：按位与会变成相乘而不是布尔与，在这里就说清楚。"""

    with pytest.raises(ContractViolation, match="bool"):
        encode_mask_png(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ContractViolation, match="二维"):
        encode_mask_png(np.zeros((2, 4, 4), dtype=bool))


def test_mask_png_rejects_truncated_payload():
    """截断的载荷要报出「解码失败」，而不是返回一个半张图。"""

    with pytest.raises(ContractViolation, match="为空"):
        decode_mask_png(b"")
    truncated = encode_mask_png(blob_scene().truth("a"))[:20]
    with pytest.raises(ContractViolation, match="解码失败"):
        decode_mask_png(truncated)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 反例对照组：契约必须有牙
# ─────────────────────────────────────────────────────────────────────────────
class _Broken(BatchCapablePredictor):
    """在参考实现上叠加**一种**破坏方式，用来验契约本身有牙。

    破坏都加在 ``predict`` 上（契约的主路径），只有 ``batch_divergent`` 故意加在
    ``predict_batch`` 上——它模拟的正是「批量那条路另起了一套参数」这种最常见的分叉。
    """

    name = "broken"

    def __init__(self, mode: str) -> None:
        self.inner = ReferenceColorPredictor()
        self.mode = mode

    def predict(self, image, prompt):
        if self.mode == "raise_out_of_frame" and prompt.points and (
            prompt.points[0].x < 0 or prompt.points[0].x > image.shape[1]
        ):
            raise ValueError("画面外的点应该被裁剪，不该报错")
        if self.mode == "whole_frame":
            height, width = image.shape[:2]
            return [MaskPrediction(mask=np.ones((height, width), dtype=bool), score=1.0)]
        results = self.inner.predict(image, prompt)
        if self.mode == "mutate":
            # 就地写回调用方的帧。两个讲究：①取 190/200 交替而不是写死一个值——写死的话
            # 第一次调用之后帧就已经是那个值，后续用例再复制一份对比就看不出差别了；
            # ②用接近背景的灰而不是 0，否则会凭空多出一个 1 px 的「连通域」，把空目标
            # 那条用例也搅浑，失败就报在错误的检查上了。
            image[0, 0, 0] = 200 if image[0, 0, 0] == 190 else 190
        return [self._corrupt(item, image.shape) for item in results]

    def _corrupt(self, item: MaskPrediction, shape) -> MaskPrediction:
        if self.mode == "float_mask":
            return MaskPrediction(mask=item.mask.astype(np.uint8), score=item.score)
        if self.mode == "half_res":
            # 内部降采样再「放大」——协议第 2 条明令禁止：坐标一旦平移，点/框就指错地方
            big = np.repeat(np.repeat(item.mask, 2, axis=0), 2, axis=1)
            return MaskPrediction(mask=big[: shape[0] // 2, : shape[1] // 2], score=item.score)
        if self.mode == "shifted":
            shifted = np.zeros_like(item.mask)
            shifted[:, 40:] = item.mask[:, :-40]
            return MaskPrediction(mask=shifted, score=item.score)
        return item

    def predict_batch(self, image, prompts):
        if self.mode == "batch_divergent":
            # 批量这条路换一套阈值：三块区域与背景的差分别是 170 / 165 / 160，门限 163
            # 正好把最小那块吃掉，于是批量与逐条给出两套数。
            return [ReferenceColorPredictor(delta=163).predict(image, prompt) for prompt in prompts]
        return [self.predict(image, prompt) for prompt in prompts]


@pytest.mark.parametrize(
    "mode,expected_check",
    [
        ("float_mask", "structure"),
        ("half_res", "structure"),
        ("shifted", "coordinate_alignment"),
        ("whole_frame", "empty_target"),
        ("batch_divergent", "batch_consistency"),
        ("mutate", "determinism_and_purity"),
        ("raise_out_of_frame", "degenerate_prompts"),
    ],
)
def test_contract_catches_a_deliberately_broken_predictor(mode, expected_check):
    """每个破坏方式都必须被抓到，且报在**对应**的那条检查上。"""

    report = run_predictor_contract(_Broken(mode))
    failed = [item.name for item in report.failures]
    assert not report.ok, f"破坏方式 {mode} 没被契约抓到——断言可能写成了空壳"
    assert expected_check in failed, f"破坏方式 {mode} 应当被 {expected_check} 抓到，实际失败项：{failed}"
    for item in report.failures:
        assert item.detail, "失败项必须带可照做的中文原因，不能只说「不通过」"


def test_whole_frame_return_is_caught_twice_on_purpose():
    """「没找到就交整帧」同时触发结构护栏与空目标条款——这是有意的双重防线。"""

    report = run_predictor_contract(_Broken("whole_frame"))
    assert {"structure", "empty_target"} <= {item.name for item in report.failures}
    structure_detail = next(item.detail for item in report.failures if item.name == "structure")
    assert "整张图" in structure_detail


def test_missing_protocol_method_is_caught_by_the_surface_check():
    """只实现 ``predict`` 没实现 ``predict_batch`` 的对象不许进主路径。"""

    class OnlyPredict:
        name = "half_implemented"

        def predict(self, image, prompt):
            return []

    report = run_predictor_contract(OnlyPredict())
    assert "surface" in {item.name for item in report.failures}


def test_wrong_image_shape_is_reported_instead_of_crashing_the_run():
    """帧不是 HxWx3 uint8 时要报「通道序转换漏了」，而不是抛一个看不懂的 OpenCV 错。"""

    bad_scene = ContractScene(image=np.zeros((4, 4), dtype=np.uint8), blobs=())
    case = ContractCase("bad_image", Prompt.from_point(1.0, 1.0), bad_scene)
    with pytest.raises(ContractViolation, match="HxWx3"):
        contract.check_prediction_structure(ReferenceColorPredictor(), case)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 契约用例集本身的性质
# ─────────────────────────────────────────────────────────────────────────────
def test_contract_cases_cover_points_boxes_and_degenerate_prompts():
    """用例集必须同时覆盖「正常提示」「退化提示」和「画面里没有目标」三类。"""

    cases = contract_cases()
    keys = {case.key for case in cases}
    assert {"point_in_a", "box_around_a", "positive_plus_negative"} <= keys
    assert {"bare_prompt", "point_outside_frame", "box_outside_frame"} <= keys
    assert {"plain_point", "plain_bare_prompt"} <= keys
    assert any(case.expect_target is not None for case in cases)
    assert any(case.expect_target is None for case in cases)


def test_contract_cases_carry_tolerances_not_equalities():
    """断言给的是带（IoU 下限 + 质心容差），不是逐像素相等——两个实现本就该有边界差异。"""

    cases = {case.key: case for case in contract_cases()}
    assert 0 < cases["point_in_a"].min_iou < 1
    assert cases["point_in_small_c"].centroid_tolerance < cases["point_in_a"].centroid_tolerance
    assert cases["positive_plus_negative"].forbid_target == "b"