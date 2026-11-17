"""上传前的帧质量预检：给一个 0–1 的质量分，命中问题就配一句能照做的补拍建议。

**与袋级失败模式分类的分工**（两层都在主路径上跑，但问的问题不同）：

* **预检（本模块）** 跑在分割**之前**，只看画面本身的统计量：清晰度、亮度分布、短边
  分辨率、液面线倾角。它不知道有几只袋子、也不认识任何一只袋子，因此答不了「哪一只袋子
  没拍全」，但能答「这一整段值不值得信」。
* **袋级分类**（:mod:`..core.failure_modes`，task 20）跑在分割**之后**，带着每只袋子的
  掩膜给结论，能答「B02 出框了」。

**三层纪律**：

1. **只降级、不拒绝**：拍坏了照样 HTTP 201、曲线照给、毫升数照给。拒绝权只留给硬校验
   （帧数 / 短边 320px 下限 / 帧序 / 袋位上限，见 :mod:`.validate`）——那些是不合格就
   不能算数的输入。
2. **共用文案层**：每条建议的中文措辞来自 ``core.failure_modes.render_reshoot_text``
   （那张表也是袋级分类的真源），两层门限又取自同一组配置项，护士端因此只会看到一套
   措辞、不会看到「预检说拍好了、袋级说拍坏了」。
3. **恒定算力**：只均匀采样 ``sample_frames`` 帧（默认 5），与短片长度、袋数都不成正比。
   1200 帧的架和 24 帧的架预检开销相同。

**质量分怎么来**：五个子分（清晰度 / 亮度 / 反光 / 分辨率 / 角度）各自归一到 [0, 1]，
按权重求加权平均；某一子项量不出来（画面里找不到可量的液面线 → 角度判不出）就把它的
权重摊回给其余子项，而不是给它 0 分——**判不出不等于拍得差**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ..core.failure_modes import (
    SEVERITY_WARNING,
    render_reshoot_text,
    sample_indices,
    surface_tilt,
)

__all__ = [
    "PRECHECK_CODES",
    "PrecheckParams",
    "FrameQuality",
    "QualityFlag",
    "SequenceQuality",
    "measure_frame_quality",
    "precheck_frames",
    "precheck_sample_indices",
]


#: 预检能命中的五类问题。码与袋级分类同名同义（``low_resolution`` 为预检独有），
#: 文案与门限也都取自同一处，护士端不需要分辨「这句话是哪个阶段说的」。
PRECHECK_CODES: tuple[str, ...] = (
    "motion_blur",
    "glare_overexposed",
    "bad_angle",
    "low_light",
    "low_resolution",
)

#: 子分权重。加权和（预检不引入 pydantic 依赖，也不给权重做配置项：五项量纲已经
#: 各自归一，权重是产品判断不是环境差异）。倾角排在清晰度之后是故意的——暗光帧的
#: 清晰度量不可信，那种帧已被剔出模糊判定，剩下的倾角才可信。
DEFAULT_WEIGHTS: dict[str, float] = {
    "sharpness": 0.30,
    "low_light": 0.20,
    "glare": 0.15,
    "resolution": 0.15,
    "tilt": 0.20,
}


@dataclass(frozen=True)
class PrecheckParams:
    """预检门限。取值与袋级分类共用同一组配置项，理由见各字段注释。"""

    #: 均匀采样多少帧（与序列长度无关）。
    sample_frames: int = 5
    #: 至少多少比例的采样帧命中才算「这一段有问题」（抗单帧偶发）。
    min_hit_ratio: float = 0.34
    # ── 清晰度 ────────────────────────────────────────────────────────
    #: 拉普拉斯方差下限。实测：拖影 ≈87、轻微手抖 ≈242，正常 ≥350，取中间值 150。
    sharpness_min: float = 150.0
    # ── 亮度分布 ──────────────────────────────────────────────────────
    #: 全画面平均灰度门限。实测：夜间暗光 ≈42–44，正常拍法 ≥80。
    low_light_mean_luma: float = 46.0
    #: 暗像素判定的灰度门限（只进证据，不单独触发）。
    dark_luma: int = 34
    # ── 反光 ──────────────────────────────────────────────────────────
    #: 近白判定：三通道最小值 ≥ 该值即算「白到没有颜色」。
    glare_channel_min: int = 242
    #: 近白像素占全画面的比例门限。
    glare_ratio_threshold: float = 0.0015
    # ── 分辨率 ────────────────────────────────────────────────────────
    #: 短边像素门限。硬下限是 320（低于此直接 422），这里是**质量下限**：
    #: 短边低于本值虽然还能算，但液面边界已经贴着采样极限，余量读数不可信。
    min_short_side: int = 480
    # ── 角度 ──────────────────────────────────────────────────────────
    #: 液面线倾角门限（度）。实测：歪 20° 的拍法 5–24°，正常拍法 ≤2.1°。
    tilt_angle_deg: float = 8.0
    #: 参与倾角拟合的最少列数（量不出来就返回 None，不猜角度）。
    tilt_min_columns: int = 16
    #: 相邻两列液面行号跳变超过此值即视为「换了一只袋子」，另起一段拟合。
    #: 歪 20° 时每列只差约 0.4 px，两只袋子之间则是几十上百 px，取 12 px 兼顾两头。
    tilt_max_jump_px: int = 12
    #: 一列里至少这么多高饱和像素才算「这一列有液体」——背景墙面上零星的高饱和列
    #: 会把液面线拟合带偏，先按列高剔掉。
    tilt_min_run_px: int = 24
    #: 液体的饱和度下限（OpenCV HSV 的 S）：与几何核阈值带的 ``saturation_min`` 同值，
    #: 保证「预检量的液面」与「主路径量的液体」是同一批像素。
    saturation_min: int = 96
    # ── 总分门禁 ──────────────────────────────────────────────────────
    #: 质量分低于此值即标 degraded（即便没有任何一类命中：五项各差一点也足以让
    #: 这一段的余量读数打问号）。
    degraded_score: float = 0.70
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def as_dict(self) -> dict[str, Any]:
        """配置摘要（进运行摘要与响应，便于信息科核对现场为什么被标 degraded）。"""
        return {
            "sample_frames": self.sample_frames,
            "min_hit_ratio": self.min_hit_ratio,
            "sharpness_min": self.sharpness_min,
            "low_light_mean_luma": self.low_light_mean_luma,
            "glare_ratio_threshold": self.glare_ratio_threshold,
            "min_short_side": self.min_short_side,
            "tilt_angle_deg": self.tilt_angle_deg,
            "degraded_score": self.degraded_score,
        }


@dataclass(frozen=True)
class FrameQuality:
    """一帧的观测量。全部是「量到了什么」，不含业务判断。"""

    frame_index: int
    width: int
    height: int
    short_side: int
    luma_mean: float
    dark_ratio: float
    glare_ratio: float
    sharpness: float
    #: 液面线倾角（度）；画面里找不到可量的液面线时为 None（判不出就说判不出）。
    tilt_deg: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "short_side": self.short_side,
            "luma_mean": round(self.luma_mean, 2),
            "dark_ratio": round(self.dark_ratio, 4),
            "glare_ratio": round(self.glare_ratio, 6),
            "sharpness": round(self.sharpness, 2),
            "tilt_deg": None if self.tilt_deg is None else round(self.tilt_deg, 2),
        }


@dataclass(frozen=True)
class QualityFlag:
    """一条预检问题：机器码 + 中文说法 + 补拍建议 + 支撑证据。

    形状与袋级分类的 ``FailureFlag`` 一致，因此响应模型（``QualityFlagOut``）两边共用。
    """

    code: str
    severity: str
    label_zh: str
    detail_zh: str
    suggestion_zh: str
    hit_frames: int
    sampled_frames: int
    evidence: dict[str, Any]

    @property
    def hit_ratio(self) -> float:
        return self.hit_frames / self.sampled_frames if self.sampled_frames else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "label_zh": self.label_zh,
            "detail_zh": self.detail_zh,
            "suggestion_zh": self.suggestion_zh,
            "hit_frames": self.hit_frames,
            "sampled_frames": self.sampled_frames,
            "hit_ratio": round(self.hit_ratio, 4),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class SequenceQuality:
    """整段短片的预检结论（上传级，一个 response 对一份）。"""

    score: float
    degraded: bool
    sampled_frames: int
    flags: tuple[QualityFlag, ...] = ()
    #: 五个子分（0–1）。量不出来的子项不出现在这里——**判不出 ≠ 拍得差**。
    sub_scores: dict[str, float] = field(default_factory=dict)
    frames: tuple[FrameQuality, ...] = ()
    #: 最该先补的那一条（护士端提示条只顶这一条）。
    primary: QualityFlag | None = None

    @property
    def state(self) -> str:
        return "degraded" if self.degraded else "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "quality_score": round(self.score, 4),
            "degraded": self.degraded,
            "sampled_frames": self.sampled_frames,
            "sub_scores": {key: round(value, 4) for key, value in self.sub_scores.items()},
            "primary": None if self.primary is None else self.primary.to_dict(),
            "flags": [flag.to_dict() for flag in self.flags],
            "frames": [item.to_dict() for item in self.frames],
        }


# ── 测量 ────────────────────────────────────────────────────────────────


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _surface_rows(saturated: np.ndarray, min_run_px: int) -> np.ndarray:
    """每列液面的行号；该列高饱和像素不足 ``min_run_px`` 时为 -1（这一列不算数）。

    袋体里液体的饱和度明显高于透明袋皮与病区背景，因此「每列最高的高饱和像素」就是
    这一列的液面。背景墙上零星的高饱和像素只有零星几个，会先被列高门限剔掉。
    """
    rows = np.full(saturated.shape[1], -1, dtype=np.int64)
    tall = saturated.sum(axis=0) >= min_run_px
    if tall.any():
        rows[tall] = saturated.argmax(axis=0)[tall]
    return rows


def _split_columns(rows: np.ndarray, max_jump: float) -> list[tuple[int, int]]:
    """把有效列按「液面行号连续」切成若干段，返回每段的 ``(起列下标, 止列下标)``。

    **一个输液架上最多挂六只袋子，液面高度各不相同**。把六条不同高度的液面硬拟合成
    一条直线，量到的不是「手机歪了多少度」，而是「这六只袋子挂得高矮差多少」——实测
    干净拍法能读出 19–22° 的假倾角。因此按列分组：相邻两列液面行号跳变超过
    ``max_jump`` 就是换了一只袋子，另起一段。
    """
    segments: list[tuple[int, int]] = []
    start: int | None = None
    previous = -1
    for index, row in enumerate(rows):
        if row < 0:
            if start is not None:
                segments.append((start, index))
                start = None
            previous = -1
            continue
        if start is None:
            start = index
        elif abs(float(row) - float(previous)) > max_jump:
            segments.append((start, index))
            start = index
        previous = int(row)
    if start is not None:
        segments.append((start, len(rows)))
    return segments


def _liquid_surface_tilt(image: np.ndarray, params: PrecheckParams) -> float | None:
    """液面线倾角（度）——不依赖任何掩膜、任何预测器。

    做法：按饱和度带取出每列液面的行号 → 按液面高度切成若干段（每段对应一只袋子）→
    每段各做一次最小二乘拟合（复用 :func:`..core.failure_modes.surface_tilt`）→ 取各段
    倾角的**中位数**。手机歪着拍时同一架上所有袋子歪向一致，中位数因此稳；某一只袋子
    恰好被手挡住或强反光时只会毁掉自己那一段，其余段照旧可用。

    没有一段够宽（``tilt_min_columns`` 列）时返回 ``None``——判不出倾角就说判不出，
    不猜一个角度出来。这一条与袋级分类一致：宁可少一个子项，也不给一个编出来的角度。
    """
    hsv = cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_RGB2HSV)
    saturated = hsv[..., 1] >= params.saturation_min
    rows = _surface_rows(saturated, params.tilt_min_run_px)
    if not (rows >= 0).any():
        return None

    height = saturated.shape[0]
    angles: list[float] = []
    for start, end in _split_columns(rows, params.tilt_max_jump_px):
        columns = rows[start:end]
        valid = columns >= 0
        if int(np.count_nonzero(valid)) < params.tilt_min_columns:
            continue
        block = np.zeros((height, columns.size), dtype=bool)
        picked = columns[valid]
        block[picked, np.flatnonzero(valid)] = True
        angle = surface_tilt(block, params.tilt_min_columns)
        if angle is not None:
            angles.append(angle)
    if not angles:
        return None
    return float(np.median(angles))


def measure_frame_quality(
    image: np.ndarray, frame_index: int = 0, params: PrecheckParams | None = None
) -> FrameQuality:
    """量一帧的拍摄质量观测量。纯统计，不做任何业务判定。"""
    params = params or PrecheckParams()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"期望 HxWx3 的 RGB 帧，实际形状 {image.shape}")

    height, width = image.shape[0], image.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    total = float(gray.size)
    channel_min = image.min(axis=2)
    return FrameQuality(
        frame_index=frame_index,
        width=int(width),
        height=int(height),
        short_side=min(int(width), int(height)),
        luma_mean=float(gray.mean()),
        dark_ratio=float(np.count_nonzero(gray <= params.dark_luma)) / total,
        glare_ratio=float(np.count_nonzero(channel_min >= params.glare_channel_min)) / total,
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        tilt_deg=_liquid_surface_tilt(image, params),
    )


def precheck_sample_indices(frame_count: int, params: PrecheckParams | None = None) -> tuple[int, ...]:
    """预检要采样哪几帧。与袋级分类共用同一个采样函数，两层看的是同一批画面。"""
    params = params or PrecheckParams()
    return sample_indices(frame_count, max(1, params.sample_frames))


# ── 判定 ────────────────────────────────────────────────────────────────


def _hit_blur(measure: FrameQuality, params: PrecheckParams) -> bool:
    return measure.sharpness < params.sharpness_min


def _hit_glare(measure: FrameQuality, params: PrecheckParams) -> bool:
    return measure.glare_ratio >= params.glare_ratio_threshold


def _hit_low_light(measure: FrameQuality, params: PrecheckParams) -> bool:
    return measure.luma_mean < params.low_light_mean_luma


def _hit_tilt(measure: FrameQuality, params: PrecheckParams) -> bool:
    return measure.tilt_deg is not None and measure.tilt_deg > params.tilt_angle_deg


def _hit_resolution(measure: FrameQuality, params: PrecheckParams) -> bool:
    return measure.short_side < params.min_short_side


#: 每类的命中判据与证据取值。抽成表而不是散在 if 里：新增一类只改这一处 + 文案表。
_JUDGES: dict[str, tuple[Any, str, str]] = {
    "motion_blur": (_hit_blur, "sharpness", "sharpness_min"),
    "glare_overexposed": (_hit_glare, "glare_ratio", "glare_ratio_threshold"),
    "low_light": (_hit_low_light, "luma_mean", "low_light_mean_luma"),
    "bad_angle": (_hit_tilt, "tilt_deg", "tilt_angle_deg"),
    "low_resolution": (_hit_resolution, "short_side", "min_short_side"),
}


def _sub_scores(measure: FrameQuality, params: PrecheckParams) -> dict[str, float | None]:
    """一帧的五个子分，各自归一到 [0, 1]；量不出来的返回 None。"""
    return {
        "sharpness": _clip01(measure.sharpness / params.sharpness_min),
        "low_light": _clip01(measure.luma_mean / params.low_light_mean_luma),
        "glare": _clip01(1.0 - measure.glare_ratio / params.glare_ratio_threshold),
        "resolution": _clip01(measure.short_side / params.min_short_side),
        "tilt": (
            None
            if measure.tilt_deg is None
            else _clip01(1.0 - measure.tilt_deg / params.tilt_angle_deg)
        ),
    }


def _score_of(measures: Sequence[FrameQuality], params: PrecheckParams) -> tuple[float, dict[str, float]]:
    """整段质量分 = 五个子分的加权平均（量不出来的子项权重摊回其余子项）。

    逐帧先算子分再跨帧平均：某一帧恰好糊在采样点上，不该把整段的分拖下来；而某一帧
    糊在**所有**采样帧上（手持抖了一整段），子分会一路低于门限，被 ``motion_blur`` 命中。
    """
    per_frame = [_sub_scores(measure, params) for measure in measures]
    totals: dict[str, float] = {}
    for key in DEFAULT_WEIGHTS:
        values = [scores[key] for scores in per_frame if scores[key] is not None]
        if values:
            totals[key] = sum(values) / len(values)

    weight_sum = sum(params.weights.get(key, 0.0) for key in totals)
    if weight_sum <= 0.0:
        return 0.0, totals
    score = sum(totals[key] * params.weights.get(key, 0.0) for key in totals) / weight_sum
    return _clip01(score), totals


def _evidence(
    code: str, matched: Sequence[FrameQuality], params: PrecheckParams
) -> dict[str, Any]:
    """证据：命中帧的实测值 + 对应门限。信息科据此判断是「拍坏了」还是「门限太严」。

    门限一律放在 ``threshold`` 键下：文案模板里的占位符是 ``{threshold}``（共用文案层
    定的），换成参数名会让护士端收到一串没填上的 ``{threshold}``。
    """
    _judge, field_name, threshold_key = _JUDGES[code]
    values = [getattr(measure, field_name) for measure in matched]
    values = [value for value in values if value is not None]
    evidence: dict[str, Any] = {"threshold": round(float(getattr(params, threshold_key)), 4)}
    if values:
        # 亮度看**最暗**的那帧（离门限最近），其余看最严重的那帧。
        worst = min(values) if code == "low_light" else max(values)
        evidence[field_name] = round(float(worst), 4)
    if code == "glare_overexposed":
        evidence["ratio"] = f"{max(m.glare_ratio for m in matched) * 100:.2f}%"
    if code == "low_light":
        evidence["dark_ratio"] = round(max(m.dark_ratio for m in matched), 4)
    return evidence


def precheck_frames(
    frames: Sequence[np.ndarray],
    params: PrecheckParams | None = None,
    *,
    frame_indices: Sequence[int] | None = None,
) -> SequenceQuality:
    """跑一遍上传前预检：采样帧 → 逐帧量 → 归成五类问题 → 汇总成一个质量分。

    ``frames`` 应当是**已经采样过**的帧（调用方按 :func:`precheck_sample_indices` 取，
    与袋级分类共用同一次解码）；整段直接传进来时按同样规则自己采样。
    ``frame_indices`` 是这些帧在原短片里的下标，用于响应里回指「第几帧」。

    命中任一类即标 ``degraded``；即便五类都没命中，质量分低于 ``degraded_score``
    也照标——「每项都差一点」同样让余量读数打问号。
    """
    params = params or PrecheckParams()
    if not frames:
        return SequenceQuality(score=0.0, degraded=False, sampled_frames=0)

    if frame_indices is None:
        indices = precheck_sample_indices(len(frames), params)
        measured = [
            measure_frame_quality(frames[index], index, params) for index in indices
        ]
    else:
        measured = [
            measure_frame_quality(image, index, params)
            for image, index in zip(frames, frame_indices)
        ]

    sampled = len(measured)
    hits: dict[str, list[FrameQuality]] = {code: [] for code in PRECHECK_CODES}
    for measure in measured:
        for code, (judge, _field, _threshold) in _JUDGES.items():
            if judge(measure, params):
                hits[code].append(measure)

    # 暗光帧判不出糊不糊：清晰度跟着对比度一起掉，光线不足与手抖在这一个量上分不开。
    # 与袋级分类同一条规则——否则护士收到的第一句建议会是「把手机靠住」，而她真正
    # 要做的第一件事是**把灯打开**。
    dark_frames = {id(measure) for measure in hits.get("low_light", ())}
    hits["motion_blur"] = [
        measure for measure in hits["motion_blur"] if id(measure) not in dark_frames
    ]

    flags: list[QualityFlag] = []
    for code in PRECHECK_CODES:
        matched = hits[code]
        if not matched:
            continue
        evidence = _evidence(code, matched, params)
        text = render_reshoot_text(code, len(matched), sampled, evidence)
        if text is None:  # pragma: no cover - PRECHECK_CODES 与文案表一一对应
            continue
        if len(matched) / sampled < params.min_hit_ratio:
            # 少数几帧偶发不算「这一段拍坏了」——不足以让护士重拍整段。
            continue
        flags.append(
            QualityFlag(
                code=code,
                severity=text["severity"],
                label_zh=text["label_zh"],
                detail_zh=text["detail_zh"],
                suggestion_zh=text["suggestion_zh"],
                hit_frames=len(matched),
                sampled_frames=sampled,
                evidence=evidence,
            )
        )

    score, sub_scores = _score_of(measured, params)
    degraded = bool(flags) or score < params.degraded_score
    primary = next((flag for flag in flags if flag.severity == SEVERITY_WARNING), None)
    return SequenceQuality(
        score=score,
        degraded=degraded,
        sampled_frames=sampled,
        flags=tuple(flags),
        sub_scores=sub_scores,
        frames=tuple(measured),
        primary=primary,
    )