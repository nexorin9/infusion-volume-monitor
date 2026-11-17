"""环境变量配置面：7 组凭据位 + 非凭据运行参数。

纪律（与 spec.md「正式对接层」一致）：

* **凭据一律读环境变量**，代码与配置文件里不出现明文；
* **缺项即启动失败**，不用静默默认值兜底——凭据缺失时必须点名是哪一个；
* 密钥字段用 ``SecretStr`` 包装，``str()``/日志里只出现掩码，避免凭据进日志。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .core.failure_modes import FailureModeParams
from .core.smoothing import SmoothingParams
from .ingest.quality import PrecheckParams
from .ingest.resize import ResizePolicy, RoiParams, UploadBudget

__all__ = [
    "CREDENTIAL_GROUPS",
    "CredentialGroup",
    "MissingCredentialError",
    "Settings",
    "get_settings",
    "reset_settings_cache",
    "missing_credentials",
    "require_settings",
]


class CredentialGroup:
    """一组凭据位的名字与所含环境变量，供启动自检按组点名缺失项。"""

    __slots__ = ("key", "label", "env_names", "example")

    def __init__(self, key: str, label: str, env_names: Sequence[str], example: str) -> None:
        self.key = key
        self.label = label
        self.env_names = tuple(env_names)
        self.example = example

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "env_names": list(self.env_names),
            "example": self.example,
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<CredentialGroup {self.key}: {', '.join(self.env_names)}>"


#: 7 组凭据位。顺序即启动自检的报错顺序（出站 → 遥测 → 模型网关）。
CREDENTIAL_GROUPS: tuple[CredentialGroup, ...] = (
    CredentialGroup(
        "nurse_exec_base_url",
        "护理执行平台出站地址",
        ("NURSE_EXEC_BASE_URL",),
        "NURSE_EXEC_BASE_URL=https://nurse-exec.example-hospital.local",
    ),
    CredentialGroup(
        "nurse_exec_api_token",
        "护理执行平台出站凭据",
        ("NURSE_EXEC_API_TOKEN",),
        "NURSE_EXEC_API_TOKEN=<院内发放的 token>",
    ),
    CredentialGroup(
        "mqtt",
        "输液泵遥测入站（MQTT）",
        ("MQTT_HOST", "MQTT_PORT", "MQTT_USERNAME", "MQTT_PASSWORD"),
        "MQTT_HOST=… / MQTT_PORT=1883 / MQTT_USERNAME=… / MQTT_PASSWORD=…",
    ),
    CredentialGroup(
        "llm_base_url",
        "模型网关地址（OpenAI-compatible）",
        ("LLM_BASE_URL",),
        "LLM_BASE_URL=https://llm-gateway.example-hospital.local/v1",
    ),
    CredentialGroup(
        "llm_api_key",
        "模型网关凭据",
        ("LLM_API_KEY",),
        "LLM_API_KEY=<院内自建端点的 key>",
    ),
    CredentialGroup(
        "llm_model",
        "模型网关模型名",
        ("LLM_MODEL",),
        "LLM_MODEL=<院内端点已部署的模型名>",
    ),
    CredentialGroup(
        "llm_timeout",
        "模型网关超时（秒）",
        ("LLM_TIMEOUT",),
        "LLM_TIMEOUT=8.0",
    ),
)

_CREDENTIAL_ENV_NAMES: tuple[str, ...] = tuple(
    name for group in CREDENTIAL_GROUPS for name in group.env_names
)


def _is_absent(name: str) -> bool:
    """未设置，或已设置但只有空白 —— 两种情况都算缺凭据。"""
    return not os.environ.get(name, "").strip()


def _blank_env_names(env_names: Iterable[str]) -> list[str]:
    """已设置但值为空串/纯空白的环境变量（同样算缺失，便于点名）。"""
    out: list[str] = []
    for name in env_names:
        raw = os.environ.get(name)
        if raw is not None and not raw.strip():
            out.append(name)
    return out


def missing_credentials() -> tuple[list[CredentialGroup], list[str]]:
    """扫描当前进程环境，返回（缺失的组，已设但为空的变量名）。

    不抛异常，供配置自检命令在启动前打印清单。
    """
    absent_groups = [g for g in CREDENTIAL_GROUPS if any(_is_absent(n) for n in g.env_names)]
    return absent_groups, _blank_env_names(_CREDENTIAL_ENV_NAMES)


class MissingCredentialError(RuntimeError):
    """凭据位缺失：按组逐项点名缺哪一个，不静默用默认值兜底。"""

    def __init__(self, missing: Sequence[CredentialGroup], blank: Sequence[str] = ()) -> None:
        self.missing_groups = tuple(missing)
        self.blank_names = tuple(blank)
        lines = ["启动失败：以下凭据位缺失或为空（不设默认值，请按 .env.example 配置后重启）"]
        for group in missing:
            absent = [name for name in group.env_names if _is_absent(name)]
            blank_in_group = [name for name in group.env_names if name in self.blank_names]
            detail = absent or blank_in_group
            lines.append(f"  - [{group.key}] {group.label}：缺 {'、'.join(detail)}")
            lines.append(f"    示例：{group.example}")
        super().__init__("\n".join(lines))


class Settings(BaseSettings):
    """服务配置面：凭据位必填，行为参数带默认值。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── M5 出站：护理执行平台（写回换液事件）──────────────────────────
    nurse_exec_base_url: str = Field(description="护理执行平台出站基址")
    nurse_exec_api_token: SecretStr = Field(description="护理执行平台出站 Bearer 凭据")

    # ── M5 入站：输液泵遥测（MQTT）──────────────────────────────────
    mqtt_host: str = Field(description="院内 MQTT broker 地址")
    mqtt_port: int = Field(description="院内 MQTT broker 端口")
    mqtt_username: str = Field(description="MQTT 用户名")
    mqtt_password: SecretStr = Field(description="MQTT 密码")
    mqtt_topic_prefix: str = Field(default="hospital/infusion", description="遥测主题前缀")

    # ── M6 模型网关（OpenAI-compatible）─────────────────────────────
    llm_base_url: str = Field(description="OpenAI-compatible 端点 base_url")
    llm_api_key: SecretStr = Field(description="模型网关 API key")
    llm_model: str = Field(description="模型名")
    llm_timeout: float = Field(description="单次调用超时（秒）")

    # ── M2 预测器实现（几何核 / 生产权重）────────────────────────────
    # 这四项只在「换实现」时用；默认全走 core/config_map.py 的注册表，
    # 业务层一律调 build_predictor()，不直接 import 任何一种实现。
    model_impl: str = Field(default="geom_kernel", description="预测器实现名（geom_kernel / weights）")
    model_id: str = Field(default="geom_kernel_base", description="注册表里的配置 ID")
    model_config_path: str | None = Field(
        default=None, description="覆盖注册表项的推理配置文件；留空即用注册表登记的那份"
    )
    model_checkpoint: str | None = Field(
        default=None, description="覆盖注册表项的生产权重路径；几何核无需权重，留空即可"
    )
    model_registry_root: str = Field(
        default="./models", description="注册表根目录：配置与权重的相对路径按它解析"
    )
    model_registry_json: str | None = Field(
        default=None,
        description="JSON 对象，覆盖/追加注册表（按 model_id 合并），供院内多环境切换",
    )
    #: 液体的饱和度下限（OpenCV HSV 的 S）。**几何核与上传前预检共用这一个门限**：
    #: 预检量的「液面线」必须是主路径量的那批液体，否则两层会对同一段短片给出不同判断。
    model_saturation_min: int = Field(default=96, description="液体饱和度下限（HSV 的 S）")

    # ── M2 面积序列平滑（core/smoothing.py 的参数面）──────────────────
    # 体积与 ETA 都吃这条曲线，一个参数设错就是「还早着呢」被误报成「快见底了」，
    # 所以六个参数全部可配，且默认值按护士手机 10–20 秒短片标定过。
    smoothing_median_window: int = Field(
        default=5, description="中值滤波窗口（按观测点计，None 不占名额）"
    )
    smoothing_alpha: float = Field(
        default=0.6, description="一阶平滑中新观测的权重；压太小会让换液点名迟到"
    )
    smoothing_rise_tolerance: float = Field(
        default=0.004, description="单帧允许的面积比上升量，超过即判反常帧"
    )
    smoothing_spike_tolerance: float = Field(
        default=0.02, description="原始值与中值偏离超过此值标尖峰（反光高光）"
    )
    smoothing_min_gap_frames: int = Field(
        default=2, description="缺帧少于此数不做受限重建（留 null）"
    )
    smoothing_max_gap_frames: int = Field(
        default=10, description="受限重建的最大缺帧数，超过则拒绝重建"
    )

    # ── 拍摄质量分类（只降级并给补拍建议，不拒绝上传）────────────────
    quality_sample_frames: int = Field(
        default=5, description="质检均匀采样帧数；与序列长度无关，保证恒定算力"
    )
    quality_min_hit_ratio: float = Field(
        default=0.34, description="一段里至少多少比例的采样帧命中才算这一段拍坏了"
    )
    quality_glare_ratio_threshold: float = Field(
        default=0.0015, description="反光过曝判定：近白像素占全画面的比例门限"
    )
    quality_low_light_mean_luma: float = Field(
        default=46.0, description="暗光判定：全画面平均灰度门限"
    )
    quality_sharpness_min: float = Field(
        default=150.0, description="运动模糊判定：拉普拉斯方差（清晰度）下限"
    )
    quality_tilt_angle_deg: float = Field(
        default=8.0, description="角度不正判定：液面线倾角门限（度）"
    )
    quality_drip_tail_ratio: float = Field(
        default=0.55, description="滴斗空管判定：腔体底部窄带 / 最宽行 的门限"
    )
    # ── 上传前帧质量预检（ingest/quality.py；与上面几项同一组门限）──────
    # 预检与袋级分类共用质量门限，保证「预检说拍好了、袋级说拍坏了」不会出现；
    # 只有两个值是预检独有的：短边质量下限与总分门禁。
    precheck_min_short_side: int = Field(
        default=480,
        description=(
            "预检的短边质量下限：短边低于此值虽然还能算，但液面边界已贴采样极限，"
            "余量读数打 degraded（硬下限 320 仍由入站硬校验把关，低于它直接 422）"
        ),
    )
    precheck_degraded_score: float = Field(
        default=0.70,
        description="五项子分加权平均低于此值即标 degraded（即便没有一类命中）",
    )

    # ── M1 入站规模控制：长边缩放 + ROI 窗口（task 35）─────────────────
    # 一轮巡视最坏 1200 帧、原生长边动辄 4000px（护士手机 12MP）。这三项决定它在
    # 病区服务器上怎么才算跑得动：长边缩到上限（几何与体积是比例量，数值不变），
    # 短边**不得**缩到入站硬校验门限之下，只把输液架那块 ROI 喂给预测器。
    frame_resize_enabled: bool = Field(
        default=True,
        description=(
            "是否在解包落盘时把长边缩到上限。默认开：像素数少一个数量级，"
            "面积比/余量/ETA 全是比例量，数值一字不变"
        ),
    )
    frame_long_edge_max: int = Field(
        default=1280,
        description=(
            "长边缩放上限（像素）。形态学核是固定像素尺度的，再大只更慢不会更准；"
            "液面边界的分辨率需求在 1080–1280 已经够（合成回归：1280 下液面线偏差 "
            "≤0.4px，见 docs/synthetic-data.md）"
        ),
    )
    frame_short_edge_floor: int = Field(
        default=320,
        description=(
            "短边下限（像素）。缩放**不得**把短边压到这条线以下，否则同一段短片会因为"
            "「缩放策略」而不是「拍法」被判 422。不得低于 320（ingest/validate.py 的硬校验）"
        ),
    )
    roi_enabled: bool = Field(
        default=True,
        description=(
            "是否按首帧候选框把预测器限制在输液架所在的那块画面。默认开：画面里其余"
            "位置（隔壁床、输液杆）同样是液体色，照单全收就是误检"
        ),
    )
    roi_margin_ratio: float = Field(
        default=0.12,
        description="ROI 窗口相对候选框向外放多少（长边比例）；液面下降袋子会变矮，要留余量",
    )
    roi_max_step_ratio: float = Field(
        default=0.05,
        description=(
            "单次调用窗口边缘最多移动多少（长边比例）：逐帧抖动的裁剪框会在面积曲线上"
            "造出假波谷，而毫升数与 ETA 挂在这条曲线下游"
        ),
    )
    roi_alpha: float = Field(
        default=0.35,
        description="ROI 窗口指数平滑系数（新观测的权重）；与 max_step 一起构成平滑+防跳变",
    )
    roi_min_side: int = Field(
        default=96,
        description="ROI 窗口短边下限（像素）；比这更小的窗没有分辨力，宁可不开窗",
    )

    # ── M1 整段上传的字节/时长闸门（task 35）──────────────────────────
    # 超限立刻退回并指向续传接口，而不是把一个大包读进内存再超时——病区网络断在
    # 90% 是常态。真要传更大的段就走分片续传（upload_resume_* 那一组）。
    upload_max_bytes: int = Field(
        default=160 * 1024 * 1024,
        description=(
            "单次整段上传的字节上限（bytes）。1200 帧 × 1280 长边的 zip 约 150MB，"
            "160MB 留一点余量；超过即 413 并指路 POST /api/v1/infusion/uploads 分片续传"
        ),
    )
    upload_max_seconds: float = Field(
        default=120.0,
        description="单次整段上传的接收时限（秒）；超限即 408 并指路分片续传",
    )

    # ── M6 网关护栏 ─────────────────────────────────────────────────
    llm_daily_call_limit: int = Field(default=2000, description="每日模型调用上限")
    llm_max_tokens_per_request: int = Field(default=1200, description="单请求 token 上限")

    # ── M6 决策融合仲裁（gateway/fusion.py）─────────────────────────
    # 默认**关**：模型在本产品里只贡献「给护士看的那句话」，结论与触发原因码
    # 永远由确定性阈值路径给出（见 spec.md 的决策来源三态）。院内还没部署端点
    # 时保持关闭即可，服务照常工作——决策_source 恒为 threshold_only。
    llm_fusion_enabled: bool = Field(
        default=False,
        description=(
            "是否在换液点名后请模型复述一遍结论并做一致性仲裁。"
            "**关掉不影响任何结论**：阈值路径独立判定，模型缺席时 decision_source=threshold_only"
        ),
    )
    llm_fusion_confidence_floor: float = Field(
        default=0.60,
        description=(
            "仲裁置信门禁：模型自评置信度低于它一律回落阈值路径（与网关的可用性门禁同源，"
            "但问的不是同一个问题）"
        ),
    )

    # ── M4 存储 ─────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///./var/infusion_monitor.db",
        description="数据库连接串；默认 SQLite，换 PostgreSQL 只改此项",
    )
    frames_root: str = Field(default="./var/frames", description="会话帧目录根路径")

    # ── M4 保留期与归档（task 44）────────────────────────────────────
    # 帧目录是像素、会话行是读数：保留期到了清像素、留读数。
    # 默认 30 天：够覆盖一次月度护理质控复盘与换袋争议追溯；病区磁盘紧张可调到 7。
    retention_session_days: float = Field(
        default=30.0,
        description="会话帧目录保留天数；到点后帧被归档或删除，**读数仍留在库里**",
    )
    retention_inference_audit_days: float = Field(
        default=90.0,
        description=(
            "推理审计（模型调用元数据）保留天数。审计链 audit_log 是飞检留档，"
            "任何保留期都不动它"
        ),
    )
    retention_archive_root: str | None = Field(
        default=None,
        description=(
            "帧目录归档根目录；留空表示到期直接删除。配了就把过期帧搬进这里"
            "（按月分目录），信息科可在飞检期内从归档区调原始帧"
        ),
    )

    # ── M4 分片续传（task 32）─────────────────────────────────────────
    # 半成品上传的保留期与在途上限：病区网络断在拍摄中途时，那半段帧要留在盘上
    # 一段时间让护士能接着传，又不能无限期占着病区的磁盘。
    upload_resume_retention_hours: float = Field(
        default=24.0,
        description=(
            "开着没提交的分片上传保留多久（小时）。到期后由 "
            "POST /api/v1/infusion/uploads/cleanup 回收暂存目录并结案为 abandoned。"
            "一轮巡视 1–2 小时，24 小时足够护士当天补传；病区磁盘紧张时可调到 6"
        ),
    )
    upload_resume_max_pending: int = Field(
        default=50,
        description=(
            "同时处于 pending 的分片上传数上限。终端断网后反复重开会很快堆到这个数，"
            "超限后新的开启请求被拒并指路清理接口，避免暂存目录把病区磁盘吃满"
        ),
    )

    # ── 换液判定参数（只产出建议，不推进状态）────────────────────────
    change_fill_ratio_threshold: float = Field(
        default=0.12, description="面积比低于此值进入换液候选"
    )
    change_consecutive_frames: int = Field(default=5, description="连续帧数下限（双条件之一）")
    change_min_interval_minutes: int = Field(default=15, description="同袋两次点名的最小间隔")
    change_max_picks_per_round: int = Field(default=3, description="单轮点名袋数上限")

    # ── M3 标定版本与偏差闸门（task 27）─────────────────────────────
    # 改版后新旧读数相对差超过这个比例，就**不自动覆盖**旧读数：那种量级的差
    # 多半是新版本自己有问题，而不是「旧版本终于被修正」。被拦下的会话保留原读数
    # 并标待复核，等护理部在复核入口结案。
    calibration_review_deviation: float = Field(
        default=0.15,
        description="标定改版的相对差复核阈值（0.15 = 差 15% 即不自动覆盖）",
    )

    # ── M3 体积不确定度与置信门禁（task 28）──────────────────────────
    # 毫升数要可信才敢拿去点名换液。置信度低于阈值就**不出数**（响应里
    # estimated_volume_ml 字段整体不出现），并把成因码交出去让护士端显示对应提示。
    # 门限取值依据写在 calibration/uncertainty.py 的字段旁，不在配置面重复解释。
    uncertainty_min_confidence: float = Field(
        default=0.60, description="放行毫升数所需的最低置信度（0.60 = 区间半宽不超过余量四成）"
    )
    uncertainty_high_occlusion_ratio: float = Field(
        default=0.35, description="遮挡帧占比超过此值即判 high_occlusion 门禁、不出数"
    )
    uncertainty_sigma_fill_floor: float = Field(
        default=0.010,
        description="面积比噪声地板（合成回归实测最差拍法偏差 0.0120，取 0.010）",
    )
    uncertainty_geometry_model_sigma: float = Field(
        default=0.030, description="腔体几何换算自身的相对误差（登记精度）"
    )
    uncertainty_occlusion_gain: float = Field(
        default=0.20, description="遮挡占比换算成相对不确定度的增益"
    )
    uncertainty_sigma_ceiling: float = Field(
        default=0.35, description="置信度归零的相对不确定度（与最低置信度合起来才是门限）"
    )

    # ── 入站 webhook 兜底通道（可选）────────────────────────────────
    webhook_signing_secret: SecretStr | None = Field(
        default=None, description="webhook 签名密钥；未设则该通道关闭"
    )
    webhook_signature_tolerance_seconds: float = Field(
        default=300.0,
        description=(
            "带时间戳头的 webhook 请求允许的最大时钟偏差（秒），超过即按重放拒收。"
            "5 分钟够覆盖病区 NTP 抖动；把它调得比网络往返时间还小，"
            "对端会开始拒掉自己的真消息"
        ),
    )
    webhook_max_body_bytes: int = Field(
        default=64 * 1024,
        description=(
            "单条 webhook 请求体上限（字节）。一条遥测几十字节，这条闸门挡的是"
            "「先读完整个 body 再验签」那种实现；超限直接 413，不落盘不解析"
        ),
    )

    # ── M5 入站遥测韧性：断线重连与新鲜度（task 39）───────────────────
    mqtt_reconnect_base_seconds: float = Field(
        default=1.0, description="断线重连的初始退避（秒）"
    )
    mqtt_reconnect_factor: float = Field(
        default=2.0, description="退避倍数：第 n 次失败等 base × factor^(n-1)"
    )
    mqtt_reconnect_max_seconds: float = Field(
        default=32.0, description="退避封顶（秒）；封顶后按固定间隔重试，不再指数增长"
    )
    mqtt_reconnect_max_attempts: int = Field(
        default=0,
        description=(
            "连续失败多少次后停止自动重连（0 = 一直重试）。遥测只是交叉校验输入，"
            "默认不放弃；设成正数则放弃后由信息科在运行摘要里看到并手工处理"
        ),
    )
    mqtt_reconnect_reset_seconds: float = Field(
        default=300.0,
        description=(
            "连稳这么久才把退避计数清零。短于这个时间的「连上又掉」继续退避，"
            "否则 broker 抖动时会一路退到封顶、再也恢复不到快速重连"
        ),
    )
    telemetry_fresh_seconds: float = Field(
        default=300.0,
        description=(
            "泵速遥测的新鲜窗口（秒）：超过即在响应里报 telemetry_state=stale。"
            "**只标注不拦数**——余量与换液判定不依赖它"
        ),
    )

    # ── M5 通道开关与出站超时 ──────────────────────────────────────────
    mqtt_enabled: bool = Field(
        default=False,
        description="是否启动 MQTT 遥测订阅线程。默认关闭：遥测只是交叉校验，"
        "缺它不影响上传、毫升估计与 ETA（ETA 退回医嘱速率）。病区部署接了泵网关"
        "再置为 true；测试与本机演示保持关闭，避免每轮都去连一个不存在的 broker。",
    )
    writeback_timeout: float = Field(
        default=5.0, description="出站写回单次请求超时（秒）"
    )
    writeback_retry_enabled: bool = Field(
        default=True,
        description=(
            "是否后台重试到期的待写回队列条目（task 37）。默认开："
            "服务重启后那笔「试了一半」的换液事件要有人继续把它送出去，"
            "而不是躺在队列里等人发现。关掉它只剩人工重放一条路。"
        ),
    )
    writeback_retry_interval_seconds: float = Field(
        default=5.0, description="待写回队列的扫描间隔（秒）"
    )

    @field_validator("nurse_exec_base_url", "llm_base_url")
    @classmethod
    def _must_be_http_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("必须是以 http:// 或 https:// 开头的地址")
        return value

    @field_validator(
        "llm_timeout",
        "change_fill_ratio_threshold",
        "writeback_timeout",
        "uncertainty_min_confidence",
        "uncertainty_high_occlusion_ratio",
        "uncertainty_sigma_fill_floor",
        "uncertainty_geometry_model_sigma",
        "uncertainty_occlusion_gain",
        "uncertainty_sigma_ceiling",
        "webhook_signature_tolerance_seconds",
        "mqtt_reconnect_base_seconds",
        "mqtt_reconnect_max_seconds",
        "mqtt_reconnect_reset_seconds",
        "telemetry_fresh_seconds",
    )
    @classmethod
    def _must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("必须为正数")
        return value

    @field_validator("change_consecutive_frames", "change_min_interval_minutes")
    @classmethod
    def _must_be_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("必须为正整数")
        return value

    @field_validator(
        "smoothing_median_window",
        "smoothing_min_gap_frames",
        "smoothing_max_gap_frames",
    )
    @classmethod
    def _smoothing_counts_are_sane(cls, value: int) -> int:
        if value < 1:
            raise ValueError("必须为正整数")
        return value

    @field_validator("smoothing_alpha")
    @classmethod
    def _alpha_in_unit_interval(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("一阶平滑权重须落在 (0, 1]")
        return value

    @field_validator(
        "smoothing_rise_tolerance",
        "smoothing_spike_tolerance",
    )
    @classmethod
    def _tolerances_are_sane(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("容差须落在 [0, 1)")
        return value

    @field_validator("smoothing_max_gap_frames")
    @classmethod
    def _max_gap_covers_min(cls, value: int, info) -> int:
        minimum = (info.data or {}).get("smoothing_min_gap_frames")
        if minimum is not None and value < minimum:
            raise ValueError("SMOOTHING_MAX_GAP_FRAMES 不得小于 SMOOTHING_MIN_GAP_FRAMES")
        return value

    @field_validator("change_fill_ratio_threshold")
    @classmethod
    def _ratio_in_unit_interval(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("面积比阈值须落在 (0, 1]")
        return value

    @field_validator(
        "quality_glare_ratio_threshold",
        "quality_min_hit_ratio",
        "quality_drip_tail_ratio",
    )
    @classmethod
    def _quality_ratios_are_sane(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("拍摄质量门限须落在 (0, 1)")
        return value

    @field_validator(
        "quality_low_light_mean_luma", "quality_sharpness_min"
    )
    @classmethod
    def _quality_levels_are_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("必须为正数")
        return value

    @field_validator("quality_sample_frames")
    @classmethod
    def _quality_sample_frames_is_sane(cls, value: int) -> int:
        if not 1 <= value <= 20:
            raise ValueError("采样帧数须落在 [1, 20]：多了拖慢质检，少了容易误判")
        return value

    @field_validator("model_saturation_min")
    @classmethod
    def _saturation_min_is_sane(cls, value: int) -> int:
        if not 1 <= value <= 255:
            raise ValueError("饱和度下限须落在 [1, 255]")
        return value

    @field_validator("quality_tilt_angle_deg")
    @classmethod
    def _tilt_threshold_is_sane(cls, value: float) -> float:
        if not 0.0 < value < 90.0:
            raise ValueError("倾角门限须落在 (0, 90) 度")
        return value

    @field_validator("precheck_min_short_side")
    @classmethod
    def _precheck_short_side_is_sane(cls, value: int) -> int:
        if not 1 <= value <= 4096:
            raise ValueError("短边质量下限须落在 [1, 4096] 像素")
        return value

    @field_validator("precheck_degraded_score")
    @classmethod
    def _precheck_score_is_sane(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("质量分门禁须落在 (0, 1]")
        return value

    @field_validator("frame_long_edge_max")
    @classmethod
    def _long_edge_max_is_sane(cls, value: int) -> int:
        if not 320 <= value <= 4096:
            raise ValueError("长边缩放上限须落在 [320, 4096] 像素")
        return value

    @field_validator("frame_short_edge_floor")
    @classmethod
    def _short_edge_floor_is_sane(cls, value: int) -> int:
        from .ingest.validate import MIN_SHORT_SIDE

        if not MIN_SHORT_SIDE <= value <= 2048:
            raise ValueError(
                f"短边下限须落在 [{MIN_SHORT_SIDE}, 2048] 像素；低于 {MIN_SHORT_SIDE} 会让"
                "缩放把画面压出入站硬校验门限，同一段短片会因为策略而不是拍法被判 422"
            )
        return value

    @field_validator("roi_margin_ratio", "roi_max_step_ratio")
    @classmethod
    def _roi_ratio_is_sane(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("ROI 比例须落在 [0, 1]")
        return value

    @field_validator("roi_alpha")
    @classmethod
    def _roi_alpha_is_sane(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("ROI 平滑系数须落在 (0, 1]")
        return value

    @field_validator("roi_min_side")
    @classmethod
    def _roi_min_side_is_sane(cls, value: int) -> int:
        if not 16 <= value <= 2048:
            raise ValueError("ROI 窗口短边下限须落在 [16, 2048] 像素")
        return value

    @field_validator("upload_max_bytes")
    @classmethod
    def _upload_max_bytes_is_sane(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("整段上传字节上限必须是正整数（置 0 会把所有上传都拒掉）")
        return value

    @field_validator("upload_max_seconds")
    @classmethod
    def _upload_max_seconds_is_sane(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("整段上传时限必须是正数（置 0 会把所有上传都拒掉）")
        return value

    @field_validator("mqtt_reconnect_factor")
    @classmethod
    def _reconnect_factor_is_sane(cls, value: float) -> float:
        if value < 1:
            raise ValueError("重连退避倍数不得小于 1（小于 1 会越退越密，把 broker 打死）")
        return value

    @field_validator("mqtt_reconnect_max_attempts")
    @classmethod
    def _reconnect_attempts_are_sane(cls, value: int) -> int:
        if value < 0:
            raise ValueError("重连次数上限不得为负（0 表示一直重试）")
        return value

    @field_validator("webhook_max_body_bytes")
    @classmethod
    def _webhook_body_cap_is_sane(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("webhook 请求体上限必须是正整数（置 0 会把整条通道拒死）")
        return value

    # ── 派生视图 ────────────────────────────────────────────────────
    @property
    def nurse_exec_events_url(self) -> str:
        """出站写回端点：{NURSE_EXEC_BASE_URL}/api/v1/infusion/events。"""
        return f"{self.nurse_exec_base_url}/api/v1/infusion/events"

    @property
    def mqtt_telemetry_topic(self) -> str:
        """入站遥测主题：hospital/infusion/+/telemetry（+ 为床号/架号段）。"""
        return f"{self.mqtt_topic_prefix.strip('/')}/+/telemetry"

    @property
    def reconnect_policy(self) -> "ReconnectPolicy":
        """断线重连退避策略（adapters.mqtt_telemetry.ReconnectPolicy）。

        配置面到算法面的唯一入口，业务代码不自己拼参数对象。
        """
        from .adapters.mqtt_telemetry import ReconnectPolicy

        return ReconnectPolicy(
            base_seconds=self.mqtt_reconnect_base_seconds,
            factor=self.mqtt_reconnect_factor,
            max_seconds=self.mqtt_reconnect_max_seconds,
            max_attempts=self.mqtt_reconnect_max_attempts,
            reset_after_seconds=self.mqtt_reconnect_reset_seconds,
        )

    @property
    def webhook_enabled(self) -> bool:
        """webhook 兜底通道是否启用（配了签名密钥才算启用）。

        没配就**没有这个端点**（404 webhook_disabled），而不是拿一个默认密钥开着它。
        """
        secret = self.webhook_signing_secret
        return secret is not None and bool(secret.get_secret_value().strip())

    @property
    def smoothing_params(self) -> SmoothingParams:
        """平滑参数（core.smoothing.SmoothingParams）。这里是配置面到算法面的唯一入口，
        业务代码不自己拼参数对象。"""
        return SmoothingParams(
            median_window=self.smoothing_median_window,
            alpha=self.smoothing_alpha,
            rise_tolerance=self.smoothing_rise_tolerance,
            spike_tolerance=self.smoothing_spike_tolerance,
            min_gap_frames=self.smoothing_min_gap_frames,
            max_gap_frames=self.smoothing_max_gap_frames,
        )

    @property
    def calibration_review_threshold(self) -> float:
        """标定改版偏差阈值。配置面到算法面的唯一入口（0.15 = 差 15% 就要复核）。"""
        return self.calibration_review_deviation

    @property
    def uncertainty_params(self) -> "UncertaintyParams":
        """体积不确定度门限（calibration.uncertainty.UncertaintyParams）。
        配置面到算法面的唯一入口，业务代码不自己拼参数对象。"""
        from .calibration.uncertainty import UncertaintyParams

        return UncertaintyParams(
            min_confidence=self.uncertainty_min_confidence,
            high_occlusion_ratio=self.uncertainty_high_occlusion_ratio,
            sigma_fill_floor=self.uncertainty_sigma_fill_floor,
            geometry_model_sigma=self.uncertainty_geometry_model_sigma,
            occlusion_sigma_gain=self.uncertainty_occlusion_gain,
            sigma_ceiling=self.uncertainty_sigma_ceiling,
        )

    @property
    def quality_params(self) -> FailureModeParams:
        """拍摄质量分类门限（core.failure_modes.FailureModeParams）。
        配置面到算法面的唯一入口，业务代码不自己拼参数对象。"""
        return FailureModeParams(
            sample_frames=self.quality_sample_frames,
            min_hit_ratio=self.quality_min_hit_ratio,
            glare_ratio_threshold=self.quality_glare_ratio_threshold,
            low_light_mean_luma=self.quality_low_light_mean_luma,
            sharpness_min=self.quality_sharpness_min,
            tilt_angle_deg=self.quality_tilt_angle_deg,
            drip_tail_ratio=self.quality_drip_tail_ratio,
        )

    @property
    def precheck_params(self) -> PrecheckParams:
        """上传前预检门限（ingest.quality.PrecheckParams）。配置面到算法面的唯一入口。

        门限**取自同一组 quality_* 配置项**——两层质检共用一套门限是有意的：护士端不该
        看到「预检说拍好了、袋级说拍坏了」。预检独有的两项（短边质量下限、总分门禁）另立配置。
        """
        return PrecheckParams(
            sample_frames=self.quality_sample_frames,
            min_hit_ratio=self.quality_min_hit_ratio,
            sharpness_min=self.quality_sharpness_min,
            glare_ratio_threshold=self.quality_glare_ratio_threshold,
            low_light_mean_luma=self.quality_low_light_mean_luma,
            min_short_side=self.precheck_min_short_side,
            tilt_angle_deg=self.quality_tilt_angle_deg,
            degraded_score=self.precheck_degraded_score,
            saturation_min=self.model_saturation_min,
        )

    @property
    def resize_policy(self) -> "ResizePolicy":
        """入站缩放策略（ingest.resize.ResizePolicy）。配置面到算法面的唯一入口。"""
        return ResizePolicy(
            long_edge_max=self.frame_long_edge_max,
            short_edge_floor=self.frame_short_edge_floor,
            enabled=self.frame_resize_enabled,
        )

    @property
    def roi_params(self) -> "RoiParams":
        """ROI 窗口门限（ingest.resize.RoiParams）。配置面到算法面的唯一入口。"""
        return RoiParams(
            enabled=self.roi_enabled,
            margin_ratio=self.roi_margin_ratio,
            max_step_ratio=self.roi_max_step_ratio,
            alpha=self.roi_alpha,
            min_side=self.roi_min_side,
        )

    @property
    def upload_budget(self) -> "UploadBudget":
        """整段上传的字节/时长闸门（ingest.resize.UploadBudget）。"""
        return UploadBudget(max_bytes=self.upload_max_bytes, max_seconds=self.upload_max_seconds)

    @property
    def sqlite_path(self) -> str | None:
        """从 DATABASE_URL 取 SQLite 文件路径；非 SQLite 形态返回 None。"""
        from .store.dsn import parse_dsn

        dsn = parse_dsn(self.database_url)
        return dsn.sqlite_path if dsn.is_sqlite else None

    @property
    def retention_policy(self) -> "RetentionPolicy":
        """保留期策略（store.retention.RetentionPolicy）。配置面到策略面的唯一入口。"""
        from .store.retention import RetentionPolicy

        return RetentionPolicy(
            session_retention_days=self.retention_session_days,
            inference_audit_retention_days=self.retention_inference_audit_days,
            archive_root=Path(self.retention_archive_root) if self.retention_archive_root else None,
        )

    def masked_credentials(self) -> dict[str, str]:
        """供运行摘要与启动日志使用：密钥只给掩码，绝不打明文。"""
        return {
            "nurse_exec_api_token": _mask(self.nurse_exec_api_token),
            "mqtt_password": _mask(self.mqtt_password),
            "llm_api_key": _mask(self.llm_api_key),
            "webhook_signing_secret": _mask(self.webhook_signing_secret),
        }

    def describe(self) -> dict:
        """可安全打印的配置摘要（不含任何明文凭据）。"""
        return {
            "nurse_exec_base_url": self.nurse_exec_base_url,
            "nurse_exec_events_url": self.nurse_exec_events_url,
            "writeback_timeout": self.writeback_timeout,
            "writeback_retry_enabled": self.writeback_retry_enabled,
            "writeback_retry_interval_seconds": self.writeback_retry_interval_seconds,
            "mqtt_host": self.mqtt_host,
            "mqtt_port": self.mqtt_port,
            "mqtt_enabled": self.mqtt_enabled,
            "mqtt_telemetry_topic": self.mqtt_telemetry_topic,
            "mqtt_reconnect": self.reconnect_policy.as_dict(),
            "telemetry_fresh_seconds": self.telemetry_fresh_seconds,
            "webhook_enabled": self.webhook_enabled,
            "webhook_signature_tolerance_seconds": self.webhook_signature_tolerance_seconds,
            "webhook_max_body_bytes": self.webhook_max_body_bytes,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_timeout": self.llm_timeout,
            "model_impl": self.model_impl,
            "model_id": self.model_id,
            "model_config_path": self.model_config_path or "(注册表默认)",
            "model_checkpoint": self.model_checkpoint or "(无)",
            "model_registry_root": self.model_registry_root,
            "model_saturation_min": self.model_saturation_min,
            "database_url": self.database_url,
            "frames_root": self.frames_root,
            "retention": self.retention_policy.as_dict(),
            "upload_resume_retention_hours": self.upload_resume_retention_hours,
            "upload_resume_max_pending": self.upload_resume_max_pending,
            "resize": self.resize_policy.as_dict(),
            "roi": self.roi_params.as_dict(),
            "upload_budget": self.upload_budget.as_dict(),
            "smoothing": self.smoothing_params.as_dict(),
            "quality": self.quality_params.as_dict(),
            "precheck": self.precheck_params.as_dict(),
            "calibration_review_deviation": self.calibration_review_deviation,
            "uncertainty": self.uncertainty_params.as_dict(),
            "credentials": self.masked_credentials(),
        }


def _mask(secret: SecretStr | None) -> str:
    if secret is None:
        return "(未配置)"
    raw = secret.get_secret_value()
    if not raw:
        return "(空)"
    if len(raw) <= 4:
        return "***"
    return f"{raw[:2]}***{raw[-2:]}"


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings()


def get_settings() -> Settings:
    """取配置；凭据缺失时抛 MissingCredentialError（点名缺哪一个）。"""
    missing, blank = missing_credentials()
    if missing or blank:
        raise MissingCredentialError(missing, blank)
    return _cached_settings()


def require_settings() -> tuple[Settings | None, list[CredentialGroup], list[str]]:
    """配置自检入口：返回（settings 或 None，缺失组，空值变量名）。

    用于启动前的 ``--check-config`` 自检命令与测试；不抛异常。
    """
    missing, blank = missing_credentials()
    if missing or blank:
        return None, missing, blank
    return Settings(), [], []


def reset_settings_cache() -> None:
    """测试与热加载用：清掉 lru_cache 的配置实例。"""
    _cached_settings.cache_clear()