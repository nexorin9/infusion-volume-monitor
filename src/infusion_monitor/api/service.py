"""FastAPI 应用工厂：常驻 HTTP API 服务（无界面）。

* 启动时按 ``config.get_settings()`` 读环境变量，**凭据缺一项就起不来**并点名缺哪个；
* SQLite 幂等迁移在 lifespan 里跑一次，重复启动不报错；
* 帧目录、库连接与上传暂存目录都挂在 ``app.state`` 上，测试可注入临时路径。
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from .. import __version__
from ..adapters.mqtt_telemetry import TelemetryChannel
from ..adapters.nurse_exec import NurseExecClient
from ..adapters.retry_queue import WritebackQueue, drain_due
from ..adapters.webhook_telemetry import WebhookTelemetryReceiver
from ..calibration.estimator import TelemetryCache
from ..config import Settings, get_settings
from ..core.config_map import build_predictor
from ..core.predictor import PerPointMaskPredictor
from ..gateway.audit import GatewayAuditor
from ..gateway.model_gateway import ModelGateway
from ..identity.change_detector import AuditPickupHistory, ChangeDetector
from ..ingest.multipart import router as tracks_router
from ..ingest.resume import router as resume_router
from ..logging_utils import install_request_logging, log_event
from ..store import (
    AuditChain,
    CalibrationRepository,
    DeviationRepository,
    EventRepository,
    InferenceAuditRepository,
    SessionRepository,
    UploadRepository,
    WritebackQueueRepository,
    apply_migrations,
    connect,
)
from ..store.migrations import current_version
from . import health, summary
from .calibration import router as calibration_router
from .audit import router as audit_router
from .claim import router as claim_router
from .contract import error_codes_router, install_error_contract
from .errors import install_error_handlers
from .pending import router as pending_router
from .pickup_payload import router as pickup_router
from .webhooks import router as webhook_router
from .transition import router as transition_router

__all__ = ["create_app", "AppStore", "app"]

#: 启动/停止事件用的 logger。放在模块级而不是 lifespan 内部，是因为 lifespan 的
#: 闭包每次建应用都重建一份 logger，测试里按名字抓 handler 会抓不到。
_startup_logger = logging.getLogger("infusion_monitor.startup")


@dataclass
class AppStore:
    """一次进程内的存储句柄。"""

    conn: sqlite3.Connection
    sessions: SessionRepository
    #: 分片续传的半成品台账（task 32）。开着没提交的那一段帧挂在它下面。
    uploads: UploadRepository
    events: EventRepository
    audit: InferenceAuditRepository
    calibrations: CalibrationRepository
    #: 自校准样本（append-only）：用护士确认的换液事件反推残留偏差项。
    deviations: DeviationRepository
    #: append-only 审计链（认领 / 重锚 / 状态迁移）。与 ``audit``（推理审计）分开：
    #: 前者回答「谁动了业务状态」，后者回答「这次结论是模型给的还是阈值给的」。
    audit_chain: AuditChain
    #: 待写回队列（task 37）：出站欠账的持久台账 + 死信 + 重放入口。
    writeback_queue: WritebackQueue

    def close(self) -> None:
        self.conn.close()


async def _retry_pump(app: FastAPI) -> None:
    """后台扫描待写回队列，把**到期**的条目重发一次。

    这是队列「活着」的那一拍。否则队列只是一张静态台账：重启后那笔试了一半的
    换液事件躺在那里，直到有人想起来去看它。

    只碰 ``queued`` 且到期的条目；``dead_letter`` 的自动预算已经用完，再自动发
    下去就是无限重试——那件事留给信息科点重放。
    """
    interval = app.state.settings.writeback_retry_interval_seconds
    while True:
        await asyncio.sleep(interval)
        try:
            results = await drain_due(
                app.state.store.writeback_queue,
                app.state.store.sessions,
                app.state.store.events,
                app.state.nurse_exec,
                audit=app.state.store.audit_chain,
                request_id="writeback-pump",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - 扫描失败不能带倒整个服务
            log_event(
                _startup_logger,
                logging.WARNING,
                "writeback.pump_failed",
                error=type(exc).__name__,
            )
            continue
        for result in results:
            if not result.ok:
                log_event(
                    _startup_logger,
                    logging.WARNING,
                    "writeback.pump_retry_failed",
                    event_id=result.event_id,
                    kind=result.kind,
                )


def create_app(
    settings: Settings | None = None,
    *,
    database_url: str | None = None,
    frames_root: str | Path | None = None,
    predictor: PerPointMaskPredictor | None = None,
    model_gateway: ModelGateway | None = None,
    change_detector: ChangeDetector | None = None,
) -> FastAPI:
    """构造应用。``settings``/``database_url``/``frames_root``/``predictor`` 供测试注入。

    ``predictor`` 默认按配置面构建（:func:`core.config_map.build_predictor`）：本轮
    ``MODEL_IMPL=geom_kernel``，几何核既是默认实现也是模型不可达时的降级路径；换生产
    权重只改 ``MODEL_IMPL`` / ``MODEL_ID`` 环境变量，业务代码不动，权重缺失则启动即
    报错而**不**静默退回几何核。

    ``model_gateway`` 默认按配置面构造。它挂在 ``app.state`` 上，是业务层调用模型
    的**唯一**入口——换院内端点只改 ``LLM_BASE_URL`` 等环境变量，业务代码不动。
    它只产出提示（``GatewayAdvice``），不推进任何业务状态。

    **所有路由处理函数都写成 ``async def``**：同步处理函数会被 Starlette 丢进线程池，
    而 SQLite 连接建在事件循环线程上，跨线程使用会抛
    ``SQLite objects created in a thread can only be used in that same thread``。
    病区侧单机部署下 SQLite 写入是毫秒级，让它跑在事件循环里比跨线程加锁更简单可靠。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved = settings or get_settings()
        url = database_url or resolved.database_url
        root = Path(frames_root or resolved.frames_root)
        root.mkdir(parents=True, exist_ok=True)
        spool = Path(tempfile.mkdtemp(prefix="infusion-upload-"))
        conn = connect(url)
        apply_migrations(conn)
        app.state.settings = resolved
        app.state.frames_root = root
        app.state.upload_spool = spool
        app.state.predictor = predictor or build_predictor(resolved)
        # 泵速遥测缓存：视觉主路径不依赖它，遥测缺失只降低交叉校验能力。
        # MQTT 订阅线程由 TelemetryChannel 持有，默认不启动（MQTT_ENABLED 未开时
        # 本机演示与测试不会去连一个不存在的 broker）。
        app.state.telemetry = TelemetryCache()
        app.state.store = AppStore(
            conn=conn,
            sessions=SessionRepository(conn),
            uploads=UploadRepository(conn),
            events=EventRepository(conn),
            audit=InferenceAuditRepository(conn),
            calibrations=CalibrationRepository(conn),
            deviations=DeviationRepository(conn),
            audit_chain=AuditChain(conn),
            writeback_queue=WritebackQueue(
                WritebackQueueRepository(conn), EventRepository(conn)
            ),
        )
        # 出站写回客户端：真实端点 + Bearer 凭据（凭据只在请求头里出现）。
        app.state.nurse_exec = NurseExecClient.from_settings(resolved)
        app.state.telemetry_channel = TelemetryChannel(
            resolved, app.state.telemetry, audit=app.state.store.audit_chain
        )
        # 入站双通道：MQTT 订阅 + webhook 兜底。**两者共用同一个 TelemetryChannel**
        # （同一份解析、去重表与遥测缓存）——切通道的灰度期里同一条遥测会从两条路各
        # 进来一次，两个各写一套的实现必然在某次固件差异上给出不同结论，而那种
        # 「两条遥测不一致」的问题比「没收到遥测」难查得多。
        app.state.telemetry_webhook = WebhookTelemetryReceiver(
            app.state.telemetry_channel,
            secret=(
                resolved.webhook_signing_secret.get_secret_value()
                if resolved.webhook_enabled
                else ""
            ),
            tolerance_seconds=resolved.webhook_signature_tolerance_seconds,
            max_body_bytes=resolved.webhook_max_body_bytes,
            audit=app.state.store.audit_chain,
            prefix=resolved.mqtt_topic_prefix,
        )
        if resolved.mqtt_enabled:
            app.state.telemetry_channel.start()
            # 退避重连线程：broker 重启会让会话失效，这时只有自己重新 connect 才走得通
            app.state.telemetry_channel.start_supervisor()
        # 模型网关：业务层调模型的唯一入口。不注入网关对象时按配置面构造真实客户端；
        # 注入只用于协议兼容性与降级路径的测试（注入的是传输层，报头与请求体照常组装）。
        gateway = model_gateway or ModelGateway.from_settings(resolved)
        gateway.auditor = gateway.auditor or GatewayAuditor(app.state.store.audit)
        app.state.model_gateway = gateway
        # 换液点名引擎：滞回带 / 每袋最小间隔 / 每轮上限三道闸都在这里。
        # 最小间隔这一道是**持久**的——上一次点名时刻从审计链的 pickup.named 行读，
        # 进程重启后不会因为内存清空而把同一袋再点一遍名。
        app.state.change_detector = change_detector or ChangeDetector(
            history=AuditPickupHistory(conn, audit=app.state.store.audit_chain)
        )
        # 待写回队列（task 37）：启动时先按事件表补齐欠账——进程重启前那笔
        # 「试了一半」的换液事件要有人接着送出去，而不是躺在事件表里没人知道。
        recovered = app.state.store.writeback_queue.reconcile(app.state.store.events)
        if recovered:
            log_event(
                _startup_logger,
                logging.WARNING,
                "writeback.queue_recovered",
                count=len(recovered),
                event_ids=[row.get("event_id") for row in recovered],
            )
        pump: asyncio.Task | None = None
        if resolved.writeback_retry_enabled:
            pump = asyncio.create_task(_retry_pump(app))
        # 启动行：一次就把「这次上线接了谁、开了哪些通道」交代清楚。
        # describe() 里的三个密钥已掩码，直接进日志不会泄漏凭据。
        log_event(
            _startup_logger,
            logging.INFO,
            "service.started",
            version=__version__,
            schema_version=current_version(conn),
            **resolved.describe(),
        )
        try:
            yield
        finally:
            if pump is not None:
                pump.cancel()
                try:
                    await pump
                except (asyncio.CancelledError, Exception):  # noqa: B014 - 关停时的异常一律吞掉
                    pass
            log_event(_startup_logger, logging.INFO, "service.stopping")
            app.state.telemetry_channel.stop()  # 含停掉退避重连线程
            # 退避重连线程停下前攒下的通道审计，在关库前落库（异线程不能直接写主连接，
            # 见 mqtt_telemetry.PENDING_AUDIT_MAX）——否则最后那几条「刚才断过」随进程没了。
            app.state.telemetry_channel.flush_pending_audit()
            app.state.model_gateway.close()
            app.state.nurse_exec.close()
            app.state.store.close()
            for leftover in spool.glob("*"):
                leftover.unlink(missing_ok=True)
            spool.rmdir()

    app = FastAPI(
        title="输液余量监测系统 · HTTP API",
        version=__version__,
        description=(
            "移动护理终端上传输液架短片帧序列，服务返回每袋余量曲线、"
            "标定后的毫升估计与预计耗尽时点，并把护士确认后的换液事件写回护理执行平台。\n\n"
            "入站：POST /api/v1/infusion/tracks（multipart/form-data，帧为 zip）。\n"
            "校验不通过返回 422，detail 为 `field / reason / suggestion` 三段式。\n\n"
            "**每个接口的 responses 里已逐条写明它会返回的错误码**，描述取自与本服务同一份\n"
            "错误码表（`GET /api/v1/error-codes` 拉机器可读版本）。"
        ),
        lifespan=lifespan,
    )
    install_error_handlers(app)
    install_error_contract(app)
    install_request_logging(app)
    app.include_router(health.router)
    app.include_router(summary.router)
    app.include_router(tracks_router)
    app.include_router(resume_router)
    app.include_router(calibration_router)
    app.include_router(claim_router)
    app.include_router(pickup_router)
    app.include_router(transition_router)
    app.include_router(pending_router)
    app.include_router(webhook_router)
    app.include_router(audit_router)
    app.include_router(error_codes_router)
    return app


#: ``uvicorn infusion_monitor.api:app`` 的入口对象。
#: 构造时不读凭据（凭据在 lifespan 启动时校验），因此 import 本模块不会因缺环境变量而失败。
app = create_app()