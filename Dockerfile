# 输液余量监测系统 —— 病区侧单机部署镜像
#
# 病区侧常见的是一台 Windows 工控机或一台 Linux 小主机，跑一个病区、几个挂点。
# 所以这个镜像刻意做得很小：不带 GPU、不带模型权重（本轮默认用确定性几何核），
# 只留 Python 运行时 + OpenCV 的无头构建。帧目录与 SQLite 挂在 volume 里，
# 容器重建不丢历史会话。
#
# 凭据**不**打进镜像。全靠运行时环境变量或 env_file 注入——镜像里出现真实 token，
# 等于把整个护理执行平台的凭据散落在每一台装过它的机器上。

FROM python:3.10-slim

# libglib2.0-0 / libgl1 是 opencv-python-headless 在部分 slim 基础镜像上的运行期
# 依赖；缺了会在 import cv2 时报 `libGL.so.1: cannot open shared object file`。
# tini 用作 init：uvicorn 要能在 SIGTERM 后把在途请求写完。
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libglib2.0-0 libgl1 curl tini \
 && rm -rf /var/lib/apt/lists/*

# 默认值都在这里，真正要改的（凭据、院内地址）由运行时 env_file 覆盖。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    FRAMES_ROOT=/data/frames \
    DATABASE_URL=sqlite:////data/infusion_monitor.db \
    LOG_FORMAT=json \
    LOG_LEVEL=INFO

WORKDIR /app

# 依赖先装：业务代码改动不会让这一层缓存失效，重建镜像快得多。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY README.md INSTALL.md ./

# 非 root 运行。帧目录与库文件都落在挂载点上，属主在 INSTALL.md 里给 chown。
RUN useradd --create-home --uid 10001 infusion \
 && mkdir -p /data/frames \
 && chown -R infusion:infusion /data /app
USER infusion

VOLUME ["/data"]
EXPOSE 8000

# 探活用 health 而不是 summary：health 永远便宜、永远 200；摘要要查四张表的聚合，
# 不该被 docker 每 30 秒打一次。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]

# 走 server:main 而不是裸 uvicorn：结构化日志在那一步装好。
# 需要调 uvicorn 自己的参数时用 `python -m infusion_monitor.server` 的等价形式，
# 或把下面替换成 uvicorn infusion_monitor.server:app（此时日志配置需自行调用）。
CMD ["python", "-m", "infusion_monitor.server"]