# 病区侧安装与运维说明

面向**在病区那台机器上把服务跑起来**的信息科同事。开发环境请看 README 的「快速开始」。

本服务是**无界面的常驻 HTTP API**：一个 uvicorn 进程 + 一个 MQTT 订阅线程，
状态落在 SQLite 与帧目录里。它不自己决定换液——只算「该换液了」并把 2–3 袋点名给护士，
换没换永远由护士在终端点确认。这一条在下面每一步都会影响到具体配置，别绕过去。

---

## 一、装之前先确认三件事

| 事项 | 为什么 | 找谁要 |
|------|--------|--------|
| 护理执行平台的出站地址与 token | 没它换液事件送不出去（事件会落 `pending_writeback` 排队，不会丢） | 护理部信息科 / 对端厂商 |
| 院内模型端点（OpenAI-compatible）的 base_url / key / 模型名 | 没它判定走确定性阈值路径，结果仍可用但标 `threshold_fallback` 需人工复核 | 院内 AI 平台 |
| MQTT broker（可选） | **不影响使用**：没有它 ETA 退回医嘱速率，视觉主路径照常。接了泵网关再开 | 后勤 / 设备科 |

> 服务在缺凭据时**故意起不来**，不会带着半个配置运行。所以「进程起不来」通常就是
> 「某几个环境变量没给全」。先跑第五节的自检命令，别急着看日志。

---

## 二、目录与权限

| 路径 | 装在哪 | 谁写 | 为什么要它 |
|------|--------|------|-----------|
| 代码 | `/opt/infusion-volume-monitor` | 只读 | — |
| 帧目录 | `/var/lib/infusion-monitor/frames` | 服务用户 | 每次巡视的原始帧；认领纠错时按帧重算曲线，删了就无法纠错 |
| 归档区 | `/var/lib/infusion-monitor/archive` | 服务用户 | 可选。配了 `RETENTION_ARCHIVE_ROOT` 才建；到点的帧搬这里而不是删掉 |
| SQLite 库 | `/var/lib/infusion-monitor/infusion_monitor.db` | 服务用户 | 会话、事件、审计、推理审计 |
| 日志 | journald（见第五节）或 `/var/log/infusion-monitor/*.log` | 服务用户 | 结构化 JSON 行，飞检与排障都靠它 |

```bash
sudo useradd --system --home /opt/infusion-volume-monitor --shell /usr/sbin/nologin infusion
sudo mkdir -p /opt/infusion-volume-monitor /var/lib/infusion-monitor/frames
sudo chown -R infusion:infusion /var/lib/infusion-monitor
sudo chmod 750 /var/lib/infusion-monitor
```

**权限给 750 就够**，不要为了省事 `chmod 777`：帧目录里是病区巡视短片，
审计表里能看出「几点谁确认了哪一袋换液」。可写的只有服务用户。

---

## 三、源码部署（推荐，便于改配置与打补丁）

```bash
cd /opt/infusion-volume-monitor
sudo -u infusion python3 -m venv .venv
sudo -u infusion .venv/bin/pip install -r requirements.txt
```

> `opencv-python-headless` 在 slim 镜像里偶发缺 `libGL.so.1`。源码装到最小化的
> Linux 上若报这个错，装系统包 `libglib2.0-0 libgl1` 即可。

---

## 四、环境变量清单

复制 `.env.example` 为 `.env` 后按院内实际填写。**7 组凭据位缺一不可**
（服务 fail fast，见第一节）。

| 组 | 变量 | 必填 | 说明 |
|----|------|------|------|
| 出站·护理执行平台 | `NURSE_EXEC_BASE_URL` | ✅ | 基址。服务在其后拼 `/api/v1/infusion/events` |
| 出站·护理执行平台 | `NURSE_EXEC_API_TOKEN` | ✅ | Bearer 凭据。只出现在请求头，不入库、不进日志 |
| 遥测·MQTT | `MQTT_HOST` `MQTT_PORT` `MQTT_USERNAME` `MQTT_PASSWORD` | ✅ | 泵速遥测入站。**`MQTT_ENABLED` 默认 false**：不接泵网关就保持 false，四项仍需填（配置面按组校验） |
| 遥测·MQTT | `MQTT_TOPIC_PREFIX` | | 默认 `hospital/infusion`，订阅 `<前缀>/+/telemetry` |
| 模型网关 | `LLM_BASE_URL` `LLM_API_KEY` `LLM_MODEL` `LLM_TIMEOUT` | ✅ | OpenAI-compatible；换院内自建端点只改这四个 |
| 分割实现 | `MODEL_IMPL` `MODEL_ID` | | 默认 `geom_kernel` / `geom_kernel_base`；**换实现只改这两项**，业务代码不动 |
| 分割实现 | `MODEL_CONFIG_PATH` `MODEL_CHECKPOINT` | | 留空 = 用注册表登记的；权重档缺权重 → **启动即失败**（不静默退回几何核） |
| 分割实现 | `MODEL_REGISTRY_ROOT` | | 默认 `./models`；配置与权重的相对路径按它解析 |
| 分割实现 | `MODEL_REGISTRY_JSON` | | 按 `model_id` 覆盖/追加注册表（试点、多环境切换） |
| 存储 | `DATABASE_URL` | | 默认 `sqlite:///./var/infusion_monitor.db`；换 PostgreSQL 只改这一项 |
| 存储 | `FRAMES_ROOT` | | 默认 `./var/frames`；分片续传的半成品在它下面的 `_staging/` |
| 保留期 | `RETENTION_SESSION_DAYS` | | 默认 `30.0`；输液架帧目录保留多久。到点由 `scripts/purge_expired_sessions.py` 清理，**读数留在库里** |
| 保留期 | `RETENTION_INFERENCE_AUDIT_DAYS` | | 默认 `90.0`；模型调用元数据的保留天数。审计链（飞检留档）任何保留期都不清 |
| 保留期 | `RETENTION_ARCHIVE_ROOT` | | 留空 = 到期直接删除；配了就按月分目录搬进归档区而不是删掉 |
| 续传 | `UPLOAD_RESUME_RETENTION_HOURS` | | 默认 `24.0`；开着没提交的分片上传保留多久，到点由清理接口回收 |
| 续传 | `UPLOAD_RESUME_MAX_PENDING` | | 默认 `50`；未提交的分片上传数上限，超了新的开启请求被拒并指路清理接口 |
| 判定参数 | `CHANGE_FILL_RATIO_THRESHOLD` `CHANGE_CONSECUTIVE_FRAMES` `CHANGE_MIN_INTERVAL_MINUTES` `CHANGE_MAX_PICKS_PER_ROUND` | | 默认 0.12 / 5 帧 / 15 分钟 / 3 袋 |
| 拍摄质检 | `QUALITY_SAMPLE_FRAMES` `QUALITY_MIN_HIT_RATIO` `QUALITY_GLARE_RATIO_THRESHOLD` `QUALITY_LOW_LIGHT_MEAN_LUMA` `QUALITY_SHARPNESS_MIN` `QUALITY_TILT_ANGLE_DEG` `QUALITY_DRIP_TAIL_RATIO` | | 默认 5 / 0.34 / 0.0015 / 46.0 / 150.0 / 8.0 / 0.55。**上传前预检与袋级分类共用这一组**，只影响「要不要建议补拍」，不影响读数 |
| 拍摄质检 | `MODEL_SATURATION_MIN` `PRECHECK_MIN_SHORT_SIDE` `PRECHECK_DEGRADED_SCORE` | | 默认 96 / 480 / 0.70。`PRECHECK_MIN_SHORT_SIDE` 是**质量**下限（硬下限仍是 320，低于它直接 422），`PRECHECK_DEGRADED_SCORE` 是五项子分的总分门禁 |
| 日志 | `LOG_LEVEL` `LOG_FORMAT` | | 默认 `INFO` / `json`；排障时 `LOG_FORMAT=human LOG_LEVEL=DEBUG` |

```bash
sudo cp .env.example /etc/infusion-monitor.env
sudo chmod 640 /etc/infusion-monitor.env
sudo chown root:infusion /etc/infusion-monitor.env
sudo -u infusion vi /etc/infusion-monitor.env     # 填院内真实值
```

### 换分割实现（几何核 → 生产权重）

1. 把权重与推理配置放到 `${MODEL_REGISTRY_ROOT}`（默认 `./models`），权限保证进程可读；
2. `python scripts/check_config.py` —— 人读输出里会列出整张注册表和当前档的配置/权重在不在，
   不用等服务起才知道；
3. 改 `.env`：`MODEL_IMPL=weights`、`MODEL_ID=weights_base`（`MODEL_CONFIG_PATH` /
   `MODEL_CHECKPOINT` 留空即用注册表登记的那份）；
4. 重启。**权重缺失时服务会拒绝启动并点名缺哪个变量、期望路径在哪**——这是刻意的：
   悄悄退回几何核看着「还能用」，但同一档 `MODEL_ID` 在两台机上会给出不同的毫升数与 ETA。

凭据文件放 `/etc` 而不是代码目录：代码目录可能被复制、被打包、被别的科室拿去参考，
`/etc` 下的 640 文件不会。仓库里**永远**只有 `.env.example`，没有任何真实值。

---

## 五、启动前自检（先跑它，再谈启动）

```bash
cd /opt/infusion-volume-monitor
sudo -u infusion env $(grep -v '^#' /etc/infusion-monitor.env | xargs) \
  .venv/bin/python scripts/check_config.py
```

退出码 **0 = 齐全**，1 = 缺项。缺项时它会逐组点名：

```
[缺失] 输液泵遥测入站（MQTT）      MQTT_HOST、MQTT_PORT、MQTT_USERNAME、MQTT_PASSWORD
       示例：MQTT_HOST=… / MQTT_PORT=1883 / MQTT_USERNAME=… / MQTT_PASSWORD=…
```

它顺带检查帧目录与 SQLite 目录**能不能写**——这两处配错在病区最常见，
症状是「上传到一半 500」，很难往配置上想。`--json` 供装机脚本判断。

---

## 六、开机自启（systemd）

`/etc/systemd/system/infusion-monitor.service`：

```ini
[Unit]
Description=输液余量监测系统（病区侧）
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=infusion
Group=infusion
WorkingDirectory=/opt/infusion-volume-monitor
EnvironmentFile=/etc/infusion-monitor.env
Environment=PYTHONPATH=/opt/infusion-volume-monitor/src
Environment=FRAMES_ROOT=/var/lib/infusion-monitor/frames
Environment=DATABASE_URL=sqlite:////var/lib/infusion-monitor/infusion_monitor.db
ExecStartPre=/opt/infusion-volume-monitor/.venv/bin/python scripts/check_config.py
ExecStart=/opt/infusion-volume-monitor/.venv/bin/python -m infusion_monitor.server
Restart=on-failure
RestartSec=5
# 只给自己要写的两个目录写权限，其余一律只读
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/var/lib/infusion-monitor
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

`ExecStartPre` 指向自检命令是个刻意的安排：配置写错时**根本不会**反复重启刷屏，
systemd 会直接在 journal 里留下「缺哪一项」，值班的人看日志就能改。

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now infusion-monitor
sudo systemctl status infusion-monitor
journalctl -u infusion-monitor -f
```

### Docker 部署

```bash
docker build -t infusion-volume-monitor:local .
docker run -d --name infusion-monitor \
  --restart unless-stopped \
  --env-file /etc/infusion-monitor.env \
  -e FRAMES_ROOT=/data/frames \
  -e DATABASE_URL=sqlite:////data/infusion_monitor.db \
  -p 8000:8000 \
  -v /var/lib/infusion-monitor:/data \
  infusion-monitor:local
```

容器内是 uid 10001 的非 root 用户，`/data` 的属主要对上：

```bash
sudo chown -R 10001:10001 /var/lib/infusion-monitor
```

---

## 七、启动后确认这三件事

```bash
curl -sS http://127.0.0.1:8000/api/v1/health | python3 -m json.tool   # 能不能用
curl -sS http://127.0.0.1:8000/api/v1/summary | python3 -m json.tool  # 现在什么状况
curl -sS http://127.0.0.1:8000/docs                                  # 接口契约（OpenAPI 3.1）
```

| health 字段 | 期望 | 不对时查什么 |
|-------------|------|-------------|
| `schema_version` | `3` | 版本不符说明库是旧的，见第八节 |
| `frames_root_writable` | `true` | 目录权限，见第二节 |
| `session_count` | 首次为 `0` | 有数但查不到会话，先跑一次真实上传 |

`/api/v1/summary` 是值班的日常入口：待换液袋数 `sessions.needs_change_count`、
没送出去的事件 `writeback.pending_writeback_count`、模型降级次数
`model.threshold_fallback_count`、遥测是否在流 `channels.telemetry.state`。

---

## 八、日常运维

| 场景 | 怎么做 |
|------|--------|
| 查某一次请求整条链 | 记下响应头 `X-Request-Id`，`journalctl -u infusion-monitor \| grep <id>` |
| 看有没有事件没送出去 | `GET /api/v1/summary` → `writeback.pending_writeback_count`；对端恢复后重放 |
| 换袋型标定版本后核对毫升数 | `GET /api/v1/calibrations/recalc`；历史会话按新版本重算，轨迹可查 |
| 导出审计包给护理部/内审 | `GET /api/v1/infusion/audit`（CSV，按 rack/床号/时间窗过滤） |
| 磁盘快满 | 帧目录按会话累积；保留期与容量测算见 task 44，`docs/` 届时有脚本 |
| 改配置 | 改 `/etc/infusion-monitor.env` → 重跑 `check_config.py` → `systemctl restart` |

### 错误码速查

完整表见 [`docs/error-codes.md`](docs/error-codes.md)（由代码生成，与 `/docs` 的 OpenAPI 描述一致）。
两类问题分开看：**API 错误码**是调用方拿到了 HTTP 错误响应；
**运行期故障**是请求成功返回但服务内部某条链路已不可用（模型连不上、MQTT 断连、
出站积压），这些只在 `/api/v1/summary` 与启动日志里出现。

### SQLite 的并发边界

病区侧单机部署下 SQLite 的写入是毫秒级，**一个病区一个进程**完全够用。
出现下列任一情况就该规划换 PostgreSQL（只改 `DATABASE_URL`，仓储层不感知方言）：

- 需要多台机器同时写同一个库（跨病区集中部署）；
- 并发上传数持续超过约 20 QPS；
- 会话表超过约 50 万行且保留期清理跟不上。

换库要做的三件事：改 `DATABASE_URL`（`postgresql://user:password@host:5432/infusion`）、
装一次驱动（`pip install 'psycopg[binary]'`）、把 `store/schema.py` 里的 SQLite 方言
DDL 翻译一份（`AUTOINCREMENT` 等）——**方言知识只住在连接层与 DDL 层**，仓储层与
迁移层零方言分支。细节见 [`docs/storage.md`](docs/storage.md)。

### 帧目录的保留期与定时清理

输液架帧目录是像素库，一天几百 MB 起步。按默认口径（单病区 12 轮/天 × 300 帧/轮
× 156 KiB/帧），**保留 30 天约占 16 GiB，不做保留期一年约 195 GiB**。所以要挂一条
定时清理：

```bash
# 先看会清掉什么（不动文件、不改库）
sudo -u infusion python3 /opt/infusion-volume-monitor/scripts/purge_expired_sessions.py --dry-run

# 每天凌晨 3 点 17 分清一次
cat >/etc/cron.d/infusion-monitor-retention <<'EOF'
17 3 * * * cd /opt/infusion-volume-monitor && /usr/bin/python3 scripts/purge_expired_sessions.py --json >> /var/log/infusion-monitor/retention.log 2>&1
EOF
sudo chmod 644 /etc/cron.d/infusion-monitor-retention
```

**清的是像素，不是读数**：面积比曲线、毫升数、ETA、换液结论都留在会话行里，护士长
复盘靠的是读数。会话还没结束、或换液事件还没送达对端的，帧目录一个都不动；审计链
（飞检留档）任何保留期都不碰。

**凭据缺失不挡清理**：这个任务只碰数据库与帧目录，不发请求、不调模型——模型网关还没
配好时磁盘照样该收。

申请磁盘前先算一遍：

```bash
python3 scripts/estimate_storage.py --wards 3 --frames-root /var/lib/infusion-monitor/frames
```

`--frames-root` 会量盘上真实帧的单帧字节（默认口径是渲染器实测下界，真照片更大）。

---

## 九、卸载

```bash
sudo systemctl disable --now infusion-monitor
sudo rm -f /etc/systemd/system/infusion-monitor.service
sudo systemctl daemon-reload
# 数据目录留着要问信息科——里面有审计留档，飞检期内不能删
sudo rm -rf /opt/infusion-volume-monitor
```

`.env` 与 `/var/lib/infusion-monitor` **不要顺手删**：前者是凭据，后者是审计与帧。