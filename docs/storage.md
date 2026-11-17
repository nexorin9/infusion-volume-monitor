# 存储演进：保留期、归档与容量测算

> 面向信息科与运维：帧目录会长大、库会长大、什么时候该换 PostgreSQL，这一份说清楚。

## 一、为什么要管

| 东西 | 增长方式 | 一年后（单病区口径，见第四章测算） |
|------|----------|------|
| **帧目录** | 每轮巡视留一段 10–20 秒短片，一个输液架一个目录 | 约 195 GiB（不做保留期） |
| **会话表 `bag_session`** | 每袋一行，含两段面积序列 JSON | 约 8,760 行 / 病区 |
| **审计链 `audit_log`** | 每轮每袋若干行（认领、点名、确认、写回） | 约 5 万行 / 病区 |

帧目录是**像素**，会话行是**读数**。这两者的生命周期不一样，所以本产品的保留策略只
清前者。

## 二、保留期策略

### 清什么

| 对象 | 保留期 | 配置项 | 到了怎么办 |
|------|--------|--------|------------|
| 输液架帧目录 | 30 天 | `RETENTION_SESSION_DAYS` | 删除，或搬进 `RETENTION_ARCHIVE_ROOT` 归档 |
| 模型调用元数据 `inference_audit` | 90 天 | `RETENTION_INFERENCE_AUDIT_DAYS` | 删除 |
| 审计链 `audit_log` | **不清** | — | 飞检留档，append-only，任何保留期都不动 |
| 出站事件 / 待写回队列 | **不清** | — | 业务记录与欠账台账，由重放流程收口 |
| 会话读数（面积比、毫升数、ETA、换液结论） | **不清** | — | 帧没了读数还在 |

### 不清什么（**三条硬纪律**）

1. **会话还没结束的**（`monitoring` / `needs_change`）：护士还指着它核对余量，
   点名刚算出来的那几袋里就有它。
2. **换液事件还没送达对端的**（出站状态 `pending` / `pending_writeback` / `failed`，
   或待写回队列里还挂着 `queued` / `dead_letter`）：信息科重放那笔欠账时还要靠原始帧核对。
3. **审计链**：飞检要回答「当时发生了什么」，靠的就是它。

**同一段短片的几只袋子共用一个帧目录**（`frames_root/{架号}_{输注单号}_{轮次}`），
所以清理的单位是**目录**：一个目录里只要还有一只袋子没到条件，整个目录都留着。
被拦下的原因逐袋记在结果里（`session_not_finished` / `writeback_unsettled` /
`writeback_queue_pending` / `frames_shared_with_active_session` / `no_event` /
`within_retention` / `bad_timestamp` / `no_frames_dir`）。

### 怎么跑

```bash
# 先看会清掉什么（不动任何文件、不改库）
python3 scripts/purge_expired_sessions.py --dry-run

# 按配置清理（cron：每天凌晨 3 点跑一次）
python3 scripts/purge_expired_sessions.py

# 保留期改成 7 天，并把过期帧搬进归档区而不是删掉
python3 scripts/purge_expired_sessions.py --retention-days 7 \
    --archive-root /var/lib/infusion-monitor/archive
```

cron 条目（`/etc/cron.d/infusion-monitor-retention`）：

```cron
17 3 * * * cd /opt/infusion-volume-monitor && /usr/bin/python3 scripts/purge_expired_sessions.py --json >> /var/log/infusion-monitor/retention.log 2>&1
```

**凭据缺失不挡清理**：保留期任务只碰数据库与帧目录，不发请求、不调模型。护理部的
模型网关还没配好时，磁盘照样该收——脚本会退回环境变量与默认值，而不是因为少一个
`LLM_API_KEY` 就整条 cron 失败。

### 归档 vs 删除

| 模式 | 配置 | 结果 |
|------|------|------|
| 直接删除（默认） | `RETENTION_ARCHIVE_ROOT` 留空 | 帧目录删除；会话行的 `frames_retention_state=purged`，`frames_dir` 保留历史位置 |
| 搬进归档区 | `RETENTION_ARCHIVE_ROOT=/var/lib/infusion-monitor/archive` | 帧按月分目录搬到 `archive/YYYY-MM/{目录名}/`；会话行的 `frames_dir` 指向新位置，`frames_retention_state=archived` |

两种模式都在 `retention_archive` 表留一行台账（哪一袋、什么时候、为什么、多少字节、
搬到哪了），并在审计链记 `session.frames_purged` / `session.frames_archived`。
**没有这张台账就答不出「这袋的帧为什么突然没了」**。

### 落盘顺序

先动文件、再写库。写库失败时下一轮会把这些会话记成 `absent`（目录本来就不在），
像素不会丢。反过来先写库再删文件，删失败就再也没人管那堆像素。

## 三、容量测算

```bash
python3 scripts/estimate_storage.py                      # 默认规模
python3 scripts/estimate_storage.py --wards 3 --beds-per-ward 40
python3 scripts/estimate_storage.py --frames-root /var/lib/infusion-monitor/frames  # 用真实帧实测
python3 scripts/estimate_storage.py --database /var/lib/infusion-monitor/infusion_monitor.db
```

### 数字从哪来

| 参数 | 默认值 | 依据 |
|------|--------|------|
| 单帧字节 | 159,744 B（156 KiB） | **本产品渲染器在规范分辨率（长边 1280px、JPEG q88）下的实测值**。这是**下界**——合成画面是平色，真手机照片纹理多，同分辨率下通常更大 |
| 每轮帧数 | 300 | 10 秒 × 30 帧/秒的手机录像。终端做抽样上传时按实际帧数改（`--frames-per-round 24`） |
| 每轮袋数 | 2 | 一架输液架挂 1–3 袋常见 |
| 每病区日均巡视轮次 | 12 | 输液巡视 1–2 小时一轮 |
| 每床每天换液次数 | 3 | 500 ml 袋按 60 ml/h 约 8 小时一袋 |

**实测优先**：`--frames-root` 量盘上真实帧的平均字节，`--database` 量真实库文件与行数
反推每行字节——两个都是实测，不是估的，脚本会把口径标出来。

### 默认规模的结论（单病区）

| 项 | 数 |
|----|----|
| 每天上传 | 12 轮 / 24 袋 / 3,600 帧 |
| 每天新增像素 | 548.4 MiB |
| 保留 30 天的帧占用 | 16.1 GiB |
| 不做保留期（一年） | 195.5 GiB |
| 库每天增长 | 595.2 KiB |
| 库一年增长 | 212.2 MiB |

**结论**：帧目录是磁盘的主项（比库大两个数量级），所以保留期策略针对的是帧；
库本身在单机部署下毫无压力。

## 四、SQLite 的并发边界与换 PostgreSQL 的判据

### 单机部署下 SQLite 够用

病区侧一个进程一个库，写入是毫秒级。写操作用 `BEGIN IMMEDIATE` 显式串行化
（先拿写锁再读，避免两路确认同时通过检查后双写）。

### 换库的判据

出现下列任一情况就该规划换 PostgreSQL：

- 需要**多台机器同时写同一个库**（跨病区集中部署）；
- 并发上传数持续超过约 **20 QPS**；
- 会话表超过约 **50 万行**且保留期清理跟不上；
- 需要按时间分区 / 分库 / 只读副本（SQLite 没有这些）。

`python3 scripts/estimate_storage.py` 每次都会按当前规模把第一条判据算出来。

### 换库要做什么

| 做什么 | 说明 |
|--------|------|
| 改 `DATABASE_URL` | `postgresql://user:password@host:5432/infusion` |
| 装驱动 | `pip install 'psycopg[binary]'`（装一次，产品代码不动） |
| 翻译 DDL | **表结构目前是 SQLite 方言**（`AUTOINCREMENT` 等），见下节 |

**不认识的连接串一律启动即报错**：把拼错的 PostgreSQL 连接串当成 SQLite 用，会让
服务在错误的库上安静地写数据——比启动失败难查得多。

## 五、方言知识住在哪（架构约定）

| 层 | 文件 | 方言知识 |
|----|------|----------|
| 连接串解析 | `store/dsn.py` | **有**：识别 scheme、拆主机/库名/用户、掩码密码 |
| 连接与事务 | `store/db.py` | **有**：选驱动、`BEGIN IMMEDIATE`、行工厂 |
| 表结构 DDL | `store/schema.py` | **有**：`AUTOINCREMENT` 等 SQLite 方言 |
| 迁移 | `store/migrations.py` | **无**：只用标准 SQL（`ALTER TABLE ADD COLUMN`、`ON CONFLICT`） |
| 仓储 | `store/repositories.py` | **无**：只发标准 SELECT/INSERT/UPDATE/DELETE |

后两层「无方言」有源码断言钉住（`tests/test_storage_dsn.py`）：仓储层与迁移层里
不得出现 `BEGIN IMMEDIATE` / `PRAGMA` / `sqlite_master` / `INSERT OR REPLACE` /
`AUTOINCREMENT`；`AUTOINCREMENT` 只允许出现在 `schema.py` 一处——换库时 DDL 翻译
只改这一处，其余模块不动。

**本轮未实连**：真实 PostgreSQL 实例本轮没有连过（本机装了 `psycopg` 但没有可用的
PG 服务）。验证只到「连接串解析正确、密码不外泄、指向 PG 时不静默退回 SQLite、
仓储层零方言分支」这一层；DDL 在 PG 上能不能跑通，需要在真库上验一次。
