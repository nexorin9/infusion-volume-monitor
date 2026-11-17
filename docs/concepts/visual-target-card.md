# 视觉目标卡（输液余量监测系统）

> 交付形态已锁定为**无界面 HTTP API 服务**（常驻 uvicorn + OpenAPI 3.1 `/docs`），
> 因此本卡的「画面」是 **API 契约面 / 终端 curl 会话 / 服务运行日志**，不是 Web 页面。
> 本环境不具备图像生成能力，故只保留本结构化卡，不放占位图。

## 画面类型

1. **服务运行日志**（常驻进程）
2. **终端 curl 会话**（上传 → 曲线 → 确认换液 → 写回）
3. **OpenAPI 3.1 契约页**（`/docs`，FastAPI 自带渲染）

## 三个状态

### 1. 起始态

一个还没被调用过的服务。屏幕上只有 `uvicorn` 启动后的滚动日志行：

```
INFO:     Application startup complete.
INFO:     loading predictor impl=geom_kernel model_id=geom_kernel_base
INFO:     schema migration ok (bag_session / infusion_event / inference_audit)
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     mqtt subscribed topic=hospital/infusion/+/telemetry
INFO:     waiting for upload on POST /api/v1/infusion/tracks
```

没有任何图形界面，只有一串诚实的日志。

### 2. 首个可用瞬间

终端里敲一条 `curl -F frames=@rack_01.zip -F rack_id=… -F infusion_order_no=…`，
回显一串 JSON：

```json
{"rack_id":"R-07A","bag_seq":"A03","frames":36,
 "area_ratio_series":[0.71,0.70,0.69,0.68,0.68,0.67,0.66,0.65,0.64,0.64,0.63,0.62],
 "estimated_volume_ml":118,"eta_minutes":26,
 "calibration_state":"calibrated","status":"monitoring","quality_flags":[]}
```

**一袋的余量曲线第一次出现在可复制的文本里**，护士不用去翻记录、不用靠口述交班。

### 3. 成形态

同一条查询端点再敲一次，回显里 `status` 变成 `needs_change`，`trigger_reason`
点名原因；护士确认换液后 `POST …/transition` 回显 `{"idempotent_hit":false}`；
服务日志同步多出两行结构化事件（`event=infusion.changed` / `event=infusion.replaced`，
含 `bag_seq`、`latency_ms`、`decision_source`）。这是「叫它一声，它认得这是哪一袋，
并把这袋的事写回了护理执行平台」的完整画面。

## 视觉语言

| 维度 | 取值 |
|------|------|
| 终端底色 | `#1b1f23` |
| JSON 字段名 | `#8ab4f8` |
| JSON 字符串值 | `#a5d6a7` |
| JSON 数字 | `#fdd663` |
| 警告 / 告警 | `#ef9a9a` |
| 日志 INFO / WARNING / ERROR | `#9e9e9e` / `#fdd663` / `#ef9a9a` |
| 字体族 | 终端等宽 `SF Mono` / `JetBrains Mono` / `Menlo`；JSON 缩进 2 空格 |
| 信息层级 | ① 每个 `bag_seq` 行的 `status` 与 `estimated_volume_ml` ② `area_ratio_series`（默认折叠，不用肉眼读） ③ `quality_flags` / `warnings` |
| 留白密度 | JSON 字段间不空行；每个 `bag_seq` 块之间空一行，让「一袋一件事」视觉上成立 |

## 必须可达的约束

- 依赖面：**Python 3.10 + FastAPI + uvicorn + Pydantic v2 + numpy/OpenCV + httpx + paho-mqtt**。
  不做 React 前端、不做 WebGL、不做视频播放器。
- `/docs` 由 FastAPI 自带 OpenAPI 渲染，不额外造轮子。
- 契约测试用 `pytest` + `httpx` / `TestClient`，产出物是**测试结果文本**，不是页面。
- **不画**任何模型效果展示页、mask 叠加预览图或交互式标注界面——那需要额外前端，
  超出本轮锁定形态；掩膜只以 `area_ratio_series` 数值序列出现在 JSON 里。

## 与实测证据的差距（诚实标注）

`docs/screenshots/` 中的实测证据 = **真实 curl 会话回显 + 真实服务日志片段 + 真实契约测试输出**。
目标侧重「一眼看懂的摘要」，实测侧重「可复制的完整 JSON」；实测截图里不会出现颜色标注，
只有纯文本终端与 JSON 回显。掩膜以数值序列呈现，不含任何像素级叠加效果。