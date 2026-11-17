# 输出样例（全量）

README 只放速查片段；这里放完整的脱敏响应与字段说明，供对接方逐字段核对。

> 全部样例来自**本地合成短片**（`scripts/make_sample_track.py` 渲染），不是真实院内短片。
> 样例标识一律用占位（`R-07` / `B01` / `TYPE-A` / `ORDER-0001`），不含可被误认为真实的
> 住院号、处方号或日期。

---

## 1. 上传响应（每袋余量曲线与毫升数）

```bash
curl -X POST http://127.0.0.1:8000/api/v1/infusion/tracks \
  -F rack_id=R-07 \
  -F infusion_order_no=ORDER-0001 \
  -F bed_no=12 \
  -F captured_at=2026-10-22T08:30:00+08:00 \
  -F 'slots=[{"bag_seq":"B01","slot_index":0,"bag_type_code":"TYPE-A","rate_text":"5ml/h"},
             {"bag_seq":"B02","slot_index":1,"bag_type_code":"TYPE-A","rate_text":"常规输液"}]' \
  -F frames=@rack_01.zip
```

```json
{"upload_key":"R-07_2026-10-22_ORDER-0001-ab12cd34",
 "rack_id":"R-07","bed_no":"12","frame_count":24,
 "first_frame":"frame_00000.jpg","last_frame":"frame_00023.jpg",
 "frame_time_source":"zip_member",
 "bags":[
   {"bag_seq":"B01","slot_index":0,"bag_type_code":"TYPE-A",
    "calibration_state":"calibrated","calibration_version":1,
    "fill_ratio":0.4393,"estimated_volume_ml":273.2,
    "eta_minutes":3279,"eta_source":"order_text","rate_ml_per_h":5.0,
    "pickup_suppressed":false,"near_empty":false,"overfilled":false,
    "area_ratio_series":[0.5890,0.5861,...],"visibility_states":["visible","visible",...]},
   {"bag_seq":"B02","slot_index":1,"bag_type_code":"TYPE-A",
    "calibration_state":"calibrated","calibration_version":1,
    "fill_ratio":0.3512,"estimated_volume_ml":217.3,
    "eta_source":"unavailable",
    "pickup_suppressed":true,"suppression_reason":"eta_unavailable",
    "area_ratio_series":[0.5891,0.5812,...],"visibility_states":["visible","visible",...]}
 ]}
```

逐字段对照：

| 字段 | 含义 | 什么时候不出现 |
|------|------|--------------|
| `fill_ratio` | 末次可见帧的液体占袋腔面积比（**观测值**） | 整段看不见（`visibility_states` 全为 occluded/lost） |
| `estimated_volume_ml` | 按袋型标定换算出的袋内毫升（保留 0.1 ml） | 未标定 / 超装 / 无观测 |
| `calibration_version` | 这个毫升数用的是该袋型第几版标定 | 未标定 |
| `eta_minutes` | 按滴速预计耗尽的分钟数（向上取整） | 滴速给不出来 |
| `eta_source` | `order_text` / `mqtt_telemetry` / `unavailable` | — |
| `pickup_suppressed` | true = 依据不足以点名，护士端只展示不催换 | — |
| `suppression_reason` | `uncalibrated` / `eta_unavailable` / `no_observation` / `overfilled` | 未抑制时不出现 |

上例中 B02 的 `rate_text` 写的是「常规输液」（解析不出 ml/h）且没有泵速遥测，
所以**毫升数照给**（217.3）但**不给时点**——这正是设计意图：护士至少知道还剩多少，
但不会被一个编出来的时点催着换袋。

---

## 2. 袋型标定：把面积比换成毫升

面积比本身换不出毫升：同一袋型不同批次的袋腔形状、弯月面挂壁、管路残留都不一样。
换算关系由**标定样例**登记——「量到的面积比」与「已知的真实毫升数」成对登记，
拟合成分段线性曲线（袋上那几道刻度带本身就是分段线性的）。

### 登记请求

```bash
curl -X POST http://127.0.0.1:8000/api/v1/infusion/calibrations \
  -H 'Content-Type: application/json' \
  -d '{"bag_type_code":"TYPE-A","capacity_ml":600.0,"dead_volume_ml":20.0,
       "meniscus_offset_ratio":0.0,
       "samples":[{"measured_fill_ratio":0.10,"known_volume_ml":58.0,"source":"称重法"},
                  {"measured_fill_ratio":0.45,"known_volume_ml":280.0,"source":"称重法"},
                  {"measured_fill_ratio":0.75,"known_volume_ml":440.0,"source":"换液确认反推"},
                  {"measured_fill_ratio":0.95,"known_volume_ml":540.0,"source":"称重法"}],
       "calibrated_by":"OP-NURSE-01","note":"按袋上刻度带登记"}'
```

### 拟合结果

```json
{"calibration_id":"cal_330cc479b47e441a","bag_type_code":"TYPE-A","version":1,
 "capacity_ml":600.0,"usable_capacity_ml":580.0,"dead_volume_ml":20.0,
 "meniscus_offset_ratio":0.0,"sample_count":4,"max_relative_error":0.0,
 "calibrated_by":"OP-NURSE-01","active":true,"recalculated_sessions":2,
 "bands":[[0.0,0.0],[0.1,58.0],[0.45,280.0],[0.75,440.0],[0.95,540.0],[1.0,580.0]]}
```

| 字段 | 含义 |
|------|------|
| `capacity_ml` | 标签标称容量（满袋） |
| `dead_volume_ml` | 管路与滴斗残留（袋内测不到的部分） |
| `usable_capacity_ml` | 有效容量 = 标称 − 残留 |
| `meniscus_offset_ratio` | 弯月面挂壁补偿：液面爬在袋壁上，量到的面积比高于真实高度比 |
| `bands` | 拟合出的分段线性带节点 `[[面积比, 袋内毫升], ...]`，两端补齐到 `[0,1]` |
| `max_relative_error` | 样例最大相对偏差；超阈值需标定人复核，**不静默拉直** |

`recalculated_sessions` = 2 表示：登记之前已有 2 个会话是 `uncalibrated`（只有面积比、
没有毫升数），登记后自动按新曲线补上了毫升数。

### 改版后的重算轨迹

```bash
curl 'http://127.0.0.1:8000/api/v1/infusion/calibrations/recalc?bag_type_code=TYPE-A'
```

| bag_seq | 版本 | 改前 | 改后 | 差值 | 原因码 |
|---------|------|------|------|------|--------|
| B01 | 无 → v1 | 补录前没有数 | 273.2 ml | — | `calibration_registered` |
| B02 | 无 → v1 | 补录前没有数 | 217.3 ml | — | `calibration_registered` |
| B01 | v1 → v2 | 273.2 ml | 292.8 ml | +19.5 | `calibration_version_changed` |
| B02 | v1 → v2 | 217.3 ml | 232.8 ml | +15.5 | `calibration_version_changed` |

两条原因码的区别是有意为之：

- `calibration_registered` —— **首次登记**，此前没有生效版本。改前那一格**留空而不是填 0**：
  改前没有数，不等于有 0 毫升。
- `calibration_version_changed` —— **改版**，此前有生效版本。`delta_ml` 是改前改后的差。

已换袋（`replaced`）的会话**不追溯修改**：护理部核对的是「当时的数对不对」，
不是「现在该是多少」。

---

## 3. 首帧候选列表（护士认领用）

```json
{"candidates":[{"candidate_id":"C01","bbox":[49,266,191,573],"area":43585,
                "stability_score":0.9667,"slot_index":0,"bound":true},
               {"candidate_id":"C02","bbox":[289,266,431,573],"area":43812,
                "stability_score":0.9999,"slot_index":1,"bound":true}]}
```

> 稳定分低的候选已被剔除，不进列表；同一槽位出现多个候选时 `bound=false`，需护士认领。

---

## 4. 遮挡与恢复（逐帧可见度）

被单或手臂挡在镜头前时，系统**不丢弃**这一帧，而是把可见度标出来：

| 状态 | 画面上发生什么 | 护士端怎么显示 | 面积比 |
|------|--------------|--------------|--------|
| `visible` | 整袋看得见 | 正常计入曲线 | 照常给 |
| `partial` | 只看到部分袋体（下段被压住） | 提示「这段没拍全，建议换个角度」 | 照常给，但标为读数偏低 |
| `occluded` | 完全看不见 | 提示「这段被挡了」 | 留空，**不插值** |
| `reappeared` | 遮挡后重新接回同一袋 | 与正常曲线连起来看 | 照常给 |
| `lost` | 连续遮挡超过丢失窗口 | 提示「这袋需要重新认领」 | 后续留空，轨迹交出 |

`visibility_states` 与 `area_ratio_series` 一一对齐：某袋在第 5–8 帧被挡时，
对应的 `area_ratio_series` 这四个位置是 `null`（**不插值、不沿用上一帧数值**），
第 9 帧记 `reappeared` 接回同一条曲线。遮挡超过丢失窗口时响应里会多一个
`track_handoff`（含最后可见帧、位置与参考余量），交给再获取流程认领；
**没交出时该字段整体不出现**，而不是返回 `null`。
---

## 5. 护士确认换液后的出站写回

护士在终端点「已换液」后，本服务 `POST {NURSE_EXEC_BASE_URL}/api/v1/infusion/events`，
带 `Authorization: Bearer $NURSE_EXEC_API_TOKEN` 与 `Idempotency-Key`。下面是对端
（本地替身）真实收到的请求：

```http
POST /api/v1/infusion/events HTTP/1.1
Authorization: Bearer <院内发放的 token>
Content-Type: application/json
Idempotency-Key: infusion-sess_<会话ID>-B01-replaced
```

```json
{"event_id":"evt_<16位>","client_event_id":"infusion-sess_<会话ID>-B01-replaced",
 "event_type":"infusion.bag.changed","session_id":"sess_<16位>","rack_id":"R-EXAMPLE-11",
 "bed_no":"EXAMPLE-BED-11","infusion_order_no":"<EXAMPLE_ORDER>","bag_seq":"B01",
 "fill_ratio":0.43934,"trigger_reason":"nurse_confirmed","decision_source":"nurse_confirmed",
 "operator_id":"N-EXAMPLE-11","occurred_at":"2026-10-22T21:58:34+08:00"}
```

注意 `estimated_volume_ml` 与 `eta_minutes` **不在请求体里**：这一袋未登记袋型标定，
门禁不通过时字段整体不出现，而不是发 `null` 让对端自己猜。补登记标定后同一袋重传，
这两格才会有值。

### 写回结果的三种落点

| 情形 | `writeback_status` | 说明 |
|------|--------------------|------|
| 对端 2xx | `sent` | 已送达；重放同一次确认不会再发第二次 |
| 对端 5xx / 超时（退避重试 ≤2 次后） | `pending_writeback` | 事件行保留、`next_retry_at` 给出可重放时点，等信息科重放 |
| 对端 4xx（业务拒绝） | `failed` | 不重试，`last_error` 留对端返回的拒绝原因 |
| 会话未被护士确认 | `pending`（不变） | **一次 HTTP 都不发**，只留一行 `writeback.refused` 审计 |

失败摘要示例（已脱敏，token 值不会出现）：

```json
{"writeback_status":"pending_writeback","ok":false,"attempts":3,"kind":"http_5xx",
 "last_error":"http_5xx: HTTP 503 {\"accepted\": true}","next_retry_at":"2026-10-22T21:59:12+08:00",
 "attempt_kinds":["http_5xx","http_5xx","http_5xx"],"backoff_delays":[1.0,4.0]}
```

字段规范与响应约定的完整表格见 `docs/nurse-exec-contract.md`（该文件由代码里的出站
字段表生成，与实现同源）。

---

## 6. 泵速遥测入站（MQTT）

订阅 `hospital/infusion/+/telemetry`，载荷为 JSON：

```json
{"bag_seq":"B01","pump_rate_ml_per_h":40.0,
 "device_sn":"PUMP-EXAMPLE-001","ts":"2026-10-22T17:41:00+08:00"}
```

输液架号取自**主题**的架号段（`+`），不从载荷猜——同一架最多 6 袋，只靠 `bag_seq`
定位不到会话。坏消息（缺字段、滴速非正、JSON 解析失败）只拒这一条并计数，不会把整条
订阅通道带掉；断线重连后 broker 重投同一份遥测按 `(架, 袋, ts)` 去重，不会二次消费。
遥测缺失时 ETA 退回输注医嘱速率，**视觉主路径不受影响**。

---

## 7. 滴速来源与冲突（ETA 引擎）

滴速有两个来源：**输注医嘱文本**（第一优先）与**泵速遥测**（医嘱认不出才退到它）。
医嘱文本的常见写法与换算：

| 医嘱原文 | 解析出的滴速 | 说明 |
|----------|--------------|------|
| `5ml/h` | 5 ml/h | 直接写明 |
| `250毫升/小时` | 250 ml/h | 单位在前 |
| `40 毫升每小时` | 40 ml/h | 「每」字写法 |
| `每小时 250 毫升` | 250 ml/h | 数值在单位后 |
| `20滴/分` | 60 ml/h | 按默认滴系数 20 滴/ml 换算 |
| `20滴/分，15滴/ml` | 80 ml/h | 医嘱写明滴系数，按写的算 |
| `5ml/kg/h` | **认不出** | 按体重给的速率缺体重，不猜 |
| `静脉推注 20ml` | **认不出** | 推注没有「还能撑多久」 |
| `500ml` | **认不出** | 体积不是速率 |

两个来源**一致**（相对偏差 ≤ 20%）→ 按医嘱给时点：

```json
{"eta_minutes":911,"eta_source":"order_text","rate_ml_per_h":40.0,
 "rate_reason":"医嘱 40 ml/h 与泵速遥测 40 ml/h 一致（相对偏差 0.0%，门限 20%），按医嘱给时点"}
```

两个来源**对不上**（相对偏差 > 20%）→ 判冲突：**不给时点、不取中间值**，
把两个值连同偏差交出来，并把该袋的换液点名压住：

```json
{"estimated_volume_ml":607.3,"eta_source":"conflict",
 "rate_reason":"医嘱 40 ml/h 与泵速遥测 20 ml/h 相对偏差 50.0%，超过 20% 门限——两个来源对不上，不取中间值，请人工确认",
 "rate_conflict":{"order_rate_ml_per_h":40.0,"telemetry_rate_ml_per_h":20.0,
                  "relative_deviation":0.5,"tolerance":0.2},
 "pickup_suppressed":true,"suppression_reason":"rate_conflict"}
```

> 响应里**没有** `eta_minutes`、**没有** `rate_ml_per_h`（不是 null 占位，是字段整体
> 不出现）；库里 `eta_minutes` 为 NULL、`rate_conflict_json` 留着冲突证据，
> `suppression_reason="rate_conflict"` 让换液点名判定跳过这一袋。
---

## 8. 体积不确定度与置信门禁

标定齐全时，每个毫升数都配一条 95% 区间与置信度。**置信度够才出数**——不够时
`estimated_volume_ml` 与 `eta_minutes` 两个键整体不出现（不是 null），但
`volume_uncertainty` 一定在场，否则护士端只会看到「没数」，无从决定下一步做什么。

### 8.1 放行：常规液位

```bash
curl -sS -X POST "$BASE/api/v1/infusion/tracks" \
  -F rack_id=R-EXAMPLE-28 -F infusion_order_no=EXAMPLE-ORDER-0028 \
  -F bed_no=B-EXAMPLE-28 -F captured_at=2026-01-01T09:41:00+08:00 \
  -F 'slots=[{"bag_seq":"B01","slot_index":0,"bag_type_code":"TYPE-A","rate_text":"40ml/h"}]' \
  -F frames=@rack_60.zip | python3 -m json.tool | head -40
```

```json
{
  "calibration_state": "calibrated",
  "estimated_volume_ml": 278.2,
  "eta_minutes": 418,
  "volume_uncertainty": {
    "confidence": 0.9313,
    "relative_sigma": 0.024,
    "low_ml": 265.1,
    "high_ml": 291.3,
    "gate": "ok",
    "reason": "",
    "hint_zh": "毫升数在可信区间内，可按余量安排换液。",
    "factors": {
      "estimation_mode": "band",
      "model_source": "band_fit_residual",
      "residual_std_after_drain_fit": 0.00469,
      "sigma_area_ratio": 0.01105,
      "sensitivity_ml_per_ratio": 533.33,
      "sigma_sensitivity_rel": 0.01638,
      "occlusion_ratio": 0.0
    }
  }
}
```

`factors` 逐项可核对：`residual_std_after_drain_fit` 是可信序列对降幅做最小二乘后的
残差（**不是**整段标准差——整段标准差量到的是「漏了多少」）；`sensitivity_ml_per_ratio`
是该液位处 `dV/dA`；除以毫升数就是 `sigma_sensitivity_rel`。

### 8.2 挡下：小液位（成因 `relative_uncertainty_too_high`）

```json
{
  "calibration_state": "low_confidence",
  "fill_ratio": 0.0412,
  "pickup_suppressed": true,
  "suppression_reason": "low_confidence",
  "volume_uncertainty": {
    "confidence": 0.0,
    "relative_sigma": 0.35,
    "low_ml": 3.8,
    "high_ml": 20.6,
    "gate": "low_confidence",
    "reason": "relative_uncertainty_too_high",
    "hint_zh": "余量太小或这一段抖动太大，算出来的毫升数落在一段挺宽的范围里，够不够换液还说不准。等液面再低一些、或把手机端稳了重拍一次。"
  }
}
```

门限不是拍上去的常量：等价于「相对不确定度 > 0.35 × (1 − 0.60) = 0.14」，而相对
不确定度 = `灵敏度 × σ_fill ÷ 毫升数`。600 ml 袋、有效容量 580 ml、σ_fill ≈ 0.010 时，
分界落在**余量约 41 ml**：20% 液位（85 ml）放行，5%（29 ml）挡下。

### 8.3 挡下：遮挡占比过高（成因 `occluded_too_much`）

```bash
python3 scripts/make_sample_track.py --out occluded.zip --frames 24 --bags 1 --fill 0.6 \
  --occlude 0:3:6 --occlude 0:10:13 --occlude 0:17:20
```

三段各 3 帧：每段都不超过丢失窗口（轨迹还接得回来），合计 9/24 = 0.375 已过高遮挡门限。

```json
{
  "calibration_state": "high_occlusion",
  "occluded_frame_count": 9,
  "frame_count": 24,
  "suppression_reason": "high_occlusion",
  "volume_uncertainty": {
    "gate": "high_occlusion",
    "reason": "occluded_too_much",
    "hint_zh": "这一段有 38% 的时间被挡着，剩下的毫升是外推出来的。等挡开以后对准输液架再拍 10–20 秒就好。"
  }
}
```

> 整段长时间被挡（超过丢失窗口）会先触发 `lost` → `no_observation`，那是更早的一条门禁，
> 不走这条。中间态（接得上但靠外推）才是 `high_occlusion` 要治的。

### 8.4 挡下：几何版本缺失（成因 `geometry_version_missing`）

腔体外形参数登记过、但没有版本号（历史 `cavity_json` 里没这个键）时，无法判断历史
读数是否还成立：

```json
{
  "calibration_state": "low_confidence",
  "suppression_reason": "low_confidence",
  "volume_uncertainty": {
    "gate": "low_confidence",
    "reason": "geometry_version_missing",
    "factors": {"geometry_revision": 0},
    "hint_zh": "这个袋型的外形参数还没登记版本号，改过以后就对不上历史读数了。先让信息科补登记版本号，毫升数会自己回来。"
  }
}
```

登记入口会分配版本号（首次登记即第 1 版）：重新登记一遍带 `geometry_revision` 的外形参数，
历史会话由改版重算一并恢复出数（同一套门限，观测证据从会话行复原，不因改版凭空放宽区间）。

### 8.5 落库形态

门禁挡下时库里**明确没有数**（`estimated_volume_ml IS NULL`、`eta_minutes IS NULL`），
但证据留档：

```sql
SELECT estimated_volume_ml, eta_minutes, calibration_state, suppression_reason,
       volume_confidence, volume_low_ml, volume_high_ml, uncertainty_gate
FROM bag_session WHERE session_id = '<会话 ID>';
--   NULL | NULL | low_confidence | low_confidence | 0.0 | 3.8 | 20.6 | low_confidence
```

逐项证据（含护士端文案）在 `uncertainty_json` 里，append-only 的 `calibration_recalc_log`
在改版重算时补上前后两版的区间，供护理部回答「这袋当时为什么报 180 毫升」。

---

## 9. 换袋后怎么认出「这不是上一轮那一袋」（重锚打分）

同一输液架的第二轮巡视：第一轮 B01 在挂点 0、药液偏蓝、液位 0.62；第二轮同一挂点上
换成偏橙的新袋、几乎是满的（对端还没推进袋序）。

```json
{
  "bag_seq": "B01",
  "reanchor_state": "unbound",
  "reanchor_reason": "appearance_changed",
  "reanchor_confidence": 0.479999,
  "reanchor_runner_up": 0.317647,
  "reanchor_driver": "colour",
  "descriptor_mode": "full",
  "candidate_state": "needs_claim"
}
```

### 9.1 逐项打分明细（内部 `CandidateScore`，响应与审计都按这套口径）

| 项 | 权重 | 同一袋跨轮 | 同挂点换新袋 | 判读 |
|----|------|-----------|------------|------|
| `geometry` 横向中心 / 袋底 / 宽度 | 0.34 | 1.000 | 1.000 | 位置没变，**位置对得上不构成接回的理由** |
| `colour` 色相直方图交集 | 0.30 | 1.000 | 0.000 | 换药液 → 颜色分布完全不像 |
| `level` 液柱高度 | 0.22 | 1.000 | 0.000（**一票否决**） | 同一袋只会变矮；变高 = 换了新袋 |
| `label` 标签区有无 / 面积 / 质心 | 0.14 | 1.000 | 1.000 | 判不出时**退出加权**，不当 0 分 |
| **总分** | — | **0.999** | **0.480** | 门限 0.62 |

### 9.2 落库与审计形态

```sql
SELECT descriptor_json, reanchor_state, reanchor_reason, previous_session_id
FROM bag_session WHERE bag_seq = 'B01' ORDER BY captured_at DESC LIMIT 1;
-- descriptor_json 含 liquid_top_y / liquid_height / liquid_fill_fraction /
--                 label_available / label_present / label_coverage / label_ink / descriptor_mode
-- reanchor_state=unbound | reanchor_reason=appearance_changed | previous_session_id=NULL

SELECT detail FROM audit_log WHERE action = 'reanchor.resolved' ORDER BY rowid DESC LIMIT 1;
-- {"reason":"appearance_changed","anchor_state":"unbound","confidence":0.479999,
--  "runner_up":0.317647,"driver":"colour","descriptor_mode":"full",
--  "scores":[["B01",0.479999,1.0],["B02",0.317647,0.86]]}
```

认准时 `previous_session_id` 指向上一轮会话（曲线续在同一条上）；认不准时它是 `NULL`
——响应里那个字段**整体不出现**，上层拿不到指针就不会把新袋接到旧曲线上。

## 审计包：CSV 明细与 manifest 首页

`GET /api/v1/infusion/audit?rack_id=R-EXAMPLE-07&occurred_from=2026-10-22T00:00:00+08:00&occurred_to=2026-10-22T23:59:59+08:00`

响应头：

```
content-type: text/csv; charset=utf-8
content-disposition: attachment; filename=audit-package.csv; filename*=UTF-8''%E5%AE%A1%E8%AE%A1%E5%8C%85_R-EXAMPLE-07_20261023.csv
x-audit-row-count: 5
x-audit-chain-digest: bb68fba4a48c2d15c3f0a1e2b7d84c69
x-audit-scope: R-EXAMPLE-07
```

响应体（`utf-8-sig` 读取，首列的 BOM 会被 Excel 与 `utf-8-sig` 吃掉）：

```
审计序号,发生时刻,操作人,动作类别,动作码,结果,对象,输液架,床号,袋序,袋型,输注单号|袋序,原状态,新状态,请求号,载荷摘要,明细(JSON)
1,2026-10-22T09:41:12+08:00,service:reanchor,其他动作,reanchor.resolved,accepted,sess_a1b2,R-EXAMPLE-07,EXAMPLE-BED-07,B01,TYPE-A,EXAMPLE-ORDER-0007|B01,,,req-7a,3f2a91c0d4e5b678,"{""anchor_state"":""bound"",""confidence"":0.9905}"
2,2026-10-22T11:20:03+08:00,service:change-detector,换液点名,state.change_proposed,accepted,sess_a1b2,R-EXAMPLE-07,EXAMPLE-BED-07,B01,TYPE-A,EXAMPLE-ORDER-0007|B01,monitoring,needs_change,req-7b,91c7aa40be2312ff,"{""margin_kind"":""volume"",""margin_value"":48.0,""trigger_reason"":""eta_within_window""}"
3,2026-10-22T11:25:40+08:00,nurse:N-07,候选认领,candidate.claimed,accepted,sess_a1b2,R-EXAMPLE-07,EXAMPLE-BED-07,B01,TYPE-A,EXAMPLE-ORDER-0007|B01,,,req-7c,77bd0e1a9c33f4d2,"{""bag_seq"":""B01"",""candidate_id"":""C01""}"
4,2026-10-22T11:30:15+08:00,nurse:N-07,护士确认,state.confirmed,accepted,sess_a1b2,R-EXAMPLE-07,EXAMPLE-BED-07,B01,TYPE-A,EXAMPLE-ORDER-0007|B01,needs_change,replaced,req-7d,5ab2c9d04e771f33,"{""client_event_id"":""infusion-sess_a1b2-B01-replaced"",""occurred_at"":""2026-10-22T11:30:15+08:00""}"
5,2026-10-22T11:30:16+08:00,service:writeback,写回成功,writeback.sent,accepted,sess_a1b2,R-EXAMPLE-07,EXAMPLE-BED-07,B01,TYPE-A,EXAMPLE-ORDER-0007|B01,,,req-7d,be10f4c827a5d391,"{""attempt_count"":1,""client_event_id"":""infusion-sess_a1b2-B01-replaced"",""writeback_status"":""sent""}"
```

同一批行的 manifest：

```json
{
  "generated_at": "2026-10-23T09:12:44+08:00",
  "scope_zh": "输液架 R-EXAMPLE-07；时间窗 2026-10-22T00:00:00+08:00 ~ 2026-10-22T23:59:59+08:00；最多 5000 行",
  "scope_key": "R-EXAMPLE-07",
  "row_count": 5,
  "window": {"from": "2026-10-22T09:41:12+08:00", "to": "2026-10-22T11:30:16+08:00"},
  "by_category": {
    "claim": 1, "calibration": 0, "pickup": 1, "confirmation": 2,
    "writeback_sent": 1, "writeback_failed": 0, "writeback_pending": 0, "other": 1
  },
  "by_action": {
    "candidate.claimed": 1, "reanchor.resolved": 1, "state.change_proposed": 1,
    "state.confirmed": 1, "writeback.sent": 1
  },
  "chain_digest": "bb68fba4a48c2d15c3f0a1e2b7d84c69",
  "digest_algorithm": "rolling-sha256(occurred_at,audit_id) 前 32 位",
  "filters": {"rack_id": "R-EXAMPLE-07", "occurred_from": "2026-10-22T00:00:00+08:00",
              "occurred_to": "2026-10-22T23:59:59+08:00", "limit": 5000}
}
```

**七类动作 ↔ 飞检的问法**（`category` 取值即导出的筛选词）：

| 类别键 | 中文名 | 回答的问句 |
|--------|--------|------------|
| `claim` | 候选认领 | 「这袋是谁认的？」 |
| `calibration` | 标定确认 | 「袋型标定是谁定的、定的是哪一版？」 |
| `pickup` | 换液点名 | 「系统点了谁、为什么没点其余的？」 |
| `confirmation` | 护士确认 | 「谁在几点确认换了哪一袋？」 |
| `writeback_sent` | 写回成功 | 「对端真的收到了吗？」 |
| `writeback_failed` | 写回失败 | 「有没有该发没发出去的？」 |
| `writeback_pending` | 写回挂起 | 「哪几条还压在队列里等重放？」 |
| `other` | 其他动作 | 「重锚、遥测、分片续传等」——归到这里而不是丢掉 |

**按架导出看不到 `calibration` 行**，这是对的：袋型标定是病区级记录，一个 `TYPE-A`
版本对全病区每一架都生效，硬给它安一个架号等于替它编一个不存在的归属。要查标定留痕，
去掉 `rack_id` 用全量导出。

## 10. 上传前帧质量预检（`precheck` 段）

整段短片一份结论，护士端据此决定要不要顶补拍提示条。下列 JSON 取自一次真 uvicorn + curl 的实跑
（`var/verify_task34.py`，合成手持抖动短片 20 帧、480×640、两袋，`blur=5`）：

```json
{
  "precheck": {
    "state": "degraded",
    "quality_score": 0.713175078790141,
    "degraded": true,
    "sampled_frames": 5,
    "sub_scores": {
      "sharpness": 0.06331508659023707,
      "low_light": 1.0,
      "glare": 1.0,
      "resolution": 1.0,
      "tilt": 0.9709027640653487
    },
    "primary": {
      "code": "motion_blur", "severity": "warning", "label_zh": "运动模糊",
      "detail_zh": "有 5/5 帧画面发糊（清晰度 9.6968，门限 150.0），袋口和液面边界化开了。",
      "suggestion_zh": "把手机靠在输液架上或用手肘夹住，稳住 10–20 秒再松开重拍。",
      "hit_frames": 5, "sampled_frames": 5, "hit_ratio": 1.0,
      "evidence": {"threshold": 150.0, "sharpness": 9.6968}
    },
    "flags": ["…同上，同一段只有这一条命中…"],
    "frames": [
      {"frame_index": 0,  "width": 480, "height": 640, "short_side": 480,
       "luma_mean": 87.71, "dark_ratio": 0.486, "glare_ratio": 0.0, "sharpness": 9.7,  "tilt_deg": 0.25},
      {"frame_index": 5,  "width": 480, "height": 640, "short_side": 480,
       "luma_mean": 89.7,  "dark_ratio": 0.486, "glare_ratio": 0.0, "sharpness": 9.44, "tilt_deg": 0.16},
      {"frame_index": 10, "width": 480, "height": 640, "short_side": 480,
       "luma_mean": 91.66, "dark_ratio": 0.486, "glare_ratio": 0.0, "sharpness": 9.66, "tilt_deg": 0.27},
      {"frame_index": 14, "width": 480, "height": 640, "short_side": 480,
       "luma_mean": 93.28, "dark_ratio": 0.486, "glare_ratio": 0.0, "sharpness": 9.47, "tilt_deg": 0.22},
      {"frame_index": 19, "width": 480, "height": 640, "short_side": 480,
       "luma_mean": 95.27, "dark_ratio": 0.486, "glare_ratio": 0.0, "sharpness": 9.22, "tilt_deg": 0.26}
    ]
  },
  "bags": [
    {"bag_seq": "B01", "precheck_score": 0.713175078790141, "precheck_degraded": true,
     "capture_degraded": true,
     "capture_quality": ["…袋级质检同样命中 motion_blur，措辞与上面逐字相同…"]}
  ]
}
```

干净短片（同场景、未加抖动）：`state="ok"`、`quality_score=1.0`、`degraded=false`、`flags=[]`，
`sub_scores` 五项全为 `1.0`——采样帧 `sharpness` 实测 923–932、`tilt_deg` 实测 `0.0`。

落库形态（`bag_session` 每一只袋子一行，逐袋各存一份同值）：

```sql
SELECT bag_seq, precheck_score, precheck_degraded, precheck_flags
  FROM bag_session
 WHERE infusion_order_no = 'EXAMPLE-ORDER-0034'
 ORDER BY bag_seq;
-- B01 | 0.713175078790141 | 1 | [{"code":"motion_blur", ..., "suggestion_zh":"把手机靠在输液架上…"}]
-- B02 | 0.713175078790141 | 1 | （同上，逐字一致）
```

### 10.1 反向优化：哪种拍法最常失败

质量分与 flag 逐袋落库，护理部想知道「我们科最常拍坏的是什么」时，一条 SQL 就够
（SQLite/PostgreSQL 均可）：

```sql
SELECT json_extract(value, '$.code')        AS problem,
       COUNT(*)                             AS hits,
       ROUND(AVG(precheck_score), 3)        AS avg_score,
       COUNT(DISTINCT rack_id)              AS racks
  FROM bag_session,
       json_each(bag_session.precheck_flags)
 WHERE precheck_degraded = 1
 GROUP BY problem
 ORDER BY hits DESC;
```

拿到的结果有两种用法：

* **改门限**：某类常年零命中 → 该门限对这批病区偏严，护士白被提示条打扰，收紧或抬高；
* **改文案**：某类命中最多 → 把它的 `suggestion_zh` 拿到科务会上读一遍，多半比调门限更管用
  （护士照不照做取决于那句话好不好懂）。

两类结论都可直接回写到 `QUALITY_*` / `PRECHECK_*` 环境变量，无需改代码。

## 11. 长输出样例归档（自 README 外移）

README 只保留每个主题两三行摘录，**完整请求/响应在此**。
上面的 §1–§10 是按业务主题索引的样例，本节按能力分组给出更长的完整片段，
含内部字段与边界取值，用于排障与对接核对。

> 本节内容全部是**真实跑出来的输出**（真 uvicorn + 真 curl + 真 SQLite / 真替身对端），
> 标识符一律脱敏（`<EXAMPLE_*>`、`N-EXAMPLE-*`、日期用样例值）。

### 袋型腔体几何：按袋子的真实形状换毫升

登记袋型时带上外形参数（超椭圆横截面 + 透视收窄 + 软袋鼓胀 + 袋壁挂液），毫升数就改由
腔体几何算：把画面上的投影面积**反解**回袋子里的真实高度，再按未变形腔体积分——
面积比乘容量的朴素做法会把袋型之间的系统性偏差一起带进来。

```bash
curl -X POST http://127.0.0.1:8000/api/v1/infusion/calibrations \
  -H 'Content-Type: application/json' \
  -d '{"bag_type_code":"TYPE-A","capacity_ml":500.0,"dead_volume_ml":20.0,
       "cavity":{"capacity_ml":500.0,"cavity_height_mm":210.0,"shape_exponent":4.0,
                 "taper":0.20,"bulge":0.055,"film_holdup_ml":0.0,
                 "label_zh":"500 ml 软袋（圆角方袋）"},
       "calibrated_by":"OP-NURSE-01"}'
# → {"version":1,"estimation_mode":"cavity","usable_capacity_ml":480.0,
#    "cavity":{"shape_exponent":4.0,"taper":0.2,"bulge":0.055,...}}
```

之后每袋的读数多三个字段，说明这袋的毫升是怎么来的：

```json
{"bag_seq":"B01","estimated_volume_ml":171.6,"eta_minutes":2059,
 "estimation_mode":"cavity","cavity_height_fraction":0.350171,"resident_ml":0.0}
```

**低液位的残留被显式扣掉。** 袋壁挂液与管路/滴斗残留永远流不出来，量级不大却正好落在
「要不要换袋」这个最要紧的区段：读数扣到只剩残留量级时，报 `0` 并标
`residual_limited=true`，而不是给一个「还剩 1.7 ml」这种谁也核对不了的数字。

### 自校准：拿护士确认过的「这袋输完了」反推残留

护士点一次「已换液」，就等于替这个袋型确认了一条残留证据：那一刻真实可读余量是 0，
而系统当时按模型给了一个数。样本按袋型攒起来取中位数（不取平均——护士也常因医嘱变更
提前换袋，那是离群点），**攒够 3 条且彼此不打架才生效**：

```bash
curl -sS 'http://127.0.0.1:8000/api/v1/infusion/calibrations/deviations?bag_type_code=TYPE-A'
# → {"count":3,"cavity_registered":true,
#    "learned_residual":{"residual_ml":7.2,"effective_residual_ml":7.2,"sample_count":3,
#                        "state":"learned","applied":true,"spread_ml":0.3},
#    "deviations":[{"session_id":"sess-…","last_fill_ratio":0.05,"model_volume_ml":7.2,
#                   "residual_ml":7.2,"eligible":1,
#                   "ineligible_reason":null,"recorded_at":"2026-10-23T09:41:00+08:00"}]}
```

| 结论状态 | 什么时候 | 系统怎么做 |
|---------|---------|-----------|
| `no_evidence` | 还没有「换液前已经很低」的确认 | 照常出数，不修正 |
| `insufficient_samples` | 有证据但不足 3 条 | 记下偏差，**不改任何读数** |
| `conflicting` | 各条反推的残差跨度超过上限 | 报出中位数供人工复核，**不生效** |
| `learned` | 证据够且一致 | 扣进读数，`resident_ml` 可见 |

半袋被换掉（医嘱变更、病人下床）不算残留证据：`ineligible_reason` 会写明
`not_near_empty`，样本照样留档——「为什么这条没参与定标」本身就是要留痕的信息。

### 标定改版：生效区间与偏差闸门

标定记录是**会改的**：换一批袋型、重新称重、护理部复核后修正，都要出新的一版。
同一袋型任何时刻只能有一个生效版本，所以每版都带一段生效区间（`effective_from` /
`effective_to`，左闭右开）。改版即交接：旧版在新区间起点收口，新版接上。

```bash
# 提前把新袋型的标定录进去，但指定下周一零点才生效 —— 当天读数一点不变
curl -sS -X POST http://127.0.0.1:8000/api/v1/infusion/calibrations \
  -H 'Content-Type: application/json' -d '{
    "bag_type_code":"TYPE-A","capacity_ml":650.0,
    "samples":[{"measured_fill_ratio":0.35,"known_volume_ml":240.0}],
    "calibrated_by":"OP-NURSE-01","effective_from":"2026-11-01T00:00:00+08:00"}'
# → {"version":2,"version_status":"scheduled","superseded_version":1,
#    "recalculated_sessions":0, "review_threshold":0.15}
#   生效时刻排到旧版之前（回填历史）→ 409，点名冲突的是第几版
```

改版会让历史会话按新版本重算。但**差得太多时不自动覆盖**：新旧读数相对差超过
`CALIBRATION_REVIEW_DEVIATION`（默认 15%）时，旧读数原样保留、会话挂上待复核、
差异落一条 append-only 告警。差 15% 以上这个量级，多半是新版本自己有问题，
而不是「旧版本终于被修正」——直接推到全院读数上，代价比晚几天修正大得多。

```bash
curl -sS 'http://127.0.0.1:8000/api/v1/infusion/calibrations/TYPE-A/reviews'
# → {"count":1,"pending_count":1,"reviews":[{"alert_id":7,"session_id":"sess-…",
#    "previous_volume_ml":425.9,"proposed_volume_ml":552.1,"delta_ml":126.2,
#    "relative_deviation":0.2286,"threshold":0.15,"severity":"review",
#    "resolution":"pending"}]}

# 护理部复核：applied = 通过，新读数落库；dismissed = 否决，旧读数继续沿用
curl -sS -X POST http://127.0.0.1:8000/api/v1/infusion/calibrations/TYPE-A/reviews/7 \
  -H 'Content-Type: application/json' \
  -d '{"resolution":"applied","reviewed_by":"OP-NURSE-02","note":"复称确认新批次袋腔更大"}'
```

版本与生效区间随时可查（信息科与护理部核对用），到点的预约版本在查询时自动转正：

```bash
curl -sS 'http://127.0.0.1:8000/api/v1/infusion/calibrations/TYPE-A/versions'
# → {"active_version":3,"scheduled_versions":[],"pending_reviews":0,"review_threshold":0.15,
#    "versions":[{"version":1,"status":"expired",
#                 "effective_from":"2026-10-22T08:00:00+08:00",
#                 "effective_to":"2026-10-22T15:04:00+08:00",
#                 "calibrated_by":"OP-NURSE-01"}, …]}
```

### 曲线平滑：抖动、尖峰与缺帧（原始值一个字节都不改）

一次上传里，每袋除了原始面积比序列，还带一条**平滑后的工作序列**和一条**受限重建序列**。
三条逐帧对齐，长度恒等于帧数——任何一帧都不会被删掉或移位：

| 字段 | 是什么 | 谁读它 |
|------|--------|--------|
| `area_ratio_series` | 几何核测到的**原始值**，一个字节都不改 | 核对用：出问题时先看它 |
| `area_ratio_smoothed` | 中值 + 一阶平滑后的工作序列；被判反常的帧**保留原值** | 展示与诊断 |
| `area_ratio_reconstructed` | 只有「前后端点都是可信观测、且缺帧 2–10 帧」时才有值，其余 `null` | 看这段的下降速度，不做余量读数 |
| `smoothing` | 尖峰帧 / 反常帧 / 重建帧 / 每段缺帧的处理原因 / 平均降幅 | 护士端提示「这段拍糊了」的依据 |

```json
{"bag_seq":"B01","frame_count":18,
 "area_ratio_series":[0.5669,0.565,0.5632,0.5595,0.5577,0.3828,0.5172,0.5521,0.5503],
 "area_ratio_smoothed":[0.565,0.565,0.5639,0.5613,0.5591,0.5549,0.5521,0.5488,0.5464],
 "area_ratio_reconstructed":[null,null,null,null,null,null,null,null,null],
 "smoothing":{"spike_frames":[5,6],"outlier_frames":[],"reconstructed_frames":[],
   "missing_frame_count":0,"drain_ratio_per_frame":0.0022,
   "latest_raw":0.5503,"latest_trusted":0.5464,"degraded":true,
   "gaps":[{"start":5,"end":6,"length":2,"reason":"reconstructed_from_endpoints","filled":false}]}}
```

> 上例第 6 帧那块镜面高光在液面上挖了个洞，测出来的面积比从 0.5577 掉到 0.3828（-0.17）。
> **原始序列照实留着这个深坑**，可信序列只走了 -0.0041，继续贴着真实缓降走——这一帧
> 若进了毫升换算，就是一次「快见底了」的误报。反过来，液面**升**上去的帧不会这样被吸收：
> 输液袋只会降，升上去只可能是拍坏了或中间换过袋，这类帧进 `outlier_frames` 保留原值、
> 退到最近一次可信读数。

被挡住的帧三条序列里都是 `null`，**不插值**；恢复之后才按前后两次可信读数补一小段，
补了哪几帧写在 `area_ratio_reconstructed` 里，六种「不补」的原因也都写在 `smoothing.gaps[].reason`：
端点不全（`no_trusted_front_endpoint` / `no_trusted_back_endpoint`）、缺得太少
（`gap_shorter_than_min`）、缺得太多（`gap_longer_than_max`）、两端液面反而升高（`endpoint_order_invalid`）。

### 拍不好时：给的是一句能照做的补拍动作，不是报错

每袋多两个字段：`capture_degraded`（这段拍坏了吗）与 `capture_quality`（问题码 + 中文建议）。
**拍坏了照样返回 `201`**，曲线与毫升数照给——被拒收的只有帧数、分辨率、帧序这些硬指标。

```json
{"bag_seq":"B01","capture_degraded":true,
 "capture_quality":[
   {"code":"motion_blur","severity":"warning","label_zh":"运动模糊",
    "detail_zh":"有 5/5 帧画面发糊（清晰度 104.2，门限 150.0），袋口和液面边界化开了。",
    "suggestion_zh":"把手机靠在输液架上或用手肘夹住，稳住 10–20 秒再松开重拍。",
    "hit_frames":5,"sampled_frames":5,"hit_ratio":1.0,
    "evidence":{"sharpness":104.2,"threshold":150.0}},
   {"code":"low_light","severity":"warning","label_zh":"暗光",
    "detail_zh":"有 5/5 帧画面太暗（最暗那帧平均灰度 27.5，门限 46.0），袋身和背景分不开。",
    "suggestion_zh":"把床头灯或手电打开，离输液架近一点再重拍。",
    "hit_frames":5,"sampled_frames":5,"hit_ratio":1.0,
    "evidence":{"luma_mean":27.47,"threshold":46.0,"dark_ratio":0.528}}]}
```

| 问题码 | 中文说法 | 什么时候判 | 给护士的补拍动作 |
|--------|----------|-----------|----------------|
| `bag_out_of_frame` | 袋体出框 | 袋体离画面边 ≤3px（证据里给实际距离） | 往后退半步，让整只袋子连同挂点进画面 |
| `motion_blur` | 运动模糊 | 清晰度（拉普拉斯方差）低于门限；**暗光帧不参与** | 把手机靠在输液架上或用手肘夹住 |
| `glare_overexposed` | 反光过曝 | 近白像素占比超门限 | 换个角度避开顶灯直射，或用手／床单在上方挡光 |
| `bad_angle` | 角度不正 | 液面线拟合倾角超门限 | 手机正对输液架，让液面横平着进画面 |
| `low_light` | 暗光 | 全画面平均灰度低于门限 | 把床头灯或手电打开，离输液架近一点 |
| `empty_drip_chamber` | 滴斗空管 | 腔体底部收成细杆（**info 级**） | 不用重拍——这是已知量偏差，余量判定不受影响 |

`capture_quality` 按「**该先修哪个**」排序，护士端把**第一条**顶到提示条就够：修好出框，
后面几条多半自动消失。六类里只有 `empty_drip_chamber` 是 info 级——它是系统已知的量偏差
（袋体与滴斗连成一块、腔体被撑大、面积比偏低），不是护士拍错了，所以**不标 `degraded`、
不占用提示条、不建议重拍**。

夜里那段拍得又暗又糊时只会报 `low_light`、**不会**同时报 `motion_blur`：暗光下对比度一起掉，
清晰度量本身不可测，报「糊」只会让护士去靠手机，而她真正该做的第一件事是**把灯打开**。
灯亮之后若仍糊，下一轮质检会单独点出「运动模糊」。

> 质检只解固定 5 帧（`QUALITY_SAMPLE_FRAMES`），24 帧和 1200 帧的短片质检开销相同。

### 一轮巡视的规模控制：落盘即缩放，只喂输液架那块画面

护士手机一轮拍 10–20 秒、动辄 1200 帧，原生长边常常 4000px。响应顶层多两个段，
答「这一段被怎么处理过」——信息科排查「为什么这段没缩」「到底裁没裁」时看这两个段：

```json
{"geometry": {
  "camera_width": 1920, "camera_height": 1280,
  "canonical_width": 1280, "canonical_height": 853,
  "scale": 0.666667, "resized": true, "reason": "long_edge_capped",
  "resized_frames": 16, "total_frames": 16,
  "scale_mismatch_with_previous_round": false},
 "roi": {
  "enabled": true, "applied": true,
  "window": {"x": 0, "y": 226, "width": 596, "height": 254},
  "applied_calls": 64, "advances": 0, "max_step_px": 0, "border_touches": 0,
  "anchor_width": 596, "anchor_height": 254, "fallback_reason": null}}
```

| 字段 | 答什么 |
|------|--------|
| `geometry.camera_*` / `canonical_*` | 护士手机原始尺寸 ↔ 服务实际分析的尺寸；护士端把候选框画回原照片靠 `scale` 换算 |
| `geometry.reason` | `long_edge_capped` 按长边缩到位 / `short_edge_floor_wins` 长宽比太极端保短边 / `short_edge_below_floor` 短边本就太小原样放行 / `disabled` 关掉了缩放 |
| `roi.window` | 实际裁剪窗（规范帧坐标）；`fallback_reason` 非空时整幅处理，理由如实报 |
| `roi.advances` / `border_touches` | 窗口动过几次、有几次是掩膜顶到窗边被迫扩窗——都是排查「曲线突然跳一下」的入口 |

两条不变量，测试逐条钉住：

| 场景 | 表现 | 为什么这样 |
|------|------|------------|
| 长边 4000px 的短片 | 落盘即缩到 1280，`resized_frames` 报告缩了几帧 | 面积比、余量、ETA 全是**比例量**，按长边缩后数值一字不变，像素数少一个数量级 |
| 长宽比极端（4000×300） | **一帧都不缩**（`short_edge_below_floor`） | 按长边缩到 1280 会把短边压到 96px，比缩之前还不可用；宁可不缩，交给校验/预检去判 |
| 帧序列本来就合规 | **一个字节都不重写** | 过一趟缩放流水线不该掉画质 |
| 邻床、输液杆也是液体色 | 只把输液架那块画面喂给预测器，掩膜照旧贴回整幅规范帧 | ROI 少喂的是**画面**，不是**坐标**——面积比、候选框、跨会话重锚全在同一个坐标系里 |
| 一架六只袋、逐只传播 | 窗口**只涨不缩**，窗内观测完全不动窗 | 跟着当前这只袋子重定位会让窗口在几只袋子之间摆，每只都被切一刀；液面下降时若跟着液体掩膜收窄，袋子上半截（标定要量的腔体上沿）会被裁掉，毫升数随之失真（实测末值 0.5252 被压成 0.2063） |
| 认不出归属的槽位退回槽位先验 | 初值窗口取**全部**候选框，不只取认领上的那几个 | 窗开大只多算几个像素；开小则出事——窗外那只袋子只剩贴着窗边 4px 的一条，几何核会锁到那条缝上 |
| 1200 帧一整段 | 峰值内存与单帧耗时**不随帧数线性膨胀** | 边收边落盘、解包流式写；这是病区那台服务器能不能跑得动的硬指标 |
| 大包传到一半断网 | 超字节 `413` / 超时限 `408`，两条都指向分片续传 | 病区网络断在 90% 是常态，不是异常；已收到的分片留在暂存目录，`GET /api/v1/infusion/uploads/{id}` 能查到断点 |

```bash
# 开关对照：同一条命令改一个环境变量，两条路给出的曲线应当逐点相同
ROI_ENABLED=true  uvicorn infusion_monitor.api:app --port 8731 &   # 裁剪喂画面
ROI_ENABLED=false uvicorn infusion_monitor.api:app --port 8732 &   # 整幅喂画面
# 实测：两只袋子 ×16/24 帧，area_ratio_series 与 fill_ratio 逐点一致
```

### 多袋并行：每条曲线落在哪个挂点上

一个输液架最多挂 6 只袋子。上传一次，整架同时跟踪；响应里每袋除了余量曲线，还带一段
「这条曲线是不是真的在输注单说的那个挂点上」的核对结论：

| 字段 | 含义 |
|------|------|
| `slot_state` | `assigned` 对得上 / `mismatched` 停在别的挂点那一列 / `ambiguous` 分不出先后 / `no_evidence` 整段没看清 / `below_score` 三通道没过门槛 / `unfilled` 这个挂点没拍到 / `extra_track` 输注单上没有的那只 |
| `slot_assigned_index` | 实际判定的挂点序号（与 `slot_index` 不一致即袋序错位） |
| `slot_score` / `slot_margin` | 挂点分配总分；与次优的分差过小即判歧义 |
| `slot_detail` | 一句给护士看的中文说明：现在是什么情况、下一步点哪里 |

```json
{"bag_seq":"B02","slot_index":1,"slot_state":"no_evidence","slot_reason":"track_never_seen",
 "slot_detail":"B02 整段都没看清，不给它配挂点，请把镜头对准这一挂点重拍一段。",
 "candidate_state":"needs_claim","occluded_frame_count":18}
{"bag_seq":"B01","slot_index":0,"slot_state":"assigned","slot_assigned_index":0,
 "slot_score":0.9316,"slot_margin":null,"slot_reason":"assigned",
 "slot_detail":"第 1 个挂点上的 B01 与这条曲线对得上，直接用。","candidate_state":"bound"}
```

> **只有 `assigned` 才是可信的袋序对应**，其余状态一律 `needs_claim`，护士在候选列表里点
> 一行认领即可。`slot_margin` 为 `null` 表示这条轨迹在其它挂点上连门限都没过——没有竞争
> 对手，也就没有可比的分差。整段被挡住的袋子**不占**任何挂点：上面 `B02` 的 `slot_state`
> 是 `no_evidence`，第 2 个挂点单独报「这段没拍到」，`B01`/`B03` 仍各自 `assigned`、读数
> 不受牵连。

### 断点续传：断点查询与分片提交

断点查询逐帧点名还缺哪些（真机实跑，24 帧的一段拍到第 12 帧时）：

```json
{
  "upload_id": "upl_4cd15c07a57d4fea",
  "status": "pending",
  "session_key": "R-EXAMPLE-32_2026-10-22_EXAMPLE-ORDER-0032-a1f142bcc8",
  "expected_frames": 24,
  "received_frames": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  "received_count": 12,
  "missing_frames": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
  "duplicate_count": 0,
  "chunk_count": 1,
  "session_ids": []
}
```

把第一片重发一次（终端超时重试）：**0 新增、12 去重**，断点不回退，磁盘上仍是先到的那一份：

```json
{
  "upload_id": "upl_4cd15c07a57d4fea",
  "accepted_frames": [],
  "duplicates_dropped": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  "duplicate_count_total": 12,
  "received_frames": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  "received_count": 12,
  "missing_frames": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
  "added_bytes": 0
}
```

提交后的响应与一次性 `/tracks` 同形（`frame_count=24`、每袋 24 个面积比采样点、
`estimated_volume_ml` 因未登记袋型而**整体不出现**）。整段重放后 `session_version`
从 1 变 2，被盖掉的那一版仍可查：

```json
{
  "session_id": "sess_5dc4a89aa2ee4ffd",
  "current_version": 2,
  "archived_count": 1,
  "versions": [
    {
      "session_id": "sess_5dc4a89aa2ee4ffd",
      "session_version": 1,
      "state": "monitoring",
      "captured_at": "2026-10-22T09:41:00+08:00",
      "frame_count": 24,
      "fill_ratio": 0.39605167409766884,
      "estimated_volume_ml": null,
      "calibration_state": "uncalibrated",
      "area_ratio_series": [0.58843, 0.57898, 0.57128, 0.56168, 0.55212, 0.54445,
                            0.53465, 0.52525, 0.51565, 0.50608, 0.49843, 0.48695,
                            0.47926, 0.46963, 0.46196, 0.45243, 0.44262, 0.43323,
                            0.42363, 0.41591, 0.4064, 0.39493, 0.38723, 0.37958],
      "archived_at": "2026-10-23T04:54:31+00:00",
      "archived_reason": "replay"
    }
  ]
}
```

### 换液点名与确认响应

```json
{"bags":[{"bag_seq":"A03","trigger_reason":"eta_within_window","eta_minutes":6,
          "confidence":0.91,"bed_no":"07-12","decision_source":"threshold_fallback"}]}
{"idempotent_hit":false,"bag_seq":"A03","status":"replaced","client_event_id":"evt-<EXAMPLE>"}
```

### 点名清单：护士站一眼扫完，按谁快输完谁排头

```bash
curl -sS "http://127.0.0.1:8800/api/v1/infusion/pickups"
```

```json
{"count":2,"bed_count":1,"generated_at":"2026-10-22T10:02:11+00:00","writeback_queued_count":0,
 "hold_reason":"未确认的 needs_change 只在清单里可见，绝不进入写回队列",
 "items":[
  {"bag_seq":"B04","bed_no":"07-12","state":"needs_change","slot_index":3,
   "trigger_reason":"eta_within_window","eta_minutes":15,"fill_ratio":0.0739,
   "estimated_volume_ml":72.4,"calibration_state":"calibrated",
   "confidence":0.6089,"confidence_source":"volume",
   "evidence_note":"按 TYPE-A 第 1 版标定换算毫升","evidence_basis":["area_ratio","calibration"],
   "client_event_id":"infusion-sess_<EXAMPLE>-B04-replaced","writeback_queued":false,
   "confirm_hint":{"session_id":"sess_<EXAMPLE>","bag_seq":"B04",
                   "client_event_id":"infusion-sess_<EXAMPLE>-B04-replaced"},
   "rank":1},
  {"bag_seq":"C01","bed_no":"07-15","state":"needs_change","slot_index":0,
   "trigger_reason":"near_empty_observed","fill_ratio":0.0091,"calibration_state":"uncalibrated",
   "confidence":0.8325,"confidence_source":"observation",
   "confidence_factors":{"visible_frames":15,"total_frames":16,"occlusion_ratio":0.0625,
                         "coverage":1.0,"low_fill_penalty":0.1363,"reason":"observed"},
   "evidence_note":"依据面积比，未标定袋型（未登记袋型），给不出毫升数；未取到滴速，不给预计耗尽时点",
   "evidence_basis":["area_ratio"],"client_event_id":"infusion-sess_<EXAMPLE>-C01-replaced",
   "writeback_queued":false,"rank":2}]}
```

两条读法：

- **C01 没有毫升数也没有 `eta_minutes`**（字段整体不出现），但它照样在清单里——
  「袋子里几乎没液体了」是直接观测，不需要标定。文案逐字写明**依据面积比，未标定袋型**，
  不拿一个编出来的毫升数糊弄，也不让护士以为那个数字不存在。
- **确认入参就摆在清单里**（`confirm_hint`）。护士端直接回填，不会因为自己编键而把
  一次换液拆成多条事件。

确认与重放（同一个确认点两次，第二次什么都不新增）：

```bash
# 原样回填 confirm_hint 里的三个字段 + 自己的工号
curl -sS -X POST http://127.0.0.1:8800/api/v1/infusion/transition \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"sess_<EXAMPLE>","bag_seq":"B04","operator_id":"N-EXAMPLE-7",
       "client_event_id":"infusion-sess_<EXAMPLE>-B04-replaced"}'
```

```json
{"session_id":"sess_<EXAMPLE>","bag_seq":"B04","state":"replaced","operator_id":"N-EXAMPLE-7",
 "client_event_id":"infusion-sess_<EXAMPLE>-B04-replaced","event_id":"evt_<EXAMPLE>",
 "created":true,"idempotent_hit":false,"confirmed_at":"2026-10-22T10:04:02+08:00",
 "pickup_evidence":{"previous_state":"needs_change","previous_trigger_reason":"eta_within_window",
                    "previous_eta_minutes":15,"previous_fill_ratio":0.0739,
                    "bed_no":"07-12","rack_id":"R-EXAMPLE-07"},
 "writeback":{"status":"sent","ok":true,"attempts":1,"kind":"ok","refused":false}}
```

> 同一个请求再发一次：`created=false`、`idempotent_hit=true`、`event_id` 与
> `confirmed_at` 与首次**完全一致**，事件表不新增行，替身对端也不收第二条请求。
> `pickup_evidence` 是确认**之前**的那份点名依据——确认会把 `trigger_reason` 覆盖成
> `nurse_confirmed`，不留一份的话事后就说不清「谁在什么时候按什么依据点的名」。

### 换液点名闸门：这一轮点了谁、为什么没点其余的

四袋都见底、但每轮上限只点 3 袋时的上传响应片段（字段均为真实返回）：

```json
{"bags":[
 {"bag_seq":"B04","eta_minutes":16,"state":"needs_change","pickup_state":"named",
  "pickup_rank":1,"trigger_reason":"eta_within_window","pickup_detail":"连续 20 帧低于 0.15，余量判定=eta"},
 {"bag_seq":"B03","eta_minutes":18,"state":"needs_change","pickup_state":"named","pickup_rank":2},
 {"bag_seq":"B02","eta_minutes":19,"state":"needs_change","pickup_state":"named","pickup_rank":3},
 {"bag_seq":"B01","eta_minutes":21,"state":"monitoring","pickup_state":"suppressed",
  "pickup_truncated":true,"pickup_rank":4,"pickup_suppress_reason":"round_cap_exceeded",
  "pickup_detail":"本轮已点名 3 袋（上限 3），B01 按 ETA 升序排在第 4 位被截断；下一轮会重新参与"}]}
```

下一轮（同一输液架、换轮次键）再传：被点过的三袋 `pickup_suppress_reason="latched"` 一条都不重复点名，
上一轮被截断的 `B01` 反而拿到名额。两轮下来 `infusion_event` 表始终是空的——**未确认不出站**。

### 出站写回事件

```json
{"event_id":"<EXAMPLE>","client_event_id":"evt-<EXAMPLE>","event_type":"infusion.bag.changed",
 "rack_id":"R-07A","bed_no":"07-12","infusion_order_no":"<EXAMPLE_ORDER>","bag_seq":"A03",
 "fill_ratio":0.39,"estimated_volume_ml":118.0,"eta_minutes":26,
 "trigger_reason":"eta_within_window","decision_source":"threshold_fallback",
 "operator_id":"N0123","occurred_at":"2026-01-01T09:12:03"}
```

> 标识均为占位示例（`<EXAMPLE_*>`），非真实住院号 / 处方号 / 患者信息。
> 终端回显与真实响应片段截图见 `docs/screenshots/`。

### 待写回清单与人工重放（欠账看得见、补得回）

```bash
# 1) 查：哪些换液事件还没送到护理执行平台
curl -s http://127.0.0.1:8000/api/v1/infusion/pending | jq '.count, .items[0]'
```

```json
// GET /api/v1/infusion/pending（脱敏示例）
{"count":1,"generated_at":"2026-01-01T10:05:12+00:00",
 "by_status":{"dead_letter":1},"dead_letter_count":1,"queued_count":0,"failed_count":0,
 "items":[{"queue_id":"wbq-<EXAMPLE>","event_id":"evt-<EXAMPLE>",
   "client_event_id":"infusion-sess_<EXAMPLE>-B01-replaced",
   "status":"dead_letter","attempt_count":3,"rack_id":"R-07A","bed_no":"07-12","bag_seq":"B01",
   "next_retry_at":"2026-01-01T09:12:19+00:00","last_error":"http_5xx: HTTP 503",
   "dead_letter_reason":"retries_exhausted","replayable":true}]}
```

```bash
# 2) 补：对端恢复后由信息科重放（只投已存在的事件，不产生第二条换液记录）
curl -s -X POST http://127.0.0.1:8000/api/v1/infusion/pending/evt-<EXAMPLE>/replay \
  -H 'Content-Type: application/json' -d '{"operator_id":"OPS-EXAMPLE"}'
```

```json
// POST /api/v1/infusion/pending/{event_id}/replay（送达后）
{"event_id":"evt-<EXAMPLE>","client_event_id":"infusion-sess_<EXAMPLE>-B01-replaced",
 "found":true,"replayed":true,"idempotent_hit":false,"writeback_status":"sent","ok":true,
 "kind":"ok","attempts":1,"queue_status":"sent","queue_attempt_count":4}
```

> 再点一次会回 `{"replayed": false, "idempotent_hit": true}`——**一次 HTTP 都不发**。
> 重放人 `operator_id` 必填，会落进审计链（`writeback.replayed`）。

### 模型判定与降级（结论从哪来）

```json
// 模型可用：采信模型结论，但状态仍不动
{"decision_source":"model_decided","usable":true,"decision":"needs_change",
 "trigger_reason":"near_empty_observed","confidence":0.86,
 "pending_manual_review":false,"fallback_reason":null,
 "model":"<EXAMPLE_MODEL>","prompt_hash":"8438191dd3713bff","attempts":1}

// 端点 5xx：退确定性阈值路径，人来兜一眼
{"decision_source":"threshold_fallback","usable":false,"decision":null,
 "pending_manual_review":true,"fallback_reason":"http_5xx","attempts":3,
 "attempt_kinds":["http_5xx","http_5xx","http_5xx"],"backoff_delays":[0.2,0.8],
 "error":"http_5xx: HTTP 503 {...}","audit_written":true}
```

每一次调用（成功、降级、压根没调）都在 `inference_audit` 留一行：`request_id` /
`model` / `prompt_hash` / `prompt_version` / `latency_ms` / `prompt_tokens` /
`completion_tokens` / `decision_source` / `fallback_reason` / `pending_manual_review`。
护士长问「这句话是模型说的还是算出来的」，答案在库里。

提示词正文不写死在代码里，放在 `src/infusion_monitor/gateway/templates/*.md`：
改提示词 = 改一个文件（可 review、可 diff、可回滚），业务代码一个字都不用动。
模板里那行输出契约（模型能说哪些结论、置信度落在什么范围）由输出 schema 渲染填入，
**不会和校验用的模型定义对不上**。`prompt_version` 记的就是模板标识
（如 `infusion_change@v1`）：结论有争议时先看版本有没有变、模板有没有改，
再讨论模型给得对不对；同一份输入的 `prompt_hash` 相同，说明提示词与观测都没变，
结论应当可复现。模板清单见 `docs/prompt-templates.md`。

### 决策仲裁：模型不是裁判，只是措辞与第二意见

开了 `LLM_FUSION_ENABLED` 之后，模型会被请来看一眼点名判定，但**只有一种情况它的
结论算数**：与阈值路径一致、且自评置信度过门禁。即便如此，被采信的也只是它给护士看的
那句人话——结论与触发原因码永远取阈值路径，因为那背后是连续帧数、余量门限与近空观测，
可复算；模型只有一句话。冲突、越界、低置信、端点不可达一律回落，并如实把
`decision_source` 改写成 `threshold_fallback`。

```json
// 一致 → 采信措辞，结论与原因码仍取阈值路径
{"decision_source":"model_decided","decision":"needs_change",
 "decision_reason":"液面已经低到袋底，护士到床旁就该看着换了",
 "arbitration":{"accepted":true,"code":"agreed_with_threshold",
   "threshold_decision":"needs_change","model_decision":"needs_change",
   "model_confidence":0.88,"trigger_reason_matched":true}}

// 冲突 → 回落，理由换成阈值路径那句可复算的话
{"decision_source":"threshold_fallback","decision":"needs_change",
 "decision_reason":"连续 24 帧低于 0.15，余量判定=near_empty",
 "arbitration":{"accepted":false,"code":"conflicts_with_threshold",
   "threshold_decision":"needs_change","model_decision":"no_change"}}

// 压根没调模型（默认）：正常路径，不是故障
{"decision_source":"threshold_only",
 "arbitration":{"accepted":false,"code":"model_not_called"}}
```

`arbitration.code` 的取值与各自的处置：`agreed_with_threshold`（一致采信）/
`conflicts_with_threshold`（冲突回落）/ `below_confidence_floor`（低置信回落）/
`model_output_unusable`（缺可执行的触发原因码，改提示词）/ `model_out_of_contract`
（说了词表以外的结论）/ `gateway_degraded`（端点不可达，找信息科）/ `model_not_called`
（本轮没调模型）/ `no_threshold_conclusion`（没有可比对的阈值结论）。

采信与回落各落一行 append-only 审计（`inference.fusion_accepted` /
`inference.fusion_rejected`，detail 带原因码、双方结论与置信度），可按架、按床导出复盘。
**无论哪一态，`needs_change → replaced` 都只能由护士在终端确认触发**——仲裁层不 import
任何仓储，也没有写状态的方法。

### 运行摘要（值班一眼看现状）

```bash
curl -sS http://127.0.0.1:8000/api/v1/summary
```

```json
{
  "sessions": {
    "total": 12,
    "by_state": {"monitoring": 9, "needs_change": 3},
    "needs_change_count": 3,
    "confirmed_count": 0
  },
  "writeback": {
    "total": 5, "pending_writeback_count": 1, "unsent_pending_count": 2,
    "sent_count": 1, "failed_count": 1
  },
  "model": {
    "total_calls": 41, "threshold_fallback_count": 6,
    "pending_manual_review_count": 6, "by_decision_source": {"model_decided": 34, "threshold_fallback": 6, "threshold_only": 1},
    "runtime_metrics": {
      "invocations": 34, "requests_sent": 38, "calls_ok": 28, "calls_failed": 6,
      "failure_rate": 0.1765, "quota_blocked": 0,
      "prompt_tokens": 48210, "completion_tokens": 7412, "total_tokens": 55622, "usage_missing": 2,
      "by_kind": {"ok": 28, "http_5xx": 3, "parse_error": 3},
      "by_decision_source": {"model_decided": 28, "threshold_fallback": 6},
      "by_model": {"ward-llm-7b": {"invocations": 34, "failure_rate": 0.1765, "total_tokens": 55622,
                                   "latency": {"count": 34, "p50_ms": 214, "p95_ms": 2610, "max_ms": 4820, "mean_ms": 701.4}}},
      "latency": {"count": 38, "p50_ms": 214, "p95_ms": 2610, "max_ms": 4820, "mean_ms": 701.4},
      "tracked_requests": 12, "request_bucket_limit": 256
    },
    "quota": {
      "policy": {"daily_call_limit": 2000, "max_tokens_per_request": 1200},
      "business_day": "2026-10-23", "used_today": 34, "remaining_today": 1966, "exhausted": false
    }
  },
  "channels": {
    "telemetry": {
      "enabled": true, "mqtt_enabled": true, "webhook_enabled": true,
      "state": "connected", "last_message_at": "2026-10-23T09:41:12+00:00",
      "cached_samples": 6, "received": 118, "duplicates": 3, "fresh_seconds": 300.0
    },
    "mqtt": {
      "enabled": true, "state": "disconnected", "topic": "hospital/infusion/+/telemetry",
      "received": 118, "duplicates": 3,
      "reconnect": {
        "attempts": 4, "backoff_delays": [1.0, 2.0, 4.0, 8.0],
        "last_outcome": "issued", "exhausted": false,
        "policy": {"base_seconds": 1.0, "factor": 2.0, "max_seconds": 32.0,
                   "max_attempts": 0, "reset_after_seconds": 300.0}
      },
      "pending_audit": {"queued": 0, "max_queued": 512, "dropped": 0}
    },
    "webhook": {
      "enabled": true, "state": "active", "accepted": 9, "rejected": 1, "duplicates": 3,
      "last_message_at": "2026-10-23T09:41:12+00:00", "last_rack_id": "R-EXAMPLE-39",
      "last_signature_ok": true, "signature_header": "x-infusion-signature",
      "tolerance_seconds": 300.0, "max_body_bytes": 65536
    }
  },
  "flushed_channel_audit": 2,
  "runtime": {"uptime_seconds": 84213.4, "request_count": 1568, "error_count": 2},
  "retention": {
    "policy": {"session_retention_days": 30.0, "inference_audit_retention_days": 90.0,
               "archive_root": null, "dry_run": false},
    "cutoff": "2026-09-23T00:00:00+00:00",
    "sessions_by_frames_state": {"kept": 12},
    "eligible_for_next_purge": 0,
    "blocked_by_reason": {"within_retention": 12},
    "archived_records": 0,
    "archived_bytes": 0
  },
  "error_codes": [{"code": "INGEST_INVALID", "status": 422}, "..."]
}
```

`needs_change_count` 是**欠着的人工动作**（系统已点名、护士还没确认），不是错误。
`pending_writeback_count` 只数**退避耗尽送不出去**的事件——护士还没确认的 `pending`
单列在 `unsent_pending_count`，两者混在一起会让护士长以为自己漏点了确认。
`retention` 段回答「盘上还剩多少帧、下一次定时清理会清哪几袋、为什么有的袋清不掉」
（`blocked_by_reason` 按原因码计数，明细在清理脚本的输出里）。

值班的人只问一句「遥测还在不在流」，看 `channels.telemetry.state` 即可（`connected` =
两条入站通道里至少一条在流）；要细看是哪条断了，`channels.mqtt` 与 `channels.webhook`
分列——**接了但断了**（`disconnected`）与**本部署根本没接泵网关**（`disabled`）是两件
完全不同的事。`mqtt.reconnect.last_outcome` 是 `issued` 而不是 `connected` 时，说明
「重连调用发出去了」，还不等于「连上了」；连没连上要等 broker 的 CONNACK 回调改状态。

`model` 里的两组数**口径不同，不要相加**：`total_calls` / `by_decision_source` 查的是
`inference_audit` 表（落过审计的所有调用，跨重启也在），`runtime_metrics` 记的是
**本进程启动以来**的用量、token、延迟分位与失败率（重启归零，因为它回答的是「这台
机器今天烧了多少」）。`quota` 是成本护栏的当日账：`exhausted=true` 之后每一轮都走
确定性阈值路径，`quota_blocked` 就是被护栏挡下、**一个 token 都没花**的次数。

### 入站遥测：webhook 兜底通道与去重

对接契约（端点、签名算法、载荷字段、拒收原因码）见
[`docs/webhook-contract.md`](webhook-contract.md)，该文件由代码常量生成、有测试比对
是否同步。**未实连范围**：真实院内泵网关的 webhook 端点本轮未实连，只对本地测试客户端
跑过签名 / 去重 / 拒收路径。

```bash
# 泵网关侧：把一条泵速 POST 过来（时间戳可选，带了就校验重放窗口）
BODY='{"bag_seq":"B01","pump_rate_ml_per_h":40.0,"device_sn":"PUMP-EXAMPLE-1","ts":"2026-10-23T09:40:00+08:00"}'
TS=$(date +%s)
SIG=$(printf '%s.%s' "$TS" "$BODY" | openssl dgst -sha256 -hmac "$WEBHOOK_SIGNING_SECRET" -hex | sed 's/^.* //')

curl -sS -X POST "http://127.0.0.1:8000/api/v1/infusion/telemetry/webhook/R-EXAMPLE-39" \
  -H "Content-Type: application/json" \
  -H "X-Infusion-Signature: sha256=$SIG" \
  -H "X-Infusion-Timestamp: $TS" \
  -d "$BODY"
```

```json
{
  "accepted": true,
  "duplicate": false,
  "rack_id": "R-EXAMPLE-39",
  "bag_seq": "B01",
  "pump_rate_ml_per_h": 40.0,
  "device_sn": "PUMP-EXAMPLE-1",
  "ts": "2026-10-23T09:40:00+08:00",
  "source": "webhook",
  "timestamp_enforced": true
}
```

同一条再送一次（同 `device_sn` + 同 `ts`）回 `202` + `{"duplicate": true, "accepted": false}`，
**不是错误**——对端拿 4xx/5xx 会当成失败去重投，而重复的成因往往就是「上次重投没收到
应答」，那会变成无限重投。MQTT 与 webhook 指向同一个泵网关时，同一份遥测从两条路各进来
一次也会被去重表认出来（`channels.*.duplicates` 计数）。

这条遥测随后出现在下一次上传响应的每个袋位里：

```json
{
  "bag_seq": "B01",
  "eta_source": "conflict",
  "suppression_reason": "rate_conflict",
  "telemetry_state": "fresh",
  "telemetry_age_seconds": 60.0,
  "telemetry_source": "webhook"
}
```

`eta_source="conflict"` 是因为输注医嘱写 5 ml/h、泵上跑 40 ml/h——两个值连同偏差一起交
出来请人确认，**不取中间值**，`eta_minutes` 不出现。没有遥测时同一袋给的是
`telemetry_state="absent"` + `eta_source="order_text"` + 正常的 `eta_minutes`：少一个
交叉校验来源，余量与换液判定一样成立。

### 一行一条的结构化日志

```json
{"ts":"2026-10-23T00:42:03.787","level":"INFO","service":"infusion-volume-monitor","logger":"infusion_monitor.http","event":"http.request","request_id":"req-e2e-ward-terminal-0013","method":"POST","path":"/api/v1/infusion/tracks","status":201,"duration_ms":140.21,"client":"10.0.3.41","slow":false}
```

调用方自带 `X-Request-Id` 就沿用它，没有就服务端生成一个，回写到响应头。业务日志行
（枚举完成、帧序列接入完成…）与审计行挂同一个 id，所以「护士端某次上传」可以一捞到底。
凭据位按名脱敏成 `***`，日志里查不到任何明文密钥。

### 保留期清理报告与容量测算

清理脚本的输出（`python3 scripts/purge_expired_sessions.py`）——**被留下多少、为什么留下**，
和删了多少一样要紧：

```text
保留期 30 天，截止 2026-09-23T08:00:00+00:00：归档 0 袋、删除 18 袋、目录本就不在 1 袋；
保留 42 袋（未完成 / 有写回欠账 / 还在保留期内），释放 214.6 MiB；裁掉推理审计 12 行。
  保留 sess_a1b2c3：会话还没结束（monitoring / needs_change），护士还在核对
  保留 sess_d4e5f6：出站换液事件还没送达对端
  保留 sess_7890ab：同一段短片的另一只袋子还没到清理条件，整个目录一起留着
  …… 另有 39 袋未列出
```

容量测算（`python3 scripts/estimate_storage.py --wards 1`）：

```text
【帧目录】
  每天上传          12 轮 / 24 袋 / 3,600 帧
  每天新增          548.4 MiB
  保留期内占用      16.1 GiB（保留 30 天）
  不做保留期        195.5 GiB（一年）

【数据库】
  每天新增行数      {'bag_session': 24, 'infusion_event': 90, 'writeback_queue': 90, ...}
  每天增长          595.2 KiB
  一年增长          212.2 MiB

【换 PostgreSQL 的判据】
  - 一年后会话表约 8,760 行，未过 500,000 行的换库判据（单机 SQLite 够用）
```

单帧字节数的默认值是**本产品渲染器在规范分辨率（长边 1280px）下的实测下界**（合成
画面是平色，真手机照片同分辨率下更大）。要拿部署环境的真值，加 `--frames-root` 直接
量盘上的真实帧，脚本会把口径标成「实测」。

### 样例数据从哪来（合成替身，不是真实院内短片）

真实输液架短片必须经医院授权采集，画面里会有床号牌、输液标签与护士的手。本仓库
自动化验收跑在**程序合成**的短片上：

```bash
# 渲一段三袋短片 + 逐帧真值清单（同一 seed 可逐字节复现）
python scripts/synthesize_rack.py --out data/synth --bags 3 --frames 24 --variant night
```

| 用途 | 数据来源 | 性质 |
|------|---------|------|
| 袋型标定与毫升换算回归 | `scripts/synthesize_rack.py` 合成短片（已知真值） | 回归基准 |
| 分割精度对照（IoU / J&F） | 公开视频分割数据集子集（脚本化下载，断网降级为 skipped） | 评测基准 |
| 单元测试边界样例 | `tests/fixtures/`、`tests/support/` | 单测夹具 |

**三者都是替身，不是最终验收边界。** 合成器不产生任何真实标识：拍摄时间写
`<EXAMPLE_CAPTURED_AT>`，袋型只给 `TYPE-A`…`TYPE-D` 泛化外形代号，不含厂商、批号或
效期；清单里有回归用例盯着这条纪律。完整说明见 [`docs/synthetic-data.md`](synthetic-data.md)。

### 评测基础从哪来（公开视频分割子集，脚本化取数）

```bash
# 只读远端 ZIP 中央目录试算，不下任何字节
python scripts/fetch_vos_subset.py --dry-run
# 实取 10 个最长序列（幂等可重跑；断网如实降级为 skipped，不伪造数据）
python scripts/fetch_vos_subset.py --relax-min-frames --max-sequences 10 --update-doc
```

| 项 | 要求 | 实际（实测自远端中央目录） |
|----|------|---------------------------|
| 序列数 | ≥ 10 | **10**（cows / koala / parkour / soapbox / drone / scooter-board / stroller / bmx-bumps / camel / dance-twirl） |
| 每序列帧数 | ≥ 150 | **90–104，未达成**：该语料 90 个序列里最长的也只 104 帧，需显式 `--relax-min-frames` |
| 首帧标注 | 需要 | 每序列均有（`Annotations/480p/<seq>/00000.png`） |
| 规模与耗时 | — | 946 帧 / 1892 个文件 / 122.9 MB；首次 77.7s，重跑 4.6s 且不下任何帧字节 |

取数按中央目录里的字节区间**只下该下的**（完整归档 794 MB，本轮实际下发约 123 MB），
逐文件 CRC32 对账 + 图像解码 + 首帧标注对账；断网或校验不过即降级为 `skipped`，
原因写在 `docs/benchmark-vos.md` 顶部，**不拿合成帧或占位图冒充真数据**。口径、许可与
降级纪律见 [`docs/benchmark-vos.md`](benchmark-vos.md)。

### 「跨帧记忆」到底值多少：一个可复现的消融对照

这批公开子集还用来回答产品主路径里的一个具体问题：护士第二轮巡视把镜头重新对准
输液架时，**逐帧重新认一遍**（阈值单帧）和**接着上一帧往下追**（跨帧记忆）差多少。
两套方案走**同一段主路径、同一个阈值核、同一组阈值**，唯一变量是每帧提示从哪来。

| 方案 | 每帧提示来自 | 全体 J&F | 含遮挡帧 J&F |
|------|--------------|---------|--------------|
| 跨帧记忆传播 | 上一帧掩膜导出的区域 + 质心正点 | **0.0646** | **0.0726** |
| 阈值单帧 | 首帧锚点，每帧重新起算 | 0.0649 | 0.0671 |

结论要照实说：**记忆在含遮挡的帧上确实更好（+0.0055），全体则基本打平（−0.0003）**。
记忆的价值是「被挡住之后还能接回同一个袋子」，不是整体分割精度更高。

两套方案的绝对值都很低（IoU 约 0.04–0.05）——纯色彩阈值核在这类自然视频上本就接近
随机水平，参数扫描下来最好的组合也只有约 0.10。**这是弱信号，不能当成精度结论。**
本环境无 GPU、无模型权重，跑的是确定性阈值核，也不是任何神经分割模型的精度。
完整对照表、参数取值依据与逐序列明细见
[`docs/benchmark-vos.md`](benchmark-vos.md)。

```bash
# 复现上表（语料已在 data/vos_subset 时）
python scripts/run_vos_benchmark.py --update-doc
```

---

## 12. 护士确认换液的完整响应与终端回显（自 README 摘录）

> 本节样例来自 `scripts/capture_demo_shots.py` 的真实运行（真 uvicorn + 真 curl +
> 真 SQLite + 本地替身对端），README 只留速查片段。

### 12.1 首次确认

护士在终端把点名清单里的 `confirm_hint` 原样回填，另加自己的工号：

```console
$ curl -sS http://127.0.0.1:49597/api/v1/infusion/pickups | jq '.items[0].confirm_hint'
{
  "session_id": "sess_c2edf96e70a648f8",
  "bag_seq": "B01",
  "client_event_id": "infusion-sess_c2edf96e70a648f8-B01-replaced"
}

$ curl -sS -X POST http://127.0.0.1:49597/api/v1/infusion/transition \
    -H 'Content-Type: application/json' \
    -d "$(jq -c '.items[0].confirm_hint + {operator_id: "N-EXAMPLE-45"}' pickups.json)" \
    -o transition.json -w '→ HTTP %{http_code}\n'
→ HTTP 200
```

`POST /api/v1/infusion/transition` 的完整响应：

```json
{
  "session_id": "sess_c2edf96e70a648f8",
  "bag_seq": "B01",
  "state": "replaced",
  "operator_id": "N-EXAMPLE-45",
  "client_event_id": "infusion-sess_c2edf96e70a648f8-B01-replaced",
  "event_id": "evt_27f24fbabf2341b2",
  "created": true,
  "idempotent_hit": false,
  "confirmed_at": "2026-10-23T14:49:41+00:00",
  "pickup_evidence": {
    "previous_state": "needs_change",
    "previous_trigger_reason": "eta_within_window",
    "previous_eta_minutes": 13,
    "previous_fill_ratio": 0.1261330124770122,
    "bed_no": "EXAMPLE-BED-45",
    "rack_id": "R-EXAMPLE-45"
  },
  "writeback": {
    "status": "sent",
    "ok": true,
    "attempts": 1,
    "kind": "ok",
    "refused": false
  }
}
```

字段逐条说明：

| 字段 | 含义 |
|------|------|
| `state` | 确认后的状态，恒为 `replaced`（终态） |
| `client_event_id` | 出站幂等键，**服务端唯一生成**，由 (会话, 袋序) 决定、不含确认时刻 |
| `event_id` | 出站事件 ID；重放时与首次完全相同 |
| `created` | 本次是否真的产生了一条新事件 |
| `idempotent_hit` | `true` = 同一次确认的重放，未新增事件、未再外发 |
| `confirmed_at` | 确认时刻（ISO8601）；重放时与首次逐字相同 |
| `pickup_evidence` | **确认之前**的判断依据留档（状态、触发原因码、ETA、液面） |
| `writeback` | 出站结果：`sent` / `pending_writeback` / `failed` / `pending`，失败也如实返回 |

`pickup_evidence` 存在的理由：确认会把 `trigger_reason` 覆盖成 `nurse_confirmed`，
所以先把原来那条留档，事后查得到「当初是按哪条理由点的名」。

### 12.2 幂等重放（同一 client_event_id 连发两次）

第二次的完整响应：

```json
{
  "session_id": "sess_c2edf96e70a648f8",
  "bag_seq": "B01",
  "state": "replaced",
  "operator_id": "N-EXAMPLE-45",
  "client_event_id": "infusion-sess_c2edf96e70a648f8-B01-replaced",
  "event_id": "evt_27f24fbabf2341b2",
  "created": false,
  "idempotent_hit": true,
  "confirmed_at": "2026-10-23T14:49:41+00:00",
  "pickup_evidence": {
    "previous_state": "replaced",
    "previous_trigger_reason": "nurse_confirmed",
    "previous_eta_minutes": 13,
    "previous_fill_ratio": 0.1261330124770122,
    "bed_no": "EXAMPLE-BED-45",
    "rack_id": "R-EXAMPLE-45"
  },
  "writeback": {
    "status": "sent",
    "ok": true,
    "attempts": 1,
    "kind": "already_sent",
    "refused": false
  }
}
```

两次逐字相同的是 `event_id` 与 `confirmed_at`；变的只有 `created`（`true` → `false`）、
`idempotent_hit`（`false` → `true`）与 `writeback.kind`（`ok` → `already_sent`）。
事件表不新增行，对端也不收第二条。

### 12.3 服务侧结构化日志（一条 JSON 一行，request_id 贯穿）

```jsonl
{"ts":"2026-10-23T22:49:40.506","level":"INFO","event":"http.request","request_id":"req_3be8061725314373"}
{"ts":"2026-10-23T22:49:40.552","level":"INFO","event":"枚举完成 候选=0 绑定=0 槽位=5 区域=整幅","request_id":"req_6c0d3629783e4756"}
{"ts":"2026-10-23T22:49:41.234","level":"INFO","event":"换液点名判定完成 rack=R-EXAMPLE-45 shift=2026-10-22-AM named=1 truncated=0 suppressed=4","request_id":"req_6c0d3629783e4756"}
{"ts":"2026-10-23T22:49:41.234","level":"INFO","event":"帧序列接入完成 rack=R-EXAMPLE-45 order=EXAMPLE-ORDER-0045 frames=20 bags=5 candidates=0 shift=2026-10-22-AM camera=640x480 canonical=640x480 roi={\"x\": 0, \"y\": 69, \"width\": 640, \"height\": 411}","request_id":"req_6c0d3629783e4756"}
{"ts":"2026-10-23T22:49:41.235","level":"INFO","event":"http.request","request_id":"req_6c0d3629783e4756"}
```

同一个 `request_id` 把「枚举 → 点名判定 → 帧序列接入 → 出站」串成一条链；
健康探活行不进业务日志。

> 「枚举完成 候选=0」不是故障：这支合成短片的液体饱和度恰在枚举稳定性探测高阈值
> （96+24=120）之下，首帧候选被判 0 分剔除，主路径改走**槽位先验起锚**，曲线、
> 毫升与点名仍照常给出。换一支饱和度更高的短片候选即正常出现。
