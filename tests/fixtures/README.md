# tests/fixtures —— 单测用的脱敏样例

> 这一目录只放**单测边界样例**与**参数清单**，不是回归基准，更不是最终验收边界。
> 完整的合成数据说明见 [`docs/synthetic-data.md`](../../docs/synthetic-data.md)。

## 这里有什么

| 路径 | 内容 |
|------|------|
| `rack-synth-scene/manifest.json` | 一个两袋场景的**参数清单 + 逐帧真值**。仓库不提交帧——同一个 `seed` 可逐字节复现 |

## 为什么只存清单不存帧

帧是二进制，且完全由 `manifest.json` 里的参数与 `seed` 决定。存一份能逐字节复现的
清单，比存几十张图片更好审计：评审者读得懂「这袋液面从 0.72 走到 0.34、对应 480 ml
可用容量」，却没法从一堆 JPEG 里看出这一点。

复现命令：

```bash
python scripts/synthesize_rack.py \
    --reproduce tests/fixtures/rack-synth-scene/manifest.json \
    --out /tmp/repro
```

## 脱敏约定

清单里的一切标识都是占位，不含也不得含：

- 具体**日期**（拍摄时间一律写 `<EXAMPLE_CAPTURED_AT>`）；
- **批号 / 效期**（袋型只写 `TYPE-A`…`TYPE-D` 这类泛化外形代号，不带厂商）；
- **住院号 / 处方号 / 身份证号**（合成器根本不产生这些字段，由调用方给）。

`tests/test_calibration_regression.py::test_fixture_manifest_is_anonymised_and_reproducible`
会扫描这份清单，一旦出现疑似真实标识就判失败。

## 夹具分层

| 层级 | 位置 | 用途 |
|------|------|------|
| 单测边界样例 | 本目录 + `tests/support/` | 帧数不足、尺寸过小、时间逆序、遮挡、标签……各类反例 |
| 回归基准 | `scripts/synthesize_rack.py` 现场渲 | 袋型标定与毫升换算的偏差预算 |
| 评测基准 | 公开视频分割数据集子集 | IoU / J&F 对照（下载脚本化，断网降级为 skipped） |

三层**都是替身**。真实院内短片须经医院授权采集，上线前必须在真数据上重跑回归。