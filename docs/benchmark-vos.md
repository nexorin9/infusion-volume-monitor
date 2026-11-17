<!-- vos-subset-status:begin -->
> **公开 VOS 子集：可用** · 10 个序列 / 946 帧 / 122.9 MB
>
> - 每序列帧数门禁：要求 150 帧 → **未达成**
> - 门禁缺口：公开语料无法满足门禁：归档 90 个序列中最长的也只有 104 帧，少于要求的每序列 150 帧（实测自远端 ZIP 中央目录，非估计值）
> - 取数方式：`--relax-min-frames` 显式放松后取语料内最长的一批序列（缺口如上，不静默替换）
> - 落盘位置：`data/vos_subset`（`data/` 已在 .gitignore 中排除，不入库）
> - 取数统计：新取 0 个文件 / 复用 1892 个 / 断点续传 0 个；校验：逐文件 CRC32 对账 + 首帧标注对账 + 图像解码
> - 本地取数耗时：4.7s
>
> 复现命令见下文「获取子集」一节；断网时本块会如实改写为 `skipped` 并给出原因，
> **不会**用合成帧或占位图冒充真数据（合成回归另见 `docs/synthetic-data.md`）。
<!-- vos-subset-status:end -->

# 公开 VOS 子集（评测基础取数与校验）

> 真实院内短片须经医院授权采集，本轮自动化验收改用**公开视频分割语料**的一个子集，
> 作为逐帧身份传播的可复现评测基础（IoU / J&F 的输入，见 `docs/synthetic-data.md` 里的
> 合成真值回归作为另一条独立口径）。本文只讲：怎么取、怎么校验、门禁缺口在哪。

## 为什么是脚本化下载，而不是"手工下载一次"

评测基础必须能在任何一台机器上按同一条命令重建，否则"可复现"只是口头话。所以
`scripts/fetch_vos_subset.py` 负责四件事：远端直读、规模可限、幂等可重跑、失败如实降级。

## 获取子集

```bash
# 1) 先试算：只读远端 ZIP 中央目录，不下任何字节
python scripts/fetch_vos_subset.py --dry-run

# 2) 取 10 个最长序列（本语料达不到每序列 150 帧门禁 → 必须显式放松，理由见顶部状态块）
python scripts/fetch_vos_subset.py --relax-min-frames --max-sequences 10 --update-doc

# 3) 只取指定序列、限总字节
python scripts/fetch_vos_subset.py --sequences koala,sheep --max-bytes 30000000

# 4) 幂等重跑：CRC32 对得上的文件一个字节都不再下
python scripts/fetch_vos_subset.py --relax-min-frames

# 5) 机器可读结果（CI / 编排层消费）
python scripts/fetch_vos_subset.py --relax-min-frames --json
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--out` | `data/vos_subset` | 落盘根目录（`data/` 已在 .gitignore 中排除） |
| `--sequences` | 空（自动按帧数从长到短选） | 逗号分隔；指定序列在归档里不存在时直接报错，不猜 |
| `--min-frames` | `150` | **每序列帧数门禁**；达不成时默认拒绝取数，不静默降门槛 |
| `--relax-min-frames` | 关 | 显式放松门禁、取语料内最长的一批；缺口照实写进 manifest 与本文顶部 |
| `--max-sequences` | `10` | 最多取几个序列（`0` = 不限） |
| `--max-bytes` | 无 | 总压缩字节上限，超出即截断（实际取到多少记在 manifest） |
| `--dry-run` | 关 | 只试算选片，不下载、不建目录 |
| `--json` | 关 | 以 JSON 打印完整结果（规模、门禁、逐文件 CRC、失败列表） |
| `--update-doc` | 关 | 把状态块刷新到本文顶部（重复执行只替换该块，正文不动） |

## 落盘结构

```
data/vos_subset/
├── JPEGImages/480p/<seq>/%05d.jpg     # 画面帧（854×480）
├── Annotations/480p/<seq>/%05d.png    # 逐帧标注（评测真值，含首帧标注）
└── manifest.json                      # 逐文件 bytes + CRC32 + 规模 + 门禁达成情况
```

标注目录是**多对象调色板 PNG**（一张图里可含多个对象），与公开评测器读取的目录约定一致；
逐帧身份传播的评测直接消费这个目录，无需再做格式转换。

## 规模门禁：要求与实际（不要互相冒充）

| 项 | 要求（任务口径） | 实际（实测自远端中央目录） |
|----|------------------|---------------------------|
| 序列数 | ≥ 10 | **10**（cows / koala / parkour / soapbox / drone / scooter-board / stroller / bmx-bumps / camel / dance-twirl） |
| 每序列帧数 | ≥ 150 | **90–104，未达成**：该公开语料 90 个序列里最长的 `cows` 也只有 104 帧 |
| 首帧标注 | 需要 | 每序列均有（`Annotations/480p/<seq>/00000.png`） |
| 帧数合计 | — | **946** 帧 / 1892 个文件（画面 + 标注）/ 122.9 MB |

脚本默认**拒绝**在门禁未达成时取数（返回 `skipped`，退出码 2）；本次是显式加
`--relax-min-frames` 才取到上述规模，缺口同时写在 `manifest.json` 的 `gate` 段和本文
顶部状态块里。评测结论引用这批数据时必须带上这个缺口，别把它写成"≥150 帧序列已验证"。

## 完整性校验做了什么

1. **逐文件 CRC32 对账**：ZIP 中央目录里带每个成员的 CRC32，落盘后逐个重算比对，
   不符即判失败、该文件不进正式路径（先写 `.tmp`，通过才原子替换）。
2. **manifest ↔ 磁盘对账**：`manifest.json` 逐文件记 `path / bytes / crc32 /
   declared_crc32 / source`，可与磁盘逐条核对（`tests/test_fetch_subset.py` 有对应用例）。
3. **图像解码**：用真实解码器把每个序列的首帧打开一遍，确认不是"长度对但解不开"的坏文件，
   并记下真实尺寸（实测 854×480）。
4. **首帧标注对账**：确认每个序列确有首帧标注，评测才有起点。

## 性能与工程取舍（实测）

| 项 | 实测 |
|----|------|
| 完整归档大小 | 832,766,765 字节（约 794 MB） |
| 取 10 个序列实际下发 | 约 123 MB 压缩负载 —— **不整包下载** |
| 中央目录读取 | 尾部 64 字节探针 + 精确取中央目录；**一个帧负载字节都不顺带下** |
| 首次取数（1892 个文件） | **77.7s** |
| 幂等重跑（1892 个文件） | **4.6s**，下发 0 个帧字节（全部 CRC 命中复用） |

相邻成员在归档里是连着放的，所以按段合并成块读（一次请求切多个成员，逐成员解压 + 校验），
而不是"一个文件一次往返"——后者在这个规模下要几十分钟。块内若出现对齐不符或某个成员坏掉，
自动退回逐成员取，并把**真正出问题的那一个成员**单独记进失败列表。

## 断网 / 校验失败时的降级纪律

- 断网、归档不可读、站点不支持 Range、成员校验不过 → 状态一律 `skipped`，退出码 2；
  原因写进 `manifest.json` 的 `skipped_reason` 与本文顶部状态块。
- **不生成任何替代数据**：不用合成帧、占位图、拷贝或随机数冒充真数据；坏文件不留在正式
  路径上（`tests/test_fetch_subset.py` 断言断网后磁盘上只剩 `manifest.json`）。
- 合成短片是另一条独立口径（回归真值用），见 `docs/synthetic-data.md`，两者不混用。

## 数据来源与许可

- 数据集：DAVIS 2017 trainval 480p，由 ETH Zürich CVG 与日内瓦大学发布；许可、引用与
  来源清单见归档内 `DAVIS/README.md`、`DAVIS/SOURCES.md`。
- 本仓库只在本地落一份评测副本用于回归，**不 redistribution**；`data/` 已在 `.gitignore` 中
  排除，不会进 Git 仓库。
- 真实院内输液短片不在本文件范围内：它必须经医院授权采集，本轮用公开子集 + 合成真值两条
  替代口径覆盖算法回归。

---

<!-- benchmark-vos-result:begin -->

> **对照基准：已运行** · 「跨帧记忆传播」vs「阈值单帧」，同一段主路径、同一套阈值核
>
> - 语料：DAVIS 2017 trainval 480p · 10 个序列 / 946 帧（本地副本见 `data/vos_subset`，不入库）
> - 阈值核：`color_band` · hue_tolerance=14 · spread_sigma=1.5 · max_hole_area=2048 · max_sprinkle_area=192
> - 评测口径：IoU / J / F / J&F，轮廓容差 bound_th=0.008；遮挡帧判据 = 真值面积 ≤ 序列峰值的 0.5 倍
> - 可复现：随机种子 0（全链路无随机源，逐位可复现）；再获取路径：disabled（两套方案只允许「记忆」一个变量）
> - 用时：54.39s

### 全体汇总

| 方案 | 序列 | 计入帧 | IoU | J | F | J&F | 遮挡帧 J&F |
|------|------|--------|-----|---|---|-----|------------|
| `memory_propagation` | 17 | 1477 | 0.0465 | 0.0465 | 0.0827 | **0.0646** | 0.0726 |
| `single_frame_threshold` | 17 | 1477 | 0.0370 | 0.0370 | 0.0928 | **0.0649** | 0.0671 |

**结论**：记忆方案在含遮挡帧上的 J&F 高出阈值单帧 **+0.0055**，全体 J&F 高出 **-0.0003**。

### 逐序列明细

| 方案 | 序列 | 计入帧 | IoU | J | F | J&F | 遮挡帧 J&F |
|------|------|--------|-----|---|---|-----|------------|
| memory_propagation | bmx-bumps#obj1 | 68 | 0.0020 | 0.0020 | 0.1359 | 0.0689 | 0.0179 |
| single_frame_threshold | bmx-bumps#obj1 | 68 | 0.0005 | 0.0005 | 0.1255 | 0.0630 | 0.0052 |
| memory_propagation | bmx-bumps#obj2 | 65 | 0.0047 | 0.0047 | 0.0495 | 0.0271 | 0.0000 |
| single_frame_threshold | bmx-bumps#obj2 | 65 | 0.0022 | 0.0022 | 0.0421 | 0.0222 | 0.0000 |
| memory_propagation | camel#obj1 | 90 | 0.2582 | 0.2582 | 0.1847 | 0.2215 | 0.1541 |
| single_frame_threshold | camel#obj1 | 90 | 0.1514 | 0.1514 | 0.1581 | 0.1548 | 0.0762 |
| memory_propagation | cows#obj1 | 104 | 0.0476 | 0.0476 | 0.0281 | 0.0378 | 0.0000 |
| single_frame_threshold | cows#obj1 | 104 | 0.0498 | 0.0498 | 0.0694 | 0.0596 | 0.0000 |
| memory_propagation | dance-twirl#obj1 | 90 | 0.0404 | 0.0404 | 0.0640 | 0.0522 | 0.0776 |
| single_frame_threshold | dance-twirl#obj1 | 90 | 0.0523 | 0.0523 | 0.0747 | 0.0635 | 0.0588 |
| memory_propagation | drone#obj1 | 76 | 0.0002 | 0.0002 | 0.0770 | 0.0386 | 0.0419 |
| single_frame_threshold | drone#obj1 | 76 | 0.0013 | 0.0013 | 0.0833 | 0.0423 | 0.0453 |
| memory_propagation | drone#obj2 | 67 | 0.0048 | 0.0048 | 0.0579 | 0.0314 | 0.0416 |
| single_frame_threshold | drone#obj2 | 67 | 0.0048 | 0.0048 | 0.0396 | 0.0222 | 0.0250 |
| memory_propagation | drone#obj3 | 91 | 0.0114 | 0.0114 | 0.0414 | 0.0264 | 0.0216 |
| single_frame_threshold | drone#obj3 | 91 | 0.0754 | 0.0754 | 0.0951 | 0.0852 | 0.0788 |
| memory_propagation | koala#obj1 | 100 | 0.0767 | 0.0767 | 0.0412 | 0.0590 | 0.0616 |
| single_frame_threshold | koala#obj1 | 100 | 0.0572 | 0.0572 | 0.0659 | 0.0616 | 0.0522 |
| memory_propagation | parkour#obj1 | 100 | 0.0526 | 0.0526 | 0.1225 | 0.0875 | 0.1404 |
| single_frame_threshold | parkour#obj1 | 100 | 0.0213 | 0.0213 | 0.1108 | 0.0661 | 0.0994 |
| memory_propagation | scooter-board#obj1 | 75 | 0.0137 | 0.0137 | 0.0587 | 0.0362 | 0.0347 |
| single_frame_threshold | scooter-board#obj1 | 75 | 0.0347 | 0.0347 | 0.0896 | 0.0621 | 0.0667 |
| memory_propagation | scooter-board#obj2 | 72 | 0.0037 | 0.0037 | 0.0723 | 0.0380 | 0.0389 |
| single_frame_threshold | scooter-board#obj2 | 72 | 0.0040 | 0.0040 | 0.1284 | 0.0662 | 0.0692 |
| memory_propagation | soapbox#obj1 | 99 | 0.0414 | 0.0414 | 0.0776 | 0.0595 | 0.0826 |
| single_frame_threshold | soapbox#obj1 | 99 | 0.0043 | 0.0043 | 0.0561 | 0.0302 | 0.0431 |
| memory_propagation | soapbox#obj2 | 99 | 0.0413 | 0.0413 | 0.0931 | 0.0672 | 0.1210 |
| single_frame_threshold | soapbox#obj2 | 99 | 0.0178 | 0.0178 | 0.0714 | 0.0446 | 0.0769 |
| memory_propagation | soapbox#obj3 | 99 | 0.0953 | 0.0953 | 0.1771 | 0.1362 | 0.2023 |
| single_frame_threshold | soapbox#obj3 | 99 | 0.0610 | 0.0610 | 0.1668 | 0.1139 | 0.1603 |
| memory_propagation | stroller#obj1 | 91 | 0.0139 | 0.0139 | 0.0461 | 0.0300 | 0.0886 |
| single_frame_threshold | stroller#obj1 | 91 | 0.0445 | 0.0445 | 0.1118 | 0.0781 | 0.0579 |
| memory_propagation | stroller#obj2 | 91 | 0.0204 | 0.0204 | 0.0695 | 0.0450 | 0.0655 |
| single_frame_threshold | stroller#obj2 | 91 | 0.0070 | 0.0070 | 0.0768 | 0.0419 | 0.0787 |
| **memory_propagation（全体）** | — | 1477 | 0.0465 | 0.0465 | 0.0827 | **0.0646** | 0.0726 |
| **single_frame_threshold（全体）** | — | 1477 | 0.0370 | 0.0370 | 0.0928 | **0.0649** | 0.0671 |

含遮挡序列（按真值面积判据自动识别）：`bmx-bumps#obj1`、`bmx-bumps#obj2`、`camel#obj1`、`dance-twirl#obj1`、`drone#obj1`、`drone#obj2`、`drone#obj3`、`koala#obj1`、`parkour#obj1`、`scooter-board#obj1`、`scooter-board#obj2`、`soapbox#obj1`、`soapbox#obj2`、`soapbox#obj3`、`stroller#obj1`、`stroller#obj2`

### 诚实标注

- **本表不是源模型精度，也不是产品分割精度。** 本环境无 CUDA、无权重，跑的是本项目自己的确定性阈值核（`color_band`）。它衡量的是**一个机制**——跨帧记忆相对单帧阈值到底带来多少增益。
- **绝对值很低，两套方案都在近乎随机的水平上。** 全体 IoU 约 0.046（单帧 0.037）。原因是纯色彩带核在这批自然视频上基本切不准——主体与背景常有相近色相与相近亮度。**因此表里的差值只能当作弱信号看，不要当成「记忆机制带来了可观的精度提升」。**
- **参数不是在这张表上调出来的，取值依据见下表。** 在 6 个序列 × 30 帧上做过一轮参数扫描（`spread_sigma` × `spread_floor` × `close_ksize` × `seed_radius`），**最好的一组也只有约 0.10 的平均 IoU**，说明瓶颈在核的能力而非调参：

| 取值 | 依据 |
|------|------|
| `spread_sigma=1.5`、`spread_floor=16` | 阈值带取「中位数 ± 1.5×1.4826×MAD」并设 16 的下限。用中位数与 MAD 而非均值与标准差：提示区域必然带进背景，均值被拉走、离散度被撑大，阈值带放宽到整幅画面都落进来。 |
| `close_ksize=0`（关掉闭运算） | 闭运算会把主体与背景之间的过渡带桥成一整块，连通域选中后直接吞掉半幅画面（实测 `camel` 第 0 帧把 7 万像素主体选成 27 万）。开运算保留，用于抹噪点。 |
| `seed_radius_fraction=0.04` | 色彩模型样本取正提示点周围一圈纯主体像素，而不是整个提示框。半径再按提示框短边的一半封顶，目标小的时候跟着缩。 |

- **记忆赢在遮挡帧上，全体上是平的。** 全体 J&F 差 -0.0003（基本打平），含遮挡帧上差 +0.0055。这正是本基准要验的那条命题——**记忆的作用是遮挡后还能接回同一个对象，不是整体分割精度更高**；逐序列明细里也能看到它逐条有输有赢（`camel`、`soapbox` 记忆明显更好，`drone#obj3`、`scooter-board` 单帧更好）。
- **语料是自然视频，不是输液短片。** 线上阈值核（`geom_kernel`）按「高饱和、中等亮度」找袋中液体，是输液架的场景先验；公开语料是牛、无人机、漂移板，因此对照改用由提示区域自估色彩模型的 `color_band` 核。两套方案用**同一份**该核与同一组阈值，变量只有「每帧提示从哪来」，所以表里的差值可归因到记忆机制本身。
- **输液场景自己的准确度口径不在本表。** 袋型标定与毫升换算的回归见 `docs/synthetic-data.md`（合成真值，含反光 / 形变 / 暗光 / 低液位变体）。
- **含遮挡序列 16 个**（判据：真值面积掉到序列峰值的一半以下，不依赖跑哪套方案）：`bmx-bumps#obj1`、`bmx-bumps#obj2`、`camel#obj1`、`dance-twirl#obj1`、`drone#obj1`、`drone#obj2`、`drone#obj3`、`koala#obj1`、`parkour#obj1`、`scooter-board#obj1`、`scooter-board#obj2`、`soapbox#obj1`、`soapbox#obj2`、`soapbox#obj3`、`stroller#obj1`、`stroller#obj2`。
- **起锚用的是首帧真值的质心点与外接框**，两套方案同起点；起锚信息量不偏向任何一方。

<!-- benchmark-vos-result:end -->
