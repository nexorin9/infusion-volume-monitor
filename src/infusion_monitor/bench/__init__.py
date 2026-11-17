"""公开 VOS 子集上的对照基准。

本包回答一个问题：**产品主路径里「跨帧记忆」那一步，到底值不值。**

做法是在同一批真实公开语料上，把两条走同一段主路径的方案各跑一遍：

- ``memory_propagation`` —— 产品主路径。``init_state`` → ``add_new_points_or_box``
  → ``propagate_in_video``，逐帧用上一帧掩膜导出提示（记忆）。
- ``single_frame_threshold`` —— 基础方案。同样三个入口、同一个确定性阈值核，
  但每一帧都退回首帧锚点重新起提示（无记忆），即「阈值单帧」。

评测口径（逐帧 IoU / 区域相似度 J / 轮廓准确度 F / J&F）与 ``sav_evaluator.py``
所转述的那套评测协议同源，GT 目录结构沿用其文档化的 ``{video_id}/{frame_id}.png``。
本包**不**宣称复现源模型精度：本环境无 CUDA、无权重，跑的是确定性几何核。
"""

from __future__ import annotations

from .metrics import (
    BOUND_TH,
    FrameMetric,
    SequenceMetric,
    aggregate,
    db_eval_boundary,
    frame_metrics,
    jaccard,
    region_similarity,
    summarise,
)

__all__ = [
    "BOUND_TH",
    "FrameMetric",
    "SequenceMetric",
    "aggregate",
    "db_eval_boundary",
    "frame_metrics",
    "jaccard",
    "region_similarity",
    "summarise",
]