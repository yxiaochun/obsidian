---
创建日期: 2026-11-16
更新日期: 2026-11-16
类型: 复盘
标签:
  - 生成式推荐
  - 推荐系统
来源:
  - "[[2025ByteDance_RankMixer.pdf]]"
状态: 待复核
---

# 2026-11-16｜论文引文简读

## 底稿论文

- **论文**：RankMixer: Scaling Up Ranking Models in Industrial Recommenders
- **一句话主题**：RankMixer 是一种面向工业推荐系统的硬件感知排序模型架构，通过多头 token 混合替代二次自注意力、用 Per-token FFN 建模特征子空间，并扩展稀疏 MoE 至十亿参数，在万亿级生产数据上显著提升 MFU 并保持推理延迟。
- **References 总量**：42
- **近 5 年候选**：21
- **今日精选**：5

## 精选引文

### 1. Hiformer: Heterogeneous Feature Interactions Learning with Transformers for Recommender Systems（2023）

- **来源**：arXiv preprint arXiv:2311.05884
- **相关性**：工业级特征交互建模，与 RankMixer 同属用 Transformer 类结构替代手工交叉模块的排序模型扩展方向。/5
- **核心思路**：用异构注意力与低秩剪枝实现工业级特征交互建模。
- **与源论文的关系**：RankMixer 在引言中讨论 CPU 时代手工交叉模块的局限，Hiformer 是同期用注意力结构做特征交互的代表工作，可作为对比基线。
- **研究价值**：提供工业级 Transformer 特征交互的工程经验，帮助理解 RankMixer 在 MFU 与延迟约束下的设计取舍。
- **历史被引**：6 篇底稿论文
- **已有精读**：[[Hiformer：用异构注意力与低秩剪枝实现工业级特征交互建模]]
- **需要核实**：候选条目年份标注为 2023 但被归入 2025 组，需人工核实正式发表年份与引用格式。

### 2. LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders（2025）

- **来源**：arXiv:2505.04421 [cs.IR]
- **相关性**：工业推荐系统扩展与长序列建模，与 RankMixer 同属字节跳动工业排序模型扩展工作。/5
- **核心思路**：全局 token 稳定超长行为序列建模。
- **与源论文的关系**：RankMixer 聚焦特征交互与排序模型扩展，LONGER 聚焦超长行为序列，二者可组合构成工业推荐扩展方案。
- **研究价值**：展示工业级推荐模型在序列维度上的扩展实践，为生成式推荐的长序列建模提供参考。
- **历史被引**：11 篇底稿论文
- **已有精读**：[[LONGER：全局token稳定超长行为序列建模]]
- **需要核实**：需人工核实 LONGER 与 RankMixer 在字节内部是否已有组合实验或引用关系。

### 3. DHEN: A deep and hierarchical ensemble network for large-scale click-through rate prediction（2022）

- **来源**：arXiv preprint arXiv:2203.11014
- **相关性**：大规模 CTR 排序模型的结构化扩展，是 RankMixer 引言中讨论的后续扩展工作之一。/5
- **核心思路**：设计深层分层集成网络以提升大规模 CTR 预测的扩展性能。
- **与源论文的关系**：RankMixer 引言将 DHEN 列为通过创新 DNN 结构提升扩展性能的后续工作，RankMixer 则进一步强调硬件对齐与 MFU。
- **研究价值**：提供排序模型结构扩展的早期工业验证，帮助定位 RankMixer 在扩展路线中的位置。
- **历史被引**：4 篇底稿论文
- **已有精读**：[[DHEN通过异构交互的分层集成扩展点击率预测能力]]
- **需要核实**：候选未提供具体实验数据与 MFU 对比，需人工核实原文结论。

### 4. Understanding scaling laws for recommendation models（2022）

- **来源**：arXiv preprint arXiv:2208.08489
- **相关性**：推荐模型缩放定律研究，为 RankMixer 的扩展动机提供理论背景。/5
- **核心思路**：研究推荐模型的缩放定律，分析参数、数据与计算对性能的影响。
- **与源论文的关系**：RankMixer 以缩放定律为动机，指出 CPU 时代模型 ROI 难以实现，该文提供缩放定律的基础分析。
- **研究价值**：为生成式推荐的规模化提供缩放定律视角，帮助评估 RankMixer 的扩展收益。
- **历史被引**：1 篇底稿论文
- **处理状态**：普通简读
- **需要核实**：候选未给出具体缩放指数或实验设置，需人工核实原文结论与 RankMixer 引用语境。

### 5. Scaling law of large sequential recommendation models（2024）

- **来源**：Proceedings of the 18th ACM Conference on Recommender Systems, 444–453
- **相关性**：序列推荐模型的缩放定律，与 RankMixer 的排序模型扩展主题相关。/5
- **核心思路**：研究大规模序列推荐模型的缩放定律。
- **与源论文的关系**：RankMixer 关注工业排序模型扩展，该文从序列推荐角度验证缩放规律，可互为补充。
- **研究价值**：高历史被引，为生成式推荐的序列建模扩展提供缩放定律证据。
- **历史被引**：13 篇底稿论文
- **处理状态**：高频引用，已尝试生成精读笔记
- **需要核实**：需人工核实该文是否涉及生成式推荐或仅限序列推荐，以及其缩放结论是否适用于排序场景。

## 使用方式

- 这是自动生成的引文简读，先按相关性挑 1～2 篇核对原文。
- 确认有价值后，再升级成知识卡片或选题；本文件只作为每日线索。
- 如某条内容信息不足，不要直接引用其结论，先查原文。
