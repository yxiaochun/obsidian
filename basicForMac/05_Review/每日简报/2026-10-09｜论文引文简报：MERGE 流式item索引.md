---
创建日期: 2026-10-09
更新日期: 2026-10-09
类型: 复盘
状态: 待复核
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2026ByteDance_MERGE.pdf]]"
---

# 2026-10-09｜论文引文简读

## 底稿论文

- **论文**：MERGE: Next-Generation Item Indexing Paradigm for Large-Scale Streaming Recommendation
- **一句话主题**：MERGE 提出一种面向大规模流式推荐的自适应 item 索引范式，通过从零动态构建聚类、监控簇占用并以细到粗合并形成层次索引，缓解 VQ 方法在偏斜非平稳 item 分布下的分配不准、簇不均衡与簇分离不足问题。
- **References 总量**：36
- **近 5 年候选**：26
- **今日精选**：5

## 精选引文

### 1. Real-time Indexing for Large-scale Recommendation by Streaming Vector Quantization Retriever（2025）

- **来源**：KDD ’25, ACM, New York, NY, USA, 4273–4283
- **相关性**：直接对应源论文所批评的 VQ 单层聚类基线 StreamingVQ，是 item 索引与流式推荐最相关的对照工作。
- **核心思路**：用流式向量量化直接进行单层聚类，提升索引即时性，并替代主要在线召回器。
- **与源论文的关系**：源论文明确指出 StreamingVQ 直接采用 VQ 做单层聚类，并将其作为工业流式推荐中 VQ 索引的代表性方法；MERGE 正是针对其分配准确率、簇均衡与簇分离问题提出新范式。
- **研究价值**：提供生成式/判别式推荐中实时 item 索引的工业基线，有助于理解 VQ 索引在流式场景的局限与 MERGE 的改进动机。
- **历史被引**：2 篇底稿论文
- **处理状态**：普通简读
- **需要核实**：候选条目未给出具体实验指标、簇数量、在线 A/B 结果；若需引用其与 MERGE 的定量对比，需人工核实原文。

### 2. Trinity: Syncretizing Multi-/Long-Tail/Long-Term Interests All in One（2024）

- **来源**：KDD ’24, ACM, New York, NY, USA, 6095–6104
- **相关性**：使用 VQ 做两级聚类构建用户统计兴趣直方图，是源论文引用的 VQ 索引代表工作，涉及多兴趣、长尾与长期兴趣。
- **核心思路**：利用 VQ 进行两级聚类，构建用户统计兴趣直方图，统一建模多兴趣、长尾和长期兴趣。
- **与源论文的关系**：源论文将 Trinity 列为 VQ 两级聚类代表，说明 VQ 在推荐索引中的广泛应用；MERGE 则主张从零自适应构建层次索引，以应对 VQ 在偏斜分布下的不足。
- **研究价值**：展示 VQ 层次聚类在工业推荐兴趣建模中的落地方式，可作为 MERGE 层次合并索引的对比参照。
- **历史被引**：1 篇底稿论文
- **处理状态**：普通简读
- **需要核实**：候选条目未提供 Trinity 的具体聚类层数、码本规模、长尾指标；与 MERGE 的直接可比性需人工核实。

### 3. Recommender systems with generative retrieval（2023）

- **来源**：Advances in Neural Information Processing Systems, Vol. 36, Curran Associates, Inc.
- **相关性**：生成式检索推荐的代表性早期工作，与源论文讨论的生成式推荐 item 索引/语义 ID 直接相关。
- **核心思路**：将推荐检索建模为生成式任务，用生成模型直接生成目标 item 的标识，而非传统向量近邻检索。
- **与源论文的关系**：源论文指出生成式推荐依赖压缩 token 空间直接生成 item，item 索引决定下游模型性能上限；该文是这一方向的基础引用之一。
- **研究价值**：帮助理解生成式召回中 item 索引/标识为何关键，为 MERGE 作为基础索引范式提供问题背景。
- **历史被引**：13 篇底稿论文
- **处理状态**：高频引用，已尝试生成精读笔记
- **需要核实**：候选条目未给出其具体标识构造方式与实验设置；若需讨论其与 MERGE 的索引差异，需人工核实原文。

### 4. Learnable Item Tokenization for Generative Recommendation（2024）

- **来源**：CIKM ’24, ACM, New York, NY, USA, 2400–2409
- **相关性**：直接研究生成式推荐中的可学习 item tokenization，与源论文的 item indexing/tokenization 主题高度一致。
- **核心思路**：面向生成式推荐学习 item tokenization，使 item 到 token 的映射可训练并服务于生成式推荐模型。
- **与源论文的关系**：源论文将 item indexing 等同于 item tokenization，并强调其对生成式推荐性能上限的决定作用；该文是可学习 tokenization 的代表工作，可作为 MERGE 的对比方向。
- **研究价值**：提供可学习 tokenization 的研究视角，有助于评估 MERGE 自适应聚类索引与学习式 tokenization 的互补或替代关系。
- **历史被引**：4 篇底稿论文
- **处理状态**：高频引用，已尝试生成精读笔记
- **需要核实**：候选条目未说明其 token 层级、码本学习目标与评测数据集；与 MERGE 的公平对比需人工核实。

### 5. Generative Recommendation with Semantic IDs: A Practitioner’s Handbook（2025）

- **来源**：CIKM ’25, ACM, New York, NY, USA, 6420–6425
- **相关性**：面向语义 ID 的生成式推荐实践指南，直接覆盖语义 ID 与生成式推荐，与源论文的 item 索引/语义 ID 主题相关。
- **核心思路**：总结生成式推荐中语义 ID 的实践方法、经验与注意事项，面向从业者提供操作指南。
- **与源论文的关系**：源论文讨论 item 索引作为生成式推荐基础，语义 ID 是其中一种离散索引形式；该文可作为语义 ID 实践层面的参考。
- **研究价值**：有助于从工程实践角度理解语义 ID 的构建、评估与落地，为 MERGE 的工业部署价值提供背景。
- **历史被引**：2 篇底稿论文
- **处理状态**：普通简读
- **需要核实**：候选条目未给出具体语义 ID 构造算法与实验结论；若需引用其具体建议，需人工核实原文。

## 使用方式

- 这是自动生成的引文简读，先按相关性挑 1～2 篇核对原文。
- 确认有价值后，再升级成知识卡片或选题；本文件只作为每日线索。
- 如某条内容信息不足，不要直接引用其结论，先查原文。
