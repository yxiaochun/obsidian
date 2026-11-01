---
创建日期: 2026-11-01
更新日期: 2026-11-01
类型: 复盘
状态: 待复核
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2026Tencent_OneRanker.pdf]]"
---

# 2026-11-01｜论文引文简读

## 底稿论文

- **论文**：OneRanker: Unified Generation and Ranking with One Model in Industrial Advertising Recommendation
- **一句话主题**：OneRanker 提出用一个模型在工业广告推荐中统一生成与排序，通过价值感知多任务解耦架构、粗到细协同目标感知机制以及输入输出双侧一致性保证，解决兴趣目标与商业价值错位、生成过程目标无感知、生成与排序脱节三大挑战，并在腾讯微信广告系统全量部署取得 GMV 正向提升。
- **References 总量**：24
- **近 5 年候选**：18
- **今日精选**：5

## 精选引文

### 1. OneRec: Unifying retrieve and rank with generative recommender and iterative preference alignment（2025）

- **来源**：arXiv preprint arXiv:2502.18965
- **相关性**：生成式推荐统一召回与排序的代表性工作，与 OneRanker 统一生成与排序的目标直接相关。/5
- **核心思路**：用生成式推荐模型统一召回与排序阶段，并通过迭代偏好对齐优化最终目标。
- **与源论文的关系**：OneRanker 在广告场景进一步统一生成与排序，并引入价值感知与目标感知机制；OneRec 是其在推荐领域的重要前置范式。
- **研究价值**：提供端到端生成式推荐统一架构的基线思路，有助于理解生成与排序融合的动机与挑战。
- **历史被引**：14 篇底稿论文
- **已有精读**：[[OneRec：统一召回与排序的端到端生成式推荐]]
- **需要核实**：候选条目为 arXiv 预印本，需核实其最终发表 venue 与实验细节；与 OneRanker 的具体差异需结合原文确认。

### 2. GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation（2025）

- **来源**：arXiv preprint arXiv:2511.10138
- **相关性**：广告推荐领域的统一生成式预训练范式，与 OneRanker 的工业广告统一建模高度相关。/5
- **核心思路**：面向大规模广告推荐提出生成式预训练单模型范式，统一建模广告推荐任务。
- **与源论文的关系**：OneRanker 同样面向广告推荐统一生成与排序，GPR 可视为同场景下的并行/前置探索。
- **研究价值**：展示广告场景下生成式统一建模的可行性与工业实践，为 OneRanker 的架构设计提供对照。
- **历史被引**：2 篇底稿论文
- **已有精读**：[[GPR：广告推荐的统一生成式预训练范式]]
- **需要核实**：候选条目为 arXiv 预印本，需核实其技术细节与实验规模；与 OneRanker 的具体架构差异需人工比对。

### 3. GRank: Towards Target-Aware and Streamlined Industrial Retrieval with a Generate-Rank Framework（2025）

- **来源**：arXiv preprint arXiv:2510.15299
- **相关性**：目标感知的生成-排序框架，与 OneRanker 解决生成过程目标无感知、生成与排序脱节的问题直接相关。/5
- **核心思路**：提出生成-排序框架，使生成阶段具备目标感知能力，并简化工业检索流程。
- **与源论文的关系**：OneRanker 同样强调目标感知与生成排序协同，GRank 提供了目标感知生成排序的参考设计。
- **研究价值**：有助于理解如何在生成阶段注入排序目标信号，避免高价值候选被系统性过滤。
- **历史被引**：2 篇底稿论文
- **已有精读**：[[GRank：无结构索引的目标感知生成式检索]]
- **需要核实**：候选条目为 arXiv 预印本，需核实其目标感知机制的具体实现与工业部署情况。

### 4. Recommender systems with generative retrieval（2023）

- **来源**：Advances in Neural Information Processing Systems 36 (2023), 10299–10315
- **相关性**：生成式检索在推荐系统中的奠基性工作，涉及语义 ID 与生成式召回，是生成式推荐的重要基础。/5
- **核心思路**：利用生成式模型基于语义 ID 进行推荐检索，替代传统向量检索。
- **与源论文的关系**：OneRanker 的生成式推荐范式建立在语义 ID 与生成式检索基础之上，该文提供基础方法论。
- **研究价值**：提供生成式检索的核心思想与语义 ID 构建方法，是理解生成式推荐演进的必读工作。
- **历史被引**：14 篇底稿论文
- **处理状态**：高频引用，已尝试生成精读笔记
- **需要核实**：需核实其语义 ID 构造与 OneRanker 所用 token 化方案的关系；本地无精读笔记，建议人工精读。

### 5. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations（2024）

- **来源**：arXiv preprint arXiv:2402.17152
- **相关性**：万亿参数序列转换器用于生成式推荐，验证推荐领域缩放定律，与生成式推荐架构与规模化直接相关。/5
- **核心思路**：提出 HSTU 序列转换器替代 DLRM，在万亿参数规模下验证生成式推荐的缩放定律。
- **与源论文的关系**：OneRanker 属于生成式推荐范式，HSTU 提供了序列建模与规模化基础，是其架构演进的重要背景。
- **研究价值**：高被引且已有精读笔记，提供生成式推荐规模化与序列建模的关键证据，对理解 OneRanker 的模型设计有支撑价值。
- **历史被引**：26 篇底稿论文
- **已有精读**：[[生成式推荐：用万亿参数序列转换器 HSTU 替代 DLRM，验证推荐领域的缩放定律]]
- **需要核实**：候选条目为 arXiv 预印本，需核实其最终发表信息；与 OneRanker 在广告排序场景的具体关联需结合原文确认。

## 使用方式

- 这是自动生成的引文简读，先按相关性挑 1～2 篇核对原文。
- 确认有价值后，再升级成知识卡片或选题；本文件只作为每日线索。
- 如某条内容信息不足，不要直接引用其结论，先查原文。
