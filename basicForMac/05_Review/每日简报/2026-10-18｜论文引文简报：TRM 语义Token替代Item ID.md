---
创建日期: 2026-10-18
更新日期: 2026-10-18
类型: 复盘
状态: 待复核
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2026ByteDance_TRM.pdf]]"
---

# 2026-10-18｜论文引文简读

## 底稿论文

- **论文**：Farewell to Item IDs: Unlocking the Scaling Potential of Large Ranking Models via Semantic Tokens
- **一句话主题**：该论文提出TRM框架，用语义token替代item ID以提升大规模排序模型的可扩展性，通过融合用户行为信息、独立学习组合知识以及联合优化判别与生成目标，在稀疏存储减少33%的同时AUC提升0.85%，并已在个性化搜索引擎A/B测试中取得用户活跃天数和改查比提升。
- **References 总量**：10
- **近 5 年候选**：5
- **今日精选**：5

## 精选引文

### 1. Recommender Systems with Generative Retrieval（2023）

- **来源**：Advances in Neural Information Processing Systems, 2023. [OpenReview](https://openreview.net/forum?id=BJ0fQUU32w)
- **相关性**：生成式召回与语义ID的早期代表性工作，直接相关于语义token和生成式检索。/5
- **核心思路**：使用基于内容的语义ID（如层次化k-means聚类）作为物品表示，并通过生成式模型（如T5）自回归地生成目标物品的语义ID，实现生成式检索。
- **与源论文的关系**：源论文TRM同样采用语义token替代item ID，并指出当前语义token忽略用户行为域的问题；该文是语义ID用于生成式召回的奠基性工作之一，为TRM提供了语义token构建的参考。
- **研究价值**：提供了语义ID生成与生成式检索的基本范式，是研究语义token在推荐系统中应用的重要基线。
- **历史被引**：1 篇底稿论文
- **处理状态**：普通简读
- **需要核实**：候选条目中仅包含该文的引用信息，未提供具体实验细节和结论，需人工核实其语义ID构建方法和生成式检索效果。

### 2. Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations（2023）

- **来源**：arXiv preprint, 2023. doi: 10.48550/arXiv.2306.08121. [arXiv](https://arxiv.org/abs/2306.08121)
- **相关性**：语义ID在排序任务中的案例研究，与源论文主题高度相关。/5
- **核心思路**：在推荐排序中引入语义ID以提升泛化能力，通过将物品映射到语义ID来缓解冷启动和稀疏性问题。
- **与源论文的关系**：源论文TRM指出语义token相比item ID具有更好的扩展潜力，并针对现有语义token的不足提出改进；该文是语义ID在排序中应用的直接相关研究，为TRM提供了问题背景和对比基线。
- **研究价值**：展示了语义ID在排序任务中的泛化优势，为研究语义token替代item ID提供了实证支持。
- **历史被引**：1 篇底稿论文
- **处理状态**：普通简读
- **需要核实**：候选条目中仅包含引用信息，未提供具体方法细节和实验结果，需人工核实其语义ID生成方式和性能提升幅度。

### 3. Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID（2025）

- **来源**：arXiv preprint, 2025. doi: 10.48550/arXiv.2504.02137. [arXiv](https://arxiv.org/abs/2504.02137)
- **相关性**：语义ID用于增强嵌入表示稳定性，与源论文关注ID嵌入不稳定问题直接相关。/5
- **核心思路**：利用语义ID来增强推荐系统中嵌入表示的稳定性，减少因物品动态变化导致的嵌入训练不稳定问题。
- **与源论文的关系**：源论文TRM的核心动机之一是item ID嵌入的不稳定性阻碍了模型扩展，而该文同样关注语义ID对嵌入稳定性的提升，为TRM提供了稳定性方面的支持证据。
- **研究价值**：提供了语义ID提升嵌入稳定性的方法，有助于理解语义token在训练稳定性方面的优势。
- **历史被引**：1 篇底稿论文
- **处理状态**：普通简读
- **需要核实**：候选条目中仅包含引用信息，未提供具体技术方案和实验数据，需人工核实其稳定性提升机制和效果。

### 4. OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment（2025）

- **来源**：arXiv preprint, 2025. doi: 10.48550/arXiv.2502.18965. [arXiv](https://arxiv.org/abs/2502.18965)
- **相关性**：生成式推荐系统，统一召回与排序，与源论文的生成式目标相关。/5
- **核心思路**：提出OneRec，使用生成式推荐模型统一召回和排序阶段，并通过迭代偏好对齐提升推荐效果。
- **与源论文的关系**：源论文TRM在训练框架中联合优化判别与生成目标，OneRec作为生成式推荐的端到端统一框架，为TRM的生成式目标提供了参考和对比。
- **研究价值**：展示了生成式推荐在工业级系统中的可行性和效果，是研究生成式推荐统一架构的重要案例。
- **历史被引**：1 篇底稿论文
- **已有精读**：[[OneRec：统一召回与排序的端到端生成式推荐]]
- **需要核实**：候选条目中仅包含引用信息，且已有精读笔记，但具体技术细节需参考笔记或原文核实。

### 5. RankMixer: Scaling Up Ranking Models in Industrial Recommenders（2025）

- **来源**：Proceedings of the 34th ACM International Conference on Information and Knowledge Management, pp. 6309–6316, 2025.
- **相关性**：大规模排序模型扩展，与源论文的缩放潜力主题相关。/5
- **核心思路**：提出RankMixer，通过token混合和模型架构设计实现工业推荐系统中排序模型的大规模扩展，提升模型MFU和效果。
- **与源论文的关系**：源论文TRM旨在释放大规模排序模型的扩展潜力，RankMixer是同期关于排序模型扩展的代表性工作，两者在扩展目标上一致，但TRM从特征表示（语义token）角度切入，而RankMixer侧重架构。
- **研究价值**：提供了排序模型扩展的另一种路径（架构层面），与TRM的语义token路径形成互补，有助于全面理解推荐模型缩放。
- **历史被引**：1 篇底稿论文
- **已有精读**：[[RankMixer：token混合让推荐模型MFU提升十倍]]
- **需要核实**：候选条目中仅包含引用信息，未提供具体实验数据和结论，需人工核实其扩展方法和效果。

## 使用方式

- 这是自动生成的引文简读，先按相关性挑 1～2 篇核对原文。
- 确认有价值后，再升级成知识卡片或选题；本文件只作为每日线索。
- 如某条内容信息不足，不要直接引用其结论，先查原文。
