---
创建日期: 2026-09-27
更新日期: 2026-09-27
类型: 复盘
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2026Alibaba_SORT.pdf]]"
---

# 2026-09-27｜论文引文简读

## 底稿论文

- **论文**：SORT: A Systematically Optimized Ranking Transformer for Industrial-scale Recommenders
- **一句话主题**：SORT 是面向工业级推荐系统的系统化优化排序 Transformer，通过请求中心样本组织、局部注意力、查询剪枝与生成式预训练解决高特征稀疏与低标签密度问题，并优化 tokenization、MHA、FFN 与训练系统，在 AliExpress 上线后订单 +7.47%、买家 +6.67%、GMV +8.65%，延迟降低 62%、吞吐提升 589%。
- **References 总量**：32
- **近 5 年候选**：21
- **今日精选**：5

## 精选引文

### 1. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations（2024）

- **来源**：arXiv preprint arXiv:2402.17152
- **相关性**：生成式推荐、HSTU、推荐缩放定律/5
- **核心思路**：提出 HSTU 序列转换器，将推荐问题转化为生成式序列建模，用万亿参数验证推荐领域的缩放定律，替代传统 DLRM。
- **与源论文的关系**：SORT 同样以 Transformer 重建工业排序模型，并强调生成式预训练与可扩展性；HSTU 是生成式推荐方向的奠基性工作，为 SORT 的生成式预训练与缩放验证提供直接参照。
- **研究价值**：提供生成式推荐替代判别式排序的范式依据，以及推荐领域缩放定律的早期证据，是理解 SORT 生成式预训练动机的关键文献。
- **历史被引**：24 篇底稿论文
- **已有精读**：[[生成式推荐：用万亿参数序列转换器 HSTU 替代 DLRM，验证推荐领域的缩放定律]]
- **需要核实**：候选条目未给出具体实验设置与工业部署细节，需人工核实其与 SORT 在样本组织、注意力机制上的具体差异。

### 2. Wukong: Towards a scaling law for large-scale recommendation（2024）

- **来源**：arXiv preprint arXiv:2403.02545
- **相关性**：推荐缩放定律、大规模排序模型/5
- **核心思路**：通过堆叠因子分解机等结构探索大规模推荐模型的缩放定律，在 100+ GFLOP/example 下持续提升质量。
- **与源论文的关系**：SORT 强调在数据规模、模型规模与序列长度上的可扩展性，Wukong 的缩放定律研究为 SORT 的规模化设计提供理论背景与对照。
- **研究价值**：帮助理解工业推荐模型如何随规模增长而持续获益，是评估 SORT 缩放表现的重要参照。
- **历史被引**：18 篇底稿论文
- **已有精读**：[[堆叠因子分解机实现推荐系统缩放定律：Wukong 在 100+ GFLOPexample 下持续提升质量]]
- **需要核实**：候选条目未说明其是否采用生成式目标，需人工核实其与 SORT 在训练目标与架构上的可比性。

### 3. MTGR: Industrial-scale generative recommendation framework in Meituan（2025）

- **来源**：Proceedings of the 34th ACM International Conference on Information and Knowledge Management, 5731–5738
- **相关性**：工业级生成式推荐、交叉特征保留/5
- **核心思路**：提出工业级生成式推荐框架 MTGR，在生成式架构中保留交叉特征，实现规模化扩展。
- **与源论文的关系**：SORT 同样面向工业级推荐，并强调灵活集成多样特征；MTGR 的交叉特征保留思路与 SORT 的特征集成优化可相互对照。
- **研究价值**：提供工业级生成式推荐在特征处理与扩展性上的落地经验，对 SORT 的工业部署与特征集成研究有直接参考价值。
- **历史被引**：12 篇底稿论文
- **已有精读**：[[MTGR：保留交叉特征的工业级生成式推荐扩展]]
- **需要核实**：候选条目未给出 MTGR 与 SORT 在注意力、tokenization 上的具体差异，需人工核实。

### 4. Rankmixer: Scaling up ranking models in industrial recommenders（2025）

- **来源**：Proceedings of the 34th ACM International Conference on Information and Knowledge Management, 6309–6316
- **相关性**：工业排序模型、token 混合、MFU 优化/5
- **核心思路**：通过 token 混合提升工业推荐排序模型的 MFU，实现排序模型的大规模扩展。
- **与源论文的关系**：SORT 同样关注工业排序 Transformer 的硬件效率与 MFU 提升，RankMixer 的 token 混合与 MFU 优化是 SORT 训练系统优化的重要对照。
- **研究价值**：为工业排序模型的效率优化与规模化提供直接经验，有助于理解 SORT 在 MFU 45% 与吞吐提升上的技术定位。
- **历史被引**：15 篇底稿论文
- **已有精读**：[[RankMixer：token混合让推荐模型MFU提升十倍]]
- **需要核实**：候选条目未说明其与 SORT 在注意力与 FFN 优化上的具体差异，需人工核实。

### 5. Scaling transformers for discriminative recommendation via generative pretraining（2025）

- **来源**：Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, 2893–2903
- **相关性**：生成式预训练、判别式推荐、Transformer 扩展/5
- **核心思路**：通过生成式预训练扩展判别式推荐中的 Transformer，缓解稀疏监督下的表示学习问题。
- **与源论文的关系**：SORT 明确采用生成式预训练作为核心优化之一，该文与 SORT 同属生成式预训练增强判别式排序的路线，且作者团队高度重合，是 SORT 的直接前序工作。
- **研究价值**：为理解 SORT 生成式预训练的设计动机与效果提供直接基础，是生成式推荐预训练方向的关键文献。
- **历史被引**：8 篇底稿论文
- **处理状态**：高频引用，已尝试生成精读笔记
- **需要核实**：候选条目未给出具体预训练任务与 SORT 的差异，需人工核实其与 SORT 的继承关系及实验对比。

## 使用方式

- 这是自动生成的引文简读，先按相关性挑 1～2 篇核对原文。
- 确认有价值后，再升级成知识卡片或选题；本文件只作为每日线索。
- 如某条内容信息不足，不要直接引用其结论，先查原文。
