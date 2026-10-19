---
创建日期: 2026-10-19
更新日期: 2026-10-19
类型: 复盘
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2026ByteDance_TokenMixer-Large.pdf]]"
---

# 2026-10-19｜论文引文简读

## 底稿论文

- **论文**：TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders
- **一句话主题**：提出TokenMixer-Large架构，通过混合-还原操作、层间残差和辅助损失解决深层梯度问题，并引入稀疏Per-token MoE实现高效参数扩展，在字节跳动多个场景中实现在线7B、离线15B参数的规模，取得电商订单1.66%提升等收益。
- **References 总量**：44
- **近 5 年候选**：20
- **今日精选**：5

## 精选引文

### 1. Rankmixer: Scaling up ranking models in industrial recommenders（2025）

- **来源**：Proceedings of the 34th ACM International Conference on Information and Knowledge Management. 6309–6316.
- **相关性**：源论文直接基于RankMixer中的TokenMixer架构进行改进，是其前身工作。/5
- **核心思路**：用轻量级token混合操作替代自注意力，通过硬件感知协同设计大幅提升模型FLOPs利用率和扩展效率。
- **与源论文的关系**：源论文明确指出TokenMixer-Large是在RankMixer的TokenMixer架构基础上，针对其深层配置的残差路径、梯度消失、MoE稀疏化不完整等瓶颈进行系统性演进。
- **研究价值**：提供了生成式推荐中高效骨干网络的设计范式，是理解TokenMixer-Large改进动机和基线性能的关键。
- **历史被引**：15 篇底稿论文
- **已有精读**：[[RankMixer：token混合让推荐模型MFU提升十倍]]
- **需要核实**：需人工核实RankMixer论文中TokenMixer的具体实现细节与源论文所述瓶颈的对应关系。

### 2. Hiformer: Heterogeneous Feature Interactions Learning with Transformers for Recommender Systems（2023）

- **来源**：arXiv preprint arXiv:2311.05884 (2023).
- **相关性**：源论文在引言中将其列为现有架构之一，并指出其设计次优和硬件利用不足的问题。/5
- **核心思路**：使用异构注意力机制和低秩剪枝来建模工业级推荐中的特征交互。
- **与源论文的关系**：源论文将HiFormer作为对比基线，指出其未能充分考虑硬件对齐协同设计，导致性能次优。
- **研究价值**：帮助理解生成式推荐架构演进中硬件效率与模型设计的权衡。
- **历史被引**：5 篇底稿论文
- **已有精读**：[[Hiformer：用异构注意力与低秩剪枝实现工业级特征交互建模]]
- **需要核实**：需人工核实HiFormer的具体架构细节及其与TokenMixer-Large在实验中的对比结果。

### 3. Wukong: Towards a Scaling Law for Large-Scale Recommendation（2024）

- **来源**：In Forty-first International Conference on Machine Learning.
- **相关性**：源论文在引言中将其列为探索推荐模型缩放定律的代表性工作，并指出其设计次优。/5
- **核心思路**：通过堆叠因子分解机设计更精细的模型结构，并验证推荐模型随规模增长的缩放定律。
- **与源论文的关系**：源论文将Wukong作为现有缩放架构之一，指出其忽视硬件对齐协同设计导致性能次优，从而引出TokenMixer-Large的改进。
- **研究价值**：提供了推荐模型缩放定律的早期工业验证，是理解生成式推荐规模化挑战的重要背景。
- **历史被引**：18 篇底稿论文
- **已有精读**：[[堆叠因子分解机实现推荐系统缩放定律：Wukong 在 100+ GFLOPexample 下持续提升质量]]
- **需要核实**：需人工核实Wukong论文中缩放实验的具体设置和结论，以及源论文对其“次优设计”的具体批评依据。

### 4. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models（2024）

- **来源**：arXiv preprint arXiv:2401.06066 (2024).
- **相关性**：源论文引入稀疏Per-token MoE进行参数扩展，DeepSeekMoE的细粒度专家分割和共享专家隔离是相关技术。/5
- **核心思路**：通过细粒度专家分割和共享专家隔离，提升MoE语言模型的专家专业化程度和效率。
- **与源论文的关系**：源论文的Sparse Per-token MoE设计可能借鉴了DeepSeekMoE的细粒度专家和共享专家思想，用于高效参数扩展。
- **研究价值**：为生成式推荐中的MoE稀疏化设计提供参考，有助于理解TokenMixer-Large的MoE实现。
- **历史被引**：4 篇底稿论文
- **已有精读**：[[DeepSeekMoE：细粒度专家分割与共享专家隔离实现极致专家专业化]]
- **需要核实**：源论文未明确引用DeepSeekMoE，需人工核实其MoE设计是否直接受该工作影响。

### 5. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity（2022）

- **来源**：Journal of Machine Learning Research 23, 120 (2022), 1–39.
- **相关性**：源论文在附录中提及尝试了Switch Transformer的负载均衡方法，与MoE稀疏化直接相关。/5
- **核心思路**：提出基于Top-1路由的稀疏专家模型，实现万亿参数规模的高效训练。
- **与源论文的关系**：源论文在附录A.2中明确提到尝试了Switch Transformer的负载均衡方法，用于解决MoE负载均衡问题。
- **研究价值**：提供了MoE负载均衡的经典方法，是理解TokenMixer-Large中MoE训练稳定性的基础。
- **历史被引**：4 篇底稿论文
- **处理状态**：高频引用，已尝试生成精读笔记
- **需要核实**：需人工核实源论文中Switch Transformer负载均衡方法的具体应用效果和对比结果。

## 使用方式

- 这是自动生成的引文简读，先按相关性挑 1～2 篇核对原文。
- 确认有价值后，再升级成知识卡片或选题；本文件只作为每日线索。
- 如某条内容信息不足，不要直接引用其结论，先查原文。
