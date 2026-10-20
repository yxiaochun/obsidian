---
创建日期: 2026-10-20
更新日期: 2026-10-20
类型: 复盘
状态: 待复核
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2026ByteDance_UGSep.pdf]]"
---

# 2026-10-20｜论文引文简读

## 底稿论文

- **论文**：Compute Only Once: UG-Separated TokenMixer for Efficient Large Recommendation Models
- **一句话主题**：提出 UG-Sep 框架，在 TokenMixer 类稠密特征交互模型中显式分离用户侧与物品侧信息流，使部分 token 保持纯用户表示从而支持跨样本复用，并辅以信息补偿与 W8A16 量化，在字节多个业务场景将推理延迟降低最多 20%。
- **References 总量**：34
- **近 5 年候选**：25
- **今日精选**：5

## 精选引文

### 1. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations（2024）

- **来源**：arXiv preprint arXiv:2402.17152
- **相关性**：生成式推荐奠基性工作（HSTU），与源论文同属大规模推荐模型缩放方向，且源论文引言引用其作为长序列建模代表。/5
- **核心思路**：用万亿参数序列转换器（HSTU）替代传统 DLRM，将推荐建模为序列生成任务，验证推荐领域的缩放定律。
- **与源论文的关系**：源论文聚焦 TokenMixer 类稠密交互模型的推理效率，HSTU 代表生成式序列建模范式；两者均追求大规模推荐模型的可扩展性，但源论文解决的是用户侧计算复用问题。
- **研究价值**：提供生成式推荐缩放定律的实证基础，是理解大规模推荐模型效率与效果权衡的关键参照。
- **历史被引**：24 篇底稿论文
- **已有精读**：[[生成式推荐：用万亿参数序列转换器 HSTU 替代 DLRM，验证推荐领域的缩放定律]]
- **需要核实**：需核实 HSTU 的具体架构细节与实验设置是否与源论文的 TokenMixer 场景可比。

### 2. Rankmixer: Scaling up ranking models in industrial recommenders（2025）

- **来源**：Proceedings of the 34th ACM International Conference on Information and Knowledge Management. 6309–6316
- **相关性**：源论文直接针对的 TokenMixer 基线架构，属于大规模稠密特征交互推荐模型。/5
- **核心思路**：通过 token 混合机制提升推荐模型的计算效率（MFU），实现工业级排序模型缩放。
- **与源论文的关系**：源论文明确指出 RankMixer 是 TokenMixer 类稠密交互架构的代表，UG-Sep 旨在解决其用户-物品特征深度纠缠导致无法复用用户侧计算的问题。
- **研究价值**：理解 TokenMixer 架构的混合机制与效率瓶颈，是评估 UG-Sep 改进效果的直接基线。
- **历史被引**：15 篇底稿论文
- **已有精读**：[[RankMixer：token混合让推荐模型MFU提升十倍]]
- **需要核实**：需核实 RankMixer 原文中 token 混合的具体实现与源论文所述纠缠问题的对应关系。

### 3. TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders（2026）

- **来源**：arXiv preprint arXiv:2602.06563
- **相关性**：源论文直接针对的另一 TokenMixer 基线，代表工业级大规模排序模型的最新缩放实践。/5
- **核心思路**：将 TokenMixer 架构扩展至七十亿参数规模，实现在线排序模型的大规模部署。
- **与源论文的关系**：源论文将 TokenMixer-Large 列为稠密特征交互架构的代表，UG-Sep 通过分离用户-物品信息流解决其推理成本问题。
- **研究价值**：提供超大规模 TokenMixer 模型的工业落地经验，是研究大模型推理效率优化的直接对象。
- **历史被引**：3 篇底稿论文
- **已有精读**：[[TokenMixer-Large：七十亿参数在线排序模型]]
- **需要核实**：需核实 TokenMixer-Large 的参数量、部署场景与源论文实验设置的一致性。

### 4. Longer: Scaling up long sequence modeling in industrial recommenders（2025）

- **来源**：Proceedings of the Nineteenth ACM Conference on Recommender Systems. 247–256
- **相关性**：源论文引言中作为长序列建模代表引用，其 KV-Cache 复用机制是 UG-Sep 试图在 TokenMixer 中复现的能力。/5
- **核心思路**：通过全局 token 稳定超长行为序列建模，并利用 KV-Cache 实现用户侧历史序列计算的复用。
- **与源论文的关系**：源论文指出 LONGER 等长序列模型可通过 KV-Cache 复用用户侧计算，而 TokenMixer 类模型难以复用；UG-Sep 首次在 TokenMixer 中实现类似复用。
- **研究价值**：提供用户侧计算复用的成熟范式，是理解 UG-Sep 设计动机与对比优势的关键参照。
- **历史被引**：12 篇底稿论文
- **已有精读**：[[LONGER：全局token稳定超长行为序列建模]]
- **需要核实**：需核实 LONGER 的 KV-Cache 复用机制与 UG-Sep 的 token 分离复用方式在计算图上的差异。

### 5. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment（2025）

- **来源**：arXiv preprint arXiv:2502.18965
- **相关性**：生成式推荐代表性工作，统一召回与排序，与源论文同属生成式推荐方向。/5
- **核心思路**：用生成式推荐模型统一召回与排序，并通过迭代偏好对齐提升效果。
- **与源论文的关系**：源论文聚焦 TokenMixer 类判别式稠密交互模型的效率，OneRec 代表生成式统一建模范式；两者均面向大规模推荐系统的可扩展性。
- **研究价值**：提供生成式推荐统一架构的工业实践，是研究生成式范式与稠密交互范式效率对比的重要参考。
- **历史被引**：13 篇底稿论文
- **已有精读**：[[OneRec：统一召回与排序的端到端生成式推荐]]
- **需要核实**：需核实 OneRec 的模型规模、推理成本与源论文实验场景的可比性。

## 使用方式

- 这是自动生成的引文简读，先按相关性挑 1～2 篇核对原文。
- 确认有价值后，再升级成知识卡片或选题；本文件只作为每日线索。
- 如某条内容信息不足，不要直接引用其结论，先查原文。
