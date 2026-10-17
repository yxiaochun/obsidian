---
创建日期: 2026-10-12
更新日期: 2026-10-12
类型: 复盘
状态: 待复核
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2026ByteDance_MixFormer.pdf]]"
---

# 2026-10-12｜论文引文简读

## 底稿论文

- **论文**：MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders
- **一句话主题**：MixFormer 提出一种统一的 Transformer 风格推荐架构，在单一骨干中联合建模用户行为序列与稠密特征交互，实现稠密容量与序列长度的协同扩展，并通过用户-物品解耦策略优化工业推理效率。
- **References 总量**：29
- **近 5 年候选**：16
- **今日精选**：5

## 精选引文

### 1. Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations（2024）

- **来源**：ICML 2024, PMLR 235, 58484–58509, [proceedings.mlr.press/v235/zhai24a.html](https://proceedings.mlr.press/v235/zhai24a.html)
- **相关性**：生成式推荐与推荐缩放定律的代表性工作，提出 HSTU 架构替代 DLRM，验证推荐领域的缩放规律。/5
- **核心思路**：使用万亿参数级序列转换器 HSTU 对用户行为序列进行生成式建模，替代传统 DLRM 架构，并系统验证推荐系统中的缩放定律。
- **与源论文的关系**：MixFormer 同样关注工业推荐中序列建模与模型容量的协同扩展；HSTU 是生成式推荐缩放方向的重要基线，与 MixFormer 的 co-scaling 动机直接相关。
- **研究价值**：为生成式推荐提供大规模序列建模与缩放定律的实证基础，是理解 MixFormer 所面对 scaling-driven regime 的关键参考文献。
- **历史被引**：22 篇底稿论文
- **已有精读**：[[生成式推荐：用万亿参数序列转换器 HSTU 替代 DLRM，验证推荐领域的缩放定律]]
- **需要核实**：候选条目仅提供标题、作者、会议与链接，未提供 HSTU 的具体架构细节、实验数据或与 MixFormer 的直接对比结果，需人工核实原文。

### 2. OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment（2025）

- **来源**：arXiv:2502.18965 [cs.IR], [arXiv:2502.18965](https://arxiv.org/abs/2502.18965)
- **相关性**：生成式推荐中统一召回与排序的端到端范式，与生成式召回/排序主题高度相关。/5
- **核心思路**：用生成式推荐模型统一召回与排序，并通过迭代偏好对齐提升推荐效果。
- **与源论文的关系**：MixFormer 强调在单一骨干中统一序列建模与特征交互；OneRec 则从生成式推荐角度统一召回与排序，两者都体现推荐架构统一化与端到端化的趋势。
- **研究价值**：为生成式推荐提供统一召回排序的工业级思路，有助于理解统一架构在推荐系统中的价值与挑战。
- **历史被引**：11 篇底稿论文
- **已有精读**：[[OneRec：统一召回与排序的端到端生成式推荐]]
- **需要核实**：候选条目未提供 OneRec 的具体模型结构、训练目标、线上指标或与 MixFormer 的关联细节，需人工核实原文。

### 3. MTGR: Industrial-Scale Generative Recommendation Framework in Meituan（2025）

- **来源**：CIKM ’25, ACM, 5731–5738, [doi:10.1145/3746252.3761565](https://doi.org/10.1145/3746252.3761565)
- **相关性**：工业级生成式推荐框架，强调保留交叉特征，与生成式推荐和工业落地主题相关。/5
- **核心思路**：在工业级生成式推荐框架中保留交叉特征，以兼顾生成式建模与特征交互能力。
- **与源论文的关系**：MixFormer 的核心动机之一是解决序列建模与稠密特征交互分离导致的协同扩展问题；MTGR 同样关注生成式推荐中特征交叉的保留，二者在工业生成式推荐的特征交互处理上具有可比性。
- **研究价值**：为生成式推荐在工业场景中如何保留和利用交叉特征提供参考，有助于评估统一架构与生成式架构的取舍。
- **历史被引**：9 篇底稿论文
- **已有精读**：[[MTGR：保留交叉特征的工业级生成式推荐扩展]]
- **需要核实**：候选条目未提供 MTGR 的具体架构、特征交叉实现方式、实验数据或与 MixFormer 的直接比较，需人工核实原文。

### 4. OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender（2025）

- **来源**：arXiv:2510.26104 [cs.IR], [arXiv:2510.26104](https://arxiv.org/abs/2510.26104)
- **相关性**：用一个 Transformer 统一特征交互与序列建模，与 MixFormer 的统一骨干和协同扩展主题高度相关。/5
- **核心思路**：在工业推荐中用一个 Transformer 同时完成特征交互与序列建模，避免二者分离。
- **与源论文的关系**：MixFormer 同样主张在单一骨干中联合建模序列行为与特征交互，OneTrans 是同一统一化方向上的直接相关工作，可作为对比基线或动机支撑。
- **研究价值**：为统一特征交互与序列建模提供具体架构参考，有助于理解 MixFormer 在统一化设计上的定位与差异。
- **历史被引**：4 篇底稿论文
- **已有精读**：[[OneTrans：一个Transformer统一特征交互与序列建模]]
- **需要核实**：候选条目未提供 OneTrans 的具体实现、实验设置、线上效果或与 MixFormer 的差异分析，需人工核实原文。

### 5. RankMixer: Scaling Up Ranking Models in Industrial Recommenders（2025）

- **来源**：CIKM ’25, ACM, 6309–6316, [doi:10.1145/3746252.3761507](https://doi.org/10.1145/3746252.3761507)
- **相关性**：工业推荐排序模型的扩展工作，提出 token 混合提升 MFU，与推荐模型缩放和工业效率主题相关。/5
- **核心思路**：通过 token 混合机制提升工业推荐排序模型的硬件利用率（MFU），实现排序模型的有效扩展。
- **与源论文的关系**：MixFormer 关注在有限计算预算下协同扩展稠密容量与序列长度，并强调工业效率优化；RankMixer 同样聚焦工业推荐模型的扩展与效率，二者在 scaling 与效率权衡上相关。
- **研究价值**：为工业推荐模型扩展提供 token 混合与效率优化思路，有助于理解 MixFormer 在效率优化方面的定位。
- **历史被引**：15 篇底稿论文
- **已有精读**：[[RankMixer：token混合让推荐模型MFU提升十倍]]
- **需要核实**：候选条目未提供 RankMixer 的具体 token 混合机制、MFU 数据、实验设置或与 MixFormer 的直接对比，需人工核实原文。

## 使用方式

- 这是自动生成的引文简读，先按相关性挑 1～2 篇核对原文。
- 确认有价值后，再升级成知识卡片或选题；本文件只作为每日线索。
- 如某条内容信息不足，不要直接引用其结论，先查原文。
