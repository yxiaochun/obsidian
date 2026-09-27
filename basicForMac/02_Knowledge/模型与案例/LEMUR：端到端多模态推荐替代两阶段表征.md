---
创建日期: 2026-09-15
更新日期: 2026-09-26
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2026ByteDance_LEMUR.pdf]]"
状态: 待复核
证据强度: 单篇工业论文证据
待验证问题:
  - "多模态表征在非搜索、非短视频场景的冷启动与长尾泛化如何？"
---

# LEMUR：端到端多模态推荐替代两阶段表征

> [!info] 一句话创新点
> LEMUR 把原始 query/document 的多模态编码、session-masked 对齐损失和 RankMixer 排序目标放进同一条可反传训练管线，再用 Memory Bank、去重和采样把历史多模态表征的计算量压到工业可部署规模。

## 论文信息

- **标题**：LEMUR: Large scale End-to-end MUltimodal Recommendation
- **作者/机构**：Xintian Han、Honggang Chen、Quan Lin 等，ByteDance
- **版本**：arXiv:2511.10962v2，2025-11-17；本地来源为 [[2026ByteDance_LEMUR.pdf]]
- **公开链接**：[arXiv:2511.10962](https://arxiv.org/abs/2511.10962)
- **领域**：工业推荐系统、多模态推荐、端到端训练
- **与研究方向的关系**：为「去两阶段、让内容表征直接受推荐目标监督」提供了一例已部署证据；它不是离散 token 生成式召回，但与 [[生成式推荐为何开始替代级联管线]] 关注的端到端化趋势相关。

## 核心问题

ID-based 推荐冷启动和泛化差；现有多模态方案常先预训练多模态模型，再冻结表征喂给推荐模型。这带来四类错位：内容表征缺少行为监督、推荐模型偏向 ID 表征、多模态模型与推荐模型更新频率不一致、长序列表征作为在线服务传输成本高。

## 七问笔记

| 问题 | 回答 |
| --- | --- |
| 研究背景 | 抖音搜索等工业系统依赖大量 ID 特征，新内容和新用户缺少行为积累；两阶段多模态表征与下游 CTR 目标脱节。 |
| 研究目的 | 从原始 query、user 和 video/document 特征端到端预测抖音搜索 CTR，让多模态编码器直接受排序目标监督，并支持实时参数更新。 |
| 创新点 | 联合训练多模态编码器与 RankMixer，配合 Memory Bank 缓存历史表征、SQDC 对齐 query-document、批内去重与采样降低成本。 |
| 研究方法 | query/document 分别经双向 Transformer 编码，表征与其他特征一起进入 RankMixer；SQDC 用真实点击作正样本并屏蔽同一 query session 内样本；历史表征按文档 ID 从 Memory Bank 取出，经改写版 LONGER Decoder 和 cosine similarity 模块建模。 |
| 实验数据 | 离线主实验来自抖音搜索日志，连续 70 天、约 30 亿样本，包含数十亿用户和数亿文档；文本输入含 query、标题、OCR、ASR 和封面 OCR。线上主实验为抖音搜索 14 天 A/B。 |
| 结果结论 | 相对 RankMixer+LONGER，AUC +0.55%，QAUC +0.81%；两阶段冻结表征仅 QAUC +0.12%，LEMUR 为 +0.81%。抖音搜索 query change rate decay 减少 0.843%，T 检验显著，全量部署超过一个月。 |
| 总体评价 | 工业证据强，消融能区分联合训练、Memory Bank、SQDC 和 session mask 的贡献；但数据/代码未公开，结论集中在字节系搜索与短视频场景，Memory Bank 的长期 staleness 和长尾覆盖仍需验证。 |

## 方法拆解

### 端到端训练

- **目标**：抖音搜索 CTR，定义为用户提交 query 后观看推荐视频超过 5 秒；训练用二元交叉熵。
- **编码**：query raw features 和 document raw features 分别输入双向 Transformer。document 文本包含标题、OCR、ASR、封面 OCR 等；输出表征与其他 ID、cross 和 user 特征一起送入 RankMixer。
- **SQDC**：Session-masked Query to Document Contrastive。批内用真实点击形成正样本，用 cosine similarity 和温度 $T$ 计算对比损失；同一 QID 的其他样本在分母中被 mask。温度 50 最优。
- **为什么不加 CIC**：CIC 原本用于对齐冻结多模态表征与 ID 表征；当多模态编码器已经和 RankMixer 联合更新时，显式对齐 ID 不再带来增益，消融中反而使 QAUC 下降 0.02%。

### Memory Bank 与序列建模

- 训练时先存储当前目标文档表征，再按用户历史文档 ID 取回表征，形成多模态历史序列。
- 用户历史窗口为一个月，训练数据跨度超过两个月，因此 Memory Bank 能覆盖序列中的文档。
- 序列模型使用改写版 LONGER Decoder：全局 token 作为 query，对历史表征做 cross-attention，再过 FFN；另用 cosine similarity 计算目标文档与历史文档相似性，并维护排序后的相似性向量。模型中有多条序列，最长 1000 条。
- 批大小 2048 时，基础模型约需 1.6 TFLOPs，document Transformer 约需 2.3 TFLOPs；若每个历史文档都重新编码，长序列成本不可行。
- 部署版只对 20% 样本执行 Transformer forward/backward，其余直接复用 Memory Bank 表征；推理时不激活 Transformer，直接取表征。
- 去重通过 `all_gather` 把某个 rank 新算的文档表征广播给其他 rank，再用自研 CUDA `Dedup Join` 算子扫描当前批内历史序列，把同一文档的旧表征替换成最新表征。

### 关键图表

| 图表 | 解读 |
| --- | --- |
| Figure 1 | 左侧是端到端管线：多模态 query/document 编码、Memory Bank、多模态序列建模和 RankMixer 共享训练通路；右侧是 SQDC 的同 query session mask。 |
| Figure 2 | Decoder 只保留 cross-attention 加 FFN，不逐历史项做自注意力，这是控制长序列成本的关键。 |
| Figure 3 | Memory Bank 当前模型输出与缓存表征的相似度随训练上升并稳定在约 0.95；序列覆盖超过 90%，当前文档覆盖超过 98%。这说明 staleness 在实验窗口内可控，但不等于长期训练也可控。 |

## 实验证据

### 离线主实验

| 模型 / 增量 | AUC | QAUC |
| --- | ---: | ---: |
| DLRM-MLP | 0.74071 | 0.63760 |
| +RankMixer | +0.49% | +0.59% |
| +LONGER | +0.89% | +1.04% |
| +LEMUR-SQDC | +1.22% | +1.51% |
| +LEMUR-SQDC-MB | +1.44% | +1.85% |
| 相对 RankMixer+LONGER 的最终提升 | +0.55% | +0.81% |

### 两阶段对照

| 方法 | QAUC | ΔQAUC |
| --- | ---: | ---: |
| baseline | 0.64393 | - |
| Two-Stage | 0.64470 | +0.12% |
| LEMUR | 0.64914 | +0.81% |

两阶段对照使用相同 SQDC 损失、文本特征和一个月抖音搜索数据预训练 Transformer，再冻结表征给下游排序任务；LEMUR 超出它 0.69% QAUC。

### 消融

| 消融 | QAUC | ΔQAUC |
| --- | ---: | ---: |
| LEMUR final online version | +0.81% | - |
| Transformer 加 stop-gradient | +0.50% | -0.31% |
| 不用短多模态序列 | +0.77% | -0.09% |
| 不用长多模态序列 | +0.56% | -0.25% |
| 不用 cosine similarity | +0.69% | -0.12% |
| 不用 SQDC loss | +0.64% | -0.17% |
| 不用 session-level mask | +0.71% | -0.10% |
| 加 CIC loss | +0.79% | -0.02% |

### 效率与采样

| 方法 | 参数 | FLOPs | QAUC |
| --- | ---: | ---: | ---: |
| baseline | 92M | 1673G | - |
| LEMUR, no-sampling | 139M | 4303G | +0.81% |
| LEMUR, p=100, q=20 | 139M | 3232G | +0.78% |
| LEMUR, p=20, q=20 | 139M | 2439G | +0.76% |
| LEMUR, p=10, q=10 | 139M | 2262G | +0.75% |

`p` 是 forward 样本比例，`q` 是 backward 样本比例。最终部署选择 `p=20/q=20`：比不采样少 0.05% QAUC，但 FLOPs 从 4303G 降到 2439G，且比 `p=10/q=10` 保留更好的长序列覆盖。

## 线上与其他场景

- **抖音搜索**：14 天 A/B，query change rate decay 减少 0.843%，T 检验显著；随后全量部署超过一个月，QAUC +0.81%。
- **抖音广告平台**：离线 AUC +0.1%。
- **TikTok Search**：在线 A/B 约实现 0.843% query change rate 下降。
- **query change rate**：定义为发生 query reformulation 的 distinct UID-query pair 数除以全部 distinct UID-query pair 数，作为负向搜索体验的代理指标。

## 批判性分析

> [!warning] 证据边界
> 论文最强的地方是工业规模、真实部署和系统消融；最需要保留怀疑的地方是外推范围和复现条件。

- **为什么说「端到端」仍带工程折衷**：多模态编码器和排序目标确实联合优化，但多数历史表征来自缓存，不逐 step 反传，缓存表征也不会随每次参数更新刷新。这不是完全严格的梯度穿透全序列，而是用 staleness/覆盖约束换取可训练性。
- **对照是否公平**：两阶段对照使用相同 SQDC、文本特征和数据，比笼统对比通用多模态预训练更有说服力；但所有主实验都来自字节内部系统，缺少公开数据集和第三方复现。
- **实验遗漏**：论文没有单独报告新 item、新用户、长尾 item 或非搜索场景的冷启动收益；抖音广告只有离线 AUC，TikTok Search 只报 query change rate，没有展示转化或收入指标。
- **指标选择**：AUC/QAUC 衡量排序质量，query change rate 更贴近搜索意图没被满足的用户行为信号；但 query change rate 也可能受界面、查询建议和流量分布影响，单指标不足以刻画体验。
- **如果改进实验**：可补充按 item 年龄/曝光频次分层的冷启动指标、Memory Bank 更新延迟的因果消融、广告与电商场景的长期在线收益，以及不同序列长度下的质量-成本曲线。

## 局限与开放问题

- 论文明确承认 Memory bank 的 staleness：缓存表征不会随训练参数实时更新，会与当前模型输出产生偏差，可能影响优化和序列模式学习。
- Memory bank 的覆盖与更新策略可能无法充分覆盖大规模系统中的长尾 item，作者担心模型偏向热门内容；未来方向包括动态更新计划和更复杂的缓存结构。
- 训练采样有效率权衡：p=100/q=20 使 QAUC 下降 0.03%，p=20/q=20 下降 0.05%，p=10/q=10 再额外下降 0.01%；最终为 ROI 和覆盖选择 20% 采样。
- 对比损失超参数敏感，过高或过低的温度都会降低效果；在联合训练框架下加入 CIC loss 反而带来轻微性能退化。

## 值得追踪的引用

- [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551)：LEMUR 的排序主干。
- [LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders](https://arxiv.org/abs/2505.04442)：LEMUR 的长序列基线和 Decoder 改写来源。
- [End-to-end training of multimodal model and ranking model](https://arxiv.org/abs/2404.06078)：两阶段/联合训练对照和 CIC loss 的来源。
- [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)：TIGER 的生成式检索范式，便于对比离散 token 与连续多模态表征路线。
- [QARM: Quantitative alignment multi-modal recommendation at Kuaishou](https://arxiv.org/abs/2411.11739)：另一家短视频平台的模态表征对齐路线。

## 术语与句式

- **End-to-end multimodal recommendation**：从原始模态特征训练表征，并让表征直接受推荐/排序目标监督。
- **Memory Bank**：按文档 ID 存取历史多模态表征的缓存机制；本文用它避免重复编码长用户历史。
- **SQDC**：in-batch query-document contrastive，加上同一 query session 的负样本 mask。
- **Query change rate**：用户把原查询改写成更具体查询的比例，可作为搜索结果未满足需求的代理信号。
- **可复用表述**：「多模态学习目标与推荐目标错位」「缓存表征的 staleness 与长尾覆盖」「用采样和去重把端到端训练压到工业可部署成本」。

## 复现清单

| 项目 | 论文给出的条件 | 复现缺口 |
| --- | --- | --- |
| 数据 | 抖音搜索连续 70 天、30 亿样本交互日志；需要 query、用户、视频/document、搜索历史、cross features 和「观看超过 5 秒」标签。 | 数据未公开。 |
| 模型 | 双向 Transformer、RankMixer、改写版 LONGER Decoder、cosine similarity 模块。 | 未提供完整网络配置和开源实现。 |
| 关键超参 | batch size 2048；Memory Bank 中 query/document 表征维度 128；query Transformer 2 层，document Transformer 4 层；SQDC 温度 50；最长序列 1000。 | 学习率、优化器、分布式并行和训练时长未完整披露。 |
| 效率系统 | FlashAttention、混合精度、`all_gather`、自研 CUDA `Dedup Join`、专用 Memory Bank 参数服务器。 | 需要工业级分布式基础设施；论文未给出硬件拓扑。 |
| 验证 | 可先复现 Table 2/3 的离线 QAUC，再消融 SQDC、Memory Bank、采样和 session mask。 | 线上 14 天 A/B、抖音广告和 TikTok Search 结果无法在公开环境直接复现。 |

## 关联

- [[TRM：语义token取代itemID释放扩展潜力]]：同属去 ID 化、增强泛化的方向；TRM 偏向语义 token 化，LEMUR 保留连续多模态表征并直接联合排序训练。
- [[生成式推荐为何开始替代级联管线]]：同属去 ID 化、增强泛化的方向。
