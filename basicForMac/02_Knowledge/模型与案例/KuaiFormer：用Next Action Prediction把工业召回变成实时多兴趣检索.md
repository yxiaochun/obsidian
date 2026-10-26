---
创建日期: 2026-10-22
更新日期: 2026-10-22
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[KuaiFormer Transformer-Based Retrieval at Kuaishou.pdf]]"
状态: 待复核
证据强度: 单篇工业论文证据
待验证问题:
  - "论文未报告最终线上配置的 query token 数、embedding 维度、层数、检索预算和资源开销，需结合后续工业论文或复现确认。"
  - "三场景线上 A/B 只有一周且未给出置信区间，长期留存、兴趣漂移和新 item 冷启动效果仍需验证。"
  - "Figure 1 示例用 4 个 query token，超参实验指出 6 个 token 是固定候选预算下的离线平衡点；最终部署选择的 token 数需要确认。"
  - "LogQ 校正与 label smoothing 的超参数、负样本实现和失效边界论文未给出，需要独立实验确认。"
---

# KuaiFormer：用Next Action Prediction把工业召回变成实时多兴趣检索

> [!info] 精读结论
> KuaiFormer 的关键不是逐 token 生成语义 ID，而是把召回训练目标从独立估计 CTR 类分数改成 Next Action Prediction：用因果 Transformer 编码实时行为，输出多个兴趣向量，再对海量 item embedding 做 ANN 检索。自适应压缩解决 256 长度序列的算力问题，LogQ 校正和 label smoothing 稳定 in-batch 训练。该方案在快手短视频场景自报离线命中率、三场景一周线上观看时长和互动指标均提升，但对生成式推荐的主要价值是“把 next-token 目标迁移到 ID 级工业召回”，不等于完整的生成式召回管线。

## 筛选与阅读判定

- **阶段 0 筛选**：精读。论文来自快手短视频生产系统，任务与我关心的生成式召回、实时兴趣建模和工业长序列约束直接重合；发现来源是 [[2026Kuaishou_GRank.pdf]]，任务底稿记录历史被引论文数为 6，外部正式引用数未核实。
- **阶段 1 一句话速览**：论文把快手短视频召回改成 Transformer 驱动的 Next Action Prediction，用多兴趣查询 token 和历史压缩支持实时大规模服务。
- **判定**：进入结构通读和逐句精读。全文覆盖方法、部署、离线对比、线上 A/B、超参和消融，不是只有摘要级构想。

## 基本信息

| 项 | 内容 |
| --- | --- |
| 论文标题 | KuaiFormer: Transformer-Based Retrieval at Kuaishou |
| 作者/机构 | Chi Liu、Jiangxia Cao、Rui Huang、Kai Zheng、Qiang Luo、Guorui Zhou；Kuaishou Technology。Kun Gai 标为 Unaffiliated |
| 期刊/会议/年份 | arXiv:2411.10057v1，2024-11-15；论文使用 ACM 模板且页眉仍是占位样式，不能据此认定已正式发表于某会议 |
| 领域 | 大规模短视频推荐、工业召回、序列推荐、Transformer、多兴趣建模 |
| 与我研究的关联 | 提供了生成式推荐的一个重要对照：不改变 ANN 召回服务形态，而是把 next-token prediction 目标迁移到工业 ID 候选池，同时用多兴趣和压缩序列解决实时约束。 |

## 一句话创新点

KuaiFormer 用因果 Transformer 直接做工业短视频召回的 Next Action Prediction，通过多个可学习 query token 自回归生成多兴趣向量、按时间粗细粒度压缩早期历史，并用 LogQ 校正和 label smoothing 稳定 in-batch softmax，从而在实时检索服务中替代传统双塔式 EBR。

## 模型结构

![[KuaiFormer：用Next Action Prediction把工业召回变成实时多兴趣检索｜模型结构图.png]]

Figure 1 以序列长度 256、4 个 query token 为例展示完整链路：item ID 和观看侧信息先经 MLP 变成 token；最早 128 个行为按 64 个一组压缩，随后 80 个行为按 16 个一组压缩，最近 48 个行为保持原粒度，共形成 55 个输入 token。因果 Transformer 与 query token 输出多个兴趣向量；训练时用 in-batch contrastive、LogQ 和 smooth loss，推理时用同一 item embedding 表做 ANN 检索并返回 TopK。

## 方法机制

| 模块 | 论文做法 | 关键作用 |
| --- | --- | --- |
| 问题定义 | 输入用户近期正反馈短视频序列 $(x_i, f_i)$，其中 $x_i$ 是 item ID，$f_i$ 是观看时长、点赞/评论/关注、视频时长、多模态类目等 10+ 侧信息；训练最大化 $P(x_{n+1}\|u_1,\ldots,u_k)$，推理用 $u_1,\ldots,u_k$ 做 ANN。 | 把召回目标显式写成“预测下一个动作”，而不是只训练一个静态匹配分。 |
| 输入表征 | 离散属性用 one-hot lookup；连续属性按业务分布分桶。视频时长在示例中最多 300 秒、均匀切成 1000 桶。侧信息与 item embedding 拼接后经 MLP 融合成 $t_i$。 | 在 ID 之外保留实时观看强度、互动类型和内容属性，让序列 token 承载行为质量。 |
| 主干 | 采用 Llama 风格 Transformer：RMSNorm、多头因果自注意力和 point-wise FFN；输出序列顶部兴趣表示。 | 用纯 Transformer 而非额外 capsule 或双塔用户塔完成兴趣编码。 |
| 自适应压缩 | 将 256 长度分成 early/middle/latest 三段：早期每 64 个行为一组、中间每 16 个一组，各用单层无掩码双向 Transformer 聚合并 mean pooling；最近 48 个行为不压缩。压缩序列长度为 2+5+48=55。 | 论文实验称 64 扩到 256 时直接建模计算资源约 6 倍，压缩后只增加约 10%，同时保留近期行为细粒度。 |
| 多兴趣 | 追加 $k$ 个可学习 query token，与压缩序列一起进入因果 Transformer。后者兴趣 token 可看见前者兴趣 token；预测分数取目标 item 与多个兴趣向量点积的最大值。 | 避免单向量无法表达短视频用户同时存在的多个兴趣，也缓解多语义正例之间的冲突。 |
| 稳定训练 | 使用 in-batch softmax；用 LogQ correction 校正热门 item 被频繁当作负样本的采样偏差；再做 label smoothing，用超参 $\alpha$ 替代严格 0/1 标签。 | 不必对十亿级候选做 full softmax，同时降低热门负采样偏差和短视频弱负样本噪声。 |
| 在线服务 | 模型在线学习、分钟级更新参数；embedding server 存放稀疏 embedding，稠密模型参数周期性同步；在线用 GPU brute-force 检索 TopK，论文也提到 Faiss/ScaNN 可作为 ANN 实现。 | 训练侧追踪实时兴趣，服务侧维持向量检索的低延迟路径。 |

## 实验证据

### 离线命中率

数据来自快手短视频推荐数据流，论文称这是其最大推荐场景，覆盖 4 亿以上用户、每天 500 亿日志。评估方式是回放真实在线请求，让各召回模型分别返回 50/100/500/1000 个候选，计算实际被用户观看 item 的命中率。表中数字为论文 Table 1 的百分数值。

| 方法 | HR@50 | HR@100 | HR@500 | HR@1000 |
| --- | ---: | ---: | ---: | ---: |
| GPRP | 3.57% | 5.21% | 11.31% | 15.70% |
| SASRec | 1.17% | 2.51% | 5.37% | 7.22% |
| DimeRec | 1.54% | 2.49% | 6.99% | 9.17% |
| ComiRec | 0.95% | 1.93% | 7.02% | 10.06% |
| Swing | 0.14% | 0.29% | 1.54% | 2.03% |
| GNN | 3.97% | 4.60% | 8.89% | 10.26% |
| KuaiFormer | 4.21% | 6.58% | 15.61% | 19.88% |

KuaiFormer 相对第二好方法提升 6.05%、26.30%、38.12%、26.62%。论文解释 SASRec 弱说明单兴趣向量覆盖不足；GPRP 利用了从召回到排序的 funnel 数据，但 KuaiFormer 未显式使用推荐漏斗数据仍超过它。

### 线上 A/B

线上 A/B 为一周、10% 用户，覆盖快手主站单页、快手极速版单页和快手双页三个场景。下表为论文 Table 2 的相对增量；Double Page 因业务优先级省略部分指标，不代表负收益。

| 场景 | 视频观看时长 | 总 App 使用时长 | 人均使用时长 | 平均单次视频观看时长 | 视频观看数 | 点赞 | 关注 | 评论 | Novel Surprise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kuaishou Single Page | +0.360% | +0.184% | +0.157% | +0.265% | +0.095% | +0.256% | +0.600% | +0.553% | +0.235% |
| Kuaishou Lite Single Page | +0.126% | +0.127% | +0.107% | +0.060% | +0.158% | +0.286% | +0.222% | +0.062% | +0.181% |
| Kuaishou Double Page | +0.411% | – | – | – | – | +0.399% | +1.254% | +1.463% | – |

三场景视频观看时长分别提升 0.360%、0.126%、0.411%。论文称在 4 亿 DAU 平台上 0.1% 级别已有统计显著性和业务价值，但没有给出置信区间、检验方法或分层结果。

### 超参与压缩消融

超参实验在快手工业数据流上进行，以训练收敛后的 offline accuracy 为指标，不是公开数据集的 AUC/Recall。

| 因素 | 论文报告结果 | 解释 |
| --- | --- | --- |
| 序列长度 | 从 8 到 256，accuracy 从 0.325 提升到 0.441。 | 更长历史持续带来上下文收益，但 64 之后边际收益变小。 |
| Query token 数 | token 1→2 增加 0.006；token 6→7 只增加 0.001。固定候选预算下每个 token 的候选数会更少。 | 更多 token 表达更多兴趣，但会递减并增加推理成本；该实验称 6 个 token 是平衡点。 |
| 层数 | Layer 1→2 从 0.391 到 0.419；Layer 4→5 从 0.433 到 0.435。 | 深度有收益但 4 到 5 层后收益很小，作者认为 4/5 层是效率平衡点。 |
| 压缩 | 64 原始序列 accuracy 0.417；256 原始序列 0.441；256 压缩序列 0.445。压缩窗口从旧到新为 64/64/16/16/16/16/16，最后 48 个 item 不压缩。压缩实验固定 1 个 query token 和 4 层 Transformer。 | 压缩比 64 原始序列更好，也略高于 256 原始序列，作者认为早期粗粒度建模兼具降算力和降噪作用。 |

## 七问笔记

| 问题 | 回答 |
| --- | --- |
| 研究背景 | 大规模短视频召回要从数十亿候选中为每天数亿用户的请求筛出数千个候选。传统 EBR 多用 DNN 双塔；SASRec/BERT4Rec 等序列模型通常压成单一兴趣，MIND/ComiRec 用 capsule 或动态路由抽取多兴趣但不是原生 Transformer，也难以处理长序列与实时性。 |
| 研究目的 | 在 4 亿用户、每天 500 亿日志、超过 5 万 QPS 的快手短视频召回中，用纯 Transformer 实时捕捉变化兴趣、表达多兴趣，并在十亿级 item 和严格延迟预算下稳定训练与推理。 |
| 创新点 | 把工业召回改写为 Next Action Prediction，配合自回归多兴趣 query token、早期粗粒度/近期细粒度的自适应压缩，以及 LogQ+label smoothing 的 in-batch 训练，实现实时多兴趣 Transformer 召回。 |
| 研究方法 | 输入 item ID 与观看侧信息，经 MLP 得到 token；Llama 风格因果 Transformer 做主干；256 长度历史压缩为 55 token；追加 $k$ 个 query token 输出多兴趣；训练用 in-batch softmax、LogQ 校正和 label smoothing；服务时用 ANN/GPU brute-force 检索 TopK。 |
| 实验数据 | 快手短视频生产数据流，4 亿以上用户、每天 500 亿日志；基线包含 Swing、ComiRec、GNN、DimeRec、GPRP 和 SASRec。指标包括回放请求的 HR@50/100/500/1000、训练收敛 accuracy，以及三个场景的一周线上 A/B。 |
| 结果结论 | 离线 HR@1000 从最强基线 GPRP 的 15.70% 提升到 19.88%；三场景视频观看时长分别提升 0.360%、0.126%、0.411%，点赞/关注/评论和 Novel Surprise 多数为正。压缩 256 序列在固定 1 query token、4 层设置下 accuracy 达 0.445，高于 256 原始序列的 0.441 和 64 序列的 0.417。 |
| 总体评价 | 生产规模、离线命中率和三场景线上收益共同支持“纯 Transformer 召回可在快手落地”。但方法收益依赖内部数据流和基础设施；论文缺少完整训练配置、置信区间、资源曲线和长期留存验证，因此应视为单篇工业论文证据。 |

## 批判性分析

### Why 层面

- **为什么要研究这个问题？** 短视频兴趣变化快、单用户每天观看数百条，用户同时有多个语义方向。传统双塔把用户压成静态匹配向量，日更或单向量模型跟不上实时兴趣；capsule 型多兴趣又不是原生 Transformer，长序列和在线服务成本更高。论文的动机在快手业务里是成立的。
- **为什么用 Next Action Prediction 而不是普通 CTR/score estimation？** 下一动作目标直接对齐“召回用户接下来会看什么”，可以从实时行为序列中更新兴趣，不需要显式把召回目标压成某个中间预估分数。它也让 Transformer 的序列建模能力成为核心，而不是把 Transformer 当作特征编码器。
- **为什么用 LogQ correction 和 label smoothing？** in-batch negatives 不均匀，热门 item 会被过度惩罚；短视频里有观看但未点赞的内容也不宜当作严格负样本。LogQ 校正采样偏差、label smoothing 降低标签噪声，这两个选择直接对应大规模召回的两个已知问题。
- **为什么用“早期粗、近期细”的压缩？** 近期动作更能代表当前意图，早期历史更像长期偏好背景。把早期 64 个一组、近期不压缩，能在 256 长度下把主 Transformer 输入降到 55。这个假设符合 recency bias，但也被 Table 3 支持：压缩 256 的 accuracy 0.445 高于原始 256 的 0.441。
- **为什么测 HR 而不是只测 Recall？** 论文的离线指标叫 Hit Rate，本质是在回放请求中检查实际观看 item 是否出现在模型 TopK 候选里，直接对应召回漏损。线上再看观看时长、互动和 Novel Surprise，能补充召回候选进入下游后的业务效果。

### 换位思考

如果重新组织论文，我会把 Table 3 的压缩收益和资源对比提前到方法细节前，因为它最能回答“为什么工业系统敢用 256 长度”。实验上还需要四组更强的对照：固定候选预算下不同 query token 数的完整服务成本；压缩窗口可学习或按内容边界自适应的版本；用排序/漏斗数据增强 KuaiFormer 的版本；以及与 DimeRec、GRank、HSTU 等生成式/序列候选在同一 item 池、同一延迟和同一 embedding 更新频率下的对比。还可以报告分层 HR、长尾 item 覆盖、置信区间和新 item 冷启动曲线。

### 优点

- **工程动机清晰**：问题不是“是否更像大模型”，而是十亿级候选、实时流和长序列约束下能否落地。
- **模块彼此对齐**：多兴趣解决语义冲突，压缩解决算力，LogQ+smoothing 解决大规模负采样和弱标签，三者的目标分别对应召回的痛点。
- **证据链较完整**：有方法、部署架构、回放离线对比、三场景线上 A/B、超参和压缩消融，而不是只有离线公开数据集结果。
- **与生成式推荐衔接自然**：它验证了 next-token prediction 的目标可以迁移到 ID 级推荐，同时仍保持 ANN 服务形态。

### 不足与边界

- **内部数据与自报结果**：数据、候选池、特征和日志管道都来自快手，论文未提供官方代码。离线对比的训练细节和基线容量不完全披露，无法排除工程实现差异。
- **A/B 窗口短**：一周、10% 用户适合验证短期收益，但不足以判断长期留存、兴趣窄化、内容生态反馈和 novelty 指标的稳定性。
- **部署关键信息缺失**：没有给出最终线上 query token 数、embedding 维度、模型层数、batch size、候选池规模、TopK 预算、P99 延迟、GPU/CPU 成本和 embedding 更新频率。
- **多兴趣评估不够细**：论文没有按兴趣 token 分析命中率、新颖性或去重贡献，也没有证明后一个兴趣 token 能看见前一个的因果设计相对独立 token 的净收益。
- **写作瑕疵**：正文把 Figure 3 的三个子图称为 Table 3(a)(b)(c)，但 Table 3 实际是压缩消融；个别指标描述存在笔误，如评估协议中“top 50/100/500/100 results”应结合 Table 1 理解为 50/100/500/1000。

> [!warning] 证据边界
> 本卡数字来自快手作者报告的 arXiv 预印本。离线指标是内部数据流上的自报回放命中率，线上是三场景一周 A/B 的相对增量，论文未给置信区间和显著性检验。不能把 4 亿 DAU 平台的相对增量直接迁移到其他业务，也不能把 Figure 1 的 4 query token 示例当成最终部署配置。

## 关键图表解读

| 图表 | 核心含义 |
| --- | --- |
| Figure 1 | 主结构图。上半部分是 serving 与 training 闭环：推理用 ANN 检索 TopK，训练用 in-batch contrastive、LogQ 与 smoothing；下半部分展示 256→55 的压缩序列和 4 个 query token。 |
| Figure 2 | 部署架构。KuaiFormer 是与 Swing、GNN、ComiRec、DimeRec、GPRP 等并列的一条召回通路；多路候选去重后进入 pre-rank、cascading rank、full rank。日志经离线训练框架更新模型，embedding server 支撑在线推理。 |
| Table 1 | 回放真实请求的离线命中率对比。KuaiFormer 在 50/100/500/1000 四个候选规模下均最高；HR@1000 19.88%，比 GPRP 15.70% 高 26.62%，说明候选覆盖率提升不是只在小 TopK 上。 |
| Table 2 | 三场景一周 A/B。主站单页/极速版单页/双页的视频观看时长分别 +0.360%/+0.126%/+0.411%；互动指标多为正，其中双页 Comments +1.463%、Follows +1.254%。 |
| Figure 3 | 序列长度、query token 数和层数的收益曲线。三者均呈递减收益：256 长度最高但 64 后变平，6 token 后收益趋平，5 层相对 4 层仅 +0.002 accuracy。 |
| Table 3 | 压缩消融。64 原始 0.417、256 原始 0.441、256 压缩 0.445。压缩没有损失精度，说明早期粗粒度不是单纯的信息丢失，而是过滤了一部分历史噪声。 |

## 值得追踪的引用

- [ ] [SASRec: Self-Attentive Sequential Recommendation](https://doi.org/10.1109/ICDM.2018.00081)：单兴趣序列推荐基线，帮助判断多兴趣模块对覆盖率差异的贡献。
- [ ] [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://doi.org/10.1145/3340531.3341898)：双向注意力序列推荐路线，与 KuaiFormer 的因果在线召回目标形成对照。
- [ ] [MIND: Multi-Interest Network with Dynamic Routing](https://doi.org/10.1145/3292500.3330665)：多兴趣召回的经典 capsule 路线，可比较 query token 与动态路由的表达和效率。
- [ ] [ComiRec: Controllable Multi-Interest Framework for Recommendation](https://doi.org/10.1145/3397271.3402744)：Table 1 中的多兴趣基线，也是聚合多样性与精度的代表。
- [ ] [PinnerFormer: Sequence Modeling for User Representation at Pinterest](https://doi.org/10.1145/3534678.3532704)：工业 Transformer 召回对照，论文用其说明日更模型难以适应快手实时兴趣变化。
- [ ] [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://doi.org/10.1145/3298689.3298686)：LogQ correction 的来源，需要理解其采样概率估计和偏置校正实现。
- [ ] [DimeRec: A Unified Framework for Enhanced Sequential Recommendation via Generative Diffusion Models](https://arxiv.org/abs/2408.11449)：扩散式序列召回基线，可对比 KuaiFormer 的 next-action 向量检索与生成式候选建模。

## 术语与句式积累

| 项 | 内容 |
| --- | --- |
| 术语 | Next Action Prediction；in-batch softmax；LogQ correction；label smoothing；multi-interest query tokens；adaptive item compression；bidirectional Transformer compression；ANN retrieval；GPU brute-force retrieval。 |
| 可复用句式 | “把召回从分数估计改为下一动作预测，才能让实时行为直接改写候选分布。”；“近期历史保留细粒度、远期历史压缩成偏好背景，是在序列深度和延迟预算之间的务实折中。”；“多兴趣不是简单多输出几个向量，而是要在固定候选预算下决定每个兴趣方向的检索配额。” |

## 生成式推荐价值判断

KuaiFormer 对我的方向有三层价值。第一，它证明 next-token prediction 目标可以脱离 NLP token 词表，直接服务 ID 候选池，只要用 in-batch softmax、LogQ 和多兴趣查询解决候选规模和语义冲突。第二，它说明“生成式目标”与“ANN 服务”并不冲突：模型可以训练成 next-action 生成器，推理时仍输出兴趣向量并走 ANN，这比逐 token 遍历索引更接近现有召回基础设施。第三，自适应压缩提醒生成式召回不要把所有历史等权 token 化；远期偏好、近期意图和实时反馈应该有不同的压缩粒度。

但它也不是完整生成式推荐：没有语义 ID 的多步生成、没有 cross-attention 重排，也没有显式生成候选 token 序列。它更适合作为“目标生成式、服务表示式”的工业中间形态，与 GRank、HSTU、MERGE 和 OneRec 对照时，关键问题是哪一层承担生成能力：训练目标、候选索引、用户表示还是在线重排。

## 复现清单

| 项 | 内容 |
| --- | --- |
| 数据 | 快手短视频生产数据流不可获取。复现需要实时正反馈观看日志、观看时长、点赞/评论/关注/分享、视频时长、类目和多模态 tag，以及持续新增 item 的候选池。公开序列推荐数据只能验证算法方向，不能复现工业规模结论。 |
| 代码 | 论文未提供官方代码或仓库链接。 |
| 环境 | 需要在线学习训练框架、分钟级参数更新、稀疏 embedding server、稠密模型参数同步，以及 GPU ANN/brute-force TopK 服务。论文提及 Faiss 和 ScaNN，但快手实践采用 GPU brute-force。 |
| 关键超参 | 论文只给出部分消融配置：256 长度压缩为 55 token；压缩窗口 64/64/16/16/16/16/16，最后 48 个 item 不压缩；压缩实验固定 1 query token、4 层 Transformer；Figure 1 示例用 4 query token；固定候选预算实验中 6 query token 是平衡点。缺失 hidden dimension、head 数、$\alpha$、batch size、学习率、TopK 和服务资源。 |
| 缺失信息 | 多 seed/置信区间、基线训练与调参细节、候选池规模、线上最终配置、A/B 随机化和显著性检验、长尾与新 item 分层结果、延迟/内存/GPU 成本曲线。 |
| 改良设想 | 用可学习或内容感知压缩边界替代固定时间位置；给早期压缩 token 加偏好摘要或语义 ID；用排序/漏斗信号做辅助监督；在固定 TopK 与固定延迟下比较 query token 分配策略；加入轻量候选重排或 target-aware 蒸馏，检验多兴趣向量的边际收益。 |

## 关联

- [[GRank：无结构索引的目标感知生成式检索]]：同一快手体系下的后续检索论文，可直接比较无结构索引、target-aware 重排与 KuaiFormer 的 next-action 向量召回。
- [[OneRec：统一召回与排序的端到端生成式推荐]]：完整生成式推荐对照，用于观察表示式召回与端到端生成的边界。
- [[TRM：语义token取代itemID释放扩展潜力]]：语义 ID 路线，可对比 ID 词表规模、扩展性和生成方式。
- [[超长行为序列建模的工程解法]]：KuaiFormer 的 adaptive item compression 是长序列建模的召回侧压缩案例。
