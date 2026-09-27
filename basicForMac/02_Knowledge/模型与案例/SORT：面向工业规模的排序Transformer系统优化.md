---
创建日期: 2026-09-15
更新日期: 2026-09-26
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2026Alibaba_SORT.pdf]]"
状态: 待复核
证据强度: 单篇工业论文证据
待验证问题:
  - "扩展优先级在稀疏标签和其他平台数据上是否可复现？"
  - "缺少公开数据集时，request-centric 样本组织的收益能否在小规模日志上重建？"
  - "线上 A/B 未报告置信区间时，业务收益能否用更长周期或更多场景验证？"
---

# SORT：面向工业规模的排序Transformer系统优化

> [!warning] 证据边界
> 本文的主结果来自阿里国际私有日志和 AliExpress 线上 A/B，没有公开数据集复现。离线 AUC 只能用于理解同一论文内的相对收益，不应直接跨论文比较。

## 论文信息

| 项目 | 内容 |
| --- | --- |
| 标题 | SORT: A Systematically Optimized Ranking Transformer for Industrial-scale Recommenders |
| 作者/机构 | Chunqi Wang 等，Alibaba International Digital Commercial Group |
| 会议/年份 | CIKM 2026 |
| DOI | [10.1145/3799682.3840109](https://doi.org/10.1145/3799682.3840109) |
| 与研究方向的关系 | 这是传统工业排序向 Transformer/scaling 路线迁移的代表性案例。它没有把排序改成端到端生成召回，而是在保留判别排序 head 的前提下，系统性改造样本组织、注意力、容量和训练系统。 |

## 一句话创新点

SORT 把一次请求的候选、用户历史和画像组织成 `[H, U, C]` 序列，再用局部注意力、query pruning、生成式预训练、DeepSeek-style MoE 和系统栈优化，把工业排序 Transformer 同时做到精度提升、低 FLOPs 和 45% 训练 MFU。

## 七问笔记

| 问题 | 回答 |
| --- | --- |
| 研究背景 | LLM 中的 Transformer 有成熟扩展性，但工业排序长期由 DIN、DeepFM、DCN 等专用交互结构主导。排序任务的 item 词表达十亿级、历史序列缺少直接监督、label 又高度二值稀疏，这使“高特征稀疏 + 低标签密度”成为直接套用 Transformer 的主要障碍。 |
| 研究目的 | 在不放弃工业排序的多目标判别任务和线上延迟约束的前提下，构建一个可扩展、可上线、硬件效率高的 ranking Transformer，并回答高稀疏特征和低标签密度下的扩展问题。 |
| 创新点 | 用请求级样本组织共享用户历史与画像，用局部注意力 + query pruning 降低候选交互成本，用 GPSD 式生成式预训练提高标签密度，再用 BOS/SEP、QKNorm、attention gate、DeepSeek-style MoE 和 RecIS/Megablocks 系统栈把精度与效率一起拉齐。 |
| 研究方法 | 在阿里工业日志上训练 0.6B requests / 50M users / 9B impressions，次日日志作测试集。对比 Transformer、HSTU、OneTrans；消融特殊 token、局部注意力、query pruning、attention gate、QKNorm、MoE；另做生成式预训练、窗口尺寸、MoE 稀疏率、数据/模型/序列长度扩展和一个月线上 A/B。 |
| 实验数据 | 离线训练集 0.6B requests、50M users、9B impressions；测试集 20M requests、4M users、0.3B impressions。任务为 CTR、CVR、AddCart 的二分类 AUC。线上覆盖 AliExpress HomePage、DetailPage、ShoppingCart、AfterPayment 四个场景。 |
| 结果结论 | Base 规模下 SORT 相对 Transformer 的 CTR/CVR/AddCart AUC 分别 +0.41pt/+0.25pt/+0.31pt；Large 规模下分别 +0.51pt/+0.34pt/+0.41pt，同时 FLOPs 从 322G 降到 188G。线上一个月均值订单 +7.47%、买家 +6.67%、GMV +8.65%，时延 -62%，吞吐 +589%。 |
| 总体评价 | 论文对精度、FLOPs、MFU、推理效率和线上业务指标都给了较强证据，方法链条清楚。但它是高度系统化的工业集成：局部注意力、query pruning、MoE、预训练和训练系统互相绑定，公开场景下的单项贡献与扩展规律仍需要独立复验。 |

## 方法机制

### 样本与 tokenization

- 传统 impression-centric 组织会把同一请求中的共享用户历史和画像重复处理 N 次；SORT 改成 request-centric sample：$S=\langle H,U,C\rangle$，其中 $H$ 是用户历史，$U$ 是用户画像，$C$ 是同请求候选集合。
- item 历史和候选 item 都映射为单 token；user profile 映射为多个 token。特征先做 embedding lookup，再 concat，最后线性投影和归一化到模型维度。
- 输入序列为 `[BOS; Tokenize(H); SEP; Tokenize(U); SEP; Tokenize(C)]`。BOS/SEP 不只是分隔符，还会作为 attention sink 帮助模型识别边界、吸收注意力权重。

### 注意力与候选隔离

- 使用 pre-normalization + RMSNorm、RoPE 和 SwishGLU。
- 多个候选采用 diagonal mask，并赋予相同 position ID；每个候选只看自己以及前置历史/画像，避免候选间互相看见。
- 用户历史大部分 token 使用局部注意力，候选和候选附近少量 token 保留因果注意力。复杂度从 $O(L^2)$ 降到 $O(WL)$；超参探索中 64/128/256 三档里 256 最好。
- query pruning 在每层剪掉远离候选的 query，最终层最多保留 128 个非候选 token。这个设计不仅降低计算量，在消融中还带来正向收益，作者解释为与推荐中的时间衰减先验一致。

### 容量与预训练

- FFN 使用 SwishGLU；容量扩展使用 DeepSeek-style MoE。相比 Switch-style MoE，它免去调 auxiliary load balancing loss，并且在稀疏率 1/8 下取得最佳结果。
- item embedding 采用 GPSD 的生成式预训练思路。Table 3 中，从零训练 Transformer 的 CTR/CVR/AddCart AUC 为 0.7175/0.8959/0.8455；只做 sparse transfer 降到 0.7162/0.8943/0.8449；sparse transfer + sparse freeze 提升到 0.7456/0.9093/0.8663。
- sparse freeze 的价值不只是首 epoch 提升，还让多 epoch 训练时验证 AUC 继续平滑增长，从而缓解 one-epoch overfitting。

### 训练与推理系统

- 训练系统基于 RecIS，MoE 用 Megablocks。稀疏模块用 operator fusion、合并内存访问、dynamic embedding 和多 process group 通信隐藏通信；稠密模块为复杂 sparse mask 自研 block-wise attention kernel，配合混合精度和梯度累积。
- Base 规模训练 MFU 从 13% 提高到 22%，Large 规模达到 45%。
- 推理通过 `torch.export` 和 AOTInductor 转静态图，并使用定制 sparse attention kernels、算子融合、半精度、KV cache、multi-context/multi-stream。相对未优化基线，吞吐 +29.4%，时延 -29.3%。

## 关键结果

### 模型对比（Table 4 摘要）

| 规模 | 模型 | CTR AUC | CVR AUC | AddCart AUC | 参数/FLOPs |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | Transformer | 0.7456 | 0.9093 | 0.8663 | 18M / 43G |
| Base | HSTU | 0.7438 | 0.9070 | 0.8644 | 18M / 43G |
| Base | OneTrans | 0.7476 | 0.9103 | 0.8677 | 54M / 24G |
| Base | SORT | 0.7497 | 0.9118 | 0.8694 | 85M / 24G |
| Large | Transformer | 0.7485 | 0.9122 | 0.8695 | 144M / 322G |
| Large | SORT | 0.7536 | 0.9156 | 0.8736 | 685M / 188G |

参数量不含约 20B embedding。组件消融中，特殊 token 和 query pruning 的单项 CTR AUC 增益最大，分别 +0.33pt 和 +0.26pt；QKNorm 的直接 AUC 增益较小，但用于降低 attention 波动和训练不稳定性。

### 扩展优先级

固定计算预算下，Figure 6 的实验覆盖：

1. 数据扩展：multi-epoch（1/2/3 epochs）和 multi-scenario。
2. 模型扩展：small/base/large。
3. 序列长度扩展：256、512、1K、2K、4K。

三者都带来收益，但数据扩展曲线最陡，因此作者建议优先做数据扩展，其次是序列长度，再是模型规模。multi-scenario 通过场景 ID 让模型区分不同来源，multi-epoch 依赖 sparse freeze 控制过拟合。

### 线上 A/B（Table 5）

| 场景 | Orders | Buyers | GMV | Latency | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| HomePage | +2.37% | +2.58% | +10.01% | -62% | +325% |
| DetailPage | +8.99% | +7.46% | +11.49% | -51% | +405% |
| ShoppingCart | +7.10% | +6.74% | +6.98% | -64% | +448% |
| AfterPayment | +11.43% | +9.91% | +6.12% | -72% | +1180% |
| Macro Average | +7.47% | +6.67% | +8.65% | -62% | +589% |

生产对照是增量训练一年以上数据的 DLRM；SORT 只使用最近三个月数据，并以 base 规模配置上线。

## 批判性分析

**为什么有效。** 论文没有把“Transformer 能 scaling”当作默认答案，而是针对排序任务的两个结构性弱点做补偿：request-centric 组织减少重复计算，让更多计算留给候选；生成式预训练用 next-item 监督增加 label density；sparse freeze 又把预训练 embedding 固定住，避免稀疏 ID embedding 在判别微调中过拟合。

**方法对比是否公平。** Base/Large 都按相同 depth 和 width 设置 Transformer、OneTrans、SORT，HSTU 为匹配参数与 FLOPs 而加宽；这说明作者意识到算力公平性。但 HSTU 没有 FFN，OneTrans 用 rule-based routing MoE，SORT 用 data-driven routing MoE，二者不只是结构差异，还包含容量分配差异。表 4 只报告单点结果，没有置信区间，严格对比仍需复跑。

**消融是否充分。** 组件消融覆盖较全，尤其是特殊 token、局部注意力、query pruning、attention gate、QKNorm 和 MoE。但许多关键收益来自系统级组合，论文没有把“纯样本组织 + 同等 FLOPs 的传统模型”作为最终归因基线，也没有给出 query pruning 在不同候选数、序列长度分布上的敏感性。

**扩展结论的边界。** 数据 > 序列 > 模型的优先级很有价值，但数据扩展只有 1/2/3 epochs 和 multi-scenario 两种形式，序列长度到 4K，模型规模到 large。它证明的是该工业域内的相对排序，不是普适 scaling law。

**优点。** 论文同时报告精度、FLOPs、MFU、推理吞吐、时延和一个月线上业务指标；生成式预训练消融清楚地表明“只迁移 embedding 不行，迁移 + 冻结才行”；扩展实验直接面向生产资源分配。

**不足。** 私有数据和约 20B embedding 使公开复现很难；没有公开数据集和开源模型；测试集只有一天日志，长期分布漂移和场景迁移未单独讨论；线上 A/B 没有报告置信区间、显著性检验或流量占比；预训练语料、GPSD 训练细节和多目标 loss 权重也没有完整披露。

## 关键图表解读

- **Figure 1**：整体架构图。用户历史、画像和候选组成一个序列，Transformer block 内使用 QKNorm、gated attention、sparse mask 和 sparse MoE；query pruning 逐层缩短序列，ranking head 只在候选 token 上输出多目标预测。
- **Figure 2**：multi-epoch 下的训练/验证 AUC。sparse freeze 让验证 AUC 继续增长，这是数据扩展实验成立的重要前提。
- **Figure 3**：局部注意力窗口和 MoE 稀疏率探索。窗口 256 最优；DeepSeek-style MoE 优于 Switch-style MoE，稀疏率 1/8 最优。
- **Figure 4**：加入特殊 token 后，BOS 吸收大量注意力权重，对应 attention sink 现象，说明边界 token 不只是形式上的分隔。
- **Figure 5**：small/base/large 下 SORT 与 Transformer 的训练 AUC 曲线，SORT 持续更高，说明优势不是单点 tune 出来的。
- **Figure 6**：数据、模型、序列长度三条扩展曲线。核心读法不是“都能涨”，而是固定预算下数据扩展的边际收益最高。
- **Table 3**：生成式预训练的关键消融。单独 sparse transfer 变差，加 sparse freeze 才大幅变好，说明预训练 embedding 的稳定性比“是否预训练”更关键。
- **Table 4**：主模型对比。SORT 在 base 和 large 规模都超过 Transformer、HSTU、OneTrans，并且 large 规模的 FLOPs 反而低于 Transformer。
- **Table 5**：四个场景一个月 A/B。GMV、订单、买家全正，但幅度差异大；AfterPayment 订单最高，HomePage 订单最低，说明结果不能只看均值。

## 值得追踪的引用

- GPSD: Scaling Transformers for Discriminative Recommendation via Generative Pretraining：SORT 生成式预训练和 sparse freeze 的直接来源。
- [HSTU: Actions Speak Louder than Words](https://arxiv.org/abs/2402.17152)：把排序重新视为序列转换任务的前置工作，SORT 与其共享 request-centric 思路但保留判别排序 head。
- [OneTrans](https://arxiv.org/abs/2510.26104)：与 SORT 的候选隔离、query pruning 和异质特征建模高度相关。
- MTGR：另一个工业级生成式推荐扩展案例，强调交叉特征不能被 scaling 简单替代。
- RankMixer：面向工业排序 MFU 的另一条 token 混合路线。

## 术语与句式积累

- **request-centric sample organization**：一次请求多个候选共享同一份用户历史和画像，样本从 `<H,U,c_j>` 变成 `<H,U,C>`。
- **local attention with query pruning**：历史 token 用局部窗口，query 逐层靠近候选剪枝；二者共同把排序计算集中到标签相关位置。
- **sparse transfer + sparse freeze**：生成式预训练得到的稀疏 embedding 先初始化排序模型，再在判别训练中冻结，以稳定表示并抑制 one-epoch overfitting。
- **data scaling > sequence scaling > model scaling**：工业排序中“更多数据”通常比“更深/更宽”更值得优先投入，前提是能用预训练和多 epoch 控制过拟合。

## 复现清单

| 项目 | 内容 |
| --- | --- |
| 数据 | 0.6B requests、50M users、9B impressions；测试集 20M requests、4M users、0.3B impressions。私有日志，缺少公开替代集。 |
| 模型 | Base：depth 6、width 512、expert intermediate 1280；Large：depth 12、width 1024、expert intermediate 2560。参数量不含约 20B embedding。 |
| 训练 | AdamW，`beta=(0.9,0.99)`，weight decay 0.01；small/base/large 学习率分别为 5e-4、2e-4、1e-4；batch size 12K；默认 1 epoch、序列长度 1K；BF16 自动混合精度，ranking head 和 MoE router 保留 FP32。 |
| 系统 | RecIS + Megablocks；sparse embedding、multi-process-group communication、block-wise sparse attention、AOTInductor、定制 sparse attention kernels、算子融合、KV cache 和 multi-stream 推理。 |
| 改良设想 | 用公开电商或内容推荐日志构建 request-centric 复现协议；固定 FLOPs 对比样本组织、query pruning 和传统 target attention；拆开 GPSD 预训练、sparse freeze、multi-epoch 与 multi-scenario 的收益；在小候选数/低稀疏平台上重测扩展优先级。 |

## 关联

- [[生成式推荐：用万亿参数序列转换器 HSTU 替代 DLRM，验证推荐领域的缩放定律]]：提供排序 Transformer 化和生成式序列建模的早期框架，SORT 在此基础上更强调工业系统效率。
- [[OneTrans：一个Transformer统一特征交互与序列建模]]：同样使用统一 Transformer 建模异质特征，但路由方式与 SORT 不同。
- [[RankMixer：token混合让推荐模型MFU提升十倍]]：同为排序 Transformer 的 MFU/规模化路线。
- [[MTGR：保留交叉特征的工业级生成式推荐扩展]]：提醒 request-centric 和 scaling 不能自动补偿交叉特征损失。
- [[GenRank：大规模生成式排序的工业验证]]：从工业验证角度补充生成式排序的训练收益与成本权衡。
