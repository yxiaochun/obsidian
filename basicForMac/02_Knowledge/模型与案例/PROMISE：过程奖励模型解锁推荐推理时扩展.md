---
创建日期: 2026-09-15
更新日期: 2026-09-26
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
  - 语义ID
来源: "[[2026Kuaishou_PROMISE.pdf]]"
状态: 待复核
证据强度: 单篇工业论文证据；包含公开基准、工业离线与在线 A/B 证据
待验证问题:
  - "PRM 对更长层级 SID 和不同数据分布是否保持收益，时延预算如何控制？"
---

# PROMISE：过程奖励模型解锁推荐推理时扩展

## 论文信息

- 作者/机构：Chengcheng Guo、Kuo Cai、Yu Zhou、Qiang Luo、Ruiming Tang、Han Li、Kun Gai、Guorui Zhou；主体来自快手科技。
- 期刊/会议/年份：2026 arXiv 论文（arXiv:2601.04674v1），PDF 首页未给出正式会议名。
- 领域：生成式推荐、过程奖励模型、测试时扩展、工业级检索。
- 与我研究的关联：直接展示如何把 LLM 领域的 PRM 与过程监督迁移到层级 Semantic ID 生成检索，并把推理时扩展从参数扩展中分离出来。

## 模型结构

![[PROMISE：过程奖励模型解锁推荐推理时扩展｜模型结构图.png|900]]

图中左侧是用户历史进入 Encoder、Decoder 生成 SID 的过程；中部把候选路径映射为 path token，经 Path-level PRM Block 和 MLP 输出奖励；右侧展示正样本取真实 SID 前缀、负样本从有效路径集合采样的训练方式。推理时，encoder-decoder 先产生 K+ 条候选，PRM 再筛出 K' 条进入下一层。

## 一句话创新点

用 Path-level PRM 对层级 Semantic ID 的中间生成路径提供稠密奖励，并在推理时扩大候选、筛回固定 beam，从而把测试时扩展引入工业生成式推荐。

## 七问笔记

| 问题 | 回答 |
|---|---|
| 研究背景 | 生成式推荐用层级 Semantic ID 把 next-item 预测拆成多次自回归生成。训练用 teacher forcing，推理却依赖模型自己生成的中间路径，早期高层 token 错误会把候选带入错误语义子空间。 |
| 研究目的 | 显式监督 SID 生成的每个中间步骤，缓解 Semantic Drift，并验证增加推理候选宽度能否比增大模型参数更高效地提升推荐质量。 |
| 创新点 | 把 Path-level PRM 和 PRM-guided Beam Search 引入工业生成式推荐，用中间路径奖励替代只看最终序列概率的传统 beam search，从而开启推荐侧测试时扩展。 |
| 研究方法 | 用 NTP loss 训练 encoder-decoder；为每个 SID 前缀构造正负样本并用逐层 InfoNCE 训练 PRM；推理时先扩到 K+ 条候选，由 PRM 筛回 K' 条，再进入下一层生成。 |
| 实验数据 | 公开侧用 Amazon Beauty 与 Sports and Outdoors；工业侧用快手在线学习日志、3 层×8192 codebook、用户序列长度 256、K=K'=1000。 |
| 结果结论 | 公开数据上超过 ActionPiece，工业离线上超过 GRank 等强基线；线上 A/B 显著提升使用时长和视频观看时长，且只增加 10% 时延。 |
| 总体评价 | 数据链路完整，方法动机和消融较充分；但收益依赖 tokenization、层级深度和延迟预算，论文没有单独 Limitations 章节，也未公开代码。 |

## 批判性分析

- Why 回答：作者把 SID 层级生成类比 CoT。高层 token 相当于粗粒度路由，一旦错误就很难在细粒度 token 阶段修复；PRM 在中间层提供稠密信号，正好补上 NTP 只监督正确路径、不知道错误路径的缺口。
- 换位思考：我会把 Semantic Drift 的定义、HRecall 指标和「只扩大 PRM 候选、不扩大 decoder beam」放在方法后立即展示，这样读者能更快理解它为什么不只是 beam search 调参。若我做扩展实验，还会报告更多 K'（小于 1000）和不同层级深度下的延迟-质量曲线。
- 优点：问题定义清楚，PRM 与生成主干端到端训练，推理结构可并行；公开基准、工业离线、线上 A/B 三层证据互补。HRecall@1/2/3@1000 消融也把收益拆回到各层错误修正上。
- 不足：工业 SID 只有 3 层，尚未证明更长层级或更复杂分布下的稳定性；公开 Beauty 的提升远小于 Sports and Outdoors，提示收益和 item 空间分布有关；A/B 指标集中在使用时长与观看时长，未给出新 item、长尾用户或满意度等子群结果。

## 关键图表解读

- Figure 1：同一名用户和上下文下，传统 beam search 在早期 token 后漂移到无关 SID，PRM-guided search 能把目标 SID 保留下来，直观解释方法的验证对象。
- Figure 2：论文主结构图，对应 NTP 主干、Path-level PRM、InfoNCE 训练和 K+/K' 推理筛选四个模块。
- Figure 3：把传统固定 beam 扩成 K+ 后由 PRM 保留 K'，说明测试时扩展只把额外计算放在轻量 PRM 上，decoder 计算基本保持不变。
- Figure 4：K+ 从 1000 扩到 6000 时，HRecall@3@1000 从 22.96% 提升到 36.37%，HRecall@2@1000 从 37.18% 提升到 49.88%，HRecall@1@1000 从 92.38% 提升到 94.50%；传统暴力扩 beam 的收益更小且显著增加 decoder 计算。
- Figure 5：与参数扩展对比，测试时扩展在相同推理 FLOPs 下取得更高 HRecall@3@1000，或用更少 FLOPs 达到同等效果。
- Figure 6：模型参数量和用户序列长度增大时性能持续提升，说明 PRM 没有阻断主干模型的正常扩展。
- Table 4：第 1、2、3 层分别启用 PRM 都有收益，三层同时启用时 HRecall@1/2/3@1000 达到 0.9431/0.4711/0.3358，证明收益是逐层纠偏叠加的结果。

## 核心问题

层级 Semantic ID 训练采用 teacher forcing，推理却依赖模型自己生成的中间路径。早期高层 token 一旦错到无关语义子空间，后续细化 token 很难纠正；模型面对分布外中间状态时还倾向回退到热门 item 分布，形成 Semantic Drift 和流行度偏置。传统 beam search 只看最终序列概率，没有对中间步骤做稠密验证。

## 方法要点

- Path-level PRM：对每条候选 SID 路径的前缀打分。正样本是真实 item 的各级 SID 前缀；负样本从有效路径集合中均匀采样，避免无意义 token 组合。奖励模型在每个层级用 InfoNCE 把正路径分数拉高、负路径分数压低，并与 NTP loss 端到端联合训练。
- 轻量 PRM 架构：把每条 SID 前缀映射成一个 path token，作为 cross-attention 的 query；复用生成模型 encoder 输出作为 key/value，叠加 1 层 cross-attention、RMSNorm、FFN 和 MLP 输出标量奖励。工业实现中生成模型 encoder/decoder 各 4 层、hidden size 1024，PRM 只有 1 层。
- PRM-guided Beam Search：先把候选 beam 扩到 K+，由 PRM 对中间路径打分后只保留 K'=1000 条进入下一步。工业配置中全局有效 beam K=K'=1000，K+ 取 4000 或 6000；K+ 扩大只增加轻量 PRM 计算，不增加 decoder 的自注意力与交叉注意力负载。
- 工业数据与 tokenization：快手日志覆盖 4 亿+日活、每天约 500 亿次用户互动、每天优化 1 亿+item；tokenizer 深度为 3，每层 codebook 大小 8192，用户序列长度 256。
- 系统部署：线上使用 K+=4000，1 个 PRM block，PRM 注意力头数减少到主生成模块的 1/4，并用 Radix Top-K 加速候选筛选；总参数量增加 15%，推理时延增加 10%。

## 线上效果

- 公开数据集：在 Amazon Sports and Outdoors 上，相较最强基线 ActionPiece，Promise 的 Recall@5/NDCG@5/Recall@10/NDCG@10 分别为 0.0450/0.0296/0.0689/0.0373，对应提升 +42.41%/+44.39%/+37.80%/+42.19%。在 Beauty 上分别为 0.0536/0.0345/0.0821/0.0437，对应提升 +4.90%/+1.47%/+5.60%/+3.07%。条件是 TIGER 切分协议、RQ-VAE 3 层 codebook 256、输入长度 40、hidden size 256，见论文 Table 1。
- 工业离线：K+=4000 时 Recall@100/NDCG@100/Recall@500/NDCG@500/Recall@1000/NDCG@1000 为 0.1494/0.00652/0.2836/0.01231/0.3358/0.01445，相较最强基线 GRank 分别 +47.92%/+38.14%/+18.36%/+13.04%/+5.66%/+1.62%；K+=6000 时相应值为 0.1609/0.00663/0.3017/0.01272/0.3637/0.01504，最高提升达 +59.31% 和 +40.47%。条件是快手在线学习数据、d=3/M=8192、序列长度 256、K=K'=1000，基线为 GRank、MPFormer、MISS、GPRP、Kuaiformer、CRM，见论文 Table 2。
- 在线 A/B：在快手主端和快手 Lite 各分配总用户的 5% 实验组与 5% 对照组，实验持续 7 天；对照为传统 beam search 生成式模型，实验组加入 PRM-guided search。Kuaishou 的总 App 使用时长 +0.121%（CI +0.04%~+0.20%）、人均使用时长 +0.120%、总视频观看时长 +0.431%；Kuaishou Lite 分别为 +0.131%（CI +0.03%~+0.23%）、+0.160%、+0.398%。条件见论文 Table 3。

## 测试时扩展与消融

- 中间层级消融：工业数据上固定 K=1000。不用 PRM 时 HRecall@1/2/3@1000 为 0.9238/0.3718/0.2296；在第 1、2、3 层都用 K_b+=4000 的 PRM 后提升到 0.9431/0.4711/0.3358。这说明早期纠偏和逐层过滤都会降低错误累计，见论文 Table 4。
- 测试时扩展：固定 decoder beam 为 1000，把 K+ 从 1000 扩到 6000，HRecall@3@1000 从 22.96% 提升到 36.37%，HRecall@2@1000 从 37.18% 提升到 49.88%，HRecall@1@1000 从 92.38% 提升到 94.50%。暴力扩大传统 beam 只带来 25.14% 级别的 HRecall@3 或更小改善，并显著增加 decoder 计算。条件见论文 Figure 4。
- 与参数扩展对比：在工业数据上，测试时扩展在相同推理 FLOPs 下取得更高 HRecall@3@1000，或用更少 FLOPs 达到同等效果。Figure 5 中基线为 22.96%，两条扩展路线可达到 26.67% 和 34.37% 附近，但测试时扩展的曲线更高效。条件见论文 4.6.1 节。
- 系统成本：K+=4000 部署版本只增加 1 个 PRM block，PRM 注意力头数为主生成模块的 1/4，并用 Radix Top-K 优化候选筛选。相比无 PRM 版本，总参数量增加 15%，推理时延增加 10%。条件见论文第 5 节。

## 值得追踪的引用

- [ ] Lightman et al., 2023, *Let's Verify Step by Step*：LLM 中过程奖励优于稀疏结果奖励的核心证据，可用来解释 PRM 迁移到推荐的动机。
- [ ] Snell et al., 2024, *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*：本文推理时扩展与参数扩展对比的思想来源。
- [ ] Rajput et al., 2023, *Recommender Systems with Generative Retrieval / TIGER*：定义层级 SID 生成式检索基准和公开数据协议。
- [ ] Hou et al., 2025, *ActionPiece*：最强公开基线，可用于检验 tokenization 变化是否改变 PRM 收益。

## 术语与句式积累

- 术语：Semantic Drift、Exposure Bias、Path-level PRM、PRM-guided Beam Search、Hierarchical Recall（HRecall@b@k）、Test-Time Scaling。
- 可复用句式：「早期高层 SID token 承担语义路由职责，错误会在后续细粒度生成中被放大」「把推理计算从 decoder beam 扩展转移到轻量路径奖励验证上，可以在固定 K' 的情况下提升中间层命中率」。

## 复现清单

- 数据：公开侧按 TIGER 协议处理 Amazon Beauty 与 Sports and Outdoors；工业侧需要在线学习日志、多模态 item embedding 和 Residual K-means 量化管线。
- 代码：论文未提供开源仓库链接，需要按 Eq. 1-13 和 Algorithm 1 自行实现 encoder-decoder、逐层 InfoNCE、K+/K' 搜索与 HRecall 评测。
- 环境：需要支持大 batch cross-attention 打分、Radix Top-K 或等价高效 Top-K 的 GPU serving 栈；公开复现可先从小规模 RQ-VAE 与 Transformer 开始。
- 改良设想：把负样本从均匀采样改成候选 beam 内错误路径、难负路径或对比量化中心，检验是否能更好覆盖真实 Semantic Drift；同时扫不同 K'/K+ 和 PRM 深度，找到显式延迟预算下的收益峰值。

## 关联

- [[GR4AD：广告生成式推荐的架构训练推理联合设计]]：动态 beam serving 配套
- [[测试时扩展与过程奖励进入推荐系统]]
- [[语义ID如何成为生成式推荐的基础设施]]：漂移源于层级 SID 结构

## 局限与开放问题

- 论文没有单独 Limitations 章节。可质疑处包括：Beauty 上的相对提升明显小于 Sports and Outdoors，说明 PRM 收益与数据分布和 item 空间有关；工业 token 空间为 3×8192，层级较浅，PRM 对更长层级 SID 的收益还需验证；推理时扩展带来的 10% 时延增加在小流量或严格尾延迟场景仍需按实际预算评估。
