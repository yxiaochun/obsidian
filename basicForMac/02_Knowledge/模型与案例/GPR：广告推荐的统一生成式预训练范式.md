---
创建日期: 2026-09-15
更新日期: 2026-09-26
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
  - 语义ID
来源: "[[2026Tencent_GPR.pdf]]"
状态: 待复核
证据强度: 单篇工业论文证据
待验证问题:
  - "仿真收益能否等价于真实线上收入，过程奖励是否强化历史热门路径？"
---

# GPR：广告推荐的统一生成式预训练范式

## 论文信息

腾讯 + 清华大学，2026 arXiv 论文（arXiv:2511.10138v3）。GPR 已全量部署在腾讯微信视频号广告系统，基线是成熟的多阶段级联系统。

## 核心问题

多阶段广告推荐存在目标错位和误差传播难以全局最优；统一生成式模型又难以满足工业落地要求。

## 一句话创新点

把广告推荐改写为「统一语义表示 + 分层理解与生成 + 业务价值后训练」的单模型端到端生成任务，让同一个模型在解码中同时完成候选生成、定向约束、排序价值估计和收益对齐。

## 模型结构

由于本次任务限制只允许修改这一张卡片，这里不另建独立 PNG 附件；以下直接嵌入原始 PDF 第 2 页的 Figure 2：

![[2026Tencent_GPR.pdf#page=2]]

Figure 2 的输入是横跨微信视频号、朋友圈、公众号等场景的异质用户旅程，经 U/O/E/I token 与共享 Semantic ID 进入模型。HSD 先对长序列做理解并生成 intent embeddings，PTD 按 Thinking-Refining-Generation 生成目标 Semantic ID，HTE 对每层语义码和最终 item 估值，解码时再由 Value-Guided Trie-Based Beam Search 注入用户定向和广告可用性约束。

## 方法要点

- One-model 目标：把“理解、思考、精修、生成、估值”放进一个生成式框架，同时输出推荐 item 与 final_value，替代级联链路。
- Unified Input Schema：用户旅程统一为 User Token（U）、Organic Token（O）、Environment Token（E）和 Item Token（I）。O 和 I 中的短视频、文章、广告等多模态内容都映射到共享的多层 Semantic ID 空间。
- RQ-Kmeans+：先用 RQ-Kmeans 生成高质量 codebook 初始化，再用 RQ-VAE 式可学习更新，encoder 端加残差连接稳定 latent 分布。动机是缓解 codebook collapse、提高利用率，并让剩余碰撞更语义一致。
- HHD（Heterogeneous Hierarchical Decoder）：HSD 使用 Hybrid Attention、Token-Aware Normalization/FFN 和 Mixture-of-Recursions 生成 intent embeddings；PTD 采用 Thinking-Refining-Generation，thinking tokens 过滤意图信息，refining module 借助扩散式去噪精修，再生成目标 Semantic ID；HTE 对每层 code 和最终 item 估值，同时作为 RL critic。
- 检索与生成：Value-Guided Trie-Based Beam Search 在解码早期应用用户定向生成的 Trie 约束，并按 HTE 估值动态调整 beam 宽度；Trie 中的合法路径已经满足年龄、性别等用户属性和广告投放约束。
- 三阶段训练：预训练用 Multi-Token Prediction，默认 4 个并行 head，每头预测一条完整 SID 路径，捕捉并行兴趣；VAFT（Value-Aware Fine-Tuning）按 action 类型和归一化 eCPM 重加权，使 conversion > click > impression；HEPO（Hierarchy Enhanced Policy Optimization）在高保真仿真环境中做分层过程奖励、GAE 和最终候选 z-score 归一化。仿真包含生产索引、特征管道、业务约束，pCTR/pCVR 模型直接从生产复制，每个请求典型生成 K=40 个候选。
- ARR（Anticipatory Request Rehearsal）：按用户活跃度每 2-4 小时或按请求率构造“下一请求”样本，用最近自然内容、已有用户特征和实时环境特征模拟未来状态，缓解历史数据滞后。

## 线上效果

- Tokenizer 消融：在腾讯广告/自然内容语料上，80% 训练、20% 测试，RQ-Kmeans+ 的 Collision/CURL1/PAS 为 20.60%/99.36%/0.992；RQ-VAE 为 23.21%/92.13%/0.985；RQ-Kmeans 为 21.40%/100%/0.986。RQ-Kmeans+ 相对 RQ-VAE 的碰撞率相对下降 11.2%，且剩余碰撞的语义相似度最高。条件见论文 Table 1。
- 行为建模：使用一年腾讯广告匿名交互训练、次日验证，任务是在百万级 catalog 中生成 top 100。HSTU 的 HitR@100 为 18.98%，OneRec 为 19.85%，完整 GPR/HHD 为 27.32%，相对 HSTU +43.9%、相对 OneRec +37.6%。所有模型共享同一 RQ-Kmeans+ tokenizer 和特征管道，见论文 Table 2。
- 结构拆解：Hybrid Attention 使 HitR@100 达 20.56%，Token-Aware FFN 达 21.98%，Token-Aware LayerNorm 达 20.76%，Mixture-of-Recursions 达 20.09%，外部 LLM thought-process token 达 20.13%；PTD 的 Thinking 和 Refining 分别达 21.75% 和 19.61%；HTE 达 19.91%；MTP 达 22.38%。以上均相对 HSTU，条件见论文 Table 2。
- Scaling：比较 0.02B、0.1B、0.2B、0.5B、1B、2B 六档 dense 参数；总参数由约 80B sparse 参数主导。Figure 5 显示更大模型训练损失更低，形成稳定 scaling law；论文未给出各档最终 loss 数值。
- 业务对齐：MTP baseline 的 nDCG/OPR/平均 final_value/最大 final_value 为 0.3868/0.5292/0.2412/0.6201；加 VAFT 后 nDCG 0.3925、OPR 0.5348；加 DPO 后 0.4383/0.5463/0.2442/0.6659；加 HEPO 后 0.4413/0.5509/0.2630/0.7619。条件是一年训练+次日验证，评估在固定日仿真中生成 K=40 候选，final_value 经 min-max 归一化，见论文 Table 3。
- 线上 A/B：v0.1（HSD+NTP+DPO）相对成熟级联带来 GMV +2.11%、GMV-Normal +2.42%、Costs +3.29%；随后 v0.2 加入无 ARR 的 HEPO 增量 GMV +0.70%；v0.3 加入 MTP+Thinking 增量 +0.63%；v0.4 加入 PTD 增量 +0.71%；v0.5 加入带 ARR 的 HEPO 增量 +0.58%。条件是微信视频号广告、五次序列上线，论文未披露每次流量比例和周期。
- 分层分析：v0.1 中低活跃用户 UG1/UG2 的 GMV 分别 +3.56%/+3.84%，高活跃 UG5 的 GMV +3.68%；UG3 的 CTCVR 最高 +4.63%。新广告（≤3 天）GMV +2.97%、CTCVR +4.02%；老广告（>3 天）GMV +1.65%、CTCVR +2.78%。条件见论文 Table 5。

## 七问笔记

| 问题 | 回答 |
|------|------|
| 研究背景 | 广告推荐必须在实时约束下同时服务用户体验、广告主 ROI 和平台收入；传统检索-预排-排序级联存在目标错位、早期漏斗损失和跨阶段工程成本，而 HSTU 类生成式方法尚未解决广告特有的稀疏转化、异质内容、定向约束和价值优化。 |
| 研究目的 | 将广告推荐重构为一个生成式 one-model 任务，用统一表示和统一解码链路替代级联，同时生成可用候选、估计业务价值并对齐收益。 |
| 创新点 | 提出 GPR：共享 Semantic ID 的 U/O/E/I 统一输入、HSD/PTD/HTE 分层解码器，以及 MTP + VAFT + HEPO 的三阶段业务对齐训练。 |
| 研究方法 | 先用 RQ-Kmeans+ 建立共享语义 token；再训练 HSD 做 Hybrid Attention、Token-Aware Normalization/FFN、Mixture-of-Recursions 和外部 LLM thought tokens；PTD 用 thinking/refining 扩散式精修后生成 SID；HTE 输出层级估值并作为 RL critic；解码用 Trie 与动态 beam width；训练走 MTP、VAFT 和 HEPO。 |
| 实验数据 | 腾讯广告与自然内容语料、一年匿名广告/自然交互序列、次日验证集、微信视频号广告线上 A/B。核心指标包括 Collision/CURL1/PAS、HitR@100、nDCG/OPR、normalized final_value、GMV、Costs、CTR/CVR/CTCVR。 |
| 结果结论 | RQ-Kmeans+ 的碰撞率 20.60% 低于 RQ-VAE/RQ-Kmeans；完整 GPR 在百万级目录 top-100 生成中 HitR@100 27.32%，相对 HSTU +43.9%、相对 OneRec +37.6%；HEPO 在仿真中提升 nDCG/final_value；线上 v0.1 GMV +2.11%、Costs +3.29%，后续组件继续带来正增量，新广告和低中活跃用户受益更明显。 |
| 总体评价 | 论文把「生成式推荐能否进入大规模广告系统」的核心难点拆得很完整，且线上证据强；但仿真收益、序列 A/B 归因和热门路径偏差仍需独立复核，工程迁移依赖腾讯的索引、特征和约束栈。 |

## 批判性分析

- Why 回答：作者从三个痛点切入——级联目标不一致、生成式模型难以满足广告约束与业务价值、历史曝光数据无法探索反事实策略。方法设计因此不是单纯的生成准确率竞赛，而是同时处理表示、解码、估值和收益对齐。
- 对比公平性：Table 2 中 HSTU、OneRec 与 GPR 共享 RQ-Kmeans+ tokenizer 和特征管道，这比只比最终模型更公平；但 encoder-decoder 与 decoder-only 的输入访问方式不同，OneRec 对非纯序列字段的利用更强，所以增益既可归因于 HHD，也包含架构对不同输入假设的适配。
- 实验设计：作者同时覆盖 tokenizer、行为建模、业务对齐、仿真和线上 A/B，证据链较完整；不过缺少 HEPO 与其他离线 RL/过程奖励方法的横向消融，也未公开线上各轮流量比例、实验周期和置信区间，序列上线后的累计归因只能依赖论文的增量声明。
- 指标选择：HitR@100 避免了语义码严格相等带来的过严惩罚，nDCG/OPR/final_value 直接对接排序质量与收益，适合广告任务；但 normalized final_value 是仿真中的相对指标，论文自己也说明不能直接映射线上收入。
- 换位思考：我会把论文顺序改为「业务约束与失败案例 → HHD → 训练 → 线上」，让读者更早看到为什么广告比普通推荐难。若继续实验，应加入历史热门/新广告曝光的因果敏感性分析、不同 beam width 与延迟曲线、HEPO 过程奖励的反事实偏差检验，以及 HHD 各模块对冷启动和长尾广告的单独贡献。

## 关键图表解读

- Figure 2：主模型结构图。输入侧统一 U/O/E/I；中间由 HSD 生成 intent embeddings，PTD 做 thinking/refining/generation，HTE 输出层级价值；右侧展示 Trie 约束下的 beam search 和 refining module。
- Figure 3：RQ-Kmeans+ 用 RQ-Kmeans 初始化多个残差 codebook，再用 VAE 式更新；encoder 残差连接缓解早期分布漂移和 dead vectors。
- Figure 4：训练流水线分三段。MTP 用 4 个并行 head 捕捉并行兴趣，VAFT 把 action 和 eCPM 注入损失，HEPO 在生产快照仿真中产生层级奖励、GAE 和最终候选 z-score。
- Figure 5：0.02B 到 2B 六个 dense 参数档的训练损失递减，论文据此主张存在 scaling law；sparse 参数总量约 80B。
- Table 1：RQ-Kmeans+ 20.60% Collision、99.36% CURL1、0.992 PAS，优于 RQ-VAE 和 RQ-Kmeans。
- Table 2：完整 GPR 27.32% HitR@100。拆解中 MTP +17.9%、Token-Aware FFN +15.8%、Thinking +14.6%、Hybrid Attention +8.3%。
- Table 3：MTP 到 VAFT、DPO、HEPO 逐步提升 nDCG、OPR 和 normalized final_value；HEPO 达 0.4413/0.5509/0.2630/0.7619。
- Table 4/5：线上五轮均报告 GMV 或 GMV-Normal 正增量；v0.1 用户分层显示低活跃和新广告更受益，UG3 的 CTCVR 增益最高。

## 值得追踪的引用

- [ ] HSTU（论文 [35]）：GPR 的主 baseline，也是理解 decoder-only 推荐和长序列缩放的基础。
- [ ] OneRec（论文 [40]）： encoder-decoder 生成式召回-排序统一方案，用于判断 GPR 增益来自广告场景还是架构范式。
- [ ] COBRA（论文 [34]）：稀疏语义 ID 与稠密表示结合的另一条路线，可对照共享 SID 空间的信息损失处理。
- [ ] MTGR（论文 [10]）：工业级生成式推荐扩展方案，可比较交叉特征保留与业务目标设计。
- [ ] Mixture-of-Recursions（论文 [3]）：GPR 借用的有效深度扩展机制，需确认其与其他递归/共享参数方案的相对效率。

## 术语与句式积累

- 术语：U/O/E/I Token、RQ-Kmeans+、HHD、HSD、PTD、HTE、Multi-Token Prediction、VAFT、HEPO、Anticipatory Request Rehearsal。
- 可复用句式：The model hierarchically decouples user-intent modeling from ad generation while exposing level-wise value estimates to trie-constrained decoding.

## 复现清单

- 数据：腾讯广告/自然内容语料、一年匿名交互序列和微信视频号线上环境均为内部数据，论文未提供公开下载或采样协议。
- 代码：论文未给出官方仓库、模型权重、codebook 尺寸、HSD/PTD 层数、优化器、学习率、batch size、训练步数或线上实验流量配置。
- 环境：可按论文协议最小化复现 RQ-Kmeans+ 与 MTP/VAFT/HEPO，但 HEPO 的奖励保真度依赖生产 pCTR/pCVR、索引、特征管道和业务约束快照。
- 改良设想：先在公开广告或电商数据上替换生产仿真器，再检验 HEPO 过程奖励是否可被去偏的倾向加权或新广告保护机制替代；同时绘制 beam width 与延迟/收益曲线，验证 HTE 动态裁剪的边际价值。

## 关联

- [[OneRanker：一个模型统一生成与排序]]：腾讯广告另一方案
- [[UniROM：广告排序的端到端统一生成架构]]：美团广告对照
- [[语义ID如何成为生成式推荐的基础设施]]

## 局限与开放问题

- 论文没有单独 Limitations 章节。可质疑处包括：仿真中的 normalized final_value 提升不能直接等于线上收入提升；线上各轮未披露流量比例和周期，序列上线之间的归因依赖团队声称的显著增量；HHD 依赖统一 schema 和广告定向约束，跨平台迁移需要重建 U/O/E/I token 与 Trie 规则；HEPO 的过程奖励来自历史成功交互的 token popularity，可能强化历史热门路径。
