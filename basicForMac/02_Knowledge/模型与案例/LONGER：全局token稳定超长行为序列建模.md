---
创建日期: 2026-09-15
更新日期: 2026-09-26
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
  - 行为序列
来源:
  - "[[2025ByteDance_LONGER.pdf]]"
  - "[[Clippings/抖音广告&电商LONGER无GSU的End2End长序列建模.md]]"
状态: 待复核
证据强度: 单篇工业论文证据；含工业离线、双域线上 A/B 与系统消融，但数据和代码未公开
待验证问题:
  - 10,000 长度下的边际收益、显存、吞吐与线上 p99 延迟拐点在哪里？
  - Global Tokens 的数量、来源和高阶交互特征在不同业务域中是否同样稳定？
  - Recent 100 查询在冷启动、长尾用户和跨域行为中是否仍优于学习式查询？
  - 未公开流量、周期和显著性检验的线上 A/B 结果能否在其他推荐平台复现？
---

# LONGER：全局token稳定超长行为序列建模

## 筛选与速览

- **精读分级建议**：S。理由是论文同时给出工业离线、双域线上 A/B、效率消融、scaling 分析和系统部署证据，长序列与推荐 scaling 主题和研究方向高度相关；但数据、代码和若干系统细节未公开，结论仍应作为待复核的工业证据。
- **一句话速览**：LONGER 用 Global Tokens 稳定长上下文注意力，用 Token Merge 与 Inner Trans 压缩序列，再用混合因果注意力、KV Cache 和 GPU 同步训练把超长行为序列建模放进工业 CVR 系统。
- **进入第二遍的理由**：论文不是单纯堆层数，而是同时处理长序列中的注意力稳定性、局部信息压缩、候选打分效率和训练系统瓶颈。

## 基本信息

- 作者/机构：Zheng Chai、Qin Ren、Xijun Xiao、Huizhi Yang、Bo Han、Sijun Zhang、Di Chen、Hui Lu、Wenlin Zhao、Lele Yu、Xionghang Xie、Shiru Ren、Xiang Sun、Yaocheng Tan、Peng Xu、Yuchao Zheng、Di Wu，均来自 ByteDance。
- 期刊/会议/年份：RecSys 2025，Prague，2025-09-22 至 2025-09-26，10 页；DOI 为 [10.1145/3705328.3748065](https://doi.org/10.1145/3705328.3748065)。
- 领域：工业推荐、超长用户行为序列、CVR 预测、推荐系统 scaling law、GPU 系统优化。
- 与我研究的关联：这不是端到端生成式推荐，但为生成式推荐中的长行为上下文、全局条件 token、候选感知注意力和工业训练效率提供了可迁移组件。

## 一句话创新点

LONGER 把目标 item、可学习 CLS、UID embedding 和高阶用户-item 交互特征组织成具备全注意力感受野的 Global Tokens，再与 Token Merge、Inner Trans 和首层 cross causal attention 结合，使长序列模型在不直接注意全部原始 token 的情况下保留候选感知、近期行为和全局锚点。

## 七问笔记

| 问题 | 回答 |
|------|------|
| 研究背景 | 工业推荐常用两阶段检索、预训练用户 embedding 或记忆增强模型处理长行为序列；这些方法降低计算量，但会造成上下游目标不一致或只能间接感知原始序列。Transformer 的二次注意力又限制端到端长上下文训练。 |
| 研究目的 | 在工业 CVR 预测中端到端建模超长用户行为，同时稳定长上下文注意力、压缩计算预算，并让训练和服务能在 GPU 集群上部署。 |
| 创新点 | 把 Global Tokens 作为信息锚点，结合相邻 token 合并、局部 Inner Transformer、首层 cross causal attention、后续 self causal attention 和 KV Cache，形成算法-系统一体的长序列推荐架构。 |
| 研究方法 | 1）将目标 item、CLS、UID 和高阶压缩交互特征作为 Global Tokens；2）相邻 token 分组为粗粒度表示，组内用 Inner Trans 保留局部交互；3）输入加入绝对时间差与可学习位置编码；4）首层用 Global Tokens 和 Recent sampled sequence tokens 作 query，对完整序列做 causal cross attention，后续层再做 self causal attention；5）训练端用 GPU 同步稠密/稀疏参数更新、BF16/FP16 混合精度和 activation recompute，服务端用 KV Cache。 |
| 实验数据 | Douyin Ads CVR 数据：2024-10-16 至 2025-02-23，130 天，5.2B 样本；前 123 天训练、后 7 天评估。序列包含 page view、click、conversion，item 侧包含广告内容、展示上下文和元数据。线上 A/B 覆盖 Douyin Ads 与 Douyin E-Commerce。 |
| 结果结论 | 工业离线 LONGER AUC=0.85290、LogLoss=0.47103，相对 base AUC 提升 1.57%，相对最强 Transformer AUC 再提升 0.21%。100 个 Recent queries 用约 54% FLOPs 达到 250 个 queries 的接近效果；Token Merge 与 Inner Trans 可减少计算并提升 AUC。Douyin Ads 三种广告形态 ADSS 提升 1.063%-2.097%，ADVV 提升 1.168%-2.151%；Douyin E-Commerce 两种形态 Order/U 提升 4.6125%-7.9222%，GMV/U 提升 5.2771%-6.5404%。 |
| 总体评价 | 论文较好地把“长序列为什么有效”拆成序列长度、参数、FLOPs 和查询策略，并用工业离线与线上结果支撑可行性。核心限制是数据与代码不可复现、缺少 MIMN/LMN/MARM 等记忆增强路线的直接对照、线上 A/B 未披露流量周期和显著性检验，scaling 曲线也仍受当前参数和序列范围限制。 |

## 结构化摘要

### 背景

作者把现有长序列路线分为三类：two-stage retrieval 先从超长序列选出候选相关短序列再端到端建模；pre-trained user embedding 先在源模型中压缩用户表示再迁移给下游；memory-augmented models 用记忆槽或缓存中间结果换取计算效率。作者认为这些方案虽然有效，但本质上是端到端长序列建模的中间形态，会丢失原始全序列信息或引入上下游不一致。

### 无 GSU 的端到端路线

传统两阶段长序列方案通常先用 **GSU**（General Search Unit）从超长行为序列中检索 top-k 相关行为，再用 **ESU**（Exact Search Unit）对短序列做精细建模；SIM、ETA、SDIM、TWIN 和 TWIN-V2 都属于这类思路。LONGER 的差异在于不设独立 GSU：完整或 Token Merge 后的行为序列仍作为 key/value 保留，候选感知与全局信息通过 Global Tokens 和首层 cross attention 直接查询长序列。

这带来两个含义：第一，检索和精排不在两个目标割裂的模块里发生，Token Merge 是按时间的通用压缩，而不是候选相关的硬检索；第二，模型没有显式的“先选出 top-k 行为”步骤，如果注意力没有找到稀疏但关键的早期行为，就不能像 GSU 那样依靠显式检索兜底。因此 LONGER 更接近端到端长序列建模，但也更依赖注意力、Global Tokens 和 Recent queries 的检索能力。

### 方法

**Global Tokens** 是全模型的锚点，放在输入序列开头。它们包含 target item representation、learnable CLS token、UID/user profile embedding 和高阶压缩的用户-item 交互特征，具备全注意力感受野，可以聚合历史、上下文与候选信息，也帮助缓解深层注意力过度集中在早期 token 的 attention sink 现象。知乎拆解进一步指出，框架图中 candidate item 单独成 token，user profile 与 context/cross features 合并成 token；全局 token 的开头位置对稳定注意力很关键。

行为 token 先加入两类时间位置信息：历史行为与候选请求之间的绝对时间差拼接进 item embedding，行为在序列中的绝对位置编码加到 item embedding。随后 Global Tokens 和序列 Tokens 分别经 MLP 映射降维到同一表征空间。

**Token Merge** 将相邻 token 分组压缩，降低 attention 的序列长度；直接 concatenation 效率高，但组内交互不足。因此 LONGER 在每个组内加入轻量 **Inner Trans**，先做局部 Transformer 编码再输出合并后的组表示。论文称典型 `n=2048, d=32` 时，`k=4` 的合并能把注意力相关 FLOPs 从 587M 降到 336M，减少 42.8%。

**Hybrid attention** 分两段：首层 cross causal attention 用 `[Global Tokens; Recent sampled sequence tokens]` 作为 query，对完整序列的 key/value 计算注意力；后续层在被采样的压缩序列上堆叠 self causal attention。可以把完整长序列理解为记忆库，把 Recent tokens 理解为检索入口：Recent token 不替代长历史，而是携带用户当前意图，决定模型应该回到长历史里查找什么。论文还给出一个边际效应证据：只采样约 40% 的完整序列 token 就能保留超过 95% 的性能改进，同时减少约 50% FLOPs。

论文发现 Recent 采样比 learnable query 和 uniform sampling 更有效。原因是 Recent token 本身就是最近浏览、点击或转化的行为表示，已经包含类目、内容形态、时间上下文和短期意图，相当于给注意力提供了一个内容感知的检索起点。例如用户最近浏览跑鞋和运动袜，这些 Recent query 更容易召回历史中的运动装备和相关兴趣，而 uniform query 可能均匀落在美妆、母婴、旅行等无关行为上，learnable query 则只是一组通用可学习向量，不天然绑定当前用户和候选上下文。这里的“query 初始化”不是参数初始化，而是指 Recent 行为向量已经为当前样本的注意力检索提供了更好的语义起点。

**系统层**采用 GPU 同步训练与存储，稠密和稀疏参数均在 GPU 侧更新，不再依赖外部 Parameter Server；稀疏 embedding 使用 HBM、CPU memory 和 SSD 的分层存储。BF16/FP16 混合精度加 activation recompute 平均带来 +18% 吞吐、-16% 训练时间和 -18% 显存，稠密层显存最高减少 28%。服务时，用户序列的 key/value 预计算后可跨候选复用，候选侧只计算其 Global Token 与缓存序列的注意力；论文报告线上吞吐退化从最高 -40% 降到 -6.8%。

### 实验

工业离线在同一预处理、调参流程和 A100 GPU 集群上比较 Base、SumPooling、TWIN、DIN(Recent50)、DIN、HSTU、Transformer 和 LONGER：

| 方法 | AUC ↑ | LogLoss ↓ | AUC 相对提升 | LogLoss 相对变化 |
|---|---:|---:|---:|---:|
| Base | 0.83968 | 0.48758 | - | - |
| SumPooling | 0.84201 | 0.48538 | +0.28% | -0.45% |
| TWIN | 0.84472 | 0.48168 | +0.60% | -1.21% |
| DIN(Recent50) | 0.84698 | 0.47830 | +0.87% | -1.90% |
| DIN | 0.84982 | 0.47452 | +1.21% | -2.68% |
| HSTU | 0.84994 | 0.47490 | +1.22% | -2.60% |
| Transformer | 0.85111 | 0.47293 | +1.36% | -3.00% |
| LONGER | 0.85290 | 0.47103 | +1.57% | -3.39% |

组件消融显示，`LONGER(w/o Merge, 2000)` 为 3.73×10^9 FLOPs、AUC 0.85111；`TokenMerge4(Concat, 500)` 降到 2.13×10^9 FLOPs、AUC 0.85232；`TokenMerge8(Concat, 250)` 为 3.03×10^9 FLOPs、AUC 0.85291；加入 Inner Trans 后达到 AUC 0.85332、LogLoss 0.47052。查询数量实验中，Recent 100 用 1.91×10^9 FLOPs 达到 AUC 0.85290，接近 Recent 250 的 AUC 0.85332，但只消耗约 54% FLOPs。

线上 A/B 分为两个域。Douyin Ads 中，Live Streaming 的 ADSS/ADVV 分别 +1.063%/+1.168%，Short Video 分别 +2.097%/+2.151%，Mall 分别 +1.816%/+1.407%。Douyin E-Commerce 中，Live Streaming 的 Order/U/GMV/U 分别 +7.9222%/+6.5404%，Short Video 分别 +4.6125%/+5.2771%。

### 结论

作者的结论是：Global Tokens、Token Merge、Inner Trans、混合因果注意力与系统级优化共同让 LONGER 在工业约束内完成端到端超长序列建模，并在广告和电商场景取得一致收益。未来工作集中在更高效的序列建模与工业跨域行为建模。

## 批判性分析

### Why 回答

- **为什么研究这个问题？** 长行为序列同时包含长期兴趣和短期意图；如果先检索或先压缩，再交给下游模型，上游目标与下游排序/转化目标容易错位。作者的动机是把长上下文信息重新放回端到端优化目标中，这个动机在 CVR 预测中成立。
- **为什么不用已有方法？** Two-stage retrieval 和预训练 embedding 隐含了“先选/先压缩”的信息瓶颈；memory-augmented models 又依赖记忆槽命中率。LONGER 的替代方案是保留原始序列作为 key/value，但把 query 压缩到 Global Tokens 和 Recent tokens，从而避开全序列二次 query 计算并让候选条件直接参与注意力。
- **为什么这样设计实验？** 离线 CVR 数据检验模型质量，Token Merge/查询数消融检验效率边界，参数和 FLOPs scaling 检验容量扩展，双域线上 A/B 检验工程可行性和业务收益。这个设计覆盖面较完整，但缺公开数据集和更多外部基线，使外部可迁移性只能间接推断。
- **为什么测这些指标？** AUC 和 LogLoss 匹配二分类 CVR 任务；FLOPs、显存、吞吐和 KV Cache 退化匹配工业部署；ADSS/ADVV 对广告主价值更敏感，Order/U 和 GMV/U 对电商转化更敏感。指标选择合理，但缺少多样性、覆盖度、冷启动和新广告/新用户的分层指标。

### 换位思考

如果重新组织论文，我会先把 Memory-augmented 路线放进离线对照，而不是只在 related work 讨论；再在同等参数量、FLOPs 和训练预算下比较 HSTU、Transformer、LONGER 和记忆模型。对于 Global Tokens，还需要单独消融 target item、CLS、UID、高阶交互特征四类 token 的贡献，而不是只报告整体机制。最后应报告不同序列长度下的 p99 latency、KV Cache 容量、命中率与更新延迟，让“10,000 长度可用”从架构主张变成完整系统证据。

如果重做实验，我会按新用户、低活用户、高频用户、新广告、成熟广告和跨域行为比例分层评估；同时对 Recent 100、learnable query、uniform query、混合 query 做同种子多轨迹实验，检验 query 选择策略的方差。这样能判断收益主要来自长序列、近期行为，还是工程协同优化。

### 优点

- 方法链条完整：全局锚点、局部合并、候选感知注意力、缓存推理和同步训练互相配合，而不是孤立堆一个长序列模块。
- 离线表不仅比较常见序列模型，也给出 Base、SumPooling、TWIN、DIN、HSTU 和 Transformer 的同表对照。
- 消融直接回答了工业上最重要的两个问题：Recent 100 是否够用，以及 Token Merge 加 Inner Trans 是否比直接 concat 更好。
- 系统部分披露了混合精度、activation recompute、分层存储和 KV Cache 的量化收益，能看出收益来自算法-系统协同。
- 双域线上 A/B 降低单业务偶然性，广告与电商指标也分别贴近各自商业目标。

### 不足

- 数据集和代码未公开，`5.2B samples` 之外的采样规则、负样本构造、特征哈希、embedding 维度、优化器、学习率、batch size、训练轮数和早停策略不足，难以精确复现。
- Table 1 没有 MIMN、LMN、MARM 等记忆增强模型，尽管 related work 已把它们作为主要路线讨论；因此 LONGER 相对“间接建模”的优势主要来自与两阶段/短序列/普通 Transformer 的比较。
- Global Tokens 的四类来源没有单独消融，无法判断收益更多来自候选感知、用户身份、CLS 正则还是高阶交互压缩。
- 线上 A/B 未披露流量比例、实验周期、置信区间、检验方法、对照模型版本和分人群结果；双域一致性强，但统计证据不完整。
- scaling 分析显示 AUC 随序列长度、参数和 FLOPs 上升，但这是当前架构和业务范围内的经验趋势；不能推广为跨业务、跨特征体系和无限资源下的普适定律。

> [!warning] 证据边界
> 论文支持“在 ByteDance 广告与电商的工业条件下，LONGER 优于所列基线并已部署”。它不自动证明 Global Tokens 对所有推荐业务都是最优，也不能替代公开数据集或跨平台的因果复现。

## 关键图表解读

- **Figure 1：总体架构**。长序列先进入 Token Merge 和 Inner Trans，Global Tokens 与采样后的序列 token 共同作为 query；首层 cross causal attention 面向完整序列，后续 self causal attention 在压缩序列中建模高阶依赖。图的关键是 Global Tokens 不是普通特征拼接，而是具有全注意力感受野的锚点。
- **Figure 2：训练框架**。数据经 Fountain 预处理后分发给多个 GPU runner，稠密和稀疏参数同步更新；GPU 侧统一参数存储，配合 HBM/CPU/SSD 分层缓存。图强调消除 Parameter Server 边界，减少通信和参数更新延迟。
- **Figure 3：KV Cache Serving**。用户序列的 key/value 预计算并缓存，候选 item 的 Global Token 只与缓存序列做注意力。该设计利用了“同一用户序列在同一请求内跨候选不变”的结构，解释了为什么长序列在线打分能复用计算。
- **Figure 4：序列长度 scaling**。论文描述增加 token 数持续提高 AUC、降低 LogLoss，并呈 power-law；更深模型从更长序列中获得更多收益，但收益随深度增加出现边际递减。该图支持长序列本身有信息量，而不是只靠更大模型。
- **Figure 5：参数与 FLOPs scaling**。固定 2 层和 2000 序列长度时，AUC 随参数量上升，报告 R²=0.9987；固定宽度 32 时，AUC 随层数与序列长度带来的 FLOPs 上升，报告 R²=0.9967。两条曲线说明在当前范围内容量与计算仍能兑换精度，但没有证明当前范围外不会饱和。

## 值得追踪的引用

- **StreamLLM / attention sinks**：LONGER 用 Global Tokens 稳定长上下文注意力的理论来源，可对比推荐场景中 attention sink 是否同样出现。
- **Perceiver 与 Q-Former**：与 LONGER 的 query 压缩思想相关，可研究 learnable query 在推荐序列中失效的原因。
- **M-FALCON**：KV Cache serving 的先例，可追踪候选感知缓存与候选 batching 的工程细节。
- **HSTU 与 Wukong**：推荐模型 scaling law 的近邻证据，可比较序列 Transformer、特征交互模型和 LONGER 的资源-收益曲线。
- **MIMN、LMN、MARM**：论文讨论但未直接对照的记忆增强路线，是判断 LONGER 是否真正优于间接建模的关键。

## 术语与句式积累

### 术语

- **GSU / ESU**：两阶段长序列建模中的 General Search Unit 和 Exact Search Unit；GSU 先从超长序列检索 top-k 相关行为，ESU 再对短序列精细建模。LONGER 不使用独立 GSU，而是通过 Token Merge、Global Tokens 和 cross attention 直接查询压缩后的长上下文。
- **Global Tokens**：目标 item、CLS、UID 和高阶交互特征组成的辅助 token，拥有全注意力感受野，用于锚定历史、上下文与候选。
- **Token Merge**：相邻 token 分组压缩为更短序列的机制。
- **Inner Trans**：在 Token Merge 组内执行轻量 Transformer 编码，缓解直接 concatenation 造成的局部交互不足。
- **Hybrid causal attention**：首层 cross causal attention 面向完整序列，后续层 self causal attention 面向采样/压缩序列。
- **GPU-synchronous dense/sparse training**：稠密与稀疏参数在同一 GPU 同步框架中更新，减少 Parameter Server 带来的通信和 staleness。

### 可复用句式

- 讨论长序列时可以写：长序列的价值不只在“更长”，还在于能否把候选、近期行为和全局兴趣放进同一个注意力目标。
- 讨论效率时可以写：先用局部 Token Merge 降低序列长度，再用少量 Global/Recent queries 查询完整序列，是比全序列 query 更划算的长上下文设计。
- 讨论部署时可以写：算法压缩只有与缓存、混合精度和同步训练协同，才能转化为可上线的吞吐与延迟收益。

## 复现清单

### 数据

- Douyin Ads CVR 数据为 2024-10-16 至 2025-02-23 的 130 天子集，共 5.2B 样本；前 123 天训练，后 7 天评估。
- 输入包含 UID、gender、超长行为序列和候选广告 item；行为包含 page view、click、conversion，item 特征包含广告内容、展示上下文和元数据。
- 数据未公开，无法直接复现官方规模；可在自有广告/电商日志中重构时间切分、行为类型和候选 item 特征。

### 代码

- 论文未给出官方代码仓库、配置文件、训练脚本或服务实现。
- 需要自行实现 Token Merge、Inner Trans、Global Tokens、hybrid causal attention、activation recompute 和候选级 KV Cache。

### 环境

- 论文说明离线实验使用 A100 GPU 集群，系统基于 TensorFlow；activation recompute 通过 `custom_gradient` 实现。
- 训练采用 BF16/FP16 混合精度；稀疏 embedding 分层存储到 HBM、CPU memory 和 SSD。
- 未披露具体 GPU 数量、拓扑、网络带宽、存储吞吐、批处理大小和在线服务硬件。

### 关键超参数与缺失信息

- 已知效率配置：典型序列长度 2000、embedding 维度 32；Token Merge 使用 group size 4 或 8；Recent query 数量为 50-250，主配置为 100。
- 已知消融对比：Recent 100 为 1.91×10^9 FLOPs；Recent 250 为 3.52×10^9 FLOPs；`LONGER(w/o Merge, 2000)` 为 3.73×10^9 FLOPs。
- 缺失信息：Global Tokens 数量、四类 token 的维度、Inner Trans 层数与隐藏维度、采样窗口、mask 细节、KV Cache 更新频率、失效策略、序列存储格式和线上一致性方案。

### 改良设想

- 对 Global Tokens 做四因素消融，分别测试 target item、CLS、UID 和高阶交互特征的独立与组合贡献。
- 在同参数量和同 FLOPs 预算下加入 MIMN、LMN、MARM、HSTU 与 vanilla Transformer 对照，分离“长序列信息收益”和“系统协同收益”。
- 把 Recent 100 扩展为按行为类型、时间衰减和候选相似度加权的动态 query，并用多种子统计其稳定性。
- 报告序列长度、KV Cache 命中率、候选数、显存、吞吐和 p99 latency 的联合曲线，寻找不同业务的部署拐点。
- 增加冷启动、新广告、低活用户和跨域行为的分层线上评估，检验 Recent query 是否系统性偏向活跃用户。

## 关联

- [[Clippings/抖音广告&电商LONGER无GSU的End2End长序列建模.md]]：中文拆解稿，补充了 GSU/ESU 对照、全局 token 组成和注意力流程的可读解释；信息仍以论文 PDF 为准。
- [[MakeItLongKeepItFast：万级序列的线性复杂度建模]]：字节长序列另一路线，可与 LONGER 的全局 token 与查询压缩对照。
- [[LASER：分段目标注意力突破长序列延迟墙]]：小红书的系统与算法协同方案，同样处理长序列在线延迟。
- [[生成式推荐：用万亿参数序列转换器 HSTU 替代 DLRM，验证推荐领域的缩放定律]]：推荐系统 scaling law 的直接对照。
- [[超长行为序列建模的工程解法]]：长序列建模的共性工程问题。
