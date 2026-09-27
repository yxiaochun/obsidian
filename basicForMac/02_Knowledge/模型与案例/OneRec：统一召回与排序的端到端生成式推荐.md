---
创建日期: 2026-09-15
更新日期: 2026-09-26
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
  - 语义ID
来源: "[[2025Kuaishou_OneRec.pdf]]"
状态: 待复核
证据强度: 单篇工业论文证据
待验证问题:
  - "reward model 偏差和多目标指标不足如何缓解？"
  - "论文正文声称相对 TIGER-1B 和未对齐 OneRec-1B 的部分提升百分比，均与 Table 1 原始数值换算不一致，需确认正确口径。"
  - "RM 同时用于离线评估和偏好对构造时，指标放大与偏好循环是否可控？"
---

# OneRec：统一召回与排序的端到端生成式推荐

## 论文信息

| 项 | 内容 |
| --- | --- |
| 论文标题 | OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment |
| 作者/机构 | Jiaxin Deng、Lejian Ren、Shiyao Wang、Qigen Hu、Kuo Cai、Weifeng Ding、Qiang Luo、Guorui Zhou；快手 |
| 时间/版本 | 2025-02-26；arXiv:2502.18965v1。PDF 内 ACM 会议字段仍是模板占位，本卡按 arXiv 版本评估 |
| 来源 | [arXiv:2502.18965](https://arxiv.org/abs/2502.18965)，本地 PDF 见 [[2025Kuaishou_OneRec.pdf]] |
| 领域 | 生成式推荐、语义 ID、自回归列表生成、偏好对齐 |
| 与研究方向的关联 | 提供端到端生成式推荐替代级联召回与排序的工业证据，核心组合是平衡语义 ID、session-wise 生成、SparseMoE 与迭代 DPO。 |
| 精读判定 | 精读。论文与研究方向直接相关，且报告快手主站短视频场景的线上 A/B 结果。 |

## 一句话创新点

OneRec 用一个 SparseMoE encoder-decoder 直接生成一个高价值 session，而不是在传统召回和排序级联中只做候选选择，并通过个性化 reward model 从自身 beam search 输出构造 self-hard negatives，再迭代执行 DPO 对齐。

## 七问笔记

| 问题 | 回答 |
| --- | --- |
| 研究背景 | 工业推荐常用召回、粗排和精排级联来平衡效率与效果；各阶段独立训练和截断会让上游结果成为下游上限，已有的生成式检索又主要停留在召回阶段。 |
| 研究目的 | 把级联管线压缩为单阶段端到端生成，同时解决 session 内候选连贯性、模型容量和推荐场景难以天然获得正负偏好样本的问题。 |
| 创新点 | 以 session-wise list generation 取代 next-item prediction，并把 SparseMoE、平衡语义 ID 和基于个性化 reward model 的 Iterative Preference Alignment 组成一个可上线框架。 |
| 研究方法 | 用户正反馈历史映射为 3 层语义 ID；T5 式 encoder 建模历史，decoder 自回归生成目标 session 的语义 ID；先做 next-token prediction，再用 reward model 对 128 个 beam search 候选选最高分和最低分构造 DPO 偏好对，并迭代更新模型。 |
| 实验数据 | 大规模快手工业数据训练离线模型；线上 A/B 在快手短视频推荐主场景进行，实验组占 1% 主流量。论文未披露离线数据的具体规模、日期窗口和划分协议。 |
| 结果结论 | 离线 Table 1 中 OneRec-1B 的 max swt 为 0.1529、max ltr 为 0.0660；加 IPA 后分别达到 0.1933 和 0.1203。线上 OneRec-1B+IPA 相对现有多阶段系统带来 Total Watch Time +1.68%、Average View Duration +6.56%。 |
| 总体评价 | 工业落地证据强，方法组合完整，尤其是把推荐缺少显式偏好对的约束转成 reward model 加 self-hard negatives；但离线指标由预训练 RM 估计，线上只报告两个时长类聚合指标，多目标、长期体验和 RM 偏差仍需复核。 |

## 方法拆解

### 平衡语义 ID

- item 表征来自与用户-item 行为分布对齐的多模态 embedding，而不是仅用文本标题表征。
- 先用 residual K-means 逐层量化残差：第 1 层残差是原始 item embedding，后续每层选择最近中心并把该中心从残差中减去，形成 $L$ 层语义 ID。
- 每层用 Balanced K-means 分配 $w=|\mathcal V|/K$ 个 item 给每个码字，缓解 RQ-VAE 类残差量化常见的 hourglass 现象和码字使用不均衡。
- 论文配置为 $K=8192$、$L=3$。这意味着每个 item 被表达为一个 3 级语义 ID，自回归生成时先预测粗层码，再逐步细化。

### Session-wise 生成

- session 定义为一次请求返回的短视频列表，论文描述为 5-10 个视频；离线实现固定 $m=5$ 个目标 item，历史行为长度 $n=256$。
- 高价值 session 的筛选条件是：实际观看视频数至少 5、session 总观看时长超过阈值、用户出现点赞、收藏或分享等交互。
- encoder 输入用户有效观看、点赞、关注、分享等正反馈历史对应的语义 ID；decoder 以 BOS 起始，按下一个 token 目标生成整个 session 的语义 ID。
- 训练损失是 session 内所有 item 语义 ID 的 cross-entropy next-token prediction。与逐 item 打分不同，列表内候选的相对关系和顺序进入同一个条件序列。

### SparseMoE 与模型扩展

- decoder 的 FFN 替换为 24 专家 SparseMoE，每个 token 通过 top-2 router 激活 2 个专家。
- 该设计在不线性增加推理 FLOPs 的前提下扩展参数容量。论文称推理时只激活约 13% 参数。
- 模型规模实验从 0.05B 扩展到 1B。相对 0.05B，0.1B 带来 max accuracy +14.45%，0.2B、0.5B 和 1B 再分别增加 5.09%、5.70% 和 5.69%，说明在实验范围内持续受益。

### Iterative Preference Alignment

- 先训练 session-wise reward model：对 session 内每个 item 做 target-aware 表示，再经 self-attention 融合列表内信息，最后用多塔 Sigmoid MLP 预测 swt、vtr、wtr 和 ltr，并用 BCE 损失训练。
- 对同一用户用当前 OneRec 做 beam search，生成 $N=128$ 个候选 session；reward 最高者作为 winner，最低者作为 loser，构造 DPO 偏好对。
- 新模型从上一轮模型初始化，损失为 next-token loss 与 DPO loss 的加权和。论文未披露 DPO 系数 $\lambda$、$\beta$ 和多目标 reward 聚合方式。
- DPO 样本比例 $r_{\text{DPO}}=1\%$。1% 达到最高观测性能平均约 95%，5% 只带来有限增益但需要 5 倍 GPU 资源，因此作者选择 1%。

### 线上部署

- 系统分为离线训练、在线推理和 DPO sample server。交互日志先用于训练种子模型，收敛后加入 DPO；模型参数同步到在线推理与 DPO 采样服务。
- 训练使用 Adam、初始学习率 $2\times 10^{-4}$、XLA 和 bfloat16 混合精度，硬件为 NVIDIA A800。
- 推理使用 key-value cache、float16 量化和 beam size 128，以平衡生成质量与时延。

## 实验证据

### 离线对比

离线基线包括 SASRec、BERT4Rec、FDSA、TIGER-0.1B/1B，以及 DPO、IPO、cDPO、rDPO、CPO、simPO 和 S-DPO 等偏好对齐方法。论文用预训练 reward model 在每次迭代的随机测试样本上估计 swt、vtr、wtr 和 ltr 的 mean/max。

| 模型 | max swt | max ltr | 论文中的定位 |
| --- | ---: | ---: | --- |
| TIGER-1B | 0.1368 | 0.0579 | 最强 point-wise 生成基线 |
| OneRec-1B | 0.1529 | 0.0660 | session-wise list generation |
| OneRec-1B+IPA | 0.1933 | 0.1203 | session-wise 生成加迭代偏好对齐 |

论文正文称 OneRec-1B+IPA 相对未对齐 OneRec-1B 的 max swt 和 max ltr 分别提升 4.04% 和 5.43%，还称 OneRec-1B 相对 TIGER-1B 的 max swt 与 max ltr 提升 1.78% 和 3.36%。这些百分比与 Table 1 原始数值的直接换算不一致；本卡在结论中优先引用 Table 1 原始值，不沿用论文正文的相对提升百分比。

### 消融与扩展

- **DPO 样本比例**：1% 到 5% 的扫描显示，进一步增加样本比例对 swt、vtr、wtr、ltr 的 mean/max 收益有限；5% 样本比例消耗 5 倍 GPU 资源。
- **偏好对齐方法**：IPA 的 max swt 和 max ltr 优于全部对比方法；DPO、cDPO、rDPO、CPO、simPO 和 S-DPO 等多数只在其中一项或均值上有改善，部分设置未稳定超过未对齐 OneRec。
- **模型规模**：0.05B 到 1B 持续提升，但论文没有做 FLOPs、显存、时延和收益的统一 scaling 曲线表，只报告 accuracy 增益。

### 线上 A/B

| 模型 | Total Watch Time | Average View Duration |
| --- | ---: | ---: |
| OneRec-0.1B | +0.57% | +4.26% |
| OneRec-1B | +1.21% | +5.01% |
| OneRec-1B+IPA | +1.68% | +6.56% |

实验在快手主站短视频推荐场景进行，实验组占 1% 主流量，对照是当时的多阶段推荐系统。论文未给出显著性检验、分人群结果、多样性和留存指标。

## 关键图表解读

- **Figure 1**：对比统一架构与级联架构。级联从约 $10^{10}$ 候选经过检索、粗排、精排逐级降到几十个结果；OneRec 则用 encoder-decoder 直接生成几十个结果，减少中间截断。
- **Figure 2**：上半部分展示语义 ID 序列进入 encoder、decoder 自回归生成 session；下半部分展示 IPA 循环，即当前模型生成多个响应、RM 打分、选高低分构造偏好对、训练下一轮模型。
- **Figure 3**：线上部署框架。日志收集与预处理进入分布式训练，训练后的 1B 参数分别同步给在线推理模型和 DPO sample server；DPO server 消费生成响应和 reward，完成偏好样本回流。
- **Figure 4**：DPO 样本比例消融。多数指标在 1% 后只有小幅波动，说明主要收益来自加入偏好对本身，而不是不断增加采样量。
- **Figure 5**：展示每层语义 ID 的 softmax 分布。IPA 让高 reward item 的目标语义 ID 出现明显置信度偏移；第 1 层熵为 6.00，第 2 层平均熵为 3.71，第 3 层熵为 0.048，说明逐层解码的不确定性递减。
- **Figure 6**：展示 0.05B 到 1B 的 accuracy 增益，支持 OneRec 在实验范围内具备模型缩放收益。

## 批判性分析

### Why 层面

- **为什么要替代级联？** 级联系统中每个阶段先优化自己的截断目标，上游候选集成为下游的硬上限。OneRec 把候选选择变成同一模型内的序列生成问题，理论上让用户历史、候选关系和业务目标直接耦合。
- **为什么用 session 而不是 next-item？** 一次请求本来就要返回一组内容。next-item 目标需要额外规则保证连贯性和多样性，session-wise 目标则把连贯性、多样性和顺序作为监督信号的一部分。
- **为什么需要 reward model？** 一个请求通常只有一次实际曝光，无法同时观察被选列表与未选列表的真实反馈。RM 把模型自生成的高分和低分响应转成偏好对，使 DPO 可以在缺少显式人工偏好标注的推荐场景中使用。
- **为什么 DPO 样本只取 1%？** 每个偏好对要对同一用户做 128 路 beam search 和 RM 打分。论文显示收益在 1% 后饱和，而成本线性增加，因此把昂贵对齐限制在小比例样本上。

### 优点

- 问题定义贴近工业约束，完整覆盖语义 ID、列表生成、容量扩展、偏好对齐、在线部署和 A/B。
- Balanced K-means 直接口针对残差量化的码字不均衡，比笼统使用 RQ-VAE 更符合工业候选池分布。
- IPA 把 self-hard negative sampling 引入生成式推荐，避免随机负样本过弱，也绕开了推荐请求难以同时获得正负曝光的问题。
- 线上结果来自快手主站短视频场景，并包含模型规模递进，说明不是只在离线小数据集上成立。

### 不足与边界

- **RM 是单点依赖**：RM 既用于离线评估，又用于偏好对构造。若它对时长目标过拟合或继承旧系统曝光偏差，DPO 会持续放大这些偏差，而论文没有分析 RM 校准、误差传播和 reward hacking。
- **多目标仍不平衡**：作者在结论中自述点赞等交互指标不足。线上只报告 Total Watch Time 和 Average View Duration，未覆盖点击、点赞、关注、分享、留存、满意度和多样性。
- **离线指标不是直接真实反馈**：swt、vtr、wtr、ltr 由预训练 RM 估计。虽然这在工业离线评估中常见，但用 RM 作为最终评估再优化 RM 选择样本，存在循环评估风险。
- **可复现信息不足**：论文没有公开数据、代码和完整数据协议；离线数据规模、基线调参、encoder/decoder 层数与隐藏维度、DPO 超参、RM 结构细节和聚合规则未披露。
- **scaling 证据有限**：0.05B 到 1B 持续提升，但未测试更大规模，也未在固定 FLOPs、时延或硬件预算下与 TIGER 做等成本对照。
- **统计信息缺失**：线上 A/B 未报告显著性检验、置信区间、实验周期和新老用户拆分，外部读者只能看到两个指标的相对提升。

> [!warning] 证据边界
> 本卡的线上结论来自快手私有主场景和论文作者报告的 A/B，不能直接外推到其他业务、候选规模或指标体系。离线表中的交互指标是 reward model 估计值；论文正文与 Table 1 之间还存在一处提升百分比口径不一致。

## 值得追踪的引用

- **DPO**：OneRec IPA 的基础目标函数，但推荐场景缺少显式偏好标注，OneRec 用 RM 自造偏好对。
- **TIGER**：语义 ID 加自回归生成的代表基线，用于对比 point-wise 生成与 session-wise 生成的差异。
- **QARM**：提供与用户-item 行为分布对齐的多模态 item embedding，是语义 ID 输入表征的上游。
- **Breaking the Hourglass Phenomenon of Residual Quantization**：说明残差量化中码字使用不均衡的来源，Balanced K-means 直接回应这个问题。
- **Hard negative sampling 理论**：IPA 的高分/低分候选构造方式可视为从模型自身分布采样难负样本。
- **EAGER**：语义与协同信息融合 tokenization 的同期方向，可用于比较平衡语义 ID 是否充分表达行为语义。

## 复现清单

| 项 | 内容 |
| --- | --- |
| 数据 | 需要快手级短视频正反馈日志、多模态 item embedding、session 反馈和线上 A/B 平台；论文未提供公开数据。 |
| 代码 | 论文未提供官方实现或代码链接。 |
| 环境 | NVIDIA A800 训练；XLA、bfloat16 混合精度；推理需要 GPU、KV cache、float16 量化和 beam search 服务。论文未披露 GPU 数量、批处理与延迟细节。 |
| 关键超参 | 每层 8192 个码、3 层语义 ID；session 长度 $m=5$；历史长度 $n=256$；24 专家、top-2 SparseMoE；$N=128$ 候选；DPO 样本比例 1%；Adam 初始学习率 $2\times 10^{-4}$。 |
| 缺失信息 | 训练数据量和划分、encoder/decoder 层数与维度、RM 目标聚合方式、DPO 系数 $\lambda$ 和 $\beta$、基线调参、A/B 周期与显著性。 |
| 改良方向 | 先用真实曝光日志校验 RM；再做 FLOPs 和 latency 约束下的 TIGER 对照；对 DPO 前后报告 RM 与真实指标的背离；补充多目标权重、reward hacking、多样性、留存和长尾候选覆盖分析。 |

## 术语与句式积累

- **session watch time（swt）**：一次请求返回的 session 内用户累计观看时长的估计。
- **session-wise list generation**：一次生成整组候选的语义 ID，而不是逐 item 独立预测。
- **self-hard negatives**：从当前模型 beam search 输出中按 reward 选出的低分响应，用来构造难负样本。
- **Iterative Preference Alignment（IPA）**：生成、RM 打分、构造偏好对、DPO 更新、再从新模型采样的循环。
- **可复用句式**：“推荐系统的偏好对齐难点不是没有 DPO，而是没有天然成对的可观测正负反馈。”；“session-wise 监督把列表连贯性从后处理规则前移到训练目标。”

## 关联

- [[PROMISE：过程奖励模型解锁推荐推理时扩展]]：解决 SID 生成中的语义漂移
- [[OneMall：快手电商多场景统一生成式推荐]]：同路线在电商的延伸
- [[生成式推荐为何开始替代级联管线]]
