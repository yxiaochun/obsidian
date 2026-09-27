---
创建日期: 2026-09-15
更新日期: 2026-09-26
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2026ByteDance_MSN.pdf]]"
状态: 待复核
证据强度: 单篇工业论文证据
精读判定: 精读
待验证问题:
  - "PKM 的负载均衡和检索质量在不同特征分布、更大记忆规模和非搜索任务中是否稳定？"
  - "Sparse-Gather 与 AirTopK 在不同硬件、QPS 和批量下能否持续维持与基线相近的时延？"
  - "Memory-Gated Fusion 与 SparseMoE、动态计算深度等其他稀疏机制组合何时才有正收益？"
---

# MSN：产品键记忆实现低成本稀疏扩展

> [!info] 一句话创新点
> MSN 把推荐模型的容量扩展从「路由到少量大专家」改成「从大参数化记忆中做稀疏值检索」，用 Product-Key Memory 把检索复杂度压到亚线性，再用 tanh 门控把检索到的个性化表示叠加到下游特征交互模块。

## 论文信息

- **标题**：MSN: A Memory-based Sparse Activation Scaling Framework for Large-scale Industrial Recommendation
- **作者/机构**：Shikang Wu、Hui Lu、Jinqiu Jin、Zheng Chai、Shiyong Hong、Junjie Zhang、Shanlei Mu、Kaiyuan Ma、Tianyi Liu、Yuchao Zheng、Zhe Wang、Jingjian Lin；ByteDance Search & AML。前四位作者贡献相同。
- **版本**：arXiv:2602.07526v1，2026-02-07；本地来源为 [[2026ByteDance_MSN.pdf]]
- **公开链接**：[arXiv:2602.07526](https://arxiv.org/abs/2602.07526)
- **领域**：工业推荐系统、推荐模型扩展、稀疏激活、记忆网络
- **与研究方向的关系**：论文不是生成式召回/排序本身，但提供了与 SparseMoE 平行的稀疏扩展路线。对生成式推荐基础模型的容量扩展、参数成本和部署约束具有直接参照价值。

## 核心问题

推荐模型扩容常受在线低时延和高吞吐约束。Sparse MoE 通过稀疏激活降低计算量，但每个激活专家是完整 FFN，权重访存随 hidden size 呈平方增长；可用专家数因此受限，每个用户可路由到的个性化组合也受限。MSN 改用大规模 key-value memory 中的稀疏参数检索：容量主要放在记忆值表里，样本只做轻量查询、Top-k 检索和门控融合，而不是激活大 FFN。

## 七问笔记

| 问题 | 回答 |
| --- | --- |
| 研究背景 | 抖音搜索等工业排序系统已有 RankMixer 类大模型，但继续扩参需要在质量、计算、访存和部署时延之间权衡。 |
| 研究目的 | 在几乎不显著增加单样本计算的前提下扩大模型容量，并让检索到的记忆表示带来细粒度个性化。 |
| 创新点 | 用 Product-Key Memory 做亚线性记忆检索，用记忆值生成门控信号调制原输入，把稀疏记忆作为可插拔模块接入 FFN 或 SMoE 主干。 |
| 研究方法 | 查询向量在 row/column 两个子键空间分别 Top-k，组成 $k^2$ 个候选后再取 Top-k；softmax 聚合对应记忆值得到 $v_o$，随后按 $\tilde{x}=x_{in}\odot\tanh(v_o)$ 融合进下游模块。训练侧加入 LayerNorm、学习率 warm-up 和 key over-parameterization。 |
| 实验数据 | 抖音搜索日志，包含超过 1,000 个异质特征，覆盖数亿用户、数十亿视频和每日数十亿交互；离线用连续三周数据，训练环境为 96 GPU。 |
| 结果结论 | 相同激活参数量下，MSN-FFN 的 Click/Finish QAUC 分别比 DLRM-MLP 高 0.54%/0.33%，优于 Wukong、FFN、GatedFFN 和两种 SMoE 配置。抖音搜索 A/B 中活跃天数 +0.0503%、换词查询率 -0.1337%、人均观看时长 +0.2958%、完播率 +0.2071%。 |
| 总体评价 | 工业证据较强，覆盖离线对比、组件消融、记忆激活分布和在线 A/B；但数据、代码、服务时延和部署硬件披露有限，结论主要集中在抖音搜索 CTR 场景，不能直接外推到所有推荐任务。 |

## 方法拆解

### Product-Key Memory 检索

- 记忆表 $V\in\mathbb{R}^{n\times d}$ 存放可学习 value 向量。若直接对全表做内积，检索复杂度是 $O(nd)$。
- MSN 用两组子键 $K_{row},K_{col}\in\mathbb{R}^{\sqrt{n}\times d}$。查询网络把输入映射成 $q_{row}$ 和 $q_{col}$，分别在两个子键空间取 Top-k。
- 两组行/列索引组成 $k^2$ 个候选，再按 $S_{row}[i]+S_{col}[j]$ 选出最终 Top-k 记忆值，最后做 softmax 加权求和得到 $v_o$。
- 该分解把全表检索复杂度降到 $O(\sqrt{n}d)$，使记忆容量可以扩展到很大规模而不把检索成本线性放大。

### 与 FFN 和 SMoE 的结构差异

- 两层 FFN 的输出可以写成隐藏激活对投影矩阵的加权和；MSN 的输出则是对大规模记忆值的稀疏加权聚合。论文把 MSN 视为稀疏近似版 FFN，但容量主要放在 value 空间。
- SMoE 输出是若干完整 FFN 专家的组合。专家参数随 hidden size 平方增长，专家数量受到显存和访存约束。
- MSN 的激活单元是 $d$ 维 value 向量而非完整 FFN，参数扩展更多体现在记忆表规模；激活计算集中在检索、聚合和门控，因此更适合受带宽限制的在线环境。

### 稳定与均衡训练

- **LayerNorm**：在相似度计算前分别归一化 query 和子键，缓解分数方差过大导致的 softmax 尖峰。
- **学习率 warm-up**：前 50k 步从基础学习率的 0.1% 线性升到基础值，早期保持低梯度，鼓励更多 value slot 被探索。
- **Key over-parameterization**：对 $K_{row}$ 和 $K_{col}$ 增加可学习线性变换 $\Theta_{row}K_{row}$ 和 $\Theta_{col}K_{col}$。即使某些 key 未被当前批次直接选中，梯度仍可通过 $\Theta$ 传播，缓解负载不均衡和记忆塌陷。

### Memory-Gated Fusion

- 论文实验中直接把 $v_o$ 作为后续输入，或与 $x_{in}$ 拼接，只有很小收益。作者认为 $d\gg k$，检索到的少量记忆值不足以替代或完整补充原输入。
- MSN 因此将 $v_o$ 作为稀疏门控信号：$\tilde{x}=x_{in}\odot\tanh(v_o)$。原输入信息被保留，记忆表示只负责调制特征交互。
- 模块插入到 per-token FFN 或 SMoE 前面，实践中配置 1 或 2 层 MSN 的质量/成本折衷最好。

### 部署优化

- **Sparse-Gather**：把稀疏访问、权重计算和梯度更新融合到一个 kernel，减少重复 memory I/O。
- **AirTopK**：原生 TensorFlow Top-k 使用 radix sort；MSN 的典型检索规模是 1024 个候选中取 32，改用 AirTopK 后平均/P99 检索开销从 13ms/26ms 降到接近生产基线水平。

## 实验证据

### 设置与指标

- 离线任务为抖音搜索 CTR 预测，主指标是 Query-Level AUC（QAUC），即在每个 query 内计算 AUC 后求平均。论文认为该指标比全样本 AUC 更贴近搜索排序的在线表现。
- 同时报告激活参数、总参数、激活比例和 batch size 2048 下的训练 FLOPs。
- 主干是抖音搜索最新 Personalized Ranking Model，基于 RankMixer。MSN-FFN 和 MSN-SMoE 分别把记忆门控模块插入 FFN 或 SMoE 前。
- 默认配置：记忆规模 $n=256^2$，检索大小 $k=32$，MSN 层数 $L=2$；Table 2 的组件消融则使用 1 层 MSN 作为默认对照。

### 离线主对比

| 模型 | Click ΔQAUC | Finish ΔQAUC | 激活/总参数 | FLOPs/batch |
| --- | ---: | ---: | ---: | ---: |
| DLRM-MLP | 0 | 0 | 118M/118M | 520G |
| Wukong | +0.25% | +0.13% | 127M/127M | 567G |
| FFN | +0.35% | +0.21% | 120M/120M | 529G |
| GatedFFN | +0.35% | +0.21% | 122M/122M | 541G |
| SMoE 2-in-4 | +0.38% | +0.22% | 120M/194M | 543G |
| SMoE 2-in-6 | +0.43% | +0.25% | 120M/267M | 546G |
| MSN-FFN | +0.54% | +0.33% | 122M/252M | 556G |
| MSN-SMoE | +0.50% | +0.31% | 122M/294M | 560G |

增量均相对 DLRM-MLP。MSN-FFN 在相近激活参数预算下优于所有对比方法；MSN-SMoE 参数总量更大，但 Click/Finish QAUC 略低于 MSN-FFN，说明两种稀疏机制直接堆叠不一定更好。

### 组件消融

| 消融主题 | 设置 | ΔQAUC | 激活/总参数 | 记忆参数占比 |
| --- | --- | ---: | ---: | ---: |
| MSN 层数 | 1 层 | 0 | 122M/194M | 37.1% |
| MSN 层数 | 2 层 | +0.05% | 122M/252M | 51.6% |
| MSN 层数 | 4 层 | -0.05% | 122M/327M | 62.9% |
| 记忆规模 | $n=512^2$ | +0.04% | 122M/398M | 69.3% |
| 记忆粒度 | pertokenV | +0.06% | 122M/1.19B | 89.7% |
| 检索规模 | $k=64$ | +0.00% | 122M/194M | 37.1% |
| 检索规模 | $k=16$ | -0.01% | 122M/194M | 37.1% |
| 检索规模 | $k=8$ | -0.05% | 122M/194M | 37.1% |
| 门控函数 | 去掉 tanh | -0.02% | 122M/194M | 37.1% |
| 门控函数 | Sigmoid | -0.05% | 122M/194M | 37.1% |

两点值得保留：一是 $k\ge 32$ 后继续扩大检索规模收益很小，但 $k=8$ 会明显截断相关信息；二是过度稀疏化并非免费，MSN 扩到 4 层后总参数增长，反而低于 1 层和 2 层配置。

### 在线 A/B

| 场景 | 活跃天数 | 换词查询率 | 人均观看时长 | 完播率 |
| --- | ---: | ---: | ---: | ---: |
| Overall | +0.0503% | -0.1337% | +0.2958% | +0.2071% |
| Double Column | +0.0481% | -0.275% | +0.3032% | +0.4549% |

基线是已在抖音搜索服务数亿用户数月的优化版 RankMixer 模型。在线收益绝对值小，但方向一致，且同时覆盖留存代理、搜索意图满足和消费深度指标。

## 关键图表解读

- **Figure 1**：展示 MSN 总体链路。输入先生成 row/column 查询，在两个子键空间分别 Top-k；索引笛卡尔组合后再取最终 Top-k，检索并加权聚合记忆值，最后用 tanh 门控调制原输入。
- **Figure 2**：对比两层 FFN 与 MSN 的参数放置。FFN 的容量在 up/down projection 矩阵中；MSN 把大部分容量放入记忆 value 表，只稀疏激活其中少量 slot。
- **Figure 3**：给出基于 RankMixer 的主干结构，并说明 MSN 模块插入在 per-token FFN 或 SMoE 之前，而不是替代整个特征交互骨干。
- **Figure 4**：显示激活记忆值的分布较为均衡，所有记忆值都有足够访问频率。这支撑 LayerNorm、warm-up 和 over-parameterization 的稳定性作用，但图示证据未给出负载熵等定量指标。

## 批判性分析

- **为什么问题成立**：推荐模型扩容不能只看理论参数量。SMoE 虽然降低了计算量，但专家参数是 $d^2$ 级，激活时访存突出，专家池又限制个性化组合。把容量移到 value 向量表，是把「每次计算的展开参数」变成「可检索存储参数」。
- **为什么用 PKM**：全表检索在数十万槽位上不可行。Product-Key 先在两个低维子空间取 Top-k，再合成候选集，用少量额外组合逻辑换取亚线性检索。
- **为什么用乘性门控**：$v_o$ 只由少量 slot 聚合而来，直接替代或拼接原输入会损失原特征交互信息。tanh 门控让原输入继续流向下游，同时允许记忆值抑制或增强某些维度。
- **对照比较合理**：实验没有只和弱基线比，而是把 RankMixer、FFN、GatedFFN、SMoE 和 Wukong 放在同一激活参数量级下。MSN-FFN 的增益不能简单归因于参数更多，因为对比控制了激活参数。
- **证据边界**：所有主实验来自抖音搜索私有日志，数据、代码和完整线上服务配置未公开。论文报告 FLOPs，但没有给出完整的在线 p50/p99 latency、硬件型号和 QPS；作者声称部署开销接近基线，外部读者只能接受定性结论。
- **稳定性证据不完整**：Figure 4 展示了激活分布，但缺少负载熵、塌陷率、训练后期路由漂移或 key 更新覆盖率的量化。key over-parameterization 也没有在 Table 2 中单独消融。
- **在线效果需要谨慎解读**：四项指标都是小百分比增益，论文未在正文中给出置信区间或显著性检验；工业推荐 A/B 中这类量级可能仍有价值，但不能把绝对幅度夸大为颠覆性提升。
- **换位思考**：若重新设计实验，我会补充公开数据集或至少另一个业务场景的迁移实验；报告 latency-matched 和 FLOPs-matched 双对照；单独消融 warm-up、LayerNorm 和 over-parameterization；并对 1、2、4 层 MSN 扫描更大记忆表与不同 $k$ 的交互效应。

## 值得追踪的引用

- **Large memory layers with product keys, NeurIPS 2019**：MSN 的 Product-Key Memory 基础，决定检索复杂度和索引结构。
- **SimVQ, ICCV 2025**：key over-parameterization 的灵感来源，可用于理解未选中 key 的梯度传播和表示塌陷缓解。
- **UltraMem / UltraMemV2**：LLM 侧的稀疏记忆网络路线，可对比记忆层数、参数规模和访存设计。
- **Adaptive Domain Scaling, SIGIR 2025**：论文把 Memory-Gated Fusion 视为广义自适应参数化的一种，可比较场景/用户调制思路。
- [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551)：MSN 的主干来源和最强离线对照组。

## 术语与句式

- **Product-Key Memory**：把全局记忆索引分解成 row/column 子键空间，先分别检索再组合候选，降低大记忆表的 Top-k 检索复杂度。
- **Memory-Gated Fusion**：用检索到的稀疏记忆表示生成门控信号，按元素调制原输入，而不是直接替换或拼接。
- **Sparse-Gather**：融合稀疏访问、权重计算和梯度更新的自定义算子。
- **AirTopK**：面向较小候选集和共享内存的快速 Top-k 算子。
- **QAUC**：先在 query 内计算 AUC，再对所有 query 平均，用于缓解搜索场景下跨 query 样本分布差异。
- **可复用句式**：「稀疏扩展要同时报告参数、FLOPs、访存和检索质量」；「记忆容量放在存储参数中，不等于每个样本都要展开大矩阵」；「稀疏模块堆叠后，优化难度可能吃掉容量收益」。

## 复现清单

| 项目 | 论文给出的条件 | 复现缺口 |
| --- | --- | --- |
| 数据 | 抖音搜索连续三周训练/评估数据，超过 1,000 个异质特征，数亿用户、数十亿视频和每日数十亿交互。 | 原始日志、特征映射和隐私处理规则不可公开复用。 |
| 模型 | RankMixer 主干；MSN 检索、tanh 门控、LayerNorm、warm-up 和 key over-parameterization。 | 完整主干层数、hidden size、tokenizer、特征交互配置和 SMoE 参数未全部披露。 |
| 关键超参 | 默认 $n=256^2$、$k=32$、$L=2$；消融中另用 1 层 MSN；前 50k 步从 0.1% 基础学习率 warm-up。 | 学习率基值、优化器、批大小分布、训练轮次和正则项未完整给出。 |
| 环境 | 96 GPU 混合分布式训练；Sparse-Gather 与 AirTopK 用于部署效率。 | GPU 型号、互联拓扑、参数服务器、服务集群、QPS 和完整延迟测量缺失。 |
| 验证 | 先复现 Table 1 的离线 QAUC，再复现层数、记忆规模、$k$ 和门控消融。 | 私有数据下无法直接验证；公开数据替代可能改变负载分布和收益量级。 |
| 代码 | 论文未提供官方实现链接。 | 需要自实现 PKM、Memory-Gated Fusion、Sparse-Gather 和 AirTopK 或寻找第三方复现。 |

## 关联

- [[RankMixer：token混合让推荐模型MFU提升十倍]]：MSN 的主干来源、SMoE 基线和工业扩展背景。
- [[TokenMixer-Large：七十亿参数在线排序模型]]：另一条 Sparse-Pertoken MoE 扩展路线，可与 MSN 的记忆值检索对比。
- [[堆叠因子分解机实现推荐系统缩放定律：Wukong 在 100+ GFLOPexample 下持续提升质量]]：离线主对比中的另一个缩放基线。
- [[Scaling Law在工业推荐系统的落地路径]]：把 MSN 放入容量扩展、计算预算和部署约束的总体框架。
