---
创建日期: 2026-09-15
更新日期: 2026-09-26
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2025Meituan_HoMer.pdf]]"
状态: 待复核
证据强度: 单篇工业论文证据
待验证问题:
  - "候选数超过 300 分 shard 后，cross-item 收益是否保持？"
---

# HoMer：同质化Transformer统一序列与集合上下文

# HoMer：同质化Transformer统一序列与集合上下文

## 精读定位

**一句话创新点**：HoMer 把历史行为升级为携带全量请求特征的 panoramic sequence，并把一次请求候选组成 set-wise 样本，用统一的 sequence encoder 与 set-wise decoder 同时建模纵向兴趣演化和平行候选竞争。

**研究关联**：论文直接回答生成式推荐中的两个输入组织问题——序列上下文应该保留多少请求级特征，以及候选集合是否应作为一等上下文进入模型。其“同一 Transformer 处理序列与集合”的路线，对生成式召回、排序和候选集合推理都有架构参考价值。

**证据评级**：A。有大规模私有业务日志、离线对比、消融、扩展曲线和在线 A/B；但数据与部署管线未开源，外推到公开数据集和其他业务时要谨慎。

## 论文信息

- 作者/机构：Shuwei Chen、Jiajun Cui、Zhengqi Xu、Fan Zhang、Jiangke Fan、Teng Zhang、Xingxing Wang，美团。
- 年份/版本：2025，[arXiv:2510.11100](https://arxiv.org/abs/2510.11100)，v2 页眉标注 2025-10-23。
- 领域：CTR 预估、序列建模、item set context、工业推荐架构。
- 业务场景：美团搜索广告的 pre-ranking 到 ranking 链路；论文结论称已部署到主流量，服务数千万用户。
- 来源：[[2025Meituan_HoMer.pdf]]

## 七问笔记

| 问题 | 回答 |
| --- | --- |
| 研究背景 | 工业 CTR 模型常把序列建模与特征交叉拆成 DIN、DCN、MoE 等专用模块。作者认为这种组合在效果、效率和扩展性上都遇到结构异质性问题。 |
| 研究目的 | 消除特征、上下文和架构三类异质性：让序列侧特征与非序列特征同粒度，让预估利用完整候选集合上下文，并用同质 Transformer 替代碎片化模块。 |
| 创新点 | 提出 panoramic sequence 和 set-wise CTR prediction，并用同质 encoder-decoder 把历史全景序列与当前候选集合放进一个模型，同时输出 exposure 和 click 概率。 |
| 研究方法 | 历史行为对齐当时全量特征形成 panoramic sequence；每请求构建一个 set-wise 样本；sequence encoder 用 self-attention 编码兴趣，set-wise decoder 先候选自交互，再 cross-attention 查询历史兴趣；训练用 click loss 加覆盖 pre-ranking 全集的 auxiliary impression loss。 |
| 实验数据 | 美团搜索广告 2025 年 4-7 月日志：420M 请求、超过 39M 用户、约 690K item、1.25B 用户行为；离线公平比较时每个请求最多使用 300 个候选。在线 A/B 使用 20% 流量，时间为 2025-09-08 至 2025-09-15。 |
| 结果结论 | HoMer 离线 AUC 0.8169、Log Loss 0.2425、42.6 GFLOPs、114.9M dense parameters；对 AUC 0.8070 的 Point-wise Baseline 提升 0.0099。在线 CTR +1.99%、RPM +2.46%。初步 kernel fusion 后 MFU 从 7.8% 到 12.2%，在线 GPU 资源降低 27%。 |
| 总体评价 | 论文把“请求级上下文完整性 + 候选集合交互 + 架构同质化”组成一个可部署方案，离线和在线证据较强。不足是结果绑定私有日志与业务管线，缺乏公开复现、方差/显著性、候选大于 300 的系统验证和曝光偏差处理细节。 |

## 方法拆解

### 三类异质性

1. **Feature heterogeneity**：point-wise 模型的行为序列常只保留类别、价格等少量 side features，而候选侧使用完整 user profile、item profile、user-item cross features 和 context features。序列兴趣表征因此比候选表征粗粒度。
2. **Context heterogeneity**：pre-ranking 会产生 tens 到 hundreds 个候选，用户对一个 item 的兴趣受同页其他候选影响；point-wise 样本把候选隔离预估，忽略集合内竞争、比较和替代关系。
3. **Architecture heterogeneity**：序列模型、特征交叉模块、专家层和多任务头反复堆叠，参数和计算难以统一扩展，也容易造成模块间的 see-saw 效应。

### Panoramic sequence

传统序列侧只把若干 side features 附着在 behavior token 上。HoMer 按时间回溯每个历史行为所属请求，把当时的 user profile、item profile、user-item cross features、context features 全部并入行为表征，得到 panoramic sequence。它不是只记录“用户看过什么”，而是记录“用户在当时完整上下文中看到什么”。论文说明可从历史 CTR 预估样本按时间聚合，线上日志服务也能改造为实时构建。

### Set-wise CTR prediction

Point-wise 范式为每个 item 生成样本，带来四类问题：

- 同一请求的 user profile、context、行为序列被重复存储和重复计算。
- 负采样通常只用 isolated exposed corpus，系统性丢掉未曝光 item 中的兴趣信号。
- 单 item 样本无法表达同请求候选之间的竞争与比较。
- serving 需要缓存和去重机制，否则会对共享特征做多次查表和前向计算。

HoMer 将一次请求构建为一个样本。user/context/panoramic sequence 只计算一次；item profile 和 user-item cross features 组成候选集合，在 decoder 中同层交互并并行预测。这样训练样本的组织方式与线上请求的真实决策结构一致。

### Homogeneous-Oriented Transformer

**Sequence encoder**：每个 behavior 先与 extensive side features 一起经过共享 tokenization layer，并加入 position/action embedding。第 $i$ 个行为可概括为

$$
E_i^0=\sigma_1(\phi(f_{pan\_seq,i}))+\sigma_2(\phi(p_i))+\sigma_3(\phi(a_i)),
$$

其中 $\phi$ 是 embedding/tokenization，$p_i$ 是位置，$a_i$ 是点击等行为动作。$L$ 层 encoder block 用 self-attention 输出用户兴趣表示 $E^L$。每个 block 使用 SiLU 激活和 LayerNorm：

$$
E^l=\sigma^l(r(\sigma^l(QK^{\top}V/\sqrt{d_k})+E^{l-1})).
$$

**Set-wise decoder**：每个候选的 item profile 和 user-item cross features 先生成 item-side embedding；user profile 与 context 生成请求级 embedding $H$；两者拼接得到 $D_i^0$。每个 decoder block 先做 cross-item interaction：

$$
\bar D^m=\sigma_D^m(r(\sigma_K^m(Q)K^{\top}V/\sqrt{d_k})+D^{m-1}),
$$

让候选在当前集合内互相注意。随后做 user-item interaction，用候选查询 encoder 输出：

$$
D^m=\sigma_D^m(r(\bar Q\bar K_E^{\top}\bar V_E/\sqrt{d_k})+\bar D^m).
$$

最终 $D_i^M$ 接入两个 MLP：

$$
p_i^{exp}=\mathrm{sigmoid}(\mathrm{MLP}_1(D_i^M)),\quad
p_i^{clk}=\mathrm{sigmoid}(\mathrm{MLP}_2(D_i^M)).
$$

**训练目标**：

$$
\mathcal{L}=\mathcal{L}_{clk}+\lambda \mathcal{L}_{imp},\quad \lambda=1.
$$

$\mathcal{L}_{clk}$ 是曝光样本上的 click cross-entropy；$\mathcal{L}_{imp}$ 覆盖 pre-ranking 生成的全部候选，用曝光 indicator $y_i^{exp}$ 提供曝光监督。这个设计使 cross-item attention 不只从稀疏点击样本学习，也能从完整候选集合的曝光结构学习。

### 部署设计

Pre-ranking 的 99th percentile 候选数不超过 300，因此 shard size 设为 300；多数请求一次模型调用即可完成 set-wise 预估，超过 300 才分 shard。变长序列和集合用 jagged tensor 组织，attention 用 Flash Attention；对基础 block 做 kernel fusion 后，线上 MFU 从 7.8% 到 12.2%，在线 GPU 资源降低 27%。

## 实验核验

离线数据来自美团搜索广告 2025 年 4-7 月日志，样本组织为 request-wise 和 set-wise 数据。指标包括 Log Loss、AUC、dense parameters 和 GFLOPs Per Request；作者采用最多 300 个候选计算每请求计算量。

| 模型 | AUC | Log Loss | GFLOPs | Dense Params |
| --- | ---: | ---: | ---: | ---: |
| Point-wise Baseline | 0.8070 | 0.2464 | 46.8G | 71.8M |
| DCNv2 | 0.8082 | 0.2438 | 99.3G | 159.1M |
| Wukong | 0.8093 | 0.2442 | 93.5G | 120.6M |
| SASRec | 0.8081 | 0.2460 | 45.7G | 58.6M |
| CIM | 0.8134 | 0.2438 | 60.1G | 67.4M |
| Point-wise HoMer | 0.8071 | 0.2464 | 40.9G | 92.3M |
| HoMer | 0.8169 | 0.2425 | 42.6G | 114.9M |

实验包含六类 point-wise 基线：DIN/DCN/MoE 组成的 Point-wise Baseline、SASRec、DCNv2、Wukong、CIM，以及去掉 cross-item interaction block 和 impression loss 的 Point-wise HoMer。所有模型都使用同一 panoramic sequence；Point-wise HoMer 增加 encoder/user-item block 数量，使 FLOPs 与 HoMer 更可比。

在线 A/B 为 20% 流量、8 天，CTR +1.99%，RPM +2.46%。论文还报告仅通过初步 kernel fusion 得到 MFU 7.8%→12.2%、在线 GPU 资源 -27%。这组结果支撑效果与效率两个主张，但未公开方差、置信区间或显著性检验。

## 消融与扩展

### Panoramic sequence 特征

| 侧特征域 | AUC | Log Loss | Side Info |
| --- | ---: | ---: | ---: |
| 无 | 0.8128 | 0.2440 | 0% |
| User | 0.8149 | 0.2434 | 14.8% |
| Item | 0.8154 | 0.2432 | 36.3% |
| User-Item | 0.8152 | 0.2431 | 37.0% |
| Context | 0.8150 | 0.2433 | 11.0% |
| 全部 | 0.8169 | 0.2425 | 100% |

单一特征域都有正贡献，但四类特征共同使用才有最大增益。这说明收益不是某个 magic feature，而是序列表征与候选表征的粒度对齐。

### Cross-item interaction

Figure 5 以 dense parameters 为横轴比较四种模型：HoMer 持续最高；去掉 auxiliary impression loss 后增益变小且高参数区出现回落；去掉 cross-item block 明显退化；Point-wise HoMer 最弱并更早饱和。对应 Table 1 中 Point-wise HoMer AUC 只有 0.8071，而完整 HoMer 是 0.8169。候选集合交互和曝光监督是相乘性设计：前者提供候选间关系通路，后者提供更完整的监督信号。

### 深度与维度

Figure 6 中只加深 encoder，AUC 提升有限；增加 cross-item block、user-item block，尤其两者同时加深，收益更大。这说明瓶颈主要在 set-wise decoder 的集合交互和候选-历史交互容量，而不是单纯延长 sequence encoder。

| Layers | Embed Dim | Token Dim | AUC | GFLOPs | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| L=8, M=8 | 16 | 256 | 0.8150 | 6.9G | 21.2M |
| L=8, M=8 | 32 | 256 | 0.8151 | 8.9G | 35.6M |
| L=8, M=8 | 16 | 512 | 0.8162 | 20.2G | 55.8M |
| L=8, M=8 | 32 | 512 | 0.8164 | 24.3G | 84.5M |
| L=8, M=16 | 16 | 256 | 0.8161 | 11.4G | 28.8M |
| L=8, M=16 | 32 | 256 | 0.8163 | 15.4G | 46.3M |
| L=8, M=16 | 16 | 512 | 0.8167 | 34.8G | 79.9M |
| L=8, M=16 | 32 | 512 | 0.8169 | 42.6G | 114.9M |

Token dimension 从 256 扩到 512 的收益明显大于 embedding dimension 从 16 扩到 32。作者建议优先扩 token 表征容量；但更宽 token 和更深 decoder 会带来明显 FLOPs 与参数增长。

## 关键图表解读

- **Figure 1**：把三类异质性画成同一条因果链。序列特征比非序列特征粗，point-wise 忽略候选集合，专用模块堆叠又放大架构复杂度。
- **Figure 2**：上方 point-wise 样本把同一请求特征重复写入多个 item 样本；下方 set-wise 样本共享 user/context/sequence，仅把 item 侧特征组织成集合，明确显示训练 schema 与线上请求 schema 对齐。
- **Figure 3**：左侧 sequence encoder 负责纵向兴趣；右侧 decoder 先做候选间 self-attention，再用 cross-attention 查询左侧历史表示；两个 MLP 分别输出 exposure 和 click。全图解释了为什么全景序列、集合交互和统一架构可以共存。
- **Figure 4**：GFLOPs-AUC 曲线显示 HoMer 的扩展线整体高于其他模型，尤其在较小计算预算下上升更陡；CIM 是强基线，但继续增加复杂度没有像 HoMer 一样稳定获益。
- **Figure 5**：按 dense parameters 的消融曲线说明 cross-item block 贡献最大，auxiliary impression loss 能让集合交互继续随规模提升；Point-wise HoMer 更早饱和。
- **Figure 6**：depth ablation 显示 set-wise decoder 是扩展重点；只加深 encoder 的边际收益低，cross-item 与 user-item block 同时加深时效果最好。

## 批判性分析

### Why 层面

**为什么要研究这个问题？**  
工业 CTR 已经同时使用序列模型和特征交叉，但两者通常以不同粒度接在一起。序列侧只看少量 side features，会丢失历史请求的决策条件；point-wise 预估又把同一候选集合拆散。作者把这两个损失统一为“异质性”，动机比单纯提出一个新注意力模块更根本。

**为什么用 panoramic sequence 而不是继续做特征交叉？**  
特征交叉发生在当前 user-item pair 上，不能恢复历史行为发生时的完整请求状态。Panoramic sequence 把历史 candidate、当时 user/context 和 cross features 一起带回来，使模型能学习“同一 item 在不同请求上下文下意义不同”。这比事后在最终表征上做交叉更符合决策过程。

**为什么用 set-wise 而不是逐 item 加候选集合特征？**  
逐 item 加入集合特征会再次复制共享表征，也无法自然建模候选之间的两两关系。Set-wise decoder 让每个候选都能看到其他候选，同时共享 sequence/context 计算，训练、存储和推理结构都更一致。

**为什么要 impression loss？**  
点击标签只存在于曝光样本，cross-item interaction 需要理解 pre-ranking 产出的完整候选集合。曝光 loss 用 $y_i^{exp}$ 为全部候选提供监督，使 decoder 学到集合级 exposure 结构，而不是只从少量正负点击样本推断竞争关系。

### 实验设计评价

论文的强处是四层证据闭环：完整模型对比、特征消融、结构消融、在线 A/B。它也没有只报告参数量，而是同时报告 dense parameters 和每请求 GFLOPs，并把 Point-wise HoMer 调到相近 FLOPs。

不足在于：

1. **私有数据限制外推**。美团搜索广告的候选分布、特征宽度、曝光机制和竞价/排序规则都可能影响结果；论文没有公开数据、代码或可复现 pipeline。
2. **缺乏统计不确定性**。离线表没有多 seed、方差或显著性检验；在线 A/B 也未报告置信区间和业务异质性。
3. **曝光偏差未充分讨论**。候选集合、位置、竞价和历史策略都会塑造曝光与点击标签。Impression loss 利用完整 pre-ranking 集合，但论文没有给出位置去偏、counterfactual correction 或策略采样校正。
4. **候选超过 300 的场景未验证**。分 shard 后每个 shard 只看到局部候选集合，作者认为小于 300 的模式可泛化，但没有给出大候选请求的分桶实验。
5. **缺关键 serving 指标**。GPU 资源下降和 MFU 提升很重要，但未见 P99 延迟、吞吐、显存峰值、失败率和不同流量峰值下的表现。

### 换位思考

如果重新组织这篇文章，我会先把“特征粒度错配”和“候选集合决策”形式化为可测量的诊断指标，再进入模型；并在公开或半公开数据上加入一个最小复现。实验上至少补三组对照：cross-item self-attention 与 DeepSets/Graph/set encoder 的比较、candidate shuffling 或 mask 稳定性、以及无 auxiliary impression loss 但使用全曝光集合采样的对照。指标上可补充按用户分组的 GAUC、校准误差和位置归一化指标。

如果迁移到生成式推荐，直接照搬 set-wise decoder 未必必要；更重要的是三点：把生成目标放在请求级候选集合上，把历史 token 与当时的请求特征绑定，并让候选间约束或竞争关系进入解码/评分阶段。HoMer 的 set-wise attention 是判别式实现，但其“请求作为样本、集合作为上下文”的组织方式可以迁移到生成式索引与候选生成。

## 近五年值得追踪文献

- **CIM / Implicit user awareness modeling via candidate item set（SIGIR 2022）**：HoMer 的强基线之一，直接代表候选集合意识建模路线，适合比较集合特征与集合注意力。
- **Recommender Systems with Generative Retrieval（NeurIPS 2023）**：语义 ID 和生成式检索的经典路线，可用于对照 HoMer 的判别式 set-wise 预估与生成式候选生成的边界。
- **Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations（2024）**：HSTU 路线强调统一序列建模与生成式推荐扩展律，可与 HoMer 的同质架构和扩展曲线对照。
- **Wukong: Towards a Scaling Law for Large-scale Recommendation（ICML 2024）**：论文基线之一，提供特征交互模型的扩展证据，适合分析同质架构是否真的比堆叠模块更可扩展。
- **Actions Speak Louder than Words: Task-Free End-to-End Learnable Item Tokenization（CIKM 2024）**：端到端 item tokenization 与 HoMer 的 panoramic tokenization 形成互补，前者优化离散化，后者优化请求上下文完整性。
- **Multi-Behavior Generative Recommendation（KDD 2025）**：把多行为与生成式推荐结合，适合研究 panoramic sequence 是否应扩展为多行为、多任务、多阶段请求上下文。
- **ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation（2025）**：上下文相关 action tokenization 是 panoramic sequence 的生成式对应问题，值得追踪如何把请求特征编入 action token。

## 术语与句式

**术语**：

- Panoramic sequence：携带历史请求全量非序列特征的行为序列。
- Set-wise CTR prediction：以请求内完整候选集合为样本的 CTR 预估范式。
- Cross-item interaction：当前请求候选之间的 self-attention 交互。
- User-item interaction：候选作为 query 对 panoramic sequence 表示的 cross-attention。
- Auxiliary impression loss：对 pre-ranking 全部候选增加曝光监督的辅助损失。
- MFU：Model FLOPs Utilization，衡量模型计算利用率。

**可复用表述**：

- “序列建模不能只记录 item，还要记录 item 出现时的完整请求状态。”
- “候选集合不是噪声，而是用户比较、替代和选择行为的上下文。”
- “架构统一的价值不在单点效果，而在训练、推理和扩展曲线的一致性。”

## 复现清单

### 数据

论文使用私有美团搜索广告日志，没有公开数据集。最小复现需要：request ID、用户画像、item 画像、user-item cross features、context features、按时间排列的行为序列、pre-ranking 候选集合、曝光 indicator 和点击标签。由于原文未说明训练/测试切分细节，外部复现必须自行定义按时间外推或按用户隔离的划分，并避免同一请求泄漏到训练与测试。

### 代码与环境

论文未提供公开代码仓库。可按以下配置先做离线复现：

- Adam optimizer，学习率 1e-4。
- One-epoch training，减少过拟合。
- 辅助曝光损失权重 $\lambda=1$。
- Encoder 层数 $L$、decoder 层数 $M$、token dimension、embedding dimension 按 Table 3 网格控制。
- Flash Attention 与 jagged tensor 用于变长序列/集合；没有这些工程组件时，可先实现 padded attention，但效率结论不可比。

### 需要补齐的复现信息

论文没有给出 GPU 型号、显存、分布式策略、完整特征 schema、embedding/sparse 参数规模、日志采样规则、训练/测试时间切分、在线排名位置处理和分 shard 的完整策略。因此数值级复现需要内部业务日志；公开研究只能复现方法结构和相对消融趋势。

### 改良设想

- 用可学习分桶、层次注意力或局部集合注意力处理超过 300 的候选，验证全集合上下文是否仍带来增益。
- 在公开 CTR/电商数据上构造 request-wise candidate set，对比 HoMer、DeepSets、set transformer、GNN 和 CIM。
- 引入位置、倾向得分或 counterfactual weighting，检查 cross-item interaction 是学到真实竞争关系，还是放大曝光策略偏差。
- 把 exposure head 与 click head 的校准分开评估，并报告 GAUC、校准误差和按候选规模/用户活跃度分桶的结果。
- 为生成式推荐设计 request-aware semantic token：把当时 context、user-item cross features 与 item ID 一起编入 token，再做生成式候选重建。

> [!warning] 证据边界
> HoMer 的效果与效率结论来自美团搜索广告的私有日志和部署管线。公开数据集、候选生成方式、排序策略或流量规模不同时，AUC 增益、MFU 和 GPU 节省都需重新验证。

## 结论与迁移

HoMer 的核心不是“又一个 Transformer”，而是把样本组织、上下文粒度和架构形态一起改掉：序列 token 携带完整历史请求，当前候选以集合形式交互，统一 encoder-decoder 让两个上下文共享计算。对生成式推荐研究最有迁移价值的部分是 request-level sample construction 和 candidate-set conditioning；如果只复用 attention 结构而不恢复全景上下文，效果和扩展性可能都会退化。

## 关联

- [[MixFormer：稠密特征与序列建模协同扩展]]：同样试图减少序列建模与稠密特征建模的割裂。
- [[OneTrans：一个Transformer统一特征交互与序列建模]]：与 HoMer 同属用统一 Transformer 替代专用模块堆叠的路线。
