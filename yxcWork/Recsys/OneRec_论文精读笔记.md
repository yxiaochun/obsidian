

> **论文标题**: OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment  
> **作者**: Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, Guorui Zhou  
> **机构**: KuaiShou Inc.（快手）  
> **发表**: arXiv:2502.18965, 2025年2月  
> **链接**: https://arxiv.org/abs/2502.18965

---

## 一、核心贡献（3句话版本）

1. **提出了OneRec**，一个基于Encoder-Decoder架构、以稀疏Mixture-of-Experts（MoE）扩规模参数的单阶段生成式推荐框架，用自回归方式直接生成用户可能感兴趣的视频列表，替代传统的多级级联排序 pipeline。
2. **突破了传统point-wise next-item prediction的局限**，提出了session-wise list generation机制，让模型一次性生成一个完整session（5-10个视频），并通过Iterative Preference Alignment（IPA）结合个性化Reward Model与DPO对齐用户偏好，解决了推荐场景下正负样本无法同时获取的痛点。
3. **在快手主场景完成大规模工业部署**（亿级DAU），离线实验中max swt比TIGER-1B提升1.78%、max ltr提升3.36%；在线A/B测试Total Watch Time提升1.68%、Average View Duration提升6.56%。

---

## 二、问题定义与动机

### 2.1 现有方法的痛点

现代推荐系统为了平衡效率与效果，普遍采用**多级级联排序架构**：召回（Retrieval）→ 粗排（Pre-ranking）→ 精排（Ranking）。每一阶段从上游接收的候选集中选出top-k传递给下一阶段。这种架构存在根本性问题：

- **各阶段独立优化**，每一阶段的效果天然成为后续阶段性能的上界，整体系统性能被前置环节"天花板"限制。
- 虽然已有工作尝试增强级联阶段间的交互（如RankFlow、GemNN等），但本质上仍维护着传统的级联范式，未能从根本上打破瓶颈。

### 2.2 生成式检索的局限

近年来，Generative Retrieval（GR）通过自回归方式直接生成候选物品的语义ID，成为 promising 的新范式。TIGER、LC-Rec、EAGER等工作在学术数据集上展现了潜力。然而：

- 现有的生成式模型**仅作为召回阶段的selector**，其推荐精度仍无法与精心设计的多级级联排序器相媲美。
- 传统point-wise生成（逐条预测下一个物品）需要依赖手工规则来保证生成结果的连贯性与多样性，难以在实际工业场景中直接替代精排。

### 2.3 OneRec的解决思路

用**统一的端到端生成模型**取代整个级联排序 pipeline，实现真正的"单阶段推荐"。核心假设是：只要模型容量足够大、生成方式足够合理，生成式模型完全有能力同时承担召回+排序的角色。

---

## 三、方法详解

### 3.1 整体架构

OneRec采用经典的**Encoder-Decoder Transformer**结构（类似T5），整体训练分为两个阶段：

1. **Session-wise List Generation 阶段**：用高质量session数据训练基础生成模型。
2. **Iterative Preference Alignment（IPA）阶段**：基于预训练Reward Model和DPO进行迭代偏好对齐。

![整体框架](https://arxiv.org/html/2502.18965v1/x3.png)

### 3.2 特征工程与语义ID量化（Semantic Tokenization）

**输入特征**：用户侧采用正向历史行为序列（有效观看、点赞、关注、分享等），每条视频用多模态embedding表示。

**语义ID生成**：
- 现有方法（如TIGER）多用RQ-VAE对多模态embedding做残差量化，但存在**code分布不平衡的"沙漏现象"（hourglass phenomenon）**——某些code被过度使用，表达能力受限。
- OneRec提出**Balanced K-means Clustering**：
  - 每一层codebook包含K=8192个cluster中心。
  - 每次迭代时，每个centroid按距离排序，严格分配w=|V|/K个最近且未被分配的item，然后重新计算centroid。
  - 重复直到分配收敛，保证每一层code被均衡使用。
- 最终每个视频被表示为L=3层的语义token序列，如 `<a_9><b_3><c_1>`。

### 3.3 Session-wise List Generation

**核心创新：从point-wise到session-wise**

传统方法只预测"下一个视频"，而OneRec的目标是生成一个**高质量session**（通常包含5-10个视频），其形式化定义为：

$$S := M(H_u)$$

其中 $H_u$ 是用户历史行为序列，$S$ 是目标session。

**高质量session的判定标准**：
- 用户在一个session中实际观看的视频数 ≥ 5
- 用户观看session的总时长超过阈值
- 用户有点赞、收藏、分享等交互行为

**模型细节**：
- Encoder：用Fully Visible Self-Attention编码用户历史行为序列。
- Decoder：用Causal Self-Attention + Cross-Attention自回归解码目标session的语义ID。
- **MoE扩展**：Decoder中的FFN替换为稀疏MoE层，共24个expert，每次前向仅激活top-2个。这使得模型可scale到1B参数，但推理时仅激活约13%的参数。
- 训练目标：标准的Next Token Prediction（NTP）交叉熵损失。

### 3.4 Iterative Preference Alignment（IPA）

这是OneRec最具创新性的模块，解决了推荐场景下DPO应用的核心难题。

**问题背景**：
- NLP中的DPO依赖人工标注的偏好对（chosen vs rejected）。
- 推荐系统中，每个用户请求只有一次展示机会，**无法同时获得正负样本**。

**解决方案**：

**Step 1: 训练个性化Reward Model（RM）**
- 输入：用户行为表示 + 目标session中各视频的target-aware表示（通过target attention融合用户行为）。
- Session内物品通过Self-Attention交互，融合上下文信息。
- 多塔预测：同时预测session watch time（swt）、view probability（vtr）、follow probability（wtr）、like probability（ltr）等多个目标。
- 用大量推荐数据预训练，以Binary Cross-Entropy为损失。

**Step 2: 自硬负样本选择（Self-hard Negative Sampling）**
- 对当前模型 $M_t$，用beam search为每个用户生成N=128个不同response。
- 用RM为每个response打分，选出**最高分作为chosen**、**最低分作为rejected**，构建偏好对。
- 相比随机采样，这种"自挖掘"的硬负样本能更有效地区分优劣。

**Step 3: 迭代DPO训练**
- 每一轮迭代中，仅随机采样1%的数据进行DPO训练（$r_{DPO}=1\%$），其余99%继续NTP训练。
- DPO损失公式：

$$\mathcal{L}_{DPO} = -\log\sigma\left(\beta\log\frac{M_{t+1}(S_u^w|H_u)}{M_t(S_u^w|H_u)} - \beta\log\frac{M_{t+1}(S_u^l|H_u)}{M_t(S_u^l|H_u)}\right)$$

- 每轮训练结束后，更新模型snapshot为 $M_{t+1}$，并用新模型继续生成偏好对，实现**自提升（self-improvement）**。

**关键发现**：仅1%的DPO样本即可带来显著增益，增加比例收益有限但GPU消耗线性增长，因此1%是效率与效果的最佳平衡点。

---

## 四、实验结果与分析

### 4.1 离线性能对比（Table 1）

| 方法类别 | 模型 | max swt ↑ | max ltr ↑ |
|---------|------|-----------|-----------|
| Point-wise判别式 | SASRec | 0.0803 | 0.0604 |
| | BERT4Rec | 0.0706 | 0.0606 |
| Point-wise生成式 | TIGER-1B | 0.1368 | 0.0579 |
| List-wise生成式 | **OneRec-1B** | **0.1529** | **0.0660** |
| + 偏好对齐 | OneRec-1B+IPA | **0.1933** | **0.1203** |

**关键结论**：
1. **Session-wise > Point-wise**：OneRec-1B比TIGER-1B的max swt高1.78%、max ltr高3.36%，证明session-wise建模在保持上下文连贯性上具有显著优势。
2. **IPA效果突出**：相比OneRec-1B基线，IPA带来max swt提升4.04%、max ltr提升5.43%。
3. **IPA优于其他DPO变体**：包括标准DPO、IPO、cDPO、rDPO、CPO、simPO、S-DPO等，IPA在各项metric上均取得最优或次优。

### 4.2 消融实验

**DPO采样比例消融（Figure 4）**：
- $r_{DPO}$ 从1%提升到5%，性能提升非常有限。
- 但5%比例需要5倍GPU资源，**1%比例可获得95%的最大收益，仅消耗20%的资源**。

**模型Scaling消融（Figure 6）**：
- 从0.05B扩展到1B，准确率持续提升。
- 0.05B → 0.1B：max accuracy +14.45%
- 0.1B → 0.2B → 0.5B → 1B：每次扩展均有约5%额外增益。
- **验证了推荐模型同样存在scaling law**。

**预测动态可视化（Figure 5）**：
- 第一层softmax输出熵最高（6.00），后续层逐渐集中（第二层平均3.71，第三层0.048）。
- 这反映了自回归解码的层次化不确定性递减：早期层继承更多不确定性，后续层借助累积上下文约束决策空间。
- OneRec+IPA相比基线，预测分布明显向高reward item偏移，验证了偏好对齐的有效性。

### 4.3 在线A/B测试（快手主场景）

| 模型 | Total Watch Time ↑ | Average View Duration ↑ |
|------|-------------------|------------------------|
| OneRec-0.1B | +0.57% | +4.26% |
| OneRec-1B | +1.21% | +5.01% |
| **OneRec-1B+IPA** | **+1.68%** | **+6.56%** |

- 在亿级DAU的短视频平台主场景，1%流量上进行严格A/B测试。
- **Total Watch Time提升1.68%，Average View Duration提升6.56%**，对平台营收有显著增量贡献。

---

## 五、系统部署架构

OneRec的线上部署包含三大核心组件：

1. **训练系统**：基于XLA和bfloat16混合精度训练，先NTP训练seed model，收敛后加入DPO loss进行偏好对齐。
2. **在线 serving 系统**：参数同步到在线推理模块，采用KV Cache + float16量化减少GPU内存占用；beam search beam size=128平衡生成质量与延迟。
3. **DPO采样服务器**：独立部署，负责实时生成偏好对数据并回传训练系统。

**关键优化**：
- MoE架构使1B模型推理时仅激活13%参数，大幅降低计算开销。
- 语义ID可离线预计算并存储，线上只需查表。

---

## 六、创新点层次分析

| 创新层次 | 具体创新 | 工程价值 |
|---------|---------|---------|
| **原理创新** | 将级联排序统一为单阶段生成任务，用自回归生成替代召回+粗排+精排 | 高风险高回报，若验证成功可根本性简化推荐架构 |
| **结构创新** | Encoder-Decoder + 稀疏MoE扩展；Session-wise list generation替代point-wise | 中等复杂度，MoE已有成熟工程实践，session-wise需改造训练数据pipeline |
| **训练策略创新** | IPA迭代偏好对齐 + 个性化Reward Model + 自硬负样本选择 | 改动相对小，在已有生成模型基础上叠加DPO训练即可 |
| **工程创新** | Balanced K-means量化解决hourglass现象；1%DPO采样实现高效对齐 | 可直接复用，量化方法对生成式检索系统有普适价值 |

---

## 七、局限性与风险

### 7.1 论文明确指出的局限

- **交互指标（如点赞likes）仍有不足**：在线分析显示，虽然watch time显著提升，但like等交互指标改善有限。作者认为未来需要增强多目标建模能力。
- 论文是arXiv预印本，尚未经过严格的同行评审，部分结论需进一步验证。

### 7.2 潜在工程风险

- **Beam Search延迟**：128个beam的串行解码对线上latency要求高的场景（如推荐需<50ms P99）仍是挑战，快手场景下具体延迟数据未披露。
- **MoE通信开销**：虽然推理时仅激活13%参数，但MoE的all-to-all通信在分布式部署中可能带来额外延迟。
- **语义ID更新成本**：当视频库（~10^10量级）发生变化时，需要重新运行Balanced K-means聚类，全量更新的计算成本较高。
- **Reward Model的bias**：RM基于历史数据训练，可能继承并放大现有系统的bias，对新用户/冷启动内容不够友好。

---

## 八、工程落地可行性评估

### 8.1 数据需求

| 维度 | 评估 |
|------|------|
| 训练数据量 | 工业规模（快手亿级DAU、海量交互日志），普通团队难以复现同等规模 |
| 特征依赖 | 依赖多模态embedding（视频内容理解），需要预训练多模态表征模型 |
| 标注成本 | 无需人工标注，利用自然交互行为（观看、点赞、关注）作为信号 |

### 8.2 计算开销

| 阶段 | 评估 |
|------|------|
| 训练GPU | NVIDIA A800，1B参数模型，DPO阶段需额外部署RM和采样服务器 |
| 推理延迟 | 依赖beam search解码，128 beam，具体P99 latency未披露；MoE仅激活13%参数有助于控制 |
| 内存占用 | 1B参数 + KV Cache，float16量化后可接受；语义ID可离线存储 |

### 8.3 实现复杂度

| 维度 | 评估 |
|------|------|
| 依赖框架 | 标准PyTorch/Transformer组件，MoE可用现有库（如Megatron-LM、Fairseq） |
| 是否有代码 | 论文未明确开源，需联系作者或自行复现 |
| 数学难度 | 标准DL知识 + DPO理论，Reward Model训练为常规多目标预测 |
| 系统改造 | 需改造训练数据pipeline（session构建）、在线serving（自回归解码）、特征系统（语义ID索引） |

### 8.4 综合落地评分：⭐⭐⭐⭐（推荐，改动适中，价值明确）

理由：
- 工业界已有快手成功部署先例，验证了工程可行性。
- 核心创新（session-wise生成 + IPA）相对独立，可逐步迁移到现有系统。
- 主要挑战在于latency优化和语义索引维护，但MoE和KV Cache等优化手段已较成熟。

---

## 九、复现步骤拆解

### Phase 1：快速验证（1-3天）

1. **确认代码可用性**：联系作者或搜索GitHub是否有开源实现。
2. **搭建简化版**：在公开数据集（如MovieLens序列数据）上实现基础Encoder-Decoder + NTP训练，验证session-wise生成的可行性。
3. **关键模块验证**：单独验证Balanced K-means量化效果，对比标准K-means的code分布均衡性。

### Phase 2：离线复现（1-2周）

1. **数据准备**：
   - 将公司内部数据构造成session格式（定义session边界和高质量session标准）。
   - 预训练或获取多模态item embedding。
   - 运行Balanced K-means生成语义ID索引。
2. **模型适配**：
   - 基于T5/Transformer架构搭建Encoder-Decoder。
   - 替换Decoder FFN为稀疏MoE（可用tutel、fairseq-moe等库）。
   - 调整embedding维度、序列长度、session大小等超参。
3. **训练策略**：
   - Phase A：纯NTP训练seed model。
   - Phase B：训练个性化Reward Model（多目标BCE loss）。
   - Phase C：IPA迭代DPO训练（建议从$r_{DPO}=1\%$开始）。
4. **离线评估**：与当前线上Baseline对比swt、vtr、wtr、ltr等指标。

### Phase 3：工程化（2-4周）

1. **推理优化**：
   - 模型量化（FP16/INT8）。
   - KV Cache加速自回归解码。
   - 语义ID离线预计算 + 在线查表。
   - 评估不同beam size下的latency-vs-quality trade-off。
2. **特征对齐**：确保训练与 serving 的语义ID完全一致，防止Training-Serving Skew。
3. **压测与上线**：P99 latency满足业务SLA后，按标准AB测试流程上线。

---

## 十、美团业务场景适配建议

### 10.1 外卖/到店推荐场景

**适配分析**：
- OneRec的session-wise生成天然适合"一次请求返回多个结果"的场景（如首页feed、商家列表）。
- **关键挑战**：外卖场景中用户意图随时间快速变化（早餐/午餐/晚餐），且LBS特征极强。

**适配建议**：
- **引入时段与地理位置特征**：在Encoder输入中显式加入时间槽（time slot）和地理位置（POI/区域）编码，让模型感知上下文。
- **动态session长度**：外卖推荐中用户浏览深度通常较浅，建议根据场景调整session长度（如3-5个商家而非10个视频）。
- **实时性保障**：外卖推荐对延迟极度敏感，建议减小beam size或引入投机解码（speculative decoding）加速。

### 10.2 搜索排序场景

**适配分析**：
- OneRec的生成式检索可直接输出item ID，避免了ANN/MIPS检索的精度损失。
- **关键挑战**：搜索场景有明确Query，需同时考虑Query-Document相关性和用户个性化。

**适配建议**：
- **Query作为Encoder输入**：将Query文本编码后拼接到用户行为序列前，作为生成条件。
- **混合索引**：对于长尾Query，生成式检索可能覆盖不足，可保留传统检索通路作为fallback。
- **多目标RM**：搜索场景需同时优化CTR、CVR、GMV等，Reward Model可扩展为多目标加权。

### 10.3 广告投放场景

**适配分析**：
- 广告推荐中CVR极度稀疏，生成式模型对稀疏信号的捕捉能力是关键。
- **关键挑战**：广告主行为建模、预算平滑、出价约束等商业逻辑难以直接嵌入生成过程。

**适配建议**：
- **增强Reward Model**：在RM中显式引入广告价值指标（eCPM、ROI），让DPO对齐商业目标。
- **两阶段生成**：先生成候选广告集合，再经过轻量级ranker进行商业过滤和排序，兼顾生成质量与商业约束。
- **冷启动优化**：新广告缺乏历史交互，可通过广告content embedding生成语义ID缓解冷启动。

### 10.4 通用改进建议

1. **特征层改进**：
   - 引入美团特有的LBS层级特征（商圈、配送范围）。
   - 时段/天气/场景等上下文特征可显式注入Encoder。

2. **训练策略改进**：
   - 针对美团数据规模和分布，调整Balanced K-means的K值和层数L。
   - DPO的$\beta$参数和$r_{DPO}$需在公司数据上调优，论文的1%不一定是最优。

3. **系统集成建议**：
   - 语义ID索引可与现有特征平台打通，离线定期更新。
   - MoE层可与现有分布式训练框架（如内部自研框架）集成，注意all-to-all通信优化。
   - Reward Model可复用现有精排模型的一部分能力，降低训练成本。

---

## 十一、关键洞察与启发

1. **推荐系统也存在Scaling Law**：OneRec从0.05B扩展到1B持续提升，验证了增大模型容量对推荐效果的正向作用。这与LLM的scaling law一致，说明工业级推荐模型可能同样受益于"大力出奇迹"。

2. **Session-wise是生成式推荐的正确打开方式**：Point-wise next-item prediction将推荐退化为序列预测任务，忽略了列表内item间的相互影响。Session-wise让模型直接学习"什么样的列表是好的"，更符合推荐系统的本质。

3. **DPO在推荐中的适配关键在于"构造偏好对"**：OneRec通过RM+自硬负样本+迭代训练，巧妙解决了推荐场景无法同时获得正负样本的难题。这种思路可推广到其他只有隐式反馈的场景。

4. **1%的DPO数据即可带来显著增益**：这说明推荐模型并不需要在全部数据上做复杂的偏好优化，少量高质量偏好对的引导远比大量普通样本有效，对工程落地非常友好。

---

## 十二、参考资料

- 论文原文：https://arxiv.org/abs/2502.18965
- 相关工作：TIGER (NeurIPS 2023), LC-Rec (ICDE 2024), EAGER (KDD 2024), S-DPO (NeurIPS 2024)
- DPO理论基础：Rafailov et al., "Direct Preference Optimization", NeurIPS 2024
