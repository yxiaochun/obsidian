# AutoML for Large Capacity Modeling of Meta's Ranking Systems 论文精读

> **论文链接**: https://arxiv.org/abs/2311.07870  
> **发表会议**: WWW 2024 Companion (ACM Web Conference 2024)  
> **作者**: Hang Yin*, Kuang-Hung Liu*, Mengying Sun, Yuxin Chen, Buyun Zhang, Jiang Liu, Vivek Sehgal, Rudresh Rajnikant Panchal, Eugen Hotaj, Xi Liu, Daifeng Guo, Jamey Zhang, Zhou Wang, Shali Jiang, Huayu Li, Zhengxing Chen, Wen-Yen Chen, Jiyan Yang, Wei Wen†  
> **机构**: Meta Platforms Inc., USA  
> **arXiv**: 2311.07870 [cs.IR]

---

## Step 1 — 论文核心信息摄取

### 1.1 论文元信息

| 项目 | 内容 |
|------|------|
| **标题** | AutoML for Large Capacity Modeling of Meta's Ranking Systems |
| **作者** | Hang Yin, Kuang-Hung Liu 等 19 人（Meta 排序/AutoML 团队） |
| **发表** | WWW 2024 Companion，新加坡，2024年5月 |
| **级别** | 顶会 companion（短论文，9页），工业界强实践导向 |
| **代码** | 未公开开源代码，但方法基于 Wen et al., 2020 的 Neural Predictor |
| **数据** | Meta 内部生产数据（Instagram CTR、Ad CVR），不公开 |

### 1.2 问题定义

**核心问题**：在 Meta 这种十亿用户规模的排序系统中，如何用 AutoML 自动发现大容量模型的架构和超参数，且**仅通过约 100 次模型评估**就能超越人类专家调优的强基线？

**现有痛点**：
- 排序模型越来越复杂（DHEN 深层堆叠结构），架构决策点和超参数敏感度都在增加；
- 大容量模型的端到端评估成本极高，AutoML 的搜索预算非常有限；
- 生产排期紧张，AutoML 必须在严格时间窗口内交付可部署的模型；
- 人类工程师调优的基线已经很强，AutoML 必须证明其 ROI（投资回报）才能被生产接纳。

**解决思路一句话**：用轻量级 Predictor 替代昂贵的真实训练评估，再让 RL 采样器在 Predictor 上高效搜索，同时通过低精度评估、训练曲线外推、Pairwise Ranking Loss、集成建模等手段把 Predictor 做准，最终只采样约 100 个模型就找到超越人类基线的架构。

### 1.3 核心贡献（3句话版本）

1. **提出了一个基于 Predictor + RL 的采样式 AutoML 框架**，联合优化神经网络架构搜索（NAS）和超参数优化（HPO），专为 Meta 级别的大容量排序模型设计。
2. **解决了大模型评估贵、样本少、生产排期紧的难题**：通过低精度评估（节省 50% 计算）、训练曲线线性外推预测长期收益、Pairwise Ranking Loss 训练 Predictor、集成多任务学习提升泛化，使得 Predictor 能在极少样本下保持高排序相关性（Kendall Tau 0.87+）。
3. **在 Instagram CTR 和 Ad CVR 生产场景中验证**，仅平均采样约 100 个模型即可在强人类基线上再获得最高 **-0.36% NE（Normalized Entropy）增益** 或 **+25% QPS 提升**；发现的 Instagram CTR 大容量模型已通过大规模线上 A/B 测试并取得统计显著收益。

---

## Step 2 — 创新点与已有方法对比

### 2.1 技术创新层次分析

| 创新层次 | 具体技术 | 工程价值 |
|----------|----------|----------|
| **原理创新** | Predictor-based Searcher + REINFORCE RL 的联合优化框架 | 中等——框架思想延续 Wen et al. 2020，但 RL 与 ROI 约束的结合有新意 |
| **结构创新** | DHEN 搜索空间设计（5 类交互模块的层级组合 + 堆叠深度） | 高——直接对应 Meta 生产架构，模块化和可扩展性强 |
| **训练策略创新** | ① 低精度评估 + 排序相关性分析；② 训练曲线线性外推长期 NE；③ Pairwise Ranking Loss；④ 集成多任务 Predictor | **最高**——这些是决定"100 次评估就能赢人类"的关键，落地成本极低但收益明确 |
| **工程创新** | FLOPs 作为跨硬件统一成本代理指标；Predictor 与生产 serving stack 高度兼容 | 高——解决了多硬件环境下 QPS 不可比的问题，保证搜出来的模型能直接部署 |

### 2.2 对比基线分析

论文主要对比了三种搜索策略在相同计算预算（150 trials）下的表现：

| 方法 | CVR Scale-up NE Gain | 特点 |
|------|----------------------|------|
| Random Search | -0.064% | 无指导采样，效率最低 |
| Bayesian Optimization | -0.082% | 比随机好，但受限于高维搜索空间和少量样本 |
| **Predictor-based RL (本文)** | **-0.093%** | 显著优于前两者，收敛快（2000 步，单 GPU 几分钟） |

**关键结论**：
- RL sampler 能在 2000 步内收敛，且收敛后的预测 NE 优于训练数据中的最佳模型，说明 RL 具备探索能力而非简单记忆；
- 通过调整 reward 中的 `α`（FLOPs 权重），可高效探索不同成本-精度帕累托前沿的模型；
- 与随机采样相比，RL 在同样的样本数下找到更优模型。

### 2.3 消融实验说明了什么最关键

论文做了多个消融，按重要性排序：

1. **Pairwise Ranking Loss vs MSE**：在预测模型排序相关性上，Pairwise Loss 的 Kendall Tau 显著高于 MSE（图 4）。这说明对于 AutoML 搜索来说，**准确的排序远比准确的绝对值更重要**。
2. **集成建模（Ensemble）**：单模型 Predictor 在验证集上泛化差（Kendall 0.38），10 模型集成后提升到 0.87（图 5）。在样本极少（~160 个）的场景下，集成是提升泛化的利器。
3. **训练曲线外推（Curve Trends）**：加入曲线趋势后，低精度评估（11B 样本）与长期评估（25B 样本）的 Kendall Tau 从 0.77 提升到 0.87（图 3a）。这防止了搜索器只选"短跑选手"而错过"长跑选手"。
4. **低精度评估**：在不损失排序质量的前提下，将训练数据量减少约 50%，直接砍掉一半评估成本。

### 2.4 局限性识别

- **短论文篇幅限制**：对 Predictor 的具体网络结构（共享层 L=2，输出头分离）和 RL 策略网络描述较简略；
- **搜索空间仍依赖人工设计**：DHEN 的 5 类交互模块和堆叠范式是人类专家预设的，AutoML 做的是"模块选择和参数调优"，而非从零发明新算子；
- **数据非平稳性**：生产数据分布持续变化，Predictor 训练用的历史模型数据会过期，需要持续更新；
- **跨任务泛化未验证**：实验只在 CTR 和 CVR 上验证，对其他排序目标（如时长、互动率）的迁移能力未讨论。

---

## Step 3 — 工程落地可行性评估

### 3.1 数据需求评估

| 维度 | 评估 |
|------|------|
| **训练数据量** | 不需要额外标注数据；Predictor 的训练数据来自历史模型评估记录（arch 编码 + NE gain + FLOPs）。论文中用了 160~190 条历史记录就能 warm start。 |
| **特征依赖** | 无特殊特征依赖，AutoML 操作的是模型架构 JSON 配置，与特征工程解耦。 |
| **数据标注成本** | 极低。每次模型评估的"标签"就是 NE loss 和 FLOPs，来自正常训练日志。 |

### 3.2 计算开销评估

| 阶段 | 评估 |
|------|------|
| **训练阶段（单次模型评估）** | 大容量 DHEN 模型训练很贵（使用 ZionEX 集群，128 张 A100）。但论文通过低精度评估砍掉 50% 数据量；且 Predictor 训练极轻量（4 层 MLP，单 GPU 几分钟）。 |
| **搜索阶段** | RL sampler 2000 步收敛，单 GPU 仅需数分钟；整个搜索流程平均只采样约 100 个模型。 |
| **推理阶段** |  discovered 模型与人工模型同属 DHEN 家族，推理延迟和 QPS 在同一量级。论文中甚至有 discovered 模型在 NE 持平情况下 QPS 提升 18.96%。 |
| **内存占用** | Predictor 本身可忽略；discovered 模型容量由搜索空间的复杂度决定（7x~56x 基准），属于可控范围。 |

### 3.3 实现复杂度评估

| 维度 | 评估 |
|------|------|
| **依赖框架** | PyTorch（Pairwise Ranking Loss 直接调 `torch.nn.MarginRankingLoss`），RL 部分为标准 REINFORCE，无自定义 CUDA 算子。 |
| **是否有代码** | 无官方开源，但 Predictor 结构（MultiTaskMLP）和 RL 算法均为标准实现，第三方复现难度中等。 |
| **数学难度** | 低。核心就是 MLP + Margin Ranking Loss + REINFORCE，无复杂贝叶斯推断或可微分架构搜索（DARTS）的梯度回传问题。 |
| **系统改造** | **极低**。这是本文最大工程优势：Predictor 输入是模型 JSON 配置的 one-hot/float 编码，输出是 NE gain 和 FLOPs；discovered 模型仍是 DHEN 结构，可直接接入现有训练（ZionEX）和 serving 栈。 |

### 3.4 综合落地评分

**⭐⭐⭐⭐⭐（强烈推荐）**

理由：
- 改动极小：不需要改特征系统、不需要改 serving 架构，只需要在现有模型训练流水线外面包一层 AutoML 调度器；
- 收益明确：在强人类基线上还能榨出 -0.09% ~ -0.36% NE 或 +25% QPS；
- 成本可控：平均 100 次评估，且低精度评估省 50% 算力；
- 已被生产验证：Instagram CTR 模型已上线 A/B 测试并统计显著。

---

## Step 4 — 复现步骤拆解

### Phase 1：快速验证（1-3 天）

1. **搭建 Predictor 原型**：
   - 输入：模型架构的 one-hot / float 编码向量（如层数、是否启用 attention、kernel size、学习率等）；
   - 网络：4 层 MLP，`[input_dim, 50, 50, 2]`，前两层共享，后两层分头预测 NE gain 和 FLOPs；
   - Loss：`torch.nn.MarginRankingLoss(margin=0.001)`；
   - 集成：训练 10 个独立模型做平均。

2. **准备历史数据**：
   - 收集过去几个月内所有训练过的模型记录，提取 `(arch_config, final_NE, FLOPs)`；
   - 对 NE 做 baseline 归一化，得到 NE gain（负值表示提升）。

3. **验证 Predictor 排序能力**：
   - 按 160:30 划分训练/测试；
   - 目标：测试集 Kendall Tau > 0.8，Pearson > 0.85。

### Phase 2：离线复现（3-7 天）

1. **接入低精度评估**：
   - 选一小批 pilot 模型做全量训练至收敛，记录学习曲线；
   - 做相关性分析，找到"排序质量不下降"的最小数据量（如全量的 50%）；
   - 后续所有评估都用这个低精度设置。

2. **加入训练曲线外推**：
   - 低精度评估末期取若干点拟合线性回归 `NE(l) = m·l + c`；
   - 用公式 `NE_long-term = NE_short-term + m·Δx` 作为 Predictor 的 accuracy label。

3. **实现 RL Sampler**：
   - 使用 REINFORCE，策略网络输出每个超参数的采样分布；
   - Reward：`-(1-α)·NE_ensemble - α·FLOPs_ensemble`；
   - 每次 RL 搜索用不同随机种子跑多轮，取 top-k 模型。

4. **定义搜索空间（参考 DHEN）**：
   - 层数（如 1~8）；
   - 各层启用的交互模块（AdvancedDLRM、Self-Attention、Linear、DCN、Convolution、None）；
   - 模块内部参数（kernel size、weight matrix size 等）；
   - 训练超参数（learning rate、init scale、layer norm 等）。

5. **离线对比实验**：
   - 与 Random Search、Bayesian Optimization（如 Optuna）在相同预算下对比 NE gain；
   - 验证论文结论：Predictor+RL > BO > Random。

### Phase 3：工程化（1-2 周）

1. **训练流水线集成**：
   - 开发一个 AutoML Controller，负责：采样 arch → 触发低精度训练 → 回收 NE/FLOPs → 更新 Predictor → RL 采样下一轮；
   - Controller 与现有训练平台（如 ZionEX、内部训练系统）通过 API 对接。

2. **特征对齐与防 Training-Serving Skew**：
   - discovered 模型仍是标准 DHEN 结构，特征处理层与 DLRM 一致，skew 风险低；
   - 重点验证：低精度评估的排序结论在全量训练上是否保持。

3. **Serving 侧验证**：
   - 对 top 候选模型做推理性能测试（QPS、latency、内存）；
   - 确认 FLOPs 与真实 QPS 的相关性（论文中 Pearson -0.97）。

4. **线上 AB 测试**：
   - 选取 NE 和 QPS 综合最优的模型上线；
   - 建议观察期至少 12 天（参考论文 Instagram 实验）。

### 关键复现检查点

- [ ] Predictor 在测试集上的 Kendall Tau 是否达到 0.8+？
- [ ] 低精度评估（如 50% 数据）与全量评估的排序相关性是否 Kendall 0.75+？
- [ ] 加入曲线外推后，排序相关性是否有明显提升？
- [ ] RL sampler 收敛后找到的模型，是否优于训练数据中的最佳模型？
- [ ] 在相同采样预算下，Predictor+RL 是否显著优于 Random / BO？

---

## Step 5 — 美团业务场景适配建议

### 5.1 适配分析框架

#### 外卖/到店推荐场景

- **特殊挑战**：用户意图实时变化（早午晚高峰需求差异大），LBS 特征强，候选集动态变化；
- **适配关键点**：
  - 本文的 AutoML 框架与业务目标无关，只优化模型架构和超参数，因此可直接套用于外卖 CTR/CVR 预估模型；
  - **低精度评估在美团的可行性**：美团推荐模型训练数据量级大，完全可以做"用 50% 近期数据做快速评估"的实验，验证排序相关性；
  - **需注意**：外卖场景存在明显的时段非平稳性，历史模型数据的分布偏移可能比 Meta 社交 feed 更严重，Predictor 需要定期 retrain 或加入时间特征。

#### 搜索排序场景

- **特殊挑战**：Query 多样性高，商家/商品百万级，相关性信号复杂；
- **适配关键点**：
  - 搜索排序模型同样采用深度交互网络（如 DIN、Transformer 交叉注意力），与 DHEN 的"交互模块堆叠"范式类似；
  - 搜索场景对 latency 更敏感，本文中 FLOPs-QPS 强相关（Pearson -0.97）的结论可直接用于搜索模型的效率-精度权衡；
  - 可将 Query 侧和 Doc 侧的交互层参数加入搜索空间。

#### 广告投放场景

- **特殊挑战**：CVR 极度稀疏（点击多转化少），多任务学习（CTR+CVR）联合建模；
- **适配关键点**：
  - 本文已在 Ad CVR 上验证，说明对稀疏标签场景有效；
  - 美团的广告模型也常用 MMoE / PLE 等多任务结构，可将任务门控网络的结构参数加入 AutoML 搜索空间；
  - **改进建议**：Predictor 的输出头可以从"NE gain + FLOPs"扩展为"CTR NE gain + CVR NE gain + FLOPs"，用多目标 RL reward 做帕累托搜索。

#### 风控场景

- **特殊挑战**：正负样本极不均衡（欺诈率 < 1%），实时决策要求毫秒级；
- **适配关键点**：
  - 风控模型通常比推荐模型小得多，AutoML 的 ROI 可能不如推荐场景明显；
  - 若风控模型也采用深度交互结构，仍可尝试，但搜索空间应更侧重轻量级架构（减少 FLOPs 权重 α）。

### 5.2 针对美团场景的改进建议

1. **特征层改进**：
   - 在模型配置编码中，加入美团特有的特征组信息（如 LBS 距离桶、时段 ID、品类层级、用户生命周期阶段）；
   - 这些特征本身不参与 AutoML 搜索，但可用于分析 discovered 模型在不同特征子集上的鲁棒性。

2. **训练策略改进**：
   - **非平稳数据处理**：Meta 的 Predictor 用历史数据 warm start，但美团外卖/闪购的数据分布随时间变化快，建议加入"模型评估时间戳"作为 Predictor 的额外输入，或采用滑动窗口只保留近期数据；
   - **多目标扩展**：对于广告和推荐，NE gain 可替换为业务指标（如 GMV、ROI、用户体验分），Predictor 的多任务头可扩展为 3+ 个目标。

3. **系统集成建议**：
   - **与现有特征平台集成**：AutoML Controller 只需调用现有训练平台 API，无需侵入特征生产链路；
   - **与模型仓库集成**：所有评估过的模型配置和指标自动入库，形成持续积累的"模型知识库"，供后续 warm start 使用；
   - **成本监控**：每次 AutoML 搜索任务自动计算总 GPU 小时数，并与预期收益（NE 提升 * 业务流量价值）做 ROI 核算，确保 AutoML 本身不亏本。

4. **冷启动加速**：
   - 美团已有大量历史训练记录，可直接用于 warm start Predictor；
   - 建议先用 1~2 周的历史数据做 Predictor 冷启动，验证排序相关性达标后再启动 RL 搜索。

---

## 一句话总结

> **Meta 这篇工作证明：在工业级大容量排序模型上，AutoML 不需要天量算力，只要 Predictor 做得准（低精度评估 + 曲线外推 + Pairwise Loss + 集成），RL 采样器做得巧，平均采样 100 个模型就能超越人类专家调优的强基线——而且已经真金白银地在线上 A/B 测试里验证过了。**

---

## 参考信息

- arXiv: https://arxiv.org/abs/2311.07870
- 相关论文：Wen et al., 2020 "Neural Predictor for Neural Architecture Search" (ECCV)
- Meta 自研架构：DHEN (Zhang et al., 2022), DLRM (Naumov et al., 2019)
- 训练系统：ZionEX (Mudigere et al., 2022)
