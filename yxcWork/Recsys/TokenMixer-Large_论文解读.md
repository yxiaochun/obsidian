# TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders — 论文解读报告

> **论文链接**：https://arxiv.org/abs/2602.06563  
> **创建时间**：2026-08-24  
> **来源**：arXiv 预印本（2026 年 2 月）
---

## 一、论文速读摘要

### 论文元信息

- **标题**：TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders
- **作者**：Yuchen Jiang, Jie Zhu, Xintian Han, Hui Lu, Kunmin Bai, Mingyu Yang 等 21 人
- **机构**：ByteDance AML & ByteDance
- **发表状态**：arXiv:2602.06563（2026 年 2 月预印本，v2 修订于 2026-02-10）
- **前作关系**：本文是 RankMixer/TokenMixer 的系统性进化版本，解决了前者在极端规模下的深层训练瓶颈

### 核心贡献（3 句话版本）

1. 提出 TokenMixer-Large，通过 Mixing & Reverting 操作修复 RankMixer 的残差设计缺陷，引入 Inter-layer Residuals 和 Auxiliary Loss 实现深层模型稳定训练，将排序模型从 1B 参数扩展至 15B（离线）/ 7B（在线流量）。
2. 提出"纯模型架构"哲学，移除 LHUC、DCNv2 等碎片化遗留算子，将广告骨干网 MFU 从 RankMixer 的 45% 进一步提升至 60%；升级为 Sparse Per-token MoE，实现"Sparse Train, Sparse Infer"统一范式。
3. 在抖音多场景全流量验证：电商订单 +1.66%、人均预览支付 GMV +2.98%；广告 ADSS +2.0%；直播收入 +1.4%，服务数亿用户。

---

## 二、背景与核心问题

### 2.1 前作 RankMixer 的成就与遗留问题

RankMixer（2025 年 7 月）通过 Multi-head Token Mixing 替代 Self-Attention，成功解决了工业排序模型"参数扩展 100 倍而延迟不变"的问题，MFU 从 4.5% 提升至 45%，并在抖音 Feed 推荐场景全流量上线 1B 参数模型。

但论文尖锐地指出，当试图进一步 scale-up 时，RankMixer/TokenMixer 架构暴露了五个致命瓶颈：

| 瓶颈 | 具体表现 | 对规模化扩展的影响 |
|------|---------|------------------|
| **Sub-optimal Residual Design** | Token Mixing 将原始 token 转换为新 token，但维度不匹配；直接相加导致语义错位 | 深层堆叠时残差路径失效，性能不增反降 |
| **Impure Model Architecture** | 历史迭代遗留 LHUC、DCNv2 等碎片化算子，属于 memory-bound 低计算强度操作 | 整体 MFU 被拖垮，即使 backbone 高效，碎片算子成为瓶颈 |
| **Insufficient Gradient Updates** | RankMixer 工业部署仅 2 层，深层配置训练不稳定，出现梯度消失 | 无法享受深层模型带来的 representational capacity |
| **Inadequate MoE Sparsification** | 采用"Dense Train, Sparse Infer"范式，训练阶段未节省成本；ReLU-MoE 激活动态不可预测 | 训练成本未降低，推理需要截断或 fallback 策略 |
| **Limited Scaling Exploration** | 仅扩展到 1B 参数 | 未触及推荐模型 scaling law 的天花板 |

### 2.2 问题定义一句话

> **工业排序模型要真正进入"大模型时代"，必须同时解决"深层训练稳定性"、"架构纯度"、"端到端稀疏化"三大工程难题，而不仅仅是替换一个特征交互算子。**

---

## 三、方法解析：TokenMixer-Large 的四大核心创新

TokenMixer-Large 的整体架构由三个模块组成：
1. **Semantic Group-wise Tokenization** — 将稀疏特征映射为语义对齐的 dense token
2. **TokenMixer-Large Block** — 核心特征交互层，包含 Mixing & Reverting、Per-token SwiGLU、Interval Residuals
3. **Sparse Per-token MoE (S-P MoE)** — 升级版的稀疏专家网络

### 3.1 创新一：Mixing & Reverting — 修复残差路径的语义错位

#### 问题根源

RankMixer 的 Token Mixing 操作将原始 token（维度 d）通过 transpose 和混合后，生成新的 mixed token（维度 d'）。在 Add & Norm 时，直接将原始 token 和混合后的 token 相加：

```python
# RankMixer 的残差路径（问题版）
mixed_tokens = token_mixing(original_tokens)  # [T, d] -> [T', d']  维度可能不同
output = original_tokens + mixed_tokens       # 直接相加：语义错位！
```

**核心问题**：
- Token Mixing 将 256 个特征域重组为新的 token 集合，token 的语义可能已经改变（比如从"用户侧特征域"变成了"交叉特征域"）
- 如果 T' ≠ T 或 token 语义已经改变，直接相加相当于让"苹果"和"橘子"做残差连接，导致信息污染

#### TokenMixer-Large 的解决方案

论文引入 **Mixing & Reverting** 的闭环范式：

```python
# TokenMixer-Large 的残差路径（修复版）
mixed_tokens = token_mixing(original_tokens)   # Step 1: Mixing  [T, d] -> [T', d']
reverted_tokens = token_reverting(mixed_tokens) # Step 2: Reverting [T', d'] -> [T, d]
output = original_tokens + reverted_tokens      # Step 3: 现在维度匹配，语义对齐！
```

**Reverting 操作的本质**：
- Mixing 阶段通过 head splitting + transpose + cross-token pooling 实现信息交互
- Reverting 阶段是 Mixing 的**逆操作**（或近似逆操作），将交互后的信息重新映射回原始 token 空间
- 这确保了残差连接的两端在**语义空间**和**维度空间**上都是对齐的

> **类比理解**：Mixing 像把多种食材放入搅拌机混合，Reverting 像把混合物倒回各自的容器——这样"加法"操作才有意义。

### 3.2 创新二：Interval Residuals + Auxiliary Loss + Down-matrix Small Init — 深层模型训练稳定三件套

#### 问题根源

RankMixer 在工业部署中仅使用 2 层。当尝试堆叠到 8 层、16 层甚至更深时：
- 梯度在反向传播时经过多层 Norm + Mixing + SwiGLU 后迅速衰减
- 深层网络失去有效的梯度信号，训练 loss 无法收敛

#### 三件套解决方案

| 技术 | 作用 | 具体机制 |
|------|------|---------|
| **Interval Residuals** | 缩短梯度传播路径 | 在每隔 N 层（而非每 1 层）引入残差连接，减少梯度需要经过的变换层数 |
| **Auxiliary Loss** | 提供中间监督信号 | 在深层中间层引入辅助损失函数，为远离输出层的参数提供直接梯度 |
| **Down-matrix Small Initialization** | 稳定初始训练阶段 | 将 down-projection 矩阵（将高维映射回低维的矩阵）初始化得更小，防止初始阶段信息爆炸 |

**三者的协同效应**：
- Interval Residuals 解决"梯度传播距离过长"问题
- Auxiliary Loss 解决"深层参数信号不足"问题
- Down-matrix Small Init 解决"训练初期不稳定"问题

论文通过消融实验验证了这三件套的必要性——移除任何一个，深层模型的收敛都会出现明显退化。

### 3.3 创新三："纯模型架构"哲学 — 从 MFU 45% 到 60%

#### 问题根源

RankMixer 虽然将 backbone 的 MFU 提升到 45%，但许多工业模型仍然保留了历史迭代中的"碎片化算子"：
- **LHUC**（Learning Hidden Unit Contributions）：参数化特征门控，低计算强度
- **DCNv2**（Deep & Cross Network v2）：显式高阶特征交叉，memory-bound 操作
- 其他遗留的手工特征交叉模块

这些算子虽然各自带来 0.1%~0.3% 的离线收益，但：
- 计算密度低，GPU 计算单元利用率差
- 引入了额外的 kernel launch 和数据搬运开销
- 与 TokenMixer backbone 的算子融合不兼容，导致整体 MFU 被拉低

#### 论文的激进洞察

> **"As models scale up, the benefits provided by low-level fragmented operators can be subsumed by stacking multiple TokenMixer Blocks."**

翻译：当模型规模足够大时，低层碎片化算子带来的微弱收益，完全可以被堆叠更多 TokenMixer-Large Block 所覆盖——而且这些纯架构的收益是**计算高效的**。

#### 实践结果

- 移除所有 LHUC、DCNv2 等碎片化算子
- 仅保留 TokenMixer-Large Block 的纯堆叠架构
- 广告骨干网络的 **MFU 从 45% 提升至 60%**
- 这意味着同等 GPU 集群下，吞吐量又提升了 33%

**哲学意义**：这是工业模型架构从"拼积木式堆叠"向"统一设计范式"的范式转移。类似 NLP 领域从"ELMo + BiLSTM + CRF"到纯 Transformer 的演进。

### 3.4 创新四：Sparse Per-token MoE — 从"Dense Train"到"Sparse Train, Sparse Infer"

#### RankMixer MoE 的问题

RankMixer 使用的 ReLU-MoE 存在两个关键缺陷：

1. **"Dense Train, Sparse Infer"**：训练阶段每个专家都参与计算（只是通过门控加权），只有推理阶段稀疏激活。这意味着训练成本没有降低，1B 参数模型训练时就需要 1B 的 FLOPs。

2. **ReLU-MoE 的激活动态不可预测**：ReLU 的激活值是连续的，导致每个 batch 中实际激活的专家数量不确定。推理时如果超出预分配资源，需要截断或 fallback——这在工业级 QPS 场景下是不可接受的。

#### TokenMixer-Large 的升级方案

**Sparse Per-token MoE (S-P MoE)** 的核心设计：

| 维度 | RankMixer (ReLU-MoE) | TokenMixer-Large (S-P MoE) |
|------|----------------------|---------------------------|
| **训练范式** | Dense Train, Sparse Infer | **Sparse Train, Sparse Infer** |
| **专家激活方式** | ReLU 连续门控，数量不确定 | **Top-k 硬路由**，每个 token 只激活固定数量专家 |
| **训练成本** | 与参数量成正比（高） | 仅与激活专家数成正比（低） |
| **推理可预测性** | 差，需要截断/fallback | **好**，每个 token 固定激活 k 个专家 |
| **工程优化** | 常规 | **FP8 量化 + Token Parallel** |

**"Sparse Train, Sparse Infer"的工程价值**：
- 训练成本与模型总容量解耦：1B 参数模型训练时只计算 100M 激活参数
- 推理延迟可精确预估：每个 token 固定走 2 个专家，无动态溢出风险
- 配合 FP8 量化和 Token Parallel，进一步压榨训练/推理效率

> **类比理解**：Dense Train 像让员工每天打卡所有部门；Sparse Train 像员工只去自己被分配的小组——会议效率更高，资源占用更少。

---

## 四、实验与效果

### 4.1 离线 Scaling Law 实验

| 场景 | 最大参数规模 | 训练范式 |
|------|------------|---------|
| 抖音广告 | 15B 参数 | 离线实验 |
| 抖音电商 | 7B 参数 | 在线流量实验 |
| 抖音电商 | 4B 参数 | 在线流量实验 |

### 4.2 在线 A/B 测试 — 多场景全流量验证

TokenMixer-Large 已部署至抖音多个核心业务场景，服务数亿用户：

| 业务场景 | 核心指标 | 提升效果 | 业务意义 |
|---------|---------|---------|---------|
| **电商** | 订单量 | **+1.66%** | 直接交易转化提升 |
| **电商** | 人均预览支付 GMV | **+2.98%** | 用户支付意愿和客单价提升 |
| **广告** | ADSS（广告深度会话质量） | **+2.0%** | 广告体验和用户价值提升 |
| **直播** | 收入 | **+1.4%** | 直播打赏/带货收入提升 |

### 4.3 关键工程指标对比

| 指标 | RankMixer | TokenMixer-Large | 变化 |
|------|-----------|-----------------|------|
| 最大参数规模 | 1B | 15B（离线）/ 7B（在线） | **7~15 倍** |
| 骨干网 MFU | 45% | **60%** | **+33%** |
| 训练范式 | Dense Train | Sparse Train | 成本大幅下降 |
| 模型纯度 | 保留碎片化算子 | **纯架构** | 算子融合更彻底 |
| 深层训练 | 最多 2 层 | 支持深层堆叠 | 表征能力大幅提升 |

---

## 五、与 RankMixer 的对比演进关系

TokenMixer-Large 不是一篇独立的论文，而是对 RankMixer 的**系统性诊断和修复**。理解两者的演进关系，对把握工业排序模型的发展方向非常关键。

```
                    RankMixer (2025.07)                TokenMixer-Large (2026.02)
                    ──────────────────                ──────────────────────────

特征交互层:         Multi-head Token Mixing    ───→    Mixing & Reverting
                       (无残差修复)                        (语义对齐残差)

特征建模层:         Per-token FFN            ───→    Per-token SwiGLU + S-P MoE
                       (ReLU-MoE, Dense Train)          (Sparse Train/Infer, Top-k 路由)

深层训练:           最多 2 层                 ───→    Interval Residuals + Aux Loss
                                                          + Down-matrix Small Init

架构纯度:           保留 LHUC/DCNv2 碎片算子   ───→    纯架构，移除所有碎片算子

最大规模:           1B 参数                  ───→    15B 参数（离线）/ 7B（在线）

MFU:                45%                      ───→    60%

在线效果:           Feed 推荐: 日活 +0.3%      ───→    电商订单 +1.66%
                       时长 +1.08%                     广告 ADSS +2.0%
                                                         直播收入 +1.4%
```

### 核心演进逻辑

**RankMixer 解决的问题**：特征交互算子从 CPU-era memory-bound 设计转向 GPU-era compute-bound 设计，实现**参数和延迟的解耦**。

**TokenMixer-Large 解决的问题**：
1. 残差路径的语义对齐（从"能工作"到"能深层工作"）
2. 深层训练的稳定性（从"浅层有效"到"深层更有效"）
3. 架构纯度（从"backbone 高效但碎片算子拖后腿"到"全链路高效"）
4. 端到端稀疏化（从"训练不省钱"到"训练推理双省钱"）
5. 规模天花板（从"1B 到头"到"15B 仍有效"）

---

## 六、工程落地可行性评估

### 6.1 综合评分：⭐⭐⭐⭐⭐（5/5）

**评分理由**：
- 工业界已多场景全流量验证，非纯学术探索
- 15B 参数离线实验和 7B 在线部署证明 scaling 天花板远超前作
- "纯架构"哲学大幅降低工程复杂度，MFU 60% 意味着 GPU 利用率接近极限
- Sparse Train 范式直接降低训练成本，ROI 极高

### 6.2 数据需求评估

- **训练数据量**：论文未明确提及，但参考前作 RankMixer 的万亿级数据，TokenMixer-Large 的 15B 参数版本需要的数据量只会更大。
- **特征依赖**：需要完善的 Semantic Group-wise Tokenization，对特征工程平台有一定要求（需要将特征按语义分组并统一维度）。
- **数据标注**：排序任务天然标签（点击、转化、支付），无需额外标注。

### 6.3 计算开销评估

| 阶段 | 评估 |
|------|------|
| **训练阶段** | Sparse Train 范式下，训练成本与激活专家数成正比，而非总参数量。15B 模型训练成本可能接近传统 1~2B 密集模型。FP8 进一步压缩训练显存和带宽需求。 |
| **推理阶段** | Top-k 硬路由确保每个 token 固定激活 k 个专家，延迟可精确预估。Token Parallel 优化并行效率。 |
| **内存占用** | 总参数 15B 需存储全部专家，但推理时只需加载激活部分。需评估显存容量是否足够。 |

### 6.4 实现复杂度评估

| 维度 | 评估 |
|------|------|
| 依赖框架 | PyTorch 标准模块 + MoE 路由逻辑；FP8 需要较新硬件（H100）和框架支持 |
| 是否有代码 | 目前未标注开源，但架构逻辑清晰，可基于 RankMixer 代码升级 |
| 数学难度 | 核心创新（Mixing/Reverting、Interval Residual、Top-k MoE）均为标准 DL 知识 |
| 系统改造 | 需要 Semantic Group-wise Tokenizer 改造输入层；移除碎片算子可能影响已有特征工程链路 |

### 6.5 风险点

1. **硬件门槛**：MFU 60% 和 FP8 需要较新 GPU（H100 级别），老旧硬件可能无法发挥全部优势。
2. **MoE 负载均衡**：Top-k 硬路由虽然可预测，但需确保负载均衡机制有效，避免专家倾斜。
3. **深层训练稳定性**：Interval Residuals 和 Aux Loss 的超参数（间隔层数、辅助损失权重）需要针对具体场景调优。
4. **"纯架构"迁移成本**：移除 LHUC/DCNv2 等碎片算子可能需要重新验证特征工程链路，存在短期效果波动的风险。

---

## 七、复现步骤拆解

### Phase 1：快速验证（1-3 天）

1. **基于 RankMixer 代码升级**：若已有 RankMixer 复现，在其基础上增加 Mixing & Reverting 模块。
2. **小规模验证**：在 Criteo/Avazu 上验证 TokenMixer-Large Block 的收敛性和效果。
3. **Ablation 验证**：验证移除 DCNv2/LHUC 后，增加 TokenMixer-Large Block 层数是否能覆盖收益。

### Phase 2：离线复现（1-2 周）

1. **Semantic Group-wise Tokenization**：将特征按语义分组（用户侧、商品侧、上下文、序列特征等），每组内特征对齐到相同维度。
2. **实现核心模块**：
   - Mixing & Reverting（transpose + pooling + inverse transpose）
   - Interval Residuals（每隔 N 层添加残差）
   - Per-token SwiGLU（每个 token 独立的 SwiGLU 层）
   - Sparse Per-token MoE（Top-k 门控 + 专家路由）
3. **训练稳定性调优**：
   - Down-matrix 小初始化（scale 设为 0.01~0.1）
   - Auxiliary Loss 权重调参（0.1~0.5 范围实验）
   - 层数逐步增加（2 → 4 → 8 → 16），观察收敛曲线
4. **纯架构迁移**：
   - 在实验分支中移除 LHUC/DCNv2
   - 对比"纯架构+深层" vs "碎片算子+浅层"的效果

### Phase 3：工程化（2-4 周）

1. **Sparse MoE 训练框架**：
   - 实现 Top-k 路由和专家选择
   - 添加负载均衡辅助损失（auxiliary load balancing loss）
   - 验证 Sparse Train 下的收敛速度
2. **FP8 量化**：
   - 在 H100 上启用 FP8 训练和推理
   - 对比 FP16/FP8 的精度和速度 trade-off
3. **Token Parallel**：
   - 优化 token 级别的并行计算，减少通信开销
4. **压测与 AB 测试**：
   - 验证 P99 latency 满足 SLA
   - 全流量 AB 测试验证核心业务指标

---

## 八、美团业务场景适配建议

### 8.1 外卖 / 到店推荐

**适配点**：
- **Semantic Group-wise Tokenization**：将 LBS 特征（用户位置、商家位置、配送距离）组织为独立的语义组，利用 Per-token SwiGLU 进行独立建模。
- **纯架构迁移**：外卖排序模型中常保留 DCN 或 FM 做显式特征交叉，可尝试移除后用深层 TokenMixer-Large Block 替代。
- **Sparse MoE 专家特化**：可为"早餐时段""午餐时段""晚餐时段"配置不同专家，通过路由机制实现时段感知的参数激活。

### 8.2 搜索排序

**适配点**：
- **Query-Document 分组**：将 Query 侧特征和 Document 侧特征分为两个语义组，Mixing 阶段实现跨组交互，Reverting 阶段保持各自语义空间。
- **深层表征**：搜索场景需要更复杂的 Query-Document 匹配逻辑，深层 TokenMixer-Large（8~16 层）的表征能力可能带来比浅层模型更显著的提升。

### 8.3 广告投放

**适配点**：
- **Sparse MoE 直接复用**：广告场景对训练成本和推理延迟最敏感，Sparse Train/Infer 范式可直接降低训练集群成本。
- **多目标专家特化**：CTR 预测和 CVR 预测可分别路由到不同专家组，实现任务感知的参数激活。
- **MFU 60% 的收益**：广告模型通常需要高频重训，GPU 利用率提升 33% 意味着同等训练成本下可以支持更大模型或更频繁更新。

### 8.4 直播 / 内容推荐

**适配点**：
- **序列特征处理**：论文提到 Semantic Group-wise Tokenizer 支持 DIN/LONGER 等序列特征提取后的 embedding，可直接接入直播场景的用户行为序列。
- **收入导向优化**：论文直播场景收入 +1.4% 的验证，为直播打赏和带货场景提供了直接的上线信心。

### 8.5 改进建议

1. **特征组设计**：美团的特征体系与抖音差异较大，Semantic Group-wise Tokenizer 的分组策略需要针对美团特征体系重新设计（如"运力特征组""时效特征组"等外卖特有分组）。
2. **冷启动适配**：深层模型对数据量要求更高，冷启动场景可能需要额外的浅层辅助网络或预训练策略。
3. **MoE 专家数调参**：15B 参数可能不适合所有场景，建议从 1B → 4B → 7B 逐步验证 scaling curve，找到业务场景的收益拐点。

---

## 九、注意事项与风险提示

1. **预印本状态**：本文目前为 arXiv 预印本，未经过严格同行评审。15B 参数离线实验的细节（如训练时间、显存占用、超参数配置）需要更多验证。
2. **硬件依赖**：MFU 60% 和 FP8 量化对硬件（H100 级别）和框架版本有较高要求，A100 或 V100 上可能无法达到论文效果。
3. **深层训练超参敏感**：Interval Residuals 的间隔层数、Auxiliary Loss 的权重、Down-matrix 的初始化 scale 对深层模型收敛至关重要，需要充分的调参实验。
4. **碎片算子移除的短期风险**：LHUC/DCNv2 等算子虽然 MFU 低，但可能在某些特定特征组合上有不可替代的作用。建议灰度验证"纯架构"是否完全覆盖碎片算子的收益。
5. **MoE 专家平衡**：Top-k 硬路由需要确保负载均衡机制有效，否则可能出现专家倾斜（少数专家承载 90% 流量），导致稀疏化失效。

---

> 如需将本文解读保存为 Markdown 文件，或进一步对比 TokenMixer-Large 与 RankMixer 的详细架构差异，请随时告知。
