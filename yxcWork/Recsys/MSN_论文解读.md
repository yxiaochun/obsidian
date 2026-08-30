---
title: A Memory-based Sparse Activation Scaling Framework for Large-scale Industrial Recommendation
source: https://arxiv.org/abs/2602.07526
created: 2026-08-24
---

---
## 一、论文速读摘要

### 论文元信息

- **标题**：MSN: A Memory-based Sparse Activation Scaling Framework for Large-scale Industrial Recommendation
- **作者**：Shikang Wu, Hui Lu, Jinqiu Jin, Zheng Chai, Shiyong Hong, Junjie Zhang, Shanlei Mu, Kaiyuan Ma, Tianyi Liu, Yuchao Zheng, Zhe Wang, Jingjian Lin
- **机构**：ByteDance Search, ByteDance AML
- **发表状态**：arXiv:2602.07526（2026 年 2 月预印本）
- **定位**：与 RankMixer / TokenMixer-Large 同团队，但聚焦**搜索场景**的另一种稀疏扩展范式——基于 Memory Network 而非 MoE

### 核心贡献（3 句话版本）

1. 提出 MSN（Memory Scaling Network），用**基于检索的稀疏参数访问**替代 MoE 的稀疏计算，通过 Product-Key Memory（PKM）将记忆检索复杂度从线性降至次线性，实现细粒度个性化且内存访问成本远低于 SMoE。
2. 设计 Memory Gating Mechanism 将检索到的个性化记忆有效融入下游特征交互模块，并配套 Norm + Warm-up + Over-parameterization 技术防止记忆检索坍塌和负载不均。
3. MSN 作为**即插即用的模型无关扩展模块**，已部署至**抖音搜索排序系统**，在已有 SMoE 基线上实现离线评估和在线 A/B 测试的显著提升。

---

## 二、问题定义：SMoE 的天花板

### 2.1 工业排序模型规模化扩展的两难

推荐模型扩展参数的常见路径：
- **扩大稀疏 Embedding 表**：参数量巨大但计算量小，主要受限于存储而非计算
- **扩大特征交互模块**：如 Wukong、DHEN、TokenMixer-Large 等，但计算开销同步增长

**核心矛盾**：参数多了 → 计算多了 → 延迟涨了 → 部署不了。

### 2.2 SMoE 为什么还不够好？

论文尖锐地指出了 SMoE（Sparse Mixture-of-Experts）在工业推理中的**两大致命缺陷**：

#### 缺陷一：内存访问成本被忽视

标准 SMoE 层的每个专家是一个完整的 FFN，参数规模 O(d²)：

```
输入: [batch, d]
门控: 选择 top-k 专家
激活: 对选中的 k 个专家，加载其 [d, 4d] 和 [4d, d] 权重矩阵
计算: 每个专家做两次矩阵乘法
```

**关键瓶颈**：
- 虽然只激活 k 个专家，但每个专家的权重需要从显存/内存中**读取**（memory access）
- 每个专家权重规模 O(d²)，假设 d=2048，一个专家的权重就是 ~16MB（FP16）
- 激活 2 个专家就要读取 ~32MB 权重
- 在带宽受限的推理场景下，**内存访问时间可能远超计算时间**

> 论文原文："This results in a memory access cost of O(d²), which becomes a dominant performance bottleneck under bandwidth-limited inference settings."

#### 缺陷二：个性化能力受限

SMoE 的个性化来源于"不同用户激活不同专家"，但这个机制有两个限制：

1. **专家数量受限**：每个专家 O(d²) 的参数规模限制了总专家数（通常 32~128 个）
2. **激活数量有限**：每次只激活 top-k 个专家（通常 k=2），用户只能从有限的专家池中获取个性化参数

论文原文："Consequently, each user can only activate a small subset from the limited expert pool, which results in limited personalization capability and suboptimal recommendation performance."

**通俗理解**：SMoE 像一家只有 64 个科室的医院，每个病人只能挂 2 个科室的号。科室之间共享很多参数（每个科室都是完整 FFN），所以科室数量做不大。而推荐系统有数亿用户，64 个科室的个性化粒度显然不够。

### 2.3 论文的核心洞察

> **"Instead of routing each interaction through a limited set of shared experts, we can dynamically retrieve a value representation from a large parameterized memory for each user."**

**核心思想转变**：
- SMoE：参数分装在少数大专家里，选几个专家来计算
- MSN：参数存放在一个巨大的**记忆表**里，为每个用户**检索**少量相关的记忆条目

类比：
- SMoE = 64 个全科医生，每人都很全面但科室数量有限
- MSN = 一个巨大的医学知识库，为每个病人检索最相关的几条医学知识

---

## 三、MSN 架构详解

### 3.1 整体架构

MSN 是一个**即插即用的扩展模块**，可以接在任何已有 backbone（如 SMoE-based 模型）上：

```
输入特征 → [Backbone (如 SMoE)] → 中间表示 h
                                      ↓
                                Query Projection: q = W_q · h
                                      ↓
                        ┌───────────────────────────────┐
                        │    Product-Key Memory (PKM)   │
                        │  - 从巨大记忆表中检索 top-k   │
                        │  - 返回 value 表示 v          │
                        └───────────────────────────────┘
                                      ↓
                        Memory Gating: h' = h ⊙ gate(v)
                                      ↓
                            输出到下游预测层
```

### 3.2 核心创新一：Product-Key Memory (PKM) — 次线性复杂度检索

#### 传统 Key-Value Memory 的问题

标准 KV Memory 的结构：
- 一个巨大的记忆表，包含 N 个条目，每个条目是一个 (key, value) 对
- 查询向量 q 与所有 N 个 key 计算相似度
- 选择 top-k 个最相似的 key，返回对应的 value

**复杂度**：检索复杂度 O(N)，其中 N 是记忆表大小。如果想扩展到百万甚至十亿级记忆条目，检索开销无法接受。

#### PKM 的解决方案：分解索引空间

Product-Key Memory 的核心思想是将单个 key 分解为**两个子 key 的笛卡尔积**：

```
传统 KV Memory:
  记忆表: [(k_1, v_1), (k_2, v_2), ..., (k_N, v_N)]
  检索: q 与 N 个 key 逐一比较 → O(N)

PKM:
  将 key 分解为两个子空间: key = (key_a, key_b)
  子 key 集合 A: [a_1, a_2, ..., a_√N]
  子 key 集合 B: [b_1, b_2, ..., b_√N]
  完整 key 空间 = A × B (笛卡尔积)，共 N 个组合
  
  检索过程:
    Step 1: q 分解为 (q_a, q_b)
    Step 2: q_a 与 A 中所有子 key 比较，选 top-k_a → O(√N)
    Step 3: q_b 与 B 中所有子 key 比较，选 top-k_b → O(√N)
    Step 4: 从候选集合 (top-k_a × top-k_b) 中精排，选最终 top-k
```

**复杂度变化**：
- 传统：O(N)
- PKM：O(√N) + O(k_a · k_b) ≈ **O(√N)**（次线性！）

**具体例子**：
- 假设 N = 1,000,000（百万级记忆条目）
- 传统检索：100 万次相似度计算
- PKM 检索：√N = 1,000 次 + 1,000 次 = 2,000 次相似度计算
- **检索效率提升 500 倍**

### 3.3 核心创新二：Memory Gating Mechanism — 记忆与特征的有效融合

检索到记忆 value 后，如何将其融入原有特征表示？论文设计了一个**门控机制**：

```python
# 原始特征表示
h = backbone_output  # [batch, d]

# 检索到的记忆 value
v = PKM_retrieve(query=h)  # [batch, k, d_v]

# 记忆门控：将记忆 value 聚合为调制信号
# 方式 1: 加权聚合 + 投影
gate_signal = Aggregate(v)  # [batch, d]

# 方式 2: 与原始特征做元素级门控
h_enhanced = h ⊙ σ(gate_signal)  # σ 为 sigmoid，输出 [batch, d]
# 或者: h_enhanced = h + gate_signal  # 残差融合
```

**设计要点**：
- 检索到的记忆 value 作为**调制信号（modulation signal）**，而非直接替代原始特征
- 通过 sigmoid 门控，模型可以自适应地决定"在多大程度上信任检索到的记忆"
- 这种设计使得 MSN 对 backbone 的侵入性很小，即插即用

### 3.4 核心创新三：Stable and Balanced Memory Optimization — 防止记忆坍塌

大规模记忆网络训练中的经典问题：

#### 问题 1：记忆检索坍塌（Memory Retrieval Collapse）

训练初期，query 和 key 的分布不稳定，模型可能只检索到少数几个"幸运"的记忆条目，大部分记忆条目永远不被访问，导致：
- 记忆利用率极低
- 被频繁访问的记忆条目过拟合
- 模型退化为只有少数有效参数的浅层网络

#### 问题 2：记忆访问不均衡（Load Imbalance）

不同 query 可能集中检索到相同的少数记忆条目，导致：
- 训练时梯度集中在少数条目上，更新不均衡
- 推理时热点记忆条目成为瓶颈

#### MSN 的三重稳定化技术

| 技术 | 作用 | 机制 |
|------|------|------|
| **Layer Normalization** | 稳定 query/key 分布 | 对 query 向量和 key 向量分别做 LN，防止数值爆炸或消失 |
| **Learning Rate Warm-up** | 温和启动记忆学习 | 训练初期使用较小的学习率，让 query 和 key 的分布先稳定下来，再正常更新 |
| **Over-Parameterization on Keys** | 增强 key 空间覆盖 | 对 key embedding 使用过参数化（更大的维度或更多的投影层），增强 key 空间的表达能力，减少"冷门 key 永远不被访问"的问题 |

### 3.5 核心创新四：Sparse-Gather + AirTopK — 工业级效率优化

#### Sparse-Gather Operator

在分布式训练/推理中，PKM 的记忆表通常分布在多个设备上。Sparse-Gather 是一个定制化的通信算子：

- **功能**：只收集（gather）被检索到的 top-k 记忆条目，而非全量同步
- **优势**：通信量从 O(N) 降到 O(k)，在分布式场景下大幅节省带宽

#### AirTopK Operator

Top-k 检索是 PKM 的核心操作，AirTopK 是一个高效的 top-k 实现：

- **功能**：在 GPU 上高效执行 top-k 选择
- **优势**：比标准 PyTorch/TensorFlow 的 top-k 算子更快，尤其适合小 k（如 k=2~8）的场景
- **适用场景**：PKM 中需要对 √N 个子 key 做 top-k 选择，AirTopK 可以加速这个过程

---

## 四、MSN 与 SMoE 的本质对比

### 4.1 从"稀疏计算"到"稀疏参数检索"的范式转移

| 维度 | SMoE | MSN |
|------|------|-----|
| **稀疏化对象** | 稀疏计算（只计算部分专家） | 稀疏参数检索（只读取部分记忆） |
| **参数组织** | 少数大专家（每个 O(d²)） | 大量小记忆条目（每个 O(d) 或更小） |
| **个性化粒度** | 粗（选几个专家） | 细（检索特定记忆条目组合） |
| **内存访问成本** | O(k · d²)（高） | O(k · d)（低） |
| **计算复杂度** | O(k · d²) | O(√N + k · d) |
| **扩展方式** | 增加专家数量（受限于 d） | 增加记忆条目数（PKM 保证次线性检索） |
| **与 backbone 关系** | 通常需要替换 backbone | **即插即用**，可接在任何 backbone 后 |

### 4.2 为什么 MSN 的内存访问成本更低？

假设 d = 2048，k = 2：

**SMoE**：
- 每个专家 = 两个矩阵 [d, 4d] + [4d, d]
- 一个专家参数量 = d × 4d + 4d × d = 8d² = 8 × 2048² ≈ **33.5M 参数**
- 激活 2 个专家需读取 ≈ **67M 参数** ≈ **134MB（FP16）**

**MSN**：
- 每个记忆条目 = 一个 value 向量，维度 d
- 一个记忆条目参数量 = d = **2048 参数**
- 检索 2 个记忆条目需读取 ≈ **4096 参数** ≈ **8KB（FP16）**
- 加上 key 检索的通信开销，总量仍在 KB~MB 级别

**差距**：MSN 的内存访问量比 SMoE 小了 **约 1 万倍**。

> 注意：这是单次前向传播的内存访问对比。SMoE 的实际收益在于专家参数可以被多个样本共享，而 MSN 的记忆条目也是共享的。核心差异在于**每个激活单元的参数规模**：SMoE 的专家是"胖"的（完整 FFN），MSN 的记忆条目是"瘦"的（单个向量）。

---

## 五、实验与效果

### 5.1 部署场景

MSN 已部署至 **Douyin Search Ranking System（抖音搜索排序系统）**，接在已有的 SMoE-based 模型上作为扩展模块。

### 5.2 实验结论（论文摘要提及）

- **离线评估**：MSN 在多个 backbone 上一致提升推荐性能
- **在线 A/B 测试**：在抖音搜索排序场景相比部署的 SOTA 模型有显著提升
- **效率**：在提升效果的同时保持高推理效率

### 5.3 作为 Plug-in 模块的价值

论文特别强调 MSN 是 **model-agnostic（模型无关）** 的：
- 可以接在 SMoE 模型上
- 可以接在 TokenMixer-Large 上
- 可以接在 DCNv2、DeepFM 等传统模型上

这意味着：如果团队已有成熟的排序 backbone，不需要推翻重来，只需在 backbone 输出后接一个 MSN 模块，即可低成本获得模型扩展的收益。

---

## 六、工程落地可行性评估

### 6.1 综合评分：⭐⭐⭐⭐⭐（5/5）

**评分理由**：
- 工业界（抖音搜索）已部署验证，非纯学术探索
- 即插即用设计大幅降低接入成本，无需改造已有 backbone
- 内存访问成本比 SMoE 低数个数量级，对带宽受限的推理场景极其友好
- PKM 的次线性检索复杂度使得记忆表可以扩展到百万甚至十亿级别

### 6.2 数据需求评估

- **训练数据量**：未明确提及，但作为扩展模块，数据量需求与 backbone 相当
- **特征依赖**：不依赖特殊特征，只需 backbone 输出的中间表示作为 query
- **数据标注**：排序任务天然标签，无需额外标注

### 6.3 计算开销评估

| 阶段 | 评估 |
|------|------|
| **训练阶段** | PKM 检索 + Memory Gating 的计算开销很小；Sparse-Gather 降低分布式通信量 |
| **推理阶段** | 内存访问 O(k·d) 远低于 SMoE 的 O(k·d²)；AirTopK 加速检索过程 |
| **存储开销** | 记忆表 N 个条目需要存储 N × d 的参数，但检索时只读取 k 个 |

### 6.4 实现复杂度评估

| 维度 | 评估 |
|------|------|
| 依赖框架 | PyTorch 标准模块 + 自定义 Sparse-Gather / AirTopK（可能需要 CUDA 开发） |
| 是否有代码 | 目前未标注开源 |
| 数学难度 | PKM 的笛卡尔积检索是标准技术，易于理解 |
| 系统改造 | **极小**，只需在 backbone 后接 MSN 模块，无需改动特征工程链路 |

### 6.5 风险点

1. **记忆表规模与检索质量的 trade-off**：记忆表越大（N 越大），个性化能力越强，但 PKM 检索的 √N 开销也会增长。需要找到业务场景下的最优 N。
2. **记忆条目冷启动**：新用户/新 query 可能检索不到高质量记忆，需要设计 fallback 策略。
3. **分布式训练复杂度**：Sparse-Gather 算子需要定制 CUDA kernel，工程实现有一定门槛。
4. **记忆表更新策略**：记忆表是参数化的，需要随模型一起训练更新。在超大规模（数十亿用户）场景下，记忆表的收敛稳定性需要验证。

---

## 七、与 RankMixer / TokenMixer-Large 的对比与互补

### 7.1 三条技术路线的定位

| 维度 | RankMixer / TokenMixer-Large | MSN |
|------|------------------------------|-----|
| **核心问题** | backbone 特征交互层的效率 | backbone 之后的个性化扩展 |
| **技术本质** | 替换/优化特征交互算子 | 在 backbone 后增加记忆检索模块 |
| **稀疏化策略** | Sparse MoE（稀疏计算） | Sparse Memory Retrieval（稀疏参数检索） |
| **与 backbone 关系** | ** backbone 本身** | ** backbone 的扩展插件** |
| **适用场景** | 从头构建新的排序模型 | 已有模型上低成本升级 |
| **部署场景** | 抖音 Feed 推荐、电商、广告、直播 | 抖音搜索排序 |

### 7.2 三者可以如何组合？

这三篇论文实际上构成了一个**完整的工业排序模型技术栈**：

```
┌─────────────────────────────────────────────────────────────┐
│                    工业排序模型技术栈                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  特征层:  Semantic Group-wise Tokenization                   │
│           (TokenMixer-Large 的输入处理)                       │
│                                                             │
│       ↓                                                     │
│  Backbone: TokenMixer-Large Block × L                       │
│            (高 MFU 统一特征交互架构)                          │
│            - Multi-head Token Mixing                         │
│            - Per-token SwiGLU + Sparse MoE                   │
│            - Mixing & Reverting + Interval Residuals         │
│                                                             │
│       ↓                                                     │
│  扩展层:  MSN (Memory Scaling Network)                       │
│            (即插即用的个性化增强)                             │
│            - Product-Key Memory Retrieval                    │
│            - Memory Gating Mechanism                         │
│            - 低内存访问成本，细粒度个性化                      │
│                                                             │
│       ↓                                                     │
│  预测层:  CTR / CVR / 时长 等多目标预测                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**组合价值**：
- TokenMixer-Large 解决了"特征交互如何高效做"
- MSN 解决了"如何在已有 backbone 上进一步做细粒度个性化"
- 两者组合：高效 backbone + 低成本个性化扩展 = 极致性价比

---

## 八、复现步骤拆解

### Phase 1：快速验证（1-2 天）

1. **搭建 PKM 原型**：
   - 实现 Product-Key Memory 的核心逻辑（query 分解 → 子空间 top-k → 笛卡尔积精排）
   - 在公开数据集（Criteo）上用小规模记忆表（N=10K）验证检索逻辑正确性
2. **接入已有 backbone**：
   - 在已有的 DeepFM / DCNv2 / 任意 backbone 后接 MSN 模块
   - 对比"backbone 单独" vs "backbone + MSN" 的离线指标

### Phase 2：离线复现（3-5 天）

1. **设计 Memory Gating**：
   - 实验加权聚合（mean pooling / attention pooling）
   - 实验门控融合（element-wise product / residual addition）
   - 对比不同融合策略的效果
2. **稳定化技术调参**：
   - Layer Norm 施加位置（query LN / key LN / 两者都做）
   - Learning Rate Warm-up 的步数和比例
   - Over-parameterization 的维度倍数
3. **记忆表规模实验**：
   - 从 N=10K → 100K → 1M 逐步扩展
   - 观察效果和检索开销的变化，找到收益拐点

### Phase 3：工程化（1-2 周）

1. **Sparse-Gather 实现**：
   - 在分布式训练框架中实现定制化的稀疏 gather 算子
   - 只同步被激活的记忆条目，避免全量同步
2. **AirTopK 集成**：
   - 集成高效的 GPU top-k 实现
   - 对小 k（2~8）场景做特别优化
3. **在线部署**：
   - 将 MSN 作为扩展模块接入线上 serving 流程
   - 监控 latency 和内存访问开销
   - AB 测试验证核心业务指标

---

## 九、美团业务场景适配建议

### 9.1 搜索排序（最契合场景）

MSN 论文的部署场景就是搜索排序，与美团的搜索业务高度契合：

- **Query 个性化**：不同 query 可以检索不同的记忆条目，实现 query 级别的个性化增强
- **即插即用**：美团搜索通常已有成熟的 backbone（如 DCN+Transformer），MSN 可以在不改动 backbone 的前提下接入
- **低延迟**：搜索排序的 latency 要求严格，MSN 的低内存访问成本非常适合

### 9.2 外卖 / 到店推荐

- **用户个性化**：外卖场景用户意图变化快（早餐/午餐/晚餐），MSN 的记忆检索可以为不同时段的用户检索不同的"时段记忆"
- **LBS 记忆**：可以为不同地理位置的用户维护地理相关的记忆条目，通过 PKM 高效检索

### 9.3 广告投放

- **低训练成本**：广告模型需要高频重训，MSN 的稀疏参数检索在训练阶段也能节省计算
- **广告主记忆**：可以为不同广告主或广告类别维护记忆条目，实现广告粒度的个性化

### 9.4 改进建议

1. **记忆表的内容设计**：
   - 论文中记忆表存储的是"通用 value 向量"
   - 美团可以设计更有语义的 memory：如"时段记忆""品类记忆""地理位置记忆"等
   - 不同语义的 memory 可以分别维护，通过多路 PKM 检索后融合

2. **冷启动策略**：
   - 新用户检索不到记忆时，可以 fallback 到全局平均记忆
   - 或设计一个"通用专家"与 MSN 并行，确保覆盖率

3. **记忆表更新频率**：
   - 记忆表参数随模型一起训练，但更新频率可以与 backbone 不同
   - 考虑 memory 的半监督更新：用高频日志微调 memory，低频全量训练 backbone

---

## 十、注意事项与风险提示

1. **预印本状态**：本文目前为 arXiv 预印本，部分实验细节（如具体在线提升幅度、记忆表规模 N 的取值）未在摘要中详述。
2. **Sparse-Gather 的工程门槛**：定制 CUDA kernel 需要 GPU 编程经验，对工程团队有一定要求。
3. **记忆表规模的上限**：虽然 PKM 将检索复杂度降到 O(√N)，但当 N 达到十亿级别时，√N 仍然很大（~31623），需要进一步优化（如层次化 PKM）。
4. **与 SMoE 的协同**：论文将 MSN 作为 SMoE 模型的扩展，但两者的收益是否 additive 还是存在重叠，需要具体场景验证。

---

> 如需将本文解读保存为 Markdown 文件，或进一步对比 MSN 与 UltraMem / LongMem 等 LLM 记忆网络的技术差异，请随时告知。
