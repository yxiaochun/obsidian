---
创建日期: 2026-09-15
更新日期: 2026-09-26
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2026ByteDance_TokenMixer-Large.pdf]]"
状态: 待复核
证据强度: 单篇工业论文证据
精读判定: 精读
待验证问题:
  - "不同稀疏比例和数据窗口下，深层模型的收敛稳定性如何？"
  - "在非字节私有数据、更小训练集和不同硬件栈上，Mixing & Reverting 与 Sparse-Pertoken MoE 的收益能否保持？"
  - "1:4 或更稀疏配置在高负载波动下是否会重新出现专家负载不均衡？"
---

# TokenMixer-Large：七十亿参数在线排序模型

## 论文信息

- 作者/机构：Yuchen Jiang 等，ByteDance AML。
- 发表信息：arXiv 预印本，2026-02-11，版本 v2；[DOI](https://doi.org/10.48550/arXiv.2602.06563)。
- 领域：工业推荐排序模型、特征交互、稀疏专家模型、Scaling Law。
- 与我的研究关联：直接承接 [[RankMixer：token混合让推荐模型MFU提升十倍]]，补足其深层残差、MoE 训练-推理稀疏化和大规模扩展证据，是理解工业排序模型纵向扩展的关键样本。

## 精读结论

TokenMixer-Large 不是把 RankMixer 简单加深，而是重排了信息通路：先用 mixing 让原始 token 聚合出混合 token，再用 reverting 恢复到原始 token 维度，保证残差两端语义对齐；随后用 Per-token SwiGLU 扩容，用 Sparse-Pertoken MoE 先稠密放大、再稀疏训练与稀疏服务，并用 interval residual、auxiliary loss、RMSNorm、gate value scaling 和 down-matrix 小初始化稳住深层训练。论文的真正主张是：推荐排序模型要继续扩展，必须同时满足语义残差、深层梯度、稀疏专家和硬件利用率四个条件。

## 模型结构

![[2026ByteDance_TokenMixer-Large.pdf#page=3]]

论文第 3 页 Figure 1 展示完整链路：原始特征与序列聚合结果先进入 tokenizer，再经过多个 TokenMixer-Large Block，每个 block 由 Norm、Mixing & Reverting、Sparse-Pertoken MoE 和 residual connection 组成，最后通过 mean pooling 输出任务 logits。主图上半部分给出 gate value scaling、shared expert、top-k 路由和 SwiGLU 的 MoE 细节。

## 七问笔记

| 问题 | 回答 |
| --- | --- |
| 研究背景 | LLM 驱动推荐模型扩展，但 Wukong、HiFormer、DHEN 等架构在硬件利用率和参数扩展上不足；RankMixer 已把 MFU 提升，却在深层、MoE 稀疏化和扩展性上遇到瓶颈。 |
| 研究目的 | 解决 RankMixer 的次优残差、深层梯度更新不足、不完整 MoE 稀疏化和扩展研究受限问题，使工业排序模型能稳定扩大到十亿参数量级并在线部署。 |
| 创新点 | 提出 Mixing & Reverting 让残差两端回到原始 token 语义，并把 MoE 从 dense-training/sparse-inference 升级为 sparse-training/sparse-inference 的 Sparse-Pertoken MoE。 |
| 研究方法 | Tokenization 先形成语义组 token 和 global token；主干堆叠 TokenMixer-Large Block，使用两层 mixing/reverting、Per-token SwiGLU、RMSNorm、interval residual 和 auxiliary loss；MoE 用先扩大后稀疏、shared expert、gate value scaling 和 down-matrix 小初始化稳定路由；系统侧采用 FP8 与 Token Parallel。 |
| 实验数据 | 主离线数据来自抖音主 feed 电商真实日志，500+ 特征、每天约 4 亿条、覆盖两年；另用抖音广告约 3 亿条/日和直播约 170 亿条/日数据。线上覆盖抖音广告、直播电商和直播。 |
| 结果结论 | CTCVR ΔAUC 对 DLRM-MLP-500M：TokenMixer-Large 500M +0.94%，超过 RankMixer +0.84%、Wukong +0.76%、HiFormer +0.44%；4B 稠密 +1.14%，1:2 稀疏 2.3B 激活参数仍为 +1.14%，FLOPs 从 29.8T 降到 15.1T；7B 稠密 +1.20%。线上带来直播电商人均预支付 GMV +2.98%、广告 ADSS +2.0%、直播收入 +1.4%。 |
| 总体评价 | 方法、消融、Scaling Law、在线 A/B 和系统优化闭环完整，能支撑“深层可扩展且可部署”的主张；但结论强依赖抖音私有数据、内部特征工程和自研部署栈，外部读者难以直接复现，MoE 超参与跨硬件稳定性仍需在更多环境验证。 |

## 方法拆解

- **Tokenization**：把特征按语义组先拼接，再用 MLP 投影到统一维度；同时从各组拼接生成一个 global token，将全局信息与增强 token 一起送入主干。
- **Mixing & Reverting**：RankMixer 的混合会把 T 个原始 token 变成 H 个 token；当 H 不等于 T 时，输入输出 token 语义错位。TokenMixer-Large 用第一层按 head 混出 H 个 token，第二层 revert 回 T 个 token，再接 Per-token SwiGLU、RMSNorm 和残差。
- **Per-token SwiGLU**：把 RankMixer 的 Per-token FFN 升级为每个 token 独立的 `FC_up`、`FC_gate`、`FC_down`，用 Swish 门控增强表达力，并保持 token 间参数隔离。
- **Residual & Normalization**：采用 Pre-Norm RMSNorm；附录显示 Post-Norm 精度虽略高但训练发散为 NaN，Sandwich Norm 下降 0.03%，因此 Pre-Norm 是深层稳定性的折中。
- **Interval residual & auxiliary loss**：每隔 2 或 3 层建立低层到高层的 interval residual，并把低层 logits 与高层 logits 联合辅助训练；最后一层不加 interval residual，以免低层信息干扰高层抽象。
- **Sparse-Pertoken MoE**：先构建稠密模型，再切分 Per-token SwiGLU 为子专家。每个 token 有一个 always-active shared expert 和 top-k 个 routed experts；训练和服务均稀疏激活。附录比较 1:2、1:4、1:8 稀疏比例，当前选择线上收益最高的 1:2。
- **Gate value scaling**：router 输出经 softmax 后会被压缩，导致选中专家的 SwiGLU 梯度不足。1:2 配置最优缩放因子为 2，1:4 配置为 4；附录解释该因子近似抵消稀疏化导致的梯度衰减。
- **Down-matrix small init**：把 `FC_down` 初始化标准差降到 0.01，让早期 SwiGLU 更接近 identity mapping。附录显示单独小初始化 `FC_down` 提升 0.03%，而整体小初始化或反转初始化位置反而退化。
- **系统优化**：自定义 MoEPermute、MoEGroupedSwiglu、MoEGroupedGemm、MoEUnpermute，并把主要计算写成 FP8；Token Parallel 让 token 维度跨设备并行，通信次数从约 4L 降到约 2L+1，配合通信-计算重叠在 batch size 320 下带来 29.2% 吞吐提升，重叠后达 96.6%。

## 实验证据

### 约 500M 参数离线对比

| 模型 | CTCVR ΔAUC | 参数量 | FLOPs/batch |
| --- | ---: | ---: | ---: |
| HiFormer | +0.46% | 570M | 28.8T |
| DCNv2 | +0.49% | 500M | 125.8T |
| DHEN | +0.63% | 413M | 103.4T |
| AutoInt | +0.75% | 546M | 138.6T |
| Wukong | +0.76% | 513M | 4.6T |
| Group Transformer | +0.81% | 580M | 4.3T |
| FAT | +0.82% | 531M | 4.5T |
| RankMixer | +0.84% | 500M | 4.6T |
| TokenMixer-Large 500M | +0.94% | 501M | 4.2T |

### 扩展与稀疏化

| 配置 | CTCVR ΔAUC | 参数量 | FLOPs/batch |
| --- | ---: | ---: | ---: |
| TokenMixer-Large 4B | +1.14% | 4.6B | 29.8T |
| TokenMixer-Large 7B | +1.20% | 7.6B | 49.0T |
| TokenMixer-Large 4B SP-MoE | +1.14% | 2.3B in 4.6B | 15.1T |

1:2 Sparse-Pertoken MoE 在激活约一半参数时匹配稠密 4B 的效果，说明“先扩大、再稀疏”在这组数据上是有效的成本-精度折中。FLOPs 减半并不自动等于线上时延减半，论文用 Token Parallel 和 FP8 补齐工程侧可行性。

### 与 RankMixer 的设计消融

| 版本 | SR | OTR | TSA | AUC | 参数量 | FLOPs |
| --- | :-: | :-: | :-: | ---: | ---: | ---: |
| Group Transformer | ✓ | ✓ | ✓ | -- | 500M | 4.2T |
| RankMixer w/o SR & OTR | ✕ | ✕ | ✕ | -0.20% | 510M | 4.2T |
| RankMixer w/o OTR | ✓ | ✕ | ✕ | -0.13% | 510M | 4.2T |
| RankMixer | ✓ | ✓ | ✕ | -0.03% | 567M | 4.2T |
| TokenMixer-Large | ✓ | ✓ | ✓ | +0.13% | 500M | 4.2T |

SR 表示标准残差，OTR 表示原 token 残差，TSA 表示残差两端 token 语义对齐。这项控制变量实验是论文方法叙事最强的部分：只补 OTR 还不够，只有语义也对齐时才获得正收益。

### 组件消融

| TokenMixer-Large 4B 设置 | ΔAUC |
| --- | ---: |
| 去掉 Global Token | -0.02% |
| 去掉 Mixing & Reverting | -0.27% |
| 去掉 Residual | -0.15% |
| 去掉 Interval Residual 与 Auxiliary Loss | -0.04% |
| Per-token SwiGLU 换成 Shared SwiGLU | -0.21% |
| Per-token SwiGLU 换成 Per-token FFN | -0.10% |

| Sparse-Pertoken MoE 设置 | ΔAUC | Δ参数 | ΔFLOPs |
| --- | ---: | ---: | ---: |
| 去掉 Shared Expert | -0.02% | 0.0% | 0.0% |
| 去掉 Gate Value Scaling | -0.03% | 0.0% | 0.0% |
| 去掉 Down-Matrix Small Init | -0.03% | 0.0% | 0.0% |
| 换成标准 Sparse MoE | -0.10% | 0.0% | 0.0% |

### 收敛与 Scaling Law

| 抖音直播配置 | 收敛天数 | ΔUAUC |
| --- | ---: | ---: |
| 90M | 14d | +0.94% |
| 500M | 30d | +0.62% |
| 2.3B | 30d | +0.41% |
| 2.3B | 60d | +0.70% |

模型越大，收敛所需样本越多；500M 到 2.3B 在 30 天窗口只获得 +0.41%，60 天窗口升到 +0.70%。Feed Ads、E-Commerce 和 Live Streaming 的稠密扩展曲线都保持正斜率，其中离线分别验证到 15B、7B 和 4B。

### 线上 A/B

| 场景 | 参数量 | 基线 | 业务指标提升 |
| --- | ---: | ---: | --- |
| 抖音广告 | 7B | RankMixer-1B | ΔAUC +0.35%，ADSS +2.0% |
| 直播电商 | 4B | RankMixer-150M | ΔAUC +0.51%，订单 +1.66%，人均预支付 GMV +2.98% |
| 直播 | 2B | RankMixer-500M | ΔUAUC +0.7%，收入 +1.4% |

## 关键图表解读

- **Figure 1**：给出 tokenizer、多块主干、Mixing & Reverting、Sparse-Pertoken MoE、global token、mean pooling 和任务输出的整体架构。
- **Figure 2**：展示 interval residual 和 auxiliary loss 的组织方式；辅助分支帮助低层学习高层目标，interval residual 为深层提供低层信息通路。
- **Figure 3**：把一个 block 拆成 permute、grouped SwiGLU/gemm、unpermute 等算子，并用星号标注 FP8 数据；主要服务时间消耗在 MoEGroupedFFN。
- **Figure 4**：三个场景的稠密 Scaling Law 都保持正向拟合；不同场景的参数上限和数据量不同，不能把某条曲线的截距直接外推到其他业务。
- **Figure 5**：在相同参数和 FLOPs 轴上比较 RankMixer 与 TokenMixer-Large，后者斜率更高，说明新设计改善的是扩展效率而不只是单点精度。
- **Figure 6**：演示 dense、1:2 和 1:4 稀疏形态。shared expert 一直激活，routed expert 按稀疏比例切分。
- **Figure 7**：1:2 配置负载相对均衡，1:8 配置负载均衡劣化。这解释了论文为何选择 1:2 而不是更激进稀疏。
- **Figure 8**：展示 SwiGLU 中 `FC_down` 小初始化的位置；它让早期输出更接近恒等映射，配合 up/gate 矩阵的常规初始化提升深层稳定性。

## 批判性分析

- **Why 动机充分**：论文不是只报告参数变大，而是先指出 RankMixer 在 token 数变化、残差语义、深层梯度、MoE 训练稀疏化上的具体矛盾，再针对每项给出设计。Mixing & Reverting 与 Table 3 的控制变量实验形成清晰因果链。
- **为什么不用已有 MoE**：RankMixer 的 ReLU-MoE 偏向 dense train/sparse infer，训练成本没有下降。TokenMixer-Large 先保证稠密版性能，再切分 SwiGLU，使训练和推理都能稀疏。选择 shared expert 是为了避免每个 token 都学习独立路由组；选择 1:2 是因为 1:8 会出现负载均衡问题。
- **为什么补多层稳定机制**：深层 stack 后，仅靠标准残差不能保证低层信息和梯度到达高层。interval residual 提供低层到高层通路，auxiliary loss 给低层可学习目标，down-matrix 小初始化控制早期输出方差。
- **实验设计缺口**：缺少跨公司或公开数据集对照；未给出特征 tokenizer 的完整组定义；MoE 的专家数、top-k、负载均衡损失和路由漂移的时间曲线披露有限；在线 A/B 未提供置信区间、分流周期、反事实校准和硬件/QPS 细节。
- **指标选择可辩护但有偏面**：AUC/UAUC 适合排序质量，CTCVR 和业务 GMV/ADSS/收入覆盖离线与在线，但没有报告训练能耗、成本、服务方差和长尾用户细分收益。
- **换位思考**：如果我来组织实验，会先固定 FLOPs 和 latency 预算，再比较 RankMixer、Group Transformer、Wukong、FAT；为每个稀疏比例补充负载均衡熵、专家利用率方差和服务时延；把 15B 离线模型的完整曲线、早期震荡、显存峰值和失效点放入主文。技术上可继续探索：用可学习的 token 分组替代固定语义组，把 gate value scaling 改成逐专家自适应，测试 1:4 稀疏在更长训练和更大流量下的稳定性。
- **优点**：工业闭环完整；Table 3 对三属性做了干净消融；组件消融、Scaling Law、在线 A/B 和系统优化互相支撑。
- **不足**：核心数据私有，外部泛化不明；方法收益高度依赖工程栈；“先扩大后稀疏”的总参数可能仍抬高内存占用；MoE 最优超参与稀疏比例强耦合，可迁移性有限。

## 值得追踪的引用

- [[RankMixer：token混合让推荐模型MFU提升十倍]]：本文直接前代，用于理解 lightweight token mixing 的原设计和瓶颈。
- [[堆叠因子分解机实现推荐系统缩放定律：Wukong 在 100+ GFLOPexample 下持续提升质量|Wukong 的推荐系统缩放实验]]：本文离线对比中的高效特征交叉基线。
- [[DeepSeekMoE：细粒度专家分割与共享专家隔离实现极致专家专业化]]：Sparse-Pertoken MoE 的共享专家与细粒度专家思想来源之一。
- ReZero：本文 down-matrix small init 的直接灵感，用于理解小初始化对深层恒等映射和收敛的作用。
- RMSNorm：本文替换 LayerNorm 的基础，需要继续核对深层推荐网络中的数值稳定与算子开销权衡。

## 术语与句式积累

- **Mixing & Reverting**：先混合 token、再恢复原始 token 数，保证残差两端语义和维度一致。
- **Sparse-Pertoken MoE**：把 Per-token SwiGLU 切成子专家，训练与服务都稀疏激活。
- **Gate Value Scaling**：放大 router 权重以缓解 softmax 后梯度不足。
- **Token Parallel**：按 token 维度切分模型参数和计算，减少多设备训练/推理通信。
- 可复用句式：“残差不仅要求形状一致，也要求 token 语义对齐。”“稀疏化只有同时降低训练和服务成本，才真正改变扩展预算。”

## 复现清单

- **数据**：抖音主 feed 电商真实训练数据，500+ 特征、每天约 4 亿条、两年数据；私有数据不可直接获得，公开数据复现需重新设计特征分组。
- **任务与指标**：CTR/CVR 任务的 AUC 与 UAUC；同时记录参数量、每 batch 2048 样本的训练 FLOPs 和 MFU。
- **代码**：论文未提供官方实现。
- **环境**：64 GPU 混合分布式框架，跨场景包含 26 个 Feed-Ads 与 38 个 Live-Streaming worker；稀疏嵌入异步更新，稠密参数同步更新；Adagrad 学习率 0.01 和 0.05。未公开 GPU 型号、互联拓扑、精确显存和集群成本。
- **关键超参**：1:2 Sparse-Pertoken MoE，shared expert 常开，routed experts top-k 激活；gate value scaling 在 1:2 为 2、1:4 为 4；`FC_down` 初始化标准差 0.01，`FC_up/FC_gate` 为 1；interval residual 每隔 2 或 3 层，最后一层不加。
- **可验证步骤**：先复现稠密 TokenMixer-Large block 和 Mixing & Reverting，再验证 Table 3 的 SR/OTR/TSA 消融；随后从稠密模型切分 Sparse-Pertoken MoE，比较 1:2、1:4 和 1:8 的精度、专家负载、吞吐和时延；最后在固定 FLOPs 预算下复现 Scaling Law 拟合。
- **缺失信息**：特征组的完整映射规则、数据采样与窗口细节、专家总数、top-k、负载均衡损失系数、线上硬件型号、服务 QPS、分桶周期和统计显著性检验。

## 关联

- [[RankMixer：token混合让推荐模型MFU提升十倍]]：前代架构，本文修补其深层扩展和 MoE 稀疏化限制。
- [[UGSep：用户侧计算复用降低大模型推理成本]]：从用户侧复用角度补充推理成本优化。
- [[DeepSeekMoE：细粒度专家分割与共享专家隔离实现极致专家专业化]]：共享专家与细粒度专家设计的通用参照。
