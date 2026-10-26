---
创建日期: 2026-10-26
类型: 笔记
标签:
  - 推荐系统
  - 生成式推荐
来源:
  - "Multi-gate Mixture-of-Experts for Multi-task Learning（KDD 2018, Google）"
  - "Kendall et al., Multi-Task Learning Using Uncertainty to Weigh Losses（CVPR 2018）"
  - 2026-10-26 与 CatPaw 的技术讨论整理
状态: 待复核
证据强度: 经典论文机制推演加公开工业报道，未做逐篇原文核对
---

# MMoE用任务私有门控路由共享专家缓解多任务冲突

MMoE（Multi-gate Mixture-of-Experts，KDD 2018）解决的核心问题是：多个相关但可能冲突的任务共享表征时的 trade-off。核心设计是**Expert 全体共享、Gate 每任务私有一份**，把任务间的冲突从共享底座内部转移到低维的门控权重上消化。

## 1. 结构演进脉络

```
Shared-Bottom → One-gate MoE → MMoE → CGC/PLE / SNR → MoE-LLM（会师）
```

### Shared-Bottom

所有任务共用一个底层网络，再接各自 Tower 输出多任务头。任务相关性高时效果好；相关性低甚至冲突时，共享底座被迫学折中表征，互相拉扯，可能出现跷跷板现象（seesaw phenomenon）：一个任务涨、另一个掉，效果可能不如单任务模型。

### One-gate MoE

底层分成多个 Expert（普通 MLP），一个 Gate（线性层 + softmax）把 Expert 输出加权求和，**全体任务共用同一份混合权重**。冲突仍未根本解决：不同任务的梯度在同一 Gate 参数上直接对抗，最终学出折中权重，本质只是把"共享一个网络"松绑为"共享一个加权方式"，解耦程度有限。

### MMoE

每个任务拥有自己独立的 Gate，各自学习对多个 Expert 的不同加权组合：

```
输入 x
  ├─ Expert_1 ... Expert_E        （共享参数，只算一次）
  ├─ Gate_a(x) → softmax → w_a    （任务 a 的门控）
  └─ Gate_b(x) → softmax → w_b    （任务 b 的门控）

h_a = Σ_i w_a[i]·Expert_i(x)  → Tower_a → 任务 a 输出
h_b = Σ_i w_b[i]·Expert_i(x)  → Tower_b → 任务 b 输出
```

关键性质：

- **Expert 参数共享**（保证特征复用与正向迁移），**Gate 参数任务私有**（梯度不再打架）。
- 任务高度相关时，各 Gate 收敛到接近的分布，自动退化为 Shared-Bottom 行为，无需人工判断任务是否该共享。
- 任务冲突时，各 Gate 学到差异很大的分布，各自偏向不同 Expert，冲突在 Gate 层表面化。
- 论文在合成数据上通过控制任务标签相关性 ρ 验证：ρ 高时三种结构接近；ρ 低时 Shared-Bottom 崩得最狠，One-gate MoE 部分缓解，MMoE 最稳。

反传行为是理解全部技巧的入口：任务 a 的梯度更新 `Gate_a / Tower_a / 全部 Expert`，任务 b 同理。**Expert 拿到的是多任务梯度的加权和（共享），Gate 只拿到自己任务的那份（隔离）**。

### 后续演进

- **CGC / PLE**（RecSys 2020, Tencent）：Expert 分为共享 + 任务专属，并堆叠为多层，针对性解决跷跷板。经验上多任务多场景大厂落地时 PLE 系居多。
- **SNR / Cross-Stitch / Sluice**：子网络级自动路由，学习共享程度。
- **MoE-LLM 一脉**（Switch / Mixtral / DeepSeek-MoE）：稀疏激活多专家路由，目标从"缓解冲突"变成"稀疏扩容"，见第 4 节。

## 2. 与 Uncertainty Weighting 的组合

MMoE 解决"梯度往哪儿喂"（结构层面），Uncertainty Weighting（CVPR 2018, Kendall et al.）解决"K 份梯度以多大强度混合"（权重层面），两者正交，工业上常叠加。

### 公式

$$L_{total}(\theta, \sigma_{1..K}) = \sum_k \frac{1}{2\sigma_k^2} L_k(\theta) + \log \sigma_k$$

第一项加权 loss，第二项 $\log \sigma_k$ 是护栏：防止网络把所有 σ 学到无穷大让加权项趋近零而躺平。

### 原理

- 源自异方差偶然不确定性的高斯极大似然：任务噪声大 → σ 大 → 权重小，不让震荡梯度带坏共享底座。
- 对固定 $L_k$ 令梯度为零可解出 $\sigma_k^2 = L_k$：**σ² 的唯一不动点就是该任务当前的 loss 水位**。
- 梯度层面（令 $s_k = \sigma_k^2$）：

$$\frac{\partial L_{total}}{\partial s_k} = \frac{s_k - L_k}{2s_k^2}$$

符号完全由 $s_k - L_k$ 决定：loss 高于水位 → σ² 被推高 → 权重坍缩；loss 低于水位 → σ² 被压低 → 权重放大。实用版里 $L_k$ 不依赖 $s_k$，这就是普通交替梯度下降，没有任何黑盒。
- 平衡点处权重 $\approx 1/(2L_k)$，等价于**每个任务的 loss 量纲自动归一化**：MSE（量级 4）和交叉熵（量级 0.1）无需人工对齐。
- 严格版分类推导采用 logit 缩放 $z_k/\sigma_k^2$，但工程实践几乎都直接套回归版加权到交叉熵上。

### 实现要点

```python
self.log_vars = nn.Parameter(torch.zeros(num_tasks))   # log 空间参数化

total = Σ_k [ 0.5 * exp(-log_vars[k]) * L_k + 0.5 * log_vars[k] ]
```

- 必须用 log 空间保证 $s_k > 0$ 且数值稳定。
- 护栏 $\log σ$ 只是对数级压力，某任务梯度为零时 σ 仍可能漂大，需监控各 σ 轨迹。
- 相比 GradNorm（要维护全局梯度范数滑动统计并 min-max 归一化任务规模），只需 K 个标量参数，开销近零，这在工业多任务排序里更常见。

## 3. 工业应用场景

- **多目标预估（最主流）**：排序模型同时预估 CTR、CVR、时长、互动、复购等。CTR 偏浅层兴趣、CVR 偏深度意图，直接共享底座必然互相干扰；MMoE/CGC/PLE 是广告与推荐排序预估层的默认底座。
- **广告多目标**：点击、转化、深层转化（付费/次留）联合建模；与 ESMM（解样本选择偏差）正交可叠加。
- **多场景建模**：Gate 思想从"任务 gate"扩展为"场景 gate"：STAR、PEPNet、一阶段多场景统一建模中，不同场景对 Expert / 特征变换 / 整层参数做不同加权，思想与 MMoE 同源（生成式推荐一侧可参见 [[TAE：基础模型加专家范式的超规模部署]]、[[MTFM：免对齐的多场景推荐基础模型]]、[[MDL：场景与任务token化统一多分布学习]]）。
- **不限于推荐**：小模型时代的多任务学习（CV 分割+深度联合估计、风控多目标检测）通用底座。

## 4. LLM 与生成式推荐中的两条 MoE 脉络会师

如今"MoE"一词下两条脉络与 MMoE 同源（多专家 + 门控加权），方向不同但在生成式推荐里重新会师：

| | MMoE 线（多任务学习脉络） | MoE-LLM 线（稀疏容量脉络） |
|---|---|---|
| 动机 | 缓解多任务冲突 | 稀疏激活、扩大容量 |
| 专家形态 | 小 MLP，每任务一个 gate | Transformer FFN 切分为多专家，token 级路由 |
| 代表 | MMoE / CGC / PLE | Switch / Mixtral / DeepSeek-MoE |

### LLM 主线

- 稀疏 FFN 路由：Mixtral（8 选 2）、DeepSeek-V2/V3 的**细粒度专家 + 共享专家**设计，与 CGC/PLE 的"共享 Expert + 任务专属 Expert"几乎同一思想在不同尺度的复现，详见 [[DeepSeekMoE：细粒度专家分割与共享专家隔离实现极致专家专业化]]。
- 多任务 LLM 微调防灾难遗忘：LoRAMoE（ICLR 2023）、MoLE（ACL 2024）把多个 LoRA adapter 当专家、任务间门控路由，backbone 冻结——"gate per task"直接移植到 PEFT 上。

### 生成式推荐侧

- **OneRec 系（快手）**：Encoder-Decoder 端到端生成式推荐明确采用 MoE + RL；多目标问题转为 ECPO / 多奖励对齐，是 MMoE 曾解决的多任务冲突在端到端框架下的新形态——从"多 loss 梯度打架"变成"多 reward 的对齐权衡"。见 [[OneRec：统一召回与排序的端到端生成式推荐]]。
- **多业务跷跷板重演**：多业务多目标联合训练一个生成式大模型时，MMoE 时代的 seesaw 原样重演（美团 MBGR 针对多业务跷跷板设计，腾讯 GPR/UniSID 处理广告+内容一体化的语义 ID 码本冲突）。解法高度 MMoE 化：语义 ID 侧用 MoE 码本（如 UniTok 的 MoE + 多码本），模型侧做模块隔离。
- **判别式 Scaling 路线的延续**：字节 [[RankMixer：token混合让推荐模型MFU提升十倍]]、[[TokenMixer-Large：七十亿参数在线排序模型]]（Sparse-Per-token MoE）与快手 [[UniMixer：把TokenMixer参数化成可学习稀疏置换矩阵统一三种推荐Scaling路线]] 走判别式扩展，但多场景多目标的实现底层仍是 PEPNet/MMoE 这套门控路由思想，只是 gate 的作用对象从 Expert 扩展到 prompt、特征变换和整层参数。
- Meta GR/HSTU 到 ULTRA-HSTU 把推荐模型推到 LLM 算力形态，使 MoE-LLM 的全部工程改造（专家并行、负载均衡、专家卸载）可平移到推荐模型，见 [[生成式推荐：用万亿参数序列转换器 HSTU 替代 DLRM，验证推荐领域的缩放定律]]。

## 5. 横向对照

| 结构 | Expert 共享方式 | Gate | 冲突处理 | 典型代价 |
|---|---|---|---|---|
| Shared-Bottom | 全部共享 | 无 | 无 | 任务冲突时互相拉扯 |
| One-gate MoE | 全部共享 | 单份共通 | 通过多 Expert 部分松绑 | Gate 梯度对抗仍折中 |
| MMoE | 全部共享 | 每任务一份 | 冲突转移到门控层 | Expert 利用不均（门控坍缩） |
| CGC/PLE | 共享 + 任务专属混合 | 任务间/任务内双层 | 显式划分共享与私有 | 结构与调参复杂度上升 |
| MoE-LLM | 全局专家池 | token 级稀疏路由 | 服务容量扩展而非任务冲突 | 负载均衡、专家并行工程 |

## 6. 局限与工程要点

- **门控坍缩**：训练后常见 1~2 个 Expert 门控权重主导、其余更新不足。缓解：gate dropout / 温度、负载均衡正则、升级 CGC/PLE。
- **没有理论保证解决全部冲突**：Expert 间仍是全输入共享，特征层面无真正私有参数。
- Expert 数量与容量折中需 A/B 验证，不是越多越好；论文实验 1/4/8 个。
- 多任务 loss 权重震荡会带动 Gate 分布震荡，配 Uncertainty Weighting 等自动调权更稳。
- 数据量过少的任务 σ 学习信号不足时，Uncertainty Weighting 退化接近固定权重，此时考虑 GradNorm / DWA。

## 7. 与全站推广工作的关联

排序公式 `eCPM + k·pGMV` 及 OCPC 多阶段目标（冷启动扶持 → 留存 → 停投打压）的联合建模，本质都是"多目标该共享还是隔离"的问题：小模型上对应 MMoE/PLE 的结构选择，生成式推荐上对应多 reward 对齐与模块分治，两套方法论贯通。

## 待办

- [ ] 核对 OneRec V2 中 MoE 的具体位置（Encoder 侧 / Decoder 侧 / 官方技术报告原文口径）
- [ ] 补充 CGC/PLE 原文的合成实验数字对照
- [ ] 确认 MBGR（美团多业务跷跷板）论文的模块隔离设计细节
