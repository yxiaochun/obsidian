---
创建日期: 2026-09-29
更新日期: 2026-11-11
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源:
  - "[[02_Knowledge/模型与案例/GR4AD：广告生成式推荐的架构训练推理联合设计.md]]"
  - "[[01_Sources/论文报告/2026Kuaishou_GR4AD.pdf]]"
  - "https://arxiv.org/html/2305.18290"
  - "https://arxiv.org/html/2402.03300v3"
  - "https://arxiv.org/html/2502.18965v1"
---

# DPO、GRPO、RSPO：从逐条偏好优化到列表级业务价值优化

三者都在监督模仿之外引入偏好或业务收益信号，区别在「优化什么目标、怎么构造对比信号、要不要采样和参考模型」。DPO 直接使用偏好对训练，也可以迭代更新偏好数据；GRPO 用同一请求下的一组采样结果构造相对优势，奖励可以来自规则或学习型评估器；RSPO 是快手 GR4AD 为广告场景设计的排序偏好优化方法。

## 原理

### DPO（Direct Preference Optimization）

给定请求 $x$、优选序列 $y_w$ 与劣选序列 $y_l$，DPO 提高新模型相对参考模型对优选序列的偏好。这里的优劣来自标注或奖励模型，不能理解成全局最优与最差。

#### 从奖励最大化到偏好损失

从带 KL 正则的奖励最大化目标出发：

$$
\max_\pi\ \mathbb{E}_{y\sim\pi(\cdot\mid x)}[r(x,y)]-\beta D_{\mathrm{KL}}\!\left(\pi(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\right)
$$

其最优策略满足：

$$
\pi^*(y\mid x)=\frac{\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)}{Z(x)}
$$

因此奖励可重写为 $r(x,y)=\beta\log[\pi^*(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]+\beta\log Z(x)$。假设偏好服从 Bradley–Terry 模型：

$$
P(y_w\succ y_l\mid x)=\sigma\!\left(r(x,y_w)-r(x,y_l)\right)
$$

代入后，同一请求下的 $\log Z(x)$ 抵消，用参数化策略拟合偏好，得到：

$$
\mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}_{(x,y_w,y_l)}\log\sigma\!\left[\beta\left(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right]
$$

这是在相应偏好模型假设下，把奖励与策略之间的关系代入偏好似然，并非任意偏好数据都能保证真实收益最优。来源：[DPO 原论文 §4](https://arxiv.org/html/2305.18290)。

#### 分差直觉：不要求当前模型整体打分更高

定义序列分数 $s_\theta(y)=\log\pi_\theta(y\mid x)$，以及新旧模型各自的优劣分差：

$$
m_\theta=s_\theta(y_w)-s_\theta(y_l),\qquad m_{\mathrm{ref}}=s_{\mathrm{ref}}(y_w)-s_{\mathrm{ref}}(y_l)
$$

单对损失就是 $-\log\sigma[\beta(m_\theta-m_{\mathrm{ref}})]$。优化方向是**让新模型的优劣分差相对参考模型增大**，不是另外要求所有序列分数上涨。

| 模型 | 优选序列 log-probability | 劣选序列 log-probability | 优劣分差 |
| --- | ---: | ---: | ---: |
| 参考模型 | −10 | −12 | 2 |
| 新模型 A | −9 | −13 | 4 |
| 新模型 B | −11 | −15 | 4 |

A、B 对这个偏好对的 DPO loss 相同。即使 B 把两条序列概率都降低了，优选序列跌得更少，也符合相对优化方向。这种现象本身不能单独证明模型退化，需要结合真实收益与其他指标判断。

#### OneRec 如何把 DPO 用于推荐列表

1. 当前模型 $M_t$ 针对用户历史生成多条 session，用 RM 选出最高分 $y_w$ 与最低分 $y_l$；两者可以包含不同商品，不限于同集合重排。
2. 新模型 $M_{t+1}$ 从 $M_t$ 初始化；本轮中 $M_t$ 作为冻结参考模型，损失为 $\mathcal{L}_{\mathrm{NTP}}+\lambda\mathcal{L}_{\mathrm{DPO}}$。
3. 下一轮用更新后的模型重新产生偏好数据与参考快照。论文用 1% 的训练请求构造 DPO 数据，以控制生成与评估成本。

整个列表的 log-probability 是各语义 ID token 条件 log-probability 的和：

$$
\log\pi_\theta(y\mid x)=\sum_{t=1}^{T}\log\pi_\theta(a_t\mid x,a_{<t})
$$

DPO 参数更新不要求穿过离散采样过程求导；RM 在这里负责构造偏好标签，其分数差没有直接作为权重进入上述标准损失。**DPO 不必显式训练 RM，但也不排斥 RM；OneRec 就依赖 RM 产生偏好对。** 来源：[OneRec §3.3 与算法 2](https://arxiv.org/html/2502.18965v1)。

### GRPO（Group Relative Policy Optimization）

以下为完整序列奖励（outcome supervision）版本。对同一请求 $x$，从采样策略 $\pi_{\mathrm{old}}$ 生成 $G$ 条序列，各自取得 $r_i=R(x,Y_i)$。奖励可由规则、业务公式或学习型 Evaluator 提供；GRPO 省掉的是 PPO 的价值网络（critic），不是奖励来源。

$$
\bar r=\frac1G\sum_i r_i,\qquad A_i=\frac{r_i-\bar r}{\operatorname{std}(r_1,\ldots,r_G)+\epsilon_{\mathrm{num}}}
$$

$\epsilon_{\mathrm{num}}$ 是防止除零的小数，区别于后面的裁剪阈值 $\varepsilon$。完整序列监督将同一个 $A_i$ 用于该序列每个生成动作；优势与奖励在策略更新中作为固定信号，不对它们反向传播。组内奖励相同时，奖励驱动的优势为零，KL 等其他项仍可能产生更新。

#### 完整损失与三个策略的分工

设序列 $Y_i$ 的长度为 $T_i$，$a_{i,t}$ 是第 $t$ 个动作（商品或语义 ID token）。概率比为：

$$
\rho_{i,t}=\frac{\pi_\theta(a_{i,t}\mid x,a_{i,<t})}{\pi_{\mathrm{old}}(a_{i,t}\mid x,a_{i,<t})}
$$

原始 GRPO 按序列长度归一化的损失可写为：

$$
\mathcal{L}_{\mathrm{GRPO}}=-\mathbb{E}\!\left[\frac1G\sum_i\frac1{T_i}\sum_t\left\{\min\!\left[\rho_{i,t}A_i,\operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_i\right]-\beta K_{i,t}\right\}\right]
$$

其中 $K_{i,t}$ 是相对参考策略的 KL 惩罚估计。原文使用 $q_{i,t}-\log q_{i,t}-1$，其中 $q_{i,t}=\pi_{\mathrm{ref}}(a_{i,t}\mid x,a_{i,<t})/\pi_\theta(a_{i,t}\mid x,a_{i,<t})$。后续变体可能改变 KL、标准化或长度归一化方式。

- $\pi_\theta$：正在更新的策略。
- $\pi_{\mathrm{old}}$：产生本批样本的策略快照，本批更新期间固定，下一批可刷新；概率比以它为分母，允许有限度复用这批采样数据。
- $\pi_{\mathrm{ref}}$：KL 约束的参考策略，可在更长阶段内固定，也可按阶段更新；不能与采样策略混为一谈。

来源：[DeepSeekMath §4.1，公式 3–4 与算法 1](https://arxiv.org/html/2402.03300v3)。

### RSPO（Ranking-Guided Softmax Preference Optimization）

场景：广告生成式推荐。两个特殊约束：优化目标是**按 eCPM 排列的列表**，per-item 奖励不足以刻画 list-wise 目标；训练样本来自**多条异构生产管线**，很多没有可靠参考分布。

用 Lambda/NDCG 框架把「列表按 eCPM 排序的 NDCG 损失」写成 softmax 对比损失的上界（GR4AD 式 16）：
$$\mathcal{L}_{\text{RSPO}}=-\mathbb{E}\left[\log_2\sigma\left(-\log\sum_{y_j\in E_i}M_{ij}\exp\left(\beta\log\tfrac{p_\theta(y_j|X)}{p_{\text{ref}}(y_j|X)}\right)-\beta\log\tfrac{p_\theta(y_i|X)}{p_{\text{ref}}(y_i|X)}\right)\right]$$
其中 $E_i=\{y_j: v_j<v_i\}$ 是奖励（eCPM）低于 $y_i$ 的候选集合，$M_{ij}$ 是标准 Lambda 权重（DCG 位置折扣 × 奖励差）。Ranking-Guided 指它是 NDCG 损失的上界；Softmax 指直接对候选集合做对比，不需要启发式构造 chosen–rejected 对。

参考模型用二值门控 $C_{ij}$ 控制（式 18）：只有参考分布可用、且当前模型与它的 log-ratio 偏差小于阈值 δ 时才启用；否则整块退化为无参考目标，避免过期或不可靠的 $p_{\text{ref}}$ 引入噪声正则。

## 对比

| | DPO | GRPO | RSPO |
|---|---|---|---|
| 范式 | 偏好对训练，可迭代刷新数据 | on-policy / 近 on-policy RL | 在线学习流中的 list-wise RL |
| 对比信号 | chosen–rejected 对（人工/启发式） | 组内奖励标准化 | 列表内按 eCPM 排序的候选集 $E_i$ |
| 奖励类型 | 标准损失使用优劣标签，不直接使用 reward 差值 | 外部标量奖励，可连续；此处讨论 outcome 版 | 连续业务价值（eCPM，稠密） |
| 优化粒度 | 一对输出；每个输出可以是完整推荐列表 | 一组输出；每个输出可以是完整推荐列表 | Lambda 权重编码候选排名与位置折扣 |
| 参考模型 | 全程必须 | KL 惩罚项 | 门控启用（可用且偏差 < δ 才用） |
| 采样需求 | 参数更新可用固定偏好对；迭代版本另行生成新数据 | 每个训练请求采 G 个输出，可短期复用 | 不强求 on-policy；接受多管线日志列表 |
| 理论锚点 | RLHF 目标解析解 | PPO 去 critic | NDCG 损失的上界 |
| 典型场景 | LLM 对齐 | 可验证奖励的推理任务 | 广告/推荐的业务价值排序 |

## GRPO 数值例子

### 推荐列表设定

候选为 A、B、C，第一位已选 A。第二位选 C 得到 $Y_+=[A,C,B]$，reward 为 3；选 B 得到 $Y_-=[A,B,C]$，reward 为 1。组均值为 2，采用总体标准差时为 1，因此优势为 $A_+=+1,A_-=-1$（忽略极小的数值稳定项）。

旧策略在前缀 $[A]$ 下，选 C、B 的概率都是 0.5。下面仅比较第二位的动作损失，忽略 KL 与其他位置，并取 $\varepsilon=0.2$：

$$
\ell=-\min\left[\rho A,\operatorname{clip}(\rho,0.8,1.2)A\right]
$$

### 概率比与局部 loss 的变化

| 当前概率 $p(C),p(B)$ | 概率比 $\rho_C,\rho_B$ | 好动作 C 的 loss | 差动作 B 的 loss | 两项平均 |
| --- | --- | ---: | ---: | ---: |
| 0.50，0.50 | 1.0，1.0 | −1.0 | +1.0 | 0 |
| 0.55，0.45 | 1.1，0.9 | −1.1 | +0.9 | −0.1 |
| 0.60，0.40 | 1.2，0.8 | −1.2 | +0.8 | −0.2 |
| 0.70，0.30 | 1.4，0.6 | −1.2 | +0.8 | −0.2 |
| 0.30，0.70 | 0.6，1.4 | −0.6 | +1.4 | +0.4 |

$\rho_C=1.2$ 表示相对旧概率增加 20%，即从 0.5 到 0.6，不是增加 0.2。表中是指定动作项的局部平均，并非含 KL 与全部 token 的完整训练 loss。

以当前概率 0.7、0.3 为例：

$$
\ell_C=-\min(1.4,1.2)=-1.2
$$

$$
\ell_B=-\min(0.6\times(-1),0.8\times(-1))=+0.8
$$

沿正确方向继续提高 C、降低 B，已经不能从这两项获得更多收益。但若反向变成 0.3、0.7，loss 仍会变差，保留纠正信号。这就是取 min 的作用：**改善足够多时停止奖励，向错误方向恶化时仍然纠正。**

在未进入裁剪平台时，$\ell=-A\rho$，因此：

$$
\frac{\partial\ell}{\partial\log\pi_\theta(a\mid s)}=-A\rho
$$

梯度下降对正优势动作倾向于提高概率，对负优势动作倾向于降低概率；进入改善方向的裁剪平台后，该动作的这个代理项梯度为零。新旧策略相同时，两项 loss 数值可以抵消为零，但对参数的梯度通常不为零。

> [!warning] clip 不是概率硬上限
> 它不强制概率比始终留在区间内；其他样本、共享参数和 KL 项仍可改变这个动作概率。重新采样并刷新旧策略后，概率比的基准也随之更新。

### 推荐场景的边界

- 组内采样发生在训练或采样服务，不等于每个线上请求都必须生成多条列表交给评估器选优；训练与 serving 的成本应分开估算。
- 完整列表 reward 可以包含位置、组合与外部性信息；但将同一个优势分给所有动作，不能细分各位置的贡献。GRPO 本身不自动解决信用分配问题。
- 奖励可以是连续业务收益，也可以来自学习型评估器。效果取决于 reward 是否可靠、采样是否有差异以及策略是否覆盖有价值的输出；并非换成 GRPO 就自动学到外部性。
- 同批采样可以短期复用，并非每做一步梯度更新就必须丢弃数据；长期偏离采样分布仍需额外处理，clip 不会消除所有分布偏差。

## 为什么 GR4AD 要新造 RSPO

GR4AD 消融（在 VSL 基础上，相对 DLRM 基线的广告收入增益）：VSL + DPO 为 +3.16%，VSL + GRPO 为 +3.21%，VSL + RSPO 为 +3.86%。RSPO 的优势来自三点组合：

1. **奖励形态匹配**：RSPO 直接利用 eCPM 排序与收益差。标准 DPO 主要使用优劣标签；GRPO 可以使用连续奖励，不能把稀疏奖励视作它的必然限制。
2. **list-wise 目标匹配业务**：RSPO 显式编码位置折扣；DPO/GRPO 是否体现列表位置与组合价值，取决于偏好、reward 和输出单位，不能仅凭算法名称认定其忽略列表信息。
3. **参考模型门控**：生产日志大量样本无 $p_{\text{ref}}$ 或参考分布因在线学习过期，DPO/GRPO 强制使用会引入噪声，RSPO 用门控规避。

此外 GR4AD 没有把 RSPO 当独立阶段，而是与 VSL 统一在一条在线学习流：用「模型排序 vs 奖励排序的秩偏差」$A^{(i)}=|r_p-r_v|/(n-1)$ 动态分配样本权重——偏差大时加重 VSL（模仿用户兴趣），偏差小时加重 RSPO（推向高价值）。这是该方案的联合训练设计，不能据此认定 DPO/GRPO 必须与监督训练分成两个阶段；OneRec 就组合了 NTP 与 DPO 损失。

> [!warning] 证据边界
> RSPO 优于 DPO/GRPO 的证据来自快手广告场景的单篇工业论文，增益可能部分来自奖励噪声、行业结构和冷启动状态；论文未做行业、账户规模、冷启动分层分析，不能外推为「推荐场景 RL 一律选 list-wise」。[[OneMall：快手电商多场景统一生成式推荐]] 在电商场景仍使用 DPO/GRPO 处理排序奖励，说明两类方案在推荐场景并存。

## 关联

- [[DPO离线与GRPO在线：机制差异与生成式推荐中的选型]]：在线/离线机制约束与 DPO/GRPO 在生成式推荐中的选型分工
- [[GR4AD：广告生成式推荐的架构训练推理联合设计]]：RSPO 的提出场景与消融证据
- [[OneMall：快手电商多场景统一生成式推荐]]：电商场景使用 DPO/GRPO 的对照
- [[PROMISE：过程奖励模型解锁推荐推理时扩展]]：推荐场景的另一种奖励建模路线
