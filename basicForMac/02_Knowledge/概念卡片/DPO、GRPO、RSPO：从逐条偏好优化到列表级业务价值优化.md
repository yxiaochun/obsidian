---
创建日期: 2026-09-29
更新日期: 2026-09-29
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源:
  - "[[02_Knowledge/模型与案例/GR4AD：广告生成式推荐的架构训练推理联合设计.md]]"
  - "[01_Sources/论文报告/2026Kuaishou_GR4AD.pdf]"
---

# DPO、GRPO、RSPO：从逐条偏好优化到列表级业务价值优化

三者解决同一个问题：SFT 只会模仿日志分布，不能直接优化下游目标（人类偏好、奖励、业务价值）。区别在「优化什么目标、怎么构造对比信号、要不要采样和参考模型」。DPO 是 LLM 对齐的离线 pairwise 方案，GRPO 是可验证奖励下的 on-policy RL，RSPO 是快手 GR4AD 为广告场景定制的 list-wise RL。

## 原理

### DPO（Direct Preference Optimization）

场景：LLM 对齐，有 (chosen, rejected) 成对偏好标注。RLHF 的 KL 约束奖励最大化目标有解析解——最优策略与参考模型的 log-ratio 隐含奖励 $r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，代回目标得闭式损失：
$$\mathcal{L}_{\text{DPO}}=-\mathbb{E}\left[\log\sigma\left(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$
不需要奖励模型、不需要在线采样，参考模型全程参与。本质是离线的 pairwise 对比学习。

### GRPO（Group Relative Policy Optimization）

场景：有明确标量奖励（DeepSeek 用于数学、代码等可验证任务）。去掉 PPO 的价值网络（critic），改用**组内基线**：同一 prompt 用当前策略采样 G 个输出，各自拿奖励 $r_i$，优势用组内标准化：
$$A_i=\frac{r_i-\text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$$
再用 PPO 式裁剪比率更新，外加对参考模型的逐 token KL 惩罚。核心直觉：不需要知道「满分是多少」，只需要知道同一题的几种解法里谁比兄弟们强。代价是必须 on-policy 采样，且数据用完即弃。

### RSPO（Ranking-Guided Softmax Preference Optimization）

场景：广告生成式推荐。两个特殊约束：优化目标是**按 eCPM 排列的列表**，per-item 奖励不足以刻画 list-wise 目标；训练样本来自**多条异构生产管线**，很多没有可靠参考分布。

用 Lambda/NDCG 框架把「列表按 eCPM 排序的 NDCG 损失」写成 softmax 对比损失的上界（GR4AD 式 16）：
$$\mathcal{L}_{\text{RSPO}}=-\mathbb{E}\left[\log_2\sigma\left(-\log\sum_{y_j\in E_i}M_{ij}\exp\left(\beta\log\tfrac{p_\theta(y_j|X)}{p_{\text{ref}}(y_j|X)}\right)-\beta\log\tfrac{p_\theta(y_i|X)}{p_{\text{ref}}(y_i|X)}\right)\right]$$
其中 $E_i=\{y_j: v_j<v_i\}$ 是奖励（eCPM）低于 $y_i$ 的候选集合，$M_{ij}$ 是标准 Lambda 权重（DCG 位置折扣 × 奖励差）。Ranking-Guided 指它是 NDCG 损失的上界；Softmax 指直接对候选集合做对比，不需要启发式构造 chosen–rejected 对。

参考模型用二值门控 $C_{ij}$ 控制（式 18）：只有参考分布可用、且当前模型与它的 log-ratio 偏差小于阈值 δ 时才启用；否则整块退化为无参考目标，避免过期或不可靠的 $p_{\text{ref}}$ 引入噪声正则。

## 对比

| | DPO | GRPO | RSPO |
|---|---|---|---|
| 范式 | 离线，闭式损失 | on-policy RL | 在线学习流中的 list-wise RL |
| 对比信号 | chosen–rejected 对（人工/启发式） | 组内奖励标准化 | 列表内按 eCPM 排序的候选集 $E_i$ |
| 奖励类型 | 隐式（偏好二值） | 外部标量奖励（稀疏、outcome 型） | 连续业务价值（eCPM，稠密） |
| 优化粒度 | point-wise（单条响应） | point-wise（单条输出） | list-wise（Lambda 权重编码位置折扣） |
| 参考模型 | 全程必须 | KL 惩罚项 | 门控启用（可用且偏差 < δ 才用） |
| 采样需求 | 无需在线采样 | 每请求采 G 个输出 | 不强求 on-policy；接受多管线日志列表 |
| 理论锚点 | RLHF 目标解析解 | PPO 去 critic | NDCG 损失的上界 |
| 典型场景 | LLM 对齐 | 可验证奖励的推理任务 | 广告/推荐的业务价值排序 |

## GRPO 数值例子

设 G=4、β=0.04、裁剪阈值 ε=0.2。对 prompt「计算 17 × 23」，当前模型采样 4 个输出，规则验证器打分 $r=[1,1,0,0]$：

1. 组内基线：mean = 0.5，std = 0.5；
2. 优势 $A_i=(r_i-\text{mean})/\text{std}=[+1,+1,-1,-1]$，答对的两个被强化，答错的被抑制；
3. 每条输出算新旧策略概率比 $\rho_i$，用 $\min(\rho_i A_i,\ \text{clip}(\rho_i,1-\epsilon,1+\epsilon)\cdot A_i)$ 更新：好输出概率最多推高到 1+ε，坏输出压过界后不再拉回；
4. 奖励上叠加 $-\beta\cdot\text{KL}(\pi_\theta\|\pi_{\text{ref}})$，防止策略跑飞；
5. 数据用完即弃，回到第 1 步继续采样——这是 on-policy 的主要成本。

换成推荐语境（输出 = 候选 SID 列表、奖励 = eCPM）时，GRPO 的短板就暴露：每秒几十万请求的 serving 场景在线采 G 个列表成本过高；逐条 advantage 丢失列表内位置信息；KL 参考分布经常不可用或过期。

## 为什么 GR4AD 要新造 RSPO

GR4AD 消融（在 VSL 基础上，相对 DLRM 基线的广告收入增益）：VSL + DPO 为 +3.16%，VSL + GRPO 为 +3.21%，VSL + RSPO 为 +3.86%。RSPO 的优势来自三点组合：

1. **奖励形态匹配**：eCPM 是连续价值信号，DPO 的二值偏好和 GRPO 的稀疏 outcome reward 都浪费了这个信息。
2. **list-wise 目标匹配业务**：广告排序按列表 NDCG 评估，DPO/GRPO 逐条优化丢失位置折扣。
3. **参考模型门控**：生产日志大量样本无 $p_{\text{ref}}$ 或参考分布因在线学习过期，DPO/GRPO 强制使用会引入噪声，RSPO 用门控规避。

此外 GR4AD 没有把 RSPO 当独立阶段，而是与 VSL 统一在一条在线学习流：用「模型排序 vs 奖励排序的秩偏差」$A^{(i)}=|r_p-r_v|/(n-1)$ 动态分配样本权重——偏差大时加重 VSL（模仿用户兴趣），偏差小时加重 RSPO（推向高价值）。这是 DPO/GRPO 两阶段训练范式不具备的生产形态。

> [!warning] 证据边界
> RSPO 优于 DPO/GRPO 的证据来自快手广告场景的单篇工业论文，增益可能部分来自奖励噪声、行业结构和冷启动状态；论文未做行业、账户规模、冷启动分层分析，不能外推为「推荐场景 RL 一律选 list-wise」。[[OneMall：快手电商多场景统一生成式推荐]] 在电商场景仍使用 DPO/GRPO 处理排序奖励，说明两类方案在推荐场景并存。

## 关联

- [[GR4AD：广告生成式推荐的架构训练推理联合设计]]：RSPO 的提出场景与消融证据
- [[OneMall：快手电商多场景统一生成式推荐]]：电商场景使用 DPO/GRPO 的对照
- [[PROMISE：过程奖励模型解锁推荐推理时扩展]]：推荐场景的另一种奖励建模路线
