---
创建日期: 2026-10-31
更新日期: 2026-11-04
类型: 选题
标签:
  - 机制设计
  - 广告技术
  - 生成式推荐
来源:
  - "[[NGA：非自回归生成式拍卖用全局外部性兼顾效果与推理效率]]"
  - "[[EGA-V2：把召回排序创意分配计费统一进单个生成模型的全链路广告框架]]"
  - "[[外卖列表广告生成式整体拍卖一期：用非自回归生成与RegretNet计费突破阶乘组合空间]]"
  - "[[MIRT：用maximal-in-range结构换取精确DSIC的whole-feed生成式拍卖]]"
  - "[[美团到餐gDNA一期实验：把自然候选当作零出价广告的整体拍卖混排]]"
---

# 非自回归混排生成器Loss设计的四条技术路线

## 切入点

五篇混排/拍卖方案的生成器训练共享同一个骨架：评估器（或 listwise 模型）打分 → 策略梯度回传给生成器。但**奖励如何组装、按 item 还是按位置分配信用、用不用组基线**，各家给出了四种不同的答案——这个选择直接决定梯度方差大小、多目标权衡的插入位置、以及激励相容信号能否渗透进生成器。

## 四条路线总览

| 路线        | 代表方案       | 奖励结构                                                        | 信用分配粒度  | 基线/方差控制                                     |
| --------- | ---------- | ----------------------------------------------------------- | ------- | ------------------------------------------- |
| VCG 式边际贡献 | NGA、EGA-V2 | $r_{y_i}=R_{\mathbf{Y}}-R_{\mathbf{Y}^{-i}}$                | item 级  | 无显式基线，边际差分天然消掉请求间共同项                        |
| 位置级折扣回报   | 外卖列表一期     | $Reward_t=\sum_k \gamma^k \delta_{t+k}$                     | 位置级     | 折扣因子 γ + 手调惩罚项；单序列采样方差大                     |
| POMO 组基线  | MIRT       | best-of-K 期望福利 $r(R;C)=\mathbb{E}_{B}[\max_{x\in R}V(x,B)]$ | range 级 | advantage = 本 range 奖励 − 其余 G−1 个 range 的均值 |
| 直接负奖励     | gDNA 一期    | $Loss=-Reward$（沿 DNA 可微排序）                                  | 序列级标量   | 无基线；allocate 本身可微，不走 score-function PG      |

## 路线一：VCG 式边际贡献（NGA / EGA-V2）

**公式**：奖励为每个 item 的边际福利贡献——

$$
r_{y_i}=R_{\mathbf{Y}}-R_{\mathbf{Y}^{-i}},\qquad L_G=-\sum_i r_{y_i}\log Z_{y_i}
$$

其中 $R_{\mathbf{Y}}$ 是含 $y_i$ 的最优序列总奖励，$R_{\mathbf{Y}^{-i}}$ 是去掉 $y_i$ 后**最优替代序列**的总奖励。这就是 VCG transfer 的定义式，用作策略梯度奖励相当于「按真实福利贡献分配信用」。

**两家的差异**：NGA 的 $R$ 含收入 + $\alpha\cdot$ 订单双目标；EGA-V2 只取收入侧 $\sum_j b_j\hat{r}^{pctr}_j$（体验项不在 PG 奖励里）。且两家都只把边际贡献用在 reward shaping 层，支付网络仍是 RegretNet 式遗憾最小化——VCG 结构在奖励与计费之间**未对齐**。

### 工程复杂度与近似

严格 VCG 的成本不在减法，而在 $R_{\mathbf{Y}^{-i}}$：它要求「去掉 $y_i$ 后重新求一条最优替代序列」，而不是只把 $y_i$ 从当前序列中删除后重打分。若入选 item 数为 $n$，严格定义需要做 $n$ 次反事实全局重排；在候选空间巨大且评估器含全局外部性时，这是主要延迟成本。因此，VCG 式边际贡献在策略梯度里通常只作为 reward shaping 信号，不等于生成器自动获得严格 DSIC。

工程上可以按「质量—延迟」梯度近似：

- **Beam 候选内过滤**：复用已打分的 beam $B=\{Y^{(1)},\dots,Y^{(B)}\}$，对每个 item 找不含它的最高分候选：$\hat{R}_{\mathbf{Y}^{-i}}=\max_{Y'\in B,\ i\notin Y'}R(Y')$。这把 $n$ 次全局重排降为一次 beam 打分加约 $O(nB)$ 过滤；但替代序列被限制在 beam 内，只是近似 VCG。
- **Mask 后重解码**：把 item $i$ mask 掉后让生成器重新生成不含它的序列。它更接近反事实重排，质量通常高于 beam 过滤，但每个 item 可能都要一次受限生成，延迟更高。
- **Leave-one-out**：直接计算 $R(\mathbf{Y})-R(\mathbf{Y}^{-i})$，其中 $\mathbf{Y}^{-i}$ 是从当前序列中删除 $i$ 后的轻量修复序列。计算最便宜，但没有考虑其他 item 能否顶上来，通常会高估边际贡献。
- **改用组基线**：若目标是稳定训练而非严格信用分配，可退回 sequence-level reward，配合 GRPO/POMO 组基线。计算更便宜，但生成器只知道整条 feed 的好坏，item 级信用变粗。

**实现细节与 NAR 红利**见 [[NGA：非自回归生成式拍卖用全局外部性兼顾效果与推理效率]]「边际贡献奖励的实现方式」节（beam 候选内过滤 vs 掩码重解码两条路）。

## 路线二：位置级折扣回报（外卖列表一期）

**公式**：

$$
Reward_t=\sum_{k=0}^{m}\gamma^k\delta_{t+k},\qquad \delta_t=imp_t\cdot isAd_t\cdot(bid_t\cdot listctr_t+cvrk\cdot listctr_t\cdot cvr_t)-k\cdot imp_t\cdot isAd_t
$$

$$
L=-\sum_{t=0}^{m}Reward_t\cdot Prob_t
$$

**特点与隐患**：多目标线性标量化发生在 reward 组装层而非评估器内；四个权重性质不同（bid 是内生价格、cvrk 是结算系数、k 与 γ 是手调旋钮）；$k\cdot imp$ 等价于**冻结的影子价格**，与「拉格朗日乘子应随约束松紧自适应」的主线结论冲突。单序列 PG 梯度方差大，文档已规划升级 GRPO（组内归一化同时消掉 bid 量纲的请求间差异）。

## 路线三：POMO 组基线（MIRT）

**公式**：生成器**出价无关**地生成 K 条候选 feed 的 range，奖励是 best-of-K 期望福利——每个上下文抽 D=8 组出价画像取 range 内最优福利。策略梯度用 POMO 基线技巧：

$$
A_g=r(R_g;C)-\frac{1}{G-1}\sum_{g'\neq g}r(R_{g'};C)
$$

每上下文生成 G=8 个 range 构基线，各 rollout 强制以不同初始 item 开始。

**与 GRPO 组的区别**：POMO 的「组」是**跨 range、不同初始 item 的确定性 rollout**；GRPO 的组是**同策略的 K 条随机采样**。前者基线更稳定（rollout 多样性由构造保证），后者实现更通用。

### GRPO loss 的设计与原理

GRPO 可以看作「无 critic 的组内相对策略梯度」。对同一上下文 $C$，旧策略采样 $G$ 条完整输出 $Y_1,\dots,Y_G$，分别得到标量奖励 $R_1,\dots,R_G$。GRPO 不训练 value model，而是直接用组内均值和标准差构造 advantage：

$$
\bar R=\frac{1}{G}\sum_{g=1}^G R_g,\qquad
\sigma=\sqrt{\frac{1}{G}\sum_{g=1}^G (R_g-\bar R)^2},\qquad
\hat A_g=\frac{R_g-\bar R}{\sigma+\epsilon_{\text{norm}}}
$$

其中组均值近似 baseline $b(C)$，消掉同一请求下 reward 的共同水平；除以 $\sigma$ 把不同请求、不同 reward 量纲拉到相近尺度，避免高价值请求的梯度淹没低价值请求。若 $R_g>\bar R$，这条 range 被加强；反之被削弱。

自回归形式下，token 级重要性比率为：

$$
\rho_{g,t}(\theta)=
\frac{\pi_\theta(y_{g,t}\mid C,Y_{g,<t})}
{\pi_{\theta_{\text{old}}}(y_{g,t}\mid C,Y_{g,<t})}
$$

完整目标通常写成 PPO 式 clipped objective 加 KL 正则：

$$
J_{\text{GRPO}}(\theta)=
\mathbb{E}
\left[
\frac{1}{G}\sum_{g=1}^G
\frac{1}{T_g}\sum_{t=1}^{T_g}
\left(
\min
\left(
\rho_{g,t}\hat A_g,
\text{clip}(\rho_{g,t},1-\epsilon_{\text{clip}},1+\epsilon_{\text{clip}})\hat A_g
\right)
-
\beta
D_{\text{KL}}
\left(
\pi_\theta\Vert\pi_{\text{ref}}
\right)
\right)
\right]
$$

训练时最小化 $-J_{\text{GRPO}}$。clip 限制单次更新中策略概率相对旧策略的偏移；KL 项约束当前策略不远离参考策略，缓解 reward hacking 和组内样本坍缩。

在 NAR 混排或 range/feed 级简化形式中，若暂时忽略 clip 和 KL，policy gradient 可写成：

$$
L_{\text{GRPO}}(\theta)
=
-\frac{1}{G}\sum_{g=1}^G
\hat A_g\log\pi_\theta(Y_g\mid C)
$$

若生成器按位置输出 $Z(\cdot)$，且 $Y_g=(y_{g,1},\dots,y_{g,m})$，则：

$$
\log\pi_\theta(Y_g\mid C)=\sum_{p=1}^m\log Z_{y_{g,p}},\qquad
L_{\text{GRPO}}(\theta)
=
-\frac{1}{G}\sum_{g=1}^G
\hat A_g\sum_{p=1}^m\log Z_{y_{g,p}}
$$

这说明标准 GRPO 的 advantage 是**整条 range/feed 共享的**：它回答「这条 range 相对同组其他 range 好多少」，不估计「去掉某个 item 后总福利会掉多少」。因此它比 VCG 式边际贡献便宜、稳定，但不做 item 级反事实归因，也不能直接推出 DSIC/VCG payment。

| 维度 | MIRT/POMO | 常见 GRPO |
| --- | --- | --- |
| 组内样本 | 强制不同初始 item 的确定性 rollout | 同策略随机采样 |
| baseline | leave-one-out：$R_g-\frac{1}{G-1}\sum_{g'\neq g}R_{g'}$ | 整组均值，再除以标准差 |
| 多样性来源 | 起点构造 | 策略随机性/温度采样 |
| 主要风险 | 起点分布可能偏离真实策略 | 策略变尖锐后组内样本趋同 |
| 信用粒度 | range 级 | 通常 sequence/range 级 |

## 路线四：直接负奖励（gDNA 一期）

**公式**：

$$
Loss=-Reward,\qquad Reward=\frac{1}{n}\sum_i\sum_j allocate_{ij}\cdot target_{ij}
$$

**关键差异**：gDNA 沿 DNA 可微排序路线，allocate 矩阵本身可微，loss 直接取负奖励端到端反传——**不需要 score-function 型策略梯度**，也没有采样方差问题。代价是分配被耦合进可微排序的近似误差；一期退化后 reward 只剩广告收益 + 固定 penaltyScore（自然体验保护单薄）。

## 横向观察

1. **信用分配粒度 vs 实现代价的权衡**：item 级边际贡献信用分配最细，但每个 item 需要一次「最优替代」近似；位置级折扣回报是标准 RL 折中；序列级标量（gDNA）实现最简但学习信号最粗——不过它靠可微性换回了梯度质量。
2. **多目标的插入位置不同**：NGA/EGA-V2 把体验项放进评估器奖励 R 内部；外卖列表/gDNA 在 reward 组装层做线性标量化，且两者的 adload/自然保护都是「冻结影子价格」（k·imp、penaltyScore），这是五个方案共同的机制设计短板。
3. **共同的下一站是组内归一化**：外卖列表已规划 GRPO——既降方差，又消掉 reward 中 bid 量纲导致的请求间梯度失衡（大出价请求淹没小出价请求）。
4. **IC 信号都不直接进生成器 loss**：所有方案的激励相容都由计费层（RegretNet 式支付网络）近似承担，生成器只对平台目标负责；MIRT 是唯一把 DSIC 放进结构保证（MIR + VCG）的例外，但其代价是生成器必须出价盲。

## 开放问题

- 边际贡献的「最优替代序列」工程近似（beam 候选内过滤）与论文理想定义的误差如何量化？
- 候选空间巨大的工业混排场景下，POMO/GRPO 的组大小 K 取多大才能兼顾方差与延迟？
- 是否存在把 IC regret 信号直接写进生成器 reward 的做法（当前所有方案都推迟到计费层，生成器对策略性出价无感知）？

## 关联

- [[NGA：非自回归生成式拍卖用全局外部性兼顾效果与推理效率]]：路线一 + 边际贡献实现细节。
- [[EGA-V2：把召回排序创意分配计费统一进单个生成模型的全链路广告框架]]：路线一的收入侧变体。
- [[外卖列表广告生成式整体拍卖一期：用非自回归生成与RegretNet计费突破阶乘组合空间]]：路线二 + GRPO 升级规划。
- [[MIRT：用maximal-in-range结构换取精确DSIC的whole-feed生成式拍卖]]：路线三 + 结构保证 IC 的例外路线。
- [[美团到餐gDNA一期实验：把自然候选当作零出价广告的整体拍卖混排]]：路线四（可微排序直接负奖励）。
- [[广告混排方案对比：外部性建模、生成范式与计费机制的路线分化]]：混排路线定位图。
