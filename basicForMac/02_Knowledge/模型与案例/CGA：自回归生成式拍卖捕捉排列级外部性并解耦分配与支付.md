---
创建日期: 2026-10-18
更新日期: 2026-10-18
类型: 笔记
标签:
  - 机制设计
  - 推荐系统
来源: "[[2025Alibaba_CGA.pdf]]"
状态: 待复核
---

# CGA：自回归生成式拍卖捕捉排列级外部性并解耦分配与支付

## 论文信息

Contextual Generative Auction with Permutation-level Externalities for Online Advertising（KDD '25）。上海交通大学（Ruitao Zhu、Chufeng Shi、Zhenzhe Zheng、Fan Wu）与阿里巴巴（Yangsu Liu、Dagui Chen、Zhenjia Ma、Jie Zhang、Jian Xu、Bo Zheng）合作，10 位作者。已在淘宝广告系统完成在线 A/B 测试（2% 流量），论文表述 baseline DNA 为「fully deployed」，即部署环境是真实生产系统。

论文自述三项贡献：(i) 形式化带排列级外部性的 multi-slot auction，并推导最优 DSIC 机制结构；(ii) 用自回归生成模型打破「allocation-after-prediction」范式，基于最优机制结构解耦 allocation 与 payment，并将 IC 约束转成可反传的事后遗憾最小化问题；(iii) 在淘宝大规模工业数据和线上 A/B 中验证收入、CTR 与机制近似效果。

## 核心问题

GSP 依赖广告 CTR 相互独立的假设；DNA、SW-VCG 等学习型拍卖虽引入外部性，但受限于「allocation-after-prediction」范式——预测阶段对最终分配结果无感知，只能建模候选集合内部的 set-level externalities，无法感知最终序列的排列上下文（permutation-level externalities），导致分配次优。论文用淘宝工业数据（Figure 1）证明：permutation-aware CTR 对坑位**非单调**（广告换到更高坑位 CTR 可能更低），这使得「升位必然增 CTR」的常规单调性直觉失效，IC 约束下的最优机制设计变得非平凡。

三个具体挑战：(i) 现有范式预测时看不到序列上下文；(ii) VCG 式遍历全排列的计算复杂度为 P(n,k)=n!/(n−k)!，淘宝 n≈50、k≈5 时无法在线实时推理；(iii) 排列级外部性下坑位 CTR 非单调，单调分配条件不再显然成立。

## 方法要点

- **理论结果：最优 DSIC 拍卖结构**。将 Myerson 拍卖改写为排列级外部性形式（Definition 3）：分配规则最大化「ironed virtual value × permutation-aware CTR」的虚拟福利；支付为 p_i = b_i − ∫₀^{b_i} Θ_i(t,b_{−i})dt / Θ_i(b)。关键引理 Lemma 1 证明：只要分配规则最大化虚拟福利，广告主获得的 CTR 随其出价单调不减，Myerson's Lemma 的单调分配条件仍成立，由此该机制是 IC/IR 约束下的收入最优机制（Corollary 1）。这一结构把**分配与支付的优化解耦**：分配只需最大化虚拟福利，最优支付规则可以独立学习。
- **Lemma 1 的证明要点**：设 $A^*$ 是当前虚拟福利最大分配，$A'$ 是目标广告提价后的虚拟福利最大分配。附录 A.1 先用反证法说明提价后的 $A'$ 仍必须包含该广告；再用两个虚拟福利最优性不等式相减，得到
  $$
  (\tilde\phi(b_t,F_t)-\tilde\phi(b'_t,F_t))\Theta(x_t;A^*)\ge
  (\tilde\phi(b_t,F_t)-\tilde\phi(b'_t,F_t))\Theta(x_t;A').
  $$
  因为 $\tilde\phi$ 随出价单调不减，差值非正，所以提价后目标广告的 $\Theta(A')\ge\Theta(A^*)$。这里的「monotone allocation」指的是广告主提价后获得的 CTR 不减，而不是某个坑位的 CTR 必须随坑位序号单调上升；后者正是 Figure 1 中被工业数据否定的直觉。
- **Generator（自回归生成模块）**：输入候选广告集合与用户特征。set-level 编码器用无位置编码的 self-attention（式 4）+ sum pooling 产出置换不变的上下文向量 c；解码器为置换等变的 GRU 自回归结构（式 5），逐步生成 k 个坑位的广告序列，分配概率 z_t^i = softmax(MLP([s_t; h_i]) + e^w·φ̃_i)_i，其中 e^w 为可学习正系数，保证虚拟价值越高分配概率越大。训练时按概率采样探索，推理时取 argmax，只部署 Generator 在线推理。
- **Generator 的关键公式**：set-level 编码器为
  $$
  [h_1,\ldots,h_n]=\text{self-attention}([e_1,\ldots,e_n]),\quad
  c=\text{MLP}\left(\left[\sum_{i=1}^n h_i;h_u\right]\right).
  $$
  因为输入集合没有固定顺序，attention 不加位置编码，再用 sum pooling 保持置换不变。GRU 解码器在坑位 $t$ 更新
  $$
  s_t=\text{GRU}(s_{t-1},h_{A_{t-1}}),\quad
  z^t_i=\operatorname{softmax}\left(\left[\text{MLP}([s_t;h_i])+e^w\tilde\phi_i\right]_{i=1}^n\right)_i,
  $$
  已被前面坑位选中的广告会被 mask 掉。$e^w$ 是可学习正系数，把 Myerson 目标注入 softmax 的偏置项。
- **Evaluator（排列感知预测模块）**：输入分配嵌入 H_A、用户嵌入与上游预估阶段的 point-wise pCTR α_A，用 Bi-LSTM + 带正弦位置编码的多头 self-attention 编码整个序列，MLP 输出 (0,2)^k 的外部性校准向量 γ_A，最终 Θ_A = min(α_A ⊙ γ_A, 1)。论文强调 Evaluator 是可替换组件，任何能捕捉序列上下文的模型都可用。
- **Evaluator 的关键公式**：带正弦位置编码的 self-attention 先算
  $$
  H^s_A=\operatorname{softmax}\left(\frac{Q_AK_A^\top}{\sqrt d}\right)V_A,
  $$
  再由 Bi-LSTM 得到前向/反向状态 $H^f_A,H^b_A$，拼接用户嵌入后输出校准向量：
  $$
  \gamma_A=2\sigma(r(r([H^s_A;H^f_A;H^b_A;h_u]))).
  $$
  校准向量被限制在 $(0,2)^k$，可同时表达邻位广告的抑制和增强作用，最终 CTR 以 $\min(\alpha_A\odot\gamma_A,1)$ 截断到合法概率。
- **PaymentNet（支付网络）**：最优支付公式含积分项，直接蒙特卡洛近似需对每个样本反复调用 Generator 和 Evaluator，推理开销大。PaymentNet 输入分配嵌入、去自身出价向量 B⁻、期望价值画像 Z⊙Θ，经 sigmoid 输出支付率 p̃∈[0,1]，支付 p = p̃⊙b，IR 约束由结构保证。
- **PaymentNet 输入的精确含义**：$H_A\in\mathbb R^{k\times d}$ 是最终序列中各广告的编码；$B^-=[b^-_{A_1},\ldots,b^-_{A_k}]\in\mathbb R^{k\times(k-1)}$ 是每个赢家看到的其他竞争者出价；$Z\odot\Theta\in\mathbb R^{k\times1}$ 用 Generator 的分配概率 $Z$ 乘 Evaluator 的排列感知 CTR $\Theta$ 近似期望价值。支付率
  $$
  \tilde p=\sigma(r(r([H_A;B^-;Z\odot\Theta])))
  $$
  天然在 $[0,1]$，所以 $p=\tilde p\odot b$ 不需要额外投影即可满足 IR。
- **三阶段训练**：先用 listwise 点击数据以 BCE（式 6）训练 Evaluator 至收敛；冻结后用 policy gradient（式 9）训练 Generator，奖励分解为 self-reward（φ̃_i·Θ_i，式 7）与 external-reward（VCG 式边际贡献：去掉该广告后其余广告虚拟福利的损失，式 8）；最后冻结 G 与 Θ，用增广拉格朗日法（式 10）在「经验事后遗憾 rgt_i = 0」约束下最大化收入训练 PaymentNet，迭代更新网络参数与拉格朗日乘子 λ。
- **训练目标的完整形式**：Evaluator 的损失是对整条曝光序列每个坑位求 BCE。Generator 奖励为
  $$
  r_{A_i}=r^{\text{self}}_{A_i}+r^{\text{external}}_{A_i}
  =\sum_{j\in A}\tilde\phi_j\Theta_j(A)-\sum_{j\in A^{-i}}\tilde\phi_j\Theta_j(A^{-i}),
  $$
  即该广告在完整序列中的虚拟福利贡献，减去删除它后其余广告序列的虚拟福利。策略损失是
  $$
  L_G=-\frac1{|D|}\sum_s\sum_{i\in k}r^{A^s_i}\log z^{A^s_i}.
  $$
  PaymentNet 阶段固定 Generator 与 Evaluator，增广拉格朗日目标写作
  $$
  L_P=-\frac1{|D|}\sum_s\sum_i P_i(A^s)\Theta_i(A^s)
  -\sum_i\lambda_i\widehat{\text{rgt}}^s_i-\frac\rho2\sum_i(\widehat{\text{rgt}}^s_i)^2,
  $$
  论文交替执行：PaymentNet 参数对固定 $\lambda$ 最小化 $L_P$；随后 $\lambda^{new}=\lambda^{old}+\rho\,\widehat{\text{rgt}}(w_P^{new})$。这项把 IC 约束转成可反传的事后遗憾惩罚。

## 线上效果

- **在线 A/B**：淘宝生产系统，2023-08-19 至 2023-08-25，占全站 2% 流量，基线为已全量部署的 DNA。相对变化：RPM +3.2%，CTR +1.4%，GMV +6.4%，广告主 ROI +3.5%，单请求响应时间相对 +1.6%（平均增加 3 ms）。论文据此推断收入提升来自分配优化而非抬高扣费（第 5.4 节）。
- **离线实验**：淘宝 2024 年 1 月真实日志。训练集 1 月 20—23 日 50 万次拍卖、110.0875 万独立广告主；测试集 1 月 25 日 10 万次拍卖、45.2671 万独立广告主；每次拍卖约 30 个候选广告，k=3 坑位（第 5.1.1 节）。价值分布预设为均匀/指数两种并据此重生成出价（避免分布估计方法引入偏差）。
- **离线主结果（均匀分布，Table 1）**：CGA RPM 1264.01、CTR 0.04692、IC 指标 Ψ 2.1%。对比：GSP RPM −9.7%、CTR −7.1%、Ψ 14.7%；DNA RPM −5.4%、CTR −3.3%、Ψ 8.4%；SW-VCG RPM −5.8%、CTR −3.5%、Ψ 5.6%；EdgeNet RPM −4.6%、CTR −2.8%、Ψ 4.9%；遍历全排列的 Optimal RPM 1318.36（比 CGA 高 4.3%，即 CGA 达到约 95% 最优收入，Ψ 0）；VCG RPM −14.1% 但 CTR +2.1%（VCG 优化社会福利牺牲收入）。指数分布下结论一致：CGA RPM 1343.07、CTR 0.04490、Ψ 3.7%；GSP RPM −12.2%、CTR −9.8%、Ψ 18.3%；DNA RPM −6.4%、CTR −4.6%、Ψ 10.1%；SW-VCG RPM −7.1%、CTR −5.0%、Ψ 7.4%；EdgeNet RPM −5.7%、CTR −3.9%、Ψ 6.7%；VCG RPM −15.8% 但 CTR +2.5%；Optimal RPM 1415.60（比 CGA 高 5.4%）、CTR +2.2%。
- **IC 指标口径**：论文的 $\Psi$ 是相对事后遗憾，即每个广告的最大策略性收益除以真实效用后在测试集上的平均。测试时把每个广告的出价替换为 $\alpha b_i$，其中 $\alpha\in\{0.2,0.4,\ldots,2.0\}$，因此它考察的是十档双向扰动下的可操纵空间，不只是线性提价。
- **实现参数**：feature embedding 维度 8，所有 attention 层 4 头；$\gamma_A$ 与支付率 $\tilde p$ 的 MLP 隐藏层大小为 128 和 32；Adam 学习率 1e-3，batch size 512。学习型基线在 learning rate {1e-5,1e-4,1e-3,1e-2}、batch size {256,512,1024,2048}、hidden size {8,16,32,64} 中做网格搜索。
- **外部性建模消融（Table 2）**：CTR 预测任务上，point-wise Logloss 0.2311/AUC 0.7366，set-level 0.2286/0.7391，permutation-level Evaluator 0.2237/0.7476——预测精度随外部性粒度细化单调提升。
- **组件消融（均匀分布，Table 3）**：去掉 Evaluator（CGA-Θ）RPM −3.9%，说明自回归模型只看前序上下文、需 Evaluator 蒸馏完整序列知识；G-E 与 PaymentNet 端到端联训 RPM −6.5% 且 Ψ 恶化到 10.5%，解耦训练更优；去掉 self-reward 最差（RPM −11.2%，Ψ 18.9%）；去掉 external-reward RPM −4.2%；用出价替代虚拟价值（CGA-φ̃）RPM 仅降 2.6%，论文推断外部性建模部分弥补了价值分布信息缺失。

## 局限与开放问题

- **单参数 CPC 设定**：每广告主一个出价、按点击扣费，多参数（如多目标、按转化结算）场景未覆盖。
- **价值分布估计被绕开而非解决**：离线实验直接预设分布重生成出价；消融显示去掉虚拟价值仅小幅掉点，但真实平台上 ƒ_i(·) 只能从出价间接估计，偏差如何传导未量化。附录 A.1 已随本 PDF 给出 Lemma 1 证明；其前提包括 ironed virtual value 对出价单调不减，以及并列最优时可固定选择同一个 $A^*$。PDF 图表内容仍存在文本提取不可达的部分，例如 Figure 1 中各广告曲线的具体数值无法逐点引用。
- **非凸优化无收敛保证**：论文自述拉格朗日方法不保证全局最优，仅以实验证明近似效果。
- **外部性建模与探索的交互**：训练期采样、推理期 argmax 的切换策略意味着线上分配没有显式探索，长期流量分布漂移下的机制稳定性是开放问题（此条为本人推断，非论文自述）。
- **Future work**（论文自述）：将 contextual generative mechanism 扩展到不同渠道的异构 item 混排；自回归生成器不限于特定生成模型，可替换更先进的方案。

## 关联

- [[DNA：可微分排序让拍卖机制进入端到端学习]]：DNA 是 CGA 的直接基线与被超越对象，其「先预测后分配」正是 CGA 要打破的范式；线上 A/B 基线即已全量部署的 DNA。
- [[混排Re-Rank：影子出价把参与度约束转化为自然与广告流的统一排序货币]]：混排方向的行业起点，坑位 i.i.d. + 乘性 gap 修正是 CGA 排列级外部性建模之前的上一代技术栈。
- [[UniROM：广告排序的端到端统一生成架构]]：同属分配与支付解耦的生成式拍卖路线，UniROM 用支付网络满足 IC/IR，CGA 用事后遗憾最小化学习最优支付，可对照阅读。
- [[全站推广为什么排序计费分离仍需激励兼容]]：CGA 是「分配/计费分离 + 支付规则可学习」的正式理论化实例，其解耦结构直接对应排序层与计费层分离的架构。
- [[点评到餐NeuralAuction实验：用序特征实现计费调控并量化激励兼容代价]] 与 [[美团到餐多位次NeuralAuction实验：逐坑位竞拍兼容多模版灵活位次]]：NeuralAuction 系列同为多坑位学习型拍卖，CGA 的排列级外部性视角是对其 set-level 建模的推进。
