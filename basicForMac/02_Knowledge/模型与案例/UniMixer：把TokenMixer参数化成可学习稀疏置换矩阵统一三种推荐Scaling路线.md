---
创建日期: 2026-10-20
更新日期: 2026-10-20
类型: 笔记
标签:
  - 推荐系统
  - 广告技术
来源:
  - "[[2026Kuaishou_UniMixer.pdf]]"
  - "[UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems (arXiv 2604.00590)](https://arxiv.org/abs/2604.00590)"
  - "[论文分享｜推荐系统的“Attention Moment”要来了？快手 UniMixer 重新思考推荐 Scaling Law（微信公众号）](https://mp.weixin.qq.com/s/-vLy8aZsU1S-XVFMIqSqZA)"
状态: 待复核
证据强度: 快手工业预印本（2026-04-01 提交，未见正式 venue，无代码），全文含附录已读；实验全部在快手广告投放场景的用户留存任务（约 7 亿样本、40 GPU）上完成，无公开数据集对比；线上为多场景 A/B（CAD 口径），未给出分场景细节
待验证问题:
  - 单一任务泛化性：只在「次日留存」二分类上验证，未在 Criteo/Avazu 等公开数据集或 CTR/CVR 任务上交叉确认
  - FLOPs 代价：UniMixer-Lite-4-Blocks 84.5M 参数对应 4.24T FLOPs，约为 RankMixer（135.5M / 1.68T）的 2.5 倍，参数效率优势是否被计算成本抵消需按各业务算力预算判断
  - basis 数优于 rank 的结论在约 5M 小模型上得出，100M 规模是否保持未验证
  - 温度退火超参（τ 从 1.0 线性退火到 0.05、退火区间 J）与 cold-start 重训策略的敏感性只有消融方向、没有超参扫描
---

# UniMixer：把TokenMixer参数化成可学习稀疏置换矩阵统一三种推荐Scaling路线

## 筛选与速览

- **一句话速览**：UniMixer（快手）把 RankMixer 式的**规则化 TokenMixer 等价改写成置换矩阵乘法**，发现该置换矩阵可分解为 $G\otimes I$ 且满足双随机、稀疏、对称三条性质，于是把它参数化成**可学习的局部+全局混合矩阵**（UniMixing），并证明 heterogeneous attention 的 value 投影、TokenMixer、Wukong 的 FM 块都是它的特例——从而把推荐领域三条互不相干的 scaling 路线（attention 系 / TokenMixer 系 / FM 系）统一进一个理论框架，轻量版 UniMixer-Lite 在参数和 FLOPs 两条 scaling 曲线上都拿到最陡斜率。
- **核心贡献3句话**：① 揭示规则化 TokenMixer 的置换矩阵本质并参数化，解除其「头数=token 数（T=H）」的硬约束，得到可学习、可优化的 UniMixing 模块（计算管线优化后复杂度 $O(L^2)\to O(L^2/B+LB)$）；② 建立统一框架：局部混合模式 $\equiv$ attention 的 value 投影，全局混合模式 $G(X,W_G)$ 涵盖 softmax 相似度（attention）、输入无关置换（TokenMixer）与 $XX^T$ 内积（FM）；③ UniMixing-Lite 用 basis 复用 + 低秩近似进一步压缩，快手广告留存任务上以 38.2M 参数超过 RankMixer 135.5M 的 AUC（0.7523 vs 0.7493），线上多场景 CAD D1-D30 平均提升 15%+。
- **与我们系列的关系**：本知识库此前的卡片都在「生成式推荐 / 混排拍卖」这条线上（生成范式、机制设计、计费），UniMixer 属于正交的另一条线——**判别式排序骨干的 scaling**。它回答的是「排序模型做大时算力花在哪里最划算」；而结论节明确说该模块可延伸到用户行为序列建模与生成式推荐任务，即生成式推荐（OneRec/HSTU 一系）的骨干设计也可以借用这套「可学习稀疏混合」的思路。

## 论文信息

- 作者/机构：Mingming Ha、Guanchen Wang、Linxun Chen、Xuan Rao、Yuexin Shi、Tianbao Ma、Zhaojie Liu、Yunqian Fan、Zilong Lu、Yanan Niu、Han Li、Kun Gai（快手）。
- 发表：arXiv 2604.00590（2026-04-01 v1，04-02 v2），cs.IR；预印本，无开源代码。
- 方法谱系：MLP-Mixer（CVI，2021）→ RankMixer（CIKM '25，非参数 token 混合）→ TokenMixer-Large（13B 规模化）→ UniMixer（参数化 + 统一理论）；FM 一侧：FM → DeepFM/DCN → Wukong（ICML '24）→ Kunlun。

## 核心问题

推荐模型的特征空间天然异质（不同 field 的 embedding 语义空间不同），不能直接搬 NLP 的 Transformer。当前三条 scaling 路线各有硬伤：

1. **Attention 系**（HiFormer、FAT、HHFT）：用 token 级独立的 Q/K/V 投影处理异质性，但内积相似度在随机初始化下被输入量值主导，attention 权重**尖锐且稀疏、对角占优**——梯度回传困难、部分 token 的训练可能停滞；大规模异质特征下 attention 分数又可能变小而失去区分度。
2. **TokenMixer 系**（RankMixer、TokenMixer-Large）：规则化、无参数的 reshape-split 混合绕开了内积相似度，但**混合模式不可学习**、缺乏场景适应性，且硬性要求头数等于 token 数（T=H），限制了可用的交互模式。
3. **FM 系**（Wukong、Kunlun）：显式低阶交互，scaled up 后性能提升受限，与 LLM 式的幂律 scaling 不符。

论文的核心问题：能否构造一个**统一的 scaling 模块**，同时吸收三条路线的优点，并给出它们之间的理论桥梁？

## 方法要点

### 第一步：TokenMixer 的等价参数化

TokenMixer 把每个 token 均分为 H 个 head，再按 head 重排后做混合。论文发现这整个操作等价于一个**大置换矩阵**乘展平输入：

$$
\mathrm{TokenMixer}(X)=\mathrm{reshape}\left(W_{\text{perm}}\,\mathrm{flatten}(X)\right)
$$

其中 $W_{\text{perm}}\in\mathbb{R}^{TD\times TD}$。附录 A 给出 2×6 输入的数值示例：混合后的张量展平后恰是 $\mathrm{flatten}(X)$ 经一个 12×12 的 0/1 置换矩阵变换的结果。这个置换矩阵有四条性质：

- **可压缩**：$W_{\text{perm}}=G\otimes I$，即两个小矩阵的 Kronecker 积（$I$ 为 $\frac{D}{T}\times\frac{D}{T}$ 单位阵，$G$ 为 $T^2\times T^2$）；
- **双随机**：每行、每列元素和均为 1；
- **稀疏**：每行/列恰有一个非零元；
- **对称**：T=H 时对称，T≠H 时不对称。

直接参数化 $W_{\text{perm}}$ 需 $O(T^2D^2)$ 参数与中间变量，不可行；参数化 $G$ 与 $I$ 后参数量降到 $O(T^4+(D/T)^2)$。

### 第二步：UniMixing——局部 + 全局可学习混合

进一步抛弃 T、D 的记法，改用**块（block）**刻画：输入 embedding 维度 $L$，块大小 $B$，块数 $L/\!/B$。每行赋予独立参数 $W_B^i$（块间交互模式各不相同），得到：

$$
\mathrm{UniMixing}(X)=\mathrm{reshape}\left(\left(W_G\otimes\{W_B^i\}_{i=1}^{L/\!/B}\right)\,\mathrm{flatten}(X),\,1,\,L\right)
$$

其中 $\otimes$ 为广义 Kronecker 积。直乘实现仍会构造 $[TD,TD]$ 中间变量，论文把计算管线改写为两步：

$$
\mathrm{reshape}\left(H,\tfrac{L}{B},B\right)=\mathrm{reshape}\left(\left[x_1W_B^1\;\middle|\;\cdots\;\middle|\;x_{L/B}W_B^{L/B}\right],\tfrac{L}{B},B\right)
$$

$$
\mathrm{UniMixing}(X)=\mathrm{reshape}\left(W_G\,\mathrm{reshape}\left(H,\tfrac{L}{B},B\right),1,L\right)
$$

复杂度从 $O(L^2)$ 降到 $O(L^2/B+LB)$，且全程无大中间变量。$W_B^i$ 控制**块内**交互模式，$W_G$ 控制**块间**交互模式；T=H 约束随之消失。

**三条性质的落地**：对称性用 $(W+W^T)/2$；双随机与稀疏用 Sinkhorn-Knopp 迭代（先取指数保证非负，再交替做行、列归一化），并用温度系数 $\tau$ 控制稀疏程度：

$$
\tilde{W}_G=\frac{W_G+W_G^T}{2},\qquad \bar{W}_G=\mathrm{Sinkhorn\text{-}Knopp}\left(\frac{\tilde{W}_G}{\tau}\right)
$$

### 第三步：统一视角——三条路线是一个公式的特例

对比 heterogeneous attention 的 $V^h$ 与上式的 $\mathrm{reshape}(H,L/B,B)$：若块数取 $T$ 且 $W_B^i=W_V^i$，局部混合**就是** attention 的 value 投影；而 $W_G$ 的角色等同于 attention 权重矩阵，只是额外要求双随机、稀疏、对称。Wukong 的 FM 块 $X_I(X_I)^TY$ 则是 attention 在 $W_Q=W_K=I$、value 不依赖输入（$V^h=W_V=Y$）时的退化形式。统一公式：

$$
\mathrm{UniMixing}(X)=\mathrm{reshape}\left(\underbrace{G(X,W_G)}_{\text{全局混合模式}}\cdot\begin{bmatrix}x_1W_B^1\\ \vdots\\ x_{L/B}W_B^{L/B}\end{bmatrix},1,L\right)
$$

| 方法 | 局部混合模式 | 全局混合模式 $G(X,W_G)$ |
| --- | --- | --- |
| Self-Attention | $XW^V$ | $\mathrm{softmax}\big((XW^Q)(XW^K)^T/\sqrt{d}\big)$ |
| Heterogeneous Attention | $X\tilde{W}^V$ | $\mathrm{softmax}\big((X\tilde{W}^Q)(X\tilde{W}^K)^T/\sqrt{d}\big)$ |
| TokenMixer | $X$ | $G$（输入无关的置换矩阵） |
| FM（Wukong） | $X_I(X_I)^T$ 型内积 | 退化 attention（$W^Q=W^K=I$、V 固定） |

三条路线的差异被归结为：**全局混合模式用什么函数度量 token 间交互强度**（softmax 内积 / 固定置换 / 显式内积）。

### UniMixing-Lite：basis 复用 + 低秩近似

块粒度越细，$W_B^i$ 数量越多、$W_G$ 越大，出现冗余。Lite 版做两处压缩：

- **局部**：只学 $b$ 个 basis 矩阵 $\{Z_\ell\}$，每个块的权重是 basis 的加权组合 $W_B^{*i}=\mathrm{Sinkhorn\text{-}Knopp}(\sum_\ell \omega_\ell^i Z_\ell)$；
- **全局**：$W_G$ 低秩化为 $W_r=\mathrm{Sinkhorn\text{-}Knopp}(A_G B_G)$，$A_G\in\mathbb{R}^{(L/\!/B)\times r}$，$B_G\in\mathbb{R}^{r\times(L/\!/B)}$。

可视化（Fig. 5）显示低秩 + basis 重构后的矩阵经 Sinkhorn-Knopp 仍接近满秩；低温 $\tau=0.05$ 下混合矩阵分布比 $\tau=1$ 明显更尖锐。

### UniMixer Block 与训练策略

- **Per-token SwiGLU**：每个 token 独立的 up/gate/down 投影（式 19），建模 token 间特征异质性；
- **SiameseNorm**：双流耦合（$\bar{X}$ 主流 + $\bar{Y}$ 残差流，逐层 $O_\ell=\mathrm{UniMixer}(\bar{X}_\ell+\mathrm{RMSNorm}(\bar{Y}_\ell))$，两流各自更新，末端融合），化解深层网络 Pre-Norm 与 Post-Norm 的张力——这是 RankMixer 加深即退化、而 UniMixer 深度可持续 scaling 的关键结构原因；
- **温度退火**：$\tau_j=\max\{\tau_{\text{start}}-(\tau_{\text{start}}-\tau_{\text{end}})j/J,\ \tau_{\text{end}}\}$，从 1.0 线性退火到 0.05。低温带来稀疏权重（性能显著更好，见消融），但梯度稀疏弱、训练不稳；数据不足时改为**两段式**：高温训好后以高温权重为初始化，再用低温（0.05）重训。

## 实验结果

**设置**：快手广告投放场景真实日志，任务为用户次留二分类（User Retention），约 7 亿用户样本、一年窗口、数百异质特征；40 GPU 混合分布式训练；指标 AUC / UAUC（用户级 AUC），效率看 dense 参数量、FLOPs/batch 与 MFU。

**与 SOTA 对比（约 100M 参数档，Table 2）**：

| 模型 | 参数量 | FLOPs/Batch | AUC | ΔAUC |
| --- | --- | --- | --- | --- |
| Heterogeneous Attention（base） | 132.7M | 1.68T | 0.744577 | – |
| Wukong | 107.1M | 1.40T | 0.744477 | -0.0100% |
| FAT | 138.4M | 1.83T | 0.744883 | +0.0306% |
| TokenMixer-Large | 103.3M | 1.27T | 0.748410 | +0.3833% |
| RankMixer | 135.5M | 1.68T | 0.749329 | +0.4752% |
| UniMixer-2-Blocks | 101.5M | 2.50T | 0.750238 | +0.5661% |
| UniMixer-Lite-4-Blocks | 38.2M | 1.26T | 0.752327 | +0.7750% |
| UniMixer-Lite-4-Blocks | 84.5M | 4.24T | 0.752718 | +0.8141% |

**Scaling law 拟合**（$\Delta AUC=c\cdot \text{Params}^{\alpha}$）：RankMixer 指数 0.116、UniMixer 0.132、UniMixer-Lite 0.142——Lite 同时拿到最大系数与最大指数，scaling 效率最强；FLOPs 维度结论一致。

**消融（6.57M 小模型，Table 3）**：去掉温度系数伤害最大（AUC -0.1645%），其次 w/o model warm-up（-0.0856%）、w/o 对称约束（-0.0573%）、w/o 块级独立局部权重（-0.0436%）、SiameseNorm 换 Post-Norm（-0.0273%）——**稀疏性是性能的第一来源**，任何一条矩阵约束被破坏都掉点。

**Lite 内部（Table 4）**：增加 basis 数 b（2→4→8，AUC +0.1055%）比增加低秩 rank r（2→256，+0.0971%）参数效率更高；**深度 scaling**：UniMixer-Lite 2→4→8 blocks 持续提升（0.749228→0.750803→0.750875），而 RankMixer 2→4 blocks 反而退化（0.747772→0.746706）——印证深度比宽度更高效，且 SiameseNorm 是深度扩展的前提。

**线上 A/B**：部署在快手多个广告投放场景，以 30 天累计活跃天数（CAD，不含安装日 D0）衡量，D1-D30 平均提升 15%+。

## 批判性分析

- **创新层次**：属于「结构创新 + 训练策略创新」的组合——置换矩阵等价改写是漂亮的理论观察，但参数化本身是直接的；真正的工程增量在于计算管线重排（避免 $[TD,TD]$ 中间变量）、三条矩阵约束的可微实现（Sinkhorn-Knopp + 对称化 + 温度）和 SiameseNorm 深度扩展。
- **实验口径偏窄**：全部实验是单一公司单一任务（拉新次留），没有公开数据集；与 RankMixer 的对比在同一数据同超参下进行，公平性尚可，但「UAUC/AUC 提升 0.3-0.8%」的绝对量级在无置信区间的情况下难以评估显著性。
- **FLOPs 不占优**：Table 2 中 UniMixer-Lite 在同参数量下 FLOPs 明显高于 RankMixer（84.5M/4.24T vs 135.5M/1.68T）。参数 scaling 曲线最陡，不等于同等算力预算下最优——论文用 FLOPs scaling 曲线（同样最陡）回应了这一点，但对预算受限的业务，「每 GPU 时买多少 AUC」才是决策量。
- **统一框架的解释力 vs 预测力**：Table 1 的统一视角主要价值是**解释**（为什么三条路线都能 scale、差异在哪里），据此「预测」出的新结构（UniMixer）确实有效，但框架没有回答「什么任务该选哪种全局混合模式」——softmax 相似度与固定置换各自的适用边界仍是经验问题。
- **博客与论文的一致性**：公众号解读的叙事（三条路线批判、核心洞察、Lite 设计、实验数字、四个深度思考）与原文逐项对得上，数值（0.749329/0.750238/0.752327、CAD +15%、深度 2→4→8）均可在原文 Table 2/4 与 5.5 节复核，未发现夸大或失实。

## 落地可行性评估

- **数据需求**：需要大规模多域异质特征日志（论文用 7 亿样本），中小规模数据下论文自己承认线性退火可能探索不足，需改用两段式 cold-start 重训——即数据量小时训练流程更复杂。
- **计算开销**：训练 40 GPU；推理侧复杂度 $O(L^2/B+LB)$ 与块大小 B 强相关，B 是核心调参对象（论文用 B=6、L=768）。无开源代码，需从零实现 Sinkhorn-Knopp 约束与两流 SiameseNorm。
- **实现复杂度**：中偏高。模块本身是标准矩阵运算（无自定义 attention 算子），但稀疏约束、温度调度、双流 norm 的数值稳定性都需要工程打磨；相比直接复用 RankMixer，改造收益（消融显示约束缺一不可）与风险并存。
- **综合评分**：⭐⭐⭐⭐——结构与训练策略均可适配现有排序模型（替换 MLP-Mixer 式骨干即可），收益数字来自真实广告场景且有线上验证，但无代码、单任务验证、FLOPs 代价需要按业务算力预算自行测算。
- **对生成式推荐的潜在迁移**：论文结论节明确说 UniMixer 可延伸到用户行为序列建模与生成式推荐任务。对本知识库关注的生成式路线（OneRec/HSTU 骨干、EGA-V2 的 encoder-decoder），「用可学习稀疏混合替代部分 attention 层」是控制生成式骨干推理成本的一个候选方向，但生成式任务的自回归解码是否保留同样的 scaling 行为完全未验证。

## 关联

- 内部：本篇是知识库中第一条「排序骨干 scaling」线的卡片，与混排/拍卖线（[[DNA：可微分排序让拍卖机制进入端到端学习]]、[[EGA-V2：把召回排序创意分配计费统一进单个生成模型的全链路广告框架]]）正交互补——前者决定「打分器多大、多贵」，后者决定「打分器输出如何变成机制」。EGA-V2 基线 MCA 使用的 Tiger/HSTU 骨干正是 attention 系 scaling 的代表。
- 外部：RankMixer（CIKM '25）、TokenMixer-Large（arXiv 2602.06563）、Wukong（ICML '24）、Kunlun（arXiv 2602.10016）、HiFormer/FAT/HHFT（attention 系）；SiameseNorm（arXiv，2026）为深度扩展的独立来源。
