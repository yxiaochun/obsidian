---
创建日期: 2026-09-15
更新日期: 2026-09-27
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2026ByteDance_OneTrans.pdf]]"
状态: 待复核
证据强度: 单篇工业论文证据
待验证问题:
  - "极端长序列中非尾部行为能否保留，更大模型如何满足线上效率约束？"
  - "统一因果架构能否在候选感知序列和外部场景中获得一致收益？"
  - "作者所称双向信息交换在多大程度上受 S 先 NS 后的因果顺序限制？"
---

# OneTrans：一个Transformer统一特征交互与序列建模

> [!warning] 证据边界
> 论文数据与代码未公开，卡片状态保持待复核；生产 A/B 和 scaling 结论不宜直接迁移到其他业务或实现。

## 论文信息

- 作者/机构：Zhaoqi Zhang、Haolei Pei、Jun Guo、Tianyu Wang、Yufei Feng、Hui Sun、Shaowei Liu、Aixin Sun；字节跳动与南洋理工大学。
- 期刊/会议/年份：WWW 2026；arXiv:2510.26104v3，最后更新 2026-02-03。
- 领域：工业推荐排序、特征交互、长序列建模、推荐模型 scaling。
- 外部记录：[OneTrans arXiv](https://arxiv.org/abs/2510.26104)。
- 与我研究的关联：OneTrans 不是生成式推荐模型，而是 DLRM-style ranking。但它把行为序列与异质特征放进同一 causal Transformer，并复用 KV caching、FlashAttention 和混合精度，直接对应「生成式推荐」关心的统一骨干、长上下文、scaling 与推理成本问题。

## 一句话创新点

OneTrans 用一个统一 tokenizer、一个 mixed-parameter causal Transformer stack 同时做行为序列建模和特征交互，并让同质的 S-tokens 共享参数、异质的 NS-tokens 使用专用参数，从而在工业排序场景获得可扩展性与线上效率。

## 模型结构

![[OneTrans：一个Transformer统一特征交互与序列建模｜模型结构图.png]]

图左侧展示统一输入路径：多条用户行为序列和用户/候选/上下文等非序列特征分别 token 化后，拼成一条以 S-tokens 在前、NS-tokens 在后的序列；中间的 OneTrans Pyramid Stack 逐层收缩尾部 S-token query，最终交给 Task Tower。右侧展示 block 细节：RMSNorm pre-norm 后先做 Mixed Causal Attention，再做 Mixed FFN；S-tokens 共享 Q/K/V 和 FFN 参数，NS-tokens 使用 token-specific 参数。

## 七问笔记

| 问题 | 回答 |
|------|------|
| 研究背景 | 工业推荐排序长期分两条路：LONGER 等扩展用户行为序列，Wukong、RankMixer 等扩展稠密特征交互。两者通常先压缩序列，再与静态特征 late fusion，导致双向信息交换不足、模块执行割裂，也难以整体扩容。 |
| 研究目的 | 用单一 Transformer 骨干替代 encode-then-interaction，让多行为序列、用户画像、候选 item 和上下文在统一 token 图中联合建模，同时满足工业排序的参数/FLOPs/时延约束。 |
| 创新点 | 见上文一句话创新点；工程侧的核心配套是 Pyramid Stack 与跨请求 KV caching。 |
| 研究方法 | 先把数值和类别特征嵌入为 NS-tokens，把 item ID、类别、价格等行为侧信息投影为 S-tokens；再用 RMSNorm pre-norm、Mixed MHA、Mixed FFN 组成 causal block。Pyramid 每层只让尾部最近 S-tokens 发 query，但 K/V 仍覆盖全序列；跨请求缓存复用 append-only 用户序列的 S-side KV。 |
| 实验数据 | 离线使用内部工业日志：29.1B impressions、27.9M users、10.2M items；日均 118.2M 曝光和 2.3M DAU。任务是 CTR 与 CVR，指标为 AUC 与 impression-weighted UAUC，按时间切分并用 next-batch 评估。线上在 Feeds 与 Mall 做 user/account hash 分流 A/B。 |
| 结果结论 | OneTrans S 相对 DCNv2+DIN 提升 CTR AUC/UAUC +1.13%/+1.77%，CVR AUC/UAUC +0.90%/+1.66%；OneTrans L 分别提升 +1.53%/+2.79% 和 +1.14%/+3.23%。与训练 FLOPs 相近的 RankMixer+Transformer 相比也有优势。Feeds 线上 click/user +7.737%、GMV/user +5.685%，Mall 线上 click/user +5.143%、GMV/user +3.670%，且 p99 时延分别下降 3.91% 和 3.26%。 |
| 总体评价 | 论文提供了离线对比、设计消融、系统效率、scaling 曲线和生产 A/B，证据链比多数单点离线实验更完整。但数据与代码未公开，绝对值和增益只在字节内部场景可验证；更大的模型规模仍受 p99 约束，Pyramid 也依赖尾部信息集中假设。 |

## 方法精读

### 统一 tokenization

- 非序列特征包含价格、CTR 等数值特征和用户 ID、item 类别等类别特征。Group-wise Tokenizer 先人工分组，再对每组用独立 MLP；Auto-Split Tokenizer 先拼接全部特征，过一次 MLP 后拆成 $L_{NS}$ 个 token。论文默认 Auto-Split，离线效果更好，也减少 kernel launch 开销。
- 多行为序列中的每个事件由 item ID 与类别、价格等 side information 拼接而成；不同行为序列先通过共享的序列级 MLP 投影到统一维度 $d$，再合并。默认使用 timestamp-aware fusion 按时间交错，并带 sequence-type indicator；timestamp-agnostic fusion 则按事件 intent 顺序拼接，并在不同序列之间插入可学习 `[SEP]`。

### Mixed causal attention

- 初始序列写成 $X^{(0)}=[\text{S-tokens};\text{NS-tokens}]$。每层先执行 $Z^{(n)}=\text{MixedMHA}(\text{Norm}(X^{(n-1)}))+X^{(n-1)}$，再执行 $X^{(n)}=\text{MixedFFN}(\text{Norm}(Z^{(n)}))+Z^{(n)}$。
- Attention 使用标准 causal mask：前面的 S-token 只能看到更早的 S-token；排在后面的每个 NS-token 可以聚合全部 S-history，也能看到前面的 NS-token。因此 NS-side 相当于把 target attention、特征交互和序列聚合放进同一张注意力表。
- Mixed parameterization 不改变 MHA 的计算形式，只改变 Q/K/V 投影和 FFN 的参数组织：所有 S-tokens 共享一套 $W_Q/W_K/W_V$ 和 FFN；每个 NS-token 各自拥有 $W_{Q,i}/W_{K,i}/W_{V,i}$ 和 FFN。RMSNorm pre-norm 用于稳定异质 token 的数值尺度。

#### RMSNorm pre-norm 的具体计算

RMSNorm 对每个 token 的 $d$ 维 hidden vector 独立归一化。设一个 token 表示为 $x=[x_1,\dots,x_d]$，先计算均方根：

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
$$

再按维度缩放：

$$
\mathrm{RMSNorm}(x)_i=\frac{x_i}{\mathrm{RMS}(x)}\gamma_i
$$

其中 $\gamma_i$ 是可学习 scale 参数。与 LayerNorm 不同，RMSNorm 不减均值、通常不加 shift/bias，因此计算更轻。例如 $x=[2,4,6,8]$ 时，$\mathrm{RMS}(x)=\sqrt{30}\approx5.477$，归一化结果约为 $[0.365,0.730,1.095,1.461]$，最后再乘 $\gamma$。

“pre-norm”指把 RMSNorm 放在子层输入之前，而不是放在残差相加之后。因此 OneTrans block 的执行顺序是：

$$
Z^{(n)}=\mathrm{MixedMHA}(\mathrm{RMSNorm}(X^{(n-1)}))+X^{(n-1)}
$$

$$
X^{(n)}=\mathrm{MixedFFN}(\mathrm{RMSNorm}(Z^{(n)}))+Z^{(n)}
$$

伪代码可以写成：

```python
def rms_norm(x, gamma, eps=1e-6):
    rms = sqrt(mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * gamma

def one_trans_block(x):
    h = rms_norm(x)
    h = mixed_causal_attention(h)
    x = x + h

    h = rms_norm(x)
    h = mixed_ffn(h)
    x = x + h

    return x
```

这样做的一般作用是：S-tokens 和 NS-tokens 来自不同特征空间，数值尺度可能差异较大；pre-norm 先把每类 token 的 hidden state 拉到较稳定的尺度，再进入 Mixed Attention/Mixed FFN。同时，pre-norm 让残差主通路保持原始 $x$，有利于深层堆叠时的梯度回传。这里的作用解释是一般机制推断，论文正文主要将其作为 block 设计和数值稳定性配套，并未单独给出 RMSNorm pre-norm 的消融。

### Pyramid stack

- Pyramid 不是在输入端硬截断历史，而是在每层只让最近的 $L'$ 个 S-tokens 发出 query；K/V 仍覆盖全序列。层间线性缩减 query 数量，OneTrans S 从 1190 到 12，OneTrans L 从 1500 到 16，每层取最接近 32 的倍数，顶层对齐 NS-token 数。
- 注意力成本变为 $O(LL'd)$，FFN 随 $L'$ 线性收缩。设计假设是 causal Transformer 会把历史信息逐步蒸馏到尾部和 NS-tokens；Table 3 显示去掉 Pyramid 后 CVR UAUC 下降 0.42%，而 TFLOPs 从 2.64T 增至 8.08T。

### KV caching 与 LLM 优化

- Stage I 对同一请求共享的 S-side 序列做 causal 建模并缓存 key/value；Stage II 对每个候选计算各自的 NS-tokens，并与缓存 S-side KV 交互。由于候选专用序列如 SIM 不能直接复用公共历史，会先 pooling 成 NS-tokens。
- 用户行为序列是 append-only 的，因此 KV caching 可跨请求复用：新请求只计算自上次以来新增行为的 $\Delta L$ 个 token，把请求内序列计算从 $O(L)$ 降到 $O(\Delta L)$。
- 系统侧再叠加 FlashAttention-2、BF16/FP16 混合精度训练、activation recomputation 和半精度推理。这些优化不改变模型公式，但让统一 Transformer 在工业 p99 预算内部署。

## 批判性分析

- Why 回答：作者要解决的痛点是两条扩展路线没有共同计算图，序列信息只能先压缩再交互。用统一 causal Transformer 后，NS-tokens 可以像 target attention 一样读取全部 S history，模型也能按深度、宽度和序列长度整体扩展。作者选择 causal attention 不是为了语言建模，而是为了把历史信息集中到尾部，从而启用 KV cache 和 Pyramid。
- 为什么这个方法而不沿用 InterFormer：InterFormer 用 summary-based bidirectional cross 连接两个模块，但仍保留独立模块和交叉架构。OneTrans 更激进，把 tokenization、注意力、FFN 和输出状态放进一个 stack，代价是必须为异质 token 专门设计 mixed parameterization。
- 为什么这样设计实验：先以内部生产链路 DCNv2+DIN 为锚点，再比较 Wukong、HiFormer、RankMixer、StackDIN、LONGER 和 RankMixer+Transformer；随后用 Table 3 拆开 tokenizer、序列融合、参数共享、attention type 和 Pyramid，用 Table 4 拆开系统优化。这个顺序能把「统一是否有效」和「哪一项工程件带来收益」分开。
- 遗漏对照实验：论文没有公开数据，缺少跨平台和跨业务场景验证；没有比较同一统一骨干下的其他 tokenization/剪枝方案；没有报告 KV cache 失效、序列漂移、缓存窗口和候选特定序列预聚合的精度损失。Pyramid 只在 OneTrans S 的完整消融中测试，OneTrans L 的泛化主要靠部署结果间接支持。
- 换位思考：如果重写论文，我会更早给出 S-tokens 与 NS-tokens 的参数量、FLOPs 和显存分解，并解释 token-specific NS 参数在数百特征下的成本；如果重做实验，我会补一个 candidate-aware sequence baseline、多个候选规模下的 cache 收益曲线，以及 fixed p99 下长度、深度、宽度的 Pareto 搜索。
- 优点：基线强且来自生产链路；同时报告 AUC、UAUC、FLOPs、MFU、训练/推理显存、p99 和业务 KPI；系统消融直接连接到训练与线上约束。
- 不足：标题层面的「bidirectional information exchange」与 token 顺序存在张力。S-tokens 在前、NS-tokens 在后时，S-token 状态不会读取候选 item 或其他 NS 特征，主要靠后面的 NS-tokens 聚合序列证据；这有利于缓存，但不是候选特征与历史 token 在每一层完全对称地互写（这一段是我的判断，论文未以该措辞展开）。CVR UAUC 样本较小，作者自己也提示波动更高；在线结果没有给出完整置信区间表。

## 关键图表解读

- **Figure 1**：对比传统两段式流水线与 OneTrans。传统方案先压缩序列，再拼接非序列特征并交给交互模块；OneTrans 让两类特征进入同一 OneTrans stack。
- **Figure 2**：系统架构。序列特征和非序列特征分别 token 化后拼接；OneTrans block 是 RMSNorm pre-norm causal Transformer，Mixed Causal Attention 和 Mixed FFN 让 S-tokens 共享 Q/K/V 与 FFN 参数，NS-tokens 每个都有专用参数。
- **Table 1/2**：离线数据与主结果。生产锚点是 10M dense 参数、0.06 TFLOPs 的 DCNv2+DIN；OneTrans S 为 91M、2.64 TFLOPs，OneTrans L 为 330M、8.62 TFLOPs。作者认为 +0.1% AUC/UAUC 有意义，+0.3% 通常对应线上显著效果。
- **Table 3**：输入与 block 消融。Auto-Split 比 Group-wise 好；有时间戳时 timestamp-aware 融合比 intent ordering 好；NS token-specific 参数优于全部共享；full attention 精度与 causal attention 接近，但会失去 KV cache 的工程收益；去掉 Pyramid 后 CVR UAUC -0.42%，TFLOPs 从 2.64T 升到 8.08T。
- **Table 4**：系统消融。相对未优化 OneTrans S，Pyramid 使训练运行时间 -28.7%；跨请求 KV caching 使训练运行时间 -30.2%、推理 p99 -29.6%；FlashAttention 使训练运行时间 -50.1%；混合精度加重计算使推理 p99 -69.1%。
- **Figure 3a**：长度、深度、宽度的 FLOPs-收益曲线。增加长度带来最大收益；深度通常比单纯加宽更有效，但深度增加串行时延，宽度更利于并行。
- **Figure 3b**：联合扩展时，OneTrans 与 RankMixer 都近似 log-linear，但 OneTrans 斜率更陡。作者解释 RankMixer-centric scaling 缺少统一骨干，其 MoE 扩容主要加宽 FFN。
- **Table 5/6**：效率与线上 A/B。OneTrans L 的 p99 为 13.2 ms，略低于 DCNv2+DIN 的 13.6 ms；Feeds 和 Mall 的核心业务指标全部为正，同时端到端 p99 也下降。

## 值得追踪的引用

- [ ] [LONGER](https://arxiv.org/abs/2505.04421)：长序列 causal Transformer 与 cross-request KV caching 的前序来源。
- [ ] [RankMixer](https://arxiv.org/abs/2507.15551)：本文的强基线和 scaling 对照，也可检查 token-specific FFN 扩容路线。
- [ ] [Wukong](https://arxiv.org/abs/2403.02545)：特征交互 scaling law 的另一路线。
- [ ] [HiFormer](https://arxiv.org/abs/2311.05884)：mixed parameterization 的思想来源，可比较异质特征组的参数划分。
- [ ] [InterFormer](https://arxiv.org/abs/2411.09852)：明确讨论序列与特征交互双向连接的对照方法。
- [ ] [HSTU](https://arxiv.org/abs/2402.17152)：生成式推荐骨干路线；作者称其与依赖丰富非序列特征的 DLRM 路线互补。

## 术语与句式积累

- S-tokens：来自多行为序列的 token；NS-tokens：来自用户、候选 item、上下文等非序列特征的 token。
- Mixed parameterization：同质 S-tokens 共享一套 Q/K/V 与 FFN，异质 NS-tokens 使用 token-specific 参数。
- Pyramid Stack：随层递减的尾部 query 集合；K/V 仍覆盖全序列，用于把长历史蒸馏到尾部和 NS-tokens。
- Cross-request KV caching：同请求候选共享 S-side KV；用户序列 append-only，跨请求只计算新增行为。
- 可复用句式：不要把「统一 Transformer」只理解为参数合并，工业可行性的关键在于统一图是否暴露了可缓存、可剪枝的计算结构。

## 复现清单

- 数据：论文使用字节内部匿名化生产日志，未公开。离线集有 29.1B 样本、27.9M users、10.2M items；按时间切分，特征在 impression 时间快照，标签按生产窗口聚合。
- 代码：论文未提供仓库。
- 环境：训练使用 16 张 H100 数据并行；训练 per-GPU batch size 2048，推理 per-GPU batch size 100；OneTrans S 为 6 层、宽度 256、4 heads，OneTrans L 为 8 层、宽度 384。
- 优化器：sparse embeddings 用 Adagrad（beta1=0.1、beta2=1.0），dense 参数用 RMSProp（lr=0.005、alpha=0.99999、momentum=0），无 weight decay；dense/sparse 梯度截断阈值分别为 90 和 120。
- Pyramid schedule：线性减少序列 query token 数，OneTrans S 从 1190 到 12，OneTrans L 从 1500 到 16；每层取最接近 32 倍数的数量，顶层对齐 NS token 数。
- 系统组件：FlashAttention-2、BF16/FP16 混合精度、activation recomputation、half-precision inference、cross-request KV caching。
- 复现缺口：没有特征 schema、embedding 表规模、行为序列定义与最大长度、缓存生命周期/失效策略、candidate-specific 序列预聚合细节；线上 A/B 的完整置信区间和观察期也未完整披露。
- 改良设想：在 fixed p99 预算下搜索长度、深度、宽度与 Pyramid 终点的 Pareto；比较尾部 query 与周期性 global query 的精度/成本；测试 S-side KV 在跨天、跨设备和序列修正场景下的稳定性；为非尾部历史增加显式检索或池化，而不是只依赖信息向尾部集中。

## 关联

- [[MixFormer：稠密特征与序列建模协同扩展]]：同属统一稠密特征与序列建模路线，但用 user-item 解耦支持请求级复用。
- [[UGSep：用户侧计算复用降低大模型推理成本]]：同属用户侧计算复用与推理成本控制思路。
- [[RankMixer：token混合让推荐模型MFU提升十倍]]：OneTrans 的强基线与 scaling 对照。
- [[生成式推荐：用万亿参数序列转换器 HSTU 替代 DLRM，验证推荐领域的缩放定律]]：用生成式/序列转导路线对照本文的 DLRM-style 统一排序。
