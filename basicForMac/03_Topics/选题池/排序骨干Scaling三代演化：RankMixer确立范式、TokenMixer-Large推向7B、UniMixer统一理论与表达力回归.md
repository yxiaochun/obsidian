---
创建日期: 2026-10-20
更新日期: 2026-10-20
类型: 选题
标签:
  - 推荐系统
来源:
  - "[[RankMixer：token混合让推荐模型MFU提升十倍]]"
  - "[[TokenMixer-Large：七十亿参数在线排序模型]]"
  - "[[UniMixer：把TokenMixer参数化成可学习稀疏置换矩阵统一三种推荐Scaling路线]]"
状态: 待复核
---

# 排序骨干Scaling三代演化：RankMixer确立范式、TokenMixer-Large推向7B、UniMixer统一理论与表达力回归

> [!info] 证据边界
> 本文是三张精读卡片的综合对比，所有数字均出自各原文（RankMixer CIKM '25、TokenMixer-Large arXiv 2602.06563、UniMixer arXiv 2604.00590）。需要注意：三代模型来自两家公司（字节、快手），**没有同数据同预算的跨代直接对比**——RankMixer 与 TokenMixer-Large 同为字节内部对比，UniMixer 的对比在快手留存任务上独立进行；跨代结论是推断性综合。

## 核心问题

推荐排序骨干的 scaling 有三条互不相干的路线（attention 系 / TokenMixer 系 / FM 系），RankMixer→TokenMixer-Large→UniMixer 恰好构成同一演化轴上的三代：**效率 → 规模 → 表达力回归**。本文梳理每一代解决了什么、带来什么、留下什么，供后续在美团场景选型排序骨干扩展方案时参考。

## 三代一览

| 维度 | RankMixer（CIKM '25，字节） | TokenMixer-Large（2026-02，字节 AML） | UniMixer（2026-04，快手） |
| --- | --- | --- | --- |
| 混合算子 | 固定置换，零参数 | 固定置换 + revert 语义对齐 | **可学习稀疏置换**（Sinkhorn-Knopp + 温度退火） |
| T=H 约束 | 硬约束 | H≠T 可行但需 reverting | **解除** |
| FFN | Per-token 两层 FFN（隐藏 kD，GELU） | Per-token **SwiGLU** | Per-token SwiGLU（沿用） |
| MoE | ReLU 路由 + 自适应 L1 + DTSI（**dense-train / sparse-infer**） | **sparse-train / sparse-infer** + shared expert + gate value scaling | 未用 MoE（走 Lite 化：basis 复用 + 低秩） |
| 深层扩展 | 无解（加深退化） | interval residual + auxiliary loss（补丁式） | **SiameseNorm**（结构性方案） |
| 主打指标 | MFU / 推理时延 | 绝对规模（7B）+ 线上 GMV | 参数效率 + scaling 指数 |
| 规模证据 | 15M→1B | 500M→15B | 5M→100M |
| 理论贡献 | 无 | 无 | 三路线统一框架 |
| 线上验证 | 抖音全量 | 抖音广告/直播电商/直播三场景 | 快手多广告场景 CAD +15% |

## 第一代 RankMixer：确立「无参数 token 混合」范式

**解决的问题**：特征空间异质，self-attention 的 Q·K 内积在异质 ID 空间里难学且昂贵（精度只差 0.03% 但 FLOPs +71.8%）。

**核心设计**：

1. Semantic tokenization：语义相近字段先拼接再投影成 T 个 token；
2. Multi-head Token Mixing：纯置换、零参数的跨 token 重排——交互发生在拼接里，学习交给 FFN；
3. Per-token FFN：每 token 独立两层 MLP，参数按特征子空间隔离；
4. ReLU 路由 MoE + 自适应 L1 + DTSI：训练稠密保证专家拿梯度，推理稀疏省算力；
5. 系统侧：MFU 4.47%→44.57%，fp16 量化，70× 参数扩展不增推理时延。

**留下的问题**：① 混合模式固定不可学习，H=T 是硬约束；② 加深就退化，深层训练无专门设计；③ MoE 只稀疏推理、训练仍稠密，训练成本没降；④ H≠T 时残差两端 token 语义错位。

## 第二代 TokenMixer-Large：纵向规模化，逐个修工程病

不换范式，把 RankMixer 推到 7B 并在线全量，每项升级对应一个具体瓶颈：

1. **Mixing & Reverting**：第二层把 H 个混合 token revert 回 T 个原始 token，残差两端语义对齐。SR/OTR/TSA 控制变量实验证明：光形状对齐不够（-0.13%），语义也对齐才有正收益（+0.13% vs 无对齐的 -0.20%）；
2. **Per-token SwiGLU**：FFN 升级为 up/gate/down 门控（消融：换回 Per-token FFN 掉 0.10%）；
3. **Sparse-Pertoken MoE**：从 DTSI 升级为训练推理都稀疏——先建稠密再切子专家，shared expert 常开（DeepSeekMoE 思想），gate value scaling 抵消 softmax 后梯度衰减（1:2 配 scaling=2），FC_down 小初始化（ReZero 思想）稳早期训练。1:2 稀疏下 2.3B 激活参数 = 4B 稠密效果，FLOPs 29.8T→15.1T；
4. **深层稳定性套件**：Pre-Norm RMSNorm（Post-Norm 精度略高但发散 NaN）+ interval residual（每 2-3 层低层直连高层）+ auxiliary loss（低层 logits 联合训练）；
5. **Global token** + 系统侧 FP8、Token Parallel（通信 4L→2L+1，通信计算重叠后吞吐 96.6%）；
6. 规模证据闭环：500M→4B→7B 持续正斜率，线上直播电商人均预支付 GMV +2.98%、广告 ADSS +2.0%、直播收入 +1.4%。

**留下的问题**：混合仍不可学习；深层问题靠 interval residual + aux loss「打补丁」，UniMixer 评价其 "does not address the root of the problem"；T=H 类约束依然存在。

## 第三代 UniMixer：换赛道——把混合本身变成可学习的

1. **理论洞察先行**：TokenMixer ≡ 置换矩阵 $W_{\text{perm}}\cdot\text{flatten}(X)$，且 $W_{\text{perm}}=G\otimes I$（可压缩、双随机、稀疏、对称四条性质）→ 参数化成 UniMixing：局部混合 $W_B^i$（块内，约等于 attention 的 V 投影）+ 全局混合 $W_G$（约等于 attention 权重矩阵），解除 T=H；计算管线重排 $O(L^2)\to O(L^2/B+LB)$，无大中间变量；
2. **三路线统一**：attention、TokenMixer、Wukong 的 FM 收进同一公式，差异只在「全局混合模式用什么函数度量 token 间交互强度」（softmax 内积 / 固定置换 / 显式内积）——前两代都没有的理论层贡献；
3. **稀疏约束可微化**：对称化 $(W+W^T)/2$ + Sinkhorn-Knopp 迭代 + 温度退火（1.0→0.05）。消融中去温度系数掉 0.1645%，是全部组件里最大的——稀疏性是这套设计的性能第一来源；
4. **深层问题从根源解决**：SiameseNorm 双流结构（主流 + 残差流耦合）替代前代的补丁组合——2→4→8 blocks 持续提升，而 RankMixer 加深直接退化；
5. **UniMixing-Lite 主打参数效率**：basis 复用（局部）+ 低秩近似（全局），38.2M 参数 AUC 0.7523 超过 RankMixer 135.5M 的 0.7493，scaling 指数 0.142 > 0.132 > 0.116。

**留下的问题**：只有快手留存单一任务、无公开数据集、无代码、FLOPs 偏高（84.5M 对应 4.24T，约为 RankMixer 同参数档的 2.5 倍）。

## 演化主线

- **RankMixer 用「零参数混合」换硬件效率**，代价是丢掉混合表达力；
- **TokenMixer-Large 在不碰混合算子的前提下治好工程病**（语义残差、深层梯度、MoE 双向稀疏、通信），换来规模上限；
- **UniMixer 把当初为效率扔掉的可学习性捡回来**——但不退回昂贵的 attention，而是用带稀疏约束的可学习置换，同时给出统一理论解释三条路线为何都能 scale。

**深层扩展的处理演进**最能体现三代差异：RankMixer 无解（加深退化）→ TokenMixer-Large 用 interval residual + aux loss 打补丁（症状缓解）→ UniMixer 用 SiameseNorm 从 norm 结构根源解决（深度 2→4→8 持续涨）。

**MoE 演进**：ReLU+L1 的 DTSI（训练稠密/推理稀疏）→ sparse-train/sparse-infer + shared expert（训练推理都稀疏）→ UniMixer 干脆放弃 MoE，把稀疏性做进混合矩阵本身。

**有意思的分叉**：后两代是字节和快手对同一前代的分叉回应——字节求规模上限（在线全量 7B），快手求理论统一和单位参数收益（100M 档打赢），两家都没有做对方做的事。

## 对美团场景的选型启示

- 若追求**推理成本敏感的排序骨干升级**（信息流/搜索精排），RankMixer/TokenMixer-Large 一系的高 MFU + 稀疏推理是直接可抄的工程路径，MoE 双向稀疏与 Token Parallel 是必配项；
- 若追求**有限算力下的最大离线收益**，UniMixer-Lite 的参数效率数据（38.2M 打 135.5M）更相关，但需自验 FLOPs 代价与留存之外任务（CTR/CVR）的迁移性；
- 三代都验证了「深度 scaling 需要专门的 norm/残差结构」——直接堆 block 会退化，这是复现任一代时都要先解决的稳定项。

## 待验证问题

- 三代分属两家公司、不同任务（抖音 CTCVR vs 快手留存），「UniMixer 参数效率优于 TokenMixer-Large」只在各自数据上成立，同预算跨代对比缺失；
- UniMixer 的可学习混合在 7B 级规模上是否仍保持 scaling 指数优势（其验证止步 100M）；
- TokenMixer-Large 的 sparse-train/sparse-infer 在快手式留存任务、UniMixer 的 SiameseNorm 在字节式 CTCVR 深层任务上是否互换有效。

## 关联

- [[RankMixer：token混合让推荐模型MFU提升十倍]]｜[[TokenMixer-Large：七十亿参数在线排序模型]]｜[[UniMixer：把TokenMixer参数化成可学习稀疏置换矩阵统一三种推荐Scaling路线]]：三代精读卡片。
- [[Scaling Law在工业推荐系统的落地路径]]：排序骨干 scaling 所处的整体技术版图。
- 本选题与混排线（[[广告混排方案对比：外部性建模、生成范式与计费机制的路线分化]]）正交：本线决定打分骨干的容量与成本，混排线决定骨干输出如何进入机制。
