---
创建日期: 2026-09-23
更新日期: 2026-09-23
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[Wukong Towards a scaling law for large-scale recommendation]]"
状态: 待复核
证据强度: 单篇论文证据
---

# 堆叠因子分解机实现推荐系统缩放定律：Wukong 在 100+ GFLOPexample 下持续提升质量

## 论文信息

- **论文标题**：Wukong: Towards a Scaling Law for Large-Scale Recommendation
- **一句话结论**：Wukong 通过堆叠因子分解机（FMB+LCB）的密集缩放策略，在六个公开数据集上取得最优 AUC，并在 146B 内部数据集上跨两个数量级复杂度（超过 100 GFLOP/example）保持缩放定律，而基线模型出现饱和或训练不稳定。
- **发现来源**：2025Meituan_MTGR
- **历史被引底稿论文数**：18

## 贡献

- 提出 Wukong 架构，完全基于堆叠的因子分解机（Factorization Machines），通过 FMB 和 LCB 的并行组合，使每层交互阶数指数增长（第 i 层可捕获 1 到 2^i 阶交互）。
- 提出协同的密集缩放策略（dense scaling），通过增加层数 l、FMB 输出嵌入数 nF、LCB 输出嵌入数 nL、压缩维度 k 和 MLP 大小来扩展模型，而非仅扩大稀疏嵌入表。
- 在六个公开数据集（Frappe、MicroVideo、MovieLens Latest、KuaiVideo、TaobaoAds、Criteo Terabyte）上，Wukong 在所有数据集上取得最优 AUC。
- 在 146B 样本、720 特征的内部数据集上，Wukong 在模型复杂度跨两个数量级（超过 100 GFLOP/example）时保持缩放定律，相对 LogLoss 持续改善，而基线模型（AFN+、DLRM、FinalMLP 等）出现饱和或训练不稳定。
- 通过消融实验验证 FMB、LCB 和残差连接各自的重要性，并展示缩放各超参数（l、nF、nL、k、MLP）对质量的贡献。

## 方法要点

- Wukong 由嵌入层、交互堆叠（Interaction Stack）和最终 MLP 组成。嵌入层将稀疏和稠密特征映射为统一维度 d 的嵌入，输出张量 X0 ∈ R^{n×d}。
- 交互堆叠由 l 个相同的 Wukong 层组成，每层包含并行的因子分解机块（FMB）和线性压缩块（LCB），输出拼接后经残差连接和层归一化：X_{i+1} = LN(concat(FMB_i(X_i), LCB_i(X_i)) + X_i)。
- FMB 包含一个 FM 和一个 MLP：FM 计算输入嵌入的两两交互，输出 2D 交互矩阵，经展平、层归一化和 MLP 转换为 (nF × d) 的向量，再重塑为 nF 个嵌入。
- LCB 执行线性重组：LCB(X_i) = W_L X_i，其中 W_L ∈ R^{nL×ni}，不增加交互阶数，确保第 i 层捕获 1 到 2^i 阶交互。
- 优化 FM：利用低秩性质，将点积交互矩阵 XX^T 乘以可学习投影矩阵 Y（形状 n×k），通过结合律先计算 X^T Y，将计算复杂度从 O(n^2 d) 降至 O(n k d)，k << n。投影矩阵 Y 可通过 MLP 对线性压缩输入进行注意力化。
- 复杂度分析：总时间复杂度约为 O(n d h log n + h^2)，其中 h 为 MLP 中最大全连接层大小。
- 缩放超参数：l（层数）、nF（FMB 输出嵌入数）、nL（LCB 输出嵌入数）、k（压缩嵌入数）、MLP（FMB 中 MLP 的层数和 FC 大小）。缩放时优先增加 l 以捕获更高阶交互，再扩大其他超参数。
- 实现细节：使用 Neo 和 NeuroShard 进行列分片嵌入表，FSDP 用于稠密部分，自动算子融合，FP16/BF16 量化（嵌入表 FP16，前向 FP16，反向 BF16）。

## 实验与证据

- 公开数据集评估：在 Frappe（0.29M 样本，10 特征）、MicroVideo（1.7M 样本，7 特征）、MovieLens Latest（2M 样本，3 特征）、KuaiVideo（13M 样本，8 特征）、TaobaoAds（26M 样本，21 特征）、Criteo Terabyte（4B 样本，39 特征）上，Wukong 在所有六个数据集上取得最优 AUC。例如：Frappe AUC 0.9868（与 FinalMLP 并列最高），MicroVideo AUC 0.7292（基线最高 MaskNet 0.7255），MovieLens Latest AUC 0.9723（与 FinalMLP 并列最高），KuaiVideo AUC 0.7414（基线最高 MaskNet 0.7376），TaobaoAds AUC 0.6488（基线最高 DCNv2 0.6457），Criteo Terabyte AUC 0.8106（基线最高 MaskNet 0.8100）。
- 公开数据集评估条件：使用 BARS 基准的预处理，五个较小数据集使用 BARS 评估框架，直接使用最佳搜索配置或默认超参数，并测试默认嵌入维度和 128 维嵌入，报告较优结果。Criteo 数据集进行单遍训练，使用近 3000 次网格搜索。
- 内部数据集评估：146B 样本，720 特征，两个任务（Task1 预测点击兴趣，Task2 预测转化）。基线包括 AFN+、AutoInt+、DLRM、DCNv2、FinalMLP、MaskNet（xDeepFM 因内存问题未报告）。训练使用 Adam（lr=0.04，beta1=0.9，beta2=1）用于稠密部分，Rowwise Adagrad（lr=0.04）用于稀疏嵌入表，嵌入维度固定 160，全局 batch size 262,144，在 128 或 256 张 H100 GPU 上运行。
- 内部数据集结果（Task1）：Wukong 在所有复杂度级别上优于所有基线，LogLoss 改善超过 0.2%。Wukong 在模型复杂度跨两个数量级（超过 100 GFLOP/example）时保持缩放定律，大约每四倍复杂度提升 0.1% 的 LogLoss 改善。基线中 AFN+、DLRM、FinalMLP 在特定复杂度后达到平台期，AutoInt+、DCNv2、MaskNet 未能进一步提升质量（AutoInt+ 和 DCNv2 出现训练不稳定，MaskNet 内存不足）。DCNv2 作为最佳基线，需要 40 倍复杂度增加才能匹配 Wukong 的质量。
- 内部数据集结果（模型大小）：Wukong 在模型大小跨所有尺度上优于基线约 0.2%，并持续改善至超过 6370 亿参数（稀疏嵌入表固定为 6270 亿参数）。
- 消融实验（组件重要性）：在内部数据集上，将 FMB 输出置零导致大幅质量下降（相对 LogLoss 从约 0.03% 恶化至 1.84%）；单独停用 LCB 或残差连接仅导致轻微下降（约 0.08% 和 0.03%），但同时停用两者导致显著下降（1.84%）。
- 消融实验（缩放各组件）：从基础配置（k=96, nF=32, nL=32, MLP=3x8192）开始，逐步加倍各超参数。增加层数 l 带来显著质量提升；增加 MLP 大小也有明显提升；增加 k 和 nF 有益；nL 在基础配置下已饱和。联合缩放 k、nF、nL 比单独缩放效果更显著。
- 与 Transformer 基线对比：在内部数据集上，将 Wukong 的独特组件应用于 AutoInt+，使用 bit-wise MLP 替代 FFN 进行 V 投影改善 LogLoss 0.34%；在自注意力后添加 bit-wise MLP 改善 0.65%；结合两者及金字塔层形状（在首层输出使用 LCB）实现 0.57% 质量提升。相比缩放后的 AutoInt+，Wukong 实现 0.08% 质量提升，同时节省 90% FLOPs。
- 数据量缩放：在内部数据集上，Wukong 模型质量随训练数据量（单遍）持续改善至 146B 样本，更大模型具有更陡峭的改善趋势，且更数据高效。

## 局限与开放问题

- 由于巨大的计算需求，尚未达到 Wukong 缩放极限的复杂度水平，无法确定其可扩展性的确切上限。
- 对 Wukong 底层原理的理论理解不足，特别是与 Transformer 等共享堆叠点积结构的架构对比，需要进一步探索。
- Wukong 在推荐系统之外的泛化能力，特别是在涉及异构输入数据源的领域，尚未充分探索。
- 内部数据集评估中，数据集大小仍远不足以让大模型收敛，这是研究的局限之一。
- 实时服务高复杂度模型面临挑战，可能的解决方案包括训练多任务基础模型摊销成本，或将大模型知识蒸馏到小模型用于服务。

## 待验证问题

- Wukong 的确切缩放极限是什么？在何种复杂度下质量会饱和？
- 如何从理论上理解 Wukong 的优越性，特别是与 Transformer 架构的对比？
- Wukong 能否泛化到推荐系统之外的领域，尤其是涉及异构输入数据源的任务？
- 如何在实际在线服务中高效部署缩放后的 Wukong 模型？
- 在更大的数据集上，Wukong 的缩放定律是否仍然成立？需要多少数据才能让最大模型收敛？

## 与知识库的关联

- 该论文与知识库中关于大规模推荐系统缩放定律、特征交互架构（如 DLRM、DCNv2、AutoInt+、FinalMLP、MaskNet、xDeepFM）以及稀疏缩放与密集缩放的对比研究密切相关。Wukong 提出的堆叠因子分解机架构和密集缩放策略，为推荐系统领域建立类似 LLM 的缩放定律提供了新方向，与知识库中关于模型缩放、训练稳定性、分布式训练（FSDP、Neo、NeuroShard）等主题形成互补。

## 核对提醒

- 论文中公开数据集结果表（Table 2）的 LogLoss 数值存在异常：Wukong 在 Frappe 上 LogLoss 为 0.1757，而 DLRM 为 0.1465，FinalMLP 为 0.1280，但论文声称 Wukong 在所有数据集上取得最优 AUC，未强调 LogLoss 最优。内部数据集结果中，相对 LogLoss 的基线为 DLRM 基础配置，0.02% 改善被视为显著。缩放定律公式 y = -100 + 99.56 x^{0.00071} 为经验拟合，需谨慎解读。论文未提供公开数据集上 Wukong 的完整超参数配置，仅提供 Criteo 的搜索空间。内部数据集的具体特征和任务细节未完全披露，可能影响复现。
