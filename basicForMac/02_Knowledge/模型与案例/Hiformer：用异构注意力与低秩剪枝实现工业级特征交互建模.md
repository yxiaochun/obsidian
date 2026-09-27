---
创建日期: 2026-09-24
更新日期: 2026-09-24
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[Hiformer Heterogeneous Feature Interactions Learning with Transformers for Recom]]"
状态: 待复核
证据强度: 单篇论文证据
---

# Hiformer：用异构注意力与低秩剪枝实现工业级特征交互建模

## 论文信息

- **论文标题**：Hiformer: Heterogeneous Feature Interactions Learning with Transformers for Recommender Systems
- **一句话结论**：针对 vanilla Transformer 在推荐系统中无法建模异构特征交互且推理延迟过高的问题，提出逐特征 QKV 投影的异构注意力层与进一步引入 Composite 投影的 Hiformer，并通过低秩近似与模型剪枝降低服务延迟，在 Google Play App 排序场景离线与在线 A/B 中均优于 AutoInt、DLRM、DCN 等基线。
- **发现来源**：2025Meituan_MTmixAtt
- **历史被引底稿论文数**：5

## 贡献

- 提出 Hiformer 模型，其核心是新颖的异构注意力层（heterogeneous attention layer），为每个特征学习独立的 Q/K/V 投影，以捕捉特征间复杂的协同效应，相比现有 vanilla Transformer 方法具有更强的模型表达能力。
- 利用低秩近似（low-rank approximation）与模型剪枝（model pruning）降低 Hiformer 的服务延迟，且不显著损害模型质量。
- 在 web-scale 数据集上进行了大量离线对比实验，证明捕捉异构特征交互的重要性，并表明 Hiformer 作为 Transformer 架构可以超越 SOTA 推荐模型。
- 通过在线 A/B 测试衡量不同模型对关键互动指标的影响，Hiformer 相比基线模型取得显著在线收益且延迟增加有限；作者称这是首次展示 Transformer 架构（Hiformer）在特征交互学习上超越 SOTA 推荐模型，并已成功部署为生产模型。

## 方法要点

- 问题定义：将推荐系统建模为对 (user, item) 特征 x 的监督学习，输入包含类别特征 xC 与稠密特征 xD，预测用户是否产生正向互动；并形式化定义了异构 z 阶特征交互（非加性映射函数 ρZ(·)）。
- 整体框架包含四层：Input Layer（类别特征、稠密特征、task embedding，task embedding 可视为 CLS token，支持多任务）、Preprocessing Layer、Feature Interaction Layer、Output Layer。
- 预处理层：类别特征通过每特征的投影矩阵 WC_i 做 embedding look-up 得到 e_i；稠密标量特征先 normalize 再 concat，经 MLP fD(·) 聚合并投影为 nD 个 embedding（nD ≪ |D|），以缩短输入 embedding 列表长度、降低二次复杂度；输出 embedding 列表长度 L = |C| + nD + t。
- 异构注意力层：将 vanilla 自注意力中所有特征共享投影矩阵的设计改为每个特征对 (i,j) 拥有唯一的语义相关性函数 φ^h_{i,j}(·,·)，实现特征语义感知与语义空间对齐；具体采用点积形式 φ^h_{i,j}(e_i,e_j) = e_i Q^h_i (e_j K^h_j)^T / sqrt(d_k)，其中 Q_i、K_j 为逐特征的 query/key 投影。
- 异构 FFN：为每个特征设计独立的 FFN，即 FFN_i^GELU(o_i) = GELU(o_i W^i_1 + b^i_1) W^i_2 + b^i_2，遵循 d_f = 4d 的设置。
- 参数与计算量：异构注意力层相比 vanilla Transformer 参数量增加且随输入 embedding 列表长度线性增长，但总 FLOPs 与同构注意力层相同（算子相同、仅参数化不同）。
- Hiformer：在异构注意力基础上进一步引入 Composite 投影提升表达能力，例如 key 投影改为 [k̂^h_1,...,k̂^h_L] = concat([e^h_1,...,e^h_L]) K̂^h，其中 K̂^h ∈ R^{Ld×Ld_k}；即先把特征 embedding 列表变换为 composite 特征作为 key/value，再在 composite 特征与 task embedding 之间学习异构交互；query 与 value 同样使用 cross/composite 投影。
- 效率优化之低秩近似：对 Composite 投影做低秩分解 K̂^h = L^h_k (R^h_k)^T，query/key/value 分别使用秩 r_k、r_v；当 r_k < Ld_k/2 且 r_v < Ld_v/2 时计算成本下降；作者通过奇异值分析观察到 V̂^h 矩阵确实呈低秩结构。
- 效率优化之模型剪枝：由于输出层只用编码后的 task embedding 做预测，最后一层 Hiformer 可只以 task embedding 作为 query、以 task embedding 与特征 embedding 作为 key/value，从而把最后一层复杂度从随 L 二次增长降为线性增长；该剪枝技术可应用于所有 Transformer 架构，与 Perceiver 的思路类似。
- 复杂度分析：Hiformer 原始推理成本为 O(L²d² + L²d + Ld²)（QKV 投影 3L²d²、注意力打分 2L²d、输出投影 Ld²、FFN 8Ld²）；低秩近似后为 O(L²d + Ld²)；剪枝后最后一层为 O(L(r_k+r_v)d + Ltd + td²)。

## 实验与证据

- 离线数据集与设置：使用 Google Play 排序模型的日志数据，标签为用户与 App 的互动（0/1），损失为 LogLoss；用 35 天滚动窗口数据训练、第 36 天数据评估，模型每天用最新数据从头重训；包含 31 个类别特征与 30 个稠密标量特征（如 app ID、app title、user language 等），类别特征词表规模从数百到数百万不等。
- 评估指标与效率测量：主指标为 AUC，作者指出在该 web-scale 评估集上 AUC 提升 0.001 即达到统计显著；同时报告以单层 Transformer 为基线归一化的 LogLoss；训练在 TPU 上进行并报告训练 QPS；延迟通过离线模拟测量，对 20 个 batch（batch size 1024）做推理，并以带剪枝的单层 Transformer 为基线归一化。
- 实现细节：基于 TensorFlow 2 与 Model Garden；除特征交互层外所有模型组件保持一致；embedding 维度统一为 128；Transformer 类模型设 d=128、H=4、d_f=512，且设 d_k=16、d_v=64（而非 d/H）；Transformer、HeteroAtt、Hiformer 仅使用编码后的 CLS/task token 做最终预测；其他方法做了超参调优；在线实验用某一天（如 2 月 1 日）数据调参并在之后日期固定。
- 离线主结果（Table 1，参数量不含 embedding 层）：AutoInt 1 层 12.39M 参数、AUC 0.7813、LogLoss -0.37096、训练 QPS 5.45e6、归一化服务延迟 2.28；DLRM 5.95M 参数、AUC 0.7819、LogLoss -0.47695、QPS 5.14e6、延迟 0.95；DCN 1 层 13.73M 参数、AUC 0.7857、LogLoss -0.79491、QPS 5.76e6、延迟 1.46。
- 离线主结果（Transformer 系列）：Transformer 1 层 0.74M 参数、AUC 0.7795、LogLoss 0（基线）、QPS 5.75e6、延迟 1.00；2 层 0.84M 参数、AUC 0.7811、LogLoss -0.31797、QPS 4.12e6、延迟 3.03；3 层 0.97M 参数、AUC 0.7838、LogLoss -0.45045、QPS 3.13e6、延迟 5.05；Transformer+PE 3 层 1.08M 参数、AUC 0.7833、LogLoss -0.39746、QPS 3.12e6、延迟 5.06。
- 离线主结果（本文模型）：HeteroAtt 1 层 2.36M 参数、AUC 0.7796、LogLoss -0.05299、QPS 5.71e6、延迟 1.01；HeteroAtt 2 层 10.50M 参数、AUC 0.7856、LogLoss -0.82141、QPS 4.10e6、延迟 3.11；Hiformer 1 层 16.68M 参数、AUC 0.7875、LogLoss -0.87440、QPS 5.69e6、延迟 1.52。即 Hiformer 单层取得最高 AUC 0.7875，优于 DCN 的 0.7857 与 HeteroAtt 2 层的 0.7856。
- Q1（同构 vs 异构）：Transformer+PE 与 vanilla Transformer 表现相近，说明仅学习每特征偏置 embedding 不足以实现异构特征交互；带异构注意力层的 HeteroAtt 显著优于 vanilla Transformer，验证异构注意力能通过变换矩阵 M_{i,j} 有效捕捉复杂特征交互；同时两层 HeteroAtt 参数量远大于两层 Transformer。
- Q2（模型性能对比）：AutoInt、DLRM、vanilla Transformer 因缺乏特征感知与语义对齐而表现相对较弱；DCN 的 Cross Net 隐式生成所有成对交叉并投影到低维空间，其成对交叉参数化各不相同，类似提供特征感知，因此与 HeteroAtt 表现相当；Hiformer 仅用一层即取得最佳性能，超过 HeteroAtt 与其他 SOTA 模型。
- Q3（服务效率）：两层 Transformer 延迟超过单层的 2 倍（因只对第二层剪枝，单层模型已剪枝），HeteroAtt 因算子相同呈现同样规律；剪枝无法应用于 AutoInt，故单层 AutoInt 比 vanilla Transformer 与 HeteroAtt 昂贵得多；单层 Hiformer 因 QKV 投影更具表达力，比单层 HeteroAtt 贵 50.05%。
- 低秩近似效果（Table 2）：设置 r_k=128、r_v=1024 时，带低秩近似的 Hiformer 为 16.68M 参数、AUC 0.7875、延迟 1.52；不带低秩近似为 59.95M 参数、AUC 0.7882、延迟 3.35；低秩近似带来 62.7% 的推理延迟节省，且无显著模型质量损失。
- 参数敏感性（Q4）：HeteroAtt 中把 Hd_k 从 d=128 降到 64 无质量损失且获得约 3% 的免费延迟改善，继续降低则质量明显下降，故选 Hd_k=64；增大 Hd_k 提升质量但显著增加延迟；为质量与延迟折中设 Hd_v=256。Hiformer 中把 Hd_k 降到 64 同样无质量损失，但与 HeteroAtt 不同几乎没有延迟收益，原因是 QK 投影做了低秩近似且 r_k=256 相对 r_v=1024 较小，query 投影主导计算成本；增大 Hd_v 延迟显著上升。作者指出 d_v 可作为在延迟代价下调节模型容量的手段，这是其他 SOTA 模型（如 DCN）不具备的。
- 在线 A/B 测试（Table 3）：控制组与实验组各随机选取 1% 用户，基线为单层 Transformer，收集 10 天互动指标；单层 HeteroAtt +1.27%*、两层 HeteroAtt +2.33%*、单层 Hiformer +2.66%*、单层 DCN +2.20%*（* 表示统计显著），Transformer 基线为 +0.00%；Hiformer 在所有模型中表现最佳并已部署到生产环境。

## 局限与开放问题

- 异构注意力层与 Hiformer 相比 vanilla Transformer 增加了参数量，且参数量随输入 embedding 列表长度线性增长；Hiformer 单层参数量达 16.68M，远高于单层 Transformer 的 0.74M。
- Hiformer 因更具表达力的 QKV 投影，单层服务延迟比单层 HeteroAtt 高 50.05%；若不使用低秩近似，延迟会显著更高（3.35 vs 1.52）。
- 由于引入 Composite 注意力层，现有高效 Transformer 架构无法直接套用于 Hiformer，需要专门设计低秩近似与剪枝方案。
- 低秩近似带来 62.7% 延迟节省的同时，AUC 从 0.7882 略降至 0.7875，存在轻微质量损失。
- 论文未给出在线 A/B 测试中 Hiformer 相对基线的具体延迟增幅数值，仅表述为「limited latency increase」。
- 实验仅基于 Google Play App 排序这一单一工业场景与单一数据集，未报告在其他数据集或推荐场景上的泛化结果。
- 论文未提供公开数据集或代码链接，离线数据为内部日志数据，外部难以复现。

## 待验证问题

- 如何将 NLP、CV 等其他领域 Transformer 架构的最新进展迁移到推荐系统？
- 新引入的特征交互组件能否用于神经架构搜索（NAS）以进一步改进推荐系统？
- Hiformer 的异构注意力与 Composite 投影设计在 Google Play 之外的推荐场景（如视频、电商、广告）中是否同样有效？
- 低秩近似中秩 r_k、r_v 的最优选择是否存在可自动搜索或理论指导的方法，而非依赖经验设定？
- 在 d_v 作为延迟-容量调节手段时，如何针对不同请求量场景自适应地选择 d_v？
- 剪枝技术虽可应用于所有 Transformer 架构，但对不同架构（如 AutoInt）无法应用的原因能否通过其他方式弥补？

## 与知识库的关联

- 与 AutoInt 直接相关：AutoInt 用多头自注意力学习特征交互，本文指出其缺乏特征感知与语义对齐，并将其作为主要对比基线（离线 AUC 0.7813、在线未列入 A/B 表）。
- 与 DCN/DCN-v2 直接相关：DCN 通过 Cross Net 显式学习随层深增长的高阶特征交叉，其成对交叉参数化不同类似提供特征感知，离线 AUC 0.7857、在线 +2.20%*，是本文认为最接近的 SOTA 竞争者。
- 与 DLRM 相关：DLRM 用因子分解机学习特征交互再用 MLP 学习隐式交互，离线 AUC 0.7819、延迟 0.95（低于基线）。
- 与 vanilla Transformer 相关：本文以单层 Transformer 作为延迟归一化基线与在线基线，并分析其同构参数共享设计不适合推荐场景中依赖上下文的特征语义。
- 与 Heterogeneous Graph Transformer 相关：两者都在注意力中考虑类型/异构性，但本文强调 Hiformer 面向 web-scale 推荐且注意力层表达力更强。
- 与 Field-aware Factorization Machine 相关：FFM 通过独立 embedding 查表实现特征感知，本文则通过异构注意力层中的信息变换实现。
- 与 PNN / kernel Factorization Machine 相关：PNN 用 kernel 矩阵考虑特征动态性，与异构注意力层动机相似，但本文基于注意力与 Q/K 投影，且进一步用 Hiformer 提升表达力并降本。
- 与 Perceiver 相关：Perceiver 也提出类似剪枝思路把 Transformer 服务成本从二次降为线性。
- 与 Wide & Deep、DeepFM、xDeepFM、Neural FM 等特征交叉工作构成相关工作脉络。
- 与 BERT4Rec、Self-Attentive Sequential Recommendation 等用户历史序列建模工作区分：本文聚焦特征交互学习而非序列建模。

## 核对提醒

- 论文 PDF 中作者单位为 Google DeepMind / Google Inc，但发现来源标注为 2025Meituan_MTmixAtt.pdf，二者不一致，引用时需核实实际发表信息。
- 论文头部标注为 Conference'17 模板（ACM ISBN 123-4567-24-567、DOI 10.475/123_4），且 arXiv 编号为 2311.05884v1（2023 年 11 月），属于预印本格式，正式发表信息需另行确认。
- Table 1 中 LogLoss 为相对单层 Transformer 的归一化值（基线为 0），并非绝对 LogLoss，解读时不可当作绝对指标。
- Table 1 中 Hiformer 的 AUC 0.7875 与 Table 2 中带低秩近似的 Hiformer AUC 0.7875 一致，但 Table 2 不带低秩近似为 0.7882，说明主表报告的是带低秩近似的结果。
- 在线 A/B 仅报告相对提升百分比，未给出绝对指标值、置信区间或延迟具体增幅。
- 论文正文中「HeteroAtt 1 层 AUC 0.7796」低于「Transformer 3 层 AUC 0.7838」，作者以「最佳性能对应层数」方式比较，引用时需注意层数差异。
- 文中公式 (2) 文字描述为「query and key projections for feature j and j」，疑为笔误（应为 feature i and j），引用公式时需谨慎。
- 低秩近似公式 (7) 中符号 L^h_v、R^h_v 与正文所述 K̂^h 的分解符号存在混用，引用时需核对原文。
- 论文未提供公开数据集与代码，所有数字均来自内部 Google Play 日志数据，外部无法独立复现。
