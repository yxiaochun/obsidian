---
创建日期: 2026-10-19
更新日期: 2026-10-29
类型: 笔记
标签:
  - 机制设计
  - 广告技术
来源:
  - "[NGA: Non-autoregressive Generative Auction with Global Externalities for Advertising Systems (arXiv 2506.05685)](https://arxiv.org/abs/2506.05685)"
  - "[arXiv HTML 全文 v1](https://arxiv.org/html/2506.05685v1)"
状态: 已精读
证据强度: 美团（上海）团队 CIKM '25 论文，已读全文；线下指标基于 reward 模型预测且绝对值经脱敏转换，线上为 2% 流量 A/B
---

# NGA：非自回归生成式拍卖用全局外部性兼顾效果与推理效率

## 筛选与速览

- **一句话速览**：NGA（Non-autoregressive Generative Auction，美团上海团队，CIKM '25）把 [[CGA：自回归生成式拍卖捕捉排列级外部性并解耦分配与支付]] 的自回归分配改成**非自回归生成 + 全局外部性建模**，以推理延迟为核心约束的学习型拍卖——它是 [[外卖列表广告生成式整体拍卖一期：用非自回归生成与RegretNet计费突破阶乘组合空间]] 业界调研表中的 NGA 原型，也是美团广告生成化路线（NGA → [[EGA-V2：把召回排序创意分配计费统一进单个生成模型的全链路广告框架]]）的第一环。
- **定位**：与 CGA 同属「模型生成 + 事后遗憾最小化计费」阵营（MIRT 批评的近似 IC 路线），差异在生成方式：非自回归一次成型换取线上低延迟，代价是外部性建模从排列级退回全局/集合级。
- **非自回归的具体做法**：不是 token 级并行解码，而是「一次前向算出全局分配概率矩阵 + 按位约束独立 argmax」，配合多塔并行评估器统一完成打分与计费。
- **核心贡献 3 句话**：
  1. 提出 Generator + Evaluator 双模块框架：双编码器（Item Encoder + Position Encoder）产出分配概率矩阵，联合建模广告间依赖与**相邻自然内容**的全局外部性（CGA 只看广告间）。
  2. 用逐位置约束解码（per-position constrained decoding）替代自回归解码，并用多塔并行 Evaluator 同时输出 listwise pCTR/pCVR/支付比例，消除「先打分后排序列」与串行评估的延迟问题。
  3. 在美团到店（LBS）真实数据离线 RPM +2.2%、CVR +6.4%、IC regret 2.9%（均优于 CGA），线上 2% 流量 A/B 对多目标 CGA：RPM +1.9%、CTR +1.6%、CVR +0.8%、RT **-6.5%**。

## 问题定义

- 场景：一次 PV 请求，n 个广告候选 + m 个自然候选（自然序列由上游按预估订单量排定、内部顺序固定），从中选出 k 个坑位的最终列表，同时定分配与支付。
- 目标：max E[Rev + α·Ord] = Σ p_i·ctr_i + α·ctr_i·cvr_i，s.t. IC、IR。
- 对 CGA 的三点批评（即本文动机）：① 忽略邻近自然内容的外部性；② 自回归生成顺序解码延迟高，坑位/候选多时不可扩展；③ 评估与支付计算串行，进一步放大延迟。

## 方法拆解

### Generator（非自回归生成）

- **双编码器**：Item Encoder 用 L 层 self-attention + FFN 编码每个候选 item → x_i；Position Encoder 用可学习位置 embedding + L 层 self-attention/cross-attention 关注所有 item 表示 → t_j。
- **分配概率矩阵** Z ∈ R^{(n+m)×k}，一次算出所有「item × 坑位」得分，非自回归：

  z_ij = softmax(ctr_i × bid_i + α × ctr_i × cvr_i + x_iᵀ t_j)

  即显式 ecpm 式打分（bid·ctr + α·ctr·cvr）与隐式外部性得分（x_iᵀ t_j）相加。
- **逐位置约束解码**：每个坑位 j 在满足业务约束的可行集 I_j 内取 argmax Z_ij。约束包括广告最靠前坑位、广告密度、同品牌去重等——这是工程上的关键设计，把硬约束放进解码而非靠模型学。

### Evaluator（多塔并行评估）

- List Encoder：对候选列表 Y=[y_1..y_k] 的 item embedding 做 L 层 transformer，建模列表内依赖与上下文。
- 与分配概率矩阵 Z 做线性投影后，**三个并行 FFN 塔**（Sigmoid(MLP)）同时输出：
  - Θ：listwise pCTR
  - Γ：listwise pCVR
  - p：支付比例（额外输入自排除竞价档案 B⁻，即每个坑位排除自身报价后的他人出价），pay_i = bid_i × p_i
- 总奖励 R = Σ (Θ_i × bid_i × p_i + α × Θ_i × Γ_i)，取 R 最高的候选列表展示（推理时 beam search，beam=20）。

### 训练

- **先训 Evaluator**：pCTR/pCVR 用真实点击/转化标签的 BCE；支付项用含事后遗憾（ex-post regret r̂gt_i）的拉格朗日损失 L_pay = -(1/|D|)Σ(Σ bid_i p_i Θ_i - Σ λ_i r̂gt_i - (ρ/2)Σ r̂gt_i²)，即 RegretNet 式的 gradient-friendly IC 近似，λ 为拉格朗日乘子、ρ 为惩罚系数。最终 L_E = w1·L_pctr + w2·L_pcvr + w3·L_pay。
- **再训 Generator（冻结 Evaluator）**：策略梯度，奖励为每个 item 的边际贡献 r_{y_i} = R_Y - R_{Y^{-i}}（Y^{-i} 为去掉 y_i 后的最优替代序列），L_G = -Σ r_{y_i} log Z_{y_i}。
- 与内部文档「Two stage，模型生成 + listwise reward 辅助训练」的对应：第一阶段 Generator 非自回归生成候选列表，第二阶段 Evaluator 以 listwise reward 选优；「辅助训练」对应 Evaluator 冻结后作为 policy gradient 的 reward 来源。

## 实验结论

- **数据**：美团到店（LBS）平台，2025-05-06 ~ 05-20，约 2.25 亿页面请求、6000 万用户、近 2 亿候选 item；前 14 天训练、最后 1 天测试。
- **设置**：n=30 广告 + m=20 自然 item，k=10 坑位；A100 80G，Adam lr=3e-3，batch 2048，α=5，beam=20，跑 5 次取均值。基线 uGSP / DNA / CGA 均固定坑位、与 NGA 同广告曝光占比。
- **离线**（绝对值已脱敏，括号为 NGA 相对最优基线提升）：

  | 模型 | RPM | CTR | CVR | Ψ (IC regret) |
  |---|---|---|---|---|
  | uGSP | 200.42 (-11.5%) | 5.26% | 5.10% | 11.4% |
  | DNA | 210.39 (-7.1%) | 5.54% | 5.14% | 8.5% |
  | CGA | 221.49 (-2.2%) | 5.75% | 4.98% | 3.8% |
  | NGA | 226.47 | 5.83% | 5.32% | **2.9%** |

- **线上 A/B**：对多目标版 CGA（固定坑位，业务约束版），2% 流量：RPM +1.9%、CTR +1.6%、CVR +0.8%、RT -6.5%。且需要生成的候选序列越多，效率收益越明显。

## 落地可行性评估

- **延迟收益是实证的**：非自回归 + 并行多塔使 RT 降 6.5%，这对混排层（候选/坑位多、SLA 紧）是刚需；CGA 类自回归在坑位增多时延迟劣势会放大。
- **IC 是近似保证**： regret 通过惩罚项压到 2.9%（线下、reward 模型口径），不是精确 DSIC；[[MIRT：用maximal-in-range结构换取精确DSIC的whole-feed生成式拍卖]] 对该路线的批评（策略无恙违反可达广告主效用 10% 量级）对应的是此类学习型拍卖的通用风险，NGA 未给出对抗策略性出价的实验。
- **依赖 reward 模型口径**：离线所有指标基于 Evaluator 预测而非真实反事实，绝对值脱敏，外部复现只能对齐相对关系。
- **支付机制完全学习，但 IR 由结构硬保证**：pay_i = bid_i × p_i，p 由网络输出（额外输入他人出价的 B⁻），且 p 经 Sigmoid 压到 (0,1)，故 pay_i 恒小于 bid_i——IR 不靠训练收敛，是输出层结构直接保证；IC 则靠遗憾最小化损失学出来（拉格朗日乘子 λ 控制 IC 强度），仍是近似保证。
- **综合评分：⭐⭐⭐⭐**——同为美团内部、混排层场景完全对口，架构改动集中在「生成方式 + 评估并行化」，工程路径清晰；主要不确定性在 IC 稳定性与约束解码在大规模候选下的召回质量。

## 复现要点

1. 数据格式：页面级样本（候选 item 特征 + 曝光/点击/转化标签 + 出价），需保留自然内容上下文以建模全局外部性。
2. 训练顺序：先 Evaluator（BCE ×2 + 遗憾最小化支付损失）→ 冻结后 policy gradient 训 Generator；关键超参 α=5（收入/订单权衡）、ρ（IC 惩罚）、w1/w2/w3。
3. 推理：分配矩阵一次算完 → 逐位置约束解码 + beam search（beam=20）→ 多塔并行打分选 R 最高列表。
4. 校验点：IC regret Ψ 用事后遗憾/效用归一化口径；约束解码的可行性兜底（可行集为空时的降级策略论文未述）。

## 论文信息

- 作者/机构：Zuowu Zheng*、Ze Wang*、Fan Yang†、Wenqing Ye、Weihua Huang、Wenqiang He、Teng Zhang、Xingxing Wang，美团（上海）。
- 发表：arXiv 2506.05685，CIKM 2025（DOI 10.1145/3746252.3760955）。
- 本卡信息边界：2026-10-27 已读全文 HTML 版，方法与实验细节来自原文；MIRT 对该路线 IC 违反的量化批评见下方「与内部文档的对应」。

## 方法架构（2026-10-27 读全文补全）

NGA 由 Generator（非自回归分配）和 Evaluator（多塔并行评估）两部分组成。

### Generator：一次前向 + 按位约束解码

1. **双编码器建模全局外部性**：
   - ItemEncoder：全部候选（n 个广告 + m 个自然结果）的 embedding 过 L 层 Transformer 自注意力，物品间相互关系在此一次性建模；
   - PositionEncoder：k 个展示位的可学习位置 embedding，经自注意力 + 对全部物品表示的交叉注意力，让每个位置同时感知其他位置与所有候选，相邻自然内容的影响在这一步引入。
2. **一次前向算出分配概率矩阵**（非自回归的核心）：


$$
z_{ij}=\text{softmax}(\text{ctr}_i\times\text{bid}_i+\alpha\cdot\text{ctr}_i\times\text{cvr}_i+\mathbf{x}_i^\top\mathbf{t}_j)
$$

得到 $\mathbf{Z}\in\mathbb{R}^{(n+m)\times k}$，所有「物品 × 位置」分数在同一个前向传播里并行算完，位置之间没有条件依赖。
3. **按位约束解码**：对每个位置独立取 argmax，候选集先按业务规则过滤（广告最小起始位、广告密度上限、品牌去重）：$i_j^*=\arg\max_{i\in\mathcal{I}_j}\mathbf{Z}_{ij}$。无 beam search 展开、无前缀状态。

> [!info] 「并行」的边界
> 打分完全并行：$\mathbf{Z}$ 矩阵一次前向算完，各位置分数不依赖前面的选择，这是非自回归延迟优势的来源。但选择环节论文只给了并行形式化公式：静态约束（最小起始位、位置级间隔）的候选集可预先并行确定，动态约束（品牌去重、广告密度上限）和同物品碰撞的消解依赖前面已选项，工程实现大概率是从前往后的轻量贪心扫描——串行的只是查表和规则过滤，无任何模型前向调用，代价微秒级。论文未交代碰撞处理与动态约束的具体执行方式，属于模糊地带。

### 多候选序列生成：在概率矩阵上做 beam search

论文 Implementation Details 明确：推理时用 beam search（beam size 20）生成多条候选序列。beam search 的作用对象是已经算完的 $\mathbf{Z}$ 矩阵，而非模型逐步输出——所有分数已躺在矩阵里，每步扩展只是查表和排序，无额外模型调用。argmax 只是 beam search 的 B=1 退化形式。

**缩微示例**（k=3 个位置、B=2、候选 {A,B,C,D}，约束为每物品限用一次；表中为各位置分配概率）：

| 物品 | 位置1 | 位置2 | 位置3 |
|---|---|---|---|
| A | 0.9 | 0.1 | 0.8 |
| B | 0.8 | 0.2 | 0.2 |
| C | 0.4 | 0.3 | 0.1 |
| D | 0.3 | 0.9 | 0.4 |

1. **位置1**：全部候选按分排序取 top2 → 假设池 {A(0.9), B(0.8)}，C、D 被剪掉；
2. **位置2**：两条假设各自扩展（已用物品剔除）：A+D=1.8、B+D=1.7、A+C=1.2、A+B=1.1、B+C=1.1、B+A=0.9 → 全局排序留 top2 {A+D, B+D}；
3. **位置3**：A+D 扩展得 A+D+B=2.0、A+D+C=1.9；B+D 扩展得 B+D+A=2.5、B+D+C=1.8 → 最终输出 {B+D+A(2.5), A+D+B(2.0)}。

三个要点：

- **存活假设数恒为 B**：每轮扩展产生的组合按累积得分全局排序后只留 B 条，复杂度 $O(k\cdot B\cdot|V|)$，不是 $B^k$ 的穷举笛卡尔积。NGA 真实参数（k=10、B=20、候选约 50）下就是 10 轮「20×50 次查表排序」；
- **累积得分救回落后者**：胜出序列 B+D+A 的首位置 B 只排第二，纯贪心链（A→D→B=2.0）永远到不了 2.5——这是论文开 B=20 而不用 argmax 的原因；
- **约束即过滤**：每物品限用一次、品牌去重、广告密度上限，处理方式相同——扩展时违反约束的组合不参与排序。

20 条候选的用途：**推理时**并行送入多塔 Evaluator，总奖励 R 最高者展示；**训练时**策略梯度的边际贡献 $r_{y_i}=R_{\mathbf{Y}}-R_{\mathbf{Y}^{-i}}$ 需要「去掉 $y_i$ 后的最优替代序列」，单条 argmax 序列无法计算。

### Evaluator：多塔并行打分与计费

**输入**（每条候选列表三样东西）：

1. 列表内物品的原始特征 $[\mathbf{e}_{y_1};\mathbf{e}_{y_2};\dots;\mathbf{e}_{y_k}]$；
2. Generator 的分配概率矩阵 $\mathbf{Z}$；
3. 剔除自身报价的 bidding profile $\mathbf{B}^-\in\mathbb{R}^{k\times(k-1)}$（仅支付塔使用，维克里式「支付只看别人报价」的结构）。

**流程**：ListEncoder（L 层自注意力 Transformer）编码列表 → $\mathbf{X}_{\text{list}}$ 与 $\mathbf{Z}$ 线性投影融合 → 三个并行塔（Sigmoid(MLP)）。ListEncoder 的列表内注意力是外部性进入评估的方式：每个位置的预测都能看到整条列表的其他物品，广告的 pCTR 会被旁边的强吸引力内容压低。

| 输出 | 形状 | 含义 |
|---|---|---|
| $\Theta_i$ | k×1 | 位置 i 的 list-wise pCTR（受列表上下文影响，非孤立物品 pCTR）|
| $\Gamma_i$ | k×1 | 位置 i 的 list-wise pCVR |
| $p_i$ | k×1 | 支付比例，最终支付 $\text{pay}_i=\text{bid}_i\times p_i$，随报价天然伸缩 |

**奖励合成**：

$$
R=\sum_{i=1}^{k}\big(\Theta_i\times\text{bid}_i\times p_i+\alpha\times\Theta_i\times\Gamma_i\big)
$$

- 收入项 $\Theta_i\cdot\text{bid}_i\cdot p_i$ = 点击概率 × 支付 = 该位置期望收入；
- 体验项 $\alpha\cdot\Theta_i\cdot\Gamma_i$ = 期望成交（LBS 订单目标），论文 α=5；
- 20 条候选并行算出 20 个 R，取最大者展示。

**训练**（先收敛后冻结，供 Generator 策略梯度使用）：

1. $\mathcal{L}_{\text{pctr}}$ / $\mathcal{L}_{\text{pcvr}}$：Θ、Γ 分别对真实点击、转化标签做交叉熵；
2. $\mathcal{L}_{\text{pay}}$：RegretNet 式拉格朗日博弈——支付塔最大化收入 $\sum_i\text{bid}_i p_i\Theta_i$，事后遗憾 $\widehat{\text{rgt}}_i$（广告主虚报报价可多赚的效用，需对每个广告主重查询机制算出）通过乘子 $\lambda_i$ 与二次惩罚反向压制，学出「虚报无利可图」的支付比例；残余遗憾即线上 IC 遗憾 Ψ=2.9% 的来源；
3. 合并为 $\mathcal{L}_E=w_1\mathcal{L}_{\text{pctr}}+w_2\mathcal{L}_{\text{pcvr}}+w_3\mathcal{L}_{\text{pay}}$。

**小结**：Evaluator 的输出不是单一分数，而是三个逐位置预测头加合成的标量奖励 R；打分 = 一次并行 Transformer 前向 + 三个 MLP，外部性由列表内注意力承载，IC 由支付塔的遗憾对抗训练近似保证。

### 训练

1. 先训 Evaluator：pCTR/pCVR 交叉熵 + 拉格朗日遗憾最小化损失（事后遗憾进入损失函数，近似保证 IC）；
2. 再训 Generator：固定 Evaluator，策略梯度优化，奖励用每个物品的边际贡献 $r_{y_i}=R_{\mathbf{Y}}-R_{\mathbf{Y}^{-i}}$。

> [!info] 与内部文档表述的对应
> 外卖列表 LaunchReview 调研表中的「Two stage，模型生成 + listwise reward 辅助训练」即对应 Generator（生成候选列表）+ Evaluator（listwise 打分选优）；「listwise reward」即总奖励 R，经策略梯度回传给生成器。

## 实验与部署数据（论文数字）

- **离线**（LBS 平台数据，2025-05-06 至 05-20，约 2.25 亿 PV 请求、6000 万用户、2 亿候选；n=30 广告、m=20 自然结果、k=10 展示位；推理时用 beam size 20 从并行解码结果生成多个候选序列）：对比 uGSP、DNA、CGA，NGA 取得最优 RPM（226.47，较 CGA +2.2%）、CTR（5.83%，+1.8%）、CVR（5.32%，+6.4%）与最低 IC 遗憾 Ψ=2.9%。
- **线上 A/B**（2% 生产流量，对比多目标版 CGA）：RPM +1.9%，CTR +1.6%，CVR +0.8%，响应时间 RT −6.5%，确认已实际部署且推理延迟低于自回归基线。

> [!warning] 外部性建模的代价
> 按位独立 argmax 意味着解码时无法像自回归那样显式建模「前面放了什么广告影响后面」，位置间依赖完全依赖编码器注意力隐式捕捉——外部性建模从 CGA 的排列级退到集合级，是效果上限换推理效率的设计选择。

## 与内部文档的对应

外卖列表 LaunchReview 的业界调研表将其归纳为：Two stage，模型生成 + listwise reward 辅助训练；非自回归、推理延迟低，但线上仍需评估模型——与论文结构一致（见上文方法拆解末尾）。MIRT 论文将其列为 feed 生成的学习型拍卖前置工作之一，实验测得该路线策略无恙（DSIC）违反可达广告主效用 10% 量级；NGA 论文自身的线下 Ψ=2.9%，二者口径不同（后者为 regret/效用比值且在 reward 模型上测），不矛盾但说明 IC 风险未被完全消除。

## 关联

- [[CGA：自回归生成式拍卖捕捉排列级外部性并解耦分配与支付]]：直接前置与线上 baseline，自回归 → 非自回归的改造对象。
- [[EGA-V2：把召回排序创意分配计费统一进单个生成模型的全链路广告框架]]：同团队后续，从混排层扩展到全链路。
- [[MIRT：用maximal-in-range结构换取精确DSIC的whole-feed生成式拍卖]]：对其近似 IC 路线的批评与替代。
- [[广告混排方案对比：外部性建模、生成范式与计费机制的路线分化]]：路线定位图。
