---
创建日期: 2026-10-19
更新日期: 2026-10-28
类型: 笔记
标签:
  - 机制设计
  - 广告技术
来源:
  - "[NGA: Non-autoregressive Generative Auction with Global Externalities for Advertising Systems (arXiv 2506.05685)](https://arxiv.org/abs/2506.05685)"
状态: 待复核
证据强度: 美团团队 CIKM '25 论文，2026-10-27 已读全文 HTML 版；实验数字来自论文表格，线上 A/B 为 2% 生产流量、指标以相对变化呈现，绝对值未公开
---

# NGA：非自回归生成式拍卖用全局外部性兼顾效果与推理效率

## 筛选与速览

- **一句话速览**：NGA（Non-autoregressive Generative Auction，美团上海团队）把 [[CGA：自回归生成式拍卖捕捉排列级外部性并解耦分配与支付]] 的自回归分配改成**非自回归生成 + 全局外部性建模**，以推理延迟为核心约束的学习型拍卖——它是 [[外卖列表广告生成式整体拍卖一期：用非自回归生成与RegretNet计费突破阶乘组合空间]] 业界调研表中的 NGA 原型，也是美团广告生成化路线（NGA → [[EGA-V2：把召回排序创意分配计费统一进单个生成模型的全链路广告框架]]）的第一环。
- **定位**：与 CGA 同属「模型生成 + 事后遗憾最小化计费」阵营（MIRT 批评的近似 IC 路线），差异在生成方式：非自回归一次成型换取线上低延迟，代价是外部性建模从排列级退回全局/集合级。
- **非自回归的具体做法**：不是 token 级并行解码，而是「一次前向算出全局分配概率矩阵 + 按位约束独立 argmax」，配合多塔并行评估器统一完成打分与计费。

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

外卖列表 LaunchReview 的业界调研表将其归纳为：Two stage，模型生成 + listwise reward 辅助训练；非自回归、推理延迟低，但线上仍需评估模型。MIRT 论文将其列为 feed 生成的学习型拍卖前置工作之一，实验测得该路线策略无恙（DSIC）违反可达广告主效用 10% 量级。

## 关联

- [[CGA：自回归生成式拍卖捕捉排列级外部性并解耦分配与支付]]：直接前置，自回归 → 非自回归的改造对象。
- [[EGA-V2：把召回排序创意分配计费统一进单个生成模型的全链路广告框架]]：同团队后续，从混排层扩展到全链路。
- [[MIRT：用maximal-in-range结构换取精确DSIC的whole-feed生成式拍卖]]：对其近似 IC 路线的批评与替代。
