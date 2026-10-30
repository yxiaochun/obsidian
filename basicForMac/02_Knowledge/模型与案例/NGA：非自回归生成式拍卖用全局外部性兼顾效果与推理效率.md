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
- 局限（论文自述 + 推断）：IC 为近似保证；自然序列顺序固定不可重排；未报告对抗性出价下的鲁棒性；论文未开源代码。

## 与内部文档的对应

外卖列表 LaunchReview 的业界调研表将其归纳为：Two stage，模型生成 + listwise reward 辅助训练；非自回归、推理延迟低，但线上仍需评估模型——与论文结构一致（见上文方法拆解末尾）。MIRT 论文将其列为 feed 生成的学习型拍卖前置工作之一，实验测得该路线策略无恙（DSIC）违反可达广告主效用 10% 量级；NGA 论文自身的线下 Ψ=2.9%，二者口径不同（后者为 regret/效用比值且在 reward 模型上测），不矛盾但说明 IC 风险未被完全消除。

## 关联

- [[CGA：自回归生成式拍卖捕捉排列级外部性并解耦分配与支付]]：直接前置与线上 baseline，自回归 → 非自回归的改造对象。
- [[EGA-V2：把召回排序创意分配计费统一进单个生成模型的全链路广告框架]]：同团队后续，从混排层扩展到全链路。
- [[MIRT：用maximal-in-range结构换取精确DSIC的whole-feed生成式拍卖]]：对其近似 IC 路线的批评与替代。
- [[广告混排方案对比：外部性建模、生成范式与计费机制的路线分化]]：路线定位图。
