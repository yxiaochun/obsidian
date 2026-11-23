---
创建日期: 2026-11-18
更新日期: 2026-11-18
类型: 笔记
标签:
  - 推荐系统
  - 生成式推荐
来源:
  - https://research.google/pubs/on-sampled-metrics-for-item-recommendation/
  - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
  - https://arxiv.org/html/2305.05065
状态: 待确认
证据强度: 指标定义及采样风险有来源支持；算例与评估方案为AI解释
讲解日期: 2026-11-17
---

# Recall与NDCG必须结合候选范围评估生成式推荐

## 一句话结论

Recall@K 衡量前 K 命中多少相关商品，NDCG@K 进一步考虑位置；候选范围、测试标签和解码方式不同，即使指标名称相同，数值也不能直接比较。

> [!info] 证据边界
> 本卡由每日讲解整理，未经用户确认。数字为自构算例；抽样指标的模型排序不一致风险来自论文；生成式推荐评估清单属于方法建议。

## 证据来源

- [On Sampled Metrics for Item Recommendation](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/)：抽样指标未必保留全量评估的模型优劣关系，即使取期望也可能不一致。
- [scikit-learn NDCG 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)：位置折扣、理想排序归一化与实现口径。
- [TIGER 原论文](https://arxiv.org/html/2305.05065)：语义 ID 生成、beam search 和无效 ID 问题。

## 机制拆解

### Recall：命中数量

对测试请求 $q$，设相关商品集合为 $R_q$，前 K 个不同商品组成 $L_q^K$：

$$
\operatorname{Recall@K}(q)=\frac{|R_q\cap L_q^K|}{|R_q|}
$$

只有一个目标商品时，它等于命中指示函数，与该设置的 Hit Rate@K 相同。多目标时，Hit Rate 只看是否至少命中一个，不能再与 Recall 混用。

默认按请求宏平均；按用户平均、按相关标签数加权会改变总体含义。无相关标签时应明确排除或记分约定，不能直接除以零。

### NDCG：位置价值

采用二元相关性 $\mathrm{rel}_{q,j}\in\{0,1\}$：

$$
\operatorname{DCG@K}(q)=\sum_{j=1}^{K}\frac{\mathrm{rel}_{q,j}}{\log_2(j+1)},\qquad
\operatorname{NDCG@K}(q)=\frac{\operatorname{DCG@K}(q)}{\operatorname{IDCG@K}(q)}
$$

IDCG 是相同相关标签集合下的理想排序得分。多级标签需要明确增益函数：直接使用相关性分值与 $2^{\mathrm{rel}}-1$ 不同；scikit-learn 文档对应直接分值增益。

### 算例

相关商品为 A、B、C，预测前五为 `[D, A, E, B, F]`：

$$
\operatorname{Recall@5}=2/3
$$

$$
\operatorname{DCG@5}=1/\log_2 3+1/\log_2 5\approx1.062
$$

$$
\operatorname{IDCG@5}=1+1/\log_2 3+1/\log_2 4\approx2.131
$$

所以 NDCG@5 约为 0.498。改成 `[A, B, D, E, F]` 后 Recall 不变，NDCG 约为 0.765。

单目标排名为 $r$ 时，若 $r\le K$，NDCG 为 $1/\log_2(r+1)$，否则为 0。

## 负采样为什么改变问题

假设全量含目标共 $N$ 个候选，目标严格排名 $r$。从另外 $N-1$ 个商品中均匀、不放回抽 $m$ 个，保留目标并按原分数重排。期望领先商品数为 $m(r-1)/(N-1)$，因此：

$$
\mathbb E[r_{\mathrm{sample}}]=1+m\frac{r-1}{N-1}
$$

取 $N=10001,r=101,m=99$，全量 Recall@10 为 0，但抽样期望排名为 1.99。这里无分数并列，公式不适用于任意热门采样或硬负例采样。

> [!warning] 不能把期望排名直接代入指标
> Recall、NDCG 是排名的非线性函数，通常 $\mathbb E[f(r)]\ne f(\mathbb E[r])$。多次重复采样降低随机波动，不自动消除相对全量指标的系统偏差；简单放大抽样排名也不能恢复真实指标。

抽样评估回答“在该抽样候选集合内能否区分目标”，全量评估回答“在定义的完整可推荐范围内能否找回目标”。两者应分别命名和报告。

## 生成式推荐的评估方案

| 实验 | 做法 | 实际回答的问题 |
| --- | --- | --- |
| 候选内排序 | 对已给定商品编码计算完整序列概率，再排序 | 候选已摆在面前时能否选对 |
| 实际解码召回 | 运行 beam 或采样，映射商品、过滤和去重后计分 | 实际搜索与后处理能否找回目标 |

不能先强行加入目标再评估，随后把结果表述成真实生成召回。建议同时报告有效商品率、最终不同商品数、补齐策略、beam 宽度和时延。

## 适用边界与待验证问题

- 固定时间切分、当时可用商品集合、历史商品过滤、并列分数规则和请求/用户聚合方式。
- 标签只覆盖已观察行为，全量候选不自动消除曝光偏差或标签不完整。
- 用 NDCG 构造训练奖励时，验证集和测试集不能再参与策略更新；评估应保留独立样本。
- 离线提升不等于在线 CTR、转化或长期满意度提升；需要对应目标的在线实验。
- 需要验证：扩大 beam 的收益是否值得成本？不同负采样分布是否改变模型排序？

## 复习

- 单目标排名从 8 到 2，Recall@10 都为 1，NDCG 从约 0.315 提高到 0.631。
- “目标加 99 个随机商品”的 Recall@10 为 90%，不能解释成全量召回率为 90%。

## 关联卡片

- [[TIGER用残差量化语义ID实现生成式召回]]：区分序列概率与实际解码输出。
- [[04_Outputs/报告/GRPO用组内相对奖励优化生成式推荐策略｜初稿]]：训练奖励与独立评估的区别。
