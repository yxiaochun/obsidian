---
创建日期: 2026-11-18
更新日期: 2026-11-18
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源:
  - https://arxiv.org/html/2402.03300v3
  - https://arxiv.org/pdf/1707.06347
状态: 待确认
证据强度: DeepSeekMath与PPO原论文支持机制；推荐奖励和采样设计为AI推演
讲解日期: 2026-11-18
---

# GRPO用组内相对奖励优化生成式推荐策略

## 一句话结论

GRPO 对同一上下文采样多个输出，用组内相对奖励构造优势，再通过裁剪策略目标更新生成模型；它省去价值模型，不省去奖励来源，也不保证相对较好的结果绝对有效。

## 机制拆解

### 同一请求内建立比较基准

对上下文 $x$，采样策略生成 $G$ 个输出：$y_i\sim\pi_{\mathrm{old}}(\cdot\mid x)$。奖励为 $r_i=R(x,y_i)$，优势定义为：

$$
\bar r=\frac1G\sum_i r_i,\qquad
s_r=\sqrt{\frac1G\sum_i(r_i-\bar r)^2},\qquad
\hat A_i=\frac{r_i-\bar r}{s_r+\delta}
$$

$\delta$ 为数值稳定项；此处采用总体标准差。组由同一请求的多次生成组成，不是不同用户混在一起比较。

组内中心化减去共同奖励水平，标准化调整尺度，但不能据此断言训练没有偏差或请求难度影响完全消失。优势与奖励在策略更新中作为固定信号。

### 结果级奖励如何影响 token

暂时忽略裁剪、KL 与长度归一化，更新方向可理解为：

$$
\Delta\theta\propto\sum_i\hat A_i\nabla_\theta\log\pi_\theta(y_i\mid x)
$$

完整输出的对数概率等于 token 对数概率之和。结果级监督把同一个优势分给该输出的所有 token：正优势鼓励该路径，负优势抑制该路径，但不能直接归因到某个商品位置或编码 token。

### 裁剪与参考约束

$$
\rho_{i,t}=\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})}
$$

$$
\ell_{i,t}=\min\left(\rho_{i,t}\hat A_i,\operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_i\right)
$$

原始目标按每条输出长度平均，再按组平均，并扣除相对参考策略的 KL 惩罚估计：

$$
J(\theta)=\mathbb E\left[\frac1G\sum_i\frac1{|y_i|}\sum_t\left(\ell_{i,t}-\beta\hat D_{i,t}\right)\right]
$$

- $\pi_\theta$：正在更新的策略。
- $\pi_{\mathrm{old}}$：生成本批样本的快照，本批更新时固定，后续刷新。
- $\pi_{\mathrm{ref}}$：KL 的参考基准，可在训练阶段内固定，也可按阶段更新。

$\hat D_{i,t}$ 在原文中采用 $q-\log q-1$，$q=\pi_{\mathrm{ref}}(y_{i,t}\mid x,y_{i,<t})/\pi_\theta(y_{i,t}\mid x,y_{i,<t})$。该项是采样估计而非每个 token 的完整精确 KL；样本分布滞后时，其期望性质需另行考虑。

取正优势 1、裁剪阈值 0.2，概率比从 1 到 1.5 时，代理项最多计到 1.2。裁剪移除改善方向上过度更新的部分激励，不把实际概率比强制锁定在区间内。

## 数值例子

同一请求四个输出奖励为 `[1, 0, 0, 0]`：均值 0.25、标准差约 0.433，忽略稳定项，优势约为 `[1.732, -0.577, -0.577, -0.577]`。

如果全部奖励为 0，优势全为 0，奖励驱动的策略项失去更新信号；KL 或其他辅助项仍可能更新。反复训练同一批零差异样本不能凭空产生探索信号。

## 生成式推荐中的适配

1. 从能生成有效商品或列表的 SFT 模型开始。
2. 同一上下文随机生成 $G$ 个输出，明确输出单位是单商品还是完整列表。
3. 用规则、评分模型或反馈提供奖励，再计算组内优势和更新策略。
4. 周期性刷新采样策略，短期复用样本时监控概率比与分布偏移。
5. 在独立评估样本上检查 Recall、NDCG、有效率、重复率与推理成本。

“命中下一件商品给 1，否则给 0”容易在大商品库形成大量零方差组；连续相关性或 CTR 预测奖励更密集，但可能被策略利用评分误差。日志未曝光的生成商品没有真实点击反馈，不能将其标为真实未点击。

$$
\mathcal L_{\mathrm{DPO}}=-\mathbb E\log\sigma\left[\beta\left(\log\frac{\pi_\theta(y^+\mid x)}{\pi_{\mathrm{ref}}(y^+\mid x)}-\log\frac{\pi_\theta(y^-\mid x)}{\pi_{\mathrm{ref}}(y^-\mid x)}\right)\right]
$$
