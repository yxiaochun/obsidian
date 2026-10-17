---
创建日期: 2026-10-17
更新日期: 2026-10-17
类型: 笔记
标签:
  - 机制设计
  - 广告技术
来源:
  - "[20250610-LaunchReview-到店餐饮美团筛选50011-基于DNA的整体拍卖算法gDNA一期（学城，内部文档）](https://km.sankuai.com/collabpage/2791029005)"
证据强度: 美团内部单场景实验（美团筛选 50011 广告位，7 天，20% 流量），含位次分段归因与扰动消融；数据未公开、未独立复核
---

# 美团到餐gDNA一期实验：把自然候选当作零出价广告的整体拍卖混排

## 筛选与速览

- **一句话速览**：gDNA（General Deep Neural Auction）是 [[美团到餐多位次NeuralAuction实验：逐坑位竞拍兼容多模版灵活位次]] 之后混排框架的换代尝试——把自然候选视为**出价为 0 的特殊广告**，与广告统一基于 rankscore 网络端到端生成混排序与计费，替代「人工模版 + 多位次 NeuralAuction 逐模版生成」的受限方案，线上 RPS +3.44%、CTR +2.46%，推全预计收入增长约 12.07 万/天，且耗时反而下降 10ms。
- **最有价值的部分**：一是「自然 = 零出价广告」这个极简的问题重构，让广告拍卖机制直接吃下混排问题；二是它诚实记录了一期如何把三个理想化设计（分数空间对齐、Listwise 校准、自然+广告统一 Reward）逐一降级为可落地的近似方案——这种「理想方案 → 落地退化」的路径比最终效果数字更有复用价值。
- **效果归因的关键证据**：广告平均曝光位次 30→33（整体后移），前 10 位曝光减少、20-40 位增加，大部分位次广告 CTR 好于或持平基线——收益不是位置靠前带来的，而是脱离固定模版后整体动态分配的效率最大化。

## 背景与问题

当前美团筛选采用**生成-评估式混排**，主流程三段：

1. **序列生成**：根据多模版配置，多位次 NeuralAuction 为每个模版生成模型认为最优的广告排序及计费，每模版一条序列；
2. **序列评估**：Listwise 模型感知序列上下文，输出单点的 list_ctr、list_cvr；
3. **序列优选**：按公式挑选最优序列：

$$
listscore = \sum_{i}\left(isAd_i\cdot\delta\cdot listctr_i\cdot price_i + \alpha\cdot listctr_i + \beta\cdot listcxr_i + \gamma\cdot listcxr_i\cdot amt_i\right)
$$

线上基线流程：

![[99_Attachments/图片/km-2791029005/线上基线流程图.svg]]

混排优化集中在两个方向：**外部性建模**（捕捉最终分配结果的整页上下文，提高预估精度）和**生成算法**（性能约束下在巨大生成空间中找更好的序列）。Listwise 侧的位次、广告上下文、自然上下文建模已比较全面；瓶颈在生成侧——位次模版人工配置且固定、受性能限制候选仅十几个，且多位次 DNA 只能捕捉粗粒度的候选层面外部性和局部位次信息，看不到整页上下文，生成的广告序不是全局最优。

## 业界调研与选型

| 业界算法 | 自然/广告 | 外部性建模 | 生成算法 |
| --- | --- | --- | --- |
| PRM | 自然重排 | transformer 建模 item 间外部性；预估阶段与最终上下文不一致，外部性不准确 | 无生成算法，直接改序 |
| PRS | 自然重排 | biLSTM 建模 item 间外部性 | two stage，规则生成 + beam search 选 k 个 reward 最大的列表 |
| MIAA | 广告+自然混排 | 考虑广告和自然整体上下文 | two stage，规则生成；TOP4 广告全排 + 模版配置位次满足性能，但大幅缩减候选空间 |
| CGA | 广告重排 | 只有广告上下文，无自然上下文 | one stage，模型生成，listwise reward 引导优化；候选空间大但自回归耗时较长 |
| NLGR | 广告重排 | 只有广告上下文，无自然上下文 | one stage，迭代寻找更优邻居列表直至停止条件；更稳定但停止条件不明确 |

趋势是 one stage 生成模型直接产出混排结果（如 CGA），但自回归耗时且未考虑自然上下文。gDNA 的选型逻辑：DNA 端到端生成广告序和计费、线上耗时短、已做少量广告上下文建模，**把自然候选加进来并完善外部性建模**，就得到一种非自回归的生成式混排算法。

## gDNA 框架：自然候选 = 出价为 0 的广告

gDNA 将自然候选视为出价为 0 的特殊广告，与广告统一基于 rankscore 网络排序，网络对自然和广告候选无感，自然候选通过设置 bid=0 参与混排：

![[99_Attachments/图片/km-2791029005/gDNA模型架构图.svg]]

相比 DNA 变化的三个模块（黄色为新增部分）：

- **Rankscore 网络**：让广告和自然候选输出可比的 rankscore：$rankScore = e^\delta\cdot ecpm + e^\alpha\cdot ctr + e^\beta\cdot cxr + e^\gamma\cdot gmv + embedding$，自然候选 bid=0、ecpm=0；
- **Listwise 校准**：序列过 listwise 模型，得到感知外部性的更准预估 listctr/listcxr 用于 loss；
- **Reward 设计**：同时考虑自然体验指标和广告收益最大化：

$$
Loss = -Reward,\qquad Reward=\frac{1}{n}\sum_{i=0}^n\sum_{j=0}^k allocate_{ij}\cdot target_{ij}
$$

$$
target_{ij} = \delta\cdot isAd\cdot listctr_{ij}\cdot price + \alpha\cdot listctr_{ij} + \delta\cdot listcxr_{ij} + \gamma\cdot gmv
$$

![[99_Attachments/图片/km-2791029005/DNA模型架构图.svg]]

## 到餐业务的三个落地挑战与一期退化

理想方案到餐场景会直接碰壁：① 特征层面，广告与自然预估分 diff 大（广告是广告归因、自然是自然归因，CVR 分差 2 倍），直接用预估分公式排序不可行；② Listwise 模型以自然 label 为目标，广告预估结果与优化目标对不齐，直接接入校准训练不稳定。一期的处理方式是逐项降级：

### 3.1 分数空间对齐

- **自然分数**保持绝对值不做变换：$rankScore_{nt} = theta\cdot flexctr + beta\cdot flexcxr$；
- **广告分数拆两部分**：广告内部排序分 $rankScore_{ad} = e^\delta\cdot ecpm + e^\alpha\cdot ctr + e^\beta\cdot cxr + e^\gamma\cdot gmv + embedding$（对齐 DNA，具有物理含义，**用于计算计费**）；混排分 $rankScore_{adMix} = w_1\cdot rankScore_{ad} + w_2\cdot(rankScore_{ad} - minRankScore_{ad})^2 + bias$——通过对 DNA 产出的 rankscore 做二阶多项式单调映射（保序变换）对齐到自然分数空间，**与自然候选直接排序决策最终位次**；
- 用 mse loss 约束可比性：$rankscoreMSELoss=\frac{1}{n}\sum_{i=1}^{n}(avgAdMixRankscore_i - avgOdRankscore_i)^2$。

gDNA 一期调整后的模型结构：

![[99_Attachments/图片/km-2791029005/gDNA一期调整后模型结构图.svg]]

长期方向是通过自然/广告统一模型输出、label 口径对齐，实现自然与广告候选直接用预估分公式可比。

### 3.2 Listwise 校准：一期退化

离线训练时载入 listwise 模型，对 rankscore 混排序列预估出更准的 listctr/listcvr 用于 reward 计算。但直接使用线上已有的自然 label listwise 模型训练不稳定（label 目标对不齐、预估 auc 准度仅持平），而彻底解决需要把广告点击/转化归因与自然归因对齐、统一模型训练——改动大、周期长。一期的退化方案：

- reward 中的 ctr 仍用**精排 ctr**，乘 posBias 刻画位次差异：$listctr_{ij}=pctr_i\cdot posBias_j$，$listcxr_{ij}=pcxr_i\cdot posBias_j$；
- **线上保留序列评估模块**：序列优选只需在自然目标下做同一流量上广告收益与自然体验的权衡置换，对 gDNA 生成的序列做**扰动**（乘不同系数生成多条序列、优选最优），实时适应流量变化。

### 3.3 Reward 设计：一期退化

统一使用自然和广告 target 训练不稳定（自然分数 auc/pcoc 不准、自然和广告预估分布差异大），一期先不引入自然分数，只基于广告预估分计算广告收益，同时增加广告的**自然体验损失**：用固定位参数 penaltyScore 代替，限制广告合理位次前移，平衡广告收益与自然损失：

$$
Reward=\frac{1}{n}\sum_{i=0}^n\sum_{j=0}^k\left(isAd_j\cdot allocate_{ij}\cdot target_{ij} + penaltyScore_j\right)
$$

$$
target_{ij} = \delta\cdot listctr_{ij}\cdot price + \alpha\cdot listctr_{ij} + \delta\cdot listcxr_{ij} + \gamma\cdot gmv
$$

后续优化自然预估准度、解决预估分布问题后，target 再引入自然预估分。

## 线上效果与收益来源

实验时间 20250519-20250525（7 天），美团筛选 50011 广告位，20% 流量：RPS +3.44%、CTR +2.46%、ACP -0.77%、CVR -2.16%、ROI +0.29%、曝光 +1.89%、OPM +0.32%，推全预计收入增长约 12.07 万/天。

线上实验效果：

![[99_Attachments/图片/km-2791029005/线上实验效果.png]]

**收入来源归因**是这份文档里最有说服力的分析：广告收入提升来自广告 CTR 提升（+2.46%）与曝光上涨（+1.89%），本质是 gDNA 脱离配置模版、在更大的位置空间上决策，整体分配更优。按每 5 个位次聚合的分段分析显示：实验组广告曝光分布更平均、整体位次后移（平均曝光位次 30→33），前 10 位曝光减少、20-40 位增加，且大部分位次分段上广告 CTR 好于或持平基线——排除了「位置红利」解释。

![[99_Attachments/图片/km-2791029005/位次分段-adload与ctr分析.png]]

**用户体验控制**：靠前位次的体验影响更大，实验组 TOP1 广告曝光率 -10.02%、整体覆盖 -7.37%、密度 +9.85%，出广告的请求减少，体验得到控制。

## 多路扰动与性能

- **扰动收益**：对照不接入序列评估/优选、直接吐出 gDNA 最优序列的对照组（20250427-20250430），接入多路扰动优选后 RPS +0.71%、CTR +0.41%、ROI +1.19%、RPM +0.71%——多条扰动序列能更好适应线上实时流量变化。

| 实验分组 | RPS | ACP | CTR(有效) | RPM | 商户CVR | 商户ROI |
| --- | --- | --- | --- | --- | --- | --- |
| 对照组（直接用 gDNA 结果吐出） | 142.003 | 1.662 | 3.85% | 64.05 | 7.006% | 5.75 |
| 实验组（接入校准流程） | +0.71%（143.015） | +0.30%（1.667） | +0.41%（3.87%） | +0.71%（64.50） | -0.14%（6.997%） | +1.19%（5.82） |

- **性能**：由于去掉了多位次 NeuralAuction 模块，整体耗时反而下降——午高峰 tp99 96.4ms→86.6ms（-10ms），avg 56ms→49.2ms（-7ms）。

基线与实验组午高峰耗时：

![[99_Attachments/图片/km-2791029005/基线午高峰耗时.png]]
![[99_Attachments/图片/km-2791029005/实验组午高峰耗时.png]]

## 下一步方向

文档规划的后续工作：广告转化目标从商户 CVR 迁移至支付 CVR，统一自然和广告 label 目标使预估分可比；在 label 对齐、提升 auc 精度的基础上于离线引入 listwise 模型获取预估准度收益（二期）；扩展到点评关键词/筛选、美团关键词等场景论证普适性。

> [!warning] 证据边界
> 这是美团内部 Launch Review 文档（美团筛选 50011 单场景、7 天、20% 流量），数据不可复现、未独立复核；一期多项设计为降级实现（posBias 代替 listwise 校准、固定 penaltyScore 代替自然体验损失建模），收益主要来自分配空间扩大而非外部性建模，二期引入 listwise 校准后的效果尚待验证。CVR -2.16% 与 ACP -0.77% 是明确代价，文档未展开归因分析。原文密级较高，注意不要外传具体数字。

## 关联

- [[美团到餐多位次NeuralAuction实验：逐坑位竞拍兼容多模版灵活位次]]：gDNA 所替代的上一代方案（序列生成段的多模版逐模版生成），本笔记中「去掉多位次 NeuralAuction 模块后耗时下降」正对应其被替换的部分。
- [[点评到餐NeuralAuction实验：用序特征实现计费调控并量化激励兼容代价]]：单位次 NeuralAuction 起点，三部曲的演化脉络为「单位次拍卖 → 多位次逐模版生成 → 整体拍卖混排」。
- [[DNA：可微分排序让拍卖机制进入端到端学习]]：gDNA 直接对应的原论文（KDD 2021，淘宝），「自然候选 = 零出价广告」是在其框架上的最小改动扩展。

原文档全部 8 张图已下载到 `99_Attachments/图片/km-2791029005/`（4 张 drawio 流程图导出为 SVG、4 张截图为 PNG）；公式按简化 Markdown 转写，结构以学城原文为准。
