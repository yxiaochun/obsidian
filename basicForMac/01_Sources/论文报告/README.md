---
创建日期: 2026-09-17
更新日期: 2026-09-17
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: 2025-2026 工业论文库
---

# 论文报告索引

这是 `01_Sources/论文报告` 的导航页，只记录论文文件、主线、对应卡片和当前证据状态。论文 PDF 保持原始证据，不改写、不覆盖。

## 当前状态

- 论文 PDF：31 篇，已统一保存在本目录。
- 对应模型案例卡：31 张，基本形成一一对应。
- 主线：端到端生成式架构、语义 ID 与索引、长序列与效率、多场景统一、LLM 推理注入。
- 当前主要缺口：References 层联系还不完整，部分论文只有结果记录，缺少与前序工作的可比较关系。

## 论文与卡片映射

| 论文 PDF | 对应知识卡片 | 核心证据 | 精读状态 |
| --- | --- | --- | --- |
| [[2026LinkedIn.pdf\|FeedSR]] | [[FeedSR：LinkedIn信息流序列排序模型落地]] | LinkedIn 信息流序列排序的工业落地 | 已精读建卡 |
| [[2026Tencent_GPR.pdf\|GPR]] | [[GPR：广告推荐的统一生成式预训练范式]] | 广告推荐的统一生成式预训练 | 已精读建卡 |
| [[2026Kuaishou_GR4AD.pdf\|GR4AD]] | [[GR4AD：广告生成式推荐的架构训练推理联合设计]] | 架构、训练、推理联合设计 | 已精读建卡 |
| [[2026Kuaishou_GRank.pdf\|GRank]] | [[GRank：无结构索引的目标感知生成式检索]] | 无结构索引与目标感知检索 | 已精读建卡 |
| [[2025Xiaohongshu_GenRank.pdf\|GenRank]] | [[GenRank：大规模生成式排序的工业验证]] | 大规模生成式排序工业验证 | 已精读建卡 |
| [[2025Meituan_HoMer.pdf\|HoMer]] | [[HoMer：同质化Transformer统一序列与集合上下文]] | 同质化 Transformer 统一上下文 | 已精读建卡 |
| [[2026Xiaohongshu_LASER.pdf\|LASER]] | [[LASER：分段目标注意力突破长序列延迟墙]] | 分段目标注意力与延迟控制 | 已精读建卡 |
| [[2026ByteDance_LEMUR.pdf\|LEMUR]] | [[LEMUR：端到端多模态推荐替代两阶段表征]] | 端到端多模态推荐表征 | 已精读建卡 |
| [[2025ByteDance_LONGER.pdf\|LONGER]] | [[LONGER：全局token稳定超长行为序列建模]] | 全局 token 稳定超长序列 | 已精读建卡 |
| [[2026ByteDance_MDL.pdf\|MDL]] | [[MDL：场景与任务token化统一多分布学习]] | 场景与任务 token 化 | 已精读建卡 |
| [[2026ByteDance_MERGE.pdf\|MERGE]] | [[MERGE：动态聚类的流式item索引范式]] | 动态聚类与流式索引 | 已精读建卡 |
| [[2026ByteDance_MSN.pdf\|MSN]] | [[MSN：产品键记忆实现低成本稀疏扩展]] | 产品键记忆与稀疏扩展 | 已精读建卡 |
| [[2026Meituan_MTFM.pdf\|MTFM]] | [[MTFM：免对齐的多场景推荐基础模型]] | 免对齐多场景基础模型 | 已精读建卡 |
| [[2025Meituan_MTGR.pdf\|MTGR]] | [[MTGR：保留交叉特征的工业级生成式推荐扩展]] | 保留交叉特征的扩展设计 | 已精读建卡 |
| [[2025Meituan_MTmixAtt.pdf\|MTmixAtt]] | [[MTmixAtt：自动token化与混合注意力的统一排序]] | 自动 token 化与混合注意力 | 已精读建卡 |
| [[2026ByteDance_MakeItLongKeepItFast.pdf\|MakeItLongKeepItFast]] | [[MakeItLongKeepItFast：万级序列的线性复杂度建模]] | 万级序列与线性复杂度 | 已精读建卡 |
| [[2026ByteDance_MixFormer.pdf\|MixFormer]] | [[MixFormer：稠密特征与序列建模协同扩展]] | 稠密特征与序列协同扩展 | 已精读建卡 |
| [[2025Kuaishou_OneLoc.pdf\|OneLoc]] | [[OneLoc：地理感知的本地生活生成式推荐]] | 地理感知本地生活推荐 | 已精读建卡 |
| [[2026Kuaishou_OneMall.pdf\|OneMall]] | [[OneMall：快手电商多场景统一生成式推荐]] | 电商多场景统一架构 | 已精读建卡 |
| [[2026Tencent_OneRanker.pdf\|OneRanker]] | [[OneRanker：一个模型统一生成与排序]] | 单模型统一生成与排序 | 已精读建卡 |
| [[2025Kuaishou_OneRec.pdf\|OneRec]] | [[OneRec：统一召回与排序的端到端生成式推荐]] | 端到端统一召回排序 | 已精读建卡 |
| [[2026ByteDance_OneTrans.pdf\|OneTrans]] | [[OneTrans：一个Transformer统一特征交互与序列建模]] | 统一特征交互与序列建模 | 已精读建卡 |
| [[2025JD_OxygenREC.pdf\|OxygenREC]] | [[OxygenREC：快慢思考让LLM推理进入电商推荐]] | 快慢思考引入 LLM 推理 | 已精读建卡 |
| [[2026Kuaishou_PROMISE.pdf\|PROMISE]] | [[PROMISE：过程奖励模型解锁推荐推理时扩展]] | 过程奖励与测试时扩展 | 已精读建卡 |
| [[2025ByteDance_RankMixer.pdf\|RankMixer]] | [[RankMixer：token混合让推荐模型MFU提升十倍]] | token 混合提升模型效率 | 已精读建卡 |
| [[2026Alibaba_SORT.pdf\|SORT]] | [[SORT：面向工业规模的排序Transformer系统优化]] | 工业规模排序系统优化 | 已精读建卡 |
| [[2026Meta_TAE.pdf\|TAE]] | [[TAE：基础模型加专家范式的超规模部署]] | 基础模型加专家超规模部署 | 已精读建卡 |
| [[2026ByteDance_TRM.pdf\|TRM]] | [[TRM：语义token取代itemID释放扩展潜力]] | 语义 token 替代 itemID | 已精读建卡 |
| [[2026ByteDance_TokenMixer-Large.pdf\|TokenMixer-Large]] | [[TokenMixer-Large：七十亿参数在线排序模型]] | 七十亿参数在线排序 | 已精读建卡 |
| [[2026ByteDance_UGSep.pdf\|UGSep]] | [[UGSep：用户侧计算复用降低大模型推理成本]] | 用户侧计算复用与成本控制 | 已精读建卡 |
| [[2025Meituan_UniROM.pdf\|UniROM]] | [[UniROM：广告排序的端到端统一生成架构]] | 广告排序端到端统一生成 | 已精读建卡 |

## 更新规则

1. 新论文先放本目录，再在这里登记标题、主线、对应卡片和状态。
2. 论文精读后优先更新对应知识卡片，不把结论只留在论文卡里。
3. 如果发现新论文与旧论文冲突，在两张卡片中都保留矛盾点，并写入研究问题索引。
4. 引用日报告诉我们“值得追的参考文献”，由人工确认后才升级为新论文或知识卡片。

## References 回流队列

这里的条目来自每日论文引文简报，只代表“值得人工核对”，不代表已经精读或采纳。确认后可以下载 PDF、升级为知识卡片，或合并进现有卡片。

| 日期 | 候选论文 | 来源线索 | 与现有主线的关系 | 状态 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 2026-09-17 | Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations | [arXiv:2402.17152](https://arxiv.org/abs/2402.17152)、ICML 2024 | 大规模序列转导与生成式推荐基础架构，可作为 [[LONGER：全局token稳定超长行为序列建模]]、[[RankMixer：token混合让推荐模型MFU提升十倍]]、[[TAE：基础模型加专家范式的超规模部署]] 的对照 | 待人工核对 | 优先精读，补入「端到端生成式架构」主线 |
| 2026-09-17 | Wukong: Towards a Scaling Law for Large-Scale Recommendation | [arXiv:2403.02545](https://arxiv.org/abs/2403.02545)、ICML 2024 | 推荐模型扩展规律的直接前序，可对照 [[Scaling Law在工业推荐系统的落地路径]]、[[RankMixer：token混合让推荐模型MFU提升十倍]]、[[TokenMixer-Large：七十亿参数在线排序模型]] | 待人工核对 | 核对扩展曲线、任务口径和特征空间 |
| 2026-09-17 | Scaling Law of Large Sequential Recommendation Models | RecSys 2024，pp. 444-453 | 序列推荐模型扩展规律，可对照 [[LONGER：全局token稳定超长行为序列建模]]、[[MakeItLongKeepItFast：万级序列的线性复杂度建模]] | 待人工核对 | 核对是否使用纯 ID 设定，确认与工业长序列方案差异 |
| 2026-09-17 | Scaling Sequential Recommendation Models with Transformers | [arXiv:2412.07585](https://arxiv.org/abs/2412.07585)、SIGIR 2024 | Transformer 序列推荐扩展，可对照 [[LONGER：全局token稳定超长行为序列建模]]、[[LASER：分段目标注意力突破长序列延迟墙]] | 待人工核对 | 核对 embedding bottleneck 与架构扩展假设 |
| 2026-09-17 | MARM: Unlocking the Future of Recommendation Systems through Memory Augmentation and Scalable Complexity | [arXiv:2411.09425](https://arxiv.org/abs/2411.09425) | 内存增强与 cache scaling，可对照 [[超长行为序列建模的工程解法]]、[[UGSep：用户侧计算复用降低大模型推理成本]] | 待人工核对 | 核对缓存规模、命中率和线上成本收益 |

### 回流优先级

1. **P0：HSTU**。它是 2024 年生成式推荐和大规模序列转导的关键前序，现有 31 篇论文中多篇使用 HSTU 作为基础块或基线，应当单独建卡。
2. **P1：Wukong 与两篇序列推荐 Scaling Law**。三者共同回答推荐模型和序列推荐模型能否 scaling，以及收益曲线受什么约束。
3. **P2：MARM**。它主要补长序列工程路线中的“内存换计算”分支，价值高但优先级略低于架构和扩展规律主线。
