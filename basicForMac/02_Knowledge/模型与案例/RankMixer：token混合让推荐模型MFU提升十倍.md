---
创建日期: 2026-09-15
更新日期: 2026-09-26
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[2025ByteDance_RankMixer.pdf]]"
状态: 待复核
证据强度: 单篇工业论文证据
精读判定: 精读
待验证问题:
  - "不同硬件和更严格 QPS 下，高 FLOPs 架构的时延与稳定性如何？"
  - "Semantic tokenization 的特征分组规则和敏感性如何影响结果？"
  - "SparseMoE 超过 8 倍稀疏扩展在 10B 规模和更长训练周期下是否稳定？"
---

# RankMixer：token混合让推荐模型MFU提升十倍

## 精读结论

RankMixer 用参数无关的 Multi-head Token Mixing 做跨特征交互，用 Per-token FFN 扩大容量并隔离异质特征子空间，再通过 DTSI 与 ReLU 路由稳定 SparseMoE。它把“参数增长”和“FLOPs 增长”解耦，把“理论 FLOPs”和“实际推理时延”解耦：抖音 Feed 的 15.8M 参数 DLRM+DCN 基线换成 1.1B 参数 RankMixer 后，FLOPs 提升 20.7 倍，但 MFU 从 4.47% 提升到 44.57%，fp32 改为 fp16，线上时延从 14.5ms 变为 14.3ms。抖音主 App 全量 A/B 显示活跃天数 +0.2908%、总时长 +1.0836%。

## 七问笔记

| 问题 | 回答 |
| --- | --- |
| 研究背景 | LLM 的缩放经验推动推荐排序模型扩大，但工业系统必须满足低时延和高 QPS；继承自 CPU 时代的 DLRM 特征交叉模块在现代 GPU 上常是 memory-bound，MFU 只有单位数百分比。 |
| 研究目的 | 找到一个同时适配推荐异质特征空间、GPU 并行性和工业成本约束的排序稠密主干，使参数能扩大而不破坏在线时延。 |
| 创新点 | 用无参数的 Multi-head Token Mixing 替代自注意力做跨 token 特征交互，并用 Per-token FFN 让每个语义 token 拥有独立参数，从而在相同或相近计算量下获得更强的缩放能力。 |
| 研究方法 | 输入先按语义分组 tokenization，再经过 L 层 RankMixer block，每层包含 Token Mixing、残差、LayerNorm、Per-token FFN、残差和 LayerNorm；SparseMoE 版本用 ReLU 路由和 Dense-Training-Sparse-Inference 缓解专家欠训练。 |
| 实验数据 | 离线使用抖音推荐线上日志和反馈标签，覆盖超过 300 个特征、数十亿用户 ID、数亿视频 ID、每日万亿级记录和两周数据；线上在抖音 Feed 与广告场景做 A/B。 |
| 结果结论 | 约 100M 稠密参数组中，RankMixer-100M 相对 DLRM-MLP 的 Finish AUC +0.64%、Finish UAUC +0.72%、Skip AUC +0.86%、Skip UAUC +1.33%；RankMixer-1B 进一步达到 +0.95%、+1.22%、+1.25%、+1.82%。广告 A/B 中 ΔAUC +0.73%、ADVV +3.90%。 |
| 总体评价 | 论文把架构设计、MFU、量化、离线缩放、消融和两类在线场景串联起来，因果链较强；但核心数据来自抖音私有生产和部署环境，特征分组、硬件、超参和 SparseMoE 细节披露不足，外部可复现性有限。 |

## 方法拆解

- **Semantic tokenization**：不把每个字段都当作一个 token，而是先把用户画像、视频特征、序列特征和交叉特征等语义相近的 embedding 分组拼接，再切分并投影到固定维度 D，得到 T 个 feature token。这样避免数百个字段导致单 token 计算碎片化，也避免全部压缩成一个 token 后高频字段压制长尾字段。
- **Multi-head Token Mixing**：每个 token 先分成 H 个 head；同一 head 下的 T 个向量拼接重组成 T 个新 token，随后做残差和 LayerNorm。作者设 H=T，使 token 数量不变。这个混合是参数无关的置换重组，因此能保持 Transformer 式高并行，同时避免自注意力的二次交互矩阵、显存读写和异质 ID 空间内积学习困难。
- **Per-token FFN**：每个 token 使用一组两层独立 MLP，隐藏维度为 kD，激活为 GELU。与所有 token 共享一个 FFN 相比，它把参数按 token/特征子空间隔离，缓解高频字段主导；单 token 前向计算复杂度不变，但总参数量随 T 增长。
- **SparseMoE**：Per-token FFN 可替换为 SparseMoE。传统 Top-k 路由对所有 token 一样分配预算，且专家数急剧增加后容易出现专家不均衡和 dying experts。RankMixer 采用 ReLU gate 加自适应 L1 正则，让高信息 token 激活更多专家；DTSI 在训练期用稠密路由保证专家获得梯度，在推理期切换到稀疏路由降低成本。
- **扩展方向**：稠密版本可用 T、D、L 和专家数 E 四个方向扩展。论文给出近似公式 `#Param ≈ 2kLTD²`、`FLOPs ≈ 4kLTD²`；100M 和 1B 配置分别为 D=768、T=16、L=2 和 D=1536、T=32、L=2。作者观察到总参数量主导质量，但更大的 D 会形成更大的 GEMM 形状并提高 MFU。

## 实验证据

### 离线对比

| 模型 | 参数量 | 每 batch 512 FLOPs | 相对 DLRM-MLP 的效果 |
| --- | ---: | ---: | --- |
| DLRM-MLP | 8.7M | 52G | Finish AUC 0.8554，Skip AUC 0.8124 |
| DLRM-MLP-100M | 95M | 185G | Finish AUC +0.15%，Skip AUC +0.15% |
| DCNv2 | 22M | 170G | Finish AUC +0.13%，Skip AUC +0.15% |
| RDCN | 22.6M | 172G | Finish AUC +0.09%，Skip AUC +0.10% |
| MoE | 47.6M | 158G | Finish AUC +0.09%，Skip AUC +0.08% |
| AutoInt | 19.2M | 307G | Finish AUC +0.10%，Skip AUC +0.12% |
| DHEN | 22M | 158G | Finish AUC +0.18%，Skip AUC +0.36% |
| HiFormer | 116M | 326G | Finish AUC +0.48% |
| Wukong | 122M | 442G | Finish AUC +0.29%，Skip AUC +0.49% |
| RankMixer-100M | 107M | 233G | Finish AUC +0.64%，Skip AUC +0.86% |
| RankMixer-1B | 1.1B | 2.1T | Finish AUC +0.95%，Skip AUC +1.25% |

训练框架采用混合分布式：稀疏嵌入异步更新，稠密部分同步更新；所有模型保持优化器一致，稠密部分用 RMSProp（learning rate 0.01），稀疏部分用 Adagrad。

### 组件消融

| 设置 | ΔAUC |
| --- | ---: |
| 移除 Multi-head Token Mixing | -0.50% |
| Per-token FFN 换成共享 FFN | -0.31% |
| 移除残差连接 | -0.07% |
| 移除 LayerNorm | -0.05% |

Token-to-FFN 路由消融中，All-Concat-MLP -0.18%，All-Share -0.25%；Self-Attention 只降低 0.03%，但参数增加 16%、FLOPs 增加 71.8%。这说明注意力在精度上接近，但计算代价明显更高。

### 线上成本与 A/B

| 指标 | OnlineBase-16M | RankMixer-1B |
| --- | ---: | ---: |
| 参数量 | 15.8M | 1.1B |
| FLOPs | 107G | 2106G |
| FLOPs/参数 | 6.8 | 1.9 |
| MFU | 4.47% | 44.57% |
| 精度 | fp32 | fp16 |
| 时延 | 14.5ms | 14.3ms |

抖音主 App 整体用户中，活跃天数 +0.2908%，总时长 +1.0836%，点赞 +2.3852%，完播 +1.9874%，评论 +0.7886%。抖音 lite 对应为 +0.1968%、+0.9869%、+1.1318%、+2.0744%、+1.1338%。低活跃用户增益最大，抖音主 App 低活跃组活跃天数 +1.7412%、总时长 +3.6434%、点赞 +8.1641%。广告场景用 1B 模型替换排序稠密部分后，ΔAUC +0.73%，ADVV +3.90%。

## 关键图表解读

- **Figure 1**：展示从数百个特征 embedding 到语义 token、RankMixer block、SparseMoE、mean pooling 和多任务输出的完整链路；Token Mixing 的要点是先分头再跨 token 重排，SparseMoE 通过 ReLU Routing 和 gate loss 控制专家激活。
- **Figure 2**：以对数横轴比较不同模型的 Finish AUC 增益与参数/FLOPs 的缩放曲线。RankMixer 在两条曲线上都更陡；Wukong 的参数曲线也较陡，但 FLOPs 曲线显示其计算增长更快，因此在 AUC-FLOPs 视角下优势缩小。
- **Figure 3**：比较专家激活率从 1、1/2、1/4 到 1/8 时的 AUC。DTSI + ReLU 路由在激进稀疏化下接近稠密模型精度，vanilla SMoE 单调退化，加入负载均衡损失只能部分缓解。论文据此提出参数/显存扩展超过 8 倍、吞吐提升 50%。
- **Figure 4**：展示不同 token 在两层 SparseMoE 中的激活专家比例。DTSI 保证训练期专家充分获得梯度，ReLU 路由让激活比例随 token 信息量动态变化，从而缓解部分专家长期不被激活的问题。

## 批判性分析

- **为什么值得做**：单纯扩大 DLRM-MLP 到 100M 只带来 0.15% Finish AUC 提升，说明工业推荐的瓶颈不是“有参数就能扩”，而是架构必须同时服务特征交互质量、GPU 利用率和在线成本。
- **为什么不用自注意力**：推荐特征 token 来自异质 ID 和统计特征空间，不同 token 的内积相似度不像自然语言 token 那样天然可比；注意力权重矩阵还带来额外显存与读写。消融显示 Self-Attention 精度只略低，但 FLOPs +71.8%，因此参数无关混合在工业预算下更合理。
- **为什么用 Per-token FFN**：共享 FFN 容易被高频字段主导。逐 token 独立参数把容量分配到特征子空间，且能与 SparseMoE 自然衔接。代价是参数量随 token 数增长，所以必须配合高 MFU 和稀疏化控制成本。
- **实验遗漏**：论文没有公开特征分组的构建规则，也没有单独消融不同 token 数、分组方式、语义聚类稳定性和错配分组的影响；线上对照硬件型号、QPS、批处理策略、缓存和冷启动分布披露有限。A/B 显著性方向清楚，但外部读者难以复现同一置信区间。
- **换位思考**：若重构这项工作，我会先固定 FLOPs 预算并补充 FLOPs-matched 的 HiFormer、Wukong、DHEN 曲线，再报告参数、显存、时延和吞吐四个轴；对 SparseMoE 补充专家数、λ、激活率方差、训练后期路由漂移和 10B 版本的稳定性实验。
- **优点**：问题定义贴近部署约束；方法从推荐异质性出发而不是简单套 Transformer；证据链覆盖离线缩放、组件消融、SparseMoE、广告与 Feed 两类在线场景。
- **不足**：核心结论依赖私有万亿级数据、内部硬件和工程栈；Semantic tokenization 的“语义分组”缺少可迁移定义；1B 稠密模型 FLOPs 仍是基线 20.7 倍，收益依赖 MFU、fp16 和部署优化的组合；长期 A/B 结果仍在提升、未收敛，说明报告点依赖实验截止时间。

## 复现清单

- **数据**：抖音推荐线上日志与反馈标签，超过 300 个特征，两周数据，包含数值、ID、交叉和序列特征；私有数据不可直接获得。
- **任务与指标**：Finish/Skip 二分类，报告 AUC 和 User-level AUC，同时记录参数量、FLOPs 和 MFU。论文将 0.0001 AUC 提升视为可信显著提升。
- **环境**：数百块 GPU，稀疏嵌入异步更新、稠密部分同步更新的混合分布式训练；推理使用 fp16。论文未给出具体 GPU 型号、网络拓扑和部署集群配置。
- **关键超参**：100M 配置为 D=768、T=16、L=2；1B 配置为 D=1536、T=32、L=2；稠密部分 RMSProp learning rate 0.01，稀疏部分 Adagrad。
- **缺失信息**：Semantic tokenization 的特征组映射、SparseMoE 的专家数、k、λ、激活预算、训练采样细节、完整验证协议和线上硬件/QPS 配置。
- **代码**：论文未提供官方实现链接。
- **改良方向**：先做特征分组敏感性扫描；再做 FLOPs-matched 与 latency-matched 双对照；测试不同 GPU 架构和更低精度；把 DTSI + ReLU 路由扩展到多任务、多场景和 10B 稠密/稀疏混合配置。

## 术语与句式积累

- **MFU**：Model Flops Utilization，实际消耗的模型 FLOPs 占硬件理论 FLOPs 的比例，用于衡量架构是否真正利用算力。
- **Token Mixing**：将 token 分头后按同一 head 跨 token 重排拼接，形成参数无关的跨特征信息交换。
- **Per-token FFN**：每个 token 一组独立 MLP，与共享 FFN 相反，用于隔离特征子空间。
- **DTSI**：Dense-Training-Sparse-Inference，训练期稠密激活专家以缓解欠训练，推理期稀疏激活以降低成本。
- **可复用句式**：“参数扩展只有在参数增长、计算增长、MFU 和精度可量化分离时才有工程意义。”；“异质特征空间的交互不应默认套用自然语言中的自注意力相似度。”

## 关联

- [[TokenMixer-Large：七十亿参数在线排序模型]]：同为工业排序模型扩展，重点比较深栈化路径与 RankMixer 的 token 混合路径。
- [[UGSep：用户侧计算复用降低大模型推理成本]]：从用户侧复用角度补充推理成本优化。
- [[堆叠因子分解机实现推荐系统缩放定律：Wukong 在 100+ GFLOPexample 下持续提升质量|Wukong 的缩放定律实验]]：RankMixer 离线对比中的关键缩放基线。
- [[LONGER：全局token稳定超长行为序列建模]]：论文中的序列模块相关工作。
- [[Scaling Law在工业推荐系统的落地路径]]：把本文的 MFU 与成本解耦证据放入工业推荐缩放框架。
