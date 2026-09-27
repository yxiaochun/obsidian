---
创建日期: 2026-09-24
更新日期: 2026-09-24
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: "[[DeepSeekMoE Towards Ultimate Expert Specialization in Mixture-of-Experts Languag]]"
状态: 待复核
证据强度: 单篇论文证据
---

# DeepSeekMoE：细粒度专家分割与共享专家隔离实现极致专家专业化

## 论文信息

- **论文标题**：DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
- **一句话结论**：DeepSeekMoE 通过细粒度专家分割和共享专家隔离两大策略，在相同专家参数和计算成本下显著提升专家专业化程度，2B 模型即可媲美 2.9B GShard，16B 模型以约 40% 计算量达到 LLaMA2 7B 水平。
- **发现来源**：2025Meituan_MTmixAtt
- **历史被引底稿论文数**：4

## 贡献

- 提出 DeepSeekMoE 架构，包含细粒度专家分割和共享专家隔离两大策略，旨在实现极致专家专业化。
- 在 2B 规模验证 DeepSeekMoE 优势：2B 模型超越 GShard 2B，并匹配 GShard 2.9B（1.5 倍专家参数和计算量），接近同等总参数稠密模型的上界性能。
- 将 DeepSeekMoE 扩展至 16B，在 2T token 上训练，以约 40% 计算量达到 DeepSeek 7B 和 LLaMA2 7B 的相当性能，并在 Open LLM Leaderboard 上大幅超越同激活参数量模型。
- 初步扩展至 145B，验证其相对 GShard 架构的持续优势，以 28.5%（甚至 18.2%）计算量达到 DeepSeek 67B 的相当性能。
- 成功对 DeepSeekMoE 16B 进行监督微调，构建对齐聊天模型，展示其适应性和多功能性。
- 公开 DeepSeekMoE 16B 模型检查点，可在单张 40GB 显存 GPU 上部署，无需量化。

## 方法要点

- 细粒度专家分割：在保持专家参数和计算成本不变的前提下，将每个专家 FFN 的中间隐藏维度缩小为原来的 1/m，从而将专家分割为 mN 个细粒度专家，并相应激活 mK 个专家，以增强激活专家组合的灵活性。
- 共享专家隔离：隔离 Ks 个专家作为始终激活的共享专家，用于捕获和整合跨上下文的通用知识，同时将路由专家的激活数量减少 Ks 以保持计算成本不变，从而减轻路由专家间的参数冗余。
- 负载均衡考虑：采用专家级平衡损失（LExpBal）防止路由崩溃，设备级平衡损失（LDevBal）促进设备间计算均衡；实践中设置较小的专家级平衡因子和较大的设备级平衡因子。
- 完整 DeepSeekMoE 公式：MoE 层输出为共享专家输出与路由专家加权输出之和，路由专家从 mN-Ks 个专家中选取 top-(mK-Ks) 个激活。
- 训练基础设施：基于 HAI-LLM 框架，集成张量并行、ZeRO 数据并行、PipeDream 流水线并行和专家并行，并开发 CUDA/Triton 内核优化门控和专家计算。

## 实验与证据

- 验证实验设置：训练数据从 DeepSeek-AI 多语言语料采样 100B token，BPE 词表 8K；模型 9 层 Transformer，隐藏维度 1280，10 个注意力头（每头 128 维），总专家参数为标准 FFN 的 16 倍，激活专家参数为标准 FFN 的 2 倍，总参数约 2B，激活参数约 0.3B；使用 AdamW 优化器，最大学习率 1.08e-3，批次大小 2K，序列长度 2K，训练 25,000 步达到 100B token。
- 验证实验基线：Dense（0.2B 总参数）、Hash Layer（2.0B 总参数，0.2B 激活参数）、Switch Transformer（2.0B 总参数，0.2B 激活参数）、GShard（2.0B 总参数，0.3B 激活参数）、DeepSeekMoE（2.0B 总参数，0.3B 激活参数，1 共享专家+63 路由专家，每专家为标准 FFN 的 0.25 倍）。
- 验证实验结果（100B token 训练后）：DeepSeekMoE 在 Pile 损失 1.808，HellaSwag 54.8%，PIQA 72.3%，ARC-easy 49.4%，ARC-challenge 34.3%，RACE-middle 44.0%，RACE-high 31.7%，HumanEval 4.9%，MBPP 2.2%，TriviaQA 16.6%，NaturalQuestions 5.7%；均优于 GShard（Pile 1.867，HellaSwag 50.5%，PIQA 70.6%，ARC-easy 43.9%，ARC-challenge 31.6%，RACE-middle 42.1%，RACE-high 30.4%，HumanEval 3.7%，MBPP 0.2%，TriviaQA 10.2%，NaturalQuestions 3.2%）。
- 与 GShard×1.5 对比（100B token）：DeepSeekMoE 达到相当性能，Pile 损失均为 1.808；GShard×1.5 总专家参数 2.83B，激活专家参数 0.35B，每 2K token FLOPs 5.8T；DeepSeekMoE 总专家参数 1.89B，激活专家参数 0.24B，FLOPs 4.3T。
- 与 Dense×16 对比（100B token）：DeepSeekMoE 接近 Dense×16 性能，Pile 损失 1.808 vs 1.806；Dense×16 总专家参数 1.89B，激活专家参数 1.89B，FLOPs 24.6T；DeepSeekMoE 激活专家参数 0.24B，FLOPs 4.3T。
- 消融实验：共享专家隔离（基于 GShard 隔离 1 个共享专家）在多数基准上提升性能；细粒度专家分割（将专家分割为 2 或 4 个更小专家，总专家数 32 或 64）持续提升整体性能；共享与路由专家比例实验：1、2、4 个共享专家分别达到 Pile 损失 1.808、1.806、1.811，最终选择 1:3 比例。
- 专家专业化分析：禁用 top 路由专家时 DeepSeekMoE 比 GShard×1.5 更敏感，表明路由专家间冗余更低；禁用共享专家并激活一个额外路由专家后，Pile 损失从 1.808 升至 2.414，表明共享专家不可替代；仅激活 4 个路由专家时 DeepSeekMoE 达到与 GShard 相当的 Pile 损失；从头训练仅激活 3 个路由专家的模型，在相同总专家参数和一半激活专家参数下仍超越 GShard。
- DeepSeekMoE 16B 设置：28 层 Transformer，隐藏维度 2048，16 个注意力头（每头 128 维），除第一层外所有 FFN 替换为 MoE 层，每层 2 共享专家+64 路由专家，每专家为标准 FFN 的 0.25 倍，每 token 路由至 2 共享专家和 6 路由专家，总参数约 16.4B，激活参数约 2.8B；训练 2T token，最大学习率 4.2e-4，批次大小 4.5K，序列长度 4K，训练 106,449 步。
- DeepSeekMoE 16B 与 DeepSeek 7B 对比（均训练 2T token）：DeepSeekMoE 16B 以 40.5% 计算量达到相当性能；Pile BPB 0.74 vs 0.75，HellaSwag 77.1% vs 75.4%，PIQA 80.2% vs 79.2%，ARC-easy 68.1% vs 67.9%，ARC-challenge 49.8% vs 48.1%，RACE-middle 61.9% vs 63.2%，RACE-high 46.4% vs 46.5%，DROP 32.9% vs 34.9%，GSM8K 18.8% vs 17.4%，MATH 4.3% vs 3.3%，HumanEval 26.8% vs 26.2%，MBPP 39.2% vs 39.0%，TriviaQA 64.8% vs 59.7%，NaturalQuestions 25.5% vs 22.2%，MMLU 45.0% vs 48.2%，WinoGrande 70.2% vs 70.5%，CLUEWSC 72.1% vs 73.1%，CEval 40.6% vs 45.0%，CMMLU 42.5% vs 47.2%，CHID 89.4% vs 89.3%。
- DeepSeekMoE 16B 与 LLaMA2 7B 对比（均训练 2T token）：DeepSeekMoE 16B 以 39.6% 计算量在多数基准上超越 LLaMA2 7B；Pile BPB 0.74 vs 0.76，HellaSwag 77.1% vs 75.6%，PIQA 80.2% vs 78.0%，ARC-easy 68.1% vs 69.1%，ARC-challenge 49.8% vs 49.0%，RACE-middle 61.9% vs 60.7%，RACE-high 46.4% vs 45.8%，DROP 32.9% vs 34.0%，GSM8K 18.8% vs 15.5%，MATH 4.3% vs 2.6%，HumanEval 26.8% vs 14.6%，MBPP 39.2% vs 21.8%，TriviaQA 64.8% vs 63.8%，NaturalQuestions 25.5% vs 25.5%，MMLU 45.0% vs 45.8%，WinoGrande 70.2% vs 69.6%，CLUEWSC 72.1% vs 64.0%，CEval 40.6% vs 33.9%，CMMLU 42.5% vs 32.6%，CHID 89.4% vs 37.9%。
- 对齐实验：对 DeepSeekMoE 16B、DeepSeek 7B、LLaMA2 7B 使用相同 1.4M SFT 数据微调，批次大小 1024，训练 8 轮，序列长度 4K，学习率 1e-5；DeepSeekMoE Chat 16B 以约 40% 计算量在多数基准上达到与 7B 稠密模型相当或更好性能，代码生成显著超越 LLaMA2 SFT 7B（HumanEval 45.7% vs 35.4%，MBPP 46.2% vs 27.8%），中文基准全面超越 LLaMA2 SFT 7B（CLUEWSC 68.2% vs 48.4%，CEval 40.0% vs 35.1%，CMMLU 49.3% vs 36.9%）。
- DeepSeekMoE 145B 初步实验：62 层 Transformer，隐藏维度 4096，32 个注意力头（每头 128 维），每层 4 共享专家+128 路由专家，每专家为标准 FFN 的 0.125 倍，每 token 路由至 4 共享专家和 12 路由专家，总参数约 144.6B，激活参数约 22.2B；训练 245B token，最大学习率 3.0e-4，批次大小 4.5K，序列长度 4K，训练 13,000 步。
- DeepSeekMoE 145B 与 DeepSeek 67B、GShard 137B 对比（均训练 245B token）：DeepSeekMoE 145B 以 28.5% 计算量达到与 DeepSeek 67B 相当性能；Pile 损失 1.876 vs 1.905 vs 1.961，HellaSwag 75.8% vs 74.8% vs 72.0%，PIQA 80.7% vs 79.8% vs 77.6%，ARC-easy 69.7% vs 69.0% vs 64.0%，ARC-challenge 48.8% vs 50.4% vs 45.8%，RACE-middle 62.1% vs 63.2% vs 59.2%，RACE-high 45.5% vs 46.9% vs 43.5%，DROP 27.8% vs 27.5% vs 21.6%，GSM8K 12.2% vs 11.8% vs 6.4%，MATH 3.1% vs 2.1% vs 1.6%，HumanEval 19.5% vs 23.8% vs 17.7%，MBPP 33.2% vs 33.6% vs 27.6%，TriviaQA 61.1% vs 57.2% vs 52.5%，NaturalQuestions 25.0% vs 22.6% vs 19.0%，MMLU 39.4% vs 45.1% vs 26.3%，WinoGrande 71.9% vs 70.7% vs 67.6%，CLUEWSC 71.9% vs 69.1% vs 65.7%，CEval 37.1% vs 40.3% vs 26.2%，CMMLU 35.9% vs 40.6% vs 25.4%，CHID 90.3% vs 88.5% vs 86.9%。
- DeepSeekMoE 142B（半激活）对比：2 共享专家+128 路由专家，仅激活 6 路由专家，总参数 142.3B，激活参数 12.2B，FLOPs 374.6T；以 18.2% 计算量达到与 DeepSeek 67B 相当性能，并超越 GShard 137B。

## 局限与开放问题

- DeepSeekMoE 16B 在多项选择题任务（如 MMLU、CEval、CMMLU）上表现不如 DeepSeek 7B，归因于其注意力参数较少（约 0.5B vs DeepSeek 7B 的 2.5B），表明注意力容量与多项选择任务性能正相关。
- DeepSeekMoE 145B 仍处于初步研究阶段，仅训练 245B token，尚未完成完整训练和最终版本。
- 细粒度专家分割在 16B 规模未采用更细粒度（如 0.125 倍），因专家过小可能降低计算效率；更大规模可继续细化。
- 共享专家与路由专家比例实验显示不同比例对性能影响不显著，但仅测试了 1、2、4 个共享专家，未探索更广泛比例。
- 设备级平衡损失在验证实验中未使用（所有专家部署在单 GPU），仅在 145B 规模使用，其通用性需进一步验证。

## 待验证问题

- 如何进一步提升 DeepSeekMoE 在多项选择题任务上的表现，以弥补注意力参数不足的短板？
- 在更大规模（如 145B 以上）下，更细粒度的专家分割（如 0.0625 倍）是否能持续带来性能提升？
- 共享专家与路由专家的最优比例是否随模型规模、任务类型或训练数据分布变化？
- DeepSeekMoE 的专家专业化程度能否通过其他可解释性方法（如知识神经元分析）进一步量化？
- 设备级平衡损失与专家级平衡损失的最优组合策略在不同并行配置下如何确定？
- DeepSeekMoE 架构能否有效迁移到多模态或非语言任务？

## 与知识库的关联

- 该论文与知识库中 MoE 相关研究（如 GShard、Switch Transformer、Hash Layer、DeepSeek 系列）紧密相关，提供了细粒度专家分割和共享专家隔离两种提升专家专业化的通用策略，可被后续 MoE 架构设计、推荐系统大规模稀疏模型、以及美团 MTmixAtt 等涉及混合专家或注意力机制的工作引用和借鉴。

## 核对提醒

- 本笔记严格依据论文全文提取，所有数字和实验条件均来自原文；但需注意：1）论文中部分实验（如 145B）为初步结果，仅训练 245B token，非最终结论；2）消融实验的具体数值（如图 3 性能归一化曲线）未在文本中给出精确数字，仅描述趋势；3）共享专家比例实验的 Pile 损失 1.808、1.806、1.811 对应 1、2、4 个共享专家，但未说明其他超参数是否完全一致；4）Open LLM Leaderboard 的具体分数未在文本中列出，仅以图 1 展示；5）部分基准（如 BBH、DROP）在验证实验中未使用，仅在 16B 和 145B 评估中出现。
