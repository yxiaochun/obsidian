---
创建日期: 2026-09-19
更新日期: 2026-09-19
类型: 复盘
标签:
  - 生成式推荐
  - 推荐系统
来源: "2025JD_OxygenREC.pdf"
---

# 2026-09-19｜论文引文简读

## 底稿论文

- **论文**：OxygenREC: An Instruction-Following Generative Framework for E-commerce Recommendation
- **一句话主题**：OxygenREC 是京东提出的工业级生成式推荐框架，通过快慢思考架构、指令引导检索、Q2I 损失和多场景对齐策略，实现端到端生成式推荐与“一次训练、多场景部署”。
- **References 总量**：72
- **近 5 年候选**：47
- **今日精选**：5

## 精选引文

### 1. Recommender Systems with Generative Retrieval（2023）

- **来源**：Advances in Neural Information Processing Systems, 36:10299–10315, 2023
- **相关性**：生成式召回/排序/5
- **核心思路**：将推荐建模为生成式检索任务，直接生成目标物品标识，而非从固定候选池中打分。
- **与源论文的关系**：源论文将生成式推荐作为核心范式，并在此基础上引入指令跟随与快慢思考，该文是生成式检索方向的基础性工作之一。
- **研究价值**：为理解生成式推荐如何替代传统多阶段级联推荐提供早期范式参考，有助于定位 OxygenREC 在生成式检索谱系中的位置。
- **需要核实**：候选条目仅给出标题、作者和发表信息，未提供具体方法细节、实验设置和结论，需人工核实其与 OxygenREC 在技术路线上的具体差异。

### 2. LightLM: A Lightweight Deep and Narrow Language Model for Generative Recommendation（2023）

- **来源**：arXiv preprint arXiv:2310.17488, 2023
- **相关性**：LLM 推荐、生成式推荐/5
- **核心思路**：提出轻量级深度窄语言模型用于生成式推荐，试图在语言模型能力与推荐效率之间取得平衡。
- **与源论文的关系**：源论文同样关注生成式推荐中的效率与工业部署问题，LightLM 的轻量化思路可与 OxygenREC 的快思考编码器-解码器骨干形成对照。
- **研究价值**：有助于研究生成式推荐模型在工业延迟约束下的轻量化设计，以及语言模型结构对推荐生成质量的影响。
- **需要核实**：候选条目未提供模型结构、训练目标和实验结果的详细信息，需人工核实其是否使用语义 ID、是否面向工业场景。

### 3. HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling（2024）

- **来源**：arXiv preprint arXiv:2409.12740, 2024
- **相关性**：LLM 推荐、生成式推荐/5
- **核心思路**：使用分层大语言模型分别进行物品建模和用户建模，以增强序列推荐能力。
- **与源论文的关系**：源论文强调 LLM 驱动的用户和物品输入建模，HLLM 的分层 LLM 思路与 OxygenREC 的 LLM 驱动输入表示有直接关联。
- **研究价值**：为研究 LLM 如何用于用户/物品表示学习、以及分层建模对推荐效果的影响提供参考。
- **需要核实**：候选条目未说明其是否采用生成式解码、是否使用语义 ID，需人工核实其与 OxygenREC 在生成式推荐框架上的异同。

### 4. IDGenRec: LLM-RecSys Alignment with Textual ID Learning（2024）

- **来源**：Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 355–364, 2024
- **相关性**：语义 ID、LLM 推荐/5
- **核心思路**：通过文本 ID 学习实现 LLM 与推荐系统之间的对齐，使物品标识具备语义可解释性。
- **与源论文的关系**：源论文涉及语义对齐机制和指令-物品一致性，IDGenRec 的文本 ID 学习与语义 ID 思路可为理解 OxygenREC 的语义对齐提供背景。
- **研究价值**：有助于研究语义 ID 在生成式推荐中的作用，以及如何通过文本标识提升 LLM 与推荐系统的对齐效果。
- **需要核实**：候选条目未提供具体 ID 构造方式、对齐损失和实验结论，需人工核实其与 OxygenREC 的 Q2I 损失和 IGR 机制的关系。

### 5. OneRec Technical Report（2025）

- **来源**：arXiv preprint arXiv:2506.13695, 2025
- **相关性**：生成式推荐、工业推荐系统/5
- **核心思路**：提出面向工业场景的生成式推荐系统 OneRec，探索端到端生成式推荐在真实业务中的落地。
- **与源论文的关系**：源论文在相关工作中提及 OneRec 系列，并将其作为生成式推荐工业实践的代表；OxygenREC 同样面向工业级生成式推荐，二者在目标场景上高度相关。
- **研究价值**：为比较不同工业生成式推荐框架的架构选择、训练策略和部署方案提供重要参照。
- **需要核实**：候选条目仅给出标题和 arXiv 链接，未提供 OneRec 的具体技术细节和实验结果，需人工核实其与 OxygenREC 在快慢思考、多场景对齐等方面的差异。

## 使用方式

- 这是自动生成的引文简读，先按相关性挑 1～2 篇核对原文。
- 确认有价值后，再升级成知识卡片或选题；本文件只作为每日线索。
- 如某条内容信息不足，不要直接引用其结论，先查原文。
