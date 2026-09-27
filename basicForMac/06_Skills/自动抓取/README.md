---
创建日期: 2026-09-15
更新日期: 2026-09-25
类型: 笔记
标签:
  - 生成式推荐
来源: Horizon 项目配置
---

# Horizon 定时抓取配置

Horizon 仓库位于 `~/Documents/Horizon`。定时抓取任务已取消，目前需要手动运行。运行后：

- 中文日报 → `00_Inbox/自动抓取/YYYY-MM-DD｜Horizon中文日报.md`
- 英文日报（原始抓取材料）→ `00_Inbox/自动抓取/YYYY-MM-DD｜Horizon英文日报.md`

复制进知识库后，脚本会调用 `06_Skills/规则/format_obsidian_markdown.sh`，按
`/Users/yangxiaochun/Documents/obsidian/obsidian-skills/skills/obsidian-markdown/SKILL.md`
对当日文件做一次 Obsidian Markdown 格式化。如果格式化失败，原始摘要仍会保留。

Horizon 只负责信息进入和第一轮摘要。正式的每日简报改由 `06_Skills/论文引文简报` 在每天 7:00 生成。

## 信息源与关键词

当前信息源定义在 `config.genrec.json`：

- arXiv cs.IR（推荐系统方向论文）
- Hugging Face / Google Research 博客
- Hacker News（门槛 50 分）
- Reddit r/MachineLearning
- OSS Insight 趋势仓库，关键词：recommender、recommendation、retrieval、rag、llm
- Google News，检索词：generative recommendation / LLM recommender system

修改信息源或关键词后，运行下方同步命令把配置复制进 Horizon 仓库再生效。

## 常用命令

```bash
# 同步配置到 Horizon 仓库
cp "06_Skills/自动抓取/config.genrec.json" ~/Documents/Horizon/data/config.json
cp -R "06_Skills/自动抓取/profile-genrec" ~/Documents/Horizon/profiles/

# 手动跑一次
~/Documents/obsidian/basicForMac/06_Skills/自动抓取/run_horizon.sh
```

## 必需的环境变量

在 `~/Documents/Horizon/.env` 中写入（任选一家模型服务商）：

```
OPENAI_API_KEY=你的密钥
```

换用 DeepSeek 等其他服务商时，同步修改 `config.genrec.json` 里的 `ai.provider`、`ai.model`、`ai.base_url` 和 `api_key_env`。
