---
创建日期: 2026-09-17
更新日期: 2026-09-25
类型: 笔记
标签:
  - 生成式推荐
  - 推荐系统
来源: 论文报告目录
---

# 论文引文简报

这个任务每天从 `01_Sources/论文报告` 中轮换一篇论文，解析论文的 References，保留近 5 年条目，并挑选最多 5 篇与生成式推荐最相关的内容生成简读。

定时任务每天 7:30 自动运行。

## 输出

`05_Review/每日简报/YYYY-MM-DD｜论文引文简报：论文短标题.md`。该文件写入后会同样使用
`/Users/yangxiaochun/Documents/obsidian/obsidian-skills/skills/obsidian-markdown/SKILL.md`
做一次 Markdown 格式化。

高历史被引引文还会输出：

`02_Knowledge/模型与案例/主题加核心观点.md`

## 规则

- 只把论文 PDF 当作底稿，不修改原文。
- References 中的年份以 2022–2026 为候选。
- 优先保留生成式推荐、LLM 推荐、语义 ID、生成式召回/排序、推荐评测相关内容。
- 先扫描 `01_Sources/论文报告` 中全部底稿论文的 References，统计每条候选的去重被引数。
- 如果候选已经对应 `02_Knowledge/模型与案例` 中的精读卡片，简报直接写入 `已有精读` 链接。
- 如果同一篇候选被超过 3 篇底稿论文引用，且还没有精读笔记，任务会尝试下载 arXiv PDF，并调用
  `/Users/yangxiaochun/Documents/obsidian/obsidian-skills/skills/intensive-paper-reading/SKILL.md`
  完成结构化精读。
- 精读完成后，会使用
  `/Users/yangxiaochun/Documents/obsidian/obsidian-skills/skills/obsidian-markdown/SKILL.md`
  对该张卡片做一次 Obsidian Markdown 格式化；只调整格式，不改变结论和数字。
- 生成的精读笔记状态为 `待复核`，同时登记到 `02_Knowledge/README.md`；下载的 PDF 保存到 `01_Sources/论文报告/ReferencesPDF`。
- 如果高被引候选没有可获取的 PDF，只保留“高频引用”标记，不伪造精读结论。
- 自动生成的内容只做线索，不直接当成最终结论。
- 想进一步沉淀时，先人工核对原文，再升级成知识卡片或选题。

## 运行

```bash
# 只检查解析结果，不调用模型
uv run generate_brief.py --dry-run

# 手动生成当天简报
uv run generate_brief.py
```

模型读取 `~/Documents/Horizon/.env` 中的 `DEEPSEEK_API_KEY`；高被引论文精读部分改由 Codex 执行
`intensive-paper-reading` skill，不再直接调用 DeepSeek。如果 skill 执行失败，当日引文简报仍会保留，
对应条目标记为“高频引用精读失败”。
