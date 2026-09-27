---
创建日期: 2026-09-17
更新日期: 2026-09-17
类型: 笔记
标签:
  - 生成式推荐
  - AI
来源: 每周输出生成任务
---

# 每周输出生成任务

这个任务每周一 9:30 由 `com.genrec.weekly-output` 触发。它会读取 `04_Outputs/README.md`，选择第一个状态为「待启动」的候选输出，生成一篇带 `｜初稿` 的文章或报告，并同步更新输出索引与对应研究专题。

任务只生成初稿，不会自动改成成稿。你需要确认核心结论后，才把文件名里的 `｜初稿` 去掉，并把状态更新为「成稿」。

## 文件

- `prompt.md`：无人值守生成初稿时使用的指令。
- `run_weekly.sh`：实际执行脚本，负责调用 Codex CLI 并写入日志。
- `com.genrec.weekly-output.plist`：LaunchAgent 配置模板。

## 日志

运行日志写入 `~/Library/Application Support/GenRecOutputs/weekly_output.log`。
