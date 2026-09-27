#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <markdown-file>" >&2
  exit 2
fi

FILE="$1"
if [[ ! -f "$FILE" ]]; then
  echo "Markdown file not found: $FILE" >&2
  exit 1
fi

FILE="$(cd "$(dirname "$FILE")" && pwd)/$(basename "$FILE")"
KB="/Users/yangxiaochun/Documents/obsidian/basicForMac"
CODEX="/Applications/ChatGPT.app/Contents/Resources/codex"
SKILL="/Users/yangxiaochun/Documents/obsidian/obsidian-skills/skills/obsidian-markdown/SKILL.md"

if [[ ! -f "$SKILL" ]]; then
  echo "Obsidian Markdown skill not found: $SKILL" >&2
  exit 1
fi

cd "$KB"

"$CODEX" exec \
  --ignore-rules \
  --sandbox workspace-write \
  --skip-git-repo-check \
  -C "$KB" \
  "请读取并严格执行 $SKILL，对 $FILE 执行一次 Obsidian Markdown 格式化 pass。必要时读取同目录 references/PROPERTIES.md、references/CALLOUTS.md 和 references/EMBEDS.md。只允许编辑这一个文件。要求 YAML frontmatter 合法，内部链接使用 [[wikilink]]，外部链接使用 [文本](https://...)，重要提醒可用 Obsidian callout，表格和代码块保持完整。不得改变研究结论、数字、来源、状态或用户表达；不使用「待补充」「TODO」「详见原文」这类占位符。完成后重新读取该文件确认可正常渲染。"
