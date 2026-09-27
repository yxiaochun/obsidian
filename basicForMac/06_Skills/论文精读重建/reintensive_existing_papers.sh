#!/bin/bash
set -uo pipefail

KB="/Users/yangxiaochun/Documents/obsidian/basicForMac"
CODEX="/Applications/ChatGPT.app/Contents/Resources/codex"
SOURCE_DIR="$KB/01_Sources/论文报告"
STATE_DIR="$KB/06_Skills/论文精读重建"
STATE_FILE="$STATE_DIR/completed.txt"
mkdir -p "$STATE_DIR"
touch "$STATE_FILE"

while IFS= read -r -d '' pdf; do
  name="$(basename "$pdf")"
  if grep -Fqx "$name" "$STATE_FILE"; then
    echo "[$(date '+%F %T')] SKIP $name"
    continue
  fi

  echo "[$(date '+%F %T')] START $name"
  prompt="请读取并严格执行 /Users/yangxiaochun/Documents/obsidian/obsidian-skills/skills/intensive-paper-reading/SKILL.md，完成这篇论文的结构化精读：$pdf 。精读卡必须新增「模型结构」小节，并嵌入论文原文中的主模型结构图；图片保存到 99_Attachments/图片/，文件名使用「知识卡片标题｜模型结构图.png」。写卡后请再读取并执行 /Users/yangxiaochun/Documents/obsidian/obsidian-skills/skills/obsidian-markdown/SKILL.md，对最终卡片做一次 Obsidian Markdown 格式化。请在 02_Knowledge/模型与案例 中找到对应卡片并更新；如果确实没有对应卡片，才新建一张。只允许修改这一张卡片和它引用的新图片文件，不要修改 PDF、原始资料、索引或无关文件。卡片状态保持「待复核」，不使用「待补充」「TODO」「详见原文」占位符。"

  if "$CODEX" exec \
    --ignore-rules \
    --sandbox workspace-write \
    --skip-git-repo-check \
    -C "$KB" \
    "$prompt" < /dev/null; then
    echo "$name" >> "$STATE_FILE"
    echo "[$(date '+%F %T')] DONE $name"
  else
    echo "[$(date '+%F %T')] FAIL $name"
  fi
done < <(find "$SOURCE_DIR" -maxdepth 1 -type f -name '*.pdf' -print0 | sort -z)

echo "[$(date '+%F %T')] ALL_FINISHED"
