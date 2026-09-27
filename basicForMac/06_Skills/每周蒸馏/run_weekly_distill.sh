#!/bin/bash
set -euo pipefail

export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"

KB="/Users/yangxiaochun/Documents/obsidian/basicForMac"
CODEX="/Applications/ChatGPT.app/Contents/Resources/codex"
PROMPT="$KB/06_Skills/每周蒸馏/weekly_distill_prompt.md"
LOG_DIR="$HOME/Library/Application Support/GenRecOutputs"
LOG_FILE="$LOG_DIR/weekly_distill.log"

mkdir -p "$LOG_DIR" "$KB/05_Review/每周蒸馏"
cd "$KB"

if [[ ! -f "$PROMPT" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 找不到每周蒸馏 prompt：$PROMPT" >> "$LOG_FILE"
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 每周蒸馏任务开始。" >> "$LOG_FILE"

# launchd may not have TCC access to read files from ~/Documents via shell
# redirection, so let Codex open the prompt from its workspace instead.
"$CODEX" exec \
  --ignore-rules \
  --sandbox workspace-write \
  --skip-git-repo-check \
  -C "$KB" \
  "请读取并严格执行 $PROMPT 中的每周蒸馏任务。" >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 每周蒸馏任务结束。" >> "$LOG_FILE"
