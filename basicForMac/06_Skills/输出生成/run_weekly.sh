#!/bin/bash
set -euo pipefail

export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

KB="/Users/yangxiaochun/Documents/obsidian/basicForMac"
CODEX="/Applications/ChatGPT.app/Contents/Resources/codex"
LOG_DIR="$HOME/Library/Application Support/GenRecOutputs"
LOG_FILE="$LOG_DIR/weekly_output.log"

mkdir -p "$LOG_DIR"
cd "$KB"

if ! rg -q "待启动" "04_Outputs/README.md"; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 没有待启动候选，任务结束。" >> "$LOG_FILE"
  exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 每周输出初稿任务开始。" >> "$LOG_FILE"

# launchd may not have TCC access to read files from ~/Documents via shell
# redirection, so let Codex open the prompt from its workspace instead.
"$CODEX" exec \
  --ignore-rules \
  --sandbox workspace-write \
  --skip-git-repo-check \
  -C "$KB" \
  "请读取并严格执行 $KB/06_Skills/输出生成/prompt.md 中的每周输出生成任务。" >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 每周输出初稿任务结束。" >> "$LOG_FILE"
