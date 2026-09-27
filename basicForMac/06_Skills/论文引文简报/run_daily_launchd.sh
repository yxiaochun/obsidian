#!/bin/bash
set -euo pipefail

export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
cd "/Users/yangxiaochun/Documents/obsidian/basicForMac/06_Skills/论文引文简报"

for attempt in 1 2 3; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 论文引文简报：第 ${attempt} 次尝试"
  if uv run generate_brief.py; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 论文引文简报生成成功"
    exit 0
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 第 ${attempt} 次尝试失败"
    [[ "$attempt" -lt 3 ]] && sleep 90
  fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 3 次尝试全部失败"
exit 1
