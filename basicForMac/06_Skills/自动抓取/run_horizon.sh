#!/bin/bash
set -euo pipefail

REPO_DIR="$HOME/Documents/Horizon"
VAULT_DIR="$HOME/Documents/obsidian/basicForMac"
FORMATTER="$VAULT_DIR/06_Skills/规则/format_obsidian_markdown.sh"
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

cd "$REPO_DIR"
uv run horizon >> "$REPO_DIR/horizon.log" 2>&1

date_tag="$(date +%F)"
echo "[$(date '+%F %T')] Horizon finished; copying summaries for $date_tag"

while IFS= read -r -d '' f; do
  case "$f" in
    *-zh.md)
      dest="$VAULT_DIR/00_Inbox/自动抓取/$date_tag｜Horizon中文日报.md"
      cp -v "$f" "$dest"
      "$FORMATTER" "$dest" || echo "[$(date '+%F %T')] Obsidian Markdown formatting failed: $dest"
      ;;
    *-en.md)
      dest="$VAULT_DIR/00_Inbox/自动抓取/$date_tag｜Horizon英文日报.md"
      cp -v "$f" "$dest"
      "$FORMATTER" "$dest" || echo "[$(date '+%F %T')] Obsidian Markdown formatting failed: $dest"
      ;;
  esac
done < <(find "$REPO_DIR/data/summaries" -maxdepth 1 -type f -name "horizon-$date_tag-*.md" -print0)
