#!/usr/bin/env bash
# Pull output/logs from apape2, then commit and push everything.
# Run from apape1 (this machine) at any time.

set -e
REPO="$HOME/zit/GS-LLM-Traders"
REMOTE="apape2.rc.binghamton.edu:~/zit/GS-LLM-Traders"

echo "=== Pulling output/ and logs/ from apape2 ==="
rsync -av "$REMOTE/output/" "$REPO/output/"
rsync -av "$REMOTE/logs/"   "$REPO/logs/"

echo "=== Git status ==="
git -C "$REPO" status --short

echo "=== Committing ==="
git -C "$REPO" add output/ logs/
git -C "$REPO" diff --cached --stat
git -C "$REPO" commit -m "Sync output and logs from apape2 $(date '+%Y-%m-%d %H:%M')" || echo "(nothing new to commit)"

echo "=== Pushing ==="
git -C "$REPO" push

echo "=== Done ==="
