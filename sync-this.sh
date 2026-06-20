#!/usr/bin/env bash
# Sync output/ and logs/ between apape1 and apape2, then commit and push.
# Run from either machine.

set -e
REPO="$HOME/zit/GS-LLM-Traders"

HOST=$(hostname)
case "$HOST" in
  promaxgb10-4ae4*)
    echo "=== On apape1: pulling output/ and logs/ from apape2 ==="
    rsync -av apape2.rc.binghamton.edu:~/zit/GS-LLM-Traders/output/ "$REPO/output/"
    rsync -av apape2.rc.binghamton.edu:~/zit/GS-LLM-Traders/logs/   "$REPO/logs/"
    ;;
  promaxgb10-4be9*)
    echo "=== On apape2: pushing output/ and logs/ to apape1 ==="
    rsync -av "$REPO/output/" apape1.rc.binghamton.edu:~/zit/GS-LLM-Traders/output/
    rsync -av "$REPO/logs/"   apape1.rc.binghamton.edu:~/zit/GS-LLM-Traders/logs/
    ;;
  *)
    echo "ERROR: Unknown host '$HOST'. Expected promaxgb10-4ae4 (apape1) or promaxgb10-4be9 (apape2)." >&2
    exit 1
    ;;
esac

echo "=== Git status ==="
git -C "$REPO" status --short

echo "=== Committing ==="
git -C "$REPO" add output/ logs/
git -C "$REPO" diff --cached --stat
git -C "$REPO" commit -m "Sync output and logs from $HOST $(date '+%Y-%m-%d %H:%M')" || echo "(nothing new to commit)"

echo "=== Pushing ==="
git -C "$REPO" push

echo "=== Done ==="
