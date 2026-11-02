#!/bin/bash
# Round 3: A vs D with the fixed turn_hints.py (commit 287c377), original game order.
cd /c/Users/Patrick/Documents/Patrick/ARC-AGI-3/.claude/worktrees/agent-a446720b35bb765b0
until grep -q "round2 done" runs/duck_ab/drive.log; do sleep 10; done
PY=third_party/duck-franzen/ARC3-Inference/.venv/Scripts/python.exe
run() { $PY kaggle/duck_grafts/local_run.py --arm $1 --games $2 --minutes 30 --concurrency 2 --seed 0 --out runs/duck_ab/$3_$1 > runs/duck_ab/$3_$1.log 2>&1; }
echo "round3 start $(date +%T)"
run A ls20,ft09,vc33,sb26 r3 & sleep 2; run D ls20,ft09,vc33,sb26 r3 & wait
echo "round3 done $(date +%T)"
