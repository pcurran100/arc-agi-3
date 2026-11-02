#!/bin/bash
# Two rounds; in each, arms A and D run side by side (2 games each in flight = 4 NIM streams).
cd /c/Users/Patrick/Documents/Patrick/ARC-AGI-3/.claude/worktrees/agent-a446720b35bb765b0
PY=third_party/duck-franzen/ARC3-Inference/.venv/Scripts/python.exe
MIN=30
run() { $PY kaggle/duck_grafts/local_run.py --arm $1 --games $2 --minutes $MIN --concurrency 2 --seed 0 --out runs/duck_ab/$3_$1 > runs/duck_ab/$3_$1.log 2>&1; }
echo "round1 start $(date +%T)"
run A ls20,ft09,vc33,sb26 r1 & sleep 2; run D ls20,ft09,vc33,sb26 r1 & wait
echo "round1 done $(date +%T)"
run D sb26,vc33,ft09,ls20 r2 & sleep 2; run A sb26,vc33,ft09,ls20 r2 & wait
echo "round2 done $(date +%T)"
