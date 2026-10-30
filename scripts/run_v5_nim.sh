#!/usr/bin/env bash
# Play v5 on public games against NVIDIA NIM (development only; Guardrail 5).
#
#   scripts/run_v5_nim.sh OUTDIR ACTIONS JOBS GAME...
#   e.g. scripts/run_v5_nim.sh docs/experiments/nim/v5_set5 80 5 ls20 tu93 re86 vc33 tn36
#
# Reads NVIDIA_TOKEN from .env (ENV_FILE overrides; never echoed) and the games
# from ENV_DIR (default ./environment_files). Keep JOBS <= 5 and never run two of
# these at once: the free key throttles at about 5 concurrent requests (D-044).
#
# BUDGET (default 12000) is Nemotron's reasoning_budget, a NIM-only workaround:
# NIM closes a request after ~340 s, so a think that runs past ~16k tokens was
# lost entirely (debug run c, 2026-10-30). The budget ends thinking and the model
# still answers. The Kaggle path has no such limit and sets no budget (D-046).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a
. "${ENV_FILE:-$ROOT/.env}"
set +a
OUT="$1"; ACTIONS="$2"; JOBS="$3"; shift 3
cd "$ROOT"
mkdir -p "$OUT"
export PYTHONPATH=src
exec "${PYTHON:-python}" scripts/eval_agent.py \
  --games "$@" --agents v5 --actions "$ACTIONS" --jobs "$JOBS" \
  --env-dir "${ENV_DIR:-environment_files}" \
  --calls-dir "$OUT" --out "$OUT/eval_s0.json" \
  --llm-url https://integrate.api.nvidia.com \
  --llm-model nvidia/nemotron-3.5-super-vl-120b-a12b --llm-key-env NVIDIA_TOKEN \
  --llm-timeout 600 \
  --llm-extra "{\"chat_template_kwargs\":{\"enable_thinking\":true,\"reasoning_budget\":${BUDGET:-12000}}}"
