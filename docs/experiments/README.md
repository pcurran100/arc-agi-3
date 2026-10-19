# Raw experiment output

Kept because `CLAUDE.md` requires that any experiment be reproducible from the
log alone, and because a decision-log entry quotes summary numbers while the
per-game detail is often what a later reader actually needs. These are the
*outputs*, not the record — the record is `docs/decision-log.md`.

Every file here is regenerable. None of it is an input to anything.

| file | produced by | recorded |
|---|---|---|
| `dynamics_sweep.txt` / `.json` | `scripts/eval_dynamics.py --out dynamics_sweep.json` | D-025, 2026-09-17 |
| `dynamics_curve.txt` / `.json` | `scripts/eval_dynamics_curve.py --carving colour_4 --test-tail 30 --games dc22,m0r0,cn04,tr87,wa30,ls20,cd82,re86` | D-025 follow-up, 2026-09-17 |
| `dynamics_sweep_objdriver.txt` / `.json` | `scripts/eval_dynamics.py --corpus corpus_obj --carvings colour_4,foreground_4` | D-026, 2026-09-24 |
| `carving_selection_exactkey.*` | first attempt, selecting on validation exact match | D-029, 2026-09-24 |
| `carving_selection.*` | `scripts/eval_carving_selection.py`; records every candidate key so rules compare offline | D-029, 2026-09-24 |
| `exploration.*` | `scripts/eval_exploration.py --actions 100` | D-031, 2026-09-24 |
| `agent_v0_nomodel.*` | `scripts/eval_agent.py --actions 500 --agents "graph_nomodel;explore"` at commit b3ff9ff with `local_min` defaulting to 0 (agent names were comma-separated then) | D-034, 2026-09-26 |
| `agent_v0_model.*` | `scripts/eval_agent.py --actions 500 --agents graph`, same code, `MODEL_TOP_K=8` | D-034, 2026-09-26 |
| `agent_v1_nomodel_2000.*` | `scripts/eval_agent.py --actions 2000 --agents graph_nomodel`, `local_min=0.3` | D-034, 2026-09-27 |
| `agent_v1_seed{1,2}.*` | `scripts/eval_agent.py --actions 500 --seed N --agents "graph_nomodel;explore"`, `local_min=0.3` | D-034, 2026-09-27 |
| `agent_v2_qwen3b_s0_PARTIAL.txt`, `llm_calls/qwen25coder3b/` | `eval_agent.py --agents v2 --llm-url` (local llama.cpp, Qwen2.5-Coder-3B Q4_K_M, CPU), seed 0 | D-035, 2026-09-27. **Incomplete: 5 of 25 games** (ar25, bp35, cd82, cn04, dc22; 0 levels). Stopped by the system for low memory; not a scoreboard row |
| `kaggle/v3_smoke_qwen36_2026-10-16/` | `scripts/push_kaggle.py kaggle/notebooks/bench_rtx6000.ipynb --slug arc3-bench-v3-smoke --env "ARC3_BENCH_GAMES=ls20 tn36" --env ARC3_BENCH_SEEDS=0 --env ARC3_BENCH_SHARDS=2 --llm-minutes 40` at commit 178f867 (Kaggle RTX Pro 6000, Qwen3.6-27B FP8, vLLM 0.19 under CPython 3.12, thinking off, presence_penalty 1.5) | D-040, 2026-10-16. **First real-model v3 run.** 45 calls; 34 unparseable (truncated or whitespace-degenerate). 2 games, seed 0: not a scoreboard row |
| `nim/v4_ls20_2026-10-19/` | `scripts/eval_agent.py --games ls20 --agents v4 --actions 60 --llm-url https://integrate.api.nvidia.com --llm-model nvidia/nemotron-3.5-super-vl-120b-a12b --llm-key-env NVIDIA_TOKEN --llm-timeout 300 --llm-extra '{"chat_template_kwargs":{"enable_thinking":true,"reasoning_budget":3000}}'` at commit 4eafd3f (prompt v4-2) | D-042, 2026-10-19. **First live v4 run.** 0/7 levels in 60 actions; 77 calls in 74 min, 28 failed with HTTP 500 (free endpoint; retries added in 34f055a). Predictions: 17 ok, 3 ok-with-extras, 23 wrong. The model learned the movement (5 cells), walls, and that the bottom bar counts moves; the store bloated to 76 live assumptions, mostly snapshots of positions (prompt fixed in 34f055a). Not a scoreboard row |
| `nim/v4_ls20_80_2026-10-19/` | same command with `--actions 80` at commit 3a02e85 (prompt v4-4); interrupted after 29 calls by the system (low memory) and resumed with `--replay` (misses fall through to the live model and are recorded) | D-042, 2026-10-19. **0/7 levels in 80 actions.** 54 calls, no failed calls after retries. Predictions: **51 ok, 7 ok-with-extras, 21 wrong** (first run 17/3/23); store held at 16 live assumptions, 2 of them goals. **Prediction accuracy rose; play did not.** The model took the bottom move-counter bar for the goal: from about t26 it pushed against the right wall to 'advance the bar', read a lost life (a red square turning grey at t42) as 'capturing a target', and spent t62-t79 on edge-blocked ACTION4 waiting for the bar to fill. Dynamics learned, goal wrong, no mechanism forced it to test the goal guess cheaply. Not a scoreboard row |

The `agent_*` runs need no corpus: they play the live engine from a cold start
and need only `environment_files/`.

Both need the corpus, which is gitignored and rebuilt deterministically:

```bash
python scripts/record_corpus.py                          # 25 games x 100 actions, ~15 s
python scripts/record_corpus.py --actions 300 --out corpus300   # ~41 s, for the curve
python scripts/record_corpus.py --driver random_objects --out corpus_obj  # ~157 s, for D-026
```

All seeds are 0. The sweep takes 7,888 s; the curve about 40 minutes on eight
games.

**A caution when comparing the two sweeps.** `dynamics_sweep` and
`dynamics_sweep_objdriver` were recorded with different drivers, so they differ
in **both** train and test transitions. A per-game delta in `exact` mixes
"better training evidence" with "a differently hard test set", on ~30 test
transitions. `changed_obj_acc` averages over hundreds of objects and is the
sturdier comparison.

**A caution when reading `dynamics_sweep.json`.** The `exact_ex_bk` field is a
diagnostic and can be vacuous — if a game changes nothing outside the induced
bookkeeping mask, "exact outside the mask" is satisfied by predicting nothing.
Check the `vacuous` flag before quoting it. The headline metric is `exact`.
