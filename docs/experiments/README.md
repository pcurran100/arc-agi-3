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

| `nim/v4fix_set5_2026-10-29/` | `scripts/eval_agent.py --games ls20 tu93 re86 vc33 tn36 --agents "v4:image=0" --actions 80 --jobs 5 --llm-url https://integrate.api.nvidia.com --llm-model nvidia/nemotron-3.5-super-vl-120b-a12b --llm-key-env NVIDIA_TOKEN --llm-timeout 600 --llm-extra '{"chat_template_kwargs":{"enable_thinking":true}}'` at commit c517377 (prompt v4-6) | D-044, 2026-10-29. **Negative: 0 levels on all 5 games**; 46–110 calls per game; re86, tn36 and tu93 not clean (explorer steps after failed calls). No `eval_s0.json`: the process ended before ls20, tu93 and vc33 returned, and the file is written only at the end. Not a scoreboard row |
| `nim/v4old_set5_ABORTED_2026-10-29/` | the same command with the source tree at 128e75f (prompt v4-4), started at the same time as the run above | D-044. **Aborted, not a result.** Two runs × 5 concurrent thinking-on requests on one free NIM key throttled both: this one stopped writing after 19:31; it was killed at ~20:25. 2–5 calls per game |
| `nim/v5_set5_2026-10-30/` | `scripts/eval_agent.py --games ls20 tu93 re86 vc33 tn36 --agents v5 --actions 80 --jobs 5 --llm-url https://integrate.api.nvidia.com --llm-model nvidia/nemotron-3.5-super-vl-120b-a12b --llm-key-env NVIDIA_TOKEN --llm-timeout 600 --llm-extra '{"chat_template_kwargs":{"enable_thinking":true,"reasoning_budget":12000}}'` at commit c558d3a (prompt v5-4) | D-046, 2026-10-30. **INVALID: network outage, not a result.** After 3-6 good calls per game the next request died with `ConnectionResetError` (683-813 s in), then every call failed in 35 s with `getaddrinfo failed` (DNS); the agent handed each action to the v1 explorer, which played 65-74 of the 80 actions per game. The two level clears (vc33 at t44, tn36 at t27) came after each game's last good call: **explorer clears, not the agent's**. Early-call findings only (21 good calls): every call but one was `think` (thinking on, 120-295 s, ~13-14k completion tokens; one vc33 call took 684 s); the single `next` call (tn36) took 4 s and 243 tokens. ls20 wrote 2 simulators judged TRUSTED (4/4, then 10/10 on replay); tu93's second simulator was a SyntaxError (`'[' was never closed`); vc33's first used `import re` (rejected) and its later ones replayed 0/5-0/7 - they modelled the INTERFACE counter #10 as the puzzle piece; re86's replayed 5/9. **Actions planned by simulator search: zero** (ls20: one search of an earlier trusted simulator found no route; the 10/10 simulator came with a subgoal plan, which code walked first, and the network died before it was searched). Why `next` almost never fired: see D-046 (each blocked exploratory step on the driven object and each NO ROUTE was MAJOR, and level-start/NEW-shape notes from the probe forced think). Not a scoreboard row |
| `nim/v5_set5b_2026-10-31/` | `scripts/eval_agent.py --games ls20 tu93 re86 vc33 tn36 --agents v5 --actions 80 --jobs 5 --llm-url https://integrate.api.nvidia.com --llm-model nvidia/nemotron-3.5-super-vl-120b-a12b --llm-key-env NVIDIA_TOKEN --llm-timeout 600 --llm-extra '{"chat_template_kwargs":{"enable_thinking":true,"reasoning_budget":12000}}'` at commit a8476a7 | D-046, 2026-10-31. **1 level on 4 valid games** (vc33 L1 at action 34, baseline 7); tu93 INVALID (network down ~3 h; halted as designed). 0.43-0.75 calls/action (v4 ~1.3), ~180-270k completion tokens/game, 26-40 min/game. Simulator-search steps: ls20 28, re86 21, vc33 11, tn36 0. 4-8 explorer steps per game, so not clean. Not a scoreboard row |

**NIM free-key concurrency (measured 2026-10-29):** keep the total number of
concurrent thinking-on requests at **about 5 per key**, across all runs. At 10
the runs above stalled for 1.5 h while a tiny request still answered in 1 s.
Run comparisons one after the other, not side by side.

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
