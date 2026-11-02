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

## Duck harness smoke runs on Kaggle (D-045, 2026-11-01)

`docs/experiments/kaggle/duck_smoke_2026-11-01/`. Each: `scripts/push_duck.py --games ls20,ft09 --game-minutes 20 --push`, RTX Pro 6000, seed as the harness sets it, one pass.

| run | flags | commit | result |
|---|---|---|---|
| `duck-base-smoke`, `duck-scale-smoke` | (none), `--graft --env ARC3_SCALE_HASH=1` | 602fc82 | **Both failed before play**: Kaggle's latest image is Python 3.13 and the Pennyroyal cp312 wheels do not install (`aiohttp==3.14.3`). Fixed by pinning Franzen's image (e36d22b) |
| `duck-mem-smoke` (A) | `--memory` | e36d22b | ft09 100 (6/6), ls20 15.45 (3/7), mean 57.72; memory pinned 2 levels |
| `duck-memscale-smoke` (B) | `--memory --graft --env ARC3_SCALE_HASH=1` | e36d22b | ft09 100 (6/6), ls20 19.84 (3/7), mean 59.92; graft applied |

A vs B is within noise (same levels cleared); these runs show only that both notebooks run end to end on the pinned image. The 2-game filter applies to offline practice runs only; a competition rerun takes the live game list and the TRUE_SUBMISSION settings.

## Duck variants on 11 public games, Kaggle (D-048, 2026-11-02)

`docs/experiments/kaggle/duck_11g_2026-11-02/`. Each: `scripts/push_duck.py <flags> --games ar25,ft09,lp85,r11l,re86,sb26,sc25,tr87,tu93,vc33,ls20 --game-minutes 20 --push`, RTX Pro 6000, one pass, 11 games concurrently (so ~70k tokens per game, about half the 2-game smoke runs: scores are not comparable with those).

| run | flags (all have `--memory`) | mean | levels |
|---|---|---|---|
| `duck-ctrl-11g` (A, control) | — | 28.88 | 38 |
| `duck-d-11g` (D) | `--hints --graft`, BUDGET+LIFE+EFFECTS, EXPOSE_RESET, SCALE_HASH | 36.73 | 41 |
| `duck-e-11g` (E) | D + `--adaptive` | 31.63 | 40 |
| `duck-f-11g` (F) | D without EXPOSE_RESET | 32.48 | 40 |

Per-game noise is very large (tr87: A 3.5, D 71.4, F 2.5), so one run per arm cannot separate the arms with confidence; a replicate of D (`duck-h-11g`) and a hints-only arm (`duck-g-11g`) were queued to measure that. Two earlier control attempts (`duck-a-11g` v1, v2) failed at startup because the kernel, first created by a push Kaggle rejected (2-GPU-session limit), never mounted its datasets; pushing under a new slug fixed it.

## Local A vs D on Nemotron via NIM (2026-11-02)

**This is NOT the Kaggle model.** The Duck harness ran on a Windows ARM laptop against the
hosted NVIDIA NIM endpoint, model `nvidia/nemotron-3.5-super-vl-120b-a12b`, not
Qwen3.8-Flash-Next under SGLang. Absolute scores are far below Kaggle's and say nothing about
the leaderboard; the only question was whether D beats A with the model held fixed.

`docs/experiments/local/duck_ab_2026-11-02/` (`summary.json`, `benchmark.json` with every
action, `stdout.log` per arm and round; transcripts stayed in the gitignored `runs/duck_ab/`).

- **Arms:** A = Franzen + solved-level memory (`push_duck.py --memory`). D = A + turn hints
  (budget, life, effects) + `EXPOSE_RESET=on` + `ARC3_SCALE_HASH=1`.
- **Runner:** `kaggle/duck_grafts/local_run.py --arm A|D --games ... --minutes 30 --concurrency 2
  --seed 0` (Franzen's env block with the endpoint swapped; changes marked LOCAL in the file).
  Driven by `drive.sh` (rounds 1-2) and `drive3.sh` (round 3), both copied here. Table:
  `python kaggle/duck_grafts/local_ab_table.py docs/experiments/local/duck_ab_2026-11-02`.
- **Fairness:** the same 4 games (ls20, ft09, vc33, sb26), a 30 min cap per game, 2 games in
  flight per arm, and both arms running at the same time, so both saw the same NIM load
  (4 streams in total). Round 2 reversed the game order and the launch order. The seed is
  sent, but concurrency makes sampling nondeterministic.
- **Code:** rounds 1-2 ran runner ce9bf44 with the hints as of 588205c. Round 3 ran the fixed
  hints, 287c377 (see below). So round 3's D is not the same arm as Kaggle's D.
- **Setup notes (Windows):** uv venv at `third_party/duck-franzen/ARC3-Inference/.venv` (x64
  CPython 3.12.12 under emulation). It was installed with `uv pip install -e` and does not
  use the lock (arc-agi 0.9.9, arcengine 0.9.3). Three shims live in the runner, not in the
  vendored tree:
  - the 67 KB Python-tool sandbox bootstrap overflowed Windows' command-line limit, so it is
    run from a file;
  - the sandbox env is given `SYSTEMROOT`, and `os.killpg` is replaced;
  - the runner re-runs itself under `-X utf8`, because a cp1252 crash in diagnostics.
  NIM rejects the request field `priority`, but the harness never sends it.

| round | game | A levels | A score | A actions | A model RESETs | D levels | D score | D actions | D model RESETs | D hint turns (budget/life/effects of prompts) | gen. tokens A / D (k) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | ls20 | 0/7 | 0.00 | 268 | 0 | 1/7 | 2.37 | 209 | 4 | 70/0/88 of 91 | 120 / 161 |
| r1 | ft09 | 0/6 | 0.00 | 49 | 0 | 0/6 | 0.00 | 58 | 0 | 46/0/59 of 63 | 193 / 173 |
| r1 | vc33 | 1/7 | 0.12 | 153 | 0 | 1/7 | 0.06 | 255 | 26 | 0/12/96 of 98 | 160 / 94 |
| r1 | sb26 | 1/8 | 2.49 | 59 | 0 | 1/8 | 2.78 | 142 | 4 | 0/0/77 of 79 | 189 / 147 |
| r1 | **mean** | | **0.65** | | | | **1.30** | | | | |
| r2 | sb26 | 1/8 | 2.78 | 81 | 0 | 1/8 | 1.07 | 262 | 17 | 0/0/59 of 61 | 222 / 162 |
| r2 | vc33 | 2/7 | 6.81 | 63 | 0 | 2/7 | 0.98 | 140 | 27 | 0/2/62 of 66 | 239 / 193 |
| r2 | ft09 | 2/6 | 14.29 | 32 | 0 | 1/6 | 4.76 | 50 | 0 | 17/0/30 of 34 | 116 / 67 |
| r2 | ls20 | 0/7 | 0.00 | 38 | 0 | 0/7 | 0.00 | 53 | 1 | 25/0/29 of 31 | 43 / 60 |
| r2 | **mean** | | **5.97** | | | | **1.70** | | | | |
| r3 | ls20 | 0/7 | 0.00 | 112 | 0 | 0/7 | 0.00 | 44 | 1 | 17/0/21 of 23 | 36 / 38 |
| r3 | ft09 | 1/6 | 3.52 | 61 | 0 | 0/6 | 0.00 | 29 | 0 | 8/0/18 of 21 | 84 / 54 |
| r3 | vc33 | 2/7 | 7.52 | 31 | 0 | 1/7 | 1.75 | 47 | 0 | 0/0/30 of 32 | 38 / 31 |
| r3 | sb26 | 1/8 | 0.69 | 59 | 0 | 1/8 | 0.66 | 37 | 1 | 0/0/19 of 24 | 40 / 30 |
| r3 | **mean** | | **2.93** | | | | **0.60** | | | | |

A: mean over all game-runs 3.18 (n=12)

D: mean over all game-runs 1.20 (n=12)

Levels cleared in total: A 11, D 9 (out of 12 game-runs each). HTTP 429 throttling (counted
in stdout) was 0/1 (A/D) in round 1, 81/97 in round 2 and 199/206 in round 3. The later
rounds therefore ran far fewer turns per game, about equally in each arm.

**Result: no evidence that D helps on Nemotron. A was ahead in 2 of 3 rounds.**
- Mean over 12 game-runs: A 3.18, D 1.20.
- Per-game spread is huge. ft09 A scored 0, 14.29 and 3.52 across the three rounds, so a
  difference of this size is within noise at n=12.
- **RESET:** the clearest mechanism is `EXPOSE_RESET`. D's model RESET 26 and 27 times on
  vc33 and 17 times on sb26, often in loops of RESETs that the harness reported as no-ops.
  Each RESET is a scored action, so D's action counts per level were higher (vc33 r2:
  2 levels in 140 actions against A's 63).
- **Kaggle:** the 11-game Kaggle runs (Qwen, 20 min) went the other way: control 28.88, D
  36.73, and F (= D without RESET) 32.48. **This local run did not change our mind about
  Kaggle**: a different model, 4 games and throttled turns. It does flag RESET as the
  component to watch for misuse.

**Bugs found in our hints and fixed in 287c377, with tests in `tests/test_turn_hints.py`.**
Both bugs were in the Kaggle D arm that scored 36.73.
1. *Lost-life line missed every life lost inside a batch.* It checked only the last
   transition. On r1 ls20, D lost 4 lives mid-batch and was told about none of them. It now
   checks every action since the previous turn message.
2. *False lost-life line on vc33.* vc33's board pieces reach into the border band, so
   clicking a toggle back restored the board and moved the band toward its start counts.
   That fired 35 times on r1 vc33, and 12 of those lines reached the model, which answered
   with RESETs. A life now also needs a band colour that was moving steadily (at least 3
   steps one way) to jump back by at least half that run.

Replaying the round-1 histories (both arms) with the fixed code: ls20 reports each real life
lost, and vc33, ft09 and sb26 report none.

**Limitation found, not fixed:** the budget line never fires on vc33. Its bar drains 1 or 2
cells per click, so it never holds the same rate for 3 steps in a row.
