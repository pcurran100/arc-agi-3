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
