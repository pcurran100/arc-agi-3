# Scoreboard

One row per agent version, so progress is a table rather than a memory. Every
row is **offline RHAE on the 25 public games, cold start, one attempt per game**,
scored by `client/budget.py` (per-level score capped at 1.15, mean over all
levels, unreached levels 0). Produced by `scripts/eval_agent.py`.

Rules, from D-034's finding that single-seed comparisons measure luck:

- A row needs **all 25 games and at least three seeds**. Report the mean and
  the per-seed values. A single-seed figure may be listed but is marked as such
  and never used to choose between versions.
- The **action cap** is part of the result. Default 500 unless stated.
- A row that used an LLM names the **model, quantisation, hardware** and the
  **number of LLM calls per game**, and points at the recorded call log.
- Rows are never edited after the fact. A corrected measurement is a new row.

The public games are also the development set, so every row flatters the agent
relative to the hidden set. The Kaggle column is the only unflattered number.

| version | tag / commit | seeds | mean RHAE | per seed | levels (of 183) | Kaggle | notes |
|---|---|---|---|---|---|---|---|
| `explore` driver + RESET | b3ff9ff | 0,1,2 | 2.07% | 2.42 / 1.76 / 2.03 | 6 / 7 / 9 | — | D-031 policy; baseline before any agent loop |
| v1 `GraphAgent`, model off | `v1-graph` | 0,1,2 | **2.42%** | 3.46 / 1.68 / 2.11 | 10 / 9 / 10 | not yet submitted | D-034. Tied with `explore` within noise |
| v1 `GraphAgent`, model on | b3ff9ff | 0 only | 1.18% | 1.18 / — / — | 10 | — | single seed. Model as one-step filter hurts (D-034) |
| v1 `GraphAgent`, model off, 2000 actions | 607b3eb | 0 only | 3.47% | 3.47 / — / — | 14 | — | single seed. Extra levels all after action 230 |

Per-game detail for every row is in the `agent_*.json` files beside this one.
