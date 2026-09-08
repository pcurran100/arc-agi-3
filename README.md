# arc-agi-3

An agent for **ARC-AGI-3**, the interactive reasoning track of ARC Prize 2026.

## Start here

| File | What it is |
|---|---|
| `KICKOFF.md` | The session-1 prompt. Paste into Claude Code to begin. |
| `CLAUDE.md` | Project instructions, loaded automatically every session. Contains the documentation protocol. |
| `docs/state-of-play.md` | **Current beliefs and current focus.** Read first, every time. |
| `docs/decision-log.md` | Append-only record of every decision, including what was rejected. |
| `docs/design-thesis.md` | The argument the project rests on, and what would falsify it. |
| `docs/benchmark.md` | Verified benchmark mechanics, scoring, prize structure. Sourced. |
| `docs/field-review.md` | Prior art: what has been tried and what it scored. Sourced. |

## The bet

ARC-AGI-3 scores agents on action efficiency against a human baseline, with a
quadratic penalty for extra actions, and explicitly does not count internal
reasoning as actions. Thinking is free; acting is expensive. That is the
condition under which model-based methods beat model-free ones.

Every current leaderboard entry is an LLM harness with no queryable model of the
environment, so none can plan, and all re-learn the mechanics on every level.
We are building an agent that induces a symbolic model of the dynamics from
observed transitions and plans against it internally, calling an LLM only to
propose candidate goals.

Current SOTA is 30.16% (Claude Opus 5). Humans complete 100%.

## Layout

```
src/arc3/
  types.py          Action, Observation (a frame *sequence*), Transition, Episode
  client/           env (the only code that spends actions), replay, RHAE budget
  perception/       geometry, diff, segmentation, object tracking
  oracle/           TEST-ONLY ground truth from engine sprites (D-010)
scripts/            fetch_environments, record_fixtures, eval_geometry, eval_tracking
fixtures/           recorded episodes, so perception tests need no engine
tests/
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
python scripts/fetch_environments.py     # anonymous key; no registration needed
python -m pytest
```

`arc-agi` is pinned at 0.9.9. Games cache to `environment_files/`, which is
gitignored and must stay that way (D-008). After the first fetch everything runs
offline.

Note the import package is `arc_agi`, not `arc`, and `make` is a method on an
`Arcade` instance.

## Status

Perception layer built and measured against engine ground truth. No agent, no
dynamics induction, no score yet.

| What | Result |
|---|---|
| Geometry, cell-size agreement with the true camera scale | 18/23 |
| Geometry soundness (no induced cell spans two colours) | 25/25 |
| Cross-frame object identity, id switch rate | 0.8% |
| Click-space pruning (D-005) | 4,096 → 15–25 |

Next: the dynamics-induction validation experiment, which is the go/no-go on the
central bet. See `docs/state-of-play.md`.

## Licence

Competition rules require open-sourcing under a permissive licence (CC0 or
MIT-0) to be eligible for prizes. Decide and add before any submission.
