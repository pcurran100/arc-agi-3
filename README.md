# arc-agi-3

An agent for **ARC-AGI-3**, the interactive reasoning track of ARC Prize 2026.

## Start here

| File | What it is |
|---|---|
| `KICKOFF.md` | The session-1 prompt. Paste into Claude Code to begin. |
| `CLAUDE.md` | Project instructions, loaded automatically every session. Contains the documentation protocol. |
| `docs/state-of-play.md` | **Current beliefs and current focus.** Read first, every time. |
| `docs/decision-log.md` | Append-only record of every decision, including what was rejected. |
| `docs/architecture.md` | **How the system is actually built.** Essay form, with worked examples and explicit non-goals. |
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
| Changed pixels explained by the objects we flag | 100% on 4 of 5 fixtures, 79% on the fifth |
| Cross-frame object identity, id switch rate | 0.8% (resting frames only) |
| Assignment algorithm, fuzzed against brute force | 4,000 cases, 0 suboptimal |
| Click-space pruning (D-005) | 4,096 → 8–196 |
| Geometry, games where any structure is found | 4 of 25 |
| Geometry, cell-size agreement with the true camera scale | 18/23 — but 16 are the trivial `cell=1` case |

Every figure is measured under **random play**, which is mostly no-ops, and the
identity figure covers resting frames only. Read them with D-015's corrections
alongside; `docs/architecture.md` §3–4 gives the full picture and the caveats.

Next: the dynamics-induction validation experiment, which is the go/no-go on the
central bet. See `docs/state-of-play.md`.

## Licence

Competition rules require open-sourcing under a permissive licence (CC0 or
MIT-0) to be eligible for prizes. Decide and add before any submission.
