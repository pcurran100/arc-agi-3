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

## Status

Pre-code. Scaffold and reasoning only. Current focus is the perception layer
(D-007).

## Licence

Competition rules require open-sourcing under a permissive licence (CC0 or
MIT-0) to be eligible for prizes. Decide and add before any submission.
