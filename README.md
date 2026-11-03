# arc-agi-3

An agent for **ARC-AGI-3**, the interactive reasoning track of ARC Prize 2026.
Final submission deadline **2 November 2026**: a Kaggle notebook, no internet
at evaluation, open source.

## Start here

| File | What it is |
|---|---|
| `CLAUDE.md` | Project instructions, loaded automatically every session. Contains the documentation protocol. |
| `docs/state-of-play.md` | Current status, beliefs, open items and risks (as of 2 Nov 2026). Older state is kept below its Archive line. |
| `docs/history.md` | **How we got here**: each phase, the ideas tried and dropped, the ideas parked, the options rejected, and the lessons. |
| `docs/decision-log.md` | Append-only record of every decision (D-001 to D-049; D-033 and D-048 are reserved), including what was rejected. The copy at the repo root is an old snapshot. |
| `docs/architecture.md` | How the system is actually built, with worked examples and explicit non-goals. §14 covers the submission harness. |
| `docs/experiments/` | Every run: `README.md` indexes them with the exact command; `scoreboard.md` holds the RHAE table for our own agents. |
| `docs/design-thesis.md` | The argument the project rests on, and what would falsify it. |
| `docs/benchmark.md` | Verified benchmark mechanics, scoring and prize structure. |
| `docs/field-review.md` | Prior art: what has been tried and what it scored. |
| `kaggle/README.md` | How to build, check and submit our own agents' notebooks (v1–v3). The Duck submission is built by `scripts/push_duck.py`. |

## The bet

ARC-AGI-3 scores agents on action efficiency against a human baseline, with a
quadratic penalty for extra actions, and does not count internal reasoning as
actions. Thinking is free; acting is expensive. That is the condition under
which model-based methods beat model-free ones.

The project began by inducing a symbolic dynamics model from transitions and
calling an LLM only to propose goals. Since D-039 **an LLM is the reasoner in
the loop**: it forms and revises hypotheses about the player, what each action
does and what the goal is, and states them in a form exact code can check.
Deterministic code is its senses and its referee. It computes every fact about
the frames, checks every prediction, and plans routes. Actions are spent to
test hypotheses or to win.

## Where things stand (2 November 2026, deadline day)

**Our own agent never beat the goal-free v1 explorer on score.** The public
Milestone #2 notebooks score ~28–31 on the leaderboard. Ours scored ~2. So the
submission is built on Daniel Franzen's vendored Duck harness, the Milestone #2
winner, with our findings added as flag-gated grafts (D-043, D-045, D-047).
**Nothing has been submitted yet.** The user ruled out entering the plain
public baseline: only a variant that improves on it goes in, and only on the
user's go-ahead (D-049). Variants are being compared against that baseline
offline.

### Our agent, five versions

Each version is still in the tree, because each is the comparison for the
next.

| Version | What it is | Best evidence | Status |
|---|---|---|---|
| **v1** `agent.py` | `GraphAgent`: exact perception, a Thompson-sampling explorer and a state graph. No goal, no LLM. | **2.42% mean offline RHAE**, 25 games × 3 seeds. Tied with the plain explorer. The induced dynamics model, used as a filter, cut the score to a third, so it ships off (D-034). | Tagged `v1-graph`. Best score of any version of ours. |
| **v2** `agent_v2.py` | An LLM proposes goal hypotheses as code; a weighted population steers v1's search. | Synthetic game: 600+ → 61 actions. With a real model, identical to v1 (D-035). | Superseded: the LLM decided nothing (D-039). |
| **v3** `v3/` | The LLM writes a world model as code; code verifies it on every transition and plans in it. | Kaggle, Qwen3.6-27B: identical to v1; 34 of 45 answers lost to truncation. One world model verified at 100% (D-040). | Superseded. |
| **v4** `v4/` | An assumption store the LLM edits; a checked prediction on every action. Since D-044: honest checking, a fact layer that finds counters and lost lives, and subgoals routed by code. | 0/7 levels on ls20, then **0 levels on 5 games** (NIM Nemotron-VL). A scripted check clears ls20 level 1 at the human baseline (D-042, D-044). | Superseded by v5. |
| **v5** `v5/` | Object permanence, an object sheet, counters out of the object list, the model's simulator replayed and searched by code, and tiered calls (D-046). | **1 level on 4 valid games**, at 0.43–0.75 calls per action (v4: ~1.3). The failure moved from cost to goal-finding. | Research line on `v5-objects`. Not a submission candidate. |

### The submission: Duck harness plus grafts

| Arm | What it is | Evidence |
|---|---|---|
| **A** | Franzen's notebook + lalalia9's solved-level memory (`push_duck.py --memory`). The author states a public score of 31.46. | Kaggle smoke, ls20 + ft09: 57.72. **Reference arm, not to be entered alone** (D-049). |
| **B** | A + our scale-invariant shape hash (`--graft --env ARC3_SCALE_HASH=1`). | Same smoke: 59.92, which is noise (D-047). |
| **C** | A + Franzen's model-visible RESET (`--env EXPOSE_RESET=on`). | Not run. |
| **D** | A + our turn hints (budget bar, lost-life detection, effect counts) + RESET + scale hash. | A-vs-D comparison on NIM (`kaggle/duck_grafts/local_run.py`) running on 2 Nov. No result yet. |

What each graft is, and what none of it does, is in `docs/architecture.md`
§14. The third-party code is credited in `third_party/*/NOTICE.md`.

## How we got here, and what we tried on the way

The full narrative is `docs/history.md`. In brief:

1. **Perception first** (D-007 to D-017). Exact segmentation, geometry and
   tracking, audited against an engine-sprite oracle. First contact with the
   engine killed four documented "facts", including Undo (D-009).
2. **Inducing the dynamics** (D-018 to D-032). An MDL rule language beat
   chance on 7 of 23 games (D-025). Dropped on the way:
   - round-robin driving (D-020);
   - static-to-interactive transfer (D-023);
   - per-game carving selection, which was a coin flip (D-029).

   An exploration policy was the first real capability (D-031).
3. **An end-to-end agent**, v1 (D-034). The learned model *hurt* as an action
   filter, and the agent had no goal.
4. **The LLM, step by step into the loop.**
   - v2 proposed goals and decided nothing.
   - v3 wrote world models and lost them to truncation.
   - v4 kept checked beliefs but held a wrong goal with confidence. Its
     forensics found six bugs of ours.
   - v5 made acting cheap but still could not find the goal.

   Tried and turned off on the way:
   - averaging hypotheses (they cancel; D-038);
   - images beside the exact grid (no measured benefit);
   - crediting goals from correct predictions (inflated a wrong goal to 0.91);
   - hard thinking caps (rejected by the user).
5. **Two tracks, then the pivot** (D-043, D-045, D-047, D-049). Four days out
   we measured ourselves against the public leaderboard for the first time,
   and switched the submission to the public harness.

**Parked and never tried:**

- a population over dynamics hypotheses;
- multi-step rollout accuracy;
- reading the 42-frame animations as evidence;
- fine-tuning, including an ARC-AGI-3 RL LoRA;
- the Qwen3.8 model comparison (D-037);
- running v4 or v5 on Kaggle with Qwen;
- grafting checked predictions and goal populations onto the Duck harness.

## Branches

The work ran on parallel branches in the last week, so the code is not yet in
one place:

| Branch | Holds |
|---|---|
| `main` | v1–v4 (D-042) and the README of 29 Oct. |
| `v4-fixes` | D-044: the v4 forensic fixes and subgoal walker. Merged into `v5-objects`. |
| `v5-objects` | v5 (`src/arc3/v5/`, D-046), its runs and architecture §13. |
| `duck-base` | The submission track: vendored harness, grafts, `push_duck.py`, `local_run.py` (D-045, D-047, D-048 wip). |
| `docs-status` | This reconciled documentation: every branch's decisions, runs and architecture in one record (2 Nov). |

## Layout

```
src/arc3/
  types.py          Action, Observation (a frame sequence), Transition, Episode
  client/           env (the only code that spends actions), replay, RHAE budget
  perception/       components, geometry, diff, six segmentations, tracking
  dynamics/         rule language, MDL induction, prediction        (v1)
  explore.py        Thompson-sampling explorer                      (D-031)
  agent.py          v1 GraphAgent                                   (D-034)
  scene.py, hypothesis.py, propose.py, agent_v2.py                  (v2, D-035)
  llm/client.py     OpenAI-compatible client: record/replay, retries, Truncated
  v3/               facts, world model sandbox, prompts, agent      (D-039)
  v4/               assumption store, expectations, agent           (D-042)
                    + router and forensic fixes                     (D-044; branch v5-objects)
  v5/               object sheet, simulator search, call tiers      (D-046; branch v5-objects)
  oracle/           TEST-ONLY ground truth from engine sprites      (D-010)
third_party/        duck-franzen (Apache-2.0), solved-level-memory, adaptive-duck  (duck-base)
kaggle/             MyAgent shim, local harness, notebooks; duck_grafts/ (turn hints, local_run)
scripts/            eval_agent (scores our agents), build_kaggle, push_kaggle,
                    push_duck (builds the submission notebook), fetch_environments,
                    and the perception/dynamics evals
fixtures/           recorded episodes, so most tests need no engine
docs/experiments/   every run's output and recorded LLM calls
tests/
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
python scripts/fetch_environments.py     # anonymous key; no registration needed
python -m pytest
```

`arc-agi` is pinned at 0.9.9. Games cache to `environment_files/`, which is
gitignored and must stay that way (D-008). After the first fetch everything
runs offline. The import package is `arc_agi`, not `arc`, and `make` is a
method on an `Arcade` instance.

Score one of our agents offline:

```bash
python scripts/eval_agent.py --agents graph_nomodel --actions 500          # v1
# v5 (on branch v5-objects):
python scripts/eval_agent.py --games ls20 --agents v5 --actions 80 \
    --llm-url https://integrate.api.nvidia.com \
    --llm-model nvidia/nemotron-3.5-super-vl-120b-a12b --llm-key-env NVIDIA_TOKEN
```

The second command calls a hosted model and is for development only. The
submission must run with no network (Guardrail 5). Every LLM run records its
calls, and `--replay` plays them back. Keep the total at about 5 concurrent
thinking-on requests per NIM key (D-044).

## Kaggle

- **The submission** (branch `duck-base`): `scripts/push_duck.py --slug NAME
  [--memory] [--graft] [--hints] [--adaptive] [--env K=V]`. It writes the
  notebook from Franzen's, with his public inputs declared and his Python 3.12
  image pinned. It is a dry run unless `--push` is given. Submitting to the
  competition is done only by the user (D-049).
- **Our own agents:** `scripts/build_kaggle.py --agent {v1,v2,v3,bench}` and
  `scripts/push_kaggle.py`. LLM versions serve Qwen3.6-27B FP8 through vLLM
  under a standalone CPython 3.12 (D-040). Details are in `kaggle/README.md`
  and `kaggle/SETUP.md`.

## Licence

Our code is MIT No Attribution (MIT-0), as the competition rules require for
prize eligibility (D-041). Vendored third-party code under `third_party/`
keeps its own licence (Apache-2.0) and NOTICE. Whether that combination meets
the rules is an open check for the user (D-045).
