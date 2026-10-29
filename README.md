# arc-agi-3

An agent for **ARC-AGI-3**, the interactive reasoning track of ARC Prize 2026.
Final submission deadline **2 November 2026**: a Kaggle notebook, no internet
at evaluation, open source.

## Start here

| File | What it is |
|---|---|
| `CLAUDE.md` | Project instructions, loaded automatically every session. Contains the documentation protocol. |
| `docs/state-of-play.md` | Current beliefs and current focus. Its header runs to session 6 (16 Oct); the v4 work since is in D-042 and `docs/experiments/README.md`. |
| `docs/decision-log.md` | Append-only record of every decision (D-001 to D-042), including what was rejected. The copy at the repo root is an old snapshot. |
| `docs/architecture.md` | How the system is actually built, with worked examples and explicit non-goals. |
| `docs/experiments/` | Every run: `README.md` indexes them with the exact command; `scoreboard.md` holds the RHAE table. |
| `docs/design-thesis.md` | The argument the project rests on, and what would falsify it. |
| `docs/benchmark.md` | Verified benchmark mechanics, scoring and prize structure. |
| `docs/field-review.md` | Prior art: what has been tried and what it scored. |
| `kaggle/README.md` | How to build, check and submit the notebook. |

## The bet

ARC-AGI-3 scores agents on action efficiency against a human baseline, with a
quadratic penalty for extra actions, and does not count internal reasoning as
actions. Thinking is free; acting is expensive. That is the condition under
which model-based methods beat model-free ones.

The project began by inducing a symbolic dynamics model from transitions and
calling an LLM only to propose goals. It no longer works that way. Since D-039
**an LLM is the reasoner in the loop**: it forms and revises hypotheses about
the player, what each action does and what the goal is, and states them in a
form exact code can check. Deterministic code is its senses and its referee. It
computes every fact about the frames and checks every prediction against what
happened. Actions are spent to test hypotheses or to win.

## Where things stand (29 October 2026)

The agent has been rebuilt four times. Each version is still in the tree,
because each is the comparison for the next.

| Version | What it is | Best evidence | Status |
|---|---|---|---|
| **v1** `agent.py` | `GraphAgent`: exact perception, a Thompson-sampling explorer and a state graph. No goal, no LLM. | **2.42% mean offline RHAE**, 25 games × 3 seeds, 9–10 of 183 levels. Tied with the plain explorer (2.07%). The induced dynamics model, used as a filter, cut the score to a third, so it ships off (D-034). | Kaggle notebook built (`submission_v1`). Tagged `v1-graph`. |
| **v2** `agent_v2.py` | An LLM proposes goal hypotheses as code; a weighted population steers search over v1's graph. | Synthetic game: 600+ → 61 actions with a correct scripted hypothesis. With a real model, identical to v1 (D-035). | Superseded. The LLM decided nothing. |
| **v3** `v3/` | The LLM writes a world model as code (`predict`, `goal`). Code verifies it on every recorded transition and plans the shortest path in it. | Synthetic: 41 actions vs 364 for v1. First real run on Kaggle (Qwen3.6-27B FP8, RTX Pro 6000): **score identical to v1**, because 34 of 45 answers were truncated or degenerate. One tn36 model verified at 100% (D-040). | Serves on Kaggle. Fixes for lost answers not yet re-run there. |
| **v4** `v4/` | An assumption store the LLM edits, plus a prediction checked on every action. Two calls: *update* edits beliefs after a surprise; *act* picks 1–6 steps, each with an expected outcome. Thinking on, image sent beside the exact grid (D-042). | Two live runs on ls20 only (NIM Nemotron-3.5-Super-VL, dev endpoint): **0/7 levels** both times. Correct predictions rose from 17/43 to 51/79 between them. The model learned the movement and walls, then took the move-counter bar for the goal and spent the rest of the run trying to fill it. | Current development line. **Not yet packaged for Kaggle.** That needs vLLM with image input, thinking and a larger context. |

The short version: perception is exact and audited, and the LLM-in-the-loop
agent learns dynamics on a real game. **No version yet beats the goal-free v1
explorer on score.** The open problem is the goal. v4's model forms a wrong
goal guess and has no mechanism that makes it test that guess cheaply.

Every RHAE figure is offline, on the 25 public games, which are also the
development set. See `docs/experiments/scoreboard.md` for the rules a number
must meet to count.

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
  v4/               assumption store, expectation language, PNG render,
                    prompts, agent                                  (D-042)
  oracle/           TEST-ONLY ground truth from engine sprites      (D-010)
scripts/            eval_agent (scores any agent), build_kaggle, push_kaggle,
                    fetch_environments, and the perception/dynamics evals
kaggle/             MyAgent shim, local harness, notebooks
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

Score an agent offline:

```bash
python scripts/eval_agent.py --agents graph_nomodel --actions 500          # v1
python scripts/eval_agent.py --games ls20 --agents v4 --actions 80 \
    --llm-url https://integrate.api.nvidia.com \
    --llm-model nvidia/nemotron-3.5-super-vl-120b-a12b --llm-key-env NVIDIA_TOKEN
```

The second command calls a hosted model and is for development only. The
submission must run with no network (Guardrail 5). Every LLM run records its
calls. `--replay` plays them back and records any new calls in the same file.
`--replay --replay-only` never calls the live model.

## Kaggle

`scripts/build_kaggle.py --agent {v1,v2,v3,bench}` writes the notebook to
`kaggle/build/`. LLM versions serve Qwen3.6-27B FP8 through vLLM on the RTX Pro
6000, under a standalone CPython 3.12 (D-040). `scripts/push_kaggle.py` pushes
runs from the CLI with their inputs declared. Uploading through the web loses
the inputs. Details are in `kaggle/README.md` and `kaggle/SETUP.md`.

## Licence

MIT No Attribution (MIT-0), as the competition rules require for prize
eligibility (D-041). See `LICENSE`.
