# ARC-AGI-3: verified benchmark facts

Everything here is from primary sources (arcprize.org, docs.arcprize.org, the
ARC-AGI-3 technical report). Sources at the bottom. If you add a fact, cite it.
If a fact turns out to be wrong, correct it here **and** note the correction in
the decision log, because we may have reasoned from it.

Last verified: 30 August 2026.

---

## What the agent sees and does

| Property | Value |
|---|---|
| Observation | 64×64 grid, each cell one of 16 colours |
| Frame delivery | single frame, or a frame sequence for animated transitions between turns |
| Action space | five directional/control keys, an **Undo** (reverts to previous state), and one cell-selection action parameterised by (x, y) |
| Turn structure | strictly turn-based — the environment does **not** change asynchronously; the agent may reason offline for as long as it likes before submitting |
| Levels | at least six per environment; progression on reaching a win condition / terminal frame |
| Instructions given | none |

**The single most important structural fact:** an action is defined as a discrete
interaction with the environment. *Internal reasoning steps do not count as
actions.* Thinking is free. Acting is expensive. Everything in our design follows
from this.

### Dataset split

| Set | Count | Purpose |
|---|---|---|
| Public Demo | 25 | development, format demonstration |
| Semi-Private | 55 | external API testing |
| Fully Private | 55 | official competition scoring |

The private sets are held out. We cannot train on them. The 25 public demo
environments are our only direct evidence about the design idiom, and they should
be treated as a **validation** set for any generative or prior-fitting work, not
a training set.

---

## Scoring: Relative Human Action Efficiency (RHAE)

Per completed level:

```
level_score = (human_baseline_actions / ai_actions) ** 2
```

- Matching human efficiency = 1.0 (100%).
- Taking **twice** as many actions as the human = **0.25**, not 0.5. The penalty
  is quadratic.
- Per-level score is **capped at 1.15×** the human baseline, so you cannot bank
  credit by being unusually efficient on one level to offset another.
- There is **no per-game action budget**. The constraint is purely the efficiency
  ratio.
- Incomplete games are hard-capped: completing 4 of 5 levels cannot exceed 66.7%
  on that game regardless of efficiency.
- Score is normalised per game and averaged across all games.

### The human baseline

Established by **first-time** players. Multiple first-time players attempt each
level; the **upper median by fewest actions** is recorded as that level's
baseline — described as typical proficient human performance rather than outlier
runs. (Methodology note: the baseline uses median player performance per level
rather than second-best.)

Humans achieve a 100% success rate at completing the environments. So a 100% RHAE
score means beating every level of every environment at or above median
first-time human action efficiency.

### Consequences we have drawn from this

1. Exploration is quadratically expensive. An agent that flails for 500 actions
   and then solves the level perfectly scores approximately zero.
2. The benchmark is explicitly measuring **one-shot comprehension**, since the
   baseline is set by people seeing the game for the first time.
3. Because scoring is per-level but mechanics are shared across the ≥6 levels of
   an environment, the cost of learning concentrates in level 1. An agent that
   persists what it learned should be near-optimal from level 2 onward. An agent
   that re-derives the mechanics each level pays that tuition six times.
4. Because internal reasoning is free, any environment step that could have been
   replaced by a step in an internal simulator is pure waste.

---

## Current performance

| System | ARC-AGI-3 | Notes |
|---|---|---|
| Claude Opus 5 (High) | **30.16%** | current SOTA; solved five Public Demo environments (AR25, FT09, LP85, R11L, S5I5) no model had previously beaten |
| Anthropic Fable-class | ~20% | ARC Prize testing, Public Demo environments |
| GPT-5.6 Sol (Max) | 7.8% | previous high score |
| Claude Opus 4.8 | 1.5% | |
| Frontier systems, March 2026 | <1% | at technical-report publication |
| **Humans** | **100% completion** | baseline by definition |

For reference, Opus 5 scores 97.5% on ARC-AGI-1 and 90.4% on ARC-AGI-2 (Max
reasoning). The v3 collapse from 90% to 30% is the gap this benchmark exists to
measure.

---

## ARC Prize 2026 — the ARC-AGI-3 track

| Item | Detail |
|---|---|
| Total pool (all tracks) | ~$2M |
| ARC-AGI-3 track | $850K |
| Grand prize | $700K for the first eligible agent at **100%**; rolls over if unclaimed |
| Top score award | $75K guaranteed — 1st $40K, 2nd $15K, 3rd $10K, 4th–5th $5K each |
| Milestone prizes | $75K across two milestones (30 June 2026 — awarded; 30 September 2026) |
| Milestone #2 deadline | **30 September 2026** |
| Final submission | **2 November 2026** |
| Paper track deadline | 8 November 2026 |

### Rules that constrain the build

- Submission is a **Kaggle notebook** only.
- **No internet access during evaluation.**
- All code and methods must be **open sourced** to be eligible, before receiving
  official private evaluation scores, under a permissive licence (CC0 / MIT-0).
- Hardware and compute limits are announced at launch — confirm current limits
  before assuming anything about model size or runtime.

### The sibling track, for context

ARC Prize 2026 also runs a separate **ARC-AGI-2** track ($700K): $275K progress
prizes, $275K grand prize judged on writeup quality, and a $150K bonus for the
first solution reaching **85%** on the private eval *within Kaggle efficiency
limits*. That bonus is unclaimed. Frontier models exceed 85% unconstrained
(Opus 5 at 90.4%) while the best Kaggle-constrained entry scored 24.03% — the
gap is compute, not capability. We are not entering that track (see D-001).

---

## Tooling: how you actually run environments

**ARC-AGI-3 is not a static dataset.** There are no JSON task files to download
the way there were for ARC-AGI-1 and ARC-AGI-2. It is a game engine, and you run
against it.

### The local runtime

The `arc-agi` PyPI package **hosts the same game engine the Kaggle gateway
runs**. The documented design promise is: *"If it works locally, it works on
Kaggle."*

```bash
pip install arc-agi        # or: uv add arc-agi
```

```python
import arc
from arc import GameAction

env = arc.make("ls20", render_mode="terminal")
obs = env.step(GameAction.ACTION1, data={"x": 32, "y": 32})
# returns frame data; game state includes WIN, GAME_OVER, etc.
```

Note the action signature: the cell-selection action is parameterised by (x, y)
in `data`, confirming the 4,096-coordinate branching factor that D-005 addresses.

### Environment discovery and caching

The `Arcade` class discovers games from local directories and remote API
endpoints, with three modes:

| Mode | Behaviour |
|---|---|
| `NORMAL` | local + API |
| `ONLINE` | API only |
| `OFFLINE` | local only |

Games download on **first run** and cache to `environments_dir` (default
`environment_files/`). After that you are **fully offline**. An `ARC_API_KEY`
from https://three.arcprize.org/ enables access to additional games.

### Starter kit

`github.com/arcprize/ARC-AGI-3-Agents` — the official agent harness. Includes a
random-agent baseline. Defaults to local execution via `arc-agi`; set
`ONLINE_ONLY=True` in `.env` to use the hosted API instead.

```bash
uv run main.py --agent=random --game=ls20
make play-local    # runs your agent against every game in the dataset, locally
```

Kaggle then runs the same code a second time against **hidden** games for the
official score.

### Consequences for us

1. **Development rollouts are free and unlimited** on the cached public games.
   The perception work and the dynamics-induction validation experiment cost
   nothing but wall-clock.
2. **That is also a trap.** RHAE punishes retries quadratically, but local
   development lets you retry infinitely. It is easy to build something that
   works when you can re-run it and scores terribly under the real metric.
   **Always evaluate on the first attempt from a cold start**, and log actions
   consumed as a first-class metric from day one — not at the end.
3. Pin the `arc-agi` version. Engine changes affect reproducibility of every
   logged experiment.

## Sources

- ARC-AGI-3 technical report — https://arxiv.org/html/2603.24621v1
- ARC-AGI-3 scoring methodology — https://docs.arcprize.org/methodology
- Measuring Human Performance on ARC-AGI-3 — https://arcprize.org/blog/arc-agi-3-human-dataset
- ARC Prize 2026, ARC-AGI-3 competition — https://arcprize.org/competitions/2026/arc-agi-3
- ARC Prize 2026, ARC-AGI-2 competition — https://arcprize.org/competitions/2026/arc-agi-2
- ARC Prize 2026 overview — https://arcprize.org/competitions/2026
- Claude Opus 5 results — https://arcprize.org/results/anthropic-claude-opus-5
- ARC Prize 2026 developer docs — https://docs.arcprize.org/arc-prize-2026
- `arc-agi` package — https://pypi.org/project/arc-agi/
- Official agent harness — https://github.com/arcprize/ARC-AGI-3-Agents
- Kaggle competition — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
