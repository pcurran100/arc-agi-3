# ARC-AGI-3: verified benchmark facts

Everything here is from primary sources (arcprize.org, docs.arcprize.org, the
ARC-AGI-3 technical report). Sources at the bottom. If you add a fact, cite it.
If a fact turns out to be wrong, correct it here **and** note the correction in
the decision log, because we may have reasoned from it.

Last verified: 30 August 2026. **Corrected against the live engine and docs on
2026-08-30 (session 1) — see D-009.** Facts established by direct measurement
against `arc-agi==0.9.9` / `arcengine==0.9.3` are marked *[measured]*; they are
the strongest evidence in this file and should be preferred over prose from any
source, including this document's own earlier text.

---

## What the agent sees and does

| Property | Value |
|---|---|
| Observation | 64×64 grid, each cell one of 16 colours. *[measured]* delivered as `list[np.ndarray]`, `shape (64, 64)`, `dtype int8` |
| Frame delivery | a **sequence** of 0–N frames per action. *[measured]* 0 frames means the action was a no-op; sequences of 42 frames observed (sb26). Intermediate frames are real observations, not decoration |
| Action space | `RESET` (id 0) plus `ACTION1`–`ACTION7`. `ACTION6` is the cell-selection action, parameterised by `(x, y)` with `0 ≤ x,y ≤ 63`. `ACTION1`–`ACTION5` and `ACTION7` take no parameters. **There is no Undo** — see the correction below |
| Legal actions | *[measured]* each frame carries `available_actions: list[int]`, a per-game mask. Observed masks range from `[6]` (click only, 5 games) to `[1,2,3,4,5,6,7]` (ar25). Stable across 60 random actions in all 25 games |
| Turn structure | strictly turn-based — the environment does **not** change asynchronously; the agent may reason offline for as long as it likes before submitting |
| Levels | *[measured]* 6–10 per environment; **183 levels across the 25 public demo games**. `win_levels` on every frame gives the count; progression increments `levels_completed` |
| Determinism | *[measured]* fully deterministic. Identical action sequences from `seed=0` produced byte-identical frame sequences in all 25 games |
| Instructions given | none |

### Correction: there is no Undo action *[measured]*

An earlier version of this file stated the action space included "an **Undo**
(reverts to previous state)". **This is wrong.** `arcengine.GameAction` defines
`RESET` and `ACTION1`–`ACTION7` only, `ARCEngine/OVERVIEW.md` states "Inputs:
limited to 6 actions plus a `RESET`", and `ARCBaseGame.perform_action` keeps no
state history of any kind. `ACTION7` is an ordinary game-specific action;
pressing it after a move reverted the frame in 1 of 6 games tested and not in
the other 5.

What `RESET` actually does (`ARCBaseGame.handle_reset`):

- if no action has yet been taken, or the game is in `WIN` — **full game reset**
- otherwise — **restart of the current level only**, and the action counter is
  *not* rewound

So the only rollback available is "throw away this level and start it again",
at the cost of every action already spent on it. `design-thesis.md` §8 reasons
from a cheap two-action probe-and-revert; that mechanism does not exist, and the
value-of-information argument has to be re-derived without it. Logged as D-009.

### Camera scaling and letterboxing *[measured]*

The 64×64 frame is **not** in general a 64×64 game board. `ARCEngine` renders
the level at the camera's resolution, upscales by the largest integer factor
that fits, and letterboxes the remainder; the 64×64 UI layer is then drawn on
top spanning the full frame.

12 of the 25 public demo games use a camera smaller than 64×64:

| Camera | Games | Implied upscale |
|---|---|---|
| 8×8 | bp35, lf52 | 8× |
| 11×11 | m0r0 | 5× |
| 16×16 | sp80 | 4× |
| 20×20, 21×21 | cn04, ar25 | 3× |
| 32×32, 32×19 | ft09, vc33, lp85 | 2× |
| 39×39, 45×45, 64×44 | tu93, ka59, dc22 | 1×, letterboxed |
| 64×64 | the remaining 13 | 1× |

Two consequences, both large:

1. **Recovering the upscale factor and the playfield rectangle is a
   first-class perception primitive**, not a nicety. It is inducible from run
   lengths and border structure, so it does not violate Guardrail 2.
2. It shrinks the click space far more than D-005 assumed. On bp35 the true
   board is 8×8 = **64** distinct cells, not 4,096, before any segmentation.

The UI layer is drawn last and always spans (0,0)–(63,63), so **UI and playfield
are spatially separable** — which is directly relevant to the design thesis §7
hunt for a score region.

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

**The baselines are available locally** *[measured]*. `EnvironmentInfo` carries
`baseline_actions: list[int]`, one entry per level, for every public demo game.
`GET https://three.arcprize.org/api/games` returns them, and they are cached in
each game's `metadata.json`. This means **we can compute our own RHAE offline**,
per level, from day one — no leaderboard round-trip needed. Do it.

| game | tags | levels | human baseline actions per level |
|---|---|---|---|
| ar25 | keyboard_click | 8 | 32, 50, 75, 37, 89, 159, 233, 73 |
| bp35 | keyboard_click | 9 | 21, 48, 44, 38, 33, 87, 86, 131, 163 |
| cd82 | keyboard_click | 6 | 55, 8, 41, 21, 23, 23 |
| cn04 | keyboard_click | 6 | 29, 54, 85, 300, 208, 113 |
| dc22 | keyboard_click | 6 | 59, 102, 67, 98, 324, 578 |
| ft09 | *(none)* | 6 | 43, 12, 23, 28, 65, 37 |
| g50t | keyboard | 7 | 78, 175, 179, 230, 96, 54, 67 |
| ka59 | keyboard_click | 7 | 28, 109, 51, 51, 33, 132, 326 |
| lf52 | click | 10 | 32, 81, 60, 71, 205, 148, 244, 109, 164, 225 |
| lp85 | click | 8 | 17, 38, 31, 16, 41, 60, 26, 159 |
| ls20 | keyboard | 7 | 22, 123, 73, 84, 96, 192, 186 |
| m0r0 | keyboard_click | 6 | 30, 111, 203, 26, 500, 237 |
| r11l | click | 6 | 22, 33, 51, 26, 52, 49 |
| re86 | keyboard_click | 8 | 26, 42, 86, 108, 189, 139, 424, 241 |
| s5i5 | click | 8 | 20, 89, 106, 54, 162, 38, 86, 83 |
| sb26 | keyboard_click | 8 | 18, 28, 18, 19, 31, 23, 58, 18 |
| sc25 | keyboard_click | 6 | 36, 6, 32, 83, 143, 50 |
| sk48 | keyboard_click | 8 | 61, 177, 101, 103, 230, 181, 125, 92 |
| sp80 | keyboard_click | 6 | 39, 58, 25, 148, 96, 152 |
| su15 | click | 9 | 22, 42, 26, 115, 36, 31, 8, 40, 41 |
| tn36 | click | 7 | 32, 72, 26, 40, 30, 55, 62 |
| tr87 | keyboard | 6 | 54, 58, 40, 45, 71, 146 |
| tu93 | keyboard_click | 9 | 19, 16, 34, 42, 123, 80, 14, 23, 111 |
| vc33 | click | 7 | 7, 18, 44, 61, 131, 34, 152 |
| wa30 | keyboard | 9 | 71, 119, 183, 98, 368, 68, 79, 442, 415 |

Read the small numbers carefully. sc25 level 2 has a baseline of **6 actions**
and cd82 level 2 of **8**. At a 1.15× cap, near-perfect play on those levels
means solving them in 7 and 9 actions respectively — with *no prior instruction*.
That is not an efficiency problem, it is a comprehension problem, and it is the
clearest single illustration of what this benchmark is measuring.

The `tags` field is a genuine free hint about modality: 4 games are keyboard-only,
7 are click-only, 13 are both, 1 (ft09) is untagged but click-only by its
`available_actions`. We should decide deliberately whether the agent is allowed
to read it — the private sets presumably carry it too, but relying on it is a
dependency worth being conscious of.

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
- Hardware and compute limits, **confirmed against docs.arcprize.org on
  2026-08-30**: the notebook selects an accelerator from `cpu`, `t4`
  (2× Nvidia T4), `p100`, or `rtx6000` (`g4-standard-48`). **`rtx6000` is
  reserved for ARC-AGI-3 notebooks** and consumes quota faster. Default is `t4`.
  The docs do not state runtime-hour or RAM limits; treat any specific figure as
  unverified until we see it in the Kaggle competition rules directly.
- "All accelerated Kaggle sessions have internet disabled" — consistent with the
  no-internet rule, and the starter kit is designed to run fully offline after
  the initial game download.
- Output artefact is `submission.parquet`.
- The Kaggle-side agent interface is two methods:
  `choose_action(frames, latest_frame) -> GameAction` and
  `is_done(frames, latest_frame) -> bool`. Note `frames` is the **history**, and
  `latest_frame` is the current `FrameDataRaw` — the harness hands us the full
  sequence, so our client's replay log should match that shape.

  *Unverified, recorded so we do not lose it:* secondary sources describe a
  $10,000 cap on runtime cost including commercial API calls, and third-party
  compute providers being acceptable. That is flatly inconsistent with "no
  internet during evaluation" for this track and is more likely a fact about the
  ARC-AGI-2 track or the Verified Testing policy. **Do not plan against it**
  until confirmed from the ARC-AGI-3 Kaggle rules page.

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
pip install arc-agi==0.9.9     # pinned; requires Python >= 3.12
```

**Correction *[measured]*:** an earlier version of this file gave the usage as
`import arc; arc.make(...)`. The distribution is named `arc-agi` but the import
package is **`arc_agi`**, and `make` is a method on an `Arcade` instance, not a
module-level function. The working call is:

```python
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction          # enums live in the engine, not the toolkit

arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
env = arc.make("ls20", seed=0, save_recording=True)   # make() also performs the reset
fd  = env.observation_space               # FrameDataRaw for the reset frame
fd  = env.step(GameAction.ACTION6, data={"x": 32, "y": 32})
```

`arc-agi==0.9.9` pulls `arcengine>=0.9.3`, plus flask, matplotlib, pydantic,
python-dotenv, pillow and requests. All vendorable; nothing needs a network at
inference time.

### `FrameDataRaw` — the actual observation object *[measured]*

| field | type | meaning |
|---|---|---|
| `frame` | `list[np.ndarray]` | 0–N frames of `(64, 64) int8`. Empty list = the action did nothing |
| `state` | `GameState` | `NOT_PLAYED` / `NOT_FINISHED` / `WIN` / `GAME_OVER` |
| `levels_completed` | `int` | levels cleared so far |
| `win_levels` | `int` | total levels in the game |
| `available_actions` | `list[int]` | the legal action ids for the next step |
| `full_reset` | `bool` | whether the last `RESET` was a full game reset |
| `action_input` | `ActionInput` | the action that produced this frame, echoed back |
| `guid` | `str` | run identifier |

Two traps in the wrapper worth knowing before writing a client:

- `LocalEnvironmentWrapper.step()` **catches every exception and returns
  `None`**. A `None` return is an error, not a no-op, and must not be silently
  treated as "nothing happened".
- Once `state` is `WIN` or `GAME_OVER`, any non-`RESET` action returns an
  **empty frame list** rather than raising. That is a distinct case from a
  legal-but-ineffective action, which also returns an empty list.

`ActionInput` also carries an optional `reasoning` blob (any JSON, ≤16 KB) that
is stored and echoed back verbatim. It is a free, in-band channel for annotating
a recording with what the agent believed at each step — worth using.

### Recording is built in *[measured]*

`arc.make(..., save_recording=True)` writes newline-delimited JSON to
`recordings/{scorecard_id}/{game_id}-{guid}.jsonl`, one event per step, with
frames included when `include_frame_data=True`. We do not have to invent a
replay format from scratch — though we will want a typed reader over it.

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
- `arc-agi` 0.9.9 release metadata — https://pypi.org/pypi/arc-agi/json
- Public demo game list and human baselines — `GET https://three.arcprize.org/api/games`
  (anonymous key from `GET /api/games/anonkey`); cached locally per game in
  `environment_files/<id>/<version>/metadata.json`
- `arcengine` 0.9.3 source, installed locally: `enums.py` (action + frame
  models), `base_game.py` (`perform_action`, `handle_reset`), `OVERVIEW.md`
  (engine contract, camera auto-scaling)
