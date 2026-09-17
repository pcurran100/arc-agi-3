# State of play

**The single source of truth for what we currently believe and what we are
currently doing.** Read this first every session. Update it before the session
ends if anything below is stale.

- **Last updated:** 2026-09-17, session 3
- **Status:** perception layer built, measured and audited. The click action
  space is now mapped against ground truth, and design thesis §8 has finally
  been re-derived. Still no agent, no dynamics, no score.
- **Days to Milestone #2 (30 Sep):** **13** — out of reach
- **Days to final submission (2 Nov):** **46** — the real target

---

## The bet, in three sentences

ARC-AGI-3 makes thinking free and acting quadratically expensive, which is the
condition under which model-based methods beat model-free ones. Every current
leaderboard entry is an LLM harness with no queryable model of the environment,
so none of them can plan, and all of them re-learn the mechanics on every level
of every game. We are building an agent that induces a symbolic model of the
dynamics from observed transitions and plans against it internally, calling an
LLM only to propose candidate goals.

Full argument: `design-thesis.md`. Benchmark facts: `benchmark.md`. Prior art:
`field-review.md`.

---

## What session 1 changed about what we believe

Four documented benchmark facts turned out to be wrong on contact with the
engine, and one of them was load-bearing. Full corrections in `benchmark.md`,
logged as D-009.

1. **There is no Undo.** The action space is `RESET` plus `ACTION1`–`ACTION7`;
   `ACTION6` is the click. The engine keeps no state history. `RESET` restarts
   the *current level* and does not rewind the action count. **Design thesis §8
   built its exploration argument on a cheap two-action probe-and-revert that
   does not exist.** §8 is annotated in place; the value-of-information argument
   is owed a fresh derivation and has not had one.
2. **The 64×64 frame is a composite**, not a game board: a camera view
   integer-upscaled and letterboxed, with a UI layer drawn last at full
   resolution. 12 of 25 games have a camera below 64×64, down to 8×8. On bp35,
   lf52 and m0r0 the **UI is most of what you see**, so "the upscale factor" is
   not a well-posed question for them.
3. **Frames arrive in sequences of 0–N, up to 42.** An action's return is an
   animation, not a frame. Design thesis §3 costs dynamics at "one labelled
   example per action"; the real figure is up to 42 sub-transitions, showing the
   *trajectory* of motion rather than its endpoints. **This makes D-002 more
   likely to work than the thesis assumed, not less.** Zero frames is also a
   legal return, meaning a legal action did nothing — a wasted action, and
   detectable.
4. **Free information the thesis never accounted for:** every frame publishes
   `available_actions` (masks range from `[6]` alone to all seven); every game
   publishes modality `tags`; and **per-level human baselines ship in
   `metadata.json`, so we can compute true RHAE offline from day one.** D-012
   allows the agent all three, behind a provider that can withhold the
   unverifiable ones, with a test that the withheld path still works.

Also established: **the engine is byte-deterministic under `seed=0`** across all
25 games, and the 25 public demo games hold **183 levels** (6–10 each).

---

## Architecture as currently conceived

Four subsystems, deliberately separable because we expect to replace at least
one wholesale (D-004).

```
                  ┌──────────────┐
   frames  ───▶   │  ONTOLOGY    │  what objects exist, which is me,
                  │  ← built     │  identity tracked across frames
                  └──────┬───────┘
                         │ objects
          ┌──────────────┼──────────────┐
          ▼                             ▼
   ┌─────────────┐              ┌──────────────┐
   │  DYNAMICS   │              │    GOAL      │
   │ induced     │              │ LLM-proposed │
   │ rules:      │              │ candidates,  │
   │ (s,a) → s'  │              │ ranked       │
   │ MDL-selected│              │              │
   └──────┬──────┘              └──────┬───────┘
          │  simulator                 │  objective
          └──────────────┬─────────────┘
                         ▼
                  ┌──────────────┐
                  │   PLANNER    │  free internal search;
                  │              │  VOI vs action cost
                  └──────┬───────┘
                         ▼
                      action
```

**Evidence economics differ per subsystem and this drives the action budget:**

| Subsystem | Evidence source | Cost to test | Persists across levels? |
|---|---|---|---|
| Ontology | every frame | passive, free | **yes** |
| Dynamics | every transition — *up to 42 sub-steps per action* | cheap | **yes** |
| Goal | win / death / level change only | expensive, sparse | no |

Early actions go to **dynamics** disambiguation, never to goal probing.

---

## Decisions made so far

D-001 to D-008 were backfilled from the design conversation. D-009 to D-014 come
from session 1 and D-015 to D-017 from session 2; all are backed by measurements. Full reasoning and rejected
alternatives in `decision-log.md`.

| ID | Decision | Confidence |
|---|---|---|
| D-001 | Enter ARC-AGI-3, not ARC-AGI-2 | high |
| D-002 | Model-based: induce symbolic dynamics, plan internally | **medium — this is the bet, still untested** |
| D-003 | Hypothesis population with soft down-weighting | high |
| D-004 | Separate ontology / dynamics / goal subsystems | high |
| D-005 | Prune the click space by object segmentation | **high — measured, 4,096 → 8–196** |
| D-006 | Exact symbolic perception, no VLM at the primitive layer | high |
| D-007 | Session 1 = perception layer only | done |
| D-008 | Develop against local `arc-agi` engine; never commit environment files | high |
| D-009 | Corrections to `benchmark.md` after first contact | — |
| D-010 | Engine sprites as a test-only ground-truth oracle | high |
| D-011 | Frame geometry as a fifth perception component, built first | high |
| D-012 | Agent may use `available_actions`, `tags`, `baseline_actions` | medium — rests on an unverifiable assumption about the private set |
| D-013 | Geometry: propose by run-length GCD, verify by cell uniformity | high |
| D-014 | Tracking by global assignment, measured against the oracle | high |
| D-015 | Corrections to the perception claims after an independent audit | — |
| D-016 | Object identity is not revisable; splits/merges out of scope for now | medium — a deliberate deferral, not a settled answer |
| D-017 | Keep a standing essay-format architecture document | high |
| D-018 | Dynamics experiment measures all six segmentations, not one | high |
| D-019 | Compute sizing and the chance baseline; success metric defined | high |
| D-020 | Round-robin driving tested and **rejected**; random retained | high — negative result, measured. Targeting sub-claim withdrawn by D-021 |
| D-021 | Ground-truth click map: keep object targets, on **class coverage** not precision; merge outcome classes online | high — measured on 19 games |
| D-022 | Exploration re-derived without Undo: revert costs *k*+1, so front-load at the root | medium — a derivation, not yet a measurement |
| D-023 | Static-to-interactive transfer **dropped**, not parked | high |

---

## What is built

```
src/arc3/
  types.py          Action, Observation (a frame *sequence*), Transition, Episode
  client/
    env.py          EnvClient — the only code that spends an action
    replay.py       ReplayClient + gzipped JSONL episodes, same Protocol
    budget.py       per-level action accounting and offline RHAE
  perception/
    components.py   connected components, 4- and 8-connectivity
    geometry.py     cell size, origin, playfield, UI/static regions   (D-013)
    diff.py         structural frame diff, and whole-animation diff
    segment.py      six candidate segmentations per frame
    track.py        object identity by global assignment              (D-014)
  drive.py          corpus-generation policies: random, round_robin  (D-020)
  oracle/           TEST-ONLY ground truth from engine sprites        (D-010)
scripts/            fetch_environments, record_fixtures, eval_geometry, eval_tracking,
                    size_pipeline, eval_baseline, eval_drivers,
                    eval_clickmap, eval_click_liveness                (D-021)
fixtures/           5 recorded episodes, 68 KB gzipped
tests/              139 tests, ~35 s
docs/architecture.md  how it all actually works, and what it does not do (D-017)
```

**Setup**, from a clean clone:

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
python scripts/fetch_environments.py     # anonymous key, no registration needed
python -m pytest
```

`arc-agi` is pinned at 0.9.9 (`arcengine` 0.9.3). The import package is
`arc_agi`, not `arc`, and `make` is an `Arcade` method — `benchmark.md` had this
wrong.

### Measured results

Corrected in session 2 by D-015. Where a figure moved, the session-1 claim is
given alongside, because the drift is itself the useful record.

| What | Result |
|---|---|
| Oracle fidelity | reproduces the engine's render **byte-exactly on all 25 games** |
| **Changed pixels explained** by the objects we flag as moved/recoloured/appeared | **100% on 4 of 5 fixtures, 79% on sp80** — the strongest evidence perception is not the weak link |
| **Assignment algorithm** fuzzed against brute-force optimal | **4,000 cases, 0 suboptimal** — the largest silent-wrongness risk, retired |
| Tracking, id switch rate | **0.8%** over 8,231 sprite-frame pairs (`colour_8`), resting frames only |
| Fragmentation | 2.55 components per ground-truth sprite — still the weak spot |
| **Geometry, games where any structure is found** | **4 of 25** (ft09, lp85, m0r0, sp80 at 150 actions) |
| Geometry, scale agreement | 18/23 (reproduces D-013) — but **16 of the 18 are the trivial `cell=1` case**. Two non-trivial successes: ft09, sp80 |
| Geometry, overestimates | **lp85: true 2, induced 4** at 150 actions (as D-013 recorded). The failing game **migrates with action count** — dc22 at 20, lp85 at 150 |
| Geometry, soundness | 0 unsound grids on 25/25 — but **verified in-sample**, so the check cannot fail by construction |
| Click pruning (D-005) | 4,096 → **8–196** candidates (sp80 8, vc33 16, ls20 29, sb26 31, bp35 196). *Was reported as 15–25.* |
| Tracking cost | 20–145 ms/frame, **824 ms on bp35** (191 components, pixel space) |
| Random-play baseline | **0 levels cleared** in 60 actions across all 25 games |
| **Click branching factor** (D-021) | **1–11 distinct outcomes on 13 of 15 live games.** sp80 is **1** — the click is a button, (x,y) decorative. Outliers: su15 143, r11l 3,096 |
| **Object targets vs lattice**, class coverage | **objects win 9, tie 5, lose 1.** Withdraws D-020's "targeting does not beat a lattice" |
| D-005 proposals vs true outcomes | **5–25× over-complete** (bp35 196 proposals for 8 outcomes) |
| Click legal but inert at level-1 start | **4 of 19 games** (ar25, m0r0, sc25, sk48) — the published mask is an upper bound |
| **Click liveness is state-gated** | **8 of 19 games.** m0r0 0%→100% by 5 actions; cd82/ka59 100%→~6% by 20 |
| `RESET` exactness | byte-identical restore, 19 games × 4,096 trials — the fact D-022 rests on |

**D-007's tarpit condition was not triggered.** Cross-frame identity works, and
it works well inside the two-week reassessment window.

**Caveats that must travel with every number above** (D-015, `architecture.md` §4):

1. The oracle can only be read for resting frames, so animated intermediate
   frames — where tracking is hardest — are unscored. Optimistic bound. (D-014)
2. The tracker is not merely unscored on animation frames, it is **not run on
   them**. Evaluation feeds one resting frame per action; deployment will feed
   whole 42-frame sequences. Different regime, no number attached.
3. **Everything is measured under random play**, whose transitions are small and
   low-diversity: 1–58 changed pixels of 4,096, dominated by recolouring, with
   few multi-object motions. Busy multi-mover scenes are absent from the
   evidence. (Corrected: an earlier draft called this distribution "mostly
   no-ops". Measured, the no-op rate is **14% overall** — 68% on sb26 and 0% on
   the other four fixtures. The problem is that changes are *tiny*, not that
   they are *absent*.)
4. Geometry soundness is verified **in-sample**. An out-of-sample check —
   induce on `frames[:n]`, verify on `frames[n:]` — does not exist and should.

---

## Immediate next step

Run the **dynamics-induction validation experiment** — the option deliberately
deferred in D-007, and the go/no-go on D-002 and therefore on most of the
project. Log a few hundred transitions across the public demo environments and
test whether object-tracking plus simple rule induction predicts held-out
transitions better than chance.

Everything it needs now exists: recorded episodes replay exactly, transitions
come out of `Episode.transitions()` with the full animation attached, and
tracking is measured. **It must not be quietly skipped because the architecture
already assumes the answer.**

Five things now shape how to run it. The first two are from session 1; the rest
are from session 2:

- **Treat the segmentation as an axis, not an input (D-018).** Run induction
  under all six carvings and report the comparison. A rule is stated in terms of
  objects, six candidate notions of object exist, and nothing selects among
  them — so fixing one arbitrarily would make a *negative* result
  uninterpretable: we could not tell "the bet is wrong" from "we carved wrong",
  on precisely the decision the project turns on. Roughly 6× compute, so run the
  full sweep on a subset and one carving on the rest.

- Use the **intermediate animation frames**, not just endpoints. Up to 42
  sub-transitions per action is far more evidence than the thesis budgeted for,
  and a rule that predicts the whole trajectory is much more constrained than one
  that predicts the endpoint.
- Hold out by **game**, not by transition. Transitions within one game are
  heavily correlated and a within-game split will flatter the result.
- ~~**Do not drive it with random play.**~~ **Withdrawn by D-020.** This
  instruction was never supported by measurement and is now contradicted by it.
  Round-robin over the action mask, with clicks aimed at segmented objects, was
  built and measured against random across all 25 games: it is indistinguishable
  on no-op rate and changed-pixel size, and **worse** on event diversity (36.9
  distinct events per game against random's 40.1). Random already reaches 99%
  action coverage at 60 actions, so round-robin's guarantee buys ~1pp. **Drive
  the experiment with random play**; the drivers live in `src/arc3/drive.py` if
  a comparison is ever wanted again.
- **Two click-only games are corpus-starved.** ft09 and lp85 run at ~90% no-ops
  under *every* driver tested. Report their dynamics results separately: they
  will have roughly a tenth of the evidence the other games do.
- **Size the compute first.** At 824 ms/frame on bp35 for one segmentation,
  tens of thousands of frames across 25 games is not a run you want to discover
  the cost of halfway through. Measure on one game, extrapolate, then commit.

**Both of D-015's smaller fixes are done (session 3).** `Tracker` and
`eval_tracking.py` now default to `colour_8`, and `segmentation_name` is
enforced in `Tracker.update` rather than stored and ignored. Enforcing it
immediately caught three tests that built `foreground_8` segmentations and fed
them to a tracker declared on a different carving — the exact silent error the
argument was meant to prevent, live in the suite since session 1.

**Two things session 3 adds to the experiment's design:**

- **Sweep the root first and merge outcome classes as you go (D-021, D-022).**
  The click's true branching factor is 1–11 on 13 of 15 live games, against
  proposal sets of 10–196. Collapsing targets that produce identical resting
  frames costs no extra actions and shrinks the corpus's click axis by 5–25×.
- **Do not let the corpus inherit the start-state assumption.** Click liveness is
  state-gated on 8 of 19 games, so a corpus driven from the root will
  systematically under-sample inputs that only come alive later — m0r0's click
  does nothing at all until roughly the fifth action. This is a *known bias in
  the corpus*, not a reason to delay, but per-game dynamics results should be
  read with it in mind alongside the ft09/lp85 starvation already flagged.

---

## Open questions and known risks

| Risk | Why it matters | Status |
|---|---|---|
| **Dynamics may not induce** from observed transitions | kills D-002 and most of the thesis | **untested — this is the next job** |
| Object identity may not track | kills D-005's pruning | **retired.** 0.8% switch rate (D-014) |
| **Fragmentation at 2.55** | one object read as several means several rules where one would do | live; watch it during dynamics induction |
| **Objecthood is not revisable** | segmentation is recomputed blind every frame; no evidence can ever change how a frame is carved, so fragmentation cannot self-correct | **open by choice** (D-016). The repair is a dynamics→segmentation feedback channel the code cannot currently receive |
| **Splits and merges are unrepresentable** | one-to-one matching means "one object became two" has no type. A split silently creates an unrelated new track | **open by choice** (D-016) |
| **Geometry earns its keep on 2 games** | D-011 built it to carry click pruning; segmentation carries that instead | accepted (D-015). Not a bug — it correctly declines on 21 games |
| **Tracking cost on cluttered games** | 824 ms/frame on bp35; the dynamics run wants tens of thousands of frames | **size this before the dynamics run, not during it** |
| ~~**§8 has no probe mechanism**~~ | with no Undo, every probe is permanent | **closed by D-022.** `RESET` restores the level start exactly, so revert costs *k*+1 and the argument is front-loading, not abandonment |
| **Do outcome classes drift with depth?** | if *what* a click does changes with state — not just *whether* it does anything — D-022's root-first ordering collapses | **open, and the highest-value cheap experiment now outstanding.** Liveness already drifts on 8 of 19 games (D-021) |
| **Click targeting fails on position-rich games** | r11l has 3,096 distinct click outcomes, su15 143; object targets reach 1.1% and 3.5% | open. Named worst cases for D-005, not averaged away |
| Hidden state may dominate | symbolic deterministic model would be the wrong representation | untested. Engine *is* deterministic given the full state, so this is about observability, not stochasticity |
| **The value function** | ranking intermediate states without knowing the goal | still the weakest link (design thesis §7). Now slightly better placed: UI and playfield are spatially separable, which is where a score region would live |
| D-012's metadata bet | if the private set withholds baselines, the planner loses its cost model | mitigated by the withheld-metadata path, which is tested |
| Scaffold headroom | if a plain harness reaches 60%+, this machinery is not worth it | watch the leaderboard and Milestone #2 |
| **Only 25 public environments** | survey reports 26.8pp inflation when evaluating on <100 tasks | discipline, not a test. Every number above is on 25 games and should be read with that in mind |

**Standing hazard:** local development allows infinite retries; RHAE punishes
them quadratically. Evaluate on first attempt from a cold start, and log
*actions consumed* alongside every result. `client/budget.py` exists to make
this the default rather than an afterthought.

---

## Parked, not dead

- **Per-region frame geometry.** The honest model of a composited frame is
  several cell sizes in one frame, not one. Deferred as not yet worth the
  complexity (D-013).
- **Oracle inside `perform_action`.** Would remove D-014's resting-frames-only
  limitation. Needs a renderer callback or an engine patch.
- **Meta-learning over procedurally generated games.** Probably right long-term;
  GPU-heavy, months of work, severe distribution-mismatch risk. See D-002.
- **Hybrid image+symbolic for the LLM-facing layer.** Tufa fed image + ASCII +
  segmentation together and won Milestone #1. Revisit when there is an
  LLM-facing layer to feed (D-006).

---

## Timeline reality

Milestone #2 on 30 September is **13 days** away and is not a realistic target.
The real deadline is **2 November**, **46 days** away. Between those sits the start of Michaelmas
term, so assume available time drops sharply from early October — front-load
anything needing sustained attention. The dynamics-induction experiment is that
thing, and it should happen next.
