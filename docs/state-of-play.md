# State of play

**The single source of truth for what we currently believe and what we are
currently doing.** Read this first every session. Update it before the session
ends if anything below is stale.

- **Last updated:** 2026-09-24, session 4
- **Status:** **the go/no-go on D-002 has been run. It is a weak pass.**
  Held-out exact-frame prediction is non-zero on 7 of 23 games where chance is
  exactly 0%, reaching 44% on dc22, and the object-level model works on 14 of
  23. Perception layer built, measured and audited; click space mapped against
  ground truth; design thesis §8 re-derived. Still no goal, no planner, no
  agent loop, no score.
- **Days to Milestone #2 (30 Sep):** **6** — out of reach
- **Days to final submission (2 Nov):** **39** — the real target

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
| D-024 | The rule language: `(action, selector) → effect`, MDL-selected | medium — deliberately at the floor of expressiveness |
| D-025 | **Dynamics experiment: weak pass. D-002 stands.** Keep all six carvings; language-limited not evidence-limited; bookkeeping layer is next | high — measured, 6 carvings × 25 games |
| D-026 | Corpus clicked random pixels and could not teach a click rule. `random_objects` is the new default | high — measured both ways |
| D-027 | **Target architecture: online, hypothesis-driven, discovery-minimising.** Single-observation hypotheses; acting and learning interleaved; frame modelled as a composite | high — sets the direction for the remaining time |

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
  drive.py          corpus policies: random, random_objects,          (D-026)
                    round_robin, lattice                              (D-020)
  dynamics/
    rules.py        the rule language and what it cannot say           (D-024)
    induce.py       observation extraction, and MDL rule selection
    predict.py      rendering a prediction, scoring, baselines
  oracle/           TEST-ONLY ground truth from engine sprites        (D-010)
scripts/            fetch_environments, record_fixtures, eval_geometry, eval_tracking,
                    size_pipeline, eval_baseline, eval_drivers,
                    eval_clickmap, eval_click_liveness                (D-021)
                    record_corpus, eval_dynamics, eval_dynamics_curve (D-025)
                    eval_click_evidence                               (D-026)
fixtures/           5 recorded episodes, 68 KB gzipped
corpus/             25 games x 100 actions, gitignored, 15 s to rebuild
tests/              197 tests, ~45 s
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
| **Dynamics: exact held-out frame** (D-025) | **best-per-game non-zero on 7/23 games, max 44.4% (dc22), against chance 0%.** Mean 4.8% under `colour_4` |
| Dynamics: object-level | changed-object accuracy ≥50% on **14/23**, ≥90% on 6/23; within 2 pixels on **13/23** |
| **Best carving is game-dependent** | `colour_4` wins 5 games, `foreground_4` wins 2 and is the only carving finding signal on ar25/cn04. Union 7 > any single 5 — D-018 is load-bearing |
| **Deformed moves: not the constraint** | `foreground_*` cuts deformation 3× (14% vs 53%) and scores **half** as well. Hypothesis refuted |
| Sweep cost | 7,888 s for 6 carvings × 25 games, against D-019's 2–2.5 h estimate |
| **Dynamics saturates at ~50 actions** | exact match gains **+0.0000** from n=50 to n=200 training transitions. **Language-limited, not evidence-limited** |
| **Four games: object model right, score ~zero** | ls20, sp80, **tn36 (0%→100% object accuracy, 0% exact)** and dc22 (100% object accuracy, 34.8% exact). A 2-pixel UI counter fails the frame comparison |
| **Click corpus was broken** (D-026) | clicks went to uniformly random pixels. Games with a learnable live click signature: **10/19 → 16/19** after aiming at objects |
| Corpus fix: effect on dynamics | click-game object accuracy **+6.8pp** (`colour_4`); exact-frame **flat**. The evidence improved, the headline did not |

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

## Immediate next step — superseded, see below

*The dynamics experiment described in this section has been run. Its result is
D-025 and the summary is in **Measured results** above. The section is kept
because its five design constraints all held up, and because the record of what
we expected is worth more than a tidy page. What is now next is at the end of the
document, under **What is next (session 4)**.*

### Original framing (sessions 1-2)

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
| ~~**Dynamics may not induce**~~ | kills D-002 and most of the thesis | **answered by D-025: it induces, weakly.** 7/23 games above a 0% chance baseline. Not refuted, not comfortably cleared |
| **16 of 23 games score exactly zero** | the mechanism works on a minority | open. D-025 argues a large part is residual bookkeeping rather than missing expressiveness, and names the test that would prove it wrong |
| **The metric conflates two failures** | sp80 has a *perfect* object model and scores 0% because a 2-pixel UI counter ticks every action | open, and deliberately not fixed by moving the goalposts. Headline stays exact-frame (D-019) |
| **No population of hypotheses** | Guardrail 1 and D-003 require one; dynamics uses the single MDL-best rule set | **open by choice**, a narrowing for the go/no-go only. First in the queue now the language is shown to work |
| **"50 actions suffice" may not survive the private set** | measured on random play over 25 public games; a game whose mechanics only appear after a long sequence would break it | open. The claim here most exposed to the private set |
| **Animation frames unused by dynamics** | up to 42 sub-transitions per action recorded, only endpoints read | open. Large unexploited evidence source |
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

---

## Objectives, in order (set by D-027, reordered by D-028)

The batch corpus pipeline has answered the question it was built for and is now
a **measurement harness, not the agent**.

**D-028 reordered this.** The sixteen zero-scoring games decompose as: **8
playfield-starved** (random play never makes the game do anything), ~1
UI-blocked, ~7 genuinely unmodelled. The largest category is an *exploration*
failure, not a learning one, and it is worth up to eight games where carving
selection is worth about one and layer separation about two.

| # | Objective | Worth | Status |
|---|---|---|---|
| 1 | **Exploration: make the game move on purpose** | up to **8 games**. vc33 changes the playfield on **0 of 99** actions under random play | **next** |
| 2 | **Layer separation: model UI and playfield apart** | **2 games** (sp80 0%→100%, dc22 44%→100%), and it is a *prerequisite* for 1 — the activity rate is what exploration optimises | with 1 |
| 3 | **Weighted hypothesis population with priors** | enables a hypothesis from a *single* observation, which the MDL point learner structurally cannot | after 2 |
| 4 | **Online loop: act on hypotheses, update from outcomes** | the actual agent | after 3 |
| ~~5~~ | ~~Carving selection~~ | ~1 game | **closed, negative (D-029).** Beats doing nothing on 51% of game-splits — a coin flip. Revisit after layer separation, when residuals can be measured on the playfield alone |
| 6 | **Multi-step rollout accuracy** | unmeasured, and planning depends on it entirely | any time |
| 7 | **End-to-end agent producing a real RHAE score** | every number here is an offline proxy | before 2 Nov |

## Current issues, named

Live problems with what is known about each. Distinct from the risk table above,
which is about the thesis; this is about the machine.

| Issue | What is known | Where |
|---|---|---|
| **UI layer defeats the headline metric** | 4+ games have a right-or-nearly-right object model and score ~0. sp80 = row 0 only, 14→0 every action. ls20 = rows 61–62, a move counter | D-025, D-026, objective 2 |
| **Learner cannot generalise from one observation** | MDL charges as much to state a rule as to log an exception, so a single event is never worth a rule. This is structural, not a threshold to tweak | D-027, objective 3 |
| **"Deformed" moves are mostly a carving artefact** | s5i5 and tn36 go from 100% deformed / 0% object accuracy under `colour_4` to **0% deformed / 100%** under `foreground_4`. Same game, same data. cd82 stays 100% under every carving and may be genuinely deformed | D-025, objective 1 |
| **8 games are playfield-starved, but less than D-028 said** | D-028 measured on the *superseded* `random` corpus. On `random_objects`: s5i5 2%→**41%**, vc33 0%→**16%**, ft09 7%→14%, su15 3%→13%, tn36 4%→11%. Targeting alone gives **2–20×**. All eight still score 0% exact | D-028, **corrected by D-030** |
| **Starved games are state-gated, not inert** | vc33 moves the playfield on **2 of 256** probes at the level start and **254 of 256** five actions later. The moves exist; a policy that notices responsiveness would find them | D-030, objective 1 |
| **`changed_obj_acc` can score blindness as success** | a change the carving cannot see cannot be scored wrong. `foreground_4` is blind to **24.7%** of object changes (`colour_4`: 2.8%). Never quote object accuracy without `blind_rate` | **D-030** |
| **D-021's click map overstates playfield effect** | it hashed the whole frame, so a click that only ticked the counter counted as a live pixel producing a distinct outcome. vc33 was recorded 100% live with 3 outcomes and has zero playfield activity. Its figures are upper bounds; recompute outside the band before reuse | D-028 |
| **No population of hypotheses** | Guardrail 1 and D-003 require one; the learner keeps only the MDL-best rule set | D-024, objective 3 |
| **42-frame animations unread** | Up to 42 sub-transitions recorded per action; dynamics reads only resting endpoints | D-024 |
| **Driver comparisons differ in train *and* test data** | D-026's two sweeps are different recordings, so per-game deltas mix better evidence with a harder test set. A clean rerun would hold the test set fixed | D-026 |
| **Object-model quality does not convert to score** | The central puzzle. Click-game object accuracy rose +6.8pp in D-026 with no headline movement | objective 2 |
| **Cannot choose a carving from 20 validation transitions** | The choice matters (oracle 7/23 vs fixed 5/23; s5i5 and tn36 swing 0%→100% object accuracy on carving alone) but no selection key is reliable. Mean residual looked like an 86% fix until a game-holdout showed 51%, i.e. chance | D-029 |

## Standing risks to the thesis

Unchanged from the risk table above. The one that matters most and is still
unmeasured: **whether predictions chain**. Every number in this repository is
single-step. Planning needs depth, and a model that is 44% right once may be
useless three steps out — or may be reliable within a well-behaved subset. The
data cannot currently distinguish those, and they have opposite consequences.
