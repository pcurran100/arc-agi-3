# State of play

**The single source of truth for what we currently believe and what we are
currently doing.** Read this first every session. Update it before the session
ends if anything below is stale.

- **Last updated:** 2026-08-30
- **Status:** pre-code. Scaffold and reasoning exist; nothing is built.
- **Days to Milestone #2 (30 Sep):** ~31 — realistically out of reach
- **Days to final submission (2 Nov):** ~64 — the real target

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

## Architecture as currently conceived

Four subsystems, deliberately separable because we expect to replace at least one
wholesale (D-004).

```
                  ┌──────────────┐
   frames  ───▶   │  ONTOLOGY    │  what objects exist, which is me,
                  │              │  identity tracked across frames
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
| Dynamics | every transition — one labelled example per action | cheap | **yes** |
| Goal | win / death / score change only | expensive, sparse | no |

Early actions go to **dynamics** disambiguation, never to goal probing.

---

## Decisions made so far

Seven, all backfilled from the design conversation and **none yet tested against
a real environment**. Full reasoning and rejected alternatives in
`decision-log.md`.

| ID | Decision | Confidence |
|---|---|---|
| D-001 | Enter ARC-AGI-3, not ARC-AGI-2 | high |
| D-002 | Model-based: induce symbolic dynamics, plan internally | **medium — this is the bet** |
| D-003 | Hypothesis population with soft down-weighting, not a single chain | high |
| D-004 | Separate ontology / dynamics / goal subsystems | high |
| D-005 | Prune the click space (4,096 → ~5–20) by object segmentation | medium |
| D-006 | Exact symbolic perception, no VLM at the primitive layer | high |
| D-007 | Session 1 = perception layer only | accepted, with a noted cost |
| D-008 | Develop against local `arc-agi` engine; never commit environment files | high |

---

## What is built

Nothing. This is the scaffold only.

**Setup not yet done** (first job of session 1):

```bash
pip install arc-agi            # pin the version
export ARC_API_KEY=...         # from https://three.arcprize.org/
```

Games cache to `environment_files/` on first run, then run fully offline. The
local engine is the same one the Kaggle gateway runs. `.gitignore`
`environment_files/` and `.env` (D-008). Clone
`arcprize/ARC-AGI-3-Agents` alongside as reference — do not vendor it.

**Standing hazard from day one:** local development allows infinite retries;
RHAE punishes them quadratically. Evaluate on first attempt from a cold start,
and log *actions consumed* alongside every result.

---

## Current focus

**The perception layer, and nothing else** (D-007).

Definition of done — all four, with tests, on the 25 public demo environments:

1. **Environment client** — connect, reset, step, receive frames, handle frame
   sequences for animated transitions, record every interaction to a replayable
   log.
2. **Frame diffing** — exact structural diff between consecutive frames. Not "the
   grid changed" but *which cells, which regions, which objects, how*.
3. **Segmentation** — connected components over the 64×64×16 grid. Multiple
   candidate segmentations, since the right notion of "object" is
   environment-dependent (4-connectivity, 8-connectivity, same-colour regions,
   bounding shapes). The survey attributes a 20–30 point advantage to robust
   multi-representation perception.
4. **Object identity across frames** — track components frame to frame through
   movement, occlusion, colour change, appearance and disappearance. **This is
   the hard one** and the one most likely to become a tarpit.

Explicitly **not** in scope yet: dynamics induction, goal proposal, any LLM call,
any planner, any submission machinery.

---

## Immediate next step after perception

Run the **dynamics-induction validation experiment** — the option deliberately
deferred in D-007. Log a few hundred transitions across the public demo
environments and test whether object-tracking plus simple rule induction predicts
held-out transitions better than chance.

This is the go/no-go on D-002 and therefore on most of the project. It should be
run as soon as perception is trustworthy, and it must not be quietly skipped
because the architecture already assumes the answer.

---

## Open questions and known risks

| Risk | Why it matters | Cheap test? |
|---|---|---|
| **Dynamics may not induce** from ~100 transitions | kills D-002 and most of the thesis | yes — the validation experiment above |
| **Object identity may not track** reliably | kills D-005's action pruning; search stays intractable | yes — hand-label a few sequences and measure |
| **Hidden state** may dominate | symbolic deterministic model is then the wrong representation | partly — look for transitions that violate every consistent rule set |
| **The value function** — ranking intermediate states without knowing the goal | the weakest link in the thesis (design thesis §7) | not yet — needs thought |
| **Scaffold headroom** may be larger than assumed | if a plain harness reaches 60%+, this machinery is not worth it | watch the leaderboard and Milestone #2 results |
| **Only 25 public environments** | survey reports 26.8pp inflation when evaluating on <100 tasks | discipline, not a test: never draw strong conclusions from this sample |

---

## Parked, not dead

- **Meta-learning over procedurally generated games.** Probably right long-term;
  GPU-heavy, months of work, severe distribution-mismatch risk. See D-002.
- **Static-to-interactive transfer as a standalone paper.** The living survey
  states nobody has established whether ARC-AGI-2 ability helps on ARC-AGI-3.
  Answerable with modest compute, and a more certain research output than winning
  the track. Worth keeping in view as a parallel objective.
- **Hybrid image+symbolic representation for the LLM-facing layer.** Tufa fed
  image + ASCII + segmentation together and won Milestone #1. Revisit when there
  is an LLM-facing layer to feed (D-006).

---

## Timeline reality

Milestone #2 on 30 September is ~31 days away and is not a realistic target given
perception-first scoping. The real deadline is **2 November**. Between those
sits the start of Michaelmas term, so assume available time drops sharply from
early October — front-load anything that needs sustained attention.
