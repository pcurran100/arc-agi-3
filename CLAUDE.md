# CLAUDE.md

Project instructions. Claude Code loads this automatically at the start of every
session. Read it, then read `docs/state-of-play.md` before doing anything else.

---

## What this project is

An agent for **ARC-AGI-3**, the interactive reasoning track of ARC Prize 2026.
Final submission deadline **2 November 2026**. Kaggle notebook, no internet at
evaluation, open-source required.

The thesis in one paragraph: ARC-AGI-3 scores agents on *action efficiency*
against a human baseline, with a quadratic penalty for extra actions, and
explicitly does not count internal reasoning as actions. That makes the exchange
rate between thinking and acting effectively infinite, which is exactly the
condition under which model-based methods beat model-free ones. We are building
an agent that **induces a symbolic model of the environment's dynamics from
observed transitions and plans against it internally**, calling an LLM only for
the narrow job its world prior is actually good for — proposing candidate goals.

Full argument in `docs/design-thesis.md`. Verified benchmark facts in
`docs/benchmark.md`. What everyone else has tried in `docs/field-review.md`.

---

## The documentation protocol

This is the part that matters most. The project is a research bet, and the value
of a research bet that fails is entirely in the record of *why* it failed and
what was ruled out. Treat the docs as the primary artefact and the code as
secondary.

### Every session

1. Read `docs/state-of-play.md` first. It is the single source of truth for what
   we currently believe and what we are currently doing.
2. Work.
3. Before the session ends, update `docs/state-of-play.md` if anything in it is
   now stale.

### Every non-trivial decision

Append an entry to `docs/decision-log.md`. Non-trivial means: it closes off an
alternative, it commits us to an interface or an approach, or a reasonable person
could have chosen differently. Library choice for a throwaway script is trivial.
"We represent objects as connected components rather than bounding boxes" is not.

Use this template, and **always fill in Options rejected** — that field is the
whole point of the log:

```markdown
## D-0NN — <short title>

- **Date:** YYYY-MM-DD
- **Status:** accepted | superseded by D-0NN | reversed
- **Context:** what prompted this, what we knew at the time
- **Decision:** what we are doing
- **Options rejected:**
  - *<option>* — why not
  - *<option>* — why not
- **Consequences:** what this commits us to, what it makes harder
- **Revisit if:** the concrete observation that would make us reopen this
```

Rules:

- The log is **append-only**. Never edit or delete a past entry to make it look
  better in hindsight.
- Never silently reverse a decision. Write a new entry, set the old one's status
  to `superseded by D-0NN`, and say in the new entry what changed our mind.
- `Revisit if` is mandatory. A decision with no falsification condition is a
  prejudice.

### Every experiment

Log it in `docs/decision-log.md` (or `docs/experiments/` once there are more than
a handful), including:

- the exact question it was meant to answer
- the setup, seeds, and how to reproduce it
- the result
- **whether it changed our mind, and if not, why not**

**Negative results are first-class.** An experiment that kills one of our
hypotheses is more valuable than one that confirms it, and it must be recorded
with the same care. If something did not work, the log says so plainly.

### Tone

Write the docs for a competent stranger reading them in six months — which may
be you. No hindsight polish, no pretending the path was straighter than it was.
If we spent two days on a dead end, the log says we spent two days on a dead end.

---

## Code conventions

- **Python.** Standard scientific stack. No dependency that cannot be vendored
  into an offline Kaggle notebook.
- **Perception is exact, not visual.** A 64×64 grid of 16 discrete colours is
  symbolic data. Use connected components, symmetry detection, periodicity,
  histograms, exact object matching. Do **not** render frames to PNG and ask a
  vision model what it sees — that trades exactness for nothing.
- **Determinism.** Seed everything, log seeds. Any experiment must be
  reproducible from the log alone.
- **Actions are the scarce resource.** Every function that can cause an
  environment step should make that obvious at the call site. Internal compute is
  free; treat it as free and spend it lavishly.
- **Typed interfaces at the seams.** Ontology, dynamics, goal and planner are
  separate subsystems (see design thesis) and must stay separable, because we
  expect to replace at least one of them wholesale.

## Guardrails

These come from specific findings, not taste. Each one has a source in
`docs/field-review.md`.

1. **No single-hypothesis chains.** Maintain a population of weighted hypotheses
   with soft down-weighting. Hard-rejecting on one disconfirming observation
   throws away hypotheses that are correct-but-conditional, and every ARC system
   above 70% on v1 maintains multiple candidates.
2. **Do not hand-craft the hypothesis space.** The Milestone #1 winner (Tufa
   Labs) found that hand-crafted tools *hurt* — letting the model improvise
   worked better. Induced structure good, human-guessed structure bad. If you
   find yourself enumerating "the kinds of games there are", stop.
3. **Do not conflate the three inference problems.** Ontology, dynamics and goal
   have different evidence economics and need different update rules. See
   design thesis §3.
4. **Do not spend actions on the goal early.** Dynamics is densely testable
   (one labelled example per action); goal is sparsely testable. Early actions
   go to dynamics disambiguation.
5. **No internet at evaluation.** Anything that requires a network call at
   inference time is disqualified. Prototype freely, but flag it.

---

## Current scope

**Session 1 and until further notice: the perception layer only.** Environment
client, frame diffing, connected-component segmentation, object identity tracked
across frames. Nothing downstream until this is exact and tested — see D-007 for
why, and for what was rejected in choosing this.
