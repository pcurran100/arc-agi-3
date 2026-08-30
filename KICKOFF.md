# Session 1 kickoff prompt

> Paste the block below into Claude Code from the root of this repo. Everything
> it needs is already in the repo — this prompt orients it and sets the scope.
> After session 1 you should not need this file again; `CLAUDE.md` and
> `docs/state-of-play.md` carry the context forward.

---

We are building an agent for **ARC-AGI-3**, the interactive reasoning track of
ARC Prize 2026. Final submission deadline is 2 November 2026.

**Before you write any code, read these in order:**

1. `CLAUDE.md` — project instructions, the documentation protocol, and five
   guardrails that each come from a specific published finding
2. `docs/state-of-play.md` — what we currently believe and what we are doing
3. `docs/design-thesis.md` — the argument the whole project rests on
4. `docs/benchmark.md` — verified mechanics, scoring, constraints
5. `docs/decision-log.md` — the seven decisions already made, and what was
   rejected in making them
6. `docs/field-review.md` — prior art; read at least §1 and §4 before designing
   anything

These were written in one design conversation before any code existed. They are
argued but untested. Treat them as a starting position to be attacked, not
received wisdom — if you think a decision is wrong, say so and we log the
disagreement.

## Scope for this session

**The perception layer only.** This is decision D-007 and the rejected
alternatives are recorded there — read it, including the noted cost of this
choice.

Do not build dynamics induction, goal proposal, any LLM call, any planner, or any
submission machinery. Everything downstream depends on perception being exact, so
a subtle bug here would silently corrupt every later result. That is the entire
argument for this scoping.

### Definition of done

Four components, each with tests, working across the 25 public demo environments:

1. **Environment client** — connect, reset, step, receive frames, handle frame
   sequences for animated transitions. Record every interaction to a replayable
   log so experiments are reproducible offline without burning actions.
2. **Frame diffing** — exact structural diff between consecutive frames. Not "the
   grid changed" but which cells, which regions, which objects, and how.
3. **Segmentation** — connected components over the 64×64×16 grid, producing
   *multiple candidate segmentations* (4-connectivity, 8-connectivity,
   same-colour regions, bounding shapes). The right notion of "object" is
   environment-dependent, and the living survey attributes a 20–30 point
   advantage to robust multi-representation perception.
4. **Object identity across frames** — track components through movement,
   occlusion, colour change, appearance and disappearance. This is the hard part
   and the most likely tarpit. If it is still unreliable after roughly two weeks,
   that is a signal to reassess the symbolic approach entirely (see D-007's
   *Revisit if*).

### Start by

Confirming the ARC-AGI-3 API surface and current Kaggle compute limits against
the live docs, since `docs/benchmark.md` was compiled on 30 August 2026 and the
competition is live. If anything there is now wrong, correct the file **and** log
the correction — we may have reasoned from the stale fact.

Then propose a module layout before implementing, and let's agree it.

## How to work

Read the documentation protocol in `CLAUDE.md` properly — it is the part of this
project I care most about.

In short: every non-trivial decision gets an entry in `docs/decision-log.md`
using the template, **including the options you rejected and why**. Never
silently reverse an earlier decision — supersede it explicitly. Every experiment
gets logged with its setup, seeds, result, and whether it changed our mind.
Negative results are first-class and get recorded with the same care as positive
ones. Update `docs/state-of-play.md` before the session ends if anything in it
has gone stale.

The project is a research bet. If it fails, the entire value of it is the record
of why and of what was ruled out. Write the docs for a competent stranger reading
them in six months.

## One standing instruction

Push back. If the thesis looks wrong once you have real frames in front of you,
say so early and loudly rather than building on it. The falsification conditions
are listed at the end of `docs/design-thesis.md` — watch for them, and tell me
if you see one.
