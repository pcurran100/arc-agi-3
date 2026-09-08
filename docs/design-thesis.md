# Design thesis

Why we are building what we are building. This document is the argument; the
decision log records the commitments that follow from it. If the argument is
wrong, most of the decisions unwind together — so it is worth being precise about
what it rests on.

Written 30 August 2026, before any code existed. Expect it to be partly wrong.

---

## 1. The asymmetry everything hangs on

ARC-AGI-3 counts environment interactions as actions and explicitly does **not**
count internal reasoning steps. Scoring is `(human_actions / ai_actions)²`.

So:

- unlimited free compute for thinking
- quadratically expensive acting
- turn-based, so no time pressure while you think

The exchange rate between thought and action is effectively infinite. Almost no
benchmark has this shape, and it is precisely the condition under which
**model-based** methods dominate model-free ones — because a world model is
exactly a device for converting expensive real actions into free imagined ones.

Current leaderboard entries do not exploit this. They think a little, act, think
a little, act. The headroom is in doing enormous amounts of internal work per
environment step.

**This is the load-bearing claim.** If it is wrong — if in practice the dynamics
are too irregular to model from a handful of transitions — the rest of this
document collapses. See §7.

## 2. Why not the obvious alternatives

**Why not just a better LLM harness?** That is what the entire Milestone #1
podium is, and it scores ~30%. The approach has no queryable model of the
environment, so it cannot plan; and it re-derives the mechanics on every level.
We think the ceiling is low, though we should hold this loosely — Poetiq's result
on ARC-AGI-2 (54% at half the cost of the prior SOTA, purely by building a better
system around frontier models) is evidence that scaffold headroom can be larger
than it looks.

**Why not deep RL?** Because RHAE disqualifies it. MuZero-class methods need
millions of frames; here the budget is a few hundred actions with a *squared*
penalty. Model-free RL is the least sample-efficient family we have and the
metric is specifically a sample-efficiency metric. This is why there is no
MuZero on the leaderboard — the metric excludes it, not lack of imagination.

**Why not a specialised net trained from scratch?** TRM (7M params) and
CompressARC (76K params) prove you do not need an LLM for static ARC. But v3's
hard part is not grid transformation — it is inferring *what the goal is* in an
undescribed environment. That needs a prior over what games are: that things can
be collected, walls block, keys open, a rising counter is good. A net trained
only on ARC grids has no such prior and cannot acquire one from held-out eval
environments. Broad world knowledge does real work here, and it is the one thing
an LLM has that a specialist does not.

## 3. Three inference problems, not one

The central structural claim. What looks like one "figure out the game" problem
is three, with different evidence economics:

| | What it is | Evidence | Cost to test |
|---|---|---|---|
| **Ontology** | what objects exist, do they persist, which one is me | every frame | cheap, passive |
| **Dynamics** | what does action *a* do to object *o* | every transition | **cheap — one labelled example per action** |
| **Goal** | what is optimised, what ends the level | win / death / score change | **expensive — sparse** |

Consequences:

- One update rule cannot serve all three. Conflating them means the update rule
  is wrong for at least two.
- **Early actions should go to dynamics, not goal.** Every action manufactures a
  dynamics training example whether you want one or not; goal evidence only
  arrives on rare events. An agent "testing what the goal is" early is spending
  the expensive resource on the cheap-to-infer-later question.
- The LLM's world prior is genuinely useful for **goal** and nearly useless for
  dynamics (which is exactly, precisely observable). So call it rarely and only
  for that.

## 4. Dynamics induction is the same problem as ARC-AGI-2

This is the transfer nobody appears to be making.

- An ARC-AGI-2 solver infers a transformation from `(input grid, output grid)`
  example pairs.
- A dynamics learner infers a transformation from
  `(frame_t, action, frame_{t+1})` triples.

Same inductive problem, different data source — and the data arrives **online**,
one pair per action, instead of upfront. All the machinery the field built in
2025 ports: DSL search, natural-language program evolution, MDL-based selection.

Practical notes:

- **Do not fit a neural transition model.** With ~100 observed transitions you
  cannot train a dynamics CNN. That is the failure mode of treating this as
  standard model-based RL.
- These are hand-built grid games, so the dynamics are likely local, sparse,
  mostly deterministic, object-level: a sprite translates one cell, a region
  toggles, a counter increments, a collision blocks. Small hypothesis space.
- **MDL is the selection criterion** — keep the shortest rule set consistent with
  everything observed. Same principle as CompressARC.

## 5. Population, not chain

Maintain a weighted **population** of hypotheses, pruned by consistency, not a
single belief that gets replaced when contradicted.

Two reasons. First, the survey lists "multiple hypotheses: diverse candidate
maintenance with consistency filtering" as one of six properties shared by every
system above 70% on ARC-AGI-1 — single-chain reasoning is a known-dead approach
in this benchmark family. Second, **hard rejection is unsound here**: there is
plausibly hidden state (an inventory flag, an internal counter, a phase), so a
hypothesis that is correct-but-conditional will produce a disconfirming
observation. Down-weight, do not delete.

## 6. Search, and the click problem

Naive worry: "we can't render forward to the end of the game, so we need a value
heuristic." Half right. Check the arithmetic:

- Five control keys → branching factor ~5 → depth 10 is ~10M nodes. Trivially
  searchable, and **free**, because internal reasoning is not an action.
- Cell-selection is parameterised over 64×64 = **4,096** coordinates. *That* is
  what explodes the branching factor.

So the binding constraint is not depth, it is the click. And the fix is
**upstream, in ontology**: do not consider clicking arbitrary cells, consider
clicking *objects*. Segment into connected components and only propose clicks on
distinct object instances — 4,096 collapses to perhaps 5–20 candidates, and real
depth becomes affordable.

**Better perception buys search depth.** That is the main reason perception is
being built first (D-007).

## 7. The value function — the genuinely unsolved part

Even with correct dynamics and a correct goal hypothesis, you cannot search to
terminal, so you need to rank intermediate states. Available signals, best first:

1. **A score region.** If a counter or score display can be localised in the
   grid, you have a *dense* reward signal and the value problem largely
   dissolves. Finding it should be a top-priority early inference, not an
   afterthought.
2. **Level transitions.** ≥6 levels per environment; a transition is unambiguous
   positive feedback.
3. **Irreversibility.** Undo exists, so actions can be cheaply classified as
   reversible or not. Irreversible state changes correlate with progress. Free
   heuristic, needs no goal knowledge.
   *(Correction, session 1: Undo does not exist — D-009. The signal survives but
   the mechanism does not: reversibility can no longer be tested by trying it,
   only predicted from the induced dynamics model, which makes this heuristic a
   consumer of §4 rather than an independent free signal.)*
4. **Reachability frontier.** Prefer actions expanding the reachable state set.
5. **Novelty / state-visitation.** Cheapest to implement, worst fit for RHAE
   because it burns actions. Tiebreak only.

This is the weakest link in the thesis and the most likely place to be wrong.

## 8. Exploration is a value-of-information problem

"Be greedy toward the assumed goal" is wrong early, because the assumed goal is
not yet trustworthy. The right early objective is **information gain** — expected
reduction in posterior entropy over the hypothesis population.

But it is not two clean phases either, because the quadratic penalty means we
cannot afford much pure exploration. It is an explicit **value-of-information
against action cost** trade on every single step. That is a well-posed decision
problem and, as far as we can tell, nobody has posed it for this benchmark.

Undo matters here: it lets you probe a risky action and revert without destroying
your position, at a cost of two actions per probe. Undo is an action and it
counts.

> **Correction, 2026-08-30 (session 1) — the paragraph above is wrong.**
> There is no Undo action. `arcengine` exposes `RESET` plus `ACTION1`–`ACTION7`
> and keeps no state history; `RESET` restarts the *current level* and does not
> rewind the action count. See `benchmark.md` § *Correction: there is no Undo
> action*, logged as D-009. Left in place rather than edited away, because the
> rest of this section was written on the assumption that cheap reversible
> probing existed.
>
> The argument has to be re-derived without it, and the conclusion moves: with
> no rollback, every probe is **permanent**, so the cost of an uninformative
> action is not "two actions" but "two actions plus whatever state damage it
> did". That pushes hard toward extracting information from *free* sources
> before spending anything — the frame sequences returned by a single action
> (up to 42 frames, §10), the published `available_actions` mask, and the
> letterbox/UI geometry — and it makes §7's irreversibility heuristic more
> important, not less, since irreversibility is now the default rather than an
> occasional hazard. This re-derivation has not yet been done properly and is
> owed its own decision-log entry.

## 9. Cross-level persistence is free score

Mechanics are shared across the ≥6 levels of an environment. Scoring is
per-level. So the cost of learning concentrates entirely in level 1, and levels
2 onward should run near-optimally if the induced model is right.

A stateless harness re-derives the mechanics every level and pays the tuition six
times. This is a concrete, measurable inefficiency in the current field and is
plausibly a large fraction of the gap between 30% and 100%.

**Ontology and dynamics persist across levels. Goal and layout are re-inferred.**

## 10. Perception is not vision

A 64×64 grid of 16 discrete colours is symbolic data. Rendering it to PNG and
asking a vision model what it sees trades exactness for nothing and produces
agents that hallucinate walls.

Use exact algorithms: connected components, symmetry detection, periodicity,
colour histograms, object matching across frames. Tufa's stack fed image + ASCII
+ segmentation together, which suggests redundant representations help *the
model* — but the primitives underneath must be exact.

---

## What would falsify this thesis

Listed so we notice if it happens, rather than rationalising.

- **Dynamics do not induce.** If simple object-tracking plus rule induction
  cannot predict held-out transitions better than chance on the public demo
  environments, §4 is wrong and the whole model-based plan goes with it.
- **Objects are not stable.** If connected-component identity cannot be tracked
  reliably across frames, §6's action-space pruning fails and search stays
  intractable.
- **Hidden state dominates.** If most environments turn out to have substantial
  unobservable state, the deterministic-symbolic model is the wrong
  representation.
- **The value problem is fatal.** If no usable progress signal can be extracted
  in most environments, correct dynamics will not be enough.
- **Scaffold headroom is larger than we think.** If a well-built plain LLM
  harness gets to 60%+, the marginal value of all this machinery drops sharply
  and we should be doing something else.

Each of these has, or should acquire, a cheap test. Write them into the decision
log as they get one.
