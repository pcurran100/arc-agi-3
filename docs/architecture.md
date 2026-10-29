# Architecture: what is built, how it works, and what it does not do

- **Last updated:** 2026-09-24, session 4
- **Scope:** the perception layer (§1-§6), dynamics induction (§7) and
  exploration (§8). Goal proposal and the planner still do not exist, and
  nothing yet plays *to win* — §8 plays to make things happen.
- **Companion documents:** `design-thesis.md` argues *why* we are building this
  kind of agent. `benchmark.md` records what the environment actually does.
  `state-of-play.md` is the short status snapshot. `decision-log.md` is the
  append-only record of choices. **This document describes the machine.**

This is written for a competent stranger — quite possibly one of us in six
months — who needs to know what the code does without reading all of it, what it
demonstrably achieves, and, most importantly, where the holes are. Every number
quoted here was measured, and §4 says how. Sections headed **What this does not
do** are not disclaimers bolted on at the end; they are the part of the document
most likely to save someone a wasted week.

---

## 1. The problem this layer solves

The environment hands us a 64×64 array of integers in the range 0–15. That is
all. There is no notion of an object, a player, a wall, or a score anywhere in
that array. A single action returns not one such array but a *sequence* of up to
42 of them — the animation the action produced — and zero frames is a legal
return meaning "the action was accepted and changed nothing".

Everything downstream in the intended architecture — dynamics induction, goal
proposal, planning — needs something better than integers. Specifically it needs
to be able to say things like *the player moved right by one cell and stopped
against a wall*. Before any of that is possible, something has to turn

> 34 of the 4,096 numbers changed

into

> one object, the 20×4 colour-9 bar, moved 4 pixels right; the other six objects
> did not move; nothing appeared or vanished.

That translation is the entire job of the perception layer. It is a
*description* problem, not an understanding problem. The layer is not supposed
to know what the bar is for.

A commitment inherited from D-006 shapes everything below: **perception is exact
and symbolic, never visual.** A 64×64 grid of 16 discrete colours is symbolic
data. We use connected components, exact shape matching, run-length arithmetic
and integer geometry. We never render a frame to an image and ask a vision model
what it sees, because that trades exactness for nothing.

---

## 2. The pipeline

Five stages. Each is a pure function of its inputs except the last, which is the
only stateful thing in the layer.

```
frames --> geometry --> segmentation --> diff --> identity --> action pruning
           (2.1)        (2.2)            (2.3)    (2.4)        (2.5)
```

### 2.1 Frame geometry — what granularity does this game move at?

Some games draw a small board scaled up. sp80's playfield is genuinely 16×16,
painted at 4× so it fills the frame. Miss that and every object is sixteen times
too big in area, every displacement is four times too long, and every rule
induced downstream is stated in the wrong units.

The original framing of this problem (D-011) was "recover the camera's upscale
factor". First contact showed that framing to be wrong: the 64×64 frame is a
*composite* — a scaled camera view, letterboxed, with a UI layer painted last at
full resolution over the top. On some games the UI is most of what you can see,
and "the upscale factor" is not a well-posed question. So the module answers a
different and better-posed one: **at what granularity does this game's content
actually move?**

The method is propose-then-verify, and neither half has a tuned threshold
(D-013):

1. **Propose.** Take the mask of pixels that change across the observed frames.
   Inside it, take the GCD of every interior colour-run length. A genuine `s`-times
   upscale makes every run a multiple of `s`. Runs touching the frame edge are
   excluded, because the frame truncates them and a truncated length would drag
   the GCD wrongly toward 1.
2. **Verify.** Check that every cell the proposal implies holds exactly one
   colour in every observed frame. If not, walk down the proposal's divisors
   until one does. A cell size of 1 always verifies, so the pipeline is total and
   its failure mode is *"no structure claimed"*, never *"wrong structure
   claimed"*.

That last property is the design's whole point. Declining to guess is a
first-class answer.

**What this does not do.** It induces exactly one cell size for the whole frame.
The honest model of a composited frame is several cell sizes in one frame — the
camera at one scale, the UI at another. That is parked, not solved (D-013), and
it is the reason the layer is near-useless on UI-heavy games. §3.1 measures how
often this bites.

### 2.2 Segmentation — what are the objects?

There is no correct answer to this question in general, and the module refuses
to pretend otherwise. It computes **six** different carvings of every frame:

| name | rule | the case it is right for |
|---|---|---|
| `colour_4`, `colour_8` | maximal same-colour connected regions | the classic ARC notion of an object |
| `foreground_4`, `foreground_8` | connected non-background regions, colour ignored | a multi-coloured sprite that should stay one object |
| `colour_mass` | every cell of one colour, connectivity ignored | scattered collectibles that behave as a set |
| `bbox` | each same-colour region's bounding box, filled | a hollow outline that means the region it encloses |

`_4` and `_8` are 4- and 8-connectivity — whether diagonal neighbours count.

Producing a population of candidate representations rather than committing to
one is deliberate, and it is the one thing the living survey attributes a
standalone 20–30 point advantage to (`field-review.md` §4). It also stays on the
right side of Guardrail 2: these are general representational rules, not guesses
about what kinds of games exist.

Segmentation runs in **cell space** when geometry verified block structure, and
in pixel space otherwise.

**What this does not do.** Nothing anywhere in the codebase ever *chooses* among
the six, or scores them, or narrows the population as evidence accumulates. They
are computed and handed onward. Downstream code names one by string and uses it.
The population exists as raw material for a judgement that has not been written.
See §3.4. D-018 makes the choice an explicit axis of the dynamics experiment,
which is the first use that does not simply pick one and hope; it does not yet
give the agent any way to choose at run time.

### 2.3 Difference — what changed?

Given two frames: which pixels changed, which colour transitions occurred and
how often, and the changed pixels grouped into connected regions.

The useful entry point is `diff_sequence`, which summarises a whole animation
rather than its endpoints. This matters more than it sounds. An action returns up
to 42 frames, and the intermediate ones show the *trajectory* of a motion rather
than only where it ended. A rule that has to predict the whole trajectory is far
more constrained — and therefore far more falsifiable — than one that only has to
predict the endpoint. Design thesis §3 budgeted "one labelled example per
action"; the real figure is up to 42 sub-transitions, which is the single
correction from first contact that made the central bet *more* likely to work
rather than less (D-009).

### 2.4 Identity — which object is which?

This is the hard stage, the one D-007 named as the project's likely tarpit, and
the one that works best.

Given a segmentation of frame *t* and of frame *t+1*, decide which component in
one *is* which component in the other, through movement, recolouring, occlusion,
appearance and disappearance.

Two commitments shape it.

**Matching is global, not greedy.** The obvious approach — walk each object and
take its nearest match — fails exactly where identity matters, on two identical
sprites passing each other. Instead every possible pairing is costed and the
cheapest overall assignment is chosen in one step, by the Hungarian /
Jonker–Volgenant algorithm. With at most a few dozen components this is
affordable, and internal compute is free. There is a regression test for the
crossing case.

The cost between two components is a ladder of exact evidence, strongest first:

| evidence | cost |
|---|---|
| identical shape *and* colouring | 0.0 |
| identical silhouette, different colouring — a toggle or state flip | 0.25 |
| neither | 0.5, plus a term from mask overlap and size ratio |

plus a saturating penalty for distance moved, so that between two otherwise
identical candidates the nearer one wins, while a long jump is still preferred
over inventing an appearance and a disappearance.

A pairing is made only if it costs less than `UNMATCHED_COST`, which is the price
of the alternative explanation — one thing vanished and a different thing
appeared. This is what sets the bar for "same object" **without a tuned
similarity threshold**: the threshold is the cost of the competing story.

**No hard rejection** (Guardrail 1, D-003). A track that loses its object is not
deleted. Its confidence decays and it goes dormant, remembering what it looked
like, so that when something matching shows up later it can be revived rather
than replaced. That is what makes occlusion survivable instead of
identity-destroying.

**The carving is declared and enforced.** A tracker is constructed for one named
segmentation and refuses to be fed another: identity across a mixed carving is
meaningless, and until session 3 that requirement was a docstring only —
`segmentation_name` was stored and never read. Enforcing it in `Tracker.update`
immediately failed three tests that had been building `foreground_8`
segmentations and feeding them to a default-constructed tracker since session 1.
The default is now `colour_8`, which is the carving D-014's 0.8% id-switch rate
was actually measured on; it had been `foreground_8`, a mismatch D-015 spotted
and session 3 closed.

**What this does not do.** Three things, and they are the most consequential
limitations in this document — §3.2, §3.3 and §3.5 treat them properly.

### 2.5 Action pruning — where is it worth clicking?

`ACTION6` takes a pixel coordinate, so its action space is 4,096 wide. Searching
that is hopeless. Since only objects are plausibly worth clicking, each
segmentation proposes one click target per component, and the union across all
six is offered. Targets are guaranteed to land *inside* their component — the
centroid when that lies within the mask, otherwise the first mask pixel in raster
order — because a click one pixel outside the object lands on something else.

The union rather than any single segmentation's list, because D-005's stated
failure mode is pruning away the winning action, and we do not yet know which
notion of object is right for a given game.

**Measured against ground truth (D-021, session 3).** Until session 3 this
component's output was a *proposal* with no denominator: 8–196 candidates, and
no measurement of how many distinct things a click can actually do.
`scripts/eval_clickmap.py` now clicks all 4,096 pixels on each of the 19 games
that publish `ACTION6`, `RESET`-ing between each, and hashes the resting frame.
On sp80, seed 0:

```
game              live%  classes  largest  obj n  obj prec  obj cov  lat prec  lat cov
sp80-589a99af     100.0        1     4096     10    100.0%   100.0%    100.0%   100.0%
vc33-5430563c     100.0        3     4064     16    100.0%   100.0%    100.0%    33.3%
ft09-0d8bbf25       7.0        8       36     71     12.7%   100.0%     12.5%   100.0%
```

Three things to read off. **sp80 has exactly one outcome class**: all 4,096
pixels produce the identical frame, so `ACTION6` there is a button and its
coordinate is decorative — the 10 proposals are 10× over-complete. **vc33 shows
why the union is the right call**: object targets reach all 3 of its outcome
classes, a 64-point lattice reaches 1. **ft09 shows precision is the wrong
headline**: 12.7% of its proposals are live, and they still reach 100% of its
distinct outcomes. A planner needs one pixel per outcome, not every live pixel.

Across the 19 games the true branching factor is **1–11 on 13 of 15 live games**,
against proposal sets of 10–196 — so the pruning is real and large but stops
5–25× short of what is there. The two exceptions, r11l (3,096 outcomes) and su15
(143), are genuinely position-rich and the object prior reaches 1.1% and 3.5% of
their outcomes respectively.

**What this does not do.** It does not merge outcome classes — the union is
still emitted whole, and nothing collapses two targets that produce identical
frames, although D-021 decides that it should. It does not re-check liveness:
the map is computed at one state, and `scripts/eval_click_liveness.py` shows
liveness is state-gated on 8 of 19 games (m0r0 is 0% live at the level start and
100% five actions later), so a target set computed once and trusted will be
wrong on those games. It does not know that four games (ar25, m0r0, sc25, sk48)
publish `ACTION6` while no pixel responds at the start state at all. And neither
script may ever be imported from the agent's code path: both reach ground truth
by brute force that is free offline and permanent at evaluation, which puts them
on the same footing as the sprite oracle (D-010).

---

## 3. What it measurably does, and does not

All measurements below are from session 2 (2026-09-08) unless marked session 3
(2026-09-17), seed 0, reproducible by the commands in §4. Where a number contradicts an earlier claim in the
repository, the earlier claim is superseded and D-015 records the correction.

### 3.0 A worked example

sp80, first non-trivial transition. Geometry has induced `cell=4`, origin (0,0),
a 16×16 cell grid.

```
TRANSITION #0: action = ACTION4, animation = 1 frame
  raw pixels changed : 34 of 4096, in 3 region(s)
  objects before/after: 7 / 7
  matched            : 7   appeared: 0   vanished: 0
  MOVED: object of colour 9, 80 px, shape 20x4
         from (12,16)-(32,20) to (16,16)-(36,20)
         displacement (4, 0), same shape=True, confidence=0.87
  UNCHANGED: 6 objects held position
```

Read the transformation: 34 scattered integer changes in, and out comes *one bar
moved right by one cell, nothing else touched, nothing created or destroyed*.
That statement is exactly the input a dynamics learner needs, and it is what the
layer is for.

### 3.1 Geometry is near-inert, and one of its answers is wrong

Across all 25 public demo games, 150 random actions, seed 0 — D-013's protocol,
which reproduces its 18/23 exactly:

- Structured geometry is found on **4 of 25 games** — ft09, lp85, m0r0, sp80.
- On the other 21 the answer is `cell=1`: no structure claimed.
- Headline agreement with the true camera scale is **18/23** on camera-visible
  games. But **16 of those 18 are `cell=1` correctly matching a true scale of
  1** — credit for correctly saying "nothing here" where there is nothing. Only
  ft09 and sp80 are non-trivial successes.
- Five games disagree: ar25 (true 3, induced 1), cn04 (3→1), lf52 (8→1),
  vc33 (2→1) — all underestimates, the safe direction — and **lp85 (true 2,
  induced 4), the one overestimate**, already on record in D-013.

So the fair statement is: geometry earns its keep on two games, is harmlessly
silent on twenty-one, and is actively wrong on one. The `18/23` is not false,
but read alone it flatters the component considerably.

**The answer depends on how long you have watched, and the failing game
migrates.** At 20 actions the overestimate is dc22 (true 1, induced 2) and lp85
underestimates; by 150 dc22 has resolved and lp85 has become the overestimate.
D-013 recorded that accuracy improves monotonically with frames, which is true,
but the *identity* of the wrong answer changes underneath it. Anything that
re-induces geometry mid-episode is therefore re-inducing a different wrong
answer at different points, not converging on a single one.

A second correction. `segment.py` states that cell-space and pixel-space are
"roughly a half-and-half split, so both paths are load-bearing". Measured, it is
4/25. The pixel-space path does nearly all the work.

### 3.2 Objecthood is a fixed rule, not a belief

This is the deepest structural limitation of the layer.

**Segmentation is recomputed from scratch on every frame, by a fixed rule,
looking only at that frame's pixels.** It has no memory. Frame 400 is carved up
exactly the way frame 1 was, gaining nothing from 399 frames of observed
interaction in between.

What *does* persist is identity — "the thing I called object 3 is this thing
now". The tracker remembers which is which. It never revises **what counts as an
object in the first place**.

The two questions are cleanly separated in the code, and only one has any
machinery behind it:

| question | status |
|---|---|
| *Which object is which?* | tracked across time, measured, works |
| *What are the objects?* | recomputed blind every frame, never learned |

There is no feedback path. `track.py` imports from `segment.py`; nothing goes the
other way. No evidence gathered by tracking can ever change how a frame is
carved.

This matters because the layer's known weak spot is **fragmentation at 2.55
components per ground-truth sprite** — we routinely see one object as two or
three. The natural repair is exactly the feedback that does not exist: *these
pieces have moved together every time we have seen them, so treat them as one.*

### 3.3 Splits and merges are structurally unrepresentable

Tested directly. Take a bar occupying columns 10–20 and split it into two pieces
at 10–14 and 16–20:

```
frame 0: one bar        ->  tracks created
frame 1: two bars       ->  matched=2, appeared=[1 new component]
result : the original track continues as one piece
         the other piece becomes a brand new, unrelated track
```

There is no record that the two were ever the same object. In reverse, merging
two objects into one:

```
frame 0: two bars       ->  tracks created
frame 1: one bar        ->  matched=1, vanished=[1]
result : one track continues; the other is marked missing,
         decays to confidence 0.6, and sits dormant forever,
         never reconciled with the thing it merged into
```

This is not a bug awaiting a patch. Assignment is **strictly one-to-one by
construction**, so "one object became two" has no representation in the type
system. The words *split* and *merge* do not appear in `track.py` at all.

A related wart surfaced by the same test: with `colour_8`, the background is
itself a component — one of the "objects" is a 4,088-pixel blob of empty space.

### 3.4 Nothing selects among the six segmentations

Stated in §2.2 and repeated here because it is easy to miss: the candidate
population is generated and never evaluated. `Tracker` takes a
`segmentation_name` argument, stores it, and **never reads it** — it is a
decorative label. The docstring warns that mixing segmentations frame-to-frame
would make identity meaningless, and nothing enforces it.

Worse, the default is `foreground_8`, which D-014 explicitly disowns: it
over-merges at 3.65 sprites per component, and its flattering 0.2% switch rate is
an artefact of collapsing whole frames into one or two blobs where identity
trivially cannot change. D-014's conclusion was that **`colour_8` is the working
default**. The code says otherwise. This is a straightforward divergence to fix.

### 3.5 Confidence barely means anything

Tracks carry a confidence that is meant to implement D-003's soft down-weighting.
On a successful match it updates as

```
confidence <- min(1, 0.5*confidence + 0.5*match_quality + 0.25)
```

The `+0.25` term dominates. A track matched *terribly* on every frame still
converges to 0.5, far above the dormancy floor of 0.1. Confidence therefore
ratchets upward almost regardless of evidence and is close to uninformative as a
quality signal. The down-weighting is softer than the design intends.

### 3.6 What the layer does deliver

Against all of the above, the core of the layer works, and this is the evidence.
For every non-trivial transition in the five recorded fixtures, using `colour_8`:

| game | transitions | changed px | change regions | moved | recoloured | appeared+vanished | **changed px explained** |
|---|---|---|---|---|---|---|---|
| ls20 | 25 | 42 | 1.8 | 1.6 | 2.8 | 0.0 | **100%** |
| sp80 | 20 | 58 | 2.2 | 1.7 | 0.8 | 0.1 | **79%** |
| bp35 | 25 | 18 | 1.7 | 0.7 | 2.7 | 0.0 | **100%** |
| vc33 | 25 | 1 | 1.0 | 0.6 | 1.4 | 0.0 | **100%** |
| sb26 | 8 | 1 | 1.0 | 0.0 | 1.9 | 0.1 | **100%** |

"Changed px explained" is the fraction of genuinely-changed pixels covered by the
objects the layer flags as moved, recoloured, appeared or vanished. At ~100% on
four of five games, the symbolic story accounts for essentially the whole pixel
delta: nothing is being silently dropped. Appearances and disappearances average
near zero, meaning the tracker almost never invents or loses an object across a
real action.

Two supporting results:

- **The assignment algorithm is correct.** `_hungarian` is hand-written rather
  than taken from scipy, to keep the offline dependency set minimal. Fuzzed
  against brute-force optimal assignment on 4,000 random rectangular cost
  matrices: **zero suboptimal results.**
- **Click pruning is real but overstated.** Measured candidates per frame:
  sp80 8, vc33 16, ls20 29, sb26 31, **bp35 196**. Against a raw space of 4,096
  that is a 20× to 500× reduction, but the previously quoted range of "15–25"
  holds on one of five fixtures. Separately, `Geometry.click_targets()` returns
  **empty on four of the five**, and on the fifth returns 256 where segmentation
  returns 8 — so in practice the pruning comes from segmentation, not from the
  geometry layer D-011 built for it.

  **Session 3 supplies the denominator this paragraph lacked** (§2.5, D-021).
  "Overstated" was the right instinct for the wrong reason: the proposal counts
  are not too *large* relative to 4,096, they are too large relative to the
  number of distinct outcomes a click actually has, which is **1–11 on 13 of the
  15 games where clicking does anything**. bp35's 196 candidates cover 8 real
  outcomes. Reproduce: `python scripts/eval_clickmap.py` (~4 min) and
  `python scripts/eval_click_liveness.py` (~10 min), seed 0, live engine
  required, no fixtures.

### 3.7 Cost

Tracking time per frame, one segmentation, `colour_8`:

| game | components | ms/frame |
|---|---|---|
| sp80 | 7 | 20 |
| ls20 | 19 | 70 |
| vc33 | 11 | 84 |
| sb26 | 26 | 145 |
| **bp35** | **191** | **824** |

The cost matrix is built by a Python double loop of full-frame array operations,
so it grows quadratically in component count. Note the correlation: the games
where geometry declines to find structure are the games that segment in pixel
space, which are the games with hundreds of components — the two failure modes
arrive together. Six segmentations multiply this further. The dynamics experiment
wants a few hundred transitions across 25 games *including intermediate animation
frames*, which is tens of thousands of frames. This should be sized before that
run, not during it.

---

## 4. How the measurements are made, and where they flatter us

Ground truth comes from the **oracle** (D-010): we reproduce the engine's
renderer with sprite indices in place of colours, giving a pixel-exact
instance-segmentation map for every frame. It reproduces the engine's own render
byte-exactly on all 25 games. It is **test and evaluation only** — nothing on the
agent's code path may import it, and `tests/test_oracle_boundary.py` enforces
that mechanically, because sprite identities will not exist on the private set.

Reproduce with:

```bash
python scripts/eval_geometry.py --actions 20 --seed 0
python scripts/eval_tracking.py --actions 60 --seed 0 --segmentation colour_8
```

Four things systematically bias the numbers toward optimism. They should travel
with any quotation of them.

1. **Ground truth exists only for resting frames.** The oracle reads the engine's
   *current* state, so only the last frame after each action can be scored. The
   intermediate frames of an animation — where tracking is hardest — are
   unscored. Fixing this needs the oracle sampled inside `perform_action`, via a
   renderer callback or an engine patch.
2. **The tracker is not merely unscored on animation frames — it is not even
   run on them.** `eval_tracking.py` feeds it one resting frame per action. In
   use it will be fed whole 42-frame animations. That is a different matching
   regime and it has no number attached.
3. **Everything is measured under random play.** Both eval scripts drive with
   uniformly random legal actions and random click coordinates. The resulting
   transitions are small and low-diversity — 1–58 changed pixels of 4,096,
   heavily dominated by recolouring, with few multi-object motions. Busy
   multi-mover scenes are simply not in this data. Geometry is especially
   exposed, since it induces from the mask of changing pixels.

   *A correction worth keeping:* an earlier draft of this section called the
   distribution "mostly no-ops", generalising from sb26. Measured across all five
   fixtures the no-op rate is **14%** — 68% on sb26 and **0% on the other four**.
   The defect in random play is that the changes it produces are trivial, not
   that it produces none.
4. **Soundness is verified in-sample.** The soundness test induces geometry from
   a set of frames and then verifies cell uniformity on the *same* frames. The
   check cannot fail by construction. This is how dc22 can be both "sound" and
   wrong. The informative test — induce on `frames[:n]`, verify on `frames[n:]` —
   does not exist.

Two smaller ones: the tracking metric assigns each ground-truth sprite the track
of whichever component covers most of its pixels, so with fragmentation at 2.55
some identity churn is averaged away before it is counted; and games where random
play reaches a terminal state during warm-up are dropped from the scored set
entirely.

---

## 5. What is deliberately not being built

Distinct from §3's *limitations*, these are **choices**. Each is a live option we
have declined, with the condition that would reopen it.

- **No VLM at the primitive layer** (D-006). We do not render frames to images and
  ask a model what it sees. The data is already symbolic; rendering it discards
  exactness for nothing. *Reopen if* a symbolic-only pipeline demonstrably cannot
  recover structure a model reads off trivially. Note the parked hybrid option in
  `state-of-play.md`: Tufa fed image + ASCII + segmentation together and won
  Milestone #1 — worth revisiting once there is an LLM-facing layer to feed.
- **No hand-crafted hypothesis space** (Guardrail 2). The Milestone #1 winner
  found hand-crafted tools actively *hurt*. The six segmentations are general
  representational rules, not a taxonomy of game types. If we ever find ourselves
  enumerating "the kinds of games there are", that is the signal to stop.
- **No per-region geometry.** The honest model of a composited frame is several
  cell sizes at once. Deferred as not yet worth the complexity (D-013), at the
  known cost quantified in §3.1.
- **No scipy, no heavyweight dependencies.** Everything must vendor into an
  offline Kaggle notebook. This is why the assignment algorithm and the
  connected-component labeller are written out rather than imported.
- **No agent, no goal, no planner.** Dynamics landed in session 3 (§7). Two of
  the four subsystems still do not exist, and neither does a loop that plays a
  game. Session 1's scope was perception only (D-007), deliberately, so that the
  expensive machinery is not built on a perception layer that turns out not to
  work.
- **No population of dynamics hypotheses**, only the single MDL-best rule set.
  This knowingly violates Guardrail 1 and D-003, which require weighted
  hypotheses with soft down-weighting. It is a narrowing for the go/no-go
  experiment: a population cannot rescue a language that cannot express the
  dynamics, so the point hypothesis is the right instrument for asking whether
  the language works at all. *Reopen* — and it is first in the queue — now that
  D-025 has shown the language does work on some games. The MDL scores already
  computed are the natural weights.
- **No use of the intermediate animation frames in dynamics.** Up to 42
  sub-transitions per action are recorded and dynamics reads only the resting
  endpoints. A rule that predicts a whole trajectory is far more constrained
  than one that predicts an endpoint, so this is a large unexploited source of
  evidence. *Reopen* when endpoint prediction stops being the bottleneck.

---

## 6. What this layer hands downstream, and what it withholds

**Hands over.** For each action: a short exact statement of the consequence —
these objects moved by these displacements, these recoloured, these appeared or
vanished — accounting for essentially all changed pixels, with near-zero spurious
appearances. Plus a pruned click space, a segmentation population, and full
animation sequences rather than endpoints.

**Withholds.** Any notion of meaning. The layer will tell you a 20×4 colour-9
rectangle moved right by 4. It does not know that this is the player, that moving
is good, that the object ahead is a wall, or that the action caused the motion.
There is no concept of *me*, of *goal*, of *rule*, or of *cause*. It describes; it
does not predict.

In the four-subsystem architecture, this layer is a partial **ontology**: that
box reads *what objects exist, which one is me, identity tracked across frames*,
and only the last clause is built. There is no self-identification, and — per
§3.2 — no revisable belief about what objects are.

The consequence for what comes next is concrete. Fragmentation at 2.55 means
dynamics induction will be asked to learn three rules where one would do, and the
natural repair is for dynamics to push back on segmentation. **The current code
has no channel to receive that push.** Whether it needs one urgently, or whether
the redundancy is tolerable, is an empirical question the dynamics experiment
will answer — and it is a question worth holding explicitly while running it.


---

## 7. Dynamics induction

Landed in session 3. This is the subsystem D-002 stakes the project on, and §7.4
reports the experiment that was its go/no-go.

### 7.1 What a hypothesis is allowed to say

A rule is `(action, selector) → effect`, with a per-action default beneath
(D-024, `src/arc3/dynamics/rules.py`):

- **effects**: `static`, `translate(dx,dy)`, `recolour(c)`, `vanish` — four,
  closed, and chosen to be exactly what `match_frames` can distinguish, so the
  language is as expressive as the perception beneath it and no more;
- **selectors**: conjunctions of attribute *equalities*, arity 0-2, over
  `colour`, `size`, `shape`, `rect`, `clicked`. Every one but the last is a
  pre-existing field of `Component`; `clicked` — did this action's click land
  inside this object — is relational rather than descriptive and general to any
  game with a pointer (D-021).

Guardrail 2 says do not hand-craft the hypothesis space, and a rule language is
one, so the constraint adopted is that **nothing may be conditioned on that
perception does not already emit.** No feature was invented for dynamics.

### 7.2 How a rule set is chosen

By minimum description length, exhaustively over all 16 partitions of arity ≤ 2
per action (`induce.py`). No search heuristic, no local optimum, no tuned
regularisation constant — the bit costs come from the sizes of the spaces being
encoded.

MDL rather than accuracy because accuracy always prefers the finest partition:
splitting on `size` gives nearly every object its own rule and scores perfectly
in training.

**One subtlety that is easy to get wrong and expensive when you do.** Only
groups that *disagree with the default* are charged for. The first version
charged bits per group in the partition, which prices the partition rather than
the description. On ls20 that made it prefer a selector with **16 errors** over
one with **1 error in 316**, because the better selector induced 37 groups of
which 35 said "static" and needed no rule written down at all.

Here is what it induces on ls20 under `colour_8`, seed 0:

```
6 rules, 4 defaults
  action4 & shape=shape#f972 -> translate(+5,+0)
  action4 & shape=shape#5539 -> translate(+5,+0)
  action3 & shape=shape#f972 -> translate(-5,+0)
  action3 & shape=shape#5539 -> translate(-5,+0)
  action2 & shape=shape#3556 -> vanish
  action2 & shape=shape#9776 -> recolour(->11)
  action1 & any -> static   [default]
```

Read it: two shapes — a 2-part sprite — move one 5-pixel cell left on ACTION3
and right on ACTION4. That is the player and its controls, induced from 70
transitions of random play with no game knowledge.

### 7.3 Rendering a prediction

The metric demands a full 64×64 frame, so predicted object states are rendered
back to a grid (`predict.py`): erase the footprint of everything that is not
staying put, then paint, in two passes over all objects rather than one pass per
object.

**What is revealed behind a departing object has to be guessed**, and this is
the single largest correctness trap in the subsystem. The obvious answer — the
frame's background colour — is wrong whenever objects rest on a surface rather
than on the void, which is most games. On ls20 the sprite crosses floor colour 3
while the frame's background is 4. Measured with ground-truth effects supplied,
erase-to-background reproduced **0 of 100** frames with a median of 27 pixels
wrong. Erasing instead to the modal colour of the one-pixel ring around the
footprint took the median residual to **2**.

### 7.4 What it measurably does, and does not

`scripts/eval_dynamics.py`, six carvings × 25 games, seed 0, 7,888 s. Full
result and interpretation in D-025.

| carving | mean exact | exact>0 | medres≤2 | mean chgobj | deformed moves |
|---|---|---|---|---|---|
| **colour_4** | **4.8%** | **5/23** | 10/23 | 36.3% | 52.5% |
| colour_8 | 4.8% | 5/23 | 10/23 | 36.4% | 53.3% |
| foreground_4 | 1.8% | 3/23 | 10/23 | 28.8% | **14.0%** |
| foreground_8 | 1.8% | 3/23 | 10/23 | 28.7% | 18.8% |
| colour_mass | 4.6% | 4/23 | 9/23 | 34.3% | 73.9% |
| bbox | 4.2% | 5/23 | 9/23 | 37.2% | 55.6% |

Against a chance baseline measured at **0%**. Best per game across all six:
exact>0 on **7/23**, within 2 pixels on **13/23**, changed-object accuracy ≥50%
on **14/23**.

Three things to read off, in order of how much they should change your picture:

1. **The best carving is game-dependent.** `colour_4` wins on five games,
   `foreground_4` on two — and is the only carving that finds anything at all on
   ar25 and cn04. The union (7) beats any single carving (5). Keeping six
   notions of object is load-bearing, not redundancy (D-018).
2. **Lower deformation does not mean better prediction.** `foreground_*` cuts
   the deformed-move rate by three times and scores less than half. Deformation
   is a symptom of a hard game, not the mechanism of failure — an important
   negative result, because the opposite was this session's working hypothesis.
3. **The metric understates the mechanism.** sp80 scores **100%
   changed-object accuracy, 2 pixels median residual, and 0% exact match.** The
   object model is perfect and the frame comparison fails, because a move
   counter in the UI ticks two pixels every action. Object-level dynamics work
   on ~60% of games; exact frame reconstruction works on 30%.

### 7.5 How much evidence does it need? Fifty actions.

`scripts/eval_dynamics_curve.py --carving colour_4 --test-tail 30`, eight games,
300-action corpus, seed 0, fixed test tail:

| training transitions | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| exact on changed | 7.1% | 11.7% | **13.5%** | **13.5%** | **13.5%** |
| changed-object accuracy | 47.7% | 55.4% | 62.3% | 63.0% | 63.4% |

**Exact match gains +0.0000 from n=50 to n=200.** Everything happens by fifty
transitions; after that the curve stops. So the subsystem is
**language-limited, not evidence-limited**, and the practical consequence is
unexpectedly cheerful: dynamics learning costs about fifty actions of *random*
play, so actions are not the scarce input to model quality. That materially
softens D-022's central worry.

The clinching case is ls20, which at n=200 scores **100% changed-object
accuracy and 0% exact match** — every object predicted correctly on held-out
transitions, and the frame comparison failing every time on two pixels of move
counter. With sp80 showing the same shape, two independent games now have a
perfect object model and a zero score, which is why §7.4's third point is the
reason the headline is 4.8% rather than an aside.

*Read the saturation point, not the endpoints.* The first version of this
script compared 7.1% to 13.5% and printed "still climbing — evidence-limited",
the exact opposite of the truth, for a curve flat across its whole second half.

**What this does not do.**

- **Nothing about objects that appear.** Rules are keyed to before-frame objects,
  so a frame containing something new is unpredictable in principle. 53 of 464
  changed transitions in the zero-scoring games are blocked this way (11%).
- **No object-to-object interaction.** "The block stops at the wall" is
  unsayable. No collision, containment or support.
- **No state beyond the current frame.** No velocity, no counters, no memory.
- **No numeric relations.** Equality only; "bigger than 4 pixels" cannot be said.
- **No move-with-deformation and no simultaneous move-and-recolour.** Both are
  silently recorded as `static`. The first accounts for 14-74% of observed moves
  depending on carving.
- **No bookkeeping layer.** The induced mask (`bookkeeping_mask`) is a scoring
  *diagnostic*, deliberately conservative — horizontal bands only, at most 4
  rows — so that it under-reports UI rather than excusing the model. It is not a
  model of the counter and nothing predicts one. D-025 names this as the next
  subsystem.
- **It is a point hypothesis, not a population.** See §5.

### 7.6 A note on positive controls

The first run of the experiment reported 0% on every game under every
condition — a clean-looking refutation of D-002. Four separate bugs, all in the
scoring and search rather than the hypothesis, each sufficient on its own:
erase-to-background rendering; an unmodelled move counter; an MDL cost that
priced partitions instead of descriptions; and an "object accuracy" of 91.6%
that was really measuring how often objects sit still (16.4% on objects that
actually changed).

What broke it open was a **positive control**: supply ground-truth effects and
ask whether the renderer can reproduce the frame at all. It could not, on 0 of
100, which localised the fault immediately. No amount of inspecting the
induction would have found it. Every one of the four is now a named regression
test in `tests/test_dynamics.py` recording the false conclusion it would
otherwise produce.


---

## 8. Exploration

The first component here that acts rather than measures. It exists because of a
decomposition, not a hunch: D-028 split the sixteen zero-scoring games and found
that **half of them are not a modelling failure at all** — the agent never makes
the game do anything, so there is nothing in the corpus to learn from. D-030 then
established the moves are there to be found, with vc33 responding to 2 of 256
probes at the level start and **254 of 256** five actions later.

### 8.1 What it optimises

`ExploringDriver` (`src/arc3/explore.py`) does Thompson sampling over a Beta
posterior on one question: *does this choice move the playfield?*

Three properties, each forced by a measurement rather than chosen:

**It counts playfield changes, not frame changes.** vc33 changed its frame on 99
of 99 actions and its playfield on **zero** — every change was a move counter
ticking. A policy rewarded for frame change would have been perfectly satisfied
by a game that never responds. `OnlineBand` estimates the bookkeeping band
incrementally, the runtime counterpart of §7's `bookkeeping_mask`, and stays
empty until warmup so it is under-confident early rather than wrong early.

**Arms are object signatures, not pixels.** Objects move, so a pixel that paid
off last turn means nothing this turn. An arm is the attribute vector §7.1's
rule language already selects on, so exploration experience transfers as objects
move and the explorer and the learner look at the same features.

**Evidence decays.** Responsiveness is state-dependent. A stationary estimate
would learn "clicking does nothing" at the level start and never revise it.
`_Arm.decay` pulls towards the prior rather than past it, so Guardrail 1's "never
hard reject" survives arbitrarily long runs of failure — an arm that has failed
twenty times is heavily down-weighted and still occasionally retried, which is
exactly what a state-gated game needs.

### 8.2 What it measurably does

`scripts/eval_exploration.py`, 25 games, 100 actions, seed 0:

| | `random` | `random_objects` | **`explore`** |
|---|---|---|---|
| playfield activity, all games | 49% | 55% | **63%** |
| playfield activity, the starved eight | **3%** | 14% | **47%** |
| **games the agent cannot move (<15%)** | **8** | 6 | **1** |
| levels cleared | 1 | 1 | **4** |

The corpus it leaves behind is correspondingly richer — on the starved games,
3% activity and 2.5 distinct playfield events per game become **46% and 30.9**.
lp85 goes from 1 distinct event to 76; vc33 from 0 to 88.

Of the four levels, **r11l is cleared by all three drivers** and is not the
explorer's doing; ft09, lp85 and tn36 are cleared only by `explore`.

### 8.3 The arm-space constraint

The single most important tuning fact, and both first guesses were wrong. The
symptom was that the explorer was *less active than uniform sampling* on
object-rich games — cd82 at 20% against `random_objects`' 79%.

| | activity | distinct events |
|---|---|---|
| `random_objects` | 38% | 27.4 |
| union of all six carvings as targets | 45% | 25.2 |
| `(colour, size)` arms | 58% | 35.8 |
| **`colour` arms** | **58%** | **44.9** |

Drawing targets from all six carvings — the obvious fix, since D-021 showed the
union is what reaches every distinct click outcome — made it **worse**, because
it doubled the arm count and Thompson sampling then spends its whole budget
establishing posteriors it never exploits. The binding constraint is **arms
against action budget**: 83-136 arms against 100 actions. Pooling to colour
gives 6-14.

**What this does not do.**

- **No goal.** It maximises "something happened in the playfield", which is a
  proxy for interestingness and explicitly not a proxy for winning. The four
  cleared levels are a side effect of poking a responsive game.
- **No model.** It knows whether an action tends to do something, never what.
- **It is worse than random where random already works** — cd82 39%→28%, dc22
  59%→34%, sk48 69%→52%. A policy spending budget on posterior estimation loses
  to uniform sampling where uniform sampling suffices.
- **No RESET policy**, though some games become *less* responsive with depth.
- **No credit assignment across steps**, which is precisely what vc33 rewards:
  the action that unlocks responsiveness gets none of the credit.
- **Three unfitted constants** (decay, warmup, band threshold). They are set to
  the values `bookkeeping_mask` already used, which makes them consistent rather
  than tuned — deliberately, after D-029 caught itself selecting a rule on test
  data.

## 9. The agent loop

`GraphAgent` (`src/arc3/agent.py`, D-034) is the first thing in the repository
that plays a game end to end and gets a competition score. Everything above
this section is either perception or a measurement instrument; this is the
loop that consumes them.

### 9.1 What it does

The engine is byte-deterministic (§benchmark, D-009), so every transition the
agent observes is a permanent, exact fact. The agent keeps those facts as a
graph: a node is a resting frame with the bookkeeping band blanked
(`state_key`), an edge is an action observed to lead from one node to another.
On visited states that graph is a perfect dynamics model, and planning against
it is shortest-path search.

Each step:

1. Record the edge just traversed, and update the explorer's arms (§8) with
   whether the playfield moved.
2. If the game is over, press `RESET` (it restarts the current level only).
3. If a planned route is in progress and still consistent, take its next step.
4. Otherwise rank the untried actions at this node by a Thompson sample from
   their arm (`("act", id)` or `("click", colour)`), and take the best, unless
   every untried action here belongs to an arm whose mean is below
   `local_min` (0.3).
5. Otherwise breadth-first search the graph, from here and from the level start
   via `RESET`, for the node with untried actions that minimises
   `path length + 4 × (1 − best arm mean)`, and walk there.

Click candidates are the centre pixels of the 64 largest non-background
`colour_8` components. Nodes are re-keyed whenever the band estimate changes,
because a key computed under one band is meaningless under another.

A worked example, sp80 (`scripts/eval_agent.py --games sp80 --agents
graph_nomodel`, seed 0): 196 distinct states in 500 actions; level 1 cleared at
action 45 against a human baseline of 39 (score 0.75), level 2 after 39 more
actions against a baseline of 58 (score capped at 1.15). Of 500 actions, 495
were exploratory and 420 moved the playfield.

Four levels are cleared at or under the human baseline: r11l (20 vs 22), tn36
(19 vs 32), sp80 level 2 (39 vs 58), vc33 level 2 (18 vs 18). Blind coverage
can do this only where the win is close to the start, and those four levels
are most of the score.

### 9.2 What it measurably does

`scripts/eval_agent.py`, 25 games, cold start, seed 0, offline RHAE from
`client/budget.py` (score capped at 1.15 per level, mean over all levels):

| | levels | games with a level | mean RHAE |
|---|---|---|---|
| `explore` driver + RESET on game over, 500 actions | 6/183 | 5/25 | 2.42% |
| **`GraphAgent`, no model, 500 actions** | **10/183** | **8/25** | **3.46%** |
| `GraphAgent`, with the induced model, 500 actions | 10/183 | 8/25 | 1.18% |
| `GraphAgent`, no model, 2000 actions | 14/183 | 11/25 | 3.47% |

**Seed 0 is the lucky seed.** On seeds 1 and 2 (`agent_v1_seed{1,2}.*`) the
graph agent scores 1.68% and 2.11%, the explore driver 1.76% and 2.03%: means
of **2.42% against 2.07%, a difference inside the noise**. sp80 scores 31.7% on
seed 0 and 0.2% on the others; vc33 15.4% against 0.4%. Only r11l and tn36
score on every seed, and the explore driver clears both too. Read every
single-seed number in this section with that in mind.

On seed 0, almost all the score comes from four games cleared quickly: sp80
(31.7%), r11l (19.2%), tn36 (16.4%), vc33 (15.4%). The four extra levels bought by
going from 500 to 2000 actions all arrive after action 230 and are worth almost
nothing under the quadratic penalty.

### 9.3 The induced model makes it worse

With `use_model=True` the agent re-induces a rule set every ten transitions
(§7, `colour_4`, last 150 transitions) and predicts the outcome of the top
eight arm-ranked untried actions; an action predicted to change nothing is
penalised 0.6, one predicted to reach a known state 0.35. Mean RHAE falls from
3.46% to 1.18%: sp80's first level goes from 45 actions to 367, vc33's second
from 43 to 218. It does open games the model-free agent does not (bp35 level 1,
tu93 two levels, cd82 at 248 rather than 470), but each late.

The likely mechanism, not yet isolated: a rule set that is exactly right on a
minority of transitions (§7.4) is confidently wrong on the rest, and a wrong
"this does nothing" prediction pushes the one productive action to the back of
the queue. That is a one-step model used as a filter, which is the weakest way
to use a model; it is not a test of planning with one.

### 9.4 What this does not do

- **No goal.** It explores until it stumbles on the win. Every level it clears
  is cleared by coverage, and the score is dominated by games where the win is
  a few actions from the start.
- **No use of the dynamics model by default.** See §9.3.
- **No transfer across levels** beyond arm statistics; the graph is discarded
  on each level change.
- **No hidden state.** Same masked frame plus same action is assumed to give
  the same result; a contradicting observation overwrites the edge.
- **The bookkeeping band is unstable on some games.** tr87 re-keys its graph on
  489 of 500 actions because a playfield region changing on about half of all
  actions is indistinguishable, by row frequency, from a counter. Restricting
  the band to edge rows fixed the re-keying and changed no score on ten games
  (D-034), so it is not what limits those games.
- **Bounded click candidates.** 64 per state; a game whose productive click is
  on a small object in a cluttered frame may never offer it.

## 10. The hypothesis layer (v2)

v1 (§9) acts, remembers and navigates, and has no idea where to go. v2
(`src/arc3/agent_v2.py`, D-035) adds the missing part: an LLM proposes **goal
hypotheses as code**, a population holds them with weights that evidence
moves, and the population's opinion of each known state steers the search. The
LLM never chooses an action.

### 10.1 The pieces

- **`arc3.llm`** — one `complete(messages)` interface. Backends: `ScriptedLLM`
  (tests), `OpenAIChatLLM` (any OpenAI-compatible server; refuses non-local
  hosts, Guardrail 5), `RecordingLLM` / `ReplayLLM` (every call to JSONL, and
  exact offline replay keyed on the full request).
- **`arc3.scene`** — a frame as objects (`colour_8` components: colour, size,
  box, centre) plus the non-background crop as hex rows. The same object is
  rendered into the prompt and passed to the hypothesis code, so the model
  writes code against exactly what it was shown.
- **`arc3.propose`** — the prompt: the scene API, what each action has been
  observed to do (`describe_effect`), what won the previous level, and how the
  existing hypotheses have fared. It describes the interface only, never kinds
  of game (Guardrail 2).
- **`arc3.hypothesis`** — parsing (tolerant of small-model formatting), a
  sandbox (whitelisted builtins, no imports or dunders, a step budget counting
  only the hypothesis's own lines), and the `Population`:

  | evidence | effect on weight |
  |---|---|
  | level won from state *s* | × (0.5 + 1.5·progress(*s*)) — ×2 if it said 1.0, ×0.5 if 0 |
  | a state it calls reached (≥ 0.99) that did not end the level | × 0.7, once per state |
  | an exception or bad return | × 0.8, at most five times |
  | identical output on ≥ 20 states | marked *flat*: excluded from ranking, kept |

  Weights floor at 0.01; nothing is deleted (Guardrail 1).
- **`HypothesisAgent`** — fills four inert hooks added to `GraphAgent`: a state
  value (the population's weighted progress), a best-first "leave for a better
  state" test (`goal_weight` = 100 value-units per step of walking), and
  on-state / on-win callbacks that feed the population. It calls the LLM 10
  actions into a level, then every 150 actions without a win, at most 4 times
  per level and 16 per game. Hypotheses that gained weight on a win carry to
  the next level at their earned weight (capped at 4).

With no LLM, or before the first proposal, every value is 0 and v2 *is* v1
with the band confined to edge rows (tested: `test_no_llm_is_exactly_v1`).

**One hypothesis at a time (D-038).** The value that steers is not the
population's average but the *focused* hypothesis's own progress. The focus is
drawn by weight; after 40 states without improving its best value it has
stalled, pays ×0.8, and another is drawn. Averaging was the first design and
failed outright when a right and a wrong hypothesis arrived together: they
cancelled to a flat landscape (synthetic room, 0 levels in 300 actions, against
2 levels with focus).

**Diagnosis.** `scripts/diagnose_v2.py RESULTS_DIR` splits a run into five
links — format, informative, steering, correct, net against v1 — and names the
first weak one. It reads the per-level population snapshots
(`HypothesisAgent.level_report`) that every results row now carries.

### 10.2 What it does with a right hypothesis — the synthetic room

`tests/test_agent_v2.py`: a 12×12 room of 4-pixel cells, ACTION1–4 move a
block, the level is won when it reaches a square in the far corner; two levels.

| agent | level 1 at | both levels |
|---|---|---|
| v1, blind (edge-row band) | not within 600 | not within 600 |
| v2, correct hypothesis | 17 | **61** |
| v2, correct then three wrong | 17 | 61 |
| v2, wrong hypotheses only | not within 600 | not within 600 |

Three things this test found, each now fixed and commented where it lives: the
unconfined band masked the rows the block moves in as a "counter", hiding it
from the hypotheses; a fixed value margin (0.15) for leaving a state let the
agent drift away from a goal one move off, which is what forced the
best-first value-against-distance trade; and a carried hypothesis entering at
the prior weight tied with a fresh wrong guess and stalled level 2.

### 10.3 What it does with a real small model — vc33

`scripts/eval_agent.py --agents v2 --games vc33 --llm-url …`, seed 0, Qwen2.5-
Coder-3B-Instruct Q4_K_M on llama.cpp, CPU (Snapdragon X Elite). Five calls at
36–80 s each; 9 hypotheses, 4 still informative at the end. Score **identical
to v1** (levels at actions 25 and 43, 15.4%). The hypotheses, verbatim goal
lines, with their final state:

```
H0 w=1.00 : maximize the number of cells of the largest colour.
H7 w=1.00 : maximize the number of cells of the most frequent colour.
H8 w=1.00 : maximize the number of cells of the largest connected region.
H1 w=0.01 [flat] false=151 : minimize the number of cells of the smallest colour.
H4 w=0.01 [flat] false=111 : minimize the number of cells of the smallest colour.
```

The evidence rules worked: "minimise the smallest colour" returns ≥ 0.99 on
almost every state, so it collected 151 false goals and sank to the floor.
What did not work is the model: every hypothesis is a colour-count statistic,
none refers to an object's position or to what the actions do, and several
blocks used `from collections import Counter`, which the sandbox rejects.

### 10.4 What this does not do

- **Goal hypotheses only.** The LLM is shown what actions did but never asked
  to predict effects; the state graph is still the only dynamics model.
- **Values exist only for visited states.** An untried action's outcome is
  unknown, so within a state the choice among untried actions is still v1's.
  The search is best-first over the known graph, not planning through
  predictions.
- **No credit for partial progress.** Only wins and false goals move weights.
- **Tested with a 3B model on a CPU only.** The Kaggle vLLM cell has not run;
  whether a 27B model's hypotheses are good enough to steer is the open
  question (D-035's revisit condition).
- **Runtime grows.** v2 builds a scene per visited state; per-step cost on the
  heaviest games has not been measured yet.

## 11. v3: the LLM reasons in the loop (D-039)

v2 (§10) let the LLM guess goals and decide nothing; on the first Kaggle run
it was identical to v1. v3 inverts the roles. The LLM holds the hypotheses —
what the goal is, which object is the player, what each action does, which
objects are static, counters or tools — and writes them as a **world model in
code**. Exact code is its senses (the fact layer), its referee (the verifier)
and its hands (the planner and the explorer). Code: `src/arc3/v3/`.

### 11.1 The loop

Per step (`v3/agent.py`, `V3Agent.act`):

1. **Observe.** The fact layer ingests the resting frame.
2. **Check.** If the last action was taken under a prediction (a planned or
   experimental step), the predicted objects are compared with the observed
   ones. A mismatch is a *surprise*: the rest of the queue is dropped and the
   exact counterexample goes into the next prompt.
3. **Act** on the queue if one exists; otherwise **decide**:
   - *carried plan* — at a new level, the previous level's model is tried
     first (planned with, every step still checked);
   - *probe* — each simple action once, and clicks on a few differently
     coloured objects (Guardrail 4: dynamics first);
   - *assess* (LLM) — on a trigger (start, surprise, game over, goal refuted,
     no plan) or every 30 actions; then the returned `predict` is verified on
     the last 80 recorded steps, and *repair* (LLM) is called with the
     counterexamples until it explains them or two repairs are spent;
   - then the model's *next*: **plan** (search its model for the shortest path
     to its goal; only if the model verifies at ≥ 90%), **experiment** (its
     test actions, each with a predicted outcome), or **explore** (12 steps
     of v1's graph explorer, used as a tool via `GraphAgent.took`).
4. A plan that runs exactly as predicted into a state its goal scores 1.0,
   without the level ending, **refutes that goal**; refuted goals are listed
   in every later prompt.

### 11.2 The fact layer, on real data

`v3/facts.py`. Tracked `colour_8` objects (background removed) with stable ids;
typed changes per step; same-colour and same-shape groups; static objects;
co-moving pairs; per-action effect tallies (clicks keyed by clicked colour);
controllable-looking objects. After 12 scripted moves on ls20
(`facts_demo.py`-style run, seed 0), an excerpt of what the model is shown:

```
MOVE TOGETHER: #7 and #8 (11 times)
CONTROLLED-LOOKING objects (moved by several actions):
  #7: ACTION1->(+0,-5), ACTION2->(+0,+5), ACTION3->(-5,+0), ACTION4->(+5,+0) (consistency 1.0)
  #8: ACTION1->(+0,-5), ACTION2->(+0,+5), ACTION3->(-5,+0), ACTION4->(+5,+0) (consistency 1.0)
  #13: ACTION1->(+1,+0), ACTION2->(+1,+0), ACTION3->(+1,+0), ACTION4->(+1,+0) (consistency 1.0)
LAST STEPS:
  t11 ACTION1: #1 reshaped; #7 moved (+0,-5); #8 moved (+0,-5); #13 moved (+1,+0), shrank 62->60; #17 grew 22->24
```

The two-part player (#7 colour 12 on #8 colour 9) is isolated without being
named; #13 is a bar losing two cells per action — a budget, if the model so
judges. Counters are deliberately *not* hidden in v3: whether they matter is a
hypothesis.

### 11.3 World model, verifier, planner

`v3/world.py`. `State` holds the frame's objects as dicts plus the frame for
collision queries (`colours_in_box`, `free`, which discount the moving
objects' own original cells). `WorldModel.compile` runs the LLM's `predict`
and `goal` in the §10 sandbox. `verify` replays `predict` on recorded steps and
returns exact mismatches ("#7 predicted at x20 y11 -> actually x20 y16").
`plan` is breadth-first to the nearest goal state (optimal in actions), then
best-first on the goal's value when the space is too large.

On real ls20 data, a hand-written model of the kind the LLM should produce
(two-part player, 5-pixel moves blocked unless the destination is maze floor,
counters ignored) predicts **18/18** recorded steps exactly, including the
wall-blocked ones (`world_demo.py`, seed 0).

### 11.4 What it does end to end — the synthetic walled room

`tests/test_v3_agent.py`: 8×8 cells, a wall with one gap, goal in the far
corner, two identical levels; a scripted model whose first `predict` ignores
walls and whose second is right.

| agent | level 1 at | both levels | LLM calls |
|---|---|---|---|
| v1, blind | 241 | 364 | 0 |
| **v3** | **24** | **41** | **2** |

The trace: 4 probe actions; first assessment (its wall-less model explains
the probe steps 100%); the plan hits the wall — surprise, counterexample;
second assessment adds walls; the optimal 17-move route wins; level 2 is
planned with the carried model, 17 moves, no call.

### 11.5 What this does not do

- **Tested end to end only with a scripted model** and one short real-model
  smoke run (§11.6). Whether a 27B model writes world models good enough to
  plan with is the open question; the Kaggle benchmark answers it.
- **One live world model.** Rivals exist only as stated hypotheses and the
  refuted-goal list; there is no population of competing `predict`s.
- **Object-level checking** (boxes, colours, existence), not cell-exact.
- **Clicks in plans target object centres only.**
- **Cost.** The fact layer segments and tracks every frame (~0.1–0.2 s per step
  on this laptop, `eval_agent.py --agents v3`, tn36/sp80/ls20); with ~110 games
  in 9 h this needs optimising before large runs.
- **No learning across games.** Each game starts from nothing but the prompt.

### 11.6 First real-model run

`eval_agent.py --agents v3 --games ls20 --actions 30 --llm-url …`, seed 0,
Qwen2.5-Coder-3B-Instruct Q4_K_M on llama.cpp (CPU), prompt `v3-1`, JSON-schema
constrained output. The loop ran as designed: 4 probe actions; *assess*
(145 s) returned schema-valid JSON — hypotheses, `predict`, `goal`, a plan
request; the verifier scored its `predict` **0%**; two *repair* calls (49 s,
34 s) stayed at 0%; the untrusted model was not planned with, and the explorer
took over. 0 levels in 30 actions. The model's content was poor: goal "reduce
the number of objects to 1", and `predict` called `state.move(7, 0)` with the
wrong arity. That led to prompt `v3-2`, which adds a neutral syntax example of
the helper API (it demonstrates calls, not a kind of game — Guardrail 2).
Whether a 27B model writes plannable world models is still unmeasured.

## 12. v4: an assumption store, checked predictions, and subgoals walked by code (D-042, D-044)

v3 (§11) handed control to the explorer until the model's simulator was
perfect, which on real games was never. v4 (`src/arc3/v4/`) asks the model for
less and checks more: it keeps an **assumption store** the model edits (what
things are, what actions do, what wins), it must **predict** the outcome of
every action it chooses, and code checks each prediction exactly. Since D-044
the model can also name **subgoals** ("bring #7 onto #5, then onto #3") that
code routes to and walks without further calls.

### 12.1 The loop

Per step (`v4/agent.py`, `V4Agent.act`):

1. **Observe** with `FactBook4` (§12.2).
2. **Check.** Assumptions with a machine-checkable rule for the action are
   checked (`store.observe`); the expectation attached to the action is
   checked (`expect.check`). A correct prediction supports the **ontology and
   dynamics** assumptions it named, once per decision. A wrong one — including
   a change the model did not mention under `rest: unchanged`, except for
   tickers — is a surprise: the queue and the plan are dropped and an update
   is due. Goals are tested by code: a failed `level_complete` prediction
   weakens the goals it named, and a goal whose `attain` condition holds
   without the level completing is weakened once per arrival.
3. **Act** on the queue; if it is empty and subgoals remain, **route** to the
   next one (§12.3) and queue the route, each step with the expectation "the
   object moves by (dx, dy)", the last with the subgoal's arrival
   expectation.
4. Otherwise **ask the model**: *update* (edit the store) after any surprise,
   no-op, novelty, return-to-start, level start or game over, and every 10
   actions unless the last 5 predictions all held; then *act* (up to 6 steps
   and/or up to 4 subgoals).

Without a usable model the step goes to v1's explorer, and every such step is
counted: a run with `explore_steps > 0` is reported as not clean.

### 12.2 The fact layer v4 adds

`v4/facts.py`, a subclass of §11.2's `FactBook` (v3's prompts are untouched,
so v3's recorded calls still replay). Each addition answers a failure measured
on ls20 on 2026-10-19 (D-044). After 14 scripted moves on real ls20
(`scripts/probe_v4_facts.py ls20 1,1,3,3,4,4,2,2,4,4,4,4,4,4`, seed 0), the new
lines are:

```
CHANGE ON EVERY ACTION, WHATEVER THE ACTION (excluded from no-op checks): #13 (last 14 steps)
SAME SHAPE UP TO SCALE/ROTATION/REFLECTION: #3 ~ #10 (x2 smaller, rotated 270)
NEXT TO the most controlled-looking object (group #7 #8):
  ACTION1 -> (+0,-5) would enter: colour 3 x25
  ACTION2 -> (+0,+5) would enter: colour 4 x25
  ACTION4 -> (+5,+0) would enter: colour 4 x25
MOVES THAT HAPPENED entered: colour 3 x250. MOVES THAT DID NOT HAPPEN had in the way: colour 4 x250.
(20 shape changes of still objects were only a mover covering or uncovering them, and are not reported as changes)
```

and the per-action tallies no longer list the bar: `ACTION4: tried 8, no
effect 5; colour-12 moved (+5,+0) x3, colour-9 moved (+5,+0) x3`. Compared with
§11.2's excerpt: the floor's `#1 reshaped` artefact is gone; the bar #13 is no
longer "controlled-looking" (an object that moves the same way under every
action is excluded); the goal box #3 is linked to the key display #10; and the
wall colour (4) is identified from evidence. The other additions are per-step
notes: an object back at its level-start position, objects restored to their
level-start state (on ls20, a lost life), a colour or shape never seen before
on the level.

### 12.3 Subgoals and routes

`v4/route.py`. A subgoal is `{object, to, relation, on_arrival, goal}` with
`relation` overlap, touch or at. `route` is a breadth-first search over the
positions of the object's co-moving group, using each action's dominant
measured displacement, refusing colours that blocked a move and were never
entered (`blocked − entered`), and refusing the exact (position, action)
pairs where a move was refused. The last was added after the real-ls20 check
below, where a move was refused with floor-coloured cells ahead and the
router kept proposing it.

Mechanics check on real ls20 (`scripts/oracle_v4_ls20.py "#5" "#3" 60`, seed 0): a
scripted "model" tries the four moves and then gives the plan "overlap #5
(expect nothing), then overlap #3 (expect level_complete)". The walker reached
#5 at t14 (the key display rotated), routed to #3 and **cleared level 1 after
22 actions — the human baseline is 22** — with 7 act calls; 18 of 19
predictions held on steps where more than the ticker changed. This measures
the walker, not a model: the subgoals were supplied.

### 12.4 What this does not do

- **Routes only translate one object.** Pushing, rotating, teleporting, moving
  obstacles and clicks are the model's to do in explicit steps.
- **Movement knowledge is per level.** The fact layer restarts at each level,
  so the moves must be re-observed (a few actions) before code can route.
- **Goal attainment is one condition.** A goal that needs something done first
  is weakened when its final condition holds without it; the model must
  restate it with the precondition.
- **Tickers are found per id over a 4-step window;** the tracker can swap ids
  between a bar's spent and live parts after a refill, and the swapped id is
  re-detected rather than corrected.
- **Thinking is not capped.** A truncated answer is retried once with twice
  the room and counted (`stats['truncated']`). vLLM's soft
  `thinking_token_budget` can be passed with `--llm-extra`; it is untested.
