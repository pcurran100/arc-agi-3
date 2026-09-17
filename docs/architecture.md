# Architecture: what is built, how it works, and what it does not do

- **Last updated:** 2026-09-08, session 2
- **Scope:** the perception layer, which as of this writing is the only
  subsystem that exists.
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
- **No agent, no dynamics, no goal, no planner.** Three of the four subsystems in
  the intended architecture do not exist, and neither does a loop that plays a
  game. Session 1's scope was perception only (D-007), deliberately, so that the
  expensive machinery is not built on a perception layer that turns out not to
  work.

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
