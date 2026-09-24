# Decision log

Append-only. Newest at the bottom. Never edit a past entry to look better in
hindsight — supersede it instead. See `CLAUDE.md` for the template and rules.

D-001 through D-007 are **backfilled** from the design conversation of
30 August 2026, before any code existed. They are recorded here so the reasoning
survives, but they carry the weakness of all backfilled decisions: they are
reconstructions, and none has yet met contact with an actual environment.

---

## D-001 — Enter the ARC-AGI-3 track, not ARC-AGI-2

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** ARC Prize 2026 runs both tracks. ARC-AGI-2 has a $150K unclaimed
  bonus at 85%; ARC-AGI-3 has a $700K grand prize at 100% plus $75K top-score
  awards. Question was which is winnable by a small team without large compute.
- **Decision:** Target ARC-AGI-3.
- **Options rejected:**
  - *ARC-AGI-2 track* — every 2025 podium finisher trained models with heavy
    synthetic data (NVARC needed hundreds of thousands of examples for 24%).
    Test-time adaptation appears in every strong v2 system. The binding
    constraint there is compute, not ideas, which is the wrong constraint for us.
  - *Both tracks* — splits attention; the two reward genuinely different things.
- **Consequences:** We optimise for action efficiency, not accuracy. Compute
  requirements are modest. We give up the more mature, better-understood problem
  for one with far less prior art — which cuts both ways.
- **Revisit if:** ARC-AGI-3 turns out to be dominated by scaffold quality alone
  and the field converges quickly, or if a cheap v2 approach emerges.

## D-002 — Model-based: induce symbolic dynamics and plan internally

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** RHAE counts only environment actions and explicitly excludes
  internal reasoning; scoring penalises extra actions quadratically. The entire
  Milestone #1 podium consists of LLM harnesses with no queryable environment
  model, scoring ~30%.
- **Decision:** Build an agent that induces a symbolic model of environment
  dynamics from observed transitions and plans against it internally. Treat
  internal compute as free and spend it lavishly.
- **Options rejected:**
  - *Better LLM harness only* — is what the podium already does; no queryable
    model, cannot plan, re-derives mechanics every level. Held loosely: Poetiq's
    54% on ARC-AGI-2 shows scaffold headroom can exceed expectations.
  - *Deep RL / MuZero-class* — needs millions of frames; RHAE budget is hundreds
    of actions with a squared penalty. The metric specifically disqualifies the
    least sample-efficient family.
  - *Specialised net trained from scratch, no LLM* — TRM and CompressARC prove
    this works for static ARC, but v3's hard part is goal inference in an
    undescribed environment, which needs a world prior a grid-trained net cannot
    have.
  - *Meta-RL over procedurally generated games* — attractive and probably right
    in the long run, but GPU-heavy, months of work, and carries severe
    distribution-mismatch risk if our generator's games are unlike ARC's. Wrong
    shape for a part-time build. **Not dead — parked.**
- **Consequences:** Commits us to symbolic representation and to perception being
  exact. Makes neural approaches awkward to bolt on later.
- **Revisit if:** the dynamics-induction validation test fails (see "What would
  falsify this thesis" at the end of `design-thesis.md`), or a plain LLM harness
  reaches 60%+.

## D-003 — Maintain a hypothesis population, not a single hypothesis

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** The natural loop design — assume, act, reject on contradiction,
  re-assume — is single-hypothesis hill-climbing. The living survey lists
  "multiple hypotheses with consistency filtering" as one of six properties
  shared by every system above 70% on ARC-AGI-1.
- **Decision:** Carry a weighted population of hypotheses, pruned by consistency
  against accumulated observations. Down-weight on disconfirmation; do not
  delete.
- **Options rejected:**
  - *Single-hypothesis chain* — known-dead in this benchmark family; also the
    observed failure mode of current LLM harnesses (commit, get confused,
    re-commit).
  - *Hard rejection on disconfirmation* — presumes determinism and full
    observability. Neither holds: hidden state (inventory flags, counters,
    phases) means a correct-but-conditional hypothesis will produce a
    disconfirming observation and be wrongly binned.
- **Consequences:** Needs a principled weighting scheme and a pruning policy.
  More bookkeeping; harder to debug than a single chain.
- **Revisit if:** population maintenance proves to cost more than it earns on
  the public demo environments.

## D-004 — Separate ontology, dynamics and goal into three inference problems

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** These have very different evidence economics. Dynamics is densely
  testable — every action yields one labelled example. Goal is sparsely testable
  — evidence only on win, death, or score change. Ontology is passively
  observable.
- **Decision:** Three separate subsystems with separate representations and
  separate update rules. Early actions are budgeted to dynamics disambiguation,
  not goal testing.
- **Options rejected:**
  - *One unified "assumptions" store* — a single update rule would be wrong for
    at least two of the three, and would cause expensive actions to be spent
    probing the goal when goal evidence arrives for free later.
- **Consequences:** Defines the module boundaries of the whole codebase. Each
  must stay independently replaceable.
- **Revisit if:** the three turn out to be so entangled in practice that
  separate update rules cannot be written.

## D-005 — Prune the click action space by object segmentation

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** Five control keys give branching factor ~5, so depth-10 search is
  ~10M nodes — trivially affordable given free internal compute. But
  cell-selection is parameterised over 64×64 = 4,096 coordinates, which is what
  actually makes search intractable.
- **Decision:** Propose clicks only on distinct segmented object instances, not
  arbitrary cells. Expected reduction from 4,096 to roughly 5–20 candidates.
- **Options rejected:**
  - *Learned click heuristic* — what Reki did (numpy-based heuristic with
    fallback); opaque, and doesn't address the underlying representation problem.
  - *Uniform sampling over coordinates* — wastes actions at quadratic cost.
  - *Deeper search with better pruning* — treats a representation problem as a
    search problem. The fix belongs upstream.
- **Consequences:** Search tractability now depends directly on segmentation
  quality. Reinforces D-007. Risk: if a game requires clicking empty cells, this
  prunes away the winning action — must be detectable and escapable.
- **Revisit if:** any public demo environment requires clicking a cell that is
  not part of a segmented object.

## D-006 — Perception primitives are exact and symbolic, never VLM vision

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** Observations are 64×64 grids of 16 discrete colours. Two of the
  three Milestone #1 entries render frames as images for a VLM.
- **Decision:** All perception primitives use exact algorithms — connected
  components, symmetry detection, periodicity, colour histograms, exact object
  matching across frames. No rendering to PNG for a vision model at the primitive
  layer.
- **Options rejected:**
  - *VLM on rendered frames* — trades exactness for nothing on discrete symbolic
    data; produces agents that hallucinate walls.
  - *Hybrid image + symbolic (Tufa's approach)* — Tufa fed image, ASCII and
    segmentation together and won, so redundant representations plausibly help
    *the model's* reasoning. Not rejected outright for the LLM-facing layer, but
    the primitives underneath must be exact. Revisit at the point we have an
    LLM-facing layer.
- **Consequences:** Perception is deterministic and unit-testable, which is
  unusual and valuable.
- **Revisit if:** some environment property proves genuinely easier to extract
  visually than symbolically.

## D-007 — Session 1 scope: perception layer only

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** Three candidate starting points, each retiring a different risk.
  Needed to pick one before writing code.
- **Decision:** Build only the substrate first — environment client, frame
  diffing, connected-component segmentation, object identity tracked across
  frames — to a high standard. Then stop and reassess with working tools in hand.
- **Options rejected:**
  - *Thin perception → dynamics-induction validation experiment* — the fastest
    go/no-go on the core thesis (can rule induction predict held-out transitions
    better than chance?), and the recommended option at the time. Rejected in
    favour of building the substrate properly first. **The concern this leaves
    open: we will have spent time before knowing whether the thesis holds.** The
    validation experiment remains the intended next step and should be run as
    soon as perception is trustworthy.
  - *Solid perception → end-to-end weak agent* — fastest route to a leaderboard
    number and retires integration risk, but tells us nothing about whether the
    interesting part works, and risks building a full loop on a thesis that may
    not hold.
- **Consequences:** No score and no thesis verdict until this is done.
  Everything downstream (D-005's action pruning, D-002's dynamics induction,
  D-004's ontology subsystem) depends on this being exact, so subtle bugs here
  would silently corrupt every later result — which is the argument for doing it
  properly first.
- **Revisit if:** perception proves to be either trivial (finish early, move on)
  or a tarpit (more than ~2 weeks without reliable cross-frame object identity,
  at which point reassess whether the symbolic approach is viable at all).

---

## D-008 — Develop against the local `arc-agi` engine; never commit environment files

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** Question arose of whether to download competition data from Kaggle
  into the repo. ARC-AGI-3 turns out not to be a static dataset — there are no
  JSON task files as there were for ARC-AGI-1/2. The `arc-agi` PyPI package hosts
  the same game engine the Kaggle gateway runs; games download on first run and
  cache to `environment_files/`, after which everything is offline. See
  `benchmark.md` § *Tooling*.
- **Decision:** Develop against the local engine. Pin the `arc-agi` version.
  `.gitignore` both `environment_files/` and `.env`. Clone
  `arcprize/ARC-AGI-3-Agents` separately as reference rather than vendoring it.
- **Options rejected:**
  - *Commit `environment_files/` to the repo* — regenerable cache, unclear
    redistribution position on the game files, and it would bloat the repo. The
    engine fetches them deterministically anyway.
  - *Develop against the hosted API (`ONLINE_ONLY=True`)* — adds latency and a
    network dependency to every iteration for no benefit, since the local engine
    is documented as identical to the Kaggle gateway.
  - *Vendor the official harness into this repo* — we want our own architecture
    (D-004); starting inside someone else's agent loop would bias it.
- **Consequences:** Development rollouts become free and unlimited on the cached
  public games, so perception work and the validation experiment cost only
  wall-clock. **This is also a hazard:** RHAE punishes retries quadratically
  while local development permits infinite retries. Mitigation, adopted now
  rather than later — evaluate on **first attempt from a cold start**, and log
  *actions consumed* as a first-class metric from day one.
- **Revisit if:** the local engine is found to diverge from the Kaggle gateway in
  any observable way, or competition rules change what may be redistributed.

---

## D-009 — Corrections to `benchmark.md` after first contact with the engine

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** `benchmark.md` was compiled from published prose on 2026-08-30
  before any code existed. Session 1 opened by installing `arc-agi==0.9.9`,
  downloading all 25 public demo environments, and measuring the API surface
  directly. Several documented facts were wrong, and one of them is load-bearing
  for `design-thesis.md` §8.
- **Decision:** Correct `benchmark.md` in place, marking every directly measured
  fact *[measured]*, and record the corrections here. Where a correction
  undermines an argument elsewhere in the docs, annotate that argument rather
  than quietly rewriting it.

  The corrections, in descending order of how much they cost us:

  1. **There is no Undo action.** `arcengine.GameAction` is `RESET` plus
     `ACTION1`–`ACTION7`; `ARCBaseGame.perform_action` keeps no state history.
     `ACTION7` is an ordinary game-specific action. `RESET` restarts the
     *current level* once any action has been taken, without rewinding the
     action count. **`design-thesis.md` §8 assumed a cheap two-action
     probe-and-revert. That mechanism does not exist.**
  2. **The 64×64 frame is often an upscaled small board.** ARCEngine renders at
     the camera resolution, integer-upscales, letterboxes, then draws a 64×64 UI
     on top. 12 of 25 public games have a camera smaller than 64×64 — bp35 and
     lf52 are 8×8, i.e. **8× upscale**. Recovering the scale factor and the
     playfield rectangle is therefore a mandatory perception primitive, and the
     click space is much smaller than D-005 assumed.
  3. **Legal actions are published.** Every frame carries `available_actions`.
     Masks range from `[6]` (click-only; 5 of 25 games) to all seven. They did
     not change across 60 random actions in any game.
  4. **Human baselines are available offline**, per level, in
     `EnvironmentInfo.baseline_actions`. We can compute true RHAE locally from
     day one.
  5. **Frames arrive in sequences of 0–N**, up to 42 observed. Zero frames means
     the action was a no-op. Levels per game are 6–10 (183 total), not "at least
     six" as an open-ended claim.
  6. **The import path was wrong**: the package is `arc_agi`, and `make` is an
     `Arcade` method, not a module-level function.
  7. **Kaggle accelerators confirmed**: `cpu` / `t4` / `p100` / `rtx6000`, with
     `rtx6000` reserved for this track. A secondary-source claim of a $10,000
     runtime-cost allowance with third-party compute is inconsistent with the
     no-internet rule and is recorded as **unverified, do not plan against it**.
  8. **The engine is deterministic** under `seed=0`: identical action sequences
     produced byte-identical frame sequences in all 25 games.
- **Options rejected:**
  - *Rewrite `design-thesis.md` §8 to remove the Undo argument* — the thesis is
    the record of what we believed and why. Silently repairing it would hide
    that we reasoned from a false premise for a day. Annotate in place instead
    and re-derive the exploration argument deliberately, as its own decision.
  - *Leave `benchmark.md` alone and note corrections only here* — the file is
    described as the source of verified facts and will be read as such. A reader
    six months from now would take the Undo claim at face value.
  - *Re-verify by reading only the published docs* — the published prose is what
    produced the errors. Measurement against the pinned engine is strictly
    better evidence, so *[measured]* facts are marked as outranking prose.
- **Consequences:** §8's value-of-information argument now has no cheap probe
  mechanism and must be redone; irreversibility is a much bigger deal than the
  thesis assumed, because mistakes are permanent within a level. D-005's
  expected click-space reduction should be restated in terms of playfield cells
  rather than 4,096. Perception gains a new required primitive (scale and
  playfield geometry) that was not in the D-007 definition of done.
- **Revisit if:** the semi-private or private environments turn out to expose a
  different action set, or a future `arc-agi` release adds history/undo. Re-run
  the measurement script against every pinned engine bump.

---

## D-010 — Use engine sprite introspection as a ground-truth oracle for perception, never as agent input

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** Downloaded games are plain Python source in
  `environment_files/<id>/<version>/`, and the running game object exposes
  `game.current_level.get_sprites()` — every sprite's name, tags, x, y, width,
  height, scale, layer, visibility and pixel data. `state-of-play.md` lists
  "object identity may not track" as a top risk with the mitigation "hand-label
  a few sequences and measure". Hand-labelling is unnecessary: the engine will
  label them exactly, for every frame of every game, for free.

  Sprite names are deliberately obfuscated random strings (`ihdgageizm`,
  `eqatonpohu`), so there is no semantic leakage — but names are *type* labels
  shared by many instances, not instance identities.
- **Decision:** Build an oracle module that extracts pixel-perfect ground-truth
  instance labels by re-running the camera render with sprite identities in
  place of colours. Use it **only** to score segmentation and tracking in the
  test suite. It is imported by tests and evaluation scripts and by nothing on
  the agent's code path; the agent sees `FrameDataRaw` and nothing else. Enforce
  this with an import-boundary test.
- **Options rejected:**
  - *Hand-label a few sequences* — the plan of record. Tiny sample, expensive,
    and biased toward the cases a human finds easy to label. The oracle covers
    all 25 games and every frame.
  - *Let the agent read sprite data at development time as a shortcut* —
    directly violates Guardrail 2 and would produce a system that cannot work on
    the private set. The whole value of the oracle depends on it never crossing
    into the agent.
  - *Read the game source to design perception* — the same failure with extra
    steps: it is exactly "enumerating the kinds of games there are". Reading
    engine source (`arcengine`) is fine and is how the corrections in D-009 were
    found; reading the 25 *game* sources to inform design is not.
- **Consequences:** Object-identity quality becomes a number we can watch move,
  computed over all 25 games, which is what makes D-007's two-week tarpit
  condition actually checkable. Costs: the oracle must model occlusion and
  layering to be honest, and it is only available on public games — so it can
  measure perception but can never be a fallback at competition time. Note also
  that the sprite decomposition is not always the target: bp35's entire 8×8
  board is a **single** sprite, so components-per-sprite is many-to-one there.
- **Revisit if:** the oracle and any reasonable segmentation disagree so
  systematically that the oracle is measuring the wrong thing, or a future
  engine version stops exposing level sprites.

---

## D-011 — Add frame geometry as a fifth perception component, built first

- **Date:** 2026-08-30
- **Status:** accepted (extends D-007's definition of done; does not supersede it)
- **Context:** D-009 established that the 64×64 frame is frequently an integer
  upscale of a smaller camera, letterboxed, with a 64×64 UI layer drawn on top —
  12 of the 25 public games, down to an 8×8 camera at 8× on bp35 and lf52.
  D-007's definition of done lists four components (client, diff, segmentation,
  identity) and none of them recovers this.
- **Decision:** Add `perception/geometry.py` as a fifth component and build it
  **before** the other three perception components. It induces, from frames
  alone: the upscale factor, the block-grid origin, the playfield rectangle, and
  the UI regions outside it. Downstream components work in *cell* space where
  geometry is confidently recovered, and fall back to pixel space where it is
  not.
- **Options rejected:**
  - *Build it last, as a normalisation layer over the existing four* — the four
    would then be developed and first measured against a representation known to
    be wrong for 12 of 25 games. On bp35 every connected component would be 64×
    its true area and every "cell" would be a rendering artefact, so the
    headline object-identity number would be uninterpretable exactly where it
    matters most.
  - *Defer to a later session, holding D-007 as written* — preserves the
    original scope shape at the cost of knowingly measuring the wrong thing.
  - *Hard-code the camera size per game from `metadata.json`* — not available on
    the private set, and it is precisely the hand-specified structure Guardrail 2
    warns against. The scale is inducible from run-length structure, so induce it.
- **Consequences:** Session 1 is now five components, not four, and D-007's
  two-week tarpit clock effectively starts later. Segmentation and tracking gain
  a dependency on geometry being right, so geometry needs its own accuracy
  number against the oracle (D-010 gives us the true camera dimensions) before
  anything is built on top of it. The click-space claim in D-005 should be
  restated in playfield cells once this lands.
- **Revisit if:** geometry cannot be recovered reliably on the 13 games where the
  camera is already 64×64 and the induced scale is degenerate at 1, or if the
  induced scale disagrees with the true camera scale on more than a couple of
  games after a principled (not threshold-tuned) criterion.

---

## D-012 — The agent may consume `available_actions`, `tags` and `baseline_actions`, behind a provider that can withhold them

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** The engine supplies three pieces of free information the design
  thesis never accounted for: the per-frame `available_actions` mask, a per-game
  modality `tags` field (keyboard / click / keyboard_click), and per-level
  `baseline_actions` — the human baseline itself. The question is which of these
  the agent is permitted to condition on at runtime, given we cannot verify what
  the private environments expose.
- **Decision:** All three. `available_actions` arrives inside every frame and is
  unambiguously part of the observation. `tags` and `baseline_actions` are read
  through a single `GameMetadata` provider, and **every consumer must behave
  correctly when the provider returns `None` for them**, enforced by a test that
  runs the agent path with metadata withheld. Knowing the level's baseline turns
  RHAE into a live budget the planner can spend against, which is a real
  advantage and worth taking.
- **Options rejected:**
  - *`available_actions` only* — the conservative boundary, and the one
    initially recommended here. Rejected: it discards a genuine planning signal
    on an assumption about the private set that is itself unverified in the
    other direction. The withholding test buys the same safety without the loss.
  - *Observation only, ignoring `available_actions` too* — would force us to
    spend actions rediscovering the legal action set, which is the one thing
    RHAE punishes hardest, to protect against a field that provably arrives in
    every frame.
  - *All three with direct access, no provider indirection* — cheaper now, but
    the dependency would be scattered across the codebase and we would discover
    at evaluation time whether it holds. The indirection is what makes the
    assumption testable.
- **Consequences:** We are betting that the private environments expose
  `baseline_actions` before play. If they do not, the provider returns `None`
  and the planner loses its cost model — so the fallback path is load-bearing
  and must be exercised in CI, not written and forgotten. Any experiment result
  that depended on baselines must record whether metadata was available, or it
  will not be comparable to a private-set run.
- **Revisit if:** we get any evidence about what the semi-private or private
  environments return from `/api/games`, or if the withheld-metadata path is
  found to have decayed. Re-check on every `arc-agi` version bump.

---

## D-013 — Geometry induction: propose by run-length GCD, verify by cell uniformity

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** D-011 put frame geometry in scope and framed it as recovering
  "the upscale factor". First contact with real frames showed that framing is
  wrong. ARCEngine composites a scaled camera view with a UI layer drawn last at
  full 64×64 resolution, and on bp35, lf52 and m0r0 the UI *is* most of what is
  visible — bp35 has an 8×8 camera whose view is almost entirely empty
  background, with the whole game painted by `RenderableUserDisplay` at 1:1.
  A single global scale does not exist for those games.
- **Decision:** Do not ask for the camera's scale. Ask **at what granularity
  this game's content moves**, which is always well defined and is what
  segmentation and click enumeration actually need. Induce it by:
  1. **Propose** — the GCD of every interior colour run inside the mask of
     pixels that change across observed frames. Interior runs only: a run
     touching the frame edge is truncated, and feeding a truncated length to a
     GCD wrongly collapses it toward 1.
  2. **Verify** — every cell the proposal implies must hold exactly one colour
     in every observed frame, judged only over cells lying entirely inside that
     mask. On failure, walk down the proposal's divisors. Cell size 1 always
     verifies, so the pipeline is total and its failure mode is "no structure
     claimed", never "wrong structure claimed".
- **Options rejected — all three were built and measured, not imagined:**
  - *MDL over the largest block-uniform rectangle.* Tried first, because minimum
    description length is the selection criterion the design thesis adopts for
    dynamics. **Result: 6/23 scale agreement, and 18 of 25 games produced an
    unsound grid** — a cell spanning two colours. It fails for a reason worth
    remembering: **a solid region is compressible at every cell size**, so
    letterboxing and flat backgrounds always win the MDL comparison while
    carrying no information about the true scale. MDL selects between models of
    the *same* region; it cannot also choose the region.
  - *Verification over the changing region's bounding box.* Straightforward and
    completely wrong: the box contains full-resolution UI detail, which vetoed
    every candidate. **Result: cell size 1 on all 25 games** — the code claimed
    nothing anywhere. Fixed by judging only cells wholly inside the change mask.
  - *Proposing from the largest connected changing region rather than the whole
    mask.* Motivated: a small animating UI element drags the global GCD to 1,
    which is what costs us ar25, cn04 and vc33. It does rescue cn04 — but a
    smaller region has fewer runs constraining the GCD, so the estimate drifts
    upward, breaking ft09 (2 → 6) and sc25 (1 → 2). **Overestimating is the
    dangerous direction**, because a too-coarse cell silently merges genuinely
    distinct game cells, whereas underestimating falls back to exact pixel
    space. Net 16/23 against 17/23, with twice the overestimates. Reverted.
- **Result as accepted**, 40 random actions per game, seed 0, measured by
  `scripts/eval_geometry.py` against the D-010 oracle:
  - **17/23 exact agreement** with the true camera scale on camera-visible
    games; **18/23 at 150 actions**.
  - **0 unsound grids across all 25 games.** This is the property everything
    downstream depends on and the one that must never regress.
  - 1 overestimate (lp85, 2 → 4); 4 safe fallbacks to pixel space (ar25, cn04,
    lf52, vc33); 2 games UI-dominated, of which m0r0 is still recovered exactly
    at 5.
  - **Accuracy improves monotonically with observed frames** — dc22's
    overestimate resolves between 40 and 150 actions. Geometry gets better for
    free as an episode proceeds, which is a reason to re-induce as evidence
    accumulates rather than fixing it at reset.
- **Consequences:** Roughly half the public demo games run in pixel space, so
  the pixel path is not a neglectable fallback — it is the common case, and both
  paths must stay correct. D-005's "4,096 → 5–20 candidates" is achieved, but by
  segmentation more than by geometry: union click targets across all
  segmentations come to 15–25 on most games. The single overestimate (lp85) is a
  live risk to segmentation there and should be watched.
- **Revisit if:** any game produces an unsound grid; a per-region geometry model
  — different cell sizes in different parts of one frame, which is the honest
  model of a composited frame — becomes worth its complexity; or the number of
  games falling back to pixel space starts limiting search depth downstream.

---

## D-014 — Track objects by global assignment; measure identity against the oracle

- **Date:** 2026-08-30
- **Status:** accepted
- **Context:** Cross-frame object identity is D-007's named tarpit and the
  falsification condition for design-thesis §6's action-space pruning. It needed
  both an implementation and, more importantly, a number.
- **Decision:** Match components between consecutive frames by **minimum-cost
  global assignment** (Hungarian / Jonker–Volgenant, written out rather than
  taken from scipy, to keep the offline dependency set to what `arc-agi` already
  installs). Pair cost is built from exact evidence in decreasing order of
  weight: identical shape and colouring, identical silhouette after recolouring,
  mask overlap and size ratio, then distance moved. A pairing is made only when
  it costs less than the alternative explanation of one disappearance plus one
  appearance, which sets the bar without a tuned similarity threshold. Unmatched
  tracks **decay in confidence and are never deleted** (Guardrail 1, D-003), and
  newly appeared components are offered to dormant tracks for reacquisition,
  which is what makes occlusion survivable.
- **Options rejected:**
  - *Greedy nearest-component matching* — fails exactly where identity matters,
    on two identical sprites passing each other. There is a regression test for
    that case. With at most a few dozen components, optimal assignment is
    affordable, and internal compute is free.
  - *Deleting a track on a frame where its object is missing* — unsound given
    hidden state (D-003), and it makes occlusion permanently destroy identity.
  - *Committing to one segmentation and tracking only that* — the right notion
    of object is environment-dependent, so the tracker is parameterised by
    segmentation name and all candidates are produced.
  - *Hand-labelling a few sequences to measure it* — the plan in
    `state-of-play.md`, superseded by D-010's oracle, which labels every frame
    of every game exactly.
- **Result**, 40 random actions per game, seed 0, `scripts/eval_tracking.py`:

  | segmentation | id switch rate | non-degenerate pairs | fragmentation | merge |
  |---|---|---|---|---|
  | `colour_8` | **0.8%** | 8,231 | 2.55 | 1.43 |
  | `colour_4` | 1.6% | 8,532 | 3.23 | 1.40 |
  | `foreground_8` | 0.2% | 2,149 | 1.33 | **3.65** |

  `foreground_8`'s 0.2% is largely an artefact and is reported as one: it
  over-merges at 3.65 sprites per component, and on several games collapses the
  frame into one or two blobs, where identity trivially cannot switch. The
  switch rate is therefore also computed over **exclusive** sprites only — those
  whose covering component covers no other sprite — which removes that credit.
  On that measure **`colour_8` is the working default at 0.8% over 8,231
  pairs**, and it is what the tracker should be pointed at.
- **A limitation stated plainly:** the oracle reads the engine's *current*
  state, so ground truth exists only for resting frames — the last frame after
  each action. The intermediate frames of an animation cannot be scored this
  way, and they are precisely where tracking is hardest. **These numbers are an
  optimistic bound on animated games.** Fixing it needs the oracle sampled
  inside `perform_action`, via a renderer callback or an engine patch.
- **Consequences:** D-007's tarpit condition has **not** been triggered —
  identity works, on the evidence available, well inside the two-week mark. The
  action-space pruning design-thesis §6 depends on is therefore viable.
  Fragmentation of 2.55 is the weak spot: we split objects the engine considers
  whole, which will show up downstream as several induced rules where one would
  do.
- **Revisit if:** the switch rate on exclusive sprites exceeds a few percent on
  any new measurement; the resting-frames-only limitation is shown to be hiding
  a real failure; or fragmentation starts blocking dynamics induction.

---


## D-015 — Corrections to the perception claims after an independent audit

- **Date:** 2026-09-08
- **Status:** accepted
- **Context:** Session 2 opened with a read-through and re-measurement of
  everything session 1 built, rather than new work. The code held up better than
  the headline numbers did. Several claims in `state-of-play.md`, `README.md` and
  the module docstrings turned out to be true-but-flattering, and one was simply
  wrong. Logged here because the numbers are quoted elsewhere and someone will
  otherwise rely on them.
- **Decision:** Restate the perception results as follows. All measured at
  seed 0, `scripts/eval_geometry.py --actions 20`, and over the five recorded
  fixtures. Full write-up in `architecture.md` §3.

  1. **Geometry fires on 4 of 25 games**, not 18 of 23. Structured grids are
     found only on four (ft09, lp85, m0r0, sp80 at 150 actions; dc22 in place of
     lp85 at 20). The `18/23` scale agreement is real but **16 of the 18 matches
     are `cell=1` correctly agreeing with a true scale of 1** — credit for
     declining to guess where there is nothing to guess. Only ft09 and sp80 are
     non-trivial successes.
  2. ~~**dc22 is a false positive.**~~ **Withdrawn — see the correction below.**

  **Correction, same session, before commit.** Items 1 and 2 above were first
  measured at `--actions 20`. D-013's established protocol is 40, with a second
  reading at 150, and D-013 had *already recorded* that dc22's overestimate
  resolves between 40 and 150 actions. Re-run at 150:

  - **18/23, exactly reproducing D-013.** dc22 resolves to `cell=1`, correct.
    The overestimate at 150 actions is **lp85 (true 2, induced 4)** — which is
    precisely what D-013 recorded, not a new finding.
  - So the claim "dc22 is a false positive" is an artefact of an under-powered
    run, and the implication that the overestimate was newly discovered was
    wrong. D-013 stands unamended on both points.
  - **The structural reframing survives**, and is slightly stronger at 150:
    structure is still found on only **4 of 25 games** (the set shifts — dc22
    drops out, lp85 comes in — leaving ft09, lp85, m0r0, sp80), and **16 of the
    18 matches are the trivial `cell=1` case**. Two non-trivial successes,
    ft09 and sp80, unchanged.
  - **A finding that is genuinely new:** the answer is *action-count dependent*
    and the overestimate **migrates between games** — dc22 at 20 actions, lp85
    at 150. D-013 noted accuracy improves monotonically with frames; it did not
    note that the identity of the failing game changes underneath it. Anything
    that re-induces geometry during play (which D-013 recommends) is therefore
    re-inducing a *different* wrong answer at different points in an episode,
    not converging on one. That is worth knowing before segmentation is allowed
    to depend on it mid-episode.

  The lesson recorded plainly: this entry nearly wrote a false correction into
  the permanent record because a measurement was run at a cheaper setting than
  the protocol the original used. **Match the original's action count before
  claiming a number moved.**
  3. **"0 unsound cell grids on 25/25" is measured in-sample.** The soundness
     test induces geometry from a set of frames and verifies uniformity on the
     same frames, so it cannot fail by construction. This is precisely how dc22
     is both "sound" and wrong. An out-of-sample check — induce on `frames[:n]`,
     verify on `frames[n:]` — does not exist and should.
  4. **Click pruning is 8–196 candidates, not 15–25.** Measured per fixture:
     sp80 8, vc33 16, ls20 29, sb26 31, bp35 196. Still a 20×–500× reduction on
     4,096, so D-005 stands, but the quoted range holds on one fixture of five.
     Separately `Geometry.click_targets()` is **empty on four of five fixtures**
     and returns 256 on the fifth where segmentation returns 8 — the pruning
     that works comes from segmentation, not from the geometry layer D-011 built
     for it.
  5. **`segment.py`'s "roughly a half-and-half split" between cell space and
     pixel space is wrong.** It is 4/25. The pixel-space path does nearly all
     the work.
  6. **Everything is measured under random play.** Transitions are small and
     low-diversity — 1–58 changed pixels of 4,096, dominated by recolouring.
     Busy multi-mover scenes are absent from the evidence.

     *Second same-session correction.* This item first read "mostly no-ops and
     wall-bumping", generalised from sb26 alone. Measured across all five
     fixtures the no-op rate is **14% overall: 68% on sb26 and 0% on the other
     four.** The criticism of random play stands but the mechanism is different —
     the changes it produces are *tiny*, not *absent*. That is the second time
     this entry over-generalised from a partial measurement; both are left
     visible rather than tidied away.
  7. **The tracker is not run on animation frames at all** in evaluation, only
     on one resting frame per action. D-014 recorded that intermediate frames go
     *unscored*; they also go *unexecuted*, which is a different and larger gap,
     because deployment will feed the tracker whole 42-frame sequences.

- **Two findings in the other direction**, recorded with equal weight:
  - **`_hungarian` is correct.** Fuzzed against brute-force optimal assignment
    on 4,000 random rectangular cost matrices: zero suboptimal results. The
    hand-rolled implementation was the largest silent-wrongness risk in the
    layer and it is now retired as a concern.
  - **The layer's output accounts for essentially the whole pixel delta.** Over
    all non-trivial fixture transitions, the objects flagged as moved,
    recoloured, appeared or vanished cover 100% of genuinely-changed pixels on
    four of five games and 79% on sp80, with appearances and disappearances
    averaging near zero. The symbolic story is not silently dropping change.
    This is the strongest evidence to date that perception is not the weak link.

- **Options rejected:**
  - *Editing the original numbers in place* — forbidden by the documentation
    protocol, and the drift between claim and measurement is itself the useful
    record.
  - *Treating the geometry finding as a bug to fix now* — it is not a bug. The
    module correctly declines to claim structure on 21 games. The problem is
    that the headline metric awards it credit for that. Reframing the metric is
    the fix; rewriting the module is not.
  - *Suppressing the dc22 false positive with a special case* — Guardrail 2. A
    game-specific patch is exactly the hand-crafting we have committed against.
    The honest response is an out-of-sample soundness check that would catch the
    class.
- **Consequences:** `README.md` and `state-of-play.md` status tables are updated
  to the corrected figures. D-005 and D-013 stand; D-011's specific claim that
  geometry would carry click pruning does not. Anyone quoting a perception number
  should quote the random-play caveat with it.
- **Revisit if:** an out-of-sample soundness check finds further unsound grids;
  or measurement under non-random play moves any of these figures materially, in
  either direction; or the migrating-overestimate effect turns out to matter once
  geometry is re-induced during live play.

---

## D-016 — Object identity is not revisable; splits and merges stay out of scope for now

- **Date:** 2026-09-08
- **Status:** accepted
- **Context:** Asked directly whether the layer forms an initial guess at what
  the objects are and revises it with evidence, the answer turned out to be no,
  in a stronger sense than anyone had written down. Segmentation is recomputed
  from scratch every frame by a fixed rule; only *identity* persists, never
  *objecthood*. `track.py` imports from `segment.py` and nothing goes the other
  way, so no evidence gathered by tracking can ever change how a frame is carved.

  Tested directly: a bar that splits in two yields one continuing track plus one
  **brand new unrelated track**, with no record they were ever the same object.
  Two objects merging yields one continuing track plus one that is marked
  missing, decays to 0.6 confidence, and sits dormant forever, never reconciled
  with what it merged into. Assignment is strictly one-to-one by construction,
  so "one object became two" has no representation in the type system; the words
  *split* and *merge* do not occur in `track.py`.
- **Decision:** Accept this for now and do **not** patch the tracker. Record it
  as a named, quantified limitation in `architecture.md` §3.2–3.3, and treat
  revisable objecthood as work belonging to the **ontology** subsystem, to be
  specified after the dynamics experiment reports.
- **Options rejected:**
  - *Extend matching to one-to-many now* — this is a real redesign, not a patch:
    it needs split/merge events in the type system, a policy for confidence
    across a split, and a new metric, since id-switch rate is not defined when
    one object becomes two. Doing it before dynamics tells us whether
    fragmentation actually blocks rule induction would be building machinery
    against a guess. Guardrail 4's logic applies: spend evidence on the densely
    testable thing first.
  - *Make segmentation stateful and accumulate an appearance model per track* —
    attractive, and probably right eventually, but it changes segmentation from a
    pure function into the layer's second stateful component, and we would be
    tuning it against random-play data that contains almost no interesting
    transitions.
  - *Reduce fragmentation by defaulting to a coarser segmentation* —
    `foreground_8` fragments least, at 1.33, but merges most, at 3.65 sprites per
    component. D-014 already rejected it for that reason. Trading fragmentation
    for merging is not progress.
  - *Doing nothing and not writing it down* — the limitation is invisible from
    the outside; the tracker reports a confident, well-formed answer in both the
    split and the merge case. It has to be recorded or it will be rediscovered
    expensively.
- **Consequences:** Dynamics induction will see one object as ~2.55 and will be
  asked to learn several rules where one would do. We are choosing to find out
  how much that costs rather than pre-emptively fixing it. If it does block
  induction, the fix is a feedback channel from dynamics to segmentation — *these
  pieces always move together, treat them as one* — which the current code has no
  way to receive, so expect an interface change, not a local edit.
- **Revisit if:** dynamics induction is measurably blocked by fragmentation; or
  any game requires reasoning about an object that genuinely splits or combines
  (a piece breaking apart, two pieces joining), which one-to-one matching cannot
  express at all.

---

## D-017 — Keep a standing essay-format architecture document

- **Date:** 2026-09-08
- **Status:** accepted
- **Context:** The documentation protocol had five documents covering *why*
  (`design-thesis.md`), *environment facts* (`benchmark.md`), *prior art*
  (`field-review.md`), *current status* (`state-of-play.md`) and *choices*
  (`decision-log.md`). Nothing described **how the system is actually built**.
  Answering that question in conversation surfaced three limitations — memoryless
  segmentation, unrepresentable splits, unread `segmentation_name` — that were
  true of the code, invisible in every existing document, and not obviously
  wrong from the outside.
- **Decision:** Add `docs/architecture.md`, maintained as continuous prose rather
  than notes: descriptive, with worked examples drawn from real recorded data,
  and with an explicit **"what this does not do"** treatment for every component
  plus a standing section on what is deliberately not being built. Keep it
  current as subsystems land. `CLAUDE.md` is amended to make this part of the
  protocol.
- **Options rejected:**
  - *Expand `state-of-play.md`* — it is deliberately a short snapshot that gets
    rewritten each session. Architecture description needs to accumulate and be
    stable; mixing the two would destroy both.
  - *Rely on module docstrings* — they are good here, but they are local by
    nature. Every limitation found this session was a property of how components
    *combine*, or of something absent, and no docstring is the natural home for
    an absence.
  - *Rely on the decision log* — it records choices at a point in time and is
    append-only, so reconstructing current behaviour means reading fourteen
    entries and mentally applying supersessions. That is the wrong artefact to
    hand a newcomer.
  - *Auto-generate from code* — would document what exists and is exactly silent
    on what does not, which is the half that matters most.
- **Consequences:** One more document to keep honest, and a real risk it drifts
  out of date and becomes worse than nothing. Mitigated by requiring that every
  quantitative claim in it names the script and seed that produced it, so staleness
  is detectable by re-running rather than by judgement.
- **Revisit if:** it goes two sessions without being updated while the code
  changes, which would mean the protocol is not being followed and the document
  should be cut rather than left misleading.

---

## D-018 — The dynamics experiment measures all six segmentations rather than fixing one

- **Date:** 2026-09-08
- **Status:** accepted
- **Context:** A dynamics rule is stated in terms of objects, but six candidate
  segmentations exist and they disagree sharply — bp35 yields 191 components
  under `colour_8` and 7 under `colour_mass`. Nothing in the codebase selects
  among them (`architecture.md` §3.4), so the choice would have been made by
  whatever string someone typed into the experiment script — and the current
  default is `foreground_8`, which D-014 explicitly disowns.

  The decisive argument is about **interpretability of a negative result.** The
  dynamics experiment is the go/no-go on D-002 and therefore on most of the
  project. If induction fails under a single arbitrary carving, we cannot
  distinguish *"the bet is wrong"* from *"we carved the objects wrong"*. That
  ambiguity sits directly on the most consequential decision we will make, and
  it is cheap to remove now and expensive to resolve afterwards.
- **Decision:** Treat the segmentation as an **axis of the experiment, not an
  input to it.** Run dynamics induction under all six carvings and report the
  comparison. The headline result becomes a table over carvings rather than a
  single number. Do **not** build a persistent per-game selector yet.
- **Options rejected:**
  - *Fix one segmentation (`colour_8`, per D-014)* — the cheapest option and the
    one that would have happened by default. Rejected because it makes a
    negative result uninterpretable on precisely the question the project turns
    on. D-014's recommendation of `colour_8` was made on *identity switch rate*,
    which is not the same objective as *rule inducibility*, and there is no
    reason yet to think the two rank the carvings the same way.
  - *Build a persistent per-game segmentation selector now* — this was the
    original proposal and it is circular. To select a carving you must score it,
    and the natural score is how well induced rules predict held-out transitions,
    which is the experiment itself. It also commits to per-game state before we
    know the best carving even varies by game.
  - *Select by a perception-side proxy such as fragmentation or merge rate* —
    those are measured against the oracle, which is test-only and may never
    inform the agent's path (D-010). Worse, D-010 records that a sprite is not
    always an object — bp35's whole board is one sprite — so lowest
    fragmentation is not obviously the right objective. It would be optimising a
    proxy we have already written down as unreliable.
  - *Run all six on all 25 games unconditionally* — compute. One carving costs
    824 ms/frame on bp35; six across tens of thousands of frames including
    intermediate animation frames is plausibly hours. Run the full sweep on a
    subset and a single carving on the rest, with the split decided by measured
    cost rather than assumed.
- **Consequences:** Roughly 6× compute on whichever games get the full sweep, so
  the run must be sized before it is launched — this reinforces the existing
  note in `state-of-play.md`. "Does dynamics induce?" no longer has a one-line
  answer, which is a real cost in communicating the result and an acceptable one.
  If one carving wins everywhere, the persistent selector deferred here may never
  be needed at all. If the winner varies by game, then D-016's deferred ontology
  work acquires its first concrete requirement *and* an objective to optimise
  against, which is a better position than guessing at one now.

  This partially answers `architecture.md` §3.4: the candidate population stops
  being generated-and-ignored, at least inside the experiment. It does not yet
  make the agent capable of choosing at run time.
- **Revisit if:** compute sizing shows the six-way sweep is infeasible even on a
  subset, in which case fall back to a principled two or three rather than one;
  or the carvings turn out near-indistinguishable on induction quality, in which
  case the axis is wasted effort and one should simply be fixed.

---

## D-019 — Compute sizing and the chance baseline, and what they settle

- **Date:** 2026-09-08
- **Status:** accepted
- **Context:** D-018 left the six-carving sweep contingent on a compute sizing it
  did not have, and guessed it might need to run on a subset of games. Separately,
  the dynamics experiment's stated success criterion is "predicts held-out
  transitions **better than chance**", and nobody had ever measured what chance
  is. Both are cheap prerequisites and both were run before designing the
  experiment, on the recorded fixtures, no engine required.

### Experiment 1 — compute sizing

- **Question:** is the six-carving sweep affordable, and where does the cost sit?
- **Setup:** full pipeline (geometry, then all six segmentations, then a tracker
  per carving) over the first 25 frames of each of the five fixtures. Seed 0,
  deterministic, no engine.
- **Result**, ms per frame for the complete six-carving pipeline:

  | game | components (`colour_8`) | ms/frame, all six |
  |---|---|---|
  | sp80 | 7 | 50 |
  | vc33 | 11 | 111 |
  | ls20 | 19 | 116 |
  | sb26 | 26 | 334 |
  | **bp35** | **191** | **2,850** |

  Mean 692 ms/frame, but the mean is meaningless — **bp35 alone is 10–50× every
  other game**, and drives the average single-handedly. Cost tracks component
  count, which tracks whether geometry found structure (§3.7 of
  `architecture.md`). Two carvings are effectively free: `colour_mass` (2–7 ms,
  it is O(colours) not O(components)) and `foreground_4/8` (0 ms on four of five
  fixtures). The expensive three are `colour_4`, `colour_8` and `bbox`.

  Mean frames per action across fixtures: **5.1**. Extrapolating to 25 games ×
  100 actions ≈ 12,700 frames: **roughly 2–2.5 hours** for all six carvings,
  assuming a handful of bp35-class games and the rest cheap.

- **Did it change our mind? Yes.** D-018 assumed the sweep would need rationing
  to a subset. It does not. Two and a half hours is a batch run, not a
  constraint. **The full six-carving sweep goes on all 25 games**, superseding
  D-018's "subset" mitigation while leaving its decision intact.
- **A structural point that survives the good news:** because the recorded
  episodes are replayed offline, the engine is stepped **once** and the sweep
  runs against recordings. Re-running a carving, or adding a seventh later, costs
  no actions and no engine time. Record once, analyse many times.

### Experiment 2 — what is "chance"?

- **Question:** what do trivial predictors score on the target "predict the exact
  resting frame after an action", so that "better than chance" means something?
- **Setup:** 120 transitions across the five fixtures. Two baselines. **B0
  static:** predict nothing changes. **B1 per-action replay:** remember the exact
  pixel delta last seen for this action id and replay it.
- **Result:**

  | | all 120 transitions | the 103 where something changed |
  |---|---|---|
  | B0 static | 14% | 0% |
  | B1 per-action replay | 14% | **0%** |

  No-op rate is **14% overall — 68% on sb26 and 0% on the other four fixtures.**

- **Did it change our mind? Yes, twice.**
  1. **The bar is on the floor, which is good news.** On transitions where
     anything happens, both trivial predictors score exactly **zero**. Exact-frame
     prediction is therefore a clean, unambiguous success criterion: any non-zero
     rate is real signal, and there is no risk of a flattering baseline being
     mistaken for induction working. B1 scoring 0% also confirms that pixel-level
     deltas do not transfer between occurrences of the same action — which is
     precisely the argument for representing dynamics over *objects* rather than
     pixels, and is the first direct evidence for it rather than an assumption.
  2. **It corrected a claim already written into three documents.** D-015 item 6,
     `architecture.md` §4 and `state-of-play.md` all said random play is "mostly
     no-ops", generalised from sb26. It is 14%. The criticism of random play
     stands, but the mechanism is that its transitions are **tiny** (1–58 changed
     pixels of 4,096, dominated by recolouring), not that they are absent. All
     three documents are corrected in place with the error left visible.

- **Decision:** Adopt **exact resting-frame match on changed transitions** as the
  primary success metric, against a measured chance baseline of 0%. Report the
  no-op fraction separately rather than letting it inflate the headline.
- **Options rejected:**
  - *A softer object-level metric as the primary* — "did we get the moving
    object's displacement right" is more forgiving and more informative about
    *why* a prediction failed, but it is scored in terms of a segmentation, and
    D-018 makes the segmentation a variable. A metric that varies with the axis
    you are sweeping cannot compare across that axis. Keep object-level scoring
    as a **secondary diagnostic**, not the headline.
  - *Including no-op transitions in the headline* — sb26 would score 68% for
    predicting nothing, and a game-weighted average would look like partial
    success where there is none.
  - *A per-pixel accuracy metric* — 4,096 pixels of which ~20 change means
    predicting "no change" scores 99.5%. Useless.
- **Consequences:** the experiment now has a defined bar, a sized budget, and no
  reason to run on a subset. What remains undecided is the **driving policy** —
  how transitions are generated — and the **rule language**. Neither is settled
  by this entry and both block the experiment proper.
- **Revisit if:** the driving policy changes the no-op rate materially, which
  would change what the headline metric is measured over; or exact-frame match
  turns out to be so harsh that every carving scores zero, in which case the
  secondary object-level diagnostic becomes the only signal and needs promoting.

---

## D-020 — Driving policy: round-robin tested and rejected; random retained

- **Date:** 2026-09-08
- **Status:** accepted — the driving decision stands; the *targeting* sub-claim
  ("object targeting does not beat a dumb lattice sweep") is **withdrawn by
  D-021**, which measured it against ground truth rather than against a proxy.
  The reasoning below is left exactly as written.
- **Context:** `state-of-play.md` carried the instruction *"do not drive the
  dynamics experiment with random play"*. That instruction rested on a
  characterisation of the fixture corpus that D-019 had already partly corrected
  — the claim that random play is "mostly no-ops" was wrong (14%, and 0% on four
  of five fixtures). Round-robin over the published action mask was proposed as
  the replacement, on two arguments: it *guarantees* action coverage where
  sampling leaves it to luck, and clicks aimed at segmented objects should beat
  clicks at random pixels.

  Both arguments were plausible and neither had been tested. Three drivers were
  implemented (`src/arc3/drive.py`) and measured against each other:
  `random` (the session-1 incumbent), `round_robin` with object-targeted clicks,
  and `lattice` — round-robin actions with clicks swept over a fixed 8-pixel grid,
  included as the null hypothesis for whether perception-guided targeting earns
  anything at all.

### Experiment — what corpus does each policy actually produce?

- **Setup:** all 25 games, 60 actions per game per driver, seed 0,
  `scripts/eval_drivers.py`. Metrics: no-op rate; mean changed pixels on
  transitions that changed; count of **distinct** changed-pixel signatures
  (a corpus of 200 copies of one event teaches one rule); and the fraction of
  legal actions tried at least once.
- **Result:**

  | driver | no-op % | mean changed px | distinct events/game | action coverage % |
  |---|---|---|---|---|
  | `random` | 20.3 | 51 | **40.1** | 99 |
  | `round_robin` | **19.3** | 51 | 36.9 | **100** |
  | `lattice` | 19.5 | **52** | 36.2 | **100** |

- **Did it change our mind? Yes — it killed the premise.** Round-robin is
  indistinguishable from random on every metric that matters, and is **worse on
  the one that matters most**: random produces ~9% more distinct events per game.
  Both of the arguments for it fail:

  1. **The coverage guarantee is worth almost nothing.** Random already reaches
     **99%** coverage at 60 actions, because most games publish only 4–7 legal
     actions. Round-robin buys one percentage point and pays three distinct
     events per game for it.
  2. **Object-targeted clicks do not beat a dumb lattice sweep** in aggregate:
     36.9 distinct events against 36.2, and 19.3% no-ops against 19.5%. D-005's
     pruning is not earning anything as a *driving* heuristic, whatever it is
     worth to a planner.

  A likely mechanism for round-robin's diversity deficit, offered as a
  hypothesis and not a finding: the engine is deterministic, so a fixed cyclic
  policy can fall into a limit cycle and revisit the same states, whereas a
  random walk breaks out of them. If true this is an argument against *any*
  strictly periodic driver, which is worth knowing before the "better methods"
  are designed.

- **Two per-game results that the averages hide, and that matter more than the
  averages:**
  - **vc33 is decisive for object targeting**: 54 mean changed pixels under
    object-targeted clicks against **1** for both random pixels and the lattice.
    A 50× difference on one game, averaged into invisibility across 25. So
    targeting is not useless — it is *game-dependent*, which is a different and
    more interesting claim than "it doesn't work".
  - **ft09 and lp85 are ~90% no-ops under every driver** (ft09 87–92%, lp85
    93–98%). Both are click-only games. No driver tested does better than
    wasting nine actions in ten on them, and this is where the corpus is
    genuinely starved. Under RHAE this is also where a real agent would haemorrhage
    score.

- **Decision:** **Retain `random` as the default corpus-generation driver.** Keep
  `round_robin` and `lattice` in the codebase as named, tested alternatives —
  they cost nothing to keep and vc33 shows the targeting axis is real. Withdraw
  the "do not drive with random play" instruction from `state-of-play.md`: it was
  never supported by measurement and is now contradicted by it.
- **Options rejected:**
  - *Adopt round-robin anyway on the coverage argument* — this was the plan of
    record and the tempting thing to do having built it. One percentage point of
    coverage against a 9% loss in event diversity is a bad trade, and adopting it
    would mean the measurement had been decorative.
  - *Delete the losing drivers* — they are tested, cost nothing, and `lattice`
    in particular is the only thing that made the object-targeting result
    interpretable. Without a null hypothesis "object targeting works" would have
    looked true on vc33 alone.
  - *Conclude the driving policy does not matter* — too strong, and the per-game
    spread contradicts it. It does not matter **in aggregate at 60 actions**. It
    matters enormously on individual games, in both directions.
  - *Design a smarter driver now* (novelty-seeking, or one that stops spending
    actions on inputs already known to be inert) — the obvious next move and
    deliberately not taken yet. The dynamics experiment is the go/no-go and does
    not need an optimal corpus, only an adequate one, and we now have evidence
    that random is adequate. Building an information-seeking driver before
    dynamics exists would also mean building it with no model to seek information
    *for*.
- **Consequences:** The dynamics experiment proceeds on random-play corpora,
  which means it can reuse the recording machinery and the existing fixtures
  unchanged, and removes a blocker that turned out to be imaginary. The ~90%
  no-op rate on click-only games is now a known, quantified hole in the corpus
  and should be reported alongside any per-game dynamics result — those two games
  will have roughly a tenth of the evidence the others do.
- **Revisit if:** dynamics induction succeeds on most games but fails on the
  click-only ones, which would point at corpus starvation rather than at the
  method; or a driver is needed for *play* rather than for corpus generation, at
  which point the objective changes completely and none of this transfers.

---

## D-021 — The ground-truth click map, and what it says about where to click

- **Date:** 2026-09-17
- **Status:** accepted
- **Context:** Two questions had been circling since D-005 and had never been
  answered against ground truth, only against proxies.

  The first is **where it is effective to click**. D-005 prunes the 4,096-pixel
  click space to 8–196 candidates by proposing one target per segmented object.
  That is a *proposal* count. Nobody had ever measured how many distinct things
  a click can actually do, so there was no denominator: 196 candidates might be
  a tenfold over-count or a tenfold under-count and the repository could not
  tell you which. D-020 then compared object-targeted clicks against a dumb
  lattice sweep, found them indistinguishable in aggregate, and concluded that
  "D-005's pruning is not earning anything as a driving heuristic". That
  conclusion rested on a proxy statistic — the count of distinct changed-pixel
  signatures produced over a rollout — chosen because no better one existed.

  The second is **when the click is worth spending an action on at all**. Six of
  the 25 games publish only `[6]`, six never publish `6`, and thirteen are
  mixed (`drive.py` docstring, re-verified this session). The published mask
  tells you the click is *legal*. Nothing told us whether it *does anything*.

  What made both answerable now is a property of the local engine that had been
  recorded but never exploited: **`RESET` restores the level start exactly.**
  Measured this session across 19 games × 4,096 clicks, the frame after
  `RESET` is byte-identical to the frame before the click every time, and the
  corroborating evidence is sp80, where 4,096 clicks each followed by a `RESET`
  produced exactly **one** distinct outcome — impossible under any drift.

  That makes the click map free *offline*. It is emphatically **not** free
  online: at evaluation every click is permanent (D-009) and a `RESET` costs an
  action without rewinding the action count. This is an instrument of the same
  kind as the sprite oracle (D-010) — ground truth to score perception against,
  never an input to the agent.

### Experiment — click every pixel, on every game that accepts clicks

- **Question:** of the 4,096 pixels, how many do anything; how many *distinct*
  things can a click do; and do the targets segmentation proposes reach them?
- **Setup:** `scripts/eval_clickmap.py`, seed 0, level-1 start state, all 19
  games that publish `6`. For each pixel: click it, hash the resting frame,
  `RESET`. Two proposal sets scored against the map — `object_targets` (the
  union of click targets over all six segmentations, which is what D-005
  proposes) and `lattice_targets(8)` (a 64-point sweep, D-020's null
  hypothesis). Reproduce with
  `python scripts/eval_clickmap.py --out clickmap.json`. ~4 minutes.
- **Two metrics, and the second is the one that matters.** *Precision* is the
  fraction of proposed targets that change anything. *Class coverage* is the
  fraction of the game's distinct outcomes that the proposal set can reach. A
  planner does not need every live pixel; it needs **one pixel per distinct
  outcome**. Precision measures waste, coverage measures whether a reachable
  move has been made unreachable — and only the second can silently cost a
  level.
- **Result:**

  | game | live % | distinct outcomes | obj targets | obj prec | obj coverage | lat prec | lat coverage |
  |---|---|---|---|---|---|---|---|
  | ar25 | **0.0** | 0 | 20 | — | — | — | — |
  | m0r0 | **0.0** | 0 | 12 | — | — | — | — |
  | sc25 | **0.0** | 0 | 30 | — | — | — | — |
  | sk48 | **0.0** | 0 | 54 | — | — | — | — |
  | sp80 | 100.0 | **1** | 10 | 100% | 100% | 100% | 100% |
  | cd82 | 100.0 | 2 | 20 | 100% | 100% | 100% | 100% |
  | ka59 | 100.0 | 2 | 16 | 100% | **100%** | 100% | 50% |
  | cn04 | 7.7 | 2 | 12 | 58% | **100%** | 6% | 50% |
  | lp85 | 2.3 | 2 | 51 | 6% | 100% | 3% | 100% |
  | dc22 | 100.0 | 3 | 48 | 100% | **100%** | 100% | 67% |
  | lf52 | 100.0 | 3 | 71 | 100% | **100%** | 100% | 67% |
  | vc33 | 100.0 | 3 | 16 | 100% | **100%** | 100% | 33% |
  | sb26 | 3.5 | 4 | 31 | 13% | 100% | 6% | 100% |
  | s5i5 | 100.0 | 5 | 37 | 100% | **100%** | 100% | 60% |
  | bp35 | 100.0 | 8 | 196 | 100% | **100%** | 100% | 88% |
  | ft09 | 7.0 | 8 | 71 | 13% | 100% | 13% | 100% |
  | tn36 | 100.0 | 11 | 96 | 100% | **100%** | 100% | 36% |
  | su15 | 82.8 | **143** | 34 | 85% | 3.5% | 88% | 2.8% |
  | r11l | 100.0 | **3096** | 47 | 100% | 1.1% | 100% | 1.7% |

- **Result 1 — the click's true branching factor is tiny on 13 of 15 games.**
  Where clicking does anything at all, the number of distinct outcomes is
  usually between **1 and 11**. On sp80 it is **one**: every one of the 4,096
  pixels produces the identical frame, so `ACTION6` there is not a spatial input
  at all, it is a **button**, and the (x, y) is decorative. Measured against
  this, D-005's proposal sets are **5–25× over-complete** (sp80 10 proposals for
  1 outcome; lp85 51 for 2; bp35 196 for 8). The pruning is real and large —
  4,096 down to tens — but it stops well short of the ~1–11 that is actually
  there.

- **Result 2 — object targeting beats the lattice, and D-020's sub-claim was
  measured on the wrong statistic.** Head to head on class coverage over the 15
  games where clicking does anything: **objects win 9, tie 5, lose 1.** The
  losses and ties are informative — objects and lattice tie at 100% on the five
  games whose outcomes are easy to reach, and objects lose only on r11l, where
  both are hopeless (1.1% against 1.7% of 3,096 outcomes). Every game where the
  two differ materially goes to objects, sometimes hugely: vc33 100% against
  33%, tn36 100% against 36%, ka59 100% against 50%.

  D-020's measurement was not wrong, and its *decision* — retain random as the
  corpus driver — is untouched. But the statistic it used, distinct
  changed-pixel signatures over a rollout, cannot see this: a lattice that
  reaches one outcome class repeatedly still accumulates signature diversity
  from the states it passes through. D-020's own vc33 anomaly (54 changed pixels
  under object targeting against 1 for the lattice, "a 50× difference on one
  game") is now explained rather than filed as a curiosity: the lattice reaches
  **one of vc33's three outcome classes**. The generalisation "object targeting
  does not beat a dumb lattice sweep" is **withdrawn**; it holds for corpus
  diversity and fails for outcome reachability, which is what a planner needs.

- **Result 3 — the published action mask is an upper bound, not a guide.** On
  **four of nineteen games** (ar25, m0r0, sc25, sk48) the engine publishes `6`
  in `available_actions` and **not one of the 4,096 pixels does anything** at the
  level-1 start state. D-012 lets the agent read the mask; this says what the
  mask is worth. A legal action is not a useful one, and on 21% of the
  click-publishing games an agent that trusts the mask at the start state will
  spend its clicks on nothing.

  A second, smaller instance of the same point: **`RESET` is never published in
  `available_actions`, yet is fully functional when stepped.** The mask is not a
  description of what works in either direction.

### Follow-up experiment — is an inert click a property of the game or of the state?

- **Question:** Result 3 is measured at one state. Does a click that does nothing
  at the level-1 start do nothing *anywhere*? The two readings have opposite
  consequences: the first would license writing the action off, the second makes
  doing so a Guardrail 1 hard rejection.
- **Setup:** `scripts/eval_click_liveness.py`, seed 0, all 19 click games. Walk a
  random rollout; at depths k = 0, 5, 10, 20, 40, replay to that state and test
  what fraction of a candidate set (object targets plus a 16-pixel lattice)
  changes the frame. Weaker question than the full map, ~2,400 engine steps per
  game instead of ~330,000. `-` means the rollout terminated before that depth.
- **Result**, percentage of candidate targets that are live:

  | game | k=0 | k=5 | k=10 | k=20 | k=40 |
  |---|---|---|---|---|---|
  | ar25 | 0 | 0 | 0 | 0 | 0 |
  | sk48 | 0 | 0 | 0 | 0 | 0 |
  | **m0r0** | **0** | **100** | **100** | **0** | **0** |
  | **sc25** | **0** | **26** | **25** | **24** | **24** |
  | **cd82** | 100 | 100 | 100 | **7** | **7** |
  | **ka59** | 100 | 100 | 100 | **6** | **6** |
  | **sp80** | 100 | 100 | 100 | **0** | – |
  | **r11l** | 100 | 100 | 100 | **0** | – |
  | **dc22** | 100 | **5** | 100 | 100 | 100 |
  | cn04 | 25 | 100 | 100 | 23 | 23 |
  | bp35, lf52, s5i5, tn36, vc33 | 100 | 100 | 100 | 100 | 100 |
  | ft09 | 15 | 15 | 15 | 15 | 15 |
  | lp85 | 4 | 4 | 4 | 4 | 4 |
  | sb26 | 9 | 8 | 8 | 8 | 8 |
  | su15 | 82 | 82 | 82 | 82 | 82 |

- **Result — click liveness is a property of the state, not of the game, on at
  least eight of nineteen.** Two of Result 3's four inert games come alive:
  **m0r0 goes 0% → 100% within five actions** and back to 0% by twenty, and
  **sc25 goes 0% → 26%** and stays. Four games move the other way — cd82, ka59,
  sp80 and r11l are fully live for ten actions and then collapse to near or
  exactly zero. dc22 is non-monotonic (100 → 5 → 100). Only **ar25 and sk48** are
  inert at every depth tested, and even that is one random trajectory, not a
  proof.

  Two games (sp80, r11l) **terminated before depth 40 under random play**, which
  is why their k=40 cell is empty; the 0% at k=20 and the termination are
  probably the same event.

- **Did it change our mind? Yes, and it caught an error in this entry's own
  first draft**, which read Result 3 as a property of the game and proposed
  writing inert actions off. That is exactly the hard rejection Guardrail 1
  forbids, and the measurement shows what it would have cost: an agent that
  wrote m0r0's click off at the start state would have discarded the only input
  that does anything five actions later.

- **Result 4 — two games are genuinely position-rich**, and they are the ones
  the object prior fails on. r11l has **3,096 distinct outcomes** from 4,096
  pixels and su15 has 143. These are not games where a handful of targets can
  stand in for the click space, and any planner that assumes a small click
  branching factor will be wrong on them. They should be watched as the
  worst case for D-005 rather than averaged away.

- **Did it change our mind? Yes, on three things.** The branching-factor result
  reframes what pruning is for; the head-to-head withdraws a generalisation from
  D-020 that was about to be built on; and the four inert games break the
  assumption that the mask indicates usefulness.

- **Decision:**
  1. **Keep object-derived targets as the click proposal set**, on class
     coverage rather than on precision. The case for D-005 is now ground truth
     rather than inference from a proxy: object targets reach **100% of the
     distinct outcomes on 13 of the 15 live games**, and win or tie against the
     lattice on 14 of 15.
  2. **Add outcome-class merging as the next step on top of it.** Because two
     targets that produce the same resting frame are the same move, the agent
     should collapse them the moment it observes it, taking the effective click
     branching factor from ~46 proposals toward the measured ~1–11. This is
     learned online from transitions the agent is taking anyway and costs no
     extra actions.
  3. **Track per-action liveness as a decaying weight, never as a rejection**
     (Guardrail 1, D-003). An input observed inert is down-weighted and retried
     when the state has changed materially; it is never struck off. The
     follow-up experiment shows why in numbers.
  4. **Treat the root sweep's liveness findings as perishable.** What a click
     *does* to an object is likely to persist; *whether it does anything* is
     state-gated on eight of nineteen games and must be re-checked after the
     state moves.
  5. **`scripts/eval_clickmap.py` and `scripts/eval_click_liveness.py` are
     offline-only**, on the D-010 footing. It
     must never be importable from the agent's code path.
- **Options rejected:**
  - *Report precision as the headline* — it is the intuitive number and it is
    the misleading one. lp85 scores 6% precision and 100% class coverage: the
    proposal set wastes most of its targets and still reaches every distinct
    move in the game. Optimising precision would prune away reachability.
  - *Conclude from Result 1 that the click space is small and hard-code a small
    branching factor* — r11l's 3,096 outcomes forbid it, and Guardrail 2 forbids
    hand-crafting the hypothesis space from an eyeballed taxonomy of 25 games.
    The branching factor is something to **measure per game at runtime**, which
    is what decision 2 does.
  - *Run the full map at several states per game* — the exhaustive version of
    the follow-up experiment, and not run: returning to a depth-*k* state costs
    *k* replayed steps per pixel, making it a ~35-minute job. The cheap
    liveness probe answered the question that mattered. What remains unmeasured
    is whether the *outcome classes* drift with depth, as opposed to mere
    liveness, and that is the live gap this entry leaves open.
  - *Use the click map as an agent input* — it is 8,192 engine steps per game
    per state, all of them permanent at evaluation. Not a temptation worth
    leaving undocumented, since the code makes it look easy.
- **Consequences:** D-005 is now measured rather than argued, and is committed
  to on the coverage argument specifically. The planner gets a concrete
  follow-on job (class merging) and two named worst-case games (r11l, su15).
  D-020's driving decision stands; its targeting generalisation does not, and
  its status is amended to note the correction rather than superseded, since the
  decision it records is unaffected.
- **Revisit if:** outcome classes measured at a non-start state differ
  materially from the start state, which would mean class merging must be
  re-learned per state rather than per level — liveness already drifts, so this
  is a live worry rather than a hypothetical, and it is the first thing to
  measure if class merging underperforms; or class coverage on the
  position-rich games matters for score, at which point the object prior is not
  the right proposal mechanism for them and something continuous is needed.

---

## D-022 — Exploration when every probe is permanent: design thesis §8, re-derived

- **Date:** 2026-09-17
- **Status:** accepted
- **Context:** This entry pays a debt that has been outstanding since session 1.
  Design thesis §8 argued for exploration using a value-of-information test of
  the form *is this worth two actions?* — one to probe, one to undo. D-009
  established that **there is no Undo**, and §8 was annotated in place rather
  than rewritten, with a note that "the argument has to be re-derived without
  it" and that the re-derivation "is owed its own decision-log entry". It never
  got one. `state-of-play.md` has carried it as **open, owed a decision entry**
  through two sessions.

  The annotation's own guess at the answer was that with no rollback "every
  probe is permanent, so the cost of an uninformative action is not two actions
  but two actions plus whatever state damage it did". **That guess is too
  pessimistic, and the reason matters.**

### The structural fact the annotation missed

`RESET` is not Undo, but it is not nothing. It restores the level start
**exactly** — verified this session across 19 games and 4,096 clicks each
(D-021), with sp80's single distinct outcome over 4,096 trials as corroboration
that no drift occurs. It costs one action and does not rewind the action count
(D-009).

So the level is **not a one-way corridor**. It is a tree rooted at the level
start, and because the engine is deterministic under a fixed seed, any state at
depth *k* is re-reachable by `RESET` followed by replaying *k* actions. Reversion
exists; it just is not cheap, and its price is not constant.

**The cost of revert is *k*+1, not 2.** §8's two-action probe-and-revert is the
special case *k*=1. It is real, and it exists only at the root.

(One caveat kept visible: this is verified at level 1. D-009 records that
`RESET` restarts the *current* level, so the tree is rooted at each level's
start, but that has not been measured at level ≥ 2 and should be before anything
leans on it. A second, smaller observation from the same measurement: `RESET`
is **never published in `available_actions` yet is fully functional when
stepped** — another instance of D-021's Result 3, that the mask is an upper
bound on legality and not a description of what works.)

### What follows

1. **Exploration cost is linear in depth, and a sweep at depth *k* is
   quadratic-ish in aggregate.** Probing *b* branches at depth *k* and returning
   each time costs *b*(*k*+1). The same *b* branches at the root cost 2*b*.
   Information is therefore roughly ***k* times more expensive to buy at depth
   *k* than at the root.**

2. **So front-load: learn everything learnable near the root before descending.**
   This replaces §8's "is it worth two actions?" with **"is it worth *k*+1
   actions, and could it have been learned at a shallower depth?"** The second
   clause is the operative one and it is new: an agent that descends before
   exhausting the root pays for the same information several times over. This is
   a concrete ordering rule, not a disposition.

   **With one measured qualification, which arrived the same day as the
   derivation.** Front-loading only pays for information that *keeps*. D-021's
   liveness probe shows that on eight of nineteen games, which inputs are live
   changes with depth — so a root sweep can learn a fact that is false fifteen
   actions later. This does not overturn the cost argument, because the root is
   still where information is cheapest and a perishable fact bought for 2 actions
   beats the same fact bought for 21. It does mean the root sweep is **not a
   one-off phase that completes**, and any design treating it as a prologue to be
   finished before play begins is wrong.

3. **The case for the model gets stronger, not weaker.** With a correct dynamics
   model, replay happens internally and the *k*+1 vanishes entirely. The value of
   D-002 therefore **scales with the depth at which exploration is needed**.
   §8's original framing, by assuming a cheap Undo, accidentally *understated*
   the case for the model — a cheap Undo is precisely the condition under which
   model-free probing stays viable. Removing it removes the alternative.

4. **Irreversibility is now priced rather than feared.** §7's irreversibility
   heuristic was about avoiding damage. Under this derivation the question is
   sharper: an action is expensive not when it is "irreversible" in the abstract
   but when **recovering from it costs more than the information is worth**, and
   that is a number. Toggles are the cheap case and are now cheaply detectable —
   an outcome class that maps *S*→*S'* such that repeating it returns to *S* is
   a toggle, and D-021's class machinery finds these for free.

### What exploration play should therefore be

The user's question — *without a goal, do we need a goal hypothesis or do we
need exploration play?* — resolves into **both, in sequence, with the RHAE
budget deciding when to switch.**

**The value function has two phases.** While the dynamics model is uncertain,
states are ranked by **expected information gain about dynamics**, which is
well-defined without knowing the goal and is what Guardrail 4 already demands
("early actions go to dynamics disambiguation"). Once a goal hypothesis has
support, states are ranked against it. The switch is not a matter of taste: per-
level human baselines ship in `metadata.json` (D-012), and RHAE's penalty above
baseline is quadratic, so **the affordable exploration budget is computable
offline per level before the agent takes its first action.**

The ordering, cheapest first:

1. **Spend nothing on what is already free.** The action mask, the `tags`, the
   UI/playfield split, and above all the up-to-42-frame animation returned by any
   action being taken anyway. D-021 weakens this tier: the mask says legal, not
   useful, and on four of nineteen click games it is legal-and-useless.
2. **Sweep at the root, where revert costs 1.** Enumerate the distinct actions;
   for the click, use the object proposal set, which D-021 measures at 100%
   outcome-class coverage on **13 of 15** live games (the exceptions are the
   position-rich pair, r11l and su15). **This sweep is exactly where
   D-021's class merging happens** — the root sweep both explores and collapses
   the click branching factor from ~46 proposals to the measured ~1–11.
3. **Descend only on actions whose outcome class is already known**, so that
   descending spends movement budget rather than information budget.
4. **Re-check liveness after the state moves, and never write an input off.**
   D-021's follow-up measured click liveness at depths 0, 5, 10, 20 and 40 and
   found it **state-gated on eight of nineteen games** — m0r0 goes from 0% live
   at the root to 100% five actions later, cd82 and ka59 go the other way. So
   the root sweep's conclusions are of two kinds with different shelf lives:
   what an action *does* to an object is a dynamics fact and should persist,
   whereas *whether it does anything here* is a state fact and expires. Only the
   first is safe to carry down the tree.
5. **Never probe for the goal** (Guardrail 4). Unchanged, but now with a cost
   model behind it rather than an intuition.

- **Decision:** Adopt the above as the replacement for design thesis §8. Record
  in the thesis that §8's VOI test is superseded by the depth-priced version
  here. The two phases of the value function, and the RHAE-derived budget that
  switches between them, are the project's answer to "what ranks states before
  a goal is known".
- **Options rejected:**
  - *Treat every action as irreversible and plan conservatively* — the
    annotation's implicit position, and it is wrong on the measurement. It would
    forgo root sweeps that cost 2 actions and return the entire click structure
    of the level.
  - *Use `RESET` freely as a general-purpose undo* — tempting once exactness is
    verified, and wrong under RHAE: the action count does not rewind, so a
    policy that resets often is quadratically punished for it. `RESET` is for
    the root sweep and for recovering from a genuinely fatal state, not for
    routine backtracking.
  - *Derive the exploration budget from a tuned constant* — the baselines are in
    the metadata and the penalty is a published formula, so a constant would be
    a worse version of a number we can compute. It would also fail silently if
    D-012's metadata bet loses.
  - *Settle the value function now* — out of scope and not owed. This entry
    fixes the *exploration* half, which is answerable from the cost structure.
    Ranking states against a *goal* remains the weakest link (design thesis §7)
    and is untouched.
  - *Defer the whole re-derivation until dynamics induction reports* — the
    argument for waiting is that a negative result on D-002 would make it moot.
    Rejected because the derivation is what says the dynamics experiment's root
    sweep is affordable in the first place, and because two sessions of "owed"
    is long enough.
- **Consequences:** §8 is no longer load-bearing in its original form; the
  depth-priced version is what the planner will be built against. Commits us to
  a root-first exploration shape, which will look wasteful on any game where the
  interesting structure is deep, and that is the cost being accepted. Gives the
  planner a defined budget input (baseline actions) and therefore a hard
  dependency on D-012's metadata bet for the switch point, mitigated by the
  withheld-metadata path.
- **Revisit if:** outcome *classes* measured at depth differ materially from
  those at the root. Note that this condition has **already partly fired**:
  D-021's follow-up shows liveness drifting with depth on eight of nineteen
  games, which is the weaker of the two versions of this worry. What is still
  unmeasured, and what would genuinely collapse the ordering, is whether the
  *outcome classes themselves* — what a click does, not whether it does
  anything — drift the same way. That measurement is the highest-value cheap
  experiment this entry leaves undone. Also revisit if `RESET` turns out not to
  restore exactly at level ≥ 2, which is assumed and verified only at level 1;
  or if a game is found where the root sweep itself is fatal, making even
  depth-1 probing unsafe (sp80 and r11l terminate under random play before depth
  40, so this is not far-fetched).

---

## D-023 — Drop static-to-interactive transfer as a parallel objective

- **Date:** 2026-09-17
- **Status:** accepted
- **Context:** `field-review.md` §5 noted that the living survey reports
  static-to-interactive transfer as largely absent — nobody has established
  whether ARC-AGI-2 ability helps on ARC-AGI-3 — and flagged it as "answerable
  with modest compute and a cheaper, more certain research output than winning
  the track". It has sat in `state-of-play.md` under *Parked, not dead* through
  three sessions, and was raised again this session as one of the project's
  standing open questions.
- **Decision:** Drop it. Remove it from *Parked, not dead* and record here that
  it was considered and declined rather than forgotten.
- **Reasoning:** the objective is only cheap for someone who already has an
  ARC-AGI-2 system to transfer *from*. **This project has done no ARC-AGI-2
  work and has none planned.** The "modest compute" estimate in `field-review.md`
  silently assumed a static solver already in hand; without one, the cheaper
  parallel objective is not cheaper, it is a second project. With ~46 days to the
  2 November deadline, the core bet (D-002) untested, and available time dropping
  from early October, there is no version of this that is affordable.
- **Options rejected:**
  - *Keep it parked* — parking is not free. `state-of-play.md` is the file every
    session reads first, and an item that will never be picked up costs attention
    every time it is read and implies an optionality that does not exist.
  - *Do it with an off-the-shelf ARC-AGI-2 solver instead of building one* —
    would answer a question about someone else's system, not about ours, and the
    transfer question is only interesting for a solver whose internals you can
    inspect.
  - *Delete the `field-review.md` §5 note as well* — no. The survey's observation
    is true and worth a competent stranger knowing. What changes is our
    relationship to it, which is what this entry records.
- **Consequences:** the project is now single-track on ARC-AGI-3 with no hedge.
  If D-002 fails, there is no parallel result to fall back on, and the value of
  the work is entirely in the negative-result record — which is what `CLAUDE.md`
  says it should be anyway.
- **Revisit if:** an ARC-AGI-2 solver appears in this repository for some other
  reason, at which point the marginal cost of the transfer experiment drops to
  roughly what `field-review.md` originally assumed.

---

## D-024 — The rule language for dynamics induction

- **Date:** 2026-09-17
- **Status:** accepted
- **Context:** D-019 closed with "what remains undecided is the **driving
  policy** and the **rule language**. Neither is settled by this entry and both
  block the experiment proper." D-020 settled the driving policy. This settles
  the rule language, which was the last thing standing between the project and
  its go/no-go experiment.

  The difficulty is Guardrail 2. A rule language *is* a hypothesis space, so
  writing one down is the very act the guardrail warns against on the evidence
  that Tufa Labs found hand-crafted tools actively *hurt*. There is no way to
  run the experiment without one, so the question is not whether to hand-craft
  but how to hand-craft as little as possible and make the remainder visible.

### The language

A rule is `(action, selector) -> effect`, with a per-action default beneath.

- **Effects** are `static`, `translate(dx,dy)`, `recolour(c)`, `vanish`. Four,
  closed.
- **Selectors** are conjunctions of attribute *equalities*, arity 0 to 2, over
  five attributes: `colour`, `size`, `shape`, `rect`, `clicked`.
- **Selection** between candidate rule sets is by minimum description length,
  exhaustively over all 16 partitions of arity <= 2 per action.

### The three constraints that keep it honest

1. **Nothing may be conditioned on that perception does not already emit.**
   `colour`, `size`, `shape` and `rect` are pre-existing fields of
   `perception.segment.Component`. Not one feature was invented for dynamics.
   The single addition is `clicked` — whether the action's click landed inside
   this object — which is *relational* rather than descriptive, applies to every
   game with a pointer, and encodes no game-specific content. D-021 is the
   argument for it: a click's effect is about *what* you clicked, not where the
   pixel was.

2. **The effect vocabulary is the tracker's report, not a taxonomy of games.**
   Those four effects are exactly what `match_frames` can distinguish. The
   language is therefore precisely as expressive as the perception beneath it and
   no more, which means it cannot smuggle in game knowledge through the back
   door — if we want a richer effect we have to make perception see it first.

3. **MDL, not accuracy, and no regularisation constant.** Accuracy alone always
   prefers the finest partition: splitting on `size` gives nearly every object
   its own rule and scores perfectly in training. The bit costs come from the
   sizes of the spaces actually being encoded, so nothing is tuned.

### What the language deliberately cannot say

Recorded here because a negative result will turn on exactly these, and because
the experiment's job is to find out whether the *simplest* object-level language
beats chance. Starting richer would mean a positive result told us nothing about
where the power came from.

- **Nothing about objects that appear.** Every rule is keyed to an object present
  in the before-frame.
- **No object-to-object interaction.** "The block stops at the wall" is
  unsayable; there is no collision, containment or support.
- **No state beyond the current frame.** No velocity, no counters, no memory.
- **No numeric relations.** Equality only, so "bigger than 4 pixels" cannot be
  said and MDL will refuse to fake it by enumeration.
- **No simultaneous move-and-recolour**, and **no move-with-deformation.** The
  second turned out to matter enormously and is the subject of D-025.

- **Options rejected:**
  - *A program-synthesis DSL over grid transformations* (the ARC-AGI-1 house
    style) — far more expressive and the obvious thing to reach for. Rejected
    for this experiment because its search cost is large enough to become the
    thing being measured: a negative result would not distinguish "dynamics do
    not induce" from "the search did not find it in the budget". The point of
    starting at the floor is that failure is attributable.
  - *A learned transition model (small CNN or transformer over frame pairs)* —
    would sidestep the whole hand-crafting problem and is what the field would
    default to. Rejected on three counts: it needs orders of magnitude more data
    than 100 actions per game, it cannot be inspected when it fails, and a
    learned model is not *queryable* in the way planning needs (D-002's whole
    argument is that we want a model we can plan against, not one we can only
    sample). Parked, not dead — it is the natural fallback if symbolic induction
    fails for reasons of expressiveness rather than principle.
  - *Rules keyed on track identity rather than attributes* — i.e. "object #7
    moves", using the tracker's persistent ids. Strictly more powerful, and it
    would solve the case where the mover is identifiable only by its history
    rather than by any attribute. Rejected *for now* because a track id does not
    generalise across levels or episodes, so a model written in track ids cannot
    persist the way design thesis section 9 requires. This is the most likely
    first extension and D-025 records the evidence for it.
  - *Letting an LLM propose the rules* — deferred by scope, not rejected on
    merit. The architecture reserves the LLM for goal proposal (D-004), and
    introducing it here would make the experiment untestable offline and
    unreproducible from a log.
  - *A weighted population of rule sets rather than the MDL-best one* — this is
    what Guardrail 1 and D-003 actually require, and the experiment violates it
    knowingly. A population cannot rescue a language that cannot express the
    dynamics, so the single best hypothesis is the right instrument for a
    go/no-go. If the experiment succeeds, the population is the first thing to
    add, and the MDL scores already computed are the natural weights.
- **Consequences:** the experiment can run, and its failures are attributable to
  a named, small set of expressive limits rather than to a search budget. Commits
  us to a language that will need at least one extension before it can model
  most games — which is a finding, not a defect, and is what D-025 reports.
  Because the language is small, every one of its limits is now a measurable
  quantity rather than a worry.
- **Revisit if:** the deformed-move rate stays high under every segmentation,
  which would mean the effect vocabulary rather than the carving is the binding
  constraint; or exact-frame match is non-zero but low everywhere, which would
  point at the absent interaction rules; or a population of hypotheses is needed
  to get off zero at all, which would reverse the narrowing above.

---

## D-025 — The dynamics-induction experiment: a weak pass on D-002

- **Date:** 2026-09-17
- **Status:** accepted
- **Context:** The go/no-go deferred since D-007 and named in `state-of-play.md`
  as "the option deliberately deferred... and the go/no-go on D-002 and
  therefore on most of the project". Design thesis falsification condition #1:
  *"If simple object-tracking plus rule induction cannot predict held-out
  transitions better than chance on the public demo environments, section 4 is
  wrong and the whole model-based plan goes with it."*

  Everything it needed was in place: D-018 made the segmentation an axis, D-019
  sized the compute and measured chance at **0%**, D-020 settled the driving
  policy, D-024 settled the rule language.

### Setup

- **Corpus:** `scripts/record_corpus.py`, 25 games x 100 actions, random driver,
  seed 0, `RESET` on terminal (recorded once, 15 s, replayed offline
  thereafter). 2,500 actions, 7,687 frames.
- **Sweep:** `scripts/eval_dynamics.py --out dynamics_sweep.json`. All six
  carvings x 25 games. **7,888 s (2 h 11 m)**, against D-019's estimate of
  2-2.5 h — the sizing held.
- **Split:** temporal and within-game, first 70% induce, last 30% test. This
  departs from the letter of `state-of-play.md`'s "hold out by game, not by
  transition", and the reason is recorded in the script docstring: dynamics are
  *per game*, so a cross-game split would measure transfer between unrelated
  games, which nothing in the thesis claims. The protection the instruction was
  after — no near-duplicate transitions across the split, no per-game tuning —
  is preserved.
- **Metric:** D-019's, unchanged. Exact resting-frame match on changed
  transitions. Chance = 0%, re-measured on this corpus by both baselines.
- **Starved games:** ft09 and lp85 reported separately per D-020.

### Result

| carving | mean exact | exact>0 | medres<=2 | mean chgobj | deformed moves |
|---|---|---|---|---|---|
| **colour_4** | **4.8%** | **5/23** | 10/23 | 36.3% | 52.5% |
| colour_8 | 4.8% | 5/23 | 10/23 | 36.4% | 53.3% |
| foreground_4 | 1.8% | 3/23 | 10/23 | 28.8% | **14.0%** |
| foreground_8 | 1.8% | 3/23 | 10/23 | 28.7% | 18.8% |
| colour_mass | 4.6% | 4/23 | 9/23 | 34.3% | 73.9% |
| bbox | 4.2% | 5/23 | 9/23 | 37.2% | 55.6% |

Best per game, across all six: **exact>0 on 7/23**, `medres<=2` on **13/23**,
changed-object accuracy >=50% on **14/23**, >=90% on **6/23**.

Per-game bests: dc22 44.4%, m0r0 26.1%, cn04 21.1%, tr87 20.0%, wa30 16.0%,
ar25 7.4%, sc25 4.2%. Starved games: **0.0%** exact at 98% no-op, as expected.

- **Verdict: the falsification condition is not met, but it is not comfortably
  cleared either.** Held-out exact-frame prediction is non-zero on 7 of 23
  games where chance is exactly zero, reaching 44% on dc22. Design thesis
  section 4 survives. It survives weakly: 16 of 23 games score zero under every
  carving, and 4.8% mean is nowhere near what planning needs.

- **Result 1 — the best carving is game-dependent, which vindicates D-018.**
  `colour_4` wins on dc22, m0r0, sc25, tr87 and wa30; `foreground_4` wins on
  ar25 and cn04 and is the *only* carving that finds signal on either. The union
  across carvings (7 games) is strictly larger than any single carving (5), so
  maintaining several notions of object is not redundancy — it is the difference
  between finding dynamics on 5 games and on 7. Had the experiment fixed one
  carving, as D-018 was written to prevent, it would have under-reported by 29%
  and attributed the loss to the method.

- **Result 2 — deformation is NOT the binding constraint, refuting the
  hypothesis this session formed before running the sweep.** Mid-experiment
  diagnosis found that 27-48% of real moves are unrepresentable because the
  object changed silhouette as it moved, and predicted that `foreground_*`,
  which keeps a multi-coloured sprite whole instead of fragmenting it, would cut
  that rate and improve prediction. **The first half was right and the second
  was wrong.** `foreground_*` cuts the deformed-move rate by a factor of
  three — 14.0% against 52.5% — and its exact-match score is **less than half**
  `colour_4`'s. Lower deformation, worse prediction.

  Within a single carving, deformation does track failure: the 18 zero-scoring
  games average 58.3% deformed against 31.9% for the seven that score. Across
  carvings it anti-correlates. So deformation is a *symptom* of a game being
  hard, not the mechanism of failure, and reducing it by merging objects costs
  more than it saves. The likely reason, offered as a hypothesis and not a
  finding: `foreground_*` merges distinct objects into one component, which
  destroys exactly the colour and size distinctions the selectors need to
  identify which object moves. Worth testing directly before acting on.

- **Result 3 — the headline metric substantially understates the mechanism, and
  sp80 is the proof.** Under `colour_4`, sp80 scores **100% changed-object
  accuracy** with a **median residual of 2 pixels** and **0% exact match**. A
  model that gets every object's behaviour right scores zero.

  This is the ls20 story from D-024's bug hunt, now visible as a general
  pattern rather than one game's quirk: the 64x64 frame is a composite with a
  bookkeeping layer drawn over it, and a move counter ticking two pixels per
  action is enough to fail an exact comparison forever. The three aggregates
  tell the story in order: exact>0 on 7/23, within-2-pixels on 13/23,
  changed-object accuracy >=50% on 14/23. **The object-level mechanism works on
  roughly 60% of games; exact frame reconstruction works on 30%.** The gap is
  small residuals, not wrong dynamics.

  D-019 chose exact-frame match because it is unambiguous and because chance is
  cleanly zero, and both remain true. What it did not anticipate is that the
  metric conflates *"the model of the game is wrong"* with *"there are two
  pixels of UI I cannot predict"*. That is a real defect in the metric, and it
  is recorded here rather than fixed by moving the goalposts after seeing the
  answer — see the rejected options.

- **Result 4 — 53 of 464 changed transitions in the zero-scoring games are
  unpredictable in principle**, because something appeared that no rule keyed to
  a before-frame object can produce (D-024's first stated limitation). That is
  11%, so it is a real ceiling but not the main story.

### Follow-up experiment — is the model evidence-limited or language-limited?

- **Question:** the sweep says dynamics induce weakly. Two very different
  diagnoses fit that, with opposite consequences. If accuracy climbs with
  training transitions, the binding constraint is **evidence**, and under RHAE
  the right response is to spend actions — D-022's exploration budget becomes
  the priority. If it plateaus, more actions buy nothing and the constraint is
  the **rule language**, so the response is expressiveness. Guessing wrong costs
  a session and, at evaluation, costs score quadratically.
- **Setup:** `scripts/eval_dynamics_curve.py --carving colour_4 --test-tail 30`,
  300-action corpus (`record_corpus.py --actions 300 --out corpus300`), seed 0,
  eight games chosen to span the outcome: five that score (dc22, m0r0, tr87,
  wa30, re86) and three that do not (cd82, cn04, ls20). Training sizes 10, 25,
  50, 100, 200 against a **fixed** test tail, so the curve is not confounded by
  a moving target.
- **Result**, mean over the eight games:

  | training transitions | 10 | 25 | 50 | 100 | 200 |
  |---|---|---|---|---|---|
  | exact on changed | 7.1% | 11.7% | **13.5%** | **13.5%** | **13.5%** |
  | changed-object accuracy | 47.7% | 55.4% | 62.3% | 63.0% | 63.4% |

- **Result — it is language-limited, and the saturation is abrupt.** Exact match
  gains **+0.0000** from n=50 to n=200 — identical to four decimal places — and
  changed-object accuracy gains 1.05pp. Everything happens by **n=50**: from 10
  to 50 transitions exact match rises 6.4pp and object accuracy 14.7pp, and
  then the curve simply stops. The three zero-scoring games are still at exactly
  zero with 200 training transitions, so their failure is not starvation either.

- **Two consequences, and the first is unexpectedly good news.**

  1. **Dynamics learning is cheap in actions: ~50 suffice.** This materially
     softens the central worry of D-022. Exploration budget was expected to be a
     hard constraint on model quality under RHAE's quadratic penalty; measured,
     the model extracts everything this language can extract from about fifty
     actions of *random* play. Actions are not the scarce input to dynamics.
     (Caveat kept in view: fifty actions of random play on 25 public games. It
     says nothing about a game needing a deep sequence before its mechanics are
     visible at all.)
  2. **More actions must not be the next investment.** Spending them on a model
     that stopped learning at n=50 is, in the script's own terms, the worst
     available trade.

- **The clinching case, and it is the same case as Result 3.** ls20 at n=200
  scores **100% changed-object accuracy and 0% exact match.** Every object's
  behaviour is predicted correctly, on held-out transitions, with 200 training
  examples — and the frame comparison fails, every time, on the two pixels of
  its move counter. sp80 is the same shape under the sweep. **Two independent
  games now show a perfect object model scoring zero**, which is what promotes
  Result 3's diagnosis from an anecdote to the reason the headline is 4.8%.

- **Did it change our mind? Yes, and it corrected the script that measured it.**
  The first version of the verdict heuristic compared the endpoints, saw 7.1% ->
  13.5%, and printed "still climbing — evidence-limited, more actions would
  help" — the exact opposite of the truth, for a curve that is flat across its
  entire second half. Endpoint comparison cannot distinguish a rising curve from
  one that rose early and stopped, which is the whole question. It now reports a
  saturation point and the second-half gain. Worth recording because the wrong
  verdict was the more comfortable one: it would have justified more data
  collection instead of harder representational work.

- **Did it change our mind? Yes, four times.** D-002 is not refuted, which was
  genuinely uncertain going in. D-018's multi-carving requirement turned out to
  be load-bearing rather than cautious. Our own mid-experiment hypothesis about
  deformation was wrong in a way that would have sent the next session down a
  blind alley had it not been measured. And the action budget turns out not to
  be the constraint on dynamics at all, which is the most useful single thing
  learned here.

- **A note on how this result was nearly the opposite.** The first run of this
  experiment reported **0% on every game under every condition** — a clean,
  publishable-looking refutation of D-002. All of it was wrong, and the four
  causes were in the scoring and the search rather than in the hypothesis:
  erase-to-background rendering, an unmodelled move counter, an MDL cost that
  priced partitions instead of descriptions, and an object-accuracy figure that
  was really measuring how often objects sit still. Each is now a named
  regression test in `tests/test_dynamics.py` recording the false conclusion it
  would otherwise cause. **The lesson is specific and worth carrying: a
  go/no-go experiment needs a positive control.** The one that broke this open
  was supplying ground-truth effects and asking whether the renderer could
  reproduce the frame at all; it could not, on 0 of 100, which localised the
  fault immediately. No amount of staring at the induction would have found it.

- **Decision:**
  1. **D-002 stands. Continue with the model-based plan**, on the strength of
     7/23 games with non-zero exact prediction against a zero baseline and
     14/23 with a working object-level model.
  2. **Keep all six carvings in the pipeline** and select per game. D-018 is
     upgraded from a methodological precaution to a load-bearing design
     decision.
  3. **The next subsystem is the bookkeeping layer, not a richer rule
     language and not more actions.** It is the cheapest large win available: it
     is spatially separable, learnable from change frequency alone, and it is
     what stands between a perfect object model and a zero score on **both**
     sp80 and ls20 — the latter demonstrated at 100% object accuracy with 200
     training transitions. Predicting a counter is a far smaller problem than
     object interaction.
  4. **Do not spend the next session buying more actions.** The curve saturates
     at n=50. Fifty actions of random play extract everything this language can
     extract, which is good news for the RHAE budget and a firm instruction
     about where not to invest.
  4. **Do not act on the deformation hypothesis.** It is refuted as a cause.
- **Options rejected:**
  - *Declare the bet vindicated* — 4.8% mean and 16 of 23 games at zero does not
    support it, and the project's value depends on not overclaiming here.
  - *Declare the bet refuted and stop* — the falsification condition as written
     is "better than chance", chance is measured at exactly 0%, and seven games
     are above it with one at 44%. Writing this off would be the mirror of the
     first run's error: accepting a headline number without checking what
     produced it.
  - *Retroactively promote `medres<=2` or changed-object accuracy to the
    headline* — the tempting move, since both look much better (13/23 and 14/23
    against 7/23), and it is exactly what D-019's "options rejected" forbade in
    advance. Changing the primary metric after seeing which one flatters the
    result would make every future comparison uninterpretable. The headline
    stays exact-frame; the others are reported beside it as the diagnostics
    D-019 designated them.
  - *Fix the metric by excluding UI regions from the comparison* — this is the
    `exact_ex_bookkeeping` figure, and it is kept strictly as a diagnostic
    because it can be vacuous: vc33 changes **nothing** outside the induced mask
    and scored 100% on it with 0% object accuracy. It now carries a denominator
    and a `vacuous` flag for that reason.
  - *Extend the rule language now* (interaction, appearance, track identity) —
    the obvious response to 16 zeros, and premature. Result 3 says a large part
    of the gap is residual bookkeeping rather than missing expressiveness, and
    the learning curve says which of the two binds. Spending a session on
    expressiveness before reading that would be guessing.
  - *Re-run the whole sweep on the bigger corpus to firm up the numbers* — 30
    test transitions per game is coarse (3.3% resolution per transition), and a
    300-action corpus is recorded. Rejected outright once the curve landed:
    accuracy is identical at n=50 and n=200, so a larger corpus would change no
    conclusion and cost another two hours.
- **Consequences:** the project continues on its stated thesis, with the first
  quantitative evidence for it rather than an argument. Commits us to per-game
  carving selection, which multiplies pipeline cost by six unless something
  later selects among them cheaply — and nothing does yet (`architecture.md`
  section 3.4). Names the bookkeeping layer as the next subsystem, ahead of the
  richer rule language that the zero-scoring games superficially argue for.
- **Revisit if:** the bookkeeping layer lands and exact-frame match does *not*
  rise substantially on the games with high changed-object accuracy — sp80,
  re86, g50t and ar25 are the specific tests, and if predicting the counter does
  not convert them, then Result 3's diagnosis is wrong and the gap is really
  missing expressiveness; or the learning curve shows the model is
  evidence-limited, in which case actions rather than expressiveness are the
  thing to spend on and D-022's exploration budget becomes the priority; or a
  cheap selector among the six carvings cannot be found, in which case the
  six-fold cost has to be paid at runtime and that changes the compute picture.

---

## D-026 — The corpus could not teach a click rule, and now mostly can

- **Date:** 2026-09-24
- **Status:** accepted
- **Context:** D-025 reported that every click-only game scored zero on
  held-out prediction, and attributed it to the rule language. That attribution
  was confounded, and the confound was one this project built for itself.

  Every corpus in the repository was recorded with `RandomDriver`, which picks a
  legal action uniformly and, when that action is the click, picks a **uniformly
  random pixel**. That choice comes from D-020, which measured random against a
  cyclic round-robin and found random better on event diversity. D-020 was
  correct on its own terms and is not being reversed. But it was decided
  **before D-021 existed**, and D-021 changed the picture completely: it mapped
  all 4,096 pixels on all 19 click-publishing games and found that object-derived
  targets reach **100% of a game's distinct click outcomes on 13 of 15** live
  games, while the live fraction of raw pixels is as low as **7.0% on ft09 and
  2.3% on lp85**.

  So the two axes had each been measured separately and the winning combination
  had never been built:

  | | random pixel | object target |
  |---|---|---|
  | **random action** | `random` — used for every corpus | **did not exist** |
  | **cyclic action** | `lattice` | `round_robin[objects]` |

  The sharper version of the problem is about *learnability*, not waste. The
  rule language's only relational feature is `clicked` (D-024). For MDL to
  condition on it, the corpus must hit the **same object more than once** — a
  single observation cannot outweigh the bits needed to state a rule. Scattering
  fifty clicks uniformly over 4,096 pixels almost never does that. The corpus
  was close to structurally incapable of teaching a click rule, which makes
  D-025's click-game zeros uninterpretable as evidence about the language.

### The driver

`RandomObjectDriver` (`src/arc3/drive.py`): uniform random over legal actions,
exactly as `RandomDriver`, with the click target drawn from the object proposal
set instead of the pixel grid. **One variable changes**, so a comparison
isolates it. A test asserts that both drivers emit an identical action sequence
on a keyboard-only game.

It is deliberately *not* an information-seeking driver: it does not prefer
untried targets, avoid objects known to be inert, or merge targets with
identical outcomes. All three are what D-021 decision 2 argues for and all three
are what D-020 deferred until there is a model to seek information for.

**One finding fell out of writing the test.** `object_targets` **includes the
background**: `colour_4`, `colour_8`, `colour_mass` and `bbox` all partition the
entire frame, so the background region is a component like any other and
contributes its own click target, usually near the middle of the frame. The
first test asserting "clicks land inside objects" failed on pixel (32,32). This
is correct behaviour for D-005, where dropping a live target costs more than
carrying a dead one, and it partly explains why D-021's proposal counts are as
large as they are. A driver spending real actions wants
`foreground_object_targets`, which excludes the background colour and falls back
to the full set rather than hard-excluding anything (Guardrail 1).

### Experiment 1 — the old proxy metrics say nothing happened

- **Setup:** `scripts/eval_drivers.py --actions 60 --seed 0`, all 25 games, now
  with four drivers.
- **Result:**

  | driver | no-op % | mean changed px | distinct events/game | coverage % |
  |---|---|---|---|---|
  | `random` | 20.3 | 51 | 40.1 | 99 |
  | `random_objects` | 20.5 | 40 | **40.7** | 99 |
  | `round_robin` | 19.3 | 51 | 36.9 | 100 |
  | `lattice` | 19.5 | 52 | 36.2 | 100 |

- **Reading: indistinguishable, and that is the second time these metrics have
  been blind to something real.** D-021 already showed that distinct
  changed-pixel signatures cannot see outcome-class structure — a driver that
  reaches one outcome repeatedly still accumulates signature diversity from the
  states it passes through. Taking this table at face value would repeat exactly
  the error D-020 made. It is reported because it is the like-for-like
  comparison with D-020, not because it settles anything.

### Experiment 2 — the metric that matches what the learner needs

- **Question:** can a corpus teach a click rule *at all*? Independent of whether
  induction works: how often is the same object clicked enough times for MDL to
  write a rule about it?
- **Setup:** `scripts/eval_click_evidence.py --corpus corpus --compare
  corpus_obj`, both corpora 25 games x 100 actions, seed 0, `colour_8`. An
  object's identity is the attribute vector the rule language selects on
  (colour, size, shape, rect), so "hit twice" means hit twice *in the sense the
  inducer cares about*. A signature needs **3** hits to count as learnable.
- **Result:**

  | | `random` | `random_objects` |
  |---|---|---|
  | clicks landing on a repeated object | 65.2% | **75.0%** |
  | learnable signatures per game | 2.4 | **4.6** |
  | learnable **live** signatures per game | 1.5 | **3.2** |
  | **games with ≥1 learnable live signature** | **10/19** | **16/19** |

- **Reading: the confound was real and is largely removed.** Six more games now
  have at least one click rule that is learnable in principle. The per-game
  gains are concentrated where D-021 predicted: s5i5 2→12 learnable live
  signatures, r11l 7→12, vc33 3→9, sc25 0→3, su15 0→3, cd82 0→1, sp80 0→1.

- **Two honest costs.**
  - **bp35 goes 4→1.** It has 196 object targets, the most of any game, so
    aiming at objects *spreads* clicks across many of them and reduces repeat
    hits. Object targeting diversifies, and on an object-rich game diversity is
    the enemy of the repetition `clicked` needs. This is a genuine trade, not
    noise, and it predicts that the best click driver is object-targeted *with
    repetition*, which neither driver does.
  - **lp85 stays at 0** under both. With 2.3% of pixels live and 97% no-ops it
    is starved past the point any targeting policy fixes, confirming D-020's
    separate reporting of it.
- **Cost:** recording rises from 15 s to 157 s for 25 games x 100 actions,
  because segmentation now runs on every click. Irrelevant offline; worth noting
  because it is per-action work an agent would also pay at evaluation.

### Experiment 3 — does the better corpus produce better dynamics?

- **Setup:** `scripts/eval_dynamics.py --corpus corpus_obj --carvings
  colour_4,foreground_4`, the two carvings that won games in D-025, 25 games,
  seed 0, same 70/30 temporal split. 464 s. Compared against D-025's numbers.
- **A caveat that has to come first.** The two sweeps run on **different
  recordings**, so both the training *and* the test transitions differ. A
  per-game change in exact match therefore conflates "better training evidence"
  with "a differently hard test set", and with only ~30 test transitions per
  game that is noisy. Changed-object accuracy averages over hundreds of objects
  rather than 30 binary outcomes and is the more trustworthy of the two.
- **Result, aggregate:**

  | carving | mean exact | exact>0 | click-game mean exact | click-game mean chgobj |
  |---|---|---|---|---|
  | colour_4 | 4.8% → **4.0%** | 5/23 → 5/23 | 4.4% → **3.3%** | 30.5% → **37.3%** |
  | foreground_4 | 1.8% → **2.1%** | 3/23 → 2/23 | 1.7% → **2.1%** | 31.1% → **32.6%** |

- **Result — the object model improves and the headline does not.** On the click
  games, changed-object accuracy rises by **+6.8pp** under `colour_4` and
  +1.5pp under `foreground_4`, while exact-frame match is flat to slightly down
  and well inside the noise of a 30-transition test set. The corpus fix did the
  thing it was designed to do — the learner now gets the click evidence it was
  missing, and its model of what objects do got better — and that did not
  convert into whole-frame predictions.

- **The standout, and it is a third instance of the same pattern.** **tn36 goes
  from 0.0% to 100.0% changed-object accuracy** and stays at **0.0% exact
  match.** Its object model went from useless to perfect and its score did not
  move. dc22 does a smaller version of the same thing: 91.7% → 100%
  changed-object accuracy, exact match 44.4% → 34.8%. With sp80 and ls20 from
  D-025, that is now **four games where the model of the game is right or nearly
  right and the frame comparison scores zero**.

  cn04 is the one game where the fix converts end to end: 0% → 3.6% exact under
  `colour_4`, and **21.1% → 35.7%** under `foreground_4`, with changed-object
  accuracy 47% → 74%.

- **Did it change our mind? Yes, and in the direction that was predicted before
  the run rather than after it.** The prediction on record was that the click
  games would move only modestly, for three reasons already measured: lp85 is
  starved past rescue, five games have a 100% deformed-move rate under `colour_*`
  so the language cannot express their changes whatever the corpus, and the UI
  residual is untouched by anything here. That is what happened.

  So the confound is now resolved **in both directions**. D-025's click-game
  zeros were not mainly a corpus artefact — the corpus was genuinely broken, it
  is now genuinely fixed, and the zeros largely remain. D-025's original
  attribution to the rule language was right, and it was right for a reason it
  had not established. It has now been established.

- **Did it change our mind? Yes.** D-025's claim that the click-only games'
  zeros are evidence about the rule language was **unsupported when it was
  made** — the corpus could not have taught those rules. Experiment 3 then
  re-establishes the claim on a corpus that can. The conclusion is unchanged and
  the evidence for it is now real, which is not the same thing as the original
  reasoning having been sound.

- **Decision:**
  1. **`random_objects` becomes the default corpus driver** for any game that
     publishes the click, on Experiment 2. `random` is retained as the named
     baseline, as are `round_robin` and `lattice`.
  2. **`foreground_object_targets` is what a driver aims at**; `object_targets`
     keeps its background-inclusive behaviour for D-005's pruning, where the
     asymmetry runs the other way.
  3. **D-025's headline stands, now on better evidence.** Dynamics induce
     weakly; the click games' zeros survive a corpus that can teach click rules.
     Its per-game click numbers are superseded by
     `docs/experiments/dynamics_sweep_objdriver.json`.
  4. **The bookkeeping layer is reinforced as the next subsystem**, not merely
     confirmed. There are now four games (sp80, ls20, tn36, dc22) whose object
     model is right or nearly right while the frame comparison scores at or near
     zero, and tn36 is the sharpest: 100% changed-object accuracy, 0% exact.
- **Options rejected:**
  - *Leave the corpus alone and treat D-025 as final* — the cheapest option and
    the wrong one. A confound identified and not measured is worse than one
    never noticed, because the record then implies it was checked.
  - *Adopt `round_robin[objects]`, which already aimed at objects* — it pairs
    object targeting with cyclic action selection, and D-020 measured cyclic
    selection as worse. Combining a winner with a loser and calling it an
    improvement is how the previous confound happened.
  - *Build an information-seeking driver now* (prefer untried targets, drop
    inert ones, merge equivalent outcomes) — it would almost certainly beat
    this, and it changes two things at once. D-021's outcome-class merging is
    the natural home for it and it belongs there, after this comparison is
    clean.
  - *Re-record every corpus and re-run every prior measurement* — the perception
    numbers (D-014, D-015) were measured under `random` and are unaffected by
    click placement in any way that has been shown. Re-running 2.2 hours of
    sweep to restate numbers about tracking would be motion, not progress.
- **Consequences:** the click-game half of the dynamics result becomes
  interpretable for the first time. Commits us to a driver whose cost scales with
  segmentation, and to the finding that the *best* click driver is
  object-targeted with deliberate repetition — which neither existing driver
  does and which is now a named gap.
- **Revisit if:** bp35's regression turns out to be the common case on
  object-rich games rather than the exception, which would mean object targeting
  needs a repetition policy before it is safe as a default — the specific
  driver that would settle it is object-targeted *with deliberate repetition*,
  which does not exist; or a like-for-like comparison on a **shared test set**
  contradicts Experiment 3's read, which is worth doing if the click games ever
  become the critical path, since the two sweeps here differ in both train and
  test data.

---

## D-027 — The target architecture: online, hypothesis-driven, discovery-minimising

- **Date:** 2026-09-24
- **Status:** accepted
- **Context:** Everything built so far collects a corpus by random play, induces
  a model in one batch afterwards, and never acts on what it learned. That was
  right for a go/no-go — it isolates the question "can dynamics be induced at
  all" from "can an agent use them" — and D-025 answered it. It is **not** the
  agent, and continuing to optimise it would be optimising the instrument.

  The direction was set by the project owner, and it is a sharper statement than
  design thesis §8 or D-022 had reached:

  > Form hypotheses from single actions. Subsequent actions are taken based on
  > hypotheses — whatever will advance towards the goal given current beliefs.
  > Hypotheses are updated from the observation that comes back. Minimise the
  > amount of pure discovery actions.

  Three commitments follow, and each contradicts something currently in the code.

### 1. A single observation must be able to create a hypothesis

The present learner cannot do this, and the reason is structural rather than
accidental. MDL charges roughly the same bits to state a rule as to record one
exception, so **one** observation of "clicking this made it vanish" is never
worth a rule and is always cheaper to store as an exception. `LEARNABLE_HITS` in
`scripts/eval_click_evidence.py` is set to 3 for exactly this reason, and D-026
measured the corpus against that bar.

A human clicking something once and watching it disappear does not withhold
judgement pending a second trial. They form *"clicking things like this makes
them vanish"* — a hypothesis carrying a prior, not a certainty, immediately
actionable and immediately falsifiable.

That is the weighted hypothesis population Guardrail 1 and D-003 have required
since before any code existed, and which D-024 knowingly suspended for the
go/no-go. The suspension is now lifted: **single-observation generalisation is a
requirement, not a nice-to-have**, and MDL over a point hypothesis is the wrong
machinery for it.

**This inverts a conclusion from D-026.** That entry found the corpus needs to
hit the same object repeatedly for a click rule to be learnable, and named an
object-targeted driver *with deliberate repetition* as the missing piece. If the
learner generalises from one observation, repetition is waste and the right
driver is the one that touches the **most distinct objects** — which is what
`random_objects` already does. The repetition driver is therefore **not built**,
and D-026's "known gap" is withdrawn: it was a workaround for a limitation we
are removing instead.

### 2. Acting and learning interleave; discovery is a cost to minimise

Currently: collect 100 random actions, then learn. Target: every action is
chosen for what it is expected to achieve *and* what it is expected to reveal,
and an action taken purely to find something out is a cost to be minimised
rather than a phase to be completed.

D-022 derived a two-phase value function — information gain while the model is
uncertain, goal-directed once a goal hypothesis has support — with an
RHAE-derived budget deciding the switch. **That framing is superseded here.**
The phases are not sequential. A good action advances the goal *and* discriminates
between live hypotheses, and pure discovery is what you do only when nothing
else does both. D-022's cost model (revert costs *k*+1, information is *k* times
dearer at depth) survives intact and is what prices the trade.

The measured fact that makes this affordable: dynamics learning **saturates at
about 50 actions** of random play (D-025). The model does not need a long
discovery phase; it needs the right fifty observations, and a hypothesis-driven
policy should need fewer.

### 3. The frame is a composite and must be modelled as one

`benchmark.md` correction #2 established that a 64x64 frame is a camera view,
integer-upscaled and letterboxed, with a UI layer drawn last at full resolution.
Every model built so far treats it as a single homogeneous surface, and is then
scored against both layers at once.

The cost of that conflation is now measured rather than suspected. **Six of 23
games reach >=90% changed-object accuracy under some carving; three of them
score exactly 0%** — s5i5, sp80 and tn36 — with median residuals of 1, 2 and 1
pixels. sp80 was traced precisely: **row 0 of the frame** changes from colour 14
to colour 0 on nearly every action and the model predicts no change, which fails
the whole-frame comparison every single time. Row 0 is a status strip. It is not
the game.

- **Decision:** adopt the three commitments above as the architecture to build
  towards. Concretely, and in the order the evidence supports:
  1. **Carving selection by held-out validation** (D-028) — cheapest, and it
     recovers a gap that is already measured.
  2. **Layer separation**, modelling UI and playfield separately.
  3. **A weighted hypothesis population with priors**, replacing MDL-over-a-point-
     hypothesis, enabling single-observation generalisation.
  4. **An online loop** that selects actions by expected goal progress plus
     expected discrimination between live hypotheses.
- **Options rejected:**
  - *Keep improving the batch pipeline* — it is a research instrument and it has
    now answered the question it was built for. D-026 spent a session making the
    corpus better and moved the headline not at all, which is the clearest
    possible signal that the corpus is no longer the constraint.
  - *Build the repetition driver as planned* — see above; it optimises for a
    learner limitation we are removing, and would make the corpus worse for the
    learner we want.
  - *Go straight to the online loop and skip 1-3* — tempting with 39 days left,
    and wrong. An online agent acting on a model that cannot predict UI pixels,
    cannot choose its own representation, and cannot learn from one observation
    would fail for reasons we could not separate. Each of 1-3 removes a named,
    measured blocker.
  - *Abandon symbolic induction for a learned transition model* — still the
    fallback if expressiveness rather than machinery turns out to be the wall
    (D-024 records the reasoning). Nothing since has made it more attractive: the
    data budget is ~50 actions per game.
- **Consequences:** the corpus/batch pipeline becomes a measurement harness
  rather than the thing being built, and its numbers stay comparable only as
  long as it is not changed underneath them. Commits us to replacing the MDL
  point-hypothesis learner, which is most of `dynamics/induce.py`. Supersedes
  D-022's sequential two-phase value function and withdraws D-026's repetition
  driver.
- **Revisit if:** single-observation hypotheses turn out to produce so many
  candidates that the population cannot be maintained or searched, in which case
  some evidence threshold has to come back and the repetition driver returns
  with it; or the online loop cannot be made to work at all inside the remaining
  time, in which case the batch pipeline plus a greedy policy is the fallback
  submission.

---

## D-028 — Decomposing the zeros: a third of the games never move the playfield

- **Date:** 2026-09-24
- **Status:** accepted
- **Context:** D-025 left 16 of 23 games scoring exactly zero and attributed it,
  loosely, to the rule language. D-027 then set an agenda — carving selection,
  then layer separation, then a hypothesis population — on the assumption that
  the zeros were a *modelling* failure. Before spending sessions on that
  ordering it was worth asking what the zeros are actually made of.

  The decomposition needs one new quantity: the **playfield activity rate**, the
  fraction of an episode's actions that change at least one pixel *outside* the
  detected bookkeeping band. It separates "the agent did something and we failed
  to predict it" from "the agent did nothing but tick a counter".

### Measurement

- **Setup:** all 25 games, 100-action random corpus, seed 0, bookkeeping band
  from `bookkeeping_mask` fitted on the training split only.
- **Result:** **8 of 25 games move the playfield on fewer than 15% of actions.**

  | game | actions changing *anything* | actions changing the **playfield** |
  |---|---|---|
  | vc33 | 99/99 | **0** |
  | lf52 | 99/99 | **2** |
  | s5i5 | 99/99 | **2** |
  | sb26 | 41/100 | **3** |
  | su15 | 42/99 | **3** |
  | tn36 | 99/99 | **4** |
  | lp85 | 1/100 | **1** |
  | ft09 | 7/100 | **7** |

  vc33 is the extreme and the clearest: **99 transitions in which the frame
  changed, and not one of them altered a single pixel outside the UI band.**
  Every one of those changes was a counter ticking.

- **All eight score 0% exact.** They are, therefore, half of the sixteen zeros,
  and they are not a modelling failure at all — there is nothing in their corpus
  to model.

### Two corrections to claims made earlier today

1. **"Three games are one or two pixels from scoring" was wrong for two of the
   three.** s5i5 and tn36 reach 100% changed-object accuracy under
   `foreground_4` with a median residual of 1 pixel, and I read that as a
   near-miss caused by the UI. It is not. Their object model is a **perfect
   model of the counter**, because the counter is essentially the only thing
   that ever moves. Only **sp80** is genuinely UI-blocked, and it is: 56 of its
   96 changed transitions alter the playfield, its object model is right, and
   excluding the band takes it from **0% to 100%**.

2. **D-021's click map conflated UI changes with playfield effects.** It scored
   a pixel as "live" and counted distinct outcomes by hashing the **whole
   frame**, so a click that only ticked the counter registered as a live pixel
   producing a distinct outcome. vc33 is the proof: D-021 recorded it as 100%
   live with 3 distinct outcomes, and the playfield activity rate is **zero**.
   Its click outcomes are counter states.

   This does not overturn D-021's *decision* — object targets still reach
   whatever outcomes exist, and the branching factor is still small — but its
   "live %" and "distinct outcomes" figures are **upper bounds on playfield
   effect, not measures of it**, and any future use must recompute them outside
   the bookkeeping band.

### What layer separation is actually worth

Measured, not estimated, from the existing sweep by excluding the detected band:

| game | exact | excluding the UI band |
|---|---|---|
| sp80 | 0.0% | **100.0%** |
| dc22 | 44.4% | **100.0%** |
| m0r0 | 26.1% | 44.4% |
| re86 | 0.0% | 6.7% |

Two games go to a perfect score and one improves substantially. Real, and
bounded: it rescues **two** games, not the eight the zeros might have suggested.

### The revised decomposition of the sixteen zeros

- **8 games — playfield-starved.** Random play never makes the game do anything.
  An **exploration** failure, not a learning one.
- **~1 game — UI-blocked** (sp80; dc22 and m0r0 are already scoring and merely
  improve). A **representation** failure, and the fix is proven.
- **~7 games — genuinely unmodelled.** Language, interaction, appearance.

- **Did it change our mind? Yes, and it reorders D-027's agenda.** The largest
  single category of failure is that **the agent never makes anything happen**.
  That is worth up to eight games. Carving selection is worth about one and
  layer separation about two. The ordering set this morning had the cheapest
  items first; it now has the *smallest* items first.

  It also converges with the direction the project owner set independently in
  D-027 — form hypotheses from single actions, act on them, minimise pure
  discovery. On a game like vc33 the problem is not that we learn badly from
  what we see; it is that random play shows us nothing to learn from. No
  learner, however good, recovers from a corpus in which the game never moves.

- **Decision:**
  1. **Promote exploration to objective 1**, ahead of carving selection and
     layer separation. The target is the playfield activity rate: an action that
     only ticks the counter is a wasted action, and that is now measurable
     *while playing* rather than only in hindsight.
  2. **Keep layer separation, at its measured value.** It converts sp80 and
     dc22 and it makes the activity rate computable, which objective 1 needs.
     It is a prerequisite, not a competitor.
  3. **Demote carving selection** to whatever its own experiment says it is
     worth; it was never going to be more than a game or two.
  4. **Recompute D-021's click-map figures outside the bookkeeping band** before
     anything relies on them again.
- **Options rejected:**
  - *Treat the starved games as unwinnable and drop them* — eight of 25 is a
    third of the public set and there is no reason to think the private set is
    kinder. Also premature: nothing has yet tried to make them move on purpose.
  - *Blame the driver and build a better one in the same style* — D-026 already
    ran that experiment. Object targeting improved click *evidence* markedly and
    moved the headline not at all, and vc33 was object-targeted in `corpus_obj`
    too. A better blind policy is not the answer; acting on a hypothesis is.
  - *Report the eight as a corpus artefact and carry on with modelling* — the
    honest reading is the opposite. A third of the games not responding to
    random play is a fact about **what these games require of an agent**, and it
    is probably the single most competition-relevant thing measured so far.
- **Consequences:** reorders the agenda around exploration. Makes the playfield
  activity rate a first-class metric, which means the bookkeeping band moves
  from a scoring diagnostic to a component the agent depends on. Invalidates the
  unqualified use of D-021's live-pixel and outcome-count figures.
- **Revisit if:** the eight starved games turn out to need a long specific
  action sequence before anything moves at all — which would make them an
  exploration problem of a much harder kind than "aim better", and would put
  them beyond reach inside the remaining time; or layer separation proves unable
  to compute a reliable activity rate online, in which case objective 1 loses
  the signal it is supposed to optimise.

---

## D-029 — Carving selection does not work at this corpus size

- **Date:** 2026-09-24
- **Status:** accepted
- **Context:** D-025 found the best carving is game-dependent — any single
  carving scores on 5 of 23 games, best-per-game on 7 — and `architecture.md`
  section 3.4 has recorded "nothing selects among the six" since session 2 as
  though it were an open research problem. It is not. **Held-out validation is
  the standard answer and had simply never been applied.** D-027 made it
  objective 1 on the grounds that it was the cheapest item with measured
  headroom.

  `scripts/eval_carving_selection.py`: split the training transitions again,
  induce under all six carvings on the inner split, keep whichever wins on the
  held-out validation split, then induce that carving on the full training set
  and score on the untouched test set. Reported against two reference points —
  the best fixed carving (what no selection gets) and an oracle that picks per
  game by looking at the test score (the ceiling).

  **The headroom is small to begin with:** 4.8% fixed against 6.1% oracle, a gap
  of 1.2 percentage points and two games.

### Experiment — four selection keys, then a check on the check

- **Setup:** 25 games, 100-action random corpus, seed 0, 70/30 train/test with a
  further 75/25 inner split. ~52 min per run. The script records every candidate
  key on the validation split so rules can be compared offline without re-running.
- **First attempt, selecting on validation exact match:** recovered **8%** of the
  headroom. Diagnosis was immediate and is the useful part: **exact match is zero
  under every carving on most games** at ~20 validation transitions, so there is
  nothing to rank and the tie-break does the work.
- **What a selection key has to be**, written down before re-running: *dense*,
  so ~20 transitions carry signal; and *carving-independent*, or the comparison
  is rigged. The second rules out changed-object accuracy, which is measured in
  terms of the carving being judged — a carving producing two big objects scores
  easily where one producing fifty cannot. That is exactly D-019's objection to
  object-level scoring as a headline and it applies with full force to using it
  as a selector.
- **Result, all four keys:**

  | selection key | mean exact | games >0 | headroom recovered |
  |---|---|---|---|
  | fixed `colour_4`, no selection | 4.8% | 5/23 | — |
  | validation exact match | 4.6% | 5/23 | **−18%** |
  | validation median residual | 4.7% | 6/23 | **−6%** |
  | validation changed-object accuracy | 4.9% | 6/23 | +8% |
  | **validation mean residual** | **5.9%** | **7/23** | **+86%** |
  | oracle (cheating) | 6.1% | 7/23 | 100% |

  Mean residual matching the oracle's game count looked like a clean win, and
  there was an a-priori story for it: median residual saturates at small
  integers and ties constantly, while the mean is continuous and sensitive to
  the catastrophic misses that actually distinguish a bad carving.

- **Then the check that mattered.** That comparison chose among four rules *by
  looking at the test set*, which is meta-level overfitting with 23 games and
  four candidates. Re-run honestly — choose the rule on a random half of the
  games, evaluate it on the other half, 200 splits:

  | | |
  |---|---|
  | rule chosen on the unseen half | `meanres` 103/200, `chgobj` 51, `exact` 46 |
  | mean headroom recovered on unseen games | **−42%** (median +46%) |
  | splits where selection beat fixed `colour_4` | **51%** |

- **Result: a coin flip.** Selection beats doing nothing on 51% of splits, the
  mean headroom is negative because the misses are catastrophic when they
  happen, and the winning *rule* is not even stable across splits. The +86% was
  an artefact of choosing the rule on the test set.

- **Did it change our mind? Yes, twice, and the second time reversed the first.**
  The initial 8% looked like "weak signal, try a better key". The better key
  looked like a decisive fix. Cross-validation showed the fix was an illusion.
  Without the game-holdout check this entry would have recorded an 86% success
  and the pipeline would have adopted a selector that is a coin flip.

- **Decision:** **Do not select carvings. Use fixed `colour_4`.** D-027's
  objective 1 is closed as a negative result, and D-028 had already demoted it
  on separate grounds — it was worth about one game where exploration is worth
  eight.
- **Options rejected:**
  - *Adopt mean-residual selection on the 86% figure* — it does not survive a
    holdout over games, and adopting it would have been exactly the error D-019
    guarded against when it refused to pick a metric after seeing which one
    flattered the result.
  - *Ensemble the six carvings instead of selecting one* — plausible, and not
    tried. It multiplies inference cost sixfold for a ceiling of 1.2pp, which is
    a bad trade at this stage. Worth revisiting only if the ceiling rises.
  - *Conclude that per-game carving choice is worthless* — too strong, and the
    oracle contradicts it: 7 games against 5 is real, and s5i5 and tn36 swing
    from 0% to 100% object accuracy on carving alone. The choice matters; we
    cannot currently *make* it from 20 validation transitions.
  - *Fix it by growing the corpus* — the learning curve saturates at n=50
    (D-025), so a bigger corpus improves the model not at all; it would only
    improve the validation signal. That is a real option and a narrow one, and
    it costs a 300-action recording plus another hour of sweep for at most
    1.2pp.
- **Consequences:** the pipeline stays on one fixed carving, so the six-fold
  sweep cost is a research expense rather than a runtime one. `architecture.md`
  section 3.4's "nothing selects among the six" is now answered: something
  could, but not reliably at this data scale.
- **Revisit if:** layer separation lands (D-028 objective 2) and residuals can
  be measured on the **playfield alone** — the current residual signal is
  contaminated by the bookkeeping band, which is a large and carving-independent
  chunk of every measurement, and cleaning it is the single most likely way to
  make this signal work; or the corpus grows enough that validation carries
  signal; or the exact-match ceiling rises far enough that 1.2pp of headroom
  becomes worth the sixfold cost.

---

## D-030 — The starved games, and a metric that was scoring blindness as success

- **Date:** 2026-09-24
- **Status:** accepted
- **Context:** D-028 made exploration objective 1 on the strength of eight games
  that move the playfield on fewer than 15% of actions. Before building an
  exploration policy it was worth asking the question that decides what kind of
  policy is needed: **is there anything at the start state that moves these
  games at all?**

  If yes, random play missed a move that exists — a *targeting* problem, cheap
  to fix. If no, the game needs a specific sequence before anything responds — a
  *depth* problem, and the condition D-028 named as putting these games beyond
  reach in the remaining time.

  `scripts/eval_playfield_map.py` answers it by brute force: probe the action
  space from a fixed state, RESET between probes, and count only changes
  **outside the bookkeeping band**. That last clause also discharges an action
  item D-028 left open — D-021's click map hashed the whole frame, so a click
  that merely ticked a counter registered as a live pixel producing a distinct
  outcome.

### Finding 1 — D-028's starvation figures were measured on a superseded corpus

This is an error in D-028 and it is mine. D-026 had already made
`random_objects` the default driver, and D-028's activity rates were computed on
`corpus/`, recorded with the old uniform-random-pixel driver. Recomputed on
`corpus_obj/`:

| game | `random` | `random_objects` |
|---|---|---|
| s5i5 | 2% | **41%** |
| vc33 | **0%** | **16%** |
| ft09 | 7% | 14% |
| su15 | 3% | 13% |
| tn36 | 4% | 11% |
| sb26 | 3% | 7% |
| lf52 | 2% | 5% |
| lp85 | 1% | 3% |

Aiming clicks at objects multiplies playfield activity by **2× to 20×** on every
one of the eight. vc33 goes from literally nothing to 16%.

**This strengthens D-028's conclusion rather than weakening it.** Its claim was
that the largest category of failure is the agent never making anything happen,
and that better acting is the lever. Here is that lever moving, from a change of
*targeting alone*, before any hypothesis-driven policy exists.

### Finding 2 — the starved games are state-gated, not inert

Probing 256 click positions plus every keyboard action, outside the band:

| game | depth 0 | depth 5 |
|---|---|---|
| vc33 | **2 of 256** | **254 of 256** |
| tn36 | 6 of 256 | 6 of 256 |

vc33 is almost completely unresponsive at the level start and almost completely
responsive five actions later. So these are **targeting-and-state** problems,
not depth-in-principle problems: the moves exist, and a policy that notices when
the game has become responsive would find them. D-028's pessimistic branch — "no
single action responds, a sequence is needed" — did not fire. This is the same
state-gating D-021 measured for click *liveness* on eight of nineteen games,
now confirmed for playfield effect specifically.

### Finding 3 — all eight still score 0%, and chasing why broke a metric

Even on the improved corpus every one of the eight scores 0% exact. Two of them,
s5i5 and tn36, do so while reporting **100% changed-object accuracy, 0%
deformed moves, zero appearances** and a median residual of 1–2 pixels *outside*
the bookkeeping band. That combination should be impossible: a perfect object
model with no unrepresentable events should reconstruct the frame.

Tracing s5i5's residual pixels: they are pixels going 11→5, 11→13, 13→11 inside
a component whose mask, size, bounding box and dominant colour never changed.
**Partial recolouring.**

The vocabulary cannot see it. `_effect_from_match` decided "same position, same
silhouette, same dominant colour" and returned `static`. The model also
predicted `static`. The two agreed, and `changed_obj_acc` scored the pair
**correct** — while the rendered frame was wrong.

**So `changed_obj_acc` was never measuring model quality. It measures agreement
with a lossy observation, and a carving that cannot see a change cannot be
scored wrong for it.** `foreground_*` carvings make this acute, because there a
component is "anything not background" and its internal colouring is not part of
its identity.

Measured across 23 games on `corpus_obj`, four carvings, seed 0:

| carving | changed-object acc | **blind rate** | exact |
|---|---|---|---|
| `colour_4` | 41.4% | **2.8%** | 4.0% |
| `bbox` | 44.1% | 3.1% | 3.6% |
| `colour_mass` | 33.4% | 7.7% | 3.5% |
| **`foreground_4`** | 29.9% | **24.7%** | **2.1%** |

`foreground_4` is blind to **a quarter of every object change it observes**,
against under 3% for the colour carvings.

**This resolves the paradox D-025 recorded and could not account for, though not
in the way this entry first guessed.** The draft of this section asserted that
`foreground_*` wins object accuracy by being blind. It does not — it has the
*lowest* object accuracy of the four, 29.9% against 41.4%. The real mechanism is
better and simpler:

> `foreground_*`'s famously low deformed-move rate — 14% against 52% for
> `colour_4`, which D-025 flagged as its one clear advantage and this session
> spent an hour hypothesising about — **is blindness, not cleanliness.** A change
> the carving cannot see is not recorded as a deformed move. It is not recorded
> as a move at all. The deformation did not go away; it stopped being visible,
> and went on to be unpredicted in the frame.

That is why D-025's central puzzle had no answer: it asked why a threefold
reduction in unrepresentable moves produced *half* the exact-match score, and
the answer is that the reduction was not real. `foreground_*` trades 24.7%
blindness for a low deformed-move rate and pays for it in every whole-frame
comparison.

- **Did it change our mind? Yes, four times, and three times against my own
  earlier claims in this session.** The starvation figures were stale. The "perfect
  object model" reading of s5i5 and tn36 — which I had already corrected once
  today, from "one pixel from scoring" to "starved" — was wrong a second time,
  and for a third reason: the model is not perfect, the observation is blind.
  And the first draft of this very entry explained the blindness backwards,
  asserting that `foreground_*` buys object accuracy by being blind when it has
  the lowest object accuracy of any carving. Each correction was found by
  insisting on a mechanism for a number rather than accepting the number, and
  the last one was caught only because the mechanism was measured instead of
  asserted.

- **Decision:**
  0. **`foreground_*` carvings are not preferred on their low deformed-move
     rate**, which is an artefact. Nothing currently selects carvings (D-029), so
     this changes no behaviour today; it removes a false argument that would
     otherwise have driven the choice when something does.
  1. **Count and report blindness.** `observe_transition` now compares pixel
     content directly when position and silhouette are unchanged, and reports
     `n_blind`; `Score.blind_rate` exposes it. The effect returned is still
     `static`, because the language genuinely cannot express a partial recolour
     — but the blindness is reported rather than scored as success.
  2. **`changed_obj_acc` is never to be quoted without `blind_rate`**, and never
     compared across carvings. D-019 refused object-level scoring as a headline
     because it is carving-dependent; this is the concrete mechanism, and it is
     worse than carving-dependence — it is a metric a representation can win by
     being less expressive.
  3. **Exploration stays objective 1**, now with direct evidence for it: a pure
     targeting change moved playfield activity 2–20× on every starved game.
  4. **The state-gating result feeds the exploration policy directly.** A policy
     should detect that a game has become responsive and exploit it, rather than
     sampling uniformly over a level where the response rate varies from 0.8% to
     99% depending on where it is standing.
- **Options rejected:**
  - *Add a partial-recolour effect to the vocabulary now* — the obvious fix, and
    premature. It is one more effect on top of a language already known to be
    missing interaction, appearance and memory, and D-027 decided the learner is
    being replaced rather than extended. Worth doing *inside* the new learner.
  - *Drop `changed_obj_acc`* — it is still the only dense per-object signal
    available and D-025 used it correctly as a diagnostic. Reporting it beside
    `blind_rate` preserves its use and removes its ability to mislead.
  - *Re-run every prior measurement with the blindness correction* — the
    headline metric is exact-frame match and is unaffected; only the object-level
    diagnostic moves. The affected claims are identified in this entry rather
    than recomputed at a cost of hours.
  - *Treat the stale-corpus error by quietly recomputing D-028* — the log is
    append-only and the error is instructive: a measurement was made against the
    old default two entries after the default changed.
- **Consequences:** object-level scoring now carries a companion figure that can
  invalidate it. Exploration keeps priority with better evidence. The explanation
  for D-025's foreground paradox closes a question that has been open since the
  sweep.
- **Revisit if:** the new learner represents partial recolouring, at which point
  the blindness disappears rather than being reported and `foreground_*` may
  become genuinely competitive; or blindness turns out to behave differently on
  the private set, where sprite colouring conventions need not match the public
  games. Note the measured answer to the obvious first question: blindness is
  **rare under `colour_*`** (2.8%) and `bbox` (3.1%), so colour-based object
  accuracy can be read close to face value and it is specifically the
  background-versus-foreground carvings that are compromised.

---

## D-031 — An exploration policy, and the first capability this project has gained

- **Date:** 2026-09-24
- **Status:** accepted
- **Context:** D-028 decomposed the sixteen zero-scoring games and found that
  half are not a modelling failure — the agent never makes the game do anything
  — making exploration objective 1, worth up to eight games where carving
  selection was worth about one (and turned out to be worth none, D-029). D-030
  then showed the moves are there to be found: vc33 responds to 2 of 256 probes
  at the level start and **254 of 256** five actions later.

  `src/arc3/explore.py`: Thompson sampling over a Beta posterior on *"does this
  choice move the playfield"*. The first component in the repository that learns
  from what comes back and changes what it does next.

### The three design points, each forced by a measurement

1. **Count playfield changes, not frame changes.** vc33 changed its frame on 99
   of 99 actions and its playfield on **zero** (D-028); a policy rewarded for
   frame change would have been perfectly satisfied by a game that never
   responds. `OnlineBand` estimates the bookkeeping band incrementally — the
   online counterpart of `bookkeeping_mask`, same rule and thresholds — and is
   deliberately empty until warmup, so it is under-confident early rather than
   wrong early. **This is D-028's objective 2, layer separation, arriving as a
   component rather than a scoring trick.**
2. **Arms are object signatures, not pixels.** Objects move, so a pixel that
   paid off last turn means nothing this turn. An arm is the attribute vector
   the rule language already selects on, so experience transfers as objects move
   and explorer and learner look at the same features.
3. **Evidence decays.** Responsiveness is state-dependent (D-021 on eight of
   nineteen games, D-030 for vc33). A stationary estimate would learn "clicking
   does nothing" at the level start and never revise it.

Guardrail 1 holds by construction: Thompson sampling never gives an arm zero
probability, and `_Arm.decay` pulls towards the prior rather than past it, so an
arm that has failed twenty times is heavily down-weighted and still occasionally
retried — exactly what a state-gated game requires.

### Experiment — playfield activity, 25 games, 100 actions, seed 0

`scripts/eval_exploration.py`. The band is computed **per driver from its own
episode**, since an agent has no access to anyone else's recording.

| | `random` | `random_objects` | **`explore`** |
|---|---|---|---|
| all 25 games | 49% | 55% | **63%** |
| the starved eight | **3%** | 14% | **47%** |
| **games still under 15% activity** | **8** | 6 | **1** |
| levels cleared | 1 | 1 | **4** |

Per game, the gains are where D-028 said the headroom was: vc33 0%→**100%**,
lp85 1%→**77%**, ft09 7%→**52%**, s5i5 2%→46%, sb26 3%→37%, tn36 4%→36%, lf52
2%→19%. **lp85 was described two sessions ago as "starved past rescue" and
needing its own investigation.** It needed a policy that notices what works.

**Levels cleared, checked rather than quoted.** r11l is cleared by all three
drivers and is not the explorer's doing. ft09, lp85 and tn36 are cleared **only**
by `explore`. So three levels are genuinely attributable — the first time
anything in this project has cleared a level another policy could not. They are
still *incidental*: the policy optimises "something happened", not winning, and
cleared them while poking a game that had become responsive.

**The honest cost.** `explore` is worse than `random` on several already-active
games: cd82 39%→28%, dc22 59%→34%, sk48 69%→52%, g50t 81%→75%, ar25 91%→85%.
A policy spending budget on posterior estimation loses to uniform sampling where
uniform sampling already works.

### Two rejected variants, both measured at equal budget

The failure being chased was that the explorer was *less active than uniform* on
object-rich games — cd82 at 20% against `random_objects`' 79%, which a policy
maximising activity should not manage.

| | activity | distinct events |
|---|---|---|
| `random_objects` | 38% | 27.4 |
| `explore`, union of all six carvings | 45% | 25.2 |
| `explore`, `(colour, size)` arms | 58% | 35.8 |
| **`explore`, `colour` arms** | **58%** | **44.9** |

- **The union of carvings made it worse**, against the expectation from D-021
  that the union is what reaches every distinct click outcome. It roughly
  doubled the arm count (tn36 136 arms to 288) and Thompson sampling then spends
  its entire budget establishing posteriors it never exploits. Kept as
  `union_targets`.
- **The binding constraint is arms against action budget.** `(colour, size)`
  gives 83–136 arms on object-rich games against 100 actions. Pooling to
  `colour` gives 6–14, is equal on activity, far better on the diversity a
  learning corpus needs, and repairs the regression (sk48 31%→52%, ar25
  72%→85%, lp85 54%→77%).

  A genuine trade, not a free win: sb26 falls 73%→37% and tn36 53%→36% under
  pooling, games where *size* distinguishes the productive object.

- **Did it change our mind? Yes.** D-028 estimated exploration was worth up to
  eight games. On its own metric it is: the count of games the agent cannot move
  falls from eight to one. Whether that converts into *score* is a separate
  question this entry does not answer.

- **Decision:**
  1. **`explore` becomes the default driver** for any new corpus. `random` and
     `random_objects` are retained as named baselines.
  2. **Both rejected variants stay in the code** (`union_targets`, `key_mode`),
     because each cost an hour to measure and deleting them invites the next
     session to re-run the same experiment.
  3. **The next refinement is a hierarchical prior**, pooling a new
     `(colour, size)` arm towards its colour's aggregate rather than starting it
     at Beta(1,1). It is the principled resolution of the pooling trade and is
     not built.
  4. **Re-record the corpus with `explore` and re-run the dynamics sweep.** The
     learner has never seen a corpus in which the starved games do anything.
- **Options rejected:**
  - *Use activity as the agent's objective* — it is a proxy for
    interestingness and explicitly **not** a proxy for winning. It earns its
    place because D-028 measured that the agent's problem is inactivity, and it
    should be replaced the moment a goal hypothesis exists.
  - *Tune the decay, warmup and band thresholds* — three constants, and tuning
    them on 25 public games is exactly the overfitting D-029 caught itself doing
    with the selection key. They are set to the same values `bookkeeping_mask`
    already used, which at least makes them consistent rather than fitted.
  - *Add a RESET policy now* — some games become *less* responsive with depth
    (cd82 and ka59 fall to ~6% by twenty actions, D-021), so noticing a dry
    spell and restarting would sometimes be right. `RESET` costs an action and
    does not rewind the count, so it needs D-022's cost model, and it is a
    second change on top of this one.
- **Consequences:** the repository now contains something that plays rather than
  only measures. Commits us to activity as the exploration objective, which is a
  placeholder. The band estimator becomes a runtime component, so its failure
  modes are now the agent's failure modes rather than a diagnostic's.
- **Revisit if:** a re-recorded corpus does not improve the dynamics results,
  which would mean activity and learnability are less related than assumed and
  the objective is wrong; or the regression on already-active games costs more
  than the starved games gain once results are scored rather than counted; or
  su15 — the one game still under 15% — turns out to need something structurally
  different, since it is now the only member of its category.

### Footnote: a bug worth recording

The first full run died with `MemoryError` inside the connected-component
labeller, which had nothing to do with it: a 64x64 grid yields at most ~16k
edges. `EnvClient` constructs its own `Arcade` whenever it is not handed one,
and each loads and caches the environment files; the harness builds one client
per game per driver, so **75 engine instances**. The traceback landed wherever
the next allocation happened to fall. Fixed with `client.env.make_arcade()` for
scripts that loop over games. Relevant beyond the harness: Kaggle caps memory,
and a misattributed resource failure in an unrelated component is exactly the
kind of thing that costs a day at the wrong moment.

---

<!-- Next entry: D-032. Copy the template from CLAUDE.md. -->
