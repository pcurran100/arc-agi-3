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

<!-- Next entry: D-020. Copy the template from CLAUDE.md. -->
