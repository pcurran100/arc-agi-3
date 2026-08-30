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

<!-- Next entry: D-009. Copy the template from CLAUDE.md. -->
