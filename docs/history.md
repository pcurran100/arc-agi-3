# History: the path to the current state, the ideas we tried, and the roads not taken

- **Written:** 2026-11-02, deadline day, from the decision log (D-001 to
  D-049), the experiment index and the commit history on every branch.
- **What this is:** the narrative that the append-only decision log cannot
  give. It covers how one idea led to the next, which ideas were tried and
  dropped, which were parked, and which were never tried. Every claim cites the
  entry that holds the evidence. Where the record is silent, this says so
  rather than filling the gap.
- **What this is not:** the record itself. When this document and
  `decision-log.md` disagree, the log wins.

---

## The short version

We set out to win ARC-AGI-3 with a **model-based** agent. The scoring makes
thinking free and acting expensive, which is the condition under which an
agent that learns how the game works and plans internally should beat one
that acts and reacts (`design-thesis.md`).

Over nine weeks the agent was built five times. Each version moved the LLM
further into the loop:

| when | version | role of the LLM | best measured result |
|---|---|---|---|
| 30 Aug – 26 Sep | perception, dynamics induction, exploration | none | perception exact and audited; induced dynamics beat chance on 7/23 games (D-025) |
| 26–27 Sep | **v1** `GraphAgent` | none | **2.42% offline RHAE**, 3 seeds × 25 games, the best score any of our own versions reached (D-034) |
| 27–30 Sep | **v2** | proposes goal hypotheses as code, ≤16 times a game | identical to v1 on Kaggle (D-035, D-039) |
| 30 Sep – 16 Oct | **v3** | writes the world model as code; code verifies it and plans in it | identical to v1; 34 of 45 answers lost to truncation (D-040) |
| 19–29 Oct | **v4** | edits an assumption store, predicts every action's outcome, names subgoals | 0 levels on ls20 twice, then 0 on 5 games (D-042, D-044) |
| 30–31 Oct | **v5** | writes a simulator that code replays and searches | 1 level on 4 valid games, at 2–3× less cost than v4 (D-046) |

None of our LLM versions beat the goal-free v1 explorer on score. Four days
before the deadline we split the work (D-043). Track A kept improving our
agent (v4 fixes, then v5). Track B vendored the public Milestone #2 winner,
Daniel Franzen's Duck harness, at ~28 on the leaderboard against our ~2, and
grafted our findings onto it (D-045). The submission comes from track B
(D-047). On deadline day the user ruled out entering the plain public
baseline: only a variant that improves on it goes in, and only with the user
present (D-049).

---

## Phase 1: perception first (30 Aug – 8 Sep; D-001 to D-017)

**What we did.** We chose ARC-AGI-3 over ARC-AGI-2 (D-001) and committed to
the model-based bet (D-002), a population of hypotheses (D-003), and three
separate inference problems: ontology, dynamics and goal (D-004). Session 1
built only perception (D-007): connected components, frame geometry, a
structural diff, six candidate segmentations, and object tracking by global
assignment (D-014). The engine's own sprites gave a byte-exact oracle for
testing, and the agent is never allowed to use it (D-010).

**What changed our mind early.**

- **There is no Undo.** The thesis's exploration argument had rested on a
  cheap probe-and-revert that does not exist (D-009). It was re-derived around
  RESET, which costs *k*+1 actions (D-022).
- **A frame is a composite.** It is a camera view upscaled into a 64×64
  canvas with a UI layer drawn on top, not a board (D-009, D-011).
- **An action returns an animation** of up to 42 frames, not one frame.
- **Per-level human baselines ship with the games**, so true RHAE can be
  computed offline (D-012).

**Ideas tried and dropped in this phase.**

- *Frame geometry as the basis for click pruning* (D-011, D-013). It finds
  structure on only 4 of 25 games. Segmentation carries the pruning instead
  (D-015).
- *An independent audit of our own claims* (D-015). This one was kept as a
  habit. It corrected several overstated numbers, such as click pruning
  "15–25", which was really 8–196.

---

## Phase 2: the dynamics-induction bet (9–25 Sep; D-018 to D-032)

**What we did.** We induced symbolic rules, `(action, selector) → effect`,
selected by minimum description length (MDL), from recorded transitions
(D-024), and measured them under all six segmentations (D-018).

**Result: a weak pass** (D-025). Exact held-out frames were predicted on 7 of
23 games, against a chance rate of 0%. The language saturated at ~50 actions
of evidence, which means it was limited by expressiveness, not by data. Four
games had a correct object model and still scored zero, because a two-pixel
UI counter broke the exact-frame comparison.

**Ideas tried and dropped.**

| idea | what happened | ref |
|---|---|---|
| Round-robin driving instead of random | no better on no-ops, worse on event diversity | D-020 |
| Clicking random pixels to record the corpus | could not teach any click rule; switched to clicking objects | D-026 |
| Learning from static ARC-AGI-2-style tasks and transferring | dropped, not parked | D-023 |
| Choosing the best segmentation per game from validation data | a coin flip (51% on game holdout) | D-029, closed negative |
| Object accuracy as the headline metric | scored blindness as success; needs `blind_rate` beside it | D-030 |

**What worked:** a Thompson-sampling exploration policy. The number of games
the agent could not make move at all fell from 8 to 1 (D-031). It is the first
capability this project gained, and it is still the core of v1.

**The finding that shaped everything after:** the "zeros" decomposed into
games that never moved, games blocked by the UI, and games that were genuinely
unmodelled (D-028, corrected by D-030). The binding constraint moved from
learning to exploring, and then (D-034) to **having no goal at all**.

---

## Phase 3: an agent and a score (26–27 Sep; D-034)

**v1** (`GraphAgent`, tag `v1-graph`) combines exact perception, the explorer
and a state graph with navigation. It scored **2.42% mean offline RHAE** over
three seeds, tied with the bare explorer (2.07%).

**The negative result that mattered:** the induced dynamics model, used as a
one-step action filter, **cut the score to a third** (3.46% → 1.18% on seed
0). It ships switched off. Single-seed tuning was shown to be noise, which is
where the scoreboard's three-seed rule comes from.

A Kaggle notebook for v1 was built for Milestone #2 (`kaggle/SETUP.md`). The
record has no Kaggle score for it.

---

## Phase 4: the LLM proposes goals (27–30 Sep; v2, D-035 to D-038)

**Idea:** an LLM proposes goal hypotheses as small Python functions. A
weighted population tests them, and they steer v1's search. The LLM never
picks an action.

**What we learned.**

- With a correct scripted hypothesis, a synthetic game fell from unsolved in
  600 actions to 61 actions. The machinery worked.
- **Averaging a population of hypotheses cancels them** (D-038). A right and a
  wrong hypothesis in the same answer produced a flat value landscape and
  0 levels. Following one hypothesis at a time, sampled by weight, gave 2
  levels. This is now a regression test.
- With a real model, v2 was **identical to v1** (Kaggle, tn36 + sp80). The
  user's verdict: an LLM consulted at most 16 times a game, and part of no
  decision, "will never yield results" (D-039).

**Tried along the way:**

- *Qwen2.5-Coder-3B on the laptop CPU* (llama.cpp). It produced only
  colour-count hypotheses, and the run was stopped by low memory after 5 of 25
  games.
- *vLLM on Kaggle's 2×T4.* It failed; the RTX 6000 works (commit `edb4551`).
- *Model choice.* We picked Qwen3.6-27B FP8, the Milestone #1 winner's model
  (D-036), and planned a head-to-head against Qwen3.8-27B (D-037). **That
  comparison was never run.** Franzen's write-up
  (`third_party/duck-franzen/WRITEUP.md`) later reported that each model
  upgrade (3.6-27B → 3.8-27B → 3.8-Flash-Next) brought a substantial
  improvement. We stayed on 3.6-27B for our own agent.

---

## Phase 5: the LLM in the loop (30 Sep – 16 Oct; v3, D-039 to D-041)

**Idea:** the LLM is the reasoner. It writes a world model as code,
`predict(state, action)` and `goal(state)`. Code checks the model against
every recorded transition, turns each mismatch into an exact counterexample,
and plans the shortest route inside a verified model. `CLAUDE.md`'s thesis was
rewritten to match, with the user's approval.

**Getting a 27B model to serve on Kaggle took four runs** (D-040):

1. Web-imported notebooks lose their inputs.
2. Kaggle's image moved to Python 3.13, which the wheels do not support.
3. The pinned 3.12 image had no GPU.
4. FlashInfer's JIT FP8 kernel fails on SM120.

The fix was vLLM under a standalone CPython 3.12, with that kernel disabled.

**Result:** score identical to v1. 34 of 45 answers were lost: truncated
after long reasoning, or degenerated into whitespace. One tn36 world model
verified at **100%**, so the model can write checkable world models. The
plumbing was losing them.

**Licence:** MIT-0, to meet the rule as quoted for prize eligibility (D-041).

---

## Phase 6: assumptions and checked predictions (19–29 Oct; v4, D-042, D-044)

**Idea**, following the user's critique of v3: put beliefs in an
**assumption store** the model edits. Every action carries a **prediction**
that code checks. Reasoning is never cut off. The image is sent beside the
exact grid. The model is consulted on every action while it is learning.
Development moved to NVIDIA NIM's free Nemotron-3.5-Super-VL, because NIM
serves no Qwen.

**Result:** on ls20 the predictions improved between runs (17/43 → 51/79
correct), but play did not: 0/7 levels. The model took the move-counter bar
for the goal and spent the run trying to fill it.

**The forensic read (D-044) found most of the failure in our code:**

- The human baseline in the prompt was read as a budget.
- Every correct prediction credited the goal named in it, so a wrong goal
  reached 0.91 confidence.
- Exact shape matching hid the goal link, which was a 2×, rotated copy of a
  glyph.
- A lost life was read as "capturing a target".
- Occlusion was reported as reshaping.
- "Predicted no move, but it moved" was scored as a success.

Each of these was fixed, and **subgoals walked by code** were added: the
model names where to go and BFS finds the route. A scripted check cleared
ls20 level 1 in 22 actions, equal to the human baseline. With the real model,
**0 levels on 5 games**, at about 1.3 calls per action.

**Tried and dropped here:**

- *Images beside the grid.* Turned off by default: there was no measured
  benefit (D-044).
- *Crediting goals from correct predictions.* Replaced by credit for level
  progress only.
- *Two NIM evaluations side by side.* One free key throttles at about 5
  concurrent thinking-on requests. The old-vs-new v4 comparison was lost to
  this (D-044, aborted).

---

## Phase 7: two tracks (29–31 Oct; D-043, D-045, D-046)

**Track A, v5** (`v5-objects`): object permanence, an object sheet in place
of the event log, counters moved out of the object list, minor surprises
logged without dropping the plan, and the model's own simulator replayed and
searched by code. Calls were tiered as `think`, `next` and `fix`.

- **Result:** 1 level on 4 valid games (vc33, level 1).
- **Cost:** 0.43–0.75 calls per action, against ~1.3 for v4.
- **The simulator was used:** 11–28 actions per game came from simulator
  search.
- **Verdict:** the cost problem was largely solved. The failure moved from
  "cannot act efficiently" to "does not find the goal" (D-046).
- **Process failures on the way:** a network outage invalidated one run
  (`v5_set5`), and a builder agent stalled and lost its draft write-up.

**Track B** (`duck-base`): Franzen's harness, vendored verbatim, with our
changes as separate, flag-gated patches (D-045). The first graft was the
scale-invariant shape hash, taken directly from the v4 forensics. Smoke runs
on Kaggle (ls20 + ft09) ran end to end once the image was pinned to
Python 3.12. A scored 57.72 and B 59.92, which is noise (D-047).

---

## Phase 8: what to submit (1–2 Nov; D-047, D-049)

The plan of 1 November (D-047) was to submit A (Franzen plus lalalia9's
solved-level memory) as the safe entry, plus one variant carrying a single
change of ours.

On 2 November the user rejected A as an entry ("there is no point just being
in line with the pack") and reserved every submission for themselves
(D-049). The candidates under test on `duck-base`:

- turn hints (D-048, in progress);
- the model-visible RESET;
- the scale hash;
- the Adaptive Duck tools.

They are compared offline on NIM with `kaggle/duck_grafts/local_run.py`. When
this was written, the A-vs-D comparison was still running.

---

## Ideas that were parked, and never tried

The condition that would reopen each idea is in its entry.

- **A population over dynamics hypotheses.** Guardrail 1 asks for one. The
  dynamics layer only ever kept the single MDL-best rule set (D-024). Goal
  hypotheses got a population in v2 and v4; simulators never did (v5 keeps
  one at a time).
- **Multi-step rollout accuracy.** Objective 6 in the old state-of-play.
  Never measured. Every dynamics number is single-step, so whether
  predictions chain is still unknown.
- **The 42-frame animations as evidence.** Recorded, then never read by the
  dynamics layer (D-024).
- **Per-region frame geometry**, and the oracle inside `perform_action`
  (D-013, D-014).
- **Revisable objecthood.** Splits and merges were deferred by D-016. v5
  reports them as events, but segmentation still never learns from dynamics.
- **Meta-learning over procedurally generated games.** Judged right in the
  long run but months of GPU work (D-002).
- **Fine-tuning.** 25 games are a validation set, not a training set (D-035).
  An ARC-AGI-3 RL LoRA (`U4AR/qwen38-arc3-rl`) was noted and not tried (D-037).
- **Other models.** Gemma-4-31B, MoE models (Qwen3.6-35B-A3B, gpt-oss-120b),
  Qwen3.8-27B and Qwen3.8-Flash-Next were all noted (D-036, D-037). None was
  tried in our own agent; Flash-Next arrived only through the Duck harness.
- **Grafts onto the Duck harness that we judged too risky to make blind:**
  code-checked predictions, and goals as a population with attainment tests.
  Both are our thesis's core. They change the harness's control flow, which
  is not something to do four days out (D-045).
- **v4/v5 on Kaggle with Qwen.** v4 was never packaged for Kaggle: that needed
  vLLM with image input, thinking on and a larger context. v5 never ran
  there either. Every v4/v5 number is on Nemotron-VL through NIM.

## Options considered and rejected outright (selected)

- **The LLM picks every action, with no world model** (the Milestone #1 style;
  D-035, D-039). Rejected twice on budget and thesis grounds. The irony is on
  the record: the submission now uses a descendant of exactly that style,
  because it scores ~28 and ours scored ~2.
- **Letting the LLM do perception from a rendered image** (D-006, D-039). The
  exact grid stayed the referee throughout. v4 added the image as a prior
  only, and it was turned off again.
- **Hard caps on thinking** (D-042, D-044, D-046). Rejected by the user:
  truncation loses the conclusion, and v3 lost three-quarters of its answers
  to it.
- **A hand-written catalogue of game types or object roles** (Guardrail 2,
  D-039, D-042). It never entered the code. The fact layers compute generic
  relations, and roles are the model's hypotheses.
- **Re-implementing the Duck approach inside our own agent** (D-045). Four
  days was not enough.

---

## What we would tell ourselves on 30 August

Each lesson is sourced to the entry that taught it. None is hindsight that the
record cannot support.

1. **Measure against the public leaderboard early.** We did not compare our
   own score with the public Milestone #2 notebooks until 29 October (D-043).
   By then the gap was ~25 points. The state-of-play of 25 September had
   written Milestone #2 off on a wrong assumption about the user's available
   time.
2. **The goal is the bottleneck, not dynamics.** Every version from v1 to v5
   failed on the goal: no goal at all (v1), goals that decided nothing (v2),
   goals never reached because the plumbing lost answers (v3), the wrong goal
   held with confidence (v4), and the goal not found (v5). The design thesis
   named the value function as "the genuinely unsolved part" (§7) in its first
   draft.
3. **Most of what looked like model failure was our code** (D-040: truncation;
   D-044: six forensic findings). Read the transcripts before blaming the
   model.
4. **Develop on the model you will run.** Our LLM versions were developed on a
   3B model on CPU, then on Nemotron-VL through NIM, and never benchmarked on
   the Kaggle model at scale. Franzen's write-up says the model upgrades were
   among his largest gains.
5. **The record held up.** Every branch's work is reconstructible from the
   log, even when three branches were writing to it at once. Writing every
   decision down, including the dead ends, is the reason this document
   exists.
