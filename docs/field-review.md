# Field review: what has been tried, and what it scored

Written 30 August 2026. Purpose is to stop us reinventing things that are already
known to work, and — more importantly — to stop us reinventing things that are
already known *not* to work.

---

## 1. ARC-AGI-3, Milestone Prize #1 (30 June 2026)

The only public evidence of what wins on the interactive track. All three
podium entries are **harnesses around small open-source LLMs**. None trained a
model.

**1st — Tufa Labs, "The Duck"** ($37.5K pool)
A small open-source LLM that writes and runs Python in a **live REPL**. Game
states are converted into Python variables. Multimodal perception via rendered
images, ASCII grids and segmentation tools together. Handles unbounded play via
"infinite play via eviction" — older messages are dropped to avoid context
exhaustion.

> Their reported finding: **"hand-crafted tools actually hurt the model; letting
> it improvise worked better."**

This is the single most useful data point in the field and it is the basis of
Guardrail 2 in `CLAUDE.md`. Note carefully what it does and does not say: it is
an argument against *human-specified* structure, not against *induced* structure.
Those get conflated.

**2nd — Reki**
Vision-language approach running Gemma-4-31B locally. Renders board frames as
labelled images, returns structured JSON actions. Maintains a "reflection memory"
refreshed roughly every 10 steps. Numpy-based click heuristic with fallback for
exploration.

**3rd — Md Boktiar Mahbub Murad, "forge"**
Same architecture as Reki, packaged as a configurable framework. Gemma-4-31B,
JSON actions, reflection memory, optional confidence prompts for conservative
moves.

### How the winner served its model offline *(added 2026-09-27, D-035)*

Read from Tufa's public repository (github.com/Tufalabs/duck-harness, notebook
`taaf-duck-harness-kaggle-share.ipynb`, `ARC3-Inference/README.md`) and writeup
(tufalabs.ai/research/duck-harness/):

- **Qwen3.6-27B in FP8, served by vLLM** as an OpenAI-compatible HTTP server
  inside the notebook (`http://127.0.0.1:1234/v1`), chosen for "Kaggle's
  single-GPU constraint". FP8 implies the Ada/Blackwell RTX 6000.
- The wheelhouse and the model snapshot are **attached Kaggle datasets**
  (`driessmit1/arc3-vllm-h100-wheelhouse-v3`,
  `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`), installed with
  `pip install --no-index --find-links`.
- The harness checks `KAGGLE_IS_COMPETITION_RERUN` and keeps a soft deadline
  before the time limit.

A third party measured the same stack (github.com/leejianrong/solve-arc-agi-3,
`docs/baselines/qwen36-vllm-artifact.md`): 35.9 GB snapshot, 5.1 GB wheelhouse,
**351 s to load, 1,819 tok/s prefill, 38.7 tok/s decode** at concurrency 1 on
the RTX Pro 6000. Our v2 packaging copies this stack (`kaggle/README.md`).

### The Duck, read from source *(added 2026-09-30)*

From `github.com/Tufalabs/duck-harness` (`ARC3-Inference/inference/agent/{prompts,tool_agent,python_tool_sandbox}.py`,
`configs/inference.json`, `example-run/summary.txt`):

- **Loop.** Every turn is a tool-calling loop with one tool, `python`
  (sandboxed, 30 s CPU, stdlib only, no numpy). The model inspects as much as it
  likes; the turn ends when its code calls `action([...])`, which may batch
  actions and stops early on a level change or game over. Each turn shows a 4×
  upscaled **PNG** of the grid, a status line and the carried text "working
  world model"; the grid itself is read through Python objects
  (`current_frame.ascii`, `.segmentation` with shape hashes, boundaries,
  containment and adjacency; `history`, `transitions`).
- **Memory is text.** Lines tagged `World model:`, `Goal model:`, `Action
  model:`, `Plan:`, `Cross-level notes:` are regex-extracted and re-injected;
  code does not persist between calls; everything but `Cross-level notes` is
  wiped at a level change; the last 30 turns are kept, older ones evicted.
- **No verification, no simulator.** The world model is prose. The prompt asks
  for BFS once dynamics are understood; in sampled transcripts BFS was written
  over ASCII cells with guessed walkable symbols, not over a checked model.
- **Engineering.** Qwen3.6-27B FP8, vLLM 0.19, thinking **on**, 64k context,
  temperature 0.6, prefix caching; 32 concurrent games per GPU in the public run.
- **Results.** 25 public games × 20 passes: **mean 1.60, median 0.07**, about
  137 actions and 59k tokens per game; best ft09 10.28, r11l 3.71, vc33 3.33.
  No ablations are in the repository; "hand-crafted tools hurt" is from the
  blog only.

### What we take from this

- The **agent-writes-code** entry beat both **agent-picks-JSON-action** entries.
  The LLM is more useful as a *program proposer* than as a policy.
- Nobody on the podium maintains a queryable model of the environment. A
  "reflection memory refreshed every ~10 steps" is a natural-language summary.
  You cannot ask a paragraph "what happens if I press left twice from here." That
  is the gap we are aiming at.
- Small open-source models are competitive. Frontier access is not the moat.

---

## 2. ARC Prize 2025 (ARC-AGI-2 track) — final results

1,455 teams, 15,154 entries. Winning solution cost ≈ $0.20/task.

| Rank | Team | Score | Approach |
|---|---|---|---|
| 1 | NVARC | 24.03% | synthetic-data ensemble, improved test-time-trained model, Tiny Recursive Model components |
| 2 | the ARChitects | 16.53% | 2D-aware **masked-diffusion** LLM with recursive self-refinement and perspective-based scoring |
| 3 | MindsAI | 12.64% | heavily engineered test-time-training pipeline: test-time fine-tuning, augmentation ensembles, tokenizer dropout, pretraining innovations |
| 4 | Lonnie | 6.67% | |
| 5 | G. Barbadillo | 6.53% | |

Note the contrast with the v3 milestone: **every** v2 podium finisher trained a
model. Harness-only does not win the static track. The two tracks reward
genuinely different things.

### Paper prizes

- **1st, $50K — Jolicoeur-Martineau, "Less is More": Tiny Recursive Model (TRM).**
  **7M parameters**, no large pretraining, ~45% on ARC-AGI-1 via recursive
  reasoning. Existence proof that ARC does not require an LLM.
- **2nd, $20K — Pourcel, Colas, Oudeyer: SOAR.** Self-improving evolutionary
  program synthesis, up to +52% **without any hand-engineered DSL**.
- **3rd, $5K — Liao & Gu: CompressARC.** Pure test-time optimisation by
  **minimum description length**, **76K parameters**, no pretraining, 20% on
  ARC-AGI-1 (20–34% reported range).

The 2025 technical report names the **"refinement loop"** — iterative per-task
optimisation guided by feedback — as the defining breakthrough of the year.

---

## 3. Frontier / unconstrained systems on ARC-AGI-2

| System | Score | Cost |
|---|---|---|
| Claude Opus 5 (Max) | 90.4% | — |
| Claude Opus 5 (High) | 88.3% | — |
| Gemini 3 Deep Think | 84.6% | — |
| Poetiq (meta-system over frontier models) | 54% | $30.57/problem |
| prior SOTA before Poetiq | 45% | $77.16/problem |
| Claude Opus 4.5 (Thinking, 64k) | 37.6% | — |

**Poetiq** is worth understanding: a meta-system that automatically constructs
full solution systems around *any* existing frontier model — "learned test time
reasoning" — explicitly without training or fine-tuning their own models. It is
the strongest evidence that the harness/scaffold layer has large untapped
headroom, which is the bet we are making on v3.

---

## 4. The living survey (arXiv 2603.13372)

### Six correlates of ≥70% performance on ARC-AGI-1

Every system above that threshold has most of these. Our design should be
checkable against the list.

1. **Guided search** — learned or neural heuristics prioritising candidates
2. **Multiple hypotheses** — diverse candidate maintenance with consistency
   filtering
3. **Test-time adaptation** — per-task fine-tuning or dynamic adjustment;
   *present in every top performer*
4. **Hybrid reasoning** — neural perception plus symbolic verification
5. **Robust perception** — multiple candidate representations, worth a
   **20–30 point** advantage on its own
6. **Compositional primitives** — reusable operations composed for novel tasks

### Selected methods

| Method | Type | v1 | v2 |
|---|---|---|---|
| Berman (natural-language programs) | synthesis | 79.6% | 29.4% |
| Pang (libraries) | synthesis | 77.1% | 26.0% |
| Franzen (Product of Experts) | test-time adaptation | 71.6% | 2.5% |
| TRM | neural, 7M params | ~45% | — |
| CompressARC | MDL, 76K params | 20–34% | — |

Best ARC-AGI-3 figure recorded in the survey: **StochasticGoose at 12.58% action
efficiency** — an 8× larger gap to human than on ARC-AGI-1.

### Open problems the survey names

1. **Compositional generalisation bottleneck.** Universal 2.5–3× degradation
   from v1 to v2 across *all* paradigms. Systems treat composition as sequential
   concatenation with O(n^d) search growth rather than hierarchical decomposition.
2. **Symbol grounding for learned primitives.** Systems remain "knowledge-bound"
   even with synthetic data — the 2025 winner needed *hundreds of thousands* of
   synthetic examples for 24%.
3. **Interactive learning capability gap.** Static-to-interactive transfer
   "largely absent". Missing: hypothesis-driven exploration, persistent memory
   across interaction sequences, goal inference from sparse feedback. **This is
   precisely our target.**
4. **Efficiency crisis.** Accuracy ≈ α + β·log(Cost) with **β ≈ 0.15**. Brute
   force to human level is economically prohibitive.
5. **Evaluation reliability.** Evaluating on <100 tasks inflates results by
   **26.8 percentage points**. Only **11%** of papers report costs.

That last point is a methodological obligation for us: **report costs, and do not
draw conclusions from small evaluation samples.** With only 25 public demo
environments we are squarely inside the danger zone.

---

## 5. Open question worth a paper on its own

The survey states that static-to-interactive transfer is largely absent — meaning
**nobody has established whether being good at ARC-AGI-2 helps at all on
ARC-AGI-3**. That is answerable with modest compute and is a cheaper, more
certain research output than winning the track. Logged as a possible parallel
objective; see `docs/state-of-play.md`.

---

## Sources

- ARC Prize 2026: Milestone Prize #1 — https://arcprize.org/blog/arc-prize-2026-milestone-1
- ARC Prize 2025 Results and Analysis — https://arcprize.org/blog/arc-prize-2025-results-analysis
- ARC Prize 2025 Technical Report — https://arxiv.org/html/2601.10904v1
- The ARC of Progress towards AGI: A Living Survey — https://arxiv.org/html/2603.13372v1
- ARC-AGI-3 technical report — https://arxiv.org/html/2603.24621v1
- Claude Opus 5 results — https://arcprize.org/results/anthropic-claude-opus-5
- Poetiq, ARC-AGI-2 verified — https://poetiq.ai/posts/arcagi_verified/
