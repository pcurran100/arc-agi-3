# Submitting to Kaggle

The submission is one notebook whose agent cell is generated from this
repository. Nothing in it needs the internet at evaluation.

## Build

```bash
python scripts/build_kaggle.py --max-actions 2000          # model-free graph agent
python scripts/build_kaggle.py --max-actions 2000 --model  # with the induced model (slower)
```

This writes `kaggle/build/my_agent.py` (the `arc3` package embedded as source
plus the `MyAgent` adapter in `kaggle/agent_shim.py`) and
`kaggle/build/submission.ipynb`. The build directory is gitignored: rebuild, do
not edit.

## Check locally before uploading

```bash
git clone --depth 1 https://github.com/arcprize/ARC-AGI-3-Agents /tmp/fw
python kaggle/local_harness.py /tmp/fw environment_files ls20 vc33 sp80
```

This instantiates `MyAgent` exactly as the framework's Swarm does and runs the
framework's own `Agent.main()` loop against the local offline engine. It checks
the adapter, the embedded importer and the action plumbing. It does **not**
check the gateway, which only exists inside Kaggle's competition rerun.

## Upload

1. Open https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 and accept
   the rules if you have not.
2. New notebook -> File -> Import notebook -> `kaggle/build/submission.ipynb`.
3. Attach the competition dataset (Add input -> the ARC-AGI-3 competition). The
   notebook installs `arc-agi` from its `arc_agi_3_wheels` folder.
4. Settings: Internet **off**, accelerator **None/CPU** (the agent uses no GPU).
5. Save Version -> Save & Run All. The commit run writes a dummy
   `submission.parquet`; that is expected.
6. Submit to Competition from the saved version. The real run happens in
   Kaggle's rerun, where the gateway plays every hidden game and writes the real
   `submission.parquet`.

## v2: the LLM-steered agent (D-035)

```bash
python scripts/build_kaggle.py --agent v2 --max-actions 1000   # -> kaggle/build/submission_v2.ipynb
```

Same notebook plus one cell that installs vLLM from an attached wheelhouse,
starts its OpenAI-compatible server on an attached model, waits for it to be
healthy (up to 20 minutes), and exports `ARC3_LLM_URL` for the agent. It
searches `/kaggle/input` for a directory with `vllm-*.whl` files and for a model
directory (`config.json` + `*.safetensors` + `tokenizer_config.json`), so no
dataset name is hard-coded. **If either is missing or the server does not come
up, the agent plays exactly as v1** — a serving failure cannot zero the run.
The LLM is not called after `--llm-hours` (default 6.5) from notebook start.

What to attach (Add input), following the Milestone #1 winner's verified stack
(github.com/Tufalabs/duck-harness):

- a **vLLM wheelhouse** built for Kaggle's Python 3.12 / CUDA 12.8 image.
  Tufa's public one is `driessmit1/arc3-vllm-h100-wheelhouse-v3`; or build your
  own in a CPU notebook with internet on (`pip download vllm==0.19.* -d wheels`,
  save as a dataset) — CPU sessions do not use GPU quota.
- a **model snapshot** (Hugging Face format). Tufa's is
  `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` (Qwen3.6-27B FP8, Apache 2.0).
  FP8 needs the RTX 6000 (Blackwell); on a T4 use a 7B AWQ model instead.

**Model settings (D-036).** Qwen3.6 *thinks by default*, which would spend the
900-token cap on reasoning. The vLLM cell therefore defaults to the model
card's non-thinking settings, all overridable by setting the environment
variable in an earlier cell:

| variable | default | thinking-mode run |
|---|---|---|
| `ARC3_LLM_EXTRA` | `{"chat_template_kwargs": {"enable_thinking": false}, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}` | `enable_thinking: true`, `top_p` 0.95, `presence_penalty` 0 |
| `ARC3_LLM_TEMPERATURE` | `0.7` | `0.6` |
| `ARC3_LLM_MAX_TOKENS` | `900` | ~`8000` (5–10× tokens and time per call) |

vLLM is launched with `--max-model-len 16384 --gpu-memory-utilization 0.90
--max-num-seqs 32`, plus `--reasoning-parser qwen3` when the model directory
name contains "qwen3". Another model can be attached instead; nothing else
names Qwen.

Accelerator: **RTX 6000** (the builder sets it). It is reserved for ARC-AGI-3
and burns quota faster than T4; the rate is not published.

Tested locally (2026-09-27) through the real framework loop against llama.cpp
on this laptop: the agent finds the server by `ARC3_LLM_URL`, calls and records,
and with the server absent plays as v1 with no errors. **Not tested: the vLLM
cell itself**, which only runs on Kaggle.

## Benchmarking v2 on Kaggle's GPU (do this before submitting v2)

The submission notebook returns one opaque number for hidden games. To learn
whether the LLM helps, score it on the 25 public games with the real model:

```bash
python scripts/build_kaggle.py --agent bench --seeds 0,1,2 --max-actions 500
# -> kaggle/build/eval_v2.ipynb
```

1. Upload the local `environment_files/` directory as a **private** Kaggle
   dataset (it is downloadable anonymously anyway; D-008 only forbids
   committing it to git).
2. Import `eval_v2.ipynb`, attach: the competition dataset, the vLLM
   wheelhouse, the model snapshot, and the environment-files dataset.
3. Accelerator RTX 6000 (or T4 with a 7B AWQ model), internet off, Save & Run
   All. Each seed plays 25 games; budget roughly 1–2 hours per seed, most of it
   the model's thinking.
4. Download `/kaggle/working/results/`: `eval_s{N}.json` (scores, same format
   as `docs/experiments/agent_*.json`) and `llm_calls/` (every prompt and
   response). Put the calls in `docs/experiments/llm_calls/<model>/` and the
   whole run can be replayed offline for free with `eval_agent.py --replay`.

Checked locally: the benchmark's `arc3` bundle and script run with nothing
from `src/` on the path and reproduce the repository's numbers (tn36 level at
19, sp80 at 45, seed 0). The vLLM cell is the untested part.

## What is known and not known about the harness

Verified from the official starter
(github.com/arcprize/ARC-AGI-3-Kaggle-Starter) and framework
(github.com/arcprize/ARC-AGI-3-Agents) source on 2026-09-26:

- The class must be named `MyAgent`, subclass `agents.agent.Agent`, and
  implement `choose_action` and `is_done`.
- The loop stops at `MAX_ACTIONS` per game (framework default 80; ours is set
  at build time) or when `is_done` returns True. RESET counts as an action.
- Competition mode permits level resets only; a full-game reset becomes a level
  reset. Our agent only ever uses RESET to restart the current level.
- The Swarm runs games in parallel threads. `GameAction.ACTION6.set_data`
  mutates a shared enum member, so `MyAgent.do_action_request` sends its own
  per-instance coordinates instead.

Not verified (could not read the Kaggle rules tab): the runtime limit, the
number of hidden games, and the exact Milestone #2 writeup requirement.
Milestone and prize eligibility require the code to be open-sourced under a
permissive licence; **this repository has no LICENSE file yet.**
