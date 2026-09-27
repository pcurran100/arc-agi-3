# Kaggle setup, step by step

Everything here is clicks on kaggle.com plus downloading four notebooks from
this repository. Nothing needs building locally: the notebooks in
`kaggle/notebooks/` are pre-built (regenerate with
`python scripts/build_kaggle.py --all` after any code change).

| notebook | what it is | accelerator |
|---|---|---|
| `submission_v1.ipynb` | the v1 agent, no LLM — the safe Milestone #2 entry | none (CPU) |
| `bench_v2_t4.ipynb` | smoke test of the LLM pipeline on the free T4s, 1 seed | GPU T4 x2 |
| `bench_v2_rtx6000.ipynb` | the real benchmark: v2 on the 25 public games, 3 seeds | GPU RTX Pro 6000 |
| `submission_v2.ipynb` | the v2 submission, once the benchmark says it wins | GPU RTX Pro 6000 |

Order: **A → B → C → D → E**. Each step is cheap enough to fail on before the
next one spends more.

---

## A. Account (once, ~10 min, free)

1. kaggle.com → your avatar → **Settings** → **Phone verification**. GPUs are
   unavailable without it.
2. Open https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 →
   **Join Competition** → accept the rules.
3. Still there, read the **Rules** tab and note: number of hidden games, the
   runtime limit, the Milestone #2 writeup requirement. (Ours says ~110 games /
   9 h, unverified.)
4. Download the four notebooks from GitHub: branch `v2-llm`, folder
   `kaggle/notebooks/` (open each file → **Download raw file**).

## B. Submit v1 for Milestone #2 (~15 min, free, CPU)

1. **Create** → **New Notebook** → **File** → **Import Notebook** → choose
   `submission_v1.ipynb`.
2. Right panel → **Add Input** → search `arc-prize-2026-arc-agi-3` → add the
   competition.
3. Right panel → **Session options**: Accelerator **None**, Internet **Off**.
4. **Save Version** → **Save & Run All (Commit)**. The commit run only writes a
   dummy `submission.parquet`; that is expected.
5. When it has finished: open the saved version → **Submit to Competition**.
   The real scoring happens in Kaggle's rerun.
6. Milestone eligibility needs the code public under a permissive licence —
   **the repository still has no LICENSE file**; that choice is yours.

## C. Upload the inputs (~30 min, free)

**C1. The public games (your own private dataset).**

1. On your machine, zip the folder `environment_files` (right-click →
   Compress). It holds 25 game folders.
2. kaggle.com → **Datasets** → **New Dataset** → drop the zip in. Title
   `arc3-public-games`. Visibility **Private**. **Create**. Kaggle unzips it.
   (It is downloadable anonymously anyway; D-008 only forbids committing it to
   git.)

**C2. Find the shared inputs** (you add them to notebooks later; here, just
check they exist and what is inside):

| what | where | check inside |
|---|---|---|
| vLLM wheelhouse | search Datasets: `arc3-vllm-h100-wheelhouse-v3` (driessmit1) | files named `vllm-0.*.whl`; version **≥ 0.17** if you want Qwen3.8 |
| Qwen3.6-27B FP8 | search Datasets: `vrfai-qwen3-6-27b-fp8-hf-snapshot` (driessmit1) | a folder with `config.json`, `tokenizer_config.json`, `*.safetensors` |
| Qwen3.8-27B FP8 | search Datasets: `qwen3-8-27b-fp8` (e.g. saltb0x) | same three kinds of file |
| Qwen2.5-7B AWQ (T4 test only) | Models: https://www.kaggle.com/models/qwen-lm/qwen2.5/transformers/7b-instruct-awq (official, QwenLM) | same three kinds of file |

If the wheelhouse is gone or private, make your own (free, CPU): new
notebook, Internet **On**, Accelerator **None**, run

```python
!pip download "vllm==0.19.*" -d /kaggle/working/wheels
```

then **Save Version**, open it → **Output** → **New Dataset**. If a model
dataset is gone, tell me — models are too large (~36 GB) to rebuild from a
notebook's output.

## D. Smoke test on the free T4s (~30–45 min of T4 quota)

Tests the one untested part — vLLM starting inside Kaggle — without spending
RTX 6000 time. It uses a 7B model, so its **score means nothing**; it only
proves the pipeline.

1. **Create** → **New Notebook** → **File** → **Import Notebook** →
   `bench_v2_t4.ipynb`.
2. **Add Input** four times: the competition; the wheelhouse; the Qwen2.5-7B
   AWQ model (from **Models**, choose Transformers / 7b-instruct-awq);
   `arc3-public-games`.
3. **Session options**: Accelerator **GPU T4 x2**, Internet **Off**.
4. Click **+ Code** above the first cell and paste:

   ```python
   import os
   os.environ['ARC3_BENCH_GAMES'] = 'tn36 sp80'
   os.environ['ARC3_BENCH_SEEDS'] = '0'
   os.environ['ARC3_BENCH_SHARDS'] = '2'
   ```

5. Run cells one at a time (Shift+Enter) and check:

   | cell | expect | if not |
   |---|---|---|
   | pip install arc-agi | no errors | the competition input is missing |
   | vLLM cell | `vllm wheel dirs: [...] \| model: /kaggle/input/...`, `GPU: ['7.5, Tesla T4', ...]`, then **`LLM up after …s`** (a few minutes) | copy me everything it prints — the tail of `/tmp/vllm.log` is the part I need |
   | the two `%%writefile` cells | `Writing /tmp/…` | — |
   | last cell | `2 processes …`, then `seed 0: 2 games, … mean RHAE …` | copy me the output and `results/log_*.txt` |

6. Right panel → **Output** (or the folder icon): `results/llm_calls/` must
   contain `.jsonl` files. That is the proof the model was actually called.
7. **Stop the session** (power button, top right). Open sessions burn quota.
8. Download `results/` (three-dot menu on the folder) and send it, or put it in
   `docs/experiments/kaggle/smoke_t4/`.

## E. The real benchmark on the RTX 6000 (~1–2 h per model, estimated)

1. First measure the RTX 6000's quota rate: note your remaining GPU hours
   (right panel), run any RTX 6000 session for 5 minutes, stop it, look again.
2. Import `bench_v2_rtx6000.ipynb`. **Add Input**: the competition, the
   wheelhouse, `arc3-public-games`, and **one** model (Qwen3.6 first).
3. **Session options**: Accelerator **GPU RTX Pro 6000**, Internet **Off**.
4. **+ Code** above the first cell:

   ```python
   import os
   os.environ['ARC3_MODEL_MATCH'] = 'qwen3-6'   # a substring of the model's folder path
   ```

   (Short on quota? Add `os.environ['ARC3_BENCH_SEEDS'] = '0'` for one seed.)
5. **Save Version** → **Save & Run All (Commit)**. It runs in the background;
   you can close the browser. It plays 25 games x 3 seeds in parallel shards.
6. When done: open the version → **Output** → download `results/`.
7. Repeat with the Qwen3.8 model attached instead and
   `ARC3_MODEL_MATCH = 'qwen3-8'`. Same seeds.
8. Put the downloads at `docs/experiments/kaggle/qwen36/` and
   `docs/experiments/kaggle/qwen38/`. From there the scoreboard, the decision
   log and offline replay are on the repository side.

## F. Submitting v2 (only if E says it beats v1)

Import `submission_v2.ipynb`; **Add Input**: competition, wheelhouse, the
winning model; set `ARC3_MODEL_MATCH` in a first cell as in E; RTX Pro 6000;
Internet Off; Save & Run All; Submit. If the model cannot start in Kaggle's
rerun, the agent plays as v1 — a serving failure cannot zero the score.

---

**Checked before handing this over:** every code cell of the four notebooks
compiles; the benchmark's run cell was executed locally with Kaggle paths
remapped and reproduced v1's seed-0 results; the vLLM cell was executed with
nothing attached and fell back to v1 as designed. **Never run anywhere:** the
vLLM cell with a real GPU, wheelhouse and model. That is what step D is for.
