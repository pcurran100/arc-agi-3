"""Push a pre-built notebook to Kaggle as a committed (batch) run, with its inputs declared.

Why this exists: importing a notebook through Kaggle's web UI does not carry its
inputs over, and the 2026-10-05 `benchmark_v3` run went out with only the
competition attached (`dataset_sources: []`), so the vLLM cell found no wheels
and no model and the agent silently played as v1. Declaring the inputs in
`kernel-metadata.json` removes that failure mode.

    python scripts/push_kaggle.py kaggle/notebooks/bench_rtx6000.ipynb \\
        --slug arc3-bench-v3-smoke --env ARC3_BENCH_GAMES="ls20 tn36" \\
        --env ARC3_BENCH_SEEDS=0 --llm-minutes 40

Then poll and fetch with the Kaggle CLI:

    kaggle kernels status <user>/<slug>
    kaggle kernels output <user>/<slug> -p <dir>

Needs the Kaggle CLI authenticated (KAGGLE_API_TOKEN or ~/.kaggle).
This spends GPU quota: the push starts a run immediately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

COMPETITION = "arc-prize-2026-arc-agi-3"
WHEELHOUSE = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
PY312_IMAGE = ("gcr.io/kaggle-images/python@sha256:"
               "dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9")
MODELS = {
    "qwen3-6": "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
    "qwen3-8": "jamalsaeedi/qwen3-8-27b-fp8",
}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True}, "outputs": [],
            "execution_count": None, "source": source}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--slug", required=True, help="kernel slug under your Kaggle user")
    ap.add_argument("--user", default="experiencebreakfast")
    ap.add_argument("--model", default="qwen3-6", choices=sorted(MODELS))
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="environment variable set in a first cell (repeatable)")
    ap.add_argument("--llm-minutes", type=float, default=None,
                    help="stop calling the model this many minutes after it is up "
                         "(overrides the notebook's LLM deadline; bounds quota)")
    ap.add_argument("--accelerator", default="NvidiaRtxPro6000")
    ap.add_argument("--docker-image", default=PY312_IMAGE)
    ap.add_argument("--timeout", type=int, default=None, help="kernel run limit, seconds")
    ap.add_argument("--dry-run", action="store_true", help="write the folder, do not push")
    args = ap.parse_args()

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    env = {"ARC3_MODEL_MATCH": args.model}
    for kv in args.env:
        k, v = kv.split("=", 1)
        env[k] = v
    first = "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
    cells = nb["cells"]
    insert_at = 1 if cells and cells[0]["cell_type"] == "markdown" else 0
    cells.insert(insert_at, code_cell(first))
    if args.llm_minutes is not None:
        # The vLLM cell sets ARC3_LLM_DEADLINE from notebook start; reset it
        # from the moment the server is up, so a slow load does not eat it.
        i = next(i for i, c in enumerate(cells) if "ARC3_LLM_DEADLINE" in "".join(c["source"]))
        cells.insert(i + 1, code_cell(
            "import os, time\n"
            f"os.environ['ARC3_LLM_DEADLINE'] = str(time.time() + {args.llm_minutes} * 60)\n"
            "print('LLM URL:', os.environ.get('ARC3_LLM_URL'), '| deadline in',"
            f" {args.llm_minutes}, 'min')\n"))
    # Always end with a summary of what the model was actually asked and answered.
    cells.append(code_cell(
        "import glob, json, collections\n"
        "files = glob.glob('/kaggle/working/results/llm_calls/*.jsonl')\n"
        "print(len(files), 'call files')\n"
        "for f in sorted(files):\n"
        "    rows = [json.loads(l) for l in open(f) if l.strip()]\n"
        "    print(f.split('/')[-1], len(rows), 'calls')\n"
        "for f in sorted(glob.glob('/kaggle/working/results/eval_s*.json')):\n"
        "    if '_k' in f: continue\n"
        "    for r in json.load(open(f))['rows']:\n"
        "        print(f.split('/')[-1], r.get('game'), r.get('levels'), r.get('score'), r.get('info', r.get('stats')))\n"))

    out = Path(tempfile.mkdtemp(prefix=f"kaggle_{args.slug}_"))
    code = out / f"{args.slug}.ipynb"
    code.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"{args.user}/{args.slug}", "title": args.slug, "code_file": code.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [WHEELHOUSE, MODELS[args.model]],
        "competition_sources": [COMPETITION], "kernel_sources": [], "model_sources": [],
        "machine_shape": args.accelerator,
        # The image of the 2026-09-30 run on which vLLM served. Newer images
        # default to Python 3.13 and cannot install the cp312 wheelhouse.
        "docker_image": args.docker_image,
        "docker_image_pinning_type": "original",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("wrote", out)
    if args.dry_run:
        return
    cmd = ["kaggle", "kernels", "push", "-p", str(out), "--accelerator", args.accelerator]
    if args.timeout:
        cmd += ["-t", str(args.timeout)]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
