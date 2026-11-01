"""Build (and optionally push) a Kaggle run of the vendored Duck harness (D-045).

The base is Daniel Franzen's Milestone #2 notebook, vendored verbatim under
``third_party/duck-franzen/kaggle/dfranzen/``. It patches Tufa's original source
bundle with Franzen's harness changes (cell 2) and serves Qwen3.8-Flash-Next
(W4A16 AutoRound + INT4 MTP drafter) under SGLang Pennyroyal 2.5.3, all from
public Kaggle inputs that this script declares in ``kernel-metadata.json``.

On top of that, optionally:

- ``--graft`` adds OUR changes to the vendored harness as a second patch, applied
  after Franzen's. The patch is ``git diff`` of ``third_party/duck-franzen/ARC3-Inference``
  against the vendoring commit, so the vendored tree stays the single source.
- ``--memory`` adds lalalia9's solved-level memory and that notebook's longer
  timeouts (``third_party/solved-level-memory/NOTICE.md``).
- ``--env K=V`` sets environment variables in a first cell (e.g. the graft flag
  ``ARC3_SCALE_HASH=1``), so the same patched code runs A and B arms.
- ``--games`` / ``--game-minutes`` shrink an interactive Save & Run to a smoke test.
  They change nothing in a real competition rerun (``KAGGLE_IS_COMPETITION_RERUN``).

    python scripts/push_duck.py --slug duck-base-smoke --games ls20,ft09 --game-minutes 20
    python scripts/push_duck.py --slug duck-scale-smoke --graft --env ARC3_SCALE_HASH=1 \\
        --games ls20,ft09 --game-minutes 20 --push

Without ``--push`` it only writes the kernel folder (a dry run). ``--push`` starts a run
immediately and spends RTX Pro 6000 quota.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUCK = ROOT / "third_party" / "duck-franzen"
NOTEBOOK = DUCK / "kaggle" / "dfranzen" / "arc-agi-3-milestone-2-solution.ipynb"
#: The commit that vendored the harness verbatim; our graft patch is the diff from it.
VENDOR_COMMIT = "b55ec89"
COMPETITION = "arc-prize-2026-arc-agi-3"
#: Franzen's public inputs, exactly as his kernel-metadata.json declares them.
DATASETS = ["dfranzen/pennyroyal-v253", "dfranzen/taaf-kaggle-source-bundle-copy"]
MODELS = [
    "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
    "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
]
FRANZEN_APPLY = ('subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", '
                 '"/kaggle/harness-changes.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)\n')
MEMORY = ROOT / "third_party" / "solved-level-memory" / "memory_patch.py"
#: (old, new) edits that, with MEMORY, turn Franzen's notebook into lalalia9's (see NOTICE.md).
MEMORY_EDITS = [
    ("SERVER_STARTUP_TIMEOUT = 12 * 60", "SERVER_STARTUP_TIMEOUT = 55 * 60"),
    ("os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'",
     "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '3600'"),
    ("bm.solver.analyzer_timeout = 900.0", "bm.solver.analyzer_timeout = 3600.0"),
]
OFFLINE_FILTER = ("    game_ids = [game_id for game_id in game_ids if not any(game_id.startswith(g) "
                  "for g in demo_excluded_games)]\n")


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True}, "outputs": [],
            "execution_count": None, "source": source}


def cell_index(cells: list[dict], needle: str) -> int:
    hits = [i for i, c in enumerate(cells) if needle in "".join(c["source"])]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one cell containing {needle!r}, found {hits}")
    return hits[0]


def replace_once(cell: dict, old: str, new: str) -> None:
    src = "".join(cell["source"])
    if src.count(old) != 1:
        raise SystemExit(f"anchor not found exactly once: {old[:70]!r}")
    cell["source"] = src.replace(old, new)


def graft_patch() -> str:
    """Our changes to the vendored harness, with paths relative to the bundle's src/."""
    out = subprocess.run(
        ["git", "diff", VENDOR_COMMIT, "--relative=third_party/duck-franzen/", "--",
         "third_party/duck-franzen/ARC3-Inference"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    if not out.strip():
        raise SystemExit("--graft given but the vendored harness has no changes")
    return out


def build(args: argparse.Namespace) -> dict:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    if args.graft:
        patch = graft_patch()
        i = cell_index(cells, "%%writefile /kaggle/harness-changes.patch")
        cells.insert(i + 1, code_cell("%%writefile /kaggle/arc3-graft.patch\n" + patch))
        j = cell_index(cells, FRANZEN_APPLY.strip())
        replace_once(cells[j], FRANZEN_APPLY, FRANZEN_APPLY + (
            'subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", '
            '"/kaggle/arc3-graft.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)\n'
            "print('ARC-AGI-3 project graft patch applied (D-045)')\n"))

    if getattr(args, "memory", False):
        for old, new in MEMORY_EDITS:
            replace_once(cells[cell_index(cells, old)], old, new)
        k = cell_index(cells, "# show config")
        replace_once(cells[k], "bm.solver.max_runtime_s_per_game = 48*60",
                     MEMORY.read_text(encoding="utf-8") + "bm.solver.max_runtime_s_per_game = 48*60")

    if args.games:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
        j = cell_index(cells, OFFLINE_FILTER.strip())
        replace_once(cells[j], OFFLINE_FILTER, OFFLINE_FILTER + (
            f"    game_ids = [g for g in game_ids if any(g.startswith(p) for p in {games!r})]\n"))
        k = cell_index(cells, "# show config")
        minutes = args.game_minutes
        replace_once(cells[k], "# show config", (
            "if not TRUE_SUBMISSION:  # smoke test (scripts/push_duck.py --games)\n"
            # Franzen's Save & Run excludes 15 games (ls20 among them); --games overrides.
            "    demo_excluded_games = []\n"
            f"    bm.solver.concurrency = {len(games)} * bm.n_passes\n"
            + (f"    bm.solver.max_runtime_s_per_game = {minutes} * 60\n" if minutes else "")
            + "\n# show config"))

    env = dict(kv.split("=", 1) for kv in args.env)
    if env:
        first = "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
        cells.insert(1, code_cell(first))
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slug", required=True, help="kernel slug under your Kaggle user")
    ap.add_argument("--user", default="experiencebreakfast")
    ap.add_argument("--graft", action="store_true", help="apply our harness changes too")
    ap.add_argument("--memory", action="store_true",
                    help="add lalalia9's solved-level memory (third_party/solved-level-memory)")
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--games", default="", help="comma-separated game-id prefixes (Save & Run)")
    ap.add_argument("--game-minutes", type=float, default=None,
                    help="per-game wall-clock cap for a smoke test (needs --games)")
    ap.add_argument("--accelerator", default="NvidiaRtxPro6000")
    ap.add_argument("--push", action="store_true", help="push and start the run (spends quota)")
    args = ap.parse_args()
    if args.game_minutes and not args.games:
        ap.error("--game-minutes needs --games")

    nb = build(args)
    out = Path(tempfile.mkdtemp(prefix=f"kaggle_{args.slug}_"))
    (out / f"{args.slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"{args.user}/{args.slug}", "title": args.slug, "code_file": f"{args.slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": DATASETS, "competition_sources": [COMPETITION],
        "kernel_sources": [], "model_sources": MODELS, "machine_shape": args.accelerator,
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("wrote", out)
    if not args.push:
        print("dry run: add --push to start it on Kaggle")
        return
    sys.exit(subprocess.call(["kaggle", "kernels", "push", "-p", str(out),
                              "--accelerator", args.accelerator]))


if __name__ == "__main__":
    main()
