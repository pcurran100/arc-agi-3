"""Can the agent choose its own carving, instead of us choosing one for it?

D-018 made the segmentation an experimental axis and D-025 found the best
carving is **game-dependent**: `colour_4` wins on five games, `foreground_4` on
two and is the only carving that finds anything at all on ar25 and cn04. The
union across carvings scores on 7 of 23 games where any single carving manages
5. `architecture.md` section 3.4 has recorded "nothing selects among the six"
since session 2, and it has been treated as an open research problem.

It is not one. **Held-out validation is the standard answer and was never
tried.** Split the training transitions again; induce under all six carvings on
the inner split; keep whichever predicts the held-out validation split best;
then induce that one carving on the full training set and score it on the real
test set. No new information, no tuned parameter, and the test set stays
untouched throughout.

Three numbers make the result readable:

- **single** — the best fixed carving applied to every game, which is what a
  system with no selection would get. The honest baseline.
- **selected** — what validation picks, per game. The achievable number.
- **oracle** — the best carving per game chosen by looking at the test score
  itself. Cheating, and it is the ceiling selection could possibly reach.

If `selected` sits near `oracle`, selection works and per-game carving choice
costs nothing but compute. If it sits near `single`, the validation signal is too
weak at this corpus size and the six-fold cost buys nothing at runtime.

**Choosing what to select ON is the whole problem, and the first attempt got it
wrong.** Selecting on validation *exact match* recovered only 8% of the
available headroom, because exact match is **zero under every carving on most
games** at this validation size — there is nothing to rank. A selection
criterion has to be:

- **dense**, not almost always zero, so ~20 validation transitions carry signal;
- **carving-independent**, or the comparison is rigged. This rules out
  changed-object accuracy, tempting though it is: it is measured in terms of the
  carving being judged, so a carving producing two big objects scores easily
  where one producing fifty cannot. That is exactly the objection D-019 raised
  when it refused object-level scoring as the headline, and it applies with full
  force to using it as a selector.

**Pixel residual satisfies both.** "How many of the 4,096 pixels did the
prediction get wrong" is measured in pixels, means the same thing under every
carving, and is graded rather than binary. This script records all three keys on
the validation split so selection rules can be compared offline from the JSON
without re-running the hour-long sweep.

Usage:
    python scripts/eval_carving_selection.py [--corpus corpus] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from arc3.client.replay import read_episode
from arc3.dynamics import induce, observe_transition, score_transitions
from arc3.dynamics.predict import bookkeeping_mask
from arc3.perception.geometry import induce_geometry

ALL_CARVINGS = (
    "colour_4",
    "colour_8",
    "foreground_4",
    "foreground_8",
    "colour_mass",
    "bbox",
)
STARVED = {"ft09", "lp85"}


def _fit_and_score(train, test, carving, geo):
    obs = [observe_transition(t, carving, geo) for t in train]
    rs = induce(obs)
    return score_transitions(test, rs, carving, geo, ignore=bookkeeping_mask(train))


def run_game(path: Path, train_frac: float = 0.7, inner_frac: float = 0.75) -> dict:
    ep = read_episode(path)
    trans = [t for t in ep.transitions() if t.action.id != 0]
    if len(trans) < 20:
        return {}
    geo = induce_geometry(ep.grids()[:40])
    cut = max(1, int(len(trans) * train_frac))
    train, test = trans[:cut], trans[cut:]
    icut = max(1, int(len(train) * inner_frac))
    inner, val = train[:icut], train[icut:]

    rec: dict = {"game": path.name.split(".")[0], "n_val": len(val), "n_test": len(test)}
    val_scores, test_scores = {}, {}
    for c in ALL_CARVINGS:
        # Selection signal: validation only. The test set is never consulted.
        v = _fit_and_score(inner, val, c, geo)
        t = _fit_and_score(train, test, c, geo)
        val_scores[c] = v
        test_scores[c] = t
        # Every candidate selection key, recorded so rules can be compared
        # offline rather than by re-running the sweep for each one.
        rec[f"val_exact_{c}"] = v.exact_on_changed
        rec[f"val_chgobj_{c}"] = v.changed_object_accuracy
        rec[f"val_medres_{c}"] = v.median_residual
        rec[f"val_meanres_{c}"] = float(np.mean(v.residuals)) if v.residuals else 0.0

    # Primary rule: lowest median pixel residual on validation. Dense, and it
    # means the same thing under every carving. Ties broken by mean residual.
    chosen = min(
        ALL_CARVINGS,
        key=lambda c: (val_scores[c].median_residual, rec[f"val_meanres_{c}"]),
    )
    oracle = max(ALL_CARVINGS, key=lambda c: test_scores[c].exact_on_changed)
    rec["chosen"] = chosen
    rec["oracle"] = oracle
    rec["selected_exact"] = test_scores[chosen].exact_on_changed
    rec["oracle_exact"] = test_scores[oracle].exact_on_changed
    rec["selected_chgobj"] = test_scores[chosen].changed_object_accuracy
    rec["oracle_chgobj"] = test_scores[oracle].changed_object_accuracy
    for c in ALL_CARVINGS:
        rec[f"test_{c}"] = test_scores[c].exact_on_changed
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    logging.disable(logging.INFO)

    paths = sorted(Path(args.corpus).glob("*.jsonl.gz"))
    rows = []
    t0 = time.time()
    hdr = f"{'game':6s} {'chosen':>14s} {'sel':>7s} {'oracle':>7s} {'oracle carving':>15s}"
    print(f"corpus={args.corpus}  selection on a held-out inner split\n")
    print(hdr)
    print("-" * len(hdr))
    for p in paths:
        r = run_game(p, args.train_frac)
        if not r or r["game"] in STARVED:
            continue
        rows.append(r)
        flag = "" if r["chosen"] == r["oracle"] else "  <-- missed"
        print(
            f"{r['game']:6s} {r['chosen']:>14s} {100 * r['selected_exact']:6.1f}% "
            f"{100 * r['oracle_exact']:6.1f}% {r['oracle']:>15s}{flag}"
        )

    if not rows:
        return
    print("-" * len(hdr))
    best_single = max(
        ALL_CARVINGS, key=lambda c: float(np.mean([r[f"test_{c}"] for r in rows]))
    )
    single = float(np.mean([r[f"test_{best_single}"] for r in rows]))
    sel = float(np.mean([r["selected_exact"] for r in rows]))
    orc = float(np.mean([r["oracle_exact"] for r in rows]))

    def n_above(key):
        return sum(1 for r in rows if r[key] > 0)

    print(f"\n{len(rows)} games, {time.time() - t0:.0f}s\n")
    print(f"{'strategy':28s} {'mean exact':>11s} {'games >0':>9s}")
    print(f"{'best single (' + best_single + ')':28s} {100 * single:10.1f}% "
          f"{sum(1 for r in rows if r[f'test_{best_single}'] > 0):5d}/{len(rows):<3d}")
    print(f"{'validation-selected':28s} {100 * sel:10.1f}% {n_above('selected_exact'):5d}/{len(rows):<3d}")
    print(f"{'oracle (cheating)':28s} {100 * orc:10.1f}% {n_above('oracle_exact'):5d}/{len(rows):<3d}")
    agree = sum(1 for r in rows if r["chosen"] == r["oracle"])
    print(f"\nvalidation picked the oracle carving on {agree}/{len(rows)} games")
    if orc > single:
        recovered = (sel - single) / (orc - single)
        print(f"fraction of the available headroom recovered: {100 * recovered:.0f}%")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
