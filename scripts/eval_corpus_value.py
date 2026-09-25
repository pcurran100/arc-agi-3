"""Which corpus teaches the better model, judged on a common test set?

Every cross-corpus comparison so far has been confounded, and the confound runs
the wrong way. `eval_dynamics.py` splits each corpus into its own train and test
halves, so a driver is scored on transitions it generated itself. A driver that
produces a *richer* corpus therefore sets itself a *harder* exam: the explore
corpus carries 42.3 distinct playfield events per game against random's 30.4,
and its exact-match score fell even as the number of games it could model at all
went up. Improving the explorer makes its own benchmark harder.

This removes the confound without recording anything new. For each game:

- build **one** test set, the concatenation of the held-out tails of *all three*
  corpora, so it contains states reached by every driver and is neutral between
  them;
- train a model on each corpus's own training split;
- score all of them on that same common set.

A fixed test set drawn from any single driver would favour whichever driver
generated it. A union favours none, and it also asks the more useful question:
does a corpus teach a model that generalises to states the agent did not itself
produce?

Usage:
    python scripts/eval_corpus_value.py --corpora corpus,corpus_obj,corpus_exp
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

STARVED = {"ft09", "lf52", "lp85", "s5i5", "sb26", "su15", "tn36", "vc33"}


def split(path: Path, train_frac: float):
    ep = read_episode(path)
    trans = [t for t in ep.transitions() if t.action.id != 0]
    cut = max(1, int(len(trans) * train_frac))
    return ep, trans[:cut], trans[cut:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="corpus,corpus_obj,corpus_exp")
    ap.add_argument("--carving", default="colour_4")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    logging.disable(logging.INFO)

    corpora = args.corpora.split(",")
    games = sorted(p.name.split(".")[0] for p in Path(corpora[0]).glob("*.jsonl.gz"))

    print(f"carving={args.carving}; every model scored on the SAME union test set\n")
    hdr = f"{'game':6s} {'test n':>7s} " + " ".join(f"{c[:13]:>14s}" for c in corpora)
    print(hdr)
    print("-" * len(hdr))

    rows = []
    t0 = time.time()
    for g in games:
        if g in ("ft09", "lp85"):
            continue
        try:
            parts = {c: split(Path(c) / f"{g}.jsonl.gz", args.train_frac) for c in corpora}
        except FileNotFoundError:
            continue
        # Geometry from the first corpus, so the representation is identical for
        # every model and only the training evidence differs.
        geo = induce_geometry(parts[corpora[0]][0].grids()[:40])
        common_test = [t for c in corpora for t in parts[c][2]]
        if not common_test:
            continue

        rec = {"game": g, "n_test": len(common_test)}
        cells = []
        for c in corpora:
            train = parts[c][1]
            rs = induce([observe_transition(t, args.carving, geo) for t in train])
            sc = score_transitions(
                common_test, rs, args.carving, geo, ignore=bookkeeping_mask(train)
            )
            rec[f"exact_{c}"] = sc.exact_on_changed
            rec[f"chgobj_{c}"] = sc.changed_object_accuracy
            rec[f"rules_{c}"] = len(rs.rules)
            cells.append(f"{100 * sc.exact_on_changed:13.1f}%")
        rows.append(rec)
        star = "*" if g in STARVED else " "
        print(f"{g:5s}{star} {rec['n_test']:7d} " + " ".join(cells))

    if not rows:
        return
    print("-" * len(hdr))

    def agg(c, subset=None):
        sel = [r for r in rows if subset is None or r["game"] in subset]
        return (
            float(np.mean([r[f"exact_{c}"] for r in sel])),
            sum(1 for r in sel if r[f"exact_{c}"] > 0),
            len(sel),
        )

    for label, subset in (("ALL", None), ("STARVED", STARVED)):
        cells = []
        for c in corpora:
            m, n, tot = agg(c, subset)
            cells.append(f"{100 * m:8.1f}% {n:2d}/{tot:<2d}")
        print(f"{label:6s} {'':7s} " + " ".join(f"{x:>14s}" for x in cells))

    print(f"\n{time.time() - t0:.0f}s   (mean exact, games>0)")
    best = max(corpora, key=lambda c: agg(c)[0])
    print(f"best corpus on a neutral test set: {best}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
