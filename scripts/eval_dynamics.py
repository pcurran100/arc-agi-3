"""The dynamics-induction validation experiment: the go/no-go on D-002.

The question, unchanged since the design conversation: **can object tracking plus
simple rule induction predict held-out transitions better than chance?** If not,
design thesis section 4 is wrong and the model-based plan goes with it.

Metric (D-019): **exact resting-frame match on transitions where something
changed**, against a measured chance baseline of 0% — both trivial predictors,
static and per-action pixel replay, score exactly zero on changed transitions,
so any non-zero rate is real signal. The no-op fraction is reported separately
rather than allowed to inflate the headline.

Axis (D-018): **all six segmentations**, because a rule is stated in terms of
objects, six candidate notions of object exist, and nothing selects among them.
Fixing one arbitrarily would make a negative result uninterpretable — we could
not tell "the bet is wrong" from "we carved wrong".

Corpus (D-020): random play, recorded once by `scripts/record_corpus.py`.

**The split is temporal and within-game**, not across games, and this departs
from the letter of the instruction in `state-of-play.md` ("hold out by game, not
by transition"). The reason: dynamics are *per game*. Inducing on games 1..k and
testing on game k+1 would measure whether mechanics transfer between unrelated
games, which nothing in the thesis claims — section 9 claims persistence across
*levels within* an environment. What the instruction was protecting against is
real, though: pooling all transitions and splitting at random puts near-duplicate
transitions on both sides. A temporal split per game is the honest version, and
the protection against tuning-to-a-game is that no parameter is fitted per game.
Reported per game, aggregated over games.

Three secondary figures, each earning its place by having caught a real error:

- **exact ex-bookkeeping.** The 64x64 frame is a composite with a UI layer drawn
  last (benchmark.md correction #2). On ls20 a two-pixel move counter in rows
  61-62 ticks every action, so a *completely correct* model of the playfield
  scored 0% exact. This figure ignores an induced bookkeeping mask, and is
  flagged `vacuous` where a game changes nothing outside it.
- **changed-object accuracy.** Plain object accuracy is ~91% for predicting
  "everything is static", the same trap D-019 rejected per-pixel accuracy for.
- **deformed-move rate.** The fraction of real moves the effect vocabulary
  cannot express because the object changed silhouette as it moved. Measured at
  27-48% under colour_8, so this is the dominant known loss, and comparing it
  across carvings is the main diagnostic for *why* a carving wins or loses.

**Do not use this to compare drivers.** It splits each corpus into its own
train and test halves, so a driver is examined on transitions it generated
itself, and a driver that produces a *richer* corpus therefore sets itself a
*harder* exam. Measured: the explore corpus scored 3.8% here against random's
4.8%, and on a common test set it is **better** — 4.5% against 3.8% — with the
apparent per-game collapses (dc22 44.4%->15.0%) turning out to be identical
scores on identical questions (32.9% under all three corpora). See D-032 and
`eval_corpus_value.py`.

This script remains correct for comparing **carvings** or **learners** on a
fixed corpus, which is what it was built for.

Usage:
    python scripts/record_corpus.py            # once
    python scripts/eval_dynamics.py [--carvings colour_8,...] [--out FILE]
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
from arc3.dynamics.predict import (
    bookkeeping_mask,
    score_replay_baseline,
    score_static_baseline,
)
from arc3.perception.geometry import induce_geometry

ALL_CARVINGS = (
    "colour_4",
    "colour_8",
    "foreground_4",
    "foreground_8",
    "colour_mass",
    "bbox",
)

#: Games D-020 measured as corpus-starved: ~90% no-ops under every driver
#: tested, so they carry roughly a tenth of the evidence the others do and are
#: reported separately rather than averaged in.
STARVED = {"ft09", "lp85"}


def run_game(path: Path, carving: str, train_frac: float = 0.7) -> dict:
    ep = read_episode(path)
    # RESET transitions are discontinuities, not moves: excluded (record_corpus).
    trans = [t for t in ep.transitions() if t.action.id != 0]
    if len(trans) < 10:
        return {}
    geo = induce_geometry(ep.grids()[:40])
    cut = max(1, int(len(trans) * train_frac))
    train, test = trans[:cut], trans[cut:]

    t0 = time.time()
    obs = [observe_transition(t, carving, geo) for t in train]
    ruleset = induce(obs)
    mask = bookkeeping_mask(train)
    sc = score_transitions(test, ruleset, carving, geo, ignore=mask)
    b0 = score_static_baseline(test)
    b1 = score_replay_baseline(test)

    return {
        "game": path.name.split(".")[0],
        "carving": carving,
        "n_train": len(train),
        "n_test": sc.n_total,
        "n_changed": sc.n_changed,
        "n_changed_outside": sc.n_changed_outside,
        "noop_frac": sc.noop_fraction,
        "exact": sc.exact_on_changed,
        "exact_ex_bk": sc.exact_ex_bookkeeping,
        "vacuous": sc.vacuous,
        "median_residual": sc.median_residual,
        "obj_acc": sc.object_accuracy,
        "changed_obj_acc": sc.changed_object_accuracy,
        "deformed_move_rate": sc.deformed_move_rate,
        "n_rules": len(ruleset.rules),
        "appeared_blocked": sc.n_appeared_blocked,
        "bk_pixels": int(mask.sum()),
        "b0_exact": b0.exact_on_changed,
        "b1_exact": b1.exact_on_changed,
        "seconds": time.time() - t0,
    }


def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--carvings", default=",".join(ALL_CARVINGS))
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    logging.disable(logging.INFO)

    paths = sorted(Path(args.corpus).glob("*.jsonl.gz"))
    if not paths:
        raise SystemExit(f"no corpus in {args.corpus}/ - run scripts/record_corpus.py first")
    carvings = args.carvings.split(",")
    print(f"{len(paths)} games x {len(carvings)} carvings, train_frac={args.train_frac}\n")

    rows: list[dict] = []
    for carving in carvings:
        per_game: list[dict] = []
        t0 = time.time()
        print(f"=== {carving} ===")
        hdr = (
            f"{'game':6s} {'test':>5s} {'chg':>5s} {'out':>5s} {'exact':>7s} "
            f"{'ex-bk':>7s} {'medres':>7s} {'chgobj':>7s} {'deform':>7s} "
            f"{'rules':>6s} {'B0':>4s} {'B1':>4s}"
        )
        print(hdr)
        print("-" * len(hdr))
        for path in paths:
            r = run_game(path, carving, args.train_frac)
            if not r:
                continue
            per_game.append(r)
            rows.append(r)
            flag = " V" if r["vacuous"] else ""
            star = "*" if r["game"] in STARVED else " "
            print(
                f"{r['game']:5s}{star} {r['n_test']:5d} {r['n_changed']:5d} "
                f"{r['n_changed_outside']:5d} {100 * r['exact']:6.1f}% "
                f"{100 * r['exact_ex_bk']:6.1f}%{flag:2s} {r['median_residual']:6.1f} "
                f"{100 * r['changed_obj_acc']:6.1f}% {100 * r['deformed_move_rate']:6.1f}% "
                f"{r['n_rules']:6d} {100 * r['b0_exact']:3.0f}% {100 * r['b1_exact']:3.0f}%"
            )
        main_rows = [r for r in per_game if r["game"] not in STARVED]
        live = [r for r in main_rows if not r["vacuous"]]
        print("-" * len(hdr))
        print(
            f"{'MEAN':6s} {'':5s} {'':5s} {'':5s} {100 * _mean(main_rows, 'exact'):6.1f}% "
            f"{100 * _mean(live, 'exact_ex_bk'):6.1f}%   "
            f"{_mean(main_rows, 'median_residual'):6.1f} "
            f"{100 * _mean(main_rows, 'changed_obj_acc'):6.1f}% "
            f"{100 * _mean(main_rows, 'deformed_move_rate'):6.1f}%"
        )
        print(
            f"  games with exact > 0: {sum(1 for r in main_rows if r['exact'] > 0)}"
            f"/{len(main_rows)};  vacuous ex-bk: {sum(1 for r in main_rows if r['vacuous'])}"
            f";  {time.time() - t0:.0f}s\n"
        )

    print("=" * 72)
    print("HEADLINE: exact resting-frame match on changed transitions, vs chance 0%")
    print(f"{'carving':14s} {'mean exact':>11s} {'mean ex-bk':>11s} {'>0 games':>9s} {'deform':>8s}")
    for carving in carvings:
        rs = [r for r in rows if r["carving"] == carving and r["game"] not in STARVED]
        live = [r for r in rs if not r["vacuous"]]
        if not rs:
            continue
        print(
            f"{carving:14s} {100 * _mean(rs, 'exact'):10.1f}% {100 * _mean(live, 'exact_ex_bk'):10.1f}% "
            f"{sum(1 for r in rs if r['exact'] > 0):5d}/{len(rs):<3d} "
            f"{100 * _mean(rs, 'deformed_move_rate'):7.1f}%"
        )
    starved = [r for r in rows if r["game"] in STARVED]
    if starved:
        print(
            f"\nstarved games (ft09, lp85), reported separately per D-020: "
            f"mean exact {100 * _mean(starved, 'exact'):.1f}%, "
            f"mean no-op {100 * _mean(starved, 'noop_frac'):.0f}%"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
