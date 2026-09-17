"""How much does dynamics induction improve with more actions?

The sweep (`eval_dynamics.py`) answers "does it beat chance". This answers the
question that follows immediately and matters for the action budget: **is the
model data-starved or language-limited?**

The two have opposite consequences. If accuracy climbs with training
transitions, the binding constraint is evidence, and D-022's exploration
budget is the thing to spend on — more actions buy a better model. If it
plateaus early, more actions buy nothing and the binding constraint is the rule
language, so the next move is expressiveness (D-024's rejected options), not
exploration. Under RHAE that distinction is worth a lot: actions are
quadratically expensive and spending them on a model that has stopped learning
is the worst available trade.

Test set is held fixed at the tail of the episode for every training size, so
the curve is not confounded by a moving target.

Read the **saturation point**, not the endpoints. A curve that rises steeply to
n=50 and is then flat to n=200 has the same endpoints as one still climbing, and
they mean opposite things.

Usage:
    python scripts/record_corpus.py --actions 300 --out corpus300
    python scripts/eval_dynamics_curve.py --carving foreground_8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from arc3.client.replay import read_episode
from arc3.dynamics import induce, observe_transition, score_transitions
from arc3.dynamics.predict import bookkeeping_mask
from arc3.perception.geometry import induce_geometry

SIZES = (10, 25, 50, 100, 200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus300")
    ap.add_argument("--carving", default="foreground_8")
    ap.add_argument("--test-tail", type=int, default=50)
    ap.add_argument("--games", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    logging.disable(logging.INFO)

    paths = sorted(Path(args.corpus).glob("*.jsonl.gz"))
    if args.games:
        wanted = set(args.games.split(","))
        paths = [p for p in paths if p.name.split(".")[0] in wanted]
    if not paths:
        raise SystemExit(f"no corpus in {args.corpus}/")

    print(f"carving={args.carving}  test=last {args.test_tail} transitions  {len(paths)} games\n")
    hdr = f"{'game':6s}" + "".join(f"{('n=' + str(n)):>10s}" for n in SIZES)
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for path in paths:
        ep = read_episode(path)
        trans = [t for t in ep.transitions() if t.action.id != 0]
        if len(trans) < max(SIZES) + args.test_tail:
            continue
        geo = induce_geometry(ep.grids()[:40])
        test = trans[-args.test_tail :]
        pool = trans[: -args.test_tail]
        cells = []
        rec = {"game": path.name.split(".")[0], "carving": args.carving}
        for n in SIZES:
            train = pool[:n]
            obs = [observe_transition(t, args.carving, geo) for t in train]
            rs = induce(obs)
            sc = score_transitions(test, rs, args.carving, geo, ignore=bookkeeping_mask(train))
            rec[f"exact_{n}"] = sc.exact_on_changed
            rec[f"chgobj_{n}"] = sc.changed_object_accuracy
            cells.append(f"{100 * sc.exact_on_changed:9.1f}%")
        rows.append(rec)
        print(f"{rec['game']:6s}" + "".join(cells))

    if rows:
        print("-" * len(hdr))
        means = [float(np.mean([r[f"exact_{n}"] for r in rows])) for n in SIZES]
        print(f"{'MEAN':6s}" + "".join(f"{100 * m:9.1f}%" for m in means))
        cmeans = [float(np.mean([r[f"chgobj_{n}"] for r in rows])) for n in SIZES]
        print(f"{'chgobj':6s}" + "".join(f"{100 * m:9.1f}%" for m in cmeans))
        # Saturation, not first-vs-last. The first version of this compared the
        # endpoints and reported "still climbing" for a curve that was flat from
        # n=50 onward: 7.1% -> 13.5% looks like growth, but every point from 50
        # to 200 was 13.5% to three significant figures. Endpoint comparison
        # cannot distinguish a rising curve from one that rose early and
        # stopped, which is the entire question being asked.
        final = means[-1]
        saturation = next(
            (n for n, v in zip(SIZES, means) if final > 0 and v >= 0.98 * final), SIZES[-1]
        )
        tail_gain = means[-1] - means[len(SIZES) // 2]
        still_rising = tail_gain > 0.01
        verdict = (
            "EVIDENCE-LIMITED: the curve is still rising in its second half, so "
            "more actions buy a better model"
            if still_rising
            else f"LANGUAGE-LIMITED: saturates at n={saturation}, and the second "
            f"half of the curve adds {100 * tail_gain:+.1f}pp. More actions buy "
            f"nothing; expressiveness is the binding constraint"
        )
        print(f"\n{SIZES[0]} -> {SIZES[-1]} transitions: "
              f"{100 * means[0]:.1f}% -> {100 * means[-1]:.1f}%")
        print(f"saturation at n={saturation};  second-half gain {100 * tail_gain:+.2f}pp")
        print(f"\n{verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
