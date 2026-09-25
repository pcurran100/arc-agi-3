"""What do the unmodellable games actually do that the language cannot say?

D-032 established that five games — vc33, s5i5, tn36, lf52, sb26 — now produce
27 to 88 distinct playfield events each and still score **exactly 0%**. Their
blocker is representation, not evidence. This profiles *which* representation.

The effect vocabulary (D-024) has four members: an object holds still,
translates, recolours, or vanishes. Everything else is silently recorded as
`static` and then mispredicted. This script opens that silence and counts what
is inside it, so the next vocabulary extension is chosen from measurements
rather than from a guess about what games are like — which is exactly the
hazard Guardrail 2 names.

Categories, all mutually exclusive, computed per matched object pair or per
unmatched object:

- **translate** / **recolour** / **vanish** — expressible today.
- **appeared** — an object in the after-frame that no before-object maps to. No
  rule keyed to a before-object can ever produce it.
- **deformed-move** — moved *and* changed silhouette. Counted by D-025 at
  14-74% depending on carving.
- **grew** / **shrank** — same position, mask changed size. A bar filling, a
  trail lengthening, a region being eaten.
- **partial recolour** — mask identical, position identical, dominant colour
  identical, and some pixels changed anyway. The blindness D-030 found scoring
  as success.
- **reshaped** — same size, same position, different mask.

Usage:
    python scripts/eval_vocabulary_gap.py [--corpus corpus_exp] [--games ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from arc3.client.replay import read_episode
from arc3.dynamics.predict import bookkeeping_mask
from arc3.perception.geometry import induce_geometry
from arc3.perception.segment import segment
from arc3.perception.track import match_frames

#: The five D-032 named: abundant evidence, zero score.
UNMODELLABLE = ("vc33", "s5i5", "tn36", "lf52", "sb26")

EXPRESSIBLE = {"translate", "recolour", "vanish", "static"}


def classify(prev, nxt, before, after) -> str:
    dx = nxt.bbox.x0 - prev.bbox.x0
    dy = nxt.bbox.y0 - prev.bbox.y0
    same_size = prev.size == nxt.size
    same_box = prev.bbox.width == nxt.bbox.width and prev.bbox.height == nxt.bbox.height
    same_colour = prev.dominant_colour == nxt.dominant_colour

    if dx or dy:
        return "translate" if (same_size and same_box) else "deformed-move"
    if not same_size:
        return "grew" if nxt.size > prev.size else "shrank"
    if not same_box:
        return "reshaped"
    if not same_colour:
        return "recolour"
    if not np.array_equal(prev.mask, nxt.mask):
        return "reshaped"
    if not np.array_equal(before[prev.mask], after[prev.mask]):
        return "partial-recolour"
    return "static"


def profile(path: Path, carving: str) -> dict:
    ep = read_episode(path)
    trans = [t for t in ep.transitions() if t.action.id != 0]
    if not trans:
        return {}
    geo = induce_geometry(ep.grids()[:40])
    band = bookkeeping_mask(trans[: max(1, int(len(trans) * 0.7))])

    counts: Counter = Counter()
    n_playfield = 0
    for t in trans:
        if t.is_noop or not ((t.before != t.after) & ~band).any():
            continue
        n_playfield += 1
        sb = segment(t.before, geo, include=(carving,))[carving]
        sa = segment(t.after, geo, include=(carving,))[carving]
        m = match_frames(sb, sa)
        by_prev = m.by_prev()
        for i, comp in enumerate(sb.components):
            mm = by_prev.get(i)
            if mm is None:
                counts["vanish"] += 1
            else:
                counts[classify(comp, sa.components[mm.next_index], t.before, t.after)] += 1
        counts["appeared"] += len(m.appeared)
    return {
        "game": path.name.split(".")[0],
        "playfield_transitions": n_playfield,
        "counts": dict(counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus_exp")
    ap.add_argument("--carving", default="colour_4")
    ap.add_argument("--games", default=",".join(UNMODELLABLE))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    logging.disable(logging.INFO)

    order = [
        "static", "translate", "recolour", "vanish",
        "deformed-move", "grew", "shrank", "reshaped", "partial-recolour", "appeared",
    ]
    rows = []
    print(f"corpus={args.corpus} carving={args.carving}")
    print("expressible today: static translate recolour vanish\n")
    hdr = f"{'game':6s} {'pf':>4s} " + " ".join(f"{k[:9]:>10s}" for k in order)
    print(hdr)
    print("-" * len(hdr))
    for g in args.games.split(","):
        p = Path(args.corpus) / f"{g}.jsonl.gz"
        if not p.exists():
            continue
        r = profile(p, args.carving)
        if not r:
            continue
        rows.append(r)
        c = r["counts"]
        print(
            f"{r['game']:6s} {r['playfield_transitions']:4d} "
            + " ".join(f"{c.get(k, 0):10d}" for k in order)
        )

    if not rows:
        return
    print("-" * len(hdr))
    tot: Counter = Counter()
    for r in rows:
        tot.update(r["counts"])
    print(f"{'TOTAL':6s} {'':4s} " + " ".join(f"{tot.get(k, 0):10d}" for k in order))

    inexpressible = {k: v for k, v in tot.items() if k not in EXPRESSIBLE}
    n_in = sum(inexpressible.values())
    n_nonstatic = sum(v for k, v in tot.items() if k != "static")
    print(
        f"\nof {n_nonstatic} non-static object events, {n_in} ({100 * n_in / max(1, n_nonstatic):.0f}%) "
        f"cannot be expressed at all"
    )
    for k, v in sorted(inexpressible.items(), key=lambda kv: -kv[1]):
        print(f"   {k:18s} {v:6d}  ({100 * v / max(1, n_in):4.1f}% of the gap)")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
