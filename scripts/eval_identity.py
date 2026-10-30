"""In-place identity stability of the agent's fact layer (D-046).

Usage::

    PYTHONPATH=src python scripts/eval_identity.py [--actions N] [--seed S] [--games g1 g2 ...]

The D-010 oracle (``eval_tracking.py``) does not score the UI layer, and the
identity failures that hurt the v4 runs were exactly there: a step counter that
refills after a lost life swapped ids between its spent and live parts (ls20),
and a growing bar was reported as "moved (-1,+0)" (tn36). This script measures
those directly on the agent's own path (``FactBook5``/``FactBook4``):

- **in-place pairs**: an object in frame *t* and an object in frame *t+1* of the
  same colour whose cells overlap at least half of the smaller one. Physically
  these are the same thing, grown, shrunk or reshaped in place.
- **in-place swaps**: in-place pairs that carry different ids. Lower is better;
  the target is zero.
- **false moves**: changes reported as ``moved`` for an object whose old and new
  cells overlap and whose shape changed - a deformation reported as a motion.

Play is scripted, not sampled from a policy: per game, a deterministic mix of
random actions (seeded) and, for movement games, long runs of one action, which
exhausts per-action counters (that is when bars refill). The numbers depend on
the script; compare versions on the same seed only.
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from arc3.client.env import EnvClient, available_environments
from arc3.types import Action


def cells(o, mask):
    ys, xs = np.nonzero(mask)
    return {(o.x + int(c), o.y + int(r)) for r, c in zip(ys, xs)}


def script(client, rng, n):
    """Actions for one game: runs of one direction plus random clicks/moves."""
    acts = [a for a in client.current.available_actions if a not in (0,)]
    simple = [a for a in acts if a != 6]
    out = []
    while len(out) < n:
        if simple and rng.random() < 0.5:
            a = rng.choice(simple)
            out += [Action.simple(a)] * rng.randint(3, 15)
        elif 6 in acts:
            out.append(Action.click(rng.randrange(64), rng.randrange(64)))
        elif simple:
            out.append(Action.simple(rng.choice(simple)))
        else:
            break
    return out[:n]


def run(gid, n, seed, book_cls):
    client = EnvClient(gid, seed=seed, record=False)
    rng = random.Random(seed)
    fb = book_cls()
    fb.observe(client.current.last, None)
    pairs = swaps = false_moves = moves = 0
    for a in script(client, rng, n):
        if client.current.state.is_terminal:
            client.step(Action.simple(0))
            fb.observe(client.current.last, Action.simple(0))
            continue
        before, bmasks = dict(fb.objects), dict(fb.masks)
        obs = client.step(a)
        if obs.last is None:
            continue
        step = fb.observe(obs.last, a)
        after, amasks = fb.objects, fb.masks
        acells = {i: cells(o, amasks[i]) for i, o in after.items() if i in amasks}
        for i, o in before.items():
            if i not in bmasks:
                continue
            bc = cells(o, bmasks[i])
            for j, p in after.items():
                if p.colour != o.colour or j not in acells:
                    continue
                ov = len(bc & acells[j])
                if ov and ov * 2 >= min(len(bc), len(acells[j])):
                    pairs += 1
                    swaps += i != j
        if step is not None:
            for c in step.changes:
                if "moved" not in c.kinds:
                    continue
                moves += 1
                b, aa = step.before.get(c.id), step.after.get(c.id)
                if b is None or aa is None or c.id not in bmasks or c.id not in amasks:
                    continue
                if b.shape != aa.shape and cells(b, bmasks[c.id]) & cells(aa, amasks[c.id]):
                    false_moves += 1
    return pairs, swaps, moves, false_moves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--book", default="v5", choices=["v4", "v5"])
    args = ap.parse_args()
    if args.book == "v4":
        from arc3.v4.facts import FactBook4 as Book
    else:
        from arc3.v5.facts import FactBook5 as Book
    games = args.games or available_environments()
    tp = ts = tm = tf = 0
    print(f"{'game':<8}{'pairs':>7}{'swaps':>7}{'swap%':>7}{'moves':>7}{'false':>7}")
    for g in games:
        p, s, m, f = run(g, args.actions, args.seed, Book)
        tp, ts, tm, tf = tp + p, ts + s, tm + m, tf + f
        print(f"{g:<8}{p:>7}{s:>7}{100 * s / max(p, 1):>7.1f}{m:>7}{f:>7}")
    print(f"\nbook {args.book}, {args.actions} actions, seed {args.seed}: in-place swaps "
          f"{ts}/{tp} ({100 * ts / max(tp, 1):.2f}%), false moves {tf}/{tm}")


if __name__ == "__main__":
    main()
