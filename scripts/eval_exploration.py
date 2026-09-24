"""Do the drivers make the game move? Scored on playfield activity, not frames.

The metric D-028 introduced and D-030 corrected: the fraction of actions that
change at least one pixel **outside the bookkeeping band**. Frame change is the
wrong target and the difference is not academic — vc33 changed its frame on 99
of 99 actions under random play and its playfield on **zero** of them, because
every one of those changes was a move counter ticking.

This is the first evaluation in the repository that scores a policy on whether
it is *doing something*, rather than on the corpus it leaves behind. It is also
the measurement objective 1 of D-028 exists to move.

The band is computed **per driver from that driver's own episode**, because an
agent has no access to anyone else's recording. That makes the comparison
slightly harder than it looks: a driver that provokes more playfield change also
gives the band estimator more to work with.

Reports the eight games D-028/D-030 identified as playfield-starved separately,
since they are where the headroom is and where the aggregate would hide it.

Usage:
    python scripts/eval_exploration.py [--actions 100] [--drivers random,...]
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

from arc3.client.env import EnvClient, available_environments, make_arcade
from arc3.drive import make_driver
from arc3.dynamics.predict import bookkeeping_mask
from arc3.types import Action

#: D-028's eight, as re-measured by D-030 on the object-targeted corpus.
STARVED = ("ft09", "lf52", "lp85", "s5i5", "sb26", "su15", "tn36", "vc33")

DRIVERS = ("random", "random_objects", "explore")


def run(gid: str, driver_name: str, n_actions: int, seed: int, arcade=None) -> dict:
    client = EnvClient(gid, seed=seed, record=True, arcade=arcade)
    assert client.episode is not None
    driver = make_driver(driver_name, seed=seed)
    resets = 0
    for _ in range(n_actions):
        a = driver.next_action(client.current)
        if a is None:
            client.step(Action.simple(0))
            resets += 1
            if driver.next_action(client.current) is None:
                break
            continue
        client.step(a)

    trans = [t for t in client.episode.transitions() if t.action.id != 0]
    if not trans:
        return {}
    # The band this driver could have learned from its own episode.
    band = bookkeeping_mask(trans)
    changed = [t for t in trans if not t.is_noop]
    playfield = [t for t in changed if ((t.before != t.after) & ~band).any()]
    px = [int(((t.before != t.after) & ~band).sum()) for t in playfield]
    return {
        "game": gid[:4],
        "driver": driver_name,
        "actions": len(trans),
        "changed": len(changed),
        "playfield": len(playfield),
        "playfield_rate": len(playfield) / len(trans),
        "mean_playfield_px": float(np.mean(px)) if px else 0.0,
        "levels": client.episode.final.levels_completed,
        "resets": resets,
        "band_px": int(band.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drivers", default=",".join(DRIVERS))
    ap.add_argument("--games", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    logging.disable(logging.INFO)

    drivers = args.drivers.split(",")
    games = available_environments()
    if args.games:
        want = set(args.games.split(","))
        games = [g for g in games if g[:4] in want]

    arcade = make_arcade()
    rows: list[dict] = []
    t0 = time.time()
    hdr = f"{'game':6s} " + " ".join(f"{d[:13]:>14s}" for d in drivers)
    print(f"playfield activity rate, {args.actions} actions, seed {args.seed}")
    print("(* = playfield-starved under random play, D-028/D-030)\n")
    print(hdr)
    print("-" * len(hdr))
    for gid in games:
        cells = []
        for d in drivers:
            r = run(gid, d, args.actions, args.seed, arcade=arcade)
            if r:
                rows.append(r)
                cells.append(f"{100 * r['playfield_rate']:13.0f}%")
            else:
                cells.append(f"{'-':>14s}")
        star = "*" if gid[:4] in STARVED else " "
        print(f"{gid[:4]:5s}{star} " + " ".join(cells))

    if not rows:
        return
    print("-" * len(hdr))

    def mean(driver, subset=None):
        sel = [r for r in rows if r["driver"] == driver and (subset is None or r["game"] in subset)]
        return float(np.mean([r["playfield_rate"] for r in sel])) if sel else 0.0

    for label, subset in (("ALL", None), ("STARVED", set(STARVED))):
        print(f"{label:6s} " + " ".join(f"{100 * mean(d, subset):13.0f}%" for d in drivers))

    print(f"\n{time.time() - t0:.0f}s")
    best = max(drivers, key=lambda d: mean(d, set(STARVED)))
    print(f"best on the starved games: {best} at {100 * mean(best, set(STARVED)):.0f}% "
          f"(random: {100 * mean('random', set(STARVED)):.0f}%)")
    lv = {d: sum(r["levels"] for r in rows if r["driver"] == d) for d in drivers}
    print(f"levels cleared: {lv}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
