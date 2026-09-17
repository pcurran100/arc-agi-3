"""Ground-truth click map: which of the 4,096 pixels actually do anything?

**This is an offline measurement instrument, not an agent capability.** It
brute-forces every pixel by RESET-ing the level between clicks, which is free
offline and impossible at evaluation, where every click is permanent (D-009:
there is no Undo). Treat its output exactly like the sprite oracle (D-010):
ground truth to score perception against, never an input to the agent. The
boundary is enforced in `tests/test_oracle_boundary.py` (D-021).

Three questions, all of which `state-of-play.md` currently answers by guess:

1. **What fraction of clicks do anything?** D-020 measured ~90% no-ops on the
   click-only games under every driver tested and could not say whether that was
   the driver's fault or the game's. The click map settles it: if only 4% of
   pixels are live, a 90% no-op rate is the *game*, and no targeting policy
   short of knowing where the live pixels are will fix it.

2. **How many distinct outcomes are there?** 4,096 pixels do not mean 4,096
   effects. The number of distinct resulting frames is the true branching factor
   of the click action space, and therefore the number D-005's pruning should be
   compared against -- not the 8-196 candidate count, which is a proposal count
   with no ground truth attached.

3. **Does segmentation find them?** D-020 compared object-targeted clicks against
   a lattice sweep on a *proxy* (distinct changed-pixel signatures) and found
   them indistinguishable in aggregate. With the map in hand the question can be
   asked directly: of the targets segmentation proposes, how many are live, and
   do they reach every distinct outcome?

Usage:  python scripts/eval_clickmap.py [--games a,b,c] [--out FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from arc3.client.env import EnvClient, available_environments
from arc3.drive import lattice_targets, object_targets
from arc3.types import CLICK, GRID_H, GRID_W, Action

RESET = 0


def _hash(frame):
    """Stable identity for a resting frame, for counting distinct outcomes."""
    if frame is None:
        return "none"
    return hashlib.blake2b(np.ascontiguousarray(frame).tobytes(), digest_size=8).hexdigest()


def click_map(game_id: str, *, seed: int = 0) -> dict:
    """Click every pixel from the level-1 start state, RESET-ing between each.

    Returns the per-pixel outcome hash, the live mask, and the timing. Costs
    2 * 4,096 engine steps and no real actions, because the whole point is that
    this is not something the agent may do.
    """
    client = EnvClient(game_id, seed=seed, record=False)
    initial = client.current.last
    if initial is None:
        raise RuntimeError(f"{game_id}: no initial frame")
    initial = initial.copy()
    base_hash = _hash(initial)

    outcome = np.empty((GRID_H, GRID_W), dtype=object)
    live = np.zeros((GRID_H, GRID_W), dtype=bool)
    nframes = np.zeros((GRID_H, GRID_W), dtype=np.int16)
    changed_px = np.zeros((GRID_H, GRID_W), dtype=np.int32)

    t0 = time.time()
    for y in range(GRID_H):
        for x in range(GRID_W):
            obs = client.step(Action.click(x, y))
            frame = obs.last
            h = _hash(frame)
            outcome[y, x] = h
            nframes[y, x] = len(obs.frames)
            if frame is not None and h != base_hash:
                live[y, x] = True
                changed_px[y, x] = int(np.count_nonzero(frame != initial))
            # restore the level start; RESET exactness is checked after the sweep
            client.step(Action.simple(RESET))
    elapsed = time.time() - t0

    restored = np.array_equal(client.current.last, initial)

    return {
        "game_id": game_id,
        "initial": initial,
        "base_hash": base_hash,
        "outcome": outcome,
        "live": live,
        "nframes": nframes,
        "changed_px": changed_px,
        "reset_exact": bool(restored),
        "seconds": elapsed,
    }


def analyse(m: dict) -> dict:
    """Turn a raw click map into the three numbers the experiment is for."""
    live = m["live"]
    outcome = m["outcome"]
    initial = m["initial"]
    total = GRID_H * GRID_W

    live_pixels = [(x, y) for y in range(GRID_H) for x in range(GRID_W) if live[y, x]]
    classes: dict = {}
    for x, y in live_pixels:
        classes.setdefault(outcome[y, x], []).append((x, y))

    def score(targets) -> dict:
        """Precision on live pixels, and coverage of distinct outcome classes."""
        targets = list(targets)
        if not targets:
            return {"n": 0, "precision": 0.0, "class_coverage": 0.0, "classes_hit": 0}
        hit = [(x, y) for (x, y) in targets if 0 <= x < GRID_W and 0 <= y < GRID_H and live[y, x]]
        reached = {outcome[y, x] for (x, y) in hit}
        return {
            "n": len(targets),
            "precision": len(hit) / len(targets),
            "class_coverage": (len(reached) / len(classes)) if classes else 0.0,
            "classes_hit": len(reached),
        }

    return {
        "game_id": m["game_id"],
        "live_pixels": len(live_pixels),
        "live_frac": len(live_pixels) / total,
        "n_classes": len(classes),
        "largest_class": max((len(v) for v in classes.values()), default=0),
        "singleton_classes": sum(1 for v in classes.values() if len(v) == 1),
        "mean_changed_px": (float(np.mean(m["changed_px"][live])) if len(live_pixels) else 0.0),
        "objects": score(object_targets(initial)),
        "lattice": score(lattice_targets(8)),
        "reset_exact": m["reset_exact"],
        "seconds": m["seconds"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="", help="comma-separated ids; default = all click games")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    logging.disable(logging.INFO)

    if args.games:
        games = args.games.split(",")
    else:
        games = []
        for g in available_environments():
            c = EnvClient(g, seed=args.seed, record=False)
            if CLICK in c.current.available_actions:
                games.append(g)

    print(f"{len(games)} games publish the click at level 1, seed={args.seed}\n")
    header = (
        f"{'game':16s} {'live%':>6s} {'classes':>8s} {'largest':>8s} "
        f"{'obj n':>6s} {'obj prec':>9s} {'obj cov':>8s} "
        f"{'lat prec':>9s} {'lat cov':>8s} {'s':>5s}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for g in games:
        r = analyse(click_map(g, seed=args.seed))
        rows.append(r)
        print(
            f"{g[:16]:16s} {100 * r['live_frac']:6.1f} {r['n_classes']:8d} "
            f"{r['largest_class']:8d} {r['objects']['n']:6d} "
            f"{100 * r['objects']['precision']:8.1f}% {100 * r['objects']['class_coverage']:7.1f}% "
            f"{100 * r['lattice']['precision']:8.1f}% {100 * r['lattice']['class_coverage']:7.1f}% "
            f"{r['seconds']:5.1f}"
        )
        if not r["reset_exact"]:
            print(f"  !! {g}: RESET did not restore the start state; row is unsound")

    if rows:
        print("-" * len(header))
        print(
            f"{'mean':16s} {100 * np.mean([r['live_frac'] for r in rows]):6.1f} "
            f"{np.mean([r['n_classes'] for r in rows]):8.1f} "
            f"{np.mean([r['largest_class'] for r in rows]):8.1f} "
            f"{np.mean([r['objects']['n'] for r in rows]):6.1f} "
            f"{100 * np.mean([r['objects']['precision'] for r in rows]):8.1f}% "
            f"{100 * np.mean([r['objects']['class_coverage'] for r in rows]):7.1f}% "
            f"{100 * np.mean([r['lattice']['precision'] for r in rows]):8.1f}% "
            f"{100 * np.mean([r['lattice']['class_coverage'] for r in rows]):7.1f}%"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
