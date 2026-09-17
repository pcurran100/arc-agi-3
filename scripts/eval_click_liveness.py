"""Does click liveness depend on the state, or is it a property of the game?

The follow-up to `eval_clickmap.py`, which maps every pixel but only at the
level-1 start state. That left the obvious question open: when the map says no
pixel does anything (four games, D-021 Result 3), is the click **dead in this
game** or merely **dead in this state**? The two have completely different
consequences — the first licenses writing the action off, the second makes doing
so a Guardrail 1 violation.

Cheap by design. Rather than re-mapping all 4,096 pixels at each depth, it walks
a random rollout and at several checkpoints tests only a candidate set (the
object targets plus a 16-pixel lattice), asking the much weaker question *does
anything here do anything*. That costs ~2,400 engine steps per game instead of
~330,000.

Reaching a depth-*k* state means `RESET` plus replaying *k* actions, which is
exact because the engine is deterministic under a fixed seed and `RESET`
restores the level start byte-for-byte (verified in D-021). Like the click map,
this is an **offline instrument only** (D-010 footing): at evaluation those
replays are permanent actions.

Reports the percentage of candidate targets that change the frame, per game per
checkpoint. A `-` means the rollout terminated before that depth.

Usage:  python scripts/eval_click_liveness.py
"""
import logging
import sys

logging.disable(logging.INFO)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from arc3.client.env import EnvClient, available_environments
from arc3.drive import RandomDriver, lattice_targets, object_targets
from arc3.types import CLICK, Action

CHECKPOINTS = [0, 5, 10, 20, 40]


def rollout_actions(game, n, seed=0):
    """The fixed action sequence a random driver takes; deterministic given seed."""
    c = EnvClient(game, record=False)
    d = RandomDriver(seed=seed)
    acts = []
    for _ in range(n):
        a = d.next_action(c.current)
        if a is None:
            break
        acts.append(a)
        c.step(a)
    return acts


def replay(client, acts):
    client.step(Action.simple(0))
    for a in acts:
        client.step(a)
    return client.current.last


def main():
    games = []
    for g in available_environments():
        c = EnvClient(g, record=False)
        if CLICK in c.current.available_actions:
            games.append(g)

    print(f"{'game':6s} " + " ".join(f"k={k:<5d}" for k in CHECKPOINTS))
    print("-" * 46)
    for g in games:
        acts = rollout_actions(g, max(CHECKPOINTS))
        c = EnvClient(g, record=False)
        cells = []
        for k in CHECKPOINTS:
            if k > len(acts):
                cells.append("  -  ")
                continue
            base = replay(c, acts[:k])
            if base is None:
                cells.append("  -  ")
                continue
            base = base.copy()
            targets = list(object_targets(base)) + lattice_targets(16)
            live = 0
            for (x, y) in targets:
                replay(c, acts[:k])
                o = c.step(Action.click(int(x), int(y)))
                if o.last is not None and not np.array_equal(o.last, base):
                    live += 1
            cells.append(f"{100*live/len(targets):4.0f}%")
        print(f"{g[:4]:6s} " + " ".join(f"{s:6s}" for s in cells))


if __name__ == "__main__":
    main()
