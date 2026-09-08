"""Record short episodes as test fixtures.

Perception must be testable without the engine and without spending an action
(D-008), so a handful of real recorded episodes live in ``fixtures/`` and the
test suite runs against those. Regenerate with::

    python scripts/record_fixtures.py [--actions N] [--seed S]

The games chosen span the cases that broke things during development: an
upscaled camera, a UI-dominated frame, a click-only game, a keyboard-only game,
and one with long animated transitions.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from arc3.client.env import EnvClient
from arc3.client.replay import write_episode
from arc3.types import Action

#: One game per structural case we care about, with the reason recorded so a
#: later reader knows why the list is what it is.
FIXTURE_GAMES = {
    "ls20": "keyboard-only, 1:1 camera, many sprites",
    "sp80": "4x upscaled camera, small board",
    "bp35": "8x camera but UI-rendered; the case that broke global scale induction",
    "vc33": "click-only, 2x camera",
    "sb26": "long animated transitions (42 frames observed for one action)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fixtures")
    args = ap.parse_args()

    out_dir = Path(args.out)
    for game, why in FIXTURE_GAMES.items():
        rng = random.Random(args.seed)
        client = EnvClient(game, seed=args.seed, record=True)
        assert client.episode is not None
        client.episode.notes = {"why": why, "recorded_by": "scripts/record_fixtures.py"}
        for _ in range(args.actions):
            acts = client.current.available_actions
            if not acts or client.current.state.is_terminal:
                break
            a = rng.choice(acts)
            client.step(
                Action.click(rng.randrange(64), rng.randrange(64))
                if a == 6
                else Action.simple(a)
            )
        path = write_episode(client.episode, out_dir / f"{game}.jsonl.gz")
        size_kb = path.stat().st_size / 1024
        print(
            f"{game:<6} {client.actions_used:>3} actions, "
            f"{len(client.episode.observations):>3} observations, "
            f"{sum(len(o.frames) for o in client.episode.observations):>4} frames, "
            f"{size_kb:>7.0f} KB  {why}"
        )


if __name__ == "__main__":
    main()
