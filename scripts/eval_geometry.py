"""Score induced geometry against the engine's true camera geometry.

Usage::

    python scripts/eval_geometry.py [--actions N] [--seed S]

Uses the D-010 oracle, so this is an evaluation script and never runs on the
agent's path. Reports two different things, because they are different claims:

- **scale agreement** - does the induced cell size match the camera's upscale
  factor? Only meaningful where the camera actually paints the frame. On games
  whose UI layer covers the camera view, the camera scale is not recoverable
  *and not useful*, so those are reported separately rather than counted as
  failures.
- **soundness** - is every induced cell genuinely one colour, and does the
  induced playfield contain the pixels that actually change? These need no
  oracle and are the properties downstream code relies on.
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from arc3.client.env import EnvClient, available_environments
from arc3.oracle import UI, ground_truth, true_geometry
from arc3.perception.geometry import induce_geometry, verify_cell
from arc3.types import Action


def rollout_frames(client: EnvClient, n_actions: int, rng: random.Random) -> list[np.ndarray]:
    frames = list(client.current.frames)
    for _ in range(n_actions):
        acts = client.current.available_actions
        if not acts or client.current.state.is_terminal:
            break
        a = rng.choice(acts)
        action = (
            Action.click(rng.randrange(64), rng.randrange(64)) if a == 6 else Action.simple(a)
        )
        frames.extend(client.step(action).frames)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hdr = (
        f"{'game':<16}{'true':<6}{'induced':<9}{'cover%':<8}{'uigrab%':<9}"
        f"{'cells':<10}{'sound':<7}{'verdict'}"
    )
    print(hdr)
    print("-" * len(hdr))

    agree = comparable = unsound = 0
    rows = []
    for gid in available_environments():
        rng = random.Random(args.seed)
        client = EnvClient(gid, seed=args.seed, record=False)
        truth = true_geometry(client._engine_game)
        gt = ground_truth(client._engine_game, rendered=client.current.last)
        ui_frac = float((gt.instance_map == UI).mean())

        frames = rollout_frames(client, args.actions, rng)
        geo = induce_geometry(frames)

        # Soundness: every induced cell must be a single colour in every frame,
        # judged over the region the induction actually claims to model.
        changed = None
        if len(frames) > 1:
            st = np.stack(frames)
            ch = (st != st[0]).any(axis=0)
            changed = ch if ch.any() else None
        sound = verify_cell(
            frames, geo.block.cell, geo.block.origin_x, geo.block.origin_y, changed
        )
        if not sound:
            unsound += 1

        induced = geo.block.cell
        cover = 100.0 * geo.block.rect.area / (64 * 64) if geo.block.is_structured else 0.0
        n_cells = geo.block.rows * geo.block.cols if geo.block.is_structured else 0
        if geo.block.was_reduced:
            n_cells = -n_cells  # flag: GCD proposal failed verification

        # The camera scale is only a fair target where the camera is visible.
        ui_dominated = ui_frac > 0.5
        if ui_dominated:
            verdict = "ui-dominated"
        else:
            comparable += 1
            if induced == truth["scale"]:
                agree += 1
                verdict = "match"
            else:
                verdict = "MISMATCH"

        rows.append((gid, truth["scale"], induced, verdict))
        print(
            f"{gid:<16}{truth['scale']:<6}{induced:<9}{cover:<8.1f}{100 * ui_frac:<9.1f}"
            f"{n_cells:<10}{'ok' if sound else 'BAD':<7}{verdict}"
        )

    print()
    print(f"scale agreement on camera-visible games: {agree}/{comparable}")
    print(f"ui-dominated games (camera scale not a fair target): {len(rows) - comparable}")
    print(f"unsound induced grids (a cell held >1 colour): {unsound}")


if __name__ == "__main__":
    main()
