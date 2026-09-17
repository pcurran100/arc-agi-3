"""Record the dynamics-induction corpus: every game, random play, seed 0.

Separate from `record_fixtures.py`, which records five short episodes for the
test suite. This records **all 25 games at length**, to be the evidence the
dynamics experiment induces from. Written once and replayed many times: the
engine is stepped here and nowhere else in the experiment, so re-running
induction, adding a seventh carving or changing the rule language costs no
engine time and no actions (D-019).

Driver is `random` (D-020, which measured round-robin and lattice against it and
found random better on event diversity).

**`RESET` on terminal, by default.** Random play ends games early — sp80 lasted
20 actions of 100, r11l 29 (it cleared a level), bp35 44 — which starves the
corpus on exactly the games that need evidence. Since `RESET` restores the level
start exactly (D-021), the recorder issues one and continues. The resulting
`RESET` transitions are real and learnable but are *discontinuities* with respect
to the action being studied, so they carry `action.id == 0` and the induction
code filters them out rather than pretending they are ordinary moves.

Usage:  python scripts/record_corpus.py [--actions 100] [--seed 0]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc3.client.env import EnvClient, available_environments
from arc3.client.replay import write_episode
from arc3.drive import make_driver
from arc3.types import Action


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--driver", default="random")
    ap.add_argument(
        "--no-reset-on-terminal",
        action="store_true",
        help="stop at the first terminal state instead of RESET-ing and continuing",
    )
    ap.add_argument("--out", default="corpus")
    args = ap.parse_args()
    logging.disable(logging.INFO)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_actions = 0
    t0 = time.time()
    print(
        f"{'game':16s} {'actions':>8s} {'frames':>8s} {'noop%':>7s} "
        f"{'resets':>7s} {'levels':>7s} {'KB':>7s}"
    )
    print("-" * 68)
    for game in available_environments():
        client = EnvClient(game, seed=args.seed, record=True)
        assert client.episode is not None
        driver = make_driver(args.driver, seed=args.seed)
        client.episode.notes = {
            "driver": args.driver,
            "recorded_by": "scripts/record_corpus.py",
            "purpose": "dynamics induction corpus",
            "reset_on_terminal": not args.no_reset_on_terminal,
        }
        resets = 0
        for _ in range(args.actions):
            action = driver.next_action(client.current)
            if action is None:
                # Terminal, or no legal action. RESET is exact (D-021), so
                # continue rather than abandon the rest of the budget.
                if args.no_reset_on_terminal:
                    break
                client.step(Action.simple(0))
                resets += 1
                if driver.next_action(client.current) is None:
                    break  # RESET did not restore a playable state
                continue
            client.step(action)

        ep = client.episode
        trans = [t for t in ep.transitions() if t.action.id != 0]
        noops = sum(1 for t in trans if t.is_noop)
        frames = sum(len(o.frames) for o in ep.observations)
        path = write_episode(ep, out_dir / f"{game[:4]}.jsonl.gz")
        total_frames += frames
        total_actions += ep.actions_used
        print(
            f"{game[:4]:16s} {ep.actions_used:8d} {frames:8d} "
            f"{100 * noops / max(1, len(trans)):6.1f}% "
            f"{resets:7d} {ep.final.levels_completed:7d} {path.stat().st_size / 1024:7.1f}"
        )

    print("-" * 68)
    print(
        f"{'total':16s} {total_actions:8d} {total_frames:8d}"
        f"   in {time.time() - t0:.0f}s -> {out_dir}/"
    )


if __name__ == "__main__":
    main()
