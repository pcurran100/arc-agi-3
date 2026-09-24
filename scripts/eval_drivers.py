"""Compare driving policies by the quality of the transition corpus they produce.

Usage::

    python scripts/eval_drivers.py [--actions N] [--seed S]

The dynamics experiment learns from whatever transitions a driver generates, so
the driver's distribution is an input to the central go/no-go and deserves to be
chosen on evidence (D-019, D-020). This script measures what each policy
actually produces. It spends real actions and needs the engine.

What is measured, and why each one:

- **no-op rate** — actions that changed nothing. Pure waste in a corpus, and
  under RHAE pure waste in a score.
- **mean changed pixels** on transitions that did change. Random play's defect
  is that its changes are *tiny*, not absent (D-019), so this is the number that
  most directly says whether a policy produces richer evidence.
- **distinct effects** — how many *different* pixel-deltas were seen. A corpus of
  200 copies of the same event teaches one rule; this is the diversity that
  induction actually feeds on.
- **action coverage** — the fraction of legal actions tried at least once. The
  point of round-robin, and the thing uniform sampling leaves to luck.
- **levels** — cleared during the rollout. Not a target, but worth noticing.
"""

from __future__ import annotations

import argparse

import numpy as np

from arc3.client.env import EnvClient, available_environments
from arc3.drive import make_driver
from arc3.types import Action

DRIVERS = ("random", "random_objects", "round_robin", "lattice")


def run(gid: str, driver_name: str, n_actions: int, seed: int) -> dict:
    client = EnvClient(gid, seed=seed, record=False)
    driver = make_driver(driver_name, seed=seed)
    legal_seen: set[int] = set(a for a in client.current.available_actions if a != 0)
    tried: set[int] = set()
    noop = changed = 0
    deltas: set[bytes] = set()
    sizes: list[int] = []
    prev = client.current.last

    for _ in range(n_actions):
        action = driver.next_action(client.current)
        if action is None:
            break
        obs = client.step(action)
        tried.add(action.id)
        legal_seen |= set(a for a in obs.available_actions if a != 0)
        after = obs.last
        if after is None or prev is None:
            noop += 1
            if after is not None:
                prev = after
            continue
        d = prev != after
        if not d.any():
            noop += 1
        else:
            changed += 1
            sizes.append(int(d.sum()))
            # Identify the *kind* of event by its changed-pixel signature.
            deltas.add(d.tobytes())
        prev = after

    total = noop + changed
    return {
        "noop_pct": 100.0 * noop / total if total else float("nan"),
        "mean_px": float(np.mean(sizes)) if sizes else 0.0,
        "distinct": len(deltas),
        "coverage": 100.0 * len(tried & legal_seen) / len(legal_seen) if legal_seen else 100.0,
        "levels": client.current.levels_completed,
        "actions": client.actions_used,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    games = available_environments()
    agg: dict[str, list[dict]] = {d: [] for d in DRIVERS}

    hdr = f"{'game':<16}" + "".join(f"{d[:12]:>34}" for d in DRIVERS)
    sub = f"{'':<16}" + "".join(f"{'noop%':>8}{'px':>8}{'distinct':>10}{'cov%':>8}" for _ in DRIVERS)
    print(hdr)
    print(sub)
    print("-" * len(sub))

    for gid in games:
        row = f"{gid[:14]:<16}"
        for d in DRIVERS:
            r = run(gid, d, args.actions, args.seed)
            agg[d].append(r)
            row += f"{r['noop_pct']:>8.0f}{r['mean_px']:>8.0f}{r['distinct']:>10}{r['coverage']:>8.0f}"
        print(row)

    print()
    print(f"{args.actions} actions per game per driver, seed {args.seed}, {len(games)} games")
    print(f"{'driver':<16}{'noop%':>8}{'mean px':>10}{'distinct/game':>15}{'coverage%':>12}{'levels':>8}")
    for d in DRIVERS:
        rs = agg[d]
        print(
            f"{d:<16}"
            f"{np.nanmean([r['noop_pct'] for r in rs]):>8.1f}"
            f"{np.mean([r['mean_px'] for r in rs]):>10.0f}"
            f"{np.mean([r['distinct'] for r in rs]):>15.1f}"
            f"{np.mean([r['coverage'] for r in rs]):>12.0f}"
            f"{sum(r['levels'] for r in rs):>8}"
        )


if __name__ == "__main__":
    main()
