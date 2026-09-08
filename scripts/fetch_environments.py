"""Download the 25 public demo environments into ``environment_files/``.

Run once. After this everything is offline (D-008)::

    python scripts/fetch_environments.py

An ``ARC_API_KEY`` is not required - the API issues an anonymous key on request,
and the 25 public demo games are available with it. Set ``ARC_API_KEY`` in
``.env`` if you have registered one; it is needed for anything beyond the public
demo set.

``environment_files/`` is gitignored and must stay that way: it is a regenerable
cache and the redistribution position on the game files is unclear (D-008).

**The downloaded game sources are readable Python.** Do not read them to inform
design - that is exactly the hand-specified structure Guardrail 2 warns against,
and anything learned from 25 games we cannot generalise from is worse than
useless on the private set. Reading the *engine* (``arcengine``) is fine and is
how the D-009 corrections were found.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import requests

ANON_KEY_URL = "https://three.arcprize.org/api/games/anonkey"
GAMES_URL = "https://three.arcprize.org/api/games"


def anonymous_key(base_url: str) -> str:
    r = requests.get(f"{base_url}/api/games/anonkey", timeout=15)
    r.raise_for_status()
    return str(r.json().get("api_key", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://three.arcprize.org")
    ap.add_argument("--environments-dir", default="environment_files")
    args = ap.parse_args()

    key = os.getenv("ARC_API_KEY", "")
    if key:
        print("using ARC_API_KEY from the environment")
    else:
        print("no ARC_API_KEY set; requesting an anonymous key")
        try:
            key = anonymous_key(args.base_url)
        except requests.RequestException as exc:
            print(f"could not obtain an anonymous key: {exc}", file=sys.stderr)
            return 1
        if not key:
            print("the API returned no key", file=sys.stderr)
            return 1
    os.environ["ARC_API_KEY"] = key

    try:
        listing = requests.get(
            f"{args.base_url}/api/games", headers={"X-Api-Key": key}, timeout=30
        )
        listing.raise_for_status()
        games = listing.json()
    except requests.RequestException as exc:
        print(f"could not list games: {exc}", file=sys.stderr)
        return 1

    print(f"{len(games)} games available")

    import arc_agi

    quiet = logging.getLogger("fetch")
    quiet.setLevel(logging.ERROR)
    arcade = arc_agi.Arcade(environments_dir=args.environments_dir, logger=quiet)

    failed = []
    for entry in sorted(games, key=lambda g: g["game_id"]):
        gid = entry["game_id"]
        baselines = entry.get("baseline_actions") or []
        try:
            env = arcade.make(gid)
        except Exception as exc:
            env, exc_text = None, f"{type(exc).__name__}: {exc}"
        else:
            exc_text = ""
        if env is None:
            failed.append(gid)
            print(f"  FAILED {gid}  {exc_text}")
            continue
        tags = ",".join(entry.get("tags") or []) or "-"
        print(f"  {gid:<16} {tags:<16} {len(baselines)} levels")

    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\ncached to {args.environments_dir}/ - everything runs offline from here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
