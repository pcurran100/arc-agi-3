"""Score segmentation and cross-frame object identity against the oracle.

Usage::

    python scripts/eval_tracking.py [--actions N] [--seed S] [--segmentation NAME]

This is the headline number for D-007: if cross-frame identity is still
unreliable after roughly two weeks, that decision says to reassess the symbolic
approach entirely. It uses the D-010 oracle and never runs on the agent's path.

**A limitation to state plainly:** the oracle reads the engine's *current* state,
so ground truth is only available for resting frames - the last frame after each
action. The intermediate frames of an animation cannot be scored this way, and
they are exactly the frames where tracking is hardest. The numbers here are
therefore an optimistic bound on animated games.

Three things are measured, because "does tracking work" is not one question:

- **id switch rate** - of sprites visible in two consecutive resting frames, how
  often did our track id for that sprite change? This is the number that matters.
- **fragmentation** - components per ground-truth sprite. Above 1 means we split
  objects the engine considers whole.
- **merge rate** - sprites per component. Above 1 means we fused distinct
  objects. Note bp35 and lf52 render entirely through the UI layer and have no
  visible sprites at all, so they are reported as unscoreable rather than as
  failures.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

import numpy as np

from arc3.client.env import EnvClient, available_environments
from arc3.oracle import UI, ground_truth
from arc3.perception.geometry import induce_geometry
from arc3.perception.segment import segment
from arc3.perception.track import Tracker
from arc3.types import Action


def sprite_track_labels(
    instance_map: np.ndarray, tracker: Tracker, seg
) -> dict[int, int]:
    """For each visible sprite, the track id covering most of its pixels."""
    track_of_component = {
        comp_index: tid for comp_index, tid in tracker._prev_track_of.items()
    }
    pixel_track = np.full(instance_map.shape, -1, dtype=np.int64)
    for comp in seg.components:
        tid = track_of_component.get(comp.index)
        if tid is not None:
            pixel_track[comp.mask] = tid

    out: dict[int, int] = {}
    for sprite in np.unique(instance_map):
        if sprite < 0:
            continue
        m = instance_map == sprite
        votes = pixel_track[m]
        votes = votes[votes >= 0]
        if votes.size:
            out[int(sprite)] = int(Counter(votes.tolist()).most_common(1)[0][0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segmentation", default="foreground_8")
    args = ap.parse_args()

    hdr = (
        f"{'game':<16}{'cell':<6}{'comps':<7}{'sprites':<9}{'frag':<7}"
        f"{'merge':<7}{'switch%':<9}{'excl%':<8}{'pairs':<7}{'note'}"
    )
    print(hdr)
    print("-" * len(hdr))

    tot_switch = tot_pairs = 0
    tot_excl_switch = tot_excl_pairs = 0
    frags: list[float] = []
    merges: list[float] = []
    unscoreable: list[str] = []
    terminal: list[str] = []

    for gid in available_environments():
        rng = random.Random(args.seed)
        client = EnvClient(gid, seed=args.seed, record=False)

        # Warm-up rollout to give geometry something to induce from. Geometry is
        # fixed for the run: re-inducing per frame would let the representation
        # shift underneath the tracker and make identity meaningless.
        warm = list(client.current.frames)
        for _ in range(min(20, args.actions)):
            acts = client.current.available_actions
            if not acts or client.current.state.is_terminal:
                break
            a = rng.choice(acts)
            warm.extend(
                client.step(
                    Action.click(rng.randrange(64), rng.randrange(64))
                    if a == 6
                    else Action.simple(a)
                ).frames
            )
        geo = induce_geometry(warm)

        if client.current.state.is_terminal:
            terminal.append(gid)
            print(
                f"{gid:<16}{geo.block.cell:<6}{'-':<7}{'-':<9}{'-':<7}{'-':<7}"
                f"{'-':<9}{'-':<8}{0:<7}terminal during warm-up"
            )
            continue

        tracker = Tracker(segmentation_name=args.segmentation)
        prev_labels: dict[int, int] | None = None
        prev_exclusive: set[int] = set()
        switches = pairs = 0
        excl_switches = excl_pairs = 0
        frag_samples: list[float] = []
        merge_samples: list[float] = []
        n_comps: list[int] = []

        for _ in range(args.actions):
            acts = client.current.available_actions
            if not acts or client.current.state.is_terminal:
                break
            a = rng.choice(acts)
            obs = client.step(
                Action.click(rng.randrange(64), rng.randrange(64))
                if a == 6
                else Action.simple(a)
            )
            frame = obs.last
            if frame is None:
                continue

            gt = ground_truth(client._engine_game, rendered=frame)
            seg = segment(frame, geo)[args.segmentation]
            tracker.update(seg)
            n_comps.append(len(seg.components))

            inst = gt.instance_map.copy()
            inst[inst == UI] = -1  # UI is not a sprite; do not score it

            # Fragmentation and merging, on this frame.
            comp_of_sprite: dict[int, set[int]] = defaultdict(set)
            sprite_of_comp: dict[int, set[int]] = defaultdict(set)
            for comp in seg.components:
                for sprite in np.unique(inst[comp.mask]):
                    if sprite >= 0:
                        comp_of_sprite[int(sprite)].add(comp.index)
                        sprite_of_comp[comp.index].add(int(sprite))
            if comp_of_sprite:
                frag_samples.append(
                    float(np.mean([len(v) for v in comp_of_sprite.values()]))
                )
            if sprite_of_comp:
                merge_samples.append(
                    float(np.mean([len(v) for v in sprite_of_comp.values()]))
                )

            # A sprite is "exclusive" when the component covering it covers no
            # other sprite. Restricting the switch rate to these removes the
            # credit a degenerate segmentation earns by merging the whole frame
            # into one blob, where identity trivially never changes.
            exclusive = {
                sprite
                for sprite, comps in comp_of_sprite.items()
                if all(len(sprite_of_comp[ci]) == 1 for ci in comps)
            }

            labels = sprite_track_labels(inst, tracker, seg)
            if prev_labels is not None:
                for sprite, tid in labels.items():
                    if sprite in prev_labels:
                        pairs += 1
                        changed_id = prev_labels[sprite] != tid
                        if changed_id:
                            switches += 1
                        if sprite in exclusive and sprite in prev_exclusive:
                            excl_pairs += 1
                            if changed_id:
                                excl_switches += 1
            prev_labels = labels
            prev_exclusive = exclusive

        if pairs == 0:
            unscoreable.append(gid)
            print(
                f"{gid:<16}{geo.block.cell:<6}"
                f"{int(np.mean(n_comps)) if n_comps else 0:<7}{'0':<9}"
                f"{'-':<7}{'-':<7}{'-':<9}{'-':<8}{0:<7}no visible sprites"
            )
            continue

        frag = float(np.mean(frag_samples)) if frag_samples else float("nan")
        merge = float(np.mean(merge_samples)) if merge_samples else float("nan")
        frags.append(frag)
        merges.append(merge)
        tot_switch += switches
        tot_pairs += pairs
        tot_excl_switch += excl_switches
        tot_excl_pairs += excl_pairs
        excl_str = f"{100 * excl_switches / excl_pairs:.1f}" if excl_pairs else "-"
        print(
            f"{gid:<16}{geo.block.cell:<6}{int(np.mean(n_comps)):<7}"
            f"{len(prev_labels or {}):<9}{frag:<7.2f}{merge:<7.2f}"
            f"{100 * switches / pairs:<9.1f}{excl_str:<8}{pairs:<7}"
            f"{'' if excl_pairs else 'no exclusive sprites'}"
        )

    print()
    print(f"segmentation: {args.segmentation}, {args.actions} actions, seed {args.seed}")
    if tot_pairs:
        print(
            f"OVERALL id switch rate: {100 * tot_switch / tot_pairs:.1f}% "
            f"over {tot_pairs} sprite-frame pairs"
        )
    if tot_excl_pairs:
        print(
            f"OVERALL id switch rate, exclusive sprites only: "
            f"{100 * tot_excl_switch / tot_excl_pairs:.1f}% over {tot_excl_pairs} pairs "
            f"<- the number to trust"
        )
    print(f"mean fragmentation (components per sprite): {np.mean(frags):.2f}")
    print(f"mean merge (sprites per component): {np.mean(merges):.2f}")
    if unscoreable:
        print(f"unscoreable (no visible sprites, UI-rendered): {', '.join(unscoreable)}")
    if terminal:
        print(f"terminal during warm-up (random play died): {', '.join(terminal)}")


if __name__ == "__main__":
    main()
