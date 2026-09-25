"""Turning observed transitions into rules, and choosing between rule sets.

Two jobs, kept separate because they fail differently.

**Observation** reads a transition through a fixed segmentation and reports, per
object present before, which of the four effects it underwent. This is pure
perception bookkeeping and its failures are perception failures.

**Induction** takes those observations and picks a rule set by **minimum
description length**: total bits to write the rules down, plus total bits to
write down the examples the rules get wrong. The hypothesis space searched is
every selector of arity 0, 1 or 2 over :data:`~arc3.dynamics.rules.ATTRIBUTES`,
which is 1 + 5 + 10 = 16 candidate partitions per action — small enough to
enumerate exhaustively, so there is no search heuristic to tune and no local
optimum to worry about.

MDL rather than accuracy because accuracy alone always prefers the finest
partition: splitting on ``size`` gives almost every object its own rule and
scores perfectly on the training set. The bit cost of naming all those rules is
what stops it, and it stops it without a tuned regularisation constant — the
costs come from the sizes of the spaces actually being encoded.

**What this does not do.**

- **It is a point hypothesis, not a population.** Guardrail 1 and D-003 require a
  weighted population of hypotheses with soft down-weighting; this returns the
  single MDL-best rule set and throws the runners-up away. That is a deliberate
  narrowing *for the experiment only*: the question on the table is whether
  object-level rules beat chance at all, and a population cannot rescue a
  language that cannot express the dynamics. If the experiment succeeds, the
  population is the first thing to add, and the MDL scores computed here are
  already the natural weights.
- **It does not use the intermediate animation frames.** `observe_transition`
  reads the resting endpoints only. The corpus carries whole animations and
  `state-of-play.md` argues at length that the trajectory is far more
  constraining than the endpoints — that remains true and unexploited. Endpoint
  prediction is the harder target and the one D-019 defined the metric on, so it
  is what is measured first.
- **It cannot represent an object that both moves and recolours.** Motion wins,
  and the recolour is silently dropped, which will cost exact-frame matches on
  any game that does both at once.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..perception.geometry import Geometry
from ..perception.segment import Segmentation, segment
from ..perception.track import match_frames
from ..types import Action, Transition
from .rules import (
    ATTRIBUTES,
    BECOME,
    RECOLOUR,
    STATIC,
    STATIC_EFFECT,
    TRANSLATE,
    VANISH,
    Effect,
    Rule,
    RuleSet,
    Selector,
    describe_object,
)


@dataclass(frozen=True, slots=True)
class ObjectEvent:
    """One object's observed fate across one transition."""

    attrs: tuple[tuple[str, Any], ...]
    effect: Effect

    def as_dict(self) -> dict[str, Any]:
        return dict(self.attrs)


@dataclass
class TransitionObservation:
    """Everything one transition contributes as evidence."""

    action: Action
    events: tuple[ObjectEvent, ...]
    n_appeared: int
    changed: bool
    n_moves: int = 0
    n_moves_deformed: int = 0
    n_blind: int = 0
    shapes: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.shapes is None:
            self.shapes = {}

    @property
    def is_predictable_in_principle(self) -> bool:
        """False when something appeared: no per-before-object rule can say it."""
        return self.n_appeared == 0


def _effect_from_match(prev, nxt, same_colour: bool, before=None, after=None) -> tuple:
    """Classify one matched pair into the closed effect vocabulary.

    Returns ``(effect, blind)``, where ``blind`` marks a pair whose pixels
    demonstrably changed while every property the vocabulary can see stayed the
    same. See below.

    Displacement is taken from the **bounding-box corner**, not the centroid.
    `match_frames` rounds a centroid difference, which is correct for scoring a
    pairing but off by a pixel when the shape changed, and a one-pixel error
    makes an exact-frame prediction fail. The corner delta is exact whenever the
    silhouette is preserved, which is the only case a translate should be
    claimed for.

    **Partial recolouring is invisible to this vocabulary, and pretending
    otherwise corrupted a headline metric.** A component can keep its mask, its
    size, its bounding box and its *dominant* colour while some of its pixels
    change colour. `foreground_*` carvings make this common, because a component
    there is "anything that is not background" and its internal colouring is not
    part of its identity. The observation then reports ``static``, the model
    predicts ``static``, the two agree, and `changed_obj_acc` scores the pair as
    correct while the rendered frame is wrong.

    Measured on s5i5 under `foreground_4`: **100% changed-object accuracy, zero
    deformed moves, zero appearances, and a 2-pixel median residual outside the
    bookkeeping band** — residuals which turned out to be pixels going 11->5 and
    13->11 inside a component whose mask never moved. The model was not perfect;
    the *observation* could not see what was wrong.

    So when the frames are supplied, the pixel content is compared directly and
    the mismatch is counted. The effect returned is still ``static``, because
    the language genuinely cannot express a partial recolour — but the blindness
    is now reported rather than scored as success.
    """
    dx = nxt.bbox.x0 - prev.bbox.x0
    dy = nxt.bbox.y0 - prev.bbox.y0
    same_silhouette = (
        prev.bbox.width == nxt.bbox.width
        and prev.bbox.height == nxt.bbox.height
        and prev.size == nxt.size
    )
    if (dx or dy) and same_silhouette:
        return Effect(TRANSLATE, dx=int(dx), dy=int(dy)), False
    if not (dx or dy) and not same_colour:
        return Effect(RECOLOUR, colour=int(nxt.dominant_colour)), False
    if not (dx or dy) and same_silhouette:
        blind = False
        if before is not None and after is not None:
            m = prev.mask
            if not np.array_equal(before[m], after[m]):
                blind = True
        if blind:
            # Same mask, same dominant colour, different pixels: a partial
            # recolour. BECOME can state it, because the template carries the
            # per-pixel colours rather than one colour.
            return Effect(BECOME, dx=0, dy=0, shape=nxt.shape_key), True
        return STATIC_EFFECT, False
    # Moved and deformed, grew, shrank, or reshaped. All four are the same
    # statement and BECOME is that statement (D-033).
    return Effect(BECOME, dx=int(dx), dy=int(dy), shape=nxt.shape_key), False


def observe_transition(
    transition: Transition,
    name: str,
    geometry: Optional[Geometry] = None,
) -> TransitionObservation:
    """Read one transition through one segmentation."""
    before, after = transition.before, transition.after
    seg_b: Segmentation = segment(before, geometry, include=(name,))[name]
    seg_a: Segmentation = segment(after, geometry, include=(name,))[name]
    matching = match_frames(seg_b, seg_a)

    events: list[ObjectEvent] = []
    shapes: dict = {}
    n_moves = n_deformed = n_blind = 0
    by_prev = matching.by_prev()
    for i, comp in enumerate(seg_b.components):
        m = by_prev.get(i)
        if m is None:
            effect = Effect(VANISH)
        else:
            nxt = seg_a.components[m.next_index]
            effect, blind = _effect_from_match(comp, nxt, m.same_colour, before, after)
            if effect.kind == BECOME and effect.shape not in shapes:
                r = nxt.bbox
                sub = nxt.mask[r.y0 : r.y1, r.x0 : r.x1]
                shapes[effect.shape] = (sub, after[r.y0 : r.y1, r.x0 : r.x1])
            if blind:
                n_blind += 1
            if (nxt.bbox.x0 - comp.bbox.x0) or (nxt.bbox.y0 - comp.bbox.y0):
                n_moves += 1
                if effect.kind != TRANSLATE:
                    n_deformed += 1
        attrs = describe_object(comp, transition.action)
        events.append(ObjectEvent(attrs=tuple(sorted(attrs.items())), effect=effect))

    return TransitionObservation(
        action=transition.action,
        events=tuple(events),
        n_appeared=len(matching.appeared),
        changed=not transition.is_noop,
        n_moves=n_moves,
        n_moves_deformed=n_deformed,
        n_blind=n_blind,
        shapes=shapes,
    )


# -- MDL ---------------------------------------------------------------------


def _bits(n: int) -> float:
    """Bits to name one of ``n`` things. 0 when there is no choice to encode."""
    return math.log2(n) if n > 1 else 0.0


def _partition_cost(
    groups: dict[Any, list[Effect]],
    n_effects: int,
    n_examples: int,
    n_candidate_partitions: int,
    arity: int,
) -> tuple[float, dict[Any, Effect], int, Effect]:
    """Bits to describe one candidate partition, plus its per-group effects.

    **Only groups that disagree with the default are charged for.** An earlier
    version of this function charged bits for every group in the partition,
    which prices the *partition* rather than the *description* and is therefore
    not MDL at all. It had a sharp consequence, caught by printing the candidate
    table on ls20: selecting on ``size`` made **1 error in 316** and was
    rejected (cost 243) in favour of ``colour`` with **16 errors** (cost 190),
    purely because ``size`` induces 37 groups. Thirty-five of those groups say
    "static", which is the default and needs no rule written down.

    The corrected accounting charges:

    - naming the chosen partition, once;
    - naming the default effect, once;
    - per *exception group*: its attribute values plus its effect;
    - per misfit example: which example, and its true effect.
    """
    chosen: dict[Any, Effect] = {}
    errors = 0
    tally: Counter = Counter()
    for key, effects in groups.items():
        best, count = Counter(effects).most_common(1)[0]
        chosen[key] = best
        errors += len(effects) - count
        tally[best] += len(effects)

    default = tally.most_common(1)[0][0]
    n_rules = sum(1 for e in chosen.values() if e != default)

    value_bits = arity * _bits(max(2, len(groups)))
    model_bits = _bits(n_candidate_partitions) + _bits(n_effects)
    model_bits += n_rules * (value_bits + _bits(n_effects))
    error_bits = errors * (_bits(max(2, n_examples)) + _bits(n_effects))
    return model_bits + error_bits, chosen, errors, default


def induce(
    observations: Sequence[TransitionObservation],
    *,
    max_arity: int = 2,
) -> RuleSet:
    """Pick the MDL-best rule set over the observed events.

    Exhaustive over selectors of arity 0..``max_arity``; no search heuristic.
    """
    by_action: dict[int, list[ObjectEvent]] = defaultdict(list)
    for obs in observations:
        for ev in obs.events:
            by_action[obs.action.id].append(ev)

    rules: list[Rule] = []
    defaults: dict[int, Effect] = {}

    attr_names = list(ATTRIBUTES)
    combos: list[tuple[str, ...]] = [()]
    for k in range(1, max_arity + 1):
        combos.extend(combinations(attr_names, k))
    n_candidates = len(combos)

    for action_id, events in by_action.items():
        if not events:
            continue
        effect_vocab = {ev.effect for ev in events}
        n_effects = max(2, len(effect_vocab))
        n_examples = len(events)

        best: Optional[tuple[float, tuple[str, ...], dict[Any, Effect], Effect]] = None
        for combo in combos:
            groups: dict[Any, list[Effect]] = defaultdict(list)
            for ev in events:
                d = ev.as_dict()
                key = tuple(d[a] for a in combo)
                groups[key].append(ev.effect)
            cost, chosen, _, default = _partition_cost(
                groups, n_effects, n_examples, n_candidates, len(combo)
            )
            if best is None or cost < best[0]:
                best = (cost, combo, chosen, default)

        assert best is not None
        _, combo, chosen, fallback = best
        defaults[action_id] = fallback
        if not combo:
            continue

        for key, effect in chosen.items():
            if effect == fallback:
                continue
            conditions = tuple(zip(combo, key))
            rules.append(Rule(action_id=action_id, selector=Selector(conditions), effect=effect))

    rules.sort(key=lambda r: -r.selector.arity)
    shapes: dict = {}
    for obs in observations:
        shapes.update(obs.shapes)
    return RuleSet(rules=tuple(rules), defaults=defaults, shapes=shapes)
