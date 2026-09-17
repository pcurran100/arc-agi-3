"""Rendering a rule set's prediction, and scoring it.

D-019 fixed the primary metric as **exact resting-frame match on transitions
where something changed**, against a measured chance baseline of 0%. That
demands a full 64x64 frame out of the model, not a per-object verdict, so the
predicted object states have to be rendered back into a grid.

Rendering is erase-then-paint in two passes. Objects that move or vanish have
their old footprint filled with the background colour first, and only then is
everything repainted; doing it per object instead lets one object's erase clip
the previous object's paint. The object's true per-pixel colours are read from
the *before* frame, because a sprite is not one colour.

**Where this is optimistic and where it is pessimistic.**

- *Pessimistic:* what is revealed behind a departing object has to be guessed.
  :func:`~arc3.dynamics.rules.reveal_colour` guesses the modal colour of the
  surrounding ring, which is right on a locally uniform surface and wrong where
  an object conceals structure. The language has no notion of depth, so this is
  a real and uncounted cost, not a scoring artefact. Erasing to the *global
  background* instead — the obvious first implementation — was measured and
  reproduced 0 of 100 frames on ls20 with perfect effects supplied.
- *Pessimistic:* any transition in which something *appeared* is unpredictable
  in principle (see :mod:`arc3.dynamics.rules`), and is scored as a miss rather
  than excluded. `n_appeared` is reported so the ceiling is visible.
- *Optimistic:* nothing. The comparison is `np.array_equal` on the whole grid.

The secondary object-level diagnostic is kept, per D-019, as a diagnostic only:
it is scored in terms of a segmentation, and D-018 makes the segmentation the
axis being swept, so a metric that varies with that axis cannot be the headline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..perception.geometry import Geometry
from ..perception.segment import background_colour, segment
from ..types import Action, Transition
from .rules import STATIC, VANISH, RuleSet, apply_effect, describe_object, reveal_colour


#: A bookkeeping band may not be thicker than this many rows (of 64). Above
#: it, the band is assumed to be playfield and is refused outright.
MAX_BAND = 4


def bookkeeping_mask(transitions, min_fraction: float = 0.5) -> np.ndarray:
    """Pixels that change on almost every action, whatever the action was.

    `benchmark.md` correction #2 established that the 64x64 frame is a
    **composite**: a camera view, letterboxed, with a UI layer drawn last at
    full resolution. This function recovers that second layer from dynamics
    alone, with no game knowledge: a move counter, timer or score bar ticks on
    every action irrespective of which action it was, so its pixels change at a
    rate no playfield object comes close to.

    It exists because of a measured surprise. On ls20 the induced playfield
    dynamics were *correct* and exact-frame prediction still scored zero: 71 of
    100 transitions were wrong by exactly two pixels, at y=61-62, marching one
    column per action as a colour-11 bar advanced over colour 3. A move counter
    made a completely correct model of the game score 0%.

    **Deliberately conservative, and horizontal only.** Two restrictions, both
    chosen so that the failure mode is *missing* UI rather than *excusing* the
    model from predicting the game:

    - Only row bands are considered. A vertical side-panel UI is missed. That is
      a stated limitation, and it makes this figure a lower bound.
    - A band thicker than :data:`MAX_BAND` rows is refused. Without the cap the
      heuristic eats the playfield: an object confined to a few rows or columns
      makes them change in 100% of transitions, and labelling those bookkeeping
      would *inflate* the score by excusing the model from the real dynamics.
      A test asserts the refusal, because the first version of this function
      accepted an 8-column playfield band.

    Detection is at **band** level (whole rows), because a counter
    advances across the frame: any individual pixel of it changes once, while
    its row changes every time. Measured on ls20, the per-pixel version returns
    an empty mask while rows 61-62 change in 99% of transitions against 10-70%
    for the 25 playfield rows - a clean separation rather than a tuned one.

    Must be computed on **training transitions only** — it is part of the model,
    and fitting it on the test set would be leakage. The threshold is the one
    tuned constant in the dynamics code and its sensitivity is reported by
    `scripts/eval_dynamics.py --bookkeeping-sweep`.
    """
    changed = [t for t in transitions if not t.is_noop]
    if not changed:
        return np.zeros((64, 64), dtype=bool)
    h, w = changed[0].before.shape
    rows = np.zeros(h, dtype=np.int32)
    for t in changed:
        ys, _ = np.nonzero(t.before != t.after)
        rows[np.unique(ys)] += 1
    thresh = max(1, int(min_fraction * len(changed)))
    mask = np.zeros((h, w), dtype=bool)
    qualifying = np.nonzero(rows >= thresh)[0]
    if len(qualifying) and len(qualifying) <= MAX_BAND:
        mask[qualifying, :] = True
    return mask


def predict_frame(
    before: np.ndarray,
    action: Action,
    ruleset: RuleSet,
    name: str,
    geometry: Optional[Geometry] = None,
) -> np.ndarray:
    """The frame the rule set says follows ``action`` from ``before``."""
    seg = segment(before, geometry, include=(name,))[name]
    bg = seg.background_colour
    if bg is None:
        bg = background_colour(before)

    canvas = before.copy()
    effects = []
    for comp in seg.components:
        effect = ruleset.effect_for(action, describe_object(comp, action))
        effects.append((comp, effect))

    # Pass 1: erase the footprint of everything that is not staying put, to the
    # colour that is actually revealed rather than to the global background.
    for comp, effect in effects:
        if effect.kind == STATIC:
            continue
        canvas[comp.mask] = reveal_colour(before, comp.mask, bg)

    # Pass 2: paint.
    for comp, effect in effects:
        if effect.kind == STATIC:
            continue
        if effect.kind == VANISH:
            continue
        apply_effect(canvas, before, comp, effect, bg)

    return canvas


@dataclass
class Score:
    """Outcome of scoring one rule set over a set of transitions."""

    n_total: int = 0
    n_changed: int = 0
    n_exact_changed: int = 0
    n_exact_all: int = 0
    n_appeared_blocked: int = 0
    n_objects: int = 0
    n_objects_correct: int = 0
    n_exact_ex_bookkeeping: int = 0
    n_changed_outside: int = 0
    n_moves: int = 0
    n_moves_deformed: int = 0
    n_changed_objects: int = 0
    n_changed_objects_correct: int = 0
    residuals: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.residuals is None:
            self.residuals = []

    @property
    def exact_on_changed(self) -> float:
        return self.n_exact_changed / self.n_changed if self.n_changed else 0.0

    @property
    def exact_overall(self) -> float:
        return self.n_exact_all / self.n_total if self.n_total else 0.0

    @property
    def exact_ex_bookkeeping(self) -> float:
        """Exact outside the bookkeeping layer, over transitions where that means something.

        The denominator is deliberately ``n_changed_outside`` rather than
        ``n_changed``. If every change a game makes falls *inside* the induced
        mask, then "exact outside the mask" is satisfied by predicting nothing
        and the figure is vacuous. That is not hypothetical: on vc33 **0%** of
        changed transitions alter a pixel outside the mask, and an earlier
        version of this property duly reported 100% for a model whose object
        accuracy was 0%. Read `vacuous` before reading this.
        """
        return (
            self.n_exact_ex_bookkeeping / self.n_changed_outside
            if self.n_changed_outside
            else 0.0
        )

    @property
    def vacuous(self) -> bool:
        """True when almost nothing changes outside the bookkeeping mask."""
        return self.n_changed > 0 and self.n_changed_outside < 0.1 * self.n_changed

    @property
    def deformed_move_rate(self) -> float:
        """Fraction of observed moves the effect vocabulary cannot express.

        An object that moves *and* changes silhouette is recorded as ``static``,
        because the language has no way to say it. Measured at 27-48% on most
        games under ``colour_8`` and 100% on vc33, so this is the dominant known
        loss in the whole pipeline, not a corner case.
        """
        return self.n_moves_deformed / self.n_moves if self.n_moves else 0.0

    @property
    def object_accuracy(self) -> float:
        return self.n_objects_correct / self.n_objects if self.n_objects else 0.0

    @property
    def changed_object_accuracy(self) -> float:
        """Accuracy on objects that actually did something.

        The headline object accuracy is near-worthless on its own: ~91% of
        observed objects are static, so "predict static for everything" scores
        ~91% and looks like competence. This is the version that cannot be
        gamed that way, and it is the same trap D-019 rejected per-pixel
        accuracy for.
        """
        return (
            self.n_changed_objects_correct / self.n_changed_objects
            if self.n_changed_objects
            else 0.0
        )

    @property
    def median_residual(self) -> float:
        return float(np.median(self.residuals)) if self.residuals else 0.0

    @property
    def noop_fraction(self) -> float:
        return (self.n_total - self.n_changed) / self.n_total if self.n_total else 0.0


def score_transitions(
    transitions,
    ruleset: RuleSet,
    name: str,
    geometry: Optional[Geometry] = None,
    ignore: Optional[np.ndarray] = None,
) -> Score:
    """Exact-frame match, plus the object-level diagnostics.

    ``ignore`` is the induced bookkeeping mask (see :func:`bookkeeping_mask`),
    used only for the secondary ``exact_ex_bookkeeping`` figure. The primary
    ``exact_on_changed`` is always the whole frame, as D-019 specified.
    """
    from .induce import observe_transition

    s = Score()
    for t in transitions:
        s.n_total += 1
        changed = not t.is_noop
        if changed:
            s.n_changed += 1

        pred = predict_frame(t.before, t.action, ruleset, name, geometry)
        wrong = pred != t.after
        hit = not bool(wrong.any())
        if hit:
            s.n_exact_all += 1
            if changed:
                s.n_exact_changed += 1
        if changed:
            s.residuals.append(int(wrong.sum()))
            truly_changed = t.before != t.after
            if ignore is not None:
                outside = bool((truly_changed & ~ignore).any())
                if outside:
                    s.n_changed_outside += 1
                    if not bool((wrong & ~ignore).any()):
                        s.n_exact_ex_bookkeeping += 1
            else:
                s.n_changed_outside += 1
                if hit:
                    s.n_exact_ex_bookkeeping += 1

        obs = observe_transition(t, name, geometry)
        if obs.n_appeared and changed:
            s.n_appeared_blocked += 1
        for ev in obs.events:
            s.n_objects += 1
            predicted = ruleset.effect_for(t.action, ev.as_dict())
            correct = predicted == ev.effect
            if correct:
                s.n_objects_correct += 1
            if not ev.effect.is_identity:
                s.n_changed_objects += 1
                if correct:
                    s.n_changed_objects_correct += 1
        s.n_moves += obs.n_moves
        s.n_moves_deformed += obs.n_moves_deformed
    return s


# -- baselines ---------------------------------------------------------------


def score_static_baseline(transitions) -> Score:
    """B0 from D-019: predict that nothing changes. Measured at 0% on changed."""
    s = Score()
    for t in transitions:
        s.n_total += 1
        if not t.is_noop:
            s.n_changed += 1
        if np.array_equal(t.before, t.after):
            s.n_exact_all += 1
    return s


def score_replay_baseline(transitions) -> Score:
    """B1 from D-019: replay the exact pixel delta last seen for this action.

    Rebuilt here rather than cited so the ladder is measured on the same corpus
    as everything else. D-019 measured it at 0% on changed transitions, and the
    reason it fails is the argument for representing dynamics over objects
    rather than pixels.
    """
    s = Score()
    last: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for t in transitions:
        s.n_total += 1
        changed = not t.is_noop
        if changed:
            s.n_changed += 1
        delta = last.get(t.action.id)
        pred = t.before.copy()
        if delta is not None:
            idx, vals = delta
            flat = pred.reshape(-1)
            inb = idx < flat.size
            flat[idx[inb]] = vals[inb]
        if np.array_equal(pred, t.after):
            s.n_exact_all += 1
            if changed:
                s.n_exact_changed += 1
        diff = np.nonzero((t.after != t.before).reshape(-1))[0]
        last[t.action.id] = (diff, t.after.reshape(-1)[diff])
    return s
