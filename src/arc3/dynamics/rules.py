"""The rule language: what a dynamics hypothesis is allowed to say.

This is the file D-019 named as the last thing blocking the dynamics experiment,
and the file Guardrail 2 makes delicate. A rule language *is* a hypothesis
space, so writing one is exactly the act the guardrail warns about — "do not
hand-craft the hypothesis space", on the evidence that Tufa Labs found
hand-crafted tools actively hurt.

Three constraints were adopted to keep the language honest rather than
game-flavoured:

1. **A rule may only condition on attributes the perception layer already
   emits.** No new features were invented for this file. ``colour``, ``size``,
   ``shape`` and ``rect`` are fields of :class:`~arc3.perception.segment.Component`
   that existed before dynamics did. The one addition, ``clicked``, is
   *relational* rather than descriptive — whether the action's click landed
   inside this object — and it is general: every game with a pointer has the
   notion, and no game-specific content is encoded by it. D-021 is the argument
   for it: a click's effect is about *what* you clicked, not about where the
   pixel was in absolute terms.

2. **The effect vocabulary is closed and tiny.** An object can hold still,
   translate, change colour, or cease to exist. These are not a taxonomy of
   games; they are the four things the *tracker* can report, so the language is
   exactly as expressive as the perception beneath it and no more.

3. **Which rules to keep is decided by description length, never by a tuned
   threshold.** See :mod:`arc3.dynamics.induce`.

**What this language cannot say**, stated plainly because the absences are what
a negative result will turn on:

- **Nothing about objects that appear.** Every rule is keyed to an object
  present in the *before* frame, so a frame containing something new is
  unpredictable in principle. Appearances are counted and reported separately.
- **No object-to-object interaction.** "The block stops when it hits a wall" is
  unsayable. Rules are per-object and context-free apart from the click
  relation, so collision, containment and support are all outside the language.
- **No state beyond the current frame.** No velocity, no counters, no memory. A
  game whose response depends on how you arrived is unmodellable.
- **No numeric relations.** A selector tests attribute *equality*, so
  "everything bigger than 4 pixels" cannot be said; it can only be approximated
  by enumerating sizes, which MDL will correctly refuse to do.

Each of those is a deliberate floor, not an oversight. The experiment asks
whether the *simplest* object-level language beats chance; starting richer would
mean a positive result told us nothing about where the power came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from ..perception.segment import Component
from ..types import Action

#: The four things the tracker can report about an object, and therefore the
#: complete effect vocabulary. ``STATIC`` is listed first because it is the
#: overwhelming majority of observations and the default prediction.
STATIC = "static"
TRANSLATE = "translate"
RECOLOUR = "recolour"
VANISH = "vanish"

#: The object is replaced by a different shape, optionally at an offset.
#:
#: Added after measuring what the four original effects could not express.
#: Profiling the five games that produce abundant evidence and score exactly
#: zero (D-032), **63% of their non-static object events were inexpressible**,
#: and 86% of that gap is one thing: an object whose mask changes.
#:
#:   deformed-move  330  (37.5% of the gap)   moved AND changed silhouette
#:   shrank         290  (33.0%)              same place, fewer pixels
#:   grew           119  (13.5%)              same place, more pixels
#:   reshaped        20  ( 2.3%)              same size, different mask
#:
#: All four are the same statement — *the object is now this other shape* — and
#: `BECOME` is that statement. It is not a taxonomy of game mechanics; it is the
#: remaining thing the tracker can report and the language could not repeat.
#:
#: The shape is carried as a `shape_key`, so two occurrences of the same form
#: are the same effect and MDL can generalise over them. The pixel template it
#: refers to lives in :attr:`RuleSet.shapes`, which keeps the effect small,
#: hashable and comparable — storing a mask inline would make every effect
#: unique and MDL would never generalise at all.
BECOME = "become"


@dataclass(frozen=True, slots=True)
class Effect:
    """What happens to one object across one transition."""

    kind: str
    dx: int = 0
    dy: int = 0
    colour: int = -1
    shape: bytes = b""

    def __str__(self) -> str:
        if self.kind == TRANSLATE:
            return f"translate({self.dx:+d},{self.dy:+d})"
        if self.kind == RECOLOUR:
            return f"recolour(->{self.colour})"
        if self.kind == BECOME:
            return f"become(shape#{hash(self.shape) & 0xFFFF:04x},{self.dx:+d},{self.dy:+d})"
        return self.kind

    @property
    def is_identity(self) -> bool:
        return self.kind == STATIC


STATIC_EFFECT = Effect(STATIC)


def _attr_colour(c: Component, a: Action) -> Any:
    return c.dominant_colour


def _attr_size(c: Component, a: Action) -> Any:
    return c.size


def _attr_shape(c: Component, a: Action) -> Any:
    return c.shape_key


def _attr_rect(c: Component, a: Action) -> Any:
    return c.is_rectangular


def _attr_clicked(c: Component, a: Action) -> Any:
    """Did this action's click land inside this object?

    False for every non-click action, which makes the attribute useless on
    keyboard-only games — correctly, since it has nothing to say there, and MDL
    will decline to split on a constant.
    """
    if not a.is_click or a.x is None or a.y is None:
        return False
    m = c.mask
    if not (0 <= a.y < m.shape[0] and 0 <= a.x < m.shape[1]):
        return False
    return bool(m[a.y, a.x])


#: The attributes a selector may test. Every one is a pre-existing perception
#: output; the list is deliberately short and deliberately not extensible by
#: game.
ATTRIBUTES: dict[str, Callable[[Component, Action], Any]] = {
    "colour": _attr_colour,
    "size": _attr_size,
    "shape": _attr_shape,
    "rect": _attr_rect,
    "clicked": _attr_clicked,
}


def describe_object(component: Component, action: Action) -> dict[str, Any]:
    """The attribute vector a rule is matched against."""
    return {name: fn(component, action) for name, fn in ATTRIBUTES.items()}


@dataclass(frozen=True, slots=True)
class Selector:
    """Which objects a rule applies to: a conjunction of attribute equalities.

    The empty conjunction is the universal selector, which is the cheapest thing
    the language can say and the hypothesis every induction starts from. Only
    *equality* is testable — there is no ``size > 4`` — so the language cannot
    express a numeric relation and MDL will refuse to fake one by enumeration.
    """

    conditions: tuple[tuple[str, Any], ...] = ()

    @property
    def arity(self) -> int:
        return len(self.conditions)

    def matches(self, attrs: dict[str, Any]) -> bool:
        return all(attrs.get(k) == v for k, v in self.conditions)

    def __str__(self) -> str:
        if not self.conditions:
            return "any"
        parts = []
        for k, v in self.conditions:
            if isinstance(v, bytes):
                v = f"shape#{hash(v) & 0xFFFF:04x}"
            parts.append(f"{k}={v}")
        return " & ".join(parts)


@dataclass(frozen=True, slots=True)
class Rule:
    """``IF action AND selector THEN effect``."""

    action_id: int
    selector: Selector
    effect: Effect

    def __str__(self) -> str:
        return f"action{self.action_id} & {self.selector} -> {self.effect}"


@dataclass
class RuleSet:
    """One induced model of a game's dynamics.

    Rules are looked up by action and then matched in order of decreasing
    selector arity, so a two-condition rule wins over a one-condition rule,
    with a per-action default beneath both. Nothing here is probabilistic: this is the
    MDL-selected point hypothesis. Maintaining a weighted *population* of these
    is what Guardrail 1 demands and is **not** done yet — see the module
    docstring of :mod:`arc3.dynamics.induce`.
    """

    rules: tuple[Rule, ...] = ()
    defaults: dict[int, Effect] = None  # type: ignore[assignment]
    #: shape_key -> bbox-relative boolean template, for rendering BECOME.
    shapes: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.defaults is None:
            self.defaults = {}
        if self.shapes is None:
            self.shapes = {}

    def effect_for(self, action: Action, attrs: dict[str, Any]) -> Effect:
        # `rules` is stored most-specific-first by the inducer.
        for r in self.rules:
            if r.action_id == action.id and r.selector.matches(attrs):
                return r.effect
        return self.defaults.get(action.id, STATIC_EFFECT)

    @property
    def size(self) -> int:
        return len(self.rules) + len(self.defaults)

    def describe(self, limit: int = 12) -> str:
        lines = [f"{len(self.rules)} rules, {len(self.defaults)} defaults"]
        for r in self.rules[:limit]:
            lines.append(f"  {r}")
        if len(self.rules) > limit:
            lines.append(f"  ... and {len(self.rules) - limit} more")
        for a, e in sorted(self.defaults.items()):
            lines.append(f"  action{a} & any -> {e}   [default]")
        return "\n".join(lines)


def reveal_colour(source: np.ndarray, mask: np.ndarray, background: int) -> int:
    """What colour is under an object when it leaves.

    The naive answer — the frame's background colour — is wrong whenever objects
    sit on a surface rather than on the void, which turns out to be most games.
    On ls20 the sprite moves across floor colour 3 while the frame's background
    is 4, so erasing to the background corrupted **every** transition and made
    exact-frame prediction impossible even when every object's effect was known
    exactly. That was measured, not guessed: with ground-truth effects supplied,
    erase-to-background reproduced 0 of 100 frames with a median of 27 pixels
    wrong.

    The fix used here is the modal colour of the one-pixel ring immediately
    outside the footprint. It is general — no game knowledge, no tuning — and it
    is right whenever an object rests on a locally uniform surface, which is the
    common case. It is *wrong* when an object straddles a boundary or conceals
    something structured, and that residual is real and reported rather than
    hidden: this function cannot recover information the frame does not contain.
    A dynamics model that needs to know what is behind an object needs hidden
    state, which the language deliberately does not have.
    """
    h, w = source.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return background
    y0, y1 = max(0, ys.min() - 1), min(h, ys.max() + 2)
    x0, x1 = max(0, xs.min() - 1), min(w, xs.max() + 2)
    window = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    window[ys - y0, xs - x0] = True
    dil = window.copy()
    dil[1:, :] |= window[:-1, :]
    dil[:-1, :] |= window[1:, :]
    dil[:, 1:] |= window[:, :-1]
    dil[:, :-1] |= window[:, 1:]
    ring = dil & ~window
    vals = source[y0:y1, x0:x1][ring]
    if vals.size == 0:
        return background
    counts = np.bincount(vals.astype(np.int64))
    return int(np.argmax(counts))


def apply_effect(
    canvas: np.ndarray,
    source: np.ndarray,
    component: Component,
    effect: Effect,
    background: int,
    shapes: Optional[dict] = None,
) -> None:
    """Draw ``component`` under ``effect`` onto ``canvas``, in place.

    ``source`` is the *before* frame, read for the object's true per-pixel
    colours: an object is not one colour, and repainting it with its dominant
    colour would corrupt every multi-coloured sprite.

    Assumes the caller has already erased the object's old footprint, so that a
    translate does not leave a copy behind. Erase-then-paint in two passes over
    all objects, never per object, or one object's paint is clipped by the next
    one's erase. See :func:`reveal_colour` for what "erased" has to mean.
    """
    if effect.kind == VANISH:
        return
    mask = component.mask
    ys, xs = np.nonzero(mask)
    if effect.kind == STATIC:
        canvas[ys, xs] = source[ys, xs]
        return
    if effect.kind == RECOLOUR:
        canvas[ys, xs] = effect.colour
        return
    if effect.kind == TRANSLATE:
        ny, nx = ys + effect.dy, xs + effect.dx
        h, w = canvas.shape
        keep = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        canvas[ny[keep], nx[keep]] = source[ys[keep], xs[keep]]
        return
    if effect.kind == BECOME:
        tmpl = shapes.get(effect.shape) if shapes else None
        if tmpl is None:
            # Shape never seen at induction time: fall back to leaving the
            # object where it was rather than erasing it, which is the less
            # destructive of the two wrong answers.
            canvas[ys, xs] = source[ys, xs]
            return
        mask_t, colour_t = tmpl
        th, tw = mask_t.shape
        y0 = component.bbox.y0 + effect.dy
        x0 = component.bbox.x0 + effect.dx
        h, w = canvas.shape
        tys, txs = np.nonzero(mask_t)
        ny, nx = tys + y0, txs + x0
        keep = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        canvas[ny[keep], nx[keep]] = colour_t[tys[keep], txs[keep]]
