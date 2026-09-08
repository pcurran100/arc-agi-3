"""Ground-truth perception labels from engine introspection.

**Test and evaluation only. Nothing on the agent's code path may import this.**
See D-010. ``tests/test_oracle_boundary.py`` enforces the rule.

The engine knows exactly which sprite painted which pixel. We reproduce
``Camera.render`` with sprite indices in place of colours, which yields a
pixel-perfect instance-segmentation map for every frame of every public demo
game - replacing the "hand-label a few sequences" plan in
``docs/state-of-play.md`` with an exhaustive automated metric.

Two honesty constraints, both load-bearing if the resulting numbers are to mean
anything:

- **Occlusion is modelled.** The map records the topmost sprite at each pixel,
  so a fully occluded sprite has no pixels and is correctly excluded from any
  "did we track it" denominator. Scoring against the sprite *list* rather than
  the rendered map would demand that perception find invisible objects.
- **A sprite is not always an object.** bp35's entire 8x8 board is a single
  sprite, so its render is many visually distinct blobs. Sprite identity is an
  upper bound on the granularity perception could recover, never a target to hit
  exactly. Report both directions of the mismatch, never just one.

Sprite ``name`` is a *type* label shared by many instances (obfuscated random
strings, so no semantic leakage); instance identity is the sprite object itself,
which the engine mutates in place across steps within a level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

MAX_DIMENSION = 64

#: Sentinel labels in an instance map. Real sprites get indices >= 0.
LETTERBOX = -1
BACKGROUND = -2
UI = -3


@dataclass(frozen=True, slots=True)
class SpriteInfo:
    """One sprite's ground truth at a moment in time."""

    index: int
    key: int  # id() of the engine object - stable instance identity within a level
    name: str  # type label, shared across instances
    tags: tuple[str, ...]
    x: int
    y: int
    width: int
    height: int
    scale: int
    layer: int
    visible: bool
    collidable: bool


@dataclass(frozen=True, slots=True)
class OracleFrame:
    """Ground truth for one rendered frame."""

    instance_map: np.ndarray  # (64, 64) int32, sentinels above or a sprite index
    colour_map: np.ndarray  # (64, 64) int8, our reconstruction of the render
    sprites: tuple[SpriteInfo, ...]
    camera_w: int
    camera_h: int
    scale: int
    x_offset: int
    y_offset: int
    letterbox_colour: int
    background_colour: int

    @property
    def playfield(self) -> tuple[int, int, int, int]:
        """Display-space ``(x0, y0, x1, y1)`` of the scaled camera view, x1/y1 exclusive."""
        return (
            self.x_offset,
            self.y_offset,
            self.x_offset + self.camera_w * self.scale,
            self.y_offset + self.camera_h * self.scale,
        )

    @property
    def ui_mask(self) -> np.ndarray:
        """Pixels a UI interface painted over the scaled camera view."""
        return self.instance_map == UI

    @property
    def letterbox_mask(self) -> np.ndarray:
        return self.instance_map == LETTERBOX

    def visible_instance_indices(self) -> tuple[int, ...]:
        """Sprite indices that actually contributed a pixel to this frame."""
        vals = np.unique(self.instance_map)
        return tuple(int(v) for v in vals if v >= 0)

    def instance_masks(self) -> dict[int, np.ndarray]:
        """Boolean mask per visible sprite index."""
        return {i: (self.instance_map == i) for i in self.visible_instance_indices()}

    def sprite(self, index: int) -> SpriteInfo:
        return self.sprites[index]


def _sprite_infos(sprites: list[Any]) -> tuple[SpriteInfo, ...]:
    out = []
    for i, s in enumerate(sprites):
        out.append(
            SpriteInfo(
                index=i,
                key=id(s),
                name=str(s.name),
                tags=tuple(s.tags or ()),
                x=int(s.x),
                y=int(s.y),
                width=int(s.width),
                height=int(s.height),
                scale=int(s.scale),
                layer=int(s.layer),
                visible=bool(s.is_visible),
                collidable=bool(s.is_collidable),
            )
        )
    return tuple(out)


def _scale_and_offset(camera: Any) -> tuple[int, int, int]:
    scale = min(MAX_DIMENSION // camera.width, MAX_DIMENSION // camera.height)
    x_offset = (MAX_DIMENSION - camera.width * scale) // 2
    y_offset = (MAX_DIMENSION - camera.height * scale) // 2
    return scale, x_offset, y_offset


def _raw_instance_render(camera: Any, sprites: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce ``Camera._raw_render`` twice: once for colour, once for index."""
    h, w = camera.height, camera.width
    colour = np.full((h, w), camera.background, dtype=np.int8)
    index = np.full((h, w), BACKGROUND, dtype=np.int32)

    for i, sprite in sorted(
        ((i, s) for i, s in enumerate(sprites) if s.is_visible), key=lambda t: t[1].layer
    ):
        pixels = sprite.render()
        sh, sw = pixels.shape
        rel_x, rel_y = sprite.x - camera.x, sprite.y - camera.y

        dx0, dx1 = max(0, rel_x), min(w, rel_x + sw)
        dy0, dy1 = max(0, rel_y), min(h, rel_y + sh)
        if dx1 <= dx0 or dy1 <= dy0:
            continue

        sx0 = max(0, -rel_x)
        sx1 = sw - max(0, (rel_x + sw) - w)
        sy0 = max(0, -rel_y)
        sy1 = sh - max(0, (rel_y + sh) - h)

        region = pixels[sy0:sy1, sx0:sx1]
        mask = region >= 0
        colour[dy0:dy1, dx0:dx1][mask] = region[mask]
        index[dy0:dy1, dx0:dx1][mask] = i

    return colour, index


def ground_truth(game: Any, rendered: Optional[np.ndarray] = None) -> OracleFrame:
    """Ground truth for ``game``'s current state.

    ``rendered`` is the frame the engine actually produced. When supplied, any
    pixel where our reconstruction disagrees is attributed to a UI interface
    (which is drawn last, over everything, spanning the full 64x64) and labelled
    :data:`UI`. Without it, UI pixels are indistinguishable from playfield ones
    and the map will be wrong wherever a UI element overlaps the camera view.
    """
    camera = game.camera
    sprites = list(game.current_level.get_sprites())
    scale, x_off, y_off = _scale_and_offset(camera)

    raw_colour, raw_index = _raw_instance_render(camera, sprites)
    if scale > 1:
        raw_colour = np.repeat(np.repeat(raw_colour, scale, axis=0), scale, axis=1)
        raw_index = np.repeat(np.repeat(raw_index, scale, axis=0), scale, axis=1)

    colour = np.full((MAX_DIMENSION, MAX_DIMENSION), camera.letter_box, dtype=np.int8)
    index = np.full((MAX_DIMENSION, MAX_DIMENSION), LETTERBOX, dtype=np.int32)
    h, w = raw_colour.shape
    colour[y_off : y_off + h, x_off : x_off + w] = raw_colour
    index[y_off : y_off + h, x_off : x_off + w] = raw_index

    if rendered is not None:
        painted = np.asarray(rendered, dtype=np.int8) != colour
        index[painted] = UI
        colour = np.asarray(rendered, dtype=np.int8).copy()

    return OracleFrame(
        instance_map=index,
        colour_map=colour,
        sprites=_sprite_infos(sprites),
        camera_w=int(camera.width),
        camera_h=int(camera.height),
        scale=int(scale),
        x_offset=int(x_off),
        y_offset=int(y_off),
        letterbox_colour=int(camera.letter_box),
        background_colour=int(camera.background),
    )


def true_geometry(game: Any) -> dict[str, int]:
    """The camera geometry perception is trying to recover, for scoring."""
    camera = game.camera
    scale, x_off, y_off = _scale_and_offset(camera)
    return {
        "camera_w": int(camera.width),
        "camera_h": int(camera.height),
        "scale": scale,
        "x_offset": x_off,
        "y_offset": y_off,
    }
