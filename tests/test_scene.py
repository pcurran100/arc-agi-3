"""Scenes: objects, band removal, rendering."""

from __future__ import annotations

import numpy as np

from arc3.scene import make_scene


def frame():
    f = np.zeros((64, 64), dtype=np.int8)
    f[10:14, 20:24] = 3      # a 4x4 block
    f[30:32, 5:15] = 7       # a 2x10 bar
    f[63, :10] = 5           # a counter in the bottom row
    return f


def test_objects_largest_first_with_boxes():
    s = make_scene(frame())
    assert [o.size for o in s.objects] == [20, 16, 10]
    bar = s.objects[0]
    assert (bar.colour, bar.x0, bar.y0, bar.w, bar.h) == (7, 5, 30, 10, 2)


def test_band_rows_hidden_from_objects_grid_and_counts():
    band = np.zeros((64, 64), dtype=bool)
    band[63, :] = True
    s = make_scene(frame(), band)
    assert all(o.colour != 5 for o in s.objects)
    assert s.count(5) == 0
    assert s.band_rows == (63,)
    assert "_" in s.grid_text()


def test_helpers():
    s = make_scene(frame())
    assert [o.colour for o in s.of_colour(3)] == [3]
    assert s.count(3) == 16
    assert s.colours() == [0, 3, 5, 7]


def test_describe_mentions_every_listed_object():
    s = make_scene(frame())
    t = s.describe()
    assert "#0 colour 7" in t and "background colour 0" in t
    assert "y10" in t
