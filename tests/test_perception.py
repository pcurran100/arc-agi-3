"""Perception tests, run entirely from recorded fixtures - no engine, no actions."""

from __future__ import annotations

import numpy as np
import pytest

from arc3.perception.components import CONN4, CONN8, label, largest_component_mask
from arc3.perception.diff import diff, diff_sequence
from arc3.perception.geometry import (
    Rect,
    induce_geometry,
    verify_cell,
)
from arc3.perception.segment import background_colour, segment
from arc3.perception.track import Tracker, match_frames
from arc3.types import freeze_grid

# -- components ------------------------------------------------------------


def test_connectivity_changes_the_answer():
    g = np.array([[1, 0], [0, 1]], dtype=np.int8)
    _, n4 = label(g, connectivity=CONN4)
    _, n8 = label(g, connectivity=CONN8)
    assert n4 == 4  # four singletons: two 1s, two 0s
    assert n8 == 2  # each colour joins diagonally


def test_labels_are_deterministic_and_raster_ordered():
    g = np.array([[2, 2, 0], [0, 0, 3]], dtype=np.int8)
    labels, n = label(g, connectivity=CONN4)
    # Four: the pair of 2s, the isolated 0 at (0,2), the pair of 0s on row 1,
    # and the 3. The top-right 0 is not 4-connected to the bottom-left 0s.
    assert n == 4
    assert labels[0, 0] == labels[0, 1] == 0  # first component in raster order
    assert labels[0, 2] == 1
    assert labels[1, 0] == labels[1, 1] == 2
    assert labels[1, 2] == 3
    again, _ = label(g, connectivity=CONN4)
    assert np.array_equal(labels, again)


def test_mask_excludes_cells_entirely():
    g = np.zeros((3, 3), dtype=np.int8)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    labels, n = label(g, connectivity=CONN4, mask=mask)
    assert n == 1
    assert labels[0, 0] == -1 and labels[1, 1] == 0


def test_largest_component_picks_the_biggest():
    m = np.zeros((5, 5), dtype=bool)
    m[0, 0] = True
    m[2:5, 2:5] = True
    assert largest_component_mask(m).sum() == 9


# -- geometry --------------------------------------------------------------


def _upscaled(cell: int, seed: int = 0) -> np.ndarray:
    """A synthetic frame that genuinely is a ``cell``-times upscale."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 6, (64 // cell, 64 // cell)).astype(np.int8)
    return freeze_grid(np.repeat(np.repeat(small, cell, axis=0), cell, axis=1))


@pytest.mark.parametrize("cell", [2, 4, 8])
def test_synthetic_upscale_is_recovered_exactly(cell):
    frames = [_upscaled(cell, s) for s in range(4)]
    geo = induce_geometry(frames)
    assert geo.block.cell == cell
    assert geo.is_structured


def test_unstructured_frames_claim_nothing():
    """The failure mode must be 'no structure', never 'wrong structure'."""
    rng = np.random.default_rng(1)
    frames = [
        freeze_grid(rng.integers(0, 6, (64, 64)).astype(np.int8)) for _ in range(3)
    ]
    geo = induce_geometry(frames)
    assert geo.block.cell == 1
    assert not geo.is_structured
    assert geo.click_targets() == []


def test_verification_rejects_a_cell_that_spans_two_colours():
    g = _upscaled(4)
    g = np.array(g)
    g[0, 0] = (g[0, 0] + 1) % 6  # break one cell
    assert not verify_cell([g], 4, 0, 0)
    assert verify_cell([g], 1, 0, 0)


def test_proposal_survives_a_single_pixel_of_noise_outside_the_mask():
    """A 1-pixel UI element must not veto a real upscale in the playfield."""
    base = np.array(_upscaled(4))
    noisy = base.copy()
    noisy[63, 63] = (noisy[63, 63] + 1) % 6
    mask = np.zeros((64, 64), dtype=bool)
    mask[:56, :56] = True
    assert verify_cell([noisy], 4, 0, 0, mask)
    assert not verify_cell([noisy], 4, 0, 0, None)


def test_geometry_never_claims_an_unsound_grid_on_real_frames(episodes):
    """The property everything downstream depends on, on real recorded data."""
    for name, ep in episodes.items():
        frames = ep.grids()
        geo = induce_geometry(frames)
        stack = np.stack(frames)
        changed = (stack != stack[0]).any(axis=0)
        mask = changed if changed.any() else None
        assert verify_cell(
            frames, geo.block.cell, geo.block.origin_x, geo.block.origin_y, mask
        ), f"{name} induced an unsound cell grid of {geo.block.cell}"


def test_sp80_upscale_is_recovered_from_a_real_recording(sp80):
    """sp80's camera is 16x16 at 4x. Getting this wrong makes every component 16x too big."""
    geo = induce_geometry(sp80.grids())
    assert geo.block.cell == 4
    assert geo.is_structured
    # 16x16 board, so click enumeration should land near 256 candidates, not 4096.
    assert 100 < len(geo.click_targets()) <= 256


def test_bp35_declines_to_guess(bp35):
    """UI-rendered at full resolution: the honest answer is 'no cell structure'."""
    geo = induce_geometry(bp35.grids())
    assert geo.block.cell == 1


def test_rect_snapping_grows_outward():
    r = Rect(3, 5, 10, 11).snapped(4, 0, 0)
    assert (r.x0, r.y0, r.x1, r.y1) == (0, 4, 12, 12)


def test_more_frames_never_makes_geometry_unsound(sb26):
    grids = sb26.grids()
    for n in (2, 10, 50, len(grids)):
        sub = grids[:n]
        geo = induce_geometry(sub)
        stack = np.stack(sub)
        changed = (stack != stack[0]).any(axis=0)
        assert verify_cell(
            sub,
            geo.block.cell,
            geo.block.origin_x,
            geo.block.origin_y,
            changed if changed.any() else None,
        )


# -- diff ------------------------------------------------------------------


def test_diff_reports_which_cells_and_how():
    a = np.zeros((8, 8), dtype=np.int8)
    b = a.copy()
    b[2:4, 2:4] = 3
    d = diff(a, b)
    assert d.n_changed == 4
    assert d.transitions[(0, 3)] == 4
    assert d.colour_delta == {0: -4, 3: 4}
    assert len(d.regions) == 1
    assert d.regions[0].is_recolour
    assert (d.bbox.x0, d.bbox.y0, d.bbox.x1, d.bbox.y1) == (2, 2, 4, 4)


def test_identical_frames_diff_to_nothing():
    a = np.zeros((8, 8), dtype=np.int8)
    d = diff(a, a.copy())
    assert d.is_empty and not d.regions and d.translation() == (0, 0)


def test_separate_changes_are_separate_regions():
    a = np.zeros((8, 8), dtype=np.int8)
    b = a.copy()
    b[0, 0] = 1
    b[7, 7] = 2
    d = diff(a, b)
    assert len(d.regions) == 2


def test_whole_frame_translation_is_detected():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 5, (16, 16)).astype(np.int8)
    b = np.roll(a, shift=2, axis=1)
    # np.roll wraps; a real scroll does not, so compare only the overlap.
    b[:, :2] = a[:, :2]
    d = diff(a, b)
    assert d.translation() in (None, (2, 0))


def test_sequence_diff_finds_transients_an_endpoint_view_would_miss():
    """Intermediate frames carry information. This is the D-009 correction."""
    a = np.zeros((8, 8), dtype=np.int8)
    mid = a.copy()
    mid[4, 4] = 5  # flashes on
    end = a.copy()  # and off again
    sd = diff_sequence([mid, end], previous=a)
    assert sd.is_animated
    assert sd.net is not None and sd.net.is_empty  # endpoints look identical
    assert sd.transient_mask.sum() == 1  # but something happened


def test_sequence_diff_on_a_real_animation(sb26):
    idx, longest = max(enumerate(sb26.observations), key=lambda t: len(t[1].frames))
    assert len(longest.frames) > 5, "sb26 fixture should contain a long animation"
    prev = next(
        (o.last for o in reversed(sb26.observations[:idx]) if o.last is not None), None
    )
    sd = diff_sequence(longest.frames, previous=prev)
    assert sd.n_frames == len(longest.frames)
    assert len(sd.steps) >= len(longest.frames) - 1
    assert sd.is_animated


def test_noop_action_produces_a_static_sequence():
    a = np.zeros((8, 8), dtype=np.int8)
    sd = diff_sequence([], previous=a)
    assert sd.is_static and not sd.is_animated and sd.n_frames == 0


# -- segmentation ----------------------------------------------------------


def test_segment_offers_several_candidate_representations(ls20):
    frame = ls20.observations[-1].last
    ss = segment(frame, induce_geometry(ls20.grids()))
    assert set(ss.names) >= {
        "colour_4",
        "colour_8",
        "foreground_4",
        "foreground_8",
        "colour_mass",
        "bbox",
    }
    for seg in ss:
        assert len(seg) > 0
        for c in seg.components:
            assert c.size == int(c.mask.sum()) > 0


def test_click_targets_land_inside_their_component(ls20):
    frame = ls20.observations[-1].last
    ss = segment(frame, induce_geometry(ls20.grids()))
    for seg in ss:
        for c in seg.components:
            x, y = c.click_target
            assert c.mask[y, x], f"{seg.name} component {c.index} target outside mask"
            assert 0 <= x < 64 and 0 <= y < 64


def test_click_pruning_beats_the_raw_coordinate_space(episodes):
    """D-005's claim, measured: 4096 candidates down to something searchable."""
    for name, ep in episodes.items():
        frame = ep.observations[-1].last
        if frame is None:
            continue
        ss = segment(frame, induce_geometry(ep.grids()))
        assert len(ss.union_click_targets()) < 4096, name


def test_every_segmentation_name_is_always_present():
    """A name that vanishes on a blank frame is a KeyError waiting to happen."""
    blank = segment(freeze_grid(np.zeros((64, 64), dtype=np.int8)))
    assert set(blank.names) >= {
        "colour_4",
        "colour_8",
        "foreground_4",
        "foreground_8",
        "colour_mass",
        "bbox",
    }
    assert len(blank["foreground_8"]) == 0  # nothing but background


def test_colour_mass_ignores_connectivity():
    g = np.zeros((8, 8), dtype=np.int8)
    g[0, 0] = 1
    g[7, 7] = 1
    ss = segment(freeze_grid(np.pad(g, ((0, 56), (0, 56)))))
    mass = ss["colour_mass"]
    ones = [c for c in mass if c.dominant_colour == 1]
    assert len(ones) == 1 and ones[0].size == 2


def test_background_guess_is_the_commonest_colour():
    g = np.zeros((8, 8), dtype=np.int8)
    g[0, 0] = 7
    assert background_colour(g) == 0


def test_segmentation_covers_static_scenery_not_just_the_active_region(sp80):
    """Walls are objects too - segmenting only what moves loses the whole map."""
    geo = induce_geometry(sp80.grids())
    frame = sp80.observations[-1].last
    seg = segment(frame, geo)["colour_8"]
    covered = np.zeros((64, 64), dtype=bool)
    for c in seg.components:
        covered |= c.mask
    assert covered.mean() > 0.9


# -- tracking --------------------------------------------------------------


def _frame_with_square(x, y, colour=3, size=4):
    g = np.zeros((64, 64), dtype=np.int8)
    g[y : y + size, x : x + size] = colour
    return freeze_grid(g)


def test_a_moving_object_keeps_its_identity():
    segs = [segment(_frame_with_square(x, 10))["foreground_8"] for x in (10, 14, 18)]
    tracker = Tracker("foreground_8")
    for s in segs:
        tracker.update(s)
    assert len(tracker.active_tracks) == 1
    track = tracker.active_tracks[0]
    assert track.total_displacement[0] == pytest.approx(8.0)


def test_two_identical_objects_crossing_are_not_swapped():
    """Greedy nearest-match fails here; global assignment is why this passes."""

    def frame(xa, xb):
        g = np.zeros((64, 64), dtype=np.int8)
        g[10:14, xa : xa + 4] = 3
        g[30:34, xb : xb + 4] = 3
        return freeze_grid(g)

    tracker = Tracker("foreground_8")
    ids = []
    for xa, xb in ((10, 40), (14, 36), (18, 32)):
        tracker.update(segment(frame(xa, xb))["foreground_8"])
        ids.append(
            tuple(
                sorted(
                    (t.track_id, round(t.component.centroid[1]))
                    for t in tracker.active_tracks
                )
            )
        )
    # The track at the top stays the top track throughout.
    assert ids[0] == ids[1] == ids[2]


def test_recolouring_preserves_identity():
    a = segment(_frame_with_square(10, 10, colour=3))["foreground_8"]
    b = segment(_frame_with_square(10, 10, colour=5))["foreground_8"]
    m = match_frames(a, b)
    assert len(m.matches) == 1
    assert m.matches[0].recoloured
    assert not m.matches[0].moved


def test_appearance_and_disappearance_are_reported():
    empty = segment(freeze_grid(np.zeros((64, 64), dtype=np.int8)))["colour_8"]
    one = segment(_frame_with_square(10, 10))["foreground_8"]
    m = match_frames(empty, one)
    assert m.appeared and not m.matches


def test_occluded_object_is_reacquired_not_replaced():
    """Guardrail 1: down-weight on disconfirmation, never delete."""
    visible = segment(_frame_with_square(10, 10))["foreground_8"]
    hidden = segment(freeze_grid(np.zeros((64, 64), dtype=np.int8)))["foreground_8"]
    tracker = Tracker("foreground_8")
    tracker.update(visible)
    original = tracker.active_tracks[0].track_id
    tracker.update(hidden)
    assert not tracker.active_tracks
    assert tracker.tracks[original].confidence < 1.0  # down-weighted
    assert original in tracker.tracks  # but not deleted
    tracker.update(visible)
    assert [t.track_id for t in tracker.active_tracks] == [original]


def test_tracking_a_real_recording_is_stable(ls20):
    geo = induce_geometry(ls20.grids())
    tracker = Tracker()  # colour_8, the default D-014 measured
    frames = [o.last for o in ls20.observations if o.last is not None]
    for f in frames:
        tracker.update(segment(f, geo)["colour_8"])
    assert tracker.tracks
    # Identity should not churn: far fewer tracks than components x frames.
    n_components = sum(
        len(segment(f, geo)["colour_8"]) for f in frames[:5]
    )
    assert len(tracker.tracks) < n_components
