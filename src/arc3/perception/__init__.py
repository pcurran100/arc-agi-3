"""Exact symbolic perception (D-006).

Pipeline order matters: geometry first, because segmentation and tracking are
both wrong at the wrong granularity (D-011).

    geometry  -> what cell size does this game's content move on?
    segment   -> several candidate carvings into objects
    track     -> which object in frame t is which in frame t+1
    diff      -> what changed, structurally, between two frames

Nothing here imports the engine; everything runs from recorded episodes.
"""

from .components import CONN4, CONN8, label
from .diff import FrameDiff, SequenceDiff, diff, diff_sequence
from .geometry import BlockGrid, Geometry, Rect, induce_geometry
from .segment import Component, Segmentation, SegmentationSet, segment
from .track import FrameMatching, Match, Track, Tracker, match_frames

__all__ = [
    "CONN4",
    "CONN8",
    "BlockGrid",
    "Component",
    "FrameDiff",
    "FrameMatching",
    "Geometry",
    "Match",
    "Rect",
    "Segmentation",
    "SegmentationSet",
    "SequenceDiff",
    "Track",
    "Tracker",
    "diff",
    "diff_sequence",
    "induce_geometry",
    "label",
    "match_frames",
    "segment",
]
