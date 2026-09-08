"""Connected-component labelling over 64x64 grids.

Kept separate from :mod:`arc3.perception.segment` because
:mod:`arc3.perception.geometry` needs it too, and geometry must not depend on
segmentation (segmentation depends on geometry).

Pure numpy, no scipy: the grids are 4096 cells, a vectorised neighbour-edge list
resolved by union-find is exact and runs in about 4 ms, and it keeps the offline
Kaggle dependency set to what ``arc-agi`` already pulls in.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

#: 4-connectivity: orthogonal neighbours only.
CONN4 = 4
#: 8-connectivity: orthogonal plus diagonal.
CONN8 = 8


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        p = self.parent
        root = a
        while p[root] != root:
            root = p[root]
        while p[a] != root:  # path compression
            p[a], a = root, p[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def new(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1


def label(
    values: np.ndarray,
    *,
    connectivity: int = CONN4,
    mask: Optional[np.ndarray] = None,
    equal_values: bool = True,
) -> tuple[np.ndarray, int]:
    """Label connected regions of ``values``.

    Args:
        values: 2D array. With ``equal_values`` (the default), neighbours join
            only when they hold the same value - components of one colour. With
            ``equal_values=False``, ``values`` is treated as a boolean mask and
            every truthy neighbour joins regardless of value, which gives
            multi-colour components.
        connectivity: :data:`CONN4` or :data:`CONN8`.
        mask: optional boolean array; ``False`` cells never belong to any
            component.

    Returns:
        ``(labels, n)`` where ``labels`` is ``int32`` with ``-1`` for excluded
        cells and ``0..n-1`` for components, numbered in raster order of their
        first pixel so the labelling is deterministic (D-006).
    """
    if connectivity not in (CONN4, CONN8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    arr = np.asarray(values)
    h, w = arr.shape
    active = np.ones((h, w), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not equal_values:
        active = active & (arr != 0)

    # Build the edge list vectorised, then union-find over it. Much faster than
    # a per-pixel scan and exactly equivalent: two active neighbours are joined
    # iff they would have been merged by a raster scan with label resolution.
    def _edges(da: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
        """Flat index pairs for neighbours offset by ``(da, dx)`` rows/cols."""
        ys0, ys1 = max(0, -da), h - max(0, da)
        xs0, xs1 = max(0, -dx), w - max(0, dx)
        if ys1 <= ys0 or xs1 <= xs0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        a_sl = (slice(ys0, ys1), slice(xs0, xs1))
        b_sl = (slice(ys0 + da, ys1 + da), slice(xs0 + dx, xs1 + dx))
        ok = active[a_sl] & active[b_sl]
        if equal_values:
            ok &= arr[a_sl] == arr[b_sl]
        if not ok.any():
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        flat = np.arange(h * w, dtype=np.int64).reshape(h, w)
        return flat[a_sl][ok], flat[b_sl][ok]

    offsets = [(0, 1), (1, 0)]
    if connectivity == CONN8:
        offsets += [(1, 1), (1, -1)]

    uf = _UnionFind(h * w)
    for da, dx in offsets:
        ia, ib = _edges(da, dx)
        for a_i, b_i in zip(ia.tolist(), ib.tolist(), strict=True):
            uf.union(a_i, b_i)

    out = np.full((h, w), -1, dtype=np.int32)
    remap: dict[int, int] = {}
    n = 0
    flat_active = active.ravel()
    flat_out = out.ravel()
    for idx in np.flatnonzero(flat_active).tolist():
        root = uf.find(idx)
        label_id = remap.get(root)
        if label_id is None:
            label_id = n
            remap[root] = label_id
            n += 1
        flat_out[idx] = label_id
    return out, n


def component_masks(labels: np.ndarray, n: int) -> list[np.ndarray]:
    """One boolean mask per component index."""
    return [labels == i for i in range(n)]


def largest_component_mask(mask: np.ndarray, *, connectivity: int = CONN8) -> np.ndarray:
    """The biggest connected region of a boolean mask, as a boolean mask."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    labels, n = label(m.astype(np.int8), connectivity=connectivity, equal_values=False)
    if n == 0:
        return m
    sizes = np.bincount(labels[labels >= 0].ravel(), minlength=n)
    return labels == int(np.argmax(sizes))


def component_sizes(labels: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    return np.bincount(labels[labels >= 0].ravel(), minlength=n)
