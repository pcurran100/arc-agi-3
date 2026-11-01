"""The scale-invariant shape graft on the vendored Duck harness (D-045).

Loads ``third_party/duck-franzen/.../segmentation.py`` by path: it is stdlib-only by
design (it is spliced into the harness's sandbox), so no harness dependencies are needed.
"""

import importlib.util
from pathlib import Path

import pytest

from arc3.client.replay import read_episode

ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / "third_party/duck-franzen/ARC3-Inference/inference/utils/segmentation.py"
CHARS = "0123456789abcdef"


def load(monkeypatch, flag: str):
    monkeypatch.setenv("ARC3_SCALE_HASH", flag)
    spec = importlib.util.spec_from_file_location(f"duck_seg_{flag}", SEG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grid_with(*shapes):
    """64x64 grid of colour 0 with each (cells, colour, (r0, c0)) painted in."""
    g = [[0] * 64 for _ in range(64)]
    for cells, colour, (r0, c0) in shapes:
        for r, c in cells:
            g[r0 + r][c0 + c] = colour
    return g


def upscale(cells, k):
    return [(r * k + i, c * k + j) for r, c in cells for i in range(k) for j in range(k)]


L = [(0, 0), (1, 0), (2, 0), (2, 1)]
L_ROT = [(0, 0), (0, 1), (0, 2), (1, 0)]        # L rotated 90 degrees


def test_flag_off_leaves_nodes_unchanged(monkeypatch):
    seg = load(monkeypatch, "0")
    assert seg.SCALE_HASH is False
    nodes = seg.segment_layer(grid_with((L, 3, (5, 5))), CHARS)["nodes"]
    assert all("scale" not in n and "scale_free_hash" not in n for n in nodes)


def test_scaled_rotated_copy_matches(monkeypatch):
    seg = load(monkeypatch, "1")
    g = grid_with((L, 3, (5, 5)), (upscale(L_ROT, 2), 4, (20, 20)), (upscale(L, 3), 5, (40, 40)))
    nodes = {n["color"]: n for n in seg.segment_layer(g, CHARS)["nodes"]}
    small, big2, big3 = nodes["3"], nodes["4"], nodes["5"]
    assert (small["scale"], big2["scale"], big3["scale"]) == (1, 2, 3)
    assert small["scale_free_hash"] == big2["scale_free_hash"] == big3["scale_free_hash"]
    # The existing hashes still tell them apart: the graft only adds keys.
    assert small["shape_hash"] != big2["shape_hash"]


def test_different_shape_does_not_match(monkeypatch):
    seg = load(monkeypatch, "1")
    t = [(0, 0), (0, 1), (0, 2), (1, 1)]
    g = grid_with((L, 3, (5, 5)), (upscale(t, 2), 4, (20, 20)))
    nodes = {n["color"]: n for n in seg.segment_layer(g, CHARS)["nodes"]}
    assert nodes["3"]["scale_free_hash"] != nodes["4"]["scale_free_hash"]


def test_not_a_scale_when_blocks_are_partial(monkeypatch):
    seg = load(monkeypatch, "1")
    # 4x4 bounding box, 2x2 blocks, but one block only half full: not a 2x upscale.
    cells = upscale([(0, 0), (1, 0), (1, 1)], 2) + [(0, 2)]
    assert seg._integer_scale(seg._normalize_cells(cells)) == 1


def test_sandbox_splice_freezes_the_flag(monkeypatch):
    seg = load(monkeypatch, "1")
    src = SEG.read_text(encoding="utf-8")
    line = next(l for l in src.splitlines() if l.startswith("SCALE_HASH = "))
    frozen = src.replace(line, f"SCALE_HASH = {seg.SCALE_HASH!r}", 1)
    assert "SCALE_HASH = True\n" in frozen and "__import__" not in frozen.split("_ORTH")[0]


def test_push_duck_builds_smoke_notebook_with_graft():
    """The notebook anchors push_duck.py edits still exist, and a smoke build with the
    graft writes our patch, applies it after Franzen's, and keeps the chosen games."""
    import argparse
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import push_duck

    args = argparse.Namespace(graft=True, games="ls20,ft09", game_minutes=20,
                              env=["ARC3_SCALE_HASH=1"])
    cells = ["".join(c["source"]) for c in push_duck.build(args)["cells"]]
    assert cells[1] == "import os\nos.environ['ARC3_SCALE_HASH'] = '1'\n"
    graft = [i for i, s in enumerate(cells) if s.startswith("%%writefile /kaggle/arc3-graft.patch")]
    franzen = [i for i, s in enumerate(cells) if s.startswith("%%writefile /kaggle/harness-changes")]
    assert graft and franzen and graft[0] > franzen[0]
    assert "segmentation.py" in cells[graft[0]]
    setup = next(s for s in cells if "/kaggle/arc3-graft.patch\"]" in s)
    assert setup.index("harness-changes.patch") < setup.index("arc3-graft.patch")
    hook = next(s for s in cells if "smoke test (scripts/push_duck.py" in s)
    assert "demo_excluded_games = []" in hook and "max_runtime_s_per_game = 20 * 60" in hook


def test_push_duck_memory_reproduces_lalalia_config():
    """--memory pins the solved-level memory before Franzen's settings in the
    customization cell and applies the three longer timeouts of lalalia9's notebook."""
    import argparse
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import push_duck

    args = argparse.Namespace(graft=False, memory=True, games="", game_minutes=None, env=[])
    text = "\n".join("".join(c["source"]) for c in push_duck.build(args)["cells"])
    hook = text[text.index("SOLVED-LEVEL MEMORY (lalalia9)"):]
    assert hook.index("SOLVED_LEVEL_MEMORY_PATCH on") < hook.index("bm.solver.max_runtime_s_per_game")
    for old, new in push_duck.MEMORY_EDITS:
        assert new in text and old not in text


def test_ls20_key_matches_its_scaled_copy(monkeypatch):
    """On real ls20 level-1 data the forensic analysis (2026-10-29) found the key display
    is the goal glyph at 2x scale and another rotation, which the rotation-only hashes
    could not see. With the graft, some 2x object shares a scale_free_hash with a 1x one."""
    fixture = ROOT / "fixtures/ls20.jsonl.gz"
    if not fixture.exists():
        pytest.skip("no ls20 fixture")
    seg = load(monkeypatch, "1")
    frame = read_episode(fixture).observations[0].frames[-1]
    nodes = seg.segment_layer(frame.tolist(), CHARS)["nodes"]
    small = {n["scale_free_hash"] for n in nodes if n["scale"] == 1 and n["pixels"] > 2}
    big = [n for n in nodes if n["scale"] == 2 and n["scale_free_hash"] in small]
    assert big, "expected a 2x object matching a 1x object on ls20 level 1"
