"""Goal hypotheses: parsing, sandbox, and soft population updates."""

from __future__ import annotations

import numpy as np
import pytest

from arc3.hypothesis import Population, compile_progress, parse_hypotheses
from arc3.scene import make_scene

GOOD = '''Here are my guesses.
```python
# GOAL: the red block reaches the right edge
def progress(scene):
    reds = scene.of_colour(2)
    if not reds:
        return 0.0
    return reds[0].x1 / 64
```
```python
# GOAL: nothing matters
def progress(scene):
    return 0.5
```
'''


def scene_with_red_at(x):
    f = np.zeros((64, 64), dtype=np.int8)
    f[10:14, x : x + 4] = 2
    return make_scene(f)


def test_parse_finds_each_block_and_goal_line():
    hs = parse_hypotheses(GOOD)
    assert [g for g, _ in hs] == ["the red block reaches the right edge", "nothing matters"]


def test_sandbox_blocks_imports_dunders_and_runaway_loops():
    with pytest.raises(ValueError):
        compile_progress("import os\ndef progress(s):\n    return 1")
    with pytest.raises(ValueError):
        compile_progress("def progress(s):\n    return s.__class__")
    pop = Population()
    pop.add_from_text("```python\n# GOAL: spin\ndef progress(scene):\n    while True:\n        pass\n```")
    h = pop.hypotheses[0]
    assert h.value(b"k", lambda: scene_with_red_at(0)) == 0.0
    assert h.n_errors == 1 and h.weight < 1.0


def test_broken_code_is_kept_but_never_ranks():
    pop = Population()
    pop.add_from_text("```python\ndef progress(scene)\n    return 1\n```")
    h = pop.hypotheses[0]
    assert h.fn is None and h.compile_error
    assert pop.live() == []


def test_values_rank_states_and_win_confirms_the_right_hypothesis():
    pop = Population()
    pop.add_from_text(GOOD)
    right, flat = pop.hypotheses
    near = scene_with_red_at(56)
    far = scene_with_red_at(0)
    assert pop.value(b"near", lambda: near) > pop.value(b"far", lambda: far)
    pop.observe_win(b"near", lambda: near)
    assert right.weight > flat.weight
    assert right.wins_confirmed == 1


def test_false_goal_is_soft():
    pop = Population()
    pop.add_from_text("```python\n# GOAL: always done\ndef progress(scene):\n    return 1.0\n```")
    h = pop.hypotheses[0]
    for i in range(3):
        pop.observe_state(bytes([i]), lambda: scene_with_red_at(0))
    assert h.false_goals == 3
    assert 0.01 <= h.weight < 0.5   # heavily down-weighted, never removed


def test_flat_hypothesis_is_excluded_from_ranking_only():
    pop = Population()
    pop.add_from_text("```python\n# GOAL: constant\ndef progress(scene):\n    return 0.3\n```")
    h = pop.hypotheses[0]
    for i in range(25):
        h.value(bytes([i]), lambda: scene_with_red_at(0))
    assert not h.informative
    assert pop.value(b"x", lambda: scene_with_red_at(0)) is None
    assert h in pop.hypotheses


def test_small_model_format_one_block_renamed_functions():
    """Measured output shape of Qwen2.5-Coder-3B on ls20 (2026-09-27)."""
    text = ("```python\n# GOAL: all one colour\ndef progress_same(scene):\n"
            "    return 1.0 if len(scene.colours()) == 1 else 0.0\n\n"
            "# GOAL: few colours\ndef progress_few(scene):\n"
            "    return 1.0 / len(scene.colours())\n```")
    pop = Population()
    hs = pop.add_from_text(text)
    assert [h.goal for h in hs] == ["all one colour", "few colours"]
    assert all(h.fn is not None for h in hs)
    assert hs[1].value(b"k", lambda: scene_with_red_at(0)) == 0.5
