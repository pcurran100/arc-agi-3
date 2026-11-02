"""Turn hints for the Duck harness (kaggle/duck_grafts/turn_hints.py, D-048), on synthetic
frames shaped like ls20's: a player block inside, a budget bar on the bottom border."""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle" / "duck_grafts"))
import turn_hints as th  # noqa: E402

F = types.SimpleNamespace
W = H = 16


def frame(player_x, bar_left, level=1):
    g = [[0] * W for _ in range(H)]
    g[8][player_x] = 9
    for x in range(W):
        g[H - 2][x] = 11 if x < bar_left else 3     # bar: colour 11 drains into colour 3
    return F(grid=tuple(map(tuple, g)), level=level, step=0)


def entries(moves, start_x=8, bar=14, level=1):
    out = [F(action="RESET", frame=frame(start_x, bar, level))]
    x = start_x
    for act, dx, cost, back in moves:
        x = start_x if back else x + dx
        bar = 14 if back else bar - cost
        out.append(F(action=act, frame=frame(x, bar, level)))
    return out


def test_budget_counts_down_per_action():
    es = entries([("LEFT", -1, 1, False), ("RIGHT", 1, 1, False), ("LEFT", -1, 1, False),
                  ("LEFT", -1, 1, False)])
    line = th.th_budget_line(th._th_attempt(es)[1])
    assert "colour 11: -1 cells per action, 10 left -> about 10 actions remain" in line


def test_blocked_moves_do_not_break_the_rate():
    es = entries([("LEFT", -1, 1, False), ("UP", 0, 0, False), ("RIGHT", 1, 1, False),
                  ("LEFT", -1, 1, False)])
    assert "colour 11: -1 cells per action" in th.th_budget_line(th._th_attempt(es)[1])


def test_walking_back_to_the_start_is_not_a_lost_life():
    es = entries([("LEFT", -1, 1, False), ("RIGHT", 1, 1, False)])
    start, trans = th._th_attempt(es)
    assert th.th_life_line(start, trans) is None


def test_lost_life_is_reported():
    es = entries([("LEFT", -1, 1, False), ("LEFT", -1, 1, False), ("LEFT", 0, 0, True)])
    start, trans = th._th_attempt(es)
    assert th.th_life_line(start, trans).startswith("BOARD BACK AT LEVEL START")


def test_effects_count_board_changes_per_action():
    es = entries([("LEFT", -1, 1, False), ("UP", 0, 0, False), ("UP", 0, 0, False)])
    assert "LEFT 1/1, UP 0/2" in th.th_effects_line(es)


def test_hints_are_off_unless_enabled(monkeypatch):
    for k in ("ARC3_HINT_BUDGET", "ARC3_HINT_LIFE", "ARC3_HINT_EFFECTS"):
        monkeypatch.delenv(k, raising=False)
    es = entries([("LEFT", -1, 1, False)] * 5)   # one kind of action needs a run of 5
    assert th.th_hints(es) == []
    monkeypatch.setenv("ARC3_HINT_BUDGET", "1")
    assert len(th.th_hints(es)) == 1
