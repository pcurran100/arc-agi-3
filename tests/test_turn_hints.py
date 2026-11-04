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
    es = entries([("LEFT", -1, 1, False)] * 3 + [("LEFT", 0, 0, True)])
    start, trans = th._th_attempt(es)
    assert th.th_life_line(start, trans).startswith("BOARD BACK AT LEVEL START: after your last action")


def test_lost_life_inside_a_batch_is_reported():
    # Local run 2026-11-02 (ls20): the model batched moves, lost a life mid-batch and moved on;
    # only the last transition was checked, so none of four lost lives was reported.
    es = entries([("LEFT", -1, 1, False)] * 3 + [("LEFT", 0, 0, True), ("RIGHT", 1, 1, False)])
    start, trans = th._th_attempt(es)
    assert th.th_life_line(start, trans) is None            # new=1: the last action only
    line = th.th_life_line(start, trans, new=3)              # the three actions of the batch
    assert line.startswith("BOARD BACK AT LEVEL START: 1 action before your last one (after LEFT)")


def test_board_toggled_back_is_not_a_lost_life():
    # vc33 (local run 2026-11-02): board pieces reach into the border band, so clicking a
    # toggle and clicking it back restores the board AND moves the band toward its start
    # counts, while the real budget bar keeps draining. The old test fired 35 times there.
    def vframe(on, bar_left):
        g = [[0] * W for _ in range(H)]
        for y in range(4, 9):
            g[y][1] = g[y][2] = 5 if on else 0             # a piece in the left band
            g[y][6] = 5 if on else 0
        for x in range(W):
            g[H - 2][x] = 7 if x < bar_left else 4        # budget: 7 drains into 4
        return F(grid=tuple(map(tuple, g)), level=1, step=0)

    es = [F(action="RESET", frame=vframe(False, 14))]
    for i in range(1, 9):
        es.append(F(action="ACTION6", frame=vframe(i % 2 == 1, 14 - i)))
    start, trans = th._th_attempt(es)
    assert all(th.th_life_line(start, trans[:k], new=k) is None for k in range(2, len(trans) + 1))


def test_hints_check_every_action_since_the_previous_turn(monkeypatch):
    # th_install passes `new`: the number of entries since the previous turn message.
    monkeypatch.setenv("ARC3_HINT_LIFE", "1")
    es = entries([("LEFT", -1, 1, False)] * 3 + [("LEFT", 0, 0, True), ("RIGHT", 1, 1, False)])
    assert any(x.startswith("BOARD BACK") for x in th.th_hints(es, new=3))
    assert not any(x.startswith("BOARD BACK") for x in th.th_hints(es, new=1))


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


def test_install_passes_the_entries_new_since_the_previous_turn(monkeypatch):
    # th_install against stand-ins for the harness modules: a turn message after a batch of
    # three actions reports a life lost in the middle of it; a resumed turn on the same
    # step repeats the same hints; the next step no longer reports it.
    monkeypatch.setenv("ARC3_HINT_LIFE", "1")
    state = {"entries": []}

    class ToolAgent:
        def analyze(self, state_path):
            return self._build_user_prompt()

        def _build_user_prompt(self):
            return "turn"

    ta = types.ModuleType("inference.agent.tool_agent")
    ta.ToolAgent = ToolAgent
    rs = types.ModuleType("inference.agent.runtime_state")
    rs.load_runtime_state = lambda path: (None, state["entries"])
    for name, mod in [("inference", types.ModuleType("inference")),
                      ("inference.agent", types.ModuleType("inference.agent")),
                      ("inference.agent.tool_agent", ta), ("inference.agent.runtime_state", rs)]:
        monkeypatch.setitem(sys.modules, name, mod)
    th.th_install()
    agent = ToolAgent()
    moves = [("LEFT", -1, 1, False)] * 3 + [("LEFT", 0, 0, True), ("RIGHT", 1, 1, False), ("LEFT", -1, 1, False)]
    state["entries"] = entries(moves[:3])
    assert "BOARD BACK" not in agent.analyze("p")
    state["entries"] = entries(moves[:5])                # a batch of two: the life is lost inside it
    assert "BOARD BACK AT LEVEL START: 1 action before your last one" in agent.analyze("p")
    assert "BOARD BACK" in agent.analyze("p")            # resumed turn, same step
    state["entries"] = entries(moves)
    assert "BOARD BACK" not in agent.analyze("p")
