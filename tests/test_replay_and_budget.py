from __future__ import annotations

import numpy as np
import pytest

from arc3.client.budget import ActionBudget, GameResult, LevelResult, level_score
from arc3.client.replay import ReplayClient, ReplayMismatch, read_episode, write_episode
from arc3.types import Action, GameState, Observation, freeze_grid


def test_fixture_episodes_round_trip(tmp_path, episodes):
    for name, ep in episodes.items():
        out = write_episode(ep, tmp_path / f"{name}.jsonl.gz")
        back = read_episode(out)
        assert back.game_id == ep.game_id
        assert back.seed == ep.seed
        assert back.engine_version == ep.engine_version
        assert len(back.observations) == len(ep.observations)
        for a, b in zip(ep.observations, back.observations, strict=True):
            assert a.action == b.action
            assert a.available_actions == b.available_actions
            assert len(a.frames) == len(b.frames)
            for fa, fb in zip(a.frames, b.frames, strict=True):
                assert np.array_equal(fa, fb)


def test_uncompressed_and_compressed_are_the_same(tmp_path, ls20):
    plain = read_episode(write_episode(ls20, tmp_path / "a.jsonl"))
    gz = read_episode(write_episode(ls20, tmp_path / "a.jsonl.gz"))
    assert len(plain.observations) == len(gz.observations)
    assert np.array_equal(plain.observations[-1].last, gz.observations[-1].last)


def test_replay_reproduces_the_recording(ls20):
    client = ReplayClient(ls20)
    assert client.actions_used == 0
    n = 0
    while not client.exhausted:
        action = client.next_action
        assert action is not None
        obs = client.step(action)
        n += 1
        assert obs.actions_used == n or obs.action == action
    assert client.actions_used == ls20.actions_used


def test_replay_refuses_to_diverge(ls20):
    """A replay that quietly returns the wrong frames is worse than no replay."""
    client = ReplayClient(ls20)
    recorded = client.next_action
    assert recorded is not None
    wrong = Action.simple(1 if recorded.id != 1 else 2)
    with pytest.raises(ReplayMismatch):
        client.step(wrong)


def test_replay_exhaustion(ls20):
    client = ReplayClient(ls20)
    client.replay_all()
    assert client.exhausted
    with pytest.raises(ReplayMismatch):
        client.step(Action.simple(1))


def test_replay_satisfies_the_client_protocol(ls20):
    from arc3.client.env import Client

    client = ReplayClient(ls20)
    assert isinstance(client, Client)
    assert client.game_id == ls20.game_id
    assert client.metadata.win_levels > 0


def test_replay_can_withhold_metadata(ls20):
    """D-012's fallback path must actually be exercisable."""
    client = ReplayClient(ls20, withhold_metadata=True)
    assert client.metadata.baseline_actions is None
    assert client.metadata.tags is None


# -- RHAE ------------------------------------------------------------------


def test_level_score_is_quadratic_and_capped():
    assert level_score(100, 100) == pytest.approx(1.0)
    # Twice the human's actions scores a quarter, not a half.
    assert level_score(100, 200) == pytest.approx(0.25)
    assert level_score(100, 400) == pytest.approx(0.0625)
    # Being far more efficient than the human does not bank unlimited credit.
    assert level_score(100, 10) == pytest.approx(1.15)
    assert level_score(100, 10, cap_ratio=True) == pytest.approx(1.15**2)


def test_level_score_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        level_score(100, 0)
    with pytest.raises(ValueError):
        level_score(0, 10)


def _obs(action, levels_completed, full_reset=False):
    return Observation(
        frames=(freeze_grid(np.zeros((64, 64))),),
        state=GameState.NOT_FINISHED,
        levels_completed=levels_completed,
        win_levels=3,
        available_actions=(1,),
        action=action,
        full_reset=full_reset,
    )


def test_budget_segments_actions_by_level():
    budget = ActionBudget("g", total_levels=3, baselines=[10, 20, 30])
    for _ in range(5):
        budget.observe(_obs(Action.simple(1), 0))
    assert budget.actions_this_level == 5
    assert budget.remaining_at_par() == 5
    budget.observe(_obs(Action.simple(1), 1))  # clears level 0 in 6 actions
    assert budget.level_index == 1
    assert budget.actions_this_level == 0
    assert budget.baseline_here() == 20

    result = budget.result()
    lv0 = next(lv for lv in result.levels if lv.index == 0)
    assert lv0.completed and lv0.actions == 6
    assert lv0.score() == pytest.approx(level_score(10, 6))


def test_budget_counts_noops_and_errors():
    """The environment charges for them, so we must too."""
    budget = ActionBudget("g", total_levels=1, baselines=[10])
    noop = Observation(
        frames=(),
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=(1,),
        action=Action.simple(1),
    )
    budget.observe(noop)
    budget.observe(
        Observation(
            frames=(),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=1,
            available_actions=(1,),
            action=Action.simple(1),
            error="boom",
        )
    )
    assert budget.actions_used == 2


def test_budget_survives_withheld_baselines():
    """D-012: with no baselines the planner loses its cost model but not its mind."""
    budget = ActionBudget("g", total_levels=2, baselines=None)
    budget.observe(_obs(Action.simple(1), 0))
    budget.observe(_obs(Action.simple(1), 1))
    assert budget.baseline_here() is None
    assert budget.remaining_at_par() is None
    result = budget.result()
    assert result.score() is None
    assert result.levels_completed == 1


def test_incomplete_game_scores_zero_for_unreached_levels():
    result = GameResult(
        game_id="g",
        levels=(
            LevelResult(0, actions=10, completed=True, baseline=10),
            LevelResult(1, actions=0, completed=False, baseline=10),
        ),
        total_levels=2,
        actions_used=10,
    )
    assert result.score() == pytest.approx(0.5)


def test_full_reset_returns_to_level_zero():
    budget = ActionBudget("g", total_levels=2, baselines=[10, 10])
    budget.observe(_obs(Action.simple(1), 0))
    budget.observe(_obs(Action.simple(0), 0, full_reset=True))
    assert budget.level_index == 0
    assert budget.actions_this_level == 0
    assert budget.actions_used == 2  # the reset still cost an action
