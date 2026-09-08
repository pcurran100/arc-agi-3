"""Live-engine tests. Marked ``engine``; skipped without ``environment_files/``.

These are the only tests that spend actions. They spend them against the local
cached engine, where actions are free (D-008) - but the habit of counting them is
not, so every one of these asserts on ``actions_used``.
"""

from __future__ import annotations

import numpy as np
import pytest

from arc3.client.env import EnvClient, available_environments
from arc3.client.replay import ReplayClient, read_episode, write_episode
from arc3.types import Action, GameState

pytestmark = pytest.mark.engine


def test_all_twenty_five_demo_environments_are_present():
    games = available_environments()
    assert len(games) == 25, f"expected 25 public demo games, found {len(games)}"


def test_reset_is_not_charged_as_an_action():
    """make() performs the reset; the engine does not bill it, so nor do we."""
    c = EnvClient("ls20", record=False)
    assert c.actions_used == 0
    assert c.current.state == GameState.NOT_FINISHED
    assert len(c.current.frames) >= 1
    assert c.current.last.shape == (64, 64)


def test_every_step_costs_exactly_one_action():
    c = EnvClient("ls20", record=False)
    for i in range(1, 6):
        obs = c.step(Action.simple(c.current.available_actions[0]))
        assert c.actions_used == i
        assert obs.actions_used == i


def test_metadata_matches_the_frame_stream():
    c = EnvClient("ls20", record=False)
    assert c.metadata.game_id.startswith("ls20")
    assert c.metadata.win_levels == c.current.win_levels
    assert c.metadata.baseline_actions is not None
    assert len(c.metadata.baseline_actions) == c.current.win_levels


def test_metadata_can_be_withheld_at_the_client():
    """D-012's fallback must be reachable from the real client, not just replay."""
    c = EnvClient("ls20", record=False, withhold_metadata=True)
    assert c.metadata.baseline_actions is None
    assert c.metadata.tags is None
    assert c.metadata.win_levels > 0  # still knows what came in the frame


def test_available_actions_are_a_strict_subset_of_the_action_space():
    for game in ("ls20", "vc33", "ar25"):
        c = EnvClient(game, record=False)
        acts = c.current.available_actions
        assert acts, f"{game} published no available actions"
        assert set(acts) <= {1, 2, 3, 4, 5, 6, 7}


def test_click_only_games_publish_only_the_click():
    c = EnvClient("vc33", record=False)
    assert c.current.available_actions == (6,)


def test_engine_is_deterministic_for_a_fixed_seed():
    """Everything logged depends on this. Verified across all 25 games in D-009."""
    seqs = []
    for _ in range(2):
        c = EnvClient("ls20", seed=0, record=False)
        frames = []
        for a in (1, 2, 3, 4, 1, 2):
            if a in c.current.available_actions:
                obs = c.step(Action.simple(a))
                frames.append(obs.last.copy() if obs.last is not None else None)
        seqs.append(frames)
    for a, b in zip(*seqs, strict=True):
        assert (a is None and b is None) or np.array_equal(a, b)


def test_a_recorded_episode_replays_identically():
    """The property the whole offline workflow rests on."""
    c = EnvClient("ls20", seed=0, record=True)
    for _ in range(8):
        acts = c.current.available_actions
        if not acts:
            break
        c.step(Action.simple(acts[0]))
    assert c.episode is not None

    replay = ReplayClient(c.episode)
    live_frames = [o.last for o in c.episode.observations]
    seen = [replay.current.last]
    while not replay.exhausted:
        action = replay.next_action
        assert action is not None
        seen.append(replay.step(action).last)
    assert len(seen) == len(live_frames)
    for a, b in zip(live_frames, seen, strict=True):
        assert (a is None and b is None) or np.array_equal(a, b)
    assert replay.actions_used == c.actions_used


def test_episode_survives_a_round_trip_to_disk(tmp_path):
    c = EnvClient("vc33", seed=0, record=True)
    for _ in range(5):
        c.step(Action.click(32, 32))
    assert c.episode is not None
    back = read_episode(write_episode(c.episode, tmp_path / "e.jsonl.gz"))
    assert back.actions_used == c.actions_used
    assert back.engine_version == c.episode.engine_version
    assert np.array_equal(back.observations[-1].last, c.episode.observations[-1].last)


def test_noop_actions_occur_and_are_not_errors():
    """bp35 returns zero frames for many legal actions. That is data, not a fault."""
    c = EnvClient("bp35", seed=0, record=False)
    saw_noop = False
    for _ in range(30):
        acts = c.current.available_actions
        obs = c.step(
            Action.click(32, 32) if acts[0] == 6 else Action.simple(acts[0])
        )
        if obs.is_noop:
            saw_noop = True
            assert not obs.is_error
            assert obs.last is None
    assert saw_noop, "expected at least one zero-frame response from bp35"


def test_long_animations_are_delivered_whole():
    """Up to 42 frames for one action. Reducing to the last frame throws this away."""
    import random

    # Which action animates is game-specific, so sample rather than assuming:
    # sb26 publishes (5, 6, 7) and only ACTION5 produces a long sequence.
    rng = random.Random(0)
    c = EnvClient("sb26", seed=0, record=False)
    longest = 0
    for _ in range(30):
        acts = c.current.available_actions
        if not acts or c.current.state.is_terminal:
            break
        a = rng.choice(acts)
        obs = c.step(
            Action.click(rng.randrange(64), rng.randrange(64))
            if a == 6
            else Action.simple(a)
        )
        longest = max(longest, len(obs.frames))
    assert longest > 5, f"sb26 should animate; longest sequence was {longest}"


def test_oracle_reproduces_the_engine_render_exactly():
    """If this drifts, every oracle-scored number in the repo is suspect."""
    from arc3.oracle import ground_truth

    for game in available_environments():
        c = EnvClient(game, record=False)
        gt = ground_truth(c._engine_game, rendered=c.current.last)
        assert np.array_equal(gt.colour_map, c.current.last), game


def test_oracle_geometry_is_self_consistent():
    from arc3.oracle import ground_truth, true_geometry

    for game in ("ls20", "sp80", "bp35", "m0r0"):
        c = EnvClient(game, record=False)
        truth = true_geometry(c._engine_game)
        gt = ground_truth(c._engine_game, rendered=c.current.last)
        x0, y0, x1, y1 = gt.playfield
        assert x1 - x0 == truth["camera_w"] * truth["scale"]
        assert y1 - y0 == truth["camera_h"] * truth["scale"]
