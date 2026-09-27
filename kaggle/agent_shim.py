# --- MyAgent: the ARC-AGI-3-Agents adapter around arc3.agent.GraphAgent (D-034) ---
#
# Everything above this line is the `arc3` package, embedded as source strings
# and served by an in-memory importer, because the Kaggle starter splices
# exactly one file into the framework and the evaluation has no internet.

import threading
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

from arc3.agent import GraphAgent
from arc3.types import Action, Observation
from arc3.types import GameState as A3State


class MyAgent(Agent):
    """Plays one game with `GraphAgent`. One instance per game, per thread."""

    MAX_ACTIONS = __MAX_ACTIONS__

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._agent = GraphAgent(seed=0, use_model=__USE_MODEL__)
        self._last_frames: tuple = ()
        self._pending_data: dict = {}

    @property
    def name(self) -> str:
        return f"{super().name}.graph.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _observation(self, f: FrameData) -> Observation:
        grids = tuple(np.asarray(g, dtype=np.int8) for g in (f.frame or []))
        if grids:
            self._last_frames = grids
        state = f.state.value if hasattr(f.state, "value") else str(f.state)
        return Observation(
            frames=grids,
            state=A3State(state),
            levels_completed=int(f.levels_completed or 0),
            win_levels=int(f.win_levels or 0),
            available_actions=tuple(int(a) for a in (f.available_actions or [])),
            full_reset=bool(getattr(f, "full_reset", False)),
        )

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Zero frames is a legal return meaning "the action changed nothing"
        # (docs/benchmark.md), so an empty frame list is only a reason to RESET
        # before any frame has ever been seen. GraphAgent carries the last grid
        # forward through no-ops itself.
        if latest_frame.state is GameState.NOT_PLAYED or (
            not latest_frame.frame and not self._last_frames
        ):
            self._pending_data = {}
            return GameAction.RESET
        try:
            a: Action = self._agent.act(self._observation(latest_frame))
        except Exception:  # never let one bad step end the game
            import logging
            logging.getLogger().exception("GraphAgent failed; sending RESET")
            self._agent = GraphAgent(seed=0, use_model=__USE_MODEL__)
            self._pending_data = {}
            return GameAction.RESET
        action = GameAction.from_id(a.id)
        self._pending_data = dict(a.data) if a.is_click else {}
        return action

    def do_action_request(self, action: GameAction) -> FrameData:
        # `GameAction.ACTION6.set_data` mutates a module-level enum member that
        # every game's thread shares, so coordinates set in one thread can be
        # sent by another. Pass this game's own coordinates instead.
        data = dict(self._pending_data) if action.is_complex() else {}
        raw = self.arc_env.step(action, data=data, reasoning=None)
        return self._convert_raw_frame_data(raw)
