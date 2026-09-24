"""The environment client - the only place in the codebase that spends actions.

Two rules this module exists to enforce:

1. **Stepping is visible.** :meth:`EnvClient.step` is the sole way to advance an
   environment, it always returns an :class:`Observation` carrying the running
   ``actions_used``, and nothing else in ``arc3`` may import ``arc_agi``.
   Actions are the scarce resource (CLAUDE.md); internal compute is free.
2. **Errors are not no-ops.** ``LocalEnvironmentWrapper.step`` catches every
   exception and returns ``None``. Silently treating that as "nothing happened"
   would corrupt dynamics induction with fabricated null transitions, so a
   ``None`` return becomes an :class:`Observation` with ``error`` set, and it is
   *not* counted as a no-op.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, runtime_checkable

from ..types import Action, Episode, GameMetadata, GameState, Observation, as_sequence

log = logging.getLogger(__name__)

#: Pinned in pyproject; recorded into every episode so replays are traceable to
#: the engine that produced them (D-008).
ENGINE_VERSION = "arc-agi==0.9.9"


@runtime_checkable
class Client(Protocol):
    """What perception and (later) the agent may assume about an environment.

    :class:`EnvClient` and :class:`~arc3.client.replay.ReplayClient` both satisfy
    this, so every test can run offline against recorded fixtures and no test can
    accidentally step a live environment.
    """

    @property
    def game_id(self) -> str: ...

    @property
    def metadata(self) -> GameMetadata: ...

    @property
    def actions_used(self) -> int: ...

    @property
    def current(self) -> Observation: ...

    def step(self, action: Action) -> Observation: ...


def make_arcade(
    *, offline: bool = True, environments_dir: str = "environment_files"
) -> Any:
    """One engine handle, for callers that build many clients in a process.

    :class:`EnvClient` constructs its own ``Arcade`` when it is not given one,
    which is right for a single agent and wrong for an evaluation harness: each
    instance loads and caches the environment files, so a script that builds one
    client per game per driver holds dozens of copies. `eval_exploration.py`
    building 75 clients exhausted memory and died inside the connected-component
    labeller, which had nothing to do with it — the traceback simply lands
    wherever the next allocation falls.

    Scripts that loop over games should build one of these and pass it to every
    client.
    """
    import arc_agi
    from arc_agi import OperationMode

    quiet = logging.getLogger("arc3.engine")
    quiet.setLevel(logging.ERROR)
    return arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE if offline else OperationMode.NORMAL,
        environments_dir=environments_dir,
        logger=quiet,
    )


class EnvClient:
    """A typed wrapper over ``arc_agi``'s environment wrapper.

    Constructing the client performs the engine's implicit reset, which the
    engine does **not** charge as an action, so ``actions_used`` starts at 0 and
    the initial observation is available as :attr:`current` before any step.
    """

    def __init__(
        self,
        game_id: str,
        *,
        arcade: Any = None,
        seed: int = 0,
        offline: bool = True,
        environments_dir: str = "environment_files",
        record: bool = True,
        withhold_metadata: bool = False,
    ) -> None:
        import arc_agi  # imported lazily: perception must not depend on the engine
        from arc_agi import OperationMode

        self._seed = seed
        if arcade is None:
            quiet = logging.getLogger("arc3.engine")
            quiet.setLevel(logging.ERROR)
            arcade = arc_agi.Arcade(
                operation_mode=OperationMode.OFFLINE if offline else OperationMode.NORMAL,
                environments_dir=environments_dir,
                logger=quiet,
            )
        self._arcade = arcade

        env = arcade.make(game_id, seed=seed)
        if env is None:
            raise RuntimeError(
                f"could not make environment {game_id!r}; "
                f"is {environments_dir}/ populated? see scripts/fetch_environments.py"
            )
        self._env = env

        info = env.info
        meta = GameMetadata(
            game_id=info.game_id,
            win_levels=int(getattr(env.observation_space, "win_levels", 0) or 0),
            title=info.title,
            tags=tuple(info.tags) if info.tags else None,
            baseline_actions=(
                tuple(int(b) for b in info.baseline_actions) if info.baseline_actions else None
            ),
        )
        self._metadata = meta.withheld() if withhold_metadata else meta

        self._actions_used = 0
        self._current = self._wrap(env.observation_space, action=None)
        self.episode: Optional[Episode] = None
        if record:
            self.episode = Episode(
                game_id=self._metadata.game_id,
                seed=seed,
                metadata=self._metadata,
                engine_version=ENGINE_VERSION,
            )
            self.episode.observations.append(self._current)

    # -- Client protocol ---------------------------------------------------

    @property
    def game_id(self) -> str:
        return self._metadata.game_id

    @property
    def metadata(self) -> GameMetadata:
        return self._metadata

    @property
    def actions_used(self) -> int:
        """Environment actions spent so far. This is the score denominator."""
        return self._actions_used

    @property
    def current(self) -> Observation:
        return self._current

    def step(self, action: Action) -> Observation:
        """Spend one action. **This is the expensive call.**

        Counts the action whatever comes back, including errors and no-ops,
        because the environment counts it too.
        """
        from arcengine import GameAction

        self._actions_used += 1
        try:
            raw = self._env.step(GameAction.from_id(action.id), data=action.data)
        except Exception as exc:  # pragma: no cover - wrapper already catches
            log.exception("engine raised on %s", action)
            obs = self._error_observation(action, f"{type(exc).__name__}: {exc}")
        else:
            if raw is None:
                # The wrapper swallowed an exception. Distinct from a no-op.
                obs = self._error_observation(action, "engine wrapper returned None")
            else:
                obs = self._wrap(raw, action=action)
        self._current = obs
        if self.episode is not None:
            self.episode.observations.append(obs)
        return obs

    # -- helpers -----------------------------------------------------------

    def _wrap(self, raw: Any, action: Optional[Action]) -> Observation:
        return Observation(
            frames=as_sequence(raw.frame),
            state=GameState(raw.state.value if hasattr(raw.state, "value") else str(raw.state)),
            levels_completed=int(raw.levels_completed),
            win_levels=int(raw.win_levels),
            available_actions=tuple(int(a) for a in raw.available_actions),
            full_reset=bool(getattr(raw, "full_reset", False)),
            action=action,
            actions_used=self._actions_used,
        )

    def _error_observation(self, action: Action, message: str) -> Observation:
        prev = self._current
        return Observation(
            frames=(),
            state=prev.state,
            levels_completed=prev.levels_completed,
            win_levels=prev.win_levels,
            available_actions=prev.available_actions,
            action=action,
            actions_used=self._actions_used,
            error=message,
        )

    # -- engine handle, for the oracle only --------------------------------

    @property
    def _engine_game(self) -> Any:
        """The live ``ARCBaseGame``. **Test-only** - see D-010.

        Reaching through this from anything on the agent's code path is
        cheating: it exposes exact sprite identities that will not exist on the
        private set. ``tests/test_oracle_boundary.py`` enforces that only
        ``arc3.oracle`` touches it.
        """
        return self._env._game


def available_environments(
    environments_dir: str = "environment_files",
) -> list[str]:
    """Game ids present in the local cache, sorted."""
    import arc_agi
    from arc_agi import OperationMode

    quiet = logging.getLogger("arc3.engine")
    quiet.setLevel(logging.ERROR)
    arcade = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE, environments_dir=environments_dir, logger=quiet
    )
    return sorted(e.game_id for e in arcade.get_environments())
