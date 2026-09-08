"""Recording and replay.

Every interaction is written to a self-describing JSONL file so that experiments
are reproducible offline without spending an action (D-008). The engine's own
``save_recording`` writes something similar, but omits the seed, the metadata and
the engine version - the three things you need to reproduce a run - so we keep
our own format and read it back into the same :class:`~arc3.types.Episode` type
the live client produces.

:class:`ReplayClient` satisfies the same :class:`~arc3.client.env.Client`
protocol as the live client, so perception tests run against real recorded
frames with no engine installed and no possibility of stepping a live
environment by accident.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import IO, Any, Iterable, Optional

from ..types import Action, Episode, GameMetadata, Observation

FORMAT_VERSION = 1


def _open(path: Path, mode: str) -> IO[str]:
    """Open ``path``, transparently gzipping when the name ends ``.gz``.

    Frames are extremely repetitive, so gzip shrinks a recorded episode by
    roughly 40x. That is the difference between fixtures being committable and
    not, and the format is otherwise unchanged.
    """
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")  # type: ignore[return-value]
    return path.open(mode, encoding="utf-8")


def write_episode(episode: Episode, path: str | Path, *, include_frames: bool = True) -> Path:
    """Write ``episode`` to a JSONL file. First line is the header."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "format_version": FORMAT_VERSION,
        "kind": "header",
        "game_id": episode.game_id,
        "seed": episode.seed,
        "engine_version": episode.engine_version,
        "metadata": episode.metadata.to_json(),
        "notes": episode.notes,
        "n_observations": len(episode.observations),
        "actions_used": episode.actions_used,
    }
    with _open(p, "w") as f:
        f.write(json.dumps(header) + "\n")
        for obs in episode.observations:
            f.write(json.dumps(obs.to_json(include_frames=include_frames)) + "\n")
    return p


def read_episode(path: str | Path) -> Episode:
    """Read an episode written by :func:`write_episode`."""
    p = Path(path)
    with _open(p, "r") as f:
        lines = [ln for ln in (line.strip() for line in f) if ln]
    if not lines:
        raise ValueError(f"empty episode file: {p}")
    header = json.loads(lines[0])
    if header.get("kind") != "header":
        raise ValueError(f"{p} does not start with an episode header")
    if header.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{p} is format version {header.get('format_version')}, "
            f"this build reads {FORMAT_VERSION}"
        )
    episode = Episode(
        game_id=header["game_id"],
        seed=int(header["seed"]),
        metadata=GameMetadata.from_json(header["metadata"]),
        engine_version=header.get("engine_version", ""),
        notes=header.get("notes", {}),
    )
    episode.observations = [Observation.from_json(json.loads(ln)) for ln in lines[1:]]
    return episode


def iter_episodes(directory: str | Path) -> Iterable[Episode]:
    """Every ``*.jsonl`` or ``*.jsonl.gz`` episode under ``directory``."""
    paths = sorted(
        list(Path(directory).glob("*.jsonl")) + list(Path(directory).glob("*.jsonl.gz"))
    )
    for p in paths:
        yield read_episode(p)


class ReplayMismatch(RuntimeError):
    """Raised when a replay is stepped with an action the recording did not take."""


class ReplayClient:
    """Replays a recorded :class:`~arc3.types.Episode` as if it were live.

    Stepping with an action other than the one recorded raises
    :class:`ReplayMismatch` rather than silently returning the wrong frames -
    a replay that quietly diverges from its recording is worse than no replay.
    """

    def __init__(self, episode: Episode, *, withhold_metadata: bool = False) -> None:
        if not episode.observations:
            raise ValueError("cannot replay an episode with no observations")
        self._episode = episode
        self._metadata = episode.metadata.withheld() if withhold_metadata else episode.metadata
        self._index = 0
        self._actions_used = 0

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "ReplayClient":
        return cls(read_episode(path), **kwargs)

    @property
    def game_id(self) -> str:
        return self._episode.game_id

    @property
    def metadata(self) -> GameMetadata:
        return self._metadata

    @property
    def actions_used(self) -> int:
        return self._actions_used

    @property
    def current(self) -> Observation:
        return self._episode.observations[self._index]

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._episode.observations) - 1

    @property
    def next_action(self) -> Optional[Action]:
        """The action the recording takes next, or None at the end."""
        if self.exhausted:
            return None
        return self._episode.observations[self._index + 1].action

    def step(self, action: Action) -> Observation:
        if self.exhausted:
            raise ReplayMismatch(
                f"replay of {self.game_id} is exhausted after "
                f"{self._actions_used} actions; asked for {action}"
            )
        nxt = self._episode.observations[self._index + 1]
        if nxt.action != action:
            raise ReplayMismatch(
                f"replay of {self.game_id} step {self._actions_used + 1}: "
                f"recorded {nxt.action}, asked for {action}"
            )
        self._index += 1
        self._actions_used += 1
        return nxt

    def replay_all(self) -> Episode:
        """Run the recording to its end and return the episode."""
        while not self.exhausted:
            action = self.next_action
            assert action is not None
            self.step(action)
        return self._episode
