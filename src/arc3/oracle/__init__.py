"""Ground-truth oracle. TEST AND EVALUATION ONLY - see D-010.

Importing this from anything on the agent's code path is cheating: it exposes
exact sprite identities that do not exist on the private evaluation set.
"""

from .sprites import (
    BACKGROUND,
    LETTERBOX,
    UI,
    OracleFrame,
    SpriteInfo,
    ground_truth,
    true_geometry,
)

__all__ = [
    "BACKGROUND",
    "LETTERBOX",
    "UI",
    "OracleFrame",
    "SpriteInfo",
    "ground_truth",
    "true_geometry",
]
