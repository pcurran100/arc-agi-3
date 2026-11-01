# Solved-level memory (third-party)

`memory_patch.py` is copied verbatim from the customization cell of the public
Kaggle notebook `vickyfym/arc-agi-3-lalalia-final` (pulled 2026-11-01), up to
the line `bm.solver.max_runtime_s_per_game = 48*60`, which begins Franzen's own
configuration and is not part of the patch.

- Author of this version: lalalia9 (Kaggle `vickyfym`), documented in their
  ARC Prize 2026 paper-track writeup "Luck, Tokens and Memory: What Moves a
  Single-GPU LLM Agent on ARC-AGI-3". Public leaderboard score of that notebook
  as stated by its author: 31.46.
- Idea and original code: rellik13 / sirikilohit, Milestone 2 solution "M85",
  CC-BY 4.0 (as stated in the patch header).
- Kaggle notebooks are published under Apache-2.0 unless stated otherwise.

`scripts/push_duck.py --memory` inserts it at the top of Franzen's
customization cell, and applies the same three timeout changes as that
notebook (server startup 12 -> 55 min, HTTP retry 900 -> 3600 s, analyzer
timeout 900 -> 3600 s), so the result is that notebook's configuration.
