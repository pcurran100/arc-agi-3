# Adaptive Duck tools (third-party)

`install_cell.py` and `selftest_cell.py` are copied verbatim from cells 17 and 19
of the public Kaggle notebook `laimerbotrolenk/lb-score-31-21` ("ARC-AGI-3 ·
Adaptive Duck", pulled 2026-11-01). Its author states a public leaderboard
score of 31.21 in the title and that the full-model practice score was not
measured. Kaggle notebooks are published under Apache-2.0 unless stated
otherwise.

What it adds to the Duck sandbox: `board()`, `scene()` (click points inside
components), JSON `notes` kept across history trimming, `solved_levels`,
`route()` (shortest paths on a mask), `bounded_search()` (A*/uniform cost in the
agent's own simulator), `check_model()` (tests the agent's simulator against
recorded transitions), `follow_plan()` (stops at the first mismatch) and
`solve_toggle()` (exact XOR toggle solving), plus system-prompt guidance.

`scripts/push_duck.py --adaptive` inserts both cells after the bundled-source
import cell, with `PROFILE = 'adaptive'`.
