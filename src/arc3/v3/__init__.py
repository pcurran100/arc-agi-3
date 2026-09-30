"""v3: the LLM reasons in the loop; exact code is its senses, tools and referee (D-039).

Modules, in the order information flows:

- `facts` — what is exactly true about the frames: tracked objects, groups,
  static objects, co-motion, transformations, per-action effects, player
  candidates. Deterministic, recomputed every step.
- `world` — the state the LLM's world model runs on, the sandbox that runs it,
  the verifier that checks it against every recorded transition, and the
  planner that searches it for the shortest path to the goal.
- `prompts` — the narrow, structured calls the LLM is asked to make.
- `agent` — the controller that ties them into assess → plan → act → verify →
  revise.
"""
