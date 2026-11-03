"""Which objects does FactBook5 classify as counters / movers on each game (scripted play)?"""
import random
import sys

sys.path.insert(0, "scripts")
from eval_identity import script  # noqa: E402

from arc3.client.env import EnvClient, available_environments  # noqa: E402
from arc3.types import Action  # noqa: E402
from arc3.v5.facts import FactBook5  # noqa: E402

for g in available_environments():
    c = EnvClient(g, seed=0, record=False)
    rng = random.Random(0)
    fb = FactBook5()
    fb.observe(c.current.last, None)
    seen_c, seen_m = {}, {}
    for a in script(c, rng, 40):
        if c.current.state.is_terminal:
            break
        o = c.step(a)
        if o.last is None:
            continue
        fb.observe(o.last, a)
        for i, d in fb.counters().items():
            ob = fb.objects[i]
            seen_c[i] = f"#{i} col{ob.colour} x{ob.x} y{ob.y} {ob.w}x{ob.h} {d}"
        for i, d in fb.movers().items():
            ob = fb.objects[i]
            seen_m[i] = f"#{i} col{ob.colour} x{ob.x} y{ob.y} {ob.w}x{ob.h} {d}"
    pcs = [p["id"] for p in fb.player_candidates()]
    print(f"{g[:4]} counters: {list(seen_c.values())} | movers: {list(seen_m.values())} | players {pcs[:3]}")
