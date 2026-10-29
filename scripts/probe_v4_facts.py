"""Print what v4's fact layer (FactBook4) reports after a fixed action sequence, and
optional routes for #7 to the given targets (D-044).

Usage: ARC3_ENV_DIR=... python scripts/probe_v4_facts.py ls20 1,1,3,3,4,4 [#5 #3] [-q]
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from arc3.client.env import EnvClient, make_arcade
from arc3.types import Action
from arc3.v4.facts import FactBook4

ED = __import__('os').environ.get('ARC3_ENV_DIR', 'environment_files')
arc = make_arcade(environments_dir=ED)
c = EnvClient(sys.argv[1], seed=0, record=False, arcade=arc, environments_dir=ED)
fb = FactBook4()
fb.observe(c.current.last)
seq = [int(a) for a in sys.argv[2].split(',')]
for a in seq:
    obs = c.step(Action.simple(a))
    s = fb.observe(obs.last, Action.simple(a))
    print(f"t{s.t} A{a}:", '; '.join(ch.describe() for ch in s.changes) or '-',
          '| noop' if fb.is_noop(s) else '', fb.notes.get(s.t), obs.state.name, obs.levels_completed)
if '-q' not in sys.argv:
    print(fb.describe())
from arc3.v4.route import Subgoal, route
for to in [a for a in sys.argv[3:] if a != '-q']:
    sg = Subgoal.parse({"object": "#7", "to": to, "relation": "overlap"})
    r, why = route(fb, sg, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
    print(to, why, [a for a, _, _ in (r or [])])
