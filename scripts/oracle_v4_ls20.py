"""Does the subgoal walker finish ls20 level 1 when told the right subgoals?
(The mechanics check behind D-044; the model is scripted, not an LLM.)"""
import json
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from arc3.client.env import EnvClient, make_arcade
from arc3.llm import ScriptedLLM
from arc3.v4.agent import V4Agent

ED = __import__('os').environ.get('ARC3_ENV_DIR', 'environment_files')
arc = make_arcade(environments_dir=ED)
c = EnvClient('ls20', seed=0, record=False, arcade=arc, environments_dir=ED)
agent = None


def mv(dx, dy):
    return {"changes": [{"target": "#7", "change": "moved", "dx": dx, "dy": dy, "colour": None}],
            "rest": "any", "outcome": "continue"}


def mind(messages):
    text = messages[-1]["content"]
    text = text[0]["text"] if isinstance(text, list) else text
    if "TASK: UPDATE" in text:
        return json.dumps({"ops": [], "summary": ""})
    fb = agent.fb
    moves = {p["id"]: p for p in fb.player_candidates()}.get(7, {}).get("moves", {})
    if len(moves) < 4:
        return json.dumps({"steps": [{"action": "ACTION1", "expect": mv(0, -5)},
                                     {"action": "ACTION3", "expect": mv(-5, 0)},
                                     {"action": "ACTION4", "expect": mv(5, 0)},
                                     {"action": "ACTION2", "expect": mv(0, 5)}],
                           "plan": [], "purpose": "learn moves", "tests": []})
    first = [] if agent.stats.get("subgoals_done") else [
        {"object": "#7", "to": sys.argv[1], "relation": "overlap",
         "on_arrival": {"changes": [], "rest": "any", "outcome": "continue"}, "goal": ""}]
    return json.dumps({"steps": [], "plan": first + [
        {"object": "#7", "to": sys.argv[2], "relation": "overlap",
         "on_arrival": {"changes": [], "rest": "any", "outcome": "level_complete"}, "goal": ""}],
        "purpose": "changer then goal", "tests": []})


agent = V4Agent(llm=ScriptedLLM(mind))
obs = c.current
for i in range(int(sys.argv[3]) if len(sys.argv) > 3 else 60):
    obs = c.step(agent.act(obs))
    if obs.levels_completed:
        print("LEVEL 1 DONE after", c.actions_used, "actions")
        break
print(agent.describe())
print("\n".join(agent.history[-14:]))
print("\n".join(agent.events[-8:]))
