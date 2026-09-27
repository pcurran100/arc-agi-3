"""Run the built kaggle/build/my_agent.py through the real ARC-AGI-3-Agents loop.

Instantiates `MyAgent` exactly as the framework's Swarm would, on a local
offline `arc_agi` environment instead of the gateway, and calls the framework's
own `Agent.main()`. This checks the adapter, the embedded importer and the
action plumbing, not the gateway.

Usage:
    python kaggle/local_harness.py <path to ARC-AGI-3-Agents checkout> <environment_files> [games...]
"""
import importlib.util
import os
import logging
import sys
import time
import types
from pathlib import Path

fw, env_dir, *games = sys.argv[1:]
games = games or ["ls20", "vc33", "ft09"]

# Load agents.agent without agents/__init__.py, which eagerly imports templates
# with dependencies we do not ship (the Kaggle notebook rewrites it for the same reason).
pkg = types.ModuleType("agents")
pkg.__path__ = [str(Path(fw) / "agents")]
sys.modules["agents"] = pkg

import arc_agi
from arc_agi import OperationMode

spec = importlib.util.spec_from_file_location(
    "my_agent", Path(__file__).parent / "build" / os.environ.get("AGENT_FILE", "my_agent.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

logging.basicConfig(level=logging.WARNING)
arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
for g in games:
    env = arcade.make(g, seed=0)
    agent = mod.MyAgent(card_id="local", game_id=g, agent_name="myagent",
                        ROOT_URL="", record=False, arc_env=env)
    t = time.time()
    agent.main()
    f = agent.frames[-1]
    print(f"{g}: levels {f.levels_completed}/{f.win_levels} state {f.state.name} "
          f"actions {agent.action_counter} {time.time() - t:.1f}s  {agent._agent.describe()}")
