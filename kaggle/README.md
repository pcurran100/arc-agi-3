# Submitting to Kaggle

The submission is one notebook whose agent cell is generated from this
repository. Nothing in it needs the internet at evaluation.

## Build

```bash
python scripts/build_kaggle.py --max-actions 2000          # model-free graph agent
python scripts/build_kaggle.py --max-actions 2000 --model  # with the induced model (slower)
```

This writes `kaggle/build/my_agent.py` (the `arc3` package embedded as source
plus the `MyAgent` adapter in `kaggle/agent_shim.py`) and
`kaggle/build/submission.ipynb`. The build directory is gitignored: rebuild, do
not edit.

## Check locally before uploading

```bash
git clone --depth 1 https://github.com/arcprize/ARC-AGI-3-Agents /tmp/fw
python kaggle/local_harness.py /tmp/fw environment_files ls20 vc33 sp80
```

This instantiates `MyAgent` exactly as the framework's Swarm does and runs the
framework's own `Agent.main()` loop against the local offline engine. It checks
the adapter, the embedded importer and the action plumbing. It does **not**
check the gateway, which only exists inside Kaggle's competition rerun.

## Upload

1. Open https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 and accept
   the rules if you have not.
2. New notebook -> File -> Import notebook -> `kaggle/build/submission.ipynb`.
3. Attach the competition dataset (Add input -> the ARC-AGI-3 competition). The
   notebook installs `arc-agi` from its `arc_agi_3_wheels` folder.
4. Settings: Internet **off**, accelerator **None/CPU** (the agent uses no GPU).
5. Save Version -> Save & Run All. The commit run writes a dummy
   `submission.parquet`; that is expected.
6. Submit to Competition from the saved version. The real run happens in
   Kaggle's rerun, where the gateway plays every hidden game and writes the real
   `submission.parquet`.

## What is known and not known about the harness

Verified from the official starter
(github.com/arcprize/ARC-AGI-3-Kaggle-Starter) and framework
(github.com/arcprize/ARC-AGI-3-Agents) source on 2026-09-26:

- The class must be named `MyAgent`, subclass `agents.agent.Agent`, and
  implement `choose_action` and `is_done`.
- The loop stops at `MAX_ACTIONS` per game (framework default 80; ours is set
  at build time) or when `is_done` returns True. RESET counts as an action.
- Competition mode permits level resets only; a full-game reset becomes a level
  reset. Our agent only ever uses RESET to restart the current level.
- The Swarm runs games in parallel threads. `GameAction.ACTION6.set_data`
  mutates a shared enum member, so `MyAgent.do_action_request` sends its own
  per-instance coordinates instead.

Not verified (could not read the Kaggle rules tab): the runtime limit, the
number of hidden games, and the exact Milestone #2 writeup requirement.
Milestone and prize eligibility require the code to be open-sourced under a
permissive licence; **this repository has no LICENSE file yet.**
