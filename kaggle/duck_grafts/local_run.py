# SPDX-License-Identifier: MIT-0
# ARC-AGI-3 project (MIT-0). Runs the vendored Duck harness (third_party/duck-franzen,
# Apache-2.0) OFFLINE on this machine against a hosted OpenAI-compatible API instead of the
# Kaggle SGLang server. Reproduces the benchmark construction of Franzen's Milestone-2
# notebook (cells 4, 16, 20) and, by arm, the changes scripts/push_duck.py makes on Kaggle:
#
#   A = Franzen + solved-level memory (--memory: memory_patch.py + its timeout edits)
#   D = A + turn hints (--hints, all three) + EXPOSE_RESET=on + ARC3_SCALE_HASH=1
#
# It must be run with the harness venv (third_party/duck-franzen/ARC3-Inference/.venv), e.g.
#
#   .venv/Scripts/python.exe kaggle/duck_grafts/local_run.py --arm A --games ls20,ft09 \
#       --minutes 25 --out runs/duck_ab/A
#
# The API key is read from NVIDIA_TOKEN in the repo-root .env and never printed.
# Every game step happens inside the harness (bm.run); this script takes no actions itself.
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEMORY = ROOT / "third_party" / "solved-level-memory" / "memory_patch.py"
HINTS = ROOT / "kaggle" / "duck_grafts" / "turn_hints.py"
DEFAULT_ENV_DIR = ROOT.parent.parent.parent / "environment_files" if ".claude" in ROOT.parts else ROOT / "environment_files"
HINT_MARKERS = {"budget": "BUDGET BAR (measured by code", "life": "BOARD BACK AT LEVEL START",
                "effects": "ACTION EFFECTS this level"}


def load_key(env_file: Path, name: str) -> str:
    for line in env_file.read_text(encoding="utf-8").splitlines():
        m = re.match(rf"\s*{name}\s*=\s*(.*)", line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    raise SystemExit(f"{name} not found in {env_file}")


def harness_env(args: argparse.Namespace) -> dict[str, str]:
    """Franzen's notebook env block (cell 4), with the server pointed at the hosted API.
    Changes from the notebook are marked LOCAL."""
    env = {
        'ARC3_WARMUP_ACTION_GAMES': '0',  # LOCAL: Kaggle keep-alive action; no server start-up here
        'ARC3_HTTP_RETRY_INITIAL_SECONDS': '3600',  # --memory edit (900 -> 3600)
        # LOCAL: a hosted free tier throttles; retry longer than the 3 x 5 s default (both arms)
        'ARC3_HTTP_RETRIES': '8', 'ARC3_HTTP_RETRY_BASE_SECONDS': '10', 'ARC3_HTTP_RETRY_MAX_SECONDS': '60',
        'MPLBACKEND': 'Agg', 'TAAF_RUN_AS_SUBMISSION': '0', 'TAAF_MINIMAL_DIAGNOSTICS': '0',
        'ONLY_RESET_LEVELS': 'true',
        'USE_TF': '0', 'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_TORCHVISION': '1',
        # LOCAL: hosted endpoint and model instead of the local SGLang server
        'LOCAL_ANALYZER_BASE_URL': args.base_url, 'OPENAI_BASE_URL': args.base_url,
        'LOCAL_ANALYZER_PROVIDER': 'vllm', 'OPENAI_PROVIDER': 'vllm',
        'LOCAL_ANALYZER_MODEL_ID': args.model, 'INFERENCE_ANALYZER_MODEL': args.model,
        'LOCAL_ANALYZER_APP_NAME': 'ARC3 Agent Harness',
        'LOCAL_ANALYZER_SEED': str(args.seed),  # LOCAL: fixed sampling seed (NIM accepts `seed`)
        'ARC3_REASONING_HISTORY_KEY': 'reasoning_content',
        'LOCAL_ANALYZER_ENABLE_THINKING': 'true',
        'LOCAL_ANALYZER_TOOL_STEPS': '0', 'LOCAL_ANALYZER_TOOL_TIMEOUT': '30',
        'LOCAL_ANALYZER_YIELD_SECONDS': '0', 'LOCAL_ANALYZER_YIELD_TOKENS': '2048',
        'LOCAL_ANALYZER_TEMPERATURE': '0.7', 'LOCAL_ANALYZER_TOP_P': '0.95', 'LOCAL_ANALYZER_TOP_K': '20',
        # context pruning: Franzen's values (Nemotron's window is larger than 128k, so they fit)
        'LOCAL_ANALYZER_CONTEXT_WINDOW': str((116 + 12) * 1024), 'LOCAL_ANALYZER_MAX_OUTPUT': str(12 * 1024),
        'ARC3_CONTEXT_DRAIN_TOKENS': str(58 * 1024), 'ARC3_HISTORY_ASSISTANT_TURNS': '150',
        'ARC3_HISTORY_TURN_DRAIN': '30', 'ARC3_HISTORY_DRAIN_COALESCE': '1',
        'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': '3072', 'ARC3_MIDDLE_TRUNCATION': '1', 'ARC3_ACTION_ECHO': '1',
        'ARC3_EXPLAIN_GAMEPLAY_CHANGED': '1', 'ARC3_NEW_CHANGED_PROMPTS': '1',
        'ARC3_LEVEL_TRANSFER_GUIDANCE': '1', 'ARC3_ACTION_INFO': '1', 'ARC3_DEDUPE_MULTICALL_LINE': '1',
        'ARC3_FRAME_DIFF_HINT': '1',
        'MULTIMODAL_CONTEXT': 'current_grid', 'MULTIMODAL_UPSCALE': '10', 'ARC3_DIFF_IMAGE': '1',
        'ARC3_GAMEOVER_DIFF_IMAGE': '1', 'ARC3_ANIMATION': '1',
        'ARC3_GUARDS_FROM_LEVEL': '2', 'ARC3_BATCH_NOOP_BLOCK': '1', 'ARC3_STALE_STATE_BLOCK': '1',
        'ARC3_MEMORY_SECTIONS': 'off',
        'ARC3_PERSISTENT_FUNCTIONS': '1', 'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game',
        'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',
        'EXPOSE_UNDO': 'on',
        # USE_PRIORITY_SCHEDULING block (the gate never binds: 10 slots > games in flight)
        'ARC3_MAX_ACTIVE_STREAMS': '10', 'ARC3_PRIORITY_REFRESH_QUEUE': '1', 'ARC3_PRIORITY_PACE': '0',
        'ARC3_PRIORITY_TAIL_FADE': '1', 'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.2',
        'ARC3_PRIORITY_TAIL_LOOKUP': 'remaining', 'ARC3_PRIORITY_SCORE_NORMALIZATION': '1',
    }
    if args.arm == "D":
        # push_duck.py puts --env in a cell before the block; none of these keys overlap it
        env.update({'EXPOSE_RESET': 'on', 'ARC3_HINT_BUDGET': '1', 'ARC3_HINT_LIFE': '1',
                    'ARC3_HINT_EFFECTS': '1', 'ARC3_SCALE_HASH': '1'})
    return env


def windows_sandbox_shims() -> None:
    """LOCAL (Windows only): the harness's Python-tool sandbox is POSIX-minded. A child Python
    started with an env lacking SYSTEMROOT dies at start-up on Windows, and os.killpg does not
    exist there (AttributeError, which its `except OSError` does not catch)."""
    if os.name != "nt":
        return
    import inference.agent.python_tool_sandbox as sb

    orig_env = sb._sandbox_env

    def _sandbox_env():
        env = orig_env()
        for k in ("SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP"):
            if os.environ.get(k):
                env[k] = os.environ[k]
        env["HOME"] = env["TMPDIR"] = os.environ.get("TEMP", env["HOME"])
        return env

    def _kill(process):
        try:
            process.kill()
        except OSError:
            pass

    # The bootstrap (~67 KB) is passed with `-c`, over Windows' 32767-character command-line
    # limit (FileNotFoundError, winerror 206 -> "Sandbox process could not start"). Run the same
    # text from a file instead; with -I the file's directory is not put on sys.path.
    import subprocess
    import tempfile
    import types

    boot = Path(tempfile.gettempdir()) / f"duck_sandbox_bootstrap_{os.getpid()}.py"
    boot.write_text(sb._SANDBOX_BOOTSTRAP, encoding="utf-8")

    def _popen(cmd, *a, **kw):
        if isinstance(cmd, list) and "-c" in cmd and cmd[cmd.index("-c") + 1] == sb._SANDBOX_BOOTSTRAP:
            i = cmd.index("-c")
            cmd = cmd[:i] + [str(boot)] + cmd[i + 2:]
        kw.pop("start_new_session", None)
        return subprocess.Popen(cmd, *a, **kw)

    sb.subprocess = types.SimpleNamespace(**{k: getattr(subprocess, k) for k in dir(subprocess)
                                             if not k.startswith("__")})
    sb.subprocess.Popen = _popen
    sb._sandbox_env = _sandbox_env
    sb._kill_process_group = _kill


class Tally:
    """Per-game counters gathered by wrapping the harness (read-only observation)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.hints = defaultdict(Counter)    # game stem -> marker -> turns with that line
        self.turns = Counter()               # game stem -> user prompts built
        self.auto_resets = Counter()         # game stem -> harness auto-resets after GAME_OVER
        self.examples = {}                   # game stem -> one prompt tail containing a hint

    def install(self) -> None:
        import inference.agent.tool_agent as ta
        import inference.framework.solver as solver

        tally = self
        orig_prompt = ta.ToolAgent._build_user_prompt

        def build_user_prompt(agent, *a, **kw):
            prompt = orig_prompt(agent, *a, **kw)
            try:
                stem = Path(str(getattr(agent, "_tally_state_path", "") or "?")).name
                with tally.lock:
                    tally.turns[stem] += 1
                    for key, marker in HINT_MARKERS.items():
                        if marker in prompt:
                            tally.hints[stem][key] += 1
                            tally.examples.setdefault(stem[:4], prompt[-1500:])
            except Exception as exc:  # observation must never change a turn
                print(f"TALLY error: {exc}", flush=True)
            return prompt

        orig_analyze = ta.ToolAgent.analyze

        def analyze(agent, state_path, *a, **kw):
            agent._tally_state_path = str(state_path)
            return orig_analyze(agent, state_path, *a, **kw)

        orig_auto = solver._HarnessGameSession._execute_auto_reset

        def auto_reset(session):
            with tally.lock:
                tally.auto_resets[session.game.game_run.game_id] += 1
            return orig_auto(session)

        ta.ToolAgent._build_user_prompt = build_user_prompt
        ta.ToolAgent.analyze = analyze
        solver._HarnessGameSession._execute_auto_reset = auto_reset


def summarise(bm, tally: Tally, wall_s: float) -> dict:
    games = []
    for run in bm.game_runs:
        ids = [rec.action.id.name for rec in run.history]
        tag = run.game_id[:4]  # state files are named after the game id; the 4-char tags are unique
        hint_counts = dict(sum((c for k, c in tally.hints.items() if k.startswith(tag)), Counter()))
        turns = sum(v for k, v in tally.turns.items() if k.startswith(tag))
        games.append({
            "game_id": run.game_id, "state": run.state, "levels_completed": run.levels_completed,
            "number_of_levels": run.number_of_levels, "final_score": run.final_score,
            "actions": len(run.history), "actions_per_level": list(run.actions_per_level),
            "base_actions_per_level": run.base_actions_per_level,
            "resets_total": ids.count("RESET"), "resets_auto": tally.auto_resets.get(run.game_id, 0),
            "resets_model": ids.count("RESET") - tally.auto_resets.get(run.game_id, 0),
            "undos": ids.count("ACTION7"),
            "generated_tokens": sum(r.generated_tokens or 0 for r in run.history) + (run.final_generated_tokens or 0),
            "uncached_input_tokens": sum(r.uncached_input_tokens or 0 for r in run.history)
            + (run.final_uncached_input_tokens or 0),
            "wallclock_s": run.final_wallclock_seconds, "turn_prompts": turns, "hint_turns": hint_counts,
            "solver_note": run.solver_note,
        })
    scores = [g["final_score"] or 0.0 for g in games]
    return {"games": games, "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "wall_s": wall_s, "hint_examples": tally.examples}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Duck harness locally against a hosted API.")
    ap.add_argument("--arm", choices=["A", "D"], required=True)
    ap.add_argument("--games", required=True, help="comma-separated game-id prefixes, run in this order")
    ap.add_argument("--minutes", type=float, default=25.0, help="per-game wall-clock cap")
    ap.add_argument("--concurrency", type=int, default=2, help="games in flight at once")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-dir", default=str(DEFAULT_ENV_DIR))
    ap.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--model", default="nvidia/nemotron-3.5-super-vl-120b-a12b")
    ap.add_argument("--key-env", default="NVIDIA_TOKEN")
    ap.add_argument("--env-file", default=str(ROOT.parent.parent.parent / ".env"
                                              if ".claude" in ROOT.parts else ROOT / ".env"))
    args = ap.parse_args()
    if not sys.flags.utf8_mode:
        # LOCAL (Windows): the harness writes its HTML/diagnostics with the locale encoding
        # (cp1252 here), which cannot encode the arrows in them. Re-run in UTF-8 mode.
        import subprocess
        sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Environment first: the harness reads most flags at import time.
    os.environ.update(harness_env(args))
    os.environ["LOCAL_ANALYZER_API_KEY"] = load_key(Path(args.env_file), args.key_env)
    os.environ.setdefault("RECORDINGS_DIR", str(out / "server_recording"))
    os.environ["PYTHONIOENCODING"] = "utf-8"

    import arc_agi
    import taaf.benchmark
    import taaf.game_api
    from inference.framework.solver import HarnessSolver

    windows_sandbox_shims()
    # --memory (memory_patch.py is inserted verbatim into the customisation cell on Kaggle)
    exec(compile(MEMORY.read_text(encoding="utf-8"), str(MEMORY), "exec"), {"__name__": "memory_patch"})
    if args.arm == "D":  # --hints: inserted after the memory patch, then th_install()
        ns = {"__name__": "turn_hints"}
        exec(compile(HINTS.read_text(encoding="utf-8"), str(HINTS), "exec"), ns)
        ns["th_install"]()
    tally = Tally()
    tally.install()

    prefixes = [g.strip() for g in args.games.split(",") if g.strip()]
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=args.env_dir)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=args.env_dir)
    available = [e.game_id for e in arcade.available_environments]
    game_ids = []
    for p in prefixes:
        hit = [g for g in available if g.startswith(p)]
        if len(hit) != 1:
            raise SystemExit(f"game prefix {p!r} matched {hit}")
        game_ids.append(hit[0])
    games = [taaf.game_api.GameAPI(env_name=g, arcade_spec=spec) for g in game_ids]

    # Franzen's customisation cell (16) with the --memory timeout edit; per-game cap from the CLI.
    solver = HarnessSolver(label="duck-harness", model="local", analyzer_timeout=3600.0,
                           max_actions_per_game=None, max_runtime_s_per_game=args.minutes * 60,
                           concurrency=args.concurrency, save_request_logs=False)
    bm = taaf.benchmark.Benchmark(label=f"duck-local-{args.arm}", games=games, solver=solver, n_passes=1,
                                  job_dir=out, game_weights=None)
    print(f"local_run arm={args.arm} games={game_ids} cap={args.minutes} min concurrency={args.concurrency} "
          f"model={args.model} seed={args.seed}", flush=True)
    t0 = time.time()
    try:
        asyncio.run(bm.run(soft_end_time=None, runtime_environment=None, minimal_diagnostics=False))
    except Exception as exc:  # teardown diagnostics; the game runs are already final
        print(f"local_run: bm.run raised {type(exc).__name__}: {exc}", flush=True)
    summary = summarise(bm, tally, time.time() - t0)
    summary.update({"arm": args.arm, "model": args.model, "minutes_cap": args.minutes, "seed": args.seed,
                    "concurrency": args.concurrency, "games_order": game_ids})
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for g in summary["games"]:
        print(f"RESULT arm={args.arm} {g['game_id']} levels={g['levels_completed']}/{g['number_of_levels']} "
              f"score={g['final_score']:.2f} actions={g['actions']} resets_model={g['resets_model']} "
              f"hints={g['hint_turns']} gen_tokens={g['generated_tokens']}", flush=True)
    print(f"MEAN arm={args.arm} {summary['mean_score']:.2f}", flush=True)


if __name__ == "__main__":
    main()
