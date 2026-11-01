# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.
# SOLVED-LEVEL MEMORY (lalalia9), for the dfranzen Milestone 2 harness.
# Adapted from rellik13/sirikilohit's Milestone 2 solution (M85, CC-BY 4.0), the only text addition in
# their write-up that helped (+2.8 on the leaderboard).
#
# Why: dfranzen keeps one rolling conversation per game and drains it from 118k to ~59k tokens when full.
# A level's turns usually fill most of that window, so after the first drain on a new level the agent
# has lost how it solved the previous ones and re-derives mechanics it already knew.
#
# What: when a level is cleared, the harness records facts it already has (level number, actions used,
# the winning attempt's last 30 actions, the tail of the reasoning in the turn that cleared it) and asks
# the model, in the next two turn messages, for one "Rule for level N:" line, captured from its reply or
# reasoning. The records are pinned as one block at the END of the system prompt.
#
# Cache: the block is only (re)written when the conversation is being drained anyway (the server must
# re-read the conversation then, so changing the system prompt costs nothing extra). The shared start
# of the system prompt is untouched, so it stays cached across games. Before the first drain on a level,
# the previous level is still in the conversation itself. Any error leaves the turn unchanged.
import re as _re_slm

import inference.agent.tool_agent as _ta_slm
from inference.agent.runtime_state import load_runtime_state as _slm_load_state

_SLM_ACTIONS = 30
_SLM_REASONING_CHARS = 900
_SLM_RULE_CHARS = 600
_SLM_FULL_LEVELS = 3
_SLM_ASK_TURNS = 2
_SLM_HEADER = "=== SOLVED LEVELS (facts from this game + your own conclusions; kept for the whole game) ==="
_SLM_FOOTER = "=== END SOLVED LEVELS ==="


def _slm_text(msg):
    parts = []
    for key in ("content", "reasoning_content", "reasoning"):
        value = msg.get(key) if isinstance(msg, dict) else None
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts += [p.get("text", "") for p in value if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return "\n".join(p for p in parts if p)


def _slm_level_actions(entries, level):
    """Actions of the winning attempt on `level`. history[i].frame is the frame AFTER history[i].action,
    so an action belongs to the level of the frame before it; a RESET starts a new attempt."""
    out, prev = [], None
    for entry in entries or []:
        action = str(getattr(entry, "action", "") or "").strip()
        if action and prev == level:
            if action.upper().startswith("RESET"):
                out = []
            else:
                out.append(action)
        lv = getattr(getattr(entry, "frame", None), "level", None)
        if isinstance(lv, int):
            prev = lv
    return out


def _slm_find_rule(text, level):
    m = _re_slm.search(r"Rule for level\s*%d\s*:[ \t]*" % int(level), text or "", _re_slm.I)
    if not m:
        return None
    lines = text[m.end():].split("\n")
    first = lines[0].strip()
    if len(first) >= 40:
        rule = first
    else:
        out = [first] if first else []
        for line in lines[1:6]:
            if not line.strip():
                if out:
                    break
                continue
            out.append(line.strip())
        rule = " ".join(out)
    rule = " ".join(rule.split())[:_SLM_RULE_CHARS]
    return rule if len(rule) >= 20 else None


def _slm_block(levels):
    if not levels:
        return ""
    lines = [_SLM_HEADER]
    full_from = max(0, len(levels) - _SLM_FULL_LEVELS)
    for i, rec in enumerate(levels):
        lines.append(f"Level {rec['level']} - cleared after {rec['n_actions']} actions.")
        lines.append(f"  Your rule: {rec['rule'] if rec.get('rule') else 'not stated'}")
        if i >= full_from:
            tail = rec["actions"]
            skipped = rec["n_actions"] - len(tail)
            lines.append("  Last actions before it cleared" + (f" ({skipped} earlier omitted)" if skipped > 0 else "")
                         + ": " + (", ".join(tail) if tail else "none recorded"))
            if rec.get("reasoning"):
                lines.append("  Your reasoning in the turn that cleared it: " + rec["reasoning"])
    lines.append("Reuse these where the new level has the same controls; verify with one probe before relying on them.")
    lines.append(_SLM_FOOTER)
    return "\n".join(lines)


def _slm_refresh_system_prompt(self):
    """Write the current block into self._system_prompt (end of prompt, replaced in place)."""
    block = _slm_block(getattr(self, "_slm_levels", []))
    if block == getattr(self, "_slm_block_text", ""):
        return False
    prompt = self._system_prompt
    old = getattr(self, "_slm_block_text", "")
    if old and old in prompt:
        prompt = prompt.replace(old, block)
    else:
        prompt = prompt.rstrip("\n") + "\n\n" + block
    self._system_prompt = prompt
    self._slm_block_text = block
    print(f"SOLVED_LEVEL_MEMORY pinned: {len(self._slm_levels)} level(s), {len(block)} chars", flush=True)
    return True


def _slm_before_turn(self, state_path):
    levels = self.__dict__.setdefault("_slm_levels", [])
    # capture rules the model has written since it was asked
    for rec in levels:
        if not rec.get("rule") and rec.get("asked"):
            for msg in reversed(self._history_messages[-8:]):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    rule = _slm_find_rule(_slm_text(msg), rec["level"])
                    if rule:
                        rec["rule"] = rule
                        break
    current = int((self._last_step_summary or {}).get("level") or 1)
    recorded = {rec["level"] for rec in levels}
    missing = [lv for lv in range(1, current) if lv not in recorded]
    if not missing:
        return
    _, entries = _slm_load_state(state_path)
    reasoning = ""
    for msg in reversed(self._history_messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            text = str(msg.get("reasoning_content") or msg.get("reasoning") or msg.get("content") or "")
            reasoning = " ".join(text.split())[-_SLM_REASONING_CHARS:]
            break
    for lv in missing:
        actions = _slm_level_actions(entries, lv)
        levels.append({"level": lv, "n_actions": len(actions), "actions": actions[-_SLM_ACTIONS:],
                       "reasoning": reasoning if lv == missing[-1] else "", "rule": None, "asks_left": _SLM_ASK_TURNS})
    levels.sort(key=lambda rec: rec["level"])


_slm_original_analyze = _ta_slm.ToolAgent.analyze
_slm_original_user_prompt = _ta_slm.ToolAgent._build_user_prompt
_slm_original_evicted = _ta_slm.ToolAgent._note_history_evicted
_slm_original_trim = _ta_slm.ToolAgent._trim_messages_for_context


def _slm_analyze(self, state_path, *args, **kwargs):
    try:
        _slm_before_turn(self, state_path)
    except Exception as exc:
        print(f"SOLVED_LEVEL_MEMORY error (turn unchanged): {exc}", flush=True)
    return _slm_original_analyze(self, state_path, *args, **kwargs)


def _slm_user_prompt(self, *args, **kwargs):
    prompt = _slm_original_user_prompt(self, *args, **kwargs)
    try:
        for rec in getattr(self, "_slm_levels", []):
            if not rec.get("rule") and rec.get("asks_left", 0) > 0:
                rec["asks_left"] -= 1
                rec["asked"] = True
                prompt += (f"\n\nBefore your tool call, write a line starting `Rule for level {rec['level']}:` "
                           "that states the goal, what each control did and the plan that won (at most 5 lines). "
                           "It is kept for the rest of the game.")
                break
    except Exception:
        pass
    return prompt


def _slm_note_evicted(self):
    _slm_original_evicted(self)
    try:
        _slm_refresh_system_prompt(self)
    except Exception as exc:
        print(f"SOLVED_LEVEL_MEMORY error (prompt unchanged): {exc}", flush=True)


def _slm_trim(self, messages, *args, **kwargs):
    result = _slm_original_trim(self, messages, *args, **kwargs)
    # a drop already forces a re-read of this conversation, so the refreshed system prompt rides along
    if (len(result) < len(messages) and result and result[0].get("role") == "system"
            and result[0].get("content") != self._system_prompt):
        result = [{**result[0], "content": self._system_prompt}, *result[1:]]
    return result


_ta_slm.ToolAgent.analyze = _slm_analyze
_ta_slm.ToolAgent._build_user_prompt = _slm_user_prompt
_ta_slm.ToolAgent._note_history_evicted = _slm_note_evicted
_ta_slm.ToolAgent._trim_messages_for_context = _slm_trim
print("SOLVED_LEVEL_MEMORY_PATCH on", flush=True)

