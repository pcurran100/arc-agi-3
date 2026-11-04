# TURN HINTS (ARC-AGI-3 project, MIT-0; D-048). Appended to the Duck harness's per-turn
# message, computed by code from the recorded frames only (no game internals):
#   ARC3_HINT_BUDGET  - colours in the frame's border bands that change by the same amount on
#                       every action (a move/time bar), with the actions left at that rate.
#                       The model planned a 43-step route on a ~21-action bar (ls20 L2, 2026-11-01).
#   ARC3_HINT_LIFE    - the board went back to its level-start state after an ordinary action:
#                       a life was probably lost. v4's model read this as progress (D-044).
#   ARC3_HINT_EFFECTS - per action, how often it changed the board this level (v5's effect table),
#                       which survives the harness's history trimming.
# Each is on when its variable is "1". Any error leaves the turn message unchanged.
import os as _os_th

BAND = 3          # border band width in cells: the 64x64 frame's UI bars sit in rows/cols 0-2, 61-63
MIN_STEPS = 3     # a bar must change by the same amount on this many consecutive actions


def _th_on(name):
    return _os_th.environ.get(name, "0").strip().lower() in ("1", "on", "true", "yes")


def _th_band_counts(grid):
    """colour -> number of cells in the border bands."""
    h = len(grid)
    w = max((len(r) for r in grid), default=0)
    out = {}
    for y, row in enumerate(grid):
        edge_row = y < BAND or y >= h - BAND
        for x, v in enumerate(row):
            if edge_row or x < BAND or x >= w - BAND:
                out[v] = out.get(v, 0) + 1
    return out


def _th_inner_diff(a, b):
    """Cells that differ outside the border bands (a, b same shape)."""
    h = len(a)
    n = 0
    for y in range(BAND, h - BAND):
        ra, rb = a[y], b[y]
        w = len(ra)
        for x in range(BAND, w - BAND):
            if ra[x] != rb[x]:
                n += 1
    return n


def _th_is_reset(action):
    return str(action or "").strip().upper().startswith("RESET")


def _th_action_kind(action):
    s = str(action or "").strip().upper()
    for sep in ("(", " ", ":"):
        if sep in s:
            s = s.split(sep, 1)[0]
    return s or "?"


def _th_attempt(entries):
    """(level_start_grid, transitions) for the current attempt of the current level.
    transitions: [(action, before_grid, after_grid)] in order, RESETs excluded."""
    if not entries:
        return None, []
    level = entries[-1].frame.level
    start = None
    trans = []
    for i in range(len(entries) - 1, 0, -1):
        cur, prev = entries[i], entries[i - 1]
        if cur.frame.level != level:
            break
        if prev.frame.level != level or _th_is_reset(cur.action):
            start = cur.frame.grid
            break
        trans.append((cur.action, prev.frame.grid, cur.frame.grid))
    trans.reverse()
    if start is None:
        start = trans[0][1] if trans else entries[-1].frame.grid
    return start, trans


def th_budget_line(trans):
    # Actions that changed nothing (e.g. a move into a wall, which on ls20 costs no budget)
    # are left out, so "per action" means per action that did something.
    steps = [t for t in trans if len(t[1]) == len(t[2]) and t[1] != t[2]]
    if len(steps) < MIN_STEPS:
        return None
    recent = steps[-8:]
    counts = [_th_band_counts(recent[0][1])] + [_th_band_counts(t[2]) for t in recent]
    parts = []
    for colour in set().union(*counts):
        series = [c.get(colour, 0) for c in counts]
        deltas = [b - a for a, b in zip(series, series[1:])]
        run = 1
        while run < len(deltas) and deltas[-run - 1] == deltas[-1]:
            run += 1
        acts = {_th_action_kind(t[0]) for t in recent[-run:]}
        # The change must not depend on the action: seen under two actions, or a long run.
        if deltas[-1] == 0 or run < MIN_STEPS or (len(acts) < 2 and run < 5):
            continue
        d, now = deltas[-1], series[-1]
        if d < 0:
            parts.append(f"colour {colour}: {d} cells per action, {now} left -> about {now // -d} "
                         "actions remain at this rate")
        else:
            parts.append(f"colour {colour}: +{d} cells per action")
    if not parts:
        return None
    return ("BUDGET BAR (measured by code over your last actions; the rate can differ between levels): "
            + "; ".join(sorted(parts)) + ". Check that a plan fits before running it.")


def _th_refilled(series, after):
    """True if a border-band colour that had been changing steadily in one direction (at least
    MIN_STEPS changes, as a draining budget bar does) jumps back by at least half of that run.
    A lost life refills the bar; board pieces that merely reach into the band oscillate."""
    for colour in set(after).union(*series):
        s = [c.get(colour, 0) for c in series]
        d = [b - a for a, b in zip(s, s[1:]) if b != a]
        if len(d) < MIN_STEPS:
            continue
        up = d[-1] > 0
        run, total = 0, 0
        for x in reversed(d):
            if (x > 0) != up:
                break
            run, total = run + 1, total + x
        jump = after.get(colour, 0) - s[-1]
        if run >= MIN_STEPS and jump and (jump > 0) != up and abs(jump) * 2 >= abs(total):
            return True
    return False


def th_life_line(start, trans, new=1):
    """A lost life among the last `new` transitions (the actions since the previous turn
    message): the board outside the bands is back at the level start AND a steadily changing
    band colour (a budget bar) jumped back. Fixed 2026-11-02 after a local run: checking only
    the last transition missed every life lost inside a batch of actions (ls20), and the old
    test "the bands moved toward their start counts" fired on vc33 whenever two clicks toggled
    the board back, because vc33's board pieces reach into the border band."""
    if len(trans) < 2:
        return None
    n = len(trans)
    bands = [_th_band_counts(trans[0][1])] + [_th_band_counts(t[2]) for t in trans]
    for i in range(n - 1, max(1, n - max(1, new)) - 1, -1):
        action, before, after = trans[i]
        if len(after) != len(start) or len(before) != len(start):
            continue
        if (_th_inner_diff(after, start) == 0 and _th_inner_diff(before, start) > 0
                and _th_refilled(bands[:i + 1], bands[i + 1])):
            ago = n - 1 - i
            when = ("after your last action" if ago == 0 else
                    f"{ago} action{'s' if ago > 1 else ''} before your last one (after {_th_action_kind(action)})")
            return (f"BOARD BACK AT LEVEL START: {when} the board (outside the border bars) returned "
                    "exactly to how this level began and a bar in the border jumped back, and you did not "
                    "RESET. Most likely a life was lost or a budget ran out; progress on this attempt is gone. "
                    "Do not read it as progress.")
    return None


def th_effects_line(entries):
    if not entries:
        return None
    level = entries[-1].frame.level
    stats = {}
    for prev, cur in zip(entries, entries[1:]):
        if cur.frame.level != level or prev.frame.level != level or _th_is_reset(cur.action):
            continue
        if len(prev.frame.grid) != len(cur.frame.grid):
            continue
        k = _th_action_kind(cur.action)
        moved, total = stats.get(k, (0, 0))
        stats[k] = (moved + (_th_inner_diff(prev.frame.grid, cur.frame.grid) > 0), total + 1)
    if not stats:
        return None
    items = [f"{k} {m}/{t}" for k, (m, t) in sorted(stats.items())]
    return ("ACTION EFFECTS this level (times the board changed outside the border bars / times used): "
            + ", ".join(items) + ".")


def th_hints(entries, new=1):
    """`new`: how many of the latest entries arrived since the previous turn message."""
    lines = []
    start, trans = _th_attempt(entries)
    if _th_on("ARC3_HINT_LIFE"):
        lines.append(th_life_line(start, trans, new))
    if _th_on("ARC3_HINT_BUDGET"):
        lines.append(th_budget_line(trans))
    if _th_on("ARC3_HINT_EFFECTS"):
        lines.append(th_effects_line(entries))
    return [x for x in lines if x]


def th_install():
    """Wrap ToolAgent so each turn message ends with the hints. Chains with other wrappers."""
    import inference.agent.tool_agent as _ta
    from inference.agent.runtime_state import load_runtime_state as _load

    orig_analyze = _ta.ToolAgent.analyze
    orig_prompt = _ta.ToolAgent._build_user_prompt

    def analyze(self, state_path, *args, **kwargs):
        self._th_state_path = state_path
        return orig_analyze(self, state_path, *args, **kwargs)

    def build_user_prompt(self, *args, **kwargs):
        prompt = orig_prompt(self, *args, **kwargs)
        try:
            path = getattr(self, "_th_state_path", None)
            if path is not None:
                _, entries = _load(path)
                n = len(entries)
                seen = getattr(self, "_th_seen", None)
                if seen is not None and seen[0] == n:
                    lines = seen[1]          # same step (a resumed turn): same hints as before
                else:
                    new = n - seen[0] if seen is not None and seen[0] < n else 1
                    lines = th_hints(entries, new)
                    self._th_seen = (n, lines)
                if lines:
                    prompt += "\n\n" + "\n".join(lines)
        except Exception as exc:
            print(f"TURN_HINTS error (turn unchanged): {exc}", flush=True)
        return prompt

    _ta.ToolAgent.analyze = analyze
    _ta.ToolAgent._build_user_prompt = build_user_prompt
    print("TURN_HINTS on:", {k: _th_on(k) for k in ("ARC3_HINT_BUDGET", "ARC3_HINT_LIFE",
                                                     "ARC3_HINT_EFFECTS")}, flush=True)
