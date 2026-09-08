import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from collections import defaultdict, Counter
from tests.conftest import load

print("Target: predict the exact resting frame after an action.\n")
print(f"{'game':7}{'trans':>6}{'noop%':>7}{'B0 static':>11}{'B1 per-action':>15}{'B1|changed':>12}")
gt=gn=0; b0=0; b1=0; b1c=0; nc=0
for g in ('ls20','sp80','bp35','vc33','sb26'):
    ep=load(g); trs=ep.transitions()
    last={}                      # action id -> last observed (changed_mask, after) delta
    n=noop=s0=s1=s1c=chg=0
    for tr in trs:
        n+=1
        is_noop = bool(np.array_equal(tr.before, tr.after))
        if is_noop: noop+=1
        else: chg+=1
        # B0: predict nothing changes
        if is_noop: s0+=1
        # B1: replay the exact pixel delta last seen for this action id
        key=(tr.action.id,)
        pred=tr.before
        if key in last:
            mask, vals = last[key]
            pred = tr.before.copy(); pred[mask]=vals
        ok = bool(np.array_equal(pred, tr.after))
        if ok: s1+=1
        if not is_noop and ok: s1c+=1
        d = tr.before != tr.after
        if d.any(): last[key]=(d, tr.after[d])
    print(f"{g:7}{n:>6}{100*noop/n:>7.0f}{100*s0/n:>10.0f}%{100*s1/n:>14.0f}%"
          f"{(f'{100*s1c/chg:.0f}%' if chg else '-'):>12}")
    gt+=n; gn+=noop; b0+=s0; b1+=s1; b1c+=s1c; nc+=chg
print(f"\n{'ALL':7}{gt:>6}{100*gn/gt:>7.0f}{100*b0/gt:>10.0f}%{100*b1/gt:>14.0f}%{100*b1c/nc:>11.0f}%")
print(f"\nno-op transitions: {gn}/{gt}.  Transitions where something changed: {nc}.")
