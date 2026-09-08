import sys, time; sys.path.insert(0, r"C:/Users/Patrick/Documents/Patrick/ARC-AGI-3")
import numpy as np
from tests.conftest import load
from arc3.perception.geometry import induce_geometry
from arc3.perception.segment import segment
from arc3.perception.track import Tracker

NAMES = ('colour_4','colour_8','foreground_4','foreground_8','colour_mass','bbox')
print(f"{'game':7}{'frames':>7}{'act':>5}{'f/act':>7}  {'seg6 ms':>8} | " +
      "".join(f"{n[:9]:>10}" for n in NAMES) + f"{'ALL6':>9}")
tot = {}
for g in ('ls20','sp80','bp35','vc33','sb26'):
    ep = load(g); grids = ep.grids(); geo = induce_geometry(grids)
    n_act = ep.actions_used
    sample = grids[:25]
    t0=time.perf_counter()
    segs=[segment(f, geo) for f in sample]
    seg_ms=(time.perf_counter()-t0)/len(sample)*1000
    row={}
    for nm in NAMES:
        tr=Tracker(nm); t0=time.perf_counter()
        for ss in segs: tr.update(ss[nm])
        row[nm]=(time.perf_counter()-t0)/len(sample)*1000
    allms = seg_ms + sum(row.values())
    tot[g]=(len(grids), n_act, allms)
    print(f"{g:7}{len(grids):>7}{n_act:>5}{len(grids)/max(1,n_act):>7.1f}  {seg_ms:>8.0f} | " +
          "".join(f"{row[n]:>10.0f}" for n in NAMES) + f"{allms:>9.0f}")

print()
fpa = np.mean([v[0]/max(1,v[1]) for v in tot.values()])
ms  = np.mean([v[2] for v in tot.values()])
worst = max(v[2] for v in tot.values())
print(f"mean frames per action across fixtures: {fpa:.1f}")
print(f"mean cost of the FULL six-carving pipeline: {ms:.0f} ms/frame  (worst game {worst:.0f})")
for acts in (100, 300):
    frames = 25*acts*fpa
    print(f"\n25 games x {acts} actions = {frames:,.0f} frames")
    print(f"   all six carvings : {frames*ms/1000/3600:6.1f} h   (worst-case mix {frames*worst/1000/3600:6.1f} h)")
    print(f"   one carving only : {frames*ms/6/1000/3600:6.1f} h  (rough)")
