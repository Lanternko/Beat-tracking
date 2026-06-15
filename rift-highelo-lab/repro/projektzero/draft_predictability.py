"""How predictable is pro champion selection, and does the predictor track the meta?

A behavioral 'policy': predict a role's pick from champion popularity in a trailing
window. Walk-forward (only past games). Measures:
  (1) top-1 accuracy + recall@5, overall and per role
  (2) META STALENESS: fresh window (last 4 wk) vs ~1-month-old window (wk -8..-5)
  (3) META-SHIFT detection: top-1 accuracy in the FIRST week of a new patch vs later
      -- the system has no explicit shift detector; its own error spike IS the detector.

Crude v1: role+trailing-popularity, no draft-state/counter-pick conditioning (that lifts
recall@5; this is the floor). Pools all leagues -> more stable popularity but blurs the
fact that regions hit a patch on different dates (noises the patch-week analysis).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter, defaultdict

yrs = [2023, 2024, 2025, 2026]
raw = pd.concat([pd.read_csv(f'/tmp/oe_data/{y}_oe.csv', low_memory=False) for y in yrs],
                ignore_index=True)
pl = raw[raw.position != 'team'][['gameid', 'league', 'date', 'patch', 'position', 'champion']].copy()
pl = pl.dropna(subset=['champion', 'position', 'date'])
pl['date'] = pd.to_datetime(pl['date'])
pl['patch'] = pl['patch'].astype(str)
wk0 = pl['date'].min()
pl['wk'] = ((pl['date'] - wk0).dt.days // 7).astype(int)
ROLES = ['top', 'jng', 'mid', 'bot', 'sup']
print(f"player-picks: {len(pl)} | weeks {pl.wk.min()}-{pl.wk.max()} | leagues {pl.league.nunique()}")

# weekly (role, champ) counts -> nested dict
wkcnt = defaultdict(lambda: defaultdict(Counter))
for (w, pos, champ), n in pl.groupby(['wk', 'position', 'champion']).size().items():
    wkcnt[w][pos][champ] += n

def window_pop(weeks):
    pop = {r: Counter() for r in ROLES}
    for w in weeks:
        for r in ROLES:
            pop[r] += wkcnt[w].get(r, Counter())
    return pop

# patch start per (league, patch) for the meta-shift analysis
pstart = pl.groupby(['league', 'patch'])['date'].min().to_dict()
pl['days_in_patch'] = pl.apply(lambda r: (r['date'] - pstart[(r['league'], r['patch'])]).days, axis=1)

# walk-forward evaluation
rows = []  # (fresh_t1, fresh_t5, stale_t1, stale_t5, position, days_in_patch)
all_w = sorted(wkcnt.keys())
for W in all_w:
    if W - 8 < all_w[0]:
        continue
    fresh = window_pop(range(W - 4, W))       # last 4 weeks
    stale = window_pop(range(W - 8, W - 4))   # ~1 month earlier
    ftop = {r: [c for c, _ in fresh[r].most_common(5)] for r in ROLES}
    stop = {r: [c for c, _ in stale[r].most_common(5)] for r in ROLES}
    gw = pl[pl.wk == W]
    for pos in ROLES:
        sub = gw[gw.position == pos]
        if sub.empty or not ftop[pos]:
            continue
        f1, f5 = ftop[pos][0], set(ftop[pos])
        s1 = stop[pos][0] if stop[pos] else None
        s5 = set(stop[pos])
        for champ, dip in zip(sub.champion.values, sub.days_in_patch.values):
            rows.append((champ == f1, champ in f5, champ == s1, champ in s5, pos, dip))

ev = pd.DataFrame(rows, columns=['f1', 'f5', 's1', 's5', 'pos', 'dip'])
print(f"evaluated picks: {len(ev)}\n")

print("=== (1) PREDICTABILITY: fresh trailing-4-week popularity, walk-forward ===")
print(f"  top-1 accuracy : {ev.f1.mean()*100:5.1f}%   (actual pick == most popular for the role)")
print(f"  recall@5       : {ev.f5.mean()*100:5.1f}%   (actual pick in the role's top-5)")
print("  by role:")
for r in ROLES:
    s = ev[ev.pos == r]
    print(f"    {r:4s}  top-1 {s.f1.mean()*100:5.1f}%   recall@5 {s.f5.mean()*100:5.1f}%")

print("\n=== (2) META STALENESS: fresh (last 4wk) vs ~1-month-old window ===")
print(f"  fresh   top-1 {ev.f1.mean()*100:5.1f}%   recall@5 {ev.f5.mean()*100:5.1f}%")
print(f"  1-mo old top-1 {ev.s1.mean()*100:5.1f}%   recall@5 {ev.s5.mean()*100:5.1f}%")
print(f"  -> staleness cost: top-1 {(ev.f1.mean()-ev.s1.mean())*100:+.1f}pp, "
      f"recall@5 {(ev.f5.mean()-ev.s5.mean())*100:+.1f}pp in ~1 month")

print("\n=== (3) META-SHIFT shock: fresh top-1 by how new the patch is ===")
for lab, mask in [('patch week 1 (days 0-6)', ev.dip <= 6),
                  ('patch week 2 (7-13)', (ev.dip >= 7) & (ev.dip <= 13)),
                  ('patch settled (14+)', ev.dip >= 14)]:
    s = ev[mask]
    print(f"  {lab:24s} n={len(s):6d}  top-1 {s.f1.mean()*100:5.1f}%  recall@5 {s.f5.mean()*100:5.1f}%")
print("  -> a dip in week 1 = the model trained on the OLD patch failing = the meta-shift signal.")
