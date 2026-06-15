"""Targeted test of the strongest form of the deviation idea: the LAST PICK.

Standard draft order ends ... B4 | R4 B5 | R5. So RED's pick5 (R5) is the absolute
last pick = the free counter-pick slot (max information, prepared-strat slot). BLUE's
pick1 (B1) is the first pick (least info, most contested). The 'hidden strat / prepared
counter' (藏招) should concentrate in a STRONG team's surprising LAST pick.

Test: among red teams (who hold last pick), does a high-surprisal pick5 predict winning
-- and does it for FAVORITES (innovation) but not UNDERDOGS (forced)? Contrast vs blue's
first pick. Averaging over 5 champs (draft_deviation_probe.py) washes this out; here we
isolate the one slot that carries the counter-pick signal.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

yrs = [2023, 2024, 2025, 2026]
raw = pd.concat([pd.read_csv(f'/tmp/oe_data/{y}_oe.csv', low_memory=False) for y in yrs],
                ignore_index=True)

# champion pick-rate per patch (meta-ness reference), from player rows
pl = raw[raw.position != 'team'].dropna(subset=['champion', 'patch']).copy()
pl['patch'] = pl['patch'].astype(str)
gpp = pl.groupby('patch')['gameid'].nunique()
cnt = pl.groupby(['patch', 'champion'])['gameid'].nunique().reset_index(name='n')
cnt['rate'] = cnt.apply(lambda r: r.n / gpp[r.patch], axis=1)
rate = {(r.patch, r.champion): r.rate for r in cnt.itertuples()}
EPS = 1e-3
def surp(patch, champ):
    return -np.log10(rate.get((str(patch), champ), 0) + EPS)

# team rows carry pick1..pick5 in DRAFT ORDER + firstPick
tm = raw[raw.position == 'team'][['gameid', 'side', 'patch', 'firstPick',
                                  'pick1', 'pick5']].dropna(subset=['pick1', 'pick5']).copy()
tm['first_pick_surp'] = [surp(p, c) for p, c in zip(tm.patch, tm.pick1)]
tm['last_pick_surp'] = [surp(p, c) for p, c in zip(tm.patch, tm.pick5)]

# join walk-forward strength + result
en = pd.read_csv('/tmp/oe_data/enriched_teams_2023_2026.csv', low_memory=False)
en['result'] = en['result'].astype(int)
m = en.merge(tm, on=['gameid', 'side'], how='inner')
p = np.clip(m.player_elo_win_perc.values, 1e-6, 1 - 1e-6)
m['strength_logit'] = np.log(p / (1 - p))     # this team's pre-game strength edge
print(f"team-games joined: {len(m)} | red (last pick) {len(m[m.side=='Red'])} | blue (first pick) {len(m[m.side=='Blue'])}\n")

def split(df, col, label):
    fav, dog = df[df.strength_logit > 0], df[df.strength_logit < 0]
    print(f"=== {label}: does a SURPRISING {col} predict this team winning? ===")
    for nm, sub in [('FAVORITE (strong)', fav), ('UNDERDOG (weak)', dog)]:
        hi = sub[sub[col] > sub[col].quantile(0.80)]   # most surprising pick (off-meta)
        lo = sub[sub[col] < sub[col].quantile(0.50)]   # ordinary/meta pick
        d = (hi.result.mean() - lo.result.mean()) * 100
        print(f"  {nm:18s} n={len(sub):5d} | surprising {col} win {hi.result.mean()*100:5.1f}% (n={len(hi)})"
              f"  vs ordinary {lo.result.mean()*100:5.1f}% (n={len(lo)})  -> {d:+.1f}pp")
    print()

# RED holds the last pick (the prepared-counter slot) -- the strongest test
split(m[m.side == 'Red'], 'last_pick_surp', 'RED LAST PICK (R5 = free counter / 藏招 slot)')
# BLUE first pick -- contrast (least information, should NOT reward surprise)
split(m[m.side == 'Blue'], 'first_pick_surp', 'BLUE FIRST PICK (B1 = most contested, no counter info)')
# also red's last pick vs the meta as a single clean number
red = m[m.side == 'Red']
print("interpretation: if 藏招/innovation is real, a strong red team's SURPRISING last pick")
print("should win MORE (positive pp) while a weak team's should win LESS (negative).")
