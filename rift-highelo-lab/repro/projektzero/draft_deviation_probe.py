"""First-cut test of the 'draft deviation as signal' idea (Kojek).

Premise: in pro, everyone drafts meta -> champion CONTENT is low-variance -> low
signal. But DEVIATION from the meta (off-meta picks) is where residual info lives.
Hypothesis: deviation is NOT monotonic with winning; it INTERACTS with strength —
a strong team's off-meta pick = prepared innovation (win), a weak team's = forced
error (lose). So we test the `offmeta x strength` interaction, not a main effect.

This is the CRUDE proxy: 'optimal' = behavioral meta = champion pick-rate in that
(patch, position). Off-meta score = mean surprisal of a side's 5 champs. NOT yet a
sequential next-pick policy (that's v2). meta-rate uses full-patch data = descriptive,
not walk-forward; the strength baseline IS walk-forward (player_elo_win_perc _before).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# --- 1. champions per (gameid, side) + patch, from raw OE ---
yrs = [2023, 2024, 2025, 2026]
raw = pd.concat([pd.read_csv(f'/tmp/oe_data/{y}_oe.csv', low_memory=False) for y in yrs],
                ignore_index=True)
pl = raw[raw.position != 'team'][['gameid', 'side', 'position', 'champion', 'patch']].dropna(
    subset=['champion', 'position'])
pl['patch'] = pl['patch'].astype(str)

# --- 2. behavioral meta-ness: champion pick-rate within (patch, position) ---
games_per_patch = pl.groupby('patch')['gameid'].nunique()
pick_cnt = pl.groupby(['patch', 'position', 'champion'])['gameid'].nunique().reset_index(name='n')
pick_cnt['meta_rate'] = pick_cnt.apply(lambda r: r['n'] / games_per_patch[r['patch']], axis=1)
rate = {(r.patch, r.position, r.champion): r.meta_rate for r in pick_cnt.itertuples()}
EPS = 1e-3
pl['surprisal'] = pl.apply(lambda r: -np.log10(rate.get((r.patch, r.position, r.champion), 0) + EPS),
                           axis=1)
# off-meta score per side per game = mean surprisal over its 5 champs (higher = more off-meta)
offmeta = pl.groupby(['gameid', 'side'])['surprisal'].mean().reset_index(name='offmeta')

# --- 3. join to walk-forward strength + result (enriched ProjektZero) ---
en = pd.read_csv('/tmp/oe_data/enriched_teams_2023_2026.csv', low_memory=False)
en['date'] = pd.to_datetime(en['date']); en['result'] = en['result'].astype(int)
en = en.merge(offmeta, on=['gameid', 'side'], how='inner')

blue = en[en.side == 'Blue'][['gameid', 'date', 'player_elo_win_perc', 'offmeta', 'result']].rename(
    columns={'player_elo_win_perc': 'p_blue', 'offmeta': 'off_blue', 'result': 'y'})
red = en[en.side == 'Red'][['gameid', 'offmeta']].rename(columns={'offmeta': 'off_red'})
g = blue.merge(red, on='gameid', how='inner').dropna().sort_values('date').reset_index(drop=True)
print(f"games with strength+draft joined: {len(g)}")

# --- 4. features ---
p = np.clip(g.p_blue.values, 1e-6, 1 - 1e-6)
g['strength_logit'] = np.log(p / (1 - p))      # blue strength edge (walk-forward)
g['offmeta_edge'] = g.off_blue - g.off_red       # >0: blue drafted more off-meta than red
# standardize for stable interaction
for c in ['strength_logit', 'offmeta_edge']:
    g[c + '_z'] = (g[c] - g[c].mean()) / g[c].std()
g['interaction'] = g['strength_logit_z'] * g['offmeta_edge_z']

# --- 5. time-split held-out log-loss: does deviation ADD over strength? ---
cut = g.date.quantile(0.8)
tr, te = g[g.date < cut], g[g.date >= cut]
y_tr, y_te = tr.y.values, te.y.values
print(f"train {len(tr)} (<{cut.date()}) | test {len(te)}\n")

specs = {
    'M0 strength only': ['strength_logit_z'],
    'M1 + offmeta_edge': ['strength_logit_z', 'offmeta_edge_z'],
    'M2 + interaction': ['strength_logit_z', 'offmeta_edge_z', 'interaction'],
}
print(f"{'model':22s} {'test acc':>9s} {'test logloss':>13s}   coefs")
for name, feats in specs.items():
    clf = LogisticRegression(C=1e6, solver='lbfgs').fit(tr[feats], y_tr)
    pte = clf.predict_proba(te[feats])[:, 1]
    acc = ((pte >= .5).astype(int) == y_te).mean()
    ll = log_loss(y_te, np.clip(pte, 1e-9, 1 - 1e-9), labels=[0, 1])
    cf = ", ".join(f"{f.replace('_z','').replace('strength_logit','str').replace('offmeta_edge','off')}={c:+.3f}"
                   for f, c in zip(feats, clf.coef_[0]))
    print(f"{name:22s} {acc*100:8.2f}% {ll:13.4f}   {cf}")

# --- 6. the innovation-vs-mistake split: off-meta among favorites vs underdogs ---
print("\n=== does off-meta drafting help the FAVORITE or the UNDERDOG? (full data) ===")
g['blue_off_meta_more'] = g.offmeta_edge > g.offmeta_edge.median()
for lab, sub in [('blue is FAVORITE (str>0)', g[g.strength_logit > 0]),
                 ('blue is UNDERDOG (str<0)', g[g.strength_logit < 0])]:
    hi = sub[sub.offmeta_edge > sub.offmeta_edge.quantile(0.75)]   # blue drafted most off-meta
    lo = sub[sub.offmeta_edge < sub.offmeta_edge.quantile(0.25)]   # blue drafted most meta
    print(f"  {lab}: n={len(sub)}")
    print(f"      blue most OFF-meta (top quartile): blue win {hi.y.mean()*100:.1f}%  (n={len(hi)})")
    print(f"      blue most ON-meta  (bot quartile): blue win {lo.y.mean()*100:.1f}%  (n={len(lo)})")
