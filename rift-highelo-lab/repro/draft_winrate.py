"""LoLDraftAI-style rebuild: predict win from draft alone on KR high-elo soloQ.

Champion-identity logistic regression (blue +1 / red -1, the DraftGap/ARAM encoding),
the honest small-data analogue of LoLDraftAI's 5v5 NN. Reports acc/logloss/brier/ECE
on a time-split held-out tail + bootstrap 90% CI, vs the matchmaking-flattened base rate.

Run from repo root:  python3 repro/draft_winrate.py
"""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

rng = np.random.default_rng(0)
con = sqlite3.connect('data/lol.db')
parts = pd.read_sql('select match_id, champion, team_id from participants', con)
matches = pd.read_sql('select match_id, patch, win_side, game_creation from matches', con)

# --- build champion-identity matrix (blue +1, red -1), label = blue win ---
champs = sorted(parts.champion.unique())
cidx = {c: i for i, c in enumerate(champs)}
M = len(champs)
matches = matches.sort_values('game_creation').reset_index(drop=True)
midpos = {m: i for i, m in enumerate(matches.match_id)}
X = np.zeros((len(matches), M), dtype=np.float32)
parts = parts[parts.match_id.isin(midpos)]
rows = parts.match_id.map(midpos).values
cols = parts.champion.map(cidx).values
signs = np.where(parts.team_id.values == 100, 1.0, -1.0).astype(np.float32)
np.add.at(X, (rows, cols), signs)
y = (matches.win_side.values == 100).astype(int)
print(f"matches {len(y)} | champs {M} | blue winrate {y.mean():.4f}")


def metrics(yt, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (np.mean((p >= 0.5).astype(int) == yt),
            log_loss(yt, p, labels=[0, 1]), brier_score_loss(yt, p))


def ece(yt, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.sum():
            e += abs(p[m].mean() - yt[m].mean()) * m.sum() / len(p)
    return e


# --- time split: last 20% by game_creation = held-out future ---
n = len(y); cut = int(n * 0.8)
Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
print(f"train {len(ytr)} | test {len(yte)} (held-out future)")

# base rate (matchmaking-flattened): constant = train blue winrate
base_p = np.full(len(yte), ytr.mean())
b = metrics(yte, base_p)
print(f"\n{'model':22s} {'acc':>8s} {'logloss':>9s} {'brier':>8s} {'ECE':>7s}")
print(f"{'base rate (constant)':22s} {b[0]*100:7.2f}% {b[1]:9.4f} {b[2]:8.4f} {'-':>7s}")

# champion-identity LR, light C sweep on a val slice of train
vcut = int(cut * 0.85)
best = None
for C in [0.05, 0.1, 0.25, 0.5, 1.0]:
    clf = LogisticRegression(C=C, max_iter=2000).fit(X[:vcut], y[:vcut])
    vll = metrics(y[vcut:cut], clf.predict_proba(X[vcut:cut])[:, 1])[1]
    if best is None or vll < best[1]:
        best = (C, vll)
C = best[0]
clf = LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr)
pte = clf.predict_proba(Xte)[:, 1]
a = metrics(yte, pte)
print(f"{'champ-identity LR':22s} {a[0]*100:7.2f}% {a[1]:9.4f} {a[2]:8.4f} {ece(yte, pte):7.4f}   (C={C})")

# random within-data split (optimistic, no patch drift) for the ceiling
perm = rng.permutation(n); rtr, rte = perm[:cut], perm[cut:]
clf_r = LogisticRegression(C=C, max_iter=2000).fit(X[rtr], y[rtr])
pr = clf_r.predict_proba(X[rte])[:, 1]
ar = metrics(y[rte], pr)
print(f"{'  (random split)':22s} {ar[0]*100:7.2f}% {ar[1]:9.4f} {ar[2]:8.4f} {ece(y[rte], pr):7.4f}")

# --- bootstrap 90% CI on time-split test (resample matches) ---
pc = np.clip(pte, 1e-9, 1 - 1e-9)
r_correct = ((pc >= 0.5).astype(int) == yte).astype(float)
r_ll = -(yte * np.log(pc) + (1 - yte) * np.log(1 - pc))
r_br = (pc - yte) ** 2
m = len(yte); B = 3000
samp = rng.integers(0, m, size=(B, m))
accs = r_correct[samp].mean(axis=1)
lls = r_ll[samp].mean(axis=1)
brs = r_br[samp].mean(axis=1)
print(f"\n=== BOOTSTRAP 90% CI (champ-identity LR, {B} resamples, {m} test games) ===")
for name, arr in [('accuracy', accs), ('logloss', lls), ('brier', brs)]:
    lo, hi = np.percentile(arr, 5), np.percentile(arr, 95)
    print(f"{name:9s} median={np.median(arr):.4f}  90%CI=[{lo:.4f}, {hi:.4f}]  width={hi-lo:.4f}")
alo, ahi = np.percentile(accs, 5), np.percentile(accs, 95)
print(f"-> draft->win accuracy {np.median(accs)*100:.2f}%  (90% CI {alo*100:.1f}-{ahi*100:.1f}%, "
      f"+/-{(ahi-alo)/2*100:.1f}pp); coin-flip 50% is {'INSIDE' if alo<=0.5<=ahi else 'OUTSIDE'} the CI")

# --- per-champion win rate + Beta 90% CI (illustrates Q4.1 sample-size limit) ---
print("\n=== per-champion presence win rate, Beta(0.5,0.5)+counts 90% CI (top played) ===")
from scipy.stats import beta as Beta
pres = parts.merge(matches[['match_id', 'win_side']], on='match_id')
pres['won'] = ((pres.team_id == 100) & (pres.win_side == 100)) | ((pres.team_id == 200) & (pres.win_side == 200))
agg = pres.groupby('champion').agg(games=('won', 'size'), wins=('won', 'sum'))
agg = agg[agg.games >= 30].sort_values('games', ascending=False)
for c, r in agg.head(8).iterrows():
    lo = Beta.ppf(0.05, r.wins + 0.5, r.games - r.wins + 0.5)
    hi = Beta.ppf(0.95, r.wins + 0.5, r.games - r.wins + 0.5)
    print(f"  {c:14s} n={int(r.games):4d} wr={r.wins/r.games*100:5.1f}%  90%CI=[{lo*100:4.1f}, {hi*100:4.1f}]  +/-{(hi-lo)/2*100:.1f}pp")
print(f"\n  (median games/champ = {int(agg.games.median())}; "
      f"typical per-champ CI width ~ {0.82/np.sqrt(agg.games.median())*200:.1f}pp -> tiers overlap)")
