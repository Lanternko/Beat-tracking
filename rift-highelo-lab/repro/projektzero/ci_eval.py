# Honest eval of ProjektZero ensemble: time-split (train -> held-out tail),
# EGPM-LR refit on train only, ensemble weights from train, bootstrap 90% CI.
# Elo/TrueSkill/SideEMA win-percs are pre-game (_before) ratings = already walk-forward.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

rng = np.random.default_rng(42)
df = pd.read_csv('/tmp/oe_data/enriched_teams_2023_2025.csv', low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df['result'] = df['result'].astype(int)
df = df.sort_values('date').reset_index(drop=True)

# Time split by game date: last 20% of games = test
gids = df[['gameid', 'date']].drop_duplicates().sort_values('date')
cut = gids.iloc[int(len(gids) * 0.80)]['date']
train = df[df['date'] < cut].copy()
test = df[df['date'] >= cut].copy()
print(f"train games {train.gameid.nunique()} (<{cut.date()}) | test games {test.gameid.nunique()} (>= cut)")

# EGPM-LR refit on TRAIN only, predict on both
feats = ['egpm_dominance_ema_before', 'opp_egpm_dominance_ema_before']
clf = LogisticRegression().fit(train[feats], train['result'])
for d in (train, test):
    d['egpm_lr_wp'] = clf.predict_proba(d[feats])[:, 1]

models = {
    'TeamElo': 'team_elo_win_perc',
    'PlayerElo': 'player_elo_win_perc',
    'TrueSkill': 'trueskill_win_perc',
    'EGPM': 'egpm_lr_wp',
    'SideEMA': 'side_ema_win_perc',
}

def metrics(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    acc = np.mean((p >= 0.5).astype(int) == y)
    return acc, log_loss(y, p, labels=[0, 1]), brier_score_loss(y, p)

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(p)
    return e

# Ensemble weights from TRAIN accuracies
train_acc = {k: metrics(train['result'].values, train[v].values)[0] for k, v in models.items()}
wsum = sum(train_acc.values())
weights = {k: train_acc[k] / wsum for k in models}
print("ensemble weights (from train acc):", {k: round(w, 3) for k, w in weights.items()})

def ensemble_wp(d):
    return sum(d[v].values * weights[k] for k, v in models.items())

# Held-out test metrics (point estimates)
print("\n=== HELD-OUT TEST (honest: EGPM+weights from train only) ===")
print(f"{'model':12s} {'acc':>8s} {'logloss':>9s} {'brier':>8s}")
y_te = test['result'].values
for k, v in models.items():
    a, ll, br = metrics(y_te, test[v].values)
    print(f"{k:12s} {a*100:7.2f}% {ll:9.4f} {br:8.4f}")
ens_te = ensemble_wp(test)
a, ll, br = metrics(y_te, ens_te)
print(f"{'Ensemble':12s} {a*100:7.2f}% {ll:9.4f} {br:8.4f}   ECE={ece(y_te, ens_te):.4f}")

# Bootstrap 90% CI (vectorized). Each game = 2 complementary team-rows, so overall
# metric = mean of per-game metric means; resample game-level means as a B x ng matrix.
test = test.reset_index(drop=True)
p_full = np.clip(ensemble_wp(test), 1e-9, 1 - 1e-9)
y_full = test['result'].values
row_correct = ((p_full >= 0.5).astype(int) == y_full).astype(float)
row_ll = -(y_full * np.log(p_full) + (1 - y_full) * np.log(1 - p_full))
row_br = (p_full - y_full) ** 2
idx_by_game = list(test.groupby('gameid').indices.values())
g_acc = np.array([row_correct[ix].mean() for ix in idx_by_game])
g_ll = np.array([row_ll[ix].mean() for ix in idx_by_game])
g_br = np.array([row_br[ix].mean() for ix in idx_by_game])
ng = len(g_acc)
B = 2000
samp = rng.integers(0, ng, size=(B, ng))
accs = g_acc[samp].mean(axis=1)
lls = g_ll[samp].mean(axis=1)
brs = g_br[samp].mean(axis=1)

def ci(a):
    return np.percentile(a, 5), np.percentile(a, 95)

print(f"\n=== BOOTSTRAP 90% CI (ensemble, {B} resamples, {ng} test games) ===")
for name, arr in [('accuracy', accs), ('logloss', lls), ('brier', brs)]:
    lo, hi = ci(arr)
    print(f"{name:9s} median={np.median(arr):.4f}  90%CI=[{lo:.4f}, {hi:.4f}]  width={hi-lo:.4f}")
acc_lo, acc_hi = ci(accs)
print(f"\n-> ensemble accuracy {np.median(accs)*100:.2f}%  (90% CI {acc_lo*100:.1f}-{acc_hi*100:.1f}%, "
      f"+/-{(acc_hi-acc_lo)/2*100:.1f}pp on {ng} games)")
