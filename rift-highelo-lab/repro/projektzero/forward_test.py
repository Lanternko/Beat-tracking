"""TRUE forward-test: train ProjektZero ratings only on games BEFORE an event,
predict that unseen event, report out-of-sample accuracy + bootstrap CI.

Elo/TrueSkill/Side-EMA win-percs are pre-game (_before) = walk-forward. EGPM-LR +
ensemble weights refit on pre-event data. Contrasts REGIONAL (big strength gaps,
predictable) vs INTERNATIONAL (top-teams-only, compressed -> coinflip).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

rng = np.random.default_rng(7)
df = pd.read_csv('/tmp/oe_data/enriched_teams_2023_2026.csv', low_memory=False)
df['date'] = pd.to_datetime(df['date']); df['result'] = df['result'].astype(int)
df['year'] = df['date'].dt.year

models = {'TeamElo': 'team_elo_win_perc', 'PlayerElo': 'player_elo_win_perc',
          'TrueSkill': 'trueskill_win_perc', 'EGPM': 'egpm_lr_wp', 'SideEMA': 'side_ema_win_perc'}
feats = ['egpm_dominance_ema_before', 'opp_egpm_dominance_ema_before']


def metrics(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (np.mean((p >= 0.5).astype(int) == y), log_loss(y, p, labels=[0, 1]), brier_score_loss(y, p))


def forward_test(league, year, label, split=None, kind=""):
    m = (df.league == league) & (df.year == year)
    if split is not None:
        m &= (df.split == split)
    test = df[m].copy()
    if len(test) == 0:
        print(f"\n### {label}: NO GAMES"); return
    cut = test.date.min()
    train = df[df.date < cut].copy()
    clf = LogisticRegression().fit(train[feats], train['result'])
    train['egpm_lr_wp'] = clf.predict_proba(train[feats])[:, 1]
    test['egpm_lr_wp'] = clf.predict_proba(test[feats])[:, 1]
    w = {k: metrics(train['result'].values, train[v].values)[0] for k, v in models.items()}
    s = sum(w.values()); w = {k: w[k] / s for k in w}
    ens = sum(test[v].values * w[k] for k, v in models.items())
    y = test['result'].values
    a, ll, br = metrics(y, ens)
    pe = metrics(y, test['player_elo_win_perc'].values)[0]
    ng = test.gameid.nunique()
    p = np.clip(ens, 1e-9, 1 - 1e-9)
    rc = ((p >= 0.5).astype(int) == y).astype(float)
    test = test.reset_index(drop=True)
    idx = list(test.groupby('gameid').indices.values())
    g_acc = np.array([rc[ix].mean() for ix in idx])
    B = 5000; samp = rng.integers(0, len(g_acc), size=(B, len(g_acc)))
    accs = g_acc[samp].mean(axis=1)
    lo, hi = np.percentile(accs, 5), np.percentile(accs, 95)
    print(f"\n### {label}  [{kind}]  {ng} games (train {train.gameid.nunique()}, cutoff {cut.date()})")
    print(f"  ENSEMBLE acc={a*100:.1f}%  PlayerElo-only={pe*100:.1f}%  logloss={ll:.4f}  brier={br:.4f}")
    print(f"  -> 90% CI [{lo*100:.1f}, {hi*100:.1f}]  +/-{(hi-lo)/2*100:.1f}pp  "
          f"(coinflip {'INSIDE' if lo <= 0.5 <= hi else 'OUTSIDE'})")


print("=== FORWARD-TEST: REGIONAL (predictable) vs INTERNATIONAL (compressed) ===")
print("\n--- 2026 (the events you asked about; train on everything before each) ---")
forward_test('LCK', 2026, "LCK 2026 Cup", split='Cup', kind="REGIONAL")
forward_test('LCK', 2026, "LCK 2026 Rounds 1-2 = road to MSI (just ended 6/14)", split='Rounds 1-2', kind="REGIONAL")
forward_test('FST', 2026, "First Stand 2026", kind="INTERNATIONAL")
forward_test('EWC', 2026, "EWC 2026", kind="mixed")
print("\n--- 2025 internationals (for contrast) ---")
forward_test('MSI', 2025, "MSI 2025", kind="INTERNATIONAL")
forward_test('WLDs', 2025, "Worlds 2025", kind="INTERNATIONAL")
