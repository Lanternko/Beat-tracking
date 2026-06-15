"""Public cross-check of the draft->win ceiling on HF gptilt Challenger ~10k
(global, 3 regions) and asia-only (KR-comparable). Same champ-identity LR as
draft_winrate.py. Confirms the matchmaking ceiling on independent data.

Needs: huggingface_hub, pyarrow.  Run from repo root: python3 repro/draft_winrate_hf.py
"""
import warnings; warnings.filterwarnings("ignore")
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

rng = np.random.default_rng(0)
REPO = 'gptilt/lol-basic-matches-challenger-10k'


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


def load(regions):
    P, Mt = [], []
    for r in regions:
        pp = hf_hub_download(REPO, f'participants/region_{r}-00000.parquet', repo_type='dataset')
        mm = hf_hub_download(REPO, f'matches/region_{r}-00000.parquet', repo_type='dataset')
        P.append(pd.read_parquet(pp, columns=['matchId', 'championName', 'teamId', 'win']))
        Mt.append(pd.read_parquet(mm, columns=['matchId', 'gameStartTimestamp', 'gameVersion']))
    parts = pd.concat(P, ignore_index=True)
    matches = pd.concat(Mt, ignore_index=True)
    g = parts.groupby('matchId').agg(n=('championName', 'size'), blue=('teamId', lambda s: (s == 100).sum()))
    good = set(g[(g.n == 10) & (g.blue == 5)].index)
    return parts[parts.matchId.isin(good)].copy(), matches[matches.matchId.isin(good)].copy()


def run(parts, matches, tag):
    champs = sorted(parts.championName.unique())
    cidx = {c: i for i, c in enumerate(champs)}
    matches = matches.sort_values('gameStartTimestamp').reset_index(drop=True)
    midpos = {m: i for i, m in enumerate(matches.matchId)}
    X = np.zeros((len(matches), len(champs)), dtype=np.float32)
    rows = parts.matchId.map(midpos).values
    cols = parts.championName.map(cidx).values
    signs = np.where(parts.teamId.values == 100, 1.0, -1.0).astype(np.float32)
    np.add.at(X, (rows, cols), signs)
    bluewin = parts[parts.teamId == 100].groupby('matchId').win.first()
    y = bluewin.reindex(matches.matchId).astype(int).values
    patches = matches.gameVersion.str.extract(r'^(\d+\.\d+)')[0].value_counts().head(3).to_dict()
    print(f"\n### {tag}: {len(y)} games | champs {len(champs)} | blue winrate {y.mean():.4f} | patches {patches}")

    n = len(y); cut = int(n * 0.8)
    base = metrics(y[cut:], np.full(n - cut, y[:cut].mean()))
    print(f"  base rate     acc={base[0]*100:5.2f}% logloss={base[1]:.4f} brier={base[2]:.4f}")
    clf = LogisticRegression(C=0.05, max_iter=2000).fit(X[:cut], y[:cut])
    p = clf.predict_proba(X[cut:])[:, 1]; yte = y[cut:]
    a = metrics(yte, p)
    print(f"  champ-LR      acc={a[0]*100:5.2f}% logloss={a[1]:.4f} brier={a[2]:.4f} ECE={ece(yte, p):.4f}")
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    rc = ((pc >= 0.5).astype(int) == yte).astype(float)
    m = len(yte); B = 3000
    samp = rng.integers(0, m, size=(B, m))
    accs = rc[samp].mean(axis=1)
    lo, hi = np.percentile(accs, 5), np.percentile(accs, 95)
    print(f"  -> draft->win {np.median(accs)*100:.2f}%  90%CI=[{lo*100:.1f}, {hi*100:.1f}]  +/-{(hi-lo)/2*100:.1f}pp"
          f"  (coinflip {'INSIDE' if lo <= 0.5 <= hi else 'OUTSIDE'})")


pa, ma = load(['asia'])
run(pa, ma, "HF Challenger ASIA (KR-comparable)")
pg, mg = load(['americas', 'asia', 'europe'])
run(pg, mg, "HF Challenger GLOBAL (3 regions)")
