"""draft-vs-execution：把勝率預測拆成「選角(draft)」vs「臨場(早期金錢)」。

5-fold CV AUC 比較：
  draft only   — 雙方 (英雄,分路) 簽到向量（team100 +1 / team200 -1）→ 純選角能預測多少
  lane@10/@14  — 五路 raw gd
  combined     — draft + lane
分解：
  combined@10 − draft  = 知道早期戰況比只看選角多預測多少（execution 的價值）
  combined@10 − lane@10 = 英雄身份比只看金錢多預測多少（打團/後期/邊帶 = 你 Q2 擔心漏掉的 β）
"""
import sqlite3

import numpy as np
from scipy.sparse import csr_matrix, hstack, lil_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import config

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def cv(X, y, C=1.0):
    return cross_val_score(LogisticRegression(max_iter=3000, C=C), X, y,
                           cv=5, scoring="roc_auc").mean()


def main():
    conn = sqlite3.connect(config.DB_PATH)
    parts = conn.execute(
        "SELECT match_id, team_id, team_position, champion, win FROM participants"
    ).fetchall()
    lane = conn.execute(
        """SELECT l.match_id, p.team_id, l.team_position, l.gd10, l.gd14
           FROM laning l JOIN participants p
           ON l.match_id = p.match_id AND l.puuid = p.puuid"""
    ).fetchall()

    champ_of, win_of = {}, {}
    for mid, team, pos, champ, win in parts:
        champ_of.setdefault(mid, []).append((team, pos, champ))
        if team == 100:
            win_of[mid] = win
    ln = {}
    for mid, team, pos, gd10, gd14 in lane:
        ln.setdefault(mid, {})[(team, pos)] = (gd10, gd14)

    feat, games = {}, []
    for mid, clist in champ_of.items():
        if mid not in win_of or len(clist) != 10:
            continue
        L = ln.get(mid, {})
        if not all((100, p) in L for p in LANES):
            continue
        dd = {}
        for team, pos, champ in clist:
            j = feat.setdefault((champ, pos), len(feat))
            dd[j] = dd.get(j, 0) + (1 if team == 100 else -1)
        games.append((dd,
                      [L[(100, p)][0] for p in LANES],
                      [L[(100, p)][1] for p in LANES],
                      win_of[mid]))

    n, nf = len(games), len(feat)
    D = lil_matrix((n, nf))
    X10 = np.zeros((n, 5)); X14 = np.zeros((n, 5)); y = np.zeros(n, int)
    for i, (dd, x10, x14, w) in enumerate(games):
        for j, v in dd.items():
            D[i, j] = v
        X10[i] = x10; X14[i] = x14; y[i] = w
    D = csr_matrix(D)
    X10s = StandardScaler().fit_transform(X10)
    X14s = StandardScaler().fit_transform(X14)

    print(f"games={n}, (英雄,分路) features={nf}, team100 winrate={y.mean() * 100:.1f}%\n")
    a_draft = cv(D, y, C=0.5)
    a_l10 = cv(X10s, y)
    a_l14 = cv(X14s, y)
    a_c10 = cv(hstack([csr_matrix(X10s), D]).tocsr(), y, C=0.5)
    a_c14 = cv(hstack([csr_matrix(X14s), D]).tocsr(), y, C=0.5)

    print(f"  draft only            AUC={a_draft:.3f}   (純選角，賽前就知道)")
    print(f"  lane gd@10 only       AUC={a_l10:.3f}")
    print(f"  lane gd@14 only       AUC={a_l14:.3f}")
    print(f"  combined (draft+@10)  AUC={a_c10:.3f}")
    print(f"  combined (draft+@14)  AUC={a_c14:.3f}\n")
    print(f"  execution 價值  = combined@10 − draft  = {a_c10 - a_draft:+.3f}")
    print(f"  英雄身份(β) 價值 = combined@10 − lane@10 = {a_c10 - a_l10:+.3f}")


if __name__ == "__main__":
    main()
