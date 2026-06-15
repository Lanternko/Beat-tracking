"""Q2 第3步：對線 -> 勝率預測。五路 gd@10/@14 -> logistic regression。

回答兩件事：
  (1) 對線狀態能把勝負預測到多準（CV accuracy / AUC）
  (2) 哪一路的早期領先最決定比賽（標準化係數 + 單變量 AUC）
"""
import sqlite3

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
LANE_ZH = {"TOP": "上路", "JUNGLE": "打野", "MIDDLE": "中路",
           "BOTTOM": "下路ADC", "UTILITY": "輔助"}


def load(conn):
    """team100 視角：每場五路的 gd@10/@14 + 勝負。"""
    rows = conn.execute(
        """SELECT l.match_id, l.team_position, l.gd10, l.gd14, l.win
           FROM laning l JOIN participants p
             ON l.match_id = p.match_id AND l.puuid = p.puuid
           WHERE p.team_id = 100"""
    ).fetchall()
    games = {}
    for mid, pos, gd10, gd14, win in rows:
        games.setdefault(mid, {})[pos] = (gd10, gd14, win)
    X10, X14, y = [], [], []
    for d in games.values():
        if not all(p in d for p in LANES):
            continue
        X10.append([d[p][0] for p in LANES])
        X14.append([d[p][1] for p in LANES])
        y.append(d["TOP"][2])  # win 對 team100 任一位置都相同
    return np.array(X10, float), np.array(X14, float), np.array(y, int)


def fit_report(X, y, label):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    acc = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()
    auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean()
    pipe.fit(X, y)
    coef = pipe.named_steps["logisticregression"].coef_[0]  # 標準化後可比
    print(f"\n=== {label} ===  n={len(y)}  CV accuracy={acc:.3f}  CV AUC={auc:.3f}")
    print("  分路重要度（標準化係數=控制其他路後的邊際效果；單變量AUC=單獨預測力）：")
    for i in np.argsort(-np.abs(coef)):
        uni = roc_auc_score(y, X[:, i])
        print(f"    {LANE_ZH[LANES[i]]:<8} coef={coef[i]:+.3f}   單變量AUC={uni:.3f}")


def gold_curve(X10, y):
    total = X10.sum(axis=1)  # = team100 總金錢差 @10
    edges = [-2000, -1000, -500, -200, 200, 500, 1000, 2000]
    labels = ["< -2k", "-2k~-1k", "-1k~-500", "-500~-200", "-200~+200",
              "+200~500", "+500~1k", "+1k~2k", "> +2k"]
    idx = np.digitize(total, edges)
    print("\n=== 10分鐘總金錢差 -> 勝率（team100 視角）===")
    print(f"  {'金錢差@10':<12}{'n':>5}{'winrate':>9}")
    for b in range(len(labels)):
        m = idx == b
        if m.sum() > 0:
            print(f"  {labels[b]:<12}{m.sum():>5}{y[m].mean() * 100:>8.0f}%")


def main():
    conn = sqlite3.connect(config.DB_PATH)
    X10, X14, y = load(conn)
    print(f"games used: {len(y)}  team100 winrate={y.mean() * 100:.1f}% (應接近 50% = 無偏)")
    fit_report(X10, y, "用 gd@10 預測勝負")
    fit_report(X14, y, "用 gd@14 預測勝負")
    gold_curve(X10, y)


if __name__ == "__main__":
    main()
