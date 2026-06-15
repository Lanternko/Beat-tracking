"""單一分路的 gd@10 -> 勝率曲線（team100 視角）。
用法：python3 lane_winrate.py [LANE=JUNGLE]   LANE: TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY

兩種口徑：
  經驗曲線 = 直接看「該路領先 X 時整體勝率」（含其他路連動，會偏高）
  模型 isolated = 只有該路領先、其餘持平時的勝率（純該路效果）
"""
import sqlite3
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

import config
from predict_winrate import LANES, load

LANE = (sys.argv[1].upper() if len(sys.argv) > 1 else "JUNGLE")
conn = sqlite3.connect(config.DB_PATH)

rows = conn.execute(
    """SELECT l.gd10, l.win FROM laning l JOIN participants p
       ON l.match_id = p.match_id AND l.puuid = p.puuid
       WHERE p.team_id = 100 AND l.team_position = ?""",
    (LANE,),
).fetchall()
gd = np.array([r[0] for r in rows], float)
win = np.array([r[1] for r in rows], int)

print(f"{LANE}: n={len(gd)}  整體勝率={win.mean() * 100:.1f}%")
print(f"\n經驗曲線（{LANE} 單路 gd@10 -> 勝率，含其他路連動）：")
edges = [-1500, -1000, -500, -250, -100, 100, 250, 500, 1000, 1500]
labels = ["< -1.5k", "-1.5k~-1k", "-1k~-500", "-500~-250", "-250~-100",
          "-100~+100", "+100~250", "+250~500", "+500~1k", "+1k~1.5k", "> +1.5k"]
idx = np.digitize(gd, edges)
for b in range(len(labels)):
    m = idx == b
    if m.sum() >= 10:
        print(f"  {labels[b]:<12}{m.sum():>4}{win[m].mean() * 100:>7.0f}%")

m = (gd >= 350) & (gd <= 650)
print(f"\n>>> 「{LANE} 領先約 500」[350,650]：n={m.sum()}  經驗勝率={win[m].mean() * 100:.0f}%")

# 模型 isolated：只有此路 +500、其餘 0
X10, _, y = load(conn)
lr = LogisticRegression(max_iter=1000).fit(X10, y)
x = np.zeros((1, 5))
x[0, LANES.index(LANE)] = 500
print(f">>> 模型 isolated（只有 {LANE} +500、其餘持平）：勝率={lr.predict_proba(x)[0, 1] * 100:.0f}%")
