"""Counter δ 矩陣：英雄 vs 英雄的對線優劣，Bayesian shrink（prior=該英雄自身對線均值 μ_A）。

δ(A,B) = mean(gd10 | A 對 B) − μ_A，再 shrink by n/(n+K) 處理稀疏。
  δ > 0：A 在此對位「比平常更壓制」B（A 占優）
  δ < 0：A 在此對位「比平常吃虧」（B counter A）
這正是 Q2 的「平均對線領先 − 該對位實際」分離出的 matchup 專屬效果（= ARAM lift 同構）。

用法：python3 build_counter_matrix.py [ROLE=TOP] [patches=16.11,16.12] [min_n=5]
"""
import sqlite3
import sys
from collections import defaultdict

import numpy as np

import config

ROLE = (sys.argv[1].upper() if len(sys.argv) > 1 else "TOP")
PATCHES = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["16.11", "16.12"])
MIN_N = int(sys.argv[3]) if len(sys.argv) > 3 else 5
K = 6  # shrinkage：對位樣本 < K 時大幅拉回 μ（可調）

conn = sqlite3.connect(config.DB_PATH)
ph = ",".join("?" * len(PATCHES))
rows = conn.execute(
    f"""SELECT l.champion, l.opp_champion, l.gd10
        FROM laning l JOIN matches m ON l.match_id = m.match_id
        WHERE l.team_position = ? AND m.patch IN ({ph})""",
    (ROLE, *PATCHES),
).fetchall()

by_champ, by_matchup = defaultdict(list), defaultdict(list)
for a, b, gd in rows:
    by_champ[a].append(gd)
    by_matchup[(a, b)].append(gd)

mu = {a: float(np.mean(v)) for a, v in by_champ.items()}

recs = []
for (a, b), v in by_matchup.items():
    n = len(v)
    if n < MIN_N:
        continue
    mean_ab = float(np.mean(v))
    raw = mean_ab - mu[a]
    delta = (n / (n + K)) * raw  # shrink toward 0
    recs.append((a, b, n, mu[a], mean_ab, delta))

n5 = sum(1 for _, v in by_matchup.items() if len(v) >= 5)
n10 = sum(1 for _, v in by_matchup.items() if len(v) >= 10)
maxn = max((len(v) for v in by_matchup.values()), default=0)
print(f"ROLE={ROLE}  patches={PATCHES}  對位觀測={len(rows)}  英雄={len(mu)}")
print(f"對位覆蓋：n>=5 的格子 {n5} 個、n>=10 {n10} 個、最大 n={maxn}（其餘太稀疏，已濾掉）\n")


def show(title, rs):
    print(title)
    print(f"  {'A (我方)':<13}{'vs B':<13}{'n':>4}{'μ_A':>7}{'此對位':>8}{'δ(shrink)':>11}")
    for a, b, n, m, act, d in rs:
        print(f"  {a:<13}{b:<13}{n:>4}{m:>+7.0f}{act:>+8.0f}{d:>+11.0f}")
    print()


recs.sort(key=lambda r: -r[5])
show("=== A 比平常更壓制 B（A 的有利對位，δ 最高）===", recs[:12])
show("=== A 比平常吃虧（B counter A，δ 最低）===", recs[-12:][::-1])
print(f"（shrink K={K}；δ 單位=金錢@10。樣本越大、δ 越可信。要更乾淨需更多 crawl。）")
