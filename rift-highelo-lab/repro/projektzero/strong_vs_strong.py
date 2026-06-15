"""Does the 69% survive when strong plays strong? Resolution analysis.
Uses player_elo_win_perc = roster/team strength only, NO champions (ProjektZero
uses zero champion info). All probs are walk-forward pre-game = out-of-sample.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

df = pd.read_csv('/tmp/oe_data/enriched_teams_2023_2026.csv', low_memory=False)
df['date'] = pd.to_datetime(df['date']); df['result'] = df['result'].astype(int)
lck = df[df.league == 'LCK'].copy()
g = lck[lck.side == 'Blue'].copy()            # one row per game; p = P(blue win)
p = g.player_elo_win_perc.values
y = g.result.values
fav_p = np.maximum(p, 1 - p)                   # model confidence in its pick
fav_win = ((p >= 0.5) == (y == 1)).astype(float)
print(f"LCK 2023-2026: {len(g)} games (one row/game)\n")

print("=== Q1: accuracy by how confident the model is (PlayerElo, champion-free) ===")
edges = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
labs = ['50-55% (toss-up)', '55-60%', '60-65%', '65-70%', '70%+ (blowout)']
for i in range(len(edges) - 1):
    m = (fav_p >= edges[i]) & (fav_p < edges[i + 1])
    if m.sum():
        print(f"  model says {labs[i]:18s} n={m.sum():4d}  favorite actually won {fav_win[m].mean()*100:5.1f}%")

# top LCK teams data-driven: top 6 by median rating among teams with >=40 games
cnt = lck.groupby('teamname').size()
med = lck.groupby('teamname')['player_elo_before'].median()
top = med[cnt >= 40].sort_values(ascending=False).head(6)
TOP = set(top.index)
print(f"\n=== Q1: STRONG vs STRONG (both teams in top-6 LCK) ===\n  top-6: {sorted(TOP)}")
has_opp = 'opponentname' in g.columns
if has_opp:
    ss = g[g.teamname.isin(TOP) & g.opponentname.isin(TOP)]
    rest = g[~(g.teamname.isin(TOP) & g.opponentname.isin(TOP))]
    for name, d in [('strong-vs-strong', ss), ('all other LCK', rest)]:
        pp = d.player_elo_win_perc.values; yy = d.result.values
        fp = np.maximum(pp, 1 - pp); fw = ((pp >= 0.5) == (yy == 1))
        print(f"  {name:18s} n={len(d):4d}  avg model confidence={fp.mean()*100:4.1f}%  "
              f"accuracy={fw.mean()*100:5.1f}%")

# playoffs across years (where top teams meet)
po = g[g.split.astype(str).str.contains('Playoff|Final|Knockout|Bracket', case=False, na=False)]
if len(po):
    pp = po.player_elo_win_perc.values; yy = po.result.values
    fw = ((pp >= 0.5) == (yy == 1))
    print(f"\n  LCK PLAYOFFS (all yrs) n={len(po)}  accuracy={fw.mean()*100:.1f}%  "
          f"avg conf={np.maximum(pp,1-pp).mean()*100:.1f}%")

# Q3: a real Bo5 - per-map prediction (T1 vs Gen.G, most recent series in data)
print("\n=== Q3: per-MAP prediction in a strong-vs-strong series (T1 vs Gen.G) ===")
pair = lck[((lck.teamname == 'T1') & (lck.opponentname == 'Gen.G')) |
           ((lck.teamname == 'Gen.G') & (lck.opponentname == 'T1'))].copy()
pair = pair[pair.teamname == 'T1'].sort_values('date')  # T1's row each map
if len(pair):
    last_day = pair.date.dt.date.iloc[-1]
    series = pair[pair.date.dt.date == last_day]
    print(f"  most recent T1-Gen.G on {last_day} ({len(series)} maps):")
    for _, r in series.iterrows():
        pt1 = r.player_elo_win_perc  # P(T1 win) since T1 is the row
        print(f"    map: T1 on {r.side:4s} side | model P(T1 win)={pt1*100:4.1f}% | "
              f"actual: {'T1 WON' if r.result == 1 else 'Gen.G won'}")
    sw = (series.player_elo_win_perc.max() - series.player_elo_win_perc.min()) * 100
    print(f"  -> per-map probs move with walk-forward Elo: win a map -> next-map P(T1) up, lose -> down")
    print(f"     (~{sw:.0f}pp swing across this series), plus a side term. It adapts to map RESULTS only,")
    print(f"     NOT to draft/counter-pick/momentum/in-game state (the menu D gap).")
