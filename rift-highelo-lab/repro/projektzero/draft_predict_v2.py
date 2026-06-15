"""v2 champion-pick predictor: sequential, draft-state-aware, player-aware, fearless-aware.

Walk-forward. Predicts each pick in true draft order (B1,R1,R2,B2,B3,R3,R4,B4,B5,R5),
removing champions already banned/taken (and, optionally, used earlier in the Bo5 series
= fearless). Blends ROLE-META popularity with the PLAYER's own champion pool. Ablation:

  M0 meta, no draft-state  -> reproduces the ~15/55 floor
  M1 + remove banned/taken -> draft-state aware
  M2 + player affinity     -> each player's pool (the big lever for top-1)
  M3 + fearless removal     -> prior-series champs gone (Bo5 sequential)

Decay: role-meta & player pools decay weekly (exp) so the model tracks the meta.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter, defaultdict

ALPHA = 0.50          # weight on player affinity vs role meta (swept: 0.5 best)
G_META, G_PLAYER = 0.80, 0.93   # weekly decay (meta moves faster than a player's pool)
ROLES = ['top', 'jng', 'mid', 'bot', 'sup']
# global draft pick order: slot -> (side, that side's pick index 1..5)
ORDER = [('Blue',1),('Red',1),('Red',2),('Blue',2),('Blue',3),
         ('Red',3),('Red',4),('Blue',4),('Blue',5),('Red',5)]

yrs = [2023, 2024, 2025, 2026]
raw = pd.concat([pd.read_csv(f'/tmp/oe_data/{y}_oe.csv', low_memory=False) for y in yrs],
                ignore_index=True)
raw['date'] = pd.to_datetime(raw['date'])
wk0 = raw['date'].min()
raw['wk'] = ((raw['date'] - wk0).dt.days // 7).astype(int)

pl = raw[raw.position != 'team'][['gameid','date','wk','position','playerid','champion','teamid','side']].dropna(
    subset=['champion','position','playerid'])
tm = raw[raw.position == 'team'][['gameid','date','wk','side','teamid','game',
                                  'ban1','ban2','ban3','ban4','ban5',
                                  'pick1','pick2','pick3','pick4','pick5']]

# per game: champion->(playerid,role) by side, bans, picks-in-order, series key
champ_meta = {}   # gameid -> dict(side -> {champ:(pid,role)})
for gid, sub in pl.groupby('gameid'):
    d = {'Blue': {}, 'Red': {}}
    for r in sub.itertuples():
        d[r.side][r.champion] = (r.playerid, r.position)
    champ_meta[gid] = d

games = []  # ordered list of (wk, date, gameid, info)
for gid, sub in tm.groupby('gameid'):
    if len(sub) != 2: continue
    info = {'wk': int(sub.wk.iloc[0]), 'date': sub.date.iloc[0], 'gameid': gid,
            'game_no': sub.game.iloc[0], 'teams': frozenset(sub.teamid),
            'bans': set(), 'picks': {}}
    for r in sub.itertuples():
        info['bans'].update([getattr(r, f'ban{i}') for i in range(1,6)])
        info['picks'][r.side] = [getattr(r, f'pick{i}') for i in range(1,6)]
    games.append(info)
games.sort(key=lambda g: (g['date'], g['gameid']))
print(f"games: {len(games)} | player-picks ~{len(pl)}")


def run(use_player, remove_taken, remove_fearless):
    role_meta = {r: Counter() for r in ROLES}
    player_pool = defaultdict(Counter)
    series_used = defaultdict(set)            # series key -> champs used so far
    last_wk = games[0]['wk']
    t1 = t5 = n = 0
    by_role = defaultdict(lambda: [0,0,0])    # role -> [t1,t5,n]
    by_gameno = defaultdict(lambda: [0,0,0])

    for g in games:
        # weekly decay so pools track the meta
        if g['wk'] > last_wk:
            dec_m, dec_p = G_META**(g['wk']-last_wk), G_PLAYER**(g['wk']-last_wk)
            for r in ROLES:
                for c in list(role_meta[r]):
                    role_meta[r][c] *= dec_m
            for pid in list(player_pool):
                for c in list(player_pool[pid]):
                    player_pool[pid][c] *= dec_p
            last_wk = g['wk']

        cm = champ_meta.get(g['gameid'])
        if cm is None: continue
        skey = (g['teams'], g['date'].date())
        # fearless only became the rule in 2025+ (verified: 2023-24 series repeat champs,
        # 2025-26 repeat-rate ~0). Removing prior-series champs pre-2025 would be wrong.
        is_fearless = remove_fearless and g['date'].year >= 2025
        prior_series = series_used[skey] if is_fearless else set()
        taken = set(g['bans']) | prior_series if remove_taken else set()
        # invert champ->(pid,role) per side to pick champion by draft slot
        revealed = []

        for side, idx in ORDER:
            champ = g['picks'][side][idx-1]
            if champ not in cm[side]:      # data hiccup (pick/role mismatch)
                if remove_taken: taken.add(champ)
                continue
            pid, role = cm[side][champ]
            # score candidates
            mtot = sum(role_meta[role].values()) or 1.0
            cand = set(c for c,_ in role_meta[role].most_common(20))
            if use_player: cand |= set(player_pool[pid])
            cand -= taken
            if champ in taken:            # actual pick can't be in taken; safety
                cand.discard(champ)
            if not cand:
                cand = {champ}
            ptot = sum(player_pool[pid].values()) or 1.0
            def score(c):
                m = role_meta[role][c]/mtot
                p = player_pool[pid][c]/ptot if use_player else 0.0
                return (ALPHA*p + (1-ALPHA)*m) if use_player else m
            ranked = sorted(cand, key=score, reverse=True)
            hit1 = ranked[0] == champ
            hit5 = champ in ranked[:5]
            t1 += hit1; t5 += hit5; n += 1
            by_role[role][0]+=hit1; by_role[role][1]+=hit5; by_role[role][2]+=1
            gn = int(g['game_no']) if not pd.isna(g['game_no']) else 1
            by_gameno[gn][0]+=hit1; by_gameno[gn][1]+=hit5; by_gameno[gn][2]+=1
            if remove_taken: taken.add(champ)
            revealed.append(champ)

        # update pools after the game
        for side in ('Blue','Red'):
            for champ,(pid,role) in cm[side].items():
                role_meta[role][champ] += 1
                player_pool[pid][champ] += 1
        series_used[skey].update(revealed)
    return t1/n, t5/n, n, by_role, by_gameno


configs = [("M0 meta, no draft-state", dict(use_player=False, remove_taken=False, remove_fearless=False)),
           ("M1 +remove banned/taken", dict(use_player=False, remove_taken=True,  remove_fearless=False)),
           ("M2 +player affinity",     dict(use_player=True,  remove_taken=True,  remove_fearless=False)),
           ("M3 +fearless (Bo5 seq)",   dict(use_player=True,  remove_taken=True,  remove_fearless=True))]
print(f"\n{'config':28s} {'top-1':>7s} {'recall@5':>9s}")
last = None
for name, kw in configs:
    a1, a5, n, br, bg = run(**kw)
    print(f"{name:28s} {a1*100:6.1f}% {a5*100:8.1f}%")
    last = (name, br, bg)

print(f"\n=== best model ({last[0]}) by role ===")
for r in ROLES:
    t1,t5,n = last[1][r]
    print(f"  {r:4s} top-1 {t1/n*100:5.1f}%  recall@5 {t5/n*100:5.1f}%  (n={n})")
print("\n=== by Bo5 game number (fearless should LOWER repeats -> earlier games easier) ===")
for gn in sorted(last[2])[:5]:
    t1,t5,n = last[2][gn]
    if n>500: print(f"  game {gn}: top-1 {t1/n*100:5.1f}%  recall@5 {t5/n*100:5.1f}%  (n={n})")
