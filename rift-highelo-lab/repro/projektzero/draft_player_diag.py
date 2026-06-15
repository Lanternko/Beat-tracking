"""Diagnostic: is PLAYER identity a real lever for top-1, or do players just pick meta?
One walk-forward pass (remove banned/taken always on). For each pick, rank candidates
under several alpha (0=pure meta ... 1=pure player pool) AND record the player's history
depth, so we can see player value separately for warm (rich pool) vs cold players.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter, defaultdict

G_META, G_PLAYER = 0.80, 0.93
ROLES = ['top','jng','mid','bot','sup']
ORDER = [('Blue',1),('Red',1),('Red',2),('Blue',2),('Blue',3),
         ('Red',3),('Red',4),('Blue',4),('Blue',5),('Red',5)]
ALPHAS = [0.0, 0.5, 0.7, 0.85, 1.0]

yrs=[2023,2024,2025,2026]
raw=pd.concat([pd.read_csv(f'/tmp/oe_data/{y}_oe.csv',low_memory=False) for y in yrs],ignore_index=True)
raw['date']=pd.to_datetime(raw['date']); wk0=raw['date'].min()
raw['wk']=((raw['date']-wk0).dt.days//7).astype(int)
pl=raw[raw.position!='team'][['gameid','side','position','playerid','champion']].dropna(subset=['champion','position','playerid'])
tm=raw[raw.position=='team'][['gameid','date','wk','side','teamid','ban1','ban2','ban3','ban4','ban5','pick1','pick2','pick3','pick4','pick5']]
cm={}
for gid,sub in pl.groupby('gameid'):
    d={'Blue':{},'Red':{}}
    for r in sub.itertuples(): d[r.side][r.champion]=(r.playerid,r.position)
    cm[gid]=d
games=[]
for gid,sub in tm.groupby('gameid'):
    if len(sub)!=2: continue
    info={'wk':int(sub.wk.iloc[0]),'date':sub.date.iloc[0],'gameid':gid,'bans':set(),'picks':{}}
    for r in sub.itertuples():
        info['bans'].update([getattr(r,f'ban{i}') for i in range(1,6)])
        info['picks'][r.side]=[getattr(r,f'pick{i}') for i in range(1,6)]
    games.append(info)
games.sort(key=lambda g:(g['date'],g['gameid']))
print(f"games {len(games)}")

role_meta={r:Counter() for r in ROLES}; player_pool=defaultdict(Counter); pgames=Counter()
last_wk=games[0]['wk']
# accumulators: per alpha -> [t1,t5]; and warm/cold split per alpha
acc={a:[0,0] for a in ALPHAS}; warm={a:[0,0] for a in ALPHAS}; cold={a:[0,0] for a in ALPHAS}
nw=nc=n=0
for g in games:
    if g['wk']>last_wk:
        dm,dp=G_META**(g['wk']-last_wk),G_PLAYER**(g['wk']-last_wk)
        for r in ROLES:
            for c in list(role_meta[r]): role_meta[r][c]*=dm
        for pid in list(player_pool):
            for c in list(player_pool[pid]): player_pool[pid][c]*=dp
        last_wk=g['wk']
    cmg=cm.get(g['gameid'])
    if cmg is None: continue
    taken=set(g['bans'])
    for side,idx in ORDER:
        champ=g['picks'][side][idx-1]
        if champ not in cmg[side]:
            taken.add(champ); continue
        pid,role=cmg[side][champ]
        mtot=sum(role_meta[role].values()) or 1.0
        ptot=sum(player_pool[pid].values()) or 1.0
        cand=set(c for c,_ in role_meta[role].most_common(20))|set(player_pool[pid])
        cand-=taken; cand.discard(champ) if champ in taken else None
        if not cand: cand={champ}
        m={c:role_meta[role][c]/mtot for c in cand}
        p={c:player_pool[pid][c]/ptot for c in cand}
        warmth = pgames[pid] >= 20
        for a in ALPHAS:
            ranked=sorted(cand,key=lambda c:a*p[c]+(1-a)*m[c],reverse=True)
            h1=ranked[0]==champ; h5=champ in ranked[:5]
            acc[a][0]+=h1; acc[a][1]+=h5
            (warm if warmth else cold)[a][0]+=h1; (warm if warmth else cold)[a][1]+=h5
        n+=1; nw+=warmth; nc+=(not warmth)
        taken.add(champ)
    for side in ('Blue','Red'):
        for champ,(pid,role) in cmg[side].items():
            role_meta[role][champ]+=1; player_pool[pid][champ]+=1;
        for pid in set(p for p,_ in cmg[side].values()): pgames[pid]+=1

print(f"\nalpha (0=meta..1=player)   top-1   recall@5   [warm n={nw} top-1 | cold n={nc} top-1]")
for a in ALPHAS:
    print(f"  a={a:.2f}   {acc[a][0]/n*100:6.1f}% {acc[a][1]/n*100:8.1f}%      "
          f"warm {warm[a][0]/max(nw,1)*100:5.1f}%   cold {cold[a][0]/max(nc,1)*100:5.1f}%")
