"""Does TEAM identity (org/coach draft habits) add over meta+player? Completes the
'team & player tendencies' direction. Full pipeline (remove taken + fearless 2025+).
configs: meta-only | +player | +player+team. If +team ~= +player, team adds nothing
beyond the player already on the roster.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter, defaultdict

G_META,G_PLAYER=0.80,0.93
ROLES=['top','jng','mid','bot','sup']
ORDER=[('Blue',1),('Red',1),('Red',2),('Blue',2),('Blue',3),('Red',3),('Red',4),('Blue',4),('Blue',5),('Red',5)]
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
side_team={}
games=[]
for gid,sub in tm.groupby('gameid'):
    if len(sub)!=2: continue
    st={r.side:r.teamid for r in sub.itertuples()}
    side_team[gid]=st
    info={'wk':int(sub.wk.iloc[0]),'date':sub.date.iloc[0],'gameid':gid,'teams':frozenset(sub.teamid),'bans':set(),'picks':{}}
    for r in sub.itertuples():
        info['bans'].update([getattr(r,f'ban{i}') for i in range(1,6)])
        info['picks'][r.side]=[getattr(r,f'pick{i}') for i in range(1,6)]
    games.append(info)
games.sort(key=lambda g:(g['date'],g['gameid']))
print(f"games {len(games)}")

gmeta={r:Counter() for r in ROLES}; ppool=defaultdict(Counter); tpool=defaultdict(Counter)
series_used=defaultdict(set); last_wk=games[0]['wk']
CFG=['meta','+player','+player+team']
T1={c:0 for c in CFG}; T5={c:0 for c in CFG}; n=0
for g in games:
    if g['wk']>last_wk:
        dm,dp=G_META**(g['wk']-last_wk),G_PLAYER**(g['wk']-last_wk)
        for r in ROLES:
            for c in list(gmeta[r]): gmeta[r][c]*=dm
        for k in list(ppool):
            for c in list(ppool[k]): ppool[k][c]*=dp
        for k in list(tpool):
            for c in list(tpool[k]): tpool[k][c]*=dp
        last_wk=g['wk']
    cmg=cm.get(g['gameid'])
    if cmg is None: continue
    st=side_team[g['gameid']]
    prior=series_used[(g['teams'],g['date'].date())] if g['date'].year>=2025 else set()
    taken=set(g['bans'])|prior; revealed=[]
    for side,idx in ORDER:
        champ=g['picks'][side][idx-1]
        if champ not in cmg[side]: taken.add(champ); continue
        pid,role=cmg[side][champ]; tid=st[side]
        mtot=sum(gmeta[role].values()) or 1.0
        ptot=sum(ppool[pid].values()) or 1.0
        ttot=sum(tpool[(tid,role)].values()) or 1.0
        cand=set(c for c,_ in gmeta[role].most_common(20))|set(ppool[pid])|set(tpool[(tid,role)]); cand-=taken
        if not cand: cand={champ}
        m={c:gmeta[role][c]/mtot for c in cand}
        p={c:ppool[pid][c]/ptot for c in cand}
        t={c:tpool[(tid,role)][c]/ttot for c in cand}
        scorers={'meta':lambda c:m[c],
                 '+player':lambda c:0.5*p[c]+0.5*m[c],
                 '+player+team':lambda c:0.5*(0.5*p[c]+0.5*t[c])+0.5*m[c]}
        for cfg in CFG:
            ranked=sorted(cand,key=scorers[cfg],reverse=True)
            T1[cfg]+=ranked[0]==champ; T5[cfg]+=champ in ranked[:5]
        n+=1; taken.add(champ); revealed.append((champ,pid,role,tid))
    for champ,pid,role,tid in revealed:
        gmeta[role][champ]+=1; ppool[pid][champ]+=1; tpool[(tid,role)][champ]+=1
    series_used[(g['teams'],g['date'].date())].update(c for c,_,_,_ in revealed)

print(f"\n{'config':16s} {'top-1':>7s} {'recall@5':>9s}")
for cfg in CFG:
    print(f"{cfg:16s} {T1[cfg]/n*100:6.1f}% {T5[cfg]/n*100:8.1f}%")
print("-> if +player+team ~= +player, team identity adds nothing beyond the rostered player.")
