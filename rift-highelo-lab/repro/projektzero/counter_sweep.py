"""Is counter-pick conditioning useful at ANY weight? One pass over M3+counter,
ranking each pick under several beta (0 = no counter ... 0.6 = counter-heavy).
If no beta beats beta=0, counter-pick (champion-identity response) is not a lever.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter, defaultdict

ALPHA=0.50; G_META,G_PLAYER=0.80,0.93
BETAS=[0.0,0.10,0.20,0.35,0.50]
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
games=[]
for gid,sub in tm.groupby('gameid'):
    if len(sub)!=2: continue
    info={'wk':int(sub.wk.iloc[0]),'date':sub.date.iloc[0],'gameid':gid,'teams':frozenset(sub.teamid),'bans':set(),'picks':{}}
    for r in sub.itertuples():
        info['bans'].update([getattr(r,f'ban{i}') for i in range(1,6)])
        info['picks'][r.side]=[getattr(r,f'pick{i}') for i in range(1,6)]
    games.append(info)
games.sort(key=lambda g:(g['date'],g['gameid']))
print(f"games {len(games)}")

gmeta={r:Counter() for r in ROLES}; ppool=defaultdict(Counter)
counter=defaultdict(lambda:defaultdict(Counter)); series_used=defaultdict(set); last_wk=games[0]['wk']
# accumulators per beta: top1, top5 ; plus counter-available subset
T1={b:0 for b in BETAS}; T5={b:0 for b in BETAS}; n=0
T1c={b:0 for b in BETAS}; nc=0
for g in games:
    if g['wk']>last_wk:
        dm,dp=G_META**(g['wk']-last_wk),G_PLAYER**(g['wk']-last_wk)
        for r in ROLES:
            for c in list(gmeta[r]): gmeta[r][c]*=dm
        for pid in list(ppool):
            for c in list(ppool[pid]): ppool[pid][c]*=dp
        last_wk=g['wk']
    cmg=cm.get(g['gameid'])
    if cmg is None: continue
    prior=series_used[(g['teams'],g['date'].date())] if g['date'].year>=2025 else set()
    taken=set(g['bans'])|prior; revealed={}
    for side,idx in ORDER:
        champ=g['picks'][side][idx-1]
        if champ not in cmg[side]: taken.add(champ); continue
        pid,role=cmg[side][champ]; opp='Red' if side=='Blue' else 'Blue'
        mtot=sum(gmeta[role].values()) or 1.0; ptot=sum(ppool[pid].values()) or 1.0
        cand=set(c for c,_ in gmeta[role].most_common(20))|set(ppool[pid]); cand-=taken
        if not cand: cand={champ}
        enemy=None
        for rc,(rs,rr) in revealed.items():
            if rs==opp and rr==role: enemy=rc; break
        ctot=sum(counter[role][enemy].values()) if enemy else 0
        base={c:ALPHA*(ppool[pid][c]/ptot)+(1-ALPHA)*(gmeta[role][c]/mtot) for c in cand}
        cnt={c:(counter[role][enemy][c]/ctot if (enemy and ctot) else 0.0) for c in cand}
        avail = enemy is not None and ctot>0
        for b in BETAS:
            ranked=sorted(cand,key=lambda c:(1-b)*base[c]+b*cnt[c] if avail else base[c],reverse=True)
            h1=ranked[0]==champ
            T1[b]+=h1; T5[b]+=champ in ranked[:5]
            if avail: T1c[b]+=h1
        n+=1; nc+=avail
        taken.add(champ); revealed[champ]=(side,role)
    for side in ('Blue','Red'):
        for champ,(pid,role) in cmg[side].items(): gmeta[role][champ]+=1; ppool[pid][champ]+=1
    for role in ROLES:
        bc=[c for c,(p,rr) in cmg['Blue'].items() if rr==role]; rc=[c for c,(p,rr) in cmg['Red'].items() if rr==role]
        if bc and rc: counter[role][rc[0]][bc[0]]+=1; counter[role][bc[0]][rc[0]]+=1
    series_used[(g['teams'],g['date'].date())].update(revealed)

print(f"\nbeta  top-1(all)  recall@5(all)   top-1 on counter-available subset (n={nc})")
for b in BETAS:
    print(f"  {b:.2f}   {T1[b]/n*100:6.1f}%    {T5[b]/n*100:6.1f}%        {T1c[b]/nc*100:6.1f}%")
print("-> if beta=0 is best, counter-pick (champ-identity response) adds no signal.")
