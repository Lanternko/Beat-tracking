"""Detection: is the predictor's own hit-rate a real-time meta-shift detector?
Run the best v2 predictor walk-forward; for each patch, measure recall@5 on its EARLY
games (first 15% by date, model still trained on the prior meta) vs SETTLED games. The
EARLY DIP should be large when the patch's TVD (realized meta shift) is large -> the
model's error spike both DETECTS and SIZES the shift, no TVD computation needed.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter, defaultdict

ALPHA=0.50; G_META,G_PLAYER=0.80,0.93
ROLES=['top','jng','mid','bot','sup']
ORDER=[('Blue',1),('Red',1),('Red',2),('Blue',2),('Blue',3),('Red',3),('Red',4),('Blue',4),('Blue',5),('Red',5)]
MAJORS={'LCK','LPL','LEC','LCS','MSI','WLDs','FST','EWC','LTA','LTAN'}

yrs=[2023,2024,2025,2026]
raw=pd.concat([pd.read_csv(f'/tmp/oe_data/{y}_oe.csv',low_memory=False) for y in yrs],ignore_index=True)
raw['date']=pd.to_datetime(raw['date']); wk0=raw['date'].min()
raw['wk']=((raw['date']-wk0).dt.days//7).astype(int)
def pk(p):
    try: a,b=str(p).split('.')[:2]; return f"{int(a)}.{int(b):02d}"
    except: return None
raw['pk']=raw['patch'].map(pk)

pl=raw[raw.position!='team'][['gameid','date','wk','position','playerid','champion','side','league','pk']].dropna(subset=['champion','position','playerid','pk'])
tm=raw[raw.position=='team'][['gameid','date','wk','side','teamid','game','pk','league','ban1','ban2','ban3','ban4','ban5','pick1','pick2','pick3','pick4','pick5']].dropna(subset=['pk'])

# per-patch TVD (major leagues) for ground-truth shift size
mp=pl[pl.league.isin(MAJORS)]
share={}
for k,sub in mp.groupby('pk'):
    if len(sub)<600: continue
    d=Counter(sub.champion); t=sum(d.values()); share[k]={c:n/t for c,n in d.items()}
order=sorted(share, key=lambda s:(int(s.split('.')[0]),int(s.split('.')[1])))
tvd={}
for i in range(1,len(order)):
    a,b=share[order[i]],share[order[i-1]]
    tvd[order[i]]=0.5*sum(abs(a.get(k,0)-b.get(k,0)) for k in set(a)|set(b))

# early/settled flag: rank within (league, patch) by date
tm=tm.sort_values('date')
tm['rk']=tm.groupby(['league','pk'])['date'].rank(method='first', pct=True)
early_games=set(tm[tm.rk<=0.15].gameid); settled_games=set(tm[tm.rk>=0.50].gameid)

cm={}
for gid,sub in pl.groupby('gameid'):
    d={'Blue':{},'Red':{}}
    for r in sub.itertuples(): d[r.side][r.champion]=(r.playerid,r.position)
    cm[gid]=d
games=[]
for gid,sub in tm.groupby('gameid'):
    if len(sub)!=2: continue
    info={'wk':int(sub.wk.iloc[0]),'date':sub.date.iloc[0],'gameid':gid,'pk':sub.pk.iloc[0],
          'teams':frozenset(sub.teamid),'bans':set(),'picks':{}}
    for r in sub.itertuples():
        info['bans'].update([getattr(r,f'ban{i}') for i in range(1,6)])
        info['picks'][r.side]=[getattr(r,f'pick{i}') for i in range(1,6)]
    games.append(info)
games.sort(key=lambda g:(g['date'],g['gameid']))

role_meta={r:Counter() for r in ROLES}; player_pool=defaultdict(Counter); series_used=defaultdict(set)
last_wk=games[0]['wk']
# per-patch: [early_hit5, early_n, settled_hit5, settled_n]
pp=defaultdict(lambda:[0,0,0,0])
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
    is_fearless = g['date'].year>=2025
    prior=series_used[(g['teams'],g['date'].date())] if is_fearless else set()
    taken=set(g['bans'])|prior
    is_early = g['gameid'] in early_games; is_settled = g['gameid'] in settled_games
    revealed=[]
    for side,idx in ORDER:
        champ=g['picks'][side][idx-1]
        if champ not in cmg[side]: taken.add(champ); continue
        pid,role=cmg[side][champ]
        mtot=sum(role_meta[role].values()) or 1.0; ptot=sum(player_pool[pid].values()) or 1.0
        cand=set(c for c,_ in role_meta[role].most_common(20))|set(player_pool[pid]); cand-=taken
        if not cand: cand={champ}
        ranked=sorted(cand,key=lambda c:ALPHA*(player_pool[pid][c]/ptot)+(1-ALPHA)*(role_meta[role][c]/mtot),reverse=True)
        h5=champ in ranked[:5]
        if is_early: pp[g['pk']][0]+=h5; pp[g['pk']][1]+=1
        elif is_settled: pp[g['pk']][2]+=h5; pp[g['pk']][3]+=1
        taken.add(champ); revealed.append(champ)
    for side in ('Blue','Red'):
        for champ,(pid,role) in cmg[side].items(): role_meta[role][champ]+=1; player_pool[pid][champ]+=1
    series_used[(g['teams'],g['date'].date())].update(revealed)

# correlate early-vs-settled recall@5 drop against patch TVD
xy=[]
for k,(eh,en,sh,sn) in pp.items():
    if en<300 or sn<300 or k not in tvd: continue
    er,sr=eh/en,sh/sn
    xy.append((tvd[k], (sr-er)*100, k, er*100, sr*100))
xy.sort()
print(f"{'patch':7s} {'TVD':>6s} {'early R@5':>9s} {'settled R@5':>11s} {'DIP(pp)':>8s}")
for t,dip,k,er,sr in xy:
    print(f"{k:7s} {t:6.3f} {er:8.1f}% {sr:10.1f}% {dip:8.1f}")
if len(xy)>=4:
    T=np.array([a for a,_,_,_,_ in xy]); D=np.array([b for _,b,_,_,_ in xy])
    r=np.corrcoef(T,D)[0,1]
    print(f"\ncorrelation(patch TVD, model recall@5 dip) = {r:+.2f}  (n={len(xy)})")
    print("-> a bigger realized meta shift => a bigger early-patch hit-rate dip.")
    print("   The deployed model's recall drop IS the shift detector (no TVD needed).")
