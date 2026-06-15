"""v3: add per-REGION meta and COUNTER-PICK conditioning on top of v2's best (M3).
M3 base = remove banned/taken + player affinity(a=0.5) + fearless(2025+).
  +region : role-meta keyed by region (LCK/LPL/EU/AM/INTL/OTHER), blended w/ global
  +counter: when the SAME-ROLE opponent is already revealed in draft order, score
            candidates by the learned response distribution P(our pick | enemy r-laner).
            (counter info is available for the later-picking laner / last pick.)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter, defaultdict

ALPHA=0.50; LAM_REGION=0.6; BETA_COUNTER=0.45
G_META,G_PLAYER=0.80,0.93
ROLES=['top','jng','mid','bot','sup']
ORDER=[('Blue',1),('Red',1),('Red',2),('Blue',2),('Blue',3),('Red',3),('Red',4),('Blue',4),('Blue',5),('Red',5)]
SLOT_SIDE_ROLEPOS={}  # filled per game
def region_of(lg):
    if lg in ('LCK','LCKC'): return 'KR'
    if lg=='LPL': return 'CN'
    if lg in ('LEC','EMEA'): return 'EU'
    if lg in ('LCS','LTA','LTAN','LTAS','NACL'): return 'AM'
    if lg in ('MSI','WLDs','FST','EWC','IEM'): return 'INTL'
    return 'OTHER'

yrs=[2023,2024,2025,2026]
raw=pd.concat([pd.read_csv(f'/tmp/oe_data/{y}_oe.csv',low_memory=False) for y in yrs],ignore_index=True)
raw['date']=pd.to_datetime(raw['date']); wk0=raw['date'].min()
raw['wk']=((raw['date']-wk0).dt.days//7).astype(int)
pl=raw[raw.position!='team'][['gameid','side','position','playerid','champion']].dropna(subset=['champion','position','playerid'])
tm=raw[raw.position=='team'][['gameid','date','wk','side','teamid','game','league','ban1','ban2','ban3','ban4','ban5','pick1','pick2','pick3','pick4','pick5']]
cm={}
for gid,sub in pl.groupby('gameid'):
    d={'Blue':{},'Red':{}}
    for r in sub.itertuples(): d[r.side][r.champion]=(r.playerid,r.position)
    cm[gid]=d
games=[]
for gid,sub in tm.groupby('gameid'):
    if len(sub)!=2: continue
    info={'wk':int(sub.wk.iloc[0]),'date':sub.date.iloc[0],'gameid':gid,'teams':frozenset(sub.teamid),
          'region':region_of(sub.league.iloc[0]),'bans':set(),'picks':{}}
    for r in sub.itertuples():
        info['bans'].update([getattr(r,f'ban{i}') for i in range(1,6)])
        info['picks'][r.side]=[getattr(r,f'pick{i}') for i in range(1,6)]
    games.append(info)
games.sort(key=lambda g:(g['date'],g['gameid']))
print(f"games {len(games)}")

def run(use_region, use_counter):
    gmeta={r:Counter() for r in ROLES}
    rmeta=defaultdict(lambda:{r:Counter() for r in ROLES})   # region -> role -> Counter
    ppool=defaultdict(Counter)
    counter=defaultdict(lambda:defaultdict(Counter))          # role -> enemy_champ -> Counter(our)
    series_used=defaultdict(set); last_wk=games[0]['wk']
    t1=t5=n=0; byrole=defaultdict(lambda:[0,0,0]); ncov=0
    for g in games:
        if g['wk']>last_wk:
            dm,dp=G_META**(g['wk']-last_wk),G_PLAYER**(g['wk']-last_wk)
            for r in ROLES:
                for c in list(gmeta[r]): gmeta[r][c]*=dm
            for reg in list(rmeta):
                for r in ROLES:
                    for c in list(rmeta[reg][r]): rmeta[reg][r][c]*=dm
            for pid in list(ppool):
                for c in list(ppool[pid]): ppool[pid][c]*=dp
            last_wk=g['wk']
        cmg=cm.get(g['gameid'])
        if cmg is None: continue
        reg=g['region']
        is_fearless=g['date'].year>=2025
        prior=series_used[(g['teams'],g['date'].date())] if is_fearless else set()
        taken=set(g['bans'])|prior
        revealed={}   # champ -> (side, role)
        for side,idx in ORDER:
            champ=g['picks'][side][idx-1]
            if champ not in cmg[side]: taken.add(champ); continue
            pid,role=cmg[side][champ]
            opp='Red' if side=='Blue' else 'Blue'
            mtot=sum(gmeta[role].values()) or 1.0; ptot=sum(ppool[pid].values()) or 1.0
            if use_region:
                rm=rmeta[reg][role]; rtot=sum(rm.values()) or 1.0
            cand=set(c for c,_ in gmeta[role].most_common(20))|set(ppool[pid])
            if use_region: cand|=set(c for c,_ in rmeta[reg][role].most_common(15))
            cand-=taken
            if not cand: cand={champ}
            # counter: is the opponent's same-role champ already revealed?
            enemy=None
            if use_counter:
                for rc,(rs,rr) in revealed.items():
                    if rs==opp and rr==role: enemy=rc; break
            ctot=sum(counter[role][enemy].values()) if (use_counter and enemy) else 0
            def sc(c):
                m=gmeta[role][c]/mtot
                if use_region: m=LAM_REGION*(rmeta[reg][role][c]/rtot)+(1-LAM_REGION)*m
                base=ALPHA*(ppool[pid][c]/ptot)+(1-ALPHA)*m
                if use_counter and enemy and ctot:
                    return (1-BETA_COUNTER)*base+BETA_COUNTER*(counter[role][enemy][c]/ctot)
                return base
            ranked=sorted(cand,key=sc,reverse=True)
            h1=ranked[0]==champ; h5=champ in ranked[:5]
            t1+=h1; t5+=h5; n+=1; byrole[role][0]+=h1; byrole[role][1]+=h5; byrole[role][2]+=1
            if use_counter and enemy: ncov+=1
            taken.add(champ); revealed[champ]=(side,role)
        # updates
        for side in ('Blue','Red'):
            for champ,(pid,role) in cmg[side].items():
                gmeta[role][champ]+=1; rmeta[reg][role][champ]+=1; ppool[pid][champ]+=1
        for role in ROLES:
            bc=[c for c,(p,rr) in cmg['Blue'].items() if rr==role]
            rc=[c for c,(p,rr) in cmg['Red'].items() if rr==role]
            if bc and rc:
                counter[role][rc[0]][bc[0]]+=1; counter[role][bc[0]][rc[0]]+=1
        series_used[(g['teams'],g['date'].date())].update(revealed)
    return t1/n,t5/n,byrole,ncov/n

print(f"\n{'config':24s} {'top-1':>7s} {'recall@5':>9s}  {'counter-cov':>11s}")
for name,kw in [("M3 base",dict(use_region=False,use_counter=False)),
                ("+region meta",dict(use_region=True,use_counter=False)),
                ("+counter-pick",dict(use_region=False,use_counter=True)),
                ("+both (v3)",dict(use_region=True,use_counter=True))]:
    a1,a5,br,cov=run(**kw)
    print(f"{name:24s} {a1*100:6.1f}% {a5*100:8.1f}%  {cov*100:10.1f}%")
    last=(name,br)
print(f"\n=== {last[0]} by role (counter helps lane-counter roles: top/mid) ===")
for r in ROLES:
    t1,t5,nn=last[1][r]
    print(f"  {r:4s} top-1 {t1/nn*100:5.1f}%  recall@5 {t5/nn*100:5.1f}%")
