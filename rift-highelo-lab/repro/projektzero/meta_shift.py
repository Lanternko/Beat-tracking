"""Understand the meta: how big is each patch's shift, how fast does the meta turn over,
and is the shift non-uniform (small pre-Worlds, large mid-season) as expected?

Metric: per patch, the champion pick-share distribution (major leagues only). The realized
meta shift between consecutive patches = Total Variation Distance TVD = 0.5*sum|p-q| in [0,1].
(This measures REALIZED shift -- conflates patch-content size with tournament/roster effects,
but that's what actually moves the meta.) Also: meta half-life (patches until TVD vs a
reference > 0.5) and TVD-by-calendar-month to test the pre-Worlds-small hypothesis.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter

MAJORS = {'LCK','LPL','LEC','LCS','MSI','WLDs','FST','EWC','LTA','LTAN','LCKC'}
yrs=[2023,2024,2025,2026]
raw=pd.concat([pd.read_csv(f'/tmp/oe_data/{y}_oe.csv',low_memory=False) for y in yrs],ignore_index=True)
raw['date']=pd.to_datetime(raw['date'])
pl=raw[(raw.position!='team') & (raw.league.isin(MAJORS))][['date','patch','position','champion']].dropna(
    subset=['champion','patch'])
pl['patch']=pl['patch'].astype(str)

def pkey(p):
    try:
        a,b=p.split('.')[:2]; return (int(a),int(b))
    except: return None
pl['pk']=pl['patch'].map(pkey)
pl=pl[pl.pk.notna()]

# per-patch champion pick-share + date, keep patches with enough games
rows=[]
for pk,sub in pl.groupby('pk'):
    if len(sub) < 600: continue          # ~>=60 games
    dist=Counter(sub.champion); tot=sum(dist.values())
    share={c:n/tot for c,n in dist.items()}
    rows.append({'pk':pk,'patch':f"{pk[0]}.{pk[1]:02d}",'date':sub.date.median(),
                 'month':sub.date.median().month,'n':len(sub),'share':share})
rows.sort(key=lambda r:r['pk'])
print(f"major-league patches with >=60 games: {len(rows)}")

def tvd(a,b):
    keys=set(a)|set(b)
    return 0.5*sum(abs(a.get(k,0)-b.get(k,0)) for k in keys)

print(f"\n{'patch':7s} {'date':>10s} {'games':>6s} {'TVD vs prev':>12s}   shift")
for i,r in enumerate(rows):
    if i==0:
        print(f"{r['patch']:7s} {str(r['date'].date()):>10s} {r['n']:6d} {'(start)':>12s}")
        continue
    t=tvd(r['share'],rows[i-1]['share'])
    bar='#'*int(t*60)
    print(f"{r['patch']:7s} {str(r['date'].date()):>10s} {r['n']:6d} {t:12.3f}   {bar}")

# meta half-life: from each patch, patches forward until TVD vs it > 0.5
hl=[]
for i in range(len(rows)):
    for j in range(i+1,len(rows)):
        if tvd(rows[i]['share'],rows[j]['share'])>0.5:
            hl.append(j-i); break
print(f"\n=== meta turnover: patches until 50% of pick-mass changes (TVD>0.5) ===")
print(f"  median {np.median(hl):.0f} patches, mean {np.mean(hl):.1f}  (n={len(hl)} measurable)")

# non-uniformity: consecutive TVD by calendar month (test 'pre-Worlds small, mid-season big')
print("\n=== shift magnitude by month (consecutive-patch TVD) ===")
tvds=[(rows[i]['month'],tvd(rows[i]['share'],rows[i-1]['share'])) for i in range(1,len(rows))]
md=pd.DataFrame(tvds,columns=['month','tvd'])
season={1:'preseason',2:'spring',3:'spring',4:'spring/MSI',5:'MSI',6:'summer',
        7:'summer',8:'pre-Worlds',9:'pre-Worlds',10:'Worlds',11:'preseason',12:'preseason'}
md['period']=md.month.map(season)
g=md.groupby('period')['tvd'].agg(['mean','count']).sort_values('mean')
for per,row in g.iterrows():
    print(f"  {per:12s} mean TVD {row['mean']:.3f}  (n={int(row['count'])})")
print(f"\n  biggest single shifts:")
big=sorted(range(1,len(rows)),key=lambda i:tvd(rows[i]['share'],rows[i-1]['share']),reverse=True)[:4]
for i in big:
    print(f"    {rows[i-1]['patch']}->{rows[i]['patch']} ({rows[i]['date'].date()}): TVD {tvd(rows[i]['share'],rows[i-1]['share']):.3f}")
