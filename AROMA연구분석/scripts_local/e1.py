# -*- coding: utf-8 -*-
import csv, json, os
from bisect import bisect_right

CTX = ['local_variance','edge_density','texture_entropy','frequency_energy','orientation_consistency']
BASE = 'D:/project/AROMA_DATASET'
DSMAP = {  # ds -> profiling dir
 'severstal':'profiling_severstal','mtd':'profiling_mtd','aitex':'profiling_aitex'
}

def cell_key(vals, edges):
    parts=[]
    for f,v in zip(CTX,vals):
        b=min(bisect_right(edges[f], v), 2)
        parts.append(str(b))
    return '_'.join(parts)

def stem(p):
    return os.path.splitext(os.path.basename(p))[0]

def load_hists(ds, pdir):
    edges = json.load(open(f'{BASE}/profiling/{pdir}/compatibility_matrix.json',encoding='utf-8'))['bin_edges']
    good={}   # image_id -> {cell:count}
    defect={}
    with open(f'{BASE}/profiling/{pdir}/context_features.csv') as f:
        r=csv.reader(f); next(r)
        for row in r:
            iid=row[0]; itype=row[1]
            vals=[float(x) for x in row[3:8]]
            ck=cell_key(vals,edges)
            pool = good if itype=='good' else defect
            d=pool.setdefault(iid,{})
            d[ck]=d.get(ck,0)+1
    # normalize
    def norm(pool):
        out={}
        for iid,cnt in pool.items():
            tot=sum(cnt.values())
            out[iid]={k:v/tot for k,v in cnt.items()}
        return out
    return norm(good), norm(defect)

def intersect(a,b):
    s=0.0
    for k,v in a.items():
        if k in b: s+=min(v,b[k])
    return s

print(f"{'ds':<10}{'N':>5}{'miss':>5}{'sim_chosen':>12}{'sim_random':>12}{'sim_best':>10}{'lift':>10}{'%>rand':>9}")
print('-'*73)
for ds,pdir in DSMAP.items():
    good,defect = load_hists(ds,pdir)
    good_list=list(good.values())
    anns=json.load(open(f'{BASE}/synth/{ds}/annotations.json',encoding='utf-8'))
    sc=[]; sr=[]; sb=[]; wins=0; miss=0; used=0
    for a in anns:
        ss=stem(a['source_roi']); cs=stem(a['normal_image'])
        if ss not in defect or cs not in good:
            miss+=1; continue
        hs=defect[ss]; hc=good[cs]
        simc=intersect(hs,hc)
        rand=sum(intersect(hs,g) for g in good_list)/len(good_list)
        best=max(intersect(hs,g) for g in good_list)
        sc.append(simc); sr.append(rand); sb.append(best)
        if simc>rand: wins+=1
        used+=1
    mc=sum(sc)/len(sc); mr=sum(sr)/len(sr); mb=sum(sb)/len(sb)
    print(f"{ds:<10}{used:>5}{miss:>5}{mc:>12.4f}{mr:>12.4f}{mb:>10.4f}{mc-mr:>+10.4f}{100*wins/used:>8.1f}%")
