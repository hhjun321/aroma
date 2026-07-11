import json, os, cv2
from collections import defaultdict
import pandas as pd

# raw base per dataset
RAW = {'mtd':'D:/project/AROMA_DATASET/mtd',
       'leather':'D:/project/AROMA_DATASET/leather',
       'aitex':'D:/project/AROMA_DATASET/aitex_tiled/aitex_tiled'}
PROF = {'mtd':'profiling_mtd','leather':'profilng_leather','aitex':'profiling_aitex'}

def colab_to_local(p, ds):
    p = p.replace('\\','/')
    # find marker and take tail after the ds-root
    if ds=='mtd':
        i = p.find('/Aroma/mtd/'); return RAW['mtd']+p[i+len('/Aroma/mtd'):] if i>=0 else None
    if ds=='leather':
        i = p.find('/leather/');  # after mvtec/leather
        return RAW['leather']+p[i+len('/leather'):] if i>=0 else None
    if ds=='aitex':
        i = p.find('/aitex_tiled/'); return RAW['aitex']+p[i+len('/aitex_tiled'):] if i>=0 else None
    return None

_dimcache={}
def dims(path):
    if path in _dimcache: return _dimcache[path]
    im = cv2.imread(path) if path and os.path.exists(path) else None
    r = (im.shape[1], im.shape[0]) if im is not None else None  # (W,H)
    _dimcache[path]=r
    return r

def classify(x,y,w,h,W,H):
    if w >= 0.8*W or h >= 0.8*H: return 'span'
    mx, my = 0.08*W, 0.08*H
    if x <= mx or y <= my or (x+w) >= (W-mx) or (y+h) >= (H-my): return 'edge'
    return 'surface'

def parse_bbox(b):
    if isinstance(b,str): return [float(v) for v in b.split(',')]
    return [float(v) for v in b]

report = {}
for ds in ['mtd','leather','aitex']:
    # ---- REAL geometry from morphology_features.csv ----
    df = pd.read_csv('D:/project/AROMA_DATASET/profiling/'+PROF[ds]+'/morphology_features.csv')
    real = defaultdict(lambda: defaultdict(int)); real_fail=0
    for _,row in df.iterrows():
        cls = row['defect_type']
        loc = colab_to_local(row['image_path'], ds)
        d = dims(loc)
        if d is None: real_fail+=1; continue
        x,y,w,h = parse_bbox(row['defect_bbox'])
        real[cls][classify(x,y,w,h,d[0],d[1])]+=1
    # ---- PLACED geometry from synth annotations ----
    synds = {'mtd':'mtd','leather':'leather','aitex':'aitex'}[ds]
    ann = json.load(open('D:/project/AROMA_DATASET/synth/'+synds+'/annotations.json',encoding='utf-8'))
    placed = defaultdict(lambda: defaultdict(int)); placed_fail=0
    for r in ann:
        if r.get('dry_run'): continue
        cls = r['class_key']
        loc = colab_to_local(r['normal_image'], ds)
        d = dims(loc)
        if d is None: placed_fail+=1; continue
        x,y,w,h = parse_bbox(r['bbox'])
        placed[cls][classify(x,y,w,h,d[0],d[1])]+=1
    report[ds]={'real':real,'placed':placed,'real_fail':real_fail,'placed_fail':placed_fail}

def es_pct(counts):
    tot=sum(counts.values())
    if tot==0: return None,0
    return 100.0*(counts.get('edge',0)+counts.get('span',0))/tot, tot

print('%-8s %-12s %6s %6s %8s %8s %7s'%('ds','class','realN','plcN','real_e+s%','plc_e+s%','gap'))
print('-'*70)
rows=[]
for ds in report:
    classes = sorted(set(report[ds]['real'])|set(report[ds]['placed']))
    for c in classes:
        rp,rn = es_pct(report[ds]['real'][c])
        pp,pn = es_pct(report[ds]['placed'][c])
        gap = (None if rp is None or pp is None else pp-rp)
        rows.append((ds,c,rn,pn,rp,pp,gap))
        print('%-8s %-12s %6d %6d %8s %8s %7s'%(ds,c,rn,pn,
            '%.1f'%rp if rp is not None else '-',
            '%.1f'%pp if pp is not None else '-',
            '%+.1f'%gap if gap is not None else '-'))
print()
for ds in report:
    print('fails', ds, 'real', report[ds]['real_fail'], 'placed', report[ds]['placed_fail'])
