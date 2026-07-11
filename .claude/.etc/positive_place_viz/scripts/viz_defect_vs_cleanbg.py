"""
좌=실제 결함 본 이미지(+defect_bbox), 우=그 결함 cluster의 top-10 compat 선택 clean-bg.
cluster별 1행. 커밋 23bd373 함수 재사용.
"""
import sys, os, csv, ast, glob, json
import numpy as np, cv2
sys.path.insert(0, "D:/project/aroma/scripts"); sys.path.insert(0, "D:/project/aroma/scripts/aroma")
import distribution_profiling as dp
import generate_defects as gd

ETC = "D:/project/aroma/.claude/.etc"
OUT = f"{ETC}/positive_place_viz"; os.makedirs(OUT, exist_ok=True)
ROW_H = 300      # 행 높이 (좌 defect 패널 높이)
THUMB = 140      # 우 clean 썸네일
NTOP = 10

def ds_dir(ds): return 'leather' if ds=='mvtec_leather' else ds
def parse_bbox(s):
    try: return tuple(int(v) for v in ast.literal_eval(s))
    except Exception: return None
def find_src(ds, dt, iid):
    for e in ('.png','.jpg','.jpeg','.bmp'):
        p=f"{ETC}/{ds_dir(ds)}/test/{dt}/{iid}{e}"
        if os.path.exists(p): return p
    return None

def fit_h(img, h):
    s=h/img.shape[0]; return cv2.resize(img,(max(1,int(img.shape[1]*s)),h))

def clean_grid(top, cols=5, thumb=THUMB):
    thumbs=[]
    for i,(sc,gp) in enumerate(top):
        im=cv2.imread(gp)
        if im is None: continue
        im=cv2.resize(im,(thumb,thumb))
        cv2.rectangle(im,(0,0),(thumb-1,thumb-1),(0,200,0),2)
        cv2.putText(im,f"#{i+1} {sc:.3f}",(3,thumb-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,220,0),1,cv2.LINE_AA)
        thumbs.append(im)
    rows=[]
    for r in range(0,len(thumbs),cols):
        ch=thumbs[r:r+cols]
        while len(ch)<cols: ch.append(np.full((thumb,thumb,3),20,np.uint8))
        rows.append(np.hstack(ch))
    return np.vstack(rows) if rows else np.zeros((thumb,thumb*cols,3),np.uint8)

def run(ds, good_dir):
    prof=f"{ETC}/profiling_tobe/{ds}"
    compat=json.load(open(f"{prof}/compatibility_matrix.json"))
    msym=compat['matrix_symmetric']; be=compat['bin_edges']
    ca={str(k):int(v) for k,v in json.load(open(f"{prof}/morphology_clusters.json"))['cluster_assignments'].items()}
    rows=list(csv.DictReader(open(f"{prof}/morphology_features.csv")))
    clusters=[c for c,r in msym.items() if r]

    # clean 이미지 스코어 (cluster별 top-10)
    goods=sorted(glob.glob(f"{good_dir}/*"))
    cells={gp:gd._normal_tile_cells(cv2.imread(gp,cv2.IMREAD_GRAYSCALE),be) for gp in goods if cv2.imread(gp,cv2.IMREAD_GRAYSCALE) is not None}
    def top_clean(c):
        s=[(gd._image_compat_score(cells[gp],msym[c]),gp) for gp in cells]
        s=[(sc,gp) for sc,gp in s if sc>=0]; s.sort(reverse=True); return s[:NTOP]

    # cluster별 대표 결함 (해당 cluster의 첫 유효 결함)
    rep={}
    for r in rows:
        iid=r['image_id']; dt=r.get('defect_type',''); cl=ca.get(iid); bb=parse_bbox(r.get('defect_bbox',''))
        if cl is None or str(cl) not in clusters or not bb or len(bb)!=4: continue
        if str(cl) in rep: continue
        src=find_src(ds,dt,iid)
        if src: rep[str(cl)]=(dt,iid,bb,src)

    blocks=[]
    for c in clusters:
        if c not in rep: continue
        dt,iid,(x,y,w,h),src=rep[c]
        # 좌: 결함 본 이미지 + bbox
        sb=cv2.imread(src); cv2.rectangle(sb,(x,y),(x+w,y+h),(0,0,255),max(2,sb.shape[1]//250))
        L=fit_h(sb,ROW_H)
        # 우: top-10 clean grid, 높이 ROW_H로 맞춤
        G=clean_grid(top_clean(c)); G=fit_h(G,ROW_H)
        div=np.full((ROW_H,8,3),80,np.uint8)
        body=np.hstack([L,div,G])
        bar=np.full((26,body.shape[1],3),30,np.uint8)
        cv2.putText(bar,f"{ds} cluster{c} | LEFT: defect {dt}#{iid} (red bbox) | RIGHT: TOP-{NTOP} selected clean-bg",
                    (6,19),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        blocks.append(np.vstack([bar,body]))
    W=max(b.shape[1] for b in blocks)
    blocks=[np.hstack([b,np.full((b.shape[0],W-b.shape[1],3),20,np.uint8)]) if b.shape[1]<W else b for b in blocks]
    montage=np.vstack(blocks)
    out=f"{OUT}/defect_vs_cleanbg_{ds}.png"
    cv2.imwrite(out,montage); print(f"saved: {out} ({montage.shape[1]}x{montage.shape[0]})  clusters={list(rep)}")

if __name__=="__main__":
    run('mvtec_leather', f"{ETC}/leather/train/good")
    run('mtd', f"{ETC}/mtd/train/good")
    print("DONE ->",OUT)
