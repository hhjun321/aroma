# AROMA 다중도메인 통합 검증 — AITEX + MTD 실행 가이드

> **실행 환경**: CPU (Stage 1 distribution profiling만; GPU 불필요)
> **전제**:
> - Google Drive 마운트 완료, `os.environ['DRIVE']` 설정됨 (예: `/content/drive/MyDrive/data/Aroma`).
> - AROMA 저장소가 `/content/AROMA`로 clone 되어 있고 `dataset_config.json`에 `aitex`/`mtd` 엔트리 존재 (본 PR에서 추가).
> - AITEX Kaggle 다운로드용 **API 토큰(KGAT_...)** 준비 (Kaggle → Settings → API). STEP 1에서 `getpass`로 입력(평문 저장 금지). + Kaggle에서 해당 데이터셋 약관 수락 완료. (구버전 폴백: `kaggle.json`을 `$DRIVE`에 보관.)
>
> **목적**: 신규 2개 도메인(AITEX 텍스타일 / MTD 자성타일)이 Stage 1 파이프라인을 통과하고, morphology가 **fallback이 아닌 ground-truth mask**로 산출되는지 검증. `mask_source == 'ground_truth'` 및 비-degenerate morphology(solidity<1, extent<1)를 확인해 mask 경로 해소가 정상인지 판정한다.

---

## 환경변수 설정

```python
import os

# AROMA 기본 (기존 셀에서 이미 설정된 경우 생략 가능)
# prepare_aitex.py 는 scripts/aroma/ 에, distribution_profiling.py 는 repo 루트
# scripts/ 에 있다. 서로 다른 디렉토리이므로 env 변수를 2개로 분리한다.
os.environ['AROMA_SCRIPTS']      = '/content/AROMA/scripts/aroma'   # prepare_aitex.py 등
os.environ['AROMA_ROOT_SCRIPTS'] = '/content/AROMA/scripts'         # distribution_profiling.py
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['DATASET_CONFIG'] = '/content/AROMA/dataset_config.json'

# 원본 데이터 다운로드/정규화 경로
os.environ['KAGGLE_DL']  = f"{os.environ['DRIVE']}/kaggle_download"
os.environ['AITEX_RAW']  = f"{os.environ['DRIVE']}/aitex_raw"     # Kaggle 3폴더 압축 해제 위치
os.environ['AITEX_OUT']  = f"{os.environ['DRIVE']}/aitex"         # prepare_aitex 정규화 출력 (dataset_config image_dir 기준)
os.environ['MTD_RAW']    = f"{os.environ['DRIVE']}/mtd_raw"       # MTD Supervisely 루트 (ds/ann, ds/img, meta.json)
os.environ['MTD_OUT']    = f"{os.environ['DRIVE']}/mtd"           # prepare_mtd 정규화 출력 (dataset_config image_dir 기준)

print("AROMA_SCRIPTS      :", os.environ['AROMA_SCRIPTS'])
print("AROMA_ROOT_SCRIPTS :", os.environ['AROMA_ROOT_SCRIPTS'])
print("AROMA_OUT     :", os.environ['AROMA_OUT'])
print("DATASET_CONFIG:", os.environ['DATASET_CONFIG'])
print("AITEX_RAW     :", os.environ['AITEX_RAW'])
print("AITEX_OUT     :", os.environ['AITEX_OUT'])
print("MTD_RAW       :", os.environ['MTD_RAW'])
print("MTD_OUT       :", os.environ['MTD_OUT'])
```

> ⚠️ `dataset_config.json`의 `aitex` `image_dir`은 `.../aitex/train/good`, `mtd` `image_dir`은 `.../mtd/train/good`(prepare_mtd 정규화 출력)를 가정한다. `$DRIVE`가 다르면 config 경로도 함께 맞춘다 (본 가이드는 `$DRIVE = /content/drive/MyDrive/data/Aroma` 가정).

---

## STEP 1 — 원본 데이터 다운로드

### 1-A. AITEX (Kaggle API → Drive)

Kaggle CLI(≥1.8.0) 설치 + **API 토큰(KGAT) 인증** 후 데이터셋을 Drive로 내려받아 압축 해제한다.

> ⚠️ **보안**: KGAT 토큰은 자격증명이다. 노트북/채팅/git에 **평문으로 남기지 말 것.** 노출되면 즉시 Kaggle에서 revoke+재발급. 아래처럼 Colab **`getpass`로 입력**하면 셀 출력·저장본에 안 남는다. (Drive 파일로 두려면 `$DRIVE/kaggle_token.txt`에 두고 읽되, 그 파일 권한/공유에 주의.)

```python
import os, getpass
# 1) KGAT 토큰 지원 버전 보장 (≥1.8.0)
!pip -q install -U "kaggle>=1.8.0"
# 2) 인증: API 토큰(KGAT_...)을 KAGGLE_API_TOKEN 으로. getpass → 출력에 안 남음
os.environ['KAGGLE_API_TOKEN'] = getpass.getpass('Kaggle API token (KGAT_...): ')
# 3) 다운로드 (-p: 대상 디렉토리)
!mkdir -p $KAGGLE_DL
!kaggle datasets download -d nexuswho/aitex-fabric-image-database -p $KAGGLE_DL
```

> 대안(kaggle.json 방식): `$DRIVE/kaggle.json`(`{"username","key"}`)을 `~/.config/kaggle/`로 복사 후 `chmod 600` — 구버전 CLI 호환. KGAT 방식이 안 되면 이걸로 폴백.

압축 해제 (`-o` 덮어쓰기로 비대화형 셀 hang 방지):

```python
import os
!mkdir -p $AITEX_RAW
!unzip -q -o $KAGGLE_DL/aitex-fabric-image-database.zip -d $AITEX_RAW
# 3폴더 확인 (Defect_images / Mask_images / NODefect_images)
!ls -R $AITEX_RAW | head -40
```

> 다운로드 zip 파일명이 다르면(`aitex-fabric-image-database.zip` 이 아니면) `!ls $KAGGLE_DL` 로 실제 파일명을 확인해 위 unzip 경로를 수정한다. 3폴더가 상위 디렉토리 하나 더 안쪽에 풀릴 수 있으니 `ls -R` 결과로 실제 `Defect_images/` 위치를 확인하고 STEP 2의 인자 경로를 맞춘다.

### 1-B. MTD (Dataset Ninja **Supervisely** 배포판 → Drive)

⚠️ MTD는 **Supervisely 포맷**을 사용한다 (abin24 GitHub `MT_*/Imgs` co-located 아님). 구조:
```
<MTD_RAW>/meta.json                 # 5 클래스(blowhole/break/crack/fray/uneven), 전부 bitmap
<MTD_RAW>/ds/img/<name>.jpg         # 1344장
<MTD_RAW>/ds/ann/<name>.jpg.json    # {size, objects:[{classTitle, bitmap:{data(base64-zlib-PNG RGBA), origin}}], tags}
```
- `objects==[]` → good(956장), `objects!=[]` → 결함(388장, 459 obj). Dataset Ninja(datasetninja.com/magnetic-tile-surface-defect)에서 받아 `$MTD_RAW`에 둔다.

원본 존재 확인:
```python
import os, glob, json, collections
R = os.environ['MTD_RAW']                       # Supervisely 루트 (ds/ann, ds/img, meta.json)
print("meta:", os.path.exists(f"{R}/meta.json"),
      "| ann:", len(glob.glob(f"{R}/ds/ann/*.json")),
      "| img:", len(glob.glob(f"{R}/ds/img/*")))
# 클래스/그룹 분포 빠른 확인
c=collections.Counter(); emp=0
for f in glob.glob(f"{R}/ds/ann/*.json"):
    o=json.load(open(f))['objects']
    emp += (not o)
    for x in o: c[x.get('classTitle')]+=1
print("free(good)=", emp, "| obj per class=", dict(c))   # 기대: good≈956, 5 class 합≈459
```

> ann≈1344, img≈1344, good≈956. 다르면 `$MTD_RAW` 경로/다운로드 확인.

---

## STEP 2 — prepare 정규화 (AITEX + MTD)

두 셋 모두 MVTec식 레이아웃(`train/good`, `test/{class}`, `ground_truth/{class}/{stem}_mask.png`)으로 정규화 → 기존 `_find_mask_path` 해소 재사용.

**2-A. AITEX** — 3폴더 → `NODefect_images → train/good`, `Defect_images → test/{ddd}`, `Mask_images → ground_truth/{ddd}/{stem}_mask.png`. mask 없거나 빈 결함 skip+count.
```python
!python $AROMA_SCRIPTS/prepare_aitex.py \
    --defect_images   $AITEX_RAW/Defect_images \
    --mask_images     $AITEX_RAW/Mask_images \
    --nodefect_images $AITEX_RAW/NODefect_images \
    --output_dir      $AITEX_OUT
```

**2-B. MTD** — Supervisely bitmap 주석을 full-frame mask로 래스터화 → `objects==[] → train/good`, 결함 class별 → `test/{class}` + `ground_truth/{class}/{stem}_mask.png`(class-union). 빈 mask skip+count.
```python
!python $AROMA_SCRIPTS/prepare_mtd.py \
    --supervisely_root $MTD_RAW \
    --output_dir       $MTD_OUT
# 산출 확인
import json, os
mani = f"{os.environ['MTD_OUT']}/mtd_manifest.json"
if os.path.exists(mani):
    c = json.load(open(mani))['counts']
    print("good           :", c['good'])              # ≈956
    print("defect_entries :", c['defect_entries'])    # ≈459
    print("per_class      :", c['per_class'])
    print("skipped_empty  :", c['skipped_empty_mask'], " no_img:", c['skipped_no_img'])
else:
    print("MISSING —", mani)
```

**출력 확인 (AITEX):**

```python
import json, os
mani = f"{os.environ['AITEX_OUT']}/aitex_manifest.json"
if os.path.exists(mani):
    with open(mani) as f:
        m = json.load(f)
    c = m['counts']
    print("normal(good)      :", c['normal'])
    print("defect matched    :", c['defect_images_matched'])
    print("defect total      :", c['defect_images_total'])
    print("mask files total  :", c['mask_files_total'])
    print("skipped(no mask)  :", c['skipped_no_mask'])
    print("unparsed code     :", c['unparsed_code'])
    print("defect_codes      :", m['defect_codes'])
    print("per_defect_code   :", c['per_defect_code'])
else:
    print(f"MISSING — {mani}")
```

> `skipped_no_mask`가 크면 (AITEX 알려진 특성: mask 없는 결함 존재) 정상이다. **`defect_codes`에 실제 존재하는 코드**를 확인해 STEP 3에서 config seed_dirs를 실제 폴더에 맞춰 정리한다.

---

## STEP 3 — dataset_config 엔트리 확인 / 정리

`dataset_config.json`에 `aitex`/`mtd` 엔트리가 존재하는지 확인한다. AITEX seed_dirs는 placeholder superset이므로, STEP 2에서 실제 생성된 `test/{ddd}` 폴더만 남기도록 정리한다.

```python
import json, os
cfg_path = os.environ['DATASET_CONFIG']
with open(cfg_path) as f:
    cfg = json.load(f)

for key in ('aitex', 'mtd'):
    if key in cfg:
        print(f"[{key}] domain={cfg[key]['domain']}")
        print(f"     image_dir={cfg[key]['image_dir']}")
        print(f"     seed_dirs({len(cfg[key]['seed_dirs'])}):")
        for d in cfg[key]['seed_dirs']:
            print(f"        exists={os.path.isdir(d)}  {d}")
    else:
        print(f"[{key}] MISSING in config")
```

**AITEX seed_dirs를 실제 생성된 폴더로 자동 정리(선택):** placeholder 중 실존하지 않는 `test/{ddd}`를 제거한다.

```python
import json, os
cfg_path = os.environ['DATASET_CONFIG']
with open(cfg_path) as f:
    cfg = json.load(f)

test_root = f"{os.environ['AITEX_OUT']}/test"
if os.path.isdir(test_root):
    real = sorted(f"{test_root}/{d}" for d in os.listdir(test_root)
                  if os.path.isdir(f"{test_root}/{d}"))
    cfg['aitex']['seed_dirs'] = real
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"aitex seed_dirs → {len(real)}개 실존 폴더로 정리:")
    for d in real:
        print("   ", d)
else:
    print(f"MISSING — {test_root} (STEP 2 먼저 실행)")
```

---

## STEP 4 — Stage 1 distribution_profiling 스모크

각 데이터셋을 소규모(`--max_images`)로 돌려 파이프라인 통과 여부를 확인한다. `--num_workers 0` 은 오류 추적이 쉬운 순차 실행.

```python
import os
os.environ['PROFILING_AITEX'] = f"{os.environ['AROMA_OUT']}/profiling/aitex"
os.environ['PROFILING_MTD']   = f"{os.environ['AROMA_OUT']}/profiling/mtd"
```

**AITEX:**

```python
!python $AROMA_ROOT_SCRIPTS/distribution_profiling.py \
    --dataset_config $DATASET_CONFIG \
    --dataset_key    aitex \
    --output_dir     $PROFILING_AITEX \
    --max_images     20 \
    --num_workers    0
```

**MTD:**

```python
!python $AROMA_ROOT_SCRIPTS/distribution_profiling.py \
    --dataset_config $DATASET_CONFIG \
    --dataset_key    mtd \
    --output_dir     $PROFILING_MTD \
    --max_images     20 \
    --num_workers    0
```

**출력 확인 (morphology_features.csv 생성 여부):**

```python
import os
for ds, env in (('aitex', 'PROFILING_AITEX'), ('mtd', 'PROFILING_MTD')):
    path = f"{os.environ[env]}/morphology_features.csv"
    if os.path.exists(path):
        n = sum(1 for _ in open(path)) - 1  # header 제외
        print(f"{ds}: morphology_features.csv rows={n}")
    else:
        print(f"{ds}: MISSING — {path}")
```

> 로그에 `fallback masks: 0` 이면 모든 결함이 GT mask로 해소된 것. `fallback masks: N`(N>0)이면 mask 경로 해소 실패 → STEP 5에서 원인 진단.

---

## STEP 5 — 검증: ground-truth mask + 비-degenerate morphology

핵심 판정. morphology가 **fallback이 아닌 ground_truth mask**로 산출됐는지, 그리고 morphology가 붕괴(bbox-as-mask → solidity/extent≈1)하지 않았는지 확인한다.

```python
import csv, os

def verify(ds, env):
    path = f"{os.environ[env]}/morphology_features.csv"
    if not os.path.exists(path):
        print(f"[{ds}] MISSING — {path}")
        return
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"[{ds}] EMPTY csv (row 0개) — 모든 결함이 drop됨 (mask 해소 실패 의심)")
        return

    n = len(rows)
    src_counts = {}
    for r in rows:
        src_counts[r['mask_source']] = src_counts.get(r['mask_source'], 0) + 1
    n_gt = src_counts.get('ground_truth', 0)

    def fnum(r, k):
        try:
            return float(r[k])
        except (ValueError, KeyError, TypeError):
            return None

    sol = [fnum(r, 'solidity') for r in rows if fnum(r, 'solidity') is not None]
    ext = [fnum(r, 'extent')   for r in rows if fnum(r, 'extent')   is not None]
    mean_sol = sum(sol) / len(sol) if sol else float('nan')
    mean_ext = sum(ext) / len(ext) if ext else float('nan')
    # bbox-as-mask 붕괴: solidity≈1 AND extent≈1 이면 mask가 bbox로 대체된 신호
    n_degen = sum(1 for r in rows
                  if (fnum(r, 'solidity') or 0) >= 0.999
                  and (fnum(r, 'extent') or 0) >= 0.999)

    print(f"[{ds}] rows={n}  mask_source={src_counts}")
    print(f"      ground_truth 비율 : {n_gt}/{n} = {n_gt/n:.1%}")
    print(f"      mean solidity={mean_sol:.3f}  mean extent={mean_ext:.3f}  "
          f"(둘 다 <1 이어야 정상; degenerate rows={n_degen})")

    ok_gt    = (n_gt == n)
    ok_morph = (mean_sol < 0.999) and (mean_ext < 0.999)
    verdict = 'PASS' if (ok_gt and ok_morph) else 'FAIL'
    print(f"      >>> {verdict}  "
          f"(mask_source 전부 ground_truth={ok_gt}, 비-degenerate morphology={ok_morph})")
    if not ok_gt:
        print(f"      ✗ fallback_* 존재 → mask 경로 해소 실패. "
              f"aitex: prepare_aitex ground_truth/{{ddd}}/ 확인 / mtd: prepare_mtd ground_truth/{{class}}/{{stem}}_mask.png 확인")
    if not ok_morph:
        print(f"      ✗ morphology 붕괴(solidity/extent≈1) → mask가 bbox로 대체됨. mask 내용 점검")
    print()

verify('aitex', 'PROFILING_AITEX')
verify('mtd',   'PROFILING_MTD')
```

**판정 기준:**

| 지표 | 기대값 | 실패 시 의미 |
|------|--------|-------------|
| `mask_source` | 전부 `ground_truth` | `fallback_sam`/`fallback_otsu` 존재 → GT mask 경로 해소 실패 |
| `mean solidity` | `< 1.0` | ≈1.0 → mask가 bbox로 붕괴 (내용 없는 mask) |
| `mean extent` | `< 1.0` | ≈1.0 → 동일 (bbox-as-mask) |
| csv rows | > 0 | 0 → 모든 결함 drop (mask 로드/추출 전부 실패) |

- **AITEX**: `fallback_*`가 나오면 → `prepare_aitex.py`가 `ground_truth/{ddd}/{stem}_mask.png`를 생성했는지, config seed_dirs의 `{ddd}` 폴더명이 `ground_truth/{ddd}` 와 일치하는지 확인 (`_find_mask_path` aitex 분기는 `image_path.parent.parent.parent/ground_truth/{defect_type}/{stem}_mask.png` 조회).
- **MTD**: `fallback_*`가 나오면 → `prepare_mtd.py`가 `ground_truth/{class}/{stem}_mask.png`를 생성했는지, config seed_dirs `test/{class}`와 클래스명이 일치하는지 확인 (`_find_mask_path` mtd 분기는 `image_path.parent.parent.parent/ground_truth/{defect_type}/{stem}_mask.png` 조회, aitex 미러).

---

## 결과 확인

두 도메인 모두 STEP 5에서 **PASS**(mask_source 전부 ground_truth + solidity/extent<1)이면 통합 성공. 이후 Severstal/MVTec leather와 함께 4셋 breadth 실험(Stage 1b 이하)으로 진행한다.

## 기대 결과

| 데이터셋 | mask_source | morphology | 비고 |
|---------|-------------|-----------|------|
| aitex | ground_truth | solidity/extent < 1 | mask 없는 결함은 prepare 단계서 skip됨(정상) |
| mtd | ground_truth | solidity/extent < 1 | defect_type = MT_{class} (Blowhole/Break/Crack/Fray/Uneven) 5종으로 구분됨. co-located .png mask는 이미지 glob(.jpg 전용)에서 제외됨 |

## 출력 파일

| 경로 | 설명 |
|------|------|
| `$AITEX_OUT/aitex_manifest.json` | prepare_aitex 정규화 매니페스트 (counts, defect_codes) |
| `$AITEX_OUT/{train/good, test/{ddd}, ground_truth/{ddd}}` | MVTec식 정규화 레이아웃 |
| `$AROMA_OUT/profiling/aitex/morphology_features.csv` | AITEX Stage1 morphology (검증 대상) |
| `$AROMA_OUT/profiling/mtd/morphology_features.csv` | MTD Stage1 morphology (검증 대상) |
