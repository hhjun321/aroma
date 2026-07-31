# clean-bg / void 게이트가 cv2 dtype 예외로 fail-open (전 환경 무력화)

## (성격: 결함 발견·증거 기록 — 수정은 미착수)

`_is_clean_background`가 **순수 검은 패치조차 clean으로 판정**한다. 원인은 `_background_quality_score`의 OpenCV 호출 조합이 cv2 4.13.0에서 미지원이라 예외가 나고, 호출부가 이를 삼켜 `True`(clean)를 반환하는 fail-open 구조다. **로컬·Colab 모두 재현.** 발견 경위: `aroma_core_compatibility_model_20260729.md` §6 시각화 자료 제작 중 void율 측정에서 5종 전부 0.0%가 나와 역추적.

---

## 1. 근인

`scripts/aroma/generate_defects.py`:

```python
# _background_quality_score :457-460
gray = gray.astype(np.float32) if gray.dtype != np.float32 else gray
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#   → cv2.error: Unsupported combination of source format (=5), destination format (=6)
```

```python
# _is_clean_background :517-521
        gray_f = gray.astype(np.float32)          # 이중 변환
        return _background_quality_score(gray_f, blur_threshold) >= min_quality
    except Exception:
        # Evaluation failure must not drop an otherwise-usable background.
        return True                                # ← fail-open
```

`_positive_place`(`:981-984`)도 같은 패턴으로 한 번 더 감싼다 → `is_void = False`.

**예외가 `>= min_quality` 비교 전에 발생하므로 임계값과 무관하다.** 0.7이든 0.5든 동일하게 fail-open.

### Laplacian 조합별 지원 (cv2 4.13.0 실측)

| src | dst | 결과 |
|---|---|---|
| uint8 | CV_64F | OK |
| uint8 | CV_32F | OK |
| **float32** | **CV_64F** | **RAISES** |
| float32 | CV_32F | OK |

### 검증

```python
>>> _is_clean_background(np.zeros((64,64), np.uint8), 0.7, 100.0)
True                    # 순수 검은 패치가 clean
>>> _background_quality_score(np.zeros((64,64), np.uint8))
cv2.error: ... Unsupported combination of source format (=5), and destination format (=6)
```

**로컬** cv2 4.13.0 / **Colab** cv2 4.13.0 동일. 즉 로컬 한정이 아니라 **환경 전역**이다.

---

## 2. 영향 범위

`_is_clean_background` / `_background_quality_score` 소비처 전수:

| 위치 | 게이트 | 무력화 결과 |
|---|---|---|
| `:431` | `_foreground_mask` void 전경 거부(`_FG_VOID_QUALITY`) | Mode A 수정([[aroma_exp4v2_foreground-void-rejection]])이 무효 |
| `:980` | `_positive_place` void 배제 | 배치 후보 필터 소실 |
| `:1096` | `_normal_tile_cells` void | 이미지 compat 점수에 void 타일 포함 |
| `:1271` | 타일 창 void 검사 | — |
| `:1552` | random fallback 위치 게이트 | 폴백이 void에 착지 가능 |
| `:2743` | `load_normal_images` pool 게이트(`--reject-clean-bg`) | 풀 필터 소실 |

실측 void율 (5종 good 이미지 64px 타일, 무작위 표본):

| 데이터셋 | 표본 | void 타일 |
|---|---|---|
| severstal | 400장 / 30,000 타일 | **0** |
| kolektor | 120장 | **0** |
| mtd | 120장 | **0** |
| mvtec_leather | 120장 | **0** |
| aitex_tiled | 200장 | **0** |

⇒ devnote에 기록된 "clean-bg 게이트 항상 ON" 정책([[project_cleanbg_gate_policy]])이 **현 환경에서 no-op**이다.

---

## 3. 수정이 단순 dtype 교체로 끝나지 않는다

`ddepth`를 `CV_32F`로 바꾸면 게이트가 되살아나지만 **반대 극단**이 된다. `min_quality=0.7` 기준 void율:

| 데이터셋 | void@0.7 | void@0.5 | quality median |
|---|---|---|---|
| severstal | 99.8% | **23.4%** | 0.580 |
| kolektor | 100.0% | 78.4% | 0.458 |
| mtd | 98.4% | **22.0%** | 0.602 |
| mvtec_leather | 100.0% | 99.6% | 0.432 |
| aitex_tiled | 98.5% | **11.5%** | 0.543 |

**원인** — 가중식 `0.30·blur + 0.30·contrast + 0.20·brightness + 0.20·noise`에서:

- `blur`: Laplacian 분산 ≥ `blur_threshold`(100)이면 1.0, 아니면 **0.3**. 매끄러운 64px 산업 표면 타일은 대개 100 미달 → 0.3 고정
- `contrast`: `min(std/128, 1.0)`. 강판 타일 std 10~30 → 0.08~0.23

두 항(가중 합 60%)이 구조적으로 낮아 quality median이 0.43~0.60에 머문다. **CASDA에서 검증된 임계 0.7은 256×256 ROI 패치를 전제한 값**이고 64px 타일로 그대로 이식된 것이 문제다.

**운영 설정은 `min_quality=0.5`** 로 하향돼 있다(사용자 확인). 그 기준에서 severstal·mtd·aitex는 11~23%로 합리적, leather 99.6%는 포화(별건).

### 수정 시 함께 정할 것

1. **dtype** — `cv2.Laplacian(gray, cv2.CV_32F)` (또는 uint8 입력 유지). 명백한 버그.
2. **fail-open 정책** — `except: return True`가 결함을 은폐한다. 최소한 `logger.warning` 1회 + 카운터. "게이트 ON인데 실제로는 OFF"가 조용히 성립하는 구조를 남기면 안 된다.
3. **`min_quality` 근거** — 0.5가 실측상 타당하나 데이터 유도값이 아니다. 64px 타일 quality 분포의 분위(예 P10~P20)로 유도할지 결정. `blur_threshold=100`도 타일 크기에 맞춰 재검토 대상.
4. **leather 포화** — 0.5에서도 99.6%. 가죽 표면 quality median 0.432. 데이터셋별 임계 또는 지표 재설계 필요 여부 판단.

---

## 4. 결과 재현성 함의

과거 devnote에 게이트가 **실제로 작동한 실측**이 있다:

- [[aroma_exp4v2_foreground-void-rejection]] (2026-06-28): `_foreground_mask` None 7.8%→18.7%, 검은배경 via_fg 33→9, 검은배경 비율 aroma 9.1%→6.0%
- [[aroma_exp4v2_clean-background-gate]] (2026-06-26): 게이트 구현·리뷰 완료

그 시점 Colab의 OpenCV는 `CV_32F → CV_64F`를 허용했다는 뜻이다. 그 후 cv2가 4.13.0으로 올라가며 게이트가 조용히 죽었다.

⇒ **동일 코드·동일 플래그가 OpenCV 버전에 따라 게이트 ON/OFF로 갈린다.** 어느 산출물이 게이트 유효 상태에서 나온 것인지 구분이 필요하다. 최소한:

- [ ] 게이트 관련 실측을 인용하는 문서에 "cv2 버전 의존" 주석
- [ ] 수정 후 재실행 시 로그에 게이트 실제 발동 수(reject 카운트)를 남겨 검증 가능하게

---

## 5. 상태

**미수정.** 사용자 판단으로 이번 작업 범위(문서·시각화) 밖으로 분리. §3의 4개 결정 사항이 정리되면 별도 dev_note로 구현.

관련: [[aroma_cleanbg_void-floor-fix]](offline `_patch_void` 별개 결함), [[aroma_exp4v2_aroma-underperformance-diagnosis]](게이트 무효 진단 전례 — 당시는 pool 게이트 R=0)
