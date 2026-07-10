# exp4v2 downstream 성능 결과 — aroma-symmetric (SGM + 타일링 + positive placement)

> 출처: `aroma_output/exp4v2/20260705_1`(aroma-sym) · aroma-CP 참조 = 3종 `20260705` / aitex `20260706_aitex`. detector = yolov8n, val=real only, synth_ratio=1.0, seeds 42/1/2.
> baseline/random은 이식값(aroma-sym run과 참조 run이 동일 — graft 정합 검증됨).
> 트랙 A(mtd·severstal·leather): multi-class, imgsz 640/rect, 100ep. 트랙 B(aitex): **single-class·tile-level**, imgsz 256, 300ep. **트랙 간 절대값 직접 비교 금지.**

---

## mAP50 (mean ± std, n=3)

| dataset | baseline | random | aroma-CP | **aroma-sym** | Δ(sym−rand) | Δ(sym−base) | Δ(sym−CP) | per-seed 부호(sym−rand) |
|---|---|---|---|---|---|---|---|---|
| **severstal** | 0.4975 ±.008 | 0.4968 ±.010 | 0.4970 ±.008 | **0.5927 ±.011** | **+0.0959** | +0.0952 | **+0.0957** | ✅ +.086/+.096/+.106 |
| mtd | 0.9204 ±.022 | 0.9310 ±.014 | 0.9348 ±.018 | 0.9172 ±.019 | −0.0138 | −0.0032 | −0.0176 | ❌ −/−/− (천장) |
| mvtec_leather | 0.8340 ±.045 | 0.9077 ±.036 | 0.8987 ±.029 | 0.8830 ±.039 | −0.0247 | +0.0490 | −0.0157 | ❌ −.031/−.092/+.049 |
| aitex † | 0.3719 ±.024 | 0.3881 ±.026 | 0.4847 | 0.3986 ±.027 | +0.0105 | +0.0267 | −0.0861 | ❌ +.029/−.020/+.022 |

† aitex = **tile-level** mAP(256 타일, 50% overlap 중복 계수) — 절대값 타 데이터셋과 직접 비교 불가. aroma-CP(0.4847)는 20260706_aitex 실측.

## mAP50-95 (참고)

| dataset | baseline | random | aroma-CP | aroma-sym |
|---|---|---|---|---|
| severstal | 0.2297 | 0.2335 | 0.2353 | **0.2940** (+0.0587 vs CP) |
| mtd | 0.6331 | 0.6612 | 0.6604 | 0.6034 |
| mvtec_leather | 0.4920 | 0.5876 | 0.5920 | 0.5174 |
| aitex † | 0.1672 | 0.1766 | 0.2181 | 0.1827 |

---

## 판독

- **severstal = 헤드라인 (개선 결정적)**: aroma-sym 0.5927로 random(+9.6pp)·baseline(+9.5pp)·aroma-CP(+9.6pp)를 **전 seed 일관 상회**(std 0.011로 낮음). mAP50-95도 +5.9pp vs CP. copy-paste(aroma-CP)는 random과 사실상 동률(flat, H1 벽)이었는데 **symmetric 게이트가 이 flat을 돌파** → 배치/게이트 개선의 순효과가 downstream에서 실측됨. AROMA 중심주장(AROMA>Random)의 직접 증거.
- **mtd = near-ceiling(측정 불가)**: baseline 0.92 천장. sym이 random 대비 −1.4pp(노이즈 범위)로 flat — **사전 등록 규칙대로 개선 무효 결론 금지**. mtd는 회귀 없음·class_key 정상 확인용.
- **mvtec_leather = random 강세**: random 0.9077이 이미 최강, sym은 baseline은 +4.9pp 상회하나 random·CP엔 미달, per-seed 비일관. leather는 copy-paste 재조합이 강해 게이트 개선 여지 작음.
- **aitex(tile-level) = random은 이기나 CP 미달**: sym이 baseline(+2.7pp)·random(+1.1pp) 상회하나 전 seed 일관은 아님. aroma-CP 0.4847이 최강(sym−CP −8.6pp) — aitex는 elongated 결함 비중이 커 aroma-sym이 대부분 AR copy_paste 폴백으로 처리됐을 가능성(ControlNet 생성 기여 제한). "생성/배치 arm도 random은 이긴다"로 분리 해석, CP 초과는 실패.

**종합**: symmetric 게이트 개선은 **headroom이 있고 배경-배치 민감도가 높은 severstal에서 결정적(+9.6pp, 전seed)**. near-ceiling(mtd)·random 강세(leather)·copy-paste 우세(aitex)에서는 이득이 없거나 CP 미달 → 개선은 **유효하나 데이터셋 조건부**. 헤드라인 arbiter = severstal.

---

## 무결성 / 정직

- **graft 정합**: 4종 baseline/random이 참조 run과 완전 동일(이식값). aroma arm만 재학습 — 비교 유효.
- **leather aroma-sym 출처**: `exp4v2_results_mvtec_leather.json`(2026-07-10 재실행, 0.8830)을 채택. 구 aggregated `exp4v2_results.json`(2026-07-08)의 leather aroma(0.8759)는 stale — 미사용.
- **개선 순효과** = aroma-sym vs aroma-CP(동일 selection·seed·real, 합성 파이프라인만 상이). severstal +9.6pp가 유일한 명확 순이득.
- **트랙 분리**: aitex(single·tile-level·256·300ep)와 3종(multi·640·100ep)은 절대값 직접 비교 금지. Δ만 트랙 내 유효.
- **aitex 생성 기여**: aroma-sym의 ControlNet 생성 비중 = 1 − ar_fallback. aitex는 낮을 수 있어 CP 미달을 "생성 novelty 부족"으로 단정 말 것(폴백 비율 병기 필요).
- 사후 튜닝 금지. 테스트 코드·pytest 금지(CLAUDE.md).
