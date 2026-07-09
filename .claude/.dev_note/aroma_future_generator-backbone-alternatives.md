# [Future Work / 참조] ControlNet 생성기 backbone 대체 검토 — SD1.5 AR 한계 대응

---

> **성격**: 구현 지시 아님. 추후 연구 착수 시 참조할 설계 검토 노트.
> **발단**: `generate_defects.py`의 ControlNet arm이 bbox를 512² 정사각으로 squash→생성→un-squash 하는 구조라, 고종횡비(AR) 결함에서 수평 스미어/평탄 왜곡 발생. 현재는 AR 게이트(`--cn_ar_threshold`, Option 3)로 AR 초과 ROI를 copy_paste 폴백. 근본 원인은 **SD1.5 base가 512² 정사각으로만 학습**된 것.
> **관련**: [[aroma_controlnet-arm_quality-filters]](AR 게이트 구현), [[aroma_step4_h1-recombination-no-info]](elongated 대면적=저정보 구간), [[project_controlnet_arm]].

## 문제 요약

- SD1.5 + ControlNet, 512² 고정 학습. 추론 시 bbox를 512²로 눌렀다 복원 → elongated(예 15×261, AR 17.4)에서 텍스처 붕괴.
- **추론 해상도만 비정사각으로 바꾸는 것(Option C)은 부당**: 학습이 512²라 train-test 분포 불일치 + SD1.5가 극단 AR에서 OOD 붕괴. 재학습 없는 해상도 변경은 data-driven으로 정당화 불가.
- **AR-bucketing 재학습(Option B)**: 같은 SD1.5 base에서 AR 버킷 학습 → AR~4까지 확장 가능하나, base가 512² 정사각 학습이라 **AR>4 극단은 여전히 불가**(64×1088을 SD1.5가 못 그림). 별도 노트로 검토 예정.
- 본 노트 = **Option 4: 생성 backbone 자체 교체** 검토.

## 핵심 판단 — backbone 교체는 "AR 꼬리"를 위한 값싼 수정이 아니다

AR 왜곡은 소수 고AR 결함(꼬리)에서만 발생. 이 꼬리를 위해 생성 backbone 전체를 교체하는 것은 비용-효과 역전. 아래 이유로 **현 시점 비권장, 재스코프 시에만**:

1. **SD1.5가 이 데이터에 오히려 적합**: 대상 데이터셋(severstal 강판, mtd 자성타일, aitex 직물)은 grayscale·소규모(수백~수천 crop). SD1.5의 가벼운 prior + 작은 해상도가 소데이터에 맞음. 상위 모델의 대용량·photoreal prior는 과적합·분포 미스매치.
2. **파이프라인 전면 재작성**: `build_train_jsonl.py`(target/hint 512² 생성), `train_controlnet.py`(SD1.5+ControlNet), hint 생성기 전부가 SD1.5+ControlNet 512² 전제. backbone 교체 = 이 전부 재설계.
3. **comparability 붕괴**: study 중간 backbone 교체 시 기존 run(20260705_cn 등)과 직접 비교 불가.
4. **연구 프레이밍**: AROMA 기여는 data-driven ROI 선택 + compat + 파이프라인이지 특정 생성기가 아님. 생성기는 교체 가능한 컴포넌트. backbone을 헤드라인에 넣으면 "합성 데이터 유용성 검증"이 "생성기 벤치마크"로 변질.

## 후보 backbone 비교 (AR 유연성 관점)

| 후보 | AR 해결 | 소·grayscale 산업 데이터 적합성 | 비용/리스크 | 판정 |
|------|---------|--------------------------------|-------------|------|
| **SDXL + ControlNet** | 네이티브 멀티-AR 버킷(1024²) | ✗ photoreal prior, 소데이터 과적합, UNet 2.6B VRAM↑ | 큼 | AR은 얻으나 데이터 부적합 |
| **SD2.1 (768²)** | 정사각 학습 유지 → 버킷 별도 필요 | 중간 | 중 | 개선 미미 |
| **PixArt-α/Σ (DiT)** | 네이티브 AR 버킷 + 효율적 | 소데이터 fine-tune 검증 부족, conditioning 생태계 미성숙 | 중~큼 | 중간 옵션, 리스크 |
| **Convolutional GAN** (DefectGAN/SDGAN/DFMGAN) | ✓ FCN → 임의 H×W 네이티브, AR 제약 소멸 | ✓ 산업 결함 few-shot 합성 문헌 표준 | mode collapse·학습 불안정, ControlNet hint/prompt 조건화 상실 | **재스코프 시 1순위** |

**요지**:
- **SDXL은 AR만 해결하고 소·grayscale 데이터엔 부적합** — 이 프로젝트에서 SD1.5 대비 개선 불명확.
- **defect-GAN 계열이 AR 관점 최강** — FCN이라 정사각 제약 자체가 없고, 산업 결함 few-shot 합성 문헌 정합성 높음. 단 ControlNet의 구조적 조건화(morphology hint + prompt)를 버리므로 **base 교체가 아니라 별도 생성 arm(다른 방법론)**으로 다뤄야 함.

## 착수 조건 (decision gate)

backbone 교체는 다음이 모두 성립할 때만 정당:
1. AR 게이트 + Option B(SD1.5 버킷팅, AR≤4)로도 회수 못 하는 **고AR & 고가치 결함이 유의미하게 남음**이 실측됨.
2. 연구 목표를 "생성기 자체 개선/비교"로 **명시적 재스코프**.
3. 기존 run과의 comparability 포기를 감수(전 arm 재실행 예산 확보).

## 권장 우선순위 (재확인)

1. **현행**: AR 게이트 + copy_paste 폴백. 임계는 AR 분포서 도출(STEP 5-0 사전 스캔, [[feedback_prescan_thresholds]]).
2. **회수 규모 측정**: copy_paste 폴백 중 ROI의 AR 2.5~4 & 소면적(H1 §4.3 고가치 c2 후보) 비율. CPU로 AR 분포에서 산출.
3. **Option B (SD1.5 버킷팅)**: 2가 크면 착수. 같은 base 유지 → 저비용 + comparability 보존. AR~4 확장.
4. **Option 4 (backbone 교체)**: 위가 부족 + 재스코프 시에만. SDXL(소데이터 부적합)보다 **defect-GAN 별도 arm** 검토. PixArt는 중간 대안.

## 미해결 / 추후 확인

- 대상 데이터셋의 AR × 면적 결합 분포 — 고AR이 소면적(고가치 thin scratch)과 얼마나 겹치는지 실측 필요(Option B 착수 근거이자 backbone 교체 필요성 판단 근거).
- defect-GAN arm 도입 시 AROMA의 ROI 선택·compat 신호를 GAN 조건으로 어떻게 주입할지(ControlNet hint 대체 설계) — 별도 설계 필요.
- PixArt-ControlNet의 소데이터 fine-tune 안정성 — 소규모 파일럿으로 검증 가능.
