// 학위논문 발표 PPT 생성기 — pptxgenjs
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const path = require("path");
const { applyTheme } = require("C:/Users/HAN/.claude/skills/synced/19abdb77-27a2-423b-a3d2-1d545964cdf9_2b3892ee-0ef1-4c64-849b-e587e11ba414/pptx/scripts/apply_theme.js");

const FIG = path.resolve(__dirname, "../figures");
const OUT = path.resolve(__dirname, "학위논문_발표.pptx");

// ---------- Theme ----------
// 참조 테마: Google Slides "Streamline" (확률론적 주변부 특징 ... .pptx)
const THEME = {
  name: "Streamline",
  headFontFace: "Raleway",
  bodyFontFace: "Lato",
  colors: {
    dk1: "000000", lt1: "FFFFFF", dk2: "595959", lt2: "F3F3F3",
    accent1: "ED561B", accent2: "158158", accent3: "8C8C8C",
    accent4: "50B432", accent5: "C0392B", accent6: "BFE3E3",
    hlink: "058DC7", folHlink: "551A8B",
  },
};
const H = THEME.colors; // hex for hex-only options

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "한호준";
pres.title = "Context-Aware 데이터 증강 기반 산업용 결함 검출 연구";
const C = pres.SchemeColor;

const W = 13.33, HH = 7.5, M = 0.6;
const CONTENT_Y = 1.6, CONTENT_H = 5.4;
const MATH = "Cambria Math";

// ---------- Layouts ----------
const BAND_H = 0.62;
function bars(y, x = M) {
  return [
    { rect: { x, y, w: 0.42, h: 0.05, fill: { color: H.accent2 }, line: { color: H.accent2, width: 0 } } },
    { rect: { x: x + 0.46, y, w: 0.42, h: 0.05, fill: { color: H.accent1 }, line: { color: H.accent1, width: 0 } } },
  ];
}
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: H.lt1 },
  objects: [
    ...bars(1.95),
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.15, w: W - 2 * M, h: 1.7, fontSize: 34, bold: false, color: H.dk1, align: "left", valign: "top", margin: 0 } } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.3, w: W - 2 * M, h: 2.4, fontSize: 16, color: H.dk2, align: "left", valign: "top", margin: 0 } } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: H.lt1 },
  objects: [
    { rect: { x: 0, y: 0, w: W, h: BAND_H, fill: { color: H.lt2 }, line: { color: H.lt2, width: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.72, w: W - 2 * M, h: 0.62, fontSize: 24, bold: false, color: H.dk1, align: "left", valign: "middle", margin: 0 } } },
    ...bars(1.38),
  ],
  slideNumber: { x: W - M - 0.8, y: HH - 0.5, w: 0.8, h: 0.35, fontSize: 10, color: H.hlink, fontFace: "Lato", align: "right" },
});

// ---------- Helpers ----------
let section = "";
function slide(title, kicker, opts = {}) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  if (kicker) s.addText(kicker, { x: M, y: 0.14, w: W - 2 * M, h: 0.36, fontSize: 11, bold: true, color: C.text2, margin: 0, isTextBox: true, valign: "middle", objectName: "kicker" });
  s.addText(title, { placeholder: "title" });
  return s;
}
function txt(s, text, x, y, w, h, o = {}) {
  s.addText(text, Object.assign({ x, y, w, h, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, valign: "top", align: "left", objectName: typeof text === "string" ? text.slice(0, 20) : "text" }, o));
}
function bullets(s, items, x, y, w, h, o = {}) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: o.gap ?? 6 } }));
  s.addText(runs, Object.assign({ x, y, w, h, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "bullets" }, o));
}
function card(s, x, y, w, h, o = {}) {
  s.addShape(pres.ShapeType.roundRect, Object.assign({ x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: H.lt2, width: 0 }, objectName: "card" }, o));
}
function label(s, text, x, y, w, h, o = {}) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06, fill: { color: o.fill || C.accent1 }, line: { color: o.fill || H.accent1, width: 0 }, objectName: "pill" });
  s.addText(text, { x, y, w, h, fontSize: o.fontSize || 12, bold: true, color: o.color || C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "pill-text" });
}
function arrow(s, x, y, w, h, o = {}) {
  s.addShape(pres.ShapeType.rightArrow, Object.assign({ x, y, w, h, fill: { color: C.accent1 }, line: { color: H.accent1, width: 0 }, objectName: "arrow" }, o));
}
function math(s, text, x, y, w, h, o = {}) {
  s.addText(text, Object.assign({ x, y, w, h, fontFace: MATH, fontSize: 16, color: C.text1, margin: 0, isTextBox: true, align: "left", valign: "middle", objectName: "math" }, o));
}
async function fig(s, file, x, y, maxW, maxH, o = {}) {
  const p = path.join(FIG, file);
  const meta = await sharp(p).metadata();
  const r = meta.width / meta.height;
  let w = maxW, h = maxW / r;
  if (h > maxH) { h = maxH; w = maxH * r; }
  const ax = o.align === "left" ? x : o.align === "right" ? x + maxW - w : x + (maxW - w) / 2;
  const ay = o.valign === "top" ? y : y + (maxH - h) / 2;
  s.addImage({ path: p, x: ax, y: ay, w, h, objectName: file });
  if (o.caption) s.addText(o.caption, { x, y: y + maxH + 0.05, w: maxW, h: 0.3, fontSize: 10, color: C.accent3, align: "center", margin: 0, isTextBox: true, objectName: "caption" });
  return { x: ax, y: ay, w, h };
}
const chartBase = (title) => ({
  showTitle: false, title, titleFontSize: 12, titleColor: H.dk1, titleFontFace: "+mn-lt",
  catAxisLabelColor: H.dk2, valAxisLabelColor: H.dk2, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  valGridLine: { color: "DDE2E6", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelFontSize: 9, dataLabelColor: H.dk1, dataLabelFormatCode: "0.00",
  legendFontSize: 10, legendColor: H.dk2, legendFontFace: "+mn-lt",
});

// 결함/배경 분류 규칙 블록 — 14p·26p 공통 구조 (입력 특성 → 분류 기준 → 결과)
function ruleBlock(s, x, y, w, h, o) {
  card(s, x, y, w, h, { fill: { color: C.accent2 } });
  label(s, o.tag, x + 0.25, y + 0.2, 1.6, 0.4, { fill: o.tagColor || C.accent1, fontSize: 11 });
  txt(s, o.title, x + 2.0, y + 0.2, w - 2.25, 0.4, { bold: true, fontSize: 14, color: C.background1, valign: "middle" });
  txt(s, [{ text: "입력 특성  ", options: { bold: true, color: H.accent6 } }, { text: o.features, options: {} }], x + 0.25, y + 0.7, w - 0.5, 0.55, { fontSize: 10.5, color: C.background1 });
  txt(s, [{ text: "분류 기준  ", options: { bold: true, color: H.accent6 } }, { text: o.criterion, options: {} }], x + 0.25, y + 1.22, w - 0.5, 0.35, { fontSize: 10.5, color: C.background1 });
  math(s, o.lines.map((t, i) => ({ text: t, options: { breakLine: i < o.lines.length - 1 } })), x + 0.25, y + 1.6, w - 0.5, h - 2.25, { fontSize: o.mathSize || 12, color: C.background1, valign: "top" });
  txt(s, [{ text: "결과  ", options: { bold: true, color: H.accent6 } }, { text: o.result, options: {} }], x + 0.25, y + h - 0.55, w - 0.5, 0.45, { fontSize: 10.5, color: C.background1, valign: "middle" });
}
function transition(s, fromTxt, toTxt, y = 5.55) {
  // 한계 → 다음 장 연결 띠
  const h = Math.min(1.2, 6.95 - y);
  card(s, M, y, W - 2 * M, h, { fill: { color: C.accent2 } });
  txt(s, fromTxt, M + 0.3, y + 0.1, 5.6, h - 0.2, { color: C.background1, fontSize: 13, valign: "middle" });
  arrow(s, M + 6.1, y + h / 2 - 0.2, 0.6, 0.4);
  txt(s, toTxt, M + 6.9, y + 0.1, 5.2, h - 0.2, { color: C.accent6, fontSize: 13, bold: true, valign: "middle" });
}

// ====================================================================
(async () => {
  // ---------------- I. 도입 ----------------
  section = "I. 도입"; pres.addSection({ title: section });
  {
    const s = pres.addSlide({ masterName: "TITLE", sectionTitle: section });
    s.addText([
      { text: "Context-Aware 데이터 증강 기반 산업용 결함 검출 연구", options: { breakLine: true, fontSize: 32 } },
      { text: "A Study on Context-Aware Data Augmentation for Industrial Defect Detection", options: { fontSize: 18, bold: true, color: H.dk2 } },
    ], { placeholder: "title" });
    s.addText([
      { text: "결함 생성(What to place)과 배치(Where to place)로 이어지는 4편 연구의 발전 궤적", options: { breakLine: true, fontSize: 14, color: H.dk2 } },
      { text: " ", options: { breakLine: true, fontSize: 8 } },
      { text: "한호준", options: { breakLine: true, fontSize: 16, color: H.dk1 } },
      { text: "학위논문 심사 발표", options: { fontSize: 13, color: H.dk2 } },
    ], { placeholder: "body" });
  }
  {
    const s = slide("연구 배경: 결함은 희소하고, 데이터는 불균형하다", "제1장 서론 · 연구 배경");
    const items = [
      ["자동 외관 검사(AVI)", "딥러닝 도입으로 검사 효율·정확도 향상. 그러나 성능은 학습 데이터 양과 질에 종속"],
      ["결함 샘플의 희소성", "실제 공정의 결함 발생률이 낮아 결함 이미지 자체를 충분히 확보하기 어려움"],
      ["클래스 불균형", "정상 ≫ 결함, 결함 간에도 편중 → 소수 결함 클래스에 과소적합 반복"],
    ];
    items.forEach(([h, d], i) => {
      const y = CONTENT_Y + i * 1.55;
      card(s, M, y, W - 2 * M, 1.3);
      s.addShape(pres.ShapeType.ellipse, { x: M + 0.3, y: y + 0.3, w: 0.7, h: 0.7, fill: { color: C.accent1 }, line: { color: H.accent1, width: 0 }, objectName: "dot" });
      txt(s, String(i + 1), M + 0.3, y + 0.3, 0.7, 0.7, { color: C.background1, bold: true, fontSize: 20, align: "center", valign: "middle" });
      txt(s, h, M + 1.3, y + 0.2, 10.5, 0.45, { bold: true, fontSize: 17 });
      txt(s, d, M + 1.3, y + 0.65, 10.8, 0.55, { fontSize: 14, color: C.text2 });
    });
    txt(s, "→ 데이터 증강으로 희소성·불균형을 완화하는 것이 본 논문의 출발점", M, 6.2, W - 2 * M, 0.5, { bold: true, color: C.accent1, fontSize: 15 });
  }
  {
    const s = slide("문제 정의: 결함 이미지 증강 = What to place + Where to place", "제1장 서론 · 문제 정의");
    const colW = (W - 2 * M - 0.4) / 2;
    [["P1  생성 문제 (What to place)", "어떤 결함 패치를 생성·증강할 것인가", ["실제 결함의 형태·구조 특성 보존", "학습에 유효한 다양성 확보", "품질이 낮으면 비현실적 데이터가 검출 성능 저하"], C.accent2],
     ["P2  배치 문제 (Where to place)", "생성·재사용한 패치를 정상 이미지의 어느 ROI에 합성할 것인가", ["결함 발생 양상은 배경 표면의 통계·형태 특성과 연관", "호환성 낮은 영역에 합성 → 실제 분포와 괴리", "기존 연구: 원위치 재사용·무작위·수작업 규칙 → 일반화 한계"], C.accent1],
    ].forEach(([h, sub, pts, col], i) => {
      const x = M + i * (colW + 0.4);
      card(s, x, CONTENT_Y, colW, 4.1);
      label(s, h, x + 0.3, CONTENT_Y + 0.3, colW - 0.6, 0.5, { fill: col, fontSize: 14 });
      txt(s, sub, x + 0.3, CONTENT_Y + 0.95, colW - 0.6, 0.6, { bold: true, fontSize: 14 });
      bullets(s, pts, x + 0.3, CONTENT_Y + 1.65, colW - 0.6, 2.3, { fontSize: 13 });
    });
    card(s, M, 5.75, W - 2 * M, 0.95, { fill: { color: C.accent2 } });
    txt(s, "본 논문의 초점: 기존 연구가 P1에 집중한 반면, P2(배치)를 데이터로부터 자동 도출하는 방향으로 고도화하고, 최종적으로 P1과 P2를 하나의 프레임워크로 통합", M + 0.3, 5.85, W - 2 * M - 0.6, 0.75, { color: C.background1, fontSize: 14, valign: "middle" });
  }
  {
    const s = slide("연구 궤적: 4편의 논문이 이루는 단계적 발전", "제1장 서론 · 연구의 발전 서사");
    const steps = [
      ["1", "제3장", "결함 패치 생성", "YOLOv8 + GAN/Diffusion\nCopy-Paste (원위치)", "P1"],
      ["2", "제4장", "ROI 탐색 도입", "ControlNet 생성 + 밝기 유사도 ROI\n+ Poisson blending", "P1 + P2(휴리스틱)"],
      ["3", "제5장 AROMA", "데이터 기반 ROI 자동화", "형태·맥락 호환성 게이트\n5개 도메인 재설계 없이 적용", "P2(데이터 드리븐)"],
      ["4", "제6장 CASDA", "도메인 특화 + 생성 통합", "철강 특화 호환성 + ControlNet\n결함 자체를 새로 생성", "P1 + P2 통합"],
    ];
    const bw = (W - 2 * M - 3 * 0.3) / 4;
    steps.forEach(([n, ch, h, d, tag], i) => {
      const x = M + i * (bw + 0.3), y = CONTENT_Y + 0.2;
      card(s, x, y, bw, 3.6);
      s.addShape(pres.ShapeType.ellipse, { x: x + 0.25, y: y + 0.25, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { color: H.accent1, width: 0 }, objectName: "num" });
      txt(s, n, x + 0.25, y + 0.25, 0.6, 0.6, { color: C.background1, bold: true, fontSize: 18, align: "center", valign: "middle" });
      txt(s, ch, x + 1.0, y + 0.35, bw - 1.2, 0.4, { fontSize: 12, color: C.accent3, bold: true });
      txt(s, h, x + 0.25, y + 1.05, bw - 0.5, 0.8, { bold: true, fontSize: 16 });
      txt(s, d, x + 0.25, y + 1.9, bw - 0.5, 1.0, { fontSize: 12, color: C.text2 });
      label(s, tag, x + 0.25, y + 3.0, bw - 0.5, 0.38, { fill: C.accent2, fontSize: 11 });
      if (i < 3) arrow(s, x + bw + 0.02, y + 1.6, 0.26, 0.35);
    });
    txt(s, "공통 질문: \"결함 후보 영역(ROI)을 어떻게 찾고, 무엇을 그 자리에 배치할 것인가\" — 각 단계는 이전 단계의 한계를 이어받아 범위를 좁히거나 확장", M, 5.5, W - 2 * M, 0.9, { fontSize: 14, bold: true, color: C.text2 });
  }

  // ---------------- II. 관련 연구·환경 ----------------
  section = "II. 관련 연구·실험 환경"; pres.addSection({ title: section });
  {
    const s = slide("관련 연구: 세 갈래 접근과 각각의 한계", "제2장 관련 연구");
    const cols = [
      ["전통적 데이터 증강", ["회전·반전·크기·밝기·대비 변환 [12–14]", "기존 결함의 외형 변형에 의존", "새로운 형태·구조 특성을 반영하기 어려움"]],
      ["생성모델 기반 합성", ["GAN [17] — Defect-GAN 등 결함 합성 [19–21]", "모드 붕괴·학습 불안정으로 다양성 제한", "Diffusion·ControlNet [24, 30] — 조건부 생성으로 제어성 향상"]],
      ["Context-Aware · Copy-Paste", ["Copy-Paste 계열: 패치 재배치, 배치는 무작위가 많음", "일부 연구는 배치 규칙 고려 — 수작업·도메인 종속", "데이터셋 변경 시 규칙 재설계 필요 → 본 논문의 공백"]],
    ];
    const cw = (W - 2 * M - 0.6) / 3;
    cols.forEach(([h, pts], i) => {
      const x = M + i * (cw + 0.3);
      card(s, x, CONTENT_Y, cw, 4.3);
      label(s, h, x + 0.25, CONTENT_Y + 0.25, cw - 0.5, 0.5, { fill: i === 2 ? C.accent1 : C.accent2, fontSize: 13 });
      bullets(s, pts, x + 0.25, CONTENT_Y + 1.0, cw - 0.5, 3.1, { fontSize: 13, gap: 10 });
    });
    txt(s, "평가: 검출 mAP·분할 mIoU/Dice, 합성 품질 FID·KID·LPIPS — 2장 7절", M, 6.0, W - 2 * M, 0.4, { fontSize: 12, color: C.accent3 });
  }
  {
    const s = slide("벤치마크 데이터셋 6종과 공통 실험 환경", "제2장 관련 연구 · 8~9절");
    const rows = [
      ["Dataset", "도메인", "규모(이미지)", "사용 장"],
      ["PCB_DATASET (PKU-HRI)", "PCB", "693", "제3장"],
      ["KolektorSDD", "금속 정류기", "399 (결함 52)", "제4장, 제5장"],
      ["AITeX", "섬유", "7,169", "제5장"],
      ["Severstal", "철강", "5,902", "제5장, 제6장"],
      ["MTD", "자기 타일", "956", "제5장"],
      ["MVTec Leather", "가죽", "245", "제5장"],
    ].map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0, color: i === 0 ? H.lt1 : H.dk1, fill: { color: i === 0 ? H.dk2 : (i % 2 ? H.lt2 : H.lt1) }, align: j === 0 ? "left" : "center", fontSize: 12 } })));
    s.addTable(rows, { x: M, y: CONTENT_Y, w: 8.0, colW: [2.8, 1.6, 1.8, 1.8], rowH: 0.42, border: { type: "solid", color: "DDE2E6", pt: 0.5 }, fontFace: THEME.bodyFontFace, objectName: "dataset-table" });
    card(s, 9.0, CONTENT_Y, W - M - 9.0, 3.0);
    txt(s, "공통 실험 환경", 9.3, CONTENT_Y + 0.25, 3, 0.4, { bold: true, fontSize: 15 });
    bullets(s, ["Google Colab · Ubuntu 20.04", "NVIDIA Tesla L4 GPU", "Python 3.12.8", "PyTorch 2.x (CUDA 11.8)"], 9.3, CONTENT_Y + 0.75, 3.3, 2.0, { fontSize: 13, gap: 8 });
    txt(s, "PCB·전기부품·직물·철강·자기타일·가죽 — 서로 다른 표면 특성에 걸쳐 제안 기법을 평가", M, 5.0, 8.0, 0.8, { fontSize: 13, color: C.text2 });
  }

  // ---------------- III. 제3장 ----------------
  section = "III. 제3장 PCB 결함 패치 증강"; pres.addSection({ title: section });
  {
    const s = slide("제3장: YOLOv8 + GAN / Diffusion 기반 PCB 결함 패치 증강", "제3장 · 방법론");
    await fig(s, "fig3-1.jpg", M, CONTENT_Y, 2.6, 2.6, { caption: "그림 3-1  PCB 6종 결함" });
    txt(s, "open · short · mouse bite · spur · pinhole/missing hole · spurious copper\nPKU-HRI PCB_DATASET 693장 → YOLOv8s/m 실시간 검출", M + 2.9, CONTENT_Y, 4.3, 1.2, { fontSize: 12, color: C.text2 });
    await fig(s, "fig3-3.jpg", M + 2.9, CONTENT_Y + 1.3, 4.3, 1.6, { caption: "그림 3-3  GAN 학습 흐름" });
    await fig(s, "fig3-5.jpg", 8.2, CONTENT_Y, 4.5, 2.2, { caption: "그림 3-5  Diffusion 학습 흐름" });
    await fig(s, "fig3-6.jpg", 8.2, CONTENT_Y + 2.7, 4.5, 1.0, { caption: "그림 3-6  Diffusion 생성 결함 패치" });
    card(s, M, 5.6, W - 2 * M, 1.1, { fill: { color: C.background2 } });
    txt(s, "증강 방식: 생성된 결함 패치를 원본 결함과 동일한 위치에 재배치(Copy-Paste). Basic / GAN / Diffusion 세 조건으로 검출 성능과 일반화 안정성 비교", M + 0.3, 5.7, W - 2 * M - 0.6, 0.9, { fontSize: 14, valign: "middle" });
  }
  {
    const s = slide("3장 결과: Diffusion 증강이 mAP에서 우세, YOLOv8m이 전반적 최고", "제3장 · 실험 결과 (표 3-2)");
    const labels = ["8s Basic", "8s GAN", "8s Diffusion", "8m Basic", "8m GAN", "8m Diffusion"];
    s.addChart(pres.ChartType.bar, [
      { name: "mAP50", labels, values: [0.94, 0.94, 0.96, 0.97, 0.96, 0.96] },
      { name: "mAP50-95", labels, values: [0.49, 0.49, 0.51, 0.54, 0.52, 0.53] },
    ], Object.assign(chartBase("YOLOv8 검출 성능 (mAP50 / mAP50-95)"), { x: M, y: CONTENT_Y, w: 7.6, h: 4.6, barGrouping: "clustered", chartColors: [H.accent2, H.accent1], valAxisMinVal: 0, valAxisMaxVal: 1.0, dataLabelPosition: "outEnd", showLegend: true, legendPos: "b", objectName: "chart-ch3" }));
    card(s, 8.5, CONTENT_Y, W - M - 8.5, 4.6);
    bullets(s, [
      "YOLOv8m Basic: P 0.96 · R 0.96 · mAP50 0.97 · mAP50-95 0.54 — 전 지표 최고",
      "Diffusion 증강: 8s·8m 모두 mAP50 0.96, mAP50-95 0.51/0.53 — GAN 대비 우세",
      "Diffusion 8s: Precision 0.97로 전체 최고 정밀도",
      "GAN 증강: Precision·Recall 소폭 하락, mAP50은 준수(0.94/0.96)",
    ], 8.8, CONTENT_Y + 0.3, W - M - 8.5 - 0.6, 4.0, { fontSize: 13, gap: 10 });
  }
  {
    const s = slide("3장 과적합 분석과 한계: 성능과 일반화의 트레이드오프", "제3장 · 표 3-3 · 한계");
    const labels = ["Basic_s", "Diffusion_s", "Gan_s"];
    s.addChart(pres.ChartType.bar, [
      { name: "Train Loss", labels, values: [1.61, 1.17, 1.20] },
      { name: "Test Loss", labels, values: [1.63, 1.70, 1.72] },
    ], Object.assign(chartBase("YOLOv8s Box Loss — Train vs Test"), { x: M, y: CONTENT_Y, w: 5.6, h: 3.9, barGrouping: "clustered", chartColors: [H.accent3, H.accent1], valAxisMinVal: 0, valAxisMaxVal: 2.0, dataLabelPosition: "outEnd", showLegend: true, legendPos: "b", objectName: "chart-loss" }));
    const gaps = [["Basic_s", "0.02", "최소 과적합", C.accent4], ["Diffusion_s", "0.53", "비교적 양호", C.accent1], ["Gan_s", "0.52", "비교적 양호", C.accent1]];
    gaps.forEach(([n, g, v, col], i) => {
      const x = 6.6 + i * 2.1;
      card(s, x, CONTENT_Y, 1.95, 1.9);
      txt(s, n, x + 0.15, CONTENT_Y + 0.15, 1.7, 0.35, { fontSize: 12, color: C.text2, align: "center" });
      txt(s, g, x + 0.15, CONTENT_Y + 0.5, 1.7, 0.8, { fontSize: 32, bold: true, color: col, align: "center", valign: "middle" });
      txt(s, "Gap · " + v, x + 0.15, CONTENT_Y + 1.35, 1.7, 0.4, { fontSize: 11, align: "center", color: C.text2 });
    });
    bullets(s, ["검출 지표만 보면 Diffusion > GAN, 그러나 두 증강 모델의 Gap 차이는 0.01 — 일반화 우열 단정 불가", "실서비스 관점: 과적합이 근소하게 낮은 GAN 증강이 상대적으로 안정적 선택지"], 6.6, CONTENT_Y + 2.1, 6.1, 1.6, { fontSize: 12, gap: 6 });
    transition(s, "한계: 증강 패치를 원본과 \"동일한 위치\"에 재배치 → 배경과의 통계·형태 호환성(Where)을 고려하지 않음", "제4장: 배경 밝기 유사도 기반 ROI 탐색으로 \"어디에 배치할 것인가\"를 처음 다룸");
  }

  // ---------------- IV. 제4장 ----------------
  section = "IV. 제4장 ControlNet + ROI + Poisson"; pres.addSection({ title: section });
  {
    const s = slide("제4장: ControlNet 결함 생성 → 유효 영역 ROI 탐색 → Poisson blending", "제4장 · 방법론 (KolektorSDD, 결함 52/399장)");
    const steps = [["fig4-2.jpg", "① ControlNet 결함·마스크 생성", "마스크 조건으로 결함 패치 생성 (그림 4-2)"], ["fig4-3.jpg", "② 유효 영역(ROI) 추출", "결함 주변 픽셀 평균 밝기와 유사한 영역을 슬라이딩 윈도우로 탐색 (그림 4-3)"], ["fig4-4.jpg", "③ Poisson blending 합성", "경계 불연속 제거, 시각적 이질감 완화 (그림 4-4)"]];
    const cw = (W - 2 * M - 0.8) / 3;
    for (let i = 0; i < 3; i++) {
      const x = M + i * (cw + 0.4);
      card(s, x, CONTENT_Y, cw, 4.3);
      await fig(s, steps[i][0], x + 0.2, CONTENT_Y + 0.2, cw - 0.4, 2.3);
      txt(s, steps[i][1], x + 0.2, CONTENT_Y + 2.65, cw - 0.4, 0.45, { bold: true, fontSize: 14 });
      txt(s, steps[i][2], x + 0.2, CONTENT_Y + 3.1, cw - 0.4, 1.0, { fontSize: 12, color: C.text2 });
      if (i < 2) arrow(s, x + cw + 0.07, CONTENT_Y + 1.9, 0.26, 0.35);
    }
    txt(s, "학습: ResNet34 · EfficientNet 분할 모델, 원본만(Real) vs 증강 비율 0.3 / 0.5 / 0.7 혼합(Mixed), 8:2 분할", M, 6.0, W - 2 * M, 0.5, { fontSize: 13, color: C.text2 });
  }
  {
    const s = slide("4장 결과: 증강 혼합 시 mIoU 0.567 → 0.892 (ResNet34, +57.4%)", "제4장 · 실험 결과 (표 4-4)");
    const labels = ["Real", "0.3", "0.5", "0.7"];
    s.addChart(pres.ChartType.line, [
      { name: "ResNet34 mIoU", labels, values: [0.5667, 0.8069, 0.8919, 0.8071] },
      { name: "EfficientNet mIoU", labels, values: [0.1682, 0.5378, 0.5347, 0.6860] },
      { name: "ResNet34 Dice", labels, values: [0.6766, 0.8799, 0.9400, 0.8813] },
      { name: "EfficientNet Dice", labels, values: [0.2531, 0.6070, 0.5862, 0.7770] },
    ], Object.assign(chartBase("증강 비율별 mIoU / Dice"), { x: M, y: CONTENT_Y, w: 7.4, h: 4.7, chartColors: [H.accent1, H.accent2, "F3A86B", H.accent3], lineSize: 2, lineDataSymbolSize: 7, valAxisMinVal: 0, valAxisMaxVal: 1.0, dataLabelPosition: "t", dataLabelFormatCode: "0.000", showLegend: true, legendPos: "b", objectName: "chart-ch4" }));
    card(s, 8.3, CONTENT_Y, W - M - 8.3, 4.7);
    bullets(s, [
      "ResNet34: 비율 0.5에서 mIoU 0.8919 최고, 0.7에서 0.8071로 하락 — 과도한 합성은 실제 분포를 왜곡",
      "EfficientNet: 0.3·0.5 정체 후 0.7에서 0.6860 최고 — 경량 모델은 더 많은 합성 데이터가 유효",
      "Dice: ResNet34 0.5 모델 0.94, EfficientNet 0.2531 → 0.777 (약 3배)",
      "두 모델 모두 증강셋에서 원본 대비 비약적 향상 → ROI 방식이 실제 결함 특성을 묘사",
    ], 8.6, CONTENT_Y + 0.3, W - M - 8.3 - 0.6, 4.1, { fontSize: 12, gap: 8 });
  }
  {
    const s = slide("4장 한계: 밝기라는 단일 통계와 휴리스틱 탐색", "제4장 · 한계 및 다음 단계");
    const cw = (W - 2 * M - 0.4) / 2;
    card(s, M, CONTENT_Y, cw, 3.7);
    label(s, "제4장이 해결한 것", M + 0.3, CONTENT_Y + 0.3, cw - 0.6, 0.45, { fill: C.accent4, fontSize: 13 });
    bullets(s, ["결함 생성(P1)에 더해 배치 위치(P2)를 처음으로 탐색", "Poisson blending으로 합성 경계 이질감 완화", "결함 희소 환경(52장)에서 분할 성능 대폭 개선"], M + 0.3, CONTENT_Y + 0.95, cw - 0.6, 2.6, { fontSize: 13, gap: 10 });
    card(s, M + cw + 0.4, CONTENT_Y, cw, 3.7);
    label(s, "남은 한계", M + cw + 0.7, CONTENT_Y + 0.3, cw - 0.6, 0.45, { fill: C.accent5, fontSize: 13 });
    bullets(s, ["ROI 추출이 결함 주변 평균 밝기 하나에만 의존", "배경 텍스처·방향성, 결함의 구조적 특성을 반영하지 못함", "사전 정의된 밝기 유사도 규칙 → 데이터셋이 바뀌면 재조정 필요"], M + cw + 0.7, CONTENT_Y + 0.95, cw - 0.6, 2.6, { fontSize: 13, gap: 10 });
    transition(s, "한계: 수작업 규칙·단일 통계량에 기반한 휴리스틱 ROI 탐색", "제5장 AROMA: 배경·결함 특성을 다차원 프로파일링하고, 배치 기준을 데이터로부터 자동 도출");
  }

  // ---------------- V. 제5장 AROMA ----------------
  section = "V. 제5장 AROMA"; pres.addSection({ title: section });
  {
    const s = slide("제5장 AROMA: 적응형 ROI 기반 형태 인지 증강", "제5장 · 개요 (Adaptive ROI-based Morphology-Aware Augmentation)");
    await fig(s, "fig5-1.png", M, CONTENT_Y, 5.6, 5.0, { caption: "그림 5-1  AROMA 파이프라인" });
    card(s, 6.6, CONTENT_Y, W - M - 6.6, 2.4, { fill: { color: C.accent2 } });
    txt(s, "데이터 드리븐", 6.9, CONTENT_Y + 0.2, 5.5, 0.4, { bold: true, fontSize: 13, color: C.accent6 });
    txt(s, "결함과 배경 간 호환성을 패치 수준의 공존 통계로부터 직접 산출하는 대칭 호환성 게이트. 도메인별 수작업 규칙 없이 동일 파이프라인을 5개 산업 데이터셋에 적용", 6.9, CONTENT_Y + 0.6, 5.5, 1.7, { fontSize: 14, color: C.background1 });
    bullets(s, [
      "프로파일링: 형태 복잡도 MCI · 맥락 복잡도 CCI 산출",
      "이산화: 결함 → 형태 클러스터 k, 배경 → 맥락 셀 c",
      "호환성: ctx_prior(k, c) — 결함측·배경측 기하평균",
      "선택 3단: 결함(ROI) → 배경 이미지 → 삽입 위치",
      "평가: AITeX · Kolektor · Severstal · MTD · MVTec Leather, YOLOv8n × 3 seeds",
    ], 6.6, CONTENT_Y + 2.7, W - M - 6.6, 2.6, { fontSize: 13, gap: 8 });
  }
  {
    const s = slide("프로파일링: 결함 형태(MCI)와 배경 텍스처(CCI)를 같은 방식으로 데이터에서 도출", "제5장 · 방법론 1 — 수식 (1)~(5), 표 5-1");
    const bw2 = (W - 2 * M - 0.3) / 2, bh2 = 3.05;
    ruleBlock(s, M, CONTENT_Y, bw2, bh2, {
      tag: "결함 형태 · MCI", title: "형태 클러스터 k — 자동 군집화",
      features: "linearity · solidity · extent · aspect ratio · eccentricity · circularity (SAM 마스크, 6개)",
      criterion: "GMM 군집화, 군집 수는 데이터셋별 BIC로 자동 결정 — 수동 임계값 없음",
      lines: ["K* = argmax_k  BIC(k)", "k ∈ {0, …, K*−1},   morph_prior(k) = P(k) = n_k / N"],
      result: "MCI = 강도 변화·분포 다양성·유형 다양성·클러스터 분리도 4개 정규화 통계량의 평균",
    });
    ruleBlock(s, M + bw2 + 0.3, CONTENT_Y, bw2, bh2, {
      tag: "배경 텍스처 · CCI", title: "텍스처 범주 — 백분위수 자동 적응", tagColor: C.accent4,
      features: "LocalVariance · OrientEntropy · FreqComplexity · TextureEntropy (정상 패치, 64px 그리드)",
      criterion: "데이터셋별 자동 계산된 백분위수 P_k로 5개 범주 분류 — 동일 알고리즘이 모든 데이터셋에 적응",
      lines: ["Smooth      = LocalVariance ≤ P₂₅", "Directional = (OrientEntropy ≤ P₂₅) ∧ (FreqComplexity ≤ P₂₅)", "Periodic    = FreqComplexity ≥ P₇₅", "Organic     = (TextureEntropy ≥ P₅₀) ∧ (LocalVariance ≥ P₇₅)", "Complex     = otherwise"],
      mathSize: 10.5,
      result: "CCI = 텍스처 엔트로피·주파수 복잡도·방향성 분산 3개 통계량의 평균 (2만 패치 표본)",
    });
    const yb = CONTENT_Y + bh2 + 0.25;
    const rows = [["Dataset", "MCI", "CCI"], ["AITeX", "0.392", "0.440"], ["Kolektor", "0.328", "0.224"], ["Severstal", "0.455", "0.306"], ["MTD", "0.561", "0.246"], ["MVTec Leather", "0.595", "0.200"]]
      .map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || (j === 2 && i > 0), color: i === 0 ? H.lt1 : (j === 2 ? H.accent1 : H.dk1), fill: { color: i === 0 ? H.dk2 : (i % 2 ? H.lt2 : H.lt1) }, align: j === 0 ? "left" : "center", fontSize: 10 } })));
    s.addTable(rows, { x: M, y: yb, w: 4.6, colW: [2.2, 1.2, 1.2], rowH: 0.28, border: { type: "solid", color: "DDE2E6", pt: 0.5 }, fontFace: THEME.bodyFontFace, objectName: "mci-cci" });
    txt(s, "MCI 0.33~0.60, CCI 0.20~0.44 — 다섯 데이터셋이 서로 다른 난이도 영역에 위치. CCI 축은 5.3절에서 context-aware 배치 효과를 조건화하는 핵심 지표", M + 4.9, yb, 3.4, 1.9, { fontSize: 11, color: C.text2 });
    await fig(s, "fig5-2.png", 8.6, yb - 0.05, W - M - 8.6, 2.0, { caption: "그림 5-2  복잡도 지형 (MCI vs CCI)" });
  }
  {
    const s = slide("이산화: 결함은 형태 클러스터 k로, 배경은 맥락 셀 c로", "제5장 · 방법론 2 — 신규 도식 ④-a: 맥락 셀 키 판독");
    await fig(s, "fig5-3.png", M, CONTENT_Y, 6.0, 2.3, { caption: "그림 5-3  데이터 기반 형태 클러스터 (GMM + BIC)" });
    await fig(s, "fig5-4a.png", M, CONTENT_Y + 2.75, 6.0, 2.3, { caption: "그림 5-4(a)  배경 맥락 특성 분포와 tertile(P33/P66) 경계 — Severstal" });
    // 도식: cell key
    const x0 = 7.1, y0 = CONTENT_Y;
    card(s, x0, y0, W - M - x0, 5.1);
    txt(s, "맥락 셀 c = 5개 텍스처 특성의 tertile 조합 키", x0 + 0.25, y0 + 0.2, 5.2, 0.4, { bold: true, fontSize: 14 });
    const feats = ["local\nvariance", "edge\ndensity", "texture\nentropy", "frequency\nenergy", "orientation\nconsistency"];
    const digits = ["2", "1", "1", "1", "1"];
    const bw = 0.82, gap = 0.16, bx = x0 + 0.35, by = y0 + 0.75;
    digits.forEach((d, i) => {
      const x = bx + i * (bw + gap);
      s.addShape(pres.ShapeType.roundRect, { x, y: by, w: bw, h: 0.8, rectRadius: 0.08, fill: { color: d === "2" ? C.accent1 : d === "1" ? C.accent2 : C.accent3 }, line: { color: H.lt1, width: 0 }, objectName: "digit" });
      txt(s, d, x, by, bw, 0.8, { color: C.background1, bold: true, fontSize: 26, align: "center", valign: "middle" });
      txt(s, feats[i], x - 0.1, by + 0.9, bw + 0.2, 0.6, { fontSize: 9, color: C.text2, align: "center" });
      if (i < 4) txt(s, "_", x + bw, by, gap, 0.8, { fontSize: 20, bold: true, align: "center", valign: "middle", color: C.text2 });
    });
    txt(s, "각 자리:  0 = 데이터셋 하위 ⅓   1 = 중간 ⅓   2 = 상위 ⅓  (P33 / P66)", x0 + 0.25, by + 1.6, 5.2, 0.4, { fontSize: 12 });
    bullets(s, [
      "\"2_1_1_1_1\" = 분산이 높고 나머지는 중간인 표면 유형 하나",
      "최대 3⁵ = 243 종, 실제 관측 셀은 데이터셋당 100~200 종",
      "등급 기준은 데이터셋 자신 → 같은 절차라도 셀 경계는 데이터셋마다 다름 (그림 5-4)",
      "셀은 범주형: 순서·이웃 없음, 한 자리만 달라도 다른 셀",
      "결함 k: 6개 형태 특성(linearity·solidity·extent·AR·eccentricity·circularity) → GMM, 개수는 BIC",
    ], x0 + 0.25, by + 2.1, W - M - x0 - 0.5, 2.3, { fontSize: 11, gap: 5 });
  }
  {
    const s = slide("ctx_prior 유도: 결함측 선호 AND 배경측 가용성의 기하평균", "제5장 · 방법론 3 — 신규 도식 ②: 인구조사 2번 → √곱 → 정규화");
    card(s, M, CONTENT_Y, 7.4, 2.05, { fill: { color: C.accent2 } });
    math(s, "ctx_prior(k, c)  ∝  √( P_def(k, c) · P_clean(c) ),    행별 max = 1", M + 0.3, CONTENT_Y + 0.15, 6.9, 0.55, { fontSize: 16, color: C.background1, bold: true });
    txt(s, [
      { text: "P_def(k,c)", options: { bold: true, color: H.accent6 } }, { text: " : 형태 클러스터 k 결함의 주변 배경 패치 중 셀 c의 비율 (결함 겹침 타일 제외)", options: { breakLine: true } },
      { text: "P_clean(c)", options: { bold: true, color: H.accent6 } }, { text: " : 전체 정상(clean) 배경 패치 중 셀 c의 비율 — 모든 k에 공용 1행", options: { breakLine: true } },
      { text: "기하평균", options: { bold: true, color: H.accent6 } }, { text: " : 어느 한쪽만 높은 셀이 과도하게 선택되는 것을 방지 → \"대칭\" 호환성", options: {} },
    ], M + 0.3, CONTENT_Y + 0.75, 6.9, 1.25, { fontSize: 12, color: C.background1 });
    // 4단계 도식 (막대)
    const cells = ["A", "B", "C", "D"];
    const pdef = [0.40, 0.20, 0.10, 0.30], pcl = [0.30, 0.40, 0.00, 0.30];
    const raw = pdef.map((v, i) => Math.sqrt(v * pcl[i]));
    const mx = Math.max(...raw), rowmax = raw.map(v => v / mx);
    const sum = raw.reduce((a, b) => a + b, 0), tgt = raw.map(v => v / sum);
    const panels = [["① 인구조사: P_def(k,·)", pdef, C.accent2], ["① 인구조사: P_clean", pcl, C.accent3], ["② √곱 (AND) → raw", raw, C.accent1], ["③ 행 max=1 (히트맵 저장)", rowmax, C.accent1], ["④ 합=1 → tgt[k]", tgt, C.accent4]];
    const py = CONTENT_Y + 2.3, ph = 2.75, pw = (W - 2 * M - 4 * 0.18) / 5;
    panels.forEach(([t, vals, col], pi) => {
      const px = M + pi * (pw + 0.18);
      card(s, px, py, pw, ph);
      txt(s, t, px + 0.1, py + 0.08, pw - 0.2, 0.4, { fontSize: 10, bold: true, align: "center" });
      const base = py + ph - 0.45, maxH = 1.6;
      const vmax = Math.max(...vals, 0.01);
      vals.forEach((v, i) => {
        const bwid = (pw - 0.5) / 4, bx = px + 0.25 + i * bwid;
        const h = v / vmax * maxH;
        if (v > 0) s.addShape(pres.ShapeType.rect, { x: bx + 0.08, y: base - h, w: bwid - 0.16, h, fill: { color: col }, line: { color: H.lt1, width: 0 }, objectName: "bar" });
        txt(s, v === 0 ? "0" : v.toFixed(2), bx, base - h - 0.28, bwid, 0.25, { fontSize: 8, align: "center", color: C.text2 });
        txt(s, cells[i], bx, base + 0.03, bwid, 0.25, { fontSize: 10, bold: true, align: "center" });
      });
      if (pi < 4) arrow(s, px + pw + 0.01, py + ph / 2 - 0.12, 0.16, 0.24);
    });
    txt(s, "C: 결함은 그 옆에 살지만 clean 배경에 없음 → 자격 박탈 (0).  ③과 ④는 같은 모양, 눈금만 다름(최적 대비 % → 전체 중 비율).  tgt[k]는 이후 배경·위치 선택의 과녁.", M, py + ph + 0.08, W - 2 * M, 0.45, { fontSize: 11, color: C.text2 });
    await fig(s, "fig5-6.png", 8.3, CONTENT_Y, W - M - 8.3, 2.05, { caption: "그림 5-6  ctx_prior 히트맵 (행 peak = 1)" });
  }
  {
    const s = slide("ROI 선발: ROI_score = ctx_prior + morph_prior 로 순위화, 쿼터로 편중 방지", "제5장 · 방법론 4 — 그림 5-7");
    await fig(s, "fig5-7.png", M, CONTENT_Y, 4.4, 5.3, { caption: "그림 5-7  ROI 선택 및 호환성 기반 배치 흐름" });
    const x0 = 5.4;
    card(s, x0, CONTENT_Y, W - M - x0, 1.5, { fill: { color: C.accent2 } });
    math(s, "ROI_score = ctx_prior(k, c_src) + morph_prior(k)", x0 + 0.3, CONTENT_Y + 0.15, 6.8, 0.55, { fontSize: 17, color: C.background1, bold: true });
    txt(s, "ctx_prior: 결함 자신의 소스 맥락 셀에서 읽은 호환성 스칼라   ·   morph_prior = P(k): 클러스터 크기 비율. 무가중 합으로 후보 결함–배경 쌍을 순위화, 상위 K개 채택", x0 + 0.3, CONTENT_Y + 0.75, 6.8, 0.7, { fontSize: 11, color: C.background1 });
    const rules = [["형태 클러스터별 기본 쿼터", "특정 형태 클러스터에 선택이 몰리지 않도록 (k, c) 쌍 단위로 배분 → 희귀 쌍 커버리지의 원천"], ["클래스별 최소 하한선", "다중 클래스 데이터셋에서 과소표집 클래스의 기아 방지 (예: Severstal c2 균형 배분)"], ["소스 크롭 최대 1회 선택", "동일 결함 크롭의 반복 사용으로 인한 다양성 붕괴 제거"], ["품질 게이트", "블러·대비·밝기·노이즈 가중합 — 하위 품질 배경 패치는 합성에서 배제"]];
    rules.forEach(([h, d], i) => {
      const y = CONTENT_Y + 1.75 + i * 0.92;
      card(s, x0, y, W - M - x0, 0.8);
      s.addShape(pres.ShapeType.ellipse, { x: x0 + 0.2, y: y + 0.2, w: 0.4, h: 0.4, fill: { color: C.accent1 }, line: { color: H.accent1, width: 0 }, objectName: "dot" });
      txt(s, h, x0 + 0.75, y + 0.08, 6.4, 0.32, { bold: true, fontSize: 12 });
      txt(s, d, x0 + 0.75, y + 0.4, 6.4, 0.38, { fontSize: 10, color: C.text2 });
    });
  }
  {
    const s = slide("배경 이미지 선택: bg_score = src_fit + class_fit + size_fit", "제5장 · 방법론 5 — 신규 도식 ⑤: 집계 반경이 넓어지는 세 적합도");
    card(s, M, CONTENT_Y, 6.6, 1.3, { fill: { color: C.accent2 } });
    math(s, "∩(a, b) = Σ_c min(a(c), b(c))      bg_score(g) = src_fit + class_fit + size_fit", M + 0.3, CONTENT_Y + 0.12, 6.1, 0.5, { fontSize: 13, color: C.background1, bold: true });
    txt(s, "모든 적합도는 후보 배경 g의 맥락 셀 히스토그램 h_g와 목표 분포의 히스토그램 교차(겹친 질량). 가장 높은 정상 이미지를 배경으로 배정", M + 0.3, CONTENT_Y + 0.65, 6.1, 0.6, { fontSize: 11, color: C.background1 });
    // 동심원 도식
    const cx = 2.6, cy = 4.75;
    const rings = [[2.9, C.accent3, "class_fit", "클래스 단위 — 같은 클래스 결함들 주변 배경의 집계 분포와 닮았나"], [1.9, C.accent2, "src_fit", "개체 단위 — 이 결함이 잘려나온 바로 그 원본 이미지의 배경 분포와 닮았나"]];
    rings.forEach(([d, col, n]) => s.addShape(pres.ShapeType.ellipse, { x: cx - d / 2, y: cy - d / 2, w: d, h: d, fill: { color: col, transparency: 55 }, line: { color: H.lt1, width: 1.5 }, objectName: n }));
    s.addShape(pres.ShapeType.ellipse, { x: cx - 0.45, y: cy - 0.45, w: 0.9, h: 0.9, fill: { color: C.accent1 }, line: { color: H.lt1, width: 1.5 }, objectName: "defect" });
    txt(s, "결함", cx - 0.45, cy - 0.45, 0.9, 0.9, { color: C.background1, bold: true, fontSize: 12, align: "center", valign: "middle" });
    txt(s, "src_fit", cx + 0.5, cy - 0.85, 0.9, 0.3, { bold: true, fontSize: 11, color: C.text1 });
    txt(s, "class_fit", cx + 0.95, cy - 1.4, 1.2, 0.3, { bold: true, fontSize: 11, color: C.text1 });
    const ex = 4.4;
    [["src_fit", "개체 단위", "이 결함이 잘려나온 그 원본 이미지의 배경 분포 p_src와의 교차 — \"결함이 원래 살던 표면과 닮았나\"", C.accent2],
     ["class_fit", "클래스 단위", "같은 클래스 결함 주변 배경의 집계 분포 p_cls와의 교차 — \"이 클래스 결함들의 표면과 닮았나\"", C.accent3],
     ["size_fit", "기하", "min(1, 0.95·W_g/w, 0.95·H_g/h) — 왜곡 없이 들어가는 정도 (1 = 무축소)", C.accent4]].forEach(([n, u, d, col], i) => {
      const y = CONTENT_Y + 1.5 + i * 1.3;
      card(s, ex, y, 7.6 - ex - 0.2, 1.2);
      label(s, n, ex + 0.15, y + 0.39, 1.0, 0.42, { fill: col, fontSize: 10 });
      txt(s, u, ex + 1.3, y + 0.1, 1.5, 0.3, { bold: true, fontSize: 11 });
      txt(s, d, ex + 1.3, y + 0.4, 7.6 - ex - 0.2 - 1.45, 0.78, { fontSize: 9.5, color: C.text2 });
    });
    await fig(s, "fig5-8.png", 7.6, CONTENT_Y, W - M - 7.6, 5.2, { caption: "그림 5-8  실제 이미지에서의 배경 할당" });
  }
  {
    const s = slide("삽입 위치 선택 ①: 핀 s · 스티커 F(s) · 둘레 R(s)", "제5장 · 방법론 6 — 신규 도식 ①: 64px 타일 격자 위의 후보 위치");
    // grid diagram
    const gx = M + 0.2, gy = CONTENT_Y + 0.3, t = 0.62, cols = 8, rows = 5;
    const F = new Set(["2,1", "3,1", "2,2", "3,2"]);
    const R = new Set(["1,0", "2,0", "3,0", "4,0", "1,1", "4,1", "1,2", "4,2", "1,3", "2,3", "3,3", "4,3"]);
    const V = new Set(["7,0", "7,1", "7,2", "7,3", "7,4"]);
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      const k = `${i},${j}`;
      const fill = F.has(k) ? C.accent1 : R.has(k) ? C.accent2 : V.has(k) ? C.text2 : C.background2;
      s.addShape(pres.ShapeType.rect, { x: gx + i * t, y: gy + j * t, w: t, h: t, fill: { color: fill, transparency: F.has(k) ? 0 : R.has(k) ? 25 : 0 }, line: { color: H.lt1, width: 1 }, objectName: "tile" });
      const lab = F.has(k) ? "F" : R.has(k) ? "R" : V.has(k) ? "void" : "";
      if (lab) txt(s, lab, gx + i * t, gy + j * t, t, t, { color: C.background1, bold: lab !== "void", fontSize: lab === "void" ? 7 : 12, align: "center", valign: "middle" });
    }
    // pin s
    s.addShape(pres.ShapeType.ellipse, { x: gx + 2 * t - 0.14, y: gy + 1 * t - 0.14, w: 0.28, h: 0.28, fill: { color: C.accent5 }, line: { color: H.lt1, width: 1.5 }, objectName: "pin" });
    txt(s, "s = (2, 1)", gx + 2 * t - 0.9, gy + 1 * t - 0.55, 1.2, 0.3, { fontSize: 11, bold: true, color: C.accent5 });
    txt(s, "64 px", gx, gy + rows * t + 0.05, t * 2, 0.3, { fontSize: 9, color: C.accent3 });
    const lx = gx + cols * t + 0.5;
    const legend = [["s", "핀 꽂는 점 — 타일 오프셋 (si, sj), 저장·소비되는 유일한 값", C.accent5], ["F(s)", "핀이 덮는 스티커 면적 — bw×bh 타일. 결함이 덮어 지우므로 존재(관측·비-void)만 확인", C.accent1], ["R(s)", "스티커 둘레 — 테두리 밴드, 합성 후 살아남아 결함과 접하는 곳. 셀 구성을 센다", C.accent2], ["void", "분산·엣지가 P15 미만인 타일 — 격자에서 제외", C.text2]];
    legend.forEach(([n, d, col], i) => {
      const y = CONTENT_Y + i * 0.78;
      s.addShape(pres.ShapeType.roundRect, { x: lx, y: y + 0.08, w: 0.75, h: 0.42, rectRadius: 0.06, fill: { color: col }, line: { color: H.lt1, width: 0 }, objectName: "lg" });
      txt(s, n, lx, y + 0.08, 0.75, 0.42, { color: C.background1, bold: true, fontSize: 12, align: "center", valign: "middle" });
      txt(s, d, lx + 0.9, y, W - M - lx - 0.9, 0.7, { fontSize: 11, color: C.text2, valign: "middle" });
    });
    card(s, M, 5.05, W - 2 * M, 1.65, { fill: { color: C.accent2 } });
    math(s, "valid(s)  ⟺  ∀ t ∈ F(s) :  observed(t) ∧ ¬void(t)          bw = ⌈ew / 64⌉,  bh = ⌈eh / 64⌉", M + 0.3, 5.15, W - 2 * M - 0.6, 0.5, { fontSize: 14, color: C.background1, bold: true });
    txt(s, "① 격자 정의(void 제외) → ② 후보 s를 (0,0)부터 타일 단위 래스터 스캔으로 전수 열거 → ③ F(s)가 격자 안·비-void인 자리만 자격 통과 → ④ 생존 자리 중 둘레 R(s)로 채점. 도장(F)의 크기·비율은 결함 bbox 자체가 정하고, 배경 크기에 맞춰 등비 축소(fit-rescale) 후 타일 올림", M + 0.3, 5.65, W - 2 * M - 0.6, 1.0, { fontSize: 11, color: C.background1 });
  }
  {
    const s = slide("삽입 위치 선택 ②: site_score = ∩(h_s, tgt[k])", "제5장 · 방법론 7 — 신규 도식 ③: 히스토그램 교차의 min-겹침");
    card(s, M, CONTENT_Y, 6.9, 1.35, { fill: { color: C.accent2 } });
    math(s, [
      { text: "h_s(c) = R(s) 타일들의 맥락 셀 정규화 히스토그램 (합 = 1)", options: { breakLine: true, fontSize: 13 } },
      { text: "site_score(k, s) = ∩(h_s, tgt[k]) = Σ_c min( h_s[c], tgt[k][c] ),     s* = argmax_{valid s} site_score", options: { fontSize: 14, bold: true } },
    ], M + 0.3, CONTENT_Y + 0.12, 6.4, 1.1, { color: C.background1 });
    // min overlap chart
    const cells = ["A", "B", "D", "C"];
    const tg = [0.373, 0.305, 0.323, 0], hs = [0.50, 0.17, 0.33, 0];
    const mn = tg.map((v, i) => Math.min(v, hs[i]));
    const cx0 = M, cy0 = CONTENT_Y + 1.6, cw = 6.9, ch = 3.4;
    card(s, cx0, cy0, cw, ch);
    txt(s, "예: 둘레 6타일 = {A,A,A,B,D,D} → h_s = {A .50, B .17, D .33}", cx0 + 0.2, cy0 + 0.1, cw - 0.4, 0.35, { fontSize: 11, bold: true });
    const base = cy0 + ch - 0.55, maxH = 1.9, grpW = (cw - 0.8) / 4;
    cells.forEach((cname, i) => {
      const gx0 = cx0 + 0.4 + i * grpW, bwid = (grpW - 0.3) / 3;
      [[tg[i], C.accent3, "tgt"], [hs[i], C.accent2, "h_s"], [mn[i], C.accent1, "min"]].forEach(([v, col, nm], j) => {
        const h = v / 0.5 * maxH, bx = gx0 + 0.15 + j * bwid;
        if (v > 0) s.addShape(pres.ShapeType.rect, { x: bx, y: base - h, w: bwid - 0.06, h, fill: { color: col }, line: { color: H.lt1, width: 0 }, objectName: "bar" });
        txt(s, v === 0 ? "0" : v.toFixed(2), bx - 0.05, base - h - 0.26, bwid + 0.04, 0.24, { fontSize: 8, align: "center", color: C.text2 });
        if (i === 0) txt(s, nm, cx0 + 0.3 + j * 0.6, cy0 + 0.5, 0.6, 0.25, { fontSize: 9, align: "left", color: col, bold: true });
      });
      txt(s, cname, gx0, base + 0.02, grpW - 0.3, 0.25, { fontSize: 12, bold: true, align: "center" });
    });
    txt(s, "score = .373 + .17 + .323 = 0.866", cx0 + 3.6, cy0 + 0.5, 3.1, 0.35, { fontSize: 12, bold: true, color: C.accent1, align: "right" });
    txt(s, "A: h_s 초과분 .127 무기여 ·  B: 목표 미달분 그대로 감점 ·  C: 둘레에 없는 셀 = 0", cx0 + 0.2, cy0 + ch - 0.22 + 0.02, cw - 0.4, 0.25, { fontSize: 9, color: C.text2 });
    await fig(s, "fig5-9.png", 7.8, CONTENT_Y, W - M - 7.8, 3.5, { caption: "그림 5-9  원본 결함의 ring 맥락과 확정된 배치 위치 (tgt[k] 질량으로 착색)" });
    bullets(s, ["왜 ring인가: footprint는 결함이 덮어 사라지는 픽셀. 합성 후 남아 경계 연속성을 결정하는 것은 둘레", "배치는 기하 휴리스틱이나 무작위가 아니라, 그 결함 형태가 실제 데이터에서 관측된 맥락과 주변이 가장 유사한 위치"], 7.8, 5.55, W - M - 7.8, 1.3, { fontSize: 11, gap: 4 });
  }
  {
    const s = slide("왜 min-겹침인가: 평균은 흔한 셀 몰빵을, 분포 매칭은 레시피를 고른다", "제5장 · 방법론 8 — 신규 도식 ④-b: 지도 1개 · 질의 N개");
    // 비교 예시
    const cw = 6.4;
    card(s, M, CONTENT_Y, cw, 2.75);
    txt(s, "tgt[k] = {A .4, B .3, D .3}  두 자리 비교", M + 0.25, CONTENT_Y + 0.12, cw - 0.5, 0.35, { bold: true, fontSize: 12 });
    const rows = [["", "자리1 (몰빵)  h_s={A 1.0}", "자리2 (배합)  h_s={A .4, B .3, D .3}"], ["타일 점수 평균", "0.40  ← 승리(!)", "0.34"], ["min-겹침 합", "0.40", "1.00  ← 승리"]]
      .map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, color: i === 0 ? H.lt1 : H.dk1, fill: { color: i === 0 ? H.dk2 : (i === 2 && j === 2) || (i === 1 && j === 1) ? "FDE9D9" : H.lt1 }, align: j === 0 ? "left" : "center", fontSize: 10 } })));
    s.addTable(rows, { x: M + 0.25, y: CONTENT_Y + 0.55, w: cw - 0.5, colW: [1.5, 2.1, 2.3], rowH: 0.42, border: { type: "solid", color: "DDE2E6", pt: 0.5 }, fontFace: THEME.bodyFontFace, objectName: "cmp" });
    txt(s, "평균은 최고 점수 셀 하나로 채운 자리가 이긴다 → 전 클러스터가 데이터셋에서 가장 흔한 표면으로 수렴. min은 초과분을 목표값에서 잘라 배합을 재현해야만 만점", M + 0.25, CONTENT_Y + 1.9, cw - 0.5, 0.8, { fontSize: 10, color: C.text2 });
    card(s, M, CONTENT_Y + 2.95, cw, 2.35, { fill: { color: C.accent2 } });
    txt(s, "실측 근거 (기여 분리)", M + 0.25, CONTENT_Y + 3.05, cw - 0.5, 0.35, { bold: true, fontSize: 12, color: C.accent6 });
    const ev = [["footprint → ring, 평균 유지", "악화 4/5", C.accent5], ["평균 → min-겹침 (분포 매칭)", "개선 5/5", C.accent4]];
    ev.forEach(([n, r, col], i) => {
      const y = CONTENT_Y + 3.45 + i * 0.6;
      txt(s, n, M + 0.25, y, 3.6, 0.5, { fontSize: 12, color: C.background1, valign: "middle" });
      label(s, r, M + 4.0, y + 0.05, 2.1, 0.4, { fill: col, fontSize: 12 });
    });
    math(s, "∩(a, b) = 1 − TV(a, b)   — 교차 최대화 ≡ total variation 거리 최소화", M + 0.25, CONTENT_Y + 4.7, cw - 0.5, 0.5, { fontSize: 11, color: C.accent6 });
    // 사다리 도식
    const lx = 7.5, lw = W - M - lx;
    card(s, lx, CONTENT_Y, lw, 5.3);
    txt(s, "단위 사다리 — 지도는 하나, 질의는 여럿", lx + 0.25, CONTENT_Y + 0.15, lw - 0.5, 0.35, { bold: true, fontSize: 13 });
    const ladder = [["데이터셋", "ctx_prior 히트맵 (matrix)", "1개", C.text2], ["클러스터 k", "tgt[k] — 행을 합=1로", "k 수만큼", C.accent1], ["이미지 g", "h_g → src/class_fit", "good 1장당 1", C.accent2], ["자리 s", "h_s → site_score", "자리당 1", C.accent4]];
    ladder.forEach(([u, w, n, col], i) => {
      const y = CONTENT_Y + 0.65 + i * 0.98, ww = lw - 0.5 - i * 0.35;
      s.addShape(pres.ShapeType.roundRect, { x: lx + 0.25 + i * 0.35, y, w: ww, h: 0.78, rectRadius: 0.08, fill: { color: col }, line: { color: H.lt1, width: 0 }, objectName: "ladder" });
      txt(s, u, lx + 0.4 + i * 0.35, y + 0.08, 1.6, 0.35, { bold: true, fontSize: 12, color: C.background1 });
      txt(s, w, lx + 0.4 + i * 0.35, y + 0.42, ww - 1.6, 0.38, { fontSize: 10, color: C.background1 });
      txt(s, n, lx + 0.25 + i * 0.35 + ww - 1.5, y + 0.08, 1.35, 0.35, { fontSize: 10, color: C.background1, align: "right" });
      if (i < 3) s.addShape(pres.ShapeType.downArrow, { x: lx + lw / 2 - 0.12, y: y + 0.79, w: 0.24, h: 0.18, fill: { color: C.accent3 }, line: { color: H.accent3, width: 0 }, objectName: "down" });
    });
    txt(s, "같은 과녁 tgt[k]에 두 번 조준: 전역(이미지 전체 h_g, 공급 확인) → 국소(둘레 h_s, 위치 특정). 2단계는 1단계가 공급한 것 이상을 만들 수 없다", lx + 0.25, CONTENT_Y + 4.65, lw - 0.5, 0.62, { fontSize: 10, color: C.text2 });
  }
  {
    const s = slide("검증 ①: 배치 커버리지와 배경 선택 유사도", "제5장 · 실험 결과 (그림 5-10, 5-12)");
    await fig(s, "fig5-10.png", M, CONTENT_Y, W - 2 * M, 2.1, { caption: "그림 5-10  ROI 배치 커버리지 비교 (형태 · 맥락 · 희귀 쌍)" });
    await fig(s, "fig5-12.png", M, CONTENT_Y + 2.55, 5.4, 2.8, { caption: "그림 5-12  배경 선택 호환성 비교" });
    const x0 = 6.4;
    card(s, x0, CONTENT_Y + 2.55, W - M - x0, 3.05);
    txt(s, "배경 텍스처 유사도 — 단측 Mann–Whitney U", x0 + 0.25, CONTENT_Y + 2.7, 6, 0.35, { bold: true, fontSize: 12 });
    const rows = [["Dataset", "Δ (AROMA − Random)", "p"], ["AITeX", "+0.071", "1.7×10⁻¹¹"], ["Severstal", "+0.032", "3.1×10⁻²³"], ["Kolektor", "+0.015", "1.2×10⁻⁴"], ["MVTec Leather", "+0.079", "2.8×10⁻⁴²"], ["MTD", "+0.005", "0.25 (n.s.)"]]
      .map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0, color: i === 0 ? H.lt1 : (i === 5 ? H.accent5 : H.dk1), fill: { color: i === 0 ? H.dk2 : (i % 2 ? H.lt2 : H.lt1) }, align: j === 0 ? "left" : "center", fontSize: 10 } })));
    s.addTable(rows, { x: x0 + 0.25, y: CONTENT_Y + 3.1, w: W - M - x0 - 0.5, colW: [2.0, 2.2, 1.9], rowH: 0.3, border: { type: "solid", color: "DDE2E6", pt: 0.5 }, fontFace: THEME.bodyFontFace, objectName: "mw" });
    txt(s, "5개 중 4개에서 유의 우위. 맥락 커버리지는 Random이 높음(MTD 0.85 vs 0.48) — 호환 셀에 집중하는 설계의 필연. 희귀 쌍 커버리지는 쿼터로 AITeX 1.00·MTD 1.00 등 동등 이상", x0 + 0.25, CONTENT_Y + 4.95, W - M - x0 - 0.5, 0.65, { fontSize: 10, color: C.text2 });
  }
  {
    const s = slide("검증 ②: 5개 데이터셋 YOLOv8n mAP@0.5 (mean, n = 3 seeds)", "제5장 · 실험 결과 (표 5-5 ~ 5-9)");
    const labels = ["AITeX", "Kolektor", "Severstal", "MTD", "MVTec Leather"];
    s.addChart(pres.ChartType.bar, [
      { name: "Baseline", labels, values: [0.2039, 0.9740, 0.4959, 0.9094, 0.8321] },
      { name: "Random", labels, values: [0.4187, 0.9938, 0.4935, 0.9465, 0.8543] },
      { name: "AROMA", labels, values: [0.4683, 0.9870, 0.5041, 0.9440, 0.8052] },
    ], Object.assign(chartBase("Baseline / Random-ROI / AROMA — mAP@0.5"), { x: M, y: CONTENT_Y, w: 8.2, h: 5.2, barGrouping: "clustered", chartColors: [H.accent3, H.accent2, H.accent1], valAxisMinVal: 0, valAxisMaxVal: 1.0, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 8, showLegend: true, legendPos: "b", objectName: "chart-ch5" }));
    const x0 = 9.1;
    const d = [["AITeX", "+4.96 pp", "CCI 0.440", C.accent4], ["Severstal", "+1.06 pp", "CCI 0.306", C.accent4], ["MTD", "−0.25 pp", "CCI 0.246", C.accent3], ["Kolektor", "−0.68 pp", "CCI 0.224", C.accent3], ["Leather", "−4.91 pp", "CCI 0.200", C.accent5]];
    txt(s, "AROMA − Random", x0, CONTENT_Y, W - M - x0, 0.35, { bold: true, fontSize: 12 });
    d.forEach(([n, v, c, col], i) => {
      const y = CONTENT_Y + 0.45 + i * 0.95;
      card(s, x0, y, W - M - x0, 0.82);
      txt(s, n, x0 + 0.2, y + 0.08, 1.6, 0.3, { bold: true, fontSize: 11 });
      txt(s, c, x0 + 0.2, y + 0.42, 1.6, 0.3, { fontSize: 10, color: C.text2 });
      txt(s, v, x0 + 1.7, y + 0.05, W - M - x0 - 1.9, 0.72, { bold: true, fontSize: 18, color: col, align: "right", valign: "middle" });
    });
    txt(s, "합성 이미지는 학습 분할에만 추가, 검증·테스트는 실제 이미지만 사용. 세 조건 동일 하이퍼파라미터", M, 6.7, 8.2, 0.3, { fontSize: 9, color: C.accent3 });
  }
  {
    const s = slide("핵심 발견: 효과의 부호가 CCI 순서대로 뒤집힌다", "제5장 · 해석 · 한계 및 다음 단계");
    // CCI 순서 띠
    const seq = [["AITeX", "0.440", "+4.96", C.accent4], ["Severstal", "0.306", "+1.06", C.accent4], ["MTD", "0.246", "−0.25", C.accent3], ["Leather", "0.200", "−4.91", C.accent5]];
    const bw = (W - 2 * M - 3 * 0.5) / 4;
    seq.forEach(([n, c, v, col], i) => {
      const x = M + i * (bw + 0.5);
      card(s, x, CONTENT_Y, bw, 1.75);
      txt(s, n, x + 0.2, CONTENT_Y + 0.12, bw - 0.4, 0.35, { bold: true, fontSize: 13 });
      txt(s, "CCI " + c, x + 0.2, CONTENT_Y + 0.47, bw - 0.4, 0.3, { fontSize: 11, color: C.text2 });
      txt(s, v + " pp", x + 0.2, CONTENT_Y + 0.8, bw - 0.4, 0.8, { bold: true, fontSize: 26, color: col, valign: "middle" });
      if (i < 3) arrow(s, x + bw + 0.1, CONTENT_Y + 0.7, 0.3, 0.35, { fill: { color: C.accent3 } });
    });
    txt(s, "CCI 높은 순 = AROMA 우위 순 (정확히 일치)", M, CONTENT_Y + 1.85, W - 2 * M, 0.35, { fontSize: 12, bold: true, color: C.accent1 });
    const cw2 = (W - 2 * M - 0.4) / 2;
    card(s, M, CONTENT_Y + 2.35, cw2, 1.75);
    label(s, "효과가 나타나는 조건", M + 0.25, CONTENT_Y + 2.55, cw2 - 0.5, 0.4, { fill: C.accent4, fontSize: 12 });
    bullets(s, ["기준 모델에 개선 여지가 있고 (AITeX baseline 0.20)", "배경 맥락 복잡도(CCI)가 높아 \"어디에 놓는가\"가 정보를 가질 때", "Kolektor처럼 baseline이 near-ceiling(0.974)이면 배치 정책 효과 자체가 소멸"], M + 0.25, CONTENT_Y + 3.05, cw2 - 0.5, 1.3, { fontSize: 11, gap: 4 });
    card(s, M + cw2 + 0.4, CONTENT_Y + 2.35, cw2, 1.75);
    label(s, "정직한 해석", M + cw2 + 0.65, CONTENT_Y + 2.55, cw2 - 0.5, 0.4, { fill: C.accent2, fontSize: 12 });
    bullets(s, ["AROMA–Random 차이는 paired t(df=2)에서 유의에 이르지 못함", "주장의 근거는 개별 격차의 크기가 아니라 CCI를 따라 부호가 순서대로 뒤집히는 패턴 자체", "균질 표면(Leather)에서는 두 배치 방식이 사실상 구분되지 않음"], M + cw2 + 0.65, CONTENT_Y + 3.05, cw2 - 0.5, 1.3, { fontSize: 11, gap: 4 });
    transition(s, "한계: 기존 결함 패치를 재배치하는 copy-paste 범위 → 기존 형태 다양성을 넘어서는 새 결함 표현 불가. 구성요소별 절제 실험 미수행", "제6장 CASDA: 철강 도메인에 특화하고 ControlNet 생성을 결합해 결함 자체를 새로 만든다", 5.85);
  }

  // ---------------- VI. 제6장 CASDA ----------------
  section = "VI. 제6장 CASDA"; pres.addSection({ title: section });
  {
    const s = slide("제6장 CASDA: 문맥 인지 강철 결함 증강 — 5단계 파이프라인", "제6장 · 개요 (Context-Aware Steel Defect Augmentation, Severstal)");
    await fig(s, "fig6-1.png", M, CONTENT_Y, 5.4, 5.2, { caption: "그림 6-1  CASDA 파이프라인" });
    card(s, 6.4, CONTENT_Y, W - M - 6.4, 2.0, { fill: { color: C.accent2 } });
    txt(s, "도메인 특화", 6.7, CONTENT_Y + 0.2, 5.5, 0.4, { bold: true, fontSize: 13, color: C.accent6 });
    txt(s, "철강 표면(Severstal, 1600×256 grayscale, 4 class)의 결함 형태–배경 표면 관계를 호환성 매트릭스로 구성하고, ControlNet 조건부 생성과 결합해 \"어디에(ROI)\"에서 \"무엇을(생성)\"까지 확장", 6.7, CONTENT_Y + 0.6, 5.6, 1.35, { fontSize: 13, color: C.background1 });
    const st = ["① 데이터셋 프로파일링 — 기하학적 ROI 특성화·배경 텍스처 분류", "② 결함–배경 호환성 분석 — 3채널 힌트·프롬프트 구성", "③ 조건부 결함 생성 — ControlNet(SD v1.5) fine-tuning", "④ ROI 탐색 및 배치 — 호환성 기반 배경 선택 + Poisson blending", "⑤ 품질 게이트(Q ≥ 0.7) → 통합 → 검출 모델 학습·평가"];
    st.forEach((t, i) => {
      const y = CONTENT_Y + 2.25 + i * 0.6;
      s.addShape(pres.ShapeType.roundRect, { x: 6.4, y, w: W - M - 6.4, h: 0.5, rectRadius: 0.06, fill: { color: i % 2 ? C.background2 : C.background1 }, line: { color: H.lt2, width: 1 }, objectName: "step" });
      txt(s, t, 6.6, y, W - M - 6.8, 0.5, { fontSize: 12, valign: "middle" });
    });
    txt(s, "분할 70:15:15 (테스트 ≈ 1,000장) · Class 2가 가장 희소", 6.4, CONTENT_Y + 5.3, W - M - 6.4, 0.3, { fontSize: 10, color: C.accent3 });
  }
  {
    const s = slide("1단계 특성화: 결함 유형과 배경 유형을 같은 방식으로 분포에서 도출", "제6장 · 방법론 — 수식 (6)~(11), 그림 6-2 ~ 6-6");
    const bw2 = (W - 2 * M - 0.3) / 2, bh2 = 3.05;
    ruleBlock(s, M, CONTENT_Y, bw2, bh2, {
      tag: "결함 유형", title: "결함 하위유형 4종 — 분포 유도 고정 임계",
      features: "λ 선형성(공분산 고유값) · σ 솔리디티(볼록껍질 면적비) · α 종횡비(타원 적합) · extent (결함 마스크, N=19,958)",
      criterion: "임계값은 전체 결함 인스턴스 분포에서 로그 스케일 Otsu 분할로 도출 (λ=0.85, σ=0.9) — 그림 6-3",
      lines: ["linear_scratch  ⟺  λ > 0.85  ∧  α > 5.0", "irregular           ⟺  σ < 0.7", "compact_blob    ⟺  α < 2.0  ∧  σ > 0.9", "general              ⟺  otherwise"],
      result: "linear_scratch / irregular / compact_blob / general → 프롬프트 F1 · 힌트 R채널",
    });
    ruleBlock(s, M + bw2 + 0.3, CONTENT_Y, bw2, bh2, {
      tag: "배경 유형", title: "배경 텍스처 5종 — 분포 유도 고정 임계", tagColor: C.accent4,
      features: "LocalVariance · EdgeDensity(Canny) · 방향성(Sobel x/y 비율) (정상 패치, 64px 슬라이딩 그리드)",
      criterion: "임계값은 정상 배경 패치 분포에서 동일 방식으로 도출 (분산 τ=14.09, 엣지 ε=0.607) — 그림 6-5·6-6",
      lines: ["Smooth    ⟺  LocalVariance ≤ 14.09", "Complex  ⟺  LocalVariance > 14.09  ∨  EdgeDensity > 0.607", "Vertical / Horizontal Stripe  ⟺  Sobel 방향성 비율 우세", "Textured  ⟺  otherwise"],
      result: "Smooth / Vertical Stripe / Horizontal Stripe / Textured / Complex → 프롬프트 F2 · 힌트 G·B채널",
    });
    const yb = CONTENT_Y + bh2 + 0.25, fh = 6.95 - yb - 0.3;
    await fig(s, "fig6-3.png", M, yb, 3.0, fh, { caption: "그림 6-3  형태 특성 분포·임계값" });
    await fig(s, "fig6-5.png", M + 3.15, yb, 1.65, fh, { caption: "그림 6-5  배경 분산 (τ)" });
    await fig(s, "fig6-6.png", M + 4.95, yb, 1.65, fh, { caption: "그림 6-6  엣지 밀도 (ε)" });
    await fig(s, "fig6-2.jpg", M + 6.9, yb, 2.6, fh, { caption: "그림 6-2  ROI 특성화" });
    await fig(s, "fig6-4.png", M + 9.65, yb, 2.45, fh, { caption: "그림 6-4  배경 텍스처 분류" });
  }
  {
    const s = slide("2단계 ControlNet 구성: 3채널 힌트 이미지 + 4필드 시맨틱 프롬프트", "제6장 · 방법론 — 신규 도식: 채널 → 힌트 → 프롬프트 조립");
    // channel diagram
    const chs = [["R", "결함 기하", "픽셀 단위 결함 마스크", "0.5", C.accent5], ["G", "표면 방향성", "Sobel 연산 기반", "0.3", C.accent4], ["B", "표면 거칠기", "7×7 국소 분산", "0.2", C.accent2]];
    chs.forEach(([c, n, d, w, col], i) => {
      const y = CONTENT_Y + i * 1.05;
      s.addShape(pres.ShapeType.roundRect, { x: M, y, w: 0.9, h: 0.9, rectRadius: 0.1, fill: { color: col }, line: { color: H.lt1, width: 0 }, objectName: "ch" });
      txt(s, c, M, y, 0.9, 0.9, { color: C.background1, bold: true, fontSize: 26, align: "center", valign: "middle" });
      txt(s, n, M + 1.05, y + 0.08, 2.0, 0.35, { bold: true, fontSize: 13 });
      txt(s, d, M + 1.05, y + 0.45, 2.2, 0.4, { fontSize: 11, color: C.text2 });
      txt(s, "× " + w, M + 3.2, y, 0.8, 0.9, { bold: true, fontSize: 16, color: col, valign: "middle" });
    });
    arrow(s, M + 4.1, CONTENT_Y + 1.3, 0.5, 0.4);
    s.addShape(pres.ShapeType.roundRect, { x: M + 4.75, y: CONTENT_Y + 0.5, w: 1.9, h: 2.0, rectRadius: 0.1, fill: { color: C.accent2 }, line: { color: H.lt1, width: 0 }, objectName: "hint" });
    txt(s, "힌트 이미지\n0.5R + 0.3G + 0.2B", M + 4.75, CONTENT_Y + 0.5, 1.9, 2.0, { color: C.background1, bold: true, fontSize: 13, align: "center", valign: "middle" });
    // prompt
    card(s, M, CONTENT_Y + 3.3, 6.65, 2.1);
    txt(s, "프롬프트 = F1 결함 유형 · F2 배경 표면 · F3 방향성 텍스처 · F4 안정성 점수 기반 표면 상태", M + 0.2, CONTENT_Y + 3.4, 6.3, 0.5, { bold: true, fontSize: 11 });
    const pr = [["a linear scratch defect", C.accent5], ["on vertical striped metal surface", C.accent4], ["with directional texture", C.accent2], ["(pristine)", C.accent1], [", class 2.", C.text2]];
    s.addText(pr.map(([t, col], i) => ({ text: t + (i < 4 ? " " : ""), options: { color: col, bold: true } })), { x: M + 0.2, y: CONTENT_Y + 3.95, w: 6.3, h: 0.8, fontSize: 13, fontFace: "Consolas", margin: 0, isTextBox: true, valign: "top", objectName: "prompt" });
    txt(s, "F1 · F2 · F3 · F4 (F4 = 0.85 → pristine) — 고정 템플릿으로 자동 생성", M + 0.2, CONTENT_Y + 4.8, 6.3, 0.4, { fontSize: 10, color: C.text2 });
    await fig(s, "fig6-8.jpg", 7.6, CONTENT_Y, W - M - 7.6, 3.4, { caption: "그림 6-8  3채널 힌트 이미지 구성 (Class 1~4)" });
    bullets(s, ["ControlNet(canny) + Stable Diffusion v1.5, AdamW lr 1e-5, min-SNR γ=5, ≤20 epoch", "추론 30 step, CFG 7.5, conditioning scale 0.7 → grayscale 변환(VAE 색 아티팩트 제거)"], 7.6, 5.5, W - M - 7.6, 1.2, { fontSize: 10, gap: 3 });
  }
  {
    const s = slide("3~5단계: 호환성 기반 배경 선택 → Poisson → 품질 게이트 → 통합", "제6장 · 방법론 · 생성 샘플 (표 6-2, 그림 6-9)");
    const st = [["③ 맥락 기반 합성", "결함 유형 × 배경 텍스처 공존 확률을 인코딩한 호환성 매트릭스에 따라 배경을 확률적으로 선택. Poisson blending으로 경계 불연속 제거"], ["④ 품질 검증 게이트", "Q = 형태 일관성 Q_morph + 선명도 Q_sharp + 아티팩트 Q_artifact + 면적비 Q_area 가중합, Q ≥ 0.7만 채택"], ["⑤ 데이터셋 통합", "채택 합성 이미지의 결함 마스크를 인코딩해 기존 학습셋에 통합"]];
    st.forEach(([h, d], i) => {
      const y = CONTENT_Y + i * 1.15;
      card(s, M, y, 6.2, 1.0);
      txt(s, h, M + 0.2, y + 0.08, 5.8, 0.35, { bold: true, fontSize: 13, color: C.accent1 });
      txt(s, d, M + 0.2, y + 0.42, 5.8, 0.55, { fontSize: 11, color: C.text2 });
    });
    const rows = [["항목", "값"], ["품질 검증 배치의 생성 후보", "2,500"], ["품질 검증 통과", "2,075 (83.0%)"], ["Class 2 표본 수", "247 → 519 (+110.1%)"], ["최종 통합 합성 표본", "2,238"], ["최종 학습 데이터셋", "7,475 (원본 5,237 + 합성 2,238, 42.7%)"]]
      .map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || i === 3, color: i === 0 ? H.lt1 : (i === 3 ? H.accent1 : H.dk1), fill: { color: i === 0 ? H.dk2 : (i % 2 ? H.lt2 : H.lt1) }, align: j === 0 ? "left" : "center", fontSize: 10 } })));
    s.addTable(rows, { x: M, y: CONTENT_Y + 3.55, w: 6.2, colW: [2.8, 3.4], rowH: 0.3, border: { type: "solid", color: "DDE2E6", pt: 0.5 }, fontFace: THEME.bodyFontFace, objectName: "aug-stats" });
    await fig(s, "fig6-9.png", 7.2, CONTENT_Y, W - M - 7.2, 4.0, { caption: "그림 6-9  동일 힌트 조건에서 생성된 결함 샘플 vs 원본 GT" });
    card(s, 7.2, CONTENT_Y + 4.4, W - M - 7.2, 1.1);
    txt(s, [{ text: "LPIPS Diversity  CASDA 0.448 > Copy-Paste 0.394", options: { bold: true, breakLine: true } }, { text: "LPIPS Realism  CASDA 0.467 < Copy-Paste 0.491 (낮을수록 우수)", options: { bold: true, breakLine: true } }, { text: "단일 힌트에서 시각적으로 다른 복수 출력 — 단순 복제가 아닌 구조적 다양성", options: { color: H.dk2 } }], 7.4, CONTENT_Y + 4.5, W - M - 7.6, 0.95, { fontSize: 11 });
  }
  {
    const s = slide("6장 결과: CASDA mAP@0.5 0.3196, 희소 Class 2 AP +22.09 pp", "제6장 · 실험 결과 (EB-YOLOv8, 표 6-3, n = 2 seeds)");
    const labels = ["mAP@0.5", "Class 1", "Class 2", "Class 3", "Class 4"];
    s.addChart(pres.ChartType.bar, [
      { name: "Baseline (Raw)", labels, values: [0.2936, 0.1490, 0.2548, 0.3966, 0.3739] },
      { name: "Copy-Paste", labels, values: [0.2722, 0.1479, 0.0550, 0.4594, 0.4266] },
      { name: "CASDA", labels, values: [0.3196, 0.2493, 0.2759, 0.4287, 0.3246] },
    ], Object.assign(chartBase("EB-YOLOv8 검출 성능 (Severstal)"), { x: M, y: CONTENT_Y, w: 7.8, h: 4.0, barGrouping: "clustered", chartColors: [H.accent3, H.accent2, H.accent1], valAxisMinVal: 0, valAxisMaxVal: 0.5, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 8, showLegend: true, legendPos: "b", objectName: "chart-ch6" }));
    const x0 = 8.7;
    const stats = [["+22.09 pp", "Class 2 AP vs Copy-Paste (0.0550 → 0.2759)", C.accent4], ["+10.03 pp", "Class 1 AP vs Raw — 최대 향상", C.accent4], ["−4.93 / −10.20 pp", "Class 4 AP vs Raw / Copy-Paste — 하락", C.accent5]];
    stats.forEach(([v, d, col], i) => {
      const y = CONTENT_Y + i * 1.32;
      card(s, x0, y, W - M - x0, 1.2);
      txt(s, v, x0 + 0.2, y + 0.1, W - M - x0 - 0.4, 0.6, { bold: true, fontSize: 22, color: col });
      txt(s, d, x0 + 0.2, y + 0.72, W - M - x0 - 0.4, 0.5, { fontSize: 10, color: C.text2 });
    });
    card(s, M, CONTENT_Y + 4.2, W - 2 * M, 1.1, { fill: { color: C.accent2 } });
    txt(s, "해석: context-aware 생성이 단순 patch 재사용보다 판별력 높은 minority-class 샘플을 제공. 다만 Class 3·4는 하락 — 전 클래스 균일 개선은 아님. n=2 소규모 실험으로 Wilcoxon(BH-FDR)은 유의 미달(d = 0.47 / 0.19) → 확정 결론이 아닌 기술 통계로 제시", M + 0.3, CONTENT_Y + 4.3, W - 2 * M - 0.6, 0.9, { fontSize: 12, color: C.background1, valign: "middle" });
  }

  // ---------------- VII. 결론 ----------------
  section = "VII. 결론"; pres.addSection({ title: section });
  {
    const s = slide("연구 요약: 생성 → 배치 도입 → 데이터 기반 자동화 → 도메인 특화·생성 통합", "제7장 결론 · 연구 요약");
    const rows = [["3장 PCB", "GAN/Diffusion 패치 증강 + 원위치 재배치", "YOLOv8m Basic mAP50-95 0.54 · Diffusion > GAN(mAP) · GAN 과적합 소폭 우위", "P1", C.accent3],
      ["4장 ROI", "ControlNet 생성 + 밝기 유사도 ROI + Poisson", "ResNet34 mIoU 0.567 → 0.892 · EfficientNet 0.168 → 0.686", "P1 + P2", C.accent2],
      ["5장 AROMA", "데이터 기반 대칭 호환성 게이트, 5개 도메인 재설계 없이 적용", "AITeX +4.96 pp · Severstal +1.06 pp · 효과는 CCI 순서와 일치, 4/5에서 배경 유사도 유의", "P2 자동화", C.accent1],
      ["6장 CASDA", "철강 특화 호환성 + ControlNet 생성 결합", "mAP 0.3196 (Raw +2.60, CP +4.74 pp) · Class 2 AP +22.09 pp · LPIPS Div 0.448", "P1 + P2 통합", C.accent1]];
    rows.forEach(([ch, m, r, tag, col], i) => {
      const y = CONTENT_Y + i * 1.3;
      card(s, M, y, W - 2 * M, 1.15);
      label(s, ch, M + 0.2, y + 0.3, 1.6, 0.5, { fill: col, fontSize: 12 });
      txt(s, m, M + 2.0, y + 0.12, 4.6, 0.95, { fontSize: 12, bold: true, valign: "middle" });
      txt(s, r, M + 6.8, y + 0.12, 4.0, 0.95, { fontSize: 11, color: C.text2, valign: "middle" });
      label(s, tag, W - M - 1.4, y + 0.3, 1.2, 0.5, { fill: C.text2, fontSize: 10 });
      if (i < 3) s.addShape(pres.ShapeType.downArrow, { x: M + 0.85, y: y + 1.15, w: 0.3, h: 0.15, fill: { color: C.accent3 }, line: { color: H.accent3, width: 0 }, objectName: "down" });
    });
    txt(s, "AROMA → CASDA: 다중 도메인 자동화와 단일 도메인 생성적 특화는 배타적이 아니라, 일반화된 배치 원리를 먼저 확립한 뒤 특정 도메인의 생성 문제에 적용하는 순차 결합", M, 6.65, W - 2 * M, 0.5, { fontSize: 11, color: C.accent1, bold: true });
  }
  {
    const s = slide("연구의 의의와 한계", "제7장 결론 · 2~3절");
    const cw = (W - 2 * M - 0.4) / 2;
    card(s, M, CONTENT_Y, cw, 5.2);
    label(s, "학술적 기여", M + 0.25, CONTENT_Y + 0.25, cw - 0.5, 0.45, { fill: C.accent4, fontSize: 13 });
    bullets(s, ["결함 생성(What)과 ROI 배치(Where) 문제의 명시적 분리와 단계적 정교화", "수작업 ROI 탐색의 데이터 기반 자동화 — 공존 통계로부터 호환성 기준을 자동 도출", "다중 도메인 일반화 → 특정 도메인 특화·생성적 확장으로의 발전 경로 제시", "6개 산업 데이터셋에서 mAP·mIoU·Dice·FID·KID·LPIPS로 실증 검증", "효과가 제한·역전되는 조건까지 규명 (증강 비율 0.7, 개선 여지 × CCI, 클래스별 불균일)"], M + 0.25, CONTENT_Y + 0.9, cw - 0.5, 4.2, { fontSize: 12, gap: 10 });
    card(s, M + cw + 0.4, CONTENT_Y, cw, 5.2);
    label(s, "한계", M + cw + 0.65, CONTENT_Y + 0.25, cw - 0.5, 0.45, { fill: C.accent5, fontSize: 13 });
    bullets(s, ["배경 통계 의존: 결함–배경 통계 관계가 결함 발생 위치를 충분히 설명한다는 전제. 공정·표면 구조 변화 시 적용 어려움", "고정 ROI 크기: 결함 크기·형태 변화를 충분히 반영하지 못함 — 작은 결함은 맥락 부족, 큰 결함은 불필요한 배경 포함", "대상 영역 미식별: 결함 위치가 특정 객체·구조에 의해 결정되는 경우 배경 통계만으로는 사전 식별 불가", "소규모 시드(AROMA n=3, CASDA n=2)로 통계적 유의성 제한"], M + cw + 0.65, CONTENT_Y + 0.9, cw - 0.5, 4.2, { fontSize: 12, gap: 10 });
  }
  {
    const s = slide("향후 연구: 세 한계에 대응하는 세 방향", "제7장 결론 · 4절");
    const pairs = [["고정 ROI 크기", "가변 ROI(variable-size) 탐색", "결함 크기·형태에 따라 ROI 범위를 동적으로 결정"], ["대상 영역 사전 식별의 부재", "객체탐지 기반 선행 위치 추정", "결함이 발생할 수 있는 대상 영역을 먼저 탐지한 뒤 그 영역의 특성으로 ROI 결정"], ["배경 통계 중심의 매칭", "공간적·구조적 정보의 통합", "형태·크기·위치·주변 구조를 함께 모델링하는 적응형 결함 합성 프레임워크"]];
    pairs.forEach(([lim, dir, d], i) => {
      const y = CONTENT_Y + i * 1.45;
      card(s, M, y, 4.6, 1.25, { fill: { color: C.background2 } });
      txt(s, "한계 " + (i + 1), M + 0.25, y + 0.12, 4.1, 0.3, { fontSize: 10, color: C.accent5, bold: true });
      txt(s, lim, M + 0.25, y + 0.42, 4.1, 0.7, { fontSize: 14, bold: true, valign: "middle" });
      arrow(s, M + 4.75, y + 0.42, 0.55, 0.4);
      card(s, M + 5.45, y, W - M - (M + 5.45), 1.25, { fill: { color: C.accent2 } });
      txt(s, dir, M + 5.7, y + 0.12, 6.5, 0.45, { fontSize: 14, bold: true, color: C.accent6 });
      txt(s, d, M + 5.7, y + 0.6, 6.5, 0.6, { fontSize: 11, color: C.background1 });
    });
    s.addShape(pres.ShapeType.rect, { x: M, y: 5.95, w: W - 2 * M, h: 0.85, fill: { color: C.accent1 }, line: { color: H.accent1, width: 0 }, objectName: "thanks" });
    txt(s, "감사합니다 — 질의응답", M, 5.95, W - 2 * M, 0.85, { fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle" });
  }

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written:", OUT);
})().catch(e => { console.error(e); process.exit(1); });
