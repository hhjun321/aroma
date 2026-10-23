"""학위논문 심사 발표 슬라이드를 deckcraft-cli로 생성한다.

사용:
    python tools/defense_deck.py --cli <deckcraft-cli.exe 경로> [--out defense_deck]

출력: <out>.deckcraft / <out>.pptx / <out>.pdf, <out>_png/ (슬라이드 미리보기)
"""
import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

W, H = 960, 540
FONT = "Malgun Gothic"
ACCENT = "#2E6FD8"
MUTED = "#5B6573"

# ---------------------------------------------------------------- 슬라이드 내용
# kind: title | bullets | figs | table | split(bullets + figs) | end
SLIDES = [
    dict(kind="title",
         title="Context-Aware 데이터 증강 기반\n산업용 결함 검출 연구",
         sub="한국기술교육대학교 컴퓨터공학과\n한호준 · 지도교수 문일영 · 2026",
         notes="네 편의 게재 논문을 하나의 연구 궤적으로 통합한 학위논문입니다."),
    dict(kind="bullets", title="발표 순서",
         bullets=["연구 배경 및 문제 정의",
                  "연구 질문과 발전 서사",
                  "제3장  YOLOv8 + GAN/Diffusion 결함 patch 증강",
                  "제4장  ControlNet + 밝기 유사도 ROI + Poisson blending",
                  "제5장  AROMA: 데이터 기반 ROI 호환성 자동화",
                  "제6장  CASDA: 강철 표면 context-aware 생성 증강",
                  "결론 · 한계 · 향후 연구"],
         numbered=True),
    dict(kind="bullets", title="연구 배경",
         bullets=["딥러닝 기반 자동 외관 검사(AVI)로 검사 효율·정확도 향상",
                  "실제 산업 현장: 결함 발생률이 낮아 결함 샘플이 근본적으로 희소",
                  "정상/결함 클래스 간 심각한 불균형 → 소수 결함 클래스 과소적합",
                  "해결 방향: 데이터 증강으로 결함 학습 데이터 확충",
                  "핵심 문제의식: 결함 후보 영역(ROI)을 어떻게 찾고, 그 자리에 무엇을 배치할 것인가"],
         notes="데이터 희소성과 불균형이 출발점입니다."),
    dict(kind="two_box", title="문제 정의: 생성(P1)과 배치(P2)",
         boxes=[("P1 · What to place",
                 "결함 patch를 어떻게 생성·증강할 것인가\n\n형태·구조 보존 + 학습에 유효한 다양성\n품질 미흡 시 비현실적 데이터가 성능 저하"),
                ("P2 · Where to place",
                 "결함 patch를 정상 이미지의 어느 ROI에 합성할 것인가\n\n결함 발생은 배경의 통계·형태 특성과 연관\n호환성 낮은 위치 → 실제 분포와 괴리")],
         footer="기존 연구는 P1 중심 · P2는 원위치 재사용/무작위 또는 도메인별 수작업 규칙 → 본 논문은 P2를 중심으로 확장"),
    dict(kind="bullets", title="연구 질문",
         bullets=["RQ1  생성모델(GAN/Diffusion) 결함 patch 증강은 검출 성능과 일반화 안정성에 어떤 영향을 주는가? (제3장)",
                  "RQ2  주변 배경 유사도 기준 ROI 선정이 원위치 재배치 대비 성능을 개선하는가? (제4장)",
                  "RQ3  데이터로부터 자동 도출된 defect–background 호환성 기준이 다수 도메인에 재설계 없이 일반화되는가? 효과는 어떤 조건에서 나타나는가? (제5장)",
                  "RQ4  자동화된 ROI 프레임워크를 도메인 특화 + 조건부 생성모델과 결합하면 copy-paste의 다양성 한계를 넘어 소수 클래스 성능을 개선하는가? (제6장)"]),
    dict(kind="table", title="연구 발전 서사: 네 편의 게재 논문",
         header=["단계", "연구", "핵심 방식", "극복한 한계"],
         rows=[["1", "YOLOv8 + 생성모델 증강", "GAN/Diffusion 결함 patch 증강 → 원위치 copy-paste", "결함 데이터 부족"],
               ["2", "결함데이터 증강 방법", "ControlNet 생성 + 밝기 유사도 ROI 탐색 + Poisson blending", "위치 무관 재배치"],
               ["3", "AROMA", "데이터 기반 defect/background 호환성 게이트 → copy-paste", "수작업 규칙 의존"],
               ["4", "CASDA", "강철 특화 조건부 prompt + 힌트 → ControlNet 생성 → 배치", "copy-paste 다양성 한계"]],
         col_w=[60, 200, 420, 200],
         footer="P1(생성) → P2(배치) 정교화 → 일반화(AROMA) → 특화·생성 재결합(CASDA)"),
    # ---- 제3장
    dict(kind="split", title="제3장 · YOLOv8 + GAN/Diffusion 결함 patch 증강",
         bullets=["PKU-HRI PCB 6종 결함(Short, Spur, Missing hole …)",
                  "GAN / Diffusion으로 결함 patch 생성",
                  "증강 patch를 원본과 동일 위치에 재배치",
                  "YOLOv8s / YOLOv8m 학습·비교"],
         figs=["fig3-4.jpg", "fig3-6.jpg"], fig_caption="위: GAN 생성 · 아래: Diffusion 생성"),
    dict(kind="table", title="제3장 · 결과: 성능–일반화 트레이드오프",
         header=["Model", "Type", "Precision", "Recall", "mAP50", "mAP50-95", "Train–Test Gap"],
         rows=[["YOLOv8s", "Basic", "0.94", "0.93", "0.94", "0.49", "0.02"],
               ["YOLOv8s", "GAN", "0.91", "0.90", "0.94", "0.49", "0.52"],
               ["YOLOv8s", "Diffusion", "0.97", "0.93", "0.96", "0.51", "0.53"],
               ["YOLOv8m", "Basic", "0.96", "0.96", "0.97", "0.54", "–"],
               ["YOLOv8m", "GAN", "0.93", "0.95", "0.96", "0.52", "–"],
               ["YOLOv8m", "Diffusion", "0.97", "0.93", "0.96", "0.53", "–"]],
         footer="Diffusion: 검출 지표 우수 · GAN: 과적합 측면 안정 → 한계: 배경과의 호환성(어디에 둘지)은 미고려"),
    # ---- 제4장
    dict(kind="split", title="제4장 · ControlNet + 밝기 유사도 ROI + Poisson blending",
         bullets=["KolektorSDD (금속 표면)",
                  "ControlNet으로 결함 이미지·마스크 생성",
                  "결함 주변 pixel 밝기 유사도를 슬라이딩 윈도우로 탐색 → 유효 영역(ROI) 선정",
                  "Poisson blending으로 이질감 없이 합성"],
         figs=["fig4-3.jpg", "fig4-4.jpg"], fig_caption="좌: 유효 영역 추출 · 우: Poisson blending 합성"),
    dict(kind="table_fig", title="제4장 · 결과: ROI 배치 도입의 효과",
         header=["증강 비율", "ResNet34 mIoU", "ResNet34 Dice", "EffNet mIoU", "EffNet Dice"],
         rows=[["Real", "0.5667", "0.6766", "0.1682", "0.2531"],
               ["0.3", "0.8069", "0.8799", "0.5378", "0.6070"],
               ["0.5", "0.8919", "0.9400", "0.5347", "0.5862"],
               ["0.7", "0.8071", "0.8813", "0.6860", "0.7770"]],
         fig="fig4-5.jpg",
         footer="ResNet34 mIoU 0.567 → 0.892 · 비율 0.7에서 하락 → 한계: 밝기 단일 통계량 휴리스틱, 텍스처·구조 미반영"),
    # ---- 제5장
    dict(kind="split", title="제5장 · AROMA 파이프라인",
         bullets=["5개 산업 데이터셋: AITeX(섬유) · Kolektor(금속) · Severstal(철강) · MTD(자기타일) · MVTec Leather(가죽)",
                  "MCI: 결함 형태 복잡도 · CCI: 배경 맥락 복잡도",
                  "patch-level 공존 통계 → 대칭 호환성 게이트(ctx_prior)",
                  "도메인별 규칙 재설계 없이 동작"],
         figs=["fig5-1.png"], fig_caption="그림 5-1. AROMA 파이프라인"),
    dict(kind="figs", title="제5장 · 데이터 기반 호환성 게이트",
         figs=["fig5-6.png", "fig5-8.png"],
         fig_caption="좌: 데이터셋별 대칭 호환성 히트맵 · 우: 실제 이미지에서의 배경 할당",
         footer="배경 선택: 5개 중 4개 데이터셋에서 AROMA 할당 배경이 Random보다 유의하게 실제 결함 맥락에 근접 (Mann–Whitney U)"),
    dict(kind="table", title="제5장 · 결과: 효과가 나타나는 조건",
         header=["데이터셋", "AROMA − Random (mAP, pp)", "개선 여지 + 배경 복잡도(CCI) 동시 충족"],
         rows=[["AITeX", "+4.96", "충족"],
               ["Severstal", "+1.06", "충족"],
               ["MTD", "−0.25", "미충족"],
               ["Kolektor", "−0.68", "미충족"],
               ["MVTec Leather", "−4.91", "미충족"]],
         col_w=[240, 300, 340],
         footer="YOLOv8n, 3 seeds paired t (df=2) — 유의성 미도달, CCI 순서와 일치하는 일관된 방향성으로 해석\n두 조건이 동시에 충족될 때만 context-aware 배치가 유효 → '언제 유효한가'의 적용 조건 제시"),
    # ---- 제6장
    dict(kind="split", title="제6장 · CASDA 5단계 파이프라인",
         bullets=["Severstal 강철 표면 (1600×256, Class 2 극희소)",
                  "1단계: ROI 형태·배경 텍스처 특성화 → 적합도 분류",
                  "2단계: defect_type·background_type 조건 prompt + 3채널 힌트",
                  "3단계: ControlNet으로 결함 자체를 새로 생성",
                  "4–5단계: 배치·품질 관리 → EB-YOLOv8 학습"],
         figs=["fig6-1.png"], fig_caption="그림 6-1. CASDA 파이프라인"),
    dict(kind="table_fig", title="제6장 · 결과: 소수 클래스 개선",
         header=["Method", "mAP@0.5", "C1 AP", "C2 AP", "C3 AP", "C4 AP"],
         rows=[["Raw", "0.2936", "0.1490", "0.2548", "0.3966", "0.3739"],
               ["Copy-Paste", "0.2722", "0.1479", "0.0550", "0.4594", "0.4266"],
               ["CASDA", "0.3196", "0.2493", "0.2759", "0.4287", "0.3246"],
               ["vs. Raw", "+2.60", "+10.03", "+2.11", "+3.21", "−4.93"],
               ["vs. CP", "+4.74", "+10.14", "+22.09", "−3.07", "−10.20"]],
         fig="fig6-9.png",
         footer="LPIPS Diversity: CASDA 0.448 vs Copy-Paste 0.394 · 2 seeds(42, 456) 기술통계, α=0.05 미통과 → 경향으로 해석"),
    # ---- 결론
    dict(kind="bullets", title="결론 및 기여",
         bullets=["ROI 문제의 명시적 분리: What to place(P1) vs Where to place(P2)",
                  "수작업 의존성의 데이터 기반 대체: 밝기 휴리스틱 → 공존 통계 호환성 게이트(AROMA)",
                  "일반화 → 특화로의 전환과 생성적 확장: AROMA 프레임워크 + ControlNet 조건부 생성(CASDA)",
                  "일관된 실증 검증: 6개 산업 데이터셋, 검출·분할 지표 + 합성 품질 지표(FID·KID·LPIPS)",
                  "실패·무효화 조건까지 규명: 성능–일반화 트레이드오프, 증강 비율 과다, 조건 미충족 시 역전, 소수 시드 한계"]),
    dict(kind="bullets", title="한계 및 향후 연구",
         bullets=["ROI 매칭이 결함 주변 배경의 통계적 속성에 크게 의존",
                  "→ 다양한 제조 현장의 결함 발생 양상을 모두 설명하지 못함",
                  "소수 시드(n=2~3)로 통계적 유의성 미확보 → multi-seed 확장 필요",
                  "향후: 가변 ROI 사이즈 대응",
                  "향후: 결함 위치가 ROI인 경우 object detection 선행"]),
    dict(kind="end", title="감사합니다", sub="Q & A"),
]


# ---------------------------------------------------------------- 명령 빌더
def cmd(cid, params=None):
    return cid if params is None else f"{cid}={json.dumps(params, ensure_ascii=False)}"


def fit(path, box):
    """box 안에 비율을 유지해 가운데 정렬한 rect."""
    x, y, w, h = box
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    rw, rh = iw * s, ih * s
    return [round(x + (w - rw) / 2, 1), round(y + (h - rh) / 2, 1), round(rw, 1), round(rh, 1)]


def title_bar(c, title):
    c.append(cmd("insert.textBox", {"rect": [40, 24, 880, 56], "text": title}))
    c += [cmd("format.font", {"family": FONT}), cmd("format.size", {"size": 28}),
          cmd("format.bold", {"on": True}), cmd("format.color", {"color": "#1E2B3C"})]
    c.append(cmd("shape.insert", {"preset": "rect", "rect": [40, 84, 120, 4]}))
    c += [cmd("shape.fill", {"color": ACCENT}), cmd("shape.line", {"none": True})]


def text_box(c, rect, text, size=18, color="#1E2B3C", bullets=False, numbered=False, align=None, bold=False):
    c.append(cmd("insert.textBox", {"rect": rect, "text": text}))
    c += [cmd("format.font", {"family": FONT}), cmd("format.size", {"size": size}),
          cmd("format.color", {"color": color})]
    if bold:
        c.append(cmd("format.bold", {"on": True}))
    if bullets:
        c.append(cmd("format.bullets", {"on": True, "char": "•", "color": ACCENT}))
    if numbered:
        c.append(cmd("format.numbering", {"on": True, "scheme": "arabicPeriod", "color": ACCENT}))
    if bullets or numbered:
        c.append(cmd("format.paragraph", {"spaceAfter": 10, "indentLeft": 22, "indentFirst": -22}))
    if align:
        c.append(cmd("format.align", {"align": align}))


def footer(c, text, y=470):
    text_box(c, [40, y, 880, 540 - y - 14], text, size=13, color=MUTED)


def pictures(c, names, box, caption=None):
    x, y, w, h = box
    cap_h = 22 if caption else 0
    n = len(names)
    gap = 12
    stacked = all(Image.open(FIG / f).size[0] / Image.open(FIG / f).size[1] > 2.5 for f in names)
    for i, f in enumerate(names):
        if stacked:  # 가로로 긴 strip 그림은 세로로 쌓는다
            cell = [x, y + i * (h - cap_h) / n, w, (h - cap_h) / n - gap]
        else:
            cw = (w - gap * (n - 1)) / n
            cell = [x + i * (cw + gap), y, cw, h - cap_h]
        c.append(cmd("insert.picture", {"path": str(FIG / f), "rect": fit(FIG / f, cell)}))
    if caption:
        text_box(c, [x, y + h - cap_h, w, cap_h], caption, size=12, color=MUTED, align="center")


def table(c, header, rows, rect, col_w=None, size=15):
    c.append(cmd("insert.table", {"rows": len(rows) + 1, "cols": len(header), "rect": rect}))
    for j, t in enumerate(header):
        c.append(cmd("text.set", {"cell": [0, j], "text": t}))
    for i, r in enumerate(rows, 1):
        for j, t in enumerate(r):
            c.append(cmd("text.set", {"cell": [i, j], "text": t}))
    c += [cmd("format.font", {"family": FONT}), cmd("format.size", {"size": size})]
    if col_w:
        c += [cmd("table.columnWidth", {"col": j, "width": w}) for j, w in enumerate(col_w)]


def build_slide(s, first):
    c = []
    if s["kind"] in ("title", "end"):
        if first:
            c.append(cmd("slide.layout", {"layout": "blank"}))
        else:
            c.append(cmd("slide.new", {"layout": "blank"}))
        c.append(cmd("shape.insert", {"preset": "rect", "rect": [0, 0, W, H]}))
        c += [cmd("shape.fill", {"color": "#1E2B3C"}), cmd("shape.line", {"none": True})]
        c.append(cmd("shape.insert", {"preset": "rect", "rect": [80, 300, 160, 5]}))
        c += [cmd("shape.fill", {"color": ACCENT}), cmd("shape.line", {"none": True})]
        big = s["kind"] == "title"
        text_box(c, [80, 120 if big else 170, 800, 170 if big else 120], s["title"],
                 size=38 if big else 48, color="#FFFFFF", bold=True)
        text_box(c, [80, 320, 800, 120], s["sub"], size=18 if big else 28, color="#C9D3E0")
    else:
        c.append(cmd("slide.new", {"layout": "blank"}))
        title_bar(c, s["title"])
        k = s["kind"]
        if k == "bullets":
            text_box(c, [50, 110, 860, 400], "\n".join(s["bullets"]), size=20,
                     bullets=not s.get("numbered"), numbered=s.get("numbered", False))
        elif k == "two_box":
            for i, (head, body) in enumerate(s["boxes"]):
                x = 50 + i * 440
                c.append(cmd("shape.insert", {"preset": "roundRect", "rect": [x, 110, 420, 260]}))
                c += [cmd("shape.fill", {"color": "#EEF3FB"}), cmd("shape.line", {"color": ACCENT, "width": 1.5})]
                text_box(c, [x + 20, 125, 380, 40], head, size=22, color=ACCENT, bold=True)
                text_box(c, [x + 20, 175, 380, 240], body, size=16)
            footer(c, s["footer"], y=400)
        elif k == "table":
            n = len(s["rows"]) + 1
            table(c, s["header"], s["rows"], [40, 110, 880, min(36 * n, 330)], s.get("col_w"))
            footer(c, s["footer"], y=440)
        elif k == "split":
            text_box(c, [50, 110, 400, 400], "\n".join(s["bullets"]), size=17, bullets=True)
            pictures(c, s["figs"], [470, 105, 460, 410], s.get("fig_caption"))
        elif k == "figs":
            pictures(c, s["figs"], [40, 100, 880, 360], s.get("fig_caption"))
            footer(c, s["footer"], y=468)
        elif k == "table_fig":
            n = len(s["rows"]) + 1
            table(c, s["header"], s["rows"], [40, 110, 500, 34 * n], size=13)
            pictures(c, [s["fig"]], [560, 100, 370, 340])
            footer(c, s["footer"], y=448)
    if s.get("notes"):
        c.append(cmd("slide.notes", {"text": s["notes"]}))
    return c


def run(cli, deck, cmds, new=False):
    args = [cli, "run"] + ([] if new else ["--in", str(deck)])
    for x in cmds:
        args += ["--cmd", x]
    args += ["--save", str(deck)]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        raise SystemExit(f"deckcraft-cli 실패:\n{r.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--out", default=str(ROOT / "defense_deck"))
    a = ap.parse_args()
    deck = Path(a.out + ".deckcraft")

    for i, s in enumerate(SLIDES):
        head = [cmd("file.new", {"theme": "Slate"})] if i == 0 else [cmd("slide.go", {"index": i - 1})]
        run(a.cli, deck, head + build_slide(s, i == 0), new=(i == 0))
        print(f"[{i + 1}/{len(SLIDES)}] {s['title'].splitlines()[0]}")

    png = Path(a.out + "_png")
    for target in [a.out + ".pptx", a.out + ".pdf"]:
        subprocess.run([a.cli, "convert", str(deck), target], check=True)
    subprocess.run([a.cli, "render", str(deck), "--all", "--scale", "1", str(png) + "/"], check=True)
    print("완료:", deck)


if __name__ == "__main__":
    main()
