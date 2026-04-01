"""debug_yolo_train.py

YOLO11 학습 TypeError 원인 파악용 디버그 스크립트.
Colab 에서 직접 실행:

  python debug_yolo_train.py --cat_dir /path/to/LSM_1 --group aroma_full
"""
import argparse
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat_dir", required=True, help="카테고리 루트 (e.g. .../LSM_1)")
    parser.add_argument("--group", default="aroma_full",
                        choices=["baseline", "aroma_full", "aroma_pruned"])
    parser.add_argument("--epochs", type=int, default=2, help="빠른 확인용 소수 epoch")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    cat_dir  = Path(args.cat_dir)
    data_dir = cat_dir / "augmented_dataset" / args.group
    train_dir = data_dir / "train"

    print("=" * 60)
    print(f"cat_dir  : {cat_dir}")
    print(f"data_dir : {data_dir}")
    print(f"train_dir: {train_dir}")
    print(f"exists   : {train_dir.exists()}")
    if train_dir.exists():
        for sub in train_dir.iterdir():
            imgs = list(sub.glob("*")) if sub.is_dir() else []
            print(f"  {sub.name}: {len(imgs)} files")
    print("=" * 60)

    # ── Step 1: YOLO import ───────────────────────────────────────────
    print("\n[Step 1] ultralytics import")
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"  ultralytics version: {ultralytics.__version__}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    # ── Step 2: 모델 로드 ─────────────────────────────────────────────
    print("\n[Step 2] YOLO('yolo11n-cls.pt') 로드")
    try:
        model = YOLO("yolo11n-cls.pt")
        print(f"  model.model type: {type(model.model)}")
        print(f"  model.ckpt_path : {getattr(model, 'ckpt_path', '(attr 없음)')}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    # ── Step 3: model.train() ─────────────────────────────────────────
    print("\n[Step 3] model.train() 시작 (verbose=True 로 전체 로그 출력)")
    train_exc = None
    try:
        model.train(
            data=str(data_dir),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=0.01,
            verbose=True,   # 디버그용 전체 출력
            exist_ok=True,
            seed=42,
            val=False,
        )
        print("  model.train() 정상 완료")
    except Exception as e:
        train_exc = e
        print(f"\n  !! model.train() 예외 발생 !!")
        print(f"  type : {type(e).__name__}")
        print(f"  msg  : {e}")
        print("\n  -- 전체 traceback --")
        traceback.print_exc()

    # ── Step 4: trainer 속성 확인 ─────────────────────────────────────
    print("\n[Step 4] trainer 속성 확인")
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        print("  model.trainer: None (trainer 미생성)")
    else:
        best = getattr(trainer, "best", "(attr 없음)")
        last = getattr(trainer, "last", "(attr 없음)")
        save_dir = getattr(trainer, "save_dir", "(attr 없음)")
        print(f"  trainer.save_dir : {save_dir}")
        print(f"  trainer.best     : {best}  (exists={Path(str(best)).exists() if best not in (None, '(attr 없음)') else 'N/A'})")
        print(f"  trainer.last     : {last}  (exists={Path(str(last)).exists() if last not in (None, '(attr 없음)') else 'N/A'})")
        print(f"  model.model      : {type(model.model) if hasattr(model, 'model') and model.model is not None else None}")
        print(f"  model.ckpt_path  : {getattr(model, 'ckpt_path', '(attr 없음)')}")

    # ── Step 5: last.pt 에서 새 인스턴스 로드 ────────────────────────
    print("\n[Step 5] trainer.last 에서 새 YOLO 인스턴스 생성")
    if trainer is not None:
        last = getattr(trainer, "last", None)
        if last is not None and Path(str(last)).exists():
            try:
                model2 = YOLO(str(last))
                print(f"  YOLO(last) 성공: {type(model2.model)}")
            except Exception:
                print("  YOLO(last) 실패:")
                traceback.print_exc()
        else:
            print(f"  last.pt 없음: {last}")
    else:
        print("  trainer 없음 — skip")

    # ── Step 6: predict 테스트 ────────────────────────────────────────
    print("\n[Step 6] model.predict() 테스트")
    test_dir = cat_dir / "augmented_dataset" / "baseline" / "test"
    test_images: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"):
        for cls_dir in test_dir.iterdir():
            if cls_dir.is_dir():
                test_images.extend([str(p) for p in cls_dir.glob(ext)])
    test_images = test_images[:4]   # 최대 4장만

    if not test_images:
        print(f"  test 이미지 없음: {test_dir}")
    else:
        print(f"  test 이미지 {len(test_images)}장으로 predict")
        try:
            results = model.predict(test_images, imgsz=args.imgsz, batch=4, verbose=False)
            for r in results:
                probs = r.probs
                print(f"    probs type={type(probs)}, data={getattr(probs, 'data', None)}")
        except Exception:
            print("  model.predict() 실패:")
            traceback.print_exc()

    print("\n[완료]")
    if train_exc:
        print(f"  학습 중 예외 발생: {type(train_exc).__name__}: {train_exc}")
    else:
        print("  학습 정상 완료")


if __name__ == "__main__":
    main()
