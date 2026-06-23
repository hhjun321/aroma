#!/usr/bin/env python3
"""
select_context_prototypes.py — CLIP → MiniBatchKMeans → medoid context selection.

Severstal has thousands of normal images. Using all of them (or a 1:1 head-slice)
as synthesis backgrounds / baseline negatives over-represents whatever the file
ordering happens to surface. Instead we pick **K representative context
prototypes** that approximate the *distribution* of the normal pool:

    all normal images
      → CLIP Vision Encoder embedding (N × 512)
      → L2 normalize
      → PCA 512→D (optional, --pca 64; --pca 0 to skip)
      → MiniBatchKMeans(n_clusters=K, random_state=seed)
      → per-cluster medoid (real image nearest the centroid, stable tie-break)
      → K context prototypes

This gives a (b) distribution-matched normal set (CASDA-comparable by distribution,
not by an explicit list intersection).

Outputs:
    <output>/context_prototypes.json   — { "prototypes": [filenames...], ... }
    <output>/context/                   — the K selected images (symlink or copy)

Deterministic: KMeans random_state=seed, medoid tie-break by sorted path.

CLIP backend: open_clip preferred, transformers CLIPModel fallback. Both imports
are deferred into the embedding function so importing this module never requires
torch/CLIP/sklearn to be installed.

Usage (Colab, GPU recommended):
    !python $AROMA_SCRIPTS/select_context_prototypes.py \\
        --image_dir /content/drive/MyDrive/data/Aroma/severstal/train/good \\
        --k 1000 --pca 64 --seed 42 \\
        --output /content/drive/MyDrive/data/Aroma/severstal/context_select
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
_DEFAULT_MODEL = "ViT-B-32"
_DEFAULT_PRETRAINED = "openai"


def _list_images(image_dir: str) -> List[str]:
    d = Path(image_dir)
    if not d.exists():
        return []
    return sorted(str(p) for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMG_EXTS)


# ---------------------------------------------------------------------------
# CLIP embedding (deferred imports; graceful errors)
# ---------------------------------------------------------------------------

def _embed_open_clip(paths: List[str], model_name: str, batch_size: int):
    """Embed via open_clip. Returns (N, D) float32 or raises ImportError."""
    import open_clip  # type: ignore
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=_DEFAULT_PRETRAINED
    )
    model = model.to(device).eval()

    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            x = torch.stack(imgs).to(device)
            emb = model.encode_image(x)
            feats.append(emb.float().cpu().numpy())
            print(f"  [open_clip] embedded {min(i + batch_size, len(paths))}/{len(paths)}")
    return np.concatenate(feats, axis=0).astype(np.float32)


def _embed_transformers(paths: List[str], batch_size: int):
    """Embed via transformers CLIPModel. Returns (N, D) float32 or raises ImportError."""
    import torch  # type: ignore
    from PIL import Image  # type: ignore
    from transformers import CLIPModel, CLIPProcessor  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            emb = model.get_image_features(**inputs)
            feats.append(emb.float().cpu().numpy())
            print(f"  [transformers] embedded {min(i + batch_size, len(paths))}/{len(paths)}")
    return np.concatenate(feats, axis=0).astype(np.float32)


def embed_clip(paths: List[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Embed images with CLIP. open_clip preferred, transformers fallback.

    Raises RuntimeError with an actionable message if neither backend is usable.
    """
    try:
        return _embed_open_clip(paths, model_name, batch_size)
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover — surface non-import failures too
        print(f"  [open_clip] failed ({exc}); trying transformers fallback")
    # transformers fallback hardcodes openai/clip-vit-base-patch32; warn if the
    # requested model differs so the run isn't silently non-reproducible.
    if model_name not in ("ViT-B-32", "ViT-B/32", "openai/clip-vit-base-patch32"):
        print(
            f"  [WARN] transformers fallback ignores --model {model_name!r}; "
            f"using openai/clip-vit-base-patch32. Install open_clip to honor it."
        )
    try:
        return _embed_transformers(paths, batch_size)
    except ImportError:
        pass
    raise RuntimeError(
        "No CLIP backend available. Install one of:\n"
        "  pip install open_clip_torch torch\n"
        "  pip install transformers torch"
    )


# ---------------------------------------------------------------------------
# Selection pipeline
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def select_prototypes(
    embeddings: np.ndarray,
    paths: List[str],
    k: int,
    pca_dim: int,
    seed: int,
) -> Tuple[List[int], np.ndarray]:
    """Return (selected_indices, reduced_embeddings).

    Pipeline: L2 normalize → PCA(opt) → MiniBatchKMeans(k) → per-cluster medoid.
    medoid = real sample nearest its cluster centroid (L2), tie-break by index.
    """
    from sklearn.cluster import MiniBatchKMeans  # type: ignore

    n = embeddings.shape[0]
    feats = _l2_normalize(embeddings.astype(np.float32))

    if pca_dim and pca_dim > 0 and pca_dim < feats.shape[1] and n > pca_dim:
        from sklearn.decomposition import PCA  # type: ignore

        feats = PCA(n_components=pca_dim, random_state=seed).fit_transform(feats)
        feats = feats.astype(np.float32)

    eff_k = min(k, n)
    if eff_k <= 0:
        return [], feats
    if eff_k == n:
        # Every image is its own prototype (deterministic order).
        return list(range(n)), feats

    km = MiniBatchKMeans(n_clusters=eff_k, random_state=seed, n_init="auto")
    labels = km.fit_predict(feats)
    centers = km.cluster_centers_

    selected: List[int] = []
    for c in range(eff_k):
        member_idx = np.where(labels == c)[0]
        if member_idx.size == 0:
            continue
        diffs = feats[member_idx] - centers[c]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        # Stable tie-break: smallest distance, then smallest original index.
        best_local = sorted(
            range(member_idx.size), key=lambda j: (float(d2[j]), int(member_idx[j]))
        )[0]
        selected.append(int(member_idx[best_local]))

    selected = sorted(set(selected))
    return selected, feats


def _stage_selected(paths: List[str], indices: List[int], context_dir: Path) -> None:
    context_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        src = paths[idx]
        dst = context_dir / Path(src).name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            os.symlink(os.path.abspath(src), dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)


def run(
    image_dir: str,
    k: int,
    pca: int,
    output: str,
    seed: int,
    model: str,
    batch_size: int = 64,
) -> dict:
    paths = _list_images(image_dir)
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    print(f"[context] embedding {len(paths)} images (model={model})")

    embeddings = embed_clip(paths, model, batch_size=batch_size)
    print(f"[context] embeddings shape={embeddings.shape}")

    indices, _feats = select_prototypes(embeddings, paths, k=k, pca_dim=pca, seed=seed)
    selected_paths = [paths[i] for i in indices]
    selected_names = [Path(p).name for p in selected_paths]
    print(f"[context] selected {len(selected_names)} prototypes (requested k={k})")

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    context_dir = out / "context"
    _stage_selected(paths, indices, context_dir)

    payload = {
        "image_dir": image_dir,
        "model": model,
        "k_requested": k,
        "k_selected": len(selected_names),
        "pca": pca,
        "seed": seed,
        "n_source": len(paths),
        "prototypes": selected_names,
        "prototype_paths": selected_paths,
    }
    (out / "context_prototypes.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[context] wrote context_prototypes.json + context/ → {out}")
    return payload


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CLIP→MiniBatchKMeans→medoid context prototype selection"
    )
    p.add_argument("--image_dir", required=True,
                   help="Normal image directory (e.g. .../severstal/train/good)")
    p.add_argument("--k", type=int, default=1000,
                   help="Number of context prototypes (default: 1000)")
    p.add_argument("--pca", type=int, default=64,
                   help="PCA dim before KMeans; 0 = skip PCA (default: 64)")
    p.add_argument("--output", required=True,
                   help="Output root (writes context_prototypes.json + context/)")
    p.add_argument("--seed", type=int, default=42,
                   help="Deterministic seed for KMeans/PCA (default: 42)")
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help=f"CLIP model name (default: {_DEFAULT_MODEL})")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Embedding batch size (default: 64)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run(
        image_dir=args.image_dir,
        k=args.k,
        pca=args.pca,
        output=args.output,
        seed=args.seed,
        model=args.model,
        batch_size=args.batch_size,
    )
    print("Done.")


if __name__ == "__main__":
    main()
