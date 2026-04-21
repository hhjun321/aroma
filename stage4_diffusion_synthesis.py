"""stage4_diffusion_synthesis.py — Diffusion 기반 결함 합성 (Option A).

Stage 4 MPB를 완전 교체. placement_map.json(Stage 3) ROI 좌표를 인페인팅 마스크로
변환하여 SD Inpainting + ControlNet으로 good 이미지에 직접 결함을 합성한다.

MPB 블러 문제 근본 해결:
    MPB: 픽셀 복사·붙여넣기 → Poisson 경계 보정 → 미세 블러 발생
    Diffusion: Latent 인코딩 → ROI 영역 Denoising → 픽셀 디코딩 (블러 없음)

CASDA 방식 준수 — ControlNet 입력 설계:
    Stage 2 변형 이미지(elastic warp)를 ControlNet 입력으로 사용하지 않는다.
    CASDA는 실제 결함 이미지(Stage 1b seed_path)를 입력으로 검증하였으므로,
    동일하게 Stage 1b 원본 seed 이미지에서 Canny 엣지를 추출한다.
    Stage 2 변형 이미지는 placement_map의 ROI 위치·크기 계산에만 사용된다.

CASDA 검증 파라미터:
    strength=0.7, guidance_scale=7.5, num_inference_steps=30
    controlnet_conditioning_scale=0.7 (학습 1.0 → 추론 0.7, 아티팩트 방지)

출력 형식: stage4_mpb_synthesis.py와 동일
    {output_dir}/defect/{image_id}.png
    {output_dir}/defect/{image_id}_mask.png
"""
import argparse
import logging
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from utils.io import load_json, validate_dir, validate_file
from utils.prompt_generator import PromptGenerator

logger = logging.getLogger(__name__)

_PIPELINE = None  # GPU 파이프라인 프로세스당 1회 로드


def _get_pipeline(controlnet_model: Optional[str], device: str):
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetInpaintPipeline,
        DDIMScheduler,
    )

    if controlnet_model and Path(controlnet_model).exists():
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch.float16
        )
        logger.info(f"Fine-tuned ControlNet loaded: {controlnet_model}")
    else:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
        )
        logger.info("Pretrained ControlNet (lllyasviel/sd-controlnet-canny) loaded")

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)

    # CASDA 검증: DDIM 스케줄러 (결정론적 생성)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    _PIPELINE = pipe
    return pipe


def _make_roi_mask(placements: List[dict], h: int, w: int) -> np.ndarray:
    """placement_map 좌표에서 바이너리 ROI 마스크 생성."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for p in placements:
        defect_path = p.get("defect_path", "")
        patch = cv2.imread(defect_path)
        if patch is None:
            continue
        ph, pw = patch.shape[:2]
        scale = float(p.get("scale", 1.0))
        ph = max(1, int(ph * scale))
        pw = max(1, int(pw * scale))
        x = max(0, min(w - pw, int(p.get("x", 0))))
        y = max(0, min(h - ph, int(p.get("y", 0))))
        mask[y : y + ph, x : x + pw] = 255
    return mask


def _canny_control_image(defect_path: str, size: tuple) -> Image.Image:
    """Stage 1b seed_path 원본 결함 이미지 → Canny 엣지 → ControlNet control image."""
    patch = cv2.imread(defect_path, cv2.IMREAD_GRAYSCALE)
    if patch is None:
        return Image.new("RGB", size, 128)
    edges = cv2.Canny(patch, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb).resize(size, Image.LANCZOS)


def _synthesize_one(
    pipe,
    bg_path: str,
    placements: List[dict],
    seed_profile: dict,
    control_pil: "Image.Image",
    prompt_gen: "PromptGenerator",
    output_dir: str,
    fmt: str,
    resolution: int,
    num_inference_steps: int,
    strength: float,
    guidance_scale: float,
    conditioning_scale: float,
    seed: int,
) -> List[str]:
    bg_pil = Image.open(bg_path).convert("RGB")
    orig_w, orig_h = bg_pil.size
    bg_resized = bg_pil.resize((resolution, resolution), Image.LANCZOS)

    # ROI 마스크 (resolution 기준으로 좌표 스케일)
    sx, sy = resolution / orig_w, resolution / orig_h
    scaled = [
        {**p,
         "x": int(p.get("x", 0) * sx),
         "y": int(p.get("y", 0) * sy),
         "scale": float(p.get("scale", 1.0)) * min(sx, sy)}
        for p in placements
    ]
    roi_mask = _make_roi_mask(scaled, resolution, resolution)
    mask_pil = Image.fromarray(roi_mask)

    # 텍스트 프롬프트: Stage 1b seed_profile → PromptGenerator (CASDA 방식)
    defect_metrics = {
        "linearity": seed_profile.get("linearity", 0.5),
        "solidity": seed_profile.get("solidity", 0.5),
        "aspect_ratio": seed_profile.get("aspect_ratio", 1.0),
    }
    bg_type = (
        placements[0].get("matched_background_type", "smooth") if placements else "smooth"
    )
    suit_score = (
        float(placements[0].get("suitability_score", 0.5)) if placements else 0.5
    )
    prompt = prompt_gen.generate_prompt(
        defect_subtype=seed_profile.get("subtype", "general"),
        background_type=bg_type,
        stability_score=0.7,
        defect_metrics=defect_metrics,
        suitability_score=suit_score,
    )
    negative_prompt = prompt_gen.generate_negative_prompt()

    # SD Inpainting + ControlNet 추론
    device_type = str(pipe.device).split(":")[0]
    generator = torch.Generator(device=str(pipe.device)).manual_seed(seed)

    with torch.autocast(device_type):
        result_pil = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=bg_resized,
            mask_image=mask_pil,
            control_image=control_pil,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=conditioning_scale,
            generator=generator,
        ).images[0]

    # grayscale 후처리: 불필요한 RGB 아티팩트 제거 후 RGB 복원
    result_pil = result_pil.convert("L").convert("RGB")

    # 원본 해상도로 복원
    result_pil = result_pil.resize((orig_w, orig_h), Image.LANCZOS)

    image_id = Path(bg_path).stem
    out_path = Path(output_dir)
    written: List[str] = []

    if fmt == "cls":
        cls_dir = out_path / "defect"
        cls_dir.mkdir(parents=True, exist_ok=True)
        img_path = str(cls_dir / f"{image_id}.png")
        result_pil.save(img_path)
        written.append(img_path)

        # 마스크 원본 해상도로 저장 (Stage 5 품질 필터링용)
        final_mask = _make_roi_mask(placements, orig_h, orig_w)
        if np.any(final_mask):
            cv2.imwrite(str(cls_dir / f"{image_id}_mask.png"), final_mask)

    return written


def run_synthesis(
    image_dir: str,
    placement_map: str,
    output_dir: str,
    seed_profile_path: str,
    format: str = "cls",
    controlnet_model: Optional[str] = None,
    device: str = "cuda",
    resolution: int = 512,
    num_inference_steps: int = 30,
    strength: float = 0.7,
    guidance_scale: float = 7.5,
    conditioning_scale: float = 0.7,
    seed: int = 42,
    max_images_per_seed: Optional[int] = None,
) -> None:
    """단일 seed에 대한 Diffusion 합성 실행."""
    validate_dir(image_dir, name="Background image_dir")
    validate_file(placement_map, name="Stage 3 placement_map.json")
    validate_file(seed_profile_path, name="Stage 1b seed_profile.json")

    seed_profile = load_json(seed_profile_path)
    entries: List[dict] = load_json(placement_map)
    img_dir = Path(image_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 유효 entry만 추린 후 수량 제한 (재현성을 위해 seed 고정)
    valid_entries = [e for e in entries if e.get("placements")]
    if max_images_per_seed is not None and len(valid_entries) > max_images_per_seed:
        rng = random.Random(seed)
        valid_entries = rng.sample(valid_entries, max_images_per_seed)

    pipe = _get_pipeline(controlnet_model, device)

    # Canny와 PromptGenerator는 seed당 1회만 계산 (이미지마다 재계산 방지)
    control_pil = _canny_control_image(
        seed_profile.get("seed_path", ""), (resolution, resolution)
    )
    prompt_gen = PromptGenerator(style="technical")

    for i, entry in enumerate(valid_entries):
        image_id: str = entry["image_id"]
        placements: List[dict] = entry.get("placements", [])
        if not placements:
            continue
        bg_path = img_dir / f"{image_id}.png"
        if not bg_path.exists():
            continue
        _synthesize_one(
            pipe=pipe,
            bg_path=str(bg_path),
            placements=placements,
            seed_profile=seed_profile,
            control_pil=control_pil,
            prompt_gen=prompt_gen,
            output_dir=output_dir,
            fmt=format,
            resolution=resolution,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            conditioning_scale=conditioning_scale,
            seed=seed + i,
        )

    logger.info(f"Stage 4 Diffusion synthesis complete → {output_dir}")


def run_synthesis_batch(
    image_dir: str,
    seed_placement_maps: List[tuple],
    output_root: str,
    cat_dir: str,
    format: str = "cls",
    controlnet_model: Optional[str] = None,
    device: str = "cuda",
    resolution: int = 512,
    num_inference_steps: int = 30,
    strength: float = 0.7,
    guidance_scale: float = 7.5,
    conditioning_scale: float = 0.7,
    seed: int = 42,
    max_images_per_seed: Optional[int] = None,
) -> None:
    """카테고리 레벨 배치 합성. GPU 단일 점유이므로 순차 처리."""
    validate_dir(image_dir, name="Background image_dir")

    pipe = _get_pipeline(controlnet_model, device)

    for seed_id, pm_path in seed_placement_maps:
        validate_file(pm_path, name=f"placement_map ({seed_id})")

        # seed_profile.json: cat_dir/stage1b_output/{seed_id}/seed_profile.json
        profile_path = Path(cat_dir) / "stage1b_output" / seed_id / "seed_profile.json"
        if not profile_path.exists():
            logger.warning(f"seed_profile.json not found for {seed_id}, skipping")
            continue

        seed_profile = load_json(str(profile_path))
        entries: List[dict] = load_json(str(pm_path))
        img_dir = Path(image_dir)
        seed_out = Path(output_root) / seed_id

        # 유효 entry만 추린 후 수량 제한 (재현성을 위해 seed 고정)
        valid_entries = [e for e in entries if e.get("placements")]
        if max_images_per_seed is not None and len(valid_entries) > max_images_per_seed:
            rng = random.Random(seed)
            valid_entries = rng.sample(valid_entries, max_images_per_seed)
            logger.info(f"  [{seed_id}] {len(entries)} → {max_images_per_seed}장 서브샘플")

        # Canny와 PromptGenerator는 seed당 1회만 계산 (이미지마다 재계산 방지)
        control_pil = _canny_control_image(
            seed_profile.get("seed_path", ""), (resolution, resolution)
        )
        prompt_gen = PromptGenerator(style="technical")

        for i, entry in enumerate(valid_entries):
            image_id: str = entry["image_id"]
            placements: List[dict] = entry.get("placements", [])
            bg_path = img_dir / f"{image_id}.png"
            if not bg_path.exists():
                continue
            _synthesize_one(
                pipe=pipe,
                bg_path=str(bg_path),
                placements=placements,
                seed_profile=seed_profile,
                control_pil=control_pil,
                prompt_gen=prompt_gen,
                output_dir=str(seed_out),
                fmt=format,
                resolution=resolution,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=guidance_scale,
                conditioning_scale=conditioning_scale,
                seed=seed + i,
            )

        logger.info(f"  [{seed_id}] → {seed_out}")

    logger.info(f"Stage 4 Diffusion batch synthesis complete → {output_root}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4 (Option A): Diffusion 기반 결함 합성 — MPB 교체"
    )
    parser.add_argument("--placement_map", required=True, help="Stage 3 placement_map.json")
    parser.add_argument("--image_dir", required=True, help="배경 이미지 디렉터리")
    parser.add_argument("--output_dir", required=True, help="출력 루트 디렉터리")
    parser.add_argument("--seed_profile", required=True, help="Stage 1b seed_profile.json")
    parser.add_argument("--format", default="cls", choices=["cls"], help="출력 포맷")
    parser.add_argument("--controlnet_model", default=None,
                        help="파인튜닝된 ControlNet 경로 (없으면 pretrained canny 사용)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--conditioning_scale", type=float, default=0.7,
                        help="CASDA 검증값: 학습 1.0 → 추론 0.7")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()
    run_synthesis(
        image_dir=args.image_dir,
        placement_map=args.placement_map,
        output_dir=args.output_dir,
        seed_profile_path=args.seed_profile,
        format=args.format,
        controlnet_model=args.controlnet_model,
        device=args.device,
        resolution=args.resolution,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        conditioning_scale=args.conditioning_scale,
        seed=args.seed,
    )
