#!/usr/bin/env python3
"""
eval_convergence.py
──────────────────
저장된 이미지로부터 FID / IS를 N별로 재계산합니다.
이미지 생성 없이 순수 평가만 수행 — 모델 로드 불필요.

사용법:
    python3 eval_convergence.py \
        --sample_dir results/MJHQ/deepcache/interval1_s8_e20_gs4.5_steps20 \
        --ref_dir    ref_images/MJHQ_steps20 \
        --method     "NVFP4 20-step" \
        --ns         20,100,250,500,750,1000 \
        --time_per_img 3.45 \
        --baseline_time 3.45 \
        --out        results/convergence/nvfp4_20step.csv
"""

import argparse
import os
import csv
from pathlib import Path

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def load_uint8(path: str, device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = (to_tensor(img) * 255).to(torch.uint8).unsqueeze(0)
    return t.to(device)


def eval_at_n(sample_dir: str, ref_dir: str, n: int, device) -> dict:
    fid_m = FrechetInceptionDistance(feature=2048, reset_real_features=True).to(device)
    is_m  = InceptionScore(splits=1).to(device)

    for i in range(n):
        s_path = os.path.join(sample_dir, f"sample_{i}.png")
        r_path = os.path.join(ref_dir,    f"ref_{i}.png")

        if not os.path.exists(s_path):
            raise FileNotFoundError(f"sample_{i}.png not found in {sample_dir}")
        if not os.path.exists(r_path):
            raise FileNotFoundError(f"ref_{i}.png not found in {ref_dir}")

        s_t = load_uint8(s_path, device)
        r_t = load_uint8(r_path, device)

        fid_m.update(r_t, real=True)
        fid_m.update(s_t, real=False)
        is_m.update(s_t)

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] loaded", flush=True)

    fid_val      = float(fid_m.compute())
    is_val, _    = is_m.compute()
    is_val       = float(is_val)

    return {"fid": round(fid_val, 3), "is": round(is_val, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir",     required=True,
                        help="Directory with sample_0.png ... sample_N-1.png")
    parser.add_argument("--ref_dir",        required=True,
                        help="Directory with ref_0.png ... ref_N-1.png")
    parser.add_argument("--method",         required=True,
                        help="Method label for CSV output")
    parser.add_argument("--ns",             default="20,100,250,500,750,1000",
                        help="Comma-separated sample counts to evaluate")
    parser.add_argument("--time_per_img",   type=float, default=0.0,
                        help="Measured time per image (sec) for this method")
    parser.add_argument("--baseline_time",  type=float, default=0.0,
                        help="Baseline time per image (sec) to compute speedup")
    parser.add_argument("--out",            required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ns = [int(x.strip()) for x in args.ns.split(",")]

    # Check max available images
    max_available = sum(
        1 for n in range(max(ns))
        if os.path.exists(os.path.join(args.sample_dir, f"sample_{n}.png"))
    )
    print(f"Method: {args.method}")
    print(f"Sample dir: {args.sample_dir}  ({max_available} images found)")
    print(f"Ref dir:    {args.ref_dir}")
    print(f"Evaluating N = {ns}")
    print()

    speedup = (args.baseline_time / args.time_per_img) if args.time_per_img > 0 else None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []

    for n in ns:
        if n > max_available:
            print(f"  N={n}: skipped (only {max_available} images available)")
            continue

        print(f"── N={n} ──────────────────────────────────────")
        metrics = eval_at_n(args.sample_dir, args.ref_dir, n, device)
        row = {
            "method":        args.method,
            "N":             n,
            "FID":           metrics["fid"],
            "IS":            metrics["is"],
            "time_per_img":  round(args.time_per_img, 3),
            "speedup":       round(speedup, 3) if speedup else "",
        }
        rows.append(row)
        print(f"  FID={metrics['fid']:.3f}  IS={metrics['is']:.4f}  "
              f"speedup={speedup:.2f}x" if speedup else
              f"  FID={metrics['fid']:.3f}  IS={metrics['is']:.4f}")
        print()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "N", "FID", "IS", "time_per_img", "speedup"])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
