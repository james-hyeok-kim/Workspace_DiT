import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from diffusers import PixArtAlphaPipeline
import modelopt.torch.quantization as mtq

# 평가 메트릭 라이브러리
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

def parse_args():
    parser = argparse.ArgumentParser(description="SVDQuant Style Evaluation for PixArt")
    parser.add_argument("--model_id", type=str, default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name", type=str, default="MJHQ", choices=["MJHQ", "sDCI"])
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--quant_method", type=str, default="INT8_DEFAULT_CFG")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="eval_results")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. 모델 로드 (Teacher 모델 준비: FP32/BF16) [cite: 328]
    print(f"🚀 Loading Teacher Model (FP32) for Similarity Reference...")
    teacher_pipe = PixArtAlphaPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32).to(device)
    teacher_pipe.set_progress_bar_config(disable=True)

    # 2. 데이터셋 설정 (MJHQ 또는 sDCI) 
    if args.dataset_name == "MJHQ":
        # MJHQ-30K: Midjourney 고화질 데이터셋 [cite: 748]
        ds = load_dataset("J_Min/MJHQ-30K", split="train", streaming=True)
        caption_key = "caption"
    else:
        # sDCI: 상세 캡션 데이터셋의 요약 버전 [cite: 755]
        ds = load_dataset("zzli/sDCI", split="train", streaming=True) # 경로 확인 필요
        caption_key = "summarized_caption"

    # 3. 메트릭 초기화 [cite: 769, 770]
    fid = FrechetInceptionDistance(feature=2048).to(device)
    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)
    ssim_metric = SSIM(data_range=1.0).to(device)

    # 4. 평가 루프
    print(f"📊 Starting Evaluation on {args.dataset_name} ({args.num_samples} samples)...")
    
    prompts = []
    for i, data in enumerate(ds):
        if i >= args.num_samples: break
        prompts.append(data[caption_key])
        
        # 실제 이미지 FID 업데이트
        real_img = data["image"].convert("RGB").resize((args.image_size, args.image_size))
        real_tensor = torch.from_numpy(np.array(real_img)).permute(2, 0, 1).unsqueeze(0).to(device)
        fid.update(real_tensor.to(torch.uint8), real=True)

    # 5. Teacher 생성 vs Student(양자화) 생성 비교
    # 5-1. Teacher 결과 생성 (메모리 절약을 위해 먼저 수행)
    teacher_tensors = []
    print("🎨 Generating Teacher Images (Reference)...")
    for prompt in tqdm(prompts):
        with torch.no_grad():
            img_t = teacher_pipe(prompt=prompt, num_inference_steps=20, size=args.image_size).images[0]
            teacher_tensors.append(torch.from_numpy(np.array(img_t)).permute(2, 0, 1).float().div(255.0))

    # 5-2. 양자화 적용 (Student 모델로 변환) [cite: 29]
    print(f"🛠 Applying Quantization: {args.quant_method}...")
    if hasattr(mtq, args.quant_method):
        config = getattr(mtq, args.quant_method)
        teacher_pipe.transformer = mtq.quantize(teacher_pipe.transformer, config)
    student_pipe = teacher_pipe # 양자화가 적용된 상태

    # 5-3. Student 결과 생성 및 Metric 계산
    print("🎨 Generating Student Images & Computing Metrics...")
    for i, prompt in enumerate(tqdm(prompts)):
        with torch.no_grad():
            img_s = student_pipe(prompt=prompt, num_inference_steps=20, size=args.image_size).images[0]
            student_tensor = torch.from_numpy(np.array(img_s)).permute(2, 0, 1).float().div(255.0).to(device)
            teacher_tensor = teacher_tensors[i].to(device)

            # Quality: CLIP Score (Text-Image Alignment) 
            clip_score.update(student_tensor.unsqueeze(0), prompt)
            
            # Quality: FID [cite: 329]
            fid.update((student_tensor * 255).unsqueeze(0).to(torch.uint8), real=False)
            
            # Similarity: SSIM (Student vs Teacher) 
            ssim_metric.update(student_tensor.unsqueeze(0), teacher_tensor.unsqueeze(0))

    # 6. 최종 결과 출력
    res_fid = fid.compute().item()
    res_clip = clip_score.compute().item()
    res_ssim = ssim_metric.compute().item()

    print("\n" + "="*50)
    print(f"✅ Final Results for {args.dataset_name}")
    print(f"⭐ [Quality] CLIP Score (C.SCR): {res_clip:.4f}")
    print(f"⭐ [Quality] FID: {res_fid:.4f}")
    print(f"⭐ [Similarity] SSIM (Teacher vs Student): {res_ssim:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()