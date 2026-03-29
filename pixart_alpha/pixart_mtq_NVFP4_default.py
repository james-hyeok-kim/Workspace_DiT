import os
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
import gc
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from torchvision.transforms import ToTensor

# Diffusers 및 Quantization
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
# import svdquant as mtq  # 공식 mtq 라이브러리
import modelopt.torch.quantization as mtq

# 데이터셋 및 지표
from datasets import load_dataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from transformers import CLIPModel, CLIPProcessor
import copy

# ==========================================
# 0. 데이터 로드 로직
# ==========================================
def get_prompts(num_samples, args):
    if args.dataset_name == "MJHQ":
        path, split, key = "xingjianleng/mjhq30k", "test", "text"
    else:
        path, split, key = "mit-han-lab/svdquant-datasets", "train", "prompt"

    try:
        dataset = load_dataset(path, split=split, streaming=True)
        prompts = []
        for i, entry in enumerate(dataset):
            if i >= num_samples: break
            prompts.append(entry[key])
        return prompts
    except Exception as e:
        print(f"⚠️ Prompt loading failed: {e}")
        return ["A professional high-quality photo"] * num_samples

# ==========================================
# 1. 메인 실행 함수
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--test_run", action="store_true")
    parser.add_argument("--ref_dir", type=str, default="./ref_images")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--model_path", type=str, default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name", type=str, default="MJHQ", choices=["MJHQ"])
    # 🎯 아래 두 줄을 추가하세요!
    parser.add_argument("--lowrank", type=int, default=32, help="SVDQuant low-rank branch rank")

    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    
    # 🎯 [핵심] 폴더 구조 분리
    # Reference 이미지는 공통으로 사용 (ref_images/MJHQ/)
    dataset_ref_dir = os.path.join(args.ref_dir, args.dataset_name)
    # 결과 이미지는 MTQ 전용 하위 폴더 생성 (results/MJHQ/mtq/)
    dataset_save_dir = os.path.join(args.save_dir, args.dataset_name, "mtq")
    
    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir, exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)

    s_target = 2 if args.test_run else args.num_samples
    prompts = get_prompts(s_target, args)
    s_count = len(prompts)
    p_count = 2 if args.test_run else min(64, s_count)
    t_count = 20

    # ------------------------------------------
    # STEP 1: Reference 이미지 생성 (FP16) - 공통 폴더
    # ------------------------------------------
    if accelerator.is_main_process:
        accelerator.print(f"🌟 [MTQ Mode] Checking Reference images in: {dataset_ref_dir}")
        pipe_ref = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
        for i in range(s_count):
            ref_path = os.path.join(dataset_ref_dir, f"ref_{i}.png")
            if not os.path.exists(ref_path):
                gen = torch.Generator(device=device).manual_seed(42 + i)
                img = pipe_ref(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
                img.save(ref_path)
        del pipe_ref
        torch.cuda.empty_cache()
        gc.collect()
    
    accelerator.wait_for_everyone()

    # ------------------------------------------
    # STEP 2: MTQ 공식 양자화 적용 (NVFP4)
    # ------------------------------------------
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    accelerator.print(f"⚙️ Applying MTQ NVFP4 SVDQuant (lowrank={args.lowrank}, alpha={args.alpha})...")

    # ------------------------------------------
    # STEP 2: MTQ 공식 양자화 적용 (NVFP4)
    # ------------------------------------------
    
    # 1. 설정값 복사 및 사용자 인자 주입
    quant_config = copy.deepcopy(mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
    
    # 'algorithm' 키 아래에 lowrank만 남기고 alpha는 제거합니다.
    if "algorithm" in quant_config:
        quant_config["algorithm"]["lowrank"] = args.lowrank
        # ❌ 아래 줄을 삭제하거나 주입하지 마세요.
        # quant_config["algorithm"]["alpha"] = args.alpha  <- 이 부분이 에러의 원인입니다.
    
    if accelerator.is_main_process:
        print(f"🎯 [CONFIG] Applied Lowrank: {quant_config['algorithm'].get('lowrank')}")
        # alpha는 라이브러리 기본값(0.5)이 자동으로 적용됩니다.

    # 2. Calibration을 위한 forward_loop 정의
    def forward_loop(model):
        for prompt in prompts[:p_count]:
            pipe(
                prompt, 
                num_inference_steps=5, 
                generator=torch.Generator(device=device).manual_seed(42)
            )

    accelerator.print(f"⚙️ Applying MTQ NVFP4 SVDQuant...")
    
    # 3. 양자화 실행
    with torch.no_grad():
        pipe.transformer = mtq.quantize(
            pipe.transformer, 
            quant_config, 
            forward_loop=forward_loop
        )

    accelerator.wait_for_everyone()

    # ------------------------------------------
    # STEP 3: 분산 이미지 생성 및 실시간 지표 업데이트
    # ------------------------------------------
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
    is_m = InceptionScore().to(device)
    fid_m = FrechetInceptionDistance(feature=2048).to(device)

    if accelerator.is_main_process:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    indices = list(range(s_count))
    with accelerator.split_between_processes(indices) as local_indices:
        for i in local_indices:
            gen = torch.Generator(device=device).manual_seed(42 + i)
            q_img = pipe(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
            
            # 🎯 MTQ 전용 폴더에 저장
            save_path = os.path.join(dataset_save_dir, f"sample_{i}.png")
            q_img.save(save_path)
            
            # 지표 업데이트
            ref_path = os.path.join(dataset_ref_dir, f"ref_{i}.png")
            r_img = Image.open(ref_path).convert("RGB")
            q_ten = ToTensor()(q_img).unsqueeze(0).to(device)
            r_ten = ToTensor()(r_img).unsqueeze(0).to(device)
            
            psnr_m.update(q_ten, r_ten)
            ssim_m.update(q_ten, r_ten)
            lpips_m.update(q_ten * 2 - 1, r_ten * 2 - 1)
            
            img_u8 = (q_ten * 255).to(torch.uint8)
            ref_u8 = (r_ten * 255).to(torch.uint8)
            is_m.update(img_u8)
            fid_m.update(ref_u8, real=True)
            fid_m.update(img_u8, real=False)
            
            print(f"📸 GPU {accelerator.process_index} -> MTQ sample_{i}.png", flush=True)

    accelerator.wait_for_everyone()

    # ------------------------------------------
    # STEP 4: 지표 계산 및 결과 저장
    # ------------------------------------------
    res_psnr = psnr_m.compute()
    res_ssim = ssim_m.compute()
    res_lpips = lpips_m.compute()
    res_is, _ = is_m.compute()
    res_fid = fid_m.compute()

    if accelerator.is_main_process:
        all_clip_scores = []
        for i in range(s_count):
            q_img = Image.open(os.path.join(dataset_save_dir, f"sample_{i}.png")).convert("RGB")
            inputs = clip_processor(text=[prompts[i]], images=q_img, return_tensors="pt", padding=True).to(device)
            all_clip_scores.append(float(clip_model(**inputs).logits_per_image.item()))

        final_res = {
            "method": "MTQ_Official_NVFP4",
            "dataset": args.dataset_name,
            "averages": {
                "psnr": float(res_psnr), "ssim": float(res_ssim),
                "lpips": float(res_lpips), "clip": np.mean(all_clip_scores)
            },
            "dist_metrics": {"FID": float(res_fid), "IS": float(res_is)}
        }
        
        # 🎯 메트릭 파일도 mtq 폴더 안에 저장
        with open(os.path.join(dataset_save_dir, "metrics.json"), "w") as f:
            json.dump(final_res, f, indent=4)
            
        print(f"\n✅ MTQ Evaluation Finished! (Saved to: {dataset_save_dir})")
        print(f"FID: {res_fid:.2f} | CLIP: {np.mean(all_clip_scores):.2f}")

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()