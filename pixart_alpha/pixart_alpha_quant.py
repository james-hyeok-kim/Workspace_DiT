import os
# ✅ [핵심 1] 메모리 단편화 방지
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ✅ [핵심 1] 메모리 단편화 방지
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
import gc
import argparse
import numpy as np
import random
import json
import random
import json
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from diffusers import PixArtAlphaPipeline
import modelopt.torch.quantization as mtq
import copy
import torch.nn.functional as F

# ✅ HAS_IR 변수 정의 (전역 선언)
HAS_IR = False
try:
    import ImageReward as RM
    HAS_IR = True
except ImportError:
    print("⚠️ Warning: ImageReward package not found. IR will be 0.0.")

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from transformers import CLIPModel, CLIPProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="SVDQuant + Differential Fine-tuning Replication")
    parser.add_argument("--model_id", type=str, default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--dataset_name", type=str, default="MJHQ", choices=["MJHQ", "sDCI"])
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--quant_method", type=str, default="TERNARY")
    parser.add_argument("--svd_dtype", type=str, default="fp16", choices=["fp16", "fp8"], help="Data type for SVD branches")
    parser.add_argument("--fp8_format", type=str, default="e4m3", choices=["e4m3", "e5m2", "hybrid"], 
                        help="FP8 format: e4m3 (precision) or e5m2 (range)")
    parser.add_argument("--block_size", type=int, default=128, choices=[16,32,64,128,256], help="Block quantization Size")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lowrank", type=int, default=64)
    parser.add_argument("--do_diff_tuning", action="store_true", help="Enable Differential Fine-tuning after quantization")
    parser.add_argument("--test_run", action="store_true", help="Minimal samples for debugging")
    return parser.parse_args()

def calculate_clip_metrics(clip_model, processor, image_pil, text, device):
    clip_model.to(device)
    inputs = processor(text=[text, "Good photo", "Bad photo"], images=image_pil, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        img_embed = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        txt_embed = outputs.text_embeds[0:1] / outputs.text_embeds[0:1].norm(p=2, dim=-1, keepdim=True)
        clip_score = (img_embed @ txt_embed.T).item() * 100
        clip_iqa = outputs.logits_per_image[:, 1:].softmax(dim=1)[:, 0].item()
    clip_model.cpu()
    return max(clip_score, 0), clip_iqa

def blockwise_fake_quantize(tensor, block_size=128, fp8_type=torch.float8_e4m3fn):
    """fp8_type에 따라 동적 스케일을 적용하는 블록 단위 양자화 (Outlier 방어 적용)"""
    orig_shape = tensor.shape
    flat = tensor.flatten()
    num_el = flat.numel()
    
    # 1. 패딩 및 리셰이프
    pad_len = (block_size - (num_el % block_size)) % block_size
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, device=tensor.device, dtype=tensor.dtype)])
    blocks = flat.view(-1, block_size)

    blocks_f32 = blocks.float()
    q_max = 448.0 if fp8_type == torch.float8_e4m3fn else 57344.0

    # ✅ [핵심 해결책] Outlier-Aware Scaling
    # 무조건 max()를 잡지 않고, 블록 내 상위 5%의 튀는 값을 제외한 '안전한 최대값'을 찾습니다.
    k = max(1, int(block_size * 0.05)) # block_size가 32면 상위 1~2개 무시
    
    # 절대값 기준으로 내림차순 정렬 후, k번째 값을 해당 블록의 최대값으로 간주
    sorted_vals, _ = torch.sort(blocks_f32.abs(), dim=1, descending=True)
    block_max = sorted_vals[:, k:k+1] 
    
    # 이렇게 하면 튀는 아웃라이어 1~2개는 q_max로 예쁘게 잘려나가고(Clamp), 
    # 나머지 95%의 파라미터들은 정밀도를 완벽하게 보존할 수 있습니다.
    scales = (block_max / q_max).clamp(min=1e-5) 
    
    # 양자화 및 클램핑 (아웃라이어들은 여기서 잘림)
    quant = (blocks_f32 / scales).clamp(-q_max, q_max)
    
    # FP8 시뮬레이션 및 복원
    quant_fp8 = quant.to(fp8_type)
    dequant = quant_fp8.to(tensor.dtype) * scales.to(tensor.dtype)
    
    return dequant.view(-1)[:num_el].view(orig_shape)

def analyze_svd_distribution(tensor):
    """
    SVD 텐서의 분포를 분석하여 적절한 FP8 포맷을 추천합니다.
    """
    with torch.no_grad():
        abs_data = tensor.float().abs()
        max_val = abs_data.max().item()
        
        # Kurtosis 계산
        mean_val = abs_data.mean()
        var_val = abs_data.var()
        if var_val == 0: return "e4m3", max_val, 0.0 # 변화가 없는 텐서
        
        kurtosis = ((abs_data - mean_val)**4).mean() / (var_val**2)
        
        # 기준: 최대값이 FP8_E4M3 범위를 위협하거나 꼬리가 너무 길면 E5M2 선택
        if max_val > 400 or kurtosis > 5.0:
            return "e5m2", max_val, kurtosis
        else:
            return "e4m3", max_val, kurtosis
        
def apply_hybrid_blockwise_quant(model, args, device):
    if args.svd_dtype != "fp8": return
    
    # ✅ [긴급 디버깅] 모델의 실제 파라미터 이름 전체를 텍스트 파일로 추출!
    print("🔍 [Debug] 현재 모델의 진짜 파라미터 이름 목록을 추출합니다...")
    os.makedirs(args.save_dir, exist_ok=True)
    param_names = [n for n, p in model.named_parameters()]
    
    debug_file_path = os.path.join(args.save_dir, "model_param_names.txt")
    with open(debug_file_path, "w") as f:
        f.write("\n".join(param_names))
    print(f"📄 모델 파라미터 목록이 저장되었습니다: {debug_file_path}")
    
    # 앞부분 이름 일부를 화면에도 출력
    print(f"👉 샘플 이름 확인: {param_names[:5]}")

    stats = {"e4m3": 0, "e5m2": 0}
    layer_logs = {}  
    print(f"🔍 [Profiler] Applying Hybrid Block-wise FP8 to Protected SVD branches...")

    for n, p in model.named_parameters():
        # ⚠️ 현재 이 조건에 걸리는 파라미터가 없어서 문제 발생 중
        # ✅ [핵심 수정] high_rank가 아니라 실제 어텐션 및 MLP의 가중치(.weight)를 정밀 타겟팅
        is_target_layer = "transformer_blocks" in n and n.endswith(".weight") and "scale_shift_table" not in n
        if is_target_layer:
            if args.fp8_format == "hybrid":
                decision, mx, kt = analyze_svd_distribution(p.data)
                target_fp8 = torch.float8_e5m2 if decision == "e5m2" else torch.float8_e4m3fn
            else:
                target_fp8 = torch.float8_e5m2 if args.fp8_format == "e5m2" else torch.float8_e4m3fn
                decision = args.fp8_format
                mx = p.data.abs().max().item()

            layer_logs[n] = {
                "fp8_format": decision.upper(),
                "max_abs_val": round(float(mx), 4),
                "shape": list(p.shape)
            }

            p.assigned_fp8_format = target_fp8
            p.data.copy_(blockwise_fake_quantize(p.data, block_size=args.block_size, fp8_type=target_fp8))
            p.requires_grad = args.do_diff_tuning
            
            stats[decision if decision in stats else "e4m3"] += 1

    print(f"\n✅ SVD Hybrid Summary: E4M3({stats['e4m3']}), E5M2({stats['e5m2']})")

    log_file_path = os.path.join(args.save_dir, f"layer_quant_log_{args.fp8_format}_rank{args.lowrank}.json")
    with open(log_file_path, "w") as f:
        json.dump(layer_logs, f, indent=4)

    print(f"📁 [Log Saved] Layer-wise quantization details saved to: {log_file_path}")

def run_diff_tuning(model, pipe, prompts, args, device, target_dtype):
    print(f"🔥 [Step 2] Starting FP8-Aware Master-Weight Tuning (Rank {args.lowrank})...")
    
    # 1️⃣ 원본 파라미터(FP8) 수집
    optimizer_params = [p for n, p in model.named_parameters() if ("high_rank" in n or "low_rank" in n) and p.requires_grad]
    
    if not optimizer_params:
        print("⚠️ Warning: No SVD parameters found! Check requires_grad setting.")
        return

    # 2️⃣ ⭐ 마스터 가중치(BF16) 생성: 업데이트를 정밀하게 기록하는 '진짜' 가중치
    master_params = [p.detach().clone().to(torch.bfloat16).requires_grad_(True) for p in optimizer_params]
    # ✅ [최적화 1] Learning Rate를 1e-4에서 1e-5로 낮춰 안정적인 미세조정 유도
    optimizer = torch.optim.AdamW(master_params, lr=1e-5, weight_decay=1e-4)
    
    p_count = 2 if args.test_run else 64
    t_count = 2 if args.test_run else 20
    s_count = 1 if args.test_run else 4
    
    calib_prompts = prompts[:p_count]
    CALIB_SIZE = 512
    added_cond_kwargs = {
        "resolution": torch.tensor([[CALIB_SIZE, CALIB_SIZE]], device=device, dtype=target_dtype), 
        "aspect_ratio": torch.tensor([[1.0]], device=device, dtype=target_dtype)
    }

    model.train()
    for p in tqdm(calib_prompts, desc="Master-Weight Tuning Loop"):
        with torch.no_grad():
            prompt_utils = pipe.encode_prompt(p, device="cpu")
            p_emb = prompt_utils[0].to(device, dtype=target_dtype)
            p_mask = prompt_utils[1].to(device, dtype=target_dtype)

        # ✅ [최적화 2] 고정된 timestep 대신, 매 프롬프트마다 랜덤한 timestep을 샘플링하여 과적합 방지
        for _ in range(t_count * s_count): 
            t_val = torch.randint(0, 1000, (1,)).item()
            t = torch.tensor([t_val], device=device, dtype=target_dtype)
            with torch.no_grad():
                # 마스터 가중치(원본)를 잠시 모델에 넣고 정답 노이즈 예측
                for i, p_orig in enumerate(optimizer_params):
                    p_orig.data.copy_(master_params[i].data.to(target_dtype))
                
                latents = torch.randn((1, model.config.in_channels, 64, 64), device=device, dtype=target_dtype)
                target_output = model(latents, encoder_hidden_states=p_emb, encoder_attention_mask=p_mask, 
                                      timestep=t, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0].detach()

            # ✅ [해결 3] 옵티마이저뿐만 아니라 모델의 Gradient도 명시적으로 초기화
            optimizer.zero_grad()
            model.zero_grad()

            # Student (Quantized) 가중치 세팅
            with torch.no_grad():
                for i, p_orig in enumerate(optimizer_params):
                    f_type = getattr(p_orig, 'assigned_fp8_format', torch.float8_e4m3fn)
                    p_orig.data.copy_(blockwise_fake_quantize(master_params[i].data, block_size=args.block_size, fp8_type=f_type))

            # Student Forward 및 증류(Distillation) Loss 계산
            with torch.autocast("cuda", dtype=target_dtype):
                output = model(latents, encoder_hidden_states=p_emb, encoder_attention_mask=p_mask, 
                               timestep=t, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
                loss = F.mse_loss(output, target_output) # 정답 출력과 현재 출력의 차이만 최소화

            loss.backward()

            # FP8 가중치에서 계산된 grad를 마스터 가중치로 전달
            for i, p_master in enumerate(master_params):
                if optimizer_params[i].grad is not None:
                    p_master.grad = optimizer_params[i].grad.to(torch.bfloat16)

            optimizer.step()
    
    # 튜닝 완료 후 최종 결과물을 FP8 상태로 다시 동기화
    with torch.no_grad():
        for i, p_orig in enumerate(optimizer_params):
            f_type = getattr(p_orig, 'assigned_fp8_format', torch.float8_e4m3fn)
            p_orig.data.copy_(blockwise_fake_quantize(master_params[i].data, block_size=args.block_size, fp8_type=f_type))

    model.eval()
    print("✅ FP8 Master-Weight Tuning Finished!")

def print_available_quant_methods():
    print("="*80)
    print(f"{'Available Quantization Methods (_CFG)':^80}")
    print("="*80)

    # 1. mtq 모듈에서 _CFG로 끝나는 모든 속성 찾기
    all_attrs = dir(mtq)
    quant_configs = [attr for attr in all_attrs if attr.endswith("_CFG")]
    quant_configs.sort()

    for cfg_name in quant_configs:
        cfg_obj = getattr(mtq, cfg_name)
        
        # 2. 핵심 정보 추출 (데이터 타입, 알고리즘 등)
        # 보통 quant_cfg 내부의 첫 번째 레이어 설정을 샘플로 가져옵니다.
        q_cfg = cfg_obj.get("quant_cfg", {})
        algo = cfg_obj.get("algorithm", "PTQ (Static)")
        
        # 대표적인 비트수 확인
        sample_key = next(iter(q_cfg)) if q_cfg else None
        bit_info = "N/A"
        if sample_key:
            sample_val = q_cfg[sample_key]
            if isinstance(sample_val, dict):
                bit_info = f"{sample_val.get('num_bits', 'FP')}bit"

        print(f"🔹 {cfg_name:<35} | {bit_info:<8} | Algo: {str(algo)}")

    print("="*80)

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # 시드 고정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    q_method = args.quant_method.upper()
    is_quant = q_method not in ["BF16", "FP16", "FP32"]
    target_dtype = torch.float16 if q_method not in ["BF16", "FP32"] else (torch.bfloat16 if q_method == "BF16" else torch.float32)

    # 데이터셋 로드
    dataset_path, split, caption_key = ("xingjianleng/mjhq30k", "test", "text") if args.dataset_name == "MJHQ" else ("mit-han-lab/svdquant-datasets", "train", "prompt")
    ds = load_dataset(dataset_path, split=split, streaming=True).shuffle(buffer_size=1000, seed=args.seed)

    print_available_quant_methods()

    # ---------------------------------------------------------
    # 단계 1: Teacher (FP16 Baseline) & 프롬프트 수집
    # ---------------------------------------------------------
    print(f"🚀 [1/3] Generating Ground-Truth with FP16 Baseline...")
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, use_safetensors=True)
    type(pipe).device = property(lambda self: torch.device(device))
    pipe.to(device)
    pipe.text_encoder.to(device="cpu", dtype=torch.float32)
    pipe.vae.enable_tiling()

    fid = FrechetInceptionDistance(feature=2048).to("cpu")
    prompts, teacher_imgs_cpu = [], []

    for i, data in enumerate(ds):
        if i >= args.num_samples: break
        prompt = data[caption_key]
        prompts.append(prompt)
        if i < args.num_samples:
            generator = torch.Generator(device=device).manual_seed(args.seed)
            latents = torch.randn((1, pipe.transformer.config.in_channels, args.image_size // 8, args.image_size // 8), generator=generator, device=device, dtype=torch.float16)
            with torch.no_grad():
                p_utils = pipe.encode_prompt(prompt, device="cpu")
                p_emb, p_mask, n_emb, n_mask = [x.to(device) for x in p_utils]
                with torch.autocast("cuda", dtype=torch.float16):
                    img_t = pipe(prompt=None, negative_prompt=None, prompt_embeds=p_emb, prompt_attention_mask=p_mask, negative_prompt_embeds=n_emb, negative_prompt_attention_mask=n_mask, generator=generator, latents=latents, num_inference_steps=20, size=args.image_size).images[0]
                teacher_imgs_cpu.append(torch.from_numpy(np.array(img_t)).permute(2, 0, 1).float().div(255.0).cpu())
                real_img = data["image"].convert("RGB").resize((args.image_size, args.image_size))
                fid.update(torch.from_numpy(np.array(real_img)).permute(2, 0, 1).unsqueeze(0).to(torch.uint8), real=True)
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 단계 2: Student 모델 준비 및 양자화 + 튜닝
    # ---------------------------------------------------------
    print(f"🚀 [2/3] Preparing Student Model ({q_method})...")
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_id, torch_dtype=target_dtype, use_safetensors=True)
    pipe.text_encoder.to(device="cpu", dtype=torch.float32)
    pipe.transformer.to(device)
    pipe.vae.to("cpu")

    if is_quant:
        # ✅ [수정] Calibration 단계 (통계 수집용)
        def forward_loop(model):
            print("📡 [Step 2-1] Initial Calibration for SVD initialization...")
            model.eval()
            num_calib = 2 if args.test_run else 32
            target_ts = np.linspace(999, 0, 5).astype(int)
            CALIB_SIZE = 512
            added_kwargs = {"resolution": torch.tensor([[CALIB_SIZE, CALIB_SIZE]], device=device, dtype=target_dtype), "aspect_ratio": torch.tensor([[1.0]], device=device, dtype=target_dtype)}
            for p in tqdm(prompts[:num_calib], desc="Calibrating"):
                with torch.no_grad():
                    prompt_utils = pipe.encode_prompt(p, device="cpu")
                    # ✅ dtype=target_dtype 추가
                    p_emb = prompt_utils[0].to(device, dtype=target_dtype) 
                    p_mask = prompt_utils[1].to(device, dtype=target_dtype)

                    for t_val in target_ts:
                        latents = torch.randn((1, model.config.in_channels, 64, 64), device=device, dtype=target_dtype)
                        t = torch.tensor([t_val], device=device, dtype=target_dtype)
                        model(latents, encoder_hidden_states=p_emb, encoder_attention_mask=p_mask, timestep=t, added_cond_kwargs=added_kwargs, return_dict=False)

        # ✅ [해결책] Pydantic 규격에 맞는 안전한 설정 생성
        def get_safe_svd_cfg():
            # mtq는 16비트(보호 모드)로 통과시키고, 나중에 우리가 직접 8비트로 깎습니다.
            return {"num_bits": 16, "type": "static"} 

        if q_method == "TERNARY_NVFP4":
            print(f"💎 [Hybrid 1] Act: 4bit | Weight: 2bit | SVD: 8bit")
            config = {
                "quant_cfg": {
                    # num_bits만 적고 format 키는 삭제 (Schema 미지원)
                    "*transformer_blocks*.weight_quantizer": {"num_bits": 2, "type": "static"},
                    "*transformer_blocks*.input_quantizer": {"num_bits": 4, "type": "dynamic"},
                    "*high_rank*": get_safe_svd_cfg(),
                    "*low_rank*": get_safe_svd_cfg()
                },
                "algorithm": {"method": "svdquant", "lowrank": args.lowrank}
            }

        elif q_method == "NVFP4_ALL":
            print(f"🚀 [Hybrid 2] Act: 4bit | Weight: 4bit | SVD: 8bit")
            # NVFP4_ALL을 수동 정의할 때도 format 키를 빼야 합니다.
            config = {
                "quant_cfg": {
                    "*transformer_blocks*.weight_quantizer": {"num_bits": 4, "type": "static"},
                    "*transformer_blocks*.input_quantizer": {"num_bits": 4, "type": "dynamic"},
                    "*high_rank*": get_safe_svd_cfg(),
                    "*low_rank*": get_safe_svd_cfg()
                },
                "algorithm": {"method": "svdquant", "lowrank": args.lowrank}
            }
            
        elif q_method == "NVFP4_SVDQUANT_DEFAULT_CFG":
            print(f"📦 [Mode] Library Default Config (Overriding SVD to 16bit)")
            config = copy.deepcopy(mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
            config["algorithm"]["lowrank"] = args.lowrank
            config["quant_cfg"].update({
                "*high_rank*": get_safe_svd_cfg(),
                "*low_rank*": get_safe_svd_cfg()
            })
        else:
            # 기타 설정 (TERNARY 등 기존 코드 대응)
            config = getattr(mtq, args.quant_method)


        # ✅ 2-1. 양자화 수행 (표준 8비트 설정으로 통과)
        pipe.transformer = mtq.quantize(pipe.transformer, config, forward_loop=forward_loop)

        # 2. 🎯 [여기가 적용 위치!] 우리가 직접 Hybrid Block-wise 적용
        apply_hybrid_blockwise_quant(pipe.transformer, args, device)

        # ✅ 2-3. (선택) 만약 튜닝을 켠다면 기존 함수 호출
        if args.do_diff_tuning:
            run_diff_tuning(pipe.transformer, pipe, prompts, args, device, target_dtype)


    # ---------------------------------------------------------
    # 단계 3: 지표 측정
    # ---------------------------------------------------------
    print(f"🎨 [3/3] Computing Metrics for {q_method} (Rank {args.lowrank})...")
    type(pipe).device = property(lambda self: torch.device(device))
    pipe.to(device)
    pipe.text_encoder.to(device="cpu", dtype=torch.float32)
    pipe.vae.enable_tiling()
    
    ssim_metric, psnr_metric, lpips_metric = SSIM(data_range=1.0).to(device), PSNR(data_range=1.0).to(device), LPIPS(net_type='vgg').to(device)
    clip_model, clip_processor = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cpu(), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    total_ir, total_clip_scr, total_clip_iqa = 0, 0, 0
    ir_scorer = None
    if HAS_IR:
        print("📥 Loading ImageReward model...")
        ir_scorer = RM.load("ImageReward-v1.0").to(device)

    for i in tqdm(range(args.num_samples)):
        prompt = prompts[i]
        generator = torch.Generator(device=device).manual_seed(args.seed)
        latents = torch.randn((1, pipe.transformer.config.in_channels, args.image_size // 8, args.image_size // 8), generator=generator, device=device, dtype=target_dtype)
        with torch.no_grad():
            p_utils = pipe.encode_prompt(prompt, device="cpu")
            p_emb, p_mask, n_emb, n_mask = [x.to(device) for x in p_utils]
            pipe.transformer.to(device)
            with torch.autocast("cuda", dtype=target_dtype):
                pred_latents = pipe(prompt=None, negative_prompt=None, prompt_embeds=p_emb, prompt_attention_mask=p_mask, negative_prompt_embeds=n_emb, negative_prompt_attention_mask=n_mask, generator=generator, latents=latents, num_inference_steps=20, size=args.image_size, output_type="latent").images
            pipe.transformer.to("cpu")
            pipe.vae.to(device)
            img_tensor = pipe.vae.decode(pred_latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            img_s = pipe.image_processor.postprocess(img_tensor, output_type="pil")[0]
            pipe.vae.to("cpu")
            
            s_tensor, t_tensor = torch.from_numpy(np.array(img_s)).permute(2, 0, 1).float().div(255.0).to(device), teacher_imgs_cpu[i].to(device)
            ssim_metric.update(s_tensor.unsqueeze(0), t_tensor.unsqueeze(0)), psnr_metric.update(s_tensor.unsqueeze(0), t_tensor.unsqueeze(0)), lpips_metric.update(s_tensor.unsqueeze(0)*2.0-1.0, t_tensor.unsqueeze(0)*2.0-1.0)
            fid.update((s_tensor.cpu() * 255).unsqueeze(0).to(torch.uint8), real=False)
            
            c_scr, c_iqa = calculate_clip_metrics(clip_model, clip_processor, img_s, prompt, device)
            total_clip_scr += c_scr; total_clip_iqa += c_iqa
            if ir_scorer:
                total_ir += ir_scorer.score(prompt, img_s)
            torch.cuda.empty_c/ache(); gc.collect()

    # 결과 저장
    res_fid = float(fid.compute().item())
    final_results = {"config": vars(args), "results": {"FID": res_fid, "IR": float(total_ir/args.num_samples), "LPIPS": float(lpips_metric.compute()), "PSNR": float(psnr_metric.compute()), "SSIM": float(ssim_metric.compute()), "CLIP_Score": float(total_clip_scr/args.num_samples)}}
    suffix = "diff_tuned" if args.do_diff_tuning else "static"
    with open(os.path.join(args.save_dir, f"results_rank_{args.lowrank}_{q_method}_{args.svd_dtype}_{args.fp8_format}_{args.block_size}_{suffix}.json"), "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"\n✅ All finished. Mode: {suffix} | FID: {res_fid:.4f}")

if __name__ == "__main__":
    main()
