import os
import torch
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 평가용 라이브러리
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

from models.dit import DiT_models
from diffusion.ddpm import DDPMScheduler
import json # 파일 맨 위 import 부분에 추가

try:
    import modelopt.torch.quantization as mtq
except ImportError:
    print("Warning: nvidia-modelopt가 설치되지 않았습니다. Fake Quantization이 생략될 수 있습니다.")

def parse_args():
    parser = argparse.ArgumentParser(description="Sample and Evaluate DiT model (FID, IS)")
    
    parser.add_argument("--model_path", type=str, required=True, help="학습된 모델 가중치 경로")
    parser.add_argument("--model_size", type=str, default="DiT-B/8", help="DiT 모델 크기")
    parser.add_argument("--dataset_name", type=str, default="CIFAR-10", help="데이터셋 이름")
    parser.add_argument("--dataset_path", type=str, default="./data", help="데이터셋 경로")
    parser.add_argument("--image_size", type=int, default=32, help="이미지 크기")
    parser.add_argument("--batch_size", type=int, default=128, help="생성 및 평가 배치 사이즈")
    parser.add_argument("--num_samples", type=int, default=1000, help="평가에 사용할 총 이미지 수 (논문 기준 보통 50000장 사용)")
    parser.add_argument("--save_dir", type=str, default="generated_samples", help="이미지 저장 폴더")
    parser.add_argument("--device", type=str, default="cuda:0")
    # 💡 NVFP4를 포함해 도커에 있는 양자화 옵션 이름 그대로 입력받도록 설정
    parser.add_argument("--quant_method", type=str, default="none", 
                        help="적용할 양자화 설정 이름 (예: NVFP4_DEFAULT_CFG, NVFP4_AWQ_FULL_CFG 등)")    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    in_channels = 3 if args.dataset_name == "CIFAR-10" else 1

    # 1. 모델 로드
    model = DiT_models[args.model_size](
        image_size=(args.image_size, args.image_size), 
        input_channel=in_channels, 
        num_labels=10
    ).to(args.device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # =================================================================
    # 2. 양자화 및 정밀도 변환 적용
    # =================================================================
    quant_method_lower = args.quant_method.lower()

    if quant_method_lower == "bf16":
        # print("✅ BF16 (Bfloat16) 정밀도로 모델 가중치를 변환합니다.")
        # model = model.bfloat16()
        print("✅ BF16 (Bfloat16) Autocast 모드로 모델 평가를 진행합니다.")

    elif quant_method_lower not in ["none", "fp32"]:
        print(f"Applying Fake Quantization method: {args.quant_method}...")

        # 🚀 현재 modelopt 환경에서 지원하는 모든 양자화 CFG 리스트 출력
        available_configs = [k for k in dir(mtq) if "CFG" in k]
        print(f"👉 현재 도커 환경에서 지원하는 양자화 옵션들: {available_configs}") 

        if hasattr(mtq, args.quant_method):
            quant_config = getattr(mtq, args.quant_method)
            print(f"✅ {args.quant_method} 모의 양자화를 적용합니다.")
            mtq.quantize(model, quant_config)
            print("Quantization applied successfully!")

            # -------------------------------------------------------------
            # 양자화된 레이어와 안 된 레이어 추출 및 출력
            # -------------------------------------------------------------
            quantized_layers = []
            unquantized_layers = []

            for name, module in model.named_modules():
                if len(list(module.children())) == 0:
                    module_type = type(module).__name__
                    if 'Quant' in module_type or hasattr(module, 'weight_quantizer'):
                        quantized_layers.append((name, module_type))
                    else:
                        unquantized_layers.append((name, module_type))

            print("\n" + "="*50)
            print("✅ 양자화가 적용된 레이어 (Quantized Layers):")
            for name, m_type in quantized_layers:
                print(f"  - {name} ({m_type})")

            print("\n" + "="*50)
            print("❌ 양자화에서 제외된 레이어 (Unquantized Layers):")
            for name, m_type in unquantized_layers:
                print(f"  - {name} ({m_type})")
            print("="*50 + "\n")

        else:
            raise ValueError(f"🚨 현재 modelopt 환경에서 '{args.quant_method}' 설정을 찾을 수 없습니다.")
    else:
        print("ℹ️ 양자화를 적용하지 않고 원본(FP32) 모델로 평가를 진행합니다.")


    # 3. 평가 메트릭 초기화 (torchmetrics)
    # FID는 2048 차원의 Inception-v3 특징을 사용합니다.
    fid = FrechetInceptionDistance(feature=2048).to(args.device)
    inception = InceptionScore().to(args.device)
    # KID 추가 (subset_size는 num_samples보다 작아야 함, 보통 50이나 100 사용)
    kid = KernelInceptionDistance(subset_size=50).to(args.device)

    # 4. 실제 이미지(Real Images)를 FID 메트릭에 업데이트
    print("Loading real images for FID computation...")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor() # 0 ~ 1 사이의 값으로 변환
    ])
    dataset = datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    real_count = 0
    for batch_x, _ in tqdm(dataloader, desc="Processing Real Images"):
        if real_count >= args.num_samples:
            break
        # torchmetrics FID는 0~255 사이의 uint8 텐서를 권장합니다.
        real_images_uint8 = (batch_x * 255).to(torch.uint8).to(args.device)
        fid.update(real_images_uint8, real=True)
        kid.update(real_images_uint8, real=True)
        real_count += batch_x.size(0)

    # 5. 디퓨전 스케줄러 (샘플링 파라미터)
    diffusion_scheduler = DDPMScheduler(
        beta_schedule="linear", noise_steps=1000, beta_start=1e-4, beta_end=2e-2
    )

    # 6. 이미지 생성 (Fake Images) 및 메트릭 업데이트
    print(f"Generating {args.num_samples} fake images for evaluation...")
    generated_count = 0
    save_flag = True # 이미지는 첫 번째 배치만 저장하기 위한 플래그

    with torch.no_grad():
        while generated_count < args.num_samples:
            current_batch_size = min(args.batch_size, args.num_samples - generated_count)
            
            x = torch.randn(current_batch_size, in_channels, args.image_size, args.image_size).to(args.device)
            y = torch.randint(0, 10, (current_batch_size,)).to(args.device)

            # Reverse Process (샘플링)
            for i in tqdm(reversed(range(1000)), total=1000, desc=f"Sampling Batch {generated_count//args.batch_size + 1}", leave=False):
                t = torch.tensor([i] * current_batch_size).to(args.device)
                
                # ---------------------------------------------------------
                # 노이즈 예측 (torch.autocast를 활용한 스마트 BF16 적용)
                # ---------------------------------------------------------
                if args.quant_method.lower() == "bf16":
                    # Autocast가 내부적으로 발생하는 Float/BFloat16 충돌을 알아서 조율해줍니다.
                    with torch.autocast(device_type=args.device.split(':')[0], dtype=torch.bfloat16):
                        predicted_noise = model(x, t, y)
                else:
                    predicted_noise = model(x, t, y)
                # ---------------------------------------------------------
                
                # 🚀 제공해주신 ddpm.py의 denoise_sample 수식과 변수명에 완벽 동기화
                shape = (current_batch_size, 1, 1, 1)
                
                # 스케줄러에서 현재 타임스텝의 파라미터 가져오기
                alpha_t = diffusion_scheduler.alpha.to(args.device)[t].view(*shape)
                alpha_cum_t = diffusion_scheduler.alpha_cum_product.to(args.device)[t].view(*shape)
                variance_t = diffusion_scheduler.variance.to(args.device)[t].view(*shape)
                
                # x_t-1 의 평균(mean) 계산
                batch_mean_t = (1 / torch.sqrt(alpha_t)) * (
                    x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cum_t)) * predicted_noise
                )
                
                # 분산(variance)을 더해 최종 x_t-1 계산 (마지막 스텝 i=0 에서는 노이즈 제외)
                if i > 0:
                    random_noise = torch.randn_like(x)
                    x = batch_mean_t + random_noise * torch.sqrt(variance_t)
                else:
                    x = batch_mean_t

            # 생성된 이미지 정규화 해제: [-1, 1] -> [0, 1]
            x_norm = (x.clamp(-1, 1) + 1) / 2.0
            x_uint8 = (x_norm * 255).to(torch.uint8)

            # 눈으로 확인하기 위해 첫 배치 저장
            if save_flag:
                save_image(x_norm, os.path.join(args.save_dir, "sample_grid.png"), nrow=8)
                save_flag = False

            # 생성된 가짜 이미지를 FID와 IS에 업데이트
            fid.update(x_uint8, real=False)
            inception.update(x_uint8)
            kid.update(x_uint8, real=False)
            
            generated_count += current_batch_size

    # 7. 최종 점수 계산
    print("\n" + "="*40)
    print("Computing metrics... (This may take a moment)")
    
    fid_score = fid.compute()
    is_mean, is_std = inception.compute()
    kid_mean, kid_std = kid.compute()

    print(f"✅ Evaluation Complete for {args.num_samples} images!")
    print(f"👉 FID Score (낮을수록 좋음): {fid_score.item():.4f}")
    print(f"👉 Inception Score (높을수록 좋음): {is_mean.item():.4f} ± {is_std.item():.4f}")
    print(f"👉 KID Score (낮을수록 좋음): {kid_mean.item():.4f} ± {kid_std.item():.4f}")
    print("="*40)

    # ==========================================
    # 8. 점수표 파일로 저장 (JSON 포맷)
    # ==========================================
    results = {
        "model_path": args.model_path,
        "model_size": args.model_size,
        "num_samples": args.num_samples,
        "quant_method": args.quant_method,
        "metrics": {
            "FID": round(fid_score.item(), 4),
            "IS_mean": round(is_mean.item(), 4),
            "IS_std": round(is_std.item(), 4),
            "KID_mean": round(kid_mean.item(), 4),
            "KID_std": round(kid_std.item(), 4)
        }
    }

    # 저장할 파일명 설정 (예: metrics.json)
    result_file_path = os.path.join(args.save_dir, "metrics.json")
    
    with open(result_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print(f"💾 평가 결과가 성공적으로 저장되었습니다: {result_file_path}")

if __name__ == "__main__":
    main()