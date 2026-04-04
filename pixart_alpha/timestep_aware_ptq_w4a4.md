# Timestep-Aware Rotation/Decomposition for PTQ W4A4 on PixArt-XL-2

## 현재 세팅 요약

| 항목 | 현재 상태 |
|---|---|
| 모델 | PixArt-XL-2-1024-MS (DiT 기반) |
| 현재 방식 | SmoothQuant + SVDQuant @ NVFP4 |
| Activation 방식 | Per-group dynamic |
| Rotation | SmoothQuant scale만 적용 (Hadamard 없음) |
| Calibration | MJHQ, 64 samples / 5000장 |
| 목표 | PTQ로 W4A4 달성 (QAT 없이) |

---

## 핵심 문제 진단

SmoothQuant의 scale은 **timestep에 무관하게 고정**된다.  
DiT의 activation 분포는 timestep마다 완전히 다른데,  
하나의 scale로 모든 timestep을 커버하려니 한계가 발생.

---

## 방향 1. Timestep-Grouped Rotation (가장 현실적)

**아이디어**: Timestep을 몇 개 그룹으로 나누고, 그룹별로 다른 Hadamard Rotation을 적용

```
T = [0, 1000] → 3~5개 그룹으로 분할
  Group 1: t ∈ [0, 200]    → R₁ (late timestep에 맞춘 rotation)
  Group 2: t ∈ [200, 600]  → R₂
  Group 3: t ∈ [600, 1000] → R₃ (early timestep에 맞춘 rotation)
```

**수학적 formulation**:

각 그룹 g에서 calibration activation $X_g$를 모아서:

$$R_g = \arg\min_R \mu(R \cdot X_g), \quad \text{s.t. } R^\top R = I$$

여기서 $\mu$는 incoherence measure (max singular value / Frobenius norm).

실제로는 Hadamard는 고정이고, **그룹별 per-channel scale $s_g$** 를 학습:

$$\hat{X}_g = \text{Hadamard}(X_g \cdot \text{diag}(s_g)^{-1})$$

**장점**:
- Weight는 offline으로 한 번만 rotate
- Activation은 runtime에 timestep group에 따라 다른 scale 적용
- 오버헤드 최소화

---

## 방향 2. Timestep-Aware SVD Decomposition (SVDQuant 확장)

**아이디어**: SVDQuant의 low-rank branch를 timestep별로 다르게 적용

현재 SVDQuant 구조:

$$Y = W X = \underbrace{L_1 L_2}_{\text{FP16 low-rank}} X + \underbrace{Q(W - L_1 L_2)}_{\text{INT4}} X$$

여기서 low-rank branch $L_1 L_2$는 activation outlier를 흡수하는 역할인데,  
outlier의 크기와 방향이 **timestep마다 달라지는 문제**가 있음.

**제안**: Timestep-conditioned residual을 low-rank branch에 추가:

$$Y_t = (L_1 L_2 + \Delta_t) X + Q(R) X$$

여기서 $\Delta_t$는 timestep group $g(t)$에 따라 선택되는 작은 rank-4 correction matrix.  
Calibration 단계에서:

$$\Delta_g = \arg\min_{\Delta, \text{rank}=4} \sum_{t \in g} \|W X_t - (L_1 L_2 + \Delta) X_t - Q(R) X_t\|^2_F$$

**장점**:
- 기존 SVDQuant 구조를 유지하면서 timestep별 보정만 추가
- rank-4면 파라미터 오버헤드 극히 작음

---

## 방향 3. Timestep-Aware Smooth + Hadamard 결합 (가장 새로운 시도)

**아이디어**: SmoothQuant의 channel-wise scale을 Hadamard rotation 이후에 timestep-adaptive하게 적용

**현재 파이프라인**:
```
X → SmoothQuant(scale s) → Quantize
```

**제안 파이프라인**:
```
X → Hadamard(H) → Timestep-adaptive scale(s_t) → Quantize
         ↑                      ↑
    outlier 분산시킴       timestep별 분포 보정
```

수학적으로:

$$\hat{X}_t = \text{diag}(s_{g(t)})^{-1} \cdot H \cdot X_t$$

$$s_{g(t),j} = \frac{\max_j(\|H X_t\|_\infty)}{\text{target\_max}}$$

**핵심**:  
Hadamard가 outlier를 먼저 분산시킨 후,  
timestep-group별 scale이 남은 분포 편차를 보정.  
현재 SmoothQuant만 쓰는 것보다 훨씬 quantization-friendly한 분포가 됨.

---

## 실험 순서 추천

### Step 1 — Timestep별 Activation 분포 분석 (선행 필수)
```python
# 각 timestep에서 activation의 kurtosis, max/std 측정
# → 어느 timestep에서 outlier가 폭발하는지 확인
for t in timesteps:
    act = get_activation(model, x_t, t)
    kurtosis[t] = scipy.stats.kurtosis(act.flatten())
    max_val[t] = act.abs().max()
    std_val[t] = act.std()
```

### Step 2 — Hadamard Rotation 추가
SmoothQuant scale 제거 후 Hadamard만 적용 → FID 변화 확인

### Step 3 — Timestep Grouping + Per-group Scale 추가 (방향 1)
Group 수를 3 → 5 → 10으로 늘려가며 FID vs 오버헤드 tradeoff 확인

### Step 4 — SVD Low-rank Correction 추가 (방향 2)
rank 4 → 8 → 16으로 늘려가며 FID recovery 측정

---

## 예상 효과

| 추가 방법 | 예상 FID 개선 | 구현 난이도 |
|---|---|---|
| Hadamard rotation만 | ★★★ | 낮음 |
| + Timestep-grouped scale | ★★★★ | 중간 |
| + SVD timestep correction | ★★★★★ | 높음 |

---

## 참고 논문

| 논문 | 핵심 기여 | 관련성 |
|---|---|---|
| SVDQuant (ICLR 2025) | Low-rank + INT4 for Diffusion | 베이스라인 |
| QuaRot (2024) | Hadamard rotation for W4A4 | Rotation 참고 |
| ConvRot (2025) | DiT-specific group-wise Hadamard | DiT rotation |
| PTQ4DiT (NeurIPS 2024) | Timestep-aware calibration for DiT | Timestep 분석 |
| TCAQ-DM (AAAI 2025) | Timestep-channel adaptive quantization | Timestep+channel |
| ViDiT-Q (ICLR 2025) | Token/condition/timestep variance 분석 | 분포 분석 참고 |
| TFMQ-DM (CVPR 2024) | Temporal feature disturbance 분석 | Timestep 중요성 |
