# Sub-4-bit quantization for Diffusion Transformers: a mathematical roadmap

**Pushing PixArt-XL-2 below NVFP4 to W2/W3 precision is feasible for weights but remains an open frontier for activations.** The most promising path combines Hadamard rotation (QuaRot/QuIP#) to eliminate outliers, SVD low-rank decomposition to absorb residual sensitivity, and GPTQ/E8-lattice vector quantization to compress the residual to INT2–INT3. For activations, timestep-aware dynamic quantization at INT4 with per-group granularity represents the practical floor; INT3 activations require aggressive mixed-precision strategies that protect attention logits, timestep embeddings, and cross-attention layers. Published results confirm W4A4 on PixArt-Σ via SVDQuant+NVFP4 matches FP16 quality, while TerDiT demonstrates ternary (1.58-bit) weights are viable through QAT—but no PTQ method has yet validated sub-4-bit weights on text-to-image DiTs. The gap from W4A4 to W3A3 is roughly an order of magnitude larger than the gap from FP16 to W4A4.

---

## 1. Hessian-based weight rounding extends naturally to INT2/INT3

GPTQ solves the layer-wise reconstruction objective **min_Q̂ ‖WX − Q̂X‖²_F** by quantizing weights column-by-column using the inverse Hessian **H = 2XX⊤**. For column q, the update rule is:

$$\hat{q}_q = \text{quant}(w_q), \quad \delta_q = \frac{w_q - \hat{q}_q}{[H^{-1}]_{qq}}, \quad w_{j>q} \leftarrow w_{j>q} - \delta_q \cdot \frac{[H^{-1}]_{q,j}}{[H^{-1}]_{qq}}$$

This second-order compensation propagates each column's rounding error optimally through remaining columns. At **3-bit with group-size 1024**, GPTQ achieves **8.45 perplexity on OPT-175B** (only +0.11 over FP16's 8.34). At **ternary with group-size 8**, it reaches 9.20 perplexity—less than 1 point above FP16. The act-order heuristic (quantizing columns in decreasing activation magnitude order) is critical for outlier-heavy models, reducing OPT-66B's 3-bit perplexity from 14.16 to 9.95.

The key limitation at 2-bit is that rounding errors become high-rank: unlike 3–4 bit where quantization errors exhibit strong low-rank structure amenable to SVD compensation, **2-bit errors require substantially higher rank for residual decomposition** (RILQ, 2024). This motivates combining GPTQ with incoherence processing.

**SpQR** extends this framework by identifying outlier weights whose quantization causes disproportionate error. The sensitivity metric **s_{ij} = (w_{ij} − quant(w_{ij}))² / (2·[H⁻¹]_{jj})** captures both rounding magnitude and the weight's second-order importance. SpQR stores the top ~0.5–1% most sensitive weights in FP16 and quantizes the rest to 3-bit with group-size 16, achieving **<1% relative perplexity loss** on LLaMA models. The bilevel quantization of group statistics (scales themselves compressed to 3-bit) is essential for keeping overhead low at aggressive bit widths.

---

## 2. Rotation and incoherence processing unlock viable 2-bit PTQ

QuIP# and QuaRot represent the two dominant rotation paradigms. Both exploit the fact that quantization error depends on the **incoherence** μ of weight matrices—matrices with uniformly distributed entries quantize far better than those with outlier concentrations.

**QuIP#** applies randomized Hadamard transforms **W̃ = U·S_U·W·S_V·V⊤** where U, V are Walsh-Hadamard matrices and S_U, S_V are random sign diagonals. The incoherence bound is **μ ∝ √(log(n/δ)/n)**, ensuring outlier energy is spread uniformly across dimensions. After rotation, weights become approximately i.i.d. sub-Gaussian with ball-shaped distribution, enabling **E8 lattice vector quantization**—an 8-dimensional codebook achieving optimal unit-ball packing in just **1 KiB** (fits in GPU L1 cache). Combined with BlockLDLQ rounding (mathematically equivalent to GPTQ), QuIP# was the **first PTQ method where 3-bit models scale better than 4-bit models** and achieves viable 2-bit quantization across LLaMA-2 scales. QTIP extends this further with trellis coded quantization for effective 256-dimensional VQ.

**QuaRot** takes a complementary approach: rather than rotating for quantization-friendly weight distributions, it uses the **computational invariance** of the transformer hidden state. The rotation y = W(Q⊤Q)x = (WQ⊤)(Qx) = W'x' is fused offline into adjacent weight matrices, producing an exactly equivalent model with outlier-free activations and weights. This enables uniform W4A4 quantization without mixed-precision. On LLaMA-2-70B at W4A4KV4, QuaRot shows **<0.5 perplexity increase**. For sub-4-bit, QuaRot serves as the pre-processing step before GPTQ—validated by the LLMC toolkit (EMNLP 2024), AMD Quark, and GPTAQ (ICML 2025), which shows QuaRot+GPTAQ reduces 2-bit perplexity by ~50% over QuaRot+GPTQ alone.

**For DiTs specifically**, ConvRot (2025) identifies that standard Hadamard transforms can amplify row-wise outliers unique to diffusion transformer architectures. It proposes **group-wise regular Hadamard transforms** that suppress both row-wise and column-wise outliers, achieving plug-and-play W4A4 for large-scale DiT models with ~20% of layers kept at W8A8 based on sensitivity.

---

## 3. Low-rank decomposition compensates quantization residuals at the cost of rank

The LoftQ framework decomposes **W ≈ Q + AB** where Q is quantized and AB is a low-rank residual in FP16. The alternating optimization iterates:

1. **Quantize**: Q_t = quant(W − A_{t-1}B_{t-1})
2. **SVD**: U, Σ, V⊤ = SVD(W − Q̃_t)  
3. **Update**: A_t = U_{:,:r}·Σ_{:r}^{1/2}, B_t = Σ_{:r}^{1/2}·V⊤_{:r,:}

At **2-bit NF2 on DeBERTaV3**, LoftQ achieves 88.0% GLUE accuracy versus QLoRA's 76.5% (+11.5%). The key insight is that the low-rank component captures the most important directions of quantization error. However, LoftQ requires subsequent fine-tuning to fully exploit the initialization.

**SVDQuant** (ICLR 2025 Spotlight) adapts this for diffusion models with a three-stage pipeline that is directly relevant to your baseline:

1. Migrate activation outliers to weights via SmoothQuant-style scaling
2. Decompose: **Ŵ = L₁L₂ + R**, where L₁L₂ is the FP16 low-rank branch  
3. Quantize residual: **Q(R)** at 4-bit with GPTQ-style optimization

The Nunchaku inference engine fuses the low-rank branch into the quantized kernel, limiting overhead to 5–10% (versus 57% without fusion). On **PixArt-Σ at W4A4 with NVFP4**, SVDQuant achieves quality **comparable to FP16** and significantly outperforms ViDiT-Q's W4A8 across FID, ImageReward, LPIPS, and PSNR.

**SVD-LLM** (ICLR 2025) demonstrates the critical integration point: combining SVD-LLM (20% compression) with GPTQ-4bit achieves **18% lower perplexity than GPTQ-3bit alone** with smaller memory (2.1 GB vs 2.8 GB). It also combines with QuIP# 2-bit under 40% SVD compression, outperforming the training-required OneBit method. The Hessian for residual quantization after SVD is H_R = E[x_rotated · x_rotated⊤], and because SVD removes the dominant spectral components, the residual has **reduced spectral norm and more uniform singular values**, making it more amenable to aggressive quantization.

**CALDERA** (2024) pushes this furthest: W ≈ Q + LR where Q is 2-bit QuIP#-quantized and LR is a (possibly low-precision) low-rank correction. It formally establishes theoretical bounds and outperforms all existing PTQ methods in the **<3 bits regime** on LLaMA-2 and LLaMA-3.

---

## 4. Ternary quantization requires QAT for diffusion transformers but GPTQ-ternary shows surprising viability

Ternary methods restrict weights to {−α, 0, +α}. TWN solves **min_{α,W^t} ‖W − αW^t‖²** with closed-form solutions: the threshold **Δ* ≈ 0.7·E(|W|)** and scaling factor **α* = mean(|W_i| : |W_i| > Δ)**. TTQ extends this with asymmetric learnable scales W^p, W^n, actually **surpassing full-precision accuracy** on AlexNet (42.5% vs 44.1% top-1 error).

**BitNet b1.58** constrains weights to {−1, 0, +1} during training (1.58 bits) with straight-through estimation. At 2B+ parameters trained on 4T tokens, it **matches FP16 LLMs** on standard benchmarks while achieving **38.8× energy reduction** at 30B scale. The scaling law indicates ternary models need approximately **double the hidden dimension** to match 16-bit perplexity.

**TerDiT** (2024) is the first ternary DiT, directly relevant to your PixArt target. Key findings:

- Added **RMS Norm after the MLP in adaLN**—critical for stable ternary training (without it, training diverges)
- TerDiT-4.2B outperforms the original full-precision DiT-XL on ImageNet 512×512 (FID 4.34 vs BitNet's 6.60)
- Demonstrates that the ternary scaling law holds for DiTs: larger models compensate for precision loss
- **EfficientDM at 2-bit weight-only PTQ fails to generate normal images**—confirming that sub-4-bit PTQ for diffusion models remains unsolved

GPTQ with group-size 8 achieves ternary quantization on OPT-175B at **9.20 perplexity** (<1 point above FP16), showing that second-order methods can handle ternary post-training on sufficiently large models. Whether this transfers to the 0.6B-parameter PixArt-XL-2 is an open question—model scale is critical for ternary robustness.

---

## 5. Activation quantization below 4-bit demands timestep-aware dynamic strategies

The activation outlier problem is catastrophic at sub-4-bit: outlier channels exhibit magnitudes **~100× larger** than typical values, and with only 4–8 quantization levels available at INT2/INT3, effective resolution for non-outlier channels collapses to 1–2 levels under per-tensor quantization.

Three paradigms address this, in increasing effectiveness:

**Smooth/migrate outliers** (SmoothQuant, OS+): The transformation Y = (X·diag(s)⁻¹)·(diag(s)·W) with **s_j = max(|X_j|)^α / max(|W_j|)^{1−α}** redistributes difficulty from activations to weights. However, SmoothQuant **fails catastrophically at W4A4** (LLaMA-7B perplexity jumps from 5.68 to 77.65). OS+ improves with channel-wise shifting for asymmetric outliers: X̂ = (X − diag(z))·diag(s)⁻¹, achieving 15.5% improvement for 4-bit BERT, but still insufficient for sub-4-bit.

**Remove outliers via rotation** (QuaRot): Hadamard rotation disperses outliers across all channels, reducing kurtosis to ~3 (near-Gaussian). This enables **uniform INT4 activations without mixed-precision**. The limitation at ≤3 bits is that fixed rotations don't adapt to layer-wise differences.

**Mixed-precision retention** (Atom, COMET): Atom keeps 128 salient outlier channels at INT8 while quantizing the rest to INT4, with per-group (g=128) dynamic scaling. On **LLaMA-65B at W4A4**, Atom shows <0.4 perplexity increase. At **W3A3, the degradation jumps to ~2.3 perplexity increase**—an order of magnitude worse—and existing methods (OmniQuant, SmoothQuant, QLLM) completely fail at W3A3.

**For diffusion transformers specifically**, the activation challenge is compounded by temporal variance. Key methods:

- **Q-Diffusion**: Calibration data must be sampled uniformly across timesteps; split quantization for bimodal shortcut activations
- **PTQ4DiT**: Spearman's ρ-guided Salience Calibration dynamically adjusts balanced salience per timestep
- **TFMQ-DM** (CVPR 2024): Identifies temporal feature disturbance—quantization errors cause timestep t's features to align with timestep t+δt, causing trajectory deviation
- **TCAQ-DM** (AAAI 2025): First competitive W4A4 on diffusion models, using timestep-channel joint reparameterization and dynamically adaptive quantizers (log₂ for power-law distributions at late timesteps, uniform for early timesteps)
- **ViDiT-Q** (ICLR 2025): Identifies four levels of variance in DiT activations—token-wise, condition-wise (CFG), timestep-wise, and time-varying channel-wise—and applies static-dynamic channel balancing with metric-decoupled mixed precision

---

## 6. DiT layers exhibit sharply different quantization sensitivity

Published DiT quantization papers converge on a clear sensitivity hierarchy for PixArt-class models:

**Most sensitive (keep at higher precision)**:
- Timestep embedding MLPs and adaLN modulation parameters (~27% of DiT parameters). Quantizing time embeddings alone increases FID by 0.81–1.04 (QuEST). TerDiT found direct ternarization of adaLN causes training collapse without added RMSNorm
- Cross-attention layers with T5 text conditioning. ViDiT-Q identifies significant condition-wise variance from classifier-free guidance's dual forward passes
- Post-softmax attention values exhibiting extreme non-uniform distributions (near power-law at late timesteps)

**Moderately sensitive**:
- Attention QKV and output projections with salient channel outliers. PTQ4DiT exploits the complementarity property: extreme values in activations and weights rarely coincide in the same channel
- First and last transformer blocks (standard practice to keep at INT8)

**Least sensitive (most compressible)**:
- FFN up/down projections in the middle blocks. However, ViDiT-Q's counterintuitive finding shows **FFN layers are MORE sensitive than attention layers** at W4A8—recommending all FFN layers (~15% of layers) stay at W8A8 while attention goes to W4A8. Blocks 6 and 26 specifically are outliers in sensitivity

This hierarchy directly informs mixed-precision bit allocation: a W2A4 pipeline should protect timestep/cross-attention/adaLN at INT8+ while aggressively quantizing mid-block attention projections.

---

## 7. A concrete W3A4 and W2A4 pipeline for PixArt-XL-2

Based on the synthesis of all methods, the most promising pipeline combines five stages:

**Stage 1 — Rotation pre-processing (offline)**. Apply ConvRot-style group-wise regular Hadamard transforms to all linear layers. This eliminates both row-wise and column-wise outliers unique to DiTs. Fuse rotations R₁, R₂ into adjacent weight matrices (zero inference cost). Apply online rotations R₃, R₄ via fast Hadamard transform for activation/KV quantization.

**Stage 2 — SVD decomposition with activation awareness**. For each linear layer, compute activation-aware SVD (ASVD-style): scale W' = W·diag(s)⁻¹ using calibration activation statistics, then decompose W' = UΣV⊤. Extract rank-r component (r ∈ {16, 32, 64}) as the FP16 low-rank branch L₁L₂. The residual R = W' − L₁L₂ has reduced spectral norm and more uniform singular values.

**Stage 3 — Aggressive weight quantization of residual**. Apply GPTQ to R at INT3 (W3A4 target) or QuIP# with E8 lattice codebook at INT2 (W2A4 target). The GPTQ objective for the residual is **min_Q tr((R−Q)·H_R·(R−Q)⊤)** where H_R = E[x_rotated·x_rotated⊤]. Because rotation has made the residual's Hessian more uniform and SVD has removed dominant spectral components, aggressive quantization is more feasible.

**Stage 4 — Timestep-aware dynamic activation quantization**. Quantize activations to INT4 using per-group (g=128) dynamic scaling with MSE-optimal clipping. Apply PTQ4DiT's Spearman's ρ-guided calibration to adjust quantization parameters per timestep. For attention softmax outputs, use TCAQ-DM's adaptive quantizer (log₂ for power-law timesteps, uniform otherwise).

**Stage 5 — Mixed-precision protection**. Keep at FP16/INT8:
- Timestep embedding MLP and adaLN modulation parameters
- Cross-attention KV projections (T5 text conditioning)  
- Post-softmax attention matrices
- FFN layers in blocks 6 and 26 (per ViDiT-Q sensitivity analysis)
- First and last transformer blocks

The full objective is:

$$\min_{Q_W, L_1, L_2} \sum_{t} \sum_{l} \left\| f_l^{\text{FP}}(X_t) - Q_X(X_t) \cdot (L_1 L_2 + Q_W(R_l))^\top \right\|^2$$

where the sum over t spans calibration timesteps with ViDiT-Q's temporal weighting.

---

## 8. Experimental recommendations for MJHQ calibration on PixArt-XL-2-1024-MS

**Calibration protocol**. Use 64 samples from MJHQ-30K stratified across the 10 categories (≈6 per category), consistent with PTQ4DiT's calibration scale. Sample calibration data across **all denoising timesteps** (PTQ4DiT uses 25 calibration timesteps; AdaTSQ recommends Fisher-guided weighting emphasizing high-sensitivity early and late timesteps). SVD-LLM shows <15% variance across calibration data choices, so 64 samples should suffice.

**Ablation experiments (priority order)**:

1. **Rotation sweep**: No rotation → standard Hadamard → ConvRot group-wise regular Hadamard. Measure FID delta at W4A4 to validate outlier elimination before pushing lower
2. **SVD rank sweep**: r ∈ {0, 8, 16, 32, 64} with W4A4 residual quantization. Plot FID vs effective bits (accounting for FP16 low-rank overhead). SVDQuant uses relatively small ranks (16–64)
3. **Weight bit-width progression**: W4A4 → W3A4 → W2A4, each with optimal rotation + SVD rank from steps 1–2. Use GPTQ for INT3, QuIP# E8 codebook for INT2
4. **Activation precision sweep**: Fix weights at W3, sweep activations: A8 → A6 → A4 → A3. This isolates the activation quantization cliff
5. **Mixed-precision protection**: Incrementally protect components (timestep → cross-attention → FFN blocks 6/26 → first/last blocks) and measure per-component FID recovery
6. **Full pipeline**: Best rotation + SVD rank-32 + GPTQ-3bit residual + dynamic A4 per-group quantization + mixed-precision protection

**Evaluation protocol**. Generate **5K–10K images** using MJHQ-30K prompts with 20–50 DPM-Solver steps and CFG scale ~4.5. Compute FID against MJHQ reference set at 1024×1024 using clean-fid. Also report per-category FID, LPIPS, PSNR, ImageReward, and CLIP score. The NVFP4 baseline FID on PixArt-Σ from SVDQuant is near-FP16; target <2 FID degradation for W3A4 and <5 for W2A4.

**Expected outcomes based on literature extrapolation**:

| Configuration | Expected FID delta vs FP16 | Confidence |
|---|---|---|
| W4A4 SVDQuant+NVFP4 (baseline) | ~0–2 | High (published) |
| W3A4 rotation+SVD(r=32)+GPTQ3 | ~3–8 | Medium (extrapolated from LLM scaling) |
| W2A4 rotation+SVD(r=64)+QuIP#2 | ~8–20 | Low (uncharted territory for DiTs) |
| W3A3 full pipeline | ~10–25 | Low (W3A3 shows order-of-magnitude jump in LLMs) |
| W2A2 | Likely non-functional without QAT | High confidence it fails |

---

## Conclusion

The mathematical toolkit for sub-4-bit DiT quantization is mature for weights but nascent for activations. **Weight quantization to INT2–INT3 is achievable** through the rotation→SVD→GPTQ/QuIP# pipeline, with each stage providing compounding benefits: rotation reduces incoherence (enabling ~0.5-bit improvement), SVD absorbs the dominant quantization error directions (worth ~0.5–1 bit), and second-order optimization minimizes residual rounding error. The critical unknown is whether PixArt-XL-2's 0.6B parameters provide sufficient redundancy for sub-4-bit compression—ternary and 2-bit results in the literature overwhelmingly come from models >7B parameters, and TerDiT's ternary success required scaling to 4.2B. **Activation quantization below INT4 remains the binding constraint**: the W4→W3 activation gap is approximately 5–10× larger than the W8→W4 gap across all published results, and no diffusion-specific method has demonstrated functional W3A3. The most impactful first experiment is validating whether ConvRot rotation + SVDQuant's rank-32 decomposition + GPTQ-3bit residual can maintain FID within 5 points of the NVFP4 baseline at W3A4—if this succeeds, progressively lowering activation bits with Atom-style mixed-precision and TCAQ-DM's timestep-adaptive quantizers becomes the logical next step.