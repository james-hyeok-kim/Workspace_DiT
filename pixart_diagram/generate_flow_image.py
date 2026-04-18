"""
generate_flow_image.py

PixArt-α XL/2 1024-MS pipeline flow diagram → PNG
Usage: python generate_flow_image.py
Output: pixart_xl_1024_flow.png (same directory)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── 노드 정의 ──────────────────────────────────────────────────────────────────
# (label, x_center, y_center, width, height, color)
NODES = [
    # Input/Output
    ("INPUT\nprompt / height / width / num_steps / guidance_scale",
     0.50, 9.80, 3.00, 0.45, "#f0f9ff"),
    ("OUTPUT\nPIL.Image  1024×1024 RGB",
     0.50, 0.20, 3.00, 0.45, "#f0f9ff"),

    # Text path
    ("T5Tokenizer\nstr → input_ids [1,120]",
     -2.20, 8.70, 2.40, 0.50, "#dbeafe"),
    ("T5EncoderModel  (24L, 4.76B)\n[1,120] → [1,120,4096]",
     -2.20, 7.90, 2.40, 0.50, "#dbeafe"),
    ("negative_encode ('')\n→ [1,120,4096]",
     -2.20, 7.10, 2.40, 0.50, "#dbeafe"),
    ("CFG concat  (uncond + cond)\n→ [2,120,4096]  mask:[2,120]",
     -2.20, 6.30, 2.40, 0.50, "#dbeafe"),

    # Latent init
    ("prepare_latents\nrandn [1,4,128,128] × init_noise_σ",
     2.80, 8.30, 2.40, 0.50, "#fef9c3"),

    # Conditions
    ("added_cond_kwargs\nresolution[2,2]  aspect_ratio[2,1]",
     2.80, 7.40, 2.40, 0.50, "#f3e8ff"),
    ("retrieve_timesteps → [T]\nDPMSolverMultistep",
     2.80, 6.60, 2.40, 0.50, "#f3e8ff"),

    # Loop header
    ("Denoising Loop × T steps",
     0.50, 5.65, 6.00, 0.40, "#bbf7d0"),

    # Loop steps
    ("CFG latent concat\n[1,4,128,128] → [2,4,128,128]",
     0.50, 5.00, 2.80, 0.50, "#dcfce7"),
    ("scheduler.scale_model_input  [2,4,128,128]",
     0.50, 4.35, 2.80, 0.40, "#dcfce7"),
    ("timestep.expand → [2]",
     0.50, 3.85, 2.80, 0.35, "#dcfce7"),

    # DiT
    ("DiT: pos_embed  Conv2d(4,1152,k=2,s=2)\n[2,4,128,128] → [2,4096,1152]  N=4096 tokens",
     0.50, 3.20, 4.00, 0.50, "#d1fae5"),
    ("adaln_single(t+res+ar)\ntimestep:[2,6912]  embedded:[2,1152]",
     -1.80, 3.20, 2.40, 0.50, "#d1fae5"),
    ("caption_projection  4096→1152→1152\n[2,120,4096] → [2,120,1152]",
     2.80, 3.20, 2.40, 0.50, "#d1fae5"),

    # Blocks (one representative block)
    ("28 × BasicTransformerBlock",
     0.50, 2.60, 4.00, 0.35, "#a7f3d0"),
    ("Self-Attn  Q/K/V:[2,16,4096,72]\nQKᵀ:[2,16,4096,4096] → out:[2,4096,1152]",
     0.50, 2.10, 3.80, 0.50, "#a7f3d0"),
    ("Cross-Attn  Q:[2,16,4096,72]  KV:[2,16,120,72]\nQKᵀ:[2,16,4096,120] → out:[2,4096,1152]",
     0.50, 1.55, 3.80, 0.50, "#a7f3d0"),
    ("FFN  1152→4608→1152  (GELU-approx)\n→ [2,4096,1152]",
     0.50, 1.00, 3.80, 0.50, "#a7f3d0"),

    # DiT output head
    ("proj_out  1152→32  + unpatchify\n[2,4096,1152] → [2,8,128,128]",
     0.50, 0.40, 3.80, 0.45, "#d1fae5"),  # will be overridden by layout

    # CFG / scheduler / VAE  (placed after loop exit)
    ("CFG guidance  uncond+4.5×(text-uncond)\n[2,8,128,128] → [1,8,128,128]",
     0.50, -0.40, 3.00, 0.50, "#dcfce7"),
    ("learned-sigma split  → [1,4,128,128]",
     0.50, -1.00, 3.00, 0.40, "#dcfce7"),
    ("scheduler.step  → latents [1,4,128,128]",
     0.50, -1.55, 3.00, 0.40, "#dcfce7"),
    ("VAE decode  AutoencoderKL\n[1,4,128,128] → [1,3,1024,1024]",
     0.50, -2.30, 3.00, 0.50, "#ffe4e6"),
]

# ── 전체 레이아웃을 세로 스택으로 재설계 ──────────────────────────────────────
ROWS = [
    # label, color
    ("INPUT\nprompt  height/width  num_steps  guidance_scale", "#e0f2fe"),

    # --- Text ---
    ("[Text] T5Tokenizer\nstr  →  input_ids [1, 120]  +  mask [1, 120]", "#dbeafe"),
    ("[Text] T5EncoderModel  (24 layers, ~4.76 B)\n[1, 120]  →  [1, 120, 4096]", "#dbeafe"),
    ("[Text] negative_encode ('')  →  [1, 120, 4096]\nCFG concat (uncond+cond)  →  [2, 120, 4096]  mask:[2, 120]", "#dbeafe"),

    # --- Latent + Cond ---
    ("[Latent] prepare_latents  randn [1,4,128,128] × init_noise_σ\n[Cond] added_cond_kwargs  resolution[2,2]  aspect_ratio[2,1]  |  retrieve_timesteps → [T]", "#fef9c3"),

    # --- Loop ---
    ("▼  Denoising Loop  ×  T steps  ▼", "#bbf7d0"),
    ("[Loop] CFG latent concat  [1,4,128,128] → [2,4,128,128]\nscheduler.scale_model_input  |  timestep.expand → [2]", "#dcfce7"),

    # --- DiT Input ---
    ("[DiT-in] pos_embed  Conv2d(4→1152, k=2,s=2)  +  flatten  +  2D sinusoidal\n[2,4,128,128]  →  [2, 4096, 1152]   N = 64×64 = 4096 tokens", "#d1fae5"),
    ("[DiT-in] adaln_single (t + resolution + aspect_ratio)\n→ timestep [2, 6×1152]  |  embedded [2, 1152]\n[DiT-in] caption_projection  4096→1152→1152  (GELU)\n[2, 120, 4096]  →  [2, 120, 1152]", "#d1fae5"),

    # --- Blocks ---
    ("▼  28 × BasicTransformerBlock  ▼", "#6ee7b7"),
    ("[Block] adaLN split → shift1, scale1, gate1, shift2, scale2, gate2\nLN1 + modulate  x·(1+scale1)+shift1", "#a7f3d0"),
    ("[Block] Self-Attention\nQ/K/V : Linear(1152→1152, bias)  →  [2, 16, 4096, 72]\nQKᵀ logits : [2, 16, 4096, 4096]  (SDPA)\nout   : [2, 4096, 1152]  +  gate1 residual", "#a7f3d0"),
    ("[Block] Cross-Attention\nQ  : [2, 16, 4096, 72]   (from hidden)\nKV : [2, 16, 120,  72]   (from caption_projection)\nQKᵀ logits : [2, 16, 4096, 120]  +  encoder_attention_mask\nout : [2, 4096, 1152]  +  residual (no gate)", "#a7f3d0"),
    ("[Block] FFN  gelu-approx\n1152 → 4608 → 1152   +  gate2 residual\nfinal output : [2, 4096, 1152]", "#a7f3d0"),

    # --- DiT Output ---
    ("[DiT-out] norm_out (LN, affine=False)  +  adaLN modulate\nproj_out  Linear(1152 → 32)  →  [2, 4096, 32]   (32 = 2²×8)\nunpatchify  einsum nhwpqc→nchpwq  →  [2, 8, 128, 128]", "#d1fae5"),

    # --- CFG + Scheduler ---
    ("[CFG] guidance  uncond + 4.5×(text-uncond)\n[2, 8, 128, 128]  →  [1, 8, 128, 128]\nlearned-sigma split  chunk(2,dim=1)[0]  →  [1, 4, 128, 128]", "#dcfce7"),
    ("[Sched] scheduler.step  (DPM-Solver)\nnoise_pred + t + latents  →  latents [1, 4, 128, 128]\n▲  repeat T times  ▲", "#dcfce7"),

    # --- VAE ---
    ("[VAE] latents / 0.18215\nAutoencoderKL decode  blocks=[128,256,512,512]  (~84 M)\n[1, 4, 128, 128]  →  [1, 3, 1024, 1024]\nimage_processor.postprocess  →  denormalize", "#ffe4e6"),

    ("OUTPUT\nPIL.Image  1024 × 1024  RGB", "#e0f2fe"),
]

ARROW_COLOR = "#475569"
BOX_RADIUS  = 0.015

def draw_flow(rows, out_path):
    n      = len(rows)
    FIG_W  = 14
    ROW_H  = 1.05     # base height per row (will be scaled by line count)
    PAD    = 0.22
    FONT   = 9.5

    # compute heights
    heights = []
    for label, _ in rows:
        lines = label.count("\n") + 1
        heights.append(max(ROW_H, lines * 0.38 + PAD * 2))

    total_h = sum(heights) + (n - 1) * 0.18 + 0.8
    fig, ax = plt.subplots(figsize=(FIG_W, total_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")

    BOX_W = 0.82
    BOX_X = (1 - BOX_W) / 2

    y = total_h - 0.4
    centers = []

    for idx, ((label, color), h) in enumerate(zip(rows, heights)):
        y_top = y
        y_bot = y - h
        cy    = (y_top + y_bot) / 2
        centers.append(cy)

        is_header = label.startswith("▼") or label.startswith("▲") or label in (rows[0][0], rows[-1][0])
        lw   = 1.8 if is_header else 1.2
        ec   = "#1e40af" if "Text" in label or "TEXT" in label else \
               "#854d0e" if "Latent" in label or "Cond" in label else \
               "#166534" if any(x in label for x in ["Loop","Block","DiT","Sched","▼","▲"]) else \
               "#9f1239" if "VAE" in label else \
               "#1e3a5f"

        fancy = FancyBboxPatch(
            (BOX_X, y_bot + PAD * 0.3),
            BOX_W, h - PAD * 0.6,
            boxstyle=f"round,pad=0.01",
            linewidth=lw,
            edgecolor=ec,
            facecolor=color,
            zorder=2,
        )
        ax.add_patch(fancy)

        weight = "bold" if is_header else "normal"
        ax.text(0.5, cy, label,
                ha="center", va="center",
                fontsize=FONT, fontfamily="monospace",
                fontweight=weight,
                wrap=False, zorder=3,
                multialignment="center")

        y = y_bot - 0.18

    # arrows
    for i in range(len(centers) - 1):
        ax.annotate("",
            xy=(0.5, centers[i+1] + heights[i+1] / 2 + 0.02),
            xytext=(0.5, centers[i] - heights[i] / 2 - 0.02),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR,
                            lw=1.5, mutation_scale=14),
            zorder=4,
        )

    ax.set_title("PixArt-α XL/2 1024-MS  —  Input → Output Flow\n(batch=1, CFG ON, guidance_scale=4.5)",
                 fontsize=12, fontweight="bold", pad=10, color="#1e293b")

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pixart_xl_1024_flow.png")
    draw_flow(ROWS, out)
