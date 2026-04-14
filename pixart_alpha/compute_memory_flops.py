#!/usr/bin/env python3
"""
Compute per-layer Memory & Bit-Weighted FLOPs comparison CSVs for:
  1. Cascade-3 Mixed-Precision (Ours)  — weight-only MXFP6/MXFP8 mixed
  2. NVFP4 SVDQuant (DEFAULT_CFG)      — W4A4 + SVD low-rank correction

FLOPs are bit-weighted (normalized to FP8 = 1.0x):
  - Matmul factor = max(act_bits, wgt_bits) / 8  (upcast to higher precision)
  - NVFP4×NVFP4 → ×0.5    FP8×FP8 → ×1.0    FP16×any → ×2.0

Scale dequantization ops are also counted:
  - Each weight element × its block scale = params ops at scale precision

Outputs:
  results/sensitivity/cascade3_memory_flops.csv
  results/sensitivity/nvfp4_svd_memory_flops.csv
"""

import csv
import math
import os

# ─────────────────────────────────────────────────────────────────────
# Architecture constants
# ─────────────────────────────────────────────────────────────────────
NUM_BLOCKS = 28
HIDDEN_DIM = 1152
FFN_DIM    = 4608
NUM_TOKENS = 64 * 64   # 1024×1024 image → 128×128 latent → 64×64 patches = 4096

LAYERS_PER_BLOCK = [
    ("attn1.to_q",     HIDDEN_DIM, HIDDEN_DIM),
    ("attn1.to_k",     HIDDEN_DIM, HIDDEN_DIM),
    ("attn1.to_v",     HIDDEN_DIM, HIDDEN_DIM),
    ("attn1.to_out.0", HIDDEN_DIM, HIDDEN_DIM),
    ("attn2.to_q",     HIDDEN_DIM, HIDDEN_DIM),
    ("attn2.to_k",     HIDDEN_DIM, HIDDEN_DIM),
    ("attn2.to_v",     HIDDEN_DIM, HIDDEN_DIM),
    ("attn2.to_out.0", HIDDEN_DIM, HIDDEN_DIM),
    ("ff.net.0.proj",  FFN_DIM,    HIDDEN_DIM),
    ("ff.net.2",       HIDDEN_DIM, FFN_DIM),
]

# ─────────────────────────────────────────────────────────────────────
# Cascade-3 config (from final_config.csv)
# ─────────────────────────────────────────────────────────────────────
CASCADE3_CONFIG = {
    0:  ("MXFP8",      64),  1:  ("MXFP6_E2M3", 32),
    2:  ("MXFP8",      64),  3:  ("MXFP6_E2M3", 32),
    4:  ("MXFP8",      64),  5:  ("MXFP6_E2M3", 32),
    6:  ("MXFP6_E2M3", 32),  7:  ("MXFP8",      64),
    8:  ("MXFP8",      64),  9:  ("MXFP6_E2M3", 32),
    10: ("MXFP8",      64),  11: ("MXFP8",      64),
    12: ("MXFP6_E2M3", 32),  13: ("MXFP8",      64),
    14: ("MXFP6_E2M3", 32),  15: ("MXFP8",      64),
    16: ("MXFP8",      64),  17: ("MXFP6_E2M3", 32),
    18: ("MXFP6_E2M3", 32),  19: ("MXFP6_E2M3", 32),
    20: ("MXFP6_E2M3", 32),  21: ("MXFP8",      64),
    22: ("MXFP6_E2M3", 64),  23: ("MXFP6_E2M3", 32),
    24: ("MXFP6_E2M3", 32),  25: ("MXFP6_E2M3", 32),
    26: ("MXFP6_E2M3", 32),  27: ("MXFP8",      64),
}

# ─────────────────────────────────────────────────────────────────────
# Format → bits
# ─────────────────────────────────────────────────────────────────────
FORMAT_BITS = {
    "FP16": 16, "MXFP8": 8, "MXFP6_E2M3": 6, "NVFP4": 4,
}
SCALE_DTYPE_BITS = {
    "FP16": 16, "E8M0": 8, "FP8": 8,
}


# ─────────────────────────────────────────────────────────────────────
# Memory helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────
def weight_mem(params, wgt_bits):
    return params * wgt_bits / 8

def scale_mem(params, bs, scale_bytes):
    return math.ceil(params / bs) * scale_bytes

def svd_mem(out_f, in_f, rank, dtype_bytes=2):
    return (rank * in_f + out_f * rank) * dtype_bytes

def smooth_mem(in_f, dtype_bytes=2):
    return in_f * dtype_bytes

def act_bytes_per_token(features, bits):
    return features * bits / 8


# ─────────────────────────────────────────────────────────────────────
# Bit-weighted FLOPs helpers
# ─────────────────────────────────────────────────────────────────────

def matmul_flops(out_f: int, in_f: int, act_bits: int, wgt_bits: int) -> float:
    """Bit-weighted matmul FLOPs (normalized to FP8=1.0x).

    Y = X @ W^T → 2 × out × in MACs
    Compute precision = max(act_bits, wgt_bits) — lower-precision operand
    is upcast to the higher precision before multiply-accumulate.
    Factor = max(act_bits, wgt_bits) / 8.

    Examples:
      FP16 × FP16  → 2 × O × I × (16/8) = 2 × O × I × 2.0
      FP8  × FP8   → 2 × O × I × (8/8)  = 2 × O × I × 1.0
      NVFP4 × NVFP4→ 2 × O × I × (4/8)  = 2 × O × I × 0.5
      FP16 × MXFP8 → 2 × O × I × (16/8) = 2 × O × I × 2.0  (upcast to FP16)
      FP16 × MXFP6 → 2 × O × I × (16/8) = 2 × O × I × 2.0  (upcast to FP16)
      NVFP4 × NVFP4→ 2 × O × I × (4/8)  = 2 × O × I × 0.5  (both 4-bit)
    """
    factor = max(act_bits, wgt_bits) / 8.0
    return 2 * out_f * in_f * factor


def scale_dequant_flops(params: int, block_size: int, scale_bits: int) -> float:
    """Scale dequantization ops: each weight element × its block scale.

    w_dequant[i] = w_quant[i] × scale[i // block_size]
    = params multiplications, each at scale precision.
    Cost factor = scale_bits / 8.
    """
    factor = scale_bits / 8.0
    return params * factor


def svd_branch_flops(out_f: int, in_f: int, rank: int, dtype_bits: int = 16) -> float:
    """SVD correction branch FLOPs (FP16 operands).

    h = X_smooth @ lora_a^T  → 2 × rank × in_f
    Y_svd = h @ lora_b^T     → 2 × out_f × rank
    Both at FP16 → factor = 16/8 = 2.0
    """
    factor = dtype_bits / 8.0
    return (2 * rank * in_f + 2 * out_f * rank) * factor


def smooth_scale_flops(in_f: int, dtype_bits: int = 16) -> float:
    """SmoothQuant: x_smooth = x * smooth_scale  → in_f element-wise muls at FP16."""
    factor = dtype_bits / 8.0
    return in_f * factor


def hadamard_rotate_flops(in_f: int, had_block_size: int = 128,
                          dtype_bits: int = 16) -> float:
    """Hadamard rotation FLOPs per token.

    _hadamard_rotate(x):
      1. sign flip: x_s = x * S_in            → in_f element-wise muls
      2. block matmul: x_3d @ H_block          → n_blocks x (1x128 @ 128x128)
         = n_blocks x 2 x 128 x 128 / 128 ops per output element x 128 elements
         = n_blocks x 2 x 128² = (in_f/128) x 2 x 128² = in_f x 256 FLOPs

    Total = in_f x (1 + 256) x (dtype_bits/8)
    """
    n_blocks = in_f // had_block_size
    factor = dtype_bits / 8.0
    sign_flops   = in_f * factor
    matmul_flops = n_blocks * 2 * had_block_size * had_block_size * factor
    return sign_flops + matmul_flops


def act_quant_flops(in_f: int, act_bits: int) -> float:
    """Activation quantization overhead (quantize + dequantize ≈ 2 ops/element)."""
    if act_bits >= 16:
        return 0  # no quantization
    factor = act_bits / 8.0
    return in_f * 2 * factor  # quant + dequant


# ─────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────
def fmt_f(val):
    if val >= 1e12: return f"{val/1e12:.2f} TFLOPs"
    if val >= 1e9:  return f"{val/1e9:.2f} GFLOPs"
    if val >= 1e6:  return f"{val/1e6:.2f} MFLOPs"
    if val >= 1e3:  return f"{val/1e3:.1f} KFLOPs"
    return f"{val:.0f}"

def fmt_b(val):
    if val >= 1024**3: return f"{val/1024**3:.2f} GB"
    if val >= 1024**2: return f"{val/1024**2:.2f} MB"
    if val >= 1024:    return f"{val/1024:.1f} KB"
    return f"{val:.0f} B"


# ─────────────────────────────────────────────────────────────────────
# Cascade-3 CSV
# ─────────────────────────────────────────────────────────────────────
def generate_cascade3_csv(out_path: str):
    F = [
        "block", "layer", "shape (OxI)",
        "act_in_dtype", "act_in_bits", "act_in_bytes/token",
        "weight_dtype", "weight_bits", "block_size", "scale_dtype", "scale_bits",
        "weight_bytes", "scale_bytes", "total_weight_bytes",
        "act_out_dtype", "act_out_bits", "act_out_bytes/token",
        # FLOPs breakdown
        "matmul_formula", "matmul_factor",
        "matmul_flops/token", "scale_dequant_flops/token", "total_flops/token",
        "matmul_flops/image", "scale_dequant_flops/image", "total_flops/image",
        "eff_bits/weight",
    ]

    rows = []
    # ── Explanation header ──
    rows.append({k: "" for k in F})
    h = {k: "" for k in F}; h["block"] = "=== BIT-WEIGHTED FLOPs CALCULATION ==="; rows.append(h)
    for lbl, desc in [
        ("Matmul (bit-weighted)",
         "2 x O x I x factor,  factor = max(act_bits, wgt_bits) / 8    [FP16×any→x2.0  FP8×FP8→x1.0  NVFP4×NVFP4→x0.5]"),
        ("Scale dequant ops",
         "params x (scale_bits / 8)    [each weight element × its block scale factor]"),
        ("Total per token",
         "matmul_flops + scale_dequant_flops"),
        ("Per image (1024x1024)",
         f"flops/token x {NUM_TOKENS} tokens    [128x128 latent / patch_size=2 = 64x64 = {NUM_TOKENS}]"),
        ("Cascade-3 note",
         "Weight-only quant → act=FP16 preserved → no SVD branch, no SmoothQuant, no act quant overhead"),
    ]:
        r = {k: "" for k in F}; r["block"] = lbl; r["layer"] = desc; rows.append(r)
    rows.append({k: "" for k in F})

    # Accumulators
    T = dict(fp16_wt=0, wt=0, sc=0, stor=0, mm=0, sd=0, tot=0, params=0, ai=0, ao=0)

    for blk in range(NUM_BLOCKS):
        fmt, bs = CASCADE3_CONFIG[blk]
        wb = FORMAT_BITS[fmt]
        ab = 16  # activation = FP16 (weight-only)

        if fmt == "MXFP8":
            sd, sb = "FP16", 16
        else:
            sd, sb = "E8M0", 8

        for ln, of, inf in LAYERS_PER_BLOCK:
            p = of * inf
            # Memory
            wm = weight_mem(p, wb)
            sm = scale_mem(p, bs, sb // 8)
            tw = wm + sm
            ai = act_bytes_per_token(inf, ab)
            ao = act_bytes_per_token(of, 16)
            # FLOPs
            mm = matmul_flops(of, inf, ab, wb)
            sdf = scale_dequant_flops(p, bs, sb)
            tot = mm + sdf

            fac = max(ab, wb) / 8.0
            formula = f"2 x {of} x {inf} x {fac}"

            rows.append({
                "block": blk, "layer": ln, "shape (OxI)": f"{of}x{inf}",
                "act_in_dtype": "FP16", "act_in_bits": ab,
                "act_in_bytes/token": round(ai, 1),
                "weight_dtype": fmt, "weight_bits": wb,
                "block_size": bs, "scale_dtype": sd, "scale_bits": sb,
                "weight_bytes": int(wm), "scale_bytes": int(sm),
                "total_weight_bytes": int(tw),
                "act_out_dtype": "FP16", "act_out_bits": 16,
                "act_out_bytes/token": round(ao, 1),
                "matmul_formula": formula, "matmul_factor": fac,
                "matmul_flops/token": int(mm),
                "scale_dequant_flops/token": int(sdf),
                "total_flops/token": int(tot),
                "matmul_flops/image": int(mm * NUM_TOKENS),
                "scale_dequant_flops/image": int(sdf * NUM_TOKENS),
                "total_flops/image": int(tot * NUM_TOKENS),
                "eff_bits/weight": round(tw * 8 / p, 2),
            })
            T["fp16_wt"] += p * 2; T["wt"] += wm; T["sc"] += sm; T["stor"] += tw
            T["mm"] += mm; T["sd"] += sdf; T["tot"] += tot
            T["params"] += p; T["ai"] += ai; T["ao"] += ao

    # ── Block summary ──
    rows.append({k: "" for k in F})
    h = {k: "" for k in F}; h["block"] = "=== BLOCK SUMMARY ==="; rows.append(h)

    for blk in range(NUM_BLOCKS):
        fmt, bs = CASCADE3_CONFIG[blk]
        wb = FORMAT_BITS[fmt]; ab = 16
        sd, sb = ("FP16", 16) if fmt == "MXFP8" else ("E8M0", 8)
        bw = bs_ = bmm = bsd = bp = 0
        for _, o, i in LAYERS_PER_BLOCK:
            p = o * i
            bw += weight_mem(p, wb); bs_ += scale_mem(p, bs, sb // 8)
            bmm += matmul_flops(o, i, ab, wb)
            bsd += scale_dequant_flops(p, bs, sb)
            bp += p
        bt = bw + bs_; btot = bmm + bsd
        rows.append({
            "block": f"Block {blk}", "layer": "10 layers",
            "shape (OxI)": f"{bp:,} params",
            "act_in_dtype": "FP16", "act_in_bits": ab, "act_in_bytes/token": "",
            "weight_dtype": fmt, "weight_bits": wb,
            "block_size": bs, "scale_dtype": sd, "scale_bits": sb,
            "weight_bytes": int(bw), "scale_bytes": int(bs_),
            "total_weight_bytes": int(bt),
            "act_out_dtype": "FP16", "act_out_bits": 16, "act_out_bytes/token": "",
            "matmul_formula": f"sum 10 layers x{max(ab,wb)/8}", "matmul_factor": max(ab, wb)/8,
            "matmul_flops/token": int(bmm), "scale_dequant_flops/token": int(bsd),
            "total_flops/token": int(btot),
            "matmul_flops/image": int(bmm * NUM_TOKENS),
            "scale_dequant_flops/image": int(bsd * NUM_TOKENS),
            "total_flops/image": int(btot * NUM_TOKENS),
            "eff_bits/weight": round(bt * 8 / bp, 2),
        })

    # ── Grand summary ──
    rows.append({k: "" for k in F})
    h = {k: "" for k in F}; h["block"] = "=== GRAND SUMMARY ==="; rows.append(h)

    fp16_mb = T["fp16_wt"] / 1024**2
    tot_mb  = T["stor"] / 1024**2
    comp    = T["fp16_wt"] / T["stor"] if T["stor"] else 0
    m8 = sum(1 for f, _ in CASCADE3_CONFIG.values() if f == "MXFP8")

    # FP16 baseline FLOPs (same MACs but all at FP16 factor=2.0)
    fp16_flops_pt = sum(2 * o * i * 2.0 for _, o, i in LAYERS_PER_BLOCK) * NUM_BLOCKS

    for lbl, val in [
        ("Method",              "Cascade-3 Mixed-Precision (Weight-Only)"),
        ("Model",               "PixArt-Sigma (600M params, 28 DiT blocks)"),
        ("Image",               f"1024x1024 → {NUM_TOKENS} tokens"),
        ("", ""),
        ("--- Config ---",      ""),
        ("MXFP8 Blocks",        f"{m8} (act=FP16, wgt=8bit → matmul factor=1.0)"),
        ("MXFP6_E2M3 Blocks",   f"{NUM_BLOCKS-m8} (act=FP16, wgt=6bit → matmul factor=0.75)"),
        ("Activation Quant",    "None — FP16 preserved"),
        ("SVD Correction",      "None"),
        ("", ""),
        ("--- Storage ---",     ""),
        ("Total Parameters",    f"{T['params']:,}"),
        ("FP16 Baseline",       f"{fp16_mb:.2f} MB"),
        ("Weight Storage",      f"{T['wt']/1024**2:.2f} MB"),
        ("Scale Storage",       f"{T['sc']/1024**2:.2f} MB"),
        ("Total Storage",       f"{tot_mb:.2f} MB"),
        ("Compression",         f"{comp:.2f}x vs FP16"),
        ("Eff Bits/Weight",     f"{T['stor']*8/T['params']:.2f}"),
        ("", ""),
        ("--- Bit-Weighted FLOPs (per token, FP8=1.0x) ---", ""),
        ("Matmul FLOPs/Token",         f"{T['mm']:,.0f}  ({fmt_f(T['mm'])})"),
        ("Scale Dequant FLOPs/Token",  f"{T['sd']:,.0f}  ({fmt_f(T['sd'])})"),
        ("Total FLOPs/Token",          f"{T['tot']:,.0f}  ({fmt_f(T['tot'])})"),
        ("", ""),
        ("--- Bit-Weighted FLOPs (per image) ---", ""),
        ("Matmul FLOPs/Image",         f"{T['mm']*NUM_TOKENS:,.0f}  ({fmt_f(T['mm']*NUM_TOKENS)})"),
        ("Scale Dequant FLOPs/Image",  f"{T['sd']*NUM_TOKENS:,.0f}  ({fmt_f(T['sd']*NUM_TOKENS)})"),
        ("Total FLOPs/Image",          f"{T['tot']*NUM_TOKENS:,.0f}  ({fmt_f(T['tot']*NUM_TOKENS)})"),
        ("vs FP16 (all x2.0)",         f"{T['tot']/fp16_flops_pt:.4f}x  (FP16 total: {fmt_f(fp16_flops_pt*NUM_TOKENS)}/image)"),
        ("", ""),
        ("--- Activation Memory ---", ""),
        ("Act In/Token (all layers)",  f"{T['ai']:.0f} bytes ({fmt_b(T['ai'])})"),
        ("Act In/Image (all layers)",  f"{T['ai']*NUM_TOKENS:.0f} bytes ({fmt_b(T['ai']*NUM_TOKENS)})"),
    ]:
        r = {k: "" for k in F}; r["block"] = lbl; r["layer"] = val; rows.append(r)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=F); w.writeheader(); w.writerows(rows)
    print(f"✅ Saved: {out_path}  ({len(rows)} rows)")
    return T


# ─────────────────────────────────────────────────────────────────────
# NVFP4 SVDQuant CSV
# ─────────────────────────────────────────────────────────────────────
def generate_nvfp4_svd_csv(out_path: str, svd_rank: int = 32, wgt_bs: int = 16):
    F = [
        "block", "layer", "shape (OxI)",
        "act_in_dtype", "act_in_bits", "act_in_bytes/token",
        "weight_dtype", "weight_bits", "block_size", "scale_dtype", "scale_bits",
        "weight_bytes", "scale_bytes", "svd_bytes", "smooth_bytes", "total_weight_bytes",
        "act_out_dtype", "act_out_bits", "act_out_bytes/token",
        # FLOPs breakdown
        "matmul_formula", "matmul_factor",
        "matmul_flops/token", "scale_dequant_flops/token",
        "svd_flops/token", "smooth_flops/token", "act_quant_flops/token",
        "total_flops/token",
        "matmul_flops/image", "scale_dequant_flops/image",
        "svd_flops/image", "smooth_flops/image", "act_quant_flops/image",
        "total_flops/image",
        "eff_bits/weight",
    ]

    rows = []
    rows.append({k: "" for k in F})
    h = {k: "" for k in F}; h["block"] = "=== BIT-WEIGHTED FLOPs CALCULATION ==="; rows.append(h)
    for lbl, desc in [
        ("Matmul (bit-weighted)",
         f"2 x O x I x factor,  factor = max(act_bits, wgt_bits)/8    [NVFP4 x NVFP4 → max(4,4)/8 = 0.5]"),
        ("Scale dequant ops",
         f"params x (scale_bits/8)    [FP16 scale → x2.0, FP8 scale → x1.0]"),
        ("SVD branch (FP16)",
         f"(2 x rank x I + 2 x O x rank) x (16/8)    [lora_a + lora_b, both FP16 → x2.0]"),
        ("SmoothQuant scale",
         f"in_features x (16/8)    [per-channel FP16 element-wise multiply → x2.0]"),
        ("Act quantization",
         f"in_features x 2 x (act_bits/8)    [quantize + dequantize overhead, 0 if FP16]"),
        ("Total per token",
         "matmul + scale_dequant + svd + smooth + act_quant"),
        ("Per image",
         f"flops/token x {NUM_TOKENS} tokens"),
    ]:
        r = {k: "" for k in F}; r["block"] = lbl; r["layer"] = desc; rows.append(r)
    rows.append({k: "" for k in F})

    # Accumulators
    T = dict(fp16_wt=0, wt=0, sc=0, sv=0, sm=0, stor=0,
             mm=0, sd=0, svf=0, smf=0, aqf=0, tot=0,
             params=0, ai=0, ao=0)

    wb = 4; ab = 4; sd_name = "FP16"; sd_bits = 16; s_each = 2

    for blk in range(NUM_BLOCKS):
        for ln, of, inf in LAYERS_PER_BLOCK:
            p = of * inf
            # Memory
            wm = weight_mem(p, wb)
            sm_bytes = scale_mem(p, wgt_bs, s_each)
            sv_bytes = svd_mem(of, inf, svd_rank, 2)
            smo_bytes = smooth_mem(inf, 2)
            tw = wm + sm_bytes + sv_bytes + smo_bytes
            ai = act_bytes_per_token(inf, ab)
            ao = act_bytes_per_token(of, 16)

            # FLOPs
            mm   = matmul_flops(of, inf, ab, wb)
            sdf  = scale_dequant_flops(p, wgt_bs, sd_bits)
            svf  = svd_branch_flops(of, inf, svd_rank, 16)
            smof = smooth_scale_flops(inf, 16)
            aqf  = act_quant_flops(inf, ab)
            tot  = mm + sdf + svf + smof + aqf

            fac = max(ab, wb) / 8.0
            formula = f"2 x {of} x {inf} x {fac}"

            rows.append({
                "block": blk, "layer": ln, "shape (OxI)": f"{of}x{inf}",
                "act_in_dtype": "NVFP4", "act_in_bits": ab,
                "act_in_bytes/token": round(ai, 1),
                "weight_dtype": "NVFP4", "weight_bits": wb,
                "block_size": wgt_bs, "scale_dtype": sd_name, "scale_bits": sd_bits,
                "weight_bytes": int(wm), "scale_bytes": int(sm_bytes),
                "svd_bytes": int(sv_bytes), "smooth_bytes": int(smo_bytes),
                "total_weight_bytes": int(tw),
                "act_out_dtype": "FP16", "act_out_bits": 16,
                "act_out_bytes/token": round(ao, 1),
                "matmul_formula": formula, "matmul_factor": fac,
                "matmul_flops/token": int(mm),
                "scale_dequant_flops/token": int(sdf),
                "svd_flops/token": int(svf),
                "smooth_flops/token": int(smof),
                "act_quant_flops/token": int(aqf),
                "total_flops/token": int(tot),
                "matmul_flops/image": int(mm * NUM_TOKENS),
                "scale_dequant_flops/image": int(sdf * NUM_TOKENS),
                "svd_flops/image": int(svf * NUM_TOKENS),
                "smooth_flops/image": int(smof * NUM_TOKENS),
                "act_quant_flops/image": int(aqf * NUM_TOKENS),
                "total_flops/image": int(tot * NUM_TOKENS),
                "eff_bits/weight": round(tw * 8 / p, 2),
            })
            T["fp16_wt"] += p*2; T["wt"] += wm; T["sc"] += sm_bytes
            T["sv"] += sv_bytes; T["sm"] += smo_bytes; T["stor"] += tw
            T["mm"] += mm; T["sd"] += sdf; T["svf"] += svf
            T["smf"] += smof; T["aqf"] += aqf; T["tot"] += tot
            T["params"] += p; T["ai"] += ai; T["ao"] += ao

    # ── Block summary ──
    rows.append({k: "" for k in F})
    h = {k: "" for k in F}; h["block"] = "=== BLOCK SUMMARY ==="; rows.append(h)

    for blk in range(NUM_BLOCKS):
        bw = bs_ = bsv = bsm = 0; bmm = bsd = bsvf = bsmf = baq = 0; bp = 0
        for _, o, i in LAYERS_PER_BLOCK:
            p = o*i; bw += weight_mem(p, wb); bs_ += scale_mem(p, wgt_bs, s_each)
            bsv += svd_mem(o, i, svd_rank, 2); bsm += smooth_mem(i, 2)
            bmm += matmul_flops(o, i, ab, wb)
            bsd += scale_dequant_flops(p, wgt_bs, sd_bits)
            bsvf += svd_branch_flops(o, i, svd_rank, 16)
            bsmf += smooth_scale_flops(i, 16); baq += act_quant_flops(i, ab)
            bp += p
        bt = bw + bs_ + bsv + bsm
        btot = bmm + bsd + bsvf + bsmf + baq
        rows.append({
            "block": f"Block {blk}", "layer": "10 layers",
            "shape (OxI)": f"{bp:,} params",
            "act_in_dtype": "NVFP4", "act_in_bits": ab, "act_in_bytes/token": "",
            "weight_dtype": "NVFP4", "weight_bits": wb,
            "block_size": wgt_bs, "scale_dtype": sd_name, "scale_bits": sd_bits,
            "weight_bytes": int(bw), "scale_bytes": int(bs_),
            "svd_bytes": int(bsv), "smooth_bytes": int(bsm),
            "total_weight_bytes": int(bt),
            "act_out_dtype": "FP16", "act_out_bits": 16, "act_out_bytes/token": "",
            "matmul_formula": f"sum 10 layers x{max(ab,wb)/8}", "matmul_factor": max(ab,wb)/8,
            "matmul_flops/token": int(bmm), "scale_dequant_flops/token": int(bsd),
            "svd_flops/token": int(bsvf), "smooth_flops/token": int(bsmf),
            "act_quant_flops/token": int(baq), "total_flops/token": int(btot),
            "matmul_flops/image": int(bmm*NUM_TOKENS), "scale_dequant_flops/image": int(bsd*NUM_TOKENS),
            "svd_flops/image": int(bsvf*NUM_TOKENS), "smooth_flops/image": int(bsmf*NUM_TOKENS),
            "act_quant_flops/image": int(baq*NUM_TOKENS), "total_flops/image": int(btot*NUM_TOKENS),
            "eff_bits/weight": round(bt*8/bp, 2),
        })

    # ── Grand summary ──
    rows.append({k: "" for k in F})
    h = {k: "" for k in F}; h["block"] = "=== GRAND SUMMARY ==="; rows.append(h)

    fp16_mb = T["fp16_wt"] / 1024**2
    tot_mb  = T["stor"] / 1024**2
    comp    = T["fp16_wt"] / T["stor"] if T["stor"] else 0
    fp16_flops_pt = sum(2*o*i*2.0 for _, o, i in LAYERS_PER_BLOCK) * NUM_BLOCKS

    for lbl, val in [
        ("Method",              "NVFP4 SVDQuant (NVFP4_SVDQUANT_DEFAULT_CFG)"),
        ("Model",               "PixArt-Sigma (600M params, 28 DiT blocks)"),
        ("Image",               f"1024x1024 → {NUM_TOKENS} tokens"),
        ("", ""),
        ("--- Config ---",      ""),
        ("Weight Format",       f"NVFP4 (4-bit) uniform, all 28 blocks, block_size={wgt_bs}"),
        ("Activation Quant",    "NVFP4 (4-bit) dynamic → matmul factor = max(4,4)/8 = 0.5"),
        ("SVD Rank",            f"{svd_rank} (FP16 lora_a + lora_b → factor x2.0)"),
        ("SmoothQuant",         "alpha=0.5, FP16 per-channel scale → factor x2.0"),
        ("Scale Dtype",         f"{sd_name} ({sd_bits}-bit) → dequant factor x{sd_bits/8}"),
        ("", ""),
        ("--- Storage ---",     ""),
        ("Total Parameters",    f"{T['params']:,}"),
        ("FP16 Baseline",       f"{fp16_mb:.2f} MB"),
        ("Weight Storage",      f"{T['wt']/1024**2:.2f} MB"),
        ("Scale Storage",       f"{T['sc']/1024**2:.2f} MB"),
        ("SVD Branch",          f"{T['sv']/1024**2:.2f} MB"),
        ("SmoothQuant Scale",   f"{T['sm']/1024**2:.4f} MB"),
        ("Total Storage",       f"{tot_mb:.2f} MB"),
        ("Compression",         f"{comp:.2f}x vs FP16"),
        ("Eff Bits/Weight",     f"{T['stor']*8/T['params']:.2f}"),
        ("", ""),
        ("--- Bit-Weighted FLOPs (per token, FP8=1.0x) ---", ""),
        ("Matmul (NVFP4×NVFP4, x0.5)",    f"{T['mm']:,.0f}  ({fmt_f(T['mm'])})"),
        ("Scale Dequant (FP16, x2.0)",     f"{T['sd']:,.0f}  ({fmt_f(T['sd'])})"),
        ("SVD Branch (FP16, x2.0)",        f"{T['svf']:,.0f}  ({fmt_f(T['svf'])})"),
        ("SmoothQuant (FP16, x2.0)",       f"{T['smf']:,.0f}  ({fmt_f(T['smf'])})"),
        ("Act Quant Overhead",             f"{T['aqf']:,.0f}  ({fmt_f(T['aqf'])})"),
        ("Total FLOPs/Token",              f"{T['tot']:,.0f}  ({fmt_f(T['tot'])})"),
        ("", ""),
        ("--- Bit-Weighted FLOPs (per image) ---", ""),
        ("Matmul FLOPs/Image",        f"{T['mm']*NUM_TOKENS:,.0f}  ({fmt_f(T['mm']*NUM_TOKENS)})"),
        ("Scale Dequant/Image",       f"{T['sd']*NUM_TOKENS:,.0f}  ({fmt_f(T['sd']*NUM_TOKENS)})"),
        ("SVD Branch/Image",          f"{T['svf']*NUM_TOKENS:,.0f}  ({fmt_f(T['svf']*NUM_TOKENS)})"),
        ("SmoothQuant/Image",         f"{T['smf']*NUM_TOKENS:,.0f}  ({fmt_f(T['smf']*NUM_TOKENS)})"),
        ("Act Quant/Image",           f"{T['aqf']*NUM_TOKENS:,.0f}  ({fmt_f(T['aqf']*NUM_TOKENS)})"),
        ("Total FLOPs/Image",         f"{T['tot']*NUM_TOKENS:,.0f}  ({fmt_f(T['tot']*NUM_TOKENS)})"),
        ("vs FP16 (all x2.0)",        f"{T['tot']/fp16_flops_pt:.4f}x  (FP16: {fmt_f(fp16_flops_pt*NUM_TOKENS)}/image)"),
        ("", ""),
        ("--- FLOPs Breakdown % ---", ""),
        ("Matmul %",            f"{T['mm']/T['tot']*100:.1f}%"),
        ("Scale Dequant %",     f"{T['sd']/T['tot']*100:.1f}%"),
        ("SVD Branch %",        f"{T['svf']/T['tot']*100:.1f}%"),
        ("SmoothQuant %",       f"{T['smf']/T['tot']*100:.1f}%"),
        ("Act Quant %",         f"{T['aqf']/T['tot']*100:.1f}%"),
        ("", ""),
        ("--- Activation Memory ---", ""),
        ("Act In/Token (NVFP4)", f"{T['ai']:.0f} bytes ({fmt_b(T['ai'])})"),
        ("Act In/Image",         f"{T['ai']*NUM_TOKENS:.0f} bytes ({fmt_b(T['ai']*NUM_TOKENS)})"),
    ]:
        r = {k: "" for k in F}; r["block"] = lbl; r["layer"] = val; rows.append(r)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=F); w.writeheader(); w.writerows(rows)
    print(f"✅ Saved: {out_path}  ({len(rows)} rows)")
    return T


# ─────────────────────────────────────────────────────────────────────
# Cascade-3 + Hadamard CSV
# ─────────────────────────────────────────────────────────────────────
def generate_cascade_hadamard_csv(out_path: str, svd_rank: int = 32,
                                   had_block_size: int = 128, act_block_size: int = 16):
    """Cascade-3 MXFP weight (per-block) + Hadamard rotation + NVFP4 activation + SVD.

    Key differences vs Cascade-3 (weight-only):
      - Activation: NVFP4 (4-bit) instead of FP16
        → matmul factor = max(NVFP4=4, wgt_bits) / 8
          MXFP8 block: max(4,8)/8 = 1.0  (was 2.0)
          MXFP6 block: max(4,6)/8 = 0.75 (was 2.0)
      - SVD error correction (rank=32 FP16 lora_a + lora_b)
      - Hadamard rotation (sign flip + block matmul, in_f x 257 per token at FP16)
      - Act quant overhead (in_f x 2 x (4/8) per token)
      - S_in buffer (in_f x 2 bytes per layer)
      - H_block (128x128 x2 bytes, shared globally = 32,768 bytes total)
      - No SmoothQuant
    """
    HAD_BLOCK_SIZE = had_block_size
    H_SHARED_BYTES = HAD_BLOCK_SIZE * HAD_BLOCK_SIZE * 2  # shared, counted once

    F = [
        "block", "layer", "shape (OxI)",
        "act_in_dtype", "act_in_bits", "act_in_bytes/token",
        "weight_dtype", "weight_bits", "block_size", "scale_dtype", "scale_bits",
        "weight_bytes", "scale_bytes", "svd_bytes", "sin_bytes", "total_weight_bytes",
        "act_out_dtype", "act_out_bits", "act_out_bytes/token",
        "matmul_formula", "matmul_factor",
        "matmul_flops/token", "scale_dequant_flops/token",
        "svd_flops/token", "hadamard_flops/token", "act_quant_flops/token",
        "total_flops/token",
        "matmul_flops/image", "scale_dequant_flops/image",
        "svd_flops/image", "hadamard_flops/image", "act_quant_flops/image",
        "total_flops/image",
        "eff_bits/weight",
    ]

    rows = []
    rows.append({k: "" for k in F})
    h = {k: "" for k in F}; h["block"] = "=== BIT-WEIGHTED FLOPs CALCULATION ==="; rows.append(h)
    for lbl, desc in [
        ("Matmul (bit-weighted)",
         "2 x O x I x max(act_bits, wgt_bits)/8   "
         "[NVFP4+MXFP8→max(4,8)/8=1.0  NVFP4+MXFP6→max(4,6)/8=0.75  FP16→x2.0]"),
        ("Scale dequant ops",
         "params x (scale_bits/8)   [MXFP8 FP16-scale→x2.0  MXFP6 E8M0-scale→x1.0]"),
        ("SVD branch (FP16)",
         "(2 x rank x I + 2 x O x rank) x (16/8)   [lora_a + lora_b, FP16→x2.0]"),
        ("Hadamard rotation (FP16)",
         "in_f x (1 + 256) x (16/8)   "
         "[sign flip (in_f) + block matmul ((in_f/128) x 2 x 128^2)]"),
        ("Act quant overhead (NVFP4)",
         "in_f x 2 x (4/8)   [quantize + dequantize, 4-bit→x0.5]"),
        ("No SmoothQuant", "SmoothQuant 없음"),
        ("Per image", f"flops/token x {NUM_TOKENS} tokens"),
    ]:
        r = {k: "" for k in F}; r["block"] = lbl; r["layer"] = desc; rows.append(r)
    rows.append({k: "" for k in F})

    T = dict(fp16_wt=0, wt=0, sc=0, sv=0, sin=0, stor=0,
             mm=0, sd=0, svf=0, hdf=0, aqf=0, tot=0,
             params=0, ai=0, ao=0)

    ab = 4  # activation = NVFP4

    for blk in range(NUM_BLOCKS):
        fmt, bs = CASCADE3_CONFIG[blk]
        wb = FORMAT_BITS[fmt]

        # scale dtype: MXFP8→FP16(16bit), MXFP6→E8M0(8bit)
        if fmt == "MXFP8":
            sd_name, sb = "FP16", 16
        else:
            sd_name, sb = "E8M0", 8

        for ln, of, inf in LAYERS_PER_BLOCK:
            p = of * inf
            # ── Memory ──
            wm       = weight_mem(p, wb)
            sm_bytes = scale_mem(p, bs, sb // 8)
            sv_bytes = svd_mem(of, inf, svd_rank, 2)
            sin_bytes = inf * 2                         # S_in per layer (FP16)
            tw = wm + sm_bytes + sv_bytes + sin_bytes
            ai = act_bytes_per_token(inf, ab)
            ao = act_bytes_per_token(of, 16)

            # ── FLOPs ──
            mm   = matmul_flops(of, inf, ab, wb)        # NVFP4 act × MXFP weight
            sdf  = scale_dequant_flops(p, bs, sb)
            svf  = svd_branch_flops(of, inf, svd_rank, 16)
            hdf  = hadamard_rotate_flops(inf, HAD_BLOCK_SIZE, 16)
            aqf  = act_quant_flops(inf, ab)
            tot  = mm + sdf + svf + hdf + aqf

            fac = max(ab, wb) / 8.0
            formula = f"2 x {of} x {inf} x {fac}"

            rows.append({
                "block": blk, "layer": ln, "shape (OxI)": f"{of}x{inf}",
                "act_in_dtype": "NVFP4", "act_in_bits": ab,
                "act_in_bytes/token": round(ai, 1),
                "weight_dtype": fmt, "weight_bits": wb,
                "block_size": bs, "scale_dtype": sd_name, "scale_bits": sb,
                "weight_bytes": int(wm), "scale_bytes": int(sm_bytes),
                "svd_bytes": int(sv_bytes), "sin_bytes": int(sin_bytes),
                "total_weight_bytes": int(tw),
                "act_out_dtype": "FP16", "act_out_bits": 16,
                "act_out_bytes/token": round(ao, 1),
                "matmul_formula": formula, "matmul_factor": fac,
                "matmul_flops/token": int(mm),
                "scale_dequant_flops/token": int(sdf),
                "svd_flops/token": int(svf),
                "hadamard_flops/token": int(hdf),
                "act_quant_flops/token": int(aqf),
                "total_flops/token": int(tot),
                "matmul_flops/image": int(mm * NUM_TOKENS),
                "scale_dequant_flops/image": int(sdf * NUM_TOKENS),
                "svd_flops/image": int(svf * NUM_TOKENS),
                "hadamard_flops/image": int(hdf * NUM_TOKENS),
                "act_quant_flops/image": int(aqf * NUM_TOKENS),
                "total_flops/image": int(tot * NUM_TOKENS),
                "eff_bits/weight": round(tw * 8 / p, 2),
            })
            T["fp16_wt"] += p * 2; T["wt"] += wm; T["sc"] += sm_bytes
            T["sv"] += sv_bytes; T["sin"] += sin_bytes; T["stor"] += tw
            T["mm"] += mm; T["sd"] += sdf; T["svf"] += svf
            T["hdf"] += hdf; T["aqf"] += aqf; T["tot"] += tot
            T["params"] += p; T["ai"] += ai; T["ao"] += ao

    # H_block shared globally: add once
    T["stor"] += H_SHARED_BYTES

    # ── Block summary ──
    rows.append({k: "" for k in F})
    h = {k: "" for k in F}; h["block"] = "=== BLOCK SUMMARY ==="; rows.append(h)

    for blk in range(NUM_BLOCKS):
        fmt, bs = CASCADE3_CONFIG[blk]
        wb = FORMAT_BITS[fmt]
        sd_name, sb = ("FP16", 16) if fmt == "MXFP8" else ("E8M0", 8)
        bw = bsc = bsv = bsin = 0; bmm = bsd = bsvf = bhdf = baq = 0; bp = 0
        for _, o, i in LAYERS_PER_BLOCK:
            p_ = o * i
            bw  += weight_mem(p_, wb); bsc  += scale_mem(p_, bs, sb // 8)
            bsv += svd_mem(o, i, svd_rank, 2); bsin += i * 2
            bmm += matmul_flops(o, i, ab, wb)
            bsd += scale_dequant_flops(p_, bs, sb)
            bsvf += svd_branch_flops(o, i, svd_rank, 16)
            bhdf += hadamard_rotate_flops(i, HAD_BLOCK_SIZE, 16)
            baq += act_quant_flops(i, ab)
            bp += p_
        bt   = bw + bsc + bsv + bsin
        btot = bmm + bsd + bsvf + bhdf + baq
        rows.append({
            "block": f"Block {blk}", "layer": "10 layers",
            "shape (OxI)": f"{bp:,} params",
            "act_in_dtype": "NVFP4", "act_in_bits": ab, "act_in_bytes/token": "",
            "weight_dtype": fmt, "weight_bits": wb,
            "block_size": bs, "scale_dtype": sd_name, "scale_bits": sb,
            "weight_bytes": int(bw), "scale_bytes": int(bsc),
            "svd_bytes": int(bsv), "sin_bytes": int(bsin),
            "total_weight_bytes": int(bt),
            "act_out_dtype": "FP16", "act_out_bits": 16, "act_out_bytes/token": "",
            "matmul_formula": f"sum 10 layers x{max(ab,wb)/8}", "matmul_factor": max(ab,wb)/8,
            "matmul_flops/token": int(bmm), "scale_dequant_flops/token": int(bsd),
            "svd_flops/token": int(bsvf), "hadamard_flops/token": int(bhdf),
            "act_quant_flops/token": int(baq), "total_flops/token": int(btot),
            "matmul_flops/image": int(bmm*NUM_TOKENS),
            "scale_dequant_flops/image": int(bsd*NUM_TOKENS),
            "svd_flops/image": int(bsvf*NUM_TOKENS),
            "hadamard_flops/image": int(bhdf*NUM_TOKENS),
            "act_quant_flops/image": int(baq*NUM_TOKENS),
            "total_flops/image": int(btot*NUM_TOKENS),
            "eff_bits/weight": round(bt*8/bp, 2),
        })

    # ── Grand summary ──
    rows.append({k: "" for k in F})
    h = {k: "" for k in F}; h["block"] = "=== GRAND SUMMARY ==="; rows.append(h)

    fp16_mb = T["fp16_wt"] / 1024**2
    tot_mb  = T["stor"] / 1024**2
    comp    = T["fp16_wt"] / T["stor"] if T["stor"] else 0
    m8 = sum(1 for f, _ in CASCADE3_CONFIG.values() if f == "MXFP8")
    fp16_flops_pt = sum(2*o*i*2.0 for _, o, i in LAYERS_PER_BLOCK) * NUM_BLOCKS

    for lbl, val in [
        ("Method",             "Cascade-3 + Hadamard (MXFP Weight + NVFP4 Act + SVD)"),
        ("Model",              "PixArt-Sigma (600M params, 28 DiT blocks)"),
        ("Image",              f"1024x1024 → {NUM_TOKENS} tokens"),
        ("", ""),
        ("--- Config ---", ""),
        ("Weight Quant",    f"Cascade-3 per-block: {m8} MXFP8 + {NUM_BLOCKS-m8} MXFP6_E2M3"),
        ("Activation Quant","NVFP4 (4-bit) dynamic, block_size=16"),
        ("Hadamard Block",  f"block_size={HAD_BLOCK_SIZE} (shared H_block={H_SHARED_BYTES} bytes)"),
        ("SVD Rank",        f"{svd_rank} (FP16 lora_a + lora_b)"),
        ("SmoothQuant",     "None"),
        ("Matmul factor",   f"MXFP8+NVFP4→max(8,4)/8=1.0x  MXFP6+NVFP4→max(6,4)/8=0.75x"),
        ("", ""),
        ("--- Storage ---", ""),
        ("Total Parameters",  f"{T['params']:,}"),
        ("FP16 Baseline",     f"{fp16_mb:.2f} MB"),
        ("Weight Storage",    f"{T['wt']/1024**2:.2f} MB"),
        ("Scale Storage",     f"{T['sc']/1024**2:.2f} MB"),
        ("SVD Branch",        f"{T['sv']/1024**2:.2f} MB"),
        ("S_in Buffers",      f"{T['sin']/1024**2:.4f} MB  (per-layer sign vectors)"),
        ("H_block (shared)",  f"{H_SHARED_BYTES/1024:.1f} KB  ({HAD_BLOCK_SIZE}x{HAD_BLOCK_SIZE} FP16, 1 copy global)"),
        ("Total Storage",     f"{tot_mb:.2f} MB"),
        ("Compression",       f"{comp:.2f}x vs FP16"),
        ("Eff Bits/Weight",   f"{T['stor']*8/T['params']:.2f}"),
        ("", ""),
        ("--- Bit-Weighted FLOPs (per token, FP8=1.0x) ---", ""),
        ("Matmul FLOPs/Token",
         f"{T['mm']:,.0f}  ({fmt_f(T['mm'])})  [MXFP8 x1.0 / MXFP6 x0.75]"),
        ("Scale Dequant/Token",
         f"{T['sd']:,.0f}  ({fmt_f(T['sd'])})  [MXFP8 FP16-scale x2.0 / MXFP6 E8M0 x1.0]"),
        ("SVD Branch/Token",
         f"{T['svf']:,.0f}  ({fmt_f(T['svf'])})  [rank={svd_rank} FP16 x2.0]"),
        ("Hadamard Rotate/Token",
         f"{T['hdf']:,.0f}  ({fmt_f(T['hdf'])})  [sign+block_matmul in_f×257 x2.0]"),
        ("Act Quant/Token",
         f"{T['aqf']:,.0f}  ({fmt_f(T['aqf'])})  [quant+dequant NVFP4 x0.5]"),
        ("Total FLOPs/Token",  f"{T['tot']:,.0f}  ({fmt_f(T['tot'])})"),
        ("", ""),
        ("--- Bit-Weighted FLOPs (per image) ---", ""),
        ("Matmul FLOPs/Image",     f"{T['mm']*NUM_TOKENS:,.0f}  ({fmt_f(T['mm']*NUM_TOKENS)})"),
        ("Scale Dequant/Image",    f"{T['sd']*NUM_TOKENS:,.0f}  ({fmt_f(T['sd']*NUM_TOKENS)})"),
        ("SVD Branch/Image",       f"{T['svf']*NUM_TOKENS:,.0f}  ({fmt_f(T['svf']*NUM_TOKENS)})"),
        ("Hadamard Rotate/Image",  f"{T['hdf']*NUM_TOKENS:,.0f}  ({fmt_f(T['hdf']*NUM_TOKENS)})"),
        ("Act Quant/Image",        f"{T['aqf']*NUM_TOKENS:,.0f}  ({fmt_f(T['aqf']*NUM_TOKENS)})"),
        ("Total FLOPs/Image",      f"{T['tot']*NUM_TOKENS:,.0f}  ({fmt_f(T['tot']*NUM_TOKENS)})"),
        ("vs FP16 (all x2.0)",     f"{T['tot']/fp16_flops_pt:.4f}x  (FP16: {fmt_f(fp16_flops_pt*NUM_TOKENS)}/image)"),
        ("", ""),
        ("--- FLOPs Breakdown % ---", ""),
        ("Matmul %",           f"{T['mm']/T['tot']*100:.1f}%"),
        ("Scale Dequant %",    f"{T['sd']/T['tot']*100:.1f}%"),
        ("SVD Branch %",       f"{T['svf']/T['tot']*100:.1f}%"),
        ("Hadamard Rotate %",  f"{T['hdf']/T['tot']*100:.1f}%"),
        ("Act Quant %",        f"{T['aqf']/T['tot']*100:.1f}%"),
        ("", ""),
        ("--- Activation Memory ---", ""),
        ("Act In/Token (NVFP4)",  f"{T['ai']:.0f} bytes ({fmt_b(T['ai'])})"),
        ("Act In/Image",          f"{T['ai']*NUM_TOKENS:.0f} bytes ({fmt_b(T['ai']*NUM_TOKENS)})"),
    ]:
        r = {k: "" for k in F}; r["block"] = lbl; r["layer"] = val; rows.append(r)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=F); w.writeheader(); w.writerows(rows)
    print(f"✅ Saved: {out_path}  ({len(rows)} rows)")
    return T


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = "results/sensitivity"

    C  = generate_cascade3_csv(os.path.join(base, "cascade3_memory_flops.csv"))
    N  = generate_nvfp4_svd_csv(os.path.join(base, "nvfp4_svd_memory_flops.csv"))
    CH = generate_cascade_hadamard_csv(
             os.path.join(base, "cascade_hadamard_memory_flops.csv"))

    fp16_fpt = sum(2*o*i*2.0 for _, o, i in LAYERS_PER_BLOCK) * NUM_BLOCKS

    W = 105
    print("\n" + "=" * W)
    print(f"  COMPARISON  (bit-weighted FLOPs, FP8=1.0x, 1024×1024 image = {NUM_TOKENS} tokens)")
    print("=" * W)
    print(f"{'':40s} {'FP16':>14s} {'Cascade-3':>14s} {'Casc+Hadamard':>14s} {'NVFP4 SVD':>14s}")
    print("-" * W)

    # ── Storage ──
    fp16_mb = C["fp16_wt"] / 1024**2
    print(f"{'--- Storage ---':40s}")
    print(f"{'  Model Storage (MB)':40s}"
          f" {fp16_mb:>13.2f}M"
          f" {C['stor']/1024**2:>13.2f}M"
          f" {CH['stor']/1024**2:>13.2f}M"
          f" {N['stor']/1024**2:>13.2f}M")
    print(f"{'  Compression vs FP16':40s}"
          f" {'1.00x':>14s}"
          f" {C['fp16_wt']/C['stor']:>13.2f}x"
          f" {CH['fp16_wt']/CH['stor']:>13.2f}x"
          f" {N['fp16_wt']/N['stor']:>13.2f}x")
    print(f"{'  Eff Bits/Weight':40s}"
          f" {'16.00':>14s}"
          f" {C['stor']*8/C['params']:>14.2f}"
          f" {CH['stor']*8/CH['params']:>14.2f}"
          f" {N['stor']*8/N['params']:>14.2f}")
    print(f"{'  Activation Dtype':40s}"
          f" {'FP16':>14s} {'FP16':>14s} {'NVFP4 (4b)':>14s} {'NVFP4 (4b)':>14s}")
    print(f"{'  Act Memory/Image':40s}"
          f" {fmt_b(C['ai']*NUM_TOKENS):>14s}"
          f" {fmt_b(C['ai']*NUM_TOKENS):>14s}"
          f" {fmt_b(CH['ai']*NUM_TOKENS):>14s}"
          f" {fmt_b(N['ai']*NUM_TOKENS):>14s}")

    # ── FLOPs/Token breakdown ──
    print()
    print(f"{'--- FLOPs/Token (FP8-normalized) ---':40s}")
    print(f"{'  Matmul':40s}"
          f" {fp16_fpt:>14,.0f}"
          f" {C['mm']:>14,.0f}"
          f" {CH['mm']:>14,.0f}"
          f" {N['mm']:>14,.0f}")
    print(f"{'  Scale Dequant':40s}"
          f" {'0':>14s}"
          f" {C['sd']:>14,.0f}"
          f" {CH['sd']:>14,.0f}"
          f" {N['sd']:>14,.0f}")
    print(f"{'  SVD Branch':40s}"
          f" {'0':>14s}"
          f" {'0':>14s}"
          f" {CH['svf']:>14,.0f}"
          f" {N['svf']:>14,.0f}")
    print(f"{'  Hadamard Rotation':40s}"
          f" {'0':>14s}"
          f" {'0':>14s}"
          f" {CH['hdf']:>14,.0f}"
          f" {'0':>14s}")
    print(f"{'  SmoothQuant':40s}"
          f" {'0':>14s}"
          f" {'0':>14s}"
          f" {'0':>14s}"
          f" {N['smf']:>14,.0f}")
    print(f"{'  Act Quant':40s}"
          f" {'0':>14s}"
          f" {'0':>14s}"
          f" {CH['aqf']:>14,.0f}"
          f" {N['aqf']:>14,.0f}")
    print(f"{'  TOTAL / token':40s}"
          f" {fp16_fpt:>14,.0f}"
          f" {C['tot']:>14,.0f}"
          f" {CH['tot']:>14,.0f}"
          f" {N['tot']:>14,.0f}")

    # ── FLOPs/Image ──
    print()
    print(f"{'--- FLOPs/Image ---':40s}")
    print(f"{'  Total':40s}"
          f" {fmt_f(fp16_fpt*NUM_TOKENS):>14s}"
          f" {fmt_f(C['tot']*NUM_TOKENS):>14s}"
          f" {fmt_f(CH['tot']*NUM_TOKENS):>14s}"
          f" {fmt_f(N['tot']*NUM_TOKENS):>14s}")
    print(f"{'  vs FP16':40s}"
          f" {'1.0000x':>14s}"
          f" {C['tot']/fp16_fpt:>13.4f}x"
          f" {CH['tot']/fp16_fpt:>13.4f}x"
          f" {N['tot']/fp16_fpt:>13.4f}x")
    print(f"{'  vs NVFP4 SVD':40s}"
          f" {fp16_fpt/N['tot']:>13.4f}x"
          f" {C['tot']/N['tot']:>13.4f}x"
          f" {CH['tot']/N['tot']:>13.4f}x"
          f" {'1.0000x':>14s}")

    # ── FLOPs breakdown % ──
    print()
    print(f"{'--- FLOPs Breakdown (Casc+Hadamard) ---':40s}")
    tot_ch = CH["tot"]
    for label, key in [("  Matmul","mm"),("  Scale Dequant","sd"),
                        ("  SVD Branch","svf"),("  Hadamard Rotation","hdf"),
                        ("  Act Quant","aqf")]:
        print(f"{label:40s} {CH[key]/tot_ch*100:>13.1f}%")

    print("=" * W)
