"""
pixart_distribution_analysis.py
================================
PixArt-Alpha DiT 모델의 weight / activation / output 분포를 분석한다.

생성 플롯:
  weight_boxplot.png           — 레이어별 weight 분포 (quantile band)
  activation_boxplot.png       — 레이어별 activation 분포
  output_boxplot.png           — 레이어별 output 분포
  stats_heatmap.png            — abs_max / kurtosis / CV / outlier_ratio 4종 ×3(W/A/O) 히트맵
  outlier_distribution.png     — weight/activation/output outlier ratio 비교
  activation_timestep_heatmap.png  — Layer × Timestep activation abs_max
  output_timestep_heatmap.png      — Layer × Timestep output abs_max
  channel_absmax_<layer>_<w/a/o>.png  — 대표 레이어 채널별 abs_max (timestep × channel)

Usage:
  python pixart_distribution_analysis.py --test_run          # 1 prompt, 5 steps
  python pixart_distribution_analysis.py --num_prompts 8 --num_steps 20
"""

import os
import json
import argparse
import time
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis as scipy_kurtosis
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler

# ==========================================
# Helpers (inlined to avoid heavy imports)
# ==========================================

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def get_module_by_name(model, name):
    for part in name.split("."):
        model = getattr(model, part)
    return model


def get_prompts(num_samples, args):
    if args.dataset_name == "MJHQ":
        path, split, key = "xingjianleng/mjhq30k", "test", "text"
    else:
        path, split, key = "mit-han-lab/svdquant-datasets", "train", "prompt"
    try:
        if load_dataset is None:
            raise ImportError("datasets not available")
        dataset = load_dataset(path, split=split, streaming=True)
        prompts = []
        for i, entry in enumerate(dataset):
            if i >= num_samples:
                break
            prompts.append(entry[key])
        return prompts
    except Exception as e:
        print(f"Warning: Dataset loading failed ({e}). Using fallback prompts.")
        fallback = [
            "A professional high-quality photo of a futuristic city with neon lights",
            "A beautiful landscape of mountains during sunset, cinematic lighting",
            "A cute robot holding a flower in a field, highly detailed digital art",
            "A gourmet burger with melting cheese and fresh vegetables on a wooden table",
            "An astronaut walking on a purple planet surface under a starry sky",
            "A golden retriever running through a meadow, motion blur, DSLR photo",
            "Abstract geometric art with vibrant colors on a dark background",
            "A medieval castle on a cliff overlooking the ocean at dusk",
        ]
        return (fallback * (num_samples // len(fallback) + 1))[:num_samples]


# ==========================================
# Utilities
# ==========================================

def ltype(name: str) -> str:
    if "attn" in name:
        return "attn"
    if "mlp" in name or "ff" in name:
        return "mlp"
    return "other"


def compute_stats(x1d: np.ndarray) -> dict:
    """x1d: 1D float numpy array (subsample OK for speed)"""
    mean = float(np.mean(x1d))
    std = float(np.std(x1d))
    abs_max = float(np.max(np.abs(x1d)))
    # Fast kurtosis (excess) via numpy — avoid scipy overhead
    if std > 1e-10 and len(x1d) >= 4:
        z = (x1d - mean) / std
        kurt = float(np.mean(z ** 4)) - 3.0
    else:
        kurt = 0.0
    outlier_ratio = float(np.mean(np.abs(x1d) > 3 * std))
    cv = std / (abs(mean) + 1e-6)
    return {
        "mean": mean,
        "std": std,
        "abs_max": abs_max,
        "kurtosis": kurt,
        "outlier_ratio": outlier_ratio,
        "cv": cv,
    }


def compute_percentiles(x1d: np.ndarray) -> dict:
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(x1d, qs)
    return {f"q{q}": float(v) for q, v in zip(qs, vals)}


# ==========================================
# Phase 2: Weight analysis
# ==========================================

def analyze_weights(transformer, target_names: list) -> dict:
    print(f"\n[Weight] Analyzing {len(target_names)} layers...")
    weight_stats = {}
    for name in target_names:
        m = get_module_by_name(transformer, name)
        w = m.weight.detach().float().cpu().numpy().flatten()
        stats = compute_stats(w)
        pcts = compute_percentiles(w)
        weight_stats[name] = {
            "shape": list(m.weight.shape),
            "layer_type": ltype(name),
            **stats,
            **pcts,
        }
    print("[Weight] Done.")
    return weight_stats


# ==========================================
# Phase 3: Activation / Output analysis
# ==========================================

N_RAW_SAMPLES = 2000   # random samples collected per layer per forward pass

def analyze_dynamics(pipe, transformer, target_names, prompts,
                     num_steps, rep_layers):
    """
    Returns dyn_stats: dict[name] -> {
      "act": {
        "per_timestep": [{mean, std, abs_max, kurtosis, outlier_ratio, cv}, ...],
        "aggregated": {...},
        "percentiles": {q1, q5, ..., q99},
        "ch_absmax_per_timestep": [[C floats] × T]  # rep_layers only
      },
      "out": {...same...}
    }
    """
    device = next(transformer.parameters()).device

    # Step counter: update via transformer pre-hook (fires before each forward pass)
    current_step = {"val": 0}
    fwd_call_count = [0]

    def transformer_pre_hook(module, inp):
        current_step["val"] = fwd_call_count[0]
        fwd_call_count[0] += 1

    pre_hook_handle = transformer.register_forward_pre_hook(transformer_pre_hook)

    # Accumulators
    # acc[name][which][t] = list of stat dicts (one per prompt)
    acc = {n: {"act": defaultdict(list), "out": defaultdict(list)} for n in target_names}
    # raw_samples[name][which] = list of 1D np arrays (for percentile/boxplot)
    raw_samples = {n: {"act": [], "out": []} for n in target_names}
    # ch_acc[name][which][t] = list of (C,) arrays (rep layers only)
    ch_acc = {n: {"act": defaultdict(list), "out": defaultdict(list)} for n in rep_layers}

    def make_hook(layer_name, which):
        is_rep = layer_name in rep_layers

        def hook(module, inp, out):
            t = current_step["val"]
            tensor = inp[0] if which == "act" else out
            x = tensor.detach().float()
            x1d_torch = x.flatten()

            # Random subsample — all stats computed on subsample for speed
            perm = torch.randperm(x1d_torch.shape[0], device=x1d_torch.device)
            x_sub = x1d_torch[perm[:N_RAW_SAMPLES]].cpu().numpy()
            raw_samples[layer_name][which].append(x_sub)

            # Scalar stats on subsample (fast)
            stats = compute_stats(x_sub)
            # abs_max: use full tensor for accuracy
            stats["abs_max"] = float(x1d_torch.abs().max().item())
            acc[layer_name][which][t].append(stats)

            if is_rep:
                x_flat = x.view(-1, x.shape[-1])
                ch_absmax = x_flat.abs().max(dim=0)[0].cpu().numpy()
                ch_acc[layer_name][which][t].append(ch_absmax)

        return hook

    hooks = []
    for name in target_names:
        m = get_module_by_name(transformer, name)
        hooks.append(m.register_forward_hook(make_hook(name, "act")))
        hooks.append(m.register_forward_hook(make_hook(name, "out")))

    print(f"\n[Dynamics] {len(hooks)} hooks on {len(target_names)} layers.")
    print(f"[Dynamics] Running {len(prompts)} prompts × {num_steps} steps...")

    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}/{len(prompts)}...", flush=True)
        gen = torch.Generator(device=device).manual_seed(42 + i)
        pipe(prompt, num_inference_steps=num_steps, generator=gen)

    for h in hooks:
        h.remove()
    pre_hook_handle.remove()

    print(f"[Dynamics] Total transformer forward calls: {fwd_call_count[0]}")
    print("[Dynamics] Aggregating stats...")

    dyn_stats = {}
    for name in target_names:
        entry = {}
        for which in ("act", "out"):
            # Per-timestep stats (averaged over prompts)
            steps = sorted(acc[name][which].keys())
            per_ts = []
            for t in steps:
                recs = acc[name][which][t]
                keys = recs[0].keys()
                avg = {k: float(np.mean([r[k] for r in recs])) for k in keys}
                per_ts.append(avg)

            # Aggregated over all timesteps
            all_vals_concat = np.concatenate(raw_samples[name][which]) if raw_samples[name][which] else np.array([0.0])
            aggregated = compute_stats(all_vals_concat)
            pcts = compute_percentiles(all_vals_concat)

            sub_entry = {
                "per_timestep": per_ts,
                "aggregated": aggregated,
                "percentiles": pcts,
            }

            # Channel-wise abs_max for representative layers
            if name in ch_acc:
                ch_per_ts = []
                for t in sorted(ch_acc[name][which].keys()):
                    recs_ch = ch_acc[name][which][t]
                    ch_per_ts.append(np.mean(recs_ch, axis=0).tolist())
                sub_entry["ch_absmax_per_timestep"] = ch_per_ts

            entry[which] = sub_entry
        dyn_stats[name] = entry

    return dyn_stats


# ==========================================
# Plotting helpers  (lerobot analyze_distributions.py 패턴)
# ==========================================

import re
from math import ceil

COLS_PER_CHUNK = 16   # block-index columns per file

# PixArt sub-type preferred order (rows in heatmap / subplots in boxplot)
PREFERRED_SUBTYPES = [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    "ff.net.0.proj", "ff.net.2",
]

ATTN_COLOR  = "#3A7FBF"
MLP_COLOR   = "#C0392B"
OTHER_COLOR = "#7F8C8D"

def _subtype_color(st: str) -> str:
    if "attn" in st: return ATTN_COLOR
    if "ff" in st:   return MLP_COLOR
    return OTHER_COLOR

def _parse_layer(name: str):
    """'transformer_blocks.14.attn1.to_q' → (14, 'attn1.to_q')"""
    m = re.match(r'transformer_blocks\.(\d+)\.(.+)', name)
    if m:
        return int(m.group(1)), m.group(2)
    return None, name

def _get_layout(all_names: list):
    """subtypes(행 순서) + block_indices(열 전체) 반환."""
    subtypes_seen, blocks_seen = set(), set()
    for n in all_names:
        bidx, st = _parse_layer(n)
        subtypes_seen.add(st)
        if bidx is not None:
            blocks_seen.add(bidx)
    ordered = [t for t in PREFERRED_SUBTYPES if t in subtypes_seen]
    extra   = sorted(subtypes_seen - set(ordered))
    return ordered + extra, sorted(blocks_seen)

def _chunk(lst, size):
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def _layer_labels_thinned(names, max_labels=40):
    n = len(names)
    step = max(1, n // max_labels)
    ticks = list(range(0, n, step))
    labels = [names[i].replace("transformer_blocks.", "b") for i in ticks]
    return ticks, labels

def _short_name(name):
    return name.replace("transformer_blocks.", "b").replace(".weight", "")

# ──────────────────────────────────────────────────────────────
# 공통: heatmap 패널 렌더 (lerobot _draw_heatmap_panel 동일)
# ──────────────────────────────────────────────────────────────
def _draw_heatmap_panel(ax, mat, row_labels, col_labels, title, cmap):
    flat = mat[~np.isnan(mat)]
    if len(flat):
        vmin = float(np.percentile(flat, 2))
        vmax = float(np.percentile(flat, 98))
        if vmin == vmax:
            vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
    else:
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(mat, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    # NaN → 회색
    nan_mask = np.isnan(mat).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    ax.imshow(nan_mask[:, :, np.newaxis] * np.array([[[0.75, 0.75, 0.75]]]),
              aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha="right")
    ax.set_xlabel("Block index", fontsize=8)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

# ──────────────────────────────────────────────────────────────
# 1. Boxplot: weight / activation / output
#    rows=sub_type subplot,  X=block_idx,  Y=value
# ──────────────────────────────────────────────────────────────
def _bxp_data_weight(name_pct: dict, subtypes, block_chunk):
    """subtype → {block_idx: bxp_stat_dict} from weight percentiles."""
    out = {st: {} for st in subtypes}
    for name, s in name_pct.items():
        bidx, st = _parse_layer(name)
        if st not in out or bidx not in block_chunk:
            continue
        whislo = s.get("q5",  s.get("p5",  0))
        whishi = s.get("q95", s.get("p95", 0))
        out[st][bidx] = {
            "med":    s.get("q50", s.get("p50", 0)),
            "q1":     s.get("q25", s.get("p25", 0)),
            "q3":     s.get("q75", s.get("p75", 0)),
            "whislo": whislo,
            "whishi": whishi,
            "mean":   s.get("mean", 0),
            "fliers": [],
            "_abs_max": s.get("abs_max"),   # red dot
            "_q99":     s.get("q99"),        # yellow dot (upper tail)
            "_q1":      s.get("q1"),         # yellow dot (lower tail)
        }
    return out


def _bxp_data_dyn(name_dyn: dict, subtypes, block_chunk, which):
    """subtype → {block_idx: bxp_stat_dict} from dynamics per_timestep."""
    out = {st: {} for st in subtypes}
    for name, entry in name_dyn.items():
        bidx, st = _parse_layer(name)
        if st not in out or bidx not in block_chunk:
            continue
        pts = entry.get(which, {}).get("per_timestep", [])
        pcts = entry.get(which, {}).get("percentiles", {})
        agg  = entry.get(which, {}).get("aggregated", {})
        if not pts and not pcts:
            continue
        # Use percentiles if available, else approximate from per_timestep
        if pcts:
            whislo = pcts.get("q5",  0)
            whishi = pcts.get("q95", 0)
            bxp = {
                "med":    pcts.get("q50", agg.get("mean", 0)),
                "q1":     pcts.get("q25", 0),
                "q3":     pcts.get("q75", 0),
                "whislo": whislo,
                "whishi": whishi,
                "mean":   agg.get("mean", 0),
                "fliers": [],
                "_abs_max": agg.get("abs_max"),  # red dot
                "_q99":     pcts.get("q99"),      # yellow dot (upper tail)
                "_q1":      pcts.get("q1"),       # yellow dot (lower tail)
            }
        else:
            vals = [s.get("abs_max", 0) for s in pts]
            whislo = float(np.percentile(vals, 5))
            whishi = float(np.percentile(vals, 95))
            bxp = {
                "med":    float(np.median(vals)),
                "q1":     float(np.percentile(vals, 25)),
                "q3":     float(np.percentile(vals, 75)),
                "whislo": whislo,
                "whishi": whishi,
                "mean":   float(np.mean(vals)),
                "fliers": [],
                "_abs_max": float(np.max(vals)),
                "_q99":     float(np.percentile(vals, 99)),
                "_q1":      float(np.percentile(vals, 1)),
            }
        out[st][bidx] = bxp
    return out


def _plot_boxplot_chunk(data_by_st, subtypes, block_chunk, title, plot_dir, fname):
    active = [st for st in subtypes if any(b in data_by_st[st] for b in block_chunk)]
    if not active:
        return

    n_cols_fig = min(3, len(active))
    n_rows_fig = ceil(len(active) / n_cols_fig)
    fig, axes = plt.subplots(n_rows_fig, n_cols_fig,
                             figsize=(6 * n_cols_fig, 4 * n_rows_fig),
                             squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for i, st in enumerate(active):
        ax = axes_flat[i]
        col = _subtype_color(st)
        blocks_here = sorted(b for b in block_chunk if b in data_by_st[st])
        if not blocks_here:
            ax.set_visible(False)
            continue
        box_stats = [data_by_st[st][b] for b in blocks_here]
        bxp = ax.bxp(box_stats, positions=range(len(blocks_here)), widths=0.6,
                     patch_artist=True, showmeans=True, showfliers=False,
                     meanprops=dict(marker="D", markersize=4,
                                    markerfacecolor="white",
                                    markeredgecolor="black", markeredgewidth=0.8))
        for patch in bxp["boxes"]:
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
        for elem in ["whiskers", "caps", "medians"]:
            for line in bxp[elem]:
                line.set_color(col)
        # manually scatter outlier dots: q99/q1=yellow, abs_max=red
        for pos_i, b in enumerate(blocks_here):
            s = data_by_st[st][b]
            whishi = s["whishi"]
            whislo = s["whislo"]
            q99 = s.get("_q99")
            q1  = s.get("_q1")
            absmax = s.get("_abs_max")
            if q99 is not None and q99 > whishi:
                ax.plot(pos_i, q99, "o", color="gold", markersize=5,
                        markeredgecolor="darkorange", markeredgewidth=0.7,
                        alpha=0.85, zorder=4)
            if q1 is not None and q1 < whislo:
                ax.plot(pos_i, q1, "o", color="gold", markersize=5,
                        markeredgecolor="darkorange", markeredgewidth=0.7,
                        alpha=0.85, zorder=4)
            if absmax is not None and absmax > whishi:
                ax.plot(pos_i, absmax, "o", color="red", markersize=5,
                        markeredgecolor="darkred", markeredgewidth=0.7,
                        alpha=0.85, zorder=5)
            if absmax is not None and -absmax < whislo:
                ax.plot(pos_i, -absmax, "o", color="red", markersize=5,
                        markeredgecolor="darkred", markeredgewidth=0.7,
                        alpha=0.85, zorder=5)
        ax.set_title(st, fontsize=9, fontweight="bold")
        ax.set_xlabel("Block index", fontsize=8)
        ax.set_ylabel("Value  (box:p25–p75 / whisker:p5–p95)", fontsize=8)
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0],[0], marker="o", color="w", markerfacecolor="red",
                   markersize=6, label="abs_max"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="gold",
                   markeredgecolor="darkorange", markersize=6, label="q99 / q1"),
        ], fontsize=7, loc="upper right", framealpha=0.6)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(len(blocks_here)))
        ax.set_xticklabels([str(b) for b in blocks_here], fontsize=7)

    for j in range(len(active), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(plot_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_quantile_band(names, label, stats_fn, plot_dir, prefix):
    """Entry point — routes to grid-based boxplot."""
    subtypes, all_blocks = _get_layout(names)
    chunks = _chunk(all_blocks, COLS_PER_CHUNK)

    # Build a unified dict for _bxp_data_weight style
    name_pct = {n: stats_fn(n) for n in names}
    for part, chunk in enumerate(chunks, 1):
        block_set = set(chunk)
        data_by_st = _bxp_data_weight(name_pct, subtypes, block_set)
        part_str = f"blocks {chunk[0]}–{chunk[-1]}"
        _plot_boxplot_chunk(
            data_by_st, subtypes, block_set,
            f"{label} Distribution  [{part_str}]\n(box: p25–p75 / whisker: p5–p95 / ◆=mean)",
            plot_dir,
            f"{prefix}_part{part}.png"
        )


# ──────────────────────────────────────────────────────────────
# 2. Stats heatmap: rows=subtype, cols=block_idx, cell=metric value
# ──────────────────────────────────────────────────────────────
def plot_stats_heatmap(names, weight_stats, dyn_stats, plot_dir):
    """One file per metric × chunk: stats_heatmap_{metric}_part{n}.png
    Each file: 1 row × 3 source (Weight / Activation / Output) panels."""
    subtypes, all_blocks = _get_layout(names)
    chunks = _chunk(all_blocks, COLS_PER_CHUNK)

    sources = [
        ("Weight",     weight_stats, None,  "YlOrRd"),
        ("Activation", dyn_stats,    "act", "Blues"),
        ("Output",     dyn_stats,    "out", "Purples"),
    ]
    metrics = [
        ("abs_max",      "abs_max",          "YlOrRd"),
        ("kurtosis",     "kurtosis (excess)", "RdBu_r"),
        ("cv",           "CV (σ/|μ|)",        "PuOr"),
        ("outlier_ratio","outlier_ratio\n(|x|>3σ)", "Reds"),
    ]

    for part, chunk in enumerate(chunks, 1):
        block_list = list(chunk)
        block_set  = set(chunk)
        col_labels = [str(b) for b in block_list]
        part_str   = f"blocks {chunk[0]}–{chunk[-1]}"
        n_cols_fig = len(block_list)
        n_rows_fig = len(subtypes)
        fw = max(14, n_cols_fig * 0.9 + 4)
        fh = n_rows_fig * 0.55 + 2.5

        for metric_key, metric_label, metric_cmap in metrics:
            fig, axes = plt.subplots(1, len(sources),
                                     figsize=(fw * len(sources) / 2.5, fh),
                                     squeeze=False)
            fig.suptitle(f"Stats Heatmap — {metric_label}  [{part_str}]\n"
                         f"(rows=sub_type, cols=block_idx)",
                         fontsize=11, fontweight="bold")

            for col_i, (src_label, src_dict, which, _src_cmap) in enumerate(sources):
                ax = axes[0][col_i]
                mat = np.full((len(subtypes), len(block_list)), np.nan)
                for name in names:
                    bidx, st = _parse_layer(name)
                    if st not in subtypes or bidx not in block_set:
                        continue
                    r = subtypes.index(st)
                    c = block_list.index(bidx)
                    if which is None:
                        v = src_dict.get(name, {}).get(metric_key)
                    else:
                        v = src_dict.get(name, {}).get(which, {}).get("aggregated", {}).get(metric_key)
                    if v is not None:
                        mat[r, c] = float(v)

                _draw_heatmap_panel(ax, mat, subtypes, col_labels,
                                    src_label, metric_cmap)

            plt.tight_layout()
            safe_metric = metric_key.replace("/", "_")
            path = os.path.join(plot_dir, f"stats_heatmap_{safe_metric}_part{part}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
# 3. Outlier distribution heatmap (W/A/O in one figure)
# ──────────────────────────────────────────────────────────────
def plot_outlier_distribution(names, weight_stats, dyn_stats, plot_dir):
    subtypes, all_blocks = _get_layout(names)
    chunks = _chunk(all_blocks, COLS_PER_CHUNK)

    for part, chunk in enumerate(chunks, 1):
        block_list = list(chunk)
        block_set  = set(chunk)
        col_labels = [str(b) for b in block_list]
        part_str   = f"blocks {chunk[0]}–{chunk[-1]}"

        fw = max(14, len(block_list) * 0.9 + 4)
        fh = len(subtypes) * 0.55 + 2.5
        fig, axes = plt.subplots(1, 3, figsize=(fw * 3 / 2.5, fh), sharey=True)
        fig.suptitle(f"Outlier Ratio  (|x|>3σ)  [{part_str}]", fontsize=11, fontweight="bold")

        srcs = [
            ("Weight",     weight_stats, None),
            ("Activation", dyn_stats,    "act"),
            ("Output",     dyn_stats,    "out"),
        ]
        for ax, (src_label, src_dict, which) in zip(axes, srcs):
            mat = np.full((len(subtypes), len(block_list)), np.nan)
            for name in names:
                bidx, st = _parse_layer(name)
                if st not in subtypes or bidx not in block_set:
                    continue
                r = subtypes.index(st)
                c = block_list.index(bidx)
                if which is None:
                    v = src_dict.get(name, {}).get("outlier_ratio")
                else:
                    v = src_dict.get(name, {}).get(which, {}).get("aggregated", {}).get("outlier_ratio")
                if v is not None:
                    mat[r, c] = float(v)

            _draw_heatmap_panel(ax, mat, subtypes, col_labels, src_label, "Reds")

        plt.tight_layout()
        path = os.path.join(plot_dir, f"outlier_heatmap_part{part}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# --------------------------------------------------
# 4. Timestep heatmap (activation / output abs_max)
# --------------------------------------------------

def plot_timestep_heatmap(names, dyn_stats, which, title, fname_prefix, plot_dir, metric="abs_max"):
    """One file per sub_type: Y=block_idx, X=timestep_idx.
    Shows how each block evolves over the denoising process."""
    num_steps = 0
    for n in names:
        pts = dyn_stats.get(n, {}).get(which, {}).get("per_timestep", [])
        if pts:
            num_steps = len(pts)
            break
    if num_steps == 0:
        print(f"  [Skip] No per_timestep data for {which}")
        return

    subtypes, all_blocks = _get_layout(names)
    t_labels = [f"t{t}" for t in range(num_steps)]
    fh = max(5, len(all_blocks) * 0.4 + 2)
    fw = max(8, num_steps * 0.7 + 3)

    for st in subtypes:
        mat = np.full((len(all_blocks), num_steps), np.nan)
        for name in names:
            bidx, subtype = _parse_layer(name)
            if subtype != st or bidx not in all_blocks:
                continue
            pts = dyn_stats.get(name, {}).get(which, {}).get("per_timestep", [])
            r = all_blocks.index(bidx)
            for t, s in enumerate(pts):
                mat[r, t] = s.get(metric, np.nan)

        if np.all(np.isnan(mat)):
            continue

        fig, ax = plt.subplots(figsize=(fw, fh))
        row_labels = [str(b) for b in all_blocks]
        _draw_heatmap_panel(ax, mat, row_labels, t_labels,
                            f"{title} — {st}  [{metric}]", "inferno")
        ax.set_xlabel("Denoising timestep", fontsize=9)
        ax.set_ylabel("Block index", fontsize=9)
        plt.tight_layout()
        safe_st = st.replace(".", "_")
        path = os.path.join(plot_dir, f"{fname_prefix}_{safe_st}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# --------------------------------------------------
# 5. Channel abs_max heatmap (representative layers)
# --------------------------------------------------

def plot_channel_absmax(names, dyn_stats, rep_layers, plot_dir):
    for name in rep_layers:
        if name not in dyn_stats:
            continue
        for which in ("act", "out"):
            ch_data = dyn_stats[name].get(which, {}).get("ch_absmax_per_timestep")
            if not ch_data:
                continue
            mat = np.array(ch_data).T   # (C, T)
            fig, ax = plt.subplots(figsize=(max(6, mat.shape[1] * 0.6), 5))
            im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")
            ax.set_xlabel("Timestep index", fontsize=9)
            ax.set_ylabel("Channel index", fontsize=9)
            ax.set_title(f"Channel-wise abs_max\n{name}  [{which}]", fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8, label="abs_max")
            plt.tight_layout()
            safe = name.replace(".", "_").replace("/", "_")
            path = os.path.join(plot_dir, f"channel_absmax_{safe}_{which}.png")
            plt.savefig(path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {path}")


# ==========================================
# Shared plot driver
# ==========================================

def _generate_plots(target_names, weight_stats, dyn_stats, rep_layers, plot_dir):
    print("\n[Phase 5] Generating plots...")

    def weight_stats_fn(name):
        return weight_stats.get(name, {})

    def act_stats_fn(name):
        s = dyn_stats.get(name, {}).get("act", {})
        return {**s.get("aggregated", {}), **s.get("percentiles", {})}

    def out_stats_fn(name):
        s = dyn_stats.get(name, {}).get("out", {})
        return {**s.get("aggregated", {}), **s.get("percentiles", {})}

    plot_quantile_band(target_names, "Weight",     weight_stats_fn, plot_dir, "weight_boxplot")
    plot_quantile_band(target_names, "Activation", act_stats_fn,    plot_dir, "activation_boxplot")
    plot_quantile_band(target_names, "Output",     out_stats_fn,    plot_dir, "output_boxplot")
    plot_stats_heatmap(target_names, weight_stats, dyn_stats, plot_dir)
    plot_outlier_distribution(target_names, weight_stats, dyn_stats, plot_dir)
    plot_timestep_heatmap(target_names, dyn_stats, "act",
                          "Activation abs_max: Layer × Timestep",
                          "activation_timestep_heatmap", plot_dir)
    plot_timestep_heatmap(target_names, dyn_stats, "out",
                          "Output abs_max: Layer × Timestep",
                          "output_timestep_heatmap", plot_dir)
    plot_channel_absmax(target_names, dyn_stats, rep_layers, plot_dir)


# ==========================================
# Main
# ==========================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str,
                   default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    p.add_argument("--num_prompts", type=int, default=4)
    p.add_argument("--num_steps",   type=int, default=20)
    p.add_argument("--output_dir",  type=str, default="results/distribution_analysis")
    p.add_argument("--dataset_name", type=str, default="MJHQ", choices=["MJHQ", "parti"])
    p.add_argument("--test_run", action="store_true",
                   help="1 prompt, 5 steps smoke test")
    p.add_argument("--plot_only", action="store_true",
                   help="Skip data collection — reload stats.json and regenerate all plots")
    return p.parse_args()


def main():
    args = parse_args()

    plot_dir  = os.path.join(args.output_dir, "plots")
    stats_path = os.path.join(args.output_dir, "stats.json")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ── plot_only: reload JSON and re-draw ──────────────────────────────────
    if args.plot_only:
        print(f"[plot_only] Loading {stats_path} ...")
        with open(stats_path) as f:
            saved = json.load(f)
        target_names = list(saved["weight"].keys())
        weight_stats = saved["weight"]
        dyn_stats    = saved["dynamics"]
        attn_l = [n for n in target_names if ltype(n) == "attn"]
        mlp_l  = [n for n in target_names if ltype(n) == "mlp"]
        rep_layers = []
        for lst in [attn_l, mlp_l]:
            if lst:
                rep_layers += [lst[0], lst[len(lst)//2], lst[-1]]
        rep_layers = list(dict.fromkeys(rep_layers))
        _generate_plots(target_names, weight_stats, dyn_stats, rep_layers, plot_dir)
        print(f"\n[Done] Plots saved to {plot_dir}/")
        for p in sorted(os.listdir(plot_dir)):
            print(f"  {p}")
        return
    # ────────────────────────────────────────────────────────────────────────

    if args.test_run:
        args.num_prompts = 1
        args.num_steps   = 5
        print("[TEST MODE] num_prompts=1, num_steps=5")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # ─────────────────────────────────────────
    # Phase 1: Load model
    # ─────────────────────────────────────────
    print("\n[Phase 1] Loading FP16 model...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer
    transformer.eval()

    skip_kw = ["x_embedder", "t_embedder", "proj_out"]
    target_names = [
        n for n, m in transformer.named_modules()
        if isinstance(m, nn.Linear) and not any(kw in n for kw in skip_kw)
    ]
    print(f"  Found {len(target_names)} target Linear layers.")

    # Representative layers (first / mid / last of attn & mlp)
    attn_layers = [n for n in target_names if ltype(n) == "attn"]
    mlp_layers  = [n for n in target_names if ltype(n) == "mlp"]
    rep_layers = []
    for lst in [attn_layers, mlp_layers]:
        if lst:
            rep_layers += [lst[0], lst[len(lst)//2], lst[-1]]
    rep_layers = list(dict.fromkeys(rep_layers))
    print(f"  Rep layers: {rep_layers}")

    prompts = get_prompts(args.num_prompts, args)[:args.num_prompts]

    # ─────────────────────────────────────────
    # Phase 2: Weight analysis
    # ─────────────────────────────────────────
    print("\n[Phase 2] Weight distribution...")
    weight_stats = analyze_weights(transformer, target_names)

    # ─────────────────────────────────────────
    # Phase 3: Activation / Output analysis
    # ─────────────────────────────────────────
    print("\n[Phase 3] Activation / Output distribution...")
    dyn_stats = analyze_dynamics(
        pipe, transformer, target_names, prompts, args.num_steps, rep_layers
    )

    # ─────────────────────────────────────────
    # Phase 4: Save JSON
    # ─────────────────────────────────────────
    print("\n[Phase 4] Saving stats JSON...")
    out = {
        "config": {
            "model_path": args.model_path,
            "num_prompts": args.num_prompts,
            "num_steps": args.num_steps,
            "dataset_name": args.dataset_name,
            "num_layers": len(target_names),
        },
        "weight": weight_stats,
        "dynamics": {
            n: {
                "act": {k: v for k, v in e["act"].items() if k != "ch_absmax_per_timestep"},
                "out": {k: v for k, v in e["out"].items() if k != "ch_absmax_per_timestep"},
            }
            for n, e in dyn_stats.items()
        },
    }
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {stats_path}")

    # ─────────────────────────────────────────
    # Phase 5: Plots
    # ─────────────────────────────────────────
    _generate_plots(target_names, weight_stats, dyn_stats, rep_layers, plot_dir)

    elapsed = time.time() - t0
    print(f"\n[Done] Total time: {elapsed/60:.1f}m")
    print(f"  Stats : {stats_path}")
    print(f"  Plots : {plot_dir}/")
    for p in sorted(os.listdir(plot_dir)):
        print(f"    {p}")


if __name__ == "__main__":
    main()
