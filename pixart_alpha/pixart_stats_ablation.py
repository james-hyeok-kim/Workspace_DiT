#!/usr/bin/env python3
"""
pixart_stats_ablation.py
========================
Ablation study: which weight-stat features best predict MXFP6 vs MXFP8 optimal format?

Uses *existing* outputs only (no model reload):
  - results/sensitivity/MJHQ/weight_stats_analysis.csv   (280 rows, 28 blocks × 10 layers)
  - results/sensitivity/MJHQ/activation_stats.csv        (280 rows, 28 blocks × 10 layers)
  - results/sensitivity/MJHQ/phase1_best_per_block.json  (28 labels: MXFP6_E2M3 or MXFP8)
  - results/sensitivity/MJHQ/phase1/b*_*.json            (per-block FID by format)

Ablation levels
  1.  Single-feature threshold sweep  (all features, both ± directions)
  2.  2-feature AND combinations
  2b. 2-feature OR  combinations
  2c. 2-feature Cascade (if f1>=T1: 8bit; elif f2>=T2: 8bit; else 6bit)
  3.  Depth-2 decision tree (sklearn, LOO-CV)
  3b. 3-feature greedy Cascade (fix best cascade-2, sweep f3)

Run:
  python pixart_stats_ablation.py
  python pixart_stats_ablation.py --save_dir results/sensitivity --dataset MJHQ
"""

import argparse
import json
import os
import csv
import math
import itertools
from collections import defaultdict

# ── optional sklearn ────────────────────────────────────────────────────────
try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    SKLEARN = True
except ImportError:
    SKLEARN = False
    print("⚠️  scikit-learn not found → skipping DT / logistic regression")


# ── helpers ─────────────────────────────────────────────────────────────────

_FAMILY = {
    "MXFP6_E2M3": "6bit", "MXFP6_E3M2": "6bit",
    "MXFP8":      "8bit", "NVFP8":       "8bit",
    "MXFP4":      "4bit", "NVFP4":       "4bit",
    "FP3_E1M1":   "3bit",
}

def load_stats_csv_v2(path: str) -> dict[int, list[dict]]:
    """Like load_stats_csv but keeps 'layer' name in each row."""
    blocks: dict[int, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            blk = int(row["block_idx"])
            blocks[blk].append({
                "layer":      row["layer"],
                "imbalance":  float(row["imbalance"]),
                "snr_MXFP4":  float(row["snr_MXFP4"]),
                "snr_MXFP6":  float(row["snr_MXFP6"]),
                "snr_MXFP8":  float(row["snr_MXFP8"]),
                "gap_4bit":   float(row["gap_4bit"]),
                "gap_6bit":   float(row["gap_6bit"]),
            })
    return dict(blocks)


def load_act_csv(path: str) -> dict[int, list[dict]]:
    """Return {block_idx: [{'layer':..., 'act_gap_6bit':..., 'act_imbalance':...}, ...]}."""
    if not os.path.exists(path):
        return {}
    blocks: dict[int, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            blk = int(row["block_idx"])
            d = {"layer": row["layer"],
                 "act_gap_6bit":   float(row["act_gap_6bit"]),
                 "act_imbalance":  float(row["act_imbalance"])}
            # optional extra columns (if Phase B hook features were added)
            for col in ("act_kurtosis", "act_dynamic_range", "act_outlier_frac"):
                if col in row:
                    d[col] = float(row[col])
            blocks[blk].append(d)
    return dict(blocks)


def classify_layer(layer_name: str) -> str:
    """Classify a layer name into a functional group.

    Groups:
      attn1_qkv  – self-attn to_q/to_k/to_v
      attn1_out  – self-attn to_out.0
      attn2_q    – cross-attn to_q  (varies per block)
      attn2_kv   – cross-attn to_k / to_v  (constant: processes fixed text encoder)
      attn2_out  – cross-attn to_out.0
      ff         – feedforward net.0.proj / net.2
    """
    if "attn1" in layer_name:
        if "to_out" in layer_name:
            return "attn1_out"
        return "attn1_qkv"
    if "attn2" in layer_name:
        if "to_out" in layer_name:
            return "attn2_out"
        if "to_k" in layer_name or "to_v" in layer_name:
            return "attn2_kv"
        return "attn2_q"
    # ff.net.*
    return "ff"


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2


def _percentile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    idx = (len(s) - 1) * p
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def load_labels(base_dir: str) -> dict[int, str]:
    """Return {block_idx: empirical_best_format}."""
    path = os.path.join(base_dir, "phase1_best_per_block.json")
    with open(path) as f:
        data = json.load(f)
    out = {}
    for key, val in data.items():
        idx = int(key.replace("block_", ""))
        fmt = val["format"] if isinstance(val, dict) else val
        out[idx] = fmt
    return out


def load_fid_by_block(base_dir: str) -> dict[int, dict[str, float]]:
    """Return {block_idx: {fmt_key: fid}} from phase1/*.json files."""
    phase1_dir = os.path.join(base_dir, "phase1")
    result: dict[int, dict[str, float]] = defaultdict(dict)
    for fname in sorted(os.listdir(phase1_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(phase1_dir, fname)) as f:
            d = json.load(f)
        parts = fname.replace(".json", "").split("_")
        blk = int(parts[0][1:])
        key = fname.replace(".json", "").split("_", 1)[1]  # e.g. MXFP6_E2M3_bs32_FP16
        fid = d.get("FID") or d.get("fid") or (d.get("metrics", {}).get("FID"))
        if fid is not None:
            result[blk][key] = float(fid)
    return dict(result)


def load_stats_csv(path: str) -> dict[int, list[dict]]:
    """Return {block_idx: [row_dict, ...]} from weight_stats_analysis.csv."""
    blocks: dict[int, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            blk = int(row["block_idx"])
            blocks[blk].append({
                "imbalance": float(row["imbalance"]),
                "snr_MXFP4":  float(row["snr_MXFP4"]),
                "snr_MXFP6":  float(row["snr_MXFP6"]),
                "snr_MXFP8":  float(row["snr_MXFP8"]),
                "gap_4bit":   float(row["gap_4bit"]),
                "gap_6bit":   float(row["gap_6bit"]),
            })
    return dict(blocks)


def block_features(layers: list[dict], block_idx: int) -> dict[str, float]:
    """Aggregate layer-level stats into per-block features."""
    gaps6   = [r["gap_6bit"]  for r in layers]
    gaps4   = [r["gap_4bit"]  for r in layers]
    imbs    = [r["imbalance"] for r in layers]
    snr8    = [r["snr_MXFP8"] for r in layers]
    snr6    = [r["snr_MXFP6"] for r in layers]

    n = len(layers)
    mean6   = sum(gaps6) / n
    max6    = max(gaps6)
    min6    = min(gaps6)
    std6    = math.sqrt(sum((x - mean6) ** 2 for x in gaps6) / n)

    mean4   = sum(gaps4) / n
    max4    = max(gaps4)

    mean_imb = sum(imbs) / n
    max_imb  = max(imbs)

    mean_snr8 = sum(snr8) / n
    mean_snr6 = sum(snr6) / n

    slope_mean = mean6 / (mean4 + 1e-8)   # gap6/gap4  (small = uniform degradation)
    slope_max  = max6  / (max4  + 1e-8)

    # relative range within block
    range6     = max6 - min6
    cv6        = std6 / (mean6 + 1e-8)    # coeff of variation

    return {
        "mean_gap6":    mean6,
        "max_gap6":     max6,
        "min_gap6":     min6,
        "std_gap6":     std6,
        "cv_gap6":      cv6,
        "range6":       range6,
        "mean_gap4":    mean4,
        "slope_mean":   slope_mean,   # mean_gap6 / mean_gap4
        "slope_max":    slope_max,    # max_gap6  / max_gap4
        "mean_imb":     mean_imb,
        "max_imb":      max_imb,
        "mean_snr8":    mean_snr8,    # absolute SNR (higher = weight is quantization-friendly)
        "mean_snr6":    mean_snr6,
        "block_idx":    float(block_idx),
        # derived
        "gap6_x_imb":   mean6 * max_imb,   # interaction term
        "max6_x_imb":   max6  * max_imb,
    }


def block_features_v2(w_layers: list[dict], a_layers: list[dict],
                      block_idx: int) -> dict[str, float]:
    """Extended per-block features with per-layer-type breakdowns.

    w_layers: rows from load_stats_csv_v2() (include 'layer' key)
    a_layers: rows from load_act_csv()       (include 'layer' key); may be []
    """
    # ── Classify each weight layer ─────────────────────────────────────────
    typed_w: dict[str, list[dict]] = defaultdict(list)
    for r in w_layers:
        typed_w[classify_layer(r["layer"])].append(r)

    # ── Classify each activation layer ────────────────────────────────────
    typed_a: dict[str, list[dict]] = defaultdict(list)
    for r in a_layers:
        typed_a[classify_layer(r["layer"])].append(r)

    # Helper: safe mean of a field across rows
    def wmean(rows, field):
        vals = [r[field] for r in rows if field in r]
        return sum(vals) / len(vals) if vals else 0.0

    def wmax(rows, field):
        vals = [r[field] for r in rows if field in r]
        return max(vals) if vals else 0.0

    def wmin(rows, field):
        vals = [r[field] for r in rows if field in r]
        return min(vals) if vals else 0.0

    # ── Weight gap6 per type ───────────────────────────────────────────────
    w_all_gap6 = [r["gap_6bit"] for r in w_layers]

    # "Sensitive" layers = all except attn2_kv (which are constant ~0.1)
    sensitive_w = [r for r in w_layers if classify_layer(r["layer"]) != "attn2_kv"]
    sens_gap6   = [r["gap_6bit"] for r in sensitive_w]

    feats: dict[str, float] = {}

    # --- Weight: per-type mean gap6 ----------------------------------------
    feats["w_attn1_mean_gap6"]  = wmean(typed_w["attn1_qkv"] + typed_w["attn1_out"], "gap_6bit")
    feats["w_attn2q_gap6"]      = wmean(typed_w["attn2_q"],   "gap_6bit")
    feats["w_attn2kv_gap6"]     = wmean(typed_w["attn2_kv"],  "gap_6bit")  # ~constant
    feats["w_attn2out_gap6"]    = wmean(typed_w["attn2_out"], "gap_6bit")
    feats["w_ff_mean_gap6"]     = wmean(typed_w["ff"],        "gap_6bit")
    feats["w_ff_max_gap6"]      = wmax (typed_w["ff"],        "gap_6bit")
    # ff.net.2 specifically
    ff2 = [r for r in typed_w["ff"] if "net.2" in r["layer"]]
    feats["w_ff2_gap6"]         = wmean(ff2, "gap_6bit")

    # --- Weight: robust aggregates (all layers) ----------------------------
    feats["w_median_gap6"]      = _median(w_all_gap6)
    feats["w_gap6_p75"]         = _percentile(w_all_gap6, 0.75)
    feats["w_gap6_p90"]         = _percentile(w_all_gap6, 0.90)

    # --- Weight: sensitive-layer aggregates (excl. attn2_kv) ---------------
    feats["w_sensitive_mean6"]  = sum(sens_gap6) / len(sens_gap6) if sens_gap6 else 0.0
    feats["w_sensitive_max6"]   = max(sens_gap6) if sens_gap6 else 0.0
    feats["w_sensitive_p75"]    = _percentile(sens_gap6, 0.75) if sens_gap6 else 0.0

    # --- Weight: imbalance per type ----------------------------------------
    feats["w_attn2q_imb"]       = wmean(typed_w["attn2_q"],  "imbalance")
    feats["w_ff_max_imb"]       = wmax (typed_w["ff"],       "imbalance")
    feats["w_sensitive_max_imb"]= wmax (sensitive_w,         "imbalance")

    # ── Activation features (empty if no act CSV) ─────────────────────────
    if a_layers:
        a_all_gap6 = [r["act_gap_6bit"] for r in a_layers]
        sensitive_a = [r for r in a_layers if classify_layer(r["layer"]) != "attn2_kv"]

        feats["a_attn2out_gap6"]  = wmean(typed_a["attn2_out"], "act_gap_6bit")
        feats["a_attn2q_gap6"]    = wmean(typed_a["attn2_q"],   "act_gap_6bit")
        feats["a_attn1_mean_gap6"]= wmean(typed_a["attn1_qkv"] + typed_a["attn1_out"], "act_gap_6bit")
        feats["a_ff_mean_gap6"]   = wmean(typed_a["ff"],         "act_gap_6bit")
        ff2_a = [r for r in typed_a["ff"] if "net.2" in r["layer"]]
        feats["a_ff2_gap6"]       = wmean(ff2_a, "act_gap_6bit")
        feats["a_max_gap6"]       = max(a_all_gap6)
        feats["a_sensitive_mean6"]= (sum(r["act_gap_6bit"] for r in sensitive_a) / len(sensitive_a)
                                     if sensitive_a else 0.0)

        feats["a_attn2out_imb"]   = wmean(typed_a["attn2_out"], "act_imbalance")
        feats["a_attn2q_imb"]     = wmean(typed_a["attn2_q"],   "act_imbalance")
        feats["a_ff_mean_imb"]    = wmean(typed_a["ff"],         "act_imbalance")
        feats["a_sensitive_max_imb"] = wmax(sensitive_a,         "act_imbalance")

        # Optional Phase B hook features
        if "act_kurtosis" in a_layers[0]:
            feats["a_mean_kurtosis"]  = sum(r["act_kurtosis"]    for r in a_layers) / len(a_layers)
            feats["a_max_dyn_range"]  = max(r["act_dynamic_range"] for r in a_layers)
            feats["a_mean_outlier"]   = sum(r["act_outlier_frac"] for r in a_layers) / len(a_layers)

    # ── Block index ────────────────────────────────────────────────────────
    feats["block_idx"] = float(block_idx)

    return feats


# ── single-feature classifier ────────────────────────────────────────────────

def sweep_single(feat_vals: list[float], labels_bin: list[int],
                 fid_deltas: list[float]) -> dict:
    """
    Sweep all thresholds for a single feature.
    labels_bin: 1 = MXFP8 optimal, 0 = MXFP6 optimal
    Returns best threshold, direction, in-sample accuracy, mean_fid_delta.
    Direction: 'high→8bit' means predict MXFP8 if feat >= T.
    """
    n = len(feat_vals)
    results = []
    for direction in ["high→8bit", "low→8bit"]:
        # For direction='high→8bit': predict 8bit if feat >= T, else 6bit
        # For direction='low→8bit':  predict 8bit if feat <= T, else 6bit
        sorted_vals = sorted(set(feat_vals))
        # add boundary candidates
        candidates = []
        for v in sorted_vals:
            candidates.append(v - 1e-6)
            candidates.append(v + 1e-6)
        for T in candidates:
            preds = []
            for v in feat_vals:
                if direction == "high→8bit":
                    preds.append(1 if v >= T else 0)
                else:
                    preds.append(1 if v <= T else 0)
            correct = sum(p == l for p, l in zip(preds, labels_bin))
            total_fid = sum(d for p, l, d in zip(preds, labels_bin, fid_deltas) if p != l)
            mean_fid = total_fid / n
            results.append((correct, -mean_fid, T, direction))

    results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = results[0]
    correct, neg_fid, T, direction = best
    return {
        "acc": correct / n,
        "correct": correct,
        "threshold": T,
        "direction": direction,
        "mean_fid_delta": -neg_fid,
    }


def loo_single(feat_vals: list[float], labels_bin: list[int]) -> float:
    """Leave-one-out CV accuracy for single-feature threshold classifier."""
    n = len(feat_vals)
    correct = 0
    for i in range(n):
        # train on n-1 points
        train_f  = [feat_vals[j] for j in range(n) if j != i]
        train_l  = [labels_bin[j] for j in range(n) if j != i]
        test_f   = feat_vals[i]
        test_l   = labels_bin[i]

        best_correct = -1
        best_T = None
        best_dir = None
        for direction in ["high→8bit", "low→8bit"]:
            sv = sorted(set(train_f))
            candidates = [v + 1e-6 for v in sv] + [v - 1e-6 for v in sv]
            for T in candidates:
                preds = [1 if (f >= T if direction == "high→8bit" else f <= T) else 0
                         for f in train_f]
                c = sum(p == l for p, l in zip(preds, train_l))
                if c > best_correct:
                    best_correct = c
                    best_T = T
                    best_dir = direction

        pred_test = 1 if (test_f >= best_T if best_dir == "high→8bit"
                          else test_f <= best_T) else 0
        if pred_test == test_l:
            correct += 1
    return correct / n


# ── 2-feature AND classifier ─────────────────────────────────────────────────

def sweep_2feature(f1_vals: list[float], f2_vals: list[float],
                   labels_bin: list[int], fid_deltas: list[float],
                   f1_dir: str, f2_dir: str) -> dict:
    """
    Grid-search 2D thresholds for AND rule.
    predict_MXFP8 = (f1 OP1 T1) AND (f2 OP2 T2)
    where OP1/OP2 are >= or <= according to direction string.
    """
    n = len(f1_vals)

    def cands(fv):
        sv = sorted(set(fv))
        return [v + 1e-6 for v in sv] + [v - 1e-6 for v in sv]

    best = (0, 0, None, None)  # (correct, -fid, T1, T2)
    for T1 in cands(f1_vals):
        for T2 in cands(f2_vals):
            preds = []
            for v1, v2 in zip(f1_vals, f2_vals):
                p1 = (v1 >= T1) if f1_dir == "high→8bit" else (v1 <= T1)
                p2 = (v2 >= T2) if f2_dir == "high→8bit" else (v2 <= T2)
                preds.append(1 if (p1 and p2) else 0)
            correct = sum(p == l for p, l in zip(preds, labels_bin))
            total_fid = sum(d for p, l, d in zip(preds, labels_bin, fid_deltas)
                            if p != l)
            if (correct, -total_fid) > (best[0], best[1]):
                best = (correct, -total_fid, T1, T2)
    c, neg_fid, T1, T2 = best
    return {"acc": c / n, "correct": c, "T1": T1, "T2": T2,
            "mean_fid_delta": -neg_fid / n}


def loo_2feature(f1_vals, f2_vals, labels_bin, f1_dir, f2_dir) -> float:
    n = len(f1_vals)
    correct = 0

    def cands(fv):
        sv = sorted(set(fv))
        return [v + 1e-6 for v in sv] + [v - 1e-6 for v in sv]

    for i in range(n):
        tf1 = [f1_vals[j] for j in range(n) if j != i]
        tf2 = [f2_vals[j] for j in range(n) if j != i]
        tl  = [labels_bin[j] for j in range(n) if j != i]

        best_c = -1; bT1 = bT2 = None
        for T1 in cands(tf1):
            for T2 in cands(tf2):
                preds = []
                for v1, v2 in zip(tf1, tf2):
                    p1 = (v1 >= T1) if f1_dir == "high→8bit" else (v1 <= T1)
                    p2 = (v2 >= T2) if f2_dir == "high→8bit" else (v2 <= T2)
                    preds.append(1 if (p1 and p2) else 0)
                c = sum(p == l for p, l in zip(preds, tl))
                if c > best_c:
                    best_c = c; bT1 = T1; bT2 = T2

        v1, v2 = f1_vals[i], f2_vals[i]
        p1 = (v1 >= bT1) if f1_dir == "high→8bit" else (v1 <= bT1)
        p2 = (v2 >= bT2) if f2_dir == "high→8bit" else (v2 <= bT2)
        pred = 1 if (p1 and p2) else 0
        if pred == labels_bin[i]:
            correct += 1
    return correct / n


# ── 2-feature OR classifier ──────────────────────────────────────────────────

def _pred_or(v1, T1, d1, v2, T2, d2) -> int:
    """predict MXFP8 if (f1 fires) OR (f2 fires)."""
    p1 = (v1 >= T1) if d1 == "high→8bit" else (v1 <= T1)
    p2 = (v2 >= T2) if d2 == "high→8bit" else (v2 <= T2)
    return 1 if (p1 or p2) else 0


def sweep_2feature_or(f1_vals, f2_vals, labels_bin, fid_deltas, f1_dir, f2_dir) -> dict:
    n = len(f1_vals)
    def cands(fv):
        sv = sorted(set(fv))
        return [v + 1e-6 for v in sv] + [v - 1e-6 for v in sv]
    best = (0, 0, None, None)
    for T1 in cands(f1_vals):
        for T2 in cands(f2_vals):
            preds = [_pred_or(v1, T1, f1_dir, v2, T2, f2_dir)
                     for v1, v2 in zip(f1_vals, f2_vals)]
            correct  = sum(p == l for p, l in zip(preds, labels_bin))
            total_fid = sum(d for p, l, d in zip(preds, labels_bin, fid_deltas) if p != l)
            if (correct, -total_fid) > (best[0], best[1]):
                best = (correct, -total_fid, T1, T2)
    c, neg_fid, T1, T2 = best
    return {"acc": c / n, "correct": c, "T1": T1, "T2": T2,
            "mean_fid_delta": -neg_fid / n}


def loo_2feature_or(f1_vals, f2_vals, labels_bin, f1_dir, f2_dir) -> float:
    n = len(f1_vals)
    correct = 0
    def cands(fv):
        sv = sorted(set(fv))
        return [v + 1e-6 for v in sv] + [v - 1e-6 for v in sv]
    for i in range(n):
        tf1 = [f1_vals[j] for j in range(n) if j != i]
        tf2 = [f2_vals[j] for j in range(n) if j != i]
        tl  = [labels_bin[j] for j in range(n) if j != i]
        best_c = -1; bT1 = bT2 = None
        for T1 in cands(tf1):
            for T2 in cands(tf2):
                preds = [_pred_or(v1, T1, f1_dir, v2, T2, f2_dir)
                         for v1, v2 in zip(tf1, tf2)]
                c = sum(p == l for p, l in zip(preds, tl))
                if c > best_c:
                    best_c = c; bT1 = T1; bT2 = T2
        pred = _pred_or(f1_vals[i], bT1, f1_dir, f2_vals[i], bT2, f2_dir)
        if pred == labels_bin[i]:
            correct += 1
    return correct / n


# ── 2-feature Cascade classifier ──────────────────────────────────────────────
# Rule: if f1 fires → MXFP8; elif f2 fires → MXFP8; else → MXFP6
# (Equivalent to OR but thresholds trained in a sequential context.)

def _pred_cascade2(v1, T1, d1, v2, T2, d2) -> int:
    if (v1 >= T1) if d1 == "high→8bit" else (v1 <= T1):
        return 1
    if (v2 >= T2) if d2 == "high→8bit" else (v2 <= T2):
        return 1
    return 0


def sweep_cascade2(f1_vals, f2_vals, labels_bin, fid_deltas, f1_dir, f2_dir) -> dict:
    """Cascade sweep; T1 threshold for stage-1 feature, T2 for stage-2."""
    n = len(f1_vals)
    def cands(fv):
        sv = sorted(set(fv))
        return [v + 1e-6 for v in sv] + [v - 1e-6 for v in sv]
    best = (0, 0, None, None)
    for T1 in cands(f1_vals):
        for T2 in cands(f2_vals):
            preds = [_pred_cascade2(v1, T1, d1, v2, T2, d2)
                     for v1, v2, d1, d2 in zip(f1_vals, f2_vals,
                                                [f1_dir]*n, [f2_dir]*n)]
            correct  = sum(p == l for p, l in zip(preds, labels_bin))
            total_fid = sum(d for p, l, d in zip(preds, labels_bin, fid_deltas) if p != l)
            if (correct, -total_fid) > (best[0], best[1]):
                best = (correct, -total_fid, T1, T2)
    c, neg_fid, T1, T2 = best
    return {"acc": c / n, "correct": c, "T1": T1, "T2": T2,
            "mean_fid_delta": -neg_fid / n}


def loo_cascade2(f1_vals, f2_vals, labels_bin, f1_dir, f2_dir) -> float:
    n = len(f1_vals)
    correct = 0
    def cands(fv):
        sv = sorted(set(fv))
        return [v + 1e-6 for v in sv] + [v - 1e-6 for v in sv]
    for i in range(n):
        tf1 = [f1_vals[j] for j in range(n) if j != i]
        tf2 = [f2_vals[j] for j in range(n) if j != i]
        tl  = [labels_bin[j] for j in range(n) if j != i]
        best_c = -1; bT1 = bT2 = None
        for T1 in cands(tf1):
            for T2 in cands(tf2):
                preds = [_pred_cascade2(v1, T1, f1_dir, v2, T2, f2_dir)
                         for v1, v2 in zip(tf1, tf2)]
                c = sum(p == l for p, l in zip(preds, tl))
                if c > best_c:
                    best_c = c; bT1 = T1; bT2 = T2
        pred = _pred_cascade2(f1_vals[i], bT1, f1_dir, f2_vals[i], bT2, f2_dir)
        if pred == labels_bin[i]:
            correct += 1
    return correct / n


# ── 3-feature Greedy Cascade classifier ───────────────────────────────────────
# Fix best (f1, T1, d1, f2, T2, d2) from cascade-2, sweep f3.

def _pred_cascade3(v1, T1, d1, v2, T2, d2, v3, T3, d3) -> int:
    if (v1 >= T1) if d1 == "high→8bit" else (v1 <= T1): return 1
    if (v2 >= T2) if d2 == "high→8bit" else (v2 <= T2): return 1
    if (v3 >= T3) if d3 == "high→8bit" else (v3 <= T3): return 1
    return 0


def loo_cascade3_greedy(f1_vals, f2_vals, f3_vals,
                        f1_dir, f2_dir, f3_dir) -> float:
    """LOO accuracy for a fixed 3-feature cascade."""
    n = len(f1_vals)
    correct = 0
    def cands(fv):
        sv = sorted(set(fv))
        return [v + 1e-6 for v in sv] + [v - 1e-6 for v in sv]
    for i in range(n):
        tf = [[f[j] for j in range(n) if j != i]
              for f in [f1_vals, f2_vals, f3_vals]]
        tl = [labels_bin_placeholder[j] for j in range(n) if j != i]
        # inner sweep
        best_c = -1; bT1 = bT2 = bT3 = None
        for T1 in cands(tf[0]):
            for T2 in cands(tf[1]):
                for T3 in cands(tf[2]):
                    preds = [_pred_cascade3(v1, T1, f1_dir, v2, T2, f2_dir, v3, T3, f3_dir)
                             for v1, v2, v3 in zip(*tf)]
                    c = sum(p == l for p, l in zip(preds, tl))
                    if c > best_c:
                        best_c = c; bT1 = T1; bT2 = T2; bT3 = T3
        pred = _pred_cascade3(f1_vals[i], bT1, f1_dir, f2_vals[i], bT2, f2_dir,
                              f3_vals[i], bT3, f3_dir)
        if pred == labels_bin_placeholder[i]:
            correct += 1
    return correct / n


labels_bin_placeholder: list[int] = []  # filled at runtime; avoids closure capture


def sweep_cascade3_greedy(all_feats: dict[str, list[float]],
                          labels_bin: list[int],
                          fid_deltas: list[float],
                          best_f1: str, best_T1: float, best_d1: str,
                          best_f2: str, best_T2: float, best_d2: str,
                          best_dir: dict[str, str]) -> tuple:
    """Given fixed (f1, f2), sweep all remaining features as f3.
    Returns (best_loo, best_f3, best_T3, best_d3, in_sample_acc).
    """
    global labels_bin_placeholder
    labels_bin_placeholder = labels_bin  # make available inside loo_cascade3_greedy

    n = len(labels_bin)
    f1_vals = all_feats[best_f1]
    f2_vals = all_feats[best_f2]

    def cands(fv):
        sv = sorted(set(fv))
        return [v + 1e-6 for v in sv] + [v - 1e-6 for v in sv]

    best = (0, 0.0, None, None, None)  # (loo_correct, in_acc, f3, T3, d3)
    candidate_feats = [f for f in all_feats if f not in (best_f1, best_f2)]

    for f3 in candidate_feats:
        f3_vals = all_feats[f3]
        f3_dir  = best_dir[f3]

        # In-sample: sweep T3 for each candidate
        best_ins = 0; best_T3_ins = None
        for T3 in cands(f3_vals):
            preds = [_pred_cascade3(v1, best_T1, best_d1, v2, best_T2, best_d2,
                                    v3, T3, f3_dir)
                     for v1, v2, v3 in zip(f1_vals, f2_vals, f3_vals)]
            c = sum(p == l for p, l in zip(preds, labels_bin))
            if c > best_ins:
                best_ins = c; best_T3_ins = T3

        # LOO
        loo_correct = 0
        for i in range(n):
            tf1 = [f1_vals[j] for j in range(n) if j != i]
            tf2 = [f2_vals[j] for j in range(n) if j != i]
            tf3 = [f3_vals[j] for j in range(n) if j != i]
            tl  = [labels_bin[j] for j in range(n) if j != i]
            best_c = -1; bT3 = None
            for T3 in cands(tf3):
                preds = [_pred_cascade3(v1, best_T1, best_d1, v2, best_T2, best_d2,
                                        v3, T3, f3_dir)
                         for v1, v2, v3 in zip(tf1, tf2, tf3)]
                c = sum(p == l for p, l in zip(preds, tl))
                if c > best_c:
                    best_c = c; bT3 = T3
            pred = _pred_cascade3(f1_vals[i], best_T1, best_d1,
                                   f2_vals[i], best_T2, best_d2,
                                   f3_vals[i], bT3,    f3_dir)
            if pred == labels_bin[i]:
                loo_correct += 1

        if (loo_correct, best_ins) > (best[0], best[1]):
            best = (loo_correct, best_ins / n, f3, best_T3_ins, f3_dir)

    return best  # (loo_correct, in_acc, f3, T3, d3)


# ── FID-weighted LOO helper ───────────────────────────────────────────────────

def fid_weighted_score(preds: list[int], labels_bin: list[int],
                       margins: list[float]) -> float:
    """FID-weighted accuracy: sum(correct_i * margin_i) / sum(margin_i)."""
    total_margin = sum(margins)
    if total_margin == 0:
        return 0.0
    weighted = sum(m for p, l, m in zip(preds, labels_bin, margins) if p == l)
    return weighted / total_margin


# ── FID delta helper ─────────────────────────────────────────────────────────

def compute_fid_deltas(labels: dict[int, str],
                       fid_by_block: dict[int, dict[str, float]]) -> dict[int, dict]:
    """
    For each block, compute cost of using MXFP6_E2M3 vs MXFP8.
    Returns {block_idx: {"delta_6→8": float, "delta_8→6": float}}
    delta_6→8 = FID(MXFP8) - FID(MXFP6)   (cost of using 8-bit when 6-bit is optimal)
    delta_8→6 = FID(MXFP6) - FID(MXFP8)   (cost of using 6-bit when 8-bit is optimal)
    misclassification_cost = delta_{predicted_wrong}
    """
    result = {}
    for blk, fmt in labels.items():
        fids = fid_by_block.get(blk, {})
        fid8 = next((v for k, v in fids.items() if "MXFP8_bs32" in k), None)
        fid6 = next((v for k, v in fids.items() if "MXFP6_E2M3_bs32" in k), None)
        if fid8 is not None and fid6 is not None:
            result[blk] = {
                "fid6": fid6, "fid8": fid8,
                "delta_8→6": max(0.0, fid6 - fid8),  # cost if MXFP8-optimal but predict 6
                "delta_6→8": max(0.0, fid8 - fid6),  # cost if MXFP6-optimal but predict 8
                "optimal": fmt,
            }
    return result


# ── decision tree (sklearn) ──────────────────────────────────────────────────

def run_dt_ablation(feat_matrix: list[dict], labels_bin: list[int],
                    feat_names: list[str]) -> None:
    if not SKLEARN:
        return
    import numpy as np

    X = np.array([[row[f] for f in feat_names] for row in feat_matrix])
    y = np.array(labels_bin)

    # Depth-2 tree
    print("\n── Decision Tree (depth=2, LOO-CV) ─────────────────────────────")
    loo = LeaveOneOut()
    dt  = DecisionTreeClassifier(max_depth=2, random_state=42)
    preds = []
    for train_idx, test_idx in loo.split(X):
        dt.fit(X[train_idx], y[train_idx])
        preds.append(dt.predict(X[test_idx])[0])
    loo_acc = sum(p == l for p, l in zip(preds, y)) / len(y)
    print(f"  DT depth=2 LOO-CV acc: {loo_acc*100:.1f}%  ({sum(p==l for p,l in zip(preds,y))}/28)")

    # Full fit for interpretability
    dt.fit(X, y)
    in_acc = sum(dt.predict(X) == y) / len(y)
    print(f"  DT depth=2 in-sample:  {in_acc*100:.1f}%")
    print("\n  Tree structure:")
    print(export_text(dt, feature_names=feat_names, spacing=3))

    # Logistic regression (L2)
    print("── Logistic Regression (LOO-CV) ────────────────────────────────")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    preds_lr = []
    for train_idx, test_idx in loo.split(X_s):
        lr.fit(X_s[train_idx], y[train_idx])
        preds_lr.append(lr.predict(X_s[test_idx])[0])
    lr_acc = sum(p == l for p, l in zip(preds_lr, y)) / len(y)
    print(f"  LR (all features) LOO-CV: {lr_acc*100:.1f}%  ({sum(p==l for p,l in zip(preds_lr,y))}/28)")

    # Feature importance
    lr.fit(X_s, y)
    coef = sorted(zip(feat_names, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    print("\n  Top feature weights (standardised):")
    for name, w in coef[:8]:
        bar = "█" * int(abs(w) * 5)
        sign = "+" if w > 0 else "-"
        print(f"    {name:20s}  {sign}{abs(w):.3f}  {bar}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir",   default="results/sensitivity")
    ap.add_argument("--dataset",    default="MJHQ")
    ap.add_argument("--top_pairs",  type=int, default=10,
                    help="Number of top 2-feature pairs to show")
    ap.add_argument("--v2", action="store_true", default=True,
                    help="Use extended per-layer-type features (v2)")
    args = ap.parse_args()

    base = os.path.join(args.save_dir, args.dataset)
    csv_path = os.path.join(base, "weight_stats_analysis.csv")

    print("=" * 72)
    print("  pixart_stats_ablation.py — Feature Ablation Study")
    print("=" * 72)

    # ── load data ────────────────────────────────────────────────────────
    labels     = load_labels(base)                         # {blk: fmt}
    stats_csv  = load_stats_csv(csv_path)                  # {blk: [layer_rows]} (no names)
    stats_csv2 = load_stats_csv_v2(csv_path)               # {blk: [layer_rows+names]}
    act_csv    = load_act_csv(os.path.join(base, "activation_stats.csv"))
    fid_data   = load_fid_by_block(base)                   # {blk: {key: fid}}
    fid_costs  = compute_fid_deltas(labels, fid_data)

    blocks = sorted(labels.keys())
    n = len(blocks)

    # ── FID margins for each block ────────────────────────────────────────
    fid_margins = []
    for blk in blocks:
        c = fid_costs.get(blk, {})
        margin = max(c.get("delta_8→6", 0.0), c.get("delta_6→8", 0.0))
        fid_margins.append(margin)

    # ── build BOTH feature matrices ───────────────────────────────────────
    feat_matrix_v1  = []   # original 16 features
    feat_matrix_v2  = []   # extended per-layer-type features
    labels_bin      = []   # 1 = MXFP8 optimal
    fid_delta_per_block = []

    for blk in blocks:
        feats_v1 = block_features(stats_csv[blk], blk)
        feat_matrix_v1.append(feats_v1)

        a_rows = act_csv.get(blk, [])
        feats_v2 = block_features_v2(stats_csv2[blk], a_rows, blk)
        feat_matrix_v2.append(feats_v2)

        is_8bit = 1 if _FAMILY[labels[blk]] == "8bit" else 0
        labels_bin.append(is_8bit)
        cost = fid_costs[blk]["delta_8→6"] if is_8bit else fid_costs[blk]["delta_6→8"]
        fid_delta_per_block.append(cost)

    # ── choose active feature matrix ──────────────────────────────────────
    feat_matrix = feat_matrix_v2 if args.v2 else feat_matrix_v1
    feat_names  = sorted(feat_matrix[0].keys())

    # ── dataset summary ──────────────────────────────────────────────────
    n8 = sum(labels_bin)
    n6 = n - n8
    print(f"\nDataset: {n} blocks  |  MXFP8-optimal: {n8}  |  MXFP6-optimal: {n6}")
    print(f"Majority-class baseline: {max(n8,n6)/n*100:.1f}%  ({max(n8,n6)}/{n})")

    # ── Level 1: single-feature ablation ────────────────────────────────
    print("\n" + "═" * 72)
    print("  Level 1: Single-Feature Threshold Classifiers")
    print("═" * 72)
    print(f"{'Feature':22s}  {'In-samp':>8}  {'LOO-CV':>8}  {'Direction':>14}  "
          f"{'Threshold':>10}  {'ΔFid':>6}")
    print("─" * 72)

    single_results = []
    for feat in feat_names:
        fv = [row[feat] for row in feat_matrix]
        res = sweep_single(fv, labels_bin, fid_delta_per_block)
        loo = loo_single(fv, labels_bin)
        single_results.append((loo, res["acc"], feat, res, loo))
        print(f"  {feat:20s}  {res['acc']*100:6.1f}%  {loo*100:6.1f}%  "
              f"  {res['direction']:12s}  {res['threshold']:10.4f}  "
              f"{res['mean_fid_delta']:6.3f}")

    single_results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print(f"\n  → Best single feature: '{single_results[0][2]}' "
          f"LOO={single_results[0][0]*100:.1f}%  in-sample={single_results[0][1]*100:.1f}%")

    # ── Level 2: 2-feature AND combinations ─────────────────────────────
    print("\n" + "═" * 72)
    print("  Level 2: 2-Feature AND Combinations (top pairs, in-sample)")
    print("═" * 72)
    print(f"{'Feature 1':22s}  {'Feature 2':22s}  {'In-samp':>8}  {'LOO-CV':>8}  {'ΔFid':>6}")
    print("─" * 72)

    # Determine best direction per feature from single-feature sweep
    best_dir = {}
    for feat in feat_names:
        fv = [row[feat] for row in feat_matrix]
        res = sweep_single(fv, labels_bin, fid_delta_per_block)
        best_dir[feat] = res["direction"]

    pair_results = []
    feat_list = feat_names[:]
    for f1, f2 in itertools.combinations(feat_list, 2):
        fv1 = [row[f1] for row in feat_matrix]
        fv2 = [row[f2] for row in feat_matrix]
        res = sweep_2feature(fv1, fv2, labels_bin, fid_delta_per_block,
                             best_dir[f1], best_dir[f2])
        loo = loo_2feature(fv1, fv2, labels_bin, best_dir[f1], best_dir[f2])
        pair_results.append((loo, res["acc"], f1, f2, res, loo))

    pair_results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for loo, acc, f1, f2, res, _ in pair_results[:args.top_pairs]:
        print(f"  {f1:20s}  {f2:20s}  {acc*100:6.1f}%  {loo*100:6.1f}%  "
              f"{res['mean_fid_delta']:6.3f}")

    best2 = pair_results[0]
    print(f"\n  → Best 2-feature AND: '{best2[2]}' + '{best2[3]}' "
          f"LOO={best2[0]*100:.1f}%  in-sample={best2[1]*100:.1f}%")
    print(f"    T1={best2[4]['T1']:.4f} ({best_dir[best2[2]]}), "
          f"T2={best2[4]['T2']:.4f} ({best_dir[best2[3]]})")

    # ── Level 2b: 2-feature OR ───────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  Level 2b: 2-Feature OR Combinations (top pairs)")
    print("═" * 72)
    print(f"{'Feature 1':28s}  {'Feature 2':28s}  {'In-samp':>8}  {'LOO-CV':>8}  {'ΔFid':>6}")
    print("─" * 72)

    or_results = []
    for f1, f2 in itertools.combinations(feat_list, 2):
        fv1 = [row[f1] for row in feat_matrix]
        fv2 = [row[f2] for row in feat_matrix]
        res = sweep_2feature_or(fv1, fv2, labels_bin, fid_delta_per_block,
                                best_dir[f1], best_dir[f2])
        loo = loo_2feature_or(fv1, fv2, labels_bin, best_dir[f1], best_dir[f2])
        preds = [_pred_or(v1, res["T1"], best_dir[f1], v2, res["T2"], best_dir[f2])
                 for v1, v2 in zip(fv1, fv2)]
        fw  = fid_weighted_score(preds, labels_bin, fid_margins)
        or_results.append((loo, res["acc"], f1, f2, res, fw))

    or_results.sort(key=lambda x: (x[0], x[1], x[5]), reverse=True)
    for loo, acc, f1, f2, res, fw in or_results[:args.top_pairs]:
        print(f"  {f1:26s}  {f2:26s}  {acc*100:6.1f}%  {loo*100:6.1f}%  "
              f"{res['mean_fid_delta']:6.3f}  fid_w={fw*100:.1f}%")

    best2b = or_results[0]
    print(f"\n  → Best 2-feature OR: '{best2b[2]}' + '{best2b[3]}' "
          f"LOO={best2b[0]*100:.1f}%  in-sample={best2b[1]*100:.1f}%  FID-w={best2b[5]*100:.1f}%")

    # ── Level 2c: 2-feature Cascade ─────────────────────────────────────
    print("\n" + "═" * 72)
    print("  Level 2c: 2-Feature Cascade (if f1 ≥ T1 → 8bit; elif f2 ≥ T2 → 8bit)")
    print("═" * 72)
    print(f"{'Feature 1 (stage-1)':28s}  {'Feature 2 (stage-2)':28s}  "
          f"{'In-samp':>8}  {'LOO-CV':>8}  {'ΔFid':>6}")
    print("─" * 72)

    cascade_results = []
    for f1, f2 in itertools.permutations(feat_list, 2):  # order matters in cascade
        fv1 = [row[f1] for row in feat_matrix]
        fv2 = [row[f2] for row in feat_matrix]
        res = sweep_cascade2(fv1, fv2, labels_bin, fid_delta_per_block,
                             best_dir[f1], best_dir[f2])
        loo = loo_cascade2(fv1, fv2, labels_bin, best_dir[f1], best_dir[f2])
        preds = [_pred_cascade2(v1, res["T1"], best_dir[f1],
                                v2, res["T2"], best_dir[f2])
                 for v1, v2 in zip(fv1, fv2)]
        fw = fid_weighted_score(preds, labels_bin, fid_margins)
        cascade_results.append((loo, res["acc"], f1, f2, res, fw))

    cascade_results.sort(key=lambda x: (x[0], x[1], x[5]), reverse=True)
    for loo, acc, f1, f2, res, fw in cascade_results[:args.top_pairs]:
        print(f"  {f1:26s}  {f2:26s}  {acc*100:6.1f}%  {loo*100:6.1f}%  "
              f"{res['mean_fid_delta']:6.3f}  fid_w={fw*100:.1f}%")

    best2c = cascade_results[0]
    print(f"\n  → Best Cascade-2: stage1='{best2c[2]}' (T={best2c[4]['T1']:.4f}, "
          f"{best_dir[best2c[2]]})  "
          f"stage2='{best2c[3]}' (T={best2c[4]['T2']:.4f}, {best_dir[best2c[3]]})  "
          f"LOO={best2c[0]*100:.1f}%")

    # ── Level 3: decision tree ───────────────────────────────────────────
    top_feats = [r[2] for r in single_results[:8]]
    print("\n" + "═" * 72)
    print(f"  Level 3: sklearn classifiers (top-8 features)")
    print("═" * 72)
    run_dt_ablation(feat_matrix, labels_bin, top_feats)

    # ── Level 3b: 3-feature Greedy Cascade ──────────────────────────────
    print("\n" + "═" * 72)
    print("  Level 3b: 3-Feature Greedy Cascade (fix best cascade-2, sweep f3)")
    print("═" * 72)

    bf1, bf2   = best2c[2], best2c[3]
    bT1, bT2   = best2c[4]["T1"], best2c[4]["T2"]
    bd1, bd2   = best_dir[bf1], best_dir[bf2]
    all_feats_map = {f: [row[f] for row in feat_matrix] for f in feat_names}

    loo_c3, ins3, bf3, bT3, bd3 = sweep_cascade3_greedy(
        all_feats_map, labels_bin, fid_delta_per_block,
        bf1, bT1, bd1, bf2, bT2, bd2, best_dir)

    fv3 = all_feats_map.get(bf3, [])
    if fv3 and bT3 is not None:
        preds3 = [_pred_cascade3(v1, bT1, bd1, v2, bT2, bd2, v3, bT3, bd3)
                  for v1, v2, v3 in zip(all_feats_map[bf1],
                                        all_feats_map[bf2], fv3)]
        fw3 = fid_weighted_score(preds3, labels_bin, fid_margins)
        ins3_acc = sum(p == l for p, l in zip(preds3, labels_bin)) / n
        print(f"  Stage-1: '{bf1}'  T={bT1:.4f}  ({bd1})")
        print(f"  Stage-2: '{bf2}'  T={bT2:.4f}  ({bd2})")
        print(f"  Stage-3: '{bf3}'  T={bT3:.4f}  ({bd3})")
        print(f"  In-sample: {ins3_acc*100:.1f}%  LOO: {loo_c3/n*100:.1f}%  "
              f"FID-weighted: {fw3*100:.1f}%  ({loo_c3}/28)")
        print()
        # Per-block breakdown
        print(f"  {'Blk':>4}  {'Pred':>10}  {'Label':>10}  {'Match':>6}  "
              f"{'FID_margin':>10}  {'Stage':>8}")
        print("  " + "─" * 56)
        for i, blk in enumerate(blocks):
            p   = preds3[i]
            lbl = labels_bin[i]
            v1_ = all_feats_map[bf1][i]
            v2_ = all_feats_map[bf2][i]
            v3_ = fv3[i]
            # determine which stage fired
            if (v1_ >= bT1) if bd1 == "high→8bit" else (v1_ <= bT1):
                stage = "s1"
            elif (v2_ >= bT2) if bd2 == "high→8bit" else (v2_ <= bT2):
                stage = "s2"
            elif (v3_ >= bT3) if bd3 == "high→8bit" else (v3_ <= bT3):
                stage = "s3"
            else:
                stage = "6bit"
            pred_str  = "MXFP8" if p else "MXFP6"
            label_str = "MXFP8" if lbl else "MXFP6"
            match = "O" if p == lbl else "X"
            print(f"  {blk:4d}  {pred_str:>10}  {label_str:>10}  {match:>6}  "
                  f"{fid_margins[i]:10.3f}  {stage:>8}")
    else:
        print("  (No f3 improved over cascade-2)")

    # ── summary ──────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  Summary")
    print("═" * 72)
    print(f"  Majority-class baseline :  {max(n8,n6)/n*100:.1f}%  ({max(n8,n6)}/{n})")
    print(f"  Weight-only (stats phase):  67.9%  (19/28)  LOO")
    print(f"  4-criteria (criteria ph.):  71.4%  (20/28)  LOO")
    for loo, acc, feat, res, _ in single_results[:3]:
        print(f"  Best single  [{feat:22s}]: {acc*100:.1f}% in-sample | {loo*100:.1f}% LOO")
    b2and = pair_results[0]
    print(f"  Best 2-AND   [{b2and[2][:12]}+{b2and[3][:12]}]: "
          f"{b2and[1]*100:.1f}% in-sample | {b2and[0]*100:.1f}% LOO")
    print(f"  Best 2-OR    [{best2b[2][:12]}+{best2b[3][:12]}]: "
          f"{best2b[1]*100:.1f}% in-sample | {best2b[0]*100:.1f}% LOO  "
          f"FID-w={best2b[5]*100:.1f}%")
    print(f"  Best Cascade-2: {best2c[0]*100:.1f}% LOO  "
          f"FID-w={best2c[5]*100:.1f}%  ({int(best2c[0]*n)}/28)")
    if fv3 and bT3 is not None:
        print(f"  Best Cascade-3: {loo_c3/n*100:.1f}% LOO  "
              f"FID-w={fw3*100:.1f}%  ({loo_c3}/28)")
    print()

    # ── save results ─────────────────────────────────────────────────────
    cascade3_entry = {}
    if fv3 and bT3 is not None:
        cascade3_entry = {
            "f1": bf1, "T1": bT1, "d1": bd1,
            "f2": bf2, "T2": bT2, "d2": bd2,
            "f3": bf3, "T3": bT3, "d3": bd3,
            "loo_acc": round(loo_c3 / n, 4),
            "in_sample_acc": round(ins3_acc, 4),
            "fid_weighted": round(fw3, 4),
        }

    out = {
        "n_blocks": n,
        "n_mxfp8_optimal": n8,
        "n_mxfp6_optimal": n6,
        "majority_baseline": max(n8, n6) / n,
        "single_features": [
            {"feature": feat, "in_sample_acc": res["acc"], "loo_acc": loo,
             "direction": res["direction"], "threshold": res["threshold"],
             "mean_fid_delta": res["mean_fid_delta"]}
            for loo, acc, feat, res, _ in single_results
        ],
        "top_pairs_AND": [
            {"f1": f1, "f2": f2, "in_sample_acc": res["acc"], "loo_acc": loo,
             "T1": res["T1"], "T2": res["T2"],
             "f1_direction": best_dir[f1], "f2_direction": best_dir[f2],
             "mean_fid_delta": res["mean_fid_delta"]}
            for loo, acc, f1, f2, res, _ in pair_results[:10]
        ],
        "top_pairs_OR": [
            {"f1": f1, "f2": f2, "in_sample_acc": res["acc"], "loo_acc": loo,
             "T1": res["T1"], "T2": res["T2"],
             "f1_direction": best_dir[f1], "f2_direction": best_dir[f2],
             "mean_fid_delta": res["mean_fid_delta"], "fid_weighted": fw}
            for loo, acc, f1, f2, res, fw in or_results[:10]
        ],
        "top_cascade2": [
            {"f1": f1, "f2": f2, "in_sample_acc": res["acc"], "loo_acc": loo,
             "T1": res["T1"], "T2": res["T2"],
             "f1_direction": best_dir[f1], "f2_direction": best_dir[f2],
             "mean_fid_delta": res["mean_fid_delta"], "fid_weighted": fw}
            for loo, acc, f1, f2, res, fw in cascade_results[:10]
        ],
        "best_cascade3": cascade3_entry,
    }
    out_path = os.path.join(base, "ablation_results_v2.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
