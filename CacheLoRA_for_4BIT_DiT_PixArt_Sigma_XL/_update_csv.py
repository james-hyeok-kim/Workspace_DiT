"""
_update_csv.py — HQDIT rows update for sweep_all_results.csv
Called by run_pipeline_parallel.sh after HQDIT runs finish.
"""
import csv, json, os, sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR   = Path(__file__).parent
RESULTS_ROOT = Path("/data/james_dit_pixart_sigma_xl_mjhq/HQDIT/MJHQ")
CSV_PATH     = SCRIPT_DIR / "results" / "sweep_all_results.csv"

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

def safe_float(v, default=float("inf")):
    try:    return float(v)
    except: return default

updated = added = 0
existing_tags = {r["tag"] for r in rows}

# ── Update existing HQDIT rows ────────────────────────────────────────────────
new_rows = []
for row in rows:
    tag = row["tag"]
    if not tag.startswith("HQDIT_"):
        new_rows.append(row)
        continue
    mp = RESULTS_ROOT / tag / "metrics.json"
    if not mp.exists():
        print(f"  REMOVE (no metrics.json): {tag}")
        continue
    m = json.load(open(mp))
    row["fid"]                = m["fid"]
    row["is"]                 = m["is"]
    row["psnr"]               = m["psnr"]
    row["ssim"]               = m["ssim"]
    row["lpips"]              = m["lpips"]
    row["clip"]               = m.get("clip", "")
    row["time_per_image_sec"] = m["time_per_image_sec"]
    row["calib_time_sec"]     = m.get("calib_time_sec", "")
    row["num_samples"]        = m.get("num_samples", row["num_samples"])
    new_rows.append(row)
    updated += 1

# ── Add newly generated HQDIT rows not yet in CSV ────────────────────────────
for mp in sorted(RESULTS_ROOT.glob("HQDIT_*/metrics.json")):
    tag = mp.parent.name
    if tag in existing_tags:
        continue
    m = json.load(open(mp))
    steps = int(tag.split("steps")[-1])
    cache_mode = "none"
    ds = de = di = ""
    lora_rank = ""
    sp = 1.0
    if "deepcache" in tag:
        cache_mode = "deepcache"
        c = tag.split("_c")[1].split("_steps")[0]
        ds, de = c.split("-"); di = "2"
        n_deep = int(de) - int(ds)
        sp = round(28 / ((28 + (28 - n_deep)) / 2), 3)
    elif "cache_lora" in tag:
        cache_mode = "cache_lora"; lora_rank = "4"
        c = tag.split("_c")[1].split("_steps")[0]
        ds, de = c.split("-"); di = "2"
        n_deep = int(de) - int(ds)
        sp = round(28 / ((28 + (28 - n_deep)) / 2), 3)
    derived_false = {k: False for k in [
        "best_fid_per_method","best_fid_none_s20","best_fid_cache_s20",
        "best_speed_per_method","best_is_per_method","best_clip_per_method",
        "pareto_s20","best_speed_in_config","best_fid_in_config"]}
    row = {
        "tag": tag, "quant_method": "HQDIT", "cache_mode": cache_mode,
        "lora_rank": lora_rank, "num_steps": steps,
        "num_samples": m.get("num_samples", 100),
        "deepcache_start": ds, "deepcache_end": de,
        "deepcache_interval": di, "speedup_est": sp,
        "fid": m["fid"], "is": m["is"], "psnr": m["psnr"],
        "ssim": m["ssim"], "lpips": m["lpips"],
        "clip": m.get("clip", ""), "time_per_image_sec": m["time_per_image_sec"],
        "calib_time_sec": m.get("calib_time_sec", ""),
        **derived_false,
    }
    new_rows.append(row)
    added += 1
    print(f"  ADDED: {tag}")

# ── Re-compute all derived boolean columns ────────────────────────────────────
derived_cols = [
    "best_fid_per_method","best_fid_none_s20","best_fid_cache_s20",
    "best_speed_per_method","best_is_per_method","best_clip_per_method",
    "pareto_s20","best_speed_in_config","best_fid_in_config",
]
for row in new_rows:
    for c in derived_cols: row[c] = False

mb = defaultdict(lambda: float("inf"))
for r in new_rows:
    f = safe_float(r["fid"])
    if f < mb[r["quant_method"]]: mb[r["quant_method"]] = f
for r in new_rows:
    if safe_float(r["fid"]) == mb[r["quant_method"]]: r["best_fid_per_method"] = True

ns20 = [safe_float(r["fid"]) for r in new_rows if r["cache_mode"]=="none" and str(r["num_steps"])=="20"]
if ns20:
    b = min(ns20)
    for r in new_rows:
        if r["cache_mode"]=="none" and str(r["num_steps"])=="20" and safe_float(r["fid"])==b:
            r["best_fid_none_s20"] = True

cs20 = [safe_float(r["fid"]) for r in new_rows if r["cache_mode"]!="none" and str(r["num_steps"])=="20"]
if cs20:
    b = min(cs20)
    for r in new_rows:
        if r["cache_mode"]!="none" and str(r["num_steps"])=="20" and safe_float(r["fid"])==b:
            r["best_fid_cache_s20"] = True

ms = defaultdict(lambda: float("inf"))
for r in new_rows:
    t = safe_float(r["time_per_image_sec"])
    if t < ms[r["quant_method"]]: ms[r["quant_method"]] = t
for r in new_rows:
    if safe_float(r["time_per_image_sec"]) == ms[r["quant_method"]]: r["best_speed_per_method"] = True

mi = defaultdict(lambda: -float("inf"))
for r in new_rows:
    v = safe_float(r["is"], -float("inf"))
    if v > mi[r["quant_method"]]: mi[r["quant_method"]] = v
for r in new_rows:
    if safe_float(r["is"], -float("inf")) == mi[r["quant_method"]]: r["best_is_per_method"] = True

mc = defaultdict(lambda: -float("inf"))
for r in new_rows:
    v = safe_float(r["clip"], -float("inf"))
    if v > mc[r["quant_method"]]: mc[r["quant_method"]] = v
for r in new_rows:
    if safe_float(r["clip"], -float("inf")) == mc[r["quant_method"]]: r["best_clip_per_method"] = True

s20 = [(i, r) for i, r in enumerate(new_rows) if str(r["num_steps"]) == "20"]
for i, r in s20:
    fi = safe_float(r["fid"]); si = safe_float(r["time_per_image_sec"])
    dominated = any(
        safe_float(r2["fid"]) <= fi and safe_float(r2["time_per_image_sec"]) <= si
        and (safe_float(r2["fid"]) < fi or safe_float(r2["time_per_image_sec"]) < si)
        for _, r2 in s20 if r2 is not r
    )
    if not dominated: new_rows[i]["pareto_s20"] = True

cfg_s = defaultdict(lambda: float("inf"))
for r in new_rows:
    k = (r["quant_method"], r["cache_mode"], r["deepcache_start"], r["deepcache_end"])
    t = safe_float(r["time_per_image_sec"])
    if t < cfg_s[k]: cfg_s[k] = t
for r in new_rows:
    k = (r["quant_method"], r["cache_mode"], r["deepcache_start"], r["deepcache_end"])
    if safe_float(r["time_per_image_sec"]) == cfg_s[k]: r["best_speed_in_config"] = True

cfg_f = defaultdict(lambda: float("inf"))
for r in new_rows:
    k = (r["quant_method"], r["cache_mode"], r["deepcache_start"], r["deepcache_end"])
    f = safe_float(r["fid"])
    if f < cfg_f[k]: cfg_f[k] = f
for r in new_rows:
    k = (r["quant_method"], r["cache_mode"], r["deepcache_start"], r["deepcache_end"])
    if safe_float(r["fid"]) == cfg_f[k]: r["best_fid_in_config"] = True

# ── Write ─────────────────────────────────────────────────────────────────────
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(new_rows)

print(f"sweep_all_results.csv: {updated} updated, {added} added → {len(new_rows)} total rows")
