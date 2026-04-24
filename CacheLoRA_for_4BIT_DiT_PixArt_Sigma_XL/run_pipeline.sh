#!/bin/bash
# run_pipeline.sh
# Step 1: HQDIT none steps {15, 20}  (missing 2 configs)
# Step 2: Update sweep_all_results.csv with new HQDIT data
# Step 3: SVDQUANT advanced sweep (4 directions, 60 runs)
#
# Usage:
#   bash run_pipeline.sh --gpu 0
#   bash run_pipeline.sh --gpu 0 --test_run

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
ENV_PYTHON="/home/jameskimh/.dit/bin/accelerate"
CSV_PATH="$SCRIPT_DIR/results/sweep_all_results.csv"

GPU_ID=""
TEST_RUN=false
TEST_FLAG=""
NUM_SAMPLES=100

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu)         GPU_ID="$2"; shift ;;
        --test_run)    TEST_RUN=true; TEST_FLAG="--test_run"; NUM_SAMPLES=2 ;;
        --num_samples) NUM_SAMPLES="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "Using GPU: $GPU_ID"
fi

echo ""
echo "########################################################"
echo "  STEP 1: HQDIT none steps {15, 20}"
echo "########################################################"

cd "$SCRIPT_DIR"

for STEP in 15 20; do
    TAG="HQDIT_none_steps${STEP}"
    RESULT_DIR="/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/HQDIT/${TAG}"
    if [ -f "$RESULT_DIR/metrics.json" ] && [ "$TEST_RUN" = false ]; then
        echo "  SKIP: $TAG (already exists)"
        continue
    fi
    echo ""
    echo "  Running: $TAG"
    $ENV_PYTHON launch \
        --num_processes 1 \
        "$PYTHON_SCRIPT" \
        --quant_method   HQDIT \
        --cache_mode     none \
        --num_steps      "$STEP" \
        --num_samples    "$NUM_SAMPLES" \
        --guidance_scale 4.5 \
        $TEST_FLAG
    echo "  Done: $TAG"
done

echo ""
echo "########################################################"
echo "  STEP 2: Update sweep_all_results.csv"
echo "########################################################"

python3 << 'PYEOF'
import csv, json, os
from pathlib import Path
from collections import defaultdict

RESULTS_ROOT = Path("/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/HQDIT")
CSV_PATH     = Path("results/sweep_all_results.csv")

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

def safe_float(v, default=float("inf")):
    try: return float(v)
    except: return default

updated = added = 0
existing_tags = {r["tag"] for r in rows}

# Update or add HQDIT rows from metrics.json
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

# Add any newly generated HQDIT rows not yet in CSV
for p in sorted(RESULTS_ROOT.glob("HQDIT_*/metrics.json")):
    tag = p.parent.name
    if tag in existing_tags:
        continue
    m = json.load(open(p))
    # Build row from metrics.json (fill derived cols as False)
    parts = tag.split("_")
    method = parts[0]
    steps  = int(tag.split("steps")[-1])
    cache_mode = "none"
    ds = de = di = sp = ""
    lora_rank = ""
    if "deepcache" in tag:
        cache_mode = "deepcache"
        c = tag.split("_c")[1].split("_steps")[0]
        ds, de = c.split("-"); di = "2"
        n_deep = int(de) - int(ds); n_total = 28; n_always = n_total - n_deep
        sp = round(n_total / ((n_total + n_always * 1) / 2), 3)
    elif "cache_lora" in tag:
        cache_mode = "cache_lora"
        lora_rank = "4"
        c = tag.split("_c")[1].split("_steps")[0]
        ds, de = c.split("-"); di = "2"
        n_deep = int(de) - int(ds); n_total = 28; n_always = n_total - n_deep
        sp = round(n_total / ((n_total + n_always * 1) / 2), 3)
    else:
        sp = 1.0

    derived_false = {k: False for k in [
        "best_fid_per_method","best_fid_none_s20","best_fid_cache_s20",
        "best_speed_per_method","best_is_per_method","best_clip_per_method",
        "pareto_s20","best_speed_in_config","best_fid_in_config"]}
    row = {
        "tag": tag, "quant_method": method, "cache_mode": cache_mode,
        "lora_rank": lora_rank, "num_steps": steps,
        "num_samples": m.get("num_samples", 100),
        "deepcache_start": ds, "deepcache_end": de,
        "deepcache_interval": di, "speedup_est": sp,
        "fid": m["fid"], "is": m["is"], "psnr": m["psnr"],
        "ssim": m["ssim"], "lpips": m["lpips"],
        "clip": m.get("clip",""), "time_per_image_sec": m["time_per_image_sec"],
        "calib_time_sec": m.get("calib_time_sec",""),
        **derived_false,
    }
    new_rows.append(row)
    added += 1
    print(f"  ADDED: {tag}")

# Re-compute derived columns
derived_cols = [
    "best_fid_per_method","best_fid_none_s20","best_fid_cache_s20",
    "best_speed_per_method","best_is_per_method","best_clip_per_method",
    "pareto_s20","best_speed_in_config","best_fid_in_config",
]
for row in new_rows:
    for c in derived_cols: row[c] = False

method_best_fid = defaultdict(lambda: float("inf"))
for r in new_rows:
    f = safe_float(r["fid"])
    if f < method_best_fid[r["quant_method"]]: method_best_fid[r["quant_method"]] = f
for r in new_rows:
    if safe_float(r["fid"]) == method_best_fid[r["quant_method"]]: r["best_fid_per_method"] = True

none_s20 = [safe_float(r["fid"]) for r in new_rows if r["cache_mode"]=="none" and str(r["num_steps"])=="20"]
if none_s20:
    b = min(none_s20)
    for r in new_rows:
        if r["cache_mode"]=="none" and str(r["num_steps"])=="20" and safe_float(r["fid"])==b: r["best_fid_none_s20"]=True

cache_s20 = [safe_float(r["fid"]) for r in new_rows if r["cache_mode"]!="none" and str(r["num_steps"])=="20"]
if cache_s20:
    b = min(cache_s20)
    for r in new_rows:
        if r["cache_mode"]!="none" and str(r["num_steps"])=="20" and safe_float(r["fid"])==b: r["best_fid_cache_s20"]=True

method_best_spd = defaultdict(lambda: float("inf"))
for r in new_rows:
    t = safe_float(r["time_per_image_sec"])
    if t < method_best_spd[r["quant_method"]]: method_best_spd[r["quant_method"]] = t
for r in new_rows:
    if safe_float(r["time_per_image_sec"]) == method_best_spd[r["quant_method"]]: r["best_speed_per_method"]=True

method_best_is = defaultdict(lambda: -float("inf"))
for r in new_rows:
    v = safe_float(r["is"], -float("inf"))
    if v > method_best_is[r["quant_method"]]: method_best_is[r["quant_method"]] = v
for r in new_rows:
    if safe_float(r["is"],-float("inf")) == method_best_is[r["quant_method"]]: r["best_is_per_method"]=True

method_best_clip = defaultdict(lambda: -float("inf"))
for r in new_rows:
    v = safe_float(r["clip"], -float("inf"))
    if v > method_best_clip[r["quant_method"]]: method_best_clip[r["quant_method"]] = v
for r in new_rows:
    if safe_float(r["clip"],-float("inf")) == method_best_clip[r["quant_method"]]: r["best_clip_per_method"]=True

s20 = [(i,r) for i,r in enumerate(new_rows) if str(r["num_steps"])=="20"]
for i,r in s20:
    fi = safe_float(r["fid"]); si = safe_float(r["time_per_image_sec"])
    dominated = any(
        safe_float(r2["fid"])<=fi and safe_float(r2["time_per_image_sec"])<=si
        and (safe_float(r2["fid"])<fi or safe_float(r2["time_per_image_sec"])<si)
        for _,r2 in s20 if r2 is not r
    )
    if not dominated: new_rows[i]["pareto_s20"] = True

cfg_spd = defaultdict(lambda: float("inf"))
for r in new_rows:
    k=(r["quant_method"],r["cache_mode"],r["deepcache_start"],r["deepcache_end"])
    t=safe_float(r["time_per_image_sec"])
    if t<cfg_spd[k]: cfg_spd[k]=t
for r in new_rows:
    k=(r["quant_method"],r["cache_mode"],r["deepcache_start"],r["deepcache_end"])
    if safe_float(r["time_per_image_sec"])==cfg_spd[k]: r["best_speed_in_config"]=True

cfg_fid = defaultdict(lambda: float("inf"))
for r in new_rows:
    k=(r["quant_method"],r["cache_mode"],r["deepcache_start"],r["deepcache_end"])
    f=safe_float(r["fid"])
    if f<cfg_fid[k]: cfg_fid[k]=f
for r in new_rows:
    k=(r["quant_method"],r["cache_mode"],r["deepcache_start"],r["deepcache_end"])
    if safe_float(r["fid"])==cfg_fid[k]: r["best_fid_in_config"]=True

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(new_rows)

print(f"CSV updated: {updated} updated, {added} added, total {len(new_rows)} rows")
PYEOF

echo ""
echo "########################################################"
echo "  STEP 3: SVDQUANT advanced sweep (4 directions)"
echo "########################################################"

bash "$SCRIPT_DIR/run_svdquant_advanced.sh" \
    ${GPU_ID:+--gpu "$GPU_ID"} \
    ${TEST_FLAG} \
    --num_samples 20

echo ""
echo "########################################################"
echo "  ALL DONE"
echo "########################################################"
