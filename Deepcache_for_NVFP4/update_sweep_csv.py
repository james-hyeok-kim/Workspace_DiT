"""
update_sweep_csv.py
/data/jameskimh/james_dit_pixart_xl_mjhq/{METHOD}/{run} 아래 모든 metrics.json을 읽어 sweep_all_results.csv를 재생성.
기존 4개 method + 신규 3개 method (FP4DIT, HQDIT, SIXBIT) 모두 포함.
"""
import glob
import json
import os
import csv

RESULTS_DIR = "/data/jameskimh/james_dit_pixart_xl_mjhq"
OUT_CSV     = os.path.join(os.path.dirname(__file__), "results", "sweep_all_results.csv")

FIELDS = [
    "tag", "quant_method", "cache_mode", "lora_rank",
    "num_steps", "num_samples",
    "deepcache_start", "deepcache_end", "deepcache_interval",
    "speedup_est",
    "fid", "is", "psnr", "ssim", "lpips", "clip",
    "time_per_image_sec", "calib_time_sec",
]

rows = []
for metrics_path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "*", "metrics.json"))):
    parts = metrics_path.replace("\\", "/").split("/")
    method, run = parts[-3], parts[-2]
    tag = f"{method}_{run}"
    if not os.path.isfile(metrics_path):
        continue
    with open(metrics_path) as f:
        d = json.load(f)
    row = {"tag": tag}
    for k in FIELDS[1:]:
        row[k] = d.get(k, "")
    rows.append(row)

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

total = len(rows)
new_methods = sum(1 for r in rows if r["quant_method"] in ("FP4DIT", "HQDIT", "SIXBIT"))
print(f"sweep_all_results.csv updated: {total} rows ({new_methods} from new methods)")
for m in ("FP4DIT", "HQDIT", "SIXBIT"):
    count = sum(1 for r in rows if r["quant_method"] == m)
    if count:
        print(f"  {m}: {count} runs")
