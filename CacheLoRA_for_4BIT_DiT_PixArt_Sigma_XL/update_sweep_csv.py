"""
update_sweep_csv.py
results/MJHQ 아래 모든 metrics.json을 읽어 sweep_all_results.csv를 재생성.
5개 method: RTN, SVDQUANT, FP4DIT, HQDIT, CONVROT (PixArt-Sigma)
"""
import json
import os
import csv

DATA_ROOT   = "/data/james_dit_pixart_sigma_xl_mjhq"
RESULTS_DIR = DATA_ROOT  # method별 하위 디렉토리를 glob으로 탐색
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
# /data/james_dit_pixart_sigma_xl_mjhq/{METHOD}/MJHQ/{TAG}/metrics.json
for method_dir in sorted(os.listdir(DATA_ROOT)):
    mjhq_dir = os.path.join(DATA_ROOT, method_dir, "MJHQ")
    if not os.path.isdir(mjhq_dir):
        continue
    for tag in sorted(os.listdir(mjhq_dir)):
        metrics_path = os.path.join(mjhq_dir, tag, "metrics.json")
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
print(f"sweep_all_results.csv updated: {total} rows")
for m in ("RTN", "SVDQUANT", "FP4DIT", "HQDIT", "CONVROT"):
    count = sum(1 for r in rows if r["quant_method"] == m)
    if count:
        print(f"  {m}: {count} runs")
