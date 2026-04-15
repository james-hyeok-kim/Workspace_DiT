"""
실험 결과를 CSV로 저장
  - results/results_summary.csv : 전체 비교표
  - experiment.md의 결과표도 업데이트
"""
import json
import csv
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
DATASET   = "MJHQ"
CSV_PATH  = os.path.join(RESULT_DIR, "results_summary.csv")

CONFIGS = ["UNIFORM_FP4", "MP_ACT_ONLY", "MP_RANK_ONLY", "MP_MODERATE", "MP_AGGRESSIVE"]

BASELINE = {
    "config":          "BASELINE (NVFP4_DEFAULT_CFG)",
    "avg_act_bits":    4.0,
    "svd_savings_pct": 0.0,
    "FID":   161.30,
    "IS":    1.7318,
    "PSNR":  15.69,
    "SSIM":  0.5902,
    "LPIPS": None,
    "CLIP":  None,
    "beats_baseline_FID": False,
    "beats_baseline_IS":  False,
    "beats_both":         False,
    "note": "reference",
}

rows = [BASELINE]

for cfg in CONFIGS:
    path = os.path.join(RESULT_DIR, cfg, DATASET, "metrics.json")
    if not os.path.exists(path):
        print(f"  [SKIP] {cfg}: metrics.json not found")
        continue

    d  = json.load(open(path))
    pm = d["primary_metrics"]
    sm = d["secondary_metrics"]
    eb = d.get("effective_bitwidth", {})

    fid  = pm["FID"]
    is_v = pm["IS"]
    beat_fid  = fid  < BASELINE["FID"]
    beat_is   = is_v > BASELINE["IS"]

    rows.append({
        "config":          cfg,
        "avg_act_bits":    round(eb.get("avg_act_bits", 0), 2),
        "svd_savings_pct": round(eb.get("svd_savings_pct", 0), 1),
        "FID":             round(fid,           4),
        "IS":              round(is_v,           4),
        "PSNR":            round(sm["PSNR"],     4),
        "SSIM":            round(sm["SSIM"],     4),
        "LPIPS":           round(sm["LPIPS"],    4),
        "CLIP":            round(sm["CLIP"],     2),
        "beats_baseline_FID": beat_fid,
        "beats_baseline_IS":  beat_is,
        "beats_both":         beat_fid and beat_is,
        "note": ("FID+IS BEAT" if beat_fid and beat_is
                 else "FID BEAT" if beat_fid
                 else "IS BEAT"  if beat_is
                 else "-"),
    })

# ---- CSV 저장 ----
os.makedirs(RESULT_DIR, exist_ok=True)
fields = ["config", "avg_act_bits", "svd_savings_pct",
          "FID", "IS", "PSNR", "SSIM", "LPIPS", "CLIP",
          "beats_baseline_FID", "beats_baseline_IS", "beats_both", "note"]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV saved: {CSV_PATH}")

# ---- 콘솔 비교표 출력 ----
print(f"\n{'Config':<28} {'ActBit':>6} {'SVDsave':>8} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>7}  Note")
print("-" * 90)
for r in rows:
    lpips_str = f"{r['LPIPS']:.4f}" if r["LPIPS"] is not None else "N/A"
    print(
        f"  {r['config']:<26} {r['avg_act_bits']:>5.1f}b {r['svd_savings_pct']:>7.1f}%"
        f" {r['FID']:>8.2f} {r['IS']:>7.4f} {r['PSNR']:>7.2f} {r['SSIM']:>7.4f}"
        f"  {r['note']}"
    )

# ---- experiment.md 결과표 업데이트 ----
md_path = os.path.join(BASE_DIR, "experiment.md")
if os.path.exists(md_path):
    header  = "| Config | Avg Act Bits | SVD Savings | FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ | vs Baseline |"
    sep     = "|--------|-------------|-------------|-------|------|--------|--------|-------------|"
    new_rows = [header, sep]
    for r in rows:
        psnr_s = f"{r['PSNR']:.2f}" if r["PSNR"] is not None else "N/A"
        ssim_s = f"{r['SSIM']:.4f}" if r["SSIM"] is not None else "N/A"
        new_rows.append(
            f"| **{r['config']}** | {r['avg_act_bits']:.2f} | {r['svd_savings_pct']:.1f}% |"
            f" {r['FID']:.2f} | {r['IS']:.4f} | {psnr_s} | {ssim_s} | {r['note']} |"
        )

    md = open(md_path).read()
    # 기존 결과표 블록 교체
    old_header = "| Config | Avg Act Bits | SVD Savings | FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ | vs Baseline |"
    if old_header in md:
        # 테이블 끝 찾기 (빈 줄 또는 ---로 구분)
        start = md.index(old_header)
        end   = md.find("\n\n", start)
        if end == -1:
            end = len(md)
        md = md[:start] + "\n".join(new_rows) + md[end:]
        open(md_path, "w").write(md)
        print(f"experiment.md 결과표 업데이트 완료: {md_path}")
    else:
        print("experiment.md 결과표 섹션을 찾지 못했습니다. 수동으로 확인하세요.")
