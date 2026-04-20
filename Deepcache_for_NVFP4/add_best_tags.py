"""
add_best_tags.py
sweep_all_results.csv에 best 관련 boolean 열을 추가한다.

추가 열 (FID — 낮을수록 좋음):
  best_fid_per_method   : 해당 method 내 전체 설정 중 FID 최소
  best_fid_none_s20     : method별 cache=none & steps=20 중 FID 최소 (공정 비교)
  best_fid_cache_s20    : method별 steps=20 전체(cache 포함) 중 FID 최소

추가 열 (IS — 높을수록 좋음):
  best_is_per_method    : 해당 method 내 전체 설정 중 IS 최대
  best_is_none_s20      : method별 cache=none & steps=20 중 IS 최대 (공정 비교)
  best_is_cache_s20     : method별 steps=20 전체(cache 포함) 중 IS 최대

기타:
  best_speed_per_method : method별 time_per_image_sec 최소
  pareto_s20            : steps=20 전체에서 FID-Time Pareto front에 속하는 점
"""
import csv

CSV_PATH = "results/sweep_all_results.csv"

rows = list(csv.DictReader(open(CSV_PATH)))

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

# ── 1. best_fid_per_method ────────────────────────────────────────────────────
# method별 최소 FID (steps/cache/range 무관)
best_fid_method = {}
for r in rows:
    fid = safe_float(r["fid"])
    if fid is None:
        continue
    m = r["quant_method"]
    if m not in best_fid_method or fid < best_fid_method[m]:
        best_fid_method[m] = fid

# ── 2. best_fid_none_s20 ─────────────────────────────────────────────────────
# method별 cache=none & steps=20 중 FID 최소
best_fid_none20 = {}
for r in rows:
    if r["cache_mode"] != "none" or r["num_steps"] != "20":
        continue
    fid = safe_float(r["fid"])
    if fid is None:
        continue
    m = r["quant_method"]
    if m not in best_fid_none20 or fid < best_fid_none20[m]:
        best_fid_none20[m] = fid

# ── 3. best_fid_cache_s20 ────────────────────────────────────────────────────
# method별 steps=20 전체(cache 포함) 중 FID 최소
best_fid_s20 = {}
for r in rows:
    if r["num_steps"] != "20":
        continue
    fid = safe_float(r["fid"])
    if fid is None:
        continue
    m = r["quant_method"]
    if m not in best_fid_s20 or fid < best_fid_s20[m]:
        best_fid_s20[m] = fid

# ── 4. best_speed_per_method ─────────────────────────────────────────────────
# method별 time_per_image_sec 최소
# cache_lora는 calibration warm-up artifact로 time이 비정상적으로 낮음 → 제외
# none / deepcache만 비교
best_speed_method = {}
for r in rows:
    if r["cache_mode"] == "cache_lora":   # artifact 제외
        continue
    t = safe_float(r["time_per_image_sec"])
    if t is None:
        continue
    m = r["quant_method"]
    if m not in best_speed_method or t < best_speed_method[m]:
        best_speed_method[m] = t

# ── 6. best_is_per_method ─────────────────────────────────────────────────────
# method별 IS 최대
best_is_method = {}
for r in rows:
    v = safe_float(r["is"])
    if v is None:
        continue
    m = r["quant_method"]
    if m not in best_is_method or v > best_is_method[m]:
        best_is_method[m] = v

# ── 7. best_clip_per_method ───────────────────────────────────────────────────
# method별 CLIP score 최대
best_clip_method = {}
for r in rows:
    v = safe_float(r["clip"])
    if v is None:
        continue
    m = r["quant_method"]
    if m not in best_clip_method or v > best_clip_method[m]:
        best_clip_method[m] = v

# ── 8. best_speed_in_config / best_fid_in_config ─────────────────────────────
# 같은 실험 설정(cache_mode, start, end, steps, lora_rank) 내에서
# method 간 speed/FID 비교

def config_key(r):
    """동일 실험 조건을 나타내는 key."""
    return (
        r["cache_mode"],
        r.get("deepcache_start", "") or "",
        r.get("deepcache_end",   "") or "",
        r["num_steps"],
        r.get("lora_rank", "")       or "",
    )

# config별 best time (cache_lora 포함 — 같은 config면 artifact도 동등하게 적용)
best_time_in_config = {}
best_fid_in_config  = {}
for r in rows:
    key = config_key(r)
    t   = safe_float(r["time_per_image_sec"])
    fid = safe_float(r["fid"])
    if t is not None:
        if key not in best_time_in_config or t < best_time_in_config[key]:
            best_time_in_config[key] = t
    if fid is not None:
        if key not in best_fid_in_config or fid < best_fid_in_config[key]:
            best_fid_in_config[key] = fid

# ── 5. pareto_s20 ────────────────────────────────────────────────────────────
# steps=20 전체에서 FID-Time Pareto front (minimize both)
s20_rows = [(i, r) for i, r in enumerate(rows)
            if r["num_steps"] == "20"
            and safe_float(r["fid"]) is not None
            and safe_float(r["time_per_image_sec"]) is not None]

pareto_idx = set()
for i, (idx_i, ri) in enumerate(s20_rows):
    fi = safe_float(ri["fid"])
    ti = safe_float(ri["time_per_image_sec"])
    dominated = False
    for j, (idx_j, rj) in enumerate(s20_rows):
        if i == j:
            continue
        fj = safe_float(rj["fid"])
        tj = safe_float(rj["time_per_image_sec"])
        if fj <= fi and tj <= ti and (fj < fi or tj < ti):
            dominated = True
            break
    if not dominated:
        pareto_idx.add(idx_i)

# ── Write updated CSV ─────────────────────────────────────────────────────────
NEW_FIELDS = [
    "best_fid_per_method",
    "best_fid_none_s20",
    "best_fid_cache_s20",
    "best_speed_per_method",
    "best_is_per_method",
    "best_clip_per_method",
    "pareto_s20",
    "best_speed_in_config",
    "best_fid_in_config",
]

original_fields = list(rows[0].keys())
all_fields = original_fields + NEW_FIELDS

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
    writer.writeheader()
    for i, r in enumerate(rows):
        m   = r["quant_method"]
        fid = safe_float(r["fid"])
        t   = safe_float(r["time_per_image_sec"])

        r["best_fid_per_method"] = (
            fid is not None and m in best_fid_method and fid == best_fid_method[m]
        )
        r["best_fid_none_s20"] = (
            r["cache_mode"] == "none"
            and r["num_steps"] == "20"
            and fid is not None
            and m in best_fid_none20
            and fid == best_fid_none20[m]
        )
        r["best_fid_cache_s20"] = (
            r["num_steps"] == "20"
            and fid is not None
            and m in best_fid_s20
            and fid == best_fid_s20[m]
        )
        r["best_speed_per_method"] = (
            r["cache_mode"] != "cache_lora"
            and t is not None
            and m in best_speed_method
            and t == best_speed_method[m]
        )
        is_ = safe_float(r["is"])
        r["best_is_per_method"] = (
            is_ is not None and m in best_is_method and is_ == best_is_method[m]
        )
        clip = safe_float(r["clip"])
        r["best_clip_per_method"] = (
            clip is not None and m in best_clip_method and clip == best_clip_method[m]
        )
        r["pareto_s20"] = i in pareto_idx

        key = config_key(r)
        r["best_speed_in_config"] = (
            t is not None
            and key in best_time_in_config
            and t == best_time_in_config[key]
        )
        r["best_fid_in_config"] = (
            fid is not None
            and key in best_fid_in_config
            and fid == best_fid_in_config[key]
        )

        writer.writerow(r)

# ── Summary ───────────────────────────────────────────────────────────────────
print("=== best_fid_per_method ===")
for r in rows:
    if r["best_fid_per_method"] == True or r["best_fid_per_method"] == "True":
        print(f"  {r['quant_method']:12s} {r['cache_mode']:12s} s{r['num_steps']:2s}  FID={safe_float(r['fid']):.1f}")

print("\n=== best_fid_none_s20 (공정 비교) ===")
for r in rows:
    if r["best_fid_none_s20"] == True or r["best_fid_none_s20"] == "True":
        print(f"  {r['quant_method']:12s}  FID={safe_float(r['fid']):.1f}  time={r['time_per_image_sec']}s")

print("\n=== best_fid_cache_s20 ===")
for r in rows:
    if r["best_fid_cache_s20"] == True or r["best_fid_cache_s20"] == "True":
        print(f"  {r['quant_method']:12s} {r['cache_mode']:12s}  FID={safe_float(r['fid']):.1f}")

print("\n=== best_speed_per_method ===")
for r in rows:
    if r["best_speed_per_method"] == True or r["best_speed_per_method"] == "True":
        print(f"  {r['quant_method']:12s} {r['cache_mode']:12s} s{r['num_steps']:2s}  time={safe_float(r['time_per_image_sec']):.2f}s")

print(f"\n=== pareto_s20 ({len(pareto_idx)} points) ===")
for i, r in enumerate(rows):
    if i in pareto_idx:
        print(f"  {r['quant_method']:12s} {r['cache_mode']:12s} s{r['num_steps']:2s}  FID={safe_float(r['fid']):.1f}  time={safe_float(r['time_per_image_sec']):.2f}s")

print("\n=== best_is_per_method ===")
for r in rows:
    if r["best_is_per_method"] == True or r["best_is_per_method"] == "True":
        print(f"  {r['quant_method']:12s} {r['cache_mode']:12s} s{r['num_steps']:2s}  IS={safe_float(r['is']):.4f}")

print("\n=== best_clip_per_method ===")
for r in rows:
    if r["best_clip_per_method"] == True or r["best_clip_per_method"] == "True":
        print(f"  {r['quant_method']:12s} {r['cache_mode']:12s} s{r['num_steps']:2s}  CLIP={safe_float(r['clip']):.2f}")

print("\n=== best_speed_in_config (설정별 가장 빠른 method) ===")
from collections import defaultdict
by_config_speed = defaultdict(list)
for r in rows:
    if r.get("best_speed_in_config") in (True, "True"):
        key = config_key(r)
        by_config_speed[key].append(r)
for key, rs in sorted(by_config_speed.items()):
    cache, cs, ce, steps, rank = key
    cfg = f"{cache} c{cs}-{ce} s{steps}" if cs else f"{cache} s{steps}"
    if rank:
        cfg += f" r{rank}"
    methods = ", ".join(r["quant_method"] for r in rs)
    t = safe_float(rs[0]["time_per_image_sec"])
    print(f"  [{cfg:35s}]  {methods:12s}  {t:.2f}s")

print("\n=== best_fid_in_config (설정별 FID 가장 좋은 method) ===")
by_config_fid = defaultdict(list)
for r in rows:
    if r.get("best_fid_in_config") in (True, "True"):
        key = config_key(r)
        by_config_fid[key].append(r)
for key, rs in sorted(by_config_fid.items()):
    cache, cs, ce, steps, rank = key
    cfg = f"{cache} c{cs}-{ce} s{steps}" if cs else f"{cache} s{steps}"
    if rank:
        cfg += f" r{rank}"
    methods = ", ".join(r["quant_method"] for r in rs)
    fid = safe_float(rs[0]["fid"])
    print(f"  [{cfg:35s}]  {methods:12s}  FID={fid:.1f}")

print(f"\nCSV updated: {CSV_PATH}  (columns: {len(all_fields)})")
