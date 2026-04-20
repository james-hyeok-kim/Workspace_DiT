"""
plot_pareto.py
Method별 개별 Pareto Front: FID vs Time/img (left) + IS vs Time/img (right)
Color = num_steps, Marker = cache_mode
모든 steps (5,10,15,20) 포함, method마다 별도 PNG 저장.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

CSV_PATH = "results/sweep_all_results.csv"
OUT_DIR   = "results"

ALL_METHODS = ["RTN", "SVDQUANT", "MRGPTQ", "FOUROVERSIX", "FP4DIT", "HQDIT", "SIXBIT"]

STEP_COLORS = {
    5:  "#aaaaaa",  # gray
    10: "#4fc3f7",  # sky blue
    15: "#ffa726",  # orange
    20: "#e53935",  # red
}

CACHE_MARKERS = {
    "none":       "o",
    "deepcache":  "s",
    "cache_lora": "^",
}

# ── Load all data ─────────────────────────────────────────────────────────────
all_rows = []
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        try:
            t     = float(r["time_per_image_sec"])
            fid   = float(r["fid"])
            is_   = float(r["is"])
            steps = int(r["num_steps"])
            if t <= 0 or fid <= 0:
                continue
        except (ValueError, KeyError):
            continue

        cache = r["cache_mode"]
        cs    = r.get("deepcache_start", "") or ""
        ce    = r.get("deepcache_end",   "") or ""
        rank  = r.get("lora_rank", "")       or ""

        if cache == "none":
            label = f"none s{steps}"
        elif cache == "deepcache":
            label = f"deepcache c{cs}-{ce} s{steps}"
        else:
            label = f"cache_lora r{rank} c{cs}-{ce} s{steps}"

        all_rows.append({
            "method":     r["quant_method"],
            "cache_mode": cache,
            "steps":      steps,
            "fid":        fid,
            "is":         is_,
            "time":       t,
            "label":      label,
        })

# ── Pareto front helper ───────────────────────────────────────────────────────
def pareto_front(xs, ys, minimize_x=True, minimize_y=True):
    dominated = [False] * len(xs)
    for i in range(len(xs)):
        for j in range(len(xs)):
            if i == j:
                continue
            x_dom = (xs[j] <= xs[i]) if minimize_x else (xs[j] >= xs[i])
            y_dom = (ys[j] <= ys[i]) if minimize_y else (ys[j] >= ys[i])
            x_str = (xs[j] < xs[i])  if minimize_x else (xs[j] > xs[i])
            y_str = (ys[j] < ys[i])  if minimize_y else (ys[j] > ys[i])
            if x_dom and y_dom and (x_str or y_str):
                dominated[i] = True
                break
    return [i for i, d in enumerate(dominated) if not d]

# ── Label placement: staggered offsets ───────────────────────────────────────
OFFSETS = [
    ( 8,  8), (-8,  8), ( 8, -8), (-8, -8),
    (14,  0), (-14,  0), ( 0, 13), ( 0,-13),
    (16,  8), (-16,  8), (16, -8), (-16, -8),
    (20,  0), (-20,  0), ( 0, 18), ( 0,-18),
]

def place_labels(ax, points):
    """points: list of (x, y, label, color)"""
    for idx, (x, y, lbl, color) in enumerate(points):
        ox, oy = OFFSETS[idx % len(OFFSETS)]
        ax.annotate(
            lbl, xy=(x, y),
            xytext=(ox, oy), textcoords="offset points",
            fontsize=6, color=color, fontweight="bold",
            ha="center", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.7),
        )

# ── Per-method plots ──────────────────────────────────────────────────────────
for method in ALL_METHODS:
    rows = [r for r in all_rows if r["method"] == method]
    if not rows:
        print(f"  No data for {method}, skipping.")
        continue

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f"Pareto Front — {method}  (steps: 5/10/15/20, all cache modes)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plot_specs = [
        (axes[0], "fid", "FID ↓",  True,  "FID vs Time/img",  True),
        (axes[1], "is",  "IS ↑",   False, "IS vs Time/img",   False),
    ]

    for ax, y_key, y_label, minimize_y, title, log_y in plot_specs:
        xs = [r["time"] for r in rows]
        ys = [r[y_key]  for r in rows]

        # ── Scatter all points ────────────────────────────────────────────
        for r in rows:
            color  = STEP_COLORS.get(r["steps"], "#888888")
            marker = CACHE_MARKERS.get(r["cache_mode"], "o")
            ax.scatter(
                r["time"], r[y_key],
                color=color, marker=marker,
                s=100, alpha=0.80,
                linewidths=0.8, edgecolors="white",
                zorder=3,
            )

        # ── Pareto front ──────────────────────────────────────────────────
        pidx = pareto_front(xs, ys, minimize_x=True, minimize_y=minimize_y)
        px = [xs[i] for i in pidx]
        py = [ys[i] for i in pidx]
        order = np.argsort(px)
        px_s = [px[i] for i in order]
        py_s = [py[i] for i in order]

        # draw line (visible even for short segments)
        ax.plot(px_s, py_s, "k--", lw=2.0, alpha=0.7, zorder=4)

        # highlight Pareto points with larger marker + black edge
        for i in pidx:
            r = rows[i]
            color  = STEP_COLORS.get(r["steps"], "#888888")
            marker = CACHE_MARKERS.get(r["cache_mode"], "o")
            ax.scatter(
                r["time"], r[y_key],
                color=color, marker=marker,
                s=220, alpha=1.0,
                linewidths=1.5, edgecolors="black",
                zorder=5,
            )

        # ── Labels on ALL points ──────────────────────────────────────────
        label_pts = []
        for r in rows:
            color = STEP_COLORS.get(r["steps"], "#888888")
            label_pts.append((r["time"], r[y_key], r["label"], color))
        place_labels(ax, label_pts)

        # ── Axes formatting ───────────────────────────────────────────────
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        # Auto x range from data + 20% margin in log space
        t_min = min(r["time"] for r in rows)
        t_max = max(r["time"] for r in rows)
        log_min = np.log10(t_min)
        log_max = np.log10(t_max)
        margin  = (log_max - log_min) * 0.12
        ax.set_xlim(10 ** (log_min - margin), 10 ** (log_max + margin))

        # Pick tick candidates that fall inside the data range
        all_candidates = [0.4, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 40]
        x_ticks = [v for v in all_candidates if t_min * 0.7 <= v <= t_max * 1.4]
        if not x_ticks:
            x_ticks = [t_min, t_max]
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda val, _: f"{val:g}s")
        )
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", labelsize=9, rotation=35)
        ax.tick_params(axis="y", labelsize=9)

        ax.set_xlabel("Time / image (s)  [log scale — faster = left]",
                      fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.12)

    # ── Legends ──────────────────────────────────────────────────────────────
    step_handles = [
        mpatches.Patch(color=c, label=f"steps={s}")
        for s, c in sorted(STEP_COLORS.items())
    ]
    cache_handles = [
        mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                      markersize=8, label=cm)
        for cm, mk in CACHE_MARKERS.items()
    ]
    pareto_handle = mlines.Line2D([], [], color="black", linestyle="--",
                                   linewidth=2.0, label="Pareto front")
    pareto_pt     = mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                                   markersize=10, markeredgecolor="black",
                                   markeredgewidth=1.5, label="Pareto point")

    axes[0].legend(handles=step_handles, title="Steps", loc="upper left",
                   fontsize=8, title_fontsize=9, framealpha=0.85)
    axes[1].legend(handles=cache_handles + [pareto_handle, pareto_pt],
                   title="Cache mode / Pareto", loc="upper left",
                   fontsize=8, title_fontsize=9, framealpha=0.85)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"pareto_{method}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  ({len(rows)} points, {len(pidx)} Pareto)")

print("Done.")
