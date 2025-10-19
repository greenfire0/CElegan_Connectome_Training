# ===== Statistical tests + plot annotations for Fig. 4/5 style panels =====
# Requirements: scipy (for tests). No questions asked defaults:
# - Metric tested: final-generation "fitness" (food targets consumed)
# - Global test: Kruskal–Wallis across all groups
# - Pairwise tests: two-sided Mann–Whitney U for all pairs
# - Multiple comparisons: Benjamini–Hochberg (FDR=0.05)
# - Annotations: significance stars on ax1 (the fitness panel)

from typing import Dict, List, Tuple
import itertools
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception as e:
    raise RuntimeError("scipy is required for significance testing") from e

# --- helpers ---------------------------------------------------------------

def _final_scores_from_runs(runs: List[List[float]]) -> np.ndarray:
    """Extract final-generation scores per run (1 value per trajectory)."""
    if not runs:
        return np.asarray([], dtype=float)
    arr = np.asarray([np.asarray(r, dtype=float)[-1] for r in runs], dtype=float)
    return arr[~np.isnan(arr)]  # guard against NaNs

def _benjamini_hochberg(pvals: List[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Return (reject_mask, qvalues) for BH-FDR."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    # ensure monotonicity
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    qvals = np.empty_like(q_sorted)
    qvals[order] = q_sorted
    reject = qvals <= alpha
    return reject, qvals

def _stars(p: float) -> str:
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."

def _collect_groups_for_metric(metric_bucket: Dict[str, List[List[float]]],
                               label_map: Dict[str, str],
                               excluded: set) -> Dict[str, np.ndarray]:
    """metric_bucket is e.g. metrics['fitness'] which maps colour->list[runs]."""
    groups = {}
    for colour, runs in metric_bucket.items():
        if colour in excluded or not runs:
            continue
        name = label_map.get(colour, colour)
        scores = _final_scores_from_runs(runs)
        if scores.size:
            groups[name] = scores
    return groups

def _kruskal_and_pairwise(groups: Dict[str, np.ndarray], alpha: float = 0.05):
    """Kruskal–Wallis across all, then MWU for all pairs with BH correction."""
    names = list(groups.keys())
    data = [groups[n] for n in names]
    # Global test
    H, p_global = stats.kruskal(*data, nan_policy="omit")
    # Pairwise
    pairs, pvals = [], []
    for a, b in itertools.combinations(names, 2):
        x, y = groups[a], groups[b]
        # two-sided, independent samples, no tie correction beyond SciPy defaults
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        pairs.append((a, b))
        pvals.append(p)
    if pvals:
        reject, qvals = _benjamini_hochberg(pvals, alpha=alpha)
    else:
        reject, qvals = np.array([]), np.array([])
    results = {
        "global": {"H": float(H), "p": float(p_global)},
        "pairwise": [
            {"a": a, "b": b, "p": float(p), "q": float(q), "reject": bool(r)}
            for (a, b), p, q, r in zip(pairs, pvals, qvals, reject)
        ],
    }
    return results

def _annotate_ax_with_pairs(ax: plt.Axes,
                            group_positions: Dict[str, float],
                            pair_results: List[dict],
                            y_top_pad: float = 0.06):
    """Draw significance brackets between group means at the top of the panel."""
    # Determine top y and allocate stacked lines
    ymin, ymax = ax.get_ylim()
    height = ymax - ymin
    level = ymax + height * 0.02
    step = height * 0.05
    used_spans = []

    def next_level(span):
        nonlocal level
        # naive stacking to reduce overlaps
        while any(not (span[1] < s[0] or span[0] > s[1]) for s in used_spans):
            level += step
            used_spans.clear()
        used_spans.append(span)
        return level

    for r in pair_results:
        a, b, q = r["a"], r["b"], r["q"]
        mark = _stars(q)
        if mark == "n.s.":   # skip annotating non-significant by default
            continue
        xa, xb = group_positions[a], group_positions[b]
        lo, hi = min(xa, xb), max(xa, xb)
        y = next_level((lo, hi))

        # bracket
        ax.plot([xa, xa, xb, xb], [y, y + step * 0.5, y + step * 0.5, y], lw=1.5)
        ax.text((xa + xb) / 2.0, y + step * 0.6, mark, ha="center", va="bottom", fontsize=20)

    # expand ylim if needed
    ax.set_ylim(ymin, max(ymax, level + step * 1.2))

# --- execution on the current figure's fitness panel --------------------

# 1) build groups from your already-computed `metrics` and `label_map`
excluded = {"gold", "teal"}
groups = _collect_groups_for_metric(metrics["fitness"], label_map, excluded)

# 2) global and pairwise tests
stats_out = _kruskal_and_pairwise(groups, alpha=0.05)

# 3) choose which pairs to annotate on the plot (all significant by default)
#    Optionally, restrict to comparisons vs. a baseline:
#    baseline = "Evolutionary"; pairs_to_plot = [r for r in stats_out["pairwise"] if baseline in (r["a"], r["b"])]
pairs_to_plot = [r for r in stats_out["pairwise"] if r["reject"]]

# 4) compute x-positions of group means as they appear in the legend order
#    Here we infer by reading plotted lines on ax1; if you plot in a fixed order,
#    you can hard-code positions instead.
handles, labels = ax1.get_legend_handles_labels()
legend_names = [lab.replace(" (mean)", "") for lab in labels]
# Map visible legend order to x positions of the metric curves (we used a single mean line per group).
# For annotation, we just place groups at evenly spaced integer x positions.
group_positions = {}
xpos = 1
for name in legend_names:
    if name in groups and name not in group_positions:
        group_positions[name] = xpos
        xpos += 1

# draw invisible anchors along x for bracket placement
for name, xp in group_positions.items():
    ax1.plot([xp], [ax1.get_ylim()[1]*0.98], alpha=0)  # invisible point to stabilize axes

# 5) annotate
_annotate_ax_with_pairs(ax1, group_positions, pairs_to_plot)

# 6) console report (pasteable into methods/supplement)
print("\n=== Fitness (final-generation) — Kruskal–Wallis + MWU with BH-FDR ===")
print(f"Kruskal–Wallis: H = {stats_out['global']['H']:.3f}, p = {stats_out['global']['p']:.3g}")
for r in sorted(stats_out["pairwise"], key=lambda z: (z["q"], z["a"], z["b"])):
    mark = _stars(r["q"])
    print(f"{r['a']} vs {r['b']}:  p = {r['p']:.3g}, q = {r['q']:.3g}  [{mark}]")

# 7) (optional) save a TSV of the pairwise stats
with open("fig_fitness_stats.tsv", "w") as f:
    f.write("group_a\tgroup_b\tp_value\tq_value\treject\n")
    for r in stats_out["pairwise"]:
        f.write(f"{r['a']}\t{r['b']}\t{r['p']:.6g}\t{r['q']:.6g}\t{int(r['reject'])}\n")
