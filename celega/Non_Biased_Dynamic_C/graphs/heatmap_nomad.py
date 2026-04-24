# nomad_change_graph_vis.py
# ======================================================================
# EXACT ConnectomeViewer layout as in the video, but only show edges
# whose change-frequency > 2 (i.e., count >= 3). No background edges.
# - Color: blue→red linear (low→high). Unused here for counts < 3.
# - Width: linear in frequency; PURE panel thinner overall.
#
# Extended analyses (writes TSVs):
#   1) Edge-class enrichment with degree-aware permutation null:
#        - edge_enrichment_hybrid.tsv / edge_enrichment_pure.tsv
#   2) Sign/gain shifts by edge class and hemisphere:
#        - sign_gain_shifts_hybrid.tsv / sign_gain_shifts_pure.tsv
#   3) Sensory→motor path-gain changes (pre vs post, up to MAX_HOPS):
#        - path_gain_changes_hybrid.tsv / path_gain_changes_pure.tsv
# ======================================================================

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from util.write_read_txt import read_arrays_from_csv_pandas
try:
    from Worm_Env.connectome2 import WormConnectome
except ImportError:
    from Worm_Env.connectome import WormConnectome

from Worm_Env.weight_dict import all_neuron_names
from graphs.connectome_graph import ConnectomeViewer

# ------------------------------- config --------------------------------
DATA_DIR_NAME     = "data_new_pentagon"
ATOL              = 1e-6
MIN_COUNT         = 3              # only show edges changed >= 3 times  ( > 2 )
MAX_FILES         = None
OUT_SVG           = "nomad_change_graphs_thresh.svg"
OUT_PNG           = "nomad_change_graphs_thresh.png"
TOPK              = 200
BASELINE_ENV_VAR  = "BASELINE_CSV"
BASELINE_DEFAULT  = "original_genome.csv"

# ConnectomeViewer params (match video)
CV_LAYOUT     = "kamada_groups"
CV_SPREAD     = 1
CV_PULSE_SIZE = 3.0
CV_GROUP_GAP  = 0.5
CV_COLOR_MODE = "energy"

# Changed-edge styling
EDGE_MIN_W_H  = 0.6
EDGE_MAX_W_H  = 3.8
EDGE_MIN_W_P  = 0.4
EDGE_MAX_W_P  = 2.2
CHANGED_EDGE_ALPHA = 0.85

# Colormap (blue→red)
CMAP_NAME     = "coolwarm"
TITLE_L       = f"NOMAD Hybrid — edges changed ≥ {MIN_COUNT} times (linear)"
TITLE_R       = f"Pure NOMAD — edges changed ≥ {MIN_COUNT} times (linear)"

# --------- analysis params (degree-aware null & path gains) ------------
N_PERM        = 200      # degree-aware permutations
DEG_BINS      = 5        # out/in-degree quantile bins
MAX_HOPS      = 4        # path-gain horizon

# Output TSVs
EDGE_ENRICH_HYB_TSV = "edge_enrichment_hybrid.tsv"
EDGE_ENRICH_PURE_TSV= "edge_enrichment_pure.tsv"
SIGN_GAIN_HYB_TSV   = "sign_gain_shifts_hybrid.tsv"
SIGN_GAIN_PURE_TSV  = "sign_gain_shifts_pure.tsv"
PATH_GAIN_HYB_TSV   = "path_gain_changes_hybrid.tsv"
PATH_GAIN_PURE_TSV  = "path_gain_changes_pure.tsv"
# -----------------------------------------------------------------------


def _repo_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _data_dir() -> str:
    return os.path.join(_repo_dir(), DATA_DIR_NAME)


def _load_csv_vector(csv_path: str) -> np.ndarray:
    rows = read_arrays_from_csv_pandas(csv_path)
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")
    return np.asarray(rows[0], dtype=float)


def _try_get_global_baseline() -> np.ndarray | None:
    data_dir = _data_dir()
    env_choice = os.environ.get(BASELINE_ENV_VAR, "").strip()
    if env_choice:
        cand = os.path.join(data_dir, env_choice) if not os.path.isabs(env_choice) else env_choice
        if os.path.isfile(cand):
            return _load_csv_vector(cand)
    default_path = os.path.join(data_dir, BASELINE_DEFAULT)
    if os.path.isfile(default_path):
        return _load_csv_vector(default_path)
    return None


def _edge_index_map(baseline_vec: np.ndarray) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]], List[str]]:
    wc = WormConnectome(weight_matrix=baseline_vec, all_neuron_names=all_neuron_names)

    if hasattr(wc, "_edge_ptr") and hasattr(wc, "_edge_w"):
        ptr = np.asarray(wc._edge_ptr)
        if ptr.ndim != 2 or ptr.shape[1] < 2:
            raise RuntimeError("Unexpected shape for wc._edge_ptr")
        ij = ptr[:, -2:].astype(int)
        if ij.shape[0] != baseline_vec.shape[0]:
            L = min(ij.shape[0], baseline_vec.shape[0])
            ij = ij[:L]
        idx_seq = np.arange(ij.shape[0], dtype=int)
        k2ij = {int(k): (int(i), int(j)) for k, (i, j) in enumerate(ij)}
        return idx_seq, k2ij, list(all_neuron_names)

    if hasattr(wc, "W"):
        W = np.asarray(wc.W)
        src, dst = np.nonzero(W != 0)
        E = len(src)
        if E != baseline_vec.shape[0]:
            L = min(E, baseline_vec.shape[0])
            src, dst = src[:L], dst[:L]
        idx_seq = np.arange(len(src), dtype=int)
        k2ij = {int(k): (int(i), int(j)) for k, (i, j) in enumerate(zip(src, dst))}
        return idx_seq, k2ij, list(all_neuron_names)

    raise RuntimeError("Cannot derive (i,j) mapping")


def _classify_files(data_dir: str) -> Dict[str, List[str]]:
    buckets = {"hybrid": [], "pure": []}
    for fn in os.listdir(data_dir):
        if not fn.lower().endswith(".csv"):
            continue
        lower = fn.lower()
        if "es_worms" in lower:
            continue
        p = os.path.join(data_dir, fn)
        if "hybrid" in lower:
            buckets["hybrid"].append(p)
        if "pure_nomad" in lower or "pure" in lower:
            buckets["pure"].append(p)
    if MAX_FILES:
        for k in buckets:
            buckets[k] = sorted(buckets[k])[:MAX_FILES]
    return buckets


def _accumulate_counts(
    files: List[str],
    k2ij: Dict[int, Tuple[int, int]],
    baseline_global: np.ndarray | None,
    atol: float
) -> Tuple[np.ndarray, int, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Returns: counts matrix, number of runs, and a list of (g0, gT) vectors per file (for downstream analyses).
    """
    N = len(all_neuron_names)
    counts = np.zeros((N, N), dtype=int)
    n_runs = 0
    pre_post_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    for path in files:
        rows = read_arrays_from_csv_pandas(path)
        if not rows:
            continue

        arr = np.asarray(rows, dtype=float)
        g0 = baseline_global if baseline_global is not None else arr[0]
        gT = arr[-1]

        L = min(len(g0), len(gT), len(k2ij))
        if L == 0:
            continue

        delta = np.abs(gT[:L] - g0[:L])
        changed_idx = np.nonzero(delta > atol)[0]

        for k in changed_idx:
            i, j = k2ij[int(k)]
            if i < N and j < N:
                counts[i, j] += 1

        pre_post_pairs.append((g0[:L].copy(), gT[:L].copy()))
        n_runs += 1

    return counts, n_runs, pre_post_pairs


def _render_viewer_into_axes(viewer: ConnectomeViewer, ax: plt.Axes):
    viewer.fig = ax.figure
    viewer.ax = ax
    ax.set_axis_off()
    baseline_color = viewer.inact_col if viewer.color_mode == "binary" else viewer.cmap(viewer.norm(0.0))
    viewer._draw_graph(baseline_color)
    viewer._add_side_labels()


def _overlay_edge_counts(ax: plt.Axes,
                         pos_by_name: Dict[str, np.ndarray],
                         names: List[str],
                         counts: np.ndarray,
                         norm: mcolors.Normalize,
                         cmap,
                         width_min: float,
                         width_max: float) -> None:
    vmin, vmax = norm.vmin, norm.vmax
    ys, xs = np.nonzero(counts >= MIN_COUNT)   # threshold here
    if ys.size == 0:
        return
    idx2name = list(names)
    denom = float(max(vmax - vmin, 1.0))
    for i, j in zip(ys, xs):
        c = int(counts[i, j])
        ni, nj = idx2name[i], idx2name[j]
        if ni not in pos_by_name or nj not in pos_by_name:
            continue
        (x0, y0) = pos_by_name[ni]
        (x1, y1) = pos_by_name[nj]
        width = width_min + (width_max - width_min) * ((c - vmin) / denom)
        ax.plot([x0, x1], [y0, y1],
                color=cmap(norm(c)), linewidth=width,
                solid_capstyle="round", alpha=CHANGED_EDGE_ALPHA, zorder=3)


def _filtered_vmax(*mats: np.ndarray) -> int:
    vals = []
    for M in mats:
        sel = M[M >= MIN_COUNT]
        if sel.size:
            vals.append(int(sel.max()))
    return max(vals) if vals else 1


def _plot_with_connectome_viewer(h_counts: np.ndarray,
                                 p_counts: np.ndarray,
                                 baseline_vec: np.ndarray,
                                 names: List[str]) -> Tuple[ConnectomeViewer, ConnectomeViewer]:
    wormL = WormConnectome(weight_matrix=baseline_vec, all_neuron_names=names)
    wormR = WormConnectome(weight_matrix=baseline_vec, all_neuron_names=names)

    viewerL = ConnectomeViewer(
        wormL, layout=CV_LAYOUT, spread=CV_SPREAD,
        pulse_size=CV_PULSE_SIZE, group_gap=CV_GROUP_GAP,
        color_mode=CV_COLOR_MODE
    )
    viewerR = ConnectomeViewer(
        wormR, layout=CV_LAYOUT, spread=CV_SPREAD,
        pulse_size=CV_PULSE_SIZE, group_gap=CV_GROUP_GAP,
        color_mode=CV_COLOR_MODE
    )

    if hasattr(viewerL, "step"): viewerL.step()
    if hasattr(viewerR, "step"): viewerR.step()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    _render_viewer_into_axes(viewerL, axes[0])
    _render_viewer_into_axes(viewerR, axes[1])

    axes[0].set_title(TITLE_L, fontsize=14, pad=10)
    axes[1].set_title(TITLE_R, fontsize=14, pad=10)

    # positions by name
    posL = {n: np.array(viewerL.pos[n], float) for n in viewerL.G.nodes()}
    posR = {n: np.array(viewerR.pos[n], float) for n in viewerR.G.nodes()}

    # color scale uses only filtered edges (>= MIN_COUNT)
    vmax = _filtered_vmax(h_counts, p_counts)
    norm = mcolors.Normalize(vmin=MIN_COUNT, vmax=vmax)
    cmap = cm.get_cmap(CMAP_NAME)

    # overlay (no gray background; only edges meeting threshold are drawn)
    _overlay_edge_counts(axes[0], posL, names, h_counts, norm, cmap, EDGE_MIN_W_H, EDGE_MAX_W_H)
    _overlay_edge_counts(axes[1], posR, names, p_counts, norm, cmap, EDGE_MIN_W_P, EDGE_MAX_W_P)

    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label(f"changed count (linear, shown ≥ {MIN_COUNT})")

    fig.savefig(OUT_SVG, dpi=300)
    fig.savefig(OUT_PNG, dpi=300)

    return viewerL, viewerR


# ========================= helper utilities ============================

def _vector_to_matrix(vec: np.ndarray, k2ij: Dict[int, Tuple[int, int]], N: int) -> np.ndarray:
    W = np.zeros((N, N), dtype=float)
    L = min(len(vec), len(k2ij))
    for k in range(L):
        i, j = k2ij[int(k)]
        if i < N and j < N:
            W[i, j] = vec[k]
    return W


def _binary_deg(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    B = (W != 0).astype(int)
    outdeg = B.sum(axis=1)
    indeg  = B.sum(axis=0)
    return outdeg, indeg


def _quantile_bins(arr: np.ndarray, bins: int) -> np.ndarray:
    if arr.size == 0:
        return arr
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(arr, qs))
    # guard: all equal
    if edges.size == 1:
        return np.zeros_like(arr, dtype=int)
    idx = np.digitize(arr, edges[1:-1], right=True)
    return idx.astype(int)


def _hemisphere(name: str) -> str:
    if len(name) and name[-1] in ("L", "R"):
        return name[-1]
    return "X"


def _get_group_map(viewer: ConnectomeViewer, names: List[str]) -> Dict[str, str]:
    """
    Try to fetch node→group labels from ConnectomeViewer; fallback to simple heuristics.
    """
    # 1) Try common attributes in ConnectomeViewer
    group_map: Dict[str, str] = {}
    if hasattr(viewer, "groups") and isinstance(viewer.groups, dict):
        for g, lst in viewer.groups.items():
            for n in lst:
                group_map[n] = str(g)

    # 2) If not populated, try node attributes
    if not group_map and hasattr(viewer, "G"):
        G = viewer.G
        for n in G.nodes():
            data = G.nodes[n]
            for key in ("group", "type", "class", "layer"):
                if key in data:
                    group_map[n] = str(data[key])
                    break

    # 3) Fallback heuristics: identify sensory/muscle based on known prefixes from your paper
    if not group_map:
        sensory_prefixes = ("FLP", "ASH", "IL1", "OLQ", "ADF", "ASG", "ASI", "ASJ")
        for n in names:
            if n.startswith(sensory_prefixes):
                group_map[n] = "sensory"
            elif "mus" in n.lower() or n.startswith(("MD", "MV")):
                group_map[n] = "muscle"
            else:
                group_map[n] = "neuron"
    # Normalize a few labels
    norm_map = {}
    for n, g in group_map.items():
        g_low = g.lower()
        if "sens" in g_low:
            norm_map[n] = "sensory"
        elif "mus" in g_low:
            norm_map[n] = "muscle"
        elif "motor" in g_low or "moto" in g_low:
            norm_map[n] = "motor"
        elif "inter" in g_low:
            norm_map[n] = "interneuron"
        else:
            norm_map[n] = g  # as-is
    return norm_map


def _sensory_indices(group_map: Dict[str, str], names: List[str]) -> List[int]:
    return [i for i, n in enumerate(names) if group_map.get(n, "").lower().startswith("sens")]


def _muscle_indices(group_map: Dict[str, str], names: List[str]) -> List[int]:
    return [i for i, n in enumerate(names) if "muscle" in group_map.get(n, "").lower()]


def _edge_list_from_k2ij(k2ij: Dict[int, Tuple[int, int]], N: int) -> np.ndarray:
    E = []
    for k, (i, j) in k2ij.items():
        if i < N and j < N:
            E.append((i, j))
    return np.asarray(E, dtype=int)


# =================== (1) Edge-class enrichment =========================

def _edge_class_enrichment(counts: np.ndarray,
                           group_map: Dict[str, str],
                           names: List[str],
                           W_base: np.ndarray,
                           k2ij: Dict[int, Tuple[int, int]],
                           n_perm: int = N_PERM,
                           deg_bins: int = DEG_BINS) -> pd.DataFrame:
    """
    Degree-aware permutation null:
      - compute out/in degree bins on the baseline graph
      - changed set = union of edges with counts > 0
      - for each permutation, select |changed| edges by matching (outbin(source), inbin(target))
        to the changed set's bin pairs (with replacement if needed)
      - summarize observed vs. expected counts per (src_group, dst_group)
    """
    N = len(names)
    Eall = _edge_list_from_k2ij(k2ij, N)
    if Eall.size == 0:
        return pd.DataFrame()

    outdeg, indeg = _binary_deg(W_base)
    outbin = _quantile_bins(outdeg, deg_bins)
    inbin  = _quantile_bins(indeg,  deg_bins)

    # union of changed edges
    changed_mask = counts > 0
    ys, xs = np.nonzero(changed_mask)
    changed_edges = np.vstack([ys, xs]).T
    M = changed_edges.shape[0]
    if M == 0:
        return pd.DataFrame()

    # map edges -> degree bin pairs
    # Build index per bin pair for sampling
    bin_pair_to_edges: Dict[Tuple[int, int], np.ndarray] = {}
    for (i, j) in Eall:
        bp = (int(outbin[i]), int(inbin[j]))
        if bp not in bin_pair_to_edges:
            bin_pair_to_edges[bp] = []
        bin_pair_to_edges[bp].append((i, j))
    for bp in list(bin_pair_to_edges.keys()):
        bin_pair_to_edges[bp] = np.asarray(bin_pair_to_edges[bp], dtype=int)

    # observed counts by edge class
    def lab(i: int) -> str:
        return group_map.get(names[i], "unknown")

    def edgeclass_counts(edge_list: np.ndarray) -> Dict[Tuple[str, str], int]:
        d: Dict[Tuple[str, str], int] = {}
        for (i, j) in edge_list:
            key = (lab(i), lab(j))
            d[key] = d.get(key, 0) + 1
        return d

    obs_counts = edgeclass_counts(changed_edges)

    # degree-aware permutations
    # replicate the changed set's bin pairs
    changed_bins = [(int(outbin[i]), int(inbin[j])) for (i, j) in changed_edges]
    class_keys = sorted(set(obs_counts.keys()))
    # Track per-class counts across permutations
    perm_mat = {ck: np.zeros(n_perm, dtype=float) for ck in class_keys}

    rng = np.random.default_rng(12345)
    for p in range(n_perm):
        sampled_edges = []
        for bp in changed_bins:
            pool = bin_pair_to_edges.get(bp, None)
            if pool is None or pool.size == 0:
                # fallback: sample any edge
                idx = rng.integers(0, Eall.shape[0])
                sampled_edges.append(Eall[idx])
            else:
                idx = rng.integers(0, pool.shape[0])
                sampled_edges.append(pool[idx])
        sampled_edges = np.asarray(sampled_edges, dtype=int)
        cc = edgeclass_counts(sampled_edges)
        for ck in class_keys:
            perm_mat[ck][p] = float(cc.get(ck, 0))

    # summarize
    rows = []
    for ck in sorted(class_keys):
        arr = perm_mat[ck]
        mu, sd = arr.mean(), arr.std(ddof=1) if arr.size > 1 else 0.0
        obs = float(obs_counts.get(ck, 0))
        z = (obs - mu) / sd if sd > 1e-12 else np.nan
        # two-sided empirical p
        more_extreme = np.sum(np.abs(arr - mu) >= np.abs(obs - mu))
        p_emp = (1 + more_extreme) / (1 + n_perm)
        rows.append({
            "src_group": ck[0],
            "dst_group": ck[1],
            "observed": int(obs),
            "expected_mean": mu,
            "expected_std": sd,
            "z": z,
            "emp_p": p_emp,
            "n_changed_edges": int(M)
        })
    return pd.DataFrame(rows)


# =========== (2) Sign / gain shifts by class and hemisphere ============

def _aggregate_sign_gain(pre_post: List[Tuple[np.ndarray, np.ndarray]],
                         k2ij: Dict[int, Tuple[int, int]],
                         names: List[str],
                         group_map: Dict[str, str],
                         atol: float) -> pd.DataFrame:
    """
    For edges with |delta| > atol across runs:
      - summarize Δw = (post - pre) per (src_group, dst_group) and hemisphere relation
      - also report sign flips (pre sign vs post sign)
    """
    N = len(names)
    hemi = [_hemisphere(n) for n in names]

    # accumulators keyed by (src_group, dst_group, hemi_rel)
    buckets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def key_for(i: int, j: int) -> Tuple[str, str, str]:
        srcg, dstg = group_map.get(names[i], "unknown"), group_map.get(names[j], "unknown")
        hrel = f"{hemi[i]}→{hemi[j]}"
        return (srcg, dstg, hrel)

    for (g0, gT) in pre_post:
        L = min(len(g0), len(gT), len(k2ij))
        for k in range(L):
            di = gT[k] - g0[k]
            if abs(di) <= atol:
                continue
            i, j = k2ij[int(k)]
            if i >= N or j >= N:
                continue
            kk = key_for(i, j)
            b = buckets.get(kk)
            if b is None:
                b = {
                    "deltas": [],
                    "pre_sign": [],
                    "post_sign": [],
                }
                buckets[kk] = b
            b["deltas"].append(di)
            b["pre_sign"].append(np.sign(g0[k]))
            b["post_sign"].append(np.sign(gT[k]))

    rows = []
    for (srcg, dstg, hrel), b in buckets.items():
        d = np.asarray(b["deltas"], float)
        ps, qs = np.asarray(b["pre_sign"], float), np.asarray(b["post_sign"], float)
        flips = np.sum(np.sign(ps) != np.sign(qs))
        pos_to_neg = np.sum((ps > 0.0) & (qs < 0.0))
        neg_to_pos = np.sum((ps < 0.0) & (qs > 0.0))
        rows.append({
            "src_group": srcg,
            "dst_group": dstg,
            "hemi": hrel,
            "n_edges": int(d.size),
            "delta_mean": float(d.mean()) if d.size else 0.0,
            "delta_median": float(np.median(d)) if d.size else 0.0,
            "frac_delta_positive": float(np.mean(d > 0.0)) if d.size else np.nan,
            "sign_flip_frac": float(flips / d.size) if d.size else np.nan,
            "pos_to_neg_count": int(pos_to_neg),
            "neg_to_pos_count": int(neg_to_pos),
            "pos_to_neg_frac": float(pos_to_neg / d.size) if d.size else np.nan,
            "neg_to_pos_frac": float(neg_to_pos / d.size) if d.size else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "src_group","dst_group","hemi","n_edges",
            "delta_mean","delta_median","frac_delta_positive","sign_flip_frac",
            "pos_to_neg_count","neg_to_pos_count","pos_to_neg_frac","neg_to_pos_frac"
        ])
    # Sort by magnitude and count
    df = pd.DataFrame(rows)
    return df.sort_values(["src_group","dst_group","hemi"]).reset_index(drop=True)


# ===== (3) Sensory→motor path gain changes (pre vs post, up to H) ======

def _path_gain_matrix(W: np.ndarray, H: int) -> np.ndarray:
    """
    Sum of powers: sum_{h=1..H} W^h
    (Directed weighted paths; no normalization.)
    """
    N = W.shape[0]
    S = np.zeros((N, N), dtype=float)
    P = W.copy()
    for _ in range(H):
        S += P
        P = P @ W
    return S


def _sensory_motor_path_changes(pre_post: List[Tuple[np.ndarray, np.ndarray]],
                                k2ij: Dict[int, Tuple[int, int]],
                                names: List[str],
                                group_map: Dict[str, str],
                                H: int = MAX_HOPS,
                                topk: int = TOPK) -> pd.DataFrame:
    N = len(names)
    sens_idx = _sensory_indices(group_map, names)
    mus_idx  = _muscle_indices(group_map, names)
    if not sens_idx or not mus_idx:
        return pd.DataFrame(columns=["sensory","muscle","gain_pre","gain_post","gain_delta","abs_delta_rank"])

    S_pre_acc = None
    S_post_acc = None
    for (g0, gT) in pre_post:
        L = min(len(g0), len(gT), len(k2ij))
        W0 = _vector_to_matrix(g0[:L], k2ij, N)
        WT = _vector_to_matrix(gT[:L], k2ij, N)
        S0 = _path_gain_matrix(W0, H)
        ST = _path_gain_matrix(WT, H)
        if S_pre_acc is None:
            S_pre_acc = S0
            S_post_acc = ST
        else:
            S_pre_acc += S0
            S_post_acc += ST

    if S_pre_acc is None or S_post_acc is None:
        return pd.DataFrame(columns=["sensory","muscle","gain_pre","gain_post","gain_delta","abs_delta_rank"])

    # average across runs
    n_runs = len(pre_post)
    S_pre = S_pre_acc / max(n_runs, 1)
    S_post= S_post_acc / max(n_runs, 1)
    Delta = S_post - S_pre

    rows = []
    for s in sens_idx:
        for m in mus_idx:
            rows.append({
                "sensory": names[s],
                "muscle": names[m],
                "gain_pre": float(S_pre[s, m]),
                "gain_post": float(S_post[s, m]),
                "gain_delta": float(Delta[s, m]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["abs_delta"] = df["gain_delta"].abs()
    df = df.sort_values("abs_delta", ascending=False).drop(columns=["abs_delta"]).reset_index(drop=True)
    if topk is not None:
        df = df.head(topk).copy()
    # rank for convenience
    df["abs_delta_rank"] = np.arange(1, len(df) + 1)
    return df


# ================================ main =================================

def _emit_topk(counts: np.ndarray, names: List[str], path_tsv: str, k: int) -> None:
    # still emit top-K but filtered to count >= MIN_COUNT
    flat = counts.ravel()
    idx_sorted = np.argsort(flat)[::-1]
    out_lines = ["count\tsrc_idx\tdst_idx\tsrc_name\tdst_name"]
    N = counts.shape[0]
    written = 0
    for lin in idx_sorted:
        c = int(flat[lin])
        if c < MIN_COUNT:
            break
        i = lin // N
        j = lin % N
        out_lines.append(f"{c}\t{i}\t{j}\t{names[i]}\t{names[j]}")
        written += 1
        if written >= k:
            break
    with open(path_tsv, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

def _print_pos_neg_summary(pre_post: List[Tuple[np.ndarray, np.ndarray]], label: str, atol: float) -> None:
    """Print total counts of positive vs negative Δw across all runs (|Δw|>atol)."""
    pos = neg = changed = 0
    for (g0, gT) in pre_post:
        d = gT - g0
        pos += int(np.sum(d >  atol))
        neg += int(np.sum(d < -atol))
        changed += int(np.sum(np.abs(d) > atol))
    print(f"[{label}] changed edges (across runs): {changed:,} | Δw>0: {pos:,} | Δw<0: {neg:,}")
# ---------------- overall sign-flip percentage ----------------
def _print_overall_sign_flip(pre_post, k2ij, atol, label: str) -> None:
    """
    Across all runs, consider only edges with |Δw| > atol and
    report how many changed sign (np.sign(pre) != np.sign(post)).
    """
    total_changed = 0
    flips = 0
    for (g0, gT) in pre_post:
        L = min(len(g0), len(gT), len(k2ij))
        if L == 0:
            continue
        d = gT[:L] - g0[:L]
        sel = np.where(np.abs(d) > atol)[0]
        if sel.size == 0:
            continue
        total_changed += sel.size
        s0 = np.sign(g0[:L][sel])
        s1 = np.sign(gT[:L][sel])
        flips += int(np.sum(s0 != s1))
    pct = 100.0 * flips / max(total_changed, 1)
    print(f"[{label}] sign flips: {flips}/{total_changed} = {pct:.2f}%")

def _print_directional_sign_flips(pre_post, k2ij, atol, label: str) -> None:
    """
    Across all runs, consider only edges with |Δw| > atol and report
    directional sign flips as percentages of all changed edges.
    """
    total_changed = 0
    pos_to_neg = 0
    neg_to_pos = 0
    for (g0, gT) in pre_post:
        L = min(len(g0), len(gT), len(k2ij))
        if L == 0:
            continue
        d = gT[:L] - g0[:L]
        sel = np.where(np.abs(d) > atol)[0]
        if sel.size == 0:
            continue
        total_changed += int(sel.size)
        pre = g0[:L][sel]
        post = gT[:L][sel]
        pos_to_neg += int(np.sum((pre > 0.0) & (post < 0.0)))
        neg_to_pos += int(np.sum((pre < 0.0) & (post > 0.0)))

    denom = max(total_changed, 1)
    pos_to_neg_pct = 100.0 * pos_to_neg / denom
    neg_to_pos_pct = 100.0 * neg_to_pos / denom
    print(
        f"[{label}] directional sign flips (% of changed edges): "
        f"positive->negative {pos_to_neg}/{total_changed} = {pos_to_neg_pct:.2f}% | "
        f"negative->positive {neg_to_pos}/{total_changed} = {neg_to_pos_pct:.2f}%"
    )

def main():
    data_dir = _data_dir()
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Missing data dir: {data_dir}")

    baseline_global = _try_get_global_baseline()
    if baseline_global is None:
        all_csvs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
        all_csvs = [p for p in all_csvs if os.path.isfile(p)]
        if not all_csvs:
            raise SystemExit("No CSV files found to infer mapping.")
        baseline_global = _load_csv_vector(all_csvs[0])

    idx_seq, k2ij, names = _edge_index_map(baseline_global)

    buckets = _classify_files(data_dir)
    hybrid_files = sorted(buckets["hybrid"])
    pure_files   = sorted(buckets["pure"])

    # accumulate counts AND collect per-run (pre,post) vectors
    h_counts, h_runs, h_pairs = _accumulate_counts(hybrid_files, k2ij, _try_get_global_baseline(), ATOL)
    p_counts, p_runs, p_pairs = _accumulate_counts(pure_files,   k2ij, _try_get_global_baseline(), ATOL)

    # figure (same as before)
    viewerL, viewerR = _plot_with_connectome_viewer(h_counts, p_counts, baseline_global, names)

    # save topK changed lists (same as you had)
    _emit_topk(h_counts, names, "top_changes_hybrid.tsv", TOPK)
    _emit_topk(p_counts, names, "top_changes_pure.tsv",   TOPK)

    # --------------- shared prep for analyses -----------------
    N = len(names)
    W0_base = _vector_to_matrix(baseline_global, k2ij, N)
    group_map = _get_group_map(viewerL, names)  # use left viewer's grouping

    # (1) Edge-class enrichment with degree-aware null
    df_enrich_h = _edge_class_enrichment(h_counts, group_map, names, W0_base, k2ij, n_perm=N_PERM, deg_bins=DEG_BINS)
    df_enrich_p = _edge_class_enrichment(p_counts, group_map, names, W0_base, k2ij, n_perm=N_PERM, deg_bins=DEG_BINS)
    if df_enrich_h is not None and not df_enrich_h.empty:
        df_enrich_h.to_csv(EDGE_ENRICH_HYB_TSV, sep="\t", index=False)
    if df_enrich_p is not None and not df_enrich_p.empty:
        df_enrich_p.to_csv(EDGE_ENRICH_PURE_TSV, sep="\t", index=False)

    # (2) Sign/gain shifts by edge class and hemisphere
    df_sg_h = _aggregate_sign_gain(h_pairs, k2ij, names, group_map, ATOL)
    df_sg_p = _aggregate_sign_gain(p_pairs, k2ij, names, group_map, ATOL)
    if not df_sg_h.empty:
        df_sg_h.to_csv(SIGN_GAIN_HYB_TSV, sep="\t", index=False)
    if not df_sg_p.empty:
        df_sg_p.to_csv(SIGN_GAIN_PURE_TSV, sep="\t", index=False)

    # (3) Sensory→motor path-gain changes (pre vs post)
    df_pg_h = _sensory_motor_path_changes(h_pairs, k2ij, names, group_map, H=MAX_HOPS, topk=TOPK)
    df_pg_p = _sensory_motor_path_changes(p_pairs, k2ij, names, group_map, H=MAX_HOPS, topk=TOPK)
    if not df_pg_h.empty:
        df_pg_h.to_csv(PATH_GAIN_HYB_TSV, sep="\t", index=False)
    if not df_pg_p.empty:
        df_pg_p.to_csv(PATH_GAIN_PURE_TSV, sep="\t", index=False)

    print(f"[hybrid] runs={h_runs}, shown edges={(h_counts>=MIN_COUNT).sum()}, "
          f"max_shown={int((h_counts[h_counts>=MIN_COUNT]).max()) if (h_counts>=MIN_COUNT).any() else 0}")
    print(f"[pure]   runs={p_runs}, shown edges={(p_counts>=MIN_COUNT).sum()}, "
          f"max_shown={int((p_counts[p_counts>=MIN_COUNT]).max()) if (p_counts>=MIN_COUNT).any() else 0}")
    print(f"Saved figures: {OUT_SVG}, {OUT_PNG}")
    print("Saved top-K lists: top_changes_hybrid.tsv, top_changes_pure.tsv")
    if df_enrich_h is not None and not df_enrich_h.empty:
        print(f"Saved enrichment: {EDGE_ENRICH_HYB_TSV}")
    if df_enrich_p is not None and not df_enrich_p.empty:
        print(f"Saved enrichment: {EDGE_ENRICH_PURE_TSV}")
    if not df_sg_h.empty:
        print(f"Saved sign/gain (hybrid): {SIGN_GAIN_HYB_TSV}")
    if not df_sg_p.empty:
        print(f"Saved sign/gain (pure):   {SIGN_GAIN_PURE_TSV}")
    if not df_pg_h.empty:
        print(f"Saved path gains (hybrid): {PATH_GAIN_HYB_TSV}")
    if not df_pg_p.empty:
        print(f"Saved path gains (pure):   {PATH_GAIN_PURE_TSV}")

    # NEW: global sign-balance summary
    _print_pos_neg_summary(h_pairs, "hybrid", ATOL)
    _print_pos_neg_summary(p_pairs, "pure", ATOL)
    # overall sign-flip percentages
    _print_overall_sign_flip(h_pairs, k2ij, ATOL, "hybrid")
    _print_overall_sign_flip(p_pairs, k2ij, ATOL, "pure")
    _print_directional_sign_flips(h_pairs, k2ij, ATOL, "hybrid")
    _print_directional_sign_flips(p_pairs, k2ij, ATOL, "pure")

if __name__ == "__main__":
    main()
