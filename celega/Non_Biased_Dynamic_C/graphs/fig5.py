# fig5.py – three-panel connectome comparison
# ==========================================
from __future__ import annotations

import os
import sys
from typing import Tuple, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from graphs.connectome_graph import ConnectomeViewer
from Worm_Env.connectome2 import WormConnectome
from Worm_Env.weight_dict import all_neuron_names
from util.write_read_txt import read_arrays_from_csv_pandas


# ──────────────────────────────────────────────
# edge helpers
# ──────────────────────────────────────────────
def _edge_lists(W: np.ndarray, names: List[str], tol: float = 1e-6):
    """Return edge list, sign list (+1/-1), magnitude list (|w| > tol)."""
    src, dst = np.where(np.abs(W) > tol)
    edges = [(names[i], names[j]) for i, j in zip(src, dst)]
    signs = [1 if W[i, j] > 0 else -1 for i, j in zip(src, dst)]
    mags = [abs(W[i, j]) for i, j in zip(src, dst)]
    return edges, signs, mags


def _filter_present(edges, signs, mags, pos):
    """Keep only edges whose endpoints exist in *pos* (avoids KeyError)."""
    keep = [(e, s, m) for e, s, m in zip(edges, signs, mags)
            if e[0] in pos and e[1] in pos]
    return ([], [], []) if not keep else zip(*keep)


def _draw_edges(ax, pos, edges, signs, mags,
                w_min=0.2, w_max=2.5, scale=True,
                color_map=("red", "blue")):
    if not edges:
        return
    if scale:
        w_max_mag = max(mags)
        widths = [w_min + (w_max - w_min) * (m / w_max_mag) for m in mags]
    else:
        widths = [w_max] * len(edges)
    colors = [color_map[0] if s > 0 else color_map[1] for s in signs]
    nx.draw_networkx_edges(nx.DiGraph(edges), pos,
                           edgelist=edges,
                           edge_color=colors,
                           width=widths,
                           arrows=False,
                           ax=ax)


def _draw_gray(ax, pos, edges, w=0.08):
    if edges:
        nx.draw_networkx_edges(nx.DiGraph(edges), pos,
                               edgelist=edges,
                               edge_color="#000000",
                               width=w,
                               arrows=False,
                               ax=ax)


# ──────────────────────────────────────────────
# CSV auto-loader
# ──────────────────────────────────────────────
def _auto_load(folder: str = "data_full_pentagon") -> Tuple[WormConnectome, ...]:
    here = os.path.dirname(__file__)
    path = os.path.join(here, folder)
    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    files = os.listdir(path)
    orig_f = next((f for f in files if "Hybrid" in f and "oginit" in f), None)
    pure_f = next((f for f in files if "Pure" in f), None)
    hyb_f = next((f for f in files if "Hybrid" in f and "oginit" not in f), None)
    if not (orig_f and pure_f and hyb_f):
        raise RuntimeError("CSV set incomplete.")

    def _row(csv: str, first=False):
        arrs = read_arrays_from_csv_pandas(os.path.join(path, csv))
        return np.asarray(arrs[0 if first else -1], float)

    wc_orig = WormConnectome(_row(orig_f, first=True), all_neuron_names)
    wc_pure = WormConnectome(_row(pure_f), all_neuron_names)
    wc_hyb = WormConnectome(_row(hyb_f), all_neuron_names)
    return wc_orig, wc_pure, wc_hyb


# ──────────────────────────────────────────────
# main plotting routine
# ──────────────────────────────────────────────
def plot_three(wc_orig, wc_pure, wc_hyb,
               tol=1e-6, node_sz=40, dpi=300):
    # shared layout
    cv = ConnectomeViewer(wc_orig,
                          layout="kamada_groups",
                          spread=1,
                          pulse_size=3.0,
                          group_gap=.5,
                          node_size=node_sz,
                          color_mode="binary")
    pos, G = cv.pos, cv.G

    fig, axs = plt.subplots(3, 1, figsize=(12, 36), dpi=dpi)

    # common label style
    label_kw = dict(font_size=5,
                    font_color="black",
                    clip_on=True,
    )

    for ax in axs:
        ax.set_axis_off()
        nx.draw_networkx_nodes(G, pos,
                               node_size=node_sz,
                               node_color="#0c0c0c",
                               ax=ax)

    # 1) ORIGINAL -------------------------------------------------
    axs[0].set_title("Original connectome", fontsize=20, pad=12)
    e, s, m = _edge_lists(wc_orig.W, all_neuron_names, tol)
    e, s, m = _filter_present(e, s, m, pos)
    _draw_edges(axs[0], pos, e, s, m,
                w_min=.1, w_max=1.2, scale=True)
    # 2) PURE NOMAD Δ --------------------------------------------
    axs[1].set_title("rE-NOMAD", fontsize=20, pad=12)
    diff = wc_pure.W - wc_orig.W
    eΔ, sΔ, mΔ = _edge_lists(diff, all_neuron_names, tol)
    eΔ, sΔ, mΔ = _filter_present(eΔ, sΔ, mΔ, pos)

    e_full, _, _ = _edge_lists(wc_pure.W, all_neuron_names, tol)
    e_keep = [e for e in e_full if e not in eΔ and e[0] in pos and e[1] in pos]
    _draw_gray(axs[1], pos, e_keep, w=.05)
    _draw_edges(
        axs[1], pos, eΔ, sΔ, mΔ,
        w_min=.2, w_max=3.0, scale=True,
        color_map=("magenta", "green")  # ← changed
    )

    # 3) HYBRID NOMAD Δ ------------------------------------------
    axs[2].set_title("mE-NOMAD", fontsize=20, pad=12)
    diff_h = wc_hyb.W - wc_orig.W
    eΔh, sΔh, mΔh = _edge_lists(diff_h, all_neuron_names, tol)
    eΔh, sΔh, mΔh = _filter_present(eΔh, sΔh, mΔh, pos)

    e_full_h, _, _ = _edge_lists(wc_hyb.W, all_neuron_names, tol)
    e_keep_h = [e for e in e_full_h if e not in eΔh and e[0] in pos and e[1] in pos]
    _draw_gray(axs[2], pos, e_keep_h, w=.05)
    _draw_edges(
        axs[2], pos, eΔh, sΔh, mΔh,
        w_min=.2, w_max=3.0, scale=True,
        color_map=("magenta", "green")  # ← changed
    )


    plt.tight_layout()
    plt.savefig("fig5.svg", dpi=300)
    print("✓ saved fig5.svg")


# ──────────────────────────────────────────────
def run():
    try:
        wc_orig, wc_pure, wc_hyb = _auto_load()
    except Exception as e:
        sys.exit(f"[fig5] {e}")
    plot_three(wc_orig, wc_pure, wc_hyb)


if __name__ == "__main__":
    run()
