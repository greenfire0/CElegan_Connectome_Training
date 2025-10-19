# fig5.py – three-panel connectome comparison (PDF-safe thin strokes)
# ===================================================================
# Keeps original API: `run()` exists and saves fig5.svg (and also fig5.pdf).
# Fixes low-zoom “fat line” artifacts while preserving prior behavior.

from __future__ import annotations

import os
import re
import sys
from typing import Tuple, List

import matplotlib as mpl
mpl.rcParams["path.simplify"] = False
mpl.rcParams["agg.path.chunksize"] = 0
mpl.rcParams["pdf.fonttype"] = 42         # keep text as text
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"     # keep text as text in SVG

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from graphs.connectome_graph import ConnectomeViewer
from Worm_Env.connectome2 import WormConnectome
from Worm_Env.weight_dict import all_neuron_names, mLeft, mRight
from util.write_read_txt import read_arrays_from_csv_pandas

# ──────────────────────────────────────────────
# SVG post-processor: add non-scaling stroke
# ──────────────────────────────────────────────
def _svg_add_non_scaling_stroke(svg_path: str) -> None:
    with open(svg_path, "r", encoding="utf-8") as f:
        svg = f.read()
    def _inject(match):
        tag, attrs, selfclose = match.groups()
        if 'vector-effect=' in attrs:
            return match.group(0)
        return f"<{tag}{attrs} vector-effect=\"non-scaling-stroke\"{selfclose}>"
    svg = re.sub(r"<(path|line|polyline)([^>]*?)(/?)>", _inject, svg)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

# ──────────────────────────────────────────────
# edge helpers
# ──────────────────────────────────────────────
def _edge_lists(W: np.ndarray, names: List[str], tol: float = 1e-6):
    src, dst = np.where(np.abs(W) > tol)
    edges = [(names[i], names[j]) for i, j in zip(src, dst)]
    signs = [1 if W[i, j] > 0 else -1 for i, j in zip(src, dst)]
    mags = [abs(W[i, j]) for i, j in zip(src, dst)]
    return edges, signs, mags

def _filter_present(edges, signs, mags, pos):
    keep = [(e, s, m) for e, s, m in zip(edges, signs, mags)
            if e[0] in pos and e[1] in pos]
    return ([], [], []) if not keep else zip(*keep)

def _style_linecollection(coll, rasterize_edges: bool):
    colls = coll if isinstance(coll, (list, tuple)) else [coll]
    for c in colls:
        try:
            c.set_capstyle("butt")
            c.set_joinstyle("miter")
            c.set_antialiased(True)
            c.set_rasterized(rasterize_edges)
        except Exception:
            pass

def _draw_edges(ax, pos, edges, signs, mags,
                w_min: float, w_max: float,
                color_map=("red", "blue"),
                rasterize_edges: bool = True):
    if not edges:
        return
    mags = np.asarray(mags, float)
    widths = w_min + (w_max - w_min) * np.sqrt(mags / max(mags.max(), 1e-12))
    colors = [color_map[0] if s > 0 else color_map[1] for s in signs]
    coll = nx.draw_networkx_edges(nx.DiGraph(edges), pos,
                                  edgelist=edges,
                                  edge_color=colors,
                                  width=widths,
                                  arrows=False,
                                  ax=ax)
    _style_linecollection(coll, rasterize_edges)

def _draw_gray(ax, pos, edges, w=0.3, rasterize_edges: bool = True):
    if edges:
        coll = nx.draw_networkx_edges(nx.DiGraph(edges), pos,
                                      edgelist=edges,
                                      edge_color="#A5A5A5",
                                      width=w,
                                      arrows=False,
                                      ax=ax)
        _style_linecollection(coll, rasterize_edges)

# ──────────────────────────────────────────────
# add "L" / "R" labels over muscle columns
# ──────────────────────────────────────────────
def _label_lr(ax, pos, dy=0.05):
    left  = [pos[n] for n in mLeft  if n in pos]
    right = [pos[n] for n in mRight if n in pos]
    if left:
        xL = np.mean([p[0] for p in left]); yL = max(p[1] for p in left) + dy
        ax.text(xL, yL, "L", ha="center", va="bottom",
                fontsize=34, fontweight="bold")
    if right:
        xR = np.mean([p[0] for p in right]); yR = max(p[1] for p in right) + dy
        ax.text(xR, yR, "R", ha="center", va="bottom",
                fontsize=34, fontweight="bold")

# ──────────────────────────────────────────────
# CSV auto-loader
# ──────────────────────────────────────────────
def _auto_load(folder: str = "data_full_pentagon") -> Tuple[WormConnectome, ...]:
    here = os.path.dirname(__file__)
    path = os.path.join(here, folder)
    if not os.path.isdir(path): raise FileNotFoundError(path)
    files = os.listdir(path)
    orig_f = next((f for f in files if "Hybrid" in f and "oginit" in f), None)
    pure_f = next((f for f in files if "Pure" in f), None)
    hyb_f  = next((f for f in files if "Hybrid" in f and "oginit" not in f), None)
    if not (orig_f and pure_f and hyb_f):
        raise RuntimeError("CSV set incomplete.")
    def _row(csv: str, first=False):
        arrs = read_arrays_from_csv_pandas(os.path.join(path, csv))
        return np.asarray(arrs[0 if first else -1], float)
    wc_orig = WormConnectome(_row(orig_f, first=True), all_neuron_names)
    wc_pure = WormConnectome(_row(pure_f), all_neuron_names)
    wc_hyb  = WormConnectome(_row(hyb_f),  all_neuron_names)
    return wc_orig, wc_pure, wc_hyb

# ──────────────────────────────────────────────
# main plotting routine
# ──────────────────────────────────────────────
def plot_three(wc_orig, wc_pure, wc_hyb,
               tol: float = 1e-6,
               node_sz: int = 40,
               figure_inches=(36, 12),
               rasterize_edges_pdf: bool = True):
    cv = ConnectomeViewer(wc_orig,
                          layout="kamada_groups",
                          spread=1,
                          pulse_size=3.0,
                          group_gap=.5,
                          node_size=node_sz,
                          color_mode="binary")
    pos, G = cv.pos, cv.G

    fig, axs = plt.subplots(1, 3, figsize=figure_inches, dpi=300)
    plt.subplots_adjust(bottom=0.05)
    for ax in axs:
        ax.set_axis_off()
        nx.draw_networkx_nodes(G, pos,
                               node_size=node_sz,
                               node_color="#838383",
                               ax=ax)
        _label_lr(ax, pos)

    # 1) Original
    e, s, m = _edge_lists(wc_orig.W, all_neuron_names, tol)
    e, s, m = _filter_present(e, s, m, pos)
    _draw_edges(axs[0], pos, e, s, m,
                w_min=0.25, w_max=1.2,
                color_map=("red", "blue"),
                rasterize_edges=rasterize_edges_pdf)

    # 2) rENOMAD Δ
    diff_pure = wc_pure.W - wc_orig.W
    eΔ, sΔ, mΔ = _edge_lists(diff_pure, all_neuron_names, tol)
    eΔ, sΔ, mΔ = _filter_present(eΔ, sΔ, mΔ, pos)
    e_full_pure, _, _ = _edge_lists(wc_pure.W, all_neuron_names, tol)
    bg_pure = [e for e in e_full_pure if e not in eΔ and e[0] in pos and e[1] in pos]
    _draw_gray(axs[1], pos, bg_pure, w=0.3, rasterize_edges=rasterize_edges_pdf)
    _draw_edges(axs[1], pos, eΔ, sΔ, mΔ,
                w_min=0.25, w_max=5.0,
                color_map=("#ff00ff", "#39ff14"),
                rasterize_edges=rasterize_edges_pdf)

    # 3) mENOMAD Δ
    diff_hyb = wc_hyb.W - wc_orig.W
    eΔh, sΔh, mΔh = _edge_lists(diff_hyb, all_neuron_names, tol)
    eΔh, sΔh, mΔh = _filter_present(eΔh, sΔh, mΔh, pos)
    e_full_hyb, _, _ = _edge_lists(wc_hyb.W, all_neuron_names, tol)
    bg_hyb = [e for e in e_full_hyb if e not in eΔh and e[0] in pos and e[1] in pos]
    _draw_gray(axs[2], pos, bg_hyb, w=0.3, rasterize_edges=rasterize_edges_pdf)
    _draw_edges(axs[2], pos, eΔh, sΔh, mΔh,
                w_min=0.25, w_max=5.0,
                color_map=("#ff00ff", "#39ff14"),
                rasterize_edges=rasterize_edges_pdf)

    for i, ax in enumerate(axs):
        ax.text(0.00, 0.95, f"({chr(97 + i)})",
                transform=ax.transAxes,
                fontsize=40, fontweight="bold",
                va="top", ha="left")

    plt.tight_layout()

    # SVG (original filename preserved) + add non-scaling strokes
    svg_path = "fig5.svg"
    fig.savefig(svg_path, format="svg", transparent=True, metadata={"Creator": "fig5.py"})
    _svg_add_non_scaling_stroke(svg_path)

    # PDF (optional; keeps text vector, edges rasterized at viewer)
    pdf_path = "fig5.pdf"
    fig.savefig(pdf_path, format="pdf", transparent=True, metadata={"Creator": "fig5.py"})
    print("✓ saved fig5.svg and fig5.pdf")

# ──────────────────────────────────────────────
# public API preserved
# ──────────────────────────────────────────────
def run():
    try:
        wc_orig, wc_pure, wc_hyb = _auto_load()
    except Exception as e:
        sys.exit(f"[fig5] {e}")
    plot_three(wc_orig, wc_pure, wc_hyb)

if __name__ == "__main__":
    run()
