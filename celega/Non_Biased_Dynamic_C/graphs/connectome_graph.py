# connectome_graph.py — revised 2025-08-01
# ==============================================================
# Drop-in replacement for your original ConnectomeViewer.
# Key extras:
#   • layout spread-factor   (spread)
#   • spike-pulse node size  (pulse_size)
#   • adjustable group gap   (group_gap)
#   • robust Kamada-Kawai    (inverse-length + ring fallback)
# ==============================================================

from __future__ import annotations
import logging
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import warnings

__all__ = ["ConnectomeViewer"]


class ConnectomeViewer:
    def __init__(
        self,
        wc,
        *,
        layout: str = "kamada_kawai",
        spread: float = 1.6,          # global zoom-out
        pulse_size: float = 2.5,      # × node_size on spike
        group_gap: float = 12.0,       # horiz. gap for grouped layouts
        threshold: Optional[float] = None,
        max_edges: int = 6_000,
        node_size: int = 40,
        color_mode: str = "binary",   # 'binary' | 'heat' | 'energy'
        colormap: str = "plasma",
        vmax: Optional[float] = None,
        inact_color: str = "#d3d3d3",
        act_color: str = "#ff5555",
        debug: bool = False,
    ) -> None:
        # ── simple logger ───────────────────────────────────────
        self.debug = bool(debug)
        self.log = logging.getLogger("ConnectomeViewer")
        if self.debug:
            logging.basicConfig(level=logging.INFO,
                                format="[\033[95m%(levelname)s\033[0m] %(message)s")
        else:
            self.log.addHandler(logging.NullHandler())

        # ── user-visible params ─────────────────────────────────
        self.wc         = wc
        self.node_size  = int(node_size)
        self.spread     = float(spread)
        self.pulse_size = float(pulse_size)
        self.group_gap  = float(group_gap)
        self.color_mode = color_mode
        self.inact_col  = inact_color
        self.act_col    = act_color

        # ── spike thresholds map ───────────────────────────────
        if threshold is not None:
            self.thr_map = np.full(wc.N, float(threshold))
        elif hasattr(wc, "_thr_map"):
            self.thr_map = wc._thr_map.astype(float)
        elif hasattr(wc, "threshold"):
            self.thr_map = np.full(wc.N, float(wc.threshold))
        else:
            self.thr_map = np.full(wc.N, 30.0)
        thr_scalar = float(np.median(self.thr_map))

        # ── colour scaling ─────────────────────────────────────
        if color_mode == "heat":
            vmax = 2.0 * thr_scalar if vmax is None else float(vmax)
            self.norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            self.cmap = cm.get_cmap(colormap)
        elif color_mode == "energy":
            vmax = thr_scalar if vmax is None else float(vmax)
            self.norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            self.cmap = cm.get_cmap("Blues_r")

        # ── state-buffer mode detection ───────────────────────
        if hasattr(wc, "post") and hasattr(wc, "curcol"):
            self._state_mode = "double"
        elif hasattr(wc, "post") and hasattr(wc, "t0"):
            self._state_mode = "double_t0"
        else:
            self._state_mode = "single"

        # ── build graph & layout ───────────────────────────────
        self.G   = self._build_graph(max_edges)
        self.pos = {n: p * self.spread for n, p in self._compute_layout(layout).items()}

        # ── initial draw ───────────────────────────────────────
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        init_color = self.inact_col if color_mode == "binary" else self.cmap(self.norm(0.0))
        self.node_coll = nx.draw_networkx_nodes(
            self.G, self.pos, node_size=self.node_size,
            node_color=[init_color], ax=self.ax
        )
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax,
                               arrows=False, alpha=0.25, width=0.3)
        self.ax.set_axis_off()
        self.fig.tight_layout()
        plt.show(block=False)

    # ==========================================================
    # public API
    # ==========================================================
    def step(self):
        # ➊ read membrane potentials (same code you have now) …
        if self._state_mode == "double":
            V = self.wc.post[:, self.wc.curcol]
        elif self._state_mode == "double_t0":
            V = self.wc.post[:, self.wc.t0]
        else:
            V = self.wc.V

        # ➋ use the visual-only spikes if provided
        spiked_now = getattr(self.wc, "spiked_vis",
                     getattr(self.wc, "spiked",
                             np.zeros_like(V, dtype=bool)))



        # node colours
        if self.color_mode == "binary":
            active = spiked_now | (np.abs(V) > self.thr_map)
            colors = np.where(active, self.act_col, self.inact_col)
        elif self.color_mode == "heat":
            colors = self.cmap(self.norm(np.abs(V)))
            colors[spiked_now] = mcolors.to_rgba(self.act_col)
        else:                                              # 'energy'
            blues  = self.cmap(self.norm(np.abs(V)))
            active = spiked_now | (np.abs(V) > self.thr_map)
            colors = np.where(active[:, None],
                              mcolors.to_rgba(self.act_col), blues)

        self.node_coll.set_color(colors)

        # pulse size
        sizes = np.where(spiked_now,
                         self.node_size * self.pulse_size,
                         self.node_size)
        self.node_coll.set_sizes(sizes)

        self.fig.canvas.draw_idle()

    update = step  # alias

    def relayout(self, layout: str = "spring") -> None:
        """Re-compute coordinates with *layout* and redraw edges."""
        self.pos = {n: p * self.spread for n, p in
                    self._compute_layout(layout).items()}
        self.ax.clear()
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax,
                               arrows=False, alpha=0.25, width=0.3)
        self.node_coll = nx.draw_networkx_nodes(
            self.G, self.pos, node_size=self.node_size, ax=self.ax
        )
        self.ax.set_axis_off()
        self.step()

    # ==========================================================
    # internal helpers
    # ==========================================================
    def _build_graph(self, max_edges: int):
        G = nx.DiGraph()
        G.add_nodes_from(self.wc.names)

        if hasattr(self.wc, "_edge_w") and hasattr(self.wc, "_edge_ptr"):
            idx = np.argsort(np.abs(self.wc._edge_w))[::-1][:max_edges]
            for (_, i, j), w in zip(np.asarray(self.wc._edge_ptr)[idx],
                                    self.wc._edge_w[idx]):
                G.add_edge(self.wc.names[i], self.wc.names[j], weight=w)
        else:
            flat = np.argsort(np.abs(self.wc.W).ravel())[::-1][:max_edges]
            src, dst = np.unravel_index(flat, self.wc.W.shape)
            for s, d in zip(src, dst):
                G.add_edge(self.wc.names[s], self.wc.names[d],
                           weight=self.wc.W[s, d])
        return G

    # ---------------- layout dispatcher -----------------------
    def _compute_layout(self, layout: str):
        if layout == "groups":
            return self._compute_group_layout()
        if layout == "kamada_groups":
            return self._compute_kamada_group_layout()
        return self._safe_standard_layout(layout)

    # ---------------- robust standard layouts ----------------
    def _safe_standard_layout(self, name: str):
        """Return a standard layout, healing Kamada-Kawai ring failures."""
        if name != "kamada_kawai":
            return self._standard_layout(name)

        # (a) global KK
        try:
            pos = self._standard_layout("kamada_kawai")
        except Exception as e:
            self.log.warning("KK raised %s → spring fallback", e)
            return self._standard_layout("spring")

        # (b) detect ring
        coords = np.vstack(list(pos.values()))
        r = np.linalg.norm(coords, axis=1)
        if r.std() / (r.mean() + 1e-12) < 5e-2:
            self.log.info("KK ring detected → spring fallback")
            return self._standard_layout("spring")
        return pos

    def _standard_layout(self, name: str):
        if name == "spring":
            return nx.spring_layout(self.G, seed=1)
        if name == "shell":
            return nx.shell_layout(self.G)
        if name == "circular":
            return nx.circular_layout(self.G)
        if name == "kamada_kawai":
            # undirected copy + inverse-length weighting
            H = self.G.to_undirected()
            for _, _, d in H.edges(data=True):
                w = abs(d.get("weight", 1.0))
                d["length"] = 1.0 / max(w, 1e-9)
            return nx.kamada_kawai_layout(H, weight="length")
        raise ValueError(f"unknown layout '{name}'")

    # ---------------- grouped layouts ------------------------
    def _compute_group_layout(self) -> Dict[str, np.ndarray]:
        names = self.wc.names
        touch   = list(self.wc.touch_idx)
        food    = list(self.wc.food_idx)
        muscle  = np.where(self.wc.muscle_mask)[0].tolist()
        other   = [i for i in range(self.wc.N)
                   if i not in set(touch + food + muscle)]

        groups  = [touch, food, other, muscle]
        x_base  = np.arange(len(groups)) * self.group_gap
        pos: Dict[str, np.ndarray] = {}
        for gx, idxs in zip(x_base, groups):
            for rank, i in enumerate(sorted(idxs)):
                pos[names[i]] = np.array([gx, -rank], float)

        rng = np.random.default_rng(42)
        for p in pos.values():
            p += rng.uniform(-0.05, 0.05, 2)
        return pos

    def _compute_kamada_group_layout(self) -> Dict[str, np.ndarray]:
        pos = self._safe_standard_layout("kamada_kawai")
        n = lambda idx: self.wc.names[idx]

        g_touch  = [n(i) for i in self.wc.touch_idx]
        g_food   = [n(i) for i in self.wc.food_idx]
        g_muscle = [n(i) for i in np.where(self.wc.muscle_mask)[0]]
        g_other  = [node for node in self.wc.names
                    if node not in g_touch + g_food + g_muscle]

        for grp, dx in zip([g_touch, g_food, g_other, g_muscle],
                           [-self.group_gap, 0, self.group_gap, 2*self.group_gap]):
            for node in grp:
                pos[node][0] += dx
        return pos
