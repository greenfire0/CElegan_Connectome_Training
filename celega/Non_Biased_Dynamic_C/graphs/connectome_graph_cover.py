import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, Optional, Sequence 
from Worm_Env.weight_dict import mLeft, mRight, muscleList

class ConnectomeViewer:
    """
    *C. elegans* connectome viewer tuned for iScience cover:

    - Dark navy background, no text.
    - Cool blue energy colormap matching the worm micrograph.
    - Full-bleed network (no margins), equal aspect.
    """

    _ALLOWED_MUSCLE_PREFIX: Sequence[str] = ("MDL", "MDR", "MVL", "MVR")

    def __init__(
        self,
        wc,
        *,
        layout: str = "kamada_kawai",
        spread: float = 1.6,
        pulse_size: float = 2.5,
        group_gap: float = 12.0,
        threshold: Optional[float] = None,
        max_edges: int = 6_000,
        node_size: int = 40,
        color_mode: str = "energy",  # 'binary' | 'heat' | 'energy'
        colormap: str = "Blues",     # ignored for 'energy' (we use custom)
        vmax: Optional[float] = None,
        inact_color: str = "#18365C",   # deep blue for inactive nodes
        act_color: str = "#9ad7ff",     # pale cyan for spikes
    ):
        self.wc = wc
        self.node_size = node_size
        self.spread = spread
        self.pulse_size = pulse_size
        self.group_gap = group_gap
        self.color_mode = color_mode
        self.inact_col = inact_color
        self.act_col = act_color

        # ---- threshold map -------------------------------------------
        if threshold is not None:
            self.thr_map = np.full(wc.N, threshold, float)
        elif hasattr(wc, "_thr_map"):
            self.thr_map = wc._thr_map.astype(float)
        elif hasattr(wc, "threshold"):
            self.thr_map = np.full(wc.N, wc.threshold, float)
        else:
            self.thr_map = np.full(wc.N, 30.0, float)
        thr_med = float(np.median(self.thr_map))

        # ---- colour scales -------------------------------------------
        if color_mode == "heat":
            vmax = 2.0 * thr_med if vmax is None else float(vmax)
            self.norm = mcolors.Normalize(0.0, vmax)
            self.cmap = cm.get_cmap(colormap)
        elif color_mode == "energy":
            vmax = thr_med if vmax is None else float(vmax)
            self.norm = mcolors.Normalize(0.0, vmax)
            # custom cool-blue gradient: very dark navy → saturated blue → pale cyan
            self.cmap = mcolors.LinearSegmentedColormap.from_list(
                "worm_blue",
                [
                    "#1c4890",  # almost black
                    "#497edf",  # deep navy
                    "#7ab6fb",  # saturated blue
                    "#bbdaff",  # pale cyan
                ],
            )
        else:
            self.norm = None
            self.cmap = None

        # ---- state mode detection ------------------------------------
        if all(hasattr(wc, a) for a in ("post", "curcol")):
            self._state_mode = "double"
        elif all(hasattr(wc, a) for a in ("post", "t0")):
            self._state_mode = "double_t0"
        else:
            self._state_mode = "single"

        # ---- node selection ------------------------------------------
        _MWHITELIST = set(muscleList)
        self._muscle_nodes = [n for n in wc.names if n in _MWHITELIST]

        self.G = self._build_graph(max_edges)
        self.name2idx = {n: i for i, n in enumerate(wc.names)}
        self.node_order = list(self.G.nodes())
        self.node_indices = np.array([self.name2idx[n] for n in self.node_order])
        self.muscle_mask_draw = np.array(
            [n in self._muscle_nodes for n in self.node_order]
        )

        # ---- layout and normalisation --------------------------------
        self.pos = {n: p * self.spread for n, p in self._compute_layout(layout).items()}
        self._normalise_positions()  # map to [0,1]×[0,1] to fill figure

        # ---- figure / axes -------------------------------------------
        # Dark navy background, full-bleed axes, no text.
        self.bg_color = self.bg_color   = (0, 0, 0, 0)
        self.edge_color = "#226ac8"  # muted blue for edges

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.patch.set_facecolor("none")
        self.fig.patch.set_alpha(0.0)
        self.ax.set_facecolor("none")
        self.ax.set_axis_off()
        self.ax.set_aspect("equal", "box")
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        baseline = (
            self.inact_col
            if color_mode == "binary"
            else self.cmap(0.35) if self.cmap is not None else self.inact_col
        )
        self._draw_graph(baseline)


    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def step(self):
        """Redraw node colours & sizes for the current timestep."""
        # select voltage vector
        if hasattr(self.wc, "V_vis"):
            V_draw = self.wc.V_vis[self.node_indices]
        else:
            if self._state_mode == "double":
                V_draw = self.wc.post[:, self.wc.curcol][self.node_indices]
            elif self._state_mode == "double_t0":
                V_draw = self.wc.post[:, self.wc.t0][self.node_indices]
            else:
                V_draw = self.wc.V[self.node_indices]

        # spike mask
        spk = getattr(
            self.wc,
            "spiked_vis",
            getattr(self.wc, "spiked", np.zeros_like(self.wc.V, bool)),
        )[self.node_indices]
        spk[self.muscle_mask_draw] = False  # muscles never flash

        # colour logic
        if self.color_mode == "binary":
            active = spk | (np.abs(V_draw) > self.thr_map[self.node_indices])
            active[self.muscle_mask_draw] = False
            colors = np.where(active, self.act_col, self.inact_col)
        elif self.color_mode == "heat":
            colors = self.cmap(self.norm(np.abs(V_draw)))
            colors[spk] = mcolors.to_rgba(self.act_col)
        else:  # 'energy'
            base = self.cmap(self.norm(np.abs(V_draw)))
            active = spk | (np.abs(V_draw) > self.thr_map[self.node_indices])
            colors = np.where(
                active[:, None], mcolors.to_rgba(self.act_col), base
            )

        self.node_coll.set_color(colors)

        # pulse size
        sizes = np.where(spk, self.node_size * self.pulse_size, self.node_size)
        self.node_coll.set_sizes(sizes)

        self.fig.canvas.draw_idle()

    update = step  # alias for animation loops

    # ------------------------------------------------------------------
    # helper functions
    # ------------------------------------------------------------------
    def _normalise_positions(self):
        """Rescale positions to [0,1]×[0,1] so the graph fills the canvas."""
        xs = np.array([p[0] for p in self.pos.values()])
        ys = np.array([p[1] for p in self.pos.values()])
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        margin = 0.05

        for k, p in self.pos.items():
            x = (p[0] - min_x) / span_x
            y = (p[1] - min_y) / span_y
            x = margin + (1 - 2 * margin) * x
            y = margin + (1 - 2 * margin) * y
            self.pos[k] = np.array([x, y])

    def _evenly_space(self, nodes, x, dy, y_shift=0.0):
        if not nodes:
            return {}
        y0 = -(len(nodes) - 1) / 3.0 * dy + y_shift
        return {n: np.array([x, y0 + k * dy]) for k, n in enumerate(sorted(nodes))}

    def _is_allowed_muscle(self, name):
        return name in self._muscle_nodes

    # ---------------- graph construction ------------------------------
    def _build_graph(self, max_edges):
        keep = [
            n
            for i, n in enumerate(self.wc.names)
            if not (self.wc.muscle_mask[i] and not self._is_allowed_muscle(n))
        ]
        keep_set = set(keep)

        G = nx.DiGraph()
        G.add_nodes_from(keep)

        if hasattr(self.wc, "_edge_w") and hasattr(self.wc, "_edge_ptr"):
            idx = np.argsort(np.abs(self.wc._edge_w))[::-1][:max_edges]
            for (_, i, j), w in zip(self.wc._edge_ptr[idx], self.wc._edge_w[idx]):
                s, d = self.wc.names[i], self.wc.names[j]
                if s in keep_set and d in keep_set:
                    G.add_edge(s, d, weight=w)
        else:
            flat = np.argsort(np.abs(self.wc.W).ravel())[::-1][:max_edges]
            src, dst = np.unravel_index(flat, self.wc.W.shape)
            for s, d in zip(src, dst):
                sn, dn = self.wc.names[s], self.wc.names[d]
                if sn in keep_set and dn in keep_set:
                    G.add_edge(sn, dn, weight=self.wc.W[s, d])
        return G

    def _draw_graph(self, init_color):
        nx.draw_networkx_edges(
            self.G,
            self.pos,
            ax=self.ax,
            arrows=False,
            alpha=0.35,
            width=0.5,
            edge_color=self.edge_color,
        )
        self.node_coll = nx.draw_networkx_nodes(
            self.G,
            self.pos,
            node_size=self.node_size,
            node_color=[init_color],
            ax=self.ax,
        )

    # ---------------- layout dispatch -------------------------------
    def _compute_layout(self, layout):
        return (
            self._compute_kamada_group_layout()
            if layout == "kamada_groups"
            else self._safe_standard_layout(layout)
        )

    def _safe_standard_layout(self, name):
        if name != "kamada_kawai":
            return self._standard_layout(name)
        try:
            pos = self._standard_layout("kamada_kawai")
        except Exception:
            return self._standard_layout("spring")
        r = np.linalg.norm(np.vstack(list(pos.values())), axis=1)
        return pos if np.std(r) >= 1e-2 else self._standard_layout("spring")

    def _standard_layout(self, name):
        if name == "spring":
            return nx.spring_layout(self.G, seed=1)
        if name == "kamada_kawai":
            H = self.G.to_undirected()
            for _, _, d in H.edges(data=True):
                w = abs(d.get("weight", 1.0))
                d["length"] = 1.0 / max(w, 1e-9)
            return nx.kamada_kawai_layout(H, weight="length")
        raise ValueError(name)

    # ---------------- grouped Kamada-Kawai --------------------
    def _compute_kamada_group_layout(self) -> Dict[str, np.ndarray]:
        """
        Touch inputs   → left columns
        Food inputs    → adjacent left columns
        Left muscles   → right inner column
        Right muscles  → right outer column
        Centre nodes   → Kamada-Kawai, inner half fanned ×6
        """
        pos = self._safe_standard_layout("kamada_kawai")
        n_by = lambda idx: self.wc.names[idx]

        g_touch = [n_by(i) for i in self.wc.touch_idx if n_by(i) in self.G]
        g_food  = [n_by(i) for i in self.wc.food_idx if n_by(i) in self.G]
        g_left  = [n for n in self.G if n in mLeft]
        g_right = [n for n in self.G if n in mRight]

        gap = self.group_gap
        x_center = 3.25 * gap
        offset   = 0.5 * gap

        x_touch = -x_center - offset
        x_food  = -x_center + offset
        x_left  =  x_center - offset
        x_right =  x_center + offset

        dy_fixed = 0.04
        shift_up = 0.15

        pos.update(self._evenly_space(g_touch, x_touch, dy_fixed,  y_shift=shift_up))
        pos.update(self._evenly_space(g_food,  x_food,  dy_fixed,  y_shift=shift_up))
        pos.update(self._evenly_space(g_left,  x_left,  dy_fixed))
        pos.update(self._evenly_space(g_right, x_right, dy_fixed))

        g_center = [
            n
            for n in self.G
            if n not in g_touch + g_food + g_left + g_right and n != "MVULVA"
        ]
        if g_center:
            pts   = np.vstack([pos[n] for n in g_center])
            cent  = pts.mean(0)
            r     = np.linalg.norm(pts - cent, axis=1)
            r_med = np.median(r)
            for n, rv in zip(g_center, r):
                scale = 6.0 if rv < r_med else 1.0
                pos[n] = cent + (pos[n] - cent) * scale

        return pos
