# genetic_dyn_video.py — Hi-res edition (centered title across both panels)
# ---------------------------------------------------------------
import os
import cv2
import numpy as np
from typing import List
from moviepy import ImageSequenceClip
from Worm_Env.c_worm           import is_food_close
from Worm_Env.celegan_env      import WormSimulationEnv
from Worm_Env.connectome2      import WormConnectome
from util.write_read_txt       import read_arrays_from_csv_pandas
from graphs.connectome_graph   import ConnectomeViewer
import matplotlib.pyplot as plt

# ───────────────────────────────
RES_SCALE = 2          # 1 = original, 2 = 4× pixels, 3 = 9× pixels …
TITLE_TEXT = "After Training With mENOMAD"
TITLE_PAD = 15  # pixels above the images
# ───────────────────────────────

def _mpl_fig_to_bgr(fig, scale: int = RES_SCALE):
    """Return a high-DPI BGR image of a Matplotlib figure."""
    orig_dpi = fig.dpi
    fig.set_dpi(orig_dpi * scale)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    argb = np.frombuffer(fig.canvas.tostring_argb(), np.uint8).reshape(h, w, 4)
    bgr  = cv2.cvtColor(argb[:, :, 1:], cv2.COLOR_RGB2BGR)
    fig.set_dpi(orig_dpi)
    return bgr

def _side_by_side_with_title(arena, net, title: str):
    """Return side-by-side image with a Matplotlib title (Matplotlib font) positioned closer to the image, no border color."""
    # Match heights, concat in BGR
    h = arena.shape[0]
    net = cv2.resize(net, (int(net.shape[1] * h / net.shape[0]), h),
                     interpolation=cv2.INTER_AREA)
    combined = cv2.hconcat([arena, net])  # BGR

    # Make a Matplotlib figure sized to the image
    dpi = 300.0
    fig_w = combined.shape[1] / dpi
    fig_h = combined.shape[0] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # Force background to white to remove colored ring
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    # Add title closer to the top of the image
    ax.text(
        0.5, 0.95,
        title,
        ha="center", va="bottom",
        fontsize=14,
        transform=ax.transAxes
    )

    fig.tight_layout(pad=0)

    # Render ➜ RGBA buffer, then to BGR ndarray
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)
    plt.close(fig)

    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    return bgr


class GeneticDynVideo:
    def __init__(self,
                 patterns          : List[int],
                 episodes          : int       = 1,
                 steps_per_episode : int       = 250,
                 tmp_dir           : str       = "tmp_img"):
        self.patterns  = patterns
        self.episodes  = episodes
        self.steps     = steps_per_episode
        self.tmp_dir   = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)

    def _orig_genome(self, csv="Hybrid_nomad29.csv"): ##Hybrid_nomad29.csv Pure_nomad15
        base_dir = os.path.dirname(__file__)
        full_path = os.path.join(os.path.join(base_dir, "data_new_pentagon"), csv)
        rows = read_arrays_from_csv_pandas(full_path)
        if not rows:
            raise FileNotFoundError(f"{csv} is empty.")
        return np.asarray(rows[-1], dtype=float)

    @staticmethod
    def _all_names():
        from Worm_Env.weight_dict import dict as wg
        names = set(wg.keys())
        for dst in wg.values():
            names.update(dst)
        return sorted(names)

    def run(self, out="simulation.mp4", fps=25):
        worm = WormConnectome(weight_matrix=self._orig_genome(),
                              all_neuron_names=self._all_names())
        env  = WormSimulationEnv(num_worms=1)
        viewer = ConnectomeViewer(
            worm,
            layout="kamada_groups",
            spread=1,
            pulse_size=3.0,
            group_gap=0.5,
            color_mode="energy"
        )

        frame_idx = 0
        for pat in self.patterns:
            for _ in range(self.episodes):
                obs = env.reset(pat)
                worm.V[:] = 0.0

                for _ in range(self.steps):
                    move = worm.move(obs[0, 0], obs[0, 4])
                    obs, _, done = env.step(move, worm_num=0, candidate=None)

                    viewer.step()
                    env.render()
                    env.ax.axis("off")
                    arena  = _mpl_fig_to_bgr(env.fig)
                    netvis = _mpl_fig_to_bgr(viewer.fig)

                    

                    frame = _side_by_side_with_title(arena, netvis, TITLE_TEXT)

                    cv2.imwrite(os.path.join(self.tmp_dir,
                                             f"frame_{frame_idx:05d}.png"),
                                frame)
                    frame_idx += 1
                    if done:
                        break

        pngs = sorted(p for p in os.listdir(self.tmp_dir) if p.endswith(".png"))
        clip = ImageSequenceClip([os.path.join(self.tmp_dir, p) for p in pngs],
                                 fps=fps)
        clip.write_videofile(out, codec="libx264", audio=False)
        for p in pngs:
            os.remove(os.path.join(self.tmp_dir, p))
        print(f"✓ saved {out}  ({len(pngs)} frames)")
    def save_cover_frame(self, out="connectome_cover.png"):
        """
        Run a single episode for the 'middle' pattern and save a high-res
        transparent connectome frame from the middle of the simulation as `out`.
        """
        if not self.patterns:
            raise ValueError("No patterns provided for GeneticDynVideo.")

        # Pick the 'middle' pattern and mid-step
        pat_idx     = len(self.patterns) // 2
        pat         = self.patterns[pat_idx]
        target_step = self.steps // 2

        worm = WormConnectome(
            weight_matrix=self._orig_genome(),
            all_neuron_names=self._all_names(),
        )

        # Use the cover-specific viewer
        from graphs.connectome_graph_cover import ConnectomeViewer as CoverConnectomeViewer

        env    = WormSimulationEnv(num_worms=1)
        viewer = CoverConnectomeViewer(
            worm,
            layout="kamada_groups",
            spread=1,
            pulse_size=3.0,
            group_gap=0.5,
            color_mode="energy",
        )

        obs = env.reset(pat)
        worm.V[:] = 0.0

        saved = False
        for t in range(self.steps):
            move = worm.move(obs[0, 0], obs[0, 4])
            obs, _, done = env.step(move, worm_num=0, candidate=None)

            viewer.step()
            env.render()      # you can drop this if you never use the arena
            env.ax.axis("off")

            if t == target_step or done:
                viewer.fig.savefig(
                    out,
                    dpi=600,
                    transparent=True,
                    facecolor="none",
                    bbox_inches="tight",
                    pad_inches=0.0,
                )
                print(f"✓ saved cover frame → {out}  (step {t}, pattern {pat})")
                saved = True
                break

        if not saved:
            # Fallback: save last state of the connectome, still transparent
            viewer.fig.savefig(
                out,
                dpi=600,
                transparent=True,
                facecolor="none",
                bbox_inches="tight",
                pad_inches=0.0,
            )
            print(f"✓ saved fallback cover frame → {out}")