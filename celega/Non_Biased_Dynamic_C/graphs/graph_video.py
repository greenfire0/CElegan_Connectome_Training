# genetic_dyn_video.py — C. elegans LIF viewer (original connectome)
# ---------------------------------------------------------------
# • Loads row-0 from Pure_nomad.csv  (untrained genome)
# • Uses the NEW WormSimulationEnv API
# • Records side-by-side arena + connectome activity to MP4
# ---------------------------------------------------------------
import os, cv2, numpy as np
from moviepy import ImageSequenceClip
from typing import List
from Worm_Env.c_worm           import is_food_close          # still needed by env.render()
from Worm_Env.celegan_env      import WormSimulationEnv      # <- your updated env
from Worm_Env.connectome            import WormConnectome         # LIF class from earlier
from util.write_read_txt       import read_arrays_from_csv_pandas
from graphs.connectome_graph   import ConnectomeViewer
import matplotlib.pyplot as plt

# ---------- helper: Matplotlib canvas ➜ BGR frame ----------
def _mpl_fig_to_bgr(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    argb = np.frombuffer(fig.canvas.tostring_argb(), np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(argb[:, :, 1:], cv2.COLOR_RGB2BGR)   # drop alpha

def _side_by_side(arena, net):
    h = arena.shape[0]
    net = cv2.resize(net, (int(net.shape[1] * h / net.shape[0]), h),
                     interpolation=cv2.INTER_AREA)
    return cv2.hconcat([arena, net])

# ------------------- recorder class -------------------------
class GeneticDynVideo:
    def __init__(self,
                 patterns          : List[int],   # env.reset(pattern_type)
                 episodes          : int       = 1,
                 steps_per_episode : int       = 250,
                 tmp_dir           : str       = "tmp_img"):
        self.patterns  = patterns
        self.episodes  = episodes
        self.steps     = steps_per_episode
        self.tmp_dir   = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)

    # ---- helpers --------------------------------------------------
    def _orig_genome(self, csv="Pure_nomad_leaky.csv"):
        base_dir = os.path.dirname(__file__)
        #full_folder = os.path.join(base_dir, "data_full_pentagon")
        full_path = os.path.join(base_dir, csv)
        rows = read_arrays_from_csv_pandas(full_path)
        if not rows:
            raise FileNotFoundError(f"{csv} is empty.")
        return np.asarray(rows[-1], dtype=float)



    @staticmethod
    def _all_names():
        from Worm_Env.weight_dict import dict as wg
        names = set(wg.keys())
        for dst in wg.values(): names.update(dst)
        return sorted(names)

    # ---- main -----------------------------------------------------
    def run(self, out="simulation.mp4", fps=25):
        # 1. build worm + viewer
        worm = WormConnectome(weight_matrix=self._orig_genome(),
                              all_neuron_names=self._all_names())
        env  = WormSimulationEnv(num_worms=1)
        viewer = ConnectomeViewer(
            worm,
            layout="kamada_groups",   # or "groups" / "kamada_kawai"
            spread=0.2,               # global zoom-out factor (default 1.6)
            pulse_size=3.0,           # how much bigger a spiking node gets
            group_gap=1.5,            # horizontal gap (only for grouped layouts)
            color_mode="energy"
        )


        frame_idx = 0
        for pat in self.patterns:
            for _ in range(self.episodes):
                obs = env.reset(pat)                # new API returns obs
                worm.V[:] = 0.0                    # reset potentials

                for _ in range(self.steps):
                    move = worm.move(obs[0, 0], obs[0, 4])  # dist , sees_food
                    obs, _, done = env.step(move, worm_num=0, candidate=None)
                    viewer.step()
                    env.render()
                    arena  = _mpl_fig_to_bgr(env.fig)
                    netvis = _mpl_fig_to_bgr(viewer.fig)
                    frame  = _side_by_side(arena, netvis)

                    cv2.imwrite(os.path.join(self.tmp_dir,
                                             f"frame_{frame_idx:05d}.png"),
                                frame)
                    frame_idx += 1
                    if done:
                        break

        # 2. encode ➜ MP4
        pngs = sorted(p for p in os.listdir(self.tmp_dir) if p.endswith(".png"))
        clip = ImageSequenceClip([os.path.join(self.tmp_dir, p) for p in pngs], fps=fps)
        clip.write_videofile(out, codec="libx264", audio=False)

        for p in pngs: os.remove(os.path.join(self.tmp_dir, p))
        print(f"✓ saved {out}  ({len(pngs)} frames)")

# ----------------------- cli ---------------------------------
if __name__ == "__main__":
    GeneticDynVideo(patterns=[5], episodes=1, steps_per_episode=100).run()
