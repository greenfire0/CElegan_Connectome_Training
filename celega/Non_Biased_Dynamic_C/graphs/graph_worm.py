# genetic_dyn_video.py
# ──────────────────────────────────────────────────────────────────────
#  Visualise the trajectory of a *single* trained worm:
#  • Loads the most-recent genome from Pure_nomad.csv
#  • Runs it in WormSimulationEnv for the chosen cue pattern
#  • Captures every frame through env.render()
#  • Encodes frames → MP4 and deletes the temporary PNGs
# ──────────────────────────────────────────────────────────────────────
import os
import cv2
import numpy as np
from tqdm import tqdm
from moviepy import ImageSequenceClip
from typing import List
from Worm_Env.connectome  import WormConnectome
from Worm_Env.celegan_env import WormSimulationEnv
from util.write_read_txt  import read_arrays_from_csv_pandas
from Worm_Env.weight_dict import muscles,muscleList,mLeft,mRight,all_neuron_names


# ──────────────────────────────────────────────────────────────────────
# helper: get one BGR frame from Matplotlib canvas
# ──────────────────────────────────────────────────────────────────────
def get_frame_from_env(env: WormSimulationEnv) -> np.ndarray:
    env.render()                                   # draw current state
    canvas = env.fig.canvas
    w, h   = canvas.get_width_height()

    # Canvas delivers ARGB bytes → drop alpha, swap to BGR
    argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    argb = argb.reshape((h, w, 4))
    rgb  = argb[:, :, 1:]                          # strip alpha
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────────────────────────────
#  Main class
# ──────────────────────────────────────────────────────────────────────
class Genetic_Dyn_Video:
    """
    Visualise one trained worm on a single cue pattern.

    Parameters
    ----------
    pattern : int
        Value passed to `env.reset(pattern)` — e.g. 0 for one-pulse task.
    total_episodes : int
        Episodes to play back.
    training_interval : int
        Timesteps in each episode (must match env.episode_len).
    tmp_img_folder : str
        Directory used for intermediate PNG frames.
    """

    def __init__(
        self,
        pattern: List[int]        = [1],           # default: one-pulse task
        total_episodes: int = 1,
        training_interval: int = 100,
        tmp_img_folder: str = "tmp_img"
    ):
        self.pattern           = pattern
        self.total_episodes    = total_episodes
        self.training_interval = training_interval
        self.tmp_img_folder    = tmp_img_folder
        os.makedirs(self.tmp_img_folder, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    #  Run + encode
    # ──────────────────────────────────────────────────────────────
    def run_video_simulation(self,
                            output_video: str = "simulation.mp4",
                            fps: int = 25) -> None:

        # 1 ─ load last saved genome
        rows = read_arrays_from_csv_pandas("Pure_nomad.csv")
        if not rows:
            raise FileNotFoundError("Pure_nomad.csv is empty.")
        genome = np.asarray(rows[-1], dtype=float)
        worm   = WormConnectome(weight_matrix=genome, all_neuron_names=all_neuron_names)


        # 2 ─ build environment
        env = WormSimulationEnv()

        # 3 ─ simulate and save PNG frames
        os.makedirs(self.tmp_img_folder, exist_ok=True)
        frame_idx = 0
        for pattern in self.pattern:                            # e.g. [0]
            for _ in range(self.total_episodes):
                env.reset(1)
                
                observation = env._get_observations()
                
                for s in range(self.training_interval):
                    movement    = worm.move(observation[0], observation[4])
                    observation,_,_ = env.step(movement,0,s)

                    frame = get_frame_from_env(env)              # BGR uint8
                    cv2.imwrite(
                        os.path.join(self.tmp_img_folder,
                                    f"frame_{frame_idx:05d}.png"),
                        frame
                    )
                    frame_idx += 1

        # 4 ─ compile → MP4
        pngs = sorted(p for p in os.listdir(self.tmp_img_folder)
                    if p.endswith(".png"))
        clip = ImageSequenceClip(
            [os.path.join(self.tmp_img_folder, p) for p in pngs],
            fps=fps
        )
        clip.write_videofile(output_video, codec="libx264", audio=False)

        # 5 ─ clean up
        for p in pngs:
            os.remove(os.path.join(self.tmp_img_folder, p))
        print(f"✔ Saved {output_video} — {len(pngs)} frames, folder cleaned.")