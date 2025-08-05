import numpy as np
import ray
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import muscles, muscleList, mLeft, mRight, all_neuron_names
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from util.write_read_txt import read_arrays_from_csv_pandas
import matplotlib.cm as cm  # For color mapping
import cv2
import math
from moviepy import ImageSequenceClip

from Worm_Env.celegan_env import WormSimulationEnv  # Your simulation environment

# MoviePy-based functions for compiling images to video
def compile_images_to_video(image_folder, output_video_path, fps=10):
    def get_all_images(image_folder):
        all_images = []
        for img in sorted(os.listdir(image_folder)):
            if img.endswith('.png'):
                img_path = os.path.join(image_folder, img)
                all_images.append(img_path)
        return all_images

    def delete_all_images(image_folder):
        for img in sorted(os.listdir(image_folder)):
            if img.endswith('.png'):
                img_path = os.path.join(image_folder, img)
                os.remove(img_path)
                print(f"Deleted image: {img_path}")

    unique_images = get_all_images(image_folder)
    clip = ImageSequenceClip(unique_images, fps=fps)
    clip.write_videofile(output_video_path, codec='libx264')
    delete_all_images(image_folder)
def get_frame_from_env(sim_env):
            sim_env.ax.clear()
            worm = sim_env.worms[0]
            sim_env.ax.plot(worm.position[0], worm.position[1], 'ro')
            sim_env.ax.plot([worm.position[0], worm.position[0] + 100 * np.cos(worm.facing_dir)],
                             [worm.position[1], worm.position[1] + 100 * np.sin(worm.facing_dir)], 'b-')
            for f in sim_env.food:
                if math.hypot(worm.position[0]-f[0], worm.position[1]-f[1]) < sim_env.range:
                    sim_env.ax.plot(f[0], f[1], 'yo')
                else:
                    sim_env.ax.plot(f[0], f[1], 'bo')
            sim_env.ax.set_xlim(0, sim_env.dimx)
            sim_env.ax.set_ylim(0, sim_env.dimy)
            sim_env.fig.canvas.draw()
            width, height = sim_env.fig.canvas.get_width_height()
            # Use tostring_argb() and convert ARGB to RGB
            buf = sim_env.fig.canvas.tostring_argb()
            buf = np.frombuffer(buf, dtype=np.uint8)
            buf = buf.reshape((height, width, 4))
            rgb = buf[:, :, 1:]  # Drop the alpha channel
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return img
class Genetic_Dyn_Video:
    def __init__(self, population_size, pattern=[3,4,5,6,7,8,9], total_episodes=10, training_interval=25, tmp_img_folder="tmp_img"):
        """
        Note: population_size is overridden to 1 since we load one candidate per food pattern.
        pattern: List of food pattern numbers. For each food pattern, the candidate genome is loaded
                 from its own file "array{pattern}.csv" (e.g. food pattern 3 uses "array3.csv").
        """
        self.population_size = 1
        self.food_patterns = pattern
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.tmp_img_folder = tmp_img_folder
        if not os.path.exists(self.tmp_img_folder):
            os.makedirs(self.tmp_img_folder)

    def run_video_simulation(self, env, output_video="food_collection_video.mp4", fps=25):
        """
        For each food pattern in self.food_patterns:
          - Loads the candidate genome from the corresponding file "array{pattern}.csv"
            (using the final genome from that file).
          - Creates a simulation environment.
          - Runs a synchronized simulation loop (total_steps = total_episodes * training_interval)
            where each environment is updated using its candidate’s move.
          - Captures a frame (via a render-like routine) from each environment.
          - Arranges the captured frames into a 2×3 montage and saves them as PNG files.
          - Finally, compiles the images into a video using MoviePy.
        """

        # For each food pattern, load the candidate from its own CSV file.
        candidates = []
        for pat in self.food_patterns:
            filename = rf"array{pat}.csv"
            all_genomes = read_arrays_from_csv_pandas(filename)
            if len(all_genomes) == 0:
                raise ValueError(f"No genomes found in {filename}")
            # Use the final genome (last row) as candidate.
            final_genome = np.array(all_genomes[-1], dtype=float)
            candidate = WormConnectome(weight_matrix=final_genome, all_neuron_names=all_neuron_names)
            candidates.append(candidate)

        # Create one simulation environment per food pattern.
        env_list = []
        for pat in self.food_patterns:
            new_env = WormSimulationEnv()
            new_env.reset(pat, num_food=36)
            env_list.append(new_env)

        # Function to capture a frame (mimicking render without displaying)


        total_steps = self.total_episodes * self.training_interval
        done_flags = [False] * len(env_list)
        last_frames = [None] * len(env_list)

        # Synchronized simulation loop.
        for step in tqdm(range(total_steps), desc="Simulating Video"):
            frame_list = []
            for i, sim_env in enumerate(env_list):
                if not done_flags[i]:
                    observation = sim_env._get_observations()
                    movement = candidates[i].move(
                        observation[0][0],
                        sim_env.worms[0].sees_food,
                        mLeft, mRight, muscleList, muscles
                    )
                    _, reward, done = sim_env.step(movement, 0, candidates[i])
                    if done:
                        done_flags[i] = True
                        last_frames[i] = get_frame_from_env(sim_env)

                # use last captured frame once an environment is done
                frame_img = last_frames[i] if done_flags[i] else get_frame_from_env(sim_env)
                frame_list.append(frame_img)

            # ensure exactly 6 frames (2 rows × 3 cols)
            if len(frame_list) > 6:
                frame_list = frame_list[:6]
            else:
                h, w, _ = frame_list[0].shape
                blank = np.zeros((h, w, 3), dtype=np.uint8)
                while len(frame_list) < 6:
                    frame_list.append(blank)

            # build the 2 × 3 montage
            row1 = cv2.hconcat(frame_list[:3])
            row2 = cv2.hconcat(frame_list[3:6])
            montage = cv2.vconcat([row1, row2])

            # ── append a horizontal viridis gradient bar encoding time ──
            bar_h, bar_w = 20, montage.shape[1]
            gradient  = np.linspace(0, 1, bar_w, dtype=np.float32)           # 0 → 1
            viridis   = cm.get_cmap("viridis")
            bar_rgb   = (viridis(gradient)[:, :3] * 255).astype(np.uint8)    # (W,3)
            bar_rgb   = np.tile(bar_rgb, (bar_h, 1, 1))                      # (H,W,3)
            bar_bgr   = cv2.cvtColor(bar_rgb, cv2.COLOR_RGB2BGR)

            montage_with_bar = cv2.vconcat([montage, bar_bgr])

            # save frame to disk
            img_filename = os.path.join(self.tmp_img_folder, f"frame_{step:05d}.png")
            cv2.imwrite(img_filename, montage_with_bar)

        # ───────────── compile PNGs → video, then clean-up ─────────────
        compile_images_to_video(self.tmp_img_folder, output_video, fps=fps)
        ray.shutdown()
        plt.close('all')


    def run_image_simulation(
            self,
            out_img: str = "nomad_vs_hybrid_trajectories.svg") -> None:
        """
        4 × 3 figure:
        rows 0–1 → PURE NOMAD
        rows 2–3 → NOMAD HYBRID
        Each panel shows the worm trajectory and “FOOD EATEN”.
        """
        # ── imports ───────────────────────────────────────────────
        import os, numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import Normalize
        from tqdm import tqdm

        csv_sets = [("array",     "PURE NOMAD"),
                    ("EVO_NOMAD", "NOMAD HYBRID")]

        # ── 1. load data ──────────────────────────────────────────
        cands, envs, food_pts = [], [], []
        for csv_prefix, _ in csv_sets:
            for pat in self.food_patterns:                       # expects 3 patterns
                csv = os.path.join("24hr", f"{csv_prefix}{pat}.csv")
                arr = read_arrays_from_csv_pandas(csv)
                if not arr:
                    raise FileNotFoundError(f"{csv} missing / empty")
                cands.append(WormConnectome(np.asarray(arr[-1], float),
                                            all_neuron_names))
                env = WormSimulationEnv();  env.reset(pat, num_food=36)
                envs.append(env);  food_pts.append(env.food.copy())

        # ── 2. simulate ──────────────────────────────────────────
        n_steps      = self.total_episodes * self.training_interval
        trajectories = [[] for _ in envs];  food_eaten = [0] * len(envs)

        for _ in tqdm(range(n_steps), desc="Simulating"):
            for i, (env, cand) in enumerate(zip(envs, cands)):
                obs = env._get_observations()
                trajectories[i].append(obs[0][1:3])
                mv = cand.move(obs[0][0], env.worms[0].sees_food,
                            mLeft, mRight, muscleList, muscles)
                _, r, _ = env.step(mv, 0, cand)
                food_eaten[i] += r

        # ── 3. plot ──────────────────────────────────────────────
        fig, axes = plt.subplots(4, 3, figsize=(17, 20), squeeze=False)
        fig.subplots_adjust(wspace=0.00)                      # tighter columns
        axes = axes.flatten()

        cmap = cm.get_cmap("viridis")
        norm = Normalize(0, max(len(t) for t in trajectories) - 1)

        for idx, (ax, traj, foods, eaten) in enumerate(
                zip(axes, trajectories, food_pts, food_eaten)):

            ax.set_xlim(0, 1600);  ax.set_ylim(0, 1200);  ax.set_aspect("equal")
            row, col = divmod(idx, 3)

            # axis labels & ticks
            if col == 0:
                ax.set_ylabel("Y POSITION", fontsize=28)
                ax.tick_params(axis="y", labelsize=24)
            else:
                ax.set_yticks([])
            if row == 3:
                ax.set_xlabel("X POSITION", fontsize=28)
                ax.tick_params(axis="x", labelsize=24)
            else:
                ax.set_xticks([])

            # food dots
            x_f, y_f = foods.T
            ax.plot(x_f, y_f, "ro", ms=7)

            # trajectory
            traj = np.asarray(traj)
            for j in range(1, len(traj)):
                ax.plot(traj[j-1:j+1, 0], traj[j-1:j+1, 1],
                        lw=3.2, color=cmap(norm(j)))

            # food‑eaten box
            ax.text(0.03, 0.06, f"FOOD EATEN: {int(eaten)}",
                    transform=ax.transAxes, fontsize=26,
                    bbox=dict(boxstyle="round",
                            facecolor="wheat",
                            alpha=0.85))

        # ── 4. colour‑bar & title (fixed positions) ───────────────
        sm = cm.ScalarMappable(norm=norm, cmap=cmap);  sm.set_array([])
        cbar_ax = fig.add_axes([0.30, 0.945, 0.70, 0.02])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=24)

        fig.text(0.65, 0.992, "TIME",
                ha="center", va="top", fontsize=38, weight="bold")
        # ── 5. pack axes tightly, then add row labels dynamically ─
        plt.tight_layout(rect=[0, 0, 1.21, 0.93])        # no fixed bottom margin

        # mid‑heights for the two row pairs, after tight‑layout
        top_mid = 0.5 * (axes[0].get_position().y1 + axes[3].get_position().y0)
        bot_mid = 0.5 * (axes[6].get_position().y1 + axes[9].get_position().y0)

        label_kw = dict(transform=fig.transFigure,
                        fontsize=30, rotation=90,
                        ha="left", va="center", weight="bold")

        fig.text(0.002, top_mid, "rE-NOMAD",   **label_kw)
        fig.text(0.002, bot_mid, "mE-NOMAD", **label_kw)

        # ── 6. save (crop all excess) ─────────────────────────────
        fig.savefig(out_img, dpi=300,
                    bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        print(f"Saved → {out_img}")