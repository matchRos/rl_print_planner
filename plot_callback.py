from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import os

class PlotTrajectoryCallback(BaseCallback):
    def __init__(self, eval_env, tcp_path, plot_freq=5000, save_path="./plots", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.tcp_path = np.array(tcp_path)
        self.plot_freq = plot_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.plot_freq == 0:
            obs, _ = self.eval_env.reset()
            done = False
            trajectory = [obs[2:4]]

            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                trajectory.append(obs[2:4])

            traj = np.array(trajectory)
            plt.figure(figsize=(6, 6))
            plt.plot(self.tcp_path[:, 0], self.tcp_path[:, 1], 'b.-', label="TCP Path")
            plt.plot(traj[:, 0], traj[:, 1], 'g.-', label="MiR Path")
            plt.axis('equal')
            plt.title(f"Policy at step {self.num_timesteps}")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(self.save_path, f"trajectory_step_{self.num_timesteps}.png")
            plt.savefig(fname)
            plt.close()
            if self.verbose:
                print(f"Saved trajectory plot: {fname}")
        return True
