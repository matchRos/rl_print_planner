from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import os

class PlotTrajectoryCallback(BaseCallback):
    def __init__(self, eval_env, tcp_path, save_path="./plots", verbose=1):
        super().__init__(verbose)
        self.best_reward = -np.inf
        self.eval_env = eval_env
        self.tcp_path = np.array(tcp_path)
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:

        if self.n_calls % 200 != 0:
            return True

        obs, _ = self.eval_env.reset()
        done = False
        trajectory = [obs[2:4]]
        killed = False
        episode_reward = 0
        step_count = 0

        while not done:
            action, _ = self.model.predict(obs)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            trajectory.append(obs[2:4])
            episode_reward += reward
            if info.get("killed_by_distance", False):
                killed = True
            step_count += 1

        if killed and episode_reward > self.best_reward:
            self.best_reward = episode_reward
            traj = np.array(trajectory)
            # Letzter erreichter TCP-Punkt
            last_tcp_idx = self.eval_env.tcp_index
            last_tcp = self.tcp_path[min(last_tcp_idx, len(self.tcp_path) - 1)]
            plt.figure(figsize=(6, 6))
            plt.plot(self.tcp_path[:, 0], self.tcp_path[:, 1], 'b.-', label="TCP Path")
            plt.plot(traj[:, 0], traj[:, 1], 'r.-', label="Best Killed MiR Path")
            plt.plot(last_tcp[0], last_tcp[1], 'kx', markersize=10, label="Last TCP Target")
            plt.axis('equal')
            plt.title(f"New Best (kill) Reward: {episode_reward:.1f} @ step {self.num_timesteps}")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(self.save_path, f"best_kill_plot_{self.num_timesteps}.png")
            plt.savefig(fname)
            plt.close()
            if self.verbose:
                print(f"New best killed episode (reward {episode_reward:.1f}) → saved {fname}")
        return True


