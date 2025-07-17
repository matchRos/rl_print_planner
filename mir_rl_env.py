import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MiRRLPathEnv(gym.Env):
    def __init__(self, tcp_path):
        super().__init__()
        self.tcp_path = tcp_path
        self.n_substeps = 10  # MiR steps per TCP point
        self.t_step = 0.1     # 100 ms per step
        self.v_max = 0.5      # max linear velocity [m/s]
        self.w_max = 1.0      # max angular velocity [rad/s]
        self.ur_offset = np.array([0.3, -0.2])  # UR10 base offset in MiR frame

        self.action_space = spaces.Box(low=np.array([-self.v_max, -self.w_max]),
                                       high=np.array([self.v_max, self.w_max]),
                                       dtype=np.float32)

        # Observation: [x_tcp, y_tcp, x_mir, y_mir, theta_mir]

        tcp_x = [p[0] for p in self.tcp_path]
        tcp_y = [p[1] for p in self.tcp_path]
        x_min, x_max = min(tcp_x) - 5, max(tcp_x) + 5
        y_min, y_max = min(tcp_y) - 5, max(tcp_y) + 5

        obs_low = np.array([x_min, y_min, x_min, y_min, -np.pi], dtype=np.float32)
        obs_high = np.array([x_max, y_max, x_max, y_max, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)


        obs, _ = self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tcp_index = 0
        self.substep = 0
        self.mir_pose = np.array([self.tcp_path[0][0] - 0.8, self.tcp_path[0][1], 0.0])
        return self._get_obs(), {}  # Rückgabe: (observation, info)


    def _get_obs(self):
        idx = min(self.tcp_index, len(self.tcp_path) - 1)
        x_tcp, y_tcp = self.tcp_path[idx]
        x, y, theta = self.mir_pose
        return np.array([x_tcp, y_tcp, x, y, theta], dtype=np.float32)

    def step(self, action):
        v, w = np.clip(action, self.action_space.low, self.action_space.high)
        x, y, theta = self.mir_pose

        dx = v * np.cos(theta) * self.t_step
        dy = v * np.sin(theta) * self.t_step
        dtheta = w * self.t_step
        self.mir_pose += np.array([dx, dy, dtheta])
        self.mir_pose[2] = (self.mir_pose[2] + np.pi) % (2 * np.pi) - np.pi

        # UR-Basis berechnen
        ur_x = x + np.cos(theta) * self.ur_offset[0] - np.sin(theta) * self.ur_offset[1]
        ur_y = y + np.sin(theta) * self.ur_offset[0] + np.cos(theta) * self.ur_offset[1]

        x_tcp, y_tcp = self.tcp_path[self.tcp_index]
        dist = np.linalg.norm([ur_x - x_tcp, ur_y - y_tcp])

        # Reward berechnen
        if dist < 0.5:
            reward = -10 * (0.5 - dist)
        elif dist > 1.1:
            reward = -10 * (dist - 1.1)
        else:
            reward = - (dist - 0.8) ** 2

        self.substep += 1
        terminated = False
        truncated = False
        if self.substep >= self.n_substeps:
            self.substep = 0
            self.tcp_index += 1
            if self.tcp_index >= len(self.tcp_path):
                terminated = True

        return self._get_obs(), reward, terminated, truncated, {}


# Dummy test
from xTCP import xTCP
from yTCP import yTCP

tcp_path = list(zip(xTCP(), yTCP()))
env = MiRRLPathEnv(tcp_path)
obs = env.reset()

trajectory = [obs[2:4]]  # track MiR positions

for _ in range(100):  # simulate 10 TCP points
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    trajectory.append(obs[2:4])
    if done:
        break

# import matplotlib.pyplot as plt
# #import ace_tools as tools

# trajectory = np.array(trajectory)
# tcp = np.array(tcp_path)

# plt.figure()
# plt.plot(tcp[:,0], tcp[:,1], 'bo-', label="TCP Path")
# plt.plot(trajectory[:,0], trajectory[:,1], 'r.-', label="MiR Path (random policy)")
# plt.legend()
# plt.axis('equal')
# plt.title("Initial RL Environment Test")
# plt.xlabel("x [m]")
# plt.ylabel("y [m]")
# plt.grid(True)
# plt.show()
#tools.display_dataframe_to_user(name="Trajectory Debug Output", dataframe=None)
