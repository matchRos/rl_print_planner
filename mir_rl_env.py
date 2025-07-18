import numpy as np
import gymnasium as gym
from gymnasium import spaces
from xMIR import xMIR
from yMIR import yMIR
from scipy.spatial import cKDTree

class MiRRLPathEnv(gym.Env):
    def __init__(self, tcp_path):
        super().__init__()
        self.tcp_path = tcp_path
        self.n_substeps = 10  # MiR steps per TCP point
        self.t_step = 0.1     # 100 ms per step
        self.v_max = 0.4      # max linear velocity [m/s]
        self.w_max = 0.8      # max angular velocity [rad/s]
        self.ur_offset = np.array([0.3, -0.2])  # UR10 base offset in MiR frame
        self.tcp_tree = cKDTree(self.tcp_path)  # nur xy-Koordinaten
        offset_path = list(zip(xMIR(), yMIR()))
        self.offset_tree = cKDTree(offset_path)

        self.action_space = spaces.Box(low=np.array([0, -self.w_max]),
                                       high=np.array([self.v_max, self.w_max]),
                                       dtype=np.float32)

        # Observation: [x_tcp, y_tcp, x_mir, y_mir, theta_mir]

        tcp_x = [p[0] for p in self.tcp_path]
        tcp_y = [p[1] for p in self.tcp_path]
        x_min, x_max = min(tcp_x) - 5, max(tcp_x) + 5
        y_min, y_max = min(tcp_y) - 5, max(tcp_y) + 5

        obs_low = np.array([x_min, y_min, x_min, y_min, -np.pi], dtype=np.float32)
        obs_high = np.array([x_max, y_max, x_max, y_max, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([
                0.0, 0.0,            # x_tcp, y_tcp
                0.0, 0.0, -np.pi,    # x_mir, y_mir, theta
                -1.0, -1.0,          # prev v, w
                0.0                  # dist_offset
            ], dtype=np.float32),
            high=np.array([
                100.0, 100.0,        # x_tcp, y_tcp
                100.0, 100.0, np.pi, # x_mir, y_mir, theta
                1.0, 1.0,            # prev v, w
                2.0                  # dist_offset
            ], dtype=np.float32)
        )
        self.prev_action = np.array([0.0, 0.0])


        obs, _ = self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        x_start = xMIR()[0]
        y_start = yMIR()[0]
        self.tcp_index = 0
        self.substep = 0
        self.prev_action = np.array([0.0, 0.0])
        self.mir_pose = np.array([x_start, y_start, 0.0])
        return self._get_obs(), {}

    def _get_obs(self):
        idx = min(self.tcp_index, len(self.tcp_path) - 1)
        x_tcp, y_tcp = self.tcp_path[idx]
        x, y, theta = self.mir_pose
        ur_x = x + np.cos(theta) * self.ur_offset[0] - np.sin(theta) * self.ur_offset[1]
        ur_y = y + np.sin(theta) * self.ur_offset[0] + np.cos(theta) * self.ur_offset[1]
        dist_offset, _ = self.offset_tree.query([ur_x, ur_y])

        obs = np.array([
            x_tcp, y_tcp,        # TCP-Punkt absolut
            x, y, theta,         # MiR Pose
            *self.prev_action,   # vorherige Aktion (v, w)
            dist_offset          # Abstand zur Offset-Kontur
        ], dtype=np.float32)
        return obs 

    def step(self, action):

        # Reward und Termination
        info = {"killed_by_distance": False}

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
        
        # Berechne Abstand
        ur_xy = np.array([ur_x, ur_y])
        nearest_dist, nearest_idx = self.tcp_tree.query(ur_xy)

        dist = np.linalg.norm([ur_x - x_tcp, ur_y - y_tcp])

        # --- Abstand zum TCP-Pfad (Kill-Bedingung) ---
        ur_xy = np.array([ur_x, ur_y])
        nearest_dist_tcp, _ = self.tcp_tree.query(ur_xy)
        if nearest_dist_tcp < 0.5 or dist > 1.1:
            reward = -1.0  # skaliert
            terminated = True
            info["killed_by_distance"] = True
            return self._get_obs(), reward, True, False, info

        # --- Winkel zum TCP-Pfad ---
        dx = x_tcp - x
        dy = y_tcp - y
        x_rel = np.cos(-theta) * dx - np.sin(-theta) * dy
        y_rel = np.sin(-theta) * dx + np.cos(-theta) * dy
        angle = np.arctan2(y_rel, x_rel)  # in MiR-Frame
        angle_error = abs(angle + np.pi / 2)
        angle_reward = max(0.0, 1.0 - angle_error / (np.pi / 2))  # ∈ [0, 1]

        # --- Abstand zur Offset-Kontur (positiv zentriert auf 0.6 m) ---
        mir_xy = self.mir_pose[:2]
        nearest_dist_offset, _ = self.offset_tree.query(mir_xy)
        offset_reward = 1.0 - abs(nearest_dist_offset) / 0.6  # [0..1], max bei 0.6 m
        offset_reward = max(0.0, offset_reward)  # keine Strafe außerhalb

        # --- Glättungsbestrafung für Aktion (Delta v, w) ---
        delta = np.abs(action - self.prev_action)
        smooth_penalty = - (0.5 * delta[0]**2 + 0.1 * delta[1]**2)  # weiche Gewichtung
        self.prev_action = action

        # --- TCP-Abstand (weiche Belohnung für 0.9 m) ---
        tcp_reward = 1.0 - abs(nearest_dist_tcp - 0.9) / 0.4  # max bei 0.9, null bei Grenze
        tcp_reward = max(0.0, tcp_reward)

        # --- Fortschritt (z. B. je TCP-Schritt) ---
        progress_reward = 0.5 #10.0 * (self.tcp_index / len(self.tcp_path))

        # --- Gesamtreward ---
        reward = (
            0.0 * tcp_reward +
            1.0 * offset_reward +
            0.0 * progress_reward +
            0.0 * smooth_penalty +  # ist negativ
            0.0 * angle_reward
        )   

        #print(nearest_dist_offset)
        #äprint(offset_reward)
        #print(f"TCP: {tcp_reward:.2f} | Offset: {offset_reward:.2f} | Smooth: {smooth_penalty:.2f} | Progress: {progress_reward:.2f} → Total: {reward:.2f}")




        self.substep += 1
        self.prev_action = action  # update für nächsten Schritt
        terminated = False
        truncated = False
        if self.substep >= self.n_substeps:
            self.substep = 0
            self.tcp_index += 1
            if self.tcp_index >= len(self.tcp_path):
                terminated = True

        return self._get_obs(), reward, terminated, truncated, info


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
