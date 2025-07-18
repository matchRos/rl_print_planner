from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from mir_rl_env import MiRRLPathEnv
from xTCP import xTCP
from yTCP import yTCP
import numpy as np
from plot_callback import PlotTrajectoryCallback

def make_env(tcp_path):
    def _init():
        return MiRRLPathEnv(tcp_path)
    return _init

if __name__ == "__main__":
    # TCP-Pfad laden
    tcp_path = list(zip(xTCP(), yTCP()))

    # SubprocVecEnv erstellen
    n_envs = 12
    env = SubprocVecEnv([make_env(tcp_path) for _ in range(n_envs)])

    # Separates eval_env + Plot Callback
    eval_env = MiRRLPathEnv(tcp_path)
    callback = PlotTrajectoryCallback(eval_env, tcp_path)

    # Optional: check_env für Debug
    check_env(eval_env, warn=True)

    # PPO trainieren
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_mir_log")
    model.learn(total_timesteps=5000000, callback=callback)

    # Modell speichern
    model.save("ppo_mir_model")
