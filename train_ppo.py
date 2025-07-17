from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from mir_rl_env import MiRRLPathEnv
from xTCP import xTCP
from yTCP import yTCP
import numpy as np
from plot_callback import PlotTrajectoryCallback

# TCP-Pfad laden
tcp_path = list(zip(xTCP(), yTCP()))

# RL-Umgebung instanziieren
env = MiRRLPathEnv(tcp_path)

# Optional: Umgebung validieren
check_env(env, warn=True)

# Callback für Trajektorien-Plot erstellen
eval_env = MiRRLPathEnv(tcp_path)  # separate evaluation environment
callback = PlotTrajectoryCallback(eval_env, tcp_path, plot_freq=5000)

# PPO-Agent konfigurieren
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_mir_log")

# Training starten
model.learn(total_timesteps=500000, callback=callback)

# Modell speichern
model.save("ppo_mir_model")
