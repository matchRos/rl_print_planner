from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from mir_rl_env import MiRRLPathEnv
from xTCP import xTCP
from yTCP import yTCP
import numpy as np

# TCP-Pfad laden
tcp_path = list(zip(xTCP(), yTCP()))

# RL-Umgebung instanziieren
env = MiRRLPathEnv(tcp_path)

# Optional: Umgebung validieren
check_env(env, warn=True)

# PPO-Agent konfigurieren
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_mir_log")

# Training starten
model.learn(total_timesteps=50000)

# Modell speichern
model.save("ppo_mir_model")
