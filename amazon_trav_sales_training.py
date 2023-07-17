import os
import time
from time import sleep
from termcolor import colored
import matplotlib.pyplot as plt

from stable_baselines3 import TD3, PPO, A2C
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from amazon_env import TSPEasyEnv, TSPMediumEnv, TSPHardEnv

# Set the parameters for the implementation
MAX_TIMESTEPS = 50000000  # Maximum number of steps to perform
SAVE_TIMESTEPS = 10000000 # save model every SAVE_TIMESTEPS step
models_dir = "models/PPO_aws_hard"
logs_dir = "logs_aws" 

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

env = TSPHardEnv()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir, device="auto")
iteration = 0
while SAVE_TIMESTEPS * iteration < MAX_TIMESTEPS:
    iteration += 1
    model.learn(total_timesteps=SAVE_TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_aws_hard")
    model.save(f"{models_dir}/traveling_salesman_aws_{SAVE_TIMESTEPS * iteration}")
env.close()