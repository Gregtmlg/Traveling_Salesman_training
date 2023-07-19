import os
import time
from time import sleep
from termcolor import colored
import matplotlib.pyplot as plt

from stable_baselines3 import TD3, PPO, A2C
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from amazon_env import TSPEasyEnv, TSPMediumEnv, TSPHardEnv, TSP33Env
from amazon_env_battery import TSPEasyBatteryEnv

models_dir = "models/PPO_easy_battery"
logs_dir = "logs_aws"

env = TSPEasyBatteryEnv()

model = PPO.load(f"{models_dir}/traveling_salesman_aws_5000000_battery.zip")


for episode in range(10):
    obs = env.reset()
    done = False
    print(env.battery_eff)
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        time.sleep(0.5)
        # print(env.bluerov.get_battery())
        # print("reward = ", reward)
env.close()