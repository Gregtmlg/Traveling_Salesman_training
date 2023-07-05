import os
import time
from time import sleep
from termcolor import colored
import matplotlib.pyplot as plt

from stable_baselines3 import TD3, PPO, A2C
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from traveling_salesman_env import TravelingSalesmanEnv

env = TravelingSalesmanEnv(max_steps=500, type_of_use="simple", nb_goals=5, action_k=0.5, step_k=2)

env.reset()
done = False
for episode in range(10):
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        print(env.bluerov.get_battery())
        print("reward = ", reward)
    env.reset()
    done = False
    print("##################### RESET #####################")

env.close()