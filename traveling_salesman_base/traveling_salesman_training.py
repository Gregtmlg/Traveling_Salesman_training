import os
import time
from time import sleep
from termcolor import colored
import matplotlib.pyplot as plt

from stable_baselines3 import TD3, PPO
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from traveling_salesman_env import TravelingSalesmanEnv

# Set the parameters for the implementation
max_timesteps = 3000000  # Maximum number of steps to perform
models_dir = "models/PPO"
logs_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

print(colored("INFO : Gym Environment creation", 'yellow'))
env = TravelingSalesmanEnv(max_steps=256, type_of_use="simple", nb_goals=5)
print(colored("INFO : Gym Environment created", 'yellow'))
start = time.time()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)
print(colored("INFO : PPO model created", 'yellow'))
model.learn(total_timesteps=max_timesteps, reset_num_timesteps=False, tb_log_name="PPO")
model.save(f"{models_dir}/traveling_salesman")
end = time.time()

print(colored("Temps de process : " + str(end - start), 'green'))

plt.figure()
plt.plot(env.reward_history)
plt.show()