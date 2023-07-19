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

# Set the parameters for the implementation
MAX_TIMESTEPS = 100000000  # Maximum number of steps to perform
SAVE_TIMESTEPS = 10000000 # save model every SAVE_TIMESTEPS step
models_dir = "models/PPO_jeudi"
logs_dir = "logs_jeudi" 

action_k = [0, 0.5, 1, 1.5, 2, 2.5, 3]
step_k = [0, 0.5, 1, 1.5, 2]

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# for ak in action_k:
#     for sk in step_k:
env = TravelingSalesmanEnv(max_steps=250, type_of_use="simple", nb_goals=5, action_k=0.5, step_k=1)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir, device="auto")
iteration = 0
while SAVE_TIMESTEPS * iteration < MAX_TIMESTEPS:
    iteration += 1
    model.learn(total_timesteps=SAVE_TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_ak_{str(1).translate(str.maketrans('', '', '.'))}_sk_{1}")
    model.save(f"{models_dir}/travel_sales_{SAVE_TIMESTEPS * iteration}_ak_{str(1).translate(str.maketrans('', '', '.'))}_sk_{1}")
env.close()


# plt.figure()
# plt.plot(env.reward_history)
# plt.show()
# env.close()