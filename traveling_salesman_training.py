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
max_timesteps = 1500000  # Maximum number of steps to perform
models_dir = "models/PPO"
logs_dir = "logs" 

action_k = [0.1, 0.5, 1, 1.5, 2, 2.5, 3]
step_k = [3, 4, 5]

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

for ak in action_k:
    for sk in step_k:
        env = TravelingSalesmanEnv(max_steps=500, type_of_use="simple", nb_goals=5, action_k=ak, step_k=sk)
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir, device="auto")
        # model.set_parameters(f"{models_dir}/traveling_salesman.zip")
        model.learn(total_timesteps=max_timesteps, reset_num_timesteps=False, tb_log_name=f"PPO_ak_{ak}_sk_{sk}")
        model.save(f"{models_dir}/traveling_salesman_ak_{str(ak).translate(str.maketrans('', '', '.'))}_sk_{sk}")
        env.close()


# plt.figure()
# plt.plot(env.reward_history)
# plt.show()
# env.close()