import gym
import numpy as np
from mobile import Bluerov
from ground_state import GroundState

class TravelingSalesmanEnv(gym.Env):
    """
        Custom environemment that follows gym interface. This environnement is created to solve traveling salesman problem with DRL. It is going to be the base for a more complex scenario.
    """

    def __init__(self, max_steps=200, type_of_use="simple", nb_goals=5):
        gym.Env.__init__(self)
        self.bluerov = Bluerov(type_of_use)
        self.ground = GroundState(nb_goals)

        self.nb_goals = nb_goals
        self.path_history = []
        self.step_max = max_steps
        self.reward_history = []
        self.episode_reward = 0

        self.init_rov_position = np.zeros(3)
        self.goals = np.zeros((nb_goals, 4))
        self.o_time = []
        self.step_counter = 0
        self.step_counter_global = 0
        self.rov_in_base = None

        x_min, x_max, y_min, y_max, z_min, z_max = self.ground.get_parameters()
        rov_x_min = [x_min]
        rov_x_max = [x_max]
        rov_y_min = [y_min]
        rov_y_max = [y_max]
        rov_z_min = [z_min]
        rov_z_max = [z_max]

        goal_min = [x_min, y_min, z_min, 0] * nb_goals
        goal_max = [x_max, y_max, z_max, 1] * nb_goals

        x_init_min = [x_min]
        x_init_max = [x_max]
        y_init_min = [y_min]
        y_init_max = [y_max]
        z_init_min = [z_min]
        z_init_max = [z_max]

        rov_in_base_min = [0]
        rov_in_base_max = [1]

        step_to_reach_goals_min = [0] * nb_goals
        step_to_reach_goals_max = [max_steps] * nb_goals
        nb_steps_min = [0]
        nb_steps_max = [max_steps]
        step_left_min = [0]
        step_left_max = [max_steps]

        self.action_space =  gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
                low=np.array(
                    rov_x_min
                    + rov_y_min
                    + rov_z_min
                    + goal_min
                    + x_init_min
                    + y_init_min
                    + z_init_min
                    + rov_in_base_min
                    + step_to_reach_goals_min
                    + nb_steps_min
                    + step_left_min
            ), 
            high=np.array(
                    rov_x_max
                    + rov_y_max
                    + rov_z_max
                    + goal_max
                    + x_init_max
                    + y_init_max
                    + z_init_max
                    + rov_in_base_max
                    + step_to_reach_goals_max
                    + nb_steps_max
                    + step_left_max
            ), shape=(34,),dtype=np.float32)

    def reset(self):
        self.reward_history.append(self.episode_reward)
        self.__reset_parameters()
        self.goals = self.ground.reset_goals()
        self.init_rov_position = self.ground.reset_init_position()
        self.bluerov.move_to(self.init_rov_position, True)
        self.path_history.append(self.init_rov_position.tolist())
        observations = self.get_observations()
        return observations
        

    def step(self, action):
        done = False
        reward_before_action = self.__compute_reward()
        current_position = self.bluerov.get_current_position()
        coordinates = self.ground.action_to_waypoint(round(action), self.path_history[-1])
        check_wpnt_inbound = self.ground.is_wpnt_inbound(coordinates)
        if check_wpnt_inbound:
            self.path_history.append(coordinates.tolist())
            self.bluerov.move_to(coordinates)
        else:
            done = True

        self.step_counter += 1
        self.step_counter_global += 1
        if self.step_counter == self.step_max:
            done = True

        
        observations = self.get_observations()
        reward = self.__compute_reward() - reward_before_action
        if (np.sum(self.goals[:,3]) == self.nb_goals) and self.rov_in_base:
            done = True
            reward += self.step_max * 0.1

        self.episode_reward += reward

        info = {}
        
        return observations, reward, done, info
    
    def get_observations(self):
        self.current_position = self.bluerov.get_current_position()
        if len(self.path_history) > 2:
            self.goals = self.ground.update_goal_reached(self.path_history[-1], self.path_history[-2])
        self.rov_in_base = int(
            (self.current_position[0] == self.init_rov_position[0]) and (self.current_position[1] == self.init_rov_position[1])
        )
        self.__update_goal_counter()
        observations = {
        'current_position' : self.current_position,
        'goals' : self.goals,
        'initial_position' : self.init_rov_position,
        'rov_in_base' : np.array([self.rov_in_base]),
        'step_to_reach_goals' : np.array(self.o_time),
        'nb_steps' : np.array([self.step_counter]),
        'step_left' : np.array([self.step_max - self.step_counter]),
        }

        observations_flat = np.concatenate([obs.flatten() for obs in observations.values()])
        return observations_flat

    def __compute_reward(self):
        return (
            np.sum(self.goals[:,3] * self.step_max / (np.asarray(self.o_time) + 0.0001))
            - self.step_counter
        )

    def __reset_parameters(self):
        self.o_time = [0] * self.nb_goals
        self.rov_in_base = 1
        self.step_counter = 0
        self.episode_reward = 0
        pass

    def __step_logger(self):
        if self.step_counter_global % 500:
          print("Learning step progress : " + str(self.step_counter_global))

    def __update_goal_counter(self):
        for ix in range(self.nb_goals):
            if self.goals[ix,3] == 0:
                self.o_time[ix] += 1