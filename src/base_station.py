from src.parameters import *
from src.data_structures import Coords3d
import numpy as np
from typing import Tuple, List
from src.drone_agent import DroneAgent, get_successful_transmit_prob
from src.channel_model import CellularUrban3GppUmi
import itertools
from src.math_tools import decision, numpy_object_array
from src.q_manager import QManager
import matplotlib.pyplot as plt

class BaseStation:
    sensing_tasks_locs = None
    last_rewards = None

    def __init__(self, n_sub_channels=DEFAULT_N_SC, n_uavs=DEFAULT_N_UAVS,
                 init_tasks_locations: List[Tuple[float, ...]] = INIT_TASK_LOCATIONS):
        self.coords = Coords3d(0, 0, 0)
        self.n_sc = n_sub_channels
        self.n_uavs = n_uavs
        self.cycle_count = 1

        if init_tasks_locations is None:
            self.sensing_tasks_locs = [Coords3d(np.random.choice(self.q_manager.state_space_x),
                                                np.random.choice(self.q_manager.state_space_y), 0)
                                       for _ in range(self.n_uavs)]
        else:
            assert (self.n_uavs == len(init_tasks_locations))
            self.sensing_tasks_locs = [Coords3d(init_loc[0], init_loc[1], 0) for init_loc in init_tasks_locations]

        self.uavs = [DroneAgent(coords=(self.sensing_tasks_locs[i]) + Coords3d(0, 0, 100).copy(),
                                sensing_task_loc=self.sensing_tasks_locs[i]) for i in
                     range(self.n_uavs)]

        self.q_manager = QManager(self.uavs)

        self.initialize_uavs_locations()

    def perform_cycle(self):
        trajs = self.select_trajectories()
        rewards_probs = self.get_cycle_reward_probabilities()
        cycle_rewards_sample = self.sample_rewards(rewards_probs)
        cycle_rewards = cycle_rewards_sample
        self.q_manager.update_q_values(cycle_rewards, self.cycle_count)
        self.cycle_count += 1
        self.reach_final_destination()
        # TODO: comment out
        for i in range(self.n_uavs):
            q_manager_loc = np.append(self.q_manager.state_spaces_2d[i][self.q_manager.state_idxs[i][0]],
                                      self.q_manager.state_space_z[self.q_manager.state_idxs[i][1]])
            bs_loc = self.uavs[i].coords.np_array()
            assert (np.all(q_manager_loc == bs_loc))

        return cycle_rewards_sample

    def reach_final_destination(self):
        for _uav in self.uavs:
            _uav.coords = _uav.next_location.copy()

    def get_cycle_reward_probabilities(self):
        trajs = self.get_trajectories()
        successful_transmit_probs = get_successful_transmit_prob(trajs)
        successful_sensing_probs = np.array([_uav.get_successful_sensing_probability_cycle() for _uav in self.uavs])
        reward_success_probs = successful_transmit_probs * successful_sensing_probs
        return reward_success_probs

    def sample_rewards(self, reward_success_probs):
        self.last_rewards = np.array([decision(_prob) for _prob in reward_success_probs])
        return self.last_rewards

    def initialize_uavs_locations(self):
        self.q_manager.initialize_uavs_locations_in_state_space()

    def select_trajectories(self):
        destinations = self.q_manager.select_action_in_cycle(self.cycle_count)
        self.set_trajectories(destinations)
        return destinations

    def set_trajectories(self, destinations):
        [_uav.set_next_location(destinations[i]) for i, _uav in enumerate(self.uavs)]

    def set_random_trajectories(self):
        [_uav.set_random_trajectory() for _uav in self.uavs]

    def get_trajectories(self):
        return [_uav.get_trajectory(start_frame_idx=1) for _uav in self.uavs]




if __name__ == '__main__':
    a = BaseStation()
    single_success_count = 0
    N_CYCLES = 3000
    rewards_per_cycle = np.zeros(N_CYCLES)
    suc_count = 0
    for i in range(N_CYCLES):
        print("Cycle:", i, "successes:", suc_count)
        rewards = a.perform_cycle()
        rewards_per_cycle[i] = rewards.sum()
        if np.any(rewards):
            suc_count += 1
        if rewards[0]:
            bla = 5

    rewards_discounted_cumsum = np.zeros(N_CYCLES)
    for i in range(len(rewards_per_cycle)):
        sum = 0
        discount = 1
        for j in range(i, len(rewards_per_cycle)):
            sum+=rewards_per_cycle[j] *discount
            discount *= DISCOUNT_RATIO
        rewards_discounted_cumsum[i]=sum

    plt.plot(rewards_discounted_cumsum)
    plt.show()
    # a.select_trajectories()
    #
    # a = a.q_manager
    # a.get_2d_actions_flags(2)
    # q_idxs, actions = a.get_current_q_idxes_and_actions(2)
    # q_values = a.get_q_values_from_idxs(2, q_idxs)
    # q_values[0] = 0.8
    # q_values[4] = 0.9
    # action = a.select_action(0.2, q_values, actions, q_idxs)
