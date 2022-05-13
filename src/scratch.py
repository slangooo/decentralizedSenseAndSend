# from src.parameters import DEFAULT_N_SC, DEFAULT_N_UAVS, AREA_DIMS, TF_FRAME_DRUATION, \
#     TB_BEACON_LENGTH, TS_SENSING_LENGTH, TC_CYCLE_LENGTH, TU_TRANSMISSION_LENGTH, SPATIAL_POINT_STEP, \
#     MAXIMUM_DRONE_HEIGHT, MINIMUM_DRONE_HEIGHT
# from src.data_structures import Coords3d
# import numpy as np
# from typing import Tuple, List
# from src.drone_agent import DroneAgent
# from src.channel_model import CellularUrban3GppUmi
# import itertools
# from src.math_tools import decision
#
#
# class BaseStation:
#     sensing_tasks_locs = None
#     state_space = None
#     subspaces_2d_sizes = None
#     state_idxs = None  # (2d_idx, h_idx), (2d_idx, h_idx)....
#     prev_state_idxs = None
#     actions_taken = None
#     prev_actions_taken = None
#     qs = None
#     actions_flags = None
#     state_spaces_2d = None
#
#     def __init__(self, n_sub_channels=DEFAULT_N_SC, n_uavs=DEFAULT_N_UAVS,
#                  init_tasks_locations: List[Tuple[float, ...]] = [(500, 0), (-350, 350), (-350, -350)]):
#         self.coords = Coords3d(0, 0, 0)
#         self.n_sc = n_sub_channels
#         self.n_uavs = n_uavs
#         self.cycle_count = 0
#
#         self.state_space_x = np.linspace(-AREA_DIMS[0], AREA_DIMS[0], int(AREA_DIMS[0] * 2 / SPATIAL_POINT_STEP) + 1)
#         self.state_space_y = np.linspace(-AREA_DIMS[1], AREA_DIMS[1], int(AREA_DIMS[1] * 2 / SPATIAL_POINT_STEP) + 1)
#         self.state_space_z = np.linspace(MINIMUM_DRONE_HEIGHT, MAXIMUM_DRONE_HEIGHT,
#                                          int((MAXIMUM_DRONE_HEIGHT - MINIMUM_DRONE_HEIGHT) / SPATIAL_POINT_STEP) + 1)
#
#         if init_tasks_locations is None:
#             self.sensing_tasks_locs = [Coords3d(np.random.choice(self.state_space_x),
#                                                 np.random.choice(self.state_space_y), 0)
#                                        for _ in range(self.n_uavs)]
#         else:
#             assert (self.n_uavs == len(init_tasks_locations))
#             self.sensing_tasks_locs = [Coords3d(init_loc[0], init_loc[1], 0) for init_loc in init_tasks_locations]
#
#         self.uavs = [DroneAgent(coords=(self.sensing_tasks_locs[i]) + Coords3d(0, 0, 100).copy(),
#                                 sensing_task_loc=self.sensing_tasks_locs[i]) for i in
#                      range(self.n_uavs)]
#
#         self.initialize_state_action_qs()
#         self.initialize_uavs_locations()
#
#     def get_cycle_reward_probabilities(self):
#         pass
#
#     def initialize_state_action_qs(self):
#         states_actions_qs = [self.uavs[i].get_single_states_actions_qs(self.state_space_x, self.state_space_y,
#                                                                        self.state_space_z, self.coords) for i in
#                              range(self.n_uavs)]
#
#         states_subspaces_2d = [i[0] for i in states_actions_qs]
#         single_2d_actions_flags = [i[1] for i in states_actions_qs]
#         single_space_qs = [i[2] for i in states_actions_qs]
#
#         qs = [None] * self.n_uavs
#         for i in range(self.n_uavs):
#             prod_len_other_spaces = 1
#             for j in range(self.n_uavs):
#                 if j == i:
#                     continue
#                 prod_len_other_spaces *= len(states_subspaces_2d[j]) * len(self.state_space_z)
#             qs[i] = single_space_qs[i] * prod_len_other_spaces
#
#         self.subspaces_2d_sizes = np.array([len(_s) for _s in states_subspaces_2d])
#         self.qs = qs
#         self.actions_flags = single_2d_actions_flags
#         self.state_spaces_2d = states_subspaces_2d
#
#     def initialize_uavs_locations(self):
#         for idx, uav in enumerate(self.uavs):
#             x, y = self.state_spaces_2d[idx][0]
#             uav.coords = Coords3d(x, y, MINIMUM_DRONE_HEIGHT)
#         self.state_idxs = [(0, 0)] * self.n_uavs
#
#     def get_current_q_idxes_and_actions(self, uav_idx):
#         action_flags = self.get_2d_actions_flags(uav_idx)
#
#         state_2d_idxes = [state_idx[0] for state_idx in self.state_idxs]
#         height_idxes = [state_idx[1] for state_idx in self.state_idxs]
#
#         main_idx = np.array([self.get_q_idx(state_2d_idxes, uav_idx)])
#
#         q_idxes = np.array(list(itertools.product(main_idx, [height_idxes[uav_idx]], range(action_flags.sum()),
#                                                   range(len(self.state_space_z)))))
#         actions = list(itertools.product(self.state_spaces_2d[uav_idx][action_flags.nonzero()],
#                                          self.state_space_z))
#         actions = np.array([np.append(action[0], action[1]) for action in actions])
#
#         return q_idxes, actions
#
#     def get_q_values_from_idxs(self, uav_idx, q_idxes):
#         q_values = []
#         for idx in q_idxes:
#             q_values.append(self.qs[uav_idx][idx[0]][idx[1]][idx[2]][idx[3]])
#         return q_values
#
#     @staticmethod
#     def select_action(exploration_probability, q_values, actions, q_idxs):
#         if decision(exploration_probability):
#             idx = np.random.randint(actions.shape[0])
#             action = actions[idx]
#             q_idx = [q_idxs[idx]]
#
#         else:
#             idx = np.argmax(q_values)
#             action = actions[idx]
#             q_idx = [q_idxs[idx]]
#
#         return action, q_idx
#
#     def get_2d_actions_flags(self, uav_idx):
#         return self.actions_flags[uav_idx][self.state_idxs[uav_idx][0]]
#
#     def get_q_idx(self, subspaces_2d_idxs, uav_idx):
#         """:param subspaces_2d_idxs: indexes of the current states for each UAV relatively to their 2D allowed subspace'
#             :param height_idx which height idx from the allowed height steps
#             :param uav_idx
#             :param allowed_action_number the order of this action relatively to
#              the allowed actions in actions_flag[subspaces_2d_idxs[uav_idx]]
#         """
#         other_idxes = np.arange(len(self.subspaces_2d_sizes)) != uav_idx
#         other_spaces_sizes = self.subspaces_2d_sizes[other_idxes] * len(self.state_space_z)
#         own_space_size = self.subspaces_2d_sizes[uav_idx]
#
#         # Each index other than the uav idx steps with the following sizes
#         idxes_steps = np.concatenate(([own_space_size], other_spaces_sizes)).cumprod()[:-1]
#         idx = (idxes_steps * np.array(subspaces_2d_idxs)[other_idxes]).sum() + subspaces_2d_idxs[uav_idx]
#
#         idx_1 = idx
#         return idx_1
#
#     def set_random_trajectories(self):
#         [_uav.set_random_trajectory() for _uav in self.uavs]
#
#     def get_trajectories(self):
#         return [_uav.get_trajectory(start_frame_idx=1) for _uav in self.uavs]
#
#
# def get_successful_transmit_prob(uavs_trajecs, frame_idx=TB_BEACON_LENGTH + TS_SENSING_LENGTH + 1,
#                                  transmission_state=np.full(DEFAULT_N_UAVS, False), t_b=TB_BEACON_LENGTH,
#                                  t_s=TS_SENSING_LENGTH, t_u=TU_TRANSMISSION_LENGTH, n_c=DEFAULT_N_SC):
#     """Algorithm 1 (probabilities fixed)"""
#
#     t_c = t_b + t_s + t_u
#     n_uav = len(transmission_state)
#     success_probs = np.zeros(len(transmission_state))
#     if frame_idx == t_b + t_s + 1:
#         success_probs = np.zeros(len(transmission_state))
#     elif frame_idx > t_c:
#         return np.zeros(n_uav)
#
#     if np.all(transmission_state):
#         return np.zeros(n_uav)
#
#     # Frame idx begins at 1
#     locations = [uavs_trajecs[i][frame_idx - 1] for i in range(n_uav)]
#     f = np.vectorize(CellularUrban3GppUmi.get_successful_transmission_probability)
#     transmit_probs = f(np.array(locations, dtype=Coords3d))
#
#     v_i = np.full(n_uav, False)
#
#     # assign subchannels
#     tr_count = 0
#     for prob_idx in (-transmit_probs).argsort():
#         if transmission_state[prob_idx] == False:
#             v_i[prob_idx] = True
#             tr_count += 1
#             if tr_count == n_c:
#                 break
#
#     non_zero_probs = np.logical_and(v_i, np.invert(transmission_state))
#     success_probs[non_zero_probs] = transmit_probs[non_zero_probs]
#
#     awaiting_transmission_count = len(transmission_state[np.invert(transmission_state)])
#     for next_state_flags in list(itertools.product([True, False], repeat=awaiting_transmission_count)):
#         next_state = transmission_state.copy()
#         next_state[np.invert(transmission_state)] = next_state_flags
#
#         state_probability = 1.0
#         for idx, states in enumerate(zip(transmission_state, next_state)):
#             i_t, i_t_1 = states
#             if i_t:
#                 state_probability *= i_t_1
#             else:  # Corrected probability model by multiplying by v_i and ...
#                 if i_t_1:
#                     state_probability *= (v_i[idx] * transmit_probs[idx])
#                 else:
#                     state_probability *= (v_i[idx] * (1 - transmit_probs[idx])) + (1 - v_i[idx])
#             if state_probability == 0:
#                 break
#
#         if state_probability > 0:
#             success_probs += state_probability * get_successful_transmit_prob(uavs_trajecs, frame_idx + 1, next_state)
#
#     return success_probs
#
#
# if __name__ == '__main__':
#     a = BaseStation()
#     a.set_random_trajectories()
#     trajs = a.get_trajectories()
#     get_successful_transmit_prob(trajs)
#     # a.get_q_idx([1, 20, 20], 3, 1, 2)
#     a.get_2d_actions_flags(2)
#     q_idxs, actions = a.get_current_q_idxes_and_actions(2)
#     q_values = a.get_q_values_from_idxs(2, q_idxs)
#     q_values[0] = 0.8
#     q_values[4] = 0.9
#     action = a.select_action(0.2, q_values, actions, q_idxs)
