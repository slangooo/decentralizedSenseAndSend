from src.parameters import DEFAULT_N_SC, DEFAULT_N_UAVS, AREA_DIMS, TF_FRAME_DRUATION, \
    TB_BEACON_LENGTH, TS_SENSING_LENGTH, TC_CYCLE_LENGTH, TU_TRANSMISSION_LENGTH, SPATIAL_POINT_STEP, \
    MAXIMUM_DRONE_HEIGHT, MINIMUM_DRONE_HEIGHT, DISCOUNT_RATIO, HEIGHT_LEVELS_PER_CYCLE
import numpy as np
from src.data_structures import Coords3d
from src.math_tools import decision
import itertools
import copy

"""Heights + action where action =[-step, 0, +step] q_action for heights are relative"""


class QManager:
    subspaces_2d_sizes = None
    state_idxs = None  # (2d_idx, h_idx), (2d_idx, h_idx)....
    prev_state_idxs = None
    actions_taken = None
    prev_actions_taken = None
    qs = None
    actions_flags = None
    state_spaces_2d = None
    last_q_idxs = None
    last_q_values = None
    discount_ratio = DISCOUNT_RATIO

    def __init__(self, uavs, n_uavs=DEFAULT_N_UAVS, base_station_coords=Coords3d(0, 0, 0)):
        self.uavs = uavs

        self.state_space_x = np.linspace(-AREA_DIMS[0], AREA_DIMS[0], int(AREA_DIMS[0] * 2 / SPATIAL_POINT_STEP) + 1)
        self.state_space_y = np.linspace(-AREA_DIMS[1], AREA_DIMS[1], int(AREA_DIMS[1] * 2 / SPATIAL_POINT_STEP) + 1)
        self.state_space_z = np.linspace(MINIMUM_DRONE_HEIGHT, MAXIMUM_DRONE_HEIGHT,
                                         int((MAXIMUM_DRONE_HEIGHT - MINIMUM_DRONE_HEIGHT) / SPATIAL_POINT_STEP) + 1)

        self.n_uavs = n_uavs
        self.base_station_coords = base_station_coords

        self.initialize_state_action_qs()
        self.next_2d_states_idxs = [None for _ in range(self.n_uavs)]

    def initialize_state_action_qs(self):
        states_actions_qs = [self.uavs[i].get_single_states_actions_qs(self.state_space_x, self.state_space_y,
                                                                       self.state_space_z, self.base_station_coords) for
                             i in range(self.n_uavs)]

        states_subspaces_2d = [i[0] for i in states_actions_qs]
        single_2d_actions_flags = [i[1] for i in states_actions_qs]
        single_space_qs = [i[2] for i in states_actions_qs]

        qs = [None] * self.n_uavs
        for i in range(self.n_uavs):
            prod_len_other_spaces = 1
            for j in range(self.n_uavs):
                if j == i:
                    continue
                prod_len_other_spaces *= len(states_subspaces_2d[j]) * len(self.state_space_z)

            tmp = []
            for _ in range(prod_len_other_spaces):
                tmp += copy.deepcopy(single_space_qs[i])
            qs[i] = tmp

        self.subspaces_2d_sizes = np.array([len(_s) for _s in states_subspaces_2d])
        self.qs = qs
        self.actions_flags = single_2d_actions_flags
        self.state_spaces_2d = states_subspaces_2d


    def initialize_uavs_locations_in_state_space(self):
        state_idxs = []
        for idx, uav in enumerate(self.uavs):
            if Coords3d(0, 0, 0).get_distance_to(self.state_spaces_2d[idx][0]) < \
                    Coords3d(0, 0, 0).get_distance_to(self.state_spaces_2d[idx][-1]):
                x, y = self.state_spaces_2d[idx][0]
                state_idxs.append(0)
            else:
                x, y = self.state_spaces_2d[idx][-1]
                state_idxs.append(len(self.state_spaces_2d[idx]) - 1)

            uav.coords = Coords3d(x, y, MAXIMUM_DRONE_HEIGHT)
        self.state_idxs = [[state_idxs[i], 0] for i in range(self.n_uavs)]

    def get_main_q_idx(self, subspaces_2d_idxs, uav_idx):
        """:param subspaces_2d_idxs: indexes of the current states for each UAV relatively to their 2D allowed subspace'
            :param uav_idx
            :return indexes of possible actions Q values in Q matrix
        """
        # return (self.subspaces_2d_sizes[:-1] * len(self.state_space_z) * subspaces_2d_idxs[1:] + subspaces_2d_idxs[0])[0]
        other_idxes = np.arange(len(self.subspaces_2d_sizes)) != uav_idx
        other_spaces_sizes = self.subspaces_2d_sizes[other_idxes] * len(self.state_space_z)
        own_space_size = self.subspaces_2d_sizes[uav_idx]

        # Each index other than the uav idx steps with the following sizes
        idxes_steps = np.concatenate(([own_space_size], other_spaces_sizes)).cumprod()[:-1]
        idx = (idxes_steps * np.array(subspaces_2d_idxs)[other_idxes]).sum() + subspaces_2d_idxs[uav_idx]

        idx_1 = idx
        return idx_1

    def get_2d_actions_flags(self, uav_idx):
        return self.actions_flags[uav_idx][self.state_idxs[uav_idx][0]]

    def select_action_in_cycle(self, cycle_number):
        exploration_probability = self.get_exploration_probability(cycle_number)
        q_idx_action = [self.get_current_q_idxes_and_actions(i) for i in range(self.n_uavs)]
        q_values = [self.get_q_values_from_idxs(i, q_idx_action[i][0]) for i in range(self.n_uavs)]
        selected_q_idx_action = [self.select_action_from_qs(exploration_probability, q_values[i], q_idx_action[i][0],
                                                            q_idx_action[i][1]) for i in range(self.n_uavs)]
        self.last_q_idxs = [selected_q_idx_action[i][0] for i in range(self.n_uavs)]
        self.last_q_values = [selected_q_idx_action[i][2] for i in range(self.n_uavs)]
        self.update_states()

        return [selected_q_idx_action[i][1] for i in range(self.n_uavs)]

    def update_states(self):
        for i in range(self.n_uavs):
            state_2d_idx = self.last_q_idxs[i][0] % self.subspaces_2d_sizes[i]
            new_2d_idx = self.actions_flags[i][state_2d_idx].nonzero()[0][self.last_q_idxs[i][2]]
            self.state_idxs[i][0] = new_2d_idx
            self.state_idxs[i][1] += self.last_q_idxs[i][3] - 1  # Because actions are either up or same or down.
            # So the states either increase or decrease by one

    def update_q_values(self, rewards, cycle_number):
        learning_rate = self.get_learning_rate(cycle_number)
        q_idx_action = [self.get_current_q_idxes_and_actions(i) for i in range(self.n_uavs)]
        q_values = [self.get_q_values_from_idxs(i, q_idx_action[i][0]) for i in range(self.n_uavs)]
        selected_q_idx_action = [self.select_action_from_qs(0, q_values[i], q_idx_action[i][0],
                                                            q_idx_action[i][1]) for i in range(self.n_uavs)]
        # next_q_idxs = [selected_q_idx_action[i, 0] for i in range(self.n_uavs)]
        next_q_values = [selected_q_idx_action[i][2] for i in range(self.n_uavs)]

        for i in range(self.n_uavs):
            self.qs[i][self.last_q_idxs[i][0]][self.last_q_idxs[i][1], self.last_q_idxs[i][2], self.last_q_idxs[i][3]] \
                = self.last_q_values[i] + learning_rate * (rewards[i] +
                                                           self.discount_ratio * next_q_values[i] - self.last_q_values[
                                                               i])
            # print(self.qs[i][self.last_q_idxs[i][0]][self.last_q_idxs[i][1], self.last_q_idxs[i][2], self.last_q_idxs[i][3]])

    @staticmethod
    def get_learning_rate(cycle_number):
        # return 1 / (cycle_number ** (2 / 8))
        # return 1 / (cycle_number ** (2 / 3))
        return 0.2

    @staticmethod
    def get_exploration_probability(cycle_number):
        # return 0.8 * np.exp(-0.0003 * cycle_number)
        # return 0.8 * np.exp(-0.03 * cycle_number)
        return 0.2

    @staticmethod
    def select_action_from_qs(exploration_probability, q_values, q_idxs, actions):
        q_values = np.array(q_values)
        if decision(exploration_probability):
            idx = np.random.randint(actions.shape[0])
            action = actions[idx]
            q_idx = q_idxs[idx]
            value = q_values[idx]

        else:
            # idx = np.argmax(q_values)
            idx = np.random.choice(np.where(q_values == q_values.max())[0])
            action = actions[idx]
            q_idx = q_idxs[idx]
            value = q_values[idx]
        # print("2", q_idx, action)
        return q_idx.copy(), action.copy(), value.copy()

    def get_current_q_idxes_and_actions(self, uav_idx):
        action_flags = self.get_2d_actions_flags(uav_idx)

        state_2d_idxes = [state_idx[0] for state_idx in self.state_idxs]
        height_idxes = [state_idx[1] for state_idx in self.state_idxs]

        main_idx = np.array([self.get_main_q_idx(state_2d_idxes, uav_idx)])

        q_idxes = list(itertools.product(main_idx, [height_idxes[uav_idx]], range(action_flags.sum()),
                                         range(HEIGHT_LEVELS_PER_CYCLE)))

        current_height = self.state_space_z[height_idxes][uav_idx]
        # possible_heights = np.arange(max(MINIMUM_DRONE_HEIGHT, current_height - SPATIAL_POINT_STEP),
        #                              min(MAXIMUM_DRONE_HEIGHT, current_height + SPATIAL_POINT_STEP)
        #                              + SPATIAL_POINT_STEP, SPATIAL_POINT_STEP)
        possible_heights = np.arange(current_height - SPATIAL_POINT_STEP,
                                     current_height + 2 * SPATIAL_POINT_STEP, SPATIAL_POINT_STEP)

        # TODO:REMOVE
        for _q_idx in q_idxes:
            state_2d_idx = _q_idx[0] % self.subspaces_2d_sizes[uav_idx]
            assert (self.actions_flags[uav_idx][state_2d_idx].sum() - 1 >= _q_idx[2])

        actions = list(itertools.product(self.state_spaces_2d[uav_idx][action_flags.nonzero()[0]],
                                         possible_heights))
        actions = [np.append(action[0], action[1]) for action in actions]

        popped_count = 0
        for idx in range(len(actions)):
            if actions[idx - popped_count][2] < MINIMUM_DRONE_HEIGHT or actions[idx - popped_count][2] > MAXIMUM_DRONE_HEIGHT:
                q_idxes.pop(idx - popped_count)
                actions.pop(idx - popped_count)
                popped_count += 1

        assert (len(q_idxes) == len(actions))
        # TODO:REMOVE
        for _q_idx, action in zip(q_idxes, actions):
            assert (action[-1] >= MINIMUM_DRONE_HEIGHT and action[-1] <= MAXIMUM_DRONE_HEIGHT)
            state_2d_idx = _q_idx[0] % self.subspaces_2d_sizes[uav_idx]
            next_action = self.state_spaces_2d[uav_idx][
                self.actions_flags[uav_idx][state_2d_idx].nonzero()[0][_q_idx[2]]]
            assert (next_action[0] == action[0] and next_action[1] == action[1])

        self.get_state_count_idx()
        return np.array(q_idxes), np.array(actions)

    def get_q_values_from_idxs(self, uav_idx, q_idxes):
        q_values = []
        for idx in q_idxes:
            q_values.append(self.qs[uav_idx][idx[0]][idx[1], idx[2], idx[3]])
        return q_values
