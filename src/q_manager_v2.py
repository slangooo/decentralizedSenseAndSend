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
    qs = None
    actions_flags = None
    states_subspaces_2d = None
    subspaces_sizes = None
    state_space_size = None
    states_2d_actions_mapping = None
    subspaces_actions_counts_per_state = None
    actions_counts = None


    def __init__(self, uavs, n_uavs=DEFAULT_N_UAVS, base_station_coords=Coords3d(0, 0, 0)):
        self.uavs = uavs

        self.state_space_x = np.linspace(-AREA_DIMS[0], AREA_DIMS[0], int(AREA_DIMS[0] * 2 / SPATIAL_POINT_STEP) + 1)
        self.state_space_y = np.linspace(-AREA_DIMS[1], AREA_DIMS[1], int(AREA_DIMS[1] * 2 / SPATIAL_POINT_STEP) + 1)
        self.state_space_z = np.linspace(MINIMUM_DRONE_HEIGHT, MAXIMUM_DRONE_HEIGHT,
                                         int((MAXIMUM_DRONE_HEIGHT - MINIMUM_DRONE_HEIGHT) / SPATIAL_POINT_STEP) + 1)
        self.z_cardinality = len(self.state_space_z)
        self.n_uavs = n_uavs
        self.base_station_coords = base_station_coords

        self.initialize_state_action_qs()
        self.next_2d_states_idxs = [None for _ in range(self.n_uavs)]

    def get_subspace_idx_from_space_idx(self, uav_idx, space_idx):
        size_prod = 1
        for idx, subspace_size in enumerate(self.subspaces_sizes):
            if idx == uav_idx:
                return int((space_idx // size_prod) % subspace_size)
            size_prod *= subspace_size

    def get_height_idx_from_subspace_idx(self, subspace_idx):
        return int(subspace_idx % self.z_cardinality)

    def get_2d_state_idx_from_subspace_idx(self, subspace_idx):
        return int(subspace_idx // self.z_cardinality)

    def get_2d_state_idx_from_space_idx(self, uav_idx, space_idx):
        subspace_idx = self.get_subspace_idx_from_space_idx(uav_idx, space_idx)
        return int(self.get_2d_state_idx_from_subspace_idx(subspace_idx))

    def get_allowed_action_from_action_idx(self, uav_idx, space_idx, action_idx):
        action_size_prod = 1
        for _uav_idx in range(self.n_uavs):
            uav_2d_idx = self.get_2d_state_idx_from_space_idx(_uav_idx, space_idx)
            if _uav_idx == uav_idx:
                uav_action_idx = (action_idx // action_size_prod) % self.subspaces_actions_counts_per_state[_uav_idx][
                    uav_2d_idx]
                break
            action_size_prod *= self.subspaces_actions_counts_per_state[_uav_idx][uav_2d_idx]
        height_idx = uav_action_idx % self.z_cardinality
        idx_2d = uav_action_idx // self.z_cardinality
        height = self.state_space_z[height_idx]
        state_2d_idx = self.states_2d_actions_mapping[uav_idx][uav_2d_idx].nonzero()[0][idx_2d]
        state_2d = self.states_subspaces_2d[uav_idx][state_2d_idx]
        return [np.append(state_2d, height), [state_2d_idx, height_idx]]

    def get_action_idx_from_actions_subspaces_idxs(self, actions_idxs, space_idx):
        action_spaces_granularities = self.actions_counts[space_idx][1].cumprod()/self.actions_counts[space_idx][1]
        action_idx = (actions_idxs * action_spaces_granularities).sum()
        return int(action_idx)

    def get_action_subspace_idx_from_action_idx(self, uav_idx, action_idx, space_idx):
        size_prod = 1
        for idx, action_subspace_size in enumerate(self.actions_counts[space_idx][1]):
            if idx == uav_idx:
                return int((action_idx // size_prod) % action_subspace_size)
            size_prod *= action_subspace_size


    def get_space_idx_from_2d_height_idxs(self, idxs_2d_height):
        """[state_2d_idx, height_idx]] taken from get_allowed_action_from_action_idx"""
        subspaces_granularities = (self.subspaces_sizes.cumprod() / self.subspaces_sizes).astype(int)
        idx = 0
        for _uav_idx in range(self.n_uavs - 1, -1, -1):
            subspace_idx = idxs_2d_height[_uav_idx][0] * self.z_cardinality + idxs_2d_height[_uav_idx][1]
            idx += subspace_idx * subspaces_granularities[_uav_idx]
        return int(idx)

    def get_idxs_of_unique_actions_per_uav(self, uav_idx, space_idx):
        actions_idxs = []
        n_actions_for_uav = self.actions_counts[space_idx][1][uav_idx]
        total_number_of_actions_in_state = self.actions_counts[space_idx][1].prod()
        n_action_successive_repetitions = (self.actions_counts[space_idx][1].cumprod() // \
                                          self.actions_counts[space_idx][1])[uav_idx]

        for action_number in range(n_actions_for_uav):
            action_idxs = []
            first_occurrence = action_number * n_action_successive_repetitions
            occurrences = np.arange(first_occurrence, total_number_of_actions_in_state,
                                   n_action_successive_repetitions * n_actions_for_uav)
            for _occ in occurrences:
                action_idxs.append(np.arange(_occ, _occ + n_action_successive_repetitions))
            actions_idxs.append(np.array(action_idxs).ravel())
        return actions_idxs

    def initialize_state_action_qs(self):
        # states_actions_qs = [self.uavs[i].get_single_states_actions_qs(self.state_space_x, self.state_space_y,
        #                                                                self.state_space_z) for
        #                      i in range(self.n_uavs)]

        self.states_subspaces_2d = [self.uavs[i].get_possible_2d_states(self.state_space_x, self.state_space_y) for i in
                                    range(self.n_uavs)]
        self.subspaces_2d_sizes = np.array([len(_s) for _s in self.states_subspaces_2d], dtype=int)
        self.subspaces_sizes = self.subspaces_2d_sizes * self.z_cardinality
        self.state_space_size = self.subspaces_sizes.prod()

        self.states_2d_actions_mapping = [self.uavs[i].get_2d_state_action_mapping() for i in range(self.n_uavs)]
        states_2d_actions_mapping_counts = []
        for _mapping in self.states_2d_actions_mapping:
            states_2d_actions_mapping_counts.append(_mapping.sum(axis=1))

        self.subspaces_actions_counts_per_state = [count * len(self.state_space_z) for count in
                                                   states_2d_actions_mapping_counts]

        # self.total_actions_per_uav_count = [states_2d_actions_mapping_counts[i].sum() for i in range(self.n_uavs)]

        self.qs = []
        self.actions_counts = []
        for space_idx in range(self.state_space_size):
            n_actions = 1
            n_action_sizes = []
            for uav_idx in range(self.n_uavs):
                idx_2d_state = self.get_2d_state_idx_from_space_idx(uav_idx, space_idx)
                n_actions *= self.subspaces_actions_counts_per_state[uav_idx][idx_2d_state]
                n_action_sizes.append(self.subspaces_actions_counts_per_state[uav_idx][idx_2d_state])
            self.qs.append(np.array([np.zeros(n_actions) for _ in range(self.n_uavs)]))
            self.actions_counts.append([np.zeros(n_actions, dtype=int), np.array(n_action_sizes, dtype=int)])
            # Note actions_count is of form [[counts of each action possible in state in state], [# of actions for each UAV]]
        # a = 5
        # #TODO:Remove
        # for space_idx in range(len(self.qs)):
        #     for action_dx in range(len(self.qs[space_idx])):
        #         subspaces_idxs = [self.get_action_subspace_idx_from_action_idx(uav_idx, action_dx, space_idx)
        #                           for uav_idx in range(self.n_uavs)]
        #         assert(self.get_action_idx_from_actions_subspaces_idxs(subspaces_idxs, space_idx) == action_dx)

    def initialize_uavs_locations_in_state_space(self):
        state_idxs = []
        for idx, uav in enumerate(self.uavs):
            if Coords3d(0, 0, 0).get_distance_to(self.states_subspaces_2d[idx][0]) > \
                    Coords3d(0, 0, 0).get_distance_to(self.states_subspaces_2d[idx][-1]):
                x, y = self.states_subspaces_2d[idx][0]
                state_idxs.append(0)
            else:
                x, y = self.states_subspaces_2d[idx][-1]
                state_idxs.append(len(self.states_subspaces_2d[idx]) - 1)

            uav.coords = Coords3d(x, y, MAXIMUM_DRONE_HEIGHT)
        self.state_idxs = [[state_idxs[i], self.z_cardinality - 1] for i in range(self.n_uavs)]

    def get_actions_policy_values(self, state_count, space_idx):
        # assert(state_count > 0)
        #For each uav get permutation of other uavs actions
        uavs_policy_values = []
        for _uav_idx in range(self.n_uavs):
            other_sizes = np.delete(self.actions_counts[space_idx][1], _uav_idx)
            other_indices = [np.arange(_size) for _size in other_sizes]

            #Other actions profiles counts
            combinations = list(itertools.product(*other_indices))
            combinations_count = []
            for _comb in combinations:
                comb_action_count = 0
                for i in range(self.actions_counts[space_idx][1][_uav_idx]):
                    actions_subspaces_idx = np.insert(_comb, _uav_idx, i)
                    action_idx = self.get_action_idx_from_actions_subspaces_idxs(actions_subspaces_idx, space_idx)
                    comb_action_count += self.actions_counts[space_idx][0][action_idx]
                combinations_count.append(comb_action_count)

            policy_values = []
            for _action_idx in range(self.actions_counts[space_idx][1][_uav_idx]):
                policy_value = 0
                for comb_idx, comb in enumerate(combinations):
                    actions_subspaces_idx = np.insert(_comb, _uav_idx, _action_idx)
                    action_idx = self.get_action_idx_from_actions_subspaces_idxs(actions_subspaces_idx, space_idx)
                    policy_value += combinations_count[comb_idx]/max(state_count, 1) * self.qs[space_idx][_uav_idx][action_idx]
                policy_values.append(policy_value)

            uavs_policy_values.append(np.array(policy_values))
        return uavs_policy_values

    def select_action_in_cycle(self, cycle_number):
        space_idx = self.get_space_idx_from_2d_height_idxs(self.state_idxs)
        self.prev_space_idx = space_idx

        # # TODO:REMOVE
        # subspaces_idxs = []
        # heights_idxs = []
        # idxs_2d = []
        # for i in range(self.n_uavs):
        #     subspaces_idxs.append(self.get_subspace_idx_from_space_idx(i, space_idx))
        #     heights_idxs.append(self.get_height_idx_from_subspace_idx(subspaces_idxs[i]))
        #     idxs_2d.append(self.get_2d_state_idx_from_subspace_idx(subspaces_idxs[i]))
        #     assert (subspaces_idxs[i] == self.get_subspace_idx_from_space_idx(i, space_idx))
        #     assert (idxs_2d[i] == self.get_2d_state_idx_from_space_idx(i, space_idx))
        #     assert (self.state_idxs[i][0] == idxs_2d[i])
        #     assert (self.state_idxs[i][1] == heights_idxs[i])
        #     assert (self.uavs[i].coords.z == self.state_space_z[heights_idxs[i]])
        #     assert (np.all([self.uavs[i].coords.x, self.uavs[i].coords.y] == self.states_subspaces_2d[i][idxs_2d[i]]))

        exploration_probability = self.get_exploration_probability(cycle_number)

        self.selected_action_idx = self.select_action(space_idx, exploration_probability)[0]

        actions = [self.get_allowed_action_from_action_idx(uav_idx, space_idx, self.selected_action_idx)
                   for uav_idx in range(self.n_uavs)]

        new_states = [action[1] for action in actions]
        new_locations = [action[0] for action in actions]

        self.update_states(new_states)
        self.current_space_idx = self.get_space_idx_from_2d_height_idxs(self.state_idxs)

        return new_locations

    def select_action(self, space_idx, exploration_probability):
        state_visits_count = self.actions_counts[space_idx][0].sum()

        max_policy_values = []
        if state_visits_count == 0 or decision(exploration_probability):
            actions_subspaces_idxs = [np.random.randint(self.actions_counts[space_idx][1][i])
                                      for i in range(self.n_uavs)]
            max_policy_values = np.zeros(self.n_uavs)
        else:
            policy_values = self.get_actions_policy_values(state_visits_count, space_idx)
            actions_subspaces_idxs = []
            for i in range(self.n_uavs):
                actions_subspaces_idxs.append(policy_values[i].argmax())
                max_policy_values.append(policy_values[i][actions_subspaces_idxs[i]])
        action_idx = self.get_action_idx_from_actions_subspaces_idxs(actions_subspaces_idxs, space_idx)

        # assert (action_idx < self.actions_counts[space_idx][1].prod())
        return action_idx, max_policy_values

    def update_states(self, new_states):
        for uav_idx, new_state in enumerate(new_states):
            self.state_idxs[uav_idx] = [new_state[0], new_state[1]]

    def update_q_values(self, rewards, cycle_number):
        learning_rate = self.get_learning_rate(cycle_number)

        next_action_policy_values = self.select_action(self.current_space_idx, 0)[1]

        self.actions_counts[self.prev_space_idx][0][self.selected_action_idx] += 1

        for uav_idx in range(self.n_uavs):
            self.qs[self.prev_space_idx][uav_idx][self.selected_action_idx] = (1 - learning_rate) *\
                 self.qs[self.prev_space_idx][uav_idx][self.selected_action_idx] + learning_rate *\
                                                (rewards[uav_idx] + DISCOUNT_RATIO * next_action_policy_values[uav_idx])

    @staticmethod
    def get_learning_rate(cycle_number):
        # return 1 / (cycle_number ** (2 / 8))
        # return 1 / (cycle_number ** (2 / 3))
        return 0.4

    @staticmethod
    def get_exploration_probability(cycle_number):
        return 0.8 * np.exp(-0.0003 * cycle_number)
        # return 0.8 * np.exp(-0.03 * cycle_number)
        # return 0.2
