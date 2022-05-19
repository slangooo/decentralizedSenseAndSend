from typing import Tuple, List, Dict
from src.data_structures import Coords3d, to_coords_3d
from src.parameters import *
from itertools import count
import numpy as np
from src.math_tools import  distance_to_line, meshgrid2
from src.math_tools import decision, numpy_object_array
from random import choice
from itertools import product
from src.channel_model import CellularUrban3GppUmi, PlosModel
from scipy.spatial.distance import cdist

ACTIONS_SIDE_LENGTH = 3  # number of actions in dimension


class DroneAgent:
    _ids = count(0)
    sensing_lambda = None
    sensing_location = None
    next_location = None
    start_location = None
    allowed_2d_states = None
    states_actions_mapping = None

    def __init__(self, drone_id: int = None, coords: Coords3d = Coords3d(0.0, 0.0, 0.0),
                 t_power=DRONE_DEFAULT_TRANSMISSION_POWER, sensing_task_loc=None, base_station_coords = Coords3d(0,0,0)):
        self.id = next(self._ids) if drone_id is None else drone_id
        self.coords = coords.copy()
        self.start_location = coords.copy()
        self.t_power = t_power
        self.last_sensing_success = False
        self.base_station_coords = base_station_coords
        if sensing_task_loc is not None:
            self.set_sensing_task(sensing_task_loc)

    # def get_random_trajectory(self):
    #     possible_actions, allowed_actions = self.get_actions()
    #     possibilities = possible_actions[allowed_actions]
    #     if not possibilities.size:
    #         return None
    #     return to_coords_3d(choice(possibilities))

    def get_possible_2d_states(self, xs, ys):
        if self.allowed_2d_states:
            return self.allowed_2d_states

        if self.sensing_location.x >= 0:
            possible_xs = xs[np.where(np.logical_and(self.base_station_coords.x <= xs, xs <= self.sensing_location.x))]
        else:
            possible_xs = xs[np.where(np.logical_and(self.base_station_coords.x >= xs, xs >= self.sensing_location.x))]

        if self.sensing_location.y >= 0:
            possible_ys = xs[np.where(np.logical_and(self.base_station_coords.y <= ys, ys <= self.sensing_location.y))]
        else:
            possible_ys = ys[np.where(np.logical_and(self.base_station_coords.y >= ys, ys >= self.sensing_location.y))]

        states_2d = np.array(np.meshgrid(possible_xs, possible_ys)).T.reshape(len(possible_xs), len(possible_ys), 2)
        f = np.vectorize(distance_to_line, excluded=[1, 2], signature='(n)->()')
        distances = f(states_2d, Coords3d(0, 0, 0).as_2d_array(), self.sensing_location.as_2d_array())
        states_mask = distances <= SPATIAL_POINT_STEP

        # allowed states in 2D
        states_2d_allowed = states_2d[states_mask]
        self.allowed_2d_states = states_2d_allowed
        return states_2d_allowed

    def get_2d_state_action_mapping(self):
        if self.states_actions_mapping:
            return self.states_actions_mapping
        # allowed actions in the form of flags. I.e, at each given state i (indexed with respect to 2D) we have
        # actions[i][d] where d (same range of i) indexes the next state from 2D states.
        actions = np.array(
            cdist(self.allowed_2d_states, self.allowed_2d_states) < np.sqrt(SPATIAL_POINT_STEP ** 2 + SPATIAL_POINT_STEP ** 2))
        self.states_actions_mapping = actions
        return actions

    def get_single_states_actions_qs(self, xs, ys, zs):
        states_2d_allowed = self.get_possible_2d_states(xs, ys)
        actions = self.get_2d_state_action_mapping()
        q_values = [np.zeros((len(zs), actions.sum(0)[i], HEIGHT_LEVELS_PER_CYCLE)) for i in
                            range(len(actions))]  # 3 possible heights

        return states_2d_allowed, actions, q_values

    def get_location_in_frame(self, frame_idx: int, t_c=TC_CYCLE_LENGTH, t_b=TB_BEACON_LENGTH) -> Coords3d:
        # frame idx is current frame number in [Tb, Tc]. Corrected mistake in paper.
        return self.start_location + (frame_idx - t_b) / (t_c - t_b) * (self.next_location - self.start_location)

    def update_location_frame_idx(self, frame_idx, t_c=TC_CYCLE_LENGTH, t_b=TB_BEACON_LENGTH):
        self.coords = self.get_location_in_frame(frame_idx, t_c, t_b).copy()

    def set_next_location(self, next_coords: Coords3d):
        self.start_location = self.coords.copy()
        self.next_location = Coords3d.from_array(next_coords)

    # def set_random_trajectory(self):
    #     next_loc = self.get_random_trajectory()
    #     self.set_next_location(next_loc)
    #     return next_loc

    def set_sensing_task(self, sensing_location: Coords3d,
                         sensing_performance_lambda: float = DEFAULT_SENSING_PERFORMANCE_LAMBDA) -> float:
        """
        :param sensing_location: Coordinates of sensing task
        :param sensing_performance_lambda: Lambda per second
        :return: Successful sensing probability
        """
        self.sensing_location = sensing_location
        self.sensing_lambda = sensing_performance_lambda

    def get_successful_sensing_probability(self, loc: Coords3d = None, duration_of_frame=TF_FRAME_DRUATION):
        assert (self.sensing_lambda is not None and self.sensing_location is not None)
        if loc is None:
            distance_to_task = self.sensing_location.get_distance_to(self.coords)
        else:
            distance_to_task = self.sensing_location.get_distance_to(loc)
        return np.exp(-self.sensing_lambda * duration_of_frame * distance_to_task)

    def get_successful_sensing_probability_cycle(self, t_b=TB_BEACON_LENGTH, t_s=TS_SENSING_LENGTH):
        locations = self.get_trajectory(t_b + 1, t_b + t_s)
        probs = list(map(self.get_successful_sensing_probability, locations))
        return np.prod(probs)

    def get_trajectory(self, start_frame_idx=TB_BEACON_LENGTH, end_frame_idx=TC_CYCLE_LENGTH):
        if self.next_location is None:
            return []
        traj = list(map(self.get_location_in_frame, range(start_frame_idx, end_frame_idx + 1)))
        if start_frame_idx < TB_BEACON_LENGTH:
            traj = traj[TB_BEACON_LENGTH - start_frame_idx:]
            traj = [traj[0]] * (TB_BEACON_LENGTH - start_frame_idx) + traj

        return traj

    def sense_task(self):
        self.last_sensing_success = decision(self.get_successful_sensing_probability())


def get_successful_transmit_prob(uavs_trajecs, frame_idx=TB_BEACON_LENGTH + TS_SENSING_LENGTH + 1,
                                 transmission_state=np.full(DEFAULT_N_UAVS, False), t_b=TB_BEACON_LENGTH,
                                 t_s=TS_SENSING_LENGTH, t_u=TU_TRANSMISSION_LENGTH, n_c=DEFAULT_N_SC):
    """Algorithm 1 (probabilities fixed)"""

    t_c = t_b + t_s + t_u
    n_uav = len(transmission_state)
    success_probs = np.zeros(len(transmission_state))
    if frame_idx == t_b + t_s + 1:
        success_probs = np.zeros(len(transmission_state))
    elif frame_idx > t_c:
        return np.zeros(n_uav)

    if np.all(transmission_state):
        return np.zeros(n_uav)

    # Frame idx begins at 1
    locations = [uavs_trajecs[i][frame_idx - 1] for i in range(n_uav)]
    f = np.vectorize(CellularUrban3GppUmi.get_successful_transmission_probability, signature='()->()')
    transmit_probs = f(numpy_object_array(locations, dtype=Coords3d))

    v_i = np.full(n_uav, False)

    # assign subchannels
    tr_count = 0
    for prob_idx in (-transmit_probs).argsort():
        if transmission_state[prob_idx] == False:
            v_i[prob_idx] = True
            tr_count += 1
            if tr_count == n_c:
                break

    non_zero_probs = np.logical_and(v_i, np.invert(transmission_state))
    success_probs[non_zero_probs] = transmit_probs[non_zero_probs]

    awaiting_transmission_count = len(transmission_state[np.invert(transmission_state)])
    for next_state_flags in list(product([True, False], repeat=awaiting_transmission_count)):
        next_state = transmission_state.copy()
        next_state[np.invert(transmission_state)] = next_state_flags

        state_probability = 1.0
        for idx, states in enumerate(zip(transmission_state, next_state)):
            i_t, i_t_1 = states
            if i_t:
                state_probability *= i_t_1
            else:  # Corrected probability model by multiplying by v_i and ...
                if i_t_1:
                    state_probability *= (v_i[idx] * transmit_probs[idx])
                    # state_probability *= transmit_probs[idx]
                else:
                    state_probability *= (v_i[idx] * (1 - transmit_probs[idx])) + (1 - v_i[idx])
                    # state_probability *= (1 - transmit_probs[idx])
            if state_probability == 0:
                break

        if state_probability > 0:
            success_probs += state_probability * get_successful_transmit_prob(uavs_trajecs, frame_idx + 1, next_state)

    return success_probs


if __name__ == '__main__':
    pass
