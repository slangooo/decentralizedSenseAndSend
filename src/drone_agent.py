from typing import Tuple, List, Dict
from src.data_structures import Coords3d, to_coords_3d
from src.parameters import *
from itertools import count
import numpy as np
from src.math_tools import  distance_to_line, meshgrid2
from src.math_tools import decision, numpy_object_array
from random import choice
from itertools import product
from src.channel_model import CellularUrban3GppUmi
from scipy.spatial.distance import cdist

ACTIONS_SIDE_LENGTH = 3  # number of actions in dimension


class DroneAgent:
    _ids = count(0)
    sensing_lambda = None
    sensing_location = None
    next_location = None
    start_location = None

    def __init__(self, drone_id: int = None, coords: Coords3d = Coords3d(0.0, 0.0, 0.0),
                 t_power=DRONE_DEFAULT_TRANSMISSION_POWER, sensing_task_loc=None):
        self.id = next(self._ids) if drone_id is None else drone_id
        self.coords = coords.copy()
        self.start_location = coords.copy()
        self.t_power = t_power
        self.last_sensing_success = False
        if sensing_task_loc is not None:
            self.set_sensing_task(sensing_task_loc)

    def get_random_trajectory(self):
        possible_actions, allowed_actions = self.get_actions()
        possibilities = possible_actions[allowed_actions]
        if not possibilities.size:
            return None
        return to_coords_3d(choice(possibilities))

    def get_single_states_actions_qs(self, xs, ys, zs, base_station_coords):
        """Return state space that can be indexed by [i][j][k] and 2D space mask index by [i][j]"""

        """Return the allowed state space defined as the points on the vertical plane between
         the base station and the sensing task with a margin of 1 spatial step
         Also the allowed actions mask per each state
         q_values are of shape [len(allowed_states), len(actions[i].nonzero()), len(zs)]
         """

        if self.sensing_location.x >= 0:
            possible_xs = xs[np.where(np.logical_and(base_station_coords.x <= xs, xs <= self.sensing_location.x))]
        else:
            possible_xs = xs[np.where(np.logical_and(base_station_coords.x >= xs, xs >= self.sensing_location.x))]

        if self.sensing_location.y >= 0:
            possible_ys = xs[np.where(np.logical_and(base_station_coords.y <= ys, ys <= self.sensing_location.y))]
        else:
            possible_ys = ys[np.where(np.logical_and(base_station_coords.y >= ys, ys >= self.sensing_location.y))]

        # #TODO: Remove or fix by adding x case
        # oscillate_ys_flag = False
        # if len(possible_ys) <= 1:
        #     possible_ys = np.array([possible_ys[0], possible_ys[0] + SPATIAL_POINT_STEP])
        #     oscillate_ys_flag = True


        states_2d = np.array(np.meshgrid(possible_xs, possible_ys)).T.reshape(len(possible_xs), len(possible_ys), 2)
        f = np.vectorize(distance_to_line, excluded=[1, 2], signature='(n)->()')
        distances = f(states_2d, Coords3d(0, 0, 0).as_2d_array(), self.sensing_location.as_2d_array())
        states_mask = distances <= SPATIAL_POINT_STEP

        # allowed states in 2D
        states_2d_allowed = states_2d[states_mask]

        # # TODO: Remove or fix by adding x case
        # if oscillate_ys_flag:
        #     for i in range(1,len(states_2d_allowed),4):
        #         states_2d_allowed[i][1] *= -1
        #     states_2d_allowed = np.concatenate((states_2d_allowed, np.array([[states_2d_allowed[-1][0], states_2d_allowed[-1][1]]])))

        # allowed actions in the form of flags. I.e, at each given state i (indexed with respect to 2D) we have
        # actions[i][d] where d (same range of i) indexes the next state from 2D states.
        actions = np.array(
            cdist(states_2d_allowed, states_2d_allowed) < np.sqrt(SPATIAL_POINT_STEP ** 2 + SPATIAL_POINT_STEP ** 2))
        # q_values[i][j] for each allowed state [i] and height [j] is a row which contains the q_values for all allowed actions in that state
        # each q_values[i][j] has len(actions[i]!=False) rows, and each element in that row correspong to different height
        # I.e, 1nd allowed action with 2nd allowed height in first state with 3rd as current height is q_values[1][3][1][2]

        # q_values = []
        # for i in range(len(states_2d_allowed)):
        #     state_2d = states_2d_allowed[i]
        #     for k in range(len(zs)):
        #         state_z = zs[k]
        #         for next_2d_state_i in actions[i].nonzero():
        #             next_2d = states_2d_allowed[next_2d_state_i]
        #             self.coords = Coords3d(state_2d[0], state_2d[1], state_z)
        #             self.set_next_location(Coords3d(next_2d[0], next_2d[1], state_z))
        #             for next_z_i in range(HEIGHT_LEVELS_PER_CYCLE):
        #                 next_z = state_z + (next_z_i - 1) * SPATIAL_POINT_STEP

        #
        #     n_actions = actions.sum(0)[i]
        # zs_array = np.array([])
        # for k in range(len(zs)):
        #     n_zs = 3 if 0 < k < len(zs) - 1 else 2
        #     zs_array = np.stack((zs_array, np.zeros(n_zs)))
        #
        # q_values = [np.tile(zs_array, (len(zs), actions.sum(0)[i], 1)) for i in range(len(actions))]

        q_values = [np.zeros((len(zs), actions.sum(0)[i], HEIGHT_LEVELS_PER_CYCLE)) for i in
                    range(len(actions))]  # 3 possible heights

        return states_2d_allowed, actions, q_values

    def get_actions(self):
        dist_to_plane = distance_to_line(self.coords.as_2d_array(),
                                         Coords3d(0, 0, 0).as_2d_array(), self.sensing_location.as_2d_array())

        max_coords = self.coords + SPATIAL_POINT_STEP
        min_coords = self.coords - SPATIAL_POINT_STEP

        xs, ys = np.linspace(min_coords.as_2d_array(), max_coords.as_2d_array(), ACTIONS_SIDE_LENGTH).T
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij', sparse=False)

        possible_2d = np.hstack([grid_x.T.reshape(-1, 1), grid_y.T.reshape(-1, 1)])

        dists_to_plane = distance_to_line(possible_2d, Coords3d(0, 0, 0).as_2d_array(),
                                          self.sensing_location.as_2d_array())

        allowed_2d = np.logical_or((dists_to_plane <= dist_to_plane), (dists_to_plane <= SPATIAL_POINT_STEP))

        possible_zs = np.linspace(self.coords.z - SPATIAL_POINT_STEP,
                                  self.coords.z + SPATIAL_POINT_STEP, ACTIONS_SIDE_LENGTH)

        allowed_z = np.logical_and(possible_zs >= MINIMUM_DRONE_HEIGHT, MAXIMUM_DRONE_HEIGHT >= possible_zs)

        possible_actions = tuple(reversed(meshgrid2(xs, ys, possible_zs)))

        possible_actions = np.vstack(list(map(np.ravel, possible_actions))).T
        allowed_actions = np.logical_and(np.tile(allowed_2d, ACTIONS_SIDE_LENGTH),
                                         allowed_z.repeat(ACTIONS_SIDE_LENGTH ** 2))

        return possible_actions, allowed_actions

    def get_location_in_frame(self, frame_idx: int, t_c=TC_CYCLE_LENGTH, t_b=TB_BEACON_LENGTH) -> Coords3d:
        # frame idx is current frame number in [Tb, Tc]. Corrected mistake in paper.
        return self.start_location + (frame_idx - t_b) / (t_c - t_b) * (self.next_location - self.start_location)

    def update_location_frame_idx(self, frame_idx, t_c=TC_CYCLE_LENGTH, t_b=TB_BEACON_LENGTH):
        self.coords = self.get_location_in_frame(frame_idx, t_c, t_b).copy()

    def set_next_location(self, next_coords: Coords3d):
        self.start_location = self.coords.copy()
        self.next_location = Coords3d.from_array(next_coords)

    def set_random_trajectory(self):
        next_loc = self.get_random_trajectory()
        self.set_next_location(next_loc)
        return next_loc

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
                else:
                    state_probability *= (v_i[idx] * (1 - transmit_probs[idx])) + (1 - v_i[idx])
            if state_probability == 0:
                break

        if state_probability > 0:
            success_probs += state_probability * get_successful_transmit_prob(uavs_trajecs, frame_idx + 1, next_state)

    return success_probs
