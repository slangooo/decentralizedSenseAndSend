import numpy as np
from src.parameters import DEFAULT_CARRIER_FREQ, NOISE_POWER_N0, DEFAULT_SNR_THRESHOLD, DRONE_DEFAULT_TRANSMISSION_POWER
from src.data_structures import Coords3d
from scipy.stats import rice, rayleigh
from src.math_tools import db2lin, lin2db
from scipy.constants import speed_of_light

class CellularUrban3GppUmi:

    @staticmethod
    def get_los_probability(drone_coords: Coords3d, base_station_coords: Coords3d) -> float:
        distance_2d = drone_coords.get_distance_to(base_station_coords, flag_2d=True)
        r_i = distance_2d
        r_c = max(294.05 * np.log10(drone_coords.z) - 432.94, 18)

        if r_i <= r_c:
            return 1
        else:
            p_0 = 233.98 * np.log10(drone_coords.z) - 0.95
            res = r_c / r_i + np.exp((-r_i / p_0)*(1 - r_c/r_i))

            return min(res, 1)

    @staticmethod
    def get_los_pathloss(drone_coords: Coords3d, base_station_coords: Coords3d,
                         carrier_freq=DEFAULT_CARRIER_FREQ) -> float:
        distance_3d = drone_coords.get_distance_to(base_station_coords)
        fspl = lin2db((4*np.pi*distance_3d*carrier_freq/speed_of_light)**2)
        res = 30.9 + (22.25 - 0.5 * np.log10(drone_coords.z)) * np.log10(distance_3d) + 20 * np.log10(carrier_freq/1e9)
        return max(fspl, res)

    @staticmethod
    def get_nlos_pathloss(los_pathloss, drone_coords: Coords3d, base_station_coords: Coords3d,
                          carrier_freq=DEFAULT_CARRIER_FREQ) -> float:
        distance_3d = drone_coords.get_distance_to(base_station_coords)
        return 32.4 + (43.2 - 7.6 * np.log10(drone_coords.z)) * np.log10(distance_3d) + 20 * np.log10(carrier_freq/1e9)

    @staticmethod
    def get_los_ss_fading_sample(drone_coords: Coords3d) -> float:
        # Small-scale LOS fading obeys Rice distribution
        k_shape_parameter = db2lin(4.217 * np.log10(drone_coords.z) + 5.787)
        return float(rice.rvs(k_shape_parameter, size=1, scale=1))

    @staticmethod
    def get_nlos_ss_fading_sample():
        # Small-scale NLOS fading obeys Rayleigh distribution
        return float(rayleigh.rvs(size=1))

    @staticmethod
    def get_successful_transmission_probability(drone_coords: Coords3d, base_station_coords: Coords3d = Coords3d(0, 0, 0),
                                                p_t=DRONE_DEFAULT_TRANSMISSION_POWER,
                                                snr_threshold=DEFAULT_SNR_THRESHOLD,
                                                carrier_freq=DEFAULT_CARRIER_FREQ) -> float:
        los_pl = CellularUrban3GppUmi.get_los_pathloss(drone_coords, base_station_coords, carrier_freq)
        nlos_pl = CellularUrban3GppUmi.get_nlos_pathloss(los_pl, drone_coords, base_station_coords, carrier_freq)

        x_los = NOISE_POWER_N0 * (10 ** (0.1 * los_pl)) * snr_threshold / p_t

        x_nlos = NOISE_POWER_N0 * 10 ** (0.1 * nlos_pl) * snr_threshold / p_t

        pr_los = CellularUrban3GppUmi.get_los_probability(drone_coords, base_station_coords)

        k_shape_parameter = db2lin(4.217 * np.log10(drone_coords.z) + 5.787)
        Fri = min(rice.cdf(x_los, k_shape_parameter, scale=1), 1)
        Fra = rayleigh.cdf(x_nlos)

        return pr_los * (1 - Fri) + (1 - pr_los) * (1 - Fra)

    @staticmethod
    def get_received_power_sample(p_t, drone_coords: Coords3d,
                                  base_station_coords: Coords3d, carrier_freq=DEFAULT_CARRIER_FREQ) -> float:

        los_gain = CellularUrban3GppUmi.get_los_ss_fading_sample(drone_coords) / 10 ** \
                   (0.1 * CellularUrban3GppUmi.get_los_pathloss(drone_coords, base_station_coords, carrier_freq))

        nlos_gain = CellularUrban3GppUmi.get_nlos_ss_fading_sample(drone_coords) / 10 ** \
                    (0.1 * CellularUrban3GppUmi.get_nlos_pathloss(drone_coords, base_station_coords, carrier_freq))

        pr_los = CellularUrban3GppUmi.get_los_probability(drone_coords, base_station_coords)

        return p_t * (pr_los * los_gain + (1 - pr_los) * nlos_gain)

    @staticmethod
    def get_received_snr_sample(p_t, drone_coords: Coords3d,
                                base_station_coords: Coords3d, carrier_freq=DEFAULT_CARRIER_FREQ) -> float:

        return CellularUrban3GppUmi.get_received_power_sample(p_t, drone_coords,
                                                              base_station_coords, carrier_freq) / NOISE_POWER_N0
