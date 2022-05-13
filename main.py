from src.base_station import BaseStation
import numpy as np
from src.parameters import DISCOUNT_RATIO
import matplotlib.pyplot as plt

if __name__ == '__main__':
    bs = BaseStation()

    N_CYCLES = 10000
    N_ITERATIONS = 1
    rewards_per_cycle = np.zeros(N_CYCLES)
    rewards_iteration = np.zeros(N_CYCLES)
    for iteration_idx in range(1, N_ITERATIONS + 1):
        for cycle_idx in range(N_CYCLES):
            rewards = bs.perform_cycle()
            rewards_iteration[cycle_idx] = rewards.sum()

            print("Iteration:", iteration_idx, "Cycle:", cycle_idx,
                  "Avg of last 20: ", rewards_iteration[max(0, cycle_idx - 20):cycle_idx].mean())
        rewards_per_cycle = rewards_per_cycle + (rewards_iteration - rewards_per_cycle) / iteration_idx

    rewards_discounted_cumsum = np.zeros(N_CYCLES)
    for i in range(len(rewards_per_cycle)):
        sum = 0
        discount = 1
        for j in range(i, len(rewards_per_cycle)):
            sum += rewards_per_cycle[j] * discount
            discount *= DISCOUNT_RATIO
        rewards_discounted_cumsum[i] = sum

    plt.plot(rewards_discounted_cumsum)
    plt.show()
