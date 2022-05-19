from src.base_station import BaseStation
import numpy as np
from src.parameters import DISCOUNT_RATIO
import matplotlib.pyplot as plt


def run_simulation(N_CYCLES=10000, N_ITERATIONS=20, q_type='SA', plot=True):
    rewards_per_cycle = np.zeros(N_CYCLES)
    for iteration_idx in range(1, N_ITERATIONS + 1):
        rewards_iteration = np.zeros(N_CYCLES)
        bs = BaseStation(q_type=q_type)
        for cycle_idx in range(N_CYCLES):
            rewards = bs.perform_cycle()
            rewards_iteration[cycle_idx] = rewards.sum()

            print("Iteration:", iteration_idx, "Cycle:", cycle_idx,
                  "Avg of last 50: ", rewards_iteration[max(0, cycle_idx - 50):cycle_idx].mean())
        rewards_per_cycle = rewards_per_cycle + (rewards_iteration - rewards_per_cycle) / iteration_idx

    rewards_discounted_cumsum = np.zeros(N_CYCLES)
    for i in range(len(rewards_per_cycle)):
        sum = 0
        discount = 1
        for j in range(i, len(rewards_per_cycle)):
            sum += rewards_per_cycle[j] * discount
            discount *= DISCOUNT_RATIO
        rewards_discounted_cumsum[i] = sum
    if plot:
        plt.plot(rewards_discounted_cumsum)
        plt.show()
    return rewards_discounted_cumsum, rewards_per_cycle


if __name__ == '__main__':
    # sa_discounted_cumsum, sa_avg_rewards_cycle = run_simulation()
    ma_discounted_cumsum, ma_avg_rewards_cycle = run_simulation(q_type='MA')
    # np.savetxt('results/sa_discounted_cumsum.txt', sa_discounted_cumsum)
    # np.savetxt('results/sa_avg_rewards_cycle.txt', sa_avg_rewards_cycle)
    # np.savetxt('results/ma_discounted_cumsum.txt', ma_discounted_cumsum)
    # np.savetxt('results/ma_avg_rewards_cycle.txt', ma_avg_rewards_cycle)
    #
    # plt.plot(sa_discounted_cumsum)
    # plt.plot(ma_discounted_cumsum)
    # plt.xlabel('Number of Cycles')
    # plt.ylabel('Average Cumulative Sum of Discounted Rewards')
    # plt.legend(['Single Agent', 'Opponent Modeling'], loc="upper left")
    # plt.show()
