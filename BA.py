from time import time

import numpy as np


# Bat Algorithm
def BA(bats, fitness, min_value, max_value, max_iterations):
    ct = time()

    A = 0.5
    alpha = 0.9
    gamma = 0.9
    num_bats, num_dimensions = bats.shape[0], bats.shape[1]
    velocities = np.zeros((num_bats, num_dimensions))
    frequencies = np.zeros(num_bats)
    best_solution = None
    Convergence = np.zeros(max_iterations)
    best_fitness = float('inf')
    min_f = np.min(frequencies)
    max_f = np.max(frequencies)

    for iteration in range(max_iterations):
        for i in range(num_bats):
            # Update frequency and velocity
            frequencies[i] = min_f + (max_f - min_f) * np.random.random()
            velocities[i] += (bats[i] - best_solution) * frequencies[i]

            # Update bat position
            new_solution = bats[i] + velocities[i]
            new_solution = np.clip(new_solution, min_value, max_value)

            # Evaluate new solution
            new_fitness = fitness(new_solution)

            # Update if the new solution is better or with probability gamma
            if np.random.random() < gamma and new_fitness < fitness(bats[i]):
                bats[i] = new_solution
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness

        # Update loudness and pulse rate
        loudness = alpha * loudness
        pulse_rate = np.exp(-A * iteration)

        min_f = np.min(frequencies)
        max_f = np.max(frequencies)
        Convergence[iteration] = best_fitness
    Time = time() - ct
    return best_fitness, Convergence, best_solution, best_fitness, Time

