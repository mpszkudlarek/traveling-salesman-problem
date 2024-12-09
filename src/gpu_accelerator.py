"""
GPU Acceleration Module for Traveling Salesman Problem Genetic Algorithm

This module provides GPU-accelerated fitness calculation using CUDA and Numba,
enabling parallel computation of route distances for large populations.
"""

import numpy as np
import numpy.typing as npt
from numba import cuda  # type: ignore


@cuda.jit
def calculate_route_distances_kernel(
    distance_matrix: npt.NDArray[np.float32],
    population: npt.NDArray[np.int32],
    fitness_scores: npt.NDArray[np.float32],
):
    """
    CUDA kernel for parallel route distance calculation.

    Computes the total route distance for each route in the population
    using parallel GPU computation.

    Args:
        distance_matrix (np.ndarray): Matrix of distances between cities
        population (np.ndarray): Population of routes to evaluate
        fitness_scores (np.ndarray): Output array to store computed fitness scores
    """
    idx = cuda.grid(1)
    if idx < population.shape[0]:
        total = 0.0
        route_len = population.shape[1]
        for i in range(route_len - 1):
            city1 = population[idx, i]
            city2 = population[idx, i + 1]
            total += distance_matrix[city1, city2]
        total += distance_matrix[population[idx, -1], population[idx, 0]]
        fitness_scores[idx] = total


class GPUAccelerator:
    """
    GPU Accelerator for Traveling Salesman Problem Genetic Algorithm.

    Provides methods to leverage CUDA for parallel fitness calculation
    of population routes.

    Attributes:
        threads_per_block (int): Number of CUDA threads per block
    """

    def __init__(self):
        """
        Initialize the GPU Accelerator with default thread configuration.
        """
        self.threads_per_block: int = 1024

    def calculate_population_fitness(
        self, population: npt.NDArray[np.int32], distance_matrix: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Calculate fitness scores for entire population using GPU acceleration.

        Args:
            population (np.ndarray): Array of routes to evaluate
            distance_matrix (np.ndarray): Matrix of distances between cities

        Returns:
            np.ndarray: Computed fitness scores for each route
        """
        pop_array = np.array(population, dtype=np.int32)
        fitness_scores = np.zeros(len(population), dtype=np.float32)

        d_matrix = cuda.to_device(distance_matrix)
        d_pop = cuda.to_device(pop_array)
        d_fitness = cuda.to_device(fitness_scores)

        blocks_per_grid = (pop_array.shape[0] + self.threads_per_block - 1) // self.threads_per_block

        calculate_route_distances_kernel[blocks_per_grid, self.threads_per_block](d_matrix, d_pop, d_fitness)

        return d_fitness.copy_to_host()
