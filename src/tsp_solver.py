"""
Implementation of Genetic Algorithm for solving the Traveling Salesman Problem (TSP).
This module provides tools for finding the shortest possible route
that visits each city exactly once and returns to the starting city
using genetic algorithm optimization.
"""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import cuda  # type: ignore

from src.base_tsp_solver import BaseTSPSolver
from src.config_parameters import GeneticConfig
from src.gpu_accelerator import GPUAccelerator
from src.load_distances import load_distances
from src.tsp_factory import TspFactory

# Define defaults for configuration
DEFAULTS = {"folder": "input"}


class DistanceMatrixError(Exception):
    """Custom exception for errors related to the distance matrix."""


class TSPSolver(BaseTSPSolver):
    """
    Traveling Salesman Problem solver using a genetic algorithm.

    This class implements a genetic algorithm to find the shortest possible route
    that visits each city exactly once and returns to the starting city.
    """

    def __init__(
        self, distance_file: str, config: Optional[Dict[str, str]] = None, use_gpu: bool = False
    ) -> None:
        """
        Initialize the solver with a distance matrix from a file.

        Args:
            distance_file (str): Name of the file containing the distance matrix.
            config (Dict[str, str]): Configuration dictionary with keys like "folder".
                                      Defaults to using values in DEFAULTS.
        """
        self.config = {**DEFAULTS, **(config or {})}
        distance_matrix, cities = self._load_distances(distance_file, self.config["folder"])

        self.distance_matrix = distance_matrix

        self.use_gpu = use_gpu
        if self.use_gpu:
            try:
                if cuda.is_available():
                    self.gpu_accelerator = GPUAccelerator()
                else:
                    self.use_gpu = False
                    print("Warning: GPU requested but CUDA is not available. Using CPU instead.")
                    self.gpu_accelerator = None
            except ImportError:
                self.use_gpu = False
                print("Warning: Numba CUDA not installed. Using CPU instead.")
                self.gpu_accelerator = None
        else:
            self.gpu_accelerator = None

        self.cities: List[str] = cities
        self.city_indices: Dict[str, int] = {city: idx for idx, city in enumerate(self.cities)}

        self.best_route: List[str] = []
        self.best_distance: float = float("inf")
        self.city_count: int = len(self.cities)

    def _load_distances(self, file_name: str, folder: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load the distance matrix from a file.
        """
        return load_distances(file_name, folder)

    @lru_cache(maxsize=1024)
    def route_distance(self, route: Tuple[str, ...]) -> float:
        """
        Calculate the total distance of a given route.

        Args:
            route (Tuple[str, ...]): A tuple of city names representing the route.

        Returns:
            float: The total distance of the route.
        """
        indices = [self.city_indices[city] for city in route]
        distances = self.distance_matrix[indices[:-1], indices[1:]]
        return float(np.sum(distances) + self.distance_matrix[indices[-1], indices[0]])

    def init_population(self, pop_size: int) -> List[Tuple[str, ...]]:
        """
        Initialize the population for the genetic algorithm.

        Args:
            pop_size (int): Number of individuals in the initial population.

        Returns:
            List[Tuple[str, ...]]: A list of routes, each represented as a tuple of city names.
        """
        cities_array = np.array(self.cities)
        return [tuple(np.random.permutation(cities_array)) for _ in range(pop_size)]

    def calculate_fitness(self, route: Tuple[str, ...]) -> float:
        """
        Calculate the fitness of a route. The fitness is the inverse of the route distance.

        Args:
            route (Tuple[str, ...]): The route to evaluate, represented as a tuple of city names.

        Returns:
            float: The fitness score (1 / distance), where lower distance yields higher fitness.
        """
        distance = self.route_distance(route)
        return 1 / distance if distance != 0 else float("inf")

    def evaluate_population(
        self, population: List[Tuple[str, ...]]
    ) -> Tuple[np.ndarray, Tuple[str, ...], float]:
        """
        Evaluate the population by calculating fitness scores and finding the best individual.

        Args:
            population (List[Tuple[str, ...]]): List of routes representing the population.

        Returns:
            Tuple:
                - np.ndarray: Fitness scores of each route.
                - Tuple[str, ...]: Best route in the current population.
                - float: Distance of the best route.
        """
        fitness_scores = np.array([self.calculate_fitness(route) for route in population])
        best_idx = np.argmax(fitness_scores)
        best_route = population[best_idx]
        best_distance = 1 / fitness_scores[best_idx]
        return fitness_scores, best_route, float(best_distance)

    def generate_new_population(self, population, fitness_scores, config):
        """
        Generate a new population by applying selection, crossover, and mutation.

        Args:
            population (List[Tuple[str, ...]]): Current population.
            fitness_scores (np.ndarray): Fitness scores of each individual.
            config (GeneticConfig): Configuration parameters.

        Returns:
            List[Tuple[str, ...]]: New population after genetic operations.
        """
        selection_method = TspFactory.get_selection_method(config.selection_config.selection_method)
        method_params = config.selection_config.get_method_params()

        crossover_method = TspFactory.get_crossover_method(config.crossover_config.crossover_method)
        mutation_method = TspFactory.get_mutation_method(config.mutation_config.mutation_method)

        new_population: List[Tuple[str, ...]] = []
        target_size = config.population_size

        while len(new_population) < target_size:
            parents = [selection_method(population, fitness_scores, **method_params) for _ in range(2)]

            if np.random.random() < config.crossover_config.crossover_rate:
                child1, child2 = crossover_method(parents[0], parents[1])
            else:
                child1, child2 = parents

            for child in (child1, child2):
                if len(new_population) < target_size:
                    mutated_child = mutation_method(child, config.mutation_config.mutation_rate)
                    new_population.append(mutated_child)

        return new_population

    def solve(self, config: GeneticConfig) -> Tuple[List[float], List[str]]:
        """
        Execute the genetic algorithm to solve the TSP.

        Args:
            config (GeneticConfig): Configuration parameters for the algorithm.

        Returns:
            Tuple[List[float], List[str]]:
                - A list of best distances at each generation.
                - The best route found by the algorithm.
        """
        population = self.init_population(config.population_size)
        best_distances = []

        for _ in range(config.generations):
            fitness_scores, current_best, current_distance = self.evaluate_population(population)
            best_distances.append(current_distance)

            if current_distance < self.best_distance:
                self.best_distance = current_distance
                self.best_route = list(current_best)

            population = self.generate_new_population(population, fitness_scores, config)

        return best_distances, self.best_route
