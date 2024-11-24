"""
Implementation of Genetic Algorithm for solving Traveling Salesman Problem (TSP).
This module provides tools for finding the shortest possible route
that visits each city exactly once
and returns to the starting city using genetic algorithm optimization.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import numpy as np

from load_distances import load_distances


class DistanceMatrixError(Exception):
    """Custom exception for distance matrix-specific errors."""


@dataclass(frozen=True)
class GeneticConfig:
    """Configuration parameters for the genetic algorithm.

    Attributes:
        generations: Number of generations to evolve, i.e., iterations,
        this is stop condition for the algorithm
        population_size: Size of the population in each generation
        crossover_rate: Probability of performing crossover (0-1)
        mutation_rate: Probability of mutation occurring (0-1)
        tournament_size: Number of individuals in tournament selection
    """

    generations: int
    population_size: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int = 3


class TSPSolver:
    """Traveling Salesman Problem solver using genetic algorithm.

    This class implements a genetic algorithm to find the shortest possible route
    that visits each city exactly once and returns to the starting city.
    """

    def __init__(self, distance_file: str, folder: str = "input"):
        """
        Initialize solver with distance matrix from file.

        Args:
            distance_file: Name of the file containing distance matrix
            folder: Folder containing the distance file
        """
        self.distance_matrix, self.cities = self._load_distances(distance_file, folder)
        self.city_indices = {city: idx for idx, city in enumerate(self.cities)}
        self.best_route: List[str] = []
        self.best_distance: float = float("inf")
        self.city_count: int = len(self.cities)

    def _load_distances(
        self, file_name: str, folder: str = "input"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load and validate distance matrix from file using the `load_distances` function.

        Returns:
            Tuple containing:
                - NumPy array of distances
                - List of city names
        """
        city_distances, cities = load_distances(file_name, folder)

        # Convert dictionary of distances to NumPy array
        num_cities = len(cities)
        matrix = np.zeros((num_cities, num_cities))

        for (city1, city2), distance in city_distances.items():
            idx1 = cities.index(city1)
            idx2 = cities.index(city2)
            matrix[idx1, idx2] = distance
            matrix[idx2, idx1] = distance  # Ensure symmetry

        return matrix, cities

    @lru_cache(maxsize=1024)
    def route_distance(self, route: tuple) -> float:
        """Calculate route distance using numpy operations and caching."""
        indices = [self.city_indices[city] for city in route]
        distances = self.distance_matrix[indices[:-1], indices[1:]]
        return float(np.sum(distances) + self.distance_matrix[indices[-1], indices[0]])

    def init_population(self, pop_size: int) -> List[tuple]:
        """Initialize population using numpy's permutation."""
        population = []
        cities_array = np.array(self.cities)
        for _ in range(pop_size):
            np.random.shuffle(cities_array)
            population.append(tuple(cities_array))
        return population

    def _single_point_crossover(self, parent1: tuple, parent2: tuple) -> tuple:
        """Single-point crossover operator."""
        size = len(parent1)
        point = np.random.randint(
            1, size
        )  # Random crossover point (excluding first and last)

        # Create children by combining parts of parents
        child1 = list(parent1[:point]) + [
            gene for gene in parent2 if gene not in parent1[:point]
        ]
        child2 = list(parent2[:point]) + [
            gene for gene in parent1 if gene not in parent2[:point]
        ]

        return tuple(child1), tuple(child2)

    def _edge_mutation(self, route: list, mutation_rate: float) -> None:
        """Edge mutation - better for preserving good sub-paths."""
        if np.random.random() < mutation_rate:
            size = len(route)
            start, end = sorted(np.random.randint(0, size, 2))
            route[start:end] = reversed(route[start:end])

    def solve(self, config: GeneticConfig) -> Tuple[List[float], List[str]]:
        """Execute optimized genetic algorithm to find best route."""
        population = self.init_population(config.population_size)
        best_distances = []

        for _ in range(config.generations):
            # Calculate fitness scores
            fitness_scores = np.array(
                [1 / self.route_distance(route) for route in population]
            )

            # Find current best
            best_idx = np.argmax(fitness_scores)
            current_best = population[best_idx]
            current_distance = self.route_distance(current_best)
            best_distances.append(current_distance)

            if current_distance < self.best_distance:
                self.best_distance = current_distance
                self.best_route = list(current_best)

            # Create new population
            new_population: List[tuple] = []

            # Generate remaining population
            while len(new_population) < config.population_size:
                # Tournament selection
                parents = [
                    self._tournament_select(
                        population, fitness_scores, config.tournament_size
                    )
                    for _ in range(2)
                ]

                # Crossover and mutation
                if np.random.random() < config.crossover_rate:
                    child1, child2 = self._single_point_crossover(
                        parents[0], parents[1]
                    )
                    children = [list(child1), list(child2)]
                else:
                    children = [list(p) for p in parents]

                for child in children:
                    self._edge_mutation(child, config.mutation_rate)
                    new_population.append(tuple(child))

            population = new_population[: config.population_size]

        return best_distances, self.best_route

    def _tournament_select(
        self, population: List[tuple], fitness_scores: np.ndarray, tournament_size: int
    ) -> tuple:
        """Tournament selection using numpy operations."""
        tournament_idx = np.random.choice(
            len(population), tournament_size, replace=False
        )
        winner_idx = tournament_idx[np.argmax(fitness_scores[tournament_idx])]
        return population[winner_idx]
