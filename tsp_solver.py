"""
Implementation of Genetic Algorithm for solving the Traveling Salesman Problem (TSP).
This module provides tools for finding the shortest possible route
that visits each city exactly once and returns to the starting city
using genetic algorithm optimization.
"""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from base_tsp_solver import BaseTSPSolver
from genetic_config import GeneticConfig
from load_distances import load_distances
from selection_factory import GeneticAlgorithmFactory

# Define defaults for configuration
DEFAULTS = {
    "folder": "input",
    "tournament_size": 3,
}


class DistanceMatrixError(Exception):
    """Custom exception for errors related to the distance matrix."""


class TSPSolver(BaseTSPSolver):
    """
    Traveling Salesman Problem solver using a genetic algorithm.

    This class implements a genetic algorithm to find the shortest possible route
    that visits each city exactly once and returns to the starting city.
    """

    def __init__(self, distance_file: str, config: Optional[Dict[str, str]] = None):
        """
        Initialize the solver with a distance matrix from a file.

        Args:
            distance_file (str): Name of the file containing the distance matrix.
            config (Dict[str, str]): Configuration dictionary with keys like "folder".
                                      Defaults to using values in DEFAULTS.
        """
        if config is None:
            config = {}
        self.config = {
            **DEFAULTS,
            **(config or {}),
        }  # Merge defaults with provided config
        # Separate unpacking and type annotation
        distance_matrix, cities = self._load_distances(
            distance_file, str(self.config["folder"])
        )

        # Assign the values to instance variables
        self.distance_matrix = distance_matrix
        self.cities: List[str] = cities  # Now correctly annotate self.cities

        # Explicitly typing the city_indices dictionary
        self.city_indices: Dict[str, int] = {
            city: idx for idx, city in enumerate(self.cities)
        }

        self.best_route: List[str] = []
        self.best_distance: float = float("inf")
        self.city_count: int = len(self.cities)  # This should now be correct

    def _load_distances(
        self, file_name: str, folder: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load and validate the distance matrix from a file.

        Args:
            file_name (str): Name of the file containing distance data.
            folder (str): Folder where the file is located.

        Returns:
            Tuple[np.ndarray, List[str]]:
                - A NumPy array representing the distance matrix.
                - A list of city names corresponding to the rows/columns of the matrix.

        Raises:
            DistanceMatrixError: If the matrix is invalid or data is missing.
        """
        city_distances, cities = load_distances(file_name, folder)

        # Convert dictionary of distances to a symmetric NumPy matrix
        num_cities = len(cities)
        matrix = np.zeros((num_cities, num_cities))

        for (city1, city2), distance in city_distances.items():
            idx1 = cities.index(city1)
            idx2 = cities.index(city2)
            matrix[idx1, idx2] = distance
            matrix[idx2, idx1] = distance  # Ensure symmetry

        return matrix, cities

    @lru_cache(maxsize=1024)
    def route_distance(self, route: Tuple[str, ...]) -> float:
        """
        Calculate the total distance of a given route.

        Args:
            route (Tuple[str, ...]): A tuple of city names representing the route.

        Returns:
            float: The total distance of the route.
        """
        indices = [
            self.city_indices[city] for city in route
        ]  # Convert city names to indices
        distances = self.distance_matrix[
            indices[:-1], indices[1:]
        ]  # Pairwise distances between cities, so for indices [0, 1, 2, 3],
        # this would be [0-1, 1-2, 2-3]
        return float(
            np.sum(distances) + self.distance_matrix[indices[-1], indices[0]]
        )  # Add the distance from the last city back to the first

    def init_population(self, pop_size: int) -> List[Tuple[str, ...]]:
        """
        Initialize the population for the genetic algorithm.

        Args:
            pop_size (int): Number of individuals in the initial population.

        Returns:
            List[Tuple[str, ...]]: A list of routes, each represented as a tuple of city names.
        """
        cities_array = np.array(self.cities)
        # Generate random permutations using permutation function from numpy
        return [tuple(np.random.permutation(cities_array)) for _ in range(pop_size)]

    def _single_point_crossover(
        self, parent1: Tuple[str, ...], parent2: Tuple[str, ...]
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """
        Perform single-point crossover between two parent routes.

        Args:
            parent1 (Tuple[str, ...]): The first parent route.
            parent2 (Tuple[str, ...]): The second parent route.

        Returns:
            Tuple[Tuple[str, ...], Tuple[str, ...]]: Two child routes resulting from crossover.
        """
        size = len(parent1)
        point = np.random.randint(
            1, size
        )  # Random crossover point, 1 because we don't want to start at the first city

        # Create children by combining parts of parents
        child1 = list(
            parent1[:point]
        ) + [  # Take the first part of parent1 up to the crossover point
            gene
            for gene in parent2
            if gene
            not in parent1[
                :point
            ]  # Add genes from parent2 that are not already in child1
        ]
        child2 = list(
            parent2[:point]
        ) + [  # Take the first part of parent2 up to the crossover point
            gene
            for gene in parent1
            if gene
            not in parent2[
                :point
            ]  # Add genes from parent1 that are not already in child2
        ]

        return tuple(child1), tuple(child2)

    def mutate(self, route: List[str], mutation_rate: float) -> None:
        """
        Apply mutation to a route by reversing a random sub-path.

        Args:
            route (List[str]): The route to mutate (list of city names).
            mutation_rate (float): Probability of mutation occurring.
        """
        if (
            np.random.random() < mutation_rate
        ):  # generate a random number between 0 and 1 and check if it's less than the mutation rate
            size = len(route)  # get the number of cities
            # Select a random sub-path by picking two distinct indices
            start, end = sorted(np.random.choice(range(size), 2, replace=False))
            # Reverse the selected sub-path
            route[start:end] = reversed(route[start:end])

    def calculate_fitness(self, route: Tuple[str, ...]) -> float:
        """
        Calculate the fitness of a route. The fitness is the inverse of the route distance.

        Args:
            route (Tuple[str, ...]): The route to evaluate, represented as a tuple of city names.

        Returns:
            float: The fitness score (1 / distance), where lower distance yields higher fitness.
        """
        distance = self.route_distance(route)  # calculate the distance of the route
        return (
            1 / distance if distance != 0 else float("inf")
        )  # Return the inverse of the distance as fitness score (avoid division by zero)

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
        fitness_scores = np.array(
            [self.calculate_fitness(route) for route in population]
        )
        best_idx = np.argmax(
            fitness_scores
        )  # Best fitness corresponds to the lowest distance
        best_route = population[best_idx]
        best_distance = (
            1 / fitness_scores[best_idx]
        )  # Convert fitness to distance (should be a scalar)

        # Ensure best_distance is a scalar (in case of possible numpy behavior)
        best_distance = float(best_distance)

        return fitness_scores, best_route, best_distance

    def generate_new_population(
        self,
        population: List[Tuple[str, ...]],
        fitness_scores: np.ndarray,
        config: GeneticConfig,
    ) -> List[Tuple[str, ...]]:
        """
        Generate a new population by applying selection, crossover, and mutation.

        Args:
            population (List[Tuple[str, ...]]): Current population.
            fitness_scores (np.ndarray): Fitness scores of each individual in the population.
            config (GeneticConfig): Configuration parameters for selection, crossover, and mutation.

        Returns:
            List[Tuple[str, ...]]: The new population after selection, crossover, and mutation.
        """
        selection_method = GeneticAlgorithmFactory.get_selection_method(
            config.selection_method
        )

        new_population: List[Tuple[str, ...]] = []

        while len(new_population) < config.population_size:
            parents = [
                selection_method(population, fitness_scores, config.tournament_size)
                for _ in range(2)
            ]

            # Apply crossover
            if np.random.random() < config.crossover_rate:
                child1, child2 = self._single_point_crossover(parents[0], parents[1])
                children = [list(child1), list(child2)]
            else:
                children = [list(p) for p in parents]

            # Apply mutation
            for child in children:
                self.mutate(child, config.mutation_rate)
                new_population.append(tuple(child))

        return new_population[
            : config.population_size
        ]  # Ensure the population size is maintained

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
            # Evaluate population
            fitness_scores, current_best, current_distance = self.evaluate_population(
                population
            )

            # Append best distance
            best_distances.append(float(current_distance))

            # Update best route if current one is better
            if current_distance < self.best_distance:
                self.best_distance = current_distance
                self.best_route = list(current_best)

            # Generate new population for the next generation
            population = self.generate_new_population(
                population, fitness_scores, config
            )

        return best_distances, self.best_route
