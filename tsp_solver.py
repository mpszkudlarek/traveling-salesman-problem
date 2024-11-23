"""
Implementation of Genetic Algorithm for solving Traveling Salesman Problem (TSP).
This module provides tools for finding the shortest possible route
that visits each city exactly once
and returns to the starting city using genetic algorithm optimization.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

from load_distances import load_distances


@dataclass
class GeneticConfig:
    """Configuration parameters for the genetic algorithm.

    Attributes:
        generations: Number of generations to evolve
        population_size: Size of the population in each generation
        crossover_rate: Probability of performing crossover (0-1)
        mutation_rate: Probability of mutation occurring (0-1)
        tournament_size: Number of individuals in tournament selection
        elitism: Whether to preserve the best individual in each generation
    """

    generations: int
    population_size: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int = 3
    elitism: bool = True


class TSPSolver:
    """Traveling Salesman Problem solver using genetic algorithm.

    This class implements a genetic algorithm to find the shortest possible route
    that visits each city exactly once and returns to the starting city.
    """

    def __init__(self, distance_file: str):
        """Initialize the TSP solver with distances from file.

        Args:
            distance_file: Path to the file containing distance matrix
        """
        self.distances, self.cities = load_distances(distance_file)
        self.best_route: List[str] = []
        self.best_distance: float = float("inf")
        self.city_count: int = len(self.cities)

    def route_distance(self, route: List[str]) -> float:
        """Calculate the total distance of a route.

        Args:
            route: List of cities in visit order

        Returns:
            Total distance of the route including return to start
        """
        return (
            sum(self.distances[(route[i], route[i + 1])] for i in range(len(route) - 1))
            + self.distances[(route[-1], route[0])]
        )

    def init_population(self, pop_size: int) -> List[List[str]]:
        """Initialize random population of routes.

        Args:
            pop_size: Size of population to generate

        Returns:
            List of random valid routes
        """
        return [random.sample(self.cities, len(self.cities)) for _ in range(pop_size)]

    def fitness(self, route: List[str]) -> float:
        """Calculate fitness value for a route.

        Args:
            route: List of cities in visit order

        Returns:
            Fitness value (inverse of route distance)
        """
        return 1 / self.route_distance(route)

    def tournament_selection(
        self, population: List[List[str]], fitnesses: List[float], tournament_size: int
    ) -> List[str]:
        """Select best individual from random tournament.

        Args:
            population: Current population of routes
            fitnesses: Fitness values for each route
            tournament_size: Number of individuals in tournament

        Returns:
            Selected route
        """
        tournament_indices = random.sample(range(len(population)), tournament_size)
        return max(
            (population[idx] for idx in tournament_indices),
            key=lambda route: fitnesses[population.index(route)],
        )

    def crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """Perform ordered crossover between two parent routes.

        Args:
            parent1: First parent route
            parent2: Second parent route

        Returns:
            New route combining features from both parents
        """
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))

        child: List[Optional[str]] = [None] * size

        for i in range(start, end):
            child[i] = parent1[i]

        remaining_cities = [city for city in parent2 if city not in child]
        j = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining_cities[j]
                j += 1

        return cast(List[str], child)

    def mutate(self, route: List[str], mutation_rate: float) -> None:
        """Perform swap mutation on a route with given probability.

        Args:
            route: Route to potentially mutate
            mutation_rate: Probability of mutation occurring
        """
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]

    def solve(self, algorithm_config: GeneticConfig) -> Tuple[List[float], List[str]]:
        """Execute genetic algorithm to find optimal route.

        Args:
            algorithm_config: Configuration parameters for the genetic algorithm

        Returns:
            Tuple containing:
                - List of best distances in each generation
                - Best route found
        """
        population = self.init_population(algorithm_config.population_size)
        best_distances: List[float] = []

        for _ in range(algorithm_config.generations):
            fitness = [self.fitness(route) for route in population]

            current_best = min(population, key=self.route_distance)
            current_distance = self.route_distance(current_best)
            best_distances.append(current_distance)

            if current_distance < self.best_distance:
                self.best_distance = current_distance
                self.best_route = current_best.copy()

            new_population: List[List[str]] = (
                [current_best.copy()] if algorithm_config.elitism else []
            )

            while len(new_population) < algorithm_config.population_size:
                parent1 = self.tournament_selection(
                    population, fitness, algorithm_config.tournament_size
                )
                parent2 = self.tournament_selection(
                    population, fitness, algorithm_config.tournament_size
                )

                if random.random() < algorithm_config.crossover_rate:
                    child1 = self.crossover(parent1, parent2)
                    child2 = self.crossover(parent1=parent2, parent2=parent1)
                else:
                    child1, child2 = parent1[:], parent2[:]

                self.mutate(child1, algorithm_config.mutation_rate)
                self.mutate(child2, algorithm_config.mutation_rate)

                new_population.extend([child1, child2])

            population = new_population[: algorithm_config.population_size]

        return best_distances, self.best_route
