"""
Module that defines the base class for solvers of the Traveling Salesman Problem (TSP).

This module includes the abstract base class `BaseTSPSolver`, which defines the interface
for all TSP solvers that use genetic algorithms. Subclasses of this base class should
implement the methods outlined here to solve the problem using specific techniques.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from src.config_parameters import GeneticConfig


class BaseTSPSolver(ABC):
    """
    Abstract base class for Traveling Salesman Problem (TSP) solvers.

    This class defines the common interface for TSP solvers using genetic algorithms.
    Any solver that inherits from this class must implement the methods defined here.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the base TSP solver.

        This method should be implemented by subclasses to initialize necessary attributes
        for solving the Traveling Salesman Problem.
        """

    @abstractmethod
    def init_population(self, pop_size: int) -> List[Tuple[str, ...]]:
        """
        Initialize the population of routes.

        Args:
            pop_size (int): The number of routes to be initialized in the population.

        Returns:
            List[Tuple[str, ...]]: A list of initialized routes,
            each represented as a tuple of city names.
        """

    @abstractmethod
    def solve(self, config: GeneticConfig) -> Tuple[List[float], List[str]]:
        """
        Solve the Traveling Salesman Problem using the genetic algorithm.

        Args:
            config (GeneticConfig): Configuration parameters for the genetic algorithm,
            such as population size,
            crossover rate, mutation rate, etc.

        Returns:
            Tuple[List[float], List[str]]: The best distances found for each
            generation and the corresponding best route.
        """

    @abstractmethod
    def route_distance(self, route: Tuple[str, ...]) -> float:
        """
        Calculate the total distance of a given route.

        Args:
            route (Tuple[str, ...]): A route represented as a tuple of city names.

        Returns:
            float: The total distance of the route.
        """

    @abstractmethod
    def calculate_fitness(self, route: Tuple[str, ...]) -> float:
        """
        Calculate the fitness of a given route.

        Fitness is generally the inverse of the route's distance, with shorter
        routes having higher fitness.

        Args:
            route (Tuple[str, ...]): A route represented as a tuple of city names.

        Returns:
            float: The fitness score of the route.
        """

    @abstractmethod
    def generate_new_population(
        self,
        population: List[Tuple[str, ...]],
        fitness_scores: np.ndarray,
        config: GeneticConfig,
    ) -> List[Tuple[str, ...]]:
        """
        Generate a new population based on the current population and fitness scores.

        This method typically performs selection, crossover, and mutation to
        generate a new set of routes for the next generation.

        Args:
            population (List[Tuple[str, ...]]): The current population of routes.
            fitness_scores (np.ndarray): The fitness scores
            corresponding to the current population.
            config (GeneticConfig): Configuration parameters for the genetic algorithm.

        Returns:
            List[Tuple[str, ...]]: The newly generated population of routes.
        """
