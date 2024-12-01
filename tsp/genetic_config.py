"""
Module that defines the configuration parameters for the genetic algorithm.

This module contains the `GeneticConfig` dataclass, which is used to store configuration
settings for the genetic algorithm. These settings include the number of generations,
population size, crossover and mutation rates, and tournament size.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneticConfig:
    """
    Configuration parameters for the genetic algorithm.

    Attributes:
        generations (int): Number of generations to evolve, i.e., iterations,
            which serves as the stop condition for the algorithm.
        population_size (int): Size of the population in each generation.
        crossover_rate (float): Probability of performing crossover (value between 0 and 1).
        mutation_rate (float): Probability of mutation occurring (value between 0 and 1).
        tournament_size (int): Number of individuals in tournament selection.
        selection_method (str): Method of selection to use in the genetic algorithm.
        crossover_method (str): Method of crossover to use in the genetic algorithm.
        chromosome_operation (str): Method of chromosome operation to use in the genetic algorithm.
    """

    generations: int
    population_size: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int
    selection_method: str
    crossover_method: str
    chromosome_operation: str
