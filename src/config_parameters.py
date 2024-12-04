"""
Module that defines the configuration parameters for the genetic algorithm.

This module contains the `GeneticConfig` dataclass, which is used to store configuration
settings for the genetic algorithm. These settings include the number of generations,
population size, crossover and mutation rates, and tournament size.
"""

from dataclasses import dataclass
from typing import Optional

from src.tsp_genetic_enums import (CrossoverMethod, MutationMethod,
                                   SelectionMethod)


@dataclass(frozen=True)
class SelectionConfig:
    """
    Configuration for selection methods in the genetic algorithm.

    Attributes:
        selection_method (SelectionMethod): The selection method to use.
        tournament_size (int): Size of the tournament for selection.
        selection_pressure (Optional[float]): Selection pressure for ranking method.
        random_seed (Optional[int]): Random seed for reproducibility.
    """

    selection_method: SelectionMethod
    tournament_size: int
    selection_pressure: Optional[float] = None
    random_seed: Optional[int] = None

    def validate(self):
        """
        Validate the configuration of the selection method.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.tournament_size <= 0:
            raise ValueError("Tournament size must be a positive integer")

        if self.selection_method not in ["tournament", "elitism", "ranking"]:
            raise ValueError(f"Invalid selection method: {self.selection_method}")

        if self.selection_method == "ranking" and self.selection_pressure is None:
            raise ValueError("Selection pressure is required for ranking selection")


@dataclass(frozen=True)
class CrossoverConfig:
    """
    Configuration for crossover methods in the genetic algorithm.

    Attributes:
        crossover_method (CrossoverMethod): The crossover method to use.
        crossover_rate (float): Probability of applying the crossover method.
    """

    crossover_method: CrossoverMethod
    crossover_rate: float

    def validate(self):
        """
        Validate the configuration of the crossover method.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")

        if self.crossover_method not in ["spc", "cx", "pmx"]:
            raise ValueError(f"Invalid crossover method: {self.crossover_method}")


@dataclass(frozen=True)
class MutationConfig:
    """
    Configuration for mutation methods in the genetic algorithm.

    Attributes:
        mutation_rate (float): Probability of applying the mutation method.
        mutation_method (MutationMethod): The mutation method to use.
    """

    mutation_rate: float
    mutation_method: MutationMethod

    def validate(self):
        """
        Validate the configuration of the mutation method.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")

        if self.mutation_method not in ["ad_swap", "inversion"]:
            raise ValueError(f"Invalid chromosome operation: {self.mutation_method}")


@dataclass(frozen=True)
class GeneticConfig:
    """
    Main configuration for the genetic algorithm.

    Attributes:
        generations (int): Number of generations to run the algorithm.
        population_size (int): Number of individuals in each generation.
        selection_config (SelectionConfig): Configuration for selection methods.
        crossover_config (CrossoverConfig): Configuration for crossover methods.
        mutation_config (MutationConfig): Configuration for mutation methods.
    """

    generations: int
    population_size: int
    selection_config: SelectionConfig
    crossover_config: CrossoverConfig
    mutation_config: MutationConfig

    def validate(self):
        """
        Validate the overall genetic algorithm configuration.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.generations <= 0:
            raise ValueError("Generations must be a positive integer")

        if self.population_size <= 0:
            raise ValueError("Population size must be a positive integer")

        # Validate nested configurations
        self.selection_config.validate()
        self.crossover_config.validate()
        self.mutation_config.validate()
