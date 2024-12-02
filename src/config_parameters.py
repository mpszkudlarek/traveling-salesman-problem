"""
Module that defines the configuration parameters for the genetic algorithm.

This module contains the `GeneticConfig` dataclass, which is used to store configuration
settings for the genetic algorithm. These settings include the number of generations,
population size, crossover and mutation rates, and tournament size.
"""

from dataclasses import dataclass
from typing import Optional

from src.selection_method_enums import (CrossoverMethod, MutationMethod,
                                        SelectionMethod)


@dataclass(frozen=True)
class SelectionConfig:
    selection_method: SelectionMethod
    tournament_size: int
    selection_pressure: Optional[float] = None
    random_seed: Optional[int] = None

    def validate(self):
        if self.tournament_size <= 0:
            raise ValueError("Tournament size must be a positive integer")

        if self.selection_method not in ["tournament", "elitism", "ranking"]:
            raise ValueError(f"Invalid selection method: {self.selection_method}")

        if self.selection_method == "ranking" and self.selection_pressure is None:
            raise ValueError("Selection pressure is required for ranking selection")


@dataclass(frozen=True)
class CrossoverConfig:
    crossover_method: CrossoverMethod
    crossover_rate: float

    def validate(self):
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")

        if self.crossover_method not in ["spc", "cx", "pmx"]:
            raise ValueError(f"Invalid crossover method: {self.crossover_method}")


@dataclass(frozen=True)
class MutationConfig:
    mutation_rate: float
    mutation_method: MutationMethod

    def validate(self):
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")

        if self.mutation_method not in ["ad_swap", "inversion"]:
            raise ValueError(f"Invalid chromosome operation: {self.mutation_method}")


@dataclass(frozen=True)
class GeneticConfig:
    generations: int
    population_size: int
    selection_config: SelectionConfig
    crossover_config: CrossoverConfig
    mutation_config: MutationConfig

    def validate(self):
        if self.generations <= 0:
            raise ValueError("Generations must be a positive integer")

        if self.population_size <= 0:
            raise ValueError("Population size must be a positive integer")

        # Validate nested configurations
        self.selection_config.validate()
        self.crossover_config.validate()
        self.mutation_config.validate()
