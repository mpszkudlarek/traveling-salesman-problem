"""
Selection method factory for genetic algorithms.

This module provides a factory class to dynamically retrieve and create
selection methods for genetic algorithms. It allows flexible selection
of different strategies (tournament, elitism, ranking) based on a string
identifier.

The factory supports:
- Tournament selection: Competitive selection of best individuals from small groups
- Elitism selection: Preference for top-performing individuals
- Ranking selection: Selection based on individual ranks with configurable pressure
"""

from typing import Callable, List, Tuple

import numpy as np
from selection_method import (elitism_selection, ranking_selection,
                              tournament_selection)

from tsp.chromosome_operation_method import (adjacent_swap_mutation,
                                             inversion_mutation)
from tsp.crossover_method import (cycle_crossover, partially_mapped_crossover,
                                  single_point_crossover)


class GeneticAlgorithmFactory:
    """
    A factory class for creating selection methods used in genetic algorithms.

    This class provides a method to retrieve different selection strategies
    based on a string identifier. It supports tournament, elitism, and ranking
    selection methods.
    """

    @classmethod
    def get_selection_method(
        cls,
        method: str,
    ) -> Callable[[List[Tuple[str, ...]], np.ndarray, int], Tuple[str, ...]]:
        """
        Retrieve a selection method based on the given method name.

        Args:
            method (str): The name of the selection method to retrieve.
                Supported methods:
                - 'tournament': Tournament selection
                - 'elitism': Elitism selection
                - 'ranking': Ranking selection

        Returns:
            Callable: A selection function that takes a population, fitness scores,
                      and a parameter (tournament size or num elites) and returns
                      a selected individual.

        Raises:
            ValueError: If an unknown selection method is specified.
        """
        if method == "tournament":
            return tournament_selection
        if method == "elitism":
            return elitism_selection
        if method == "ranking":
            return ranking_selection

        raise ValueError(f"Unknown selection method: {method}")

    @classmethod
    def get_crossover_method(
        cls, method: str
    ) -> Callable[
        [Tuple[str, ...], Tuple[str, ...]], Tuple[Tuple[str, ...], Tuple[str, ...]]
    ]:
        """
        Retrieve a crossover method based on the given method name.

        Args:
            method (str): The name of the crossover method to retrieve.
                Supported methods:
                - 'spc': single point crossover
                - 'cx': cycle crossover
                - 'pmx': partially mapped crossover

        Returns:
            Callable: A crossover function that takes two parents and returns two children.


        Raises:
            ValueError: If an unknown crossover method is specified.
        """
        if method == "spc":
            return single_point_crossover
        if method == "cx":
            return cycle_crossover
        if method == "pmx":
            return partially_mapped_crossover

        raise ValueError(f"Unknown selection method: {method}")

    @classmethod
    def get_chromosome_operation_method(
        cls, method: str
    ) -> Callable[[Tuple[str, ...], float], Tuple[str, ...]]:
        """
        Retrieve a chromosome operation method based on the given method name.

        Args:
            method (str): The name of the chromosome operation method to retrieve.
                Supported methods:
                - 'ad_swap': adjacent swap mutation
                - 'inversion': Inversion mutation

        Returns:
            Callable: A chromosome operation function that takes a route and mutation
                      rate and returns a mutated route.

        Raises:
            ValueError: If an unknown chromosome operation method is specified.
        """
        if method == "ad_swap":
            return adjacent_swap_mutation
        if method == "inversion":
            return inversion_mutation

        raise ValueError(f"Unknown chromosome operation method: {method}")
