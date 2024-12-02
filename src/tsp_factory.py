from typing import Callable, List, Tuple, Union

import numpy as np

from src.crossover_method import (cycle_crossover, partially_mapped_crossover,
                                  single_point_crossover)
from src.mutation_method import adjacent_swap_mutation, inversion_mutation
from src.selection_method import (elitism_selection, ranking_selection,
                                  tournament_selection)
from src.selection_method_enums import (CrossoverMethod, MutationMethod,
                                        SelectionMethod)


class TspFactory:
    """
    A factory class for creating methods used in genetic algorithms.

    Provides methods for selection, crossover, and mutation strategies.
    """

    @classmethod
    def get_selection_method(
        cls,
        method: SelectionMethod,
    ) -> Callable[[List[Tuple[str, ...]], np.ndarray, int], Tuple[str, ...]]:
        """
        Retrieve a selection method based on the given method enumeration.
        """
        method_map: dict[
            SelectionMethod,
            Callable[[List[Tuple[str, ...]], np.ndarray, int], Tuple[str, ...]],
        ] = {
            SelectionMethod.TOURNAMENT: tournament_selection,
            SelectionMethod.ELITISM: elitism_selection,
            SelectionMethod.RANKING: ranking_selection,
        }

        if method not in method_map:
            raise ValueError(f"Unsupported selection method: {method}")

        return method_map[method]

    @classmethod
    def get_crossover_method(
        cls, method: CrossoverMethod
    ) -> Callable[
        [Tuple[str, ...], Tuple[str, ...]], Tuple[Tuple[str, ...], Tuple[str, ...]]
    ]:
        """
        Retrieve a crossover method based on the given method enumeration.
        """
        method_map: dict[
            CrossoverMethod,
            Callable[
                [Tuple[str, ...], Tuple[str, ...]],
                Tuple[Tuple[str, ...], Tuple[str, ...]],
            ],
        ] = {
            CrossoverMethod.SINGLE_POINT: single_point_crossover,
            CrossoverMethod.CYCLE: cycle_crossover,
            CrossoverMethod.PARTIALLY_MAPPED: partially_mapped_crossover,
        }

        if method not in method_map:
            raise ValueError(f"Unsupported crossover method: {method}")

        return method_map[method]

    @classmethod
    def get_mutation_method(
        cls, method: MutationMethod
    ) -> Callable[[Tuple[str, ...], float], Tuple[str, ...]]:
        """
        Retrieve a chromosome operation method based on the given method enumeration.
        """
        method_map: dict[
            MutationMethod,
            Callable[[Tuple[str, ...], float], Tuple[str, ...]],
        ] = {
            MutationMethod.ADJACENT_SWAP: adjacent_swap_mutation,
            MutationMethod.INVERSION: inversion_mutation,
        }

        if method not in method_map:
            raise ValueError(f"Unsupported chromosome operation method: {method}")

        return method_map[method]

    @classmethod
    def validate_method(
        cls,
        method_type: str,
        method: Union[SelectionMethod, CrossoverMethod, MutationMethod],
    ) -> bool:
        """
        Validate if a given method exists for a specific method type.

        Args:
            method_type (str): The type of the method ('selection', 'crossover', 'mutation').
            method (Union[SelectionMethod, CrossoverMethod, MutationMethod]): The method
                   to validate.

        Returns:
            bool: True if the method is valid, False otherwise.
        """
        try:
            if method_type == "selection":
                if not isinstance(method, SelectionMethod):
                    return False
                cls.get_selection_method(method)
            elif method_type == "crossover":
                if not isinstance(method, CrossoverMethod):
                    return False
                cls.get_crossover_method(method)
            elif method_type == "mutation":
                if not isinstance(method, MutationMethod):
                    return False
                cls.get_mutation_method(method)
            else:
                return False
            return True
        except ValueError:
            return False
