"""
mutation_method.py

Contains implementations of mutation methods for genetic algorithms solving the
Traveling Salesman Problem (TSP). The methods include adjacent swap, inversion,
and insertion mutations, applied to routes (chromosomes).
"""

from typing import List, Tuple, Union

import numpy as np


def adjacent_swap_mutation(
    route: Union[List[str], Tuple[str, ...]], mutation_rate: float
) -> Tuple[str, ...]:
    """
    Apply adjacent swap mutation to a route.

    Args:
        route (Union[List[str], Tuple[str, ...]]): The current route.
        mutation_rate (float): Probability of applying mutation.

    Returns:
        Tuple[str, ...]: Mutated route.
    """
    route_list = list(route)

    if np.random.random() < mutation_rate:
        size = len(route_list)
        idx = int(np.random.randint(0, size - 1))
        route_list[idx], route_list[idx + 1] = route_list[idx + 1], route_list[idx]

    return tuple(route_list)


def inversion_mutation(
    route: Union[List[str], Tuple[str, ...]], mutation_rate: float
) -> Tuple[str, ...]:
    """
    Apply inversion mutation to a route by reversing a random sub-path.

    Args:
        route (Union[List[str], Tuple[str, ...]]): The current route.
        mutation_rate (float): Probability of applying mutation.

    Returns:
        Tuple[str, ...]: Mutated route.
    """
    route_list = list(route)
    if np.random.random() < mutation_rate:
        size = len(route_list)
        indices = np.random.choice(range(size), 2, replace=False)
        start, end = map(int, sorted(indices))
        route_list[start:end] = list(
            reversed(route_list[start:end])
        )

    return tuple(route_list)


def insertion_mutation(
    route: Union[List[str], Tuple[str, ...]], mutation_rate: float
) -> Tuple[str, ...]:
    """
    Apply insertion mutation to a route by moving one city to a new position.

    Args:
        route (Union[List[str], Tuple[str, ...]]): The current route.
        mutation_rate (float): Probability of applying mutation.

    Returns:
        Tuple[str, ...]: Mutated route.
    """
    route_list = list(route)

    if np.random.random() < mutation_rate:
        size = len(route_list)
        idx = int(np.random.randint(0, size))
        city = route_list.pop(idx)
        new_position = int(
            np.random.randint(0, size - 1)
        )
        route_list.insert(new_position, city)

    return tuple(route_list)
