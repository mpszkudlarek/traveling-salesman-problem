from typing import List, Tuple, Union

import numpy as np


def adjacent_swap_mutation(
    route: Union[List[str], Tuple[str, ...]], mutation_rate: float
) -> Tuple[str, ...]:
    """
    Apply adjacent swap mutation to a route.
    """
    route_list = list(route)  # Convert to list for manipulation

    if np.random.random() < mutation_rate:
        size = len(route_list)
        idx = np.random.randint(0, size - 1)  # Choose a pair of adjacent cities
        route_list[idx], route_list[idx + 1] = route_list[idx + 1], route_list[idx]

    return tuple(route_list)


def inversion_mutation(
    route: Union[List[str], Tuple[str, ...]], mutation_rate: float
) -> Tuple[str, ...]:
    """
    Apply inversion mutation to a route by reversing a random sub-path.
    """
    route_list = list(route)  # Convert to list for manipulation

    if np.random.random() < mutation_rate:
        size = len(route_list)
        start, end = sorted(np.random.choice(range(size), 2, replace=False))
        route_list[start:end] = reversed(route_list[start:end])

    return tuple(route_list)


def insertion_mutation(
    route: Union[List[str], Tuple[str, ...]], mutation_rate: float
) -> Tuple[str, ...]:
    """
    Apply insertion mutation to a route by moving one city to a new position.
    """
    route_list = list(route)  # Convert to list for manipulation

    if np.random.random() < mutation_rate:
        size = len(route_list)
        idx = np.random.randint(0, size)
        city = route_list.pop(idx)
        new_position = np.random.randint(0, size - 1)
        route_list.insert(new_position, city)

    return tuple(route_list)
