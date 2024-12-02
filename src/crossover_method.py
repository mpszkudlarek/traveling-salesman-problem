"""
crossover_method.py

Contains implementations of crossover methods for genetic algorithms solving the
Traveling Salesman Problem (TSP). The methods include single-point crossover,
cycle crossover (CX), and partially mapped crossover (PMX).
"""

from typing import Tuple

import numpy as np


def single_point_crossover(
    parent1: Tuple[str, ...], parent2: Tuple[str, ...]
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
    point = np.random.randint(1, size)

    child1 = list(parent1[:point]) + [
        gene for gene in parent2 if gene not in parent1[:point]
    ]
    child2 = list(parent2[:point]) + [
        gene for gene in parent1 if gene not in parent2[:point]
    ]

    return tuple(child1), tuple(child2)


def cycle_crossover(
    parent1: Tuple[str, ...], parent2: Tuple[str, ...]
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Perform cycle crossover (CX) between two parent routes.

    Args:
        parent1 (Tuple[str, ...]): The first parent route.
        parent2 (Tuple[str, ...]): The second parent route.

    Returns:
        Tuple[Tuple[str, ...], Tuple[str, ...]]: Two child routes resulting from crossover.
    """
    size = len(parent1)
    child1 = list(parent1)
    child2 = list(parent2)

    filled1 = [False] * size
    filled2 = [False] * size

    def create_cycle(start_idx):
        indices_in_cycle = []
        current_idx = start_idx
        while current_idx not in indices_in_cycle:
            indices_in_cycle.append(current_idx)
            current_idx = parent1.index(parent2[current_idx])
        return indices_in_cycle

    for i in range(size):
        if not filled1[i]:
            cycle_indices = create_cycle(i)

            for idx in cycle_indices:
                child1[idx] = parent1[idx]
                child2[idx] = parent2[idx]
                filled1[idx] = True
                filled2[idx] = True

    for i in range(size):
        if not filled1[i]:
            child1[i] = parent2[i]
        if not filled2[i]:
            child2[i] = parent1[i]

    return tuple(child1), tuple(child2)


def partially_mapped_crossover(
    parent1: Tuple[str, ...], parent2: Tuple[str, ...]
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Perform partially mapped crossover (PMX) between two parent routes.

    Args:
        parent1 (Tuple[str, ...]): The first parent route.
        parent2 (Tuple[str, ...]): The second parent route.

    Returns:
        Tuple[Tuple[str, ...], Tuple[str, ...]]: Two child routes resulting from crossover.
    """
    size = len(parent1)
    child1 = list(parent1)
    child2 = list(parent2)

    point1, point2 = sorted(np.random.choice(range(size), 2, replace=False))

    mapping1 = {parent1[i]: parent2[i] for i in range(point1, point2)}
    mapping2 = {parent2[i]: parent1[i] for i in range(point1, point2)}

    for i in range(point1, point2):
        child1[i], child2[i] = parent2[i], parent1[i]

    def resolve_conflicts(child, mapping):
        for i in range(size):
            if point1 <= i < point2:
                continue
            while child[i] in mapping:
                child[i] = mapping[child[i]]

    resolve_conflicts(child1, mapping1)
    resolve_conflicts(child2, mapping2)

    return tuple(child1), tuple(child2)
