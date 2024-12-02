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
    point = np.random.randint(
        1, size
    )  # Random crossover point, 1 because we don't want to start at the first city

    # Create children by combining parts of parents
    child1 = list(
        parent1[:point]
    ) + [  # Take the first part of parent1 up to the crossover point
        gene
        for gene in parent2
        if gene
        not in parent1[:point]  # Add genes from parent2 that are not already in child1
    ]
    child2 = list(
        parent2[:point]
    ) + [  # Take the first part of parent2 up to the crossover point
        gene
        for gene in parent1
        if gene
        not in parent2[:point]  # Add genes from parent1 that are not already in child2
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

    # Helper function to create cycles
    def create_cycle(start_idx):
        cycle_indices = []
        current_idx = start_idx
        while current_idx not in cycle_indices:
            cycle_indices.append(current_idx)
            current_idx = parent1.index(parent2[current_idx])
        return cycle_indices

    # Start creating cycles and filling children
    for i in range(size):
        if not filled1[i]:  # Start a new cycle if the position isn't filled
            cycle_indices = create_cycle(i)

            # Alternate cycles between child1 and child2
            for idx in cycle_indices:
                child1[idx] = parent1[idx]
                child2[idx] = parent2[idx]
                filled1[idx] = True
                filled2[idx] = True

    # Fill remaining positions from the opposite parent
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

    # Select two random crossover points
    point1, point2 = sorted(np.random.choice(range(size), 2, replace=False))

    # Create mappings for the crossover section
    mapping1 = {parent1[i]: parent2[i] for i in range(point1, point2)}
    mapping2 = {parent2[i]: parent1[i] for i in range(point1, point2)}

    # Swap the crossover section
    for i in range(point1, point2):
        child1[i], child2[i] = parent2[i], parent1[i]

    # Resolve conflicts using the mappings
    def resolve_conflicts(child, mapping):
        for i in range(size):
            if point1 <= i < point2:
                continue
            while child[i] in mapping:
                child[i] = mapping[child[i]]

    resolve_conflicts(child1, mapping1)
    resolve_conflicts(child2, mapping2)

    return tuple(child1), tuple(child2)
