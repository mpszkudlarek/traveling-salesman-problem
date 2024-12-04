"""
Selection methods for genetic algorithms.

This module provides various selection strategies for genetic algorithms,
including tournament selection, elitism selection, and ranking selection.
These methods are used to choose individuals from a population based on
their fitness scores, supporting different selection pressures and strategies.

Supported selection methods:
- Tournament Selection: Randomly selects the best individual from a small tournament
- Elitist  Selection: Selects from the top-performing individuals
- Rank Selection: Selects individuals based on their rank, with configurable selection pressure
Each of those algorithms are based from: https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)
"""

from typing import List, Optional, Tuple

import numpy as np


def tournament_selection(
    population: List[Tuple[str, ...]],
    fitness_scores: np.ndarray,
    tournament_percent: float,
    random_seed: Optional[int] = None,
) -> Tuple[str, ...]:
    """
    Select an individual using tournament selection.

    Args:
        population (List[Tuple[str, ...]]): The current population of routes.
        fitness_scores (np.ndarray): Fitness scores for each individual.
        tournament_percent (flora): Percent of population to use in tournament (0.0 - 1.0).
        random_seed (int, optional): A seed for the random number
                                        generator to ensure reproducibility.


    Returns:
        Tuple[str, ...]: The selected route.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if not 0 < tournament_percent <= 0:
        raise ValueError("Tournament size must be between 0 and 1.")

    tournament_size = max(2, int(len(population) * tournament_percent))

    tournament_indices = np.random.choice(
        len(population), tournament_size, replace=False
    )

    tournament_scores = fitness_scores[tournament_indices]
    selected_idx = tournament_indices[np.argmax(tournament_scores)]
    return population[selected_idx]


def elitist_selection(
    population: List[Tuple[str, ...]],
    fitness_scores: np.ndarray,
    num_elites: int,
    random_seed: Optional[int] = None,
) -> Tuple[str, ...]:
    """
    Select a single elite individual, with preference for the top individuals.

    Args:
        population (List[Tuple[str, ...]]): The current population of routes.
        fitness_scores (np.ndarray): Fitness scores for each individual.
        num_elites (int): Number of top individuals to consider for selection.
        random_seed (int, optional): A seed for the random number
                                        generator to ensure reproducibility..


    Returns:
        Tuple[str, ...]: The selected elite route.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if num_elites < 1 or num_elites > len(population):
        raise ValueError(f"Number of elites must be between 1 and {len(population)}.")

    # Sort indices based on fitness scores in descending order
    elite_indices = np.argsort(fitness_scores)[::-1][:num_elites]

    # Randomly select from the top num_elites individuals
    selected_idx = np.random.choice(elite_indices)

    return population[selected_idx]


def rank_selection(
    population: List[Tuple[str, ...]],
    fitness_scores: np.ndarray,
    selection_pressure: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Tuple[str, ...]:
    """
    Select an individual using ranking selection.

    Args:
        population (List[Tuple[str, ...]]): The current population of routes.
        fitness_scores (np.ndarray): Fitness scores for each individual.
        selection_pressure (float, optional): Controls selection bias.
            Higher values create stronger selection pressure.
            If None, a dynamic default can be set.
        random_seed (int, optional): A seed for the random number
                                        generator to ensure reproducibility..

    Returns:
        Tuple[str, ...]: The selected route.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if selection_pressure is None:
        selection_pressure = 1.5

    if not 1.0 <= selection_pressure <= 2.0:
        raise ValueError("Selection pressure must be between 1.0 and 2.0.")

    population_size = len(population)

    if population_size < 2:
        raise ValueError("Population size must be at least 2 for ranking selection.")

    ranks = np.argsort(np.argsort(fitness_scores)[::-1]) + 1
    selection_probs = (selection_pressure / population_size) - (
        (2 * selection_pressure - 2)
        * (ranks - 1)
        / (population_size * (population_size - 1))
    )

    selection_probs /= np.sum(selection_probs)

    selected_idx = np.random.choice(population_size, p=selection_probs)

    return population[selected_idx]
