"""
Selection methods for genetic algorithms.

This module provides various selection strategies for genetic algorithms,
including tournament selection, elitism selection, and ranking selection.
These methods are used to choose individuals from a population based on
their fitness scores, supporting different selection pressures and strategies.

Supported selection methods:
- Tournament Selection: Randomly selects the best individual from a small tournament
- Elitism Selection: Selects from the top-performing individuals
- Ranking Selection: Selects individuals based on their rank, with configurable selection pressure
"""

from typing import List, Tuple

import numpy as np


def tournament_selection(
    population: List[Tuple[str, ...]], fitness_scores: np.ndarray, tournament_size: int
) -> Tuple[str, ...]:
    """
    Select an individual using tournament selection.

    Args:
        population (List[Tuple[str, ...]]): The current population of routes.
        fitness_scores (np.ndarray): Fitness scores for each individual.
        tournament_size (int): Number of individuals competing in the tournament.

    Returns:
        Tuple[str, ...]: The selected route.
    """
    tournament_indices = np.random.choice(
        len(population), tournament_size, replace=False
    )
    tournament_scores = fitness_scores[tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_scores)]
    return population[winner_idx]


def elitism_selection(
    population: List[Tuple[str, ...]], fitness_scores: np.ndarray, num_elites: int
) -> Tuple[str, ...]:
    """
    Select a single elite individual, with preference for the top individuals.

    Args:
        population (List[Tuple[str, ...]]): The current population of routes.
        fitness_scores (np.ndarray): Fitness scores for each individual.
        num_elites (int): Number of top individuals to consider for selection.

    Returns:
        Tuple[str, ...]: The selected elite route.
    """
    # Sort indices based on fitness scores in descending order
    elite_indices = np.argsort(fitness_scores)[::-1][:num_elites]

    # Randomly select from the top num_elites individuals
    selected_idx = np.random.choice(elite_indices)

    return population[selected_idx]


def ranking_selection(
    population: List[Tuple[str, ...]],
    fitness_scores: np.ndarray,
    selection_pressure: float = 1.5,
) -> Tuple[str, ...]:
    """
    Select an individual using ranking selection.

    Args:
        population (List[Tuple[str, ...]]): The current population of routes.
        fitness_scores (np.ndarray): Fitness scores for each individual.
        selection_pressure (float, optional): Controls selection bias.
            Higher values create stronger selection pressure.
            Defaults to 1.5.

    Returns:
        Tuple[str, ...]: The selected route.
    """
    # Get the number of individuals
    population_size = len(population)

    # Create ranking probabilities
    # Uses linear ranking selection formula
    ranks = np.argsort(np.argsort(fitness_scores)[::-1]) + 1
    selection_probs = (2 - selection_pressure) / population_size + (
        2 * ranks * (selection_pressure - 1)
    ) / (population_size * (population_size - 1))

    # Normalize probabilities
    selection_probs /= np.sum(selection_probs)

    # Select an individual based on ranking probabilities
    selected_idx = np.random.choice(population_size, p=selection_probs)

    return population[selected_idx]
