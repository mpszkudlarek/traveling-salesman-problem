"""
This module provides functionality for running and analyzing TSP (Traveling Salesman Problem)
experiments using genetic algorithms with various configuration parameters.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.config_parameters import CrossoverConfig, GeneticConfig, MutationConfig, SelectionConfig
from src.enums.tsp_genetic_enums import CrossoverMethod, MutationMethod, SelectionMethod
from src.tsp_solver import TSPSolver


@dataclass
class TSPResult:
    """
    Contains the results of a single TSP experiment run.

    Attributes:
        best_distances: List of best distances for each generation
        best_route: Best route found during the experiment
        final_distance: Final best distance achieved
        crossover_method: Crossover method used
        selection_method: Selection method used
        mutation_method: Mutation method used
    """

    best_distances: List[float]
    best_route: Tuple[str, ...]
    final_distance: float
    crossover_method: CrossoverMethod
    selection_method: SelectionMethod
    mutation_method: MutationMethod


@dataclass
class ExperimentParams:
    """Configuration parameters for TSP experiment."""

    mutation_rate: float
    crossover_rate: float
    population_size: int
    generations: int
    selection_method: SelectionMethod = SelectionMethod
    crossover_method: CrossoverMethod = CrossoverMethod
    mutation_method: MutationMethod = MutationMethod
    tournament_percent: Optional[float] = None
    selection_pressure: Optional[float] = None
    use_gpu: bool = True


def run_experiment(params: ExperimentParams) -> TSPResult:
    """
    Run a single TSP experiment with specified parameters.

    Args:
        params: ExperimentParams containing all experiment configuration

    Returns:
        TSPResult containing the experiment results
    """
    selection_params = {}
    if params.selection_method == SelectionMethod.TOURNAMENT:
        selection_params["tournament_percent"] = params.tournament_percent or 0.2
    elif params.selection_method == SelectionMethod.RANKING:
        selection_params["selection_pressure"] = params.selection_pressure or 1.5

    selection_config = SelectionConfig(selection_method=params.selection_method, **selection_params)

    crossover_config = CrossoverConfig(
        crossover_method=params.crossover_method, crossover_rate=params.crossover_rate
    )

    mutation_config = MutationConfig(mutation_rate=params.mutation_rate, mutation_method=params.mutation_method)

    config = GeneticConfig(
        generations=params.generations,
        population_size=params.population_size,
        selection_config=selection_config,
        crossover_config=crossover_config,
        mutation_config=mutation_config,
    )

    solver = TSPSolver(distance_file="10.in", config={"folder": "../input"}, use_gpu=params.use_gpu)

    best_distances, best_route = solver.solve(config)

    return TSPResult(
        best_distances=best_distances,
        best_route=tuple(best_route),
        final_distance=solver.best_distance,
        crossover_method=crossover_config.crossover_method,
        selection_method=selection_config.selection_method,
        mutation_method=mutation_config.mutation_method,
    )
