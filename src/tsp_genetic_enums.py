"""
tsp_genetic_enums.py

Defines enumerations for selection, crossover, and mutation methods
used in genetic algorithms for solving the Traveling Salesman Problem (TSP).
"""

from enum import Enum


class SelectionMethod(Enum):
    """
    Enumeration of selection methods for genetic algorithms.

    Methods:
        TOURNAMENT: Selects candidates using a tournament-based approach.
        ELITISM: Retains the best individuals across generations.
        RANKING: Selects individuals based on their ranking in the population.
    """

    TOURNAMENT = "tournament"
    ELITISM = "elitism"
    RANKING = "ranking"

    def __str__(self):
        return self.value


class CrossoverMethod(Enum):
    """
    Enumeration of crossover methods for genetic algorithms.

    Methods:
        SINGLE_POINT: Performs crossover at a single point in the chromosome.
        CYCLE: Applies cycle crossover to exchange segments of chromosomes.
        PARTIALLY_MAPPED: Uses partially mapped crossover for gene exchange.
    """

    SINGLE_POINT = "spc"
    CYCLE = "cx"
    PARTIALLY_MAPPED = "pmx"

    def __str__(self):
        return self.value


class MutationMethod(Enum):
    """
    Enumeration of mutation methods for genetic algorithms.

    Methods:
        ADJACENT_SWAP: Swaps two adjacent genes in the chromosome.
        INVERSION: Reverses the order of genes in a segment of the chromosome.
        INSERTION: Inserts a gene from one position into another.
    """

    ADJACENT_SWAP = "adjacent_swap"
    INVERSION = "inversion"
    INSERTION = "insertion"

    def __str__(self):
        return self.value
