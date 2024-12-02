from enum import Enum


class SelectionMethod(Enum):
    TOURNAMENT = "tournament"
    ELITISM = "elitism"
    RANKING = "ranking"


class CrossoverMethod(Enum):
    SINGLE_POINT = "spc"
    CYCLE = "cx"
    PARTIALLY_MAPPED = "pmx"


class MutationMethod(Enum):
    ADJACENT_SWAP = "ad_swap"
    INVERSION = "inversion"
    INSERTION = "insertion"
