# For your project to be successful you must implement at least 2 selection mechanisms

# ga_rd/selection.py
import random
from typing import List
from ga_rd.Individual_RD import IndividualRD

"""
Two selection mechanisms for the Running Dinner GA:
- Tournament selection (1. random selection of k individuals 2. return the one with lowest fitness)
- Rank-based selection (1. sort population by fitness 2. sample with weights proportional to rank)

Both mechanisms are implemented as functions that take a population of
IndividualRD objects and return a single selected individual.
"""

def tournament_selection(
    population: List[IndividualRD],
    k: int = 3
) -> IndividualRD:
    """
    Tournament selection (with replacement).

    Args:
        population (List[IndividualRD]): current population to select from.
        k (int, optional): number of random contestants in each tournament. Defaults to 3.

    Returns:
        IndividualRD: the tournament winner (individual with lowest fitness, since we minimize).
    """
    # pick k individuals at random (with replacement)
    contestants = random.choices(population, k=k)
    # return the one with the best (lowest) fitness
    return min(contestants, key=lambda ind: ind.fitness)


def rank_selection(
    population: List[IndividualRD]
) -> IndividualRD:
    """
    Rank-based selection.

    Sorts individuals by ascending fitness (best first), then
    assigns weights = [N, N-1, ..., 1], and samples one individual
    with probability proportional to its rank weight.

    Args:
        population (List[IndividualRD]): current population to select from.

    Returns:
        IndividualRD: a selected individual.
    """
    # sort by fitness (best/lowest first)
    ranked = sorted(population, key=lambda ind: ind.fitness)
    n = len(ranked)
    # best individual gets weight=n, worst gets weight=1
    weights = list(range(n, 0, -1))
    # random.choices returns a list of size k, so we take [0]
    return random.choices(ranked, weights=weights, k=1)[0]
