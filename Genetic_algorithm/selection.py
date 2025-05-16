import random
from typing import List
from Genetic_algorithm.base_individual import Individual

def tournament_selection(
    population: List[Individual],
    maximization: bool = False,
    k: int = 3
) -> Individual:
    """
    k-way tournament (with replacement), unified for minimization or maximization.

    Parameters
    ----------
    population : List[Individual]
        Current population.
    maximization : bool, default=False
        If False, lower fitness is better (minimize). If True, higher fitness is better.
    k : int, default=3
        Tournament size (number of contestants).

    Returns
    -------
    Individual
        The tournament winner.
    """
    # pick k random contestants (with replacement)
    contestants = random.choices(population, k=k)
    # choose best according to the mode
    if maximization:
        # higher fitness wins
        return max(contestants, key=lambda ind: ind.fitness)
    else:
        # lower fitness wins
        return min(contestants, key=lambda ind: ind.fitness)


def rank_selection(
    population: List[Individual],
    maximization: bool = False
) -> Individual:
    """
    Rank-based roulette selection.

    1. Sort population by fitness.
    2. Assign weights N, N-1, …, 1 (best gets highest weight).
    3. Sample one individual proportionally to its weight.

    Parameters
    ----------
    population : List[Individual]
        Current population.
    maximization : bool, default False
        If False, lower fitness is better; if True, higher fitness is better.

    Returns
    -------
    Individual
        The selected individual.
    """
    # best-first sort: reverse=True if maximize, else reverse=False
    ranked = sorted(
        population,
        key=lambda ind: ind.fitness,
        reverse=maximization
    )
    n = len(ranked)
    weights = list(range(n, 0, -1))  # best gets weight n, worst gets 1
    return random.choices(ranked, weights=weights, k=1)[0]