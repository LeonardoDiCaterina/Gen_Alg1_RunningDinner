import random
from typing import List
from Genetic_algorithm.solution_rd import SolutionRD


def tournament_selection(
    population: List[SolutionRD],
    maximization: bool = True,
    k: int = 3
) -> SolutionRD:
    """
    k-way tournament (with replacement), unified for minimization or maximization.

    Parameters
    ----------
    population : List[SolutionRD]
        Current population.
    maximization : bool, default=True
        If False, lower fitness is better (minimize). If True, higher fitness is better.
    k : int, default=3
        Tournament size (number of contestants).

    Returns
    -------
    SolutionRD
        The tournament winner.
    """
    # Randomly select k individuals from the population
    contestants = random.choices(population, k=k)
    
    # Determine the winner based on fitness
    if maximization:
        winner = max(contestants, key=lambda ind: float(ind))
    else:
        winner = min(contestants, key=lambda ind: float(ind))
    
    return winner


def rank_selection(
    population: List[SolutionRD],
    maximization: bool = True
) -> SolutionRD:
    """
    Rank-based roulette selection.

    1. Sort population by fitness.
    2. Assign weights N, N-1, …, 1 (best gets highest weight).
    3. Sample one individual proportionally to its weight.

    Parameters
    ----------
    population : List[Individual]
        Current population.
    maximization : bool, default True
        If False, lower fitness is better; if True, higher fitness is better.

    Returns
    -------
    Individual
        The selected individual.
    """
    # Sort population by fitness
    sorted_population = sorted(population, reverse=maximization)
    
    # Assign weights
    weights = list(range(len(sorted_population), 0, -1))
    
    # Calculate total weight
    total_weight = sum(weights)
    
    # Calculate selection probabilities
    probabilities = [weight / total_weight for weight in weights]
    
    # Select an individual based on the calculated probabilities
    return random.choices(sorted_population, weights=probabilities, k=1)[0]

    