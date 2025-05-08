# ga_rd/algorithm.py

import random
from typing import List
from .Individual_RD import IndividualRD
from .selection import tournament_selection, rank_selection
from .crossover import one_point_crossover, uniform_crossover
from .mutation import swap_mutation, scramble_mutation, inversion_mutation
from copy import deepcopy
from ga_rd.Solution_RD import SolutionRD

def run_ga(
    pop_size: int,
    house_coords,
    n_participants: int,
    capacity: int,
    a: float,
    generations: int,
    selection_method: str   = 'tournament',
    tournament_size: int    = 3,
    afterparty_house: int   = None
) -> List[IndividualRD]:
    """
    Run a generational GA for the Running Dinner problem.

    Parameters:
    - pop_size [int]: Number of individuals in the population.
    - house_coords [np.ndarray]: Array of shape (n_houses, 2) with house latitude/longitude.
    - n_participants [int]: Total number of diners (must fit into the course capacities).
    - capacity [int]: Seating capacity of each house (including the host).
    - a [float]: Weight factor balancing distance vs. mixing in fitness.
    - generations [int]: How many generations to evolve.
    - selection_method: {'tournament','rank'} Parent selection strategy.
    - tournament_size [int]: k for tournament selection (only if method='tournament').
    - afterparty_house [int or None]: If provided, index of the house for the final afterparty leg.

    Returns List[IndividualRD] (The final evolved population)
    """
    # 1) INITIALIZE population
    pop: List[IndividualRD] = []
    for _ in range(pop_size):
        ind = IndividualRD(
            house_coords=house_coords,
            n_participants=n_participants,
            capacity_of_houses=capacity,
            a=a,
            afterparty_house=afterparty_house
        )
        ind.random_representation()
        pop.append(ind)

    # 2) EVOLUTIONARY LOOP
    for gen in range(1, generations+1):
        new_pop: List[IndividualRD] = []

        # fill next generation
        while len(new_pop) < pop_size:
            # 2a) PARENT SELECTION
            if selection_method == 'tournament':
                p1 = tournament_selection(pop, tournament_size)
                p2 = tournament_selection(pop, tournament_size)
            else:
                p1 = rank_selection(pop)
                p2 = rank_selection(pop)

            # 2b) CROSSOVER
            c1 = p1.crossover(p2)
            c2 = p2.crossover(p1)

            # 2c) MUTATION
            new_pop.extend([c1.mutation(), c2.mutation()])

        # 2d) REPLACE old population
        pop = new_pop[:pop_size]

    # 3) RETURN final population
    return pop
