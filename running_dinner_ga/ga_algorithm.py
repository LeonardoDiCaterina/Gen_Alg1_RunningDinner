# running_dinner_ga/ga_algorithm.py

import time
import random
from typing import List
from .Individual_RD import IndividualRD

def tournament_selection(
    population: List[IndividualRD],
    k: int = 3
) -> IndividualRD:
    contestants = random.choices(population, k=k)
    return min(contestants, key=lambda ind: ind.fitness)

def rank_selection(
    population: List[IndividualRD]
) -> IndividualRD:
    ranked = sorted(population, key=lambda ind: ind.fitness)
    weights = list(range(len(ranked), 0, -1))
    return random.choices(ranked, weights=weights, k=1)[0]

def run_ga(
    pop_size: int,
    house_coords,
    n_participants: int,
    capacity: int,
    a: float,
    generations: int,
    selection_method: str = 'tournament',
    tournament_size: int = 3
) -> List[IndividualRD]:

    # 1) Initialize population
    pop = [
        IndividualRD(house_coords, n_participants, capacity, a)
        for _ in range(pop_size)
    ]
    for ind in pop:
        ind.random_representation()

    # 2) Evolutionary loop
    for gen in range(generations):
        print(f"\n>>> Generation {gen+1}/{generations} start")
        start_time = time.perf_counter()

        new_pop = []
        loop = 0

        while len(new_pop) < pop_size:
            loop += 1

            # Parent selection
            if selection_method == 'tournament':
                p1 = tournament_selection(pop, tournament_size)
                p2 = tournament_selection(pop, tournament_size)
            else:
                p1 = rank_selection(pop)
                p2 = rank_selection(pop)

            # Crossover
            c1 = p1.crossover(p2)
            c2 = p2.crossover(p1)

            # Mutation
            child1 = c1.mutation()
            child2 = c2.mutation()

            new_pop.extend([child1, child2])

            # **Debug print every loop**:
            print(f"   loop {loop}: new_pop size = {len(new_pop)}", end="\r")

        # Trim in case we overfilled
        pop = new_pop[:pop_size]

        elapsed = time.perf_counter() - start_time
        print(f"\n<<< Generation {gen+1} done in {elapsed:.3f}s (loops: {loop})")

    print("\nGA complete.")
    return pop
