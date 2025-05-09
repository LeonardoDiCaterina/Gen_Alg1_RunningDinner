# ga_rd/algorithm.py
import random
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Dict, Any
from .Individual_RD import IndividualRD
from .selection import tournament_selection, rank_selection

def run_ga(
    pop_size: int,
    participant_homes: np.ndarray,
    house_coords: np.ndarray,
    host_idxs: list[int],
    capacity: int,
    a: float,
    generations: int,
    selection_method: str = 'tournament',
    tournament_size: int = 3,
    crossover_method: str = 'uniform',
    crossover_prob: float = 0.9,
    mutation_method: str = 'swap',
    mutation_prob: float = 0.1,
    elitism: int = 1,
    early_stopping: int = None,
    afterparty_house: int = None,
    verbose: bool = False
) -> Tuple[List[IndividualRD], Dict[str, Any]]:
    """
    Run a generational GA for the Running Dinner problem.
    
    Args:
      - pop_size:           how many Individuals in each generation
      - participant_homes:  shape = (n_participants, 2)
      - house_coords:       shape = (n_houses, 2)
      - host_idxs:          list of house indices for each course
      - capacity:           seats per house (including host)
      - a:                  distance-vs-mixing weight in f = (m/n)*a*d
      - generations:        number of generations to evolve
      - selection_method:   'tournament' or 'rank'
      - tournament_size:    k for tournament selection
      - crossover_method:   'one_point', 'uniform', 'pmx', or 'random'
      - crossover_prob:     probability of applying crossover
      - mutation_method:    'swap', 'scramble', or 'inversion'
      - mutation_prob:      probability of applying mutation
      - elitism:            number of best individuals to preserve
      - early_stopping:     stop if no improvement for this many generations
      - afterparty_house:   optional index in [0..n_houses) for final leg
      - verbose:            whether to print progress
    
    Returns:
      Tuple of:
      - final population of IndividualRD
      - stats dictionary with history and metrics
    """
    
    # infer number of diners
    n_participants = participant_homes.shape[0]
    
    # Statistics tracking
    stats = {
        'best_fitness': [],
        'avg_fitness': [],
        'worst_fitness': [],
        'best_solution': None,
        'generations_run': 0,
        'early_stopped': False,
        'convergence_gen': None
    }
    
    # 1) initialize population
    pop: List[IndividualRD] = []
    for _ in range(pop_size):
        ind = IndividualRD(
            participant_homes=participant_homes,
            house_coords=house_coords,
            host_idxs=host_idxs,
            capacity_of_houses=capacity,
            a=a,
            afterparty_house=afterparty_house,
            crossover_method=crossover_method,
            crossover_prob=crossover_prob,
            mutation_method=mutation_method,
            mutation_prob=mutation_prob
        )
        ind.random_representation()
        pop.append(ind)
    
    # Track best solution
    best_fitness = float('inf')
    best_generation = 0
    
    # 2) main GA loop
    for gen in range(1, generations + 1):
        stats['generations_run'] = gen
        
        # Calculate fitness for all individuals
        fitnesses = [ind.fitness for ind in pop]
        
        # Update stats
        gen_best = min(fitnesses)
        gen_avg = sum(fitnesses) / len(fitnesses)
        gen_worst = max(fitnesses)
        
        stats['best_fitness'].append(gen_best)
        stats['avg_fitness'].append(gen_avg)
        stats['worst_fitness'].append(gen_worst)
        
        # Update best solution if improved
        if gen_best < best_fitness:
            best_fitness = gen_best
            best_generation = gen
            stats['best_solution'] = pop[fitnesses.index(gen_best)]
        
        # Print progress if verbose
        if verbose and gen % 10 == 0:
            print(f"Generation {gen}: Best={gen_best:.2f}, Avg={gen_avg:.2f}, Worst={gen_worst:.2f}")
        
        # Check early stopping
        if early_stopping and (gen - best_generation) >= early_stopping:
            if verbose:
                print(f"Early stopping at generation {gen}: No improvement for {early_stopping} generations")
            stats['early_stopped'] = True
            stats['convergence_gen'] = best_generation
            break
        
        # Elitism: keep best individuals
        if elitism > 0:
            elite = sorted(pop, key=lambda ind: ind.fitness)[:elitism]
            elite = [deepcopy(ind) for ind in elite]  # Make sure they're independent copies
        else:
            elite = []
        
        # Create new population
        new_pop: List[IndividualRD] = elite.copy()
        
        # Fill the rest of the population with offspring
        while len(new_pop) < pop_size:
            # parent selection
            if selection_method == 'tournament':
                p1 = tournament_selection(pop, tournament_size)
                p2 = tournament_selection(pop, tournament_size)
            else:
                p1 = rank_selection(pop)
                p2 = rank_selection(pop)
            
            # crossover → two children
            c1, c2 = p1.crossover(p2)
            
            # mutation
            c1 = c1.mutation()
            c2 = c2.mutation()
            
            # add to new population
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        
        # Update population
        pop = new_pop
    
    return pop, stats

