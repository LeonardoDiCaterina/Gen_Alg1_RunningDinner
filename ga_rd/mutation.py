# For your project to be successful you must implement at least 3 mutation operators

# ga_rd/mutation.py
from __future__ import annotations
import random
import numpy as np
from copy import deepcopy
from typing import TYPE_CHECKING

# Only import IndividualRD for type checking—won't happen at runtime
if TYPE_CHECKING:
    from .Individual_RD import IndividualRD

"""
Three mutation operators for the Running Dinner GA:
- swap_mutation: swap two positions within the same genome segment (houses or participants)
- scramble_mutation: shuffle a contiguous slice within a chosen segment
- inversion_mutation: reverse a contiguous slice within a chosen segment

Each function takes an IndividualRD and returns a new mutated IndividualRD.
"""

# Maximum number of retries for mutation attempts
MAX_MUTATION_RETRIES = 5

def swap_mutation(ind: IndividualRD, max_retries: int = MAX_MUTATION_RETRIES, verbose: bool = False) -> IndividualRD:
    """
    Swap two genes within one segment. Retry until genome is still valid.
    """
    for attempt in range(1, max_retries+1):
        mutant = deepcopy(ind)
        n_h   = mutant.solution.n_houses
        L     = len(mutant.solution.genome)  # Use len() instead of .size

        # choose segment
        if random.random() < 0.5:
            i, j = random.sample(range(0, n_h), 2)
        else:
            i, j = random.sample(range(n_h, L), 2)

        # perform swap
        mutant.solution.genome_swap(i, j)
        mutant._fitness = None

        # validate
        if mutant.solution.check_validity_of_genome():
            if verbose: 
                print(f"[swap_mutation] Success on attempt {attempt}: swapped {i}<->{j}")
            return mutant

    if verbose:
        print("[swap_mutation] All retries failed, returning original")
    return ind  # fallback: no change

def scramble_mutation(ind: IndividualRD, max_retries: int = MAX_MUTATION_RETRIES, verbose: bool = False) -> IndividualRD:
    """
    Scramble a slice within one segment. Retry until genome is valid.
    """
    for attempt in range(1, max_retries+1):
        mutant = deepcopy(ind)
        genome = mutant.solution.genome.copy()  # Direct access to genome
        n_h    = mutant.solution.n_houses
        L      = len(genome)  # Use len() instead of .size

        # choose segment
        if random.random() < 0.5:
            seg0, seg1 = 0, n_h
        else:
            seg0, seg1 = n_h, L

        # Ensure we have at least 2 elements to scramble
        if seg1 - seg0 < 2:
            continue

        i, j = sorted(random.sample(range(seg0, seg1), 2))
        if j - i < 2:  # Need at least 2 elements to scramble
            continue
            
        genome[i:j] = np.random.permutation(genome[i:j])
        mutant.solution.set_genome(genome)
        mutant._fitness = None

        if mutant.solution.check_validity_of_genome():
            if verbose:
                print(f"[scramble_mutation] Success on attempt {attempt}: scrambled {i}:{j}")
            return mutant

    if verbose:
        print("[scramble_mutation] All retries failed, returning original")
    return ind

def inversion_mutation(ind: IndividualRD, max_retries: int = MAX_MUTATION_RETRIES, verbose: bool = False) -> IndividualRD:
    """
    Invert a slice within one segment. Retry until genome is valid.
    """
    for attempt in range(1, max_retries+1):
        mutant = deepcopy(ind)
        genome = mutant.solution.genome.copy()  # Direct access to genome
        n_h    = mutant.solution.n_houses
        L      = len(genome)  # Use len() instead of .size

        if random.random() < 0.5:
            seg0, seg1 = 0, n_h
        else:
            seg0, seg1 = n_h, L

        # Ensure we have at least 2 elements to invert
        if seg1 - seg0 < 2:
            continue

        i, j = sorted(random.sample(range(seg0, seg1), 2))
        if j - i < 2:  # Need at least 2 elements to invert
            continue
            
        genome[i:j] = genome[i:j][::-1]  # Invert the slice
        mutant.solution.set_genome(genome)
        mutant._fitness = None

        if mutant.solution.check_validity_of_genome():
            if verbose:
                print(f"[inversion_mutation] Success on attempt {attempt}: inverted {i}:{j}")
            return mutant

    if verbose:
        print("[inversion_mutation] All retries failed, returning original")
    return ind