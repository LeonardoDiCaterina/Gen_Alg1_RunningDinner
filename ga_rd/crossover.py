# For your project to be successful you must implement at least 2 crossover operators

# ga_rd/crossover.py
import numpy as np
import random
from copy import deepcopy
from typing import Tuple
from .Solution_RD import SolutionRD
from .repair import _repair_houses, _repair_participants

"""
Two crossover operators for the Running Dinner GA:
- pmx_crossover: Partially Mapped Crossover (PMX) 
- Uniform crossover

The crossover operators are implemented as functions that take two
SolutionRD parents and return a new SolutionRD child.
The child genome is then repaired to ensure it satisfies all hard constraints.
"""

def pmx_crossover(
    parent1: SolutionRD,
    parent2: SolutionRD,
    max_retries: int = 10,
    verbose: bool = False
) -> tuple[SolutionRD, SolutionRD]:
    """
    Partially Mapped Crossover (PMX) producing two offspring, with repair & validation.
    
    This specialized crossover works well for permutation problems like the running dinner.
    PMX preserves relative order and position of elements from parents while avoiding duplicates.
    
    Algorithm:
    1) Select two random crossover points
    2) Copy the segment between points from parent1 to child1 (and parent2 to child2)
    3) Map corresponding elements between the segments to build replacement mappings
    4) Fill remaining positions using the mapping rules to prevent duplicates
    
    Args:
        parent1, parent2: the two parents
        max_retries: how many attempts before fallback
        verbose: print progress if True
        
    Returns:
        (child1, child2): two valid SolutionRD offspring
    """
    L = len(parent1.genome)
    n_h = parent1.n_houses
    
    for attempt in range(1, max_retries + 1):
        # We'll apply PMX to the participant segments only
        # Select segment of genome to focus on (house assignments or one of the participant blocks)
        if random.random() < 0.5:
            # House assignments segment
            segment_start = 0
            segment_end = n_h
        else:
            # One of the participant blocks
            course = random.randint(0, parent1.n_courses - 1)
            block_size = parent1.n_participants + parent1.empty_spots
            segment_start = n_h + course * block_size
            segment_end = segment_start + block_size
        
        # Select crossover points within the selected segment
        cp1, cp2 = sorted(random.sample(range(segment_start, segment_end), 2))
        
        if verbose:
            print(f"[PMX] attempt {attempt}, segment=[{segment_start}:{segment_end}], " 
                  f"crossover points=[{cp1}:{cp2}]")
        
        # Create initial children by copying parents
        g1 = parent1.genome.copy()
        g2 = parent2.genome.copy()
        
        # Copy segment between crossover points
        segment1 = parent1.genome[cp1:cp2].copy()
        segment2 = parent2.genome[cp1:cp2].copy()
        
        # Step 1: Copy the segment from parent1 to child2 and parent2 to child1
        g1[cp1:cp2] = segment2
        g2[cp1:cp2] = segment1
        
        # Step 2: Create mappings for the swapped segments
        map1 = {}  # Maps from parent2 to parent1
        map2 = {}  # Maps from parent1 to parent2
        for i in range(len(segment1)):
            map1[segment2[i]] = segment1[i]
            map2[segment1[i]] = segment2[i]
        
        # Step 3: Fill in the remaining positions using the mappings
        for i in list(range(segment_start, cp1)) + list(range(cp2, segment_end)):
            # For child1
            value1 = parent1.genome[i]
            while value1 in map1:
                value1 = map1[value1]
            g1[i] = value1
            
            # For child2
            value2 = parent2.genome[i]
            while value2 in map2:
                value2 = map2[value2]
            g2[i] = value2
        
        # Repair both children
        g1 = _repair_houses(g1, parent1)
        g1 = _repair_participants(g1, parent1)
        g2 = _repair_houses(g2, parent1)
        g2 = _repair_participants(g2, parent1)
        
        # Create SolutionRD objects from the repaired genomes
        c1 = deepcopy(parent1)
        c2 = deepcopy(parent1)
        c1.set_genome(g1)
        c2.set_genome(g2)
        
        # Check validity
        valid1 = c1.check_validity_of_genome(verbose=False)
        valid2 = c2.check_validity_of_genome(verbose=False)
        
        if valid1 and valid2:
            if verbose:
                print(f"[PMX] success on attempt {attempt}")
            return c1, c2
    
    # Fallback if no valid children found
    if verbose:
        print("[PMX] retries exhausted; falling back per child")
    
    fallback1 = deepcopy(parent1 if parent1.fitness <= parent2.fitness else parent2)
    fallback2 = deepcopy(parent1 if parent1.fitness <= parent2.fitness else parent2)
    return fallback1, fallback2

def uniform_crossover(
    parent1: SolutionRD,
    parent2: SolutionRD,
    max_retries: int = 10,
    verbose: bool = False
) -> tuple[SolutionRD, SolutionRD]:
    """
    Uniform crossover producing two offspring, with repair & validation.

    Repeated up to `max_retries`:
      1) Draw random mask of length L
      2) child1.genome = where(mask, p1, p2); child2 = where(mask, p2, p1)
      3) Repair & validate both
      4) If both valid, return them; else retry

    Fallback: invalid child → fitter parent.

    Args:
        parent1, parent2: the two parents
        max_retries: attempts before fallback
        verbose: print progress if True

    Returns:
        (child1, child2): two valid SolutionRD offspring
    """
    L = len(parent1.genome)
    for attempt in range(1, max_retries + 1):
        if verbose:
            print(f"[uniform] attempt {attempt}")

        mask = np.random.rand(L) < 0.5
        g1 = np.where(mask, parent1.genome, parent2.genome)
        g2 = np.where(mask, parent2.genome, parent1.genome)

        g1 = _repair_houses(g1, parent1)
        g1 = _repair_participants(g1, parent1)
        g2 = _repair_houses(g2, parent1)
        g2 = _repair_participants(g2, parent1)

        c1 = deepcopy(parent1); c1.set_genome(g1)
        c2 = deepcopy(parent1); c2.set_genome(g2)

        if c1.check_validity_of_genome(verbose=False) and c2.check_validity_of_genome(verbose=False):
            if verbose:
                print(f"[uniform] success on attempt {attempt}")
            return c1, c2

    if verbose:
        print("[uniform] retries exhausted; falling back per child")

    fallback1 = deepcopy(parent1 if parent1.fitness <= parent2.fitness else parent2)
    fallback2 = deepcopy(parent1 if parent1.fitness <= parent2.fitness else parent2)
    return fallback1, fallback2





