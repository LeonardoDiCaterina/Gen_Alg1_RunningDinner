# For your project to be successful you must implement at least 2 crossover operators

# ga_rd/crossover.py
import numpy as np
from copy import deepcopy
from typing import Tuple
from ga_rd.Solution_RD import SolutionRD

"""
Two crossover operators for the Running Dinner GA:
- XX NEED TO COMPLETE XX crossover and then update 
- Uniform crossover

The crossover operators are implemented as functions that take two
SolutionRD parents and return a new SolutionRD child.
The child genome is then repaired to ensure it satisfies all hard constraints.
"""

def uniform_crossover(
    parent1: SolutionRD,
    parent2: SolutionRD,
    max_retries: int = 10,
    verbose: bool = False
) -> SolutionRD:
    """
    Uniform crossover with repair and validation, with optional verbose output.
    
    Tries up to `max_retries` to produce a valid child genome:
      1) Mix genes via uniform mask
      2) Repair house segment and participant blocks
      3) Validate; if valid, return child
      4) Otherwise retry
    
    If all retries fail, falls back to the fitter of the two parents.
    
    Args:
        parent1 (SolutionRD): First parent solution.
        parent2 (SolutionRD): Second parent solution.
        max_retries (int): Max attempts before fallback.
        verbose (bool): If True, print debug info on success or final fallback.
    Returns:
        SolutionRD: A valid offspring solution.
    """
    for attempt in range(1, max_retries + 1):
        # 1) Mix genes
        g1, g2 = parent1.genome, parent2.genome
        mask   = np.random.rand(len(g1)) < 0.5
        newg   = np.where(mask, g1, g2)

        # 2) Repair both segments
        newg = _repair_houses(newg, parent1)
        newg = _repair_participants(newg, parent1)

        # 3) Wrap into a child and validate
        child = deepcopy(parent1)
        child.set_genome(newg)
        if child.check_validity_of_genome():
            if verbose:
                print(f"[uniform_crossover] Success on attempt {attempt}")
            return child

    # 4) All retries failed -> fallback to fitter parent
    if verbose:
        print("[uniform_crossover] All retries failed, falling back to fitter parent")
    fallback = parent1 if parent1.fitness <= parent2.fitness else parent2
    return deepcopy(fallback)


def _repair_houses(self, genome: np.ndarray, template: SolutionRD) -> np.ndarray:
        # How many slots in the “house segment”?
        n_h = template.n_houses
        # desired might look like {0:3, 1:3, 2:3, -1:1} if you have 10 total houses: 3 starter, 3 main, 3 dessert, 1 empty.
        desired = {c: template.houses_per_course for c in range(template.n_courses)}
        desired[-1] = template.empty_houses

        # current counts. I.e., count what we actually have in newg[:n_h]
        # E.g. current_counts might be {0:4, 1:2, 2:3, -1:1} if uniform crossover gave us 4 starters instead of 3.
        vals, cnts = np.unique(genome[:n_h], return_counts=True)
        current_counts = dict(zip(vals.tolist(), cnts.tolist()))

        # find over/under counts 
        # I.e., Compare “have” vs. “want” to list which values are over and under represented.
        over, under = [], []
        for val, want in desired.items():
            have = current_counts.get(val, 0)
            if have > want:
                over.extend([val] * (have - want))
            elif have < want:
                under.extend([val] * (want - have))

        # if no over/under counts, return genome
        if not over and not under:
            return genome

        # positions of over-represented values
        positions = {v: np.where(genome[:n_h] == v)[0].tolist() for v in set(over)}
        np.random.shuffle(under)

        # replace excess occurrences
        for v in over:
            idx_list = positions[v]
            idx = idx_list.pop(np.random.randint(len(idx_list)))
            genome[idx] = under.pop()
        return genome


def _repair_participants(
    genome: np.ndarray,
    template: SolutionRD
) -> np.ndarray:
    """
    Repair each course's participant block so that:
      - Each guest ID 0..n_participants-1 appears exactly once per block
      - Exactly `empty_spots` placeholder values (-1) per block

    Args:
        genome (np.ndarray): Full genome array (houses + participant blocks).
        template (SolutionRD): A reference solution carrying block sizes.

    Returns:
        np.ndarray: The genome with corrected participant blocks.
    """
    # Compute offsets and block length
    n_h   = template.n_houses                 # number of house-assignment genes
    n_p   = template.n_participants            # number of distinct guests
    e_sp  = template.empty_spots               # number of empty seats per course
    blk   = n_p + e_sp                         # length of each course's block

    # Loop over each course block
    for course in range(template.n_courses):
        start = n_h + course * blk             # index where this block begins
        block = genome[start : start + blk]    # slice out the block

        # Count occurrences of each value in the block
        vals, cnts = np.unique(block, return_counts=True)
        count      = dict(zip(vals.tolist(), cnts.tolist()))

        # Build desired counts: 1 of each guest, plus e_sp of -1
        desired = {p: 1 for p in range(n_p)}
        if e_sp:
            desired[-1] = e_sp

        # Identify over- and under-represented values
        over, under = [], []
        for v, want in desired.items():
            have = count.get(v, 0)
            if have > want:
                # too many of v → will need to replace (have-want) occurrences
                over.extend([v] * (have - want))
            elif have < want:
                # too few of v → need to add (want-have) occurrences
                under.extend([v] * (want - have))

        # If any imbalance, perform swaps
        if over:
            # map each over-represented value to its indices in the block
            pos_map = {
                v: list(np.where(block == v)[0])
                for v in set(over)
            }
            np.random.shuffle(under)            # randomize replacement order

            # For each extra occurrence, replace one at random
            for v in over:
                idx_list = pos_map[v]           # positions holding the surplus value
                i = idx_list.pop(
                    np.random.randint(len(idx_list))
                )                               # pick one index to replace
                block[i] = under.pop()          # replace with a needed value

            # Write the repaired block back into the genome
            genome[start : start + blk] = block

    # Return the fully repaired genome array
    return genome





