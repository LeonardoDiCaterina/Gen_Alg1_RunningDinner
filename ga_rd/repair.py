# ga_rd/repair.py
import numpy as np
from copy import deepcopy

def _repair_houses(genome: np.ndarray, template: "SolutionRD") -> np.ndarray:
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


def _repair_participants(genome: np.ndarray, template: "SolutionRD") -> np.ndarray:
    n_h   = template.n_houses
    n_p   = template.n_participants
    e_sp  = template.empty_spots
    blk   = n_p + e_sp

    for course in range(template.n_courses):
        start = n_h + course * blk
        block = genome[start : start + blk]

        # Remove all -1s if e_sp == 0
        if e_sp == 0:
            block = block[block != -1]

        # Count occurrences
        vals, cnts = np.unique(block, return_counts=True)
        count = dict(zip(vals.tolist(), cnts.tolist()))

        # Build desired counts
        desired = {p: 1 for p in range(n_p)}
        if e_sp:
            desired[-1] = e_sp

        # Identify over- and under-represented values
        over, under = [], []
        for v, want in desired.items():
            have = count.get(v, 0)
            if have > want:
                over.extend([v] * (have - want))
            elif have < want:
                under.extend([v] * (want - have))

        # Replace over-represented with under-represented
        pos_map = {v: list(np.where(block == v)[0]) for v in set(over)}
        np.random.shuffle(under)
        for v in over:
            idx_list = pos_map[v]
            i = idx_list.pop(np.random.randint(len(idx_list)))
            block[i] = under.pop()

        genome[start : start + blk] = block

    return genome

