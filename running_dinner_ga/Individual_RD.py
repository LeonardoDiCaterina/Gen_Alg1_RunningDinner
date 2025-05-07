# running_dinner_ga/Individual_RD.py

from Classes.Individual import Individual
from .Solution_RD import SolutionRD
from copy import deepcopy
import random
import numpy as np

class IndividualRD(Individual):
    """
    GA interface wrapping the domain SolutionRD.
    """
    def __init__(self, house_coords, n_participants, capacity_of_houses, a):
        super().__init__()
        self.solution = SolutionRD(house_coords, n_participants, capacity_of_houses, a)
        self.solution.generate_genome

    def random_representation(self):
        self.solution.generate_genome
        self._fitness = None

    def check_representation(self):
        return self.solution.check_validity_of_genome()

    def calculate_fitness(self):
        return self.solution.fitness

    @property
    def fitness(self):
        # cache via base class .fitness property
        return super().fitness

    def mutation(self) -> "IndividualRD":
        """
        Swap two positions *within* the same segment:
        keep trying until genome_swap() succeeds.
        """
        mutant = deepcopy(self)
        n_h = mutant.solution.n_houses
        L   = mutant.solution.genome.size

        # keep trying until we get a valid swap
        while True:
            # pick segment at random
            if random.random() < 0.5:
                i, j = random.sample(range(0, n_h), 2)
            else:
                i, j = random.sample(range(n_h, L), 2)

            try:
                mutant.solution.genome_swap(i, j)
                break
            except ValueError:
                # indices crossed segments? try again
                continue

        return mutant

    def _mutate_swap(self, sol):
        i, j = np.random.choice(len(sol.get_genome), 2, replace=False)
        sol.genome_swap(i, j)

    def _mutate_scramble(self, sol):
        genome = sol.get_genome
        i, j = sorted(np.random.choice(len(genome), 2, replace=False))
        genome[i:j] = np.random.permutation(genome[i:j])
        sol.set_genome(genome)

    def _mutate_inversion(self, sol):
        genome = sol.get_genome
        i, j = sorted(np.random.choice(len(genome), 2, replace=False))
        genome[i:j] = genome[i:j][::-1]
        sol.set_genome(genome)

    def crossover(self, other):
        ops = [self._crossover_one_point, self._crossover_uniform]
        op = np.random.choice(ops)
        child_sol = op(self.solution, other.solution)
        child = IndividualRD(
            child_sol.house_coords,
            child_sol.n_participants,
            child_sol.capacity_of_houses,
            child_sol.a
        )
        child.solution = child_sol
        child._fitness = None
        return child


    def semantic_key(self):
        return tuple(self.solution.get_genome.tolist())
    
    
    def _repair_houses(self, genome: np.ndarray, template: SolutionRD) -> np.ndarray:
        n_h = template.n_houses
        # desired counts per course and empties
        desired = {c: template.houses_per_course for c in range(template.n_courses)}
        desired[-1] = template.empty_houses

        # current counts
        vals, cnts = np.unique(genome[:n_h], return_counts=True)
        current_counts = dict(zip(vals.tolist(), cnts.tolist()))

        # find over/under counts
        over, under = [], []
        for val, want in desired.items():
            have = current_counts.get(val, 0)
            if have > want:
                over.extend([val] * (have - want))
            elif have < want:
                under.extend([val] * (want - have))

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

    def _repair_participants(self, genome: np.ndarray, template: SolutionRD) -> np.ndarray:
        n_h      = template.n_houses
        n_p      = template.n_participants
        e_sp     = template.empty_spots
        block_len = n_p + e_sp

        for b in range(template.n_courses):
            start = n_h + b * block_len
            end   = start + block_len
            block = genome[start:end]

            # count current vs. desired
            vals, cnts = np.unique(block, return_counts=True)
            count = dict(zip(vals.tolist(), cnts.tolist()))
            desired = {p: 1 for p in range(n_p)}
            if e_sp:
                desired[-1] = e_sp

            over, under = [], []
            for val, want in desired.items():
                have = count.get(val, 0)
                if have > want:
                    over.extend([val] * (have - want))
                elif have < want:
                    under.extend([val] * (want - have))

            if over:
                positions = {v: np.where(block == v)[0].tolist() for v in set(over)}
                np.random.shuffle(under)
                for v in over:
                    pos_list = positions[v]
                    idx_b   = pos_list.pop(np.random.randint(len(pos_list)))
                    block[idx_b] = under.pop()
                genome[start:end] = block

        return genome

    def _crossover_one_point(self, s1, s2):
        g1, g2      = s1.get_genome, s2.get_genome
        cp          = np.random.randint(1, len(g1))
        new_genome  = np.concatenate([g1[:cp], g2[cp:]])

        # repair both segments
        new_genome = self._repair_houses(new_genome, s1)
        new_genome = self._repair_participants(new_genome, s1)

        sol = deepcopy(s1)
        sol.set_genome(new_genome)
        return sol

    def _crossover_uniform(self, s1, s2):
        g1, g2      = s1.get_genome, s2.get_genome
        mask        = np.random.rand(len(g1)) < 0.5
        new_genome  = np.where(mask, g1, g2)

        # repair both segments
        new_genome = self._repair_houses(new_genome, s1)
        new_genome = self._repair_participants(new_genome, s1)

        sol = deepcopy(s1)
        sol.set_genome(new_genome)
        return sol