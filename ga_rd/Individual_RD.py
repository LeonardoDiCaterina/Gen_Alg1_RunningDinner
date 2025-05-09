# ga_rd/Individual_RD.py
from Classes.Individual import Individual
from .Solution_RD import SolutionRD
from copy import deepcopy
import numpy as np
from typing import Tuple
import random
from .crossover import uniform_crossover, pmx_crossover
from .mutation import swap_mutation, scramble_mutation, inversion_mutation

class IndividualRD(Individual):
    """
    Wraps a domain SolutionRD into a generic GA Individual interface.
    """

    def __init__(
        self,
        participant_homes:  np.ndarray,
        house_coords:       np.ndarray,
        host_idxs:          list[int],
        capacity_of_houses: int,
        a:                  float = 1.0,
        afterparty_house:   int   = None,
        crossover_method:   str   = 'uniform',
        crossover_prob:     float = 0.9,
        mutation_method:    str   = 'swap',
        mutation_prob:      float = 0.1
    ):
        super().__init__()
        # pass host_idxs into SolutionRD
        self.solution = SolutionRD(
            participant_homes  = participant_homes,
            house_coords       = house_coords,
            host_idxs          = host_idxs,
            capacity_of_houses = capacity_of_houses,
            a                  = a
        )

        # optional afterparty
        if afterparty_house is not None:
            self.solution.afterparty_house = afterparty_house

        # crossover settings
        valid_x = {'pmx', 'uniform', 'random'}
        if crossover_method not in valid_x:
            raise ValueError(f"crossover_method must be in {valid_x}")
        self.crossover_method = crossover_method
        self.crossover_prob   = crossover_prob

        # mutation settings
        valid_m = {'swap','scramble','inversion'}
        if mutation_method not in valid_m:
            raise ValueError(f"mutation_method must be in {valid_m}")
        self.mutation_method = mutation_method
        self.mutation_prob   = mutation_prob

    def random_representation(self, max_attempts=2000):
        for attempt in range(max_attempts):
            self.solution.generate_genome()
            self.solution.repair_houses()
            self.solution.repair_participants()
            self.solution.secure_owner_to_houses()
            # REPAIR AGAIN after seating owners!
            self.solution.repair_participants()
            self._fitness = None
            if self.solution.check_validity_of_genome():
                return
        # If we get here, all attempts failed
        # Print last genome with verbose for debugging
        self.solution.check_validity_of_genome(verbose=True)
        print(self.solution.genome)
        raise RuntimeError(
            f"random_representation: genome still invalid after {max_attempts} attempts at house-repair, participant-repair, and secure_owner_to_houses()!"
        )


    def check_representation(self) -> bool:
        return self.solution.check_validity_of_genome()

    def calculate_fitness(self) -> float:
        return self.solution.fitness

    @property
    def fitness(self) -> float:
        return super().fitness

    def crossover(self, other: "IndividualRD") -> tuple["IndividualRD","IndividualRD"]:
        """
        With probability self.crossover_prob, apply the chosen crossover operator
        to produce *two* offspring; otherwise just copy (self, other) through.
        """
        # 1) Maybe skip crossover entirely
        if random.random() > self.crossover_prob:
            c1 = deepcopy(self)
            c2 = deepcopy(other)
            c1._fitness = None
            c2._fitness = None
            return c1, c2

        # 2) Pick the correct standalone crossover function
        if   self.crossover_method == 'pmx':
            fn = pmx_crossover
        elif self.crossover_method == 'uniform':
            fn = uniform_crossover
        else:  # 'random'
            fn = random.choice([pmx_crossover, uniform_crossover])

        # 3) Call it, getting two SolutionRD children back
        sol1, sol2 = fn(
            self.solution,
            other.solution,
            max_retries=getattr(self, 'max_retries', 10),
            verbose=getattr(self, 'verbose', False)
        )

        # 4) Wrap each SolutionRD into its own IndividualRD
        def wrap(sol: SolutionRD) -> "IndividualRD":
            child = IndividualRD(
                participant_homes   = sol.participant_homes,
                house_coords        = sol.house_coords,
                host_idxs           = sol.host_idxs,
                capacity_of_houses  = sol.capacity_of_houses,
                a                    = sol.a,
                afterparty_house     = getattr(sol, 'afterparty_house', None),
                crossover_method     = self.crossover_method,
                crossover_prob       = self.crossover_prob,
                mutation_method      = self.mutation_method,
                mutation_prob        = self.mutation_prob
            )
            child.solution = sol
            child._fitness = None
            return child

        return wrap(sol1), wrap(sol2)
    
    def mutation(self) -> "IndividualRD":
        # copy
        child = deepcopy(self)
        # maybe skip
        if random.random() > self.mutation_prob:
            child._fitness = self._fitness
            return child

        # choose operator
        if   self.mutation_method == 'swap':    mutant = swap_mutation(child)
        elif self.mutation_method == 'scramble': mutant = scramble_mutation(child)
        elif self.mutation_method == 'inversion':mutant = inversion_mutation(child)
        else: raise ValueError(f"Unknown mutation: {self.mutation_method}")

        mutant._fitness = None
        return mutant

    def semantic_key(self) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
        """
        A “fuzzy” key for semantic equality.

        We iterate over each house index h=0..n_houses-1 and pull out:
          - which course that house hosts (an int in {0,1,2,-1})
          - which guest IDs actually sit there (ignore any -1 placeholders, treat as a set)

        Two individuals with the same (course, set_of_guests) for every house
        will produce the same key, even if the internal ordering within each
        house’s guest‐list differs.

        Returns:
            A tuple of length n_houses, each entry is
              (course_idx, sorted_tuple_of_guest_IDs)
        """
        sol = self.solution
        key = []
        for h in range(sol.n_houses):
            course_idx, members = sol.get_house(h)  # members is e.g. np.array or list
            # filter out empty‐seat markers, sort the remaining guest IDs
            real = [int(x) for x in members if x >= 0]
            real.sort()
            key.append((int(course_idx), tuple(real)))
        return tuple(key)