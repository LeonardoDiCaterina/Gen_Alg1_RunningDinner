# ga_rd/Individual_RD.py
from Classes.Individual import Individual
from .Solution_RD import SolutionRD
from copy import deepcopy
import random
import numpy as np
from ga_rd.crossover import uniform_crossover
from ga_rd.mutation import swap_mutation, scramble_mutation, inversion_mutation

class IndividualRD(Individual):
    """
    GA interface wrapping the domain SolutionRD.
    """
    def __init__(
        self,
        house_coords: np.ndarray,
        n_participants: int,
        capacity_of_houses: int,
        a: float,
        afterparty_house: int = None,
        crossover_method: str = 'uniform',
        crossover_prob: float = 0.9,
        mutation_method: str = 'swap',
        mutation_prob: float = 0.1
    ):
        super().__init__()
        self.solution = SolutionRD(
            house_coords=house_coords,
            participant_homes=house_coords[:n_participants],
            n_participants=n_participants,
            capacity_of_houses=capacity_of_houses,
            a=a
        )

        # optionally tag on an afterparty house index
        if afterparty_house is not None:
           self.solution.afterparty_house = afterparty_house

        # store desired crossover style: 'one_point', 'uniform', or 'random'
        valid_crossover = {'one_point', 'uniform', 'random'}
        if crossover_method not in valid_crossover:
            raise ValueError(f"crossover_method must be one of {valid_crossover}")
        if not (0.0 <= crossover_prob <= 1.0):
            raise ValueError("crossover_prob must be between 0 and 1")
        self.crossover_method = crossover_method
        self.crossover_prob = crossover_prob

        # Mutation setup
        valid_mutation = {'swap', 'scramble', 'inversion'}
        if mutation_method not in valid_mutation:
            raise ValueError(f"Invalid mutation_method: {mutation_method}")
        if not (0.0 <= mutation_prob <= 1.0):
            raise ValueError("mutation_prob must be between 0 and 1")
        self.mutation_method = mutation_method
        self.mutation_prob = mutation_prob

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
        Apply a selected mutation operator with prob `mutation_prob`.

        Returns:
            IndividualRD: either an unmodified copy (no mutation) or a mutated offspring.
        """
        # Work on a copy to preserve the original
        child = deepcopy(self)

        # Decide whether to mutate at all
        if random.random() > self.mutation_prob:
            # No mutation: preserve cached fitness
            child._fitness = self._fitness
            return child

        # Otherwise perform the chosen mutation
        if self.mutation_method == 'swap':
            mutated = swap_mutation(child)
        elif self.mutation_method == 'scramble':
            mutated = scramble_mutation(child)
        elif self.mutation_method == 'inversion':
            mutated = inversion_mutation(child)
        else:
            raise ValueError(f"Unknown mutation_method: {self.mutation_method}")

        # Invalidate fitness so it'll be recalculated
        mutated._fitness = None
        return mutated

    def crossover(self, other: "IndividualRD") -> "IndividualRD":
        """
        With prob (1 - crossover_prob), perform no crossover and
        return a copy of self. Otherwise, apply the selected crossover operator,
        wrap the resulting SolutionRD into a new IndividualRD, and return it.
        """
        # Only crossover if random number is greater than the crossover_prob
        if random.random() > self.crossover_prob:
            # Skip crossover
            child = deepcopy(self)
            child._fitness = None
            return child

        # Otherwise, perform crossover
        if self.crossover_method == 'one_point':
            cross_fn = one_point_crossover
        elif self.crossover_method == 'uniform':
            cross_fn = uniform_crossover
        else:  # 'random'
            cross_fn = random.choice([one_point_crossover, uniform_crossover])

        # Call the standalone crossover
        child_sol = cross_fn(
            self.solution,
            other.solution,
            max_retries=getattr(self, 'max_retries', 10),
            verbose=getattr(self, 'verbose', False)
        )

        # Wrap and return the offspring
        child = IndividualRD(
            house_coords=child_sol.house_coords,
            n_participants=child_sol.n_participants,
            capacity_of_houses=child_sol.capacity_of_houses,
            a=child_sol.a,
            afterparty_house=getattr(child_sol, 'afterparty_house', None),
            crossover_method=self.crossover_method,
            crossover_prob=self.crossover_prob,
            mutation_method=self.mutation_method,
            mutation_prob=self.mutation_prob
        )
        child.solution = child_sol
        child._fitness = None
        return child


    def semantic_key(self):
        return tuple(self.solution.get_genome.tolist())
    