#Individual.py
from abc import ABC, abstractmethod
from copy import deepcopy


class Individual(ABC):

    def __init__(self):
        self._mutation_probability = 0.1
        self._crossover_probability = 0.5
        self.initial_population = 5
        self._fitness = None

    @property
    def mutation_probability(self):
        return self._mutation_probability

    @mutation_probability.setter
    def mutation_probability(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Mutation probability must be an int or float.")
        if not (0 <= value <= 1):
            raise ValueError("Mutation probability must be between 0 and 1.")
        self._mutation_probability = value

    @property
    def crossover_probability(self):
        return self._crossover_probability

    @crossover_probability.setter
    def crossover_probability(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Crossover probability must be an int or float.")
        if not (0 <= value <= 1):
            raise ValueError("Crossover probability must be between 0 and 1.")
        self._crossover_probability = value

    @abstractmethod
    def random_representation(self) -> None:
        """ Generate a random representation of the individual."""
        pass
    
    @abstractmethod
    def check_representation(self) -> bool:
        """ Check if the representation is valid."""
        return True
    @property
    def fitness(self):
        if self._fitness is None:
            self._fitness = self.calculate_fitness()  # Only calculate when needed but once it is calculated it is stored
        return self._fitness
    
    @abstractmethod
    def calculate_fitness(self):
        pass
    
    # Method for mutation
    @abstractmethod
    def mutation(self):
        pass

    # Method for crossover
    @abstractmethod
    def crossover(self, other):
        pass
    
    
    
    @abstractmethod
    def semantic_key(self):
        """Return a hashable object that defines semantic equivalence.
        This is used to check if two individuals are semantically equivalent.
        In Grouping problems for example, the semantic key is the ordered set of the elements in the genome.

        Returns:
            hashable: A hashable object that defines semantic equivalence.
        
        """
        pass

    def __hash__(self):
        return hash(self.semantic_key())    


    def __iter__(self):
        for _ in range(self.initial_population):
            yield next(self)
        
    def __next__(self):
        copy = self.copy_Individual(delete_fitness=True)
        print(f"mutating: {copy}")
        return copy.mutation()

    def __matmul__(self, other):
        return self.crossover(other)
    
    def __pow__(self, power):
         return [self.crossover(self) for _ in range(power)]

    def __call__(self):
        return self.fitness

    def __float__(self):
        return float(self.fitness) if self.fitness is not None else float('-inf')

    # Comparison Methods
    def __eq__(self, other):
        if not isinstance(other, Individual):
            return False
        return self.semantic_key() == other.semantic_key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return float(self) < float(other)
    
    def __le__(self, other):
        return float(self) <= float(other)
    
    def __gt__(self, other):
        return float(self) > float(other)
    
    def __ge__(self, other):
        return float(self) >= float(other)
    
    
    def __len__(self):
        """ Return the length of the genome. """
        return len(self.genome)

    # String Representation
    def __str__(self):
        return f"Fitness: {self.fitness}"
    
    # Deep Copy
    def copy_Individual(self, delete_fitness=True):
        """
        Returns a deep copy of the individual, optionally deleting the fitness attribute.
        This is useful for creating a new individual that is a copy of the current one and are ready to be mutated or crossed over.
        The fitness of the new individual will be set to None if delete_fitness is True.
        This is necessary because the fitness of the new individual will be calculated again after mutation or crossover.
        
        Args:
            delete_fitness (bool): If True, the fitness of the new individual will be set to None.
        Returns:
            Individual: A deep copy of the individual.
        
        """
        new_individual = deepcopy(self)
        if delete_fitness:
            new_individual._fitness = None
        return new_individual