from copy import deepcopy
# import abc
from abc import ABC, abstractmethod
import numpy as np
class Solution(ABC):
    def __init__(self):
        
        self.genome = np.arange(10)
        self.default_n_mutations = 1
        
    @property
    @abstractmethod
    def fitness(self):
        raise  NotImplementedError("Subclasses must implement the fitness property.")
    
    def __lt__(self,other):
        """
        Less than operator for the solution class.

        Args:
            other (soluton): The other solution.
        Returns:
            bool: True if the current fitness is less than the other fitness.
        """
        return self.fitness < other.fitness
    def __le__(self,other):
        """
        Less than or equal operator for the solution class.
        Args:
            other (soluton): The other solution.
        Returns:
            bool: True if the current fitness is less than or equal to the other fitness.
        """
        return self.fitness <= other.fitness
    def __gt__(self,other):
        """
        Greater than operator for the solution class.
        Args:
            other (soluton): The other solution.
        Returns:
            bool: True if the current fitness is greater than the other fitness.
        """
        return self.fitness > other.fitness
    def __ge__(self,other):
        """
        Greater than or equal operator for the solution class.
        Args:
            other (soluton): The other solution.
        Returns:
            bool: True if the current fitness is greater than or equal to the other fitness.
        """
        return self.fitness >= other.fitness
    
    #the equality and inequality operators are not implemented the same way as the others,
    # we are going to use them to check if the genome is the same

    
    def __eq__(self,other):
        """
        Equality operator for the solution class.
        Args:
            other (soluton): The other solution.
        Returns:
            bool: True if the current solution is equal to the other solution.
        """
        return np.array_equal(self.get_genome,other.get_genome)
    def __ne__(self,other):
        """
        Not equal operator for the solution class.
        Args:
            other (soluton): The other solution.
        Returns:
            bool: True if the current solution is not equal to the other solution.
        """
        return not np.array_equal(self.get_genome,other.get_genome)
    
    # the hash function is used to hash the solution in dictionaries, we are going to use the genome of the solution as the hash    
    def __hash__(self):
        """
        Hash function for the solution class.
        Args:
            other (soluton): The other solution.
        Returns:
            int: The hash of the solution.
        """
        return hash(self.get_genome.tobytes())
    
    """ now we are going to take care of the representation of the solution and the string representation of the solution """
    def __repr__(self):
        """
        Representation of the solution class.
        For debugging purposes.
        Returns:
            str: The representation of the solution.
        """
        return f'Solution({self.fitness})'
    
    # we need to implement copy_solution to copy the solution
    @property
    def copy_solution(self):
        """
        Copy the solution.
        Returns:
            Solution: The copied solution.
        """
        new_solution = deepcopy(self)
        return new_solution
    
    @property
    def get_genome(self):
        """
        Getter for the partecipants.
        Returns:
            np.array: The partecipants.
        """
        return np.array(self.genome)
    
    def set_genome(self, new_genome: np.array):
        """
        Setter for the genome.
        Args:
            new_genome (np.array): The new genome.
        """
        
        self.genome = new_genome
    
    
    # to implement the crossover operator we are going to use the addition and the subtraction of the two solutions
    def __add__(self,other):
        """
        Addition operator for the solution class.
        Args:
            other (soluton): The other solution.
        Returns:
            Solution: The new solution.
        """
        new_solution = self.copy_solution
        genome_a = self.get_genome
        genome_b = other.get_genome
        # we are going to use the intersection of the two solutions to create a new solution
        
        intersection = (genome_a == genome_b).astype(int)
        
        # shuffle only the different partecipants keeping the intersection in place        
        genome_a[intersection == 0] = np.random.permutation(genome_a [intersection == 0])
        
        new_solution.set_genome(genome_a)
        
        
                
        return new_solution    
    
    # the subtraction operator is going to be used to return the differnce in fitness between two solutions
    def __sub__(self,other):
        """
        Subtraction operator for the solution class.
        Args:
            other (soluton): The other solution.
        Returns:
            Solution: The new solution.
        """
     
        return self.fitness - other.fitness
        
        

    
    
    # I'm going t make a swap operator to swap the partecipants of two solutions
    def swap(self, p1: int, p2: int):
        mutant = self.copy_solution
        
        # swap the partecipants
        place1 = np.where(self.get_genome == p1)
        place2 = np.where(self.get_genome == p2)
        
        mutant_genome = mutant.get_genome
        mutant_genome[place1] = p2
        mutant_genome[place2] = p1
        mutant.set_genome(mutant_genome)
        
        return mutant
    
    def get_default_n_mutations(self):
        """
        Getter for the default number of mutations.
        Returns:
            int: The default number of mutations.
        """
        return int(self.default_n_mutations)
    
    def set_default_n_mutations(self, n_mutations: int):
        """
        Setter for the default number of mutations.
        Args:
            n_mutations (int): The new default number of mutations.
        """
        if isinstance(n_mutations, int):
            self.default_n_mutations = n_mutations
        else:
            raise ValueError("The number of mutations must be an int")    
    
    def __mul__(self,args):
        """
        
        Args:
            other (int): The number of mutated solutions to return.
        Returns:
            Solution: The new solution.
        """
        if isinstance(args, int):
            n_children = args
            n_mutations = self.default_n_mutations
        elif isinstance(args, tuple):
            n_children = args[0]
            n_mutations = args[1]
        elif isinstance(args, Solution):
            return NotImplemented
        else:
            raise ValueError("The argument must be an int or a tuple of two ints")        
        if isinstance(args, int):
            n_children = args
            n_mutations = self.default_n_mutations
        elif isinstance(args, tuple):
            n_children = args[0]
            n_mutations = args[1]
        elif isinstance(args, Solution):
            return NotImplemented
        else:
            raise ValueError("The argument must be an int or a tuple of two ints")        
        unique_solutions = {self.copy_solution}  # initialize with a copy of self
        for _ in range(n_children):
            new_solution = self.copy_solution
            for __ in range(n_mutations):
                p1_index = np.random.randint(0, len(self.get_genome))
                p2_index = np.random.randint(0, len(self.get_genome))
                while p1_index == p2_index:
                    p2_index = np.random.randint(0, len(self.get_genome))
                new_solution = new_solution.swap(p1_index, p2_index)
            unique_solutions.add(new_solution)
        return unique_solutions
            
    def __pow__(self,args):
        """
        
        Args:
            other (int): The number of mutated solutions to return.
        Returns:
            Solution: a list of new solutions.
        """
        return list(self * args)
        
    # the division operator operator is going ot be a tool to inspect the difference in the genome of two solutions
    def __floordiv__(self,other):
        """
        Subtraction operator for the solution class.
        Args:
            other (soluton): The other solution.
        Returns:
            Solution: The new solution.
        """
        genome_a = (self.get_genome).astype(float)
        genome_b = (other.get_genome).astype(float)
        
        mask = (genome_a != genome_b).astype(bool)
        print(genome_a[mask].astype(int))
        print("-"*50)
        print("*"*20 + " Mask " + "*"*25)
        print(mask.astype(int))
        print("-"*50)
        print("-"*50)
        print("*"*20 + " Genome A " + "*"*20)
        genome_a[np.where(self.get_genome == other.get_genome)] = np.nan
        print(genome_a)
        print("-"*50)
        print("*"*20 + " Genome B " + "*"*20)
        genome_b[np.where(self.get_genome == other.get_genome)] = np.nan
        print(genome_b)
        print("-"*50)
        print("-"*10 + " Delta Fitness " + "-"*10)
        print(self - other)
        print("-"*50)
        print("-"*50)
        
