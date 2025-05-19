import random
import numpy as np
from Genetic_algorithm.genome import Genome
from Genetic_algorithm.fitness import ResourceFitness
from Genetic_algorithm.mutations import logistic_mutation, social_mutation, logistic_mutation_2
from Genetic_algorithm.crossovers import social_crossover, logistic_crossover_2, full_crossover
from Genetic_algorithm.base_individual import Individual
from copy import deepcopy
# This class implements a solution for the Resource Distribution problem using a genetic algorithm.

class SolutionRD(Individual):
    def __init__(self,fitness_instance,genome_class,mutation_functions, crossover_functions, prob_social_mutation=0.5, prob_social_crossover=0.5):
        super().__init__()
        self.prob_social_mutation = prob_social_mutation
        self.prob_social_crossover = prob_social_crossover
        # at some point I'll check if the genome_class is a Genome and the fitness_class is a ResourceFitness
        # but not today
        self.mutation_functions = mutation_functions
        self.crossover_functions = crossover_functions
        
        self.genome_class = genome_class
        self.random_representation()
        self.fitness_instance = fitness_instance

    def random_representation(self):
        self.genome = self.genome_class()

    def mutation(self):
        new_genome = None
        if random.random() < self.prob_social_mutation:
            new_genome =  self.mutation_functions[0](self.genome)
            
        else:
            new_genome  = self.mutation_functions[1](self.genome)
    
        self.genome = new_genome
        #print("new genome", self.genome)
        #return new_individual
        
    def crossover(self, other):
        
        new_individual_1 = self.copy_Individual(delete_fitness=True)
        new_individual_2 = other.copy_Individual(delete_fitness=True)
        
        new_genome_1 = None
        new_genome_2 = None
        
        if random.random() < self.prob_social_crossover:
            new_genome_1,new_genome_2 = self.crossover_functions[0](self.genome, other.genome)
        else:
            new_genome_1,new_genome_2 = self.crossover_functions[1](self.genome, other.genome)

        new_individual_1.genome = new_genome_1
        new_individual_2.genome = new_genome_2
        return new_individual_1, new_individual_2
    
    def calculate_fitness(self):
        result = self.fitness_instance.evaluate(self.genome)
        return result
    
    def check_representation(self):
        return True
    
    def semantic_key(self):
        return self.genome.semantic_key()
    
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
        new_individual.genome = self.genome
        new_individual.fitness_instance = self.fitness_instance
        new_individual.mutation_functions = self.mutation_functions
        new_individual.crossover_functions = self.crossover_functions
        return new_individual
    
    def __iter__(self):
        for _ in range(self.initial_population):
            yield next(self)
        
    def __next__(self):
        return SolutionRD(self.fitness_instance, self.genome_class, self.mutation_functions, self.crossover_functions, self.prob_social_mutation, self.prob_social_crossover)