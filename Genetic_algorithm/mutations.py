import random
import Genetic_algorithm.config as config
from Genetic_algorithm.genome_old import Genome
from copy import deepcopy
import numpy as np

# make sure the input genome is type Genome

def logistic_mutation(genome): #this one
    new_genome = deepcopy(genome)
    rand_1 = random.randint(0, config.N_HOUSES-1)
    rand_2 = random.randint(0, config.N_HOUSES-1)
    while rand_1 == rand_2:
        rand_2 = random.randint(0, config.N_HOUSES-1)
    new_genome.swap_house_assigments(rand_1, rand_2)
    return new_genome
    

def social_mutation(genome): # this one
    new_genome = deepcopy(genome)
    rand_1 = random.randint(0, config.LEN_COURSE-1)
    rand_2 = random.randint(0, config.LEN_COURSE-1)
    while rand_1 == rand_2:
        rand_2 = random.randint(0, config.LEN_COURSE-1)
    rand_course = random.randint(0, config.N_COURSES-1)
    new_genome.swap_course_assignments(rand_course, rand_1, rand_2)
    return new_genome

def pacman_mutation(array):

    # ran int between 0 and len(genome)
    window_size = random.randint(0, len(array)-1)
    # random int between 0 and len(genome) - window_size
    new_genome = np.concat((array[window_size:], array[:window_size]))
    return new_genome
    
def logistic_mutation_2(genome): # this one

    logistic_array = genome.house_assignments.copy()
    new_logistic_array = pacman_mutation(logistic_array)
    new_genome = deepcopy(genome)
    new_genome.house_assignments = new_logistic_array
    new_genome.secure_all_owner_to_houses()
    return new_genome