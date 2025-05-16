import random
import Genetic_algorithm.config as config
from Genetic_algorithm.genome import Genome
from copy import deepcopy

# make sure the input genome is type Genome

def logistic_mutation(genome):
    new_genome = deepcopy(genome)
    rand_1 = random.randint(0, config.N_HOUSES-1)
    rand_2 = random.randint(0, config.N_HOUSES-1)
    while rand_1 == rand_2:
        rand_2 = random.randint(0, config.N_HOUSES-1)
    new_genome.swap_house_assigments(rand_1, rand_2)
    return new_genome
    

def social_mutation(genome):
    new_genome = deepcopy(genome)
    rand_1 = random.randint(0, config.LEN_COURSE-1)
    rand_2 = random.randint(0, config.LEN_COURSE-1)
    while rand_1 == rand_2:
        rand_2 = random.randint(0, config.LEN_COURSE-1)
    rand_course = random.randint(0, config.N_COURSES-1)
    new_genome.swap_course_assignments(rand_course, rand_1, rand_2)
    return new_genome