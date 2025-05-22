from copy import deepcopy
import numpy as np
from Genetic_algorithm import config

def full_crossover(genome1, genome2): # this one
    logistic1 = genome1.house_assignments.copy()
    logistic2 = genome2.house_assignments.copy()

    new_gen1 = deepcopy(genome1)
    new_gen1.house_assignments = logistic2
    new_gen1.fix_course_assignments(inplace=True)

    new_gen2 = deepcopy(genome2)
    new_gen2.house_assignments = logistic1
    new_gen2.fix_course_assignments(inplace=True)

    return new_gen1, new_gen2


    
def pillar_crossover(array1, array2):
    """
    Perform a pillar crossover between two arrays.
    
    Args:
        array1 (numpy.ndarray): The first array.
        array2 (numpy.ndarray): The second array.
    
    Returns:
        tuple: A tuple containing the two new arrays after crossover.
    """
    #boolean array to see where the two arrays are equal
    mask = array1 == array2
    #shuffle all the elements of the array1 that are not in the mask
    shuffled_array1 = array1.copy()
    shuffled_array1[~mask] = np.random.permutation(array1[~mask])
    #shuffle all the elements of the array2 that are not in the mask
    shuffled_array2 = array2.copy()
    shuffled_array2[~mask] = np.random.permutation(array2[~mask])
    return shuffled_array1, shuffled_array2

def social_crossover(genome1, genome2): # this one
    
    new_g1 = np.array([])
    new_g2 = np.array([])
    for i in range(config.N_COURSES):
        course_i_1 = genome1.get_course_participants(i)
        course_i_2 = genome2.get_course_participants(i)
        
        new_course_i_1, new_course_i_2 = pillar_crossover(course_i_1, course_i_2)
        new_g1 = np.concatenate((new_g1, new_course_i_1))
        new_g2 = np.concatenate((new_g2, new_course_i_2))
    
    new_gen1 = deepcopy(genome1)
    new_gen1.course_assignments = new_g1
    #new_gen1.secure_all_owner_to_courses()
    new_gen2 = deepcopy(genome2)
    new_gen2.course_assignments = new_g2
    #new_gen2.secure_all_owner_to_houses()
    return new_gen1, new_gen2


def logistic_crossover_2(genome1, genome2): # this one
    logistic1 = genome1.house_assignments.copy()
    logistic2 = genome2.house_assignments.copy()
    
    new_logistic1, new_logistic2 = pillar_crossover(logistic1, logistic2)
    new_gen1 = deepcopy(genome1)
    new_gen1.house_assignments = new_logistic1
    new_gen1.fix_course_assignments()
    new_gen2 = deepcopy(genome2)
    new_gen2.house_assignments = new_logistic2
    new_gen2.fix_course_assignments()
    return new_gen1, new_gen2


