from copy import deepcopy
import numpy as np

def logistic_crossover(genome1, genome2):
    logistic1 = genome1.house_assignments.copy()
    logistic2 = genome2.house_assignments.copy()

    new_gen1 = deepcopy(genome1)
    new_gen1.house_assignments = logistic2
    new_gen1.secure_all_owner_to_houses()
    new_gen2 = deepcopy(genome2)
    new_gen2.house_assignments = logistic1
    new_gen2.secure_all_owner_to_houses()
    
    
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

def social_crossover(genome1, genome2):
    
    
    logistic1 = genome1.course_assignments.copy()
    logistic2 = genome2.course_assignments.copy()

    offspring1, offspring2 = pillar_crossover(logistic1, logistic2)
    new_gen1 = deepcopy(genome1)
    new_gen2 = deepcopy(genome2)
    new_gen1.course_assignments = offspring1
    new_gen2.course_assignments = offspring2
    new_gen1.secure_all_owner_to_houses()
    new_gen2.secure_all_owner_to_houses()
    return new_gen1, new_gen2


