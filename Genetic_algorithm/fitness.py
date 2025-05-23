import numpy as np
import Genetic_algorithm.config as config
#from Genetic_algorithm.genome import Genome
from copy import deepcopy

class ResourceFitness:
    def __init__(self, data_matrix, alpha=1.0, beta=1.0):
        self.data_matrix = data_matrix
        # weights for the two fitness functions
        self.alpha = alpha
        self.beta = beta
        # contants to normalize the scores
        self.max_logistic_score = np.nanmax(data_matrix)*config.N_COURSES
        self.max_social_score = (config.HOUSE_CAPACITY - 1) * config.N_COURSES
        # final fitness score
        self.fitness_score = None
        self._number_of_calls = 0
        
    @property
    def number_of_calls(self):
        return self._number_of_calls

    @number_of_calls.setter
    def number_of_calls(self, value):
        if value < 0:
            raise ValueError("number_of_calls cannot be negative")
        self._number_of_calls = value    
        
        
    def evaluate(self, genome):
        
        social_score = self._calculate_social_fitness(genome)
        logistic_score = self._calculate_logistic_fitness(genome)
        self.number_of_calls += 1
        return self.alpha * social_score + self.beta * logistic_score
    
    
    def partecipation_matrix (self, genome):
        """
        Create a participation matrix for the genome.
        
        The participation matrix is a square matrix of size N_PARTICIPANTS x N_PARTICIPANTS,
        where each element (i, j) indicates whether participant i has met participant j.
        
        Args:
            genome (Genome): The genome object containing the house and course assignments.
            
        Returns:
            numpy.ndarray: The participation matrix of size N_PARTICIPANTS x N_PARTICIPANTS.    

        """
        
        part_mat = np.zeros((config.N_PARTICIPANTS, config.N_PARTICIPANTS), dtype=int)
        for i in range(config.N_HOUSES):
            house = genome.get_house_partecipants(i)
            if len(house) == 0:
                continue
            for j in range(len(house)):
                if house[j] != -1:
                    for k in range(j+1, len(house)):
                        if house[k] != -1 and house[j] > house[k]:
                            part_mat[house[j]][house[k]] = 1


        return part_mat

    def _calculate_social_fitness(self,genome):
        """
        Calculate the social fitness of the genome.
        The social fitness is calculated as the total number of meetings between participants,
        normalized by the number of participants.
        
        The participation matrix is used to count the number of meetings.
        The total number of meetings is the sum of all elements in the participation matrix.
        The social fitness is then calculated as the total number of meetings divided by the number of participants.
        

        Args:
            genome (Genome): The genome object containing the house and course assignments.

        Returns:
            float: The average number of meetings per participant.
        """
        total_meetings = np.nansum(self.partecipation_matrix(genome))
        return total_meetings/config.N_PARTICIPANTS

    def _calculate_logistic_fitness(self, genome) -> float:
        """
        For each partecipant, compute:
          1) home → first course (appetizer)
          2) appetizer → main-course
          3) main-course → dessert

        Sum over all partecipants and invert.
        """
        total_distance = 0.0

        for partecipant in range(config.N_PARTICIPANTS):
            # build their ordered stops
            itinerary = genome.get_partecipant_itinerary(partecipant)
            stops = [h for (_, h) in itinerary if h != -1]
            if not stops:
                continue

            # home → first stop
            a, b = partecipant, stops[0]
            if a > b:
                a, b = b, a
            total_distance += self.data_matrix[a, b]

            # between-course legs
            for prev_house, curr_house in zip(stops, stops[1:]):
                a, b = prev_house, curr_house
                if a > b:
                    a, b = b, a
                total_distance += self.data_matrix[a, b]

        if total_distance <= 0:
            return float("inf")
        return self.max_logistic_score / total_distance