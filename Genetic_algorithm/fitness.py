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

    def _get_distance(self, loc1_idx, loc2_idx) -> float:
            """
            Helper to get distance from self.data_matrix, handling index order and invalid locations.
            Assumes participant_idx can be used as a location_idx for their home.
            Returns float('inf') on error or if locations are invalid.
            """
            if loc1_idx == -1 or loc2_idx == -1:
                # print(f"Warning: Invalid location index (-1) in distance calculation between {loc1_idx} and {loc2_idx}.")
                return 0.0 # Returning 0 for an invalid leg, effectively ignoring it.
                           # Alternatively, could return float('inf') if this path should be heavily penalized.

            # Ensure indices are integers for matrix access
            a, b = int(loc1_idx), int(loc2_idx)

            if a > b: # Ensure consistent indexing if matrix is triangular or symmetric with specific access
                a, b = b, a

            try:
                distance = self.data_matrix[a, b]
                return float(distance) # Ensure it's a float
            except IndexError:
                print(f"Warning: Index out of bounds in distance matrix access for locations {a} (from {loc1_idx}), {b} (from {loc2_idx}). Matrix shape: {getattr(self.data_matrix, 'shape', 'N/A')}")
                return float('inf') 
            except TypeError: 
                print(f"Warning: data_matrix is not properly initialized, not subscriptable, or indices {a}, {b} are of wrong type.")
                return float('inf')

    def _get_raw_total_travel_distance(self, genome) -> float:
            """
            Calculates the sum of travel distances for all participants.
            The path for each participant is: home -> course1 -> course2 -> course3.
            Travel from the last course back to home is NOT included.
            Returns float('inf') if any leg has an infinite distance (error).
            """
            total_distance_for_all = 0.0

            for participant_idx in range(self.config.N_PARTICIPANTS):
                # itinerary is a list of (course_idx, house_idx) tuples, ordered by course.
                itinerary = genome.get_participant_itinerary(participant_idx) 

                # Extract the sequence of houses the participant visits for courses.
                # These are assumed to be in the correct course order (e.g., appetizer, main, dessert).
                stops = [house_idx for (_course_idx, house_idx) in itinerary if house_idx != -1] 

                if not stops: # Participant has no valid course assignments
                    continue

                current_participant_path_distance = 0.0
                # Initial location is home, assumed to be participant_idx
                current_location = participant_idx 

                # Iterate through the sequence of houses visited for courses
                for stop_house_idx in stops:
                    leg_distance = self._get_distance(current_location, stop_house_idx)
                    if leg_distance == float('inf'): # If any leg is problematic, the whole path sum is problematic
                        current_participant_path_distance = float('inf')
                        break 
                    current_participant_path_distance += leg_distance
                    current_location = stop_house_idx # Update location for the next leg

                if current_participant_path_distance == float('inf'):
                    total_distance_for_all = float('inf') # Propagate error
                    break
                total_distance_for_all += current_participant_path_distance

            return total_distance_for_all

    def calculate_average_participant_distance(self, genome) -> float:
            """
            Calculates the average distance a participant has to travel.
            This covers travel from home to the first course, and between subsequent courses.
            Travel from the last course back to home is NOT included in this calculation.
            Returns float('inf') if an error occurs in distance calculation.
            """
            if not hasattr(self.config, 'N_PARTICIPANTS') or self.config.N_PARTICIPANTS == 0:
                return 0.0 # Or np.nan, or raise error, depending on desired behavior for no participants

            raw_total_distance = self._get_raw_total_travel_distance(genome)

            if raw_total_distance == float('inf'):
                return float('inf') # Propagate error indication

            return raw_total_distance / self.config.N_PARTICIPANTS

    def _find_house_for_course(self, participant_itinerary, target_course_idx) -> int:
            """Helper to find which house a participant attends for a specific course."""
            for course_idx, house_idx in participant_itinerary:
                if course_idx == target_course_idx:
                    return house_idx
            return -1 # Participant not assigned to this course or house not found for this course

    def calculate_average_unique_people_met(self, genome) -> float:
            """
            Calculates the average number of unique individuals each participant meets
            across all courses they attend.
            """
            if not hasattr(self.config, 'N_PARTICIPANTS') or self.config.N_PARTICIPANTS == 0:
                return 0.0
            if not hasattr(self.config, 'N_COURSES') or self.config.N_COURSES <= 0:
                print("Warning: N_COURSES not defined or invalid in config. Cannot calculate unique people met.")
                return 0.0

            sum_of_unique_people_met_counts = 0

            # Pre-calculate all itineraries to avoid redundant calls to genome.get_participant_itinerary
            all_itineraries = [genome.get_participant_itinerary(i) for i in range(self.config.N_PARTICIPANTS)]

            for p1_idx in range(self.config.N_PARTICIPANTS):
                people_met_by_p1 = set() # Use a set to store unique participant IDs met
                itinerary_p1 = all_itineraries[p1_idx] 

                if not itinerary_p1: # If p1 has no itinerary, they meet no one.
                    sum_of_unique_people_met_counts += 0
                    continue

                for p2_idx in range(self.config.N_PARTICIPANTS):
                    if p1_idx == p2_idx: # A participant cannot meet themselves
                        continue
                    
                    itinerary_p2 = all_itineraries[p2_idx]
                    if not itinerary_p2: # If p2 has no itinerary, p1 cannot meet them.
                        continue

                    # Check for meetings at each course
                    for course_num in range(self.config.N_COURSES): # Assumes courses are indexed 0, 1, ..., N_COURSES-1
                        house_p1_at_course = self._find_house_for_course(itinerary_p1, course_num)
                        house_p2_at_course = self._find_house_for_course(itinerary_p2, course_num)

                        # If both are at a valid house (-1 means not assigned/invalid) 
                        # AND they are at the same house for this specific course
                        if house_p1_at_course != -1 and house_p1_at_course == house_p2_at_course:
                            people_met_by_p1.add(p2_idx) # Add p2_idx to the set of unique people p1 met

                sum_of_unique_people_met_counts += len(people_met_by_p1)

            return sum_of_unique_people_met_counts / self.config.N_PARTICIPANTS