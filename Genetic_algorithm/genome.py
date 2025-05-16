import random
import numpy as np
import Genetic_algorithm.config as config

class Genome:
    def __init__(self):    
        
        self.house_assignments, self.course_assignments = self.generate_random_genome()
        
    # make a getter and a setter for house_assignments and course_assignments
    @property
    def house_assignments(self):
        return self._house_assignments
    @house_assignments.setter
    def house_assignments(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("house_assignments must be a numpy array.")
        self._house_assignments = value
    @property
    def course_assignments(self):
        return self._course_assignments
    @course_assignments.setter
    def course_assignments(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("course_assignments must be a numpy array.")
        self._course_assignments = value
    
    

    def encode(self):
        concatenated = np.concatenate((self.house_assignments, self.course_assignments))
        return concatenated

    def decode(self):
        return self.house_assignments, self.course_assignments
    
    def generate_random_genome(cls):
        random_house_assignments = np.repeat(np.arange(config.N_COURSES),np.ceil(config.MIN_N_HOUSES/config.N_COURSES))
        empty_houses_array = np.full(config.N_HOUSES - config.MIN_N_HOUSES, -1)
        random_house_assignments = np.concatenate((random_house_assignments, empty_houses_array)).astype(int)
        np.random.shuffle(random_house_assignments)
        
        course_partecipants = np.arange(config.N_PARTICIPANTS)
        empty_spots_array = np.full(config.EMPTY_SPOTS, -1)
        course_partecipants =  np.concatenate((course_partecipants, empty_spots_array))
        
        randdom_course_assignments =[]
        for _ in range(config.N_COURSES):
            np.random.shuffle(course_partecipants)
            randdom_course_assignments.append(course_partecipants.copy())
        
        randdom_course_assignments = np.concatenate(randdom_course_assignments)
        # scure owners
        return random_house_assignments, randdom_course_assignments
    
    def get_course_position(self, course_index):
        if (course_index < 0 or course_index >= config.N_COURSES) and course_index != -1:
            raise IndexError("Index out of range for course assignments.")
        
        start = course_index * config.LEN_COURSE
        end = start + config.LEN_COURSE
        
        return start, end
    
    def get_houses_in_course(self, course_index):
        if (course_index < 0 or course_index >= config.N_COURSES) and course_index != -1:
            raise IndexError("Index out of range for course assignments.")
        
        # in the house assignments, give me the ones that are equal to the course idx
        houses_numbers = np.where(self.house_assignments == course_index) [0]
        # sort the houses numbers
        houses_numbers = np.sort(houses_numbers)
        return houses_numbers
    
    def get_house_position(self, idx):
        if idx < 0 or idx >= config.N_HOUSES:
            raise IndexError("Index out of range for house assignments.")
        
        # get the course of the house
        course_idx = self.house_assignments[idx]
        # get the houses in the course
        houses_in_course = self.get_houses_in_course(course_idx)
        # get the index of the house in the course

        if len(houses_in_course) != 0:
            idx_in_course = np.where(houses_in_course == idx)[0][0]
            #get the course position
            course_start, course_end = self.get_course_position(course_idx)
            # get the start and end of the house in the course
            start = course_start + idx_in_course*config.HOUSE_CAPACITY
            end = start + config.HOUSE_CAPACITY
        
            #check that end is not out of range
            if end > len(self.course_assignments):
                raise IndexError("---get_house_position---Index out of range for course assignments.")
            if end > course_end:
                raise IndexError("---get_house_position---Index out of range for course assignments.")
        
            return start, end
        return -1, -1

    def get_course_participants(self, course_index):
        if (course_index < 0 or course_index >= config.N_COURSES) and course_index != -1:
            raise IndexError("Index out of range for course assignments.")
        
        start, end = self.get_course_position(course_index)
        return self.course_assignments[start:end]
    
    def get_house_partecipants(self, house_index):
        if house_index < 0 or house_index >= config.N_HOUSES:
            raise IndexError("Index out of range for house assignments.")
        if self.house_assignments[house_index] == -1:
           #print("House is empty.")
           return []
        start, end = self.get_house_position(house_index)
        if start == -1 and end == -1:
            return np.array([])
        return self.course_assignments[start:end]
    
    def locate_participant_at_given_course(self, course_index, participant_number):
        
        if participant_number < 0 or participant_number >= config.N_PARTICIPANTS:
            raise IndexError("Index out of range for course assignments.")
        
        course_start, course_end = self.get_course_position(course_index)
        houses_at_course = self.get_houses_in_course(course_index)
        for house in houses_at_course:
            house_start, house_end = self.get_house_position(house)
            if participant_number in self.course_assignments[house_start:house_end]:
                return house
        raise ValueError(f"Participant {participant_number} not found in course{course_index}.")
    
    def swap_house_assigments(self, idx1, idx2):
        if idx1 < 0 or idx1 >= config.N_HOUSES:
            raise IndexError("Index out of range for house assignments.")
        if idx2 < 0 or idx2 >= config.N_HOUSES:
            raise IndexError("Index out of range for house assignments.")
        
        self.house_assignments[idx1], self.house_assignments[idx2] = self.house_assignments[idx2], self.house_assignments[idx1]
        self.secure_all_owner_to_houses()
           
    def swap_course_assignments(self,course, idx1, idx2):
        if idx1 < 0 or idx1 >= config.LEN_COURSE:
            raise IndexError("Index out of range for course assignments.")
        if idx2 < 0 or idx2 >= config.LEN_COURSE:
            raise IndexError("Index out of range for course assignments.")
        if course < 0 or course >= config.N_COURSES:
            raise IndexError("Index out of range for course assignments.")
        course_start, _ = self.get_course_position(course)
        
        self.course_assignments[course_start + idx1], self.course_assignments[course_start +idx2] = self.course_assignments[course_start + idx2], self.course_assignments[course_start + idx1]
        
        
    def secure_single_owner_to_house(self, house_number):
        
        if house_number < 0 or house_number >= config.N_HOUSES:
            raise IndexError("Index out of range for house assignments.")
        course_index = self.house_assignments[house_number]
        
        owner_location = self.locate_participant_at_given_course(course_index, house_number)
        house_start, house_end = self.get_house_position(house_number)
        
        if owner_location not in self.course_assignments[house_start:house_end]:
            #get a random number between house start and house end
            random_parecipant_index = random.randint(house_start, house_end-1)
            
            new_owner_location = self.course_assignments[random_parecipant_index]
            new_participant_location = self.course_assignments[owner_location]
            self.course_assignments[owner_location] = new_participant_location
            self.course_assignments[random_parecipant_index] = new_owner_location
            
            
            #self.course_assignments[owner_location], self.course_assignments[random_parecipant_index] = self.course_assignments[random_parecipant_index], self.course_assignments[owner_location]
        
    def secure_all_owner_to_houses(self):
        """
        Secure the owners to their houses in the genome.
        
        This method iterates through all the houses in the genome and checks if the owner of each house is present in the house.
        If the owner is not present, it calls the get_owner_to_house method to swap the owner with a participant in the course.
        The method ensures that each house has its owner present in the genome.
        
        Args:
            int: The number of houses in the genome.
        
        Returns:
            None
        """
        
        for owner in range(config.N_HOUSES):
            if self.house_assignments[owner] != -1:
                self.secure_single_owner_to_house(owner)
        
    def __str__(self):
        return ''.join(map(str, self.encode()))
    
    def semantic_key(self):
        """
        Generate a semantic key for the genome.
        
        The semantic key is a string representation of the genome, where each house assignment is represented by its index.
        The course assignments are represented by their indices as well.
        
        Returns:
            str: The semantic key of the genome.
        """
        hased_houses = self.house_assignments.copy()
        hashed_courses = self.course_assignments.copy()
        for house_start in range(int(len(hased_houses)/config.HOUSE_CAPACITY)):
            hashed_courses[house_start:house_start+config.HOUSE_CAPACITY] = np.sort(hashed_courses[house_start:house_start+config.HOUSE_CAPACITY])
        hased_genome = np.concatenate((hased_houses, hashed_courses))
        arr_bytes = hased_genome.tobytes()
        return str(hash(arr_bytes))
    
    def get_partecipant_itinerary(self, participant_index):
        """
        Get the itinerary of a participant in the genome.
        
        The itinerary is a list of tuples, where each tuple contains the course index and the house index for each course.
        
        Args:
            participant_index (int): The index of the participant.
        
        Returns:
            list: A list of tuples representing the itinerary of the participant.
        """
        if participant_index < 0 or participant_index >= config.N_PARTICIPANTS:
            raise IndexError("Index out of range for course assignments.")
        
        itinerary = []
        for course_index in range(config.N_COURSES):
            house_index = self.locate_participant_at_given_course(course_index, participant_index)
            if house_index != -1:
                itinerary.append((course_index, house_index))
        return itinerary
            