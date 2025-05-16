import numpy as np
HOUSE_CAPACITY = 6
N_COURSES = 3
N_PARTICIPANTS = 10
HOUSES_PER_COURSE = int(np.ceil(N_PARTICIPANTS/HOUSE_CAPACITY))
MIN_N_HOUSES = HOUSES_PER_COURSE * N_COURSES
N_HOUSES = MIN_N_HOUSES + 1
EMPTY_SPOTS = HOUSES_PER_COURSE * HOUSE_CAPACITY - N_PARTICIPANTS
EMPTY_HOUSES = N_HOUSES - MIN_N_HOUSES
LEN_COURSE = N_PARTICIPANTS + EMPTY_SPOTS
#min_n_houses, n_houses, empty_spots 

#n_spot_in_course = int(np.ceil(self.min_n_houses/self.n_courses)*self.capacity_of_houses)


        #self.houses_per_course = int(np.ceil(self.n_partecipants/self.capacity_of_houses))
        #self.min_n_houses = self.houses_per_course * self.n_courses
        #self.n_houses = self.min_n_houses + 1
        #self.empty_spots = self.houses_per_course * self.capacity_of_houses - self.n_partecipants
        #self.empty_houses = self.n_houses - self.min_n_houses