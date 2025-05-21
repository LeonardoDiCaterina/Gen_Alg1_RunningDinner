import random
import numpy as np
import Genetic_algorithm.config as config

class Genome:
    def __init__(self):
        # generate and secure a valid random genome
        self._house_assignments, self._course_assignments = self.generate_random_genome()

    @property
    def house_assignments(self):
        return np.array(self._house_assignments, dtype=int)

    @house_assignments.setter
    def house_assignments(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("house_assignments must be a numpy array.")
        self._house_assignments = value

    @property
    def course_assignments(self):
        return np.array(self._course_assignments, dtype=int)

    @course_assignments.setter
    def course_assignments(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("course_assignments must be a numpy array.")
        self._course_assignments = value

    def encode(self):
        return np.concatenate((self.house_assignments, self.course_assignments))

    def decode(self):
        return self.house_assignments, self.course_assignments

    @classmethod
    def generate_random_genome(cls):
        # 1) random house assignments (exactly MIN_N_HOUSES courses then empty)
        counts = config.MIN_N_HOUSES // config.N_COURSES
        remainder = config.MIN_N_HOUSES % config.N_COURSES
        base = np.repeat(
            np.arange(config.N_COURSES),
            counts
        )
        if remainder > 0:
            base = np.concatenate((base, np.arange(remainder)))
        empty_h = np.full(config.N_HOUSES - config.MIN_N_HOUSES, -1)
        houses = np.concatenate((base, empty_h)).astype(int)
        np.random.shuffle(houses)

        # 2) random course assignments
        pool = np.concatenate((
            np.arange(config.N_PARTICIPANTS),
            np.full(config.EMPTY_SPOTS, -1)
        ))
        courses = []
        for _ in range(config.N_COURSES):
            np.random.shuffle(pool)
            courses.append(pool.copy())
        courses = np.concatenate(courses)

        # 3) secure hosts into their blocks
        temp = cls.__new__(cls)
        temp._house_assignments  = houses.copy()
        temp._course_assignments = courses.copy()
        temp.secure_all_owner_to_houses()
        return temp._house_assignments, temp._course_assignments

    def get_course_position(self, course_index):
        if course_index < 0 or course_index >= config.N_COURSES:
            raise IndexError("Course index out of range.")
        start = course_index * config.LEN_COURSE
        return start, start + config.LEN_COURSE

    def get_houses_in_course(self, course_index):
        if course_index < 0 or course_index >= config.N_COURSES:
            raise IndexError("Course index out of range.")
        return np.sort(np.where(self.house_assignments == course_index)[0])

    def get_house_position(self, house_index):
        if house_index < 0 or house_index >= config.N_HOUSES:
            raise IndexError("House index out of range.")
        course = self.house_assignments[house_index]
        if course == -1:
            return -1, -1
        houses = self.get_houses_in_course(course)
        idxs = np.where(houses == house_index)[0]
        if idxs.size == 0:
            return -1, -1
        start, _ = self.get_course_position(course)
        start += idxs[0] * config.HOUSE_CAPACITY
        return start, start + config.HOUSE_CAPACITY

    def get_course_participants(self, course_index):
        start, end = self.get_course_position(course_index)
        return self.course_assignments[start:end]

    def get_house_partecipants(self, house_index):
        # preserve original typo
        if house_index < 0 or house_index >= config.N_HOUSES:
            raise IndexError("House index out of range.")
        course = self.house_assignments[house_index]
        if course == -1:
            return np.array([], dtype=int)
        start, end = self.get_house_position(house_index)
        if start == -1:
            return np.array([], dtype=int)
        return self.course_assignments[start:end]

    def locate_participant_at_given_course(self, course_index, participant_number):
        if participant_number < 0 or participant_number >= config.N_PARTICIPANTS:
            raise IndexError("Participant index out of range.")
        course_start, course_end = self.get_course_position(course_index)
        houses = self.get_houses_in_course(course_index)
        for h in houses:
            hs, he = self.get_house_position(h)
            if participant_number in self.course_assignments[hs:he]:
                return h
        raise ValueError(f"Participant {participant_number} not found in course {course_index}.")

    def swap_house_assigments(self, idx1, idx2):
        if idx1 < 0 or idx1 >= config.N_HOUSES or idx2 < 0 or idx2 >= config.N_HOUSES:
            raise IndexError("House index out of range.")
        self.house_assignments[idx1], self.house_assignments[idx2] = (
            self.house_assignments[idx2], self.house_assignments[idx1]
        )
        self.secure_all_owner_to_houses()

    def swap_course_assignments(self, course, idx1, idx2):
        if idx1 < 0 or idx1 >= config.LEN_COURSE or idx2 < 0 or idx2 >= config.LEN_COURSE:
            raise IndexError("Course seat index out of range.")
        if course < 0 or course >= config.N_COURSES:
            raise IndexError("Course index out of range.")
        start, _ = self.get_course_position(course)
        ra = self.course_assignments
        ra[start + idx1], ra[start + idx2] = ra[start + idx2], ra[start + idx1]
        self._course_assignments = ra

    def secure_single_owner_to_house(self, house_number):
        if house_number < 0 or house_number >= config.N_HOUSES:
            raise IndexError("House index out of range.")
        course = self.house_assignments[house_number]
        if course == -1:
            return
        start, end = self.get_house_position(house_number)
        owner = house_number
        positions = np.where(self._course_assignments == owner)[0]
        if not any(start <= p < end for p in positions):
            if positions.size == 0:
                return
            p0 = positions[0]
            swap = random.randint(start, end - 1)
            ca = self._course_assignments
            ca[p0], ca[swap] = ca[swap], ca[p0]
            self._course_assignments = ca

    def secure_all_owner_to_houses(self):
        for h in range(config.N_HOUSES):
            if self._house_assignments[h] != -1:
                self.secure_single_owner_to_house(h)

    def __str__(self):
        return ''.join(map(str, self.encode()))

    def semantic_key(self):
        hased_h = self.house_assignments.copy()
        hased_c = self.course_assignments.copy()
        for i in range(int(len(hased_h) / config.HOUSE_CAPACITY)):
            start = i * config.HOUSE_CAPACITY
            hased_c[start:start + config.HOUSE_CAPACITY] = np.sort(
                hased_c[start:start + config.HOUSE_CAPACITY]
            )
        arrb = np.concatenate((hased_h, hased_c)).tobytes()
        return str(hash(arrb))

    def get_partecipant_itinerary(self, participant_index):
        """
        Get the itinerary of a participant in the genome as a fixed‐length
        list of (course_index, house_index) tuples—one per course.
        If the participant isn’t at a course, that entry is (-1, -1).
        """
        if participant_index < 0 or participant_index >= config.N_PARTICIPANTS:
            raise IndexError("Participant index out of range.")

        # 1) Pre-fill with “missing” markers so length == N_COURSES
        itinerary = [(-1, -1) for _ in range(config.N_COURSES)]

        # 2) For each course, try to find which house they sit at
        for course_index in range(config.N_COURSES):
            try:
                house_index = self.locate_participant_at_given_course(
                    course_index, participant_index
                )
            except ValueError:
                # not seated in this course
                continue
            itinerary[course_index] = (course_index, house_index)

        # 3) Return a list of length N_COURSES
        return itinerary
