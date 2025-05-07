# running_dinner_ga/Solution_RD.py

import numpy as np
from Classes.Solution import Solution

class SolutionRD(Solution):
    """
    Domain representation for Running Dinner.
    """
    def __init__(self, house_coords, n_participants, capacity_of_houses, a=1.0):
        super().__init__()
        self.house_coords = house_coords
        self.n_participants = n_participants
        self.capacity_of_houses = capacity_of_houses
        self.a = a
        self.n_courses = 3
        self.houses_per_course = int(np.ceil(n_participants / capacity_of_houses))
        self.min_n_houses = self.houses_per_course * self.n_courses
        self.n_houses = self.min_n_houses + 1
        self.empty_houses = self.n_houses - self.min_n_houses # number of “empty” houses beyond the min
        self.empty_spots = self.houses_per_course * capacity_of_houses - n_participants
        self.generate_genome # build initial random genome

    @property
    def fitness(self):
        m = self.n_courses * (self.capacity_of_houses - 1)
        n = self._compute_avg_meetings()
        d = self._compute_total_distance()
        return (m / n) * self.a * d

    @property
    def generate_genome(self):
        """
        Generate a valid genome:
          - exactly `houses_per_course` houses for each course 0,1,2
          - exactly `empty_houses` slots of -1
          - for each course block: a random permutation of participants + empty spots
        """
        # 1) Houses segment
        #    - `houses_per_course` copies of each course index
        #    - `empty_houses` copies of -1
        occ = np.repeat(np.arange(self.n_courses), self.houses_per_course)
        empt = np.full(self.empty_houses, -1, dtype=int)
        houses = np.concatenate((occ, empt))
        np.random.shuffle(houses)

        # 2) Participant‐blocks segment
        genome = houses.copy()
        participants = np.arange(self.n_participants)
        empt_spots = np.full(self.empty_spots, -1, dtype=int)

        # build one block per course
        for _ in range(self.n_courses):
            block = np.concatenate((participants, empt_spots))
            np.random.shuffle(block)
            genome = np.concatenate((genome, block))

        self.genome = genome

    def _compute_avg_meetings(self):
        total_met = 0
        for p in range(self.n_participants):
            met = set()
            for c in range(self.n_courses):
                owners = self.get_current_course_owners(c)
                for h in owners:
                    members = self.get_house(h)[1]
                    if p in members:
                        met.update(x for x in members if x not in (-1, p))
                        break
            total_met += len(met)
        return total_met / self.n_participants

    def _compute_total_distance(self):
        dist = 0.0
        for p in range(self.n_participants):
            path = []
            for c in range(self.n_courses):
                owners = self.get_current_course_owners(c)
                for h in owners:
                    members = self.get_house(h)[1]
                    if p in members:
                        path.append(h)
                        break
            for i in range(len(path) - 1):
                c0 = self.house_coords[path[i]]
                c1 = self.house_coords[path[i+1]]
                dist += np.linalg.norm(c0 - c1)
        return dist

    # --- Constraint and helper methods ---
    def get_course_position(self, course_index):
        if course_index < 0 or course_index >= self.n_courses:
            return -1
        return self.n_houses + course_index * (self.capacity_of_houses * self.houses_per_course)

    def get_house_position(self, house_index):
        course_idx = self.genome[house_index]
        if course_idx == -1:
            return (-1, [])
        owned = np.sum(self.genome[:house_index] == course_idx) - 1
        offset = self.get_course_position(course_idx)
        return (course_idx, offset + owned * self.capacity_of_houses)

    def get_house(self, house_index):
        ci, pos = self.get_house_position(house_index)
        if ci == -1:
            return (ci, [])
        return (ci, self.genome[pos:pos + self.capacity_of_houses])

    def get_course(self, course_index):
        house_idxs = np.where(self.genome[:self.n_houses] == course_index)[0]
        participants = []
        for h in house_idxs:
            participants.extend(self.get_house(h)[1])
        return (house_idxs, np.array(participants))

    def get_current_course_owners(self, course_index):
        return self.get_course(course_index)[0]

    def genome_swap(self, i, j):
        # allow swap only if BOTH indices are in the same segment
        same_segment = (i < self.n_houses and j < self.n_houses) \
                    or (i >= self.n_houses and j >= self.n_houses)
        if not same_segment:
            raise ValueError("Swap indices must be in same genome segment")
        # now perform the swap
        self.genome[i], self.genome[j] = self.genome[j], self.genome[i]

    def check_validity_of_genome(self, verbose=False):
        # implement all your checks; return True if valid else False
        # (omitted for brevity)
        return True

# copy over your get_course_position, get_house_position,
# get_house, get_course, get_current_course_owners, genome_swap,
# and check_validity_of_genome methods from your existing code.
