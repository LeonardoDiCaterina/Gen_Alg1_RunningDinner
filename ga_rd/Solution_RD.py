# ga_rd/Solution_RD.py

import numpy as np
from Classes.Solution import Solution

class SolutionRD(Solution):
    """
    Domain representation for the Running Dinner problem.

    Genome layout:
      - First n_houses entries: which course each house hosts
          0 = starter, 1 = main, 2 = dessert, -1 = empty house
      - Then n_courses blocks, each of length (n_participants + empty_spots):
          a permutation of participant IDs [0..n_participants-1] plus -1 placeholders
    """

    def __init__(
        self,
        house_coords: np.ndarray,
        participant_homes: np.ndarray,
        n_participants: int,
        capacity_of_houses: int,
        a: float = 1.0
    ):
        super().__init__()
        # Coordinates of each house (lat/lon), shape = (n_houses, 2)
        self.house_coords        = house_coords
        # Coordinates of each guest’s home, shape = (n_participants, 2)
        self.participant_homes   = participant_homes

        # Number of guests
        self.n_participants      = n_participants
        # Max guests per house (including the host)
        self.capacity_of_houses  = capacity_of_houses
        # Weight between distance and mixing in f = (m/n)*a*d
        self.a                   = a

        # Fixed problem parameters
        self.n_courses           = 3
        # Houses needed per course (rounded up)
        self.houses_per_course   = int(np.ceil(n_participants / capacity_of_houses))
        # Minimum houses across all courses
        self.min_n_houses        = self.houses_per_course * self.n_courses
        # Total houses (plus one “spare” for remainder)
        self.n_houses            = self.min_n_houses + 1
        # Number of truly empty houses beyond the minimum
        self.empty_houses        = self.n_houses - self.min_n_houses
        # Empty seats per course block
        self.empty_spots         = self.houses_per_course * capacity_of_houses - n_participants

        # Build an initial random genome
        self.generate_genome()

    @property
    def fitness(self) -> float:
        """
        f(x) = (m / n) * a * d
          m = maximum possible distinct meetings per guest = n_courses * (capacity_of_houses - 1)
          n = actual average distinct meetings per guest
          d = total travel distance (home->starter->main->dessert->optional afterparty)
        Lower is better.
        """
        m = self.n_courses * (self.capacity_of_houses - 1)
        n = self._compute_avg_meetings()
        d = self._compute_total_distance()
        return (m / n) * self.a * d

    def generate_genome(self) -> None:
        """
        Populate self.genome with a valid, random assignment:
          1) House segment: exactly `houses_per_course` of each [0,1,2] plus `empty_houses` of -1
          2) For each course, a random shuffle of [0..n_participants-1] + `empty_spots` of -1
        """
        # 1) Houses
        occ   = np.repeat(np.arange(self.n_courses), self.houses_per_course)
        empt  = np.full(self.empty_houses, -1, dtype=int)
        houses= np.concatenate((occ, empt))
        np.random.shuffle(houses)

        # 2) Participant blocks
        genome      = houses.copy()
        participants= np.arange(self.n_participants)
        empt_spots  = np.full(self.empty_spots, -1, dtype=int)

        for _ in range(self.n_courses):
            block = np.concatenate((participants, empt_spots))
            np.random.shuffle(block)
            genome = np.concatenate((genome, block))

        self.genome = genome

    def _compute_avg_meetings(self) -> float:
        """Compute the average number of *distinct* new people each guest meets."""
        total_met = 0
        for p in range(self.n_participants):
            met = set()
            for c in range(self.n_courses):
                # which houses host course c?
                for h in self.get_current_course_owners(c):
                    members = self.get_house(h)[1]
                    if p in members:
                        # add everyone except -1 and themselves
                        met.update(x for x in members if x not in (-1, p))
                        break
            total_met += len(met)
        return total_met / self.n_participants

    def _compute_total_distance(self) -> float:
        """
        Sum Euclidean distances for each guest:
          home -> starter -> main -> dessert -> (optional afterparty)
        """
        dist = 0.0
        for p in range(self.n_participants):
            prev = self.participant_homes[p]
            # Courses
            for c in range(self.n_courses):
                for h in self.get_current_course_owners(c):
                    members = self.get_house(h)[1]
                    if p in members:
                        curr = self.house_coords[h]
                        dist += np.linalg.norm(curr - prev)
                        prev = curr
                        break
            # Afterparty leg if provided
            if hasattr(self, 'afterparty_house'):
                ap = self.afterparty_house
                dist += np.linalg.norm(self.house_coords[ap] - prev)
        return dist

    # --- GENOME HELPERS & VALIDITY CHECK ------------------------------------

    def get_course_position(self, course_index: int) -> int:
        """Index offset in genome where this course’s participants begin."""
        if not (0 <= course_index < self.n_courses):
            return -1
        return self.n_houses + course_index * (self.capacity_of_houses * self.houses_per_course)

    def get_house_position(self, house_index: int):
        """
        Given a house slot, return (course_idx, start_pos) of its participants in the genome.
        course_idx = -1 if empty.
        """
        course_idx = self.genome[house_index]
        if course_idx == -1:
            return -1, []
        owned = np.sum(self.genome[:house_index] == course_idx) - 1
        start = self.get_course_position(course_idx) + owned * self.capacity_of_houses
        return course_idx, start

    def get_house(self, house_index: int):
        """Return (course_idx, members_array) for a given house slot."""
        ci, start = self.get_house_position(house_index)
        if ci == -1:
            return ci, []
        return ci, self.genome[start : start + self.capacity_of_houses]

    def get_course(self, course_index: int):
        """Return (house_indices, flattened_members) for course_index."""
        houses = np.where(self.genome[:self.n_houses] == course_index)[0]
        parts = []
        for h in houses:
            parts.extend(self.get_house(h)[1])
        return houses, np.array(parts)

    def get_current_course_owners(self, course_index: int):
        """Convenience: which houses host the given course."""
        return self.get_course(course_index)[0]

    def genome_swap(self, i: int, j: int):
        """
        Swap two genes only if they are in the same segment:
          - both < n_houses  (house assignments)
          - or both >= n_houses (participant blocks)
        """
        same = (i < self.n_houses and j < self.n_houses) \
            or (i >= self.n_houses and j >= self.n_houses)
        if not same:
            raise ValueError("Swap indices must be in same genome segment")
        self.genome[i], self.genome[j] = self.genome[j], self.genome[i]

    def check_validity_of_genome(self, verbose: bool = False) -> bool:
        """
        Verify all hard constraints:
          - correct count of each course in first segment
          - each course block has correct seat count and no duplicates
          - each full house includes its owner
        """
        # 1) Houses count
        hseg = self.genome[:self.n_houses]
        counts = {c: np.sum(hseg == c) for c in range(self.n_courses)}
        empt   = np.sum(hseg == -1)
        if any(counts[c] != self.houses_per_course for c in counts) or empt != self.empty_houses:
            if verbose:
                print("House‐segment counts wrong:", counts, "empty:", empt)
            return False

        # 2) Course blocks
        for c in range(self.n_courses):
            _, block = self.get_course(c)
            if len(block) != self.n_participants + self.empty_spots:
                if verbose: print(f"Course {c} wrong length {len(block)}")
                return False
            # empties first, then each guest exactly once
            sorted_b = np.sort(block)
            if not np.array_equal(sorted_b[:self.empty_spots],
                                  np.full(self.empty_spots, -1)):
                if verbose: print(f"Course {c} empty spots wrong")
                return False
            if not np.array_equal(sorted_b[self.empty_spots:], np.arange(self.n_participants)):
                if verbose: print(f"Course {c} participants wrong")
                return False

        # 3) Owner in house check
        for h in range(self.n_houses):
            ci, members = self.get_house(h)
            if ci >= 0 and len(members) == self.capacity_of_houses:
                if h not in members:
                    if verbose: print(f"Owner {h} missing from house {h}")
                    return False

        return True
