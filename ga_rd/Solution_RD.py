# ga_rd/Solution_RD.py
import numpy as np
from Classes.Solution import Solution
from .repair import _repair_houses, _repair_participants

class SolutionRD(Solution):
    """
    Domain model for Running Dinner:
      - first n_houses genes:  which course  0/1/2 or -1
      - then 3 blocks of (n_participants+empty_spots) each, a permuted mix
    """

    def __init__(
        self,
        participant_homes:   np.ndarray,
        house_coords:        np.ndarray,
        host_idxs:          list[int],
        capacity_of_houses:  int,
        a:                   float = 1.0
    ):
        super().__init__()
        # guest homes = one per participant
        self.participant_homes   = participant_homes
        # course locations = hosts
        self.house_coords        = house_coords
        self.host_idxs         = host_idxs

        # how many diners?
        self.n_participants      = participant_homes.shape[0]
        self.capacity_of_houses  = capacity_of_houses
        self.a                   = a

        self.n_courses           = 3
        self.houses_per_course   = int(
            np.ceil(self.n_participants / capacity_of_houses)
        )
        self.min_n_houses        = self.houses_per_course * self.n_courses
        self.n_houses            = self.min_n_houses + 1
        self.empty_houses        = self.n_houses - self.min_n_houses
        self.empty_spots         = (
            self.houses_per_course * capacity_of_houses
            - self.n_participants
        )

        # initial genome
        self.generate_genome()

    @property
    def fitness(self) -> float:
        m = self.n_courses * (self.capacity_of_houses - 1)
        n = self._compute_avg_meetings()
        d = self._compute_total_distance()
        return (m / n) * self.a * d

    def generate_genome(self) -> None:
        occ   = np.repeat(np.arange(self.n_courses), self.houses_per_course)
        empt  = np.full(self.empty_houses, -1, dtype=int)
        houses = np.concatenate((occ, empt))
        np.random.shuffle(houses)

        genome = houses.copy()
        participants = np.arange(self.n_participants)
        empt_spots   = np.full(self.empty_spots, -1, dtype=int)
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

    def get_house_position(self, house_index: int) -> tuple[int,int]:
        course_idx = self.genome[house_index]
        if course_idx == -1:
            return -1, -1

        # all house-slots for this course, in genome order
        slots = np.where(self.genome[:self.n_houses] == course_idx)[0]
        # find which position within that list corresponds to our house_index
        slot_idx = int(np.where(slots == house_index)[0][0])

        start_pos = self.get_course_position(course_idx) \
                    + slot_idx * self.capacity_of_houses
        return course_idx, start_pos


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
                owner_id = self.host_idxs[h]    # real participant‐ID
                if owner_id not in members:
                    if verbose:
                        print(f"Owner {owner_id} missing from house {h}")
                    return False
        # 4) No duplicates in houses
        return True
    
    def repair_houses(self):
        """Ensure the first n_houses entries have exactly the right counts."""
        self.genome = _repair_houses(self.genome, self)

    def repair_participants(self):
        """Ensure each course‐block has exactly one of each guest and the right -1s."""
        self.genome = _repair_participants(self.genome, self)
        
    def secure_owner_to_houses(self):
        """
        Ensure that for each house‐slot h (0..n_houses-1), the true host
        participant `self.host_idxs[h]` is seated in the capacity‐sized block
        for that house.  If not, swap the owner in, evicting an empty seat
        if possible, else a random other guest.
        """
        for h in range(self.n_houses):
            course_idx = self.genome[h]
            if course_idx == -1:
                # this “house” is empty — no host needed
                continue

            owner = self.host_idxs[h]

            # 1) Find all the house‐slots that host this course
            course_houses = np.where(self.genome[:self.n_houses] == course_idx)[0]
            # 2) Which index within that list is our house h?
            slot_in_course = int(np.where(course_houses == h)[0][0])

            # 3) Locate the start of the seats‐block for this course‐house
            block_start = (
                self.get_course_position(course_idx)
                + slot_in_course * self.capacity_of_houses
            )
            block = self.genome[block_start : block_start + self.capacity_of_houses]

            # 4) If owner already in the block, we’re good
            if owner in block:
                continue

            # 5) Otherwise find where the owner *currently* sits in this course
            full_start = self.get_course_position(course_idx)
            full_len   = len(course_houses) * self.capacity_of_houses
            full_block = self.genome[full_start : full_start + full_len]
            owner_pos_in_full = int(np.where(full_block == owner)[0][0])
            owner_abs_idx     = full_start + owner_pos_in_full

            # 6) Pick a seat to replace: prefer an empty spot if any
            empties = np.where(block == -1)[0]
            if len(empties) > 0:
                replace_offset = int(empties[0])
            else:
                replace_offset = np.random.randint(self.capacity_of_houses)
            victim_abs_idx = block_start + replace_offset

            # 7) Swap them
            self.genome[victim_abs_idx], self.genome[owner_abs_idx] = (
                self.genome[owner_abs_idx],
                self.genome[victim_abs_idx]
            )

    def set_genome(self, genome):
        """
        Set the genome, ensuring it's a numpy array
        """
        self.genome = np.array(genome)
        
    def get_genome(self):
        """
        Return a copy of the genome
        """
        return self.genome.copy()
