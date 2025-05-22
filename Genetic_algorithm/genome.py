import random
import numpy as np
import Genetic_algorithm.config as config


class Genome:
    """A single solution (chromosome) for the running–dinner optimisation.

    A genome is composed of two concatenated integer arrays:

    * **house_assignments** – length = ``N_HOUSES``.  
      Entry *i* tells **which course** (``0 = appetiser``, ``1 = main``,
      ``2 = dessert``) is served in house *i*.  ``-1`` means that the house is
      unused.

    * **course_assignments** – length = ``LEN_COURSE × N_COURSES``.  
      It is split into ``N_COURSES`` blocks.  Each block contains
      ``LEN_COURSE = N_PARTICIPANTS + EMPTY_SPOTS`` seats and is itself divided
      into contiguous sub‑blocks of ``HOUSE_CAPACITY`` seats (one sub‑block per
      house serving that course).  The value in a seat is the **participant
      id** or ``-1`` for an empty seat.

    Three *hard* constraints must always hold – both in randomly generated
    individuals and after every variation operator (mutation, crossover, …):

    1. **Every participant sits *exactly once* in each course.**
    2. **The host of every house is seated in their own house.**  (Participant
       *i* is the host of house *i*.)
    3. **Every course block contains precisely ``EMPTY_SPOTS`` empty seats.**

    The class provides a *repair loop* that enforces those constraints, and all
    public methods that can break them call the loop automatically.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self):
        self._house_assignments, self._course_assignments = (
            self.generate_random_genome())
        # paranoia – but cheap: be 100 % sure of validity
        self.fix_course_assignments(inplace=True)

    # ------------------------------------------------------------------ #
    # Array properties                                                    #
    # ------------------------------------------------------------------ #
    @property
    def house_assignments(self):
        return np.array(self._house_assignments, dtype=int)

    @house_assignments.setter
    def house_assignments(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("house_assignments must be a numpy array")
        self._house_assignments = value

    @property
    def course_assignments(self):
        return np.array(self._course_assignments, dtype=int)

    @course_assignments.setter
    def course_assignments(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("course_assignments must be a numpy array")
        self._course_assignments = value

    # ------------------------------------------------------------------ #
    # Encoding & decoding                                                 #
    # ------------------------------------------------------------------ #
    def encode(self):
        """Return a flat numpy array suitable for GA operators."""
        return np.concatenate((self.house_assignments, self.course_assignments))

    def decode(self):
        return self.house_assignments, self.course_assignments

    # ------------------------------------------------------------------ #
    # Random genome                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def generate_random_genome(cls):
        """Create **one** random genome *already satisfying* all hard
        constraints."""
        # 1) House assignments ------------------------------------------------
        reps   = config.MIN_N_HOUSES // config.N_COURSES
        rem    = config.MIN_N_HOUSES %  config.N_COURSES
        houses = np.repeat(np.arange(config.N_COURSES), reps)
        if rem:
            houses = np.concatenate((houses, np.arange(rem)))
        houses = np.concatenate((houses,
                                 np.full(config.N_HOUSES - len(houses), -1)))
        np.random.shuffle(houses)

        # 2) Course assignments ----------------------------------------------
        base_pool = np.concatenate((np.arange(config.N_PARTICIPANTS),
                                     np.full(config.EMPTY_SPOTS, -1)))
        courses = []
        for _ in range(config.N_COURSES):
            np.random.shuffle(base_pool)
            courses.append(base_pool.copy())
        courses = np.concatenate(courses)

        # 3) Finalise into a valid genome ------------------------------------
        temp = cls.__new__(cls)
        temp._house_assignments  = houses.astype(int)
        temp._course_assignments = courses.astype(int)
        temp.secure_all_owner_to_houses()
        temp.fix_course_assignments(inplace=True)
        return temp._house_assignments, temp._course_assignments
        return temp._house_assignments, temp._course_assignments

    # ------------------------------------------------------------------ #
    # Positional helpers                                                 #
    # ------------------------------------------------------------------ #
    def get_course_position(self, course_index):
        """Return *(start, end)* indices (Python slice style) of the course
        block inside *course_assignments*."""
        if not 0 <= course_index < config.N_COURSES:
            raise IndexError("Course index out of range")
        start = course_index * config.LEN_COURSE
        return start, start + config.LEN_COURSE

    def get_houses_in_course(self, course_index):
        """`np.array` of houses that host *course_index* (sorted)."""
        return np.sort(np.where(self._house_assignments == course_index)[0])

    def get_house_position(self, house_index):
        """Return *(start, end)* indices of the seat sub‑block corresponding
        to *house_index* inside *course_assignments*.  Returns ``(-1, -1)`` for
        unused houses."""
        course = self._house_assignments[house_index]
        if course == -1:
            return -1, -1
        houses = self.get_houses_in_course(course)
        local  = np.where(houses == house_index)[0][0]
        cs, _  = self.get_course_position(course)
        start  = cs + local * config.HOUSE_CAPACITY
        return start, start + config.HOUSE_CAPACITY

    # ------------------------------------------------------------------ #
    # Validation                                                         #
    # ------------------------------------------------------------------ #
    def _course_valid(self, course_index):
        cs, ce = self.get_course_position(course_index)
        block  = self._course_assignments[cs:ce]

        # empty seats count
        if (block == -1).sum() != config.EMPTY_SPOTS:
            return False

        # duplicates & missing participants
        players = block[block != -1]
        if players.size != config.N_PARTICIPANTS:
            return False
        if np.unique(players).size != players.size:
            return False

        # each host present in its house
        for h in self.get_houses_in_course(course_index):
            hs, he = self.get_house_position(h)
            if not (self._course_assignments[hs:he] == h).any():
                return False
        return True

    def is_valid(self):
        return all(self._course_valid(c) for c in range(config.N_COURSES))

    # ------------------------------------------------------------------ #
    # Repair helpers                                                     #
    # ------------------------------------------------------------------ #
    def _fix_hosts(self):
        """Ensure every host sits in their own house (one pass)."""
        for h in range(config.N_HOUSES):
            course = self._house_assignments[h]
            if course == -1:
                continue  # unused house
            hs, he = self.get_house_position(h)
            if (self._course_assignments[hs:he] == h).any():
                continue  # already seated

            # 1) Replace an empty seat in that house if possible -------------
            empties = np.where(self._course_assignments[hs:he] == -1)[0]
            if len(empties):
                self._course_assignments[hs + empties[0]] = h
                continue

            # 2) Re‑use a duplicate somewhere else in the course -------------
            cs, ce = self.get_course_position(course)
            block  = self._course_assignments[cs:ce]
            vals, counts = np.unique(block[block != -1], return_counts=True)
            dupes = vals[counts > 1]
            if len(dupes):
                idx = np.where(block == dupes[0])[0][-1]
                self._course_assignments[cs + idx] = h
                continue

            # 3) Last resort: steal a random seat in own house ---------------
            swap_idx = random.randint(hs, he - 1)
            self._course_assignments[swap_idx] = h

    def _deduplicate_and_fill(self):
        """For every course block remove duplicates and insert missing
        participants (tries not to disturb hosts)."""
        full_set = set(range(config.N_PARTICIPANTS))
        for c in range(config.N_COURSES):
            cs, ce = self.get_course_position(c)
            block  = self._course_assignments[cs:ce]

            # --- 1) remove extras ------------------------------------------
            players = block[block != -1]
            uniq, counts = np.unique(players, return_counts=True)
            for val, cnt in zip(uniq, counts):
                if cnt > 1:
                    # keep first occurrence only
                    idxs = np.where(block == val)[0][1:]
                    block[idxs] = -1

            # --- 2) add missing participants -------------------------------
            present  = set(block[block != -1])
            missing  = list(full_set - present)
            empties  = np.where(block == -1)[0]
            for seat, p in zip(empties, missing):
                block[seat] = p

            self._course_assignments[cs:ce] = block

    def _ensure_empty_seats(self):
        """Adjust every course block to contain exactly `EMPTY_SPOTS` seats
        marked with -1 while keeping hosts seated."""
        for c in range(config.N_COURSES):
            cs, ce = self.get_course_position(c)
            block  = self._course_assignments[cs:ce]
            diff   = (block == -1).sum() - config.EMPTY_SPOTS
            if diff == 0:
                continue  # already fine

            elif diff > 0:  # too many empties -> fill some
                present = set(block[block != -1])
                addable = list(set(range(config.N_PARTICIPANTS)) - present)
                empties = np.where(block == -1)[0]
                for i in range(diff):
                    block[empties[i]] = addable[i % len(addable)]

            else:  # diff < 0 -> need more empties
                need   = -diff
                idxs   = list(range(len(block)))
                random.shuffle(idxs)
                for local in idxs:
                    val = block[local]
                    if val == -1:
                        continue
                    # do not remove a host from their house
                    house_offset = local // config.HOUSE_CAPACITY
                    house        = self.get_houses_in_course(c)[house_offset]
                    if val == house:
                        continue
                    block[local] = -1
                    need -= 1
                    if need == 0:
                        break

            self._course_assignments[cs:ce] = block

    # ------------------------------------------------------------------ #
    # Public repair helpers                                              #
    # ------------------------------------------------------------------ #
    def secure_all_owner_to_houses(self):
        """*Public helper* used by external code and by the random‑genome
        factory: ensure every host is seated once in their own house (single
        pass).  This does *not* guarantee overall validity – it is meant to be
        followed by :py:meth:`fix_course_assignments`."""
        self._fix_hosts()

    # ------------------------------------------------------------------ #
    # Public repair                                                      #
    # ------------------------------------------------------------------ #
    def _rebuild_course_assignments(self):
        """Completely regenerate *course_assignments* so that it is *guaranteed*
        to satisfy all hard constraints, given the current
        *house_assignments*.  Used as a fallback when the incremental repair
        loop stalls (should be extremely rare)."""
        for course in range(config.N_COURSES):
            cs, ce = self.get_course_position(course)
            block  = np.full(config.LEN_COURSE, -1, dtype=int)

            # --- place hosts in their houses -----------------------------
            houses = self.get_houses_in_course(course)
            for i, house in enumerate(houses):
                start = i * config.HOUSE_CAPACITY
                block[start] = house  # first seat of the house sub‑block

            # --- fill remaining seats with a permutation -----------------
            hosts_set = set(houses)
            participants = [p for p in range(config.N_PARTICIPANTS)
                            if p not in hosts_set]
            random.shuffle(participants)
            seat_ptr = 0
            for seat_idx in range(config.LEN_COURSE):
                if block[seat_idx] != -1:
                    continue  # already host
                if seat_ptr < len(participants):
                    block[seat_idx] = participants[seat_ptr]
                    seat_ptr += 1
                # else leave as -1 (empty seat) – by construction we hit
                # exactly EMPTY_SPOTS empties when done

            self._course_assignments[cs:ce] = block
        assert self.is_valid(), "Rebuild failed to produce a valid genome"

    # ------------------------------------------------------------------ #
    # Public repair                                                      #
    # ------------------------------------------------------------------ #
    def fix_course_assignments(self, *, inplace=False):
        """Repair *course_assignments* so that **all hard constraints hold**.

        The function keeps iterating the deterministic repair passes until the
        genome becomes valid.  In the extremely unlikely event that the
        incremental loop makes no further progress, the whole seating plan is
        rebuilt from scratch – *guaranteeing* success and preventing the GA
        from ever crashing.
        """
        if not inplace:
            genome = Genome.__new__(Genome)
            genome._house_assignments  = self._house_assignments.copy()
            genome._course_assignments = self._course_assignments.copy()
        else:
            genome = self

        max_passes_without_progress = 10  # safety – usually 1–2 are enough
        for _ in range(max_passes_without_progress):
            if genome.is_valid():
                return genome
            before = genome._course_assignments.copy()
            genome._fix_hosts()
            genome._deduplicate_and_fill()
            genome._fix_hosts()
            genome._ensure_empty_seats()
            if genome.is_valid():
                return genome
            if np.array_equal(before, genome._course_assignments):
                # no progress – bail out to full rebuild
                break

        # Fallback: build a fresh valid seating from scratch ----------------
        genome._rebuild_course_assignments()
        return genome
    
    # ------------------------------------------------------------------ #
    # GA‑level utilities                                                 #
    # ------------------------------------------------------------------ #
    def get_house_partecipants(self, house_index):
        """Return an array of participant ids seated in *house_index* for the
        course that house hosts.  (Misspelling kept for backward
        compatibility.)  If the house is unused the array is empty."""
        if not 0 <= house_index < config.N_HOUSES:
            raise IndexError("House index out of range")
        course = self._house_assignments[house_index]
        if course == -1:
            return np.array([], dtype=int)
        hs, he = self.get_house_position(house_index)
        block = self._course_assignments[hs:he]
        return block[block != -1]

    # official spelling alias (optional)
    get_house_participants = get_house_partecipants

    def get_course_participants(self, course):
        """Return a 1‑D `np.ndarray` of participant ids (no -1) seated in the
        *whole* course block ``course``.  Empty seats are filtered out, order
        is preserved."""
        cs, ce = self.get_course_position(course)
        block = self._course_assignments[cs:ce]
        return block[block != -1]

    # ------------------------------------------------------------------ #
    # GA‑level utilities                                                 #
    # ------------------------------------------------------------------ #
    def locate_participant_at_given_course(self, course_index, participant):
        cs, _ = self.get_course_position(course_index)
        houses = self.get_houses_in_course(course_index)
        for h in houses:
            hs, he = self.get_house_position(h)
            if participant in self._course_assignments[hs:he]:
                return h
        raise ValueError(f"Participant {participant} not found in course {course_index}")

    def swap_house_assigments(self, idx1, idx2):
        """Swap the *course* hosted by two houses and repair."""
        self._house_assignments[idx1], self._house_assignments[idx2] = (
            self._house_assignments[idx2], self._house_assignments[idx1])
        # changing hosts/course assignments may invalidate seating
        self.fix_course_assignments(inplace=True)

    def swap_course_assignments(self, course, idx1, idx2):
        """Swap two seats *within the same course block* (cheap)."""
        cs, _ = self.get_course_position(course)
        ca = self._course_assignments
        ca[cs + idx1], ca[cs + idx2] = ca[cs + idx2], ca[cs + idx1]
        self._course_assignments = ca
        # local swap can only break duplicates / host seating in that block
        self.fix_course_assignments(inplace=True)

    # ------------------------------------------------------------------ #
    # Misc.                                                               #
    # ------------------------------------------------------------------ #
    def __str__(self):
        return ''.join(map(str, self.encode()))

    def semantic_key(self):
        """Return a hash that is *invariant* to the within‑house seat order –
        handy for duplicate detection in the GA population."""
        h = self.house_assignments.copy()
        c = self.course_assignments.copy()
        # sort seats inside every house sub‑block
        for course in range(config.N_COURSES):
            houses = self.get_houses_in_course(course)
            for i, hidx in enumerate(houses):
                start, end = self.get_house_position(hidx)
                c[start:end] = np.sort(c[start:end])
        return str(hash(np.concatenate((h, c)).tobytes()))

    def get_partecipant_itinerary(self, participant):  # typo kept for API
        if not 0 <= participant < config.N_PARTICIPANTS:
            raise IndexError("Participant index out of range")
        itinerary = [(-1, -1)] * config.N_COURSES
        for c in range(config.N_COURSES):
            try:
                house = self.locate_participant_at_given_course(c, participant)
                itinerary[c] = (c, house)
            except ValueError:
                pass
        return itinerary
