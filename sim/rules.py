"""
Crystal Champions — Headless Rule Engine (Standard Mode)

This is a pure-Python simulation of the *official* Crystal Champions rules
sourced from WorldWinner's own help centre (Zendesk article 41420330679059)
and the corporate blog posts. It is deliberately **headless** — no graphics,
no DOM, no animation. Its purpose is to give the planner a deterministic,
seeded simulator it can evaluate thousands of games against.

Key rule references
-------------------
- Board: 8×8, 7 colours (green, orange, blue, white, red, purple, yellow)
- Swap: adjacent orthogonally; illegal if the swap creates no new match
  (EXCEPT void / special↔special swaps)
- Match: 3+ in a row/column, no diagonals
- Gravity: gems fall down; new gems fall in from top
- Game length: 180 s
- Levels: 6 levels, score thresholds 500 / 1,475 / 3,075 / 5,450 / 8,750
  Multipliers: ×1.0, ×1.5, ×2.0, ×2.5, ×3.0, ×3.5
  Level-up: board cleared and refilled; unused specials carry over
- No legal move → board refilled (costs time, no score penalty)
- Hint: −50 points (flat, NOT scaled by the level multiplier);
  auto-hint after 15 s idle (also −50)

Scoring (all multiplied by the level multiplier *at swap time*)
---------------------------------------------------------------
Chain table (per cascade **wave**):
  1st=10, 2nd=20, 3rd=30, 4th=40, 5th=60, 6th=80, 7th=110, 8th=160, 9th+=210
  ``chain_points(n)`` answers "what is the chain bonus for wave n?"  Every
  match in the same wave shares that wave's chain bonus.
Match size bonus: 3 gems +0, 4 gems +10, 5+ gems +20
Special detonation bonus (running count within the move):
  1st=20, 2nd=40, 3rd=50, 4th=60, subsequent +10 each
  ``detonation_points(n)`` answers "what does the n-th detonation of this
  move award?" — it is **incremental**, not cumulative.
Starlight detonation: +5 per gem destroyed (× multiplier)
Void swap: 75 for the Void + 20 per normal gem + 60 per special gem destroyed

Special Gems (Standard mode: 3 tiers)
-------------------------------------
Fire      — created by a line of 4 → detonates a 3×3 area around it
Starlight — created by a true L/T (two perpendicular 3-lines sharing exactly
            one cell) → destroys its full row AND column
Void      — created by a line of 5 → SWAP with any gem removes that gem and ALL
            gems of that colour.  Void does not need a match to be swapped.
            Void+Void swap clears the entire board.
            A Void destroyed by another gem's blast wipes its own colour
            (same effect as being swapped).

Special combinations (geometry unpublished for most; we implement best guesses)
-------------------------------------------------------------------------------
Fire+Fire           → 6×6 detonation (documented)
Fire+Starlight      → 3-row + 3-col cross (centred on midpoint)
Starlight+Starlight → full row+col of both gems
Void+Void           → clear entire board (documented)
Void+Special        → convert all gems of Void's colour into that Special,
                      then detonate them

No Rocket Gem / Hypercube / PowerUp mode: Crystal Champions Standard mode has
only the three specials above, so ``'rocket'`` was removed from ``SPECIALS``.

Time model
----------
Every swap costs ``MOVE_TIME_COST`` (1.0 s) of wall clock.  On top of that,
animations cost extra: a level-up refill costs ``LEVEL_UP_TIME_COST`` (1.0 s)
per level gained and a deadlock shuffle costs ``SHUFFLE_TIME_COST`` (1.0 s).
``MoveResult.time_cost`` reports the **total** (base move + animations) so the
planner and the controller can reason about the clock without guessing.

Assumptions / simplifications (documented so they can be corrected)
-------------------------------------------------------------------
1. Gem colours: 7 (g/o/b/w/r/p/y). The rules never state the count; this matches
   Bejeweled Blitz ancestry and our CV analysis.
2. Initial fill: random uniform, re-rolled until no matches exist.
3. Refill: random uniform from the active colour set.
4. The chain counter advances **once per cascade wave**, not once per match.
   (Earlier revisions advanced it per match, which inflated a single cascade to
   ``chain_count=14``.  Wave-based is how the real game's chain lookup behaves.)
5. Special creation priority: L/T (starlight) > 5 (void) > 4 (fire).
   "L/T" means two *perpendicular* matches of exactly 3 sharing one cell.
   If a longer line (4+) is involved the shape is ambiguous and we prefer fire
   / void — a 4+3 L-shape therefore makes a Fire gem, not a Starlight.
6. Special creation position: the swap cell if it belongs to the creating match;
   else the geometric centre / shared cell.
7. A gem with a special still has its base colour and participates in matches.
8. Detonation bonus table is global across the whole move (not per-wave).
9. Starlight's "+5 per gem destroyed" counts every unique cell cleared by
   that starlight's row+col, including other specials (which then detonate).
10. Void swap destroys the target gem, the void gem, and all gems of the
    target's colour. Points: 75 (void) + 20×normals + 60×specials, all ×mult.
11. Level-up: board is cleared and refilled; specials remain in their cells
    (carried over). This may create immediate matches — allowed.
    The level check happens **after** every point of the move (match, detonation,
    combo, void) has been added, so a big void swap can level you up.
12. Time cost of deadlock refill / level-up animation is 1.0 s each (see above).
13. Chain detonations are resolved as a worklist (BFS), so a special caught in
    another special's blast detonates recursively.
14. A cascade wave that clears zero cells is a stall: the loop breaks instead of
    spinning forever, and an absolute wave cap raises RuntimeError.

Usage
-----
    game = Game(seed=42)
    moves = game.legal_moves()
    result = game.apply_swap(*moves[0])
    print(f"Scored {result.points} pts, chain {result.chain_count}")
"""

from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Dict, Any

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
COLOURS = ('g', 'o', 'b', 'w', 'r', 'p', 'y')   # green, orange, blue, white, red, purple, yellow
SPECIALS = ('fire', 'starlight', 'void')        # no rocket: Standard mode only

CHAIN_TABLE = (10, 20, 30, 40, 60, 80, 110, 160, 210)   # 1-based, per cascade wave


def chain_points(k: int) -> int:
    """Chain bonus for cascade wave ``k`` (1-based). Waves beyond the table
    keep the last value (210)."""
    return CHAIN_TABLE[k-1] if 1 <= k <= len(CHAIN_TABLE) else (CHAIN_TABLE[-1] if k > 0 else 0)


DETON_TABLE = (20, 40, 50, 60)   # 1st, 2nd, 3rd, 4th detonation of the move


def detonation_points(n: int) -> int:
    """Incremental bonus for the ``n``-th detonation of one move (1-based).

    n=1 → 20, n=2 → 40, n=3 → 50, n=4 → 60, n>4 → 60 + 10*(n-4).
    The four first detonations therefore total 20+40+50+60 = 170.
    """
    if n <= 0:
        return 0
    if n <= len(DETON_TABLE):
        return DETON_TABLE[n-1]
    return DETON_TABLE[-1] + 10 * (n - len(DETON_TABLE))


LEVEL_MULT = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
LEVEL_THRESH = (500, 1475, 3075, 5450, 8750)

# Wall-clock model (see module docstring "Time model")
MOVE_TIME_COST = 1.0        # every swap
LEVEL_UP_TIME_COST = 1.0    # per level gained (board clear + refill animation)
SHUFFLE_TIME_COST = 1.0     # deadlock shuffle

# Safety valve for the cascade loop
MAX_CASCADE_WAVES = 60

MATCH_SIZE_BONUS = {3: 0, 4: 10}


def size_bonus(sz: int) -> int:
    """Match-size bonus: +0 for 3, +10 for 4, +20 for 5 or more."""
    if sz >= 5:
        return 20
    return MATCH_SIZE_BONUS.get(sz, 0)


DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
@dataclass
class Match:
    cells: List[Tuple[int, int]]
    size: int
    horizontal: bool


@dataclass
class MoveResult:
    points: int
    matches: int                 # number of matches resolved (all waves)
    detonations: int
    chain_count: int             # number of cascade waves
    specials_created: int
    void_swap: bool
    level_ups: int
    time_cost: float             # total seconds consumed: MOVE_TIME_COST + animations
    events: List[Dict[str, Any]]
    board_before: Tuple
    board_after: Tuple
    deadlocked: bool = False     # True when the settled board had to be shuffled


# ----------------------------------------------------------------------
# Game engine
# ----------------------------------------------------------------------
class Game:
    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        level: int = 1,
        score: float = 0.0,
        time_left: float = 180.0,
        n_colours: int = 7,
        rows: int = 8,
        cols: int = 8,
        mode: str = 'standard',
        time_limit: Optional[float] = None,
    ):
        # ``time_limit`` is the legacy spelling of ``time_left``; honour it.
        if time_limit is not None:
            time_left = time_limit
        self.rng = random.Random(seed)
        self.rows, self.cols = rows, cols
        self.palette = COLOURS[:n_colours]
        self.mode = mode
        self.time_left = float(time_left)
        self.score = int(score)
        self.level = max(1, min(int(level), 6))
        self.colour: List[List[Optional[str]]] = [[None]*cols for _ in range(rows)]
        self.special: List[List[Optional[str]]] = [[None]*cols for _ in range(rows)]
        self.hints_used = 0
        self.deadlocks = 0
        self.total_moves = 0
        self._fill_full()

    # -------------------- basic helpers --------------------
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _rand_colour(self) -> str:
        return self.rng.choice(self.palette)

    def _empty_board(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.colour[r][c] = None
                self.special[r][c] = None

    # -------------------- board state serialisation --------------------
    def board_tuple(self) -> Tuple:
        """Immutable snapshot for cloning / comparison."""
        return (
            tuple(tuple(row) for row in self.colour),
            tuple(tuple(row) for row in self.special),
            self.score, self.level, self.time_left
        )

    def restore_board(self, snap: Tuple):
        self.colour = [list(row) for row in snap[0]]
        self.special = [list(row) for row in snap[1]]
        self.score, self.level, self.time_left = snap[2], snap[3], snap[4]

    def clone(self) -> 'Game':
        g = Game.__new__(Game)
        g.rng = random.Random()
        g.rng.setstate(self.rng.getstate())
        g.rows, g.cols = self.rows, self.cols
        g.palette = self.palette
        g.mode = self.mode
        g.time_left = self.time_left
        g.score = self.score
        g.level = self.level
        g.colour = [row[:] for row in self.colour]
        g.special = [row[:] for row in self.special]
        g.hints_used = self.hints_used
        g.deadlocks = self.deadlocks
        g.total_moves = self.total_moves
        return g

    # -------------------- initial fill (no matches) --------------------
    def _has_any_match(self) -> bool:
        return bool(self._find_matches())

    def _fill_full(self):
        """Fill board randomly until it has zero matches."""
        for _attempt in range(1000):
            for r in range(self.rows):
                for c in range(self.cols):
                    self.colour[r][c] = self._rand_colour()
                    self.special[r][c] = None
            if not self._has_any_match():
                return
        raise RuntimeError("Could not generate a match-free board")

    # -------------------- match finding --------------------
    def _find_matches(self) -> List[Match]:
        """Return all horizontal/vertical runs of 3+ same colour.
        Cells with specials still match on their base colour."""
        matches: List[Match] = []
        # Horizontal
        for r in range(self.rows):
            c = 0
            while c < self.cols:
                col = self.colour[r][c]
                if col is None:
                    c += 1
                    continue
                start = c
                while c < self.cols and self.colour[r][c] == col:
                    c += 1
                length = c - start
                if length >= 3:
                    matches.append(Match([(r, cc) for cc in range(start, c)], length, True))
        # Vertical
        for c in range(self.cols):
            r = 0
            while r < self.rows:
                col = self.colour[r][c]
                if col is None:
                    r += 1
                    continue
                start = r
                while r < self.rows and self.colour[r][c] == col:
                    r += 1
                length = r - start
                if length >= 3:
                    matches.append(Match([(rr, c) for rr in range(start, r)], length, False))
        return matches

    def _run_lengths_through(self, r: int, c: int) -> Tuple[int, int]:
        """(horizontal_run, vertical_run) of same-colour gems through (r, c).

        Only the *contiguous* run counts — a hole (None) breaks it.  Used for
        cheap "did this swap create a match" tests.
        """
        col = self.colour[r][c]
        if col is None:
            return 0, 0
        left = c
        while left - 1 >= 0 and self.colour[r][left-1] == col:
            left -= 1
        right = c
        while right + 1 < self.cols and self.colour[r][right+1] == col:
            right += 1
        up = r
        while up - 1 >= 0 and self.colour[up-1][c] == col:
            up -= 1
        down = r
        while down + 1 < self.rows and self.colour[down+1][c] == col:
            down += 1
        return right - left + 1, down - up + 1

    # -------------------- legal move generation --------------------
    def legal_moves(self) -> List[Tuple[int, int, int, int]]:
        """Return all (r1,c1,r2,c2) swaps that are legal.
        Legal if: (a) swap creates >=1 NEW match through one of the swapped
        cells, OR (b) involves a Void gem, OR (c) special↔special swap."""
        moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                for dr, dc in ((0, 1), (1, 0)):  # right, down to avoid dupes
                    nr, nc = r+dr, c+dc
                    if not self.in_bounds(nr, nc):
                        continue
                    if self._swap_is_legal(r, c, nr, nc):
                        moves.append((r, c, nr, nc))
        return moves

    def _swap_is_legal(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """True iff the swap is allowed.

        A normal swap is legal only if it *creates* a match that runs through one
        of the two swapped cells.  Checking "does the board contain any match
        after the swap" is wrong: one pre-existing match on the board made
        107/112 of all swaps look legal.
        """
        s1, s2 = self.special[r1][c1], self.special[r2][c2]
        # Special↔special always legal
        if s1 and s2:
            return True
        # Void swap always legal (even with no match)
        if s1 == 'void' or s2 == 'void':
            return True
        # Normal swap: simulate, then look only at the two swapped cells
        if self.colour[r1][c1] is None or self.colour[r2][c2] is None:
            return False
        # Swapping two identical colours changes nothing, so it cannot create a
        # match — the game rejects it.
        if self.colour[r1][c1] == self.colour[r2][c2]:
            return False
        self.colour[r1][c1], self.colour[r2][c2] = self.colour[r2][c2], self.colour[r1][c1]
        self.special[r1][c1], self.special[r2][c2] = self.special[r2][c2], self.special[r1][c1]
        try:
            legal = False
            for (r, c) in ((r1, c1), (r2, c2)):
                h, v = self._run_lengths_through(r, c)
                if h >= 3 or v >= 3:
                    legal = True
                    break
        finally:
            # revert
            self.colour[r1][c1], self.colour[r2][c2] = self.colour[r2][c2], self.colour[r1][c1]
            self.special[r1][c1], self.special[r2][c2] = self.special[r2][c2], self.special[r1][c1]
        return legal

    # -------------------- void swap resolution --------------------
    def _resolve_void_swap(self, r_void: int, c_void: int, r_target: int,
                           c_target: int) -> Tuple[int, Set[Tuple[int, int]], int, int]:
        """Resolve a void swap. Returns (points, cells_cleared, normals_destroyed, specials_destroyed)."""
        target_colour = self.colour[r_target][c_target]
        void_special = self.special[r_void][c_void]
        if void_special != 'void':
            raise ValueError(f"_resolve_void_swap called without a void at {(r_void, c_void)}")
        # Cells to clear: both swapped cells + all of target_colour
        cleared = {(r_void, c_void), (r_target, c_target)}
        normals = 0
        specials = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.colour[r][c] == target_colour:
                    cleared.add((r, c))
                    if self.special[r][c]:
                        specials += 1
                    else:
                        normals += 1
        # Points: 75 + 20*normals + 60*specials, all × mult
        mult = LEVEL_MULT[min(self.level-1, 5)]
        pts = int((75 + 20*normals + 60*specials) * mult)
        # Clear them
        for r, c in cleared:
            self.colour[r][c] = None
            self.special[r][c] = None
        return pts, cleared, normals, specials

    # -------------------- gravity & refill --------------------
    def _gravity_and_refill(self):
        for c in range(self.cols):
            # compact down
            write = self.rows - 1
            for r in range(self.rows-1, -1, -1):
                if self.colour[r][c] is not None:
                    if r != write:
                        self.colour[write][c] = self.colour[r][c]
                        self.special[write][c] = self.special[r][c]
                        self.colour[r][c] = None
                        self.special[r][c] = None
                    write -= 1
            # refill top
            for r in range(write, -1, -1):
                self.colour[r][c] = self._rand_colour()
                self.special[r][c] = None

    # -------------------- special creation from matches --------------------
    def _create_specials_from_matches(
        self,
        matches: List[Match],
        swap_pos: Optional[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], str]:
        """Determine specials created this wave.

        Priority: L/T (starlight) > 5-line (void) > 4-line (fire).
        Returns dict {(r,c): special_kind}.
        """
        created: Dict[Tuple[int, int], str] = {}

        # --- L/T detection: one cell shared by two PERPENDICULAR 3-lines ---
        per_cell: Dict[Tuple[int, int], List[Match]] = {}
        for m in matches:
            for cell in m.cells:
                per_cell.setdefault(cell, []).append(m)
        for cell, ms in per_cell.items():
            if len(ms) < 2:
                continue
            if len({m.horizontal for m in ms}) < 2:
                continue                      # both horizontal: not an L/T
            if all(m.size == 3 for m in ms):
                created[cell] = 'starlight'
            # else: a 4+ line is involved → ambiguous shape, prefer fire/void.
            # The per-match pass below handles it (4 → fire, 5+ → void).

        # --- 5-line (void) and 4-line (fire) ---
        for m in matches:
            if m.size >= 5:
                kind = 'void'
            elif m.size == 4:
                kind = 'fire'
            else:
                continue
            if all(c in created for c in m.cells):
                continue                      # whole match consumed by a starlight
            pos = self._special_pos(m, swap_pos, created)
            if pos is not None and pos not in created:
                created[pos] = kind
        return created

    def _special_pos(
        self,
        match: Match,
        swap_pos: Optional[Tuple[int, int]],
        taken: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Preferred position for a special gem from this match."""
        taken = taken or {}
        if swap_pos and swap_pos in match.cells and swap_pos not in taken:
            return swap_pos
        idx = len(match.cells) // 2
        for k in (0, idx, -1):
            cell = match.cells[k]
            if cell not in taken:
                return cell
        for cell in match.cells:
            if cell not in taken:
                return cell
        return None

    # -------------------- detonation effects --------------------
    def _detonate(self, r: int, c: int, kind: str) -> Set[Tuple[int, int]]:
        """Return the set of cells affected by detonating the special at (r,c)."""
        affected: Set[Tuple[int, int]] = set()
        if kind == 'fire':
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r+dr, c+dc
                    if self.in_bounds(nr, nc):
                        affected.add((nr, nc))
        elif kind == 'starlight':
            for cc in range(self.cols):
                affected.add((r, cc))
            for rr in range(self.rows):
                affected.add((rr, c))
        elif kind == 'void':
            # A Void caught in another gem's blast wipes its own colour, exactly
            # like a Void swap does.
            affected.add((r, c))
            col = self.colour[r][c]
            if col is not None:
                for rr in range(self.rows):
                    for cc in range(self.cols):
                        if self.colour[rr][cc] == col:
                            affected.add((rr, cc))
        return affected

    def _special_combo(self, specs: List[Tuple[int, int, str]],
                       cleared: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Resolve special↔special combos. specs = list of (r,c,kind) being swapped.
        Returns additional cleared cells from the combo."""
        if len(specs) != 2:
            return set()
        (r1, c1, k1), (r2, c2, k2) = specs
        kinds = tuple(sorted([k1, k2]))
        extra: Set[Tuple[int, int]] = set()

        if kinds == ('fire', 'fire'):
            # documented: 6×6 around the midpoint
            mr, mc = (r1+r2)//2, (c1+c2)//2
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = mr+dr, mc+dc
                    if self.in_bounds(nr, nc):
                        extra.add((nr, nc))
        elif kinds == ('fire', 'starlight'):
            # 3 rows + 3 columns crossing at the midpoint
            mr, mc = (r1+r2)//2, (c1+c2)//2
            for dc in range(-1, 2):
                for rr in range(self.rows):
                    if self.in_bounds(rr, mc+dc):
                        extra.add((rr, mc+dc))
            for dr in range(-1, 2):
                for cc in range(self.cols):
                    if self.in_bounds(mr+dr, cc):
                        extra.add((mr+dr, cc))
        elif kinds == ('starlight', 'starlight'):
            for cc in range(self.cols):
                extra.add((r1, cc))
            for rr in range(self.rows):
                extra.add((rr, c1))
            for cc in range(self.cols):
                extra.add((r2, cc))
            for rr in range(self.rows):
                extra.add((rr, c2))
        elif kinds == ('void', 'void'):
            # documented: entire board
            for r in range(self.rows):
                for c in range(self.cols):
                    extra.add((r, c))
        elif 'void' in kinds:
            # Void+Special → convert all gems of the Void's colour into that
            # special, then detonate them.
            void_idx = 0 if kinds[0] == 'void' else 1
            vr, vc, _ = specs[void_idx]
            other_kind = kinds[1-void_idx]
            target_colour = self.colour[vr][vc]
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.colour[r][c] == target_colour:
                        self.special[r][c] = other_kind
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.special[r][c] == other_kind:
                        extra.update(self._detonate(r, c, other_kind))
        return extra

    # -------------------- cascade resolution --------------------
    def _resolve_cascade(self, swap_pos: Tuple[int, int], is_void_swap: bool) -> MoveResult:
        """Run the full cascade after a swap (or void swap).

        The board must already be *settled* (no holes) when this is called.
        Returns a MoveResult whose ``time_cost`` is 0.0 — ``apply_swap`` adds the
        base move cost, level-up animations and deadlock shuffle.
        """
        mult = LEVEL_MULT[min(self.level-1, 5)]
        total_pts = 0
        waves = 0
        deton_count = 0
        match_count = 0
        specials_created_total = 0
        events: List[Dict[str, Any]] = []

        while True:
            matches = self._find_matches()
            if not matches:
                break

            waves += 1
            if waves > MAX_CASCADE_WAVES:
                raise RuntimeError(
                    f"cascade exceeded {MAX_CASCADE_WAVES} waves at swap {swap_pos} "
                    f"(void_swap={is_void_swap}): board still matches after every wave. "
                    f"Unknown cells="
                    f"{sum(1 for row in self.colour for c in row if c is None)}"
                )

            # Special creation this wave
            created = self._create_specials_from_matches(
                matches, swap_pos if waves == 1 else None)
            specials_created_total += len(created)
            match_count += len(matches)

            # Cells to clear this wave (a cell that spawns a special survives)
            to_clear: Set[Tuple[int, int]] = set()
            for m in matches:
                for cell in m.cells:
                    if cell not in created:
                        to_clear.add(cell)

            # --- chain detonation: worklist so blasts recurse ---
            pending: deque = deque()
            for cell in sorted(to_clear):
                sp = self.special[cell[0]][cell[1]]
                if sp:
                    pending.append((cell[0], cell[1], sp))
            wave_detonations: List[Tuple[int, int, str]] = []
            visited: Set[Tuple[int, int, str]] = set()
            while pending:
                r, c, sp = pending.popleft()
                if (r, c, sp) in visited:
                    continue
                visited.add((r, c, sp))
                if self.special[r][c] != sp:
                    continue                      # stale entry
                wave_detonations.append((r, c, sp))
                for (nr, nc) in self._detonate(r, c, sp):
                    to_clear.add((nr, nc))
                    sp2 = self.special[nr][nc]
                    if sp2 and (nr, nc, sp2) not in visited:
                        pending.append((nr, nc, sp2))

            # Newly created specials are never consumed by a blast in the same wave
            to_clear -= set(created)

            # --- progress guard: a wave that clears nothing would loop forever ---
            if not to_clear:
                events.append({
                    'type': 'cascade_stalled',
                    'wave': waves,
                    'matches': len(matches),
                    'created': sorted(created.items()),
                })
                break

            # --- score the matches: chain advances once per WAVE ---
            wave_chain_pts = chain_points(waves)
            for m in matches:
                base = wave_chain_pts + size_bonus(m.size)
                pts = int(base * mult)
                total_pts += pts
                events.append({
                    'type': 'match',
                    'cells': m.cells,
                    'size': m.size,
                    'chain': waves,
                    'points': pts,
                })

            # --- score the detonations ---
            for r, c, kind in wave_detonations:
                deton_count += 1
                det_pts = int(detonation_points(deton_count) * mult)
                total_pts += det_pts
                events.append({
                    'type': 'detonation',
                    'kind': kind,
                    'pos': (r, c),
                    'detonation_number': deton_count,
                    'points': det_pts,
                })
                if kind == 'starlight':
                    affected = self._detonate(r, c, 'starlight')
                    gems_destroyed = sum(
                        1 for (rr, cc) in affected if self.colour[rr][cc] is not None)
                    sl_pts = int(gems_destroyed * 5 * mult)
                    total_pts += sl_pts
                    events.append({
                        'type': 'starlight_bonus',
                        'gems': gems_destroyed,
                        'points': sl_pts,
                    })

            # --- apply clearing ---
            for r, c in to_clear:
                self.colour[r][c] = None
                self.special[r][c] = None

            # --- apply created specials (they kept their colour) ---
            for (r, c), kind in created.items():
                if self.colour[r][c] is not None:
                    self.special[r][c] = kind

            # --- gravity + refill ---
            self._gravity_and_refill()

        return MoveResult(
            points=total_pts,
            matches=match_count,
            detonations=deton_count,
            chain_count=waves,
            specials_created=specials_created_total,
            void_swap=is_void_swap,
            level_ups=0,
            time_cost=0.0,
            events=events,
            board_before=(),
            board_after=self.board_tuple(),
        )

    # -------------------- level-up --------------------
    def _apply_level_ups(self) -> Tuple[int, float, List[Dict[str, Any]]]:
        """Promote as many levels as ``self.score`` now allows.

        MUST be called after every point of the move has been added to
        ``self.score`` — otherwise a void swap worth 475 on 400 points never
        levels up.
        """
        level_ups = 0
        time_cost = 0.0
        events: List[Dict[str, Any]] = []
        while self.level < 6 and self.score >= LEVEL_THRESH[self.level-1]:
            self.level += 1
            level_ups += 1
            time_cost += LEVEL_UP_TIME_COST
            events.append({
                'type': 'level_up',
                'level': self.level,
                'multiplier': LEVEL_MULT[min(self.level-1, 5)],
                'threshold': LEVEL_THRESH[self.level-2] if self.level >= 2 else 0,
            })
            # Level-up: clear the board but keep the special gems, then refill.
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.special[r][c] is None:
                        self.colour[r][c] = None
            self._gravity_and_refill()
        return level_ups, time_cost, events

    # -------------------- move finalisation --------------------
    def _finish(self, res: MoveResult, before: Tuple) -> MoveResult:
        """Book-keep a resolved move: score, level-ups, deadlock, clock."""
        res.board_before = before
        self.score += res.points

        # Level check happens only now that all points of the move are in.
        level_ups, lu_time, lu_events = self._apply_level_ups()
        res.level_ups = level_ups
        res.events.extend(lu_events)

        cost = MOVE_TIME_COST + lu_time

        # Deadlock: the settled board has no legal move → shuffle, cost time.
        if not self.has_legal_move():
            cost += self.shuffle_board()
            res.deadlocked = True
            res.events.append({
                'type': 'deadlock_shuffle',
                'time_cost': SHUFFLE_TIME_COST,
            })

        res.time_cost = cost
        res.board_after = self.board_tuple()
        self.time_left -= res.time_cost
        return res

    # -------------------- public swap --------------------
    def apply_swap(self, r1: int, c1: int, r2: int, c2: int) -> MoveResult:
        """Execute a swap, resolve cascade, update score/level/time."""
        self.total_moves += 1
        before = self.board_tuple()

        s1, s2 = self.special[r1][c1], self.special[r2][c2]

        # ---- Special↔special swap (includes Void+Void) ----
        if s1 and s2:
            self.special[r1][c1], self.special[r2][c2] = s2, s1
            self.colour[r1][c1], self.colour[r2][c2] = self.colour[r2][c2], self.colour[r1][c1]
            combo_cleared = self._special_combo([(r1, c1, s2), (r2, c2, s1)], set())
            mult = LEVEL_MULT[min(self.level-1, 5)]
            combo_pts = 0
            combo_events: List[Dict[str, Any]] = []
            for (r, c) in combo_cleared:
                if self.in_bounds(r, c) and self.colour[r][c] is not None:
                    had_special = self.special[r][c]
                    self.colour[r][c] = None
                    self.special[r][c] = None
                    if had_special:
                        combo_pts += int(60 * mult)
                        combo_events.append(
                            {'type': 'combo_detonate_special', 'pos': (r, c), 'kind': had_special})
                    else:
                        combo_pts += int(20 * mult)
            combo_events.insert(0, {
                'type': 'special_combo',
                'kinds': sorted([s1, s2]),
                'cells': len(combo_cleared),
                'points': combo_pts,
            })
            # settle the holes left by the combo before looking for matches
            self._gravity_and_refill()
            res = self._resolve_cascade((r1, c1), is_void_swap=False)
            res.points += combo_pts
            res.events = combo_events + res.events
            return self._finish(res, before)

        # ---- Void swap (single void only) ----
        if s1 == 'void' or s2 == 'void':
            vr, vc = (r1, c1) if s1 == 'void' else (r2, c2)
            tr, tc = (r2, c2) if s1 == 'void' else (r1, c1)
            void_pts, cleared, normals, specials = self._resolve_void_swap(vr, vc, tr, tc)
            events: List[Dict[str, Any]] = [{
                'type': 'void_swap',
                'void_pos': (vr, vc),
                'target_pos': (tr, tc),
                'normals': normals,
                'specials': specials,
                'points': void_pts,
            }]
            # settle the holes the void left behind, then cascade
            self._gravity_and_refill()
            res = self._resolve_cascade((vr, vc), is_void_swap=True)
            res.points += void_pts
            res.events = events + res.events
            res.void_swap = True
            return self._finish(res, before)

        # ---- Normal swap ----
        self.colour[r1][c1], self.colour[r2][c2] = self.colour[r2][c2], self.colour[r1][c1]
        self.special[r1][c1], self.special[r2][c2] = self.special[r2][c2], self.special[r1][c1]

        res = self._resolve_cascade((r1, c1), is_void_swap=False)
        return self._finish(res, before)

    # -------------------- deadlock / no moves --------------------
    def has_legal_move(self) -> bool:
        for r in range(self.rows):
            for c in range(self.cols):
                for dr, dc in ((0, 1), (1, 0)):
                    nr, nc = r+dr, c+dc
                    if self.in_bounds(nr, nc) and self._swap_is_legal(r, c, nr, nc):
                        return True
        return False

    def shuffle_board(self) -> float:
        """No legal move → refill entire board. Costs time, no score penalty.

        Returns the time cost (``SHUFFLE_TIME_COST``); ``apply_swap`` folds it
        into ``MoveResult.time_cost``.  Retries until the new board actually has
        a legal move (a match-free random board is not guaranteed to have one).
        """
        self.deadlocks += 1
        for _attempt in range(50):
            self._fill_full()
            if self.has_legal_move():
                return SHUFFLE_TIME_COST
        return SHUFFLE_TIME_COST

    # -------------------- hint --------------------
    def hint(self) -> Optional[Tuple[int, int, int, int]]:
        """Return a legal move (first found) and charge a flat −50 points."""
        moves = self.legal_moves()
        if not moves:
            return None
        self.score -= 50          # spec: flat −50, not scaled by the multiplier
        self.hints_used += 1
        return moves[0]

    # -------------------- level multiplier getter --------------------
    def current_multiplier(self) -> float:
        return LEVEL_MULT[min(self.level-1, 5)]

    def next_threshold(self) -> Optional[int]:
        if self.level >= 6:
            return None
        return LEVEL_THRESH[self.level-1]

    def progress_to_next(self) -> float:
        nxt = self.next_threshold()
        if nxt is None:
            return 1.0
        prev = LEVEL_THRESH[self.level-2] if self.level > 1 else 0
        return (self.score - prev) / (nxt - prev)

    # -------------------- game over check --------------------
    def is_over(self) -> bool:
        return self.time_left <= 0


# ----------------------------------------------------------------------
# Baseline greedy policy (for quick smoke tests)
# ----------------------------------------------------------------------
def greedy_pick(game: Game) -> Optional[Tuple[int, int, int, int]]:
    """Evaluate all legal moves by 1-ply simulation, pick max immediate points."""
    moves = game.legal_moves()
    if not moves:
        return None
    best = None
    best_score = -1
    for m in moves:
        g = game.clone()
        res = g.apply_swap(*m)
        if res.points > best_score:
            best_score = res.points
            best = m
    return best


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _selftest() -> int:
    import threading
    import time

    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}{(' - ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("=== Crystal Champions Rule Engine Self-Test ===")

    # ---- constants that must never drift ----
    print("\n-- frozen tables --")
    check("chain table", CHAIN_TABLE == (10, 20, 30, 40, 60, 80, 110, 160, 210))
    check("thresholds", LEVEL_THRESH == (500, 1475, 3075, 5450, 8750))
    check("multipliers", LEVEL_MULT == (1.0, 1.5, 2.0, 2.5, 3.0, 3.5))
    check("specials (no rocket)", SPECIALS == ('fire', 'starlight', 'void'))

    # ---- BUG 1: detonation bonus is incremental ----
    print("\n-- bug 1: detonation_points is incremental --")
    per = [detonation_points(n) for n in range(1, 5)]
    check("per-detonation bonuses", per == [20, 40, 50, 60], f"{per}")
    total4 = sum(detonation_points(n) for n in range(1, 5))
    check("total for 4 detonations == 170", total4 == 170, f"{total4} (was 360)")
    check("5th detonation == 70", detonation_points(5) == 70, f"{detonation_points(5)}")

    # ---- BUG 3: legality is "this swap created a match" ----
    print("\n-- bug 3: legality on a board with one pre-existing match --")
    g = Game(seed=7)
    for c in range(3):
        g.colour[7][c] = 'r'
    moves = g.legal_moves()
    check("not ~108 legal swaps", len(moves) < 40, f"{len(moves)}/112 legal (was 109)")

    # cross-check the fast path against an explicit before/after match comparison
    def legality_slow(gm: Game, r1: int, c1: int, r2: int, c2: int) -> bool:
        s1, s2 = gm.special[r1][c1], gm.special[r2][c2]
        if s1 and s2:
            return True
        if s1 == 'void' or s2 == 'void':
            return True
        if gm.colour[r1][c1] == gm.colour[r2][c2]:
            return False

        def keys_touching(cells, matches):
            out = set()
            for m in matches:
                if any(x in cells for x in m.cells):
                    out.add((tuple(m.cells), m.horizontal))
            return out

        cells = ((r1, c1), (r2, c2))
        before_keys = keys_touching(cells, gm._find_matches())
        gm.colour[r1][c1], gm.colour[r2][c2] = gm.colour[r2][c2], gm.colour[r1][c1]
        after_keys = keys_touching(cells, gm._find_matches())
        gm.colour[r1][c1], gm.colour[r2][c2] = gm.colour[r2][c2], gm.colour[r1][c1]
        return bool(after_keys - before_keys)

    mismatches = 0
    for seed in (1, 42, 123, 999, 31337):
        gm = Game(seed=seed)
        for r in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (1, 0)):
                    nr, nc = r+dr, c+dc
                    if gm.in_bounds(nr, nc):
                        fast = gm._swap_is_legal(r, c, nr, nc)
                        slow = legality_slow(gm, r, c, nr, nc)
                        if fast != slow:
                            mismatches += 1
    check("fast legality == before/after match comparison (5 seeds x 112 swaps)",
          mismatches == 0, f"{mismatches} mismatches")

    # ---- BUG 4: hint penalty is flat ----
    print("\n-- bug 4: hint penalty --")
    penalties = {}
    for lvl in (1, 3, 6):
        gh = Game(seed=1)
        gh.level = lvl
        before = gh.score
        gh.hint()
        penalties[lvl] = gh.score - before
    check("flat -50 at L1/L3/L6", all(v == -50 for v in penalties.values()),
          f"{penalties} (L6 was -175)")

    # ---- BUG 2: level-up after ALL points ----
    print("\n-- bug 2: 400 points + void swap worth 475 must level up --")
    lu_ok = False
    lu_detail = "no seed produced a clean 475-point void swap"
    for seed in range(300):
        gv = Game(seed=seed)
        # void at (0,0); make (0,1) the swap partner, colour 'r'
        gv.special[0][0] = 'void'
        gv.colour[0][0] = 'p'
        gv.colour[0][1] = 'r'
        r_cells = [(0, 1)]
        for (r, c) in [(r, c) for r in range(8) for c in range(8)]:
            if len(r_cells) >= 20:
                break
            if (r, c) in r_cells or (r, c) == (0, 0):
                continue
            old = gv.colour[r][c]
            gv.colour[r][c] = 'r'
            if gv._has_any_match():
                gv.colour[r][c] = old
            else:
                r_cells.append((r, c))
        if len(r_cells) != 20:
            continue
        gv.score = 400
        gv.level = 1
        res = gv.apply_swap(0, 0, 0, 1)
        if res.points != 475:      # cascade added points; keep looking for a clean case
            continue
        lu_detail = (f"seed={seed} void=475 total={res.points} score={gv.score} "
                     f"level={gv.level} level_ups={res.level_ups}")
        lu_ok = res.level_ups == 1 and gv.level == 2 and gv.score == 875
        break
    check("400 + 475 -> level 2, level_ups=1", lu_ok, lu_detail)

    # ---- BUG 9: chain advances per wave ----
    print("\n-- bug 9: chain index is per wave --")
    worst = 0
    for seed in (42, 7, 123, 999, 5):
        gc = Game(seed=seed)
        mv = greedy_pick(gc)
        if not mv:
            continue
        c2 = gc.clone()
        r2 = c2.apply_swap(*mv)
        worst = max(worst, r2.chain_count)
        print(f"    seed {seed}: move {mv} -> {r2.points} pts, "
              f"matches={r2.matches}, chain(waves)={r2.chain_count}")
    check("chain_count never exceeds a sane wave count", worst <= 8, f"max waves={worst}")

    # ---- BUG 10: L/T detection is perpendicular & prefers fire ----
    print("\n-- bug 10: 4+3 L-shape makes a fire, not a starlight --")
    gl = Game(seed=2)
    for r in range(8):
        for c in range(8):
            gl.colour[r][c] = COLOURS[(r * 3 + c * 5 + (r * c) % 3) % 7]
    # build a clean deterministic L: horizontal 4 at row 0 + vertical 3 at col 3
    gl.colour[0][0] = gl.colour[0][1] = gl.colour[0][2] = gl.colour[0][3] = 'g'
    gl.colour[1][3] = gl.colour[2][3] = 'g'
    for r in range(8):
        for c in range(8):
            gl.special[r][c] = None
    ms = gl._find_matches()
    created = gl._create_specials_from_matches(ms, swap_pos=(0, 0))
    kinds = sorted(created.values())
    check("4+3 L -> fire", kinds == ['fire'],
          f"created={ {k: v for k, v in created.items()} } matches={[(m.size, m.horizontal) for m in ms]}")

    # genuine L/T of two 3-lines -> starlight
    gs = Game(seed=2)
    for r in range(8):
        for c in range(8):
            gs.colour[r][c] = COLOURS[(r * 3 + c * 5 + (r * c) % 3 + 1) % 7]
    gs.colour[0][0] = gs.colour[0][1] = gs.colour[0][2] = 'g'
    gs.colour[1][0] = gs.colour[2][0] = 'g'
    # break the accidental vertical extension if any
    if gs.colour[3][0] == 'g':
        gs.colour[3][0] = 'o'
    if gs.colour[0][3] == 'g':
        gs.colour[0][3] = 'o'
    ms2 = gs._find_matches()
    created2 = gs._create_specials_from_matches(ms2, swap_pos=None)
    check("true 3+3 L -> starlight", 'starlight' in created2.values(),
          f"created={created2}")

    # ---- BUG 11: a detonated void wipes its colour ----
    print("\n-- bug 11: a void destroyed by a blast wipes its colour --")
    gvv = Game(seed=8)
    for r in range(8):
        for c in range(8):
            gvv.colour[r][c] = 'b'
            gvv.special[r][c] = None
    r_cells = [(0, 0), (1, 4), (2, 7), (4, 2), (6, 6), (7, 1)]
    for (r, c) in r_cells[1:]:
        gvv.colour[r][c] = 'r'
    gvv.colour[3][3] = 'r'
    gvv.special[3][3] = 'void'
    affected = gvv._detonate(3, 3, 'void')
    total_r = sum(1 for r in range(8) for c in range(8) if gvv.colour[r][c] == 'r')
    check("void blast returns its whole colour",
          sum(1 for (r, c) in affected if gvv.colour[r][c] == 'r') == total_r
          and len(affected) == total_r,
          f"affected={len(affected)} of {total_r} 'r' gems (1 = self only, old behaviour)")

    # ---- BUG 5+6: recursive detonation & no hang ----
    print("\n-- bugs 5+6: recursive detonations and the stall guard --")

    def cascade_probe(holder: Dict[str, Any], kind: str) -> None:
        # 'stall': an isolated 3x3 block where EVERY matched cell spawns a
        # special -> to_clear is empty -> old code looped forever.
        # 'refill': a 3x3 block merged into a random board -> real cascading.
        g6 = Game(seed=5)
        if kind == 'stall':
            pal = ['o', 'b', 'w', 'r', 'p', 'y']
            g6.colour = [[pal[(r * 2 + c * 3) % 6] for c in range(8)] for r in range(8)]
            g6.special = [[None] * 8 for _ in range(8)]
            rows = range(3, 6)
        else:
            rows = range(3)
        for r in rows:
            for c in rows:
                g6.colour[r][c] = 'g'
        t0 = time.perf_counter()
        try:
            res6 = g6._resolve_cascade((0, 0), False)
            holder['result'] = (res6.points, res6.chain_count,
                                [e['type'] for e in res6.events if e['type'] != 'match'],
                                round(time.perf_counter() - t0, 3))
        except Exception as exc:                        # noqa: BLE001
            holder['result'] = f"{type(exc).__name__}: {exc}"

    for kind in ('stall', 'refill'):
        holder: Dict[str, Any] = {}
        th = threading.Thread(target=cascade_probe, args=(holder, kind), daemon=True)
        th.start()
        th.join(10.0)
        if kind == 'stall':
            ok = (not th.is_alive() and 'result' in holder
                  and 'cascade_stalled' in str(holder.get('result')))
            label = "all-special wave breaks instead of spinning (stall guard)"
        else:
            ok = (not th.is_alive() and 'result' in holder
                  and 'cascade_stalled' not in str(holder.get('result')))
            label = "3x3 refill cascade resolves (does not hang)"
        check(label, ok, f"{kind}: {holder.get('result')}")

    # recursive fire -> fire chain
    gr = Game(seed=9)
    for r in range(8):
        for c in range(8):
            gr.colour[r][c] = COLOURS[(r * 2 + c * 3 + 1) % 7]
    gr.colour[0][0] = gr.colour[0][1] = gr.colour[0][2] = 'g'
    gr.special[0][1] = 'fire'          # fire inside the match
    gr.special[0][2] = 'starlight'     # starlight 1 cell away inside the blast
    res5 = gr._resolve_cascade((0, 0), False)
    det_kinds = [e['kind'] for e in res5.events if e['type'] == 'detonation']
    check("starlight inside a fire blast detonates too",
          det_kinds.count('fire') >= 1 and det_kinds.count('starlight') >= 1,
          f"detonations={det_kinds}")

    # ---- BUG 7: deadlock wired into the move ----
    print("\n-- bug 7: deadlock shuffle is applied and timed --")
    gd = Game(seed=4)
    # colour=(r+c)%3 is a genuine deadlock: no swap can ever make 3 in a line
    gd.colour = [[COLOURS[(r + c) % 3] for c in range(8)] for r in range(8)]
    gd.special = [[None]*8 for _ in range(8)]
    dead_legal = gd.legal_moves()
    check("(r+c)%3 board really has no legal move", len(dead_legal) == 0,
          f"{len(dead_legal)} legal moves")
    cost = gd.shuffle_board()
    check("shuffle_board returns a time cost", cost == SHUFFLE_TIME_COST, f"{cost}")
    check("shuffle_board leaves a playable board", gd.has_legal_move(),
          f"deadlocks={gd.deadlocks}")

    # apply_swap must detect the deadlock itself and fold the cost into time_cost
    gd2 = Game(seed=14)
    mv_d = greedy_pick(gd2)
    if mv_d:
        real_have = gd2.has_legal_move
        calls = {'n': 0}

        def one_deadlock() -> bool:
            calls['n'] += 1
            return False if calls['n'] == 1 else real_have()

        gd2.has_legal_move = one_deadlock        # type: ignore[assignment]
        rt7 = gd2.apply_swap(*mv_d)
        check("apply_swap reports deadlocked + shuffle time",
              rt7.deadlocked and rt7.time_cost >= MOVE_TIME_COST + SHUFFLE_TIME_COST,
              f"deadlocked={rt7.deadlocked} time_cost={rt7.time_cost} "
              f"(base {MOVE_TIME_COST} + shuffle {SHUFFLE_TIME_COST})")
        check("deadlock shuffle event logged",
              any(e['type'] == 'deadlock_shuffle' for e in rt7.events))

    # ---- BUG 8: named time constants ----
    print("\n-- bug 8: named time constants --")
    gt = Game(seed=6)
    mv = greedy_pick(gt)
    if mv:
        t0 = gt.time_left
        rt = gt.apply_swap(*mv)
        check("time_cost includes MOVE_TIME_COST", rt.time_cost >= MOVE_TIME_COST,
              f"time_cost={rt.time_cost} clock {t0}->{gt.time_left:.2f}")
        check("no move limit", not hasattr(gt, 'moves_left') or gt.moves_left is None)

    # ---- end-to-end sanity ----
    print("\n-- end-to-end 10-move greedy runs --")
    for seed in (1, 42, 123, 999):
        gg = Game(seed=seed)
        for _step in range(10):
            if gg.is_over() or not gg.has_legal_move():
                break
            m = greedy_pick(gg)
            if not m:
                dead = gg.shuffle_board()
                gg.time_left -= dead
                continue
            gg.apply_swap(*m)
        print(f"    seed {seed}: score={gg.score} level={gg.level} "
              f"moves={gg.total_moves} time={gg.time_left:.2f} deadlocks={gg.deadlocks}")
        check(f"seed {seed} made moves", gg.total_moves > 0)

    print("\n=== " + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}") + " ===")
    return 0 if not failures else 1


if __name__ == '__main__':
    import sys
    sys.exit(_selftest())
