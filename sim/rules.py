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
- Swap: adjacent orthogonally; illegal if no match created (EXCEPT void/special swaps)
- Match: 3+ in a row/column, no diagonals
- Gravity: gems fall down; new gems fall in from top
- Game length: 180 s
- Levels: 6 levels, score thresholds 500 / 1,475 / 3,075 / 5,450 / 8,750
  Multipliers: ×1.0, ×1.5, ×2.0, ×2.5, ×3.0, ×3.5
  Level-up: board cleared and refilled; unused specials carry over
- No legal move → board refilled (costs time, no score penalty)
- Hint: −50 points; auto-hint after 15 s idle (also −50)

Scoring (all multiplied by current level multiplier)
----------------------------------------------------
Chain table (per match, simultaneous or cascaded):
  1st=10, 2nd=20, 3rd=30, 4th=40, 5th=60, 6th=80, 7th=110, 8th=160, 9th+=210
Match size bonus: 3 gems +0, 4 gems +10, 5+ gems +20
Special detonation bonus (running count within the move):
  1st=20, 2nd=40, 3rd=50, 4th=60, subsequent +10 each
Starlight detonation: +5 per gem destroyed (× multiplier)
Void swap: 75 for the Void + 20 per normal gem + 60 per special gem destroyed

Special Gems (Standard mode: 3 tiers)
-------------------------------------
Fire      — created by a line of 4 → detonates a 3×3 area around it
Starlight — created by two conjoined matches of 3 (L or T) → destroys its full row AND column
Void      — created by a line of 5 → SWAP with any gem removes that gem and ALL gems of that colour
            Void does not need to form a match to be swapped.
            Void+Void swap clears the entire board.

Special combinations (geometry unpublished for most; we implement best guesses)
--------------------------------------------------------------------------------
Fire+Fire         → 6×6 detonation (documented)
Fire+Starlight    → 3-row + 3-col cross (centred on midpoint)
Fire+Rocket       → 3 rows + 3 cols (not in Standard; placeholder)
Starlight+Starlight → full row+col of both gems
Rocket+Rocket     → opposing lines to edge (documented for PowerUp)
Void+Void         → clear entire board (documented)
Void+Special      → convert all gems of Void's colour into that Special, then detonate them

Assumptions / simplifications (documented so they can be corrected)
--------------------------------------------------------------------
1. Gem colours: 7 (g/o/b/w/r/p/y). The rules never state the count; this matches
   Bejeweled Blitz ancestry and our CV analysis.
2. Initial fill: random uniform, re-rolled until no matches exist.
3. Refill: random uniform from the active colour set.
4. Simultaneous matches in one wave each advance the chain counter.
5. Special creation priority: L/T (starlight) > 5 (void) > 4 (fire).
   If a cell participates in both a 4-line and an L, starlight wins.
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
12. Time cost of deadlock refill / level-up animation is modelled as 1.0 s
    of wall-clock time consumed.
13. No PowerUp mode, no Rocket Gem, no Hyperspeed in this Standard-mode engine.
    Those can be added later.

Usage
-----
    game = Game(seed=42)
    moves = game.legal_moves()
    result = game.apply_swap(*moves[0])
    print(f"Scored {result.points} pts, chain {result.chain_count}")
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional, Dict, Any
from copy import deepcopy

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
COLOURS = ('g', 'o', 'b', 'w', 'r', 'p', 'y')   # green, orange, blue, white, red, purple, yellow
SPECIALS = ('fire', 'rocket', 'starlight', 'void')

CHAIN_TABLE = (10, 20, 30, 40, 60, 80, 110, 160, 210)   # 1-based
def chain_points(k: int) -> int:
    return CHAIN_TABLE[k-1] if k <= len(CHAIN_TABLE) else CHAIN_TABLE[-1]

DETON_TABLE = (20, 40, 50, 60)   # 1st, 2nd, 3rd, 4th
def detonation_points(n: int) -> int:
    if n <= 0: return 0
    total = 0
    for i in range(n):
        total += DETON_TABLE[i] if i < len(DETON_TABLE) else DETON_TABLE[-1] + 10*(i - len(DETON_TABLE) + 1)
    return total

LEVEL_MULT = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
LEVEL_THRESH = (500, 1475, 3075, 5450, 8750)

MATCH_SIZE_BONUS = {3: 0, 4: 10}
# 5+ gems treated as +20 (rules: "+20 for 5-gem match")
def size_bonus(sz: int) -> int:
    if sz >= 5: return 20
    return MATCH_SIZE_BONUS.get(sz, 0)

DIRS = ((0,1),(1,0),(0,-1),(-1,0))

# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
@dataclass
class Match:
    cells: List[Tuple[int,int]]
    size: int
    horizontal: bool

@dataclass
class MoveResult:
    points: int
    matches: int
    detonations: int
    chain_count: int
    specials_created: int
    void_swap: bool
    level_ups: int
    time_cost: float
    events: List[Dict[str, Any]]
    board_before: Tuple
    board_after: Tuple

# ----------------------------------------------------------------------
# Game engine
# ----------------------------------------------------------------------
class Game:
    def __init__(
        self,
        seed: Optional[int] = None,
        n_colours: int = 7,
        rows: int = 8,
        cols: int = 8,
        time_limit: float = 180.0,
        mode: str = 'standard'
    ):
        self.rng = random.Random(seed)
        self.rows, self.cols = rows, cols
        self.palette = COLOURS[:n_colours]
        self.mode = mode
        self.time_left = time_limit
        self.score = 0
        self.level = 1
        self.colour = [[None]*cols for _ in range(rows)]
        self.special = [[None]*cols for _ in range(rows)]
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
        for attempt in range(1000):
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
        matches = []
        visited = set()

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
                    cells = [(r, cc) for cc in range(start, c)]
                    matches.append(Match(cells, length, True))
                    visited.update(cells)
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
                    cells = [(rr, c) for rr in range(start, r)]
                    matches.append(Match(cells, length, False))
                    visited.update(cells)
        return matches

    # -------------------- legal move generation --------------------
    def legal_moves(self) -> List[Tuple[int,int,int,int]]:
        """Return all (r1,c1,r2,c2) swaps that are legal.
        Legal if: (a) swap creates >=1 match, OR (b) involves a Void gem,
        OR (c) special↔special swap."""
        moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                for dr, dc in ((0,1),(1,0)):  # right, down to avoid dupes
                    nr, nc = r+dr, c+dc
                    if not self.in_bounds(nr, nc): continue
                    if self._swap_is_legal(r, c, nr, nc):
                        moves.append((r, c, nr, nc))
        return moves

    def _swap_is_legal(self, r1, c1, r2, c2) -> bool:
        s1, s2 = self.special[r1][c1], self.special[r2][c2]
        # Special↔special always legal
        if s1 and s2: return True
        # Void swap always legal (even with no match)
        if s1 == 'void' or s2 == 'void': return True
        # Normal swap: simulate and check for any match
        self.colour[r1][c1], self.colour[r2][c2] = self.colour[r2][c2], self.colour[r1][c1]
        self.special[r1][c1], self.special[r2][c2] = self.special[r2][c2], self.special[r1][c1]
        has = self._has_any_match()
        # revert
        self.colour[r1][c1], self.colour[r2][c2] = self.colour[r2][c2], self.colour[r1][c1]
        self.special[r1][c1], self.special[r2][c2] = self.special[r2][c2], self.special[r1][c1]
        return has

    # -------------------- void swap resolution --------------------
    def _resolve_void_swap(self, r_void, c_void, r_target, c_target) -> Tuple[int, Set[Tuple[int,int]], int, int]:
        """Resolve a void swap. Returns (points, cells_cleared, normals_destroyed, specials_destroyed)."""
        target_colour = self.colour[r_target][c_target]
        void_special = self.special[r_void][c_void]
        assert void_special == 'void'
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
    def _create_specials_from_matches(self, matches: List[Match], swap_pos: Optional[Tuple[int,int]]) -> Dict[Tuple[int,int], str]:
        """Determine specials created this wave.
        Priority: L/T (starlight) > 5-line (void) > 4-line (fire).
        Returns dict {(r,c): special_kind}."""
        created = {}
        # Detect L/T: two matches sharing exactly one cell
        shared_cells = {}
        for m in matches:
            for cell in m.cells:
                shared_cells[cell] = shared_cells.get(cell, 0) + 1
        for cell, count in shared_cells.items():
            if count >= 2:
                created[cell] = 'starlight'
        # Now 5-line and 4-line for matches not already spawning a starlight
        for m in matches:
            # Skip matches whose cells are all used by starlight creation
            if all(c in created for c in m.cells):
                continue
            if m.size >= 5:
                pos = self._special_pos(m, swap_pos)
                if pos not in created:
                    created[pos] = 'void'
            elif m.size == 4:
                pos = self._special_pos(m, swap_pos)
                if pos not in created:
                    created[pos] = 'fire'
        return created

    def _special_pos(self, match: Match, swap_pos: Optional[Tuple[int,int]]) -> Tuple[int,int]:
        """Preferred position for a special gem from this match."""
        if swap_pos and swap_pos in match.cells:
            return swap_pos
        # centre-ish
        idx = len(match.cells) // 2
        return match.cells[idx]

    # -------------------- detonation effects --------------------
    def _detonate(self, r: int, c: int, kind: str) -> Set[Tuple[int,int]]:
        """Return set of cells affected by detonating special at (r,c)."""
        affected = set()
        if kind == 'fire':
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    nr, nc = r+dr, c+dc
                    if self.in_bounds(nr, nc):
                        affected.add((nr, nc))
        elif kind == 'starlight':
            for cc in range(self.cols):
                affected.add((r, cc))
            for rr in range(self.rows):
                affected.add((rr, c))
        elif kind == 'rocket':
            # Not in Standard mode; placeholder: full row + full col
            for cc in range(self.cols):
                affected.add((r, cc))
            for rr in range(self.rows):
                affected.add((rr, c))
        elif kind == 'void':
            # Void detonation (when destroyed by another special) acts as colour wipe?
            # Undocumented; we treat it as no extra effect beyond being cleared.
            affected.add((r, c))
        return affected

    def _special_combo(self, specs: List[Tuple[int,int,str]], cleared: Set[Tuple[int,int]]) -> Set[Tuple[int,int]]:
        """Resolve special↔special combos. specs = list of (r,c,kind) being swapped.
        Returns additional cleared cells from the combo."""
        if len(specs) != 2: return set()
        (r1,c1,k1), (r2,c2,k2) = specs
        kinds = tuple(sorted([k1, k2]))
        extra = set()
        # Fire+Fire → 6×6 around midpoint (documented)
        if kinds == ('fire','fire'):
            mr, mc = (r1+r2)//2, (c1+c2)//2
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = mr+dr, mc+dc
                    if self.in_bounds(nr, nc):
                        extra.add((nr, nc))
        # Fire+Starlight → 3-row + 3-col cross at midpoint
        elif kinds == ('fire','starlight'):
            mr, mc = (r1+r2)//2, (c1+c2)//2
            for dc in range(-1, 2):
                for rr in range(self.rows):
                    extra.add((rr, mc+dc))
            for dr in range(-1, 2):
                for cc in range(self.cols):
                    extra.add((mr+dr, cc))
        # Starlight+Starlight → both full row+col
        elif kinds == ('starlight','starlight'):
            for cc in range(self.cols):
                extra.add((r1, cc))
            for rr in range(self.rows):
                extra.add((rr, c1))
            for cc in range(self.cols):
                extra.add((r2, cc))
            for rr in range(self.rows):
                extra.add((rr, c2))
        # Void+Void → entire board (documented)
        elif kinds == ('void','void'):
            for r in range(self.rows):
                for c in range(self.cols):
                    extra.add((r, c))
        # Void+Special → convert all gems of Void's colour into that Special, then detonate
        elif 'void' in kinds:
            void_idx = 0 if kinds[0] == 'void' else 1
            vr, vc, _ = specs[void_idx]
            other_kind = kinds[1-void_idx]
            target_colour = self.colour[vr][vc]
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.colour[r][c] == target_colour:
                        self.special[r][c] = other_kind
            # Now detonate all those converted specials
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.special[r][c] == other_kind:
                        extra.update(self._detonate(r, c, other_kind))
        return extra

    # -------------------- cascade resolution --------------------
    def _resolve_cascade(self, swap_pos: Tuple[int,int], is_void_swap: bool) -> MoveResult:
        """Run the full cascade after a swap (or void swap)."""
        mult = LEVEL_MULT[min(self.level-1, 5)]
        total_pts = 0
        chain_idx = 0
        deton_count = 0
        specials_created_total = 0
        events = []
        level_ups = 0
        time_cost = 0.0

        if is_void_swap:
            # Void swap already handled points/clearing; just cascade from resulting board
            pass
        else:
            # First wave: the swap itself created matches
            pass

        while True:
            matches = self._find_matches()
            if not matches:
                break

            # Special creation this wave
            created = self._create_specials_from_matches(matches, swap_pos if chain_idx == 0 else None)
            specials_created_total += len(created)

            # Cells to clear this wave
            to_clear = set()
            for m in matches:
                for cell in m.cells:
                    if cell not in created:
                        to_clear.add(cell)

            # Detonate specials that are in to_clear
            wave_detonations = []
            for r, c in list(to_clear):
                sp = self.special[r][c]
                if sp:
                    wave_detonations.append((r, c, sp))
                    to_clear.update(self._detonate(r, c, sp))

            # Special↔special combos if the swap involved two specials
            if chain_idx == 0:
                s1 = self.special[swap_pos[0]][swap_pos[1]]
                # find the other swapped cell
                # We don't have the other pos here easily; call _special_combo separately
                pass  # handled in apply_swap

            # Score matches
            for m in matches:
                chain_idx += 1
                base = chain_points(chain_idx) + size_bonus(m.size)
                total_pts += int(base * mult)
                events.append({
                    'type': 'match',
                    'cells': m.cells,
                    'size': m.size,
                    'chain': chain_idx,
                    'points': int(base * mult)
                })

            # Score detonations
            for r, c, kind in wave_detonations:
                deton_count += 1
                det_pts = int(detonation_points(deton_count) * mult)
                total_pts += det_pts
                events.append({
                    'type': 'detonation',
                    'kind': kind,
                    'pos': (r, c),
                    'detonation_number': deton_count,
                    'points': det_pts
                })
                # Starlight extra +5 per gem destroyed
                if kind == 'starlight':
                    affected = self._detonate(r, c, 'starlight')
                    gems_destroyed = len([cell for cell in affected if self.colour[cell[0]][cell[1]] is not None])
                    sl_pts = int(gems_destroyed * 5 * mult)
                    total_pts += sl_pts
                    events.append({
                        'type': 'starlight_bonus',
                        'gems': gems_destroyed,
                        'points': sl_pts
                    })

            # Apply clearing
            for r, c in to_clear:
                self.colour[r][c] = None
                self.special[r][c] = None

            # Apply created specials
            for (r, c), kind in created.items():
                self.special[r][c] = kind

            # Gravity + refill
            self._gravity_and_refill()

        # Check level-ups
        while self.level < 6 and self.score + total_pts >= LEVEL_THRESH[self.level-1]:
            self.level += 1
            level_ups += 1
            time_cost += 1.0  # animation cost
            # Level-up: clear board, refill, keep specials
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.special[r][c] is None:
                        self.colour[r][c] = None
            self._gravity_and_refill()
            # After refill, specials remain in their cells (already kept)

        return MoveResult(
            points=total_pts,
            matches=chain_idx,
            detonations=deton_count,
            chain_count=chain_idx,
            specials_created=specials_created_total,
            void_swap=is_void_swap,
            level_ups=level_ups,
            time_cost=time_cost,
            events=events,
            board_before=(),  # filled by caller
            board_after=self.board_tuple()
        )

    # -------------------- public swap --------------------
    def apply_swap(self, r1: int, c1: int, r2: int, c2: int) -> MoveResult:
        """Execute a swap, resolve cascade, update score/level/time."""
        self.total_moves += 1
        before = self.board_tuple()

        s1, s2 = self.special[r1][c1], self.special[r2][c2]

        # Special↔special swap (check FIRST, includes Void+Void)
        if s1 and s2:
            kinds = tuple(sorted([s1, s2]))
            # Swap the specials and colours
            self.special[r1][c1], self.special[r2][c2] = s2, s1
            self.colour[r1][c1], self.colour[r2][c2] = self.colour[r2][c2], self.colour[r1][c1]
            # Combo resolution
            combo_cleared = self._special_combo([(r1,c1,s2), (r2,c2,s1)], set())
            combo_pts = 0
            combo_events = []
            for r, c in combo_cleared:
                if self.in_bounds(r, c) and self.colour[r][c] is not None:
                    had_special = self.special[r][c]
                    self.colour[r][c] = None
                    self.special[r][c] = None
                    mult = LEVEL_MULT[min(self.level-1, 5)]
                    if had_special:
                        combo_pts += int(60 * mult)
                        combo_events.append({'type': 'combo_detonate_special', 'pos': (r,c), 'kind': had_special})
                    else:
                        combo_pts += int(20 * mult)
            # Then cascade from resulting board
            res = self._resolve_cascade((r1, c1), is_void_swap=False)
            res.points += combo_pts
            res.events = combo_events + res.events
            res.board_before = before
            self.score += res.points
            self.time_left -= (1.0 + res.time_cost)
            return res

        # Void swap (single void only)
        if s1 == 'void' or s2 == 'void':
            vr, vc = (r1, c1) if s1 == 'void' else (r2, c2)
            tr, tc = (r2, c2) if s1 == 'void' else (r1, c1)
            void_pts, cleared, normals, specials = self._resolve_void_swap(vr, vc, tr, tc)
            events = [{
                'type': 'void_swap',
                'void_pos': (vr, vc),
                'target_pos': (tr, tc),
                'normals': normals,
                'specials': specials,
                'points': void_pts
            }]
            res = self._resolve_cascade((vr, vc), is_void_swap=True)
            res.points += void_pts
            res.events = events + res.events
            res.void_swap = True
            res.board_before = before
            self.time_left -= (1.0 + res.time_cost)
            self.score += res.points
            return res

        # Normal swap
        self.colour[r1][c1], self.colour[r2][c2] = self.colour[r2][c2], self.colour[r1][c1]
        self.special[r1][c1], self.special[r2][c2] = self.special[r2][c2], self.special[r1][c1]

        res = self._resolve_cascade((r1, c1), is_void_swap=False)
        res.board_before = before
        self.score += res.points
        self.time_left -= (1.0 + res.time_cost)
        return res

    # -------------------- deadlock / no moves --------------------
    def has_legal_move(self) -> bool:
        return len(self.legal_moves()) > 0

    def shuffle_board(self) -> float:
        """No legal move → refill entire board. Costs time, no score penalty."""
        self.deadlocks += 1
        self._fill_full()
        return 1.0  # time cost

    # -------------------- hint --------------------
    def hint(self) -> Optional[Tuple[int,int,int,int]]:
        """Return a legal move (first found) and charge −50×mult."""
        moves = self.legal_moves()
        if not moves:
            return None
        mult = LEVEL_MULT[min(self.level-1, 5)]
        penalty = int(50 * mult)
        self.score -= penalty
        self.hints_used += 1
        return moves[0]

    # -------------------- level multiplier getter --------------------
    def current_multiplier(self) -> float:
        return LEVEL_MULT[min(self.level-1, 5)]

    def next_threshold(self) -> Optional[int]:
        if self.level >= 6: return None
        return LEVEL_THRESH[self.level-1]

    def progress_to_next(self) -> float:
        nxt = self.next_threshold()
        if nxt is None: return 1.0
        prev = LEVEL_THRESH[self.level-2] if self.level > 1 else 0
        return (self.score - prev) / (nxt - prev)

    # -------------------- game over check --------------------
    def is_over(self) -> bool:
        return self.time_left <= 0

# ----------------------------------------------------------------------
# Baseline greedy policy (for quick smoke tests)
# ----------------------------------------------------------------------
def greedy_pick(game: Game) -> Optional[Tuple[int,int,int,int]]:
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
if __name__ == '__main__':
    import sys
    print("=== Crystal Champions Rule Engine Self-Test ===")
    for seed in [1, 42, 123, 999]:
        print(f"\n--- Seed {seed} ---")
        game = Game(seed=seed)
        print(f"Initial board:")
        for r in range(8):
            row = ''.join(game.colour[r][c][0] for c in range(8))
            print(f"  {row}")
        moves = game.legal_moves()
        print(f"Legal moves: {len(moves)}")
        if moves:
            mv = greedy_pick(game)
            if mv:
                g2 = game.clone()
                res = g2.apply_swap(*mv)
                print(f"Greedy move {mv} -> {res.points} pts, chain {res.chain_count}, deton {res.detonations}, specials_created {res.specials_created}")
                print(f"  Score: {game.score} -> {g2.score}, Level: {game.level} -> {g2.level}, Time left: {g2.time_left:.1f}")
        # Simulate a few moves
        for step in range(10):
            if game.is_over(): break
            if not game.has_legal_move():
                game.time_left -= game.shuffle_board()
                print(f"  Deadlock refill, time left {game.time_left:.1f}")
                continue
            mv = greedy_pick(game)
            if not mv: break
            res = game.apply_swap(*mv)
            print(f"  Move {step+1}: {mv} -> +{res.points} pts (total {game.score}), L{game.level}, t={game.time_left:.1f}")
            if res.level_ups:
                print(f"    LEVEL UP -> L{game.level}")
        print(f"Final: score={game.score}, level={game.level}, moves={game.total_moves}, time={game.time_left:.1f}, hints={game.hints_used}, deadlocks={game.deadlocks}")
    print("\n=== Self-test complete ===")