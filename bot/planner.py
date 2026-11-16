"""
Planner — connects detector output to the rule engine.

Takes an 8x8 grid of gem types from the detector, builds a `sim.rules.Game`
state, and evaluates swaps. The best-move search is a real 2-ply:

  stage 1 — evaluate every legal move on the live board, keep the top
            `lookahead` by immediate points;
  stage 2 — for each of those, play the move, evaluate EVERY legal reply on
            the resulting board, and keep the best one.

Ranking: `combined = points + 0.6 * followup_points`.

Budget: < 600 ms per call on a 2-core i3 (measured ~40 ms here).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sim.rules import COLOURS, Game, MoveResult

# Detector emits LONG type names; the rule engine uses short codes.
DET_TO_GAME: Dict[str, str] = {
    "green": "g",
    "orange": "o",
    "blue": "b",
    "white": "w",
    "red": "r",
    "purple": "p",
    "yellow": "y",
}
GAME_TO_DET: Dict[str, str] = {v: k for k, v in DET_TO_GAME.items()}

FOLLOWUP_WEIGHT = 0.6
ROWS = 8
COLS = 8


def _to_game_code(value: Any) -> Optional[str]:
    """Long name or short code -> short code. '?'/unknown -> None (hole)."""
    if value is None:
        return None
    if value in DET_TO_GAME:            # long name
        return DET_TO_GAME[value]
    if value in COLOURS:                # already a short code
        return value
    return None


def grid_to_game(grid: List[List[str]], level: int = 1, score: float = 0.0,
                 time_left: float = 180.0) -> Game:
    """
    8x8 grid of long names or short codes -> `sim.rules.Game`.

    '?' and anything unrecognised become a hole (colour None). The caller's
    grid is never modified.
    """
    if len(grid) != ROWS or any(len(row) != COLS for row in grid):
        raise ValueError(f"grid must be {ROWS}x{COLS}")

    game = Game(seed=0)                 # fixed seed => stable suggestions at 1 Hz
    game.level = level
    game.score = float(score)
    game.time_left = float(time_left)

    for r in range(ROWS):
        for c in range(COLS):
            game.colour[r][c] = _to_game_code(grid[r][c])
            game.special[r][c] = None   # the detector does not classify specials

    return game


def evaluate_moves(game: Game) -> List[Tuple[Tuple[int, int, int, int], MoveResult]]:
    """1-ply: every legal move, sorted by immediate points (descending)."""
    results: List[Tuple[Tuple[int, int, int, int], MoveResult]] = []
    for mv in game.legal_moves():
        g = game.clone()
        results.append((mv, g.apply_swap(*mv)))
    results.sort(key=lambda item: item[1].points, reverse=True)
    return results


def _best_reply(game: Game) -> Optional[Dict[str, Any]]:
    """Best single move on `game` (1-ply). None when there is no legal move."""
    best_points: Optional[int] = None
    best_move: Optional[Tuple[int, int, int, int]] = None
    for fmv in game.legal_moves():
        g = game.clone()
        pts = g.apply_swap(*fmv).points
        if best_points is None or pts > best_points:
            best_points, best_move = pts, fmv
    if best_move is None:
        return None
    return {"move": best_move, "points": int(best_points)}


def best_move(grid: List[List[str]], level: int = 1, score: float = 0.0,
              time_left: float = 180.0, lookahead: int = 6) -> Optional[Dict[str, Any]]:
    """
    Two-ply best move (docs/CONTRACTS.md §2).
    
    NEW: First checks for special gem patterns (Void, Starlight, Fire),
    then falls back to normal 3-match detection.

    Returns None when there is no legal move, else:
      {"move", "points", "followup", "combined", "chain", "specials",
       "detonations", "level_ups", "pattern_type", "is_pattern"}

    Never mutates `grid`.
    """
    
    # ── NEW: Check for special gem patterns first ──────────────────────
    # Allow up to 10 unknown cells - pattern detection can still work
    unknown_count = sum(1 for row in grid for cell in row if cell == '?' or cell is None)
    
    # Log pattern detection attempt
    print(f"[DEBUG] Unknown cells: {unknown_count}, checking patterns...")
    
    if unknown_count < 10:  # Allow up to 10 unknowns
        try:
            # Convert long names to short codes for pattern detector
            short_grid = []
            for row in grid:
                short_row = []
                for cell in row:
                    if cell in DET_TO_GAME:
                        short_row.append(DET_TO_GAME[cell])
                    elif cell in COLOURS:
                        short_row.append(cell)
                    else:
                        short_row.append('?')  # Keep unknowns
                short_grid.append(short_row)
            
            # Check patterns
            from bot.pattern_detector import PatternDetector, PatternType
            detector = PatternDetector()
            print(f"[DEBUG] Running pattern detection...")
            opportunities = detector.find_all_opportunities(short_grid)
            
            print(f"[DEBUG] Found {len(opportunities)} pattern opportunities")
            
            if opportunities:
                best_opp = opportunities[0]
                r, c = best_opp.swap_position
                
                print(f"[DEBUG] Best pattern: {best_opp.pattern_type.value} at ({r},{c})")
                
                # Convert swap direction to second position
                direction_map = {
                    'right': (r, c + 1),
                    'left': (r, c - 1),
                    'down': (r + 1, c),
                    'up': (r - 1, c)
                }
                r2, c2 = direction_map.get(best_opp.swap_direction, (r, c))
                
                # Validate bounds
                if 0 <= r2 < 8 and 0 <= c2 < 8:
                    print(f"[DEBUG] Returning pattern move: ({r},{c}) -> ({r2},{c2})")
                    
                    return {
                        "move": (r, c, r2, c2),
                        "points": 1000,  # High priority for patterns
                        "combined": 1000,
                        "chain": 0,
                        "specials": 0,
                        "detonations": 0,
                        "level_ups": 0,
                        "followup": None,
                        "pattern_type": best_opp.pattern_type.value,
                        "is_pattern": True,
                        "pattern_description": str(best_opp)
                    }
                else:
                    print(f"[DEBUG] Pattern out of bounds: ({r2},{c2})")
            else:
                print(f"[DEBUG] No patterns found, falling back to normal")
                
        except Exception as e:
            print(f"[DEBUG] Pattern detection error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[DEBUG] Too many unknowns ({unknown_count}), skipping patterns")
    
    # ── ORIGINAL: Normal 3-match detection ─────────────────────────────
    game = grid_to_game(grid, level=level, score=score, time_left=time_left)
    moves = game.legal_moves()
    if not moves:
        return None

    # ── stage 1: every legal move on the live board ──────────────────
    first_ply: List[Tuple[Tuple[int, int, int, int], MoveResult, Game]] = []
    for mv in moves:
        g = game.clone()
        res = g.apply_swap(*mv)
        first_ply.append((mv, res, g))

    # Deterministic ordering: points desc, then move tuple for stable ties.
    first_ply.sort(key=lambda item: (-item[1].points, item[0]))
    candidates = first_ply[:max(1, int(lookahead))]

    # ── stage 2: best reply from each candidate's resulting board ────
    best: Optional[Dict[str, Any]] = None
    for mv, res, g in candidates:
        followup = _best_reply(g)
        followup_points = followup["points"] if followup else 0
        combined = res.points + FOLLOWUP_WEIGHT * followup_points
        cand = {
            "move": mv,
            "points": int(res.points),
            "followup": followup,
            "combined": float(combined),
            "chain": int(res.chain_count),
            "specials": int(res.specials_created),
            "detonations": int(res.detonations),
            "level_ups": int(res.level_ups),
        }
        if best is None or cand["combined"] > best["combined"]:
            best = cand

    return best


def best_moves(grid: List[List[str]], level: int = 1, score: float = 0.0,
               time_left: float = 180.0, lookahead: int = 6, top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Return top N moves (1-ply only for speed).
    
    Returns list of dicts sorted by points descending:
      [{"move", "points", "combined"}, ...]
    
    Never mutates `grid`.
    """
    game = grid_to_game(grid, level=level, score=score, time_left=time_left)
    moves = game.legal_moves()
    if not moves:
        return []

    # Evaluate all legal moves (1-ply only for speed)
    results: List[Tuple[Tuple[int, int, int, int], int]] = []
    for mv in moves:
        g = game.clone()
        pts = g.apply_swap(*mv).points
        results.append((mv, int(pts)))

    # Sort by points descending
    results.sort(key=lambda item: (-item[1], item[0]))

    # Return top N
    top_moves = []
    for mv, pts in results[:top_n]:
        top_moves.append({
            "move": mv,
            "points": pts,
            "combined": float(pts),
        })

    return top_moves


def format_move(move: Tuple[int, int, int, int]) -> str:
    """Format a move as 'R3C2 -> R3C5' (1-based, human readable)."""
    r1, c1, r2, c2 = move
    return f"R{r1 + 1}C{c1 + 1} -> R{r2 + 1}C{c2 + 1}"


def describe(result: Optional[Dict[str, Any]]) -> str:
    """One-line readout for the snackbar."""
    if result is None:
        return "no legal move"
    text = f"{format_move(result['move'])}  +{result['points']}"
    follow = result.get("followup")
    if follow:
        text += f"  then +{follow['points']}"
    return text


# ─── Self-test ────────────────────────────────────────────────────────

def _real_grid() -> Tuple[List[List[str]], str]:
    """Read a real board from a reference screenshot via detector.grid."""
    import glob
    import os

    import cv2

    from detector.constants import DEFAULT_REF_DIR
    from detector.grid import calibrate, capture_region, read_grid

    shots = sorted(glob.glob(os.path.join(DEFAULT_REF_DIR, "*.png")))
    if not shots:
        raise RuntimeError(f"no reference screenshots in {DEFAULT_REF_DIR}")
    img = cv2.imread(shots[-2] if len(shots) > 1 else shots[0])
    grid = calibrate(full_bgr=img)
    region = capture_region(grid)
    crop = img[region["top"]:region["top"] + region["height"],
               region["left"]:region["left"] + region["width"]].copy()
    rows, stats = read_grid(crop, grid)
    return rows, os.path.basename(shots[-2] if len(shots) > 1 else shots[0])


def _main() -> int:
    import time

    print("=== bot.planner self-test ===")
    rows, shot = _real_grid()
    print(f"board read from {shot} (unknown cells: none count below)")
    for row in rows:
        print("  " + " ".join(t[:5] for t in row))

    game = grid_to_game(rows)
    print(f"legal moves: {len(game.legal_moves())}")

    snapshot = [row[:] for row in rows]
    t0 = time.perf_counter()
    result = best_move(rows)
    first_call = 1000 * (time.perf_counter() - t0)
    if result is None:
        print("no legal move!")
        return 1

    print(f"\nbest move: {format_move(result['move'])}  +{result['points']} pts"
          f"  chain={result['chain']} specials={result['specials']}"
          f" detonations={result['detonations']} level_ups={result['level_ups']}")
    follow = result["followup"]
    if follow:
        print(f"  follow-up: {format_move(follow['move'])}  +{follow['points']} pts")
    else:
        print("  follow-up: none")
    print(f"  combined: {result['combined']:.1f}")
    print(f"  first call: {first_call:.1f} ms")

    # Top 5 with their follow-ups.
    game = grid_to_game(rows)
    scored = []
    for mv in game.legal_moves():
        g = game.clone()
        res = g.apply_swap(*mv)
        scored.append((res.points, mv, g))
    scored.sort(key=lambda t: (-t[0], t[1]))
    print("\ntop 5 by immediate points:")
    for pts, mv, g in scored[:5]:
        follow = _best_reply(g)
        combined = pts + FOLLOWUP_WEIGHT * (follow["points"] if follow else 0)
        tail = format_move(follow["move"]) + f" +{follow['points']}" if follow else "-"
        print(f"  {format_move(mv):<22} +{pts:<5} then {tail:<22} combined {combined:.1f}")

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        best_move(rows)
        times.append(1000 * (time.perf_counter() - t0))
    times.sort()
    print(f"\nbest_move over 20 calls: mean {sum(times) / len(times):.1f} ms, "
          f"median {times[len(times) // 2]:.1f} ms, max {times[-1]:.1f} ms "
          f"(budget 600 ms)")

    if snapshot != rows:
        print("ERROR: input grid was mutated!", flush=True)
        return 1
    if times[-1] > 600.0:
        print("ERROR: over the 600 ms budget", flush=True)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
