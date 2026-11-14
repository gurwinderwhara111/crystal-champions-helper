"""
Planner — connects detector output to the rule engine.

Takes an 8x8 grid of gem types from the detector, builds a Game state,
evaluates all legal moves via the rule engine, returns the best swap.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import sys
import os

# Add sim to path
SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sim')
sys.path.insert(0, SIM_DIR)

from rules import Game, MoveResult, COLOURS

# Detector uses short codes: g,o,b,w,r,p,y
# Rule engine uses same codes - perfect match!
DET_TO_GAME = {c: c for c in COLOURS}
GAME_TO_DET = {c: c for c in COLOURS}

# Special gem mapping: detector doesn't detect specials yet
# For now, assume no specials on board (will be added later)
# Detector flags: 'cage', 'swirl' - not yet mapped to specials


def grid_to_game(grid: List[List[str]], level: int = 1, score: int = 0, time_left: float = 180.0) -> Game:
    """Convert detector 8x8 grid to a Game instance."""
    g = Game(seed=None)  # seed doesn't matter - we'll overwrite board
    g.level = level
    g.score = score
    g.time_left = time_left
    
    # Overwrite board with detected grid
    for r in range(8):
        for c in range(8):
            det = grid[r][c]
            if det in DET_TO_GAME:
                g.colour[r][c] = DET_TO_GAME[det]
                g.special[r][c] = None
            else:
                g.colour[r][c] = None
                g.special[r][c] = None
    
    return g


def evaluate_moves(game: Game, depth: int = 1) -> List[Tuple[Tuple[int,int,int,int], MoveResult]]:
    """Evaluate all legal moves, return list of (move, result) sorted by points descending."""
    moves = game.legal_moves()
    results = []
    
    for mv in moves:
        g = game.clone()
        res = g.apply_swap(*mv)
        results.append((mv, res))
    
    # Sort by points descending
    results.sort(key=lambda x: x[1].points, reverse=True)
    return results


def best_move(grid: List[List[str]], level: int = 1, score: int = 0, time_left: float = 180.0,
              depth: int = 1) -> Optional[Tuple[Tuple[int,int,int,int], MoveResult]]:
    """Return the best move for the given board state."""
    game = grid_to_game(grid, level, score, time_left)
    evaluated = evaluate_moves(game, depth)
    if not evaluated:
        return None
    return evaluated[0]


def format_move(mv: Tuple[int,int,int,int]) -> str:
    """Format move as human-readable string."""
    r1, c1, r2, c2 = mv
    return f"({r1},{c1}) -> ({r2},{c2})"


# Self-test
if __name__ == '__main__':
    # Test with a known grid from the reference image
    test_grid = [
        ['g','o','b','b','g','o','r','p'],
        ['b','r','o','o','r','w','y','p'],
        ['r','g','r','w','o','b','g','b'],
        ['o','w','o','y','g','b','p','o'],
        ['y','y','b','w','g','y','p','o'],
        ['r','r','o','w','w','g','o','g'],
        ['p','g','g','o','g','p','r','g'],
        ['b','p','w','b','b','w','p','b'],
    ]
    
    print("Testing planner...")
    result = best_move(test_grid, level=1, score=0, time_left=180.0)
    if result:
        mv, res = result
        print(f"Best move: {format_move(mv)}")
        print(f"  Points: {res.points}")
        print(f"  Chain: {res.chain_count}")
        print(f"  Specials created: {res.specials_created}")
        print(f"  Detonations: {res.detonations}")
        print(f"  Level ups: {res.level_ups}")
    else:
        print("No legal moves!")