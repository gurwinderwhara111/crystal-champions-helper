"""
Bot components for Crystal Champions.
"""

from .planner import grid_to_game, evaluate_moves, best_move, format_move
from .clicker import move_mouse, click_at, double_click_cell
from .auto import AutoPlayer

__all__ = [
    'grid_to_game',
    'evaluate_moves',
    'best_move',
    'format_move',
    'move_mouse',
    'click_at',
    'double_click_cell',
    'AutoPlayer',
]

__version__ = '2.0.0'