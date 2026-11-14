"""
Simulation / Rule Engine module for Crystal Champions.
"""

from .rules import Game, MoveResult, COLOURS, greedy_pick

__all__ = [
    'Game',
    'MoveResult',
    'COLOURS',
    'greedy_pick',
]

__version__ = '2.0.0'