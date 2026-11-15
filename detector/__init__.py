"""
Detector module for Crystal Champions gem detection.
"""

from .board import find_board, gem_mask, BoardFinder
from .cell import (
    cell_core, cell_blob, cell_silhouette,
    detect_variants, gem_patch, silhouette,
)
from .classify import (
    Classifier, build_classifier, get_classifier,
    core_hue, hue_to_type, shape_family,
)
from .constants import TYPES, TYPE_INDEX
from .detect import detect_board, analyse_board

__all__ = [
    'find_board', 'gem_mask', 'BoardFinder',
    'cell_core', 'cell_blob', 'cell_silhouette',
    'detect_variants', 'gem_patch', 'silhouette',
    'Classifier', 'build_classifier', 'get_classifier',
    'core_hue', 'hue_to_type', 'shape_family',
    'TYPES', 'TYPE_INDEX',
    'detect_board', 'analyse_board',
]

__version__ = '2.1.0'
