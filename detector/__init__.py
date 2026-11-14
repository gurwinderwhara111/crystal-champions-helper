"""
Detector module for Crystal Champions gem detection.
"""

from .board import find_board, BoardFinder
from .cell import (
    cell_core, cell_blob, cell_silhouette,
    detect_variants, gem_patch
)
from .classify import (
    Classifier, TYPES, TYPE_INDEX,
    hue_to_type, shape_family, core_hue,
    build_classifier, get_classifier
)
from .detect import detect_board, analyse_board

__all__ = [
    'find_board', 'BoardFinder',
    'cell_core', 'cell_blob', 'cell_silhouette',
    'detect_variants', 'gem_patch',
    'Classifier', 'TYPES', 'TYPE_INDEX',
    'hue_to_type', 'shape_family', 'core_hue',
    'build_classifier', 'get_classifier',
    'detect_board', 'analyse_board',
]

__version__ = '2.0.0'