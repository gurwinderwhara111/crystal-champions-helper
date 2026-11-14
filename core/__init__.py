"""
Core shared modules for Crystal Champions Bot.
"""

from .logging_system import CrystalLogger, get_logger, export_all_training_data
from .hud_parser import HUDParser, HUDState, parse_hud
from .auto_player import IntegratedAutoPlayer, AutoConfig, create_auto_player

__all__ = [
    'CrystalLogger',
    'get_logger',
    'export_all_training_data',
    'HUDParser',
    'HUDState',
    'parse_hud',
    'IntegratedAutoPlayer',
    'AutoConfig',
    'create_auto_player',
]

__version__ = '2.0.0'