"""
Planner V2 with Pattern Detection
================================
Integrates special gem pattern detection with existing move finding.

Priority order:
1. Void + Void combo (if two voids exist)
2. Create Void (5-in-row)
3. Create Starlight (T/L shape)
4. Create Fire (4-in-row)
5. Normal 3-match (original)

Usage: Drop-in replacement for planner.py
"""
from typing import Optional, Dict, List, Tuple, Any
from .pattern_detector import PatternDetector, PatternType

# Import original functions
from .planner import best_move as original_best_move


def best_move_with_patterns(board: List[List[str]], grid: Dict) -> Optional[Dict]:
    """
    Find best move considering special gem patterns FIRST, then normal moves.
    
    Args:
        board: 8x8 grid of gem colors ('R', 'G', 'B', 'Y', 'P', 'O', 'W', '?')
        grid: Grid configuration
        
    Returns:
        Dict with move info or None
    """
    detector = PatternDetector()
    
    # Step 1: Find ALL special pattern opportunities
    opportunities = detector.find_all_opportunities(board)
    
    if opportunities:
        # Get the highest priority opportunity
        best_opp = opportunities[0]
        
        # Convert to move format
        r, c = best_opp.swap_position
        
        # Calculate second position based on direction
        direction_map = {
            'right': (r, c + 1),
            'left': (r, c - 1),
            'down': (r + 1, c),
            'up': (r - 1, c)
        }
        
        r2, c2 = direction_map[best_opp.swap_direction]
        
        # Validate bounds
        if 0 <= r2 < 8 and 0 <= c2 < 8:
            # Determine arrow color based on pattern type
            color_map = {
                PatternType.VOID: 'red',
                PatternType.STARLIGHT_T: 'blue',
                PatternType.STARLIGHT_L: 'blue',
                PatternType.FIRE: 'orange'
            }
            
            return {
                "move": ((r, c), (r2, c2)),
                "points": 0,  # Pattern creation value
                "description": str(best_opp),
                "pattern_type": best_opp.pattern_type.value,
                "priority": "special_gem",
                "arrow_color": color_map[best_opp.pattern_type]
            }
    
    # Step 2: Fall back to normal 3-match detection
    return original_best_move(board, grid)


# Keep compatibility
def best_move(board: List[List[str]], grid: Optional[Dict] = None) -> Optional[Dict]:
    """Main entry point - tries patterns first, then normal moves."""
    if grid is None:
        grid = {}
    return best_move_with_patterns(board, grid)
