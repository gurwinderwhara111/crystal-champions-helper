"""
Pattern Detector for Special Gems
================================
Detects opportunities to create Fire, Starlight, and Void gems.

Safe implementation - doesn't modify existing code.
Call this from planner when you want special gem detection.
"""
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    FIRE = "fire"           # 4 in a row
    VOID = "void"           # 5 in a row
    STARLIGHT_T = "starlight_t"  # T shape
    STARLIGHT_L = "starlight_l"  # L shape


@dataclass
class PatternOpportunity:
    """Represents a pattern that can be created with one swap."""
    pattern_type: PatternType
    gem_color: str
    swap_position: Tuple[int, int]  # (row, col) of first gem to swap
    swap_direction: str  # 'right' or 'down'
    description: str
    
    def __str__(self):
        return f"{self.pattern_type.value.upper()}: {self.description}"


class PatternDetector:
    """
    Detects special gem creation opportunities in the 8x8 grid.
    
    Usage:
        detector = PatternDetector()
        opportunities = detector.find_all_opportunities(board)
        
        # board is 8x8 list of strings: ['R', 'G', 'B', 'Y', 'P', 'O', '?', '?']
        # ? = unknown, special gems will also show as ? for now
    """
    
    def __init__(self):
        self.rows = 8
        self.cols = 8
    
    def find_all_opportunities(self, board: List[List[str]]) -> List[PatternOpportunity]:
        """
        Find ALL special gem opportunities on the board.
        
        Returns list sorted by priority (Void > Starlight > Fire)
        """
        opportunities = []
        
        # Find each pattern type
        opportunities.extend(self.find_void_opportunities(board))
        opportunities.extend(self.find_starlight_opportunities(board))
        opportunities.extend(self.find_fire_opportunities(board))
        
        # Sort by priority
        priority_order = {
            PatternType.VOID: 0,
            PatternType.STARLIGHT_T: 1,
            PatternType.STARLIGHT_L: 2,
            PatternType.FIRE: 3
        }
        
        opportunities.sort(key=lambda x: priority_order[x.pattern_type])
        
        return opportunities
    
    def find_fire_opportunities(self, board: List[List[str]]) -> List[PatternOpportunity]:
        """
        Find swaps that create 4-in-a-row (Fire Gem).
        
        Pattern: [X][X][X][X] horizontally or vertically
        Need one swap to complete 4.
        """
        opportunities = []
        
        # Check horizontal 4-in-row potentials
        for row in range(self.rows):
            for col in range(self.cols - 3):  # Need 4 consecutive
                # Pattern: [A][A][A][?] or [A][A][?][A] or [A][?][A][A] or [?][A][A][A]
                window = [
                    (row, col), (row, col+1), (row, col+2), (row, col+3)
                ]
                
                gems = [board[r][c] for r, c in window]
                
                # Check if we have 3 of same and 1 different (or empty)
                if self._is_fire_opportunity(gems):
                    color = self._get_majority_color(gems)
                    if color and color != '?':
                        # Find which position needs the swap
                        swap_info = self._find_fire_swap_horizontal(board, row, col, color)
                        if swap_info:
                            opportunities.append(PatternOpportunity(
                                pattern_type=PatternType.FIRE,
                                gem_color=color,
                                swap_position=swap_info[0],
                                swap_direction=swap_info[1],
                                description=f"4-in-row horizontal at row {row}, cols {col}-{col+3}"
                            ))
        
        # Check vertical 4-in-row potentials
        for row in range(self.rows - 3):
            for col in range(self.cols):
                window = [
                    (row, col), (row+1, col), (row+2, col), (row+3, col)
                ]
                
                gems = [board[r][c] for r, c in window]
                
                if self._is_fire_opportunity(gems):
                    color = self._get_majority_color(gems)
                    if color and color != '?':
                        swap_info = self._find_fire_swap_vertical(board, row, col, color)
                        if swap_info:
                            opportunities.append(PatternOpportunity(
                                pattern_type=PatternType.FIRE,
                                gem_color=color,
                                swap_position=swap_info[0],
                                swap_direction=swap_info[1],
                                description=f"4-in-row vertical at col {col}, rows {row}-{row+3}"
                            ))
        
        return opportunities
    
    def find_void_opportunities(self, board: List[List[str]]) -> List[PatternOpportunity]:
        """
        Find swaps that create 5-in-a-row (Void Gem).
        
        Pattern: [X][X][X][X][X] straight line
        """
        opportunities = []
        
        # Check horizontal 5-in-row
        for row in range(self.rows):
            for col in range(self.cols - 4):  # Need 5 consecutive
                window = [
                    (row, col), (row, col+1), (row, col+2), 
                    (row, col+3), (row, col+4)
                ]
                
                gems = [board[r][c] for r, c in window]
                
                if self._is_void_opportunity(gems):
                    color = self._get_majority_color(gems)
                    if color and color != '?':
                        swap_info = self._find_void_swap_horizontal(board, row, col, color)
                        if swap_info:
                            opportunities.append(PatternOpportunity(
                                pattern_type=PatternType.VOID,
                                gem_color=color,
                                swap_position=swap_info[0],
                                swap_direction=swap_info[1],
                                description=f"5-in-row horizontal at row {row}, cols {col}-{col+4}"
                            ))
        
        # Check vertical 5-in-row
        for row in range(self.rows - 4):
            for col in range(self.cols):
                window = [
                    (row, col), (row+1, col), (row+2, col),
                    (row+3, col), (row+4, col)
                ]
                
                gems = [board[r][c] for r, c in window]
                
                if self._is_void_opportunity(gems):
                    color = self._get_majority_color(gems)
                    if color and color != '?':
                        swap_info = self._find_void_swap_vertical(board, row, col, color)
                        if swap_info:
                            opportunities.append(PatternOpportunity(
                                pattern_type=PatternType.VOID,
                                gem_color=color,
                                swap_position=swap_info[0],
                                swap_direction=swap_info[1],
                                description=f"5-in-row vertical at col {col}, rows {row}-{row+4}"
                            ))
        
        return opportunities
    
    def find_starlight_opportunities(self, board: List[List[str]]) -> List[PatternOpportunity]:
        """
        Find swaps that create T or L shape (Starlight Gem).
        
        T-shape: 3 vertical + 2 horizontal (centered)
        L-shape: 3 vertical + 2 horizontal (at corner)
        """
        opportunities = []
        
        # Check T-shape (vertical stem + horizontal bar)
        for row in range(self.rows - 2):
            for col in range(1, self.cols - 1):  # Need space on both sides
                # T-shape pattern
                #    [C]
                #    [C]
                # [C][C][C]
                
                t_positions = [
                    (row, col),      # top of stem
                    (row+1, col),    # middle of stem
                    (row+2, col),    # bottom center
                    (row+2, col-1),  # bottom left
                    (row+2, col+1)   # bottom right
                ]
                
                gems = [board[r][c] for r, c in t_positions]
                
                if self._is_starlight_opportunity(gems):
                    color = self._get_majority_color(gems)
                    if color and color != '?':
                        swap_info = self._find_starlight_swap(board, t_positions, color, 'T')
                        if swap_info:
                            opportunities.append(PatternOpportunity(
                                pattern_type=PatternType.STARLIGHT_T,
                                gem_color=color,
                                swap_position=swap_info[0],
                                swap_direction=swap_info[1],
                                description=f"T-shape at center ({row+1},{col})"
                            ))
        
        # Check L-shape
        for row in range(self.rows - 2):
            for col in range(self.cols - 2):
                # L-shape pattern (corner at bottom-right)
                # [C]
                # [C]
                # [C][C]
                
                l_positions = [
                    (row, col),      # top
                    (row+1, col),    # middle
                    (row+2, col),    # bottom left
                    (row+2, col+1)   # bottom right (corner)
                ]
                
                # Need one more to make 5 - either (row+2, col+2) or (row+1, col+1)
                # L-shape with 5: [C][C][C][C] vertical + 2 horizontal at end
                
                # Actually L-shape is: 3 vertical + 2 horizontal extending from bottom
                l_positions_full = [
                    (row, col),      
                    (row+1, col),    
                    (row+2, col),    
                    (row+2, col+1),  # horizontal extension
                    # 5th position could be (row+2, col+2) or we need different config
                ]
                
                # Actually let's use the correct L-shape: 3 in one direction, 2 in perpendicular at corner
                gems = [board[r][c] for r, c in l_positions]
                
                if self._is_starlight_opportunity(gems, need=4):  # L is 4 + 1 swap
                    color = self._get_majority_color(gems)
                    if color and color != '?':
                        swap_info = self._find_starlight_swap(board, l_positions, color, 'L')
                        if swap_info:
                            opportunities.append(PatternOpportunity(
                                pattern_type=PatternType.STARLIGHT_L,
                                gem_color=color,
                                swap_position=swap_info[0],
                                swap_direction=swap_info[1],
                                description=f"L-shape at corner ({row+2},{col})"
                            ))
        
        return opportunities
    
    # Helper methods
    def _is_fire_opportunity(self, gems: List[str]) -> bool:
        """Check if 4 gems can become 4-in-row with one swap."""
        # Need 3 same colors and 1 that can be swapped
        non_unknown = [g for g in gems if g != '?']
        if len(non_unknown) < 3:
            return False
        
        # Check if 3 are the same
        from collections import Counter
        counts = Counter(non_unknown)
        return any(count >= 3 for count in counts.values())
    
    def _is_void_opportunity(self, gems: List[str]) -> bool:
        """Check if 5 gems can become 5-in-row with one swap."""
        non_unknown = [g for g in gems if g != '?']
        if len(non_unknown) < 4:
            return False
        
        from collections import Counter
        counts = Counter(non_unknown)
        return any(count >= 4 for count in counts.values())
    
    def _is_starlight_opportunity(self, gems: List[str], need: int = 4) -> bool:
        """Check if gems can form T/L shape with one swap."""
        non_unknown = [g for g in gems if g != '?']
        if len(non_unknown) < need:
            return False
        
        from collections import Counter
        counts = Counter(non_unknown)
        return any(count >= need for count in counts.values())
    
    def _get_majority_color(self, gems: List[str]) -> Optional[str]:
        """Get the most common non-unknown color."""
        from collections import Counter
        non_unknown = [g for g in gems if g != '?']
        if not non_unknown:
            return None
        counts = Counter(non_unknown)
        return counts.most_common(1)[0][0]
    
    def _find_fire_swap_horizontal(self, board, row, start_col, color):
        """Find what swap completes the 4-in-row horizontally."""
        # Check each position in the window
        for offset in range(4):
            col = start_col + offset
            if board[row][col] != color and board[row][col] != '?':
                # This position needs the correct color
                # Check if we can swap with adjacent
                if col > 0 and board[row][col-1] == color:
                    return ((row, col-1), 'right')
                if col < self.cols - 1 and board[row][col+1] == color:
                    return ((row, col+1), 'left')
                if row > 0 and board[row-1][col] == color:
                    return ((row-1, col), 'down')
                if row < self.rows - 1 and board[row+1][col] == color:
                    return ((row+1, col), 'up')
        return None
    
    def _find_fire_swap_vertical(self, board, start_row, col, color):
        """Find what swap completes the 4-in-row vertically."""
        for offset in range(4):
            row = start_row + offset
            if board[row][col] != color and board[row][col] != '?':
                if row > 0 and board[row-1][col] == color:
                    return ((row-1, col), 'down')
                if row < self.rows - 1 and board[row+1][col] == color:
                    return ((row+1, col), 'up')
                if col > 0 and board[row][col-1] == color:
                    return ((row, col-1), 'right')
                if col < self.cols - 1 and board[row][col+1] == color:
                    return ((row, col+1), 'left')
        return None
    
    def _find_void_swap_horizontal(self, board, row, start_col, color):
        """Find what swap completes the 5-in-row horizontally."""
        for offset in range(5):
            col = start_col + offset
            if board[row][col] != color and board[row][col] != '?':
                if col > 0 and board[row][col-1] == color:
                    return ((row, col-1), 'right')
                if col < self.cols - 1 and board[row][col+1] == color:
                    return ((row, col+1), 'left')
        return None
    
    def _find_void_swap_vertical(self, board, start_row, col, color):
        """Find what swap completes the 5-in-row vertically."""
        for offset in range(5):
            row = start_row + offset
            if board[row][col] != color and board[row][col] != '?':
                if row > 0 and board[row-1][col] == color:
                    return ((row-1, col), 'down')
                if row < self.rows - 1 and board[row+1][col] == color:
                    return ((row+1, col), 'up')
        return None
    
    def _find_starlight_swap(self, board, positions, color, shape_type):
        """Find what swap completes the starlight pattern."""
        # Simplified - check if any position in pattern can be swapped
        for row, col in positions:
            if board[row][col] != color and board[row][col] != '?':
                # Try swapping with adjacent same color
                for dr, dc, direction in [(-1,0,'down'), (1,0,'up'), (0,-1,'right'), (0,1,'left')]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and board[nr][nc] == color:
                        return ((nr, nc), direction)
        return None


# Standalone test function
def test_pattern_detector():
    """Test the pattern detector with sample boards."""
    detector = PatternDetector()
    
    # Test board with fire opportunity (3 red + 1 blue in row)
    test_board = [
        ['R', 'R', 'R', 'B', 'G', 'G', 'Y', 'Y'],
        ['G', 'G', 'B', 'B', 'B', 'B', 'R', 'R'],  # Row 1: 4 blues
        ['Y', 'Y', 'R', 'R', 'R', 'R', 'G', 'G'],  # Row 2: 4 reds
        ['P', 'P', 'Y', 'Y', 'B', 'B', 'R', 'R'],
        ['R', 'R', 'G', 'G', 'Y', 'Y', 'P', 'P'],
        ['B', 'B', 'R', 'R', 'G', 'G', 'Y', 'Y'],
        ['P', 'P', 'Y', 'Y', 'R', 'R', 'B', 'B'],
        ['G', 'G', 'B', 'B', 'P', 'P', 'Y', 'Y']
    ]
    
    print("Testing Pattern Detector...")
    print("=" * 50)
    
    opportunities = detector.find_all_opportunities(test_board)
    
    if opportunities:
        print(f"\nFound {len(opportunities)} opportunities:")
        for opp in opportunities:
            print(f"  -> {opp}")
    else:
        print("\nNo opportunities found in test board.")
    
    return opportunities


if __name__ == "__main__":
    test_pattern_detector()
