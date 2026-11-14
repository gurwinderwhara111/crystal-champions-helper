"""
HUD Parser — Extracts score, level, multiplier, timer from HUD strip
=====================================================================

Uses template matching with digit templates for robust OCR without Tesseract.
Works with the 430x46 HUD crop captured by the collector.
"""

from __future__ import annotations
import cv2
import numpy as np
import os
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

# ─── Data Classes ──────────────────────────────────────────────────────

@dataclass
class HUDState:
    """Parsed HUD state."""
    score: int = 0
    level: int = 1
    multiplier: float = 1.0
    timer_seconds: int = 180
    moves_remaining: Optional[int] = None
    timestamp: float = 0.0
    confidence: float = 0.0
    raw_ocr: Dict[str, str] = None  # raw digit strings for debugging

@dataclass
class DigitTemplate:
    """Template for a single digit."""
    digit: str
    template: np.ndarray
    w: int
    h: int

# ─── Template Manager ──────────────────────────────────────────────────

class DigitTemplateManager:
    """Manages digit templates for template-matching OCR."""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.templates: Dict[str, DigitTemplate] = {}
        self.template_dir = template_dir or str(
            Path(__file__).parent.parent / "assets" / "digit_templates"
        )
        self._load_templates()
        
    def _load_templates(self):
        """Load digit templates from directory."""
        if not os.path.exists(self.template_dir):
            # Create default templates programmatically
            self._create_default_templates()
            return
            
        for i in range(10):
            path = os.path.join(self.template_dir, f"digit_{i}.png")
            if os.path.exists(path):
                tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if tmpl is not None:
                    self.templates[str(i)] = DigitTemplate(
                        digit=str(i),
                        template=tmpl,
                        w=tmpl.shape[1],
                        h=tmpl.shape[0]
                    )
                    
        # Also load colon for timer
        colon_path = os.path.join(self.template_dir, "colon.png")
        if os.path.exists(colon_path):
            tmpl = cv2.imread(colon_path, cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                self.templates[':'] = DigitTemplate(
                    digit=':',
                    template=tmpl,
                    w=tmpl.shape[1],
                    h=tmpl.shape[0]
                )
                
        # Load 'x' for multiplier (e.g., "x1.5")
        x_path = os.path.join(self.template_dir, "x.png")
        if os.path.exists(x_path):
            tmpl = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                self.templates['x'] = DigitTemplate(
                    digit='x',
                    template=tmpl,
                    w=tmpl.shape[1],
                    h=tmpl.shape[0]
                )
                
    def _create_default_templates(self):
        """Create basic digit templates programmatically."""
        # Create simple 7-segment style digits
        for i in range(10):
            img = np.zeros((24, 14), dtype=np.uint8)
            # Simple font rendering
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(i), (1, 18), font, 0.5, 255, 1, cv2.LINE_AA)
            self.templates[str(i)] = DigitTemplate(
                digit=str(i),
                template=img,
                w=14,
                h=24
            )
            
        # Colon
        img = np.zeros((24, 6), dtype=np.uint8)
        cv2.circle(img, (3, 8), 2, 255, -1)
        cv2.circle(img, (3, 16), 2, 255, -1)
        self.templates[':'] = DigitTemplate(digit=':', template=img, w=6, h=24)
        
        # 'x' for multiplier
        img = np.zeros((24, 14), dtype=np.uint8)
        cv2.putText(img, 'x', (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
        self.templates['x'] = DigitTemplate(digit='x', template=img, w=14, h=24)

# ─── HUD Parser ────────────────────────────────────────────────────────

class HUDParser:
    """Parses HUD strip for score, level, multiplier, timer."""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.tm = DigitTemplateManager(template_dir)
        self.debug_mode = False
        self.debug_dir = None
        
    def enable_debug(self, debug_dir: str):
        """Enable debug image saving."""
        self.debug_mode = True
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
    def preprocess_hud(self, hud_bgr: np.ndarray) -> np.ndarray:
        """Preprocess HUD crop for digit detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(hud_bgr, cv2.COLOR_BGR2GRAY)
        
        # The HUD has dark text on light background or vice versa
        # Normalize contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Threshold - HUD text is typically dark on light or light on dark
        # Try adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
    
    def find_digit_regions(self, thresh: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find bounding boxes of individual digits."""
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            # Filter by size - digits are roughly 10-20px wide, 15-30px tall
            if 8 <= w <= 30 and 12 <= h <= 40 and area >= 20:
                boxes.append((x, y, w, h))
                
        # Sort left-to-right
        boxes.sort(key=lambda b: b[0])
        return boxes
    
    def match_digit(self, digit_img: np.ndarray) -> Tuple[str, float]:
        """Match a digit image against templates."""
        best_digit = '?'
        best_score = -1.0
        
        # Resize digit to standard height for matching
        h, w = digit_img.shape
        if h < 10 or w < 4:
            return '?', 0.0
            
        scale = 24.0 / h
        new_w = int(w * scale)
        resized = cv2.resize(digit_img, (new_w, 24), interpolation=cv2.INTER_AREA)
        
        for digit, tmpl in self.tm.templates.items():
            if resized.shape != tmpl.template.shape:
                # Resize template to match
                tmpl_resized = cv2.resize(tmpl.template, (resized.shape[1], resized.shape[0]))
            else:
                tmpl_resized = tmpl.template
                
            # Template matching
            result = cv2.matchTemplate(resized, tmpl_resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            
            if score > best_score:
                best_score = score
                best_digit = digit
                
        return best_digit, best_score
    
    def parse_hud(self, hud_bgr: np.ndarray) -> HUDState:
        """Parse full HUD strip and extract all values."""
        timestamp = time.time()
        
        if hud_bgr is None or hud_bgr.size == 0:
            return HUDState(timestamp=timestamp, confidence=0.0)
            
        # Preprocess
        thresh = self.preprocess_hud(hud_bgr)
        
        # Find digit regions
        boxes = self.find_digit_regions(thresh)
        
        if not boxes:
            return HUDState(timestamp=timestamp, confidence=0.0)
            
        # Group boxes into logical regions based on x-position
        # HUD layout (typical): [Score] [Level/Multiplier] [Timer] [Moves]
        # We'll parse all digits and then segment by position
        
        digits = []
        for x, y, w, h in boxes:
            digit_img = thresh[y:y+h, x:x+w]
            digit, conf = self.match_digit(digit_img)
            digits.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'digit': digit, 'conf': conf
            })
            
        if not digits:
            return HUDState(timestamp=timestamp, confidence=0.0)
            
        # Sort by x position
        digits.sort(key=lambda d: d['x'])
        
        # Extract raw digit string
        raw_digits = ''.join(d['digit'] for d in digits)
        
        # Heuristic parsing based on known HUD layout
        # Score is usually first (leftmost), then level/multiplier, then timer
        
        # Find potential timer (contains colon)
        timer_str = ""
        timer_idx = -1
        for i, d in enumerate(digits):
            if d['digit'] == ':':
                # Found timer separator
                timer_idx = i
                break
                
        # Parse timer if found
        timer_seconds = 180
        if timer_idx >= 0:
            # Timer format: MM:SS or M:SS
            timer_digits = digits[max(0, timer_idx-2):timer_idx+3]
            timer_str = ''.join(d['digit'] for d in timer_digits)
            timer_seconds = self._parse_timer(timer_str)
            
        # Parse score (leftmost digits before level/multiplier)
        # Score is typically the largest number on the left
        score = self._parse_score(digits, timer_idx)
        
        # Parse level/multiplier
        level, multiplier = self._parse_level_multiplier(digits, timer_idx)
        
        # Parse moves remaining (rightmost small number)
        moves = self._parse_moves(digits, timer_idx)
        
        # Calculate overall confidence
        confidences = [d['conf'] for d in digits if d['conf'] > 0]
        avg_conf = np.mean(confidences) if confidences else 0.0
        
        return HUDState(
            score=score,
            level=level,
            multiplier=multiplier,
            timer_seconds=timer_seconds,
            moves_remaining=moves,
            timestamp=timestamp,
            confidence=avg_conf,
            raw_ocr={'digits': raw_digits, 'timer': timer_str}
        )
    
    def _parse_timer(self, timer_str: str) -> int:
        """Parse timer string like '1:23' or '0:55' into seconds."""
        try:
            parts = timer_str.split(':')
            if len(parts) == 2:
                minutes = int(''.join(filter(str.isdigit, parts[0])))
                seconds = int(''.join(filter(str.isdigit, parts[1])))
                return minutes * 60 + seconds
        except:
            pass
        return 180
    
    def _parse_score(self, digits: List[Dict], timer_idx: int) -> int:
        """Extract score from leftmost digits."""
        # Score digits are before timer and level
        # Heuristic: take first 4-6 digits as score
        score_digits = []
        for i, d in enumerate(digits):
            if timer_idx >= 0 and i >= timer_idx:
                break
            if d['digit'].isdigit():
                score_digits.append(d['digit'])
            elif d['digit'] == 'x':  # multiplier prefix
                break
        try:
            return int(''.join(score_digits[:7])) if score_digits else 0
        except:
            return 0
    
    def _parse_level_multiplier(self, digits: List[Dict], timer_idx: int) -> Tuple[int, float]:
        """Extract level and multiplier."""
        # Look for 'x' followed by multiplier, or level indicator
        level = 1
        multiplier = 1.0
        
        for i, d in enumerate(digits):
            if d['digit'] == 'x':
                # Next digit(s) are multiplier
                mult_digits = []
                for j in range(i+1, min(i+3, len(digits))):
                    if digits[j]['digit'].isdigit() or digits[j]['digit'] == '.':
                        mult_digits.append(digits[j]['digit'])
                    else:
                        break
                try:
                    multiplier = float(''.join(mult_digits)) if mult_digits else 1.0
                    # Map multiplier to level
                    mult_to_level = {1.0: 1, 1.5: 2, 2.0: 3, 2.5: 4, 3.0: 5, 3.5: 6}
                    level = mult_to_level.get(multiplier, 1)
                except:
                    pass
                break
                
        return level, multiplier
    
    def _parse_moves(self, digits: List[Dict], timer_idx: int) -> Optional[int]:
        """Extract moves remaining (small number on right)."""
        # Moves is typically a small number (1-7) on the far right
        if timer_idx >= 0:
            # Look after timer
            for d in reversed(digits[timer_idx:]):
                if d['digit'].isdigit():
                    try:
                        val = int(d['digit'])
                        if 1 <= val <= 10:
                            return val
                    except:
                        pass
        return None


# ─── Convenience Function ────────────────────────────────────────────

def parse_hud(hud_bgr: np.ndarray, template_dir: str = None, debug_dir: str = None) -> HUDState:
    """Quick HUD parsing function."""
    parser = HUDParser(template_dir)
    if debug_dir:
        parser.enable_debug(debug_dir)
    return parser.parse_hud(hud_bgr)


# ─── Self-Test ────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    
    # Test with a sample HUD crop if available
    test_hud = r"D:\random works by gs\Crystal Champions helper\detector\dataset\20261114_090312\frames\hud_f00000.png"
    
    if os.path.exists(test_hud):
        hud = cv2.imread(test_hud)
        print(f"Testing HUD parser on {test_hud}")
        print(f"HUD shape: {hud.shape}")
        
        parser = HUDParser()
        parser.enable_debug("debug_hud")
        result = parser.parse_hud(hud)
        
        print(f"Score: {result.score}")
        print(f"Level: {result.level}")
        print(f"Multiplier: {result.multiplier}")
        print(f"Timer: {result.timer_seconds}s")
        print(f"Moves: {result.moves_remaining}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Raw OCR: {result.raw_ocr}")
    else:
        print(f"Test HUD not found: {test_hud}")
        print("Run collector first to generate HUD frames")