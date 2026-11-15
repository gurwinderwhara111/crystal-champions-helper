"""
Board finding module for Crystal Champions.
Locates the 8x8 board using gem blob detection and lattice fitting.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from .constants import (
    BLOB_ASPECT_RANGE, BLOB_FILL_RATIO, EVENNESS_WEIGHT,
    GEM_BRIGHT_S, GEM_BRIGHT_V, GRID_TOLERANCE_RATIO,
    MAX_BLOB_AREA_RATIO, MIN_BLOBS, MIN_BLOB_AREA_RATIO,
    PEARL_S, PEARL_V,
    PITCH_SEARCH_MAX, PITCH_SEARCH_MIN, PITCH_SEARCH_STEP,
)


class BoardFinder:
    """Finds the 8x8 board in an image using gem blob detection."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
    def find_board(self, img: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Locate the 8x8 board.
        
        Strategy: find gem blobs, then for a range of candidate pitches fit a
        lattice on the x and y axes independently. A candidate is only accepted if
        it produces 8 CONSECUTIVE occupied grid lines that explain the blobs AND
        the spacing of those lines is uniform (the real board is perfectly even).
        Theme and crop independent.
        
        Returns:
            dict with keys: centres, xs, ys, pitch, mask
            or None if not found
        """
        pts, mask = self._gem_blobs(img)
        if len(pts) < MIN_BLOBS:
            if self.verbose:
                print(f"    Too few blobs: {len(pts)} < {MIN_BLOBS}")
            return None
            
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        
        best = None
        for s in np.arange(PITCH_SEARCH_MIN, PITCH_SEARCH_MAX, PITCH_SEARCH_STEP):
            gx, cx = self._axis_grid(xs, s)
            if gx is None:
                continue
            gy, cy = self._axis_grid(ys, s)
            if gy is None:
                continue
            even = min(self._evenness(gx), self._evenness(gy))
            score = cx * cy * even
            if best is None or score > best[0]:
                best = (score, gx, gy, s, cx, cy)
                
        if best is None:
            return None
            
        score, gx, gy, s, cx, cy = best
        if self.verbose:
            print(f"    blobs={len(pts)} pitch={s:.1f} cover_x={cx:.2f} cover_y={cy:.2f} even={score:.3f}")
            
        pitch = float(np.mean(np.diff(gx)))
        centres = [(x, y) for y in gy for x in gx]
        if self.verbose:
            print(f"    xs={[round(v, 1) for v in gx]}")
            print(f"    ys={[round(v, 1) for v in gy]}  pitch={pitch:.1f}")
            
        return {
            "centres": centres,
            "xs": gx,
            "ys": gy,
            "pitch": pitch,
            "mask": mask
        }
        
    def _gem_blobs(self, img: np.ndarray) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """Find gem centroids using connected components on gem mask."""
        m = gem_mask(img)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        pts = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < MIN_BLOB_AREA_RATIO or a > MAX_BLOB_AREA_RATIO:
                continue
            if w < 30 or h < 30 or w > 130 or h > 130:
                continue
            r = w / float(h)
            if r < BLOB_ASPECT_RANGE[0] or r > BLOB_ASPECT_RANGE[1]:
                continue
            if a / float(w * h) < BLOB_FILL_RATIO:
                continue
            pts.append((float(cent[i][0]), float(cent[i][1])))
        return pts, m
        
    def _axis_grid(self, coords: np.ndarray, s: float) -> Tuple[Optional[List[float]], float]:
        """
        Given 1-D coordinates and a candidate pitch, find the 8 CONSECUTIVE grid
        lines that hold the most points. Returns (lines, coverage) or (None, 0).
        """
        c = np.asarray(coords, dtype=np.float64)
        if len(c) < 8 or s is None or s <= 0:
            return None, 0.0
            
        tol = s * GRID_TOLERANCE_RATIO
        best_phase, best_hits = 0.0, -1
        
        for ph in np.arange(0.0, s, 1.0):
            k = np.round((c - ph) / s)
            hits = int((np.abs(c - (ph + k * s)) < tol).sum())
            if hits > best_hits:
                best_hits, best_phase = hits, float(ph)
                
        phase = best_phase
        k = np.round((c - phase) / s).astype(int)
        counts = {}
        vals = {}
        for kk, cc in zip(k, c):
            counts[int(kk)] = counts.get(int(kk), 0) + 1
            vals.setdefault(int(kk), []).append(cc)
            
        best_run, best_score = None, -1
        for k0 in range(min(counts) - 1, max(counts) + 2):
            run = range(k0, k0 + 8)
            sc = sum(counts.get(k, 0) for k in run)
            if sc > best_score:
                best_score, best_run = sc, run
                
        if best_run is None or best_score < 8:
            return None, 0.0
            
        lines = []
        for k in best_run:
            v = vals.get(k)
            if not v:
                return None, 0.0
            lines.append(float(np.mean(v)))
            
        lines.sort()
        coverage = best_score / float(len(c))
        return lines, coverage
        
    def _evenness(self, lines: Optional[List[float]]) -> float:
        """How uniform is the spacing of these grid lines? 1.0 = perfect."""
        if lines is None or len(lines) < 3:
            return 0.0
        d = np.diff(np.asarray(lines, dtype=np.float64))
        if d.mean() <= 0:
            return 0.0
        return 1.0 / (1.0 + EVENNESS_WEIGHT * float(np.std(d) / d.mean()))


def find_board(img: np.ndarray, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Locate the 8x8 board.
    
    Returns dict with: centres, xs, ys, pitch, mask
    """
    finder = BoardFinder(verbose=verbose)
    return finder.find_board(img)


def gem_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of bright gem pixels (excludes wooden cage and red swirl aura)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    bright = (v > GEM_BRIGHT_V) & (s > GEM_BRIGHT_S)
    pearl = (v > PEARL_V) & (s <= PEARL_S)
    m = (bright | pearl).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))