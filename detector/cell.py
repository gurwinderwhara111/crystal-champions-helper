"""
Cell processing module for Crystal Champions.
Extracts core, blob, silhouette, patch, and variant flags from each cell.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple, Set

from .constants import *


def cell_core(bgr: np.ndarray, cx: float, cy: float, pitch: float) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    """
    Central square of the cell -> pure gem core (normalised to CORE_SIZE).
    """
    half = pitch * CORE_HALF_RATIO
    x0, x1 = int(cx - half), int(cx + half)
    y0, y1 = int(cy - half), int(cy + half)
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(bgr.shape[1], x1)
    y1 = min(bgr.shape[0], y1)
    core = bgr[y0:y1, x0:x1]
    if core.size == 0:
        return None, (x0, y0, x1, y1)
    return cv2.resize(core, (CORE_SIZE, CORE_SIZE), interpolation=cv2.INTER_AREA), (x0, y0, x1, y1)


def cell_blob(bgr: np.ndarray, cx: float, cy: float, pitch: float) -> Optional[Dict[str, Any]]:
    """
    Bright-mask blob of the gem: pixels bright enough to be gem material.
    
    NOTE: this is NOT the true outline - gems have dark engraved facets
    so a brightness mask leaves holes. Kept because extent measured this way
    is SCALE CONSISTENT (cropped screenshots are downscaled, and a
    background-difference silhouette inflates its edge band at small scales,
    which shifts the extent bands per screenshot).
    
    Use cell_silhouette() when the true outline is needed.
    """
    half = pitch * BLOB_HALF_RATIO
    x0, x1 = int(cx - half), int(cx + half)
    y0, y1 = int(cy - half), int(cy + half)
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(bgr.shape[1], x1)
    y1 = min(bgr.shape[0], y1)
    win = bgr[y0:y1, x0:x1]
    if win.size == 0:
        return None
        
    m = gem_mask(win)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
        
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_BLOB_AREA:
        return None
        
    x, y, w, h = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    extent = area / float(w * h) if w and h else 0.0
    solidity = area / hull_area if hull_area else 0.0
    aspect = w / float(h) if h else 0.0
    
    return {
        "win": win, "contour": c, "area": area,
        "bbox": (x, y, w, h), "extent": extent, "solidity": solidity,
        "aspect": aspect, "corners": len(approx),
        "win_shape": (win.shape[0], win.shape[1]),
        "centroid": (x + w / 2.0, y + h / 2.0),
    }


def cell_silhouette(bgr: np.ndarray, cx: float, cy: float, pitch: float) -> Optional[Dict[str, Any]]:
    """
    The gem's TRUE outline, segmented against the TILE BACKGROUND.
    
    Thresholding on brightness does not work: gems have dark engraved facets
    (red diamond has near-black face, orange hexagon has dark lower-left facet),
    so a brightness threshold punches holes in the outline.
    
    Here the background colour is estimated from the border ring of the crop
    and every pixel that differs from it is gem material; the outer contour is
    then filled so dark facets don't count as holes.
    
    Caveat: the edge band grows when the screenshot is downscaled, so the
    measured extent drifts with scale. Good for shape identification, not for
    a fixed threshold.
    """
    half = pitch * SILHOUETTE_HALF_RATIO
    x0, x1 = int(cx - half), int(cx + half)
    y0, y1 = int(cy - half), int(cy + half)
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(bgr.shape[1], x1)
    y1 = min(bgr.shape[0], y1)
    win = bgr[y0:y1, x0:x1]
    if win.size == 0:
        return None
        
    h, w = win.shape[:2]
    if h < 6 or w < 6:
        return None
        
    # Background from border ring
    ring = np.concatenate([
        win[:3].reshape(-1, 3), win[-3:].reshape(-1, 3),
        win[:, :3].reshape(-1, 3), win[:, -3:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    
    # Difference from background
    diff = np.abs(win.astype(np.float32) - bg.astype(np.float32)).max(axis=2)
    m = (diff > 26).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
        
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_SILHOUETTE_AREA:
        return None
        
    solid = np.zeros_like(m)
    cv2.drawContours(solid, [c], -1, 255, -1)
    c = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    area = cv2.contourArea(c)
    x, y, w2, h2 = cv2.boundingRect(c)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    
    return {
        "win": win, "contour": c, "area": area,
        "bbox": (x, y, w2, h2),
        "extent": area / float(w2 * h2) if w2 and h2 else 0.0,
        "solidity": area / hull_area if hull_area else 0.0,
        "aspect": w2 / float(h2) if h2 else 0.0,
        "corners": len(approx),
        "win_shape": (h, w),
        "centroid": (x + w2 / 2.0, y + h2 / 2.0),
    }


def silhouette(blob: Optional[Dict], size: int = 32) -> Optional[np.ndarray]:
    """Normalised binary silhouette of the gem, for shape-only matching."""
    if blob is None:
        return None
    wh, ww = blob["win_shape"]
    m = np.zeros((wh, ww), np.uint8)
    cv2.drawContours(m, [blob["contour"]], -1, 255, -1)
    x, y, w, h = blob["bbox"]
    crop = m[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    s = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    s -= s.mean()
    n = np.linalg.norm(s)
    return s / (n + 1e-6)


def detect_variants(bgr: np.ndarray, cx: float, cy: float, pitch: float, gem_bbox: Optional[Tuple] = None) -> Set[str]:
    """
    Flags for the wooden cage and the red swirl aura.
    
    Both WRAP the gem, so the trick is to REMOVE the gem first: the bright
    gem body is subtracted from the red / tan masks, and whatever is left
    over is the wrapper. A red gem then has no red left, and an orange gem
    has no tan left, so neither can be mistaken for a swirl or a cage.
    """
    half = int(pitch * VARIANT_HALF_RATIO) + 2
    x0, x1 = int(cx - half), int(cx + half)
    y0, y1 = int(cy - half), int(cy + half)
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(bgr.shape[1], x1)
    y1 = min(bgr.shape[0], y1)
    win = bgr[y0:y1, x0:x1]
    flags = set()
    if win.size == 0:
        return flags
        
    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    
    # Bright gem body mask -> removed from wrapper masks
    body = ((vv > BODY_V_HIGH) & (ss > BODY_S_HIGH)) | ((vv > BODY_V_MID) & (ss <= BODY_S_HIGH))
    body = cv2.dilate(body.astype(np.uint8), np.ones((BODY_DILATE_KERNEL, BODY_DILATE_KERNEL), np.uint8)).astype(bool)
    
    def wrapper(mask: np.ndarray, min_side: float) -> bool:
        """
        A wrapper (cage / swirl) is a HOLLOW frame that fills most of the cell.
        Three tests:
          * bounding box at least min_side * pitch
          * roughly square (a frame is not a streak)
          * it encloses a hole (RETR_CCOMP gives it a child contour)
        """
        m = (mask & ~body).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((WRAPPER_OPEN_KERNEL, WRAPPER_OPEN_KERNEL), np.uint8))
        cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            return False
        for i, c in enumerate(cnts):
            area = cv2.contourArea(c)
            if area < WRAPPER_MIN_AREA_RATIO * pitch * pitch:
                continue
            x, y, w, h2 = cv2.boundingRect(c)
            side = max(w, h2)
            if side < min_side * pitch:
                continue
            if not (WRAPPER_ASPECT_RANGE[0] < w / float(max(h2, 1)) < WRAPPER_ASPECT_RANGE[1]):
                continue
            if hier[0][i][3] != -1:  # has a hole -> it is a frame
                return True
        return False
    
    red = ((hh < SWIRL_HUE_LOW) | (hh > SWIRL_HUE_HIGH)) & (ss > SWIRL_SAT) & (vv > SWIRL_VAL)
    tan = ((hh >= CAGE_HUE_LOW) & (hh <= CAGE_HUE_HIGH)) & (ss > CAGE_SAT_LOW) & (ss < CAGE_SAT_HIGH) & (vv > CAGE_VAL_LOW) & (vv < CAGE_VAL_HIGH)
    
    if wrapper(red, 0.78):
        flags.add("swirl")
    if wrapper(tan, 0.72):
        flags.add("cage")
    return flags


def gem_patch(bgr: np.ndarray, cx: float, cy: float, pitch: float, sil: Optional[Dict]) -> Optional[np.ndarray]:
    """
    Normalised FULL-APPEARANCE patch of the gem: the true silhouette, masked
    out of the cell crop and resized to a fixed square.
    
    Captures everything that makes a crystal unique - outline, the
    facet engraving inside it, and colour - so a template match on this
    identifies all seven types, including red and purple (both diamonds,
    but engraved differently).
    """
    if sil is None:
        return None
    half = pitch * PATCH_HALF_RATIO
    x0, x1 = int(cx - half), int(cx + half)
    y0, y1 = int(cy - half), int(cy + half)
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(bgr.shape[1], x1)
    y1 = min(bgr.shape[0], y1)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [sil["contour"]], -1, 255, -1)
    # Square it: crop to silhouette bbox, pad to square, resize
    x, y, bw, bh = sil["bbox"]
    side = max(bw, bh)
    cxm, cym = x + bw // 2, y + bh // 2
    a = max(0, cxm - side // 2 - 2)
    b = max(0, cym - side // 2 - 2)
    cc = min(w, a + side + 4)
    dd = min(h, b + side + 4)
    sub = crop[b:dd, a:cc]
    msub = mask[b:dd, a:cc]
    if sub.size == 0:
        return None
    out = cv2.resize(sub, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    mres = cv2.resize(msub, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
    mres = (mres > 127).astype(np.float32)[..., None]
    return (out.astype(np.float32) * mres)