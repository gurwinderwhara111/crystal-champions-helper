"""
Classification module for Crystal Champions gem detection.
Four classifiers: RULE, TEMPLATE, CENTROID, SHAPE.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .constants import *


class Classifier:
    """
    Four classifiers over the same reference set:
      * `rule_predict`    - hue bands + silhouette veto (primary, explainable)
      * `template_predict`- FULL-APPEARANCE template match (outline + engraving + colour)
      * `centroid_predict`- nearest neighbour in (hue, sat, val)
      * `shape_predict`   - silhouette template match only (colour blind)
    """
    
    def __init__(self, refs: Optional[Dict] = None):
        self.refs = refs or {}
        
    # ─── Reference Building ───────────────────────────────────────────
    
    @staticmethod
    def build_refs(samples: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """samples: {type: [{"core": patch, "silh": vector, "patch": patch}, ...]}"""
        refs = {}
        for t, patches in samples.items():
            refs[t] = list(patches)
        return refs
    
    @staticmethod
    def _unit_vec(patch: np.ndarray) -> np.ndarray:
        """Per-channel unit normalised feature (hue, sat, val) -> 3 channels."""
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
        f = np.stack([hsv[..., 0] / 180.0,
                      hsv[..., 1] / 255.0,
                      hsv[..., 2] / 255.0], axis=-1)
        out = np.empty_like(f)
        for c in range(3):
            ch = f[..., c]
            ch = ch - ch.mean()
            out[..., c] = ch / (np.linalg.norm(ch) + 1e-6)
        return out
    
    # ─── NCC (Core Template Match) ────────────────────────────────────
    
    def ncc_predict(self, core: Optional[np.ndarray]) -> Tuple[Optional[str], float]:
        """Template match the gem core against every learned reference."""
        if core is None or not self.refs:
            return None, 0.0
        c = self._unit_vec(cv2.resize(core, (CORE_SIZE, CORE_SIZE)))
        scores = {}
        for t, arr in self.refs.items():
            best = -1.0
            for ref in arr:
                r = self._unit_vec(ref["core"])
                s = float((c * r).sum() / 3.0)  # mean per-channel correlation
                if s > best:
                    best = s
            scores[t] = best
        order = sorted(scores.items(), key=lambda kv: -kv[1])
        margin = order[0][1] - (order[1][1] if len(order) > 1 else 0.0)
        return order[0][0], margin
    
    # ─── Full Appearance Template Match ──────────────────────────────
    
    @staticmethod
    def _patch_vec(patch: np.ndarray) -> np.ndarray:
        """Per-channel unit-normalised HSV feature of a gem patch."""
        hsv = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        f = np.stack([hsv[..., 0] / 180.0,
                      hsv[..., 1] / 255.0,
                      hsv[..., 2] / 255.0], axis=-1)
        out = np.empty_like(f)
        for c in range(3):
            ch = f[..., c]
            ch = ch - ch.mean()
            out[..., c] = ch / (np.linalg.norm(ch) + 1e-6)
        return out
    
    def template_predict(self, patch: Optional[np.ndarray]) -> Tuple[Optional[str], float]:
        """
        Match the whole gem appearance against the learned reference patches.
        Learns the actual artwork - outline, engraved facets, highlights and colour.
        """
        if patch is None or not self.refs:
            return None, 0.0
        q = self._patch_vec(patch)
        scores = {}
        for t, arr in self.refs.items():
            best = -1.0
            for ref in arr:
                p = ref.get("patch")
                if p is None:
                    continue
                s = float((q * self._patch_vec(p)).sum() / 3.0)
                if s > best:
                    best = s
            if best > -1.0:
                scores[t] = best
        if not scores:
            return None, 0.0
        order = sorted(scores.items(), key=lambda kv: -kv[1])
        margin = order[0][1] - (order[1][1] if len(order) > 1 else 0.0)
        return order[0][0], margin
    
    # ─── Nearest Colour Centroid ─────────────────────────────────────
    
    def centroid_predict(self, core: Optional[np.ndarray]) -> Tuple[Optional[str], float]:
        """
        Nearest-neighbour in (hue, saturation, value) space against reference
        centroids. Independent of hard coded hue bands.
        """
        if core is None or not self.refs:
            return None, 0.0
        hue, sat, val, _, white = core_hue(core)
        if hue is None:
            # Pearl candidate: compare whiteness with pearl reference
            best, berr = None, 1e9
            for t, arr in self.refs.items():
                for ref in arr:
                    rh, rs, rv, _, rw = core_hue(ref["core"])
                    if rh is None:
                        err = abs(white - rw)
                        if err < berr:
                            berr, best = err, t
            if best is None:
                return None, 0.0
            return best, 1.0 - min(berr, 1.0)
        scores = {}
        for t, arr in self.refs.items():
            best = 1e9
            for ref in arr:
                rh, rs, rv, _, rw = core_hue(ref["core"])
                if rh is None:
                    continue
                dh = abs(hue - rh)
                dh = min(dh, 180.0 - dh)
                d = dh + abs(sat - rs) * 0.25 + abs(val - rv) * 0.10
                best = min(best, d)
            if best < 1e8:
                scores[t] = best
        if not scores:
            return None, 0.0
        order = sorted(scores.items(), key=lambda kv: kv[1])
        margin = (order[1][1] - order[0][1]) if len(order) > 1 else 0.0
        return order[0][0], margin
    
    # ─── Shape Only Template Match ───────────────────────────────────
    
    def shape_predict(self, blob: Optional[Dict]) -> Tuple[Optional[str], float]:
        """
        Classify by SILHOUETTE alone. Completely independent of colour.
        Cross-check that catches a colour mistake (and vice versa).
        """
        s = silhouette(blob)
        if s is None or not self.refs:
            return None, 0.0
        scores = {}
        for t, arr in self.refs.items():
            best = -1.0
            for ref in arr:
                rs = ref.get("silh")
                if rs is None:
                    continue
                sc = float((s * rs).sum())
                if sc > best:
                    best = sc
            if best > -1.0:
                scores[t] = best
        if not scores:
            return None, 0.0
        order = sorted(scores.items(), key=lambda kv: -kv[1])
        margin = order[0][1] - (order[1][1] if len(order) > 1 else 0.0)
        return order[0][0], margin
    
    # ─── Rule Based (Primary) ────────────────────────────────────────
    
    def rule_predict(self, core: Optional[np.ndarray], blob: Optional[Dict]) -> Tuple[Optional[str], float]:
        """
        Colour first (hue bands, circular median, whiteness for pearl),
        then the silhouette as an INDEPENDENT VETO.
        
        The veto only fires on strong contradiction - measured silhouette
        ranges overlap between green/orange and between blue/white, so those
        pairs are never vetoed against each other.
        """
        hue, sat, val, _, white = core_hue(core)
        t = hue_to_type(hue, sat, val, white)
        if t is None:
            return None, 0.0
        
        # Colour confidence: how far hue sits from band edges
        edges = HUE_EDGES
        conf = 0.97
        if t in edges:
            lo, hi, pad = edges[t]
            if lo < hue < hi:
                margin = min(hue - lo, hi - hue)
            else:  # red wraps around 0
                margin = min(hue - lo + 180.0, hi + 180.0 - hue)
            conf = 0.97 if margin > pad * 0.5 else 0.72
        if t == "white":
            conf = 0.97 if white > WHITENESS_HIGH_CONF else 0.75
        
        # Silhouette veto
        fam = shape_family(blob["extent"]) if blob else None
        allowed = FAMILY_SET.get(t)
        if fam is not None and allowed is not None and fam not in allowed:
            # Contradiction -> do not guess
            return None, 0.0
        return t, conf


# ─── Hue / Colour Functions ──────────────────────────────────────────

def core_hue(core: Optional[np.ndarray]) -> Tuple[Optional[float], float, float, float, float]:
    """
    Dominant hue of the gem core.
    
    * CIRCULAR MEDIAN of hue, not mean. Orange hexagon has a yellow X;
      mean hue slides into yellow, median ignores it (X covers < half).
    * Pearl is white/iridescent -> hue meaningless. Identified by WHITENESS
      = median over pixels of min(B,G,R)/max(B,G,R).
      Coloured gems ~0.1, pearl ~0.7.
    """
    if core is None or core.size == 0:
        return None, 0.0, 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.float64), hsv[..., 1].astype(np.float64), hsv[..., 2].astype(np.float64)
    b, g, r = core[..., 0].astype(np.float64), core[..., 1].astype(np.float64), core[..., 2].astype(np.float64)
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    ratio = np.where(mx > 20, mn / np.maximum(mx, 1), 0.0)
    whiteness = float(np.median(ratio))
    if whiteness > WHITENESS_PEARL:
        return None, 0.0, 0.0, 0.0, whiteness
    
    sel = (s > CORE_SAT_HIGH) & (v > CORE_VAL_HIGH)
    if sel.sum() < CORE_MIN_PIXELS_HIGH:
        sel = (s > CORE_SAT_MID) & (v > CORE_VAL_MID)
    if sel.sum() < CORE_MIN_PIXELS_MID:
        sel = (v > CORE_VAL_LOW) & (s > CORE_SAT_LOW)
    if sel.sum() < CORE_MIN_PIXELS_LOW:
        return None, 0.0, 0.0, 0.0, whiteness
    hh, ss, vv = h[sel], s[sel], v[sel]
    
    # Circular median
    uniq = np.unique(hh)
    best, bestscore = None, 1e18
    for u in uniq:
        d = np.abs(hh - u)
        d = np.minimum(d, 180.0 - d)
        sc = float(d.sum())
        if sc < bestscore:
            bestscore, best = sc, float(u)
    return best, float(np.median(ss)), float(np.median(vv)), float(len(hh)), whiteness


def hue_to_type(hue: Optional[float], sat: float, val: float, whiteness: float = 0.0) -> Optional[str]:
    """Map a core hue to one of the 7 types (or None)."""
    if whiteness > WHITENESS_PEARL:
        return "white"
    if hue is None:
        return None
    if hue < 6.5 or hue > 172:
        return "red"
    if hue < 21:
        return "orange"
    if hue < 33:
        return "yellow"
    if hue < 82:
        return "green"
    if hue < 128:
        return "blue"
    if hue < 170:
        return "purple"
    return None


def shape_family(extent: Optional[float], solidity: Optional[float] = None, aspect: Optional[float] = None, corners: Optional[int] = None) -> Optional[str]:
    """
    Coarse shape family from the extent of the BRIGHT-MASK blob.
    
    Measured on all 384 labelled cells (this mask is scale consistent):
        purple 0.53 | yellow 0.53 | red 0.60 | blue 0.74 | orange 0.78
        green 0.81 | white 0.68
    """
    if extent is None:
        return None
    if extent < EXTENT_BANDS["sharp"]:
        return "sharp"
    if extent < EXTENT_BANDS["mid"]:
        return "mid"
    if extent < EXTENT_BANDS["pearl"]:
        return "pearl"
    if extent < EXTENT_BANDS["ball"]:
        return "ball"
    return "blocky"


# ─── Reference Building ──────────────────────────────────────────────

DEFAULT_REF_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'images of cristels')

_classifier_cache = None


def build_classifier(ref_dir: str = None) -> 'Classifier':
    """Build a Classifier from reference images in the given directory."""
    if ref_dir is None:
        ref_dir = DEFAULT_REF_DIR
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"Reference directory not found: {ref_dir}")
    
    # Load images
    images = []
    for f in sorted(os.listdir(ref_dir)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            p = os.path.join(ref_dir, f)
            im = cv2.imread(p)
            if im is not None:
                images.append((f, im))
    if not images:
        raise RuntimeError(f"No images found in {ref_dir}")
    
    def analyse(im, verbose=False):
        from .board import find_board
        from .cell import cell_core, cell_blob, cell_silhouette, detect_variants, gem_patch, silhouette
        board = find_board(im, verbose=verbose)
        if board is None:
            return None
        cells = []
        for (cx, cy) in board["centres"]:
            core, _ = cell_core(im, cx, cy, board["pitch"])
            blob = cell_blob(im, cx, cy, board["pitch"])
            sil = cell_silhouette(im, cx, cy, board["pitch"])
            flags = detect_variants(im, cx, cy, board["pitch"],
                                    blob["bbox"] if blob else None)
            cells.append({"cx": cx, "cy": cy, "core": core, "blob": blob,
                          "sil": sil,
                          "patch": gem_patch(im, cx, cy, board["pitch"], sil),
                          "flags": flags})
        return {"board": board, "cells": cells}
    
    by_name = {f: im for f, im in images}
    samples = {t: [] for t in TYPES}
    for t, (frag, (r, c)) in SEEDS.items():
        fname = next((f for f in by_name if frag in f), None)
        if fname is None:
            continue
        res = analyse(by_name[fname])
        if res is None:
            continue
        idx = (r - 1) * 8 + (c - 1)
        cell = res["cells"][idx]
        if cell["core"] is None:
            continue
        samples[t].append({"core": cell["core"], "silh": silhouette(cell["blob"]),
                            "patch": cell["patch"]})
    
    # Pass 2: harvest confident cells
    clf0 = Classifier(Classifier.build_refs(samples))
    for f, im in images:
        res = analyse(im)
        if res is None:
            continue
        for cell in res["cells"]:
            if cell["core"] is None:
                continue
            t, conf = clf0.rule_predict(cell["core"], cell["blob"])
            if t is None or conf < SEED_CONFIDENCE_THRESHOLD:
                continue
            if len(samples[t]) >= MAX_REFS_PER_TYPE:
                continue
            samples[t].append({"core": cell["core"],
                               "silh": silhouette(cell["blob"]),
                               "patch": cell["patch"]})
    
    return Classifier.build_refs(samples)


def get_classifier(ref_dir: str = None) -> 'Classifier':
    """Get or build the global Classifier (cached)."""
    global _classifier_cache
    if _classifier_cache is None:
        refs = build_classifier(ref_dir)
        _classifier_cache = Classifier(refs)
    return _classifier_cache