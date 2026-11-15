"""
High-level detection API for Crystal Champions.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from .board import find_board
from .cell import (cell_core, cell_blob, cell_silhouette, detect_variants,
                   gem_patch, silhouette)
from .classify import Classifier, build_classifier, get_classifier
from .constants import DEFAULT_REF_DIR, SEEDS, TYPES, TYPE_INDEX


def analyse_board(im: np.ndarray, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Detect the board and read every cell.
    
    Returns dict with board info and cell details.
    """
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


def detect_board(img: np.ndarray, roi: Dict = None, clf: Classifier = None) -> List[List[str]]:
    """
    Detect the 8x8 board and classify every cell.
    
    Args:
        img: BGR image (full frame or cropped board)
        roi: Optional dict with keys left, top, width, height. 
             If provided AND image size matches roi (cropped board),
             uses the ROI grid directly. Otherwise, runs find_board() on img.
        clf: Optional Classifier instance. If None, uses global cached one.
    
    Returns:
        8x8 list of lists of gem type strings ('green', 'red', etc.) or '?' for unknown.
    """
    if clf is None:
        clf = get_classifier()
    
    # Decide whether to use ROI grid or find_board
    use_roi_grid = False
    if roi is not None:
        h, w = img.shape[:2]
        # If image dimensions match ROI (within 5%), assume pre-cropped board
        if abs(w - roi['width']) < roi['width'] * 0.05 and abs(h - roi['height']) < roi['height'] * 0.05:
            use_roi_grid = True
    
    if use_roi_grid:
        # Derive grid from ROI (assumes perfect 8x8 grid on pre-cropped board)
        left, top, w, h = roi['left'], roi['top'], roi['width'], roi['height']
        pitch = w / 8.0
        centres = []
        for r in range(8):
            for c in range(8):
                cx = left + c * pitch + pitch * 0.5
                cy = top + r * pitch + pitch * 0.5
                centres.append((cx, cy))
        board_pitch = pitch
    else:
        # Find board in the image (handles full frames, different scales, margins)
        board = find_board(img)
        if board is None:
            return [['?' for _ in range(8)] for _ in range(8)]
        centres = board["centres"]
        board_pitch = board["pitch"]
    
    # Classify each cell
    grid = [['?' for _ in range(8)] for _ in range(8)]
    for idx, (cx, cy) in enumerate(centres):
        r, c = divmod(idx, 8)
        core, _ = cell_core(img, cx, cy, board_pitch)
        blob = cell_blob(img, cx, cy, board_pitch)
        if core is None or blob is None:
            grid[r][c] = '?'
            continue
        t, _ = clf.rule_predict(core, blob)
        grid[r][c] = t if t is not None else '?'
    
    return grid


# ─── Reference Building (from test_detect.py logic) ──────────────────

def build_references_from_images(images: List[Tuple[str, np.ndarray]]) -> Dict[str, List[Dict]]:
    """Build template references from images using seeds + confident harvest."""
    by_name = {f: im for f, im in images}
    samples = {t: [] for t in TYPES}
    
    # Pass 1: hand-picked seeds
    for t, (frag, (r, c)) in SEEDS.items():
        fname = next((f for f in by_name if frag in f), None)
        if fname is None:
            continue
        res = analyse_board(by_name[fname])
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
        res = analyse_board(im)
        if res is None:
            continue
        for cell in res["cells"]:
            if cell["core"] is None:
                continue
            t, conf = clf0.rule_predict(cell["core"], cell["blob"])
            if t is None or conf < 0.95:
                continue
            if len(samples[t]) >= 12:
                continue
            samples[t].append({"core": cell["core"],
                               "silh": silhouette(cell["blob"]),
                               "patch": cell["patch"]})
    return Classifier.build_refs(samples)


def load_reference_images(ref_dir: Optional[str] = None) -> List[Tuple[str, np.ndarray]]:
    """Load reference images from directory."""
    if ref_dir is None:
        ref_dir = DEFAULT_REF_DIR
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"Reference directory not found: {ref_dir}")
    images = []
    for f in sorted(os.listdir(ref_dir)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            p = os.path.join(ref_dir, f)
            im = cv2.imread(p)
            if im is not None:
                images.append((f, im))
    return images