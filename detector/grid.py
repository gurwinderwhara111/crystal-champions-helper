"""
`detector.grid` — the 1 Hz manual-loop grid reader.

This module is the ONLY place that knows the board lattice. It has two jobs:

  * `calibrate()` — run the expensive `find_board()` once and freeze the
    result as an explicit lattice (`x0, y0, pitch_x, pitch_y`) plus a tight
    capture region.
  * `read_grid()` — read all 64 cells from a crop WITHOUT calling
    `find_board()`, which is what makes a 1 Hz loop affordable.

Coordinate system (see docs/CONTRACTS.md §0)
-------------------------------------------
    screen px : sx = x0 + c * pitch_x , sy = y0 + r * pitch_y
    crop px   : cx = (x0 - left) + c * pitch_x , cy = (y0 - top) + r * pitch_y

`x0, y0` is the centre of cell (0, 0). `left, top` is the top-left of the
capture region. The region is TIGHT (margin = 0.55 * pitch) so that the
snackbar overlay on the left of the screen can never fall inside the crop.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .board import find_board
from .cell import cell_blob, cell_core
from .classify import Classifier, build_classifier
from .constants import DEFAULT_REF_DIR, TYPES

# ─── Paths (project root is one level above this package) ─────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
GRID_PATH = os.path.join(CONFIG_DIR, "grid.json")
CACHE_PATH = os.path.join(CONFIG_DIR, "classifier.pkl")

# ─── Sanity limits ────────────────────────────────────────────────────

PITCH_MIN = 60.0
PITCH_MAX = 110.0
GRID_SIZE = 8
MARGIN_RATIO = 0.55        # capture margin, in cells, around the board

REF_IMAGE_EXT = (".png", ".jpg", ".jpeg")


def _warn(msg: str) -> None:
    """Log to stderr. Style rule: never `except: pass`."""
    sys.stderr.write(f"[detector.grid] {msg}\n")


def _ref_images() -> List[str]:
    """Absolute paths of the reference screenshots, sorted."""
    if not os.path.isdir(DEFAULT_REF_DIR):
        raise FileNotFoundError(f"Reference directory not found: {DEFAULT_REF_DIR}")
    return [
        os.path.join(DEFAULT_REF_DIR, f)
        for f in sorted(os.listdir(DEFAULT_REF_DIR))
        if f.lower().endswith(REF_IMAGE_EXT)
    ]


def _new_mss():
    """Create an mss grabber, tolerating old and new API names."""
    import mss
    factory = getattr(mss, "MSS", None) or mss.mss
    return factory()


def _screen_size() -> Optional[Tuple[int, int]]:
    """Primary monitor size in pixels, or None when there is no display."""
    try:
        import mss  # imported lazily: keeps the package importable headless
    except ImportError:
        _warn("mss is not installed; cannot query screen size")
        return None
    try:
        with _new_mss() as sct:
            mon = sct.monitors[1]
            return int(mon["width"]), int(mon["height"])
    except Exception as exc:                    # pragma: no cover - env dependent
        _warn(f"cannot query screen size: {exc!r}")
        return None


# ─── Lattice ──────────────────────────────────────────────────────────

def lattice_from_board(board: Dict[str, Any],
                       bounds: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """
    Pure function: fitted `find_board` result -> `grid` dict.

    `board` must contain 8 `xs` and 8 `ys` grid-line centres (as returned by
    `detector.board.find_board`). `bounds` = (width, height) of the coordinate
    space the board lives in (the screenshot / the screen). The capture region
    is clamped to those bounds.

    Raises RuntimeError if the lattice is not physically plausible.
    """
    xs = np.asarray(board["xs"], dtype=np.float64).ravel()
    ys = np.asarray(board["ys"], dtype=np.float64).ravel()
    if xs.size != GRID_SIZE or ys.size != GRID_SIZE:
        raise RuntimeError(
            f"board lattice needs {GRID_SIZE} x-lines and {GRID_SIZE} y-lines, "
            f"got {xs.size} and {ys.size}")

    # Separate pitches: the real board is 81.55 px wide and 81.49 px tall
    # and the fitted lines spread by up to ~5 px. Never collapse to one value.
    pitch_x = float(np.mean(np.diff(xs)))
    pitch_y = float(np.mean(np.diff(ys)))

    for name, p in (("pitch_x", pitch_x), ("pitch_y", pitch_y)):
        if not (PITCH_MIN <= p <= PITCH_MAX):
            raise RuntimeError(
                f"{name}={p:.2f} outside sanity range {PITCH_MIN:.0f}-{PITCH_MAX:.0f}")

    # x0/y0 = least-squares intercept of the line index->position fit with the
    # slope pinned to the measured pitch. More robust than xs[0] alone.
    idx = np.arange(GRID_SIZE, dtype=np.float64)
    x0 = float(np.mean(xs - pitch_x * idx))
    y0 = float(np.mean(ys - pitch_y * idx))

    margin_x = MARGIN_RATIO * pitch_x
    margin_y = MARGIN_RATIO * pitch_y

    left = int(round(x0 - margin_x))
    top = int(round(y0 - margin_y))
    right = int(round(x0 + (GRID_SIZE - 1) * pitch_x + margin_x))
    bottom = int(round(y0 + (GRID_SIZE - 1) * pitch_y + margin_y))

    if bounds is not None:
        bw, bh = int(bounds[0]), int(bounds[1])
        # The full lattice must be inside the bounds; clamping is only allowed
        # to shave the margin, never the board itself.
        need_left = x0 - pitch_x / 2.0
        need_top = y0 - pitch_y / 2.0
        need_right = x0 + (GRID_SIZE - 1) * pitch_x + pitch_x / 2.0
        need_bottom = y0 + (GRID_SIZE - 1) * pitch_y + pitch_y / 2.0
        if need_left < 0 or need_top < 0 or need_right > bw or need_bottom > bh:
            raise RuntimeError(
                f"board is outside the capture bounds: need "
                f"({need_left:.0f},{need_top:.0f})-({need_right:.0f},{need_bottom:.0f}) "
                f"inside 0,0-{bw},{bh}")
        left = max(0, left)
        top = max(0, top)
        right = min(bw, right)
        bottom = min(bh, bottom)

    grid = {
        "x0": x0,
        "y0": y0,
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
        "cols": GRID_SIZE,
        "rows": GRID_SIZE,
    }
    _check_invariants(grid)
    return grid


def _check_invariants(grid: Dict[str, Any]) -> None:
    """The §0 invariants. Raises RuntimeError on violation."""
    x0, y0 = grid["x0"], grid["y0"]
    px, py = grid["pitch_x"], grid["pitch_y"]
    left, top = grid["left"], grid["top"]
    right, bottom = left + grid["width"], top + grid["height"]

    if not (left <= x0 - px / 2.0 and top <= y0 - py / 2.0):
        raise RuntimeError("capture region does not cover the top-left cell")
    if not (right >= x0 + (GRID_SIZE - 1) * px + px / 2.0):
        raise RuntimeError("capture region does not cover the right column")
    if not (bottom >= y0 + (GRID_SIZE - 1) * py + py / 2.0):
        raise RuntimeError("capture region does not cover the bottom row")
    if grid["width"] <= 0 or grid["height"] <= 0:
        raise RuntimeError("capture region is empty")


# ─── Calibration ──────────────────────────────────────────────────────

def _grab_primary() -> np.ndarray:
    """Screenshot of the primary monitor as BGR."""
    with _new_mss() as sct:
        mon = sct.monitors[1]
        raw = np.asarray(sct.grab(mon))          # BGRA
    return np.ascontiguousarray(raw[..., :3])


def calibrate(full_bgr: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Find the board and return a `grid` dict (see docs/CONTRACTS.md §0).

    If `full_bgr` is None the primary monitor is grabbed with mss.
    Raises RuntimeError("board not found") when `find_board` fails.
    """
    if full_bgr is None:
        full_bgr = _grab_primary()
    h, w = full_bgr.shape[:2]
    board = find_board(full_bgr)
    if board is None:
        raise RuntimeError("board not found")
    return lattice_from_board(board, bounds=(w, h))


# ─── Persistence ──────────────────────────────────────────────────────

def save_grid(grid: Dict[str, Any], path: str = GRID_PATH) -> None:
    """Write the grid dict as JSON (creating config/ if needed)."""
    _check_invariants(grid)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(grid, fh, indent=2)
        fh.write("\n")


_GRID_KEYS = ("x0", "y0", "pitch_x", "pitch_y",
              "left", "top", "width", "height", "cols", "rows")


def load_grid(path: str = GRID_PATH) -> Optional[Dict[str, Any]]:
    """
    Read + validate a grid dict. Returns None if the file is missing, is not
    JSON, is missing keys, has a pitch outside 60-110, or if the capture
    region is not inside the screen bounds.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            grid = json.load(fh)
    except (OSError, ValueError) as exc:
        if os.path.exists(path):
            _warn(f"cannot read {path}: {exc!r}")
        return None

    if not isinstance(grid, dict):
        _warn(f"{path}: not a JSON object")
        return None

    missing = [k for k in _GRID_KEYS if k not in grid]
    if missing:
        _warn(f"{path}: missing keys {missing}")
        return None

    try:
        for k in _GRID_KEYS:
            grid[k] = float(grid[k]) if k not in ("left", "top", "width", "height",
                                                  "cols", "rows") else int(grid[k])
        if grid["cols"] != GRID_SIZE or grid["rows"] != GRID_SIZE:
            raise ValueError(f"expected an {GRID_SIZE}x{GRID_SIZE} grid, "
                             f"got {grid['cols']}x{grid['rows']}")
        for name in ("pitch_x", "pitch_y"):
            if not (PITCH_MIN <= grid[name] <= PITCH_MAX):
                raise ValueError(
                    f"{name}={grid[name]:.2f} outside {PITCH_MIN:.0f}-{PITCH_MAX:.0f}")
        if grid["width"] <= 0 or grid["height"] <= 0:
            raise ValueError("empty capture region")
        _check_invariants(grid)
        size = _screen_size()
        if size is not None:
            sw, sh = size
            if (grid["left"] < 0 or grid["top"] < 0
                    or grid["left"] + grid["width"] > sw
                    or grid["top"] + grid["height"] > sh):
                raise ValueError(
                    f"capture region {grid['left']},{grid['top']} "
                    f"{grid['width']}x{grid['height']} is outside the screen "
                    f"{sw}x{sh} - recalibrate")
    except (TypeError, ValueError, RuntimeError) as exc:
        _warn(f"{path}: invalid grid ({exc})")
        return None

    return grid


def capture_region(grid: Dict[str, Any]) -> Dict[str, int]:
    """mss monitor dict for the board region (see §0)."""
    return {
        "left": int(round(grid["left"])),
        "top": int(round(grid["top"])),
        "width": int(round(grid["width"])),
        "height": int(round(grid["height"])),
    }


# ─── Classifier cache ─────────────────────────────────────────────────

_clf_cache: Optional[Classifier] = None


def _cache_is_fresh() -> bool:
    """True when the pickle exists and is newer than every reference image."""
    if not os.path.exists(CACHE_PATH):
        return False
    try:
        cache_mtime = os.path.getmtime(CACHE_PATH)
        newest = max(os.path.getmtime(p) for p in _ref_images())
    except OSError as exc:
        _warn(f"cannot stat classifier cache: {exc!r}")
        return False
    return cache_mtime >= newest


def _load_cached_classifier() -> Optional[Classifier]:
    try:
        with open(CACHE_PATH, "rb") as fh:
            clf = pickle.load(fh)
    except Exception as exc:                    # corrupt / stale object graph
        _warn(f"classifier cache unusable ({exc!r}); rebuilding")
        return None
    if not isinstance(clf, Classifier) or not clf.refs:
        _warn("classifier cache does not contain a usable Classifier; rebuilding")
        return None
    return clf


def _save_cached_classifier(clf: Classifier) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(clf, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, CACHE_PATH)
    except OSError as exc:
        _warn(f"cannot write classifier cache: {exc!r}")


def get_classifier(rebuild: bool = False) -> Classifier:
    """
    Cached classifier: ~7 s cold build, ~40 ms warm load.

    Invalidated when `rebuild=True`, when the pickle is older than any file in
    `images of cristels/`, or when the pickle does not load.
    """
    global _clf_cache
    if rebuild:
        _clf_cache = None
    elif _clf_cache is not None:
        return _clf_cache
    elif _cache_is_fresh():
        clf = _load_cached_classifier()
        if clf is not None:
            _clf_cache = clf
            return clf

    clf = build_classifier()
    _save_cached_classifier(clf)
    _clf_cache = clf
    return clf


# ─── Reading ──────────────────────────────────────────────────────────

def read_grid(crop: np.ndarray, grid: Dict[str, Any],
              clf: Optional[Classifier] = None) -> Tuple[List[List[str]], Dict[str, Any]]:
    """
    Read the 8x8 board out of `crop` using the frozen lattice.

    `crop` is a BGR array of exactly `capture_region(grid)`. `find_board` is
    deliberately NOT called: the lattice is already known.

    Returns (rows, stats) where rows is 8x8 long type names or '?' and stats
    is {"unknown": int, "known": int, "cells": [[...]]}.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        raise ValueError("read_grid got an empty crop")
    if clf is None:
        clf = get_classifier()

    x0, y0 = float(grid["x0"]), float(grid["y0"])
    pitch_x, pitch_y = float(grid["pitch_x"]), float(grid["pitch_y"])
    left, top = int(round(grid["left"])), int(round(grid["top"]))
    pitch = 0.5 * (pitch_x + pitch_y)          # extractors take one pitch

    rows: List[List[str]] = [["?"] * GRID_SIZE for _ in range(GRID_SIZE)]
    unknown = 0
    for r in range(GRID_SIZE):
        cy = (y0 - top) + r * pitch_y
        for c in range(GRID_SIZE):
            cx = (x0 - left) + c * pitch_x
            core, _ = cell_core(crop, cx, cy, pitch)
            if core is None:
                unknown += 1
                continue
            blob = cell_blob(crop, cx, cy, pitch)
            t, _conf = clf.rule_predict(core, blob)
            if t is None or t not in TYPES:
                unknown += 1
                continue
            rows[r][c] = t

    cells = [row[:] for row in rows]
    stats = {
        "unknown": unknown,
        "known": GRID_SIZE * GRID_SIZE - unknown,
        "cells": cells,
    }
    return rows, stats


def crop_from_screenshot(img: np.ndarray, grid: Dict[str, Any]) -> np.ndarray:
    """Test helper: cut `capture_region(grid)` out of a full screenshot."""
    reg = capture_region(grid)
    return img[reg["top"]:reg["top"] + reg["height"],
               reg["left"]:reg["left"] + reg["width"]].copy()


# ─── Self-test ────────────────────────────────────────────────────────

def _main() -> int:
    import glob

    print("=== detector.grid self-test ===")
    shots = sorted(glob.glob(os.path.join(DEFAULT_REF_DIR, "*.png")))
    if not shots:
        print(f"no reference screenshots in {DEFAULT_REF_DIR}", file=sys.stderr)
        return 1

    import cv2
    shot = shots[-2] if len(shots) > 1 else shots[0]
    img = cv2.imread(shot)
    print(f"screenshot: {os.path.basename(shot)}  {img.shape[1]}x{img.shape[0]}")

    t0 = time.perf_counter()
    grid = calibrate(full_bgr=img)
    print(f"calibrate: {1000 * (time.perf_counter() - t0):.0f} ms")
    for k in ("x0", "y0", "pitch_x", "pitch_y", "left", "top", "width", "height"):
        print(f"  {k:<8} {grid[k]:.2f}")
    reg = capture_region(grid)
    if reg["left"] <= 12 + 360:
        print(f"  ! capture region starts at x={reg['left']} - snackbar "
              f"(x=12..372) may overlap", file=sys.stderr)
    else:
        print(f"  snackbar zone x=12..372 is clear of the crop (left={reg['left']})")

    t0 = time.perf_counter()
    clf = get_classifier()
    print(f"classifier: {1000 * (time.perf_counter() - t0):.0f} ms  "
          f"({sum(len(v) for v in clf.refs.values())} refs)")

    crop = crop_from_screenshot(img, grid)
    print(f"crop: {crop.shape[1]}x{crop.shape[0]}")
    t0 = time.perf_counter()
    rows, stats = read_grid(crop, grid, clf)
    print(f"read_grid: {1000 * (time.perf_counter() - t0):.1f} ms  "
          f"unknown={stats['unknown']}")
    for r in rows:
        print("  " + " ".join(f"{t[:5]:<5}" for t in r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
