"""
Constants for Crystal Champions gem detection.

This module is the SINGLE SOURCE OF TRUTH for tunables. It deliberately
re-exports nothing and defines `__all__`, so `from .constants import *`
no longer leaks `np` (or anything else) into consumers. New code should
still prefer explicit imports.
"""

from __future__ import annotations
import os

# ─── Gem Types ──────────────────────────────────────────────────────────

# The 7 gem types in the game, in a stable order.
TYPES = ["green", "red", "blue", "purple", "yellow", "orange", "white"]
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}

# OpenCV HSV hue centres measured from the actual game art (H is 0..179).
HUE_CENTRE = {
    "red":    1.5,
    "orange": 13.0,
    "yellow": 24.0,
    "green":  60.0,
    "blue":   104.0,
    "purple": 153.0,
}

# Silhouette signature (extent = area/bbox_area) per shape family.
# MEASURED on the real game art across 384 hand-labelled cells.
SHAPE_FAMILY = {
    "green":  "blocky",
    "red":    "mid",
    "blue":   "ball",
    "purple": "sharp",
    "yellow": "sharp",
    "orange": "blocky",
    "white":  "pearl",
}

# Which shape families are acceptable for each colour.
# Only red and purple share an outline (both are diamonds).
FAMILY_SET = {
    "green":  {"blocky"},
    "orange": {"blocky"},
    "blue":   {"ball", "blocky", "pearl"},
    "white":  {"pearl", "ball", "blocky", "mid"},
    "red":    {"mid", "sharp"},
    "purple": {"sharp"},
    "yellow": {"sharp"},
}

# ─── Detection Parameters ──────────────────────────────────────────────

CORE_SIZE = 40          # normalised gem core patch size (px)
PATCH_SIZE = 48         # normalised full-gem patch size (px)

# Gem mask thresholds
GEM_BRIGHT_V = 185
GEM_BRIGHT_S = 50
PEARL_V = 195
PEARL_S = 50

# Core hue detection thresholds
CORE_SAT_HIGH = 130
CORE_VAL_HIGH = 85
CORE_SAT_MID = 90
CORE_VAL_MID = 80
CORE_SAT_LOW = 40
CORE_VAL_LOW = 60
CORE_MIN_PIXELS_HIGH = 25
CORE_MIN_PIXELS_MID = 15
CORE_MIN_PIXELS_LOW = 10

# Hue band edges (for confidence calculation)
HUE_EDGES = {
    "red":    (172.0, 6.5, 6.5),
    "orange": (6.5, 21.0, 6.5),
    "yellow": (21.0, 33.0, 6.5),
    "green":  (33.0, 82.0, 8.0),
    "blue":   (82.0, 128.0, 8.0),
    "purple": (128.0, 172.0, 8.0),
}

# Whiteness threshold for pearl
WHITENESS_PEARL = 0.58
WHITENESS_HIGH_CONF = 0.75
WHITENESS_LOW_CONF = 0.75

# Silhouette extent bands (measured on 384 labelled cells)
EXTENT_BANDS = {
    "sharp":   0.575,   # purple, yellow
    "mid":     0.635,   # red
    "pearl":   0.695,   # white
    "ball":    0.765,   # blue
    "blocky":  1.0,     # green, orange
}

SHAPE_FAMILY_FROM_EXTENT = {
    "sharp":   ("purple", "yellow"),
    "mid":     ("red",),
    "pearl":   ("white",),
    "ball":    ("blue",),
    "blocky":  ("green", "orange"),
}

# ─── Variant Detection (cage / swirl) ──────────────────────────────────

# Body mask thresholds (high to exclude cage/swirl)
BODY_V_HIGH = 210
BODY_S_HIGH = 50
BODY_V_MID = 200
BODY_S_LOW = 50
BODY_DILATE_KERNEL = 5

# Wrapper detection
WRAPPER_OPEN_KERNEL = 3
WRAPPER_MIN_AREA_RATIO = 0.015
WRAPPER_MIN_SIDE_RATIO = 0.72
WRAPPER_ASPECT_RANGE = (0.55, 1.85)

# Red swirl HSV range
SWIRL_HUE_LOW = 8
SWIRL_HUE_HIGH = 172
SWIRL_SAT = 110
SWIRL_VAL = 60

# Tan cage HSV range
CAGE_HUE_LOW = 8
CAGE_HUE_HIGH = 32
CAGE_SAT_LOW = 45
CAGE_SAT_HIGH = 160
CAGE_VAL_LOW = 50
CAGE_VAL_HIGH = 225

# ─── Board Finding ────────────────────────────────────────────────────

PITCH_SEARCH_MIN = 38.0
PITCH_SEARCH_MAX = 165.0
PITCH_SEARCH_STEP = 1.0
MIN_BLOBS = 20
MIN_COVERAGE = 0.0
GRID_TOLERANCE_RATIO = 0.30
EVENNESS_WEIGHT = 3.0

# Grid refinement
SNAP_WINDOW_RATIO = 26  # pixels around grid line

# Cell extraction ratios
CORE_HALF_RATIO = 0.22
BLOB_HALF_RATIO = 0.46
SILHOUETTE_HALF_RATIO = 0.44
PATCH_HALF_RATIO = 0.44
VARIANT_HALF_RATIO = 0.58

# Minimum areas
MIN_BLOB_AREA = 60
MIN_SILHOUETTE_AREA = 200
MIN_BLOB_AREA_RATIO = 500
MAX_BLOB_AREA_RATIO = 6000
BLOB_ASPECT_RANGE = (0.5, 2.0)
BLOB_FILL_RATIO = 0.35

# ─── Classifier ────────────────────────────────────────────────────────

MAX_REFS_PER_TYPE = 12
SEED_CONFIDENCE_THRESHOLD = 0.95
HARVEST_CONFIDENCE_THRESHOLD = 0.95

# ─── SEEDS for reference building ──────────────────────────────────────

# Keys are (filename substring, (row, col)) with 1-based row/col.
# Verified against ground truth grid of Screenshot 2026-11-13 112536.png
SEEDS = {
    "green":  ("112536", (1, 6)),
    "red":    ("112536", (1, 4)),
    "blue":   ("112536", (2, 1)),
    "purple": ("112536", (1, 8)),
    "yellow": ("112536", (2, 7)),
    "orange": ("112536", (1, 7)),
    "white":  ("112536", (2, 6)),
}

# ─── Reference Image Directory ────────────────────────────────────────

# Directory name, relative to the project root.
REF_DIR_NAME = "images of cristels"

# Absolute path of the reference screenshots. `detector/` lives one level
# below the project root, so a single '..' is correct.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REF_DIR = os.path.join(_PROJECT_ROOT, REF_DIR_NAME)

# ─── Public API ───────────────────────────────────────────────────────

# `np` and `_PROJECT_ROOT` are intentionally NOT exported.
__all__ = [
    # types
    "TYPES", "TYPE_INDEX", "HUE_CENTRE", "SHAPE_FAMILY", "FAMILY_SET",
    # detection parameters
    "CORE_SIZE", "PATCH_SIZE",
    "GEM_BRIGHT_V", "GEM_BRIGHT_S", "PEARL_V", "PEARL_S",
    "CORE_SAT_HIGH", "CORE_VAL_HIGH", "CORE_SAT_MID", "CORE_VAL_MID",
    "CORE_SAT_LOW", "CORE_VAL_LOW",
    "CORE_MIN_PIXELS_HIGH", "CORE_MIN_PIXELS_MID", "CORE_MIN_PIXELS_LOW",
    "HUE_EDGES",
    "WHITENESS_PEARL", "WHITENESS_HIGH_CONF", "WHITENESS_LOW_CONF",
    "EXTENT_BANDS", "SHAPE_FAMILY_FROM_EXTENT",
    # variant detection
    "BODY_V_HIGH", "BODY_S_HIGH", "BODY_V_MID", "BODY_S_LOW",
    "BODY_DILATE_KERNEL",
    "WRAPPER_OPEN_KERNEL", "WRAPPER_MIN_AREA_RATIO", "WRAPPER_MIN_SIDE_RATIO",
    "WRAPPER_ASPECT_RANGE",
    "SWIRL_HUE_LOW", "SWIRL_HUE_HIGH", "SWIRL_SAT", "SWIRL_VAL",
    "CAGE_HUE_LOW", "CAGE_HUE_HIGH", "CAGE_SAT_LOW", "CAGE_SAT_HIGH",
    "CAGE_VAL_LOW", "CAGE_VAL_HIGH",
    # board finding
    "PITCH_SEARCH_MIN", "PITCH_SEARCH_MAX", "PITCH_SEARCH_STEP",
    "MIN_BLOBS", "MIN_COVERAGE", "GRID_TOLERANCE_RATIO", "EVENNESS_WEIGHT",
    "SNAP_WINDOW_RATIO",
    # cell extraction
    "CORE_HALF_RATIO", "BLOB_HALF_RATIO", "SILHOUETTE_HALF_RATIO",
    "PATCH_HALF_RATIO", "VARIANT_HALF_RATIO",
    "MIN_BLOB_AREA", "MIN_SILHOUETTE_AREA", "MIN_BLOB_AREA_RATIO",
    "MAX_BLOB_AREA_RATIO", "BLOB_ASPECT_RANGE", "BLOB_FILL_RATIO",
    # classifier
    "MAX_REFS_PER_TYPE", "SEED_CONFIDENCE_THRESHOLD",
    "HARVEST_CONFIDENCE_THRESHOLD", "SEEDS",
    # reference images
    "REF_DIR_NAME", "DEFAULT_REF_DIR",
]
