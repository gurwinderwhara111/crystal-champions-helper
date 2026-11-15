"""
Self-test for the detector package.

Reads all 6 screenshots from `images of cristels/`, runs calibrate + read_grid,
and prints per-image accuracy vs `detector/ground_truth.json`.

Rows 6-7 (0-indexed) are byte-identical across 5 of the 6 boards — this is
flagged, and accuracy is reported both ways.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .constants import DEFAULT_REF_DIR
from .grid import calibrate, capture_region, read_grid


def _ground_truth() -> Dict[str, List[List[str]]]:
    """Load the hand-labelled grids."""
    gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"ground truth not found: {gt_path}")
    raw = json.load(open(gt_path))
    # Raw format: {"112400": {"types": [[...], ...]}, ...}
    result = {}
    for key, val in raw.items():
        if isinstance(val, dict) and "types" in val:
            result[key] = val["types"]
    return result


def _test_detector() -> int:
    """Run the self-test. Returns exit code."""
    print("=== detector self-test ===")

    # Load reference images
    ref_files = sorted(
        f
        for f in os.listdir(DEFAULT_REF_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not ref_files:
        print(f"ERROR: no reference images in {DEFAULT_REF_DIR}")
        return 1

    print(f"found {len(ref_files)} reference images")

    # Load ground truth
    try:
        gt = _ground_truth()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Track aggregate stats
    total_cells = 0
    total_correct = 0
    total_wrong = 0
    total_refused = 0

    # Track stats excluding rows 6-7 (the contaminated rows)
    total_cells_clean = 0
    total_correct_clean = 0
    total_wrong_clean = 0
    total_refused_clean = 0

    timings: List[float] = []

    for fname in ref_files:
        # Match to ground truth by substring (e.g. "112400" -> "Screenshot 2026-11-13 112400.png")
        gt_key = next((k for k in gt if k in fname), None)
        if gt_key is None:
            print(f"  {fname}: no ground truth, skipping")
            continue

        img = cv2.imread(os.path.join(DEFAULT_REF_DIR, fname))
        if img is None:
            print(f"  {fname}: failed to load")
            continue

        # Calibrate on this image
        try:
            grid = calibrate(full_bgr=img)
        except RuntimeError as e:
            print(f"  {fname}: calibrate FAILED: {e}")
            continue

        # Capture region and read the grid
        region = capture_region(grid)
        crop = img[region["top"] : region["top"] + region["height"],
                   region["left"] : region["left"] + region["width"]]

        t0 = time.perf_counter()
        rows, stats = read_grid(crop, grid)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        # Compare
        expected = gt[gt_key]
        correct = wrong = refused = 0
        correct_clean = wrong_clean = refused_clean = 0

        for r in range(8):
            for c in range(8):
                exp = expected[r][c] if r < len(expected) and c < len(expected[r]) else "?"
                got = rows[r][c] if r < len(rows) and c < len(rows[r]) else "?"

                total_cells += 1
                if r not in (6, 7):  # exclude contaminated rows
                    total_cells_clean += 1

                if got == "?":
                    refused += 1
                    total_refused += 1
                    if r not in (6, 7):
                        refused_clean += 1
                        total_refused_clean += 1
                elif got == exp:
                    correct += 1
                    total_correct += 1
                    if r not in (6, 7):
                        correct_clean += 1
                        total_correct_clean += 1
                else:
                    wrong += 1
                    total_wrong += 1
                    if r not in (6, 7):
                        wrong_clean += 1
                        total_wrong_clean += 1

        summary = f"+{correct} -{wrong} ?{refused}"
        print(f"  {fname}: {summary}")

    # Summary
    if total_cells == 0:
        print("ERROR: no cells tested")
        return 1

    print(f"\n--- Results (all 64 cells) ---")
    print(f"  cells:   {total_cells}")
    print(f"  correct: {total_correct} ({100.0 * total_correct / total_cells:.2f}%)")
    print(f"  wrong:   {total_wrong}")
    print(f"  refused: {total_refused}")

    if total_cells_clean > 0:
        print(f"\n--- Results (excluding rows 6-7, which are duplicated in GT) ---")
        print(f"  cells:   {total_cells_clean}")
        print(
            f"  correct: {total_correct_clean} ({100.0 * total_correct_clean / total_cells_clean:.2f}%)"
        )
        print(f"  wrong:   {total_wrong_clean}")
        print(f"  refused: {total_refused_clean}")

    if timings:
        print(
            f"\nread_grid timing: mean {1000 * sum(timings) / len(timings):.1f} ms, "
            f"max {1000 * max(timings):.1f} ms"
        )

    # Budget check
    if max(timings) > 1.0:
        print(f"ERROR: read_grid too slow ({1000 * max(timings):.0f} ms > 1000 ms budget)")
        return 1

    # Quality check
    if total_wrong > 0:
        print(f"ERROR: {total_wrong} wrong classifications")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_test_detector())
