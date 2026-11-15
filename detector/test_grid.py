"""
Real, runnable verification of the detector lattice + planner.

    python -m detector.test_grid          (from the project root)

Checks, in order:

  1. `calibrate` / `lattice_from_board` on a real screenshot, and that the
     capture region is tight enough to exclude the snackbar overlay.
  2. `read_grid` accuracy against `detector/ground_truth.json` for all six
     screenshots — reported WITH and WITHOUT rows 6-7 (see the warning below).
  3. The ground-truth contamination check itself.
  4. `read_grid` timing over 20 iterations (1 Hz feasibility).
  5. `bot.planner.best_move` timing over 20 calls (600 ms budget).
  6. Cache behaviour: cold build, warm load, corrupt-pickle recovery.

Exit code is 0 only if every hard check passes.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2

from .constants import DEFAULT_REF_DIR, TYPES
from .grid import (CACHE_PATH, GRID_PATH, GRID_SIZE, MARGIN_RATIO, PITCH_MAX,
                   PITCH_MIN, calibrate, capture_region, crop_from_screenshot,
                   get_classifier, lattice_from_board, load_grid, read_grid,
                   save_grid)

GROUND_TRUTH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ground_truth.json")

SUSPECT_ROWS = (6, 7)          # 0-indexed bottom two rows
SNACKBAR_LEFT, SNACKBAR_RIGHT = 12, 12 + 360

FAILURES: List[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    """Print a PASS/FAIL line and remember failures."""
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def load_ground_truth() -> Dict[str, Any]:
    with open(GROUND_TRUTH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def gt_key_for(filename: str, gt: Dict[str, Any]) -> Optional[str]:
    for key in gt:
        if key != "_comment" and key in filename:
            return key
    return None


def accuracy(found: List[List[str]], truth: List[List[str]],
             skip_rows: Tuple[int, ...] = ()) -> Tuple[int, int]:
    ok = total = 0
    for r in range(GRID_SIZE):
        if r in skip_rows:
            continue
        for c in range(GRID_SIZE):
            total += 1
            if found[r][c] == truth[r][c]:
                ok += 1
    return ok, total


# ─── 1. Calibration ───────────────────────────────────────────────────

def check_calibration(shots: List[str]) -> Optional[Dict[str, Any]]:
    rule("1. CALIBRATION (explicit lattice, tight capture region)")
    shot = next(s for s in shots if "112536" in s)
    img = cv2.imread(shot)
    h, w = img.shape[:2]
    print(f"  image: {os.path.basename(shot)}  {w}x{h}")

    t0 = time.perf_counter()
    grid = calibrate(full_bgr=img)
    print(f"  calibrate(): {1000 * (time.perf_counter() - t0):.0f} ms")

    # The pure function must agree with the screenshot-driven path.
    from .board import find_board
    board = find_board(img)
    assert board is not None
    pure = lattice_from_board(board, bounds=(w, h))

    print(f"  pitch_x = {grid['pitch_x']:.3f}   pitch_y = {grid['pitch_y']:.3f}"
          f"   (delta {abs(grid['pitch_x'] - grid['pitch_y']):.3f} px)")
    check(PITCH_MIN <= grid["pitch_x"] <= PITCH_MAX,
          "pitch_x inside 60-110", f"{grid['pitch_x']:.2f}")
    check(PITCH_MIN <= grid["pitch_y"] <= PITCH_MAX,
          "pitch_y inside 60-110", f"{grid['pitch_y']:.2f}")
    check(abs(grid["pitch_x"] - pure["pitch_x"]) < 1e-9
          and abs(grid["x0"] - pure["x0"]) < 1e-9,
          "lattice_from_board(board) == calibrate(image)")

    reg = capture_region(grid)
    print(f"  capture region: left={reg['left']} top={reg['top']} "
          f"{reg['width']}x{reg['height']}")

    # Tightness: the crop must not reach the snackbar parked at x=12..372.
    check(reg["left"] > SNACKBAR_RIGHT,
          "snackbar zone is outside the crop",
          f"crop left={reg['left']} > snackbar right={SNACKBAR_RIGHT}")

    margin_left = grid["x0"] - reg["left"]
    margin_right = (reg["left"] + reg["width"]) - (
        grid["x0"] + (GRID_SIZE - 1) * grid["pitch_x"])
    margin_top = grid["y0"] - reg["top"]
    margin_bottom = (reg["top"] + reg["height"]) - (
        grid["y0"] + (GRID_SIZE - 1) * grid["pitch_y"])
    want_x = MARGIN_RATIO * grid["pitch_x"]
    want_y = MARGIN_RATIO * grid["pitch_y"]
    print(f"  margins  L={margin_left:.1f} R={margin_right:.1f} "
          f"T={margin_top:.1f} B={margin_bottom:.1f}  (target ≈{want_x:.1f}/{want_y:.1f})")
    check(all(0.5 * p - 1.0 <= m <= 0.55 * p + 1.0
              for m, p in ((margin_left, grid["pitch_x"]),
                           (margin_right, grid["pitch_x"]),
                           (margin_top, grid["pitch_y"]),
                           (margin_bottom, grid["pitch_y"]))),
          "margins are ≈0.55*pitch on every side")

    # The crop is what read_grid sees; the lattice must be fully inside it.
    check(reg["left"] <= grid["x0"] - grid["pitch_x"] / 2
          and reg["top"] <= grid["y0"] - grid["pitch_y"] / 2
          and reg["left"] + reg["width"] >= grid["x0"] + 7 * grid["pitch_x"] + grid["pitch_x"] / 2
          and reg["top"] + reg["height"] >= grid["y0"] + 7 * grid["pitch_y"] + grid["pitch_y"] / 2,
          "§0 capture-region invariants hold")

    # ── persistence ──
    save_grid(grid)
    back = load_grid()
    check(back is not None and abs(back["pitch_x"] - grid["pitch_x"]) < 1e-9,
          "save_grid -> load_grid round-trip", GRID_PATH)

    bad = dict(grid); bad["pitch_x"] = 150.0
    tmp = GRID_PATH + ".bad"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(bad, fh)
    check(load_grid(tmp) is None, "load_grid rejects pitch 150")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"x0": 1.0, "y0": 2.0}, fh)
    check(load_grid(tmp) is None, "load_grid rejects missing keys")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({**grid, "left": 5000}, fh)
    check(load_grid(tmp) is None, "load_grid rejects off-screen region")
    os.remove(tmp)

    return grid


# ─── 2. read_grid accuracy ────────────────────────────────────────────

def check_accuracy(shots: List[str], gt: Dict[str, Any]) -> Dict[str, Any]:
    rule("2. read_grid ACCURACY vs detector/ground_truth.json")
    clf = get_classifier()
    print(f"  classifier: {sum(len(v) for v in clf.refs.values())} refs")
    print(f"  {'image':<14} {'all rows':>10} {'rows 0-5':>10}   unknown")

    total_ok = total_all = 0
    trimmed_ok = trimmed_all = 0
    results = {}
    for shot in shots:
        fname = os.path.basename(shot)
        key = gt_key_for(fname, gt)
        if key is None:
            print(f"  {fname}: no ground truth, skipped")
            continue
        img = cv2.imread(shot)
        h, w = img.shape[:2]
        grid = calibrate(full_bgr=img)
        crop = crop_from_screenshot(img, grid)
        rows, stats = read_grid(crop, grid, clf)
        truth = gt[key]["types"]

        ok, n = accuracy(rows, truth)
        ok_t, n_t = accuracy(rows, truth, skip_rows=SUSPECT_ROWS)
        total_ok += ok; total_all += n
        trimmed_ok += ok_t; trimmed_all += n_t
        results[key] = {"rows": rows, "ok": ok, "n": n,
                        "ok_trim": ok_t, "n_trim": n_t,
                        "unknown": stats["unknown"]}
        print(f"  {key:<14} {ok:>3}/{n:<3} ({100 * ok / n:5.1f}%) "
              f"{ok_t:>3}/{n_t:<3} ({100 * ok_t / n_t:5.1f}%)   {stats['unknown']}")

    print(f"  {'TOTAL':<14} {total_ok:>3}/{total_all} ({100 * total_ok / total_all:5.1f}%) "
          f"{trimmed_ok:>3}/{trimmed_all} ({100 * trimmed_ok / trimmed_all:5.1f}%)")
    check(total_ok / total_all >= 0.95, "overall accuracy >= 95% (all rows)")
    print(f"\n  NOTE: 'rows 0-5' excludes the two suspect ground-truth rows "
          f"(see section 3).")
    return results


# ─── 3. ground-truth contamination ────────────────────────────────────

def check_ground_truth_integrity(gt: Dict[str, Any]) -> None:
    rule("3. GROUND TRUTH CONTAMINATION WARNING")
    print("  !!! WARNING: rows 6-7 of detector/ground_truth.json are BYTE-IDENTICAL")
    print("  !!! across 5 of the 6 boards. Those two rows were almost certainly")
    print("  !!! transcribed once and duplicated into the other files, so any")
    print("  !!! accuracy number that includes them is contaminated: a detector")
    print("  !!! that is right on one board is right on five.")
    print("  !!! Both variants are reported above and below. Do not quote the")
    print("  !!! all-rows figure as an independent measurement.\n")

    keys = [k for k in gt if k != "_comment"]
    sigs: Dict[Tuple[str, ...], List[str]] = {}
    for k in keys:
        sig = tuple("|".join(gt[k]["types"][r]) for r in SUSPECT_ROWS)
        sigs.setdefault(sig, []).append(k)

    for sig, group in sorted(sigs.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(group)} board(s) share rows 6-7: {', '.join(sorted(group))}")
        if len(group) > 1:
            for r, row in zip(SUSPECT_ROWS, sig):
                print(f"      row {r}: {row}")

    dup = max(len(g) for g in sigs.values())
    print()
    check(dup >= 5, "reported (not hidden): bottom-two-row duplication",
          f"{dup} of {len(keys)} boards identical")

    if len(keys) > 1:
        same_full = 0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if gt[keys[i]]["types"] == gt[keys[j]]["types"]:
                    same_full += 1
                    print(f"  !! whole boards identical: {keys[i]} == {keys[j]}")
        if same_full == 0:
            print("  (the six boards differ above row 6, so the top 6 rows are "
                  "independent evidence)")


# ─── 4. read_grid timing ──────────────────────────────────────────────

def check_read_timing(shots: List[str]) -> None:
    rule("4. read_grid TIMING (1 Hz loop)")
    clf = get_classifier()
    shot = next(s for s in shots if "112536" in s)
    img = cv2.imread(shot)
    grid = calibrate(full_bgr=img)
    crop = crop_from_screenshot(img, grid)

    read_grid(crop, grid, clf)           # warm numpy/cv2 caches
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        read_grid(crop, grid, clf)
        times.append(1000 * (time.perf_counter() - t0))
    times.sort()
    mean = sum(times) / len(times)
    print(f"  {GRID_SIZE}x{GRID_SIZE} cells, 20 iterations: "
          f"mean {mean:.1f} ms, median {times[10]:.1f} ms, "
          f"min {times[0]:.1f} ms, max {times[-1]:.1f} ms")
    print(f"  budget at 1 Hz: 1000 ms/tick "
          f"(crop grab + read_grid + best_move must fit)")

    check(mean < 250.0, "read_grid well inside a 1 Hz tick", f"{mean:.1f} ms mean")
    # A 2-core i3 is roughly 2-3x slower than this machine. Still comfortable.
    print(f"  on a 2-core i3 (≈2.5x slower): ≈{2.5 * mean:.0f} ms/call "
          f"-> {1000 / (2.5 * mean):.0f} Hz headroom before the planner is added")

    # Prove read_grid uses the frozen lattice and never re-finds the board.
    from . import grid as grid_mod
    original = grid_mod.find_board
    grid_mod.find_board = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("read_grid called find_board"))
    try:
        read_grid(crop, grid, clf)
        called = False
    except AssertionError:
        called = True
    finally:
        grid_mod.find_board = original
    check(not called, "read_grid does NOT call find_board")


# ─── 5. best_move timing ──────────────────────────────────────────────

def check_planner_timing(shots: List[str], gt: Dict[str, Any]) -> None:
    rule("5. best_move TIMING (600 ms budget)")
    from bot.planner import best_move, format_move

    boards: List[Tuple[str, List[List[str]]]] = []
    for shot in shots:
        fname = os.path.basename(shot)
        key = gt_key_for(fname, gt)
        if key is None:
            continue
        img = cv2.imread(shot)
        grid = calibrate(full_bgr=img)
        rows, _ = read_grid(crop_from_screenshot(img, grid), grid)
        boards.append((key, rows))

    all_times: List[float] = []
    for key, rows in boards:
        snapshot = [row[:] for row in rows]
        per_board = []
        for _ in range(4):
            t0 = time.perf_counter()
            res = best_move(rows)
            per_board.append(1000 * (time.perf_counter() - t0))
            if res is None:
                print(f"  {key}: no legal move on this board")
                break
        all_times.extend(per_board)
        if per_board:
            print(f"  {key}: {format_move(res['move'])} +{res['points']} "
                  f"-> combined {res['combined']:.0f}   "
                  f"{sum(per_board) / len(per_board):.1f} ms/call")
        if snapshot != rows:
            FAILURES.append(f"best_move mutated the grid ({key})")

    all_times.sort()
    n = len(all_times)
    mean = sum(all_times) / n
    print(f"\n  {n} calls: mean {mean:.1f} ms, median {all_times[n // 2]:.1f} ms, "
          f"max {all_times[-1]:.1f} ms")
    check(all_times[-1] < 600.0, "every call under 600 ms",
          f"worst {all_times[-1]:.1f} ms")
    check(not any("mutated" in f for f in FAILURES), "input grid is never mutated")


# ─── 6. classifier cache ──────────────────────────────────────────────

def check_cache() -> None:
    rule("6. CLASSIFIER CACHE (config/classifier.pkl)")
    from . import grid as grid_mod

    clf = get_classifier()
    has = os.path.exists(CACHE_PATH)
    size_kb = os.path.getsize(CACHE_PATH) / 1024 if has else 0
    check(has, "cache file exists", f"{CACHE_PATH} ({size_kb:.0f} KB)")

    grid_mod._clf_cache = None
    t0 = time.perf_counter()
    get_classifier()
    warm = 1000 * (time.perf_counter() - t0)
    check(warm < 100.0, "warm load < 100 ms", f"{warm:.1f} ms")

    t0 = time.perf_counter()
    get_classifier(rebuild=True)
    cold = 1000 * (time.perf_counter() - t0)
    print(f"  cold rebuild: {cold:.0f} ms (was ~7 s on the audit machine)")

    # Corrupt pickle must be survivable.
    with open(CACHE_PATH, "rb") as fh:
        good = fh.read()
    with open(CACHE_PATH, "wb") as fh:
        fh.write(b"not a pickle at all")
    grid_mod._clf_cache = None
    try:
        clf2 = get_classifier()
        ok = isinstance(clf2, grid_mod.Classifier) and bool(clf2.refs)
    except Exception as exc:                  # a crash here is a real failure
        ok = False
        print(f"  corrupt-cache recovery raised {exc!r}")
    check(ok, "corrupt pickle is detected and rebuilt")

    # Stale cache (older than a reference image) must be rebuilt.
    import glob as _glob
    refs = sorted(_glob.glob(os.path.join(DEFAULT_REF_DIR, "*.png")))
    oldest = min(os.path.getmtime(p) for p in refs)
    os.utime(CACHE_PATH, (oldest - 7200, oldest - 7200))
    check(not grid_mod._cache_is_fresh(), "stale cache is detected by mtime")

    with open(CACHE_PATH, "wb") as fh:        # restore a good cache
        fh.write(good)
    grid_mod._clf_cache = None


# ─── main ─────────────────────────────────────────────────────────────

def main() -> int:
    shots = sorted(glob.glob(os.path.join(DEFAULT_REF_DIR, "*.png")))
    if not shots:
        print(f"no screenshots in {DEFAULT_REF_DIR}", file=sys.stderr)
        return 2
    print(f"python {sys.version.split()[0]}   mss/cv2/numpy on "
          f"{sys.platform}")
    print(f"screenshots: {len(shots)} in {DEFAULT_REF_DIR}")

    gt = load_ground_truth()
    check_calibration(shots)
    check_accuracy(shots, gt)
    check_ground_truth_integrity(gt)
    check_read_timing(shots)
    check_planner_timing(shots, gt)
    check_cache()

    rule("SUMMARY")
    if FAILURES:
        print(f"  {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"    - {f}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
