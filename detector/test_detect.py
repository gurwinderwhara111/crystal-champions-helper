"""
test_detect.py - run the detector over the screenshots in
"images of cristels" and print the 8x8 grid it reads for each.

Usage:
    python test_detect.py            # detect + print grids
    python test_detect.py --save     # also dump annotated crop sheets
"""

import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemvision as gv

IMG_DIR = r"D:\random works by gs\Crystal Champions helper\images of cristels"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hand-picked, unambiguous seed cells used to build the template references.
# Keys are (filename substring, (row, col)) with 1-based row/col.
# Verified against the ground truth grid of Screenshot 2026-11-13 112536.png:
#   row1 = G O B B G O R P      row2 = B R O O R W Y P
SEEDS = {
    "green":  ("112536", (1, 6)),
    "red":    ("112536", (1, 4)),
    "blue":   ("112536", (2, 1)),
    "purple": ("112536", (1, 8)),
    "yellow": ("112536", (2, 7)),
    "orange": ("112536", (1, 7)),
    "white":  ("112536", (2, 6)),
}

GLYPH = {"green": "G", "red": "R", "blue": "B", "purple": "P",
         "yellow": "Y", "orange": "O", "white": "W", None: "?"}


def load_images():
    out = []
    for f in sorted(os.listdir(IMG_DIR)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            p = os.path.join(IMG_DIR, f)
            im = cv2.imread(p)
            if im is not None:
                out.append((f, im))
    return out


def analyse(im, verbose=False):
    """Detect the board and read every cell. Returns dict."""
    board = gv.find_board(im, verbose=verbose)
    if board is None:
        return None
    cells = []
    for (cx, cy) in board["centres"]:
        core, _ = gv.cell_core(im, cx, cy, board["pitch"])
        blob = gv.cell_blob(im, cx, cy, board["pitch"])
        flags = gv.detect_variants(im, cx, cy, board["pitch"],
                                   blob["bbox"] if blob else None)
        cells.append({"cx": cx, "cy": cy, "core": core, "blob": blob,
                      "flags": flags})
    return {"board": board, "cells": cells}


def build_references(images):
    """
    Collect reference cores + silhouettes.

    Two passes:
      1. hand-picked seeds (one unambiguous example of each of the 7 types)
      2. every cell whose RULE classification is high confidence, which gives
         many examples per type and makes the cross-checkers meaningful
    """
    by_name = {f: im for f, im in images}
    samples = {t: [] for t in gv.TYPES}
    for t, (frag, (r, c)) in SEEDS.items():
        fname = next((f for f in by_name if frag in f), None)
        if fname is None:
            print(f"  !! seed file for {t} not found")
            continue
        res = analyse(by_name[fname])
        if res is None:
            print(f"  !! could not locate board in {fname}")
            continue
        idx = (r - 1) * 8 + (c - 1)
        cell = res["cells"][idx]
        if cell["core"] is None:
            print(f"  !! seed {t} has no core")
            continue
        samples[t].append({"core": cell["core"], "silh": gv.silhouette(cell["blob"])})
        print(f"  seed {t:7s} <- {fname} r{r}c{c}")

    # pass 2: harvest confident cells
    clf0 = gv.Classifier(gv.Classifier.build_refs(samples))
    added = 0
    for f, im in images:
        res = analyse(im)
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
                               "silh": gv.silhouette(cell["blob"])})
            added += 1
    print(f"  harvested {added} additional confident reference cells")
    return gv.Classifier.build_refs(samples)


def print_grid(res, tag, labels):
    print(f"\n  {tag}")
    for r in range(8):
        row = ""
        for c in range(8):
            cell = res["cells"][r * 8 + c]
            lab = labels[r * 8 + c]
            g = GLYPH.get(lab, "?")
            fl = cell["flags"]
            if "cage" in fl:
                g = g.lower()
            if "swirl" in fl:
                g = g + "*"
            row += g + " "
        print("   ", row)


def main():
    images = load_images()
    print(f"loaded {len(images)} images from {IMG_DIR}")

    print("\nbuilding template references from hand-picked seeds...")
    refs = build_references(images)
    clf = gv.Classifier(refs)

    results = {}
    for f, im in images:
        print(f"\n=== {f}  ({im.shape[1]}x{im.shape[0]}) ===")
        res = analyse(im, verbose=True)
        if res is None:
            print("  BOARD NOT FOUND")
            continue
        labels_rule, labels_ncc, agree = [], [], 0
        for cell in res["cells"]:
            t1, c1 = clf.rule_predict(cell["core"], cell["blob"])
            t2, c2 = clf.ncc_predict(cell["core"])
            labels_rule.append(t1)
            labels_ncc.append(t2)
            if t1 is not None and t1 == t2:
                agree += 1
        print_grid(res, "RULE (hue + shape)", labels_rule)
        print_grid(res, "NCC  (template cores)", labels_ncc)
        n = len(res["cells"])
        print(f"    rule-vs-ncc agreement: {agree}/{n} = {100.0*agree/n:.1f}%")
        results[f] = {"rule": labels_rule, "ncc": labels_ncc,
                      "flags": [sorted(c["flags"]) for c in res["cells"]]}

    with open(os.path.join(OUT_DIR, "detection_results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {os.path.join(OUT_DIR, 'detection_results.json')}")


if __name__ == "__main__":
    main()
