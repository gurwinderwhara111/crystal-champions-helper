"""
make_atlas.py - build a visual reference atlas of the 7 crystal types.

For every screenshot in "images of cristels" this script
  1. locates the 8x8 board (gemvision.find_board),
  2. reads the ground-truth type of every cell (ground_truth.json),
  3. picks ONE clean, unobstructed example of each of the 7 types
     (no wooden cage, no red swirl aura, solid un-clipped silhouette),
  4. writes a magnified photo montage and a pure-silhouette montage into
     docs/shapes/.

It also writes the combined all-types montages and the variant examples
(wooden cage, red swirl aura).

Nothing outside docs/ is touched. detector/gemvision.py and
detector/ground_truth.json are read-only inputs.

Usage:
    python detector/make_atlas.py
"""

import os
import sys
import json
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gemvision as gv  # noqa: E402

ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "images of cristels")
GT_PATH = os.path.join(HERE, "ground_truth.json")
DOCS = os.path.join(ROOT, "docs")
OUT = os.path.join(DOCS, "shapes")

TYPES = list(gv.TYPES)

# The screenshot used for the combined "all types" montages (the full
# desktop silver-chrome capture).
BEST_IMAGE = "112536"

# ---------------------------------------------------------------- shape names
# Precise, descriptive names of the OUTLINE. Written from the measured
# geometry (convex-hull vertex count, extent) plus visual inspection of the
# artwork; see docs/SHAPE_ATLAS.md for the numbers behind each name.
SHAPE_NAME = {
    "green":  "rounded cube / square with chamfered corners",
    "red":    "brilliant-cut diamond (pointed crown + pointed pavilion)",
    "blue":   "faceted ball / rounded decagon",
    "purple": "square rotated 45 deg (rhombus), one corner clipped",
    "yellow": "triangle / pyramid",
    "orange": "hexagon, point at the top",
    "white":  "sphere / circle",
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- helpers

def load_images():
    """[(stamp, path, bgr)] for every screenshot, sorted by stamp."""
    out = []
    for f in sorted(os.listdir(IMG_DIR)):
        if not f.lower().endswith(".png"):
            continue
        stamp = f.split(" ")[-1].replace(".png", "")
        im = cv2.imread(os.path.join(IMG_DIR, f))
        if im is not None:
            out.append((stamp, os.path.join(IMG_DIR, f), im))
    return out


def cell_window(im, cx, cy, pitch, frac=0.46):
    """The square crop of one cell, as used by the detector."""
    half = int(pitch * frac)
    h, w = im.shape[:2]
    x0, x1 = max(0, int(cx - half)), min(w, int(cx + half))
    y0, y1 = max(0, int(cy - half)), min(h, int(cy + half))
    return im[y0:y1, x0:x1].copy()


def clean_mask(sil):
    """Filled binary mask of the silhouette, cropped to its bbox (bool)."""
    m = np.zeros(sil["win_shape"], np.uint8)
    cv2.drawContours(m, [sil["contour"]], -1, 255, -1)
    x, y, w, h = sil["bbox"]
    crop = m[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    return crop > 0


def window_mask(sil):
    """Filled binary mask of the silhouette in the full cell window (bool)."""
    m = np.zeros(sil["win_shape"], np.uint8)
    cv2.drawContours(m, [sil["contour"]], -1, 255, -1)
    return m > 0


def hull_vertices(mask, eps=0.03):
    """
    Number of vertices of the convex hull of a mask, approximated with
    approxPolyDP at `eps` * hull perimeter.
    """
    m = (mask * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    ap = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
    return int(len(ap))


def label(img, text, org, scale=0.5, thick=1, colour=(255, 255, 255)):
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, thick, cv2.LINE_AA)


def fit_text(img, text, org, scale, thick, colour, max_w):
    """Shrink the font until the text fits `max_w` pixels wide."""
    while scale > 0.2:
        (w, _h), _ = cv2.getTextSize(text, FONT, scale, thick)
        if w <= max_w:
            break
        scale -= 0.02
    label(img, text, org, scale, thick, colour)
    return scale


# ---------------------------------------------------------------- selection

def collect_cells():
    """
    Walk every screenshot and every cell, and keep everything needed to
    choose a good example of each type.

    A cell is only eligible if the ground truth says it carries no variant
    AND the variant detector agrees (no cage, no swirl).
    """
    gt = json.load(open(GT_PATH, encoding="utf-8"))
    per_type = {t: [] for t in TYPES}
    for stamp, path, im in load_images():
        board = gv.find_board(im)
        if board is None:
            print(f"  !! no board in {stamp}")
            continue
        pitch = board["pitch"]
        g = gt[stamp]
        cage = set(map(tuple, g["cage"]))
        swirl = set(map(tuple, g["swirl"]))
        for i, (cx, cy) in enumerate(board["centres"]):
            r, c = divmod(i, 8)              # 0-based
            r1, c1 = r + 1, c + 1            # 1-based, as in ground_truth
            if (r1, c1) in cage or (r1, c1) in swirl:
                continue
            flags = gv.detect_variants(im, cx, cy, pitch)
            if flags:
                continue                    # detector thinks it is wrapped
            t = g["types"][r][c]
            sil = gv.cell_silhouette(im, cx, cy, pitch)
            if sil is None:
                continue
            blob = gv.cell_blob(im, cx, cy, pitch)
            if blob is None:
                continue
            m = clean_mask(sil)
            if m is None:
                continue
            wh, ww = sil["win_shape"]
            area_frac = sil["area"] / float(wh * ww)
            # A neighbouring gem bleeding into the crop would inflate the
            # silhouette well past the cell; a single gem covers ~45-80 %
            # of the 0.88*pitch window.
            if area_frac > 0.90:
                continue
            # The silhouette must not wrap the whole crop window, which is
            # what happens when the cell crop picks up neighbouring gems.
            ring = np.concatenate([m[0, :], m[-1, :], m[:, 0], m[:, -1]])
            border_cov = float((ring > 0).mean())
            if border_cov > 0.65:
                continue
            # The outline must be a single solid body.
            if sil["solidity"] < 0.70:
                continue
            # Colour, measured on the pure gem core.
            core, _ = gv.cell_core(im, cx, cy, pitch)
            hue, sat, val, npx, white = gv.core_hue(core) if core is not None \
                else (None, 0.0, 0.0, 0, 0.0)
            per_type[t].append(dict(
                stamp=stamp, r=r1, c=c1, cx=cx, cy=cy, pitch=pitch,
                extent=sil["extent"], aspect=sil["aspect"], solidity=sil["solidity"],
                area=sil["area"], bbox=sil["bbox"], corners=sil["corners"],
                mask=m, win=(wh, ww), sil_area=sil["area"], sil=sil,
                win_mask=window_mask(sil),
                blob_extent=blob["extent"], blob_corners=blob["corners"],
                hv=hull_vertices(m, 0.03),
                hue=hue, sat=sat, val=val, whiteness=white,
            ))
    return per_type, gt


def choose_best(cells):
    """
    Pick the best single example out of a list of candidate cells.

    The best is the silhouette closest to the group median extent, with a
    preference for a high-solidity (clean, solid, convex) outline and a
    square aspect.
    """
    if not cells:
        return None
    exts = np.array([c["extent"] for c in cells])
    med_ext = float(np.median(exts))
    best, best_score = None, None
    for c in cells:
        # distance from the group median shape
        d = abs(c["extent"] - med_ext) * 4.0
        # a clean outline is convex and solid
        d += (1.0 - c["solidity"]) * 2.0
        # prefer a square outline
        d += abs(c["aspect"] - 1.0) * 1.5
        # and one with a clear, multi-corner hull
        d += (0 if c["hv"] >= 3 else 1.0)
        if best_score is None or d < best_score:
            best, best_score = c, d
    return best


def choose_best_global(per_type):
    """
    Pick one example per type across the WHOLE corpus.

    The group median used as the reference shape is the median over every
    clean cell of that type in every screenshot, so the chosen example is
    representative of the type rather than of one screenshot.
    """
    chosen = {}
    stats = {}
    for t in TYPES:
        cells = per_type[t]
        if not cells:
            continue
        exts = np.array([c["extent"] for c in cells])
        med_ext = float(np.median(exts))
        best, best_score = None, None
        for c in cells:
            d = abs(c["extent"] - med_ext) * 4.0
            d += (1.0 - c["solidity"]) * 2.0
            d += abs(c["aspect"] - 1.0) * 1.5
            d += (0 if c["hv"] >= 3 else 1.0)
            if best_score is None or d < best_score:
                best, best_score = c, d
        chosen[t] = best
        stats[t] = dict(n=len(cells), med_ext=med_ext,
                        ext_min=float(exts.min()), ext_max=float(exts.max()))
    return chosen, stats


# ---------------------------------------------------------------- montages

def photo_tile(im, cand, size, upscale=3):
    """Magnified photo-real crop of one cell, with the type + shape name."""
    crop = cell_window(im, cand["cx"], cand["cy"], cand["pitch"])
    big = cv2.resize(crop, (size * upscale, size * upscale),
                     interpolation=cv2.INTER_LANCZOS4)
    bar = 62
    tile = np.zeros((big.shape[0] + bar, big.shape[1], 3), np.uint8)
    tile[bar:] = big
    fit_text(tile, cand["t"].upper(), (6, 22), 0.62, 2, (255, 255, 255),
             big.shape[1] - 12)
    fit_text(tile, SHAPE_NAME[cand["t"]], (6, 42), 0.44, 1, (190, 215, 255),
             big.shape[1] - 12)
    label(tile, f"r{cand['r']}c{cand['c']}  {cand['stamp']}",
          (6, 56), 0.36, 1, (160, 160, 160))
    return tile


def outline_tile(cand, size, upscale=3):
    """
    Pure filled white silhouette of the same cell on black.

    The mask is rendered in the full cell window (not cropped to the
    bounding box) so the outline keeps its true aspect ratio and sits in
    the same frame as the matching photo tile.
    """
    m = cand["win_mask"]
    big = cv2.resize(m.astype(np.uint8) * 255, (size * upscale, size * upscale),
                     interpolation=cv2.INTER_NEAREST)
    tile = np.zeros((big.shape[0] + 62, big.shape[1], 3), np.uint8)
    tile[62:, :, 0] = big
    tile[62:, :, 1] = big
    tile[62:, :, 2] = big
    fit_text(tile, cand["t"].upper(), (6, 22), 0.62, 2, (255, 255, 255),
             big.shape[1] - 12)
    fit_text(tile, SHAPE_NAME[cand["t"]], (6, 42), 0.44, 1, (190, 215, 255),
             big.shape[1] - 12)
    label(tile, f"r{cand['r']}c{cand['c']}  {cand['stamp']}",
          (6, 56), 0.36, 1, (160, 160, 160))
    return tile


def grid_from_tiles(tiles, cols=4):
    rows = []
    for i in range(0, len(tiles), cols):
        chunk = list(tiles[i:i + cols])
        while len(chunk) < cols:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)


def build_montage(cands, im, path, outline=False, cols=4):
    """Grid montage of the given candidates."""
    size, upscale = 120, 3
    tiles = []
    for t in TYPES:
        c = cands.get(t)
        if c is None:
            continue
        c = dict(c)
        c["t"] = t
        tiles.append(outline_tile(c, size, upscale) if outline
                     else photo_tile(im, c, size, upscale))
    if not tiles:
        return None
    grid = grid_from_tiles(tiles, cols)
    pad = np.zeros((grid.shape[0], 24, 3), np.uint8)
    out = np.hstack([grid, pad])
    cv2.imwrite(path, out)
    return out


def build_combined(cands, im, path, outline=False, cols=4):
    """One labelled montage with the single best example of each type."""
    size, upscale = 170, 3
    tiles = []
    for t in TYPES:
        c = cands.get(t)
        if c is None:
            continue
        c = dict(c)
        c["t"] = t
        tiles.append(outline_tile(c, size, upscale) if outline
                     else photo_tile(im, c, size, upscale))
    grid = grid_from_tiles(tiles, cols)
    cv2.imwrite(path, grid)
    return grid


# ---------------------------------------------------------------- variants

def variant_examples(gt):
    """
    Crop one example of the wooden cage and one of the red swirl aura.

    Ground truth gives the cells (1-based row, col). The cage list is only
    recorded for the cells we are certain about, so the list is searched in
    every screenshot until a hit is found.
    """
    out = {}
    for kind in ("cage", "swirl"):
        for stamp, path, im in load_images():
            g = gt[stamp]
            if not g[kind]:
                continue
            board = gv.find_board(im)
            if board is None:
                continue
            pitch = board["pitch"]
            r1, c1 = g[kind][0]
            idx = (r1 - 1) * 8 + (c1 - 1)
            cx, cy = board["centres"][idx]
            t = g["types"][r1 - 1][c1 - 1]
            crop = cell_window(im, cx, cy, pitch, frac=0.62)
            out.setdefault(kind, []).append((stamp, t, r1, c1, crop))
    return out


def variant_tile(stamp, t, r, c, crop, title, note, size=380):
    big = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)
    bar = 78
    tile = np.zeros((size + bar, size, 3), np.uint8)
    tile[bar:] = big
    fit_text(tile, title, (6, 24), 0.62, 2, (255, 255, 255), size - 12)
    fit_text(tile, note, (6, 44), 0.44, 1, (200, 220, 255), size - 12)
    label(tile, f"{t} gem at r{r}c{c}  ({stamp})", (6, 62), 0.38, 1,
          (170, 170, 170))
    return tile


# ---------------------------------------------------------------- main

def main():
    os.makedirs(OUT, exist_ok=True)
    per_type, gt = collect_cells()
    chosen, stats = choose_best_global(per_type)

    print("\nclean-cell statistics:")
    for t in TYPES:
        s = stats.get(t)
        if s:
            print(f"  {t:7s} clean cells={s['n']:3d}  extent med={s['med_ext']:.3f} "
                  f"[{s['ext_min']:.3f},{s['ext_max']:.3f}]")

    print("\nchosen examples:")
    for t in TYPES:
        c = chosen.get(t)
        if c:
            print(f"  {t:7s} {c['stamp']} r{c['r']}c{c['c']}  "
                  f"extent={c['extent']:.3f} aspect={c['aspect']:.3f} "
                  f"solidity={c['solidity']:.3f} hullV={c['hv']}")

    written = []
    for stamp, path, im in load_images():
        # one best example per type, chosen WITHIN this screenshot so that
        # every montage shows that screenshot's own artwork
        cands = {}
        for t in TYPES:
            sub = [c for c in per_type[t] if c["stamp"] == stamp]
            b = choose_best(sub)
            if b is not None:
                cands[t] = b
        if not cands:
            continue
        p1 = os.path.join(OUT, f"photos_{stamp}.png")
        p2 = os.path.join(OUT, f"outlines_{stamp}.png")
        build_montage(cands, im, p1, outline=False)
        build_montage(cands, im, p2, outline=True)
        written += [p1, p2]
        print(f"  wrote {os.path.relpath(p1, ROOT)} ({len(cands)} types: "
              f"{','.join(t for t in TYPES if t in cands)})")

    # combined montages: the single best example of each type, taken from
    # the reference screenshot (112536).
    ref_im = None
    for stamp, path, im in load_images():
        if stamp == BEST_IMAGE:
            ref_im = im
    best_cands = {}
    for t in TYPES:
        sub = [c for c in per_type[t] if c["stamp"] == BEST_IMAGE]
        b = choose_best(sub)
        if b is None:                      # fall back to the global best
            b = chosen[t]
        if b is not None:
            best_cands[t] = b
    p3 = os.path.join(OUT, "all_types_photo.png")
    p4 = os.path.join(OUT, "all_types_outline.png")
    build_combined(best_cands, ref_im, p3, outline=False)
    build_combined(best_cands, ref_im, p4, outline=True)
    written += [p3, p4]
    print(f"  wrote {os.path.relpath(p3, ROOT)} ({len(best_cands)} types)")
    print(f"  wrote {os.path.relpath(p4, ROOT)} ({len(best_cands)} types)")

    # variants
    var = variant_examples(gt)
    for kind in ("cage", "swirl"):
        items = var.get(kind, [])
        if not items:
            continue
        stamp, t, r, c, crop = items[0]
        if kind == "cage":
            title = "WOODEN CAGE"
            note = "tan/brown square frame wrapped around the gem"
        else:
            title = "RED SWIRL AURA"
            note = "red vortex ring around the gem"
        tile = variant_tile(stamp, t, r, c, crop, title, note)
        pv = os.path.join(OUT, f"variant_{kind}.png")
        cv2.imwrite(pv, tile)
        written.append(pv)
        print(f"  wrote {os.path.relpath(pv, ROOT)}")

    # measurement dump used to write docs/SHAPE_ATLAS.md
    meas = {}
    for t in TYPES:
        cells = per_type[t]
        if not cells:
            continue
        e = np.array([c["extent"] for c in cells])
        a = np.array([c["aspect"] for c in cells])
        ar = np.array([c["area"] for c in cells])
        be = np.array([c["blob_extent"] for c in cells])
        hv = np.array([c["hv"] for c in cells])
        hvc = Counter(hv.tolist())
        b = chosen[t]
        meas[t] = dict(
            n=len(cells),
            extent_med=float(np.median(e)), extent_min=float(e.min()),
            extent_max=float(e.max()),
            aspect_med=float(np.median(a)),
            area_med=float(np.median(ar)),
            blob_extent_med=float(np.median(be)),
            hull_mode=hvc.most_common(1)[0][0],
            hull_hist={int(k): int(v) for k, v in sorted(hvc.items())},
            example=dict(stamp=b["stamp"], r=b["r"], c=b["c"],
                         extent=b["extent"], aspect=b["aspect"],
                         solidity=b["solidity"], area=b["area"],
                         hv=b["hv"], blob_extent=b["blob_extent"]),
        )
    with open(os.path.join(OUT, "measurements.json"), "w", encoding="utf-8") as fh:
        json.dump(meas, fh, indent=1)
    written.append(os.path.join(OUT, "measurements.json"))
    print("  wrote docs/shapes/measurements.json")


if __name__ == "__main__":
    main()
