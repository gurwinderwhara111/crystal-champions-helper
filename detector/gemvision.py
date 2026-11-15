"""
gemvision.py - Crystal Champions gem detection (TEST HARNESS, detection only)

Scope: locate the 8x8 board, segment every cell, and classify each gem by
shape AND colour. No planning, no clicking.

Design notes
------------
* Board finding is theme-agnostic: we project a "gem mask" onto the x and y
  axes and find the 8 dominant peaks (8 columns / 8 rows). Works for both the
  silver-purple chrome and the gold chrome, cropped or full-screen shots.
* Colour comes from the *core* of the cell (central ~30%), which is guaranteed
  to be pure gem material - free of the wooden cage, the red swirl aura and
  the tile background.
* Shape comes from the *full* cell window blob, measured against the tile
  background so the silhouette is exact.
* Two independent classifiers are produced so we can cross-check:
    - `hue`    : rule based (hue band + shape consistency)
    - `ncc`    : normalised template matching against learned reference cores
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------- constants

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
#
# MEASURED on the real game art across 384 hand-labelled cells, not guessed
# from ideal geometry (the first guess was wrong - see PRD section 10):
#   purple  0.52-0.54   sharp   (diamond)
#   yellow  0.51-0.52   sharp   (triangle)   <- same extent as purple
#   red     0.56-0.63   mid     (diamond)
#   white   0.62-0.77   pearl   (ball)
#   blue    0.60-0.76   ball
#   green   0.77-0.85   blocky  (rounded cube)
#   orange  0.77-0.80   blocky  (hexagon)    <- same extent as green
#
# So extent cleanly separates {purple,yellow}, {red}, {white,blue} and
# {green,orange}, but green and orange are geometrically identical in
# silhouette and can only be told apart by colour.
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
#
# Every one of the 7 crystals has its OWN distinctive look - its silhouette
# plus the facet pattern engraved inside it:
#     green  rounded cube, vertical highlight bar
#     orange hexagon, 4-pointed star engraved
#     blue   rounded hexagon / faceted ball, diagonal + horizontal bands
#     red    diamond, V / heart engraving
#     purple diamond, concentric diamond engraving
#     yellow triangle, inner facet
#     white  sphere, hexagonal tessellation
#
# Only red and purple share an OUTLINE (both are diamonds); even they differ
# in the engraving inside, so a full-appearance template separates all seven.
# The bands below are therefore only a safety veto on the sharp end, where the
# silhouette families are unambiguous.
FAMILY_SET = {
    "green":  {"blocky"},
    "orange": {"blocky"},
    "blue":   {"ball", "blocky", "pearl"},
    "white":  {"pearl", "ball", "blocky", "mid"},
    "red":    {"mid", "sharp"},
    "purple": {"sharp"},
    "yellow": {"sharp"},
}

CORE_SIZE = 40          # normalised gem core patch size (px)


# ---------------------------------------------------------------- helpers

def gem_mask(bgr):
    """Binary mask of bright gem pixels (the wooden cage and the red swirl
    aura are deliberately excluded - they are darker than the gem itself)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    bright = (v > 185) & (s > 50)
    pearl = (v > 195) & (s <= 50)
    m = (bright | pearl).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _peaks(profile, min_sep):
    """Local maxima of a smoothed 1-D profile."""
    p = profile.astype(np.float64)
    k = np.ones(5, np.float64) / 5.0
    p = np.convolve(p, k, mode="same")
    cand = []
    for i in range(1, len(p) - 1):
        if p[i] >= p[i - 1] and p[i] > p[i + 1] and p[i] > 0.15 * p.max():
            cand.append((p[i], i))
    cand.sort(reverse=True)
    chosen = []
    for val, idx in cand:
        if all(abs(idx - c) >= min_sep for c in chosen):
            chosen.append(idx)
        if len(chosen) >= 8:
            break
    return sorted(chosen)


def _refine_lattice(peaks, n=8):
    """Given rough peaks, fit a uniform n-cell lattice and return cell centres."""
    if len(peaks) < n:
        return None
    # pitch estimates from consecutive peak spacings
    ests = []
    for i in range(len(peaks) - 1):
        for j in range(i + 1, min(i + 4, len(peaks))):
            k = j - i
            if k >= 1:
                ests.append((peaks[j] - peaks[i]) / k)
    ests.sort()
    pitch = ests[len(ests) // 2] if ests else 0.0

    # least squares fit of peak_i = a + pitch * round((peak_i - a)/pitch)
    a = peaks[0]
    for _ in range(8):
        idxs = np.array([round((p - a) / pitch) for p in peaks], dtype=np.float64)
        vals = np.array(peaks, dtype=np.float64)
        good = (idxs >= -1) & (idxs <= n)
        if good.sum() >= 2:
            a = float(np.mean(vals[good] - idxs[good] * pitch))
    centres = [a + pitch * k for k in range(n)]
    return centres


def _gem_blobs(img):
    """Centroids of compact bright blobs that look like gems."""
    m = gem_mask(img)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    pts = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 500 or a > 6000:
            continue
        if w < 30 or h < 30 or w > 130 or h > 130:
            continue
        r = w / float(h)
        if r < 0.5 or r > 2.0:
            continue
        if a / float(w * h) < 0.35:
            continue
        pts.append((float(cent[i][0]), float(cent[i][1])))
    return pts, m


def _pitch_from_diffs(vals, lo=42.0, hi=140.0):
    """Estimate the lattice pitch from 1-D pairwise differences."""
    v = np.asarray(vals, dtype=np.float64)
    if len(v) < 6:
        return None
    diffs = []
    n = len(v)
    for i in range(n):
        d = np.abs(v[i + 1:] - v[i])
        d = d[(d >= lo) & (d <= hi * 6)]
        diffs.append(d)
    diffs = np.concatenate(diffs) if diffs else np.array([])
    if len(diffs) < 12:
        return None
    scores = []
    for s in np.arange(lo, hi, 0.5):
        k = np.round(diffs / s)
        ok = (k >= 1) & (k <= 6)
        if ok.sum() < 8:
            continue
        err = np.abs(diffs[ok] - k[ok] * s)
        scores.append((s, float(ok.sum()) - 3.0 * float((err > 2.0).sum()) - 0.15 * float(err.sum())))
    if not scores:
        return None
    scores.sort(key=lambda t: -t[1])
    best = scores[0][1]
    good = [s for s, sc in scores if sc >= best * 0.92 and sc > 0]
    if not good:
        good = [scores[0][0]]
    return float(max(good))


def _axis_grid(coords, s, n=8):
    """
    Given 1-D coordinates and a candidate pitch, find the n CONSECUTIVE grid
    lines that hold the most points. Returns (lines, coverage) or (None, 0).

    The "consecutive" requirement is what kills harmonic pitches: with a
    half-pitch the real columns land on every 2nd slot, so no run of 8
    consecutive slots holds the board.
    """
    c = np.asarray(coords, dtype=np.float64)
    if len(c) < 8 or s is None or s <= 0:
        return None, 0.0
    tol = s * 0.30
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
        run = range(k0, k0 + n)
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


def _evenness(lines):
    """How uniform is the spacing of these grid lines? 1.0 = perfect."""
    if lines is None or len(lines) < 3:
        return 0.0
    d = np.diff(np.asarray(lines, dtype=np.float64))
    if d.mean() <= 0:
        return 0.0
    return 1.0 / (1.0 + 3.0 * float(np.std(d) / d.mean()))


def find_board(img, verbose=False):
    """
    Locate the 8x8 board.

    Strategy: find the gem blobs, then for a range of candidate pitches fit a
    lattice on the x and y axes independently. A candidate is only accepted if
    it produces 8 CONSECUTIVE occupied grid lines that explain the blobs AND
    the spacing of those lines is uniform (the real board is perfectly even).
    Theme and crop independent.
    """
    pts, m = _gem_blobs(img)
    if len(pts) < 20:
        return None
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])

    best = None
    for s in np.arange(38.0, 165.0, 1.0):
        gx, cx = _axis_grid(xs, s)
        if gx is None:
            continue
        gy, cy = _axis_grid(ys, s)
        if gy is None:
            continue
        even = min(_evenness(gx), _evenness(gy))
        score = cx * cy * even
        if best is None or score > best[0]:
            best = (score, gx, gy, s, cx, cy)
    if best is None:
        return None
    score, gx, gy, s, cx, cy = best
    if verbose:
        print(f"    blobs={len(pts)} pitch={s} cover_x={cx:.2f} cover_y={cy:.2f} even={score:.3f}")

    pitch = float(np.mean(np.diff(gx)))
    centres = [(x, y) for y in gy for x in gx]
    if verbose:
        print(f"    xs={[round(v, 1) for v in gx]}")
        print(f"    ys={[round(v, 1) for v in gy]}  pitch={pitch:.1f}")
    return {"centres": centres, "xs": gx, "ys": gy, "pitch": pitch, "mask": m}


def _snap(img, grid, axis):
    """Move each grid line onto the local centroid of the gem mass near it."""
    m = gem_mask(img)
    h, w = m.shape
    out = []
    for g in grid:
        lo = int(max(0, g - 26))
        hi = int(min((w if axis == 0 else h), g + 26))
        sl = m[:, lo:hi] if axis == 0 else m[lo:hi, :]
        prof = sl.sum(axis=1 if axis == 0 else 0).astype(np.float64)
        if prof.sum() <= 0:
            out.append(g)
            continue
        wts = np.arange(len(prof), dtype=np.float64)
        c = float((prof * wts).sum() / prof.sum()) + lo
        out.append(c)
    return out


# ---------------------------------------------------------------- per cell

def cell_core(bgr, cx, cy, pitch):
    """Central square of the cell -> pure gem core (normalised to CORE_SIZE)."""
    half = pitch * 0.22
    x0, x1 = int(cx - half), int(cx + half)
    y0, y1 = int(cy - half), int(cy + half)
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(bgr.shape[1], x1)
    y1 = min(bgr.shape[0], y1)
    core = bgr[y0:y1, x0:x1]
    if core.size == 0:
        return None, (x0, y0, x1, y1)
    return cv2.resize(core, (CORE_SIZE, CORE_SIZE), interpolation=cv2.INTER_AREA), (x0, y0, x1, y1)


def cell_blob(bgr, cx, cy, pitch):
    """
    Bright-mask blob of the gem: the pixels that are bright enough to be gem
    material.

    NOTE: this is NOT the true outline - the gems have dark engraved facets
    so a brightness mask leaves holes in them. It is kept because the extent
    measured this way is SCALE CONSISTENT (the cropped screenshots are
    downscaled, and a background-difference silhouette inflates its edge
    band at small scales, which shifts the extent bands per screenshot).

    Use cell_silhouette() when the real outline is needed.
    """
    half = pitch * 0.46
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
    if area < 60:
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


def cell_silhouette(bgr, cx, cy, pitch):
    """
    The gem's TRUE outline, segmented against the TILE BACKGROUND.

    Thresholding on brightness does not work: the gems have dark engraved
    facets (the red diamond has a near-black face, the orange hexagon a dark
    lower-left facet), so a brightness threshold punches holes in the outline.
    Here the background colour is estimated from the border ring of the crop
    and every pixel that differs from it is gem material; the outer contour is
    then filled so the dark facets do not count as holes.

    Caveat: the edge band of the mask grows when the screenshot is a
    downscaled version of the render, so the measured extent drifts with
    scale. Good for shape identification, not for a fixed threshold.
    """
    half = pitch * 0.44
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
    ring = np.concatenate([
        win[:3].reshape(-1, 3), win[-3:].reshape(-1, 3),
        win[:, :3].reshape(-1, 3), win[:, -3:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    diff = np.abs(win.astype(np.float32) - bg.astype(np.float32)).max(axis=2)
    m = (diff > 26).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
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


def silhouette(blob, size=32):
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


def core_hue(core):
    """
    Dominant hue of the gem core.

    Two ideas that matter here:
      * CIRCULAR MEDIAN of the hue, not the mean. The orange hexagon has a
        big yellow X painted across it; a mean hue slides into yellow, the
        median ignores it because the X covers < half of the gem.
      * The pearl is white/iridescent, so its hue is meaningless. It is
        identified by WHITENESS = median over pixels of min(B,G,R)/max(B,G,R).
        Every coloured gem sits near 0.1; the pearl sits near 0.7.
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
    if whiteness > 0.58:
        return None, 0.0, 0.0, 0.0, whiteness

    sel = (s > 130) & (v > 85)
    if sel.sum() < 25:
        sel = (s > 90) & (v > 80)
    if sel.sum() < 15:
        # dark core (cage shadow, swirl overlay, deep shading) -> loosen
        sel = (v > 60) & (s > 40)
    if sel.sum() < 10:
        return None, 0.0, 0.0, 0.0, whiteness
    hh, ss, vv = h[sel], s[sel], v[sel]
    # circular median
    uniq = np.unique(hh)
    best, bestscore = None, 1e18
    for u in uniq:
        d = np.abs(hh - u)
        d = np.minimum(d, 180.0 - d)
        sc = float(d.sum())
        if sc < bestscore:
            bestscore, best = sc, float(u)
    return best, float(np.median(ss)), float(np.median(vv)), float(len(hh)), whiteness


def hue_to_type(hue, sat, val, whiteness=0.0):
    """Map a core hue to one of the 7 types (or None)."""
    if whiteness > 0.58:
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


def shape_family(extent, solidity=None, aspect=None, corners=None):
    """
    Coarse shape family from the extent of the BRIGHT-MASK blob.

    Measured on all 384 labelled cells (this mask is scale consistent, which
    the true-silhouette mask is not):
        purple 0.53 | yellow 0.53 | red 0.60 | blue 0.74 | orange 0.78
        green 0.81 | white 0.68
    """
    if extent is None:
        return None
    if extent < 0.575:
        return "sharp"
    if extent < 0.635:
        return "mid"
    if extent < 0.695:
        return "pearl"
    if extent < 0.765:
        return "ball"
    return "blocky"


def detect_variants(bgr, cx, cy, pitch, gem_bbox=None):
    """
    Flags for the wooden cage and the red swirl aura.

    Both WRAP the gem, so the trick is to REMOVE the gem first: the bright
    gem body is subtracted from the red / tan masks, and whatever is left
    over is the wrapper. A red gem then has no red left, and an orange gem
    has no tan left, so neither can be mistaken for a swirl or a cage.
    """
    half = int(pitch * 0.58) + 2
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

    # bright gem body mask -> removed from the wrapper masks.
    # Threshold is high (V>210) on purpose: the swirl aura sits at V~186 and
    # the wooden cage at V~169, so both must SURVIVE this mask.
    body = ((vv > 210) & (ss > 50)) | ((vv > 200) & (ss <= 50))
    body = cv2.dilate(body.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    def wrapper(mask, min_side):
        """
        A wrapper (cage / swirl) is a HOLLOW frame that fills most of the
        cell. Three tests:
          * bounding box at least min_side * pitch
          * roughly square (a frame is not a streak)
          * it encloses a hole (RETR_CCOMP gives it a child contour)
        """
        m = (mask & ~body).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            return False
        for i, c in enumerate(cnts):
            area = cv2.contourArea(c)
            if area < 0.015 * pitch * pitch:
                continue
            x, y, w, h2 = cv2.boundingRect(c)
            side = max(w, h2)
            if side < min_side * pitch:
                continue
            if not (0.55 < w / float(max(h2, 1)) < 1.85):
                continue
            if hier[0][i][3] != -1:          # has a hole -> it is a frame
                return True
        return False

    red = ((hh < 8) | (hh > 172)) & (ss > 110) & (vv > 60)
    tan = ((hh >= 8) & (hh <= 32)) & (ss > 45) & (ss < 160) & (vv > 50) & (vv < 225)
    if wrapper(red, 0.78):
        flags.add("swirl")
    if wrapper(tan, 0.72):
        flags.add("cage")
    return flags


# ---------------------------------------------------------------- classify

PATCH_SIZE = 48        # normalised full-gem patch size


def gem_patch(bgr, cx, cy, pitch, sil):
    """
    Normalised FULL-APPEARANCE patch of the gem: the true silhouette, masked
    out of the cell crop and resized to a fixed square.

    This captures everything that makes a crystal unique - outline, the
    facet engraving inside it, and colour - so a template match on this
    identifies all seven types, including the two that share an outline
    (red and purple, both diamonds, but engraved differently).
    """
    if sil is None:
        return None
    half = pitch * 0.44
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
    # square it: crop to the silhouette bbox, pad to a square, resize
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


class Classifier:
    """
    Four classifiers over the same reference set:
      * `rule_predict`    - hue bands + silhouette veto (primary, explainable)
      * `template_predict`- FULL-APPEARANCE template match (outline + engraving
                           + colour). This is the one that exploits the fact
                           that every crystal has its own distinct look.
      * `centroid_predict`- nearest neighbour in (hue, sat, val)
      * `shape_predict`   - silhouette template match only (colour blind)
    """

    def __init__(self, refs=None):
        self.refs = refs or {}

    # -- reference based ------------------------------------------------
    @staticmethod
    def build_refs(samples):
        """samples: {type: [{"core": patch, "silh": vector}, ...]}"""
        refs = {}
        for t, patches in samples.items():
            refs[t] = list(patches)
        return refs

    @staticmethod
    def _unit_vec(patch):
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

    def ncc_predict(self, core):
        """Template match the gem core against every learned reference."""
        if core is None or not self.refs:
            return None, 0.0
        c = self._unit_vec(cv2.resize(core, (CORE_SIZE, CORE_SIZE)))
        scores = {}
        for t, arr in self.refs.items():
            best = -1.0
            for ref in arr:
                r = self._unit_vec(ref["core"])
                s = float((c * r).sum() / 3.0)   # mean per-channel correlation
                if s > best:
                    best = s
            scores[t] = best
        order = sorted(scores.items(), key=lambda kv: -kv[1])
        margin = order[0][1] - (order[1][1] if len(order) > 1 else 0.0)
        return order[0][0], margin

    # -- full appearance template match --------------------------------
    @staticmethod
    def _patch_vec(patch):
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

    def template_predict(self, patch):
        """
        Match the whole gem appearance against the learned reference patches.

        The reference set holds the real gems, so this learns the actual
        artwork - outline, engraved facets, highlights and colour - rather
        than a hand written description of it.
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

    # -- nearest colour centroid --------------------------------------
    def centroid_predict(self, core):
        """
        Nearest-neighbour in (hue, saturation, value) space against reference
        centroids. Independent of the hard coded hue bands, so it is a real
        cross-check of the rule based classifier.
        """
        if core is None or not self.refs:
            return None, 0.0
        hue, sat, val, _, white = core_hue(core)
        if hue is None:
            # pearl candidate: compare whiteness with the pearl reference
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

    # -- shape only template match (independent of colour) --------------
    def shape_predict(self, blob):
        """
        Classify by SILHOUETTE alone. Completely independent of colour, so it
        is the cross-check that catches a colour mistake (and vice versa).
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

    # -- rule based -----------------------------------------------------
    def rule_predict(self, core, blob):
        """
        Colour first (hue bands, circular median, whiteness for the pearl),
        then the silhouette as an INDEPENDENT VETO.

        The veto only fires on a strong contradiction - the measured silhouette
        ranges overlap between green/orange and between blue/white, so those
        pairs are never vetoed against each other.
        """
        hue, sat, val, _, white = core_hue(core)
        t = hue_to_type(hue, sat, val, white)
        if t is None:
            return None, 0.0

        # colour confidence: how far the hue sits from the band edges
        edges = {"red": (172.0, 6.5, 6.5), "orange": (6.5, 21.0, 6.5),
                 "yellow": (21.0, 33.0, 6.5), "green": (33.0, 82.0, 8.0),
                 "blue": (82.0, 128.0, 8.0), "purple": (128.0, 172.0, 8.0)}
        conf = 0.97
        if t in edges:
            lo, hi, pad = edges[t]
            if lo < hue < hi:
                margin = min(hue - lo, hi - hue)
            else:                      # red wraps around 0
                margin = min(hue - lo + 180.0, hi + 180.0 - hue)
            conf = 0.97 if margin > pad * 0.5 else 0.72
        if t == "white":
            conf = 0.97 if white > 0.75 else 0.75

        # silhouette veto
        fam = shape_family(blob["extent"]) if blob else None
        allowed = FAMILY_SET.get(t)
        if fam is not None and allowed is not None and fam not in allowed:
            # contradiction -> do not guess
            return None, 0.0
        return t, conf


# ---------------------------------------------------------------- high-level API

DEFAULT_REF_DIR = os.path.join(os.path.dirname(__file__), '..', 'images of cristels')

_classifier_cache = None

def build_classifier(ref_dir: str = None) -> 'Classifier':
    """Build a Classifier from reference images in the given directory.
    Uses the same seed/harvest logic as test_detect.py."""
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
    
    # Build references using the same logic as test_detect.py
    SEEDS = {
        "green":  ("112536", (1, 6)),
        "red":    ("112536", (1, 4)),
        "blue":   ("112536", (2, 1)),
        "purple": ("112536", (1, 8)),
        "yellow": ("112536", (2, 7)),
        "orange": ("112536", (1, 7)),
        "white":  ("112536", (2, 6)),
    }
    
    def analyse(im, verbose=False):
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
            if t is None or conf < 0.95:
                continue
            if len(samples[t]) >= 12:
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


def detect_board(img, roi: dict = None, clf: 'Classifier' = None) -> list:
    """
    Detect the 8x8 board and classify every cell.
    
    Args:
        img: BGR image (full frame or cropped board)
        roi: Optional dict with keys left, top, width, height. 
             If provided AND the image size matches roi (cropped board),
             uses the ROI grid directly. Otherwise, runs find_board() on img.
        clf: Optional Classifier instance. If None, uses the global cached one.
    
    Returns:
        8x8 list of lists of gem type strings ('green', 'red', etc.) or '?' for unknown.
    """
    if clf is None:
        clf = get_classifier()
    
    # Decide whether to use ROI grid or find_board
    use_roi_grid = False
    if roi is not None:
        h, w = img.shape[:2]
        # If image dimensions match ROI (within 5%), assume it's a pre-cropped board
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