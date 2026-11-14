"""
collect.py - record the REAL Crystal Champions board while you play.

Why this exists
---------------
Static screenshots only ever show a settled board. The real game is played on
a MOVING board (you are supposed to swap while gems are falling - "Live
Swaps"), and our detector has never been tested against a board in motion.
This tool grabs the board region at high frequency during a live 3 minute
match, so we end up with a labelled-by-motion dataset:

  * gems mid-fall
  * the +10 / +60 point popups floating over the board
  * special gems detonating
  * the board clearing and refilling on a level up
  * the Hyperspeed / PowerUp transitions

It also pairs every frame with a crop of the HUD, so we can read the score,
the level and the timer for each frame and turn the recording into a
(board state -> score delta) dataset that the planner can be calibrated on.

It NEVER takes focus. Run it with pythonw.exe so no console window appears,
and the game will not pause.

Usage
-----
    python collect.py --seconds 200            # record ~3.3 minutes
    python collect.py --seconds 200 --fps 20
    python collect.py --recal                   # re-find the board, then record
    python collect.py --calibrate               # just find the board, print the ROI

Run it BEFORE you start the match, press "Start match" in the browser, then
let it record the whole round. Ctrl+C or close the window to stop.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemvision as gv

try:
    import mss
except ImportError:
    print("MSS is required:  python -m pip install mss")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "dataset")
CFG = os.path.join(HERE, "roi.json")

# how much the board must change before we keep the frame (0..1)
CHANGE_THRESHOLD = 0.004
# the HUD strip above the board (score / level / timer)
HUD_H = 46
HUD_W = 430


def find_roi(im):
    """Locate the board and return (left, top, width, height) plus the HUD box.
    Validates that the board has enough blobs and proper dimensions."""
    board = gv.find_board(im)
    if board is None:
        return None
    
    # Validate: must have enough blobs (near 64 for full board)
    pts, _ = gv._gem_blobs(im)
    if len(pts) < 40:  # Need at least ~40 visible gems for reliable grid
        return None
    
    xs, ys, pitch = board["xs"], board["ys"], board["pitch"]
    
    # Validate pitch is in expected range (60-100 for full-res, 38-50 for half-res)
    if pitch < 35 or pitch > 110:
        return None
    
    # Validate grid spans ~8 cells
    if len(xs) < 7 or len(ys) < 7:
        return None
    
    left = int(min(xs) - pitch * 0.55)
    top = int(min(ys) - pitch * 0.55)
    right = int(max(xs) + pitch * 0.55)
    bottom = int(max(ys) + pitch * 0.55)
    w, h = right - left, bottom - top
    
    # Validate ROI size (should be ~500-900px for full board)
    if w < 300 or h < 300 or w > 1000 or h > 1000:
        return None
    
    hud = (max(0, left), max(0, top - HUD_H - 12), min(HUD_W, w + 120), HUD_H)
    return {"left": left, "top": top, "width": w, "height": h, "hud": hud}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=200.0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--recal", action="store_true")
    ap.add_argument("--wait", type=float, default=600.0,
                    help="seconds to keep waiting for the board to appear")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(OUT_ROOT, stamp)
    frames_dir = os.path.join(outdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    logpath = os.path.join(outdir, "collect.log")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            with open(logpath, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass

    log(f"collector started, outdir={outdir}")

    with mss.MSS() as sct:
        # ---- locate the board, WAITING for it to appear ----
        roi = None
        if not args.recal and os.path.exists(CFG):
            roi = json.load(open(CFG))
            log(f"using saved ROI {roi}")

        deadline = time.time() + args.wait
        tries = 0
        while roi is None and time.time() < deadline:
            tries += 1
            shot = np.asarray(sct.grab(sct.monitors[1]))[:, :, :3]
            roi = find_roi(shot)
            if roi is None:
                if tries % 10 == 0:
                    log(f"waiting for the game board... ({tries})")
                time.sleep(1.0)
                continue
            json.dump(roi, open(CFG, "w"), indent=1)
            log(f"board found: {roi['left']},{roi['top']} "
                f"{roi['width']}x{roi['height']}")
        if roi is None:
            log("GAVE UP waiting for the board. Nothing recorded.")
            return
        if args.calibrate:
            log(json.dumps(roi))
            return

        region = {"left": roi["left"], "top": roi["top"],
                  "width": roi["width"], "height": roi["height"]}
        hud = {"left": roi["hud"][0], "top": roi["hud"][1],
               "width": roi["hud"][2], "height": roi["hud"][3]}

        manifest = open(os.path.join(outdir, "manifest.jsonl"), "a",
                        buffering=1)
        prev = None
        kept = 0
        t0 = time.time()
        interval = 1.0 / args.fps
        log(f"RECORDING for {args.seconds:.0f}s at {args.fps:.0f} fps")
        log(">>> GO TO CHROME AND PLAY NOW <<<")

        try:
            while (time.time() - t0) < args.seconds:
                loop_start = time.time()
                arr = np.asarray(sct.grab(region))[:, :, :3]

                keep = prev is None
                if not keep:
                    diff = cv2.absdiff(arr, prev).mean() / 255.0
                    keep = diff > CHANGE_THRESHOLD

                if keep:
                    name = f"f{kept:05d}.png"
                    cv2.imwrite(os.path.join(frames_dir, name), arr)
                    hudarr = np.asarray(sct.grab(hud))[:, :, :3]
                    cv2.imwrite(os.path.join(frames_dir, "hud_" + name),
                                hudarr)
                    manifest.write(json.dumps({
                        "frame": name,
                        "t": round(time.time() - t0, 3),
                        "wall": datetime.now().isoformat(timespec="seconds"),
                    }) + "\n")
                    prev = arr
                    kept += 1
                    if kept % 25 == 0:
                        log(f"  {kept} frames  t={time.time()-t0:6.1f}s")

                sleep = interval - (time.time() - loop_start)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            log("stopped by user.")

    manifest.close()
    log(f"DONE: {kept} frames in {outdir}")


if __name__ == "__main__":
    main()
