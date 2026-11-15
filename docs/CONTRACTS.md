# CONTRACTS — manual-mode rewrite

Every module below is a **frozen interface**. Three agents are working in parallel.
Do not change a signature. Do not rename a key. If something is impossible, still
implement the exact signature and raise a clear exception.

All UI work happens on the Tk main thread. All detection/planning happens on a
worker thread. Nothing in this contract may block the Tk thread for more than
~5 ms.

---

## 0. Coordinate system (read this first)

`grid` dict = the calibrated board lattice, in **screen pixels**:

```python
{
  "x0": float,        # screen x of the centre of cell (row 0, col 0)
  "y0": float,        # screen y of the centre of cell (row 0, col 0)
  "pitch_x": float,   # horizontal distance between cell centres
  "pitch_y": float,   # vertical distance between cell centres
  "left": int,        # x of the TOP-LEFT of the capture region (screen px)
  "top": int,         # y of the TOP-LEFT of the capture region (screen px)
  "width": int,       # width  of the capture region
  "height": int,      # height of the capture region
  "cols": 8,
  "rows": 8,
}
```

Invariants:

* `left <= x0 - pitch_x/2` and `top <= y0 - pitch_y/2`
* `left + width >= x0 + 7*pitch_x + pitch_x/2`
* the capture region is tight around the board (margin ≈ `0.55 * pitch`) so that
  the snackbar overlay is **never inside the crop** and never pollutes detection.

Cell centre for `(r, c)`, in screen px:

```
sx = x0 + c * pitch_x
sy = y0 + r * pitch_y
```

Cell centre for `(r, c)`, in crop px (crop captured at `left, top`):

```
cx = (x0 - left) + c * pitch_x
cy = (y0 - top)  + r * pitch_y
```

---

## 1. `detector/grid.py`

```python
GRID_PATH  = <project_root>/config/grid.json
CACHE_PATH = <project_root>/config/classifier.pkl

def calibrate(full_bgr=None) -> dict:
    """Find the board on screen and return a `grid` dict (see §0).
    If full_bgr is None, grab the primary monitor with mss.
    Raises RuntimeError("board not found") if find_board fails."""

def save_grid(grid: dict, path: str = GRID_PATH) -> None: ...
def load_grid(path: str = GRID_PATH) -> dict | None: ...
    # returns None if the file is missing or fails validation

def capture_region(grid: dict) -> dict:
    """Return the mss monitor dict for grabbing the board region:
    {"left": int, "top": int, "width": int, "height": int}"""

def get_classifier(rebuild: bool = False) -> "Classifier":
    """Cached classifier. Loads CACHE_PATH if present, else builds from
    `images of cristels` and writes the cache. Rebuild ~7 s, cache load ~40 ms."""

def read_grid(crop, grid: dict, clf=None) -> tuple[list[list[str]], dict]:
    """crop = BGR ndarray of exactly `capture_region(grid)`.
    Returns (rows, stats):
      rows  : 8x8 list of long type names ('green','red','blue','purple',
              'yellow','orange','white') or '?' when refused.
      stats : {"unknown": int, "known": int, "cells": [[...]]}
    """
```

`grid.json` example:

```json
{"x0": 900.4, "y0": 205.1, "pitch_x": 81.6, "pitch_y": 81.5,
 "left": 855, "top": 160, "width": 665, "height": 665,
 "cols": 8, "rows": 8}
```

---

## 2. `bot/planner.py`

```python
DET_TO_GAME = {"green":"g","orange":"o","blue":"b","white":"w",
               "red":"r","purple":"p","yellow":"y"}

def grid_to_game(grid, level=1, score=0.0, time_left=180.0) -> "Game":
    """8x8 long names or short codes -> sim.rules.Game.
    '?' or unknown -> None (hole)."""

def best_move(grid, level=1, score=0.0, time_left=180.0, lookahead=6) -> dict | None:
    """Returns None when there is no legal move. Otherwise:

    {
      "move":      (r1, c1, r2, c2),
      "points":    int,            # immediate points of this swap
      "followup":  {"move": (r1,c1,r2,c2), "points": int} | None,
      "combined":  float,          # ranking score used to sort
      "chain":     int,
      "specials":  int,
      "detonations": int,
      "level_ups": int,
    }

    Two-ply: evaluate every legal move, take the top `lookahead` by immediate
    points, and for each of those evaluate the best reply from the resulting
    board. `combined` currently = points + 0.6 * followup_points.
    Must not mutate the caller's grid.
    """

def format_move(move: tuple) -> str: ...
```

`best_move` must finish in **< 600 ms** on a 2-core i3 for a normal board.

---

## 3. `sim/rules.py` (unchanged public API)

```python
Game(seed=None, level=1, score=0.0, time_left=180.0, mode="standard")
g.clone() -> Game
g.legal_moves() -> list[tuple[int,int,int,int]]
g.apply_swap(r1,c1,r2,c2) -> MoveResult
g.has_legal_move() -> bool
MoveResult: .points .chain_count .specials_created .detonations .level_ups .time_cost
COLOURS = ('g','o','b','w','r','p','y')
```

---

## 4. `panel/arrow_overlay.py`

```python
class ArrowOverlay:
    """Borderless, click-through, never-focused arrow overlay drawn ON the board."""

    def __init__(self, master): ...          # master = the Tk root

    def set_grid(self, grid: dict) -> None:
        """Position the overlay window over capture_region(grid)."""

    def show_move(self, a: tuple[int, int], b: tuple[int, int],
                  label: str = "") -> None:
        """a, b are (row, col). Draws an arrow from a to b plus rings on both
        cells and an optional text label. Non-blocking."""

    def hide(self) -> None:
        """Hide and BLOCK until the pixels are actually off screen.
        The caller grabs the screen immediately after this returns, so this
        must not be fire-and-forget: withdraw() then update() then a short
        settle sleep (~25-40 ms)."""

    def set_visible(self, visible: bool) -> None: ...
    def destroy(self) -> None: ...
```

Required window styles: `WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW`.
Colour-key transparency via tkinter `-transparentcolor` so only the arrow is opaque.
The overlay must never appear in a screenshot taken while it is hidden.

---

## 5. `panel/snackbar.py`

```python
class SnackBar:
    """Compact always-on-top bar. ~360 x 46 px. Draggable by its body.

    Layout (single row):
        [status dot] [readout text ............] [SCAN] [PAUSE] [X]
    """

    def __init__(self, master, on_scan, on_pause, on_close): ...

    def set_status(self, text: str, kind: str = "idle") -> None:
        """kind in {"idle","scanning","paused","ok","error"} -> dot colour."""

    def set_readout(self, text: str) -> None:
        """Short line, e.g. 'R3C2 -> R3C5   +340   then +180'."""

    def set_paused(self, paused: bool) -> None:
        """Flips the PAUSE button label to RESUME/PAUSE."""

    def set_move_cells(self, a, b) -> None:
        """Optional: colour the readout. a/b are (row,col) or None."""

    def place_default(self) -> None:
        """Park it in the dead zone left of the board (x=12, y=centred)."""

    def destroy(self) -> None: ...
```

Must use `WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW` so clicking a button never
takes focus away from Chrome. Buttons must still receive clicks — NOACTIVATE
windows do get mouse input; what they must not do is activate.

Exactly three buttons. No other widgets that steal keyboard focus.

---

## 6. `core/manual_controller.py`

Pure logic. **No tkinter import.** All UI is injected as callbacks.

```python
class ManualController:
    def __init__(self, grid: dict, *,
                 on_before_capture=None,   # () -> None   MUST block until overlay hidden
                 on_after_capture=None,    # () -> None   overlay may show again
                 on_move=None,             # (dict | None) -> None   planner result
                 on_status=None,           # (str, str) -> None   (text, kind)
                 on_grid=None,             # (rows, stats) -> None
                 interval: float = 1.0,
                 max_unknown: int = 10): ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    @property
    def paused(self) -> bool: ...
    @property
    def last_result(self) -> dict | None: ...
```

Tick sequence, exactly:

```
1. on_before_capture()          # overlay hides, blocks
2. crop = mss.grab(capture_region(grid))
3. on_after_capture()           # overlay may reappear (controller does not show it)
4. rows, stats = detector.grid.read_grid(crop, grid, clf)
5. if stats["unknown"] > max_unknown -> on_status("UNCERTAIN", "error"),
   on_move(None), sleep remainder, continue
6. result = planner.best_move(rows, level, score, time_left)
7. on_move(result); on_grid(rows, stats)
8. sleep so that the tick period is `interval`
```

Track predicted score/level internally from `result["points"]` so the planner has
a multiplier even with no HUD OCR. Expose them as `.predicted_score` and
`.level`.

`stop()` must join the worker within 2 s and must not raise.

---

## 7. Bug fixes that MUST land (from the audit)

| file:line | bug |
|---|---|
| `detector/classify.py:310` | `DEFAULT_REF_DIR` uses `'..','..'` -> resolves above the repo. Must be one `'..'`. |
| `detector/detect.py:145` | same two-level bug in `load_reference_images` |
| `detector/cell.py:52` | calls `gem_mask()` but only imports `constants` -> `NameError`. Import from `.board`. |
| `detector/classify.py:162` | `silhouette()` not in scope -> `NameError`. Import from `.cell`. |
| `detector/classify.py:282` | `hue < 170 -> purple` but band ends at 172 -> purple with hue 170–172 becomes `?`. Use `< 172`. |
| `detector/classify.py:386` | `build_classifier` returns a dict, annotated `-> Classifier`. Return a `Classifier`. |
| `bot/planner.py:21` | `DET_TO_GAME` maps short code -> short code. Must map long names. |
| `sim/rules.py:100-105` | `detonation_points` returns a cumulative sum and is fed the running counter -> 2nd detonation gives 60 instead of 40. |
| `sim/rules.py:563` | level-up checked before combo (620) and void (641) points are added -> missed level-ups. |
| `sim/rules.py:292` | `_swap_is_legal` tests "any match on board" not "swap created a match" -> 108/112 swaps reported legal. |
| `sim/rules.py:676` | hint penalty `int(50*mult)` = -175 at L6; spec says flat -50. |
| `sim/rules.py:500-504` | chain detonation only one level deep (snapshot `list(to_clear)`); deeper specials never fire. |
| `sim/rules.py` wave loop | if every matched cell is a created-special cell, `to_clear` is empty -> infinite `while True`. Needs a progress guard. |
| `core/hud_parser.py:210` | `time.time()` with `import time` only inside `__main__` -> `NameError`. |
| `core/logging_system.py:255` | `f"pitch={pitch:.1f if pitch else 'N/A'}"` -> `ValueError` every frame. |
| `core/auto_player.py:110` | `datetime.now()` with no `import datetime`. |
| `core/auto_player.py:766,792` | `json.dump` with no module-level `import json`. |
| `core/auto_player.py:677` | `_check_round_end` returns True whenever there is no legal move -> ends the session on the first settled board. |

---

## 8. Style rules

* Python 3.13, no new third-party dependencies. `numpy`, `cv2`, `mss`, `tkinter`, `ctypes` only.
* Type hints on every public function.
* Never `except: pass`. Log or set a status.
* No `sys.path` mutation at import time. Use package-relative imports (`from detector.grid import ...`).
* Every module gets `if __name__ == "__main__":` self-test that runs standalone.
