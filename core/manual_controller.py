"""
Manual controller — the tick loop for the 1 Hz manual-mode assistant.

This module is pure logic. It has NO tkinter import. All UI is injected as
callbacks so the controller can be tested without a display and so the UI
thread is clearly separated from the worker thread.

See docs/CONTRACTS.md §6 for the contract.
"""

from __future__ import annotations

import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ─── Inactive UI: all communication goes through callbacks ─────────────

# () -> None       The overlay MUST hide and BLOCK until hidden.
OnBeforeCapture = Callable[[], None]

# () -> None       Overlay may now show again (controller does not show it).
OnAfterCapture = Callable[[], None]

# (dict | None)    Planner result from best_move(), None when uncertain.
OnMove = Callable[[Optional[Dict[str, Any]]], None]

# (text, kind)     kind in {"idle","scanning","paused","ok","error"}
OnStatus = Callable[[str, str], None]

# (rows, stats)    The 8x8 grid of long names and detection stats.
OnGrid = Callable[[List[List[str]], Dict[str, Any]], None]


class ManualController:
    """
    Runs a 1 Hz tick loop that detects the board and suggests moves.
    
    Usage:
        ctrl = ManualController(
            grid,
            on_before_capture=overlay.hide_and_wait,
            on_after_capture=overlay.show,
            on_move=snackbar.set_readout,
            on_status=snackbar.set_status,
            on_grid=overlay.render_grid,
        )
        ctrl.start()
        # ...
        ctrl.stop()
    
    The tick sequence is (per CONTRACTS.md §6):
        1. on_before_capture()    # blocks until overlay hidden
        2. crop = mss.grab(region)
        3. on_after_capture()
        4. rows, stats = read_grid(crop, grid, clf)
        5. if unknown > max_unknown: on_status("UNCERTAIN"), on_move(None)
        6. result = planner.best_move(rows, level, score, remaining_time)
        7. on_move(result), on_grid(rows, stats)
        8. sleep so that the tick period is `interval`
    
    The controller tracks predicted score/level internally and passes them
    to the planner so that the level multiplier is accounted for.
    """

    def __init__(
        self,
        grid: Dict[str, Any],
        *,
        on_before_capture: OnBeforeCapture,
        on_after_capture: OnAfterCapture,
        on_move: OnMove,
        on_status: OnStatus,
        on_grid: OnGrid,
        interval: float = 1.0,
        max_unknown: int = 10,
    ) -> None:
        self.grid = grid
        self._on_before_capture = on_before_capture
        self._on_after_capture = on_after_capture
        self._on_move = on_move
        self._on_status = on_status
        self._on_grid = on_grid
        self.interval = max(0.1, interval)
        self.max_unknown = max(0, max_unknown)

        # Internal state
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Predicted game state
        self._predicted_score = 0.0
        self._level = 1
        self._level_thresholds = [500, 1475, 3075, 5450, 8750]
        self._time_start: Optional[float] = None

        # Cached classifier
        self._clf = None

        # Last result for external polling
        self._last_result: Optional[Dict[str, Any]] = None

    # ─── Public API ────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def predicted_score(self) -> float:
        return self._predicted_score

    @property
    def level(self) -> int:
        return self._level

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        return self._last_result

    def start(self) -> None:
        """Start the tick loop. Returns immediately."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._paused = False
            self._time_start = time.monotonic()
            self._thread = threading.Thread(target=self._tick_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the tick loop and join the thread (max 2 s)."""
        with self._lock:
            if not self._running:
                return
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    # ─── Tick Loop ─────────────────────────────────────────────────────

    def _tick_loop(self) -> None:
        """Worker thread: 1 Hz detect + plan."""
        import mss

        from detector.grid import get_classifier, read_grid, capture_region
        from bot.planner import best_move

        # Warm-start the classifier (first call is ~7 s)
        try:
            self._clf = get_classifier()
        except Exception as e:
            self._on_status(f"classifier failed: {e}", "error")
            self._running = False
            return

        while self._running:
            tick_start = time.monotonic()

            # ── 1. Pause gate ───────────────────────────────────────────
            while self._paused and self._running:
                time.sleep(0.1)

            if not self._running:
                break

            # ── 2. Capture (with overlay hidden) ─────────────────────────
            try:
                self._on_before_capture()
                region = capture_region(self.grid)

                with mss.mss() as sct:
                    shot = sct.grab(region)
                    crop = np.array(shot)[:, :, :3]  # BGRA -> BGR
            except Exception as e:
                self._on_status(f"capture failed: {e}", "error")
                time.sleep(0.5)
                continue
            finally:
                self._on_after_capture()

            # ── 3. Detect ───────────────────────────────────────────────
            try:
                rows, stats = read_grid(crop, self.grid, self._clf)
            except Exception as e:
                self._on_status(f"detect failed: {e}", "error")
                time.sleep(0.5)
                continue

            unknown = stats.get("unknown", 0)
            if unknown > self.max_unknown:
                self._on_status(f"UNCERTAIN — {unknown} cells unread", "error")
                self._on_move(None)
                self._on_grid(rows, stats)
                self._last_result = None
                self._sleep_remaining(tick_start)
                continue

            # ── 4. Plan ─────────────────────────────────────────────────
            try:
                remaining = self._remaining_time()
                result = best_move(
                    rows,
                    level=self._level,
                    score=self._predicted_score,
                    time_left=remaining,
                )
            except Exception as e:
                self._on_status(f"plan failed: {e}", "error")
                self._on_move(None)
                self._on_grid(rows, stats)
                self._last_result = None
                self._sleep_remaining(tick_start)
                continue

            # ── 5. Update internal state ─────────────────────────────────
            if result is not None:
                self._predicted_score += result["points"]
                # Level-up check
                while (
                    self._level < 6
                    and self._predicted_score >= self._level_thresholds[self._level - 1]
                ):
                    self._level += 1

            # ── 6. Callbacks ─────────────────────────────────────────────
            self._on_status("ok", "ok")
            self._on_move(result)
            self._on_grid(rows, stats)
            self._last_result = result

            # ── 7. Sleep ─────────────────────────────────────────────────
            self._sleep_remaining(tick_start)

    def _remaining_time(self) -> float:
        """Seconds remaining in the 180 s game, or 180 if not started."""
        if self._time_start is None:
            return 180.0
        return max(0.0, 180.0 - (time.monotonic() - self._time_start))

    def _sleep_remaining(self, tick_start: float) -> None:
        """Sleep until the tick period ends."""
        elapsed = time.monotonic() - tick_start
        remaining = self.interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


# ─── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import os

    import numpy as np

    # Stub callbacks
    events: List[str] = []

    def on_before_capture():
        events.append("before")
        print("  on_before_capture")

    def on_after_capture():
        events.append("after")
        print("  on_after_capture")

    def on_move(result):
        events.append(f"move:{result is not None}")
        print(f"  on_move: {result is not None}")

    def on_status(text, kind):
        events.append(f"status:{text}:{kind}")
        print(f"  on_status: {text} ({kind})")

    def on_grid(rows, stats):
        events.append("grid")
        print(f"  on_grid: {stats['unknown']} unknown")

    # Try to load a saved grid
    grid_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "grid.json"
    )
    if not os.path.exists(grid_path):
        print(f"No grid.json at {grid_path} — run SCAN first")
        raise SystemExit(0)

    grid = json.load(open(grid_path))
    print(f"Loaded grid: {grid['x0']:.1f},{grid['y0']:.1f} pitch {grid['pitch_x']:.1f}x{grid['pitch_y']:.1f}")

    # Create and run for 3 ticks
    ctrl = ManualController(
        grid,
        on_before_capture=on_before_capture,
        on_after_capture=on_after_capture,
        on_move=on_move,
        on_status=on_status,
        on_grid=on_grid,
        interval=1.0,
        max_unknown=10,
    )

    print("\nStarting 3-second test...")
    ctrl.start()
    time.sleep(3.5)
    ctrl.stop()
    print(f"\nEvents captured: {len(events)}")
    print("OK")
