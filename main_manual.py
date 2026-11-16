"""
Manual mode entry point — Crystal Champions 1 Hz assistant.

This is the default user experience: a compact snackbar that parks in the
dead zone left of the game board, and arrows that draw over the two gems
to swap. The user taps the gems manually; the bot does NOT click.

Usage:
    python main.py manual
    
Or on Windows:
    double-click run_manual.cmd

Workflow:
    1. Open Chrome → Crystal Champions → PRACTICE mode (no cash!)
    2. Run this script
    3. Click SCAN to calibrate the board
    4. Watch the arrows update every second
    5. Tap the two highlighted gems
    6. ESC or CLOSE to quit

The arrow automatically hides while the screen is captured so that the
detector reads clean pixels. This is guaranteed by the controller calling
on_before_capture() which blocks until the overlay window is fully hidden.
"""

from __future__ import annotations

import json
import os
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# ─── Project root ──────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent

# ─── Theme tokens (kept in sync with panel/theme.py) ───────────────────

BG = "#1a1a2e"
BG_DEEP = "#0d0d1a"
FG = "#e0e0e0"
FG_DIM = "#666666"
ACCENT = "#00d4aa"
WARN = "#ff6b6b"
OK = "#4ade80"
ERROR = "#ef4444"
SCAN_BLUE = "#3b82f6"

TRANSPARENT_KEY = "#010101"  # color key for the arrow overlay


def _project_root() -> Path:
    """Return the project root (one level above this file)."""
    return Path(__file__).resolve().parent


def _config_dir() -> Path:
    return _project_root() / "config"


def _grid_path() -> Path:
    return _config_dir() / "grid.json"


# ─── Controller callbacks ────────────────────────────────────────────────


class ManualUI:
    """Holds the UI elements and controller. Single Tk root."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.overrideredirect(True)

        # State
        self.grid: Optional[Dict[str, Any]] = None
        self.controller: Optional[Any] = None  # ManualController
        self.arrow_overlay: Optional[Any] = None
        self.snackbar: Optional[Any] = None

        # Build UI
        self._setup_window()
        self._setup_snackbar()
        self._setup_arrow_overlay()
        self.root.after(50, self._load_grid_if_present)

    def _setup_window(self) -> None:
        """Apply WS_EX_NOACTIVATE to the root so clicks pass through."""
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_NOACTIVATE = 0x08000000
        WS_EX_TOOLWINDOW = 0x00000080

        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
        if hwnd == 0:
            hwnd = self.root.winfo_id()

        exstyle = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        new_style = exstyle | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)

    def _setup_snackbar(self) -> None:
        """Create the compact snackbar (delegates to panel/snackbar.py)."""
        from panel.snackbar import SnackBar

        self.snackbar = SnackBar(
            self.root,
            on_scan=self._on_scan,
            on_pause=self._on_pause,
            on_close=self._on_close,
        )
        self.snackbar.place_default()
        self.snackbar.set_status("idle", "idle")

    def _setup_arrow_overlay(self) -> None:
        """Create the arrow overlay window."""
        from panel.arrow_overlay import ArrowOverlay

        self.arrow_overlay = ArrowOverlay(self.root)

    # ─── Grid management ───────────────────────────────────────────────

    def _load_grid_if_present(self) -> None:
        """Try to load a previously saved grid."""
        path = _grid_path()
        if path.exists():
            try:
                self.grid = json.load(open(path))
                self.arrow_overlay.set_grid(self.grid)
                self.snackbar.set_status("READY - click PAUSE to start", "ok")
                self.snackbar.set_readout(f"Grid loaded: {self.grid['width']}x{self.grid['height']}")
            except Exception as e:
                self.snackbar.set_status(f"grid load error: {e}", "error")
        else:
            self.snackbar.set_status("click SCAN to calibrate", "idle")

    def _save_grid(self) -> None:
        if self.grid is None:
            return
        _config_dir().mkdir(parents=True, exist_ok=True)
        json.dump(self.grid, open(_grid_path(), "w"), indent=2)

    # ─── Button callbacks ──────────────────────────────────────────────

    def _on_scan(self) -> None:
        """Run board calibration in a background thread."""
        import threading
        
        self.snackbar.set_status("scanning...", "scanning")
        
        # Stop controller if running
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

        def _scan_worker():
            from detector.grid import calibrate
            try:
                grid = calibrate()
                # Update UI on main thread
                self.root.after(0, lambda: self._on_scan_success(grid))
            except Exception as e:
                self.root.after(0, lambda: self.snackbar.set_status(f"scan failed: {e}", "error"))

        thread = threading.Thread(target=_scan_worker, daemon=True)
        thread.start()

    def _on_scan_success(self, grid):
        """Called on main thread after successful scan."""
        self.grid = grid
        self.arrow_overlay.set_grid(self.grid)
        self._save_grid()
        self.snackbar.set_status("READY", "ok")

    def _on_pause(self) -> None:
        """Toggle pause/resume."""
        if self.controller is None:
            # Start the controller
            self._start_controller()
        else:
            if self.controller.paused:
                self.controller.resume()
                self.snackbar.set_paused(False)
            else:
                self.controller.pause()
                self.snackbar.set_paused(True)

    def _on_close(self) -> None:
        """Shut down."""
        if self.controller is not None:
            self.controller.stop()
        if self.arrow_overlay is not None:
            self.arrow_overlay.destroy()
        if self.snackbar is not None:
            self.snackbar.destroy()
        self.root.destroy()

    # ─── Controller wiring ──────────────────────────────────────────────

    def _start_controller(self) -> None:
        """Create and start the ManualController."""
        if self.grid is None:
            self.snackbar.set_status("SCAN first", "error")
            return

        from core.manual_controller import ManualController

        self.controller = ManualController(
            self.grid,
            on_before_capture=self.arrow_overlay.hide,
            on_after_capture=lambda: None,
            on_move=self._on_move,
            on_status=self._on_status,
            on_grid=self._on_grid,
            interval=0.8,
            max_unknown=10,
        )
        self.controller.start()
        self.snackbar.set_paused(False)
        self.snackbar.set_status("RUNNING", "ok")

    def _on_move(self, result: Optional[Dict[str, Any]], top_moves: List[Dict[str, Any]] = []) -> None:
        """Callback when planner produces a move."""
        if result is None:
            self.snackbar.set_readout("no legal move")
            self.arrow_overlay.hide()
            return

        # Show move with color based on pattern type
        move = result["move"]
        r1, c1, r2, c2 = move
        
        # Debug: print what type of move we got
        if result.get("is_pattern"):
            print(f"[MAIN] Pattern detected: {result.get('pattern_type')} at ({r1},{c1})->({r2},{c2})")
        else:
            print(f"[MAIN] Normal move: +{result.get('points', 0)} at ({r1},{c1})->({r2},{c2})")
        
        # Determine arrow color based on pattern type
        if result.get("is_pattern"):
            pattern_type = result.get("pattern_type", "")
            if pattern_type == "void":
                color = "#ff0000"  # Red
                label = "VOID"
                print(f"[MAIN] Showing RED arrow for VOID")
            elif pattern_type == "starlight_t" or pattern_type == "starlight_l":
                color = "#0088ff"  # Blue
                label = "STARLIGHT"
                print(f"[MAIN] Showing BLUE arrow for STARLIGHT")
            elif pattern_type == "fire":
                color = "#ff8800"  # Orange
                label = "FIRE"
                print(f"[MAIN] Showing ORANGE arrow for FIRE")
            else:
                color = None  # Default green
                label = ""
                print(f"[MAIN] Showing GREEN arrow (unknown pattern)")
            
            self.arrow_overlay.show_move((r1, c1), (r2, c2), label, colour=color)
        else:
            # Normal move - green
            print(f"[MAIN] Showing GREEN arrow for normal move")
            self.arrow_overlay.show_move((r1, c1), (r2, c2), f"+{result['points']}")
        
        # Build readout
        if result.get("is_pattern"):
            readout = f"{result.get('pattern_description', 'Special pattern')}"
        else:
            readout = f"R{r1+1}C{c1+1} → R{r2+1}C{c2+1}  +{result['points']}"
            followup = result.get("followup")
            if followup:
                readout += f"  then +{followup['points']}"
        
        self.snackbar.set_readout(readout)

    def _on_status(self, text: str, kind: str) -> None:
        self.snackbar.set_status(text, kind)

    def _on_grid(self, rows: list, stats: dict) -> None:
        """Grid callback (currently unused but required by contract)."""
        pass

    # ─── Main loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    """Entry point."""
    ui = ManualUI()
    ui.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
