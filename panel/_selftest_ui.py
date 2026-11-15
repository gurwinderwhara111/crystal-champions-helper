"""
UI self-test for the compact panel (not part of the app).
=========================================================

Run:  python panel/_selftest_ui.py

Creates real, visible Tk windows on the desktop for ~3 s, exercises
ArrowOverlay.show_move / hide / destroy and SnackBar.set_status / set_readout /
set_paused / destroy, and prints the measured Win32 extended styles plus the
IsWindowVisible transition that the capture loop depends on.

Exits 0 on success, 3 on watchdog timeout, 1 on any exception.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
import tkinter as tk
import traceback
from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import arrow_overlay  # noqa: E402
import snackbar  # noqa: E402

GRID = {
    "x0": 240.8, "y0": 190.8, "pitch_x": 81.6, "pitch_y": 81.5,
    "left": 200, "top": 150, "width": 665, "height": 665,
    "cols": 8, "rows": 8,
}

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_GetForegroundWindow = _user32.GetForegroundWindow
_GetForegroundWindow.restype = wintypes.HWND
_GetForegroundWindow.argtypes = ()
_WindowFromPoint = _user32.WindowFromPoint
_WindowFromPoint.restype = wintypes.HWND
_WindowFromPoint.argtypes = (wintypes.POINT,)

_LAYERED = 0x00080000
_TRANSPARENT = 0x00000020
_NOACTIVATE = 0x08000000
_TOOLWINDOW = 0x00000080

try:
    import mss
    import numpy as np
except ImportError as exc:  # screenshot proof is optional
    mss = None
    np = None
    _SHOT_ERR = str(exc)
else:
    _SHOT_ERR = ""


def shot():
    """Grab exactly capture_region(GRID) as BGRx."""
    if mss is None:
        return None
    with mss.mss() as sct:
        return np.array(sct.grab({
            "left": GRID["left"], "top": GRID["top"],
            "width": GRID["width"], "height": GRID["height"],
        }))


def at_point(x: int, y: int) -> int:
    return int(_WindowFromPoint(wintypes.POINT(int(x), int(y))) or 0)


def decode(ex: int) -> str:
    return "|".join(
        name for name, bit in (
            ("LAYERED", _LAYERED),
            ("TRANSPARENT", _TRANSPARENT),
            ("NOACTIVATE", _NOACTIVATE),
            ("TOOLWINDOW", _TOOLWINDOW),
        )
        if ex & bit
    )


def fg() -> int:
    return int(_GetForegroundWindow() or 0)


def watchdog(seconds: float = 14.0) -> None:
    def _kill() -> None:
        time.sleep(seconds)
        print("WATCHDOG: selftest did not finish in time", flush=True)
        os._exit(3)

    threading.Thread(target=_kill, daemon=True).start()


def main() -> int:
    watchdog()

    root = tk.Tk()
    root.withdraw()
    print(f"root withdrawn; screen {root.winfo_screenwidth()}x{root.winfo_screenheight()}",
          flush=True)

    # ── ArrowOverlay ─────────────────────────────────────────────────
    ov = arrow_overlay.ArrowOverlay(root)
    ov.set_grid(GRID)
    print(f"overlay hwnd=0x{ov.hwnd:X}  window={ov.win.geometry()}", flush=True)

    fg_before = fg()
    ov.show_move((3, 2), (3, 5), "+340")
    root.update()
    ex = ov.apply_styles()
    print(f"overlay exstyle = 0x{ex:08X}  -> {decode(ex)}", flush=True)
    print(f"overlay IsWindowVisible after show = {int(ov.is_visible())}", flush=True)
    print(f"foreground hwnd before/after show = 0x{fg_before:X} / 0x{fg():X} "
          f"(focus steal: {fg_before != fg()})", flush=True)
    for bit, name in ((_LAYERED, "LAYERED"), (_TRANSPARENT, "TRANSPARENT"),
                      (_NOACTIVATE, "NOACTIVATE"), (_TOOLWINDOW, "TOOLWINDOW")):
        print(f"  {name:<11} bit set: {bool(ex & bit)}", flush=True)

    # Click-through: WindowFromPoint must NOT resolve to the overlay window.
    px = GRID["left"] + GRID["width"] // 2
    py = GRID["top"] + GRID["height"] // 2
    hit = at_point(px, py)
    print(f"click-through: WindowFromPoint({px},{py}) = 0x{hit:X} "
          f"overlay = 0x{ov.hwnd:X} -> passes through: {hit != ov.hwnd}", flush=True)

    # Capture-safety: arrow pixels must be on screen now and gone after hide().
    shots: dict[str, object] = {}
    if mss is None:
        print(f"screenshot check skipped (mss/numpy: {_SHOT_ERR})", flush=True)
    else:
        shots["with_arrow"] = shot()
        if shots["with_arrow"] is not None:
            print(f"screenshot with arrow: shape={shots['with_arrow'].shape}", flush=True)

    state: dict[str, object] = {}

    def step_hide() -> None:
        t0 = time.perf_counter()
        ov.hide()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        vis = int(ov.is_visible())
        state["hidden_visible"] = vis
        print(f"overlay hide() blocked {dt_ms:.1f} ms; "
              f"IsWindowVisible after hide = {vis}", flush=True)
        root.after(300, step_show_again)

    def step_show_again() -> None:
        ov.show_move((1, 1), (1, 4), "+120")
        root.update()
        print(f"overlay IsWindowVisible after 2nd show = {int(ov.is_visible())}", flush=True)
        root.after(800, step_teardown_overlay)

    def step_teardown_overlay() -> None:
        ov.destroy()
        ov.destroy()  # must be safe twice
        print("overlay destroyed twice: ok", flush=True)
        root.after(100, build_snackbar)

    # ── SnackBar ─────────────────────────────────────────────────────
    def build_snackbar() -> None:
        calls: list[str] = []
        bar = snackbar.SnackBar(
            root,
            on_scan=lambda: calls.append("scan"),
            on_pause=lambda: calls.append("pause"),
            on_close=lambda: calls.append("close"),
        )
        state["bar"] = bar
        root.update()
        ex = bar.apply_styles()
        print(f"snackbar hwnd=0x{bar.hwnd:X}  geometry={bar.win.geometry()}", flush=True)
        print(f"snackbar exstyle = 0x{ex:08X}  -> {decode(ex)}", flush=True)
        print(f"snackbar IsWindowVisible = {int(bar.is_visible())}", flush=True)
        for bit, name in ((_NOACTIVATE, "NOACTIVATE"), (_TOOLWINDOW, "TOOLWINDOW")):
            print(f"  {name:<11} bit set: {bool(ex & bit)}", flush=True)
        print(f"snackbar bar size = {bar.win.winfo_width()}x{bar.win.winfo_height()}",
              flush=True)

        bar.set_status("idle", "idle")
        bar.set_readout("R3C2 -> R3C5   +340   then +180")
        root.update()
        print(f"readout(long)  = {bar.readout.cget('text')!r}", flush=True)
        bar.set_status("calibrating\u2026", "scanning")
        root.update()
        print(f"readout(scan)  = {bar.readout.cget('text')!r}", flush=True)
        bar.set_paused(True)
        root.update()
        print(f"pause label = {bar.btn_pause._label.cget('text')!r}", flush=True)
        root.after(600, step_bar2)

    def step_bar2() -> None:
        bar: snackbar.SnackBar = state["bar"]  # type: ignore[assignment]
        bar.set_readout("UNCERTAIN \u2014 14 cells unread")
        bar.set_status("UNCERTAIN \u2014 14 cells unread", "error")
        bar.set_move_cells((3, 2), (3, 5))
        bar.set_paused(False)
        root.update()
        print(f"readout(err)   = {bar.readout.cget('text')!r}", flush=True)
        print(f"readout width  = {bar.readout.winfo_width()} px", flush=True)
        print(f"pause label = {bar.btn_pause._label.cget('text')!r}", flush=True)
        # Simulated click on SCAN goes through the same handler path.
        bar._do_scan()
        bar._do_pause()
        root.after(600, finish)

    def finish() -> None:
        bar: snackbar.SnackBar = state["bar"]  # type: ignore[assignment]
        bar.destroy()
        bar.destroy()  # safe twice
        print("snackbar destroyed twice: ok", flush=True)
        print(f"foreground hwnd at exit = 0x{fg():X}", flush=True)
        root.destroy()

    root.after(800, step_hide)
    root.mainloop()

    if state.get("hidden_visible") != 0:
        print("FAIL: overlay was still visible after hide()", flush=True)
        return 1
    print("SELFTEST OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
