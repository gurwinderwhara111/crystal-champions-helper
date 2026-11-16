"""
Borderless click-through arrow overlay, drawn on top of the game board.
======================================================================

Contract §4.  The window sits exactly over ``capture_region(grid)`` so that
crop coordinates are window coordinates::

    window_x = (x0 - left) + col * pitch_x
    window_y = (y0 - top)  + row * pitch_y

Style bits (all required):

    WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW

``WS_EX_TRANSPARENT`` lets every click fall through to Chrome underneath.
``WS_EX_NOACTIVATE`` keeps focus where it is - the game pauses on focus loss.
Combined with ``-transparentcolor`` only the drawn arrow is opaque.

The single most important method is :meth:`ArrowOverlay.hide`: it withdraws the
window and *blocks* until the pixels are gone, because the capture loop grabs
the screen the instant it returns.
"""

from __future__ import annotations

import ctypes
import math
import time
import tkinter as tk
from ctypes import wintypes

try:  # normal package import
    from . import theme
except ImportError:  # running this file directly (self-test)
    import theme  # type: ignore[no-redef]

# ── Win32 ─────────────────────────────────────────────────────────────
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020

_user32 = ctypes.WinDLL("user32", use_last_error=True)


def _bind(fn, argtypes, restype):
    if fn is None:
        return None
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


_GetWindowLong = _bind(
    getattr(_user32, "GetWindowLongPtrW", None) or _user32.GetWindowLongW,
    (wintypes.HWND, ctypes.c_int),
    ctypes.c_ssize_t,
)
_SetWindowLong = _bind(
    getattr(_user32, "SetWindowLongPtrW", None) or _user32.SetWindowLongW,
    (wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t),
    ctypes.c_ssize_t,
)
_GetParent = _bind(_user32.GetParent, (wintypes.HWND,), wintypes.HWND)
_SetWindowPos = _bind(
    _user32.SetWindowPos,
    (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
     ctypes.c_int, ctypes.c_int, ctypes.c_uint),
    wintypes.BOOL,
)
_IsWindowVisible = _bind(_user32.IsWindowVisible, (wintypes.HWND,), wintypes.BOOL)
_ShowWindow = _bind(_user32.ShowWindow, (wintypes.HWND, ctypes.c_int), wintypes.BOOL)
_GetForegroundWindow = _bind(_user32.GetForegroundWindow, (), wintypes.HWND)

SW_SHOWNOACTIVATE = 4
SW_HIDE = 0

# How long to let DWM composite the withdrawal before the caller screenshots.
SETTLE_S = 0.035

OVERLAY_EX_STYLE = (
    WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
)


def resolve_hwnd(win: tk.Misc) -> int:
    """Real top-level HWND of a Tk window (parent frame, not the child)."""
    try:
        wid = int(win.winfo_id())
    except (tk.TclError, ValueError):
        return 0
    parent = int(_GetParent(wintypes.HWND(wid)) or 0) if _GetParent else 0
    return parent or wid


def read_ex_style(hwnd: int) -> int:
    """Current GWL_EXSTYLE of ``hwnd`` (0 when the handle is bogus)."""
    if not hwnd:
        return 0
    return int(_GetWindowLong(wintypes.HWND(hwnd), GWL_EXSTYLE))


class ArrowOverlay:
    """Borderless, click-through, never-focused arrow overlay drawn ON the board."""

    #: ring radius as a fraction of the cell pitch
    RING_FRAC = 0.36
    #: ring line thickness as a fraction of the cell pitch
    RING_WIDTH_FRAC = 0.075

    def __init__(self, master: tk.Misc) -> None:
        self.master = master
        self.grid: dict | None = None
        self.hwnd: int = 0
        self._destroyed = False
        self._master_visible = True       # hotkey master toggle
        self._shown = False
        self._last_move: tuple[tuple[int, int], tuple[int, int], str] | None = None

        self.win = tk.Toplevel(master)
        self.win.withdraw()
        self.win.overrideredirect(True)
        try:
            self.win.attributes("-topmost", True)
            self.win.attributes("-transparentcolor", theme.TRANSPARENT_KEY)
        except tk.TclError:
            # No -transparentcolor support: still usable, just not keyed out.
            pass
        self.win.configure(bg=theme.TRANSPARENT_KEY)
        self.win.title("cc-arrows")

        self.canvas = tk.Canvas(
            self.win,
            bg=theme.TRANSPARENT_KEY,
            bd=0,
            highlightthickness=0,
            takefocus=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self.win.update_idletasks()
        self.apply_styles()

    # ── Win32 plumbing ────────────────────────────────────────────────
    def apply_styles(self) -> int:
        """(Re)assert the extended styles.  Idempotent; returns the ex-style."""
        if self._destroyed:
            return 0
        hwnd = resolve_hwnd(self.win)
        if not hwnd:
            return 0
        self.hwnd = hwnd
        cur = read_ex_style(hwnd)
        want = cur | OVERLAY_EX_STYLE
        if want != cur:
            _SetWindowLong(wintypes.HWND(hwnd), GWL_EXSTYLE, want)
        # Make the frame change take effect without moving / activating.
        _SetWindowPos(
            wintypes.HWND(hwnd), None, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        )
        return read_ex_style(hwnd)

    # ── Geometry ──────────────────────────────────────────────────────
    def set_grid(self, grid: dict) -> None:
        """Position the overlay window exactly over capture_region(grid)."""
        if self._destroyed:
            return
        self.grid = dict(grid)
        w = int(grid["width"])
        h = int(grid["height"])
        left = int(grid["left"])
        top = int(grid["top"])
        self.canvas.configure(width=w, height=h)
        self.win.geometry(f"{w}x{h}+{left}+{top}")
        self.win.update_idletasks()
        self.apply_styles()
        self._prime_geometry()
        # Redraw the current move against the new lattice.
        if self._last_move is not None:
            a, b, label = self._last_move
            self._draw(a, b, label)

    def _prime_geometry(self) -> None:
        """Map once while empty so the WM really adopts the requested size.

        Tk ignores ``wm geometry`` until the toplevel has been mapped at least
        once.  The window is fully colour-keyed out at this point, so this
        brief map is invisible; it happens during calibration only.
        """
        if self._destroyed:
            return
        shown = self._shown
        try:
            if not shown:
                self.win.deiconify()
                self.win.update()
            self.win.withdraw()
            self.win.update_idletasks()
        except tk.TclError:
            return
        self._shown = False

    def cell_to_window(self, r: int, c: int) -> tuple[float, float]:
        """Cell (row, col) -> pixel inside the overlay window."""
        g = self.grid
        if g is None:
            raise RuntimeError("ArrowOverlay.set_grid() must be called first")
        x = (float(g["x0"]) - float(g["left"])) + c * float(g["pitch_x"])
        y = (float(g["y0"]) - float(g["top"])) + r * float(g["pitch_y"])
        return x, y

    # ── Drawing ───────────────────────────────────────────────────────
    def clear(self) -> None:
        if self._destroyed:
            return
        try:
            self.canvas.delete("all")
        except tk.TclError:
            pass

    def _fit_chip(self, x: float, y: float, text: str, anchor: str) -> None:
        """Draw a text chip with a dark background, clamped inside the canvas."""
        c = self.canvas
        tid = c.create_text(
            x, y, text=text, anchor=anchor,
            fill=theme.FG, font=theme.FONT_MONO_BOLD,
        )
        x1, y1, x2, y2 = c.bbox(tid)
        pad = 4
        w = self._canvas_w
        h = self._canvas_h
        dx = dy = 0.0
        if x1 - pad < 0:
            dx = pad - (x1 - pad) + 1
        elif x2 + pad > w:
            dx = (w - pad) - (x2 + pad) - 1
        if y1 - pad < 0:
            dy = pad - (y1 - pad) + 1
        elif y2 + pad > h:
            dy = (h - pad) - (y2 + pad) - 1
        if dx or dy:
            c.move(tid, dx, dy)
            x1, y1, x2, y2 = c.bbox(tid)
        rid = c.create_rectangle(
            x1 - pad, y1 - pad, x2 + pad, y2 + pad,
            fill=theme.BG_DEEP, outline=theme.GRID_LINE, width=1,
        )
        c.tag_lower(rid, tid)

    def _badge(self, x: float, y: float, n: str, colour: str) -> None:
        """Small numbered badge marking tap order."""
        c = self.canvas
        rad = 9
        c.create_oval(
            x - rad, y - rad, x + rad, y + rad,
            fill=theme.BG_DEEP, outline=colour, width=2,
        )
        c.create_text(
            x, y, text=n, fill=colour, font=theme.FONT_UI_SMALL, anchor="center",
        )

    def _draw(self, a: tuple[int, int], b: tuple[int, int], label: str, colour: str = None) -> None:
        """(Re)build the arrow graphic.  Pure canvas work, never blocks."""
        if self._destroyed or self.grid is None:
            return
        g = self.grid
        c = self.canvas
        
        # Use provided colour or default
        if colour is None:
            colour = theme.ARROW_NOW
            
        self._canvas_w = int(g["width"])
        self._canvas_h = int(g["height"])
        # Don't clear canvas if we're drawing multiple moves
        # c.delete("all")  -- removed, will be called in show_moves

        ax, ay = self.cell_to_window(int(a[0]), int(a[1]))
        bx, by = self.cell_to_window(int(b[0]), int(b[1]))
        pitch = (float(g["pitch_x"]) + float(g["pitch_y"])) / 2.0
        rad = max(10.0, pitch * self.RING_FRAC)
        ring_w = max(3, int(round(pitch * self.RING_WIDTH_FRAC)))

        # Pickup cell: coloured ring, thick.
        c.create_oval(
            ax - rad, ay - rad, ax + rad, ay + rad,
            outline=colour, width=ring_w,
        )
        # Destination cell: coloured ring, thick.
        c.create_oval(
            bx - rad, by - rad, bx + rad, by + rad,
            outline=colour, width=ring_w,
        )

        # Shaft from ring edge to ring edge, head at B.
        dx, dy = bx - ax, by - ay
        dist = math.hypot(dx, dy)
        if dist >= 1.0:
            ux, uy = dx / dist, dy / dist
            sx, sy = ax + ux * (rad + 2), ay + uy * (rad + 2)
            ex, ey = bx - ux * (rad + 3), by - uy * (rad + 3)
            c.create_line(
                sx, sy, ex, ey,
                fill=colour, width=max(3, ring_w - 1),
                arrow=tk.LAST, arrowshape=(16, 20, 7), capstyle=tk.ROUND,
            )
            mid_x, mid_y = (sx + ex) / 2.0, (sy + ey) / 2.0
            # Offset the chip perpendicular to the arrow so it never covers it.
            ox, oy = -uy, ux
            chip_x = mid_x + ox * 22
            chip_y = mid_y + oy * 22
            anchor = "center"
        else:
            chip_x, chip_y = ax + rad + 6, ay - rad - 6
            anchor = "w"

        # Tap order badges: 1 = pick up, 2 = drop.
        self._badge(ax - rad * 0.78, ay - rad * 0.78, "1", colour)
        self._badge(bx - rad * 0.78, by - rad * 0.78, "2", colour)

        if label:
            self._fit_chip(chip_x, chip_y, label, anchor)

    def show_move(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
        label: str = "",
        colour: str = None,
    ) -> None:
        """Draw an arrow a -> b with rings and an optional label. Non-blocking."""
        if self._destroyed:
            return
        if self.grid is None:
            raise RuntimeError("ArrowOverlay.set_grid() must be called first")
        a = (int(a[0]), int(a[1]))
        b = (int(b[0]), int(b[1]))
        self._last_move = (a, b, label)
        self.canvas.delete("all")
        self._draw(a, b, label, colour)  # Pass colour!
        self._show_window()

    def show_moves(
        self,
        moves: list,
    ) -> None:
        """Draw top 3 moves with different colors.
        
        Args:
            moves: List of dicts with keys 'a', 'b', 'label', 'colour'
                   e.g. [{'a': (3,2), 'b': (3,5), 'label': '+340', 'colour': '#00ff00'}, ...]
        """
        if self._destroyed:
            return
        if self.grid is None:
            raise RuntimeError("ArrowOverlay.set_grid() must be called first")
        
        self.canvas.delete("all")
        
        # Draw each move with its assigned colour
        for i, move in enumerate(moves[:3]):  # Max 3 moves
            a = (int(move['a'][0]), int(move['a'][1]))
            b = (int(move['b'][0]), int(move['b'][1]))
            label = move.get('label', '')
            colour = move.get('colour', theme.MOVE_1 if i == 0 else theme.MOVE_2 if i == 1 else theme.MOVE_3)
            
            self._draw(a, b, label, colour)
        
        self._show_window()

    def _show_window(self) -> None:
        if self._destroyed or not self._master_visible or self._shown:
            return
        self.apply_styles()
        try:
            # deiconify() is focus-safe here: WS_EX_NOACTIVATE suppresses the
            # activation, and it keeps Tk's own mapped-state consistent so the
            # withdraw() in hide() really is a ShowWindow(SW_HIDE).
            self.win.deiconify()
            self.win.update_idletasks()
        except tk.TclError:
            return
        self._shown = True

    # ── Visibility ────────────────────────────────────────────────────
    def hide(self) -> None:
        """Hide and BLOCK until the pixels are actually off screen.

        The capture loop grabs the screen the moment this returns, so this is
        deliberately synchronous: withdraw -> flush -> settle sleep.
        """
        if self._destroyed:
            return
        try:
            self.win.withdraw()
            if self.hwnd:
                # Belt and braces: even if Tk thinks it is already unmapped,
                # force the HWND itself hidden.
                _ShowWindow(wintypes.HWND(self.hwnd), SW_HIDE)
            self.win.update_idletasks()
        except tk.TclError:
            return
        self._shown = False
        time.sleep(SETTLE_S)

    def is_visible(self) -> bool:
        """True when the OS reports the overlay window as visible."""
        if self._destroyed or not self.hwnd:
            return False
        return bool(_IsWindowVisible(wintypes.HWND(self.hwnd)))

    def set_visible(self, visible: bool) -> None:
        """Master toggle (hotkey).  False hides arrows entirely."""
        if self._destroyed:
            return
        self._master_visible = bool(visible)
        if not visible:
            self.hide()
        else:
            self.apply_styles()
            if self._last_move is not None:
                a, b, label = self._last_move
                self._draw(a, b, label)
                self._show_window()

    # ── Teardown ──────────────────────────────────────────────────────
    def destroy(self) -> None:
        """Safe to call twice."""
        if self._destroyed:
            return
        self._destroyed = True
        try:
            self.win.destroy()
        except tk.TclError:
            pass
        self.win = None  # type: ignore[assignment]
        self.hwnd = 0
        self._shown = False


# ── Standalone demo (§8) ──────────────────────────────────────────────
if __name__ == "__main__":
    DEMO_GRID = {
        "x0": 240.8, "y0": 190.8, "pitch_x": 81.6, "pitch_y": 81.5,
        "left": 200, "top": 150, "width": 665, "height": 665,
        "cols": 8, "rows": 8,
    }
    root = tk.Tk()
    root.withdraw()
    ov = ArrowOverlay(root)
    ov.set_grid(DEMO_GRID)
    ov.show_move((3, 2), (3, 5), "+340")
    print(f"exstyle = 0x{ov.apply_styles():08X}  hwnd=0x{ov.hwnd:X}")
    root.after(900, lambda: (ov.hide(), print("visible after hide:", ov.is_visible())))
    root.after(1200, lambda: ov.show_move((1, 1), (1, 4), "+120"))
    root.after(2200, lambda: (ov.destroy(), root.destroy()))
    root.mainloop()
