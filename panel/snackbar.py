"""
Compact always-on-top snackbar, 360 x 46.
=========================================

Contract §5.  One row, three buttons, nothing else::

    [●]  [ readout text, single line, ellipsised ]   [SCAN] [PAUSE] [X]

The bar is draggable by its body/readout but never by the buttons, and it uses
``WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW`` so clicking SCAN/PAUSE/X never pulls
focus away from Chrome (the game pauses when Chrome loses focus).
"""

from __future__ import annotations

import ctypes
import tkinter as tk
import tkinter.font as tkfont
from ctypes import wintypes

try:  # normal package import
    from . import theme
except ImportError:  # running this file directly (self-test)
    import theme  # type: ignore[no-redef]

# ── Win32 ─────────────────────────────────────────────────────────────
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
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

BAR_EX_STYLE = WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW

# ── Layout ────────────────────────────────────────────────────────────
BAR_W = 360
BAR_H = 46
BTN_H = 22
BTN_SCAN_W = 36
BTN_PAUSE_W = 54
BTN_CLOSE_W = 20
BTN_GAP = 3
READOUT_FALLBACK_W = 216
DEFAULT_X = 12

_X_HOVER_BG = "#3a2126"


def resolve_hwnd(win: tk.Misc) -> int:
    """Real top-level HWND of a Tk window (parent frame, not the child)."""
    try:
        wid = int(win.winfo_id())
    except (tk.TclError, ValueError):
        return 0
    parent = int(_GetParent(wintypes.HWND(wid)) or 0) if _GetParent else 0
    return parent or wid


def read_ex_style(hwnd: int) -> int:
    if not hwnd:
        return 0
    return int(_GetWindowLong(wintypes.HWND(hwnd), GWL_EXSTYLE))


class _FlatButton(tk.Frame):
    """Small flat button built from a Frame + Label (exact pixel size, no focus)."""

    def __init__(
        self,
        master: tk.Misc,
        text: str,
        command,
        width: int,
        hover_bg: str = theme.BG_HOVER,
    ) -> None:
        super().__init__(
            master,
            width=width,
            height=BTN_H,
            bg=theme.BG_DEEP,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.pack_propagate(False)
        self.command = command
        self._hover_bg = hover_bg
        self._label = tk.Label(
            self,
            text=text,
            bg=theme.BG_DEEP,
            fg=theme.FG,
            font=theme.FONT_UI_BOLD,
            bd=0,
            highlightthickness=0,
            padx=0,
            pady=0,
            cursor="hand2",
            takefocus=0,
        )
        self._label.pack(fill="both", expand=True)
        for w in (self, self._label):
            w.bind("<Button-1>", self._on_press, add="+")
            w.bind("<ButtonRelease-1>", self._on_release, add="+")
            w.bind("<Enter>", self._on_enter, add="+")
            w.bind("<Leave>", self._on_leave, add="+")

    def _paint(self, bg: str) -> None:
        self.configure(bg=bg)
        self._label.configure(bg=bg)

    def _on_press(self, _event: tk.Event) -> str:
        self._paint(self._hover_bg)
        return "break"

    def _on_release(self, _event: tk.Event) -> str:
        self._paint(self._hover_bg)
        if callable(self.command):
            self.command()
        return "break"

    def _on_enter(self, _event: tk.Event) -> None:
        self._paint(self._hover_bg)

    def _on_leave(self, _event: tk.Event) -> None:
        self._paint(theme.BG_DEEP)

    def set_text(self, text: str) -> None:
        self._label.configure(text=text)

    def set_width(self, width: int) -> None:
        self.configure(width=width)


class SnackBar:
    """Compact always-on-top bar. ~360 x 46 px. Draggable by its body."""

    def __init__(self, master: tk.Misc, on_scan, on_pause, on_close) -> None:
        self.master = master
        self.on_scan = on_scan
        self.on_pause = on_pause
        self.on_close = on_close

        self.hwnd: int = 0
        self._destroyed = False
        self._paused = False
        self._raw_text = ""
        self._drag_offset: tuple[int, int] | None = None

        self.win = tk.Toplevel(master)
        self.win.withdraw()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg=theme.BG)
        self.win.title("cc-snackbar")

        self.body = tk.Frame(self.win, bg=theme.BG, bd=0, highlightthickness=0)
        self.body.pack(fill="both", expand=True)

        # Status dot.
        self.dot = tk.Canvas(
            self.body, width=10, height=10, bg=theme.BG,
            bd=0, highlightthickness=0, takefocus=0,
        )
        self._dot_id = self.dot.create_oval(0, 0, 9, 9, fill=theme.FG_DIM, outline="")
        self.dot.pack(side=tk.LEFT, padx=(6, 4))

        # Buttons (packed right-to-left so they read SCAN PAUSE X).
        self.btn_close = _FlatButton(
            self.body, "✕", self._do_close, BTN_CLOSE_W, hover_bg=_X_HOVER_BG,
        )
        self.btn_close.pack(side=tk.RIGHT, padx=(BTN_GAP, 4))
        self.btn_pause = _FlatButton(
            self.body, "PAUSE", self._do_pause, BTN_PAUSE_W,
        )
        self.btn_pause.pack(side=tk.RIGHT, padx=(BTN_GAP, 0))
        self.btn_scan = _FlatButton(
            self.body, "SCAN", self._do_scan, BTN_SCAN_W,
        )
        self.btn_scan.pack(side=tk.RIGHT, padx=(0, 0))

        # Readout line.
        self._font = tkfont.Font(family="Consolas", size=8)
        self.readout = tk.Label(
            self.body,
            text="",
            bg=theme.BG,
            fg=theme.FG,
            font=theme.FONT_MONO,
            anchor="w",
            bd=0,
            highlightthickness=0,
            padx=0,
            pady=0,
            takefocus=0,
        )
        self.readout.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Dragging: body, readout and dot - never the buttons.
        for w in (self.body, self.readout, self.dot):
            w.bind("<Button-1>", self._drag_start, add="+")
            w.bind("<B1-Motion>", self._drag_move, add="+")
            w.bind("<ButtonRelease-1>", self._drag_end, add="+")
        self.readout.bind("<Configure>", self._on_configure, add="+")

        self.win.update_idletasks()
        self.apply_styles()
        self.place_default()
        self._show_window()

    # ── Win32 ─────────────────────────────────────────────────────────
    def apply_styles(self) -> int:
        """Assert NOACTIVATE|TOOLWINDOW. Idempotent; returns the ex-style."""
        if self._destroyed:
            return 0
        hwnd = resolve_hwnd(self.win)
        if not hwnd:
            return 0
        self.hwnd = hwnd
        cur = read_ex_style(hwnd)
        want = cur | BAR_EX_STYLE
        if want != cur:
            _SetWindowLong(wintypes.HWND(hwnd), GWL_EXSTYLE, want)
        _SetWindowPos(
            wintypes.HWND(hwnd), None, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        )
        return read_ex_style(hwnd)

    def _show_window(self) -> None:
        if self._destroyed:
            return
        self.apply_styles()
        try:
            # Focus-safe: WS_EX_NOACTIVATE suppresses activation for deiconify.
            self.win.deiconify()
            self.win.update_idletasks()
        except tk.TclError:
            return
        self._refit()

    def is_visible(self) -> bool:
        if self._destroyed or not self.hwnd:
            return False
        return bool(_IsWindowVisible(wintypes.HWND(self.hwnd)))

    # ── Placement / dragging ──────────────────────────────────────────
    def place_default(self) -> None:
        """Park it in the dead zone left of the board (x=12, y=centred)."""
        if self._destroyed:
            return
        sh = self.win.winfo_screenheight()
        y = max(0, (sh - BAR_H) // 2)
        self.win.geometry(f"{BAR_W}x{BAR_H}+{DEFAULT_X}+{y}")

    def _drag_start(self, event: tk.Event) -> None:
        self._drag_offset = (
            int(event.x_root) - self.win.winfo_x(),
            int(event.y_root) - self.win.winfo_y(),
        )

    def _drag_move(self, event: tk.Event) -> None:
        if self._drag_offset is None:
            return
        dx, dy = self._drag_offset
        x = int(event.x_root) - dx
        y = int(event.y_root) - dy
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        # Clamp so it can never be dragged fully off screen.
        x = max(0, min(x, sw - BAR_W))
        y = max(0, min(y, sh - BAR_H))
        self.win.geometry(f"+{x}+{y}")

    def _drag_end(self, _event: tk.Event) -> None:
        self._drag_offset = None

    # ── Content ───────────────────────────────────────────────────────
    def _fit_text(self, text: str, avail: int) -> str:
        if avail <= 8:
            return ""
        if self._font.measure(text) <= avail:
            return text
        ell = "\u2026"
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._font.measure(text[:mid] + ell) <= avail:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + ell if lo else ell

    def _refit(self) -> None:
        if self._destroyed:
            return
        w = self.readout.winfo_width()
        if w <= 1:
            w = READOUT_FALLBACK_W
        self.readout.configure(text=self._fit_text(self._raw_text, w - 1))

    def _on_configure(self, _event: tk.Event) -> None:
        self._refit()

    def _set_line(self, text: str) -> None:
        self._raw_text = str(text)
        self._refit()

    def set_status(self, text: str, kind: str = "idle") -> None:
        """kind in {"idle","scanning","paused","ok","error"} -> dot colour."""
        if self._destroyed:
            return
        colour = theme.DOT_COLOURS.get(kind, theme.FG_DIM)
        self.dot.itemconfigure(self._dot_id, fill=colour)
        if text:
            self._set_line(text)

    def set_readout(self, text: str) -> None:
        """Short line, e.g. 'R3C2 -> R3C5   +340   then +180'."""
        self._set_line(text)

    def set_paused(self, paused: bool) -> None:
        """Flips the PAUSE button label to RESUME/PAUSE."""
        if self._destroyed:
            return
        self._paused = bool(paused)
        self.btn_pause.set_text("RESUME" if self._paused else "PAUSE")
        self.btn_pause.set_width(BTN_PAUSE_W)

    def set_move_cells(self, a, b) -> None:
        """Optional: colour the readout. a/b are (row,col) or None."""
        if self._destroyed:
            return
        colour = theme.ACCENT if (a is not None and b is not None) else theme.FG
        self.readout.configure(fg=colour)

    # ── Button handlers ───────────────────────────────────────────────
    def _do_scan(self) -> None:
        if callable(self.on_scan):
            self.on_scan()

    def _do_pause(self) -> None:
        if callable(self.on_pause):
            self.on_pause()

    def _do_close(self) -> None:
        if callable(self.on_close):
            self.on_close()

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


# ── Standalone demo (§8) ──────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    bar = SnackBar(
        root,
        on_scan=lambda: bar.set_status("scanning\u2026", "scanning"),
        on_pause=lambda: bar.set_paused(not bar._paused),
        on_close=lambda: (bar.destroy(), root.destroy()),
    )
    bar.set_status("idle", "idle")
    bar.set_readout("R3C2 -> R3C5   +340   then +180")
    print(f"exstyle = 0x{bar.apply_styles():08X}  hwnd=0x{bar.hwnd:X}")
    root.after(1200, lambda: bar.set_readout("UNCERTAIN \u2014 14 cells unread"))
    root.after(2200, lambda: (bar.destroy(), root.destroy()))
    root.mainloop()
