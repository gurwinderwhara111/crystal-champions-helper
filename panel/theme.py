"""
Shared UI tokens for the compact manual-mode panel.
===================================================

One dark palette, one accent.  Plain module-level strings only - no classes,
no logic - so both ``snackbar`` and ``arrow_overlay`` stay visually identical.

``TRANSPARENT_KEY`` is the tkinter ``-transparentcolor`` key.  It is a colour
that is deliberately never used by any drawn arrow pixel; everything the overlay
does NOT draw becomes fully transparent (and click-through) on screen.
"""

from __future__ import annotations

# ── Surfaces ──────────────────────────────────────────────────────────
BG = "#15171d"          # snackbar body
BG_DEEP = "#0b0c10"     # buttons, label chip on the arrow
BG_HOVER = "#232733"    # button hover

# ── Text ──────────────────────────────────────────────────────────────
FG = "#e8eaee"
FG_DIM = "#7b818f"

# ── Accents / state ───────────────────────────────────────────────────
ACCENT = "#00d4aa"      # pickup ring, "scanning" dot, primary accent
ACCENT_2 = "#ffb454"    # destination ring / arrow head tint
WARN = "#f0a53a"        # paused
OK = "#3ddc84"          # ok
ERR = "#ff5c5c"         # error

# ── Arrow colours (manual mode) ───────────────────────────────────────
ARROW_NOW = "#00ff00"      # GREEN - current move to tap NOW
ARROW_NEXT = "#ff8c00"     # ORANGE - follow-up/next move text

# ── Top-3 move colours ────────────────────────────────────────────────
MOVE_1 = "#00ff00"         # GREEN - best move
MOVE_2 = "#00bfff"         # BLUE - 2nd best move
MOVE_3 = "#ffff00"         # YELLOW - 3rd best move

# ── Misc ──────────────────────────────────────────────────────────────
GRID_LINE = "#2a2f3a"

# Keyed-out colour.  MUST NOT appear in any arrow graphic.
TRANSPARENT_KEY = "#010203"

# ── Fonts (tk font-spec strings: "<family> <size> [style]") ───────────
FONT_MONO = "Consolas 8"
FONT_MONO_BOLD = "Consolas 8 bold"
FONT_UI = "{Segoe UI} 8"
FONT_UI_BOLD = "{Segoe UI} 8 bold"
FONT_UI_SMALL = "{Segoe UI} 7 bold"

# ── Status dot colours, keyed by SnackBar.set_status(kind=...) ────────
DOT_COLOURS = {
    "idle": FG_DIM,
    "scanning": ACCENT,
    "paused": WARN,
    "ok": OK,
    "error": ERR,
}

__all__ = [
    "BG", "BG_DEEP", "BG_HOVER", "FG", "FG_DIM",
    "ACCENT", "ACCENT_2", "WARN", "OK", "ERR",
    "ARROW_NOW", "ARROW_NEXT",
    "MOVE_1", "MOVE_2", "MOVE_3",
    "GRID_LINE", "TRANSPARENT_KEY",
    "FONT_MONO", "FONT_MONO_BOLD", "FONT_UI", "FONT_UI_BOLD", "FONT_UI_SMALL",
    "DOT_COLOURS",
]
