"""
Mouse click execution using Windows SendInput (no external deps).
"""

import ctypes
from ctypes import wintypes
import time

# Windows constants
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

# ULONG_PTR is pointer-sized unsigned integer
if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong

# Structures
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("mi", MOUSEINPUT),
    ]

def send_input(inputs):
    """Send array of INPUT structures."""
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    ctypes.windll.user32.SendInput(n, ctypes.byref(arr), ctypes.sizeof(INPUT))

def move_mouse(x: int, y: int):
    """Move mouse to absolute screen coordinates (0-65535)."""
    # Convert to absolute coordinates (0-65535)
    screen_w = ctypes.windll.user32.GetSystemMetrics(0)
    screen_h = ctypes.windll.user32.GetSystemMetrics(1)
    ax = int(x * 65535 / screen_w)
    ay = int(y * 65535 / screen_h)
    
    inp = INPUT(INPUT_MOUSE, MOUSEINPUT(ax, ay, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, 0))
    send_input([inp])

def click_at(x: int, y: int, delay: float = 0.05):
    """Click at absolute screen coordinates."""
    move_mouse(x, y)
    time.sleep(delay)
    # Left down
    inp_down = INPUT(INPUT_MOUSE, MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE, 0, 0))
    # Left up
    inp_up = INPUT(INPUT_MOUSE, MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE, 0, 0))
    send_input([inp_down, inp_up])

def double_click_cell(roi: dict, pitch: float, r1: int, c1: int, r2: int, c2: int, delay_between: float = 0.15):
    """Click two cells in sequence (swap = tap-tap)."""
    # Calculate screen coordinates for cell centres
    left, top = roi['left'], roi['top']
    x1 = int(left + c1 * pitch + pitch * 0.5)
    y1 = int(top + r1 * pitch + pitch * 0.5)
    x2 = int(left + c2 * pitch + pitch * 0.5)
    y2 = int(top + r2 * pitch + pitch * 0.5)
    
    click_at(x1, y1)
    time.sleep(delay_between)
    click_at(x2, y2)


# Test
if __name__ == '__main__':
    print("Clicker module loaded. Test with: click_at(100, 100)")