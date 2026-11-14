"""
Crystal Champions — Control Panel Overlay (Integrated)
=======================================================

Enhanced panel with:
- Full auto-player integration (core.auto_player.IntegratedAutoPlayer)
- Real-time HUD display (score, level, multiplier, timer)
- Move prediction display
- Live 8x8 grid with confidence
- Detailed logging panel
- HUD calibrator
- Board RECOVER / CLOSE
- Hotkey kill switch (Ctrl+Shift+X)
- Training data export button
"""

from __future__ import annotations
import sys
import os
import time
import threading
import queue
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import ctypes
from pathlib import Path
from datetime import datetime

# Add project paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

try:
    import cv2
    import numpy as np
    import mss
    from detector import detect_board, find_board
    from core.auto_player import IntegratedAutoPlayer, AutoConfig
    from core.logging_system import get_logger
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# ─── Windows Constants ────────────────────────────────────────────────

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
LWA_ALPHA = 0x00000002

# Hotkey
VK_X = 0x58
VK_CONTROL = 0x11
VK_SHIFT = 0x10

# ─── Panel Geometry ──────────────────────────────────────────────────

PANEL_W = 380
PANEL_H = 720
DEADZONE_X = 20
DEADZONE_Y = 50

# ─── Colors ──────────────────────────────────────────────────────────

BG_COLOR = '#1a1a2e'
FG_COLOR = '#e0e0e0'
ACCENT_COLOR = '#00d4aa'
WARN_COLOR = '#ff6b6b'
DIM_COLOR = '#666666'
BTN_BG = '#2d2d44'
BTN_HOVER = '#3d3d5c'
BTN_DISABLED = '#1a1a2e'
GRID_BG = '#0d0d1a'
GEM_COLORS = {
    'g': '#4ade80', 'o': '#fb923c', 'b': '#60a5fa',
    'w': '#f8fafc', 'r': '#f87171', 'p': '#a78bfa', 'y': '#facc15',
    '?': '#333333'
}

# ─── Main Panel Class ────────────────────────────────────────────────

class OverlayPanel:
    """Enhanced control panel with full auto-player integration."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        
        # State
        self.state = 'BOOT'  # BOOT -> READY -> PLAYING -> DONE
        self.roi = None
        self.hud_roi = None
        self.running = False
        self.warmup_done = False
        self.close_stage = 0
        
        # Config paths
        self.roi_path = Path(__file__).parent.parent / "detector" / "roi.json"
        self.hud_roi_path = Path(__file__).parent.parent / "detector" / "hud_roi.json"
        
        # Auto-player
        self.auto_player = None
        self.auto_config = AutoConfig()
        
        # Logging
        self.logger = get_logger("panel", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.logger.register_ui_callback(self._on_log_message)
        
        # UI update queue
        self.ui_queue = queue.Queue()
        
        # Setup
        self._setup_window()
        self._setup_ui()
        self._register_hotkey()
        self._position_panel()
        self._load_rois()
        
        # Start UI update loop
        self.root.after(50, self._process_ui_queue)
        
        # Start warmup
        self.root.after(100, self._warmup)
        
    def _setup_window(self):
        """Apply Windows extended styles."""
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
        if hwnd == 0:
            hwnd = self.root.winfo_id()
            
        exstyle = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        new_style = exstyle | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_LAYERED
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
        ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, 210, LWA_ALPHA)
        
    def _position_panel(self):
        self.root.geometry(f'{PANEL_W}x{PANEL_H}+{DEADZONE_X}+{DEADZONE_Y}')
        self.root.deiconify()
        
    # ─── UI Setup ─────────────────────────────────────────────────────
    
    def _setup_ui(self):
        self.root.configure(bg=BG_COLOR)
        
        # Title bar
        title_frame = tk.Frame(self.root, bg='#0f0f1a', height=36)
        title_frame.pack(fill='x', padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text='CRYSTAL CHAMPIONS', bg='#0f0f1a', fg=ACCENT_COLOR,
                 font=('Consolas', 10, 'bold')).pack(side='left', padx=12, pady=8)
        
        self.state_label = tk.Label(title_frame, text='BOOT', bg='#0f0f1a', fg=DIM_COLOR,
                                    font=('Consolas', 9))
        self.state_label.pack(side='right', padx=12, pady=8)
        
        title_frame.bind('<Button-1>', self._drag_start)
        title_frame.bind('<B1-Motion>', self._drag_motion)
        
        # Status / Log display
        log_frame = tk.LabelFrame(self.root, text='LOG', bg=BG_COLOR, fg=DIM_COLOR,
                                  font=('Consolas', 8), bd=1, relief='solid')
        log_frame.pack(fill='x', padx=12, pady=4)
        
        self.status_text = tk.Text(log_frame, height=6, bg='#111122', fg=FG_COLOR,
                                   font=('Consolas', 7), relief='flat', bd=0,
                                   wrap='word', state='disabled')
        self.status_text.pack(fill='x', padx=4, pady=4)
        self._log('Initializing...')
        
        # HUD Display
        hud_frame = tk.LabelFrame(self.root, text='HUD (score/level/timer)', bg=BG_COLOR, fg=DIM_COLOR,
                                  font=('Consolas', 8), bd=1, relief='solid')
        hud_frame.pack(fill='x', padx=12, pady=4)
        
        self.hud_labels = {}
        hud_grid = tk.Frame(hud_frame, bg=BG_COLOR)
        hud_grid.pack(fill='x', padx=8, pady=6)
        
        for i, (key, label) in enumerate([('score', 'SCORE'), ('level', 'LEVEL'), 
                                           ('mult', 'MULT'), ('timer', 'TIMER'),
                                           ('moves', 'MOVES')]):
            frame = tk.Frame(hud_grid, bg=BG_COLOR)
            frame.pack(side='left', expand=True, fill='x', padx=2)
            tk.Label(frame, text=label, bg=BG_COLOR, fg=DIM_COLOR,
                     font=('Consolas', 7)).pack()
            self.hud_labels[key] = tk.Label(frame, text='--', bg=BG_COLOR, fg=ACCENT_COLOR,
                                             font=('Consolas', 10, 'bold'))
            self.hud_labels[key].pack()
            
        # HUD Calibrator
        hud_cal_frame = tk.LabelFrame(self.root, text='HUD CALIBRATOR', bg=BG_COLOR, fg=DIM_COLOR,
                                      font=('Consolas', 8), bd=1, relief='solid')
        hud_cal_frame.pack(fill='x', padx=12, pady=4)
        
        self.hud_status = tk.Label(hud_cal_frame, text='Not calibrated', bg=BG_COLOR, fg=DIM_COLOR,
                                   font=('Consolas', 8))
        self.hud_status.pack(pady=4)
        
        hud_btn_frame = tk.Frame(hud_cal_frame, bg=BG_COLOR)
        hud_btn_frame.pack(fill='x', padx=8, pady=6)
        
        self.hud_cut_btn = self._make_button(hud_btn_frame, 'CUT REGION', self._start_hud_calibration,
                                             state='normal', width=12)
        self.hud_cut_btn.pack(side='left', padx=2, expand=True, fill='x')
        
        self.hud_pause_btn = self._make_button(hud_btn_frame, 'PAUSE', self._toggle_hud_capture,
                                               state='disabled', width=8)
        self.hud_pause_btn.pack(side='left', padx=2)
        
        self.hud_recal_btn = self._make_button(hud_btn_frame, 'RECALIBRATE', self._start_hud_calibration,
                                               state='disabled', width=10)
        self.hud_recal_btn.pack(side='right', padx=2, expand=True, fill='x')
        
        # Move Prediction Display
        pred_frame = tk.LabelFrame(self.root, text='MOVE PREDICTION', bg=BG_COLOR, fg=DIM_COLOR,
                                   font=('Consolas', 8), bd=1, relief='solid')
        pred_frame.pack(fill='x', padx=12, pady=4)
        
        self.pred_labels = {}
        pred_grid = tk.Frame(pred_frame, bg=BG_COLOR)
        pred_grid.pack(fill='x', padx=8, pady=6)
        
        for key, label in [('move', 'MOVE'), ('points', 'PTS'), ('chain', 'CHAIN'),
                           ('specials', 'SPEC'), ('det', 'DET')]:
            frame = tk.Frame(pred_grid, bg=BG_COLOR)
            frame.pack(side='left', expand=True, fill='x', padx=2)
            tk.Label(frame, text=label, bg=BG_COLOR, fg=DIM_COLOR,
                     font=('Consolas', 7)).pack()
            self.pred_labels[key] = tk.Label(frame, text='--', bg=BG_COLOR, fg=ACCENT_COLOR,
                                              font=('Consolas', 10, 'bold'))
            self.pred_labels[key].pack()
            
        # 8x8 Live Grid
        grid_frame = tk.LabelFrame(self.root, text='LIVE GRID (confidence)', bg=BG_COLOR, fg=DIM_COLOR,
                                   font=('Consolas', 8), bd=1, relief='solid')
        grid_frame.pack(fill='both', expand=True, padx=12, pady=4)
        
        self.grid_cells = []
        for r in range(8):
            row_frame = tk.Frame(grid_frame, bg=BG_COLOR)
            row_frame.pack(fill='x', padx=2, pady=1)
            row_cells = []
            for c in range(8):
                lbl = tk.Label(row_frame, text='--', width=3, height=1,
                               bg=GRID_BG, fg=DIM_COLOR, font=('Consolas', 7, 'bold'),
                               relief='flat', bd=1)
                lbl.pack(side='left', padx=1, pady=1)
                row_cells.append(lbl)
            self.grid_cells.append(row_cells)
            
        # Control Buttons
        btn_frame = tk.Frame(self.root, bg=BG_COLOR)
        btn_frame.pack(fill='x', padx=12, pady=8)
        
        self.recover_btn = self._make_button(btn_frame, 'RECOVER', self._on_recover,
                                             state='disabled', width=10)
        self.recover_btn.pack(side='left', padx=4, expand=True, fill='x')
        
        self.play_btn = self._make_button(btn_frame, 'PLAY', self._on_play,
                                          state='disabled', width=10)
        self.play_btn.pack(side='left', padx=4, expand=True, fill='x')
        
        self.pause_btn = self._make_button(btn_frame, 'PAUSE', self._on_pause,
                                           state='disabled', width=10)
        self.pause_btn.pack(side='left', padx=4, expand=True, fill='x')
        
        self.close_btn = self._make_button(btn_frame, 'CLOSE', self._on_close,
                                           state='disabled', width=10, warn=True)
        self.close_btn.pack(side='right', padx=4, expand=True, fill='x')
        
        # Debug / Export Buttons
        debug_frame = tk.Frame(self.root, bg=BG_COLOR)
        debug_frame.pack(fill='x', padx=12, pady=4)
        
        self.export_btn = self._make_button(debug_frame, 'EXPORT DATA', self._export_data,
                                            state='normal', width=12)
        self.export_btn.pack(side='left', padx=2, expand=True, fill='x')
        
        self.debug_btn = self._make_button(debug_frame, 'DEBUG FRAME', self._save_debug_frame,
                                           state='normal', width=12)
        self.debug_btn.pack(side='right', padx=2, expand=True, fill='x')
        
        # Hotkey hint
        tk.Label(self.root, text='Kill: Ctrl+Shift+X', bg=BG_COLOR, fg=DIM_COLOR,
                 font=('Consolas', 7)).pack(side='bottom', pady=4)
        
    def _make_button(self, parent, text, command, state='normal', width=10, warn=False):
        bg = WARN_COLOR if warn else BTN_BG
        active_bg = '#ff8b8b' if warn else BTN_HOVER
        fg = '#ffffff' if warn else FG_COLOR
        
        btn = tk.Button(parent, text=text, command=command, state=state,
                        bg=bg, fg=fg, activebackground=active_bg, activeforeground=fg,
                        font=('Consolas', 9, 'bold'), relief='flat', bd=0,
                        width=width, height=2, cursor='hand2')
        
        def on_enter(e):
            if btn['state'] != 'disabled':
                btn.configure(bg=active_bg)
        def on_leave(e):
            if btn['state'] != 'disabled':
                btn.configure(bg=bg)
        
        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)
        return btn
        
    def _drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y
        
    def _drag_motion(self, event):
        x = self.root.winfo_x() + event.x - self._drag_x
        y = self.root.winfo_y() + event.y - self._drag_y
        self.root.geometry(f'+{x}+{y}')
        
    # ─── Logging ──────────────────────────────────────────────────────
    
    def _log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        if hasattr(self, 'status_text'):
            self.status_text.configure(state='normal')
            self.status_text.insert('end', line + '\n')
            self.status_text.see('end')
            self.status_text.configure(state='disabled')
        if self.logger:
            self.logger.info(msg)
            
    def _on_log_message(self, msg: str):
        """Callback from logger for UI updates."""
        self.ui_queue.put(('log', msg))
        
    def _process_ui_queue(self):
        """Process UI updates from background threads."""
        try:
            while True:
                msg_type, data = self.ui_queue.get_nowait()
                if msg_type == 'log':
                    self._log(data)
                elif msg_type == 'hud':
                    self._update_hud_display(data)
                elif msg_type == 'prediction':
                    self._update_prediction_display(data)
                elif msg_type == 'grid':
                    self._render_grid(data)
                elif msg_type == 'state':
                    self._set_state(data)
        except queue.Empty:
            pass
        self.root.after(50, self._process_ui_queue)
        
    # ─── HUD Calibrator ──────────────────────────────────────────────
    
    def _start_hud_calibration(self):
        if hasattr(self, 'hud_calibrating') and self.hud_calibrating:
            return
        self.hud_calibrating = True
        self._log('HUD calibration: drag rectangle on screen, ENTER=confirm, ESC=cancel')
        
        self.hud_selector = tk.Toplevel(self.root)
        self.hud_selector.overrideredirect(True)
        self.hud_selector.attributes('-topmost', True)
        self.hud_selector.attributes('-alpha', 0.3)
        self.hud_selector.configure(bg='black')
        
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.hud_selector.geometry(f'{sw}x{sh}+0+0')
        
        self.hud_canvas = tk.Canvas(self.hud_selector, bg='black', highlightthickness=0, cursor='crosshair')
        self.hud_canvas.pack(fill='both', expand=True)
        
        self.hud_instr = tk.Label(self.hud_selector, 
            text='DRAG to select HUD strip (score/level/timer)  |  ENTER=Confirm  ESC=Cancel',
            bg='#000000', fg='#00ff00', font=('Consolas', 14, 'bold'))
        self.hud_instr.place(relx=0.5, rely=0.05, anchor='center')
        
        self.hud_rect_id = None
        self.hud_rect_start = None
        self.hud_rect_current = None
        
        self.hud_canvas.bind('<Button-1>', self._hud_on_drag_start)
        self.hud_canvas.bind('<B1-Motion>', self._hud_on_drag_motion)
        self.hud_canvas.bind('<ButtonRelease-1>', self._hud_on_drag_end)
        self.hud_selector.bind('<Return>', self._hud_on_confirm)
        self.hud_selector.bind('<Escape>', self._hud_on_cancel)
        self.hud_selector.focus_force()
        
    def _hud_on_drag_start(self, event):
        self.hud_rect_start = (event.x, event.y)
        if self.hud_rect_id:
            self.hud_canvas.delete(self.hud_rect_id)
        self.hud_rect_id = self.hud_canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                             outline='#00ff00', width=2, dash=(4, 4))
        
    def _hud_on_drag_motion(self, event):
        if self.hud_rect_id and self.hud_rect_start:
            x1, y1 = self.hud_rect_start
            x2, y2 = event.x, event.y
            self.hud_canvas.coords(self.hud_rect_id, x1, y1, x2, y2)
        
    def _hud_on_drag_end(self, event):
        if self.hud_rect_start:
            x1, y1 = self.hud_rect_start
            x2, y2 = event.x, event.y
            rx = self.hud_selector.winfo_rootx()
            ry = self.hud_selector.winfo_rooty()
            self.hud_rect_current = (min(x1, x2) + rx, min(y1, y2) + ry, abs(x2 - x1), abs(y2 - y1))
    
    def _hud_on_confirm(self, event=None):
        if self.hud_rect_current:
            x, y, w, h = self.hud_rect_current
            if w > 10 and h > 10:
                self.hud_roi = {'left': x, 'top': y, 'width': w, 'height': h}
                self._save_hud_roi()
                self._update_hud_ui(True)
                self._log(f'HUD region set: {self.hud_roi}')
                # Update auto-player if running
                if self.auto_player:
                    self.auto_player.hud_roi = self.hud_roi
            else:
                self._log('HUD region too small, cancelled')
        self._close_hud_selector()
        
    def _hud_on_cancel(self, event=None):
        self._log('HUD calibration cancelled')
        self._close_hud_selector()
        
    def _close_hud_selector(self):
        if self.hud_selector:
            self.hud_selector.destroy()
            self.hud_selector = None
            self.hud_canvas = None
            self.hud_instr = None
            self.hud_rect_id = None
            self.hud_rect_start = None
            self.hud_rect_current = None
        self.hud_calibrating = False
        
    def _update_hud_ui(self, calibrated: bool):
        if calibrated and self.hud_roi:
            self.hud_status.configure(text=f'Calibrated: {self.hud_roi["left"]},{self.hud_roi["top"]} {self.hud_roi["width"]}x{self.hud_roi["height"]}', fg=ACCENT_COLOR)
            self.hud_pause_btn.configure(state='normal')
            self.hud_recal_btn.configure(state='normal')
            self.hud_cut_btn.configure(text='RE-CUT')
        else:
            self.hud_status.configure(text='Not calibrated', fg=DIM_COLOR)
            self.hud_pause_btn.configure(state='disabled')
            self.hud_recal_btn.configure(state='disabled')
            self.hud_cut_btn.configure(text='CUT REGION')
            
    def _toggle_hud_capture(self):
        if self.auto_player:
            self.auto_player.config.hud_parse_interval = 0 if self.auto_player.config.hud_parse_interval > 0 else 5
            self.hud_pause_btn.configure(text='START' if self.auto_player.config.hud_parse_interval == 0 else 'PAUSE')
            self._log(f'HUD capture {"PAUSED" if self.auto_player.config.hud_parse_interval == 0 else "RESUMED"}')
            
    def _load_rois(self):
        # Load board ROI
        if self.roi_path.exists():
            try:
                self.roi = json.load(open(self.roi_path))
                self._log(f'Board ROI loaded: {self.roi}')
            except Exception as e:
                self._log(f'Board ROI load error: {e}')
                
        # Load HUD ROI
        if self.hud_roi_path.exists():
            try:
                self.hud_roi = json.load(open(self.hud_roi_path))
                self._update_hud_ui(True)
                self._log(f'HUD ROI loaded: {self.hud_roi}')
            except Exception as e:
                self._log(f'HUD ROI load error: {e}')
                
    def _save_hud_roi(self):
        if self.hud_roi:
            json.dump(self.hud_roi, open(self.hud_roi_path, 'w'), indent=1)
            self._log(f'HUD ROI saved: {self.hud_roi}')
            
    def _save_roi(self):
        if self.roi:
            json.dump(self.roi, open(self.roi_path, 'w'), indent=1)
            self._log(f'Board ROI saved: {self.roi}')
            
    # ─── Hotkey ──────────────────────────────────────────────────────
    
    def _register_hotkey(self):
        self._log('Hotkey armed: Ctrl+Shift+X')
        self.hotkey_thread = threading.Thread(target=self._hotkey_loop, daemon=True)
        self.hotkey_thread.start()
        
    def _hotkey_loop(self):
        user32 = ctypes.windll.user32
        while getattr(self, 'running', True) or self.state != 'DONE':
            if (user32.GetAsyncKeyState(VK_CONTROL) & 0x8000 and
                user32.GetAsyncKeyState(VK_SHIFT) & 0x8000 and
                user32.GetAsyncKeyState(VK_X) & 0x8000):
                self.root.after(0, self._kill_switch)
                break
            time.sleep(0.05)
            
    def _kill_switch(self):
        self._log('KILL SWITCH ACTIVATED')
        self.running = False
        if self.auto_player:
            self.auto_player.stop()
        self.root.after(500, self.root.destroy)
        
    # ─── Warmup ──────────────────────────────────────────────────────
    
    def _warmup(self):
        def work():
            try:
                self._log('Warming up detector...')
                with mss.MSS() as sct:
                    shot = np.asarray(sct.grab(sct.monitors[1]))[:, :, :3]
                board = find_board(shot)
                if board is None:
                    self.root.after(0, lambda: self._log('ERROR: Board not found on warmup'))
                    self.root.after(0, lambda: self._set_state('READY'))
                    return
                    
                xs, ys, pitch = board['xs'], board['ys'], board['pitch']
                left = int(min(xs) - pitch * 0.55)
                top = int(min(ys) - pitch * 0.55)
                right = int(max(xs) + pitch * 0.55)
                bottom = int(max(ys) + pitch * 0.55)
                self.roi = {'left': left, 'top': top, 'width': right-left, 'height': bottom-top}
                
                json.dump(self.roi, open(self.roi_path, 'w'), indent=1)
                
                cells = detect_board(shot, self.roi)
                identified = sum(1 for row in cells for c in row if c != '?')
                self.root.after(0, lambda: self._log(f'Warmup OK: {identified}/64 cells identified'))
                self.root.after(0, lambda: self._set_state('READY'))
                self.warmup_done = True
                
            except Exception as e:
                self.root.after(0, lambda: self._log(f'Warmup error: {e}'))
                self.root.after(0, lambda: self._set_state('READY'))
                
        threading.Thread(target=work, daemon=True).start()
        
    # ─── RECOVER ─────────────────────────────────────────────────────
    
    def _on_recover(self):
        if self.state not in ('READY', 'PLAYING'):
            return
        self._log('RECOVER: scanning board...')
        self.recover_btn.configure(state='disabled', text='SCANNING...')
        
        def work():
            try:
                with mss.MSS() as sct:
                    shot = np.asarray(sct.grab(sct.monitors[1]))[:, :, :3]
                board = find_board(shot)
                if board is None:
                    self.root.after(0, lambda: self._log('RECOVER FAILED: board not found'))
                    self.root.after(0, lambda: self._enable_recover())
                    return
                    
                xs, ys, pitch = board['xs'], board['ys'], board['pitch']
                left = int(min(xs) - pitch * 0.55)
                top = int(min(ys) - pitch * 0.55)
                right = int(max(xs) + pitch * 0.55)
                bottom = int(max(ys) + pitch * 0.55)
                self.roi = {'left': left, 'top': top, 'width': right-left, 'height': bottom-top}
                
                json.dump(self.roi, open(self.roi_path, 'w'), indent=1)
                
                cells = detect_board(shot, self.roi)
                identified = sum(1 for row in cells for c in row if c != '?')
                
                self.root.after(0, lambda: self._log(f'RECOVER OK: {identified}/64 identified, ROI locked'))
                self.root.after(0, lambda: self._enable_recover())
                self.root.after(0, lambda: self._update_roi_display())
                
            except Exception as e:
                self.root.after(0, lambda: self._log(f'RECOVER error: {e}'))
                self.root.after(0, lambda: self._enable_recover())
                
        threading.Thread(target=work, daemon=True).start()
        
    def _enable_recover(self):
        self.recover_btn.configure(state='normal', text='RECOVER')
        
    def _update_roi_display(self):
        if self.roi:
            self._log(f'ROI updated: {self.roi["left"]},{self.roi["top"]} {self.roi["width"]}x{self.roi["height"]}')
            
    # ─── PLAY / PAUSE ────────────────────────────────────────────────
    
    def _on_play(self):
        if self.state not in ('READY', 'PAUSED'):
            return
            
        if not self.roi:
            self._log('ERROR: No board ROI - run RECOVER first')
            messagebox.showerror('Error', 'No board ROI - run RECOVER first')
            return
            
        self._log('Starting auto-play...')
        self._set_state('PLAYING')
        
        # Create auto-player
        self.auto_player = IntegratedAutoPlayer(
            roi=self.roi,
            hud_roi=self.hud_roi,
            config=AutoConfig(
                move_delay=0.9,
                cascade_wait=1.0,
                save_debug_frames=True
            ),
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            panel_callback=lambda msg: self.ui_queue.put(('log', msg))
        )
        
        # Register callbacks for UI updates
        self.auto_player.logger.register_ui_callback(
            lambda msg: self.ui_queue.put(('log', msg))
        )
        
        # Start auto-player
        self.auto_player.start()
        self.running = True
        
        # Start UI update loop for auto-player data
        self._start_auto_player_ui_loop()
        
        self.play_btn.configure(text='STOP', command=self._on_stop)
        self.pause_btn.configure(state='normal')
        self.recover_btn.configure(state='normal')
        self.close_btn.configure(state='normal')
        
    def _on_pause(self):
        if self.auto_player:
            if self.auto_player.paused:
                self.auto_player.resume()
                self.pause_btn.configure(text='PAUSE')
                self._log('Auto-play RESUMED')
            else:
                self.auto_player.pause()
                self.pause_btn.configure(text='RESUME')
                self._log('Auto-play PAUSED')
                
    def _on_stop(self):
        self._log('Stopping auto-play...')
        if self.auto_player:
            self.auto_player.stop()
            self.auto_player = None
        self.running = False
        self._set_state('READY')
        self.play_btn.configure(text='PLAY', command=self._on_play)
        self.pause_btn.configure(state='disabled', text='PAUSE')
        self._log('Auto-play stopped')
        
    def _start_auto_player_ui_loop(self):
        """Poll auto-player for UI updates."""
        def poll():
            if not self.running or not self.auto_player:
                return
                
            try:
                ap = self.auto_player
                
                # Update HUD display
                hud = ap.current_hud
                self.ui_queue.put(('hud', {
                    'score': hud.score,
                    'level': hud.level,
                    'mult': hud.multiplier,
                    'timer': hud.timer_seconds,
                    'moves': hud.moves_remaining
                }))
                
                # Update prediction display
                if ap.move_history:
                    last = ap.move_history[-1]
                    mv = last.move
                    pred = last.predicted
                    self.ui_queue.put(('prediction', {
                        'move': f'({mv[0]},{mv[1]})->({mv[2]},{mv[3]})',
                        'points': pred.points,
                        'chain': pred.chain_count,
                        'specials': pred.specials_created,
                        'det': pred.detonations
                    }))
                
                # Update grid
                self.ui_queue.put(('grid', ap.current_grid))
                
            except Exception as e:
                pass
                
            if self.running:
                self.root.after(200, poll)
                
        self.root.after(200, poll)
        
    # ─── UI Updates ──────────────────────────────────────────────────
    
    def _update_hud_display(self, hud: dict):
        self.hud_labels['score'].configure(text=str(hud.get('score', '--')))
        self.hud_labels['level'].configure(text=str(hud.get('level', '--')))
        self.hud_labels['mult'].configure(text=f"x{hud.get('mult', '--')}")
        self.hud_labels['timer'].configure(text=f"{hud.get('timer', '--')}s")
        self.hud_labels['moves'].configure(text=str(hud.get('moves', '--')))
        
    def _update_prediction_display(self, pred: dict):
        self.pred_labels['move'].configure(text=pred.get('move', '--'))
        self.pred_labels['points'].configure(text=str(pred.get('points', '--')))
        self.pred_labels['chain'].configure(text=str(pred.get('chain', '--')))
        self.pred_labels['specials'].configure(text=str(pred.get('specials', '--')))
        self.pred_labels['det'].configure(text=str(pred.get('det', '--')))
        
    def _render_grid(self, grid: List[List[str]]):
        for r in range(8):
            for c in range(8):
                val = grid[r][c] if r < len(grid) and c < len(grid[r]) else '?'
                lbl = self.grid_cells[r][c]
                lbl.configure(text=val.upper() if val != '?' else '?',
                              bg=GEM_COLORS.get(val, '#333333'),
                              fg='#000000' if val in ('w','y') else '#ffffff')
                              
    def _set_state(self, state: str):
        self.state = state
        self.state_label.configure(text=state)
        if state == 'READY':
            self.recover_btn.configure(state='normal')
            self.play_btn.configure(state='normal')
            self.close_btn.configure(state='normal')
        elif state == 'PLAYING':
            self.recover_btn.configure(state='normal')
            self.play_btn.configure(state='normal')
            self.pause_btn.configure(state='normal')
            self.close_btn.configure(state='normal')
        elif state == 'DONE':
            self.recover_btn.configure(state='disabled')
            self.play_btn.configure(state='disabled')
            self.pause_btn.configure(state='disabled')
            self.close_btn.configure(state='normal', text='EXIT')
            
    # ─── CLOSE ───────────────────────────────────────────────────────
    
    def _on_close(self):
        if self.close_stage == 0:
            self.close_stage = 1
            self.close_btn.configure(text='CONFIRM', bg='#ffaa00', activebackground='#ffcc00')
            self._log('Close armed — click again to exit')
            self.root.after(3000, self._rearm_close)
        else:
            self._log('Shutting down...')
            self.running = False
            if self.auto_player:
                self.auto_player.stop()
            self.root.destroy()
            
    def _rearm_close(self):
        if self.close_stage == 1:
            self.close_stage = 0
            self.close_btn.configure(text='CLOSE', bg=WARN_COLOR, activebackground='#ff8b8b')
            
    # ─── Export / Debug ──────────────────────────────────────────────
    
    def _export_data(self):
        """Export all training data."""
        self._log('Exporting training data...')
        try:
            if self.auto_player:
                self.auto_player._export_session_data()
            if self.logger:
                self.logger.export_training_data()
            self._log('Training data exported successfully')
            messagebox.showinfo('Export', 'Training data exported successfully')
        except Exception as e:
            self._log(f'Export failed: {e}')
            messagebox.showerror('Export Error', str(e))
            
    def _save_debug_frame(self):
        """Save current frame for debugging."""
        self._log('Saving debug frame...')
        try:
            with mss.MSS() as sct:
                shot = np.asarray(sct.grab(sct.monitors[1]))[:, :, :3]
            fname = Path(__file__).parent.parent / "data" / "debug" / f"manual_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fname.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(fname), shot)
            self._log(f'Debug frame saved: {fname}')
        except Exception as e:
            self._log(f'Debug frame save failed: {e}')
            
    def run(self):
        self.root.mainloop()


# ─── Main Entry ──────────────────────────────────────────────────────

def main():
    panel = OverlayPanel()
    panel.run()


if __name__ == '__main__':
    main()