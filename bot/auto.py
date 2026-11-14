"""
Auto-play loop (legacy) — integrates detector, planner, clicker.
Runs in background thread, started by panel when entering PLAYING state.

Note: This is the legacy auto-player. Use core.auto_player.IntegratedAutoPlayer
for the new fully integrated version with HUD parsing, verification, etc.
"""

from __future__ import annotations
import time
import threading
import sys
import os

# Add paths
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BOT_DIR)
sys.path.insert(0, ROOT_DIR)

import mss
import numpy as np
from detector import detect_board
from sim import Game
from bot.planner import best_move
from bot.clicker import double_click_cell


class AutoPlayer:
    """Legacy auto-player - use core.auto_player.IntegratedAutoPlayer for new version."""
    
    def __init__(self, roi: dict, panel_callback=None):
        self.roi = roi
        self.panel_callback = panel_callback  # for logging to panel
        self.running = False
        self.thread = None
        self.paused = False
        self.last_move_time = 0
        self.move_delay = 1.0  # minimum seconds between moves
        
    def _log(self, msg: str):
        if self.panel_callback:
            self.panel_callback(msg)
        else:
            print(f"[AUTO] {msg}")
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self._log("Auto-play started")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self._log("Auto-play stopped")
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False
    
    def _loop(self):
        with mss.MSS() as sct:
            monitor = sct.monitors[1]
            while self.running:
                if self.paused:
                    time.sleep(0.2)
                    continue
                
                # Rate limit
                now = time.time()
                if now - self.last_move_time < self.move_delay:
                    time.sleep(0.05)
                    continue
                
                try:
                    # Capture full screen
                    full = np.asarray(sct.grab(monitor))[:, :, :3]
                    
                    # Detect board
                    grid = detect_board(full, self.roi)
                    
                    # Check for unknown cells
                    unknown = sum(1 for row in grid for c in row if c == '?')
                    if unknown > 8:  # too many unknowns, skip
                        self._log(f"Skip: {unknown} unknown cells")
                        time.sleep(0.5)
                        continue
                    
                    # Plan best move
                    # TODO: get actual level/score/time from HUD OCR
                    result = best_move(grid, level=1, score=0, time_left=180.0)
                    
                    if result is None:
                        self._log("No legal moves")
                        time.sleep(1.0)
                        continue
                    
                    mv, res = result
                    r1, c1, r2, c2 = mv
                    
                    self._log(f"Move: ({r1},{c1})->({r2},{c2})  +{res.points} pts  chain={res.chain_count}")
                    
                    # Execute swap (tap-tap)
                    pitch = self.roi['width'] / 8.0
                    double_click_cell(self.roi, pitch, r1, c1, r2, c2)
                    
                    self.last_move_time = time.time()
                    
                    # Wait for board to settle (cascade animation)
                    time.sleep(0.8)
                    
                except Exception as e:
                    self._log(f"Error: {e}")
                    time.sleep(1.0)


# Self-test
if __name__ == '__main__':
    # Test with a mock ROI
    roi = {'left': 788, 'top': 72, 'width': 661, 'height': 661}
    auto = AutoPlayer(roi)
    print("AutoPlayer ready. Call auto.start() to begin.")