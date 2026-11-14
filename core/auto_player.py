"""
Fully Integrated Auto-Player — Detector → HUD → Planner → Clicker → Verify
============================================================================

Complete auto-play loop with:
- Live board detection (MSS + gemvision)
- HUD parsing (score, level, multiplier, timer)
- Move planning (rule engine with cascade simulation)
- Click execution (SendInput tap-tap)
- Post-click verification
- Cascade handling
- Round-end detection
- Self-healing / auto-recovery
- Comprehensive logging & debug data collection
"""

from __future__ import annotations
import time
import threading
import queue
import sys
import os
import traceback
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
import mss

# Add project paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "detector"))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "core"))

import gemvision as gv
from rules import Game, MoveResult, COLOURS
from planner import grid_to_game, evaluate_moves, best_move, format_move
from clicker import double_click_cell
from logging_system import CrystalLogger, get_logger, log_detection, log_move_eval, log_click, log_game_state, log_error
from hud_parser import HUDParser, HUDState

# ─── Data Classes ──────────────────────────────────────────────────────

@dataclass
class AutoConfig:
    """Configuration for auto-player."""
    # Timing
    move_delay: float = 0.9          # min seconds between moves
    cascade_wait: float = 1.0        # wait after click for cascade to settle
    verify_wait: float = 0.3         # wait before verifying click
    max_move_time: float = 5.0       # max time to plan+execute a move
    
    # Detection
    max_unknown_cells: int = 8       # skip frame if more unknowns
    min_board_confidence: float = 0.7
    re_detect_interval: int = 30     # frames between full board re-detect
    
    # Click
    click_delay: float = 0.15        # delay between tap-tap
    click_jitter: float = 1.0        # pixel jitter for human-like clicks
    
    # Recovery
    auto_recover: bool = True        # auto-run RECOVER on detection failure
    max_consecutive_failures: int = 5
    recover_cooldown: float = 3.0    # seconds between recover attempts
    
    # Round end
    end_timer_threshold: int = 3     # seconds left to consider round over
    end_score_stall: int = 10        # frames with no score change = round end
    
    # HUD
    hud_parse_interval: int = 5      # parse HUD every N frames
    
    # Debug
    save_debug_frames: bool = True
    max_debug_frames: int = 500

@dataclass
class MoveRecord:
    """Record of a completed move for analysis."""
    frame_id: int
    timestamp: float
    move: Tuple[int, int, int, int]
    predicted: MoveResult
    actual_score_delta: int
    actual_grid: List[List[str]]
    verification: Dict[str, Any]
    hud_before: HUDState
    hud_after: Optional[HUDState]

# ─── Main Auto-Player ──────────────────────────────────────────────────

class IntegratedAutoPlayer:
    """Fully integrated auto-player with all components connected."""
    
    def __init__(self, 
                 roi: Dict[str, int],
                 hud_roi: Optional[Dict[str, int]] = None,
                 config: Optional[AutoConfig] = None,
                 session_id: Optional[str] = None,
                 panel_callback: Optional[Callable[[str], None]] = None):
        
        self.roi = roi
        self.hud_roi = hud_roi
        self.config = config or AutoConfig()
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.panel_callback = panel_callback
        
        # State
        self.running = False
        self.paused = False
        self.thread: Optional[threading.Thread] = None
        
        # Frame/move counters
        self.frame_id = 0
        self.move_id = 0
        self.consecutive_failures = 0
        self.last_move_time = 0
        self.last_recover_time = 0
        self.last_score = 0
        self.score_stall_frames = 0
        
        # Current state
        self.current_grid: List[List[str]] = [['?' for _ in range(8)] for _ in range(8)]
        self.current_hud: HUDState = HUDState()
        self.cascading = False
        self.round_ended = False
        
        # Move history
        self.move_history: List[MoveRecord] = []
        
        # Components
        self.logger = get_logger("auto_player", self.session_id)
        if panel_callback:
            self.logger.register_ui_callback(panel_callback)
            
        self.hud_parser = HUDParser()
        if self.config.save_debug_frames:
            debug_dir = ROOT / "debug_data" / self.session_id / "hud"
            self.hud_parser.enable_debug(str(debug_dir))
            
        # Clicker state
        self.pitch = roi['width'] / 8.0
        
        # Debug frames
        self.debug_frame_count = 0
        self.debug_dir = ROOT / "debug_data" / self.session_id / "frames"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Last successful board detection
        self.last_good_grid: Optional[List[List[str]]] = None
        self.last_good_roi = roi
        
    # ─── Lifecycle ────────────────────────────────────────────────────
    
    def start(self):
        """Start the auto-play loop."""
        if self.running:
            return
            
        self.logger.info(f"Starting auto-player (session: {self.session_id})")
        self.logger.info(f"Board ROI: {self.roi}")
        self.logger.info(f"HUD ROI: {self.hud_roi}")
        self.logger.info(f"Config: move_delay={self.config.move_delay}s, cascade_wait={self.config.cascade_wait}s")
        
        self.running = True
        self.paused = False
        self.round_ended = False
        self.consecutive_failures = 0
        self.score_stall_frames = 0
        self.last_move_time = 0
        self.last_recover_time = 0
        self.move_id = 0
        self.frame_id = 0
        self.move_history.clear()
        
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the auto-play loop."""
        self.logger.info("Stopping auto-player...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        self._export_session_data()
        self.logger.info("Auto-player stopped")
        
    def pause(self):
        self.paused = True
        self.logger.info("Auto-player PAUSED")
        
    def resume(self):
        self.paused = False
        self.logger.info("Auto-player RESUMED")
        
    def trigger_recover(self):
        """Manually trigger board re-detection."""
        self.logger.info("Manual RECOVER triggered")
        self.consecutive_failures = self.config.max_consecutive_failures  # force recover
        
    # ─── Main Loop ────────────────────────────────────────────────────
    
    def _main_loop(self):
        """Main auto-play loop."""
        with mss.MSS() as sct:
            monitor = sct.monitors[1]
            frame_count = 0
            
            while self.running:
                loop_start = time.perf_counter()
                
                try:
                    if self.paused:
                        time.sleep(0.2)
                        continue
                        
                    if self.round_ended:
                        self.logger.info("Round ended - stopping")
                        break
                        
                    # Rate limiting
                    now = time.time()
                    if now - self.last_move_time < self.config.move_delay:
                        time.sleep(0.02)
                        continue
                        
                    # 1. CAPTURE
                    with self.logger.time_operation("capture"):
                        full = np.asarray(sct.grab(monitor))[:, :, :3]
                        
                    # 2. HUD PARSE (periodic)
                    hud_changed = False
                    if frame_count % self.config.hud_parse_interval == 0 and self.hud_roi:
                        with self.logger.time_operation("hud_parse"):
                            hud_crop = self._crop_hud(full)
                            if hud_crop is not None:
                                new_hud = self.hud_parser.parse_hud(hud_crop)
                                if new_hud.confidence > 0.3:
                                    hud_changed = self._update_hud(new_hud)
                                    
                    # 3. BOARD DETECTION
                    with self.logger.time_operation("board_detect"):
                        detection_result = self._detect_board(full)
                        
                    if detection_result is None:
                        self._handle_detection_failure()
                        frame_count += 1
                        continue
                        
                    grid, board_found, pitch, cell_details = detection_result
                    
                    # Log detection frame
                    if self.config.save_debug_frames and self.debug_frame_count < self.config.max_debug_frames:
                        self._save_debug_frame(full, grid, cell_details)
                        
                    # Log detection
                    log_detection(self.logger,
                        board_found=board_found,
                        roi=self.roi,
                        pitch=pitch,
                        grid=grid,
                        cell_details=cell_details,
                        capture_latency=0,  # measured in context manager
                        detect_latency=0,
                        image=full if self.config.save_debug_frames else None)
                    
                    # Update state
                    self.current_grid = grid
                    self.frame_id += 1
                    frame_count += 1
                    self.consecutive_failures = 0
                    self.last_good_grid = [row[:] for row in grid]
                    
                    # Check for unknown cells
                    unknown = sum(1 for row in grid for c in row if c == '?')
                    if unknown > self.config.max_unknown_cells:
                        self.logger.warning(f"Too many unknown cells: {unknown}, skipping frame")
                        time.sleep(0.1)
                        continue
                        
                    # 4. LOG GAME STATE
                    log_game_state(self.logger,
                        grid=grid,
                        level=self.current_hud.level,
                        score=self.current_hud.score,
                        time_left=self.current_hud.timer_seconds,
                        multiplier=self.current_hud.multiplier,
                        unknown_cells=unknown,
                        specials=[],  # detector doesn't identify specials yet
                        cascading=self.cascading)
                        
                    # 5. CHECK ROUND END
                    if self._check_round_end():
                        self.round_ended = True
                        self.logger.info("Round end detected")
                        break
                        
                    # 6. PLAN MOVE
                    with self.logger.time_operation("plan_move"):
                        plan_result = self._plan_move(grid)
                        
                    if plan_result is None:
                        self.logger.warning("No legal moves - checking for deadlock")
                        self._handle_no_moves()
                        continue
                        
                    mv, predicted = plan_result
                    r1, c1, r2, c2 = mv
                    
                    # Log move evaluation
                    # We need to reconstruct evaluations for logging
                    game = grid_to_game(grid, self.current_hud.level, self.current_hud.score, self.current_hud.timer_seconds)
                    evaluations = evaluate_moves(game)
                    eval_list = []
                    for m, res in evaluations:
                        eval_list.append({
                            'move': m,
                            'points': res.points,
                            'chain': res.chain_count,
                            'specials_created': res.specials_created,
                            'detonations': res.detonations,
                            'level_ups': res.level_ups
                        })
                    
                    chosen = {
                        'move': mv,
                        'points': predicted.points,
                        'chain': predicted.chain_count,
                        'specials_created': predicted.specials_created,
                        'detonations': predicted.detonations,
                        'level_ups': predicted.level_ups
                    }
                    
                    log_move_eval(self.logger,
                        board_grid=grid,
                        level=self.current_hud.level,
                        score=self.current_hud.score,
                        time_left=self.current_hud.timer_seconds,
                        evaluations=eval_list,
                        chosen=chosen,
                        eval_latency=0)  # measured in context manager
                    
                    # 6. EXECUTE CLICK
                    with self.logger.time_operation("execute_click"):
                        click_success = self._execute_click(mv)
                        
                    if not click_success:
                        self.logger.error(f"Click failed for move {format_move(mv)}")
                        continue
                        
                    # 7. VERIFY CLICK
                    time.sleep(self.config.verify_wait)
                    verification = self._verify_click(grid, mv, predicted)
                    
                    # 8. WAIT FOR CASCADE
                    self.cascading = True
                    time.sleep(self.config.cascade_wait)
                    self.cascading = False
                    
                    # 9. RECORD MOVE
                    self._record_move(grid, mv, predicted, verification)
                    
                    self.last_move_time = time.time()
                    self.move_id += 1
                    
                    # Check score stall
                    if self.current_hud.score == self.last_score:
                        self.score_stall_frames += 1
                    else:
                        self.score_stall_frames = 0
                        self.last_score = self.current_hud.score
                        
                except Exception as e:
                    log_error(self.logger, f"Main loop error: {e}", e, 
                             context={'frame_id': self.frame_id, 'move_id': self.move_id})
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.config.max_consecutive_failures:
                        self._trigger_recovery()
                    time.sleep(1.0)
                    
                # Frame rate limiting
                elapsed = time.perf_counter() - loop_start
                if elapsed < 0.05:  # max ~20 fps
                    time.sleep(0.05 - elapsed)
                    
    # ─── Board Detection ──────────────────────────────────────────────
    
    def _detect_board(self, full: np.ndarray) -> Optional[Tuple[List[List[str]], bool, float, List[Dict]]]:
        """Detect board and return grid, board_found, pitch, cell_details."""
        try:
            # Use gemvision's detect_board with our ROI
            grid = gv.detect_board(full, self.roi)
            
            # Also run find_board to get pitch and validate
            board = gv.find_board(full)
            if board is None:
                return None
                
            pitch = board['pitch']
            centres = board['centres']
            
            # Collect cell details for debugging
            cell_details = []
            for idx, (cx, cy) in enumerate(centres):
                r, c = divmod(idx, 8)
                core, _ = gv.cell_core(full, cx, cy, pitch)
                blob = gv.cell_blob(full, cx, cy, pitch)
                sil = gv.cell_silhouette(full, cx, cy, pitch)
                flags = gv.detect_variants(full, cx, cy, pitch, blob["bbox"] if blob else None)
                
                cell_details.append({
                    'r': r, 'c': c,
                    'cx': cx, 'cy': cy,
                    'detected': grid[r][c],
                    'core_shape': core.shape if core is not None else None,
                    'blob_extent': blob['extent'] if blob else None,
                    'sil_extent': sil['extent'] if sil else None,
                    'flags': list(flags) if flags else []
                })
                
            return grid, True, pitch, cell_details
            
        except Exception as e:
            log_error(self.logger, f"Board detection failed: {e}", e)
            return None
            
    def _handle_detection_failure(self):
        """Handle board detection failure."""
        self.consecutive_failures += 1
        self.logger.warning(f"Detection failure #{self.consecutive_failures}")
        
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self._trigger_recovery()
            
    def _trigger_recovery(self):
        """Trigger automatic board recovery."""
        now = time.time()
        if now - self.last_recover_time < self.config.recover_cooldown:
            return
            
        if not self.config.auto_recover:
            return
            
        self.logger.info("Triggering auto-recovery (RECOVER)")
        self.last_recover_time = now
        self.consecutive_failures = 0
        
        # Re-detect board from full screen
        with mss.MSS() as sct:
            full = np.asarray(sct.grab(sct.monitors[1]))[:, :, :3]
            board = gv.find_board(full)
            if board:
                xs, ys, pitch = board['xs'], board['ys'], board['pitch']
                left = int(min(xs) - pitch * 0.55)
                top = int(min(ys) - pitch * 0.55)
                right = int(max(xs) + pitch * 0.55)
                bottom = int(max(ys) + pitch * 0.55)
                self.roi = {'left': left, 'top': top, 'width': right-left, 'height': bottom-top}
                self.pitch = self.roi['width'] / 8.0
                
                # Save ROI
                import json
                json.dump(self.roi, open(ROOT / "detector" / "roi.json", 'w'), indent=1)
                
                self.logger.info(f"RECOVERY: new ROI {self.roi}")
            else:
                self.logger.error("RECOVERY FAILED: could not find board")
                
    # ─── HUD Handling ────────────────────────────────────────────────
    
    def _crop_hud(self, full: np.ndarray) -> Optional[np.ndarray]:
        """Crop HUD region from full frame."""
        if not self.hud_roi:
            return None
            
        h, w = full.shape[:2]
        x = max(0, self.hud_roi['left'])
        y = max(0, self.hud_roi['top'])
        w = min(self.hud_roi['width'], full.shape[1] - x)
        h = min(self.hud_roi['height'], full.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return None
            
        return full[y:y+h, x:x+w].copy()
        
    def _update_hud(self, new_hud: HUDState) -> bool:
        """Update HUD state and return True if changed significantly."""
        old = self.current_hud
        self.current_hud = new_hud
        
        # Check for significant changes
        changed = False
        if abs(new_hud.score - old.score) > 0:
            changed = True
            self.logger.info(f"HUD: score {old.score} -> {new_hud.score}")
        if new_hud.level != old.level:
            changed = True
            self.logger.info(f"HUD: level {old.level} -> {new_hud.level} (mult x{new_hud.multiplier})")
        if new_hud.timer_seconds != old.timer_seconds:
            changed = True
            self.logger.debug(f"HUD: timer {old.timer_seconds} -> {new_hud.timer_seconds}")
            
        return changed
        
    # ─── Move Planning ────────────────────────────────────────────────
    
    def _plan_move(self, grid: List[List[str]]) -> Optional[Tuple[Tuple[int,int,int,int], MoveResult]]:
        """Plan best move using current HUD state."""
        try:
            result = best_move(
                grid,
                level=self.current_hud.level,
                score=self.current_hud.score,
                time_left=self.current_hud.timer_seconds
            )
            return result
        except Exception as e:
            log_error(self.logger, f"Move planning failed: {e}", e)
            return None
            
    # ─── Click Execution ──────────────────────────────────────────────
    
    def _execute_click(self, mv: Tuple[int,int,int,int]) -> bool:
        """Execute tap-tap click for the move."""
        r1, c1, r2, c2 = mv
        
        # Calculate screen coordinates
        left, top = self.roi['left'], self.roi['top']
        x1 = int(left + c1 * self.pitch + self.pitch * 0.5)
        y1 = int(top + r1 * self.pitch + self.pitch * 0.5)
        x2 = int(left + c2 * self.pitch + self.pitch * 0.5)
        y2 = int(top + r2 * self.pitch + self.pitch * 0.5)
        
        # Add small jitter for human-like behavior
        import random
        jitter = self.config.click_jitter
        x1 += random.randint(-jitter, jitter)
        y1 += random.randint(-jitter, jitter)
        x2 += random.randint(-jitter, jitter)
        y2 += random.randint(-jitter, jitter)
        
        coords = ((x1, y1), (x2, y2))
        click_start = time.perf_counter()
        
        try:
            from clicker import click_at
            
            # First click
            click_at(x1, y1)
            time.sleep(self.config.click_delay)
            
            # Second click
            click_at(x2, y2)
            
            latency = time.perf_counter() - click_start
            
            log_click(self.logger,
                move=mv,
                coords=((x1, y1), (x2, y2)),
                latency=latency,
                success=True)
                
            return True
            
        except Exception as e:
            latency = time.perf_counter() - click_start
            log_click(self.logger,
                move=mv,
                coords=((x1, y1), (x2, y2)),
                latency=latency,
                success=False,
                error=str(e))
            return False
            
    # ─── Click Verification ───────────────────────────────────────────
    
    def _verify_click(self, 
                      old_grid: List[List[str]], 
                      mv: Tuple[int,int,int,int],
                      predicted: MoveResult) -> Dict[str, Any]:
        """Verify that click produced expected board change."""
        verification = {
            'board_changed': False,
            'score_delta': 0,
            'predicted_points': predicted.points,
            'actual_points': 0,
            'match': False,
            'errors': []
        }
        
        try:
            # Re-detect board after click
            with mss.MSS() as sct:
                full = np.asarray(sct.grab(sct.monitors[1]))[:, :, :3]
            new_grid = gv.detect_board(full, self.roi)
            
            # Check if board changed
            changed = False
            for r in range(8):
                for c in range(8):
                    if old_grid[r][c] != new_grid[r][c]:
                        changed = True
                        break
                        
            verification['board_changed'] = changed
            verification['new_grid'] = new_grid
            
            # Calculate actual score delta from HUD
            actual_delta = self.current_hud.score - self.last_score
            verification['actual_points'] = actual_delta
            verification['predicted_points'] = predicted.points
            
            # Check if prediction matches
            if abs(actual_delta - predicted.points) <= 10:  # small tolerance
                verification['match'] = True
            else:
                verification['errors'].append(
                    f"Score mismatch: predicted {predicted.points}, got {actual_delta}"
                )
                
            self.logger.info(f"Verification: board_changed={changed}, "
                           f"predicted={predicted.points}, actual={actual_delta}, "
                           f"match={verification['match']}")
                           
        except Exception as e:
            verification['errors'].append(f"Verification error: {e}")
            log_error(self.logger, f"Verification failed: {e}", e)
            
        return verification
        
    # ─── Move Recording ───────────────────────────────────────────────
    
    def _record_move(self,
                     grid: List[List[str]],
                     mv: Tuple[int,int,int,int],
                     predicted: MoveResult,
                     verification: Dict[str, Any]):
        """Record completed move for analysis."""
        record = MoveRecord(
            frame_id=self.frame_id,
            timestamp=time.time(),
            move=mv,
            predicted=predicted,
            actual_score_delta=verification.get('actual_points', 0),
            actual_grid=verification.get('new_grid', grid),
            verification=verification,
            hud_before=self.current_hud,
            hud_after=None  # will be filled on next HUD parse
        )
        self.move_history.append(record)
        self.move_id += 1
        
        # Update last good grid
        self.last_good_grid = [row[:] for row in grid]
        
    # ─── Round End Detection ──────────────────────────────────────────
    
    def _check_round_end(self) -> bool:
        """Check if round has ended."""
        # Timer near zero
        if self.current_hud.timer_seconds <= self.config.end_timer_threshold:
            self.logger.info(f"Round end: timer at {self.current_hud.timer_seconds}s")
            return True
            
        # Score stalled for many frames
        if self.score_stall_frames >= self.config.end_score_stall:
            self.logger.info(f"Round end: score stalled for {self.score_stall_frames} frames")
            return True
            
        # No legal moves and not cascading
        if not self.cascading:
            game = grid_to_game(self.current_grid, self.current_hud.level, 
                               self.current_hud.score, self.current_hud.timer_seconds)
            if not game.has_legal_move():
                self.logger.info("Round end: no legal moves")
                return True
                
        return False
        
    def _handle_no_moves(self):
        """Handle situation with no legal moves."""
        self.logger.warning("No legal moves - waiting for board refill")
        time.sleep(2.0)  # wait for auto-refill
        
    # ─── Debug Frame Saving ───────────────────────────────────────────
    
    def _save_debug_frame(self, full: np.ndarray, grid: List[List[str]], cell_details: List[Dict]):
        """Save debug frame with overlay."""
        if self.debug_frame_count >= self.config.max_debug_frames:
            return
            
        try:
            # Create annotated frame
            annotated = full.copy()
            
            # Draw board ROI
            cv2.rectangle(annotated, 
                (self.roi['left'], self.roi['top']),
                (self.roi['left'] + self.roi['width'], self.roi['top'] + self.roi['height']),
                (0, 255, 0), 2)
                
            # Draw HUD ROI
            if self.hud_roi:
                cv2.rectangle(annotated,
                    (self.hud_roi['left'], self.hud_roi['top']),
                    (self.hud_roi['left'] + self.hud_roi['width'], self.hud_roi['top'] + self.hud_roi['height']),
                    (255, 0, 255), 2)
                    
            # Draw grid overlay on board region
            board_img = annotated[self.roi['top']:self.roi['top']+self.roi['height'],
                                 self.roi['left']:self.roi['left']+self.roi['width']]
            
            if board_img.size > 0:
                cell_w = board_img.shape[1] / 8
                cell_h = board_img.shape[0] / 8
                for r in range(8):
                    for c in range(8):
                        x = int(c * cell_w)
                        y = int(r * cell_h)
                        val = grid[r][c]
                        color = (0, 255, 0) if val != '?' else (0, 0, 255)
                        cv2.putText(board_img, val[0].upper() if val != '?' else '?',
                                  (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                  
            # Save
            self.debug_frame_count += 1
            fname = self.debug_dir / f"frame_{self.frame_id:06d}_move{self.move_id:04d}.png"
            cv2.imwrite(str(fname), annotated)
            
        except Exception as e:
            self.logger.debug(f"Debug frame save failed: {e}")
            
    # ─── Session Export ───────────────────────────────────────────────
    
    def _export_session_data(self):
        """Export all session data for training/analysis."""
        self.logger.info("Exporting session data...")
        
        # Export logger data
        self.logger.export_training_data()
        
        # Export move history
        move_file = ROOT / "training_data" / self.session_id / "move_history.json"
        move_file.parent.mkdir(parents=True, exist_ok=True)
        
        moves_data = []
        for m in self.move_history:
            moves_data.append({
                'frame_id': m.frame_id,
                'timestamp': m.timestamp,
                'move': m.move,
                'predicted_points': m.predicted.points,
                'actual_points': m.actual_score_delta,
                'verification': m.verification,
                'hud_before_score': m.hud_before.score,
                'hud_before_level': m.hud_before.level
            })
            
        with open(move_file, 'w') as f:
            json.dump(moves_data, f, indent=2)
            
        self.logger.info(f"Exported {len(self.move_history)} moves to {move_file}")
        
        # Create summary
        self._create_session_summary()
        
    def _create_session_summary(self):
        """Create human-readable session summary."""
        summary = {
            'session_id': self.session_id,
            'start_time': self.move_history[0].timestamp if self.move_history else time.time(),
            'end_time': time.time(),
            'total_moves': self.move_id,
            'total_frames': self.frame_id,
            'final_score': self.current_hud.score,
            'final_level': self.current_hud.level,
            'moves_per_minute': self.move_id / max((time.time() - self.move_history[0].timestamp) / 60, 1) if self.move_history else 0,
            'avg_predicted_score': sum(m.predicted.points for m in self.move_history) / max(len(self.move_history), 1),
            'avg_actual_score': sum(m.actual_score_delta for m in self.move_history) / max(len(self.move_history), 1),
            'verification_match_rate': sum(1 for m in self.move_history if m.verification.get('match', False)) / max(len(self.move_history), 1),
            'errors': [e['message'] for e in self.logger.errors]
        }
        
        summary_file = ROOT / "training_data" / self.session_id / "session_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Session summary: {summary}")

# ─── Convenience Functions ────────────────────────────────────────────

def create_auto_player(roi: Dict, hud_roi: Dict = None, **kwargs) -> IntegratedAutoPlayer:
    """Factory function to create auto-player with default config."""
    config = AutoConfig(**kwargs)
    return IntegratedAutoPlayer(roi, hud_roi, config)


# ─── Self-Test ────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Test with mock ROI
    roi = {'left': 788, 'top': 72, 'width': 661, 'height': 661}
    hud_roi = {'left': 788, 'top': 14, 'width': 430, 'height': 46}
    
    config = AutoConfig(
        move_delay=1.0,
        cascade_wait=1.0,
        save_debug_frames=True
    )
    
    player = IntegratedAutoPlayer(roi, hud_roi, config, session_id="test_session")
    print("AutoPlayer created successfully")
    print(f"Session ID: {player.session_id}")
    print(f"Board ROI: {roi}")
    print(f"HUD ROI: {hud_roi}")
    print(f"Config: {config}")