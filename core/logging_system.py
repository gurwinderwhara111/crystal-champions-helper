"""
Unified Logging & Debugging System for Crystal Champions Bot
=============================================================

Provides structured logging, debug data collection, and training data export
for all components: detector, planner, clicker, auto-player, panel.
"""

from __future__ import annotations
import os
import sys
import json
import time
import threading
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from contextlib import contextmanager
import queue

# ─── Configuration ──────────────────────────────────────────────────────
LOG_ROOT = Path(__file__).parent.parent / "logs"
DEBUG_ROOT = Path(__file__).parent.parent / "debug_data"
TRAINING_ROOT = Path(__file__).parent.parent / "training_data"

for p in (LOG_ROOT, DEBUG_ROOT, TRAINING_ROOT):
    p.mkdir(parents=True, exist_ok=True)

# ─── Data Classes ───────────────────────────────────────────────────────

@dataclass
class DetectionFrame:
    """Single frame detection result for training/debug."""
    timestamp: float
    frame_id: int
    board_found: bool
    roi: Optional[Dict[str, int]]
    pitch: Optional[float]
    grid: List[List[str]]
    unknown_count: int
    cell_details: List[Dict[str, Any]]  # per-cell debug info
    capture_latency_ms: float
    detect_latency_ms: float
    image_path: Optional[str] = None  # saved debug image

@dataclass
class MoveEvaluation:
    """Planner move evaluation for training."""
    timestamp: float
    frame_id: int
    board_grid: List[List[str]]
    level: int
    score: int
    time_left: float
    legal_moves: int
    evaluations: List[Dict[str, Any]]  # all moves with scores
    chosen_move: Optional[Dict[str, Any]]
    evaluation_latency_ms: float

@dataclass
class ClickEvent:
    """Click execution record."""
    timestamp: float
    frame_id: int
    move: Tuple[int, int, int, int]
    target_coords: Tuple[Tuple[int, int], Tuple[int, int]]  # (x1,y1), (x2,y2)
    click_latency_ms: float
    success: bool
    error: Optional[str] = None

@dataclass
class GameStateSnapshot:
    """Complete game state at a point in time."""
    timestamp: float
    frame_id: int
    grid: List[List[str]]
    level: int
    score: int
    time_left: float
    multiplier: float
    unknown_cells: int
    specials_on_board: List[Dict[str, Any]]
    cascading: bool

@dataclass
class SessionSummary:
    """End-of-session summary for analysis."""
    session_id: str
    start_time: float
    end_time: float
    total_frames: int
    total_moves: int
    final_score: int
    final_level: int
    moves_per_minute: float
    avg_move_score: float
    cascade_stats: Dict[str, int]
    detection_accuracy: Dict[str, float]
    errors: List[str]

# ─── Core Logger ────────────────────────────────────────────────────────

class CrystalLogger:
    """Centralized logging with structured output and debug collection."""
    
    def __init__(self, component: str, session_id: Optional[str] = None):
        self.component = component
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_counter = 0
        self.move_counter = 0
        
        # Structured data queues
        self.detection_frames: List[DetectionFrame] = []
        self.move_evaluations: List[MoveEvaluation] = []
        self.click_events: List[ClickEvent] = []
        self.game_snapshots: List[GameStateSnapshot] = []
        self.errors: List[Dict[str, Any]] = []
        
        # Setup file logging
        self._setup_file_logging()
        
        # Debug image saving
        self.save_debug_images = True
        self.debug_image_dir = DEBUG_ROOT / self.session_id / component
        self.debug_image_dir.mkdir(parents=True, exist_ok=True)
        
        # Callbacks for real-time UI updates
        self.ui_callbacks: List[Callable[[str], None]] = []
        
    def _setup_file_logging(self):
        """Setup rotating file handlers for different log levels."""
        log_file = LOG_ROOT / f"{self.session_id}_{self.component}.log"
        debug_file = LOG_ROOT / f"{self.session_id}_{self.component}_DEBUG.log"
        
        self.logger = logging.getLogger(f"crystal.{self.component}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # Main log - INFO and above
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        self.logger.addHandler(fh)
        
        # Debug log - everything
        dh = logging.FileHandler(debug_file, encoding='utf-8')
        dh.setLevel(logging.DEBUG)
        dh.setFormatter(logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%H:%M:%S'
        ))
        self.logger.addHandler(dh)
        
        # Console - WARNING and above only
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        self.logger.addHandler(ch)
        
    def register_ui_callback(self, callback: Callable[[str], None]):
        """Register a callback for real-time UI log updates."""
        self.ui_callbacks.append(callback)
        
    def _notify_ui(self, level: str, msg: str):
        """Send log to registered UI callbacks."""
        for cb in self.ui_callbacks:
            try:
                cb(f"[{level}] {msg}")
            except Exception:
                pass
                
    # ─── Standard Logging ─────────────────────────────────────────────
    
    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, extra=kwargs)
        self._notify_ui('DEBUG', msg)
        
    def info(self, msg: str, **kwargs):
        self.logger.info(msg, extra=kwargs)
        self._notify_ui('INFO', msg)
        
    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, extra=kwargs)
        self._notify_ui('WARN', msg)
        
    def error(self, msg: str, exc: Optional[Exception] = None, **kwargs):
        if exc:
            self.logger.error(f"{msg}: {exc}", exc_info=True, extra=kwargs)
            self.errors.append({
                'timestamp': time.time(),
                'message': msg,
                'exception': str(exc),
                'traceback': traceback.format_exc()
            })
        else:
            self.logger.error(msg, extra=kwargs)
            self.errors.append({
                'timestamp': time.time(),
                'message': msg,
                'exception': None,
                'traceback': None
            })
        self._notify_ui('ERROR', msg)
        
    def critical(self, msg: str, exc: Optional[Exception] = None, **kwargs):
        self.logger.critical(msg, exc_info=exc is not None, extra=kwargs)
        self._notify_ui('CRITICAL', msg)
        
    # ─── Structured Data Collection ───────────────────────────────────
    
    def log_detection_frame(self, 
                           board_found: bool,
                           roi: Optional[Dict],
                           pitch: Optional[float],
                           grid: List[List[str]],
                           cell_details: List[Dict],
                           capture_latency: float,
                           detect_latency: float,
                           image: Optional[Any] = None):
        """Log a complete detection frame for training/debug."""
        self.frame_counter += 1
        unknown = sum(1 for row in grid for c in row if c == '?')
        
        frame = DetectionFrame(
            timestamp=time.time(),
            frame_id=self.frame_counter,
            board_found=board_found,
            roi=roi,
            pitch=pitch,
            grid=[row[:] for row in grid],
            unknown_count=unknown,
            cell_details=cell_details,
            capture_latency_ms=capture_latency * 1000,
            detect_latency_ms=detect_latency * 1000
        )
        self.detection_frames.append(frame)
        
        # Save debug image if provided
        if self.save_debug_images and image is not None:
            img_path = self.debug_image_dir / f"frame_{self.frame_counter:06d}.png"
            try:
                import cv2
                cv2.imwrite(str(img_path), image)
                frame.image_path = str(img_path)
            except Exception as e:
                self.debug(f"Failed to save debug image: {e}")
                
        self.info(f"Frame {self.frame_counter}: board={board_found}, unknown={unknown}, "
                  f"pitch={pitch:.1f if pitch else 'N/A'}, "
                  f"capture={capture_latency*1000:.1f}ms, detect={detect_latency*1000:.1f}ms")
        
        return frame
    
    def log_move_evaluation(self,
                           board_grid: List[List[str]],
                           level: int,
                           score: int,
                           time_left: float,
                           evaluations: List[Dict],
                           chosen: Optional[Dict],
                           eval_latency: float):
        """Log planner move evaluation for training."""
        self.move_counter += 1
        self.frame_counter += 1
        
        eval_record = MoveEvaluation(
            timestamp=time.time(),
            frame_id=self.frame_counter,
            board_grid=[row[:] for row in board_grid],
            level=level,
            score=score,
            time_left=time_left,
            legal_moves=len(evaluations),
            evaluations=evaluations,
            chosen_move=chosen,
            evaluation_latency_ms=eval_latency * 1000
        )
        self.move_evaluations.append(eval_record)
        
        if chosen:
            mv = chosen['move']
            self.info(f"Move {self.move_counter}: ({mv[0]},{mv[1]})->({mv[2]},{mv[3]}) "
                      f"+{chosen['points']} pts chain={chosen['chain']} "
                      f"specials={chosen.get('specials_created',0)} "
                      f"eval={eval_latency*1000:.1f}ms")
        else:
            self.warning(f"Move {self.move_counter}: NO LEGAL MOVES FOUND")
            
        return eval_record
    
    def log_click(self,
                  move: Tuple[int, int, int, int],
                  coords: Tuple[Tuple[int, int], Tuple[int, int]],
                  latency: float,
                  success: bool,
                  error: Optional[str] = None):
        """Log click execution."""
        click = ClickEvent(
            timestamp=time.time(),
            frame_id=self.frame_counter,
            move=move,
            target_coords=coords,
            click_latency_ms=latency * 1000,
            success=success,
            error=error
        )
        self.click_events.append(click)
        
        if success:
            self.info(f"Click: ({move[0]},{move[1]})->({move[2]},{move[3]}) "
                      f"at {coords} latency={latency*1000:.1f}ms")
        else:
            self.error(f"Click FAILED: {error}")
            
    def log_game_state(self,
                       grid: List[List[str]],
                       level: int,
                       score: int,
                       time_left: float,
                       multiplier: float,
                       unknown_cells: int,
                       specials: List[Dict],
                       cascading: bool):
        """Log complete game state snapshot."""
        snap = GameStateSnapshot(
            timestamp=time.time(),
            frame_id=self.frame_counter,
            grid=[row[:] for row in grid],
            level=level,
            score=score,
            time_left=time_left,
            multiplier=multiplier,
            unknown_cells=unknown_cells,
            specials_on_board=specials,
            cascading=cascading
        )
        self.game_snapshots.append(snap)
        
    def log_error(self, msg: str, exc: Optional[Exception] = None, context: Optional[Dict] = None):
        """Log error with full context."""
        self.error(msg, exc)
        if context:
            self.errors[-1]['context'] = context
            
    # ─── Export & Persistence ─────────────────────────────────────────
    
    def export_training_data(self, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """Export all collected data for training/analysis."""
        if output_dir is None:
            output_dir = TRAINING_ROOT / self.session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Detection frames
        det_file = output_dir / f"detection_frames_{self.component}.json"
        with open(det_file, 'w') as f:
            json.dump([asdict(f) for f in self.detection_frames], f, indent=2)
        files['detection'] = str(det_file)
        
        # Move evaluations
        eval_file = output_dir / f"move_evaluations_{self.component}.json"
        with open(eval_file, 'w') as f:
            json.dump([asdict(m) for m in self.move_evaluations], f, indent=2)
        files['evaluations'] = str(eval_file)
        
        # Click events
        click_file = output_dir / f"click_events_{self.component}.json"
        with open(click_file, 'w') as f:
            json.dump([asdict(c) for c in self.click_events], f, indent=2)
        files['clicks'] = str(click_file)
        
        # Game snapshots
        snap_file = output_dir / f"game_snapshots_{self.component}.json"
        with open(snap_file, 'w') as f:
            json.dump([asdict(s) for s in self.game_snapshots], f, indent=2)
        files['snapshots'] = str(snap_file)
        
        # Errors
        err_file = output_dir / f"errors_{self.component}.json"
        with open(err_file, 'w') as f:
            json.dump(self.errors, f, indent=2)
        files['errors'] = str(err_file)
        
        # Session summary
        summary = self._create_session_summary()
        sum_file = output_dir / f"session_summary_{self.component}.json"
        with open(sum_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        files['summary'] = str(sum_file)
        
        self.info(f"Exported training data to {output_dir}: {files}")
        return files
    
    def _create_session_summary(self) -> SessionSummary:
        """Create end-of-session summary."""
        total_moves = len(self.move_evaluations)
        total_frames = len(self.detection_frames)
        
        cascade_stats = {}
        total_score = 0
        for eval in self.move_evaluations:
            if eval.chosen_move:
                total_score += eval.chosen_move.get('points', 0)
                chain = eval.chosen_move.get('chain', 0)
                cascade_stats[f'chain_{chain}'] = cascade_stats.get(f'chain_{chain}', 0) + 1
                
        detection_board_found = sum(1 for f in self.detection_frames if f.board_found)
        detection_accuracy = {
            'frames_with_board': detection_board_found,
            'total_frames': total_frames,
            'board_find_rate': detection_board_found / max(total_frames, 1),
            'avg_unknown_cells': sum(f.unknown_count for f in self.detection_frames) / max(total_frames, 1)
        }
        
        return SessionSummary(
            session_id=self.session_id,
            start_time=self.detection_frames[0].timestamp if self.detection_frames else time.time(),
            end_time=time.time(),
            total_frames=total_frames,
            total_moves=total_moves,
            final_score=total_score,
            final_level=0,  # would need to track
            moves_per_minute=total_moves / max((time.time() - self.detection_frames[0].timestamp) / 60, 1) if self.detection_frames else 0,
            avg_move_score=total_score / max(total_moves, 1) if total_moves else 0,
            cascade_stats=cascade_stats,
            detection_accuracy=detection_accuracy,
            errors=[e['message'] for e in self.errors]
        )
    
    @contextmanager
    def time_operation(self, operation: str):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.debug(f"{operation} took {elapsed:.2f}ms")

# ─── Global Logger Factory ────────────────────────────────────────────

_loggers: Dict[str, CrystalLogger] = {}

def get_logger(component: str, session_id: Optional[str] = None) -> CrystalLogger:
    """Get or create a logger for a component."""
    key = f"{component}_{session_id or 'default'}"
    if key not in _loggers:
        _loggers[key] = CrystalLogger(component, session_id)
    return _loggers[key]

def export_all_training_data(session_id: Optional[str] = None):
    """Export training data from all loggers."""
    for logger in _loggers.values():
        if session_id is None or logger.session_id == session_id:
            logger.export_training_data()

def set_global_session(session_id: str):
    """Set session ID for all future loggers."""
    global _loggers
    _loggers.clear()
    # Update existing loggers
    for logger in _loggers.values():
        logger.session_id = session_id

# ─── Convenience Functions ────────────────────────────────────────────

def log_detection(logger: CrystalLogger, **kwargs):
    return logger.log_detection_frame(**kwargs)

def log_move_eval(logger: CrystalLogger, **kwargs):
    return logger.log_move_evaluation(**kwargs)

def log_click(logger: CrystalLogger, **kwargs):
    return logger.log_click(**kwargs)

def log_game_state(logger: CrystalLogger, **kwargs):
    return logger.log_game_state(**kwargs)

def log_error(logger: CrystalLogger, msg: str, exc: Exception = None, **kwargs):
    return logger.log_error(msg, exc, kwargs)