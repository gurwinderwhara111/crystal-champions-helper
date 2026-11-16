"""
Test Pattern Detection on Recorded Videos
==========================================
Analyzes gameplay videos to find special gem opportunities.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from bot.pattern_detector import PatternDetector, PatternType


def analyze_video_with_patterns(video_path: str) -> Dict[str, Any]:
    """Analyze video and detect special gem patterns."""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {Path(video_path).name}")
    print(f"{'='*70}")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Failed to open video"}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Duration: {duration:.1f}s | Frames: {frame_count} | FPS: {fps:.1f}")
    
    # Load grid config
    grid_path = Path(__file__).parent / "config" / "grid.json"
    if not grid_path.exists():
        print("[ERROR] No grid config found. Run SCAN first.")
        return {"error": "No grid config"}
    
    grid = json.load(open(grid_path))
    print("[OK] Grid config loaded")
    
    # Load classifier
    try:
        import pickle
        clf_path = Path(__file__).parent / "config" / "classifier.pkl"
        with open(clf_path, 'rb') as f:
            clf = pickle.load(f)
        print("[OK] Classifier loaded")
    except Exception as e:
        print(f"[ERROR] Classifier: {e}")
        return {"error": str(e)}
    
    # Pattern detector
    detector = PatternDetector()
    
    # Stats
    stats = {
        "total_frames": 0,
        "frames_with_patterns": 0,
        "void_found": 0,
        "starlight_found": 0,
        "fire_found": 0,
        "normal_moves": 0,
        "pattern_examples": []
    }
    
    # Sample every 0.8 seconds
    sample_interval = 0.8
    current_time = 0.0
    
    from detector.grid import capture_region, read_grid
    
    while current_time < duration:
        frame_idx = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        stats["total_frames"] += 1
        
        try:
            # Get board region
            region = capture_region(grid)
            crop = frame[region["top"]:region["top"] + region["height"],
                       region["left"]:region["left"] + region["width"]].copy()
            
            # Read grid
            rows, det_stats = read_grid(crop, grid, clf)
            
            # Check for unknowns
            unknown = det_stats.get("unknown", 0)
            if unknown > 10:
                current_time += sample_interval
                continue
            
            # Convert to short codes for pattern detection
            # Map: green->g, orange->o, blue->b, white->w, red->r, purple->p, yellow->y, ?->?
            color_map = {
                'green': 'g', 'orange': 'o', 'blue': 'b', 'white': 'w',
                'red': 'r', 'purple': 'p', 'yellow': 'y'
            }
            
            short_grid = []
            for row in rows:
                short_row = []
                for cell in row:
                    if cell in color_map:
                        short_row.append(color_map[cell])
                    elif cell == '?':
                        short_row.append('?')
                    else:
                        short_row.append('?')
                short_grid.append(short_row)
            
            # Find patterns
            opportunities = detector.find_all_opportunities(short_grid)
            
            if opportunities:
                stats["frames_with_patterns"] += 1
                
                # Count by type
                for opp in opportunities:
                    if opp.pattern_type == PatternType.VOID:
                        stats["void_found"] += 1
                    elif opp.pattern_type in [PatternType.STARLIGHT_T, PatternType.STARLIGHT_L]:
                        stats["starlight_found"] += 1
                    elif opp.pattern_type == PatternType.FIRE:
                        stats["fire_found"] += 1
                
                # Save first few examples
                if len(stats["pattern_examples"]) < 5:
                    stats["pattern_examples"].append({
                        "time": current_time,
                        "patterns": [str(opp) for opp in opportunities[:3]]
                    })
            else:
                stats["normal_moves"] += 1
                
        except Exception as e:
            pass
        
        current_time += sample_interval
    
    cap.release()
    
    # Print results
    print(f"\n--- RESULTS ---")
    print(f"Frames analyzed: {stats['total_frames']}")
    print(f"Frames with patterns: {stats['frames_with_patterns']}")
    print(f"  - Void (5-in-row): {stats['void_found']}")
    print(f"  - Starlight (T/L): {stats['starlight_found']}")
    print(f"  - Fire (4-in-row): {stats['fire_found']}")
    print(f"Normal moves only: {stats['normal_moves']}")
    
    if stats["pattern_examples"]:
        print(f"\n--- PATTERN EXAMPLES ---")
        for ex in stats["pattern_examples"]:
            print(f"  At {ex['time']:.1f}s: {', '.join(ex['patterns'])}")
    
    return stats


def main():
    videos = [
        r"d:\random works by gs\Crystal Champions helper\2nd gameplay video close capture.mp4",
        r"d:\random works by gs\Crystal Champions helper\Recording 2026-11-15 110245.mp4",
    ]
    
    all_results = []
    
    for video_path in videos:
        if Path(video_path).exists():
            result = analyze_video_with_patterns(video_path)
            result["video"] = Path(video_path).name
            all_results.append(result)
        else:
            print(f"\n[ERROR] Video not found: {video_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    
    for result in all_results:
        if "error" in result:
            print(f"\n{result['video']}: ERROR - {result['error']}")
        else:
            print(f"\n{result['video']}:")
            print(f"  Frames: {result['total_frames']}")
            print(f"  Pattern frames: {result['frames_with_patterns']}")
            print(f"  Void opportunities: {result['void_found']}")
            print(f"  Starlight opportunities: {result['starlight_found']}")
            print(f"  Fire opportunities: {result['fire_found']}")


if __name__ == "__main__":
    main()
