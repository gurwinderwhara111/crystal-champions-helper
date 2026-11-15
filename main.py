#!/usr/bin/env python3
"""
Crystal Champions Bot - Main Entry Point
=========================================

Usage:
    python main.py manual         # Manual mode (default) - arrows + snackbar
    python main.py panel          # Launch control panel (legacy)
    python main.py collect        # Run data collector
    python main.py detect         # Test detection on reference images
    python main.py test           # Run self-tests
    python main.py auto           # Run auto-player (headless, legacy)

The 'manual' mode is the recommended default. It shows a compact snackbar
with SCAN/PAUSE/CLOSE buttons and draws arrows on the two gems to swap.
The user taps manually; the bot does NOT click.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_manual():
    """Launch manual mode (default)."""
    from main_manual import main
    main()


def cmd_panel():
    """Launch the control panel (legacy - use manual mode instead)."""
    from panel.overlay import main as panel_main
    panel_main()


def cmd_collect():
    """Run the data collector."""
    from detector.collect import main as collect_main
    collect_main()


def cmd_detect():
    """Test detection on reference images."""
    from detector.test_detect import main as test_main
    test_main()


def cmd_test():
    """Run all self-tests."""
    print("=== Crystal Champions Bot - Self Tests ===\n")
    
    # Test detector
    print("1. Testing detector...")
    import subprocess
    result = subprocess.run([sys.executable, "-m", "detector.test_detect"], 
                           cwd=ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print()
    
    # Test rule engine
    print("2. Testing rule engine...")
    result = subprocess.run([sys.executable, "-m", "sim.rules"], 
                           cwd=ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print()
    
    # Test planner
    print("3. Testing planner...")
    result = subprocess.run([sys.executable, "-m", "bot.planner"], 
                           cwd=ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print()
    
    # Test clicker
    print("4. Testing clicker...")
    result = subprocess.run([sys.executable, "-m", "bot.clicker"], 
                           cwd=ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print()
    
    # Test detector module
    print("5. Testing detector module...")
    result = subprocess.run([sys.executable, "-m", "detector"], 
                           cwd=ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print()
    
    print("=== All tests completed ===")


def cmd_auto():
    """Run auto-player headless (for testing)."""
    print("Auto-player headless mode - use panel for full integration")
    # This would run the integrated auto-player headless
    from core.auto_player import IntegratedAutoPlayer, AutoConfig
    
    # Example ROI - adjust for your setup
    roi = {'left': 788, 'top': 72, 'width': 661, 'height': 661}
    hud_roi = {'left': 788, 'top': 14, 'width': 430, 'height': 46}
    
    config = AutoConfig(
        move_delay=1.0,
        cascade_wait=1.0,
        save_debug_frames=True
    )
    
    player = IntegratedAutoPlayer(roi, hud_roi=hud_roi, config=config)
    print("AutoPlayer created. Press Ctrl+C to stop.")
    
    try:
        player.start()
        # Keep running
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        player.stop()


def main():
    # Default to manual mode if no command given
    if len(sys.argv) < 2:
        cmd_manual()
        return

    cmd = sys.argv[1].lower()

    commands = {
        'manual': cmd_manual,
        'panel': cmd_panel,
        'collect': cmd_collect,
        'detect': cmd_detect,
        'test': cmd_test,
        'auto': cmd_auto,
    }

    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)

    commands[cmd]()


if __name__ == '__main__':
    main()