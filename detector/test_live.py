import cv2
import gemvision as gv
import mss
import numpy as np

with mss.MSS() as sct:
    shot = np.asarray(sct.grab(sct.monitors[1]))[:, :, :3]

board = gv.find_board(shot, verbose=True)
print(f'Board found: {board is not None}')
if board:
    print(f'  pitch: {board["pitch"]:.1f}')
    print(f'  centres: {len(board["centres"])}')
    xs, ys = board['xs'], board['ys']
    left = int(min(xs) - board["pitch"] * 0.55)
    top = int(min(ys) - board["pitch"] * 0.55)
    right = int(max(xs) + board["pitch"] * 0.55)
    bottom = int(max(ys) + board["pitch"] * 0.55)
    print(f'  ROI: left={left}, top={top}, w={right-left}, h={bottom-top}')
    
    # Test detection on full frame with ROI
    roi = {'left': left, 'top': top, 'width': right-left, 'height': bottom-top}
    grid = gv.detect_board(shot, roi)
    print('Grid:')
    for row in grid:
        print(' ', ' '.join(r[0].upper() if r != '?' else '?' for r in row))