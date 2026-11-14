import cv2
import gemvision as gv
import os

frame_dir = r'D:\random works by gs\Crystal Champions helper\detector\dataset\20261114_090312\frames'
frames = sorted([f for f in os.listdir(frame_dir) if f.startswith('f') and f.endswith('.png')])

# Test on frames where board is found
total_cells = 0
identified = 0
unknown = 0
frames_with_board = 0

for idx in range(0, len(frames), 25):  # sample every 25th frame
    if idx >= len(frames):
        break
    path = os.path.join(frame_dir, frames[idx])
    im = cv2.imread(path)
    if im is None:
        continue
    board = gv.find_board(im)
    if board is None:
        continue
    
    frames_with_board += 1
    roi = {'left': 0, 'top': 0, 'width': im.shape[1], 'height': im.shape[0]}
    grid = gv.detect_board(im, roi)
    
    for row in grid:
        for c in row:
            total_cells += 1
            if c == '?':
                unknown += 1
            else:
                identified += 1

print(f'Frames with board found: {frames_with_board}')
print(f'Total cells: {total_cells}')
print(f'Identified: {identified} ({100*identified/total_cells:.1f}%)')
print(f'Unknown: {unknown} ({100*unknown/total_cells:.1f}%)')