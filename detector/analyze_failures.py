import cv2
import gemvision as gv
import os

frame_dir = r'D:\random works by gs\Crystal Champions helper\detector\dataset\20261114_090312\frames'
frames = sorted([f for f in os.listdir(frame_dir) if f.startswith('f') and f.endswith('.png')])

print("Analyzing board-find failures...")
print()

failures = []
for idx in range(len(frames)):
    path = os.path.join(frame_dir, frames[idx])
    im = cv2.imread(path)
    if im is None:
        continue
    board = gv.find_board(im)
    if board is None:
        # Check blob count
        pts, _ = gv._gem_blobs(im)
        failures.append((frames[idx], im.shape, im.mean(), len(pts)))

print(f'Total frames: {len(frames)}')
print(f'Board find failures: {len(failures)}')
print()
print('First 20 failures:')
for fname, shape, mean, blobs in failures[:20]:
    print(f'  {fname}: shape={shape}, mean={mean:.1f}, blobs={blobs}')

# Also check what the unknown cells look like
print()
print("Sampling unknown cells from successful frames...")
for idx in range(0, len(frames), 50):
    if idx >= len(frames):
        break
    path = os.path.join(frame_dir, frames[idx])
    im = cv2.imread(path)
    if im is None:
        continue
    board = gv.find_board(im)
    if board is None:
        continue
    roi = {'left': 0, 'top': 0, 'width': im.shape[1], 'height': im.shape[0]}
    grid = gv.detect_board(im, roi)
    
    unknown_positions = []
    for r in range(8):
        for c in range(8):
            if grid[r][c] == '?':
                unknown_positions.append((r, c))
    if unknown_positions:
        print(f'  {frames[idx]}: {len(unknown_positions)} unknowns at {unknown_positions[:10]}')