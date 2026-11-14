import cv2
import gemvision as gv
import os

frame_dir = r'D:\random works by gs\Crystal Champions helper\detector\dataset\20261114_090312\frames'
frames = sorted([f for f in os.listdir(frame_dir) if f.startswith('f') and f.endswith('.png')])

print(f'Testing detector on {len(frames)} frames from new dataset...')
print()

for idx in [0, 100, 300, 500, 800, 1200, 1500, 1900]:
    if idx < len(frames):
        path = os.path.join(frame_dir, frames[idx])
        im = cv2.imread(path)
        if im is None:
            print(f'{frames[idx]}: FAILED TO LOAD')
            continue
        board = gv.find_board(im)
        if board:
            print(f'{frames[idx]}: board FOUND, pitch={board["pitch"]:.1f}, centres={len(board["centres"])}')
        else:
            print(f'{frames[idx]}: board NOT FOUND (shape={im.shape}, mean={im.mean():.1f})')