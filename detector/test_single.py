import cv2
import gemvision as gv

path = r'D:\random works by gs\Crystal Champions helper\images of cristels\Screenshot 2026-11-13 112414.png'
im = cv2.imread(path)
print(f'Image: {im.shape}')

board = gv.find_board(im, verbose=True)
if board:
    print(f'Board found: pitch={board["pitch"]:.1f}, centres={len(board["centres"])}')
    roi = {'left': 0, 'top': 0, 'width': im.shape[1], 'height': im.shape[0]}
    grid = gv.detect_board(im, roi)
    print('Grid:')
    for row in grid:
        print(' ', ' '.join(r[0].upper() if r != '?' else '?' for r in row))
    
    total = 64
    unknown = sum(1 for row in grid for c in row if c == '?')
    print(f'Identified: {total-unknown}/{total} ({100*(total-unknown)/total:.1f}%)')
else:
    print('Board NOT found')