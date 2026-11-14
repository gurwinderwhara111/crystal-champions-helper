import cv2
import gemvision as gv

cap = cv2.VideoCapture(r'D:\random works by gs\Crystal Champions helper\2nd gameplay video close capture.mp4')

for i in range(0, 1784, 100):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue
    board = gv.find_board(frame)
    if board is None:
        continue
    roi = {'left': 0, 'top': 0, 'width': frame.shape[1], 'height': frame.shape[0]}
    grid = gv.detect_board(frame, roi)
    
    unknown_positions = []
    for r in range(8):
        for c in range(8):
            if grid[r][c] == '?':
                unknown_positions.append((r, c))
    
    # Show grid
    print(f'Frame {i}: {len(unknown_positions)} unknowns')
    for r in range(8):
        row = ''
        for c in range(8):
            val = grid[r][c]
            row += (val[0].upper() if val != '?' else '?') + ' '
        print(f'  {row}')
    print()

cap.release()