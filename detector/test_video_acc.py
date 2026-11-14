import cv2
import gemvision as gv

cap = cv2.VideoCapture(r'D:\random works by gs\Crystal Champions helper\2nd gameplay video close capture.mp4')
print('Testing classification accuracy on video frames...')

total_cells = 0
identified = 0
unknown = 0
frames_tested = 0

for i in range(0, 1784, 50):  # every 50th frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue
    board = gv.find_board(frame)
    if board is None:
        continue
    frames_tested += 1
    roi = {'left': 0, 'top': 0, 'width': frame.shape[1], 'height': frame.shape[0]}
    grid = gv.detect_board(frame, roi)
    for row in grid:
        for c in row:
            total_cells += 1
            if c == '?':
                unknown += 1
            else:
                identified += 1

print(f'Frames tested: {frames_tested}')
print(f'Total cells: {total_cells}')
print(f'Identified: {identified} ({100*identified/total_cells:.1f}%)')
print(f'Unknown: {unknown} ({100*unknown/total_cells:.1f}%)')
cap.release()