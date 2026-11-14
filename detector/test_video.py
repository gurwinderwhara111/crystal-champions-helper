import cv2
import gemvision as gv

cap = cv2.VideoCapture(r'D:\random works by gs\Crystal Champions helper\2nd gameplay video close capture.mp4')
print('Testing board detection on video frames...')
for i in range(0, 1784, 100):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue
    board = gv.find_board(frame)
    if board:
        print(f'Frame {i}: board FOUND, pitch={board["pitch"]:.1f}, centres={len(board["centres"])}')
    else:
        pts, _ = gv._gem_blobs(frame)
        print(f'Frame {i}: NO BOARD (blobs={len(pts)}, mean={frame.mean():.1f})')
cap.release()