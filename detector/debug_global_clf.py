import cv2
import gemvision as gv
import os

cap = cv2.VideoCapture(r'D:\random works by gs\Crystal Champions helper\2nd gameplay video close capture.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()
cap.release()

board = gv.find_board(frame)
clf = gv.get_classifier()  # global cached classifier with templates

print(f"Board found: pitch={board['pitch']:.1f}")
centres = board['centres']

for r in range(8):
    for c in range(8):
        idx = r * 8 + c
        cx, cy = centres[idx]
        core, _ = gv.cell_core(frame, cx, cy, board['pitch'])
        blob = gv.cell_blob(frame, cx, cy, board['pitch'])
        sil = gv.cell_silhouette(frame, cx, cy, board['pitch'])
        patch = gv.gem_patch(frame, cx, cy, board['pitch'], sil)
        
        if core is None or blob is None:
            print(f'  r{r}c{c}: NO CORE/BLOB')
            continue
            
        # Test all 4 classifiers
        t_rule, c_rule = clf.rule_predict(core, blob)
        t_ncc, c_ncc = clf.ncc_predict(core)
        t_temp, c_temp = clf.template_predict(patch) if patch is not None else (None, 0)
        t_cent, c_cent = clf.centroid_predict(core)
        t_shape, c_shape = clf.shape_predict(blob)
        
        # Show all
        vals = []
        if t_rule: vals.append(f"rule={t_rule}({c_rule:.2f})")
        if t_ncc: vals.append(f"ncc={t_ncc}({c_ncc:.2f})")
        if t_temp: vals.append(f"temp={t_temp}({c_temp:.2f})")
        if t_cent: vals.append(f"cent={t_cent}({c_cent:.2f})")
        if t_shape: vals.append(f"shape={t_shape}({c_shape:.2f})")
        
        print(f'  r{r}c{c}: {" | ".join(vals)}')