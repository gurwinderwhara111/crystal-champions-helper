"""
Extract high-confidence reference cells from the close-capture video
to retrain the classifier on the actual game appearance.
"""
import cv2
import gemvision as gv
import os
import numpy as np

video_path = r'D:\random works by gs\Crystal Champions helper\2nd gameplay video close capture.mp4'
outdir = r'D:\random works by gs\Crystal Champions helper\detector\video_refs'
os.makedirs(outdir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Video: {total_frames} frames at {cap.get(cv2.CAP_PROP_FPS):.1f} fps')

# Build classifier first (uses reference images)
clf = gv.get_classifier()

# Sample frames throughout the video
sample_indices = list(range(0, total_frames, 50))  # every 50th frame
print(f'Sampling {len(sample_indices)} frames...')

collected = {t: [] for t in gv.TYPES}
max_per_type = 15

for frame_idx in sample_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue
    
    board = gv.find_board(frame)
    if board is None:
        continue
    
    roi = {'left': 0, 'top': 0, 'width': frame.shape[1], 'height': frame.shape[0]}
    grid = gv.detect_board(frame, roi)
    
    # Collect high-confidence cells (where rule_predict agrees with majority)
    centres = board['centres']
    pitch = board['pitch']
    
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            cx, cy = centres[idx]
            
            core, _ = gv.cell_core(frame, cx, cy, pitch)
            blob = gv.cell_blob(frame, cx, cy, pitch)
            sil = gv.cell_silhouette(frame, cx, cy, pitch)
            patch = gv.gem_patch(frame, cx, cy, pitch, sil)
            
            if core is None or blob is None:
                continue
            
            # Get rule prediction (our primary)
            t_rule, c_rule = clf.rule_predict(core, blob)
            if t_rule is None or c_rule < 0.95:
                continue
            
            # Verify with other classifiers (consensus)
            t_ncc, _ = clf.ncc_predict(core)
            t_temp, _ = clf.template_predict(patch) if patch is not None else (None, 0)
            t_cent, _ = clf.centroid_predict(core)
            
            votes = [t_rule]
            if t_ncc: votes.append(t_ncc)
            if t_temp: votes.append(t_temp)
            if t_cent: votes.append(t_cent)
            
            # Require at least 2 agreeing
            if votes.count(t_rule) < 2:
                continue
            
            # Check if we need more of this type
            if len(collected[t_rule]) >= max_per_type:
                continue
            
            # Save reference
            collected[t_rule].append({
                'core': core.copy(),
                'blob': {k: v for k, v in blob.items() if k != 'contour'},
                'silh': gv.silhouette(blob),
                'patch': patch.copy() if patch is not None else None,
                'source_frame': frame_idx,
                'position': (r, c)
            })
            
            # Save visual for inspection
            cv2.imwrite(os.path.join(outdir, f'{t_rule}_r{r}c{c}_f{frame_idx}.png'), 
                        cv2.resize(core, (80, 80), interpolation=cv2.INTER_AREA))

cap.release()

print('\nCollected references:')
for t in gv.TYPES:
    print(f'  {t}: {len(collected[t])}')

# Save as pickle for later use
import pickle
with open(os.path.join(outdir, 'video_refs.pkl'), 'wb') as f:
    pickle.dump(collected, f)

print(f'\nSaved to {outdir}/video_refs.pkl')