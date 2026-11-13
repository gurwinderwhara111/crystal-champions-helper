"""
make_verify.py - build an annotated 8x8 montage per screenshot so the
reading can be checked by eye.

Each tile is the actual cell crop with the predicted label drawn on it.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemvision as gv
from test_detect import analyse, load_images, build_references, GLYPH

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify")


def main():
    os.makedirs(OUT, exist_ok=True)
    images = load_images()
    clf = gv.Classifier(build_references(images))

    for f, im in images:
        res = analyse(im)
        if res is None:
            print(f, "NO BOARD")
            continue
        pitch = res["board"]["pitch"]
        tiles = []
        for cell in res["cells"]:
            half = pitch * 0.46
            x0, x1 = int(cell["cx"] - half), int(cell["cx"] + half)
            y0, y1 = int(cell["cy"] - half), int(cell["cy"] + half)
            x0, y0 = max(0, x0), max(0, y0)
            x1 = min(im.shape[1], x1)
            y1 = min(im.shape[0], y1)
            crop = im[y0:y1, x0:x1].copy()
            t, conf = clf.rule_predict(cell["core"], cell["blob"])
            crop = cv2.resize(crop, (76, 76), interpolation=cv2.INTER_AREA)
            g = GLYPH.get(t, "?")
            fl = cell["flags"]
            if "cage" in fl:
                g = g.lower()
            if "swirl" in fl:
                g = g + "*"
            cv2.rectangle(crop, (0, 0), (75, 75), (255, 255, 255), 1)
            cv2.putText(crop, g, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(crop, g, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)

        rows = []
        for r in range(8):
            row = np.hstack(tiles[r * 8:(r + 1) * 8])
            rows.append(row)
        grid = np.vstack(rows)
        pad = np.zeros((grid.shape[0], 40, 3), np.uint8)
        out = np.hstack([grid, pad])
        name = f.split(" ")[-1].replace(".png", "")
        cv2.imwrite(os.path.join(OUT, f"v_{name}.png"), out)
        print("wrote", os.path.join(OUT, f"v_{name}.png"), out.shape)


if __name__ == "__main__":
    main()
