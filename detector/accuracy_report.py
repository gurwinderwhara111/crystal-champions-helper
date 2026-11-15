"""
accuracy_report.py - score the detector against the hand-transcribed ground
truth and print a confusion matrix.

Three independent classifiers are scored:
  RULE     : hue bands (hard coded) cross-checked against silhouette shape
  CENTROID : nearest neighbour in (hue, sat, val) against learned references
  NCC      : per-channel correlation against learned reference cores
"""
import os
import sys
import json
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemvision as gv
from test_detect import analyse, load_images, build_references

HERE = os.path.dirname(os.path.abspath(__file__))
GT_FILE = os.path.join(HERE, "ground_truth.json")
CONFIDENT = 0.80          # shape/hue agreement threshold


def main():
    gt = json.load(open(GT_FILE))
    images = load_images()
    clf = gv.Classifier(build_references(images))

    totals = collections.Counter()
    confusion = {m: collections.Counter() for m in ("RULE", "CENTROID", "SHAPE", "TEMPLATE")}
    errors = []

    for f, im in images:
        key = None
        for k in gt:
            if k in f:
                key = k
                break
        if key is None:
            continue
        res = analyse(im)
        if res is None:
            print(f"{key}: BOARD NOT FOUND")
            continue
        truth = gt[key]["types"]
        for idx, cell in enumerate(res["cells"]):
            r, c = divmod(idx, 8)
            t = truth[r][c]
            t_rule, conf = clf.rule_predict(cell["core"], cell["blob"])
            t_cen, _ = clf.centroid_predict(cell["core"])
            t_shape, _ = clf.shape_predict(cell["blob"])
            t_tpl, _ = clf.template_predict(cell["patch"])
            totals["cells"] += 1
            for name, pred in (("RULE", t_rule), ("CENTROID", t_cen),
                               ("SHAPE", t_shape), ("TEMPLATE", t_tpl)):
                confusion[name][(t, pred)] += 1
                if pred == t:
                    totals[name + "_ok"] += 1
                else:
                    totals[name + "_bad"] += 1
            if t_rule != t:
                errors.append((key, r + 1, c + 1, t, t_rule, conf))

    print("\n================ ACCURACY ================")
    n = totals["cells"]
    for name in ("RULE", "CENTROID", "SHAPE", "TEMPLATE"):
        ok = totals[name + "_ok"]
        print(f"  {name:9s} {ok}/{n} = {100.0*ok/n:.2f}%   errors={totals[name+'_bad']}")
    print(f"\n  total cells scored: {n}")
    # what a confidence gate would do
    low = [e for e in errors if e[5] < 0.80]
    print(f"  cells the shape cross-check flagged as low confidence: {len(low)}"
          f"  -> gated accuracy {100.0*(n-len(low))/n:.2f}% with {len(low)} unknown")

    print("\n  confusion matrix (RULE)  rows = truth, cols = prediction")
    types = gv.TYPES
    print("            " + "".join(f"{t[:5]:>7s}" for t in types) + "   UNK")
    for t in types:
        row = confusion["RULE"]
        line = f"  {t:10s} "
        for p in types:
            line += f"{row[(t,p)]:7d}"
        line += f"{row[(t,None)]:6d}"
        print(line)

    print("\n  disagreements vs ground truth:")
    if not errors:
        print("    none")
    for e in errors:
        print(f"    {e[0]}  r{e[1]}c{e[2]}: truth={e[3]:7s} pred={str(e[4]):7s} conf={e[5]:.2f}")

    # variant check (only the cells we are certain about)
    print("\n  variant flags (cage / swirl):")
    tp = fp = fn = 0
    for f, im in images:
        key = next((k for k in gt if k in f), None)
        if key is None:
            continue
        res = analyse(im)
        want_cage = {tuple(p) for p in gt[key].get("cage", [])}
        want_swirl = {tuple(p) for p in gt[key].get("swirl", [])}
        for idx, cell in enumerate(res["cells"]):
            r, c = divmod(idx, 8)
            fl = cell["flags"]
            for name, want in (("cage", want_cage), ("swirl", want_swirl)):
                has = name in fl
                should = (r + 1, c + 1) in want
                if has and should:
                    tp += 1
                elif has and not should:
                    fp += 1
                elif should and not has:
                    fn += 1
    print(f"    true positives={tp}  false positives={fp}  missed={fn}")
    if tp + fp:
        print(f"    precision={100.0*tp/max(1,tp+fp):.1f}%  recall={100.0*tp/max(1,tp+fn):.1f}%")

    out = {"cells": n,
           "rule_accuracy": totals["RULE_ok"] / n,
           "centroid_accuracy": totals["CENTROID_ok"] / n,
           "shape_accuracy": totals["SHAPE_ok"] / n,
           "template_accuracy": totals["TEMPLATE_ok"] / n,
           "errors": [{"image": e[0], "row": e[1], "col": e[2], "truth": e[3],
                       "pred": e[4], "conf": e[5]} for e in errors]}
    with open(os.path.join(HERE, "accuracy.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\n  wrote accuracy.json")


if __name__ == "__main__":
    main()
