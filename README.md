# Crystal Champions Helper

A vision + planning assistant for **Crystal Champions** (the 8x8 match-3 on
`worldwinner.com`), written in Python for Windows.

It reads the board straight off the screen with **MSS**, works out what every
gem is (shape **and** colour), and — in the full build — plans the best swap
and plays it with two synthetic taps.

> **Status:** the **detection** stage is finished and measured (99.74 % type
> accuracy over 384 hand-labelled cells). The planner, the overlay panel and
> the auto-play loop are specified in [`PRD.md`](PRD.md) but not built yet.

---

## What is in here

| Path | What it is |
|---|---|
| [`PRD.md`](PRD.md) | Full product + technical spec: why the system exists, every decision we made, the measured numbers, the known gaps and the roadmap |
| [`detector/gemvision.py`](detector/gemvision.py) | The detection library: board location, cell segmentation, gem classification |
| [`detector/test_detect.py`](detector/test_detect.py) | Runs the detector over the sample screenshots and prints the 8x8 grid it reads |
| [`detector/accuracy_report.py`](detector/accuracy_report.py) | Scores the detector against the hand-transcribed ground truth, prints a confusion matrix |
| [`detector/make_verify.py`](detector/make_verify.py) | Builds an annotated 8x8 montage per screenshot so a human can check the reading at a glance |
| [`detector/ground_truth.json`](detector/ground_truth.json) | The 6 screenshots transcribed by eye — the reference the detector is scored against |
| [`detector/accuracy.json`](detector/accuracy.json) | The machine-written result of the last scoring run |
| `images of cristels/` | The 6 sample screenshots (615×670 up to 1915×927, cropped board + full desktop, silver and gold chrome) |

## Running it

```powershell
cd detector
python test_detect.py        # read every screenshot, print the 8x8 grid
python accuracy_report.py    # score against ground truth
python make_verify.py        # write verify\v_*.png montages for eyeballing
```

No installation, no services, no drivers, no registry changes. It only needs
`opencv-python`, `numpy` and (later) `mss`.

## Measured accuracy

| Classifier | What it uses | Accuracy |
|---|---|---|
| **RULE** (primary) | hue band + silhouette shape cross-check | **383 / 384 = 99.74 %** |
| CENTROID | nearest neighbour in (hue, sat, val) vs learned references | 321 / 384 = 83.6 % |
| SHAPE | silhouette template match only (colour-blind) | 317 / 384 = 82.6 % |

The single RULE error was a washed-out orange hexagon that reads as yellow —
and the shape cross-check flagged it with **confidence 0.35**, so a
confidence gate would have turned it into an *unknown* rather than a wrong
move. Confusions are zero across every other pair.

## The 7 gem types

| Type | Shape | Colour (OpenCV H) |
|---|---|---|
| green | rounded cube | ~55 |
| red | diamond | ~2 (wraps at 179) |
| blue | faceted ball | ~102 |
| purple | diamond | ~145 |
| yellow | triangle / pyramid | ~24 |
| orange | hexagon | ~17 |
| white | iridescent pearl | hue is meaningless → whiteness test |

Two of them are the *same shape* (red / purple diamonds) and two more are the
*same shape* (blue ball / white pearl). That is exactly why the classifier
uses shape and colour as two independent channels instead of trusting either
one alone.

## Licence

Personal project, for a private game between friends. See `PRD.md` for the
terms-of-service note — read it before pointing this at a real-money session.
