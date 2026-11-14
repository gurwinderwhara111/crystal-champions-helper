# Crystal Champions Helper

A vision + planning assistant for **Crystal Champions** (the 8x8 match-3 on
`worldwinner.com`), written in Python for Windows.

It reads the board straight off the screen with **MSS**, works out what every
gem is (shape **and** colour), and — in the full build — plans the best swap
and plays it with two synthetic taps.

> **Status:** the **detection** stage is finished and measured (98.96 % of 384
> hand-labelled cells identified, 0 wrong). The planner, the overlay panel and
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
| **RULE** (primary) | hue band + silhouette **veto** | **380 / 384 = 98.96 % identified, 0 wrong** |
| CENTROID | nearest neighbour in (hue, sat, val) vs learned references | 370 / 384 = 96.35 % |
| SHAPE | silhouette template match only (colour-blind) | 349 / 384 = 90.89 % |
| TEMPLATE | full-appearance template match (outline + engraving + colour) | 357 / 384 = 92.97 % |

The metric that matters for an auto-player is **precision, not recall**:
of the 380 cells the system claims to know, **380 are right**. It refuses 4
cells — 3 of them the red-swirl special gems we have not modelled yet (the
swirl distorts the measured shape, so the veto fires), and 1 a washed-out
orange caught mid-spawn-animation. It never guesses wrong on this corpus, and
there are **zero confusions** in the confusion matrix.

The single earlier error (a washed-out orange hexagon read as yellow) is now
also refused rather than answered.

## The 7 gem types

| Type | Silhouette | Engraving inside | Extent | Colour (OpenCV H) |
|---|---|---|---|---|
| green | rounded cube / square | vertical specular bar | 0.81 | ~55 |
| orange | hexagon | 4-pointed star | 0.78 | ~17 |
| blue | rounded hexagon / faceted ball | diagonal + horizontal bands | 0.74 | ~102 |
| white | sphere, iridescent | hexagonal tessellation | 0.68 | hue is meaningless → whiteness test |
| red | diamond | V / heart | 0.60 | ~2 (wraps at 179) |
| purple | diamond | concentric diamond | 0.53 | ~145 |
| yellow | triangle | inner facet | 0.53 | ~24 |

**Seven types, six distinct silhouettes.** Only **red and purple share an
outline** — both are diamonds — and they are told apart by the engraving
inside them (V/heart vs concentric diamond) and by colour. Every other type
is unique by outline alone, and the measured extent falls into five families
(sharp / mid / pearl / ball / blocky).

That is why the classifier uses shape and colour as two independent channels
instead of trusting either one alone, and why the full-appearance TEMPLATE
classifier can name all seven: it matches the outline, the engraving and the
colour together, against reference patches taken from real gems.

The detector keeps **two** shape masks on purpose: a bright-mask blob (scale
consistent across the corpus, used for the veto) and a background-segmented
silhouette (the gem's true outline, used for identification). See §7.5 of the
PRD for why one of them cannot do both jobs.

## Licence

Personal project, for a private game between friends. See `PRD.md` for the
terms-of-service note — read it before pointing this at a real-money session.
