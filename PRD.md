# PRD — Crystal Champions Helper

**Document type:** product + technical specification, written from the design
session. It captures *what we are building*, *why each decision was made*, and
*what we measured* — so that a future session (or a future human) can pick up
the work without re-deriving any of it.

**Status:** Stage 1 (detection) **complete and validated**. Stages 2–4
(panel, planner, auto-play) **specified, not built**.

---

## 1. The problem, in one paragraph

A friend and I play **Crystal Champions**, an 8×8 match-3 on
`worldwinner.com`, against each other. Whoever scores more pays for the next
week's party. I am the weaker player, so I want an edge. The edge I want is
**not a better reflex** — it is a machine that looks at the board and tells me,
or does for me, the swap that is worth the most points.

The game runs in a canvas inside Chrome, on a Windows laptop. We are not
allowed to touch the page: no console injection, no browser extension, no
DOM scraping. Everything has to come from **outside** the browser, the way a
human's eyes work — screen pixels in, synthetic mouse taps out.

---

## 2. What the game actually is (research findings)

Derived from the two supplied recordings and six gameplay screenshots.

### 2.1 Board and gems

* **8 columns × 8 rows** on a dark navy tile grid, inside a rounded silver
  (or gold) frame, on a purple nebula background.
* **7 gem types.** Shape and colour are *almost* one-to-one, but **not quite**,
  and that is the crux of the whole detection problem:

  | # | Type | Shape | OpenCV hue (H, 0–179) |
  |---|---|---|---|
  | 1 | green | rounded cube / square | ~55 |
  | 2 | red | diamond | ~2 (wraps at 179) |
  | 3 | blue | faceted ball | ~102 |
  | 4 | purple / magenta | diamond | ~145 |
  | 5 | yellow | triangle / pyramid | ~24 |
  | 6 | orange | hexagon | ~17 |
  | 7 | white | iridescent pearl | meaningless → whiteness test |

  * **red and purple are both diamonds.**
  * **blue and the white pearl are both round.**
  * The orange hexagon has a bright yellow/white **X painted across it** and a
    white specular sparkle — this is what breaks naive hue averaging.

### 2.2 Cell variants

* **Wooden cage** — a tan/brown square frame around a gem. The gem is still
  visible and still matchable.
* **Red swirl aura** — a red vortex ring around a gem. Appears on gems of any
  colour (seen on a yellow triangle, a blue ball and a white pearl).

### 2.3 HUD

* **Top of board:** a strip of seven small icons, one per gem type.
* **Left:** a pink counter, a `x1`/`x2` multiplier, a star, and the **score**.
* **Bottom:** a small square button, a magenta striped **moves-remaining**
  meter, a **moves-remaining** number, and a **countdown timer**.
* Screenshots prove the striped bar and the number agree: 7 moves → bar full,
  6 → bar 6/7 striped, 4 → 4/7, 2 → 2/7.
* **A second chrome exists**: the same game in a *gold* theme (gold frame,
  gold HUD) and a **HYPERSPEED** mode with its own `6:34.02` timer and a
  `0:55` countdown. The gems themselves are identical art in both themes.

### 2.4 Rules we have from the "HOW TO PLAY" dialog

* *Matches* — "Swap Gems to make 3+ matches of the same colour in a row or
  column. Match 4+ to create Special Gems!"
* *Hint* — "Find swaps that create multiple matches in one go!"
* The dialog has **4 pages**; the supplied recording only ever shows page 1,
  so **the special-gem rules are still unknown to us**.

### 2.5 Open questions we could not answer from the media

* Is the goal score-based, or "collect N of each colour"? (the top strip may
  be a legend rather than objectives)
* What exactly are the special gems — striped? wrapped? rainbow? What does the
  cage need to be broken?
* Does the board refill from the top after a match (gravity)? What is the
  actual score formula? What ends the round?
* → **One complete round recording, start to finish, would answer all four.**

---

## 3. The machine it has to run on

Measured, not assumed.

| | |
|---|---|
| CPU | 11th Gen Intel Core i3-1115G4 @ 3.0 GHz — 2 cores / 4 threads, 2.5 MB L2, 6 MB L3 |
| RAM | **8 GB** (2 × 4 GB SK Hynix DDR4-3200, soldered) — only ~0.7 GB free at rest |
| GPU | Intel Iris Xe / UHD Graphics — **no dGPU, no dedicated VRAM** |
| Disk | NVMe Intel SSDPEKNU512GZ, 476.9 GB, **20 GB free** |
| OS | Windows 11 Home, 64-bit, build 26200 |
| Display | single 1920×1080 internal panel |
| Python | 3.13.7 with `mss`, `dxcam`, `numpy`, `opencv-python`, `onnxruntime`, `pillow`, `tkinter` |

**Consequences baked into the design:**

* **No torch.** A CNN framework would eat memory we do not have. `onnxruntime`
  is present if a small model is ever needed.
* **2 cores.** Every hot loop must be bounded. A 2 fps detection loop costs
  ~15 % of one core — acceptable.
* **~0.7 GB free RAM.** Everything is a small array; nothing is loaded that
  does not have to be.
* **One 1920×1080 screen.** No multi-monitor edge cases.

---

## 4. Constraints we agreed on (and why they matter)

| Constraint | Consequence for the design |
|---|---|
| Python only | everything below is Python + OpenCV + numpy |
| **MSS** for capture | MSS was chosen because other capture routes made the game pause. Measured: full-HD grab **34 ms**, 700×700 region grab **16.7 ms** |
| **Region-only** capture | we never grab the whole desktop once the ROI is known |
| **Never resize / zoom / minimise / scale** the browser window | the board's screen coordinates are **constant** → we hardcode the ROI after one calibration instead of hunting for the board every frame |
| **Tap-tap** swap, not drag | two synthetic clicks ~150–250 ms apart: gem A then gem B |
| **Never switch windows** | the app must create **no focusable window**: no console, no taskbar entry, no Alt-Tab target |
| **Cannot change the window** | the game pauses on focus loss — that is almost certainly what "capturing normally pauses the game" actually means |
| **Let the system settle before the match** | warm-up (imports, model load, dummy grabs, one full detect, one dummy plan) must complete *before* the round starts, so no cold path can fire mid-round |
| **Do not mess with the personal laptop** | no installs at runtime, no registry, no services, no drivers, no firewall rules, no autostart, no scheduler. One folder, double-click to run, delete the folder to remove it |
| **Fully automated** | the panel buttons are *manual overrides*, not controls. The system self-heals on its own |

---

## 5. Target architecture

```
 ┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐
 │ 1. CAPTURE  │──▶│ 2. LOCATE    │──▶│ 3. READ      │──▶│ 4. PLAN    │──▶│ 5. ACT      │
 │ MSS region  │   │ board + 8x8  │   │ 64 cells,    │   │ rank all   │   │ overlay +    │
 │ ~17 ms      │   │ lattice      │   │ shape+colour │   │ swaps      │   │ 2 taps       │
 └─────────────┘   └──────────────┘   └──────────────┘   └────────────┘   └──────────────┘
```

### 5.1 Capture

* **MSS** region grab of the calibrated board rectangle. ~17 ms.
* The region is fixed; no full-screen grabs during play.
* **The overlay must never overlap the board**, or it poisons the capture.
  This is a hard architectural constraint, not a style choice.

### 5.2 Locate

Because the window never changes size, the board rectangle is calibrated once
and stored. Each frame we only **validate** it (cheap) and re-locate
automatically if validation fails.

### 5.3 Read — the part that is built and measured

See §7.

### 5.4 Plan (not yet built)

* Enumerate all 112 adjacent pairs in an 8×8 board; keep those that produce a
  match.
* Score each candidate:
  * deterministic **first cascade** (exact),
  * **Monte-Carlo** the rest of the cascade — refill is random, so cascades 2+
    can only be estimated; sample the real colour distribution and average,
  * **special-gem creation** (match-4 → striped, match-5 / L-T → wrapped,
    rainbow) and the **combo matrix** (striped+striped, striped+wrapped,
    wrapped+wrapped, rainbow+basic) — these are where the big points live,
  * **objective weighting** — if the level is "collect N of each colour", the
    best move is not the biggest match, it is the match that advances a needed
    colour.
* Keep it fast: with ~35 s per move, a 50 ms planner is already 0.1 % of the
  budget. **Do not** run the expensive Monte-Carlo over all 112 swaps; prune
  to the valid ones first, and optionally run the deep search in a background
  thread during the previous move's animation.

### 5.5 Act (not yet built)

* Overlay: layered topmost window, `WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW |
  WS_EX_LAYERED`, click-through, no taskbar entry, no console
  (`pythonw.exe`). It must never steal focus from Chrome.
* Input: two `SendInput` clicks on the two gem centres, ~180 ms apart.
* **The cursor problem.** Auto-play moves the real pointer. A pointer sitting
  on the board corrupts the cell under it — which is the cell we are about to
  read or click. Options: park the cursor in the dead zone between reads, or
  hide it for the duration. Parking is preferred (no system setting touched).

---

## 6. The control panel

Design agreed in session:

```
┌──────────────────────────────────┐
│ ● PLAYING   8fps  conf 97%       │   status line
│ best: R7C3↔R7C4  +180  3↑        │   current plan
├──────────────────────────────────┤
│   [  RECOVER  ]     [  CLOSE  ]  │
└──────────────────────────────────┘
```

* **Two buttons, not three.** SCAN and CALIBRATE were merged because the
  system is fully automated and the panel is only a manual override:
  **RECOVER** re-reads the board, validates the ROI, re-learns the palette and
  re-locates the board *if validation failed*. It is idempotent — if nothing is
  broken it just shows the reading and changes nothing. It also pops the
  diagnostic 8×8 grid so the user can see what the machine sees.
* **CLOSE** is two-stage mid-round: first click **disarms** (stops clicking
  immediately), second click exits the process. No accidental exit mid-round.
* **A hotkey kill** (F8) so it can be stopped without aiming at the panel.
* The panel lives in the **purple dead zone to the left of the board**, never
  over the 8×8.
* Buttons stay **greyed out** until warm-up completes; then the panel is
  "settled" and the buttons go live.
* State machine: `BOOT → READY → PLAYING → DONE`.

---

## 7. Detection — design, lessons, and measured results

### 7.1 Finding the board (theme-agnostic)

First attempt — peak-picking the projection of a "gem mask" — **failed**: the
purple nebula background is itself saturated, so it produced bogus peaks and
an 8-column lattice that ran off the screen.

The working method:

1. **Gem mask** = bright pixels: `V > 185 & S > 50` (plus `V > 195 & S ≤ 50`
   for the pearl). The wooden cage (V≈169) and the swirl aura (V≈186) are
   deliberately *outside* this mask, so they never pollute the blob list.
2. **Blob filter**: area 500–6000, width/height 30–130, aspect 0.5–2.0,
   fill ≥ 0.35. This keeps the 64 gems and throws away UI text, the objective
   strip icons and the background streaks.
3. **Pitch search**: for each candidate pitch `s` in 38…165 px, fit a lattice
   independently on the x and the y axis:
   * find the best **phase**,
   * keep only a run of **8 CONSECUTIVE** occupied grid lines (this is what
     kills harmonic pitches: with a half-pitch the real columns land on every
     2nd slot, so no run of 8 consecutive slots holds the board),
   * score by coverage × coverage × **evenness** of the spacing (the real
     board is perfectly regular; a bad fit is lumpy).
4. Refine each grid line to the local centroid of the gem mass.

*Lesson:* the "consecutive run" constraint is what makes this robust; the
evenness term is what fixed a 112414 regression where an off-grid candidate
won on coverage alone.

Result: **all 6 screenshots, both chomes, cropped and full-desktop, board
located correctly**, pitch 72.7 px (cropped) and 81.5 px (full desktop).

### 7.2 Reading a cell

Two independent channels, deliberately kept separate:

**Colour — from the cell *core*** (central 22 % of the pitch, guaranteed to be
pure gem material, free of cage, swirl and tile background).

* **Circular median of hue, not the mean.** The orange hexagon has a big
  yellow X across it; a *mean* hue slides into yellow and the gem is
  misread. The median ignores it because the X covers less than half the gem.
  This single change took the orange/yellow confusion from 4 errors to 1.
* **Whiteness test for the pearl.** The pearl is iridescent, so its hue is
  meaningless. `whiteness = median over pixels of min(B,G,R)/max(B,G,R)`:
  every coloured gem sits near 0.1, the pearl near 0.7–0.95. Clean
  separation, no overlap.
* Hue bands, tuned on the real art: red < 6.5 or > 172, orange < 21,
  yellow < 33, green < 82, blue < 128, purple < 170.

**Shape — from the cell *window* blob** (the bright blob of the gem body).

`extent = area / bbox_area` separates the shapes almost on its own:

| shape | extent | extra discriminator |
|---|---|---|
| rounded cube (green) | ~0.90 | — |
| circle (blue / pearl) | ~0.785 | corners ≤ 3 |
| hexagon (orange) | ~0.75 | corners ≥ 4 |
| diamond (red / purple) | ~0.50 | aspect 1.0, 4 sharp corners |
| triangle (yellow) | ~0.50 | aspect ≈ 1.15, 3 corners, flat bottom |

**Cross-check.** The shape family is compared against the shape implied by the
colour. Agreement → confidence 0.98. Disagreement → confidence 0.35, i.e. the
cell is *suspect* and the bot refuses to act on it.

### 7.3 Variants (cage / swirl)

The first attempt compared wrapper colour against the gem's own bounding box
and produced 68 false positives, because the gems **fill their cell** — a red
gem looks like a swirl and an orange gem looks like a cage.

The working method:

1. Subtract the **bright gem body** (`V > 210`) from the red and tan masks.
   The threshold is high on purpose: the swirl sits at V≈186 and the cage at
   V≈169, so both must *survive* the mask.
2. A wrapper is a **hollow frame** that fills most of the cell, so:
   * bounding box ≥ 0.72–0.78 × pitch,
   * roughly square (a frame is not a streak),
   * it **encloses a hole** (found via `RETR_CCOMP`: a frame gets a child
     contour; a solid gem does not),
   * area ≥ 1.5 % of the cell.

Current numbers: **precision 100 %, recall 6.5 %** — the hole test is strict
and misses most real cages. This is the biggest open gap in the read stage
and is called out in §10.

### 7.4 Measured accuracy

Ground truth: the 6 screenshots were **transcribed by eye**, cell by cell
(384 cells), into `detector/ground_truth.json` — independently of the
detector. The annotated montages in `detector/verify/` are the audit trail.

| Classifier | Channel | Result |
|---|---|---|
| **RULE** | hue band + shape cross-check | **383 / 384 = 99.74 %** |
| CENTROID | nearest neighbour in (hue, sat, val) | 321 / 384 = 83.6 % |
| SHAPE | silhouette template match, colour-blind | 317 / 384 = 82.6 % |

Confusion matrix for RULE — **zero confusions** except one:

```
              green  red  blue  purple  yellow  orange  white
green            68    0     0       0       0      0      0
red               0   50     0       0       0      0      0
blue              0    0    66       0       0      0      0
purple            0    0     0      54       0      0      0
yellow            0    0     0       0      29      0      0
orange            0    0     0       0       1     58      0
white             0    0     0       0       0      0     58
```

The one error: `112522 r5c8` — a washed-out orange hexagon, apparently
mid-spawn-animation, read as yellow. **It was flagged with confidence 0.35**,
so with a confidence gate of 0.80 it becomes an *unknown*, not a wrong answer.
Gated accuracy is therefore 100 % of the answers given, with 1 cell unknown.

---

## 8. Reliability rules (the part that keeps a bad read from costing a round)

1. **Never click on a partially-read board.** Every cell must produce a
   confident read; otherwise wait and re-read.
2. **Re-verify the two cells you are about to touch** with a second,
   independent pass before clicking.
3. **Verify the click afterwards.** If the board did not change as predicted,
   the click missed → retry once → then disarm and report.
4. **Compare the actual result to the prediction** and log the difference.
   This is the self-correcting loop: the system learns the real scoring
   formula instead of assuming one.
5. **Detect round end** (timer 0 / moves 0 / results screen) and stop
   clicking — never click into a results screen.
6. **Bounded retries.** Never loop forever on one candidate move.

---

## 9. The bot loop (fully automated)

```
warm up  →  detect board  →  validate ROI  →  read 64 cells (multi-frame vote)
        →  any UNKNOWN? wait / re-read / auto-RECOVER
        →  plan  →  re-verify the two target cells  →  tap A, tap B
        →  verify the board changed  →  compare with prediction, log
        →  detect round end? stop  →  else loop
```

---

## 10. Known gaps, risks and open decisions

| # | Item | Severity |
|---|---|---|
| 1 | **Variant detection recall is 6.5 %** — cages and swirls are mostly missed. A caged cell must not be swapped blindly. | high |
| 2 | **Special gems are unidentified.** Only one charged gem (a red swirl) has ever been seen. The HOW TO PLAY pages 2–4 were never captured. | high |
| 3 | **The score formula is unknown.** We are guessing at points per gem and per cascade. | high |
| 4 | **The objective strip** may be a legend, not objectives. If it is objectives with counts, we need OCR on a ~12 px strip. | medium |
| 5 | **Whether the board refills from the top** (gravity) is unknown; it decides whether cascade simulation is meaningful at all. | medium |
| 6 | **The cursor** must be parked off-board during reads, or hidden. | medium |
| 7 | **Terms of service.** `worldwinner.com` is a real-cash play platform. Automated play there may breach their terms and could put the account and any payout at risk. Building and tuning against a free/practice round or a local clone first, and only enabling it for real money when we are comfortable, is the sane order of operations. | high |
| 8 | **Three OpenCV packages are installed** (`opencv-python`, `opencv-python-headless`, `opencv-contrib-python`) and they all write the same `cv2` module. Clean up to one. | low |
| 9 | **Only 20 GB free disk** on the laptop. | low |
| 10 | **MSS's `mss.mss()` is deprecated** in 10.2 — use `mss.MSS()`. | low |

---

## 11. Roadmap

* **Stage 1 — detection** ✅ *done and measured* (this repo).
  Next: fix variant recall, learn the special gems from a full-round
  recording.
* **Stage 2 — the panel** — two-button control panel, `pythonw` launcher,
  RECOVER/CLOSE, hotkey kill, diagnostic 8×8 view.
* **Stage 3 — the planner** — move enumeration, cascade simulation, special
  combos, objective weighting, and the persistent move store with
  predicted-vs-actual logging.
* **Stage 4 — auto-play** — tap-tap input, click verification, round-end
  detection, cursor parking, self-healing.

---

## 12. Repository layout

```
.
├── PRD.md                     ← this document
├── README.md
├── .gitignore
├── images of cristels/        ← 6 sample screenshots (the test corpus)
└── detector/
    ├── gemvision.py           ← detection library
    ├── test_detect.py         ← run over the corpus, print the 8×8 grids
    ├── accuracy_report.py     ← score vs ground truth + confusion matrix
    ├── make_verify.py         ← annotated montages for eyeball auditing
    ├── ground_truth.json      ← hand-transcribed reference labels
    └── accuracy.json          ← machine-written result of the last run
```

---

## 13. Design decisions log (the "why" behind each choice)

| Decision | Alternative rejected | Reason |
|---|---|---|
| MSS for capture | DXcam, Windows Graphics Capture | MSS is the only one that does not disturb the game; WGC would be the better technical choice but adds a dependency and was not needed. DXcam is fastest but the margin did not matter at a 35 s/move budget |
| Region-only capture | full-screen grab | 34 ms → 17 ms, and it is the input to everything else |
| Hardcoded ROI + validation | re-detect the board every frame | the window never changes size, so the ROI is constant; detection is only needed after a failure |
| Two channels (shape, colour) | colour only | red/purple and blue/pearl are same-shape; colour alone cannot separate them |
| Circular median hue | mean hue | the orange hexagon's X highlight drags a mean into yellow |
| Whiteness test for the pearl | hue band | the pearl's hue is meaningless by construction |
| Consecutive-8-slot lattice constraint | peak picking | kills harmonic pitches; peak picking failed on the saturated background |
| Confidence gating (no guess) | always answer | a skipped move costs nothing, a wrong move costs the round |
| Two-button panel | three buttons | the system is automated; SCAN and CALIBRATE are the same code path |
| `pythonw.exe`, no console | console app | the game pauses on focus loss; a console window would steal focus and kill the round |
| Overlay outside the board | overlay on the board | the overlay would be captured by MSS and poison the read |
| No ML model | small CNN via ONNX | 8 GB RAM with 0.7 GB free, 2 cores; the rule-based classifier hit 99.74 % and is explainable. ONNX stays available if specials need it |
