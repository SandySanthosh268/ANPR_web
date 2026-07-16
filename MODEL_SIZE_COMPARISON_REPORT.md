# Plate Model Size Comparison — YOLOv8s vs YOLOv8m

## 1. Objective

Compare the current production plate-detection weight (`vechile_plate_yolov8s.pt`,
"s"/small) against the larger "m"/medium variant (`vechile_plate_yolov8m.pt`,
~2.3x the file size — 52MB vs 22MB) on the same test video, same 5fps
decimation, same tracking/OCR/validator code — only `PLATE_MODEL_PATH` in
`app/config.py` differs between the two runs.

## 2. Result Summary

| | s-model (current production) | m-model |
|---|---|---|
| Detect avg time | 233.4ms | **497.3ms** (~2.1x slower) |
| Detect min/max | 202.4 / 452.2ms | 452.6 / 859.4ms |
| OCR avg time (CPU, both forced) | 186.7ms | 182.4ms (~same, expected — OCR itself doesn't depend on which detector found the box) |

Metrics below compare the two runs directly (single value, not one per model —
computed over the frame-pairs matched between them, see §3):

| Comparison metric | Value |
|---|---|
| Matched detections (bbox-center within 30px) | 265 |
| Plate-confidence identical | 0/265 (values differ every time — expected, different weights) |
| OCR status mismatches (accepted/rejected/no_text) | 33/265 |
| **OCR validated-plate text mismatches** | **48/265** |

## 3. Detection coordinates and raw detection counts

### Bounding-box coordinates

The two weights never produce bit-identical boxes for the same physical
plate — different training/weights means different learned box regression.
Typical drift is a **5-10px shift** on one or more edges, e.g. (all `frames.csv`,
5fps run):

| Frame | s-model bbox | m-model bbox |
|---|---|---|
| 1000 | `(117,732)-(300,797)` conf=0.744 | `(122,731)-(296,791)` conf=0.740 |
| 440 | `(720,654)-(827,713)` conf=0.587 | `(726,661)-(824,707)` conf=0.487 |
| 905 | `(880,564)-(1006,600)` conf=0.723 | `(881,560)-(1008,602)` conf=0.519 |

This is why `compare_frames.py` matches by nearest bbox-center within a
tolerance (`--match-radius`, default 30px) rather than exact coordinates —
exact matching finds zero rows across two different model weights.

### Raw detection counts — s-model fires ~30% more boxes at both frame rates

| | Every frame (1328 src frames) | 5th frame / 5fps (265 src frames) |
|---|---|---|
| s-model total detections | 2179 | 424 |
| m-model total detections | 1619 | 329 |
| Matched (bbox-center within 30px) | 1299 | 265 |
| Only in s-model (m missed it) | 880 | 159 |
| Only in m-model (s missed it) | 320 | 64 |
| s/m ratio | 1.35x | 1.29x |

The ~30% gap holds **regardless of decimation rate** — so it's not a
frame-rate artifact, it's a genuine model-behavior difference: s-model
fires on more marginal/borderline boxes (closer to the confidence
threshold), m-model is more selective/conservative. OCR mismatch
proportion is also consistent across both rates (~17-19%: 48/265 at 5fps,
225/1299 at every-frame), and the same `TN03D9766` pattern repeats at
every-frame too — s-model repeatedly rejects/no-texts frames the m-model
reads cleanly (e.g. frames 93, 100, 103, 115, 169, 178, 210, 244, 248, 249,
490 in the every-frame run).

## 4. Accuracy — m-model reads plates more reliably

Of the 48 mismatched readings, the clearest signal is on the long-running
`TN03D9766` track (a near-stationary vehicle visible for most of the video):

| | Count |
|---|---|
| s-model failed (`no_text`/`rejected`), m-model correctly read `TN03D9766` | **9** (frames 210, 525, 550, 690, 845, 1085, 1120, 1165, 1320) |
| m-model failed, s-model correctly read `TN03D9766` | 5 (frames 100, 115, 490, 725, 730) |

Other plates where the m-model's reading is visibly closer to correct:

| Plate | s-model reading | m-model reading |
|---|---|---|
| `TN10BL7364` | `NL03L736` / `TN1OBL7364` (near-miss) | `TN10BL7364` (frames 30, 40 — clean) |
| `TN10BY5351` | `TN1OBY5351` (I/1 confusion) | `TN10BY5351` (frames 440, 445 — clean) |
| `TN04AY8104` | `NL4AY8104` (state-code misread) | `TN04AY8104` (frames 905, 920 — clean) |
| `TN05DB4398` | `TN0SOB4398` (stuck on this misread) | `TN05DB4398` (frames 1190, 1195 — closer) |

The full 48-row side-by-side detail is reproducible via:
```bash
python compare_frames.py new_results/plate_tracking_5fps/frames.csv new_results/model_m_5fps/frames.csv
```

The same comparison at every-frame rate (225 mismatches/1299 matched) is
reproducible via:
```bash
python compare_frames.py new_results/plate_tracking_everyframe/frames.csv new_results/model_m_everyframe/frames.csv
```

## 5. Interpretation

The larger model appears to produce **tighter, more precise plate bounding
boxes**, which feeds a cleaner crop into PaddleOCR — this is a detection
quality improvement, not an OCR change (the OCR engine and validator are
byte-identical between the two runs). It also fires on fewer marginal boxes
overall (§3) — trading raw detection coverage/recall for precision. The
accuracy gain is real but comes at a **~2.1x per-frame cost**.

## 6. Recommendation

- **On the current CPU-only deployment**: adopting the m-model would make
  the existing frame-drop problem worse (`PRODUCTION_VS_OFFLINE_FRAME_REPORT.md`
  already shows CPU production delivering only ~40% of the nominal 5fps
  target with the s-model; doubling detection time would push this lower
  still). Not recommended until GPU deployment.
- **On GPU** (where detection is already ~18x faster per
  `ANPR_COLAB_VS_CPU_REPORT.md`, and frame delivery hit ~99.6% of nominal
  per `PRODUCTION_VS_OFFLINE_FRAME_REPORT.md` §4): the m-model's 2.1x cost
  is easily absorbed, and its accuracy improvement is a clear win. Worth
  switching to once GPU deployment happens.

## Raw data

- `new_results/plate_tracking_5fps/results.csv` / `frames.csv` — s-model, 5fps.
- `new_results/model_m_5fps/results.csv` / `frames.csv` — m-model, 5fps.
- `new_results/plate_tracking_everyframe/results.csv` / `frames.csv` — s-model, every frame.
- `new_results/model_m_everyframe/results.csv` / `frames.csv` — m-model, every frame.
- `backend/compare_frames.py` — updated to match by nearest bbox-center
  (not exact coordinates) so cross-model comparisons work; this is also
  what generates the mismatch-detail table used in §3.
