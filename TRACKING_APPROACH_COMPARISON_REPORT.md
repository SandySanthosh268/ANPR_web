# Plate-Tracking vs Vehicle-Tracking (ByteTrack) — Comparison Report

## 1. Objective

Compare two ways of assigning a stable identity to a detected plate, across the
same test video (`sample_video.mp4`), at two frame-processing rates:

1. **Plate-tracking** (this project's original, custom approach) — tracks the
   *plate box* directly, using a nearest-center matcher with velocity
   prediction (`app/tracking/plate_tracker.py`, `_Candidate`/`_match`).
2. **Vehicle-tracking** (Ultralytics ByteTrack) — tracks the *vehicle box*
   (car/auto/motorcycle/etc.) via `model.track()`, and a plate borrows its
   identity from whichever tracked vehicle box contains it.

Both approaches use the exact same detection model, OCR engine, and
validator — only the *identity-assignment* logic differs.

## 2. Result Summary

| Approach | Frame rate | Tracks | Tracks with a valid plate | Plate hit rate |
|---|---|---|---|---|
| Plate-tracking | 5fps (production rate) | 58 | **29** | **50%** |
| Plate-tracking | every frame | 45 | 19 | 42% |
| Vehicle-tracking (ByteTrack) | 5fps (production rate) | 5 | **2** | **40%†** |
| Vehicle-tracking (ByteTrack) | every frame | 20 | 16 | 80% |

† Only 5 tracks total were ever formed at 5fps — most vehicles that
crossed the scene were never tracked at all, so this hit-rate percentage is
misleading on its own; the real story is in the tracks column (see §3).

**Bottom line: at production's actual frame rate (5fps decimation, forced by
CPU throughput — see `ANPR_COLAB_VS_CPU_REPORT.md`), plate-tracking finds
~14x more valid plates than vehicle-tracking (29 vs 2).** Vehicle-tracking
only wins when every frame is processed, which the current CPU-only
deployment cannot sustain.

## 3. Why Vehicle-Tracking Fails at 5fps (and Plate-Tracking Doesn't)

ByteTrack links a vehicle box in frame N to a vehicle box in frame N+1 by
**IoU (Intersection over Union)** — how much the two boxes overlap. This
depends on how far the vehicle moved *relative to its own box size* between
the two frames the tracker actually sees, not on the vehicle's absolute
size.

- **Every-frame processing**: consecutive tracked frames are ~0.04s apart
  (native 24.89fps). A vehicle moves very little pixel-wise between them, so
  IoU stays high and ByteTrack confirms the same track_id reliably (80% hit
  rate, 20 well-formed tracks).
- **5fps decimation**: consecutive tracked frames are ~0.2s apart — 5x the
  gap. A moving vehicle's box shifts far enough between these two sampled
  frames that IoU collapses toward zero. ByteTrack then either assigns a
  brand-new track_id (losing continuity) or drops the vehicle entirely.
  Only the one **near-stationary** vehicle in the test video (a parked auto)
  kept a stable ByteTrack id at 5fps, because its box barely moved between
  samples regardless of the gap — everything else (moving cars, buses,
  motorcycles) lost tracking almost immediately. This produced only **5
  tracks total** at 5fps, versus 20 at every-frame.

This is a **known, previously-documented limitation of ByteTrack for this
pipeline** — the original `PlateTracker` docstring (before this comparison)
already recorded that ByteTrack confirmed a plate-box track for only ~7% of
processed frames at production's decimation rate, which is why the custom
matcher was built in the first place. This test confirms the same
IoU-vs-decimation problem also affects the *larger* vehicle boxes once the
gap between processed frames grows large enough or the vehicle moves fast
enough — a bigger box makes detection easier, but does not make
frame-to-frame **matching** immune to a large motion gap.

**Plate-tracking avoids this entirely** by not relying on IoU/overlap at
all. It predicts where a plate *should* be now (last known position +
estimated velocity × elapsed time) and matches the nearest actual detection
to that prediction within a generous pixel radius (`_MATCH_RADIUS_PX=150`).
This tolerates decimation-sized motion gaps far better, since it reasons
about *expected position* rather than requiring the boxes to literally
overlap.

## 4. Plate-Tracking Is Not Perfect Either — Known Residual Issue

Plate-tracking still occasionally fragments the *same physical vehicle*
into multiple track_ids — e.g. in the 5fps test:

```
TN03D9766  -> track 3  (frames 10-15)
           -> track 8  (frames 35-315)
           -> track 15 (frames 375-1325)
```

This happens when the plate detector fails to detect the plate for more
than `_MAX_MISSED_SECONDS` (1.0s of real/video time) in a row — e.g. due to
motion blur, occlusion by another vehicle, or a confidence dip below
`PLATE_CONF_THRESHOLD`. When the plate reappears after that gap, the
tracker treats it as a new sighting and assigns a fresh id. This is a
separate, smaller-magnitude issue from the ByteTrack/IoU problem above —
it fragments an otherwise-successfully-read plate into 2-3 track_ids
instead of losing the reading entirely, and mitigations (larger
`_MAX_MISSED_SECONDS`, or de-duplicating by validated plate text
downstream) are being considered separately.

## 5. Attempted Fix: Tuning ByteTrack's `match_thresh`

ByteTrack's own config (`bytetrack.yaml`) exposes `match_thresh: 0.8` — the
IoU threshold used to link a box across frames. Since the failure mode above
is specifically "IoU collapses due to the decimation gap," lowering this
threshold was tested directly: a custom config
(`app/tracking/decimated_bytetrack.yaml`) with `match_thresh: 0.3` instead
of `0.8`, everything else unchanged.

| | Stock (`match_thresh=0.8`) @ 5fps | Tuned (`match_thresh=0.3`) @ 5fps |
|---|---|---|
| Tracks | 5 | **11** |
| Valid plates | 2 | **7** |

This is a real, measurable improvement (3.5x more valid plates) — worth
keeping if pursuing this approach further. **However, inspecting which
tracks actually improved shows most of the gain came from this pipeline's
own fallback matcher (the nearest-center matcher, disjoint id range
`1,000,000+`), not from ByteTrack itself successfully tracking more
vehicles.** Only one vehicle (the same near-stationary auto that always
survived) kept a genuine ByteTrack id across the whole video; every other
successful plate reading came through the fallback path. Lowering
`match_thresh` still leaves vehicle-tracking (7/11 valid plates) well short
of plate-tracking's 29/58 at the same 5fps rate — a single threshold tune
does not fully close the gap for genuinely fast-moving vehicles at this
decimation rate.

## 6. Recommendation

- **Keep plate-tracking (the custom nearest-center/velocity matcher) as the
  production tracking approach**, given the current CPU-only deployment
  processes video at 5fps decimation.
- **Revisit vehicle-tracking (ByteTrack) if/when the pipeline moves to
  GPU inference** (see `ANPR_COLAB_VS_CPU_REPORT.md` — detection is ~18x
  faster on a T4 GPU), since every-frame processing would then be
  affordable and ByteTrack's 80% hit rate at that rate clearly beats
  plate-tracking's own every-frame result (42%). If GPU deployment isn't
  imminent but vehicle-tracking is still preferred for other reasons, keep
  the tuned `match_thresh=0.3` config (`decimated_bytetrack.yaml`) as a
  partial mitigation rather than the stock default.

## Raw data

- `new_results/local_results_5fps_nogate.csv` / `_frames.csv` — plate-tracking, 5fps, OCR gate removed.
- `new_results/local_results_everyframe.csv` / `_frames.csv` — plate-tracking, every frame, OCR gate removed.
- `new_results/vehicle_tracking_5fps/results.csv` / `frames.csv` — vehicle-tracking (ByteTrack), stock `match_thresh=0.8`, 5fps, OCR gate removed.
- `new_results/vehicle_tracking_5fps_tuned/results.csv` / `frames.csv` — vehicle-tracking (ByteTrack), tuned `match_thresh=0.3`, 5fps, OCR gate removed.
- `new_results/vehicle_tracking_everyframe/results.csv` / `frames.csv` — vehicle-tracking (ByteTrack), every frame, OCR gate removed.
