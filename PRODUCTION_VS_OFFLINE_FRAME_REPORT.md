# Production (`main.py`) vs Offline Script — Frame Delivery & Accuracy Report

## 1. Objective

Two related questions, both using the same test video (`sample_video.mp4`,
1328 source frames, native ~24.89fps, 53.22s):

1. At the **same nominal decimation rate (5fps, `PROCESSING_FPS=5`)**, does
   the real production process (`python -m app.main`, with concurrent HLS
   transcoding + API server) actually process/deliver as many frames as the
   clean, single-threaded offline script (`run_pipeline_offline.py`)?
2. Separately, how do **every-frame** vs **every-5th-frame** processing
   compare for plate-tracking (already covered in
   `TRACKING_APPROACH_COMPARISON_REPORT.md`, restated here for the
   frame-count angle specifically)?

Both production and the offline script use the exact same `PlateTracker`,
`PlateReader`, and validator — the only difference in §2 is that production
also runs concurrent ffmpeg (HLS) and the API server, competing for the same
CPU.

## 2. CPU 5fps: Offline Script vs Production

| | Offline script (`run_pipeline_offline.py --target-fps 5`) | **Production (`python -m app.main`, `PROCESSING_FPS=5`)** |
|---|---|---|
| Source frames | 1328 | 1328 |
| Decimation stride | 5 | 5 |
| Nominal target frames (1328 / stride) | 265 | 265 |
| **Frames actually processed/sent** | **265** (100% of target) | **105** (~40% of target) |
| Frames missed vs. nominal target | 0 | **160** |
| Frame-queue drops (`dropped_frames`) | n/a (no queue, synchronous) | 154 |
| Avg frames delivered per 2s HLS segment | n/a | 4.04 (nominal target: 10) |
| Total processing time | 146.5s | ~76s wall-clock (one Play-to-finish cycle) |
| `vehicle_count` / tracks | 58 (raw tracker count) | 40 (`PlateIdentity`-corrected) |
| Distinct validated plates | 19 | 18 |
| **Plates found in both** | **10** | **10** |

### Why production misses ~60% of the nominal target

The offline script is single-threaded and synchronous — every decimated
frame it reads gets processed immediately, no exceptions. Production runs
the same detection/tracking/OCR work **concurrently with ffmpeg HLS
transcoding and the FastAPI server**, all competing for the same limited
CPU threads (`torch.set_num_threads`, see `app/config.py`). When detection
can't keep up with the incoming (already-decimated) frame rate, `FrameQueue`
drops the oldest queued frame to avoid unbounded lag (`app/services/frame_queue.py`)
— this is a *second* layer of frame loss on top of the 5fps decimation
itself, and it's what the 154 `dropped_frames` / 160-frame shortfall
reflects.

### Plates found — overlap and differences

**Found in both (10):** the well-tracked, longer-duration vehicles read
identically either way — same underlying detection/tracking/validator code.

**Found only in one or the other:** almost entirely near-miss OCR variants
of the *same* physical vehicles (e.g. production's `TN05OB4398` /
`TN02ATA104` vs offline's `TN0SOB4398` / `NL4AY8104` — same vehicle,
different attempt's reading accepted) rather than genuinely different
vehicles. A few vehicles visible only briefly were caught in one run and
not the other, consistent with production's extra frame loss shifting
*which* exact frames got a shot at OCR.

## 3. Every-Frame vs Every-5th-Frame (Plate-Tracking, Offline Script)

| | Every frame (`--target-fps 0`) | 5th frame (`--target-fps 5`, production's rate) |
|---|---|---|
| Source frames | 1328 | 1328 |
| Frames processed | **1328** (100%) | **265** (20%) |
| Total processing time | 739.6s (~12.3 min) | 146.5s (~2.4 min) |
| Processed-fps (throughput) | 1.8 | 1.8 |
| Tracks (raw, no `PlateIdentity` correction) | 45 | 58 |
| **Tracks with a valid plate** (includes fragmentation duplicates — see below) | 19 | **29** |
| **Distinct plate strings** (fragmentation duplicates collapsed) | 17 | **19** |

Two different counts, easy to conflate: "tracks with a valid plate" counts
every track row that has *some* accepted reading, which still double-counts
a plate PlateTracker fragmented into 2-3 track_ids (e.g. 5fps's `TN03D9766`
across 3 tracks, `TN23AP4115` across 3 — see
`TRACKING_APPROACH_COMPARISON_REPORT.md` §4). "Distinct plate strings"
collapses those duplicates, so it's the more meaningful "how many actual
vehicles were successfully read" number — **17 at every-frame vs 19 at
5fps**, a much smaller gap than the raw 19-vs-29 track count suggests. The
Appendix's plate lists in this report use the distinct-string counting.

Processing every frame gives the tracker more chances to observe each
vehicle, and — matching this — produces **fewer, less-fragmented tracks**
(45 vs 58) since a plate rarely goes undetected long enough to exceed
`_MAX_MISSED_SECONDS` between *consecutive* frames. It still finds slightly
**fewer** distinct plates overall (17 vs 19), consistent with
`OCR_COOLDOWN_FRAMES`/attempt budgeting interacting differently at each
decimation rate (see the earlier "why does 10fps give fewer plates than
5fps" discussion in this session) — which exact frame a track's OCR
attempts land on varies with the sampling phase, and that can matter more
than raw frame count.

## 4. Production on GPU (Colab T4) — Same 5fps Config

Tested via `colab_main_pipeline_test.ipynb`, which runs the *actual*
`python -m app.main` process (FastAPI + HLS + frame queue, not just the
detect/track/OCR loop) on a Colab T4 GPU, same video, same `PROCESSING_FPS=5`.

| | CPU production | **GPU production (Colab T4)** |
|---|---|---|
| Avg frames delivered per 2s segment | 4.04 (of nominal 10) | **9.96** (of nominal 10) |
| Total frames delivered | 105 (of ~265 target) | **259** (of ~265 target) |
| Frame-queue drops (`dropped_frames`) | 154 | **0** |
| `vehicle_count` (`PlateIdentity`-corrected) | 40 | 46 |
| Distinct validated plates | 18 | **22** |

**GPU delivers essentially the full nominal frame rate** (9.96/10 per
segment, 0 queue drops) — confirming the CPU-contention bottleneck in §2/§3
is specifically a CPU-throughput problem, not a decimation-rate or
tracking-logic problem. Once detection is fast enough (GPU) to keep up with
the incoming decimated frame rate, `FrameQueue` never needs to drop
anything, and every decimated frame gets its full detection+OCR treatment.

This also explains the higher plate count on GPU (22 vs 18) — with
essentially zero frame loss, more vehicles get their full share of
processed frames and OCR attempts within the video's duration, not just
faster per-frame processing.

## 5. Configuration Note

This report required correcting `app/config.py`'s `PROCESSING_FPS`, which
had drifted to `10` during this session's testing (confirmed via
production's startup log: `processing stride=2` instead of the expected
`stride=5`) — it's been reset to **`5`** to match the offline "5fps"
baseline used throughout this project's other reports. If you intend to run
production at a different rate going forward, re-run this comparison at
that rate rather than assuming these numbers transfer.

## Appendix: Full Plate Lists

The actual distinct validated plate strings behind every count in this
report, for full transparency.

| Offline, 5fps (19) | Offline, every-frame (17) | CPU production, 5fps (18) | GPU production, 5fps (22) |
|---|---|---|---|
| MZ8J3333 | TN03D9766 | NL03L736 | MZ8J3333 |
| NL4AY8104 | TN04AY8104 | TN02ATA104 | TN02AT8104 |
| TN03D9766 | TN073736 | TN03CS1012 | TN02ATA104 |
| TN04AY8104 | TN09BT8635 | TN03D9766 | TN03CS1012 |
| TN073735 | TN09CS1812 | TN04AY8104 | TN03CS1812 |
| TN09BT8535 | TN09CX5000 | TN05OB4398 | TN03D9766 |
| TN09CS1812 | TN09Z174 | TN073735 | TN04AY8104 |
| TN09CX5000 | TN0SOB4398 | TN09CS1812 | TN05OB4398 |
| TN0SOB4398 | TN10BL7364 | TN10BI3974 | TN073735 |
| TN10BL7364 | TN10BY3974 | TN10BL7364 | TN09CS1812 |
| TN10BY3974 | TN10CC0947 | TN10BY3974 | TN09CX5000 |
| TN10CC0947 | TN1OBY5351 | TN10BY5351 | TN10BI3974 |
| TN1OBI3974 | TN1ZBJ9333 | TN10CC0947 | TN10BL7364 |
| TN1OBY5351 | TN23AP4115 | TN12BJ9333 | TN10BY3974 |
| TN1ZBJ9333 | TN37OF3523 | TN23AP4113 | TN10BY5351 |
| TN23AP4115 | TN41AR0082 | TN23AP4115 | TN10CC0947 |
| TN37OF3523 | TN976500 | TN41AR0082 | TN12BJ9333 |
| TN41AR0082 | | TN976500 | TN23AP4113 |
| TN976500 | | | TN23AP4115 |
| | | | TN37OF3523 |
| | | | TN41AR0082 |
| | | | TN976500 |

Note the same handful of physical vehicles keep reappearing across every
column under slightly different spellings (`TN10BL7364`/`TN10BY5351`,
`TN0SOB4398`/`TN05OB4398`, `TN23AP4113`/`TN23AP4115`, `TN12BJ9333` etc.) —
these are OCR near-miss variants of the same plate, not different vehicles;
see `TRACKING_APPROACH_COMPARISON_REPORT.md` §4 and the earlier
`plate_validator.py` digit/letter-confusion work for why a single vehicle
can still surface more than one accepted spelling across many OCR attempts.

## Raw data

- `new_results/plate_tracking_5fps/results.csv` / `frames.csv` — offline, 5fps.
- `new_results/plate_tracking_everyframe/results.csv` / `frames.csv` — offline, every frame.
- `backend/run.log` — CPU production run's full log (5fps, this report's §2 numbers).
- `files/colab_main_pipeline_test.ipynb` — the GPU production test notebook (§4).
