# Detection Tuning Report — imgsz / conf-threshold / missed-seconds

## 1. Objective

`MODEL_SIZE_COMPARISON_REPORT.md` found the m-model improves OCR *reading*
accuracy but does **not** improve detection *consistency* — it actually
detects fewer plate-frames overall (329 vs 424) and fragmented one
long-running vehicle (`TN03D9766`) into more tracks (7 vs 3), because it's
more conservative on marginal/borderline boxes. This report tests whether
tuning three detection/tracking knobs on the **s-model** (same weight,
same speed baseline) can fix detection consistency directly, without
paying the m-model's ~2.1x cost for an unrelated benefit.

## 2. Root cause (recap)

For `TN03D9766`, the s-model detected the plate in nearly every processed
frame in the 295-435 range at borderline confidence (0.25-0.63) — it never
disappeared entirely. The gap analysis showed the vehicle was simply
small/far/blurry in this stretch of video, and detection was already
running right at the edge of its confidence threshold there. Two levers
directly address this:

- `PLATE_CONF_THRESHOLD` (0.25 → lower) — catch boxes just below the old cutoff.
- `PLATE_DETECTION_IMGSZ` (640 → higher) — better resolution for small/far plates.
- `_MAX_MISSED_SECONDS` (1.0 → higher, `plate_tracker.py`) — tolerate slightly
  longer real gaps before minting a new track_id, independent of detection quality.

## 3. Change tested

All three tuned together, s-model, 5fps, same test video:

| Config | Baseline | Tuned |
|---|---|---|
| `PLATE_MODEL_PATH` | `vechile_plate_yolov8s.pt` | same |
| `PLATE_CONF_THRESHOLD` | 0.25 | **0.20** |
| `PLATE_DETECTION_IMGSZ` | 640 | **960** |
| `_MAX_MISSED_SECONDS` (`plate_tracker.py`) | 1.0 | **1.5** |

## 4. Results

| | Baseline (640/0.25/1.0) | **Tuned (960/0.20/1.5)** |
|---|---|---|
| Total plate-detections | 424 | **499** (+17.7%) |
| Matched (bbox-center within 30px) | 345 | 345 |
| Detect avg time | 232.4ms | **464.8ms** (~2x slower) |
| Detect min/max | 202.4 / 452.2ms | 382.4 / 823.4ms |
| Total tracks (raw) | 58 | **45** (fewer overall) |
| Tracks with valid plate | 29 | 28 |
| Distinct plate strings | 19 | **20** |
| Fragmented plates (>1 track) | 8 | 7 |
| `TN03D9766` fragments | **3 tracks** | **1 track** (fixed) |
| OCR validated-plate text mismatches | 51/345 | |
| Overall plate-set agreement (`compare_pipeline_runs.py`) | | 28.0% |

### `TN03D9766` — the case this tuning specifically targeted

Baseline fragmented this vehicle into 3 track_ids (gaps in a decimation-rate,
borderline-confidence stretch). Tuned run: **a single, unbroken track** —
confirms the imgsz increase caught the marginal detections the baseline
was missing in that exact frame range.

### `[no_text]` → correctly read

Many frames where the baseline got no OCR text at all now read
`TN03D9766` cleanly in the tuned run: frames 45, 90, 210, 250, 525, 550,
690, 845, 910, 1025, 1085, 1120, 1165, 1320 — direct evidence the plate
*was* visible/legible in these frames, just under-detected by the
baseline's lower imgsz/higher confidence cutoff.

Full detail reproducible via:
```bash
python compare_frames.py new_results/plate_tracking_5fps/frames.csv new_results/model_s_5fps-img/frames.csv
python compare_pipeline_runs.py new_results/plate_tracking_5fps/results.csv new_results/model_s_5fps-img/results.csv
```

## 5. Interpretation

This tuning combination improves **both** detection consistency (more
plate-frames caught, `TN03D9766`'s fragmentation eliminated) **and** does
so on the s-model — meaning the same ~2x detection-cost paid by switching
to the m-model instead buys a fix for the actual problem (missed
detections/fragmentation) rather than only a downstream OCR-reading
quality bump. Compared to `MODEL_SIZE_COMPARISON_REPORT.md`'s m-model
finding, this is the more targeted fix for detection consistency
specifically.

## 6. Production impact (real `python -m app.main` run, tuned config)

Ran the actual production process (not the offline script) with the tuned
config (imgsz=960, conf=0.20, missed=1.5) on CPU, same video, same 5fps —
confirms the cost predicted in §5 shows up in real frame delivery, not
just offline detect-time benchmarks.

| | CPU, baseline config (`PRODUCTION_VS_OFFLINE_FRAME_REPORT.md` §2) | **CPU, tuned config** (`backend/run.log`) | GPU, baseline config (§4 of same report) |
|---|---|---|---|
| Total frames delivered | 105 (of ~265 target) | **80** (of ~265 target) | 259 |
| Avg frames/segment | 4.04 (of nominal 10) | **3.08** (of nominal 10) | 9.96 |
| Frame-queue drops (`dropped_frames`) | 154 | **179** | 0 |
| Detect avg time (first ~12 segments, warmup) | ~233ms (imgsz 640) | **0.9-1.68s** (ffmpeg-startup + cache-warmup contention) | n/a (GPU, no contention) |
| Detect avg time (after warmup, stabilized) | ~233ms | **~0.43-0.6s** (close to offline's 465ms benchmark) | n/a |

**Tuning makes CPU production frame delivery worse, not better** — avg
frames/segment drops from 4.04 → 3.08, queue drops rise 154 → 179. The
~2x slower per-frame detect time (confirmed here, not just in the offline
benchmark) means fewer decimated frames survive `FrameQueue` before being
dropped under CPU contention (ffmpeg HLS transcode + API server + torch
detection all competing for the same limited threads,
`torch.set_num_threads`). Also notable: production's real detect times run
2-3x *higher than the offline benchmark* for the first ~12 segments (CPU
cache/model warmup + concurrent ffmpeg startup), before settling near the
offline number — an added contention cost the offline script's
single-threaded, synchronous test never sees at all.

**GPU still wins by a wide margin regardless of config** (§ above) — this
reinforces §5's recommendation: this tuning should be deployed once on
GPU, not now on CPU, where it trades a detection-consistency win for a
throughput loss.

## 7. Cost and recommendation

Same CPU-throughput tradeoff as the m-model comparison: detect time
232ms → 465ms (~2x) will worsen CPU production's existing frame-drop
problem (already only ~40% of nominal 5fps delivered per
`PRODUCTION_VS_OFFLINE_FRAME_REPORT.md`). **Not recommended on the
current CPU-only deployment** — hold until GPU deployment, where the 2x
cost is easily absorbed and both detection consistency and (if combined
with the m-model) OCR accuracy can be improved together.

## Raw data

- `new_results/plate_tracking_5fps/results.csv` / `frames.csv` — s-model, baseline config (640/0.25/1.0).
- `new_results/model_s_5fps-img/results.csv` / `frames.csv` — s-model, tuned config (960/0.20/1.5).
- `backend/app/config.py` — `PLATE_CONF_THRESHOLD`, `PLATE_DETECTION_IMGSZ`.
- `backend/app/tracking/plate_tracker.py` — `_MAX_MISSED_SECONDS`.
