# ANPR Pipeline — Flow & Reference Report

Plain reference document: flow diagram, file-by-file summary, config reference, and measured inference timing. No vehicle-detection stage — plates are detected and tracked directly on the full frame.

---

## 1. Flow Diagram

```
                         Source video file
                                 |
                 +---------------+---------------+
                 |                               |
          DETECTION PATH                    VIDEO PATH
          (Python process)                  (ffmpeg process)
                 |                               |
          VideoReader                        ffmpeg reads
     (reads full-res frames,                 the same file
      decimated to PROCESSING_FPS)           independently
                 |                               |
          FrameQueue                    Downscale to 960px wide
     (bounded, drops oldest                 + encode (libx264,
      frame under load)                       ultrafast preset)
                 |                               |
     Resize FULL FRAME to 640x640         Write 2-second .ts
    (FIRST resize — for YOLO to find       segments + manifest
     WHERE the plate is, not to read it)
                 |                               |
         Plate Detection (YOLOv8s)       Serve at /hls/*
                 |                        (always no-store)
      Map plate box back to
        ORIGINAL frame coords
                 |
      Crop plate from ORIGINAL
        (full-resolution) frame
                 |
      Assign track_id (velocity-
      predicting nearest-center
            matcher)
                 |
        OcrGate (throttle check:
      first sighting, or improved
       quality / cooldown elapsed)
                 |
        OcrWorker (background
       thread, bounded job queue)
                 |
      Resize crop to 64px height
      (SECOND resize — the small
       plate crop, not the frame;
      standardizes text size for OCR)
                 |
             PaddleOCR
                 |
      plate_validator.normalize_plate()
      (format check + state-code fix)
                 |
            SegmentStore
     (boxes by HLS segment index,
      OCR attempts list — every
      attempt, not just successes)
                 |                               |
                 +---------------+---------------+
                                 |
                     Browser: hls.js
        (plays the HLS stream; fires FRAG_CHANGED
                on every segment boundary)
                                 |
              GET /api/detections/{id}?segment=N
              GET /api/detections/{id}/plates
                       (polled every 2s)
                                 |
                 +---------------+---------------+
                 |                               |
          CanvasOverlay                   DetectionTable
     (draws boxes on the video,        (lists every OCR attempt:
      scaled to the REAL source        accepted / rejected / no
      resolution, not the HLS          text — with crop image
      preview's downscaled one)         and confidence)
```

Both the detection path and the video path are gated by the same start signal: nothing runs until the frontend's Play button calls `POST /api/cameras/{id}/start`. For a video-file source, one pass through this diagram is one "cycle" — the cycle ends when the file is exhausted, and a later Play click starts a fresh one (no auto-loop).

---

## 2. File-by-File Summary

### Backend

| File | What it does |
|---|---|
| `app/main.py` | Process entry point; runs one detection cycle per Play click and logs per-segment throughput. |
| `app/config.py` | All tunable settings in one place (models, thresholds, FPS, HLS, DB). |
| `app/camera/frame_source.py` | Defines the `Frame` data shape and the abstract frame-source interface. |
| `app/camera/video_reader.py` | Reads a video file frame-by-frame, decimated to `PROCESSING_FPS`, paced like a live feed. |
| `app/camera/rtsp_reader.py` | Reads a live RTSP stream on its own thread, with auto-reconnect. |
| `app/services/frame_queue.py` | Bounded queue between the reader thread and the pipeline; drops the oldest frame under load. |
| `app/services/pipeline.py` | Wires plate detection, tracking, OCR submission, and result emission for every frame. |
| `app/tracking/plate_tracker.py` | Detects plates on the full frame and assigns track IDs with a custom velocity-predicting matcher. |
| `app/detection/plate_detector.py` | Loads and exposes the YOLO plate-detection model. |
| `app/services/ocr_gate.py` | Per-track throttle deciding whether a plate crop is worth OCR-ing again. |
| `app/services/ocr_worker.py` | Runs OCR on a background thread with its own bounded job queue. |
| `app/ocr/plate_reader.py` | Preprocesses a plate crop and runs PaddleOCR, concatenating multi-line reads. |
| `app/ocr/plate_validator.py` | Validates plate format against real Indian state codes and corrects common OCR misreads. |
| `app/services/segment_store.py` | Per-cycle in-memory store: detections by HLS segment, plus the OCR-attempts list. |
| `app/services/hls_service.py` | Spawns ffmpeg to transcode the source into an HLS stream for the browser. |
| `app/services/pipeline_controller.py` | Gate that holds the detection loop until the frontend clicks Play. |
| `app/services/api_server.py` | Runs the FastAPI app on a background thread. |
| `app/api/cameras.py` | REST endpoints: camera info, and starting a camera. |
| `app/api/detections.py` | REST endpoints: per-segment detections, and the OCR-attempts list. |
| `app/websocket/server.py` | FastAPI app instance, CORS, no-store caching middleware, `/hls` static mount. |
| `app/database/db_sink.py` | Optional PostgreSQL upsert of the best reading per (camera, track). |
| `app/database/schema.py` | Database table definition. |
| `app/services/result_sink.py` | `DetectionResult` data shape and sink interfaces (print / DB / composite). |
| `app/utils/logger.py` | Shared logger configuration. |

### Frontend

| File | What it does |
|---|---|
| `src/pages/CameraView.jsx` | Top-level page: fetches camera info, handles Play, polls OCR results. |
| `src/components/LivePlayer.jsx` | Wraps hls.js and syncs detection fetches to the video's current HLS segment. |
| `src/components/CanvasOverlay.jsx` | Draws detection boxes on a canvas over the video, scaled to the real source resolution. |
| `src/components/DetectionTable.jsx` | Lists every OCR attempt with its crop image and status. |
| `src/services/api.js` | Thin axios wrapper for the backend REST endpoints. |

---

## 3. Configuration Reference

| Key | Current value | One-line meaning |
|---|---|---|
| `PLATE_MODEL_PATH` | `number_plate_yolov8s_v2.pt` | Which YOLO weight detects plates. |
| `PLATE_CONF_THRESHOLD` | 0.25 | Minimum detector confidence to count as a plate. |
| `PLATE_DETECTION_IMGSZ` | 640 | Resolution YOLO resizes to internally before detecting. |
| `OCR_LANG` | en | PaddleOCR language model. |
| `OCR_CONF_THRESHOLD` | 0.4 | Confidence a reading must reach before OcrGate stops retrying. |
| `OCR_MAX_ATTEMPTS_PER_TRACK` | 5 | Max OCR retries per tracked plate. |
| `OCR_COOLDOWN_FRAMES` | 10 | Minimum processed-frame gap between retries. |
| `FRAME_QUEUE_MAXSIZE` | 5 | How many frames can buffer before the oldest is dropped. |
| `PROCESSING_FPS` | 5 | Target frame rate detection runs at (decimated from source). |
| `RTSP_RECONNECT_INITIAL_DELAY` | 1.0s | Delay before the first RTSP reconnect attempt. |
| `RTSP_RECONNECT_MAX_DELAY` | 30.0s | Cap on RTSP reconnect backoff. |
| `DATABASE_URL` | unset by default | Optional Postgres connection string; DB persistence skipped if unset. |
| `HLS_OUTPUT_DIR` | `backend/hls_output` | Where ffmpeg writes HLS segments/manifest. |
| `FFMPEG_BINARY` | `ffmpeg` | The ffmpeg executable to invoke. |
| `HLS_SEGMENT_SECONDS` | 2 | Target duration of each HLS segment — also the detection-sync granularity. |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed frontend origin(s) for the API. |
| `PUBLIC_HOST` / `PUBLIC_PORT` | `localhost` / 8765 | Host/port the browser uses to reach the API. |

---

## 4. Inference Timing — Measured Calculations

All numbers measured directly on this CPU-only environment (plate-detection numbers from real per-segment backend logs; OCR numbers from timing `PlateReader.read()` on 15 real plate crops, PaddleOCR pre-warmed).

### 4.1 Per-stage cost

| Stage | Model / engine | Typical | Observed range | Runs on |
|---|---|---|---|---|
| Plate detection | YOLOv8s, 640 imgsz, full frame | 0.38 s | 0.35 s – 1.4 s | Main pipeline thread, every processed frame |
| OCR | PaddleOCR | 0.12 s | 0.066 s – 0.4 s | Background thread, per OCR-eligible sighting |
| Model warmup (one-time) | YOLOv8s + PaddleOCR | 0.9 s | 0.70 s – 1.08 s | Once, at process start |

### 4.2 Derived totals

| Metric | Value | How it's calculated |
|---|---|---|
| Max sustainable frame rate | ~2.6 fps | 1 ÷ 0.38 s (plate-detection cost alone) |
| Actual sustained frame rate | ~2.0 – 2.5 fps | Measured from per-segment logs |
| Configured decimation target | 5 fps | `PROCESSING_FPS` |
| Total time, one plate read (detect + first OCR) | ~0.50 s | 0.38 s + 0.12 s |
| Additional cost per OCR retry | +0.07 – 0.4 s | One more OCR call only (detection not repeated) |
| Worst case, all 5 OCR attempts used | ~0.98 s | 0.38 s + (5 × 0.12 s) |

**Note on this table:** an earlier internal assumption stated OCR at "~1–2 s per call" — that was never directly measured and turned out to be wrong. Plate detection, not OCR, is the larger cost per frame; OCR still runs on a background thread because even ~0.12 s inline would stall frame consumption on every sighting, and multiple plates per frame compound that.

### 4.3 GPU Comparison (Google Colab, NVIDIA T4)

Same plate-detection model (`number_plate_yolov8s_v2.pt`) and settings (`conf=0.25`, `imgsz=640`) as Section 4.1, run against **every frame** of the same source video (no sampling/decimation) — CPU column reused from Section 4.1, GPU column measured directly on a Colab T4 instance (`colab_gpu_benchmark.ipynb`).

| | avg | min | max | max sustainable fps |
|---|---|---|---|---|
| CPU (this machine) | 0.380 s | 0.350 s | 1.400 s | 2.6 |
| CUDA (T4, Colab) | 0.016 s | 0.011 s | 0.037 s | 62.1 |

**Double detection (vehicle detection → crop → plate detection) on the same T4**, for comparison against the project's original two-stage architecture (see Section 1/6) — measured on all 1,328 frames of the same source video, `yolov8n.pt` for vehicle detection + the same plate model on each vehicle crop:

| | avg | min | max | max sustainable fps | plate coverage |
|---|---|---|---|---|---|
| Single-stage (direct, full frame) | 0.016 s | 0.011 s | 0.037 s | 62.1 | — |
| Double detection (vehicle → crop → plate) | 0.053 s | 0.009 s | 0.193 s | 19.0 | 93.4% (1,240/1,328 frames) |

Of the double-detection cost, vehicle detection itself averages 0.016 s/frame and the summed plate-detection calls (one per vehicle in frame) average 0.036 s/frame. **Double detection is ~3.3× slower per frame than single-stage** on the same GPU — the extra vehicle-detection pass plus one plate-detection call per vehicle (not per frame) adds up, especially on frames with multiple vehicles. This is on top of the tracking-continuity tradeoffs (Section 1) that motivated dropping the vehicle-detection stage in the first place.

**Speedup: 23.6× faster on GPU (avg case).**

**What this changes:** at 62 fps max sustainable rate, a GPU eliminates the Section 4.2/5 shortfall entirely — `PROCESSING_FPS=5` (or even a much higher target) would be met with large headroom to spare, since GPU inference alone is ~12× faster than the current 5 fps budget requires. Detection would stop being the bottleneck; OCR (still CPU-bound via PaddleOCR unless also moved to GPU) would become the larger relative cost instead.

---

## 5. Segment Duration & Frame Coverage

Each HLS segment is `HLS_SEGMENT_SECONDS = 2` seconds long — this is both the video chunk size ffmpeg writes and the sync granularity the frontend fetches detections at (`video_time // HLS_SEGMENT_SECONDS` decides which segment a frame's detections belong to).

How many *processed* frames land inside one 2-second segment depends on how many frames the pipeline actually gets through in that window — which is capped by the plate-detection cost (Section 4.1), not by the segment boundary itself.

| Quantity | Value | How it's calculated |
|---|---|---|
| Segment duration | 2 s | `HLS_SEGMENT_SECONDS` |
| Target frames per segment | 10 frames | `PROCESSING_FPS` (5) × segment duration (2 s) |
| Actual frames per segment (measured) | 2 – 8 frames, typically 4 – 5 | Directly counted in backend logs (`Segment N done: X frames`) |
| Actual frames per segment (formula) | ~4 – 5 frames | Actual sustained rate (2.0–2.5 fps, Section 4.2) × 2 s |
| Shortfall vs. target | ~50 – 60% fewer frames than the 10-frame target | Plate detection (0.38 s/frame) can't keep up with the 5 fps (0.2 s/frame) budget |

**What this means in practice:** a segment doesn't get one detection per rendered video frame — it gets whatever the pipeline managed to process in that 2-second window, usually 4–5 sampled frames rather than a continuous 10. This is why a fast-moving plate can appear in one segment's detections and be missing from the next, even though it was visible the whole time on screen — the video plays at full frame rate, but detection only sampled a few points within it.

---

## 6. Known Limits

- CPU is the hard ceiling: ~2–2.5 fps sustained against a 5 fps target. Only faster hardware (GPU) or a smaller/faster detection model would raise this.
- OCR runs one call at a time on a single background thread; a 12-slot queue absorbs short bursts but sustained multi-plate frames still evict jobs (logged, not silent).
- Track fragmentation is reduced (velocity-prediction matching) but not eliminated — erratic motion across a large decimated frame gap can still split one physical plate into multiple track IDs.
- OCR crop images are in-memory only; they do not survive a process restart or a fresh cycle, and are not yet written to the database.
- One camera per process — no shared inference service or multi-camera batching yet.
