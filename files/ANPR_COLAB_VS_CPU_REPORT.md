# ANPR Pipeline: Colab (GPU) vs Local (CPU) — Test Report

## 1. Objective

Verify that running the ANPR pipeline on a GPU (Google Colab, Tesla T4) produces the **same
detection/OCR results** as the current CPU-only production setup, and measure the actual speed
difference. Two separate questions were tested:

1. **Correctness** — does GPU change *what* gets detected/read?
2. **Performance** — how much faster is GPU, per pipeline stage?

## 2. Test Setup

- **Same code, not a re-implementation**: both runs imported the actual production classes —
  `PlateDetector`, `PlateTracker`, `PlateReader`, `OcrGate`, `normalize_plate` — from a zipped copy
  of `backend/app/`, run via `run_pipeline_offline.py` locally and an equivalent Colab notebook
  (`colab_full_pipeline_test.ipynb`) on Drive.
- **Same video** (`sample_video.mp4`, 1920×1080, ~12.5fps, 53s) and **same model weight**
  (`vechile_plate_yolov8s.pt`) in both environments.
- **Local machine**: CPU-only (no GPU) — matches current production.
- **Colab**: Tesla T4 GPU runtime, plate detection (YOLO/torch) on GPU; OCR (PaddleOCR) forced to
  CPU (`use_gpu=False`) for the main comparison run, due to a cuDNN version clash — see §5.
- **Output**: per-track summary CSVs (`local_results.csv` / `colab_results.csv`) and per-frame
  detail CSVs with bbox coordinates + inference timing (`local_results_frames.csv` /
  `colab_frames.csv`), compared with `compare_pipeline_runs.py` (track-level) and

### 2.1 Local machine spec

| Spec | Value |
|---|---|
| CPU | 11th Gen Intel Core i5-1135G7 @ 2.40GHz |
| Cores / Threads | 4 cores / 8 threads |
| RAM (total) | 15GB (10GB available at test time, 4GB swap) |
| Disk (ROM) | 937GB total, 590GB free (`/`, NVMe SSD) |
| OS | Ubuntu 24.04 LTS, kernel 6.17 |
| GPU | None (CPU-only, matches production) |

### 2.2 Colab spec & config used

| Spec | Value |
|---|---|
| GPU | Tesla T4, 15GB GDDR6 VRAM, 2,560 CUDA cores, 320 Tensor cores |
| Runtime type | GPU (T4), free tier |
| CUDA version (torch) | 12.8 |
| cuDNN version (torch) | 9.19.0 |
| System RAM | ~12.7GB (Colab free-tier standard profile) |
| Disk (`/content`, ephemeral) | 113GB total, 66GB available (checked via `df -h /content`) |
| Session limit | 12 hours max, ~90 min idle timeout |
| Key installed packages | `ultralytics`, `opencv-python-headless`, `numpy==1.26.4` (pinned last, force-reinstalled), `paddlepaddle-gpu==2.6.2`, `paddleocr==2.7.3` |
| App code | `backend/app/` zipped and unzipped into `/content/backend`, imported directly — not re-typed |

Full breakdown of Colab's free/Pro/Pro+ tiers and hardware limits: see `COLAB_T4_SPECS.md`.
  `compare_frames.py` (frame-level, bbox-matched).

## 3. Accuracy Results — ✅ 100% match

| | Local (CPU) | Colab (GPU detect / CPU OCR) |
|---|---|---|
| Frames processed | 1,328 | 1,328 |
| Tracks seen | 115 | 115 |
| Distinct validated plates | 4 | 4 |
| Plate-set agreement | — | **100.0%** |

**Matched plates, confidence-for-confidence identical:**

| Plate | Local confidence | Colab confidence | Δ |
|---|---|---|---|
| TN03CS1012 | 0.688 | 0.688 | +0.000 |
| TN03D9766 | 0.990 | 0.990 | +0.000 |
| TN41AR0082 | 0.917 | 0.917 | +0.000 |
| TN976500 | 0.986 | 0.986 | +0.000 |

**Frame-level check** (`compare_frames.py`, matching by frame_id + plate bbox, not track_id since
IDs are assigned independently per run):

- **2,179 / 2,179 detected plate-frames matched** between local and Colab (same frame, same bbox).
- **0 mismatches** in plate-detection confidence, OCR status (accepted/rejected/no_text), or
  validated plate text, across every matched row.

**Conclusion: GPU does not change correctness.** Same model + same code + same video → identical
detections and identical OCR readings, CPU or GPU.

## 4. Performance Results

| Stage | Local (CPU) | Colab (GPU) | Speedup |
|---|---|---|---|
| Plate Detection | 318.5 ms (Min: 200.9 ms, Max: 635.6 ms) | 17.4 ms (Min: 11.5 ms, Max: 90.2 ms) | **~18× Faster** |
| OCR | 86.4 ms | 49.8 ms | ~1.7× Faster* |
| **Plate + OCR (one full process, when OCR runs)** | **416.6 ms** | **77.5 ms** | **~5.4× Faster** |
| **Grand total — all frames (whole video)** | **424.87 s (~7.1 min)** | **42.25 s** | **~10× Faster** |

\* OCR ran on CPU in both environments for this comparison (see §5) — the difference here reflects
Colab's CPU vs the local test machine's CPU, not a GPU effect.

**Detection is the expensive, per-frame stage** — at ~18x faster on GPU, this is the main real-world
win for throughput (more frames/sec sustainable, larger frame queues not needed, better keep-up
with live camera feeds).

### 4.1 Total inference time (whole video, ~1,328 frames / 53s)

Summed across every frame actually run through each stage (detection time is per-frame, deduplicated
across multiple plates found in the same frame; OCR time is summed over every OCR call made):

| Stage | Local (CPU) total | Colab (GPU detect / CPU OCR) total | Colab (GPU detect / GPU OCR experiment) total |
|---|---|---|---|
| Plate detection (1,225-1,226 frames with a detection) | **388.86s** | **21.43s** | 26.30s |
| OCR (417-569 calls) | **36.01s** | **20.81s** | 26.23s |
| **Combined total** | **424.87s** (~7.1 min) | **42.25s** | 52.53s |

- **Detection total: ~18x faster on GPU** (388.86s → 21.43s) — this is the dominant cost on CPU and
  the main thing GPU fixes.
- **OCR total (CPU-forced on both sides): ~1.7x faster on Colab's CPU** than the local test
  machine's CPU (36.01s → 20.81s) — not a GPU effect, just a faster CPU in the Colab VM.
- **End-to-end, the whole video processed ~10x faster overall on Colab** (424.87s → 42.25s) even
  with OCR still running on CPU in both places.
- The GPU-OCR experiment column total (52.53s) is *higher* than the CPU-OCR column (42.25s)
  because it includes a one-time ~5.8s cuDNN/JIT warmup cost on its first detection call (see §5,
  item 4) and made more OCR attempts overall (569 vs 417 — OcrGate's throttling is timing-sensitive,
  so a run with a warmup spike diverges slightly in *when* it retries OCR, though not in the final
  validated plates). Per-call OCR average was still faster on GPU (see §4 table).

### 4.2 Per-frame total (YOLO detection + OCR combined, for one frame)

Every frame pays the detection cost; only a minority of frames also trigger an OCR call that
frame (`OcrGate` throttles OCR to first-sighting + occasional retries, not every frame — see
`app/services/ocr_gate.py`). Average **combined** cost per frame, split by whether OCR ran that
frame or not:

| | Local (CPU) | Colab (GPU detect / CPU OCR) | Colab (GPU detect / GPU OCR experiment) |
|---|---|---|---|
| **All frames** (avg) | 346.8 ms | **34.5 ms** | 42.8 ms |
| Frames **with** an OCR call (avg, detect+OCR) | 416.6 ms (n=348) | **77.5 ms** (n=349) | 95.7 ms (n=411) |
| Frames **without** OCR (avg, detection only) | 319.1 ms (n=877) | **17.3 ms** (n=877) | 16.2 ms (n=815) |

So for **one typical frame**:
- No OCR triggered that frame → **~319ms (CPU) vs ~17ms (GPU)**, detection only.
- OCR triggered that frame → **~417ms (CPU) vs ~78ms (GPU)**, detection + one OCR call.
- Across the whole video, ~28% of frames (348/1225) triggered an OCR call; the rest were
  detection-only.

### 4.3 Stage averages + grand totals, side by side

Per-call average for each stage individually, plus each stage's total time over the whole video
(same numbers as §4 and §4.1, collected here in one table for quick reference):

| | Local (CPU) | Colab (GPU detect / CPU OCR) | Colab (GPU detect / GPU OCR experiment) |
|---|---|---|---|
| **Plate detection — avg per call** | 318.5 ms | **17.4 ms** | 22.6 ms |
| **Plate detection — total (1,225-1,226 calls)** | 388.86 s | **21.43 s** | 26.30 s |
| **OCR — avg per call** | 86.4 ms | **49.8 ms** | 46.1 ms |
| **OCR — total (417-569 calls)** | 36.01 s | **20.81 s** | 26.23 s |
| **Grand total (detection + OCR)** | **424.87 s** | **42.25 s** | 52.53 s |

## 5. Environment Issues Encountered (Colab setup)

Getting a clean Colab run required working through several environment conflicts, all resolved
during testing:

1. **numpy ABI mismatch** — `opencv-python-headless` installed against a different numpy major
   version than Colab's preinstalled one (`numpy.core.multiarray failed to import`). Fixed by
   installing `numpy==1.26.4` **last**, with `--force-reinstall --no-deps`, so nothing installed
   afterward could re-corrupt it, followed by one runtime restart.
2. **Stale Python import cache** — after unzipping `app.zip` mid-session, `ModuleNotFoundError: No
   module named 'app'` persisted even after the folder existed, because Python's `FileFinder`
   cached the "not found" result from before extraction. Fixed with `importlib.invalidate_caches()`
   or a session restart.
3. **cuDNN sublibrary version mismatch** (`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`) — PaddleOCR's
   GPU predictor (via `paddlepaddle-gpu==2.6.2`) and torch's bundled cuDNN loaded conflicting
   versions in the same process. Worked around by forcing just the OCR engine onto CPU
   (`PaddleOCR(..., use_gpu=False)`), leaving plate detection (torch/YOLO) on GPU.
4. **GPU-OCR experiment** — attempted resolving cuDNN properly by upgrading to
   `paddlepaddle-gpu==3.3.0` (cu126 build). One run succeeded and showed OCR running on GPU with
   correct results and a modest **~1.29x** OCR speedup over CPU (see `results/with_ocr/` vs
   `results/without_ocr/` in `backend/`), but a later attempt in the same environment broke `torch`
   entirely (`libtorch_cuda.so: undefined symbol: ncclCommShrink`) — the paddle 3.3.0 install
   downgraded a shared CUDA library (`nvidia-nccl-cu12`) that torch's compiled binary required at
   a different exact version. **Conclusion: GPU OCR is possible but not reliably reproducible** in
   this shared torch+paddle Colab environment without deeper, fragile version-pinning across both
   frameworks' CUDA dependencies — not recommended to depend on for now.

## 6. Recommendations

1. **Accuracy is verified — no correctness risk in a CPU→GPU move** for this pipeline's detection
   and OCR stages, given identical model/code.
2. **GPU primarily benefits the detection stage** (~18x) — if/when deploying on GPU hardware,
   expect the biggest win there; OCR stays CPU-bound unless the paddle/torch cuDNN conflict is
   solved more robustly (e.g. isolating OCR in a separate process/environment with its own pinned
   CUDA stack).
3. **Colab is a testing tool, not a production target** for this project — see
   `COLAB_T4_SPECS.md` for session-length/idle-timeout limits and why a dedicated GPU VM is the
   right choice for a real always-on multi-camera deployment.
4. Raw data backing this report: `backend/local_results.csv`, `backend/local_results_frames.csv`,
   `backend/results/without_ocr/`, `backend/results/with_ocr/`, and the comparison tools
   `backend/compare_pipeline_runs.py` / `backend/compare_frames.py`.
