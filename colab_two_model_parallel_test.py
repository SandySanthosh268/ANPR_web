# ==============================================================================
# Two-model test: vehicle detection (YOLOv8s) + plate detection (YOLOv8s),
# run on the SAME video, three different ways -- to answer: does running two
# models "at once" actually overlap on the GPU, or does it just add up?
#
# Run this in Google Colab (Runtime -> Change runtime type -> GPU -> T4).
# Paste each "# --- cell ---" block into its own Colab cell, in order.
#
# Three modes measured over the whole video, each a separate full pass:
#   A) SEQUENTIAL      -- vehicle then plate, one after another (baseline)
#   B) THREADED-NAIVE  -- both launched via Python threads, but on the GPU's
#                         one default CUDA stream (still effectively serialized)
#   C) THREADED-STREAMS-- both launched via Python threads, each on its OWN
#                         CUDA stream (genuine attempt at concurrent GPU work)
# ==============================================================================

# --- cell: install deps ---
!pip install -q ultralytics opencv-python-headless

# --- cell: mount Drive ---
from google.colab import drive
drive.mount('/content/drive')

DRIVE_FOLDER = '/content/drive/MyDrive/anpr_pipeline_test'

import os
VEHICLE_MODEL_PATH = f'{DRIVE_FOLDER}/yolov11s.pt'          # vehicle model (YOLOv8s size)
PLATE_MODEL_PATH = f'{DRIVE_FOLDER}/vechile_plate_yolov8s.pt'  # plate model (YOLOv8s size)
VIDEO_PATH = f'{DRIVE_FOLDER}/sample_video.mp4'

for p in (VEHICLE_MODEL_PATH, PLATE_MODEL_PATH, VIDEO_PATH):
    assert os.path.exists(p), f"Missing: {p} -- check the filename/folder in Drive"
print("Found vehicle model, plate model, and video.")

# --- cell: load both models ---
import time
import threading
import cv2
import torch
from ultralytics import YOLO

CONF = 0.25
IMGSZ = 640
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

vehicle_model = YOLO(VEHICLE_MODEL_PATH)
vehicle_model.to(device)
plate_model = YOLO(PLATE_MODEL_PATH)
plate_model.to(device)

# Warm up both (first call always pays a one-time cold-start cost -- exclude from timing)
cap = cv2.VideoCapture(VIDEO_PATH)
ok, warm_frame = cap.read()
if ok:
    vehicle_model.predict(warm_frame, conf=CONF, imgsz=IMGSZ, verbose=False)
    plate_model.predict(warm_frame, conf=CONF, imgsz=IMGSZ, verbose=False)
cap.release()
print("Both models loaded and warmed up.")

# --- cell: helper to run one full pass over the video in a given mode ---
import contextlib

def _nullcontext():
    return contextlib.nullcontext()


def run_pass(mode: str):
    """mode: 'sequential' | 'threaded_naive' | 'threaded_streams'"""
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vehicle_ms_list = []
    plate_ms_list = []
    frame_wall_ms_list = []
    frame_id = 0

    # Separate CUDA streams, created once, reused every frame (mode C only)
    if mode == "threaded_streams" and device == "cuda":
        vehicle_stream = torch.cuda.Stream()
        plate_stream = torch.cuda.Stream()

    t_pass_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        result = {}

        def run_vehicle(stream=None):
            t0 = time.perf_counter()
            ctx = torch.cuda.stream(stream) if stream is not None else _nullcontext()
            with ctx:
                vehicle_model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)
                if device == "cuda":
                    torch.cuda.current_stream().synchronize()
            result["vehicle_ms"] = (time.perf_counter() - t0) * 1000

        def run_plate(stream=None):
            t0 = time.perf_counter()
            ctx = torch.cuda.stream(stream) if stream is not None else _nullcontext()
            with ctx:
                plate_model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)
                if device == "cuda":
                    torch.cuda.current_stream().synchronize()
            result["plate_ms"] = (time.perf_counter() - t0) * 1000

        t_frame_start = time.perf_counter()

        if mode == "sequential":
            run_vehicle()
            run_plate()
        elif mode == "threaded_naive":
            t1 = threading.Thread(target=run_vehicle)
            t2 = threading.Thread(target=run_plate)
            t1.start(); t2.start()
            t1.join(); t2.join()
        elif mode == "threaded_streams":
            t1 = threading.Thread(target=run_vehicle, args=(vehicle_stream if device == "cuda" else None,))
            t2 = threading.Thread(target=run_plate, args=(plate_stream if device == "cuda" else None,))
            t1.start(); t2.start()
            t1.join(); t2.join()
        else:
            raise ValueError(mode)

        frame_wall_ms = (time.perf_counter() - t_frame_start) * 1000
        frame_wall_ms_list.append(frame_wall_ms)
        vehicle_ms_list.append(result["vehicle_ms"])
        plate_ms_list.append(result["plate_ms"])

        if frame_id % 200 == 0:
            print(f"  [{mode}] ...{frame_id}/{total_frames} frames")

    cap.release()
    total_elapsed = time.perf_counter() - t_pass_start

    return {
        "mode": mode,
        "frames": frame_id,
        "total_elapsed_s": total_elapsed,
        "fps": frame_id / total_elapsed,
        "avg_frame_wall_ms": sum(frame_wall_ms_list) / len(frame_wall_ms_list),
        "avg_vehicle_ms": sum(vehicle_ms_list) / len(vehicle_ms_list),
        "avg_plate_ms": sum(plate_ms_list) / len(plate_ms_list),
    }


# --- cell: run all three modes and compare ---
results = []
for mode in ("sequential", "threaded_naive", "threaded_streams"):
    print(f"\n=== Running mode: {mode} ===")
    results.append(run_pass(mode))

print("\n" + "=" * 90)
print(f"{'Mode':20}{'FPS':>10}{'Total(s)':>12}{'AvgFrame(ms)':>16}{'AvgVehicle(ms)':>17}{'AvgPlate(ms)':>15}")
for r in results:
    print(f"{r['mode']:20}{r['fps']:>10.1f}{r['total_elapsed_s']:>12.2f}"
          f"{r['avg_frame_wall_ms']:>16.1f}{r['avg_vehicle_ms']:>17.1f}{r['avg_plate_ms']:>15.1f}")

seq = results[0]
for r in results[1:]:
    speedup = seq["avg_frame_wall_ms"] / r["avg_frame_wall_ms"]
    print(f"\n{r['mode']} vs sequential: {speedup:.2f}x on avg per-frame wall time "
          f"({'faster' if speedup > 1 else 'slower or no real overlap'})")
