# ==============================================================================
# Two-model test: vehicle detection model + plate detection model, each with
# its OWN separate input video (like two different camera feeds), started at
# the same time -- to answer: if 2+ cameras each run their own model on the
# same GPU concurrently, what's the real FPS/inference impact?
#
# Unlike the earlier threading test (colab_two_model_parallel_test.py), this
# uses two SEPARATE OS PROCESSES (subprocess), not Python threads -- each
# process gets its own Python interpreter (no shared GIL) and its own CUDA
# context. This is the realistic architecture for "N cameras, N models" and
# the only way to get genuine concurrent execution instead of GIL-serialized
# threads (see the threaded_naive/threaded_streams results from last time --
# both were ~4x SLOWER than sequential, purely from Python-level overhead).
#
# Run this in Google Colab (Runtime -> Change runtime type -> GPU -> T4).
# Paste each "# --- cell ---" block into its own Colab cell, in order.
# ==============================================================================

# --- cell: install deps ---
!pip install -q ultralytics opencv-python-headless

# --- cell: mount Drive and set paths ---
from google.colab import drive
drive.mount('/content/drive')

DRIVE_FOLDER = '/content/drive/MyDrive/anpr_pipeline_test'

import os
VEHICLE_MODEL_PATH = f'{DRIVE_FOLDER}/yolov8s.pt'
PLATE_MODEL_PATH = f'{DRIVE_FOLDER}/vechile_plate_yolov8s.pt'

# Two DIFFERENT inputs, like two different camera feeds. If you only have one
# video, this just points both at the same file -- still a valid test (each
# process independently reads/decodes its own copy), just not literally
# different footage. Upload a second video and change VIDEO_B_PATH for a
# more realistic 2-camera test.
VIDEO_A_PATH = f'{DRIVE_FOLDER}/sample_video.mp4'   # vehicle model's input
VIDEO_B_PATH = f'{DRIVE_FOLDER}/sample_video_d.mp4'   # plate model's input -- change if you have a 2nd video

for p in (VEHICLE_MODEL_PATH, PLATE_MODEL_PATH, VIDEO_A_PATH, VIDEO_B_PATH):
    assert os.path.exists(p), f"Missing: {p} -- check the filename/folder in Drive"
print("Found both models and both video inputs.")

# --- cell: write the standalone worker script to disk ---
# Each worker is a fresh, independent Python process -- loads ONE model,
# processes ONE video start-to-finish, and writes its own timing stats to a
# JSON file. Nothing here is shared with the other worker (no GIL, no shared
# CUDA context) -- only the physical GPU hardware is shared between them.
worker_code = r'''
import argparse
import json
import time

import cv2
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    model.to(device)

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Warm up (excluded from timing)
    ok, warm_frame = cap.read()
    if ok:
        model.predict(warm_frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    times = []
    frame_id = 0
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        t0 = time.perf_counter()
        model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    cap.release()
    total_elapsed = time.perf_counter() - t_start

    result = {
        "label": args.label,
        "device": device,
        "frames": frame_id,
        "total_s": total_elapsed,
        "fps": frame_id / total_elapsed if total_elapsed > 0 else 0,
        "avg_ms": sum(times) / len(times) if times else 0,
        "min_ms": min(times) if times else 0,
        "max_ms": max(times) if times else 0,
    }
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"[{args.label}] done: {result['fps']:.1f} fps, avg {result['avg_ms']:.1f}ms, "
          f"total {result['total_s']:.2f}s over {result['frames']} frames")


if __name__ == "__main__":
    main()
'''

with open("/content/single_model_worker.py", "w") as f:
    f.write(worker_code)
print("Worker script written to /content/single_model_worker.py")

# --- cell: run both models as separate processes, started at the same time ---
import subprocess
import time
import json
import sys

VEHICLE_OUT = "/content/vehicle_result.json"
PLATE_OUT = "/content/plate_result.json"

def launch(model_path, video_path, label, out_path):
    return subprocess.Popen([
        sys.executable, "/content/single_model_worker.py",
        "--model", model_path, "--video", video_path,
        "--label", label, "--out", out_path,
    ])

print("Starting both processes at the same time...\n")
t_start = time.perf_counter()

p_vehicle = launch(VEHICLE_MODEL_PATH, VIDEO_A_PATH, "vehicle", VEHICLE_OUT)
p_plate = launch(PLATE_MODEL_PATH, VIDEO_B_PATH, "plate", PLATE_OUT)

p_vehicle.wait()
p_plate.wait()

combined_wall_s = time.perf_counter() - t_start

vehicle_result = json.load(open(VEHICLE_OUT))
plate_result = json.load(open(PLATE_OUT))

print("\n" + "=" * 70)
print("Run individually, each own process/GPU-context, own input video:")
print(f"  Vehicle model: {vehicle_result['fps']:>6.1f} fps  avg={vehicle_result['avg_ms']:>6.1f}ms  "
      f"total={vehicle_result['total_s']:>6.2f}s  ({vehicle_result['frames']} frames)")
print(f"  Plate model:   {plate_result['fps']:>6.1f} fps  avg={plate_result['avg_ms']:>6.1f}ms  "
      f"total={plate_result['total_s']:>6.2f}s  ({plate_result['frames']} frames)")

sum_if_sequential_s = vehicle_result["total_s"] + plate_result["total_s"]
print(f"\nSum of both totals, AS IF run one after another: {sum_if_sequential_s:.2f}s")
print(f"Actual combined wall time, running at the SAME time (2 processes): {combined_wall_s:.2f}s")
print(f"\nOverlap gained: {sum_if_sequential_s / combined_wall_s:.2f}x "
      f"(1.0x = no real overlap/fully serialized, 2.0x = perfect overlap, no GPU contention)")
