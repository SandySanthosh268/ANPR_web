# ==============================================================================
# ANPR plate-detection: CPU vs GPU inference benchmark
#
# Run this in Google Colab (Runtime -> Change runtime type -> GPU -> T4).
# Paste each "# --- cell ---" block into its own Colab cell, in order.
# ==============================================================================

# --- cell: install deps ---
!pip install -q ultralytics opencv-python-headless

# --- cell: mount Google Drive ---
# One-time setup on your own computer (before this step):
#   1. Open drive.google.com in a browser
#   2. Create a folder, e.g. "anpr_benchmark"
#   3. Upload into it: number_plate_yolov8s_v2.pt  and  vid1.mp4
#      (backend/models/number_plate_yolov8s_v2.pt and your sample video)
# This step will pop up a Google sign-in / permission prompt — approve it.
from google.colab import drive
drive.mount('/content/drive')

# Adjust this path to wherever you put the folder in your Drive.
DRIVE_FOLDER = '/content/drive/MyDrive/anpr_benchmark'

MODEL_PATH = f'{DRIVE_FOLDER}/number_plate_yolov8s_v2.pt'
VIDEO_PATH = f'{DRIVE_FOLDER}/vid1.mp4'

import os
assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH} — check the folder/filename in Drive"
assert os.path.exists(VIDEO_PATH), f"Video not found at {VIDEO_PATH} — check the folder/filename in Drive"
print("Found both files in Drive.")

# --- alternative cell: upload files directly instead of Drive ---
# Skip this if you used the Drive cell above. Only needed if you'd rather
# upload through the browser each session instead of using Drive.
# from google.colab import files
# print("Upload the model weight (.pt) first...")
# uploaded_model = files.upload()
# print("Now upload a sample video (.mp4)...")
# uploaded_video = files.upload()
# MODEL_PATH = list(uploaded_model.keys())[0]
# VIDEO_PATH = list(uploaded_video.keys())[0]

# --- cell: confirm GPU is active ---
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

# --- cell: benchmark ---
import cv2
import time
from ultralytics import YOLO

# Same settings as production (backend/app/config.py)
CONF_THRESHOLD = 0.25
IMGSZ = 640
STRIDE = 6          # matches PROCESSING_FPS=5 decimation on a ~28-30fps source
NUM_FRAMES = 200     # how many sampled frames to benchmark

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.to(device)
print(f"Model loaded on: {device}")

cap = cv2.VideoCapture(VIDEO_PATH)

# Warm up (first call always pays a one-time cold-start cost, exclude from timing)
ok, warm_frame = cap.read()
if ok:
    model.predict(warm_frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind after warmup

times = []
detections_found = 0
frame_id = 0
checked = 0

while checked < NUM_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    if frame_id % STRIDE != 0:
        continue
    checked += 1

    t0 = time.perf_counter()
    results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)
    torch.cuda.synchronize() if device == "cuda" else None  # ensure GPU work finished before stopping the clock
    dt = time.perf_counter() - t0
    times.append(dt)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        detections_found += 1

cap.release()

avg_t = sum(times) / len(times)
print(f"\n=== Results ({device.upper()}) ===")
print(f"Frames benchmarked: {len(times)}")
print(f"Frames with a detection: {detections_found} ({detections_found/len(times):.1%})")
print(f"avg={avg_t:.4f}s  min={min(times):.4f}s  max={max(times):.4f}s")
print(f"Max sustainable fps: {1/avg_t:.1f}")

# --- cell: compare against known CPU baseline ---
# These numbers were measured directly on the project's CPU-only machine
# (see ANPR_FLOW_REPORT.md, Section 4.1), same model + same settings.
cpu_avg = 0.38
cpu_min = 0.35
cpu_max = 1.4

print("\n=== CPU vs GPU comparison ===")
print(f"{'':12}{'avg':>10}{'min':>10}{'max':>10}{'max fps':>12}")
print(f"{'CPU':12}{cpu_avg:>10.3f}{cpu_min:>10.3f}{cpu_max:>10.3f}{1/cpu_avg:>12.1f}")
print(f"{device.upper():12}{avg_t:>10.3f}{min(times):>10.3f}{max(times):>10.3f}{1/avg_t:>12.1f}")
print(f"\nSpeedup: {cpu_avg/avg_t:.1f}x faster on {device.upper()} (avg case)")
