import os
from pathlib import Path

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Avoid oversubscribing the host: on CPU, torch defaults to using every core,
# which fights the concurrently-running ffmpeg HLS transcode (and anything
# else on the machine) for CPU time instead of leaving it any headroom.
if DEVICE == "cpu":
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

# backend/app/config.py -> backend/ -> backend/models
BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR / "models"

VEHICLE_MODEL_PATH = str(MODELS_DIR / "yolov8n.pt")
VEHICLE_CONF_THRESHOLD = 0.4
# YOLO resizes to this internally then rescales boxes back to the original
# frame's coordinates automatically — running at native 1920x1080 every
# frame is the single biggest CPU cost in the pipeline (cost scales roughly
# with resolution squared); 640 detects vehicle-scale objects fine at a
# fraction of the compute.
VEHICLE_DETECTION_IMGSZ = 640
# COCO class_id -> vehicle_type. COCO has no auto-rickshaw/van/pickup classes;
# truck is used as a stopgap for those until a custom-trained model replaces this.
VEHICLE_CLASS_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

PLATE_MODEL_PATH = str(MODELS_DIR / "best.pt")
PLATE_CONF_THRESHOLD = 0.20

OCR_LANG = "en"
OCR_CONF_THRESHOLD = 0.50
OCR_MAX_ATTEMPTS_PER_TRACK = 5
OCR_COOLDOWN_FRAMES = 10

FRAME_QUEUE_MAXSIZE = 5

# Decimate a video file's frame rate down to this before it ever reaches the
# pipeline — a fixed, uniform stride (unlike the earlier load-adaptive
# skipping) that cuts CPU workload substantially while keeping consecutive
# processed frames close enough together for ByteTrack to still follow
# normal traffic motion. None/0 disables decimation (process every frame).
PROCESSING_FPS = 10

RTSP_RECONNECT_INITIAL_DELAY = 1.0
RTSP_RECONNECT_MAX_DELAY = 30.0

DATABASE_URL = os.environ.get("DATABASE_URL")
DB_POOL_MIN_CONN = 1
DB_POOL_MAX_CONN = 5

HLS_OUTPUT_DIR = BACKEND_DIR / "hls_output"
FFMPEG_BINARY = "ffmpeg"
HLS_SEGMENT_SECONDS = 2

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")

# Host/port the browser should use to reach the FastAPI app (WebSocket + HLS +
# REST all share this one process/port). Distinct from the bind host
# (main.py's --ws-host, e.g. 0.0.0.0), which isn't a valid address to connect
# to from a browser.
PUBLIC_HOST = os.environ.get("PUBLIC_HOST", "localhost")
PUBLIC_PORT = int(os.environ.get("PUBLIC_PORT", "8765"))

