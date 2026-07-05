import cv2
import time
from ultralytics import YOLO

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "backend/models/yolov8n.pt"
VIDEO_PATH = "/home/ubuntu/Videos/anpr_videos/vid1.mp4"

# Vehicle class names in your model
# Change according to your training classes
# COCO class indices (as used by stock yolov8n.pt)
CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# -----------------------------
# Load Model
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# Open Video
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video")
    exit()

prev_time = time.time()

cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vehicle Detection", 960, 540)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # -----------------------------
    # YOLO Inference
    # -----------------------------
    start = time.perf_counter()

    results = model(
        frame,
        conf=0.30,
        verbose=False
    )

    inference_time = (time.perf_counter() - start) * 1000  # milliseconds

    vehicle_count = 0

    # -----------------------------
    # Draw detections
    # -----------------------------
    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in CLASS_NAMES:
                continue

            vehicle_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{CLASS_NAMES[cls]} {conf:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # -----------------------------
    # FPS Calculation
    # -----------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # -----------------------------
    # Display Information
    # -----------------------------
    cv2.putText(
        frame,
        f"FPS : {fps:.2f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Inference : {inference_time:.2f} ms",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles : {vehicle_count}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Vehicle Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:      # ESC
        break

cap.release()
cv2.destroyAllWindows()