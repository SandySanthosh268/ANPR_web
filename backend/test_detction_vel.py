import cv2
import time
from ultralytics import YOLO

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "/home/ubuntu/Documents/sandy_files/ANPR_web3/backend/backend/models/best.pt"
VIDEO_PATH = "/home/ubuntu/Videos/anpr_videos/vid1.mp4"
CONF_THRESHOLD = 0.20

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

cv2.namedWindow("Plate Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Plate Detection", 960, 540)

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
        conf=CONF_THRESHOLD,
        verbose=False
    )

    inference_time = (time.perf_counter() - start) * 1000  # milliseconds

    plate_count = 0

    # -----------------------------
    # Draw detections
    # -----------------------------
    for result in results:

        boxes = result.boxes

        for box in boxes:

            conf = float(box.conf[0])

            plate_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"plate {conf:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
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
        f"Plates : {plate_count}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Plate Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:      # ESC
        break

cap.release()
cv2.destroyAllWindows()