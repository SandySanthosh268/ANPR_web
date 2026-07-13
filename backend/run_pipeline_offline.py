"""Run the real detection+tracking+OCR+validation pipeline over a video file
and dump a per-track results CSV — no FastAPI/DB/websocket/RTSP involved.

Uses the exact same classes as the production pipeline (PlateDetector,
PlateTracker, PlateReader, OcrGate, normalize_plate — see
app/services/pipeline.py) so this is not a re-implementation that could
drift from production behavior, just the same logic run synchronously
against a file instead of a live camera.

Purpose: produce a baseline CSV on this (CPU) machine, then run the paired
Colab notebook (colab_full_pipeline_test.ipynb) on the same video with the
same models and diff the two CSVs with compare_pipeline_runs.py to check
whether GPU inference changes *what* gets detected/read, not just how fast.

Writes two CSVs:
  --out         one row per track: best/validated plate reading summary.
  --frames-out  one row per detected plate *per frame*: bbox coordinates,
                detection confidence/inference time, and (on frames where
                OCR ran) the raw OCR text/confidence/inference time and
                whether it passed validation.

Run from the backend/ directory:
    python run_pipeline_offline.py --source /path/to/vid1.mp4 --out local_results.csv --frames-out local_frames.csv
"""

import argparse
import csv
import time

import cv2

from app.detection.plate_detector import PlateDetector
from app.ocr.plate_reader import PlateReader
from app.ocr.plate_validator import normalize_plate
from app.services.ocr_gate import OcrGate
from app.tracking.plate_tracker import PlateTracker


def _crop(frame, bbox):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


class TrackRecord:
    def __init__(self, vehicle_type: str, first_frame: int):
        self.vehicle_type = vehicle_type
        self.first_frame = first_frame
        self.last_frame = first_frame
        self.ocr_attempts = 0
        self.best_text: str | None = None
        self.best_confidence = 0.0
        self.readings: set[str] = set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline full-pipeline run (detect+track+OCR+validate)")
    parser.add_argument("--source", required=True, help="Video file path")
    parser.add_argument("--out", default="pipeline_results.csv", help="Track-summary output CSV path")
    parser.add_argument(
        "--frames-out", default=None,
        help="Per-frame detection CSV path (default: derived from --out, e.g. pipeline_results_frames.csv)",
    )
    args = parser.parse_args()
    frames_out = args.frames_out or args.out.rsplit(".", 1)[0] + "_frames.csv"

    detector = PlateDetector()
    tracker = PlateTracker(detector)
    ocr_reader = PlateReader()
    ocr_gate = OcrGate()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    records: dict[int, TrackRecord] = {}
    frame_id = 0
    t_start = time.perf_counter()

    frames_file = open(frames_out, "w", newline="")
    frames_writer = csv.writer(frames_file)
    frames_writer.writerow([
        "frame_id", "track_id", "vehicle_type",
        "vehicle_bbox_x1", "vehicle_bbox_y1", "vehicle_bbox_x2", "vehicle_bbox_y2",
        "plate_bbox_x1", "plate_bbox_y1", "plate_bbox_x2", "plate_bbox_y2",
        "plate_confidence", "detect_inference_ms",
        "ocr_ran", "ocr_raw_text", "ocr_confidence", "ocr_validated_plate",
        "ocr_status", "ocr_inference_ms",
    ])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        t_detect = time.perf_counter()
        tracked = tracker.track(frame)
        detect_ms = (time.perf_counter() - t_detect) * 1000

        for plate in tracked:
            record = records.get(plate.track_id)
            if record is None:
                record = TrackRecord(plate.vehicle_type, frame_id)
                records[plate.track_id] = record
            record.last_frame = frame_id

            vx1, vy1, vx2, vy2 = plate.vehicle_bbox if plate.vehicle_bbox else ("", "", "", "")
            px1, py1, px2, py2 = plate.bbox

            crop = _crop(frame, plate.bbox)
            ocr_ran = crop.size > 0 and ocr_gate.should_run(plate.track_id, frame_id, crop)
            ocr_raw_text = ocr_confidence = ocr_validated = ocr_status = ""
            ocr_ms = ""

            if ocr_ran:
                t_ocr = time.perf_counter()
                result = ocr_reader.read(crop)
                ocr_ms = f"{(time.perf_counter() - t_ocr) * 1000:.1f}"

                if result is None:
                    ocr_status = "no_text"
                else:
                    ocr_gate.record_attempt(plate.track_id, frame_id, result.confidence, crop)
                    record.ocr_attempts += 1
                    ocr_raw_text = result.text
                    ocr_confidence = f"{result.confidence:.3f}"

                    normalized = normalize_plate(result.text)
                    if normalized is not None:
                        ocr_status = "accepted"
                        ocr_validated = normalized
                        record.readings.add(normalized)
                        if result.confidence > record.best_confidence:
                            record.best_text = normalized
                            record.best_confidence = result.confidence
                    else:
                        ocr_status = "rejected"

            frames_writer.writerow([
                frame_id, plate.track_id, plate.vehicle_type,
                vx1, vy1, vx2, vy2,
                px1, py1, px2, py2,
                f"{plate.confidence:.3f}", f"{detect_ms:.1f}",
                ocr_ran, ocr_raw_text, ocr_confidence, ocr_validated,
                ocr_status, ocr_ms,
            ])

    frames_file.close()
    cap.release()
    elapsed = time.perf_counter() - t_start
    print(f"Processed {frame_id} frames in {elapsed:.1f}s ({frame_id/elapsed:.1f} fps)")
    print(f"Tracks seen: {len(records)}")
    print(f"Wrote {frames_out}")

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track_id", "vehicle_type", "first_frame", "last_frame",
            "ocr_attempts", "best_plate", "best_confidence", "all_readings",
        ])
        for track_id, r in sorted(records.items()):
            writer.writerow([
                track_id, r.vehicle_type, r.first_frame, r.last_frame,
                r.ocr_attempts, r.best_text or "", f"{r.best_confidence:.3f}",
                ";".join(sorted(r.readings)),
            ])
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()


