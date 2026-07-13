"""Compare two run_pipeline_offline.py / colab notebook *_frames.csv files
row-by-row -- unlike compare_pipeline_runs.py (which only compares the final
validated-plate summary), this checks every detected plate on every frame:
same bbox/confidence, same OCR text, and the per-stage inference-time delta.

Rows are matched by (frame_id, plate bbox) rather than track_id, since track
IDs are assigned independently in each run and aren't comparable across runs
-- the physical plate position in a given frame is.

Usage:
    python compare_frames.py without_ocr/colab_frames.csv with_ocr/colab_frames.csv
"""

import argparse
import csv
from statistics import mean


def _bbox_key(row: dict, prefix: str) -> tuple:
    return (
        row[f"{prefix}_x1"], row[f"{prefix}_y1"],
        row[f"{prefix}_x2"], row[f"{prefix}_y2"],
    )


def load_rows(path: str) -> dict[tuple, dict]:
    rows: dict[tuple, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["frame_id"],) + _bbox_key(row, "plate_bbox")
            rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two per-frame pipeline CSVs")
    parser.add_argument("run_a", help="e.g. without_ocr/colab_frames.csv")
    parser.add_argument("run_b", help="e.g. with_ocr/colab_frames.csv")
    args = parser.parse_args()

    a = load_rows(args.run_a)
    b = load_rows(args.run_b)

    both_keys = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))

    print(f"{args.run_a}: {len(a)} detected plate-frames")
    print(f"{args.run_b}: {len(b)} detected plate-frames")
    print(f"Matched (same frame + same plate bbox): {len(both_keys)}")
    if only_a:
        print(f"Only in {args.run_a}: {len(only_a)} (e.g. {only_a[:5]})")
    if only_b:
        print(f"Only in {args.run_b}: {len(only_b)} (e.g. {only_b[:5]})")

    conf_mismatches = 0
    ocr_status_mismatches = 0
    ocr_text_mismatches = 0
    detect_ms_a, detect_ms_b = [], []
    ocr_ms_a, ocr_ms_b = [], []

    for key in both_keys:
        ra, rb = a[key], b[key]

        if ra["plate_confidence"] != rb["plate_confidence"]:
            conf_mismatches += 1

        detect_ms_a.append(float(ra["detect_inference_ms"]))
        detect_ms_b.append(float(rb["detect_inference_ms"]))

        if ra["ocr_ran"] == "True" and rb["ocr_ran"] == "True":
            if ra["ocr_status"] != rb["ocr_status"]:
                ocr_status_mismatches += 1
            if ra["ocr_validated_plate"] != rb["ocr_validated_plate"]:
                ocr_text_mismatches += 1
            if ra["ocr_inference_ms"]:
                ocr_ms_a.append(float(ra["ocr_inference_ms"]))
            if rb["ocr_inference_ms"]:
                ocr_ms_b.append(float(rb["ocr_inference_ms"]))

    print(f"\nPlate-confidence mismatches: {conf_mismatches}/{len(both_keys)}")
    print(f"OCR status (accepted/rejected/no_text) mismatches: {ocr_status_mismatches}")
    print(f"OCR validated-plate text mismatches: {ocr_text_mismatches}")

    print(f"\n=== Detection timing (ms) ===")
    print(f"{args.run_a}: avg={mean(detect_ms_a):.1f}  min={min(detect_ms_a):.1f}  max={max(detect_ms_a):.1f}")
    print(f"{args.run_b}: avg={mean(detect_ms_b):.1f}  min={min(detect_ms_b):.1f}  max={max(detect_ms_b):.1f}")

    if ocr_ms_a and ocr_ms_b:
        print(f"\n=== OCR timing (ms), rows where both ran OCR ===")
        print(f"{args.run_a}: avg={mean(ocr_ms_a):.1f}  min={min(ocr_ms_a):.1f}  max={max(ocr_ms_a):.1f}  (n={len(ocr_ms_a)})")
        print(f"{args.run_b}: avg={mean(ocr_ms_b):.1f}  min={min(ocr_ms_b):.1f}  max={max(ocr_ms_b):.1f}  (n={len(ocr_ms_b)})")
        print(f"Speedup ({args.run_b} vs {args.run_a}): {mean(ocr_ms_a)/mean(ocr_ms_b):.2f}x")


if __name__ == "__main__":
    main()
