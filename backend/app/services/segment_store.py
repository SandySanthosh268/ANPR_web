import threading

from app.services.result_sink import DetectionResult


class SegmentStore:
    """In-memory, per-camera store of detection results grouped by HLS
    segment index. Replaces the earlier WebSocket per-frame stream: the
    frontend fetches a segment's detections directly when hls.js starts
    playing that segment (FRAG_CHANGED), which is inherently synchronized —
    no separate timeline-matching protocol needed. Reset fresh each time the
    backend starts a new source-loop cycle (a new SegmentStore is created and
    registered), consistent with everything else in the per-cycle pipeline.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._segments: dict[int, list[dict]] = {}
        # Under CPU backpressure the bounded frame queue can drop every frame
        # belonging to a whole segment, so add_frame() is never called for
        # it — tracked separately so get_segment() can tell "genuinely not
        # reached yet" (404) apart from "reached, but the pipeline had
        # nothing for it" (200, empty frames — not an error).
        self._highest_segment_seen = -1
        self.total_segments: int | None = None
        self.duration_s: float | None = None
        self.hls_ready = False
        # Best OCR reading per track_id so far — populated asynchronously by
        # OcrWorker, independent of which segment/frame triggered it. Polled
        # by the frontend's Detection Table, decoupled from segment fetches
        # (an OCR result can resolve well after its triggering segment was
        # already served to the player).
        self._plate_results: dict[int, dict] = {}

    def add_frame(self, segment_index: int, video_time: float, results: list[DetectionResult]) -> None:
        frame_entry = {
            "video_timestamp_sec": video_time,
            "video_timestamp_ms": round(video_time * 1000),
            "detections": [
                {
                    "track_id": r.track_id,
                    "vehicle_type": r.vehicle_type,
                    "vehicle_bbox": list(r.vehicle_bbox),
                    "plate_bbox": list(r.plate_bbox) if r.plate_bbox else None,
                    "plate": r.plate,
                    "vehicle_confidence": r.vehicle_confidence,
                    "ocr_confidence": r.ocr_confidence,
                }
                for r in results
            ],
        }
        with self._lock:
            self._segments.setdefault(segment_index, []).append(frame_entry)
            self._highest_segment_seen = max(self._highest_segment_seen, segment_index)

    def get_segment(self, index: int) -> dict | None:
        with self._lock:
            frames = self._segments.get(index)
            reached = index <= self._highest_segment_seen
        if frames is not None:
            return {"segment": index, "frames": frames}
        if reached:
            return {"segment": index, "frames": []}
        return None

    def mark_source_ended(self, total_segments: int, duration_s: float) -> None:
        with self._lock:
            self.total_segments = total_segments
            self.duration_s = duration_s
            self.hls_ready = True

    def record_plate_result(
        self, track_id: int, vehicle_type: str, plate: str, ocr_confidence: float
    ) -> None:
        with self._lock:
            existing = self._plate_results.get(track_id)
            if existing is not None and existing["ocr_confidence"] >= ocr_confidence:
                return
            self._plate_results[track_id] = {
                "track_id": track_id,
                "vehicle_type": vehicle_type,
                "plate": plate,
                "ocr_confidence": ocr_confidence,
            }

    def get_plate_results(self) -> list[dict]:
        with self._lock:
            return list(self._plate_results.values())


_registry: dict[str, SegmentStore] = {}
_registry_lock = threading.Lock()


def register(camera_id: str, store: SegmentStore) -> None:
    with _registry_lock:
        _registry[camera_id] = store


def get(camera_id: str) -> SegmentStore | None:
    with _registry_lock:
        return _registry.get(camera_id)
