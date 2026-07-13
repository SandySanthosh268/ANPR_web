import threading

from app.services.result_sink import DetectionResult

# Caps memory/response size for a long-running session — old attempts are
# dropped oldest-first once this many have accumulated.
_MAX_OCR_ATTEMPTS = 300


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
        # Detection bboxes are in the raw source frame's pixel coordinates
        # (VideoReader reads the source directly, independent of the HLS
        # transcode). The frontend needs this to scale boxes correctly since
        # the HLS preview stream is encoded at a *different* resolution
        # (downscaled for faster ffmpeg encoding) than the source it detects
        # on — video.videoWidth alone isn't the coordinate space bboxes live in.
        self.frame_width: int | None = None
        self.frame_height: int | None = None
        # Every OCR attempt (accepted, rejected, or nothing readable) —
        # populated asynchronously by OcrWorker, independent of which
        # segment/frame triggered it. Polled by the frontend's Detection
        # Table, decoupled from segment fetches (an OCR result can resolve
        # well after its triggering segment was already served to the
        # player). Kept as a bounded list, not deduped per track, so failed
        # attempts stay visible instead of being silently overwritten by a
        # later better one.
        self._ocr_attempts: list[dict] = []
        self._next_attempt_id = 1
        # Mirrors PlateTracker.total_vehicle_count for this camera — updated
        # each frame from the pipeline (see Pipeline.process) rather than
        # computed here, since counting "new track" is the tracker's job.
        self.vehicle_count = 0

    def set_vehicle_count(self, count: int) -> None:
        with self._lock:
            self.vehicle_count = count

    def set_source_resolution(self, width: int, height: int) -> None:
        with self._lock:
            self.frame_width = width
            self.frame_height = height

    def add_frame(self, segment_index: int, video_time: float, results: list[DetectionResult]) -> None:
        frame_entry = {
            "video_timestamp_sec": video_time,
            "video_timestamp_ms": round(video_time * 1000),
            "detections": [
                {
                    "track_id": r.track_id,
                    "vehicle_type": r.vehicle_type,
                    "vehicle_bbox": list(r.vehicle_bbox) if r.vehicle_bbox else None,
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

    def record_ocr_attempt(
        self,
        track_id: int,
        vehicle_type: str,
        plate: str | None,
        ocr_confidence: float | None,
        image: str,
        status: str,
    ) -> None:
        with self._lock:
            self._ocr_attempts.append(
                {
                    "id": self._next_attempt_id,
                    "track_id": track_id,
                    "vehicle_type": vehicle_type,
                    "plate": plate,
                    "ocr_confidence": ocr_confidence,
                    "image": image,
                    "status": status,
                }
            )
            self._next_attempt_id += 1
            if len(self._ocr_attempts) > _MAX_OCR_ATTEMPTS:
                self._ocr_attempts.pop(0)

    def get_plate_results(self) -> list[dict]:
        with self._lock:
            return list(reversed(self._ocr_attempts))


_registry: dict[str, SegmentStore] = {}
_registry_lock = threading.Lock()


def register(camera_id: str, store: SegmentStore) -> None:
    with _registry_lock:
        _registry[camera_id] = store


def get(camera_id: str) -> SegmentStore | None:
    with _registry_lock:
        return _registry.get(camera_id)
