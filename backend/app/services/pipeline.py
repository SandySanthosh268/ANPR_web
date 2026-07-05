import time

import numpy as np

from app.camera.frame_source import Frame
from app.detection.plate_detector import PlateDetector
from app.detection.vehicle_detector import VehicleDetector
from app.ocr.plate_reader import PlateReader
from app.services import segment_store
from app.services.ocr_gate import OcrGate
from app.services.ocr_worker import OcrJob, OcrWorker
from app.services.result_sink import DetectionResult, ResultSink
from app.tracking.bytetrack import ByteTracker, TrackedVehicle
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ROLLING_WINDOW = 5
_TARGET_FRAME_BUDGET_SECONDS = 1.0 / 15  # aim to keep up with ~15 FPS on CPU
_FORCE_PLATE_REFRESH_EVERY = 5  # frames to skip at most before retrying plate detection


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


class Pipeline:
    """Wires camera -> vehicle detection+tracking -> plate detection -> OCR
    -> result aggregation -> sink, per the ANPR processing flow.

    Adapts to CPU load: vehicle tracking always runs every frame (it's cheap,
    ~0.1-0.3s, and ByteTrack's ID matching depends on seeing consecutive
    frames with small motion between them — skipping frames here caused fast
    ID switches for moving vehicles while static ones looked fine, since zero
    motion always matches). Only the expensive per-vehicle plate detection +
    OCR stage is skipped under load, with a periodic forced retry so it
    doesn't get permanently stuck skipping.
    """

    def __init__(self, camera_id: str, sink: ResultSink):
        self.camera_id = camera_id
        self.sink = sink

        self.vehicle_detector = VehicleDetector()
        self.tracker = ByteTracker(self.vehicle_detector)
        self.plate_detector = PlateDetector()
        self.ocr_reader = PlateReader()
        self.ocr_gate = OcrGate()
        # OCR runs on its own thread — it's ~1-2s per call on CPU, and running
        # it inline on the main loop stalled frame consumption long enough to
        # overflow the frame queue and drop everything that arrived meanwhile.
        self.ocr_worker = OcrWorker(self.ocr_reader, self.ocr_gate, self._on_ocr_result)

        self._recent_durations: list[float] = []
        self._frames_since_plate_attempt = 0

        self._warmup()

    def _warmup(self) -> None:
        # YOLO and PaddleOCR both pay a one-time cold-start cost (thread pool
        # init, graph compilation) on their first real inference call — often
        # several seconds. Paying it here, before frames start flowing, keeps
        # it from silently eating the start of a live/paced video stream.
        start = time.monotonic()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            self.tracker.track(dummy_frame)
            self.plate_detector.detect(dummy_crop)
            self.ocr_reader.read(dummy_crop)
        except Exception:
            logger.exception("Model warmup failed (non-fatal, continuing)")
        logger.info("Model warmup took %.2fs", time.monotonic() - start)

    def _should_skip_plate_stage(self) -> bool:
        # Capped so this never sticks permanently: a frame that skips the
        # plate stage still keeps its vehicle_bbox (tracking always ran), so
        # staleness here only costs a missed plate read, not a missing box.
        if self._frames_since_plate_attempt >= _FORCE_PLATE_REFRESH_EVERY:
            return False
        if len(self._recent_durations) < _ROLLING_WINDOW:
            return False
        avg = sum(self._recent_durations) / len(self._recent_durations)
        return avg > _TARGET_FRAME_BUDGET_SECONDS * 2

    def _record_duration(self, duration: float) -> None:
        self._recent_durations.append(duration)
        if len(self._recent_durations) > _ROLLING_WINDOW:
            self._recent_durations.pop(0)

    def process(self, frame: Frame) -> list[DetectionResult]:
        start = time.monotonic()
        tracked = self.tracker.track(frame.image)

        skip_plate = self._should_skip_plate_stage()
        self._frames_since_plate_attempt = 0 if not skip_plate else self._frames_since_plate_attempt + 1

        results: list[DetectionResult] = []
        for vehicle in tracked:
            results.append(self._process_vehicle(frame, vehicle, skip_plate))

        self._record_duration(time.monotonic() - start)
        return results

    def _on_ocr_result(self, track_id: int, vehicle_type: str, plate_text: str, confidence: float) -> None:
        # Runs on the OcrWorker thread, once a result comes back — arrives
        # asynchronously relative to whichever frame triggered it, so it's
        # surfaced through a separate per-track store (polled by the
        # frontend's Detection Table) rather than retrofitted into a
        # frame-by-frame detections list that may already be served.
        store = segment_store.get(self.camera_id)
        if store is not None:
            store.record_plate_result(track_id, vehicle_type, plate_text, confidence)

    def _process_vehicle(
        self, frame: Frame, vehicle: TrackedVehicle, skip_plate: bool
    ) -> DetectionResult:
        vehicle_crop = _crop(frame.image, vehicle.bbox)
        plate_bbox: tuple[int, int, int, int] | None = None
        plate_detection_confidence: float | None = None

        plate = None if skip_plate else self.plate_detector.detect(vehicle_crop)
        if plate is not None:
            px1, py1, px2, py2 = plate.bbox
            vx1, vy1, _, _ = vehicle.bbox
            plate_bbox = (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)
            plate_detection_confidence = plate.confidence
            plate_crop = _crop(vehicle_crop, plate.bbox)

            if self.ocr_gate.should_run(
                vehicle.track_id, frame.frame_id, plate_crop
            ) and not self.ocr_worker.is_pending(vehicle.track_id):
                self.ocr_worker.submit(
                    OcrJob(
                        track_id=vehicle.track_id,
                        frame_id=frame.frame_id,
                        vehicle_type=vehicle.vehicle_type,
                        plate_crop=plate_crop,
                    )
                )

        result = DetectionResult(
            camera_id=self.camera_id,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            track_id=vehicle.track_id,
            vehicle_type=vehicle.vehicle_type,
            vehicle_bbox=vehicle.bbox,
            plate_bbox=plate_bbox,
            plate=None,
            vehicle_confidence=vehicle.confidence,
            plate_confidence=plate_detection_confidence,
            ocr_confidence=None,
        )
        self.sink.emit(result)
        return result
