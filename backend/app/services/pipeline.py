import base64
import time

import cv2
import numpy as np

from app.camera.frame_source import Frame
from app.detection.plate_detector import PlateDetector
from app.ocr.plate_reader import PlateReader
from app.services import segment_store
from app.services.ocr_gate import OcrGate
from app.services.ocr_worker import OcrJob, OcrWorker
from app.services.result_sink import DetectionResult, ResultSink
from app.tracking.plate_tracker import PlateTracker, TrackedPlate
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


def _encode_jpeg(image: np.ndarray) -> str:
    # A data URL keeps this self-contained in the JSON the Detection Table
    # already polls — no separate image-serving endpoint/file storage needed
    # for what's a small plate-crop thumbnail.
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


class Pipeline:
    """Wires camera -> plate detection+tracking -> OCR -> result aggregation
    -> sink. No *separate* vehicle-detection stage: the plate model is a
    combined vehicle+plate weight, so PlateTracker finds both in one pass
    and reports each plate's containing vehicle_type/vehicle_bbox alongside it.
    """

    def __init__(self, camera_id: str, sink: ResultSink):
        self.camera_id = camera_id
        self.sink = sink

        self.plate_detector = PlateDetector()
        self.tracker = PlateTracker(self.plate_detector)
        self.ocr_reader = PlateReader()
        self.ocr_gate = OcrGate()
        # OCR runs on its own thread — it's ~1-2s per call on CPU, and running
        # it inline on the main loop stalled frame consumption long enough to
        # overflow the frame queue and drop everything that arrived meanwhile.
        self.ocr_worker = OcrWorker(self.ocr_reader, self.ocr_gate, self._on_ocr_result)

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
            self.ocr_reader.read(dummy_crop)
        except Exception:
            logger.exception("Model warmup failed (non-fatal, continuing)")
        logger.info("Model warmup took %.2fs", time.monotonic() - start)

    def process(self, frame: Frame) -> list[DetectionResult]:
        tracked = self.tracker.track(frame.image)
        store = segment_store.get(self.camera_id)
        if store is not None:
            store.set_vehicle_count(self.tracker.total_vehicle_count)
        return [self._process_plate(frame, plate) for plate in tracked]

    def _on_ocr_result(
        self,
        track_id: int,
        frame_id: int,
        timestamp: float,
        vehicle_type: str,
        plate_text: str | None,
        confidence: float | None,
        plate_crop: np.ndarray,
        status: str,
    ) -> None:
        # Runs on the OcrWorker thread, once a result comes back — arrives
        # asynchronously relative to whichever frame triggered it, so it's
        # surfaced through a separate per-track store (polled by the
        # frontend's Detection Table) rather than retrofitted into a
        # frame-by-frame detections list that may already be served. Records
        # every OCR attempt (accepted, rejected, or nothing readable), not
        # just successful ones, so failures are visible for debugging rather
        # than silently disappearing.
        store = segment_store.get(self.camera_id)
        if store is not None:
            store.record_ocr_attempt(
                track_id, vehicle_type, plate_text, confidence, _encode_jpeg(plate_crop), status
            )

        # The regular per-frame emit() below always has plate=None (OCR
        # hasn't resolved yet at that point) — without this second emit, a
        # validated plate reading never reached the DB sink at all, only
        # the in-memory store above. Only "accepted" carries a plate value;
        # other statuses still emit so last_seen/vehicle_confidence in the
        # DB row keep advancing for a track that's still being read.
        self.sink.emit(
            DetectionResult(
                camera_id=self.camera_id,
                frame_id=frame_id,
                timestamp=timestamp,
                track_id=track_id,
                vehicle_type=vehicle_type,
                vehicle_bbox=None,
                plate_bbox=None,
                plate=plate_text if status == "accepted" else None,
                vehicle_confidence=None,
                plate_confidence=None,
                ocr_confidence=confidence if status == "accepted" else None,
            )
        )

    def _process_plate(self, frame: Frame, plate: TrackedPlate) -> DetectionResult:
        plate_crop = _crop(frame.image, plate.bbox)

        if self.ocr_gate.should_run(
            plate.track_id, frame.frame_id, plate_crop
        ) and not self.ocr_worker.is_pending(plate.track_id):
            self.ocr_worker.submit(
                OcrJob(
                    track_id=plate.track_id,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    vehicle_type=plate.vehicle_type,
                    plate_crop=plate_crop,
                )
            )

        result = DetectionResult(
            camera_id=self.camera_id,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            track_id=plate.track_id,
            vehicle_type=plate.vehicle_type,
            vehicle_bbox=plate.vehicle_bbox,
            plate_bbox=plate.bbox,
            plate=None,
            vehicle_confidence=None,
            plate_confidence=plate.confidence,
            ocr_confidence=None,
        )
        self.sink.emit(result)
        return result
