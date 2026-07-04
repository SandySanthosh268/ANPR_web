from dataclasses import dataclass

import numpy as np

from app.config import VEHICLE_CLASS_MAP, VEHICLE_CONF_THRESHOLD, VEHICLE_DETECTION_IMGSZ
from app.detection.vehicle_detector import VehicleDetector
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrackedVehicle:
    track_id: int
    bbox: tuple[int, int, int, int]
    vehicle_type: str
    confidence: float


class ByteTracker:
    """Assigns stable track IDs to detected vehicles across frames using
    Ultralytics' built-in ByteTrack integration, rather than a hand-rolled
    implementation.
    """

    def __init__(self, detector: VehicleDetector):
        self._detector = detector

    def track(self, frame: np.ndarray) -> list[TrackedVehicle]:
        results = self._detector.model.track(
            frame,
            conf=VEHICLE_CONF_THRESHOLD,
            classes=list(VEHICLE_CLASS_MAP.keys()),
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            imgsz=VEHICLE_DETECTION_IMGSZ,
        )
        tracked: list[TrackedVehicle] = []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return tracked
        for box, track_id in zip(boxes, boxes.id):
            cls_id = int(box.cls.item())
            vehicle_type = VEHICLE_CLASS_MAP.get(cls_id)
            if vehicle_type is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            tracked.append(
                TrackedVehicle(
                    track_id=int(track_id.item()),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    vehicle_type=vehicle_type,
                    confidence=float(box.conf.item()),
                )
            )
        return tracked
