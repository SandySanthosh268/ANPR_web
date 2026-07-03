from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from app.config import DEVICE, VEHICLE_CLASS_MAP, VEHICLE_CONF_THRESHOLD, VEHICLE_MODEL_PATH
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VehicleDetection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    vehicle_type: str
    confidence: float


class VehicleDetector:
    """Detects vehicles in a frame using a pretrained YOLO model, filtered to
    the vehicle-relevant COCO classes. The underlying `model` is also used by
    tracking.bytetrack.ByteTracker so the model is only loaded once.
    """

    def __init__(self, model_path: str = VEHICLE_MODEL_PATH, device: str = DEVICE):
        self.model = YOLO(model_path)
        self.model.to(device)
        logger.info("Loaded vehicle detector %s on %s", model_path, device)

    def detect(self, frame: np.ndarray) -> list[VehicleDetection]:
        results = self.model.predict(
            frame, conf=VEHICLE_CONF_THRESHOLD, classes=list(VEHICLE_CLASS_MAP.keys()), verbose=False
        )
        detections: list[VehicleDetection] = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            vehicle_type = VEHICLE_CLASS_MAP.get(cls_id)
            if vehicle_type is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                VehicleDetection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    vehicle_type=vehicle_type,
                    confidence=float(box.conf.item()),
                )
            )
        return detections
