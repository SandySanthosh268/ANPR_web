import os
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from app.config import DEVICE, PLATE_CONF_THRESHOLD, PLATE_MODEL_PATH
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PlateDetection:
    bbox: tuple[int, int, int, int]  # coordinates relative to the vehicle crop
    confidence: float


class PlateDetector:
    """Detects a license plate within a cropped vehicle image.

    COCO pretrained YOLO has no license-plate class, so this expects a
    separately pretrained plate-detection YOLO weight (single class: 'plate').
    Point PLATE_MODEL_PATH at one of the small open license-plate YOLO models
    (e.g. from Roboflow Universe / Ultralytics HF hub) until a custom-trained
    model is available. Loading is lazy so a missing weight file doesn't break
    pipeline startup for vehicle-only testing.
    """

    def __init__(self, model_path: str = PLATE_MODEL_PATH, device: str = DEVICE):
        self._model_path = model_path
        self._device = device
        self._model: YOLO | None = None

    def _ensure_loaded(self) -> YOLO:
        if self._model is None:
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(
                    f"Plate detection model not found at '{self._model_path}'. "
                    "Download a pretrained license-plate YOLO weight (e.g. from "
                    "Roboflow Universe or the Ultralytics HF hub) and place it "
                    "at this path, or update PLATE_MODEL_PATH in app/config.py."
                )
            self._model = YOLO(self._model_path)
            self._model.to(self._device)
            logger.info("Loaded plate detector %s on %s", self._model_path, self._device)
        return self._model

    def detect(self, vehicle_crop: np.ndarray) -> PlateDetection | None:
        if vehicle_crop.size == 0:
            return None
        model = self._ensure_loaded()
        results = model.predict(vehicle_crop, conf=PLATE_CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        best = max(boxes, key=lambda b: float(b.conf.item()))
        x1, y1, x2, y2 = best.xyxy[0].tolist()
        return PlateDetection(
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            confidence=float(best.conf.item()),
        )
