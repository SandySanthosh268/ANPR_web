import threading
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import OCR_CONF_THRESHOLD, OCR_COOLDOWN_FRAMES, OCR_MAX_ATTEMPTS_PER_TRACK


def _sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class _TrackOcrState:
    best_confidence: float = 0.0
    best_sharpness: float = 0.0
    attempts: int = 0
    last_ocr_frame: int = -10**9


class OcrGate:
    """Per-track throttle deciding when a plate crop is worth running OCR on.

    Without this, OCR (the most expensive stage) would run on every frame a
    track is visible, even for a vehicle stopped at a light for 10+ seconds.
    OCR runs only for a track's first sighting, or when quality plausibly
    improved (higher plate-crop sharpness than the previous attempt) and the
    track hasn't already hit its confidence target, attempt cap, or cooldown.
    """

    def __init__(
        self,
        conf_threshold: float = OCR_CONF_THRESHOLD,
        max_attempts: int = OCR_MAX_ATTEMPTS_PER_TRACK,
        cooldown_frames: int = OCR_COOLDOWN_FRAMES,
    ):
        self._conf_threshold = conf_threshold
        self._max_attempts = max_attempts
        self._cooldown_frames = cooldown_frames
        self._state: dict[int, _TrackOcrState] = {}
        # should_run() runs on the main pipeline thread; record_attempt() runs
        # on the OcrWorker background thread once a result comes back — both
        # touch the same per-track state, so this needs a lock.
        self._lock = threading.Lock()

    def should_run(self, track_id: int, frame_id: int, plate_crop: np.ndarray) -> bool:
        with self._lock:
            state = self._state.setdefault(track_id, _TrackOcrState())
            if state.attempts == 0:
                return True
            if state.best_confidence >= self._conf_threshold:
                return False
            if state.attempts >= self._max_attempts:
                return False
            if frame_id - state.last_ocr_frame < self._cooldown_frames:
                return False
            best_sharpness = state.best_sharpness
        return _sharpness(plate_crop) > best_sharpness

    def record_attempt(
        self, track_id: int, frame_id: int, confidence: float, plate_crop: np.ndarray
    ) -> None:
        sharpness = _sharpness(plate_crop)
        with self._lock:
            state = self._state.setdefault(track_id, _TrackOcrState())
            state.attempts += 1
            state.last_ocr_frame = frame_id
            state.best_confidence = max(state.best_confidence, confidence)
            state.best_sharpness = max(state.best_sharpness, sharpness)

    def forget(self, track_id: int) -> None:
        with self._lock:
            self._state.pop(track_id, None)
