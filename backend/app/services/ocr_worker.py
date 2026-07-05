import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.ocr.plate_reader import PlateReader
from app.ocr.plate_validator import is_valid_plate
from app.services.ocr_gate import OcrGate
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OcrJob:
    track_id: int
    frame_id: int
    vehicle_type: str
    plate_crop: np.ndarray


class OcrWorker:
    """Runs OCR — the pipeline's single most expensive stage, ~1-2s per call
    on CPU with PaddleOCR — on a dedicated background thread.

    Without this, one OCR call blocks the main per-frame loop for its full
    duration; during those ~2s the incoming (already-decimated) frame queue
    fills up and starts dropping frames, since nothing is draining it. Moving
    OCR off that thread lets tracking + plate detection keep consuming frames
    at their own (much faster) pace while OCR catches up whenever it can.

    A track can have at most one outstanding OCR job at a time (`_pending`),
    matching what OcrGate already assumed when OCR was synchronous.
    """

    def __init__(
        self,
        ocr_reader: PlateReader,
        ocr_gate: OcrGate,
        on_result: Callable[[int, str, str, float], None],
        maxsize: int = 4,
    ):
        self._ocr_reader = ocr_reader
        self._ocr_gate = ocr_gate
        self._on_result = on_result
        self._queue: queue.Queue[OcrJob] = queue.Queue(maxsize=maxsize)
        self._pending: set[int] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_pending(self, track_id: int) -> bool:
        with self._lock:
            return track_id in self._pending

    def submit(self, job: OcrJob) -> None:
        with self._lock:
            if job.track_id in self._pending:
                return
            self._pending.add(job.track_id)
        # If OCR is already backlogged, drop the oldest queued job rather than
        # blocking the caller (the main pipeline thread) or growing unbounded
        # — a missed attempt just means OcrGate's normal retry logic tries
        # again on a later frame.
        if self._queue.full():
            try:
                dropped = self._queue.get_nowait()
                with self._lock:
                    self._pending.discard(dropped.track_id)
            except queue.Empty:
                pass
        self._queue.put_nowait(job)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                result = self._ocr_reader.read(job.plate_crop)
                if result is not None:
                    # Still counts as an attempt even if the text fails
                    # validation below — that's what throttles retries via
                    # OcrGate's cooldown/max-attempts, regardless of whether
                    # the reading was usable.
                    self._ocr_gate.record_attempt(
                        job.track_id, job.frame_id, result.confidence, job.plate_crop
                    )
                    if is_valid_plate(result.text):
                        self._on_result(job.track_id, job.vehicle_type, result.text, result.confidence)
                    else:
                        logger.debug(
                            "Rejected OCR read %r for track %d (fails plate format check)",
                            result.text,
                            job.track_id,
                        )
            except Exception:
                logger.exception("OCR worker failed on track %d", job.track_id)
            finally:
                with self._lock:
                    self._pending.discard(job.track_id)
