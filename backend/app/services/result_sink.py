import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    camera_id: str
    frame_id: int
    timestamp: float
    track_id: int
    vehicle_type: str
    vehicle_bbox: tuple[int, int, int, int]
    plate_bbox: tuple[int, int, int, int] | None
    plate: str | None
    vehicle_confidence: float
    plate_confidence: float | None
    ocr_confidence: float | None


class ResultSink(ABC):
    """Output boundary for aggregated detection results. DB storage and
    WebSocket broadcast (added in a later pass) are just new subclasses —
    pipeline.py never needs to change.
    """

    @abstractmethod
    def emit(self, result: DetectionResult) -> None:
        ...


class PrintSink(ResultSink):
    def emit(self, result: DetectionResult) -> None:
        print(json.dumps(asdict(result)))


class CallbackSink(ResultSink):
    def __init__(self, callback: Callable[[DetectionResult], None]):
        self._callback = callback

    def emit(self, result: DetectionResult) -> None:
        self._callback(result)


class CompositeSink(ResultSink):
    """Fans a result out to multiple sinks, e.g. PrintSink + DbSink at once."""

    def __init__(self, sinks: list[ResultSink]):
        self._sinks = sinks

    def emit(self, result: DetectionResult) -> None:
        for sink in self._sinks:
            sink.emit(result)
