from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    frame_id: int
    timestamp: float
    # Content-relative position in seconds (the decoder's own PTS for a video
    # file; None for a live source with no such timeline, e.g. RTSP). Used to
    # correlate with the frontend video player's currentTime — computed from
    # the actual decoded position rather than frame_id/fps, so it stays
    # correct even with variable frame rate or dropped frames.
    video_time: float | None = None


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next available frame, or None if the source has no frame ready."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        ...

    def release(self) -> None:
        pass
