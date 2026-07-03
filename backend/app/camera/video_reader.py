import time

import cv2

from app.camera.frame_source import Frame, FrameSource
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VideoReader(FrameSource):
    """Reads frames from a local video file, sequentially, at its native pace."""

    def __init__(self, path: str):
        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        self._frame_id = 0
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._start_time: float | None = None
        logger.info("Opened video file %s", path)

    def read(self) -> Frame | None:
        if not self._cap.isOpened():
            return None
        ok, image = self._cap.read()
        if not ok:
            return None

        # Decoding is CPU-bound and much faster than real playback, which
        # would otherwise blow through the whole file before inference can
        # keep up (see frame_queue.py's drop-oldest behavior). Pacing reads to
        # the source's own fps makes this behave like a live feed, so the
        # queue only drops frames under genuine CPU backpressure.
        if self.fps:
            if self._start_time is None:
                self._start_time = time.monotonic()
            target_time = self._start_time + self._frame_id / self.fps
            delay = target_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        self._frame_id += 1
        # CAP_PROP_POS_MSEC reflects the decoder's actual PTS for the frame
        # just read, not a frame_id/fps estimate — stays correct even if the
        # source has variable frame rate or the occasional dropped frame.
        video_time = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return Frame(image=image, frame_id=self._frame_id, timestamp=time.time(), video_time=video_time)

    @property
    def is_open(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        self._cap.release()
