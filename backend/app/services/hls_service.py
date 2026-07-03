import subprocess

from app.config import FFMPEG_BINARY, HLS_OUTPUT_DIR, HLS_SEGMENT_SECONDS
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HlsService:
    """Transcodes the same source the detection pipeline reads into an HLS
    stream, served by the FastAPI app's StaticFiles mount at /hls. Uses a VOD
    playlist (not a live sliding window) since the current use case is a
    recorded video file — see plan for why RTSP live sync is out of scope here.
    """

    def __init__(self, camera_id: str, source: str):
        self._camera_id = camera_id
        self._source = source
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        out_dir = HLS_OUTPUT_DIR / self._camera_id
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            FFMPEG_BINARY,
            "-y",
            "-i",
            self._source,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-threads",
            "2",
            "-g",
            "50",
            "-sc_threshold",
            "0",
            "-an",
            "-f",
            "hls",
            "-hls_time",
            str(HLS_SEGMENT_SECONDS),
            "-hls_playlist_type",
            "vod",
            "-hls_segment_filename",
            str(out_dir / "seg_%05d.ts"),
            str(out_dir / "index.m3u8"),
        ]
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        logger.info("Started HLS transcode for %s -> %s", self._camera_id, out_dir)

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        logger.info("Stopped HLS transcode for %s", self._camera_id)
