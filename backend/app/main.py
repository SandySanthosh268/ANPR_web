import argparse
import threading
import time

from app import config
from app.camera.frame_source import Frame, FrameSource
from app.camera.rtsp_reader import RTSPReader
from app.camera.video_reader import VideoReader
from app.config import DATABASE_URL, FRAME_QUEUE_MAXSIZE, HLS_SEGMENT_SECONDS, PROCESSING_FPS
from app.services import segment_store
from app.services.api_server import ApiServer
from app.services.frame_queue import FrameQueue
from app.services.hls_service import HlsService
from app.services.pipeline import Pipeline
from app.services.result_sink import CompositeSink, PrintSink, ResultSink
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _is_rtsp(source: str) -> bool:
    return source.lower().startswith("rtsp://")


def _reader_loop(reader: FrameSource, frame_queue: FrameQueue, stop_event: threading.Event) -> None:
    while not stop_event.is_set() and reader.is_open:
        frame = reader.read()
        if frame is None:
            if isinstance(reader, VideoReader):
                break
            time.sleep(0.01)
            continue
        frame_queue.put(frame)
    stop_event.set()


def _build_sink() -> ResultSink:
    sinks: list[ResultSink] = [PrintSink()]
    if DATABASE_URL:
        from app.database.db_sink import DbSink
        from app.database.schema import init_schema

        init_schema()
        sinks.append(DbSink())
        logger.info("DATABASE_URL set — persisting results to PostgreSQL")
    else:
        logger.info("DATABASE_URL not set — results will only be printed, not persisted")
    return CompositeSink(sinks)


def _run_one_cycle(source: str, camera_id: str, pipeline: Pipeline) -> None:
    """Runs the reader+HLS+pipeline loop for one pass over `source`. For a
    video file this ends when the file is exhausted; for RTSP it runs until
    interrupted (RTSPReader reconnects internally, so one "cycle" is the
    whole process lifetime).
    """
    reader: FrameSource = (
        RTSPReader(source) if _is_rtsp(source) else VideoReader(source, target_fps=PROCESSING_FPS)
    )
    frame_queue = FrameQueue(maxsize=FRAME_QUEUE_MAXSIZE)

    hls_service = HlsService(camera_id=camera_id, source=source)
    hls_service.start()

    # Fresh store each cycle, matching the fresh reader/HLS output — a client
    # fetching mid-cycle only ever sees this cycle's segments, never a mix.
    store = segment_store.SegmentStore()
    segment_store.register(camera_id, store)

    stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=_reader_loop, args=(reader, frame_queue, stop_event), daemon=True
    )
    reader_thread.start()

    current_segment_index = -1
    last_video_time = 0.0

    try:
        while not stop_event.is_set() or frame_queue.qsize() > 0:
            frame: Frame | None = frame_queue.get(timeout=1.0)
            if frame is None:
                continue
            results = pipeline.process(frame)

            if frame.video_time is not None:
                last_video_time = frame.video_time
                current_segment_index = int(frame.video_time // HLS_SEGMENT_SECONDS)
                store.add_frame(current_segment_index, frame.video_time, results)

            if frame_queue.dropped_frames:
                logger.debug("Dropped frames so far: %d", frame_queue.dropped_frames)
    finally:
        if current_segment_index >= 0:
            store.mark_source_ended(current_segment_index + 1, last_video_time)
        stop_event.set()
        reader.release()
        hls_service.stop()
        logger.info("Cycle finished, dropped frames: %d", frame_queue.dropped_frames)


def run(source: str, camera_id: str, api_host: str, api_port: int) -> None:
    # Keep the cameras API's advertised port in sync with what we actually bind.
    config.PUBLIC_PORT = api_port

    # Loaded once and reused across cycles — reloading models on every replay
    # of a video file would be wasteful, and ByteTrack/the OCR gate degrade
    # gracefully across a source restart (new track_ids get assigned once
    # motion prediction can no longer match, same as a real camera scene cut).
    pipeline = Pipeline(camera_id=camera_id, sink=_build_sink())

    api_server = ApiServer(host=api_host, port=api_port)
    api_server.start()

    is_rtsp = _is_rtsp(source)
    try:
        while True:
            _run_one_cycle(source, camera_id, pipeline)
            if is_rtsp:
                break
            logger.info("Video source ended — restarting for continuous playback")
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ANPR detection pipeline")
    parser.add_argument("--source", required=True, help="Path to a video file or an rtsp:// URL")
    parser.add_argument("--camera-id", required=True, help="Identifier for this camera")
    parser.add_argument("--api-host", default="0.0.0.0", help="Host to bind the API server to")
    parser.add_argument("--api-port", type=int, default=8765, help="Port for the API server")
    args = parser.parse_args()
    run(args.source, args.camera_id, args.api_host, args.api_port)


if __name__ == "__main__":
    main()
