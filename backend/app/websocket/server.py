from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.cameras import router as cameras_router
from app.api.detections import router as detections_router
from app.config import CORS_ORIGINS, HLS_OUTPUT_DIR

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_manifest(request: Request, call_next):
    # The .m3u8 manifest's content changes across each source-loop cycle at
    # the *same* URL (fresh segments, or none yet) — without this, the
    # browser can cache an early/empty snapshot indefinitely and never
    # re-fetch, leaving hls.js permanently stuck.
    response = await call_next(request)
    if request.url.path.endswith(".m3u8"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    elif response.status_code == 404:
        # A segment .ts file is safe to cache once it exists (immutable
        # after ffmpeg finishes writing it) — but hls.js can request it
        # fractions of a second *before* ffmpeg has finished, getting a
        # genuine 404. Without this, the browser's default heuristic caching
        # can cache that 404 and keep serving it even after the real file
        # exists, permanently "losing" that segment and stalling playback.
        response.headers["Cache-Control"] = "no-store"
    return response


HLS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/hls", StaticFiles(directory=HLS_OUTPUT_DIR), name="hls")

app.include_router(cameras_router)
app.include_router(detections_router)
