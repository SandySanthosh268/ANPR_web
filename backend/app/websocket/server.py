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
    # re-fetch, leaving hls.js permanently stuck. Segment .ts files are safe
    # to cache (immutable once written) and are unaffected by this.
    response = await call_next(request)
    if request.url.path.endswith(".m3u8"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


HLS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/hls", StaticFiles(directory=HLS_OUTPUT_DIR), name="hls")

app.include_router(cameras_router)
app.include_router(detections_router)
