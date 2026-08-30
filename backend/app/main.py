import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database import init_db
from app.routers import cv, jobs
from app.routers.jobs import MasterCvUnavailableError

# Root logger, explicitly to stdout -- uvicorn's default log config only
# touches its own "uvicorn.*" loggers, so without this, logging.exception()
# calls elsewhere in the app fall back to the lastResort handler (stderr).
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="SmashApply API", version="0.2.0", lifespan=lifespan)

# CORS is added first (and thus wraps every other layer, including the
# exception handlers below) so that error responses carry CORS headers too --
# an unhandled exception that bypasses CORSMiddleware entirely comes back with
# no Access-Control-Allow-Origin header, which the browser reports as a bare
# "TypeError: Failed to fetch" instead of the real error. Wide open for local
# dev since this only ever runs against a local Ollama instance.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Content-Disposition isn't in the browser's default CORS-safelisted response
    # headers, so without this the frontend's fetch() can't read the dynamic
    # filename and silently falls back to a generic name.
    expose_headers=["Content-Disposition"],
)

app.include_router(jobs.router)
app.include_router(cv.router)


@app.exception_handler(MasterCvUnavailableError)
async def master_cv_unavailable_handler(request: Request, exc: MasterCvUnavailableError):
    return JSONResponse(status_code=400, content={"error": "Master CV file not found. Please upload again."})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Starlette special-cases a handler registered for the bare `Exception`
    # class: it's pulled out and bound to ServerErrorMiddleware, which is
    # always the OUTERMOST layer of the stack -- unlike a handler for any
    # other exception type, it sits outside CORSMiddleware no matter what
    # order add_middleware was called in, so the CORS headers below have to
    # be set by hand here or this response comes back with none at all.
    logging.exception("Unhandled error while processing %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please try again."},
        headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Credentials": "true"},
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}
