from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.routers import cv, jobs

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="SmashApply API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
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


@app.get("/api/health")
def health():
    return {"status": "ok"}
