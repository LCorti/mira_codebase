import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.registry import start_helpers, stop_helpers
from app.routers import (
    datasets,
    jobs,
    methods,
    metrics,
    models,
    projects,
    results,
    tags,
    tasks,
    tests,
    web,
)

logging.getLogger("uvicorn.access").addFilter(
    lambda record: "/jobs/batch/show" not in record.getMessage()
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_helpers()
    yield
    stop_helpers()


app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",  # Django frontend
        "http://127.0.0.1:8000",  # Alternative localhost
        "http://localhost:3000",  # Common development port
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(web.router)

app.include_router(models.router)
app.include_router(datasets.router)
app.include_router(metrics.router)
app.include_router(tasks.router)

app.include_router(projects.router)
app.include_router(tests.router)
app.include_router(methods.router)
app.include_router(jobs.router)
app.include_router(results.router)
app.include_router(tags.router)


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
