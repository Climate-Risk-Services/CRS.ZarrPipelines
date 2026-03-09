"""
app/api/main.py
FastAPI application — HTTP trigger for Cloud Run Job pipeline executions.
"""

from fastapi import FastAPI

from app.api.routers.pipeline import router as pipeline_router

app = FastAPI(
    title="CRS ZarrPipelines API",
    description="HTTP trigger for climate hazard scoring and aggregation pipeline",
    version="0.2.0",
)

app.include_router(pipeline_router, prefix="/pipeline")


@app.get("/health")
def health():
    return {"status": "ok"}
