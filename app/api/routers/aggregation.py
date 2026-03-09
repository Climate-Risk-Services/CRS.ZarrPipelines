"""
app/api/routers/aggregation.py
Endpoints for GADM aggregation jobs.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.domain.pipeline import ALL_HAZARD_CODES, aggregate_all_hazards, aggregate_hazard
from app.utils.job_store import create_job, get_job, update_job

router = APIRouter()


class AggregateRequest(BaseModel):
    gadm_levels: list[int] = [1]


def _run_aggregate_hazard(job_id: str, hazard_code: str, gadm_levels: list):
    update_job(job_id, "running")
    try:
        aggregate_hazard(hazard_code, gadm_levels=gadm_levels)
        update_job(job_id, "completed")
    except Exception as exc:
        update_job(job_id, "failed", error=str(exc))


def _run_aggregate_all(job_id: str, gadm_levels: list):
    update_job(job_id, "running")
    try:
        aggregate_all_hazards(gadm_levels=gadm_levels)
        update_job(job_id, "completed")
    except Exception as exc:
        update_job(job_id, "failed", error=str(exc))


@router.post("/aggregate/all")
def post_aggregate_all(body: AggregateRequest, background_tasks: BackgroundTasks):
    job_id = create_job(hazard="all", step="aggregate")
    background_tasks.add_task(_run_aggregate_all, job_id, body.gadm_levels)
    return {"job_id": job_id}


@router.post("/aggregate/{hazard_code}")
def post_aggregate_hazard(
    hazard_code: str, body: AggregateRequest, background_tasks: BackgroundTasks
):
    if hazard_code not in ALL_HAZARD_CODES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown hazard code '{hazard_code}'. "
            f"Valid codes: {ALL_HAZARD_CODES}",
        )
    job_id = create_job(hazard=hazard_code, step="aggregate")
    background_tasks.add_task(
        _run_aggregate_hazard, job_id, hazard_code, body.gadm_levels
    )
    return {"job_id": job_id}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job
