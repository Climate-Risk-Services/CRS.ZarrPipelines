"""
app/api/routers/scoring.py
Endpoints for hazard scoring jobs.
"""

from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.domain.pipeline import ALL_HAZARD_CODES, score_all_hazards, score_hazard
from app.utils.job_store import create_job, get_job, update_job

router = APIRouter()


class ScoreRequest(BaseModel):
    scales: List[str] = ["5", "10"]


def _run_score_hazard(job_id: str, hazard_code: str, scales: List[str]):
    update_job(job_id, "running")
    try:
        score_hazard(hazard_code, scales=scales)
        update_job(job_id, "completed")
    except Exception as exc:
        update_job(job_id, "failed", error=str(exc))


def _run_score_all(job_id: str, scales: List[str]):
    update_job(job_id, "running")
    try:
        score_all_hazards(scales=scales)
        update_job(job_id, "completed")
    except Exception as exc:
        update_job(job_id, "failed", error=str(exc))


@router.post("/score/all")
def post_score_all(body: ScoreRequest, background_tasks: BackgroundTasks):
    job_id = create_job(hazard="all", step="score")
    background_tasks.add_task(_run_score_all, job_id, body.scales)
    return {"job_id": job_id}


@router.post("/score/{hazard_code}")
def post_score_hazard(
    hazard_code: str, body: ScoreRequest, background_tasks: BackgroundTasks
):
    if hazard_code not in ALL_HAZARD_CODES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown hazard code '{hazard_code}'. "
            f"Valid codes: {ALL_HAZARD_CODES}",
        )
    job_id = create_job(hazard=hazard_code, step="score")
    background_tasks.add_task(_run_score_hazard, job_id, hazard_code, body.scales)
    return {"job_id": job_id}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job
