"""
app/api/routers/pipeline.py
HTTP trigger for the Cloud Run Job pipeline.

POST /pipeline/run                    → launch a Cloud Run Job execution
GET  /pipeline/status/{execution_id}  → query execution status from Cloud Run
"""

import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


def _job_full_name() -> str:
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    region = os.environ.get("CLOUD_RUN_REGION", "europe-west4")
    job = os.environ.get("PIPELINE_JOB_NAME", "crs-pipeline-job")
    return f"projects/{project}/locations/{region}/jobs/{job}"


class RunRequest(BaseModel):
    flow: str = "full"                          # full | score | agg | combine
    hazard_codes: Optional[List[str]] = None    # None = all 14
    scales: List[str] = ["5", "10"]
    gadm_levels: List[int] = [1]


@router.post("/run")
def run_pipeline(body: RunRequest):
    """Launch a Cloud Run Job execution for the pipeline."""
    from google.cloud import run_v2

    if body.flow not in ("full", "score", "agg", "combine"):
        raise HTTPException(status_code=400, detail="flow must be one of: full, score, agg, combine")

    env_overrides = [
        run_v2.EnvVar(name="FLOW", value=body.flow),
        run_v2.EnvVar(name="SCALES", value=",".join(body.scales)),
        run_v2.EnvVar(name="GADM_LEVELS", value=",".join(str(x) for x in body.gadm_levels)),
    ]
    if body.hazard_codes:
        env_overrides.append(
            run_v2.EnvVar(name="HAZARD_CODES", value=",".join(body.hazard_codes))
        )

    client = run_v2.JobsClient()
    request = run_v2.RunJobRequest(
        name=_job_full_name(),
        overrides=run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(env=env_overrides)
            ]
        ),
    )
    operation = client.run_job(request=request)
    execution_id = operation.metadata.name.split("/")[-1]

    return {"execution_id": execution_id, "status": "launched", "flow": body.flow}


@router.get("/status/{execution_id}")
def get_status(execution_id: str):
    """Query Cloud Run for the status of a pipeline execution."""
    from google.cloud import run_v2

    client = run_v2.ExecutionsClient()
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    region = os.environ.get("CLOUD_RUN_REGION", "europe-west4")
    job = os.environ.get("PIPELINE_JOB_NAME", "crs-pipeline-job")
    name = f"projects/{project}/locations/{region}/jobs/{job}/executions/{execution_id}"

    try:
        execution = client.get_execution(name=name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    status = "running"
    for condition in execution.conditions:
        if condition.type_ == "Completed":
            status = "completed" if condition.status == "True" else "failed"

    return {
        "execution_id": execution_id,
        "status": status,
        "start_time": execution.start_time.isoformat() if execution.start_time else None,
        "completion_time": execution.completion_time.isoformat() if execution.completion_time else None,
    }
