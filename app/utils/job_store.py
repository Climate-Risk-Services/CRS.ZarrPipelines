"""
app/utils/job_store.py
In-memory job state tracking for async pipeline tasks.
"""

import uuid
from datetime import datetime
from typing import Dict, Optional


class JobStatus:
    def __init__(self, job_id: str, hazard: str, step: str):
        self.job_id = job_id
        self.hazard = hazard
        self.step = step
        self.status = "pending"  # pending | running | completed | failed
        self.started_at = datetime.utcnow().isoformat()
        self.finished_at: Optional[str] = None
        self.error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "hazard": self.hazard,
            "step": self.step,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


_jobs: Dict[str, JobStatus] = {}


def create_job(hazard: str, step: str) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobStatus(job_id, hazard, step)
    return job_id


def update_job(job_id: str, status: str, error: Optional[str] = None):
    if job_id not in _jobs:
        return
    _jobs[job_id].status = status
    if status in ("completed", "failed"):
        _jobs[job_id].finished_at = datetime.utcnow().isoformat()
    if error:
        _jobs[job_id].error = error


def get_job(job_id: str) -> Optional[Dict]:
    job = _jobs.get(job_id)
    return job.to_dict() if job else None
