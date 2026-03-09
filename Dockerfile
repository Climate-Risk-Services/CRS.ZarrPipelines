FROM python:3.11-slim

# System deps for geopandas / GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install deps into the system Python (no venv inside container)
RUN uv sync --no-dev --system

COPY app/ ./app/
COPY main.py ./
COPY job_entrypoint.py ./

# GCP Application Default Credentials are injected at runtime via
# Workload Identity (Cloud Run) — no key file baked in.
ENV PYTHONUNBUFFERED=1

# Default: start the FastAPI service (Cloud Run Service).
# Cloud Run Job overrides this with: python job_entrypoint.py
CMD ["python", "main.py"]
